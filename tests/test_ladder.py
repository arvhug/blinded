import math

import numpy as np
import pytest

from blinded.ladder import (
    BlockDraw,
    Ladder,
    Score,
    floor,
    rank_ic_score,
    round_to,
    sharpe_score,
    step,
)
from blinded.metrics import rank

known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case


def score(value, offsets):
    return Score(value, value + np.asarray(offsets, dtype=float))


@known
def test_steps_and_rounding():
    assert [step(x) for x in (0.002, 0.005, 0.011, 0.05, 0.051, 0.3, 7.0)] == [
        0.002, 0.005, 0.02, 0.05, 0.1, 0.5, 10.0,
    ]  # fmt: skip
    # 0.0123 / 0.005 = 2.46 -> 2 -> 0.010; 0.06 / 0.05 = 1.2 -> 1 -> 0.05
    assert round_to(0.0123, 0.005) == 0.01
    assert round_to(0.06, 0.05) == 0.05
    assert round_to(0.0175, 0.005) == 0.02  # 3.5 -> 4 (half to even)
    assert round_to(-0.013, 0.01) == -0.01


@known
def test_a_feature_phase_by_hand():
    lad = Ladder("feature")
    # 1. reps 0.0123 -+ 0.002: sd 0.002 against the no-skill baseline, floor 0.005;
    #    0.0123 - 0 > 0.005, released at round(0.0123 / 0.005) * 0.005 = 0.010
    d1 = lad.submit(score(0.0123, [-0.002, 0.002, -0.002, 0.002]), trial_id=1)
    assert d1.eta_bootstrap == pytest.approx(0.002)
    assert (d1.released, d1.best, d1.eta) == (True, 0.01, 0.005)
    # 2. moves exactly with submission 1, so the paired difference has sd 0: step = floor;
    #    0.0140 - 0.010 = 0.004 < 0.005, not released
    d2 = lad.submit(score(0.0140, [-0.002, 0.002, -0.002, 0.002]), trial_id=2)
    assert d2.eta_bootstrap == pytest.approx(0.0, abs=1e-12)
    assert (d2.released, d2.best, d2.eta) == (False, 0.01, 0.005)
    # 3. difference from submission 1's replicates: 0.0297 + (-0.013, 0.009, -0.013, 0.009),
    #    sd 0.011, step 0.02; 0.042 - 0.010 = 0.032 > 0.02, released at 2.1 -> 2 -> 0.04
    d3 = lad.submit(score(0.0420, [-0.015, 0.011, -0.015, 0.011]), trial_id=3)
    assert d3.eta_bootstrap == pytest.approx(0.011)
    assert (d3.released, d3.best, d3.best_before, d3.eta) == (True, 0.04, 0.01, 0.02)
    # 4. worse than the best: not released, the best stays
    d4 = lad.submit(score(0.0300, [0.0] * 4), trial_id=4)
    assert (d4.released, d4.best) == (False, 0.04)
    assert lad.best_trial == 3
    assert d4.public() == {"released": False, "best": 0.04}


@known
def test_strategy_floor_stops_a_trivial_improvement():
    lad = Ladder("strategy")
    # replicates all equal: bootstrap sd 0, so the step is the 0.05 floor
    assert lad.submit(score(0.04, [0.0] * 5)).released is False  # 0.04 < 0.05
    d = lad.submit(score(0.06, [0.0] * 5))
    assert (d.released, d.best, d.eta) == (True, 0.05, 0.05)  # 1.2 steps -> 1
    # 0.09 - 0.05 = 0.04 < 0.05: not released, though 0.09 beats the 0.06 behind the best
    assert lad.submit(score(0.09, [0.0] * 5)).released is False


@known
def test_the_best_never_moves_down():
    rng = np.random.default_rng(3)
    lad = Ladder("feature")
    history = []
    for _ in range(300):
        value = rng.normal(0.02, 0.03)
        d = lad.submit(Score(value, value + rng.normal(0, rng.uniform(0, 0.02), 50)))
        if d.released:
            # released only past the old best plus a step, so at least half a step above it
            assert d.best_before is None or d.best > d.best_before + d.eta / 2 - 1e-12
            assert value - (d.best_before or 0.0) > d.eta
        else:
            assert d.best == d.best_before
        history.append(d.best if d.best is not None else -np.inf)
    assert all(b >= a for a, b in zip(history, history[1:], strict=False))
    assert 1 <= sum(np.diff(history) > 0) < 30


@edge
def test_degenerate_submissions():
    lad = Ladder("feature")
    d = lad.submit(Score(np.nan, np.full(10, np.nan)))
    assert (d.released, d.best, d.eta) == (False, None, 0.005)
    d = lad.submit(Score(0.004, np.full(10, 0.004)))
    assert d.released is False  # under the floor, however precise


@cross
def test_rank_ic_replicates_match_resampling_by_hand():
    rng = np.random.default_rng(0)
    n, block = 13, 4
    signal, target = rng.normal(size=n), rng.normal(size=n)
    signal[2] = np.nan  # dropped inside its block
    draw = BlockDraw.make(n, block, n_boot=5, seed=9)
    got = rank_ic_score(signal, target, draw)
    ok = np.isfinite(signal)
    x, y = np.full(n, np.nan), np.full(n, np.nan)
    x[ok], y[ok] = rank(signal[ok]), rank(target[ok])
    assert got.value == pytest.approx(np.corrcoef(x[ok], y[ok])[0, 1], abs=1e-12)
    for k in range(5):
        idx = np.concatenate(
            [(s + np.arange(m)) % n for s, m in zip(draw.starts[k], draw.lengths, strict=True)]
        )
        keep = idx[ok[idx]]
        assert got.replicates[k] == pytest.approx(np.corrcoef(x[keep], y[keep])[0, 1], abs=1e-12)


@known
def test_sharpe_from_blocks_by_hand():
    # blocks of 2 steps: sums 2, 4, 6; mean 4, sd 2; 12 blocks an hour: 2 * sqrt(12)
    pnl = np.array([1.0, 1, 2, 2, 3, 3])
    got = sharpe_score(pnl, BlockDraw.make(6, 2, n_boot=4, seed=1), blocks_per_hour=12)
    assert got.value == pytest.approx(2 * math.sqrt(12))


@cross
def test_sharpe_replicates_match_resampling_by_hand():
    rng = np.random.default_rng(5)
    pnl = rng.normal(0.1, 1, 23)
    draw = BlockDraw.make(23, 5, n_boot=6, seed=2)
    got = sharpe_score(pnl, draw, blocks_per_hour=12)
    # 23 steps in blocks of 5: four blocks, the last of 8 steps (5 and the remainder of 3), so
    # a replicate draws four blocks of 5, 5, 5 and 8 steps, circularly, 23 steps in all
    for k in range(6):
        sums = [
            pnl[(s + np.arange(m)) % 23].sum()
            for s, m in zip(draw.starts[k][:4], [5, 5, 5, 8], strict=True)
        ]
        expect = np.mean(sums) / np.std(sums, ddof=1) * math.sqrt(12)
        assert got.replicates[k] == pytest.approx(expect, rel=1e-12)
    blocks = [pnl[:5].sum(), pnl[5:10].sum(), pnl[10:15].sum(), pnl[15:].sum()]
    assert got.value == pytest.approx(np.mean(blocks) / np.std(blocks, ddof=1) * math.sqrt(12))


@known
def test_every_step_counts_in_a_block_sharpe():
    # 5 steps in blocks of 2: the remainder, the fifth step, is folded into the second block
    # (10 and -70), not dropped (10 and 0): mean -30, sd 40 sqrt 2, so the Sharpe at 1,800
    # blocks an hour is -30 / (40 sqrt 2) x 30 sqrt 2 = -22.5, where dropping it gave
    # 5 / sqrt 50 x sqrt 1,800 = +30.0
    pnl = np.array([1.0, 9, 0, 0, -70])
    draw = BlockDraw.make(5, 2, n_boot=3, seed=1)
    got = sharpe_score(pnl, draw, blocks_per_hour=1800)
    assert got.value == pytest.approx(-22.5, abs=1e-12)
    # and every replicate is two drawn blocks of 2 and 3 steps: five steps, as the window
    for k in range(3):
        a, b = draw.starts[k][:2]
        sums = [pnl[(a + np.arange(2)) % 5].sum(), pnl[(b + np.arange(3)) % 5].sum()]
        expect = np.mean(sums) / np.std(sums, ddof=1) * math.sqrt(1800)
        assert got.replicates[k] == pytest.approx(expect, rel=1e-12, nan_ok=True)


@known
def test_pairing_shrinks_the_step_for_near_copies():
    rng = np.random.default_rng(1)
    n = 3000
    target = rng.normal(size=n)
    a = 0.1 * target + rng.normal(size=n)
    b = a + 0.01 * rng.normal(size=n)
    draw = BlockDraw.make(n, 100, n_boot=400, seed=4)
    sa, sb = rank_ic_score(a, target, draw), rank_ic_score(b, target, draw)
    alone = np.std(sb.replicates)
    paired = np.std(sb.replicates - sa.replicates)
    assert paired < alone / 10


@known
def test_the_strategy_floor_scales_with_the_block():
    # 0.05 sqrt(5 min / block): with 8-hour blocks sqrt(5 / 480) = 0.10206..., 0.0051031
    eight = Ladder("strategy", floor=floor("strategy", "8h"))
    assert eight.floor == pytest.approx(0.05 * math.sqrt(5 / 480), abs=1e-12)
    assert eight.floor == pytest.approx(0.005103103630798288, abs=1e-12)
    # five-minute blocks, every brief so far: exactly the old 0.05, so nothing published moves
    assert floor("strategy", "5min") == 0.05 == Ladder("strategy").floor
    # the rank IC floor is unitless and stays
    assert floor("feature", "8h") == floor("feature", "5min") == 0.005
    # with 8-hour blocks a strategy whose replicates do not move steps by 0.01, not 0.05
    assert eight.submit(score(0.03, [0.0] * 5)).released is True
    assert eight.best == 0.03


@known
def test_under_rules_3_a_first_release_needs_two_standard_errors():
    # one null score, sd 0.004 (step 0.005): 0.007 clears the step but not 2 sd = 0.008
    reps = np.random.default_rng(3).normal(0.0, 0.004, 4000)
    reps = (reps - reps.mean()) / reps.std() * 0.004
    for rules, released in ((2, True), (3, False)):
        ladder = Ladder("feature", floor=0.005, rules=rules)
        assert ladder.submit(Score(0.007, reps + 0.007)).released is released
    # above 2 sd it is released under rules 3 too, rounded to the step as before
    ladder = Ladder("feature", floor=0.005, rules=3)
    d = ladder.submit(Score(0.0085, reps + 0.0085))
    assert d.released and d.best == 0.01 and d.eta == 0.005


@known
def test_under_rules_3_a_worse_submission_is_not_released():
    # the statistics audit's case (audit-stats/hand_checks.py, 5): 0.089 is released and
    # rounded to 0.08 on a 0.02 step; 0.086, paired and 0.003 below it with a 0.005 step,
    # clears the rounded best (0.086 - 0.08 > 0.005) and was released, though worse
    r1 = np.random.default_rng(1).normal(0.089, 0.012, 1000)
    r2 = r1 - 0.003 + np.random.default_rng(2).normal(0, 0.001, 1000)
    for rules, again in ((2, True), (3, False)):
        ladder = Ladder("feature", floor=0.005, rules=rules)
        first = ladder.submit(Score(0.089, r1))
        assert first.released and first.best == 0.08 and first.eta == 0.02
        assert ladder.submit(Score(0.086, r2)).released is again
