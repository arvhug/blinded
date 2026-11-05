"""DSR against the numerical example in Bailey and López de Prado (2014), p. 9-10; PBO by hand,
by brute force and by Monte Carlo, since the PBO paper's own examples use unseeded random data."""

import itertools
import math
from math import comb
from statistics import NormalDist

import numpy as np
import pytest

from blinded.deflate import (
    NotEnough,
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    pbo,
    probabilistic_sharpe,
)

known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case

# the paper's example: annualised SR 2.5 over 5 years of daily data, 250 days a year,
# N = 100 trials whose annualised SRs have variance 1/2, skewness -3, kurtosis 10
SR = 2.5 / math.sqrt(250)
VAR = 0.5 / 250
T = 1250


@known
def test_dsr_paper_example():
    dsr, sr0 = deflated_sharpe(SR, n_trials=100, var_trials=VAR, t=T, skew=-3, kurt=10)
    assert round(sr0, 4) == 0.1132  # "~ 0.1132, non-annualized"
    assert round(dsr, 4) == 0.9004  # "= 0.9004 < 0.95"


@known
def test_dsr_paper_example_with_46_trials():
    dsr, _ = deflated_sharpe(SR, n_trials=46, var_trials=VAR, t=T, skew=-3, kurt=10)
    assert round(dsr, 4) == 0.9505  # "after running only N=46 independent trials"


@known
def test_dsr_paper_example_with_normal_returns_and_88_trials():
    dsr, _ = deflated_sharpe(SR, n_trials=88, var_trials=VAR, t=T, skew=0, kurt=3)
    assert round(dsr, 4) == 0.9505  # "DSR = 0.9505 after N=88 independent trials"


@cross
def test_sr0_from_the_quantiles():
    # the two standard normal quantiles, from an independent implementation
    from scipy.special import ndtri

    z1, z2 = ndtri(1 - 1 / 100), ndtri(1 - 1 / (100 * math.e))
    expect = math.sqrt(VAR) * ((1 - 0.5772156649) * z1 + 0.5772156649 * z2)
    assert expected_max_sharpe(100, VAR) == pytest.approx(expect, rel=1e-9)
    # one trial: nothing was selected, the threshold is the mean
    assert expected_max_sharpe(1, VAR) == 0.0


@known
def test_psr_by_hand():
    # SR 0.1 per period over T = 101, normal: (0.1 - 0) * sqrt(100) / sqrt(1 + 0.5 * 0.01)
    got = probabilistic_sharpe(0.1, 0.0, t=101, skew=0, kurt=3)
    assert got == pytest.approx(NormalDist().cdf(1 / math.sqrt(1.005)))
    # negative skew and fat tails widen the error, so the same SR is less convincing
    assert probabilistic_sharpe(0.1, 0.0, t=101, skew=-3, kurt=10) < got


@cross
@pytest.mark.parametrize("n", [10, 100, 1000])
def test_expected_maximum_against_monte_carlo(n):
    # the paper's appendix 2 checks equation (1) the same way
    rng = np.random.default_rng(n)
    sd = 0.3
    maxima = rng.normal(0, sd, size=(4000, n)).max(axis=1)
    approx = expected_max_sharpe(n, sd**2)
    assert approx == pytest.approx(maxima.mean(), rel=0.03)


@known
def test_moments_by_hand():
    # returns 1, 2, 3, 6: mean 3, deviations -2, -1, 0, 3; m2 = 14/4, m3 = 18/4, m4 = 98/4
    sr, skew, kurt = moments(np.array([1.0, 2, 3, 6]))
    assert sr == pytest.approx(3 / math.sqrt(14 / 3))
    assert skew == pytest.approx(4.5 / 3.5**1.5)
    assert kurt == pytest.approx(24.5 / 3.5**2)


@known
def test_combination_count():
    # the paper's Figure 1 has six combinations for S = 4. For S = 16 it says 12,780,
    # but C(16, 8) = 12,870: a transposition in the text
    assert comb(4, 2) == 6 and comb(16, 8) == 12_870
    rng = np.random.default_rng(0)
    assert pbo(rng.normal(size=(16, 8)), partitions=16).n_combinations == 12_870
    assert pbo(rng.normal(size=(4, 3)), partitions=4, min_trials=3).n_combinations == 6


@known
def test_pbo_by_hand():
    # four blocks A-D, three trials, performance = mean
    #   trial 1: +3 +3 -3 -3   trial 2: 1 1 1 1   trial 3: -2 -2 +2 +2
    # train {A,B}: trial 1 wins (3); on {C,D} it is worst (-3): rank 1, omega 1/4, logit < 0
    # train {C,D}: trial 3 wins (2); on {A,B} it is worst (-2): logit < 0
    # train {A,C}, {A,D}, {B,C}, {B,D}: trials 1 and 3 average 0, trial 2 wins with 1,
    #   and is best on the other half too: rank 3, omega 3/4, logit ln 3
    m = np.array([[3.0, 1, -2], [3, 1, -2], [-3, 1, 2], [-3, 1, 2]])
    got = pbo(m, partitions=4, metric="mean", min_trials=3)
    assert got.pbo == pytest.approx(2 / 6)
    assert sorted(np.round(got.logits, 6)) == sorted(
        [round(math.log(1 / 3), 6)] * 2 + [round(math.log(3), 6)] * 4
    )
    # the in-sample winner lost money out of sample in those same two
    assert got.prob_oos_loss == pytest.approx(2 / 6)


def brute_force_pbo(m, s):
    """Algorithm 2.3 written out with loops, as a reference."""
    t, n = m.shape
    rows = t // s
    parts = [m[i * rows : (i + 1) * rows] for i in range(s)]
    logits = []
    for c in itertools.combinations(range(s), s // 2):
        train = np.vstack([parts[i] for i in c])
        test = np.vstack([parts[i] for i in range(s) if i not in c])
        sharpe_in = train.mean(axis=0) / train.std(axis=0, ddof=1)
        sharpe_out = test.mean(axis=0) / test.std(axis=0, ddof=1)
        best = int(np.argmax(sharpe_in))
        order = sharpe_out.argsort().argsort() + 1  # 1 = worst; no ties in continuous data
        omega = order[best] / (n + 1)
        logits.append(math.log(omega / (1 - omega)))
    return np.mean(np.array(logits) <= 0), np.array(logits)


@cross
def test_pbo_matches_a_brute_force_implementation():
    rng = np.random.default_rng(7)
    m = rng.normal(0.02, 1, size=(83, 9)) + np.linspace(0, 0.1, 9)
    want, want_logits = brute_force_pbo(m, 8)
    got = pbo(m, partitions=8)
    assert got.pbo == pytest.approx(want)
    assert got.logits == pytest.approx(want_logits)


@cross
def test_pbo_of_pure_noise_is_about_one_half():
    # with no skill the out-of-sample rank of the in-sample winner is uniform on 1..N;
    # for N = 10, P(rank <= 5) = 1/2
    rng = np.random.default_rng(11)
    values = [pbo(rng.normal(size=(64, 10)), partitions=16).pbo for _ in range(60)]
    assert np.mean(values) == pytest.approx(0.5, abs=0.06)


@cross
def test_pbo_with_one_real_edge_is_low():
    rng = np.random.default_rng(12)
    m = rng.normal(size=(160, 10))
    m[:, 3] += 0.8  # one trial with a Sharpe of 0.8 per period, the rest noise
    assert pbo(m, partitions=16).pbo < 0.05


@edge
def test_pbo_guard():
    rng = np.random.default_rng(0)
    with pytest.raises(NotEnough, match="not enough trials/blocks for PBO"):
        pbo(rng.normal(size=(17, 5)))
    with pytest.raises(NotEnough, match="not enough trials/blocks for PBO"):
        pbo(rng.normal(size=(12, 10)))
    with pytest.raises(ValueError, match="even"):
        pbo(rng.normal(size=(20, 10)), partitions=5)


@edge
def test_flat_trials_rank_as_zero():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(32, 8))
    m[:, 0] = 0.0  # never traded
    got = pbo(m, partitions=16)
    assert np.isfinite(got.logits).all()


@known
def test_summary_from_a_ledger(tmp_path):
    from blinded.deflate import summary
    from blinded.ledger import Ledger

    led = Ledger(tmp_path / "l.sqlite")
    led.start_run("r")
    rng = np.random.default_rng(5)
    ids = []
    for i in range(10):
        tid = led.add_trial("r", kind="strategy", status="accepted", code_hash=f"h{i}")
        led.add_blocks(tid, "val", rng.normal(0.1 * (i == 4), 1, 17).tolist())
        ids.append(tid)
    led.add_trial("r", kind="strategy", status="rejected_lookahead", code_hash="leak")
    got = summary(led, ids[4])
    # N counts the rejected trial too; T is the 17 validation blocks
    assert (got["n_trials"], got["blocks"]) == (11, 17)
    sr, skew, kurt = moments(np.array([b["pnl_bps"] for b in led.blocks(ids[4], "val")]))
    sharpes = [moments(np.array([b["pnl_bps"] for b in led.blocks(t, "val")]))[0] for t in ids]
    dsr, sr0 = deflated_sharpe(sr, 11, np.var(sharpes, ddof=1), 17, skew, kurt)
    assert got["dsr"] == pytest.approx(dsr)
    assert got["sharpe_per_sqrt_hour"] == pytest.approx(sr * math.sqrt(12))
    assert got["sr0_per_sqrt_hour"] == pytest.approx(sr0 * math.sqrt(12))
    assert got["pbo"] is not None and got["pbo_reason"] is None


@edge
def test_summary_says_when_pbo_has_too_little(tmp_path):
    from blinded.deflate import summary
    from blinded.ledger import Ledger

    led = Ledger(tmp_path / "l.sqlite")
    led.start_run("r")
    rng = np.random.default_rng(1)
    for i in range(3):
        tid = led.add_trial("r", kind="strategy", status="accepted", code_hash=f"h{i}")
        led.add_blocks(tid, "val", rng.normal(0, 1, 17).tolist())
    got = summary(led, tid)
    assert got["pbo"] is None and "not enough trials/blocks for PBO" in got["pbo_reason"]
