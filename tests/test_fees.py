"""The fee table and the fee verdict (blinded.fees): profitable at the brief's fee,
fee-limited, or edge-limited, on the score's blocks and draws."""

import math

import numpy as np
import pytest

from blinded import brief as briefs
from blinded import fees
from blinded.archetypes import PERIODS
from blinded.fees import Tier
from blinded.ladder import BlockDraw

pytestmark = pytest.mark.usefixtures("rules2")  # hand numbers under rules 2
known = pytest.mark.known_answer
cross = pytest.mark.cross_check


def raw_brief(**execution):
    window = "2026-10-01T00:00:01Z/2026-10-01T00:00:06Z"
    return {"name": "tardis-binance-solusdt-perp-mm", "target": {"instrument": "X",
            "horizons": ["1s"]}, "grid": {"step": "1s", "warmup": "0s"},
            "data": {"path": "x.parquet", "instrument_column": "symbol",
                     "streams": {"book": {"fields": {"bid": "bid"}}}},
            "execution": {"latency_ms": 500, **execution},
            "splits": {"embargo": "2s", "windows": {"train": window, "val": window,
                                                    "test": window}},
            "stats": {"block": "1s", "n_boot": 50, "seed": 0}}  # fmt: skip


@known
def test_the_table_has_one_row_per_tier_with_its_source():
    rows = fees.tiers()
    assert len({t.key for t in rows}) == len(rows) >= 80
    assert {t.venue for t in rows} == set(fees.VENUES)
    assert all(t.source.startswith("https://") and t.fetched for t in rows)
    # Deribit's public API, get_instruments, read on 2026-10-20: maker_commission 0.00015 and
    # taker_commission 0.00035 on the perpetuals, 0.0003 both on options
    perp, opt = (
        fees.tier("deribit/inverse-perpetual/standard"),
        fees.tier("deribit/options/standard"),
    )
    assert (perp.maker_bps, perp.taker_bps, opt.maker_bps, opt.taker_bps) == (1.5, 3.5, 3.0, 3.0)
    # a market maker program sets only the maker fee
    program = [t for t in rows if not math.isfinite(t.taker_bps)]
    assert program and all(t.maker_bps <= 0 for t in program)


@known
def test_a_brief_takes_its_fees_from_a_named_tier_and_refuses_an_unknown_one():
    b = briefs.parse(raw_brief(fee_tier="deribit/inverse-perpetual/standard"))
    assert (b.execution.maker_fee_bps, b.execution.taker_fee_bps) == (1.5, 3.5)
    assert b.execution.fee_tier == "deribit/inverse-perpetual/standard"
    with pytest.raises(ValueError, match="not in the fee table"):
        briefs.parse(raw_brief(fee_tier="deribit/inverse-perpetual/vip99"))
    with pytest.raises(ValueError, match="drop maker_fee_bps"):
        briefs.parse(raw_brief(fee_tier="deribit/inverse-perpetual/standard", maker_fee_bps=2))
    program = next(t for t in fees.tiers() if not math.isfinite(t.taker_bps))
    with pytest.raises(ValueError, match="give taker_fee_bps"):
        briefs.parse(raw_brief(fee_tier=program.key))
    b = briefs.parse(raw_brief(fee_tier=program.key, taker_fee_bps=4.0))
    assert (b.execution.maker_fee_bps, b.execution.taker_fee_bps) == (program.maker_bps, 4.0)
    # without a tier the brief's own fees, as before
    b = briefs.parse(raw_brief(taker_fee_bps=5.0, maker_fee_bps=2.0))
    assert (b.execution.maker_fee_bps, b.execution.taker_fee_bps, b.execution.fee_tier) == (
        2.0, 5.0, None)  # fmt: skip
    assert fees.venue_of(b) == ("binance", ["usdm-perpetual"], "the brief's name")


def lower_by_hand(rest, units, block, hours, n_boot, seed, f):
    """The interval's lower end at fee ``f``, studentised, resampled here from the draw's own
    starts: each draw's total less the pass's, over the draw's own standard error (its block
    sums' standard deviation times the square root of their number), and the 95th percentile
    of that ratio times the pass's own error taken off the pass's total."""
    draw = BlockDraw.make(len(rest), block, n_boot, seed)
    k = len(rest) // block
    lengths = [block] * (k - 1) + [len(rest) - (k - 1) * block]
    n = len(rest)
    v = rest - f * units
    own = np.array([v[sum(lengths[:j]) : sum(lengths[: j + 1])].sum() for j in range(k)])
    total, se = own.sum(), own.std(ddof=1) * np.sqrt(k)
    ratios = []
    for row in draw.starts:
        sums = np.array([v[(s + np.arange(m)) % n].sum()
                         for s, m in zip(row[:k], lengths, strict=True)])  # fmt: skip
        if sums.std(ddof=1) > 0:
            ratios.append((sums.sum() - total) / (sums.std(ddof=1) * np.sqrt(k)))
    return float((total - np.quantile(ratios, 0.95) * se) / hours)


@known
def test_the_fee_verdict_on_the_hand_example():
    # the hand example at a 2 bps maker fee: RA, scored, nets 0, -10, -20, 0, -90 by 1 s step and
    # rests 0, 10, 10, 0, 10 units (2.6, 3.8, 5.7 s); the flatten's 5 bps taker fee stays. At a
    # maker fee f a step nets rest - f x units, rest = 0, 10, 0, 0, -70: the break-even fee
    # is -60 / 30 = -2.0 bps, a 2 bps rebate. Five blocks are fewer than the 20 the Ladder
    # needs to release, so the judge claims no interval: too few blocks to say, f0 alone.
    # Asked on two blocks, the interval's lower end reaches zero only at a deeper rebate than
    # any tier publishes, so it is edge-limited, and it loses before fees.
    from tests.test_maker import judge_brief, run

    s = run("RA")
    rest, units = fees.per_step(s)
    assert np.allclose(rest, [0, 10, 0, 0, -70]) and np.allclose(units, [0, 10, 10, 0, 10])
    few = fees.judge(s, judge_brief(2.0), s.step_ns)
    assert (few["kind"], few["blocks"], few["periods"]) == (None, 5, PERIODS)
    assert few["f0_bps"] == pytest.approx(-2.0, abs=1e-12) and "interval" not in few
    assert fees.words(few).startswith("Fee verdict: too few blocks to say. The pass holds 5 "
                                      "blocks of 1 steps, fewer than the 20")  # fmt: skip
    assert "break-even maker fee, where the mean net is zero, is -2.00 bps" in fees.words(few)
    v = fees.verdict(rest, units, 1, s.hours, 2.0, "maker", 50, 0, rules=s.rules, periods=2)
    assert (v["side"], v["fee_bps"], v["block_steps"], v["blocks"]) == ("maker", 2.0, 1, 5)
    assert v["f0_bps"] == pytest.approx(-2.0, abs=1e-12)
    assert v["kind"] == "edge-limited" and v["interval"][0] < 0
    assert v["edge_case"] == "loses before fees"
    star = v["f_star_bps"]
    assert star < min(t.maker_bps for t in fees.tiers())
    assert lower_by_hand(rest, units, 1, 5 / 3600, 50, 0, star - 1e-6) > 0
    assert lower_by_hand(rest, units, 1, 5 / 3600, 50, 0, star + 1e-6) <= 0
    assert v["interval"][0] == pytest.approx(lower_by_hand(rest, units, 1, 5 / 3600, 50, 0, 2.0))
    text = fees.words(v)
    assert text.startswith("Fee verdict: edge-limited, it loses before fees.")
    assert text.endswith("the break-even fee is -2.00 bps, so even with no fee its mean net is "
                         "below zero.")  # fmt: skip


TABLE = (
    Tier("binance", "usdm-perpetual", "vip0", "VIP 0", 2.0, 5.0, "none", "https://a", "d", True),
    Tier("binance", "usdm-perpetual", "mm", "MM", -0.5, math.nan, "program", "https://a", "d",
         True),
    Tier("mexc", "usdm-perpetual", "base", "base", 0.0, 2.0, "none", "https://b", "d", False),
)  # fmt: skip


def planted(edge: float, n: int = 3600, seed: int = 3):
    """A maker whose every 1 s step earns ``edge`` bps on one unit before the fee, with noise."""
    rng = np.random.default_rng(seed)
    return edge + rng.normal(0, 0.5, n), np.ones(n)


@known
@pytest.mark.parametrize("edge, kind", [(3.0, "profitable"), (1.0, "fee-limited"),
                                        (-1.0, "edge-limited")])  # fmt: skip
def test_planted_strategies_get_their_fee_verdicts(edge, kind):
    # an hour of 1 s steps at a 2 bps brief fee: +3 before fees is profitable at it; +1 is
    # not (it nets -1 a step) but is at a fee a little under +1; -1 loses before fees and needs
    # a rebate of about 1 bps, deeper than the table's deepest, -0.5
    rest, units = planted(edge)
    v = fees.verdict(rest, units, 60, 1.0, 2.0, "maker", 200, 7, "binance",
                     ["usdm-perpetual"], "test", table=TABLE)  # fmt: skip
    assert v["kind"] == kind
    assert v["f0_bps"] == pytest.approx(rest.mean(), abs=1e-12)
    assert v["f_star_bps"] < v["f0_bps"]  # the lower end needs a lower fee than the mean
    assert lower_by_hand(rest, units, 60, 1.0, 200, 7, v["f_star_bps"]) == pytest.approx(
        0, abs=1e-5
    )
    if kind == "fee-limited":
        assert 0.0 < v["f_star_bps"] < 1.0
        assert v["tiers_own"] == ["binance/usdm-perpetual/mm"]
        assert v["tiers_other"] == ["mexc/usdm-perpetual/base"]
        text = fees.words(v, TABLE)
        assert (
            f"consistently profitable at a maker fee of {v['f_star_bps']:.2f} bps or lower" in text
        )
        assert "on its venue, Binance USD-M perpetual: MM (-0.5 bps)" in text
        assert "MEXC USD-M perpetual base (0 bps, unverified)" in text
    if kind == "edge-limited":
        assert -1.1 < v["f_star_bps"] < -0.5 - 0.3
        assert "below every published tier" in fees.words(v, TABLE)
        assert "loses before fees" in fees.words(v, TABLE)


@known
def test_a_verdict_needs_the_blocks_the_ladder_needs_to_release():
    # one rule for every pass: an interval on fewer than PERIODS whole blocks of the score's
    # block claims nothing. +3 bps a step on 19 one-minute blocks is too few blocks to say,
    # with the break-even fee alone; on 20 it is profitable, as on the hour above
    rest, units = planted(3.0, n=19 * 60)
    few = fees.verdict(rest, units, 60, 19 / 60, 2.0, "maker", 200, 7, table=TABLE)
    assert (few["kind"], few["blocks"], few["note"]) == (None, 19, "no fee verdict: too few "
                                                         "blocks to say")  # fmt: skip
    assert few["f0_bps"] == pytest.approx(rest.mean(), abs=1e-12)
    assert "f_star_bps" not in few and "interval" not in few
    rest, units = planted(3.0, n=20 * 60)
    assert fees.verdict(rest, units, 60, 1 / 3, 2.0, "maker", 200, 7, table=TABLE)["kind"] == (
        "profitable")  # fmt: skip
    never = fees.verdict(rest, 0 * units, 60, 1 / 3, 2.0, "maker", 200, 7, table=TABLE)
    assert never["note"] == "no fee verdict: it never filled" and "f0_bps" not in never


@known
def test_an_edge_limited_verdict_says_which_case_it_is():
    # +0.3 bps a step with noise of sd 40 a step, demeaned so f0 is +0.3 exactly: over the
    # hour the mean net at fee 0 is +1,080 bps, positive before fees, but the sum's sd of
    # about 40 x 60 = 2,400 puts the 5% lower end near 1,080 - 1.645 x 2,400 < 0, and it
    # reaches zero only at a fee near 0.3 - 1.645 x 2,400 / 3,600 = -0.8 bps (-0.68 on these
    # draws), deeper than the table's -0.5: not significant at any fee a venue offers. -1 a
    # step loses before fees.
    noise = np.random.default_rng(3).normal(0, 40.0, 3600)
    rest, units = 0.3 + noise - noise.mean(), np.ones(3600)
    v = fees.verdict(rest, units, 60, 1.0, 2.0, "maker", 200, 7, "binance", ["usdm-perpetual"],
                     "test", table=TABLE)  # fmt: skip
    assert v["kind"] == "edge-limited" and v["f_star_bps"] < -0.5
    assert v["f0_bps"] == pytest.approx(0.3, abs=1e-12)
    assert v["edge_case"] == "not significant at any published fee"
    text = fees.words(v, TABLE)
    assert text.startswith("Fee verdict: edge-limited, positive before fees but not "
                           "significant at any fee a venue offers.")  # fmt: skip
    assert f"up to the break-even of {v['f0_bps']:.2f} bps" in text
    rest, units = planted(-1.0)
    v = fees.verdict(rest, units, 60, 1.0, 2.0, "maker", 200, 7, table=TABLE)
    assert v["edge_case"] == "loses before fees" and v["f0_bps"] < 0


@known
def test_the_products_come_from_the_instruments_traded():
    # a brief that names no tier and no product: the venue from its name, each product from an
    # instrument the pass trades (a _PERP name a USD-M perpetual, else spot), in leg order, and
    # the venue's own tiers of those products listed first, each under its product
    raw = raw_brief(taker_fee_bps=5.0) | {"name": "binance-carry8-2024-04-01-450d-15m"}
    b = briefs.parse(raw)
    assert fees.venue_of(b) == ("binance", [], "the brief's name")
    how = "the brief's name and the instruments traded"
    assert fees.venue_of(b, ("LINKUSDT_PERP",)) == ("binance", ["usdm-perpetual"], how)
    assert fees.venue_of(b, ("ETHUSDT", "ETHUSDT_PERP")) == (
        "binance", ["spot", "usdm-perpetual"], how)  # fmt: skip
    spot = Tier("binance", "spot", "vip0", "VIP 0", 0.0, 1.0, "none", "https://c", "d", True)
    rest, units = planted(1.0)
    v = fees.verdict(rest, units, 60, 1.0, 2.0, "maker", 200, 7, "binance",
                     ["spot", "usdm-perpetual"], how, table=(*TABLE, spot))  # fmt: skip
    assert v["kind"] == "fee-limited" and v["product"] is None
    assert v["tiers_own"] == ["binance/spot/vip0", "binance/usdm-perpetual/mm"]
    assert v["tiers_other"] == ["mexc/usdm-perpetual/base"]
    assert ("on its venue, Binance spot: VIP 0 (0 bps); Binance USD-M perpetual: MM (-0.5 bps)"
            in fees.words(v, (*TABLE, spot)))  # fmt: skip
