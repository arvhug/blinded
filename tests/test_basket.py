"""The basket mechanism: N legs in the sim, the cross-sectional score, ``bl.basket`` and D9.

The three-leg sim adds a leg C to the two-leg example of ``test_pairs.py``
(grid 1 to 5 s, latency 0.5 s, 7.5 bps; A asks for +1, +1, -1, -1, 0 and B for
the opposite). C's book is a copy of A's.

The run tests use twenty synthetic perpetuals on 15-minute bars over twenty
days (``xs_run``): enough legs for the archetypes block to imply a basket, and
validation of 4 days against the 80 hours short-term reversal needs.
"""

import shutil
import textwrap
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

import blinded as bl
from blinded import Panel, contract, demo, dies, metrics, truncation
from blinded import brief as briefs
from blinded.brief import Window
from blinded.contract import Legs
from blinded.daemon import Daemon
from blinded.durations import label
from blinded.evaluate import _all_horizons, score_xs
from blinded.ladder import BlockDraw, xs_rank_ic_score
from blinded.sim import simulate_legs
from blinded.vault import RunDir

pytestmark = pytest.mark.usefixtures("rules2")  # hand numbers under rules 2
known = pytest.mark.known_answer
cross_check = pytest.mark.cross_check
edge = pytest.mark.edge_case
T0 = datetime(2024, 1, 1, tzinfo=UTC)
WINDOW = Window(
    "w", pd.Timestamp(T0 + timedelta(seconds=1)), pd.Timestamp(T0 + timedelta(seconds=6))
)
LONG_A = np.array([1.0, 1.0, -1.0, -1.0, 0.0])


def at(*seconds):
    return [T0 + timedelta(seconds=s) for s in seconds]


def quotes(bids, asks):
    n = len(bids)
    return pl.DataFrame({"ts": at(0.0, 1.2, 2.2, 3.2), "bid": bids, "ask": asks,
                         "bid_qty": [1.0] * n, "ask_qty": [1.0] * n})  # fmt: skip


def prints(seconds, prices, sides):
    n = len(seconds)
    return pl.DataFrame({"ts": at(*seconds), "price": prices, "qty": [1.0] * n, "side": sides})


@pytest.fixture(scope="module")
def panel():
    a = quotes([99.98, 99.99, 100.19, 100.09], [100.02, 100.01, 100.21, 100.11])
    b = quotes([49.98, 49.99, 50.09, 50.04], [50.02, 50.01, 50.11, 50.06])
    a_prints = prints([0.5, 2.6, 3.7, 4.4], [99.98, 100.19, 100.11, 100.09], [-1.0, -1, 1, -1])
    b_prints = prints([0.5, 2.6, 3.7], [49.98, 50.11, 50.04], [-1.0, 1, -1])
    return Panel.from_frames(
        at(1, 2, 3, 4, 5),
        book={"A": a, "B": b, "C": a.clone()},
        trades={"A": a_prints, "B": b_prints, "C": a_prints.clone()},
    )


def make_brief():
    raw = {
        "data": {"path": "x.parquet", "instrument_column": "s", "streams": {}},
        "target": {"instrument": "A", "horizons": ["1s"]},
        "grid": {"step": "1s", "warmup": "0s"},
        "execution": {"latency_ms": 500, "taker_fee_bps": 7.5},
        "splits": {"embargo": "1min"},
        "families": {"basket": {"instruments": ["A", "B", "C"]}},
    }
    return briefs.parse(raw)


@known
def test_a_third_leg_held_flat_changes_nothing(panel):
    # every number is the two-leg answer of test_pairs.py: -12 gross on 8 fills, 60 of fees,
    # -72 net, round trips -36 and -36, exposure 8 / 5, and C never trades
    sim = simulate_legs(Legs({"A": LONG_A, "B": -LONG_A, "C": 0 * LONG_A}), panel, WINDOW,
                        make_brief())  # fmt: skip
    m = sim.metrics()
    assert sim.gross.sum() == pytest.approx(-12)
    assert (m["fills"], m["fills_A"], m["fills_B"], m["fills_C"]) == (8, 4, 4, 0)
    assert m["fees_bps"] == pytest.approx(60)
    assert sim.net.sum() == pytest.approx(-72)
    assert sim.run_net == pytest.approx([-36, -36])
    assert m["gross_exposure"] == pytest.approx(1.6)


@known
def test_a_third_leg_traded_like_the_first_adds_its_numbers(panel):
    # C is A's book with A's positions, so it adds A's +6 gross on 4 fills: -12 + 6 = -6
    # gross, 12 fills, 12 x 7.5 = 90 of fees, -96 net over 1/720 h = -69,120 bps/h.
    # Exposure 3 legs held at each of the first four points: (3 + 3 + 3 + 3 + 0) / 5 = 2.4.
    # Round trips are A's runs: the long A -7, B -29, C -7 = -43; the short A -17, B -19,
    # C -17 = -53; -43 - 53 = -96, the net.
    sim = simulate_legs(Legs({"A": LONG_A, "B": -LONG_A, "C": LONG_A}), panel, WINDOW,
                        make_brief())  # fmt: skip
    m = sim.metrics()
    assert sim.gross.sum() == pytest.approx(-6)
    assert (m["fills"], m["fills_C"]) == (12, 4)
    assert m["fees_bps"] == pytest.approx(90)
    assert sim.net.sum() == pytest.approx(-96)
    assert m["net_bps_per_hour"] == pytest.approx(-69_120)
    assert m["gross_exposure"] == pytest.approx(2.4)
    assert sim.run_net == pytest.approx([-43, -53])
    assert sim.run_net.sum() == pytest.approx(-96)


@known
def test_breadth_of_the_three_legs(panel):
    # nets: A and C +6 gross - 30 fees = -24 each, B -18 - 30 = -48; none positive, the
    # total is negative so no top share, and the best leg is A (the first of the two -24s).
    # Each leg is held at four of five points: 12 / 5 = 2.4 legs. Trades at steps 0 (open),
    # 2 (flip) and 4 (close) in every leg, 4 units a leg: 12 / 3 = 4 a rebalance.
    sim = simulate_legs(Legs({"A": LONG_A, "B": -LONG_A, "C": LONG_A}), panel, WINDOW,
                        make_brief())  # fmt: skip
    out = dies.breadths(sim)
    assert out["leg_net_bps"] == pytest.approx({"A": -24, "B": -48, "C": -24})
    assert (out["positive_share"], out["top_share"], out["top_leg"]) == (0.0, None, "A")
    assert out["mean_legs_held"] == pytest.approx(2.4)
    assert (out["rebalances"], out["turnover_per_rebalance"]) == (3, 4.0)


@known
def test_a_basket_round_trip_is_each_legs_own_run(panel):
    # A's and C's long runs -7, their short runs -17; B's short run -29 and long run -19:
    # six round trips, which sum to the net
    sim = simulate_legs(Legs({"A": LONG_A, "B": -LONG_A, "C": LONG_A}), panel, WINDOW,
                        make_brief())  # fmt: skip
    nets = sorted(p["net"] for p in dies.leg_paths(sim))
    assert nets == pytest.approx(sorted([-7, -17, -29, -19, -7, -17]))
    assert sum(nets) == pytest.approx(-96)


# -- the cross-sectional score -------------------------------------------------------------

X3 = np.array([[1.0, 2, 3], [1, 2, 3]])
R3 = np.array([[0.01, 0.02, 0.03], [0.03, 0.01, 0.02]])


@known
def test_the_cross_sectional_score_of_three_legs():
    # c = 1 at the first point; at the second the returns rank (3, 1, 2) against (1, 2, 3):
    # d^2 = 4 + 1 + 1 = 6, c = 1 - 6 x 6 / (3 x 8) = -0.5; the score is 0.25.
    # markout with k = 1: (300 - 100) / 2 = 100 bps, then (200 - 300) / 2 = -50; mean 25
    assert metrics.xs_rank_ic(X3, R3).tolist() == pytest.approx([1.0, -0.5])
    out = metrics.xs_markout(X3, R3)
    assert out["markout_bps"] == pytest.approx(25.0)
    # top leg's return 300 then 200 bps: 250; bottom's 100 then 300, negated: -200
    assert (out["top_bps"], out["bottom_bps"]) == pytest.approx((250.0, -200.0))


@known
def test_score_xs_reports_the_score_under_the_single_leg_keys():
    raw = {
        "data": {"path": "x.parquet", "instrument_column": "s", "streams": {}},
        "target": {"instrument": "A", "horizons": ["1s"]},
        "grid": {"step": "1s", "warmup": "0s"},
        "execution": {"latency_ms": 0, "taker_fee_bps": 7.5},
        "splits": {"embargo": "1min"},
        "stats": {"n_boot": 50},
        "families": {"basket": {"instruments": ["A", "B", "C"]}},
    }
    brief = briefs.parse(raw)
    ts = pl.Series("ts", at(1, 2)).dt.cast_time_unit("ns")
    signal = Legs({leg: X3[:, i] for i, leg in enumerate("ABC")})
    # the same returns at every horizon the decay reads
    labels = [label(h) for h in _all_horizons(brief)]
    targets = {leg: pl.DataFrame({"ts": ts, **dict.fromkeys(labels, R3[:, i])})
               for i, leg in enumerate("ABC")}  # fmt: skip
    win = Window("w", pd.Timestamp(T0), pd.Timestamp(T0 + timedelta(seconds=10)))
    row = score_xs(signal, ts, targets, win, brief)["horizons"].row(0, named=True)
    assert (row["n"], row["rank_ic"], row["ic"]) == (2, pytest.approx(0.25), pytest.approx(0.25))
    assert row["markout_bps"] == pytest.approx(25.0)


@edge
def test_a_point_with_fewer_than_three_legs_has_no_score():
    x = np.array([[1.0, 2.0, np.nan], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    r = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.3, np.nan, 0.1]])
    # two finite legs; tied features (constant ranks); two finite legs again
    assert np.isnan(metrics.xs_rank_ic(x, r)).all()


@cross_check
def test_xs_rank_ic_score_matches_a_pandas_groupby():
    rng = np.random.default_rng(20)
    n, k = 600, 20
    x = rng.normal(size=(n, k))
    r = 0.1 * x + rng.normal(size=(n, k))
    x[rng.random((n, k)) < 0.05] = np.nan
    r[rng.random((n, k)) < 0.05] = np.nan
    r[:5] = np.nan  # points with no score at all
    long = pd.DataFrame({"t": np.repeat(np.arange(n), k), "x": x.ravel(), "r": r.ravel()})
    long = long.dropna()

    def spearman(g):
        if len(g) < 3:
            return np.nan
        return g["x"].rank().corr(g["r"].rank())

    c = long.groupby("t").apply(spearman, include_groups=False).reindex(range(n)).to_numpy()
    draw = BlockDraw.make(n, 25, 200, seed=4)
    score = xs_rank_ic_score(x, r, draw)
    assert score.value == pytest.approx(np.nanmean(c), abs=1e-12)
    # each replicate: the mean of c over the drawn blocks' points, wrapping at the end
    expected = []
    for row in range(len(draw.starts)):
        idx = np.concatenate(
            [(s + np.arange(m)) % n for s, m in zip(draw.starts[row], draw.lengths, strict=True)]
        )
        expected.append(np.nanmean(c[idx]))  # fmt: skip
    np.testing.assert_allclose(score.replicates, expected, rtol=0, atol=1e-12)


# -- bl.basket -------------------------------------------------------------------------


@known
def test_basket_holds_the_top_and_bottom_through_each_window():
    # 4-row windows of 4 s; the scores at the rows before the second and third window starts
    # are (3, 1, 2) and (1, 2, 3); top = 1: (+1, -1, 0), then (-1, 0, +1). Leg 1 flips (two
    # fills), leg 2 closes, leg 3 opens. Rows 0 to 3 have no earlier row and are null.
    ts = pl.Series("ts", at(*range(12))).dt.cast_time_unit("ns")
    frame = pl.DataFrame({"ts": ts, "a": [None, None, None, 3.0, 9, 9, 9, 1, 0, 0, 0, 0],
                          "b": [None, None, None, 1.0, 9, 9, 9, 2, 0, 0, 0, 0],
                          "c": [None, None, None, 2.0, 9, 9, 9, 3, 0, 0, 0, 0]})  # fmt: skip
    out = frame.select(**bl.basket({k: pl.col(k) for k in "abc"}, top=1, every="4s"))
    assert out["a"].to_list() == [None] * 4 + [1.0] * 4 + [-1.0] * 4
    assert out["b"].to_list() == [None] * 4 + [-1.0] * 4 + [0.0] * 4
    assert out["c"].to_list() == [None] * 4 + [0.0] * 4 + [1.0] * 4


@edge
def test_basket_is_flat_with_too_few_scores_and_skips_a_null_leg():
    ts = pl.Series("ts", at(*range(8))).dt.cast_time_unit("ns")
    frame = pl.DataFrame({"ts": ts, "a": [5.0] * 8, "b": [None] * 8, "c": [1.0] * 8,
                          "d": [3.0] * 8})  # fmt: skip
    one = frame.select(**bl.basket({k: pl.col(k) for k in "abcd"}, top=1, every="4s"))
    # b has no score: a is the top, c the bottom, d in the middle, b out of the rank
    assert one.row(4) == (1.0, 0.0, -1.0, 0.0)
    two = frame.select(**bl.basket({k: pl.col(k) for k in "abcd"}, top=2, every="4s"))
    # three finite scores cannot fill two a side: flat (null) throughout
    assert all(v is None for row in two.rows() for v in row)


@known
def test_a_basket_of_max_position_two_has_gross_four():
    ts = pl.Series("ts", at(*range(8))).dt.cast_time_unit("ns")
    frame = pl.DataFrame({"ts": ts, **{k: [float(i)] * 8 for i, k in enumerate("abcdef")}})
    out = frame.select(**bl.basket({k: pl.col(k) for k in "abcdef"}, top=2, every="4s",
                                   max_position=2.0))  # fmt: skip
    # top 2 (e, f) hold +2 / 2 = +1 each, bottom 2 (a, b) -1 each: gross 4 = 2 x max_position
    assert out.row(4) == (-1.0, -1.0, 0.0, 0.0, 1.0, 1.0)


@cross_check
def test_basket_is_causal(synthetic_panel):
    p = synthetic_panel
    legs = ["XRPUSDT", "XRPUSDT_PERP"]

    def fn(panel):
        score = {s: bl.features.ret(bl.features.mid(s), "10s") for s in legs}
        score["XRPUSDT_LAG"] = bl.features.lag(bl.features.mid("XRPUSDT"), "1s")
        return bl.basket(score, top=1, every="30s")

    window = bl.Window("synthetic", p.time(0) + pd.Timedelta("5min"), p.time(-1))
    verdict = truncation.check(fn, p, window, seed=1)
    assert verdict.ok, str(verdict)
    assert np.isfinite(verdict.full["XRPUSDT"]).mean() > 0.9


# -- through the daemon, on twenty synthetic perpetuals ---------------------------------

LEGS = [f"S{i:02d}" for i in range(20)]


def xs_file(path, days=20, seed=3):
    """Twenty perpetuals on 15-minute bars: a common walk plus each name's own, with a
    reversal planted at 4 h, so the four-hour reversal score has something to rank."""
    rng = np.random.default_rng(seed)
    n = days * 96
    start = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
    common = np.cumsum(rng.normal(0, 0.002, n))
    parts = []
    for leg in LEGS:
        shock = rng.normal(0, 0.003, n)
        # half of each shock is permanent, half undone 16 bars (4 h) later
        own = 0.5 * np.cumsum(shock) + 0.5 * np.convolve(shock, np.ones(16), "full")[:n]
        close = 100 * np.exp(common + own)
        bars = pl.DataFrame({"open_time": start + np.arange(n) * 900_000,
                             "close_time": start + np.arange(n) * 900_000 + 899_999,
                             "close": close, "volume": 10.0, "quote_volume": close * 10,
                             "count": 5, "taker_buy_volume": 6.0})  # fmt: skip
        parts.append(demo.bar_rows(bars, leg, 1.0, "15m"))
    pl.concat(parts).sort("ts", maintain_order=True).write_parquet(path)
    return path


def xs_brief_raw(data: str) -> dict:
    return {
        "blinded": 0, "name": "xs-synthetic",
        "data": {"path": data, "format": "parquet",
                 "time": {"column": "ts", "unit": "ns", "tz": "UTC", "clock": "exchange"},
                 "instrument_column": "symbol",
                 "streams": {"book": {"filter": "kind == 'book'",
                                      "fields": {"bid": "bid0_px", "ask": "ask0_px"}},
                             "trades": {"filter": "kind == 'trades'",
                                        "fields": {"price": "price", "qty": "qty", "side": "side",
                                                   "n": "n"},
                                        "side_values": {"buy": "B", "sell": "S"}}},
                 "cost_model": {"kind": "half_spread", "fill": "next bar close",
                                "source": "synthetic bookTicker 2023-12-31",
                                "half_spread_bps": dict.fromkeys(LEGS, 1.0)}},
        "target": {"instrument": LEGS[-1], "price": "mid", "horizons": ["15m", "1h", "4h"],
                   "primary_horizon": "4h"},
        "grid": {"step": "15min", "start": "2024-01-01T00:00:00Z", "warmup": "1d"},
        "execution": {"latency_ms": 900_000, "taker_fee_bps": 5.0, "min_rebalance": 0.1},
        "splits": {"fractions": [0.5, 0.25, 0.25], "embargo": "1d"},
        "stats": {"block": "4h", "n_boot": 200, "seed": 1},
        "archetypes": {"instruments": LEGS},
    }  # fmt: skip


@pytest.fixture(scope="module")
def xs_run(tmp_path_factory):
    from blinded import vault

    root = tmp_path_factory.mktemp("xs")
    data = xs_file(root / "xs.parquet")
    brief = root / "xs.brief.yaml"
    brief.write_text(yaml.safe_dump(xs_brief_raw(str(data)), sort_keys=False))
    return vault.init(brief, tmp_path_factory.mktemp("xs-runs") / "run")


@pytest.fixture
def xs_copy(xs_run, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(xs_run.root, root)
    return RunDir(root)


def judge(run, source, rid="r1"):
    (run.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
    (run.inbox / f"{rid}.json").write_text('{"kind": "eval", "file": "f.py", "name": null}')
    daemon = Daemon(run)
    return daemon.handle(run.inbox / f"{rid}.json"), daemon


REVERSAL = """
import polars as pl

import blinded as bl
from blinded import features as F

rev = bl.short_term_reversal()


@rev.feature("the names that fell most over the last four hours bounce")
def rev_score(panel):
    return F.xs_zscore(F.each(panel.legs, lambda s: -F.ret(pl.col(f"{s}.mid"), "4h")))


@rev.strategy(requires=["rev_score"])
def rev_basket(panel, features):
    return bl.basket(features["rev_score"], top=4, every="4h")
"""


@known
def test_a_basket_feature_and_strategy_are_judged_by_the_daemon(xs_copy):
    response, daemon = judge(xs_copy, REVERSAL)
    feat, strat = response["trials"]
    assert (feat["status"], strat["status"]) == ("accepted", "accepted"), response
    assert (feat["family"], feat["edge"], feat["archetype"]) == (
        "basket",
        "liquidity",
        "short_term_reversal",
    )
    # the planted reversal ranks the legs: a positive cross-sectional rank IC at 4 h
    assert feat["train"]["horizons"][label(pd.Timedelta("4h"))]["rank_ic"] > 0.05
    assert feat["validation"]["released"] is True
    m = strat["train"]["rows"]["brief"]
    # four names a side at a quarter each: gross exposure 2 while in position
    assert m["gross_exposure"] == pytest.approx(2.0, abs=0.02)
    assert m["fills_S00"] > 0


@known
def test_a_basket_that_leaves_out_a_leg_is_an_error_row(xs_copy):
    source = """
    import blinded as bl

    rev = bl.short_term_reversal()


    @rev.feature("nineteen of the twenty")
    def nineteen_f(panel):
        return {s: panel.col(s, "mid") for s in panel.legs[:19]}


    @rev.strategy()
    def nineteen_s(panel, features):
        return {s: 0.0 * panel.col(s, "mid") for s in panel.legs[:19]}
    """
    response, daemon = judge(xs_copy, source)
    keys = ", ".join(LEGS)
    feat, strat = response["trials"]
    assert (feat["status"], strat["status"]) == ("error", "error")
    rule = "a basket feature returns a mapping with one value per leg"
    assert feat["reason"] == f"{rule}, {keys}, only"
    assert strat["reason"] == (f"a basket strategy returns a mapping with a position for each "
                               f"leg, {keys}, only")  # fmt: skip


@known
def test_a_basket_feature_standardised_on_the_whole_sample_is_lookahead(xs_copy):
    source = """
    import polars as pl

    import blinded as bl
    from blinded import features as F

    rev = bl.short_term_reversal()


    @rev.feature("each name's return less the whole sample's mean cross-section")
    def planted(panel):
        rets = F.each(panel.legs, lambda s: F.ret(pl.col(f"{s}.mid"), "4h"))
        mean = F.xs_mean(rets)
        return {s: rets[s] - mean[s].mean() for s in panel.legs}
    """
    response, _ = judge(xs_copy, source)
    (trial,) = response["trials"]
    assert trial["status"] == "rejected_lookahead"


@cross_check
def test_the_report_draws_the_basket_and_its_breadth(xs_copy):
    from blinded.report import gather, recompute, render

    judge(xs_copy, REVERSAL)
    diag = recompute(xs_copy, None)["diagnostics"]["archetypes"]
    rev = diag["liquidity"]["short_term_reversal"]
    assert rev["strategy"]["name"] == "rev_basket"
    legs = rev["strategy"]["pnl"]["val"]["legs"]
    assert sorted(legs) == LEGS
    # the bold line is the sum of the thin ones
    total = np.sum([leg for leg in legs.values()], axis=0)
    np.testing.assert_allclose(rev["strategy"]["pnl"]["val"]["net"], total, atol=1e-9)
    assert rev["dies"]["breadth"]["val"]["legs"] == 20
    assert rev["dies"]["breadth"]["val"]["mean_legs_held"] == pytest.approx(8.0, abs=0.1)
    # statistical arbitrage has no strategy, so it is drawn at the template's defaults
    stat = diag["relative_value"]["statistical_arbitrage"]
    assert stat["strategy"]["name"] is None and "bl.basket(top=4" in stat["strategy"]["why"]
    html = render(gather(xs_copy))
    assert "Net P&amp;L of rev_basket, leg by leg" in html or "leg by leg" in html
    assert "D9: did one name carry it" in html


@edge
def test_a_families_brief_takes_a_basket_of_three_or_more():
    raw = make_brief().raw | {"families": {"basket": {"instruments": ["A", "B"]}}}
    with pytest.raises(ValueError, match="a basket needs at least 3 different instruments"):
        briefs.parse(raw)
    brief = make_brief()
    bound = contract.bind(bl.strategy(family="basket")(lambda p, f: {}), brief)
    assert (bound.family, bound.legs) == ("basket", ("A", "B", "C"))
