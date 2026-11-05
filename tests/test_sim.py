"""The taker sim on a five-point book small enough to follow by hand.

Grid 1, 2, 3, 4, 5 s. Book of A (time: bid / ask, mid):

    0.0: 99.98 / 100.02, 100.00
    1.2: 99.99 / 100.01, 100.00    after the decision at 1 s, before its fill at 1.5 s
    2.2: 100.19 / 100.21, 100.20
    3.2: 100.09 / 100.11, 100.10

Positions asked for at 1..5 s: +1, +1, -1, -1, 0. The reference notional is
the mark at the first decision plus latency: 100.00 in every case below, so
0.01 of price is 1 bps.
"""

import math
import textwrap
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

import blinded as bl
from blinded import brief as briefs
from blinded.brief import Window
from blinded.panel import Panel
from blinded.sim import (
    held,
    linear,
    maker_proxy,
    report,
    simulate,
    threshold,
    walk_forward_scale,
)

pytestmark = pytest.mark.usefixtures("rules2")  # hand numbers under rules 2
known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case
T0 = datetime(2024, 1, 1, tzinfo=UTC)
POSITION = np.array([1.0, 1.0, -1.0, -1.0, 0.0])
HALF = pd.Timedelta("500ms")
WINDOW = Window(
    "w", pd.Timestamp(T0 + timedelta(seconds=1)), pd.Timestamp(T0 + timedelta(seconds=6))
)


def at(*seconds):
    return [T0 + timedelta(seconds=s) for s in seconds]


def hand_panel(delay_s=0.0, n=None):
    """The book above; every row available ``delay_s`` after its stamp."""
    book = pl.DataFrame(
        {
            "ts": at(0.0, 1.2, 2.2, 3.2),
            "bid": [99.98, 99.99, 100.19, 100.09],
            "ask": [100.02, 100.01, 100.21, 100.11],
            "bid_qty": [1.0] * 4,
            "ask_qty": [1.0] * 4,
        }
    )
    trades = pl.DataFrame(
        {
            "ts": at(0.5, 2.6, 3.7, 4.4),
            "price": [99.98, 100.19, 100.11, 100.09],
            "qty": [1.0] * 4,
            "side": [-1.0, -1.0, 1.0, -1.0],
        }
    )
    if n is not None:
        trades = trades.with_columns(n=pl.Series(n))
    late = (pl.col("ts") + pd.Timedelta(seconds=delay_s)).alias("available_at")
    book, trades = book.with_columns(late), trades.with_columns(late)
    return Panel.from_frames(at(1, 2, 3, 4, 5), book={"A": book}, trades={"A": trades})


@pytest.fixture(scope="module")
def panel():
    return hand_panel()


def run(panel, position=POSITION, latency_ms=500, fee_bps=7.5, **kw):
    return simulate(
        position,
        panel,
        WINDOW,
        "A",
        latency=pd.Timedelta(milliseconds=latency_ms),
        fee_bps=fee_bps,
        **kw,
    )


@known
def test_fills_happen_at_the_touch_after_the_latency(panel):
    sim = run(panel)
    # marks at 1.5 .. 6.5 s: 100.00, 100.20, 100.10, 100.10, 100.10, 100.10
    # held +1 from 1.5 to 2.5: +20 bps; to 3.5: -10; short from 3.5 on: flat price, 0
    assert sim.mid_pnl == pytest.approx([20, -10, 0, 0, 0])
    # buy 1 at the ask as-of 1.5 s, 100.01 (not 100.02, the ask at the decision): 1 bps
    # sell 2 at the bid as-of 3.5 s, 100.09: 2 x 1 bps; buy 1 at 100.11 as-of 5.5 s: 1 bps
    assert sim.spread_bps == pytest.approx([1, 0, 2, 0, 1])
    # the long bought at 100.01 and sold at 100.09 (+8), the short sold at 100.09
    # and bought at 100.11 (-2): 6 bps gross, which is 10 of mid P&L less 4 of spread
    assert sim.gross.sum() == pytest.approx(6)


@known
def test_fee_arithmetic_and_the_flip_as_two_fills(panel):
    sim = run(panel)
    # traded 1 + 2 + 1 = 4 units at 7.5 bps: 30 bps; the flip at 3 s is a close and an open
    assert sim.fee_paid == pytest.approx([7.5, 0, 15, 0, 7.5])
    assert sim.fills.tolist() == [1, 0, 2, 0, 1]
    m = sim.metrics()
    assert (m["fills"], m["fees_bps"], m["spread_bps"]) == (4, pytest.approx(30), pytest.approx(4))
    assert sim.net.sum() == pytest.approx(6 - 30)


@known
def test_per_hour_units(panel):
    m = run(panel).metrics()
    # 5 one-second steps are 1/720 of an hour
    assert m["hours"] == pytest.approx(1 / 720)
    assert m["net_bps_per_hour"] == pytest.approx(-24 * 720)  # -17,280
    assert m["gross_bps_per_hour"] == pytest.approx(6 * 720)
    assert m["turnover_per_hour"] == pytest.approx(4 * 720)  # 4 units traded
    assert m["net_bps_per_fill"] == pytest.approx(-24 / 4)
    assert m["gross_bps_per_fill"] == pytest.approx(6 / 4)


@known
def test_drawdown_round_trips_and_hit_rate(panel):
    sim = run(panel)
    # per-step net 11.5, -10, -17, 0, -8.5; cumulative 11.5, 1.5, -15.5, -15.5, -24
    assert sim.metrics()["max_drawdown_bps"] == pytest.approx(11.5 + 24)
    # long: 20 - 10 mid, open 1 + 7.5, close (half the flip) 1 + 7.5 -> -7
    # short: 0 mid, open (other half) 1 + 7.5, close 1 + 7.5 -> -17
    assert sim.run_net == pytest.approx([-7, -17])
    assert sim.metrics()["hit_rate"] == 0
    free = run(panel, fee_bps=0)
    # without fees: long 10 - 1 - 1 = 8, short 0 - 1 - 1 = -2
    assert free.run_net == pytest.approx([8, -2])
    assert free.metrics()["hit_rate"] == 0.5
    assert free.run_net.sum() == pytest.approx(free.gross.sum())


@known
def test_a_feed_delay_does_not_move_the_fills(panel):
    # Every row arrives 0.9 s after its stamp, so at 3 s the panel still shows the 0.0 and
    # 1.2 s rows (mid 100.00): the 2.2 s row arrives at 3.1 s.
    late = hand_panel(delay_s=0.9)
    assert late.frame["A.mid"].to_list() == pytest.approx([100.0, 100.0, 100.0, 100.2, 100.1])
    # The order sent at 3 s still reaches the book at stamp 3.5 s, the row stamped 3.2 s and
    # available at 4.1 s: the fill reads as-of available_at 3.5 + 0.9 = 4.4 s and sells at
    # 100.09. Every fill and mark lands where it did without the delay, so the P&L is the
    # hand example's: +8 on the long, -2 on the short, 6 gross. The decision saw stamps up
    # to 2.1 s and the fill is at 3.5 s: the latency plus the delay.
    sim = run(late)
    assert sim.mid_pnl == pytest.approx([20, -10, 0, 0, 0])
    assert sim.spread_bps == pytest.approx([1, 0, 2, 0, 1])
    assert sim.gross.sum() == pytest.approx(6)
    # Reading as-of available_at 3.5 s instead would see stamps up to 2.6 s and sell at the
    # 2.2 s row's 100.19: the fills of a sim with no latency at all, 25 gross (below).
    assert run(panel, latency_ms=0).gross.sum() == pytest.approx(25)


@known
def test_latency_zero_fills_at_the_decision_quotes(panel):
    sim = run(panel, latency_ms=0)
    # marks at 1..6 s: 100.00, 100.00, 100.20, 100.10, 100.10, 100.10
    assert sim.mid_pnl == pytest.approx([0, 20, 10, 0, 0])
    # buy at 100.02 (2 bps), sell 2 at 100.19 (2 x 1), buy at 100.11 (1)
    assert sim.spread_bps == pytest.approx([2, 0, 2, 0, 1])
    # long 100.02 -> 100.19 is +17, short 100.19 -> 100.11 is +8
    assert sim.gross.sum() == pytest.approx(25)


@known
def test_min_rebalance_ignores_small_changes(panel):
    sim = run(panel, position=np.array([0.05, 0.3, 0.25, 0, 0]), fee_bps=0, min_rebalance=0.1)
    # 0.05 from flat: ignored; 0.3: traded; 0.25 is 0.05 away: ignored; 0: traded
    assert sim.position.tolist() == [0, 0.3, 0.3, 0, 0]
    assert sim.fills.tolist() == [0, 1, 0, 1, 0]
    # 0.3 held from 2.5 to 3.5 s: 0.3 x (100.10 - 100.20) = -3 bps; then flat price
    # spread 0.3 x 1 bps on the buy at 100.21 and on the sell at 100.09
    assert sim.mid_pnl.sum() == pytest.approx(-3)
    assert sim.gross.sum() == pytest.approx(-3.6)


@known
def test_sharpe_from_block_pnl(panel):
    sim = run(panel, block=pd.Timedelta("2s"))
    # blocks of two steps: 11.5 - 10 = 1.5 and -17 + 0, with the fifth step's -8.5 folded
    # into it, -25.5; mean -12, sd sqrt(2 * 13.5^2); 1,800 two-second blocks an hour:
    # -12 / (13.5 sqrt 2) x 30 sqrt 2 = -26.67 (dropping the fifth step gave -7.75 / 13.08
    # x 42.43 = -25.13)
    expect = -12 / math.sqrt(2 * 13.5**2) * math.sqrt(1800)
    assert sim.blocks == pytest.approx([1.5, -25.5])
    assert sim.metrics()["sharpe_per_sqrt_hour"] == pytest.approx(expect)


@known
def test_maker_proxy(panel):
    sim = run(panel, fee_bps=0)
    # 2 sells and 1 buy print inside the 5 s window: 0.4 and 0.2 a second
    pb, ps = 1 - math.exp(-0.4), 1 - math.exp(-0.2)
    got = maker_proxy(sim, panel, WINDOW, "A", horizon=pd.Timedelta("1s"))
    # buy 1 resting at 99.99 as-of 1.5, marked at 100.20 as-of 2.5: 21 bps
    # sell 2 resting at 100.11 as-of 3.5, marked 100.10: 1 bps each; buy 1 at 100.09: 1 bps
    assert got["fill_probability_buy"] == pytest.approx(pb)
    assert got["fill_probability_sell"] == pytest.approx(ps)
    assert got["net_bps_per_hour"] == pytest.approx((21 * pb + 2 * ps + 1 * pb) * 720)
    assert got["net_bps_per_fill"] == pytest.approx((21 + 2 + 1) / 4)
    assert got["fills"] == pytest.approx(2 * pb + ps)


@known
def test_maker_proxy_counts_n_trades_a_thinned_row_stands_for():
    # the same prints standing for 5, 2, 3 and 1 raw trades: inside the window the sells
    # at 2.6 and 4.4 s are 2 + 1 = 3 trades and the buy at 3.7 s is 3, 0.6 a second each
    p = hand_panel(n=[5, 2, 3, 1])
    sim = run(p, fee_bps=0)
    got = maker_proxy(sim, p, WINDOW, "A", horizon=pd.Timedelta("1s"))
    prob = 1 - math.exp(-0.6)
    assert got["fill_probability_buy"] == pytest.approx(prob)
    assert got["fill_probability_sell"] == pytest.approx(prob)
    assert got["net_bps_per_hour"] == pytest.approx((21 + 2 + 1) * prob * 720)


@known
def test_report_rows(panel):
    raw = {
        "data": {"path": "x.parquet", "instrument_column": "s", "streams": {}},
        "target": {"instrument": "A", "horizons": ["1s"]},
        "grid": {"step": "1s", "warmup": "0s"},
        "execution": {"latency_ms": 500, "taker_fee_bps": 7.5},
        "splits": {"embargo": "1min"},
    }
    rows = report(POSITION, panel, WINDOW, briefs.parse(raw))
    assert rows["row"].to_list() == [
        "brief", "latency 0", "fees 0", "latency 0, fees 0", "maker proxy",
    ]  # fmt: skip
    net = dict(zip(rows["row"], rows["gross_bps_per_hour"] / 720, strict=True))
    assert net["brief"] == pytest.approx(6) and net["latency 0"] == pytest.approx(25)
    assert rows["note"][1].startswith("unrealistic") and rows["note"][4].startswith("upper bound")


@edge
def test_held_clips_and_treats_nan_as_flat():
    assert held(np.array([np.nan, 2.0, -3.0]), max_position=1.5).tolist() == [0, 1.5, -1.5]
    # exactly min_rebalance away is traded
    assert held(np.array([0.1, 0.15]), min_rebalance=0.1).tolist() == [0.1, 0.1]


def scale_frame(values, step_s=1):
    times = [T0 + timedelta(seconds=step_s * i) for i in range(len(values))]
    return pl.DataFrame({"ts": times, "f": values})


@known
def test_walk_forward_scale():
    # 2 s clock windows: rows 0-1, 2-3, 4-5
    frame = scale_frame([1.0, 3.0, 5.0, 7.0, 2.0, 4.0])
    got = frame.select(walk_forward_scale(pl.col("f"), "2s"))["f"].to_list()
    # first window: no history; second: sd(1, 3) = sqrt(2); third: sd(1, 3, 5, 7) = 2.582
    assert got[:2] == [None, None]
    assert got[2:4] == pytest.approx([math.sqrt(2)] * 2)
    assert got[4:] == pytest.approx([math.sqrt(20 / 3)] * 2)


@known
def test_walk_forward_scale_skips_nan():
    # NaN in rows 0 and 3 is missing data; the rest is the frame above shifted by one window
    nan = float("nan")
    frame = scale_frame([nan, 1.0, 3.0, nan, 5.0, 7.0, 2.0, 4.0])
    got = frame.select(walk_forward_scale(pl.col("f"), "2s"))["f"].to_list()
    # windows 1-2 have under two values behind them; then sd(1, 3), then sd(1, 3, 5, 7)
    assert got[:4] == [None] * 4
    assert got[4:6] == pytest.approx([math.sqrt(2)] * 2)
    assert got[6:] == pytest.approx([math.sqrt(20 / 3)] * 2)


@known
def test_linear_treats_nan_as_null():
    nan = float("nan")
    with_nan = scale_frame([1.0, 3.0, 5.0, nan, 2.0, 4.0])
    with_null = scale_frame([1.0, 3.0, 5.0, None, 2.0, 4.0])
    a = with_nan.select(linear(pl.col("f"), max_position=10, every="2s"))["f"].to_list()
    b = with_null.select(linear(pl.col("f"), max_position=10, every="2s"))["f"].to_list()
    assert a[3] is None and a == b


@known
def test_linear_divides_by_the_walk_forward_scale():
    frame = scale_frame([1.0, 3.0, 5.0, 7.0, 2.0, 4.0])
    got = frame.select(linear(pl.col("f"), max_position=10, every="2s"))["f"].to_list()
    s2, s3 = math.sqrt(2), math.sqrt(20 / 3)
    assert got[2:] == pytest.approx([5 / s2, 7 / s2, 2 / s3, 4 / s3])
    clipped = frame.select(linear(pl.col("f"), every="2s"))["f"].to_list()
    assert clipped[2:] == pytest.approx([1, 1, 2 / s3, 1])


@known
def test_threshold_hysteresis():
    # rows 0-1 are -1/sqrt(2) and +1/sqrt(2), so the scale of the next 2 s window is 1
    r = 1 / math.sqrt(2)
    z = [0.0, 2.5, 1.0, 0.3, -2.1, -1.0, 0.4, None]
    frame = scale_frame([-r, r, *z], step_s=1)
    out = frame.select(threshold(pl.col("f"), z_in=2.0, z_out=0.5, every="2s"))
    got = out.to_series().to_list()
    # rows 0-1 have no scale yet: flat. Rows 2-3 divide by 1: 0 -> flat, 2.5 -> long.
    # Later windows refit the scale, so the state machine itself is checked below.
    assert got[:4] == [0.0, 0.0, 0.0, 1.0]


@known
def test_threshold_state_machine_with_a_fixed_scale(monkeypatch):
    import blinded.sim as simmod

    monkeypatch.setattr(simmod, "_scale", lambda f, every: pl.lit(1.0))
    z = [0.0, 2.5, 1.0, 0.3, -2.1, -1.0, 0.4, None, 3.0, -0.6, 0.49]
    got = scale_frame(z).select(simmod.threshold(pl.col("f"))).to_series().to_list()
    # flat; above 2: long; inside the band: still long; under 0.5: flat; below -2: short;
    # still short; flat; missing: no change; long; -0.6 is inside the band: long; flat
    assert got == [0, 1, 1, 0, -1, -1, 0, 0, 1, 1, 0]


@known
def test_a_leading_nan_does_not_fix_the_position():
    # 1 s windows, so each row's scale is the sd of the rows before it, NaN left out.
    # Rows 0-2 have under two values behind them: flat. Row 3: sd(-r, r) = 1, z = 0: flat.
    # Row 4: sd(-r, r, 0) = sqrt(1/2), z = 2.5 * sqrt(2) = 3.54: long.
    r = 1 / math.sqrt(2)
    frame = scale_frame([float("nan"), -r, r, 0.0, 2.5])
    got = frame.select(threshold(pl.col("f"), z_in=2.0, z_out=0.5, every="1s")).to_series()
    assert got.to_list() == [0.0, 0.0, 0.0, 0.0, 1.0]


@known
def test_threshold_holds_through_nan_as_through_null(monkeypatch):
    import blinded.sim as simmod

    monkeypatch.setattr(simmod, "_scale", lambda f, every: pl.lit(1.0))
    z = [0.0, 2.5, float("nan"), 0.3, -2.1, float("nan"), 0.4]
    got = scale_frame(z).select(simmod.threshold(pl.col("f"))).to_series().to_list()
    # long, NaN: still long, flat, short, NaN: still short, flat
    assert got == [0, 1, 1, 0, -1, -1, 0]


@edge
@pytest.mark.parametrize("build", [walk_forward_scale, linear, threshold])
def test_the_feature_is_computed_once(build):
    # polars computes each copy of a subexpression again, and the scale uses its input
    # about thirty times over: a costly feature passed in whole was computed that often
    calls = []

    def counted(s):
        calls.append(len(s))
        return s

    f = pl.col("f").map_batches(counted, return_dtype=pl.Float64)
    scale_frame([1.0, 3.0, 5.0, 7.0, 2.0, 4.0]).select(build(f, every="2s"))
    assert calls == [6]


@edge
def test_threshold_rejects_an_inverted_band():
    with pytest.raises(ValueError, match="z_out"):
        threshold(pl.col("f"), z_in=0.5, z_out=1.0)


STRATEGIES = """
import polars as pl

import blinded as bl
from blinded import features as F


@bl.feature("perp leads spot")
def basis(panel):
    b = F.basis("XRPUSDT_PERP", "XRPUSDT")
    return b - F.roll_mean(b, "5m")


@bl.strategy(requires=["basis"])
def follow(panel, features):
    return bl.threshold(features["basis"], z_in=1.5, z_out=0.25, every="2m")


@bl.strategy(requires=["basis"])
def sized(panel, features):
    return bl.linear(features["basis"], every="2m")


@bl.strategy(requires=["basis"])
def peek(panel, features):
    return pl.col("XRPUSDT.mid").shift(-10) > pl.col("XRPUSDT.mid")


@bl.strategy(requires=["nowhere"])
def orphan(panel, features):
    return features["nowhere"]
"""


@pytest.fixture(scope="module")
def strategies(tmp_path_factory):
    path = tmp_path_factory.mktemp("strat") / "strategies.py"
    path.write_text(textwrap.dedent(STRATEGIES))
    return {s.name: s for s in bl.load(path).strategies}


@known
def test_built_in_strategies_pass_the_truncation_test(
    strategies, synthetic_brief, synthetic_panel
):
    for name in ("follow", "sized"):
        result = strategies[name].evaluate(synthetic_brief, panel=synthetic_panel)
        assert result.ok, result.reason
        # the synthetic file keeps every book row with its sizes and every trade: a sixth
        # row executes the positions on the fill engine
        assert result.rows["row"].to_list() == ["brief", "latency 0", "fees 0",
                                                "latency 0, fees 0", "maker proxy",
                                                "maker-taker"]  # fmt: skip
        brief, free = result.row("brief"), result.row("latency 0, fees 0")
        assert brief["fills"] > 20
        # with fees, every unit traded costs 7.5 bps more than without
        assert brief["fees_bps"] == pytest.approx(
            7.5 * brief["turnover_per_hour"] * brief["hours"]
        )
        assert free["fees_bps"] == 0
    assert "taker sim" in repr(result) and "maker share" in repr(result)


@known
def test_a_leaky_strategy_is_rejected(strategies, synthetic_brief, synthetic_panel):
    result = strategies["peek"].evaluate(synthetic_brief, panel=synthetic_panel)
    assert result.status == "rejected_lookahead"
    assert result.rows.is_empty()


@edge
def test_a_missing_feature_is_an_error(strategies, synthetic_brief, synthetic_panel):
    result = strategies["orphan"].evaluate(synthetic_brief, panel=synthetic_panel)
    assert result.status == "error" and "nowhere" in result.reason


@known
def test_features_are_found_in_the_workspace_layout(tmp_path, synthetic_brief, synthetic_panel):
    (tmp_path / "features").mkdir()
    (tmp_path / "strategies").mkdir()
    feature_src, _, _ = STRATEGIES.partition("@bl.strategy")
    (tmp_path / "features" / "basis.py").write_text(feature_src)
    (tmp_path / "strategies" / "s.py").write_text(
        "import blinded as bl\n\n\n@bl.strategy(requires=['basis'])\n"
        "def s(panel, features):\n    return bl.threshold(features['basis'])\n"
    )
    (strat,) = bl.load(tmp_path / "strategies" / "s.py").strategies
    assert strat.evaluate(synthetic_brief, panel=synthetic_panel).ok


@cross
@pytest.mark.parametrize("latency_ms", [0, 50, 300])
def test_gross_equals_the_cash_from_fills(
    strategies, synthetic_brief, synthetic_panel, latency_ms
):
    from blinded.evaluate import scoring_window

    brief = briefs.load(synthetic_brief)
    result = strategies["follow"].evaluate(brief, panel=synthetic_panel)
    win = scoring_window(brief, synthetic_panel, "train")
    latency = pd.Timedelta(milliseconds=latency_ms)
    sim = simulate(result.position, synthetic_panel, win, "XRPUSDT", latency=latency)
    # independently: pay the ask or receive the bid on every change, mark what is left
    g = synthetic_panel.grid_ns
    i0, i1 = np.searchsorted(g, [win.start.value, win.end.value])
    t = g[i0:i1] + latency.value
    book = synthetic_panel.stream_of("XRPUSDT", "bid")
    h = sim.position
    d = np.diff(np.concatenate([[0.0], h]))
    cash = -np.sum(d * np.where(d > 0, book.asof("ask", t), book.asof("bid", t)))
    end = book.asof("mid", np.array([g[i1 - 1] + synthetic_panel.step_ns + latency.value]))[0]
    ref = book.asof("mid", t[:1])[0]
    assert sim.fills.sum() > 20
    assert sim.gross.sum() == pytest.approx((cash + h[-1] * end) / ref * 1e4, rel=1e-9, abs=1e-9)


@known
def test_the_hold_of_a_strategy_in_legs_counts_every_legs_runs():
    from blinded.sim import MultiSim, Sim, hold_steps

    def leg(h):
        return Sim(*[None] * 3, np.array(h, float), *[None] * 6, 1.0)

    a = leg([1, 1, 1, 1, 0, 0])  # one run of 4
    b = leg([-1, -1, 1, 1, 1, 1])  # runs of 2 and 4
    both = MultiSim(*[None] * 3, a.position, *[None] * 6, 1.0, legs={"A": a, "B": b})
    # (16 + 4 + 16) / (4 + 2 + 4)
    assert hold_steps(both) == pytest.approx(3.6)
    # the first leg alone, which is all MultiSim.position holds, would say 4
    assert hold_steps(a) == 4.0


def touch_brief(**execution):
    """A brief whose file keeps every book row with its sizes and every trade at its own price:
    the five-point book is then its own touch stream, and the sim table has a maker-taker
    row."""
    raw = {
        "data": {"path": "x.parquet", "instrument_column": "s",
                 "streams": {"book": {"fields": {"bid": "bid", "ask": "ask", "bid_qty": "bq",
                                                 "ask_qty": "aq"}},
                             "trades": {"fields": {"price": "p", "qty": "q", "side": "s"}}}},
        "target": {"instrument": "A", "horizons": ["1s"]},
        "grid": {"step": "1s", "warmup": "0s"},
        "execution": {"latency_ms": 500, "taker_fee_bps": 7.5, "deadline_ms": 1000}
        | execution,
        "splits": {"embargo": "1min"},
    }  # fmt: skip
    return briefs.parse(raw)


def taker_path(s):
    b = s.book
    return [(round((int(t) - int(pd.Timestamp(T0).value)) / 1e9, 1), int(side), round(p * 0.01, 2),
             float(q), bool(tk))
            for t, side, p, q, tk in zip(b.fill_ns, b.fill_side, b.fill_price, b.fill_qty,
                                         b.fill_taker, strict=True)]  # fmt: skip


@known
def test_maker_taker_posts_at_the_touch_and_crosses_at_the_deadline(panel):
    from blinded import sim_maker

    # the +1, +1, -1, -1, 0 positions, each change posted at the touch as-of its arrival and
    # crossed at the far touch 1 s after it when not filled. Every row of the book shows one
    # unit a side, and every trade is one unit, so a queue model that joins behind the
    # displayed unit never fills: a trade at the price only uses up the unit ahead.
    def mt(model):
        return sim_maker.simulate_maker_taker(POSITION, panel, WINDOW, "A", model, latency=HALF,
                                              deadline=pd.Timedelta("1s"),
                                              taker_fee_bps=7.5)  # fmt: skip

    # 1 s: buy 1, posted at the bid as-of 1.5 s, 99.99 (the 1.2 s row). The 2.6 s sell at
    # 100.19 is above it, nothing trades at or through 99.99 by 2.5 s, and the deadline
    # crosses at the ask as-of 2.5 s, 100.21, where the taker row bought at 100.01.
    # 3 s: known long 1 (the 2.5 s fill is known at 3.0 s), target -1: sell 2 at the ask as-of
    # 3.5 s, 100.11, behind the unit displayed there. The 3.7 s buy of 1 at 100.11 uses up
    # that unit; at 4.5 s the deadline sells both at the bid, 100.09.
    # 5 s: known short 1, target 0: buy 1 at the bid as-of 5.5 s, 100.09; no trade comes, and
    # its deadline is the end of the last step, 6.5 s, where it crosses at the ask, 100.11.
    path = [(2.5, 1, 100.21, 1.0, True), (4.5, -1, 100.09, 2.0, True),
            (6.5, 1, 100.11, 1.0, True)]  # fmt: skip
    for model in ("RA", "PR", "T"):
        s = mt(model)
        assert taker_path(s) == path, model
        assert [(int(x), int(p)) for x, p in zip(s.book.order_side, s.book.order_price,
                                                 strict=True)] == [(1, 9999), (-1, 10011),
                                                                   (1, 10009)]  # fmt: skip
        # long 1 from 100.21 to 100.09 is -12, short 1 from 100.09 to 100.11 is -2: -14
        # gross; four units at the 7.5 bps taker fee is 30; nothing rested, so no maker fee
        assert s.gross.sum() == pytest.approx(-14) and s.fee_paid.sum() == pytest.approx(30)
        assert s.net.sum() == pytest.approx(-44) and s.book.flatten is None
        row = sim_maker.maker_taker_row(s, pd.Timedelta("1s"))
        assert (row["fills"], row["maker_share"]) == (3, 0.0)
        assert row["net_bps_per_hour"] == pytest.approx(-44 * 720)
    # F is at the front: the 3.7 s buy of 1 at 100.11 fills one of the two units resting,
    # at the maker fee (0 here), and the deadline crosses the other at 100.09: gross -10 on
    # the long (100.21 to 100.11) and -2 on the short, 3 taker units, 22.5 bps
    f = mt("F")
    assert taker_path(f)[1] == (3.7, -1, 100.11, 1.0, False)
    assert f.gross.sum() == pytest.approx(-12) and f.fee_paid.sum() == pytest.approx(22.5)
    assert sim_maker.maker_taker_row(f, pd.Timedelta("1s"))["maker_share"] == 0.25
    # the sim table's sixth row is the worse of RA and PR: the same path, RA on the tie
    rows = report(POSITION, panel, WINDOW, touch_brief())
    assert rows["row"].to_list()[-1] == "maker-taker"
    by = rows.filter(pl.col("row") == "maker-taker").to_dicts()[0]
    assert (by["fills"], by["maker_share"]) == (3, 0.0)
    assert by["net_bps_per_hour"] == pytest.approx(-44 * 720)
    assert "1s deadline, under RA" in by["note"]
    brief = rows.filter(pl.col("row") == "brief").to_dicts()[0]
    assert brief["net_bps_per_hour"] == pytest.approx(-24 * 720)  # the taker row, unchanged
    assert brief["maker_share"] is None  # a column of the maker-taker row alone


@known
def test_maker_taker_keeps_a_resting_order_while_its_target_stands(panel):
    from blinded import sim_maker

    # with a 3 s deadline the 1 s buy (99.99 from 1.5 s, deadline 4.5 s) is still resting at
    # 2 s, when the target is the same, and is kept rather than posted again at the 2.2 s
    # touch. At 3 s the target turns to -1: it is cancelled at 3.5 s, and since nothing
    # filled the sell is one unit, at 100.11 from 3.5 s. The 3.7 s buy uses up the unit
    # ahead of it; at 4 s the target stands; at 5 s it is flat, the sell is cancelled at
    # 5.5 s, and the known inventory already is the target, so nothing is sent. Nothing
    # traded: a change reversed before its deadline costs nothing in this row.
    s = sim_maker.simulate_maker_taker(POSITION, panel, WINDOW, "A", "RA", latency=HALF,
                                       deadline=pd.Timedelta("3s"),
                                       taker_fee_bps=7.5)  # fmt: skip
    b = s.book
    assert [(int(x), int(p), float(z), st) for x, p, z, st in
            zip(b.order_side, b.order_price, b.order_size, b.order_status, strict=True)] == [
        (1, 9999, 1.0, "cancelled"), (-1, 10011, 1.0, "cancelled")]  # fmt: skip
    assert [(int(x) - int(pd.Timestamp(T0).value)) / 1e9 for x in b.order_end] == [3.5, 5.5]
    assert taker_path(s) == []
    assert s.position.tolist() == [0.0] * 5 and s.net.sum() == 0.0


@edge
def test_the_deadline_is_a_brief_field():
    assert touch_brief().execution.deadline == pd.Timedelta("1s")
    assert touch_brief(deadline_ms=None).execution.deadline == pd.Timedelta("2s")
    with pytest.raises(ValueError, match="deadline_ms must be positive"):
        touch_brief(deadline_ms=0)


@known
def test_under_rules_3_each_fee_is_on_its_trades_notional(panel):
    # the trades above at P_ref, the mark at the first fill, 100.00: buy 1 at 100.01, sell 2 at
    # 100.09, buy 1 at 100.11. Under rules 2 each paid 7.5 bps a unit; under rules 3 7.5 bps
    # of its notional over P_ref (the statistics audit's c1_feebasis.py)
    old, new = run(panel, rules=2), run(panel, rules=3)
    assert old.fee_paid == pytest.approx([7.5, 0, 15, 0, 7.5])
    assert new.fee_paid == pytest.approx([7.5 * 1.0001, 0, 15 * 1.0009, 0, 7.5 * 1.0011])
    assert new.spread_bps == pytest.approx(old.spread_bps) and new.fills.tolist() == [
        1,
        0,
        2,
        0,
        1,
    ]


@known
def test_under_rules_3_a_taker_pays_to_leave_its_last_position(panel):
    # long 1 from 1.5 s to the end. The end decision would be at 6 s, its fill at 6.5 s: the
    # bid as-of 6.5 s is 100.09 against a mark of 100.10, so leaving costs 1 bps of spread and
    # the fee on 100.09, both in the last step. Under rules 2 the last position was marked
    # to 100.10 and left, free (the simulator audit's c2_exit.py)
    long = np.ones(5)
    old, new = run(panel, position=long, rules=2), run(panel, position=long, rules=3)
    assert old.spread_bps == pytest.approx([1, 0, 0, 0, 0])
    assert new.spread_bps == pytest.approx([1, 0, 0, 0, 1])
    assert old.fee_paid == pytest.approx([7.5, 0, 0, 0, 0])
    assert new.fee_paid == pytest.approx([7.5 * 1.0001, 0, 0, 0, 7.5 * 1.0009])
    assert (old.fills.sum(), new.fills.sum()) == (1, 2)
    assert new.mid_pnl == pytest.approx(old.mid_pnl)
    assert new.net.sum() == pytest.approx(old.net.sum() - 1 - 7.5 * 1.0009 - 7.5 * 0.0001)
