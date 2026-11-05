"""The death checks of the "How it dies" panel, on books small enough to follow
by hand: the five-point book of test_sim.py and the two-leg example of test_pairs.py."""

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest
from test_pairs import T0, make_brief, pair, prints, quotes
from test_sim import POSITION, WINDOW, at, hand_panel

from blinded import brief as briefs
from blinded import dies, sim
from blinded.brief import Window
from blinded.panel import Panel
from blinded.report import _rows, _run

pytestmark = pytest.mark.usefixtures("rules2")  # hand numbers under rules 2
known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case


def one_leg_brief(**execution):
    raw = {
        "data": {"path": "x.parquet", "instrument_column": "s", "streams": {}},
        "target": {"instrument": "A", "horizons": ["1s"]},
        "grid": {"step": "1s", "warmup": "0s"},
        "execution": {"latency_ms": 500, "taker_fee_bps": 7.5, **execution},
        "splits": {"embargo": "1min"},
        "stats": {"block": "5min"},
    }
    return briefs.parse(raw)


@pytest.fixture(scope="module")
def panel():
    return hand_panel()


def five_point(panel, latency_ms=500, fee_bps=7.5, position=POSITION, **kw):
    latency = pd.Timedelta(milliseconds=latency_ms)
    return sim.simulate(position, panel, WINDOW, "A", latency=latency, fee_bps=fee_bps, **kw)


# -- D1 costs ---------------------------------------------------------------------------


@known
def test_the_break_even_fee_of_the_five_point_book(panel):
    s = five_point(panel)
    # +6 bps gross (test_sim.py) on fills of 1, 2 and 1 units: four units, 1.5 bps a unit
    assert s.gross.sum() == pytest.approx(6)
    assert np.abs(s.trade).sum() == 4
    assert dies.breakeven_fee_bps(s.gross.sum(), np.abs(s.trade).sum()) == pytest.approx(1.5)
    # and at 7.5 bps a unit the four units pay 30: 6 - 30 = -24
    assert s.net.sum() == pytest.approx(-24)


@known
def test_the_fee_latency_grid_holds_the_report_rows(panel):
    brief = one_leg_brief()
    rows = _rows(sim.report(POSITION, panel, WINDOW, brief, "A"))
    s = five_point(panel)
    out = dies.costs(_run(POSITION, panel, WINDOW, brief, "A"), s, rows, brief)
    g = out["grid"]
    # a file that is not thinned keeps every stamp: 0, L/2, L and 2L
    assert g["latencies_ms"] == [0, 250, 500, 1000]
    assert g["fees"] == ["0", "1", "2.5", "5", "brief"]
    cell = {(lat, f): v for lat, row in zip(g["latencies_ms"], g["net_bps_per_hour"], strict=True)
            for f, v in zip(g["fees"], row, strict=True)}  # fmt: skip
    assert cell[(500, "brief")] == pytest.approx(rows["brief"]["net_bps_per_hour"])
    assert cell[(0, "0")] == pytest.approx(rows["latency 0, fees 0"]["net_bps_per_hour"])
    # by hand, over 5 s = 1/720 h: at 500 ms, +6 gross less 4 units of each fee
    for fee in ("0", "1", "2.5", "5"):
        assert cell[(500, fee)] == pytest.approx((6 - 4 * float(fee)) * 720)
    # at latency 0 the fills are 100.02, 100.19 and 100.11, +25 gross (test_sim.py)
    assert cell[(0, "0")] == pytest.approx(25 * 720)
    assert out["breakeven_fee_bps"] == pytest.approx(1.5)


@known
def test_the_grid_keeps_one_bar_on_bars():
    raw = one_leg_brief().raw | {"grid": {"step": "1h", "warmup": "0s"},
                                 "execution": {"latency_ms": 3_600_000}}  # fmt: skip
    raw["splits"] = {"embargo": "1d"}
    assert dies.latencies(briefs.parse(raw)) == [pd.Timedelta(0), pd.Timedelta("1h")]


# -- D3 capacity ----------------------------------------------------------------------


@known
def test_touch_shares_by_hand():
    # a buy of one unit against an ask size of 4
    assert dies.touch_shares(np.array([1.0]), np.array([4.0]))["median"] == 0.25
    # fills of 1 and 3 against 4 and 2: 0.25 and 1.5, median 0.875, half of them above 1
    out = dies.touch_shares(np.array([1.0, -3.0]), np.array([4.0, 2.0]))
    assert (out["shares"], out["median"], out["above_one"]) == ([0.25, 1.5], 0.875, 0.5)


@known
def test_capacity_reads_the_far_touch_at_each_fill():
    # the five-point book with sizes: the buy decided at 1 s fills at 1.5 s against the ask of
    # the 1.2 s row, size 4; the sell of 3 decided at 3 s fills at 3.5 s against the bid of
    # the 3.2 s row, size 2
    book = pl.DataFrame({"ts": at(0.0, 1.2, 2.2, 3.2), "bid": [99.98, 99.99, 100.19, 100.09],
                         "ask": [100.02, 100.01, 100.21, 100.11], "bid_qty": [9.0, 9.0, 9.0, 2.0],
                         "ask_qty": [9.0, 4.0, 9.0, 9.0]})  # fmt: skip
    trades = pl.DataFrame({"ts": at(0.5, 2.6, 3.7, 4.4), "price": [99.98, 100.19, 100.11, 100.09],
                           "qty": [1.0] * 4, "side": [-1.0, -1.0, 1.0, -1.0]})  # fmt: skip
    panel = Panel.from_frames(at(1, 2, 3, 4, 5), book={"A": book}, trades={"A": trades})
    s = five_point(panel, position=np.array([1.0, 1.0, -2.0, -2.0, -2.0]), max_position=3.0)
    assert s.trade.tolist() == [1, 0, -3, 0, 0]
    out = dies.capacity(s, panel, WINDOW, "A")["A"]
    assert (out["median"], out["above_one"], out["n"]) == (0.875, 0.5, 2)
    # 4 units traded against the three prints of size 1 inside [1 s, 6 s)
    assert out["participation"] == pytest.approx(4 / 3)


@edge
def test_a_book_without_sizes_has_no_touch_share(panel):
    book = panel.events("book", "A").frame.drop("bid_qty", "ask_qty")
    bare = Panel.from_frames(at(1, 2, 3, 4, 5), book={"A": book.drop("available_at")})
    out = dies.capacity(five_point(bare), bare, WINDOW, "A")["A"]
    assert out["median"] is None and out["touch"] == dies.NOT_MEASURABLE
    assert out["participation"] is None


# -- D4 worst window ------------------------------------------------------------------


@known
def test_window_nets_by_hand():
    out = dies.worst_window([10.0, -4.0, 7.0])
    assert out["worst_bps"] == -4 and out["concentration"] == pytest.approx(10 / 13)
    assert (out["windows"], out["losing_windows"]) == (3, 1)
    assert dies.worst_window([3.0, -5.0])["concentration"] is None  # total -2: none
    # cumulative 2, -1, -2, 3, 1: under water at the second and third steps, then the fifth;
    # the deepest fall is from 2 to -2
    net = np.array([2.0, -3.0, -1.0, 5.0, -2.0])
    assert dies.underwater_steps(net) == 2
    assert dies.max_drawdown(net) == 4


@cross
def test_window_nets_match_a_pandas_groupby():
    rng = np.random.default_rng(8)
    ts = pd.date_range("2024-01-01 00:00:07", periods=7200, freq="1s", tz="UTC")
    net = rng.normal(size=len(ts))
    got = dies.window_nets(net, ts.as_unit("ns").asi8, pd.Timedelta("5min").value)
    want = pd.Series(net, index=ts).groupby(ts.floor("5min")).sum()
    assert [pd.Timestamp(t, tz="UTC") for t, _ in got] == list(want.index)
    np.testing.assert_allclose([v for _, v in got], want.to_numpy(), rtol=0, atol=1e-12)


# -- D5 half-life ---------------------------------------------------------------------


@known
def test_a_signal_half_life_from_its_autocorrelation():
    # rho 0.5: one step halves it, -ln 2 / ln 0.5 = 1 step of 100 ms
    assert dies.signal_half_life(0.5, 0.1) == pytest.approx(0.1)
    assert dies.signal_half_life(1.0, 0.1) is None and dies.signal_half_life(-0.2, 0.1) is None
    # an AR(1) path with rho 0.5 reads its own half-life back, within its sampling error
    rng = np.random.default_rng(1)
    x = np.zeros(20_000)
    for i in range(1, len(x)):
        x[i] = 0.5 * x[i - 1] + rng.normal()
    out = dies.half_life(x, 0, len(x), 0.1)
    assert out["signal_half_life_s"] == pytest.approx(0.1, rel=0.05)


@known
def test_a_spread_half_life_against_the_random_walk_line():
    item = {"random_walk_half_life_s": 287.0, "windows": [
        {"segment": "val", "phi": 0.9, "half_life_s": 10.0, "reverts": True},
        {"segment": "val", "phi": 0.95, "half_life_s": 30.0, "reverts": True},
        {"segment": "val", "phi": 1.01, "half_life_s": None, "reverts": False},
        {"segment": "train", "phi": 0.5, "half_life_s": 1.0, "reverts": True}]}  # fmt: skip
    out = dies.half_life(None, 0, 0, 1.0, item, "val")
    # the median of 10 and 30, and one of the three validation windows did not revert
    assert out == {"spread_median_half_life_s": 20.0, "random_walk_half_life_s": 287.0,
                   "no_reversion_share": pytest.approx(1 / 3)}  # fmt: skip


# -- D6 adverse excursion ---------------------------------------------------------------


def one_run(mid, spread, fee, trade):
    h = np.cumsum(trade)
    prev = np.r_[0.0, h[:-1]]
    return h, [(prev, np.asarray(trade, float), np.asarray(mid, float),
                np.asarray(spread, float), np.asarray(fee, float))]  # fmt: skip


@known
def test_a_net_path_inside_one_run():
    # long from the first step; the entry costs 0.5 of spread and 0.5 of fee, and the mid
    # P&L is 3, -3, -1, 5: a net path of +2, -3, -1, +5, cumulative 2, -1, -2, 3
    h, parts = one_run([3, -3, -1, 5], [0.5, 0, 0, 0], [0.5, 0, 0, 0], [1, 0, 0, 0])
    (path,) = dies.run_paths(h, parts)
    assert path["path"] == pytest.approx([2, -1, -2, 3])
    assert (path["net"], path["cost"], path["length"]) == (pytest.approx(3), 1.0, 4)
    out = dies.mae([path])
    assert (out["worst_bps"], out["median_close_bps"]) == (pytest.approx(-2), pytest.approx(3))
    # under water by 2, more than its cost of 1, and closed positive
    assert out["recovered_share"] == 1.0
    assert path["net"] == pytest.approx(sim._round_trips(h, parts[0][0], parts)[0])


@cross
def test_run_paths_add_up_to_the_sims_round_trips():
    book = {
        "A": quotes([99.98, 99.99, 100.19, 100.09], [100.02, 100.01, 100.21, 100.11]),
        "B": quotes([49.98, 49.99, 50.09, 50.04], [50.02, 50.01, 50.11, 50.06]),
    }
    trades = {
        "A": prints([0.5, 2.6, 3.7, 4.4], [99.98, 100.19, 100.11, 100.09], [-1.0, -1, 1, -1]),
        "B": prints([0.5, 2.6, 3.7], [49.98, 50.11, 50.04], [-1.0, 1, -1]),
    }
    panel = Panel.from_frames([T0 + timedelta(seconds=s) for s in (1, 2, 3, 4, 5)], book=book,
                              trades=trades)  # fmt: skip
    two = sim.simulate_legs(pair(), panel, WINDOW, make_brief())
    # the documented pair: round trips of -36 and -36 (test_pairs.py)
    paths = dies.run_paths(two.position, dies.parts(two))
    assert [p["net"] for p in paths] == pytest.approx([-36, -36])
    # and a seeded leg with flips, partial closes and rests at zero
    rng = np.random.default_rng(5)
    n = 400
    h = np.round(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n) * (rng.random(n) < 0.3), 1)
    h = pd.Series(np.where(h == 0, np.nan, h)).ffill(limit=3).fillna(0).to_numpy()
    prev = np.r_[0.0, h[:-1]]
    trade = h - prev
    parts = [(prev, trade, rng.normal(size=n), np.abs(trade) * 0.3, np.abs(trade) * 1.0)]
    nets = [p["net"] for p in dies.run_paths(h, parts)]
    np.testing.assert_allclose(nets, sim._round_trips(h, prev, parts), rtol=0, atol=1e-9)


# -- D9 breadth and D10 whipsaw ---------------------------------------------------------


@known
def test_breadth_of_three_legs():
    out = dies.breadth({"A": 5.0, "B": -2.0, "C": 1.0})
    # two legs of three made money; the best leg made 5 of a total 4
    assert out["positive_share"] == pytest.approx(2 / 3)
    assert out["top_share"] == pytest.approx(5 / 4) and out["top_leg"] == "A"


@known
def test_whipsaw_from_round_trips_of_1_3_and_20_steps():
    assert dies.whipsaw([1, 3, 20], [-1.0, 2.0, 5.0], 2)["short_share"] == pytest.approx(1 / 3)
    # the same round trips as a simulation on a 1 s grid against a band whose low end is 2 s
    h = np.array([1.0, 0, -1, -1, -1, 0] + [1.0] * 20)
    prev = np.r_[0.0, h[:-1]]
    trade = h - prev
    mid = np.zeros(len(h))
    mid[0], mid[2], mid[6] = -1.0, 2.0, 5.0  # nets of -1, 2 and 5, no costs
    zeros = np.zeros(len(h))
    s = sim.Sim(pd.Timedelta(0), 0.0, len(h) / 3600, h, trade, mid, zeros, zeros,
                (trade != 0).astype(int), zeros, 3600.0)  # fmt: skip
    out = dies.whipsaws(s, pd.Timedelta("2s"), 10**9)
    assert out["lo_steps"] == 2 and out["round_trips"] == 3
    assert out["short_share"] == pytest.approx(1 / 3)
    # two of three won; mean win 3.5 against a mean loss of 1; the best tenth is one round
    # trip of three, ceil(0.3), which made 5 of the total 6
    assert out["hit_rate"] == pytest.approx(2 / 3) and out["win_loss"] == pytest.approx(3.5)
    assert out["best_tenth_share"] == pytest.approx(5 / 6)


# -- D2 race --------------------------------------------------------------------------


@known
def test_the_race_compares_returns_from_the_decision_and_from_the_fill():
    x = np.arange(1.0, 6.0)
    # from the decision the ranks agree exactly (rank IC 1); from the fill they are reversed
    out = dies.race(x, {"1s": x.copy()}, {"1s": x[::-1].copy()}, 0, 5, 1.0, 3.0, 1.0)
    assert (out["peak_horizon"], out["peak_rank_ic"]) == ("1s", pytest.approx(1.0))
    assert out["rank_ic_left"] == pytest.approx(-1.0)
    assert out["ic_left_share"] == pytest.approx(-1.0)
    # an entry of 3 bps against a half spread of 1: 2 bps moved while the order flew
    assert out["in_flight_bps"] == 2.0
    # a feature traded against its sign (direction -1) reads the same curves negated
    flipped = dies.race(-x, {"1s": x.copy()}, {"1s": x[::-1].copy()}, 0, 5, -1.0, None, 1.0)
    assert flipped["peak_rank_ic"] == pytest.approx(1.0) and flipped["in_flight_bps"] is None
    assert math.isclose(flipped["ic_left_share"], -1.0)


@known
def test_d11_reads_the_second_venues_cadence_back():
    # B's book on a 20 ms beat over 10 s; the curve peaks at 40 ms (C 0.9), and the lag
    # nearest the 20 ms cadence is 20 ms (C 0.3): the peak is past the cadence
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    at = [t0 + pd.Timedelta(milliseconds=20 * i) for i in range(500)]
    book = pl.DataFrame({"ts": at, "bid": 1.0, "ask": 1.1, "bid_qty": 1.0, "ask_qty": 1.0})
    grid = [t0 + pd.Timedelta(seconds=s) for s in range(1, 10)]
    panel = Panel.from_frames(grid, book={"A": book.clone(), "B": book})
    win = Window("train", t0, t0 + pd.Timedelta("10s"))
    curve = {"leader": "A", "follower": "B", "lags_ms": [0.0, 20.0, 40.0, 100.0],
             "segments": {"train": {"corr": [0.1, 0.3, 0.9, 0.2], "lag_at_peak_ms": 40.0,
                                    "c_at_peak": 0.9}}}  # fmt: skip
    out = dies.venue_feed(panel, win, "B", [curve], "train")
    assert (out["median_gap_ms"], out["bin_max_ms"], out["share_max"], out["ok"]) == (
        pytest.approx(20.0),
        20,
        1.0,
        True,
    )
    (c,) = out["curves"]
    assert (c["cadence_lag_ms"], c["c_at_cadence"], c["peak_within_cadence"]) == (20.0, 0.3, False)
