"""The carry sim, its target and ``bl.carry``, on the five-point example of ``test_pairs.py``.

Grid 1, 2, 3, 4, 5 s, latency 0.5 s, fee 7.5 bps; fills at 1.5, 2.5, 3.5, 4.5 and
5.5 s. Leg A asks for +1, +1, -1, -1, 0 and leg B, now the perpetual, for the
opposite. B's book (time: bid / ask, mid):

    0.0: 49.98 / 50.02, 50.00
    1.2: 49.99 / 50.01, 50.00
    2.2: 50.09 / 50.11, 50.10
    3.2: 50.04 / 50.06, 50.05

B's reference notional is its mark at the first fill, 1.5 s: 50.00. B settles
funding at 2.0 s (rate +0.0001) and at 4.0 s (rate -0.0002).
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

import blinded as bl
from blinded import Panel, dies
from blinded import brief as briefs
from blinded.brief import Window
from blinded.contract import Legs
from blinded.durations import label
from blinded.sim import report, simulate, simulate_legs
from blinded.target import carry_returns, feature_returns, spread_returns

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


def quotes(bids, asks, seconds=(0.0, 1.2, 2.2, 3.2)):
    n = len(bids)
    return pl.DataFrame({"ts": at(*seconds), "bid": bids, "ask": asks,
                         "bid_qty": [1.0] * n, "ask_qty": [1.0] * n})  # fmt: skip


def prints(seconds, prices, sides):
    n = len(seconds)
    return pl.DataFrame({"ts": at(*seconds), "price": prices, "qty": [1.0] * n, "side": sides})


def settlements(seconds, rates):
    return pl.DataFrame({"ts": at(*seconds), "rate": rates})


def pair_panel(funding=None, borrow=None):
    book = {
        "A": quotes([99.98, 99.99, 100.19, 100.09], [100.02, 100.01, 100.21, 100.11]),
        "B": quotes([49.98, 49.99, 50.09, 50.04], [50.02, 50.01, 50.11, 50.06]),
    }
    trades = {
        "A": prints([0.5, 2.6, 3.7, 4.4], [99.98, 100.19, 100.11, 100.09], [-1.0, -1, 1, -1]),
        "B": prints([0.5, 2.6, 3.7], [49.98, 50.11, 50.04], [-1.0, 1, -1]),
    }
    return Panel.from_frames(at(1, 2, 3, 4, 5), book=book, trades=trades, funding=funding,
                             borrow=borrow)  # fmt: skip


FUNDING = {"B": settlements([2.0, 4.0], [0.0001, -0.0002])}


def make_brief(**execution):
    raw = {
        "data": {"path": "x.parquet", "instrument_column": "s", "streams": {}},
        "target": {"instrument": "A", "horizons": ["1s"]},
        "grid": {"step": "1s", "warmup": "0s"},
        "execution": {"latency_ms": 500, "taker_fee_bps": 7.5, **execution},
        "splits": {"embargo": "1min"},
        "families": {"pairs": {"pairs": [["A", "B"]]}},
    }
    return briefs.parse(raw)


def pair():
    return Legs({"A": LONG_A, "B": -LONG_A})


@known
def test_funding_accrues_at_each_settlement_by_hand():
    sim = simulate_legs(pair(), pair_panel(FUNDING), WINDOW, make_brief())
    a, b = sim.legs["A"], sim.legs["B"]
    # A has no funding stream, so no funding at all
    assert a.funding is None
    # at 2.0 s the live fill is the one at 1.5 s, B short one unit; B's mark as-of 2.0 s is
    # the 1.2 s row, 50.00, against a reference of 50.00: -(-1)(0.0001)(1e4)(1) = +1.0 bps.
    # At 4.0 s the live fill is 3.5 s, B long one; the mark is the 3.2 s row, 50.05:
    # -(+1)(-0.0002)(1e4)(50.05 / 50.00) = +2.002 bps, booked at the 3.5 s step
    np.testing.assert_allclose(b.funding, [1.0, 0.0, 2.002, 0.0, 0.0], rtol=0, atol=1e-12)
    assert sim.metrics()["funding_bps"] == pytest.approx(3.002, abs=1e-12)
    # the pair nets -72 without funding (test_pairs.py): -72 + 3.002
    assert sim.net.sum() == pytest.approx(-68.998, abs=1e-12)
    assert sim.metrics()["net_bps_per_hour"] == pytest.approx(-68.998 * 720, abs=1e-9)
    # funding is not a fee and not a price move: gross, fees and mid are those of test_pairs
    assert sim.gross.sum() == pytest.approx(-12, abs=1e-12)
    assert sim.fee_paid.sum() == pytest.approx(60, abs=1e-12)
    # round trips are A's runs: the long run (steps 0, 1) gets B's +1.0, -36 + 1.0; the short
    # run (steps 2, 3) gets +2.002, -36 + 2.002
    np.testing.assert_allclose(sim.run_net, [-35.0, -33.998], rtol=0, atol=1e-12)
    assert sim.run_net.sum() == pytest.approx(sim.net.sum(), abs=1e-12)


@known
def test_a_stamp_exactly_on_a_fill_counts_the_position_after_it():
    # +0.0001 at 3.5 s, the fill that turns B from short to long: B is long after it, and
    # pays -(+1)(0.0001)(1e4)(50.05 / 50.00) = -1.001 at that step
    on = simulate_legs(pair(), pair_panel({"B": settlements([3.5], [0.0001])}), WINDOW,
                       make_brief()).legs["B"]  # fmt: skip
    np.testing.assert_allclose(on.funding, [0.0, 0.0, -1.001, 0.0, 0.0], rtol=0, atol=1e-12)
    # 0.1 ms earlier the short of the 2.5 s step is live: +1.001 at step 1
    before = simulate_legs(pair(), pair_panel({"B": settlements([3.4999], [0.0001])}), WINDOW,
                           make_brief()).legs["B"]  # fmt: skip
    np.testing.assert_allclose(before.funding, [0.0, 1.001, 0.0, 0.0, 0.0], rtol=0, atol=1e-12)


@known
def test_settlements_outside_the_held_span_pay_nothing():
    # before the first fill (1.5 s) the leg is flat; from 6.5 s, the end of the last step,
    # the window is over; 6.4 s is still the last step's, where B is flat
    panel = pair_panel({"B": settlements([0.5, 1.4999, 6.4, 6.5], [0.01] * 4)})
    b = simulate_legs(pair(), panel, WINDOW, make_brief()).legs["B"]
    np.testing.assert_allclose(b.funding, np.zeros(5), rtol=0, atol=0)


@known
def test_every_report_row_but_the_maker_proxy_carries_the_funding():
    rows = report(pair(), pair_panel(FUNDING), WINDOW, make_brief())
    by = {r["row"]: r for r in rows.iter_rows(named=True)}
    # at latency 0 the fills are at 1, 2, 3, 4, 5 s. The 2.0 s stamp is on a fill: the
    # position after it, B short from the 2 s step, at the mark as-of 2.0 s, 50.00, over a
    # reference read at 1.0 s, 50.00: +1.0. The 4.0 s stamp is on the 4 s fill, B long, mark
    # 50.05: +2.002. Fees change no position. So all four taker rows carry 3.002
    for name in ("brief", "latency 0", "fees 0", "latency 0, fees 0"):
        assert by[name]["funding_bps"] == pytest.approx(3.002, abs=1e-12), name
    assert by["brief"]["net_bps_per_hour"] / 720 == pytest.approx(-72 + 3.002, abs=1e-9)
    assert by["fees 0"]["net_bps_per_hour"] / 720 == pytest.approx(-12 + 3.002, abs=1e-9)
    # the maker proxy is the edge of passive fills at the horizon, not a holding P&L: as before
    plain = {r["row"]: r for r in report(pair(), pair_panel(), WINDOW, make_brief())
             .iter_rows(named=True)}  # fmt: skip
    assert by["maker proxy"]["funding_bps"] is None
    assert by["maker proxy"]["net_bps_per_hour"] == plain["maker proxy"]["net_bps_per_hour"]


@known
def test_without_a_funding_stream_nothing_changes():
    sim = simulate_legs(pair(), pair_panel(), WINDOW, make_brief())
    assert sim.funding is None and all(x.funding is None for x in sim.legs.values())
    assert "funding_bps" not in sim.metrics()
    # zero rates accrue zeros, and every figure is the one without a stream
    zero = simulate_legs(pair(), pair_panel({"B": settlements([2.0, 4.0], [0.0, 0.0])}),
                         WINDOW, make_brief())  # fmt: skip
    assert zero.metrics()["funding_bps"] == 0.0
    plain, with_zero = sim.metrics(), zero.metrics()
    assert {k: v for k, v in with_zero.items() if k != "funding_bps"} == plain
    np.testing.assert_array_equal(zero.run_net, sim.run_net)


@known
def test_one_leg_on_a_perpetual_pays_funding_too():
    # B alone, short then long as in the pair: the same +1.0 and +2.002
    s = simulate(-LONG_A, pair_panel(FUNDING), WINDOW, "B", latency=pd.Timedelta("500ms"))
    np.testing.assert_allclose(s.funding, [1.0, 0.0, 2.002, 0.0, 0.0], rtol=0, atol=1e-12)
    assert s.metrics()["funding_bps"] == pytest.approx(3.002, abs=1e-12)


# -- D7 -----------------------------------------------------------------------------------


@known
def test_d7_splits_the_funding_by_side_and_sets_it_against_the_basis():
    panel = pair_panel(FUNDING)
    s = simulate_legs(pair(), panel, WINDOW, make_brief())
    d7 = dies.funding(s, panel, WINDOW, "B")
    # +1.0 while B was short (step 0), +2.002 while it was long (step 2)
    assert d7["funding_bps"] == pytest.approx(3.002, abs=1e-12)
    assert d7["short_bps"] == pytest.approx(1.0, abs=1e-12)
    assert d7["long_bps"] == pytest.approx(2.002, abs=1e-12)
    # A's marks at 1.5 .. 6.5 s are 100.00, 100.20, 100.10, then flat: mid +20, -10, 0, 0, 0
    # bps; B's -20, +10, 0, 0, 0 (test_pairs.py). The pair is hedged on the marks, so the
    # basis line is flat at zero and the funding line is all there is
    np.testing.assert_allclose(d7["lines"]["mid_bps"], np.zeros(5), rtol=0, atol=1e-9)
    np.testing.assert_allclose(d7["lines"]["funding_bps"], [1.0, 1.0, 3.002, 3.002, 3.002],
                               rtol=0, atol=1e-12)  # fmt: skip
    # two settlements in [1 s, 6 s): (0.0001 - 0.0002) / 2 = -0.5 bps
    assert (d7["settlements"], d7["mean_rate_bps"]) == (2, pytest.approx(-0.5, abs=1e-12))
    # D1 counts the funding in the gross: -12 + 3.002 over 8 units traded
    costs = dies.costs(lambda lat, fee: s, s, {}, make_brief())
    assert costs["gross_bps"] == pytest.approx(-8.998, abs=1e-12)
    assert costs["funding_bps"] == pytest.approx(3.002, abs=1e-12)
    assert costs["breakeven_fee_bps"] == pytest.approx(-8.998 / 8, abs=1e-12)
    assert dies.funding(simulate_legs(pair(), pair_panel(), WINDOW, make_brief()), pair_panel(),
                        WINDOW, "B") == {"note": dies.NO_FUNDING}  # fmt: skip


@known
def test_round_trip_paths_end_at_the_round_trips_with_funding_in_them():
    s = simulate_legs(pair(), pair_panel(FUNDING), WINDOW, make_brief())
    paths = dies.run_paths(s.position, dies.parts(s))
    assert [p["net"] for p in paths] == pytest.approx([-35.0, -33.998], abs=1e-12)


# -- the target ------------------------------------------------------------------------------

UP = 100 * np.exp(0.001)  # spot's mid after 2.0 s: a log return of exactly 10 bps


def carry_panel(funding_seconds, perp_after=50.0):
    """Spot S: 100 until 2.0 s, then UP; perpetual P: 50 throughout, or ``perp_after`` from
    2.6 s. One settlement of +0.0001 at each of ``funding_seconds``."""
    book = {"S": quotes([100.0, UP], [100.0, UP], (0.0, 2.0)),
            "P": quotes([50.0, perp_after], [50.0, perp_after], (0.0, 2.6))}  # fmt: skip
    funding = {"P": settlements(funding_seconds, [0.0001] * len(funding_seconds))}
    return Panel.from_frames(at(1, 2, 3, 4, 5), book=book, funding=funding)


def carry_at_1s(panel, h="2s"):
    """``carry_returns`` of S over P at the 1 s decision, latency 0.5 s: held 1.5 to 3.5 s."""
    out = carry_returns(panel, "S", "P", (pd.Timedelta(h),), pd.Timedelta("500ms"))
    return out[h].to_numpy()[0] * 1e4


@known
def test_carry_returns_add_the_funding_of_the_held_span():
    # the spread return from 1.5 to 3.5 s is ln(UP / 100) - ln(50 / 50) = 10 bps; one
    # settlement of +0.0001 inside [1.5, 3.5) at a mark ratio of 1 adds 1 bps: 11
    assert carry_at_1s(carry_panel([2.5])) == pytest.approx(11.0, abs=1e-9)
    # one stamped on the entry fill counts, as the sim books it to the position after it
    assert carry_at_1s(carry_panel([1.5])) == pytest.approx(11.0, abs=1e-9)
    # one just before the entry, or on the exit fill, is outside: 10
    assert carry_at_1s(carry_panel([1.4])) == pytest.approx(10.0, abs=1e-9)
    assert carry_at_1s(carry_panel([3.5])) == pytest.approx(10.0, abs=1e-9)
    # two inside: 12
    assert carry_at_1s(carry_panel([1.5, 3.4])) == pytest.approx(12.0, abs=1e-9)


@edge
def test_a_settlement_before_the_first_mark_leaves_the_later_returns_alone():
    # a settlement at 0.0 s, before any book row of P (the first is stamped 0.0 s too, so
    # move it to -1 s on a panel whose P book starts at 0.5 s): returns from 1.5 s are the 11
    book = {"S": quotes([100.0, UP], [100.0, UP], (0.0, 2.0)),
            "P": quotes([50.0], [50.0], (0.5,))}  # fmt: skip
    funding = {"P": settlements([-1.0, 2.5], [0.0001, 0.0001])}
    panel = Panel.from_frames(at(1, 2, 3, 4, 5), book=book, funding=funding)
    assert carry_at_1s(panel) == pytest.approx(11.0, abs=1e-9)


@known
def test_carry_returns_weigh_each_settlement_by_the_perpetual_move_since_entry():
    # P is 50 at entry (1.5 s) and 50.05 from 2.6 s, so a settlement at 3.0 s pays
    # 0.0001 x 50.05 / 50 = 1.001 bps of the entry notional; the spread return is now
    # 10 bps less ln(50.05 / 50)
    got = carry_at_1s(carry_panel([3.0], perp_after=50.05))
    assert got == pytest.approx(10.0 - np.log(50.05 / 50) * 1e4 + 1.001, abs=1e-9)


@known
def test_a_basis_feature_is_scored_against_carry_returns_and_a_pair_against_the_spread():
    panel = carry_panel([2.5])
    hs, lat = (pd.Timedelta("2s"),), pd.Timedelta("500ms")
    carry = feature_returns(panel, ("S", "P"), "mid", hs, lat, "carry")
    pairs = feature_returns(panel, ("S", "P"), "mid", hs, lat, "pairs")
    np.testing.assert_array_equal(
        carry["2s"].to_numpy(), carry_returns(panel, "S", "P", hs, lat)["2s"].to_numpy()
    )
    np.testing.assert_array_equal(
        pairs["2s"].to_numpy(), spread_returns(panel, "S", "P", hs, lat)["2s"].to_numpy()
    )


# -- bl.carry ---------------------------------------------------------------------------------


@known
def test_bl_carry_enters_on_a_high_rate_and_leaves_below_rate_out():
    legs = bl.carry("S", "P")
    rates = pl.DataFrame({"P.rate": [0.00005, 0.0002, 0.00015, -0.00001, 0.0]})
    got = rates.select(**legs)
    # 0.00005 is under rate_in: flat. 0.0002 is over it: long spot, short perp. 0.00015
    # holds (not under rate_out = 0). -0.00001 is under 0: flat. 0.0 is not under 0 and not
    # over 0.0001: still flat
    assert got["S"].to_list() == [0, 1, 1, 0, 0]
    assert got["P"].to_list() == [0, -1, -1, 0, 0]


@known
def test_bl_carry_goes_the_other_way_on_a_low_rate():
    legs = bl.carry("S", "P", rate_in=0.0001, rate_out=0.00002, max_position=0.5)
    rates = pl.DataFrame({"P.rate": [None, -0.0002, -0.00005, -0.00001, 0.0002, 0.00001,
                                     -0.0003, float("nan")]})  # fmt: skip
    got = rates.select(**legs)
    # no rate yet: flat. -0.0002 < -0.0001: short spot, long perp. -0.00005 holds (not above
    # -rate_out = -0.00002). -0.00001 is above it: flat. 0.0002: long spot. 0.00001 is under
    # rate_out: flat. -0.0003: short spot. A missing rate holds
    assert got["S"].to_list() == [0, -0.5, -0.5, 0, 0.5, 0, -0.5, -0.5]
    assert got["P"].to_list() == [0, 0.5, 0.5, 0, -0.5, 0, 0.5, 0.5]


@edge
def test_bl_carry_needs_rate_out_inside_rate_in():
    with pytest.raises(ValueError, match="need 0 <= rate_out <= rate_in"):
        bl.carry("S", "P", rate_in=0.0001, rate_out=0.0002)
    with pytest.raises(ValueError, match="need 0 <= rate_out <= rate_in"):
        bl.carry("S", "P", rate_out=-0.0001)


@cross_check
def test_bl_carry_reads_the_rate_as_of_the_decision():
    # settlements at 2.0 and 4.0 s of +0.0002 and -0.0002 on P; the panel's P.rate at 1..5 s
    # is the last settled one: none, 0.0002, 0.0002, -0.0002, -0.0002
    panel = Panel.from_frames(
        at(1, 2, 3, 4, 5),
        book={"P": quotes([50.0], [50.0], (0.0,))},
        funding={"P": settlements([2.0, 4.0], [0.0002, -0.0002])},
    )
    assert panel.frame["P.rate"].to_list() == [None, 0.0002, 0.0002, -0.0002, -0.0002]
    got = panel.frame.select(**bl.carry("S", "P"))
    assert got["S"].to_list() == [0, 1, 1, -1, -1]


# -- the funding rows of the carry file ------------------------------------------------------

HOUR_MS = 3_600_000


def settled(calc_ms, rates, interval=8):
    return pl.DataFrame({"calc_time": calc_ms, "funding_interval_hours": [interval] * len(rates),
                         "last_funding_rate": rates})  # fmt: skip


@known
def test_a_settlement_is_stamped_on_its_boundary_and_seen_a_second_later(tmp_path):
    from blinded import demo

    base = int(pd.Timestamp("2024-04-02", tz="UTC").value // 1_000_000)
    # calc_time 7 ms after midnight and 1 ms before 08:00: stamped at 00:00 and 08:00
    rows, found = demo.funding_rows(settled([base + 7, base + 8 * HOUR_MS - 1], [1e-4, -2e-5]),
                                    "X_PERP", "15m")  # fmt: skip
    assert rows["ts"].dt.epoch("ms").to_list() == [base, base + 8 * HOUR_MS]
    assert (rows["kind"].to_list(), rows["rate"].to_list()) == (["funding"] * 2, [1e-4, -2e-5])
    assert found == {"offset_ms": [-1, 7], "intervals": [8], "hours": [0, 8]}
    assert tuple(rows.columns) == demo.CARRY_LAYOUT
    # through a brief whose funding stream is available a second after its stamp, the
    # decision at 00:00 sees no rate (the settlement's calc_time, 00:00:00.007, is after it)
    # and the one at 00:15 sees 1e-4
    path = tmp_path / "f.parquet"
    rows.write_parquet(path)
    raw = {"data": {"path": str(path), "instrument_column": "symbol",
                    "streams": {"funding": {"filter": "kind == 'funding'",
                                            "fields": {"rate": "rate"}}},
                    "availability": {"default_delay_ms": 0, "overrides": [
                        {"stream": "funding", "delay_ms": 1000}]}},
           "target": {"instrument": "X_PERP", "horizons": ["15m"]},
           "grid": {"step": "15min", "start": "2024-04-02T00:00:00Z", "warmup": "0s"},
           "splits": {"embargo": "1h"}}  # fmt: skip
    panel = Panel.from_brief(briefs.parse(raw))
    assert panel.frame["X_PERP.rate"].to_list()[:2] == [None, 1e-4]
    ev = panel.events("funding", "X_PERP")
    assert (ev.delay_ns, ev.at[0] - base * 1_000_000) == (10**9, 10**9)


@edge
def test_a_settlement_off_its_boundary_or_two_on_one_fail_the_build():
    from blinded import demo

    base = int(pd.Timestamp("2024-04-02", tz="UTC").value // 1_000_000)
    with pytest.raises(demo.BuildError, match="lies 1000 ms from the nearest 15m boundary"):
        demo.funding_rows(settled([base + 1000], [1e-4]), "X_PERP", "15m")
    with pytest.raises(demo.BuildError, match="two settlements on one 15m boundary"):
        demo.funding_rows(settled([base + 3, base + 5], [1e-4, 1e-4]), "X_PERP", "15m")


@edge
def test_a_funding_file_with_another_header_fails(tmp_path):
    import zipfile

    from blinded import demo

    path = tmp_path / "X-fundingRate-2024-04.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("X.csv", "calc_time,last_funding_rate\n1711929600000,0.0001\n")
    with pytest.raises(demo.BuildError, match="the header is"):
        demo.read_funding(path)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("X.csv", f"{demo.FUNDING_HEADER}\n1711929600000,8,0.00070161\n")
    got = demo.read_funding(path)
    assert got.row(0) == (1711929600000, 8, 0.00070161)


# -- the spot cost day ----------------------------------------------------------------------

COINS = ["BTC", "ETH"]


def listing(types=("trades", "book_ticker"), since="2019-03-30T00:00:00.000Z"):
    return {"datasets": {"symbols": [
        {"id": f"{c}USDT", "type": "spot", "dataTypes": list(types), "availableSince": since,
         "availableTo": "2026-10-18T00:00:00.000Z"} for c in COINS]}}  # fmt: skip


@edge
def test_the_spot_book_is_verified_on_tardis_and_falls_back_a_month(tmp_path):
    from blinded import demo

    calls, lines = [], []
    # listed only from 2024-02-15: 2024-03-01 passes, so only the kline HEADs stop the build
    late = listing(since="2024-02-15T00:00:00.000Z")
    with pytest.raises(demo.BuildError, match="not on data.binance.vision"):
        demo.carry(tmp_path, coins=COINS, details=lambda ex: late, head=lambda url: None,
                   fetch_day=lambda *a: calls.append(a), log=lines.append)  # fmt: skip
    assert not any("passing over" in line for line in lines)
    # listed from 2024-03-05: 2024-03-01 is passed over with its reason, then 2024-02-01 too
    lines.clear()
    later = listing(since="2024-03-05T00:00:00.000Z")
    with pytest.raises(demo.BuildError, match="no Tardis day for the spot half spreads"):
        demo.carry(tmp_path, coins=COINS, details=lambda ex: later, head=lambda url: 1,
                   fetch_day=lambda *a: calls.append(a), log=lines.append)  # fmt: skip
    assert [line.split(":")[0] for line in lines] == [
        "passing over the spot book on 2024-03-01",
        "passing over the spot book on 2024-02-01",
    ]
    assert "binance BTCUSDT covers 2024-03-05 to 2026-10-18, not 2024-03-01" in lines[0]
    # no book_ticker at all
    assert demo.spot_book_problem(listing(types=("trades",)), "BTCUSDT", demo.CARRY_SPOT_DAYS[0]
                                  ) == "binance BTCUSDT has no book_ticker"  # fmt: skip
    assert calls == []  # nothing was downloaded


@known
def test_spot_half_spreads_come_from_every_two_sided_row_of_the_tardis_day(tmp_path):
    import gzip

    from blinded import demo

    day = demo.CARRY_SPOT_DAYS[0]
    fetched = []

    def fetch_day(exchange, data_types, symbols, d, raw):
        fetched.append((exchange, tuple(data_types), tuple(symbols), d))
        raw.mkdir(parents=True, exist_ok=True)
        # three two-sided rows at half spreads of 1, 1 and 3 bps (0.01, 0.01 and 0.03 on a mid
        # of 100) and one crossed row, dropped: the median is 1 bps
        body = (
            "exchange,symbol,timestamp,local_timestamp,ask_amount,ask_price,bid_price,"
            "bid_amount\n"
            + "".join(
                f"binance,{symbols[0]},1,2,1,{a},{b},1\n"
                for b, a in ((99.99, 100.01), (99.99, 100.01), (99.97, 100.03), (100.02, 100.0))
            )
        )
        path = raw / demo.raw_name(exchange, data_types[0], d, symbols[0])
        path.write_bytes(gzip.compress(body.encode()))

    table = demo.tardis_half_spreads(["BTCUSDT"], day, tmp_path / "raw", tmp_path / "costs",
                                     fetch_day, log=lambda s: None)  # fmt: skip
    row = table["symbols"]["BTCUSDT"]
    assert row["half_spread_bps"] == pytest.approx(1.0, abs=1e-9)
    assert (row["rows"], row["two_sided_rows"]) == (4, 3)
    assert fetched == [("binance", ("book_ticker",), ("BTCUSDT",), day)]
    # cached: a second call fetches nothing
    again = demo.tardis_half_spreads(["BTCUSDT"], day, tmp_path / "raw", tmp_path / "costs",
                                     fetch_day, log=lambda s: None)  # fmt: skip
    assert again == table and len(fetched) == 1


@cross_check
def test_the_carry_brief_gives_the_e_column_on_the_built_windows():
    from test_archetypes import DATASETS, TABLE_B, cell

    from blinded import archetypes, demo

    hs = {s: 1.0 for c in demo.CARRY_COINS for s in (f"{c}USDT", f"{c}USDT_PERP")}
    raw = demo.carry_brief("binance-carry8-2024-04-01-450d-15m.parquet", list(demo.CARRY_COINS),
                           hs, demo.CARRY_SPOT_DAYS[0], demo.CARRY_PERP_DAY,
                           demo.CARRY_START)  # fmt: skip
    raw["splits"]["windows"] = DATASETS["e"]["windows"]
    brief = briefs.parse(raw)
    menu = archetypes.menu(brief, archetypes.facts(brief))
    assert {n: cell(v) for n, v in menu.verdicts.items()} == {n: c[4] for n, c in TABLE_B.items()}
    # spot pays 10 bps and the perpetual 5; the funding stream is seen a second after its stamp
    assert (brief.execution.fee("BTCUSDT"), brief.execution.fee("BTCUSDT_PERP")) == (10, 5)
    assert brief.data.delay("funding", "BTCUSDT_PERP") == pd.Timedelta("1s")
    assert brief.data.delay("book", "BTCUSDT_PERP") == pd.Timedelta(0)
    assert brief.data.cost_model.day == pd.Timestamp("2024-03-27", tz="UTC")


# -- through the daemon, the report and replay, on a synthetic spot and perpetual -------------


def carry_file(path, days=36, seed=5, borrow=None):
    """A spot S and its perpetual S_PERP on 15-minute bars, with funding every 8 hours: the
    rate holds +5 bps for three days, then -3 bps for three, around a little noise, and the
    perpetual is the spot times a basis of under a basis point of noise. So the last settled
    rate predicts the next one, and the carry of the next 8 hours is mostly funding. With
    ``borrow``, ``[(day, daily rate), ...]``, S has a borrow row each day at the rate of the
    last change on or before it."""
    from blinded import demo

    rng = np.random.default_rng(seed)
    n = days * 96
    start = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    perp = close * np.exp(rng.normal(0, 5e-5, n))
    parts = []
    for inst, c in (("S", close), ("S_PERP", perp)):
        bars = pl.DataFrame({"open_time": start + np.arange(n) * 900_000,
                             "close_time": start + np.arange(n) * 900_000 + 899_999,
                             "close": c, "volume": 10.0, "quote_volume": c * 10, "count": 5,
                             "taker_buy_volume": 6.0})  # fmt: skip
        parts.append(demo.bar_rows(bars, inst, 1.0, "15m").with_columns(
            rate=pl.lit(None, pl.Float64)))  # fmt: skip
    k = np.arange(days * 3)
    regime = np.where((k // 9) % 2 == 0, 5e-4, -3e-4)
    rates = regime + rng.normal(0, 2e-5, len(k))
    calc = start + k * 8 * HOUR_MS + rng.integers(0, 10, len(k))  # a few ms late, as archived
    rows, _ = demo.funding_rows(settled(calc.tolist(), rates.tolist()), "S_PERP", "15m")
    parts.append(rows)
    if borrow:
        first = pd.Timestamp("2024-01-01").date()
        # a row at 00:00 every day, as the history has them, at the rate from the last change
        history = [[start + d * 24 * HOUR_MS, [r for day, r in borrow if day <= d][-1]]
                   for d in range(days)]  # fmt: skip
        parts.append(demo.borrow_rows(history, "S", first, first + timedelta(days=days))[0])
    pl.concat(parts).sort("ts", maintain_order=True).write_parquet(path)
    return path


def carry_brief_raw(data: str) -> dict:
    from blinded import demo

    # the builder's brief on one coin, renamed S and S_PERP, at its own fees and stamps
    raw = demo.carry_brief(data, ["S"], {"SUSDT": 1.0, "SUSDT_PERP": 1.0},
                           demo.CARRY_SPOT_DAYS[0], demo.CARRY_PERP_DAY,
                           pd.Timestamp("2024-01-01").date())  # fmt: skip
    inst = ["S", "S_PERP"]
    raw |= {"name": "carry-synthetic",
            "stats": {"block": "8h", "n_boot": 200, "seed": 1},
            "execution": raw["execution"] | {"fees": {"S": 10.0}},
            "archetypes": {"instruments": inst, "carry": [inst],
                           "params": {"basis_trade": {"rate_in": 0.0001, "rate_out": 0.0},
                                      "pairs_trading": {"window": "2d", "every": "4h",
                                                        "z_in": 2.0, "z_out": 0.5}}}}  # fmt: skip
    raw["target"]["instrument"] = "S_PERP"
    raw["data"]["path"] = data
    raw["data"]["cost_model"] |= {"source": "synthetic bookTicker 2023-12-31",
                                  "half_spread_bps": {"S": 1.0, "S_PERP": 1.0}}  # fmt: skip
    return raw


@pytest.fixture(scope="module")
def carry_run(tmp_path_factory):
    import yaml

    from blinded import vault

    root = tmp_path_factory.mktemp("carry")
    data = carry_file(root / "carry.parquet")
    brief = root / "carry.brief.yaml"
    brief.write_text(yaml.safe_dump(carry_brief_raw(str(data)), sort_keys=False))
    return vault.init(brief, tmp_path_factory.mktemp("carry-runs") / "run")


@pytest.fixture
def carry_copy(carry_run, tmp_path):
    import shutil

    from blinded.vault import RunDir

    shutil.copytree(carry_run.root, tmp_path / "run")
    return RunDir(tmp_path / "run")


BASIS = """
import polars as pl

import blinded as bl

carry = bl.basis_trade(spot="S", perp="S_PERP")


@carry.feature("a high last funding rate pays a short perpetual over the next 8 hours")
def last_rate(panel):
    return pl.col("S_PERP.rate")


@carry.strategy()
def collect(panel, features):
    return bl.carry("S", "S_PERP", rate_in=0.0001, rate_out=0.0)
"""


def judge(run, source, rid="r1"):
    import textwrap

    from blinded.daemon import Daemon

    (run.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
    (run.inbox / f"{rid}.json").write_text('{"kind": "eval", "file": "f.py", "name": null}')
    return Daemon(run).handle(run.inbox / f"{rid}.json")


def by_hand_funding(panel, win, latency, position_perp):
    """The perpetual's funding over ``win`` with pandas, apart from the sim: each settlement
    pays minus the position whose fill (decision plus latency) was the last at or before it,
    times the rate, times the mark there over the mark at the first fill."""
    g = pd.Series(panel.grid_ns)
    inside = ((g >= win.start.value) & (g < win.end.value)).to_numpy()
    fills = pd.DataFrame({"t": g[inside].to_numpy() + latency.value,
                          "h": np.asarray(position_perp, dtype=float)[inside]})  # fmt: skip
    marks = panel.events("book", "S_PERP")
    ref = marks.asof("mid", fills["t"].to_numpy()[:1])[0]
    ev = panel.events("funding", "S_PERP")
    f = pd.DataFrame({"t": ev.frame["ts"].dt.epoch("ns").to_numpy(), "rate": ev.column("rate")})
    end = fills["t"].iloc[-1] + panel.step_ns
    f = f[(f["t"] >= fills["t"].iloc[0]) & (f["t"] < end)]
    held = pd.merge_asof(f, fills, on="t", direction="backward", allow_exact_matches=True)
    m = marks.asof("mid", held["t"].to_numpy())
    return float((-held["h"] * held["rate"] * 1e4 * m / ref).sum())


@known
def test_a_basis_trade_is_scored_against_carry_and_simulated_with_funding(carry_copy):
    from blinded import contract
    from blinded.evaluate import scoring_window

    response = judge(carry_copy, BASIS)
    feat, strat = response["trials"]
    assert (feat["status"], strat["status"]) == ("accepted", "accepted"), response
    assert (feat["edge"], feat["archetype"], feat["legs"]) == ("carry", "basis_trade", "S/S_PERP")
    # the last rate ranks the next 8 hours' carry: the planted regimes; against the spread
    # return alone it would rank noise
    h8 = label(pd.Timedelta("8h"))
    ic = feat["train"]["horizons"][h8]["rank_ic"]
    assert ic > 0.5
    panel = carry_copy.panel("train")
    brief = carry_copy.brief()
    win = scoring_window(brief, panel, "train")
    lat = brief.latency
    x = panel.eval(pl.col("S_PERP.rate")).to_numpy()
    spread = spread_returns(panel, "S", "S_PERP", (pd.Timedelta("8h"),), lat)[h8].to_numpy()
    carry = carry_returns(panel, "S", "S_PERP", (pd.Timedelta("8h"),), lat)[h8].to_numpy()
    g = panel.grid_ns
    inside = (g >= win.start.value) & (g < win.end.value)
    frame = pd.DataFrame({"x": x[inside], "carry": carry[inside], "spread": spread[inside]})
    assert ic == pytest.approx(frame["x"].corr(frame["carry"], method="spearman"), abs=1e-9)
    assert abs(frame["x"].corr(frame["spread"], method="spearman")) < 0.2
    # the strategy's train funding, against pandas
    rows = strat["train"]["rows"]
    legs = contract.values(bl.carry("S", "S_PERP"), panel, ("S", "S_PERP"))
    want = by_hand_funding(panel, win, lat, legs["S_PERP"])
    assert rows["brief"]["funding_bps"] == pytest.approx(want, rel=1e-9)
    assert want > 0  # it collects in both regimes
    assert rows["maker proxy"].get("funding_bps") is None


@cross_check
def test_the_report_draws_the_basis_trade_with_its_funding_and_replays(carry_copy):
    from blinded import report

    judge(carry_copy, BASIS)
    diag = report.recompute(carry_copy, None)["diagnostics"]["archetypes"]
    basis = diag["carry"]["basis_trade"]
    (item,) = basis["pairs"]
    assert item["pair"] == ["S", "S_PERP"] and item["name"] == "collect"
    d7 = basis["dies"]["funding"]
    assert set(d7) == {"train", "val"}
    # D7's total is the strategy's own sim on the segment, and its lines end at the sums
    assert d7["train"]["funding_bps"] == pytest.approx(
        d7["train"]["short_bps"] + d7["train"]["long_bps"], abs=1e-9
    )
    assert d7["val"]["lines"]["funding_bps"][-1] == pytest.approx(d7["val"]["funding_bps"])
    # the settlements stamped in validation: one every 8 hours from 2024-01-01 00:00
    val = carry_copy.brief().splits.fixed["val"]
    stamps = pd.date_range("2024-01-01", periods=36 * 3, freq="8h", tz="UTC")
    assert d7["val"]["settlements"] == sum(val[0] <= t < val[1] for t in stamps) > 0
    assert basis["dies"]["mae"]["val"]["round_trips"] > 0
    path = report.report(carry_copy.root)
    html = path.read_text()
    assert "D7: what carry paid or cost" in html and "Funding against the basis" in html
    check = report.replay(carry_copy.root)
    assert check["ok"], [c for c in check["checks"] if not c["ok"]][:5]
    assert any(c["name"].endswith("funding.val.funding_bps") for c in check["checks"])


@known
def test_archetypes_md_prints_the_funding_on_train_and_the_fees(carry_run):
    text = (carry_run.workspace / "ARCHETYPES.md").read_text()
    menu = carry_run.menu()
    f = menu.facts["funding"]["S_PERP"]
    brief = carry_run.brief()
    win = brief.splits.fixed["train"]
    first = win[0] + brief.grid.warmup
    # the settlements stamped in train after the 1-day warm-up: every 8 hours from day 1
    assert f["settlements"] == int((win[1] - first) / pd.Timedelta("8h"))
    assert "Funding on train, the mean rate per settlement" in text
    assert f"S_PERP {f['mean_rate_bps']:.3g} bps" in text
    assert "costs 2 x (10 + 5) = 30 bps before any spread or basis move" in text


# -- borrow on a short spot leg ---------------------------------------------------------------


def rates(seconds, values):
    return pl.DataFrame({"ts": at(*seconds), "borrow_rate": values})


# 8.64 a day is 86,400 bps a day, a bps a second, so each one-second step short costs about a bps
BORROW = {"A": rates([0.0], [8.64])}


@known
def test_a_short_spot_leg_pays_borrow_by_hand():
    sim = simulate_legs(pair(), pair_panel(borrow=BORROW), WINDOW, make_brief())
    a, b = sim.legs["A"], sim.legs["B"]
    # A is short from the 3.5 s fill to the 5.5 s one: two steps of a second, each valued at
    # A's mark when it starts, 100.10 (the 3.2 s row), over a reference of 100.00:
    # -(1)(8.64)(1e4)(1 / 86,400)(100.10 / 100.00) = -1.001 a step. Long steps pay nothing
    np.testing.assert_allclose(a.borrow, [0.0, 0.0, -1.001, -1.001, 0.0], rtol=0, atol=1e-12)
    # B has no borrow stream, so its short pays nothing and it has no borrow at all
    assert b.borrow is None
    assert sim.metrics()["borrow_bps"] == pytest.approx(-2.002, abs=1e-12)
    # borrow is neither a fee nor a price move: gross and fees are those of test_pairs.py, and
    # the pair's -72 net loses the 2.002
    assert sim.gross.sum() == pytest.approx(-12, abs=1e-12)
    assert sim.fee_paid.sum() == pytest.approx(60, abs=1e-12)
    assert sim.net.sum() == pytest.approx(-74.002, abs=1e-12)
    # the round trips are A's runs: the long run as before, the short run -36 - 2.002
    np.testing.assert_allclose(sim.run_net, [-36.0, -38.002], rtol=0, atol=1e-12)


@known
def test_each_step_pays_the_rate_in_force_when_it_starts():
    # a rate published exactly at 3.5 s is in force for the step the 3.5 s fill starts; one
    # published at 3.6 s first applies to the 4.5 s step. 0 before either
    on = simulate_legs(pair(), pair_panel(borrow={"A": rates([0.0, 3.5], [0.0, 8.64])}),
                       WINDOW, make_brief()).legs["A"]  # fmt: skip
    np.testing.assert_allclose(on.borrow, [0.0, 0.0, -1.001, -1.001, 0.0], rtol=0, atol=1e-12)
    late = simulate_legs(pair(), pair_panel(borrow={"A": rates([0.0, 3.6], [0.0, 8.64])}),
                         WINDOW, make_brief()).legs["A"]  # fmt: skip
    np.testing.assert_allclose(late.borrow, [0.0, 0.0, 0.0, -1.001, 0.0], rtol=0, atol=1e-12)
    # and a rate that halves at 4.0 s halves the second step's
    half = simulate_legs(pair(), pair_panel(borrow={"A": rates([0.0, 4.0], [8.64, 4.32])}),
                         WINDOW, make_brief()).legs["A"]  # fmt: skip
    np.testing.assert_allclose(half.borrow, [0.0, 0.0, -1.001, -0.5005, 0.0], rtol=0, atol=1e-12)


@known
def test_every_taker_row_carries_the_borrow_of_its_own_fills():
    by = {r["row"]: r for r in report(pair(), pair_panel(borrow=BORROW), WINDOW, make_brief())
          .iter_rows(named=True)}  # fmt: skip
    for name in ("brief", "fees 0"):
        assert by[name]["borrow_bps"] == pytest.approx(-2.002, abs=1e-12), name
    # at latency 0 A is short from the 3 s fill to the 5 s one, valued at the marks as-of 3.0 s
    # (the 2.2 s row, 100.20) and 4.0 s (100.10) over a reference as-of 1.0 s (the 0.0 s row,
    # 100.00): -1.002 - 1.001
    for name in ("latency 0", "latency 0, fees 0"):
        assert by[name]["borrow_bps"] == pytest.approx(-2.003, abs=1e-12), name
    assert by["brief"]["net_bps_per_hour"] / 720 == pytest.approx(-72 - 2.002, abs=1e-9)


@known
def test_without_a_borrow_stream_or_a_short_nothing_is_owed():
    plain = simulate_legs(pair(), pair_panel(), WINDOW, make_brief())
    assert plain.borrow is None and "borrow_bps" not in plain.metrics()
    long = simulate(np.ones(5), pair_panel(borrow=BORROW), WINDOW, "A",
                    latency=pd.Timedelta("500ms"))  # fmt: skip
    np.testing.assert_array_equal(long.borrow, np.zeros(5))


@known
def test_the_agent_reads_the_borrow_and_the_judge_charges_it(tmp_path):
    import yaml

    from blinded import contract, vault

    # S costs 2.5 bps a day to borrow for the first 18 days, then 5
    data = carry_file(tmp_path / "carry.parquet", borrow=[(0, 2.5e-4), (18, 5e-4)])
    path = tmp_path / "carry.brief.yaml"
    path.write_text(yaml.safe_dump(carry_brief_raw(str(data)), sort_keys=False))
    run = vault.init(path, tmp_path / "run")
    rules = (run.workspace / "RULES.md").read_text()
    assert "## Borrow" in rules and "`SYM.borrow_rate` is what borrowing SYM costs a day" in rules
    menu = (run.workspace / "ARCHETYPES.md").read_text()
    # train is the first 18 days less the 1-day warm-up, all at 2.5 bps
    assert (
        "- Borrow: the reverse trade, short spot and long the perpetual, pays to borrow the "
        "coin, `SYM.borrow_rate` as-of each step, for as long as it is held. On train, the "
        "mean rate a day in force: S 2.5 bps (2.5 at most)."
    ) in menu
    panel = run.panel("train")
    assert "S.borrow_rate" in panel.frame.columns
    # bl.carry goes short spot on the file's -3 bps days and pays there, and only there
    legs = contract.values(bl.carry("S", "S_PERP"), panel)
    win = Window("train", pd.Timestamp(panel.grid[96]), pd.Timestamp(panel.grid[-1]))
    s = simulate_legs(legs, panel, win, run.brief()).legs["S"]
    assert (s.position < 0).any() and (s.position > 0).any()
    np.testing.assert_array_equal(s.borrow < 0, s.position < 0)
    # 15 minutes short at 2.5 bps a day is 2.5 / 96 bps, valued at the mark as-of the fill
    # that starts the step (one bar after the decision) over the mark at the first fill
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [win.start.value, win.end.value])
    marks = panel.stream_of("S", "mid")
    m = marks.asof("mid", g[i0:i1] + run.brief().latency.value + marks.delay_ns)
    want = np.where(s.position < 0, -2.5 / 96 * m / m[np.isfinite(m)][0], 0.0)
    np.testing.assert_allclose(s.borrow, want, rtol=1e-12, atol=1e-15)


# -- the whole builder, on a faked network ---------------------------------------------------


def fake_archive(perp_basis=1.001):
    """``fetch_url`` for the carry builder without the network: every kline month of a
    symbol at a close of 100 on spot and 100 x ``perp_basis`` on the perpetual, and a
    settlement of 1e-4 every 8 hours, 3 ms late. Like the real one, it keeps a file it
    already has under that name."""
    import hashlib
    import io
    import zipfile

    from blinded import demo

    def zipped(path, name, body):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name.replace(".zip", ".csv"), body)
        path.write_bytes(buf.getvalue())
        digest = hashlib.sha256(buf.getvalue()).hexdigest()
        path.with_name(path.name + ".CHECKSUM").write_text(f"{digest}  {path.name}\n")

    def fetch_url(url, raw):
        raw.mkdir(parents=True, exist_ok=True)
        name = url.rsplit("/", 1)[1]
        path = raw / name
        if path.is_file():
            return path
        month = name.removesuffix(".zip")[-7:]
        first = pd.Timestamp(f"{month}-01", tz="UTC")
        last = first + pd.offsets.MonthBegin(1)
        if "fundingRate" in url:
            t = np.arange(first.value // 1_000_000, last.value // 1_000_000, 8 * HOUR_MS) + 3
            body = demo.FUNDING_HEADER + "\n" + "".join(f"{x},8,0.0001\n" for x in t)
        else:
            t = np.arange(first.value // 1_000_000, last.value // 1_000_000, 900_000)
            close = 100.0 * (perp_basis if "/futures/" in url else 1.0)
            body = demo.KLINE_HEADER + "\n" + "".join(
                f"{x},{close},{close},{close},{close},10,{x + 899_999},{10 * close},5,6,"
                f"{6 * close},0\n" for x in t)  # fmt: skip
        zipped(path, name, body)
        return path

    return fetch_url


@cross_check
def test_the_carry_builder_keeps_spot_and_perpetual_apart(tmp_path, monkeypatch):
    import gzip
    import json

    from test_archetypes import DATASETS

    from blinded import demo

    monkeypatch.setattr(demo, "fetch_url", fake_archive())
    costs = tmp_path / "costs"
    costs.mkdir()
    (costs / "bookTicker-2024-03-27.json").write_text(json.dumps({"day": "2024-03-27",
        "symbols": {"BTCUSDT": {"half_spread_bps": 0.5, "rows": 4, "two_sided_rows": 4,
                                "sha256": "ab" * 32, "zip_bytes": 1}}}))  # fmt: skip

    def fetch_day(exchange, data_types, symbols, day, raw):
        raw.mkdir(parents=True, exist_ok=True)
        body = (
            "exchange,symbol,timestamp,local_timestamp,ask_amount,ask_price,bid_price,"
            "bid_amount\nbinance,BTCUSDT,1,2,1,100.02,99.98,1\n"
        )  # 2 bps
        (raw / demo.raw_name(exchange, data_types[0], day, symbols[0])).write_bytes(
            gzip.compress(body.encode()))  # fmt: skip

    asked = []

    def fetch_json(url):  # a rate at 00:00 every day, 0.4 bps, as the history has them
        asked.append(url)
        q = dict(x.split("=") for x in url.split("?")[1].split("&"))
        days = range(int(q["startTime"]), int(q["endTime"]) + 86_400_000, 86_400_000)
        return {"data": [{"timestamp": str(t), "dailyInterestRate": "0.00004"} for t in days]}

    built = demo.carry(tmp_path / "out", tmp_path / "raw", costs, tmp_path / "tardis",
                       coins=["BTC"], log=lambda s: None, head=lambda url: 1,
                       details=lambda ex: listing(), fetch_day=fetch_day,
                       fetch_json=fetch_json)  # fmt: skip
    # 451 days from the day before train, 30 at a time
    assert len(asked) == 16 and all("asset=BTC&vipLevel=0" in u for u in asked)
    rows = pl.read_parquet(built.data)
    # 450 days of 96 bars, three rows each, on spot and perpetual, and 1,350 settlements
    assert rows.height == 2 * 43_200 * 3 + 1_350 + 450 == built.rows
    mids = rows.filter(pl.col("kind") == "book").group_by("symbol").agg(pl.col("mid").mean())
    assert dict(mids.iter_rows()) == {"BTCUSDT": 100.0, "BTCUSDT_PERP": pytest.approx(100.1)}
    # the spot day's 2 bps on spot, the cached 0.5 on the perpetual
    assert built.spreads == {"BTCUSDT": 2.0, "BTCUSDT_PERP": 0.5}
    funding = rows.filter(pl.col("kind") == "funding")
    assert (funding["ts"].dt.epoch("ms") % (15 * 60_000) == 0).all()
    assert built.funding["BTCUSDT_PERP"]["offset_ms"] == [3, 3]
    brief = briefs.load(built.brief)
    wins = {k: "/".join(str(t) for t in v) for k, v in built.windows.items()}
    assert {k: tuple(pd.Timestamp(x) for x in v.split("/")) for k, v in wins.items()} == {
        k: tuple(pd.Timestamp(x) for x in v.split("/"))
        for k, v in DATASETS["e"]["windows"].items()}  # fmt: skip
    assert brief.data.delay("funding", "BTCUSDT_PERP") == pd.Timedelta("1s")
    # a borrow row per day on spot, from the rate in force at the start, the day before's out
    borrow = rows.filter(pl.col("kind") == "borrow")
    assert borrow.height == 450 and set(borrow["symbol"]) == {"BTCUSDT"}
    assert borrow["ts"][0] == datetime(2024, 4, 1) and (borrow["rate"] == 4e-5).all()
    assert built.borrow["BTCUSDT"] == {"rows": 450, "first": "2024-04-01T00:00:00+00:00",
                                       "max_gap_hours": 24.0}  # fmt: skip
    assert brief.data.delay("borrow", "BTCUSDT") == pd.Timedelta("1s")
    assert (costs / "borrow-2024-04-01-2025-06-25.json").is_file()
    assert "| BTCUSDT | 450 | 2024-04-01T00:00 | 24 |" in built.notes.read_text()
    notes = built.notes.read_text()
    assert built.sha256[("BTCUSDT", "2024-04")] != built.sha256[("BTCUSDT_PERP", "2024-04")]
    assert "| BTCUSDT_PERP | 1,350 | 3 to 3 | 00, 08, 16 | 8 |" in notes
    # the zips are gone, the Tardis file kept
    assert not (tmp_path / "raw").exists()
    assert (tmp_path / "tardis" / demo.raw_name("binance", "book_ticker", demo.CARRY_SPOT_DAYS[0],
                                               "BTCUSDT")).is_file()  # fmt: skip


@edge
def test_one_file_for_spot_and_perpetual_stops_the_carry_build(tmp_path, monkeypatch):
    from blinded import demo

    fake = fake_archive()

    def same_folder(url, raw):  # the old bug: both markets fetched into one folder
        return fake(url, raw.parent)

    monkeypatch.setattr(demo, "fetch_url", same_folder)
    costs = tmp_path / "costs"
    costs.mkdir()
    import json

    (costs / "bookTicker-2024-03-27.json").write_text(json.dumps({"day": "2024-03-27",
        "symbols": {"BTCUSDT": {"half_spread_bps": 0.5, "rows": 4, "two_sided_rows": 4,
                                "sha256": "ab" * 32, "zip_bytes": 1}}}))  # fmt: skip
    (costs / f"tardis-binance-book_ticker-{demo.CARRY_SPOT_DAYS[0]}.json").write_text(
        json.dumps({"day": str(demo.CARRY_SPOT_DAYS[0]), "symbols": {"BTCUSDT": {
            "half_spread_bps": 2.0, "rows": 1, "two_sided_rows": 1, "sha256": "cd" * 32,
            "gz_bytes": 1}}}))  # fmt: skip
    with pytest.raises(demo.BuildError, match="the spot and perpetual klines of BTCUSDT "
                                              "2024-04 are one file"):  # fmt: skip
        demo.carry(tmp_path / "out", tmp_path / "raw", costs, coins=["BTC"], log=lambda s: None,
                   head=lambda url: 1, details=lambda ex: listing(),
                   fetch_day=lambda *a: None)  # fmt: skip
    assert not (tmp_path / "out").exists()
