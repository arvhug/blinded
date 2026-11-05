"""The two-leg sim on a five-point example, and ``bl.pairs`` on a pair that reverts.

Grid 1, 2, 3, 4, 5 s, latency 0.5 s, fee 7.5 bps. Leg A is the book of
``test_sim.py``; leg B (time: bid / ask, mid):

    0.0: 49.98 / 50.02, 50.00
    1.2: 49.99 / 50.01, 50.00
    2.2: 50.09 / 50.11, 50.10
    3.2: 50.04 / 50.06, 50.05

A asks for +1, +1, -1, -1, 0 and B for the opposite. B's reference notional
is its mark at 1.5 s, 50.00, so 0.01 of B's price is 2 bps.
"""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

import blinded as bl
from blinded import Panel, truncation
from blinded import brief as briefs
from blinded.brief import Window
from blinded.contract import Legs
from blinded.features import cross
from blinded.sim import maker_proxy, report, simulate_legs

pytestmark = pytest.mark.usefixtures("rules2")  # hand numbers under rules 2
known = pytest.mark.known_answer
cross_check = pytest.mark.cross_check
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


@pytest.fixture(scope="module", params=["no funding stream", "funding at zero"])
def panel(request):
    """The two legs, and the same with B settling funding at 0 at 2.0 and 4.0 s: every figure
    below holds for both (``test_carry.py`` has the funding)."""
    funding = None
    if request.param == "funding at zero":
        funding = {"B": pl.DataFrame({"ts": at(2.0, 4.0), "rate": [0.0, 0.0]})}
    book = {
        "A": quotes([99.98, 99.99, 100.19, 100.09], [100.02, 100.01, 100.21, 100.11]),
        "B": quotes([49.98, 49.99, 50.09, 50.04], [50.02, 50.01, 50.11, 50.06]),
    }
    trades = {
        "A": prints([0.5, 2.6, 3.7, 4.4], [99.98, 100.19, 100.11, 100.09], [-1.0, -1, 1, -1]),
        "B": prints([0.5, 2.6, 3.7], [49.98, 50.11, 50.04], [-1.0, 1, -1]),
    }
    return Panel.from_frames(at(1, 2, 3, 4, 5), book=book, trades=trades, funding=funding)


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
def test_leg_b_by_hand(panel):
    b = simulate_legs(pair(), panel, WINDOW, make_brief()).legs["B"]
    # marks at 1.5 .. 6.5 s: 50.00, 50.10, 50.05, 50.05, 50.05, 50.05. Short from 1.5 to
    # 2.5 s: -0.10, -20 bps; to 3.5 s: +0.05, +10; long from 3.5 s on a flat price: 0
    assert b.mid_pnl == pytest.approx([-20, 10, 0, 0, 0])
    # sell 1 at 49.99 (mark 50.00), buy 2 at 50.06 (mark 50.05), sell 1 at 50.04 (50.05):
    # 0.01 a unit each, 2 bps a unit
    assert b.spread_bps == pytest.approx([2, 0, 4, 0, 2])
    # the short sold 49.99 and bought 50.06, -0.07 = -14 bps; the long bought 50.06 and
    # sold 50.04, -0.02 = -4 bps: -18 gross
    assert b.gross.sum() == pytest.approx(-18)
    # cash: +49.99 - 2 x 50.06 + 50.04 = -0.09, flat at the end: -0.09 / 50.00 = -18 bps
    cash = 49.99 - 2 * 50.06 + 50.04
    assert b.gross.sum() == pytest.approx(cash / 50.00 * 1e4)
    assert b.fills.tolist() == [1, 0, 2, 0, 1]


@known
def test_the_pair_adds_its_legs(panel):
    sim = simulate_legs(pair(), panel, WINDOW, make_brief())
    a, b = sim.legs["A"], sim.legs["B"]
    np.testing.assert_allclose(sim.net, a.net + b.net)
    # A makes +6 gross on four fills (test_sim.py), B -18 on four: -12 gross, eight fills,
    # 8 x 7.5 = 60 of fees, -72 net over 1/720 of an hour
    assert sim.gross.sum() == pytest.approx(-12)
    m = sim.metrics()
    assert (m["fills"], m["fills_A"], m["fills_B"]) == (8, 4, 4)
    assert m["fees_bps"] == pytest.approx(60)
    assert m["net_bps_per_hour"] == pytest.approx(-72 * 720)
    # one notional in each leg while in position, none at the end: mean 8 / 5
    exposure = np.abs(a.position) + np.abs(b.position)
    assert exposure.tolist() == [2, 2, 2, 2, 0]
    assert m["gross_exposure"] == pytest.approx(1.6)
    assert m["turnover_per_hour"] == pytest.approx(8 * 720)


@known
def test_round_trips_are_runs_of_the_first_leg_charged_with_both(panel):
    sim = simulate_legs(pair(), panel, WINDOW, make_brief())
    # A's long run nets -7 and its short run -17 (test_sim.py). B in the first run: mid -20
    # + 10, its open 2 + 7.5 and half its flip 2 + 7.5: -29, so -36 in all. In the second:
    # mid 0, the other half of the flip 9.5 and the close 2 + 7.5: -19, so -36.
    assert sim.run_net == pytest.approx([-36, -36])
    assert sim.run_net.sum() == pytest.approx(sim.net.sum())
    free = simulate_legs(pair(), panel, WINDOW, make_brief(), fees={"A": 0.0, "B": 0.0})
    # without fees: A +8 and -2; B -10 - 2 - 2 and 0 - 2 - 2
    assert free.run_net == pytest.approx([-6, -6])


@known
def test_report_rows_for_two_legs(panel):
    rows = report(pair(), panel, WINDOW, make_brief(fees={"B": 10.0}))
    by = {r["row"]: r for r in rows.iter_rows(named=True)}
    assert list(by) == ["brief", "latency 0", "fees 0", "latency 0, fees 0", "maker proxy"]
    # each leg at its own fee: 4 x 7.5 + 4 x 10 = 70 on -12 gross
    assert by["brief"]["fees_bps"] == pytest.approx(70)
    assert by["brief"]["net_bps_per_hour"] / 720 == pytest.approx(-82)
    assert by["fees 0"]["net_bps_per_hour"] / 720 == pytest.approx(-12)
    # latency 0: A fills at 100.02, 100.19, 100.11 for +25 (test_sim.py); B sells 49.98,
    # buys 2 at 50.11, sells 50.04: -0.13 and -0.07 of 50.00, -40; -15 for the pair
    assert by["latency 0, fees 0"]["gross_bps_per_hour"] / 720 == pytest.approx(25 - 40)
    # the maker proxy of each leg at its own trade rate, added up. A: as in test_sim.py.
    # B: 1 buy and 1 sell print in the window, 0.2 a second each. A resting sell at 50.01
    # (1.5 s) marked at 50.10 is -18 bps; the buy of 2 at 50.04 marked 50.05, +2 each; the
    # sell at 50.06 marked 50.05, +2.
    pa_buy, pa_sell, pb = 1 - math.exp(-0.4), 1 - math.exp(-0.2), 1 - math.exp(-0.2)
    a_bps = 21 * pa_buy + 2 * pa_sell + 1 * pa_buy
    b_bps = (-18 + 2 * 2 + 2) * pb
    proxy = by["maker proxy"]
    assert proxy["net_bps_per_hour"] == pytest.approx((a_bps + b_bps) * 720)
    assert proxy["fills"] == pytest.approx(2 * pa_buy + pa_sell + 3 * pb)
    # per unit over all 8 units: (21 + 2 + 1) for A and (-18 + 4 + 2) for B
    assert proxy["net_bps_per_fill"] == pytest.approx((24 - 12) / 8)
    one = simulate_legs(pair(), panel, WINDOW, make_brief()).legs["A"]
    alone = maker_proxy(one, panel, WINDOW, "A", horizon=pd.Timedelta("1s"))
    assert alone["net_bps_per_hour"] == pytest.approx(a_bps * 720)


def reverting_pair(hours=6, seed=4):
    """A 1 s grid: A's log mid a random walk, B's the same plus AR(1) noise of half-life 60 s."""
    rng = np.random.default_rng(seed)
    n = hours * 3600
    a = np.log(100.0) + np.cumsum(rng.normal(0, 1e-4, n))
    phi = 2 ** (-1 / 60)
    shocks = rng.normal(0, 2e-5, n)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + shocks[t]
    grid = pd.date_range(pd.Timestamp(T0), periods=n, freq="1s")

    def book(v):
        m = np.exp(v)
        idx = grid - pd.Timedelta("500ms")
        return pd.DataFrame({"bid": m, "ask": m, "bid_qty": 1.0, "ask_qty": 1.0}, index=idx)

    return Panel.from_pandas(grid, book={"A": book(a), "B": book(a + noise)})


@cross_check
def test_pairs_trades_against_the_z_score_and_reads_no_future():
    p = reverting_pair()
    args = dict(z_in=2.0, z_out=0.5, window="2h", every="2h")
    legs = {leg: p.eval(expr).to_numpy() for leg, expr in bl.pairs("A", "B", **args).items()}
    z = p.eval(cross.spread_z("A", "B", "2h", "2h")).to_numpy()
    beta = p.eval(cross.hedge_ratio("A", "B", "2h", "2h")).to_numpy()
    # above z_in A is short, below -z_in long; B always holds -beta of A
    high, low = np.nan_to_num(z) > 2, np.nan_to_num(z) < -2
    assert high.sum() > 50 and low.sum() > 50
    assert (legs["A"][high] == -1).all() and (legs["A"][low] == 1).all()
    np.testing.assert_allclose(legs["B"][4 * 3600 :], -beta[4 * 3600 :] * legs["A"][4 * 3600 :])
    whole = Window("all", p.time(0), p.time(-1) + pd.Timedelta("1s"))
    verdict = truncation.check(lambda q: bl.pairs("A", "B", **args), p, whole, seed=2)
    assert verdict.ok, str(verdict)
    assert list(verdict.full) == ["A", "B"]


@cross_check
def test_the_spread_reverts_at_the_planted_half_life():
    p = reverting_pair()
    # half-life over the 7,199 steps inside each 2 h window, read at its last second; with
    # 7,200 s against a mean reversion time of 60 / ln 2 = 87 s the estimate is good to
    # about sqrt(2 x 87 / 7200) = 16%
    s = cross.pair_spread("A", "B", "2h", "2h")
    hl = p.eval(cross.half_life(s, 7199)).to_numpy()
    ends = [2 * 7200 - 1, 3 * 7200 - 1]
    assert all(30 < hl[i] < 120 for i in ends), hl[ends]
