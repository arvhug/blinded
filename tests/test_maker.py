"""The fill engine of a market-making strategy, on a hand example.

Grid 1 to 5 s, window [1 s, 6 s), every latency 0.5 s, tick 0.01, size 10
units, ``bl.quote(bid=0, ask=0, size=10, skew=0, max_inventory=10)``: join the
touch on both sides while a full fill keeps the known inventory within ten.
Maker fee 2 bps a unit, taker 5. The touch stream (stamp: bid / size, ask /
size) and the trades (stamp: side, quantity at price):

    0.0: 99.99 / 50, 100.01 / 40
    1.2: 99.99 / 30, 100.01 / 40     20 cancelled at the bid before the quote arrives
    1.6: 99.99 / 50, 100.01 / 40     20 joined behind the quote
    1.8: 99.99 / 40, 100.01 / 40     10 cancelled, nobody traded
    2.1:                                 sell 25 at 99.99
    2.2: 99.99 / 15, 100.01 / 40     after the 2.1 s trade
    2.6:                                 sell 20 at 99.99
    2.7: 99.98 / 60, 100.00 / 25     the 99.99 level is gone; the ask improved
    2.9:                                 sell 5 at 99.98
    3.3: 99.99 / 20, 100.01 / 40     the ask backed off
    3.8:                                 buy 5 at 100.01
    4.4: 100.01 / 10, 100.02 / 40    the market jumped
    4.6: 99.97 / 80, 99.99 / 30      and fell back
    4.7:                                 sell 10 at 99.97
    5.7:                                 sell 10 at 99.96
    5.8: 99.95 / 50, 99.97 / 30      after the 5.7 s trade

P_ref is the mid as-of 1.5 s, 100.00, so 0.01 of price on one unit is 1 bps
and on the ten units of the hand table 10 bps: every P&L number below is ten
times the hand table's, which is in bps of the ten units' notional.
"""

import math
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest
from conftest import local

import blinded as bl
from blinded import brief as briefs
from blinded import sim, sim_maker, truncation
from blinded.brief import Window
from blinded.contract import ContractError, Quotes, output_rule, values
from blinded.panel import Panel
from blinded.sim_maker import Decisions, Touch, decompose, run_engine, simulate_maker

pytestmark = pytest.mark.usefixtures("rules2")  # hand numbers under rules 2
known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case
T0 = datetime(2026, 10, 1, tzinfo=UTC)
S = 10**9
HALF = pd.Timedelta("500ms")
WINDOW = Window("w", pd.Timestamp(T0 + timedelta(seconds=1)),
                pd.Timestamp(T0 + timedelta(seconds=6)))  # fmt: skip
BOOK = [(0.0, 99.99, 50, 100.01, 40), (1.2, 99.99, 30, 100.01, 40), (1.6, 99.99, 50, 100.01, 40),
        (1.8, 99.99, 40, 100.01, 40), (2.2, 99.99, 15, 100.01, 40), (2.7, 99.98, 60, 100.00, 25),
        (3.3, 99.99, 20, 100.01, 40), (4.4, 100.01, 10, 100.02, 40), (4.6, 99.97, 80, 99.99, 30),
        (5.8, 99.95, 50, 99.97, 30)]  # fmt: skip
TRADES = [(2.1, -1, 25, 99.99), (2.6, -1, 20, 99.99), (2.9, -1, 5, 99.98), (3.8, 1, 5, 100.01),
          (4.7, -1, 10, 99.97), (5.7, -1, 10, 99.96)]  # fmt: skip
POLICY = dict(bid=0, ask=0, size=10, skew=0, max_inventory=10)


def at(*seconds):
    return [T0 + timedelta(seconds=s) for s in seconds]


def secs(ns) -> float:
    return round((int(ns) - int(pd.Timestamp(T0).value)) / S, 6)


def hand_panel(extra_trades=(), funding=None, borrow=None) -> Panel:
    book = pl.DataFrame({"ts": at(*(r[0] for r in BOOK)), "bid": [r[1] for r in BOOK],
                         "bid_qty": [float(r[2]) for r in BOOK], "ask": [r[3] for r in BOOK],
                         "ask_qty": [float(r[4]) for r in BOOK]})  # fmt: skip
    rows = sorted([*TRADES, *extra_trades])
    trades = pl.DataFrame({"ts": at(*(r[0] for r in rows)),
                           "side": [float(r[1]) for r in rows],
                           "qty": [float(r[2]) for r in rows],
                           "price": [r[3] for r in rows]})  # fmt: skip
    streams = {}
    if funding:
        streams["funding"] = {"X": pl.DataFrame({"ts": at(*(f[0] for f in funding)),
                                                 "rate": [f[1] for f in funding]})}  # fmt: skip
    if borrow:
        streams["borrow"] = {
            "X": pl.DataFrame(
                {"ts": at(*(b[0] for b in borrow)), "borrow_rate": [b[1] for b in borrow]}
            )
        }
    return Panel.from_frames(at(1, 2, 3, 4, 5), book={"X": book}, trades={"X": trades},
                             **streams)  # fmt: skip


def run(model, panel=None, quote=None, **kwargs):
    kwargs = dict(tick=0.01, latency=HALF, maker_fee_bps=2.0, taker_fee_bps=5.0,
                  block=pd.Timedelta("1s")) | kwargs  # fmt: skip
    q = bl.quote(**POLICY) if quote is None else quote
    return simulate_maker(q, panel or hand_panel(), WINDOW, "X", model, **kwargs)


def fills(s):
    b = s.book
    return [
        (secs(t), int(side), int(p), float(q), bool(th))
        for t, side, p, q, th in zip(
            b.fill_ns, b.fill_side, b.fill_price, b.fill_qty, b.fill_through, strict=True
        )
    ]


def orders(s):
    """Each order as (sent, side, price in ticks, arrival, status)."""
    b = s.book
    return [(secs(t), int(side), int(p), secs(a), st) for t, side, p, a, st in
            zip(b.order_sent, b.order_side, b.order_price, b.order_arrival, b.order_status,
                strict=True)]  # fmt: skip


@pytest.fixture(scope="module")
def hand():
    return {m: run(m) for m in sim_maker.MODELS}


@known
def test_the_hand_example_fills_under_each_model(hand):
    # ticks: 99.99 is 9999. T's bid at 99.99 is alone once the 2.7 s row drops the bid to
    # 99.98, and the sell at 99.98 at 2.9 s goes through it. RA reaches the front of 30 when
    # the 2.6 s sell takes it to -15; PR at 2.1 s (24 - 25 = -1: one unit), then 9 at 2.6 s;
    # F on the first sell at its price. Every model's new ask at 100.00, alone inside the
    # 3.3 s book, is swept by the buy at 100.01 at 3.8 s, and the bid at 99.97 by the sell at
    # 99.96 at 5.7 s.
    swept = [(3.8, -1, 10000, 10.0, True), (5.7, 1, 9997, 10.0, True)]
    assert fills(hand["T"]) == [(2.9, 1, 9999, 10.0, True), *swept]
    assert fills(hand["RA"]) == [(2.6, 1, 9999, 10.0, False), *swept]
    assert fills(hand["PR"]) == [(2.1, 1, 9999, 1.0, False), (2.6, 1, 9999, 9.0, False), *swept]
    assert fills(hand["F"]) == [(2.1, 1, 9999, 10.0, False), *swept]
    # the window ends long ten: sold at the bid as-of 6.5 s, 99.95, as a taker
    for s in hand.values():
        assert (secs(s.book.flatten[0]), *s.book.flatten[1:]) == (6.5, -1, 9995, 10.0)
        assert s.metrics()["fills"] == len(s.book.fill_ns)


@known
def test_the_queue_ahead_of_the_first_bid():
    # RA: 30 at arrival (1.5 s, the 1.2 s row); min(30, 50) at 1.6 s; min(30, 40) at 1.8 s;
    # 30 - 25 at 2.1 s; min(5, 15) at 2.2 s; 5 - 20 < 0 at 2.6 s, filled, at the front.
    # PR: at 1.8 s the level fell from 50 to 40 with no trade, so a cancel of 10 came from
    # ahead in proportion, 10 x 30 / 50 = 6: 24; 24 - 25 = -1 at 2.1 s fills one unit and
    # leaves it at the front; the 2.2 s row shows 40 - 25 = 15, no cancel, so it stays 0.
    # F is at the front from arrival; T never knows the queue.
    panel = hand_panel()
    touch = sim_maker.touch_of(panel, "X", 0.01)
    dec = Decisions.make(_policy_arrays(), panel.grid_ns, touch, None)
    paths = {}
    for m in sim_maker.MODELS:
        trace = []
        run_engine(touch, dec, panel.grid_ns[-1] + S, m, HALF.value, trace=trace)
        paths[m] = [(secs(t), a) for t, _, oid, a in trace if oid == 0]
    assert paths["RA"] == [(1.5, 30.0), (1.6, 30.0), (1.8, 30.0), (2.1, 5.0), (2.2, 5.0),
                           (2.6, 0.0)]  # fmt: skip
    assert paths["PR"] == [(1.5, 30.0), (1.6, 30.0), (1.8, 24.0), (2.1, 0.0), (2.2, 0.0),
                           (2.6, 0.0)]  # fmt: skip
    assert paths["F"] == [(1.5, 0.0), (2.1, 0.0)]
    assert len(paths["T"]) == 1 and math.isnan(paths["T"][0][1])


def _policy_arrays(n=5, **change):
    p = POLICY | change
    return Quotes({k: np.full(n, np.nan if v is None else float(v)) for k, v in p.items()})


@known
def test_the_decisions_rejection_and_deep_bid(hand):
    # at 3 s RA and T know no fill yet (theirs are at 2.6 and 2.9 s, known from 3.1 and 3.4 s)
    # and post a bid at the 2.7 s touch, 99.98, behind the 3.3 s book's 99.99 when it arrives
    # at 3.5 s: deep, never the touch, cancelled at 4.5 s. F knows its 2.1 s fill (long 10)
    # and PR its one unit: a full bid would take either past ten, so both post an ask only.
    # At 4 s every model knows it is long ten and asks 100.01, the 3.3 s touch; it arrives at
    # 4.5 s against the 4.4 s best bid of 100.01 and is rejected.
    for m, s in hand.items():
        sent = [o for o in orders(s) if o[0] == 3.0]
        bids = [o for o in sent if o[1] == 1]
        assert bids == ([(3.0, 1, 9998, 3.5, "cancelled")] if m in ("RA", "T") else []), m
        assert [o for o in sent if o[1] == -1] == [(3.0, -1, 10000, 3.5, "filled")]
        assert [o for o in orders(s) if o[0] == 4.0] == [(4.0, -1, 10001, 4.5, "rejected")]
        assert s.metrics()["post_only_rejected"] == 1
    deep = hand["RA"].book
    assert np.isnan(deep.order_ahead[2]) and deep.order_status[2] == "cancelled"


def _hftbacktest_hand(model: str):
    """The hand example through hftbacktest (the ``check`` extra), under ``RA`` (its
    ``risk_adverse_queue_model``) or ``PR`` (``power_prob_queue_model(1.0)``): the fills as
    (stamp in s, side, price in ticks, units) and the expired GTX orders as (stamp, side,
    price), each in the order hftbacktest reported them.

    The rules that make it the hand example, each a translation and none a fudge:

    - a touch row is depth events on the exchange clock, ``local_ts = exch_ts + 1`` so the
      feed latency is positive: the level the touch moved away from towards the far side
      (a bid that fell, an ask that rose) is deleted, and a level left behind when the touch
      moves in keeps its last size, which the touch stream does not know. No resting order
      is ever on such a level here (RA's deep bid at 99.98 rests on the 2.7 s row's 60 and
      never fills in either engine);
    - lot 1, so ``is_filled``'s rounding of a fill to whole lots touches none of these;
    - hftbacktest's local position counts a partially filled order only once it is filled,
      and then only the last response's quantity (``proc/local.rs`` applies a fill to the
      state on status Filled alone): PR's 2.1 s unit would never reach it. So the policy
      keeps its inventory from the fills it has been told of, as the engine's known
      inventory is, and reads them after every feed row and order response
      (``wait_next_feed(True, ...)`` up to each decision; ``wait_order_response`` with
      ``WAIT_ORDER_RESPONSE_ANY`` waits for an order of that id in this binding, so it only
      times out), so that a partial fill and the rest of its order are two fills;
    - hftbacktest fills a resting order whose price the book moves through without a trade
      (``on_best_bid_update``); the hand book never does that to a resting order.
    """
    hb = pytest.importorskip("hftbacktest")
    from hftbacktest.types import event_dtype
    from numba import njit

    depth = hb.DEPTH_EVENT | hb.EXCH_EVENT | hb.LOCAL_EVENT
    trade = hb.TRADE_EVENT | hb.EXCH_EVENT | hb.LOCAL_EVENT
    rows, old = [], None
    for t, b, bq, a, aq in BOOK:
        ts, b, a = round(t * S), round(b * 100), round(a * 100)
        if old is not None:
            rows += [(depth | hb.BUY_EVENT, ts, p, 0.0) for p in range(old[0], b, -1)]
            rows += [(depth | hb.SELL_EVENT, ts, p, 0.0) for p in range(old[1], a)]
        rows += [(depth | hb.BUY_EVENT, ts, b, float(bq)),
                 (depth | hb.SELL_EVENT, ts, a, float(aq))]  # fmt: skip
        old = b, a
    rows += [(trade | (hb.BUY_EVENT if side > 0 else hb.SELL_EVENT), round(t * S),
              round(p * 100), float(q)) for t, side, q, p in TRADES]  # fmt: skip
    rows.sort(key=lambda r: r[1])  # stable: a row's deletions, then its two levels
    data = np.zeros(len(rows), dtype=event_dtype)
    for i, (ev, ts, p, q) in enumerate(rows):
        data[i] = (ev, ts, ts + 1, p * 0.01, q, 0, 0, 0.0)

    filled_, cancelled, expired = hb.FILLED, hb.CANCELED, hb.EXPIRED
    gtx, limit = hb.GTX, hb.LIMIT

    @njit
    def told(hbt, seen, out, n):
        # every fill and expiry the strategy has been told of since the last look
        vals = hbt.orders(0).values()
        while vals.has_next():
            o = vals.get()
            got = o.qty - o.leaves_qty
            side = 1.0 if o.side == 1 else -1.0
            if got - seen[o.order_id, 0] > 1e-9:
                out[n, 0], out[n, 1] = o.exch_timestamp, side
                out[n, 2], out[n, 3] = o.exec_price_tick, got - seen[o.order_id, 0]
                seen[o.order_id, 0] = got
                n += 1
            if o.status == expired and seen[o.order_id, 1] == 0:
                out[n, 0], out[n, 1] = o.exch_timestamp, side
                out[n, 2], out[n, 3] = o.price_tick, 0.0
                seen[o.order_id, 1] = 1
                n += 1
        return n

    @njit
    def policy(hbt, out):
        # decisions at 1 to 5 s and the end at 6 s; join the touch while a full fill keeps the
        # known inventory within ten; keep a live order whose price stands
        seen = np.zeros((64, 2))
        n, ids, nid = 0, np.zeros(2, dtype=np.int64), 1
        hbt.elapse(1_000_000_000)  # from the first row, at 0, to the first decision
        for k in range(6):
            g = (k + 1) * 1_000_000_000
            while hbt.current_timestamp < g:
                hbt.wait_next_feed(True, g - hbt.current_timestamp)
                n = told(hbt, seen, out, n)
            known = 0.0
            for i in range(n):
                known += out[i, 1] * out[i, 3]
            book, orders = hbt.depth(0), hbt.orders(0)
            bid = book.best_bid_tick if k < 5 and known + 10.0 <= 10.0 + 1e-9 else -1
            ask = book.best_ask_tick if k < 5 and known - 10.0 >= -10.0 - 1e-9 else -1
            want = (bid, ask)
            for s in range(2):
                v = ids[s]
                if v != 0 and orders.__contains__(v):
                    o = orders.get(v)
                    if o.status != filled_ and o.status != cancelled and o.status != expired:
                        if want[s] == o.price_tick:
                            continue
                        hbt.cancel(0, v, False)
                ids[s] = 0
                if want[s] >= 0:
                    ids[s], nid = nid, nid + 1
                    if s == 0:
                        hbt.submit_buy_order(0, ids[s], want[s] * 0.01, 10.0, gtx, limit, False)
                    else:
                        hbt.submit_sell_order(0, ids[s], want[s] * 0.01, 10.0, gtx, limit, False)
        hbt.elapse(1_000_000_000)
        return told(hbt, seen, out, n)

    asset = (hb.BacktestAsset().data([data]).linear_asset(1.0)
             .constant_order_latency(HALF.value, HALF.value).partial_fill_exchange()
             .trading_value_fee_model(0.0002, 0.0005).tick_size(0.01).lot_size(1.0)
             .last_trades_capacity(0))  # fmt: skip
    queue = asset.risk_adverse_queue_model if model == "RA" else asset.power_prob_queue_model
    asset = queue() if model == "RA" else queue(1.0)
    hbt = hb.HashMapMarketDepthBacktest([asset])
    out = np.zeros((64, 4))
    n = policy(hbt, out)
    hbt.close()
    del data  # the backtest read it in place until here
    got = [(round(t / S, 6), int(side), int(p), float(q)) for t, side, p, q in out[:n]]
    return [f for f in got if f[3] > 0], [f[:3] for f in got if f[3] == 0]


@cross
@pytest.mark.parametrize("model", ["RA", "PR"])
def test_hftbacktest_fills_the_hand_example_as_the_engine_does(hand, model):
    # the same fills to the stamp, price and quantity, and the one GTX
    # order expired at 4.5 s, the ask at 100.01 against the 4.4 s best bid of 100.01. The
    # rules that translate one engine's input into the other's are in _hftbacktest_hand.
    theirs, expired = _hftbacktest_hand(model)
    mine = [f[:4] for f in fills(hand[model])]
    assert sorted(theirs) == mine
    assert mine == {"RA": [(2.6, 1, 9999, 10.0), (3.8, -1, 10000, 10.0), (5.7, 1, 9997, 10.0)],
                    "PR": [(2.1, 1, 9999, 1.0), (2.6, 1, 9999, 9.0), (3.8, -1, 10000, 10.0),
                           (5.7, 1, 9997, 10.0)]}[model]  # fmt: skip
    rejected = [(a, side, p) for _, side, p, a, st in orders(hand[model]) if st == "rejected"]
    assert expired == rejected == [(4.5, -1, 10001)]


@known
def test_the_hand_example_pnl(hand):
    # cash, price units: -999.9 + 1000.0 - 999.7 + 999.5 = -0.1 in every model, 10 bps of one
    # unit at 100.00 (the hand table's -1). Captured against the mid at each fill: T's 2.9 s buy is
    # at the mid of the 2.7 s row, 99.99 (0), its 3.8 s sell at the 3.3 s mid 100.00 (0) and
    # its 5.7 s buy one cent under 99.98 (+10): +10; the others buy at 2.1 or 2.6 s a cent
    # under 100.00: +20. Fees: 30 units at 2 bps and the flatten's 10 at 5: 110 (the hand table's
    # 11). Net -10 - 110 = -120 (the hand table's -12).
    for m, s in hand.items():
        k = s.metrics()
        assert float(s.gross.sum()) == pytest.approx(-10.0, abs=1e-9)
        assert k["captured_bps"] == pytest.approx(10.0 if m == "T" else 20.0, abs=1e-9)
        assert k["fees_bps"] == pytest.approx(110.0)
        assert float(s.net.sum()) == pytest.approx(-120.0, abs=1e-9)
        assert (k["units"], k["flatten_units"]) == (30.0, 10.0)
        # inventory and flatten, the rest of the gross: -2 x 10 for T, -3 x 10 for the others
        assert k["captured_bps"] + float(s.gross.sum() - k["captured_bps"]) == pytest.approx(-10)
    # RA by step, t = 1.5, 2.5, ..., 5.5 s, the last step to the flatten at 6.5 s, marks
    # 100.00, 100.00, 100.00, 100.015, 99.98, then 99.96: the 2.6 s buy is marked at 100.00 at
    # both ends of its step (0) and earns a cent (-10 of spread); held through step 2 it makes
    # 10 x 0.015 while the 3.8 s sale gives the same back from its stamp to 4.5 s; the 5.7 s
    # buy loses 10 x 0.02 to 6.5 s (-20) and earns a cent (-10), and the flatten pays a cent
    # (+10).
    ra = hand["RA"]
    assert ra.position.tolist() == [0, 0, 10, 0, 0]
    assert np.allclose(ra.mid_pnl, [0, 0, 0, 0, -20], atol=1e-9)
    assert np.allclose(ra.spread_bps, [0, -10, 0, 0, 0], atol=1e-9)
    assert np.allclose(ra.fee_paid, [0, 20, 20, 0, 70])
    assert np.allclose(ra.net, [0, -10, -20, 0, -90], atol=1e-9)
    # round trips: bought at 99.99, sold at 100.00 (+10 - 40 of fees); bought at 99.97 and
    # flattened at 99.95 (-20 - 20 - 50)
    assert np.allclose(ra.run_net, [-30, -90], atol=1e-9)
    assert sim.hold_steps(ra) == 1.0  # one run of one step
    # F holds from its 2.1 s fill through steps 1 and 2: one run of two
    assert hand["F"].position.tolist() == [0, 10, 10, 0, 0] and sim.hold_steps(hand["F"]) == 2.0


@known
def test_markouts_one_second_after_the_first_buy(hand):
    # F's 2.1 s buy: mid 100.00 then 99.99 at 3.1 s (the 2.7 s row): -10 (the hand table's -1).
    # RA's 2.6 s buy: 100.00 then 100.00 at 3.6 s (the 3.3 s row): 0. T's 2.9 s buy: 99.99
    # then 100.00 at 3.9 s: +10.
    first = {}
    for m in ("F", "RA", "T"):
        b, touch = hand[m].book, hand[m].touch
        t = b.fill_ns[0]
        move = touch.mid(np.array([t + S]))[0] - touch.mid(np.array([t]))[0]
        first[m] = round(b.fill_qty[0] * move / 100.0 * 1e4, 9)
    assert first == {"F": -10.0, "RA": 0.0, "T": 10.0}
    # the same sums over all the fills: the 3.8 s sale reads +20 at 4.8 s (100.00 to 99.98)
    # and the 5.7 s buy -20 at 6.7 s (99.98 to 99.96), so the 1 s totals are the first buy's
    for m in ("F", "RA", "T"):
        assert decompose(hand[m])["adverse_bps"]["1s"] == pytest.approx(first[m], abs=1e-9)


@known
def test_the_response_latency_decides_what_a_decision_knows():
    # with fills known at once, RA at 3 s knows its 2.6 s fill (long 10), so it posts no bid.
    # At 4 s it also knows the 3.8 s sale, flat, and bids at the 3.3 s touch, 99.99: deep at
    # 4.5 s under the 4.4 s book; at 4.6 s the ask comes down to 99.99, its price, with no
    # trade, and the book crossing fills it there (before the book-crossing rule it waited for the
    # sell at 99.97 at 4.7 s to sweep it, at the same price). At 5 s it knows it is long ten
    # and only asks, and its ask at 99.99 waits behind 30 that nothing trades. Cash -999.9 +
    # 1000.0 - 999.9 + 999.5 = -0.3: -30.
    late, now = run("RA"), run("RA", response_latency=pd.Timedelta(0))
    assert [o for o in orders(late) if o[:2] == (3.0, 1)] == [(3.0, 1, 9998, 3.5, "cancelled")]
    assert [o for o in orders(now) if o[:2] == (3.0, 1)] == []
    assert fills(now) == [(2.6, 1, 9999, 10.0, False), (3.8, -1, 10000, 10.0, True),
                          (4.6, 1, 9999, 10.0, False)]  # fmt: skip
    assert now.book.fill_crossed.tolist() == [False, False, True]
    assert float(now.gross.sum()) == pytest.approx(-30.0, abs=1e-9)


@known
def test_a_stale_quote_fills_until_its_cancel_lands():
    # a buy of 50 at 100.02 at 3.4 s goes through the ask at 100.01, which the decision at 3 s
    # moved to 100.00. With the cancel 0.5 s away the old ask is still resting and sells its
    # ten; with the cancel 0.2 s away it was pulled at 3.2 s, the new one lands at 3.5 s, and
    # the buy fills neither.
    panel = hand_panel(extra_trades=[(3.4, 1, 50, 100.02)])
    slow = run("RA", panel)
    fast = run("RA", panel, cancel_latency=pd.Timedelta("200ms"))
    assert (3.4, -1, 10001, 10.0, True) in fills(slow)
    assert not any(f[0] == 3.4 for f in fills(fast))
    assert [o for o in orders(fast) if o[0] == 1.0 and o[1] == -1] == [
        (1.0, -1, 10001, 1.5, "cancelled")]  # fmt: skip
    assert fast.book.order_end[1] == pd.Timestamp(T0 + timedelta(seconds=3.2)).value


@known
def test_on_one_stamp_the_cancel_and_the_new_order_come_before_the_trade():
    # a buy of 50 at 100.02 at 3.5 s, the stamp on which the decision at 3 s pulls the ask at
    # 100.01 and lands the one at 100.00: the trade sees only the new order, alone inside the
    # 3.3 s book, and goes through it, so the ten sell at 100.00 there, not at 100.01
    s = run("RA", hand_panel(extra_trades=[(3.5, 1, 50, 100.02)]))
    assert [f for f in fills(s) if f[0] == 3.5] == [(3.5, -1, 10000, 10.0, True)]


@known
@pytest.mark.parametrize("model", sim_maker.MODELS)
def test_the_far_touch_reaching_a_resting_price_fills_it_with_no_trade(model):
    # a bid joins 99.99 at 1.5 s and an ask 100.01. At 2.2 s the ask comes down to 99.99 with
    # no trade: our bid is not in the recorded book, so the seller who brought the ask to
    # 99.99 would have hit it first; RA, PR and F buy ten there, flagged crossed. T needs the
    # book strictly through the price, which the 2.6 s row (ask 99.98) is. The decision at
    # 3 s moves the ask to the 2.6 s touch, 99.98, landing at 3.5 s, so the old ask at
    # 100.01 rests until then, and the 3.2 s row's bid at 100.01 sells it under RA, PR and F;
    # not under T, whose bid there is not through it. No trade prints all window.
    book = [(0.0, 9999, 50, 10001, 40), (2.2, 9998, 60, 9999, 30), (2.6, 9997, 60, 9998, 30),
            (3.2, 10001, 10, 10002, 40)]  # fmt: skip
    ns = np.array([int(r[0] * S) for r in book], dtype=np.int64)
    col = [np.array([r[i] for r in book]) for i in range(1, 5)]
    empty = np.zeros(0)
    touch = Touch(0.01, ns, col[0].astype(np.int64), col[2].astype(np.int64),
                  col[1].astype(float), col[3].astype(float), empty.astype(np.int64),
                  empty.astype(np.int64), empty, empty.astype(np.int64))  # fmt: skip
    grid = np.array([1, 2, 3], dtype=np.int64) * S
    q = Quotes({"bid": np.zeros(3), "ask": np.zeros(3), "size": np.full(3, 10.0),
                "skew": np.zeros(3), "max_inventory": np.full(3, 10.0)})  # fmt: skip
    b = run_engine(touch, Decisions.make(q, grid, touch, None), 4 * S, model, S // 2)
    got = [(round(int(t) / S, 3), int(s), int(p), float(x), bool(c)) for t, s, p, x, c
           in zip(b.fill_ns, b.fill_side, b.fill_price, b.fill_qty, b.fill_crossed,
                  strict=True)]  # fmt: skip
    if model == "T":
        assert got == [(2.6, 1, 9999, 10.0, True)]
    else:
        assert got == [(2.2, 1, 9999, 10.0, True), (3.2, -1, 10001, 10.0, True)]
    assert not b.fill_through.any() and not b.fill_taker.any()


@known
def test_post_only_rejects_a_bid_that_would_cross():
    # two ticks under the 99.99 / 100.01 touch, -2 improves the bid to 100.01: at the ask, so
    # rejected at 1.5 s; -1 puts it at 100.00, inside the spread, alone
    s = run("RA", quote=bl.quote(bid=-2, ask=None, size=10, max_inventory=10))
    assert orders(s)[0] == (1.0, 1, 10001, 1.5, "rejected")
    s = run("RA", quote=bl.quote(bid=-1, ask=None, size=10, max_inventory=10))
    assert orders(s)[0][:4] == (1.0, 1, 10000, 1.5) and s.book.order_ahead[0] == 0.0


@known
def test_offsets_round_away_from_the_touch():
    # a half tick behind the bid rounds to a whole tick behind it (99.98); half a tick inside
    # the ask rounds back to the touch (100.01): the less aggressive quote in both cases
    s = run("RA", quote=bl.quote(bid=0.5, ask=-0.5, size=10, max_inventory=10))
    assert [o[:3] for o in orders(s)[:2]] == [(1.0, 1, 9998), (1.0, -1, 10001)]


@known
def test_a_size_under_one_lot_quotes_nothing():
    # lot 1: 0.9 rounds down to 0, so no order at all; 10.7 rounds down to 10
    assert orders(run("RA", quote=bl.quote(**POLICY | {"size": 0.9}), lot=1.0)) == []
    s = run("RA", quote=bl.quote(**POLICY | {"size": 10.7}), lot=1.0)
    assert s.book.order_size[0] == 10.0


@known
def test_skew_lean_and_fair_move_the_quotes():
    # PR knows long one at 3 s. With skew 1 a tick per unit, its bid sits one tick below the
    # 2.7 s touch's 99.98 and its ask one tick below 100.00: 99.97 and 99.99 (the cap is out
    # of the way at 100)
    s = run("PR", quote=bl.quote(bid=0, ask=0, size=10, skew=1, max_inventory=100))
    assert [o[1:3] for o in orders(s) if o[0] == 3.0] == [(1, 9997), (-1, 9999)]
    # lean 1 raises both from the 0.0 s touch: the bid to 100.00, the ask to 100.02
    s = run("RA", quote=bl.quote(bid=0, ask=0, size=10, lean=1, max_inventory=10))
    assert [o[1:3] for o in orders(s)[:2]] == [(1, 10000), (-1, 10002)]
    # fair 99.995: the bid from floor(9999.5) = 99.99, the ask from ceil(9999.5) = 100.00
    s = run("RA", quote=bl.quote(bid=0, ask=0, size=10, max_inventory=10, fair=99.995))
    assert [o[1:3] for o in orders(s)[:2]] == [(1, 9999), (-1, 10000)]


def small_touch(sells: list[float], n=5) -> tuple[Touch, Decisions]:
    """A book fixed at 99.99 / 100.01 and a sell of each size at 99.99, 0.1 s after each of
    the first decisions at 1, 2, ... s."""
    ns = [int(pd.Timestamp(t).value) for t in at(0.0)]
    book = Touch(0.01, np.array(ns), np.array([9999]), np.array([10001]), np.array([50.0]),
                 np.array([50.0]), np.array([(k + 1.1) * S for k in range(len(sells))],
                                           dtype=np.int64) + ns[0],
                 np.full(len(sells), 9999), np.array(sells, dtype=float),
                 np.full(len(sells), -1))  # fmt: skip
    grid = np.array([ns[0] + (k + 1) * S for k in range(n)], dtype=np.int64)
    return book, grid


@known
def test_max_inventory_withdraws_the_bid_exactly_at_the_cap():
    # size 1, max_inventory 2, F, every latency 0: the bid at 1 s fills at 1.1 s (long 1),
    # the one at 2 s at 2.1 s (long 2); at 3 s a full fill would take it to 3, so no bid, and
    # the sells at 3.1 and 4.1 s fill nothing. The ask (short 1 is within the cap) rests.
    touch, grid = small_touch([1.0, 1.0, 1.0, 1.0])
    dec = Decisions.make(_policy_arrays(size=1, max_inventory=2), grid, touch, None)
    b = run_engine(touch, dec, grid[-1] + S, "F", 0)
    assert [secs(t) for t in b.fill_ns] == [1.1, 2.1]
    assert [secs(t) for t, side in zip(b.order_sent, b.order_side, strict=True) if side > 0] == [
        1.0, 2.0]  # fmt: skip
    assert b.flatten[1:] == (-1, 9999, 2.0)


@known
def test_the_flatten_pays_its_half_spread_and_the_taker_fee(hand):
    # RA's last step: the 5.7 s buy earns a cent on ten (-10) and the flatten at 99.95 pays a
    # cent against the 6.5 s mid 99.96 (+10); fees 10 units at 2 and 10 at 5
    ra = hand["RA"]
    assert ra.spread_bps[-1] == pytest.approx(0.0, abs=1e-9)
    assert ra.fee_paid[-1] == pytest.approx(10 * 2 + 10 * 5)
    assert ra.metrics()["flatten_units"] == 10.0


@known
def test_funding_reads_the_inventory_at_the_settlement_from_the_fills():
    # a settlement of +1 bps at 2.7 s: RA is long ten from its 2.6 s fill, though the grid
    # sampled it flat at 2.5 s, and pays 10 x 0.0001 x 1e4 x 99.99 / 100 = 9.999 at step 1;
    # T's bid fills only at 2.9 s, so it pays nothing. On a fill's own stamp, 2.6 s, the
    # position after it pays: 10 x 100.00 / 100 = 10.
    ra = run("RA", hand_panel(funding=[(2.7, 0.0001)]))
    assert np.allclose(ra.funding, [0, -9.999, 0, 0, 0])
    assert run("T", hand_panel(funding=[(2.7, 0.0001)])).funding.sum() == 0.0
    on = run("RA", hand_panel(funding=[(2.6, 0.0001)]))
    assert np.allclose(on.funding, [0, -10.0, 0, 0, 0])
    assert ra.metrics()["funding_bps"] == pytest.approx(-9.999)
    assert float(ra.net.sum()) == pytest.approx(-120.0 - 9.999)
    assert float(ra.run_net.sum()) == pytest.approx(-120.0 - 9.999)


@known
def test_a_short_inventory_pays_borrow_as_the_taker_sim_charges_it():
    # an ask only: it rests at 100.01, moves to 100.00 at 3.5 s and is swept at 3.8 s, short
    # ten; RA does not know at 4 s and asks 100.01, rejected; at 5 s it knows it is short ten,
    # so no ask. Short at t = 4.5 and 5.5 s. At 8.64 a day, a second short on ten units at
    # mark m costs 10 x 8.64 x 1e4 / 86400 x m / 100: 10.0015 at 100.015 and 9.998 at 99.98,
    # the accrual of blinded.sim.borrowed.
    panel = hand_panel(borrow=[(0.0, 8.64)])
    s = run("RA", panel, quote=bl.quote(bid=None, ask=0, size=10, max_inventory=10))
    assert s.position.tolist() == [0, 0, 0, -10, -10]
    assert np.allclose(s.borrow, [0, 0, 0, -10.0015, -9.998])
    assert s.metrics()["borrow_bps"] == pytest.approx(-19.9995)


def hand_brief(**execution):
    raw = {"name": "hand", "data": {"path": "x.parquet", "instrument_column": "symbol",
                                    "streams": {"book": {"fields": {"bid": "bid"}}}},
           "target": {"instrument": "X", "horizons": ["1s"]}, "grid": {"step": "1s"},
           "execution": {"latency_ms": 500, "taker_fee_bps": 5.0, "maker_fee_bps": 2.0,
                         "tick_size": 0.01} | execution,
           "stats": {"block": "1s"}}  # fmt: skip
    return briefs.parse(raw)


@known
def test_the_rows_and_the_worse_of_the_two_queue_models():
    # by 1 s step, RA nets 0, -10, -20, 0, -90 and PR -1, -9, -20, 0, -90 (its 2.1 s unit
    # earns its cent a step earlier): the same mean, -24, and PR's spread is the smaller
    # (squares about the mean 5,702 against 5,720), so on the brief's 1 s blocks PR's Sharpe,
    # -24 / sqrt(5702 / 4) x 60 = -38.14, is below RA's -38.08; but PR holds two steps, and
    # on its own 2 s blocks, the fifth step folded into the second (-10 and -110), it is
    # -36.0, so RA is scored, as the judge scores it
    panel = hand_panel()
    rows = sim.report(bl.quote(**POLICY), panel, WINDOW, hand_brief(), "X")
    by = {r["row"]: r for r in rows.iter_rows(named=True)}
    assert list(by) == ["brief", "RA", "PR", "T", "F", "latency 0", "fees 0", "rebate 0.5",
                        "fee 1", "fee 2 (Binance VIP0)"]  # fmt: skip
    assert by["RA"]["sharpe_per_sqrt_hour"] == pytest.approx(-24 / math.sqrt(5720 / 4) * 60)
    assert by["PR"]["sharpe_per_sqrt_hour"] == pytest.approx(-24 / math.sqrt(5702 / 4) * 60)
    assert by["brief"]["model"] == "RA" and by["latency 0"]["model"] == "RA"
    assert by["latency 0"]["latency_ms"] == 0.0
    # fees are arithmetic on RA's fills: 30 maker units and the flatten's 10, gross -10, over
    # 5 s = 1/720 h. Fees 0: -10. Rebate 0.5: -10 + 15 - 50 = -45. 1: -10 - 30 - 50 = -90.
    # 2: -120, the brief row.
    for name, net in (("fees 0", -10), ("rebate 0.5", -45), ("fee 1", -90),
                      ("fee 2 (Binance VIP0)", -120), ("brief", -120)):  # fmt: skip
        assert by[name]["net_bps_per_hour"] == pytest.approx(net * 720), name
        assert by[name]["units"] == 30.0
    assert by["fees 0"]["fees_bps"] == 0.0 and by["rebate 0.5"]["fees_bps"] == pytest.approx(35)


def random_day(seed: int, minutes: int = 20):
    """A random walk of a one- or two-tick book with random sizes, trades at and through the
    touch, a funding settlement every five minutes, and quotes that move with it."""
    rng = np.random.default_rng(seed)
    n = minutes * 60 * 20
    t = np.sort(rng.uniform(0, minutes * 60, n))
    mid = 10_000 + np.cumsum(rng.choice([-1, 0, 0, 0, 1], n))
    bid = mid - rng.choice([0, 1], n)
    ask = bid + rng.choice([1, 1, 1, 2], n)
    book = pl.DataFrame({"ts": at(*t), "bid": bid * 0.01, "ask": ask * 0.01,
                         "bid_qty": rng.integers(1, 40, n).astype(float),
                         "ask_qty": rng.integers(1, 40, n).astype(float)})  # fmt: skip
    k = rng.integers(0, n, n // 4)
    side = rng.choice([-1.0, 1.0], len(k))
    through = rng.random(len(k)) < 0.1
    price = np.where(side > 0, ask[k] + through, bid[k] - through) * 0.01
    tt = t[k] + rng.uniform(0, 0.05, len(k))
    order = np.argsort(tt, kind="stable")
    trades = pl.DataFrame({"ts": at(*tt[order]), "side": side[order], "price": price[order],
                           "qty": rng.integers(1, 15, len(k)).astype(float)[order]})  # fmt: skip
    funding = pl.DataFrame(
        {
            "ts": at(*np.arange(150, minutes * 60, 300.0)),
            "rate": rng.normal(0, 1e-4, len(range(150, minutes * 60, 300))),
        }
    )
    grid = at(*np.arange(1, minutes * 60, 0.1))
    panel = Panel.from_frames(grid, book={"X": book}, trades={"X": trades},
                              funding={"X": funding})  # fmt: skip
    m = len(grid)

    def held(choices):  # a choice held for 5 s at a time, so that orders rest
        return np.repeat(rng.choice(choices, m // 50 + 1), 50)[:m]

    quotes = Quotes({"bid": held([0.0, 0.0, 1.0, 0.5, np.nan]),
                     "ask": held([0.0, 0.0, 1.0, -1.0]), "size": np.full(m, 3.0),
                     "skew": np.full(m, 0.7), "max_inventory": np.full(m, 9.0),
                     "lean": held([0.0, 0.0, 1.0, -1.0])})  # fmt: skip
    return panel, quotes, Window("w", pd.Timestamp(grid[0]), pd.Timestamp(grid[-1]))


@cross
@pytest.mark.parametrize("model", sim_maker.MODELS)
def test_the_gross_telescopes_to_the_cash_of_every_fill(model):
    # the sum of the per-step gross is the cash of the fills, the flatten's included, since the
    # flatten leaves nothing to mark: sum of -s q p, over P_ref
    panel, quotes, win = random_day(3)
    s = simulate_maker(quotes, panel, win, "X", model, tick=0.01,
                       latency=pd.Timedelta("150ms"), cancel_latency=pd.Timedelta("50ms"),
                       response_latency=pd.Timedelta("100ms"), maker_fee_bps=1.0,
                       taker_fee_bps=4.0, block=pd.Timedelta("1min"))  # fmt: skip
    b = s.book
    side = np.append(b.fill_side, b.flatten[1])
    qty = np.append(b.fill_qty, b.flatten[3])
    price = np.append(b.fill_price, b.flatten[2]) * 0.01
    cash = -(side * qty * price).sum() / s.ref * 1e4
    # the flatten crosses the inventory rounded to 1e-9 units, so a pro-rata residue under
    # that is held past it, worth at most its size at the day's prices
    residue = abs(float((side * qty).sum()))
    assert residue < 1e-9
    tol = 1e-9 * abs(cash) + residue * price.max() / s.ref * 1e4
    assert len(b.fill_ns) > 50
    assert float(s.gross.sum()) == pytest.approx(cash, abs=tol)
    fees = qty[:-1].sum() * 1.0 + qty[-1] * 4.0
    assert float(s.net.sum()) == pytest.approx(cash - fees + s.funding.sum(), abs=tol)
    assert float(s.run_net.sum()) == pytest.approx(float(s.net.sum()), abs=tol)
    # the position at each step start is every fill stamped by then
    t = panel.grid_ns[np.searchsorted(panel.grid_ns, win.start.value) :][: len(s.position)]
    t = t + pd.Timedelta("150ms").value
    inv = np.array([(b.fill_side * b.fill_qty)[b.fill_ns <= x].sum() for x in t])
    assert np.allclose(s.position, inv)
    # the cap holds the known inventory, so the true one never passes it by more than a size
    assert np.abs(s.position).max() <= 9.0 + 3.0


RAW = local("data/tardis/raw")
SOL = ("binance-futures_book_ticker_2026-10-01_SOLUSDT.csv.gz",
       "binance-futures_trades_2026-10-01_SOLUSDT.csv.gz")  # fmt: skip
NOON = pd.Timestamp("2026-10-01 12:00", tz="UTC").value


@pytest.fixture(scope="module")
def sol_noon():
    """The SOL perpetual's first ten minutes of 12:00 on 2026-10-01 from Tardis' raw files, on
    the receive clock, with the minute before for the book the first decision reads."""
    if not all((RAW / name).is_file() for name in SOL):
        pytest.skip(f"the SOL raw files are not in {RAW}")
    lo, hi = NOON - 60 * S, NOON + 601 * S
    book = pl.read_csv(RAW / SOL[0], columns=["local_timestamp", "ask_amount", "ask_price",
                                              "bid_price", "bid_amount"])  # fmt: skip
    trades = pl.read_csv(RAW / SOL[1], columns=["local_timestamp", "side", "price", "amount"])
    keep = (pl.col("local_timestamp") * 1000).is_between(lo, hi)
    book = book.filter(keep & (pl.col("ask_price") > pl.col("bid_price"))).to_pandas()
    trades = trades.filter(keep).to_pandas()
    book = pd.DataFrame(
        {
            "ts": book.local_timestamp * 1000,
            "bid": book.bid_price,
            "ask": book.ask_price,
            "bq": book.bid_amount,
            "aq": book.ask_amount,
        }
    )
    trades = pd.DataFrame({"ts": trades.local_timestamp * 1000, "price": trades.price,
                           "qty": trades.amount,
                           "side": np.where(trades.side == "buy", 1, -1)})  # fmt: skip
    # Tardis writes each file in receive order, so a stable sort keeps file order on a tie
    return (book.sort_values("ts", kind="stable", ignore_index=True),
            trades.sort_values("ts", kind="stable", ignore_index=True))  # fmt: skip


def pandas_fills(book, trades, grid, end, model, size=10.0, cap=10.0, latency=50_000_000):
    """The fill list of the join-the-touch policy, replayed order by order with pandas.

    Written apart from the engine: each order is a dict, replayed lazily on its own slice of
    the merged events up to the decision that needs its fills, with the touch it arrives on
    from ``merge_asof``. Fills are known at once (no response latency), cancels take as long
    as orders, prices are in cents.
    """
    cents = book.assign(bid=(book.bid * 100).round().astype(int),
                        ask=(book.ask * 100).round().astype(int))  # fmt: skip
    # by stamp, book rows first, then each file's own order (a sort on several keys is not
    # stable, so the order is a key)
    events = pd.concat(
        [
            cents.assign(kind=0, seq=np.arange(len(cents))),
            trades.assign(kind=1, seq=np.arange(len(trades)),
                          price=(trades.price * 100).round().astype(int)),
        ]
    )  # fmt: skip
    events = events.sort_values(["ts", "kind", "seq"], ignore_index=True)
    ev = list(events.itertuples(index=False))
    stamps = events.ts.to_numpy()
    known = pd.merge_asof(pd.DataFrame({"ts": grid}), cents, on="ts")  # what a decision sees
    view = {1: None, -1: None}
    active, fills = [], []

    def touch_at(t):
        return pd.merge_asof(pd.DataFrame({"ts": [t]}), cents, on="ts").iloc[0]

    def advance(o, upto):
        # replay ``o`` through every event stamped by ``upto`` and before its cancel lands
        if o["done"] or o["arrive"] > upto:
            return
        if o["at"] is None:  # it arrives
            row = touch_at(o["arrive"])
            if (o["side"] > 0 and o["price"] >= row.ask) or (
                o["side"] < 0 and o["price"] <= row.bid
            ):
                o.update(done=True, dead=o["arrive"])
                return
            mine, size_ = (row.bid, row.bq) if o["side"] > 0 else (row.ask, row.aq)
            better = o["price"] > mine if o["side"] > 0 else o["price"] < mine
            if model == "F":
                o["ahead"] = 0.0
            elif model == "T":
                o["ahead"] = None
            elif o["price"] == mine:
                o["ahead"], o["shown"], o["since"] = size_, size_, 0.0
            else:
                o["ahead"] = 0.0 if better else None
            # book rows on the arrival stamp are in that touch already; trades on it are not
            first = np.searchsorted(stamps, o["arrive"], side="left")
            while first < len(ev) and ev[first].ts == o["arrive"] and ev[first].kind == 0:
                first += 1
            o["at"] = first

        # a book row on the cancel's stamp still sees the order (rows come before cancels);
        # a trade on it does not
        def seen(e):
            return e.ts < o["cancel"] or (e.ts == o["cancel"] and e.kind == 0)

        while o["at"] < len(ev) and ev[o["at"]].ts <= upto and seen(ev[o["at"]]):
            e = ev[o["at"]]
            o["at"] += 1
            p, s = o["price"], o["side"]
            if e.kind == 0:
                level, shown = (e.bid, e.bq) if s > 0 else (e.ask, e.aq)
                # the far touch at the price (under T, through it) fills the rest there
                far = e.ask if s > 0 else e.bid
                gap = (p - far) if s > 0 else (far - p)
                if gap > 0 or (gap == 0 and model != "T"):
                    fills.append((e.ts, s, p, o["left"], o["at"]))
                    o.update(left=0.0, done=True, dead=e.ts)
                    return
                if model in ("T", "F"):
                    continue
                if level == p:
                    if o["ahead"] is None:
                        o["ahead"], o["shown"], o["since"] = shown, shown, 0.0
                    elif model == "RA":
                        o["ahead"] = min(o["ahead"], shown)
                    else:
                        expect = o["shown"] - o["since"]
                        a = o["ahead"]
                        if shown < expect:
                            a -= (expect - shown) * a / expect
                        o["ahead"], o["shown"], o["since"] = max(0.0, min(a, shown)), shown, 0.0
                elif (level < p) if s > 0 else (level > p):
                    o["ahead"], o["shown"], o["since"] = 0.0, 0.0, 0.0
                continue
            if e.side == s:  # an aggressor on our own side trades against the other book
                continue
            through = e.price < p if s > 0 else e.price > p
            got = 0.0
            if through:
                got = o["left"]
            elif e.price == p and model != "T" and o["ahead"] is not None:
                o["ahead"] -= e.qty
                o["since"] = o.get("since", 0.0) + e.qty
                if o["ahead"] < 0:
                    got, o["ahead"] = min(o["left"], -o["ahead"]), 0.0
            if got > 0:
                fills.append((e.ts, s, p, got, o["at"]))
                o["left"] -= got
                if o["left"] <= 1e-9 * size:
                    o.update(done=True, dead=e.ts)
                    return

    def decide(g, k):
        for o in active:
            advance(o, g)
        q = round(sum(s * x for t, s, _, x, _ in fills if t <= g), 9)
        if k is None:  # the end: everything out is cancelled
            want = {1: None, -1: None}
        else:
            row = known.iloc[k]
            want = {1: int(row.bid) if q + size <= cap + 1e-9 else None,
                    -1: int(row.ask) if q - size >= -cap - 1e-9 else None}  # fmt: skip
        for s in (1, -1):
            o = view[s]
            alive = o is not None and not (o["dead"] is not None and o["dead"] <= g)
            if alive and want[s] == o["price"]:
                continue
            if alive:
                o["cancel"] = g + latency
            view[s] = None
            if want[s] is not None:
                view[s] = {"side": s, "price": want[s], "arrive": g + latency, "cancel": np.inf,
                           "left": size, "at": None, "done": False, "dead": None,
                           "since": 0.0}  # fmt: skip
                active.append(view[s])

    for k, g in enumerate(grid):
        decide(int(g), k)
    decide(end, None)
    for o in active:
        advance(o, end + latency)
    # in the order of the events that made them
    return [f[:4] for f in sorted(fills, key=lambda f: f[4])]


@cross
@pytest.mark.parametrize("model", sim_maker.MODELS)
def test_the_engine_agrees_with_a_pandas_replay_on_ten_minutes_of_sol(sol_noon, model):
    # the policy of the hand example at size 10 on the SOL perpetual, 12:00 to 12:10 on the
    # 100 ms grid, 50 ms to send and cancel, fills known at once: the same fills, stamp,
    # side, price and quantity, from the engine and from the replay above
    book, trades = sol_noon
    touch = Touch(0.01, book.ts.to_numpy(), sim_maker.to_ticks(book.bid.to_numpy(), 0.01),
                  sim_maker.to_ticks(book.ask.to_numpy(), 0.01), book.bq.to_numpy(float),
                  book.aq.to_numpy(float), trades.ts.to_numpy(),
                  sim_maker.to_ticks(trades.price.to_numpy(), 0.01), trades.qty.to_numpy(float),
                  trades.side.to_numpy())  # fmt: skip
    grid = NOON + np.arange(6000, dtype=np.int64) * 100_000_000
    end = NOON + 600 * S
    dec = Decisions.make(_policy_arrays(len(grid)), grid, touch, 0.01)
    b = run_engine(touch, dec, end, model, 50_000_000, 50_000_000, 0)
    mine = [(int(t), int(s), int(p), round(float(q), 9)) for t, s, p, q in
            zip(b.fill_ns, b.fill_side, b.fill_price, b.fill_qty, strict=True)]  # fmt: skip
    theirs = [(int(t), s, p, round(q, 9)) for t, s, p, q in
              pandas_fills(book, trades, grid, end, model)]  # fmt: skip
    assert len(mine) > 20
    assert mine == theirs


@known
def test_bl_quote_has_the_six_keys_and_fair_when_given():
    assert isinstance(bl.quote(), Quotes)
    assert list(bl.quote()) == ["bid", "ask", "size", "skew", "max_inventory", "lean"]
    assert list(bl.quote(fair=99.995))[-1] == "fair"


@known
def test_quotes_are_evaluated_key_by_key():
    # the bid size as-of 1, 2, 3, 4, 5 s is 50 (the 0.0 s row), 40 (1.8 s), 60 (2.7 s), 20
    # (3.3 s), 80 (4.6 s); a number is repeated on every point and None withdraws everywhere
    panel = hand_panel()
    q = values(bl.quote(bid=pl.col("X.bid_qty") / 10, ask=None, size=2), panel)
    assert isinstance(q, Quotes)
    assert q["bid"].tolist() == [5.0, 4.0, 6.0, 2.0, 8.0]
    assert np.isnan(q["ask"]).all() and q["size"].tolist() == [2.0] * 5
    assert (q["skew"].tolist(), q["max_inventory"].tolist()) == ([0.0] * 5, [1.0] * 5)


@edge
def test_quotes_belong_to_a_market_making_strategy_and_only_to_one():
    panel = hand_panel()
    with pytest.raises(
        ContractError, match=re.escape(output_rule("strategy", ("X",), template="maker"))
    ):
        values(pl.col("X.mid"), panel, ("X",), "maker")
    with pytest.raises(ContractError, match="a strategy trading X returns one position"):
        values(bl.quote(), panel, ("X",), "threshold/linear")
    with pytest.raises(ContractError, match="a market-making strategy returns bl.quote"):
        values(Quotes(bid=0, ask=0), panel)  # size, skew and max_inventory missing


@known
def test_the_truncation_test_reads_a_quote_key_by_key(synthetic_panel):
    # a lean from a centred mean sees the future exactly where the centred mean itself does:
    # the same cut, the same first point, and the key that disagreed is named
    centred = pl.col("XRPUSDT.mid").rolling_mean(21, center=True)
    p = synthetic_panel
    win = Window("w", p.time(0) + pd.Timedelta("5min"), p.time(-1))
    plain = truncation.check(lambda panel: centred, p, win, seed=3)
    quoted = truncation.check(lambda panel: bl.quote(lean=centred), p, win, seed=3)
    assert plain.status == quoted.status == "rejected_lookahead"
    a, b = plain.violation, quoted.violation
    assert (a.cut, a.first) == (b.cut, b.first)
    assert (b.key, b.leg) == ("lean", None) and ", key lean:" in str(b)
    # an honest quote passes, every key compared
    fine = truncation.check(lambda panel: bl.quote(lean=pl.col("XRPUSDT.mid").diff()), p, win)
    assert fine.ok and isinstance(fine.full, Quotes)


def judge_brief(maker_fee: float):
    window = "2026-10-01T00:00:01Z/2026-10-01T00:00:06Z"
    raw = {"name": "hand", "data": {"path": "x.parquet", "instrument_column": "symbol",
                                    "streams": {"book": {"fields": {"bid": "bid"}}}},
           "target": {"instrument": "X", "horizons": ["1s"]},
           "grid": {"step": "1s", "warmup": "0s"},
           "execution": {"latency_ms": 500, "taker_fee_bps": 5.0,
                         "maker_fee_bps": maker_fee, "tick_size": 0.01},
           "splits": {"embargo": "2s", "windows": {"train": window, "val": window,
                                                   "test": window}},
           "stats": {"block": "1s", "n_boot": 50, "seed": 0}}  # fmt: skip
    return briefs.parse(raw)


@known
@pytest.mark.parametrize("fee, scored", [(0.0, "PR"), (2.0, "RA"), (-5.0, "RA")])
def test_the_judge_scores_the_worse_of_the_two_queue_models(fee, scored):
    # the hand example as validation, every model its own pass. RA holds one step at a time,
    # so its blocks are the brief's 1 s; PR holds two steps (long from 2.1 s), so it is
    # bootstrapped in 2 s blocks, the fifth step, which holds the -70 flatten, folded into the
    # second: blocks of steps 1 and 2, and 3 to 5. With no maker fee RA nets 0, 10, 0, 0, -70,
    # -12 / sqrt(4280 / 4) x 60 = -22.01; PR nets 1, 9, 0, 0, -70, blocks 10 and -70, mean
    # -30, sd 40 sqrt 2: -30 / (40 sqrt 2) x 30 sqrt 2 = -22.5, the lower, and scored. At a
    # 2 bps fee RA nets 0, -10, -20, 0, -90: -24 / sqrt(5720 / 4) x 60 = -38.08, the lower;
    # PR -1, -9, -20, 0, -90, blocks -10 and -110, mean -60, sd 50 sqrt 2: -36.0. At a 5 bps
    # rebate the nets are 0, 60, 50, 0, -20 and 6, 54, 50, 0, -20 (each maker unit earns 5,
    # the flatten pays 50): RA 18 / sqrt(4880 / 4) x 60 = 30.92, the lower; PR blocks 60 and
    # 30, mean 45, sd 15 sqrt 2: 90.0. Before the fold PR's second block was steps 3 and 4
    # alone, and PR read -90.0 at 2 bps (scored) and +30.0 with no fee (RA scored).
    from blinded import daemon
    from blinded.contract import Binding

    panel = hand_panel()
    strat = bl.market_making(instrument="X")(lambda panel, features: bl.quote(**POLICY))
    bound = Binding("directional", ("X",), edge="liquidity", archetype="market_making",
                    template="maker")  # fmt: skip

    def fn(p):
        return values(bl.quote(**POLICY), p, ("X",), "maker")

    out = daemon._judge_steps({"kind": "strategy", "name": "join"}, strat, "strategy", fn,
                              judge_brief(fee), lambda seg: panel, "val", bound)  # fmt: skip
    val = out["val"]
    want = {(0.0, "RA"): -12 / math.sqrt(4280 / 4) * 60, (0.0, "PR"): -22.5,
            (2.0, "RA"): -24 / math.sqrt(5720 / 4) * 60, (2.0, "PR"): -36.0,
            (-5.0, "RA"): 18 / math.sqrt(4880 / 4) * 60, (-5.0, "PR"): 90.0}  # fmt: skip
    assert val["fill_model"] == scored
    assert val["value"] == pytest.approx(want[(fee, scored)])
    assert val["value"] == pytest.approx(min(want[(fee, "RA")], want[(fee, "PR")]))
    # every model's validation numbers are kept; the brief row is the scored model's
    assert set(val["models"]) == set(sim_maker.MODELS)
    assert val["metrics"]["rows"]["brief"] == val["metrics"]["rows"][scored]
    assert val["hold_steps"] == (2.0 if scored == "PR" else 1.0)
    assert len(val["replicates"]) == 50


@known
def test_a_declared_touch_file_reads_as_the_panels_own_rows():
    # the hand example's book rows in the touch file's layout, prices in ticks, give the same
    # stream as the panel's own book rows, the trades from the panel either way
    panel = hand_panel()
    rows = pl.DataFrame({"ts": at(*(r[0] for r in BOOK)), "bid": [round(r[1] * 100) for r in BOOK],
                         "ask": [round(r[3] * 100) for r in BOOK],
                         "bid_qty": [float(r[2]) for r in BOOK],
                         "ask_qty": [float(r[4]) for r in BOOK]})  # fmt: skip
    a, b = sim_maker.touch_from(rows, panel, "X", 0.01), sim_maker.touch_of(panel, "X", 0.01)
    for name in ("book_ns", "bid", "ask", "bid_qty", "ask_qty", "trade_ns", "price", "qty",
                 "side"):  # fmt: skip
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


def judged(fee: float, quote: dict | None = None, segment: str = "val"):
    """The daemon's judgement of a quote on the hand example as every segment."""
    from blinded import daemon
    from blinded.contract import Binding

    q = POLICY if quote is None else quote
    panel = hand_panel()
    strat = bl.market_making(instrument="X")(lambda panel, features: bl.quote(**q))
    bound = Binding("directional", ("X",), edge="liquidity", archetype="market_making",
                    template="maker")  # fmt: skip

    def fn(p):
        return values(bl.quote(**q), p, ("X",), "maker")

    item = {"kind": "strategy", "name": "join", "family": "directional", "group": "liquidity"}
    return daemon._judge_steps(item, strat, "strategy", fn, judge_brief(fee),
                               lambda seg: panel, segment, bound)  # fmt: skip


@known
def test_the_page_and_the_judge_take_the_same_model_when_the_blocks_disagree():
    # at a 2 bps maker fee the nets by step are RA 0, -10, -20, 0, -90 and PR -1, -9, -20, 0,
    # -90: the same mean, -24, and PR's squares about it the smaller (5,702 against 5,720), so
    # on the brief's 1 s blocks PR is the lower, -24 / sqrt(5702 / 4) x 60 = -38.14 against
    # RA's -24 / sqrt(5720 / 4) x 60 = -38.08. The judge's rule measures each model on blocks
    # as long as its holds: RA holds one step, so its blocks are the brief's; PR holds two
    # (long from 2.1 s), so its blocks are 2 s with the fifth step folded into the second,
    # -10 and -110, and its Sharpe is -60 / (50 sqrt 2) x 30 sqrt 2 = -36.0. RA's is the
    # lower, and RA is what the judge scores and what the page's bracket marks.
    ra = np.array([0.0, -10, -20, 0, -90])
    pr = np.array([-1.0, -9, -20, 0, -90])

    def sr(x, per_hour):
        return x.mean() / x.std(ddof=1) * math.sqrt(per_hour)

    assert sr(pr, 3600) == pytest.approx(-24 / math.sqrt(5702 / 4) * 60)
    assert sr(ra, 3600) == pytest.approx(-24 / math.sqrt(5720 / 4) * 60)
    assert sr(pr, 3600) < sr(ra, 3600)  # the brief's blocks would pick PR
    sims = {m: run(m) for m in ("RA", "PR")}
    for m, nets in (("RA", ra), ("PR", pr)):
        assert np.allclose(sims[m].net, nets), m
    held = {m: sim_maker.held(s) for m, s in sims.items()}
    assert held["RA"]["sharpe"] == pytest.approx(sr(ra, 3600))
    assert (held["RA"]["block_steps"], held["RA"]["blocks"]) == (1, 5)
    assert held["PR"]["sharpe"] == pytest.approx(-36.0)
    assert (held["PR"]["block_steps"], held["PR"]["blocks"]) == (2, 2)
    assert sim_maker.worse(sims["RA"], sims["PR"]).model == "RA"
    # the judge, the sim table's brief row and the report's bracket all take RA
    val = judged(2.0)["val"]
    assert val["fill_model"] == "RA" and val["value"] == pytest.approx(sr(ra, 3600))
    assert val["models"]["PR"]["held"]["sharpe"] == pytest.approx(-36.0)
    rows = sim.report(bl.quote(**POLICY), hand_panel(), WINDOW, judge_brief(2.0), "X")
    assert rows.filter(pl.col("row") == "brief")["model"].item() == "RA"


@known
def test_a_score_that_measures_nothing_is_null_and_never_reaches_the_ladder():
    # a bid alone, never an ask: RA buys ten at 2.6 s and holds them to the flatten, so its
    # inventory is one run of three steps (3.5, 4.5 and 5.5 s) and its own block is three
    # steps, of which the five-step window holds one; PR buys one at 2.1 s and nine at 2.6 s,
    # a run of four steps, one block of four. A Sharpe of one block is NaN for both, so the
    # score is null, with its reason, and on test the summary's score is null too.
    quote = dict(bid=0, ask=None, size=10, skew=0, max_inventory=10)
    for segment in ("val", "test"):
        item = judged(2.0, quote, segment)
        out = item[segment]
        assert out["value"] is None and out["fill_model"] == "RA"
        assert out["block_steps"] == 3 and out["own_blocks"] == 1
        assert out["score_note"] == ("no Sharpe: its holds give 1 block of 3 steps on the "
                                     "window, and a Sharpe needs two")  # fmt: skip
        assert out["models"]["PR"]["held"] == {"sharpe": None, "hold_steps": 4.0,
                                               "block_steps": 4, "blocks": 1}  # fmt: skip
    summary = item["test_summary"]
    assert summary["score"] is None and summary["own_blocks"] == 1
    assert summary["score_note"].startswith("no Sharpe")
    assert summary["rows"]["brief"]["net_bps_per_hour"] is not None  # the rows still stand


@known
@pytest.mark.parametrize("fee", [0.0, 2.0])
def test_a_runs_manifest_picks_the_rules_it_is_judged_under(tmp_path, fee):
    # the hand example as validation, judged through a run directory's brief: its manifest's
    # judge_rules decides the rules, and a manifest without one is rules 1. Under rules 1 PR's
    # 2 s blocks are steps 1 and 2, then 3 and 4, the -70 flatten in step 5 dropped: no fee,
    # blocks 10 and 0, mean 5, sd 5 sqrt 2: 5 / (5 sqrt 2) x 30 sqrt 2 = +30.0, so RA (-22.01)
    # is scored; at 2 bps blocks -10 and -20: -15 / (5 sqrt 2) x 30 sqrt 2 = -90.0, scored.
    # Under rules 2 the flatten folds into the second block: -22.5 (PR scored) and -36.0 (RA
    # scored, -38.08). RA holds one step, so its 1 s blocks hold every step under both. No book
    # row of the example reaches a resting price, so the crossing changes no fill here.
    import json

    import yaml

    from blinded import daemon
    from blinded.contract import Binding
    from blinded.vault import RunDir

    panel = hand_panel()
    strat = bl.market_making(instrument="X")(lambda panel, features: bl.quote(**POLICY))
    bound = Binding("directional", ("X",), edge="liquidity", archetype="market_making",
                    template="maker")  # fmt: skip

    def fn(p):
        return values(bl.quote(**POLICY), p, ("X",), "maker")

    ra = {0.0: -12 / math.sqrt(4280 / 4) * 60, 2.0: -24 / math.sqrt(5720 / 4) * 60}[fee]
    want = {None: (ra, {0.0: 30.0, 2.0: -90.0}[fee]), 1: (ra, {0.0: 30.0, 2.0: -90.0}[fee]),
            2: (ra, {0.0: -22.5, 2.0: -36.0}[fee])}  # fmt: skip
    for rules, (want_ra, want_pr) in want.items():
        root = tmp_path / f"run-{rules}"
        (root / "vault").mkdir(parents=True)
        (root / "vault" / "brief.yaml").write_text(yaml.safe_dump(judge_brief(fee).raw))
        manifest = {"run_id": "hand", "windows": {}, "segments": {}}
        if rules is not None:
            manifest["judge_rules"] = rules
        (root / "vault" / "manifest.json").write_text(json.dumps(manifest))
        brief = RunDir.open(root).brief()
        assert brief.judge_rules == (rules or 1)
        out = daemon._judge_steps({"kind": "strategy", "name": "join"}, strat, "strategy", fn,
                                  brief, lambda seg: panel, "val", bound)  # fmt: skip
        val = out["val"]
        got = {m: val["models"][m]["held"]["sharpe"] for m in ("RA", "PR")}
        assert got["RA"] == pytest.approx(want_ra) and got["PR"] == pytest.approx(want_pr)
        assert val["value"] == pytest.approx(min(want_ra, want_pr))
        assert val["fill_model"] == ("RA" if want_ra < want_pr else "PR")


@known
@pytest.mark.parametrize("model", sim_maker.MODELS)
def test_under_rules_1_a_book_row_reaching_a_resting_price_fills_nothing(model):
    # the four-row book of the crossing test above, where no trade prints all window: under
    # rules 2 the far touch reaching each resting order fills it, under rules 1 only a trade
    # fills, so nothing does and both orders rest until they are cancelled or the window ends
    book = [(0.0, 9999, 50, 10001, 40), (2.2, 9998, 60, 9999, 30), (2.6, 9997, 60, 9998, 30),
            (3.2, 10001, 10, 10002, 40)]  # fmt: skip
    ns = np.array([int(r[0] * S) for r in book], dtype=np.int64)
    col = [np.array([r[i] for r in book]) for i in range(1, 5)]
    empty = np.zeros(0)
    touch = Touch(0.01, ns, col[0].astype(np.int64), col[2].astype(np.int64),
                  col[1].astype(float), col[3].astype(float), empty.astype(np.int64),
                  empty.astype(np.int64), empty, empty.astype(np.int64))  # fmt: skip
    grid = np.array([1, 2, 3], dtype=np.int64) * S
    q = Quotes({"bid": np.zeros(3), "ask": np.zeros(3), "size": np.full(3, 10.0),
                "skew": np.zeros(3), "max_inventory": np.full(3, 10.0)})  # fmt: skip
    old = run_engine(touch, Decisions.make(q, grid, touch, None), 4 * S, model, S // 2, rules=1)
    new = run_engine(touch, Decisions.make(q, grid, touch, None), 4 * S, model, S // 2, rules=2)
    assert len(old.fill_ns) == 0 and len(new.fill_ns) == (1 if model == "T" else 2)
    with pytest.raises(ValueError, match="judge_rules"):
        run_engine(touch, Decisions.make(q, grid, touch, None), 4 * S, model, S // 2, rules=4)


@known
@pytest.mark.parametrize("model", ["RA", "PR", "T", "F"])
@pytest.mark.parametrize("rules", [1, 2, 3])
def test_a_trade_and_the_row_that_shows_it_on_one_stamp(model, rules):
    # the simulator audit's hand example (audit-sim/scripts/a4_hand.py). The book at 0 s: bid
    # 100.00 x 10, ask 100.01 x 10. A bid of 1 joins at 100.00 at 1.1 s (decided at 1 s, 100 ms
    # latency), 10 ahead. At 2.0 s one packet carries a sell of 6 at 100.00 and the book row
    # after it, bid 100.00 x 4: in truth 10 - 6 = 4 are still ahead and nothing fills. Read
    # row first (rules 1 and 2), RA clamps the queue ahead to the row's 4 and the trade then
    # takes 6 from it again, filling the order at 2.0 s; PR scales its share of the queue
    # the same way. Read trade first (rules 3): 4 ahead, then a row of 4, no fill. Neither
    # bracket reads the queue: F, the front of it, fills on the trade at its price under any
    # order, and T, trades through it only, never fills on a trade at its price
    s = 1_000_000_000
    touch = Touch(tick=0.01, book_ns=np.array([0, 2 * s, 3 * s]),
                  bid=np.array([10000, 10000, 10000]), ask=np.array([10001] * 3),
                  bid_qty=np.array([10.0, 4.0, 4.0]), ask_qty=np.array([10.0] * 3),
                  trade_ns=np.array([2 * s]), price=np.array([10000]), qty=np.array([6.0]),
                  side=np.array([-1]))  # fmt: skip
    g = np.array([1 * s, 2 * s + 500_000_000])
    n = len(g)
    dec = Decisions(g, np.array([10000.0] * n), np.array([np.nan] * n), np.zeros(n),
                    np.full(n, np.nan), np.zeros(n), np.full(n, 1e9), np.ones(n))  # fmt: skip
    book = run_engine(touch, dec, int(4 * s), model=model, latency_ns=100_000_000, rules=rules)
    fills = list(zip((book.fill_ns / s).tolist(), book.fill_qty.tolist(), strict=True))
    queued = model in ("RA", "PR")
    if queued and rules < 3 or model == "F":
        assert fills == [(2.0, 1.0)]
    else:
        assert fills == []
