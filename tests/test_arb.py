"""The hedge leg of a maker-taker arbitrage, on a hand example.

Grid 1 to 5 s, window [1 s, 6 s). Venue A (the quote venue) and venue B (the
hedge venue), ticks 0.0001 on both. A's latencies, order, cancel and response,
are 0.5 s, B's order latency 0.5 s, so a hedge reaches B 1.0 s after its fill.
P_ref is B's mid as-of 1.5 s, 1.0000, and the size 100 units, so 0.01 USD is
1 bps of the fill and 100 bps of one unit, the unit every P&L array is in. A's
maker fee 0, B's taker fee 10 bps. The policy is
``bl.quote(bid=30, ask=30, size=100, skew=0, max_inventory=1e9, fair=(B's bid,
B's ask))``: the bid 30 ticks under floor(bid_B / tick), the ask 30 over
ceil(ask_B / tick).

B's touch (stamp: bid / size, ask / size), and the depth rows at the same
stamps, three levels a side (the example gives the levels at 3.0 and 5.0 s,
where the two hedges arrive; the others are written here so that the slower
hedge of the 1.5 s variant has a book to walk):

    0.0: 0.9995 / 300, 1.0005 / 300
    2.0: 0.9990 / 150, 1.0000 / 200
    3.0: 0.9988 / 40,  0.9998 / 100    bids 0.9988 / 40, 0.9987 / 50, 0.9986 / 200
    4.1: 1.0040 / 100, 1.0050 / 100    bids 1.0040 / 100, 1.0039 / 100, 1.0038 / 100
    5.0: 1.0042 / 80,  1.0052 / 60     asks 1.0052 / 60, 1.0053 / 100, 1.0054 / 300

A's touch: 0.0 s 0.9900 / 500, 1.0100 / 400; 4.3 s 1.0030 / 200, 1.0080 /
100. A's trades: 2.3 s a sell of 100 at 0.9960; 4.2 s a buy of 100 at 1.0035.

No book row of A reaches a resting price, so the book-crossing rule (rules 2) fills
nothing here: A's rows are at 0.0 s, before any order, and at 4.3 s, when the
only live order is the bid at 0.9958 (from 3.5 s), under A's ask of 1.0080;
the ask at 1.0028 had filled at 4.2 s. The numbers stand under rules 2.
"""

import dataclasses
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

import blinded as bl
from blinded.brief import Stock, Window
from blinded.contract import values
from blinded.panel import Panel
from blinded.sim_maker import (
    Depth,
    arb_fills,
    simulate_arb,
    stock_path,
)

pytestmark = pytest.mark.usefixtures("rules2")  # hand numbers under rules 2
known = pytest.mark.known_answer
edge = pytest.mark.edge_case
T0 = datetime(2026, 10, 1, tzinfo=UTC)
S = 10**9
HALF = pd.Timedelta("500ms")
TICK = 0.0001
WINDOW = Window("w", pd.Timestamp(T0 + timedelta(seconds=1)),
                pd.Timestamp(T0 + timedelta(seconds=6)))  # fmt: skip
A_BOOK = [(0.0, 0.9900, 500, 1.0100, 400), (4.3, 1.0030, 200, 1.0080, 100)]
A_TRADES = [(2.3, -1, 100, 0.9960), (4.2, 1, 100, 1.0035)]
B_BOOK = [(0.0, 0.9995, 300, 1.0005, 300), (2.0, 0.9990, 150, 1.0000, 200),
          (3.0, 0.9988, 40, 0.9998, 100), (4.1, 1.0040, 100, 1.0050, 100),
          (5.0, 1.0042, 80, 1.0052, 60)]  # fmt: skip
# three levels a side at each of B's stamps: (bids, asks), each [(price, size), ...]
B_DEPTH = {
    0.0: ([(0.9995, 300), (0.9994, 300), (0.9993, 300)],
          [(1.0005, 300), (1.0006, 300), (1.0007, 300)]),
    2.0: ([(0.9990, 150), (0.9989, 200), (0.9988, 200)],
          [(1.0000, 200), (1.0001, 200), (1.0002, 200)]),
    3.0: ([(0.9988, 40), (0.9987, 50), (0.9986, 200)],
          [(0.9998, 100), (0.9999, 100), (1.0000, 200)]),
    4.1: ([(1.0040, 100), (1.0039, 100), (1.0038, 100)],
          [(1.0050, 100), (1.0051, 100), (1.0052, 100)]),
    5.0: ([(1.0042, 80), (1.0041, 100), (1.0040, 100)],
          [(1.0052, 60), (1.0053, 100), (1.0054, 300)]),
}  # fmt: skip
BIG = Stock(coin={"A": 1000.0, "B": 1000.0}, usdt={"A": 1000.0, "B": 1000.0},
            rebalance_at=1.0, withdrawal_fee_coin=0.0, withdrawal_fee_usdt=0.0)  # fmt: skip


def at(*seconds):
    return [T0 + timedelta(seconds=s) for s in seconds]


def secs(ns) -> float:
    return round((int(ns) - int(pd.Timestamp(T0).value)) / S, 6)


def book_frame(rows):
    return pl.DataFrame({"ts": at(*(r[0] for r in rows)), "bid": [r[1] for r in rows],
                         "bid_qty": [float(r[2]) for r in rows], "ask": [r[3] for r in rows],
                         "ask_qty": [float(r[4]) for r in rows]})  # fmt: skip


def trade_frame(rows):
    return pl.DataFrame({"ts": at(*(r[0] for r in rows)), "side": [float(r[1]) for r in rows],
                         "qty": [float(r[2]) for r in rows],
                         "price": [r[3] for r in rows]})  # fmt: skip


def depth_of(levels: dict) -> Depth:
    stamps = sorted(levels)
    n = len(levels[stamps[0]][0])

    def col(side, i, k):
        return np.array([levels[t][side][i][k] for t in stamps], dtype=float)

    return Depth(np.array([int(pd.Timestamp(x).value) for x in at(*stamps)], dtype=np.int64),
                 np.column_stack([col(0, i, 0) for i in range(n)]),
                 np.column_stack([col(0, i, 1) for i in range(n)]),
                 np.column_stack([col(1, i, 0) for i in range(n)]),
                 np.column_stack([col(1, i, 1) for i in range(n)]))  # fmt: skip


def hand_panel(a_trades=A_TRADES, a_book=A_BOOK, b_book=B_BOOK) -> Panel:
    return Panel.from_frames(at(1, 2, 3, 4, 5), book={"A": book_frame(a_book),
                                                      "B": book_frame(b_book)},
                             trades={"A": trade_frame(a_trades)})  # fmt: skip


def policy(**kw):
    return bl.quote(**({"bid": 30, "ask": 30, "size": 100, "skew": 0, "max_inventory": 1e9,
                        "fair": (pl.col("B.bid"), pl.col("B.ask"))} | kw))  # fmt: skip


def run(model="RA", panel=None, quote=None, depth=None, stock=BIG, **kwargs):
    panel = panel or hand_panel()
    kwargs = dict(tick=TICK, latency=HALF, hedge_latency=HALF, maker_fee_bps=0.0,
                  hedge_fee_bps=10.0, block=pd.Timedelta("1s")) | kwargs  # fmt: skip
    q = values(policy() if quote is None else quote, panel)
    return simulate_arb(q, panel, WINDOW, "A", "B", stock, model,
                        depth=depth or depth_of(B_DEPTH), **kwargs)  # fmt: skip


def usd(x):
    return round(float(x), 6)


@known
@pytest.mark.parametrize("model", ["RA", "PR", "T", "F"])
def test_the_hand_example_both_round_trips(model):
    s = run(model)
    f = arb_fills(s)
    # both fills are prints through a lone quote inside A's 0.9900 / 1.0100 spread, so the
    # four models agree: a buy of 100 at 0.9965 at 2.3 s, a sale of 100 at 1.0028 at 4.2 s
    assert [secs(t) for t in f["ns"]] == [2.3, 4.2]
    assert f["side"].tolist() == [1.0, -1.0]
    assert [round(p, 4) for p in f["price"]] == [0.9965, 1.0028]
    assert f["qty"].tolist() == [100.0, 100.0]
    # each hedge 1.0 s after its fill: 2.3 + 0.5 (response) + 0.5 (B's order) = 3.3 s, 5.2 s
    assert [secs(t) for t in f["hedge_ns"]] == [3.3, 5.2]
    # the first walk, a sale of 100 into the 3.0 s bids: 40 at 0.9988 = 39.952, 50 at
    # 0.9987 = 49.935, 10 at 0.9986 = 9.986; 99.873 in all, vwap 0.99873
    assert usd(f["vwap"][0] * 100) == 99.873
    # the second, a purchase of 100 from the 5.0 s asks: 60 at 1.0052 = 60.312, 40 at
    # 1.0053 = 40.212; 100.524, vwap 1.00524
    assert usd(f["vwap"][1] * 100) == 100.524
    # the first round trip: captured 100 x (0.9995 - 0.9965) = +0.30 (B's 2.0 s mid), the
    # delay 100 x (0.9993 - 0.9995) = -0.02 (the 3.0 s mid at the hedge), the slippage
    # 100 x (0.99873 - 0.9993) = -0.057, the fee 10 bps x 100 x 1.0000 = 0.100; net +0.123
    assert [usd(f[k][0]) for k in ("captured_usd", "delay_usd", "slippage_usd", "fee_usd",
                                   "net_usd")] == [0.30, -0.02, -0.057, 0.1, 0.123]  # fmt: skip
    # the second: captured -100 x (1.0045 - 1.0028) = -0.17 (the 4.1 s mid: B moved 45 bps
    # in the 1.1 s since the quote was priced), the delay -100 x (1.0047 - 1.0045) = -0.02,
    # the slippage -100 x (1.00524 - 1.0047) = -0.054, the fee 0.100; net -0.344
    assert [usd(f[k][1]) for k in ("captured_usd", "delay_usd", "slippage_usd", "fee_usd",
                                   "net_usd")] == [-0.17, -0.02, -0.054, 0.1, -0.344]  # fmt: skip
    # in bps of the 100 USD at P_ref: +30 - 2 - 5.7 - 10 = +12.3 and -17 - 2 - 5.4 - 10 = -34.4
    # (arb_fills' own bps are of each fill's notional at B's mid then, as the probe's are:
    # 0.123 / 99.95 = +12.31 and -0.344 / 100.45 = -34.25)
    assert [round(float(x / (100 * s.ref) * 1e4), 6) for x in f["net_usd"]] == [12.3, -34.4]
    assert [round(float(x), 2) for x in f["net_bps"]] == [12.31, -34.25]
    assert f["picked_off"].tolist() == [False, True]
    # B's hedge-side touch at the fill: the bid 0.9990 above the bought 0.9965, not through;
    # the ask 1.0050 above the sold 1.0028, through (B's bid 1.0040 is already above it too)
    assert f["touch_through"].tolist() == [False, True]
    assert not f["unreachable"].any()
    assert not f["crossed"].any()  # no crossing under rules 2 (the module's docstring)
    # the total: A -99.65 + 100.28 = +0.63, B +99.873 - 100.524 = -0.651, fees -0.200;
    # -0.221 USD, -2210 bps of one unit at 1.0000
    assert usd(s.net.sum() * s.ref * 1e-4) == -0.221
    assert round(float(s.net.sum()), 6) == -2210.0
    assert round(float(s.gross.sum()), 6) == -210.0
    assert round(float(s.fee_paid.sum()), 6) == 2000.0
    # markouts of B's mid 1 s after each fill: (0.9993 - 0.9995) / 0.9995 = -2 bps, and
    # -(1.0047 - 1.0045) / 1.0045 = -2 bps
    later = s.mark.mid(f["ns"] + S)
    assert [round(float(x), 1) for x in f["side"] * (later - f["mid_fill"]) / f["mid_fill"]
            * 1e4] == [-2.0, -2.0]  # fmt: skip


@known
def test_the_hand_example_per_step_and_the_stock_after_each_leg():
    s = run("RA")
    # steps (1.5, 2.5], (2.5, 3.5], ... to 6.5 s: the fill in step 0, its hedge in step 1,
    # the second fill in step 2 and its hedge in step 3; in bps of one unit at 1.0000
    assert [round(float(x), 6) for x in s.net] == [3000.0, -1770.0, -1700.0, -1740.0, 0.0]
    assert [round(float(x), 6) for x in s.position] == [0.0, 100.0, 0.0, -100.0, 0.0]
    assert s.units.tolist() == [100.0, 0.0, 100.0, 0.0, 0.0]
    assert s.taker_units.tolist() == [0.0, 100.0, 0.0, 100.0, 0.0]
    rows = stock_path(s)["rows"]
    # (stamp, leg, coin A, USDT A, coin B, USDT B), from 1000 of each
    got = [(secs(r[0]), r[1], *(round(x, 6) for x in r[2:])) for r in rows]
    assert got == [
        (2.3, "fill", 1100.0, 900.35, 1000.0, 1000.0),  # A: +100 coin, -99.65 USDT
        (3.3, "hedge", 1100.0, 900.35, 900.0, 1099.773),  # B: -100 coin, +99.873 - 0.100
        (4.2, "fill", 1000.0, 1000.63, 900.0, 1099.773),  # A: -100 coin, +100.28
        (5.2, "hedge", 1000.0, 1000.63, 1000.0, 999.149),  # B: +100 coin, -100.524 - 0.100
    ]
    # both venues flat in coin, USDT down 0.221 between them
    assert round(1000.63 + 999.149 - 2000.0, 6) == -0.221
    m = s.metrics()
    assert (m["fills"], m["units"], m["hedged_units"]) == (2, 200.0, 200.0)
    assert round(m["picked_off_share"], 6) == 0.5
    assert m["rebalances"] == 0 and m["stock_withdrawn_share"] == 0.0


@known
def test_a_slower_hedge_reads_the_depth_as_of_its_own_arrival():
    # hedge_latency_ms 1500: each hedge reaches B 2.0 s after its fill. The first, at 4.3 s,
    # walks the 4.1 s bids: 100 at 1.0040, against the 4.1 s mid 1.0045: captured +0.30, the
    # delay 100 x (1.0045 - 0.9995) = +0.50, the slippage 100 x (1.0040 - 1.0045) = -0.05,
    # the fee 0.100, net +0.65. The second fill is known at 4.7 s; 4.7 + 1.5 = 6.2 s is past
    # the window's end decision at 6.0 s, so it is sent then, min(4.7, 6.0) + 1.5 = 6.2 s,
    # and still reads the 5.0 s row: the same walk and the same -0.344. Had the hedge read
    # the depth as-of the fill's stamp, the first would have walked the 2.0 s bids.
    s = run("RA", hedge_latency=pd.Timedelta("1500ms"))
    f = arb_fills(s)
    assert [secs(t) for t in f["hedge_ns"]] == [4.3, 6.2]
    assert usd(f["vwap"][0] * 100) == 100.4
    assert usd(f["vwap"][1] * 100) == 100.524
    assert [usd(f[k][0]) for k in ("captured_usd", "delay_usd", "slippage_usd",
                                   "net_usd")] == [0.30, 0.50, -0.05, 0.65]  # fmt: skip
    assert [usd(f[k][1]) for k in ("captured_usd", "delay_usd", "slippage_usd",
                                   "net_usd")] == [-0.17, -0.02, -0.054, -0.344]  # fmt: skip
    # +0.65 - 0.344 = +0.306 USD; the window's last step now ends at 6.0 + 1.5 = 7.5 s
    assert usd(s.net.sum() * s.ref * 1e-4) == 0.306


@edge
def test_a_hedge_past_the_depth_is_marked_at_the_last_level_and_flagged():
    # the 3.0 s bids hold 30: 10 at 0.9988, 10 at 0.9987, 10 at 0.9986; the hedge of 100 takes
    # them and marks the other 70 at the last level's 0.9986: 9.988 + 9.987 + 9.986 + 69.902
    # = 99.863, vwap 0.99863, flagged unreachable, 90 units beyond level 0
    thin = dict(B_DEPTH)
    thin[3.0] = ([(0.9988, 10), (0.9987, 10), (0.9986, 10)], thin[3.0][1])
    f = arb_fills(run("RA", depth=depth_of(thin)))
    assert f["unreachable"].tolist() == [True, False]
    assert usd(f["vwap"][0] * 100) == 99.863
    assert f["beyond"].tolist() == [90.0, 40.0]


@known
def test_a_trade_through_fills_no_more_than_its_print():
    # the 2.3 s sale prints 60, not 100: capped at the print, the bid fills 60 at 0.9965 and
    # its other 40 rests until the 2 s decision's cancel lands at 2.5 s; uncapped, the default
    # rule, it fills all 100
    trades = [(2.3, -1, 60, 0.9960), (4.2, 1, 100, 1.0035)]
    capped = run("RA", panel=hand_panel(a_trades=trades))
    full = run("RA", panel=hand_panel(a_trades=trades), through="full")
    assert capped.book.fill_qty.tolist() == [60.0, 100.0]
    assert full.book.fill_qty.tolist() == [100.0, 100.0]
    # each fill's hedge is its own quantity: 60 sold into the 3.0 s bids, 40 at 0.9988 =
    # 39.952 and 20 at 0.9987 = 19.974, 59.926 in all
    assert usd(arb_fills(capped)["vwap"][0] * 60) == 59.926


@known
def test_a_bid_is_withdrawn_when_its_hedge_cannot_be_funded_and_back_once_a_transfer_lands():
    # B holds 150 coin, A 1000; the size is 100. Without the 4.2 s buy and the 4.3 s row
    # (whose bid of 1.0030 would cross the resting ask at 1.0028) only the bid fills.
    # 1 s: B's 150 covers one fill: both sides rest. 2 s: the bid sent at 1 s may have filled
    # unknown to the decision (it did, at 2.3 s), so its replacement would need 200: the bid
    # is withdrawn. 3 s: the fill is known and its hedge sent, B holds 50: withdrawn; A holds
    # 1100 of 1150 coin, over 0.9 of the two, so (1100 - 50) / 2 = 525 leave A, arriving
    # 1.5 s later, at 4.5 s, less the 2 coin fee. 4 s: B still holds 50: withdrawn. 5 s:
    # B holds 50 + 523 = 573: the bid is back.
    stock = Stock(coin={"A": 1000.0, "B": 150.0}, usdt={"A": 1000.0, "B": 1000.0},
                  rebalance_at=0.9, transfer=pd.Timedelta("1500ms"),
                  withdrawal_fee_coin=2.0)  # fmt: skip
    s = run("RA", panel=hand_panel(a_trades=A_TRADES[:1], a_book=A_BOOK[:1]), stock=stock)
    assert (s.withdrawn & 1).tolist() == [0, 1, 1, 1, 0]
    assert (s.withdrawn & 2).tolist() == [0, 0, 0, 0, 0]
    assert round(s.metrics()["stock_withdrawn_share"], 6) == 0.6
    bids = s.book.order_side > 0
    assert [secs(t) for t in s.book.order_sent[bids]] == [1.0, 5.0]
    (x,) = s.transfers
    assert (secs(x["sent_ns"]), secs(x["arrives_ns"]), x["currency"], x["amount"]) == (
        3.0, 4.5, "coin", 525.0)  # fmt: skip
    rows = stock_path(s)["rows"]
    assert [(secs(r[0]), r[1], round(r[2], 6), round(r[4], 6)) for r in rows] == [
        (2.3, "fill", 1100.0, 150.0), (3.0, "sent", 575.0, 150.0),
        (3.3, "hedge", 575.0, 50.0), (4.5, "arrived", 575.0, 573.0)]  # fmt: skip


@known
def test_a_rebalance_pays_its_withdrawal_fee_at_bs_mid():
    # the transfer of the test above, 2 coin at B's mid as-of 3.0 s, 0.9993: 1.9986 USD, in
    # bps of one unit at 1.0000 19986, in the step that holds 3.0 s, (2.5 s, 3.5 s]
    stock = Stock(coin={"A": 1000.0, "B": 150.0}, usdt={"A": 1000.0, "B": 1000.0},
                  rebalance_at=0.9, transfer=pd.Timedelta("1500ms"),
                  withdrawal_fee_coin=2.0)  # fmt: skip
    s = run("RA", panel=hand_panel(a_trades=A_TRADES[:1], a_book=A_BOOK[:1]), stock=stock)
    assert [round(float(x), 6) for x in s.transfer_fee] == [0.0, 19986.0, 0.0, 0.0, 0.0]
    m = s.metrics()
    assert (m["rebalances"], round(m["rebalance_fee_bps"], 6)) == (1, 19986.0)
    # the fee rows keep it: at no hedge fee the net is the gross less the transfer
    free = s.refee(0.0, 0.0)
    assert round(float(free.net.sum() - free.gross.sum()), 6) == -19986.0


def _random_day(seed: int):
    """Thirty minutes of a thin venue A quoted around a deep venue B that walks: A's book
    moves every few seconds, prints come about every 5 s through or at A's touch, and B's
    depth shows ten levels a tick apart at every row of its touch."""
    rng = np.random.default_rng(seed)
    secs_ = np.sort(rng.uniform(0, 1800, 2400))
    mid = 1.0 + np.cumsum(rng.normal(0, 2e-4, len(secs_)))
    half = np.round(rng.uniform(3, 8, len(secs_))) * TICK
    b_book = [(float(t), round(m - h, 4), 200.0, round(m + h, 4), 200.0)
              for t, m, h in zip(secs_, mid, half, strict=True)]  # fmt: skip
    depth = {
        r[0]: (
            [(round(r[1] - i * TICK, 4), 30.0) for i in range(10)],
            [(round(r[3] + i * TICK, 4), 30.0) for i in range(10)],
        )
        for r in b_book
    }
    a_secs = np.sort(rng.uniform(0, 1800, 400))
    a_mid = np.interp(a_secs, secs_, mid) + rng.normal(0, 3e-4, len(a_secs))
    a_book = [(float(t), round(m - 0.006, 4), 50.0, round(m + 0.006, 4), 50.0)
              for t, m in zip(a_secs, a_mid, strict=True)]  # fmt: skip
    t_secs = np.sort(rng.uniform(1, 1800, 360))
    side = rng.choice([-1, 1], len(t_secs))
    t_mid = np.interp(t_secs, secs_, mid)
    px = np.round((t_mid + side * rng.uniform(0, 0.007, len(t_secs))) / TICK) * TICK
    trades = [(float(t), int(sd), float(q), round(float(p), 4)) for t, sd, q, p in
              zip(t_secs, side, rng.uniform(5, 120, len(t_secs)), px, strict=True)]  # fmt: skip
    grid = at(*np.arange(60, 1790, 0.5))
    panel = Panel.from_frames(grid, book={"A": book_frame(a_book), "B": book_frame(b_book)},
                              trades={"A": trade_frame(trades)})  # fmt: skip
    return panel, depth_of(depth)


@known
@pytest.mark.parametrize("model", ["RA", "PR", "T", "F"])
def test_the_gross_telescopes_to_the_cash_of_both_legs(model):
    panel, depth = _random_day(7)
    win = Window("w", pd.Timestamp(T0 + timedelta(seconds=60)),
                 pd.Timestamp(T0 + timedelta(seconds=1790)))  # fmt: skip
    q = values(policy(bid=20, ask=20, size=40), panel)
    s = simulate_arb(q, panel, win, "A", "B", BIG, model, tick=TICK, latency=HALF,
                     hedge_latency=pd.Timedelta("200ms"), maker_fee_bps=1.0,
                     hedge_fee_bps=10.0, depth=depth)  # fmt: skip
    f = arb_fills(s)
    assert len(f["side"]) > 20
    # every hedge lands inside the window's steps and the two venues end flat in coin, so the
    # gross is the cash of both legs, in bps of one unit at P_ref
    cash = float(np.sum(-f["side"] * f["qty"] * f["price"] + f["side"] * f["qty"] * f["vwap"]))
    assert abs(float(s.gross.sum()) - cash / s.ref * 1e4) < 1e-6
    assert abs(float(s.position[-1])) < 1e-9
    # and the decomposition adds up fill by fill: captured + delay + slippage is the cash
    parts = f["captured_usd"] + f["delay_usd"] + f["slippage_usd"]
    assert abs(float(parts.sum()) - cash) < 1e-9
    # the fees: 1 bps on the maker units, 10 on the hedged units, at P_ref
    assert abs(float(s.fee_paid.sum()) - float(f["qty"].sum()) * 11.0) < 1e-6


@known
def test_the_arbitrage_rows_and_the_fee_rows_are_arithmetic():
    from blinded import brief as briefs
    from blinded.sim_maker import HEDGE_FEE_ROWS, arb_report

    raw = {"data": {"path": "x.parquet", "instrument_column": "symbol", "streams": {}},
           "target": {"instrument": "A", "horizons": ["1s"]},
           "execution": {"latency_ms": 500, "hedge_latency_ms": 500, "hedge_fee_bps": 10,
                         "tick_size": TICK, "lot_size": 1},
           "splits": {"embargo": "2s"}, "stats": {"block": "1s"},
           "stock": {"A": {"coin": 1000, "usdt": 1000}, "B": {"coin": 1000, "usdt": 1000},
                     "rebalance_at": 1.0, "withdrawal_fee_coin": 0}}  # fmt: skip
    brief = briefs.parse(raw)
    panel = hand_panel()
    rows = arb_report(values(policy(), panel), panel, WINDOW, brief, "A", "B",
                      depth=depth_of(B_DEPTH))  # fmt: skip
    by = {r["row"]: r for r in rows.iter_rows(named=True)}
    assert list(by) == ["brief", "RA", "PR", "T", "F", "RA uncapped", "latency 0", "hedge 1 s",
                        *(n for n, _ in HEDGE_FEE_ROWS)]  # fmt: skip
    # the two prints equal the size, so the cap changes nothing here
    assert by["RA uncapped"]["net_bps_per_hour"] == by["RA"]["net_bps_per_hour"]
    hours = by["brief"]["hours"]
    gross = by["brief"]["gross_bps_per_hour"] * hours  # -210
    for name, fee in HEDGE_FEE_ROWS:
        # fees change no fill: the net is the gross less the fee on the 200 hedged units
        assert abs(by[name]["net_bps_per_hour"] * hours - (gross - 200 * fee)) < 1e-9
    assert round(by["hedge fee 10"]["net_bps_per_hour"] * hours, 6) == -2210.0
    # the hedge 1 s row: each hedge 0.5 + 0.5 s after its fill, as the brief's, so the same
    assert by["hedge 1 s"]["net_bps_per_hour"] == by["brief"]["net_bps_per_hour"]


# -- the archetype: its refusals and its binding -----------------------------------


def _arb_raw():
    """An arbitrage brief as `blinded demo arb` writes it, files not read: the quote venue A
    with its touch file, the hedge venue B with its depth file, the pair declared."""
    return {
        "name": "arb",
        "data": {"path": "x.parquet", "instrument_column": "symbol",
                 "streams": {"book": {"filter": "kind == 'book'",
                                      "fields": {"bid": "bid", "ask": "ask", "bid_qty": "bq",
                                                 "ask_qty": "aq"}},
                             "trades": {"filter": "kind == 'trades'",
                                        "fields": {"price": "price", "qty": "qty",
                                                   "side": "side"}}},
                 "touch": {"path": "t.parquet", "tick_size": 1e-5, "instrument": "A"},
                 "depth": {"path": "d.parquet", "venue": "B", "levels": 25, "lag_ms": 100}},
        "target": {"instrument": "A", "horizons": ["1s", "10s"], "primary_horizon": "10s"},
        "execution": {"latency_ms": 50, "hedge_latency_ms": 50, "hedge_fee_bps": 10,
                      "tick_size": 1e-5, "lot_size": 1},
        "archetypes": {"instruments": ["A", "B"], "cross_venue": [["A", "B"]]},
        "stock": {"A": {"coin": 3225, "usdt": 500}, "B": {"coin": 3225, "usdt": 500},
                  "rebalance_at": 0.8, "transfer_minutes": 30, "withdrawal_fee_coin": 0},
    }  # fmt: skip


@known
@pytest.mark.parametrize(
    "change, line",
    [
        (
            lambda r: r["data"].pop("depth"),
            "needs the hedge venue's depth at every change (a depth file); the file has none",
        ),
        (
            lambda r: r["stock"].update(withdrawal_fee_coin=None),
            "stock.withdrawal_fee_coin is unset: look it up on both venues' withdrawal pages "
            "for A and its network",
        ),
        (
            lambda r: r["archetypes"].update(cross_venue=[]),
            "needs the quote and hedge instruments declared as one product under "
            "archetypes.cross_venue; the depth file's venue B is under none",
        ),
        (
            lambda r: r.pop("stock"),
            "needs the stock block: each venue's coin and USDT, for A and B",
        ),
    ],
)
def test_the_arbitrage_is_refused_with_the_line_that_says_why(change, line):
    from blinded import archetypes
    from blinded import brief as briefs
    from blinded.contract import ContractError, bind

    raw = _arb_raw()
    assert archetypes.arb_lack(briefs.parse(raw)) is None
    change(raw)
    brief = briefs.parse(raw)
    assert archetypes.arb_lack(brief)[1] == line
    menu = archetypes.Menu("arb", {"cross_venue_making": archetypes.check(
        archetypes.REGISTRY["cross_venue_making"], _facts(brief), brief)}, (), {})  # fmt: skip
    assert menu.line("cross_venue_making") == (
        f"cross_venue_making (liquidity & flow): not runnable on arb: {line}."
    )
    with pytest.raises(ContractError, match="not runnable on arb"):
        bind(bl.cross_venue_making(quote="A", hedge="B")(lambda p, f: None), brief, menu)


def _facts(brief):
    from blinded import archetypes
    from blinded import brief as briefs

    wins = briefs.windows(brief, pd.Timestamp("2026-10-01", tz="UTC"),
                          pd.Timestamp("2026-10-02", tz="UTC"))  # fmt: skip
    return archetypes.facts(brief, windows=wins)


@known
def test_the_binding_names_the_hedge_and_the_maker_on_the_hedge_venue_is_refused():
    from blinded import archetypes
    from blinded import brief as briefs
    from blinded.contract import ContractError, bind

    brief = briefs.parse(_arb_raw())
    menu = archetypes.menu(brief, _facts(brief))
    assert menu.verdicts["cross_venue_making"].runs
    bound = bind(bl.cross_venue_making(quote="A", hedge="B")(lambda p, f: None), brief, menu)
    assert (bound.legs, bound.hedge, bound.template) == (("A",), "B", "arb")
    with pytest.raises(ContractError, match="quotes A and hedges on B, the venue of its depth"):
        bind(bl.cross_venue_making(quote="B", hedge="A")(lambda p, f: None), brief, menu)
    # market making binds to the touch file's instrument only: B's fills belong to the hedge
    with pytest.raises(ContractError, match="B is the hedge venue of an arbitrage"):
        bind(bl.market_making(instrument="B")(lambda p, f: None), brief, menu)
    assert bind(bl.market_making(instrument="A")(lambda p, f: None), brief, menu).hedge is None


@known
def test_the_death_checks_on_the_hand_example():
    from blinded import dies

    s = run("RA")
    d14 = dies.picked_off(s)
    # of the two fills the second is picked off, captured -0.17 on 100.28 at B's 1.0045 mid:
    # -0.17 / 100.45 = -16.92 bps of the fill
    assert (d14["fills"], d14["picked_off_share"], d14["touch_through_share"]) == (2, 0.5, 0.5)
    assert round(d14["picked_captured_bps"], 2) == -16.92
    slow = run("RA", hedge_latency=pd.Timedelta("1500ms"))
    d15 = dies.hedge(s, slow, {"net_bps_per_hour": 1.0}, {"net_bps_per_hour": 2.0})
    # the delays -0.02 / 99.95 and -0.02 / 100.45 of each fill, -2.00 and -1.99 bps; the
    # slippages -0.057 / 99.95 = -5.70 and -0.054 / 100.45 = -5.38
    assert round(d15["delay_bps"], 4) == round((-0.02 / 99.95 - 0.02 / 100.45) / 2 * 1e4, 4)
    assert round(d15["slippage_bps"], 4) == round((-0.057 / 99.95 - 0.054 / 100.45) / 2 * 1e4,
                                                  4)  # fmt: skip
    assert d15["unreachable_share"] == 0.0
    # B's bid 0.9990 at 2.3 s is 25 bps over the bought 0.9965, over the 10 bps of fees: a
    # visible cross, still there at the hedge's walk (0.99873 over 0.9965); the sold 1.0028 is
    # under B's 1.0050 ask, no cross
    assert (d15["visible_share"], d15["reached_share"]) == (0.5, 1.0)
    d16 = dies.stock(s)
    # one buy of 99.65 USD and one sale of 100.28 over the window's 5 s: -0.63 USD, or
    # -0.63 x 86400 / 5 = -10886.4 USD a day onto the quote venue
    assert d16["buy_share"] == 0.5
    assert round(d16["drift_usd_a_day"], 6) == round(-0.63 * 86400 / 5, 6)
    assert (d16["rebalances"], d16["withdrawn_share"]) == (0, 0.0)
    d17 = dies.counterparty(s, None, s.start_ns, s.start_ns + 5 * S)
    # A's two prints, 99.6 and 100.35 USD, in the 5 s from 1.5 s: 34,560 prints a day
    assert round(d17["prints_a_day"], 6) == 34560.0 and d17["line"].startswith("the fills are")


@known
def test_the_score_is_the_worse_of_the_two_queue_models_on_a_planted_window():
    from blinded.sim_maker import held, worse

    # the bid joins A's touch at 0.9965 behind 100 displayed (the 1.0 s quote off B's 0.0 s
    # bid of 0.9995); 50 join behind it at 1.6 s and 60 cancel at 1.8 s with no trade. RA takes
    # the cancel as behind it (ahead stays min(100, 90) = 90); PR takes 60 x 100 / 150 = 40
    # from ahead (60 left). The 1.9 s sale of 70 at 0.9965 then fills PR for 10 and RA for
    # nothing, so the two models' nets, and Sharpes, differ: the score is the lower.
    book = [(0.0, 0.9965, 100, 1.0100, 400), (1.6, 0.9965, 150, 1.0100, 400),
            (1.8, 0.9965, 90, 1.0100, 400)]  # fmt: skip
    trades = [(1.9, -1, 70, 0.9965), (4.2, 1, 100, 1.0035)]
    panel = hand_panel(a_trades=trades, a_book=book)
    ra, pr = run("RA", panel=panel), run("PR", panel=panel)
    assert ra.book.fill_qty.tolist() != pr.book.fill_qty.tolist()
    assert pr.book.fill_qty[0] == 10.0 and pr.book.fill_ns[0] - ra.start_ns == int(0.4 * S)
    a, b = held(ra)["sharpe"], held(pr)["sharpe"]
    assert a != b
    assert worse(ra, pr).model == ("RA" if a < b else "PR")


@known
@pytest.mark.parametrize("rules, fee", [(2, 2.0), (3, 3.0)])
def test_a_usdt_transfer_pays_its_own_fee_under_rules_3(rules, fee):
    # A holds 5000 USDT and B 100, over 0.9 of the two: at the first decision, 1 s, half the
    # difference, 2450 USDT, leaves A. Under rules 3 it pays withdrawal_fee_usdt, 3 USDT;
    # under rules 1 and 2 the coin's fee, 2 coin, at B's mid then, 1.0000: 2 USDT
    stock = Stock(coin={"A": 1000.0, "B": 1000.0}, usdt={"A": 5000.0, "B": 100.0},
                  rebalance_at=0.9, transfer=pd.Timedelta("1500ms"),
                  withdrawal_fee_coin=2.0, withdrawal_fee_usdt=3.0)  # fmt: skip
    s = run("RA", stock=stock, rules=rules)
    x = next(x for x in s.transfers if x["currency"] == "usdt")
    assert (secs(x["sent_ns"]), x["amount"], usd(x["fee_usd"])) == (1.0, 2450.0, fee)
    arrived = [r for r in stock_path(s)["rows"] if r[1] == "arrived"][0]
    assert usd(arrived[5]) == usd(100.0 + 2450.0 - fee)
    # charged to the net as bps of one unit at P_ref, in the step that holds 1 s
    assert usd(s.transfer_fee.sum()) == usd(fee / s.ref * 1e4)


@known
@pytest.mark.parametrize("unset", ["withdrawal_fee_coin", "withdrawal_fee_usdt"])
def test_under_rules_3_both_withdrawal_fees_must_be_set(unset):
    stock = Stock(coin={"A": 1000.0, "B": 1000.0}, usdt={"A": 1000.0, "B": 1000.0},
                  rebalance_at=1.0, withdrawal_fee_coin=0.0, withdrawal_fee_usdt=0.0)  # fmt: skip
    stock = dataclasses.replace(stock, **{unset: None})
    with pytest.raises(ValueError) as err:
        run("RA", stock=stock, rules=3)
    words = str(err.value)
    assert "withdrawal_fee_coin and stock.withdrawal_fee_usdt must both be set" in words
    assert f"{unset} is unset" in words
    # under rules 2 only the coin's fee is needed, as before
    if unset == "withdrawal_fee_usdt":
        assert run("RA", stock=stock, rules=2).fills.sum() > 0


@known
def test_under_rules_3_a_hedge_past_the_depth_pays_a_further_spread():
    # a sale of 1000 into B's bids at 0.5 s: 300 each at 0.9995, 0.9994 and 0.9993, and 100 the
    # file does not show. Under rules 2 those 100 were sold at the last level, 0.9993; under
    # rules 3 a further spread below it, the row's 1.0005 less 0.9995: 0.9983. Flagged either way
    depth = depth_of(B_DEPTH)
    stamp = np.array([int(pd.Timestamp(at(0.5)[0]).value)])
    old = depth.walk(stamp, np.array([-1]), np.array([1000.0]), rules=2)
    new = depth.walk(stamp, np.array([-1]), np.array([1000.0]), rules=3)
    shown = 300 * (0.9995 + 0.9994 + 0.9993)
    assert old["vwap"][0] == pytest.approx((shown + 100 * 0.9993) / 1000)
    assert new["vwap"][0] == pytest.approx((shown + 100 * 0.9983) / 1000)
    assert old["unreachable"][0] and new["unreachable"][0]


@known
def test_under_rules_3_a_hedge_the_depth_cannot_price_is_never_at_the_mid():
    from blinded.sim_maker import mark_of, unpriced

    panel = hand_panel()
    mark = mark_of(panel, "B")
    stamp = np.array([int(pd.Timestamp(at(0.5)[0]).value)])
    # B's touch at 0.5 s: 0.9995 / 1.0005. A sale under rules 2 took the mid, 1.0000; under
    # rules 3 the bid less a spread, 0.9985, and a purchase the ask plus one, 1.0015
    assert unpriced(mark, stamp, np.array([-1]), 2)[0] == pytest.approx(1.0)
    assert unpriced(mark, stamp, np.array([-1]), 3)[0] == pytest.approx(0.9985)
    assert unpriced(mark, stamp, np.array([1]), 3)[0] == pytest.approx(1.0015)
    # before B shows any price there is none to take: the pass is refused, not hedged free
    early = np.array([int(pd.Timestamp(at(-1.0)[0]).value)])
    with pytest.raises(ValueError, match="cannot be priced"):
        unpriced(mark, early, np.array([-1]), 3)
