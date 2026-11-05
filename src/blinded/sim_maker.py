"""Quotes to fills to P&L: the fill engine of a market-making strategy.

A market-making strategy returns quote parameters on the grid
(:func:`quote`, a :class:`~blinded.contract.Quotes`). The engine turns them
into resting orders, one a side, and walks the window's events in stamp
order: every top-of-book row of the instrument, with its sizes, and every
trade at its own price. Book rows come before trades on a shared stamp. An
order fills when a trade goes through its price, or, under a queue model,
when the trades at its price have used up the queue ahead of it. Four
models of that queue are run, each its own closed loop:

- ``T``, trade-through only: the queue is never known, so only a trade
  strictly through the price fills;
- ``RA``, risk-averse: the order joins the back of the displayed size, the
  queue ahead shrinks by each trade at the price, and a cancel is assumed
  to be behind the order, so the queue only ever falls to the displayed
  size (hftbacktest's ``RiskAdverseQueueModel``);
- ``PR``, pro rata: as RA, but a cancel at the level comes from ahead of the
  order in proportion to the queue ahead (hftbacktest's ``ProbQueueModel``
  with the power function of exponent 1);
- ``F``, front of the queue: any trade at or through the price fills.

Under every model a book row whose far touch reaches a resting order's
price fills its remainder there with no trade (``crossed``), under T only a
book strictly through it (:func:`run_engine`).

For a fixed quote sequence the fill sets are nested, T in RA in PR in F,
and no model bounds the P&L. The judge scores the worse of RA and PR by net
Sharpe (:func:`worse`); T and F bracket the fill count.

Clock. All on the file's clock. A decision at :math:`g` knows the book
available by :math:`g` and the fills stamped by :math:`g - L_r`; an order
it sends reaches the book at :math:`g + L`, a cancel at :math:`g + L_c`.
A side whose desired price and size equal the resting order's keeps it and
its place in the queue. A new order that would cross the book at its
arrival is rejected (post-only, as Binance's GTX) and the side stays empty
until a decision that knows of the rejection. At the end of the window
every order is cancelled at :math:`g_\\text{end} + L_c` and the inventory
is crossed at the far touch as-of :math:`g_\\text{end} + L`, paying the
taker fee, with :math:`g_\\text{end}` the grid point after the window's
last.

Units. Prices are whole ticks inside the engine. Sizes and inventory are
units of the instrument (one coin), and every P&L number is in basis points
of one unit at :math:`P_\\text{ref}`, the mid as-of the first decision plus
:math:`L`, as in :mod:`blinded.sim`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from blinded import sim as taker
from blinded.brief import Brief, Window
from blinded.contract import Quotes
from blinded.durations import label
from blinded.ladder import RULES, check_rules, sharpe_blocks
from blinded.panel import Events, Panel

MODELS = ("RA", "PR", "T", "F")
QUEUE_MODELS = ("RA", "PR")
NAMES = {
    "RA": "risk-averse queue: cancels are behind the order",
    "PR": "pro-rata queue: cancels come from ahead of the order in proportion",
    "T": "trade-through only: a bracket, never scored",
    "F": "front of the queue: a bracket, never scored",
}
# what the ledger keeps of each model on validation
MODEL_KEYS = ("net_bps_per_hour", "sharpe_per_sqrt_hour", "fills", "units", "captured_bps",
              "adverse_bps_10s")  # fmt: skip
# the maker fee of each fee row, in bps a unit; "fees 0" also waives the taker fee of the
# flatten, the others keep the brief's
FEE_ROWS = (("fees 0", 0.0), ("rebate 0.5", -0.5), ("fee 1", 1.0), ("fee 2 (Binance VIP0)", 2.0))
ADVERSE = pd.Timedelta("10s")
_EPS = 1e-9
_NEVER = 2**62
_CHUNK = 1 << 18
_CODE = {"RA": 0, "PR": 1, "T": 2, "F": 3}


def quote(bid=0, ask=0, size=1.0, skew=0.0, max_inventory=1.0, lean=0, fair=None) -> Quotes:
    """Quote parameters for a market-making strategy, at every grid point.

    Each argument is a number, a polars expression or a Series on the grid.

    Parameters
    ----------
    bid, ask
        Offset of each quote from its own side of the touch, in ticks: 0
        joins the touch, 1 rests one tick behind it, -1 improves on it by a
        tick (and is rejected when that would cross the book). Null withdraws
        the side.
    size
        The resting size of each side, in units of the instrument, rounded
        down to the lot; null, 0 or under one lot withdraws both sides.
    skew
        Ticks per unit of known inventory, the declared feedback: a long
        inventory lowers both quotes, a short one raises them.
    max_inventory
        No bid while a full fill of it would take the known inventory above
        ``max_inventory``, no ask while one would take it below minus that;
        null withdraws both sides. With ``size`` 1 and ``max_inventory`` 2 the
        bid rests while long 1 and is withdrawn at long 2.
    lean
        A signed shift of both quotes, in ticks: positive raises them, so the
        bid moves in and the ask moves out.
    fair
        A price to measure the offsets from instead of the touch: the bid
        from ``floor(fair / tick)``, the ask from ``ceil(fair / tick)``. A
        pair ``(bid_fair, ask_fair)`` gives each side its own price, as an
        arbitrage quote priced off the other venue's touch does: the bid
        from ``floor(bid_fair / tick)``, the ask from ``ceil(ask_fair / tick)``.

    The sim applies the inventory, which never enters the strategy: with
    :math:`q` the known inventory, the bid sits
    :math:`\\lceil \\text{bid} + \\text{skew}\\,q - \\text{lean} \\rceil` ticks
    below the touch's bid and the ask
    :math:`\\lceil \\text{ask} - \\text{skew}\\,q + \\text{lean} \\rceil` above
    the touch's ask, a fraction of a tick rounding away from the touch.

    Examples
    --------
    >>> import blinded as bl
    >>> q = bl.quote(bid=0, ask=0, size=1.0, skew=1.0, max_inventory=2.0)
    >>> sorted(q)
    ['ask', 'bid', 'lean', 'max_inventory', 'size', 'skew']
    >>> "fair" in bl.quote(fair=99.995)
    True
    >>> sorted(k for k in bl.quote(fair=(99.99, 100.01)) if k.startswith("fair"))
    ['fair_ask', 'fair_bid']
    """
    out = Quotes(bid=bid, ask=ask, size=size, skew=skew, max_inventory=max_inventory, lean=lean)
    if isinstance(fair, tuple | list):
        if len(fair) != 2:
            raise ValueError("fair is one price, or a pair (bid_fair, ask_fair)")
        out["fair_bid"], out["fair_ask"] = fair
    elif fair is not None:
        out["fair"] = fair
    return out


# -- the touch stream -------------------------------------------------------


@dataclass(frozen=True)
class Touch:
    """What the fill engine reads of one instrument: every top-of-book row with its sizes and
    every trade at its own price, prices in whole ticks.

    ``book_ns`` and ``trade_ns`` are the rows' stamps on the file's clock, in
    file order on a tie; ``side`` is the aggressor's, +1 a buy that lifts asks
    and -1 a sell that hits bids. ``delay_ns`` is how long after its stamp a
    book row is available to a decision.
    """

    tick: float
    book_ns: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_qty: np.ndarray
    ask_qty: np.ndarray
    trade_ns: np.ndarray
    price: np.ndarray
    qty: np.ndarray
    side: np.ndarray
    delay_ns: int = 0

    def row(self, stamps: np.ndarray) -> np.ndarray:
        """The index of the last book row stamped at or before each stamp; -1 before any."""
        return np.searchsorted(self.book_ns, np.asarray(stamps), side="right") - 1

    def mid(self, stamps: np.ndarray) -> np.ndarray:
        """The mid price as-of each stamp, NaN before the first row."""
        j = self.row(stamps)
        out = np.full(len(j), np.nan)
        ok = j >= 0
        out[ok] = (self.bid[j[ok]] + self.ask[j[ok]]) * (self.tick / 2)
        return out


def infer_tick(*prices: np.ndarray) -> float:
    """The smallest step between two distinct prices, to six significant figures.

    For a file whose brief gives no ``execution.tick_size``: over hours of a
    book that is one or two ticks wide, both sides visit neighbouring prices.

    Examples
    --------
    >>> infer_tick(np.array([99.99, 100.0, 100.01]), np.array([100.02]))
    0.01
    >>> infer_tick(np.array([1.4, 1.4000000000000001, 1.4001]))  # 14000 ticks, twice
    0.0001
    """
    p = np.unique(np.concatenate([np.asarray(x, dtype=float) for x in prices]))
    p = p[np.isfinite(p)]
    gaps = np.diff(p)
    # a price written as a multiple of the tick in floats can land a few ulps either side
    gaps = gaps[gaps > 1e-9 * np.abs(p).max()] if len(p) else gaps
    if not len(gaps):
        raise ValueError("the book shows a single price, so no tick can be read from it")
    return float(f"{gaps.min():.6g}")


def to_ticks(price: np.ndarray, tick: float, what: str = "price") -> np.ndarray:
    """Prices as whole ticks, refusing one that is not.

    Examples
    --------
    >>> to_ticks(np.array([99.99, 100.0]), 0.01).tolist()
    [9999, 10000]
    """
    x = np.asarray(price, dtype=float) / tick
    r = np.rint(x)
    bad = np.abs(x - r) > 1e-6
    if bad.any():
        raise ValueError(f"a {what} of {np.asarray(price)[bad][0]!r} is not a whole number of "
                         f"ticks of {tick:g}")  # fmt: skip
    return r.astype(np.int64)


@dataclass(frozen=True)
class TouchSource:
    """Where a declared touch stream's rows are: the touch files of one instrument, in stamp
    order across the list (a run's segments up to the one it reads), and their tick.

    A panel built from a brief with ``data.touch`` carries one
    (:attr:`blinded.panel.Panel.touch_source`), and :func:`touch_of` reads it in
    place of the panel's own book rows, which on a thinned file are only the
    lattice's.
    """

    paths: tuple[Path, ...]
    instrument: str
    tick: float

    def rows(self, start_ns: int | None = None, end_ns: int | None = None) -> pl.DataFrame:
        """The rows stamped before ``end_ns``, from the last one at or before ``start_ns``.

        A read as-of any stamp from ``start_ns`` to ``end_ns`` sees what it
        would see in the whole stream. The files are scanned, so only the rows
        kept are held: an hour before ``start_ns`` first, the whole prefix only
        when that hour has no row at or before it.
        """
        lf = pl.scan_parquet([str(p) for p in self.paths])
        at = pl.col("ts").cast(pl.Datetime("ns")).dt.epoch("ns")
        if end_ns is not None:
            lf = lf.filter(at < end_ns)
        if start_ns is None:
            return lf.collect()
        head = lf.filter(at >= start_ns - _HOUR).collect()
        if not head.height or _ns(head["ts"])[0] > start_ns:
            head = lf.collect()
        i = int(np.searchsorted(_ns(head["ts"]), start_ns, side="right")) - 1
        return head.slice(max(i, 0))


_HOUR = 3_600_000_000_000
# what the engine and its accounts read after a window's last decision: the cancel and the
# flatten after the latencies, and the markouts of the last fills at up to a minute
_AFTER = 300_000_000_000


def _ns(ts: pl.Series) -> np.ndarray:
    return ts.dt.epoch("ns").to_numpy() if ts.dtype.is_temporal() else ts.to_numpy()


def touch_of(panel: Panel, instrument: str, tick: float | None = None,
             window: Window | None = None) -> Touch:  # fmt: skip
    """The touch stream of ``instrument``: the panel's declared stream, or its own book and
    trade rows.

    A panel whose brief declares a touch file for ``instrument``
    (:class:`TouchSource`) reads the file's rows, with ``window`` its rows
    from the last at or before the window's first decision to five minutes
    after its end, which is everything a pass on that window reads, and its
    tick. Otherwise the stream is the panel's own rows, for a file that keeps
    every book row with its sizes and every trade at its own stamp, as the
    Tardis layout does: rows with a missing or crossed price are left out, and
    ``tick`` defaults to :func:`infer_tick` on the book.
    """
    source = getattr(panel, "touch_source", None)
    if source is not None and source.instrument == instrument:
        lo = hi = None
        if window is not None:
            delay = panel.events("trades", instrument).delay_ns
            lo = int(window.start.value) - delay - panel.step_ns
            hi = int(window.end.value) + panel.step_ns + _AFTER
        return touch_from(source.rows(lo, hi), panel, instrument, source.tick)
    book, trades = panel.events("book", instrument), panel.events("trades", instrument)
    ts = book.frame["ts"].dt.epoch("ns").to_numpy()
    bid, ask = book.column("bid"), book.column("ask")
    bq, aq = book.column("bid_qty"), book.column("ask_qty")
    ok = np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (bid > 0)
    ok &= np.isfinite(bq) & np.isfinite(aq)
    tick = infer_tick(bid[ok], ask[ok]) if tick is None else tick
    tt = trades.frame["ts"].dt.epoch("ns").to_numpy()
    px, qty, side = trades.column("price"), trades.column("qty"), trades.column("side")
    good = np.isfinite(px) & np.isfinite(qty) & (qty > 0) & np.isin(side, (-1.0, 1.0))
    return Touch(
        tick=float(tick),
        book_ns=ts[ok],
        bid=to_ticks(bid[ok], tick),
        ask=to_ticks(ask[ok], tick),
        bid_qty=bq[ok],
        ask_qty=aq[ok],
        trade_ns=tt[good],
        price=to_ticks(px[good], tick, "trade price"),
        qty=qty[good],
        side=side[good].astype(np.int64),
        delay_ns=book.delay_ns,
    )


def touch_from(rows: pl.DataFrame, panel: Panel, instrument: str, tick: float) -> Touch:
    """The touch stream of a file that declares one (``data.touch``): its book rows from
    ``rows``, the touch file's layout (``ts``, ``bid`` and ``ask`` in ticks, ``bid_qty``,
    ``ask_qty``, in stamp order), and its trades from the panel's ``trades`` stream, whose
    availability delay the book shares."""
    trades = panel.events("trades", instrument)
    tt = trades.frame["ts"].dt.epoch("ns").to_numpy()
    px, qty, side = trades.column("price"), trades.column("qty"), trades.column("side")
    good = np.isfinite(px) & np.isfinite(qty) & (qty > 0) & np.isin(side, (-1.0, 1.0))
    ts = rows["ts"]
    ns = ts.dt.epoch("ns").to_numpy() if ts.dtype.is_temporal() else ts.to_numpy()
    return Touch(
        tick=float(tick),
        book_ns=ns.astype(np.int64),
        bid=rows["bid"].to_numpy().astype(np.int64),
        ask=rows["ask"].to_numpy().astype(np.int64),
        bid_qty=rows["bid_qty"].to_numpy().astype(float),
        ask_qty=rows["ask_qty"].to_numpy().astype(float),
        trade_ns=tt[good],
        price=to_ticks(px[good], tick, "trade price"),
        qty=qty[good],
        side=side[good].astype(np.int64),
        delay_ns=trades.delay_ns,
    )


# -- the engine -------------------------------------------------------------


@dataclass
class Book:
    """What one engine pass did: the fills, the orders and the counts.

    Fills are in stamp order: ``fill_ns``, ``fill_side`` (+1 bought, -1
    sold), ``fill_price`` in ticks, ``fill_qty``, ``fill_through`` (a trade
    went through the price), ``fill_order``, ``fill_taker`` (the order's
    remainder crossed at its deadline, in maker-taker execution; False for
    every fill of a market-making strategy) and ``fill_crossed`` (a book row
    put the far touch at or through the resting order's price: the whole
    remainder filled at its price, a maker fill with no trade). Orders hold, by id,
    ``order_side``, ``order_price``, ``order_size``, ``order_sent``,
    ``order_arrival``, ``order_ahead`` (the queue ahead at arrival, NaN when
    unknown), ``order_end`` and ``order_status`` (``filled``, ``cancelled``,
    ``rejected``, ``dropped`` for one cancelled before it arrived, ``crossed``
    for one whose remainder was crossed at its deadline, ``open`` for one
    resting at the end, or ``sent`` for one still on its way).
    ``flatten`` is the end-of-window crossing, ``(stamp, side, price in ticks,
    qty)``, or None when the inventory was flat.
    ``quoted`` holds, at each decision, which sides had an order resting just
    before it: 1 the bid, 2 the ask, 3 both.
    """

    fill_ns: np.ndarray
    fill_side: np.ndarray
    fill_price: np.ndarray
    fill_qty: np.ndarray
    fill_through: np.ndarray
    fill_order: np.ndarray
    order_side: np.ndarray
    order_price: np.ndarray
    order_size: np.ndarray
    order_sent: np.ndarray
    order_arrival: np.ndarray
    order_ahead: np.ndarray
    order_end: np.ndarray
    order_status: np.ndarray
    flatten: tuple | None
    rejected: int
    requotes: int
    quoted: np.ndarray
    fill_taker: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    fill_crossed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


@dataclass(frozen=True)
class Decisions:
    """The quote parameters at each decision of a window, with the inventory left out.

    ``ref_bid`` and ``ref_ask`` are the prices the offsets count from, in
    ticks (the touch the decision knows, or ``fair``), and ``base_bid`` and
    ``base_ask`` the offsets at zero inventory, ``bid - lean`` and ``ask +
    lean``; a side is off where any of these is NaN. ``size`` is rounded to
    the lot, 0 for no quotes.
    """

    ns: np.ndarray
    ref_bid: np.ndarray
    ref_ask: np.ndarray
    base_bid: np.ndarray
    base_ask: np.ndarray
    skew: np.ndarray
    cap: np.ndarray
    size: np.ndarray

    @classmethod
    def make(cls, quotes: Quotes, grid_ns: np.ndarray, touch: Touch, lot: float | None
             ) -> Decisions:  # fmt: skip
        """The decisions at ``grid_ns`` from quote arrays on the same points."""
        n = len(grid_ns)

        def key(name: str, default: float) -> np.ndarray:
            v = quotes.get(name)
            return np.full(n, default) if v is None else np.asarray(v, dtype=float)

        known = touch.row(grid_ns - touch.delay_ns)
        ref_bid = np.where(known >= 0, touch.bid[np.maximum(known, 0)], np.nan).astype(float)
        ref_ask = np.where(known >= 0, touch.ask[np.maximum(known, 0)], np.nan).astype(float)
        fair = key("fair", np.nan) / touch.tick
        own = np.isfinite(fair)
        ref_bid[own] = np.floor(fair[own] + _EPS)
        ref_ask[own] = np.ceil(fair[own] - _EPS)
        # a two-sided fair, each side its own price: an arbitrage quote off another venue's touch
        if "fair_bid" in quotes or "fair_ask" in quotes:
            fb, fa = key("fair_bid", np.nan) / touch.tick, key("fair_ask", np.nan) / touch.tick
            ref_bid = np.where(np.isfinite(fb), np.floor(fb + _EPS), np.nan)
            ref_ask = np.where(np.isfinite(fa), np.ceil(fa - _EPS), np.nan)
        lean = np.nan_to_num(key("lean", 0.0))
        size = key("size", np.nan)
        if lot is not None:
            size = np.floor(size / lot + _EPS) * lot
        size = np.where(np.isfinite(size) & (size > 0), size, 0.0)
        return cls(grid_ns, ref_bid, ref_ask, key("bid", np.nan) - lean,
                   key("ask", np.nan) + lean, np.nan_to_num(key("skew", 0.0)),
                   key("max_inventory", np.nan), size)  # fmt: skip


def run_engine(
    touch: Touch,
    decisions: Decisions,
    end_ns: int,
    model: str = "RA",
    latency_ns: int = 0,
    cancel_ns: int | None = None,
    response_ns: int | None = None,
    trace: list | None = None,
    targets: np.ndarray | None = None,
    deadline_ns: int | None = None,
    rules: int = RULES,
    through: str = "full",
    gate=None,
) -> Book:
    """One closed-loop pass of the fill engine under ``model``.

    ``decisions`` are made at their stamps, in order, and ``end_ns`` is the
    end decision, the grid point after the last, which cancels everything at
    ``end_ns + cancel_ns`` and crosses the inventory at ``end_ns +
    latency_ns``. The cancel and response latencies default to the order
    latency; a cancel may not be slower than the order.

    With ``targets``, the inventory to hold after each decision, the pass is
    maker-taker execution of a position strategy instead
    (:func:`simulate_maker_taker`): a decision whose target differs from the
    one its live order was sent for cancels that order and sends one for the
    difference between the target and the known inventory, at its side's
    price in ``decisions`` (the touch as-of the arrival); an order rests until
    it fills or until ``deadline_ns`` after its arrival, when its remainder is
    crossed at the far touch as-of that stamp, a taker fill; the end decision
    cancels nothing, every deadline falls at ``end_ns + latency_ns`` at the
    latest, and there is no flatten: the inventory left is the last target's.
    On a stamp a deadline comes after the cancels and before the arrivals.

    The order of things on one stamp :math:`t`, under rules 1 and 2: book rows;
    cancels, then arrivals, stamped :math:`t`; trades; the decision at :math:`t`.
    Under rules 3: cancels, then arrivals; trades; book rows; the decision; with no
    trade on the stamp, book rows come before the cancels and arrivals. A
    feed stamps a trade and the book row that shows it alike, on 11 to 14% of
    the touch trade groups of the SOL and LTC files, and the row reflects the
    trade, so the trade goes first. So a trade
    sees an order that arrives on its stamp, a cancelled order sees no trade
    on the cancel's stamp, and a decision at :math:`t` sees the book row and
    the fills stamped :math:`t`.

    A book row that puts the far touch at or through a resting order's price
    (an ask at or below a live bid, a bid at or above a live ask) fills the
    order's whole remainder at its price, flagged ``crossed``: the order is
    not in the recorded book, so the seller who brought the ask down to it
    would have hit it first, and no trade printed only because no recorded
    bid was there. Under T only a book strictly through the price does, as
    only a trade strictly through it does. A row on a cancel's stamp comes
    before the cancel unless a trade shares the stamp, so it can cross the order.

    ``through`` is what a trade through a resting order's price fills:
    ``"full"`` the order's whole remainder (the default, which market making
    keeps), ``"print"`` no more than that trade's own quantity, since the
    aggressor that walked past the level had at least that much and no more
    is known (the arbitrage template's rule).

    ``gate``, when given, is consulted at every decision of a quote and may
    withdraw a side (an arbitrage strategy's stock, :class:`StockGate`): it is
    called as ``gate(k, td, lim, kf, fills, want_bid, want_ask, reserve_bid,
    reserve_ask)`` with the fills the decision knows (the first ``kf`` of
    ``fills``, the engine's lists of stamps, sides, prices and quantities),
    the prices each side wants (None for no quote) and, for each side, the
    units that could still fill without the decision knowing: the size it
    wants, plus every order of that side it sent whose end it cannot yet know
    (stamped after ``lim``). It returns whether each side may rest.

    It is a plain Python loop over the window's events: flat lists, scalar
    state, no dicts in the hot path. ``trace``, when given, receives ``(stamp,
    side, order, ahead)`` each time an order's queue ahead is set or read
    again: at its arrival, on a book row at or through its price and on a
    trade at its price; ``ahead`` is NaN while unknown.
    """
    if model not in _CODE:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    lat = int(latency_ns)
    lc = lat if cancel_ns is None else int(cancel_ns)
    lr = lat if response_ns is None else int(response_ns)
    if lc > lat:
        raise ValueError("a cancel may not take longer than the order it replaces")
    if through not in ("full", "print"):
        raise ValueError(f"through must be 'full' or 'print', got {through!r}")
    capped = through == "print"
    code = _CODE[model]
    queue, pro, front = code in (0, 1), code == 1, code == 3
    # how far the far touch must come to cross a resting order: to its price, or under T
    # strictly through it
    reach = 1 if code == 2 else 0
    # the crossing is rules 2's (blinded.ladder.RULES): under rules 1 only a trade fills
    cross = check_rules(rules) >= 2
    g = decisions.ns
    n = len(g)
    if n == 0:
        raise ValueError("no decisions in the window")

    # the book as it stood at the first decision, then every event after it up to the flatten
    first = int(g[0])
    stop = int(end_ns) + lat
    j0 = int(touch.row(np.array([first]))[0])
    if j0 >= 0:
        cb, ca = int(touch.bid[j0]), int(touch.ask[j0])
        cbq, caq = float(touch.bid_qty[j0]), float(touch.ask_qty[j0])
    else:
        cb = ca = None
        cbq = caq = 0.0
    b0, b1 = np.searchsorted(touch.book_ns, [first, stop], side="right")
    t0, t1 = np.searchsorted(touch.trade_ns, [first, stop], side="right")
    nb, nt = b1 - b0, t1 - t0
    book_ns, trade_ns = touch.book_ns[b0:b1], touch.trade_ns[t0:t1]
    # merge the two sorted streams: a row's place is its own index plus what of the other
    # stream comes before it. Under rules 3 a trade comes before the book row on its stamp, the
    # row that already shows it: read first, the row clamps the queue ahead to the size after
    # the trade and the trade then takes from it again. Under rules 1 and 2 the row came first
    trades_first = check_rules(rules) >= 3
    side_b, side_t = ("right", "left") if trades_first else ("left", "right")
    at_book = np.arange(nb) + np.searchsorted(trade_ns, book_ns, side=side_b)
    at_trade = np.arange(nt) + np.searchsorted(book_ns, trade_ns, side=side_t)

    # a book row is (bid, ask, bid size, ask size) in ticks and units, a trade (price, side,
    # qty, 0), and the kind 0 for a book row, 1 for a trade
    pairs = (
        (book_ns, trade_ns, np.int64),
        (0, 1, np.int8),
        (touch.bid[b0:b1], touch.price[t0:t1], np.int64),
        (touch.ask[b0:b1], touch.side[t0:t1], np.int64),
        (touch.bid_qty[b0:b1], touch.qty[t0:t1], float),
        (touch.ask_qty[b0:b1], 0.0, float),
    )
    columns = tuple(_merged(nb + nt, at_book, at_trade, *pair) for pair in pairs)
    del at_book, at_trade, pairs

    G = g.astype(np.int64).tolist()
    taking = targets is not None
    tg = np.asarray(targets, dtype=float).tolist() if taking else []
    dl = _NEVER if deadline_ns is None else int(deadline_ns)
    ref_b = decisions.ref_bid.tolist()
    ref_a = decisions.ref_ask.tolist()
    base_b, base_a = decisions.base_bid.tolist(), decisions.base_ask.tolist()
    skew, cap, size = decisions.skew.tolist(), decisions.cap.tolist(), decisions.size.tolist()
    ceil, isfinite = math.ceil, math.isfinite

    # orders, by id
    o_side, o_price, o_size, o_sent, o_arr = [], [], [], [], []
    o_ahead, o_end, o_status, o_dead = [], [], [], []
    # fills
    f_ns, f_side, f_px, f_qty, f_thr, f_oid, f_tak, f_crs = [], [], [], [], [], [], [], []
    # the engine's resting order on each side: id (-1 none), price, remaining, queue ahead
    # (-1 unknown) and, for PR, the displayed size and the trades at the price since the last
    # book row there
    bo = ao = -1
    bp = ap = 0
    brem = arem = 0.0
    bA = aA = -1.0
    bd = ad = bs = as_ = 0.0
    inv = 0.0
    # the strategy's view: the last order it sent on each side, -1 none
    vb = va = -1
    # maker-taker: the strategy's live order and the target it was sent for
    vo, vt = -1, math.nan
    kf, known = 0, 0.0
    cancels: deque = deque()
    arrivals: deque = deque()
    deadlines: deque = deque()
    # with a gate: the orders each side sent whose end the strategy may not know yet
    sent_b: deque = deque()
    sent_a: deque = deque()
    fills = (f_ns, f_side, f_px, f_qty)
    k = 0  # next decision; n is the end decision
    rejected = requotes = 0
    quoted = np.zeros(n, dtype=np.int8)

    def fill(t, side, oid, price, qty, through, taker=False, crossed=False):
        f_ns.append(t)
        f_side.append(side)
        f_px.append(price)
        f_qty.append(qty)
        f_thr.append(through)
        f_oid.append(oid)
        f_tak.append(taker)
        f_crs.append(crossed)

    # the events are read as Python lists a chunk at a time: lists index fastest, and a whole
    # day of them would hold gigabytes
    nev = len(columns[0])
    e = c0 = c1 = 0
    while True:
        # the next market event, or past the end
        if e < nev:
            if e >= c1:
                c0, c1 = e, min(e + _CHUNK, nev)
                T, K, F1, F2, F3, F4 = (c[c0:c1].tolist() for c in columns)
            x = e - c0
            t, kd = T[x], K[x]
        else:
            t, kd = stop + 1, 1
        # pending decisions, cancels and arrivals due before this event, in time order:
        # cancels, then arrivals, then the decision, on one stamp
        while True:
            tc = cancels[0][0] if cancels else _NEVER
            tl = deadlines[0][0] if deadlines else _NEVER
            ta = arrivals[0][0] if arrivals else _NEVER
            td = G[k] if k < n else (int(end_ns) if k == n else _NEVER)
            if tl < tc and tl <= ta and tl <= td:
                if not (tl < t or (tl == t and kd == 1)):
                    break
                _, oid = deadlines.popleft()
                if o_status[oid] != "open":
                    continue
                # the remainder crosses at the far touch as-of the deadline: a taker fill
                if oid == bo:
                    if ca is not None and brem > _EPS * o_size[oid]:
                        fill(tl, 1, oid, ca, brem, False, True)
                        inv += brem
                    bo = -1
                elif oid == ao:
                    if cb is not None and arem > _EPS * o_size[oid]:
                        fill(tl, -1, oid, cb, arem, False, True)
                        inv -= arem
                    ao = -1
                o_status[oid], o_end[oid], o_dead[oid] = "crossed", tl, tl
            elif tc <= ta and tc <= td:
                if not (tc < t or (tc == t and kd == 1)):
                    break
                _, oid = cancels.popleft()
                st = o_status[oid]
                if st == "sent":
                    o_status[oid] = "dropped"
                    o_end[oid] = tc
                elif oid == bo:
                    o_status[oid], o_end[oid] = "cancelled", tc
                    bo = -1
                elif oid == ao:
                    o_status[oid], o_end[oid] = "cancelled", tc
                    ao = -1
            elif ta <= td:
                if not (ta < t or (ta == t and kd == 1)):
                    break
                _, oid = arrivals.popleft()
                if o_status[oid] == "dropped":
                    continue
                p = o_price[oid]
                if o_side[oid] > 0:
                    if ca is not None and p >= ca:
                        o_status[oid], o_end[oid], o_dead[oid] = "rejected", ta, ta
                        rejected += 1
                        continue
                    if bo >= 0:  # cannot happen with cancels no slower than orders
                        o_status[bo], o_end[bo] = "cancelled", ta
                    bo, bp, brem, bd, bs = oid, p, o_size[oid], 0.0, 0.0
                    if front:
                        bA = 0.0
                    elif not queue or cb is None or p < cb:
                        bA = -1.0
                    elif p == cb:
                        bA = bd = cbq
                    else:
                        bA = 0.0
                    o_status[oid], o_ahead[oid] = "open", bA
                    if trace is not None:
                        trace.append((ta, 1, oid, bA if bA >= 0 else math.nan))
                    if taking:
                        deadlines.append((min(ta + dl, stop), oid))
                else:
                    if cb is not None and p <= cb:
                        o_status[oid], o_end[oid], o_dead[oid] = "rejected", ta, ta
                        rejected += 1
                        continue
                    if ao >= 0:
                        o_status[ao], o_end[ao] = "cancelled", ta
                    ao, ap, arem, ad, as_ = oid, p, o_size[oid], 0.0, 0.0
                    if front:
                        aA = 0.0
                    elif not queue or ca is None or p > ca:
                        aA = -1.0
                    elif p == ca:
                        aA = ad = caq
                    else:
                        aA = 0.0
                    o_status[oid], o_ahead[oid] = "open", aA
                    if trace is not None:
                        trace.append((ta, -1, oid, aA if aA >= 0 else math.nan))
                    if taking:
                        deadlines.append((min(ta + dl, stop), oid))
            else:
                if not td < t:
                    break
                lim = td - lr
                while kf < len(f_ns) and f_ns[kf] <= lim:
                    known += f_side[kf] * f_qty[kf]
                    kf += 1
                known = round(known, 9)
                if k == n:  # the end: cancel everything the strategy believes is out
                    for v in () if taking else (vb, va):
                        if v >= 0 and o_dead[v] > lim:
                            cancels.append((td + lc, v))
                    vb = va = -1
                    k += 1
                    continue
                quoted[k] = (bo >= 0) + 2 * (ao >= 0)
                if taking:
                    # keep a live order while the target it was sent for stands; else cancel it
                    # and send one for what the known inventory lacks of the target
                    tgt = tg[k]
                    alive = vo >= 0 and o_dead[vo] > lim
                    if alive and tgt == vt:
                        k += 1
                        continue
                    if alive:
                        cancels.append((td + lc, vo))
                        requotes += 1
                    vo, vt = -1, tgt
                    d = round(tgt - known, 9)
                    side = 1 if d > 0 else -1
                    ref = ref_b[k] if side > 0 else ref_a[k]
                    if d != 0 and isfinite(ref):
                        vo = len(o_side)
                        o_side.append(side)
                        o_price.append(int(ref))
                        o_size.append(abs(d))
                        o_sent.append(td)
                        o_arr.append(td + lat)
                        o_ahead.append(-1.0)
                        o_end.append(_NEVER)
                        o_status.append("sent")
                        o_dead.append(_NEVER)
                        arrivals.append((td + lat, vo))
                    k += 1
                    continue
                z = size[k]
                want_b = want_a = None
                if z > 0 and isfinite(cap[k]):
                    # a side is quoted only while its full fill keeps the known inventory
                    # within the cap
                    sk, room = skew[k], cap[k] + _EPS
                    if known + z <= room and isfinite(base_b[k]) and isfinite(ref_b[k]):
                        want_b = int(ref_b[k]) - ceil(base_b[k] + sk * known - _EPS)
                    if known - z >= -room and isfinite(base_a[k]) and isfinite(ref_a[k]):
                        want_a = int(ref_a[k]) + ceil(base_a[k] - sk * known - _EPS)
                if gate is not None:
                    while sent_b and o_end[sent_b[0]] <= lim:
                        sent_b.popleft()
                    while sent_a and o_end[sent_a[0]] <= lim:
                        sent_a.popleft()
                    keep_b = vb >= 0 and o_dead[vb] > lim and o_price[vb] == want_b
                    keep_a = va >= 0 and o_dead[va] > lim and o_price[va] == want_a
                    keep_b, keep_a = keep_b and o_size[vb] == z, keep_a and o_size[va] == z
                    rb = z + sum(o_size[v] for v in sent_b if o_end[v] > lim and not (
                        keep_b and v == vb))  # fmt: skip
                    ra = z + sum(o_size[v] for v in sent_a if o_end[v] > lim and not (
                        keep_a and v == va))  # fmt: skip
                    ok_b, ok_a = gate(k, td, lim, kf, fills, want_b, want_a, rb, ra)
                    want_b = want_b if ok_b else None
                    want_a = want_a if ok_a else None
                for side in (1, -1):
                    v, want = (vb, want_b) if side > 0 else (va, want_a)
                    alive = v >= 0 and o_dead[v] > lim
                    if alive and want is not None and o_price[v] == want and o_size[v] == z:
                        continue
                    if alive:
                        cancels.append((td + lc, v))
                        requotes += want is not None
                    if want is None:
                        v = -1
                    else:
                        v = len(o_side)
                        o_side.append(side)
                        o_price.append(want)
                        o_size.append(z)
                        o_sent.append(td)
                        o_arr.append(td + lat)
                        o_ahead.append(-1.0)
                        o_end.append(_NEVER)
                        o_status.append("sent")
                        o_dead.append(_NEVER)
                        arrivals.append((td + lat, v))
                        if gate is not None:
                            (sent_b if side > 0 else sent_a).append(v)
                    if side > 0:
                        vb = v
                    else:
                        va = v
                k += 1
        if e >= nev:
            break
        e += 1
        if kd == 0:
            cb, ca, cbq, caq = F1[x], F2[x], F3[x], F4[x]
            # the far touch at or through a resting order's price fills its remainder there
            if cross and bo >= 0 and ca <= bp - reach:
                fill(t, 1, bo, bp, brem, False, False, True)
                inv += brem
                o_status[bo], o_end[bo], o_dead[bo] = "filled", t, t
                bo = -1
            if cross and ao >= 0 and cb >= ap + reach:
                fill(t, -1, ao, ap, arem, False, False, True)
                inv -= arem
                o_status[ao], o_end[ao], o_dead[ao] = "filled", t, t
                ao = -1
            if bo >= 0 and queue:
                if cb == bp:
                    if bA < 0:
                        bA = bd = cbq
                        bs = 0.0
                    elif pro:
                        ex = bd - bs
                        if cbq < ex:
                            bA -= (ex - cbq) * bA / ex
                        bA = max(0.0, min(bA, cbq))
                        bd, bs = cbq, 0.0
                    elif cbq < bA:
                        bA = cbq
                elif cb < bp:
                    bA = bd = bs = 0.0
                if trace is not None and cb <= bp:
                    trace.append((t, 1, bo, bA))
            if ao >= 0 and queue:
                if ca == ap:
                    if aA < 0:
                        aA = ad = caq
                        as_ = 0.0
                    elif pro:
                        ex = ad - as_
                        if caq < ex:
                            aA -= (ex - caq) * aA / ex
                        aA = max(0.0, min(aA, caq))
                        ad, as_ = caq, 0.0
                    elif caq < aA:
                        aA = caq
                elif ca > ap:
                    aA = ad = as_ = 0.0
                if trace is not None and ca >= ap:
                    trace.append((t, -1, ao, aA))
            continue
        px, y = F1[x], F3[x]
        if F2[x] < 0:  # a sell hits the bids
            if bo < 0:
                continue
            if px < bp:
                got, thr = (min(brem, y) if capped else brem), True
            elif px == bp and bA >= 0:
                bA -= y
                bs += y
                if trace is not None:
                    trace.append((t, 1, bo, max(bA, 0.0)))
                if bA >= 0:
                    continue
                got, thr, bA = min(brem, -bA), False, 0.0
            else:
                continue
            fill(t, 1, bo, bp, got, thr)
            inv += got
            brem -= got
            if brem <= _EPS * o_size[bo]:
                o_status[bo], o_end[bo], o_dead[bo] = "filled", t, t
                bo = -1
        else:  # a buy lifts the asks
            if ao < 0:
                continue
            if px > ap:
                got, thr = (min(arem, y) if capped else arem), True
            elif px == ap and aA >= 0:
                aA -= y
                as_ += y
                if trace is not None:
                    trace.append((t, -1, ao, max(aA, 0.0)))
                if aA >= 0:
                    continue
                got, thr, aA = min(arem, -aA), False, 0.0
            else:
                continue
            fill(t, -1, ao, ap, got, thr)
            inv -= got
            arem -= got
            if arem <= _EPS * o_size[ao]:
                o_status[ao], o_end[ao], o_dead[ao] = "filled", t, t
                ao = -1

    inv = round(inv, 9)
    flatten = None
    # under rules 3 a maker-taker pass flattens what its last target left too, as the taker
    # sim now crosses its last position
    if inv != 0 and cb is not None and (not taking or check_rules(rules) >= 3):
        flatten = (stop, -1 if inv > 0 else 1, cb if inv > 0 else ca, abs(inv))
    ahead = np.array(o_ahead, dtype=float)
    ahead[ahead < 0] = np.nan
    return Book(
        fill_ns=np.array(f_ns, dtype=np.int64),
        fill_side=np.array(f_side, dtype=np.int64),
        fill_price=np.array(f_px, dtype=np.int64),
        fill_qty=np.array(f_qty, dtype=float),
        fill_through=np.array(f_thr, dtype=bool),
        fill_order=np.array(f_oid, dtype=np.int64),
        order_side=np.array(o_side, dtype=np.int64),
        order_price=np.array(o_price, dtype=np.int64),
        order_size=np.array(o_size, dtype=float),
        order_sent=np.array(o_sent, dtype=np.int64),
        order_arrival=np.array(o_arr, dtype=np.int64),
        order_ahead=ahead,
        order_end=np.array(o_end, dtype=np.int64),
        order_status=np.array(o_status, dtype=object),
        flatten=flatten,
        rejected=rejected,
        requotes=requotes,
        quoted=quoted,
        fill_taker=np.array(f_tak, dtype=bool),
        fill_crossed=np.array(f_crs, dtype=bool),
    )


def _merged(n: int, at_book, at_trade, book_values, trade_values, dtype) -> np.ndarray:
    """One column of the merged event stream: the book rows' values and the trades' in place."""
    out = np.zeros(n, dtype=dtype)
    out[at_book] = book_values
    out[at_trade] = trade_values
    return out


# -- the accounting ---------------------------------------------------------


@dataclass
class MakerSim(taker.Sim):
    """One closed-loop pass of a market-making strategy over one window, under one fill model.

    The per-step arrays are :class:`blinded.sim.Sim`'s, over the steps from
    :math:`t_i = g_i + L` to :math:`t_{i+1}`, the last ending at the flatten:
    ``position`` is the inventory as-of :math:`t_i`, and a fill stamped in
    :math:`(t_i, t_{i+1}]` belongs to step :math:`i`, so that

    .. math::

        \\text{mid\\_pnl}_i = \\frac{10^4}{P_\\text{ref}} \\Big( h_i (m_{i+1} - m_i)
            + \\sum_f s_f q_f (m_{i+1} - m(t_f)) \\Big), \\qquad
        \\text{spread\\_bps}_i = -\\frac{10^4}{P_\\text{ref}} \\sum_f s_f q_f (m(t_f) - p_f),

    and ``gross`` telescopes to the cash of every fill, the flatten's
    included. A maker's ``spread_bps`` is negative: earned. ``fills`` counts
    the engine's fills of a step, partial or not, and ``units`` the units
    that rested before they filled; the flatten is one taker fill in the
    last step, with its half spread in ``spread_bps`` and its fee in
    ``fee_paid``, counted in neither. ``taker_units`` holds each step's
    crossed units at the taker fee: the flatten's, and in maker-taker
    execution each deadline's (:func:`simulate_maker_taker`).
    ``captured`` is what the maker fills earned against the mid at their
    stamps, ``adverse`` what the mid did to them over the next 10 seconds,
    in total. ``book`` keeps the engine's fills and orders.
    """

    model: str = "RA"
    book: Book | None = None
    touch: Touch | None = None
    ref: float = float("nan")
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    units: np.ndarray = field(default_factory=lambda: np.array([]))
    taker_units: np.ndarray = field(default_factory=lambda: np.array([]))
    # under rules 3 the units each step's fees are charged on, notional over P_ref
    fee_taker_units: np.ndarray | None = None
    captured: np.ndarray = field(default_factory=lambda: np.array([]))
    adverse: float = 0.0
    flatten_units: float = 0.0
    runs: dict = field(default_factory=dict)
    per_block: int = 1
    step_ns: int = 0
    start_ns: int = 0

    def metrics(self) -> dict:
        """:meth:`blinded.sim.Sim.metrics`, with turnover counted in units filled, and the
        maker's own: units filled, captured, adverse selection at 10 s, inventory beyond it,
        post-only rejections, requotes, orders sent, the share of decisions with a side
        resting, and the mean absolute inventory."""
        out = super().metrics()
        units, captured = float(self.units.sum()), float(self.captured.sum())
        quoted = self.book.quoted if self.book is not None else np.zeros(0)
        out |= {
            "turnover_per_hour": units / self.hours,
            "units": units,
            "captured_bps": captured,
            "adverse_bps_10s": self.adverse,
            "inventory_bps": float(self.gross.sum()) - captured - self.adverse,
            "post_only_rejected": self.book.rejected if self.book is not None else 0,
            "requotes": self.book.requotes if self.book is not None else 0,
            "orders": len(self.book.order_side) if self.book is not None else 0,
            "quoted_share": float(np.mean(quoted > 0)) if len(quoted) else float("nan"),
            "mean_abs_inventory": float(np.mean(np.abs(self.position))),
            "flatten_units": self.flatten_units,
        }
        return out

    def charged(self) -> tuple[np.ndarray, np.ndarray]:
        """The units each step's maker and taker fees are charged on: the units filled, or
        under rules 3 their notional over P_ref."""
        made = self.units if self.fee_units is None else self.fee_units
        took = self.taker_units if self.fee_taker_units is None else self.fee_taker_units
        return made, took

    def refee(self, maker_fee_bps: float, taker_fee_bps: float | None = None) -> MakerSim:
        """The same fills at other fees: fees change no fill, so the fee rows are arithmetic.

        ``taker_fee_bps``, the flatten's, defaults to this sim's.
        """
        taker_fee = self.taker_fee_bps if taker_fee_bps is None else taker_fee_bps
        made, took = self.charged()
        fee = made * maker_fee_bps + took * taker_fee
        out = replace(self, fee_bps=maker_fee_bps, fee_paid=fee, maker_fee_bps=maker_fee_bps,
                      taker_fee_bps=taker_fee)  # fmt: skip
        out.blocks = _blocks(out.net, self.per_block, self.rules)
        out.run_net = _run_net(self.runs, maker_fee_bps, taker_fee)
        return out


def simulate_maker(
    quotes: Quotes,
    panel: Panel,
    window: Window,
    instrument: str,
    model: str = "RA",
    tick: float | None = None,
    lot: float | None = None,
    latency: pd.Timedelta = taker._ZERO,
    cancel_latency: pd.Timedelta | None = None,
    response_latency: pd.Timedelta | None = None,
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 0.0,
    block: pd.Timedelta = taker._BLOCK,
    touch: Touch | None = None,
    funding: Events | None = None,
    borrow: Events | None = None,
    rules: int = RULES,
) -> MakerSim:
    """Rest ``quotes`` on ``instrument`` over ``window`` under one fill model and account for it.

    The decisions are the window's grid points; :func:`run_engine` fills them
    against ``touch`` (by default :func:`touch_of` the panel, with ``tick``),
    and the fills are marked to the touch's mid as :class:`MakerSim` says. A
    maker fill pays ``maker_fee_bps`` a unit (negative for a rebate) and the
    flatten ``taker_fee_bps``. A perpetual's ``funding`` stream accrues as
    :func:`blinded.sim.accrue` does, on the inventory held at each settlement
    stamp, and a spot instrument's ``borrow`` stream as
    :func:`blinded.sim.borrowed` does, on the inventory as-of each step's
    start; both default to the panel's streams of ``instrument``.
    """
    from blinded.contract import values

    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [window.start.value, window.end.value])
    if i1 - i0 < 1:
        raise ValueError("the window holds no grid points")
    if not all(isinstance(v, np.ndarray) and len(v) == len(g) for v in quotes.values()):
        quotes = values(quotes, panel)
    if touch is None:
        touch = touch_of(panel, instrument, tick, window)
    G = g[i0:i1]
    end = int(G[-1] + panel.step_ns)
    params = Quotes({k: np.asarray(v, dtype=float)[i0:i1] for k, v in quotes.items()})
    lat = latency.value
    lc = lat if cancel_latency is None else cancel_latency.value
    lr = lat if response_latency is None else response_latency.value
    book = run_engine(touch, Decisions.make(params, G, touch, lot), end, model, lat, lc, lr,
                      rules=rules)  # fmt: skip
    if funding is None:
        funding = taker.funding_of(panel, instrument)
    if borrow is None:
        borrow = taker.borrow_of(panel, instrument)
    return account(book, touch, G, end, panel.step_ns, latency, model, maker_fee_bps,
                   taker_fee_bps, block, funding, borrow, rules)  # fmt: skip


def account(book: Book, touch: Touch, grid_ns: np.ndarray, end_ns: int, step_ns: int,
            latency: pd.Timedelta, model: str, maker_fee_bps: float, taker_fee_bps: float,
            block: pd.Timedelta = taker._BLOCK, funding: Events | None = None,
            borrow: Events | None = None, rules: int = RULES) -> MakerSim:  # fmt: skip
    """The per-step P&L of one engine pass (:class:`MakerSim`), its blocks under the judging
    ``rules`` the engine ran under."""
    lat = latency.value
    t = grid_ns.astype(np.int64) + lat
    t_end = int(end_ns) + lat
    t_next = np.concatenate([t[1:], [t_end]])
    n = len(t)
    ns, side, qty = book.fill_ns, book.fill_side, book.fill_qty
    price = book.fill_price * touch.tick
    # a maker fill rested; a taker one crossed, at a deadline (maker-taker) or the flatten
    maker = ~book.fill_taker if len(book.fill_taker) == len(ns) else np.ones(len(ns), bool)
    counted = np.ones(len(ns), dtype=bool)
    flat = 0.0
    if book.flatten is not None:
        stamp, s, p, q = book.flatten
        ns, side = np.append(ns, stamp), np.append(side, s)
        price, qty = np.append(price, p * touch.tick), np.append(qty, q)
        maker, flat = np.append(maker, False), float(q)
        counted = np.append(counted, False)
    signed = side * qty
    m = touch.mid(t)
    m_next = touch.mid(t_next)
    finite = m[np.isfinite(m)]
    ref = float(finite[0]) if len(finite) else float("nan")
    at = touch.mid(ns)
    # step i holds the fills stamped in (t_i, t_{i+1}]; a fill on t_0 itself, the first
    # arrival, is already in h_0 and only its spread and fee go to step 0
    i = np.searchsorted(t, ns, side="left") - 1
    step = np.maximum(i, 0)
    after = np.round(np.cumsum(signed), 9)
    known = np.searchsorted(ns, t, side="right") - 1
    h = np.where(known >= 0, after[np.maximum(known, 0)] if len(after) else 0.0, 0.0)

    def per_step(weights: np.ndarray, where=None) -> np.ndarray:
        keep = np.ones(len(ns), dtype=bool) if where is None else where
        return np.bincount(step[keep], weights=weights[keep], minlength=n)[:n]

    intra = np.where(i >= 0, signed * (m_next[step] - at), 0.0)
    mid_pnl = np.nan_to_num(h * (m_next - m) + per_step(intra)) / ref * 1e4
    edge = signed * (at - price)
    spread = -np.nan_to_num(per_step(edge)) / ref * 1e4
    units = per_step(qty, maker)
    crossed = per_step(qty, ~maker)
    fee_units = fee_crossed = None
    notional = check_rules(rules) >= 3
    if notional:
        # each fill's fee on its own notional, the fill price over P_ref
        worth = qty * price / ref
        fee_units, fee_crossed = per_step(worth, maker), per_step(worth, ~maker)
        fee = fee_units * maker_fee_bps + fee_crossed * taker_fee_bps
    else:
        fee = units * maker_fee_bps + crossed * taker_fee_bps
    later = touch.mid(ns + ADVERSE.value)
    adverse = float(np.nansum((signed * (later - at))[maker]) / ref * 1e4)

    def held_at(stamps):
        j = np.searchsorted(ns, stamps, side="right") - 1
        return np.where(j >= 0, after[np.maximum(j, 0)] if len(after) else 0.0, 0.0)

    paid = owed = None
    settled = (np.zeros(0, dtype=np.int64), np.zeros(0))
    if funding is not None:
        paid, *settled = taker.accrue(h, t, t_end, funding, touch.mid, ref, held=held_at,
                                      each=True)  # fmt: skip
    if borrow is not None:
        owed = taker.borrowed(h, t, t_next, borrow, m, ref)
    per_block = max(int(block.value // step_ns), 1)
    sim = MakerSim(
        latency=latency,
        fee_bps=maker_fee_bps,
        hours=n * step_ns / taker.HOUR.value,
        position=h,
        trade=per_step(signed),
        mid_pnl=mid_pnl,
        spread_bps=spread,
        fee_paid=fee,
        fills=np.bincount(step[counted], minlength=n)[:n],
        blocks=np.array([]),
        blocks_per_hour=taker.HOUR.value / (per_block * step_ns),
        funding=paid,
        borrow=owed,
        rules=rules,
        model=model,
        book=book,
        touch=touch,
        ref=ref,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        units=units,
        taker_units=crossed,
        captured=np.nan_to_num(per_step(edge, maker)) / ref * 1e4,
        adverse=adverse,
        flatten_units=flat,
        per_block=per_block,
        step_ns=int(step_ns),
        start_ns=int(t[0]),
        fee_units=fee_units,
        fee_taker_units=fee_crossed,
    )
    sim.blocks = _blocks(sim.net, per_block, rules)
    # funding goes to the run open at its settlement, borrow to the run open at its step
    stamps = np.concatenate([settled[0], t])
    carried = np.concatenate([settled[1], np.zeros(n) if owed is None else owed])
    sim.runs = _runs(ns, signed, price, maker, ref, stamps, carried, notional)
    sim.run_net = _run_net(sim.runs, maker_fee_bps, taker_fee_bps)
    return sim


def _blocks(net: np.ndarray, per_block: int, rules: int) -> np.ndarray:
    return sharpe_blocks(net, per_block, rules)


def _runs(ns, signed, price, maker, ref, stamps, carried, notional: bool = False) -> dict:
    """The round trips of a maker's inventory, from its fills: runs of one sign.

    Each fill's cash goes to the run it opens or extends; the part of a fill
    that closes a run goes to that run and the rest opens the next. Each
    run's cash, maker units and taker units are kept so that its net can be
    priced at any fee (:func:`_run_net`). ``carried`` (funding at its
    settlement, borrow at its step's start) goes to the run open at its
    stamp, which is the run holding the inventory it was charged on: a run
    holds from its first fill's stamp up to, not including, its last's. The
    flatten closes the last run, so the runs' nets add up to the window's.
    With ``notional`` (rules 3) the units a run's fees are charged on are each
    part's notional over ``ref``, as the pass's own fees are.
    """
    cash, mu, tu, starts, ends = [], [], [], [], []
    inv = 0.0
    for stamp, sq, p, mk in zip(ns.tolist(), signed.tolist(), price.tolist(), maker.tolist(),
                                strict=True):  # fmt: skip
        left = abs(sq)
        s = 1.0 if sq > 0 else -1.0
        while left > _EPS:
            if inv == 0 or (inv > 0) == (s > 0):
                if inv == 0:
                    cash.append(0.0)
                    mu.append(0.0)
                    tu.append(0.0)
                    starts.append(stamp)
                    ends.append(_NEVER)
                part = left
            else:
                part = min(left, abs(inv))
            cash[-1] -= s * part * p
            worth = part * p / ref if notional else part
            if mk:
                mu[-1] += worth
            else:
                tu[-1] += worth
            inv = round(inv + s * part, 9)
            left -= part
            if inv == 0:
                ends[-1] = stamp
    starts_a, ends_a = np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)
    run_carry = np.zeros(len(starts))
    if len(starts):
        r = np.searchsorted(starts_a, stamps, side="right") - 1
        inside = (r >= 0) & (carried != 0)
        inside[inside] &= stamps[inside] < ends_a[r[inside]]
        np.add.at(run_carry, r[inside], carried[inside])
    return {"cash_bps": np.array(cash) / ref * 1e4, "maker_units": np.array(mu),
            "taker_units": np.array(tu), "carried_bps": run_carry,
            "start_ns": starts_a, "end_ns": ends_a}  # fmt: skip


def _run_net(runs: dict, maker_fee_bps: float, taker_fee_bps: float) -> np.ndarray:
    if not runs or not len(runs["cash_bps"]):
        return np.array([])
    return (runs["cash_bps"] - runs["maker_units"] * maker_fee_bps
            - runs["taker_units"] * taker_fee_bps + runs["carried_bps"])  # fmt: skip


# -- the rows and the score -------------------------------------------------


def held(s: MakerSim) -> dict:
    """The hold rule on one pass: the Sharpe the judge scores, and the block it is measured in.

    The block is the brief's (``per_block`` steps) or the pass's time-weighted
    mean hold (:func:`blinded.sim.hold_steps`) rounded up to whole steps,
    whichever is longer, since one hold is one bet however many blocks it
    spans (``docs/ladder.md``). ``sharpe`` is the net Sharpe per square-root
    hour over that block's whole blocks from the window's start, under the
    pass's judging rules (:func:`blinded.ladder.sharpe_blocks`: under rules 2
    the remainder folded into the last, under rules 1 dropped), exactly the
    point value the daemon gives the Ladder; ``blocks`` is how many there
    are, and ``releasable`` is False when the block is the hold's and
    validation would hold fewer than :data:`blinded.archetypes.PERIODS` of
    them. With fewer than two blocks, or no variation, ``sharpe`` is NaN: there
    is nothing to measure.
    """
    from blinded.archetypes import PERIODS
    from blinded.ladder import block_sharpe

    hold = taker.hold_steps(s)
    own = max(s.per_block, math.ceil(hold))
    k = len(s.net) // own
    blocks = sharpe_blocks(s.net, own, s.rules)
    sharpe = block_sharpe(blocks, taker.HOUR.value / (own * s.step_ns))
    return {"sharpe": sharpe, "hold_steps": hold, "block_steps": own, "blocks": k,
            "releasable": own == s.per_block or k >= PERIODS}  # fmt: skip


def worse(a: MakerSim, b: MakerSim) -> MakerSim:
    """The one of two passes with the lower Sharpe under the hold rule (:func:`held`), the
    Sharpe the judge scores; a pass with none (no P&L to measure) is the worse, and on a tie
    the first."""
    sa, sb = (held(s)["sharpe"] for s in (a, b))
    if not np.isfinite(sa):
        return a
    if not np.isfinite(sb):
        return b
    return b if sb < sa else a


def models(quotes: Quotes, panel: Panel, window: Window, brief: Brief,
           instrument: str | None = None, touch: Touch | None = None,
           which: tuple[str, ...] = MODELS, zero: bool = False
           ) -> dict[str, MakerSim]:  # fmt: skip
    """The passes of ``which`` models at the brief's latencies and fees, or at every latency 0
    with ``zero``."""
    ex = brief.execution
    inst = instrument or brief.target.instrument
    if touch is None:
        touch = touch_of(panel, inst, ex.tick_size, window)
    if not all(isinstance(v, np.ndarray) and len(v) == len(panel) for v in quotes.values()):
        from blinded.contract import values

        quotes = values(quotes, panel)
    if zero:
        lat = dict(latency=taker._ZERO, cancel_latency=None, response_latency=None)
    else:
        lat = dict(latency=ex.latency, cancel_latency=ex.cancel, response_latency=ex.response)
    return {m: simulate_maker(quotes, panel, window, inst, m, touch.tick, ex.lot_size,
                              maker_fee_bps=ex.maker_fee_bps, taker_fee_bps=ex.fee(inst),
                              block=brief.stats.block, touch=touch, rules=brief.judge_rules,
                              **lat)
            for m in which}  # fmt: skip


def report(quotes: Quotes, panel: Panel, window: Window, brief: Brief,
           instrument: str | None = None, touch: Touch | None = None,
           sims: dict[str, MakerSim] | None = None, scored: str | None = None,
           zero: MakerSim | None = None) -> pl.DataFrame:  # fmt: skip
    """The maker's sim table: the scored model, the four models, latency 0 and the fee rows.

    ``brief`` is the worse of RA and PR (:func:`worse`, by the hold rule's
    Sharpe, as the judge scores); ``RA``, ``PR``,
    ``T`` and ``F`` are the four models at the brief's latencies and fees;
    ``latency 0`` is the scored model with every latency at 0 (unrealistic);
    ``fees 0``, ``rebate 0.5``, ``fee 1`` and ``fee 2 (Binance VIP0)`` are
    the scored model's own fills at those maker fees (:meth:`MakerSim.refee`;
    ``fees 0`` waives the flatten's taker fee too). The ``model`` column names
    the model each row ran. ``sims``, when given, are the four passes already
    made, ``scored`` the model the judge scored, when it is known, and
    ``zero`` that model's pass at every latency 0, when it is made.
    """
    ex = brief.execution
    inst = instrument or brief.target.instrument
    if touch is None:
        touch = touch_of(panel, inst, ex.tick_size, window)
    sims = sims or models(quotes, panel, window, brief, inst, touch)
    scored = worse(sims["RA"], sims["PR"]) if scored is None else sims[scored]
    if zero is None:
        (zero,) = models(quotes, panel, window, brief, inst, touch, (scored.model,),
                         True).values()  # fmt: skip
    rows = [
        (
            "brief",
            scored,
            "the worse of the two queue models by the Sharpe the judge "
            f"scores, on blocks as long as its holds: {scored.model}",
        )
    ]
    rows += [(m, sims[m], NAMES[m]) for m in MODELS]
    rows.append(("latency 0", zero, taker.UNREALISTIC))
    for name, fee in FEE_ROWS:
        flatten = 0.0 if name == "fees 0" else None
        rows.append((name, scored.refee(fee, flatten), f"{scored.model}'s fills at this fee"))
    table = [{"row": name, "model": s.model, **s.metrics(), "note": note}
             for name, s, note in rows]  # fmt: skip
    return pl.DataFrame(table, infer_schema_length=None)


HORIZONS = (pd.Timedelta("1s"), ADVERSE, pd.Timedelta("60s"))


def decompose(s: MakerSim, horizons: tuple[pd.Timedelta, ...] = HORIZONS) -> dict:
    """Where a maker's P&L went, in bps of one unit and per unit filled.

    ``captured`` is :math:`\\sum_f s_f q_f (m(t_f) - p_f)` over the maker
    fills, ``adverse`` at each horizon :math:`h` is :math:`\\sum_f s_f q_f
    (m(t_f + h) - m(t_f))`, and ``inventory`` at :math:`h` the rest of the
    gross, the holding beyond :math:`h` and the flatten; ``net`` is the gross
    less fees plus funding and borrow.
    """
    b = s.book
    maker = ~b.fill_taker if len(b.fill_taker) == len(b.fill_ns) else np.ones(len(b.fill_ns),
                                                                              bool)  # fmt: skip
    signed = b.fill_side * b.fill_qty
    at = s.touch.mid(b.fill_ns)
    captured = float(s.captured.sum())
    gross = float(s.gross.sum())
    units = float(s.units.sum())
    out = {"units": units, "captured_bps": captured, "gross_bps": gross,
           "fees_bps": float(s.fee_paid.sum()), "net_bps": float(s.net.sum()),
           "adverse_bps": {}, "inventory_bps": {}}  # fmt: skip
    for h in horizons:
        later = s.touch.mid(b.fill_ns + h.value)
        adverse = float(np.nansum((signed * (later - at))[maker]) / s.ref * 1e4)
        key = label(h)
        out["adverse_bps"][key] = adverse
        out["inventory_bps"][key] = gross - captured - adverse
    per = (lambda x: x / units) if units else (lambda x: float("nan"))
    out["per_unit"] = {"captured_bps": per(captured),
                       "adverse_bps": {k: per(v) for k, v in out["adverse_bps"].items()},
                       "fees_bps": per(out["fees_bps"]),
                       "net_bps": per(out["net_bps"])}  # fmt: skip
    return out


# -- maker-taker execution of a position strategy ------------------------------


DEADLINE = pd.Timedelta("2s")


def has_touch(brief: Brief, instrument: str) -> bool:
    """Whether the fill engine can run on ``instrument``: a touch file declared for it, or,
    with none declared, a file that keeps every book row with its sizes and every trade at
    its own price (:func:`blinded.archetypes.touch_lack`)."""
    from blinded.archetypes import touch_lack

    if brief.data.touch is not None:
        return brief.data.touch.instrument == instrument
    return touch_lack(brief) is None


def simulate_maker_taker(
    position: np.ndarray,
    panel: Panel,
    window: Window,
    instrument: str,
    model: str = "RA",
    tick: float | None = None,
    latency: pd.Timedelta = taker._ZERO,
    cancel_latency: pd.Timedelta | None = None,
    response_latency: pd.Timedelta | None = None,
    deadline: pd.Timedelta = DEADLINE,
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 0.0,
    min_rebalance: float = 0.0,
    max_position: float = 1.0,
    block: pd.Timedelta = taker._BLOCK,
    touch: Touch | None = None,
    funding: Events | None = None,
    borrow: Events | None = None,
    rules: int = RULES,
) -> MakerSim:
    """A position strategy executed as a maker that takes at a deadline, under one fill model.

    ``position`` is what the taker sim (:func:`blinded.sim.simulate`) would
    hold after each decision, through :func:`blinded.sim.held`. A decision
    whose target differs from the one its live order was sent for cancels
    that order (at :math:`g + L_c`) and posts the difference between the
    target and the inventory it knows (fills stamped by :math:`g - L_r`) at
    the touch on its side as-of the arrival :math:`g + L`: a buy at the best
    bid, a sell at the best ask, where the taker sim fills at the far touch
    then. The order rests in the queue of ``model``; whatever of it is
    unfilled ``deadline`` after its arrival is crossed at the far touch as-of
    that stamp and pays ``taker_fee_bps``, and what filled while resting pays
    ``maker_fee_bps``. Every deadline falls by the end of the window's last
    step at the latest, and the inventory left is marked at the mid there, as
    the taker sim marks its last position, with no flatten.
    """
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [window.start.value, window.end.value])
    if i1 - i0 < 1:
        raise ValueError("the window holds no grid points")
    h = taker.held(np.asarray(position, dtype=float)[i0:i1], min_rebalance, max_position)
    if touch is None:
        touch = touch_of(panel, instrument, tick, window)
    G = g[i0:i1]
    end = int(G[-1] + panel.step_ns)
    lat = latency.value
    lc = lat if cancel_latency is None else cancel_latency.value
    lr = lat if response_latency is None else response_latency.value
    j = touch.row(G + lat)
    ok = j >= 0
    ref_b = np.where(ok, touch.bid[np.maximum(j, 0)], np.nan).astype(float)
    ref_a = np.where(ok, touch.ask[np.maximum(j, 0)], np.nan).astype(float)
    zero = np.zeros(len(G))
    decisions = Decisions(G, ref_b, ref_a, zero, zero, zero, np.full(len(G), np.inf), zero)
    book = run_engine(touch, decisions, end, model, lat, lc, lr, targets=h,
                      deadline_ns=deadline.value, rules=rules)  # fmt: skip
    if funding is None:
        funding = taker.funding_of(panel, instrument)
    if borrow is None:
        borrow = taker.borrow_of(panel, instrument)
    return account(book, touch, G, end, panel.step_ns, latency, model, maker_fee_bps,
                   taker_fee_bps, block, funding, borrow, rules)  # fmt: skip


def maker_taker(position: np.ndarray, panel: Panel, window: Window, brief: Brief,
                instrument: str, touch: Touch | None = None) -> MakerSim:  # fmt: skip
    """:func:`simulate_maker_taker` at the brief's latencies, fees, deadline and position
    rules, under RA and PR, and the worse of the two (:func:`worse`), the archetype's rule."""
    ex = brief.execution
    if touch is None:
        touch = touch_of(panel, instrument, ex.tick_size, window)
    sims = [simulate_maker_taker(position, panel, window, instrument, m, touch.tick, ex.latency,
                                 ex.cancel, ex.response, ex.deadline, ex.maker_fee_bps,
                                 ex.fee(instrument), ex.min_rebalance, ex.max_position,
                                 brief.stats.block, touch, rules=brief.judge_rules)
            for m in QUEUE_MODELS]  # fmt: skip
    return worse(*sims)


def maker_taker_row(s: MakerSim, deadline: pd.Timedelta) -> dict:
    """The ``maker-taker`` row of :func:`blinded.sim.report`: the taker table's columns, with
    ``fills`` every fill, resting or crossed, and ``maker_share``, the units that filled while
    resting over every unit traded."""
    out = taker.Sim.metrics(s)
    rested, crossed = float(s.units.sum()), float(s.taker_units.sum())
    traded = rested + crossed
    out["maker_share"] = rested / traded if traded else float("nan")
    note = (f"posted at the touch, crossed at a {label(deadline)} deadline, under {s.model}, the "
            "worse of the two queue models; diagnostic, never in the Ladder")  # fmt: skip
    return {"row": "maker-taker", **out, "note": note}


# -- what the report draws ---------------------------------------------------

# the offsets of a maker's fill fan, from the fill: dense near it, every 10 s further out
FAN_OFFSETS = ("-10s", "-5s", "-2s", "-1s", "-500ms", "-200ms", "-100ms", "0ms", "100ms", "200ms",
               "500ms", "1s", "2s", "5s", "10s", "20s", "30s", "40s", "50s", "60s")  # fmt: skip
FAN_GAP = pd.Timedelta("70s")


def _offset_ns(text: str) -> int:
    return -pd.Timedelta(text[1:]).value if text.startswith("-") else pd.Timedelta(text).value


def fill_fan(s: MakerSim, end_ns: int) -> dict:
    """The mid's move from each resting fill at :data:`FAN_OFFSETS`, in the fill's direction, in
    bps: mean and standard error weighted by the fill's units, fills less than
    :data:`FAN_GAP` apart one draw (:func:`blinded.markouts.fan`), moves past ``end_ns`` left
    out. ``captured_bps_per_unit`` is where the fill price sat against the mid in the same
    direction, positive when it was better than the mid, so the fan's marker at 0 is minus it.
    """
    from blinded import markouts as mk

    b = s.book
    rest = ~b.fill_taker if len(b.fill_taker) == len(b.fill_ns) else np.ones(len(b.fill_ns),
                                                                             bool)  # fmt: skip
    ns, side, qty = b.fill_ns[rest], b.fill_side[rest], b.fill_qty[rest]
    offsets = [_offset_ns(o) for o in FAN_OFFSETS]
    moves = mk.mid_moves(s.touch.mid, ns, offsets, int(end_ns))
    fan = mk.fan(moves, side.astype(float), qty, mk.clusters(ns, FAN_GAP.value))
    units = float(s.units.sum())
    captured = float(s.captured.sum()) / units if units else None
    return {"n": fan["n"], "mean_bps": fan["mean_bps"], "se_bps": fan["se_bps"],
            "captured_bps_per_unit": captured}  # fmt: skip


def _fill_steps(s: MakerSim) -> np.ndarray:
    """The step each of the engine's fills belongs to: step i holds :math:`(t_i, t_{i+1}]`."""
    t = s.start_ns + s.step_ns * np.arange(len(s.net))
    return np.maximum(np.searchsorted(t, s.book.fill_ns, side="left") - 1, 0)


def pnl_lines(s: MakerSim, points: int = 600) -> dict:
    """Where a maker's P&L goes, cumulative by step: what the resting fills captured against the
    mid at their stamps, that plus the mid's move over the next 10 s (adverse selection), and
    the net. The gap between the second and the third is the inventory held past 10 s, the
    flatten and the fees."""
    from blinded import markouts as mk

    b = s.book
    rest = ~b.fill_taker if len(b.fill_taker) == len(b.fill_ns) else np.ones(len(b.fill_ns),
                                                                             bool)  # fmt: skip
    signed = b.fill_side * b.fill_qty
    moved = signed * (s.touch.mid(b.fill_ns + ADVERSE.value) - s.touch.mid(b.fill_ns))
    adverse = np.bincount(_fill_steps(s)[rest], weights=np.nan_to_num(moved[rest]),
                          minlength=len(s.net))[: len(s.net)] / s.ref * 1e4  # fmt: skip
    captured = np.cumsum(s.captured)
    keep = mk.downsample(len(s.net), points)
    hours = keep * (s.step_ns / taker.HOUR.value)
    return {"hours": hours.tolist(), "captured": captured[keep].tolist(),
            "adverse": (captured + np.cumsum(adverse))[keep].tolist(),
            "net": np.cumsum(s.net)[keep].tolist()}  # fmt: skip


EXCERPT = pd.Timedelta("30min")


def excerpt(s: MakerSim, span: pd.Timedelta = EXCERPT, points: int = 900) -> dict:
    """The ``span`` of the window with the most fills of ``s``: the touch, the resting quotes
    and the inventory sampled at up to ``points`` step starts, and every fill in it.

    The span starts at the step start from which it holds the most fills (the
    first, on a tie), or at the window's start when the window is shorter.
    Prices are in units of the price; ``bid_quote`` and ``ask_quote`` are the
    price of the order resting on each side at the sample, None when none
    rests; ``inventory`` is every fill's units up to the sample, signed. Fills
    are ``(seconds from the start, side, price, units)``.
    """
    b, tick = s.book, s.touch.tick
    n = len(s.net)
    t = s.start_ns + s.step_ns * np.arange(n)
    width = int(span.value)
    ns = b.fill_ns
    if t[-1] + s.step_ns - t[0] <= width or not len(ns):
        k = 0
    else:
        last = int(np.searchsorted(t, t[-1] + s.step_ns - width, side="right"))
        starts = t[:last]
        counts = np.searchsorted(ns, starts + width) - np.searchsorted(ns, starts)
        k = int(np.argmax(counts))
    start = int(t[k])
    inside = t[(t >= start) & (t < start + width)]
    every = max(1, -(-len(inside) // points))
    at = inside[::every]
    j = s.touch.row(at)
    ok = j >= 0
    bid = np.where(ok, s.touch.bid[np.maximum(j, 0)] * tick, np.nan)
    ask = np.where(ok, s.touch.ask[np.maximum(j, 0)] * tick, np.nan)
    quotes = {}
    rested = np.isin(b.order_status, ("filled", "cancelled", "open", "crossed"))
    for side, key in ((1, "bid_quote"), (-1, "ask_quote")):
        mine = rested & (b.order_side == side)
        arr, end, px = b.order_arrival[mine], b.order_end[mine], b.order_price[mine]
        order = np.argsort(arr, kind="stable")
        arr, end, px = arr[order], end[order], px[order]
        r = np.searchsorted(arr, at, side="right") - 1
        live = (r >= 0) & (at < end[np.maximum(r, 0)])
        quotes[key] = [float(px[i] * tick) if live[m] else None for m, i in enumerate(r)]
    held = np.round(np.cumsum(b.fill_side * b.fill_qty), 9)
    w = np.searchsorted(ns, at, side="right") - 1
    inventory = np.where(w >= 0, held[np.maximum(w, 0)] if len(held) else 0.0, 0.0)
    span_fills = (ns >= start) & (ns < start + width)
    return {"start_ns": start, "seconds": ((at - start) / 1e9).tolist(),
            "bid": [None if not np.isfinite(v) else float(v) for v in bid],
            "ask": [None if not np.isfinite(v) else float(v) for v in ask], **quotes,
            "inventory": inventory.tolist(), "model": s.model,
            "fills": [[(int(x) - start) / 1e9, int(sd), float(p * tick), float(q)]
                      for x, sd, p, q in zip(ns[span_fills], b.fill_side[span_fills],
                                             b.fill_price[span_fills], b.fill_qty[span_fills],
                                             strict=True)]}  # fmt: skip


# -- maker-taker arbitrage: the hedge leg ----------------------------------------
#
# A quote rests on venue A (the quote venue) priced off venue B (the hedge venue); every fill
# is hedged on B as a taker the moment it is known, walking B's depth as-of the hedge's
# arrival. The maker leg is the engine above, unchanged but for
# the print cap; the hedge leg, the two-venue stock and the accounting are below.

ARB_MODELS = MODELS
# what the ledger keeps of each model of an arbitrage on validation
ARB_KEYS = ("net_bps_per_hour", "sharpe_per_sqrt_hour", "fills", "units", "captured_bps",
            "delay_bps", "slippage_bps", "picked_off_share", "stock_withdrawn_share")  # fmt: skip
# the hedge fees of the arbitrage fee rows, in bps, with A's maker fee at 0
HEDGE_FEE_ROWS = (("hedge fee 10", 10.0), ("hedge fee 7.5", 7.5), ("hedge fee 5", 5.0),
                  ("hedge fee 0", 0.0))  # fmt: skip
HEDGE_SLOW = pd.Timedelta("1s")
UNSET_FEE = ("stock.withdrawal_fee_coin is unset: look it up on both venues' withdrawal pages "
             "for {coin} and its network")  # fmt: skip
# under rules 3 each currency's transfer pays its own fee, and both must be set
UNSET_FEES = ("stock.withdrawal_fee_coin and stock.withdrawal_fee_usdt must both be set, each "
              "transfer paying its own currency's fee: look both up on both venues' withdrawal "
              "pages for {coin}, USDT and their networks; {unset} unset")  # fmt: skip


def unset_fees(stock, coin: str, rules: int) -> str | None:
    """Why an arbitrage on ``stock`` is not scored under ``rules``, or None: under rules 3
    both withdrawal fees must be set, under rules 1 and 2 the coin's."""
    if check_rules(rules) < 3:
        return None if stock.withdrawal_fee_coin is not None else UNSET_FEE.format(coin=coin)
    unset = [k for k in ("withdrawal_fee_coin", "withdrawal_fee_usdt")
             if getattr(stock, k) is None]  # fmt: skip
    if not unset:
        return None
    return UNSET_FEES.format(
        coin=coin, unset=" and ".join(unset) + (" is" if len(unset) == 1 else " are")
    )


@dataclass(frozen=True)
class Mark:
    """The hedge venue's touch, in price units, read as-of any stamp: the fair both legs of
    an arbitrage are marked to."""

    ts: np.ndarray
    bid: np.ndarray
    ask: np.ndarray

    def _at(self, values: np.ndarray, stamps) -> np.ndarray:
        j = np.searchsorted(self.ts, np.asarray(stamps), side="right") - 1
        out = np.full(len(j), np.nan)
        ok = j >= 0
        out[ok] = values[j[ok]]
        return out

    def mid(self, stamps) -> np.ndarray:
        """B's mid as-of each stamp, NaN before its first row."""
        return (self._at(self.bid, stamps) + self._at(self.ask, stamps)) / 2

    def bid_at(self, stamps) -> np.ndarray:
        return self._at(self.bid, stamps)

    def ask_at(self, stamps) -> np.ndarray:
        return self._at(self.ask, stamps)


def mark_of(panel: Panel, instrument: str) -> Mark:
    """The hedge venue's touch from the panel's book rows of ``instrument``: every two-sided
    row on its own stamp, as the arbitrage file keeps them."""
    book = panel.events("book", instrument)
    ts = book.frame["ts"].dt.epoch("ns").to_numpy()
    bid, ask = book.column("bid"), book.column("ask")
    ok = np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (bid > 0)
    return Mark(ts[ok].astype(np.int64), bid[ok].astype(float), ask[ok].astype(float))


@dataclass(frozen=True)
class Depth:
    """The hedge venue's levels at every snapshot: ``ts`` and, per row, ``levels`` prices and
    sizes a side, in price units and units (NaN prices for missing levels hold nothing)."""

    ts: np.ndarray
    bid_px: np.ndarray
    bid_qty: np.ndarray
    ask_px: np.ndarray
    ask_qty: np.ndarray

    @classmethod
    def from_rows(cls, rows: pl.DataFrame, levels: int) -> Depth:
        def side(name: str) -> tuple[np.ndarray, np.ndarray]:
            px = np.column_stack([rows[f"{name}{i}_px"].to_numpy().astype(float)
                                  for i in range(levels)])  # fmt: skip
            qty = np.column_stack([rows[f"{name}{i}_qty"].to_numpy().astype(float)
                                   for i in range(levels)])  # fmt: skip
            qty = np.where(np.isfinite(px) & np.isfinite(qty) & (qty > 0), qty, 0.0)
            return px, qty

        (bp, bq), (ap, aq) = side("bid"), side("ask")
        return cls(_ns(rows["ts"]).astype(np.int64), bp, bq, ap, aq)

    def walk(self, stamps: np.ndarray, sides: np.ndarray, qty: np.ndarray,
             rules: int = RULES) -> dict:  # fmt: skip
        """Each hedge order taken from the book as-of its stamp: a sale (``sides`` -1) into
        the bids, a purchase (+1) from the asks, level by level for ``qty``.

        ``vwap`` is the volume-weighted price; where the levels hold less than
        ``qty`` the rest is marked past the last level and ``unreachable`` is
        set: under rules 3 a further spread (the row's level-0 ask less bid)
        beyond it, under rules 1 and 2 at it. ``beyond`` is the units taken past
        level 0. Before the first snapshot the whole order is unreachable and
        its price NaN.
        """
        n = len(stamps)
        out = {"vwap": np.full(n, np.nan), "unreachable": np.zeros(n, dtype=bool),
               "beyond": np.zeros(n), "row": np.full(n, -1, dtype=np.int64)}  # fmt: skip
        if not n:
            return out
        j = np.searchsorted(self.ts, np.asarray(stamps), side="right") - 1
        out["row"] = j
        for i in range(n):
            if j[i] < 0:
                out["unreachable"][i] = True
                continue
            if sides[i] > 0:
                px, size = self.ask_px[j[i]], self.ask_qty[j[i]]
            else:
                px, size = self.bid_px[j[i]], self.bid_qty[j[i]]
            left, cash, last = float(qty[i]), 0.0, math.nan
            for p, q in zip(px.tolist(), size.tolist(), strict=True):
                if q <= 0:
                    continue
                take = min(left, q)
                cash += take * p
                left -= take
                last = p
                if left <= _EPS * float(qty[i]):
                    left = 0.0
                    break
            if left > 0:
                out["unreachable"][i] = True
                if math.isfinite(last) and check_rules(rules) >= 3:
                    # what the file does not show costs a further spread beyond what it does
                    spread = float(self.ask_px[j[i]][0] - self.bid_px[j[i]][0])
                    worse = last + spread if sides[i] > 0 else last - spread
                    cash = cash + left * worse if math.isfinite(spread) else math.nan
                elif math.isfinite(last):
                    cash += left * last
                else:
                    cash = math.nan
            out["vwap"][i] = cash / float(qty[i]) if qty[i] else math.nan
            top = size[0] if len(size) else 0.0
            out["beyond"][i] = max(float(qty[i]) - float(top), 0.0)
        return out


@dataclass(frozen=True)
class DepthSource:
    """Where a declared depth file's rows are (``data.depth``): the vault's depth files of a
    run's segments up to the one a panel reads, in stamp order, their venue and levels.

    A panel built from a brief with ``data.depth`` carries one
    (:attr:`blinded.panel.Panel.depth_source`); :func:`depth_of` reads it for
    the window a pass runs on, so only the hedge leg ever reads the file.
    """

    paths: tuple[Path, ...]
    venue: str
    levels: int

    def rows(self, start_ns: int | None = None, end_ns: int | None = None) -> pl.DataFrame:
        """The rows stamped before ``end_ns``, from the last one at or before ``start_ns``."""
        return TouchSource(self.paths, self.venue, 1.0).rows(start_ns, end_ns)


def depth_of(panel: Panel, window: Window | None = None) -> Depth:
    """The hedge venue's depth the panel's brief declares, the rows a pass on ``window``
    reads: from the last at or before its first decision to five minutes after its end."""
    source = getattr(panel, "depth_source", None)
    if source is None:
        raise ValueError("an arbitrage needs the hedge venue's depth (data.depth); this panel "
                         "has none")  # fmt: skip
    lo = hi = None
    if window is not None:
        lo = int(window.start.value) - panel.step_ns
        hi = int(window.end.value) + panel.step_ns + _AFTER
    return Depth.from_rows(source.rows(lo, hi), source.levels)


def unpriced(mark: Mark, stamps: np.ndarray, sides: np.ndarray, rules: int) -> np.ndarray:
    """The price of hedges the depth file cannot price (none of it yet, or an empty side).

    Under rules 1 and 2 B's mid. Under rules 3 never the mid: B's far touch, a
    purchase at its ask and a sale at its bid, and a further spread beyond it,
    as past the depth's last level; where B's touch has no price either the
    pass cannot be priced and is refused, never given a free hedge.
    """
    if check_rules(rules) < 3:
        return mark.mid(stamps)
    bid, ask = mark.bid_at(stamps), mark.ask_at(stamps)
    spread = ask - bid
    out = np.where(np.asarray(sides) > 0, ask + spread, bid - spread)
    if len(out) and not np.isfinite(out).all():
        raise ValueError("a hedge falls where the hedge venue shows no price, in its depth or "
                         "its touch: the pass cannot be priced")  # fmt: skip
    return out


def hedge_stamps(fill_ns: np.ndarray, end_ns: int, response_ns: int, hedge_ns: int
                 ) -> np.ndarray:  # fmt: skip
    """When each fill's hedge reaches the hedge venue: sent once the fill is known, at its
    stamp plus the response latency, or at the window's end decision if that is sooner,
    and arriving ``hedge_ns`` later; never before the fill itself."""
    sent = np.minimum(fill_ns.astype(np.int64) + int(response_ns), int(end_ns))
    return np.maximum(fill_ns.astype(np.int64), sent + int(hedge_ns))


class StockGate:
    """The two-venue stock as an arbitrage strategy knows it at each decision, and the rule
    that withdraws a side it could not fund.

    The ledger is four numbers, each venue's coin and USDT. A fill on A moves
    A's (coin :math:`+s q`, USDT :math:`-s q p` less the maker fee) and its
    hedge moves B's (coin :math:`-s q`, USDT :math:`+s q\\,\\text{vwap}` less
    the hedge fee), fees in USD at :math:`P_\\text{ref}` as every fee here is.
    A decision knows the fills stamped by :math:`g - L_r` and counts both
    legs of each, since its hedge is then sent. A bid rests only while B holds
    the coin its hedges would sell and A the USDT it would pay, for every unit
    of that side that could still fill unknown to the decision (the engine's
    reserve); an ask only while A holds the coin and B the USDT, at B's ask.
    Then the rebalance: when one venue holds more than ``rebalance_at`` of
    the two venues' coin, half the difference leaves it at once and reaches
    the other after the transfer time, less ``withdrawal_fee_coin`` units; the
    USDT likewise, less ``withdrawal_fee_usdt`` USDT under rules 3, and under
    rules 1 and 2 less the coin fee at B's mid. One transfer of each is in
    flight at a time.
    """

    def __init__(self, stock, quote: str, hedge: str, tick: float, mark: Mark,
                 depth: Depth, end_ns: int, response_ns: int, hedge_ns: int, ref: float,
                 maker_fee_bps: float, hedge_fee_bps: float, n: int,
                 rules: int = RULES):  # fmt: skip
        why = unset_fees(stock, coin_of(quote), rules)
        if why is not None:
            raise ValueError(why)
        self.rules = check_rules(rules)
        self.fee_usdt = stock.withdrawal_fee_usdt if self.rules >= 3 else None
        self.coin = [float(stock.coin[quote]), float(stock.coin[hedge])]
        self.usdt = [float(stock.usdt[quote]), float(stock.usdt[hedge])]
        self.at, self.fee_coin = stock.rebalance_at, float(stock.withdrawal_fee_coin)
        self.transfer_ns = int(stock.transfer.value)
        self.tick, self.mark, self.depth = tick, mark, depth
        self.end_ns, self.lr, self.lb = int(end_ns), int(response_ns), int(hedge_ns)
        # a fee a unit at P_ref under rules 1 and 2; under rules 3 a fraction of the notional
        self.notional = self.rules >= 3
        scale = 1.0 if self.notional else ref
        self.fee_a = maker_fee_bps * 1e-4 * scale
        self.fee_b = hedge_fee_bps * 1e-4 * scale
        self.k = 0
        self.flight: list = []  # (arrival, currency, venue it arrives at, amount)
        self.transfers: list[dict] = []
        self.withdrawn = np.zeros(n, dtype=np.int8)  # 1 the bid, 2 the ask, at each decision

    def __call__(self, k, td, lim, kf, fills, want_b, want_a, rb, ra):
        f_ns, f_side, f_px, f_qty = fills
        while self.k < kf:
            i = self.k
            s, q, p = f_side[i], f_qty[i], f_px[i] * self.tick
            self.coin[0] += s * q
            self.usdt[0] -= s * q * p + q * self.fee_a * (p if self.notional else 1.0)
            (t_h,) = hedge_stamps(np.array([f_ns[i]]), self.end_ns, self.lr, self.lb)
            walked = self.depth.walk(np.array([t_h]), np.array([-s]), np.array([q]), self.rules)
            got = walked["vwap"][0]
            if not math.isfinite(got):
                got = float(unpriced(self.mark, np.array([t_h]), np.array([-s]), self.rules)[0])
            self.coin[1] -= s * q
            self.usdt[1] += s * q * got - q * self.fee_b * (got if self.notional else 1.0)
            self.k += 1
        while self.flight and self.flight[0][0] <= td:
            _, what, to, amount = self.flight.pop(0)
            (self.coin if what == "coin" else self.usdt)[to] += amount
        self._rebalance(td)
        ok_b = want_b is None or (
            self.coin[1] >= rb - _EPS and self.usdt[0] >= rb * want_b * self.tick - _EPS
        )
        ok_a = True
        if want_a is not None:
            ask_b = float(self.mark.ask_at(np.array([td]))[0])
            ok_a = self.coin[0] >= ra - _EPS and (
                math.isfinite(ask_b) and self.usdt[1] >= ra * ask_b - _EPS
            )
        self.withdrawn[k] = (not ok_b) + 2 * (not ok_a)
        return ok_b, ok_a

    def _rebalance(self, td: int) -> None:
        busy = {f[1] for f in self.flight}
        mid = float(self.mark.mid(np.array([td]))[0])
        for what, held in (("coin", self.coin), ("usdt", self.usdt)):
            total = held[0] + held[1]
            if what in busy or total <= 0 or max(held) <= self.at * total:
                continue
            frm = 0 if held[0] > held[1] else 1
            amount = (held[frm] - held[1 - frm]) / 2
            if what == "coin":
                fee, fee_usd = self.fee_coin, self.fee_coin * mid
            elif self.fee_usdt is not None:
                fee = fee_usd = self.fee_usdt
            else:
                fee = fee_usd = self.fee_coin * mid  # rules 1 and 2
            held[frm] -= amount
            self.flight.append((td + self.transfer_ns, what, 1 - frm, amount - fee))
            self.transfers.append({"sent_ns": int(td), "arrives_ns": int(td + self.transfer_ns),
                                   "currency": what, "from": frm, "amount": float(amount),
                                   "fee_usd": float(fee_usd)})  # fmt: skip


def coin_of(instrument: str) -> str:
    """The coin of an instrument named ``COINUSDT_VENUE``: ``ATUSDT_MEXC`` is ATUSDT."""
    return instrument.split("_")[0]


@dataclass
class ArbSim(MakerSim):
    """One closed-loop pass of an arbitrage strategy: the maker leg on A and every fill's hedge
    on B, under one fill model.

    The per-step arrays are :class:`MakerSim`'s, marked to B's mid for both
    venues: a maker fill belongs to the step of its stamp :math:`t_f`, its
    hedge to the step of its arrival :math:`t_h`, and ``position`` is the net
    coin the two venues hold beyond the stock, :math:`s_f q_f` from
    :math:`t_f` to :math:`t_h`. So ``gross`` telescopes to the cash of both
    legs and that position at B's last mid; ``spread_bps`` holds what each
    leg paid against B's mid at its own stamp (the maker leg's capture,
    negative, and the hedge's slippage), ``mid_pnl`` the delay. ``units`` are
    the maker fills' units (A's maker fee), ``taker_units`` the hedges' (the
    hedge fee), and ``transfer_fee`` each step's rebalance fees, in
    ``fee_paid`` too. ``hedges`` holds each fill's hedge (:func:`arb_fills`
    reads it), ``withdrawn`` each decision's sides withdrawn for stock and
    ``transfers`` the rebalances booked.
    """

    hedge_fee_bps: float = 0.0
    transfer_fee: np.ndarray = field(default_factory=lambda: np.array([]))
    hedges: dict = field(default_factory=dict)
    mark: Mark | None = None
    withdrawn: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    transfers: list = field(default_factory=list)
    through: str = "print"
    quote: str = ""
    hedge: str = ""
    stock: object = None

    def metrics(self) -> dict:
        """:meth:`MakerSim.metrics` and the arbitrage's own: units
        hedged, what each leg captured, the delay and the slippage in bps of one unit, the
        shares of fills picked off, with B's touch already through the price, and hedged
        past B's depth, the rebalances and their fees, and the share of decisions with a side
        withdrawn for stock."""
        out = super().metrics()
        f = arb_fills(self)
        n = len(f["side"])
        units = float(self.units.sum())
        out |= {
            "hedged_units": float(self.taker_units.sum()),
            "captured_bps": float(np.sum(f["captured_usd"]) / self.ref * 1e4),
            "delay_bps": float(np.sum(f["delay_usd"]) / self.ref * 1e4),
            "slippage_bps": float(np.sum(f["slippage_usd"]) / self.ref * 1e4),
            "picked_off_share": float(np.mean(f["captured_usd"] < 0)) if n else math.nan,
            "touch_through_share": float(np.mean(f["touch_through"])) if n else math.nan,
            "unreachable_share": float(np.mean(f["unreachable"])) if n else math.nan,
            "rebalances": len(self.transfers),
            "rebalance_fee_bps": float(self.transfer_fee.sum()),
            "stock_withdrawn_share": float(np.mean(self.withdrawn > 0))
            if len(self.withdrawn) else math.nan,
            "net_bps_per_unit": float(self.net.sum()) / units if units else math.nan,
        }  # fmt: skip
        for key in ("flatten_units", "inventory_bps", "adverse_bps_10s"):
            out.pop(key, None)
        return out

    def refee(self, maker_fee_bps: float, taker_fee_bps: float | None = None) -> ArbSim:
        """The same fills and hedges at other fees: A's maker fee and the hedge fee; the
        rebalance fees stay."""
        hedge_fee = self.hedge_fee_bps if taker_fee_bps is None else taker_fee_bps
        made, took = self.charged()
        fee = made * maker_fee_bps + took * hedge_fee + self.transfer_fee
        out = replace(self, fee_bps=maker_fee_bps, fee_paid=fee, maker_fee_bps=maker_fee_bps,
                      taker_fee_bps=hedge_fee, hedge_fee_bps=hedge_fee)  # fmt: skip
        out.blocks = _blocks(out.net, self.per_block, self.rules)
        out.run_net = _run_net(self.runs, maker_fee_bps, hedge_fee)
        return out


def simulate_arb(
    quotes: Quotes,
    panel: Panel,
    window: Window,
    quote: str,
    hedge: str,
    stock,
    model: str = "RA",
    tick: float | None = None,
    lot: float | None = None,
    latency: pd.Timedelta = taker._ZERO,
    cancel_latency: pd.Timedelta | None = None,
    response_latency: pd.Timedelta | None = None,
    hedge_latency: pd.Timedelta = taker._ZERO,
    maker_fee_bps: float = 0.0,
    hedge_fee_bps: float = 0.0,
    block: pd.Timedelta = taker._BLOCK,
    touch: Touch | None = None,
    depth: Depth | None = None,
    mark: Mark | None = None,
    through: str = "print",
    rules: int = RULES,
) -> ArbSim:
    """Rest ``quotes`` on ``quote`` (venue A) and hedge every fill on ``hedge`` (venue B), under
    one fill model, and account for both legs.

    The engine (:func:`run_engine`) runs on A's touch with the print cap
    (``through``) and the stock rule (:class:`StockGate`). For each fill at
    :math:`t_f` the hedge, the opposite side for the fill's quantity, is sent
    at :math:`t_f + L_r` and reaches B at :math:`t_h = t_f + L_r + L_B`
    (:func:`hedge_stamps`; a fill known after the window's end decision is
    hedged from that decision), and walks B's depth as-of :math:`t_h`
    (:meth:`Depth.walk`). Fills are hedged one by one, never netted. The maker
    fill pays ``maker_fee_bps`` and the hedge ``hedge_fee_bps``, both on the
    units at :math:`P_\\text{ref}`, B's mid as-of the first decision plus
    :math:`L`; both legs are marked to B's mid (:class:`ArbSim`).
    """
    from blinded.contract import values

    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [window.start.value, window.end.value])
    if i1 - i0 < 1:
        raise ValueError("the window holds no grid points")
    if not all(isinstance(v, np.ndarray) and len(v) == len(g) for v in quotes.values()):
        quotes = values(quotes, panel)
    if touch is None:
        touch = touch_of(panel, quote, tick, window)
    if depth is None:
        depth = depth_of(panel, window)
    if mark is None:
        mark = mark_of(panel, hedge)
    G = g[i0:i1]
    end = int(G[-1] + panel.step_ns)
    params = Quotes({k: np.asarray(v, dtype=float)[i0:i1] for k, v in quotes.items()})
    lat = latency.value
    lc = lat if cancel_latency is None else cancel_latency.value
    lr = lat if response_latency is None else response_latency.value
    lb = int(hedge_latency.value)
    m0 = mark.mid(np.array([int(G[0]) + lat]))[0]
    ref = float(m0) if np.isfinite(m0) else float(np.nanmean(mark.mid(G)))
    gate = StockGate(stock, quote, hedge, touch.tick, mark, depth, end, lr, lb, ref,
                     maker_fee_bps, hedge_fee_bps, len(G), rules)  # fmt: skip
    book = run_engine(touch, Decisions.make(params, G, touch, lot), end, model, lat, lc, lr,
                      rules=rules, through=through, gate=gate)  # fmt: skip
    return account_arb(book, touch, mark, depth, G, end, panel.step_ns, latency, lr, lb, model,
                       maker_fee_bps, hedge_fee_bps, ref, gate, block, rules, through, quote,
                       hedge, stock)  # fmt: skip


def account_arb(book: Book, touch: Touch, mark: Mark, depth: Depth, grid_ns: np.ndarray,
                end_ns: int, step_ns: int, latency: pd.Timedelta, response_ns: int,
                hedge_ns: int, model: str, maker_fee_bps: float, hedge_fee_bps: float,
                ref: float, gate: StockGate, block: pd.Timedelta = taker._BLOCK,
                rules: int = RULES, through: str = "print", quote: str = "", hedge: str = "",
                stock=None) -> ArbSim:  # fmt: skip
    """The per-step P&L of one arbitrage pass (:class:`ArbSim`): both legs as trades marked to
    B's mid, the steps :class:`MakerSim`'s, the last one ending at the later of the end
    decision plus :math:`L` and plus :math:`L_B`, so that every hedge falls in it."""
    lat = latency.value
    t = grid_ns.astype(np.int64) + lat
    t_end = int(end_ns) + max(lat, int(hedge_ns))
    t_next = np.concatenate([t[1:], [t_end]])
    n = len(t)
    fn, fs, fq = book.fill_ns, book.fill_side.astype(float), book.fill_qty
    fp = book.fill_price * touch.tick
    th = hedge_stamps(fn, end_ns, response_ns, hedge_ns)
    walked = depth.walk(th, -fs, fq, rules)
    missing = ~np.isfinite(walked["vwap"])
    vwap = np.where(missing, unpriced(mark, th, -fs, rules), walked["vwap"])
    # every trade of both legs, in stamp order: the maker fills (side s at the fill's price)
    # and the hedges (side -s at the walk's price)
    ns = np.concatenate([fn, th])
    signed = np.concatenate([fs * fq, -fs * fq])
    price = np.concatenate([fp, vwap])
    maker = np.concatenate([np.ones(len(fn), bool), np.zeros(len(fn), bool)])
    order = np.argsort(ns, kind="stable")
    ns, signed, price, maker = ns[order], signed[order], price[order], maker[order]
    qty = np.abs(signed)
    m = mark.mid(t)
    m_next = mark.mid(t_next)
    at = mark.mid(ns)
    i = np.searchsorted(t, ns, side="left") - 1
    step = np.maximum(i, 0)
    after = np.round(np.cumsum(signed), 9)
    known = np.searchsorted(ns, t, side="right") - 1
    h = np.where(known >= 0, after[np.maximum(known, 0)] if len(after) else 0.0, 0.0)

    def per_step(weights: np.ndarray, where=None) -> np.ndarray:
        keep = np.ones(len(ns), dtype=bool) if where is None else where
        return np.bincount(step[keep], weights=weights[keep], minlength=n)[:n]

    intra = np.where(i >= 0, signed * (m_next[step] - at), 0.0)
    mid_pnl = np.nan_to_num(h * (m_next - m) + per_step(intra)) / ref * 1e4
    edge = signed * (at - price)
    spread = -np.nan_to_num(per_step(edge)) / ref * 1e4
    units = per_step(qty, maker)
    hedged = per_step(qty, ~maker)
    sent = np.array([x["sent_ns"] for x in gate.transfers], dtype=np.int64)
    fee_usd = np.array([x["fee_usd"] for x in gate.transfers], dtype=float)
    tstep = np.maximum(np.searchsorted(t, sent, side="left") - 1, 0)
    transfer = np.bincount(tstep, weights=fee_usd / ref * 1e4, minlength=n)[:n]
    fee_units = fee_hedged = None
    notional = check_rules(rules) >= 3
    if notional:
        # both legs' fees on their own notional: the fill's price, the hedge's walk
        worth = qty * price / ref
        fee_units, fee_hedged = per_step(worth, maker), per_step(worth, ~maker)
        fee = fee_units * maker_fee_bps + fee_hedged * hedge_fee_bps + transfer
    else:
        fee = units * maker_fee_bps + hedged * hedge_fee_bps + transfer
    per_block = max(int(block.value // step_ns), 1)
    sim = ArbSim(
        latency=latency,
        fee_bps=maker_fee_bps,
        hours=n * step_ns / taker.HOUR.value,
        position=h,
        trade=per_step(signed),
        mid_pnl=mid_pnl,
        spread_bps=spread,
        fee_paid=fee,
        fills=np.bincount(step[maker], minlength=n)[:n],
        blocks=np.array([]),
        blocks_per_hour=taker.HOUR.value / (per_block * step_ns),
        rules=rules,
        model=model,
        book=book,
        touch=touch,
        ref=ref,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=hedge_fee_bps,
        units=units,
        taker_units=hedged,
        captured=np.nan_to_num(per_step(edge, maker)) / ref * 1e4,
        adverse=0.0,
        flatten_units=0.0,
        per_block=per_block,
        step_ns=int(step_ns),
        start_ns=int(t[0]),
        hedge_fee_bps=hedge_fee_bps,
        transfer_fee=transfer,
        fee_units=fee_units,
        fee_taker_units=fee_hedged,
        hedges={
            "ns": th,
            "vwap": vwap,
            "unreachable": walked["unreachable"],
            "beyond": walked["beyond"],
            "unpriced": missing,
        },  # fmt: skip
        mark=mark,
        withdrawn=gate.withdrawn,
        transfers=list(gate.transfers),
        through=through,
        quote=quote,
        hedge=hedge,
        stock=stock,
    )
    sim.blocks = _blocks(sim.net, per_block, rules)
    carried = np.zeros(n)
    sim.runs = _runs(ns, signed, price, maker, ref, t, carried, notional)
    sim.run_net = _run_net(sim.runs, maker_fee_bps, hedge_fee_bps)
    return sim


def arb_fills(s: ArbSim) -> dict:
    """Each fill of an arbitrage pass and its hedge, decomposed.

    In USD of the fill: ``captured_usd`` :math:`s q (m_B(t_f) - p)`, the
    maker leg against B's mid at the fill; ``delay_usd`` :math:`s q (m_B(t_h)
    - m_B(t_f))`, B's move while the fill waited for its hedge;
    ``slippage_usd`` :math:`s q (\\text{vwap} - m_B(t_h))`, the hedge against
    B's mid at its arrival; ``fee_usd`` both legs' fees at :math:`P_\\text{ref}`;
    ``net_usd`` their sum. In bps of the fill's notional at :math:`m_B(t_f)`,
    the same under ``_bps``. ``picked_off`` is captured below zero (B's mid
    had already passed the price); ``touch_through`` is B's hedge-side touch at
    the fill already past the price; ``visible`` is that touch better than the
    price by more than both fees; ``reached`` such a cross still there for the
    hedge's walk.
    """
    b, mark = s.book, s.mark
    side, qty = b.fill_side.astype(float), b.fill_qty
    price = b.fill_price * s.touch.tick
    tf, th, vwap = b.fill_ns, s.hedges.get("ns", np.zeros(0)), s.hedges.get("vwap", np.zeros(0))
    mid_f, mid_h = mark.mid(tf), mark.mid(th)
    hedge_touch = np.where(side > 0, mark.bid_at(tf), mark.ask_at(tf))
    fee = qty * (s.maker_fee_bps + s.hedge_fee_bps) * 1e-4 * s.ref
    captured = side * qty * (mid_f - price)
    delay = side * qty * (mid_h - mid_f)
    slip = side * qty * (vwap - mid_h)
    notional = qty * mid_f
    fees_bps = s.maker_fee_bps + s.hedge_fee_bps

    def bps(x):
        return np.where(notional > 0, x / np.where(notional > 0, notional, 1.0) * 1e4, np.nan)

    return {
        "ns": tf, "hedge_ns": th, "side": side, "qty": qty, "price": price, "vwap": vwap,
        "mid_fill": mid_f, "mid_hedge": mid_h,
        "captured_usd": captured, "delay_usd": delay, "slippage_usd": slip, "fee_usd": fee,
        "net_usd": captured + delay + slip - fee,
        "captured_bps": bps(captured), "delay_bps": bps(delay), "slippage_bps": bps(slip),
        "fee_bps": bps(fee), "net_bps": bps(captured + delay + slip - fee),
        "picked_off": captured < 0,
        "touch_through": side * (hedge_touch - price) < 0,
        "visible": side * (hedge_touch - price) / mid_f * 1e4 > fees_bps,
        "reached": side * (vwap - price) / mid_f * 1e4 > fees_bps,
        "unreachable": s.hedges.get("unreachable", np.zeros(0, dtype=bool)),
        "beyond": s.hedges.get("beyond", np.zeros(0)),
        "through": b.fill_through, "crossed": b.fill_crossed,
    }  # fmt: skip


def transfer_terms(s: ArbSim) -> dict:
    """The rebalancing transfers of one arbitrage pass: how many, how many a day, and the fees
    they paid in USD, for the break-even fee a transfer the report prints."""
    days = s.hours / 24 if s.hours else float("nan")
    fee = float(np.sum([x["fee_usd"] for x in s.transfers]))
    return {"count": len(s.transfers), "a_day": len(s.transfers) / days if days else None,
            "fee_usd": fee}  # fmt: skip


def stock_path(s: ArbSim) -> dict:
    """The two venues' stock after every leg and transfer of a pass, as the ledger moved it:
    ``(stamp, what, coin A, USDT A, coin B, USDT B)`` rows in stamp order, the transfers in
    flight counted at neither venue until they arrive."""
    st, quote, hedge = s.stock, s.quote, s.hedge
    coin = [float(st.coin[quote]), float(st.coin[hedge])]
    usdt = [float(st.usdt[quote]), float(st.usdt[hedge])]
    f = arb_fills(s)
    fee_a, fee_b = s.maker_fee_bps * 1e-4 * s.ref, s.hedge_fee_bps * 1e-4 * s.ref
    if check_rules(s.rules) >= 3:  # on each leg's notional
        fee_a, fee_b = s.maker_fee_bps * 1e-4, s.hedge_fee_bps * 1e-4
    events = []
    for i in range(len(f["side"])):
        sd, q = f["side"][i], f["qty"][i]
        pa, pb = (f["price"][i], f["vwap"][i]) if check_rules(s.rules) >= 3 else (1.0, 1.0)
        cash_a = -sd * q * f["price"][i] - q * fee_a * pa
        cash_b = sd * q * f["vwap"][i] - q * fee_b * pb
        events.append((int(f["ns"][i]), 0, "fill", (0, sd * q, cash_a)))
        events.append((int(f["hedge_ns"][i]), 1, "hedge", (1, -sd * q, cash_b)))
    for x in s.transfers:
        frm, amount = x["from"], x["amount"]
        fee = s.stock.withdrawal_fee_coin if x["currency"] == "coin" else x["fee_usd"]
        dc, du = (amount, 0.0) if x["currency"] == "coin" else (0.0, amount)
        events.append((x["sent_ns"], 2, "sent", (frm, -dc, -du)))
        get = (amount - fee, 0.0) if x["currency"] == "coin" else (0.0, amount - fee)
        events.append((x["arrives_ns"], 3, "arrived", (1 - frm, get[0], get[1])))
    events.sort(key=lambda e: (e[0], e[1]))
    rows = []
    for stamp, _, what, (venue, dc, du) in events:
        coin[venue] += dc
        usdt[venue] += du
        rows.append((stamp, what, coin[0], usdt[0], coin[1], usdt[1]))
    return {"rows": rows, "coin": coin, "usdt": usdt}


def hedge_terms(brief: Brief, hedge: str) -> tuple[pd.Timedelta, float]:
    """The hedge leg's latency and taker fee: the brief's ``hedge_latency_ms`` and
    ``hedge_fee_bps``, else the order latency and the hedge instrument's taker fee."""
    ex = brief.execution
    lat = ex.latency if ex.hedge_latency is None else ex.hedge_latency
    fee = ex.fee(hedge) if ex.hedge_fee_bps is None else ex.hedge_fee_bps
    return lat, fee


def arb_models(quotes: Quotes, panel: Panel, window: Window, brief: Brief, quote: str,
               hedge: str, touch: Touch | None = None, depth: Depth | None = None,
               mark: Mark | None = None, which: tuple[str, ...] = MODELS,
               zero: bool = False, slow: bool = False, through: str = "print"
               ) -> dict[str, ArbSim]:  # fmt: skip
    """The arbitrage passes of ``which`` models at the brief's latencies and fees, the print
    cap on; with ``zero`` every latency 0, the hedge's too; with ``slow`` the hedge reaching
    B :data:`HEDGE_SLOW` after its fill (the ``hedge 1 s`` row: B's order latency is that
    less the response latency)."""
    ex = brief.execution
    if brief.stock is None:
        raise ValueError("an arbitrage needs the brief's stock block: each venue's coin and "
                         "USDT")  # fmt: skip
    if touch is None:
        touch = touch_of(panel, quote, ex.tick_size, window)
    if depth is None:
        depth = depth_of(panel, window)
    if mark is None:
        mark = mark_of(panel, hedge)
    if not all(isinstance(v, np.ndarray) and len(v) == len(panel) for v in quotes.values()):
        from blinded.contract import values

        quotes = values(quotes, panel)
    hedge_lat, hedge_fee = hedge_terms(brief, hedge)
    if zero:
        lat = dict(latency=taker._ZERO, cancel_latency=None, response_latency=None,
                   hedge_latency=taker._ZERO)  # fmt: skip
    else:
        lat = dict(latency=ex.latency, cancel_latency=ex.cancel, response_latency=ex.response,
                   hedge_latency=hedge_lat)  # fmt: skip
        if slow:
            lat["hedge_latency"] = max(HEDGE_SLOW - ex.response, taker._ZERO)
    return {m: simulate_arb(quotes, panel, window, quote, hedge, brief.stock, m, touch.tick,
                            ex.lot_size, maker_fee_bps=ex.maker_fee_bps, hedge_fee_bps=hedge_fee,
                            block=brief.stats.block, touch=touch, depth=depth, mark=mark,
                            through=through, rules=brief.judge_rules, **lat)
            for m in which}  # fmt: skip


ARB_NAMES = NAMES | {"RA uncapped": "RA with a trade through filling the whole order (the "
                                    "default rule): capacity, never scored"}  # fmt: skip


def arb_report(quotes: Quotes, panel: Panel, window: Window, brief: Brief, quote: str,
               hedge: str, touch: Touch | None = None, depth: Depth | None = None,
               mark: Mark | None = None, sims: dict[str, ArbSim] | None = None,
               scored: str | None = None, extra: dict | None = None
               ) -> pl.DataFrame:  # fmt: skip
    """An arbitrage strategy's sim table.

    ``brief`` is the worse of RA and PR (:func:`worse`); ``RA``, ``PR``,
    ``T`` and ``F`` the four models with through fills capped at the print;
    ``RA uncapped`` RA with the default rule, the capacity row; ``latency 0`` the
    scored model with every latency 0; ``hedge 1 s`` the scored model with its
    hedge reaching B a second after each fill; and ``hedge fee 10`` to
    ``hedge fee 0`` the scored model's own fills and hedges at those hedge
    fees, A's maker fee 0 (:meth:`ArbSim.refee`). ``extra``, when given,
    holds passes already made of the scored model under ``latency 0``,
    ``hedge 1 s`` and ``RA uncapped``.
    """
    ex = brief.execution
    if touch is None:
        touch = touch_of(panel, quote, ex.tick_size, window)
    if depth is None:
        depth = depth_of(panel, window)
    if mark is None:
        mark = mark_of(panel, hedge)
    common = dict(touch=touch, depth=depth, mark=mark)
    sims = sims or arb_models(quotes, panel, window, brief, quote, hedge, **common)
    s = worse(sims["RA"], sims["PR"]) if scored is None else sims[scored]
    extra = dict(extra or {})
    if "latency 0" not in extra:
        (extra["latency 0"],) = arb_models(
            quotes, panel, window, brief, quote, hedge, which=(s.model,), zero=True, **common
        ).values()
    if "hedge 1 s" not in extra:
        (extra["hedge 1 s"],) = arb_models(
            quotes, panel, window, brief, quote, hedge, which=(s.model,), slow=True, **common
        ).values()
    if "RA uncapped" not in extra:
        (extra["RA uncapped"],) = arb_models(
            quotes, panel, window, brief, quote, hedge, which=("RA",), through="full", **common
        ).values()
    rows = [("brief", s, "the worse of the two queue models by the Sharpe the judge scores, on "
                         f"blocks as long as its holds: {s.model}")]  # fmt: skip
    rows += [(m, sims[m], NAMES[m]) for m in MODELS]
    rows.append(("RA uncapped", extra["RA uncapped"], ARB_NAMES["RA uncapped"]))
    rows.append(("latency 0", extra["latency 0"], taker.UNREALISTIC))
    rows.append(
        (
            "hedge 1 s",
            extra["hedge 1 s"],
            f"{s.model} with each hedge reaching the hedge venue 1 s after its fill",
        )
    )
    for name, fee in HEDGE_FEE_ROWS:
        rows.append(
            (
                name,
                s.refee(0.0, fee),
                f"{s.model}'s fills and hedges at this hedge fee, the quote venue's maker fee 0",
            )
        )
    table = [{"row": name, "model": x.model, **x.metrics(), "note": note}
             for name, x, note in rows]  # fmt: skip
    return pl.DataFrame(table, infer_schema_length=None)


DAY_HOURS = 24.0


def arb_table(s: ArbSim) -> dict:
    """One arbitrage pass in the probe's terms, the report's per-fill
    table: fills, units and USD traded; captured, delay, slippage, fee and net, each the mean
    over fills in bps of the fill's notional at B's mid; the net in USD over the window and a
    day at that rate; the shares picked off, with B's touch through and filled by a trade
    through the price; the buy share; the drift, USD of coin a day moved onto the quote venue
    (bought there and sold on the hedge venue); and the median wait from an order's arrival to
    its fill, in seconds."""
    f = arb_fills(s)
    n = len(f["side"])
    usd = f["qty"] * f["price"]
    scale = DAY_HOURS / s.hours if s.hours else math.nan

    def mean(key):
        return float(np.nanmean(f[key])) if n else None

    b = s.book
    wait = (b.fill_ns - b.order_arrival[b.fill_order]) / 1e9 if n else np.zeros(0)
    net_usd = float(s.net.sum()) * s.ref * 1e-4
    return {
        "fills": n, "units": float(f["qty"].sum()), "usd_traded": float(usd.sum()),
        "captured_bps": mean("captured_bps"), "delay_bps": mean("delay_bps"),
        "slippage_bps": mean("slippage_bps"), "fee_bps": mean("fee_bps"),
        "net_bps": mean("net_bps"), "net_usd": net_usd, "net_usd_a_day": net_usd * scale,
        "picked_off_share": float(np.mean(f["picked_off"])) if n else None,
        "touch_through_share": float(np.mean(f["touch_through"])) if n else None,
        "through_share": float(np.mean(f["through"])) if n else None,
        "buy_share": float(np.mean(f["side"] > 0)) if n else None,
        "drift_usd_a_day": float(np.sum(f["side"] * usd)) * scale,
        "wait_s_median": float(np.median(wait)) if n else None,
        "size_usd": float(np.median(f["qty"]) * s.ref) if n else None,
        "hours": float(s.hours),
    }  # fmt: skip


def arb_pnl_lines(s: ArbSim, points: int = 600) -> dict:
    """Where an arbitrage's P&L goes, cumulative by step, in USD: what the maker fills captured
    against B's mid, that plus the hedge delay, that plus the hedge's slippage (the gross), and
    the net after both legs' fees and the rebalances."""
    from blinded import markouts as mk

    f = arb_fills(s)
    n = len(s.net)
    t = s.start_ns + s.step_ns * np.arange(n)
    at = np.maximum(np.searchsorted(t, f["ns"], side="left") - 1, 0)
    hedged = np.maximum(np.searchsorted(t, f["hedge_ns"], side="left") - 1, 0)

    def cum(weights, steps):
        return np.cumsum(np.bincount(steps, weights=np.nan_to_num(weights), minlength=n)[:n])

    captured = cum(f["captured_usd"], at)
    delay = cum(f["delay_usd"], hedged)
    slip = cum(f["slippage_usd"], hedged)
    keep = mk.downsample(n, points)
    hours = keep * (s.step_ns / taker.HOUR.value)
    return {"hours": hours.tolist(), "captured": captured[keep].tolist(),
            "delayed": (captured + delay)[keep].tolist(),
            "gross": (captured + delay + slip)[keep].tolist(),
            "net": (np.cumsum(s.net) * s.ref * 1e-4)[keep].tolist()}  # fmt: skip


def arb_excerpt(s: ArbSim, span: pd.Timedelta = EXCERPT, points: int = 900) -> dict:
    """The ``span`` of the window with the most fills: both venues' touches and the resting
    quotes sampled at up to ``points`` step starts (:func:`excerpt`), every fill on A and
    hedge on B of a fill in the span: fills ``(seconds, side, price, units)``, hedges
    ``(seconds, side, vwap, units)`` with the side the hedge traded, one for one."""
    out = excerpt(s, span, points)
    start = out["start_ns"]
    at = start + (np.asarray(out["seconds"]) * 1e9).astype(np.int64)
    out["hedge_bid"] = [None if not np.isfinite(v) else float(v) for v in s.mark.bid_at(at)]
    out["hedge_ask"] = [None if not np.isfinite(v) else float(v) for v in s.mark.ask_at(at)]
    f = arb_fills(s)
    width = int(span.value)
    inside = (f["ns"] >= start) & (f["ns"] < start + width)
    out["hedges"] = [
        [(int(x) - start) / 1e9, int(-sd), float(p), float(q)]
        for x, sd, p, q in zip(
            f["hedge_ns"][inside],
            f["side"][inside],
            f["vwap"][inside],
            f["qty"][inside],
            strict=True,
        )
    ]
    out.pop("inventory", None)
    return out
