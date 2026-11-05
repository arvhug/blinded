"""Positions to fills to P&L: the taker simulator, its diagnostic rows, the maker proxy, and
the built-in strategies.

Units. A position of 1 is one unit of reference notional: the quantity worth
the mark price at the first scored decision. Every P&L number is in basis
points of that notional, so a long held from price :math:`a` to price
:math:`b` makes :math:`10^4 (b - a) / P_\\text{ref}` and a round trip
telescopes to exit minus entry. A fill of size :math:`|\\Delta|` pays
:math:`|\\Delta|` times the fee in bps. A strategy in two legs holds one
notional of each leg per unit, each priced at its own mark, and its P&L is
the sum of the legs'. A perpetual with a ``funding`` stream also pays or
collects funding at each settlement, and a short position in a spot
instrument with a ``borrow`` stream pays its rate every step, both in bps of
the same notional (:func:`simulate`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl

from blinded.brief import Brief, Window
from blinded.contract import Legs, Quotes
from blinded.durations import Span, polars_span
from blinded.features import cross
from blinded.features._expr import once
from blinded.ladder import RULES, block_sharpe, check_rules, sharpe_blocks
from blinded.panel import Events, Panel

HOUR = pd.Timedelta("1h")
_ZERO = pd.Timedelta(0)
_BLOCK = pd.Timedelta("5min")
UNREALISTIC = "unrealistic, diagnostic only"
UPPER_BOUND = "upper bound: front of the queue, no adverse selection; never in the Ladder"
# columns that only one row of :func:`report` has: the other rows leave them out of their
# summaries rather than carry a null, so a table without that row reads as it did before it
ONE_ROW = ("maker_share",)


def held(target: np.ndarray, min_rebalance: float = 0.0, max_position: float = 1.0) -> np.ndarray:
    """The positions actually held, from the positions a strategy asks for.

    NaN means flat, targets are clipped to ``+-max_position``, and a change
    smaller than ``min_rebalance`` from the position held is not traded.

    Examples
    --------
    >>> held(np.array([0, 0.05, 0.12, 0.15, 0.3, np.nan, 3.0]), min_rebalance=0.1).tolist()
    [0.0, 0.0, 0.12, 0.12, 0.3, 0.0, 1.0]
    """
    t = np.clip(
        np.nan_to_num(np.asarray(target, dtype=float), nan=0.0), -max_position, max_position
    )
    if min_rebalance <= 0:
        return t
    out = np.empty_like(t)
    h = 0.0
    for i, x in enumerate(t):
        if abs(x - h) >= min_rebalance - 1e-12:
            h = x
        out[i] = h
    return out


@dataclass
class Sim:
    """One simulation of one position series over one window."""

    latency: pd.Timedelta
    fee_bps: float
    hours: float
    position: np.ndarray
    trade: np.ndarray
    mid_pnl: np.ndarray
    spread_bps: np.ndarray
    fee_paid: np.ndarray
    fills: np.ndarray
    blocks: np.ndarray
    blocks_per_hour: float
    run_net: np.ndarray = field(default_factory=lambda: np.array([]))
    funding: np.ndarray | None = None
    borrow: np.ndarray | None = None
    rules: int = RULES
    # the units each step's fee is charged on: under rules 3 each trade's notional over
    # P_ref, the exit's included; None, under rules 1 and 2, for the units traded
    fee_units: np.ndarray | None = None

    @property
    def gross(self) -> np.ndarray:
        """Per-step P&L at the fill prices, before fees."""
        return self.mid_pnl - self.spread_bps

    @property
    def net(self) -> np.ndarray:
        """Per-step P&L after fees, funding and borrow included: neither is a fee or a price
        move.

        ``funding`` is None when no leg has a funding stream, and ``borrow``
        when no leg has a borrow stream, so every number of a file without them
        is what it was before they existed.
        """
        return self._carried(self.gross - self.fee_paid)

    @property
    def held_pnl(self) -> np.ndarray:
        """What holding the position made at each step: the mid P&L, plus funding and borrow.
        The round trips book them to the run open at that step."""
        return self._carried(self.mid_pnl)

    def _carried(self, x: np.ndarray) -> np.ndarray:
        """``x`` plus funding and borrow where the sim has them, ``x`` itself where not."""
        for carried in (self.funding, self.borrow):
            if carried is not None:
                x = x + carried
        return x

    def metrics(self) -> dict:
        """The per-hour summary: P&L, Sharpe, drawdown, fills, fees, turnover, hit rate."""
        net, gross = self.net.sum(), self.gross.sum()
        fills = int(self.fills.sum())
        cum = np.concatenate([[0.0], np.cumsum(self.net)])
        runs = self.run_net
        carried = {
            name: float(x.sum())
            for name, x in (("funding_bps", self.funding), ("borrow_bps", self.borrow))
            if x is not None
        }
        return {
            "latency_ms": self.latency / pd.Timedelta("1ms"),
            "fee_bps": self.fee_bps,
            "hours": self.hours,
            "net_bps_per_hour": net / self.hours,
            "gross_bps_per_hour": gross / self.hours,
            "sharpe_per_sqrt_hour": block_sharpe(self.blocks, self.blocks_per_hour),
            "max_drawdown_bps": float(np.max(np.maximum.accumulate(cum) - cum)),
            "fills": fills,
            "fees_bps": float(self.fee_paid.sum()),
            "spread_bps": float(self.spread_bps.sum()),
            **carried,
            "turnover_per_hour": float(np.abs(self.trade).sum()) / self.hours,
            "net_bps_per_fill": net / fills if fills else np.nan,
            "gross_bps_per_fill": gross / fills if fills else np.nan,
            "hit_rate": float(np.mean(runs > 0)) if len(runs) else np.nan,
            "round_trips": len(runs),
        }


@dataclass
class MultiSim(Sim):
    """A strategy in several legs: each leg simulated on its own book, the P&L added up.

    ``mid_pnl``, ``spread_bps``, ``fee_paid``, ``fills``, ``net`` and
    ``blocks`` are sums over the legs, and ``trade`` is the units traded in
    all of them. ``position`` is the first leg's: its runs of one sign are the
    round trips, each charged with every leg's P&L and costs. ``fee_bps`` is
    the mean of the legs' fees and ``legs`` keeps each leg's own :class:`Sim`.
    """

    legs: dict[str, Sim] = field(default_factory=dict)

    def metrics(self) -> dict:
        """:meth:`Sim.metrics`, plus the mean gross exposure and the fills of each leg."""
        exposure = sum(np.abs(s.position) for s in self.legs.values())
        out = super().metrics() | {"gross_exposure": float(np.mean(exposure))}
        return out | {f"fills_{leg}": int(s.fills.sum()) for leg, s in self.legs.items()}


def simulate(
    position: np.ndarray,
    panel: Panel,
    window: Window,
    instrument: str,
    price: str = "mid",
    latency: pd.Timedelta = _ZERO,
    fee_bps: float = 0.0,
    min_rebalance: float = 0.0,
    max_position: float = 1.0,
    block: pd.Timedelta = _BLOCK,
    funding: Events | None = None,
    borrow: Events | None = None,
    rules: int = RULES,
) -> Sim:
    """Trade a position series as a taker and mark it to ``price``.

    ``position[i]`` is what the strategy wants to hold after deciding at grid
    point :math:`g_i`; only points inside ``window`` trade, starting flat. A
    change :math:`\\Delta_i` fills at :math:`t_i = g_i + L`, buying at the ask
    and selling at the bid as-of :math:`t_i`, read from the events rather than
    the grid. The position is then held until :math:`t_{i+1}`. A change from
    long to short, or back, is two fills: one closing, one opening. The fill
    times are stamps, so with an availability delay :math:`d` the book at
    :math:`t_i` is the one available by :math:`t_i + d`.

    ``funding`` holds the instrument's settlements, ``rate`` at each stamp;
    by default the panel's ``funding`` stream of ``instrument``, when it has
    one. At a settlement stamp :math:`f` from :math:`t_0` to the end of the
    last step, the position held is :math:`h_i` for the largest :math:`i`
    with :math:`t_i \\le f`, so a stamp exactly on a fill counts the position
    after it, and the accrual

    .. math::

        a_f = -h_i \\cdot \\text{rate}_f \\cdot 10^4 \\cdot m(f) / P_\\text{ref}

    bps is added to ``funding`` at step :math:`i`, with :math:`m(f)` the mark
    as-of :math:`f` (plus its delay): a long pays a positive rate, a short
    collects it. Without a funding stream ``funding`` is None.

    ``borrow`` holds what borrowing the coin costs, ``borrow_rate`` a day as
    a fraction at each published stamp; by default the panel's ``borrow``
    stream of ``instrument``, when it has one. A short position pays the rate
    :math:`b_i` as-of the stamp :math:`t_i` that starts its step, for as long
    as the step lasts:

    .. math::

        \\text{borrow}_i = -\\max(-h_i, 0)\\, b_i\\, 10^4\\,
        \\frac{t_{i+1} - t_i}{1\\,\\text{day}}\\, \\frac{m_i}{P_\\text{ref}}

    bps at step :math:`i`, the coins owed valued at the mark when the step
    starts. A long or flat step pays nothing. Without a borrow stream
    ``borrow`` is None.

    Under rules 3 (:data:`blinded.ladder.RULES`) a fee is charged on the trade's
    own notional, :math:`|\\Delta_i|\\, p_i / P_\text{ref}` units at the fill price
    :math:`p_i`, as the spread, funding and borrow already are, and a position
    still open when the window ends pays to leave: it is crossed at the far
    touch as-of the end, :math:`g_{\text{last}} + \text{step} + L`, its half
    spread and taker fee charged to the last step, as the maker engine's
    flatten is. Under rules 1 and 2 the fee was on the units traded at
    :math:`P_\text{ref}` and the last position was marked to the mid and left.
    """
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [window.start.value, window.end.value])
    if i1 - i0 < 1:
        raise ValueError("the window holds no grid points")
    h = held(np.asarray(position, dtype=float)[i0:i1], min_rebalance, max_position)
    t = g[i0:i1] + latency.value
    t_next = np.concatenate([g[i0 + 1 : i1], [g[i1 - 1] + panel.step_ns]]) + latency.value
    book, marks = panel.stream_of(instrument, "bid"), panel.stream_of(instrument, price)
    fill_at = t + book.delay_ns
    bid, ask = book.asof("bid", fill_at), book.asof("ask", fill_at)
    mark = marks.asof(price, t + marks.delay_ns)
    mark_next = marks.asof(price, t_next + marks.delay_ns)
    ref = mark[np.isfinite(mark)][0]

    prev = np.concatenate([[0.0], h[:-1]])
    trade = h - prev
    fill = np.where(trade > 0, ask, np.where(trade < 0, bid, mark))
    mid_pnl = np.nan_to_num(h * (mark_next - mark)) / ref * 1e4
    spread = np.nan_to_num(np.abs(trade) * np.abs(fill - mark)) / ref * 1e4
    fee = np.abs(trade) * fee_bps
    flips = (prev * h) < 0
    fills = (trade != 0).astype(int) + flips
    fee_units = None
    if check_rules(rules) >= 3:
        # each trade's fee on its own notional: the fill price over P_ref
        px = np.where(np.isfinite(fill), fill, np.where(np.isfinite(mark), mark, ref))
        fee_units = np.where(trade != 0, np.abs(trade) * px / ref, 0.0)
        last = float(h[-1])
        if last != 0:
            # what is still held at the end crosses at the far touch as-of the end plus L
            at = np.array([t_next[-1] + book.delay_ns])
            far = float((book.asof("bid", at) if last > 0 else book.asof("ask", at))[0])
            m_end = float(mark_next[-1])
            far = far if np.isfinite(far) else m_end
            spread = spread.copy()
            spread[-1] += float(np.nan_to_num(abs(last) * abs(far - m_end) / ref * 1e4))
            fee_units[-1] += abs(last) * (far if np.isfinite(far) else ref) / ref
            fills = fills.copy()
            fills[-1] += 1
        fee = fee_units * fee_bps
    if funding is None:
        funding = funding_of(panel, instrument)
    paid = None
    if funding is not None:
        paid = accrue(h, t, t_next[-1], funding, lambda at: marks.asof(price, at + marks.delay_ns),
                      ref)  # fmt: skip

    if borrow is None:
        borrow = borrow_of(panel, instrument)
    owed = None if borrow is None else borrowed(h, t, t_next, borrow, mark, ref)

    step = pd.Timedelta(panel.step_ns, unit="ns")
    n = len(h)
    per_block = max(int(block / step), 1)
    net = mid_pnl - spread - fee
    for carried in (paid, owed):
        if carried is not None:
            net = net + carried
    # the blocks of its Sharpe under the run's judging rules (blinded.ladder.RULES)
    blocks = sharpe_blocks(net, per_block, rules)
    sim = Sim(
        latency=latency,
        fee_bps=fee_bps,
        hours=n * step / HOUR,
        position=h,
        trade=trade,
        mid_pnl=mid_pnl,
        spread_bps=spread,
        fee_paid=fee,
        fills=fills,
        blocks=blocks,
        blocks_per_hour=HOUR / (per_block * step),
        funding=paid,
        borrow=owed,
        rules=rules,
        fee_units=fee_units,
    )
    sim.run_net = _round_trips(h, prev, [(prev, trade, sim.held_pnl, spread, fee)])
    return sim


def funding_of(panel: Panel, instrument: str) -> Events | None:
    """The ``funding`` stream of ``instrument``, or None when the panel has none."""
    try:
        return panel.events("funding", instrument)
    except KeyError:
        return None


def borrow_of(panel: Panel, instrument: str) -> Events | None:
    """The ``borrow`` stream of ``instrument``, or None when the panel has none."""
    try:
        return panel.events("borrow", instrument)
    except KeyError:
        return None


def accrue(h: np.ndarray, t: np.ndarray, end: int, funding: Events, mark, ref: float,
           held=None, each: bool = False):  # fmt: skip
    """The funding of each step, in bps of the reference notional ``ref``.

    ``h[i]`` is held from fill stamp ``t[i]`` to ``t[i + 1]``, the last to
    ``end``. Each settlement stamp :math:`f` in ``[t[0], end)`` books
    :math:`-h_i\\,\\text{rate}_f\\,10^4\\,m(f)/P_\\text{ref}` to the largest :math:`i` with
    :math:`t_i \\le f`; ``mark(stamps)`` gives :math:`m`. The stamp is the
    settlement's ``ts``, not when the rate became available: who pays is
    decided by the position held at the settlement. ``held(stamps)``, when
    given, is that position instead of :math:`h_i`: a maker's inventory moves
    inside a step, and it pays on what it holds at the stamp
    (:func:`blinded.sim_maker.simulate_maker`). With ``each``, the stamps and
    accruals of the settlements come back too, as ``(out, stamps, accruals)``.
    """
    out = np.zeros(len(h))
    f = funding.frame["ts"].dt.epoch("ns").to_numpy()
    rate = funding.column("rate")
    inside = (f >= t[0]) & (f < end) & np.isfinite(rate)
    f, rate = f[inside], rate[inside]
    a = np.zeros(0)
    if len(f):
        i = np.searchsorted(t, f, side="right") - 1
        a = np.nan_to_num(-(h[i] if held is None else held(f)) * rate * 1e4 * mark(f) / ref)
        np.add.at(out, i, a)
    return (out, f, a) if each else out


def borrowed(h: np.ndarray, t: np.ndarray, t_next: np.ndarray, borrow: Events, mark: np.ndarray,
             ref: float) -> np.ndarray:  # fmt: skip
    """What a short position pays to borrow the coin at each step, in bps of ``ref``.

    ``h[i]`` is held from ``t[i]`` to ``t_next[i]`` and ``mark[i]`` is the mark
    at ``t[i]``. A short pays the ``borrow_rate`` (a day, as a fraction) as-of
    the stamp that starts its step, for as long as the step lasts,

    .. math::

        -\\max(-h_i, 0)\\, b_i\\, 10^4\\, \\frac{t_{i+1} - t_i}{1\\,\\text{day}}\\,
        \\frac{m_i}{P_\\text{ref}},

    and a long or flat step pays nothing.
    """
    rate = borrow.asof("borrow_rate", t + borrow.delay_ns)
    days = (t_next - t) / pd.Timedelta("1D").value
    return -np.nan_to_num(np.maximum(-h, 0.0) * rate * 1e4 * days * mark / ref)


def simulate_legs(
    legs: Legs,
    panel: Panel,
    window: Window,
    brief: Brief,
    latency: pd.Timedelta | None = None,
    fees: dict[str, float] | None = None,
) -> MultiSim:
    """Trade each leg with :func:`simulate` and add the legs up.

    Each leg fills on its own book, is marked on its own ``target.price``,
    is priced in units of its own reference notional and pays its own fee:
    ``fees[leg]``, by default the brief's fee for that instrument. ``latency``
    defaults to the brief's.
    """
    ex = brief.execution
    latency = ex.latency if latency is None else latency
    fees = {leg: ex.fee(leg) for leg in legs} if fees is None else fees
    sims = {
        leg: simulate(pos, panel, window, leg, brief.target.price, latency, fees[leg],
                      ex.min_rebalance, ex.max_position, brief.stats.block,
                      rules=brief.judge_rules)
        for leg, pos in legs.items()
    }  # fmt: skip
    return _combine(sims)


def _combine(sims: dict[str, Sim]) -> MultiSim:
    first = next(iter(sims.values()))

    def total(name: str) -> np.ndarray:
        return sum(getattr(s, name) for s in sims.values())

    parts = []
    for s in sims.values():
        prev = np.concatenate([[0.0], s.position[:-1]])
        parts.append((prev, s.trade, s.held_pnl, s.spread_bps, s.fee_paid))
    lead_prev = np.concatenate([[0.0], first.position[:-1]])
    paid = [s.funding for s in sims.values() if s.funding is not None]
    owed = [s.borrow for s in sims.values() if s.borrow is not None]
    charged = [s.fee_units for s in sims.values() if s.fee_units is not None]
    return MultiSim(
        latency=first.latency,
        fee_bps=float(np.mean([s.fee_bps for s in sims.values()])),
        hours=first.hours,
        position=first.position,
        trade=sum(np.abs(s.trade) for s in sims.values()),
        mid_pnl=total("mid_pnl"),
        spread_bps=total("spread_bps"),
        fee_paid=total("fee_paid"),
        fills=total("fills"),
        blocks=total("blocks"),
        blocks_per_hour=first.blocks_per_hour,
        run_net=_round_trips(first.position, lead_prev, parts),
        funding=sum(paid) if paid else None,
        borrow=sum(owed) if owed else None,
        rules=first.rules,
        legs=dict(sims),
        fee_units=sum(charged) if len(charged) == len(sims) else None,
    )


def hold_steps(s: Sim) -> float:
    r"""The time-weighted mean length, in grid steps, of the runs of one sign of each leg's
    position: :math:`\sum L^2 / \sum L` over every run of every leg.

    It is the expected length of the hold that a random step in position belongs to,
    so one long hold dominates it as it dominates the P&L: ten 1-hour holds and one
    28-day hold give 662 hours, about 27.6 days, where the median would give 1 hour. 0 when the
    strategy is never in position.

    Examples
    --------
    >>> import numpy as np
    >>> run = lambda h: Sim(*[None] * 3, np.array(h, float), *[None] * 6, 1.0)
    >>> hold_steps(run([0, 1, 1, 1, 0, -1, 0, 0]))  # runs of 3 and 1: (9 + 1) / 4
    2.5
    >>> hold_steps(run([1, 1, -1, -1, -1, -1])), hold_steps(run([0, 0]))
    (3.3333333333333335, 0.0)
    """
    legs = s.legs.values() if isinstance(s, MultiSim) and s.legs else [s]
    total = squares = 0.0
    for leg in legs:
        sign = np.sign(np.nan_to_num(np.asarray(leg.position, dtype=float)))
        edges = np.flatnonzero(np.diff(sign) != 0) + 1
        bounds = np.concatenate([[0], edges, [len(sign)]])
        lengths = np.diff(bounds)[sign[bounds[:-1]] != 0] if len(sign) else np.array([])
        total += float(lengths.sum())
        squares += float((lengths.astype(float) ** 2).sum())
    return squares / total if total else 0.0


def _round_trips(h, prev, legs) -> np.ndarray:
    """Net P&L of each run of one sign of ``h``, entry and exit costs included.

    ``legs`` holds ``(prev, trade, mid_pnl, spread, fee)`` for each leg traded
    alongside ``h``: its P&L while a run is open goes to that run, and the cost
    of a trade to the run it opens or, for the part that reduces the leg, to
    the run it closes. An open run is marked.
    """
    sign, before = np.sign(h), np.sign(prev)
    starts = (sign != 0) & (sign != before)
    run = np.where(sign != 0, np.cumsum(starts) - 1, -1)
    if not starts.any():
        return np.array([])
    prev_run = np.concatenate([[-1], run[:-1]])
    total = np.zeros(int(starts.sum()))
    held_here = run >= 0
    for p, trade, mid_pnl, spread, fee in legs:
        cost_per_unit = np.divide(
            spread + fee, np.abs(trade), out=np.zeros_like(fee), where=trade != 0
        )
        was = np.sign(p)
        closing = (was != 0) & (np.sign(trade) == -was)
        close_qty = np.where(closing, np.minimum(np.abs(trade), np.abs(p)), 0.0)
        open_qty = np.abs(trade) - close_qty
        np.add.at(
            total, run[held_here], mid_pnl[held_here] - (open_qty * cost_per_unit)[held_here]
        )
        closed = (close_qty > 0) & (prev_run >= 0)
        np.add.at(total, prev_run[closed], -(close_qty * cost_per_unit)[closed])
    return total


def maker_proxy(
    sim: Sim,
    panel: Panel,
    window: Window,
    instrument: str,
    horizon: pd.Timedelta,
    price: str = "mid",
    maker_fee_bps: float = 0.0,
) -> dict:
    """What the same position changes would have earned as passive quotes, at best.

    A buy rests at the bid as-of :math:`t_i`, a sell at the ask. It fills with
    probability :math:`1 - e^{-\\lambda h}`, where :math:`\\lambda` is the
    window's rate of trades on the other side (sells for a resting buy, each
    counted ``n`` times when the stream has ``n``), and is marked at ``price``
    as-of :math:`t_i + h`. That assumes the order is at the front of the
    queue and that fills are no more likely when the price is about to move
    against it, so it is an upper bound.
    """
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [window.start.value, window.end.value])
    t = g[i0:i1] + sim.latency.value
    book, marks = panel.stream_of(instrument, "bid"), panel.stream_of(instrument, price)
    bid, ask = book.asof("bid", t + book.delay_ns), book.asof("ask", t + book.delay_ns)
    ref = marks.asof(price, t[:1] + marks.delay_ns)[0]
    later = marks.asof(price, t + horizon.value + marks.delay_ns)
    try:
        trades = panel.events("trades", instrument)
    except KeyError:
        return {}
    at = trades.at
    side = trades.column("side")
    count = trades.column("n") if "n" in trades.frame.columns else np.ones(len(trades))
    inside = (at >= window.start.value) & (at < window.end.value)
    seconds = sim.hours * 3600
    rate = {+1: count[inside & (side < 0)].sum() / seconds,
            -1: count[inside & (side > 0)].sum() / seconds}  # fmt: skip
    h_s = horizon / pd.Timedelta("1s")
    prob = {s: 1 - math.exp(-r * h_s) for s, r in rate.items()}
    buy, sell = sim.trade > 0, sim.trade < 0
    edge = np.where(buy, later - bid, np.where(sell, ask - later, 0.0)) / ref * 1e4
    p = np.where(buy, prob[+1], np.where(sell, prob[-1], 0.0))
    size = np.abs(sim.trade)
    expected = np.nansum(size * p * (edge - maker_fee_bps))
    traded = size.sum()
    per_fill = float(np.nansum(size * edge) / traded) - maker_fee_bps if traded else np.nan
    return {
        "latency_ms": sim.latency / pd.Timedelta("1ms"),
        "fee_bps": maker_fee_bps,
        "hours": sim.hours,
        "net_bps_per_hour": expected / sim.hours,
        "fills": float(np.sum(p[size > 0])),
        "net_bps_per_fill": per_fill,
        "fill_probability_buy": prob[+1],
        "fill_probability_sell": prob[-1],
    }


def report(
    position: np.ndarray | Legs,
    panel: Panel,
    window: Window,
    brief: Brief,
    instrument: str | None = None,
    hedge: str | None = None,
) -> pl.DataFrame:
    """The brief's row, the three unrealistic diagnostic rows and the maker proxy.

    The rows are ``brief`` (the brief's latency and taker fee), ``latency 0``,
    ``fees 0``, ``latency 0, fees 0`` and ``maker proxy``. One position series
    trades ``instrument``, the brief's target by default; where the file has a
    touch stream of it (:func:`blinded.sim_maker.has_touch`) a sixth row,
    ``maker-taker``, executes the same positions on the fill engine, posting
    each change at the touch and crossing what is left at the brief's
    deadline, under the worse of RA and PR (:func:`blinded.sim_maker.maker_taker`;
    diagnostic, never in the Ladder). :class:`Legs` go through
    :func:`simulate_legs`, each leg at its own fee, with the maker proxy added
    up over the legs. A market-making strategy's quotes
    (:class:`~blinded.contract.Quotes`) get the fill engine's rows instead,
    :func:`blinded.sim_maker.report`, and an arbitrage's quotes on ``instrument``
    hedged on ``hedge`` the arbitrage's, :func:`blinded.sim_maker.arb_report`.
    """
    if isinstance(position, Quotes):
        from blinded import sim_maker

        if hedge is not None:
            inst = instrument or brief.target.instrument
            return sim_maker.arb_report(position, panel, window, brief, inst, hedge)
        return sim_maker.report(position, panel, window, brief, instrument)
    ex = brief.execution
    zero = pd.Timedelta(0)
    if isinstance(position, Legs):
        fees, free = {leg: ex.fee(leg) for leg in position}, dict.fromkeys(position, 0.0)

        def run(latency, fee):
            return simulate_legs(position, panel, window, brief, latency, fee)

        def proxy(sim):
            return _maker_legs(sim, panel, window, brief)
    else:
        instrument = instrument or brief.target.instrument
        fees, free = ex.fee(instrument), 0.0
        common = dict(
            panel=panel,
            window=window,
            instrument=instrument,
            price=brief.target.price,
            min_rebalance=ex.min_rebalance,
            max_position=ex.max_position,
            block=brief.stats.block,
            rules=brief.judge_rules,
        )

        def run(latency, fee):
            return simulate(position, latency=latency, fee_bps=fee, **common)

        def proxy(sim):
            return maker_proxy(sim, panel, window, instrument, brief.target.primary_horizon,
                               brief.target.price, ex.maker_fee_bps)  # fmt: skip

    variants = [
        ("brief", ex.latency, fees, ""),
        ("latency 0", zero, fees, UNREALISTIC),
        ("fees 0", ex.latency, free, UNREALISTIC),
        ("latency 0, fees 0", zero, free, UNREALISTIC),
    ]
    rows, sims = [], []
    for name, latency, fee, note in variants:
        sims.append(run(latency, fee))
        rows.append({"row": name, **sims[-1].metrics(), "note": note})
    passive = proxy(sims[0])
    if passive:
        rows.append({"row": "maker proxy", **passive, "note": UPPER_BOUND})
    if not isinstance(position, Legs):
        from blinded import sim_maker

        if sim_maker.has_touch(brief, instrument):
            both = sim_maker.maker_taker(position, panel, window, brief, instrument)
            rows.append(sim_maker.maker_taker_row(both, ex.deadline))
    return pl.DataFrame(rows, infer_schema_length=None)


def _maker_legs(main: MultiSim, panel: Panel, window: Window, brief: Brief) -> dict:
    """The maker proxy of each leg, at that leg's own trade rate, added up."""
    t, fee = brief.target, brief.execution.maker_fee_bps
    parts = {leg: maker_proxy(s, panel, window, leg, t.primary_horizon, t.price, fee)
             for leg, s in main.legs.items()}  # fmt: skip
    if not all(parts.values()):
        return {}
    traded = {leg: float(np.abs(s.trade).sum()) for leg, s in main.legs.items()}
    edge = sum(
        (parts[leg]["net_bps_per_fill"] + fee) * traded[leg] for leg in parts if traded[leg]
    )
    units = sum(traded.values())
    return {
        "latency_ms": main.latency / pd.Timedelta("1ms"),
        "fee_bps": fee,
        "hours": main.hours,
        "net_bps_per_hour": sum(p["net_bps_per_hour"] for p in parts.values()),
        "fills": sum(p["fills"] for p in parts.values()),
        "net_bps_per_fill": edge / units - fee if units else np.nan,
    }


def walk_forward_scale(f: pl.Expr, every: Span = "30m") -> pl.Expr:
    """The standard deviation of ``f`` over every row before the current clock window.

    Windows are ``every`` long and aligned to the clock (``ts.dt.truncate``).
    The scale is refitted at the start of each window from all earlier rows,
    expanding, and held for the window, so no row's scale uses its own
    window. Null in the first window, and wherever the spread is zero.
    NaN counts as missing, like null: one NaN would otherwise poison every
    later sum.
    """
    return once(lambda f: _scale(f, every), f)


def linear(f: pl.Expr, max_position: float = 1.0, every: Span = "30m") -> pl.Expr:
    """Position proportional to the feature: ``clip(f / scale, +-max_position)``.

    ``scale`` is :func:`walk_forward_scale`, fitted on earlier windows only.
    A NaN feature gives a null position, as a null one does.
    """

    def position(f):
        f = f.fill_nan(None)
        return (f / _scale(f, every)).clip(-max_position, max_position)

    return once(position, f)


def threshold(f: pl.Expr, z_in: float = 2.0, z_out: float = 0.5, every: Span = "30m") -> pl.Expr:
    """Long or short one unit when ``f / scale`` passes ``+-z_in``; flat back inside ``+-z_out``.

    In between, the position is whatever it was, which is the hysteresis
    that keeps turnover down. ``scale`` is :func:`walk_forward_scale`.
    A missing or NaN feature leaves the position unchanged.
    """
    _check_band(z_in, z_out)

    def position(f):
        f = f.fill_nan(None)
        return _hysteresis(f / _scale(f, every), z_in, z_out)

    return once(position, f)


def _scale(f: pl.Expr, every: Span) -> pl.Expr:
    # f appears about thirty times below, so the public functions pass it in as a
    # column, through once
    f = f.fill_nan(None)
    c = f - f.drop_nulls().first()
    present = c.is_not_null().cast(pl.Int64).cum_sum()
    s1 = c.fill_null(0.0).cum_sum()
    s2 = (c * c).fill_null(0.0).cum_sum()
    var = ((s2 - s1 * s1 / present) / (present - 1)).clip(lower_bound=0.0)
    sd = pl.when(present > 1).then(var.sqrt())
    key = pl.col("ts").dt.truncate(polars_span(every))
    at_start = key != key.shift(1)
    scale = pl.when(at_start).then(sd.shift(1)).fill_null(strategy="forward")
    return pl.when(scale > 0).then(scale)


def pairs(
    a: str,
    b: str,
    z_in: float = 2.0,
    z_out: float = 0.5,
    window: Span = "2h",
    every: Span = "30m",
    max_position: float = 1.0,
) -> dict[str, pl.Expr]:
    """Short the spread of ``a`` over ``b`` when its z-score is above ``z_in``, long below
    ``-z_in``, flat back inside ``+-z_out``.

    The z-score is :func:`blinded.features.cross.spread_z`; ``a`` holds
    :math:`u = -\\operatorname{state}(z)` and ``b`` holds :math:`-\\tilde\\beta u`,
    with :math:`\\tilde\\beta` from :func:`blinded.features.cross.hedge_ratio`,
    both clipped to ``+-max_position``. Everything is fitted on the window
    before each clock window of ``every``, so ``b`` also trades when the ratio
    is refitted while a position is held. Return it from a strategy with
    ``family="pairs", pair=(a, b)``.
    """
    u = -_hysteresis(cross.spread_z(a, b, window, every), z_in, z_out)
    beta = cross.hedge_ratio(a, b, window, every)
    lim = (-max_position, max_position)
    return {a: u.clip(*lim), b: (-beta * u).clip(*lim)}


def carry(
    spot: str,
    perp: str,
    rate_in: float = 0.0001,
    rate_out: float = 0.0,
    max_position: float = 1.0,
) -> dict[str, pl.Expr]:
    """Long ``spot`` and short ``perp`` while the perpetual's last settled funding rate is
    high, the other way round while it is low, and flat in between.

    The rate is ``pl.col(f"{perp}.rate")``, the funding stream as-of the
    decision, so a decision reads the last settlement available by then.
    Long spot and short perp (``+max_position``, ``-max_position``) from a
    rate above ``rate_in`` until one below ``rate_out``; short spot and long
    perp from a rate below ``-rate_in`` until one above ``-rate_out``; else
    the position held, flat before any settlement. A short perpetual collects
    a positive rate. Return it from a strategy of ``bl.basis_trade``.

    Examples
    --------
    >>> import blinded as bl
    >>> legs = bl.carry("BTCUSDT", "BTCUSDT_PERP")
    >>> sorted(legs)
    ['BTCUSDT', 'BTCUSDT_PERP']
    """
    if not 0 <= rate_out <= rate_in:
        raise ValueError(f"need 0 <= rate_out <= rate_in, got rate_in={rate_in}, "
                         f"rate_out={rate_out}")  # fmt: skip
    rate = pl.col(f"{perp}.rate").fill_nan(None)

    def state(enter: pl.Expr, leave: pl.Expr) -> pl.Expr:
        on = pl.when(enter).then(1.0).when(leave).then(0.0).otherwise(None)
        return on.fill_null(strategy="forward").fill_null(0.0)

    # the two states cannot both be on: entering one is leaving the other, since
    # -rate_in <= -rate_out <= rate_out <= rate_in
    u = state(rate > rate_in, rate < rate_out) - state(rate < -rate_in, rate > -rate_out)
    return {spot: u * max_position, perp: 0.0 - u * max_position}  # 0.0 - u: no -0.0


def basket(
    score: Mapping[str, pl.Expr],
    top: int = 4,
    every: Span = "4h",
    max_position: float = 1.0,
) -> dict[str, pl.Expr]:
    """Long the ``top`` legs with the highest score and short the ``top`` with the lowest, held
    through each clock window of ``every``.

    At the start of each window (aligned as :func:`walk_forward_scale` and
    :func:`blinded.features.cross.hedge_ratio` align theirs), each leg's score
    is read at the row before the window start and the legs with a finite
    score are ranked; ties go in leg order, the earlier leg ranked lower. The
    top ``top`` hold ``+max_position / top`` and the bottom ``top`` hold
    ``-max_position / top`` through the window, the rest 0, so the gross
    exposure is ``2 max_position`` while in position. With fewer than ``2
    top`` finite scores the window is flat (null). Everything is read at a
    window start from earlier rows, so it is causal. Return it from a strategy
    of a basket archetype, ``bl.statistical_arbitrage`` or
    ``bl.short_term_reversal``.
    """
    from blinded.features.cross import _held
    from blinded.features.xs import Cross, as_list

    if top < 1:
        raise ValueError(f"top must be at least 1, got {top}")
    held = _held(as_list(score), every)
    e, w = pl.element(), max_position / top
    rank, n = e.rank("ordinal"), e.count()
    side = pl.when(rank > n - top).then(w).when(rank <= top).then(-w).otherwise(0.0)
    return Cross(score, held.list.eval(pl.when(n >= 2 * top).then(side)))


def _hysteresis(z: pl.Expr, z_in: float, z_out: float) -> pl.Expr:
    """1 above ``z_in``, -1 below ``-z_in``, 0 inside ``+-z_out``, else the last state."""
    _check_band(z_in, z_out)
    state = (
        pl.when(z > z_in)
        .then(1.0)
        .when(z < -z_in)
        .then(-1.0)
        .when(z.abs() < z_out)
        .then(0.0)
        .otherwise(None)
    )
    return state.fill_null(strategy="forward").fill_null(0.0)


def _check_band(z_in: float, z_out: float) -> None:
    if not 0 <= z_out <= z_in:
        raise ValueError(f"need 0 <= z_out <= z_in, got z_in={z_in}, z_out={z_out}")
