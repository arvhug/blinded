"""How an archetype dies: the numbers of the report's "How it dies" panel.

Each archetype names its death checks (``checks`` in
:data:`blinded.archetypes.REGISTRY`, titled in :data:`~blinded.archetypes.DEATHS`).
Every check is a function of the vault and the brief: :mod:`blinded.report`
computes it in ``recompute`` for the archetype's pick on each segment and
``blinded replay`` checks it. The arithmetic is here, pure numpy over a
simulation (:class:`~blinded.sim.Sim`) and the arrays it was built from, so each
number can be checked by hand:

- D1 ``costs``: the fee per unit traded that would eat the gross P&L, and the
  net P&L over a grid of fees and latencies;
- D2 ``race``: how much of a feature's information is left once the order
  fills, and what the order paid in flight;
- D3 ``capacity``: the share of the far touch one fill takes, and of the
  window's traded volume the strategy is;
- D4 ``worst_window``: the net P&L per block window, its worst, its
  concentration, the longest stretch under water and the drawdown;
- D5 ``half_life``: how long the signal, or a pair's spread, lasts;
- D6 ``mae``: how far each round trip went against it before it closed;
- D7 ``funding``: what funding paid or cost, beside what the basis did;
- D8 ``crowding``: what this data cannot answer at all;
- D9 ``breadth``: whether one leg of a basket carried it;
- D10 ``whipsaw``: whether a trend got chopped;
- D11 ``venue_feed``: whether a lead across venues is the second venue's feed
  schedule;
- D12 ``toxicity``: for a market maker, what its fills captured and gave back,
  and how they came;
- D13 ``queue``: for a market maker, where in the queue its orders sat.

A market-making strategy's D1 and D3 are their own (:func:`maker_costs`,
:func:`maker_capacity`): a quote pays the maker fee on every unit filled, and
its size is measured against the queue it joins.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from blinded import markouts as mk
from blinded import metrics, sim

FEES = (0.0, 1.0, 2.5, 5.0)  # the grid's fees in bps a unit traded, beside the brief's
NOT_MEASURABLE = "not measurable from this data"
BELOW_THE_BAR = "not measurable below the bar: the fill is the next bar's close"


# -- the arithmetic -------------------------------------------------------------


def breakeven_fee_bps(gross_bps: float, traded: float) -> float | None:
    """The fee per unit traded at which the gross P&L is all paid away.

    Examples
    --------
    The documented five-point book: +6 bps gross over four fills of one unit.

    >>> breakeven_fee_bps(6.0, 4.0)
    1.5
    """
    return float(gross_bps / traded) if traded > 0 else None


def touch_shares(sizes: np.ndarray, touch_qty: np.ndarray) -> dict:
    """Each fill's size over the size at the far touch as-of the fill: median, share above 1.

    Examples
    --------
    Fills of 1 and 3 against touches of 4 and 2:

    >>> out = touch_shares(np.array([1.0, 3.0]), np.array([4.0, 2.0]))
    >>> out["shares"], out["median"], out["above_one"]
    ([0.25, 1.5], 0.875, 0.5)
    """
    ok = np.isfinite(touch_qty) & (touch_qty > 0)
    share = np.abs(sizes[ok]) / touch_qty[ok]
    if not len(share):
        return {"shares": [], "median": None, "above_one": None, "n": 0}
    return {"shares": share.tolist(), "median": float(np.median(share)),
            "above_one": float(np.mean(share > 1)), "n": int(len(share))}  # fmt: skip


def window_nets(net: np.ndarray, grid_ns: np.ndarray, every_ns: int) -> list[tuple[int, float]]:
    """The net P&L summed inside each clock window of ``every_ns``, as ``(start ns, net)``.

    Windows are aligned as :func:`blinded.markouts.window_ics` aligns them.

    Examples
    --------
    >>> window_nets(np.array([4.0, 6.0, -4.0, 7.0]), np.array([0, 5, 10, 20]), 10)
    [(0, 10.0), (10, -4.0), (20, 7.0)]
    """
    key = grid_ns - grid_ns % every_ns
    starts, first = np.unique(key, return_index=True)
    sums = np.add.reduceat(net, first) if len(net) else np.array([])
    return [(int(s), float(v)) for s, v in zip(starts, sums, strict=True)]


def worst_window(nets: list[float]) -> dict:
    """The worst window's net, and the best window's share of the total when that is positive.

    Examples
    --------
    >>> out = worst_window([10.0, -4.0, 7.0])
    >>> out["worst_bps"], out["concentration"] == 10 / 13
    (-4.0, True)
    """
    if not nets:
        return {"worst_bps": None, "best_bps": None, "total_bps": None, "concentration": None,
                "windows": 0, "losing_windows": 0}  # fmt: skip
    total = float(sum(nets))
    return {"worst_bps": float(min(nets)), "best_bps": float(max(nets)), "total_bps": total,
            "concentration": float(max(nets)) / total if total > 0 else None,
            "windows": len(nets), "losing_windows": int(sum(v < 0 for v in nets))}  # fmt: skip


def underwater_steps(net: np.ndarray) -> int:
    """The longest run of steps on which the cumulative net is below its running best.

    The path starts at 0 before the first step.

    Examples
    --------
    Cumulative 2, -1, -2, 3, 1: under water at steps 2 and 3, then again at 5.

    >>> underwater_steps(np.array([2.0, -3.0, -1.0, 5.0, -2.0]))
    2
    """
    cum = np.cumsum(net)
    best = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    under = cum < best
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return longest


def max_drawdown(net: np.ndarray) -> float:
    """The largest fall of the cumulative net from its running best, as :meth:`Sim.metrics`."""
    cum = np.r_[0.0, np.cumsum(net)]
    return float(np.max(np.maximum.accumulate(cum) - cum))


def signal_half_life(rho: float, step_s: float) -> float | None:
    """:math:`-\\ln 2 / \\ln \\rho_1` steps, in seconds; None unless :math:`0 < \\rho_1 < 1`.

    Examples
    --------
    A first-order autocorrelation of 0.5 on a 100 ms grid:

    >>> signal_half_life(0.5, 0.1)
    0.1
    """
    if rho is None or not np.isfinite(rho) or not 0 < rho < 1:
        return None
    return -math.log(2) / math.log(rho) * step_s


def run_paths(h: np.ndarray, parts: list[tuple]) -> list[dict]:
    """Every round trip's net P&L step by step, by the rule of :func:`blinded.sim._round_trips`.

    A round trip is a run of one sign of ``h``, the first leg's position.
    ``parts`` holds ``(prev, trade, mid_pnl, spread, fee)`` for each leg: its
    mid P&L while the run is open goes to the run, a trade's cost to the run it
    opens, and the part that reduces a leg to the run it closes. Each item has
    the run's first and last step, ``path`` (the cumulative net at each step it
    touched, in order), ``net`` (the last of them) and ``cost``, the spread and
    fees charged to it. The nets equal :attr:`Sim.run_net`.
    """
    sign, before = np.sign(h), np.sign(np.r_[0.0, h[:-1]])
    starts = (sign != 0) & (sign != before)
    if not starts.any():
        return []
    run = np.where(sign != 0, np.cumsum(starts) - 1, -1)
    prev_run = np.r_[-1, run[:-1]]
    n_runs = int(starts.sum())
    steps = np.arange(len(h))
    ids, at, value, cost = [], [], [], np.zeros(n_runs)
    held = run >= 0
    for p, trade, mid_pnl, spread, fee in parts:
        per_unit = np.divide(spread + fee, np.abs(trade), out=np.zeros_like(fee),
                             where=trade != 0)  # fmt: skip
        was = np.sign(p)
        closing = (was != 0) & (np.sign(trade) == -was)
        close_qty = np.where(closing, np.minimum(np.abs(trade), np.abs(p)), 0.0)
        open_qty = np.abs(trade) - close_qty
        ids.append(run[held])
        at.append(steps[held])
        value.append((mid_pnl - open_qty * per_unit)[held])
        np.add.at(cost, run[held], (open_qty * per_unit)[held])
        closed = (close_qty > 0) & (prev_run >= 0)
        ids.append(prev_run[closed])
        at.append(steps[closed])
        value.append(-(close_qty * per_unit)[closed])
        np.add.at(cost, prev_run[closed], (close_qty * per_unit)[closed])
    ids, at, value = (np.concatenate(v) for v in (ids, at, value))
    order = np.lexsort((at, ids))
    ids, at, value = ids[order], at[order], value[order]
    out = []
    bounds = np.r_[np.flatnonzero(np.diff(ids)) + 1, len(ids)]
    for lo, hi in zip(np.r_[0, bounds[:-1]], bounds, strict=True):
        steps_here, v = at[lo:hi], value[lo:hi]
        keys, first = np.unique(steps_here, return_index=True)
        path = np.cumsum(np.add.reduceat(v, first))
        r = int(ids[lo])
        out.append({"run": r, "first": int(keys[0]), "last": int(keys[-1]),
                    "path": path.tolist(), "net": float(path[-1]), "cost": float(cost[r]),
                    "length": int((run == r).sum())})  # fmt: skip
    return out


def mae(paths: list[dict]) -> dict:
    """The adverse excursion of each round trip: the lowest its cumulative net went, or 0.

    The median and the worst, and the share of round trips that were under
    water by more than their own cost and still closed positive.

    Examples
    --------
    One round trip whose net path is +2, -3, -1, +5: cumulative 2, -1, -2, 3.

    >>> out = mae([{"path": [2.0, -1.0, -2.0, 3.0], "net": 3.0, "cost": 1.5}])
    >>> out["worst_bps"], out["median_close_bps"], out["recovered_share"]
    (-2.0, 3.0, 1.0)
    """
    if not paths:
        return {"round_trips": 0, "median_bps": None, "worst_bps": None,
                "median_close_bps": None, "recovered_share": None}  # fmt: skip
    worst = np.array([min(0.0, min(p["path"])) for p in paths])
    deep = [w < -p["cost"] and p["net"] > 0 for w, p in zip(worst, paths, strict=True)]
    return {"round_trips": len(paths), "median_bps": float(np.median(worst)),
            "worst_bps": float(worst.min()),
            "median_close_bps": float(np.median([p["net"] for p in paths])),
            "recovered_share": float(np.mean(deep))}  # fmt: skip


def breadth(leg_nets: dict[str, float]) -> dict:
    """The share of legs with a positive net, and the best leg's share of the total.

    Examples
    --------
    >>> out = breadth({"A": 5.0, "B": -2.0, "C": 1.0})
    >>> out["positive_share"] == 2 / 3, out["top_share"] == 5 / 4
    (True, True)
    """
    nets = np.array(list(leg_nets.values()), dtype=float)
    total = float(nets.sum())
    top = max(leg_nets, key=leg_nets.get) if leg_nets else None
    share = float(nets.max()) / total if total > 0 else None
    return {"positive_share": float(np.mean(nets > 0)) if len(nets) else None,
            "top_leg": top, "top_share": share}  # fmt: skip


def whipsaw(lengths: list[int], nets: list[float], lo_steps: float) -> dict:
    """Round trips closed within one ``lo`` of opening, the hit rate, the mean win over the mean
    loss, and the share of the net made in the best tenth of round trips.

    Examples
    --------
    Round trips of 1, 3 and 20 steps against a holding of 2 steps:

    >>> whipsaw([1, 3, 20], [-1.0, 2.0, 5.0], 2)["short_share"] == 1 / 3
    True
    """
    n = len(lengths)
    if not n:
        return {"round_trips": 0, "short_share": None, "hit_rate": None, "win_loss": None,
                "best_tenth_share": None}  # fmt: skip
    v = np.array(nets, dtype=float)
    wins, losses = v[v > 0], v[v < 0]
    k = math.ceil(0.1 * n)
    total = float(v.sum())
    ratio = float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None
    best = float(np.sort(v)[::-1][:k].sum()) / total if total > 0 else None
    return {"round_trips": n, "short_share": float(np.mean(np.array(lengths) <= lo_steps)),
            "hit_rate": float(np.mean(v > 0)), "win_loss": ratio,
            "best_tenth_share": best}  # fmt: skip


# -- the checks on one segment -----------------------------------------------------------


def parts(s: sim.Sim) -> list[tuple]:
    """``(prev, trade, mid_pnl, spread, fee)`` of each leg of a simulation, a leg's funding in
    its ``mid_pnl`` (:attr:`~blinded.sim.Sim.held_pnl`), as the round trips book it."""
    legs = s.legs.values() if isinstance(s, sim.MultiSim) else [s]
    return [(np.r_[0.0, x.position[:-1]], x.trade, x.held_pnl, x.spread_bps, x.fee_paid)
            for x in legs]  # fmt: skip


NO_FUNDING = "no funding stream on this file"


def funding(s: sim.Sim, panel, win, perp: str) -> dict:
    """D7 on one segment: what the perpetual leg's funding paid, split by the side it was on,
    against what holding the position made on the marks, and the settlements of the window.

    ``funding_bps`` is the leg's funding over the window, ``short_bps`` and
    ``long_bps`` the parts accrued while the perpetual was short and long;
    ``mid_bps`` is the mid P&L of every leg, which for spot against its
    perpetual is the basis change. ``lines`` holds both cumulatively, by hour.
    ``settlements`` and ``mean_rate_bps`` are the perpetual's settlements
    stamped inside the window and their mean rate.
    """
    legs = s.legs if isinstance(s, sim.MultiSim) else {perp: s}
    x = legs.get(perp)
    if x is None or x.funding is None:
        return {"note": NO_FUNDING}
    ev = panel.events("funding", perp)
    f = ev.frame["ts"].dt.epoch("ns").to_numpy()
    rate = ev.column("rate")[(f >= win.start.value) & (f < win.end.value)]
    rate = rate[np.isfinite(rate)]
    paid = x.funding
    cum_f, cum_m = np.cumsum(paid), np.cumsum(s.mid_pnl)
    keep = mk.downsample(len(paid))
    step_h = panel.step_ns / 3.6e12
    return {"funding_bps": float(paid.sum()), "short_bps": float(paid[x.position < 0].sum()),
            "long_bps": float(paid[x.position > 0].sum()), "mid_bps": float(s.mid_pnl.sum()),
            "settlements": int(len(rate)),
            "mean_rate_bps": float(np.mean(rate) * 1e4) if len(rate) else None,
            "lines": {"hours": (keep * step_h).tolist(), "funding_bps": cum_f[keep].tolist(),
                      "mid_bps": cum_m[keep].tolist()}}  # fmt: skip


def latencies(brief) -> list[pd.Timedelta]:
    """The latencies of the D1 grid: 0 and the brief's, and half and twice it where the file
    keeps those stamps; one bar only on bars."""
    lat = brief.latency
    out = [pd.Timedelta(0), lat]
    if not brief.grid.bars:
        out += [x for x in (lat / 2, lat * 2) if brief.data.exact(x)]
    return sorted(set(out))


def costs(run, s: sim.Sim, rows: dict, brief) -> dict:
    """D1 on one segment: the break-even fee, the five sim rows and the fee-latency grid.

    ``run(latency, fee)`` simulates the pick at that latency and fee (``None``
    for the brief's own); ``rows`` are :func:`blinded.sim.report`'s rows. With
    funding, ``gross_bps`` includes it, so the break-even fee is what the
    position made before fees, funding included: for a basis trade the
    funding is the edge. Borrow on a short spot leg is in it the same way.
    """
    lats = latencies(brief)
    labels = [*(f"{f:g}" for f in FEES), "brief"]
    grid = [[run(lat, f).metrics()["net_bps_per_hour"] for f in (*FEES, None)] for lat in lats]
    gross = float(s.gross.sum())
    funded = {}
    for name, carried in (("funding_bps", s.funding), ("borrow_bps", s.borrow)):
        if carried is not None:
            funded[name] = float(carried.sum())
            gross += funded[name]
    traded = float(np.abs(s.trade).sum())
    keep = ("net_bps_per_hour", "gross_bps_per_hour", "fills")
    return {"gross_bps": gross, **funded, "traded": traded,
            "breakeven_fee_bps": breakeven_fee_bps(gross, traded),
            "rows": {name: {k: _num(r.get(k)) for k in keep} for name, r in rows.items()},
            "grid": {"fees": labels, "latencies_ms": [x / pd.Timedelta("1ms") for x in lats],
                     "net_bps_per_hour": grid}}  # fmt: skip


def race(x: np.ndarray, at_decision: dict, at_fill: dict, a: int, b: int, direction: float,
         entry_bps: float | None, half_spread: float | None) -> dict:  # fmt: skip
    """D2 on one segment: the rank IC of the feature against returns from the decision and
    from the fill, at each horizon.

    ``peak`` is the largest rank IC in the traded direction (``direction``,
    the sign of the train IC) on returns that start at the decision;
    ``ic_left_share`` is the rank IC at that horizon on returns that start at
    the fill, which a taker can hold, over the peak. ``in_flight_bps`` is the
    order's average entry cost less the median half spread: what the price
    moved while the order was on its way.
    """
    curve0 = {h: mk.rank_ic(x[a:b], r[a:b]) for h, r in at_decision.items()}
    curve1 = {h: mk.rank_ic(x[a:b], r[a:b]) for h, r in at_fill.items()}
    signed = {h: None if v is None else direction * v for h, v in curve0.items()}
    known = {h: v for h, v in signed.items() if v is not None}
    peak_h = max(known, key=known.get) if known else None
    peak = known.get(peak_h) if peak_h else None
    left = curve1.get(peak_h) if peak_h else None
    left = None if left is None else direction * left
    share = left / peak if peak and left is not None and peak > 0 else None
    flight = None if None in (entry_bps, half_spread) else entry_bps - half_spread
    return {"horizons": list(curve0), "rank_ic_from_decision": [signed[h] for h in curve0],
            "rank_ic_from_fill": [None if curve1[h] is None else direction * curve1[h]
                                  for h in curve0],
            "peak_horizon": peak_h, "peak_rank_ic": peak, "rank_ic_left": left,
            "ic_left_share": share, "entry_bps": entry_bps, "half_spread_bps": half_spread,
            "in_flight_bps": flight}  # fmt: skip


def capacity(s: sim.Sim, panel, win, inst: str) -> dict:
    """D3 on one segment, per leg: the touch share of each fill and the participation.

    The far touch is the ask size for a buy and the bid size for a sell,
    as-of the fill. Participation is the units traded over the volume traded
    in the window. A book without sizes, as on bars, has no touch share.
    ``inst`` names the leg of a one-leg simulation.
    """
    legs = s.legs if isinstance(s, sim.MultiSim) else {inst: s}
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [win.start.value, win.end.value])
    out = {}
    for leg, x in legs.items():
        traded = np.nonzero(x.trade)[0]
        book = panel.stream_of(leg, "bid")
        at = g[i0 + traded] + x.latency.value + book.delay_ns
        side = np.sign(x.trade[traded])
        row = {"units": float(np.abs(x.trade).sum())}
        if {"bid_qty", "ask_qty"} <= book.fields:
            touch = np.where(side > 0, book.asof("ask_qty", at), book.asof("bid_qty", at))
            row |= {k: v for k, v in touch_shares(x.trade[traded], touch).items()
                    if k != "shares"}  # fmt: skip
        else:
            row |= {"median": None, "above_one": None, "n": 0, "touch": NOT_MEASURABLE}
        try:
            trades = panel.events("trades", leg)
        except KeyError:
            row["participation"] = None
        else:
            inside = (trades.at >= win.start.value) & (trades.at < win.end.value)
            volume = float(np.nansum(trades.column("qty")[inside]))
            row["volume"] = volume
            row["participation"] = row["units"] / volume if volume > 0 else None
        out[leg] = row
    return out


def worst_windows(s: sim.Sim, panel, win, brief) -> dict:
    """D4 on one segment: net P&L per window of the brief's block, the stretch under water."""
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [win.start.value, win.end.value])
    windows = window_nets(s.net, g[i0:i1], brief.stats.block.value)
    step_h = panel.step_ns / 3.6e12
    return worst_window([v for _, v in windows]) | {
        "underwater_hours": underwater_steps(s.net) * step_h,
        "max_drawdown_bps": max_drawdown(s.net),
        "block": str(brief.stats.block),
    }


def half_life(x, a: int, b: int, step_s: float, pair=None, seg=None) -> dict:
    """D5 on one segment: the signal's half-life from its first-order autocorrelation, and for
    a pair the median window half-life of its spread against a random walk's. A basket's
    signal, one array per leg, has the mean of its legs' autocorrelations."""
    out = {}
    if isinstance(x, dict):
        rhos = [_autocorr(v[a:b]) for v in x.values()]
        rhos = [r for r in rhos if r is not None]
        rho = float(np.mean(rhos)) if rhos else None
        out |= {"autocorr": _num(rho), "signal_half_life_s": signal_half_life(rho, step_s)}
    elif x is not None:
        rho = _autocorr(x[a:b])
        out |= {"autocorr": _num(rho), "signal_half_life_s": signal_half_life(rho, step_s)}
    if pair is not None:
        rows = [w for w in pair["windows"] if w["segment"] == seg and w["phi"] is not None]
        lives = [w["half_life_s"] for w in rows if w["reverts"] and w["half_life_s"] is not None]
        out |= {"spread_median_half_life_s": float(np.median(lives)) if lives else None,
                "random_walk_half_life_s": pair["random_walk_half_life_s"],
                "no_reversion_share": (sum(w["reverts"] is False for w in rows) / len(rows)
                                       if rows else None)}  # fmt: skip
    return out


def _autocorr(v: np.ndarray) -> float | None:
    live = v[np.isfinite(v)]
    return metrics.autocorr(live) if len(live) > 2 else None


def leg_paths(s: sim.Sim) -> list[dict]:
    """The round trips of :func:`run_paths`: of the first leg, as :class:`~blinded.sim.MultiSim`
    keys them, or for a basket (more than two legs) of every leg on its own, pooled, since
    a basket's legs open and close at different times."""
    if isinstance(s, sim.MultiSim) and len(s.legs) > 2:
        return [p for x in s.legs.values() for p in run_paths(x.position, parts(x))]
    return run_paths(s.position, parts(s))


def breadths(s: sim.MultiSim) -> dict:
    """D9 on one segment: :func:`breadth` of the legs' nets, the mean number of legs held, and
    the units traded per rebalance, a rebalance being a step on which any leg trades. In the
    three-leg example of ``tests/test_basket.py`` every leg is held for four of five steps,
    3 x 4 / 5 = 2.4 legs, and the legs trade at steps 0, 2 and 4, 12 units in all, 4 a
    rebalance.
    """
    nets = {leg: float(x.net.sum()) for leg, x in s.legs.items()}
    held = sum((x.position != 0).astype(int) for x in s.legs.values())
    traded = sum((x.trade != 0).astype(int) for x in s.legs.values()) > 0
    units = float(sum(np.abs(x.trade).sum() for x in s.legs.values()))
    rebalances = int(traded.sum())
    return breadth(nets) | {
        "legs": len(nets), "mean_legs_held": float(np.mean(held)), "rebalances": rebalances,
        "turnover_per_rebalance": units / rebalances if rebalances else None,
        "leg_net_bps": nets,
    }  # fmt: skip


def venue_feed(panel, win, inst: str, curves: list[dict], seg: str) -> dict:
    """D11 on one segment: the feed cadence of ``inst``'s book over the window, from its
    receive stamps (:func:`blinded.demo.venue_cadence`), and for each curve across the venues
    :math:`C` at its peak against :math:`C` at the lag nearest the median gap: a lead no
    longer than the cadence cannot be told from the schedule."""
    from blinded.demo import venue_cadence

    ts = panel.events("book", inst).frame["ts"].dt.epoch("ns").to_numpy()
    inside = ts[(ts >= win.start.value) & (ts < win.end.value)]
    cad = venue_cadence(inside / 1e6) if len(inside) > 2 else None
    out = {"instrument": inst, "median_gap_ms": None, "bin_max_ms": None, "share_max": None,
           "ok": None, "curves": []}  # fmt: skip
    if cad is not None:
        out |= {k: cad[k] for k in ("median_gap_ms", "bin_max_ms", "share_max", "ok")}
    for c in curves:
        row = c["segments"].get(seg) or {}
        corr, lags = row.get("corr") or [], c["lags_ms"]
        known = [(lag, v) for lag, v in zip(lags, corr, strict=True) if v is not None]
        near = None
        if cad is not None and known:
            near = min(known, key=lambda lv: (abs(lv[0] - cad["median_gap_ms"]), lv[0]))
        out["curves"].append({
            "leader": c["leader"], "follower": c["follower"],
            "lag_at_peak_ms": row.get("lag_at_peak_ms"), "c_at_peak": row.get("c_at_peak"),
            "cadence_lag_ms": near[0] if near else None, "c_at_cadence": near[1] if near else None,
            "peak_within_cadence": (None if cad is None or row.get("lag_at_peak_ms") is None
                                    else abs(row["lag_at_peak_ms"]) <= cad["median_gap_ms"]),
        })  # fmt: skip
    return out


def whipsaws(s: sim.Sim, lo: pd.Timedelta, step_ns: int) -> dict:
    """D10 on one segment, from the round trips of :func:`run_paths`."""
    paths = run_paths(s.position, parts(s))
    lo_steps = lo.value / step_ns
    return whipsaw([p["length"] for p in paths], [p["net"] for p in paths], lo_steps) | {
        "lo_steps": lo_steps
    }


def _num(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# -- a market maker's checks (blinded.sim_maker) ---------------------------------------

MAKER_FEES = (-0.5, 0.0, 1.0, 2.0)  # the D1 grid's maker fees in bps a unit, beside the brief's
CAVEAT_SHARE = 0.1  # the size over the median displayed size past which the caveat prints
CAVEAT = ("the engine assumes the market does not react to your quote, and this size is over "
          "a tenth of the median size displayed at the touch")  # fmt: skip


def maker_costs(scored, zero, rows: dict, latency_ms: float) -> dict:
    """D1 for a market-making strategy on one segment, on the scored model's fills.

    ``scored`` and ``zero`` are the pass of the scored queue model at the
    brief's latencies and at every latency 0
    (:class:`~blinded.sim_maker.MakerSim`); ``rows`` are
    :func:`blinded.sim_maker.report`'s. The break-even fee is the gross,
    funding and borrow included, over the units filled while resting: the
    maker fee a unit at which nothing is left. The grid is the net per hour at
    each maker fee of :data:`MAKER_FEES` and the brief's, at latency 0 and the
    brief's: fees change no fill, so a fee is arithmetic on a pass, and the
    latency is a second pass. The flatten keeps the brief's taker fee.
    """
    gross = float(scored.gross.sum())
    funded = {}
    for name, carried in (("funding_bps", scored.funding), ("borrow_bps", scored.borrow)):
        if carried is not None:
            funded[name] = float(carried.sum())
            gross += funded[name]
    units = float(scored.units.sum())
    keep = ("net_bps_per_hour", "gross_bps_per_hour", "fills")
    passes = sorted({0.0: zero, float(latency_ms): scored}.items())
    grid = [[s.refee(f).metrics()["net_bps_per_hour"] for f in MAKER_FEES]
            + [s.metrics()["net_bps_per_hour"]] for _, s in passes]  # fmt: skip
    return {"gross_bps": gross, **funded, "traded": units,
            "breakeven_fee_bps": breakeven_fee_bps(gross, units),
            "rows": {name: {k: _num(r.get(k)) for k in keep} for name, r in rows.items()},
            "grid": {"fees": [*(f"{f:g}" for f in MAKER_FEES), "brief"],
                     "latencies_ms": [lat for lat, _ in passes], "net_bps_per_hour": grid},
            "model": scored.model}  # fmt: skip


def aggressors(touch, start_ns: int, end_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """The aggressor orders inside ``[start_ns, end_ns)``: trades sharing a stamp and a side
    summed, as ``(side, units)``."""
    t, side, qty = touch.trade_ns, touch.side, touch.qty
    inside = (t >= start_ns) & (t < end_ns)
    t, side, qty = t[inside], side[inside], qty[inside]
    if not len(t):
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    new = np.r_[True, (np.diff(t) != 0) | (np.diff(side) != 0)]
    starts = np.flatnonzero(new)
    return side[starts], np.add.reduceat(qty, starts)


def maker_capacity(s, start_ns: int, end_ns: int) -> dict:
    """D3 for a market-making strategy on one segment: how big its quotes were against the
    queue they joined and the orders that filled them.

    Per resting fill, its order's size over the size displayed on the
    order's side of the touch as-of its arrival (median, and the share over
    1); the median order size over the median aggressor order of the window
    (trades sharing a stamp and a side, summed); the share of fills that were
    partial, less than the order's size; and on each side the units filled
    over the units the other side's aggressors traded in the window (a bid is
    filled by sells). ``caveat`` is True when the median order is over a
    tenth of the median displayed size, where the engine's assumption that
    the market does not react to the quote stops being harmless.
    """
    b, touch = s.book, s.touch
    rest = ~b.fill_taker if len(b.fill_taker) == len(b.fill_ns) else np.ones(len(b.fill_ns),
                                                                             bool)  # fmt: skip
    oid = b.fill_order[rest]
    size = b.order_size[oid]
    j = touch.row(b.order_arrival[oid])
    shown = np.where(b.order_side[oid] > 0, touch.bid_qty[np.maximum(j, 0)],
                     touch.ask_qty[np.maximum(j, 0)]).astype(float)  # fmt: skip
    shown[j < 0] = np.nan
    shares = touch_shares(size, shown)
    a_side, a_qty = aggressors(touch, start_ns, end_ns)
    sent = b.order_size[np.isin(b.order_status, ("filled", "cancelled", "open", "crossed"))]
    median_order = float(np.median(sent)) if len(sent) else None
    median_aggressor = float(np.median(a_qty)) if len(a_qty) else None
    qty = b.fill_qty[rest]
    side = b.fill_side[rest]
    out = {
        "fills": int(rest.sum()),
        "median_touch_share": shares["median"],
        "touch_share_above_one": shares["above_one"],
        "median_order": median_order,
        "median_aggressor": median_aggressor,
        "order_over_aggressor": (
            median_order / median_aggressor if median_order and median_aggressor else None
        ),
        "partial_share": float(np.mean(qty < size - 1e-9 * size)) if len(qty) else None,
    }
    for name, mine, theirs in (("bid", 1, -1), ("ask", -1, 1)):
        traded = float(a_qty[a_side == theirs].sum())
        filled = float(qty[side == mine].sum())
        out[f"participation_{name}"] = filled / traded if traded > 0 else None
    rows = (touch.book_ns >= start_ns) & (touch.book_ns < end_ns)
    shown_all = np.concatenate([touch.bid_qty[rows], touch.ask_qty[rows]])
    displayed = float(np.median(shown_all)) if len(shown_all) else None
    out["median_displayed"] = displayed
    out["caveat"] = bool(median_order is not None and displayed
                         and median_order > CAVEAT_SHARE * displayed)  # fmt: skip
    return out


def _how_filled(s) -> dict:
    """The share of a pass's resting fills by how they came: a trade through the price, the
    book's far touch reaching the price with no trade (``crossed_share``, under rules 2 only:
    :data:`blinded.ladder.RULES`), an order alone inside the spread at its arrival, and the
    queue at the touch traded out."""
    b, touch = s.book, s.touch
    rest = ~b.fill_taker if len(b.fill_taker) == len(b.fill_ns) else np.ones(len(b.fill_ns),
                                                                             bool)  # fmt: skip
    n = int(rest.sum())
    if not n:
        none = {"through_share": None, "crossed_share": None, "inside_share": None,
                "queue_share": None}  # fmt: skip
        if getattr(s, "rules", 2) < 2:
            del none["crossed_share"]
        return none
    oid = b.fill_order[rest]
    j = np.maximum(touch.row(b.order_arrival[oid]), 0)
    px, buy = b.order_price[oid], b.order_side[oid] > 0
    inside = np.where(buy, px > touch.bid[j], px < touch.ask[j])
    through = b.fill_through[rest]
    if getattr(s, "rules", 2) < 2:
        # under rules 1 (blinded.ladder.RULES) no book row fills a resting order
        inside &= ~through
        return {"through_share": float(through.mean()), "inside_share": float(inside.mean()),
                "queue_share": float(1.0 - through.mean() - inside.mean())}  # fmt: skip
    crossed = (b.fill_crossed[rest] if len(b.fill_crossed) == len(b.fill_ns)
               else np.zeros(n, bool))  # fmt: skip
    inside &= ~through & ~crossed
    queue = 1.0 - through.mean() - crossed.mean() - inside.mean()
    return {"through_share": float(through.mean()), "crossed_share": float(crossed.mean()),
            "inside_share": float(inside.mean()), "queue_share": float(queue)}  # fmt: skip


def toxicity(sims: dict) -> dict:
    """D12 on one segment: for each fill model's pass (:class:`~blinded.sim_maker.MakerSim`),
    what its resting fills captured against the mid at their stamps, the mid's move over 1, 10
    and 60 s after them (adverse selection), and the rest of the gross (inventory beyond 10 s
    and the flatten), each per unit filled; how the fills came (:func:`_how_filled`); and
    adverse over captured at 10 s, the multiple of what a fill earned that the next 10 s took
    back (negative when the mid moved against the fill)."""
    from blinded import sim_maker

    out = {}
    for m, s in sims.items():
        d = sim_maker.decompose(s)
        units = d["units"]

        def per(x, u=units):
            return x / u if u else None

        adv = d["adverse_bps"]
        out[m] = {"units": units, "captured_bps_per_unit": per(d["captured_bps"]),
                  **{f"adverse_{h}_bps_per_unit": per(v) for h, v in adv.items()},
                  "inventory_bps_per_unit": per(d["inventory_bps"]["10s"]),
                  "adverse_over_captured": (adv["10s"] / d["captured_bps"]
                                            if d["captured_bps"] else None),
                  **_how_filled(s)}  # fmt: skip
    return out


def queue(sims: dict) -> dict:
    """D13 on one segment: for each fill model's pass, where in the queue its orders sat.

    The median and 90th percentile wait from an order's arrival to each of
    its fills, in seconds; the share of the orders that rested and were
    cancelled with nothing filled; post-only rejections over orders sent; the
    mean queue ahead at arrival in multiples of the order's size (unknown for
    T, which never knows it, and for an order deeper than the touch); the
    share of orders that arrived deeper than the touch; the share of
    decisions with a bid, and with an ask, resting; and the mean and largest
    absolute inventory over the window's steps.
    """
    out = {}
    for m, s in sims.items():
        b, touch = s.book, s.touch
        rest = ~b.fill_taker if len(b.fill_taker) == len(b.fill_ns) else np.ones(
            len(b.fill_ns), bool)  # fmt: skip
        wait = (b.fill_ns[rest] - b.order_arrival[b.fill_order[rest]]) / 1e9
        rested = np.isin(b.order_status, ("filled", "cancelled", "open", "crossed"))
        filled = np.zeros(len(b.order_side), dtype=bool)
        filled[b.fill_order[rest]] = True
        cancelled = (b.order_status == "cancelled") & ~filled
        j = np.maximum(touch.row(b.order_arrival), 0)
        deep = np.where(b.order_side > 0, b.order_price < touch.bid[j],
                        b.order_price > touch.ask[j]) & rested  # fmt: skip
        ahead = b.order_ahead[rested] / b.order_size[rested]
        sent = len(b.order_side)
        q = b.quoted.astype(int)
        out[m] = {"orders": sent,
                  "median_wait_s": float(np.median(wait)) if len(wait) else None,
                  "p90_wait_s": float(np.quantile(wait, 0.9)) if len(wait) else None,
                  "cancelled_unfilled_share": (float(cancelled.sum() / rested.sum())
                                               if rested.any() else None),
                  "rejected_share": b.rejected / sent if sent else None,
                  "mean_ahead_sizes": (float(np.nanmean(ahead)) if np.isfinite(ahead).any()
                                       else None),
                  "deeper_share": float(deep.sum() / rested.sum()) if rested.any() else None,
                  "bid_quoted_share": float(np.mean((q & 1) > 0)) if len(q) else None,
                  "ask_quoted_share": float(np.mean((q & 2) > 0)) if len(q) else None,
                  "mean_abs_inventory": float(np.mean(np.abs(s.position))),
                  "max_abs_inventory": float(np.max(np.abs(s.position)))}  # fmt: skip
    return out


# -- an arbitrage: the hedge leg's death checks -------------

COUNTERPARTY = ("the fills are this flow's prints landing through your quote; the replay "
                "assumes it does not change when you are there")  # fmt: skip


def arb_costs(scored, rows: dict) -> dict:
    """D1 for an arbitrage on one segment: the break-even hedge fee, what the two legs made
    before fees (captured, delay and slippage) over the units hedged, in bps of one unit at
    :math:`P_\\text{ref}`, the quote venue's maker fee and the rebalances already taken off;
    and the sim rows' net, the hedge fee rows among them."""
    units = float(scored.taker_units.sum())
    maker = float(scored.units.sum()) * scored.maker_fee_bps
    rest = float(scored.gross.sum()) - maker - float(scored.transfer_fee.sum())
    keep = ("net_bps_per_hour", "gross_bps_per_hour", "fills", "net_bps_per_unit")
    return {"gross_bps": float(scored.gross.sum()), "hedged_units": units,
            "breakeven_hedge_fee_bps": breakeven_fee_bps(rest, units),
            "hedge_fee_bps": scored.hedge_fee_bps,
            "rows": {name: {k: _num(r.get(k)) for k in keep} for name, r in rows.items()},
            "model": scored.model}  # fmt: skip


def arb_capacity(s, uncapped, start_ns: int, end_ns: int) -> dict:
    """D3 for an arbitrage on one segment: the quote's size over the median print on the quote
    venue, in USD; the print-capped units over the uncapped pass's (``uncapped``, RA with a
    through fill taking the whole order); the units filled over the window's volume on that
    side (a bid filled by sells); and the share of hedged units taken past the hedge venue's
    level 0."""
    from blinded.sim_maker import arb_fills

    f = arb_fills(s)
    touch = s.touch
    t = touch.trade_ns
    inside = (t >= start_ns) & (t < end_ns)
    px, qty, side = touch.price[inside] * touch.tick, touch.qty[inside], touch.side[inside]
    usd = px * qty
    size = float(np.median(f["qty"] * f["price"])) if len(f["qty"]) else None
    sells, buys = float(qty[side < 0].sum()), float(qty[side > 0].sum())
    bought = float(f["qty"][f["side"] > 0].sum())
    sold = float(f["qty"][f["side"] < 0].sum())
    full = float(uncapped.units.sum())
    return {
        "size_usd": size, "print_usd_median": float(np.median(usd)) if len(usd) else None,
        "size_over_print": size / float(np.median(usd)) if size and len(usd) else None,
        "capped_over_uncapped": float(s.units.sum()) / full if full else None,
        "bid_share_of_sells": bought / sells if sells else None,
        "ask_share_of_buys": sold / buys if buys else None,
        "beyond_level0_share": float(f["beyond"].sum() / f["qty"].sum())
        if f["qty"].sum() else None,
    }  # fmt: skip


def picked_off(s) -> dict:
    """D14 on one segment: the share of fills picked off (captured below zero: the hedge
    venue's mid had already passed the price), the share with the hedge venue's touch already
    through it, the mean captured of the picked-off fills in bps, and both shares by hour of
    the window."""
    from blinded.sim_maker import arb_fills

    f = arb_fills(s)
    n = len(f["side"])
    hour = ((f["ns"] - s.start_ns) // 3_600_000_000_000).astype(int) if n else np.zeros(0, int)
    hours = int(math.ceil(s.hours)) if s.hours else 0
    by = []
    for h in range(hours):
        mine = hour == h
        k = int(mine.sum())
        by.append({"hour": h, "fills": k,
                   "picked_off_share": float(np.mean(f["picked_off"][mine])) if k else None,
                   "touch_through_share": float(np.mean(f["touch_through"][mine])) if k
                   else None})  # fmt: skip
    picked = f["picked_off"]
    return {"fills": n,
            "picked_off_share": float(np.mean(picked)) if n else None,
            "touch_through_share": float(np.mean(f["touch_through"])) if n else None,
            "picked_captured_bps": float(np.mean(f["captured_bps"][picked]))
            if picked.any() else None,
            "by_hour": by}  # fmt: skip


def hedge(s, slow, brief_row: dict, slow_row: dict) -> dict:
    """D15 on one segment: the hedge's delay cost, slippage and fee a fill, in bps of the
    fill; the share of hedges past the hedge venue's depth (``unreachable``); the visible
    crosses at the fill (the hedge venue's touch better than the price by more than both
    fees) and the share of them still there for the hedge's walk; and the ``hedge 1 s``
    row's net a fill and an hour against the brief's."""
    from blinded.sim_maker import arb_fills

    f, g = arb_fills(s), arb_fills(slow)
    n = len(f["side"])
    visible = f["visible"]

    def mean(d, key):
        return float(np.nanmean(d[key])) if len(d["side"]) else None

    return {"fills": n, "delay_bps": mean(f, "delay_bps"),
            "slippage_bps": mean(f, "slippage_bps"), "fee_bps": mean(f, "fee_bps"),
            "unreachable_share": float(np.mean(f["unreachable"])) if n else None,
            "visible_share": float(np.mean(visible)) if n else None,
            "reached_share": float(np.mean(f["reached"][visible])) if visible.any() else None,
            "net_bps": mean(f, "net_bps"), "slow_net_bps": mean(g, "net_bps"),
            "slow_delay_bps": mean(g, "delay_bps"),
            "net_bps_per_hour": _num(brief_row.get("net_bps_per_hour")),
            "slow_net_bps_per_hour": _num(slow_row.get("net_bps_per_hour"))}  # fmt: skip


def stock(s) -> dict:
    """D16 on one segment: the buy share, the drift (USD of coin a day moved onto the quote
    venue), the days the smaller of the two stocks the drift drains would last at that rate,
    the share of decisions with each side withdrawn for stock, and the rebalances booked and
    their fees in USD."""
    from blinded.sim_maker import arb_table

    table = arb_table(s)
    st = s.stock
    mark = float(s.ref)
    drift = table["drift_usd_a_day"]
    # buying on the quote venue drains the hedge venue's coin and the quote venue's USDT
    if drift > 0:
        drained = min(st.coin[s.hedge] * mark, st.usdt[s.quote])
    elif drift < 0:
        drained = min(st.coin[s.quote] * mark, st.usdt[s.hedge])
    else:
        drained = None
    w = s.withdrawn
    return {
        "buy_share": table["buy_share"],
        "drift_usd_a_day": drift,
        "days_to_exhaust": drained / abs(drift) if drained is not None and drift else None,
        "bid_withdrawn_share": float(np.mean(w & 1 > 0)) if len(w) else None,
        "ask_withdrawn_share": float(np.mean(w & 2 > 0)) if len(w) else None,
        "withdrawn_share": float(np.mean(w > 0)) if len(w) else None,
        "rebalances": len(s.transfers),
        "coin_rebalances": sum(x["currency"] == "coin" for x in s.transfers),
        "rebalance_fee_usd": float(np.sum([x["fee_usd"] for x in s.transfers])),
        "stock_usd": {v: st.coin[v] * mark + st.usdt[v] for v in (s.quote, s.hedge)},
    }


def counterparty(s, panel, start_ns: int, end_ns: int) -> dict:
    """D17 on one segment: who prints on the quote venue: prints a day
    and USD, the size distribution's p10, p50 and p90 and the largest 1 USD bucket's share, the
    gap's coefficient of variation, the share printed at a price never shown as the touch
    within a second (:func:`blinded.bookstats.flow_facts`), and the line."""
    from blinded import bookstats

    touch = s.touch
    t = touch.trade_ns
    inside = (t >= start_ns) & (t < end_ns)
    if inside.sum() < 2:
        return {"note": "fewer than two prints on the quote venue in the window"}
    flow = bookstats.flow_facts(
        touch.book_ns,
        touch.bid * touch.tick,
        touch.ask * touch.tick,
        t[inside],
        touch.side[inside].astype(float),
        touch.price[inside] * touch.tick,
        touch.qty[inside],
    )
    hours = (end_ns - start_ns) / 3.6e12
    return {"prints_a_day": flow["trades"] * 24 / hours, "usd_a_day": flow["usd"] * 24 / hours,
            "usd_p10": flow["usd_p10"], "usd_p50": flow["usd_p50"], "usd_p90": flow["usd_p90"],
            "bucket_max": flow["bucket_max"], "gap_cv": flow["gap_cv"],
            "never_shown": flow["never_shown"], "at_touch": flow["at_touch"],
            "line": COUNTERPARTY}  # fmt: skip
