"""Clock diagnostics for DATA.md: event rates, gaps, staleness and cross-stream lags."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from blinded.brief import Brief
from blinded.durations import label, words
from blinded.evaluate import scoring_window
from blinded.panel import Events, Panel

RETURN = pd.Timedelta("100ms")
MAX_LAG = pd.Timedelta("500ms")
CLEAR_PEAK = 0.1
LEAD_STEPS = 10
FINE = pd.Timedelta("10ms")


def stream_table(panel: Panel) -> pl.DataFrame:
    """Per stream and instrument: rate, inter-event gaps and grid staleness.

    Rows sharing a timestamp count as one event for the gap statistics.
    """
    grid = panel.grid_ns
    step = panel.step_ns
    span_s = (grid[-1] - grid[0]) / 1e9
    rows = []
    for stream, sym in panel.streams:
        ev = panel.events(stream, sym)
        distinct = np.unique(ev.at)
        gaps_ms = np.diff(distinct) / 1e6
        rows.append(
            {
                "stream": stream,
                "instrument": sym,
                "rows": len(ev),
                "duplicate_rows": len(ev) - len(distinct),
                "events_per_s": len(distinct) / span_s,
                "median_gap_ms": float(np.median(gaps_ms)) if len(gaps_ms) else np.nan,
                "p90_gap_ms": float(np.quantile(gaps_ms, 0.9)) if len(gaps_ms) else np.nan,
                "stale_share": staleness(ev, grid, step),
            }
        )
    return pl.DataFrame(rows)


def staleness(ev: Events, grid_ns: np.ndarray, step_ns: int) -> float:
    """Share of grid points whose as-of event is older than one grid step, or missing."""
    idx = np.searchsorted(ev.at, grid_ns, side="right") - 1
    age = np.where(idx >= 0, grid_ns - ev.at[np.maximum(idx, 0)], np.iinfo(np.int64).max)
    return float(np.mean(age > step_ns))


def returns_at(ev: Events, field: str, times_ns: np.ndarray, ret_ns: int) -> np.ndarray:
    """Log return of an as-of price over ``(t - ret, t]`` at each time."""
    return np.log(ev.asof(field, times_ns)) - np.log(ev.asof(field, times_ns - ret_ns))


def xcorr_lag(
    a: Events, field_a: str, b: Events, field_b: str, grid_ns: np.ndarray
) -> dict[str, float]:
    """Lag of ``b`` behind ``a`` that maximises the correlation of 100 ms returns.

    The lag is searched over +-500 ms, first in 10 ms steps and then in 1 ms
    steps around the best coarse lag. A positive lag means ``a`` moves first.
    The price of ``b`` is read as-of ``g + lag``, so the lag is not tied to the
    grid step.
    """
    ra = returns_at(a, field_a, grid_ns, RETURN.value)

    def corr(lag_ms: int) -> float:
        rb = returns_at(b, field_b, grid_ns + lag_ms * 1_000_000, RETURN.value)
        ok = np.isfinite(ra) & np.isfinite(rb)
        if ok.sum() < 3 or ra[ok].std() == 0 or rb[ok].std() == 0:
            return np.nan
        return float(np.corrcoef(ra[ok], rb[ok])[0, 1])

    limit = int(MAX_LAG / pd.Timedelta("1ms"))
    coarse = {lag: corr(lag) for lag in range(-limit, limit + 1, 10)}
    if all(np.isnan(v) for v in coarse.values()):
        return {"lag_ms": np.nan, "corr_at_lag": np.nan, "corr_at_0": np.nan}
    best = max(coarse, key=lambda k: -np.inf if np.isnan(coarse[k]) else coarse[k])
    fine = {lag: corr(lag) for lag in range(max(best - 10, -limit), min(best + 10, limit) + 1)}
    best = max(fine, key=lambda k: -np.inf if np.isnan(fine[k]) else fine[k])
    return {"lag_ms": float(best), "corr_at_lag": fine[best], "corr_at_0": coarse[0]}


def lead_lag_curve(
    leader: Events,
    follower: Events,
    grid_ns: np.ndarray,
    step_ns: int,
    lags_ns,
    field: str = "mid",
) -> list[float | None]:
    r"""The cross-correlation of one-step log returns, :math:`C(\tau)`, at each lag.

    .. math::

        C(\tau) = \operatorname{corr}\big(r_\ell(g),\ r_f(g + \tau)\big), \qquad
        r_X(t) = \ln m_X(t) - \ln m_X(t - \Delta),

    over the grid points ``grid_ns``, with :math:`\Delta` = ``step_ns`` and each
    mid read as-of its time from the events, so a lag need not be a whole
    step. :math:`r_\ell(g)` is the leader's last step as the decision at
    :math:`g` saw it; :math:`r_f(g + \tau)` is the follower's step that ends
    :math:`\tau` later. A follower that copies the leader :math:`c` later
    peaks at :math:`\tau = c`; a negative peak means the follower moved
    first. Points whose follower read falls after the last grid point are left
    out. None where fewer than three pairs are finite or one side is constant.

    Examples
    --------
    A grid of 10, 20, 30, 40 ns; the leader doubles at 5 and the follower 10 later, at 15.
    At lag 0 the two jumps sit in different steps, single spikes in four points, which
    correlate at -1/3; at lag 10 they line up:

    >>> import polars as pl
    >>> def ev(at, mid):
    ...     t = pl.Series(at).cast(pl.Datetime("ns"))
    ...     return Events(pl.DataFrame({"ts": t, "available_at": t, "bid": mid, "ask": mid}))
    >>> lead, follow = ev([0, 5], [1.0, 2.0]), ev([0, 15], [1.0, 2.0])
    >>> c = lead_lag_curve(lead, follow, np.array([10, 20, 30, 40]), 10, [0, 10])
    >>> [round(v, 4) for v in c]
    [-0.3333, 1.0]
    """
    r_lead = np.log(leader.asof(field, grid_ns)) - np.log(leader.asof(field, grid_ns - step_ns))
    out = []
    for lag in lags_ns:
        at = grid_ns + int(lag)
        r = np.log(follower.asof(field, at)) - np.log(follower.asof(field, at - step_ns))
        ok = np.isfinite(r_lead) & np.isfinite(r) & (at <= grid_ns[-1])
        if ok.sum() < 3 or np.std(r_lead[ok]) == 0 or np.std(r[ok]) == 0:
            out.append(None)
        else:
            out.append(float(np.corrcoef(r_lead[ok], r[ok])[0, 1]))
    return out


def lead_lag_lags(step_ns: int, fill_ns: int, raw: bool = True) -> list[int]:
    """The lags of :func:`lead_lag_curve` for the report and DATA.md, in ns, sorted.

    Whole grid steps from -10 to +10, and ``fill_ns + step_ns``: the
    follower's first whole step after a taker's fill, ``fill_ns`` being the
    latency plus the follower's feed delay. With the raw events (a file that
    is not thinned) also 10 ms steps across one grid step either side of zero,
    and the mirror of the fill point as a control.

    Examples
    --------
    >>> ms = 1_000_000
    >>> [x // ms for x in lead_lag_lags(100 * ms, 50 * ms, raw=False)][8:]
    [-200, -100, 0, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    """
    lags = {k * step_ns for k in range(-LEAD_STEPS, LEAD_STEPS + 1)} | {fill_ns + step_ns}
    if raw:
        fine = FINE.value
        lags |= set(range(-step_ns, step_ns + 1, fine)) | {-(fill_ns + step_ns)}
    return sorted(lags)


def lead_lag_summary(lags_ns, corr, fill_ns: int, step_ns: int) -> dict:
    """Where :math:`C(\tau)` peaks, and how much of it a taker can still trade.

    ``c_after_fill`` is :math:`C` at the follower's first whole step after the
    fill, ``fill_ns + step_ns``: the step a taker who decided at :math:`g`
    holds in full. ``best_after_fill`` is the largest :math:`C` from there
    on. ``usable_by_taker`` is a rule of thumb, not a test: yes when that is
    positive and at least half the peak. Ties at the peak go to the lag
    nearest zero.

    Examples
    --------
    A peak of 0.9 at 100 ns, and 0.6 on the first whole step after a fill at 50 ns:

    >>> out = lead_lag_summary([0, 100, 150], [0.2, 0.9, 0.6], fill_ns=50, step_ns=100)
    >>> out["lag_at_peak_ms"], out["c_after_fill"], out["usable_by_taker"]
    (0.0001, 0.6, True)
    """
    after = fill_ns + step_ns
    known = [(lag, c) for lag, c in zip(lags_ns, corr, strict=True) if c is not None]
    if not known:
        return {"lag_at_peak_ms": None, "c_at_peak": None, "c_after_fill": None,
                "best_after_fill": None, "usable_by_taker": False}  # fmt: skip
    lag, peak = max(known, key=lambda lc: (lc[1], -abs(lc[0])))
    later = [c for lag_, c in known if lag_ >= after]
    best = max(later) if later else None
    at_fill = dict(known).get(after)
    usable = best is not None and peak > 0 and best > 0 and best >= peak / 2
    return {"lag_at_peak_ms": lag / 1e6, "c_at_peak": peak, "c_after_fill": at_fill,
            "best_after_fill": best, "usable_by_taker": bool(usable)}  # fmt: skip


def lead_lag(brief: Brief, panel: Panel, window, family=None) -> list[dict]:
    """:func:`lead_lag_curve` and :func:`lead_lag_summary` for every leader and follower of the
    brief's lead-lag family, over the grid points of ``window``.

    The lags are :func:`lead_lag_lags`, with the fine steps only when the file
    holds the raw events, and only those whose reads a thinned file keeps.
    The fill is the latency plus the follower's feed delay. Follower reads
    stop at the window's last grid point.
    """
    family = family or brief.family("lead_lag")
    if family is None:
        return []
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [window.start.value, window.end.value])
    step = panel.step_ns
    price = brief.target.price
    out = []
    for leader in family.leaders:
        for follower in family.followers:
            if leader == follower:
                continue
            lead, follow = panel.stream_of(leader, price), panel.stream_of(follower, price)
            fill = brief.latency.value + follow.delay_ns
            raw = brief.data.thinned is None and not brief.grid.bars
            lags = [
                lag
                for lag in lead_lag_lags(step, fill, raw=raw)
                if brief.data.exact(pd.Timedelta(lag - follow.delay_ns, unit="ns"))
            ]
            corr = lead_lag_curve(lead, follow, g[i0:i1], step, lags, price)
            out.append({"leader": leader, "follower": follower,
                        "lags_ms": [lag / 1e6 for lag in lags], "fill_ms": fill / 1e6,
                        "step_ms": step / 1e6, "corr": corr,
                        **lead_lag_summary(lags, corr, fill, step)})  # fmt: skip
    return out


def touch_alignment(book: Events, trades: Events, offsets_ms=range(-200, 21)) -> dict:
    """Share of trades printed at the touch when the book is read at ``ts + offset``.

    A buy should print at the ask and a sell at the bid of the book just before
    it. If the share peaks at a negative offset, trades are stamped late
    relative to the book by that much; ties go to the offset nearest zero. The
    book is read strictly before the shifted time, so a book update stamped
    with the trade itself is excluded.
    """
    at = trades.at
    side, price = trades.column("side"), trades.column("price")
    bid, ask = book.column("bid"), book.column("ask")
    shares = {}
    for off in offsets_ms:
        idx = np.searchsorted(book.at, at + off * 1_000_000, side="left") - 1
        ok = idx >= 0
        touch = np.where(side[ok] > 0, ask[idx[ok]], bid[idx[ok]])
        shares[off] = float(np.mean(price[ok] == touch))
    best = max(shares, key=lambda off: (shares[off], -abs(off)))
    return {"best_offset_ms": best, "share_at_best": shares[best], "share_at_0": shares[0]}


def lag_table(panel: Panel) -> pl.DataFrame:
    """Cross-correlation lags: trades vs book of each instrument, and book vs book."""
    grid = panel.grid_ns
    have = set(panel.streams)
    pairs = [
        (("book", s, "mid"), ("trades", s, "price"))
        for s in panel.symbols
        if ("book", s) in have and ("trades", s) in have
    ]
    books = [s for s in panel.symbols if ("book", s) in have]
    for x, y in itertools.combinations(books, 2):
        pairs.append((("book", x, "mid"), ("book", y, "mid")))
    rows = []
    for (sa, xa, fa), (sb, xb, fb) in pairs:
        lag = xcorr_lag(panel.events(sa, xa), fa, panel.events(sb, xb), fb, grid)
        clear = bool(lag["corr_at_lag"] >= CLEAR_PEAK)
        rows.append({"first": f"{sa} {xa}", "second": f"{sb} {xb}", **lag, "clear_peak": clear})
    return pl.DataFrame(rows)


def touch_table(panel: Panel) -> pl.DataFrame:
    rows = []
    for sym in panel.symbols:
        if ("book", sym) in panel.streams and ("trades", sym) in panel.streams:
            fit = touch_alignment(panel.events("book", sym), panel.events("trades", sym))
            rows.append({"instrument": sym, **fit})
    return pl.DataFrame(rows)


def write_data_md(brief: Brief, panel: Panel, out: Path) -> str:
    """Write DATA.md for a brief and return its text."""
    text = render(brief, panel)
    Path(out).write_text(text)
    return text


def render(brief: Brief, panel: Panel) -> str:
    thinned = brief.data.thinned is not None
    lines = [
        f"# DATA: {brief.name}",
        "",
        f"File `{brief.data.path.name}`, sha256 `{_sha256(brief.data.path)[:16]}`.",
        f"Clock: {brief.data.clock}. Grid: {label(brief.grid.step)} steps, {len(panel):,} "
        f"points, {panel.time(0)} to {panel.time(-1)}; warm-up {label(brief.grid.warmup)}.",
        "",
        "Scoring windows (warm-up removed):",
        *[f"- {scoring_window(brief, panel, name)}" for name in ("train", "val", "test")],
        "",
    ]
    if thinned:
        lines += _thinned_md(brief)
    kept = " On this thinned file they describe the kept rows, not the market." if thinned else ""
    lines += [
        "## Streams",
        "",
        "Gaps are between distinct timestamps; `stale_share` is the share of grid "
        "points whose latest event is older than one grid step." + kept,
        "",
        _markdown(stream_table(panel)),
        "",
    ]
    if thinned:
        lines += [
            "The cross-stream lags and trade stamps of the raw file are in the raw-file "
            "section at the end; on the kept rows they would measure the thinning.",
            "",
        ]
    elif brief.grid.bars:
        lines += [
            f"The millisecond tables (cross-stream lag, trade stamps against the book) are left "
            f"out: the grid step is {words(brief.grid.step)}, and 100 ms returns on bars mean "
            "nothing.",
            "",
        ]
    else:
        lines += [
            "## Cross-stream lag",
            "",
            "Lag that maximises the correlation of 100 ms log returns over +-500 ms, "
            "searched in 10 ms then 1 ms steps. Positive: the first stream moves first. "
            f"A peak correlation under {CLEAR_PEAK} is not a clear peak and its lag is noise.",
            "",
            _markdown(lag_table(panel)),
            "",
            "## Trade stamps against the book",
            "",
            "Share of trades printed at the touch (buy at the ask, sell at the bid) "
            "when the book is read just before trade time + offset, offsets -200 to "
            "+20 ms in 1 ms steps. A peak at a negative offset means trades are "
            "stamped late relative to the book.",
            "",
            _markdown(touch_table(panel)),
            "",
        ]
    if brief.data.touch is not None and getattr(panel, "touch_source", None) is not None:
        lines += _touch_md(brief, panel)
    book = brief.data.streams.get("book")
    if book is not None and "age_ms" in book.fields:
        lines += _venue_facts_md(brief, panel)
    if brief.data.depth is not None and brief.data.depth.path.is_file():
        lines += _depth_md(brief, panel)
    if brief.data.cost_model is not None:
        lines += _cost_md(brief)
    if brief.archetypes is not None and brief.archetypes.cross_venue:
        lines += _venues_md(brief, panel)
    if brief.has_families or brief.has_archetypes:
        lines += _families_md(brief, panel)
    lines += ["## Caveats", "", *_caveats(brief)]
    return "\n".join(lines) + "\n"


def _thinned_md(brief: Brief) -> list[str]:
    thin = brief.data.thinned
    stamps = ", ".join(_offset(o) for o in thin.offsets)
    return [
        "## A thinned file",
        "",
        f"For every grid point g this file keeps the last book row at or before each of the "
        f"stamps {stamps}, and drops the rest; the judge reads the book only there, so the "
        "panel, the fills and every forward return are what the raw file would give. Each "
        "kept book row carries `n`, the raw updates it stands for, so `cum_book` counts them "
        "all. Trades are summed per side between the panel's reads: one row per side and "
        "step at the stamp of that side's last trade, `qty` the side's total, `price` its "
        "volume-weighted price and `n` its number of trades. Volumes, notionals, trade counts "
        "and the side of the last trade are exact at every grid point; `price` and `qty` of "
        "the last trade are the step's per-side VWAP and total, not one print.",
        "",
        "Reads at other stamps would see the kept row before them, so the judge leaves out "
        "the horizons and offsets the file cannot serve: decay horizons under one step that "
        "do not end on a kept stamp, and markout offsets likewise. Lead-lag inside one step "
        "is in the raw-file section, where it was measured before thinning.",
        "",
    ]


def _touch_md(brief: Brief, panel: Panel) -> list[str]:
    """The touch stream the fill engine reads, over this workspace's rows
    (:func:`blinded.bookstats.touch_facts`)."""
    from blinded import bookstats, sim_maker
    from blinded.demo_mm import facts_lines

    spec = brief.data.touch
    t = sim_maker.touch_of(panel, spec.instrument)
    end = int(panel.grid_ns[-1]) + panel.step_ns
    facts = bookstats.touch_facts(t.book_ns, t.bid, t.ask, t.bid_qty, t.ask_qty, t.tick, end,
                                  t.trade_ns, t.side, t.price, t.qty)  # fmt: skip
    return [
        "## The touch stream",
        "",
        f"The fill engine reads every top-of-book row of {spec.instrument} from the touch file "
        f"`{spec.path.name}` ({facts['rows']:,} rows here), in ticks of {spec.tick_size:g}, and "
        "each trade at its own stamp; features never see it. Over this workspace's rows:",
        "",
        *facts_lines(facts),
        "",
        "A fall of the best bid is traded out when the sells at the old price from 100 ms "
        "before its last row to the fall reach the size it showed, pulled when there were none, "
        "and partly each otherwise. Where most levels are traded out, a quote at the back of "
        "the touch fills when the level is swept; where many are pulled, the two queue models "
        "disagree most.",
        "",
    ]


def _book_arrays(panel: Panel, sym: str) -> dict:
    ev = panel.events("book", sym)
    out = {"ts": ev.frame["ts"].dt.epoch("ns").to_numpy()}
    return out | {f: ev.column(f) for f in ("bid", "ask", "bid_qty", "ask_qty", "age_ms")
                  if f in ev.fields}  # fmt: skip


def _venue_facts_md(brief: Brief, panel: Panel) -> list[str]:
    """Each venue's quotes and prints over this workspace's rows, and each declared
    cross-venue pair's touch against the other's."""
    from blinded import bookstats
    from blinded.demo_mm import cross_lines, flow_table, venue_table

    end = int(panel.grid_ns[-1]) + panel.step_ns
    touch = brief.data.touch
    books, facts, flows = {}, {}, {}
    for sym in panel.symbols:
        try:
            b, t = _book_arrays(panel, sym), panel.events("trades", sym)
        except KeyError:
            continue
        books[sym] = b
        tt = t.frame["ts"].dt.epoch("ns").to_numpy()
        px, qty, side = t.column("price"), t.column("qty"), t.column("side")
        tick = touch.tick_size if touch is not None and touch.instrument == sym else None
        facts[sym] = bookstats.venue_facts(b["ts"], b["bid"], b["ask"], b["bid_qty"], end, tt,
                                           px, qty, b.get("age_ms"), tick)  # fmt: skip
        flows[sym] = bookstats.flow_facts(b["ts"], b["bid"], b["ask"], tt, side, px, qty)
    out = [
        "## The venues on this file",
        "",
        "Every book row of each venue is on the one receive clock; `age_ms` is how long after "
        "the exchange's stamp the recorder received it, none where the venue sends no stamp. "
        "Over this workspace's rows:",
        "",
        *venue_table(facts),
        "",
    ]
    spec = brief.archetypes
    for a, b in spec.cross_venue if spec is not None else ():
        if a in books and b in books:
            out += [*cross_lines(a, b, bookstats.cross_facts(books[a], books[b], end)), ""]
    out += [
        "## Who prints on each venue",
        "",
        "A flow of near-uniform sizes at a regular pace, printing at prices nobody showed as "
        "the touch, is one counterparty, not a market: a quote that fills against it fills "
        "against that one process, which may change when the quote is there.",
        "",
        *flow_table(flows),
        "",
    ]
    return out


def _depth_md(brief: Brief, panel: Panel) -> list[str]:
    """The depth file against the touch of its venue, over this workspace's rows."""
    from blinded import bookstats
    from blinded.demo_mm import snapshot_lines

    spec = brief.data.depth
    depth = pl.read_parquet(spec.path)
    b = _book_arrays(panel, spec.venue)
    d = {c: depth[c].to_numpy() for c in ("bid0_px", "ask0_px", "bid0_qty", "ask0_qty")}
    ts = depth["ts"].dt.epoch("ns").to_numpy()
    out = bookstats.snapshot_agreement(b["ts"], b["bid"], b["ask"], b["bid_qty"], b["ask_qty"],
                                       ts, d["bid0_px"], d["ask0_px"], d["bid0_qty"],
                                       d["ask0_qty"])  # fmt: skip
    return [
        "## The depth file",
        "",
        f"`{spec.path.name}` holds {spec.venue}'s book to {spec.levels} levels a side at every "
        f"snapshot ({len(ts):,} rows here), read only by the hedge leg as-of its arrival; its "
        f"level 0 trails the touch stream by about {spec.lag_ms:g} ms. Over this workspace's "
        "rows:",
        "",
        *snapshot_lines(out, spec.venue),
        "",
    ]


def _cost_md(brief: Brief) -> list[str]:
    cost = brief.data.cost_model
    rows = pl.DataFrame([{"instrument": k, "half_spread_bps": v}
                         for k, v in cost.half_spread_bps.items()])  # fmt: skip
    return [
        "## Fills on bars",
        "",
        cost.line()[0].upper() + cost.line()[1:] + ". Each grid point is a bar boundary: the "
        "panel at g holds the bar that closed 1 ms before it, `SYM.mid` is that bar's close and "
        "`SYM.bid`, `SYM.ask` the close minus and plus the half spread below. An order decided "
        "at g fills one bar later, at the next bar's close plus or minus the half spread, for "
        "longs and shorts alike: neither the best nor the worst price of the bar, and no "
        "impact. The trades are two rows per bar, the taker sells then the taker buys, at the "
        "bar's VWAP; their trade counts `n` are an estimate.",
        "",
        _markdown(rows),
        "",
    ]


def _venues_md(brief: Brief, panel: Panel) -> list[str]:
    """The feed cadence of the second venue of each cross-venue pair, on this panel."""
    from blinded.demo import cadence_line, venue_cadence

    out = ["## Venues", ""]
    for a, b in brief.archetypes.cross_venue:
        ts = panel.events("book", b).frame["ts"].dt.epoch("ns").to_numpy()
        line = cadence_line(b, venue_cadence(ts / 1e6))
        out += [f"{a} and {b} are one product on two venues. {line[0].upper()}{line[1:]}; D11 "
                "of the report repeats it on each segment. On the receive clock a lead of one "
                "venue over the other can be market lead or feed path.", ""]  # fmt: skip
    return out


def _offset(o: pd.Timedelta) -> str:
    ms = o / pd.Timedelta("1ms")
    return "g" if ms == 0 else f"g {'+' if ms > 0 else '-'} {abs(ms):g} ms"


def _families_md(brief: Brief, panel: Panel) -> list[str]:
    """The lead-lag table and the pairs table on train, for the families the brief has."""
    win = scoring_window(brief, panel, "train")
    out = []
    lead = lead_lag(brief, panel, win)
    if lead:
        fill, step = lead[0]["fill_ms"], lead[0]["step_ms"]
        keep = ("leader", "follower", "lag_at_peak_ms", "c_at_peak", "c_after_fill",
                "best_after_fill", "usable_by_taker")  # fmt: skip
        out += [
            "## Lead-lag on train",
            "",
            f"C(tau) is the correlation of the leader's last {step:g} ms log return at each "
            f"grid point with the follower's {step:g} ms return ending tau later, over whole "
            "steps from -10 to +10"
            + ("" if brief.data.thinned else " and 10 ms steps across one step either side")
            + ". A follower that copies with a delay peaks at that delay. `c_after_fill` is C "
            f"on the follower's first whole step after a taker's fill ({fill:g} ms after the "
            f"decision, so tau = {fill + step:g} ms), the step a taker holds; "
            "`best_after_fill` the largest C from there on; `usable_by_taker` says yes when "
            "that is positive and at least half the peak, a rule of thumb.",
            "",
            _markdown(pl.DataFrame([{k: r[k] for k in keep} for r in lead])),
            "",
        ]
    rows = pairs_table(brief, panel, win)
    if rows:
        d = brief.family("pairs").defaults
        shown = ""
        if len(rows) > PAIRS_SHOWN:
            rows = sorted(rows, key=_by_half_life)[:PAIRS_SHOWN]
            shown = (f" The {PAIRS_SHOWN} of the {len(brief.family('pairs').pairs)} pairs with "
                     "the shortest median half-life are shown.")  # fmt: skip
        out += [
            "## Pairs on train",
            "",
            f"Each pair A over B fitted as `bl.pairs` fits it at the brief's defaults: OLS of "
            f"ln m_A on ln m_B over the trailing {d['window']}, refitted at each {d['every']} "
            "clock window. `hedge_ratio` and `residual_sd_bps` are the fit in force at the "
            "end of train; `median_half_life_s` is the AR(1) half-life of the spread in each "
            "clock window, the median over windows (none when most did not revert), and "
            "`no_reversion` the share of windows where it did not come back at all. A pairs "
            "feature is scored against the spread return at equal notional, A's forward log "
            "return less B's." + shown,
            "",
            _markdown(pl.DataFrame(rows)),
            "",
        ]
    return out


PAIRS_SHOWN = 10  # DATA.md lists this many pairs when the brief implies more


def _by_half_life(row: dict) -> tuple:
    """Shortest median half-life first; pairs without one, which mostly did not revert, last."""
    v = row["median_half_life_s"]
    return (not np.isfinite(v), v if np.isfinite(v) else 0.0)


def pairs_table(brief: Brief, panel: Panel, window) -> list[dict]:
    """Per pair of the brief's pairs family, the fit at its defaults over ``window``."""
    from blinded.durations import to_ns
    from blinded.features import cross
    from blinded.markouts import window_half_lives

    family = brief.family("pairs")
    if family is None:
        return []
    d = family.defaults
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [window.start.value, window.end.value])
    out = []
    for a, b in family.pairs:
        beta, mu, var, y, x = cross._fit(a, b, d["window"], d["every"])
        fit = panel.frame.lazy().select(beta=beta, dev=y - beta * x - mu, sd=var.sqrt()).collect()
        beta = fit["beta"].fill_null(np.nan).to_numpy()[i0:i1]
        sd = fit["sd"].fill_null(np.nan).to_numpy()[i0:i1]
        dev = fit["dev"].fill_null(np.nan).to_numpy()[i0:i1]
        lives = window_half_lives(dev, g[i0:i1], to_ns(d["every"]), panel.step_ns / 1e9)
        fitted = [w for w in lives if w["phi"] is not None]
        # a window that did not revert counts as an infinite half-life, one that overshot
        # every step (phi <= 0) as none
        life = [np.inf if w["reverts"] is False else w["half_life_s"] for w in fitted]
        life = np.array([v for v in life if v is not None])
        median = float(np.median(life)) if len(life) else np.nan
        last = np.flatnonzero(np.isfinite(beta) & np.isfinite(sd))
        out.append({
            "pair": f"{a} over {b}",
            "hedge_ratio": float(beta[last[-1]]) if len(last) else np.nan,
            "residual_sd_bps": float(sd[last[-1]]) * 1e4 if len(last) else np.nan,
            "median_half_life_s": median if np.isfinite(median) else np.nan,
            "no_reversion": (sum(w["reverts"] is False for w in fitted) / len(fitted)
                             if fitted else np.nan),
            "windows": len(fitted),
        })  # fmt: skip
    return out


def _caveats(brief: Brief) -> list[str]:
    notes = [
        "- Rows with identical timestamps resolve by file order, last wins.",
        "- Which instruments are in the file, and why, is for the user to own.",
    ]
    if brief.data.clock == "receive":
        notes.append(
            "- Receive clock: a lead of one stream over another can be market "
            "lead or feed delay; these numbers cannot tell them apart."
        )
    elif brief.data.default_delay_ms or brief.data.delay_overrides:
        notes.append(
            f"- Exchange clock: the brief assumes each row reaches you "
            f"{brief.data.default_delay_ms:g} ms after its stamp (overrides aside), so a "
            "decision at g sees stamps up to g less that delay, and an order's latency runs "
            "from g." + _overrides(brief)
        )
    else:
        notes.append(
            "- Exchange clock: nothing here says when a row reached you; set "
            "availability delays in the brief."
        )
    return notes


def _overrides(brief: Brief) -> str:
    """The brief's availability overrides in words, e.g. a funding stream stamped on its
    settlement and seen a second later; nothing when there are none."""
    rules = brief.data.delay_overrides
    if not rules:
        return ""
    each = [f"the {r.get('stream', 'every')} stream"
            + (f" of {r['instrument']}" if r.get("instrument") else "")
            + f" {float(r['delay_ms']):g} ms" for r in rules]  # fmt: skip
    return " Overrides: " + "; ".join(each) + "."


def _markdown(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "(none)"
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in frame.iter_rows()]
    return "\n".join([head, rule, *body])


def _cell(value) -> str:
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if np.isnan(value) else f"{value:.4g}"
    if isinstance(value, int | np.integer):
        return f"{value:,}"
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
