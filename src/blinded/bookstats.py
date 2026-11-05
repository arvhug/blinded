"""Measurements of a recorded book: how wide and deep the touch is, how a touch level ends,
how a depth snapshot agrees with the touch stream, and what one venue's quotes and flow look
like against another's.

Every function takes flat arrays (stamps in ns on one clock, prices in price units or in
ticks as named) and returns plain numbers, so the builders of :mod:`blinded.demo_mm` print
them over a whole day in ``DATA-raw.md`` and :mod:`blinded.diagnostics` prints them over
train in ``DATA.md`` from the same code.
"""

from __future__ import annotations

import numpy as np

MS = 1_000_000
SECOND = 1_000 * MS
# a fall of the best bid counts the sells at the old price from this long before the row
# that showed the old level last, to the row that shows the fall
FALL_LOOKBACK = 100 * MS
# a level is traded out when the sells at its price reach its displayed size, to rounding
TRADED_OUT = 0.999


def tw_quantile(ts: np.ndarray, x: np.ndarray, q: float, end: int) -> float:
    """The time-weighted quantile ``q`` of a step function ``x`` that holds from ``ts[i]`` to
    ``ts[i + 1]``, the last value to ``end``.

    Examples
    --------
    >>> tw_quantile(np.array([0, 10, 20]), np.array([1.0, 5.0, 3.0]), 0.5, 100)
    3.0
    """
    w = np.diff(np.append(ts, end)).astype(float)
    if not len(x) or w.sum() <= 0:
        return float("nan")
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(int(np.searchsorted(cw, q)), len(x) - 1)])


def time_share(ts: np.ndarray, flag: np.ndarray, end: int) -> float:
    """The share of the time from ``ts[0]`` to ``end`` that ``flag`` holds, as a step function.

    Examples
    --------
    >>> time_share(np.array([0, 10, 40]), np.array([True, False, True]), 100)
    0.7
    """
    w = np.diff(np.append(ts, end)).astype(float)
    return float(w[np.asarray(flag, dtype=bool)].sum() / w.sum()) if w.sum() > 0 else float("nan")


def falls(ts: np.ndarray, bid: np.ndarray, bid_qty: np.ndarray, sell_ts: np.ndarray,
          sell_price: np.ndarray, sell_qty: np.ndarray) -> dict:  # fmt: skip
    """How each fall of the best bid came about: traded out, pulled, or partly each.

    A fall is a book row whose bid is below the row before's. The sells at
    the old price stamped from :data:`FALL_LOOKBACK` before the row before to
    the falling row are what the level was traded with: at least the displayed
    size (to :data:`TRADED_OUT`) is traded out, none at all is pulled, and
    anything between is partly each. ``bid`` and ``sell_price`` are in one
    unit (ticks, or prices on one tick: they are compared to half of it).

    Examples
    --------
    Rows every 200 ms; three falls of a 10-lot bid: swept by 10 at 100 ms, by nothing, and
    by 4 at 900 ms.

    >>> ts = np.arange(7) * 200 * MS
    >>> bid = np.array([5, 4, 5, 4, 5, 4, 4])
    >>> out = falls(ts, bid, np.full(7, 10.0), np.array([100, 900]) * MS, np.array([5, 5]),
    ...             np.array([10.0, 4.0]))
    >>> out["falls"], [round(out[k], 4) for k in ("traded_out", "pulled", "partly")]
    (3, [0.3333, 0.3333, 0.3333])
    """
    down = np.flatnonzero(bid[1:] < bid[:-1]) + 1
    if not len(down):
        return {"falls": 0, "traded_out": np.nan, "pulled": np.nan, "partly": np.nan}
    old, shown, before = bid[down - 1], bid_qty[down - 1], ts[down - 1]
    lo = np.searchsorted(sell_ts, before - FALL_LOOKBACK, side="left")
    hi = np.searchsorted(sell_ts, ts[down], side="right")
    traded = np.zeros(len(down))
    for i in range(len(down)):
        a, b = lo[i], hi[i]
        if b > a:
            at = np.abs(sell_price[a:b] - old[i]) < 0.5
            traded[i] = sell_qty[a:b][at].sum()
    out_ = traded >= shown * TRADED_OUT
    none = traded == 0
    a, b = float(out_.mean()), float(none.mean())
    return {"falls": int(len(down)), "traded_out": a, "pulled": b, "partly": 1 - a - b}


def touch_facts(ts: np.ndarray, bid: np.ndarray, ask: np.ndarray, bid_qty: np.ndarray,
                ask_qty: np.ndarray, tick: float, end: int, trade_ts: np.ndarray,
                trade_side: np.ndarray, trade_price: np.ndarray,
                trade_qty: np.ndarray) -> dict:  # fmt: skip
    """The rows of one touch stream over ``[ts[0], end)``.

    ``bid`` and ``ask`` are in ticks of ``tick``; ``trade_price`` too, and
    ``trade_side`` is +1 for a buy, -1 for a sell. The spread's shares are of
    time on the stream's clock; sizes are medians over rows; the falls are
    :func:`falls` on the sells.
    """
    mid = (bid + ask) * (tick / 2)
    wide = ask - bid
    hours = (end - ts[0]) / 3600e9 if len(ts) else float("nan")
    med_mid = float(np.median(mid)) if len(mid) else float("nan")
    changes = np.flatnonzero(bid[1:] != bid[:-1]) + 1
    same = bid[1:] == bid[:-1]
    sell = trade_side < 0
    return {
        "rows": int(len(ts)),
        "rows_per_second": len(ts) / (hours * 3600) if hours else float("nan"),
        "distinct_stamps": int(len(np.unique(ts))),
        "mid": med_mid,
        "tick": tick,
        "tick_bps": tick / med_mid * 1e4,
        "one_tick_share": time_share(ts, wide == 1, end),
        "two_tick_share": time_share(ts, wide == 2, end),
        "bid_qty_median": float(np.median(bid_qty)),
        "ask_qty_median": float(np.median(ask_qty)),
        "touch_usd_median": float(np.median(bid_qty)) * med_mid,
        "trades": int(len(trade_ts)),
        "trades_per_second": len(trade_ts) / (hours * 3600) if hours else float("nan"),
        "bid_change_median_ms": float(np.median(np.diff(ts[changes]))) / MS
        if len(changes) > 1
        else float("nan"),  # fmt: skip
        "bid_same_rows": int(same.sum()),
        "bid_size_changes": int(same.sum() - np.sum(same & (bid_qty[1:] == bid_qty[:-1]))),
        **falls(ts, bid, bid_qty, trade_ts[sell], trade_price[sell], trade_qty[sell]),
    }


def aggressors(stamp: np.ndarray, price: np.ndarray, qty: np.ndarray, tick: float) -> dict:
    """Trades that share an exchange stamp, taken as one aggressor order: how many, their
    size's median, p90 and p99, and the share that walked more than one tick.

    Examples
    --------
    >>> out = aggressors(np.array([1, 1, 2]), np.array([1.0, 1.02, 1.0]), np.array([1.0, 2, 4]),
    ...                  0.01)
    >>> out["orders"], out["size_median"], out["walked"]
    (2, 3.5, 0.5)
    """
    _, first, inverse = np.unique(stamp, return_index=True, return_inverse=True)
    size = np.bincount(inverse, weights=qty)
    hi = np.full(len(first), -np.inf)
    lo = np.full(len(first), np.inf)
    np.maximum.at(hi, inverse, price)
    np.minimum.at(lo, inverse, price)
    return {"orders": int(len(first)), "size_median": float(np.median(size)),
            "size_p90": float(np.quantile(size, 0.9)), "size_p99": float(np.quantile(size, 0.99)),
            "walked": float(np.mean((hi - lo) / tick > 0.5))}  # fmt: skip


def delay_ms(local: np.ndarray, exchange: np.ndarray) -> dict:
    """The collector's delay, ``local - exchange``, median and p99 in ms; None for both when
    the venue sends no exchange stamp (the two are equal on every row)."""
    d = np.asarray(local, dtype=np.int64) - np.asarray(exchange, dtype=np.int64)
    if not len(d) or not np.any(d):
        return {"median": None, "p99": None}
    return {"median": float(np.median(d)) / MS, "p99": float(np.quantile(d, 0.99)) / MS}


def snapshot_agreement(t_ts, t_bid, t_ask, t_bq, t_aq, s_ts, s_bid, s_ask, s_bq, s_aq,
                       lags_ms=(0, 10, 100)) -> dict:  # fmt: skip
    """A depth snapshot's level 0 against the touch stream.

    ``forward[lag]`` is, over the touch rows, the share whose touch the
    snapshot as-of the row's stamp plus ``lag`` ms shows at its level 0, by
    price (both sides) and by size (both sides); ``reverse`` the same over the
    snapshot rows, the touch as-of each; ``crossed`` the share of touch rows
    whose bid is at or above the as-of snapshot's ask or whose ask is at or
    below its bid. A stamp before the other stream's first row reads its first
    row. Prices in one unit on both streams.
    """
    out: dict = {"touch_rows": int(len(t_ts)), "snapshot_rows": int(len(s_ts)), "forward": {}}

    def asof(ts, at):
        return np.clip(np.searchsorted(ts, at, side="right") - 1, 0, len(ts) - 1)

    for lag in lags_ms:
        k = asof(s_ts, t_ts + lag * MS)
        price = (s_bid[k] == t_bid) & (s_ask[k] == t_ask)
        size = np.isclose(s_bq[k], t_bq) & np.isclose(s_aq[k], t_aq)
        out["forward"][lag] = (float(price.mean()), float(size.mean()))
    k = asof(t_ts, s_ts)
    price = (s_bid == t_bid[k]) & (s_ask == t_ask[k])
    size = np.isclose(s_bq, t_bq[k]) & np.isclose(s_aq, t_aq[k])
    out["reverse"] = (float(price.mean()), float(size.mean()))
    k = asof(s_ts, t_ts)
    out["crossed"] = float(np.mean((t_bid >= s_ask[k]) | (t_ask <= s_bid[k])))
    return out


def venue_facts(ts: np.ndarray, bid: np.ndarray, ask: np.ndarray, bid_qty: np.ndarray,
                end: int, trade_ts: np.ndarray, trade_price: np.ndarray, trade_qty: np.ndarray,
                age_ms: np.ndarray | None = None, tick: float | None = None) -> dict:  # fmt: skip
    """One venue's quotes and prints over ``[ts[0], end)``:
    the mid and its tick in bps; the spread in bps, time-weighted median, p10 and p90;
    the bid's median size in USD; rows a second, the gap between rows (median and p99 ms)
    and the share of the time with no row for 60 s; the collector delay from ``age_ms``
    (receive less exchange stamp, None where the venue sends none); trades and USD.
    Prices in price units."""
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid * 1e4
    span = end - ts[0] if len(ts) else 0
    gaps = np.diff(np.append(ts, end))
    med = tw_quantile(ts, mid, 0.5, end)
    usd = trade_price * trade_qty
    d = {"median": None, "p99": None}
    if age_ms is not None and np.any(age_ms):
        d = {"median": float(np.median(age_ms)), "p99": float(np.quantile(age_ms, 0.99))}
    return {
        "rows": int(len(ts)), "rows_per_second": len(ts) / (span / 1e9) if span else float("nan"),
        "mid": med, "tick": tick, "tick_bps": tick / med * 1e4 if tick else None,
        "spread_bps": tw_quantile(ts, spread, 0.5, end),
        "spread_p10": tw_quantile(ts, spread, 0.1, end),
        "spread_p90": tw_quantile(ts, spread, 0.9, end),
        "touch_usd": float(np.median(bid_qty * bid)) if len(ts) else float("nan"),
        "gap_median_ms": float(np.median(np.diff(ts))) / MS if len(ts) > 1 else float("nan"),
        "gap_p99_ms": float(np.quantile(np.diff(ts), 0.99)) / MS if len(ts) > 1 else float("nan"),
        "silent_60s": float(gaps[gaps > 60 * SECOND].sum() / span) if span else float("nan"),
        "delay_ms": d, "trades": int(len(trade_ts)), "usd": float(usd.sum()),
    }  # fmt: skip


def cross_facts(a: dict, b: dict, end: int, distance_bps: float = 30.0) -> dict:
    """Venue A's touch against venue B's, time-weighted on the merged clock from the later
    first row to ``end``: A's mid less B's in bps of B's,
    median and absolute median; the share of time A's ask is at or below B's bid (a visible
    cross); the share a bid ``distance_bps`` under B's bid is above A's best bid and under
    its ask, so it would be A's best. ``a`` and ``b`` hold ``ts``, ``bid``, ``ask``."""
    ts = np.union1d(a["ts"], b["ts"])
    ts = ts[ts >= max(a["ts"][0], b["ts"][0])]
    ja = np.searchsorted(a["ts"], ts, side="right") - 1
    jb = np.searchsorted(b["ts"], ts, side="right") - 1
    ab, aa, bb, ba = a["bid"][ja], a["ask"][ja], b["bid"][jb], b["ask"][jb]
    bm = (bb + ba) / 2
    off = ((ab + aa) / 2 - bm) / bm * 1e4
    q = bb * (1 - distance_bps * 1e-4)
    return {"mid_off_bps": tw_quantile(ts, off, 0.5, end),
            "mid_off_abs_bps": tw_quantile(ts, np.abs(off), 0.5, end),
            "crossed": time_share(ts, aa <= bb, end),
            "distance_bps": distance_bps,
            "bid_is_best": time_share(ts, (q > ab) & ~(q >= aa), end)}  # fmt: skip


def flow_facts(book_ts: np.ndarray, bid: np.ndarray, ask: np.ndarray, trade_ts: np.ndarray,
               side: np.ndarray, price: np.ndarray, qty: np.ndarray) -> dict:  # fmt: skip
    """Who prints on a venue: trades and USD; USD a trade at
    p10, p50 and p90; the smallest and largest share of trades in a 1 USD bucket from 1 to
    34 USD; the gap between trades, median s and coefficient of variation (1 for a Poisson
    flow, 0 for a clock); trades an hour from the first, least and most; the buy share; the
    share printed at the touch shown just before (a buy at the ask, a sell at the bid); and
    the share at a price never shown as the touch on its side within a second either way.
    ``side`` is +1 for a buy."""
    usd = price * qty
    gaps = np.diff(trade_ts) / 1e9
    hours = np.bincount(((trade_ts - trade_ts[0]) // (3600 * SECOND)).astype(int))
    buckets = np.round(usd)
    shares = [float(np.mean(buckets == v)) for v in range(1, 35)]
    j = np.searchsorted(book_ts, trade_ts, side="right") - 1
    k = np.maximum(j, 0)
    buy = side > 0
    at = np.where(buy, price == ask[k], price == bid[k]) & (j >= 0)
    lo = np.clip(np.searchsorted(book_ts, trade_ts - SECOND, side="left"), 0, len(book_ts) - 1)
    hi = np.clip(np.searchsorted(book_ts, trade_ts + SECOND, side="right") - 1, 0,
                 len(book_ts) - 1)  # fmt: skip
    seen = np.zeros(len(trade_ts), dtype=bool)
    for i in range(len(trade_ts)):
        shown = ask[lo[i] : hi[i] + 1] if buy[i] else bid[lo[i] : hi[i] + 1]
        seen[i] = bool(np.any(shown == price[i]))
    return {
        "trades": int(len(trade_ts)), "usd": float(usd.sum()),
        "usd_p10": float(np.quantile(usd, 0.1)), "usd_p50": float(np.median(usd)),
        "usd_p90": float(np.quantile(usd, 0.9)),
        "bucket_min": min(shares), "bucket_max": max(shares),
        "gap_median_s": float(np.median(gaps)) if len(gaps) else float("nan"),
        "gap_cv": float(gaps.std() / gaps.mean()) if len(gaps) and gaps.mean() else float("nan"),
        "hour_min": int(hours.min()), "hour_max": int(hours.max()),
        "buy_share": float(buy.mean()), "at_touch": float(at.mean()),
        "never_shown": float(1 - seen.mean()),
    }  # fmt: skip
