"""Markouts and the other signal diagnostics the report draws.

Pure numpy over arrays the report hands in, so each can be checked by hand:
the markout fan around a signal's extreme moments or a strategy's orders,
with its standard error, the signal's decay and its IC per clock window, and
the cost a taker has to clear.

Markouts are on the clock of the decision, when the order is sent: offset 0 is
the send, the latency later is the fill, and negative offsets show what the
price did before the order went out.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from blinded.metrics import rank

FAN_BEFORE = "10s"
FAN_AFTER = "60s"
FAN_GAP = "70s"  # before + after: no two events' paths share a second of price
FAN_TAILS = (0.05, 0.95)  # the train quantiles past which a feature's value is an event
DECAY_HORIZONS = ("100ms", "250ms", "500ms", "1s", "2s", "5s", "10s", "20s", "30s", "60s")


def rank_ic(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman correlation over the points where both are finite; None under 3 points."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(np.corrcoef(rank(x[ok]), rank(y[ok]))[0, 1])


def events(
    x: np.ndarray, t_ns: np.ndarray, lo: float, hi: float, gap_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Where ``x`` is above ``hi`` or below ``lo``, each at least ``gap_ns`` after the last kept.

    Returns the indices and a sign for each, +1 above ``hi`` and -1 below
    ``lo``. The scan takes the first point past either edge, skips
    ``gap_ns``, and takes the next, so a signal that stays extreme counts once
    per gap. With the gap as long as the window the fan draws, no two events'
    paths overlap and their mean has an honest standard error.

    Examples
    --------
    One point a second and a gap of 2 s: the point at 2 s is past an edge but
    too soon after the one at 1 s.

    >>> idx, sign = events(np.array([0.0, 5.0, -5.0, 0.0, -5.0]), np.arange(5) * 10**9,
    ...                    -1.0, 1.0, 2 * 10**9)
    >>> idx.tolist(), sign.tolist()
    ([1, 4], [1.0, -1.0])
    """
    past = np.flatnonzero((x > hi) | (x < lo))
    times = t_ns[past]
    kept, i = [], 0
    while i < len(past):
        kept.append(past[i])
        i = int(np.searchsorted(times, times[i] + gap_ns, side="left"))
    idx = np.asarray(kept, dtype=np.int64)
    return idx, np.where(x[idx] > hi, 1.0, -1.0)


def clusters(t_ns: np.ndarray, gap_ns: int) -> np.ndarray:
    """A cluster id for each of a sorted run of times, new wherever the last is ``gap_ns`` back.

    Orders closer together than the fan's window share price, so the fan's
    standard error counts each cluster as one draw.

    Examples
    --------
    >>> clusters(np.array([0, 1, 5, 6, 20]), 3).tolist()
    [0, 0, 1, 1, 2]
    """
    if len(t_ns) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.r_[0, np.cumsum(np.diff(t_ns) >= gap_ns)]


def fan(
    moves: list[np.ndarray],
    sign: np.ndarray,
    weight: np.ndarray | None = None,
    cluster: np.ndarray | None = None,
) -> dict:
    """The mean move in the traded direction at each offset, its standard error, longs and shorts.

    ``moves`` holds one array per offset with one value per event, as
    :func:`mid_moves` gives, and each is multiplied by the event's ``sign``.
    The mean is weighted by ``weight`` (one each when None). The standard
    error treats events that share a ``cluster`` id as one draw (each event
    its own when None):

    .. math:: \\mathrm{SE} = \\frac{1}{\\sum_i w_i}\\sqrt{\\frac{C}{C-1}\\sum_c
              \\Big(\\sum_{i \\in c} w_i (v_i - \\bar v)\\Big)^2}

    over :math:`C` clusters, which for unweighted events in clusters of one is
    the sample standard deviation over :math:`\\sqrt n`. An event whose move is
    NaN at an offset, past the end of the data, is left out there.

    Examples
    --------
    A long that made 2 bps, a short whose price fell 4 bps, a long that lost 1:

    >>> out = fan([np.array([2.0, -4.0, -1.0])], np.array([1.0, -1.0, 1.0]))
    >>> [round(out[k][0], 6) for k in ("mean_bps", "se_bps", "long_bps", "short_bps")]
    [1.666667, 1.452966, 0.5, 4.0]
    """
    sign = np.asarray(sign, dtype=float)
    w = np.ones(len(sign)) if weight is None else np.asarray(weight, dtype=float)
    c = np.arange(len(sign)) if cluster is None else np.asarray(cluster)
    out = {"n": int(len(sign)), "n_long": int((sign > 0).sum()),
           "n_short": int((sign < 0).sum()), "mean_bps": [], "se_bps": [], "long_bps": [],
           "short_bps": []}  # fmt: skip
    for m in moves:
        v = sign * m
        ok = np.isfinite(v)
        mean, se = _clustered(v[ok], w[ok], c[ok])
        out["mean_bps"].append(mean)
        out["se_bps"].append(se)
        for key, side in (("long_bps", sign > 0), ("short_bps", sign < 0)):
            out[key].append(_clustered(v[ok & side], w[ok & side], c[ok & side])[0])
    return out


def _clustered(v: np.ndarray, w: np.ndarray, c: np.ndarray) -> tuple[float | None, float | None]:
    total = w.sum()
    if len(v) == 0 or total <= 0:
        return None, None
    mean = float((w * v).sum() / total)
    _, inverse = np.unique(c, return_inverse=True)
    sums = np.bincount(inverse, weights=w * (v - mean))
    k = len(sums)
    se = float(np.sqrt(k / (k - 1) * (sums**2).sum()) / total) if k > 1 else None
    return mean, se


def window_ics(
    x: np.ndarray, y: np.ndarray, grid_ns: np.ndarray, every_ns: int
) -> list[tuple[int, float | None]]:
    """Rank IC inside each clock window of ``every_ns``, as ``(window start ns, ic)``.

    Windows are aligned to the clock (``start = t - t mod every``), as
    ``ts.dt.truncate`` aligns them.
    """
    key = grid_ns - grid_ns % every_ns
    starts, first = np.unique(key, return_index=True)
    bounds = np.r_[first, len(key)]
    return [(int(s), rank_ic(x[a:b], y[a:b]))
            for s, a, b in zip(starts, bounds[:-1], bounds[1:], strict=True)]  # fmt: skip


def mid_moves(
    mark: Callable[[np.ndarray], np.ndarray], t_ns: np.ndarray, offsets_ns: list[int], last_ns: int
) -> list[np.ndarray]:
    """``(mark(t + offset) / mark(t) - 1)`` in bps for each offset, NaN past ``last_ns``.

    Examples
    --------
    >>> mid = lambda at: np.where(at < 10, 100.0, 101.0)
    >>> [np.round(m, 4).tolist() for m in mid_moves(mid, np.array([5, 12]), [-10, 0, 10], 20)]
    [[0.0, -99.0099], [0.0, 0.0], [100.0, nan]]
    """
    base = mark(t_ns)
    out = []
    for off in offsets_ns:
        at = t_ns + off
        m = (mark(at) / base - 1) * 1e4
        m[at > last_ns] = np.nan
        out.append(m)
    return out


def spread_moves(
    mark_a: Callable[[np.ndarray], np.ndarray],
    mark_b: Callable[[np.ndarray], np.ndarray],
    beta: np.ndarray,
    t_ns: np.ndarray,
    offsets_ns: list[int],
    last_ns: int,
) -> list[np.ndarray]:
    """The move of a pair's log spread from each decision, in bps, NaN past ``last_ns``.

    :math:`[\\ln m_a(t + \\tau) - \\ln m_a(t)] - \\beta_t [\\ln m_b(t + \\tau) - \\ln
    m_b(t)]`, with the hedge ratio held at its value at the decision: the P&L of
    one unit of the spread bought at :math:`t`, before costs.

    Examples
    --------
    >>> a = lambda at: np.where(at < 10, 100.0, 101.0)
    >>> b = lambda at: np.full(len(at), 50.0)
    >>> moves = spread_moves(a, b, np.array([1.0]), np.array([5]), [0, 10], 20)
    >>> [round(float(m[0]), 4) for m in moves]  # ln(101 / 100) in bps
    [0.0, 99.5033]
    """
    la, lb = np.log(mark_a(t_ns)), np.log(mark_b(t_ns))
    out = []
    for off in offsets_ns:
        at = t_ns + off
        move = (np.log(mark_a(at)) - la) - beta * (np.log(mark_b(at)) - lb)
        move[at > last_ns] = np.nan
        out.append(move * 1e4)
    return out


def entry_cost(
    side: np.ndarray, size: np.ndarray, fill: np.ndarray, sent: np.ndarray
) -> float | None:
    """Size-weighted mean of ``side * (fill - sent) / sent`` in bps: what getting in cost.

    ``sent`` is the mark when the order was sent, so this is the half spread
    crossed plus whatever the price did while the order was in flight.

    Examples
    --------
    >>> round(entry_cost(np.array([1.0, -1.0]), np.array([1.0, 1.0]),
    ...                  np.array([100.02, 99.99]), np.array([100.0, 100.0])), 9)
    1.5
    """
    v = side * (fill - sent) / sent * 1e4
    ok = np.isfinite(v)
    w = size[ok]
    return float((v[ok] * w).sum() / w.sum()) if w.sum() > 0 else None


def half_spread_bps(bid: np.ndarray, ask: np.ndarray) -> float | None:
    """The median half spread in bps of the mid, over the points with a two-sided book.

    Examples
    --------
    >>> round(half_spread_bps(np.array([99.99, 99.98]), np.array([100.01, 100.02])), 9)
    1.5
    """
    mid = (bid + ask) / 2
    s = (ask - bid) / mid * 1e4 / 2
    s = s[np.isfinite(s)]
    return float(np.median(s)) if len(s) else None


def downsample(n: int, points: int = 600) -> np.ndarray:
    """At most ``points`` evenly spaced indices into a length-``n`` series, first and last kept.

    Examples
    --------
    >>> downsample(10, 4).tolist()
    [0, 3, 6, 9]
    """
    if n <= points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, points).round().astype(int))


def window_half_lives(
    s: np.ndarray, grid_ns: np.ndarray, every_ns: int, step_s: float
) -> list[dict]:
    """How fast ``s`` reverts inside each clock window of ``every_ns``.

    In each window, :math:`\\hat\\varphi` is the OLS slope of :math:`s_t` on
    :math:`s_{t-1}` with an intercept, over consecutive points that are both
    finite, and the half-life is :math:`-\\ln 2 / \\ln\\hat\\varphi` steps of
    ``step_s`` seconds. ``reverts`` is False where :math:`\\hat\\varphi \\ge
    1`, which has no half-life; a window with :math:`\\hat\\varphi \\le 0`, or
    fewer than three pairs, has none either.

    Examples
    --------
    Each value is half the one before, in one window of 10 s on a 1 s grid:

    >>> out = window_half_lives(np.array([8.0, 4.0, 2.0, 1.0, 0.5]), np.arange(5) * 10**9,
    ...                         10 * 10**9, 1.0)
    >>> out[0]["phi"], out[0]["half_life_s"], out[0]["reverts"]
    (0.5, 1.0, True)
    """
    key = grid_ns - grid_ns % every_ns
    starts, first = np.unique(key, return_index=True)
    bounds = np.r_[first, len(key)]
    out = []
    for start, a, b in zip(starts, bounds[:-1], bounds[1:], strict=True):
        now, before = s[a + 1 : b], s[a : b - 1]
        ok = np.isfinite(now) & np.isfinite(before)
        phi = None
        if ok.sum() >= 3 and np.std(before[ok]) > 0:
            x, y = before[ok] - before[ok].mean(), now[ok] - now[ok].mean()
            phi = float((x * y).sum() / (x * x).sum())
        reverts = phi is not None and phi < 1
        life = -math.log(2) / math.log(phi) * step_s if phi is not None and 0 < phi < 1 else None
        out.append({"start": int(start), "phi": phi, "half_life_s": life,
                    "reverts": bool(reverts) if phi is not None else None})  # fmt: skip
    return out


def random_walk_half_life(n: int, step_s: float, reps: int = 200, seed: int = 0) -> float:
    """The median half-life :func:`window_half_lives` reads on a random walk of ``n`` points.

    The AR(1) slope fitted with an intercept sits below 1 on a random walk,
    by about 4 / ``n`` in the median (the Dickey-Fuller bias), so a window
    reads a finite half-life with no reversion at all. A spread has to be
    well under this line before its half-life says anything. A walk whose
    slope is 1 or more counts as an infinite half-life.

    Examples
    --------
    On 30-minute windows of a 500 ms grid, a random walk reads about five minutes:

    >>> round(random_walk_half_life(3600, 0.5) / 60)
    5
    """
    rng = np.random.default_rng(seed)
    walk = rng.standard_normal((reps, n)).cumsum(axis=1)
    x = walk[:, :-1] - walk[:, :-1].mean(axis=1, keepdims=True)
    y = walk[:, 1:] - walk[:, 1:].mean(axis=1, keepdims=True)
    phi = (x * y).sum(axis=1) / (x * x).sum(axis=1)
    life = np.full(reps, np.inf)
    inside = (phi > 0) & (phi < 1)
    life[inside] = -math.log(2) / np.log(phi[inside]) * step_s
    return float(np.median(life))
