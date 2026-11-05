"""The truncation test: recompute a feature on truncated panels and compare."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from blinded.brief import Window
from blinded.contract import Legs, Output, Quotes, run
from blinded.panel import Panel

QUANTILES = (0.30, 0.50, 0.70, 0.85, 0.95)
RTOL = 1e-7
ATOL = 1e-10


@dataclass(frozen=True)
class Violation:
    """Where a recomputation disagreed with the full run.

    ``cut`` is ``None`` for a determinism failure (two full runs disagreed).
    ``leg`` names the instrument whose positions disagreed, for a strategy
    with several legs, and ``key`` the quote parameter that disagreed, for a
    market-making strategy (:class:`~blinded.contract.Quotes`).
    """

    cut: pd.Timestamp | None
    first: pd.Timestamp
    n_bad: int
    n_compared: int
    max_diff: float
    nan_mismatch: int
    leg: str | None = None
    key: str | None = None

    def __str__(self) -> str:
        where = "two full runs" if self.cut is None else f"cut at {self.cut.isoformat()}"
        where += f", leg {self.leg}" if self.leg else ""
        where += f", key {self.key}" if self.key else ""
        finite = np.isfinite(self.max_diff)
        size = f"max |diff| {self.max_diff:.3g}" if finite else "no finite diff"
        nans = f", {self.nan_mismatch} NaN-vs-number" if self.nan_mismatch else ""
        return (
            f"{where}: first mismatch at {self.first.isoformat()}, "
            f"{self.n_bad}/{self.n_compared} points differ ({size}{nans})"
        )


@dataclass(frozen=True)
class Verdict:
    """Outcome of the determinism check and the truncation test."""

    status: str
    cuts: tuple[pd.Timestamp, ...]
    violation: Violation | None = None
    full: np.ndarray | Legs | None = None

    @property
    def ok(self) -> bool:
        return self.status == "accepted"

    def __str__(self) -> str:
        if self.ok:
            return f"passed: deterministic, matches the full run at all {len(self.cuts)} cuts"
        return f"{self.status}: {self.violation}"


def cut_times(
    panel: Panel, window: Window, seed: int, tail: pd.Timedelta | None = None
) -> list[pd.Timestamp]:
    """Cut times inside the scoring window, the 0.50 cut first.

    The cuts sit at :data:`QUANTILES` of the window's grid points, at one seeded
    random point, and last in the middle of the window's last block of ``tail``
    (the brief's ``stats.block``; without one, the last 2% of the window), so a
    leak confined to the window's end still meets a cut: the 0.95 cut alone
    leaves the last 5% unchecked, and at most half a block is now.

    Every cut sits 0.37 of a grid step after a grid point, so it never
    coincides with one and the open/closed side of each boundary is exercised.
    Each cut then moves, within the next 100 grid points, to the one where for
    as many streams as possible the next event comes after the cut and is
    nearer than the last one, so a forward or nearest join has a future row to
    take. Ties go to the point where the most columns of the frame change at
    the next grid step, so a leak that reads later grid rows sees them differ.
    """
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [window.start.value, window.end.value])
    points = g[i0:i1]
    if len(points) == 0:
        raise ValueError("the scoring window contains no grid points")
    offset = int(0.37 * panel.step_ns)
    rng = np.random.default_rng(seed)
    picks = [0.50, *(q for q in QUANTILES if q != 0.50)]
    idx = [int(q * (len(points) - 1)) for q in picks] + [int(rng.integers(len(points)))]
    step = pd.Timedelta(panel.step_ns, unit="ns")
    last = max(int(tail / step) if tail else len(points) // 50, 1)
    idx.append(max(len(points) - 1 - last // 2, 0))
    cuts = []
    for i in idx:
        i = _future_close(panel, i0, points, i, offset)
        cuts.append(pd.Timestamp(int(points[i] + offset), tz="UTC").tz_convert(panel.tz))
    return cuts


def _future_close(
    panel: Panel, i0: int, points: np.ndarray, i: int, offset_ns: int, span: int = 100
) -> int:
    """Index near ``i`` that maximises the streams whose next event is after the cut and nearer.

    ``points`` are the grid points from frame row ``i0``. Ties are broken by
    the number of frame columns that change between a point and the next.
    """
    g = points[i : i + span]
    score = np.zeros(len(g), dtype=np.int64)
    for key in panel.streams:
        at = panel.events(*key).at
        nxt = np.searchsorted(at, g, side="right")
        has_both = (nxt > 0) & (nxt < len(at))
        ahead = np.where(has_both, at[np.minimum(nxt, len(at) - 1)] - g, 0)
        behind = np.where(has_both, g - at[np.maximum(nxt - 1, 0)], 0)
        score += has_both & (ahead > offset_ns) & (ahead < behind)
    rows = panel.frame.slice(i0 + i, len(g) + 1).drop("ts").to_numpy().astype(np.float64)
    changed = np.zeros(len(g), dtype=np.int64)
    if len(rows) > 1:
        a, b = rows[:-1], rows[1:]
        moved = (a != b) & ~(np.isnan(a) & np.isnan(b))
        changed[: len(moved)] = moved.sum(axis=1)
    return i + int(np.argmax(score * (rows.shape[1] + 1) + changed))


def precheck(
    fn: Callable[[Panel], Output],
    panel: Panel,
    window: Window,
    seed: int = 0,
    tail: pd.Timedelta | None = None,
) -> tuple[Verdict | None, np.ndarray]:
    """The fast pre-check: one full run and the 0.50 cut.

    A blatant leak fails here after two evaluations; a third, a second full
    run, then tells look-ahead from non-determinism. Returns the failing
    verdict (or None) and the full run, for :func:`check` to reuse.
    """
    full = run(fn, panel)
    scale = _scale(full)
    t = cut_times(panel, window, seed, tail)[0]
    short = panel.cut(t)
    bad = compare(_head(full, len(short)), run(fn, short), short.grid_ns, scale, t, panel.tz)
    if bad is None:
        return None, full
    again = run(fn, panel)
    noise = compare(full, again, panel.grid_ns, scale, cut=None, tz=panel.tz)
    if noise is not None:
        return Verdict("rejected_nondeterministic", (), noise), full
    return Verdict("rejected_lookahead", (t,), bad), full


def check(
    fn: Callable[[Panel], Output],
    panel: Panel,
    window: Window,
    seed: int = 0,
    full: np.ndarray | None = None,
    skip: int = 0,
    tail: pd.Timedelta | None = None,
) -> Verdict:
    """Run the determinism check, then the truncation test.

    Parameters
    ----------
    fn
        The feature function.
    panel
        The full panel the feature is scored on.
    window
        The scoring window; cuts are placed inside it.
    seed
        Seed for the random cut.
    full, skip
        A full run already made and the number of leading cuts already
        passed, as after :func:`precheck`.
    tail
        The length of the window's last block, inside which the last cut sits
        (:func:`cut_times`).

    The judge passes a ``fn`` that runs each call in a fresh process
    (:class:`blinded.isolate.Isolated`), so nothing one evaluation leaves in
    module state reaches another.

    Returns
    -------
    Verdict
        ``accepted``, ``rejected_nondeterministic`` or ``rejected_lookahead``,
        with the first violation found.
    """
    full = run(fn, panel) if full is None else full
    scale = _scale(full)
    again = run(fn, panel)
    bad = compare(full, again, panel.grid_ns, scale, cut=None, tz=panel.tz)
    if bad is not None:
        return Verdict("rejected_nondeterministic", (), bad)

    cuts = cut_times(panel, window, seed, tail)
    for t in cuts[skip:]:
        short = panel.cut(t)
        part = run(fn, short)
        bad = compare(_head(full, len(short)), part, short.grid_ns, scale, cut=t, tz=panel.tz)
        if bad is not None:
            return Verdict("rejected_lookahead", tuple(cuts), bad)
    return Verdict("accepted", tuple(cuts), full=full)


def compare(
    full: np.ndarray | Legs,
    part: np.ndarray | Legs,
    grid_ns: np.ndarray,
    scale: float | dict[str, float],
    cut: pd.Timestamp | None,
    tz: str = "UTC",
) -> Violation | None:
    """Compare two runs point by point. NaN equals NaN; NaN against a number fails.

    ``grid_ns`` holds the grid times in integer nanoseconds, for reporting
    where the first mismatch is. Positions in several legs are compared leg by
    leg, each with its own ``scale``, and the first leg that disagrees is named;
    quotes likewise, key by key.
    """
    if isinstance(full, Legs):
        for leg, values in full.items():
            bad = compare(values, part[leg], grid_ns, scale[leg], cut, tz)
            if bad is not None:
                name = "key" if isinstance(full, Quotes) else "leg"
                return dataclasses.replace(bad, **{name: leg})
        return None
    equal = np.isclose(full, part, rtol=RTOL, atol=ATOL * scale, equal_nan=True)
    if equal.all():
        return None
    bad = ~equal
    both = bad & np.isfinite(full) & np.isfinite(part)
    diff = np.abs(full[both] - part[both]).max() if both.any() else np.nan
    first = pd.Timestamp(int(grid_ns[int(np.argmax(bad))]), tz="UTC").tz_convert(tz)
    return Violation(
        cut=cut,
        first=first,
        n_bad=int(bad.sum()),
        n_compared=len(full),
        max_diff=float(diff),
        nan_mismatch=int((np.isnan(full) != np.isnan(part)).sum()),
    )


def _head(values: np.ndarray | Legs, n: int) -> np.ndarray | Legs:
    if isinstance(values, Legs):
        return type(values)({leg: v[:n] for leg, v in values.items()})
    return values[:n]


def _scale(values: np.ndarray | Legs) -> float | dict[str, float]:
    if isinstance(values, Legs):
        return {leg: _scale(v) for leg, v in values.items()}
    finite = np.abs(values[np.isfinite(values)])
    scale = float(np.median(finite)) if finite.size else 0.0
    return scale if scale > 0 else 1.0
