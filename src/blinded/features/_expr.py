"""Building blocks shared by the feature library: windows, look-backs and safe division."""

from __future__ import annotations

import numpy as np
import polars as pl

from blinded.durations import Span, to_ns

TS = "ts"
Window = int | Span


def is_rows(window) -> bool:
    """True for a window given as a number of rows."""
    return isinstance(window, int | np.integer) and not isinstance(window, bool)


def rows(window) -> int:
    if window < 1:
        raise ValueError(f"window must be at least 1 point, got {window}")
    return int(window)


def span(window: Span) -> str:
    """A positive time span as a polars duration string."""
    ns = to_ns(window)
    if ns <= 0:
        raise ValueError(f"a time window must be positive, got {window!r}")
    return f"{ns}ns"


def roll(x: pl.Expr, how: str, window: Window, min_periods: int | None = None, **kw) -> pl.Expr:
    """``x.rolling_<how>(n)`` over ``n`` rows, or ``x.rolling_<how>_by("ts", w)`` over a span.

    Time windows are :math:`(t - w, t]`, closed on the right like pandas, and
    emit from the first row unless ``min_periods`` says otherwise.
    """
    if is_rows(window):
        by_rows = getattr(x, f"rolling_{how}")
        return by_rows(window_size=rows(window), min_samples=min_periods, **kw)
    by = getattr(x, f"rolling_{how}_by")
    return by(TS, window_size=span(window), min_samples=min_periods or 1, closed="right", **kw)


def count(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    """Number of non-null values of ``x`` in each window."""
    return roll(x.is_not_null().cast(pl.Float64), "sum", window, min_periods)


def back(x: pl.Expr, horizon: Window) -> pl.Expr:
    """Value of ``x`` ``horizon`` rows back, or at the last row with ``ts <= t - horizon``.

    Time look-backs assume ``ts`` strictly increases, as the panel grid does.
    """
    if is_rows(horizon):
        if horizon < 0:
            raise ValueError("a negative lag reads the future")
        return x.shift(int(horizon))
    ns = to_ns(horizon)
    if ns < 0:
        raise ValueError("a negative lag reads the future")
    if ns == 0:
        return x
    # j, the number of rows with ts <= t - h, is the row number plus one minus the rows in
    # (t - h, t]. A rolling count finds those in one pass, where search_sorted bisects for
    # every row and is about four times slower here. x shifted down one row holds x[j - 1]
    # at j, and the shift's null at j = 0. The clip keeps tied timestamps, which polars
    # windows count ahead of the current row, from wrapping to the end of the column.
    ts = pl.col(TS)
    in_window = ts.is_not_null().cast(pl.Int64).rolling_sum_by(ts, f"{ns}ns", closed="right")
    j = (pl.int_range(1, pl.len() + 1) - in_window).clip(lower_bound=0)
    return x.shift(1).gather(j)


def once(build, *inputs: pl.Expr) -> pl.Expr:
    """``build(*inputs)``, with each input computed once however often ``build`` uses it.

    Polars computes every copy of a subexpression in a tree again, so a
    function that uses its input ten times costs ten inputs, and nesting
    such functions multiplies the cost. Here the inputs become columns beside
    ``ts`` first, and ``build`` runs on those columns. The result is named
    after the first input, as an expression built on it directly would be.

    Examples
    --------
    >>> df = pl.DataFrame({"ts": [1, 2, 3], "x": [1.0, 2.0, 4.0]})
    >>> df.select(once(lambda v: v * v - v, pl.col("x") + 1))
    shape: (3, 1)
    ┌──────┐
    │ x    │
    │ ---  │
    │ f64  │
    ╞══════╡
    │ 2.0  │
    │ 6.0  │
    │ 20.0 │
    └──────┘
    """
    names = [f"_in{i}" for i in range(len(inputs))]

    def run(s: pl.Series) -> pl.Series:
        frame = s.struct.unnest()
        frame.columns = [*names, TS]
        return frame.select(build(*map(pl.col, names))).to_series().alias(s.name)

    # the struct, and so the result, takes the name of its first field
    first, *rest = (e.cast(pl.Float64) for e in inputs)
    fields = [first, *(e.alias(n) for n, e in zip(names[1:], rest, strict=True))]
    return pl.struct(*fields, pl.col(TS).alias("_ts")).map_batches(run, return_dtype=pl.Float64)


def nonzero(scale: pl.Expr) -> pl.Expr:
    """``scale`` where it is positive, else null, so a zero spread gives null rather than inf."""
    return pl.when(scale > 0).then(scale)


def ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """``num / den``, null where ``den`` is 0."""
    return num / pl.when(den != 0).then(den)


def window_sum(cumulative: pl.Expr, window: Span) -> pl.Expr:
    r"""Sum over :math:`(t - w, t]` from a cumulative column: :math:`C_t - C_{t-w}`.

    ``cumulative`` is one of the panel's ``cum_*`` columns, a running total over
    the events available so far, so the difference counts exactly the events
    with ``t - w < available_at <= t`` when :math:`w` is a whole number of
    grid steps. Before the first row at or before :math:`t - w` the total
    there is taken as 0.

    Examples
    --------
    >>> import polars as pl
    >>> from blinded.features import example_panel
    >>> example_panel().eval(window_sum(pl.col("SPOT.cum_qty"), "1s")).to_list()  # 2+1, 3, 0, 4
    [3.0, 3.0, 0.0, 4.0]
    """
    span(window)
    return cumulative - back(cumulative, window).fill_null(0)
