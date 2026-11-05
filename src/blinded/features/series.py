r"""Transforms of one series, as polars expressions; the value at ``t`` uses inputs up to ``t``.

Each function takes an expression and returns one, so transforms chain and
nothing is computed until the expression is evaluated on a frame, by the
judge or by :meth:`blinded.Panel.eval`. Windows are a number of rows
(``3000``) or a time span (``"5m"``, ``"5min"``, ``"100ms"``) over the
frame's ``ts`` column. Integer windows give null until the window is full.
Time windows cover :math:`(t - w, t]` and emit from the first row, as in
pandas. Exponential halflives are likewise a number of rows or a span; a span
decays by the time elapsed between rows, so it also works on irregular times.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from blinded.durations import to_ns
from blinded.features._expr import (
    TS,
    Window,
    back,
    count,
    is_rows,
    nonzero,
    roll,
    rows,
    span,
)

__all__ = [
    "bollinger_b",
    "diff",
    "ewm_cross",
    "ewm_mean",
    "ewm_std",
    "ewm_zscore",
    "expanding_zscore",
    "lag",
    "meanrev",
    "momentum",
    "realized_vol",
    "ret",
    "roll_beta",
    "roll_corr",
    "roll_kurt",
    "roll_max",
    "roll_mean",
    "roll_median",
    "roll_min",
    "roll_quantile",
    "roll_rank",
    "roll_skew",
    "roll_std",
    "roll_sum",
    "roll_winsorize",
    "rsi",
    "time_of_day",
    "zscore",
]


def lag(x: pl.Expr, n: Window = 1) -> pl.Expr:
    r"""Value ``n`` rows (or a time span) back, :math:`x_{t-n}`.

    Time lags take the last value at or before :math:`t - n`. Negative lags
    raise, since they would read the future.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    >>> df.select(lag(pl.col("x"), 1)).to_series().to_list()
    [None, 1.0, 2.0]
    """
    return back(x, n)


def diff(x: pl.Expr, n: Window = 1) -> pl.Expr:
    r"""Change over ``n`` rows or a time span, :math:`x_t - x_{t-n}`.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 4.0, 9.0]})
    >>> df.select(diff(pl.col("x"))).to_series().to_list()
    [None, 3.0, 5.0]
    """
    return x - back(x, n)


def ret(x: pl.Expr, horizon: Window = 1, log: bool = True) -> pl.Expr:
    r"""Backward return over ``horizon``: :math:`\ln(x_t / x_{t-h})`, or :math:`x_t/x_{t-h} - 1`.

    Always looks back; there is no forward version in this library.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [100.0, 110.0, 99.0]})
    >>> df.select(ret(pl.col("x"), 1, log=False).round(4)).to_series().to_list()
    [None, 0.1, -0.1]
    """
    ratio = x / back(x, horizon)
    return ratio.log() if log else ratio - 1


def roll_mean(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing mean, :math:`\frac{1}{|W_t|}\sum_{s \in W_t} x_s`.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    >>> df.select(roll_mean(pl.col("x"), 2)).to_series().to_list()
    [None, 1.5, 2.5, 3.5]
    """
    return roll(x, "mean", window, min_periods)


def roll_std(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing sample standard deviation (divisor :math:`n - 1`).

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 3.0, 5.0, 5.0]})
    >>> df.select(roll_std(pl.col("x"), 2).round(4)).to_series().to_list()
    [None, 1.4142, 1.4142, 0.0]
    """
    return roll(x, "std", window, min_periods)


def roll_sum(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing sum, :math:`\sum_{s \in W_t} x_s`.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    >>> df.select(roll_sum(pl.col("x"), 3)).to_series().to_list()
    [None, None, 6.0, 9.0]
    """
    return roll(x, "sum", window, min_periods)


def roll_min(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing minimum, :math:`\min_{s \in W_t} x_s`.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [3.0, 1.0, 2.0, 5.0]})
    >>> df.select(roll_min(pl.col("x"), 2)).to_series().to_list()
    [None, 1.0, 1.0, 2.0]
    """
    return roll(x, "min", window, min_periods)


def roll_max(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing maximum, :math:`\max_{s \in W_t} x_s`.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [3.0, 1.0, 2.0, 5.0]})
    >>> df.select(roll_max(pl.col("x"), 2)).to_series().to_list()
    [None, 3.0, 2.0, 5.0]
    """
    return roll(x, "max", window, min_periods)


def roll_median(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing median, :math:`Q_{0.5}` over :math:`W_t`.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [3.0, 1.0, 2.0, 5.0]})
    >>> df.select(roll_median(pl.col("x"), 3)).to_series().to_list()
    [None, None, 2.0, 2.0]
    """
    return roll(x, "median", window, min_periods)


def roll_quantile(x: pl.Expr, window: Window, q: float, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing quantile :math:`Q_q` over :math:`W_t`, linear interpolation, window only.

    Polars defaults to the nearest rank; this asks for linear, as pandas does.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> df.select(roll_quantile(pl.col("x"), 5, 0.25)).to_series().to_list()
    [None, None, None, None, 2.0]
    """
    return roll(x, "quantile", window, min_periods, quantile=q, interpolation="linear")


def roll_rank(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Percentile of the current value within its trailing window, in :math:`(0, 1]`.

    :math:`\mathrm{rank}_{W_t}(x_t) / |W_t|`; ties share their average rank.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [3.0, 1.0, 2.0, 5.0]})
    >>> df.select(roll_rank(pl.col("x"), 3).round(4)).to_series().to_list()
    [None, None, 0.6667, 1.0]
    """
    rank = roll(x, "rank", window, min_periods, method="average")
    return rank / count(x, window, min_periods)


def _moment(x: pl.Expr, window: Window, min_periods: int | None, how: str) -> pl.Expr:
    if is_rows(window):
        if how == "skew":
            return x.rolling_skew(rows(window), bias=False, min_samples=min_periods)
        return x.rolling_kurtosis(rows(window), bias=False, min_samples=min_periods)
    # no rolling_skew_by or rolling_kurtosis_by in polars: the rolling context runs the
    # aggregation once per (t - w, t] window, natively
    agg = x.skew(bias=False) if how == "skew" else x.kurtosis(bias=False)
    out = agg.rolling(index_column=TS, period=span(window))
    return out if min_periods is None else pl.when(count(x, window) >= min_periods).then(out)


def roll_skew(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing sample skewness, adjusted Fisher-Pearson :math:`G_1 = g_1 \sqrt{n(n-1)}/(n-2)`.

    Needs 3 points.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 10.0]})
    >>> df.select(roll_skew(pl.col("x"), 4).round(4)).to_series().to_list()
    [None, None, None, 1.7636]
    """
    return _moment(x, window, min_periods, "skew")


def roll_kurt(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing sample excess kurtosis, bias-corrected :math:`G_2`.

    :math:`G_2 = \frac{(n-1)\,((n+1)\, g_2 + 6)}{(n-2)(n-3)}`; needs 4 points.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    >>> df.select(roll_kurt(pl.col("x"), 4).round(4)).to_series().to_list()
    [None, None, None, -1.2]
    """
    return _moment(x, window, min_periods, "kurtosis")


def _cov(x: pl.Expr, y: pl.Expr, window: Window) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Windowed covariance over rows where both are present, plus the masked series."""
    both = x.is_not_null() & y.is_not_null()
    xm, ym = pl.when(both).then(x), pl.when(both).then(y)
    n = roll(both.cast(pl.Float64), "sum", window)
    mx, my, mxy = (roll(v, "mean", window) for v in (xm, ym, xm * ym))
    return (mxy - mx * my) * (n / (n - 1)), xm, ym


def roll_corr(x: pl.Expr, y: pl.Expr, window: Window) -> pl.Expr:
    r"""Trailing Pearson correlation, :math:`\mathrm{cov}_W(x, y) / (\sigma_W(x)\, \sigma_W(y))`.

    Only rows where both are present count.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 7.0]})
    >>> df.select(roll_corr(pl.col("x"), pl.col("y"), 2).round(4)).to_series().to_list()
    [None, 1.0, 1.0]
    """
    cov, xm, ym = _cov(x, y, window)
    return cov / nonzero((roll(xm, "var", window) * roll(ym, "var", window)).sqrt())


def roll_beta(y: pl.Expr, x: pl.Expr, window: Window) -> pl.Expr:
    r"""Trailing OLS slope of ``y`` on ``x``, :math:`\mathrm{cov}_W(y, x) / \mathrm{var}_W(x)`.

    Null where ``x`` is constant over the window.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    >>> x = pl.col("x")
    >>> df.select(roll_beta(2 * x + 1, x, 3).round(4)).to_series().to_list()
    [None, None, 2.0, 2.0]
    """
    cov, _, _ = _cov(y, x, window)
    return cov / nonzero(roll(x, "var", window))


def zscore(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Trailing z-score, :math:`(x_t - \mu_W) / \sigma_W`, the window including :math:`t`.

    The mean and deviation come from the trailing window, never the full
    sample. Null where the window is constant.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 3.0, 3.0]})
    >>> df.select(zscore(pl.col("x"), 3).round(4)).to_series().to_list()
    [None, None, 1.0, 0.5774, None]
    """
    mean = roll(x, "mean", window, min_periods)
    return (x - mean) / nonzero(roll(x, "std", window, min_periods))


def meanrev(x: pl.Expr, window: Window, min_periods: int | None = None) -> pl.Expr:
    r"""Mean-reversion signal, :math:`-(x_t - \mu_W) / \sigma_W`.

    The negated z-score, so a positive value says ``x`` sits below its trailing
    mean and the forecast is up.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    >>> df.select(meanrev(pl.col("x"), 3).round(4)).to_series().to_list()
    [None, None, -1.0]
    """
    return -zscore(x, window, min_periods)


def expanding_zscore(x: pl.Expr, min_periods: int = 2) -> pl.Expr:
    r"""Z-score against everything up to :math:`t`: :math:`(x_t - \bar x_{\le t}) / s_{\le t}`.

    It depends on where the sample starts, which is fixed, and not on where it
    ends. The running sums are taken of :math:`x - x_0`, the first value,
    which keeps :math:`S_2 - S_1^2/n` from cancelling on prices. Null while
    the history is constant.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 3.0, 2.0]})
    >>> df.select(expanding_zscore(pl.col("x")).round(4)).to_series().to_list()
    [None, 0.7071, 0.0]
    """
    c = x - x.drop_nulls().first()
    n = c.cum_count().cast(pl.Float64)
    s1, s2 = c.cum_sum(), (c * c).cum_sum()
    var = (s2 - s1 * s1 / n) / (n - 1)
    return pl.when(n >= min_periods).then((c - s1 / n) / nonzero(var).sqrt())


def _alpha(halflife: float) -> float:
    if halflife <= 0:
        raise ValueError(f"a halflife must be positive, got {halflife}")
    return float(1 - np.exp(-np.log(2) / halflife))


def _is_number(value) -> bool:
    number = isinstance(value, int | float | np.integer | np.floating)
    return number and not isinstance(value, bool)


def ewm_mean(x: pl.Expr, halflife: float | Window) -> pl.Expr:
    r"""Exponentially weighted mean, :math:`m_t = (1-\alpha_t) m_{t-1} + \alpha_t x_t`.

    For a halflife of :math:`h` rows, :math:`\alpha = 1 - 2^{-1/h}`; for a
    span :math:`h`, :math:`\alpha_t = 1 - 2^{-\Delta t / h}` with
    :math:`\Delta t` the time since the previous row, which on a regular grid
    is the same thing. Seeded at the first observation, :math:`m_0 = x_0`,
    never at a sample statistic.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [0.0, 4.0, 4.0]})
    >>> df.select(ewm_mean(pl.col("x"), 1)).to_series().to_list()
    [0.0, 2.0, 3.0]
    """
    if _is_number(halflife):
        return x.ewm_mean(alpha=_alpha(halflife), adjust=False)
    return x.ewm_mean_by(TS, half_life=span(halflife))


def ewm_std(x: pl.Expr, halflife: float | Window) -> pl.Expr:
    r"""Exponentially weighted standard deviation about the EW mean.

    :math:`v_t = (1-\alpha_t)\,(v_{t-1} + \alpha_t (x_t - m_{t-1})^2)`,
    :math:`v_0 = 0`; returns :math:`\sqrt{v_t}`. Seeded at the first
    observation.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [0.0, 4.0]})
    >>> df.select(ewm_std(pl.col("x"), 1)).to_series().to_list()
    [0.0, 2.0]
    """
    if _is_number(halflife):
        return x.ewm_std(alpha=_alpha(halflife), adjust=False, bias=True)
    # polars has no ewm_std_by. The recursion is v_t = (1 - a_t) v_{t-1} + a_t z_t with
    # z_t = (1 - a_t)(x_t - m_{t-1})^2, so it is an EW mean of z with the same weights.
    present = x.is_not_null()
    ts = pl.col(TS)
    elapsed = (ts - pl.when(present).then(ts).forward_fill().shift(1)).dt.total_nanoseconds()
    keep = (elapsed * (-np.log(2) / to_ns(halflife))).exp()
    previous = ewm_mean(x, halflife).forward_fill().shift(1)
    z = pl.when(present).then((keep * (x - previous) ** 2).fill_null(0.0))
    return z.ewm_mean_by(TS, half_life=span(halflife)).sqrt()


def ewm_zscore(x: pl.Expr, halflife: float | Window) -> pl.Expr:
    r"""EW z-score, :math:`(x_t - m_t) / \sqrt{v_t}` with :func:`ewm_mean` and :func:`ewm_std`.

    Null where the EW deviation is zero, including the first point.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [0.0, 4.0]})
    >>> df.select(ewm_zscore(pl.col("x"), 1)).to_series().to_list()
    [None, 1.0]
    """
    return (x - ewm_mean(x, halflife)) / nonzero(ewm_std(x, halflife))


def ewm_cross(x: pl.Expr, fast: float | Window, slow: float | Window) -> pl.Expr:
    r"""Difference of a fast and a slow EW mean (the MACD line), :math:`m^{fast}_t - m^{slow}_t`.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [0.0, 4.0]})
    >>> df.select(ewm_cross(pl.col("x"), 1, 2).round(4)).to_series().to_list()
    [0.0, 0.8284]
    """
    return ewm_mean(x, fast) - ewm_mean(x, slow)


def realized_vol(x: pl.Expr, window: Window) -> pl.Expr:
    r"""Realised volatility, :math:`\sqrt{\sum_{s \in W_t} (\ln x_s - \ln x_{s-1})^2}`.

    Not annualised: it is the volatility over the window itself.

    Examples
    --------
    >>> import numpy as np, polars as pl
    >>> df = pl.DataFrame({"x": np.exp([0.0, 0.3, -0.1])})
    >>> df.select(realized_vol(pl.col("x"), 2).round(4)).to_series().to_list()
    [None, None, 0.5]
    """
    r = x.log().diff()
    # a rolling sum keeps a running total, so a flat window after a busy one can come out
    # a hair below zero; a sum of squares cannot be negative
    return roll_sum(r * r, window).clip(lower_bound=0.0).sqrt()


def momentum(x: pl.Expr, window: Window) -> pl.Expr:
    r"""Volatility-scaled momentum, :math:`\ln(x_t / x_{t-w}) / RV_w(t)`.

    The log return over the window divided by :func:`realized_vol` over the
    same window, so a random walk gives a value of order one.

    Examples
    --------
    >>> import numpy as np, polars as pl
    >>> df = pl.DataFrame({"x": np.exp([0.0, 0.3, 0.7])})
    >>> df.select(momentum(pl.col("x"), 2).round(4)).to_series().to_list()
    [None, None, 1.4]
    """
    return ret(x, window, log=True) / nonzero(realized_vol(x, window))


def rsi(x: pl.Expr, period: int = 14) -> pl.Expr:
    r"""Wilder's relative strength index, :math:`100\, G_t / (G_t + L_t)`.

    :math:`G` and :math:`L` are Wilder averages of gains and losses: the
    plain mean of the first ``period`` changes, then
    :math:`G_t = G_{t-1} + (g_t - G_{t-1}) / n`. The seed uses only the first
    ``period`` changes. The period is a number of rows. Null while there have
    been no moves at all.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 1.0, 3.0]})
    >>> df.select(rsi(pl.col("x"), 2).round(4)).to_series().to_list()
    [None, None, 50.0, 83.3333]
    """
    if not is_rows(period):
        raise TypeError(f"rsi takes its period as a number of rows, got {period!r}")
    n = rows(period)
    change = x.diff()
    gains = _wilder(change.clip(lower_bound=0.0), n)
    losses = _wilder((-change).clip(lower_bound=0.0), n)
    return 100 * gains / nonzero(gains + losses)


def _wilder(values: pl.Expr, n: int) -> pl.Expr:
    row = pl.int_range(pl.len())
    seed_row = row.filter(values.is_not_null()).first() + n - 1
    seeded = (
        pl.when(row == seed_row)
        .then(values.rolling_mean(n, min_samples=1))
        .when(row > seed_row)
        .then(values)
    )
    return seeded.ewm_mean(alpha=1 / n, adjust=False)


def bollinger_b(x: pl.Expr, window: Window, k: float = 2.0) -> pl.Expr:
    r"""Position inside trailing Bollinger bands, :math:`(x_t - \mu_W + k\sigma_W) / (2k\sigma_W)`.

    0 at the lower band, 1 at the upper, 0.5 on the mean. Null where the
    window is constant.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    >>> df.select(bollinger_b(pl.col("x"), 3, k=1)).to_series().to_list()
    [None, None, 1.0]
    """
    sd = nonzero(roll(x, "std", window))
    return (x - roll(x, "mean", window) + k * sd) / (2 * k * sd)


def roll_winsorize(x: pl.Expr, window: Window, q: float = 0.01) -> pl.Expr:
    r"""Clip ``x`` to its trailing :math:`[Q_q, Q_{1-q}]` quantiles.

    The bounds come from the trailing window, not the full sample.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    >>> df.select(roll_winsorize(pl.col("x"), 5, 0.25)).to_series().to_list()
    [None, None, None, None, 4.0]
    """
    lower = roll_quantile(x, window, q)
    upper = roll_quantile(x, window, 1 - q)
    return pl.when(lower.is_not_null()).then(x.clip(lower, upper))


def time_of_day(ts: pl.Expr | None = None) -> pl.Expr:
    r"""Hours since midnight in the column's timezone, a float in :math:`[0, 24)`.

    Reads the frame's ``ts`` column unless given another datetime expression.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> df = pl.DataFrame({"ts": [datetime(2024, 1, 1, 6, 30), datetime(2024, 1, 1, 18, 45)]})
    >>> df.select(time_of_day()).to_series().to_list()
    [6.5, 18.75]
    """
    t = pl.col(TS) if ts is None else ts
    h, m, s, ns = (getattr(t.dt, part)().cast(pl.Int64) for part in _CLOCK)
    return (((h * 60 + m) * 60 + s) * 1_000_000_000 + ns) / 3.6e12


_CLOCK = ("hour", "minute", "second", "nanosecond")
