r"""Features across two instruments, on mids read as-of each grid point.

Each function takes symbols and returns an expression. Examples use
:func:`blinded.features.example_panel`.

The lead-lag functions compare a leader's trailing log return,
:math:`r_\ell(t, k) = \ln m_\ell(g_t) - \ln m_\ell(g_{t-k})`, with the
follower's over the same rows: the gap is the move the follower has not yet
made.

The pairs functions work on log mids, :math:`y = \ln m_a` and :math:`x =
\ln m_b`, so the hedge ratio is a ratio of notionals and the spread of a spot
and its perpetual is their log basis. The ratio, the mean and the scale are
fitted by OLS of :math:`y` on :math:`x` over the trailing ``window`` at the row
before each clock window of ``every``, and held through that window, so the
first clock window has none and no row's fit uses its own window.
"""

from __future__ import annotations

import math

import polars as pl

from blinded.durations import Span, polars_span
from blinded.features import series
from blinded.features._expr import TS, Window, back, nonzero, once, roll
from blinded.features.book import mid, spread

__all__ = [
    "basis",
    "basis_z",
    "half_life",
    "hedge_ratio",
    "lagged_corr",
    "lead_gap",
    "lead_gap_z",
    "lead_ret",
    "pair_spread",
    "rel_spread",
    "ret_beta",
    "ret_corr",
    "spread_z",
]


def basis(a: str, b: str, log: bool = False) -> pl.Expr:
    r"""Basis of ``a`` over ``b``, :math:`m_a / m_b - 1`, or :math:`\ln(m_a / m_b)`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(basis("PERP", "SPOT").round(4)).to_list()  # 11 / 10.5 - 1, ...
    [0.0476, 0.0, 0.0652, -0.02]
    """
    ratio = mid(a) / mid(b)
    return ratio.log() if log else ratio - 1


def basis_z(
    a: str,
    b: str,
    window: Window | None = None,
    halflife: float | Window | None = None,
) -> pl.Expr:
    r"""Basis z-score, trailing :math:`(B_t - \mu_W)/\sigma_W` or exponentially weighted.

    Give exactly one of ``window`` or ``halflife``. See
    :func:`blinded.features.series.zscore` and
    :func:`blinded.features.series.ewm_zscore`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(basis_z("PERP", "SPOT", window=3).round(2)).to_list()
    [None, None, 0.82, -0.79]
    """
    if (window is None) == (halflife is None):
        raise ValueError("give exactly one of window or halflife")
    b_ = basis(a, b)
    return series.zscore(b_, window) if window is not None else series.ewm_zscore(b_, halflife)


def lead_ret(leader: str, horizon: Window = 1, log: bool = True) -> pl.Expr:
    r"""The leader's own backward mid return, :math:`\ln(m_t / m_{t-h})`.

    Used as a lead-lag feature for another instrument: it only looks back.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(lead_ret("PERP").round(4)).to_list()  # ln(12.25 / 11) at the third
    [None, 0.0, 0.1076, 0.0]
    """
    return series.ret(mid(leader), horizon, log=log)


def lead_gap(leader: str, follower: str, horizon: Window = 1) -> pl.Expr:
    r"""The leader's trailing log return less the follower's, :math:`r_\ell(t, k) - r_f(t, k)`.

    Positive when the leader has risen further than the follower over the
    last ``horizon``: the move the follower has not made yet, if it is going
    to.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> # row 2: ln(12.25 / 11) - ln(11.5 / 11) = 0.10763 - 0.04445
    >>> example_panel().eval(lead_gap("PERP", "SPOT").round(4)).to_list()
    [None, -0.0465, 0.0632, -0.0834]
    """
    return lead_ret(leader, horizon) - lead_ret(follower, horizon)


def lead_gap_z(leader: str, follower: str, horizon: Window = 1, window: Window = "5m") -> pl.Expr:
    r""":func:`lead_gap` divided by its trailing standard deviation over ``window``.

    There is no mean to take off: two instruments that move together have a
    gap near zero on average, so the gap is scaled and kept centred on zero.
    Null where the gap was constant over the window.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> # gaps -0.04652, 0.06318, -0.08338; sd of the first two is 0.10970 / sqrt 2 = 0.07757,
    >>> # so row 2 is 0.06318 / 0.07757
    >>> example_panel().eval(lead_gap_z("PERP", "SPOT", 1, 2).round(4)).to_list()
    [None, None, 0.8145, -0.8046]
    """
    gap = lead_gap(leader, follower, horizon)
    return gap / nonzero(series.roll_std(gap, window))


def lagged_corr(leader: str, follower: str, lag: Window = 1, window: Window = "5m") -> pl.Expr:
    r"""Trailing correlation of the follower's one-step return with the leader's ``lag`` earlier.

    :math:`\rho_W\big(r_f(t, 1),\ r_\ell(t - k, 1)\big)` over the last
    ``window``: how strongly the leader's move ``lag`` ago has been followed
    lately. ``lag`` is a number of rows or a time span, read backwards.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> # SPOT returns 0.0465, 0.0445, 0.0834 against PERP's a row earlier, -, 0, 0.1076:
    >>> # the last two pairs both rise, so the correlation over them is 1
    >>> example_panel().eval(lagged_corr("PERP", "SPOT", 1, 2).round(4)).to_list()
    [None, None, None, 1.0]
    """
    return series.roll_corr(_step_ret(follower), back(_step_ret(leader), lag), window)


def rel_spread(a: str, b: str) -> pl.Expr:
    r"""Spread of ``a`` relative to ``b``, both in bps of their mids, :math:`s_a / s_b`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(rel_spread("SPOT", "PERP").round(4)).to_list()  # (1/10.5)/(1/11)
    [1.0476, 2.0, 2.1304, 1.96]
    """
    return spread(a, bps=True) / spread(b, bps=True)


def ret_corr(a: str, b: str, window: Window) -> pl.Expr:
    r"""Trailing correlation of one-step log mid returns, :math:`\rho_W(r_a, r_b)`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(ret_corr("SPOT", "PERP", 2).round(4)).to_list()
    [None, None, -1.0, -1.0]
    """
    return series.roll_corr(_step_ret(a), _step_ret(b), window)


def ret_beta(a: str, b: str, window: Window) -> pl.Expr:
    r"""Trailing beta of ``a``'s one-step log mid returns on ``b``'s, :math:`\beta_W(r_a, r_b)`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(ret_beta("SPOT", "PERP", 2).round(4)).to_list()
    [None, None, -0.0192, -0.3617]
    """
    return series.roll_beta(_step_ret(a), _step_ret(b), window)


def hedge_ratio(a: str, b: str, window: Window = "2h", every: Span = "30m") -> pl.Expr:
    r"""Walk-forward hedge ratio of ``a`` on ``b``, :math:`\tilde\beta_t = \beta_{w(t) - 1}`.

    :math:`\beta` is the OLS slope of :math:`y` on :math:`x` over the trailing
    ``window``, :math:`\mathrm{cov}_W(y, x) / \mathrm{var}_W(x)`, read at the
    last row before the clock window that holds :math:`t`. Null through the
    first clock window. A window given as a number of rows has no fit until it
    holds that many; a time window is fitted on whatever rows it holds, from
    the first, so the earliest fits can rest on a handful of rows. Null where
    :math:`x` did not move over the window.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> # windows of 2 s hold rows [0], [1, 2], [3]; the fit for row 3 is over rows 0-2:
    >>> # y = ln(11, 11, 12.25), x = ln(10.5, 11, 11.5), cov / var = 1.1740
    >>> example_panel().eval(hedge_ratio("PERP", "SPOT", 3, "2s").round(4)).to_list()
    [None, None, None, 1.174]
    """
    return once(lambda y, x: _moments(y, x, window, every)[0], *_logs(a, b))


def pair_spread(a: str, b: str, window: Window = "2h", every: Span = "30m") -> pl.Expr:
    r"""Spread of ``a`` over ``b``, :math:`s_t = y_t - \tilde\beta_t x_t`, on the current log mids.

    Not :func:`blinded.features.book.spread`, the bid-ask spread of one book.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> # row 3: ln 12.25 - 1.17395 * ln 12.5 = -0.4596
    >>> example_panel().eval(pair_spread("PERP", "SPOT", 3, "2s").round(4)).to_list()
    [None, None, None, -0.4596]
    """
    return mid(a).log() - hedge_ratio(a, b, window, every) * mid(b).log()


def spread_z(a: str, b: str, window: Window = "2h", every: Span = "30m") -> pl.Expr:
    r"""The spread standardised walk-forward, :math:`z_t = (s_t - \tilde\mu_t) / \tilde\sigma_t`.

    :math:`\tilde\mu = \bar y - \tilde\beta \bar x` and :math:`\tilde\sigma^2 =
    \mathrm{var}_W(y) - 2\tilde\beta\,\mathrm{cov}_W(y, x) + \tilde\beta^2
    \mathrm{var}_W(x)`, the residual variance of the fit (divisor :math:`W`),
    come from the same window as :math:`\tilde\beta` and are held with it, so
    a time window is fitted on whatever rows it holds, as in
    :func:`hedge_ratio`. Null through the first clock window, and where
    :math:`\tilde\sigma` is zero up to rounding, as for an exact fit.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> # rows 0-2: mean y - 1.17395 mean x = -0.38044, residual sd 0.025943;
    >>> # row 3: (-0.45956 + 0.38044) / 0.025943 = -3.0500
    >>> example_panel().eval(spread_z("PERP", "SPOT", 3, "2s").round(4)).to_list()
    [None, None, None, -3.05]
    """

    def z(y, x):
        beta, mu, var = _moments(y, x, window, every)
        return (y - beta * x - mu) / var.sqrt()

    return once(z, *_logs(a, b))


def half_life(x: pl.Expr, window: Window) -> pl.Expr:
    r"""Trailing half-life of ``x`` in grid steps, :math:`-\ln 2 / \ln \hat\varphi`.

    :math:`\hat\varphi` is the OLS slope of :math:`x_t` on :math:`x_{t-1}`, with
    an intercept, over the last ``window`` pairs. Infinite where
    :math:`\hat\varphi \ge 1`, which is no reversion; null where
    :math:`\hat\varphi < 0`, an overshoot every step.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"s": [8.0, 4.0, 2.0, 1.0, 0.5]})
    >>> # each value is half the last: slope 0.5, -ln 2 / ln 0.5 = 1 step
    >>> df.select(half_life(pl.col("s"), 4).round(6)).to_series().to_list()
    [None, None, None, None, 1.0]
    """
    # The means of the lagged values are the means of the current ones a row earlier. Each
    # copy of x in the tree is evaluated again, so x appears as few times as possible.
    mean = roll(x, "mean", window)
    before = mean.shift(1)
    phi = (roll(x * x.shift(1), "mean", window) - mean * before) / (
        roll(x.pow(2), "mean", window).shift(1) - before.pow(2)
    )
    # abs turns the -0 that phi = 1 gives into +0, so that its half-life is +inf
    return (math.log(2) / (-phi.log()).clip(lower_bound=0.0).abs()).fill_nan(None)


def _fit(a: str, b: str, window: Window, every: Span):
    """The held slope, intercept and residual variance, and the ``y`` and ``x`` they fit."""
    y, x = _logs(a, b)
    return (*_moments(y, x, window, every), y, x)


def _logs(a: str, b: str) -> tuple[pl.Expr, pl.Expr]:
    """The log mids of ``a`` and ``b``, measured from their first values.

    That leaves the slope and the residuals as they are and keeps the moments
    from cancelling.
    """
    return _from_first(mid(a).log()), _from_first(mid(b).log())


def _moments(y: pl.Expr, x: pl.Expr, window: Window, every: Span):
    """The held slope, intercept and residual variance of ``y`` on ``x``.

    A residual variance under 1e-12 of the variance of ``y`` is rounding, and null.
    """
    both = y.is_not_null() & x.is_not_null()
    ym, xm = pl.when(both).then(y), pl.when(both).then(x)
    mx, my, mxx, myy, mxy = (roll(v, "mean", window) for v in (xm, ym, xm * xm, ym * ym, xm * ym))
    var_x, var_y, cov = mxx - mx * mx, myy - my * my, mxy - mx * my
    beta = cov / nonzero(var_x)
    resid = var_y - beta * cov
    resid = pl.when(resid > 1e-12 * var_y).then(resid)
    return _held(beta, every), _held(my - beta * mx, every), _held(resid, every)


def _held(x: pl.Expr, every: Span) -> pl.Expr:
    """``x`` at the last row before the current clock window, for every row of the window."""
    key = pl.col(TS).dt.truncate(polars_span(every))
    start = pl.when(key != key.shift(1)).then(pl.int_range(pl.len()))
    return x.shift(1).gather(start.forward_fill().fill_null(0))


def _from_first(x: pl.Expr) -> pl.Expr:
    return x - x.drop_nulls().first()


def _step_ret(sym: str) -> pl.Expr:
    return mid(sym).log().diff()
