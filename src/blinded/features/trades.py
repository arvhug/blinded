r"""Trade-flow features over trailing windows :math:`(t - w, t]` of trade arrivals.

Windows here are time spans, since they count events, not grid points. A
window with no trades gives 0 for sums and counts and null for ratios. Each
sum is the difference of one of the panel's cumulative columns at :math:`t`
and at :math:`t - w` (:func:`~blinded.features.window_sum`), which counts
exactly the trades available in the window when :math:`w` is a whole number
of grid steps, as 1 s, 10 s or 5 min are on a 100 ms grid. Other windows
reach back to the last grid point at or before :math:`t - w`, so they run up
to one step long. Examples use :func:`blinded.features.example_panel`.
"""

from __future__ import annotations

import polars as pl

from blinded.durations import Span
from blinded.features import series
from blinded.features._expr import Window, ratio, window_sum
from blinded.features.book import mid

__all__ = [
    "avg_trade_size",
    "kyle_lambda",
    "signed_volume",
    "trade_count",
    "trade_imbalance",
    "vwap",
    "vwap_dev",
]


def _sum(sym: str, total: str, window: Span) -> pl.Expr:
    return window_sum(pl.col(f"{sym}.cum_{total}"), window)


def signed_volume(sym: str, window: Span) -> pl.Expr:
    r"""Buy volume minus sell volume, :math:`\sum_{i \in W_t} s_i q_i`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(signed_volume("SPOT", "1s")).to_list()  # 2 - 1, 3, none, -4
    [1.0, 3.0, 0.0, -4.0]
    """
    return _sum(sym, "signed_qty", window)


def trade_count(sym: str, window: Span) -> pl.Expr:
    r"""Number of trades in the window, :math:`N_{W_t}`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(trade_count("SPOT", "1s")).to_list()
    [2.0, 1.0, 0.0, 1.0]
    """
    return _sum(sym, "trades", window).cast(pl.Float64)


def trade_imbalance(sym: str, window: Span) -> pl.Expr:
    r"""Volume imbalance, :math:`(V_B - V_S) / (V_B + V_S)`; null with no trades.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(trade_imbalance("SPOT", "1s").round(4)).to_list()  # (2 - 1) / 3
    [0.3333, 1.0, None, -1.0]
    """
    buys, sells = _sum(sym, "buy_qty", window), _sum(sym, "sell_qty", window)
    return ratio(buys - sells, buys + sells)


def vwap(sym: str, window: Span) -> pl.Expr:
    r"""Volume-weighted average trade price, :math:`\sum p_i q_i / \sum q_i`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(vwap("SPOT", "1s").round(4)).to_list()  # (2*11 + 1*10) / 3
    [10.6667, 12.0, None, 12.0]
    """
    return ratio(_sum(sym, "notional", window), _sum(sym, "qty", window))


def vwap_dev(sym: str, window: Span) -> pl.Expr:
    r"""Mid against the trailing VWAP in bps, :math:`10^4\,(m_t / \mathrm{VWAP}_W - 1)`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(vwap_dev("SPOT", "1s").round(2)).to_list()  # (10.5/(32/3) - 1) 1e4
    [-156.25, -833.33, None, 416.67]
    """
    return (mid(sym) / vwap(sym, window) - 1) * 1e4


def avg_trade_size(sym: str, window: Span) -> pl.Expr:
    r"""Mean trade size, :math:`\sum q_i / N`; null with no trades.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(avg_trade_size("SPOT", "1s")).to_list()
    [1.5, 3.0, None, 4.0]
    """
    return ratio(_sum(sym, "qty", window), _sum(sym, "trades", window))


def kyle_lambda(sym: str, window: Window) -> pl.Expr:
    r"""Price impact per unit of signed volume, a trailing regression slope.

    For each grid step, :math:`r_t = \ln m_t - \ln m_{t-1}` and :math:`v_t` is
    the signed volume that arrived during the step. :math:`\lambda_t` is the
    OLS slope of :math:`r` on :math:`v` over the trailing ``window`` of grid
    points (see :func:`blinded.features.series.roll_beta`).

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(kyle_lambda("SPOT", 3).round(4)).to_list()  # v = 3, 0, -4
    [None, None, None, -0.0055]
    """
    r = mid(sym).log().diff()
    v = pl.col(f"{sym}.cum_signed_qty").diff()
    return series.roll_beta(r, v, window)
