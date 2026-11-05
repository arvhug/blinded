r"""Top-of-book features for one instrument, read as-of each grid point.

Each function takes a symbol and returns an expression over the panel's
``"SYM.field"`` columns. Examples use :func:`blinded.features.example_panel`.
"""

from __future__ import annotations

import polars as pl

from blinded.durations import Span, to_ns
from blinded.features._expr import window_sum

__all__ = [
    "depth_imbalance",
    "imbalance",
    "micro_minus_mid",
    "microprice",
    "mid",
    "quote_rate",
    "spread",
    "spread_ticks",
]


def _col(sym: str, field: str) -> pl.Expr:
    return pl.col(f"{sym}.{field}")


def mid(sym: str) -> pl.Expr:
    r"""Mid price, :math:`(b + a) / 2`, the ``SYM.mid`` column.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(mid("SPOT")).to_list()
    [10.5, 11.0, 11.5, 12.5]
    """
    return _col(sym, "mid")


def spread(sym: str, bps: bool = False) -> pl.Expr:
    r"""Quoted spread, :math:`a - b`, or :math:`10^4 (a - b) / m` in bps of the mid.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> p = example_panel()
    >>> p.eval(spread("SPOT")).to_list()
    [1.0, 2.0, 1.0, 1.0]
    >>> p.eval(spread("SPOT", bps=True).round(2)).to_list()  # 1 / 10.5 * 1e4, ...
    [952.38, 1818.18, 869.57, 800.0]
    """
    width = _col(sym, "ask") - _col(sym, "bid")
    return width / mid(sym) * 1e4 if bps else width


def spread_ticks(sym: str, tick: float) -> pl.Expr:
    r"""Spread in ticks, :math:`\mathrm{round}((a - b) / \text{tick})`, halves to even.

    The tick size is a contract constant; pass it. It is never inferred from
    the data, which would make it a full-sample statistic.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(spread_ticks("SPOT", tick=0.5)).to_list()
    [2.0, 4.0, 2.0, 2.0]
    """
    return (spread(sym) / tick).round(0)


def microprice(sym: str) -> pl.Expr:
    r"""Size-weighted mid, :math:`(b\,q_a + a\,q_b) / (q_b + q_a)`.

    Leans towards the side with less size, which is the side more likely to
    move.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(microprice("SPOT")).to_list()  # (10*1 + 11*3) / 4 = 10.75, ...
    [10.75, 11.0, 11.5, 12.25]
    """
    return _col(sym, "microprice")


def imbalance(sym: str) -> pl.Expr:
    r"""Top-of-book size imbalance, :math:`(q_b - q_a) / (q_b + q_a)`, in :math:`[-1, 1]`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(imbalance("SPOT")).to_list()  # (3 - 1) / 4 = 0.5, ...
    [0.5, 0.0, 0.0, -0.5]
    """
    bq, aq = _col(sym, "bid_qty"), _col(sym, "ask_qty")
    return (bq - aq) / (bq + aq)


def depth_imbalance(sym: str, levels: int = 5) -> pl.Expr:
    r"""Size imbalance over the first ``levels`` levels of the book,
    :math:`(\sum_k q^b_k - \sum_k q^a_k) / (\sum_k q^b_k + \sum_k q^a_k)`, in :math:`[-1, 1]`.

    Level 0 is the touch, ``SYM.bid_qty`` and ``SYM.ask_qty``; level :math:`k`
    is ``SYM.bid{k}_qty`` and ``SYM.ask{k}_qty``, which a file built with
    five levels carries as-of the same stamps. One level is :func:`imbalance`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(depth_imbalance("SPOT", levels=1)).to_list()
    [0.5, 0.0, 0.0, -0.5]
    """
    if levels < 1:
        raise ValueError(f"levels must be at least 1, got {levels}")
    names = [("bid_qty", "ask_qty")] + [(f"bid{k}_qty", f"ask{k}_qty") for k in range(1, levels)]
    bids = pl.sum_horizontal(_col(sym, b) for b, _ in names)
    asks = pl.sum_horizontal(_col(sym, a) for _, a in names)
    return (bids - asks) / (bids + asks)


def micro_minus_mid(sym: str, bps: bool = False) -> pl.Expr:
    r"""Microprice minus mid, :math:`\mu - m`, in price units or :math:`10^4 (\mu - m)/m`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(micro_minus_mid("SPOT")).to_list()
    [0.25, 0.0, 0.0, -0.25]
    """
    m = mid(sym)
    gap = microprice(sym) - m
    return gap / m * 1e4 if bps else gap


def quote_rate(sym: str, window: Span) -> pl.Expr:
    r"""Book updates per second over the trailing window, :math:`N_{(t - w, t]} / w`.

    Examples
    --------
    >>> from blinded.features import example_panel
    >>> example_panel().eval(quote_rate("SPOT", "2s")).to_list()  # 1 in (-1, 1], then 2 per 2 s
    [0.5, 1.0, 1.0, 1.0]
    """
    return window_sum(_col(sym, "cum_book"), window) / (to_ns(window) / 1e9)
