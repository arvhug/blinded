r"""Features across a cross-section: one expression per leg, combined row by row.

A basket feature returns a mapping with one value per leg, the instruments of
the brief's basket family in order (``panel.legs``). :func:`each` builds one
from a function of a symbol; the ``xs_`` functions take such a mapping and
return a :class:`Cross` with the same keys, each row computed from that row's
values across the legs only, so they are as causal as their inputs. Nulls and
NaN are left out of a row's mean, spread and ranks.

A :class:`Cross` keeps the legs as one list expression and hands out each leg
as an element of it, so the judge evaluates the cross-section once rather than
once per leg, and the ``xs_`` functions nest without multiplying the work.

Examples use three legs whose values on one row are 1, 2 and 3.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import polars as pl

__all__ = ["each", "xs_demean", "xs_mean", "xs_rank", "xs_zscore"]


class Cross(dict):
    """A cross-section: ``{leg: expression}``, each leg the element of one list expression,
    ``values``, in the order of the keys. Use it as the mapping it is."""

    def __init__(self, legs: Iterable[str], values: pl.Expr):
        legs = list(legs)
        super().__init__({leg: values.list.get(i, null_on_oob=True) for i, leg in enumerate(legs)})
        self.values = values


def as_list(d: Mapping[str, pl.Expr]) -> pl.Expr:
    """The legs of ``d`` as one list expression in key order, NaN read as null."""
    if isinstance(d, Cross):
        return d.values
    return pl.concat_list(*(v.cast(pl.Float64).fill_nan(None) for v in d.values()))


def each(legs: Iterable[str], fn: Callable[[str], pl.Expr]) -> dict[str, pl.Expr]:
    """``{leg: fn(leg)}`` for every leg, in order.

    Examples
    --------
    >>> import polars as pl
    >>> sorted(each(["A", "B"], lambda s: pl.col(f"{s}.mid")))
    ['A', 'B']
    """
    return {leg: fn(leg) for leg in legs}


def _row(d: Mapping[str, pl.Expr], fn: Callable[[pl.Expr], pl.Expr]) -> Cross:
    return Cross(d, as_list(d).list.eval(fn(pl.element())))


def xs_mean(d: Mapping[str, pl.Expr]) -> Cross:
    """The row's mean across the legs, on every leg.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    >>> df.select(**xs_mean({k: pl.col(k) for k in "abc"})).row(0)
    (2.0, 2.0, 2.0)
    """
    return _row(d, lambda e: e.fill_null(0.0) * 0.0 + e.mean())


def xs_demean(d: Mapping[str, pl.Expr]) -> Cross:
    """Each leg less the row's mean across the legs.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    >>> df.select(**xs_demean({k: pl.col(k) for k in "abc"})).row(0)
    (-1.0, 0.0, 1.0)
    """
    return _row(d, lambda e: e - e.mean())


def xs_zscore(d: Mapping[str, pl.Expr]) -> Cross:
    r"""Each leg less the row's mean, over the row's sample standard deviation across the legs.

    :math:`(x_i - \bar x) / s` with :math:`s^2 = \sum_i (x_i - \bar x)^2 / (n - 1)`
    over the :math:`n` legs that have a value. Null where fewer than two do, or
    where they are all equal.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    >>> df.select(**xs_zscore({k: pl.col(k) for k in "abc"})).row(0)  # s = 1
    (-1.0, 0.0, 1.0)
    """
    return _row(d, lambda e: pl.when(e.std() > 0).then((e - e.mean()) / e.std()))


def xs_rank(d: Mapping[str, pl.Expr]) -> Cross:
    """Each leg's rank across the legs in its row, 1 for the smallest, ties sharing the average
    rank; null where the leg has no value.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [3.0], "b": [1.0], "c": [2.0]})
    >>> df.select(**xs_rank({k: pl.col(k) for k in "abc"})).row(0)
    (3.0, 1.0, 2.0)
    """
    return _row(d, lambda e: e.rank())
