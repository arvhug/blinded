r"""The consolidated price of a coin over its venues, as a column of the panel.

``F.theo`` reads the output of the Kalman filter that ``blinded init`` ran once over the
whole file with parameters fitted on train (:mod:`blinded.theo`): causal by construction,
since the filter is a one-pass recursion, so it passes the truncation test as ``F.mid``
does.
"""

from __future__ import annotations

import polars as pl

__all__ = ["theo"]

_FIELDS = ("theo", "sd_bps", "innovation_z")


def theo(coin: str, venues=None, field: str = "theo") -> pl.Expr:
    r"""The theo of ``coin`` as-of each grid point: the filter's output at the last event
    available by then, the ``THEO.<coin>.<field>`` column.

    ``field`` is ``theo`` (the consolidated price, in the anchor venue's
    terms), ``sd_bps`` (its posterior sd, in bps), ``innovation_z`` (the last
    event's innovation over its predicted sd) or ``basis:<venue>`` (that
    venue's fitted basis over the theo, in bps). ``venues`` must be None or
    the brief's own legs (``data.theo.legs``): the theo is computed once, at
    init, over those.

    Examples
    --------
    >>> theo("SOLUSDT").meta.output_name()
    'THEO.SOLUSDT.theo'
    >>> theo("SOLUSDT", field="basis:SOLUSDT_OKX").meta.output_name()
    'THEO.SOLUSDT.basis_SOLUSDT_OKX'
    """
    if venues is not None and not isinstance(venues, list | tuple):
        raise TypeError("venues is a list of the theo's legs, or None")
    if field.startswith("basis:"):
        name = "basis_" + field.split(":", 1)[1]
    elif field in _FIELDS:
        name = field
    else:
        raise ValueError(f"field must be one of {', '.join(_FIELDS)} or basis:<venue>, "
                         f"got {field!r}")  # fmt: skip
    return pl.col(f"THEO.{coin}.{name}")
