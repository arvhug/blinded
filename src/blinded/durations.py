"""Time spans written polars-style (``"5m"``, ``"100ms"``) or pandas-style (``"5min"``).

Everything that takes a span goes through :func:`to_ns`, so the two spellings
mean the same thing everywhere: in the brief, in panel windows and in the
feature library.
"""

from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

_NS = {
    "ns": 1,
    "us": 1_000,
    "µs": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60_000_000_000,
    "min": 60_000_000_000,
    "h": 3_600_000_000_000,
    "d": 86_400_000_000_000,
    "w": 604_800_000_000_000,
}
# "min" before "m" and "ms" before "m" and "s": alternation takes the first that fits
_UNIT = r"(ns|us|µs|ms|min|s|m|h|d|w)"
_PART = re.compile(rf"(\d+(?:\.\d*)?)\s*{_UNIT}")
_WHOLE = re.compile(rf"\s*(?:\d+(?:\.\d*)?\s*{_UNIT}\s*)+")
_BARE = re.compile(r"\s*[-+]?\d+(?:\.\d*)?\s*")

Span = str | dt.timedelta | np.timedelta64


def to_ns(value: Span) -> int:
    """A span in integer nanoseconds.

    Strings combine number-unit parts (``"1h30m"``, ``"250ms"``, ``"5min"``);
    anything else pandas can parse (``"00:05:00"``, ``"1 day"``) also works.
    Calendar units (``"1mo"``) and bare numbers are rejected: a bare number
    has no unit, and a month has no fixed length.

    Examples
    --------
    >>> to_ns("5m") == to_ns("5min") == 300 * 10**9
    True
    >>> to_ns("1h30m") // 60_000_000_000
    90
    """
    bare = isinstance(value, str) and _BARE.fullmatch(value)
    if bare or isinstance(value, bool | int | float | np.integer | np.floating):
        n = str(value).strip()
        raise ValueError(f"a time span needs a unit, e.g. '{n}s' or '{n}ms'")
    if isinstance(value, str) and _WHOLE.fullmatch(value):
        return round(sum(float(n) * _NS[u] for n, u in _PART.findall(value)))
    try:
        return pd.Timedelta(value).value
    except ValueError as exc:
        raise ValueError(f"cannot read {value!r} as a time span: {exc}") from None


def polars_span(value: Span) -> str:
    """A span as a polars duration string in nanoseconds, e.g. ``"300000000000ns"``."""
    return f"{to_ns(value)}ns"


def timedelta(value: Span) -> pd.Timedelta:
    """A span as a ``pandas.Timedelta`` with nanosecond resolution."""
    return pd.Timedelta(to_ns(value), unit="ns")


def label(value: Span) -> str:
    """Short label for a span: ``100ms``, ``10s``, ``5min``.

    Examples
    --------
    >>> [label(s) for s in ("100ms", "10s", "90s", "5m")]
    ['100ms', '10s', '90s', '5min']
    """
    ms = to_ns(value) / 1e6
    if ms < 1000:
        return f"{ms:g}ms"
    if ms < 120_000 or ms % 60_000:
        return f"{ms / 1000:g}s"
    return f"{ms / 60_000:g}min"


_WORDS = (("d", 86_400_000_000_000), ("h", 3_600_000_000_000), ("min", 60_000_000_000),
          ("s", 1_000_000_000), ("ms", 1_000_000), ("µs", 1_000), ("ns", 1))  # fmt: skip


def words(value: Span) -> str:
    """A span as a person writes it: whole weeks as weeks, anything else in its parts.

    For sentences, where :func:`label` is for column names: ``40 weeks``,
    ``3 h 25 min``, ``110 d 12 h``, ``500 ms``.

    Examples
    --------
    >>> [words(s) for s in ("40w", "14d", "3h25m", "110d12h", "500ms", "10us", "0s")]
    ['40 weeks', '2 weeks', '3 h 25 min', '110 d 12 h', '500 ms', '10 µs', '0 s']
    """
    ns = to_ns(value)
    week = 7 * _NS["d"]
    if ns >= week and ns % week == 0:
        n = ns // week
        return f"{n} week" + ("s" if n != 1 else "")
    parts = []
    for unit, size in _WORDS:
        n, ns = divmod(ns, size)
        if n:
            parts.append(f"{n} {unit}")
    return " ".join(parts) or "0 s"
