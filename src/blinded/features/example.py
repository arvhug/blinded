"""A four-point panel small enough to check every example by hand."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from blinded.panel import Panel

__all__ = ["example_panel"]

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _at(seconds: list[float]) -> list[datetime]:
    return [_T0 + timedelta(seconds=s) for s in seconds]


def example_panel() -> Panel:
    """The panel used in the feature library's examples.

    Grid: 1 s steps at 00:00:01, 00:00:02, 00:00:03, 00:00:04 UTC on 2024-01-01.
    Times below are seconds after midnight.

    ``SPOT`` book (each row is the as-of state at the next grid point):

    ====  ===  ===  =======  =======
    time  bid  ask  bid_qty  ask_qty
    ====  ===  ===  =======  =======
    0.5   10   11   3        1
    1.5   10   12   1        1
    2.5   11   12   2        2
    3.5   12   13   1        3
    ====  ===  ===  =======  =======

    ``PERP`` book: bid 10.5 / ask 11.5 at 0.2, then bid 12 / ask 12.5 at 2.2,
    sizes 1 throughout. Mids on the grid are 11, 11, 12.25, 12.25.

    ``SPOT`` trades: buy 2 at 11 (0.6), sell 1 at 10 (0.9), buy 3 at 12 (1.7),
    sell 4 at 12 (3.2). No trade falls in (2, 3].

    Examples
    --------
    >>> example_panel().frame["SPOT.mid"].to_list()
    [10.5, 11.0, 11.5, 12.5]
    """
    spot = pl.DataFrame(
        {
            "ts": _at([0.5, 1.5, 2.5, 3.5]),
            "bid": [10.0, 10.0, 11.0, 12.0],
            "ask": [11.0, 12.0, 12.0, 13.0],
            "bid_qty": [3.0, 1.0, 2.0, 1.0],
            "ask_qty": [1.0, 1.0, 2.0, 3.0],
        }
    )
    perp = pl.DataFrame(
        {
            "ts": _at([0.2, 2.2]),
            "bid": [10.5, 12.0],
            "ask": [11.5, 12.5],
            "bid_qty": [1.0, 1.0],
            "ask_qty": [1.0, 1.0],
        }
    )
    trades = pl.DataFrame(
        {
            "ts": _at([0.6, 0.9, 1.7, 3.2]),
            "price": [11.0, 10.0, 12.0, 12.0],
            "qty": [2.0, 1.0, 3.0, 4.0],
            "side": [1.0, -1.0, 1.0, -1.0],
        }
    )
    grid = _at([1, 2, 3, 4])
    return Panel.from_frames(grid, book={"SPOT": spot, "PERP": perp}, trades={"SPOT": trades})
