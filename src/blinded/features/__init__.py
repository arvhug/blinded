"""A library of standard features as polars expressions, each causal by construction.

Everything is available directly (``bl.features.ewm_zscore``,
``bl.features.basis``) and also through its submodule (``bl.features.cross.basis``).
The test suite runs every function here through the truncation test.
"""

from blinded.features import book, consolidated, cross, series, trades, xs
from blinded.features._expr import window_sum
from blinded.features.book import *  # noqa: F403
from blinded.features.consolidated import theo
from blinded.features.cross import *  # noqa: F403
from blinded.features.example import example_panel
from blinded.features.series import *  # noqa: F403
from blinded.features.trades import *  # noqa: F403
from blinded.features.xs import *  # noqa: F403

__all__ = [
    "book",
    "consolidated",
    "cross",
    "example_panel",
    "series",
    "trades",
    "window_sum",
    "xs",
    *series.__all__,
    *book.__all__,
    *cross.__all__,
    *trades.__all__,
    *xs.__all__,
    "theo",
]
