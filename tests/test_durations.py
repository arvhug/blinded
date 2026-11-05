import datetime as dt

import pandas as pd
import pytest

from blinded.durations import polars_span, to_ns

known = pytest.mark.known_answer
edge = pytest.mark.edge_case


@known
@pytest.mark.parametrize(
    "text, ns",
    [
        ("5m", 300_000_000_000),
        ("5min", 300_000_000_000),
        ("100ms", 100_000_000),
        ("1h30m", 5_400_000_000_000),
        ("2h 30min", 9_000_000_000_000),
        ("1.5s", 1_500_000_000),
        ("10us", 10_000),
        ("00:05:00", 300_000_000_000),
    ],
)
def test_polars_and_pandas_spellings(text, ns):
    assert to_ns(text) == ns


@known
def test_timedeltas_and_polars_strings():
    assert to_ns(dt.timedelta(milliseconds=250)) == 250_000_000
    assert to_ns(pd.Timedelta("1s")) == 1_000_000_000
    assert polars_span("5min") == "300000000000ns"


@edge
@pytest.mark.parametrize("value", [5, 2.5, "5", "1mo"])
def test_bare_numbers_and_calendar_units_are_rejected(value):
    with pytest.raises(ValueError):
        to_ns(value)
