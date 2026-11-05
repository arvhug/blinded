import pandas as pd
import pytest
import yaml
from conftest import EXAMPLE_BRIEF

from blinded import brief as briefs

known = pytest.mark.known_answer
edge = pytest.mark.edge_case


def ts(text):
    return pd.Timestamp(text, tz="UTC")


def raw_brief(**overrides):
    raw = yaml.safe_load(EXAMPLE_BRIEF.read_text())
    raw["data"]["path"] = "x.parquet"
    for section, values in overrides.items():
        raw[section] = {**raw.get(section, {}), **values}
    return raw


@known
def test_xrp_span_splits_on_whole_minutes():
    # the XRP grid runs 02:00:00.1 .. 07:59:59.8, so the span is 5:59:59.7
    # train end: 02:00:00.1 + 0.50 * 5:59:59.7 = 04:59:59.95, nearest minute 05:00
    # val end:   02:00:00.1 + 0.75 * 5:59:59.7 = 06:29:59.875, nearest minute 06:30
    # embargo 5 min after each; test ends one 100 ms step after the last point
    brief = briefs.parse(raw_brief())
    first, last = ts("2024-11-28 02:00:00.1"), ts("2024-11-28 07:59:59.8")
    w = briefs.windows(brief, first, last)
    assert (w["train"].start, w["train"].end) == (first, ts("2024-11-28 05:00"))
    assert (w["val"].start, w["val"].end) == (ts("2024-11-28 05:05"), ts("2024-11-28 06:30"))
    assert w["test"].start == ts("2024-11-28 06:35")
    assert w["test"].end == ts("2024-11-28 07:59:59.9")


@known
def test_default_embargo_is_five_horizons_or_five_minutes():
    # horizons up to 60 s: max(5 * 60 s, 5 min) = 5 min
    horizons = (pd.Timedelta("10s"), pd.Timedelta("60s"))
    assert briefs.default_embargo(horizons) == pd.Timedelta("5min")
    # 120 s: 5 * 120 s = 10 min
    assert briefs.default_embargo((pd.Timedelta("120s"),)) == pd.Timedelta("10min")
    # 1 s: the 5 min floor
    assert briefs.default_embargo((pd.Timedelta("1s"),)) == pd.Timedelta("5min")
    raw = raw_brief(target={"horizons": ["1s", "2min"], "primary_horizon": "1s"})
    del raw["splits"]["embargo"]
    assert briefs.parse(raw).splits.embargo == pd.Timedelta("10min")


@edge
def test_embargo_must_outlast_latency_plus_the_longest_horizon():
    # 50 ms latency + 60 s horizon reaches 60.05 s past the last train point
    with pytest.raises(ValueError, match="longer than latency"):
        briefs.parse(raw_brief(splits={"embargo": "60s"}))
    assert briefs.parse(raw_brief(splits={"embargo": "61s"})).splits.embargo == pd.Timedelta("61s")


@edge
def test_coarse_grid_rounds_to_its_own_step():
    # 5 min bars over 02:00 .. 07:55: span 5:55, half 2:57:30 -> 04:57:30, nearest 5 min 05:00
    brief = briefs.parse(raw_brief(grid={"step": "5min", "warmup": "0s"}))
    w = briefs.windows(brief, ts("2024-11-28 02:00"), ts("2024-11-28 07:55"))
    assert w["train"].end == ts("2024-11-28 05:00")
    assert w["test"].end == ts("2024-11-28 08:00")


@known
def test_execution_block_defaults():
    raw = raw_brief()
    raw["execution"] = {"latency_ms": 50}
    ex = briefs.parse(raw).execution
    assert (ex.latency, ex.taker_fee_bps, ex.max_position, ex.min_rebalance) == (
        pd.Timedelta("50ms"),
        0.0,
        1.0,
        0.0,
    )
    ex = briefs.parse(raw_brief()).execution
    assert (ex.taker_fee_bps, ex.maker_fee_bps, ex.min_rebalance) == (7.5, 0.0, 0.1)
