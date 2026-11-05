import pandas as pd
import pytest
import yaml
from conftest import EXAMPLE_BRIEF

from blinded import brief as briefs

known = pytest.mark.known_answer
edge = pytest.mark.edge_case
UTC = "UTC"


def ts(text):
    return pd.Timestamp(text, tz=UTC)


@pytest.fixture
def example(monkeypatch):
    monkeypatch.setenv("BLINDED_DATA_PATH", "/data/somewhere.parquet")
    return briefs.load(EXAMPLE_BRIEF)


@known
def test_example_brief_parses(example):
    assert str(example.data.path) == "/data/somewhere.parquet"
    assert example.data.streams["book"].filter == "is_trade == false"
    assert example.data.streams["trades"].side_values == {"buy": "B", "sell": "S"}
    assert example.target.primary_horizon == pd.Timedelta("10s")
    assert example.latency == pd.Timedelta("50ms")
    assert example.grid.step == pd.Timedelta("100ms")


@known
def test_named_windows_for_six_hours(example):
    w = briefs.windows(example, ts("2024-11-28 02:00"), ts("2024-11-28 08:00"))
    # 50/25/25 of six hours with a 5 min embargo before val and before test
    assert (w["train"].start, w["train"].end) == (ts("2024-11-28 02:00"), ts("2024-11-28 05:00"))
    assert (w["val"].start, w["val"].end) == (ts("2024-11-28 05:05"), ts("2024-11-28 06:30"))
    assert w["test"].start == ts("2024-11-28 06:35")
    assert w["test"].end == ts("2024-11-28 08:00:00.1")  # one step past the last point


@known
def test_explicit_window(example):
    w = briefs.window("2024-11-28T04:59:55.5/2024-11-28T07:59:40", example, ts("2024"), ts("2025"))
    assert (w.start, w.end) == (ts("2024-11-28 04:59:55.5"), ts("2024-11-28 07:59:40"))


@known
def test_delay_overrides():
    raw = yaml.safe_load(EXAMPLE_BRIEF.read_text())
    raw["data"]["path"] = "x.parquet"
    raw["data"]["availability"] = {
        "default_delay_ms": 5,
        "overrides": [{"instrument": "P", "stream": "trades", "delay_ms": 50}],
    }
    data = briefs.parse(raw).data
    assert data.delay("trades", "P") == pd.Timedelta("50ms")
    assert data.delay("book", "P") == pd.Timedelta("5ms")
    assert data.delay("trades", "S") == pd.Timedelta("5ms")


@edge
def test_relative_data_path_resolves_against_the_brief(tmp_path):
    raw = yaml.safe_load(EXAMPLE_BRIEF.read_text())
    raw["data"]["path"] = "events.parquet"
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(raw))
    assert briefs.load(tmp_path / "b.yaml").data.path == tmp_path / "events.parquet"


@edge
def test_unset_variable_is_an_error(monkeypatch):
    monkeypatch.delenv("BLINDED_DATA_PATH", raising=False)
    with pytest.raises(ValueError, match="BLINDED_DATA_PATH"):
        briefs.load(EXAMPLE_BRIEF)


@edge
def test_bad_values_are_rejected(example):
    raw = yaml.safe_load(EXAMPLE_BRIEF.read_text())
    raw["data"]["path"] = "x"
    with pytest.raises(ValueError, match="unit"):
        briefs.parse({**raw, "grid": {"step": 100}})
    with pytest.raises(ValueError, match="sum"):
        briefs.parse({**raw, "splits": {"fractions": [0.5, 0.5, 0.5]}})
    with pytest.raises(ValueError, match="primary_horizon"):
        briefs.parse({**raw, "target": {**raw["target"], "primary_horizon": "5s"}})
    with pytest.raises(ValueError, match="START/END"):
        briefs.window("tomorrow", example, ts("2024"), ts("2025"))


def with_families(families, **execution):
    raw = yaml.safe_load(EXAMPLE_BRIEF.read_text())
    raw["data"]["path"] = "x.parquet"
    raw["families"] = families
    raw["execution"] |= execution
    return briefs.parse(raw)


@edge
def test_a_brief_without_families_has_one_directional_family_on_the_target(example):
    (family,) = example.families
    assert (family.name, family.share, family.instruments) == ("directional", 1.0, ("XRPUSDT",))
    assert not example.has_families
    assert example.family("pairs") is None


@known
def test_families_keep_brief_order_and_share_equally_by_default():
    brief = with_families(
        {"pairs": {"pairs": [["XRPUSDT_PERP", "XRPUSDT"]]}, "directional": None,
         "lead_lag": {"leaders": ["XRPUSDT_PERP"], "followers": ["XRPUSDT"]}}
    )  # fmt: skip
    assert [f.name for f in brief.families] == ["pairs", "directional", "lead_lag"]
    assert [f.share for f in brief.families] == pytest.approx([1 / 3] * 3)
    pairs, directional, lead_lag = brief.families
    assert pairs.pairs == (("XRPUSDT_PERP", "XRPUSDT"),)
    assert pairs.defaults == {"window": "2h", "every": "30m", "z_in": 2.0, "z_out": 0.5}
    assert directional.instruments == ("XRPUSDT",)  # the target, when none are listed
    assert lead_lag.targets == ("XRPUSDT",) and lead_lag.symbols == ("XRPUSDT_PERP", "XRPUSDT")


@known
def test_explicit_shares_and_pairs_defaults():
    brief = with_families(
        {"directional": {"share": 0.25, "instruments": ["XRPUSDT", "XRPUSDT_PERP"]},
         "pairs": {"share": 0.75, "pairs": [["XRPUSDT_PERP", "XRPUSDT"]],
                   "defaults": {"window": "10m", "z_in": 3.0}}}
    )  # fmt: skip
    directional, pairs = brief.families
    assert (directional.share, pairs.share) == (0.25, 0.75)
    assert directional.targets == ("XRPUSDT", "XRPUSDT_PERP")
    assert pairs.defaults == {"window": "10m", "every": "30m", "z_in": 3.0, "z_out": 0.5}
    assert brief.has_families


@known
def test_a_per_instrument_fee_overrides_the_taker_fee():
    ex = with_families(None, taker_fee_bps=7.5, fees={"XRPUSDT": 10.0}).execution
    assert ex.fee("XRPUSDT") == 10.0 and ex.fee("XRPUSDT_PERP") == 7.5


@edge
@pytest.mark.parametrize(
    "families, message",
    [
        ({"momentum": {}}, "unknown families"),
        ({"directional": {"share": 0.5}, "pairs": {"pairs": [["A", "B"]]}}, "every family"),
        ({"directional": {"share": 0.6}, "pairs": {"share": 0.6, "pairs": [["A", "B"]]}},
         "sum to 1"),
        ({"directional": {"share": 0}, "pairs": {"share": 1, "pairs": [["A", "B"]]}}, "positive"),
        ({"pairs": {"pairs": [["A", "A"]]}}, "two different"),
        ({"pairs": {"pairs": [["A", "B", "C"]]}}, "two different"),
        ({"pairs": {}}, "needs pairs"),
        ({"lead_lag": {"leaders": ["A"], "followers": ["A"]}}, "not the same"),
        ({"pairs": {"pairs": [["A", "B"]], "defaults": {"z_in": 0.5, "z_out": 1.0}}}, "z_out"),
        (["directional", "pairs"], "map each family"),
    ],
)  # fmt: skip
def test_bad_families_are_rejected(families, message):
    with pytest.raises(ValueError, match=message):
        with_families(families)


def thinned_raw(**changes):
    """The example brief on a 500 ms grid, its file thinned for a 50 ms delay and 50 ms latency."""
    raw = yaml.safe_load(EXAMPLE_BRIEF.read_text())
    raw["data"]["path"] = "x.parquet"
    raw["data"]["availability"] = {"default_delay_ms": 50, "overrides": []}
    raw["data"]["thinned"] = {"step": "500ms", "offsets_ms": [-50, 0, 50]}
    raw["grid"]["step"] = "500ms"
    raw["target"]["horizons"] = ["500ms", "1s", "10s", "60s"]
    for key, value in changes.items():
        section, name = key.split("__")
        raw[section][name] = value
    return raw


@known
def test_a_thinned_file_keeps_the_panel_the_fills_and_whole_step_returns():
    from blinded.evaluate import _all_horizons

    brief = briefs.parse(thinned_raw())
    thin = brief.data.thinned
    assert thin.step == pd.Timedelta("500ms")
    # the panel reads stamp g - 50 ms, latency 0 reads g and the fills g + 50 ms, each a
    # whole number of steps from a kept offset; 100 ms and 250 ms are not
    keeps = [brief.data.exact(pd.Timedelta(f"{ms}ms")) for ms in (-50, 0, 50, 550, 100, 250)]
    assert keeps == [True, True, True, True, False, False]
    # the decay ladder's 100 ms and 200 ms end at g + 150 and g + 250 ms: left out
    assert [h / pd.Timedelta("1s") for h in _all_horizons(brief)] == [0.5, 1, 2, 5, 10, 30, 60]


@edge
@pytest.mark.parametrize(
    "changes, message",
    [
        ({"grid__step": "100ms"}, "grid.step is 100ms"),
        ({"execution__latency_ms": 100}, "fills at 100ms"),
        ({"target__horizons": ["750ms", "10s"], "target__primary_horizon": "10s"},
         "750ms are not whole steps"),
    ],
)  # fmt: skip
def test_a_brief_cannot_read_stamps_its_thinned_file_dropped(changes, message):
    with pytest.raises(ValueError, match=message):
        briefs.parse(thinned_raw(**changes))


@edge
def test_a_thinned_file_refuses_another_feed_delay():
    raw = thinned_raw()
    raw["data"]["availability"]["overrides"] = [{"instrument": "XRPUSDT", "delay_ms": 20}]
    with pytest.raises(ValueError, match="the panel at a 20ms delay"):
        briefs.parse(raw)


def maker_raw(**execution) -> dict:
    return {"data": {"path": "x.parquet", "instrument_column": "symbol",
                     "streams": {"book": {"fields": {"bid": "bid"}}}},
            "target": {"instrument": "X", "horizons": ["1s"]},
            "execution": {"latency_ms": 50} | execution}  # fmt: skip


@known
def test_the_fill_engines_execution_fields_default_to_the_order_latency(tmp_path):
    brief = briefs.parse(maker_raw())
    ex = brief.execution
    assert (ex.tick_size, ex.lot_size, ex.cancel_latency, ex.response_latency) == (None,) * 4
    assert ex.cancel == ex.response == pd.Timedelta("50ms")
    ex = briefs.parse(maker_raw(tick_size=0.01, lot_size=0.001, cancel_latency_ms=20,
                                response_latency_ms=0)).execution  # fmt: skip
    assert (ex.tick_size, ex.lot_size) == (0.01, 0.001)
    assert (ex.cancel, ex.response) == (pd.Timedelta("20ms"), pd.Timedelta(0))
    raw = maker_raw()
    raw["data"]["touch"] = {"path": "x.touch.parquet", "tick_size": 0.01}
    touch = briefs.parse(raw, base=tmp_path).data.touch
    assert (touch.path, touch.tick_size) == (tmp_path / "x.touch.parquet", 0.01)
    # the touch belongs to the target unless it names its instrument
    assert touch.instrument == raw["target"]["instrument"]
    raw["data"]["touch"]["instrument"] = "OTHER"
    assert briefs.parse(raw, base=tmp_path).data.touch.instrument == "OTHER"
    raw["data"]["depth"] = {"path": "x.depth.parquet", "venue": "OTHER", "levels": 5,
                            "lag_ms": 100}  # fmt: skip
    depth = briefs.parse(raw, base=tmp_path).data.depth
    assert (depth.path, depth.venue, depth.levels, depth.lag_ms) == (
        tmp_path / "x.depth.parquet",
        "OTHER",
        5,
        100.0,
    )


@edge
def test_the_fill_engines_fields_are_checked():
    # one resting order a side: a cancel slower than its replacement would leave two
    with pytest.raises(ValueError, match="must not be longer than latency_ms"):
        briefs.parse(maker_raw(cancel_latency_ms=60))
    with pytest.raises(ValueError, match="tick_size must be positive"):
        briefs.parse(maker_raw(tick_size=0))
    with pytest.raises(ValueError, match="response_latency_ms must not be negative"):
        briefs.parse(maker_raw(response_latency_ms=-1))
    raw = maker_raw()
    raw["data"]["touch"] = {"path": "x.touch.parquet"}
    with pytest.raises(ValueError, match="data.touch needs a positive tick_size"):
        briefs.parse(raw)
    raw = maker_raw()
    raw["data"]["depth"] = {"path": "x.depth.parquet"}
    with pytest.raises(ValueError, match="data.depth needs a venue"):
        briefs.parse(raw)


def _arb_raw():
    raw = yaml.safe_load(EXAMPLE_BRIEF.read_text())
    raw["data"]["path"] = "x.parquet"
    raw["data"]["depth"] = {"path": "d.parquet", "venue": "B", "levels": 25, "lag_ms": 100}
    raw["execution"] |= {"latency_ms": 50, "hedge_latency_ms": 50, "hedge_fee_bps": 10}
    raw["stock"] = {
        "A": {"coin": 3225, "usdt": 500},
        "B": {"coin": 3225, "usdt": 500},
        "rebalance_at": 0.8,
        "transfer_minutes": 30,
        "withdrawal_fee_coin": None,
    }
    return raw


@known
def test_the_hedge_and_the_stock_are_read():
    b = briefs.parse(_arb_raw())
    assert b.execution.hedge_latency == pd.Timedelta("50ms")
    assert b.execution.hedge_fee_bps == 10.0
    assert b.stock.coin == {"A": 3225.0, "B": 3225.0}
    assert b.stock.usdt == {"A": 500.0, "B": 500.0}
    assert (b.stock.rebalance_at, b.stock.transfer) == (0.8, pd.Timedelta("30min"))
    assert b.stock.withdrawal_fee_coin is None  # unset until looked up
    assert b.data.depth.lag_ms == 100.0
    plain = briefs.parse({**_arb_raw(), "stock": None})
    assert plain.stock is None
    raw = _arb_raw()
    del raw["execution"]["hedge_latency_ms"], raw["execution"]["hedge_fee_bps"]
    assert briefs.parse(raw).execution.hedge_latency is None
    assert briefs.parse(raw).execution.hedge_fee_bps is None


@edge
@pytest.mark.parametrize(
    "change, message",
    [
        # 30 ms of hedge latency and 50 of response is 80 ms after the fill, under the 100 ms
        # the snapshot trails the touch: the hedge would read a book older than its fill
        (lambda r: r["execution"].update(hedge_latency_ms=30), "older than the fill"),
        (lambda r: r["stock"].pop("withdrawal_fee_coin"), "withdrawal_fee_coin must be given"),
        (lambda r: r["stock"].update(withdrawal_fee_coin=-1), "must not be negative"),
        (lambda r: r["stock"].update(rebalance_at=0.4), "rebalance_at"),
        (lambda r: r["stock"].pop("B"), "names 1 venues"),
        (lambda r: r["stock"]["A"].update(btc=1), "coin and usdt"),
    ],
)
def test_the_hedge_and_the_stock_are_checked(change, message):
    raw = _arb_raw()
    change(raw)
    with pytest.raises(ValueError, match=message):
        briefs.parse(raw)


@known
def test_a_hedge_as_slow_as_the_depth_lag_is_allowed():
    raw = _arb_raw()
    raw["execution"] |= {"hedge_latency_ms": 70, "response_latency_ms": 30}
    assert briefs.parse(raw).execution.hedge_latency == pd.Timedelta("70ms")
