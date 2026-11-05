import numpy as np
import pandas as pd
import polars as pl
import pytest
from conftest import brief_dict, write_brief
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal

from blinded import Panel
from blinded import brief as briefs
from blinded import features as F

nan = np.nan
known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case
T0 = pd.Timestamp("2024-11-28 02:00")


def book_row(t, sym, bid, ask, bq=1.0, aq=1.0):
    return dict(
        t=t,
        symbol=sym,
        is_trade=False,
        side=None,
        price=nan,
        qty=nan,
        bid0_px=bid,
        ask0_px=ask,
        bid0_qty=bq,
        ask0_qty=aq,
    )


def trade_row(t, sym, side, price, qty):
    return dict(
        t=t,
        symbol=sym,
        is_trade=True,
        side=side,
        price=price,
        qty=qty,
        bid0_px=price - 1,
        ask0_px=price + 1,
        bid0_qty=1.0,
        ask0_qty=1.0,
    )


def build(tmp_path, rows, edit=None):
    """Write rows in the XRP layout, build the example brief around them, load the panel."""
    frame = pd.DataFrame(rows)
    frame.index = T0 + pd.to_timedelta(frame.pop("t"), unit="s")
    frame.index.name = "ts"
    frame["mid"] = (frame.bid0_px + frame.ask0_px) / 2
    frame["side"] = frame.side.astype("category")
    frame.to_parquet(tmp_path / "events.parquet")
    raw = brief_dict(tmp_path / "events.parquet")
    if edit is not None:
        edit(raw)
    return Panel.from_brief(briefs.load(write_brief(raw, tmp_path / "brief.yaml")))


@pytest.fixture
def rows():
    return [
        book_row(0.05, "A", 10, 12),
        book_row(0.10, "A", 11, 13),
        book_row(0.25, "A", 12, 14),
        book_row(0.25, "A", 13, 15),
        trade_row(0.12, "A", "B", 13, 2),
        trade_row(0.12, "A", "S", 11, 5),
        trade_row(0.31, "A", "B", 15, 1),
        book_row(0.37, "A", 14, 16),
    ]


@known
def test_grid_runs_from_the_first_event_rounded_up_to_the_last_rounded_down(tmp_path, rows):
    p = build(tmp_path, rows)
    seconds = (p.grid_ns - T0.tz_localize("UTC").value) / 1e9
    assert seconds.round(3).tolist() == [0.1, 0.2, 0.3]


@known
def test_asof_is_backward_inclusive_and_last_duplicate_wins(tmp_path, rows):
    p = build(tmp_path, rows)
    # 0.1 sees the row stamped 0.10; 0.2 still sees it; 0.3 sees the second 0.25 row
    assert_allclose(p.frame["A.mid"], [12, 12, 14])
    assert_allclose(p.frame["A.bid"], [11, 11, 13])


@known
def test_trade_fields_and_side_mapping(tmp_path, rows):
    p = build(tmp_path, rows)
    # two trades at 0.12: the sell is the later row
    assert_allclose(p.frame["A.price"], [nan, 11, 11], equal_nan=True)
    assert_allclose(p.frame["A.side"], [nan, -1, -1], equal_nan=True)


@known
def test_flow_over_a_trailing_window(tmp_path, rows):
    p = build(tmp_path, rows)
    # 200 ms: (-0.1, 0.1] none; (0, 0.2] +2 - 5; (0.1, 0.3] +2 - 5
    assert_allclose(p.eval(F.signed_volume("A", "200ms")), [0, -3, -3])
    assert_allclose(p.eval(F.window_sum(pl.col("A.cum_qty"), "100ms")), [0, 7, 0])


@known
def test_availability_delay_hides_rows_until_they_arrive(tmp_path, rows):
    def delay_trades(raw):
        raw["data"]["availability"]["overrides"] = [
            {"instrument": "A", "stream": "trades", "delay_ms": 90}
        ]

    p = build(tmp_path, rows, delay_trades)
    # trades at 0.12 arrive at 0.21: invisible at 0.2, visible at 0.3. The 0.31 trade
    # arrives at exactly 0.40, which also stretches the grid to a fourth point.
    assert_allclose(p.frame["A.price"], [nan, nan, 11, 15], equal_nan=True)
    assert_allclose(p.frame["A.mid"], [12, 12, 14, 15])  # the book stream has no delay


@known
def test_events_carry_their_availability_delay(tmp_path, rows):
    def delay_trades(raw):
        raw["data"]["availability"]["overrides"] = [
            {"instrument": "A", "stream": "trades", "delay_ms": 90}
        ]

    p = build(tmp_path, rows, delay_trades)
    assert p.events("trades", "A").delay_ns == 90_000_000
    assert p.events("book", "A").delay_ns == 0
    assert p.cut(T0 + pd.Timedelta("237ms")).events("trades", "A").delay_ns == 90_000_000


@edge
def test_a_delay_that_varies_within_a_stream_is_refused():
    t = [T0.tz_localize("UTC") + pd.Timedelta(seconds=s) for s in (0.5, 1.5)]
    book = pl.DataFrame({"ts": t, "available_at": [t[0], t[1] + pd.Timedelta("10ms")],
                         "bid": [1.0, 1.0], "ask": [2.0, 2.0]})  # fmt: skip
    p = Panel.from_frames([t[0] + pd.Timedelta("1s")], book={"A": book}, step="1s")
    with pytest.raises(ValueError, match="one delay"):
        _ = p.events("book", "A").delay_ns


@known
def test_n_counts_the_raw_events_a_thinned_row_stands_for():
    t0 = T0.tz_localize("UTC")

    def at(*seconds):
        return [t0 + pd.Timedelta(seconds=s) for s in seconds]

    book = pl.DataFrame({"ts": at(0.5, 1.5, 2.5), "bid": [1.0] * 3, "ask": [2.0] * 3,
                         "n": [2, 1, 3]})  # fmt: skip
    trades = pl.DataFrame({"ts": at(0.6, 1.7), "price": [1.5, 1.5], "qty": [5.0, 1.0],
                           "side": [1.0, -1.0], "n": [4, 2]})  # fmt: skip
    p = Panel.from_frames(at(1, 2, 3), book={"A": book}, trades={"A": trades})
    # rows would count 1, 2, 3 updates; n says 2, 2 + 1 and 2 + 1 + 3
    assert p.frame["A.cum_book"].to_list() == [2, 3, 6]
    # 4 raw trades by 1 s, 4 + 2 by 2 s; in each 1 s window 4, 2 and 0
    assert p.frame["A.cum_trades"].to_list() == [4, 6, 6]
    assert_allclose(p.eval(F.trade_count("A", "1s")), [4, 2, 0])
    assert "A.n" not in p.frame.columns
    assert p.frame["A.cum_qty"].to_list() == [5.0, 6.0, 6.0]  # quantities are already totals


@known
def test_cut_keeps_events_and_grid_points_up_to_t(tmp_path, rows):
    p = build(tmp_path, rows)
    short = p.cut(T0 + pd.Timedelta("237ms"))
    assert len(short) == 2  # 0.1 and 0.2
    assert short.book("A")["bid"].to_list() == [10, 11]  # the 0.25 rows come after the cut
    assert len(short.trades("A")) == 2


@known
def test_microprice_field(tmp_path):
    p = build(tmp_path, [book_row(0.05, "A", 10, 11, bq=3, aq=1), book_row(0.15, "A", 10, 11)])
    assert_allclose(p.frame["A.microprice"], [10.75])  # (10*1 + 11*3) / 4


@known
def test_grid_start_from_the_brief(tmp_path, rows):
    p = build(tmp_path, rows, lambda raw: raw["grid"].update(start="2024-11-28 02:00:00.2"))
    assert len(p) == 2


@cross
def test_asof_matches_a_loop(tmp_path):
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 20, 400)).round(3)
    rows = [book_row(x, "A", 100 + i, 101 + i) for i, x in enumerate(t)]
    p = build(tmp_path, rows)
    expected = []
    for seconds in (p.grid_ns - T0.tz_localize("UTC").value) / 1e9:
        seen = np.nonzero(t <= seconds + 1e-9)[0]
        expected.append(100.5 + seen[-1] if len(seen) else nan)
    assert_allclose(p.frame["A.mid"], expected, equal_nan=True)


@edge
def test_grid_point_before_any_trade_is_nan(tmp_path, rows):
    p = build(tmp_path, rows)
    assert p.frame["A.price"][0] is None


@edge
def test_unknown_field_and_stream_raise_clear_errors(tmp_path, rows):
    p = build(tmp_path, rows)
    with pytest.raises(KeyError, match="no field 'vwap'"):
        p.col("A", "vwap")
    with pytest.raises(KeyError, match="no 'trades' stream for 'B'"):
        p.trades("B")


@edge
def test_microsecond_grid_is_held_in_nanoseconds():
    grid = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC"), periods=2, freq="1s").as_unit("us")
    quotes = pd.DataFrame(
        {"bid": [1.0], "ask": [3.0], "bid_qty": [1.0], "ask_qty": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00.5", tz="UTC")]),
    )
    p = Panel.from_pandas(grid, book={"A": quotes})
    assert_allclose(p.frame["A.mid"], [nan, 2.0], equal_nan=True)


@known
def test_frame_is_wide_with_one_column_per_instrument_and_field(tmp_path, rows):
    p = build(tmp_path, rows)
    assert p.frame.columns[0] == "ts"
    assert p.frame.schema["ts"] == pl.Datetime("ns", "UTC")
    assert p.frame["A.mid"].to_list() == [12, 12, 14]
    assert p.frame["A.price"].to_list() == [None, 11, 11]
    # trades at 0.12 (buy 2 at 13, sell 5 at 11) and 0.31 (buy 1 at 15, after 0.3)
    assert p.frame["A.cum_trades"].to_list() == [0, 2, 2]
    assert p.frame["A.cum_signed_qty"].to_list() == [0, -3, -3]
    assert p.frame["A.cum_notional"].to_list() == [0, 81, 81]  # 13*2 + 11*5
    # book rows at 0.05, 0.10 | 0.25, 0.25 | 0.37
    assert p.frame["A.cum_book"].to_list() == [2, 2, 4]


@known
def test_col_names_a_frame_column(tmp_path, rows):
    p = build(tmp_path, rows)
    assert p.eval(p.col("A", "bid")).to_list() == [11, 11, 13]
    with pytest.raises(KeyError, match="no field 'vwap'"):
        p.col("A", "vwap")


@known
def test_to_pandas_is_indexed_by_ts(tmp_path, rows):
    p = build(tmp_path, rows)
    df = p.to_pandas()
    assert df.index.name == "ts"
    assert (df.index.asi8 == p.grid_ns).all()
    assert_allclose(df["A.price"], [nan, 11, 11], equal_nan=True)


@edge
def test_to_pandas_takes_names_or_a_list(tmp_path, rows):
    p = build(tmp_path, rows)
    assert p.to_pandas("A.bid", "A.price").columns.tolist() == ["A.bid", "A.price"]
    assert p.to_pandas(["A.bid", "A.price"]).columns.tolist() == ["A.bid", "A.price"]
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        p.to_pandas(["A.vwap"])


def rebuilt(panel, t):
    """A panel built from scratch out of the events available by ``t``."""
    t_ns = pd.Timestamp(t).value
    events = {key: panel.events(*key).head(t_ns) for key in panel.streams}
    grid = panel.grid_ns[panel.grid_ns <= t_ns]
    return Panel.build(events, grid, panel.tz, panel.step_ns)


def assert_cut_matches_rebuild(panel, t):
    cut, fresh = panel.cut(t), rebuilt(panel, t)
    assert_frame_equal(cut.frame, fresh.frame)
    for key in panel.streams:
        assert_frame_equal(cut.events(*key).frame, fresh.events(*key).frame)


@known
def test_cut_frame_equals_a_rebuild_on_a_small_panel(tmp_path, rows):
    p = build(tmp_path, rows)
    for ms in ("120ms", "237ms", "300ms", "1s"):
        assert_cut_matches_rebuild(p, T0.tz_localize("UTC") + pd.Timedelta(ms))


@known
def test_cut_frame_equals_a_rebuild_for_every_column_type(dataset):
    _, _, panel = dataset
    # as-of floats, derived mid and microprice, integer event counts, cumulative sums
    assert set(panel.frame.schema.values()) == {pl.Float64, pl.Int64, pl.Datetime("ns", "UTC")}
    assert {"XRPUSDT.microprice", "XRPUSDT.cum_book", "XRPUSDT.cum_notional"} <= set(
        panel.frame.columns
    )
    span = panel.grid_ns[-1] - panel.grid_ns[0]
    for q in (0.0, 0.3137, 0.75):
        t = pd.Timestamp(int(panel.grid_ns[0] + q * span) + 37_000_000, tz="UTC")
        assert_cut_matches_rebuild(panel, t)


def test_columns_no_stream_reads_change_nothing(synthetic_brief):
    # each stream reads only its own columns: a wide column of nulls, as a file's trade rows
    # carry for the book's levels, leaves every event table as it was
    from blinded.panel import load_events

    brief = briefs.load(synthetic_brief)
    frame = pl.read_parquet(brief.data.path)
    wide = frame.with_columns(pl.lit(None, pl.Float64).alias(f"junk{k}") for k in range(8))
    plain, padded = load_events(brief, frame), load_events(brief, wide)
    assert plain.keys() == padded.keys()
    for key in plain:
        assert_frame_equal(plain[key].frame, padded[key].frame)
