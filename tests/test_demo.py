import gzip
import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from blinded import brief as briefs
from blinded import demo, markouts
from blinded import vault as vaults
from blinded.panel import Panel, load_events, make_grid
from blinded.sim import simulate

pytestmark = pytest.mark.usefixtures("rules2")  # hand numbers under rules 2
known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case

DAY = date(2026, 10, 1)
T0 = 1790812800_000000  # 2026-10-01 00:00:00 UTC in microseconds

QUOTES = "exchange,symbol,timestamp,local_timestamp,ask_amount,ask_price,bid_price,bid_amount\n"
TRADES = "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"


def write(path, header, rows):
    with gzip.open(path, "wt") as fh:
        fh.write(header + "".join(",".join(map(str, r)) + "\n" for r in rows))


@pytest.fixture
def raw(tmp_path):
    """Two book files and two trades files; local stamps are offsets from midnight in us."""
    spot_q = [
        ("binance", "XRPUSDT", T0 + 1000, T0 + 1000, 5.0, 1.5001, 1.5000, 7.0),
        ("binance", "XRPUSDT", T0 + 3000, T0 + 3000, 5.0, 1.5000, 1.5000, 7.0),  # locked: dropped
        ("binance", "XRPUSDT", T0 + 5000, T0 + 5000, 6.0, 1.5003, 1.5001, 8.0),
    ]
    spot_t = [
        ("binance", "XRPUSDT", T0 + 500, T0 + 600, 1, "buy", 1.5001, 2.0),  # before any quote
        ("binance", "XRPUSDT", T0 + 4000, T0 + 4100, 2, "sell", 1.5000, 3.0),
        ("binance", "XRPUSDT", T0 + 4900, T0 + 5000, 3, "buy", 1.5003, 4.0),  # ties the quote
    ]
    perp_q = [("binance-futures", "XRPUSDT", T0 + 1000, T0 + 2000, 9.0, 1.4999, 1.4998, 9.0)]
    perp_t = [("binance-futures", "XRPUSDT", T0 + 2000, T0 + 2500, 4, "sell", 1.4998, 10.0)]
    for exchange, kind, rows in (
        ("binance", "book_ticker", spot_q),
        ("binance", "trades", spot_t),
        ("binance-futures", "book_ticker", perp_q),
        ("binance-futures", "trades", perp_t),
    ):
        write(
            tmp_path / demo.raw_name(exchange, kind, DAY, "XRPUSDT"),
            QUOTES if kind == "book_ticker" else TRADES,
            rows,
        )
    return tmp_path


@known
def test_rows_land_in_the_xrp_layout(raw):
    frame = demo.build(raw, DAY, ["XRPUSDT"])
    assert tuple(frame.columns) == demo.COLUMNS
    assert frame.schema["ts"] == pl.Datetime("ns")
    # receive-clock offsets in us: spot quotes 1000 and 5000 (3000 is locked), perp quote 2000,
    # perp trade 2500, spot trades 4100 and 5000 (600 has no book yet)
    us = (frame["ts"].dt.epoch("us") - T0).to_list()
    assert us == [1000, 2000, 2500, 4100, 5000, 5000]
    assert frame["symbol"].to_list() == [
        "XRPUSDT", "XRPUSDT_PERP", "XRPUSDT_PERP", "XRPUSDT", "XRPUSDT", "XRPUSDT",
    ]  # fmt: skip
    # at 5000 the quote and the trade tie; the quote comes first, as in its file
    assert frame["is_trade"].to_list() == [False, False, True, True, False, True]


@known
def test_a_trade_carries_the_book_received_at_or_before_it(raw):
    trades = demo.build(raw, DAY, ["XRPUSDT"]).filter("is_trade")
    rows = trades.select("side", "price", "qty", "bid0_px", "ask0_px", "mid").rows()
    assert rows[0] == ("S", 1.4998, 10.0, 1.4998, 1.4999, (1.4998 + 1.4999) / 2)
    # spot sell at 4100 sees the quote at 1000 (the locked one at 3000 was dropped)
    assert rows[1] == ("S", 1.5, 3.0, 1.5, 1.5001, (1.5 + 1.5001) / 2)
    # spot buy at 5000 ties the quote at 5000: a backward join includes it
    assert rows[2] == ("B", 1.5003, 4.0, 1.5001, 1.5003, (1.5001 + 1.5003) / 2)
    book = demo.build(raw, DAY, ["XRPUSDT"]).filter(~pl.col("is_trade"))
    assert book["side"].null_count() == book.height == 3


@known
def test_the_brief_reads_the_demo_file(raw, tmp_path):
    frame = demo.build(raw, DAY, ["XRPUSDT"])
    frame.write_parquet(tmp_path / "demo.parquet")
    raw_brief = demo.brief_for("demo.parquet", "XRPUSDT", DAY)
    (tmp_path / "demo.brief.yaml").write_text(yaml.safe_dump(raw_brief))
    brief = briefs.load(tmp_path / "demo.brief.yaml")
    example = yaml.safe_load(
        (Path(__file__).parents[1] / "examples" / "demo.brief.yaml").read_text()
    )
    assert raw_brief["target"]["horizons"] == example["target"]["horizons"]
    assert (brief.data.clock, brief.execution.taker_fee_bps) == ("receive", 7.5)
    assert brief.latency == pd.Timedelta("50ms")
    events = Panel.from_brief(brief, frame).streams
    assert sorted(events) == [
        ("book", "XRPUSDT"), ("book", "XRPUSDT_PERP"),
        ("trades", "XRPUSDT"), ("trades", "XRPUSDT_PERP"),
    ]  # fmt: skip


@edge
def test_a_day_that_is_not_free_needs_a_key(tmp_path, monkeypatch):
    pytest.importorskip("tardis_dev")
    monkeypatch.delenv("TARDIS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="not the first of a month"):
        demo.download(date(2026, 10, 2), ["XRPUSDT"], tmp_path)


# -- the Binance builder ----------------------------------------------------------

MS = 1_000_000
STEP = 500 * MS
STAMPS = [-50 * MS, 0, 50 * MS]  # a 50 ms feed delay and a 50 ms latency


@known
def test_the_kept_book_rows_are_the_ones_each_read_returns():
    # reads at g - 50, g and g + 50 ms for g = 0, 500, 1000 and 1500 ms, inside [0, 1500)
    points = demo.read_points(0, 1500 * MS, STEP, STAMPS)
    assert (points // MS).tolist() == [0, 50, 450, 500, 550, 950, 1000, 1050, 1450]
    book = np.array([10, 40, 60, 300, 450, 460, 490, 520, 990, 1200]) * MS
    idx, n = demo.thin_book(book, points)
    # 0 reads nothing yet; 50 reads the row at 40; 450 the one at 450; 500 the one at 490;
    # 550 and 950 the one at 520; 1000 and 1050 the one at 990; 1450 the one at 1200
    assert idx.tolist() == [1, 4, 6, 7, 8, 9]
    # each stands for itself and the dropped rows since the last kept one: 10-40, 60-450,
    # 460-490, then one each; ten rows in all
    assert n.tolist() == [2, 3, 2, 1, 1, 1]


@known
def test_trades_are_summed_per_side_between_the_panels_reads():
    # the panel reads trades up to g - 50 ms: at -50, 450, 950, 1450 ms
    trades = pl.DataFrame({
        "ts": np.array([100, 200, 300, 450, 700, 1460]) * MS,
        "side": ["B", "S", "B", "S", "B", "S"],
        "price": [10.0, 9.0, 11.0, 8.0, 12.0, 7.0],
        "qty": [1.0, 2.0, 3.0, 1.0, 1.0, 5.0],
    })  # fmt: skip
    rows = demo.thin_trades(trades, STEP, -50 * MS).rows()
    ms = [(t // MS, *rest) for t, *rest in rows]
    assert ms == [
        (200, "S", 9.0, 2.0, 1),  # the one sell before the read at 450
        (300, "B", (10 * 1 + 11 * 3) / 4, 4.0, 2),  # two buys: VWAP 43 / 4 = 10.75
        (450, "S", 8.0, 1.0, 1),  # stamped on the read itself: a row of its own
        (700, "B", 12.0, 1.0, 1),
        (1460, "S", 7.0, 5.0, 1),
    ]


def synthetic_day(seconds: int = 600, seed: int = 3):
    """A random-walk perp: book updates every few ms, trades at the touch, stamps in ms."""
    rng = np.random.default_rng(seed)
    t_book = np.sort(rng.integers(0, seconds * 1000, seconds * 40))
    mid = 100 + np.cumsum(rng.choice([-0.01, 0.0, 0.01], len(t_book)))
    book = pl.DataFrame({"t": t_book, "event_time": t_book + 5, "bid": mid - 0.01,
                         "ask": mid + 0.01, "bid_qty": rng.uniform(1, 9, len(t_book)),
                         "ask_qty": rng.uniform(1, 9, len(t_book))})  # fmt: skip
    t_trade = np.sort(rng.integers(0, seconds * 1000, seconds * 6))
    i = np.searchsorted(t_book, t_trade, side="right") - 1
    ok = i >= 0
    t_trade, i = t_trade[ok], i[ok]
    buy = rng.random(len(i)) < 0.5
    trades = pl.DataFrame({"t": t_trade, "side": np.where(buy, "B", "S"),
                           "price": np.where(buy, mid[i] + 0.01, mid[i] - 0.01),
                           "qty": rng.uniform(0.1, 3, len(i)).round(3)})  # fmt: skip
    return book, trades


@pytest.fixture(scope="module")
def thinned(tmp_path_factory):
    """Ten minutes of one synthetic perp, raw and thinned, with the demo's brief."""
    day = date(2024, 3, 27)
    book, trades = synthetic_day()
    t0 = int(pd.Timestamp(day).value) // MS
    book, trades = (f.with_columns(pl.col("t") + t0) for f in (book, trades))
    lattice = {"origin_ns": t0 * MS, "step_ns": STEP, "offsets_ns": STAMPS, "read_ns": -50 * MS}
    kept_book, kept_trades, _ = demo.thin_symbol_day(
        book, trades, "X_PERP", day, lattice, demo.Carry(), last_day=True
    )
    raw = demo.binance_brief("x.parquet", ["X"], day, 1, "500ms", 50, 50)
    raw["families"] = {"directional": {}}
    brief = briefs.parse(raw, base=tmp_path_factory.mktemp("binance"))
    return book, trades, kept_book, kept_trades, brief


@cross
def test_a_panel_from_the_kept_rows_equals_one_from_the_raw_rows(thinned):
    from blinded import sim
    from blinded.brief import Window
    from blinded.panel import load_events
    from blinded.target import forward_returns

    book, trades, kept_book, kept_trades, brief = thinned
    ident = demo.identity_check(book, trades, kept_book, kept_trades, brief, hours=10 / 60)
    # grid points from 0.5 s while a fill at g + 50 ms stays inside the ten minutes: to 599.5 s
    assert ident["grid_points"] == 1199 and ident["kept_rows"] < ident["raw_rows"] / 2
    # the same again by hand: both panels on one grid, every column, the targets and the sim
    raw_rows = pl.concat([
        demo._book_rows(book.select((pl.col("t") * MS).alias("ts"), "bid", "ask", "bid_qty",
                                    "ask_qty", n=pl.lit(1)), "X_PERP"),
        demo._trade_rows(trades.select((pl.col("t") * MS).alias("ts"), "side", "price", "qty",
                                       n=pl.lit(1)), "X_PERP"),
    ]).sort("ts", maintain_order=True)  # fmt: skip
    kept = pl.concat([kept_book, kept_trades]).sort("ts", maintain_order=True)
    t0 = int(pd.Timestamp("2024-03-27").value)
    grid = np.arange(t0 + STEP, t0 + 590 * 10**9, STEP)
    panels = [Panel.build(load_events(brief, f), grid, "UTC", STEP) for f in (raw_rows, kept)]
    a, b = (p.frame.drop("X_PERP.price", "X_PERP.qty") for p in panels)
    summed = "^.*cum_(qty|signed_qty|buy_qty|sell_qty|notional)$"
    assert a.select(pl.all().exclude(summed)).equals(b.select(pl.all().exclude(summed)))
    np.testing.assert_allclose(a["X_PERP.cum_notional"], b["X_PERP.cum_notional"], rtol=1e-12)
    h = brief.target.horizons
    r = [forward_returns(p, "X_PERP", "mid", h, brief.latency) for p in panels]
    assert r[0].equals(r[1])
    rng = np.random.default_rng(1)
    pos = rng.choice([-1.0, 0.0, 1.0], len(grid))
    win = Window("x", panels[0].time(10), panels[0].time(-10))
    s = [sim.simulate(pos, p, win, "X_PERP", "mid", brief.latency, 7.5) for p in panels]
    assert np.array_equal(s[0].net, s[1].net) and s[0].fills.sum() > 100
    m = [sim.maker_proxy(x, p, win, "X_PERP", pd.Timedelta("10s"))
         for x, p in zip(s, panels, strict=True)]  # fmt: skip
    assert m[0] == m[1]


@edge
def test_a_book_on_a_100ms_cadence_fails_the_batching_check():
    book, trades = synthetic_day(60)
    t = (book["t"] // 100) * 100  # every update moved onto a 100 ms beat
    with pytest.raises(demo.BuildError, match="the book looks batched: median gap 100 ms"):
        demo.raw_checks(book.with_columns(t=t), trades, "X 2024-03-27")


@edge
def test_one_stamp_going_backwards_fails_the_build():
    book, trades = synthetic_day(60)
    t = book["t"].to_numpy().copy()
    t[100], t[101] = t[101] + 1, t[100]  # one pair out of order
    with pytest.raises(demo.BuildError, match=r"stamps go backwards \(1 in the book"):
        demo.raw_checks(book.with_columns(t=pl.Series(t)), trades, "X 2024-03-27")


@edge
def test_a_changed_header_fails_with_the_header_in_the_message(tmp_path):
    import zipfile

    path = tmp_path / "X-trades-2024-03-27.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("X-trades-2024-03-27.csv", "id,price,qty,time,is_buyer_maker\n1,2,3,4,true\n")
    with pytest.raises(demo.BuildError, match="the header is 'id,price,qty,time,is_buyer_maker'"):
        demo.read_archive(path, "trades")


@edge
def test_four_days_are_refused_before_anything_is_downloaded(tmp_path, monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("downloaded")

    monkeypatch.setattr(demo, "fetch", no_network)
    # four perpetuals, four days, 172,800 steps a day, 0.81 x 3 + 1.2 = 3.63 rows a step
    with pytest.raises(demo.BuildError, match="over the 10,000,000 limit"):
        demo.binance("2024-03-27", 4, tmp_path, log=print)
    assert "about 10,036,224 rows" in capsys.readouterr().out
    with pytest.raises(AssertionError, match="downloaded"):
        demo.binance("2024-03-27", 4, tmp_path, force=True)


@known
def test_the_binance_brief_has_three_families_and_keeps_its_own_stamps():
    raw = demo.binance_brief("f.parquet", list(demo.SYMBOLS), date(2024, 3, 27), 3, "500ms",
                             50, 50)  # fmt: skip
    brief = briefs.parse(raw)
    assert [f.name for f in brief.families] == ["directional", "lead_lag", "pairs"]
    lead, pairs = brief.family("lead_lag"), brief.family("pairs")
    assert lead.leaders == ("BTCUSDT_PERP", "ETHUSDT_PERP")
    assert lead.followers == ("SOLUSDT_PERP", "XRPUSDT_PERP")
    assert pairs.pairs == (("ETHUSDT_PERP", "BTCUSDT_PERP"), ("XRPUSDT_PERP", "SOLUSDT_PERP"))
    assert (brief.data.clock, brief.data.default_delay_ms, brief.latency) == (
        "exchange", 50.0, pd.Timedelta("50ms"))  # fmt: skip
    assert [o / pd.Timedelta("1ms") for o in brief.data.thinned.offsets] == [-50, 0, 50]
    assert brief.grid.start == pd.Timestamp("2024-03-27", tz="UTC")


# -- klines: bars as rows ----------------------------------------------

BAR0 = pd.Timestamp("2024-06-03", tz="UTC")
MIN = 60_000  # ms


def bars(closes, volume=10.0, step_ms=MIN, start=BAR0):
    t0 = int(start.value // 1_000_000)
    n = len(closes)
    return pl.DataFrame({
        "open_time": [t0 + i * step_ms for i in range(n)],
        "close_time": [t0 + (i + 1) * step_ms - 1 for i in range(n)],
        "close": [float(c) for c in closes], "volume": [volume] * n,
        "quote_volume": [volume * c for c in closes], "count": [5] * n,
        "taker_buy_volume": [volume * 0.6] * n,
    })  # fmt: skip


def bar_brief(path="x.parquet", step="1min", latency_ms=60_000, hs=None, source="2024-06-01"):
    hs = {"X": 10.0} if hs is None else hs
    return briefs.parse({
        "name": "bars",
        "data": {"path": path, "instrument_column": "symbol",
                 "streams": {"book": {"filter": "kind == 'book'",
                                      "fields": {"bid": "bid0_px", "ask": "ask0_px"}},
                             "trades": {"filter": "kind == 'trades'",
                                        "fields": {"price": "price", "qty": "qty",
                                                   "side": "side", "n": "n"},
                                        "side_values": {"buy": "B", "sell": "S"}}},
                 "cost_model": {"kind": "half_spread", "fill": "next bar close",
                                "source": f"data.binance.vision bookTicker {source}",
                                "half_spread_bps": hs}},
        "target": {"instrument": "X", "horizons": [step]},
        "grid": {"step": step, "start": BAR0.isoformat(), "warmup": "0s"},
        "execution": {"latency_ms": latency_ms, "taker_fee_bps": 7.5},
        "splits": {"embargo": "3h"},
    })  # fmt: skip


@known
def test_four_bars_become_book_rows_at_their_close_stamps():
    rows = demo.bar_rows(bars([100, 100, 120, 100]), "X", 10.0, "1m")
    book = rows.filter(pl.col("kind") == "book")
    # each bar's close, less and plus 10 bps of itself, at open_time + 1 min - 1 ms
    assert book["bid0_px"].to_list() == pytest.approx([99.9, 99.9, 119.88, 99.9])
    assert book["ask0_px"].to_list() == pytest.approx([100.1, 100.1, 120.12, 100.1])
    assert book["mid"].to_list() == [100.0, 100.0, 120.0, 100.0]
    stamps = (book["ts"].dt.epoch("ms") - BAR0.value // 1_000_000).to_list()
    assert stamps == [59_999, 119_999, 179_999, 239_999]
    # book, sell, buy at each stamp
    assert rows["kind"].to_list()[:3] == ["book", "trades", "trades"]
    assert rows["side"].to_list()[:3] == [None, "S", "B"]


def bar_panel(closes, latency_ms):
    rows = demo.bar_rows(bars(closes), "X", 10.0, "1m")
    brief = bar_brief(latency_ms=latency_ms)
    events = load_events(brief, rows)
    return Panel.build(events, make_grid(brief, events), "UTC", brief.grid.step.value)


@known
def test_a_taker_on_bars_fills_at_the_next_bars_close():
    # grid points at the bar boundaries 0, 1, 2 and 3 min; long one unit decided at the
    # second boundary between bars (2 min), flat at the third (3 min)
    from blinded.brief import Window

    window = Window("w", BAR0 + pd.Timedelta("2min"), BAR0 + pd.Timedelta("4min"))
    position = np.array([0.0, 0.0, 1.0, 0.0])
    panel = bar_panel([100, 100, 120, 100], 60_000)
    s = simulate(position, panel, window, "X", "mid", pd.Timedelta("1min"), fee_bps=7.5)
    # latency one bar: the buy fills at 3 min on the bar that closed then, 120.12 against a
    # mark of 120, which is the reference: 10 bps of spread. The sell fills at 4 min at 99.9
    # against 100: 0.1 / 120 = 8.3333 bps. Mid P&L (100 - 120) / 120 x 1e4 = -1666.6667;
    # gross -1685.0, fees 2 x 7.5 = 15, net -1700.0 (to rounding: 1e-12 here)
    assert s.mid_pnl.sum() == pytest.approx(-20 / 120 * 1e4)
    assert s.spread_bps.tolist() == pytest.approx([10.0, 0.1 / 120 * 1e4])
    assert s.gross.sum() == pytest.approx(-1685.0, abs=1e-9)
    assert s.fee_paid.sum() == 15.0
    assert s.net.sum() == pytest.approx(-1700.0, abs=1e-9)
    # latency 0: the same decisions fill at 100.1 and 119.88 against a reference of 100:
    # mid +2000, spreads 10 and 12, gross 1978, net 1963
    zero = simulate(position, bar_panel([100, 100, 120, 100], 0), window, "X", "mid",
                    pd.Timedelta(0), fee_bps=7.5)  # fmt: skip
    assert zero.mid_pnl.sum() == pytest.approx(2000.0)
    assert zero.spread_bps.tolist() == pytest.approx([10.0, 12.0])
    assert zero.net.sum() == pytest.approx(1963.0, abs=1e-9)


@known
def test_a_bars_trades_split_by_the_taker_buy_volume():
    one = pl.DataFrame({"open_time": [0], "close_time": [MIN - 1], "close": [101.0],
                        "volume": [10.0], "quote_volume": [1000.0], "count": [5],
                        "taker_buy_volume": [6.0]})  # fmt: skip
    rows = demo.bar_rows(one, "X", 10.0, "1m").filter(pl.col("kind") == "trades")
    # VWAP 1000 / 10 = 100 for both; 6 bought and 4 sold; round(5 x 6 / 10) = 3 buys, 2 sells
    assert rows.select("side", "qty", "price", "n").rows() == [("S", 4.0, 100.0, 2),
                                                              ("B", 6.0, 100.0, 3)]  # fmt: skip
    # a bar with nothing traded: trade rows of size 0 and count 0 at the close
    idle = one.with_columns(volume=pl.lit(0.0), quote_volume=pl.lit(0.0), count=pl.lit(0),
                            taker_buy_volume=pl.lit(0.0))  # fmt: skip
    rows = demo.bar_rows(idle, "X", 10.0, "1m").filter(pl.col("kind") == "trades")
    assert rows.select("qty", "price", "n").rows() == [(0.0, 101.0, 0), (0.0, 101.0, 0)]


@known
def test_a_half_spread_is_read_back_from_the_rows():
    rows = demo.bar_rows(bars([100.0] * 3), "X", 2.0, "1m").filter(pl.col("kind") == "book")
    got = markouts.half_spread_bps(rows["bid0_px"].to_numpy(), rows["ask0_px"].to_numpy())
    assert got == pytest.approx(2.0, rel=1e-12)


def kline_zip(path, frame, header=True, micro=False):
    out = frame.with_columns(open=pl.col("close"), high=pl.col("close"), low=pl.col("close"),
                             taker_buy_quote_volume=pl.lit(0.0), ignore=pl.lit(0))  # fmt: skip
    if micro:
        out = out.with_columns(pl.col("open_time", "close_time") * 1000)
    buf = io.StringIO()
    out.select(demo.KLINE_COLUMNS).write_csv(buf, include_header=header)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(path.with_suffix(".csv").name, buf.getvalue())
    return path


@edge
def test_headerless_and_microsecond_klines_parse_like_a_headed_one(tmp_path):
    frame = bars([100.5, 101.25, 99.75])
    headed = demo.read_klines(kline_zip(tmp_path / "a.zip", frame))
    found = []
    bare = demo.read_klines(kline_zip(tmp_path / "b.zip", frame, header=False), found)
    assert found == ["no header"]
    found = []
    micro = demo.read_klines(kline_zip(tmp_path / "c.zip", frame, header=False, micro=True),
                             found)  # fmt: skip
    assert found == ["no header", "microseconds"]
    assert headed.equals(frame.select(headed.columns))
    assert bare.equals(headed) and micro.equals(headed)
    assert demo.bar_rows(micro, "X", 1.0, "1m").equals(demo.bar_rows(headed, "X", 1.0, "1m"))


@edge
def test_another_kline_header_fails(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("bad.csv", "open_time,close\n1,2\n")
    with pytest.raises(demo.BuildError, match="the header is 'open_time,close'"):
        demo.read_klines(path)


@edge
def test_a_symbol_month_missing_0_2_percent_of_its_bars_fails():
    june = bars([100.0] * 2880, step_ms=15 * MIN, start=pd.Timestamp("2024-06-01", tz="UTC"))
    start, end = date(2024, 6, 1), date(2024, 7, 1)
    gaps = demo.bar_gaps(june, "15m", start, end)
    assert gaps["2024-06"] == {"expected": 2880, "present": 2880, "missing": 0}
    # six of 2,880 is 0.21%, over the 0.1% limit; two (0.07%) passes
    holed = june.filter(~pl.int_range(pl.len()).is_in([10, 11, 400, 401, 402, 2000]))
    with pytest.raises(demo.BuildError, match=r"XRPUSDT 2024-06: 6 of 2880 bars missing"):
        demo.missing_check("XRPUSDT", demo.bar_gaps(holed, "15m", start, end))
    fewer = june.filter(~pl.int_range(pl.len()).is_in([10, 11]))
    assert demo.missing_check("XRPUSDT", demo.bar_gaps(fewer, "15m", start, end))[0] == "2024-06"


@edge
def test_missing_files_stop_the_build_before_any_download():
    sizes = {"https://x/a.zip": 10, "https://x/b.zip": None, "https://x/c.zip": None}
    with pytest.raises(demo.BuildError, match="not on data.binance.vision: b.zip, c.zip"):
        demo.head_sizes(list(sizes), head=sizes.get)


def bar_file(tmp_path, hs=1.0, days=3):
    rows = demo.bar_rows(bars([100.0 + (i % 7) for i in range(24 * days)], step_ms=60 * MIN),
                         "X", hs, "1h")  # fmt: skip
    path = tmp_path / "data" / "bars.parquet"
    path.parent.mkdir()
    rows.write_parquet(path)
    return path


def bar_brief_file(tmp_path, data, **kw):
    brief = bar_brief(str(data), step="1h", latency_ms=3_600_000, **kw)
    path = tmp_path / "bars.brief.yaml"
    path.write_text(yaml.safe_dump(brief.raw, sort_keys=False))
    return path


@known
def test_init_checks_the_cost_model_against_the_file(tmp_path):
    data = bar_file(tmp_path)
    run = vaults.init(bar_brief_file(tmp_path, data, hs={"X": 1.0}), tmp_path / "ok")
    text = (run.workspace / "DATA.md").read_text()
    assert "## Fills on bars" in text and "The millisecond tables" in text
    assert "## Cross-stream lag" not in text
    rules = (run.workspace / "RULES.md").read_text()
    assert "## Bars" in rules and "There is no book inside a bar." in rules
    # the file was built at 1 bps; a brief that says 1.0000001 is off by 1e-7 relative
    with pytest.raises(ValueError, match="not its brief's half spread of 1 bps"):
        vaults.init(bar_brief_file(tmp_path, data, hs={"X": 1.0000001}), tmp_path / "off")
    assert not (tmp_path / "off").exists()
    # a day inside train: the bars start on 2024-06-03
    with pytest.raises(ValueError, match="2024-06-03 is not before train"):
        vaults.init(bar_brief_file(tmp_path, data, hs={"X": 1.0}, source="2024-06-03"),
                    tmp_path / "late")  # fmt: skip
    assert not (tmp_path / "late").exists()


@edge
def test_a_one_tick_spread_on_a_large_price_passes_the_cost_check():
    # BTC spot on 2024-03-01: one tick of 0.01 on about 61,000 is 0.000809 bps. The doubles of
    # (ask - bid) / (ask + bid) are good to about 1e-12 bps, which is 1.8e-9 of it, so the
    # check allows 1e-10 bps for rounding on top of 1e-9 relative
    rows = demo.bar_rows(bars([61234.56, 61987.65, 60111.11]), "X", 0.000809, "1m")
    start = pd.Timestamp(BAR0)
    worst = vaults.check_cost_model(bar_brief(hs={"X": 0.000809}), rows, start)
    assert worst["X"] < 1e-8
    # a brief at the next four-figure value, 0.0008091, is 1.2e-4 off: refused
    with pytest.raises(ValueError, match="not its brief's half spread of 0.0008091 bps"):
        vaults.check_cost_model(bar_brief(hs={"X": 0.0008091}), rows, start)


@edge
def test_a_cost_model_must_be_a_known_kind_with_positive_spreads():
    raw = bar_brief().raw
    raw["data"]["cost_model"]["kind"] = "vwap"
    with pytest.raises(ValueError, match="cost_model.kind must be one of"):
        briefs.parse(raw)
    raw["data"]["cost_model"] |= {"kind": "half_spread", "half_spread_bps": {"X": 0.0}}
    with pytest.raises(ValueError, match="must be positive"):
        briefs.parse(raw)
    cost = bar_brief().data.cost_model
    assert cost.day == pd.Timestamp("2024-06-01", tz="UTC")
    assert cost.line() == (
        "fills are a model: the next bar's close plus or minus a half "
        "spread of 10 bps per symbol, measured on data.binance.vision "
        "bookTicker 2024-06-01, before train; no book existed at the fill"
    )


@edge
def test_an_hourly_grid_keeps_its_last_bar():
    # three days of hourly bars from 2024-06-03 00:00: the last closes at 23:59:59.999 on the
    # 5th, so the grid runs 00:00 on the 3rd to 23:00 on the 5th, 72 points. np.arange sized
    # in floating point dropped the last: the 1 ns past the end is below a float's precision
    # at 72 x 3.6e12 ns
    rows = demo.bar_rows(bars([100.0] * 72, step_ms=60 * MIN), "X", 1.0, "1h")
    brief = bar_brief(step="1h", latency_ms=3_600_000)
    grid = make_grid(brief, load_events(brief, rows))
    assert len(grid) == 72
    assert pd.Timestamp(int(grid[-1]), tz="UTC") == BAR0 + pd.Timedelta("2D23h")


# -- the XRP perpetual on two venues ---------------------------------------


def bybit_raw(root, cadence_ms=20, seconds=0.14, hours=(0,)):
    """Binance's perp files of the ``raw`` fixture's layout, and Bybit quotes on a beat of
    ``cadence_ms`` for the first ``seconds`` of each of ``hours``, with three trades. Ten
    hand-written Bybit rows when the defaults hold."""
    perp_q = [("binance-futures", "XRPUSDT", T0 + 1000, T0 + 2000, 9.0, 1.4999, 1.4998, 9.0)]
    perp_t = [("binance-futures", "XRPUSDT", T0 + 2000, T0 + 2500, 4, "sell", 1.4998, 10.0)]
    n = round(seconds * 1000 / cadence_ms)
    stamps = [T0 + h * 3_600_000_000 + 1000 * cadence_ms * i for h in hours for i in range(n)]
    bybit_q = [("bybit", "XRPUSDT", t, t + 300, 12.0 + i, 1.5001 + 0.0001 * (i % 2), 1.5000,
                11.0) for i, t in enumerate(stamps)]  # fmt: skip
    root.mkdir(parents=True, exist_ok=True)
    bybit_t = [
        ("bybit", "XRPUSDT", T0 + 25_000, T0 + 25_400, "a", "buy", 1.5001, 3.0),
        ("bybit", "XRPUSDT", T0 + 45_000, T0 + 45_400, "b", "sell", 1.5000, 2.0),
        ("bybit", "XRPUSDT", T0 + 85_000, T0 + 85_400, "c", "buy", 1.5002, 1.0),
    ]
    for exchange, kind, rows in (
        ("binance-futures", "book_ticker", perp_q),
        ("binance-futures", "trades", perp_t),
        ("bybit", "quotes", bybit_q),
        ("bybit", "trades", bybit_t),
    ):
        write(root / demo.raw_name(exchange, kind, DAY, "XRPUSDT"),
              QUOTES if kind != "trades" else TRADES, rows)  # fmt: skip
    return root


LISTED = {
    "datasets": {
        "symbols": [
            {
                "id": "XRPUSDT",
                "type": "perpetual",
                "dataTypes": ["trades", "quotes", "book_ticker"],
                "availableSince": "2021-05-13T00:00:00.000Z",
                "availableTo": "2026-10-18T00:00:00.000Z",
            }
        ]
    }
}


@known
def test_ten_bybit_rows_build_the_tagged_instrument_in_the_tardis_layout(tmp_path):
    raw = bybit_raw(tmp_path / "raw")
    frame = demo.venue_frame(raw, demo.VENUES[0])
    assert tuple(frame.columns) == demo.COLUMNS
    bybit = frame.filter(pl.col("symbol") == "XRPUSDT_PERP_BYBIT")
    book, fills = bybit.filter(~pl.col("is_trade")), bybit.filter(pl.col("is_trade"))
    # seven quotes 20 ms apart, stamped on receive (local) time, 0.3 ms after the exchange's
    assert book.height == 7 and fills.height == 3
    assert book["ts"].dt.epoch("us").to_list() == [T0 + 20_000 * i + 300 for i in range(7)]
    assert book["mid"].to_list()[:2] == pytest.approx([1.50005, 1.50010])
    # the 25 ms buy arrives at 25.4 ms and carries the 20.3 ms quote: ask 1.5002, size 13
    first = fills.row(0, named=True)
    assert (first["side"], first["price"], first["qty"]) == ("B", 1.5001, 3.0)
    assert (first["ask0_px"], first["ask0_qty"]) == (1.5002, 13.0)
    assert frame.filter(pl.col("symbol") == "XRPUSDT_PERP").height == 2


@edge
def test_a_second_venue_on_a_100ms_cadence_is_refused_and_20ms_passes():
    beat = {ms: np.arange(0.0, 60_000.0, ms) for ms in (20, 100)}
    fast = demo.venue_cadence(beat[20])
    assert fast["ok"] and fast["median_gap_ms"] == 20
    slow = demo.venue_cadence(beat[100])
    assert not slow["ok"] and slow["median_gap_ms"] == 100


@known
def test_the_cross_venue_build_writes_the_cadence_and_the_brief(tmp_path):
    raw = bybit_raw(tmp_path / "raw", cadence_ms=20, seconds=180)
    fetched = []
    built = demo.crossvenue(tmp_path / "out", raw, log=lambda s: None,
                            details=lambda ex: LISTED,
                            fetch_day=lambda *a: fetched.append(a))  # fmt: skip
    assert fetched == []  # every file was already in raw
    # every row fits, so the file keeps the 14 hours the rule allows
    assert built.data.name == "tardis-binance-bybit-xrpusdt-perp-2026-10-01-14h.parquet"
    assert built.cadence["median_gap_ms"] == pytest.approx(20.0)
    notes = built.notes.read_text()
    assert "Venue feed cadence: XRPUSDT_PERP_BYBIT's book, median gap 20 ms" in notes
    brief = yaml.safe_load(built.brief.read_text())
    assert brief["archetypes"]["cross_venue"] == [["XRPUSDT_PERP", "XRPUSDT_PERP_BYBIT"]]
    assert brief["target"]["instrument"] == "XRPUSDT_PERP"
    # and DATA.md's venue line, from the panel of the brief
    from blinded.diagnostics import _venues_md

    parsed = briefs.load(built.brief)
    text = "\n".join(_venues_md(parsed, Panel.from_brief(parsed)))
    assert ("Venue feed cadence: XRPUSDT_PERP_BYBIT's book, median gap 20 ms between distinct "
            "receive stamps") in text  # fmt: skip


@edge
def test_a_slow_second_venue_stops_the_build(tmp_path):
    raw = bybit_raw(tmp_path / "raw", cadence_ms=100, seconds=180)
    with pytest.raises(demo.BuildError, match="median gap of 100 ms, at or above 50"):
        demo.crossvenue(tmp_path / "out", raw, log=lambda s: None, details=lambda ex: LISTED,
                        fetch_day=lambda *a: None)  # fmt: skip
    assert not (tmp_path / "out").exists()


@known
def test_the_file_keeps_the_most_whole_hours_that_fit_the_row_budget(tmp_path, monkeypatch):
    # 100 Bybit quotes in each of hours 0, 1 and 2, three Bybit trades and two Binance rows in
    # hour 0: 105 rows in 1 h, 205 in 2 h, 305 in 3 h; a budget of 250 keeps 2 hours
    raw = bybit_raw(tmp_path / "raw", cadence_ms=20, seconds=2, hours=(0, 1, 2))
    monkeypatch.setattr(demo, "CROSS_VENUE_ROWS", 250)
    built = demo.crossvenue(tmp_path / "out", raw, log=lambda s: None, details=lambda ex: LISTED,
                            fetch_day=lambda *a: None)  # fmt: skip
    assert (built.counts[1], built.counts[2], built.counts[3]) == (105, 205, 305)
    assert (built.hours, built.rows) == (2, 205)
    assert built.data.name.endswith("-2h.parquet")


@edge
def test_the_venue_fallbacks_go_in_order_and_say_why(tmp_path):
    no_quotes = {"datasets": {"symbols": [dict(LISTED["datasets"]["symbols"][0],
                                               dataTypes=["trades"])]}}  # fmt: skip
    said = []
    with pytest.raises(demo.BuildError, match="no second venue for the XRP perpetual") as err:
        demo.crossvenue(tmp_path / "out", tmp_path / "raw", log=said.append,
                        details=lambda ex: no_quotes, fetch_day=lambda *a: None)  # fmt: skip
    # Bybit has no quotes on either day; OKX does not list XRPUSDT under that name
    assert said == [
        "passing over bybit XRPUSDT on 2026-10-01: bybit XRPUSDT has no quotes",
        "passing over okex-swap XRP-USDT-SWAP on 2026-10-01: okex-swap does not list "
        "XRP-USDT-SWAP",
        "passing over bybit XRPUSDT on 2026-09-01: bybit XRPUSDT has no quotes",
    ]
    assert "nothing was downloaded" in str(err.value)
    since = "2026-10-05T00:00:00.000Z"
    late = {
        "datasets": {"symbols": [dict(LISTED["datasets"]["symbols"][0], availableSince=since)]}
    }
    assert demo.venue_problem(late, demo.VENUES[0]) == (
        "bybit XRPUSDT covers 2026-10-05 to 2026-10-18, not 2026-10-01"
    )


@cross
def test_touch_prices_read_the_rows_read_archive_does(tmp_path):
    header = demo.HEADERS["bookTicker"]
    body = "1,100.0,2,100.2,3,1000,1001\n2,100.1,2,100.2,3,1500,1501\n"
    path = tmp_path / "X-bookTicker-2024-03-27.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("X-bookTicker-2024-03-27.csv", header + "\n" + body)
    bid, ask = demo.touch_prices(path)
    book = demo.book_frame(demo.read_archive(path, "bookTicker"))
    np.testing.assert_array_equal(bid, book["bid"].to_numpy())
    np.testing.assert_array_equal(ask, book["ask"].to_numpy())
