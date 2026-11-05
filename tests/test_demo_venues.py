"""``blinded demo arb`` and ``blinded demo venues`` on hand-written Tardis files of one coin on
two or three spot venues, stamps in ms after midnight on the receive clock.

The quote venue (``mexc``) is 200 bps wide and the hedge venue (``binance``) 10 bps; every
venue's snapshot shows its touch at level 0, so the snapshot check passes, and each hand-written
number below is read off the rows.
"""

import gzip

import numpy as np
import polars as pl
import pytest
import yaml

from blinded import demo_mm
from blinded.brief import load
from blinded.demo import BuildError, raw_name

known = pytest.mark.known_answer
edge = pytest.mark.edge_case
DAY = "2026-10-01"
T0 = 1_790_812_800_000_000
COIN = "TESTUSDT"
# (ms, bid, ask, bid size, ask size); the exchange's stamp is 5 ms earlier on mexc and okex
BOOKS = {
    "mexc": [(10, 0.9900, 1.0100, 50.0, 40.0), (500, 0.9901, 1.0101, 30.0, 40.0),
             (2000, 0.9902, 1.0102, 20.0, 10.0)],
    "binance": [(5, 0.9995, 1.0005, 300.0, 300.0), (300, 0.9996, 1.0006, 200.0, 100.0),
                (1500, 0.9995, 1.0005, 100.0, 150.0)],
    "okex": [(8, 0.9994, 1.0004, 10.0, 10.0), (900, 0.9993, 1.0003, 12.0, 11.0)],
}  # fmt: skip
TRADES = {"mexc": [(700, "sell", 0.9950, 20.0), (2100, "buy", 1.0050, 15.0)],
          "binance": [(310, "buy", 1.0006, 5.0), (1600, "sell", 0.9995, 7.0)],
          "okex": [(950, "buy", 1.0003, 1.0)]}  # fmt: skip


def write(path, header, rows):
    with gzip.open(path, "wt") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


def raw_files(raw, books=BOOKS, snapshot_shift=0.0):
    raw.mkdir(parents=True, exist_ok=True)
    day = demo_mm.Date.fromisoformat(DAY)
    for ex, rows in books.items():
        sym = demo_mm.venue_symbol(COIN, ex)
        lag = 0 if ex == "binance" else 5000
        write(
            raw / raw_name(ex, "book_ticker", day, sym),
            "exchange,symbol,timestamp,local_timestamp,ask_amount,ask_price,bid_price,bid_amount",
            [
                (ex, sym, T0 + ms * 1000 - lag, T0 + ms * 1000, aq, a, b, bq)
                for ms, b, a, bq, aq in rows
            ],
        )
        write(
            raw / raw_name(ex, "trades", day, sym),
            "exchange,symbol,timestamp,local_timestamp,id,side,price,amount",
            [
                (ex, sym, T0 + ms * 1000 - lag, T0 + ms * 1000, i, s, p, q)
                for i, (ms, s, p, q) in enumerate(TRADES[ex])
            ],
        )
        header = "exchange,symbol,timestamp,local_timestamp," + ",".join(
            f"asks[{k}].price,asks[{k}].amount,bids[{k}].price,bids[{k}].amount" for k in range(25)
        )
        snaps = []
        for ms, b, a, bq, aq in rows:  # one snapshot 2 ms after each touch row, its book
            cells = []
            for k in range(25):
                cells += [
                    round(a + k * 1e-4 + snapshot_shift, 4),
                    aq + k,
                    round(b - k * 1e-4, 4),
                    bq + k,
                ]
            snaps.append([ex, sym, T0 + ms * 1000, T0 + (ms + 2) * 1000, *cells])
        write(raw / raw_name(ex, "book_snapshot_25", day, sym), header, snaps)  # fmt: skip


def listing(ex):
    sym = demo_mm.venue_symbol(COIN, ex)
    return {
        "datasets": {
            "symbols": [
                {
                    "id": sym,
                    "type": "spot",
                    "dataTypes": list(demo_mm.ARB_TYPES),
                    "availableSince": "2025-01-01T00:00:00.000Z",
                }
            ]
        }
    }


def no_fetch(*args):
    raise AssertionError(f"nothing should be fetched: {args}")


def build_arb(tmp_path, **kw):
    raw_files(tmp_path / "raw", **kw)
    return demo_mm.arb(COIN, DAY, tmp_path / "out", raw=tmp_path / "raw", log=lambda m: None,
                       details=listing, fetch_day=no_fetch)  # fmt: skip


def ms(frame):
    return ((frame["ts"].dt.epoch("ns") - T0 * 1000) // 1_000_000).to_list()


@known
def test_the_arb_file_keeps_every_row_of_both_venues(tmp_path):
    b = build_arb(tmp_path)
    data = pl.read_parquet(b.data)
    book = data.filter(pl.col("kind") == "book")
    # every book row of both venues on one clock, file order on a tie
    assert ms(book) == [5, 10, 300, 500, 1500, 2000]
    assert book["symbol"].cast(pl.String).to_list() == [
        "TESTUSDT_BINANCE", "TESTUSDT_MEXC", "TESTUSDT_BINANCE", "TESTUSDT_MEXC",
        "TESTUSDT_BINANCE", "TESTUSDT_MEXC"]  # fmt: skip
    assert book["n"].to_list() == [1] * 6
    # mexc stamps its rows 5 ms before the recorder sees them; binance spot sends none
    assert book["age_ms"].to_list() == [0.0, 5.0, 0.0, 5.0, 0.0, 5.0]
    # the hedge venue's level 1 from its snapshot as-of each of its rows: the snapshot 2 ms
    # after a row is not yet there, so the first row has none and the others the one before
    hedge = book.filter(pl.col("symbol") == "TESTUSDT_BINANCE")
    assert hedge["bid1_px"].to_list() == [None, 0.9994, 0.9995]
    assert book.filter(pl.col("symbol") == "TESTUSDT_MEXC")["bid1_px"].null_count() == 3
    trades = data.filter(pl.col("kind") == "trades")
    assert ms(trades) == [310, 700, 1600, 2100]
    # each with its own venue's touch: binance's 300 ms row, mexc's 500 ms row, ...
    assert trades["bid0_qty"].to_list() == [200.0, 30.0, 100.0, 20.0]
    touch = pl.read_parquet(b.touch)
    assert ms(touch) == [10, 500, 2000] and touch["bid"].to_list() == [9900, 9901, 9902]
    depth = pl.read_parquet(b.depth)
    assert ms(depth) == [7, 302, 1502] and depth.width == 1 + 4 * 25
    assert depth["ask24_px"].to_list() == [1.0029, 1.003, 1.0029]


@known
def test_the_arb_brief_declares_the_touch_the_depth_and_the_stock(tmp_path):
    b = build_arb(tmp_path)
    brief = load(b.brief)
    assert (brief.data.touch.instrument, brief.data.touch.tick_size) == ("TESTUSDT_MEXC", 1e-4)
    d = brief.data.depth
    assert (d.venue, d.levels, d.lag_ms, d.path) == ("TESTUSDT_BINANCE", 25, 100.0, b.depth)
    assert brief.archetypes.cross_venue == (("TESTUSDT_MEXC", "TESTUSDT_BINANCE"),)
    raw = yaml.safe_load(b.brief.read_text())
    stock = raw["stock"]
    assert stock["withdrawal_fee_coin"] is None and stock["rebalance_at"] == 0.8
    # 500 USD of coin at the quote venue's mid, 1.0001 time-weighted: 499 whole coins
    assert stock["TESTUSDT_MEXC"] == {"coin": 499.0, "usdt": 500.0}
    ex = raw["execution"]
    assert (ex["hedge_fee_bps"], ex["maker_fee_bps"], ex["hedge_latency_ms"]) == (10.0, 0.0, 50)
    assert ex["fees"] == {"TESTUSDT_BINANCE": 10.0}
    # the venues' spreads over the day, time-weighted: mexc about 200 bps, binance 10
    assert b.facts["TESTUSDT_MEXC"]["spread_bps"] == pytest.approx(2e4 * 0.02 / 2.0001, rel=1e-3)
    assert b.snapshots["TESTUSDT_MEXC"]["reverse"] == (1.0, 1.0)
    notes = b.notes.read_text()
    assert "## 2026-10-01" in notes and "Who prints on each" in notes


@edge
def test_a_tight_quote_venue_and_a_snapshot_off_its_touch_are_refused(tmp_path):
    tight = dict(BOOKS, mexc=[(10, 0.9998, 1.0002, 5.0, 5.0), (500, 0.9997, 1.0002, 5.0, 5.0)])
    with pytest.raises(BuildError, match="under 3 hedge fees of 10 bps"):
        build_arb(tmp_path / "a", books=tight)
    with pytest.raises(BuildError, match="shows its level 0 prices on 0.0% of them"):
        build_arb(tmp_path / "b", snapshot_shift=0.0003)
    assert not (tmp_path / "a" / "out").exists() and not (tmp_path / "b" / "out").exists()


@known
def test_the_consolidated_file_holds_every_venue_on_one_clock(tmp_path):
    raw_files(tmp_path / "raw")
    legs = (("binance", COIN), ("okex", COIN), ("mexc", COIN))
    b = demo_mm.venues(COIN, DAY, tmp_path / "out", raw=tmp_path / "raw", legs=legs,
                       log=lambda m: None, details=listing, fetch_day=no_fetch)  # fmt: skip
    data = pl.read_parquet(b.data)
    book = data.filter(pl.col("kind") == "book")
    assert ms(book) == [5, 8, 10, 300, 500, 900, 1500, 2000]
    assert book.group_by("symbol").len().sort("symbol")["len"].to_list() == [3, 3, 2]
    assert data.filter(pl.col("kind") == "trades").height == 5
    brief = load(b.brief)
    assert brief.target.instrument == "TESTUSDT_BINANCE"
    assert brief.archetypes.cross_venue == (("TESTUSDT_OKX", "TESTUSDT_BINANCE"),
                                            ("TESTUSDT_MEXC", "TESTUSDT_BINANCE"))  # fmt: skip
    assert "age_ms" in brief.data.streams["book"].fields
    assert brief.data.touch is None and brief.data.thinned is None
    f = b.facts["TESTUSDT_OKX"]
    assert (f["rows"], f["trades"], f["delay_ms"]["median"]) == (2, 1, 5.0)
    assert b.facts["TESTUSDT_BINANCE"]["delay_ms"]["median"] is None
    assert np.isclose(f["usd"], 1.0003)
    assert "The venues over the day:" in b.notes.read_text()
