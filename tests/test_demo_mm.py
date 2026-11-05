"""``blinded demo mm`` on hand-written Tardis files: ten ``book_ticker`` rows, six trades, three
``book_snapshot_5`` rows and two ``derivative_ticker`` rows of one perpetual in the first 300
ms of 2026-10-01, every stamp in ms after midnight on the receive clock.

The lattice is a 100 ms grid read at 0 and at the 50 ms latency, so a read every 50 ms. The
book rows are at 10, 20, 40, 55, 60, 120, 130, 160, 240 and 260 ms; the last row at or before
each read is kept, standing for the rows since the one kept before it: 40 for 10, 20 and 40
(n 3), 60 for 55 and 60 (n 2), 130 for 120 and 130 (n 2), then 160, 240 and 260 alone.
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
T0 = 1_790_812_800_000_000  # 2026-10-01 00:00 UTC in Tardis's microseconds
SYM = "TESTUSDT"
BOOK_MS = [10, 20, 40, 55, 60, 120, 130, 160, 240, 260]
BIDS = [100.00, 100.00, 100.00, 99.99, 99.99, 100.00, 100.01, 100.01, 100.00, 100.00]
BID_QTY = [5.0, 4.0, 3.0, 8.0, 9.0, 2.0, 1.0, 6.0, 7.0, 4.0]
TRADES = [
    (15, "sell", 100.00, 1.0),
    (40, "buy", 100.01, 2.0),
    (55, "sell", 99.99, 0.5),
    (125, "buy", 100.01, 1.5),
    (160, "sell", 100.01, 3.0),
    (250, "sell", 100.00, 1.0),
]
SNAP_MS = [5, 58, 150]


def write(path, header, rows):
    with gzip.open(path, "wt") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


def snapshot_row(ms, bid, ask):
    cells = []
    for k in range(5):
        cells += [round(ask + 0.01 * k, 2), 10.0 + k, round(bid - 0.01 * k, 2), 20.0 + k]
    return ["binance-futures", SYM, T0 + ms * 1000 - 300, T0 + ms * 1000, *cells]


def raw_files(raw, funding_after_ms=300):
    raw.mkdir(parents=True, exist_ok=True)
    name = {t: raw / raw_name("binance-futures", t, demo_mm.Date.fromisoformat(DAY), SYM)
            for t in demo_mm.MM_TYPES}  # fmt: skip
    write(
        name["book_ticker"],
        "exchange,symbol,timestamp,local_timestamp,ask_amount,ask_price,bid_price,bid_amount",
        [
            (
                "binance-futures",
                SYM,
                T0 + ms * 1000 - 2000,
                T0 + ms * 1000,
                3.0 + i,
                round(b + 0.01, 2),
                b,
                q,
            )
            for i, (ms, b, q) in enumerate(zip(BOOK_MS, BIDS, BID_QTY, strict=True))
        ],
    )
    write(
        name["trades"],
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount",
        [
            ("binance-futures", SYM, T0 + ms * 1000 - 1000, T0 + ms * 1000, i, s, p, q)
            for i, (ms, s, p, q) in enumerate(TRADES)
        ],
    )
    # the touch as-of each snapshot's stamp: the first row (5 ms reads it), 55 and 130
    write(
        name["book_snapshot_5"],
        "exchange,symbol,timestamp,local_timestamp,"
        + ",".join(
            f"asks[{k}].price,asks[{k}].amount,bids[{k}].price,bids[{k}].amount" for k in range(5)
        ),
        [
            snapshot_row(5, 100.00, 100.01),
            snapshot_row(58, 99.99, 100.00),
            snapshot_row(150, 100.01, 100.02),
        ],
    )
    write(name["derivative_ticker"], "exchange,symbol,timestamp,local_timestamp,"
          "funding_timestamp,funding_rate,predicted_funding_rate,open_interest,last_price,"
          "index_price,mark_price",
          [("binance-futures", SYM, T0, T0 + funding_after_ms * 1000, T0, 0.0001, "", "", "",
            "", ""),
           ("binance-futures", SYM, T0 + 1_500_000, T0 + 1_500_000, T0 + 8 * 3600 * 10**6,
            0.0002, "", "", "", "", "")])  # fmt: skip
    return name


def listing(types=demo_mm.MM_TYPES, kind="perpetual"):
    return {
        "datasets": {
            "symbols": [
                {
                    "id": SYM,
                    "type": kind,
                    "dataTypes": list(types),
                    "availableSince": "2025-01-01T00:00:00.000Z",
                }
            ]
        }
    }


def no_fetch(*args):
    raise AssertionError(f"nothing should be fetched: {args}")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("mm")
    raw_files(root / "raw")
    logs = []
    b = demo_mm.mm(SYM, DAY, root / "out", raw=root / "raw", log=logs.append,
                   details=lambda ex: listing(), fetch_day=no_fetch)  # fmt: skip
    return b, logs


def ms(frame, col="ts"):
    return ((frame[col].dt.epoch("ns") - T0 * 1000) // 1_000_000).to_list()


@known
def test_the_lattice_keeps_the_last_row_at_or_before_each_read(built):
    b, _ = built
    data = pl.read_parquet(b.data)
    book = data.filter(pl.col("kind") == "book")
    assert ms(book) == [40, 60, 130, 160, 240, 260]
    assert book["n"].to_list() == [3, 2, 2, 1, 1, 1]
    assert book["bid0_px"].to_list() == [100.00, 99.99, 100.01, 100.01, 100.00, 100.00]
    assert book["bid0_qty"].to_list() == [3.0, 9.0, 1.0, 6.0, 7.0, 4.0]
    # levels 1 to 4 from the snapshot as-of each kept stamp: 40 reads the 5 ms snapshot, 60
    # and 130 the 58 ms one, 160 and later the 150 ms one
    assert book["bid1_px"].to_list() == [99.99, 99.98, 99.98, 100.00, 100.00, 100.00]
    assert book["ask4_qty"].to_list() == [14.0] * 6
    assert book["bid2_qty"].to_list() == [22.0] * 6
    assert "bid0_px" in book.columns and "bid5_px" not in book.columns


@known
def test_every_trade_keeps_its_stamp_and_the_touch_as_received(built):
    b, _ = built
    data = pl.read_parquet(b.data)
    trades = data.filter(pl.col("kind") == "trades")
    assert ms(trades) == [15, 40, 55, 125, 160, 250]
    assert trades["side"].cast(pl.String).to_list() == ["S", "B", "S", "B", "S", "S"]
    # the touch at or before each: 10, 40 (the book row on its stamp comes first), 55, 120,
    # 160 and 240 ms
    assert trades["bid0_qty"].to_list() == [5.0, 3.0, 8.0, 2.0, 6.0, 7.0]
    assert trades["is_trade"].all() and trades["n"].to_list() == [1] * 6
    # on a shared stamp the book row comes before the trade
    at40 = data.filter(pl.col("ts").dt.epoch("ns") == T0 * 1000 + 40_000_000)
    assert at40["kind"].cast(pl.String).to_list() == ["book", "trades"]


@known
def test_the_funding_rows_and_the_touch_file(built):
    b, logs = built
    data = pl.read_parquet(b.data)
    funding = data.filter(pl.col("kind") == "funding")
    # the rate of the last row naming each settlement of the day, stamped at the settlement
    assert ms(funding) == [0, 8 * 3_600_000]
    assert funding["rate"].to_list() == [0.0001, 0.0002]
    assert [f["after_ms"] for f in b.funding] == [300.0, 1500.0 - 8 * 3_600_000]
    touch = pl.read_parquet(b.touch)
    assert ms(touch) == BOOK_MS
    assert touch["bid"].to_list() == [round(x * 100) for x in BIDS]
    assert (touch["ask"] - touch["bid"]).to_list() == [1] * 10
    assert touch["bid"].dtype == pl.Int64 and touch["bid_qty"].to_list() == BID_QTY
    assert b.touch_check["stamps_checked"] == 6
    assert any("thinning identity holds" in line for line in logs)


@known
def test_the_brief_declares_the_lattice_the_touch_and_the_funding_delay(built):
    b, _ = built
    brief = load(b.brief)
    assert brief.data.touch.path == b.touch and brief.data.touch.tick_size == 0.01
    assert brief.data.touch.instrument == f"{SYM}_PERP" == brief.target.instrument
    assert [o.value // 10**6 for o in brief.data.thinned.offsets] == [0, 50]
    assert brief.data.delay("funding", f"{SYM}_PERP").value == 10**9
    assert brief.data.delay("book", f"{SYM}_PERP").value == 0
    assert set(brief.data.streams["book"].fields) >= {"bid1_qty", "ask4_px", "n"}
    raw = yaml.safe_load(b.brief.read_text())
    assert (raw["execution"]["maker_fee_bps"], raw["execution"]["taker_fee_bps"]) == (2.0, 5.0)
    assert raw["archetypes"]["params"]["market_making"] == {"size": 1.0, "skew": 0.25,
                                                            "max_inventory": 2.0}  # fmt: skip
    notes = b.notes.read_text()
    assert "## 2026-10-01" in notes and "falls of the best bid" in notes
    # two falls: 100.00 (3 shown at 40 ms) to 99.99 at 55 ms, with 1 sold at 100.00 at 15 ms,
    # inside the 100 ms before 40 ms: partly; 100.01 (6 shown at 160 ms) to 100.00 at 240 ms,
    # with 3 sold at 100.01 at 160 ms: partly
    assert (b.facts["falls"], b.facts["partly"]) == (2, 1.0)


@edge
def test_a_late_funding_rate_a_missing_type_and_a_paid_day_are_refused(tmp_path):
    raw_files(tmp_path / "raw", funding_after_ms=1000)
    with pytest.raises(BuildError, match="last arrived 1,000 ms after"):
        demo_mm.mm(SYM, DAY, tmp_path / "out", raw=tmp_path / "raw", log=lambda m: None,
                   details=lambda ex: listing(), fetch_day=no_fetch)  # fmt: skip
    assert not (tmp_path / "out").exists()
    with pytest.raises(BuildError, match="has no derivative_ticker; nothing was downloaded"):
        demo_mm.mm(
            SYM,
            DAY,
            tmp_path / "out",
            raw=tmp_path / "raw",
            log=lambda m: None,
            details=lambda ex: listing(demo_mm.MM_TYPES[:3]),
            fetch_day=no_fetch,
        )
    with pytest.raises(BuildError, match="is a spot, not a perpetual"):
        demo_mm.mm(
            SYM,
            DAY,
            tmp_path / "out",
            raw=tmp_path / "raw",
            log=lambda m: None,
            details=lambda ex: listing(kind="spot"),
            fetch_day=no_fetch,
        )
    with pytest.raises(BuildError, match="not a free first-of-month day"):
        demo_mm.mm(SYM, "2026-10-02", tmp_path / "out", raw=tmp_path / "raw",
                   log=lambda m: None, details=lambda ex: listing(), fetch_day=no_fetch,
                   api_key="")  # fmt: skip


@edge
def test_a_snapshot_that_is_not_the_touchs_book_is_refused(tmp_path):
    names = raw_files(tmp_path / "raw")
    with gzip.open(names["book_snapshot_5"], "rt") as fh:
        text = fh.read().replace("100.01,", "100.05,")
    with gzip.open(names["book_snapshot_5"], "wt") as fh:
        fh.write(text)
    with pytest.raises(BuildError, match="shows its level 0 prices on"):
        demo_mm.mm(SYM, DAY, tmp_path / "out", raw=tmp_path / "raw", log=lambda m: None,
                   details=lambda ex: listing(), fetch_day=no_fetch)  # fmt: skip


@known
def test_the_bookstats_falls_by_hand():
    from blinded import bookstats

    ts = np.array(BOOK_MS) * 10**6
    bid = np.array([round(x * 100) for x in BIDS])
    sells = [(m, p, q) for m, s, p, q in TRADES if s == "sell"]
    out = bookstats.falls(ts, bid, np.array(BID_QTY), np.array([m * 10**6 for m, _, _ in sells]),
                          np.array([round(p * 100) for _, p, _ in sells]),
                          np.array([q for _, _, q in sells]))  # fmt: skip
    assert out == {"falls": 2, "traded_out": 0.0, "pulled": 0.0, "partly": 1.0}
