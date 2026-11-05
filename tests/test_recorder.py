"""The recorder: MEXC's protobuf decode, the two venues' streams into rows and gap rows, the
Binance local book, the clock, and the writer path against the Tardis AT day."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from conftest import local

from blinded import proto, recorder
from blinded.recorder import HOUR_NS, MS, Binance, LocalBook, Mexc, Recording

known = pytest.mark.known_answer
edge = pytest.mark.edge_case

DAY = date(2026, 10, 21)
T0 = recorder.day_start(DAY) + 10 * HOUR_NS  # 10:00 UTC
T1 = T0 + HOUR_NS


@known
def test_a_hand_encoded_deals_frame_decodes_to_its_fields():
    # written from the wire format by hand: field 1 (deals) length-delimited, one item of
    # price "0.1505" (0a 06), quantity "20" (12 02), tradeType 1 (18 01), time 300 (20 ac 02:
    # 300 is 0b10_0101100, low seven bits first with the continuation bit) and tradeId "a1"
    # (2a 02), 21 bytes; field 2 (eventType) "x"; in the wrapper after channel "c" and
    # sendTime 1000 (30 e8 07) as field 314, whose key 314 << 3 | 2 = 2514 is the varint d2 13
    item = "0a06302e31353035" "12023230" "1801" "20ac02" "2a026131"  # fmt: skip
    deals = "0a15" + item + "120178"
    frame = bytes.fromhex("0a0163" + "30e807" + "d2131a" + deals)
    w = proto.wrapper(frame)
    assert w == {"channel": "c", "sendTime": 1000, "body": "publicAggreDeals",
                 "publicAggreDeals": {"deals": [{"price": "0.1505", "quantity": "20",
                                                 "tradeType": 1, "time": 300, "tradeId": "a1"}],
                                      "eventType": "x"}}  # fmt: skip
    assert proto.encode("PushDataV3ApiWrapper", w) == frame


@edge
def test_an_unknown_field_is_skipped_and_a_cut_frame_refused():
    # field 9 (wire type 0, value 5) is in no schema: skipped, as protobuf does
    data = bytes.fromhex("0a03302e31" + "4805")
    assert proto.decode("PublicAggreBookTickerV3Api", data) == {"bidPrice": "0.1"}
    with pytest.raises(proto.DecodeError):
        proto.decode("PublicAggreBookTickerV3Api", bytes.fromhex("0a05302e31"))
    assert len(proto.schema()) == 21  # the 16 files' messages, items included
    assert proto.COMMIT in (proto.HERE / "__init__.py").read_text()
    assert (proto.HERE / "LICENSE").read_text().lstrip().startswith("Apache License")


def mexc_frame(t_ms, body, values, conn=1):
    """A wrapper frame as MEXC sends it, sent ``t_ms`` after 10:00."""
    return proto.encode("PushDataV3ApiWrapper", {"channel": "ch", "symbol": "ATUSDT",
                                                 "sendTime": (T0 // MS) + t_ms - 5,
                                                 body: values})  # fmt: skip


def ticker(v, bid="0.15000", ask="0.15100", bq="500", aq="400"):
    return {"bidPrice": bid, "bidQuantity": bq, "askPrice": ask, "askQuantity": aq,
            "version": str(v), "lastOrderCreateTime": 1}  # fmt: skip


def deals(*items):
    return {"deals": [{"price": p, "quantity": q, "tradeType": s, "time": (T0 // MS) + tm,
                       "tradeId": i} for p, q, s, tm, i in items], "eventType": "d"}  # fmt: skip


def limit(v, n=3):
    return {"bids": [{"price": f"{0.15 - k * 1e-4:.5f}", "quantity": "10"} for k in range(n)],
            "asks": [{"price": f"{0.151 + k * 1e-4:.5f}", "quantity": "20"} for k in range(n)],
            "eventType": "l", "version": str(v)}  # fmt: skip


def aggre(lo, hi):
    return {"bids": [], "asks": [], "eventType": "a", "fromVersion": str(lo), "toVersion": str(hi)}


# the hand-written stream: 20 MEXC frames and 20 Binance events over 10:00 and 11:00, one gap
# planted in each venue's 10:00 hour (MEXC's depth versions jump from 503 to 507; Binance's
# depth U from 1007 to 1010), the 11:00 hour clean
MEXC = [  # (ms after 10:00, body, values, connection)
    (100, "publicAggreBookTicker", ticker(1001), 1),  # kept
    (110, "publicAggreBookTicker", ticker(1001), 1),  # the heartbeat: same version
    (120, "publicAggreBookTicker", ticker(1001), 1),
    (130, "publicAggreBookTicker", ticker(1002, bid="0.15010"), 1),  # kept
    (200, "publicAggreDeals", deals(("0.15050", "20", 1, 190, "a1"),
                                    ("0.15060", "5", 2, 195, "a2")), 1),  # two trades
    (300, "publicAggreDepths", aggre(501, 501), 1),
    (400, "publicAggreDepths", aggre(502, 503), 1),
    (500, "publicLimitDepths", limit(77), 1),  # kept
    (510, "publicLimitDepths", limit(77), 1),  # same version
    (600, "publicAggreDepths", aggre(507, 508), 1),  # the gap: 504 expected
    (700, "publicAggreBookTicker", ticker(1003, ask="0.15090"), 1),  # kept
    (800, "publicAggreDeals", deals(("0.15060", "5", 2, 195, "a2"),
                                    ("0.15070", "1", 1, 790, "a3")), 2),  # a2 seen: one trade
    (900, "publicAggreBookTicker", ticker(1002, bid="0.15010"), 2),  # older: the other socket
    (1000, "publicAggreDepths", aggre(509, 509), 1),
    (HOUR_NS // MS + 100, "publicAggreBookTicker", ticker(1004, bq="300"), 1),  # kept
    (HOUR_NS // MS + 200, "publicAggreDeals", deals(("0.15080", "64", 2, 3_600_190, "a4")), 1),
    (HOUR_NS // MS + 300, "publicAggreDepths", aggre(510, 512), 1),
    (HOUR_NS // MS + 400, "publicLimitDepths", limit(78, 2), 1),  # kept
    (HOUR_NS // MS + 500, "publicAggreBookTicker", ticker(1004, bq="300"), 1),
    (HOUR_NS // MS + 600, "publicAggreBookTicker", ticker(1005, aq="100"), 1),  # kept
]  # fmt: skip


def bt(u, b="0.1500", B="100", a="0.1502", A="80"):
    return {"stream": "atusdt@bookTicker", "data": {"u": u, "s": "ATUSDT", "b": b, "B": B,
                                                    "a": a, "A": A}}  # fmt: skip


def tr(t_ms, tid, p, q, m):
    return {"stream": "atusdt@trade", "data": {"e": "trade", "E": (T0 // MS) + t_ms - 3,
                                               "s": "ATUSDT", "t": tid, "p": p, "q": q,
                                               "T": (T0 // MS) + t_ms - 4, "m": m}}  # fmt: skip


def dp(t_ms, U, u, b=(), a=()):
    data = {"e": "depthUpdate", "E": (T0 // MS) + t_ms - 2, "s": "ATUSDT", "U": U, "u": u,
            "b": [list(x) for x in b], "a": [list(x) for x in a]}  # fmt: skip
    return {"stream": "atusdt@depth@100ms", "data": data}


SNAPSHOT_1 = (1000, [["0.1500", "100"], ["0.1499", "50"]], [["0.1502", "80"], ["0.1503", "60"]])
SNAPSHOT_2 = (1012, [["0.1498", "70"], ["0.1497", "30"]], [["0.1501", "40"]])
BINANCE = [  # (ms after 10:00, event); "snapshot" is a REST answer, not one of the 20
    (50, ("snapshot", SNAPSHOT_1)),
    (150, bt(2000)),  # kept
    (160, dp(160, 995, 1000, [("0.1500", "90")])),  # older than the snapshot: dropped
    (170, dp(170, 1001, 1003, [("0.1500", "120")])),  # straddles 1001: applied, a row
    (180, tr(180, 5001, "0.1501", "10", False)),  # the buyer took: B
    (190, tr(190, 5002, "0.1500", "3", True)),  # S
    (250, bt(2001, B="120")),  # kept
    (260, bt(2001, B="120")),  # the same update id: dropped
    (270, dp(270, 1004, 1005, (), [("0.1502", "0")])),  # a level removed: a row
    (280, dp(280, 1006, 1006, [("0.1490", "5")])),  # a third bid level: a row
    (300, dp(300, 1006, 1006, [("0.1490", "5")])),  # the other socket's copy: dropped
    (400, dp(400, 1010, 1012, [("0.1499", "1")])),  # the gap: U 1010, 1007 expected
    (450, tr(450, 5003, "0.1500", "7", True)),  # kept
    (500, dp(500, 1013, 1014, [("0.1498", "75")])),  # waits for the new snapshot
    (600, ("snapshot", SNAPSHOT_2)),  # 1010-1012 dropped, 1013-1014 applied: a row
    (700, bt(2005, b="0.1498", B="75", a="0.1501", A="40")),  # kept
    (800, dp(800, 1015, 1015, (), [("0.1502", "9")])),  # a row
    (HOUR_NS // MS + 100, tr(3_600_100, 5004, "0.1501", "64", False)),  # kept
    (HOUR_NS // MS + 200, bt(2010, b="0.1498", B="75", a="0.1501", A="30")),  # kept
    (HOUR_NS // MS + 300, dp(3_600_300, 1016, 1016, (), [("0.1501", "30")])),  # a row
    (HOUR_NS // MS + 400, dp(3_600_400, 1017, 1017, [("0.1497", "30")])),  # unchanged: no row
]  # fmt: skip


@pytest.fixture
def recorded(tmp_path):
    rec = Recording(tmp_path, "ATUSDT", log=lambda m: None, started_ns=T0 - 1, host="test")
    asked = []
    mexc, binance = Mexc(rec, "ATUSDT"), Binance(rec, "ATUSDT", snapshot=lambda: asked.append(1))
    events = [(T0 + ms * MS, "mexc", (body, values, conn)) for ms, body, values, conn in MEXC]
    events += [(T0 + ms * MS, "binance", e) for ms, e in BINANCE]
    for t, venue, e in sorted(events, key=lambda x: x[0]):
        if venue == "mexc":
            body, values, conn = e
            mexc.on_frame(t, mexc_frame((t - T0) // MS, body, values), conn)
        elif isinstance(e, tuple):  # the REST snapshot
            binance.seeded(t, *e[1])
        else:
            binance.on_frame(t, json.dumps(e))
    lines = [rec.close_hour(T1 + HOUR_NS)]
    return rec, tmp_path / DAY.isoformat(), asked, lines


def rows(day_dir, stream):
    return pl.concat([pl.read_parquet(p) for p in
                      sorted((day_dir / "parts").glob(f"*.{stream}.parquet"))]).rows()  # fmt: skip


@known
def test_the_hand_stream_gives_its_rows_to_the_digit(recorded):
    rec, d, asked, _ = recorded
    ms = MS
    send = T0 - 5 * ms
    assert rows(d, "quote_book") == [
        (T0 + 100 * ms, send + 100 * ms, 0.15, 0.151, 500.0, 400.0),
        (T0 + 130 * ms, send + 130 * ms, 0.1501, 0.151, 500.0, 400.0),
        (T0 + 700 * ms, send + 700 * ms, 0.15, 0.1509, 500.0, 400.0),
        (T1 + 100 * ms, send + HOUR_NS + 100 * ms, 0.15, 0.151, 300.0, 400.0),
        (T1 + 600 * ms, send + HOUR_NS + 600 * ms, 0.15, 0.151, 500.0, 100.0),
    ]  # fmt: skip
    # a deals frame's trades share its stamp; the exchange stamp is each deal's own time
    assert rows(d, "quote_trades") == [
        (T0 + 200 * ms, T0 + 190 * ms, "B", 0.1505, 20.0),
        (T0 + 200 * ms, T0 + 195 * ms, "S", 0.1506, 5.0),
        (T0 + 800 * ms, T0 + 790 * ms, "B", 0.1507, 1.0),
        (T1 + 200 * ms, T1 + 190 * ms, "S", 0.1508, 64.0),
    ]  # fmt: skip
    depth = rows(d, "quote_depth")
    assert [r[:6] for r in depth] == [(T0 + 500 * ms, send + 500 * ms, 0.15, 10.0, 0.151, 20.0),
                                      (T1 + 400 * ms, send + HOUR_NS + 400 * ms, 0.15, 10.0,
                                       0.151, 20.0)]  # fmt: skip
    assert depth[1][2 + 4 * 2 :] == (None,) * (4 * (recorder.QUOTE_LEVELS - 2))
    assert rows(d, "hedge_book") == [
        (T0 + 150 * ms, T0 + 150 * ms, 0.15, 0.1502, 100.0, 80.0),
        (T0 + 250 * ms, T0 + 250 * ms, 0.15, 0.1502, 120.0, 80.0),
        (T0 + 700 * ms, T0 + 700 * ms, 0.1498, 0.1501, 75.0, 40.0),
        (T1 + 200 * ms, T1 + 200 * ms, 0.1498, 0.1501, 75.0, 30.0),
    ]  # fmt: skip
    assert rows(d, "hedge_trades") == [
        (T0 + 180 * ms, T0 + 177 * ms, "B", 0.1501, 10.0),
        (T0 + 190 * ms, T0 + 187 * ms, "S", 0.15, 3.0),
        (T0 + 450 * ms, T0 + 447 * ms, "S", 0.15, 7.0),
        (T1 + 100 * ms, T1 + 97 * ms, "B", 0.1501, 64.0),
    ]  # fmt: skip
    # the rebuilt book's levels, bid then ask a level: one row a change of the top 25, none
    # between the gap and the rebuild, the rebuild's own row at the snapshot's arrival
    top = [(r[0], r[1], [x for x in r[2:] if x is not None]) for r in rows(d, "hedge_depth")]
    assert top == [
        (T0 + 170 * ms, T0 + 168 * ms, [0.15, 120.0, 0.1502, 80.0, 0.1499, 50.0, 0.1503, 60.0]),
        (T0 + 270 * ms, T0 + 268 * ms, [0.15, 120.0, 0.1503, 60.0, 0.1499, 50.0]),
        (T0 + 280 * ms, T0 + 278 * ms, [0.15, 120.0, 0.1503, 60.0, 0.1499, 50.0, 0.149, 5.0]),
        (T0 + 600 * ms, T0 + 600 * ms, [0.1498, 75.0, 0.1501, 40.0, 0.1497, 30.0]),
        (T0 + 800 * ms, T0 + 798 * ms, [0.1498, 75.0, 0.1501, 40.0, 0.1497, 30.0, 0.1502, 9.0]),
        (T1 + 300 * ms, T1 + 298 * ms, [0.1498, 75.0, 0.1501, 30.0, 0.1497, 30.0, 0.1502, 9.0]),
    ]  # fmt: skip
    assert asked == [1]  # one new snapshot asked for, at the gap


@known
def test_the_planted_gaps_give_their_gap_rows_and_a_gapped_hour(recorded):
    rec, d, _, lines = recorded
    gaps = [(g["venue"], g["stream"], g["kind"], g["before"], g["after"],
             (g["t_before"] - T0) // MS, (g["t_after"] - T0) // MS)
            for g in recorder.gaps_of(d)]  # fmt: skip
    assert gaps == [
        ("binance", "depth", "sequence", 1006, 1010, 280, 400),
        ("mexc", "aggre.depth", "sequence", 503, 507, 400, 600),
        ("binance", "depth", "resync", 1006, 1014, 280, 600),
    ]
    status = recorder.hour_statuses(d, DAY)
    assert status[T0] == ("gapped", "binance depth resync; binance depth sequence; "
                                    "mexc aggre.depth sequence")  # fmt: skip
    assert status[T1] == ("judged", "")
    assert status[T0 - HOUR_NS] == ("unobserved", "the recorder was not running")
    assert sum(s == "unobserved" for s, _ in status.values()) == 22
    assert [x["rows"] for x in recorder.hours_of(d)] == [
        {"quote_book": 3, "quote_trades": 3, "quote_depth": 1, "hedge_book": 3,
         "hedge_trades": 3, "hedge_depth": 5},
        {"quote_book": 2, "quote_trades": 1, "quote_depth": 1, "hedge_book": 1,
         "hedge_trades": 1, "hedge_depth": 1},
    ]  # fmt: skip
    notes = (d / "DATA-raw.md").read_text()
    assert "| mexc | aggre.depth | sequence | 503 | 507 | 10:00:00.400 | 10:00:00.600 | 0.200 |" \
        in notes  # fmt: skip
    assert notes.count("## 1") == 2 and "Gap rows: none." in notes


@edge
def test_an_hour_the_recorder_began_inside_is_not_judged(tmp_path):
    rec = Recording(tmp_path, "ATUSDT", log=lambda m: None, started_ns=T0 + 5 * MS, host="t")
    m = Mexc(rec, "ATUSDT")
    m.on_frame(T0 + 10 * MS, mexc_frame(10, "publicAggreBookTicker", ticker(1)))
    rec.close_hour(T1)
    status = recorder.hour_statuses(tmp_path / DAY.isoformat(), DAY)
    assert status[T0] == ("unobserved", "the recorder ran only part of the hour")
    assert (
        "Recorded from 10:00:00.005 to 11:00:00.000 only"
        in (tmp_path / DAY.isoformat() / "DATA-raw.md").read_text()
    )


@known
def test_the_local_book_takes_binances_rule():
    book = LocalBook(levels=2)
    assert book.event(5, 6, [], []) == "waiting"
    book.snapshot(10, [["1.0", "5"]], [["1.1", "4"]])
    assert book.event(8, 10, [], []) == "dropped"  # at or under lastUpdateId
    assert book.event(12, 13, [], []) == "gap"  # does not straddle 11
    assert book.event(9, 12, [["0.9", "2"], ["0.8", "1"]], []) == "applied"
    assert book.top_levels() == (((1.0, 5.0), (0.9, 2.0)), ((1.1, 4.0),))
    assert book.event(13, 13, [["1.0", "0"]], []) == "applied"  # removed
    assert book.top_levels()[0] == ((0.9, 2.0), (0.8, 1.0))
    assert book.event(12, 14, [], [["1.1", "3"]]) == "applied"  # a straddling copy
    assert book.event(16, 16, [], []) == "gap"


@known
def test_the_clock_offset_is_ntps_and_a_slow_clock_is_refused():
    # sent at 1,000 ms, back at 1,360 ms, the server read 1,222: the midpoint is 1,180, so
    # this clock is 42 ms behind the server's
    s = recorder.clock_sample("mexc", 1_000 * MS, 1_360 * MS, 1_222)
    assert (s["rtt_ms"], s["offset_ms"]) == (360.0, 42.0)
    samples = [recorder.clock_sample("mexc", k * MS, (k + 300 + k % 7) * MS, k + 150 + 40)
               for k in range(0, 8000, 1000)]  # fmt: skip
    summary = recorder.clock_summary(samples)["mexc"]
    assert summary["n"] == 8 and summary["rtt_p90"] == pytest.approx(305.3)
    recorder.check_clock({"mexc": summary})
    late = recorder.clock_summary([recorder.clock_sample("binance", 0, 100 * MS, 301)])
    with pytest.raises(recorder.ClockError, match=r"^the recorder's clock is \+251 ms off "
                       r"binance; fix the clock, the stamps are never corrected$"):  # fmt: skip
        recorder.check_clock(late)


@known
def test_the_forward_latencies_are_each_venues_p90_round_trip_rounded_up(tmp_path):
    lines = [recorder.clock_sample("mexc", 0, int((330 + 10 * k) * MS), 165) for k in range(8)]
    lines += [recorder.clock_sample("binance", 0, int((270 + 30 * k) * MS), 135) for k in range(8)]
    (tmp_path / "clock.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    # p90 of 330..400 is 393 ms, of 270..480 is 459: rounded up to the 100 ms grid
    assert recorder.day_latencies(tmp_path) == {
        "latency_ms": 400, "cancel_latency_ms": 400, "response_latency_ms": 400,
        "hedge_latency_ms": 500}  # fmt: skip


SF = local("data/tardis")


@known
@pytest.mark.skipif(not (SF / "arb/atusdt").is_dir(), reason="the Tardis AT day is not here")
def test_the_tardis_at_day_through_the_writer_is_the_stored_file_byte_for_byte(tmp_path):
    from blinded import demo_mm
    from blinded.demo import raw_name

    day = date(2026, 10, 1)
    start = recorder.day_start(day)
    within = (pl.col("ts") >= start) & (pl.col("ts") < start + recorder.DAY_NS)

    def raw(ex, kind):
        return SF / "raw" / raw_name(ex, kind, day, "ATUSDT")

    frames = {
        "quote_book": demo_mm.read_ticker(raw("mexc", "book_ticker")),
        "quote_trades": demo_mm.read_trades(raw("mexc", "trades")),
        "hedge_book": demo_mm.read_ticker(raw("binance", "book_ticker")),
        "hedge_trades": demo_mm.read_trades(raw("binance", "trades")),
        "hedge_depth": demo_mm.read_snapshot(raw("binance", "book_snapshot_25"), 25),
    }
    frames = {s: f.filter(within).select(list(recorder.STREAMS[s])) for s, f in frames.items()}
    names = list(frames)
    order = pl.concat([pl.DataFrame({"ts": f["ts"], "k": [k] * f.height,
                                     "i": range(f.height)}) for k, f in enumerate(frames.values())]
                      ).sort("ts", maintain_order=True)  # fmt: skip
    table = {s: f.rows() for s, f in frames.items()}
    rec = Recording(tmp_path, "ATUSDT", log=lambda m: None, started_ns=start, host="replay")
    for k, i in zip(order["k"].to_list(), order["i"].to_list(), strict=True):
        rec.add(names[k], table[names[k]][i])
    rec.close_hour(start + recorder.DAY_NS)
    built = rec.build_day(day, latencies={"latency_ms": 50})
    stored = SF / "arb/atusdt/tardis-arb-atusdt-mexc-binance-2026-10-01"
    for kind, suffix in (("data", ""), ("touch", ".touch"), ("depth", ".depth")):
        assert built["paths"][kind].read_bytes() == Path(f"{stored}{suffix}.parquet").read_bytes()
    assert built["rows"]["data"] == 145_157
    # the parts stay: they hold the quote venue's own depth
    assert list((tmp_path / day.isoformat() / "parts").glob("*.parquet"))


def test_a_wait_ends_by_the_wall_clock_when_the_machine_slept(tmp_path, monkeypatch):
    # the event loop's clock stops while the Mac sleeps: a wait taken on it in one piece ended
    # late by every hour slept (the 21 Oct recording ran past its 24 hours and missed its day
    # build); taken a check at a time against the wall clock, it ends when the wall clock says
    import asyncio
    import time as _time

    rec = recorder.Recorder(tmp_path, hours=1)
    monkeypatch.setattr(recorder, "SLEEP_CHECK_S", 0.01)
    wall = [_time.time_ns()]
    monkeypatch.setattr(recorder.time, "time_ns", lambda: wall[0])

    async def slept():
        target = wall[0] + recorder.HOUR_NS
        waiting = asyncio.ensure_future(rec._sleep_until(target))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        wall[0] = target + 1  # the machine wakes an hour later; the loop's clock saw 50 ms
        await asyncio.wait_for(waiting, timeout=2.0)

    asyncio.run(slept())
