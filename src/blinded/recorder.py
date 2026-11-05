"""The recorder: one coin's public market data on MEXC and Binance spot, on one clock.

``blinded record`` keeps one websocket connection a venue:
MEXC's ``aggre.bookTicker@10ms``, ``aggre.deals@10ms``, ``aggre.depth@10ms`` and
``limit.depth@20``, protobuf decoded by :mod:`blinded.proto`; Binance's ``bookTicker``,
``trade`` and ``depth@100ms`` from the market-data-only host, the depth rebuilt into a
local book. Public endpoints only: no key, no account, no order.

Every row is stamped with ``time.time_ns()`` on the raw frame, before it is parsed, and
the stamp is never corrected. A constant offset of this clock moves every stamp alike,
so the order of rows across venues, the fill engine and every as-of read are untouched;
only ``age_ms``, the receive stamp less the exchange's, carries the offset. So the
recorder measures the offset against both venues' ``GET /api/v3/time`` at start and every
hour (server time less the midpoint of the round trip, eight samples a venue), refuses to
start when the median is over :data:`CLOCK_LIMIT_MS` against either venue, and writes
every sample down.

What is kept, and why. MEXC pushes its book ticker every 10 ms whether or not the book
changed, so a book row is kept only when ``version`` moves, which is the cadence of
Tardis's ``book_ticker``. A deals frame of ``k`` items gives ``k`` trades on the frame's
stamp, in frame order. Binance's ``bookTicker`` carries no exchange stamp and every one is
kept (its ``age_ms`` is 0, as in the Tardis file); its trades are kept with ``E``. Its
depth stream is applied to a book seeded from ``GET /api/v3/depth`` by Binance's rule
(drop events with ``u`` at or under the snapshot's ``lastUpdateId``, the first applied
event must straddle it, then each ``U`` is the previous ``u + 1``), and a row of the top 25
levels is kept whenever they change. MEXC's ``limit.depth@20`` rows feed only the hour's
snapshot check in ``DATA-raw.md``: no sim reads the quote venue's depth.

Gaps. A sequence break (MEXC's depth ``fromVersion`` against the last ``toVersion``,
Binance's trade ids and depth ``U``), a silence of MEXC's 100-a-second book ticker
over :data:`HEARTBEAT_S`, and a connection lost are each written as a gap row: the venue,
the stream, the sequence values and the receive stamps either side. A Binance depth gap
rebuilds the book from a new snapshot and no depth row is written until it is rebuilt.
Connections are replaced before they are dropped (make before break) every
:data:`ROTATE_S`, under both venues' 24-hour limit, and a close is never waited on; with
two connections open every stream drops what it has already seen, by its sequence field.

The writer path (:class:`Recording`): rows are buffered by stream and written each hour
as part files, ``parts/HH.N.<stream>.parquet``, with the hour's section of ``DATA-raw.md``
(rows, gap rows, the clock samples, the Tardis-style checks over the hour and each part's
SHA-256) and one line of ``hours.jsonl``. At 00:05 UTC the day's parts become the three
files ``blinded demo arb`` writes, through the same code
(:func:`blinded.demo_mm.arb_frames`), with its brief and the day's section; the parts are
then deleted.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import heapq
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from array import array
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path

import numpy as np
import polars as pl

from blinded import demo_mm, proto

HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
MS = 1_000_000
HEARTBEAT_S = 2.0  # MEXC's book ticker comes 100 times a second: 2 s of silence is a dead socket
PING_S = 20.0  # MEXC drops a connection after a minute without data unless pinged
ROTATE_S = 12 * 3600.0  # both venues end a connection after 24 hours
CLOCK_LIMIT_MS = 250.0
CLOCK_SAMPLES = 8
BUILD_AFTER_NS = 5 * 60 * 10**9  # the day is built at 00:05 UTC
SLEEP_CHECK_S = 60.0  # the longest wait taken on the loop's clock (Recorder._sleep_until)
GRID_MS = 100  # the forward brief's latencies are rounded up to the grid step
LEVELS = demo_mm.DEPTH_LEVELS
QUOTE_LEVELS = 20

MEXC_WS = "wss://wbs-api.mexc.com/ws"
MEXC_REST = "https://api.mexc.com"
BINANCE_WS = "wss://data-stream.binance.vision/stream?streams="
BINANCE_REST = "https://data-api.binance.vision"


def _levels(levels: int) -> list[str]:
    return [f"{s}{k}_{f}" for k in range(levels) for s in ("bid", "ask") for f in ("px", "qty")]


_BOOK = {"ts": pl.Int64, "exch": pl.Int64, "bid": pl.Float64, "ask": pl.Float64,
         "bid_qty": pl.Float64, "ask_qty": pl.Float64}  # fmt: skip
_TRADES = {"ts": pl.Int64, "exch": pl.Int64, "side": pl.String, "price": pl.Float64,
           "qty": pl.Float64}  # fmt: skip
# the stream a row belongs to, and its columns: the readers' of blinded.demo_mm, so the day's
# files are built by the same code from a Tardis day or from a recorded one
STREAMS = {
    "quote_book": _BOOK,
    "quote_trades": _TRADES,
    "quote_depth": {"ts": pl.Int64, "exch": pl.Int64,
                    **{c: pl.Float64 for c in _levels(QUOTE_LEVELS)}},
    "hedge_book": _BOOK,
    "hedge_trades": _TRADES,
    "hedge_depth": {"ts": pl.Int64, "exch": pl.Int64, **{c: pl.Float64 for c in _levels(LEVELS)}},
}  # fmt: skip
NAMES = {"quote_book": "book ticker, kept when version moves", "quote_trades": "deals",
         "quote_depth": "limit.depth@20, kept when version moves (the check only)",
         "hedge_book": "bookTicker", "hedge_trades": "trade",
         "hedge_depth": "depth@100ms rebuilt, kept when the top 25 change"}  # fmt: skip


class Columns:
    """One stream's rows of an hour, a typed array a column: about 8 bytes a number, where a
    tuple of Python numbers costs 30. A missing float is held as NaN and written as null,
    which is how a Tardis file's missing levels read."""

    def __init__(self, schema: dict):
        self.names = list(schema)
        self.kinds = [schema[c] for c in self.names]
        self.cols = [array("q") if k == pl.Int64 else array("d") if k == pl.Float64 else []
                     for k in self.kinds]  # fmt: skip
        self.n = 0

    def append(self, row) -> None:
        for col, kind, x in zip(self.cols, self.kinds, row, strict=True):
            col.append(math.nan if x is None and kind == pl.Float64 else x)
        self.n += 1

    def __len__(self) -> int:
        return self.n

    def frame(self) -> pl.DataFrame:
        out = {}
        for name, kind, col in zip(self.names, self.kinds, self.cols, strict=True):
            if kind == pl.Float64:
                out[name] = pl.Series(
                    name,
                    np.frombuffer(col, dtype=np.float64) if len(col) else np.zeros(0),
                    dtype=pl.Float64,
                    nan_to_null=True,
                )
            elif kind == pl.Int64:
                out[name] = pl.Series(
                    name,
                    np.frombuffer(col, dtype=np.int64)
                    if len(col)
                    else np.zeros(0, dtype=np.int64),
                    dtype=pl.Int64,
                )
            else:
                out[name] = pl.Series(name, col, dtype=kind)
        return pl.DataFrame(out)


def _buffers() -> dict[str, Columns]:
    return {s: Columns(schema) for s, schema in STREAMS.items()}


def iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, UTC).isoformat(timespec="milliseconds")


def hour_of(ns: int) -> int:
    return ns // HOUR_NS * HOUR_NS


def day_of(ns: int) -> Date:
    return datetime.fromtimestamp(ns // DAY_NS * 86_400, UTC).date()


def day_start(day: Date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * 10**9


def stem(coin: str, day: Date, quote: str = "mexc", hedge: str = "binance") -> str:
    return f"live-arb-{coin.lower()}-{quote}-{hedge}-{day.isoformat()}"


def _sha(path: Path) -> str:
    return demo_mm.file_sha256(path)


@dataclass
class Gap:
    """A stretch a stream cannot vouch for: its venue and stream, what was seen (``kind``:
    ``sequence``, ``silent``, ``reconnect`` or ``resync``), the sequence values either side
    and the receive stamps either side, in ns."""

    venue: str
    stream: str
    kind: str
    before: object
    after: object
    t_before: int
    t_after: int

    def row(self) -> dict:
        return {"venue": self.venue, "stream": self.stream, "kind": self.kind,
                "before": self.before, "after": self.after, "t_before": int(self.t_before),
                "t_after": int(self.t_after),
                "seconds": round((self.t_after - self.t_before) / 1e9, 3)}  # fmt: skip


# -- the writer path ---------------------------------------------------------------------


class Recording:
    """One coin's recording under ``root``: a directory a UTC day, rows buffered by stream
    and written an hour at a time (the writer path; the module's docstring).

    ``add`` takes a row in its stream's column order (:data:`STREAMS`); the row goes to the
    hour of its own stamp. ``close_hour`` writes the open hour's parts, its section of
    ``DATA-raw.md`` and its line of ``hours.jsonl``; ``build_day`` turns a day's parts into
    the three files. ``started_ns`` is when the recorder began; an hour it began inside is
    marked as such, and the forward judge does not score it.
    """

    def __init__(self, root: str | Path, coin: str = "ATUSDT", quote: str = "mexc",
                 hedge: str = "binance", log: Callable[[str], None] = print,
                 started_ns: int | None = None, host: str | None = None):  # fmt: skip
        self.root = Path(root).expanduser()
        self.coin, self.quote, self.hedge = coin.upper(), quote, hedge
        self.log = log
        self.started_ns = started_ns
        self.host = host or os.uname().nodename
        self.hour: int | None = None
        self.buf: dict[str, Columns] = _buffers()
        self.frames: dict[str, int] = {}
        self.gaps: list[Gap] = []
        self.clock: list[dict] = []
        self.dropped = 0  # rows stamped before the open hour (the clock stepped back)
        self.totals: dict[str, int] = dict.fromkeys(STREAMS, 0)
        # what happens to an hour once it is taken: written here, or handed to a thread
        self.on_hour: Callable[[dict], object] = self.write_hour

    # where things go
    def day_dir(self, day: Date) -> Path:
        return self.root / day.isoformat()

    def _open(self, ns: int) -> None:
        d = self.day_dir(day_of(ns))
        (d / "parts").mkdir(parents=True, exist_ok=True)
        notes = d / "DATA-raw.md"
        if not notes.exists():
            notes.write_text(head_notes(self.coin, day_of(ns), self.quote, self.hedge, self.host))

    def add(self, stream: str, row: tuple) -> None:
        """One row of ``stream``; the hour it falls in is opened, the one before closed."""
        h = hour_of(int(row[0]))
        if self.hour is None:
            self.hour = h
            self._open(h)
        elif h > self.hour:
            taken = self.take(h)
            if taken is not None:
                self.on_hour(taken)
        elif h < self.hour:
            self.dropped += 1
            return
        self.buf[stream].append(row)

    def frame(self, channel: str, n: int = 1) -> None:
        self.frames[channel] = self.frames.get(channel, 0) + n

    def gap(self, g: Gap) -> None:
        self.gaps.append(g)
        self.log(f"gap: {g.venue} {g.stream} {g.kind} {g.before} -> {g.after}, "
                 f"{(g.t_after - g.t_before) / 1e9:.2f} s")  # fmt: skip
        d = self.day_dir(day_of(g.t_after))
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "gaps.jsonl", "a") as fh:
            fh.write(json.dumps(g.row()) + "\n")

    def clock_samples(self, samples: list[dict]) -> None:
        self.clock += samples
        if not samples:
            return
        d = self.day_dir(day_of(samples[-1]["t1"]))
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "clock.jsonl", "a") as fh:
            for s in samples:
                fh.write(json.dumps(s) + "\n")

    def take(self, until: int | None = None) -> dict | None:
        """Swap out the open hour's rows, frames, gaps and clock samples, for
        :meth:`write_hour`; ``until`` is the next hour's start, None at a stop."""
        if self.hour is None:
            return None
        out = {"hour": self.hour, "rows": self.buf, "frames": self.frames, "gaps": self.gaps,
               "clock": self.clock, "dropped": self.dropped, "until": until,
               "started_ns": self.started_ns}  # fmt: skip
        self.buf = _buffers()
        self.frames, self.gaps, self.clock, self.dropped = {}, [], [], 0
        self.hour = until
        if until is not None:
            self._open(until)
        return out

    def close_hour(self, until: int | None = None) -> dict | None:
        """Write the open hour (:meth:`take` then :meth:`write_hour`) and open ``until``."""
        taken = self.take(until)
        return None if taken is None else self.write_hour(taken)

    def write_hour(self, taken: dict) -> dict:
        """The hour's parts, its ``DATA-raw.md`` section and its ``hours.jsonl`` line."""
        h = taken["hour"]
        d = self.day_dir(day_of(h))
        parts = d / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        hh = datetime.fromtimestamp(h / 1e9, UTC).strftime("%H")
        n = 0
        while any(parts.glob(f"{hh}.{n}.*.parquet")):
            n += 1
        written, counts = {}, {}
        for stream in STREAMS:
            rows = taken["rows"][stream]
            counts[stream] = len(rows)
            self.totals[stream] += len(rows)
            frame = rows.frame()
            path = parts / f"{hh}.{n}.{stream}.parquet"
            frame.write_parquet(path)
            written[stream] = {"file": path.name, "bytes": path.stat().st_size,
                               "sha256": _sha(path)}  # fmt: skip
        start = taken["started_ns"]
        end = taken["until"] if taken["until"] is not None else time.time_ns()
        line = {"hour": iso(h), "hour_ns": h, "part": n,
                "from_ns": max(h, start) if start is not None else h,
                "to_ns": min(end, h + HOUR_NS), "rows": counts, "frames": taken["frames"],
                "gaps": [g.row() for g in taken["gaps"]], "dropped": taken["dropped"],
                "clock": clock_summary(taken["clock"]), "parts": written,
                "host": self.host}  # fmt: skip
        with open(d / "hours.jsonl", "a") as fh:
            fh.write(json.dumps(line) + "\n")
        try:
            checks = hour_checks(taken["rows"], self.coin, self.quote, self.hedge, h,
                                 min(end, h + HOUR_NS))  # fmt: skip
        except Exception as exc:  # the notes never stop the recording
            checks = [f"The checks over the hour were not computed: {type(exc).__name__}: {exc}"]
        with open(d / "DATA-raw.md", "a") as fh:
            fh.write(hour_notes(line, taken["clock"], checks, self.quote, self.hedge))
        self.log(f"hour {iso(h)} written: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
        return line

    def build_day(self, day: Date, latencies: dict | None = None,
                  keep_parts: bool = True) -> dict:  # fmt: skip
        """The day's three files, its brief and its ``DATA-raw.md`` section from its parts.
        The parts are kept unless ``keep_parts`` is False: they hold the quote venue's own
        depth, which no built file does, and a day of them is a few MB (2.5 MB for 22 Oct's
        first 15 hours)."""
        d = self.day_dir(day)
        streams = read_parts(d)
        built = build_files(streams, self.coin, day, d, self.quote, self.hedge,
                            latencies or day_latencies(d))  # fmt: skip
        with open(d / "DATA-raw.md", "a") as fh:
            fh.write(day_notes(built))
        if not keep_parts:
            for p in (d / "parts").glob("*.parquet"):
                p.unlink()
        self.log(f"day {day} built: {built['rows']['data']:,} rows, {built['rows']['touch']:,} "
                 f"touch rows, {built['rows']['depth']:,} depth rows")  # fmt: skip
        return built


def read_parts(day_dir: Path) -> dict[str, pl.DataFrame]:
    """Each stream's rows of a day from its part files, hour by hour, in stamp order (a
    stable sort: a sorted day comes back as it went in)."""
    parts = Path(day_dir) / "parts"
    out = {}
    for stream, schema in STREAMS.items():
        files = sorted(parts.glob(f"*.{stream}.parquet"),
                       key=lambda p: tuple(int(x) for x in p.name.split(".")[:2]))  # fmt: skip
        frames = [pl.read_parquet(p) for p in files]
        frame = pl.concat(frames) if frames else pl.DataFrame(schema=schema)
        out[stream] = frame.sort("ts", maintain_order=True)
    return out


def _legs(streams: dict[str, pl.DataFrame], coin: str, quote: str, hedge: str, start: int,
          end: int) -> tuple[demo_mm.Leg, demo_mm.Leg]:  # fmt: skip
    """The two venues' legs as :func:`blinded.demo_mm.read_leg` makes them from Tardis files:
    the touch two-sided rows only, both inside ``[start, end)``."""
    within = (pl.col("ts") >= start) & (pl.col("ts") < end)
    legs = []
    for venue, side in ((quote, "quote"), (hedge, "hedge")):
        book = streams[f"{side}_book"].filter(within)
        bid, ask = pl.col("bid"), pl.col("ask")
        ticker = book.filter((bid > 0) & (ask > bid))
        trades = streams[f"{side}_trades"].filter(within)
        tick, lot = demo_mm.SPOT_STEPS.get((venue, coin), (None, None))
        if tick is None:
            from blinded.sim_maker import infer_tick

            tick = infer_tick(ticker["bid"].to_numpy(), ticker["ask"].to_numpy())
        legs.append(demo_mm.Leg(venue, demo_mm.instrument(coin, venue), tick, lot, ticker,
                                trades))  # fmt: skip
    return legs[0], legs[1]


def build_files(streams: dict[str, pl.DataFrame], coin: str, day: Date, out: Path,
                quote: str = "mexc", hedge: str = "binance", latencies: dict | None = None,
                brief: bool = True) -> dict:  # fmt: skip
    """Write the day's data, touch and depth files (and its brief) from the day's rows of each
    stream: the layout of ``blinded demo arb``, by :func:`blinded.demo_mm.arb_frames`."""
    start = day_start(day)
    end = start + DAY_NS
    a, b = _legs(streams, coin, quote, hedge, start, end)
    within = (pl.col("ts") >= start) & (pl.col("ts") < end)
    snapshot = streams["hedge_depth"].filter(within)
    frame, touch, depth = demo_mm.arb_frames(a, b, snapshot)
    s = stem(coin, day, quote, hedge)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"data": out / f"{s}.parquet", "touch": out / f"{s}.touch.parquet",
             "depth": out / f"{s}.depth.parquet", "brief": out / f"{s}.brief.yaml"}  # fmt: skip
    frame.write_parquet(paths["data"])
    touch.write_parquet(paths["touch"])
    depth.write_parquet(paths["depth"])
    built = {"day": day, "coin": coin, "legs": (a, b), "paths": paths,
             "rows": {"data": frame.height, "touch": touch.height, "depth": depth.height,
                      f"{a.instrument} book": a.ticker.height,
                      f"{b.instrument} book": b.ticker.height,
                      f"{a.instrument} trades": a.trades.height,
                      f"{b.instrument} trades": b.trades.height},
             "streams": streams, "latencies": latencies}  # fmt: skip
    if brief and a.ticker.height:
        import yaml

        names = demo_mm.BuiltVenueFile(paths["data"], paths["brief"], out / "DATA-raw.md", coin,
                                       day, [a, b], touch=paths["touch"],
                                       depth=paths["depth"])  # fmt: skip
        mid = float(np.median((a.ticker["bid"] + a.ticker["ask"]).to_numpy() / 2))
        raw = demo_mm.arb_brief(names, a, b, mid, demo_mm.HEDGE_FEE_BPS,
                                latencies or {"latency_ms": 50})  # fmt: skip
        raw["name"] = s
        paths["brief"].write_text(yaml.safe_dump(raw, sort_keys=False))
    return built


# -- the clock -------------------------------------------------------------------------


def clock_sample(venue: str, t0: int, t1: int, server_ms: int) -> dict:
    """One reading, NTP-style: the offset is the server's time less the midpoint of the round
    trip, in ms; positive when this clock is behind."""
    return {"venue": venue, "t0": int(t0), "t1": int(t1), "server_ms": int(server_ms),
            "rtt_ms": (t1 - t0) / MS, "offset_ms": server_ms - (t0 + t1) / 2 / MS}  # fmt: skip


def clock_summary(samples: list[dict]) -> dict:
    """Each venue's median offset, its range, and the round trip's median and p90, in ms."""
    out = {}
    for venue in sorted({s["venue"] for s in samples}):
        off = np.array([s["offset_ms"] for s in samples if s["venue"] == venue])
        rtt = np.array([s["rtt_ms"] for s in samples if s["venue"] == venue])
        out[venue] = {"n": int(len(off)), "offset_ms": float(np.median(off)),
                      "offset_min": float(off.min()), "offset_max": float(off.max()),
                      "rtt_ms": float(np.median(rtt)),
                      "rtt_p90": float(np.percentile(rtt, 90))}  # fmt: skip
    return out


class ClockError(RuntimeError):
    """The recorder's clock is too far from a venue's to start."""


def check_clock(summary: dict, limit_ms: float = CLOCK_LIMIT_MS) -> None:
    """Refuse a start when either venue's median offset is over ``limit_ms``."""
    for venue, s in summary.items():
        if abs(s["offset_ms"]) > limit_ms:
            raise ClockError(f"the recorder's clock is {s['offset_ms']:+.0f} ms off {venue}; "
                             "fix the clock, the stamps are never corrected")  # fmt: skip


def day_latencies(day_dir: Path, quote: str = "mexc", hedge: str = "binance") -> dict:
    """The forward brief's latencies from a day's clock samples:
    the p90 round trip to the quote venue for the order, cancel and response latencies, the
    p90 to the hedge venue for the hedge's, each rounded up to the grid step."""
    path = Path(day_dir) / "clock.jsonl"
    samples = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def p90(venue):
        rtt = [s["rtt_ms"] for s in samples if s["venue"] == venue]
        if not rtt:
            return None
        return int(math.ceil(float(np.percentile(rtt, 90)) / GRID_MS) * GRID_MS)

    q, h = p90(quote), p90(hedge)
    if q is None or h is None:
        return {"latency_ms": 50}
    return {"latency_ms": q, "cancel_latency_ms": q, "response_latency_ms": q,
            "hedge_latency_ms": h}  # fmt: skip


# -- the notes --------------------------------------------------------------------------


def head_notes(coin: str, day: Date, quote: str, hedge: str, host: str) -> str:
    s = stem(coin, day, quote, hedge)
    return "\n".join([
        f"# The recorded rows behind {s}.parquet",
        "",
        f"Recorded by `blinded record` on {host}: {coin} spot on `{quote}` (instrument "
        f"{demo_mm.instrument(coin, quote)}, the quote venue) and `{hedge}` "
        f"({demo_mm.instrument(coin, hedge)}, the hedge venue), from both venues' public "
        "websocket streams, every stamp this host's receive clock, `time.time_ns()` on the "
        "raw frame before it is parsed, never corrected. MEXC's book ticker is kept when its "
        "`version` moves, every deal at its frame's stamp; Binance's `bookTicker` and "
        "`trade` as sent, its depth rebuilt from `depth@100ms` on a REST snapshot, a row "
        "whenever the top 25 levels change. Each hour below is the rows of that hour, its gap "
        "rows, the clock against both venues' `/api/v3/time` and the checks of "
        "`blinded demo arb`; the day's files are built at 00:05 UTC.",
        "",
    ]) + "\n"  # fmt: skip


def _t(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, UTC).strftime("%H:%M:%S.%f")[:-3]


def hour_notes(line: dict, samples: list[dict], checks: list[str], quote: str,
               hedge: str) -> str:  # fmt: skip
    h = line["hour_ns"]
    lines = [f"## {_t(h)[:5]} to {_t(h + HOUR_NS)[:5]} UTC", ""]
    whole = line["from_ns"] == h and line["to_ns"] == h + HOUR_NS
    lines.append("Recorded over the whole hour." if whole else
                 f"Recorded from {_t(line['from_ns'])} to {_t(line['to_ns'])} only: the "
                 "recorder was not running the whole hour, so it is not judged.")  # fmt: skip
    lines += ["", "| stream | rows | part | bytes | sha256 |", "|---|---|---|---|---|"]
    for stream, p in line["parts"].items():
        venue = quote if stream.startswith("quote") else hedge
        lines.append(f"| {venue} {NAMES[stream]} | {line['rows'][stream]:,} | {p['file']} | "
                     f"{p['bytes']:,} | `{p['sha256'][:16]}` |")  # fmt: skip
    span = max((line["to_ns"] - line["from_ns"]) / 1e9, 1e-9)
    frames = "; ".join(f"{k} {v:,} ({v / span:.2f} a second)"
                       for k, v in sorted(line["frames"].items()))  # fmt: skip
    lines += ["", f"Frames received: {frames or 'none'}.", ""]
    if line["gaps"]:
        lines += ["Gap rows:", "", "| venue | stream | kind | before | after | from | to | s |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for g in line["gaps"]:
            lines.append(f"| {g['venue']} | {g['stream']} | {g['kind']} | {g['before']} | "
                         f"{g['after']} | {_t(g['t_before'])} | {_t(g['t_after'])} | "
                         f"{g['seconds']:.3f} |")  # fmt: skip
    else:
        lines.append("Gap rows: none.")
    if line["dropped"]:
        lines.append(f"{line['dropped']} rows were stamped before the hour opened (the clock "
                     "stepped back) and dropped.")  # fmt: skip
    lines.append("")
    if samples:
        lines += ["The clock, offset = server time less the midpoint of the round trip, ms:", "",
                  "| venue | sample at | offset | round trip |", "|---|---|---|---|"]  # fmt: skip
        for s in samples:
            lines.append(f"| {s['venue']} | {_t(s['t0'])} | {s['offset_ms']:+.1f} | "
                         f"{s['rtt_ms']:.1f} |")  # fmt: skip
        for venue, c in line["clock"].items():
            flag = " (over the limit: the hour is marked clock)" if abs(
                c["offset_ms"]) > CLOCK_LIMIT_MS else ""  # fmt: skip
            lines.append(f"\n{venue}: median offset {c['offset_ms']:+.1f} ms "
                         f"({c['offset_min']:+.1f} to {c['offset_max']:+.1f}), round trip "
                         f"median {c['rtt_ms']:.0f} ms, p90 {c['rtt_p90']:.0f}"
                         f"{flag}.")  # fmt: skip
    else:
        lines.append("No clock samples fell in this hour.")
    lines += ["", *checks, ""]
    return "\n".join(lines) + "\n"


def hour_checks(rows: dict[str, Columns], coin: str, quote: str, hedge: str, start: int,
                end: int) -> list[str]:  # fmt: skip
    """``blinded demo arb``'s checks over one hour's rows: each venue's spread, touch and
    prints, the two venues against each other, each depth stream against its touch, and who
    prints."""
    streams = {s: r.frame() for s, r in rows.items()}
    return checks_lines(streams, coin, quote, hedge, start, end, "the hour")


def checks_lines(streams, coin, quote, hedge, start, end, what: str) -> list[str]:
    from blinded import bookstats

    a, b = _legs(streams, coin, quote, hedge, start, end)
    if not a.ticker.height or not b.ticker.height:
        return [f"No checks over {what}: a venue has no two-sided book row."]
    facts = {leg.instrument: leg.facts(end) for leg in (a, b)}
    out = [f"Over {what}:", "", *demo_mm.venue_table(facts), ""]
    tops = [{c: leg.ticker[c].to_numpy() for c in ("ts", "bid", "ask")} for leg in (a, b)]
    out += [
        *demo_mm.cross_lines(
            a.instrument, b.instrument, bookstats.cross_facts(*tops, end, demo_mm.ARB_DISTANCE_BPS)
        ),
        "",
    ]
    for leg, name in ((a, "quote_depth"), (b, "hedge_depth")):
        snap = streams[name].filter((pl.col("ts") >= start) & (pl.col("ts") < end))
        if not snap.height:
            out += [f"{leg.instrument}: no depth rows to check against its touch.", ""]
            continue
        s = bookstats.snapshot_agreement(
            *(leg.ticker[c].to_numpy() for c in ("ts", "bid", "ask", "bid_qty", "ask_qty")),
            *(snap[c].to_numpy() for c in ("ts", "bid0_px", "ask0_px", "bid0_qty", "ask0_qty")),
        )  # fmt: skip
        s["delay_ms"] = {"touch": bookstats.delay_ms(leg.ticker["ts"], leg.ticker["exch"]),
                         "snapshot": bookstats.delay_ms(snap["ts"], snap["exch"])}  # fmt: skip
        out += [*demo_mm.snapshot_lines(s, f"{leg.instrument}'s depth"), ""]
    if a.trades.height and b.trades.height:
        out += [
            "Who prints:",
            "",
            *demo_mm.flow_table({leg.instrument: leg.flow() for leg in (a, b)}),
            "",
        ]
    return out


def day_notes(built: dict) -> str:
    a, b = built["legs"]
    day = built["day"]
    lines = [f"## The day, {day.isoformat()}", "",
             "Built at 00:05 UTC from the day's parts by `blinded.demo_mm.arb_frames`, the "
             "code `blinded demo arb` builds a Tardis day with:", "",
             "| rows | count |", "|---|---|",
             *(f"| {k} | {v:,} |" for k, v in built["rows"].items()), "",
             "| file | bytes | sha256 |", "|---|---|---|"]  # fmt: skip
    for p in built["paths"].values():
        if p.exists():
            lines.append(f"| {p.name} | {p.stat().st_size:,} | `{_sha(p)}` |")
    lat = built.get("latencies") or {}
    lines += ["", "Latencies for the day's forward brief, the p90 round trip to each venue "
              "rounded up to the grid step: " + ", ".join(f"{k} {v}" for k, v in lat.items())
              + ".", ""]  # fmt: skip
    start = day_start(day)
    try:
        # over what was recorded: the time-weighted figures would carry the last row on
        last = max((int(f["ts"].max()) for f in built["streams"].values() if f.height),
                   default=start)  # fmt: skip
        lines += checks_lines(built["streams"], built["coin"], a.exchange, b.exchange, start,
                              min(start + DAY_NS, last + 1), "the day")  # fmt: skip
    except Exception as exc:
        lines.append(f"The day's checks were not computed: {type(exc).__name__}: {exc}")
    return "\n".join(lines) + "\n\n"


# -- the venues ----------------------------------------------------------------------------


class Mexc:
    """MEXC's four channels of one coin into a :class:`Recording`: protobuf frames, deduped
    on each stream's sequence field so that two open connections give one stream."""

    venue = "mexc"

    def __init__(self, rec: Recording, coin: str):
        self.rec, self.coin = rec, coin
        self.version: int | None = None
        self.depth_to: int | None = None
        self.limit_version: int | None = None
        self.t_depth: int | None = None
        self.deal_ids: deque = deque(maxlen=4096)
        self.deal_set: set = set()
        self.last_book_ns: dict[int, int] = {}  # each connection's last book ticker frame
        self.last_ns: int | None = None  # the last frame of any connection

    def channels(self) -> list[str]:
        c = self.coin
        return [f"spot@public.aggre.bookTicker.v3.api.pb@10ms@{c}",
                f"spot@public.aggre.deals.v3.api.pb@10ms@{c}",
                f"spot@public.aggre.depth.v3.api.pb@10ms@{c}",
                f"spot@public.limit.depth.v3.api.pb@{c}@{QUOTE_LEVELS}"]  # fmt: skip

    def on_frame(self, t: int, raw: bytes, conn: int = 0) -> None:
        """One binary frame received at ``t`` (ns) on connection ``conn``."""
        w = proto.wrapper(raw)
        body = w.get("body")
        send = int(w.get("sendTime") or 0) * MS
        self.last_ns = t
        self.rec.frame(f"mexc {body}")
        if body == "publicAggreBookTicker":
            self.last_book_ns[conn] = t
            b = w[body]
            v = int(b.get("version") or 0)
            if self.version is not None and v <= self.version:
                return
            self.version = v
            self.rec.add("quote_book", (t, send, float(b.get("bidPrice") or "nan"),
                                        float(b.get("askPrice") or "nan"),
                                        float(b.get("bidQuantity") or "nan"),
                                        float(b.get("askQuantity") or "nan")))  # fmt: skip
        elif body == "publicAggreDeals":
            for d in w[body].get("deals", []):
                key = d.get("tradeId") or (d.get("time"), d.get("price"), d.get("quantity"))
                if key in self.deal_set:
                    continue
                if len(self.deal_ids) == self.deal_ids.maxlen:
                    self.deal_set.discard(self.deal_ids[0])
                self.deal_ids.append(key)
                self.deal_set.add(key)
                side = "B" if d.get("tradeType") == 1 else "S"
                row = (t, int(d.get("time") or 0) * MS, side, float(d["price"]),
                       float(d["quantity"]))  # fmt: skip
                self.rec.add("quote_trades", row)
        elif body == "publicAggreDepths":
            d = w[body]
            lo, hi = int(d.get("fromVersion") or 0), int(d.get("toVersion") or 0)
            if self.depth_to is not None and hi <= self.depth_to:
                return
            if self.depth_to is not None and lo != self.depth_to + 1:
                self.rec.gap(Gap("mexc", "aggre.depth", "sequence", self.depth_to, lo,
                                 self.t_depth or t, t))  # fmt: skip
            self.depth_to, self.t_depth = hi, t
        elif body == "publicLimitDepths":
            d = w[body]
            v = int(d.get("version") or 0)
            if self.limit_version is not None and v <= self.limit_version:
                return
            self.limit_version = v
            pairs = [[(x["price"], x["quantity"]) for x in d.get(k, [])] for k in ("bids", "asks")]
            self.rec.add("quote_depth", (t, send, *_flat(*pairs, QUOTE_LEVELS)))


def _flat(bids, asks, levels: int) -> list:
    """``(price, quantity)`` levels as ``bid0_px, bid0_qty, ask0_px, ask0_qty, ...``, None
    past the book's end."""
    out = []
    for k in range(levels):
        for side in (bids, asks):
            out += [float(side[k][0]), float(side[k][1])] if k < len(side) else [None, None]
    return out


class LocalBook:
    """Binance's depth stream applied to a REST snapshot, by Binance's rule.

    Events wait until a snapshot arrives. An event whose ``u`` is at or under the last
    applied update id is dropped (a duplicate, or older than the snapshot). The first event
    applied after a snapshot must straddle it, ``U <= lastUpdateId + 1 <= u``, and every
    later one must start at the previous ``u + 1``; a later one that straddles is applied
    too, since each level's quantity is absolute and the levels it repeats are already at
    the values it carries, which is what a second connection's copy of the same events looks
    like. An event that starts past ``u + 1`` is a gap: the book is cleared and waits for a
    new snapshot.
    """

    def __init__(self, levels: int = LEVELS):
        self.levels = levels
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_u: int | None = None
        self.synced = False
        self.ready = False  # a snapshot is in
        self.pending: deque = deque(maxlen=20_000)
        self.top: tuple | None = None

    def snapshot(self, last_update_id: int, bids, asks) -> None:
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.last_u = int(last_update_id)
        self.synced, self.ready, self.top = False, True, None

    def clear(self) -> None:
        self.bids, self.asks = {}, {}
        self.synced = self.ready = False
        self.top = None

    def event(self, U: int, u: int, bids, asks) -> str:
        """``applied``, ``dropped``, ``waiting`` (no snapshot yet) or ``gap``."""
        if not self.ready:
            self.pending.append((U, u, bids, asks))
            return "waiting"
        if u <= self.last_u:
            return "dropped"
        if not U <= self.last_u + 1 <= u:
            return "gap"
        for book, rows in ((self.bids, bids), (self.asks, asks)):
            for p, q in rows:
                p, q = float(p), float(q)
                if q > 0:
                    book[p] = q
                else:
                    book.pop(p, None)
        self.last_u, self.synced = u, True
        return "applied"

    def top_levels(self) -> tuple:
        bids = heapq.nlargest(self.levels, self.bids.items())
        asks = heapq.nsmallest(self.levels, self.asks.items())
        return tuple(bids), tuple(asks)


class Binance:
    """Binance's three streams of one coin into a :class:`Recording`: JSON frames of the
    combined stream, the depth applied to a :class:`LocalBook`."""

    venue = "binance"

    def __init__(self, rec: Recording, coin: str, snapshot: Callable | None = None):
        self.rec, self.coin = rec, coin
        self.book = LocalBook()
        self.last_u: int | None = None  # bookTicker
        self.last_t: int | None = None  # trade ids
        self.t_trade: int | None = None
        self.t_depth: int | None = None
        self.snapshot = snapshot  # called with no argument when the book needs one
        self.gap_open: tuple | None = None  # (last u, stamp) of a depth gap not yet rebuilt
        self.last_ns: int | None = None

    def streams(self) -> str:
        s = self.coin.lower()
        return "/".join([f"{s}@bookTicker", f"{s}@trade", f"{s}@depth@100ms"])

    def on_frame(self, t: int, raw: str | bytes, conn: int = 0) -> None:
        m = json.loads(raw)
        stream, d = m.get("stream") or "", m.get("data") or {}
        self.last_ns = t
        if stream.endswith("@bookTicker"):
            self.rec.frame("binance bookTicker")
            u = int(d["u"])
            if self.last_u is not None and u <= self.last_u:
                return
            self.last_u = u
            self.rec.add("hedge_book", (t, t, float(d["b"]), float(d["a"]), float(d["B"]),
                                        float(d["A"])))  # fmt: skip
        elif stream.endswith("@trade"):
            self.rec.frame("binance trade")
            tid = int(d["t"])
            if self.last_t is not None and tid <= self.last_t:
                return
            if self.last_t is not None and tid != self.last_t + 1:
                self.rec.gap(Gap("binance", "trade", "sequence", self.last_t, tid,
                                 self.t_trade or t, t))  # fmt: skip
            self.last_t, self.t_trade = tid, t
            self.rec.add("hedge_trades", (t, int(d["E"]) * MS, "S" if d["m"] else "B",
                                          float(d["p"]), float(d["q"])))  # fmt: skip
        elif "@depth" in stream:
            self.rec.frame("binance depth")
            self.depth(t, int(d["U"]), int(d["u"]), d.get("b", []), d.get("a", []),
                       int(d.get("E") or 0) * MS)  # fmt: skip

    def depth(self, t: int, U: int, u: int, bids, asks, exch: int) -> None:
        was = self.book.last_u
        got = self.book.event(U, u, bids, asks)
        if got == "waiting":
            return
        if got == "gap":
            self.rec.gap(Gap("binance", "depth", "sequence", was, U, self.t_depth or t, t))
            self.gap_open = (was, self.t_depth or t)
            self.book.clear()
            self.book.pending.append((U, u, bids, asks))
            if self.snapshot is not None:
                self.snapshot()
            return
        if got == "applied":
            self.t_depth = t
            if self.gap_open is not None:
                self.rec.gap(Gap("binance", "depth", "resync", self.gap_open[0], u,
                                 self.gap_open[1], t))  # fmt: skip
                self.gap_open = None
            self._emit(t, exch)

    def seeded(self, t: int, last_update_id: int, bids, asks) -> None:
        """A REST snapshot arrived at ``t``: seed the book and apply what waited for it."""
        pending = list(self.book.pending)
        self.book.pending.clear()
        self.book.snapshot(last_update_id, bids, asks)
        for U, u, b, a in pending:
            got = self.book.event(U, u, b, a)
            if got == "gap":
                # the snapshot is older than what waited: ask again
                self.book.clear()
                self.book.pending.extend(pending)
                if self.snapshot is not None:
                    self.snapshot()
                return
        if self.book.synced:
            self.t_depth = t
            if self.gap_open is not None:
                self.rec.gap(Gap("binance", "depth", "resync", self.gap_open[0],
                                 self.book.last_u, self.gap_open[1], t))  # fmt: skip
                self.gap_open = None
            self._emit(t, 0)

    def _emit(self, t: int, exch: int) -> None:
        top = self.book.top_levels()
        if top == self.book.top:
            return
        self.book.top = top
        bids, asks = top
        self.rec.add("hedge_depth", (t, exch or t, *_flat(bids, asks, LEVELS)))


# -- the process -------------------------------------------------------------------------


@dataclass
class Status:
    started_ns: int
    reconnects: dict = field(default_factory=dict)
    rotations: dict = field(default_factory=dict)
    gaps: int = 0
    errors: int = 0
    connected: dict = field(default_factory=dict)


def rss_bytes(pid: int | None = None) -> int:
    """The resident memory of ``pid`` (this process by default), from ``ps``."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid or os.getpid())],
                         capture_output=True, text=True)  # fmt: skip
    text = out.stdout.strip()
    return int(text) * 1024 if text.isdigit() else 0


class Recorder:
    """The asyncio process: both venues' connections, the watchdog, the clock, the hour and
    the day. Needs ``aiohttp`` (``pip install 'blinded[live]'``)."""

    def __init__(self, root: str | Path, coin: str = "ATUSDT", hours: float | None = None,
                 log: Callable[[str], None] = print):  # fmt: skip
        self.root = Path(root).expanduser()
        self.coin = coin.upper()
        self.hours = hours
        self.log = log
        self.started_ns = time.time_ns()
        self.rec = Recording(self.root, self.coin, log=log, started_ns=self.started_ns)
        self.mexc = Mexc(self.rec, self.coin)
        self.binance = Binance(self.rec, self.coin, snapshot=self._want_snapshot)
        self.status = Status(self.started_ns)
        self.stop = asyncio.Event()
        self._snap_task: asyncio.Task | None = None
        self._conn_ids = 0
        self.writes: list[asyncio.Task] = []

    # REST
    async def _get(self, url: str) -> tuple[int, int, object]:
        t0 = time.time_ns()
        async with self.session.get(url) as r:
            r.raise_for_status()
            body = await r.json(content_type=None)
        return t0, time.time_ns(), body

    async def clock(self) -> list[dict]:
        out = []
        for venue, base in (("mexc", MEXC_REST), ("binance", BINANCE_REST)):
            for _ in range(CLOCK_SAMPLES):
                try:
                    t0, t1, body = await self._get(f"{base}/api/v3/time")
                    out.append(clock_sample(venue, t0, t1, int(body["serverTime"])))
                except Exception as exc:
                    self.log(f"clock sample of {venue} failed: {type(exc).__name__}: {exc}")
                await asyncio.sleep(0.5)
        return out

    def _want_snapshot(self) -> None:
        if self._snap_task is None or self._snap_task.done():
            self._snap_task = asyncio.get_running_loop().create_task(self._snapshot())

    async def _snapshot(self) -> None:
        for attempt in range(10):
            try:
                _, t1, body = await self._get(f"{BINANCE_REST}/api/v3/depth?symbol={self.coin}"
                                              "&limit=5000")  # fmt: skip
                self.binance.seeded(t1, int(body["lastUpdateId"]), body["bids"], body["asks"])
                return
            except Exception as exc:
                self.log(f"depth snapshot failed: {type(exc).__name__}: {exc}")
                await asyncio.sleep(min(2**attempt, 30))

    # connections
    async def _open(self, venue: str):
        import aiohttp

        self._conn_ids += 1
        cid = self._conn_ids
        timeout = aiohttp.ClientWSTimeout(ws_receive=None, ws_close=1.0)
        if venue == "mexc":
            ws = await self.session.ws_connect(MEXC_WS, max_msg_size=1 << 22, timeout=timeout,
                                               autoping=True)  # fmt: skip
            await ws.send_str(json.dumps({"method": "SUBSCRIPTION",
                                          "params": self.mexc.channels()}))  # fmt: skip
        else:
            ws = await self.session.ws_connect(BINANCE_WS + self.binance.streams(),
                                               max_msg_size=1 << 22, timeout=timeout,
                                               heartbeat=30.0, autoping=True)  # fmt: skip
        conn = {"id": cid, "ws": ws, "venue": venue, "first": asyncio.Event(), "last": None,
                "opened": time.time_ns(), "dead": asyncio.Event()}  # fmt: skip
        conn["task"] = asyncio.get_running_loop().create_task(self._read(conn))
        if venue == "mexc":
            conn["ping"] = asyncio.get_running_loop().create_task(self._ping(conn))
        return conn

    async def _read(self, conn: dict) -> None:
        import aiohttp

        ws = conn["ws"]
        adapter = self.mexc if conn["venue"] == "mexc" else self.binance
        try:
            while True:
                msg = await ws.receive()
                t = time.time_ns()
                if msg.type not in (aiohttp.WSMsgType.BINARY, aiohttp.WSMsgType.TEXT):
                    break
                if conn["venue"] == "mexc" and msg.type == aiohttp.WSMsgType.TEXT:
                    continue  # MEXC's acknowledgements and pongs
                try:
                    adapter.on_frame(t, msg.data, conn["id"])
                except Exception as exc:  # one bad frame does not drop the connection
                    self.status.errors += 1
                    if self.status.errors <= 20:
                        self.log(f"{conn['venue']} frame not read: {type(exc).__name__}: {exc}")
                conn["last"] = t
                if not conn["first"].is_set():
                    conn["first"].set()
        except Exception as exc:
            self.log(f"{conn['venue']} connection {conn['id']}: {type(exc).__name__}: {exc}")
        finally:
            conn["dead"].set()

    async def _ping(self, conn: dict) -> None:
        with contextlib.suppress(Exception):
            while not conn["dead"].is_set():
                await asyncio.sleep(PING_S)
                await conn["ws"].send_str(json.dumps({"method": "PING"}))

    def _drop(self, conn: dict) -> None:
        """Close without waiting for the close handshake."""
        for key in ("ping",):
            if key in conn:
                conn[key].cancel()
        conn["closing"] = asyncio.get_running_loop().create_task(conn["ws"].close())
        conn["task"].cancel()

    async def _feed(self, venue: str) -> None:
        """One venue's connection, replaced before it is dropped every :data:`ROTATE_S`,
        and replaced at once when it dies or, on MEXC, when its book ticker goes silent."""
        conn = await self._connect(venue, None)
        rotate_at = time.monotonic() + ROTATE_S
        while not self.stop.is_set():
            await asyncio.sleep(0.25)
            now = time.time_ns()
            silent = False
            if venue == "mexc" and conn["first"].is_set():
                last = self.mexc.last_book_ns.get(conn["id"], conn["opened"])
                silent = now - last > HEARTBEAT_S * 1e9
            if conn["dead"].is_set() or silent:
                kind = "silent" if silent and not conn["dead"].is_set() else "reconnect"
                last = conn["last"] or conn["opened"]
                if venue == "mexc":
                    last = self.mexc.last_book_ns.get(conn["id"], last)
                self._drop(conn)
                conn = await self._connect(venue, (kind, last))
                rotate_at = time.monotonic() + ROTATE_S
            elif time.monotonic() > rotate_at:
                try:
                    new = await self._open(venue)
                    await asyncio.wait_for(new["first"].wait(), 15.0)
                except Exception as exc:
                    self.log(f"{venue} rotation failed: {type(exc).__name__}: {exc}")
                    rotate_at = time.monotonic() + 60
                    continue
                self._drop(conn)
                conn = new
                self.status.rotations[venue] = self.status.rotations.get(venue, 0) + 1
                rotate_at = time.monotonic() + ROTATE_S
        self._drop(conn)

    async def _connect(self, venue: str, lost: tuple | None) -> dict:
        delay = 1.0
        while True:
            conn = None
            try:
                conn = await self._open(venue)
                await asyncio.wait_for(conn["first"].wait(), 15.0)
                break
            except Exception as exc:
                self.log(f"{venue} connect failed: {type(exc).__name__}: {exc}")
                if conn is not None:
                    with contextlib.suppress(Exception):
                        self._drop(conn)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
        self.status.connected[venue] = iso(time.time_ns())
        if lost is not None:
            kind, last = lost
            self.status.reconnects[venue] = self.status.reconnects.get(venue, 0) + 1
            stream = "bookTicker" if kind == "silent" else "connection"
            self.rec.gap(Gap(venue, stream, kind, None, None, int(last), int(conn["last"])))
            if venue == "binance":
                # the depth stream's book cannot be trusted across a lost connection: rebuild it
                self.binance.gap_open = (self.binance.book.last_u, int(last))
                self.binance.book.clear()
                self._want_snapshot()
        elif venue == "binance":
            self._want_snapshot()
        return conn

    # the hour, the day, the status
    async def _hours(self) -> None:
        while not self.stop.is_set():
            now = time.time_ns()
            nxt = hour_of(now) + HOUR_NS
            await self._sleep_until(nxt + 1_000_000_000)
            if self.stop.is_set():
                return
            # the clock first, so its samples are in the hour they are written with
            if self.rec.hour is not None and self.rec.hour < hour_of(time.time_ns()):
                await self._close_hour(hour_of(time.time_ns()))
            samples = await self.clock()
            self.rec.clock_samples(samples)
            for venue, s in clock_summary(samples).items():
                if abs(s["offset_ms"]) > CLOCK_LIMIT_MS:
                    self.log(f"the clock is {s['offset_ms']:+.0f} ms off {venue}: this hour is "
                             "marked clock")  # fmt: skip
            if datetime.now(UTC).hour == 0:
                await self._sleep_until(day_start(day_of(time.time_ns())) + BUILD_AFTER_NS)
            # every hour, not only at 00:05: a machine asleep through midnight builds the day
            # in its first hour awake (build_due does nothing when no past day is due)
            await self.build_due()

    async def _close_hour(self, until: int | None) -> None:
        taken = self.rec.take(until)
        if taken is not None:
            await asyncio.to_thread(self.rec.write_hour, taken)

    async def build_due(self) -> None:
        """Build every past day that has parts and no data file, each in its own process: a
        busy day's build holds the whole day in memory, and the recording must not."""
        today = day_of(time.time_ns())
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            try:
                day = Date.fromisoformat(d.name)
            except ValueError:
                continue
            data = d / f"{stem(self.coin, day)}.parquet"
            if day < today and not data.exists() and any((d / "parts").glob("*.parquet")):
                argv = [sys.executable, "-m", "blinded.recorder", "build", str(self.root),
                        day.isoformat(), "--coin", self.coin]  # fmt: skip
                proc = await asyncio.create_subprocess_exec(
                    *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
                )
                out, _ = await proc.communicate()
                for line in out.decode(errors="replace").splitlines()[-5:]:
                    self.log(line)
                if proc.returncode:
                    self.log(f"day {day} not built (exit {proc.returncode}); its parts are kept")

    async def _status(self) -> None:
        path = self.root / "status.json"
        while not self.stop.is_set():
            line = {"t": iso(time.time_ns()), "pid": os.getpid(), "started": iso(self.started_ns),
                    "rss_bytes": rss_bytes(), "maxrss_bytes": resource.getrusage(
                        resource.RUSAGE_SELF).ru_maxrss,
                    "rows_written": self.rec.totals,
                    "rows_open_hour": {s: len(r) for s, r in self.rec.buf.items()},
                    "frames_open_hour": self.rec.frames,
                    "gaps_open_hour": len(self.rec.gaps), "reconnects": self.status.reconnects,
                    "rotations": self.status.rotations, "connected": self.status.connected,
                    "frame_errors": self.status.errors,
                    "binance_book_synced": self.binance.book.synced}  # fmt: skip
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(line, indent=1) + "\n")
            os.replace(tmp, path)
            with open(self.root / "status.jsonl", "a") as fh:
                fh.write(json.dumps(line) + "\n")
            await self._sleep_until(time.time_ns() + 60 * 10**9)

    async def _sleep_until(self, ns: int) -> None:
        """Until the wall clock reaches ``ns``, or stop. The event loop's clock stops while
        the machine sleeps, so the wait is taken a minute at a time against the wall clock: one
        long wait would end late by every minute the machine slept."""
        while not self.stop.is_set():
            left = (ns - time.time_ns()) / 1e9
            if left <= 0:
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.stop.wait(), min(left, SLEEP_CHECK_S))

    async def run(self) -> None:
        import aiohttp

        self.root.mkdir(parents=True, exist_ok=True)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            self.session = session
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self.stop.set)
            self.rec.on_hour = lambda taken: loop.create_task(
                asyncio.to_thread(self.rec.write_hour, taken)
            )
            samples = await self.clock()
            summary = clock_summary(samples)
            for venue in ("mexc", "binance"):
                if venue not in summary:
                    raise ClockError(f"no clock sample from {venue}: the recorder does not "
                                     "start without one")  # fmt: skip
            check_clock(summary)
            self.rec.clock_samples(samples)
            self.log(
                "clock: "
                + "; ".join(
                    f"{v} {s['offset_ms']:+.1f} ms, round trip {s['rtt_ms']:.0f} ms"
                    for v, s in summary.items()
                )
            )
            await self.build_due()
            tasks = [
                asyncio.create_task(self._feed("mexc")),
                asyncio.create_task(self._feed("binance")),
                asyncio.create_task(self._hours()),
                asyncio.create_task(self._status()),
            ]
            if self.hours is not None:
                await self._sleep_until(self.started_ns + int(self.hours * HOUR_NS))
                self.stop.set()
            try:
                await asyncio.gather(*tasks)
            finally:
                self.stop.set()
                for t in tasks:
                    t.cancel()
                await self._close_hour(None)


def record(
    root: str | Path,
    coin: str = "ATUSDT",
    hours: float | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Run the recorder until stopped (or for ``hours``)."""
    asyncio.run(Recorder(root, coin, hours, log).run())


# -- reading a recording -------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def hours_of(day_dir: Path) -> list[dict]:
    """Every ``hours.jsonl`` line of a day, in the order written."""
    return read_jsonl(Path(day_dir) / "hours.jsonl")


def gaps_of(day_dir: Path) -> list[dict]:
    return read_jsonl(Path(day_dir) / "gaps.jsonl")


def clock_of(day_dir: Path) -> list[dict]:
    return read_jsonl(Path(day_dir) / "clock.jsonl")


def day_built(day_dir: Path, coin: str, day: Date) -> dict | None:
    """The day's three files when it is built, else None."""
    s = stem(coin, day)
    paths = {k: Path(day_dir) / f"{s}{x}" for k, x in (("data", ".parquet"),
             ("touch", ".touch.parquet"), ("depth", ".depth.parquet"))}  # fmt: skip
    return paths if all(p.exists() for p in paths.values()) else None


def hash_parts(day_dir: Path) -> str:
    """One SHA-256 over a day's part files, names and contents, for an open day."""
    h = hashlib.sha256()
    for p in sorted((Path(day_dir) / "parts").glob("*.parquet")):
        h.update(p.name.encode())
        h.update(_sha(p).encode())
    return h.hexdigest()


def hour_statuses(day_dir: Path, day: Date) -> dict[int, tuple[str, str]]:
    """Each hour of a recorded day, by its start in ns: ``(status, why)``.

    ``unobserved`` when no part covers the whole hour (the recorder was not running all of
    it); ``gapped`` when a gap row's span, from the receive stamp before it to the one
    after, touches the hour; ``clock`` when the hour's median offset against either venue
    is over :data:`CLOCK_LIMIT_MS`; ``judged`` otherwise.
    """
    start = day_start(day)
    lines = {}
    for line in hours_of(day_dir):
        lines.setdefault(line["hour_ns"], []).append(line)
    gaps = gaps_of(day_dir)
    for line in (x for v in lines.values() for x in v):
        gaps += [g for g in line["gaps"] if g not in gaps]
    out = {}
    for k in range(24):
        h = start + k * HOUR_NS
        mine = lines.get(h, [])
        whole = len(mine) == 1 and mine[0]["from_ns"] <= h and mine[0]["to_ns"] >= h + HOUR_NS
        if not whole:
            why = ("the recorder was not running" if not mine else
                   "the recorder ran only part of the hour")  # fmt: skip
            out[h] = ("unobserved", why)
            continue
        hit = [g for g in gaps if g["t_before"] < h + HOUR_NS and g["t_after"] >= h]
        if hit:
            kinds = sorted({f"{g['venue']} {g['stream']} {g['kind']}" for g in hit})
            out[h] = ("gapped", "; ".join(kinds))
            continue
        off = {v: c["offset_ms"] for v, c in (mine[0].get("clock") or {}).items()
               if abs(c["offset_ms"]) > CLOCK_LIMIT_MS}  # fmt: skip
        if off:
            out[h] = ("clock", "; ".join(f"{v} {x:+.0f} ms" for v, x in off.items()))
            continue
        out[h] = ("judged", "")
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m blinded.recorder build ROOT DAY [--coin C]``: build one recorded day."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m blinded.recorder")
    parser.add_argument("what", choices=["build"])
    parser.add_argument("root", type=Path)
    parser.add_argument("day")
    parser.add_argument("--coin", default="ATUSDT")
    args = parser.parse_args(argv)
    rec = Recording(args.root, args.coin, log=lambda m: print(m, flush=True))
    rec.build_day(Date.fromisoformat(args.day))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
