"""Public demo data: a free Tardis day of Binance spot and perp, and days of Binance
perpetuals from Binance's own archive.

Tardis (tardis.dev) serves its CSV datasets for the first day of every month
without an API key. ``blinded demo tardis`` downloads, for each symbol, a top
of book file and the ``trades`` file of ``binance`` (spot) and
``binance-futures`` (USD-M perpetual) for that day, and builds one parquet in
the layout the XRP brief reads: a ``ts`` column on the receive clock, one row
per event, book rows with ``bid0_px``, ``ask0_px``, ``bid0_qty``, ``ask0_qty``
and ``mid``, trade rows with ``side`` (``B``/``S``), ``price`` and ``qty`` plus
the book as it stood when the trade arrived. Spot keeps its symbol and the perp
gets ``_PERP``. A brief beside the parquet points at it.

The top of book comes from ``book_ticker`` by default: Binance's own
best-bid-and-offer stream, one row per change. Tardis also has ``quotes``,
the same columns, which on Binance spot arrive on a 100 ms cadence (median gap
100.2 ms on 2026-10-01, against 2.0 ms for ``book_ticker``), so a spot book
built from it lags the trades.

The download uses the ``tardis-dev`` client (``pip install 'blinded[demo]'``):
the dataset endpoint sits behind Cloudflare and refuses a bare ``curl``.
Redistribution terms for the files are unclear, so keep them out of
repositories, as with any market data.

``blinded demo binance`` builds several days of USD-M perpetuals from
``data.binance.vision``, top of book and trades on the exchange clock, thinned
to the stamps the judge reads (:func:`binance`).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from blinded import durations

EXCHANGES = {"binance": "", "binance-futures": "_PERP"}
BOOK_TYPES = ("book_ticker", "quotes")
_BOOK = ("bid0_px", "ask0_px", "bid0_qty", "ask0_qty")
COLUMNS = ("ts", "symbol", "side", "price", "qty", *_BOOK, "mid", "is_trade")


def raw_name(exchange: str, data_type: str, day: Date, symbol: str) -> str:
    """The file name ``tardis-dev`` gives a download."""
    return f"{exchange}_{data_type}_{day.isoformat()}_{symbol}.csv.gz"


def download(
    day: Date, symbols: list[str], raw: Path, book: str = "book_ticker", api_key: str | None = None
) -> list[Path]:
    """Fetch the book and trades files for ``day`` into ``raw``; files already there are kept."""
    if book not in BOOK_TYPES:
        raise ValueError(f"book must be one of {BOOK_TYPES}, got {book!r}")
    try:
        from tardis_dev import download_datasets
    except ImportError:
        raise ImportError(
            "blinded demo needs the tardis-dev client: pip install 'blinded[demo]'"
        ) from None
    key = api_key if api_key is not None else os.environ.get("TARDIS_API_KEY", "")
    if day.day != 1 and not key:
        raise ValueError(
            f"{day} is not the first of a month; Tardis serves other days only with an "
            "API key (set TARDIS_API_KEY)"
        )
    raw.mkdir(parents=True, exist_ok=True)
    for exchange in EXCHANGES:
        download_datasets(
            exchange=exchange,
            data_types=[book, "trades"],
            symbols=symbols,
            from_date=day.isoformat(),
            to_date=(day + timedelta(days=1)).isoformat(),
            api_key=key,
            download_dir=str(raw),
        )
    kinds = (book, "trades")
    return [raw / raw_name(ex, k, day, s) for ex in EXCHANGES for k in kinds for s in symbols]


def _receive_time() -> pl.Expr:
    return pl.from_epoch(pl.col("local_timestamp"), time_unit="us").dt.cast_time_unit("ns")


def book_rows(path: Path, symbol: str) -> pl.DataFrame:
    """Book rows from a Tardis ``book_ticker`` or ``quotes`` file.

    Rows with an empty side, or a bid at or above the ask, are dropped.
    """
    frame = pl.read_csv(path)
    bid, ask = pl.col("bid_price"), pl.col("ask_price")
    return frame.filter((bid > 0) & (ask > bid)).select(
        _receive_time().alias("ts"),
        pl.lit(symbol).alias("symbol"),
        pl.lit(None, pl.String).alias("side"),
        pl.lit(None, pl.Float64).alias("price"),
        pl.lit(None, pl.Float64).alias("qty"),
        bid.alias("bid0_px"),
        ask.alias("ask0_px"),
        pl.col("bid_amount").alias("bid0_qty"),
        pl.col("ask_amount").alias("ask0_qty"),
        ((bid + ask) / 2).alias("mid"),
        pl.lit(False).alias("is_trade"),
    )


def trades(path: Path, symbol: str, book: pl.DataFrame) -> pl.DataFrame:
    """Trade rows from a Tardis ``trades`` file, each with the book received at or before it.

    Trades that arrive before the first quote of their symbol have no book and are dropped.
    """
    frame = pl.read_csv(path)
    rows = frame.select(
        _receive_time().alias("ts"),
        pl.lit(symbol).alias("symbol"),
        pl.col("side").replace_strict({"buy": "B", "sell": "S"}, return_dtype=pl.String),
        pl.col("price").cast(pl.Float64),
        pl.col("amount").cast(pl.Float64).alias("qty"),
    ).with_row_index("_order")
    state = book.select("ts", *_BOOK, "mid")
    joined = rows.sort("ts", maintain_order=True).join_asof(state, on="ts", strategy="backward")
    return (
        joined.sort("_order")
        .drop("_order")
        .filter(pl.col("mid").is_not_null())
        .with_columns(pl.lit(True).alias("is_trade"))
        .select(COLUMNS)
    )


def build(raw: Path, day: Date, symbols: list[str], book: str = "book_ticker") -> pl.DataFrame:
    """One frame of every book and trade row for ``symbols`` on ``day``, ordered by receive time.

    Within a symbol the rows keep the order of their files, book before
    trades, so rows that share a receive time resolve as the files list them.
    """
    parts = []
    for exchange, suffix in EXCHANGES.items():
        for symbol in symbols:
            name = symbol + suffix
            quotes = book_rows(raw / raw_name(exchange, book, day, symbol), name)
            fills = trades(raw / raw_name(exchange, "trades", day, symbol), name, quotes)
            parts += [quotes, fills]
    frame = pl.concat(parts).sort("ts", maintain_order=True)
    return frame.with_columns(pl.col("side").cast(pl.Categorical))


def brief_for(data_file: str, symbol: str, day: Date) -> dict:
    """A brief in the shape of ``examples/demo.brief.yaml`` for the demo file."""
    return {
        "blinded": 0,
        "name": f"tardis-binance-{symbol.lower()}-{day.isoformat()}",
        "data": {
            "path": data_file,
            "format": "parquet",
            "time": {"column": "ts", "unit": "ns", "tz": "UTC", "clock": "receive"},
            "instrument_column": "symbol",
            "streams": {
                "book": {
                    "filter": "is_trade == false",
                    "fields": {
                        "bid": "bid0_px",
                        "ask": "ask0_px",
                        "bid_qty": "bid0_qty",
                        "ask_qty": "ask0_qty",
                    },
                },
                "trades": {
                    "filter": "is_trade == true",
                    "fields": {"price": "price", "qty": "qty", "side": "side"},
                    "side_values": {"buy": "B", "sell": "S"},
                },
            },
            "availability": {"default_delay_ms": 0, "overrides": []},
        },
        "target": {
            "instrument": symbol,
            "price": "mid",
            "horizons": ["100ms", "1s", "10s", "60s"],
            "primary_horizon": "10s",
        },
        "grid": {"step": "100ms", "start": None, "warmup": "5min"},
        "execution": {
            "latency_ms": 50,
            "taker_fee_bps": 7.5,
            "maker_fee_bps": 0.0,
            "max_position": 1.0,
            "min_rebalance": 0.1,
        },
        "splits": {"scheme": "time", "fractions": [0.5, 0.25, 0.25], "embargo": "5min"},
        "stats": {"se": "both", "block": "5min", "n_boot": 1000, "seed": 20241128},
    }


def tardis(
    day: Date | str,
    symbols: list[str],
    out: str | Path,
    hours: float | None = None,
    book: str = "book_ticker",
    api_key: str | None = None,
) -> Path:
    """Download one Tardis day and write the demo parquet and its brief under ``out``.

    Parameters
    ----------
    day
        A first of the month, unless an API key is given.
    symbols
        Binance symbols such as ``XRPUSDT``; each gives a spot and a ``_PERP`` instrument.
        The first is the target.
    out
        Directory for ``raw/`` (the downloads), the parquet and ``brief.yaml``.
    hours
        Keep only rows received in the first ``hours`` hours of the day (UTC).
    book
        ``book_ticker`` (every change of the best bid and offer) or ``quotes``.

    Returns
    -------
    Path
        The brief.
    """
    day = Date.fromisoformat(day) if isinstance(day, str) else day
    out = Path(out).expanduser()
    download(day, symbols, out / "raw", book, api_key)
    frame = build(out / "raw", day, symbols, book)
    stem = f"binance-{'-'.join(s.lower() for s in symbols)}-{day.isoformat()}"
    if book != "book_ticker":
        stem += f"-{book}"
    if hours is not None:
        start = pl.datetime(day.year, day.month, day.day, time_unit="ns")
        end = start + pl.duration(microseconds=int(round(hours * 3600 * 1e6)))
        frame = frame.filter(pl.col("ts") < end)
        stem += f"-{hours:g}h"
    data = out / f"{stem}.parquet"
    frame.write_parquet(data)
    brief = out / f"{stem}.brief.yaml"
    brief.write_text(yaml.safe_dump(brief_for(data.name, symbols[0], day), sort_keys=False))
    return brief


def summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows per instrument and stream, with the first and last receive time."""
    return (
        frame.group_by("symbol", "is_trade", maintain_order=True)
        .agg(
            pl.len().alias("rows"),
            pl.col("ts").min().alias("first"),
            pl.col("ts").max().alias("last"),
        )
        .sort("symbol", "is_trade")
    )


# -- Binance USD-M archive -----------------------------------------------------

ARCHIVE = "https://data.binance.vision/data/futures/um/daily"
HEADERS = {
    "bookTicker": "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,"
    "transaction_time,event_time",
    "trades": "id,price,qty,quote_qty,time,is_buyer_maker",
}
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
ROWS_GUIDE, ROWS_LIMIT = 8_500_000, 10_000_000
# what a thinned symbol-day keeps per grid step, measured on the four perpetuals over 27 to
# 29 March 2024: 5,030,813 book rows and 2,495,529 trade rows in 12 x 172,800 steps. Two kept
# stamps 50 ms apart often read the same book row, and a quiet half-second has one side's
# trades or none.
BOOK_ROWS_PER_STAMP, TRADE_ROWS_PER_STEP = 0.81, 1.2
BATCH_MEDIAN_MS, BATCH_BIN_SHARE = 50.0, 0.20
CLOCK_FAIL_MS = 5
TOUCH_SAMPLE = 200_000
MS = 1_000_000
SUMMED = ("cum_qty", "cum_signed_qty", "cum_buy_qty", "cum_sell_qty", "cum_notional")


class BuildError(ValueError):
    """A raw file failed one of the builder's checks."""


def archive_name(kind: str, symbol: str, day: Date) -> str:
    """The archive's name for one day of one symbol, ``BTCUSDT-bookTicker-2024-03-27.zip``."""
    return f"{symbol}-{kind}-{day.isoformat()}.zip"


def fetch(kind: str, symbol: str, day: Date, raw: Path, attempts: int = 3) -> Path:
    """Download one zip and its ``.CHECKSUM`` into ``raw`` and check the SHA-256.

    A zip already there that matches its checksum is kept, so an interrupted
    build picks up where it stopped. A mismatch after the last attempt raises.
    """
    if kind not in HEADERS:
        raise ValueError(f"kind must be one of {sorted(HEADERS)}, got {kind!r}")
    name = archive_name(kind, symbol, day)
    return fetch_url(f"{ARCHIVE}/{kind}/{symbol}/{name}", raw, attempts)


def fetch_url(url: str, raw: Path, attempts: int = 3) -> Path:
    """Download one archive zip at ``url`` and its ``.CHECKSUM`` into ``raw``, as :func:`fetch`."""
    raw.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[1]
    path, check = raw / name, raw / f"{name}.CHECKSUM"
    if not check.is_file():
        check.write_bytes(_get(url + ".CHECKSUM"))
    want = check.read_text().split()[0].lower()
    for attempt in range(attempts):
        if path.is_file() and file_sha256(path) == want:
            return path
        part = path.with_name(path.name + ".part")
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as resp, part.open("wb") as out:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() == want:
            part.replace(path)
            return path
        part.unlink()
        time.sleep(2 * (attempt + 1))
    raise BuildError(f"{name} does not match its .CHECKSUM after {attempts} downloads")


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_archive(path: Path, kind: str) -> pl.DataFrame:
    """The CSV inside one archive zip, its header checked against the archive's layout."""
    with zipfile.ZipFile(path) as zf:
        (member,) = zf.infolist()
        data = zf.read(member)
    header = data[: data.index(b"\n")].decode().strip()
    if header != HEADERS[kind]:
        want = HEADERS[kind]
        raise BuildError(f"{Path(path).name}: the header is {header!r}, expected {want!r}")
    return pl.read_csv(io.BytesIO(data))


def book_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """A ``bookTicker`` file as stamps in ms, the touch, and the event time."""
    return raw.select(
        pl.col("transaction_time").alias("t"),
        pl.col("event_time"),
        pl.col("best_bid_price").cast(pl.Float64).alias("bid"),
        pl.col("best_ask_price").cast(pl.Float64).alias("ask"),
        pl.col("best_bid_qty").cast(pl.Float64).alias("bid_qty"),
        pl.col("best_ask_qty").cast(pl.Float64).alias("ask_qty"),
    )


def trades_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """A ``trades`` file as stamps in ms, the aggressor's side, price and quantity.

    ``is_buyer_maker`` means the buyer's order was resting, so the seller hit it:
    a sell.
    """
    return raw.select(
        pl.col("time").alias("t"),
        pl.when(pl.col("is_buyer_maker")).then(pl.lit("S")).otherwise(pl.lit("B")).alias("side"),
        pl.col("price").cast(pl.Float64),
        pl.col("qty").cast(pl.Float64),
    )


# -- the raw checks -------------------------------------------------------------


def gap_check(t_ms: np.ndarray, median_limit: float | None = BATCH_MEDIAN_MS) -> dict:
    """Stamps in file order: backward steps, and the gaps between distinct stamps.

    ``share_max`` is the largest share of gaps in one 10 ms bin from 20 to
    1000 ms, ``bin_max_ms`` where it is. A stream rebuilt on a cadence, like
    Tardis's spot ``quotes`` at 100 ms, piles its gaps into one bin. Fails
    on any backward step, a bin over 20%, or, with ``median_limit``, a median
    gap at or over it.

    Examples
    --------
    >>> out = gap_check(np.arange(0, 10_000, 100), median_limit=None)  # every 100 ms
    >>> out["share_max"], out["bin_max_ms"], out["ok"]
    (1.0, 100, False)
    """
    backward = int(np.sum(np.diff(t_ms) < 0))
    gaps = np.diff(np.unique(t_ms)).astype(float)
    median = float(np.median(gaps)) if len(gaps) else np.nan
    bins = np.arange(20, 1001, 10)
    counts, _ = np.histogram(gaps, bins=bins)
    share = counts / len(gaps) if len(gaps) else counts * 0.0
    worst = int(np.argmax(share)) if len(share) else 0
    ok = backward == 0 and share[worst] <= BATCH_BIN_SHARE
    if median_limit is not None:
        ok = ok and median < median_limit
    return {"backward": backward, "median_gap_ms": median, "share_max": float(share[worst]),
            "bin_max_ms": int(bins[worst]), "ok": bool(ok)}  # fmt: skip


def raw_checks(book: pl.DataFrame, trades: pl.DataFrame, name: str) -> dict:
    """The checks of one symbol-day on its raw rows; raises :class:`BuildError` on a failure.

    Stamps never go backwards; the book is not batched (median gap under 50
    ms, no 10 ms bin over 20% of gaps) and neither are the trades (no bin
    over 20%: trades are sparse, so their median gap says nothing);
    ``event_time - transaction_time`` is reported, a median over 50 ms as a
    warning; and the two streams agree on the clock: trades print at the touch
    of the book read within 5 ms of their stamp, and book mid and trade price
    move together within 5 ms.
    """
    from blinded.diagnostics import touch_alignment, xcorr_lag

    b, t = book["t"].to_numpy(), trades["t"].to_numpy()
    out = {"book_rows": book.height, "trade_rows": trades.height}
    out["book"] = gap_check(b)
    out["trades"] = gap_check(t, median_limit=None)
    lag = (book["event_time"] - book["t"]).to_numpy()
    out["event_lag_median_ms"] = float(np.median(lag))
    out["event_lag_p99_ms"] = float(np.quantile(lag, 0.99))
    book_ev, trades_ev = _events(book), _events(trades)
    every = max(1, trades.height // TOUCH_SAMPLE)
    sample = _events(trades.gather_every(every))
    touch = touch_alignment(book_ev, sample)
    out["touch_offset_ms"], out["touch_share"] = touch["best_offset_ms"], touch["share_at_best"]
    start = int(b[0]) * MS - int(b[0]) * MS % (100 * MS) + 100 * MS
    grid = np.arange(start, int(b[-1]) * MS, 100 * MS)
    xc = xcorr_lag(book_ev, "mid", trades_ev, "price", grid)
    out["xcorr_lag_ms"], out["xcorr_corr"] = xc["lag_ms"], xc["corr_at_lag"]
    problems = []
    if out["book"]["backward"] or out["trades"]["backward"]:
        problems.append(f"stamps go backwards ({out['book']['backward']} in the book, "
                        f"{out['trades']['backward']} in the trades)")  # fmt: skip
    if not out["book"]["ok"] and not out["book"]["backward"]:
        problems.append(f"the book looks batched: median gap {out['book']['median_gap_ms']:g} "
                        f"ms, {out['book']['share_max']:.0%} of gaps near "
                        f"{out['book']['bin_max_ms']} ms")  # fmt: skip
    if not out["trades"]["ok"] and not out["trades"]["backward"]:
        problems.append(f"the trades look batched: {out['trades']['share_max']:.0%} of gaps "
                        f"near {out['trades']['bin_max_ms']} ms")  # fmt: skip
    if abs(out["touch_offset_ms"]) > CLOCK_FAIL_MS:
        problems.append(f"trades match the touch best {out['touch_offset_ms']} ms off their "
                        "stamp")  # fmt: skip
    if not np.isfinite(out["xcorr_lag_ms"]) or abs(out["xcorr_lag_ms"]) > CLOCK_FAIL_MS:
        problems.append(f"book and trades move together {out['xcorr_lag_ms']} ms apart")
    if problems:
        raise BuildError(f"{name}: " + "; ".join(problems))
    out["warnings"] = []
    if out["event_lag_median_ms"] > 50:
        out["warnings"].append(f"event_time trails transaction_time by a median "
                               f"{out['event_lag_median_ms']:g} ms")  # fmt: skip
    return out


def _events(frame: pl.DataFrame):
    """Raw rows as :class:`~blinded.panel.Events`, stamps in ns, no delay."""
    from blinded.panel import Events

    at = pl.from_epoch(frame["t"], time_unit="ms").dt.cast_time_unit("ns")
    cols = {"ts": at, "available_at": at}
    if "bid" in frame.columns:
        cols |= {"bid": frame["bid"], "ask": frame["ask"]}
    else:
        cols |= {"price": frame["price"],
                 "side": frame["side"].replace_strict({"B": 1.0, "S": -1.0})}  # fmt: skip
    return Events(pl.DataFrame(cols))


# -- thinning -------------------------------------------------------------------


def read_points(start_ns: int, end_ns: int, step_ns: int, offsets_ns, origin_ns: int = 0):
    """Every stamp ``g + o`` in ``[start_ns, end_ns)``: ``g`` a grid point, ``origin_ns``
    plus a whole number of steps, and ``o`` one of ``offsets_ns``.

    Examples
    --------
    >>> read_points(0, 1000, 500, [-50, 0, 50]).tolist()
    [0, 50, 450, 500, 550, 950]
    """
    offsets = np.asarray(sorted(offsets_ns), dtype=np.int64)
    lo = (start_ns - origin_ns - int(offsets[-1])) // step_ns
    hi = -((origin_ns + int(offsets[0]) - end_ns) // step_ns)
    g = origin_ns + step_ns * np.arange(lo, hi + 1, dtype=np.int64)
    points = np.unique((g[:, None] + offsets[None, :]).ravel())
    return points[(points >= start_ns) & (points < end_ns)]


def thin_book(ts_ns: np.ndarray, points_ns: np.ndarray, weight=None) -> tuple:
    """The rows an as-of read at each point returns, and how many raw rows each stands for.

    ``ts_ns`` is non-decreasing. The row a read at ``p`` returns is the last
    one stamped at or before ``p``. A kept row's ``n`` counts it and every
    dropped row since the kept row before it; ``weight`` gives a row its own
    count (a row carried over from the day before stands for the rows after
    that day's last kept one).

    Examples
    --------
    Rows at 0, 10, 20, 30 and 40 read at 15 and 40: rows 1 and 4 are kept, standing for
    rows 0-1 and 2-4.

    >>> idx, n = thin_book(np.array([0, 10, 20, 30, 40]), np.array([15, 40]))
    >>> idx.tolist(), n.tolist()
    ([1, 4], [2, 3])
    """
    idx = np.unique(np.searchsorted(ts_ns, points_ns, side="right") - 1)
    idx = idx[idx >= 0]
    w = np.ones(len(ts_ns), dtype=np.int64) if weight is None else np.asarray(weight)
    cum = np.cumsum(w)
    return idx, np.diff(np.r_[0, cum[idx]])


def thin_trades(trades: pl.DataFrame, step_ns: int, read_ns: int, origin_ns: int = 0):
    """Trades summed per side between the panel's reads, at ``origin + k step + read_ns``.

    ``trades`` has ``ts`` in ns, ``side``, ``price`` and ``qty``. For each
    step and side, one row at the stamp of that side's last trade after one
    read and before the next, ``qty`` the side's total, ``price`` its
    volume-weighted price and ``n`` its number of trades. Trades stamped on a
    read itself get a row of their own, so a window that starts or ends there
    counts them as the raw rows would. Rows sharing a stamp keep the order of
    their last trades, so the last row up to any read has the side of the last
    trade.

    Examples
    --------
    Reads every 100 at 50, 150, ...: two buys and a sell before 50, a buy on 150.

    >>> t = pl.DataFrame({"ts": [10, 20, 30, 150], "side": ["B", "S", "B", "B"],
    ...                   "price": [10.0, 9.0, 12.0, 11.0], "qty": [1.0, 5.0, 3.0, 2.0]})
    >>> thin_trades(t, 100, 50).rows()
    [(20, 'S', 9.0, 5.0, 1), (30, 'B', 11.5, 4.0, 2), (150, 'B', 11.0, 2.0, 1)]
    """
    u = pl.col("ts") - origin_ns - read_ns
    return (
        trades.with_row_index("order")
        .with_columns(k=-((-u) // step_ns), on_read=(u % step_ns) == 0)
        .group_by("k", "on_read", "side", maintain_order=True)
        .agg(
            pl.col("ts").last(),
            pl.col("qty").sum(),
            (pl.col("price") * pl.col("qty")).sum().alias("notional"),
            pl.len().alias("n"),
            pl.col("order").last(),
        )
        .sort("ts", "order")
        .select(
            "ts",
            "side",
            (pl.col("notional") / pl.col("qty")).alias("price"),
            "qty",
            pl.col("n").cast(pl.Int64),
        )
    )


def _book_rows(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Book rows in the XRP layout, from ``ts`` in ns, the touch and ``n``."""
    bid, ask = pl.col("bid"), pl.col("ask")
    return frame.select(
        pl.col("ts").cast(pl.Datetime("ns")),
        pl.lit(symbol).alias("symbol"),
        pl.lit(None, pl.String).alias("side"),
        pl.lit(None, pl.Float64).alias("price"),
        pl.lit(None, pl.Float64).alias("qty"),
        bid.alias("bid0_px"),
        ask.alias("ask0_px"),
        pl.col("bid_qty").alias("bid0_qty"),
        pl.col("ask_qty").alias("ask0_qty"),
        ((bid + ask) / 2).alias("mid"),
        pl.lit(False).alias("is_trade"),
        pl.col("n").cast(pl.Int64),
    )


def _trade_rows(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Trade rows in the XRP layout, the book columns empty, from ``ts`` in ns and ``n``."""
    none = pl.lit(None, pl.Float64)
    return frame.select(
        pl.col("ts").cast(pl.Datetime("ns")),
        pl.lit(symbol).alias("symbol"),
        pl.col("side"),
        pl.col("price"),
        pl.col("qty"),
        *(none.alias(c) for c in (*_BOOK, "mid")),
        pl.lit(True).alias("is_trade"),
        pl.col("n").cast(pl.Int64),
    )


@dataclass
class Carry:
    """A symbol's last raw book row of the day before, if no read kept it, and the raw rows
    since the last kept one, which it stands for."""

    row: pl.DataFrame | None = None
    pending: int = 0


def thin_symbol_day(
    book: pl.DataFrame,
    trades: pl.DataFrame,
    symbol: str,
    day: Date,
    lattice: dict,
    carry: Carry,
    last_day: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, Carry]:
    """One symbol-day thinned to the read lattice, in the XRP layout plus ``n``.

    ``lattice`` holds ``origin_ns`` (the first grid point), ``step_ns``,
    ``offsets_ns`` (the kept stamps around each grid point) and ``read_ns``
    (where the panel reads, minus the feed delay). The book keeps, for every
    kept stamp in the day, the last row at or before it, with the last raw row
    of the day before in front when no read of that day kept it. On the last
    day the last raw row is kept too, so the file ends where the raw one does.
    """
    start = int(pd.Timestamp(day).value)
    rows = book.select((pl.col("t").cast(pl.Int64) * MS).alias("ts"), "bid", "ask", "bid_qty",
                       "ask_qty").with_columns(w=pl.lit(1, pl.Int64))  # fmt: skip
    if carry.row is not None:
        rows = pl.concat([carry.row.with_columns(w=pl.lit(carry.pending, pl.Int64)), rows])
    ts = rows["ts"].to_numpy()
    points = read_points(start, start + 86_400 * 10**9, lattice["step_ns"], lattice["offsets_ns"],
                         lattice["origin_ns"])  # fmt: skip
    weight = rows["w"].to_numpy()
    idx, n = thin_book(ts, points, weight)
    total = int(weight.sum())
    after = total - int(np.cumsum(weight)[idx[-1]]) if len(idx) else total
    if last_day and after:
        idx, n = np.r_[idx, len(ts) - 1], np.r_[n, after]
        after = 0
    next_carry = Carry()
    if after:
        next_carry = Carry(rows.slice(len(ts) - 1, 1).drop("w"), after)
    kept = rows.drop("w")[idx].with_columns(n=pl.Series(n, dtype=pl.Int64))
    tr = trades.select((pl.col("t").cast(pl.Int64) * MS).alias("ts"), "side", "price", "qty")
    summed = thin_trades(tr, lattice["step_ns"], lattice["read_ns"], lattice["origin_ns"])
    return _book_rows(kept, symbol), _trade_rows(summed, symbol), next_carry


def identity_check(
    book: pl.DataFrame,
    trades: pl.DataFrame,
    kept_book: pl.DataFrame,
    kept_trades: pl.DataFrame,
    brief,
    hours: float = 1.0,
) -> dict:
    """Build the panel from the raw rows and from the kept ones over the first ``hours`` and
    check they agree; raises :class:`BuildError` where they do not.

    Every panel column must be equal except the last trade's ``price`` and
    ``qty``, which the thinned file holds as the step's per-side VWAP and
    total. The summed volumes and notionals are equal to a relative 1e-9: the
    same trades added in another order. And the book's bid and ask must agree
    at every fill time, with and without the latency.
    """
    from blinded.panel import Panel, load_events

    symbol = kept_book["symbol"][0]
    origin = int(kept_book["ts"].dt.epoch("ns").min()) // brief.grid.step.value
    origin *= brief.grid.step.value
    end = origin + int(hours * 3600e9)
    raw_book = book.select((pl.col("t").cast(pl.Int64) * MS).alias("ts"), "bid", "ask",
                           "bid_qty", "ask_qty", n=pl.lit(1))  # fmt: skip
    raw_trades = trades.select((pl.col("t").cast(pl.Int64) * MS).alias("ts"), "side", "price",
                               "qty", n=pl.lit(1))  # fmt: skip
    raw = pl.concat([_book_rows(raw_book, symbol), _trade_rows(raw_trades, symbol)])
    thin = pl.concat([kept_book, kept_trades])
    within = pl.col("ts").dt.epoch("ns") < end
    raw, thin = raw.filter(within), thin.filter(within)
    step = brief.grid.step.value
    # a fill at g + L reads the book at stamp g + L, which must fall inside the hour
    grid = np.arange(origin + step, end - brief.latency.value, step, dtype=np.int64)
    panels = []
    for frame in (raw, thin):
        events = load_events(brief, frame.sort("ts", maintain_order=True))
        panels.append((Panel.build(events, grid, brief.data.tz, step), events))
    (p_raw, e_raw), (p_thin, e_thin) = panels
    vague = {f"{symbol}.price", f"{symbol}.qty"}
    columns = [c for c in p_raw.frame.columns if c not in vague]
    if set(p_raw.frame.columns) != set(p_thin.frame.columns):
        raise BuildError(f"thinning identity: the panels have different columns for {symbol}")
    for col in columns:
        a, b = p_raw.frame[col], p_thin.frame[col]
        if col.split(".")[-1] in SUMMED:
            # the same trades added in another order, and price x qty from a VWAP: equal to
            # rounding, not to the bit
            x, y = (s.cast(pl.Float64).fill_null(np.nan).to_numpy() for s in (a, b))
            same = np.isclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            same = (a == b).fill_null(a.is_null() & b.is_null()).to_numpy()
        if not same.all():
            bad = int(np.flatnonzero(~same)[0])
            raise BuildError(f"thinning identity: {col} differs at {p_raw.time(bad)}")
    ev_raw, ev_thin = e_raw[("book", symbol)], e_thin[("book", symbol)]
    reads = 0
    for latency in (brief.latency.value, 0):
        at = grid + latency + ev_raw.delay_ns
        for side in ("bid", "ask"):
            if not np.array_equal(ev_raw.asof(side, at), ev_thin.asof(side, at), equal_nan=True):
                raise BuildError(f"thinning identity: the {side} at the fills differs ({symbol})")
            reads += len(at)
    return {"symbol": symbol, "grid_points": len(grid), "columns": len(columns),
            "reads": reads, "raw_rows": raw.height, "kept_rows": thin.height}  # fmt: skip


def binance_brief(data_file: str, symbols: list[str], start: Date, days: int, step: str,
                  latency_ms: float, delay_ms: float) -> dict:  # fmt: skip
    """The brief of a thinned Binance file: four perpetuals, three families, exchange clock.

    The first half of the symbols lead the second half, and the pairs are
    the second of each half over the first (ETH over BTC, XRP over SOL with
    the default four). The grid starts at midnight of the first day, so the
    lattice ``blinded init`` builds is the one the file was thinned to.
    """
    inst = [f"{s}_PERP" for s in symbols]
    half = max(1, len(inst) // 2)
    lead, follow = inst[:half], inst[half:] or inst[:1]
    pairs = [[inst[i + 1], inst[i]] for i in range(0, len(inst) - 1, 2)]
    families = {
        "directional": {"instruments": inst, "share": 0.3},
        "lead_lag": {"leaders": lead, "followers": follow, "share": 0.3},
        "pairs": {"pairs": pairs, "share": 0.4,
                  "defaults": {"window": "2h", "every": "30m", "z_in": 2.0, "z_out": 0.5}},
    }  # fmt: skip
    offsets = sorted({-float(delay_ms), 0.0, float(latency_ms)})
    return {
        "blinded": 0,
        "name": data_file.removesuffix(".parquet"),
        "data": {
            "path": data_file,
            "format": "parquet",
            "time": {"column": "ts", "unit": "ns", "tz": "UTC", "clock": "exchange"},
            "instrument_column": "symbol",
            "streams": {
                "book": {
                    "filter": "is_trade == false",
                    "fields": {
                        "bid": "bid0_px",
                        "ask": "ask0_px",
                        "bid_qty": "bid0_qty",
                        "ask_qty": "ask0_qty",
                        "n": "n",
                    },
                },
                "trades": {
                    "filter": "is_trade == true",
                    "fields": {"price": "price", "qty": "qty", "side": "side", "n": "n"},
                    "side_values": {"buy": "B", "sell": "S"},
                },
            },
            "availability": {"default_delay_ms": float(delay_ms), "overrides": []},
            "thinned": {"step": step, "offsets_ms": [o if o % 1 else int(o) for o in offsets]},
        },
        "target": {
            "instrument": inst[-1],
            "price": "mid",
            "horizons": ["500ms", "1s", "10s", "60s"],
            "primary_horizon": "10s",
        },
        "grid": {"step": step, "start": f"{start.isoformat()}T00:00:00Z", "warmup": "5min"},
        "execution": {
            "latency_ms": float(latency_ms),
            "taker_fee_bps": 7.5,
            "maker_fee_bps": 0.0,
            "max_position": 1.0,
            "min_rebalance": 0.1,
        },
        "splits": {"scheme": "time", "fractions": [0.5, 0.25, 0.25], "embargo": "5min"},
        "stats": {"se": "both", "block": "5min", "n_boot": 1000, "seed": 20241128},
        "families": families,
    }


def estimate_rows(symbols: int, days: int, step: str, kept_stamps: int) -> int:
    """Rows a thinned file of ``symbols`` x ``days`` should hold, from the lattice and the
    rows per step measured on 27 to 29 March 2024.

    Examples
    --------
    >>> estimate_rows(4, 3, "500ms", 3), estimate_rows(4, 4, "500ms", 3)
    (7527168, 10036224)
    """
    steps = 86_400 * 10**9 // pd.Timedelta(step).value
    per_step = BOOK_ROWS_PER_STAMP * kept_stamps + TRADE_ROWS_PER_STEP
    return int(symbols * days * steps * per_step)


@dataclass
class Built:
    """What :func:`binance` wrote, and the numbers it checked on the way."""

    data: Path
    brief: Path
    notes: Path
    rows: int
    days: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    fine: dict = field(default_factory=dict)


def binance(
    start: Date | str,
    days: int,
    out: str | Path,
    symbols: list[str] | tuple[str, ...] = SYMBOLS,
    step: str = "500ms",
    latency_ms: float = 50.0,
    delay_ms: float = 50.0,
    keep_raw: bool = False,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> Built:
    """Download days of Binance USD-M perpetuals, check them, thin them and write one parquet.

    Parameters
    ----------
    start
        The first day, ``YYYY-MM-DD``.
    days
        How many consecutive days.
    out
        Directory for ``raw/`` (the zips, deleted at the end unless ``keep_raw``),
        ``parts/`` (one thinned symbol-day each), the parquet, its brief and
        ``DATA-raw.md``.
    step, latency_ms, delay_ms
        The grid step, and the latency and feed delay the brief sets; together
        they fix the stamps the file keeps (:func:`read_points`).
    force
        Build even when the rows estimated from the lattice pass 10 million.

    Each symbol-day is read from its zip, checked (:func:`raw_checks`), thinned
    (:func:`thin_symbol_day`) and written to ``parts/`` before the next is
    read, so memory holds one raw day at a time; the first hour of the first
    symbol-day is checked panel against panel (:func:`identity_check`), and
    the first day's raw books give the fine lead-lag curves of ``DATA-raw.md``.
    """
    from blinded import brief as briefs

    start = Date.fromisoformat(start) if isinstance(start, str) else start
    symbols = [s.upper() for s in symbols]
    out = Path(out).expanduser()
    stamps = sorted({-float(delay_ms), 0.0, float(latency_ms)})
    estimate = estimate_rows(len(symbols), days, step, len(stamps))
    note = "over" if estimate > ROWS_GUIDE else "under"
    log(f"about {estimate:,} rows for {len(symbols)} symbols x {days} days at {step}, "
        f"{note} the {ROWS_GUIDE:,} guideline")  # fmt: skip
    if estimate > ROWS_LIMIT and not force:
        raise BuildError(f"about {estimate:,} rows is over the {ROWS_LIMIT:,} limit one judgement "
                         "fits in; use fewer days or symbols, or --force")  # fmt: skip
    raw, parts = out / "raw", out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    short = "-".join(s.lower().removesuffix("usdt") for s in symbols)
    name = f"binance-um-{short}-{start.isoformat()}-{days}d-{step}"
    raw_brief = binance_brief(f"{name}.parquet", symbols, start, days, step, latency_ms, delay_ms)
    brief = briefs.parse(raw_brief, base=out)
    lattice = {"origin_ns": int(pd.Timestamp(start).value), "step_ns": brief.grid.step.value,
               "offsets_ns": [int(o * MS) for o in stamps],
               "read_ns": -int(delay_ms * MS)}  # fmt: skip
    built = Built(out / f"{name}.parquet", out / f"{name}.brief.yaml", out / "DATA-raw.md", 0)
    carry = {s: Carry() for s in symbols}
    first_books: dict[str, pl.DataFrame] = {}
    files = []
    for i in range(days):
        day = start + timedelta(days=i)
        built.days[day] = {}
        for sym in symbols:
            zips = {kind: fetch(kind, sym, day, raw) for kind in HEADERS}
            files += list(zips.values())
            book = book_frame(read_archive(zips["bookTicker"], "bookTicker"))
            trades = trades_frame(read_archive(zips["trades"], "trades"))
            checks = raw_checks(book, trades, f"{sym} {day}")
            checks["sha256"] = {k: (raw / f"{p.name}.CHECKSUM").read_text().split()[0]
                                for k, p in zips.items()}  # fmt: skip
            kept_book, kept_trades, carry[sym] = thin_symbol_day(
                book, trades, f"{sym}_PERP", day, lattice, carry[sym], last_day=i == days - 1
            )
            checks["kept_book"], checks["kept_trades"] = kept_book.height, kept_trades.height
            if not built.identity:
                built.identity = identity_check(book, trades, kept_book, kept_trades, brief)
                log(f"thinning identity holds on the first hour of {sym} {day}: "
                    f"{built.identity['grid_points']:,} grid points, "
                    f"{built.identity['columns']} columns")  # fmt: skip
            if i == 0:
                first_books[f"{sym}_PERP"] = book.select("t", "bid", "ask")
            pl.concat([kept_book, kept_trades]).write_parquet(parts / f"{sym}-{day}.parquet")
            built.days[day][sym] = checks
            log(f"{sym} {day}: {book.height:,} book rows and {trades.height:,} trades, kept "
                f"{kept_book.height:,} and {kept_trades.height:,}")  # fmt: skip
            del book, trades, kept_book, kept_trades
    built.fine = fine_lead_lag(first_books, brief, start)
    del first_books
    frame = pl.concat([pl.read_parquet(parts / f"{s}-{start + timedelta(days=i)}.parquet")
                       for i in range(days) for s in symbols])  # fmt: skip
    frame = frame.sort("ts", maintain_order=True).with_columns(pl.col("side").cast(pl.Categorical))
    frame.write_parquet(built.data)
    built.rows = frame.height
    built.brief.write_text(yaml.safe_dump(raw_brief, sort_keys=False))
    built.notes.write_text(raw_notes(built, brief, symbols, start, days))
    log(f"{built.rows:,} rows, {'over' if built.rows > ROWS_GUIDE else 'under'} the "
        f"{ROWS_GUIDE:,} guideline")  # fmt: skip
    for path in parts.glob("*.parquet"):
        path.unlink()
    parts.rmdir()
    if not keep_raw:
        for path in files:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".CHECKSUM").unlink(missing_ok=True)
        if not any(raw.iterdir()):
            raw.rmdir()
    return built


def fine_lead_lag(books: dict[str, pl.DataFrame], brief, day: Date) -> dict:
    """The lead-lag curve at 10 ms steps over +-500 ms of every leader and follower, on one
    day's raw books: 100 ms returns on a 100 ms grid, the follower read as-of each lag.

    The fill is the brief's latency plus its feed delay after the decision,
    ``first_after_fill`` the follower's first whole 100 ms step after it.
    """
    from blinded.diagnostics import lead_lag_curve, lead_lag_summary

    family = brief.family("lead_lag")
    if family is None:
        return {}
    step = 100 * MS
    origin = int(pd.Timestamp(day).value)
    grid = np.arange(origin + step, origin + 86_400 * 10**9, step, dtype=np.int64)
    lags = list(range(-500 * MS, 500 * MS + 1, 10 * MS))
    fill = brief.latency.value + int(brief.data.default_delay_ms * MS)
    events = {}
    out = {}
    for leader in family.leaders:
        for follower in family.followers:
            if leader == follower or leader not in books or follower not in books:
                continue
            for sym in (leader, follower):
                if sym not in events:
                    events[sym] = _events(books[sym])
            corr = lead_lag_curve(events[leader], events[follower], grid, step, lags)
            out[(leader, follower)] = {"lags_ms": [lag // MS for lag in lags], "corr": corr,
                                       **lead_lag_summary(lags, corr, fill, step)}  # fmt: skip
    return out


def raw_notes(built: Built, brief, symbols: list[str], start: Date, days: int) -> str:
    """``DATA-raw.md``: the raw-file checks per symbol-day and the fine lead-lag curves.

    A header, then one ``## YYYY-MM-DD`` section per day, so that
    ``blinded init`` can append only the days the agent may see.
    """
    from blinded.diagnostics import _cell

    thin = brief.data.thinned
    ident = built.identity
    stamps = ", ".join(_offset_text(o) for o in thin.offsets)
    lines = [
        f"# The raw files behind {built.data.name}",
        "",
        f"Built by `blinded demo binance` from data.binance.vision: USD-M perpetuals "
        f"{', '.join(symbols)}, {days} day{'s' if days > 1 else ''} from {start}, top of book "
        "(`bookTicker`) and "
        "trades. Stamps are the exchange's: `transaction_time` for the book and `time` for "
        "trades, the matching engine's clock in ms; `event_time`, when the message left, is "
        "not used. Each zip matched its `.CHECKSUM` (SHA-256 below).",
        "",
        f"The brief's latency is {brief.latency / pd.Timedelta('1ms'):g} ms of order and "
        f"{brief.data.default_delay_ms:g} ms of feed: a decision at g sees stamps up to g - "
        f"{brief.data.default_delay_ms:g} ms and its order fills at stamp g + "
        f"{brief.latency / pd.Timedelta('1ms'):g} ms. The file keeps, for every grid point g "
        f"of the {durations.label(thin.step)} grid, the last book row at or before {stamps}, "
        "with `n` the raw rows each stands for, and the trades summed per side between the "
        "panel's reads.",
        "",
        f"Thinning identity, on the first hour of {ident['symbol']}: a panel from the "
        f"{ident['raw_rows']:,} raw rows and one from the {ident['kept_rows']:,} kept rows "
        f"agree at all {ident['grid_points']:,} grid points in {ident['columns']} columns "
        "(all but the last trade's price and size), and the bid and ask agree at all "
        f"{ident['reads']:,} fill reads, at the brief's latency and at zero.",
        "",
        "Checks on each symbol-day: stamps never go backwards; the book is not batched "
        f"(median gap between distinct stamps under {BATCH_MEDIAN_MS:g} ms, no 10 ms bin from "
        f"20 to 1000 ms holding over {BATCH_BIN_SHARE:.0%} of gaps) and neither are the trades "
        "(the bin test only: trades are sparse); trades print at the touch of the book best "
        f"within {CLOCK_FAIL_MS} ms of their stamp (`touch`, on up to {TOUCH_SAMPLE:,} trades "
        "spread over the day); and the mid and the trade price move together within "
        f"{CLOCK_FAIL_MS} ms (`mid_vs_trade`, 100 ms returns searched over +-500 ms).",
        "",
    ]
    for day, per in built.days.items():
        lines += [f"## {day}", ""]
        head = ["symbol", "book rows", "trades", "book median gap ms", "largest 10 ms bin",
                "event - transaction ms (median, p99)", "touch ms (share)",
                "mid_vs_trade ms (corr)", "kept book rows", "kept trades", "book sha256",
                "trades sha256"]  # fmt: skip
        lines += ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
        for sym, c in per.items():
            cells = [
                sym,
                f"{c['book_rows']:,}",
                f"{c['trade_rows']:,}",
                _cell(c["book"]["median_gap_ms"]),
                f"{c['book']['share_max']:.1%} at {c['book']['bin_max_ms']} ms",
                f"{c['event_lag_median_ms']:g}, {c['event_lag_p99_ms']:g}",
                f"{c['touch_offset_ms']} ({c['touch_share']:.2f})",
                f"{_cell(c['xcorr_lag_ms'])} ({c['xcorr_corr']:.2f})",
                f"{c['kept_book']:,}",
                f"{c['kept_trades']:,}",
                f"`{c['sha256']['bookTicker'][:16]}`",
                f"`{c['sha256']['trades'][:16]}`",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        warnings = [f"- {sym}: {w}" for sym, c in per.items() for w in c.get("warnings", [])]
        lines += ["", *warnings] if warnings else []
        lines.append("")
        if day == start and built.fine:
            lines += _fine_md(built.fine, brief)
    return "\n".join(lines) + "\n"


def _fine_md(fine: dict, brief) -> list[str]:
    fill = brief.latency / pd.Timedelta("1ms") + brief.data.default_delay_ms
    pairs = list(fine)
    summary = ["| leader | follower | lag at the peak, ms | C at the peak | C at 0 | "
               "C on the first step after the fill | best C from there on | usable by a taker |",
               "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for leader, follower in pairs:
        f = fine[(leader, follower)]
        zero = f["corr"][f["lags_ms"].index(0)]
        summary.append(
            f"| {leader} | {follower} | {_num_text(f['lag_at_peak_ms'])} | "
            f"{_num_text(f['c_at_peak'])} | {_num_text(zero)} | {_num_text(f['c_after_fill'])} | "
            f"{_num_text(f['best_after_fill'])} | {'yes' if f['usable_by_taker'] else 'no'} |"
        )
    curve = ["| lag, ms | " + " | ".join(f"{a} then {b}" for a, b in pairs) + " |",
             "|---|" + "---|" * len(pairs)]  # fmt: skip
    for i, lag in enumerate(fine[pairs[0]]["lags_ms"]):
        cells = [_num_text(fine[p]["corr"][i]) for p in pairs]
        curve.append(f"| {lag} | " + " | ".join(cells) + " |")
    return [
        "### Lead-lag inside half a second, on this day's raw books",
        "",
        "The half-second grid cannot see a lead shorter than a step, so it is measured "
        "here, once, before thinning: the correlation of the leader's last 100 ms log return "
        "on a 100 ms grid with the follower's 100 ms return ending a lag later, at 10 ms "
        f"steps. The fill is {fill:g} ms after the decision, so the first whole 100 ms step "
        f"after it ends at {fill + 100:g} ms. This day is all in train.",
        "",
        *summary,
        "",
        *curve,
        "",
    ]


def _num_text(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _offset_text(o: pd.Timedelta) -> str:
    ms = o / pd.Timedelta("1ms")
    return "g" if ms == 0 else f"g {'+' if ms > 0 else '-'} {abs(ms):g} ms"


def notes_until(text: str, until: pd.Timestamp | None) -> str:
    """``DATA-raw.md`` without the day or month sections that end after ``until``; all of it
    for None.

    A section is headed ``## YYYY-MM-DD`` (a day) or ``## YYYY-MM`` (a month).

    Examples
    --------
    >>> text = "# head\\n\\n## 2024-03-27\\na\\n## 2024-03-28\\nb\\n"
    >>> notes_until(text, pd.Timestamp("2024-03-28 12:00", tz="UTC"))
    '# head\\n\\n## 2024-03-27\\na\\n'
    >>> notes_until("## 2024-03\\na\\n## 2024-04\\nb\\n", pd.Timestamp("2024-04-30", tz="UTC"))
    '## 2024-03\\na\\n'
    """
    if until is None:
        return text
    keep, out = True, []
    for line in text.splitlines(keepends=True):
        end = _section_end(line[3:].strip()) if line.startswith("## ") else None
        if end is not None:
            keep = end <= until
        if keep:
            out.append(line)
    return "".join(out)


def _section_end(text: str) -> pd.Timestamp | None:
    """Where a ``YYYY-MM-DD`` or ``YYYY-MM`` heading's period ends, or None for another one."""
    try:
        return pd.Timestamp(Date.fromisoformat(text), tz="UTC") + pd.Timedelta("1D")
    except ValueError:
        pass
    try:
        first = Date.fromisoformat(f"{text}-01")
    except ValueError:
        return None
    return pd.Timestamp(first, tz="UTC") + pd.offsets.MonthBegin(1)


# -- Binance klines: bars as rows the judge reads -------------------------------

VISION = "https://data.binance.vision/data"
KLINE_HEADER = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                "taker_buy_volume,taker_buy_quote_volume,ignore")  # fmt: skip
KLINE_COLUMNS = tuple(KLINE_HEADER.split(","))
KLINE_LAYOUT = ("ts", "symbol", "kind", "bid0_px", "ask0_px", "mid", "side", "price", "qty", "n")
MICROSECONDS = 10**14  # an open_time above this is in microseconds: 2025-01 onward, some files
MISSING_LIMIT = 0.001  # the share of a symbol-month's bars that may be missing
DOWNLOAD_CAP = 10 * 10**9
TREND_SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT")
TREND_START, TREND_END, TREND_COST_DAY = Date(2023, 5, 17), Date(2026, 10, 1), Date(2023, 5, 16)
TREND_PERIODS_LO = pd.Timedelta("14D")  # trend following's shortest typical holding


def kline_url(symbol: str, interval: str, month: str, market: str = "futures/um") -> str:
    """The archive's monthly kline zip, ``.../BTCUSDT/1h/BTCUSDT-1h-2023-05.zip``."""
    return f"{VISION}/{market}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def book_ticker_url(symbol: str, day: Date) -> str:
    """The archive's USD-M ``bookTicker`` zip of one day."""
    return f"{ARCHIVE}/bookTicker/{symbol}/{archive_name('bookTicker', symbol, day)}"


def months(start: Date, end: Date) -> list[str]:
    """``YYYY-MM`` of every month holding a day in ``[start, end)``.

    Examples
    --------
    >>> months(Date(2023, 5, 17), Date(2023, 8, 1))
    ['2023-05', '2023-06', '2023-07']
    """
    out, d = [], Date(start.year, start.month, 1)
    while d < end:
        out.append(f"{d.year}-{d.month:02d}")
        d = Date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return out


def head_sizes(urls: list[str], head: Callable[[str], int | None] | None = None) -> dict:
    """``Content-Length`` of every URL by an HTTP HEAD, before a byte is fetched.

    Raises :class:`BuildError` naming every file the archive does not have, so
    a missing symbol or day stops the build instead of being skipped.
    """
    head = head or _head
    sizes = {url: head(url) for url in urls}
    missing = [url.rsplit("/", 1)[1] for url, size in sizes.items() if size is None]
    if missing:
        raise BuildError(f"not on data.binance.vision: {', '.join(missing)}; nothing was "
                         "downloaded")  # fmt: skip
    return sizes


def _head(url: str) -> int | None:
    import urllib.error

    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            return int(resp.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def read_klines(path: Path, quirks: list[str] | None = None) -> pl.DataFrame:
    """The bars of one kline zip, ``open_time`` and ``close_time`` in ms.

    Two quirks of the archive: older futures files have no header row (the
    first field parses as an integer), and files from 2025 on may stamp in
    microseconds (a stamp above :data:`MICROSECONDS`), which are divided down
    to ms. Any other header fails. The quirks found are appended to ``quirks``.
    """
    with zipfile.ZipFile(path) as zf:
        (member,) = zf.infolist()
        data = zf.read(member)
    first = data[: data.index(b"\n")].decode().strip()
    headed = not first.split(",")[0].strip().isdigit()
    if headed and first != KLINE_HEADER:
        raise BuildError(f"{Path(path).name}: the header is {first!r}, expected "
                         f"{KLINE_HEADER!r}")  # fmt: skip
    ints = ("open_time", "close_time", "count")
    frame = pl.read_csv(io.BytesIO(data), has_header=headed, new_columns=list(KLINE_COLUMNS),
                        schema_overrides={c: pl.Int64 if c in ints else pl.Float64
                                          for c in KLINE_COLUMNS})  # fmt: skip
    ms = {c: pl.when(pl.col(c) > MICROSECONDS).then(pl.col(c) // 1000).otherwise(pl.col(c))
          for c in ("open_time", "close_time")}  # fmt: skip
    if quirks is not None:
        quirks += [] if headed else ["no header"]
        quirks += ["microseconds"] if (frame["open_time"] > MICROSECONDS).any() else []
    keep = ("open_time", "close_time", "close", "volume", "quote_volume", "count",
            "taker_buy_volume")  # fmt: skip
    return frame.with_columns(**ms).select(keep)


def bar_rows(
    bars: pl.DataFrame, symbol: str, half_spread_bps: float, interval: str
) -> pl.DataFrame:
    """Each bar as one book row and two trade rows, sell then buy, at its close stamp.

    The stamp is ``open_time + interval - 1 ms``. The book row has ``mid`` the
    close and ``bid``, ``ask`` the close times ``1 -+ hs / 1e4``, with no sizes.
    The sell row has ``qty = volume - taker_buy_volume``, the buy row
    ``taker_buy_volume``; both are priced at the bar's VWAP, ``quote_volume /
    volume``, or the close when nothing traded. ``n`` splits the trade count by
    volume, ``round(count x taker_buy_volume / volume)`` buys and the rest
    sells: an estimate, since the archive gives the count and not its split.

    Examples
    --------
    >>> bars = pl.DataFrame({"open_time": [0], "close_time": [59_999], "close": [100.0],
    ...                      "volume": [10.0], "quote_volume": [1000.0], "count": [5],
    ...                      "taker_buy_volume": [6.0]})
    >>> rows = bar_rows(bars, "X", 10.0, "1m").select("kind", "side", "price", "qty", "n")
    >>> rows.rows()[1:]
    [('trades', 'S', 100.0, 4.0, 2), ('trades', 'B', 100.0, 6.0, 3)]
    """
    step = durations.to_ns(interval) // MS
    traded = pl.col("volume") > 0
    vwap = pl.when(traded).then(pl.col("quote_volume") / pl.col("volume")).otherwise("close")
    buys = pl.when(traded).then((pl.col("count") * pl.col("taker_buy_volume") / pl.col("volume"))
                                .round(0)).otherwise(0).cast(pl.Int64)  # fmt: skip
    base = bars.select(
        ((pl.col("open_time") + step - 1) * MS).cast(pl.Datetime("ns")).alias("ts"),
        pl.lit(symbol).alias("symbol"),
        "close",
        vwap.alias("vwap"),
        (pl.col("volume") - pl.col("taker_buy_volume")).clip(lower_bound=0.0).alias("sell_qty"),
        pl.col("taker_buy_volume").alias("buy_qty"),
        buys.alias("n_buy"),
        (pl.col("count") - buys).alias("n_sell"),
    ).with_row_index("_bar")
    none = pl.lit(None, pl.Float64)
    hs = half_spread_bps / 1e4
    book = base.select("_bar", "ts", "symbol", kind=pl.lit("book"),
                       bid0_px=pl.col("close") * (1 - hs), ask0_px=pl.col("close") * (1 + hs),
                       mid=pl.col("close"), side=pl.lit(None, pl.String), price=none, qty=none,
                       n=pl.lit(None, pl.Int64), _order=pl.lit(0))  # fmt: skip
    sides = [
        base.select("_bar", "ts", "symbol", kind=pl.lit("trades"), bid0_px=none, ask0_px=none,
                    mid=none, side=pl.lit(side), price="vwap", qty=f"{key}_qty",
                    n=f"n_{key}", _order=pl.lit(order))
        for order, side, key in ((1, "S", "sell"), (2, "B", "buy"))
    ]  # fmt: skip
    rows = pl.concat([book, *sides]).sort("_bar", "_order")
    return rows.select(KLINE_LAYOUT)


def bar_gaps(bars: pl.DataFrame, interval: str, start: Date, end: Date) -> dict[str, dict]:
    """Bars expected, present and missing per month inside ``[start, end)``.

    A bar is expected at every whole ``interval`` from ``start``. Stamps off
    that grid, a duplicate, or a close time that is not ``open_time +
    interval - 1 ms`` raise :class:`BuildError`.
    """
    step = durations.to_ns(interval) // MS
    t0, t1 = (int(pd.Timestamp(d, tz="UTC").value // MS) for d in (start, end))
    t = bars["open_time"].to_numpy()
    if np.any((t - t0) % step):
        raise BuildError(f"bars open off the {interval} grid")
    if len(np.unique(t)) != len(t):
        raise BuildError("a bar appears twice")
    if not (bars["close_time"] - bars["open_time"] == step - 1).all():
        raise BuildError(f"a bar's close_time is not open_time + {interval} - 1 ms")
    out = {}
    for month in months(start, end):
        lo = max(t0, int(pd.Timestamp(f"{month}-01", tz="UTC").value // MS))
        nxt = pd.Timestamp(f"{month}-01", tz="UTC") + pd.offsets.MonthBegin(1)
        hi = min(t1, int(nxt.value // MS))
        expected = -(-(hi - lo) // step)
        present = int(((t >= lo) & (t < hi)).sum())
        out[month] = {"expected": expected, "present": present, "missing": expected - present}
    return out


def missing_check(symbol: str, gaps: dict[str, dict]) -> tuple[str, float]:
    """The month of :func:`bar_gaps` missing the largest share of its bars, and that share;
    raises :class:`BuildError` naming the symbol and month when it is over
    :data:`MISSING_LIMIT`."""
    worst = max(gaps, key=lambda m: gaps[m]["missing"] / gaps[m]["expected"])
    share = gaps[worst]["missing"] / gaps[worst]["expected"]
    if share > MISSING_LIMIT:
        raise BuildError(f"{symbol} {worst}: {gaps[worst]['missing']} of "
                         f"{gaps[worst]['expected']} bars missing, over "
                         f"{MISSING_LIMIT:.1%}")  # fmt: skip
    return worst, share


def half_spreads(symbols: list[str], day: Date, raw: Path, cache: Path,
                 log: Callable[[str], None] = print) -> dict:  # fmt: skip
    """The median half spread of each symbol over one ``bookTicker`` day, in bps of the mid.

    Every top-of-book row of the day counts once (:func:`blinded.markouts.half_spread_bps`).
    The result, with the rows and each zip's SHA-256, is cached as
    ``cache/bookTicker-DAY.json``, so a day measured once is not fetched again. Only
    the two price columns are parsed (:func:`touch_prices`).
    """
    from blinded.markouts import half_spread_bps

    path = cache / f"bookTicker-{day.isoformat()}.json"
    table = json.loads(path.read_text()) if path.is_file() else {"day": day.isoformat(),
                                                                 "symbols": {}}  # fmt: skip
    for sym in symbols:
        if sym in table["symbols"]:
            log(f"{sym} {day}: half spread {table['symbols'][sym]['half_spread_bps']:.4g} bps, "
                f"from {path.name}")  # fmt: skip
            continue
        zipped = fetch("bookTicker", sym, day, raw)
        bid, ask = touch_prices(zipped)
        two_sided = (bid > 0) & (ask > bid)
        table["symbols"][sym] = {
            "half_spread_bps": half_spread_bps(bid[two_sided], ask[two_sided]),
            "rows": len(bid), "two_sided_rows": int(two_sided.sum()),
            "sha256": (raw / f"{zipped.name}.CHECKSUM").read_text().split()[0],
            "zip_bytes": zipped.stat().st_size,
        }  # fmt: skip
        log(f"{sym} {day}: {len(bid):,} book rows, half spread "
            f"{table['symbols'][sym]['half_spread_bps']:.4g} bps")  # fmt: skip
        del bid, ask
        cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, indent=1) + "\n")
    return table


def touch_prices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The best bid and ask of every row of one ``bookTicker`` zip, its header checked.

    The same rows as :func:`read_archive` then :func:`book_frame`, with only
    the two price columns parsed, so a day of BTC (178 MB zipped) fits in
    memory beside the build.
    """
    with zipfile.ZipFile(path) as zf:
        (member,) = zf.infolist()
        data = zf.read(member)
    header = data[: data.index(b"\n")].decode().strip()
    if header != HEADERS["bookTicker"]:
        raise BuildError(f"{Path(path).name}: the header is {header!r}, expected "
                         f"{HEADERS['bookTicker']!r}")  # fmt: skip
    cols = ["best_bid_price", "best_ask_price"]
    frame = pl.read_csv(io.BytesIO(data), columns=cols,
                        schema_overrides=dict.fromkeys(cols, pl.Float64))  # fmt: skip
    del data
    return frame["best_bid_price"].to_numpy(), frame["best_ask_price"].to_numpy()


def four_figures(x: float) -> float:
    """``x`` to four significant figures, as the brief records a measured half spread.

    Examples
    --------
    >>> four_figures(0.0123456), four_figures(1.23456)
    (0.01235, 1.235)
    """
    return float(f"{x:.4g}")


def _bar_data(data_file: str, symbols: list[str], hs: dict[str, float], cost_day: Date) -> dict:
    """The ``data`` block of a bar file's brief: the kline layout and its cost model."""
    return {
        "path": data_file,
        "format": "parquet",
        "time": {"column": "ts", "unit": "ns", "tz": "UTC", "clock": "exchange"},
        "instrument_column": "symbol",
        "streams": {
            "book": {"filter": "kind == 'book'",
                     "fields": {"bid": "bid0_px", "ask": "ask0_px"}},
            "trades": {"filter": "kind == 'trades'",
                       "fields": {"price": "price", "qty": "qty", "side": "side", "n": "n"},
                       "side_values": {"buy": "B", "sell": "S"}},
        },
        "availability": {"default_delay_ms": 0, "overrides": []},
        "cost_model": {
            "kind": "half_spread",
            "fill": "next bar close",
            "source": f"data.binance.vision bookTicker {cost_day.isoformat()}",
            "half_spread_bps": {f"{s}_PERP": hs[s] for s in symbols},
        },
    }  # fmt: skip


def trend_brief(data_file: str, symbols: list[str], hs: dict[str, float], cost_day: Date,
                start: Date) -> dict:  # fmt: skip
    """The brief of the trend file: hourly bars of four perpetuals."""
    inst = [f"{s}_PERP" for s in symbols]
    return {
        "blinded": 0,
        "name": data_file.removesuffix(".parquet"),
        "data": _bar_data(data_file, symbols, hs, cost_day),
        "target": {"instrument": inst[-1], "price": "mid", "horizons": ["1d", "1w", "2w"],
                   "primary_horizon": "1w"},
        "grid": {"step": "1h", "start": f"{start.isoformat()}T00:00:00Z", "warmup": "4w"},
        "execution": {"latency_ms": 3_600_000, "taker_fee_bps": 5.0, "maker_fee_bps": 0.0,
                      "max_position": 1.0, "min_rebalance": 0.1},
        "splits": {"scheme": "time", "fractions": [0.5, 0.25, 0.25], "embargo": "3w"},
        "stats": {"se": "both", "block": "2w", "n_boot": 1000, "seed": 20241128},
        "archetypes": {
            "instruments": inst,
            "params": {"trend_following": {"lookback": "4w", "every": "1w"},
                       "pairs_trading": {"window": "1w", "every": "1d", "z_in": 2.0,
                                         "z_out": 0.5}},
        },
    }  # fmt: skip


def xs_brief(data_file: str, symbols: list[str], hs: dict[str, float], cost_day: Date,
             start: Date) -> dict:  # fmt: skip
    """The brief of the cross-section file: twenty perpetuals on 15-minute
    bars, the two basket archetypes and pairs trading at set parameters."""
    inst = [f"{s}_PERP" for s in symbols]
    return {
        "blinded": 0,
        "name": data_file.removesuffix(".parquet"),
        "data": _bar_data(data_file, symbols, hs, cost_day),
        "target": {"instrument": inst[-1], "price": "mid",
                   "horizons": ["15m", "1h", "4h", "1d"], "primary_horizon": "4h"},
        "grid": {"step": "15min", "start": f"{start.isoformat()}T00:00:00Z", "warmup": "1d"},
        "execution": {"latency_ms": 900_000, "taker_fee_bps": 5.0, "maker_fee_bps": 0.0,
                      "max_position": 1.0, "min_rebalance": 0.1},
        "splits": {"scheme": "time", "fractions": [0.5, 0.25, 0.25], "embargo": "2d"},
        "stats": {"se": "both", "block": "4h", "n_boot": 1000, "seed": 20241128},
        "archetypes": {
            "instruments": inst,
            "params": {"statistical_arbitrage": {"every": "4h", "top": 4, "lookback": "1d"},
                       "short_term_reversal": {"every": "4h", "top": 4, "lookback": "4h"},
                       "pairs_trading": {"window": "2d", "every": "4h", "z_in": 2.0,
                                         "z_out": 0.5}},
        },
    }  # fmt: skip


@dataclass(frozen=True)
class BarFile:
    """What a bar file is: its symbols and days, the cost day, the bar, and the holding period
    whose span rule the build asserts (``lo``, the shortest typical holding of ``holder``)."""

    prefix: str
    symbols: tuple[str, ...]
    start: Date
    end: Date
    cost_day: Date
    interval: str
    lo: pd.Timedelta
    holder: str
    command: str
    brief: Callable[..., dict]

    def name(self, symbols: list[str]) -> str:
        days = (self.end - self.start).days
        return f"{self.prefix}{len(symbols)}-{self.start.isoformat()}-{days}d-{self.interval}"


TREND = BarFile("binance-um-trend", TREND_SYMBOLS, TREND_START, TREND_END, TREND_COST_DAY,
                "1h", TREND_PERIODS_LO, "trend following", "blinded demo trend",
                trend_brief)  # fmt: skip
XS_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
              "AVAXUSDT", "LINKUSDT", "DOTUSDT", "ARBUSDT", "LTCUSDT", "TRXUSDT", "ATOMUSDT",
              "UNIUSDT", "NEARUSDT", "APTUSDT", "FILUSDT", "ETCUSDT", "BCHUSDT")  # fmt: skip
# short-term reversal's band starts at 4 h, statistical arbitrage's at 1 h: the longer rules
XS = BarFile("binance-um-xs", XS_SYMBOLS, Date(2024, 4, 1), Date(2025, 6, 25), Date(2024, 3, 27),
             "15m", pd.Timedelta("4h"), "short-term reversal", "blinded demo crosssection",
             xs_brief)  # fmt: skip


@dataclass
class BuiltBars:
    """What :func:`bars` wrote, and the numbers it checked on the way."""

    data: Path
    brief: Path
    notes: Path
    costs: Path
    rows: int
    downloaded: int
    gaps: dict = field(default_factory=dict)
    quirks: dict = field(default_factory=dict)
    sha256: dict = field(default_factory=dict)
    spreads: dict = field(default_factory=dict)
    windows: dict = field(default_factory=dict)


def trend(
    out: str | Path,
    raw: str | Path | None = None,
    costs: str | Path | None = None,
    symbols: list[str] | tuple[str, ...] = TREND_SYMBOLS,
    keep_raw: bool = False,
    log: Callable[[str], None] = print,
    head: Callable[[str], int | None] | None = None,
) -> BuiltBars:
    """Build the trend file: four USD-M perpetuals on hourly bars from
    2023-05-17 to 2026-10-01, bid and ask from half spreads measured on 2023-05-16.
    The build stops before writing the brief if train or validation is shorter than 20
    holding periods of two weeks. See :func:`bars`."""
    return bars(TREND, out, raw, costs, symbols, keep_raw, log, head)


def crosssection(
    out: str | Path,
    raw: str | Path | None = None,
    costs: str | Path | None = None,
    symbols: list[str] | tuple[str, ...] = XS_SYMBOLS,
    keep_raw: bool = False,
    log: Callable[[str], None] = print,
    head: Callable[[str], int | None] | None = None,
) -> BuiltBars:
    """Build the cross-section file: twenty USD-M perpetuals on 15-minute
    bars from 2024-04-01 to 2025-06-25, bid and ask from half spreads measured on
    2024-03-27. The build stops before writing the brief if train or validation is shorter
    than 20 holding periods of 4 hours, short-term reversal's. See :func:`bars`."""
    return bars(XS, out, raw, costs, symbols, keep_raw, log, head)


def bars(
    spec: BarFile,
    out: str | Path,
    raw: str | Path | None = None,
    costs: str | Path | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    keep_raw: bool = False,
    log: Callable[[str], None] = print,
    head: Callable[[str], int | None] | None = None,
) -> BuiltBars:
    """Build a bar file: monthly klines of USD-M perpetuals as rows, and its brief.

    Every file is checked to exist by an HTTP HEAD first (:func:`head_sizes`)
    and the download is refused past :data:`DOWNLOAD_CAP`. Each zip is checked
    against its ``.CHECKSUM``. The half spreads are measured on the cost day
    (:func:`half_spreads`), which lies before train, rounded to four figures
    and written into the brief; the bars become rows with :func:`bar_rows`. A
    symbol-month missing more than :data:`MISSING_LIMIT` of its bars fails.
    The windows ``blinded init`` will fix are computed from the built file,
    and the build stops, before writing the brief, if train or validation is
    shorter than :data:`~blinded.archetypes.PERIODS` holding periods of
    ``spec.lo``. Raw zips are deleted at the end unless ``keep_raw``.
    """
    from blinded import brief as briefs
    from blinded.archetypes import PERIODS
    from blinded.panel import load_events, make_grid

    out = Path(out).expanduser()
    raw = Path(raw).expanduser() if raw else out / "raw"
    costs = Path(costs).expanduser() if costs else out / "costs"
    symbols = [s.upper() for s in (symbols or spec.symbols)]
    start, end, cost_day, interval = spec.start, spec.end, spec.cost_day, spec.interval
    if not cost_day < start:
        raise BuildError(f"the cost day {cost_day} must lie before train, which starts {start}")
    kline = {(s, m): kline_url(s, interval, m) for s in symbols for m in months(start, end)}
    books = {s: book_ticker_url(s, cost_day) for s in symbols}
    urls = [*books.values(), *kline.values()]
    log(f"checking {len(urls)} files on data.binance.vision before fetching any")
    sizes = head_sizes(urls, head)
    cached = costs / f"bookTicker-{cost_day.isoformat()}.json"
    have = json.loads(cached.read_text())["symbols"] if cached.is_file() else {}
    planned = sum(sizes[u] for s, u in books.items() if s not in have)
    planned += sum(sizes[u] for u in kline.values())
    todo = sum(s not in have for s in symbols)
    log(f"about to fetch {planned / 1e6:,.1f} MB ({len(kline)} kline months"
        f"{f', {todo} bookTicker days' if todo else ''}), "
        f"against the {DOWNLOAD_CAP / 1e9:g} GB cap")  # fmt: skip
    if planned > DOWNLOAD_CAP:
        raise BuildError(f"{planned / 1e9:.2f} GB is over the {DOWNLOAD_CAP / 1e9:g} GB cap")
    table = half_spreads(symbols, cost_day, raw, costs, log)
    hs = {s: four_figures(table["symbols"][s]["half_spread_bps"]) for s in symbols}
    name = spec.name(symbols)
    built = BuiltBars(out / f"{name}.parquet", out / f"{name}.brief.yaml", out / "DATA-raw.md",
                      cached, 0, 0, spreads=hs)  # fmt: skip
    t0, t1 = (int(pd.Timestamp(d, tz="UTC").value // MS) for d in (start, end))
    parts, fetched, running = [], [], 0
    for sym in symbols:
        frames = []
        for month in months(start, end):
            path = fetch_url(kline[(sym, month)], raw)
            fetched.append(path)
            running += path.stat().st_size
            built.sha256[(sym, month)] = (raw / f"{path.name}.CHECKSUM").read_text().split()[0]
            built.quirks[(sym, month)] = []
            frame = read_klines(path, built.quirks[(sym, month)])
            frames.append(frame.filter((pl.col("open_time") >= t0) & (pl.col("open_time") < t1)))
        frame = pl.concat(frames).sort("open_time")
        gaps = bar_gaps(frame, interval, start, end)
        zero = pl.from_epoch(frame.filter(pl.col("volume") == 0)["open_time"], time_unit="ms")
        for month, n in zero.dt.strftime("%Y-%m").value_counts().iter_rows():
            gaps[month]["zero_volume"] = n
        built.gaps[sym] = gaps
        worst, share = missing_check(sym, gaps)
        log(f"{sym}: {frame.height:,} bars, {sum(g['missing'] for g in gaps.values())} missing "
            f"(worst {worst}, {share:.2%}); {running / 1e6:,.1f} MB fetched so far")  # fmt: skip
        parts.append(bar_rows(frame, f"{sym}_PERP", hs[sym], interval))
    built.downloaded = running + sum(
        table["symbols"][s]["zip_bytes"] for s in symbols if s not in have
    )
    rows = pl.concat(parts).sort("ts", maintain_order=True)
    raw_brief = spec.brief(built.data.name, symbols, hs, cost_day, start)
    brief = briefs.parse(raw_brief, base=out)
    grid = make_grid(brief, load_events(brief, rows))
    first, last = (pd.Timestamp(int(grid[i]), tz="UTC") for i in (0, -1))
    wins = briefs.windows(brief, first, last)
    need = PERIODS * spec.lo
    spans = {"train": wins["train"].end - (wins["train"].start + brief.grid.warmup),
             "val": wins["val"].end - wins["val"].start}  # fmt: skip
    built.windows = {k: (wins[k].start, wins[k].end) for k in ("train", "val", "test")}
    log("windows: " + "; ".join(f"{k} {a} to {b}" for k, (a, b) in built.windows.items()))
    short = {k: v for k, v in spans.items() if v < need}
    if short:
        found = ", ".join(f"{k} {durations.words(v)}" for k, v in short.items())
        raise BuildError(f"the split leaves {found} against the {durations.words(need)} "
                         f"{spec.holder} needs; nothing written")  # fmt: skip
    if cost_day >= first.date():
        raise BuildError(f"the cost day {cost_day} is not before train ({first})")
    out.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(built.data)
    built.rows = rows.height
    built.brief.write_text(yaml.safe_dump(raw_brief, sort_keys=False))
    built.notes.write_text(bar_notes(spec, built, table, symbols, spans, need))
    log(f"{built.rows:,} rows; validation {durations.words(spans['val'])} against "
        f"{durations.words(need)}; wrote {built.data.name}")  # fmt: skip
    if not keep_raw:
        for path in fetched:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".CHECKSUM").unlink(missing_ok=True)
        for sym in symbols:
            zipped = raw / archive_name("bookTicker", sym, cost_day)
            zipped.unlink(missing_ok=True)
            zipped.with_name(zipped.name + ".CHECKSUM").unlink(missing_ok=True)
        if raw.is_dir() and not any(raw.iterdir()):
            raw.rmdir()
    return built


def bar_notes(spec: BarFile, built: BuiltBars, table: dict, symbols: list[str], spans: dict,
              need) -> str:  # fmt: skip
    """``DATA-raw.md`` of a bar file: the cost day, then one section per month.

    The cost day (``## YYYY-MM-DD``) lies before train, so every run keeps
    it; a month section (``## YYYY-MM``) is cut from a protected run's DATA.md
    when the month ends after train (:func:`notes_until`).
    """
    day = table["day"]
    bar = durations.words(durations.timedelta(spec.interval))
    last = (pd.Timestamp(spec.end) - durations.timedelta(spec.interval)).strftime("%H:%M")
    lines = [
        f"# The raw files behind {built.data.name}",
        "",
        f"Built by `{spec.command}` from data.binance.vision: USD-M perpetuals "
        f"{', '.join(symbols)}, monthly {spec.interval} klines from {spec.start} to {spec.end} "
        f"(the last bar opens at {last} the day before), and one `bookTicker` day per symbol "
        "for the fill model. Each zip matched its `.CHECKSUM`.",
        "",
        f"Each bar is one book row and two trade rows at its close stamp, open_time + {bar} - 1 "
        "ms: the book row's bid and ask are the close times 1 -+ the half spread below, with "
        "no sizes; the sell row holds the volume less the taker buys, the buy row the taker "
        "buys, both at the bar's VWAP; `n` splits the bar's trade count by volume, an "
        "estimate. A decision at a bar boundary sees the bar that closed 1 ms before it and "
        "fills one bar later, at the next close plus or minus the half spread.",
        "",
        f"The split leaves train {durations.words(spans['train'])} and validation "
        f"{durations.words(spans['val'])} against the {durations.words(need)} (20 holding "
        f"periods of {durations.words(spec.lo)}) {spec.holder} needs.",
        "",
        f"## {day}",
        "",
        "The fill model's day, before train. Half spread: the median over every two-sided "
        "`bookTicker` row of the day of (ask - bid) / (2 mid), in bps; the brief keeps it to "
        "four figures.",
        "",
        "| symbol | book rows | two-sided rows | half spread bps | in the brief | zip sha256 |",
        "|---|---|---|---|---|---|",
    ]  # fmt: skip
    for sym in symbols:
        c = table["symbols"][sym]
        lines.append(f"| {sym} | {c['rows']:,} | {c['two_sided_rows']:,} | "
                     f"{c['half_spread_bps']:.6g} | {built.spreads[sym]:g} | "
                     f"`{c['sha256'][:16]}` |")  # fmt: skip
    lines.append("")
    for month in months(spec.start, spec.end):
        lines += [f"## {month}", "",
                  "| symbol | bars expected | present | missing | no volume | file quirks | "
                  "zip sha256 |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        for sym in symbols:
            g = built.gaps[sym][month]
            quirks = ", ".join(built.quirks[(sym, month)]) or "none"
            lines.append(f"| {sym} | {g['expected']:,} | {g['present']:,} | {g['missing']} | "
                         f"{g.get('zero_volume', 0)} | {quirks} | "
                         f"`{built.sha256[(sym, month)][:16]}` |")  # fmt: skip
        lines.append("")
    return "\n".join(lines) + "\n"


# -- one perpetual on two venues: the Tardis free day ---------------------------------

VENUE_MEDIAN_MS = 50.0  # a second venue's book with a median gap at or above this is refused
CROSS_VENUE_ROWS = 7_800_000  # the rows of the 14-hour Tardis file, measured at 2.4 to 2.9 GB
CROSS_VENUE_HOURS = 14
VENUE_TYPES = ("quotes", "trades")
PERP = "XRPUSDT_PERP"


@dataclass(frozen=True)
class Venue:
    """A second venue for the XRP perpetual: its Tardis exchange and symbol, the day, the tag
    its instrument gets, and its name in words."""

    exchange: str
    symbol: str
    day: Date
    tag: str
    title: str

    @property
    def instrument(self) -> str:
        return PERP + self.tag


# Bybit first, the most liquid XRP perpetual outside Binance; OKX second;
# then Bybit on the month before, with Binance's files for that day fetched too
VENUES = (
    Venue("bybit", "XRPUSDT", Date(2026, 10, 1), "_BYBIT", "Bybit"),
    Venue("okex-swap", "XRP-USDT-SWAP", Date(2026, 10, 1), "_OKX", "OKX"),
    Venue("bybit", "XRPUSDT", Date(2026, 9, 1), "_BYBIT", "Bybit"),
)


def venue_cadence(t_ms: np.ndarray) -> dict:
    """The feed cadence of a second venue's book, from its receive stamps in ms.

    :func:`gap_check`'s numbers; ``ok`` when no stamp goes backwards and the
    median gap is under :data:`VENUE_MEDIAN_MS`. The 20% bin rule does not
    apply: a venue that pushes its book on a 10 to 20 ms schedule piles its
    gaps into one bin and is still accepted, with the cadence printed, since a
    lead shorter than it cannot be told from the schedule. A book rebuilt
    every 100 ms, like Tardis's spot ``quotes``, fails.

    Examples
    --------
    >>> every = {ms: np.arange(0, 10_000, float(ms)) for ms in (20, 100)}
    >>> venue_cadence(every[20])["ok"], venue_cadence(every[100])["ok"]
    (True, False)
    """
    out = gap_check(np.asarray(t_ms, dtype=float), median_limit=None)
    out["ok"] = bool(out["backward"] == 0 and out["median_gap_ms"] < VENUE_MEDIAN_MS)
    return out


def cadence_line(instrument: str, cadence: dict) -> str:
    """DATA.md's venue line, which D11 repeats."""
    lo = cadence["bin_max_ms"]
    return (f"venue feed cadence: {instrument}'s book, median gap {cadence['median_gap_ms']:.4g} "
            f"ms between distinct receive stamps; the largest 10 ms bin, {lo} to {lo + 10} ms, "
            f"holds {cadence['share_max']:.0%} of gaps. A lead of the other venue shorter than "
            "this cannot be told from the schedule")  # fmt: skip


def venue_problem(details: dict, venue: Venue, key: str = "") -> str | None:
    """Why ``venue`` cannot be had for its day, from ``tardis_dev.get_exchange_details``; None
    when its symbol is a listed perpetual with ``quotes`` and ``trades`` covering the day and
    the day is free (the first of a month) or a key is set."""
    listed = (details.get("datasets") or {}).get("symbols") or []
    sym = next((s for s in listed if s.get("id") == venue.symbol), None)
    where = f"{venue.exchange} {venue.symbol}"
    if sym is None:
        return f"{venue.exchange} does not list {venue.symbol}"
    if sym.get("type") != "perpetual":
        return f"{where} is a {sym.get('type')}, not a perpetual"
    missing = [t for t in VENUE_TYPES if t not in (sym.get("dataTypes") or ())]
    if missing:
        return f"{where} has no {', '.join(missing)}"
    day = pd.Timestamp(venue.day, tz="UTC")
    since = pd.Timestamp(sym["availableSince"])
    until = pd.Timestamp(sym["availableTo"]) if sym.get("availableTo") else None
    if since > day or (until is not None and until < day + pd.Timedelta("1D")):
        return (f"{where} covers {since.date()} to {until.date() if until else 'now'}, not "
                f"{venue.day}")  # fmt: skip
    if venue.day.day != 1 and not key:
        return f"{venue.day} is not a free first-of-month day"
    return None


@dataclass
class BuiltVenues:
    """What :func:`crossvenue` wrote, and the numbers it checked on the way."""

    data: Path
    brief: Path
    notes: Path
    venue: Venue
    hours: int
    rows: int
    cadence: dict
    files: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    tried: list = field(default_factory=list)
    streams: list = field(default_factory=list)


def crossvenue(
    out: str | Path,
    raw: str | Path | None = None,
    log: Callable[[str], None] = print,
    details: Callable[[str], dict] | None = None,
    fetch_day: Callable[..., None] | None = None,
    venues: tuple[Venue, ...] = VENUES,
    api_key: str | None = None,
) -> BuiltVenues:
    """Build the cross-venue file: the XRP perpetual on Binance and on a
    second venue from one Tardis free day, on the receive clock.

    The second venue is the first of ``venues`` whose symbol Tardis lists as a
    perpetual with ``quotes`` and ``trades`` on its day
    (:func:`venue_problem`); each one passed over is printed with its reason,
    and none left is a :class:`BuildError` before anything is downloaded. The
    venue's ``quotes`` and ``trades`` are fetched, and Binance's
    ``book_ticker`` and ``trades`` for that day when ``raw`` lacks them; files
    already there are kept, as Tardis gz is kept. The rows are the
    :func:`build` layout: Binance as ``XRPUSDT_PERP``, the second venue with
    its tag. The file keeps the largest whole number of hours from midnight,
    at most :data:`CROSS_VENUE_HOURS`, whose rows fit in
    :data:`CROSS_VENUE_ROWS`, and the second venue's book must pass
    :func:`venue_cadence`. ``details`` and ``fetch_day`` replace the Tardis
    client in tests.
    """
    key = api_key if api_key is not None else os.environ.get("TARDIS_API_KEY", "")
    out = Path(out).expanduser()
    raw = Path(raw).expanduser() if raw else out / "raw"
    if details is None or fetch_day is None:
        try:
            import tardis_dev
        except ImportError:
            raise ImportError(
                "blinded demo crossvenue needs the tardis-dev client: pip install 'blinded[demo]'"
            ) from None
        details = details or tardis_dev.get_exchange_details
        fetch_day = fetch_day or tardis_fetcher(tardis_dev.download_datasets, key)
    tried, venue = [], None
    for v in venues:
        problem = venue_problem(details(v.exchange), v, key)
        if problem is None:
            venue = v
            break
        tried.append((v, problem))
        log(f"passing over {v.exchange} {v.symbol} on {v.day}: {problem}")
    if venue is None:
        reasons = "; ".join(p for _, p in tried)
        raise BuildError(f"no second venue for the XRP perpetual: {reasons}; nothing was "
                         "downloaded")  # fmt: skip
    log(f"second venue: {venue.title} ({venue.exchange} {venue.symbol}) on {venue.day}")
    wanted = [("binance-futures", "book_ticker", "XRPUSDT"),
              ("binance-futures", "trades", "XRPUSDT"),
              *((venue.exchange, t, venue.symbol) for t in VENUE_TYPES)]  # fmt: skip
    for exchange, kind, symbol in wanted:
        if not (raw / raw_name(exchange, kind, venue.day, symbol)).is_file():
            log(f"fetching {exchange} {kind} {symbol} {venue.day} through tardis-dev")
            fetch_day(exchange, [kind], [symbol], venue.day, raw)
    files = {}
    for exchange, kind, symbol in wanted:
        path = raw / raw_name(exchange, kind, venue.day, symbol)
        if not path.is_file():
            raise BuildError(f"{path.name} did not arrive; nothing written")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    frame = venue_frame(raw, venue)
    start = pd.Timestamp(venue.day, tz="UTC").value
    ts = frame["ts"].dt.epoch("ns").to_numpy()
    counts = {h: int(np.searchsorted(ts, start + h * 3_600_000_000_000))
              for h in range(1, CROSS_VENUE_HOURS + 1)}  # fmt: skip
    fits = [h for h, n in counts.items() if n <= CROSS_VENUE_ROWS]
    if not fits:
        raise BuildError(f"one hour holds {counts[1]:,} rows, over {CROSS_VENUE_ROWS:,}")
    hours = max(fits)
    frame = frame.slice(0, counts[hours])
    book = frame.filter((pl.col("symbol") == venue.instrument) & ~pl.col("is_trade"))
    cadence = venue_cadence(book["ts"].dt.epoch("ns").to_numpy() / 1e6)
    log(cadence_line(venue.instrument, cadence))
    if not cadence["ok"]:
        raise BuildError(f"{venue.instrument}'s book has a median gap of "
                         f"{cadence['median_gap_ms']:.4g} ms, at or above "
                         f"{VENUE_MEDIAN_MS:g}: a book rebuilt on a cadence manufactures a "
                         "lead; nothing written")  # fmt: skip
    short = "okx" if venue.exchange.startswith("okex") else venue.exchange
    name = f"tardis-binance-{short}-xrpusdt-perp-{venue.day.isoformat()}-{hours}h"
    built = BuiltVenues(out / f"{name}.parquet", out / f"{name}.brief.yaml",
                        out / "DATA-raw.md", venue, hours, frame.height, cadence, files, counts,
                        tried)  # fmt: skip
    clock = "%H:%M:%S%.3f"
    stamps = (pl.col("first").dt.strftime(clock), pl.col("last").dt.strftime(clock))
    built.streams = summary(frame).with_columns(*stamps).rows()
    out.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(built.data)
    built.brief.write_text(yaml.safe_dump(crossvenue_brief(built.data.name, name, venue),
                                          sort_keys=False))  # fmt: skip
    built.notes.write_text(venue_notes(built))
    log(f"{built.rows:,} rows in {hours} h; wrote {built.data.name}")
    return built


def tardis_fetcher(download_datasets, key: str):
    """``fetch_day(exchange, data_types, symbols, day, raw)`` through ``tardis-dev``."""

    def fetch_day(exchange, data_types, symbols, day, raw):
        raw.mkdir(parents=True, exist_ok=True)
        download_datasets(exchange=exchange, data_types=list(data_types), symbols=list(symbols),
                          from_date=day.isoformat(),
                          to_date=(day + timedelta(days=1)).isoformat(), api_key=key,
                          download_dir=str(raw))  # fmt: skip

    return fetch_day


def venue_frame(raw: Path, venue: Venue) -> pl.DataFrame:
    """Every book and trade row of the XRP perpetual on Binance and on ``venue``, by receive
    time, in the layout of :func:`build`: Binance's ``book_ticker`` and the venue's
    ``quotes``, each venue's trades with its own book as received."""
    parts = []
    for exchange, book, symbol, name in (("binance-futures", "book_ticker", "XRPUSDT", PERP),
                                         (venue.exchange, "quotes", venue.symbol,
                                          venue.instrument)):  # fmt: skip
        quotes = book_rows(raw / raw_name(exchange, book, venue.day, symbol), name)
        fills = trades(raw / raw_name(exchange, "trades", venue.day, symbol), name, quotes)
        parts += [quotes, fills]
    frame = pl.concat(parts).sort("ts", maintain_order=True)
    return frame.with_columns(pl.col("side").cast(pl.Categorical))


def crossvenue_brief(data_file: str, name: str, venue: Venue) -> dict:
    """The brief of the cross-venue file: the Tardis brief on the Binance perpetual, with the
    two venues under ``archetypes``, declared one product."""
    brief = brief_for(data_file, PERP, venue.day)
    brief["name"] = name
    brief["archetypes"] = {"instruments": [PERP, venue.instrument],
                           "cross_venue": [[PERP, venue.instrument]]}  # fmt: skip
    return brief


def venue_notes(built: BuiltVenues) -> str:
    """``DATA-raw.md`` of the cross-venue file.

    The head holds what is not about any one window: the venue and why, and
    the files. The day's section (``## YYYY-MM-DD``) holds the row counts and
    the second venue's cadence over the whole file, and a protected run's
    DATA.md cuts it (:func:`notes_until`); DATA.md measures the cadence on
    train for itself.
    """
    v = built.venue
    passed = "".join(f" {t.title} ({t.exchange} {t.symbol}) on {t.day} was passed over: "
                     f"{why}." for t, why in built.tried)  # fmt: skip
    lines = [
        f"# The raw files behind {built.data.name}",
        "",
        f"Built by `blinded demo crossvenue` from the Tardis free day {v.day}: the XRP USD-M "
        f"perpetual on Binance (`binance-futures` `book_ticker` and `trades` of XRPUSDT, "
        f"instrument {PERP}) and on {v.title} (`{v.exchange}` `quotes` and `trades` of "
        f"{v.symbol}, instrument {v.instrument}), every row on the receive clock of Tardis's "
        f"recorder (`local_timestamp`), the first {built.hours} hours of the day.{passed} "
        f"{v.title} is the second venue because its linear perpetual is the most liquid XRP "
        "perpetual outside Binance. Spot is left out, to keep the file inside the row budget. "
        "Tardis files have no checksum; their sizes and SHA-256 are these:",
        "",
        "| file | bytes | sha256 |",
        "|---|---|---|",
        *(f"| {n} | {f['bytes']:,} | `{f['sha256'][:16]}` |" for n, f in built.files.items()),
        "",
        f"The file keeps the largest whole number of hours from midnight, at most "
        f"{CROSS_VENUE_HOURS}, whose rows fit in {CROSS_VENUE_ROWS:,}, the rows of the 14-hour "
        "Tardis file, whose judgement was measured at 2.4 to 2.9 GB.",
        "",
        f"## {v.day.isoformat()}",
        "",
        "Rows by the hour the file is cut at:",
        "",
        "| hours | rows |",
        "|---|---|",
        *(f"| {h} | {n:,} |" for h, n in built.counts.items()),
        "",
        "| instrument | trades | rows | first | last |",
        "|---|---|---|---|---|",
        *(f"| {s} | {'yes' if t else 'no'} | {n:,} | {a} | {b} |"
          for s, t, n, a, b in built.streams),
        "",
        "The build's check, over the whole file. "
        + cadence_line(v.instrument, built.cadence)[0].upper()
        + cadence_line(v.instrument, built.cadence)[1:] + ".",
        "",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


# -- eight coins' spot and perpetual with funding ----------------------

CARRY_COINS = ("BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LINK")
CARRY_START, CARRY_END = Date(2024, 4, 1), Date(2025, 6, 25)
CARRY_SPOT_DAYS = (Date(2024, 3, 1), Date(2024, 2, 1))  # Tardis free days, the fallback second
CARRY_PERP_DAY = Date(2024, 3, 27)  # the cross-section's cost day; its table is cached
CARRY_LO = pd.Timedelta("8h")  # the basis trade's shortest typical holding
CARRY_CAP = 10**9  # what one carry build may download, Tardis included
FUNDING_HEADER = "calc_time,funding_interval_hours,last_funding_rate"
# a settlement's calc_time lies a few ms from its boundary; one further than this fails, and
# the rate is available this long after the boundary, so never before calc_time
FUNDING_SLACK = pd.Timedelta("1s")
SPOT_FEE_BPS, PERP_FEE_BPS = 10.0, 5.0  # Binance VIP0 taker fees
BORROW_URL = "https://www.binance.com/bapi/margin/v1/public/margin/vip/spec/history-interest-rate"
BORROW_REQUEST = timedelta(days=30)  # the span of one request to the margin history
BORROW_GAP = pd.Timedelta("2D")  # the longest the history may go without a published rate


def funding_url(symbol: str, month: str) -> str:
    """The archive's monthly USD-M ``fundingRate`` zip of one symbol."""
    return (f"{VISION}/futures/um/monthly/fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{month}.zip")  # fmt: skip


def read_funding(path: Path) -> pl.DataFrame:
    """The settlements of one ``fundingRate`` zip: ``calc_time`` in ms, the interval in hours
    and ``last_funding_rate``. Any other header fails."""
    with zipfile.ZipFile(path) as zf:
        (member,) = zf.infolist()
        data = zf.read(member)
    first = data[: data.index(b"\n")].decode().strip()
    if first != FUNDING_HEADER:
        raise BuildError(f"{Path(path).name}: the header is {first!r}, expected "
                         f"{FUNDING_HEADER!r}")  # fmt: skip
    return pl.read_csv(io.BytesIO(data), schema_overrides={
        "calc_time": pl.Int64, "funding_interval_hours": pl.Int64,
        "last_funding_rate": pl.Float64})  # fmt: skip


def funding_rows(settled: pl.DataFrame, symbol: str, interval: str) -> tuple[pl.DataFrame, dict]:
    """Settlements as ``kind == 'funding'`` rows, stamped on the bar boundary they belong to.

    The archive's ``calc_time`` lies a few ms from the settlement it records
    (00:00:00.007, or 07:59:59.999). Each row's ``ts`` is the nearest bar
    boundary, so every settlement lands on a grid point and the sim books it to
    the position held there; the brief makes the stream available
    :data:`FUNDING_SLACK` after its stamp, so no rate is seen before its
    ``calc_time``. A ``calc_time`` :data:`FUNDING_SLACK` or more from every
    boundary, or two settlements on one boundary, raise :class:`BuildError`.
    Returns the rows (the kline layout plus ``rate``) and the offsets found:
    ``calc_time`` less the stamp, smallest and largest, in ms.

    Examples
    --------
    >>> s = pl.DataFrame({"calc_time": [1712016000007, 1712044799999],
    ...                   "funding_interval_hours": [8, 8], "last_funding_rate": [1e-4, -2e-5]})
    >>> rows, found = funding_rows(s, "BTCUSDT_PERP", "15m")
    >>> rows["ts"].dt.strftime("%m-%d %H:%M:%S%.3f").to_list(), found["offset_ms"]
    (['04-02 00:00:00.000', '04-02 08:00:00.000'], [-1, 7])
    """
    step = durations.to_ns(interval) // MS
    t = settled["calc_time"].to_numpy()
    stamp = (t + step // 2) // step * step
    off = t - stamp
    slack = FUNDING_SLACK.value // MS
    far = np.abs(off) >= slack
    if far.any():
        when = pd.Timestamp(int(t[far][0]), unit="ms", tz="UTC")
        raise BuildError(f"{symbol}: a settlement at {when} lies {int(off[far][0])} ms from the "
                         f"nearest {interval} boundary, {slack} ms or more")  # fmt: skip
    if len(np.unique(stamp)) != len(stamp):
        raise BuildError(f"{symbol}: two settlements on one {interval} boundary")
    none = pl.lit(None, pl.Float64)
    rows = pl.DataFrame({"ts": pl.Series(stamp * MS).cast(pl.Datetime("ns")),
                         "rate": settled["last_funding_rate"]}).select(
        "ts", symbol=pl.lit(symbol), kind=pl.lit("funding"), bid0_px=none, ask0_px=none,
        mid=none, side=pl.lit(None, pl.String), price=none, qty=none,
        n=pl.lit(None, pl.Int64), rate="rate")  # fmt: skip
    found = {"offset_ms": [int(off.min()), int(off.max())] if len(off) else [None, None],
             "intervals": sorted(set(settled["funding_interval_hours"].to_list())),
             "hours": sorted({int(h) for h in (stamp // 3_600_000) % 24})}  # fmt: skip
    return rows.select(*CARRY_LAYOUT), found


CARRY_LAYOUT = (*KLINE_LAYOUT, "rate")


def spot_book_problem(details: dict, symbol: str, day: Date, key: str = "") -> str | None:
    """Why Tardis cannot give ``binance`` spot ``book_ticker`` of ``symbol`` on ``day``, from
    ``tardis_dev.get_exchange_details("binance")``; None when it is listed as spot with
    ``book_ticker`` covering the day and the day is free (a first of the month) or a key is
    set."""
    listed = (details.get("datasets") or {}).get("symbols") or []
    sym = next((s for s in listed if s.get("id") == symbol), None)
    if sym is None:
        return f"binance does not list {symbol}"
    if sym.get("type") != "spot":
        return f"binance {symbol} is a {sym.get('type')}, not spot"
    if "book_ticker" not in (sym.get("dataTypes") or ()):
        return f"binance {symbol} has no book_ticker"
    start = pd.Timestamp(day, tz="UTC")
    since = pd.Timestamp(sym["availableSince"])
    until = pd.Timestamp(sym["availableTo"]) if sym.get("availableTo") else None
    if since > start or (until is not None and until < start + pd.Timedelta("1D")):
        return (f"binance {symbol} covers {since.date()} to {until.date() if until else 'now'}, "
                f"not {day}")  # fmt: skip
    if day.day != 1 and not key:
        return f"{day} is not a free first-of-month day"
    return None


def tardis_half_spreads(symbols: list[str], day: Date, raw: Path, cache: Path, fetch_day,
                        log: Callable[[str], None] = print) -> dict:  # fmt: skip
    """The median half spread of each spot symbol over one Tardis ``binance`` ``book_ticker``
    day, in bps of the mid, as :func:`half_spreads` measures a ``bookTicker`` day.

    Every two-sided row of the day counts once. The gz files are kept in
    ``raw``, as the Tardis builders keep theirs; the result, with each file's
    rows, size and SHA-256 (Tardis has no checksums), is cached as
    ``cache/tardis-binance-book_ticker-DAY.json``. ``fetch_day(exchange,
    data_types, symbols, day, raw)`` downloads one file (:func:`tardis_fetcher`).
    """
    from blinded.markouts import half_spread_bps

    path = cache / f"tardis-binance-book_ticker-{day.isoformat()}.json"
    table = json.loads(path.read_text()) if path.is_file() else {"day": day.isoformat(),
                                                                 "symbols": {}}  # fmt: skip
    for sym in symbols:
        if sym in table["symbols"]:
            log(f"{sym} {day}: half spread {table['symbols'][sym]['half_spread_bps']:.4g} bps, "
                f"from {path.name}")  # fmt: skip
            continue
        gz = raw / raw_name("binance", "book_ticker", day, sym)
        if not gz.is_file():
            log(f"fetching binance book_ticker {sym} {day} through tardis-dev")
            fetch_day("binance", ["book_ticker"], [sym], day, raw)
        if not gz.is_file():
            raise BuildError(f"{gz.name} did not arrive; nothing written")
        frame = pl.read_csv(
            gz,
            columns=["bid_price", "ask_price"],
            schema_overrides={"bid_price": pl.Float64, "ask_price": pl.Float64},
        )
        bid, ask = frame["bid_price"].to_numpy(), frame["ask_price"].to_numpy()
        two_sided = (bid > 0) & (ask > bid)
        table["symbols"][sym] = {
            "half_spread_bps": half_spread_bps(bid[two_sided], ask[two_sided]),
            "rows": len(bid), "two_sided_rows": int(two_sided.sum()),
            "sha256": file_sha256(gz), "gz_bytes": gz.stat().st_size,
        }  # fmt: skip
        log(f"{sym} {day}: {len(bid):,} book rows, {gz.stat().st_size / 1e6:,.1f} MB, half "
            f"spread {table['symbols'][sym]['half_spread_bps']:.4g} bps")  # fmt: skip
        del frame, bid, ask
        cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, indent=1) + "\n")
    return table


def borrow_history(coins: list[str], start: Date, end: Date, costs: Path,
                   log: Callable[[str], None] = print,
                   fetch_json: Callable[[str], dict] | None = None) -> dict:  # fmt: skip
    """Each coin's VIP 0 daily borrow rate as Binance's public margin interest history published
    it, from the day before ``start`` to ``end``: ``{coin: [[ms, rate], ...]}`` in time order.

    The history has a row at 00:00 UTC each day and one at each change; a
    request covers :data:`BORROW_REQUEST` and may return rows past its end,
    so each keeps only its own span. The day before ``start`` is there so the
    rate in force at ``start`` is known. Cached as
    ``costs/borrow-<start>-<end>.json``; ``fetch_json(url)`` replaces the
    network in tests.

    Examples
    --------
    >>> import tempfile
    >>> ms = lambda d: int(pd.Timestamp(d, tz="UTC").value // MS)
    >>> rows = {"data": [{"timestamp": str(ms("2024-04-01")), "dailyInterestRate": "0.00005568"},
    ...                  {"timestamp": str(ms("2024-04-03")), "dailyInterestRate": "0.01"}]}
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     got = borrow_history(["ETH"], Date(2024, 4, 1), Date(2024, 4, 2), Path(tmp),
    ...                          lambda m: None, lambda url: rows)
    >>> got["ETH"]
    [[1711929600000, 5.568e-05]]
    """
    cached = costs / f"borrow-{start.isoformat()}-{end.isoformat()}.json"
    if cached.is_file():
        return json.loads(cached.read_text())
    if fetch_json is None:

        def fetch_json(url: str) -> dict:
            return json.loads(_get(url))

    out = {}
    for coin in coins:
        got: dict[int, float] = {}
        a = start - timedelta(days=1)
        while a < end:
            b = min(a + BORROW_REQUEST, end)
            lo, hi = (int(pd.Timestamp(d, tz="UTC").value // MS) for d in (a, b))
            page = fetch_json(f"{BORROW_URL}?asset={coin}&vipLevel=0&startTime={lo}&endTime={hi}")
            for r in page.get("data") or []:
                t, rate = int(r["timestamp"]), float(r["dailyInterestRate"])
                if lo <= t < hi:
                    if got.get(t, rate) != rate:
                        raise BuildError(f"Binance's margin history gives {coin} two rates at {t}")
                    got[t] = rate
            a = b
        out[coin] = [[t, got[t]] for t in sorted(got)]
        log(f"{coin} borrow: {len(got):,} published rates from {start - timedelta(days=1)} "
            f"to {end}")  # fmt: skip
    costs.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out) + "\n")
    return out


def borrow_rows(history: list, symbol: str, start: Date, end: Date) -> tuple[pl.DataFrame, dict]:
    """One coin's published borrow rates as ``kind == 'borrow'`` rows of its spot instrument,
    from the rate in force at ``start`` to the last before ``end``.

    ``history`` is :func:`borrow_history`'s list for the coin. Each row is
    stamped when the rate was published, with ``rate`` the daily rate as a
    fraction; the brief maps it to ``borrow_rate`` and the sim charges a short
    the rate as-of each step. No rate in force at ``start``, or a gap longer
    than :data:`BORROW_GAP` between two, raises :class:`BuildError`. Returns the
    rows (the carry layout) and what was found: the rows, the first stamp and
    the longest gap in hours.

    Examples
    --------
    >>> ms = lambda d: int(pd.Timestamp(d, tz="UTC").value // MS)
    >>> h = [[ms("2024-03-31"), 1e-4], [ms("2024-04-01"), 2e-4], [ms("2024-04-01 06:00"), 3e-4],
    ...      [ms("2024-04-02"), 4e-4]]
    >>> rows, found = borrow_rows(h, "ETHUSDT", Date(2024, 4, 1), Date(2024, 4, 2))
    >>> rows["rate"].to_list(), found["max_gap_hours"]
    ([0.0002, 0.0003], 18.0)
    """
    t = np.array([x[0] for x in history], dtype=np.int64)
    rate = np.array([x[1] for x in history], dtype=float)
    lo, hi = (int(pd.Timestamp(d, tz="UTC").value // MS) for d in (start, end))
    first = int(np.searchsorted(t, lo, side="right")) - 1
    if first < 0:
        raise BuildError(f"{symbol}: no borrow rate in force at {start}; nothing written")
    keep = slice(first, int(np.searchsorted(t, hi, side="left")))
    t, rate = t[keep], rate[keep]
    gaps = np.diff(np.append(t, hi)) / 3.6e6
    if gaps.max() * pd.Timedelta("1h") > BORROW_GAP:
        at = pd.Timestamp(int(t[int(np.argmax(gaps))]), unit="ms", tz="UTC")
        raise BuildError(f"{symbol}: the borrow history goes {gaps.max():.0f} hours without a "
                         f"rate from {at}; nothing written")  # fmt: skip
    none = pl.lit(None, pl.Float64)
    rows = pl.DataFrame({"ts": pl.Series(t * MS).cast(pl.Datetime("ns")), "rate": rate}).select(
        "ts", symbol=pl.lit(symbol), kind=pl.lit("borrow"), bid0_px=none, ask0_px=none,
        mid=none, side=pl.lit(None, pl.String), price=none, qty=none,
        n=pl.lit(None, pl.Int64), rate="rate")  # fmt: skip
    found = {"rows": rows.height,
             "first": pd.Timestamp(int(t[0]), unit="ms", tz="UTC").isoformat(),
             "max_gap_hours": float(gaps.max())}  # fmt: skip
    return rows, found


def carry_brief(data_file: str, coins: list[str], hs: dict[str, float], spot_day: Date,
                perp_day: Date, start: Date) -> dict:  # fmt: skip
    """The brief of the carry file: eight coins' spot and perpetual on
    15-minute bars with a funding stream on each perpetual and a borrow stream on each spot,
    spot fees 10 bps and perpetual 5 bps."""
    inst = [s for c in coins for s in (f"{c}USDT", f"{c}USDT_PERP")]
    data = _bar_data(data_file, [], {}, perp_day)
    data["streams"]["funding"] = {"filter": "kind == 'funding'", "fields": {"rate": "rate"}}
    data["streams"]["borrow"] = {"filter": "kind == 'borrow'", "fields": {"borrow_rate": "rate"}}
    slack = int(FUNDING_SLACK / pd.Timedelta("1ms"))
    data["availability"] = {
        "default_delay_ms": 0,
        "overrides": [{"stream": "funding", "delay_ms": slack},
                      {"stream": "borrow", "delay_ms": slack}],
    }  # fmt: skip
    data["cost_model"]["source"] = (f"tardis binance book_ticker {spot_day.isoformat()} (spot) "
                                    f"and data.binance.vision bookTicker {perp_day.isoformat()} "
                                    "(perpetuals)")  # fmt: skip
    data["cost_model"]["half_spread_bps"] = {s: hs[s] for s in inst}
    return {
        "blinded": 0,
        "name": data_file.removesuffix(".parquet"),
        "data": data,
        "target": {"instrument": inst[-1], "price": "mid", "horizons": ["1h", "8h", "1d"],
                   "primary_horizon": "8h"},
        "grid": {"step": "15min", "start": f"{start.isoformat()}T00:00:00Z", "warmup": "1d"},
        "execution": {"latency_ms": 900_000, "taker_fee_bps": PERP_FEE_BPS,
                      "maker_fee_bps": 0.0, "max_position": 1.0, "min_rebalance": 0.1,
                      "fees": {f"{c}USDT": SPOT_FEE_BPS for c in coins}},
        "splits": {"scheme": "time", "fractions": [0.5, 0.25, 0.25], "embargo": "2d"},
        "stats": {"se": "both", "block": "8h", "n_boot": 1000, "seed": 20241128},
        "archetypes": {
            "instruments": inst,
            "carry": [[f"{c}USDT", f"{c}USDT_PERP"] for c in coins],
            "params": {"basis_trade": {"rate_in": 0.0001, "rate_out": 0.0},
                       "pairs_trading": {"window": "2d", "every": "4h", "z_in": 2.0,
                                         "z_out": 0.5}},
        },
    }  # fmt: skip


@dataclass
class BuiltCarry(BuiltBars):
    """What :func:`carry` wrote: :class:`BuiltBars`, plus the spot cost day, the Tardis files
    and the funding settlements' stamps."""

    spot_day: Date | None = None
    spot_costs: Path | None = None
    tardis_bytes: int = 0
    funding: dict = field(default_factory=dict)
    tried: list = field(default_factory=list)
    borrow: dict = field(default_factory=dict)


def carry(
    out: str | Path,
    raw: str | Path | None = None,
    costs: str | Path | None = None,
    tardis_raw: str | Path | None = None,
    coins: list[str] | tuple[str, ...] = CARRY_COINS,
    keep_raw: bool = False,
    log: Callable[[str], None] = print,
    head: Callable[[str], int | None] | None = None,
    details: Callable[[str], dict] | None = None,
    fetch_day: Callable[..., None] | None = None,
    api_key: str | None = None,
    cap: int = CARRY_CAP,
    fetch_json: Callable[[str], dict] | None = None,
) -> BuiltCarry:
    """Build the carry file: eight coins' spot and USD-M perpetual on
    15-minute bars from 2024-04-01 to 2025-06-25, with every funding settlement.

    Before a byte is fetched: every kline and ``fundingRate`` month is checked
    by an HTTP HEAD (:func:`head_sizes`), and Tardis's exchange details must
    list each spot symbol with ``book_ticker`` on a free day, 2024-03-01 or
    else 2024-02-01 (:func:`spot_book_problem`; each day passed over is printed
    with its reason). Spot half spreads are measured on that Tardis day
    (:func:`tardis_half_spreads`), perpetual ones are the cross-section's
    table of 2024-03-27 (:func:`half_spreads`, cached), both before train.
    Each bar is :func:`bar_rows`; each settlement a funding row stamped on its
    boundary (:func:`funding_rows`). A symbol-month missing more than
    :data:`MISSING_LIMIT` of its bars fails, and the build stops before
    writing the brief if train or validation is shorter than 20 holding
    periods of 8 hours, the basis trade's. The download is refused past
    ``cap``, checked before each file. Binance zips are deleted at the end
    unless ``keep_raw``; Tardis gz files are kept in ``tardis_raw``. Each
    coin's borrow rates are borrow rows of its spot instrument
    (:func:`borrow_history`, cached in ``costs``, and :func:`borrow_rows`).
    ``head``, ``details``, ``fetch_day`` and ``fetch_json`` replace the
    network in tests.
    """
    from blinded import brief as briefs
    from blinded.archetypes import PERIODS
    from blinded.panel import load_events, make_grid

    key = api_key if api_key is not None else os.environ.get("TARDIS_API_KEY", "")
    out = Path(out).expanduser()
    raw = Path(raw).expanduser() if raw else out / "raw"
    costs = Path(costs).expanduser() if costs else out / "costs"
    tardis_raw = Path(tardis_raw).expanduser() if tardis_raw else raw
    coins = [c.upper() for c in coins]
    symbols = [f"{c}USDT" for c in coins]  # the archive's names, of spot and perpetual alike
    start, end = CARRY_START, CARRY_END
    if details is None or fetch_day is None:
        try:
            import tardis_dev
        except ImportError:
            raise ImportError(
                "blinded demo carry needs the tardis-dev client: pip install 'blinded[demo]'"
            ) from None
        details = details or tardis_dev.get_exchange_details
        fetch_day = fetch_day or tardis_fetcher(tardis_dev.download_datasets, key)
    tried, spot_day = [], None
    listing = details("binance")
    for day in CARRY_SPOT_DAYS:
        problems = [p for s in symbols if (p := spot_book_problem(listing, s, day, key))]
        if not problems:
            spot_day = day
            break
        tried.append((day, "; ".join(problems)))
        log(f"passing over the spot book on {day}: {'; '.join(problems)}")
    if spot_day is None:
        raise BuildError(
            "no Tardis day for the spot half spreads: "
            + "; ".join(f"{d}: {p}" for d, p in tried)
            + "; nothing was downloaded"
        )
    for day in (spot_day, CARRY_PERP_DAY):
        if not day < start:
            raise BuildError(f"the cost day {day} must lie before train, which starts {start}")
    month_list = months(start, end)
    urls = {(s, m, kind): url for s in symbols for m in month_list for kind, url in (
        ("spot", kline_url(s, "15m", m, market="spot")), ("perp", kline_url(s, "15m", m)),
        ("funding", funding_url(s, m)))}  # fmt: skip
    cached = costs / f"bookTicker-{CARRY_PERP_DAY.isoformat()}.json"
    have = json.loads(cached.read_text())["symbols"] if cached.is_file() else {}
    books = {s: book_ticker_url(s, CARRY_PERP_DAY) for s in symbols if s not in have}
    log(f"checking {len(urls) + len(books)} files on data.binance.vision before fetching any; "
        f"Tardis lists binance book_ticker on {spot_day} for {', '.join(symbols)}")  # fmt: skip
    sizes = head_sizes([*urls.values(), *books.values()], head)
    planned = sum(sizes.values())
    log(f"about to fetch {planned / 1e6:,.1f} MB from data.binance.vision ({len(urls)} monthly "
        f"files{f', {len(books)} bookTicker days' if books else ''}) and up to {len(symbols)} "
        f"Tardis book_ticker days, against the {cap / 1e9:g} GB cap")  # fmt: skip
    if planned > cap:
        raise BuildError(f"{planned / 1e9:.2f} GB is over the {cap / 1e9:g} GB cap")
    running = [0]

    def counted(fetch):
        def call(exchange, data_types, symbols, day, where):
            if running[0] + planned > cap:
                raise BuildError(f"{running[0] / 1e9:.2f} GB fetched from Tardis already; the "
                                 f"next file would pass the {cap / 1e9:g} GB cap")  # fmt: skip
            fetch(exchange, data_types, symbols, day, where)
            for sym in symbols:
                for kind in data_types:
                    gz = where / raw_name(exchange, kind, day, sym)
                    running[0] += gz.stat().st_size if gz.is_file() else 0

        return call

    spot_table = tardis_half_spreads(symbols, spot_day, tardis_raw, costs, counted(fetch_day),
                                     log)  # fmt: skip
    perp_table = half_spreads(symbols, CARRY_PERP_DAY, raw, costs, log)
    hs = {s: four_figures(spot_table["symbols"][s]["half_spread_bps"]) for s in symbols}
    hs |= {f"{s}_PERP": four_figures(perp_table["symbols"][s]["half_spread_bps"])
           for s in symbols}  # fmt: skip
    name = f"binance-carry{len(coins)}-{start.isoformat()}-{(end - start).days}d-15m"
    built = BuiltCarry(
        out / f"{name}.parquet",
        out / f"{name}.brief.yaml",
        out / "DATA-raw.md",
        cached,
        0,
        0,
        spreads=hs,
        spot_day=spot_day,
        tried=tried,
        spot_costs=costs / f"tardis-binance-book_ticker-{spot_day}.json",
    )
    built.tardis_bytes = running[0]
    t0, t1 = (int(pd.Timestamp(d, tz="UTC").value // MS) for d in (start, end))
    parts, fetched, binance_bytes = [], [], 0
    # spot and futures kline zips have the same names (BTCUSDT-15m-2024-04.zip), and
    # fetch_url keeps a file it already has, so each market gets its own folder
    where = {"spot": raw / "spot", "perp": raw / "futures", "funding": raw / "futures"}
    for sym in symbols:
        for kind, inst in (("spot", sym), ("perp", f"{sym}_PERP")):
            frames = []
            for month in month_list:
                path = fetch_url(urls[(sym, month, kind)], where[kind])
                fetched.append(path)
                binance_bytes += path.stat().st_size
                checksum = path.with_name(f"{path.name}.CHECKSUM")
                built.sha256[(inst, month)] = checksum.read_text().split()[0]
                if kind == "perp" and built.sha256[(inst, month)] == built.sha256[(sym, month)]:
                    raise BuildError(f"the spot and perpetual klines of {sym} {month} are one "
                                     "file; nothing written")  # fmt: skip
                built.quirks[(inst, month)] = []
                frame = read_klines(path, built.quirks[(inst, month)])
                frames.append(frame.filter((pl.col("open_time") >= t0)
                                           & (pl.col("open_time") < t1)))  # fmt: skip
            frame = pl.concat(frames).sort("open_time")
            gaps = bar_gaps(frame, "15m", start, end)
            zero = pl.from_epoch(frame.filter(pl.col("volume") == 0)["open_time"], time_unit="ms")
            for month, n in zero.dt.strftime("%Y-%m").value_counts().iter_rows():
                gaps[month]["zero_volume"] = n
            built.gaps[inst] = gaps
            worst, share = missing_check(inst, gaps)
            parts.append(bar_rows(frame, inst, hs[inst], "15m").with_columns(
                rate=pl.lit(None, pl.Float64)))  # fmt: skip
            log(f"{inst}: {frame.height:,} bars, {sum(g['missing'] for g in gaps.values())} "
                f"missing (worst {worst}, {share:.2%}); "
                f"{(binance_bytes + running[0]) / 1e6:,.1f} MB fetched so far")  # fmt: skip
        settled = []
        for month in month_list:
            path = fetch_url(urls[(sym, month, "funding")], where["funding"])
            fetched.append(path)
            binance_bytes += path.stat().st_size
            checksum = path.with_name(f"{path.name}.CHECKSUM")
            built.sha256[(f"{sym}_PERP funding", month)] = checksum.read_text().split()[0]
            settled.append(read_funding(path))
        rows, found = funding_rows(pl.concat(settled).sort("calc_time"), f"{sym}_PERP", "15m")
        rows = rows.filter((pl.col("ts") >= datetime(start.year, start.month, start.day))
                           & (pl.col("ts") < datetime(end.year, end.month, end.day)))  # fmt: skip
        stamps = rows["ts"].dt.strftime("%Y-%m").value_counts()
        built.funding[f"{sym}_PERP"] = found | {
            "rows": rows.height,
            "by_month": dict(stamps.iter_rows()),
        }
        parts.append(rows)
        log(
            f"{sym}_PERP: {rows.height:,} settlements, calc_time {found['offset_ms'][0]} to "
            f"{found['offset_ms'][1]} ms from its boundary, at hours {found['hours']}"
        )
    history = borrow_history(coins, start, end, costs, log, fetch_json)
    for coin in coins:
        rows, built.borrow[f"{coin}USDT"] = borrow_rows(history[coin], f"{coin}USDT", start, end)
        parts.append(rows)
    built.downloaded = (
        binance_bytes
        + running[0]
        + sum(perp_table["symbols"][s]["zip_bytes"] for s in symbols if s not in have)
    )
    rows = pl.concat(parts).sort("ts", maintain_order=True)
    raw_brief = carry_brief(built.data.name, coins, hs, spot_day, CARRY_PERP_DAY, start)
    brief = briefs.parse(raw_brief, base=out)
    grid = make_grid(brief, load_events(brief, rows))
    first, last = (pd.Timestamp(int(grid[i]), tz="UTC") for i in (0, -1))
    off_grid = rows.filter(pl.col("kind") == "funding")["ts"].dt.epoch("ns").to_numpy()
    if np.any((off_grid - grid[0]) % (grid[1] - grid[0])):
        raise BuildError("a funding stamp is not on the grid")
    wins = briefs.windows(brief, first, last)
    need = PERIODS * CARRY_LO
    spans = {"train": wins["train"].end - (wins["train"].start + brief.grid.warmup),
             "val": wins["val"].end - wins["val"].start}  # fmt: skip
    built.windows = {k: (wins[k].start, wins[k].end) for k in ("train", "val", "test")}
    log("windows: " + "; ".join(f"{k} {a} to {b}" for k, (a, b) in built.windows.items()))
    short = {k: v for k, v in spans.items() if v < need}
    if short:
        found = ", ".join(f"{k} {durations.words(v)}" for k, v in short.items())
        raise BuildError(f"the split leaves {found} against the {durations.words(need)} "
                         "the basis trade needs; nothing written")  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(built.data)
    built.rows = rows.height
    built.brief.write_text(yaml.safe_dump(raw_brief, sort_keys=False))
    built.notes.write_text(carry_notes(built, spot_table, perp_table, coins, spans, need))
    log(f"{built.rows:,} rows; validation {durations.words(spans['val'])} against "
        f"{durations.words(need)}; wrote {built.data.name}")  # fmt: skip
    if not keep_raw:
        for path in fetched:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".CHECKSUM").unlink(missing_ok=True)
        for sym in books:
            zipped = raw / archive_name("bookTicker", sym, CARRY_PERP_DAY)
            zipped.unlink(missing_ok=True)
            zipped.with_name(zipped.name + ".CHECKSUM").unlink(missing_ok=True)
        for folder in (*set(where.values()), raw):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
    return built


def carry_notes(built: BuiltCarry, spot_table: dict, perp_table: dict, coins: list[str],
                spans: dict, need) -> str:  # fmt: skip
    """``DATA-raw.md`` of the carry file: the two cost days, then one section per month with
    each instrument's bars and each perpetual's settlements."""
    spots = [f"{c}USDT" for c in coins]
    passed = "".join(f" {d} was passed over: {why}." for d, why in built.tried)
    lines = [
        f"# The raw files behind {built.data.name}",
        "",
        f"Built by `blinded demo carry`: spot and USD-M perpetual {', '.join(spots)} (the "
        "perpetual as SYM_PERP), monthly 15m klines from data.binance.vision "
        f"(`spot/monthly/klines` and `futures/um/monthly/klines`) from {CARRY_START} to "
        f"{CARRY_END} (the last bar opens at 23:45 the day before), every funding settlement "
        "of each perpetual (`futures/um/monthly/fundingRate`), a Tardis `binance` "
        f"`book_ticker` day per spot symbol and the cross-section's `bookTicker` day for the "
        f"perpetuals, for the fill model.{passed} Each zip matched its `.CHECKSUM`; Tardis files "
        "have no checksum, and their sizes and SHA-256 are below.",
        "",
        "Each bar is one book row and two trade rows at its close stamp, open_time + 15 min - "
        "1 ms, as in the other bar files. Each settlement is one funding row, `kind == "
        "'funding'`, with `rate` the archive's `last_funding_rate`, stamped on the 15-minute "
        "boundary nearest its `calc_time`, which lies a few ms from it; the brief makes the "
        f"funding stream available {durations.words(FUNDING_SLACK)} after its stamp, so a "
        "rate is never seen before its `calc_time`, and the sim books it to the position held "
        "at the stamp. Fees are Binance's VIP0 taker rates, 10 bps on spot and 5 bps on the "
        "perpetual: a round trip in both legs costs 30 bps before any spread or basis move.",
        "",
        "Each coin's borrow rate is one borrow row of its spot instrument per published rate, "
        "`kind == 'borrow'`, with `rate` the daily rate as a fraction, from Binance's public "
        f"margin interest history (VIP 0, `{BORROW_URL}`), which has a row at 00:00 UTC every "
        "day and one at each change. The first row is the rate in force at the start; the "
        f"stream is available {durations.words(FUNDING_SLACK)} after each stamp, and a short "
        "spot position pays the rate as-of each step.",
        "",
        "| spot | borrow rows | first | longest gap, hours |",
        "|---|---|---|---|",
        *(f"| {sym} | {b['rows']:,} | {b['first'][:16]} | {b['max_gap_hours']:g} |"
          for sym, b in built.borrow.items()),
        "",
        f"The split leaves train {durations.words(spans['train'])} and validation "
        f"{durations.words(spans['val'])} against the {durations.words(need)} (20 holding "
        f"periods of {durations.words(CARRY_LO)}) the basis trade needs.",
        "",
        f"## {built.spot_day.isoformat()}",
        "",
        "The spot fill model's day, before train: Tardis `binance` `book_ticker`, every change "
        "of Binance's best bid and offer. Half spread: the median over every two-sided row of "
        "the day of (ask - bid) / (2 mid), in bps; the brief keeps four figures.",
        "",
        "| symbol | book rows | two-sided rows | half spread bps | in the brief | gz bytes | "
        "sha256 |",
        "|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for sym in spots:
        c = spot_table["symbols"][sym]
        lines.append(f"| {sym} | {c['rows']:,} | {c['two_sided_rows']:,} | "
                     f"{c['half_spread_bps']:.6g} | {built.spreads[sym]:g} | {c['gz_bytes']:,} | "
                     f"`{c['sha256'][:16]}` |")  # fmt: skip
    lines += ["", f"## {CARRY_PERP_DAY.isoformat()}", "",
              "The perpetuals' fill model's day, before train: the cross-section file's "
              "`bookTicker` table, measured the same way.", "",
              "| symbol | book rows | two-sided rows | half spread bps | in the brief | "
              "zip sha256 |", "|---|---|---|---|---|---|"]  # fmt: skip
    for sym in spots:
        c = perp_table["symbols"][sym]
        lines.append(f"| {sym}_PERP | {c['rows']:,} | {c['two_sided_rows']:,} | "
                     f"{c['half_spread_bps']:.6g} | {built.spreads[sym + '_PERP']:g} | "
                     f"`{c['sha256'][:16]}` |")  # fmt: skip
    lines.append("")
    lines += ["Funding stamps: each perpetual's settlements, how far its `calc_time` lay from "
              "the boundary it is stamped on (ms, smallest and largest), the hours of the day "
              "they settle at and the archive's `funding_interval_hours`.", "",
              "| perpetual | settlements | calc_time less stamp, ms | hours (UTC) | interval h |",
              "|---|---|---|---|---|"]  # fmt: skip
    for inst, f in built.funding.items():
        lo, hi = f["offset_ms"]
        lines.append(f"| {inst} | {f['rows']:,} | {lo} to {hi} | "
                     f"{', '.join(f'{h:02d}' for h in f['hours'])} | "
                     f"{', '.join(str(i) for i in f['intervals'])} |")  # fmt: skip
    lines.append("")
    for month in months(CARRY_START, CARRY_END):
        lines += [f"## {month}", "",
                  "| instrument | bars expected | present | missing | no volume | file quirks | "
                  "kline zip sha256 | settlements | fundingRate zip sha256 |",
                  "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for sym in spots:
            for inst in (sym, f"{sym}_PERP"):
                g = built.gaps[inst][month]
                quirks = ", ".join(built.quirks[(inst, month)]) or "none"
                funded = inst in built.funding
                settled = built.funding[inst]["by_month"].get(month, 0) if funded else ""
                sha = f"`{built.sha256[(inst + ' funding', month)][:16]}`" if funded else ""
                lines.append(
                    f"| {inst} | {g['expected']:,} | {g['present']:,} | {g['missing']} | "
                    f"{g.get('zero_volume', 0)} | {quirks} | "
                    f"`{built.sha256[(inst, month)][:16]}` | {settled} | {sha} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"
