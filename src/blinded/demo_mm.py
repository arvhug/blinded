"""Market-making files from free Tardis days.

``blinded demo mm`` builds one USD-M perpetual's day for the fill engine: a data file in
the Tardis layout whose book rows are thinned to the read
lattice and carry five levels from ``book_snapshot_5``, with every trade at its own stamp
and the funding settlements of ``derivative_ticker``; a touch file of every
``book_ticker`` row in ticks, which only the fill engine reads; a brief; and
``DATA-raw.md`` with the raw files' sizes and SHA-256 and the day's measurements.
``blinded demo arb`` and ``blinded demo venues`` build the arbitrage pair and the
consolidated book from the same pieces.

Every stamp is Tardis's ``local_timestamp``, its collector's receive clock, in ns.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from blinded import bookstats
from blinded.demo import (
    FUNDING_SLACK,
    MS,
    BuildError,
    file_sha256,
    raw_name,
    read_points,
    tardis_fetcher,
    thin_book,
)

PERP_EXCHANGE = "binance-futures"
MM_TYPES = ("book_ticker", "trades", "book_snapshot_5", "derivative_ticker")
# ticks and lots of the two perpetuals, from fapi.binance.com/fapi/v1/exchangeInfo as
# read on 19 Oct 2026; the builder checks the book against the tick
PERP_STEPS = {"SOLUSDT": (0.01, 0.01), "LTCUSDT": (0.01, 0.001)}
# a depth snapshot must show the touch stream's prices at its level 0 on this share of its
# rows, the touch read as-of each
SNAPSHOT_SHARE = 0.90
DAY_NS = 86_400 * 10**9
STEP = "100ms"
LAYOUT = ("ts", "symbol", "kind", "is_trade", "side", "price", "qty", "bid0_px", "ask0_px",
          "bid0_qty", "ask0_qty", "mid")  # fmt: skip


def level_columns(levels: int, first: int = 1) -> list[str]:
    """The book's level columns past the touch, ``bid1_px, bid1_qty, ask1_px, ask1_qty, ...``.

    Examples
    --------
    >>> level_columns(2)
    ['bid1_px', 'bid1_qty', 'ask1_px', 'ask1_qty']
    """
    return [f"{s}{k}_{f}" for k in range(first, levels) for s in ("bid", "ask")
            for f in ("px", "qty")]  # fmt: skip


def listing_problem(details: dict, exchange: str, symbol: str, day: Date, types,
                    kind: str | None = None, key: str = "") -> str | None:  # fmt: skip
    """Why Tardis cannot give ``types`` of ``symbol`` on ``exchange`` for ``day``, from
    ``tardis_dev.get_exchange_details(exchange)``; None when it is listed (as ``kind`` when
    given: ``perpetual``, ``spot``) with every type, covering the day, and the day is free
    (a first of the month) or a key is set."""
    listed = (details.get("datasets") or {}).get("symbols") or []
    sym = next((s for s in listed if s.get("id") == symbol), None)
    where = f"{exchange} {symbol}"
    if sym is None:
        return f"{exchange} does not list {symbol}"
    if kind is not None and sym.get("type") != kind:
        return f"{where} is a {sym.get('type')}, not a {kind}"
    missing = [t for t in types if t not in (sym.get("dataTypes") or ())]
    if missing:
        return f"{where} has no {', '.join(missing)}"
    start = pd.Timestamp(day, tz="UTC")
    since = pd.Timestamp(sym["availableSince"])
    until = pd.Timestamp(sym["availableTo"]) if sym.get("availableTo") else None
    if since > start or (until is not None and until < start + pd.Timedelta("1D")):
        return f"{where} covers {since.date()} to {until.date() if until else 'now'}, not {day}"
    if day.day != 1 and not key:
        return f"{day} is not a free first-of-month day"
    return None


def _client(details, fetch_day, key: str):
    if details is not None and fetch_day is not None:
        return details, fetch_day
    try:
        import tardis_dev
    except ImportError:
        raise ImportError("blinded demo needs the tardis-dev client: pip install "
                          "'blinded[demo]'") from None  # fmt: skip
    return (details or tardis_dev.get_exchange_details,
            fetch_day or tardis_fetcher(tardis_dev.download_datasets, key))  # fmt: skip


def fetch_files(wanted, day: Date, raw: Path, fetch_day, log) -> dict:
    """Fetch every ``(exchange, type, symbol)`` not already in ``raw``; returns each file's
    bytes and SHA-256 (Tardis files have no checksums)."""
    files = {}
    for exchange, kind, symbol in wanted:
        path = raw / raw_name(exchange, kind, day, symbol)
        if not path.is_file():
            log(f"fetching {exchange} {kind} {symbol} {day} through tardis-dev")
            fetch_day(exchange, [kind], [symbol], day, raw)
        if not path.is_file():
            raise BuildError(f"{path.name} did not arrive; nothing written")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return files


# -- reading Tardis files ----------------------------------------------------------------


def _stamps(frame: pl.DataFrame, name: str) -> None:
    t = frame["local_timestamp"]
    if not t.is_sorted():
        raise BuildError(f"{name}: local_timestamp goes backwards; nothing written")


def read_ticker(path: Path) -> pl.DataFrame:
    """A ``book_ticker`` file as ``ts`` and ``exch`` (receive and exchange stamps, ns) and the
    touch, every two-sided row (a bid above 0 and an ask above it) in file order."""
    cols = ["timestamp", "local_timestamp", "ask_amount", "ask_price", "bid_price", "bid_amount"]
    f = pl.read_csv(path, columns=cols, schema_overrides={c: pl.Float64 for c in cols[2:]})
    _stamps(f, Path(path).name)
    bid, ask = pl.col("bid_price"), pl.col("ask_price")
    return f.filter((bid > 0) & (ask > bid)).select(
        ts=pl.col("local_timestamp") * 1000, exch=pl.col("timestamp") * 1000, bid=bid, ask=ask,
        bid_qty=pl.col("bid_amount"), ask_qty=pl.col("ask_amount"))  # fmt: skip


def read_trades(path: Path) -> pl.DataFrame:
    """A ``trades`` file as ``ts``, ``exch``, the aggressor's ``side`` (``B``/``S``),
    ``price`` and ``qty``, in file order."""
    cols = ["timestamp", "local_timestamp", "side", "price", "amount"]
    f = pl.read_csv(path, columns=cols, schema_overrides={"price": pl.Float64,
                                                          "amount": pl.Float64})  # fmt: skip
    _stamps(f, Path(path).name)
    return f.select(ts=pl.col("local_timestamp") * 1000, exch=pl.col("timestamp") * 1000,
                    side=pl.col("side").replace_strict({"buy": "B", "sell": "S"},
                                                       return_dtype=pl.String),
                    price="price", qty=pl.col("amount"))  # fmt: skip


def read_snapshot(path: Path, levels: int) -> pl.DataFrame:
    """A ``book_snapshot_N`` file's first ``levels`` levels as ``ts``, ``exch`` and
    ``bid{k}_px``, ``bid{k}_qty``, ``ask{k}_px``, ``ask{k}_qty`` from level 0."""
    names = {}
    for k in range(levels):
        for side, s in (("bids", "bid"), ("asks", "ask")):
            names[f"{side}[{k}].price"] = f"{s}{k}_px"
            names[f"{side}[{k}].amount"] = f"{s}{k}_qty"
    cols = ["timestamp", "local_timestamp", *names]
    f = pl.read_csv(path, columns=cols, schema_overrides={c: pl.Float64 for c in names})
    _stamps(f, Path(path).name)
    return f.select(ts=pl.col("local_timestamp") * 1000, exch=pl.col("timestamp") * 1000,
                    **{v: pl.col(k) for k, v in names.items()})  # fmt: skip


def funding_settlements(path: Path, start: int, end: int) -> tuple[pl.DataFrame, list[dict]]:
    """The settlements in ``[start, end)`` of a ``derivative_ticker`` file: for each
    ``funding_timestamp`` S, the ``funding_rate`` of the last row that names S.

    Binance's mark-price stream shows the rate of the next settlement and switches to the
    one after a moment past it, so the last row naming S carries the rate settled at S. It
    must arrive within :data:`~blinded.demo.FUNDING_SLACK` of S, the delay the brief gives
    the stream, so no rate is read before it was received; one later raises
    :class:`~blinded.demo.BuildError`. Returns the rows (``ts`` S, ``rate``) and, per
    settlement, the rate, the rows naming it and how long after S the last one arrived.
    """
    f = pl.read_csv(path, columns=["local_timestamp", "funding_timestamp", "funding_rate"],
                    schema_overrides={"funding_rate": pl.Float64})  # fmt: skip
    _stamps(f, Path(path).name)
    last = (f.drop_nulls("funding_timestamp")
            .group_by("funding_timestamp", maintain_order=True)
            .agg(pl.col("local_timestamp").last(), pl.col("funding_rate").last(), pl.len())
            .with_columns(s=pl.col("funding_timestamp") * 1000)
            .filter((pl.col("s") >= start) & (pl.col("s") < end)).sort("s"))  # fmt: skip
    found, slack = [], FUNDING_SLACK.value
    for s, local, rate, n in last.select("s", "local_timestamp", "funding_rate", "len").rows():
        after = local * 1000 - s
        when = pd.Timestamp(s, tz="UTC")
        if after >= slack:
            raise BuildError(f"the rate settled at {when} last arrived {after / MS:,.0f} ms after "
                             f"it, past the stream's {slack // MS:,} ms delay")  # fmt: skip
        found.append({"at": when, "rate": rate, "rows": n, "after_ms": after / MS})
    rows = pl.DataFrame({"ts": last["s"], "rate": last["funding_rate"]},
                        schema={"ts": pl.Int64, "rate": pl.Float64})  # fmt: skip
    return rows, found


# -- the layout ----------------------------------------------------------------------


def book_layout(book: pl.DataFrame, symbol: str, extra: list[str]) -> pl.DataFrame:
    """Book rows (``ts`` in ns, the touch, ``n``, and ``extra`` level columns) in the layout."""
    none = pl.lit(None, pl.Float64)
    return book.select(
        pl.col("ts").cast(pl.Datetime("ns")), symbol=pl.lit(symbol), kind=pl.lit("book"),
        is_trade=pl.lit(False), side=pl.lit(None, pl.String), price=none, qty=none,
        bid0_px="bid", ask0_px="ask", bid0_qty="bid_qty", ask0_qty="ask_qty",
        mid=(pl.col("bid") + pl.col("ask")) / 2,
        **{c: pl.col(c) if c in book.columns else none for c in extra},
        n=pl.col("n").cast(pl.Int64), rate=none)  # fmt: skip


def trade_layout(trades: pl.DataFrame, book: pl.DataFrame, symbol: str,
                 extra: list[str]) -> pl.DataFrame:  # fmt: skip
    """Every trade at its own stamp with the touch received at or before it, in the layout;
    a trade before the first book row has no book and is dropped."""
    none = pl.lit(None, pl.Float64)
    j = np.searchsorted(book["ts"].to_numpy(), trades["ts"].to_numpy(), side="right") - 1
    keep = j >= 0
    at = book[j[keep]].select(b="bid", a="ask", bq="bid_qty", aq="ask_qty")
    rows = pl.concat([trades.filter(pl.Series(keep)), at], how="horizontal")
    return rows.select(
        pl.col("ts").cast(pl.Datetime("ns")), symbol=pl.lit(symbol), kind=pl.lit("trades"),
        is_trade=pl.lit(True), side="side", price="price", qty="qty", bid0_px="b", ask0_px="a",
        bid0_qty="bq", ask0_qty="aq", mid=(pl.col("b") + pl.col("a")) / 2,
        **{c: none for c in extra}, n=pl.lit(1, pl.Int64), rate=none)  # fmt: skip


def funding_layout(rows: pl.DataFrame, symbol: str, extra: list[str]) -> pl.DataFrame:
    none = pl.lit(None, pl.Float64)
    return rows.select(
        pl.col("ts").cast(pl.Datetime("ns")), symbol=pl.lit(symbol), kind=pl.lit("funding"),
        is_trade=pl.lit(False), side=pl.lit(None, pl.String), price=none, qty=none,
        bid0_px=none, ask0_px=none, bid0_qty=none, ask0_qty=none, mid=none,
        **{c: none for c in extra}, n=pl.lit(None, pl.Int64), rate="rate")  # fmt: skip


def finish(parts: list[pl.DataFrame]) -> pl.DataFrame:
    """One frame by receive stamp; on a shared stamp, book rows, then trades, then funding, as
    the parts are listed. ``symbol``, ``kind`` and ``side`` are categorical."""
    frame = pl.concat(parts).sort("ts", maintain_order=True)
    return frame.with_columns(pl.col("symbol", "kind", "side").cast(pl.Categorical))


def lattice(ts: np.ndarray, start: int, end: int, step_ns: int, offsets_ns) -> tuple:
    """The book rows a read at each lattice stamp in ``[start, end)`` returns, each with the
    raw rows it stands for, and the last raw row too when no read keeps it, so the file
    ends where the raw one does (:func:`blinded.demo.thin_book`)."""
    points = read_points(start, end, step_ns, offsets_ns, start)
    idx, n = thin_book(ts, points)
    after = len(ts) - (int(idx[-1]) + 1 if len(idx) else 0)
    if after:
        idx, n = np.r_[idx, len(ts) - 1], np.r_[n, after]
    return idx, n


def levels_asof(snapshot: pl.DataFrame, stamps: np.ndarray, levels: int) -> pl.DataFrame:
    """The snapshot's levels 1 to ``levels - 1`` as-of each stamp, null before its first row."""
    cols = level_columns(levels)
    j = np.searchsorted(snapshot["ts"].to_numpy(), stamps, side="right") - 1
    out = {}
    for c in cols:
        x = snapshot[c].to_numpy()
        v = np.where(j >= 0, x[np.maximum(j, 0)], np.nan)
        out[c] = pl.Series(c, v, dtype=pl.Float64, nan_to_null=True)
    return pl.DataFrame(out)


def touch_layout(book: pl.DataFrame, tick: float) -> pl.DataFrame:
    """The touch file: every book row's stamp, its prices in ticks, its sizes."""
    from blinded.sim_maker import to_ticks

    return pl.DataFrame({
        "ts": pl.Series(book["ts"]).cast(pl.Datetime("ns")),
        "bid": to_ticks(book["bid"].to_numpy(), tick, "bid"),
        "ask": to_ticks(book["ask"].to_numpy(), tick, "ask"),
        "bid_qty": book["bid_qty"], "ask_qty": book["ask_qty"],
    })  # fmt: skip


def identity(raw_book: pl.DataFrame, kept: pl.DataFrame, others: list[pl.DataFrame], brief,
             symbol: str, hours: float = 1.0) -> dict:  # fmt: skip
    """The panel from every raw book row and from the lattice's rows over the first ``hours``
    must agree on every column, and the book on every read at a grid point and at a grid
    point plus the latency; raises :class:`~blinded.demo.BuildError` where not.

    ``raw_book`` and ``kept`` are in the layout (each raw row with ``n`` 1), ``others`` the
    trade and funding rows, which both files share.
    """
    from blinded.panel import Panel, load_events

    step = brief.grid.step.value
    origin = int(kept["ts"].dt.epoch("ns").min()) // step * step
    end = origin + int(hours * 3600e9)
    within = pl.col("ts").dt.epoch("ns") < end
    frames = [finish([b.filter(within), *(o.filter(within) for o in others)])
              for b in (raw_book, kept)]  # fmt: skip
    grid = np.arange(origin + step, end - brief.latency.value, step, dtype=np.int64)
    built = []
    for frame in frames:
        events = load_events(brief, frame)
        built.append((Panel.build(events, grid, brief.data.tz, step), events))
    (p_raw, e_raw), (p_thin, e_thin) = built
    if p_raw.frame.columns != p_thin.frame.columns:
        raise BuildError(f"thinning identity: the panels have different columns for {symbol}")
    for col in p_raw.frame.columns:
        a, b = p_raw.frame[col], p_thin.frame[col]
        same = (a == b).fill_null(a.is_null() & b.is_null()).to_numpy()
        if not same.all():
            at = p_raw.time(int(np.argmin(same)))
            raise BuildError(f"thinning identity: {col} differs at {at}")
    reads = 0
    for lat in (0, brief.latency.value):
        for f in ("bid", "ask", "bid_qty", "ask_qty"):
            a = e_raw[("book", symbol)].asof(f, grid + lat)
            b = e_thin[("book", symbol)].asof(f, grid + lat)
            if not np.array_equal(a, b, equal_nan=True):
                raise BuildError(f"thinning identity: the {f} read at grid + {lat // MS} ms "
                                 f"differs ({symbol})")  # fmt: skip
            reads += len(grid)
    return {"grid_points": len(grid), "columns": len(p_raw.frame.columns), "reads": reads,
            "raw_rows": frames[0].height, "kept_rows": frames[1].height}  # fmt: skip


def snapshot_check(touch: pl.DataFrame, snapshot: pl.DataFrame, name: str) -> dict:
    """:func:`blinded.bookstats.snapshot_agreement` of a snapshot against the touch stream it
    belongs to; raises :class:`~blinded.demo.BuildError` when the touch read as-of the
    snapshot's rows shows its level 0 prices on fewer than :data:`SNAPSHOT_SHARE` of them."""
    out = bookstats.snapshot_agreement(
        *(touch[c].to_numpy() for c in ("ts", "bid", "ask", "bid_qty", "ask_qty")),
        *(snapshot[c].to_numpy() for c in ("ts", "bid0_px", "ask0_px", "bid0_qty", "ask0_qty")),
    )
    out["delay_ms"] = {
        "touch": bookstats.delay_ms(touch["ts"], touch["exch"]),
        "snapshot": bookstats.delay_ms(snapshot["ts"], snapshot["exch"]),
    }
    if out["reverse"][0] < SNAPSHOT_SHARE:
        raise BuildError(f"{name}: the touch as-of each snapshot row shows its level 0 prices on "
                         f"{out['reverse'][0]:.1%} of them, under {SNAPSHOT_SHARE:.0%}: the two "
                         "streams are not one book; nothing written")  # fmt: skip
    return out


def day_facts(touch: pl.DataFrame, trades: pl.DataFrame, tick: float, end: int) -> dict:
    """The rows of one instrument from its raw rows: the touch's
    (:func:`blinded.bookstats.touch_facts`), the aggressors' and the feed delay."""
    from blinded.sim_maker import to_ticks

    side = np.where(trades["side"].to_numpy() == "B", 1, -1)
    facts = bookstats.touch_facts(
        touch["ts"].to_numpy(),
        to_ticks(touch["bid"].to_numpy(), tick),
        to_ticks(touch["ask"].to_numpy(), tick),
        touch["bid_qty"].to_numpy(),
        touch["ask_qty"].to_numpy(),
        tick,
        end,
        trades["ts"].to_numpy(),
        side,
        to_ticks(trades["price"].to_numpy(), tick, "trade price"),
        trades["qty"].to_numpy(),
    )
    facts["aggressors"] = bookstats.aggressors(trades["exch"].to_numpy(),
                                               trades["price"].to_numpy(),
                                               trades["qty"].to_numpy(), tick)  # fmt: skip
    facts["delay_ms"] = bookstats.delay_ms(touch["ts"], touch["exch"])
    return facts


# -- the perpetual's file -----------------------------------------


def shedding_skew(size: float, max_inventory: float) -> float:
    """The default ``skew`` of a market-making brief: the largest power of two, in ticks a
    unit, that keeps every shedding quote at or behind the touch at any inventory the cap
    allows with one more order in flight, ``skew * (max_inventory + size) < 1``.

    On a book one tick wide, a shedding quote a whole tick inside the touch is on the far
    side, and post-only rejects it at every decision while the cap withdraws the other side:
    the inventory cannot be shed. With the offset rounded away from the touch, a skew under
    one tick over the largest inventory the quote can know keeps it at the touch at worst.

    Examples
    --------
    >>> shedding_skew(1.0, 2.0)  # under 1/3
    0.25
    >>> shedding_skew(1.0, 3.0)  # 1/4 itself would reach a whole tick
    0.125
    """
    reach = float(max_inventory) + float(size)
    return 2.0 ** -(math.floor(math.log2(reach)) + 1)


def mm_brief(data: str, touch: str, symbol: str, tick: float, lot: float | None, day: Date,
             latency_ms: float, levels: int) -> dict:  # fmt: skip
    """The brief of a market-making file."""
    inst = f"{symbol}_PERP"
    fields = {"bid": "bid0_px", "ask": "ask0_px", "bid_qty": "bid0_qty", "ask_qty": "ask0_qty",
              "n": "n", **{c: c for c in level_columns(levels)}}  # fmt: skip
    lat = int(latency_ms) if float(latency_ms).is_integer() else float(latency_ms)
    return {
        "blinded": 0,
        "name": data.removesuffix(".parquet"),
        "data": {
            "path": data, "format": "parquet",
            "time": {"column": "ts", "unit": "ns", "tz": "UTC", "clock": "receive"},
            "instrument_column": "symbol",
            "streams": {
                "book": {"filter": "kind == 'book'", "fields": fields},
                "trades": {"filter": "kind == 'trades'",
                           "fields": {"price": "price", "qty": "qty", "side": "side"},
                           "side_values": {"buy": "B", "sell": "S"}},
                "funding": {"filter": "kind == 'funding'", "fields": {"rate": "rate"}},
            },
            "availability": {"default_delay_ms": 0, "overrides": [
                {"stream": "funding", "delay_ms": int(FUNDING_SLACK / pd.Timedelta("1ms"))}]},
            "thinned": {"step": STEP, "offsets_ms": [0, lat]},
            "touch": {"path": touch, "tick_size": tick, "instrument": inst},
        },
        "target": {"instrument": inst, "price": "mid", "horizons": ["100ms", "1s", "10s", "60s"],
                   "primary_horizon": "10s"},
        "grid": {"step": STEP, "start": f"{day.isoformat()}T00:00:00Z", "warmup": "5min"},
        "execution": {"latency_ms": lat, "taker_fee_bps": 5.0, "maker_fee_bps": 2.0,
                      "max_position": 2.0, "min_rebalance": 0.1, "tick_size": tick,
                      **({"lot_size": lot} if lot else {})},
        "splits": {"scheme": "time", "fractions": [0.5, 0.25, 0.25], "embargo": "5min"},
        "stats": {"se": "both", "block": "5min", "n_boot": 1000, "seed": 20241128},
        # skew a quarter of a tick a unit at size 1 and cap 2, not one or a half:
        # at an inventory of the cap plus a partial fill in flight, half a tick a unit puts the
        # shedding quote a tick inside a one-tick book, where post-only rejects it at every
        # decision (shedding_skew)
        "archetypes": {"instruments": [inst],
                       "params": {"market_making": {"size": 1.0,
                                                    "skew": shedding_skew(1.0, 2.0),
                                                    "max_inventory": 2.0}}},
    }  # fmt: skip


@dataclass
class BuiltMM:
    """What :func:`mm` wrote, and the numbers it checked on the way."""

    data: Path
    touch: Path
    brief: Path
    notes: Path
    symbol: str
    day: Date
    hours: float
    tick: float
    lot: float | None
    rows: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    touch_check: dict = field(default_factory=dict)
    snapshot: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    funding: list = field(default_factory=list)


def mm(
    symbol: str,
    day: Date | str,
    out: str | Path,
    raw: str | Path | None = None,
    levels: int = 5,
    latency_ms: float = 50.0,
    hours: float | None = None,
    log: Callable[[str], None] = print,
    details: Callable[[str], dict] | None = None,
    fetch_day: Callable[..., None] | None = None,
    api_key: str | None = None,
) -> BuiltMM:
    """Build the market-making file of one Binance USD-M perpetual's free Tardis day.

    Verifies the symbol and its four types against ``get_exchange_details``
    before anything is fetched, fetches what ``raw`` lacks, and writes under
    ``out``: the data file (book rows thinned to the lattice of a 100 ms grid
    from midnight read at 0 and ``latency_ms``, each with ``n`` and the
    snapshot's levels 1 to ``levels - 1`` as-of its stamp; every trade with
    the touch as received; the funding settlements of the day, from
    :func:`funding_settlements`), the touch file, the brief and ``DATA-raw.md``.
    With ``hours`` the day is cut at that many hours from midnight. Refuses
    (:class:`~blinded.demo.BuildError`) a book off its tick, a snapshot that
    is not the touch stream's book (:func:`snapshot_check`), and a thinned
    file whose first hour's panel is not the raw file's (:func:`identity`); the
    touch file must pass :func:`blinded.vault.check_touch` on every stamp.
    ``details`` and ``fetch_day`` replace the Tardis client in tests.
    """
    from blinded import brief as briefs
    from blinded.sim_maker import infer_tick
    from blinded.vault import check_touch

    day = Date.fromisoformat(day) if isinstance(day, str) else day
    symbol = symbol.upper()
    key = api_key if api_key is not None else os.environ.get("TARDIS_API_KEY", "")
    out = Path(out).expanduser()
    raw = Path(raw).expanduser() if raw else out / "raw"
    types = MM_TYPES if levels > 1 else tuple(t for t in MM_TYPES if t != "book_snapshot_5")
    details, fetch_day = _client(details, fetch_day, key)
    problem = listing_problem(details(PERP_EXCHANGE), PERP_EXCHANGE, symbol, day, types,
                              "perpetual", key)  # fmt: skip
    if problem:
        raise BuildError(f"{problem}; nothing was downloaded")
    files = fetch_files([(PERP_EXCHANGE, t, symbol) for t in types], day, raw, fetch_day, log)
    start = int(pd.Timestamp(day).value)
    end = start + (int(hours * 3600e9) if hours else DAY_NS)
    path = {t: raw / raw_name(PERP_EXCHANGE, t, day, symbol) for t in types}
    within = (pl.col("ts") >= start) & (pl.col("ts") < end)
    ticker = read_ticker(path["book_ticker"]).filter(within)
    trades = read_trades(path["trades"]).filter(within)
    # a symbol whose tick is not on record takes the book's smallest step; either way every
    # price must be a whole number of ticks (touch_layout)
    tick, lot = PERP_STEPS.get(symbol) or (
        infer_tick(ticker["bid"].to_numpy(), ticker["ask"].to_numpy()),
        None,
    )
    inst = f"{symbol}_PERP"
    extra = level_columns(levels)
    touch = touch_layout(ticker, tick)
    lat_ns = int(round(latency_ms * MS))
    idx, n = lattice(ticker["ts"].to_numpy(), start, end, pd.Timedelta(STEP).value,
                     sorted({0, lat_ns}))  # fmt: skip
    kept = ticker[idx].with_columns(n=pl.Series(n, dtype=pl.Int64))
    snapshot = None
    if levels > 1:
        snapshot = read_snapshot(path["book_snapshot_5"], levels).filter(within)
        kept = pl.concat([kept, levels_asof(snapshot, kept["ts"].to_numpy(), levels)],
                         how="horizontal")  # fmt: skip
    settled, found = funding_settlements(path["derivative_ticker"], start, end)
    others = [trade_layout(trades, ticker, inst, extra), funding_layout(settled, inst, extra)]
    kept_rows = book_layout(kept, inst, extra)
    stem = f"tardis-binance-{symbol.lower()}-perp-mm-{day.isoformat()}"
    if hours:
        stem += f"-{hours:g}h"
    built = BuiltMM(out / f"{stem}.parquet", out / f"{stem}.touch.parquet",
                    out / f"{stem}.brief.yaml", out / "DATA-raw.md", symbol, day,
                    hours or 24.0, tick, lot, files=files, funding=found)  # fmt: skip
    raw_brief = mm_brief(built.data.name, built.touch.name, symbol, tick, lot, day, latency_ms,
                         levels)  # fmt: skip
    brief = briefs.parse(raw_brief, base=out)
    first = ticker.filter(pl.col("ts") < start + 3_600 * 10**9).with_columns(n=pl.lit(1))
    if snapshot is not None:
        first = pl.concat([first, levels_asof(snapshot, first["ts"].to_numpy(), levels)],
                          how="horizontal")  # fmt: skip
    built.identity = identity(book_layout(first, inst, extra), kept_rows, others, brief, inst)
    ident = built.identity
    log(f"thinning identity holds on the first hour: {ident['grid_points']:,} grid points, "
        f"{ident['columns']} columns, {ident['reads']:,} reads")  # fmt: skip
    if snapshot is not None:
        built.snapshot = snapshot_check(ticker, snapshot, f"{inst}'s book_snapshot_{levels}")
    frame = finish([kept_rows, *others])
    del kept, first
    built.touch_check = check_touch(brief, frame, touch)
    log(f"the touch file is the data file's book at all {built.touch_check['stamps_checked']:,} "
        "of its stamps")  # fmt: skip
    built.facts = day_facts(ticker, trades, tick, end)
    built.rows = {"ticker": ticker.height, "kept": kept_rows.height, "trades": others[0].height,
                  "funding": others[1].height, "snapshot": 0 if snapshot is None
                  else snapshot.height, "data": frame.height}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(built.data)
    touch.write_parquet(built.touch)
    built.brief.write_text(yaml.safe_dump(raw_brief, sort_keys=False))
    built.notes.write_text(mm_notes(built, latency_ms, levels))
    log(
        f"{frame.height:,} rows ({kept_rows.height:,} book, {others[0].height:,} trades, "
        f"{others[1].height} funding) and {touch.height:,} touch rows; wrote {built.data.name}"
    )
    return built


def _num(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "none"
    return f"{x:,.{digits}g}" if isinstance(x, float) else f"{x:,}"


def _pct(x, digits: int = 1) -> str:
    return "none" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def facts_lines(f: dict) -> list[str]:
    """The table for one instrument, as markdown rows."""
    d, a = f.get("delay_ms") or {}, f.get("aggressors")
    rows = [
        ("mid, median; tick", f"{_num(f['mid'], 6)}; {f['tick']:g} = {f['tick_bps']:.3f} bps"),
        ("spread one tick wide, share of time; two ticks",
         f"{_pct(f['one_tick_share'], 2)}; {_pct(f['two_tick_share'], 2)}"),
        ("book rows (a second); distinct stamps",
         f"{f['rows']:,} ({f['rows_per_second']:.0f}); {f['distinct_stamps']:,}"),
        ("trades (a second)", f"{f['trades']:,} ({f['trades_per_second']:.1f})"),
    ]  # fmt: skip
    if a:
        rows += [("aggressor orders (trades sharing an exchange stamp); size median; p90; p99",
                  f"{a['orders']:,}; {_num(a['size_median'])}; {_num(a['size_p90'])}; "
                  f"{_num(a['size_p99'])}"),
                 ("aggressors walking more than one tick", _pct(a["walked"]))]  # fmt: skip
    if d:
        rows.append(("feed delay, receive less exchange stamp, median; p99 (ms)",
                     "none (no exchange stamp)" if d["median"] is None
                     else f"{d['median']:.2f}; {d['p99']:.1f}"))  # fmt: skip
    rows += [
        ("displayed at the touch, median bid; ask; bid in USD",
         f"{_num(f['bid_qty_median'])}; {_num(f['ask_qty_median'])}; "
         f"{f['touch_usd_median']:,.0f}"),
        ("falls of the best bid; traded out; pulled; partly",
         f"{f['falls']:,}; {_pct(f['traded_out'])}; {_pct(f['pulled'])}; {_pct(f['partly'])}"),
        ("median time between changes of the best bid price",
         f"{_num(f['bid_change_median_ms'])} ms"),
        ("rows at an unchanged bid price; of them with the bid's size changed",
         f"{f['bid_same_rows']:,}; {f['bid_size_changes']:,}"),
    ]  # fmt: skip
    return ["| measure | value |", "|---|---|", *(f"| {k} | {v} |" for k, v in rows)]


def snapshot_lines(s: dict, what: str) -> list[str]:
    """The depth check's column for one venue, as a paragraph."""
    fw, d = s["forward"], s.get("delay_ms")

    def delay(x):
        return "none (no exchange stamp)" if x["median"] is None else f"{x['median']:.1f}"

    lags = ", ".join(f"{_pct(p)} and {_pct(q)} at {lag} ms" for lag, (p, q) in fw.items())
    when = (f"; collector delay, median ms, touch {delay(d['touch'])}, snapshot "
            f"{delay(d['snapshot'])}") if d else ""  # fmt: skip
    return [
        f"{what}: {s['touch_rows']:,} touch rows and {s['snapshot_rows']:,} snapshot rows{when}. "
        f"The snapshot's level 0 as-of each touch row's stamp plus a lag shows the touch's "
        f"prices and sizes on {lags}; the touch as-of each snapshot row shows its level 0 "
        f"prices on {_pct(s['reverse'][0])} and sizes on {_pct(s['reverse'][1])} of them "
        f"(the build refuses under {SNAPSHOT_SHARE:.0%}); {_pct(s['crossed'])} of touch rows "
        "cross the as-of snapshot's level 0.",
    ]


def mm_notes(built: BuiltMM, latency_ms: float, levels: int) -> str:
    """``DATA-raw.md`` of a market-making file: a head about the files, and the day's section,
    which a protected run's DATA.md cuts (:func:`blinded.demo.notes_until`)."""
    inst = f"{built.symbol}_PERP"
    ident = built.identity
    head = [
        f"# The raw files behind {built.data.name}",
        "",
        f"Built by `blinded demo mm` from the Tardis free day {built.day}: the {inst} USD-M "
        f"perpetual (`binance-futures` `book_ticker`, `trades`, `book_snapshot_5` and "
        f"`derivative_ticker` of {built.symbol}), every stamp Tardis's `local_timestamp`, the "
        f"recorder's receive clock, the first {built.hours:g} hours of the day. Tardis files have "
        "no checksum; their sizes and SHA-256 are these:",
        "",
        "| file | bytes | sha256 |",
        "|---|---|---|",
        *(f"| {n} | {f['bytes']:,} | `{f['sha256'][:16]}` |" for n, f in built.files.items()),
        "",
        f"The data file keeps the book rows a read at a grid point or at a grid point plus "
        f"{latency_ms:g} ms returns (a 100 ms grid from midnight), each with `n`, the raw rows it "
        f"stands for, and levels 1 to {levels - 1} of the snapshot as-of its stamp; every trade "
        "at its own stamp with the touch as received; and the day's funding settlements, each "
        "the rate of the last `derivative_ticker` row that names it, available 1 s after it. "
        f"The touch file `{built.touch.name}` holds every two-sided `book_ticker` row, prices in "
        f"ticks of {built.tick:g}; only the fill engine reads it. Tick {built.tick:g} and lot "
        f"{built.lot if built.lot else 'none'} from Binance's exchangeInfo (read 19 Oct 2026); "
        "every book price is a whole number of ticks.",
        "",
        f"## {built.day.isoformat()}",
        "",
        "| rows | count |",
        "|---|---|",
        *(f"| {k} | {v:,} |" for k, v in built.rows.items()),
        "",
        f"The build's checks. Thinning identity on the first hour: {ident['grid_points']:,} grid "
        f"points, every one of {ident['columns']} panel columns equal between the raw book and "
        f"the kept rows, and {ident['reads']:,} reads of the book at grid points and fills equal. "
        "The touch file equals the data file's book at all "
        f"{built.touch_check['stamps_checked']:,} of its stamps.",
        "",
    ]  # fmt: skip
    if built.snapshot:
        head += [*snapshot_lines(built.snapshot, f"{inst}'s book_snapshot_{levels}"), ""]
    head += [f"The touch stream over the {built.hours:g} hours:", "",
             *facts_lines(built.facts), "", "Funding settlements:", "",
             "| settlement | rate | rows naming it | last arrived after |", "|---|---|---|---|",
             *(f"| {f['at']:%Y-%m-%d %H:%M} | {f['rate']:.8f} | {f['rows']:,} | "
               f"{f['after_ms']:,.0f} ms |" for f in built.funding), ""]  # fmt: skip
    return "\n".join(head) + "\n"


# -- venues on one clock: the arbitrage pair and the consolidated book -------

TAGS = {"mexc": "MEXC", "binance": "BINANCE", "okex": "OKX", "bybit-spot": "BYBIT"}
# spot ticks and lots from both venues' exchangeInfo as read on 20 Oct
# 2026; a pair not here takes the book's smallest step
SPOT_STEPS = {("mexc", "ATUSDT"): (1e-5, 1.0), ("binance", "ATUSDT"): (1e-4, None),
              ("mexc", "SOLUSDT"): (0.01, 0.01), ("binance", "SOLUSDT"): (0.01, None)}  # fmt: skip
ARB_TYPES = ("book_ticker", "trades", "book_snapshot_25")
VENUE_TYPES = ("book_ticker", "trades")
HEDGE_FEE_BPS = 10.0  # Binance spot, regular tier
MEXC_TAKER_BPS = {"ATUSDT": 8.0, "SOLUSDT": 5.0}  # MEXC's exchangeInfo, takerCommission
ARB_DISTANCE_BPS = 30.0
SPREAD_FEES = 3  # the quote venue's spread must be this many hedge fees wide
DEPTH_LEVELS = 25
DEPTH_LAG_MS = 100  # Binance spot's depth stream trails its bookTicker by up to 100 ms (1.8)
STOCK = {"coin_usd": 500.0, "usdt": 500.0, "rebalance_at": 0.8, "transfer_minutes": 30}
CONSOLIDATED = (("binance", "SOLUSDT"), ("okex", "SOL-USDT"), ("bybit-spot", "SOLUSDT"),
                ("mexc", "SOLUSDT"))  # fmt: skip
OTHER_DAYS = (Date(2026, 9, 1), Date(2026, 8, 1), Date(2026, 7, 1))


def instrument(coin: str, exchange: str) -> str:
    """``ATUSDT`` on ``mexc`` is ``ATUSDT_MEXC``."""
    return f"{coin}_{TAGS[exchange]}"


def venue_symbol(coin: str, exchange: str) -> str:
    """Tardis's symbol of a spot coin on one exchange: OKX writes ``SOL-USDT``."""
    return f"{coin.removesuffix('USDT')}-USDT" if exchange == "okex" else coin


@dataclass
class Leg:
    """One venue's raw rows of a day: the touch, the trades, and its instrument and tick."""

    exchange: str
    instrument: str
    tick: float
    lot: float | None
    ticker: pl.DataFrame
    trades: pl.DataFrame

    def book_rows(self, extra: list[str]) -> pl.DataFrame:
        """Every book row in the layout, with ``n`` 1 and ``age_ms``, how long after the
        exchange's stamp the recorder received it (0 where the venue sends no stamp)."""
        book = self.ticker.with_columns(n=pl.lit(1, pl.Int64),
                                        age_ms=(pl.col("ts") - pl.col("exch")) / MS)  # fmt: skip
        return book_layout(book, self.instrument, extra)

    def trade_rows(self, extra: list[str]) -> pl.DataFrame:
        return trade_layout(self.trades, self.ticker, self.instrument, extra)

    def facts(self, end: int) -> dict:
        t = self.ticker
        age = ((t["ts"] - t["exch"]) / MS).to_numpy()
        return bookstats.venue_facts(t["ts"].to_numpy(), t["bid"].to_numpy(), t["ask"].to_numpy(),
                                     t["bid_qty"].to_numpy(), end, self.trades["ts"].to_numpy(),
                                     self.trades["price"].to_numpy(),
                                     self.trades["qty"].to_numpy(), age, self.tick)  # fmt: skip

    def flow(self) -> dict:
        t, x = self.ticker, self.trades
        side = np.where(x["side"].to_numpy() == "B", 1, -1)
        return bookstats.flow_facts(t["ts"].to_numpy(), t["bid"].to_numpy(), t["ask"].to_numpy(),
                                    x["ts"].to_numpy(), side, x["price"].to_numpy(),
                                    x["qty"].to_numpy())  # fmt: skip


def read_leg(raw: Path, exchange: str, coin: str, day: Date, start: int, end: int) -> Leg:
    """One spot venue's ``book_ticker`` and ``trades`` of ``coin`` in ``[start, end)``; its
    tick from :data:`SPOT_STEPS` or the book's smallest step, every price a whole tick."""
    from blinded.sim_maker import infer_tick, to_ticks

    sym = venue_symbol(coin, exchange)
    within = (pl.col("ts") >= start) & (pl.col("ts") < end)
    ticker = read_ticker(raw / raw_name(exchange, "book_ticker", day, sym)).filter(within)
    trades = read_trades(raw / raw_name(exchange, "trades", day, sym)).filter(within)
    tick, lot = SPOT_STEPS.get((exchange, coin)) or (
        infer_tick(ticker["bid"].to_numpy(), ticker["ask"].to_numpy()),
        None,
    )
    for name, x in (
        ("bid", ticker["bid"]),
        ("ask", ticker["ask"]),
        ("trade price", trades["price"]),
    ):
        to_ticks(x.to_numpy(), tick, f"{exchange} {sym} {name}")
    return Leg(exchange, instrument(coin, exchange), tick, lot, ticker, trades)


def depth_layout(snapshot: pl.DataFrame, levels: int) -> pl.DataFrame:
    """The depth file: every snapshot row's stamp and its levels in price units."""
    cols = [f"{s}{k}_{f}" for s in ("bid", "ask") for k in range(levels) for f in ("px", "qty")]
    return snapshot.select(pl.col("ts").cast(pl.Datetime("ns")), *cols)


def _verify(details, legs, types, day, key) -> None:
    found = (listing_problem(details(ex), ex, sym, day, types, "spot", key) for ex, sym in legs)
    problems = [p for p in found if p]
    if problems:
        raise BuildError("; ".join(problems) + "; nothing was downloaded")


def _streams(extra: list[str]) -> dict:
    fields = {"bid": "bid0_px", "ask": "ask0_px", "bid_qty": "bid0_qty", "ask_qty": "ask0_qty",
              "n": "n", **{c: c for c in extra}}  # fmt: skip
    return {"book": {"filter": "kind == 'book'", "fields": fields},
            "trades": {"filter": "kind == 'trades'",
                       "fields": {"price": "price", "qty": "qty", "side": "side"},
                       "side_values": {"buy": "B", "sell": "S"}}}  # fmt: skip


def _venue_brief(data: str, day: Date, target: str, extra: list[str], execution: dict,
                 archetypes: dict) -> dict:  # fmt: skip
    return {
        "blinded": 0, "name": data.removesuffix(".parquet"),
        "data": {"path": data, "format": "parquet",
                 "time": {"column": "ts", "unit": "ns", "tz": "UTC", "clock": "receive"},
                 "instrument_column": "symbol", "streams": _streams(extra),
                 "availability": {"default_delay_ms": 0, "overrides": []}},
        "target": {"instrument": target, "price": "mid",
                   "horizons": ["100ms", "1s", "10s", "60s"], "primary_horizon": "10s"},
        "grid": {"step": STEP, "start": f"{day.isoformat()}T00:00:00Z", "warmup": "5min"},
        "execution": execution,
        "splits": {"scheme": "time", "fractions": [0.5, 0.25, 0.25], "embargo": "5min"},
        "stats": {"se": "both", "block": "5min", "n_boot": 1000, "seed": 20241128},
        "archetypes": archetypes,
    }  # fmt: skip


@dataclass
class BuiltVenueFile:
    """What :func:`arb` or :func:`venues` wrote, and what it measured on the way."""

    data: Path
    brief: Path
    notes: Path
    coin: str
    day: Date
    legs: list
    touch: Path | None = None
    depth: Path | None = None
    rows: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    cross: dict = field(default_factory=dict)
    flow: dict = field(default_factory=dict)
    snapshots: dict = field(default_factory=dict)
    touch_check: dict = field(default_factory=dict)
    other_days: dict = field(default_factory=dict)


ARB_EXTRA = [*level_columns(5), "age_ms"]  # the data file's columns past the touch


def arb_frames(a: Leg, b: Leg, hedge_snapshot: pl.DataFrame
               ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:  # fmt: skip
    """The arbitrage file's three frames from the two legs' raw rows and the hedge venue's
    snapshot rows (``ts``, ``exch`` and ``bid{k}_px`` ... ``ask{k}_qty`` from level 0): the
    data file (every book row of both venues with ``age_ms`` and the hedge venue's levels 1
    to 4 as-of each of its rows, every trade with its own venue's touch), the quote venue's
    touch file and the hedge venue's depth file. ``blinded demo arb`` builds a Tardis day
    with it and the recorder (:mod:`blinded.recorder`) a recorded one."""
    hedge_book = pl.concat([b.ticker, levels_asof(hedge_snapshot, b.ticker["ts"].to_numpy(),
                                                  5)], how="horizontal")  # fmt: skip
    hb = Leg(b.exchange, b.instrument, b.tick, b.lot, hedge_book, b.trades)
    frame = finish([a.book_rows(ARB_EXTRA), hb.book_rows(ARB_EXTRA), a.trade_rows(ARB_EXTRA),
                    b.trade_rows(ARB_EXTRA)])  # fmt: skip
    return frame, touch_layout(a.ticker, a.tick), depth_layout(hedge_snapshot, DEPTH_LEVELS)


def arb_brief(built: BuiltVenueFile, a: Leg, b: Leg, mid: float, hedge_fee_bps: float,
              latencies: dict) -> dict:  # fmt: skip
    """The arbitrage file's brief, as a mapping: ``latencies`` are ``latency_ms`` and any of
    ``cancel_latency_ms``, ``response_latency_ms`` and ``hedge_latency_ms`` (the order
    latency when not given); the stock is :data:`STOCK` at ``mid``, the quote venue's, and
    its withdrawal fees are unset."""
    coins = float(np.floor(STOCK["coin_usd"] / mid))
    size = float(np.floor(10.0 / mid / (a.lot or 1)) * (a.lot or 1))

    def ms(x):
        return int(x) if float(x).is_integer() else float(x)

    lat = {k: ms(v) for k, v in latencies.items() if v is not None}
    lat.setdefault("hedge_latency_ms", lat["latency_ms"])
    order = ("latency_ms", "cancel_latency_ms", "response_latency_ms", "hedge_latency_ms")
    execution = {**{k: lat[k] for k in order if k in lat},
                 "taker_fee_bps": MEXC_TAKER_BPS.get(built.coin, hedge_fee_bps),
                 "maker_fee_bps": 0.0,
                 "hedge_fee_bps": hedge_fee_bps, "fees": {b.instrument: hedge_fee_bps},
                 "max_position": 1.0, "min_rebalance": 0.1, "tick_size": a.tick,
                 **({"lot_size": a.lot} if a.lot else {})}  # fmt: skip
    archetypes = {"instruments": [a.instrument, b.instrument],
                  "cross_venue": [[a.instrument, b.instrument]],
                  "params": {"market_making": {"size": size, "skew": 0.0,
                                               "max_inventory": 10 * size}}}  # fmt: skip
    raw = _venue_brief(built.data.name, built.day, a.instrument, ARB_EXTRA, execution,
                       archetypes)  # fmt: skip
    raw["data"]["touch"] = {"path": built.touch.name, "tick_size": a.tick,
                            "instrument": a.instrument}  # fmt: skip
    raw["data"]["depth"] = {"path": built.depth.name, "venue": b.instrument,
                            "levels": DEPTH_LEVELS, "lag_ms": DEPTH_LAG_MS}  # fmt: skip
    raw["stock"] = {a.instrument: {"coin": coins, "usdt": STOCK["usdt"]},
                    b.instrument: {"coin": coins, "usdt": STOCK["usdt"]},
                    "rebalance_at": STOCK["rebalance_at"],
                    "transfer_minutes": STOCK["transfer_minutes"],
                    "withdrawal_fee_coin": None,
                    "withdrawal_fee_usdt": None}  # fmt: skip
    return raw


def arb(
    coin: str = "ATUSDT",
    day: Date | str = "2026-10-01",
    out: str | Path = ".",
    raw: str | Path | None = None,
    quote: str = "mexc",
    hedge: str = "binance",
    hedge_fee_bps: float = HEDGE_FEE_BPS,
    latency_ms: float = 50.0,
    log: Callable[[str], None] = print,
    details: Callable[[str], dict] | None = None,
    fetch_day: Callable[..., None] | None = None,
    api_key: str | None = None,
) -> BuiltVenueFile:
    """Build the arbitrage file: ``coin`` on a thin quote
    venue and a deep hedge venue, spot, one free Tardis day on the receive clock.

    Verifies both venues' ``book_ticker``, ``trades`` and ``book_snapshot_25`` against
    ``get_exchange_details`` before anything is fetched. Writes the data file (every book
    row of both venues with ``age_ms``, the hedge venue's levels 1 to 4 as-of each of its
    rows, every trade with its own venue's touch as received), the quote venue's touch file,
    the hedge venue's depth file (every snapshot row, 25 levels), the brief and
    ``DATA-raw.md``. Refuses a day whose quote venue's spread, time-weighted median, is under
    three hedge fees, and a snapshot that is not its touch stream's book
    (:func:`snapshot_check`); the touch file must pass :func:`blinded.vault.check_touch`.
    """
    from blinded import brief as briefs
    from blinded.vault import check_touch

    day = Date.fromisoformat(day) if isinstance(day, str) else day
    coin = coin.upper()
    key = api_key if api_key is not None else os.environ.get("TARDIS_API_KEY", "")
    out = Path(out).expanduser()
    raw = Path(raw).expanduser() if raw else out / "raw"
    details, fetch_day = _client(details, fetch_day, key)
    pairs = [(quote, venue_symbol(coin, quote)), (hedge, venue_symbol(coin, hedge))]
    _verify(details, pairs, ARB_TYPES, day, key)
    files = fetch_files([(ex, t, s) for ex, s in pairs for t in ARB_TYPES], day, raw, fetch_day,
                        log)  # fmt: skip
    start = int(pd.Timestamp(day).value)
    end = start + DAY_NS
    a, b = (read_leg(raw, ex, coin, day, start, end) for ex in (quote, hedge))
    facts = {leg.instrument: leg.facts(end) for leg in (a, b)}
    spread = facts[a.instrument]["spread_bps"]
    if spread < SPREAD_FEES * hedge_fee_bps:
        raise BuildError(f"{a.instrument}'s spread on {day} is {spread:.1f} bps, time-weighted "
                         f"median, under {SPREAD_FEES} hedge fees of {hedge_fee_bps:g} bps: a "
                         "day with a maker on the quote venue, where the probe found 1 to 14 "
                         "fills a day; nothing written")  # fmt: skip
    within = (pl.col("ts") >= start) & (pl.col("ts") < end)
    snaps = {leg.instrument: read_snapshot(raw / raw_name(leg.exchange, "book_snapshot_25", day,
                                                          venue_symbol(coin, leg.exchange)),
                                           DEPTH_LEVELS).filter(within)
             for leg in (a, b)}  # fmt: skip
    built = BuiltVenueFile(Path(), Path(), out / "DATA-raw.md", coin, day, [a, b], files=files,
                           facts=facts)  # fmt: skip
    for leg in (a, b):
        built.snapshots[leg.instrument] = snapshot_check(
            leg.ticker, snaps[leg.instrument], f"{leg.instrument}'s book_snapshot_25"
        )
    del snaps[a.instrument]
    frame, touch, depth = arb_frames(a, b, snaps[b.instrument])
    stem = f"tardis-arb-{coin.lower()}-{quote}-{hedge}-{day.isoformat()}"
    built.data, built.brief = out / f"{stem}.parquet", out / f"{stem}.brief.yaml"
    built.touch, built.depth = out / f"{stem}.touch.parquet", out / f"{stem}.depth.parquet"
    raw_brief = arb_brief(built, a, b, facts[a.instrument]["mid"], hedge_fee_bps,
                          {"latency_ms": latency_ms})  # fmt: skip
    brief = briefs.parse(raw_brief, base=out)
    built.touch_check = check_touch(brief, frame, touch)
    log(f"the touch file is the data file's book of {a.instrument} at all "
        f"{built.touch_check['stamps_checked']:,} of its stamps")  # fmt: skip
    tops = [{c: leg.ticker[c].to_numpy() for c in ("ts", "bid", "ask")} for leg in (a, b)]
    built.cross = bookstats.cross_facts(*tops, end, ARB_DISTANCE_BPS)
    built.flow = {leg.instrument: leg.flow() for leg in (a, b)}
    for other in OTHER_DAYS:
        path = raw / raw_name(quote, "book_ticker", other, venue_symbol(coin, quote))
        if path.is_file():
            t = read_ticker(path)
            s0 = int(pd.Timestamp(other).value)
            sp = ((t["ask"] - t["bid"]) / (t["ask"] + t["bid"]) * 2e4).to_numpy()
            built.other_days[other] = bookstats.tw_quantile(t["ts"].to_numpy(), sp, 0.5,
                                                            s0 + DAY_NS)  # fmt: skip
    built.rows = {"data": frame.height, "touch": touch.height, "depth": depth.height,
                  **{f"{leg.instrument} book": leg.ticker.height for leg in (a, b)},
                  **{f"{leg.instrument} trades": leg.trades.height for leg in (a, b)}}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(built.data)
    touch.write_parquet(built.touch)
    depth.write_parquet(built.depth)
    built.brief.write_text(yaml.safe_dump(raw_brief, sort_keys=False))
    built.notes.write_text(arb_notes(built, hedge_fee_bps))
    log(f"{frame.height:,} rows, {touch.height:,} touch rows, {depth.height:,} depth rows; wrote "
        f"{built.data.name}")  # fmt: skip
    return built


def venues(
    coin: str = "SOLUSDT",
    day: Date | str = "2026-10-01",
    out: str | Path = ".",
    raw: str | Path | None = None,
    legs: tuple[tuple[str, str], ...] = CONSOLIDATED,
    log: Callable[[str], None] = print,
    details: Callable[[str], dict] | None = None,
    fetch_day: Callable[..., None] | None = None,
    api_key: str | None = None,
) -> BuiltVenueFile:
    """Build the consolidated book's stream: ``coin``'s spot
    touch and trades on each of ``legs`` (``(exchange, coin)``), every row on the one receive
    clock with ``age_ms``, each venue an instrument of one file, the first venue the target.

    Verifies each venue's ``book_ticker`` and ``trades`` before fetching, and writes the data
    file, the brief and ``DATA-raw.md`` with each venue's rows, cadence, collector delay,
    spread, touch and flow, and the snapshot check for each
    venue whose ``book_snapshot_25`` is in ``raw``. Nothing is thinned: the theo
    reads every quote.
    """
    from blinded import brief as briefs
    from blinded.panel import load_events

    day = Date.fromisoformat(day) if isinstance(day, str) else day
    coin = coin.upper()
    key = api_key if api_key is not None else os.environ.get("TARDIS_API_KEY", "")
    out = Path(out).expanduser()
    raw = Path(raw).expanduser() if raw else out / "raw"
    details, fetch_day = _client(details, fetch_day, key)
    pairs = [(ex, venue_symbol(coin, ex)) for ex, _ in legs]
    _verify(details, pairs, VENUE_TYPES, day, key)
    files = fetch_files([(ex, t, s) for ex, s in pairs for t in VENUE_TYPES], day, raw,
                        fetch_day, log)  # fmt: skip
    start = int(pd.Timestamp(day).value)
    end = start + DAY_NS
    read = [read_leg(raw, ex, coin, day, start, end) for ex, _ in legs]
    built = BuiltVenueFile(Path(), Path(), out / "DATA-raw.md", coin, day, read, files=files)
    built.facts = {leg.instrument: leg.facts(end) for leg in read}
    built.flow = {leg.instrument: leg.flow() for leg in read}
    within = (pl.col("ts") >= start) & (pl.col("ts") < end)
    for leg in read:
        path = raw / raw_name(
            leg.exchange, "book_snapshot_25", day, venue_symbol(coin, leg.exchange)
        )
        if path.is_file():
            snap = read_snapshot(path, 1).filter(within)
            built.snapshots[leg.instrument] = snapshot_check(
                leg.ticker, snap, f"{leg.instrument}'s book_snapshot_25"
            )
    extra = ["age_ms"]
    frame = finish([*(leg.book_rows(extra) for leg in read),
                    *(leg.trade_rows(extra) for leg in read)])  # fmt: skip
    stem = f"tardis-venues-{coin.lower()}-{'-'.join(TAGS[ex].lower() for ex, _ in legs)}-{day}"
    built.data, built.brief = out / f"{stem}.parquet", out / f"{stem}.brief.yaml"
    names = [leg.instrument for leg in read]
    fees = {leg.instrument: MEXC_TAKER_BPS.get(coin, HEDGE_FEE_BPS) if leg.exchange == "mexc"
            else HEDGE_FEE_BPS for leg in read}  # fmt: skip
    execution = {"latency_ms": 50, "taker_fee_bps": HEDGE_FEE_BPS, "maker_fee_bps": 10.0,
                 "fees": fees, "max_position": 1.0, "min_rebalance": 0.1}  # fmt: skip
    archetypes = {"instruments": names,
                  "cross_venue": [[n, names[0]] for n in names[1:]]}  # fmt: skip
    raw_brief = _venue_brief(built.data.name, day, names[0], extra, execution, archetypes)
    brief = briefs.parse(raw_brief, base=out)
    events = load_events(brief, frame)
    for leg in read:  # the file holds every row it read, in order
        if len(events[("book", leg.instrument)]) != leg.ticker.height:
            raise BuildError(f"{leg.instrument}: the file lost book rows; nothing written")
    built.rows = {**{f"{leg.instrument} book": leg.ticker.height for leg in read},
                  **{f"{leg.instrument} trades": leg.trades.height for leg in read},
                  "data": frame.height}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(built.data)
    built.brief.write_text(yaml.safe_dump(raw_brief, sort_keys=False))
    built.notes.write_text(venue_notes(built))
    log(f"{frame.height:,} rows on {len(read)} venues; wrote {built.data.name}")
    return built


def venue_table(facts: dict[str, dict]) -> list[str]:
    """The venues' quotes and prints as markdown, one column a venue."""
    names = list(facts)

    def delay(d):
        return (
            "none (no exchange stamp)"
            if d["median"] is None
            else (f"{d['median']:.1f}; {d['p99']:.1f}")
        )

    rows = [
        ("mid, time-weighted median; tick in bps",
         lambda f: f"{_num(f['mid'], 4)}; "
         + (f"{f['tick_bps']:.2f}" if f["tick_bps"] else "none")),
        ("spread, time-weighted median (p10, p90), bps",
         lambda f: f"{f['spread_bps']:.2f} ({f['spread_p10']:.1f}, {f['spread_p90']:.1f})"),
        ("touch, median USD, bid", lambda f: f"{f['touch_usd']:,.0f}"),
        ("book rows (a second); median gap ms; p99",
         lambda f: f"{f['rows']:,} ({f['rows_per_second']:.1f}); {f['gap_median_ms']:.1f}; "
                   f"{f['gap_p99_ms']:,.0f}"),
        ("share of the time with no row for 60 s", lambda f: _pct(f["silent_60s"])),
        ("collector delay, receive less exchange stamp, median; p99 ms",
         lambda f: delay(f["delay_ms"])),
        ("trades; USD", lambda f: f"{f['trades']:,}; {f['usd']:,.0f}"),
    ]  # fmt: skip
    out = [f"| | {' | '.join(names)} |", "|---|" + "---|" * len(names)]
    return out + [f"| {label} | {' | '.join(fn(facts[n]) for n in names)} |" for label, fn in rows]


def cross_lines(a: str, b: str, c: dict) -> list[str]:
    """The rows of one venue against another, as a sentence."""
    return [f"{a} against {b}, time-weighted on the merged clock: {a}'s mid sits "
            f"{c['mid_off_bps']:+.1f} bps from {b}'s (absolute median "
            f"{c['mid_off_abs_bps']:.1f}); "
            f"{a}'s ask is at or below {b}'s bid {_pct(c['crossed'])} of the time (a visible "
            f"cross); a bid {c['distance_bps']:g} bps under {b}'s bid would be {a}'s best bid "
            f"{_pct(c['bid_is_best'])} of the time."]  # fmt: skip


def flow_table(flows: dict[str, dict]) -> list[str]:
    """Who prints on each venue as markdown, one column a venue."""
    names = list(flows)
    rows = [
        ("trades; USD", lambda f: f"{f['trades']:,}; {f['usd']:,.0f}"),
        ("USD a trade, p10; p50; p90",
         lambda f: f"{f['usd_p10']:.1f}; {f['usd_p50']:.1f}; {f['usd_p90']:.1f}"),
        ("share in each 1 USD bucket from 1 to 34 USD",
         lambda f: f"{_pct(f['bucket_min'])} to {_pct(f['bucket_max'])}"),
        ("gap between trades, median s; coefficient of variation",
         lambda f: f"{f['gap_median_s']:.1f}; {f['gap_cv']:.2f}"),
        ("trades an hour, least; most", lambda f: f"{f['hour_min']:,}; {f['hour_max']:,}"),
        ("buy share", lambda f: f"{f['buy_share']:.2f}"),
        ("printed at the touch shown just before", lambda f: _pct(f["at_touch"], 0)),
        ("printed at a price never shown as the touch within 1 s",
         lambda f: _pct(f["never_shown"], 0)),
    ]  # fmt: skip
    out = [f"| | {' | '.join(names)} |", "|---|" + "---|" * len(names)]
    return out + [f"| {label} | {' | '.join(fn(flows[n]) for n in names)} |" for label, fn in rows]


def _files_table(files: dict) -> list[str]:
    return [
        "| file | bytes | sha256 |",
        "|---|---|---|",
        *(f"| {n} | {f['bytes']:,} | `{f['sha256'][:16]}` |" for n, f in files.items()),
    ]


def arb_notes(built: BuiltVenueFile, hedge_fee_bps: float) -> str:
    """``DATA-raw.md`` of the arbitrage file: the files and what was built from them in the
    head, the day's measurements in its section, which a protected run's DATA.md cuts."""
    a, b = built.legs
    other = "; ".join(f"{d}: {x:.1f} bps" for d, x in built.other_days.items())
    lines = [
        f"# The raw files behind {built.data.name}",
        "",
        f"Built by `blinded demo arb` from the Tardis free day {built.day}: {built.coin} spot on "
        f"`{a.exchange}` (instrument {a.instrument}, the quote venue) and `{b.exchange}` "
        f"({b.instrument}, the hedge venue), `book_ticker`, `trades` and `book_snapshot_25` of "
        "each, every stamp Tardis's `local_timestamp`, the recorder's receive clock. Tardis "
        "files have no checksum; their sizes and SHA-256 are these:",
        "",
        *_files_table(built.files),
        "",
        "The data file keeps every book row of both venues, with `age_ms`, how long after the "
        "exchange's stamp the recorder received the row (0 where the venue sends none), the "
        "hedge venue's levels 1 to 4 from its snapshot as-of each of its rows, and every trade "
        "with its own venue's touch as received. Nothing is thinned: the quote venue changes "
        "its touch about once a second, and the hedge leg reads the hedge venue's touch at "
        "fill stamps that are on no lattice. The touch file "
        f"`{built.touch.name}` holds every two-sided row of {a.instrument} in ticks of "
        f"{a.tick:g}; the depth file `{built.depth.name}` every snapshot row of "
        f"{b.instrument}, {DEPTH_LEVELS} levels a side in price units. Only the sims read the "
        f"two. Ticks: {a.instrument} {a.tick:g}, {b.instrument} {b.tick:g}; lot "
        f"{a.lot if a.lot else 'none'} on the quote venue. Fees in the brief: maker 0 and "
        f"taker {MEXC_TAKER_BPS.get(built.coin, hedge_fee_bps):g} bps on the quote venue (its "
        f"exchangeInfo), the hedge {hedge_fee_bps:g} bps (Binance's regular spot tier); the "
        "stock block's `withdrawal_fee_coin` and `withdrawal_fee_usdt` are unset: look both "
        "up on both venues' withdrawal pages before an arbitrage strategy is scored.",
        "",
        f"The quote venue's spread on the other free days in the raw directory, time-weighted "
        f"median: {other or 'none there'}. A wide day is a month with no maker on the quote "
        "venue; the build refuses a day under three hedge fees.",
        "",
        f"## {built.day.isoformat()}",
        "",
        "| rows | count |", "|---|---|", *(f"| {k} | {v:,} |" for k, v in built.rows.items()),
        "",
        f"The build's checks: the touch file equals the data file's book of {a.instrument} at "
        f"all {built.touch_check['stamps_checked']:,} of its stamps; each venue's snapshot is "
        "its touch stream's book:",
        "",
    ]  # fmt: skip
    for inst, s in built.snapshots.items():
        lines += [*snapshot_lines(s, inst), ""]
    lines += ["The two venues over the day:", "",
              *venue_table(built.facts), "", *cross_lines(a.instrument, b.instrument, built.cross),
              "", "Who prints on each:", "",
              *flow_table(built.flow), ""]  # fmt: skip
    return "\n".join(lines) + "\n"


def venue_notes(built: BuiltVenueFile) -> str:
    """``DATA-raw.md`` of the consolidated book's file."""
    names = ", ".join(f"`{leg.exchange}` ({leg.instrument})" for leg in built.legs)
    lines = [
        f"# The raw files behind {built.data.name}",
        "",
        f"Built by `blinded demo venues` from the Tardis free day {built.day}: {built.coin} spot "
        f"on {names}, `book_ticker` and `trades` of each, every row on Tardis's "
        "`local_timestamp`, the one receive clock, so cross-venue order is as one observer saw "
        "it. Every book row is kept with `age_ms`, how long after the exchange's stamp the "
        "recorder received it (0 where the venue sends none, as Binance spot does not); every "
        "trade with its own venue's touch as received. Tardis files have no checksum; their "
        "sizes and SHA-256 are these:",
        "",
        *_files_table(built.files),
        "",
        f"## {built.day.isoformat()}",
        "",
        "| rows | count |", "|---|---|", *(f"| {k} | {v:,} |" for k, v in built.rows.items()),
        "",
        "The venues over the day:",
        "",
        *venue_table(built.facts),
        "",
        "Who prints on each:",
        "",
        *flow_table(built.flow),
        "",
    ]  # fmt: skip
    if built.snapshots:
        lines += ["Each venue's `book_snapshot_25` against its touch stream, where "
                  "the raw directory has it:", ""]  # fmt: skip
        for inst, s in built.snapshots.items():
            lines += [*snapshot_lines(s, inst), ""]
    return "\n".join(lines) + "\n"
