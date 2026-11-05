"""Event streams joined backward as-of onto a decision grid, as one wide polars frame."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
import polars as pl

from blinded import brief as briefs
from blinded.brief import Brief

_DERIVED = {
    "mid": ("bid", "ask"),
    "microprice": ("bid", "ask", "bid_qty", "ask_qty"),
    "signed_qty": ("side", "qty"),
    "buy_qty": ("side", "qty"),
    "sell_qty": ("side", "qty"),
    "notional": ("price", "qty"),
}
_FILTER = re.compile(r"^\s*(\w+)\s*(==|!=)\s*(.+?)\s*$")


class Events:
    """The events of one stream for one instrument, ordered by ``available_at``.

    Ties keep file order, so an as-of lookup at a duplicated timestamp sees the
    last row, as a feed handler would. ``delay_ns`` is the stream's
    availability delay, ``available_at - ts``; it is worked out from the rows
    when not given.
    """

    def __init__(self, frame: pl.DataFrame, delay_ns: int | None = None):
        self.frame = frame
        self.at = frame["available_at"].dt.epoch("ns").to_numpy()
        self._columns: dict[str, np.ndarray] = {}
        self._delay = delay_ns

    def __len__(self) -> int:
        return len(self.at)

    @property
    def delay_ns(self) -> int:
        """How long after its stamp an event becomes usable, the same for every event.

        An order sent at :math:`g` reaches the book as it stood at stamp
        :math:`g + L`, which is the book as-of ``available_at`` :math:`g + L +
        d`; the sim and the targets read fills, marks and returns there.
        """
        if self._delay is None:
            lag = self.at - self.frame["ts"].dt.epoch("ns").to_numpy()
            if len(lag) and lag.min() != lag.max():
                raise ValueError("available_at must be ts plus one delay for the whole stream")
            self._delay = int(lag[0]) if len(lag) else 0
        return self._delay

    @property
    def fields(self) -> set[str]:
        names = set(self.frame.columns) - {"ts", "available_at", "n"}
        return names | {d for d, needs in _DERIVED.items() if set(needs) <= names}

    def column(self, name: str) -> np.ndarray:
        """Per-event values of a field, derived fields included, NaN where missing."""
        if name not in self._columns:
            self._columns[name] = self._compute(name)
        return self._columns[name]

    def _compute(self, name: str) -> np.ndarray:
        if name == "mid":
            return (self.column("bid") + self.column("ask")) / 2
        if name == "microprice":
            bid, ask = self.column("bid"), self.column("ask")
            bq, aq = self.column("bid_qty"), self.column("ask_qty")
            return (bid * aq + ask * bq) / (bq + aq)
        if name == "signed_qty":
            return self.column("side") * self.column("qty")
        if name == "buy_qty":
            return np.where(self.column("side") > 0, self.column("qty"), 0.0)
        if name == "sell_qty":
            return np.where(self.column("side") < 0, self.column("qty"), 0.0)
        if name == "notional":
            return self.column("price") * self.column("qty")
        if name == "count":
            return np.ones(len(self))
        return self.frame[name].cast(pl.Float64).fill_null(np.nan).to_numpy()

    def head(self, t_ns: int) -> Events:
        """Events available at or before ``t_ns``."""
        n = int(np.searchsorted(self.at, t_ns, side="right"))
        return Events(self.frame.head(n), self._delay)

    def asof(self, name: str, times_ns: np.ndarray) -> np.ndarray:
        """Value of the last event with ``available_at <= t`` for each time, NaN before any."""
        idx = np.searchsorted(self.at, times_ns, side="right") - 1
        values = self.column(name)
        out = np.full(len(times_ns), np.nan)
        known = idx >= 0
        out[known] = values[idx[known]]
        return out


class Panel:
    """What a feature receives: a wide frame on the decision grid, plus the raw events.

    ``panel.frame`` has a ``ts`` column, the grid, and one column per
    instrument and field, named ``"SYM.field"``. Every value at grid time
    :math:`g` comes from events with ``available_at <= g``:

    - each stream field as-of :math:`g`: ``bid``, ``ask``, ``bid_qty``,
      ``ask_qty`` from the book and ``price``, ``qty``, ``side`` (+1 buy,
      -1 sell) of the last trade, plus ``mid`` and ``microprice``;
    - ``rate`` of a perpetual with a ``funding`` stream: the last funding rate
      settled, one row per settlement at its stamp;
    - ``borrow_rate`` of a spot instrument with a ``borrow`` stream: what
      borrowing the coin costs a day, as a fraction, one row per published
      rate at its stamp;
    - cumulative sums over the events so far: ``cum_book``, ``cum_trades``,
      ``cum_funding`` and ``cum_borrow`` count events, ``cum_qty``,
      ``cum_buy_qty``, ``cum_sell_qty``, ``cum_signed_qty`` and ``cum_notional``
      add them up. The sum over
      :math:`(g - w, g]` is the value at :math:`g` minus the value at
      :math:`g - w`. A stream with an ``n`` field, the number of raw events
      each row stands for in a thinned file, counts with ``n`` instead.

    Missing values (a grid point before a stream's first event) are null.

    Parameters
    ----------
    frame
        The wide frame, ``ts`` first.
    events
        Mapping from ``(stream, instrument)`` to :class:`Events`.
    step_ns
        The grid step in nanoseconds.
    """

    def __init__(self, frame: pl.DataFrame, events: dict[tuple[str, str], Events], step_ns: int,
                 legs: tuple[str, ...] | None = None):  # fmt: skip
        self.frame = frame
        self._events = events
        self._legs = tuple(legs) if legs else None
        self.step_ns = int(step_ns)
        self.grid_ns = frame["ts"].dt.epoch("ns").to_numpy()
        self.tz = frame.schema["ts"].time_zone
        # the fill engine's stream of a file that declares one (data.touch): where its rows
        # are, never read by a feature (:class:`blinded.sim_maker.TouchSource`); and an
        # arbitrage file's hedge depth, read only by the hedge leg (``DepthSource``)
        self.touch_source = None
        self.depth_source = None

    @classmethod
    def build(
        cls, events: dict[tuple[str, str], Events], grid_ns: np.ndarray, tz: str, step_ns: int
    ) -> Panel:
        """Join every stream backward as-of onto the grid."""
        ts = pl.DataFrame({"ts": _datetimes(np.asarray(grid_ns, dtype=np.int64), tz)})
        blocks = [ts]
        seen: set[str] = set()
        for (stream, sym), ev in sorted(events.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            block = _block(stream, sym, ev, ts)
            clash = seen & set(block.columns)
            if clash:
                raise ValueError(f"{sorted(clash)} come from more than one stream")
            seen |= set(block.columns)
            blocks.append(block)
        return cls(pl.concat(blocks, how="horizontal"), events, step_ns)

    @classmethod
    def from_brief(
        cls, brief: Brief | str | os.PathLike, frame: pl.DataFrame | None = None, theo=None
    ) -> Panel:
        """Load the brief's data file and build the full panel.

        ``brief`` is a parsed :class:`~blinded.brief.Brief` or a path to
        ``brief.yaml``. ``frame``, if given, is used instead of reading the
        file: the raw rows, in the file's layout. A brief with ``data.theo``
        adds the theo's stored output as the instrument ``THEO.<coin>``
        (:mod:`blinded.theo`), from its ``path``, or from ``theo``, the files
        of a run's segments in order.
        """
        if not isinstance(brief, Brief):
            brief = briefs.load(brief)
        events = load_events(brief, frame)
        spec = brief.data.theo
        if spec is not None and (theo is not None or (frame is None and spec.path)):
            from blinded.theo import events_frame

            # available when the last of its legs' book rows is: the filter's output at an
            # event cannot be read before the row it was computed from
            paths = theo if theo is not None else (spec.path,)
            delay = max(brief.data.delay("book", leg).value for leg in spec.legs)
            frame_ = events_frame(paths, delay)
            events[("theo", spec.instrument)] = Events(frame_, delay)
        panel = cls.build(events, make_grid(brief, events), brief.data.tz, brief.grid.step.value)
        basket = brief.family("basket")
        panel._legs = basket.instruments if basket is not None else None
        touch = brief.data.touch
        if frame is None and touch is not None:
            from blinded.sim_maker import TouchSource

            panel.touch_source = TouchSource((touch.path,), touch.instrument, touch.tick_size)
        depth = brief.data.depth
        if frame is None and depth is not None:
            from blinded.sim_maker import DepthSource

            panel.depth_source = DepthSource((depth.path,), depth.venue, depth.levels)
        return panel

    @classmethod
    def from_frames(
        cls,
        grid,
        book: Mapping[str, pl.DataFrame] | None = None,
        trades: Mapping[str, pl.DataFrame] | None = None,
        step=None,
        funding: Mapping[str, pl.DataFrame] | None = None,
        borrow: Mapping[str, pl.DataFrame] | None = None,
    ) -> Panel:
        """Build a panel from polars event frames.

        Parameters
        ----------
        grid
            Decision times: a polars Datetime series, a ``DatetimeIndex`` or a
            list of datetimes, evenly spaced. Naive times are read as UTC.
        book
            ``{instrument: frame}`` with ``ts``, optionally ``available_at``
            (defaults to ``ts``), and ``bid``, ``ask``, ``bid_qty``, ``ask_qty``.
        trades
            ``{instrument: frame}`` with ``ts``, optionally ``available_at``,
            and ``price``, ``qty``, ``side`` (+1 buy, -1 sell).
        step
            The grid step; inferred from the grid when it has two or more points.
        funding
            ``{perpetual: frame}`` with ``ts``, the settlement stamp, optionally
            ``available_at``, and ``rate``, the rate settled there. The panel
            then has ``SYM.rate``, the last settled rate as-of each grid point,
            and the sim accrues it (:func:`blinded.sim.simulate`).
        borrow
            ``{spot: frame}`` with ``ts``, optionally ``available_at``, and
            ``borrow_rate``, the daily rate of borrowing the coin published
            there. A short position pays the rate as-of each step
            (:func:`blinded.sim.simulate`).
        """
        grid_ns, tz = _grid_ns(grid)
        events = {}
        streams = (("book", book or {}), ("trades", trades or {}), ("funding", funding or {}),
                   ("borrow", borrow or {}))  # fmt: skip
        for stream, frames in streams:
            for sym, frame in frames.items():
                events[(stream, sym)] = Events(_event_frame(frame, tz))
        return cls.build(events, grid_ns, tz, _step_ns(grid_ns, step))

    @classmethod
    def from_pandas(
        cls,
        grid: pd.DatetimeIndex,
        book: Mapping[str, pd.DataFrame] | None = None,
        trades: Mapping[str, pd.DataFrame] | None = None,
    ) -> Panel:
        """Build a panel from pandas frames indexed by availability time.

        Parameters
        ----------
        grid
            Decision times, an evenly spaced ``DatetimeIndex``.
        book
            ``{instrument: frame}`` with columns ``bid``, ``ask``, ``bid_qty``,
            ``ask_qty``.
        trades
            ``{instrument: frame}`` with columns ``price``, ``qty`` and
            ``side`` (+1 buy, -1 sell).
        """
        tz = str(grid.tz) if grid.tz is not None else "UTC"
        frames = {
            name: {sym: _from_pandas(frame, tz) for sym, frame in (group or {}).items()}
            for name, group in (("book", book), ("trades", trades))
        }
        step = grid.freq if grid.freq is not None else None
        return cls.from_frames(grid, frames["book"], frames["trades"], step=step)

    @property
    def grid(self) -> pl.Series:
        """The decision times, the frame's ``ts`` column."""
        return self.frame["ts"]

    @property
    def step(self) -> pd.Timedelta:
        return pd.Timedelta(self.step_ns, unit="ns")

    def __len__(self) -> int:
        return len(self.grid_ns)

    def time(self, i: int) -> pd.Timestamp:
        """The ``i``-th grid point as a timestamp; negative ``i`` counts from the end."""
        return pd.Timestamp(int(self.grid_ns[i]), tz="UTC").tz_convert(self.tz)

    @property
    def symbols(self) -> list[str]:
        return sorted({sym for _, sym in self._events})

    @property
    def legs(self) -> tuple[str, ...]:
        """The legs of a basket: the instruments of the brief's basket family, in the brief's
        order, so a basket feature can write ``F.each(panel.legs, ...)``; every instrument,
        sorted, in a panel without one."""
        return self._legs or tuple(self.symbols)

    @property
    def streams(self) -> list[tuple[str, str]]:
        return sorted(self._events)

    def fields(self, sym: str) -> list[str]:
        """The fields of ``sym`` in the frame."""
        prefix = f"{sym}."
        return [c[len(prefix) :] for c in self.frame.columns if c.startswith(prefix)]

    def col(self, sym: str, field: str) -> pl.Expr:
        """The frame column ``"SYM.field"`` as an expression, checked to exist.

        Examples
        --------
        >>> from blinded.features import example_panel
        >>> p = example_panel()
        >>> p.eval(p.col("SPOT", "mid")).to_list()
        [10.5, 11.0, 11.5, 12.5]
        """
        name = f"{sym}.{field}"
        if name not in self.frame.columns:
            raise KeyError(f"{sym!r} has no field {field!r}; available: {self.fields(sym)}")
        return pl.col(name)

    def eval(self, expr: pl.Expr) -> pl.Series:
        """Evaluate an expression on the frame, lazily, as the judge does."""
        return self.frame.lazy().select(expr).collect().to_series()

    def to_pandas(self, *columns: str | list[str]) -> pd.DataFrame:
        """The frame, or only ``columns`` of it, as a pandas DataFrame indexed by ``ts``.

        Nulls become NaN. Naming the columns you need converts only those; they
        can be given one by one or as a list.

        Examples
        --------
        >>> from blinded.features import example_panel
        >>> example_panel().to_pandas("SPOT.mid", "PERP.mid").iloc[0].tolist()
        [10.5, 11.0]
        >>> example_panel().to_pandas(["SPOT.mid", "PERP.mid"]).columns.tolist()
        ['SPOT.mid', 'PERP.mid']
        """
        names = [c for arg in columns for c in ([arg] if isinstance(arg, str) else arg)]
        frame = self.frame.select("ts", *names) if names else self.frame
        return frame.to_pandas().set_index("ts")

    def events(self, stream: str, sym: str) -> Events:
        try:
            return self._events[(stream, sym)]
        except KeyError:
            raise KeyError(f"no {stream!r} stream for {sym!r}; have {self.streams}") from None

    def book(self, sym: str) -> pl.DataFrame:
        """Book events for one instrument, with ``available_at``."""
        return self.events("book", sym).frame

    def trades(self, sym: str) -> pl.DataFrame:
        """Trade events for one instrument, with ``available_at``."""
        return self.events("trades", sym).frame

    def cut(self, t: pd.Timestamp | str) -> Panel:
        """A panel holding only events with ``available_at <= t`` and grid points ``<= t``.

        The frame is a row slice of this one. That equals a rebuild from the
        kept events because every value at :math:`g \\le t` is computed from
        events available by :math:`g`; ``tests/test_panel.py`` checks it.
        """
        t_ns = _to_ns(t, self.tz)
        events = {key: ev.head(t_ns) for key, ev in self._events.items()}
        n = int(np.searchsorted(self.grid_ns, t_ns, side="right"))
        out = Panel(self.frame.slice(0, n), events, self.step_ns, self._legs)
        out.touch_source = self.touch_source
        out.depth_source = self.depth_source
        return out

    def with_columns(self, columns: Mapping[str, np.ndarray]) -> Panel:
        """The same panel with extra float columns on the grid, nulls where NaN."""
        extra = [
            pl.Series(name, values, dtype=pl.Float64, nan_to_null=True)
            for name, values in columns.items()
        ]
        out = Panel(self.frame.with_columns(extra), self._events, self.step_ns, self._legs)
        out.touch_source = self.touch_source
        out.depth_source = self.depth_source
        return out

    def stream_of(self, sym: str, field: str) -> Events:
        """The event table of ``sym`` that carries ``field``."""
        mine = [ev for (_, s), ev in sorted(self._events.items()) if s == sym]
        found = [ev for ev in mine if field in ev.fields]
        if not found:
            have = sorted({f for ev in mine for f in ev.fields})
            raise KeyError(f"{sym!r} has no field {field!r}; available: {have}")
        if len(found) > 1:
            raise KeyError(f"field {field!r} of {sym!r} is in more than one stream")
        return found[0]

    def __repr__(self) -> str:
        span = f"{self.time(0)} .. {self.time(-1)}" if len(self) else "empty"
        return f"Panel({len(self)} grid points, {span}, symbols={self.symbols})"


def _block(stream: str, sym: str, ev: Events, ts: pl.DataFrame) -> pl.DataFrame:
    """The columns one stream of one instrument contributes to the frame."""
    fields = [c for c in ev.frame.columns if c not in ("ts", "available_at", "n")]
    values = [pl.col(f).cast(pl.Float64).fill_nan(None).alias(f"{sym}.{f}") for f in fields]
    count = pl.int_range(1, pl.len() + 1, dtype=pl.Int64)
    if "n" in ev.frame.columns:
        count = pl.col("n").cast(pl.Int64).fill_null(1).cum_sum()
    sums = {f"{sym}.cum_{stream}": count}
    sums |= {f"{sym}.{name}": expr.cum_sum() for name, expr in _summable(fields).items()}
    right = ev.frame.select("available_at", *values, **sums)
    joined = ts.join_asof(right, left_on="ts", right_on="available_at", strategy="backward")
    out = joined.drop("ts", "available_at").with_columns(pl.col(*sums).fill_null(0))
    bid, ask, bq, aq = (pl.col(f"{sym}.{f}") for f in ("bid", "ask", "bid_qty", "ask_qty"))
    if {"bid", "ask"} <= set(fields):
        out = out.with_columns(((bid + ask) / 2).alias(f"{sym}.mid"))
    if {"bid", "ask", "bid_qty", "ask_qty"} <= set(fields):
        out = out.with_columns(((bid * aq + ask * bq) / (bq + aq)).alias(f"{sym}.microprice"))
    return out


def _summable(fields: list[str]) -> dict[str, pl.Expr]:
    """Per-event quantities that add up over a window; missing values add nothing."""
    have = set(fields)
    qty = pl.col("qty").cast(pl.Float64).fill_nan(None)
    side = pl.col("side").cast(pl.Float64).fill_nan(None)
    price = pl.col("price").cast(pl.Float64).fill_nan(None)
    out = {}
    if "qty" in have:
        out["cum_qty"] = qty.fill_null(0.0)
    if {"qty", "side"} <= have:
        out["cum_signed_qty"] = (side * qty).fill_null(0.0)
        out["cum_buy_qty"] = pl.when(side > 0).then(qty).otherwise(0.0).fill_null(0.0)
        out["cum_sell_qty"] = pl.when(side < 0).then(qty).otherwise(0.0).fill_null(0.0)
    if {"qty", "price"} <= have:
        out["cum_notional"] = (price * qty).fill_null(0.0)
    return out


def load_events(brief: Brief, frame: pl.DataFrame | None = None) -> dict[tuple[str, str], Events]:
    """Read the data file (or take ``frame``) and split it per stream and instrument."""
    spec = brief.data
    frame = pl.read_parquet(spec.path) if frame is None else frame
    frame = frame.with_columns(event_times(frame, brief).alias("_ts"))
    out = {}
    for stream in spec.streams.values():
        columns = [_field_expr(stream, name, src) for name, src in stream.fields.items()]
        test = _filter_expr(stream.filter) if stream.filter else None
        # only the columns this stream reads travel through the filter and the split: a file
        # whose trade rows carry the book's columns as nulls would copy them all otherwise
        need = {"_ts", spec.instrument_column, *(r for c in columns for r in c.meta.root_names())}
        need |= set(test.meta.root_names()) if test is not None else set()
        rows = frame.select(c for c in frame.columns if c in need)
        rows = rows.filter(test) if test is not None else rows
        parts = rows.partition_by(spec.instrument_column, as_dict=True, maintain_order=True)
        for (sym,), part in parts.items():
            delay = spec.delay(stream.name, sym)
            table = part.select(
                pl.col("_ts").alias("ts"),
                (pl.col("_ts") + pl.duration(nanoseconds=delay.value)).alias("available_at"),
                *columns,
            ).sort("available_at", maintain_order=True)
            out[(stream.name, sym)] = Events(table, delay.value)
    return out


def make_grid(brief: Brief, events: dict[tuple[str, str], Events]) -> np.ndarray:
    """Regular grid at ``brief.grid.step`` covering the events, as integer nanoseconds.

    It starts at ``grid.start`` if given, else at the first event rounded up to
    a step, and ends at the last event rounded down.
    """
    step = brief.grid.step.value
    first = min(int(ev.at[0]) for ev in events.values() if len(ev))
    last = max(int(ev.at[-1]) for ev in events.values() if len(ev))
    if brief.grid.start is not None:
        start = _to_ns(brief.grid.start, brief.data.tz)
    else:
        start = -(-first // step) * step
    end = last // step * step
    # whole steps in integers: np.arange sizes its output in floating point, and on an hourly
    # grid in nanoseconds the end would fall off it
    return start + step * np.arange((end - start) // step + 1, dtype=np.int64)


def _datetimes(ns: np.ndarray, tz: str) -> pl.Series:
    utc = pl.Series("ts", ns, dtype=pl.Int64).cast(pl.Datetime("ns"))
    return utc.dt.replace_time_zone("UTC").dt.convert_time_zone(tz).set_sorted()


def _grid_ns(grid) -> tuple[np.ndarray, str]:
    if isinstance(grid, pd.DatetimeIndex):
        tz = str(grid.tz) if grid.tz is not None else "UTC"
        utc = grid.tz_localize("UTC") if grid.tz is None else grid
        return utc.as_unit("ns").asi8, tz
    series = pl.Series(grid) if not isinstance(grid, pl.Series) else grid
    if not isinstance(series.dtype, pl.Datetime):
        raise TypeError(f"the grid must hold datetimes, got {series.dtype}")
    tz = series.dtype.time_zone or "UTC"
    return series.dt.cast_time_unit("ns").dt.epoch("ns").to_numpy(), tz


def _step_ns(grid_ns: np.ndarray, step) -> int:
    if step is not None:
        return pd.Timedelta(step).value
    gaps = np.unique(np.diff(grid_ns))
    if len(gaps) != 1:
        raise ValueError("the grid must be evenly spaced with at least two points, or pass step")
    return int(gaps[0])


def _event_frame(frame: pl.DataFrame, tz: str) -> pl.DataFrame:
    if "available_at" not in frame.columns:
        frame = frame.with_columns(pl.col("ts").alias("available_at"))
    times = {}
    for name in ("ts", "available_at"):
        col = frame[name].dt.cast_time_unit("ns")
        if col.dtype.time_zone is None:
            col = col.dt.replace_time_zone("UTC")
        times[name] = col.dt.convert_time_zone(tz)
    fields = [c for c in frame.columns if c not in times]
    out = frame.select(*fields).with_columns(**times).select("ts", "available_at", *fields)
    return out.sort("available_at", maintain_order=True)


def _from_pandas(frame: pd.DataFrame, tz: str) -> pl.DataFrame:
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize(tz)
    order = np.argsort(index.asi8, kind="stable")
    at = _datetimes(index.as_unit("ns").asi8[order], tz)
    data = {"ts": at, "available_at": at}
    data |= {col: frame[col].to_numpy(dtype=float)[order] for col in frame.columns}
    return pl.DataFrame(data)


def event_times(frame: pl.DataFrame, brief: Brief) -> pl.Series:
    """The time column of raw rows as nanosecond datetimes in the brief's timezone."""
    return _timestamps(frame[brief.data.time_column], brief.data)


def _timestamps(column: pl.Series, spec) -> pl.Series:
    if column.dtype.is_integer():
        column = pl.from_epoch(column, time_unit=spec.time_unit)
    if column.dtype.time_zone is None:
        column = column.dt.replace_time_zone(spec.tz)
    return column.dt.convert_time_zone(spec.tz).dt.cast_time_unit("ns")


def _filter_expr(text: str) -> pl.Expr:
    match = _FILTER.match(text)
    if match is None:
        raise ValueError(f"stream filter must look like 'column == value', got {text!r}")
    column, op, literal = match.groups()
    value = _literal(literal)
    col = pl.col(column)
    return col == value if op == "==" else col != value


def _literal(text: str):
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if text[0] in "'\"" and text[-1] == text[0]:
        return text[1:-1]
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text


def _field_expr(stream, name: str, source: str) -> pl.Expr:
    if name != "side":
        return pl.col(source).alias(name)
    buy, sell = stream.side_values.get("buy", "B"), stream.side_values.get("sell", "S")
    side = pl.col(source).cast(pl.String)
    return pl.when(side == str(buy)).then(1.0).when(side == str(sell)).then(-1.0).alias("side")


def _to_ns(t, tz) -> int:
    ts = pd.Timestamp(t)
    ts = ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
    return ts.value
