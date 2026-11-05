"""Load and validate a dataset brief (``brief.yaml``)."""

from __future__ import annotations

import functools
import itertools
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd
import yaml

from blinded import durations
from blinded.ladder import RULES, check_rules

_ENV_VAR = re.compile(r"\$\{(\w+)\}")
FAMILIES = ("directional", "lead_lag", "pairs", "basket")
PAIRS_DEFAULTS = {"window": "2h", "every": "30m", "z_in": 2.0, "z_out": 0.5}


@dataclass(frozen=True)
class Stream:
    """One event stream: a row filter plus a mapping from canonical to source columns."""

    name: str
    filter: str | None
    fields: dict[str, str]
    side_values: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Thinned:
    """The stamps a thinned data file keeps.

    For every grid point :math:`g` and every offset :math:`o`, the file holds
    the last book row stamped at or before :math:`g + o`, so a read at any of
    those stamps sees exactly what the raw file would show. A read at any
    other stamp sees the kept row before it. ``blinded demo binance`` writes
    this block into the brief of the file it builds.
    """

    step: pd.Timedelta
    offsets: tuple[pd.Timedelta, ...]

    def keeps(self, offset: pd.Timedelta) -> bool:
        """Whether a read at stamp :math:`g + \text{offset}` is exact for every grid point.

        Examples
        --------
        >>> t = Thinned(pd.Timedelta("500ms"), tuple(pd.to_timedelta([-50, 0, 50], unit="ms")))
        >>> [t.keeps(pd.Timedelta(ms, unit="ms")) for ms in (50, 450, 550, 100, -500)]
        [True, True, True, False, True]
        """
        step = self.step.value
        return any((offset.value - o.value) % step == 0 for o in self.offsets)


@dataclass(frozen=True)
class CostModel:
    """How a bar file's bid and ask were made, since bars have none.

    ``kind`` is ``half_spread``: each book row is the bar's close plus and
    minus ``half_spread_bps[instrument]``, measured on the top-of-book day
    named in ``source``, and a taker fills at the next bar's close at that
    spread (``fill``). ``day`` is the date in ``source``; it must lie before
    train. ``blinded demo trend`` writes this block into the brief it builds,
    and ``blinded init`` checks it against the file.
    """

    kind: str
    fill: str
    source: str
    half_spread_bps: dict[str, float] = field(default_factory=dict)
    day: pd.Timestamp | None = None

    def line(self) -> str:
        """The sentence every page that shows a number from the file carries."""
        spreads = list(self.half_spread_bps.values())
        if spreads and min(spreads) != max(spreads):
            hs = f"{min(spreads):g} to {max(spreads):g} bps per symbol"
        elif spreads:
            hs = f"{spreads[0]:g} bps per symbol"
        else:
            hs = "a measured number of bps per symbol"
        return (f"fills are a model: the next bar's close plus or minus a half spread of {hs}, "
                f"measured on {self.source}, before train; no book existed at the "
                "fill")  # fmt: skip


@dataclass(frozen=True)
class TouchFile:
    """The touch stream of a market-making file, which only the fill engine reads.

    ``path`` holds every top-of-book row of the instrument with its sizes,
    ``ts``, ``bid``, ``ask`` (in ticks of ``tick_size``), ``bid_qty`` and
    ``ask_qty``; a run directory splits it beside the data file. A Tardis file
    that keeps every book row and trade needs none: the engine reads the data
    file's own rows (:func:`blinded.sim_maker.touch_of`).
    """

    path: Path
    tick_size: float
    instrument: str | None = None


@dataclass(frozen=True)
class DepthFile:
    """The order book's levels on one instrument at every snapshot, which only the hedge leg
    of an arbitrage strategy and the data notes read.

    ``path`` holds Tardis ``book_snapshot_25`` rows as ``ts``, ``bid0_px`` ..
    ``bid{levels-1}_px``, ``bid0_qty`` .., ``ask0_px`` .., ``ask0_qty`` .., in
    price units; ``venue`` is the instrument they belong to; ``lag_ms`` is how
    far the snapshot's level 0 trails the touch stream, as measured. A run
    directory splits it beside the data file, as it splits the touch file.
    """

    path: Path
    venue: str
    levels: int = 25
    lag_ms: float = 0.0


@dataclass(frozen=True)
class Data:
    path: Path
    time_column: str
    time_unit: str
    tz: str
    clock: str
    instrument_column: str
    streams: dict[str, Stream]
    default_delay_ms: float
    delay_overrides: tuple[dict, ...]
    thinned: Thinned | None = None
    cost_model: CostModel | None = None
    touch: TouchFile | None = None
    depth: DepthFile | None = None
    theo: object = None  # blinded.theo.TheoSpec

    @property
    def bars(self) -> bool:
        """True for a bar file: one with a declared fill model (:class:`CostModel`)."""
        return self.cost_model is not None

    def delay(self, stream: str, instrument: str) -> pd.Timedelta:
        """Availability delay for one stream of one instrument."""
        ms = self.default_delay_ms
        for rule in self.delay_overrides:
            same_stream = rule.get("stream", stream) == stream
            same_instrument = rule.get("instrument", instrument) == instrument
            if same_stream and same_instrument:
                ms = float(rule["delay_ms"])
        return pd.Timedelta(milliseconds=ms)

    def exact(self, offset: pd.Timedelta) -> bool:
        """Whether this file answers a read at stamp :math:`g + \text{offset}` exactly.

        Always, unless the file is thinned (:class:`Thinned`).
        """
        return self.thinned is None or self.thinned.keeps(offset)


@dataclass(frozen=True)
class Target:
    instrument: str
    price: str
    horizons: tuple[pd.Timedelta, ...]
    primary_horizon: pd.Timedelta


BARS = pd.Timedelta("1min")


@dataclass(frozen=True)
class Grid:
    step: pd.Timedelta
    start: pd.Timestamp | None
    warmup: pd.Timedelta

    @property
    def bars(self) -> bool:
        """True for a grid of a minute or more, on which a horizon or a lag shorter than one
        step reads the same bar at both ends."""
        return self.step >= BARS


@dataclass(frozen=True)
class Execution:
    """How positions become fills: latency, fees, position limit and the rebalance threshold.

    ``fees`` holds per-instrument taker fees that override ``taker_fee_bps``.
    The fill engine of a market-making strategy (:mod:`blinded.sim_maker`) also
    reads ``tick_size`` and ``lot_size``, the instrument's price and quantity
    steps (None: the tick is read from the training book at ``blinded init``,
    and sizes are not rounded), and ``cancel_latency`` and
    ``response_latency``, how long a cancel takes to reach the book and a fill
    to come back; each is None for the order latency, :attr:`latency`.
    ``deadline`` (``deadline_ms``, 2 s by default) is how long the maker-taker
    row of a position strategy rests an order before it crosses
    (:func:`blinded.sim_maker.simulate_maker_taker`). ``fee_tier``
    (``venue/product/tier``) names the row of the package's fee table
    (:mod:`blinded.fees`) whose maker and taker fees the brief took.
    ``hedge_latency`` (``hedge_latency_ms``) and ``hedge_fee_bps`` are an
    arbitrage strategy's hedge leg's: how long its taker order takes to reach
    the hedge venue once a fill is known, and its taker fee there
    (:func:`blinded.sim_maker.simulate_arb`); None where the brief has no hedge.
    """

    latency: pd.Timedelta
    taker_fee_bps: float
    maker_fee_bps: float
    max_position: float
    min_rebalance: float
    fees: dict[str, float] = field(default_factory=dict)
    tick_size: float | None = None
    lot_size: float | None = None
    cancel_latency: pd.Timedelta | None = None
    response_latency: pd.Timedelta | None = None
    deadline: pd.Timedelta = pd.Timedelta("2s")
    fee_tier: str | None = None
    hedge_latency: pd.Timedelta | None = None
    hedge_fee_bps: float | None = None

    def fee(self, instrument: str) -> float:
        """The taker fee of one instrument, in bps."""
        return self.fees.get(instrument, self.taker_fee_bps)

    @property
    def cancel(self) -> pd.Timedelta:
        """The cancel latency in force: ``cancel_latency``, else the order latency."""
        return self.latency if self.cancel_latency is None else self.cancel_latency

    @property
    def response(self) -> pd.Timedelta:
        """The response latency in force: ``response_latency``, else the order latency."""
        return self.latency if self.response_latency is None else self.response_latency


@dataclass(frozen=True)
class Family:
    """One kind of strategy a run studies, and its share of each phase's sessions.

    ``instruments`` belong to the directional family and to a basket, whose
    legs they are in order, ``leaders`` and ``followers`` to lead-lag, and
    ``pairs`` to the pairs family, where ``(A, B)`` means the spread of A over
    B. ``defaults`` are the pairs
    parameters the report uses when no pairs strategy was submitted.
    """

    name: str
    share: float
    instruments: tuple[str, ...] = ()
    leaders: tuple[str, ...] = ()
    followers: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()
    defaults: dict = field(default_factory=dict)

    @property
    def targets(self) -> tuple[str, ...]:
        """What a single-leg feature may predict and a single-leg strategy trade."""
        return self.followers if self.name == "lead_lag" else self.instruments

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every instrument the family names, in order, once each."""
        legs = [*self.instruments, *self.leaders, *self.followers]
        return tuple(dict.fromkeys(legs + [s for pair in self.pairs for s in pair]))


@dataclass(frozen=True)
class Archetypes:
    """The brief's ``archetypes`` block: the run studies strategy archetypes grouped by edge.

    ``instruments`` are the run's (``blinded init`` writes every instrument
    in the file when the block leaves them out); ``only`` limits the run to
    some archetypes; ``shares`` are per edge; ``cross_venue`` lists pairs
    that are one product on two venues and ``carry`` spot and perpetual
    pairs, ``[spot, perp]``; ``params`` override an archetype's template
    defaults, which the report draws with when no strategy of it exists.
    """

    instruments: tuple[str, ...] = ()
    only: tuple[str, ...] = ()
    shares: dict[str, float] = field(default_factory=dict)
    cross_venue: tuple[tuple[str, str], ...] = ()
    carry: tuple[tuple[str, str], ...] = ()
    params: dict[str, dict] = field(default_factory=dict)

    def params_for(self, name: str) -> dict:
        """The template parameters of archetype ``name``: the registry's, then the brief's."""
        from blinded.archetypes import REGISTRY

        return dict(REGISTRY[name].params) | dict(self.params.get(name) or {})


@dataclass(frozen=True)
class Group:
    """What the Ladders, sessions and streaks are keyed by: an edge in a brief with
    ``archetypes``, a family otherwise. ``members`` are the edge's runnable archetypes."""

    name: str
    title: str
    share: float
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class Splits:
    """Split fractions and embargo; ``fixed`` holds boundaries a run directory pinned at init."""

    fractions: tuple[float, float, float]
    embargo: pd.Timedelta
    fixed: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=dict)


@dataclass(frozen=True)
class Stats:
    block: pd.Timedelta
    n_boot: int
    seed: int


@dataclass(frozen=True)
class Stock:
    """What an arbitrage strategy holds on its two venues before its first fill: the
    ``stock`` block.

    ``coin`` and ``usdt`` hold each venue's coin in units and its USDT, by
    instrument. When one venue holds more than ``rebalance_at`` of the two
    venues' coin (or USDT), half the difference is sent to the other and
    arrives ``transfer`` later. A transfer of coin costs ``withdrawal_fee_coin``
    units of coin, and one of USDT ``withdrawal_fee_usdt`` USDT. Neither fee has a
    default: each is None until the brief sets it, and the judge refuses to score
    an arbitrage strategy until both are set. A run judged under rules 1 or 2
    charged a USDT transfer the coin fee at the hedge venue's mid and has no
    ``withdrawal_fee_usdt``; its briefs still load.
    """

    coin: dict[str, float]
    usdt: dict[str, float]
    rebalance_at: float = 0.8
    transfer: pd.Timedelta = pd.Timedelta("30min")
    withdrawal_fee_coin: float | None = None
    withdrawal_fee_usdt: float | None = None

    @property
    def venues(self) -> tuple[str, ...]:
        return tuple(self.coin)


@dataclass(frozen=True)
class Brief:
    """A parsed brief. Everything the judge does is a function of this plus the data.

    ``judge_rules`` is the judging rules' version (:data:`blinded.ladder.RULES`). A run's
    is its manifest's, set by :meth:`blinded.vault.RunDir.brief` whatever the brief file
    says; a brief read on its own takes a top-level ``judge_rules``, else the current rules,
    or rules 1 when it is the workspace copy of a run whose ``blinded init`` wrote none.
    ``stock`` is an arbitrage file's two-venue stock (:class:`Stock`), None elsewhere.
    """

    name: str
    source: Path | None
    data: Data
    target: Target
    grid: Grid
    latency: pd.Timedelta
    execution: Execution
    splits: Splits
    stats: Stats
    raw: dict
    families: tuple[Family, ...] = ()
    archetypes: Archetypes | None = None
    judge_rules: int = RULES
    stock: Stock | None = None

    @property
    def has_families(self) -> bool:
        """True when the brief has a ``families`` block; without one the run has one family."""
        return bool(self.raw.get("families"))

    @property
    def has_archetypes(self) -> bool:
        """True when the brief has an ``archetypes`` block. Its families are then implied."""
        return self.archetypes is not None

    def family(self, name: str) -> Family | None:
        return next((f for f in self.families if f.name == name), None)

    @functools.cached_property
    def groups(self) -> tuple[Group, ...]:
        """The sessions' groups: the edges with a runnable archetype, else the families.

        In a brief with ``archetypes`` the menu decides which edges can run
        (:func:`blinded.archetypes.menu`), from the brief's own instruments,
        streams and fixed windows, so this needs a brief that ``blinded
        init`` wrote. A brief with ``families`` has one group per family, and
        one without either has the one group ``directional``.
        """
        if not self.has_archetypes:
            return tuple(Group(f.name, f.name, f.share) for f in self.families)
        from blinded import archetypes

        menu = archetypes.menu(self, archetypes.facts(self))
        return tuple(Group(e, archetypes.EDGES[e].title, menu.shares[e], menu.members(e))
                     for e in menu.edges)  # fmt: skip

    def group(self, name: str) -> Group | None:
        return next((g for g in self.groups if g.name == name), None)


@dataclass(frozen=True)
class Window:
    """A half-open scoring window ``[start, end)``."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __str__(self) -> str:
        return f"{self.label} [{self.start.isoformat()}, {self.end.isoformat()})"


def load(path: str | os.PathLike) -> Brief:
    """Read a brief from a YAML file.

    Relative data paths resolve against the brief's directory and ``${VAR}``
    references are expanded from the environment.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    brief = parse(raw, base=path.parent, source=path)
    if "judge_rules" not in raw and (path.parent / ".judge").is_dir():
        # a run's workspace copy from before the rules had a version: the run's are rules 1
        brief = replace(brief, judge_rules=1)
    return brief


def parse(raw: dict, base: Path | None = None, source: Path | None = None) -> Brief:
    """Build a :class:`Brief` from an already-loaded mapping."""
    data = raw["data"]
    target = raw["target"]
    grid = raw.get("grid", {})
    splits = raw.get("splits", {})
    stats = raw.get("stats", {})
    execution = raw.get("execution", {})

    horizons = tuple(_duration(h) for h in target["horizons"])
    primary = _duration(target.get("primary_horizon", target["horizons"][0]))
    if primary not in horizons:
        raise ValueError(f"primary_horizon {primary} is not one of the horizons {horizons}")
    if target.get("price", "mid") not in ("mid", "microprice"):
        raise ValueError("target.price must be 'mid' or 'microprice'")

    fractions = tuple(float(f) for f in splits.get("fractions", (0.5, 0.25, 0.25)))
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"splits.fractions must be three numbers summing to 1, got {fractions}")

    latency = pd.Timedelta(milliseconds=float(execution.get("latency_ms", 0)))
    fee_tier = _fee_tier(execution)
    embargo = _embargo(splits.get("embargo"), horizons, latency)
    time = data.get("time", {})
    availability = data.get("availability", {})
    families, archetypes = _blocks(raw)
    brief = Brief(
        name=raw.get("name", "unnamed"),
        source=source,
        data=Data(
            path=_data_path(data["path"], base),
            time_column=time.get("column", "ts"),
            time_unit=time.get("unit", "ns"),
            tz=time.get("tz", "UTC"),
            clock=time.get("clock", "receive"),
            instrument_column=data["instrument_column"],
            streams={name: _stream(name, spec) for name, spec in data["streams"].items()},
            default_delay_ms=float(availability.get("default_delay_ms", 0)),
            delay_overrides=tuple(availability.get("overrides") or ()),
            thinned=_thinned(data.get("thinned")),
            cost_model=_cost_model(data.get("cost_model")),
            touch=_touch(data.get("touch"), base, target["instrument"]),
            depth=_depth(data.get("depth"), base),
            theo=_theo(data.get("theo"), base),
        ),
        target=Target(
            instrument=target["instrument"],
            price=target.get("price", "mid"),
            horizons=horizons,
            primary_horizon=primary,
        ),
        grid=Grid(
            step=_duration(grid.get("step", "100ms")),
            start=pd.Timestamp(grid["start"]) if grid.get("start") else None,
            warmup=_duration(grid.get("warmup", "5min")),
        ),
        latency=latency,
        execution=Execution(
            latency=latency,
            taker_fee_bps=fee_tier[0],
            maker_fee_bps=fee_tier[1],
            fee_tier=fee_tier[2],
            max_position=float(execution.get("max_position", 1.0)),
            min_rebalance=float(execution.get("min_rebalance", 0.0)),
            fees={str(k): float(v) for k, v in (execution.get("fees") or {}).items()},
            tick_size=_positive(execution, "tick_size"),
            lot_size=_positive(execution, "lot_size"),
            cancel_latency=_latency(execution, "cancel_latency_ms", latency),
            response_latency=_latency(execution, "response_latency_ms"),
            deadline=_deadline(execution),
            hedge_latency=_latency(execution, "hedge_latency_ms"),
            hedge_fee_bps=_fee(execution, "hedge_fee_bps"),
        ),
        splits=Splits(
            fractions=fractions,
            embargo=embargo,
            fixed=_fixed_windows(splits.get("windows"), time.get("tz", "UTC")),
        ),
        stats=Stats(
            block=_duration(stats.get("block", "5min")),
            n_boot=int(stats.get("n_boot", 1000)),
            seed=int(stats.get("seed", 0)),
        ),
        raw=raw,
        families=families,
        archetypes=archetypes,
        judge_rules=check_rules(raw.get("judge_rules", RULES)),
        stock=_stock(raw.get("stock")),
    )
    _check_thinned(brief)
    _check_hedge(brief)
    return brief


def _blocks(raw: dict) -> tuple[tuple[Family, ...], Archetypes | None]:
    """The families and the archetypes block; a brief has one or the other, or neither."""
    target = raw["target"]["instrument"]
    if "archetypes" not in raw:
        return _families(raw.get("families"), target), None
    if raw.get("families"):
        raise ValueError("a brief has an archetypes block or a families block, not both")
    spec = _archetypes(raw.get("archetypes") or {})
    return _implied(spec, target), spec


def default_embargo(horizons: tuple[pd.Timedelta, ...]) -> pd.Timedelta:
    """``max(5 x the longest horizon, 5 minutes)``."""
    return max(5 * max(horizons), pd.Timedelta("5min"))


def windows(brief: Brief, first: pd.Timestamp, last: pd.Timestamp) -> dict[str, Window]:
    """Named windows for a data span: ``train``, ``val``, ``test`` and ``all``.

    ``first`` and ``last`` are the first and last grid points. Train,
    validation and test are consecutive time fractions of the span, with an
    embargo at the start of validation and of test. The two split points are
    rounded to the nearest whole minute (or grid step, if that is longer), so
    the boundaries do not move with the exact time of the first and last
    event. All windows are half-open; test ends one step after the last point.
    A brief written by ``blinded init`` carries the boundaries it fixed under
    ``splits.windows``, and those are returned as they are.
    """
    if brief.splits.fixed:
        named = {k: Window(k, start, end) for k, (start, end) in brief.splits.fixed.items()}
        return named | {"all": Window("all", first, last + brief.grid.step)}
    span = last - first
    f_train, f_val, _ = brief.splits.fractions
    embargo = brief.splits.embargo
    unit = max(brief.grid.step, pd.Timedelta("1min"))
    train_end = (first + f_train * span).round(unit)
    val_end = (first + (f_train + f_val) * span).round(unit)
    after_last = last + brief.grid.step
    return {
        "train": Window("train", first, train_end),
        "val": Window("val", train_end + embargo, val_end),
        "test": Window("test", val_end + embargo, after_last),
        "all": Window("all", first, after_last),
    }


def window(spec: str, brief: Brief, first: pd.Timestamp, last: pd.Timestamp) -> Window:
    """Resolve ``train``/``val``/``test``/``all`` or an explicit ``START/END``."""
    named = windows(brief, first, last)
    if spec in named:
        return named[spec]
    if "/" not in spec:
        raise ValueError(f"window must be one of {sorted(named)} or START/END, got {spec!r}")
    start, end = (_timestamp(part, brief.data.tz) for part in spec.split("/", 1))
    if end <= start:
        raise ValueError(f"window end {end} is not after its start {start}")
    return Window(spec, start, end)


def _embargo(value, horizons: tuple[pd.Timedelta, ...], latency: pd.Timedelta) -> pd.Timedelta:
    if value is None:
        return default_embargo(horizons)
    embargo = _duration(value)
    reach = latency + max(horizons)
    if embargo <= reach:
        raise ValueError(
            f"splits.embargo {embargo} must be longer than latency + the longest horizon "
            f"({reach}): the last train target would reach into validation"
        )
    return embargo


def _positive(execution: dict, key: str) -> float | None:
    """``execution.tick_size`` or ``lot_size``: None when absent, else a positive number."""
    if execution.get(key) is None:
        return None
    value = float(execution[key])
    if not value > 0:
        raise ValueError(f"execution.{key} must be positive, got {execution[key]}")
    return value


def _fee_tier(execution: dict) -> tuple[float, float, str | None]:
    """The taker and maker fees, and ``execution.fee_tier`` when it names a tier of the fee
    table (:func:`blinded.fees.tier`): its fees, refused when the tier is unknown or when the
    brief also gives a fee the tier sets. A market maker program publishes no taker fee (the
    ordinary tier sets it), so with one of those the brief gives ``taker_fee_bps``."""
    taker, maker, key = (execution.get(k) for k in ("taker_fee_bps", "maker_fee_bps", "fee_tier"))
    if not key:
        return float(taker or 0.0), float(maker or 0.0), None
    from blinded import fees

    t = fees.tier(str(key))
    if maker is not None:
        raise ValueError(f"execution.fee_tier {key} sets the maker fee: drop maker_fee_bps")
    if t.taker_bps == t.taker_bps:  # published
        if taker is not None:
            raise ValueError(f"execution.fee_tier {key} sets the taker fee: drop taker_fee_bps")
        taker = t.taker_bps
    elif taker is None:
        raise ValueError(f"execution.fee_tier {key} publishes no taker fee (the venue's "
                         "ordinary tier sets it): give taker_fee_bps")  # fmt: skip
    return float(taker), t.maker_bps, t.key


def _fee(execution: dict, key: str) -> float | None:
    """A fee in bps that may be absent: None, else a number."""
    if execution.get(key) is None:
        return None
    return float(execution[key])


_STOCK_KEYS = {"rebalance_at", "transfer_minutes", "withdrawal_fee_coin", "withdrawal_fee_usdt"}


def _stock(spec) -> Stock | None:
    """The ``stock`` block: each venue's ``coin`` and ``usdt``, keyed by instrument, and the
    rebalance rule. ``withdrawal_fee_coin`` must be named, and may be null until it is looked
    up (:class:`Stock`)."""
    if not spec:
        return None
    venues = {k: v for k, v in spec.items() if k not in _STOCK_KEYS}
    coin, usdt = {}, {}
    for name, held in venues.items():
        if not isinstance(held, dict) or set(held) - {"coin", "usdt"}:
            raise ValueError(f"stock.{name} must give coin and usdt, the venue's stock")
        coin[str(name)], usdt[str(name)] = (float(held.get(k) or 0.0) for k in ("coin", "usdt"))
        if coin[str(name)] < 0 or usdt[str(name)] < 0:
            raise ValueError(f"stock.{name} must not be negative")
    if len(venues) != 2:
        raise ValueError(f"stock names {len(venues)} venues; an arbitrage holds stock on two")
    at = float(spec.get("rebalance_at", 0.8))
    if not 0.5 < at <= 1.0:
        raise ValueError(f"stock.rebalance_at must be above 0.5 and at most 1, got {at}")
    minutes = float(spec.get("transfer_minutes", 30))
    if minutes < 0:
        raise ValueError(f"stock.transfer_minutes must not be negative, got {minutes}")
    if "withdrawal_fee_coin" not in spec:
        raise ValueError("stock.withdrawal_fee_coin must be given: look it up on both venues' "
                         "withdrawal pages, or write null until then")  # fmt: skip
    fees = {}
    for key in ("withdrawal_fee_coin", "withdrawal_fee_usdt"):
        fee = spec.get(key)
        if fee is not None and float(fee) < 0:
            raise ValueError(f"stock.{key} must not be negative, got {fee}")
        fees[key] = None if fee is None else float(fee)
    return Stock(coin=coin, usdt=usdt, rebalance_at=at, transfer=pd.Timedelta(minutes=minutes),
                 **fees)  # fmt: skip


def _check_hedge(brief: Brief) -> None:
    """A hedge reads the depth file as-of its arrival, the fill's stamp plus the response
    latency on the quote venue plus ``hedge_latency_ms``; a depth file's level 0 trails the
    touch by ``lag_ms``, so a hedge that arrives sooner after its fill than that would read a
    snapshot older than the fill."""
    ex, depth = brief.execution, brief.data.depth
    if ex.hedge_latency is None or depth is None:
        return
    delay = ex.response + ex.hedge_latency
    lag = pd.Timedelta(milliseconds=depth.lag_ms)
    if delay < lag:
        raise ValueError(
            f"execution.hedge_latency_ms ({durations.label(ex.hedge_latency)}) plus the response "
            f"latency ({durations.label(ex.response)}) is under data.depth.lag_ms "
            f"({durations.label(lag)}): the hedge would read a depth snapshot older than the "
            "fill it hedges"
        )


def _deadline(execution: dict) -> pd.Timedelta:
    """``execution.deadline_ms``: how long the maker-taker row rests an order before it
    crosses; 2 s when absent, and positive."""
    value = execution.get("deadline_ms")
    if value is None:
        return pd.Timedelta("2s")
    if not float(value) > 0:
        raise ValueError(f"execution.deadline_ms must be positive, got {value}")
    return pd.Timedelta(milliseconds=float(value))


def _latency(execution: dict, key: str, order: pd.Timedelta | None = None
             ) -> pd.Timedelta | None:  # fmt: skip
    """``execution.cancel_latency_ms`` or ``response_latency_ms``: None when absent.

    A cancel may not be slower than the order (``order``): the fill engine
    holds one resting order a side, and a cancel that lands after its
    replacement would leave two.
    """
    if execution.get(key) is None:
        return None
    value = pd.Timedelta(milliseconds=float(execution[key]))
    if value < pd.Timedelta(0):
        raise ValueError(f"execution.{key} must not be negative, got {execution[key]}")
    if order is not None and value > order:
        raise ValueError(f"execution.{key} ({durations.label(value)}) must not be longer than "
                         f"latency_ms ({durations.label(order)}): the replacement order would "
                         "reach the book before the cancel of the one it replaces")  # fmt: skip
    return value


def _touch(spec, base: Path | None, target: str) -> TouchFile | None:
    """``data.touch``: the touch stream's path, resolved as ``data.path`` is, its tick, and the
    instrument it belongs to, the target unless named."""
    if not spec:
        return None
    if spec.get("tick_size") is None or not float(spec["tick_size"]) > 0:
        raise ValueError("data.touch needs a positive tick_size: its prices are in ticks")
    return TouchFile(path=_data_path(str(spec["path"]), base), tick_size=float(spec["tick_size"]),
                     instrument=str(spec.get("instrument") or target))  # fmt: skip


def _depth(spec, base: Path | None) -> DepthFile | None:
    """``data.depth``: the depth file's path, the instrument it belongs to, its levels and the
    measured lag of its level 0 behind the touch."""
    if not spec:
        return None
    if not spec.get("venue"):
        raise ValueError("data.depth needs a venue: the instrument whose levels it holds")
    levels = int(spec.get("levels", 25))
    if levels < 1:
        raise ValueError(f"data.depth.levels must be at least 1, got {levels}")
    return DepthFile(path=_data_path(str(spec["path"]), base), venue=str(spec["venue"]),
                     levels=levels, lag_ms=float(spec.get("lag_ms") or 0.0))  # fmt: skip


def _theo(spec, base: Path | None):
    """``data.theo``: the coin, its legs, the anchor (one leg's name, or weights by leg) and,
    in a run's brief, the file holding the filter's output (:class:`blinded.theo.TheoSpec`)."""
    if not spec:
        return None
    from blinded.theo import TheoSpec

    legs = tuple(str(x) for x in spec.get("legs") or ())
    if len(legs) < 2 or not spec.get("coin"):
        raise ValueError("data.theo needs a coin and two legs or more: the instruments quoting it")
    anchor = spec.get("anchor")
    if isinstance(anchor, str):
        anchor = {anchor: 1.0}
    anchor = {str(k): float(v) for k, v in (anchor or {}).items()}
    if not anchor or set(anchor) - set(legs) or sum(anchor.values()) <= 0:
        raise ValueError("data.theo.anchor must name a leg, or give legs weights that sum above 0")
    path = _data_path(str(spec["path"]), base) if spec.get("path") else None
    return TheoSpec(coin=str(spec["coin"]), legs=legs, anchor=anchor, path=path)


def _thinned(spec) -> Thinned | None:
    if not spec:
        return None
    offsets = tuple(pd.Timedelta(milliseconds=float(o)) for o in spec["offsets_ms"])
    return Thinned(step=_duration(spec["step"]), offsets=offsets)


COST_KINDS = ("half_spread",)
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _cost_model(spec) -> CostModel | None:
    """``data.cost_model``; the day is the last ``YYYY-MM-DD`` in ``source``."""
    if not spec:
        return None
    kind = spec.get("kind")
    if kind not in COST_KINDS:
        raise ValueError(f"data.cost_model.kind must be one of {COST_KINDS}, got {kind!r}")
    spreads = {str(k): float(v) for k, v in (spec.get("half_spread_bps") or {}).items()}
    bad = [k for k, v in spreads.items() if not v > 0]
    if bad:
        raise ValueError(f"data.cost_model.half_spread_bps must be positive; {bad} are not")
    source = str(spec.get("source", "a top-of-book day"))
    days = _DAY.findall(source)
    return CostModel(kind=kind, fill=str(spec.get("fill", "next bar close")), source=source,
                     half_spread_bps=spreads,
                     day=pd.Timestamp(days[-1], tz="UTC") if days else None)  # fmt: skip


def _check_thinned(brief: Brief) -> None:
    """A thinned file must keep every stamp the judge reads: the panel's, the fills' and the
    ends of the forward returns."""
    thin = brief.data.thinned
    if thin is None:
        return
    name = durations.label
    where = f"the data file is thinned to a {name(thin.step)} grid"
    if brief.grid.step != thin.step:
        raise ValueError(f"{where}, but grid.step is {name(brief.grid.step)}")
    ms = [brief.data.default_delay_ms, *(float(r["delay_ms"]) for r in brief.data.delay_overrides)]
    for delay in (pd.Timedelta(milliseconds=m) for m in ms):
        if not thin.keeps(-delay):
            raise ValueError(f"{where} and does not keep the panel at a {name(delay)} delay")
    if not thin.keeps(brief.latency):
        raise ValueError(f"{where} and does not keep the fills at {name(brief.latency)}")
    odd = [name(h) for h in brief.target.horizons if h.value % thin.step.value]
    if odd:
        raise ValueError(f"{where}; horizons {', '.join(odd)} are not whole steps of it")


def _families(spec, target: str) -> tuple[Family, ...]:
    """The brief's families; without a ``families`` block, one directional family on the target."""
    if not spec:
        return (Family("directional", 1.0, instruments=(target,)),)
    if not isinstance(spec, dict):
        raise ValueError("families must map each family's name to its settings")
    unknown = sorted(set(spec) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown families {unknown}; the families are {', '.join(FAMILIES)}")
    bodies = {name: body or {} for name, body in spec.items()}
    given = {name: body["share"] for name, body in bodies.items() if "share" in body}
    if given and len(given) != len(bodies):
        raise ValueError("give every family a share, or none for equal shares")
    shares = {n: float(s) for n, s in given.items()} or {n: 1 / len(bodies) for n in bodies}
    if any(s <= 0 for s in shares.values()) or abs(sum(shares.values()) - 1) > 1e-9:
        raise ValueError(f"family shares must be positive and sum to 1, got {shares}")
    return tuple(_family(name, shares[name], body, target) for name, body in bodies.items())


def _family(name: str, share: float, body: dict, target: str) -> Family:
    if name == "directional":
        return Family(name, share, instruments=tuple(body.get("instruments") or (target,)))
    if name == "lead_lag":
        leaders, followers = tuple(body.get("leaders") or ()), tuple(body.get("followers") or ())
        if not any(a != b for a in leaders for b in followers):
            raise ValueError("lead_lag needs a leader and a follower that are not the same")
        return Family(name, share, leaders=leaders, followers=followers)
    if name == "basket":
        from blinded.metrics import XS_MIN_LEGS

        legs = tuple(str(s) for s in body.get("instruments") or ())
        if len(set(legs)) < XS_MIN_LEGS or len(set(legs)) != len(legs):
            raise ValueError(f"a basket needs at least {XS_MIN_LEGS} different instruments, "
                             f"got {list(legs)}")  # fmt: skip
        return Family(name, share, instruments=legs)
    pairs = tuple(tuple(str(s) for s in pair) for pair in body.get("pairs") or ())
    if not pairs:
        raise ValueError("the pairs family needs pairs: [[A, B], ...]")
    for pair in pairs:
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"a pair is two different instruments, got {list(pair)}")
    defaults = PAIRS_DEFAULTS | dict(body.get("defaults") or {})
    if not 0 <= float(defaults["z_out"]) <= float(defaults["z_in"]):
        raise ValueError(f"pairs defaults need 0 <= z_out <= z_in, got {defaults}")
    return Family(name, share, pairs=pairs, defaults=defaults)


def _archetypes(spec) -> Archetypes:
    from blinded.archetypes import EDGES, REGISTRY

    if not isinstance(spec, dict):
        raise ValueError("archetypes must be a mapping: instruments, only, shares, ...")
    unknown = sorted(set(spec) - {"instruments", "only", "shares", "cross_venue", "carry",
                                  "params"})  # fmt: skip
    if unknown:
        raise ValueError(f"unknown keys in archetypes: {unknown}")
    instruments = tuple(str(s) for s in spec.get("instruments") or ())
    if len(set(instruments)) != len(instruments):
        raise ValueError("archetypes.instruments lists an instrument twice")
    only = tuple(str(n) for n in spec.get("only") or ())
    bad = [n for n in only if n not in REGISTRY]
    if bad:
        raise ValueError(f"archetypes.only names {bad}, which are not archetypes; the "
                         f"archetypes are {', '.join(REGISTRY)}")  # fmt: skip
    shares = {str(k): float(v) for k, v in (spec.get("shares") or {}).items()}
    for edge in shares:
        if edge not in EDGES:
            raise ValueError(f"{edge} is not an edge; the edges are {', '.join(EDGES)}")
    if shares and (any(v <= 0 for v in shares.values()) or abs(sum(shares.values()) - 1) > 1e-9):
        raise ValueError(f"archetypes.shares must be positive and sum to 1, got {shares}")
    pairs = {}
    for key in ("cross_venue", "carry"):
        pairs[key] = tuple(tuple(str(s) for s in p) for p in spec.get(key) or ())
        for pair in pairs[key]:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(f"an archetypes.{key} pair is two different instruments, "
                                 f"got {list(pair)}")  # fmt: skip
    params = {str(k): dict(v or {}) for k, v in (spec.get("params") or {}).items()}
    bad = [n for n in params if n not in REGISTRY]
    if bad:
        raise ValueError(f"archetypes.params names {bad}, which are not archetypes")
    return Archetypes(instruments, only, shares, pairs["cross_venue"], pairs["carry"], params)


def _implied(spec: Archetypes, target: str) -> tuple[Family, ...]:
    """The mechanism families an archetypes block implies, so that binding, the judge's
    targets and the report's family machinery run unchanged.

    ``directional`` over every instrument; ``lead_lag`` with every instrument
    both leader and follower; ``pairs`` over every unordered pair in list
    order, then the declared cross-venue and carry pairs; ``basket`` over
    every instrument when there are at least 20. Without instruments listed
    (before ``blinded init`` writes them) the one instrument is the target.
    """
    from blinded.archetypes import CROSS_SECTION

    inst = spec.instruments or (target,)
    out = [Family("directional", 1.0, instruments=inst)]
    if len(inst) > 1:
        out.append(Family("lead_lag", 1.0, leaders=inst, followers=inst))
    pairs = list(itertools.combinations(inst, 2))
    seen = {frozenset(p) for p in pairs}
    for pair in (*spec.cross_venue, *spec.carry):
        if frozenset(pair) not in seen:
            seen.add(frozenset(pair))
            pairs.append(pair)
    if pairs:
        out.append(Family("pairs", 1.0, pairs=tuple(pairs),
                          defaults=spec.params_for("pairs_trading")))  # fmt: skip
    if len(inst) >= CROSS_SECTION:
        out.append(Family("basket", 1.0, instruments=inst))
    share = 1 / len(out)
    return tuple(Family(f.name, share, f.instruments, f.leaders, f.followers, f.pairs,
                        f.defaults) for f in out)  # fmt: skip


def _fixed_windows(spec: dict | None, tz: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if not spec:
        return {}
    # a forward brief (blinded.forward) fixes one window, the recorded day it judges
    if set(spec) not in ({"train", "val", "test"}, {"forward"}):
        raise ValueError("splits.windows needs train, val and test (or, in a forward brief, "
                         f"forward alone), got {sorted(spec)}")  # fmt: skip
    out = {}
    for name, text in spec.items():
        start, end = (_timestamp(part, tz) for part in str(text).split("/", 1))
        out[name] = (start, end)
    return out


def _stream(name: str, spec: dict) -> Stream:
    return Stream(
        name=name,
        filter=spec.get("filter"),
        fields=dict(spec["fields"]),
        side_values=dict(spec.get("side_values") or {}),
    )


def _data_path(value: str, base: Path | None) -> Path:
    def expand(match: re.Match) -> str:
        var = match.group(1)
        if var not in os.environ:
            raise ValueError(f"data.path uses ${{{var}}}, which is not set in the environment")
        return os.environ[var]

    path = Path(_ENV_VAR.sub(expand, value)).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def _duration(value) -> pd.Timedelta:
    return durations.timedelta(value)


def _timestamp(text: str, tz: str) -> pd.Timestamp:
    ts = pd.Timestamp(text.strip())
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
