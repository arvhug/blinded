"""The feature contract: decorators, loading feature files, and the static scan."""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib.util
import inspect
import itertools
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from blinded.panel import Panel

_FORBIDDEN_MODULES = {"os", "subprocess", "socket", "pickle", "requests", "urllib", "ctypes"}
_FORBIDDEN_CALLS = {"open", "eval", "exec", "__import__"}
# a call named like a write (to_*, write*, save*, dump*) is allowed only when it is one of
# these, conversions that never take a path: the five that the 628 published trial files
# make (survey of 20 Oct 2026) and their kind in polars, pandas and numpy. Every other to_*
# can write a file given a path: to_json, to_html, to_xml, to_markdown, to_latex, to_string,
# to_feather, to_hdf, to_excel, to_stata, to_sql, to_parquet, to_csv, to_pickle
SAFE_CALLS = frozenset({
    "to_list", "to_series", "to_numpy", "to_pandas", "to_timedelta", "to_frame", "to_dict",
    "to_dicts", "to_arrow", "to_physical", "to_struct", "to_dummies", "to_numeric",
    "to_datetime", "to_period", "to_timestamp", "to_pydatetime", "to_flat_index", "to_offset",
})  # fmt: skip
_WRITE_LIKE = ("to_", "write", "save", "dump")
_CLOCK_READS = {("time", "time"), ("time", "time_ns"), ("datetime", "now"), ("datetime", "utcnow")}
_AGGREGATES = {
    "count", "kurt", "kurtosis", "last", "len", "max", "mean", "median", "min",
    "n_unique", "nunique", "quantile", "rank", "skew", "std", "sum", "var",
}  # fmt: skip
# pandas window objects: x.rolling(5).mean() aggregates per window, not over the column
_WINDOWED = {"rolling", "expanding", "ewm", "resample", "groupby", "group_by", "group_by_dynamic"}
_ROW_WISE = {"fill_null", "fillna", "clip", "where", "mask", "add", "sub", "mul", "div", "truediv"}
# protected mode: code runs inside the judge, next to validation data
ALLOWED_IMPORTS = {
    "__future__", "blinded", "collections", "dataclasses", "functools", "itertools", "math",
    "numpy", "pandas", "polars", "statistics", "typing",
}  # fmt: skip
_PROTECTED_CALLS = {
    "__import__", "breakpoint", "compile", "delattr", "eval", "exec", "getattr", "globals",
    "input", "locals", "open", "setattr", "vars",
}  # fmt: skip
# calls that take a path and read or write it themselves, the ones not named read_*, scan_*,
# to_*, write*, save* or dump*: the I/O of numpy, pandas and polars at the versions pinned
# (survey of 21 Oct 2026), none of which the 628 published trial files use
_FILE_METHODS = {
    "fromfile", "genfromtxt", "load", "loadtxt", "memmap", "save", "savetxt", "savez",
    "savez_compressed", "sink_csv", "sink_ipc", "sink_parquet", "tofile", "fromregex",
    "open_memmap", "DataSource", "serialize", "deserialize", "ExcelFile", "ExcelWriter",
    "HDFStore", "sql", "SQLContext", "register_plugin_function",
}  # fmt: skip
_DUNDER_OK = {"__name__", "__doc__", "__init__", "__call__"}
# protected mode: attributes that reach a module the imports refuse, through a library that
# holds it (pd.io.common.os, np.ctypeslib.ctypes), or memory past an array's end
_MODULE_ATTRS = {
    "os", "sys", "io", "subprocess", "socket", "builtins", "shutil", "ctypes", "ctypeslib",
    "pathlib", "importlib", "tempfile", "pickle", "marshal", "gc", "inspect", "threading",
    "multiprocessing", "signal", "mmap", "sqlite3", "as_strided", "plugins",
}  # fmt: skip
_module_ids = itertools.count()

Series = pl.Expr | pl.Series | pd.Series
Output = Series | Mapping[str, Series]


class ContractError(ValueError):
    """A feature or strategy that does not fit the brief, or returned the wrong shape.

    The message is written from the decorator's arguments and the brief, never
    from data.
    """


class Legs(dict):
    """A strategy's positions in several instruments: one float64 array on the grid per leg."""


QUOTE_KEYS = ("bid", "ask", "size", "skew", "max_inventory", "lean", "fair", "fair_bid",
              "fair_ask")  # fmt: skip


class Quotes(Legs):
    """A market-making strategy's quote parameters: one float64 array on the grid per key.

    The keys are those of :data:`QUOTE_KEYS`, ``lean`` and the fairs optional (``fair``, or
    ``fair_bid`` and ``fair_ask``, each side's own price):
    :func:`blinded.sim_maker.quote` builds one, and the fill engine
    (:mod:`blinded.sim_maker`) turns it into resting orders. Its keys are
    parameters rather than instruments, but it is compared, padded and cut
    key by key exactly as a strategy's legs are, so the truncation test runs
    on it unchanged and names the key that disagreed.
    """


# the templates whose strategies return quotes: market making and maker-taker arbitrage
QUOTING = ("maker", "arb")
_TEMPLATES = {"directional": "threshold/linear", "lead_lag": "threshold/linear",
              "pairs": "pairs", "basket": "basket"}  # fmt: skip


@dataclass(frozen=True)
class Binding:
    """Where a feature or strategy sits in the brief.

    ``legs`` is the one instrument it predicts or trades, or its pair in the
    brief's order. ``sign`` is -1 for a pair declared the other way round: a
    feature's values are then negated so that high still means A over B rises.
    ``archetype`` and ``edge`` are the decorator's tag, ``template`` the sim
    that turns its positions into fills, and ``group`` what its Ladder and
    sessions are keyed by: the edge in a brief with ``archetypes``, else the
    family.
    """

    family: str
    legs: tuple[str, ...]
    sign: float = 1.0
    edge: str | None = None
    archetype: str | None = None
    # both follow from the rest and the brief, so they take no part in equality
    template: str = field(default="threshold/linear", compare=False)
    group: str = field(default="", compare=False)
    # an arbitrage's hedge venue (``bl.cross_venue_making``), where the sim hedges each fill
    hedge: str | None = field(default=None, compare=False)

    @property
    def label(self) -> str:
        return "/".join(self.legs)


class Feature:
    """A feature function registered with :func:`feature`.

    Calling it runs the wrapped function unchanged. ``target`` is the
    instrument whose return it predicts (None for the brief's target), and
    ``pair`` the pair whose spread return it predicts, for the pairs family.
    ``archetype`` and ``roles`` are set by an archetype's decorator
    (:mod:`blinded.archetypes`), never by :func:`feature`.
    """

    def __init__(
        self,
        fn: Callable[[Panel], Output],
        hypothesis: str,
        family: str = "directional",
        target: str | None = None,
        pair: tuple[str, str] | None = None,
        archetype: str | None = None,
        roles: dict[str, str] | None = None,
    ):
        functools.update_wrapper(self, fn)
        self.fn = fn
        self.name = fn.__name__
        self.hypothesis = hypothesis
        self.family = family
        self.target = target
        self.pair = None if pair is None else tuple(pair)
        self.archetype = archetype
        self.roles = dict(roles or {})

    @property
    def legs(self) -> tuple[str, ...]:
        """The pair, or the target; empty when the target is left to the brief."""
        return self.pair or ((self.target,) if self.target else ())

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def evaluate(self, brief, window: str = "train", latency_ms: float | None = None):
        """Run the judge on this feature and return an :class:`~blinded.evaluate.Result`."""
        from blinded.evaluate import evaluate

        return evaluate(self, brief, window=window, latency_ms=latency_ms)

    def __repr__(self) -> str:
        return f"<feature {self.name}: {self.hypothesis}>"


class Strategy:
    """A strategy registered with :func:`strategy`, or with an archetype's decorator.

    Calling it runs the wrapped function unchanged. A strategy of
    ``bl.market_making`` returns quotes (:class:`Quotes`) rather than a
    position.
    """

    def __init__(
        self,
        fn: Callable,
        requires: list[str],
        family: str | None = "directional",
        trades: str | None = None,
        pair: tuple[str, str] | None = None,
        archetype: str | None = None,
        roles: dict[str, str] | None = None,
    ):
        functools.update_wrapper(self, fn)
        self.fn = fn
        self.name = fn.__name__
        self.requires = list(requires)
        self.family = family
        self.trades = trades
        self.pair = None if pair is None else tuple(pair)
        self.archetype = archetype
        self.roles = dict(roles or {})

    @property
    def legs(self) -> tuple[str, ...]:
        """The pair, or the instrument traded; empty when that is left to the brief."""
        return self.pair or ((self.trades,) if self.trades else ())

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def evaluate(self, brief, window: str = "train", latency_ms: float | None = None, **kwargs):
        """Run the judge on this strategy; see :func:`~blinded.evaluate.evaluate_strategy`."""
        from blinded.evaluate import evaluate_strategy

        return evaluate_strategy(self, brief, window=window, latency_ms=latency_ms, **kwargs)

    def __repr__(self) -> str:
        return f"<strategy {self.name} requires={self.requires}>"


def feature(
    hypothesis: str,
    family: str = "directional",
    target: str | None = None,
    pair: tuple[str, str] | None = None,
) -> Callable[[Callable], Feature]:
    """Register a function ``panel -> expression`` as a feature.

    The function may return a polars expression over ``panel.frame`` (the
    usual case), a polars Series with one value per grid point, or a pandas
    Series indexed like ``panel.to_pandas()``.

    Parameters
    ----------
    hypothesis
        One line saying what the feature is meant to capture.
    family
        ``directional``, ``lead_lag``, ``pairs`` or ``basket``; it must be in the brief.
        A basket feature returns a mapping with one value per leg of the
        brief's basket family, ``panel.legs``.
    target
        The instrument whose forward return the feature predicts: one of the
        directional family's instruments or the lead-lag family's followers.
        Defaults to the brief's target instrument.
    pair
        ``(A, B)``, for the pairs family only: the feature predicts the
        forward return of A minus B, so high means the spread of A over B is
        expected to rise.

    Examples
    --------
    >>> import blinded as bl
    >>> @bl.feature("perp leads spot; the basis mean-reverts")
    ... def basis(panel):
    ...     return panel.asof("PERP", "mid") / panel.asof("SPOT", "mid") - 1
    """
    if callable(hypothesis):
        raise TypeError('bl.feature needs a hypothesis: @bl.feature("what this captures")')

    def register(fn: Callable) -> Feature:
        return Feature(fn, hypothesis, family, target, pair)

    return register


def strategy(
    requires: list[str] = (),
    family: str = "directional",
    trades: str | None = None,
    pair: tuple[str, str] | None = None,
) -> Callable[[Callable], Strategy]:
    """Register a function ``(panel, features) -> position`` as a strategy.

    ``features`` maps each name in ``requires`` to a polars expression for that
    feature's values; the function returns the position to hold, in
    ``[-max_position, max_position]``, as an expression or a Series, NaN for
    flat. A pairs strategy returns a mapping with a position for each leg of
    its ``pair``, in units of one notional per leg; a single-leg strategy may
    return a one-key mapping for the instrument it ``trades``.

    Examples
    --------
    >>> import blinded as bl
    >>> @bl.strategy(requires=["basis"])
    ... def basis_threshold(panel, features):
    ...     return bl.threshold(features["basis"], z_in=2.0, z_out=0.5)
    """

    def register(fn: Callable) -> Strategy:
        return Strategy(fn, requires, family, trades, pair)

    return register


def bind(item: Feature | Strategy, brief, menu=None) -> Binding:
    """Check a feature or strategy against the brief and find its legs.

    Raises :class:`ContractError` when the family is not in the brief, the
    target or traded instrument is not one of the family's, or the pair is not
    one of its pairs. A pair given the other way round is put in the brief's
    order, with ``sign`` -1.

    In a brief with ``archetypes`` the item must carry an archetype the
    ``menu`` runs (:func:`blinded.archetypes.menu`; computed from the brief
    when not given), and a pair declared as one product on two venues or as
    a carry pair belongs to that archetype alone. The message of a refusal is
    the menu's line. In any other brief, market making is refused with its
    line on a file without a touch stream
    (:func:`blinded.archetypes.touch_lack`).
    """
    from blinded import archetypes

    name = getattr(item, "archetype", None)
    arch = archetypes.REGISTRY.get(name) if name else None
    no_touch = arch is not None and arch.needs.touch and archetypes.touch_lack(brief) is not None
    if no_touch and not brief.has_archetypes:
        verdict = archetypes.Verdict("refused", "sim", archetypes.touch_line(brief))
        raise ContractError(archetypes.line(arch, brief.name, verdict))
    if brief.has_archetypes:
        if arch is None:
            raise ContractError(
                "this run studies archetypes; decorate with one of the archetypes in ARCHETYPES.md"
            )
        if menu is None:
            menu = archetypes.menu(brief, archetypes.facts(brief))
        refusal = menu.line(name)
        if refusal is not None:
            raise ContractError(refusal)
        _declared_pair_rule(item, arch, brief.archetypes)
    if arch is not None and arch.needs.touch and arch.template == "maker":
        _touch_rule(item, brief)
    hedge = _hedge_rule(item, brief) if arch is not None and arch.template == "arb" else None
    bound = _bind_family(item, brief)
    return dataclasses.replace(
        bound,
        edge=arch.edge if arch else None,
        archetype=name if arch else None,
        template=arch.template if arch else _TEMPLATES[bound.family],
        group=arch.edge if brief.has_archetypes else bound.family,
        hedge=hedge,
    )


def _hedge_rule(item, brief) -> str:
    """An arbitrage quotes the file's quote venue and hedges on its depth file's venue, the two
    declared as one product (:func:`blinded.archetypes.arb_pair`)."""
    from blinded import archetypes

    lack = archetypes.arb_lack(brief)
    if lack is not None:
        verdict = archetypes.Verdict("refused", lack[0], lack[1])
        raise ContractError(archetypes.line(archetypes.REGISTRY["cross_venue_making"],
                                            brief.name, verdict))  # fmt: skip
    quote = getattr(item, "trades", None) or getattr(item, "target", None)
    hedge = (item.roles or {}).get("hedge")
    pair = archetypes.arb_pair(brief)
    if (quote, hedge) != pair:
        raise ContractError(f"cross_venue_making on this file quotes {pair[0]} and hedges on "
                            f"{pair[1]}, the venue of its depth file; got quote={quote}, "
                            f"hedge={hedge}")  # fmt: skip
    return hedge


def _touch_rule(item, brief) -> None:
    """On a thinned file the fill engine has a stream only for the touch file's instrument;
    on an arbitrage file (one with a depth file) only for the quote venue's."""
    touch, inst = brief.data.touch, getattr(item, "trades", None) or getattr(item, "target", None)
    if touch is None or inst in (None, touch.instrument):
        return
    if brief.data.thinned is not None:
        raise ContractError(f"market making on this file quotes {touch.instrument}, the "
                            f"instrument of its touch stream; the book of {inst} is thinned to "
                            "the read lattice")  # fmt: skip
    if brief.data.depth is not None and inst == brief.data.depth.venue:
        raise ContractError(f"market making on this file quotes {touch.instrument}, the "
                            f"instrument of its touch stream; {inst} is the hedge venue of "
                            "an arbitrage, whose fills bl.cross_venue_making hedges "
                            "there")  # fmt: skip


def _declared_pair_rule(item, arch, spec) -> None:
    """A cross-venue pair is cross_venue_arbitrage's and a carry pair basis_trade's, only."""
    if item.pair is None:
        return
    a, b = item.pair
    pair = frozenset(item.pair)
    cross = {frozenset(p) for p in spec.cross_venue}
    carry = {frozenset(p) for p in spec.carry}
    if arch.name == "pairs_trading" and pair in cross:
        raise ContractError(f"pair ({a}, {b}) is one product on two venues; use "
                            "bl.cross_venue_arbitrage")  # fmt: skip
    if arch.name == "pairs_trading" and pair in carry:
        raise ContractError(f"pair ({a}, {b}) is a carry pair; use bl.basis_trade")
    if arch.name == "cross_venue_arbitrage" and pair not in cross:
        listed = ", ".join(f"({x}, {y})" for x, y in spec.cross_venue) or "none"
        raise ContractError(f"pair ({a}, {b}) is not under archetypes.cross_venue: {listed}")
    if arch.name == "basis_trade" and pair not in carry:
        listed = ", ".join(f"({x}, {y})" for x, y in spec.carry) or "none"
        raise ContractError(f"pair ({a}, {b}) is not under archetypes.carry: {listed}")


def _bind_family(item: Feature | Strategy, brief) -> Binding:
    """The mechanism check: the family is the brief's, and so are the legs."""
    family = brief.family(item.family)
    if family is None:
        raise ContractError(f"family {item.family} is not in the brief")
    if family.name == "basket":
        if item.pair is not None or (item.target if isinstance(item, Feature) else item.trades):
            raise ContractError("a basket takes every instrument of the brief; give it no legs")
        return Binding("basket", family.instruments)
    if family.name == "pairs":
        if item.pair is None or len(item.pair) != 2:
            raise ContractError("the pairs family needs pair=(A, B), one of the brief's pairs")
        a, b = item.pair
        if (a, b) in family.pairs:
            return Binding("pairs", (a, b))
        if (b, a) in family.pairs:
            return Binding("pairs", (b, a), -1.0)
        listed = ", ".join(f"({x}, {y})" for x, y in family.pairs)
        raise ContractError(f"pair ({a}, {b}) is not one of the brief's pairs: {listed}")
    if item.pair is not None:
        raise ContractError(f"pair= is for the pairs family, not {family.name}")
    declared = item.target if isinstance(item, Feature) else item.trades
    instrument = declared or brief.target.instrument
    if instrument not in family.targets:
        role = "followers" if family.name == "lead_lag" else "instruments"
        raise ContractError(
            f"{instrument} is not one of the {family.name} family's {role}: "
            + ", ".join(family.targets)
        )
    return Binding(family.name, (instrument,))


def output_rule(kind: str, legs: tuple[str, ...], family: str | None = None,
                template: str | None = None) -> str:  # fmt: skip
    """What a feature or strategy with these legs must return, for an error message.

    A strategy bound to the ``maker`` template returns quotes, any other a position.
    """
    if kind == "strategy" and template == "maker":
        return ("a market-making strategy returns bl.quote(...), a mapping with the keys bid, "
                "ask, size, skew, max_inventory and, optionally, lean and fair")  # fmt: skip
    if kind == "strategy" and template == "arb":
        return ("an arbitrage strategy returns bl.quote(...), a mapping with the keys bid, ask, "
                "size, skew, max_inventory and fair, a price off the hedge venue, or "
                "fair=(bid_fair, ask_fair), one for each side")  # fmt: skip
    if family == "basket":
        keys = ", ".join(legs)
        if kind == "feature":
            return f"a basket feature returns a mapping with one value per leg, {keys}, only"
        return f"a basket strategy returns a mapping with a position for each leg, {keys}, only"
    if kind == "feature":
        return "a feature returns one value per grid point, not a mapping"
    if len(legs) > 1:
        keys = ", ".join(legs)
        return f"a pairs strategy returns a mapping with a position for each leg, {keys}, only"
    return f"a strategy trading {legs[0]} returns one position, or a mapping with that one key"


@dataclass
class Module:
    """A loaded feature file."""

    path: Path
    source: str
    features: list[Feature] = field(default_factory=list)
    strategies: list[Strategy] = field(default_factory=list)


def load(path: str | Path) -> Module:
    """Import a feature file and collect the features and strategies it defines."""
    path = Path(path).resolve()
    name = f"_blinded_user_{next(_module_ids)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    found = Module(path, path.read_text())
    for value in vars(module).values():
        if isinstance(value, Feature):
            found.features.append(value)
        elif isinstance(value, Strategy):
            found.strategies.append(value)
    return found


@dataclass
class Scan:
    """Result of the static pre-check: hard errors and advisory warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def scan(source: str, lines: range | None = None, protected: bool = False) -> Scan:
    """Static pre-check of a feature file's source.

    Rejects I/O, process, network and clock access anywhere in the file. Warns
    about constructs that usually mean look-ahead, in pandas or polars, only
    within ``lines`` if given: negative shifts, centred windows, forward and
    nearest as-of joins, backward fills, reversed cumulative sums,
    interpolation, and a whole-column aggregate such as ``x.mean()`` combined
    with ``x`` row by row. The truncation test is what decides.

    With ``protected``, as in a run directory, it also rejects imports outside
    :data:`ALLOWED_IMPORTS`; calls to ``open``, ``exec``, ``eval``,
    ``__import__``, ``getattr`` and the like; any library call that reads or
    writes files (``read_*``, ``scan_*``, ``np.load``, ``tofile``, ...); and
    dunder attributes such as ``__builtins__`` or ``__subclasses__``.
    """
    result = Scan()
    tree = ast.parse(source)
    if protected:
        _protected(tree, result)
    for node in ast.walk(tree):
        warn = lines is None or getattr(node, "lineno", None) in lines
        if isinstance(node, ast.Import | ast.ImportFrom):
            for module in _imported(node):
                root = module.split(".")[0]
                if protected and root not in ALLOWED_IMPORTS:
                    continue  # already reported by _protected
                if root in _FORBIDDEN_MODULES:
                    result.errors.append(f"line {node.lineno}: imports {module}")
        elif isinstance(node, ast.Call):
            _check_call(node, result, warn=warn)
        if warn:
            _warn_broadcast(node, result)
    return result


def _protected(tree: ast.AST, result: Scan) -> None:
    for node in ast.walk(tree):
        line = f"line {getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import | ast.ImportFrom):
            for module in _imported(node):
                if module.split(".")[0] not in ALLOWED_IMPORTS:
                    result.errors.append(f"{line}: imports {module}, which is not allowed here")
                    continue
                # what an import brings in is held to the attribute rules: `from polars
                # import read_parquet` reads files as `pl.read_parquet` does
                names = module.split(".")[1:]
                if isinstance(node, ast.ImportFrom):
                    names += [alias.name for alias in node.names]
                for name in names:
                    why = _refused_name(name)
                    if why:
                        result.errors.append(f"{line}: imports {name}, which {why}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _PROTECTED_CALLS - _FORBIDDEN_CALLS:
                result.errors.append(f"{line}: calls {node.func.id}()")
        elif isinstance(node, ast.Attribute):
            name = node.attr
            if name.startswith("__") and name.endswith("__") and name not in _DUNDER_OK:
                result.errors.append(f"{line}: uses {name}")
            elif _refused_name(name):
                result.errors.append(f"{line}: .{name} {_refused_name(name)}")
        elif isinstance(node, ast.Name) and node.id == "__builtins__":
            result.errors.append(f"{line}: uses __builtins__")


def _refused_name(name: str) -> str | None:
    """Why protected mode refuses an attribute, or a name imported, called ``name``; None
    when it does not."""
    if name.startswith("__") and name.endswith("__"):
        return None if name in _DUNDER_OK else "is a dunder name"
    if name.startswith("_"):
        return "is a library's private name"
    if name in _FILE_METHODS or name.startswith(("read_", "scan_")):
        return "reads or writes files"
    if name.startswith(_WRITE_LIKE) and name not in SAFE_CALLS:
        return "can write data"
    if name in _MODULE_ATTRS:
        return "reaches a module that is not allowed here"
    return None


def scan_feature(feat: Feature) -> Scan | None:
    """Scan the file that defines ``feat``; None when there is no file (a REPL)."""
    try:
        source = Path(inspect.getsourcefile(feat.fn)).read_text()
        body, first = inspect.getsourcelines(feat.fn)
    except (TypeError, OSError):
        return None
    return scan(source, range(first, first + len(body)))


def run(fn: Callable[[Panel], Output], panel: Panel) -> np.ndarray | Legs:
    """Call a feature and return its values on the grid as float64, nulls as NaN."""
    return values(fn(panel), panel)


def values(out: Output, panel: Panel, legs: tuple[str, ...] | None = None,
           template: str | None = None) -> np.ndarray | Legs:  # fmt: skip
    """A feature's output as one float64 per grid point, or a strategy's as :class:`Legs`.

    An expression is evaluated as ``panel.frame.lazy().select(expr)``; a polars
    Series must have one value per grid point; a pandas Series must be
    indexed by the grid's timestamps. A mapping from instrument to any of
    those gives :class:`Legs`. With ``legs``, the output must be exactly those
    legs: a mapping with one key per leg, in which case one leg comes back as
    a plain array, or for one leg a single series.

    :class:`Quotes` come back as :class:`Quotes`, each key evaluated as a
    series is and a number repeated on every point. With ``template``
    ``maker`` the output must be quotes, and with any other template or
    ``legs`` it must not be; either mistake is a :class:`ContractError` with
    :func:`output_rule`.
    """
    maker = template in QUOTING
    if isinstance(out, Quotes):
        if not maker and (legs is not None or template is not None):
            raise ContractError(output_rule("strategy", legs or (), template=template))
        return _quotes(out, panel)
    if maker:
        raise ContractError(output_rule("strategy", legs or (), template=template))
    if isinstance(out, Mapping):
        arrays = out if isinstance(out, Legs) else _arrays(out, panel)
        if legs is None:
            return Legs(arrays)
        if set(arrays) != set(legs):
            raise ContractError(output_rule("strategy", legs))
        return arrays[legs[0]] if len(legs) == 1 else Legs({leg: arrays[leg] for leg in legs})
    if legs is not None and len(legs) > 1:
        raise ContractError(output_rule("strategy", legs))
    return _array(out, panel)


def extend(out: np.ndarray | Legs, n: int) -> np.ndarray | Legs:
    """Values on the first points of a grid, padded with NaN to ``n`` points; :class:`Legs` and
    :class:`Quotes` key by key, keeping their type."""
    if isinstance(out, Legs):
        return type(out)({leg: extend(v, n) for leg, v in out.items()})
    padded = np.full(n, np.nan)
    padded[: len(out)] = out
    return padded


def _arrays(out: Mapping, panel: Panel) -> dict[str, np.ndarray]:
    """Each value of a mapping as :func:`_array` gives it. A cross-section built by the
    ``xs_`` functions or :func:`blinded.basket` (:class:`blinded.features.xs.Cross`) is one
    list expression, evaluated once and split by leg; any other mapping leg by leg."""
    from blinded.features.xs import Cross

    if not isinstance(out, Cross):
        return {str(k): _array(v, panel) for k, v in out.items()}
    values = panel.frame.lazy().select(out.values.alias("v")).collect()
    split = values.select(pl.col("v").list.get(i, null_on_oob=True).cast(pl.Float64).alias(str(k))
                          for i, k in enumerate(out))  # fmt: skip
    return {str(k): split[str(k)].to_numpy() for k in out}


def _quotes(out: Quotes, panel: Panel) -> Quotes:
    """Each key of a :class:`Quotes` as one float64 per grid point: a number or None repeated,
    an array of the grid's length kept, and the expressions evaluated in one pass."""
    unknown = sorted(set(out) - set(QUOTE_KEYS))
    missing = [k for k in QUOTE_KEYS[:5] if k not in out]
    if unknown or missing:
        raise ContractError(output_rule("strategy", (), template="maker"))
    n = len(panel)
    arrays, exprs = {}, {}
    for key, v in out.items():
        if v is None or isinstance(v, int | float) and not isinstance(v, bool):
            arrays[key] = np.full(n, np.nan if v is None else float(v))
        elif isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n:
            arrays[key] = v.astype(np.float64)
        elif isinstance(v, pl.Expr):
            exprs[key] = v.cast(pl.Float64).alias(key)
        else:
            arrays[key] = _array(v, panel)
    if exprs:
        # with ts beside them, a literal is repeated on every point rather than given once
        frame = panel.frame.lazy().select("ts", *exprs.values()).collect()
        arrays |= {key: frame[key].to_numpy() for key in exprs}
    return Quotes({key: arrays[key] for key in out})


def _array(out: Series, panel: Panel) -> np.ndarray:
    if isinstance(out, pl.Expr):
        out = panel.frame.lazy().select(out.alias("feature")).collect().to_series()
    if isinstance(out, pl.Series):
        if len(out) != len(panel):
            raise ValueError(
                f"a feature must give one value per grid point ({len(panel)}); got {len(out)}"
            )
        return out.cast(pl.Float64).to_numpy()
    if isinstance(out, pd.Series):
        if not _on_grid(out.index, panel):
            raise ValueError(
                f"a pandas Series must be indexed by the grid ({len(panel)} points, as in "
                f"panel.to_pandas()); got {len(out)} points"
            )
        return out.to_numpy(dtype=np.float64, na_value=np.nan)
    raise TypeError(
        "a feature must return a polars expression or Series, or a pandas Series; "
        f"got {type(out).__name__}"
    )


def _on_grid(index, panel: Panel) -> bool:
    if not isinstance(index, pd.DatetimeIndex) or len(index) != len(panel):
        return False
    if index.tz is None:
        index = index.tz_localize(panel.tz)
    return np.array_equal(index.as_unit("ns").asi8, panel.grid_ns)


def _imported(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    return [alias.name for alias in node.names]


def _check_call(node: ast.Call, result: Scan, warn: bool) -> None:
    func = node.func
    line = f"line {node.lineno}"
    if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALLS:
        result.errors.append(f"{line}: calls {func.id}()")
    if not isinstance(func, ast.Attribute):
        return
    owner = func.value.id if isinstance(func.value, ast.Name) else None
    if func.attr.startswith(_WRITE_LIKE) and func.attr not in SAFE_CALLS:
        result.errors.append(f"{line}: .{func.attr}() can write data; of the calls named to_*, "
                             "write*, save* and dump* only conversions such as .to_numpy() "
                             "are allowed")  # fmt: skip
    if (owner, func.attr) in _CLOCK_READS:
        result.errors.append(f"{line}: reads the clock with {owner}.{func.attr}()")
    if warn:
        _warn_lookahead(node, func, line, result)


def _warn_lookahead(node: ast.Call, func: ast.Attribute, line: str, result: Scan) -> None:
    warn = result.warnings.append
    if func.attr == "merge_asof" and _keyword(node, "direction") in ("forward", "nearest"):
        warn(f"{line}: merge_asof(direction={_keyword(node, 'direction')!r})")
    if func.attr == "join_asof" and _keyword(node, "strategy") in ("forward", "nearest"):
        warn(f"{line}: join_asof(strategy={_keyword(node, 'strategy')!r}) takes later rows")
    if func.attr == "shift" and _negative_literal(node):
        warn(f"{line}: shift() by a negative amount reads the future")
    if _keyword(node, "center") is True:
        warn(f"{line}: center=True puts future points in the window")
    backward = func.attr == "fill_null" and _keyword(node, "strategy") == "backward"
    backward |= func.attr == "fillna" and _keyword(node, "method") in ("bfill", "backfill")
    if backward or func.attr in ("bfill", "backfill", "backward_fill"):
        warn(f"{line}: {func.attr}() fills backwards, from later values")
    if func.attr.startswith("cum") and _keyword(node, "reverse") is True:
        warn(f"{line}: {func.attr}(reverse=True) accumulates from the end")
    if func.attr in ("interpolate", "interpolate_by"):
        warn(f"{line}: {func.attr}() fills a gap using the next value")


def _warn_broadcast(node: ast.AST, result: Scan) -> None:
    """Warn about ``x - x.mean()`` and the like: one number from the whole column on every row."""
    if isinstance(node, ast.BinOp | ast.Compare):
        operands = [node.left, node.right] if isinstance(node, ast.BinOp) else [node.left]
        operands += node.comparators if isinstance(node, ast.Compare) else []
    elif isinstance(node, ast.Call) and _method(node) in _ROW_WISE:
        operands = [*node.args, *(kw.value for kw in node.keywords)]
    else:
        return
    for operand in operands:
        while isinstance(operand, ast.UnaryOp):
            operand = operand.operand
        agg = _whole_column(operand)
        if agg is not None:
            result.warnings.append(
                f"line {operand.lineno}: .{agg}() of the whole column is broadcast onto every row"
            )


def _whole_column(node: ast.AST) -> str | None:
    """The aggregation's name when ``node`` is ``x.mean()``, ``np.std(x)`` or ``pl.max(...)``."""
    if _method(node) not in _AGGREGATES:
        return None
    receiver = node.func.value
    while isinstance(receiver, ast.Call | ast.Attribute | ast.Subscript):
        if _method(receiver) in _WINDOWED:
            return None
        receiver = receiver.func if isinstance(receiver, ast.Call) else receiver.value
    return node.func.attr


def _method(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _keyword(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def _negative_literal(node: ast.Call) -> bool:
    args = list(node.args) + [kw.value for kw in node.keywords if kw.arg in ("periods", "n")]
    for arg in args[:1]:
        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            return True
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int | float) and arg.value < 0:
            return True
    return False
