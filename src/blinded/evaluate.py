"""Judge one feature: static scan, determinism, truncation test, then metrics."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from blinded import brief as briefs
from blinded import metrics, sim, truncation
from blinded.brief import Brief, Window
from blinded.contract import (
    ContractError,
    Feature,
    Legs,
    Strategy,
    bind,
    extend,
    load,
    output_rule,
    run,
    scan_feature,
    values,
)
from blinded.durations import label
from blinded.panel import Panel
from blinded.target import feature_returns


@dataclass
class Result:
    """What the judge says about one feature.

    ``status`` is one of ``accepted``, ``rejected_static``,
    ``rejected_nondeterministic``, ``rejected_lookahead`` or ``error``.
    ``legs`` is the instrument whose return was scored, or the pair whose
    spread return was, in the brief's order. ``archetype`` and ``edge`` are
    set when an archetype's decorator registered the feature.
    """

    name: str
    hypothesis: str
    status: str
    reason: str
    window: Window
    latency: pd.Timedelta
    family: str = "directional"
    legs: tuple[str, ...] = ()
    archetype: str | None = None
    edge: str | None = None
    verdict: truncation.Verdict | None = None
    horizons: pl.DataFrame = field(default_factory=pl.DataFrame)
    decay: pl.DataFrame = field(default_factory=pl.DataFrame)
    stability: pl.DataFrame = field(default_factory=pl.DataFrame)
    turnover: float = np.nan
    autocorr: float = np.nan
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "accepted"

    def to_frame(self) -> pl.DataFrame:
        """Metrics by horizon, one row per horizon in the brief."""
        return self.horizons.clone()

    def to_pandas(self) -> pd.DataFrame:
        """The same table as a pandas DataFrame indexed by horizon label."""
        if self.horizons.is_empty():
            return pd.DataFrame()
        return self.horizons.to_pandas().set_index("horizon")

    def __repr__(self) -> str:
        from blinded.render import format_result

        return format_result(self)


@dataclass
class StrategyResult:
    """What the judge says about one strategy: the truncation verdict, then the sim rows.

    ``rows`` has one row each for the brief's latency and fee, the three
    unrealistic diagnostics (latency 0, fees 0, both) and the maker proxy, and
    where the file has a touch stream the maker-taker row
    (:func:`blinded.sim.report`).
    ``position`` is :class:`~blinded.contract.Legs` for a strategy in two legs.
    ``archetype`` and ``edge`` are set when an archetype's decorator
    registered the strategy.
    """

    name: str
    requires: list[str]
    status: str
    reason: str
    window: Window
    latency: pd.Timedelta
    family: str | None = "directional"
    legs: tuple[str, ...] = ()
    archetype: str | None = None
    edge: str | None = None
    verdict: truncation.Verdict | None = None
    rows: pl.DataFrame = field(default_factory=pl.DataFrame)
    position: np.ndarray | Legs | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "accepted"

    def row(self, name: str = "brief") -> dict:
        """One row of the sim table as a dict."""
        return self.rows.filter(pl.col("row") == name).to_dicts()[0]

    def to_frame(self) -> pl.DataFrame:
        return self.rows.clone()

    def to_pandas(self) -> pd.DataFrame:
        if self.rows.is_empty():
            return pd.DataFrame()
        return self.rows.to_pandas().set_index("row")

    def __repr__(self) -> str:
        from blinded.render import format_strategy

        return format_strategy(self)


def evaluate(
    feat: Feature,
    brief: Brief | str | Path,
    window: str = "train",
    latency_ms: float | None = None,
    panel: Panel | None = None,
) -> Result:
    """Run the judge on one feature.

    Parameters
    ----------
    feat
        A function registered with :func:`blinded.feature`.
    brief
        A :class:`~blinded.brief.Brief` or a path to ``brief.yaml``.
    window
        ``train``, ``val``, ``test``, ``all`` or ``START/END``.
    latency_ms
        Overrides ``execution.latency_ms`` from the brief.
    panel
        A prebuilt full panel, to avoid reloading the data.
    """
    if not isinstance(brief, Brief):
        brief = briefs.load(brief)
    full = panel if panel is not None else panel_for(brief)
    win = scoring_window(brief, full, window)
    latency = brief.latency if latency_ms is None else pd.Timedelta(milliseconds=latency_ms)
    static = scan_feature(feat)
    base = dict(
        name=feat.name,
        hypothesis=feat.hypothesis,
        window=win,
        latency=latency,
        family=feat.family,
        legs=feat.legs,
        **_tags(feat),
        warnings=(static.warnings if static else ["no source file; static scan skipped"])
        + _lattice_warning(brief, latency),
    )
    try:
        bound = bind(feat, brief, _menu(brief, full))
    except ContractError as exc:
        return Result(status="error", reason=str(exc), **base)
    base["legs"] = bound.legs
    if static and static.errors:
        return Result(status="rejected_static", reason="; ".join(static.errors), **base)

    short = full.cut(win.end)
    try:
        verdict = truncation.check(feat.fn, short, win, brief.stats.seed, tail=brief.stats.block)
    except Exception as exc:
        reason = describe_error(exc, inspect.getsourcefile(feat.fn))
        return Result(status="error", reason=reason, **base)
    if not verdict.ok:
        reason = str(verdict.violation)
        return Result(status=verdict.status, reason=reason, verdict=verdict, **base)
    basket = bound.family == "basket"
    keys = set(verdict.full) if isinstance(verdict.full, Legs) else None
    if (keys is not None) != basket or (basket and keys != set(bound.legs)):
        return Result(status="error", reason=output_rule("feature", bound.legs, bound.family),
                      **base)  # fmt: skip

    t = brief.target
    targets = feature_returns(
        full, bound.legs, t.price, _all_horizons(brief), latency, bound.template
    )
    scores = score_for(bound, verdict.full, short.frame["ts"], targets, win, brief)
    return Result(status="accepted", reason=str(verdict), verdict=verdict, **scores, **base)


def evaluate_strategy(
    strat: Strategy,
    brief: Brief | str | Path,
    window: str = "train",
    latency_ms: float | None = None,
    panel: Panel | None = None,
    features: dict[str, Feature] | None = None,
) -> StrategyResult:
    """Run the judge on one strategy: static scan, truncation test, then the sim.

    The truncation test runs on the composed function ``panel -> position``:
    the required features are computed on the (possibly cut) panel and handed
    to the strategy, so a leak in either is caught.

    Parameters
    ----------
    strat
        A function registered with :func:`blinded.strategy`.
    features
        The features named in ``requires``; found with :func:`find_features`
        when not given.
    """
    if not isinstance(brief, Brief):
        brief = briefs.load(brief)
    full = panel if panel is not None else panel_for(brief)
    win = scoring_window(brief, full, window)
    if latency_ms is not None:
        latency = pd.Timedelta(milliseconds=latency_ms)
        # a cancel or a response no slower than the order it follows
        slow = {k: min(v, latency) for k in ("cancel_latency", "response_latency")
                if (v := getattr(brief.execution, k)) is not None}  # fmt: skip
        ex = dataclasses.replace(brief.execution, latency=latency, **slow)
        brief = dataclasses.replace(brief, latency=latency, execution=ex)
    static = scan_feature(strat)
    base = dict(
        name=strat.name,
        requires=strat.requires,
        window=win,
        latency=brief.latency,
        family=strat.family,
        legs=strat.legs,
        **_tags(strat),
        warnings=(static.warnings if static else ["no source file; static scan skipped"])
        + _lattice_warning(brief, brief.latency),
    )
    try:
        bound = bind(strat, brief, _menu(brief, full))
    except ContractError as exc:
        return StrategyResult(status="error", reason=str(exc), **base)
    base["legs"] = bound.legs
    if static and static.errors:
        return StrategyResult(status="rejected_static", reason="; ".join(static.errors), **base)
    try:
        features = find_features(strat) if features is None else features
    except KeyError as exc:
        return StrategyResult(status="error", reason=str(exc.args[0]), **base)

    short = full.cut(win.end)
    composed = positions(strat, features, bound.legs, bound.template)
    try:
        verdict = truncation.check(composed, short, win, brief.stats.seed, tail=brief.stats.block)
    except ContractError:
        reason = output_rule("strategy", bound.legs, bound.family, bound.template)
        return StrategyResult(status="error", reason=reason, **base)
    except Exception as exc:
        reason = describe_error(exc, inspect.getsourcefile(strat.fn))
        return StrategyResult(status="error", reason=reason, **base)
    if not verdict.ok:
        reason = str(verdict.violation)
        return StrategyResult(status=verdict.status, reason=reason, verdict=verdict, **base)
    position = extend(verdict.full, len(full))
    rows = sim.report(position, full, win, brief, bound.legs[0], hedge=bound.hedge)
    return StrategyResult(
        status="accepted",
        reason=str(verdict),
        verdict=verdict,
        rows=rows,
        position=position,
        **base,
    )


def positions(
    strat: Strategy,
    features: dict[str, Feature],
    legs: tuple[str, ...] | None = None,
    template: str | None = None,
) -> Callable[[Panel], pl.Series | Legs]:
    """The composed function ``panel -> position`` that the truncation test runs.

    Each required feature is computed on the panel it is given and added to
    the frame as ``feature:<name>``; the strategy receives ``{name:
    pl.col("feature:<name>")}``; a basket feature, one value per leg, gives
    ``{name: {leg: pl.col("feature:<name>:<leg>")}}``. With ``legs``, from
    :func:`~blinded.contract.bind`, the strategy must return exactly those
    legs, and a strategy in two legs gives :class:`~blinded.contract.Legs`.
    With ``template`` ``maker`` (a market-making strategy) it must return
    :class:`~blinded.contract.Quotes` instead, and gives them.
    """

    def composed(panel: Panel) -> pl.Series | Legs:
        columns, exprs = {}, {}
        for n, f in features.items():
            v = run(f.fn, panel)
            if isinstance(v, Legs):
                columns |= {f"feature:{n}:{leg}": x for leg, x in v.items()}
                exprs[n] = {leg: pl.col(f"feature:{n}:{leg}") for leg in v}
            else:
                columns[f"feature:{n}"] = v
                exprs[n] = pl.col(f"feature:{n}")
        wide = panel.with_columns(columns)
        out = values(strat.fn(wide, exprs), wide, legs, template)
        return out if isinstance(out, Legs) else pl.Series("position", out)

    return composed


def find_features(strat: Strategy) -> dict[str, Feature]:
    """The features a strategy requires, by name.

    Looked up in the strategy's own file first, then in every ``.py`` file of
    a ``features/`` directory beside the strategy's directory (the workspace
    layout: ``strategies/s.py`` next to ``features/``).
    """
    found = {v.name: v for v in strat.fn.__globals__.values() if isinstance(v, Feature)}
    missing = [n for n in strat.requires if n not in found]
    source = inspect.getsourcefile(strat.fn)
    if missing and source is not None:
        folder = Path(source).resolve().parent.parent / "features"
        for path in sorted(folder.glob("*.py")) if folder.is_dir() else []:
            for feat in load(path).features:
                found.setdefault(feat.name, feat)
    missing = [n for n in strat.requires if n not in found]
    if missing:
        raise KeyError(f"required features not found: {', '.join(missing)}")
    return {n: found[n] for n in strat.requires}


def _tags(item) -> dict:
    from blinded.archetypes import REGISTRY

    name = getattr(item, "archetype", None)
    return {"archetype": name, "edge": REGISTRY[name].edge} if name in REGISTRY else {}


def _menu(brief: Brief, panel: Panel):
    """The archetype menu for a local judgement: from the brief and the panel, as at init."""
    if not brief.has_archetypes:
        return None
    from blinded import archetypes

    return archetypes.menu(brief, archetypes.facts(brief, panel))


def scoring_window(brief: Brief, panel: Panel, spec: str) -> Window:
    """Resolve a window spec and drop the grid warm-up from its start."""
    win = briefs.window(spec, brief, panel.time(0), panel.time(-1))
    start = max(win.start, panel.time(0) + brief.grid.warmup)
    return Window(win.label, start, win.end)


def score(
    signal: np.ndarray, ts: pl.Series, targets: pl.DataFrame, win: Window, brief: Brief
) -> dict:
    """Metrics of a signal against forward returns inside the scoring window.

    ``signal`` and ``ts`` cover a prefix of the grid ``targets`` was built on.
    """
    g = ts.dt.epoch("ns").to_numpy()
    n = len(g)
    inside = (g >= win.start.value) & (g < win.end.value)
    step = brief.grid.step
    block = max(int(brief.stats.block / step), 1)

    def target(h: pd.Timedelta) -> np.ndarray:
        return targets[label(h)].to_numpy()[:n]

    rows = []
    for h in brief.target.horizons:
        x, y = _paired(signal, target(h), inside)
        row = metrics.ic_summary(x, y, h / step, block, brief.stats.n_boot, brief.stats.seed)
        row.update(metrics.markout(x, y))
        rows.append({"horizon": label(h), **_flatten(row)})
    decay = []
    for h in _all_horizons(brief):
        x, y = _paired(signal, target(h), inside)
        decay.append(
            {
                "horizon": label(h),
                "n": len(x),
                "ic": metrics.correlation(x, y)[0],
                "rank_ic": metrics.correlation(metrics.rank(x), metrics.rank(y))[0],
            }
        )
    live = signal[inside & np.isfinite(signal)]
    primary = target(brief.target.primary_horizon)
    return {
        "horizons": pl.DataFrame(rows),
        "decay": pl.DataFrame(decay),
        "stability": metrics.stability(
            ts.filter(pl.Series(inside)), signal[inside], primary[inside], min_n=_half(step)
        ),
        "turnover": metrics.turnover(live),
        "autocorr": metrics.autocorr(live),
    }


def score_for(bound, signal, ts: pl.Series, targets, win: Window, brief: Brief) -> dict:
    """:func:`score` of a feature in its family: :func:`score_xs` for a basket's, else
    :func:`score` of the signal times the binding's sign."""
    if bound.family == "basket":
        return score_xs(signal, ts, targets, win, brief)
    return score(bound.sign * signal, ts, targets, win, brief)


def score_xs(signal: Legs, ts: pl.Series, targets: dict[str, pl.DataFrame], win: Window,
             brief: Brief) -> dict:  # fmt: skip
    r"""Metrics of a basket feature against its legs' forward returns inside the window.

    The same keys as :func:`score`, from the cross-sectional rank IC
    :math:`c_t` of :func:`blinded.metrics.xs_rank_ic` at each point: ``ic``
    and ``rank_ic`` are the mean of :math:`c_t` (there is no Pearson
    version), its Newey-West error uses the series of :math:`c_t` with the
    horizon's lags and its bootstrap resamples blocks of it, ``n`` counts the
    finite :math:`c_t`, and the markouts are those of a basket long the top
    fifth and short the bottom fifth (:func:`blinded.metrics.xs_markout`).
    Decay and IC per window follow from :math:`c_t` at each horizon;
    ``turnover`` and ``autocorr`` are the means over the legs of each leg's
    own.
    """
    g = ts.dt.epoch("ns").to_numpy()
    n = len(g)
    inside = (g >= win.start.value) & (g < win.end.value)
    step = brief.grid.step
    block = max(int(brief.stats.block / step), 1)
    legs = list(targets)
    x = np.column_stack([np.asarray(signal[leg], dtype=float)[:n] for leg in legs])

    def target(h: pd.Timedelta) -> np.ndarray:
        return np.column_stack([targets[leg][label(h)].to_numpy()[:n] for leg in legs])

    rows = []
    for h in brief.target.horizons:
        r = target(h)
        c = metrics.xs_rank_ic(x[inside], r[inside])
        row = metrics.xs_ic_summary(c, h / step, block, brief.stats.n_boot, brief.stats.seed)
        row.update(metrics.xs_markout(x[inside], r[inside]))
        rows.append({"horizon": label(h), **_flatten(row)})
    decay = []
    for h in _all_horizons(brief):
        c = metrics.xs_rank_ic(x[inside], target(h)[inside])
        ok = np.isfinite(c)
        mean = float(np.mean(c[ok])) if ok.any() else np.nan
        decay.append({"horizon": label(h), "n": int(ok.sum()), "ic": mean, "rank_ic": mean})
    primary = metrics.xs_rank_ic(x[inside], target(brief.target.primary_horizon)[inside])
    half = _half(step)
    live = [col[np.isfinite(col)] for col in x[inside].T]
    return {
        "horizons": pl.DataFrame(rows),
        "decay": pl.DataFrame(decay),
        "stability": metrics.xs_stability(ts.filter(pl.Series(inside)), primary, min_n=half),
        "turnover": _leg_mean(metrics.turnover, live),
        "autocorr": _leg_mean(metrics.autocorr, live),
    }


def _leg_mean(fn, legs: list[np.ndarray]) -> float:
    vals = [fn(v) for v in legs if len(v) > 2]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


@functools.lru_cache(maxsize=4)
def _cached_panel(path: str, brief_mtime: float, data_mtime: float) -> Panel:
    return Panel.from_brief(briefs.load(path))


def panel_for(brief: Brief) -> Panel:
    """The full panel for a brief, cached while neither the brief nor the data file changes."""
    if brief.source is None:
        return Panel.from_brief(brief)
    mtimes = (Path(brief.source).stat().st_mtime, Path(brief.data.path).stat().st_mtime)
    return _cached_panel(str(brief.source), *mtimes)


def describe_error(exc: Exception, source: str | None = None) -> str:
    """Exception type, message and the innermost line in ``source`` (the feature's file)."""
    frames = traceback.extract_tb(exc.__traceback__)
    mine = [f for f in frames if source is not None and Path(f.filename) == Path(source)]
    where = f" at {Path(mine[-1].filename).name}:{mine[-1].lineno}" if mine else ""
    return f"{type(exc).__name__}{where}: {exc}"


def _half(step: pd.Timedelta, every: str = "30min") -> int:
    return int(pd.Timedelta(every) / step / 2)


def _all_horizons(brief: Brief) -> list[pd.Timedelta]:
    """The brief's horizons and the decay ladder's, less those whose returns end at a stamp a
    thinned file does not keep, and on bars those shorter than a bar."""
    ladder = {h for h in metrics.DECAY_LADDER if not brief.grid.bars or h >= brief.grid.step}
    spans = set(brief.target.horizons) | ladder
    return sorted(h for h in spans if brief.data.exact(brief.latency + h))


def _lattice_warning(brief: Brief, latency: pd.Timedelta) -> list[str]:
    if brief.data.exact(latency):
        return []
    return [
        f"latency {label(latency)} is off the stamps this thinned file keeps: fills and "
        "returns read the last kept row before them, so the numbers are approximate"
    ]


def _paired(f: np.ndarray, r: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = inside & np.isfinite(f) & np.isfinite(r)
    return f[ok], r[ok]


def _flatten(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, tuple):
            out[f"{key}_lo"], out[f"{key}_hi"] = value
        else:
            out[key] = value
    return out


evaluate_feature = evaluate  # alias: the package binds `evaluate` as a name too
