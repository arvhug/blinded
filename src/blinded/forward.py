"""Forward paper: the released strategy judged on days recorded after it was frozen.

``blinded forward RUN_DIR --live-data DIR`` takes a run whose test has been scored and
judges its final, unchanged, on the days :mod:`blinded.recorder` wrote under ``DIR``.
Nothing is refitted and nothing is chosen: the vault's
brief, judging rules, theo parameters and the final's code and features, a forward brief a
day that differs from the vault's only in where the data is, the day's grid start, the
latencies and one window, ``forward``, from 00:05 to the day's end.

The rows did not exist when the strategy was released, so no choice of day, distance or
size can have read them; the fills are still the engine's, so forward paper says whether
the market still pays what the locked test said it paid, not whether a print would have
reached a real order.

How a day is judged. The panel is built from the day's rows and two hours of the day
before (:data:`WARMUP`) up to the last closed hour plus the five minutes the hedges and
markouts read. The quotes are the final's code evaluated in a fenced worker
(:mod:`blinded.isolate`), the daemon's pool and fence: without ``srt`` this refuses, as the
daemon does, unless ``BLINDED_UNFENCED=1``. The pass is the judge's: both queue models
through :func:`blinded.sim_maker.arb_models`, the worse scored, the stock carried through
the day by one continuous pass, re-run from 00:05 each hour, so a day's numbers are one
pass. Its per-step net summed by hour, and its fills counted by hour, are the hourly
series.

An hour is judged only when the recorder covered all of it with no gap and its clock
under 250 ms (:func:`blinded.recorder.hour_statuses`); any other hour is printed hatched
with its reason and adds nothing to any statistic. The verdict is a one-sided CUSUM on the
hourly net and on the hourly fill count against the locked test's own hours
(:func:`cusum`), with ``k = 0.5`` and each threshold set so that an unchanged strategy
alarms in 10% of weeks once the error of a reference of so few hours is counted
(:func:`threshold`).

Everything is written under ``vault/forward/``, which the hook and the OS sandbox deny to
an agent: the forward briefs, the theo of each day, ``ledger.sqlite`` (one row a judged hour
and a day row, appended, never rewritten) and ``forward.html``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from blinded import archetypes, contract, isolate, recorder, sim_maker
from blinded import brief as briefs
from blinded.ledger import code_hash
from blinded.panel import Panel
from blinded.vault import RunDir, Window, sha256

WARMUP = pd.Timedelta("2h")  # rows of the day before a day's panel starts from
K, H = 0.5, 5.0  # the CUSUM's k, and the h it would take were the reference's m and s known
RATE = 0.10  # the share of unchanged weeks a threshold lets alarm (:func:`threshold`)
LEVEL = 0.90  # the forward hours' own interval (:func:`own_interval`), as the locked test's
MAX_EXCLUDED = 4  # a day with more excluded hours gets no day row
HOUR_NS = recorder.HOUR_NS
PORT = 8766


class ForwardError(RuntimeError):
    """A run forward paper cannot judge, or a day it cannot read."""


def forward_dir(run: RunDir) -> Path:
    return run.vault / "forward"


# -- the final -------------------------------------------------------------------------------


@dataclass
class Final:
    """The run's final strategy as the vault holds it: its trial, its file among the vault's
    trial copies, the accepted features' files, and the hash of its source."""

    trial_id: int
    name: str
    code_path: str
    accepted: dict[str, str]
    source_sha: str
    archetype: str | None = None


def _vault_copy(run: RunDir, path: str) -> str:
    """The run's own copy of a trial file: the vault keeps every submission under its
    content's hash, which is checked; the ledger's path may name a directory the run was
    first written in."""
    copy = run.trials / Path(path).name
    if not copy.is_file():
        raise ForwardError(f"{copy} is missing: the vault keeps every trial's file")
    if code_hash(copy.read_text()) != copy.stem:
        raise ForwardError(f"{copy.name} does not hash to its name")
    return str(copy)


def final_of(run: RunDir) -> Final:
    """The scored final; refuses a run whose locked test is not in."""
    ledger = run.ledger()
    try:
        scored = ledger.test_score(run.run_id)
        if scored is None:
            raise ForwardError("the run's test has not been scored: forward paper judges a "
                               "final whose locked test is in")  # fmt: skip
        row = ledger.trial(scored["trial_id"])
        accepted = {r["name"]: _vault_copy(run, r["code_path"])
                    for r in ledger.trials("feature") if r["status"] == "accepted"}  # fmt: skip
    finally:
        ledger.close()
    path = _vault_copy(run, row["code_path"])
    source = Path(path).read_text()
    if row["code_hash"] and code_hash(source, row["name"]) != row["code_hash"]:
        raise ForwardError(f"{Path(path).name} is not the source trial {row['trial_id']} was "
                           "judged on")  # fmt: skip
    return Final(int(row["trial_id"]), row["name"], path, accepted,
                 sha256(path), row["archetype"])  # fmt: skip


# -- a day -----------------------------------------------------------------------------------


@dataclass
class Day:
    """What one judgement reads: the day's start and end, its own files and the day
    before's (warm-up), each file list in time order, and each hour's status."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    data: tuple[Path, ...]
    touch: tuple[Path, ...]
    depth: tuple[Path, ...]
    theo: tuple[Path, ...] | None = None
    statuses: dict[int, tuple[str, str]] = field(default_factory=dict)
    warm: bool = True  # rows of the day before are in
    upto: pd.Timestamp | None = None  # the last closed hour's end, for an open day
    hashes: dict = field(default_factory=dict)
    clock: dict = field(default_factory=dict)
    host: str | None = None

    @property
    def closed(self) -> bool:
        return self.upto is None or self.upto >= self.end


def segment_day(run: RunDir, whole: bool = False) -> Day:
    """The vault's own test segment as a day: from the end of validation, its rows and two
    hours of validation's before it, every hour judged. ``blinded forward --segment test``
    judges it; at the vault's latencies it must give the locked test's numbers.

    ``whole``: the vault's whole span instead, from the start of train, its first two hours
    a warm-up as a first recorded day's are. Its hours give :func:`reference` the shape of
    an hour's net and fills, never their level, which the locked test alone sets."""
    wins = run.windows()
    m = run.manifest
    start = wins["train"].start if whole else wins["val"].end
    files = {kind: tuple(run.side_path(s, kind) for s in ("train", "val", "test"))
             for kind in ("touch", "depth")}  # fmt: skip
    theo = (tuple(run.side_path(s, "theo") for s in ("train", "val", "test"))
            if m.get("theo") else None)  # fmt: skip
    data = tuple(run.segment_path(s) for s in ("train", "val", "test"))
    hours = range(int(start.value // HOUR_NS), int(-(-wins["test"].end.value // HOUR_NS)))
    return Day("vault" if whole else "test", start, wins["test"].end, data, files["touch"],
               files["depth"], theo, {h * HOUR_NS: ("judged", "") for h in hours},
               warm=not whole,
               hashes={"segments": {s: m["segments"][s]["sha256"] for s in m["segments"]}},
               host="the vault")  # fmt: skip


def recorded_days(live: Path, coin: str = "ATUSDT") -> list[Date]:
    out = []
    for d in sorted(Path(live).iterdir()) if Path(live).is_dir() else []:
        try:
            day = Date.fromisoformat(d.name)
        except ValueError:
            continue
        if recorder.hours_of(d):
            out.append(day)
    return out


def _files_of(live: Path, day: Date, coin: str, scratch: Path) -> dict[str, Path] | None:
    """A recorded day's data, touch and depth files: the built ones, or for a day still open
    the same three built from its parts into ``scratch``."""
    d = Path(live) / day.isoformat()
    built = recorder.day_built(d, coin, day)
    if built is not None:
        return built
    if not any((d / "parts").glob("*.parquet")):
        return None
    out = scratch / day.isoformat()
    b = recorder.build_files(recorder.read_parts(d), coin, day, out, brief=False)
    return {k: b["paths"][k] for k in ("data", "touch", "depth")}


def live_day(live: Path, day: Date, coin: str, scratch: Path, now_ns: int | None = None) -> Day:
    """A recorded day as :class:`Day`: its files and the day before's, its hours' statuses,
    the hashes of what was read, the clock and the recorder's host."""
    now_ns = now_ns or datetime.now(UTC).timestamp() * 1e9
    d = Path(live) / day.isoformat()
    files = _files_of(live, day, coin, scratch)
    if files is None:
        raise ForwardError(f"{d} has no rows")
    prev = _files_of(live, day.fromordinal(day.toordinal() - 1), coin, scratch)
    start = pd.Timestamp(recorder.day_start(day), tz="UTC")
    end = start + pd.Timedelta("1D")
    closed = [line["hour_ns"] + HOUR_NS for line in recorder.hours_of(d)]
    upto = min(pd.Timestamp(max(closed), tz="UTC"), end) if closed else start
    if upto > pd.Timestamp(int(now_ns), tz="UTC"):
        upto = pd.Timestamp(int(now_ns) // HOUR_NS * HOUR_NS, tz="UTC")
    statuses = recorder.hour_statuses(d, day)
    lines = recorder.hours_of(d)
    built = recorder.day_built(d, coin, day) is not None
    hashes = {
        "files": {k: sha256(p) for k, p in files.items()}
        if built
        else {"parts": recorder.hash_parts(d)},
        "data_raw_md": sha256(d / "DATA-raw.md") if (d / "DATA-raw.md").exists() else None,
    }
    clock = recorder.clock_summary(recorder.clock_of(d)) if recorder.clock_of(d) else {}
    pick = (lambda k: (prev[k], files[k])) if prev else (lambda k: (files[k],))
    return Day(day.isoformat(), start, end, pick("data"), pick("touch"), pick("depth"), None,
               statuses, warm=prev is not None, upto=upto, hashes=hashes, clock=clock,
               host=lines[-1].get("host") if lines else None)  # fmt: skip


# -- the forward brief -----------------------------------------------------------------------


def forward_brief(run: RunDir, day: Day, latencies: dict, final: Final,
                  write: bool = True) -> tuple[dict, briefs.Brief]:  # fmt: skip
    """The vault's brief with the day's files, its grid start, ``latencies`` (any of
    ``latency_ms``, ``cancel_latency_ms``, ``response_latency_ms``, ``hedge_latency_ms``)
    and one window, ``forward``, from the brief's warm-up past the day's start to its end;
    everything else byte for byte the vault's, and a ``forward:`` block naming the run,
    the final and the day's hashes. Written to ``vault/forward/<day>.brief.yaml``."""
    raw = yaml.safe_load((run.vault / "brief.yaml").read_text())
    vault = run.brief()
    raw["data"]["path"] = str(day.data[-1])
    if "touch" in raw["data"]:
        raw["data"]["touch"] = dict(raw["data"]["touch"]) | {"path": str(day.touch[-1])}
    if "depth" in raw["data"]:
        raw["data"]["depth"] = dict(raw["data"]["depth"]) | {"path": str(day.depth[-1])}
    if "theo" in raw["data"] and day.theo:
        raw["data"]["theo"] = dict(raw["data"]["theo"]) | {"path": str(day.theo[-1])}
    raw["grid"]["start"] = day.start.isoformat()
    for key, value in latencies.items():
        raw["execution"][key] = value
    begin = day.start + vault.grid.warmup
    raw["splits"]["windows"] = {"forward": f"{begin.isoformat()}/{day.end.isoformat()}"}
    raw["judge_rules"] = run.judge_rules
    raw["forward"] = {"run": run.run_id, "run_dir": str(run.root), "day": day.label,
                      "final": {"trial_id": final.trial_id, "name": final.name,
                                "source_sha256": final.source_sha},
                      "hashes": day.hashes}  # fmt: skip
    brief = briefs.parse(raw)
    if write:
        out = forward_dir(run)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{day.label}.brief.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return raw, brief


def day_theo(run: RunDir, brief: briefs.Brief, frame: pl.DataFrame, label: str) -> Path:
    """The day's theo, filtered over the panel's rows with train's parameters at full rate,
    never refitted, into ``vault/forward/<day>.theo.parquet``."""
    from blinded import theo as theos

    meta = json.loads((run.vault / "theo.json").read_text())
    params = theos.TheoParams.from_dict(meta["params"])
    out = forward_dir(run) / f"{label}.theo.parquet"
    theos.refilter(frame, brief, brief.data.theo, params).write_parquet(out)
    return out


def panel_of(brief: briefs.Brief, day: Day, until: pd.Timestamp) -> Panel:
    """The panel of a judgement: the day's rows from :data:`WARMUP` before its start to
    ``until`` plus the five minutes a pass reads after its last decision, with the touch and
    depth files as the vault's panels carry them."""
    lo = int((day.start - WARMUP).value)
    hi = int(until.value) + sim_maker._AFTER
    at = pl.col("ts").cast(pl.Datetime("ns")).dt.epoch("ns")
    frame = pl.concat([pl.scan_parquet(str(p)).filter((at >= lo) & (at < hi)).collect()
                       for p in day.data])  # fmt: skip
    panel = Panel.from_brief(brief, frame, theo=day.theo)
    spec = brief.data.touch
    if spec is not None:
        panel.touch_source = sim_maker.TouchSource(day.touch, spec.instrument, spec.tick_size)
    depth = brief.data.depth
    if depth is not None:
        panel.depth_source = sim_maker.DepthSource(day.depth, depth.venue, depth.levels)
    return panel


# -- the judgement ---------------------------------------------------------------------------


def fenced_pool(run: RunDir):
    """The daemon's pool: a fresh worker an evaluation, under ``srt`` (refused without it,
    unless ``BLINDED_UNFENCED=1``), reading nothing of the run but its trial files. Each
    start is written to ``vault/forward/fence.jsonl``."""
    from blinded.daemon import DEFAULT_LIMITS
    from blinded.sandbox import protected

    srt = isolate.fence()
    out = forward_dir(run)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fence.jsonl", "a") as fh:
        fh.write(json.dumps({"started": pd.Timestamp.now(tz="UTC").isoformat(),
                             "srt": srt is not None}) + "\n")  # fmt: skip
    pool = isolate.Pool(DEFAULT_LIMITS, None, protected(run), [str(run.trials)])
    return pool, srt is not None


def quotes_of(run: RunDir, final: Final, brief: briefs.Brief, panel: Panel, pool):
    """The final's quotes on ``panel``, computed in a fenced worker and checked again by
    :func:`blinded.contract.values`; this process never imports the code. Returns the
    output and the binding."""
    from blinded.daemon import _required

    declared = [isolate.declared(d) for d in isolate.describe(pool, final.code_path)]
    local = {d.name for d in declared if isinstance(d, contract.Feature)}
    (obj,) = [d for d in declared if isinstance(d, contract.Strategy) and d.name == final.name]
    menu = run.menu() if brief.has_archetypes else None
    bound = contract.bind(obj, brief, menu)
    required = _required(obj, local, final.accepted)
    spec = isolate.Spec(final.code_path, obj.name, "strategy", required, bound.legs,
                        bound.template, None)  # fmt: skip
    return isolate.Isolated(pool, spec)(panel), bound


@dataclass
class Judgement:
    """One pass over a day: the scored model's pass, its hourly series and the day's terms."""

    day: Day
    window: Window
    scored: sim_maker.ArbSim
    hours: list[dict]
    arb: dict
    transfers: dict
    latencies: dict
    brief_sha: str
    srt: bool
    judged_at: str
    pnl: dict = field(default_factory=dict)
    stock: dict = field(default_factory=dict)


def judge(run: RunDir, day: Day, latencies: dict, final: Final | None = None, pool=None,
          srt: bool | None = None, write: bool = True) -> Judgement:  # fmt: skip
    """One continuous pass of the final over ``day`` from its window's start to the last
    closed hour (the day's end when closed): the judge's call, both queue models, the worse
    scored. ``latencies`` go into the forward brief."""
    final = final or final_of(run)
    if final.archetype != "cross_venue_making":
        raise ForwardError(f"forward paper judges an arbitrage final; {final.name} is "
                           f"{final.archetype or 'not an archetype strategy'}")  # fmt: skip
    raw, brief = forward_brief(run, day, latencies, final, write=False)
    until = min(day.upto or day.end, day.end)
    win = Window("forward", day.start + brief.grid.warmup, until)
    if win.end <= win.start:
        raise ForwardError(f"{day.label}: no closed hour after its warm-up yet")
    if brief.data.theo is not None and day.theo is None:
        lo = int((day.start - WARMUP).value)
        hi = int(until.value) + sim_maker._AFTER
        at = pl.col("ts").cast(pl.Datetime("ns")).dt.epoch("ns")
        rows = pl.concat([pl.scan_parquet(str(p)).filter((at >= lo) & (at < hi)).collect()
                          for p in day.data])  # fmt: skip
        day = dataclasses.replace(day, theo=(day_theo(run, brief, rows, day.label),))
    # written once the day's theo is known, so the brief names it
    raw, brief = forward_brief(run, day, latencies, final, write=write)
    panel = panel_of(brief, day, until)
    own = pool is None
    if own:
        pool, srt = fenced_pool(run)
    try:
        quotes, bound = quotes_of(run, final, brief, panel, pool)
    finally:
        if own:
            pool.close()
    (quote,) = bound.legs or (brief.target.instrument,)
    _, hedge = archetypes.arb_pair(brief)
    sims = sim_maker.arb_models(quotes, panel, win, brief, quote, hedge,
                                which=sim_maker.QUEUE_MODELS)  # fmt: skip
    scored = sim_maker.worse(sims["RA"], sims["PR"])
    hours = hourly(scored, panel, win, day, quote, hedge)
    j = Judgement(day, win, scored, hours, sim_maker.arb_table(scored),
                  sim_maker.transfer_terms(scored), dict(latencies),
                  sha256_text(yaml.safe_dump(raw, sort_keys=False)), bool(srt),
                  pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))  # fmt: skip
    j.pnl = sim_maker.arb_pnl_lines(scored)
    path = sim_maker.stock_path(scored)
    rows = path["rows"]
    every = max(1, math.ceil(len(rows) / 2000))  # the page's chart needs no more
    j.stock = {"rows": rows[::every] + ([rows[-1]] if rows and (len(rows) - 1) % every else [])}
    return j


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def hourly(s: sim_maker.ArbSim, panel: Panel, win: Window, day: Day, quote: str,
           hedge: str) -> list[dict]:  # fmt: skip
    """The pass's per-step net summed by hour of the decision and its fills counted by hour
    of the fill, with each hour's decomposition, shares and status."""
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [win.start.value, win.end.value])
    G = g[i0:i1]
    usd = s.net * s.ref * 1e-4
    f = sim_maker.arb_fills(s)
    fill_hour = f["ns"] // HOUR_NS * HOUR_NS
    step_hour = G // HOUR_NS * HOUR_NS
    sent = np.array([x["sent_ns"] // HOUR_NS * HOUR_NS for x in s.transfers], dtype=np.int64)
    gap = _mid_gap(panel, quote, hedge, i0, i1)
    out = []
    for h in np.unique(step_hour):
        steps = step_hour == h
        mine = fill_hour == h
        n = int(mine.sum())

        def mean(key, mine=mine, n=n):
            return float(np.nanmean(f[key][mine])) if n else None

        status, why = day.statuses.get(int(h), ("unobserved", "no recorder line"))
        # an hour the recorder missed says so, warm or not
        if status == "judged" and not day.warm and h < day.start.value + WARMUP.value:
            status, why = "warmup", "the first recorded day's first two hours: no day before"
        usd_traded = f["qty"][mine] * f["price"][mine]
        out.append({
            "hour_ns": int(h), "hour": pd.Timestamp(int(h), tz="UTC").isoformat(),
            "status": status, "reason": why, "steps": int(steps.sum()),
            "net_usd": float(usd[steps].sum()), "fills": n,
            "units": float(f["qty"][mine].sum()),
            "captured_bps": mean("captured_bps"), "delay_bps": mean("delay_bps"),
            "slippage_bps": mean("slippage_bps"), "fee_bps": mean("fee_bps"),
            "net_bps": mean("net_bps"),
            "picked_off": float(np.mean(f["picked_off"][mine])) if n else None,
            "buys": float(np.mean(f["side"][mine] > 0)) if n else None,
            "withdrawn": float(np.mean(s.withdrawn[steps] > 0)) if steps.any() else None,
            "rebalances": int((sent == h).sum()),
            "drift_usd": float(np.sum(f["side"][mine] * usd_traded)),
            "mid_gap_bps": gap.get(int(h)),
        })  # fmt: skip
    return out


def _mid_gap(panel: Panel, quote: str, hedge: str, i0: int, i1: int) -> dict[int, float]:
    """The quote venue's mid less the hedge venue's, bps of the hedge's, median by hour."""
    cols = [f"{v}.{x}" for v in (quote, hedge) for x in ("bid", "ask")]
    if not all(c in panel.frame.columns for c in cols):
        return {}
    x = panel.frame.slice(i0, i1 - i0).select(
        pl.col("ts").dt.epoch("ns").alias("t"),
        ((pl.col(cols[0]) + pl.col(cols[1])) / (pl.col(cols[2]) + pl.col(cols[3])) - 1) * 1e4,
    )
    x = x.rename({x.columns[1]: "gap"}).drop_nulls()
    by = x.group_by((pl.col("t") // HOUR_NS * HOUR_NS).alias("h")).agg(pl.col("gap").median())
    return {int(h): float(v) for h, v in by.iter_rows()}


# -- the ledger ------------------------------------------------------------------------------

_HOUR_COLS = ("day", "hour", "hour_ns", "judged_at", "status", "reason", "steps", "net_usd",
              "fills", "units", "captured_bps", "delay_bps", "slippage_bps", "fee_bps",
              "net_bps", "picked_off", "buys", "withdrawn", "rebalances", "drift_usd",
              "mid_gap_bps", "rows_json", "gaps", "clock_json", "brief_sha256", "data_json",
              "strategy_sha256", "srt")  # fmt: skip
_DAY_COLS = ("day", "judged_at", "upto", "closed", "hours_judged", "hours_excluded", "arb_json",
             "transfers_json", "latencies_json", "brief_sha256", "data_json", "strategy_sha256",
             "srt", "note")  # fmt: skip
_REF_COLS = ("judged_at", "latencies_json", "hours_json", "m_net", "s_net", "m_fills",
             "s_fills", "n", "arb_json", "ci_json", "h_net", "h_fills",
             "calibration_json")  # fmt: skip


class ForwardLedger:
    """``vault/forward/ledger.sqlite``: every hour of every judgement, the day rows, and the
    reference the CUSUM measures against. Rows are only ever added; the page reads the
    latest of each and says how many there were."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        for name, cols in (("hours", _HOUR_COLS), ("days", _DAY_COLS),
                           ("reference", _REF_COLS)):  # fmt: skip
            self.db.execute(f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(cols)})")
            # a ledger from an earlier version gains any new columns, its rows null in them
            have = {r[1] for r in self.db.execute(f"PRAGMA table_info({name})")}
            for c in cols:
                if c not in have:
                    self.db.execute(f"ALTER TABLE {name} ADD COLUMN {c}")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def _add(self, table: str, cols: tuple, row: dict) -> None:
        self.db.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES "
                        f"({', '.join('?' * len(cols))})", [row.get(c) for c in cols])  # fmt: skip

    def add(self, j: Judgement, final: Final) -> None:
        day, data = j.day, json.dumps(j.day.hashes, sort_keys=True)
        lines = {}
        for h in j.hours:
            lines[h["hour_ns"]] = h
            self._add(
                "hours",
                _HOUR_COLS,
                h
                | {
                    "day": day.label,
                    "judged_at": j.judged_at,
                    "clock_json": json.dumps(day.clock),
                    "brief_sha256": j.brief_sha,
                    "data_json": data,
                    "strategy_sha256": final.source_sha,
                    "srt": int(j.srt),
                },
            )
        excluded = [h for h in j.hours if h["status"] != "judged"]
        judged = len(j.hours) - len(excluded)
        note = None
        if len(excluded) > MAX_EXCLUDED:
            note = f"too few hours: {len(excluded)} of {len(j.hours)} excluded"
        self._add("days", _DAY_COLS, {
            "day": day.label, "judged_at": j.judged_at, "upto": j.window.end.isoformat(),
            "closed": int(day.closed), "hours_judged": judged, "hours_excluded": len(excluded),
            "arb_json": None if note else json.dumps(j.arb),
            "transfers_json": None if note else json.dumps(j.transfers),
            "latencies_json": json.dumps(j.latencies), "brief_sha256": j.brief_sha,
            "data_json": data, "strategy_sha256": final.source_sha, "srt": int(j.srt),
            "note": note})  # fmt: skip
        self.db.commit()

    def add_reference(self, ref: dict) -> None:
        self._add("reference", _REF_COLS, ref)
        self.db.commit()

    def reference(self, latencies: dict) -> dict | None:
        rows = self.db.execute("SELECT * FROM reference WHERE latencies_json = ? ORDER BY rowid",
                               (json.dumps(latencies, sort_keys=True),)).fetchall()  # fmt: skip
        return dict(rows[-1]) if rows else None

    def latest_hours(self) -> list[dict]:
        """Each (day, hour)'s row from the latest judgement of its day, in time order."""
        rows = self.db.execute("SELECT * FROM hours ORDER BY rowid").fetchall()
        last = {}
        for r in self.db.execute("SELECT day, MAX(judged_at) AS at FROM hours GROUP BY day"):
            last[r["day"]] = r["at"]
        out = {(r["day"], r["hour_ns"]): dict(r) for r in rows if r["judged_at"] == last[r["day"]]}
        return sorted(out.values(), key=lambda r: r["hour_ns"])

    def days(self) -> list[dict]:
        """The latest row of each day, with how many judgements it had."""
        rows = [dict(r) for r in self.db.execute("SELECT * FROM days ORDER BY rowid")]
        out = {}
        for r in rows:
            n = out.get(r["day"], {}).get("judgements", 0) + 1
            out[r["day"]] = r | {"judgements": n}
        return list(out.values())


# -- the reference and the verdict -----------------------------------------------------------


def reference(run: RunDir, latencies: dict, final: Final, pool=None, srt=None) -> dict:
    """The locked test's hourly net and fills, recomputed by this engine at the forward
    latencies on the vault's test segment (:func:`segment_day`): their means and sds over the
    test window's whole hours; each CUSUM's threshold (:func:`threshold`) and what it buys
    (:func:`false_alarms`); and the report's 90% interval of the final's net, for the page's
    second comparison.

    The thresholds count the error of a mean and sd taken from so few hours. They are
    bootstrapped from the whole vault day's hours at the same latencies when it has more
    whole hours than the test, the test's own otherwise: a CUSUM standardised by its own
    sample's mean and sd alarms as often whatever the hours' level and spread, so train's
    and validation's hours, which the search saw, lend only the shape of an hour's net and
    fills (their skew, their zeros), never the level the forward hours are held to."""
    step = int(pd.Timedelta(run.brief().grid.step).value)

    def whole_hours(day):
        j = judge(run, day, latencies, final, pool, srt, write=False)
        return j, [h for h in j.hours if h["status"] == "judged" and h["steps"] * step >= HOUR_NS]

    j, whole = whole_hours(segment_day(run))
    net = np.array([h["net_usd"] for h in whole])
    fills = np.array([h["fills"] for h in whole], dtype=float)
    _, wider = whole_hours(segment_day(run, whole=True))
    if len(wider) <= len(whole):
        wider = whole
    source = "the vault day" if wider is not whole else "the test"
    calibration = {"hours": len(wider), "from": source}
    thresholds = {}
    for name, key, mine in (("net", "net_usd", net), ("fills", "fills", fills)):
        x = np.array([h[key] for h in wider], dtype=float)
        h = threshold(x, len(whole))
        thresholds[name] = h
        # what the page's own CUSUM, at this reference's m and s, would see: the same shape
        # moved to the locked test's mean and sd, then halved or stopped
        at_ref = mine.mean() + mine.std(ddof=1) * (x - x.mean()) / x.std(ddof=1)
        calibration[name] = {
            "h": h,
            "false_alarm": false_alarms(x, h=h, n=len(whole), seed=2)["share"],
            "halving": false_alarms(at_ref, h=h, seed=3, scale=0.5),
            "stop": false_alarms(at_ref, h=h, seed=4, scale=0.0),
            "known": false_alarms(x, n=None, seed=5)["share"],  # h = 5, m and s known
            "estimated_at_5": false_alarms(x, n=len(whole), seed=6)["share"],  # h = 5
        }
    ci = None
    report = run.root / "report.json"
    if report.exists():
        r = json.loads(report.read_text())
        rec = (r.get("recompute") or {}).get("final") or {}
        lo_hi = (r.get("verdict") or {}).get("ci")
        if lo_hi and rec.get("ref"):
            # the verdict's interval is bps of one unit an hour at the test pass's P_ref
            ci = {"usd_per_hour": [x * 1e-4 * rec["ref"] for x in lo_hi],
                  "hours": (rec.get("arb") or {}).get("hours")}  # fmt: skip
    return {"judged_at": j.judged_at, "latencies_json": json.dumps(latencies, sort_keys=True),
            "hours_json": json.dumps([{k: h[k] for k in ("hour", "net_usd", "fills")}
                                      for h in whole]),
            "m_net": float(net.mean()), "s_net": float(net.std(ddof=1)),
            "m_fills": float(fills.mean()), "s_fills": float(fills.std(ddof=1)),
            "n": len(whole), "arb_json": json.dumps(j.arb), "ci_json": json.dumps(ci),
            "h_net": thresholds["net"], "h_fills": thresholds["fills"],
            "calibration_json": json.dumps(calibration)}  # fmt: skip


def cusum(
    x: np.ndarray, m: float, s: float, k: float = K, h: float = H
) -> tuple[np.ndarray, int | None]:
    """The lower one-sided CUSUM of ``x`` in units of the reference: :math:`z_t = (x_t - m)/s`,
    :math:`C_t = \\max(0, C_{t-1} - z_t - k)` from :math:`C_0 = 0`; returns every
    :math:`C_t` and the index of the first :math:`C_t > h`, or None.

    Examples
    --------
    >>> c, at = cusum(np.array([1.0, 0.0, -1.0, -1.0]), m=1.0, s=1.0, k=0.5, h=2.0)
    >>> c.tolist(), at
    ([0.0, 0.5, 2.0, 3.5], 3)
    """
    z = (np.asarray(x, dtype=float) - m) / s
    c, out, first = 0.0, np.zeros(len(z)), None
    for t, zt in enumerate(z):
        c = max(0.0, c - zt - k)
        out[t] = c
        if first is None and c > h:
            first = t
    return out, first


def _weeks(x: np.ndarray, k: float, hours: int, weeks: int, seed: int, scale: float,
           n: int | None) -> np.ndarray:  # fmt: skip
    """Every :math:`C_t` of ``weeks`` weeks of ``hours`` hours drawn with replacement from
    ``x``, each multiplied by ``scale``, standardised by ``x``'s own mean and sd (``n`` None:
    the reference known) or, each week, by the mean and sd of ``n`` hours drawn from ``x``
    (a reference of ``n`` hours, with its estimates' error); a week whose reference has no
    spread is dropped."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(weeks, hours), replace=True)
    if n is None:
        m, s = np.full(weeks, x.mean()), np.full(weeks, x.std(ddof=1))
    else:
        ref = rng.choice(x, size=(weeks, n), replace=True)
        m, s = ref.mean(axis=1), ref.std(axis=1, ddof=1)
    keep = s > 0
    z = (draws[keep] * scale - m[keep, None]) / s[keep, None]
    c = np.zeros_like(z)
    prev = np.zeros(len(z))
    for t in range(hours):
        prev = np.maximum(0.0, prev - z[:, t] - k)
        c[:, t] = prev
    return c


def false_alarms(x: np.ndarray, k: float = K, h: float = H, hours: int = 168,
                 weeks: int = 20_000, seed: int = 1, scale: float = 1.0,
                 n: int | None = None) -> dict:  # fmt: skip
    """How often the CUSUM alarms within ``hours`` on hours drawn with replacement from
    ``x`` itself, each multiplied by ``scale`` (1, no change: the false-alarm rate; 0.5, a
    halving: the detection delay), against ``x``'s own mean and sd, or with ``n`` against
    those of ``n`` hours drawn from ``x`` each week, as a reference of ``n`` hours estimates
    them: the share of weeks with an alarm and the median and p90 hours to it."""
    over = _weeks(x, k, hours, weeks, seed, scale, n) > h
    alarmed = over.any(axis=1)
    first = over.argmax(axis=1)[alarmed] + 1
    return {"share": float(alarmed.mean()),
            "median_hours": float(np.median(first)) if len(first) else None,
            "p90_hours": float(np.percentile(first, 90)) if len(first) else None,
            "siegmund_arl_hours": (math.exp(2 * k * h) - 2 * k * h - 1) / (2 * k * k)}  # fmt: skip


def threshold(x: np.ndarray, n: int | None, k: float = K, rate: float = RATE,
              hours: int = 168, weeks: int = 20_000, seed: int = 1) -> float:  # fmt: skip
    """The smallest ``h``, to 0.1, at which the CUSUM alarms within ``hours`` in at most
    ``rate`` of the weeks :func:`false_alarms` draws with no change, the reference estimated
    from ``n`` hours (None: known): the ``1 - rate`` quantile of a week's largest
    :math:`C_t`, rounded up.

    A mean and sd from a few hours are off by a fraction of an sd, which the CUSUM reads as
    drift, so the threshold that holds the rate grows as the reference shrinks.

    Examples
    --------
    >>> x = np.random.default_rng(0).exponential(size=22)
    >>> threshold(x, None, weeks=4000) < threshold(x, 22, weeks=4000) < threshold(x, 5, weeks=4000)
    True
    """
    top = _weeks(x, k, hours, weeks, seed, 1.0, n).max(axis=1)
    return math.ceil(float(np.quantile(top, 1 - rate)) * 10 - 1e-9) / 10


def _hours(n: int) -> str:
    return f"{n} judged hour{'' if n == 1 else 's'}"


def judged_totals(hours: list[dict]) -> dict:
    """The judged hours' count, fills and net, and their net a fill and buy share weighted by
    each hour's fills: what a day adds to the forward record, its excluded hours left out.

    Examples
    --------
    >>> hours = [{"status": "judged", "fills": 3, "net_usd": 0.5, "net_bps": 10.0, "buys": 1.0},
    ...          {"status": "gapped", "fills": 9, "net_usd": 9.0, "net_bps": 50.0, "buys": 0.0},
    ...          {"status": "judged", "fills": 1, "net_usd": -0.1, "net_bps": -2.0, "buys": 0.0}]
    >>> t = judged_totals(hours)
    >>> t["hours"], t["fills"], round(t["net_usd"], 6), t["net_bps"], t["buys"]
    (2, 4, 0.4, 7.0, 0.75)
    """
    judged = [h for h in hours if h["status"] == "judged"]
    fills = sum(h["fills"] for h in judged)

    def weighted(key):
        if not fills:
            return None
        return sum(h[key] * h["fills"] for h in judged if h["fills"]) / fills

    return {"hours": len(judged), "fills": fills, "net_usd": sum(h["net_usd"] for h in judged),
            "net_bps": weighted("net_bps"), "buys": weighted("buys")}  # fmt: skip


def own_interval(net: np.ndarray, level: float = LEVEL, n_boot: int = 9_999, seed: int = 7,
                 periods: int | None = None) -> dict:  # fmt: skip
    """The forward hours' own interval for the mean hourly net, and where it lies against
    zero (``side``: ``above``, ``includes``, ``below``; None under ``periods`` hours).

    The interval is studentised (bootstrap-t), each judged hour a block, and claims
    something only on ``periods`` hours (:data:`blinded.archetypes.PERIODS`, 20), as the
    judge's fee verdict claims one only on 20 blocks. An hour is its own block because an
    hour's net barely depends on the last: on the AT day's hedged 5-minute block nets the
    variance of an hour's sum is 2.88 times that of twelve independent blocks and of two
    hours' 2.92 times that of twenty-four, so the dependence is spent within the hour. Drawn
    20, 48 and 168 at a time from that day's hours, the interval covered the mean 0.90 to
    0.92 of the time, also with the hours' upper tail stretched threefold.

    It answers what the CUSUM cannot: the CUSUM asks whether the forward hours differ from
    the locked test's, and a reference of a few hours is too loose to see a halving within a
    week; this asks whether the forward hours make money, which a week of them measures
    about five times more tightly than the locked test measured its own (the square root of
    168 hours over the AT test's 5.9).

    Examples
    --------
    >>> own_interval(np.ones(5))["side"] is None
    True
    >>> r = own_interval(np.linspace(0.1, 1.0, 24))
    >>> r["side"], r["lo"] < r["mean"] < r["hi"]
    ('above', True)
    >>> own_interval(np.linspace(-1.0, 1.0, 24))["side"]
    'includes'
    """
    periods = archetypes.PERIODS if periods is None else periods
    x = np.asarray(net, dtype=float)
    n = len(x)
    out = {"n": n, "level": level, "periods": periods,
           "mean": float(x.mean()) if n else None}  # fmt: skip
    if n < periods:
        return out | {"side": None, "lo": None, "hi": None}
    se = float(x.std(ddof=1)) / math.sqrt(n)
    if se == 0:
        lo = hi = float(x.mean())
    else:
        d = x[np.random.default_rng(seed).integers(0, n, size=(n_boot, n))]
        sd = d.std(axis=1, ddof=1)
        ok = sd > 0
        t = (d.mean(axis=1)[ok] - x.mean()) / (sd[ok] / math.sqrt(n))
        q_lo, q_hi = np.quantile(t, ((1 - level) / 2, 1 - (1 - level) / 2))
        lo, hi = float(x.mean() - q_hi * se), float(x.mean() - q_lo * se)
    side = "above" if lo > 0 else "below" if hi < 0 else "includes"
    return out | {"side": side, "lo": lo, "hi": hi}


def own_line(own: dict) -> str:
    """The page's sentence on :func:`own_interval`."""
    if own["side"] is None:
        return (f"The forward hours' own interval for the hourly net waits for "
                f"{own['periods']} judged hours; {own['n']} so far.")  # fmt: skip
    says = {
        "above": "above zero: forward, the strategy has made money",
        "includes": "across zero: forward cannot yet say the strategy makes money",
        "below": "below zero: forward, the strategy has lost money",
    }[own["side"]]
    return (f"The forward hours' own {own['level']:.0%} interval for the hourly net, over "
            f"{_hours(own['n'])}: {own['lo']:+.3f} to {own['hi']:+.3f} USD an hour (mean "
            f"{own['mean']:+.3f}), {says}.")  # fmt: skip


def verdict(hours: list[dict], ref: dict) -> dict:
    """Both CUSUMs over the judged hours, the forward hours' own interval
    (:func:`own_interval`) and the page's sentences."""
    judged = [h for h in hours if h["status"] == "judged"]
    net = np.array([h["net_usd"] for h in judged])
    fills = np.array([h["fills"] for h in judged], dtype=float)
    out = {"n": len(judged), "net": float(net.sum()), "fills": int(fills.sum())}
    if not judged:
        out["line"] = "No hour has been judged yet."
        return out
    out["own"] = own_interval(net)
    out["own_line"] = own_line(out["own"])
    h_net, h_fills = ref.get("h_net") or H, ref.get("h_fills") or H
    c_net, a_net = cusum(net, ref["m_net"], ref["s_net"], h=h_net)
    c_fills, a_fills = cusum(fills, ref["m_fills"], ref["s_fills"], h=h_fills)
    out |= {"cusum_net": c_net.tolist(), "cusum_fills": c_fills.tolist(),
            "alarm_net": a_net, "alarm_fills": a_fills, "h_net": h_net, "h_fills": h_fills,
            "hours": [h["hour"] for h in judged]}  # fmt: skip
    alarms = [(a, name, h, x, m) for a, name, h, x, m in (
        (a_net, "hourly net", h_net, net.mean(), ref["m_net"]),
        (a_fills, "hourly fills", h_fills, fills.mean(), ref["m_fills"]))
        if a is not None]  # fmt: skip
    if alarms:
        parts = [f"the CUSUM on the {name} crossed {h:g} at {judged[a]['hour'][:16]} UTC, "
                 f"{len(judged) - 1 - a} judged hours ago; the forward mean is {x:.3f} against "
                 f"the locked test's {m:.3f}" for a, name, h, x, m in alarms]  # fmt: skip
        out["line"] = "The forward period disagrees with the locked test: " + "; ".join(parts)
        out["agrees"] = False
    else:
        out["line"] = (
            f"Agrees so far: {_hours(len(judged))}, net {net.sum():+.2f} USD "
            f"against {ref['m_net'] * len(judged):+.2f} expected, fills "
            f"{int(fills.sum()):,} against {ref['m_fills'] * len(judged):,.0f}."
        )
        out["agrees"] = True
    ci = json.loads(ref.get("ci_json") or "null")
    if ci:
        lo, hi = (x * len(judged) for x in ci["usd_per_hour"])
        t = ci["hours"]
        a, b = (x * t for x in ci["usd_per_hour"])
        out["ci_line"] = (f"The locked test's own 90% interval, {a:+.2f} to {b:+.2f} USD over "
                          f"{t:.1f} hours, scaled to {_hours(len(judged))}: {lo:+.2f} to "
                          f"{hi:+.2f} USD; forward made {net.sum():+.2f}.")  # fmt: skip
    return out


def calibration_line(ref: dict) -> str:
    """The page's sentence on the two thresholds: what they are set for and what they buy."""
    cal = json.loads(ref.get("calibration_json") or "null")
    if not cal:
        return f"The CUSUM uses k = {K:g} and h = {H:g}, in those sds."
    net, fills = cal["net"], cal["fills"]

    def caught(change):
        out = []
        for stat, label in (("net", "the net"), ("fills", "fills")):
            r = cal[stat][change]
            if r["share"] >= 0.5:
                out.append(f"in a median {r['median_hours']:.0f} hours on {label}")
            elif r["share"] > 0:
                out.append(f"within a week in {r['share']:.0%} of weeks on {label}")
            else:
                out.append(f"not within a week on {label}")
        return " and ".join(out)

    return (f"The CUSUM uses k = {K:g}, and h = {net['h']:g} on the net and {fills['h']:g} on "
            f"fills, in those sds: the thresholds at which an unchanged strategy alarms in "
            f"{RATE:.0%} of weeks once the error of a mean and sd from {ref['n']} hours is "
            f"counted, bootstrapped from {cal['hours']} whole hours of {cal['from']} (h = "
            f"{H:g} would alarm in {net['estimated_at_5']:.0%} and {fills['estimated_at_5']:.0%}"
            f" of weeks). Against this reference a stop is caught {caught('stop')}, a halving "
            f"{caught('halving')}.")  # fmt: skip


# -- the command -----------------------------------------------------------------------------


def latencies_of(live: Path, day: Date) -> dict:
    """The measured latencies of a recorded day (:func:`blinded.recorder.day_latencies`)."""
    return recorder.day_latencies(Path(live) / day.isoformat())


def run_forward(
    run_dir: str | Path,
    live: str | Path | None = None,
    latency_ms: float | None = None,
    segment: str | None = None,
    log: Callable[[str], None] = print,
    coin: str | None = None,
    skip: set[str] | None = None,
) -> list[Judgement]:
    """Judge every recorded day under ``live`` up to its last closed hour (or, with
    ``segment="test"``, the vault's own test segment), append to the forward ledger and
    rewrite the page. ``latency_ms`` sets every latency, the hedge's too, in place of the
    day's measured ones."""
    run = RunDir(Path(run_dir).expanduser().resolve())
    run.verify()
    final = final_of(run)
    brief = run.brief()
    coin = coin or sim_maker.coin_of(brief.target.instrument)
    fixed = None
    if latency_ms is not None:
        ms = int(latency_ms) if float(latency_ms).is_integer() else float(latency_ms)
        fixed = {"latency_ms": ms, "cancel_latency_ms": ms, "response_latency_ms": ms,
                 "hedge_latency_ms": ms}  # fmt: skip
    # the fence first: without srt nothing is written
    pool, srt = fenced_pool(run)
    ledger = ForwardLedger(forward_dir(run) / "ledger.sqlite")
    out = []
    try:
        with tempfile.TemporaryDirectory(prefix="blinded-forward-") as scratch:
            if segment == "test":
                days = [(segment_day(run), fixed or _vault_latencies(brief))]
            elif segment is not None:
                raise ForwardError("--segment takes test only")
            else:
                if live is None:
                    raise ForwardError("--live-data names the recorder's directory")
                days = []
                for d in recorded_days(Path(live), coin):
                    if d.isoformat() in (skip or set()):
                        continue
                    day = live_day(Path(live), d, coin, Path(scratch))
                    days.append((day, fixed or latencies_of(Path(live), d)))
            for day, lat in days:
                j = judge(run, day, lat, final, pool, srt)
                ledger.add(j, final)
                t = judged_totals(j.hours)
                log(f"{day.label}: {t['hours']} of {len(j.hours)} hours judged, {t['fills']} "
                    f"fills and net {t['net_usd']:+.4f} USD in them")  # fmt: skip
                out.append(j)
            if out:
                lat = out[-1].latencies
                ref = ledger.reference(lat)
                if ref is None:
                    ref = reference(run, lat, final, pool, srt)
                    ledger.add_reference(ref)
                write_page(run, final, ledger, ref, out[-1])
    finally:
        pool.close()
        ledger.close()
    return out


def watch(run_dir: str | Path, live: str | Path, latency_ms: float | None = None,
          log: Callable[[str], None] = print) -> None:  # fmt: skip
    """Judge the recorded days, then again ten minutes after every hour, when the recorder
    has written the hour that closed; a day is judged once more after it closes, and not
    again."""
    import time

    done: set[str] = set()
    while True:
        js = run_forward(run_dir, live, latency_ms, log=log, skip=done)
        done |= {j.day.label for j in js if j.day.closed}
        now = time.time()
        time.sleep(max(60.0, 3600 - now % 3600 + 600))


def _vault_latencies(brief: briefs.Brief) -> dict:
    ex = brief.execution

    def ms(td):
        x = td.value / 1e6
        return int(x) if float(x).is_integer() else x

    out = {"latency_ms": ms(ex.latency)}
    for key, td in (("cancel_latency_ms", ex.cancel_latency),
                    ("response_latency_ms", ex.response_latency),
                    ("hedge_latency_ms", ex.hedge_latency)):  # fmt: skip
        if td is not None:
            out[key] = ms(td)
    return out


# -- the page --------------------------------------------------------------------------------


def write_page(run: RunDir, final: Final, ledger: ForwardLedger, ref: dict,
               last: Judgement) -> Path:  # fmt: skip
    hours = ledger.latest_hours()
    # the vault's own test segment (--segment test) is the known answer, not a forward day:
    # its hours enter the verdict only on a page with no recorded day
    live = [h for h in hours if h["day"] != "test"]
    html = page(run, final, live or hours, ledger.days(), ref, last, every=hours)
    path = forward_dir(run) / "forward.html"
    path.write_text(html)
    return path


def page(run: RunDir, final: Final, hours: list[dict], days: list[dict], ref: dict,
         last: Judgement, every: list[dict] | None = None) -> str:  # fmt: skip
    """``forward.html``: a static page, no JavaScript, the charts as inline SVG. ``hours``
    are the verdict's; ``every`` (``hours`` if not given) are every day's, for the days'
    table."""
    from html import escape

    from blinded import plots

    v = verdict(hours, ref)
    lat = ", ".join(f"{k} {x}" for k, x in last.latencies.items())
    clock = "; ".join(f"{k} {c['offset_ms']:+.1f} ms (round trip median {c['rtt_ms']:.0f} ms)"
                      for k, c in last.day.clock.items()) or "none recorded"  # fmt: skip
    head = [
        f"<h1>Forward paper: {escape(final.name)}</h1>",
        f"<p>Run <code>{escape(run.run_id)}</code> ({escape(str(run.root))}), final trial "
        f"{final.trial_id}, source sha256 <code>{final.source_sha[:16]}</code>. Recorder host: "
        f"{escape(last.day.host or 'unknown')}. Clock against the venues on the last day: "
        f"{escape(clock)}. Latencies of the last judgement, ms: {escape(lat)}. Judged in "
        f"{'srt' if last.srt else 'no OS sandbox (BLINDED_UNFENCED=1)'}.</p>",
        f"<p><strong>{escape(v['line'])}</strong></p>",
    ]
    if v.get("own_line"):
        head.append(f"<p>{escape(v['own_line'])}</p>")
    if v.get("ci_line"):
        head.append(f"<p>{escape(v['ci_line'])}</p>")
    head.append(f"<p>Reference: the locked test's whole hours at these latencies, n = "
                f"{ref['n']}: hourly net {ref['m_net']:+.3f} USD (sd {ref['s_net']:.3f}), "
                f"hourly fills {ref['m_fills']:.1f} (sd {ref['s_fills']:.1f}). "
                f"{calibration_line(ref)}</p>")  # fmt: skip
    charts = []
    if hours:
        charts.append(plots.forward_hourly(hours))
        charts.append(plots.forward_cumulative(hours, ref, v))
        if v.get("cusum_net"):
            charts.append(plots.forward_cusum(v))
    if last.pnl:
        charts.append(plots.forward_decomposition(last.pnl, last.day.label))
    if last.stock.get("rows"):
        charts.append(plots.forward_stock(last.stock["rows"], last.day.label))
    rows = []
    for d in days:
        cells = [d["day"], d["upto"][:16], d["hours_judged"], d["hours_excluded"],
                 d["judgements"]]  # fmt: skip
        if d["arb_json"]:
            # the judged hours only: the pass ran through the excluded ones, but they add nothing
            t = judged_totals([h for h in every or hours if h["day"] == d["day"]])
            cells += [
                t["fills"],
                f"{t['net_bps']:+.2f}" if t["net_bps"] is not None else "",
                f"{t['net_usd']:+.4f}",
                f"{t['buys']:.2f}" if t["buys"] is not None else "",
            ]
        else:
            cells += [d["note"] or "", "", "", ""]
        rows.append("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in cells) + "</tr>")
    table = ("<table><tr><th>day</th><th>to</th><th>hours judged</th><th>excluded</th>"
             "<th>judgements</th><th>fills</th><th>net a fill, bps</th><th>net USD</th>"
             "<th>buys</th></tr>" + "".join(rows) + "</table><p>Fills, net and buys are the "
             "judged hours' alone.</p>")  # fmt: skip
    hrows = []
    for h in hours[-48:]:
        cells = [h["hour"][:16], h["status"], h["reason"] or "", f"{h['net_usd']:+.4f}",
                 h["fills"], "" if h["net_bps"] is None else f"{h['net_bps']:+.2f}",
                 "" if h["mid_gap_bps"] is None else f"{h['mid_gap_bps']:+.1f}"]  # fmt: skip
        hrows.append("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in cells) + "</tr>")
    htable = ("<table><tr><th>hour, UTC</th><th>status</th><th>why</th><th>net USD</th>"
              "<th>fills</th><th>net a fill, bps</th><th>quote venue's mid less the hedge "
              "venue's, bps</th></tr>" + "".join(hrows) + "</table>")  # fmt: skip
    style = ("body{font-family:sans-serif;max-width:60rem;margin:1rem auto;padding:0 1rem;"
             "color:#1a1a1a;background:#fff}table{border-collapse:collapse;font-size:.85rem}"
             "td,th{border:1px solid #ccc;padding:.2rem .4rem;text-align:right}"
             "svg{max-width:100%;height:auto}")  # fmt: skip
    body = "\n".join([*head, *charts, "<h2>Days</h2>", table,
                      "<h2>The last 48 hours</h2>", htable])  # fmt: skip
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>Forward paper</title>"
        f"<style>{style}</style></head><body>\n{body}\n</body></html>\n"
    )


def serve(run_dir: str | Path, port: int = PORT) -> None:
    """Serve ``vault/forward/forward.html`` on 127.0.0.1 only, and nothing else."""
    import http.server

    run = RunDir(Path(run_dir).expanduser().resolve())
    path = forward_dir(run) / "forward.html"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path not in ("/", "/forward.html") or not path.exists():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"serving {path} at http://127.0.0.1:{port}/", flush=True)
        httpd.serve_forever()
