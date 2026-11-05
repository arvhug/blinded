"""The judge daemon: watches the workspace inbox, owns the vault, answers in the outbox.

A request is a pair of files dropped by ``blinded submit``: ``<id>.py``, the
code, and ``<id>.json``, written last. The daemon scans the code, judges it in
a child process with a wall-time and memory limit, writes a ledger row per
feature or strategy, puts validation scores through the Ladder, regenerates
``LEDGER.md`` and writes ``outbox/<id>.json``. Responses carry train metrics in
full and, from validation, the truncation verdict and the Ladder's
``released`` and ``best``, nothing else.

A run in naive mode has the same daemon with the protections off:
no static scan, no truncation test, no Ladder, and exact metrics at zero
latency on whatever window the request names, ``all`` by default. Every call
is still a ledger row.

Trials are tagged with the iteration the driver wrote to ``vault/iteration``
before starting the session; a request cannot set it.

Each feature and strategy is checked against the brief's families before
anything of it runs (:func:`blinded.contract.bind`), and judged in its family:
a feature against its instrument's forward return or its pair's spread
return (with the funding a short perpetual collects, for a basis trade), a
strategy on its own instrument or on both legs of its pair, funding accrued
where a leg has a funding stream. Each
group has its own Ladder for each kind, so a group can release whatever the
others have released. A group is a family, or in a brief with ``archetypes``
an edge, whose archetypes share its Ladders; there every submission must
carry an archetype the menu written at init runs (``vault/menu.json``), and
anything else is an ``error`` row with the menu's line before any of it runs.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import json
import math
import multiprocessing as mp
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from blinded import contract, isolate, sim, sim_maker, truncation
from blinded.archetypes import PERIODS, REGISTRY
from blinded.contract import ContractError, Legs
from blinded.durations import label
from blinded.ladder import BlockDraw, Ladder, rank_ic_score, sharpe_score, xs_rank_ic_score
from blinded.ladder import floor as ladder_floor
from blinded.ledger import AlreadyScored, code_hash
from blinded.panel import Panel
from blinded.target import feature_returns
from blinded.vault import RunDir

POLL_SECONDS = 0.2


@dataclass(frozen=True)
class Limits:
    """Per-evaluation wall time, per-submission wall time and resident memory."""

    evaluation_seconds: float = 120.0
    submission_seconds: float = 900.0
    rss_bytes: int = 4 * 1024**3


DEFAULT_LIMITS = Limits()


class Ladders(dict):
    """The Ladders of a run, keyed by ``(kind, group)``: the group is the family, or the edge
    in a run with archetypes.

    In a run with one group the kind alone names its Ladder.
    """

    def __missing__(self, key):
        found = [ladder for (kind, _), ladder in self.items() if kind == key]
        if isinstance(key, str) and len(found) == 1:
            return found[0]
        raise KeyError(key)


def record_fence(run: RunDir, srt: str | None) -> None:
    """Add this judge's start to ``vault/fence.jsonl``: whether it ran judged code inside
    ``srt`` or, by ``BLINDED_UNFENCED=1``, without it (:func:`blinded.isolate.fence`)."""
    line = {"started": pd.Timestamp.now(tz="UTC").isoformat(), "srt": srt is not None}
    with open(run.vault / "fence.jsonl", "a") as fh:
        fh.write(json.dumps(line) + "\n")


class Daemon:
    """Judge requests for one run directory."""

    def __init__(self, run: RunDir, limits: Limits = DEFAULT_LIMITS):
        self.run = run
        self.limits = limits
        run.verify()
        record_fence(run, isolate.fence())
        self.ledger = run.ledger()
        self.run_id = run.run_id
        self.naive = run.mode == "naive"
        brief = run.brief()
        self.families = brief.has_families or brief.has_archetypes
        self.archetypes = brief.has_archetypes
        self.ladders = Ladders()
        for group in brief.groups:
            for kind in ("feature", "strategy"):
                floor = ladder_floor(kind, brief.stats.block)
                ladder = Ladder(kind, floor=floor, rules=brief.judge_rules)
                self.ladders[(kind, group.name)] = ladder
                rows = self.ledger.releases(kind, group=group.name)
                if rows:
                    last = rows[-1]
                    reps = np.load(self._replicates_path(last["trial_id"]))
                    shown = json.loads(self.ledger.trial(last["trial_id"])["val_metrics_json"]
                                       or "{}")  # fmt: skip
                    ladder.restore(last["best_after"], last["trial_id"], reps, shown.get("value"))

    def serve(self, once: bool = False) -> None:
        """Process requests until interrupted; with ``once``, process what is there and return."""
        while True:
            for request in sorted(self.run.inbox.glob("*.json"), key=lambda p: p.stat().st_mtime):
                self.handle(request)
            if once:
                return
            time.sleep(POLL_SECONDS)

    def handle(self, request_path: Path) -> dict:
        """Judge one request and write its response; returns the response."""
        rid = request_path.stem
        try:
            request = json.loads(request_path.read_text())
        except (OSError, json.JSONDecodeError):
            request = {}
        archive = self.run.vault / "requests"
        archive.mkdir(exist_ok=True)
        code_file = self.run.inbox / f"{rid}.py"
        if request.get("kind") != "eval":
            self.ledger.probe(self.run_id, "inbox", f"request of kind {request.get('kind')!r}")
            response = {"id": rid, "error": "the inbox takes eval requests only"}
        elif not code_file.is_file():
            response = {"id": rid, "error": "no code file came with the request"}
        else:
            response = self._evaluate(rid, request, code_file.read_text())
        for path in (request_path, code_file):
            if path.exists():
                shutil.move(path, archive / path.name)
        self.run.write_ledger_md(self.ledger)
        _write_json(self.run.outbox / f"{rid}.json", response)
        return response

    def iteration(self) -> int | None:
        """The driver's current iteration, from ``vault/iteration``; None outside a run."""
        try:
            return int((self.run.vault / "iteration").read_text().strip())
        except (OSError, ValueError):
            return None

    def _evaluate(self, rid: str, request: dict, source: str) -> dict:
        name = request.get("name")
        stem = Path(request.get("file", "submission.py")).stem
        common = dict(submitted_by="agent", iteration=self.iteration())
        if self.naive:
            return self._evaluate_naive(rid, request, source, name or stem, common)
        try:
            static = contract.scan(source, protected=True)
        except SyntaxError as exc:
            static = contract.Scan(errors=[f"line {exc.lineno}: syntax error"])
        if static.errors:
            reason = "; ".join(static.errors)
            kind = "strategy" if "strategy" in source else "feature"
            tid = self.ledger.add_trial(
                self.run_id,
                kind=kind,
                name=name or stem,
                code_hash=code_hash(source, name) if not reason.endswith("syntax error") else None,
                status="rejected_static",
                reason_short=reason[:200],
                **_declared_tags(source),
                **common,
            )
            return {
                "id": rid,
                "trials": [_rejected(tid, kind, name or stem, "rejected_static", reason)],
            }

        digest = code_hash(source)
        path = self.run.trials / f"{digest}.py"
        path.write_text(source)
        job = {"code_path": str(path), "name": name, "accepted": self._accepted_features()}
        started = time.monotonic()
        outcome, payload = run_isolated(self.run, job, self.limits)
        wall = time.monotonic() - started
        if outcome != "done":
            kind = _declared_kind(source, name)
            tid = self.ledger.add_trial(
                self.run_id,
                kind=kind,
                name=name or stem,
                code_hash=code_hash(source, name),
                code_path=str(path),
                status=outcome,
                reason_short=payload,
                wall_seconds=wall,
                **_declared_tags(source),
                **common,
            )
            return {
                "id": rid,
                "trials": [_rejected(tid, kind, name or stem, outcome, payload)],
            }
        tags = _declared_tags(source)
        if "crash" in payload:
            items = [{"kind": "feature", "name": name or stem, "status": "error",
                      "reason": payload["crash"], **tags}]  # fmt: skip
        else:
            items = payload["items"]
        if not items:
            items = [{"kind": "feature", "name": name or stem, "status": "error",
                      "reason": "no @bl.feature or @bl.strategy by that name",
                      **tags}]  # fmt: skip
        trials = []
        for item in items:
            trials.append(self._record(item, source, str(path), wall / len(items), common))
        return {"id": rid, "trials": trials}

    def _evaluate_naive(self, rid: str, request: dict, source: str, name: str, common) -> dict:
        window = str(request.get("window") or "all")
        kind = "strategy" if "strategy" in source else "feature"

        def failed(status: str, reason: str, **fields) -> dict:
            tid = self.ledger.add_trial(
                self.run_id,
                kind=kind,
                name=name,
                status=status,
                reason_short=reason[:200],
                **_declared_tags(source),
                **fields,
                **common,
            )
            return {"id": rid, "trials": [_rejected(tid, kind, name, status, reason)]}

        try:
            ast.parse(source)
        except SyntaxError as exc:
            return failed("error", f"line {exc.lineno}: syntax error")
        path = self.run.trials / f"{code_hash(source)}.py"
        path.write_text(source)
        job = {
            "code_path": str(path),
            "name": request.get("name"),
            "naive": True,
            "window": window,
            "accepted": self._accepted_features(),
        }
        started = time.monotonic()
        outcome, payload = run_isolated(self.run, job, self.limits)
        wall = time.monotonic() - started
        where = dict(code_hash=code_hash(source, name), code_path=str(path), wall_seconds=wall)
        if outcome != "done":
            return failed(outcome, payload, **where)
        if "crash" in payload or not payload["items"]:
            return failed("error", payload.get("crash", "no @bl.feature or @bl.strategy"), **where)
        trials = []
        for item in payload["items"]:
            metrics = item.get("metrics")
            tid = self.ledger.add_trial(
                self.run_id,
                kind=item["kind"],
                name=item["name"],
                hypothesis=item.get("hypothesis"),
                code_hash=code_hash(source, item["name"]),
                code_path=str(path),
                status=item["status"],
                reason_short=item.get("reason", "")[:200],
                train_metrics_json=None if metrics is None else {**metrics, "window": window},
                wall_seconds=wall / len(payload["items"]),
                family=item.get("family") or "directional",
                legs=item.get("legs"),
                edge=item.get("edge"),
                archetype=item.get("archetype"),
                **common,
            )
            out = _rejected(
                tid, item["kind"], item["name"], item["status"], item.get("reason", "")
            )
            out |= self._where(item)
            out["window"] = window
            if "metrics" in item:
                out["metrics"] = item["metrics"]
            trials.append(out)
        return {"id": rid, "trials": trials}

    def _record(self, item: dict, source: str, path: str, wall: float, common: dict) -> dict:
        kind, val = item["kind"], item.get("val") or {}
        family = item.get("family") or "directional"
        tid = self.ledger.add_trial(
            self.run_id,
            kind=kind,
            name=item["name"],
            hypothesis=item.get("hypothesis"),
            code_hash=code_hash(source, item["name"]),
            code_path=path,
            status=item["status"],
            reason_short=item.get("reason", "")[:200],
            train_metrics_json=item.get("train"),
            val_metrics_json={k: v for k, v in val.items() if k not in ("replicates", "blocks")},
            wall_seconds=wall,
            family=family,
            legs=item.get("legs"),
            edge=item.get("edge"),
            archetype=item.get("archetype"),
            **common,
        )
        out = {"trial_id": tid, "kind": kind, "name": item["name"], "status": item["status"],
               "reason": item.get("reason", "")}  # fmt: skip
        out |= self._where(item)
        if "train" in item:
            out["train"] = item["train"]
        if not val:
            return out
        group = item.get("group") or family
        ladder = self.ladders[(kind, group)]
        public = {"truncation": val.get("truncation", "error")}
        if item["status"] == "accepted" and val.get("truncation") == "passed":
            self.ledger.add_blocks(tid, "val", val["blocks"], val.get("block_ic"))
            from blinded.ladder import Score

            value = val.get("value")
            if value is None or not np.isfinite(value) or not val.get("releasable", True):
                # no number, or too few of its own holds: nothing for the Ladder to judge,
                # whatever the replicates say, and the agent hears what it would of a loss
                decision = {"released": False, "best": ladder.best}
                self.ledger.update_trial(tid, released_json=decision, released_bool=False,
                                         ladder_best_before=ladder.best,
                                         ladder_best_after=ladder.best)  # fmt: skip
                public |= decision
            else:
                decision = ladder.submit(Score(value, val["replicates"]), trial_id=tid)
                if decision.released:
                    np.save(self._replicates_path(tid), val["replicates"])
                    self.ledger.add_release(self.run_id, kind, tid, decision.best_before,
                                            decision.best, decision.eta, family,
                                            group)  # fmt: skip
                self.ledger.update_trial(
                    tid,
                    released_json=decision.public(),
                    released_bool=decision.released,
                    ladder_best_before=decision.best_before,
                    ladder_best_after=decision.best,
                )
                public |= decision.public()
        else:
            public |= {"released": False, "best": ladder.best}
            if "reason" in val:
                public["reason"] = val["reason"]
        out["validation"] = public
        return out

    def _where(self, item: dict) -> dict:
        """The family and legs for a response, in a run with families; with archetypes, the
        edge and archetype too."""
        if not self.families:
            return {}
        out = {"family": item.get("family") or "directional", "legs": item.get("legs")}
        if self.archetypes:
            out |= {"edge": item.get("edge"), "archetype": item.get("archetype")}
        return out

    def _accepted_features(self) -> dict[str, str]:
        """The latest accepted feature of each name, by its vault copy."""
        out = {}
        for row in self.ledger.trials("feature"):
            if row["status"] == "accepted":
                out[row["name"]] = row["code_path"]
        return out

    def _replicates_path(self, trial_id: int) -> Path:
        return self.run.trials / f"{trial_id}.val.npy"


def score_test(run: RunDir, trial_id: int, limits: Limits = DEFAULT_LIMITS) -> dict:
    """Score one trial on the test segment. Allowed once per run; a second call raises.

    Called by the driver or by a person (``blinded test``), never through the inbox.
    """
    ledger = run.ledger()
    if ledger.test_score(run.run_id) is not None:
        raise AlreadyScored(f"run {run.run_id} has already been scored on test")
    row = ledger.trial(trial_id)
    if row is None or row["status"] != "accepted":
        raise ValueError(f"trial {trial_id} is not an accepted trial")
    job = {
        "code_path": row["code_path"],
        "name": row["name"],
        "accepted": {
            r["name"]: r["code_path"]
            for r in ledger.trials("feature")
            if r["status"] == "accepted"
        },  # fmt: skip
        "segment": "test",
    }
    outcome, payload = run_isolated(run, job, limits)
    if outcome != "done" or "crash" in payload:
        raise RuntimeError(f"test scoring failed: {payload if outcome == 'done' else outcome}")
    (item,) = [i for i in payload["items"] if i["name"] == row["name"]]
    metrics = item.get("test_summary") or {"status": item["status"], "reason": item.get("reason")}
    ledger.record_test(run.run_id, trial_id, metrics)
    if item.get("test_blocks"):
        ledger.add_blocks(trial_id, "test", item["test_blocks"])
    return metrics


def run_isolated(run: RunDir, job: dict, limits: Limits) -> tuple[str, object]:
    """Judge ``job`` in a spawned child; returns ``("done", result)``, ``("timeout", why)``
    or ``("oom", why)``.

    The child reports before each feature evaluation. The parent kills it when
    one evaluation outlasts ``evaluation_seconds``, the whole submission
    ``submission_seconds``, or its resident memory passes ``rss_bytes``. RSS is
    read with ``ps`` because ``RLIMIT_AS`` is not enforced on macOS.
    """
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    job = job | {"limits": limits}
    proc = ctx.Process(target=_worker, args=(child, str(run.root), job), daemon=True)
    proc.start()
    child.close()
    started = last_tick = time.monotonic()
    try:
        while True:
            if parent.poll(0.1):
                try:
                    kind, value = parent.recv()
                except EOFError:
                    return "done", {"crash": "the evaluation process died"}
                if kind == "done":
                    return "done", value
                if kind == "limit":
                    return value
                last_tick = time.monotonic()
            now = time.monotonic()
            if now - last_tick > limits.evaluation_seconds:
                return "timeout", f"one evaluation ran past {limits.evaluation_seconds:g} s"
            if now - started > limits.submission_seconds:
                return "timeout", f"the submission ran past {limits.submission_seconds:g} s"
            rss = _rss(proc.pid)
            if rss > limits.rss_bytes:
                return "oom", f"resident memory passed {limits.rss_bytes / 1024**3:g} GB"
            if not proc.is_alive() and not parent.poll():
                return "done", {"crash": f"the evaluation process exited ({proc.exitcode})"}
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=5)
        parent.close()


def _rss(pid: int) -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    text = out.stdout.strip()
    return int(text) * 1024 if text.isdigit() else 0


def _worker(conn, root: str, job: dict) -> None:
    try:
        result = _judge(conn, RunDir(Path(root)), job)
    except isolate.Limit as exc:
        conn.send(("limit", (exc.outcome, exc.why)))
        conn.close()
        return
    except isolate.Failed as exc:
        result = {"crash": exc.reason}
    except BaseException as exc:
        result = {"crash": _describe(exc, job["code_path"])}
    conn.send(("done", result))
    conn.close()


def _judge(conn, run: RunDir, job: dict) -> dict:
    """Judge one file. Its code never runs here: each evaluation is a fresh worker
    (:mod:`blinded.isolate`), the file's features and strategies are read in one, and this
    process only binds, compares and scores what the workers send back."""
    from blinded.sandbox import protected

    limits = job.get("limits") or DEFAULT_LIMITS
    deny, allow = protected(run), [str(run.trials)]
    with isolate.Pool(limits, lambda: conn.send(("tick", None)), deny, allow) as pool:
        return _judge_in(conn, run, job, pool)


def _judge_in(conn, run: RunDir, job: dict, pool: isolate.Pool) -> dict:
    brief = run.brief()
    menu = run.menu() if brief.has_archetypes else None
    declared = [isolate.declared(d) for d in isolate.describe(pool, job["code_path"])]
    name = job.get("name")
    local = {d.name for d in declared if isinstance(d, contract.Feature)}
    feats = [d for d in declared if isinstance(d, contract.Feature) and name in (None, d.name)]
    strats = [d for d in declared if isinstance(d, contract.Strategy) and name in (None, d.name)]
    segment = job.get("segment", "val")
    panels: dict[str, Panel] = {}

    def panel(seg: str) -> Panel:
        if seg not in panels:
            panels[seg] = run.panel(seg)
        return panels[seg]

    def isolated(obj, kind: str, bound: contract.Binding, required=None) -> isolate.Isolated:
        basket = kind == "feature" and bound.family == "basket"
        spec = isolate.Spec(job["code_path"], obj.name, kind, required or {}, bound.legs,
                            bound.template, bound.legs if basket else None)  # fmt: skip
        return isolate.Isolated(pool, spec, before=lambda: conn.send(("tick", None)))

    judge_one = _judge_one
    if job.get("naive"):
        judge_one = functools.partial(_judge_naive, window=job.get("window", "all"))
    items = []
    for kind, objs in (("feature", feats), ("strategy", strats)):
        for obj in objs:
            head = {"kind": kind, "name": obj.name, "family": obj.family, **_tags(obj.archetype)}
            try:
                bound = contract.bind(obj, brief, menu)
            except ContractError as exc:
                items.append(head | {"status": "error", "reason": str(exc)})
                continue
            required = None
            if kind == "strategy":
                try:
                    required = _required(obj, local, job["accepted"])
                except KeyError as exc:
                    items.append(head | {"status": "error", "reason": exc.args[0]})
                    continue
            fn = isolated(obj, kind, bound, required)
            items.append(judge_one(obj, kind, fn, brief, panel, segment, bound))
    return {"items": items}


def _judge_naive(obj, kind, fn, brief, panel, segment, bound, window="all") -> dict:
    """In-sample metrics on ``window`` of the whole file, at zero latency, nothing checked."""
    from blinded.evaluate import _all_horizons, score_for, scoring_window

    item = _item(obj, kind, bound)
    zero = pd.Timedelta(0)
    # every latency 0: the cancel and response latencies follow the order's
    ex = dataclasses.replace(brief.execution, latency=zero, cancel_latency=None,
                             response_latency=None)  # fmt: skip
    brief = dataclasses.replace(brief, latency=zero, execution=ex)
    try:
        full = panel("test")
        win = scoring_window(brief, full, window)
        out = fn(full)
        values = out if isinstance(out, Legs) else out.to_numpy()
        if kind == "feature":
            targets = feature_returns(full, bound.legs, brief.target.price,
                                      _all_horizons(brief), zero, bound.template)  # fmt: skip
            scores = score_for(bound, values, full.frame["ts"], targets, win, brief)
            metrics = _feature_summary(scores)
        else:
            metrics = _strategy_summary(sim.report(values, full, win, brief, bound.legs[0],
                                                   hedge=bound.hedge))  # fmt: skip
    except Exception as exc:
        return item | {"status": "error", "reason": _failure(exc, obj, kind, bound)}
    return item | {"status": "accepted", "reason": f"in-sample on {win}", "metrics": metrics}


def _required(strat, local: set[str], accepted: dict[str, str]) -> dict[str, str | None]:
    """Where each required feature comes from: the file of the one accepted in the ledger,
    else ``None`` for one defined in the same file."""
    out = {}
    for need in strat.requires:
        if need in accepted:
            out[need] = accepted[need]
        elif need in local:
            out[need] = None
        else:
            raise KeyError(f"required feature {need!r} is not an accepted feature")
    return out


def _item(obj, kind, bound) -> dict:
    return {"kind": kind, "name": obj.name, "hypothesis": getattr(obj, "hypothesis", None),
            "family": bound.family, "legs": bound.label, "edge": bound.edge,
            "archetype": bound.archetype, "group": bound.group}  # fmt: skip


def _tags(archetype: str | None) -> dict:
    """The ledger's ``archetype`` and ``edge`` for an item tagged ``archetype``."""
    if archetype not in REGISTRY:
        return {}
    return {"archetype": archetype, "edge": REGISTRY[archetype].edge}


def _judge_one(obj, kind, fn, brief, panel, segment, bound) -> dict:
    item = _item(obj, kind, bound)
    try:
        return _judge_steps(item, obj, kind, fn, brief, panel, segment, bound)
    except Exception as exc:
        return item | {"status": "error", "reason": _failure(exc, obj, kind, bound)}


def _judge_steps(item, obj, kind, fn, brief, panel, segment, bound) -> dict:
    from blinded.evaluate import _all_horizons, score_for, scoring_window

    seed = brief.stats.seed
    price = brief.target.price
    train = panel("train")
    win = scoring_window(brief, train, "train")
    try:
        tail = brief.stats.block
        bad, full = truncation.precheck(fn, train, win, seed, tail)
        verdict = bad or truncation.check(fn, train, win, seed, full=full, skip=1, tail=tail)
    except Exception as exc:
        return item | {"status": "error", "reason": _failure(exc, obj, kind, bound)}
    if not verdict.ok:
        return item | {"status": verdict.status, "reason": str(verdict.violation)}
    item |= {"status": "accepted", "reason": str(verdict)}
    if kind == "feature":
        h = _all_horizons(brief)
        targets = feature_returns(train, bound.legs, price, h, brief.latency, bound.template)
        scores = score_for(bound, verdict.full, train.frame["ts"], targets, win, brief)
        item["train"] = _feature_summary(scores)
    else:
        rows = sim.report(verdict.full, train, win, brief, bound.legs[0], hedge=bound.hedge)
        item["train"] = _strategy_summary(rows)

    wp = panel(segment)
    vwin = scoring_window(brief, wp, segment)
    try:
        vverdict = truncation.check(fn, wp.cut(vwin.end), vwin, seed, tail=brief.stats.block)
    except Exception as exc:
        # the agent hears only that it failed here: an exception's name and line, or a
        # refusal's line, are the judged code's to choose, so on this panel they could
        # carry its data; the vault keeps the words
        item[segment] = {"truncation": "error", "reason": _blind_failure(exc, segment),
                         "detail": _failure(exc, obj, kind, bound)}  # fmt: skip
        return item
    if not vverdict.ok:
        item[segment] = {"truncation": "failed"}
        reason = f"failed the truncation test on the {segment} panel"
        return item | {"status": vverdict.status, "reason": reason}
    signal = contract.extend(vverdict.full, len(wp))
    g = wp.grid_ns
    i0, i1 = np.searchsorted(g, [vwin.start.value, vwin.end.value])
    step = pd.Timedelta(wp.step_ns, unit="ns")
    block = max(int(brief.stats.block / step), 1)
    draw = BlockDraw.make(i1 - i0, block, brief.stats.n_boot, seed)
    if kind == "feature" and bound.family == "basket":
        primary = brief.target.primary_horizon
        r = feature_returns(wp, bound.legs, price, (primary,), brief.latency, bound.template)
        x = np.column_stack([signal[leg][i0:i1] for leg in bound.legs])
        y = np.column_stack([r[leg][label(primary)].to_numpy()[i0:i1] for leg in bound.legs])
        sc = xs_rank_ic_score(x, y, draw)
        block_ic = _block_xs_ic(x, y, block)
        rows = None
        out = {"truncation": "passed", "value": sc.value, "replicates": sc.replicates,
               "block_ic": block_ic, "blocks": [None] * len(block_ic)}  # fmt: skip
    elif kind == "feature":
        primary = brief.target.primary_horizon
        r = feature_returns(wp, bound.legs, price, (primary,), brief.latency, bound.template)
        x, y = bound.sign * signal[i0:i1], r[label(primary)].to_numpy()[i0:i1]
        sc = rank_ic_score(x, y, draw)
        k = len(x) // block
        block_ic = [_rank_corr(x[j * block : (j + 1) * block], y[j * block : (j + 1) * block])
                    for j in range(k)]  # fmt: skip
        rows = None
        out = {"truncation": "passed", "value": sc.value, "replicates": sc.replicates,
               "block_ic": block_ic, "blocks": [None] * k}  # fmt: skip
    else:
        ex, (inst, *_) = brief.execution, bound.legs
        n = int(i1 - i0)

        def scored(main):
            # one hold is one bet however many blocks it spans: bootstrap in blocks at least as
            # long as the strategy's own holds, and release nothing with fewer than PERIODS
            hold = sim.hold_steps(main)
            own = max(block, math.ceil(hold))
            d = draw if own == block else BlockDraw.make(n, own, brief.stats.n_boot, seed)
            return (sharpe_score(main.net, d, pd.Timedelta("1h") / (own * step),
                                 rules=brief.judge_rules), hold, own)  # fmt: skip

        extra = {}
        if bound.template == "arb":
            # the maker leg's every model its own closed loop, each fill hedged on the hedge
            # venue; the Ladder scores the worse of the two queue models by the hold rule's
            # Sharpe on net per block, both legs, as for market making
            sims = sim_maker.arb_models(signal, wp, vwin, brief, inst, bound.hedge)
            main = sim_maker.worse(sims["RA"], sims["PR"])
            model = main.model
            sc, hold, own = scored(main)
            rows = sim_maker.arb_report(signal, wp, vwin, brief, inst, bound.hedge, sims=sims,
                                        scored=model)  # fmt: skip
            extra = {"fill_model": model,
                     "models": {m: {k: _num(v) for k, v in sims[m].metrics().items()
                                    if k in sim_maker.ARB_KEYS}
                                | {"held": _held(sim_maker.held(sims[m]))}
                                for m in sim_maker.MODELS}}  # fmt: skip
        elif bound.template == "maker":
            # every model is its own closed loop; the Ladder scores the worse of the two
            # queue models by the hold rule's Sharpe (sim_maker.held, the rule the report's
            # bracket prints), with that model's replicates under the phase's shared draws
            sims = sim_maker.models(signal, wp, vwin, brief, inst)
            main = sim_maker.worse(sims["RA"], sims["PR"])
            model = main.model
            sc, hold, own = scored(main)
            rows = sim_maker.report(signal, wp, vwin, brief, inst, sims=sims, scored=model)
            extra = {"fill_model": model,
                     "models": {m: {k: _num(v) for k, v in sims[m].metrics().items()
                                    if k in sim_maker.MODEL_KEYS}
                                | {"held": _held(sim_maker.held(sims[m]))}
                                for m in sim_maker.MODELS}}  # fmt: skip
        else:
            if isinstance(signal, Legs):
                main = sim.simulate_legs(signal, wp, vwin, brief)
            else:
                main = sim.simulate(signal, wp, vwin, inst, price, brief.latency,
                                    ex.fee(inst), ex.min_rebalance, ex.max_position,
                                    brief.stats.block, rules=brief.judge_rules)  # fmt: skip
            sc, hold, own = scored(main)
            rows = sim.report(signal, wp, vwin, brief, inst)
        out = {"truncation": "passed", "value": sc.value, "replicates": sc.replicates,
               "blocks": main.blocks.tolist(), "metrics": _strategy_summary(rows),
               "hold_steps": hold, "hold_hours": hold * step / pd.Timedelta("1h"),
               "block_steps": own, "own_blocks": n // own,
               "releasable": own == block or n // own >= PERIODS, **extra}  # fmt: skip
    # a NaN is never a score: a Sharpe of fewer than two blocks, or of a P&L that never moved,
    # measures nothing, so the ledger holds null and says why, and the Ladder is not asked
    note = _unmeasured(out, kind)
    if note:
        out |= {"value": None, "score_note": note}
    if segment == "test":
        summary = {"score": out["value"]}
        if kind == "strategy":
            summary |= {k: out[k] for k in ("hold_steps", "block_steps", "own_blocks")}
        if note:
            summary["score_note"] = note
        item["test_summary"] = summary | (out.get("metrics") or {})
        item["test_blocks"] = out["blocks"] if kind == "strategy" else None
    item[segment] = out
    return item


def _unmeasured(out: dict, kind: str) -> str | None:
    """Why a validation or test score is no number, or None when it is one."""
    value = out.get("value")
    if value is not None and np.isfinite(value):
        return None
    if kind == "feature":
        return "no rank correlation: the feature or the return did not vary on the window"
    own, blocks = out.get("block_steps"), out.get("own_blocks")
    if blocks is not None and blocks < 2:
        return (f"no Sharpe: its holds give {blocks} block{'s' if blocks != 1 else ''} of "
                f"{own} steps on the window, and a Sharpe needs two")  # fmt: skip
    return "no Sharpe: its net P&L did not vary from block to block"


def _held(rule: dict) -> dict:
    """The hold rule's numbers for the ledger: the Sharpe it scores, null when it measures
    nothing, the hold and the block in steps, and how many blocks."""
    return {"sharpe": _num(rule["sharpe"]), "hold_steps": rule["hold_steps"],
            "block_steps": rule["block_steps"], "blocks": rule["blocks"]}  # fmt: skip


def contract_path(obj) -> str | None:
    import inspect

    try:
        return inspect.getsourcefile(obj.fn)
    except TypeError:
        return None


def _feature_summary(scores: dict) -> dict:
    keep = ("n", "n_eff", "ic", "ic_se_nw", "rank_ic", "rank_ic_se_nw", "rank_ic_se_boot",
            "markout_bps")  # fmt: skip
    horizons = {row["horizon"]: {k: _num(row[k]) for k in keep}
                for row in scores["horizons"].iter_rows(named=True)}  # fmt: skip
    return {"horizons": horizons, "turnover": _num(scores["turnover"]),
            "autocorr": _num(scores["autocorr"])}  # fmt: skip


def _strategy_summary(rows: pl.DataFrame) -> dict:
    """The sim rows by name, and the brief row's net and Sharpe; for a market-making strategy
    also ``fill_model``, the model its brief row ran."""
    out = {}
    for row in rows.iter_rows(named=True):
        out[row["row"]] = {k: _num(v) for k, v in row.items()
                           if k not in ("row", "note", "model")
                           and not (k in sim.ONE_ROW and v is None)}  # fmt: skip
    brief = out.get("brief", {})
    summary = {"rows": out, "sharpe_per_sqrt_hour": brief.get("sharpe_per_sqrt_hour"),
               "net_bps_per_hour": brief.get("net_bps_per_hour")}  # fmt: skip
    if "model" in rows.columns:
        summary["fill_model"] = rows.filter(pl.col("row") == "brief")["model"].item()
    return summary


def _block_xs_ic(x: np.ndarray, y: np.ndarray, block: int) -> list[float]:
    """The mean cross-sectional rank IC inside each whole block of a window, NaN where none of
    its points has one."""
    from blinded.metrics import xs_rank_ic

    c = xs_rank_ic(x, y)
    out = []
    for j in range(len(c) // block):
        part = c[j * block : (j + 1) * block]
        out.append(float(np.mean(part[np.isfinite(part)])) if np.isfinite(part).any() else np.nan)
    return out


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    from blinded.metrics import rank

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(rank(x[ok]), rank(y[ok]))[0, 1])


def _num(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _failure(exc: BaseException, obj, kind: str, bound: contract.Binding) -> str:
    """Why an evaluation failed, as the agent may hear it.

    A :class:`~blinded.contract.ContractError` while running is a wrong shape of
    output; the reason is the rule, written from the binding, since the
    exception could have come from the agent's code with anything in it.
    """
    if isinstance(exc, ContractError):
        return contract.output_rule(kind, bound.legs, bound.family, bound.template)
    if isinstance(exc, isolate.Failed):
        return exc.reason
    return _describe(exc, contract_path(obj))


def _blind_failure(exc: BaseException, segment: str) -> str:
    """Why an evaluation off the train panel failed, as the agent may hear it: one of two
    fixed sentences, nothing the judged code chose."""
    where = {"val": "validation"}.get(segment, segment)
    if isinstance(exc, isolate.Failed) and exc.reason.startswith("the judge refused it"):
        return f"the judge refused it on the {where} panel"
    return f"it raised an error on the {where} panel"


def _declared_tags(source: str) -> dict:
    """``family``, and ``archetype`` and ``edge`` when the file names one, for a row whose
    judgement never ran or never finished."""
    archetype = _declared_archetype(source)
    if archetype is None:
        return {"family": _declared_family(source)}
    return {"family": REGISTRY[archetype].mechanism or "directional", **_tags(archetype)}


def _declared_archetype(source: str) -> str | None:
    """The archetype the file's decorators call, when they all call the same one.

    Counts every call of a registry name, ``bl.lead_lag(...)`` or a bare
    ``lead_lag(...)``, so ``x = bl.lead_lag(...)`` followed by
    ``@x.feature(...)`` counts once for ``lead_lag``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    named = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            word = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if word in REGISTRY:
                named.add(word)
    return named.pop() if len(named) == 1 else None


def _declared_family(source: str) -> str:
    """The family the decorators in ``source`` name, when they all name the same one."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "directional"
    named = {
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "family" and isinstance(kw.value, ast.Constant)
    }
    return named.pop() if len(named) == 1 else "directional"


def _declared_kind(source: str, name: str | None) -> str:
    """``strategy`` when what was judged is decorated as a strategy, else ``feature``.

    For a judgement that never finished, so nothing came back to say which it
    was. With ``name`` only the function of that name counts; without, the
    file counts as a strategy when it declares strategies only. An archetype
    used as the decorator, ``@bl.pairs_trading(...)`` or a name bound to one,
    marks a strategy, as do ``.strategy(...)`` and ``.feature(...)`` of it.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "feature"
    bound = set()  # names assigned an archetype call: x = bl.lead_lag(...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            word = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if word in REGISTRY:
                bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    kinds = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (name is None or node.name == name):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                word = getattr(target, "attr", None) or getattr(target, "id", "")
                if word in ("feature", "strategy"):
                    kinds.add(word)
                elif word in REGISTRY or word in bound:
                    kinds.add("strategy")
    return "strategy" if kinds == {"strategy"} else "feature"


def _describe(exc: BaseException, source: str | None) -> str:
    """Exception type and the line in the submitted file, never the message: it may hold data."""
    frames = traceback.extract_tb(exc.__traceback__)
    mine = [f for f in frames if source is not None and Path(f.filename) == Path(source)]
    where = f" at line {mine[-1].lineno}" if mine else ""
    return f"{type(exc).__name__}{where}"


def _rejected(tid: int, kind: str, name: str, status: str, reason: str) -> dict:
    return {"trial_id": tid, "kind": kind, "name": name, "status": status, "reason": reason}


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)
