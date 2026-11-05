"""The loop: one fresh agent session per iteration, phases, stopping rules and resume.

``blinded run BRIEF --run-dir DIR`` builds the run directory, starts the judge
daemon, and runs sessions until a phase rule or a hard stop ends the run:

1. **Features.** Ends once 60% of ``max_iterations`` (rounded down, at
   least 1) have run, or when every family is closed. A family closes after
   ``feature_streak_sessions`` (3) of its sessions in a row without a release
   of one of its features, or when the agent returns ``phase_done`` in two
   of its sessions in a row.
2. **Strategies.** Ends when every family has had
   ``strategy_streak_sessions`` (3) of its sessions in a row without a
   strategy release, or at the budget.
3. **Finalise.** One session that names the final strategy. The last of
   ``max_iterations`` is kept for it. A trial that was never released is
   overridden with the released strategy whose released best is highest,
   in any family; with none, the final strategy is flat.
4. **Test.** The driver scores the final strategy on test, once.

Each session of a phase is for one group: of the groups still open, the one
with the fewest sessions so far for its ``share``, ties to the first in the
brief. A group is a family, or in a brief with ``archetypes`` an edge, whose
runnable archetypes share its sessions, streaks and Ladders. A brief with
neither block has one group, and every session is its.

Hard stops, checked before every session: any probe (the run halts and
nothing is scored on test), ``wall_hours``, ``max_usd``, ``max_iterations``,
and three sessions in a row that add no ledger row. After any hard stop but
a probe, the driver finalises without the agent: the last released strategy,
or flat.

In naive mode there is no Ladder, so the two release rules are
off; the agent's own pick is final, and the driver scores it on the locked
test with the brief's latency and fees.

All state is on disk. Every count the rules use (iterations, spend, wall
time, the runs of sessions without a release, the run of empty sessions) is
recomputed from the ledger, so ``--resume`` after a crash continues from the
ledger; the crashed session's trials count, and its cost, if it never wrote
a result, counts as 0. ``driver/state.json`` keeps the phase and the family
of each iteration and the outcome.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

from blinded import prompts, sandbox, transcript, vault
from blinded.harness.base import Harness, IterationResult, SessionLimits
from blinded.ledger import now
from blinded.vault import RunDir

STUCK = 3
PHASE_DONE = 2
_KIND = {"features": "feature", "strategies": "strategy"}


@dataclass(frozen=True)
class Budget:
    """How long and how much: the brief's ``budget`` block, overridden by command-line flags.

    ``iteration_usd`` and ``iteration_minutes`` cap one session; ``max_usd``
    and ``wall_hours`` the run. Spend is the sessions' own ``total_cost_usd``.
    """

    wall_hours: float = 8.0
    max_iterations: int = 40
    max_usd: float = 60.0
    model: str = "opus"
    effort: str | None = "high"
    max_turns: int = 60
    iteration_usd: float = 3.0
    iteration_minutes: float = 45.0
    feature_streak_sessions: int = 3
    strategy_streak_sessions: int = 3

    def __post_init__(self):
        for name in ("feature_streak_sessions", "strategy_streak_sessions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"budget.{name} must be a whole number of sessions, at least 1")

    @classmethod
    def from_raw(cls, raw: dict | None, **overrides) -> Budget:
        names = {f.name for f in fields(cls)}
        given = {k: v for k, v in (raw or {}).items() if k in names}
        given |= {k: v for k, v in overrides.items() if v is not None and k in names}
        return cls(**given)

    def streak(self, phase: str) -> int:
        """Sessions in a row without a release that close a family's phase."""
        if phase == "features":
            return self.feature_streak_sessions
        return self.strategy_streak_sessions

    @property
    def feature_iterations(self) -> int:
        """Iterations after which the features phase ends: 60% of the run, rounded down."""
        return max(1, math.floor(0.6 * self.max_iterations))


@dataclass(frozen=True)
class Standing:
    """One group in one phase: its sessions, its run of sessions without a release of the
    phase's kind, its run of ``phase_done``, its released best, and why it closed.

    ``family`` is the group's name: a family, or an edge in a run with archetypes.
    """

    family: str
    share: float
    sessions: int
    streak: int
    phase_done: int
    best: float | None
    closed: str | None


@dataclass(frozen=True)
class Status:
    """Where the run stands, recomputed from the ledger before every session.

    ``family`` is the family the next session is for, and ``non_releases``
    and ``phase_done`` are that family's runs; ``standings`` has every
    family's.
    """

    iterations: int
    spent_usd: float
    wall_seconds: float
    phase: str
    non_releases: int
    phase_done: int
    empty: int
    probes: int
    phase_end: str | None = None
    family: str | None = None
    standings: tuple[Standing, ...] = field(default_factory=tuple)


def start(brief: str | Path, run_dir: str | Path, mode: str = "protected") -> RunDir:
    """Build a new run directory for the driver (``blinded init`` plus the mode)."""
    return vault.init(brief, run_dir, mode=mode)


class Driver:
    """Run the loop for one run directory.

    Parameters
    ----------
    run
        An initialised run directory.
    budget
        Limits for the run and for each session.
    harness
        Runs the sessions; default :class:`~blinded.harness.claude_cli.ClaudeCLI`.
    python
        Interpreter for the daemon and for ``blinded`` on the agent's ``PATH``.
    """

    def __init__(
        self,
        run: RunDir,
        budget: Budget | None = None,
        harness: Harness | None = None,
        python: str | None = None,
    ):
        self.rundir = run
        self.mode = run.mode
        self.python = python or sys.executable
        self.state_path = run.driver / "state.json"
        state = self._read_state()
        if budget is None:
            budget = Budget.from_raw(
                state.get("budget") if state else run.brief().raw.get("budget")
            )
        self.budget = budget
        if harness is None:
            from blinded.harness.claude_cli import ClaudeCLI

            harness = ClaudeCLI(
                run, prompts.SCHEMA, run.root / "agent" / "system_append.md", self.python
            )
        self.harness = harness
        self.state = state or {}
        brief = run.brief()
        self.groups = brief.groups
        self.archetypes = brief.has_archetypes
        self.has_families = brief.has_families or brief.has_archetypes
        self.dropped = run.menu().dropped_line if self.archetypes else None

    # -- the loop ---------------------------------------------------------------

    def run(self, resume: bool = False) -> dict:
        """Run until the end of the finalise session or a hard stop; returns the final state."""
        self._begin(resume)
        if self.state.get("status") in ("finished", "halted"):
            return self.state
        self._recover()
        stop = None
        with self._daemon():
            while True:
                status = self.status()
                if self._last_phase(status) == "finalise":
                    stop = "finalised"
                    break
                stop = self._hard_stop(status)
                if stop:
                    break
                phase = status.phase
                self._iterate(status.iterations + 1, phase, status)
                if phase == "finalise":
                    stop = "finalised"
                    break
        return self._finish(stop)

    def status(self) -> Status:
        """Counts for the stopping rules, all from the ledger."""
        ledger = self.rundir.ledger()
        try:
            iters = ledger.iterations(self.rundir.run_id)
            trials = ledger.trials()
        finally:
            ledger.close()
        phases = {int(k): v for k, v in self.state.get("phases", {}).items()}
        done = len(iters)
        spent = sum(r["cost_usd_est"] or 0.0 for r in iters)
        wall = sum(_seconds(r["started_at"], r["ended_at"]) for r in iters)
        probes = sum(1 for t in trials if t["kind"] == "probe")
        working = [r["iteration"] for r in iters if phases.get(r["iteration"]) != "finalise"]
        per_iter = {n: 0 for n in working}
        for t in trials:
            if t["kind"] != "probe" and t["iteration"] in per_iter:
                per_iter[t["iteration"]] += 1
        empty = _trailing(working, lambda n: per_iter[n] == 0)
        summaries = {r["iteration"]: _summary(r) for r in iters}
        phase, end, standings = self._next_phase(done, phases, summaries, trials)
        family = _assign(standings)
        mine = next((st for st in standings if st.family == family), None)
        streak, flagged = (mine.streak, mine.phase_done) if mine else (0, 0)
        return Status(done, spent, wall, phase, streak, flagged, empty, probes, end, family,
                      tuple(standings))  # fmt: skip

    def _next_phase(self, done, phases, summaries, trials):
        """The phase of the next session, why the last one ended, and each family's standing."""
        if done >= self.budget.max_iterations - 1:
            return "finalise", "the last iteration is kept for naming the final strategy", []
        phase = phases.get(done, "features") if done else "features"
        end = None
        standings = self._standings(phase, phases, done, summaries, trials)
        if phase == "features":
            if done >= self.budget.feature_iterations:
                end = f"60% of {self.budget.max_iterations} iterations ran"
            elif all(st.closed for st in standings):
                end = self._all_closed(standings)
            if end:
                phase = "strategies"
                standings = self._standings(phase, phases, done, summaries, trials)
        if phase == "strategies" and all(st.closed for st in standings):
            end, phase, standings = self._all_closed(standings), "finalise", []
        return phase, end, standings

    def _standings(self, phase, phases, done, summaries, trials) -> list[Standing]:
        """Every group's sessions, runs and released best in ``phase``, from the ledger.

        A session counts against a group's run when no trial of the phase's
        kind and that group was released in it, whatever else it submitted.
        The streak rule applies in protected mode; ``phase_done`` closes a
        group only in the features phase.
        """
        kind = _KIND.get(phase)
        if kind is None:
            return []
        assigned = self.state.get("families", {})
        first = self.groups[0].name
        held = [n for n in sorted(phases) if phases[n] == phase and n <= done]
        column = "edge" if self.archetypes else "family"
        released = {}
        for t in trials:
            if t["kind"] == kind and t["released_bool"]:
                released.setdefault(t[column], {})[t["iteration"]] = t["ladder_best_after"]
        out = []
        for fam in self.groups:
            mine = [n for n in held if assigned.get(str(n), first) == fam.name]
            hits = released.get(fam.name, {})
            streak = _trailing(mine, lambda n, hits=hits: n not in hits)
            flagged = _trailing(mine, lambda n: _phase_done(summaries.get(n, {})))
            limit = self.budget.streak(phase)
            closed = None
            if self.mode == "protected" and streak >= limit:
                closed = f"{limit} sessions in a row without a {kind} release"
            elif phase == "features" and flagged >= PHASE_DONE:
                closed = "the agent returned phase_done in two sessions in a row"
            best = max((v for v in hits.values() if v is not None), default=None)
            out.append(Standing(fam.name, fam.share, len(mine), streak, flagged, best, closed))
        return out

    def _all_closed(self, standings: list[Standing]) -> str:
        if len(standings) == 1:
            return standings[0].closed
        names = {g.name: g.title for g in self.groups} if self.archetypes else {}
        closed = "; ".join(f"{names.get(st.family, st.family)}, {st.closed}" for st in standings)
        return f"every {'edge' if self.archetypes else 'family'} is closed: {closed}"

    def _last_phase(self, status: Status) -> str | None:
        return self.state.get("phases", {}).get(str(status.iterations))

    def _hard_stop(self, status: Status) -> str | None:
        b = self.budget
        if status.probes:
            return "probe"
        if b.max_usd - status.spent_usd < 0.01:
            return "max_usd"
        if status.wall_seconds >= b.wall_hours * 3600:
            return "wall_hours"
        if status.iterations >= b.max_iterations:
            return "max_iterations"
        if status.empty >= STUCK:
            return "stuck"
        return None

    def _iterate(self, n: int, phase: str, status: Status) -> IterationResult:
        b = self.budget
        self.state.setdefault("phases", {})[str(n)] = phase
        if status.family is not None:
            self.state.setdefault("families", {})[str(n)] = status.family
        if status.phase_end and phase != self.state.get("phase"):
            self.state.setdefault("phase_ends", []).append(
                {"after_iteration": n - 1, "to": phase, "why": status.phase_end}
            )
        self.state["phase"] = phase
        self._save()
        (self.rundir.vault / "iteration").write_text(f"{n}\n")
        self._write_budget(n, status)
        log = self.rundir.driver / "logs" / f"iter_{n:03d}.jsonl"
        ledger = self.rundir.ledger()
        ledger.start_iteration(self.rundir.run_id, n, str(log))
        ledger.close()
        limits = SessionLimits(
            model=b.model,
            effort=b.effort,
            max_turns=b.max_turns,
            budget_usd=round(min(b.iteration_usd, b.max_usd - status.spent_usd), 2),
            wall_seconds=min(b.iteration_minutes * 60, b.wall_hours * 3600 - status.wall_seconds),
        )
        family = self._family(status.family) if self.has_families else None
        prompt = prompts.iteration(phase, self.mode, family)
        result = self.harness.run(prompt, self.rundir.workspace, limits, log=log)
        self._drain()
        self._settle(n, phase, result)
        return result

    def _settle(self, n: int, phase: str, result: IterationResult, ended_at: str | None = None):
        """Record a finished session: probes, sandbox events, cost, turns and summary."""
        others: list[dict] = []
        denials = {
            "permission_denials": result.permission_denials,
            "session_id": result.session_id,
        }
        probes = sandbox.record_denials(self.rundir, denials, others)
        refusals = []
        if result.transcript is not None:
            added, refusals = transcript.record(
                self.rundir, transcript.events(result.transcript), iteration=n
            )
            probes += added
        events = [{"iteration": n, "source": "denial", **d} for d in others]
        events += [{"iteration": n, "source": "sandbox", **r.to_dict()} for r in refusals]
        if events:
            with (self.rundir.driver / "events.jsonl").open("a") as fh:
                fh.writelines(json.dumps(e) + "\n" for e in events)
        summary = {
            "phase": phase,
            "outcome": result.outcome,
            "summary": result.summary,
            "probes": probes,
            "refusals": len(refusals),
        }
        ledger = self.rundir.ledger()
        ledger.finish_iteration(
            self.rundir.run_id,
            n,
            session_id=result.session_id,
            ended_at=ended_at or now(),
            cost_usd_est=result.cost_usd,
            turns=result.turns,
            summary_json=summary,
        )
        self.rundir.write_ledger_md(ledger)
        ledger.close()

    def _recover(self) -> None:
        """Close the books on a session that was running when the driver stopped."""
        from blinded.harness.claude_cli import parse

        ledger = self.rundir.ledger()
        open_rows = [r for r in ledger.iterations(self.rundir.run_id) if r["ended_at"] is None]
        ledger.close()
        for row in open_rows:
            path = Path(row["transcript_path"])
            result = parse(path)
            if result.outcome == "no_result":
                result.outcome = "crashed"
            ended = row["started_at"]
            if path.is_file():
                ended = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(
                    timespec="milliseconds"
                )
            phase = self.state.get("phases", {}).get(str(row["iteration"]), "features")
            self._settle(row["iteration"], phase, result, ended_at=ended)

    # -- finishing --------------------------------------------------------------

    def _finish(self, stop: str | None) -> dict:
        from blinded.daemon import score_test
        from blinded.ledger import AlreadyScored

        self.state["stop"] = stop
        (self.rundir.vault / "iteration").unlink(missing_ok=True)
        final = self.state.get("final") or self._final_choice(stop)
        self.state["final"] = final
        self._save()
        ledger = self.rundir.ledger()
        scored = ledger.test_score(self.rundir.run_id)
        ledger.close()
        if stop == "probe":
            self.state["status"] = "halted"
        elif final.get("trial_id") is not None and scored is None:
            try:
                if self.mode == "naive":
                    rescore(self.rundir, final["trial_id"])
                else:
                    score_test(self.rundir, final["trial_id"])
            except AlreadyScored:
                pass
            except (RuntimeError, ValueError) as exc:
                self.state["test_error"] = f"{type(exc).__name__}: {exc}"[:300]
        if stop != "probe":
            self.state["status"] = "finished"
        ledger = self.rundir.ledger()
        ledger.update_run(
            self.rundir.run_id, finished_at=now(), final_trial_id=final.get("trial_id")
        )
        ledger.close()
        self._save()
        return self.state

    def _final_choice(self, stop: str | None) -> dict:
        """The final strategy: the agent's pick if it stands, else the driver's."""
        ledger = self.rundir.ledger()
        try:
            iters = ledger.iterations(self.rundir.run_id)
            strategies = ledger.trials("strategy")
        finally:
            ledger.close()
        phases = self.state.get("phases", {})
        finals = [_summary(r) for r in iters if phases.get(str(r["iteration"])) == "finalise"]
        named = (
            _trial_id((finals[-1].get("summary") or {}).get("final_trial_id")) if finals else None
        )
        by_id = {t["trial_id"]: t for t in strategies}
        if self.mode == "naive":
            return _naive_choice(named, by_id, finals)
        released = [t for t in strategies if t["released_bool"]]
        choice = {"agent_pick": named, "finalised_by": "agent" if finals else "driver"}
        if named is not None and named in {t["trial_id"] for t in released}:
            return choice | {"trial_id": named, "override": None}
        if not finals:
            why = f"no finalise session ({stop})"
        elif named is None:
            why = "the agent named no trial"
        elif named not in by_id:
            why = f"trial {named} is not a strategy trial"
        else:
            why = f"trial {named} was never released"
        if released:
            # the Ladder only releases improvements, so within a family the last is the best
            top = max(released, key=lambda t: (_or_lowest(t["ladder_best_after"]), t["trial_id"]))
            best = top["trial_id"]
            why += f"; the released strategy with the best released score, trial {best}, is final"
            return choice | {"trial_id": best, "override": why}
        why += "; no strategy was ever released, so the final strategy is flat"
        return choice | {"trial_id": None, "override": why}

    # -- files ------------------------------------------------------------------

    def _begin(self, resume: bool) -> None:
        if self.state and not resume:
            raise FileExistsError(
                f"{self.rundir.root} already has a run; "
                f"continue it with --resume {self.rundir.run_id}"
            )
        if resume and not self.state:
            raise FileNotFoundError(f"{self.rundir.root} has no driver state to resume")
        agent = self.rundir.root / "agent"
        (agent / "system_append.md").write_text(prompts.SYSTEM_APPEND)
        (agent / "iteration_schema.json").write_text(json.dumps(prompts.SCHEMA, indent=2) + "\n")
        notes = self.rundir.workspace / "NOTES.md"
        if not notes.exists():
            notes.write_text("# NOTES\n")
        if not self.state:
            self.state = {
                "run_id": self.rundir.run_id,
                "mode": self.mode,
                "status": "running",
                "budget": asdict(self.budget),
                "harness": self.harness.name,
                "started_at": now(),
                "phases": {},
            }
            ledger = self.rundir.ledger()
            ledger.update_run(
                self.rundir.run_id,
                budget_json=asdict(self.budget),
                harness=self.harness.name,
                claude_version=self.harness.version(),
                model=self.budget.model,
            )
            ledger.close()
        else:
            self.state["budget"] = asdict(self.budget)
            self.state.setdefault("resumed_at", []).append(now())
        self._save()

    def _family(self, name: str | None):
        """The session's family, or its group (an edge) in a run with archetypes."""
        if self.archetypes:
            return next((g for g in self.groups if g.name == name), None)
        return self.rundir.brief().family(name)

    def _write_budget(self, n: int, status: Status) -> None:
        b = self.budget
        kind = _KIND.get(status.phase)
        lines = [
            "# BUDGET",
            "",
            f"Iteration {n} of {b.max_iterations}; the last names the final strategy.",
            f"Phase: {status.phase.upper()}.",
        ]
        if self.archetypes and status.family:
            lines.append(f"This session is for the {self._family(status.family).title} edge.")
        elif self.has_families and status.family:
            lines.append(f"This session is for the {status.family} family.")
        lines += [
            f"Spent: ${status.spent_usd:.2f} of ${b.max_usd:.2f} (the sessions' own accounting).",
            f"Wall time used: {_hm(status.wall_seconds)} of {_hm(b.wall_hours * 3600)}.",
        ]
        limit = b.streak(status.phase)
        if self.has_families and kind:
            lines += ["", self._standings_table(status, limit), ""]
            if self.dropped:
                lines += [self.dropped[0].upper() + self.dropped[1:] + ".", ""]
        elif self.mode == "protected" and kind:
            lines.append(
                f"Sessions in a row without a {kind} release: {status.non_releases} "
                f"(the phase ends at {limit})."
            )
        if status.phase == "features":
            unit, one = ("edge", "an edge") if self.archetypes else ("family", "a family")
            each = (
                f"every {unit} is closed; {one} closes after `phase_done` in two of its "
                "sessions in a row"
                if self.has_families
                else f"after `phase_done` in two sessions in a row (now {status.phase_done})"
            )
            lines.append(
                f"The features phase ends after iteration {b.feature_iterations} at the latest, "
                f"or {each}."
            )
        for end in self.state.get("phase_ends", []):
            moved = f"The run moved to {end['to']} after iteration {end['after_iteration']}"
            lines.append(f"{moved}: {end['why']}.")
        (self.rundir.workspace / "BUDGET.md").write_text("\n".join(lines) + "\n")

    def _standings_table(self, status: Status, limit: int) -> str:
        protected = self.mode == "protected"
        head = ["edge", "members"] if self.archetypes else ["family"]
        head += ["share", "sessions this phase"]
        head += ["without a release in a row", "released best"] if protected else []
        head += ["open"]
        rows = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
        for st in status.standings:
            cells = [st.family]
            if self.archetypes:
                group = self._family(st.family)
                cells = [group.title, ", ".join(group.members)]
            cells += [f"{st.share:g}", str(st.sessions)]
            if protected:
                best = "none yet" if st.best is None else f"{st.best:g}"
                cells += [f"{st.streak} of {limit}", best]
            cells.append(f"closed: {st.closed}" if st.closed else "open")
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    def _read_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        self.rundir.driver.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(self.state_path)

    # -- the daemon -------------------------------------------------------------

    @contextlib.contextmanager
    def _daemon(self):
        pid_file = self.rundir.driver / "daemon.pid"
        _stop_pid(pid_file)
        log = (self.rundir.driver / "daemon.log").open("a")
        proc = subprocess.Popen(
            [self.python, "-m", "blinded.cli", "daemon", str(self.rundir.root)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid_file.write_text(f"{proc.pid}\n")
        self._proc = proc
        try:
            yield proc
        finally:
            _stop_group(proc)
            pid_file.unlink(missing_ok=True)
            log.close()

    def _drain(self, timeout: float = 900.0) -> None:
        """Wait until the daemon has answered every request the session left in the inbox."""
        deadline = time.monotonic() + timeout
        while any(self.rundir.inbox.glob("*.json")) and time.monotonic() < deadline:
            if getattr(self, "_proc", None) is not None and self._proc.poll() is not None:
                return
            time.sleep(0.2)


def rescore(run: RunDir, trial_id: int) -> dict:
    """Score a naive run's final pick as the protected judge would, on the locked test, once.

    The code goes through the protected static scan first; then
    :func:`blinded.daemon.score_test` runs the truncation test and scores the
    test window with the brief's latency and fees. A rejection is the test
    result.
    """
    from blinded import contract
    from blinded.daemon import score_test

    ledger = run.ledger()
    row = ledger.trial(trial_id)
    try:
        static = contract.scan(Path(row["code_path"]).read_text(), protected=True)
        if static.errors:
            metrics = {"status": "rejected_static", "reason": "; ".join(static.errors)[:200]}
            ledger.record_test(run.run_id, trial_id, metrics)
            return metrics
    finally:
        ledger.close()
    return score_test(run, trial_id)


def _naive_choice(named: int | None, by_id: dict, finals: list[dict]) -> dict:
    choice = {"agent_pick": named, "finalised_by": "agent" if finals else "driver"}
    if named is not None and named in by_id and by_id[named]["status"] == "accepted":
        return choice | {"trial_id": named, "override": None}
    best, best_sr = None, -math.inf
    for tid, row in by_id.items():
        sr = (json.loads(row["train_metrics_json"] or "{}")).get("sharpe_per_sqrt_hour")
        if row["status"] == "accepted" and sr is not None and sr > best_sr:
            best, best_sr = tid, sr
    if best is None:
        return choice | {"trial_id": None, "override": "no strategy was submitted"}
    return choice | {"trial_id": best, "override": f"the best claimed Sharpe, trial {best}"}


def _assign(standings: list[Standing]) -> str | None:
    """The open family with the fewest sessions for its share; ties to the first."""
    open_ = [st for st in standings if not st.closed]
    if not open_:
        return None
    return min(open_, key=lambda st: round(st.sessions / st.share, 9)).family


def _or_lowest(value: float | None) -> float:
    return -math.inf if value is None else value


def _phase_done(summary: dict) -> bool:
    return bool((summary.get("summary") or {}).get("phase_done"))


def _summary(row) -> dict:
    try:
        return json.loads(row["summary_json"] or "{}")
    except json.JSONDecodeError:
        return {}


def _trial_id(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _trailing(items, pred) -> int:
    count = 0
    for item in reversed(list(items)):
        if not pred(item):
            break
        count += 1
    return count


def _seconds(start: str | None, end: str | None) -> float:
    if not start:
        return 0.0
    a = datetime.fromisoformat(start)
    b = datetime.fromisoformat(end) if end else datetime.now(UTC)
    return max((b - a).total_seconds(), 0.0)


def _hm(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}:{minutes % 60:02d}"


def _stop_group(proc: subprocess.Popen, grace: float = 10.0) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def _stop_pid(pid_file: Path) -> None:
    """Stop a daemon a crashed driver left running, so two never share an inbox."""
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGTERM)
    pid_file.unlink(missing_ok=True)
