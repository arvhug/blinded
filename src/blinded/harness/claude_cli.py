"""The ``claude -p`` harness: one headless Claude Code session per iteration.

The command line is :func:`blinded.sandbox.command`, run from the workspace
with blinded's ``bin`` first on ``PATH``, in its own process group so that a
session past its wall time is stopped with everything it started. The session
streams JSON events (``--output-format stream-json --verbose``) to the
transcript file; the last event, ``type: result``, carries ``total_cost_usd``,
``num_turns``, ``session_id``, ``permission_denials`` and, because the
command passes ``--json-schema``, the agent's summary as ``structured_output``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from blinded import sandbox
from blinded.harness.base import Harness, IterationResult, SessionLimits
from blinded.transcript import events
from blinded.vault import RunDir


class ClaudeCLI(Harness):
    """Run sessions with the ``claude`` binary on ``PATH``.

    Parameters
    ----------
    run
        The run directory; the sandbox settings come from ``run/agent/``.
    schema
        JSON Schema for the session summary, as a dict.
    system_append
        File appended to Claude Code's system prompt.
    python
        The interpreter whose ``bin`` directory holds ``blinded``; default the current one.
    """

    name = "claude-cli"

    def __init__(
        self,
        run: RunDir,
        schema: dict | None = None,
        system_append: Path | None = None,
        python: str | None = None,
        binary: str = "claude",
    ):
        self.rundir = run
        self.schema = schema
        self.system_append = system_append
        self.python = python
        self.binary = binary

    def version(self) -> str | None:
        try:
            out = subprocess.run([self.binary, "--version"], capture_output=True, text=True,
                                 timeout=30)  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() or None

    def command(self, prompt: str, limits: SessionLimits) -> list[str]:
        argv = sandbox.command(
            self.rundir,
            prompt,
            model=limits.model,
            max_turns=limits.max_turns,
            budget_usd=limits.budget_usd,
            output="stream-json",
            effort=limits.effort,
            schema=None if self.schema is None else json.dumps(self.schema),
            system_append=self.system_append,
        )
        return [self.binary, *argv[1:]]

    def run(
        self, prompt: str, workspace: Path, limits: SessionLimits, log: Path | None = None
    ) -> IterationResult:
        log = log or self.rundir.driver / "logs" / f"session-{int(time.time())}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        timed_out = False
        with log.open("w") as out, log.with_suffix(".stderr").open("w") as err:
            proc = subprocess.Popen(
                self.command(prompt, limits),
                cwd=workspace,
                env=sandbox.environment(self.python),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=limits.wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop(proc)
        result = parse(log)
        result.wall_seconds = time.monotonic() - started
        if timed_out and result.outcome == "no_result":
            result.outcome = "timeout"
        return result


def stop(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """Terminate a session's process group, then kill it if it is still there."""
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


def parse(path: Path) -> IterationResult:
    """Read a stream-json transcript into an :class:`IterationResult`.

    Without a ``result`` event (the session was killed or crashed) the
    outcome is ``no_result``, the cost is unknown and recorded as 0, and the
    session id comes from the first event that has one.
    """
    stream = events(path)
    final = next((e for e in reversed(stream) if e.get("type") == "result"), None)
    session = next((e["session_id"] for e in stream if e.get("session_id")), None)
    if final is None:
        return IterationResult(outcome="no_result", session_id=session, transcript=Path(path))
    summary = final.get("structured_output")
    return IterationResult(
        outcome=str(final.get("subtype") or "unknown"),
        session_id=final.get("session_id") or session,
        cost_usd=float(final.get("total_cost_usd") or 0.0),
        turns=int(final.get("num_turns") or 0),
        summary=summary if isinstance(summary, dict) else None,
        permission_denials=list(final.get("permission_denials") or []),
        transcript=Path(path),
        is_error=bool(final.get("is_error")),
    )
