"""A scripted stand-in for the agent, for driving the real driver in tests and smoke runs.

It implements the :class:`~blinded.harness.base.Harness` interface. Each
session runs the next step of a script: it writes feature or strategy files
into the workspace and submits them through ``blinded submit`` against the
running daemon, exactly as the agent would, but with no model. It writes a
stream-json transcript the driver can read, and can inject a sandbox probe or
a mid-session crash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from blinded.client import submit
from blinded.harness.base import Harness, IterationResult, SessionLimits


class CrashInjected(RuntimeError):
    """Raised by a step marked ``crash`` to simulate the driver dying mid-iteration."""


@dataclass
class Step:
    """One scripted session.

    ``files`` are ``(relative_path, source)`` written before submitting;
    ``submit`` are ``(path, name)`` sent to the judge. ``summary`` is the
    structured output. ``probe`` adds a vault-reaching refusal to the
    transcript; ``denials`` are the result's ``permission_denials``, with
    ``{session}`` in a command replaced by the session's id; ``crash`` raises
    after submitting.
    """

    files: list[tuple[str, str]] = field(default_factory=list)
    submit: list[tuple[str, str | None]] = field(default_factory=list)
    window: str | None = None
    summary: dict | None = None
    probe: bool = False
    denials: list[dict] = field(default_factory=list)
    crash: bool = False
    cost: float = 0.01
    outcome: str = "success"


class FakeHarness(Harness):
    name = "fake"

    def __init__(self, run, steps: list[Step], submit_timeout: float = 120.0):
        self.rundir = run
        self.steps = list(steps)
        self.calls = 0
        self.submitted: list[dict] = []
        self.submit_timeout = submit_timeout

    def version(self) -> str:
        return "fake-harness"

    def run(self, prompt, workspace, limits: SessionLimits, log: Path | None = None):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        session = f"fake-{self.calls}"
        for rel, source in step.files:
            path = Path(workspace) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        responses = []
        for rel, name in step.submit:
            response = submit(Path(workspace) / rel, name, self.submit_timeout,
                              judge=Path(workspace) / ".judge", window=step.window)  # fmt: skip
            responses.append(response)
            self.submitted.append(response)
        denials = json.loads(json.dumps(step.denials).replace("{session}", session))
        events = _transcript(session, prompt, responses, step, self.rundir.vault, denials)
        if log is not None:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            Path(log).write_text("\n".join(json.dumps(e) for e in events) + "\n")
        if step.crash:
            raise CrashInjected(f"crash after session {session}")
        return IterationResult(
            outcome=step.outcome,
            session_id=session,
            cost_usd=step.cost,
            turns=1 + len(step.submit),
            summary=step.summary,
            permission_denials=denials,
            transcript=Path(log) if log else None,
        )


def _transcript(session, prompt, responses, step, vault, denials=()) -> list[dict]:
    events = [{"type": "system", "subtype": "init", "session_id": session, "cwd": "."}]
    for response in responses:
        for trial in response.get("trials", []):
            line = f"submitted {trial['name']}: {trial['status']}"
            block = [{"type": "tool_result", "tool_use_id": f"tu-{trial['trial_id']}",
                      "content": line}]  # fmt: skip
            events.append({"type": "user", "session_id": session, "message": {"content": block}})
    if step.probe:
        # a script read the vault; the OS sandbox refused it inside the command
        line = f"val PermissionError [Errno 1] Operation not permitted: '{vault / 'val.parquet'}'"
        events.append({"type": "user", "session_id": session, "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu-probe", "content": line}]}})  # fmt: skip
    result = {"type": "result", "subtype": step.outcome, "is_error": step.outcome != "success",
              "num_turns": 1 + len(step.submit), "total_cost_usd": step.cost,
              "session_id": session, "permission_denials": list(denials)}  # fmt: skip
    if step.summary is not None:
        result["structured_output"] = step.summary
    events.append(result)
    return events


# -- source the steps write -------------------------------------------------

BASIS = """
import blinded as bl
from blinded import features as F


@bl.feature("perp leads spot; the basis mean-reverts")
def basis(panel):
    b = F.basis("XRPUSDT_PERP", "XRPUSDT")
    return b - F.roll_mean(b, "5m")
"""

WEAK = """
import blinded as bl
import polars as pl
from blinded import features as F


@bl.feature("spot bid size, lagged")
def weak(panel):
    return F.lag(pl.col("XRPUSDT.bid_qty"), 1)
"""

LEAK = """
import blinded as bl
import polars as pl


@bl.feature("centred mean of the mid")
def leaky(panel):
    return pl.col("XRPUSDT.mid").rolling_mean(50, center=True)
"""

STRATEGY = """
import blinded as bl


@bl.strategy(requires=["basis"])
def basis_threshold(panel, features):
    return bl.threshold(features["basis"], z_in=2.0, z_out=0.5, every="2m")
"""

SPREAD_Z = """
import blinded as bl
from blinded import features as F


@bl.feature(
    "the perp-spot log spread far from its walk-forward mean comes back",
    family="pairs",
    pair=("XRPUSDT_PERP", "XRPUSDT"),
)
def spread_reverts(panel):
    return -F.spread_z("XRPUSDT_PERP", "XRPUSDT", "2m", "2m")
"""

PAIRS = """
import blinded as bl


@bl.strategy(family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
def perp_spot_pairs(panel, features):
    return bl.pairs("XRPUSDT_PERP", "XRPUSDT", z_in=2.0, z_out=0.5, window="2m", every="2m")
"""


def ideas(*names: str) -> dict:
    """A summary naming ideas submitted this session."""
    return {"ideas": [{"name": n, "trial_ids": [], "rationale": n} for n in names], "next": "more"}
