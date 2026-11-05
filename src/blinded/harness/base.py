"""The interface between the driver and whatever runs the agent.

The driver needs one thing from a harness: run a fresh session with a prompt
in the workspace, within limits, and say what it cost and what the agent
reported. The judge never sees the harness, so ``claude -p`` today and the
Agent SDK later are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SessionLimits:
    """Limits on one session: the model, the turns, the spend and the wall time."""

    model: str = "opus"
    effort: str | None = "high"
    max_turns: int = 60
    budget_usd: float = 3.0
    wall_seconds: float = 1800.0


@dataclass
class IterationResult:
    """What one session did, as far as the driver needs to know.

    ``outcome`` is the session's own result subtype (``success``,
    ``error_max_turns``, ``error_max_budget_usd``, ...) or, when there was no
    result, ``timeout`` or ``no_result``. ``summary`` is the structured
    output the iteration schema asks for, or None if the session ended
    without one.
    """

    outcome: str
    session_id: str | None = None
    cost_usd: float = 0.0
    turns: int = 0
    summary: dict | None = None
    permission_denials: list[dict] = field(default_factory=list)
    transcript: Path | None = None
    wall_seconds: float = 0.0
    is_error: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome == "success" and not self.is_error


class Harness(ABC):
    """Runs one agent session per call."""

    name = "harness"

    def version(self) -> str | None:
        """The version of what runs the sessions, for the ledger; None if unknown."""
        return None

    @abstractmethod
    def run(
        self, prompt: str, workspace: Path, limits: SessionLimits, log: Path | None = None
    ) -> IterationResult:
        """Run one session in ``workspace`` and return its result.

        ``log`` is where the session's transcript goes, when the harness keeps one.
        """
