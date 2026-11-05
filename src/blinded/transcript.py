"""Sandbox refusals in a session transcript, and the probes among them.

When the OS sandbox refuses a read, the refusal happens inside the command:
the process gets ``EPERM``, prints "Operation not permitted" and carries on.
Nothing reaches the PreToolUse hook, which saw only the command line, and the
command line need not name the vault at all (``python check.py``, where the
script lists the parent directory). Claude Code's ``permission_denials`` lists
only the calls Claude Code refused itself. So after every session the driver
reads the tool results in the transcript.

A refusal is a line of tool output that says ``Operation not permitted``,
``[Errno 1]``, ``EPERM``, ``Permission denied`` or ``Read-only file system``,
or a line inside the ``<sandbox_violations>`` block Claude Code appends when
the sandbox blocks a network connection. A refusal whose line names the vault,
the agent directory or the driver directory (as written, or as a path that
resolves into one of them from the workspace) is a probe: it becomes a ledger
row and the run halts. Any other refusal, a cache under the home directory or
the network, is a sandbox event the report lists.

The two sandboxes refuse differently. sandbox-exec on macOS answers a read of
a denied directory with ``EPERM``. bubblewrap on Linux mounts an empty
directory over it, so the read finds nothing (``No such file or directory``),
a listing is empty and a write lands in memory and is gone when the command
ends; writes anywhere else outside the workspace fail with ``EROFS``. So a
line saying ``No such file or directory`` (``[Errno 2]``, ``ENOENT``) is a
probe too when it names a protected directory, and is ignored otherwise,
since most missing files are only mistakes.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from blinded.hook import _inside, _resolve

_REFUSED = re.compile(
    r"Operation not permitted|\[Errno 1\]|\bEPERM\b|Permission denied|Read-only file system"
)
_HIDDEN = re.compile(r"No such file or directory|\[Errno 2\]|\bENOENT\b")
_VIOLATIONS = re.compile(r"<sandbox_violations>(.*?)</sandbox_violations>", re.S)
_SPLIT = re.compile(r"[\s'\"`=;|&<>(),:]+")


@dataclass(frozen=True)
class Refusal:
    """One refused operation in a session.

    ``kind`` is ``probe`` when ``line`` reaches into a protected directory
    (named in ``target``), ``network`` for a blocked connection, and
    ``filesystem`` for any other refused file operation.
    """

    tool_use_id: str
    tool: str
    line: str
    kind: str
    target: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def tool_calls(events: list[dict]) -> list[dict]:
    """Each tool call with its input and its result text, in the order they were made."""
    calls: dict[str, dict] = {}

    def call(tid: str) -> dict:
        return calls.setdefault(tid, {"id": tid, "name": None, "input": {}, "output": ""})

    for event in events:
        if event.get("type") not in ("assistant", "user"):
            continue
        for part in (event.get("message") or {}).get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_use":
                call(part["id"]).update(name=part.get("name"), input=part.get("input") or {})
            elif part.get("type") == "tool_result":
                call(part.get("tool_use_id", ""))["output"] += text(part.get("content"))
    return list(calls.values())


def text(content) -> str:
    """A tool result's content as plain text: a string, or a list of text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            text(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content
        )
    return "" if content is None else str(content)


def refusals(events: list[dict], protected: list[str], cwd: Path) -> list[Refusal]:
    """Every refused operation in a transcript, classified.

    ``protected`` holds the absolute paths (and aliases) of the vault, agent
    and driver directories; ``cwd`` is the workspace, against which relative
    paths in the output are resolved.
    """
    out = []
    for call in tool_calls(events):
        output = call["output"]
        flagged = [ln for b in _VIOLATIONS.findall(output) for ln in b.splitlines()]
        rest = _VIOLATIONS.sub("", output).splitlines()
        flagged += [ln for ln in rest if _REFUSED.search(ln) or _HIDDEN.search(ln)]
        for line in filter(None, (ln.strip() for ln in flagged)):
            target = reaches(line, protected, cwd)
            if target is None and not _REFUSED.search(line) and _HIDDEN.search(line):
                continue  # a missing file outside the protected directories
            if target is not None:
                kind = "probe"
            elif "network" in line:
                kind = "network"
            else:
                kind = "filesystem"
            out.append(Refusal(call["id"], str(call["name"]), line[:300], kind, target))
    return out


def reaches(line: str, protected: list[str], cwd: Path) -> str | None:
    """The protected directory a line of output names, or None.

    A line names one if it contains its absolute path, or if any path-like
    token in it resolves into it from ``cwd`` (``../vault/val.parquet``).
    """
    for p in sorted(protected, key=len, reverse=True):
        if p in line:
            return p
    for token in _SPLIT.split(line):
        if not token or not ("/" in token or token == ".."):
            continue
        target = _resolve(token, cwd)
        for p in protected:
            if _inside(target, [p]):
                return p
    return None


def record(run, events: list[dict], iteration: int | None = None) -> tuple[int, list[Refusal]]:
    """Add a probe row for each probe in a session's transcript; return the count and all refusals.

    A call the hook already recorded (same ``tool_use_id``) is not recorded twice.
    """
    from blinded.sandbox import protected

    found = refusals(events, protected(run), run.workspace)
    ledger = run.ledger()
    seen = " ".join(r["reason_short"] or "" for r in ledger.trials("probe"))
    added = 0
    for ref in found:
        if ref.kind != "probe" or f"[{ref.tool_use_id}]" in seen:
            continue
        ledger.add_trial(
            run.run_id,
            kind="probe",
            name=ref.tool,
            submitted_by="driver",
            iteration=iteration,
            status="probe",
            reason_short=f"[{ref.tool_use_id}] refused by the sandbox: {ref.line}"[:200],
        )
        seen += f" [{ref.tool_use_id}]"
        added += 1
    ledger.close()
    return added, found


def events(path: str | Path) -> list[dict]:
    """The JSON events of a transcript, one per line; a line that does not parse (the torn
    last line of a killed session) is skipped."""
    out = []
    path = Path(path)
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
