"""The PreToolUse hook: deny any tool call that reaches for the vault, and record it as a probe.

Claude Code runs the hook before every tool call, outside the Bash sandbox,
with the call as JSON on stdin. This module uses only the standard library and
is loaded by file path from ``DIR/agent/hook.py``, so a hook call does not
import polars.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import re
import shlex
import sys
from pathlib import Path

_FILE_TOOLS = {"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"}
_SEARCH_TOOLS = {"Glob", "Grep"}
_TEST_SCORING = re.compile(r"\bblinded\s+test\b|--test\b|--segment[=\s]+test\b")
# a segment's rows, and the touch, depth and theo files cut beside them
_SEGMENT_FILES = re.compile(r"\b(val|test)(\.touch|\.depth|\.theo)?\.parquet\b")
SAVED_OUTPUT = "read a saved tool result with the Read tool; Bash cannot reach it"
_WILDCARD = re.compile(r"[*?\[]")


def decide(event: dict, config: dict) -> str | None:
    """Why the call must be denied, or None to let it through.

    ``config`` holds ``protected``: the vault, the agent directory and their
    aliases as absolute paths.
    """
    tool = event.get("tool_name", "")
    args = event.get("tool_input") or {}
    cwd = Path(event.get("cwd") or ".")
    protected = config["protected"]
    if tool == "Bash":
        command = str(args.get("command", ""))
        if any(p in command for p in protected) or re.search(r"(^|[\s/'\"=])vault/", command):
            return "the command names the vault"
        tokens = re.split(r"[\s'\"=;|&<>(),]+", command) + _shell_words(command)
        paths = [t for t in tokens if t and ("/" in t or t in ("..", "vault"))]
        if any(_inside(_resolve(t, cwd), protected) for t in paths):
            return "the command names the vault"
        if _SEGMENT_FILES.search(command):
            return "the command names a validation or test file"
        if _TEST_SCORING.search(command):
            return "the test segment is scored once, by the driver"
        session = _session(event)
        rest = _saved_output().sub(lambda m: "" if _own(m, session) else m[0], command)
        if ".claude/" in rest or ".claude " in rest:
            return "settings are not the agent's to change"
        return SAVED_OUTPUT if rest != command else None
    paths = []
    if tool in _FILE_TOOLS:
        paths = [args.get("file_path") or args.get("notebook_path") or ""]
    elif tool in _SEARCH_TOOLS:
        paths = [args.get("path") or "", args.get("pattern") or "", args.get("glob") or ""]
    for raw in filter(None, map(str, paths)):
        target = _resolve(raw, cwd)
        if _inside(target, protected):
            return f"{tool} on the vault"
        if any(p in raw for p in protected) or raw.startswith("vault/") or "/vault/" in raw:
            return f"{tool} on the vault"
        if tool in _FILE_TOOLS and tool != "Read" and ".claude" in Path(target).parts:
            return "settings are not the agent's to change"
        saved, session = _saved_output(), _session(event)
        named = str(Path(raw).expanduser())
        own = any(m and _own(m, session) for m in map(saved.fullmatch, (target, named)))
        if _inside(target, [_resolve("~/.claude", cwd)]) and not own:
            return "Claude Code's own files are not the agent's to read"
    return None


def _session(event: dict) -> str:
    """The event's ``session_id``, or the name of its transcript."""
    return event.get("session_id") or Path(event.get("transcript_path") or "").stem


def _saved_output() -> re.Pattern:
    """Paths into any session's saved tool results, from ``~`` or absolute.

    Claude Code writes a tool result too long to show to
    ``~/.claude/projects/<workspace>/<session>/tool-results/`` and tells the
    agent to read it there, so that one directory under ``~/.claude`` is the
    agent's own. :func:`_own` says whether a match is.
    """
    home = re.escape(str(Path.home().resolve()))
    stop = r"\s'\";|&<>()"
    return re.compile(
        rf"(?:~|{home})/\.claude/projects/(?P<ws>[^/{stop}]+)/(?P<session>[^/{stop}]+)"
        rf"/tool-results(?![^/{stop}])[^{stop}]*"
    )


def _own(m: re.Match, session: str) -> bool:
    """Whether a path :func:`_saved_output` matched stays in ``session``'s saved results.

    A path through ``..`` is a way out of that directory. A wildcard in the
    workspace or session part, as in ``*/6289cd80*``, counts when it could
    match ``session`` and every directory it matches on disk is that session's.
    """
    if not session or ".." in m[0]:
        return False
    if not _WILDCARD.search(m["ws"] + m["session"]):
        return m["session"] == session
    if not fnmatch.fnmatchcase(session, m["session"]):
        return False
    found = glob.glob(str(Path.home() / ".claude" / "projects" / m["ws"] / m["session"]))
    return all(Path(d).name == session for d in found)


def _shell_words(command: str) -> list[str]:
    """The words of a command as the shell reads them, quotes removed and joined, so that
    ``../vau"l"t/menu.json`` is ``../vault/menu.json``; each word split again at the
    shell's operators and at ``=``. A command the shell could not parse (an unclosed quote)
    gives its words up to there. Any word that resolves inside the vault is a probe. The
    hook reads text; the OS sandbox, which sees the system calls, is the real gate."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";|&<>()")
    lexer.whitespace_split = True
    words = []
    try:
        for word in lexer:
            words.append(word)
    except ValueError:
        pass
    return [part for w in words for part in re.split(r"[=,:]", w) if part]


def _inside(target: str, protected: list[str]) -> bool:
    return any(target == p or target.startswith(p.rstrip("/") + "/") for p in protected)


def _resolve(raw: str, cwd: Path) -> str:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = cwd / path
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def deny(reason: str, recorded: bool = True) -> dict:
    """The hook's answer for a denied call."""
    tail = " The attempt was recorded." if recorded else ""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"blinded: {reason}.{tail}",
        }
    }


def main(config_path: str, ledger_module) -> int:
    """Read one tool call from stdin, answer on stdout, log a denial to the ledger."""
    config = json.loads(Path(config_path).read_text())
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    reason = decide(event, config)
    if reason is None:
        return 0
    if reason == SAVED_OUTPUT:  # a wrong tool, not a probe
        print(json.dumps(deny(reason, recorded=False)))
        return 0
    detail = json.dumps({"tool": event.get("tool_name"), "input": event.get("tool_input")})
    tag = f"[{event.get('tool_use_id', '')}] {reason}: {detail}"
    try:
        ledger = ledger_module.Ledger(config["ledger"])
        ledger.probe(config["run_id"], str(event.get("tool_name")), tag)
        ledger.close()
    except Exception as exc:  # the denial stands even if the ledger cannot be written
        print(f"blinded hook: could not record the probe: {exc}", file=sys.stderr)
    print(json.dumps(deny(reason)))
    return 0
