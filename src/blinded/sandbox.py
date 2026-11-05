"""The agent's sandbox: ``agent-settings.json``, the PreToolUse hook, and the ``claude -p`` line.

Written to ``DIR/agent/``, outside the workspace, so the agent's file tools
cannot reach them. ``DIR/driver/``, where the driver keeps its state and the
session transcripts, is protected the same way. The defences,
from the outside in:

1. The vault is outside the workspace and never passed with ``--add-dir``;
   ``--restricted`` confines the file tools to the working directory.
2. Deny rules on the vault and the agent and driver directories, as absolute
   ``//`` paths, which also cover shell file commands Claude Code recognises.
3. The OS sandbox on every Bash command and its children: no reads of the
   vault, the agent and driver directories or the home directory (except the
   workspace and the Python that runs blinded), writes only in the workspace,
   no network.
4. ``--restricted`` loads only managed settings and ``--settings``, so a
   ``.claude/settings.json`` written into the workspace, and the user's own
   ``~/.claude/settings.json``, are never read. ``--strict-mcp-config`` drops
   MCP servers configured anywhere else.
5. The PreToolUse hook, which denies calls that name the vault or ask for the
   test segment and records each as a probe in the ledger.
6. The judge daemon runs outside the sandbox; the agent never needs the vault.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import blinded
from blinded.vault import RunDir

TOOLS = "Bash,Read,Edit,Write,Glob,Grep"
PROTECTED = ("vault", "agent", "driver")


def aliases(path: Path) -> list[str]:
    """A path as given and resolved, with and without macOS's ``/private`` prefix."""
    out = {str(Path(path).absolute()), str(Path(path).resolve())}
    for p in list(out):
        if p.startswith("/private/"):
            out.add(p.removeprefix("/private"))
    return sorted(out)


def protected(run: RunDir) -> list[str]:
    """The vault, agent and driver directories of ``run``, with their aliases."""
    return sorted({a for name in PROTECTED for a in aliases(run.root / name)})


def write(run: RunDir, python: str | None = None) -> Path:
    """Write ``agent-settings.json``, ``hook.json`` and ``hook.py`` under ``DIR/agent/``."""
    agent = run.root / "agent"
    agent.mkdir(exist_ok=True)
    (run.root / "driver" / "logs").mkdir(parents=True, exist_ok=True)
    guarded = protected(run)
    src = Path(blinded.__file__).resolve().parent
    config = {"protected": guarded, "ledger": str(run.ledger_path), "run_id": run.run_id}
    (agent / "hook.json").write_text(json.dumps(config, indent=2) + "\n")
    (agent / "hook.py").write_text(_HOOK.format(src=src, config=agent / "hook.json"))
    hook = f"{shlex.quote(python or sys.executable)} {shlex.quote(str(agent / 'hook.py'))}"
    settings = {
        "permissions": {
            "deny": deny_rules(run),
            "allow": ["Edit(features/**)", "Edit(strategies/**)", "Edit(NOTES.md)"],
        },
        "hooks": {
            "PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": hook}]}]
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": True,
            "filesystem": {
                "denyRead": [*guarded, "~/"],
                "allowRead": [*aliases(run.workspace), *_python_dirs(src)],
            },
            "network": {"allowedDomains": []},
        },
    }
    path = agent / "agent-settings.json"
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return path


def deny_rules(run: RunDir) -> list[str]:
    rules = [f"{tool}(/{p}/**)" for p in protected(run) for tool in ("Read", "Edit")]
    return [*rules, "Edit(.claude/**)", "Bash(claude *)", "WebFetch", "WebSearch"]


def command(
    run: RunDir,
    prompt: str,
    model: str = "opus",
    max_turns: int = 60,
    budget_usd: float = 3.0,
    output: str = "json",
    effort: str | None = None,
    schema: str | None = None,
    system_append: Path | None = None,
) -> list[str]:
    """The ``claude -p`` argv for one agent session in ``run``'s workspace.

    ``schema`` is a JSON Schema, as text, for the session's structured output;
    ``system_append`` a file appended to the system prompt. Sessions are not
    saved under ``~/.claude`` and load no MCP servers or skills.
    """
    argv = ["claude", "-p", prompt, "--restricted", "--tools", TOOLS]
    argv += ["--settings", str(run.root / "agent" / "agent-settings.json")]
    argv += ["--permission-mode", "dontAsk", "--permission-prompts", "none"]
    argv += ["--disallowedTools", *deny_rules(run)]
    argv += ["--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"]
    argv += ["--max-turns", str(max_turns), "--max-budget-usd", f"{budget_usd:g}"]
    argv += ["--model", model]
    if effort:
        argv += ["--effort", effort]
    if system_append is not None:
        argv += ["--append-system-prompt-file", str(system_append)]
    if schema is not None:
        argv += ["--json-schema", schema]
    argv += ["--output-format", output]
    return argv + (["--verbose"] if output == "stream-json" else [])


def record_denials(run: RunDir, result: dict, others: list[dict] | None = None) -> int:
    """Add a probe row for each denial in a session's result that reached for the vault.

    Claude Code refuses some calls before PreToolUse hooks run: a file tool
    pointed outside the working directory under ``--restricted``, or at a path
    a deny rule covers. Those never reach the hook, but they are listed in the
    result's ``permission_denials``, which the driver reads after every
    session. A denial is a probe when the hook's own rule
    (:func:`blinded.hook.decide`) would have refused it: the vault, the agent or
    driver directory, a validation or test file, test scoring, or settings.
    The rest (a write outside ``features/``, a web fetch, a Bash read of the
    session's own saved tool result) are appended to ``others`` when it is
    given. Rows are matched on ``tool_use_id``, so a call
    the hook already recorded is not recorded twice. Returns how many probe
    rows were added.
    """
    from blinded.hook import SAVED_OUTPUT, decide

    config = {"protected": protected(run)}
    ledger = run.ledger()
    seen = " ".join(r["reason_short"] or "" for r in ledger.trials("probe"))
    added = 0
    for denial in result.get("permission_denials") or []:
        tid = denial.get("tool_use_id", "")
        event = {
            "tool_name": denial.get("tool_name"),
            "tool_input": denial.get("tool_input") or {},
            "cwd": str(run.workspace),
            "session_id": result.get("session_id"),
        }
        if decide(event, config) in (None, SAVED_OUTPUT):
            if others is not None:
                others.append(denial)
            continue
        if tid and f"[{tid}]" in seen:
            continue
        detail = json.dumps({"tool": denial.get("tool_name"), "input": denial.get("tool_input")})
        ledger.add_trial(
            run.run_id,
            kind="probe",
            name=str(denial.get("tool_name")),
            submitted_by="driver",
            status="probe",
            reason_short=f"[{tid}] denied before the hook: {detail}"[:200],
        )
        added += 1
    ledger.close()
    return added


def environment(python: str | None = None) -> dict[str, str]:
    """The environment for the session: blinded's ``bin`` first on PATH."""
    env = dict(os.environ)
    bin_dir = str(Path(python or sys.executable).parent)
    env["PATH"] = os.pathsep.join([bin_dir, env.get("PATH", "")])
    return env


def _python_dirs(src: Path) -> list[str]:
    dirs = {sys.prefix, sys.base_prefix, str(src.parent)}
    return sorted({a for d in dirs for a in aliases(Path(d))})


_HOOK = '''\
"""PreToolUse hook for one run directory, written by blinded. Standard library only."""

import importlib.util
import sys


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ledger = _load("blinded_ledger", "{src}/ledger.py")
hook = _load("blinded_hook", "{src}/hook.py")
sys.exit(hook.main("{config}", ledger))
'''
