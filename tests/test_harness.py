import json
import os
import shutil
import stat
import sys
import time

import pytest

from blinded import sandbox
from blinded.harness import SessionLimits
from blinded.harness.claude_cli import ClaudeCLI, parse
from blinded.vault import RunDir

known = pytest.mark.known_answer
edge = pytest.mark.edge_case

SCHEMA = {"type": "object", "properties": {"next": {"type": "string"}}, "required": ["next"]}

EVENTS = [
    {"type": "system", "subtype": "init", "session_id": "s-1", "cwd": "."},
    {
        "type": "assistant",
        "session_id": "s-1",
        "message": {
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}
            ]
        },
    },  # fmt: skip
    {
        "type": "user",
        "session_id": "s-1",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "DATA.md\nRULES.md",
                    "is_error": False,
                }
            ]
        },
    },  # fmt: skip
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 7,
        "total_cost_usd": 0.0123,
        "session_id": "s-1",
        "structured_output": {"next": "try the basis"},
        "permission_denials": [{"tool_name": "WebFetch", "tool_use_id": "toolu_2"}],
    },
]


@pytest.fixture
def run(run_dir, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run_dir.root, root)
    run = RunDir(root)
    sandbox.write(run)
    return run


def stub(tmp_path, body: str):
    """A stand-in for the claude binary: records its argv and cwd, then runs ``body``."""
    path = tmp_path / "claude-stub"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys, time\n"
        f"open({str(tmp_path / 'argv.json')!r}, 'w').write("
        "json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'path': os.environ['PATH']}))\n"
        + body
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def emit(events) -> str:
    return "".join(f"print(json.dumps({e!r}), flush=True)\n" for e in events)


@known
def test_a_session_is_run_in_the_workspace_and_read_from_its_transcript(run, tmp_path):
    harness = ClaudeCLI(run, SCHEMA, run.root / "agent" / "system_append.md",
                        binary=stub(tmp_path, emit(EVENTS)))  # fmt: skip
    limits = SessionLimits(model="sonnet", effort="low", max_turns=12, budget_usd=0.4)
    log = run.driver / "logs" / "iter_1.jsonl"
    result = harness.run("do it", run.workspace, limits, log=log)
    assert (result.outcome, result.ok, result.session_id) == ("success", True, "s-1")
    assert (result.cost_usd, result.turns) == (0.0123, 7)
    assert result.summary == {"next": "try the basis"}
    assert result.permission_denials == [{"tool_name": "WebFetch", "tool_use_id": "toolu_2"}]
    assert result.transcript == log and len(log.read_text().splitlines()) == 4
    called = json.loads((tmp_path / "argv.json").read_text())
    argv = called["argv"]
    assert called["cwd"] == str(run.workspace.resolve())
    assert called["path"].split(os.pathsep)[0] == os.path.dirname(sys.executable)
    assert argv[:2] == ["-p", "do it"]
    pairs = {
        flag: argv[argv.index(flag) + 1]
        for flag in ("--model", "--effort", "--max-turns", "--max-budget-usd", "--output-format")
    }
    assert pairs == {"--model": "sonnet", "--effort": "low", "--max-turns": "12",
                     "--max-budget-usd": "0.4", "--output-format": "stream-json"}  # fmt: skip
    assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA
    assert "--verbose" in argv and "--restricted" in argv


def gone(pid: int, within: float = 10.0) -> bool:
    """Whether process ``pid`` has disappeared, waiting up to ``within`` seconds.

    A killed process lingers as a zombie until its parent reaps it; under load
    that can take longer than any fixed pause, so poll instead of sleeping once.
    """
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


@edge
def test_a_session_past_its_wall_time_is_stopped_with_its_children(run, tmp_path):
    pids = tmp_path / "pids"
    body = emit(EVENTS[:1]) + (
        "child = subprocess.Popen(['sleep', '60'])\n"
        f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}')\n"
        "time.sleep(60)\n"
    )
    harness = ClaudeCLI(run, binary=stub(tmp_path, body))
    # the wall leaves the stub seconds to start under load, and the 60 s sleeps are far past
    # it plus the 10 s grace between SIGTERM and SIGKILL
    wall = 4.0
    started = time.monotonic()
    result = harness.run("x", run.workspace, SessionLimits(wall_seconds=wall))
    elapsed = time.monotonic() - started
    assert wall <= elapsed < wall + 10
    assert (result.outcome, result.session_id, result.cost_usd) == ("timeout", "s-1", 0.0)
    session, child = (int(p) for p in pids.read_text().split())
    assert gone(session) and gone(child)


@edge
def test_a_crashed_session_has_no_result_and_a_torn_last_line_is_skipped(tmp_path):
    log = tmp_path / "t.jsonl"
    log.write_text(json.dumps(EVENTS[0]) + "\n" + json.dumps(EVENTS[1]) + "\n" + '{"type": "us')
    result = parse(log)
    assert (result.outcome, result.session_id, result.summary, result.ok) == (
        "no_result",
        "s-1",
        None,
        False,
    )
    assert parse(tmp_path / "missing.jsonl").outcome == "no_result"


@edge
def test_a_session_that_ran_out_of_turns_is_not_ok(tmp_path):
    final = EVENTS[-1] | {"subtype": "error_max_turns", "is_error": True}
    final.pop("structured_output")
    log = tmp_path / "t.jsonl"
    log.write_text(json.dumps(final) + "\n")
    result = parse(log)
    assert (result.outcome, result.ok, result.summary, result.turns) == (
        "error_max_turns",
        False,
        None,
        7,
    )
