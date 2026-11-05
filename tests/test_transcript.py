import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from blinded import sandbox, transcript
from blinded.vault import RunDir

known = pytest.mark.known_answer
edge = pytest.mark.edge_case

FIXTURES = Path(__file__).parent / "transcripts"


@pytest.fixture
def run(run_dir, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run_dir.root, root)
    run = RunDir(root)
    sandbox.write(run)
    return run


def fixture(name: str, run: RunDir) -> list[dict]:
    """A real transcript, recorded on a synthetic run, with its paths moved to ``run``."""
    text = (FIXTURES / name).read_text()
    text = text.replace("{RUN}", str(run.root)).replace("{HOME}", str(Path.home()))
    return [json.loads(line) for line in text.splitlines()]


def bash(tid: str, command: str, output: str) -> list[dict]:
    return [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": command}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": tid, "content": output}]}},
    ]  # fmt: skip


@known
def test_a_script_that_reads_the_vault_is_a_probe_the_hook_never_saw(run):
    # recorded with claude -p 2.1.301 on macOS: `python check_paths.py` lists the run root
    # and opens */manifest.json; the hook passes the command, the sandbox refuses the vault
    events = fixture("script_reads_the_vault.jsonl", run)
    added, found = transcript.record(run, events, iteration=3)
    assert added == 1
    assert [r.kind for r in found] == ["probe"] * 3
    (ref,) = [r for r in found if r.target == str(run.vault)]
    assert (ref.kind, ref.tool) == ("probe", "Bash")
    assert ref.target == str(run.vault)
    assert ref.line.startswith("vault PermissionError [Errno 1] Operation not permitted")
    (row,) = run.ledger().trials("probe")
    assert (row["submitted_by"], row["iteration"], row["status"]) == ("driver", 3, "probe")
    assert row["reason_short"].startswith(
        "[toolu_01UZeMde8iNHsaLGqL4rB7g6] refused by the sandbox"
    )
    # agent/ and driver/ were missing files in the same output, which on Linux is how a denied
    # directory looks: they are probes too, in the same call, so still one row
    assert {r.target for r in found} == {str(run.vault), str(run.root / "agent"), str(run.driver)}
    assert transcript.record(run, events)[0] == 0


@known
def test_the_home_directory_and_the_network_are_events_not_probes(run):
    # recorded the same way: `ls ~/Library` and an https request from python
    events = fixture("home_and_network.jsonl", run)
    added, found = transcript.record(run, events)
    assert added == 0
    assert [r.kind for r in found] == ["filesystem", "network"]
    assert found[0].line == f"ls: {Path.home()}/Library: Operation not permitted"
    assert found[1].line == "deny network-outbound example.com:443 (user denied)"
    assert run.ledger().trials("probe") == []


@edge
def test_relative_paths_and_list_content_are_resolved(run):
    events = [
        *bash("t1", "find .. -name '*.npy'", "find: ../driver/logs: Operation not permitted"),
        *bash("t2", "ls ..", "ls: ..: Operation not permitted"),
        *bash("t3", "python x.py", "OSError: [Errno 1] EPERM"),
    ]
    events[1]["message"]["content"][0]["content"] = [
        {"type": "text", "text": "find: ../driver/logs: Operation not permitted"}
    ]
    found = transcript.refusals(events, sandbox.protected(run), run.workspace)
    assert [(r.tool_use_id, r.kind) for r in found] == [
        ("t1", "probe"),
        ("t2", "filesystem"),
        ("t3", "filesystem"),
    ]
    assert found[0].target == str(run.driver)


@edge
def test_a_call_the_hook_recorded_is_not_recorded_again(run):
    command = f"cat {run.vault}/val.parquet"
    event = {"tool_name": "Bash", "tool_use_id": "toolu_H", "cwd": str(run.workspace),
             "tool_input": {"command": command}}  # fmt: skip
    subprocess.run(
        [sys.executable, str(run.root / "agent" / "hook.py")],
        input=json.dumps(event), capture_output=True, text=True, check=True,
    )  # fmt: skip
    events = bash("toolu_H", command, f"cat: {run.vault}/val.parquet: Operation not permitted")
    assert transcript.record(run, events) == (0, transcript.refusals(
        events, sandbox.protected(run), run.workspace))  # fmt: skip
    assert len(run.ledger().trials("probe")) == 1


@known
def test_a_missing_file_in_a_protected_directory_is_a_probe(run):
    # bubblewrap mounts an empty directory over a denied one: the read finds nothing
    hidden = f"vault FileNotFoundError [Errno 2] No such file or directory: '{run.vault}/c.txt'"
    events = bash("t1", "python look.py", hidden)
    events += bash("t2", "cat ../vault/x", "cat: ../vault/x: No such file or directory")
    events += bash("t3", "touch ../x", "touch: cannot touch '../x': Read-only file system")
    found = transcript.refusals(events, sandbox.protected(run), run.workspace)
    assert [(r.tool_use_id, r.kind, r.target) for r in found] == [
        ("t1", "probe", str(run.vault)),
        ("t2", "probe", str(run.vault)),
        ("t3", "filesystem", None),
    ]  # fmt: skip


@edge
def test_ordinary_output_has_no_refusals(run):
    events = bash("t", "blinded submit features/a.py", '{"trials": [{"status": "accepted"}]}')
    events += bash("u", "cat features/x.py", "cat: features/x.py: No such file or directory")
    assert transcript.refusals(events, sandbox.protected(run), run.workspace) == []
    assert transcript.tool_calls(events)[0]["input"] == {"command": "blinded submit features/a.py"}
