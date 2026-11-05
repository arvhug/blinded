"""The OS sandbox on its own: blinded's settings run through the runtime Claude Code uses.

Claude Code's Bash sandbox is Anthropic's sandbox runtime, ``srt``:
sandbox-exec on macOS and bubblewrap on Linux. These tests hand ``srt`` the
filesystem and network rules :func:`blinded.sandbox.write` puts in
``agent-settings.json`` and run the probes an agent session could try from its
workspace, so one file is checked on both kernels. They need ``srt`` on PATH
(``npm install -g @anthropic-ai/sandbox-runtime``), or its path in
``BLINDED_SRT``, and are skipped without it.

Each protected directory holds a canary, and a probe passes when the canary
never reaches the output and nothing it writes reaches the disk, whatever the
error. The two sandboxes refuse differently (macOS with ``EPERM``, Linux by
hiding the directory), and the driver learns of a probe only from the
session's output, so the probes also check that the scan of a transcript
(:mod:`blinded.transcript`) records each read as a probe.
"""

import json
import os
import shutil
import subprocess
import sys
import uuid

import pytest

from blinded import sandbox, transcript
from blinded.transcript import _REFUSED
from blinded.vault import RunDir

SRT = os.environ.get("BLINDED_SRT") or shutil.which("srt")
pytestmark = [
    pytest.mark.os_sandbox,
    pytest.mark.skipif(SRT is None, reason="srt is not installed"),
]


@pytest.fixture
def boxed(run_dir, tmp_path):
    """A copy of the run with a canary in each protected directory and in a fake home,
    and a function that runs a shell command in its workspace under the OS sandbox."""
    root = tmp_path / "run"
    shutil.copytree(run_dir.root, root)
    run = RunDir(root)
    box = json.loads(sandbox.write(run).read_text())["sandbox"]
    home = tmp_path / "home"
    home.mkdir()
    canary = uuid.uuid4().hex
    for where in (run.vault, run.root / "agent", run.driver, home):
        (where / "canary.txt").write_text(canary)
    # the sixth, seventh and eighth canaries: the touch, depth and theo files the vault cuts
    # beside the segments, which only the judge's sims and the panel read
    for kind in ("touch", "depth", "theo"):
        (run.vault / f"val.{kind}.parquet").write_text(canary)
    # and forward paper's page and ledger, which blinded forward writes under the vault
    (run.vault / "forward").mkdir()
    for name in ("forward.html", "ledger.sqlite"):
        (run.vault / "forward" / name).write_text(canary)
    rules = {
        "filesystem": {
            "denyRead": box["filesystem"]["denyRead"],
            "allowRead": box["filesystem"]["allowRead"],
            "allowWrite": sandbox.aliases(run.workspace),
            "denyWrite": [],
        },
        "network": {"allowedDomains": box["network"]["allowedDomains"], "deniedDomains": []},
    }
    settings = tmp_path / "srt-settings.json"
    settings.write_text(json.dumps(rules))
    # srt reads ~ from HOME, so the session's "~/" rule lands on the fake home
    env = sandbox.environment() | {"HOME": str(home)}

    def sh(command: str) -> subprocess.CompletedProcess:
        return subprocess.run([SRT, "--settings", str(settings), "-c", command],
                              cwd=run.workspace, env=env, capture_output=True, text=True,
                              timeout=120)  # fmt: skip

    return run, home, canary, sh


def kept_out(out: subprocess.CompletedProcess, canary: str) -> bool:
    return canary not in out.stdout + out.stderr


def scanned(run: RunDir, command: str, out: subprocess.CompletedProcess) -> list[str]:
    """What the driver's scan makes of this output, as one Bash call in a transcript."""
    events = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t", "name": "Bash", "input": {"command": command}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t", "content": out.stdout + out.stderr}]}},
    ]  # fmt: skip
    return [r.kind for r in transcript.refusals(events, sandbox.protected(run), run.workspace)]


@pytest.mark.known_answer
def test_the_workspace_and_blindeds_python_still_work(boxed):
    run, _, _, sh = boxed
    py = sys.executable
    out = sh(f"cat DATA.md > /dev/null && echo x > features/probe.txt && "
             f"{py} -c 'import blinded, polars; print(\"ok\")'")  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
    assert (run.workspace / "features" / "probe.txt").read_text() == "x\n"


@pytest.mark.known_answer
def test_a_read_of_any_protected_directory_is_refused(boxed):
    run, home, canary, sh = boxed
    py = sys.executable
    for where in (run.vault, run.root / "agent", run.driver):
        for command in (f"cat {where}/canary.txt",
                        f"{py} -c \"print(open('{where}/canary.txt').read())\"",
                        f"cat ../{where.name}/canary.txt"):  # fmt: skip
            out = sh(command)
            assert out.returncode != 0 and kept_out(out, canary), command
            assert "probe" in scanned(run, command, out), (command, out.stderr)
    # the home directory, where Claude Code keeps every session's transcript
    out = sh("cat ~/canary.txt")
    assert out.returncode != 0 and kept_out(out, canary), out.stderr


@pytest.mark.known_answer
def test_the_vaults_touch_and_depth_files_are_refused(boxed):
    run, _, canary, sh = boxed
    py = sys.executable
    for kind in ("touch", "depth"):
        path = run.vault / f"val.{kind}.parquet"
        for command in (f"cat {path}", f"cat ../vault/val.{kind}.parquet",
                        f"{py} -c \"print(open('{path}').read())\""):  # fmt: skip
            out = sh(command)
            assert out.returncode != 0 and kept_out(out, canary), command
            assert "probe" in scanned(run, command, out), (command, out.stderr)


@pytest.mark.known_answer
def test_forward_papers_page_and_ledger_are_refused(boxed):
    run, _, canary, sh = boxed
    py = sys.executable
    for name in ("forward.html", "ledger.sqlite"):
        path = run.vault / "forward" / name
        for command in (f"cat {path}", f"cat ../vault/forward/{name}",
                        f"{py} -c \"print(open('{path}').read())\""):  # fmt: skip
            out = sh(command)
            assert out.returncode != 0 and kept_out(out, canary), command
            assert "probe" in scanned(run, command, out), (command, out.stderr)


@pytest.mark.known_answer
def test_a_script_that_never_names_the_vault_still_cannot_read_it(boxed):
    # the probe the hook cannot see: walk the run directory and open whatever is there
    run, _, canary, sh = boxed
    (run.workspace / "look.py").write_text(
        "import os\n"
        "up = os.path.dirname(os.getcwd())\n"
        "for d in sorted(os.listdir(up)):\n"
        "    p = os.path.join(up, d, 'canary.txt')\n"
        "    try:\n"
        "        print(d, open(p).read())\n"
        "    except OSError as e:\n"
        "        print(d, type(e).__name__, e)\n"
    )
    out = sh(f"{sys.executable} look.py")
    assert kept_out(out, canary), out.stdout
    lines = dict(line.split(" ", 1) for line in out.stdout.splitlines())
    for name in ("vault", "agent", "driver"):
        assert lines[name].split()[0] in ("PermissionError", "FileNotFoundError"), lines[name]
    # the command never names the vault, so only the scan can catch it
    assert scanned(run, "python look.py", out).count("probe") == 3


@pytest.mark.known_answer
def test_writes_outside_the_workspace_are_refused(boxed):
    run, home, _, sh = boxed
    # into a protected directory: refused on macOS; on Linux it lands in the empty directory
    # mounted over it and is gone when the command ends. Either way the disk is untouched.
    for target in (run.vault / "planted.txt", run.root / "agent" / "planted.txt",
                   run.driver / "planted.txt", home / "planted.txt"):  # fmt: skip
        sh(f"echo x > {target}")
        assert not target.exists(), target
    # anywhere else outside the workspace: refused on both, EPERM or EROFS
    target = run.root / "escape.txt"
    out = sh(f"echo x > {target}")
    assert out.returncode != 0 and not target.exists()
    assert _REFUSED.search(out.stderr), out.stderr


@pytest.mark.known_answer
def test_the_network_is_closed(boxed):
    _, _, _, sh = boxed
    probe = ("import socket\n"
             "try:\n"
             "    socket.create_connection(('1.1.1.1', 80), timeout=5)\n"
             "    print('connected')\n"
             "except OSError as e:\n"
             "    print('refused', e)\n")  # fmt: skip
    out = sh(f'{sys.executable} -c "{probe}"')
    assert "connected" not in out.stdout and "refused" in out.stdout, out.stdout + out.stderr
