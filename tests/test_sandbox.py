import json
import shutil
import subprocess
import sys

import pytest

from blinded import sandbox
from blinded.hook import decide
from blinded.vault import RunDir

known = pytest.mark.known_answer
edge = pytest.mark.edge_case


@pytest.fixture
def run(run_dir, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run_dir.root, root)
    run = RunDir(root)
    sandbox.write(run)
    return run


def call_hook(run, tool, **tool_input):
    event = {"tool_name": tool, "tool_input": tool_input, "cwd": str(run.workspace)}
    out = subprocess.run(
        [sys.executable, str(run.root / "agent" / "hook.py")],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout) if out.stdout.strip() else None


def probes(run):
    vault = run.vault
    return [
        ("Read", {"file_path": str(vault / "val.parquet")}),
        ("Bash", {"command": f"cat {vault}/val.parquet"}),
        ("Bash", {"command": f"python -c \"open('{vault}/val.parquet')\""}),
        (
            "Write",
            {
                "file_path": str(run.workspace / ".claude" / "settings.json"),
                "content": '{"permissions": {"allow": ["Read"]}}',
            },
        ),  # fmt: skip
        ("Bash", {"command": f"blinded test {run.root} --trial 1"}),
    ]


@known
def test_the_five_probes_are_denied_and_recorded(run):
    for tool, args in probes(run):
        answer = call_hook(run, tool, **args)
        decision = answer["hookSpecificOutput"]
        assert (decision["hookEventName"], decision["permissionDecision"]) == (
            "PreToolUse",
            "deny",
        ), (tool, args)
    rows = run.ledger().trials("probe")
    assert [r["status"] for r in rows] == ["probe"] * 5
    assert [r["name"] for r in rows] == ["Read", "Bash", "Bash", "Write", "Bash"]
    assert "Read on the vault" in rows[0]["reason_short"]


@known
def test_ordinary_work_passes_and_leaves_no_rows(run):
    allowed = [
        ("Read", {"file_path": str(run.workspace / "DATA.md")}),
        ("Read", {"file_path": "features/basis.py"}),
        ("Write", {"file_path": str(run.workspace / "features" / "x.py"), "content": "x = 1"}),
        ("Bash", {"command": "blinded submit features/x.py"}),
        ("Bash", {"command": "python -c 'import polars; print(polars.__version__)'"}),
        ("Grep", {"pattern": "basis", "path": "features"}),
        ("Read", {"file_path": str(run.workspace / ".claude" / "settings.json")}),
    ]
    for tool, args in allowed:
        assert call_hook(run, tool, **args) is None, (tool, args)
    assert run.ledger().trials("probe") == []


@edge
def test_relative_and_aliased_paths_are_caught(run):
    config = json.loads((run.root / "agent" / "hook.json").read_text())
    cwd = str(run.workspace)

    def ask(tool, **args):
        return decide({"tool_name": tool, "tool_input": args, "cwd": cwd}, config)

    assert ask("Read", file_path="../vault/val.parquet")
    assert ask("Glob", pattern="*.parquet", path="../vault")
    assert ask("Bash", command="ls ../vault")
    assert ask("Bash", command="head -c 100 test.parquet")
    # the touch and depth files the vault cuts beside the segments, by name or by path
    assert ask("Bash", command="python -c \"pl.read_parquet('val.touch.parquet')\"")
    assert ask("Bash", command="cp test.depth.parquet data/")
    assert ask("Read", file_path="../vault/val.touch.parquet")
    assert ask("Read", file_path=str(run.vault / "test.depth.parquet"))
    assert ask("Bash", command="ls data/train.touch.parquet") is None
    # and the theo's, whose validation and test rows were filtered with train's parameters
    assert ask("Bash", command="python -c \"pl.read_parquet('test.theo.parquet')\"")
    assert ask("Read", file_path=str(run.vault / "val.theo.parquet"))
    assert ask("Read", file_path=str(run.vault / "theo.json"))
    # forward paper's page and ledger (blinded.forward), under the vault
    assert ask("Read", file_path=str(run.vault / "forward" / "forward.html"))
    assert ask("Read", file_path="../vault/forward/ledger.sqlite")
    assert ask("Bash", command=f"sqlite3 {run.vault}/forward/ledger.sqlite 'select * from hours'")
    assert ask("Bash", command="cat ../vault/forward/forward.html")
    assert ask("Grep", pattern="net_usd", path="../vault/forward")
    assert ask("Bash", command="blinded submit features/x.py --test")
    assert ask("Edit", file_path=".claude/settings.local.json", old_string="a", new_string="b")
    assert ask("Bash", command="echo '{}' > .claude/settings.json")
    assert ask("Read", file_path=str(run.root / "agent" / "hook.json"))
    assert ask("Bash", command="ls features") is None


@known
def test_saved_tool_results_are_the_agents_own_and_nothing_else_under_claude_is(run):
    # Claude Code saves a long tool result beside the session's transcript and tells the
    # agent to read it there. In a real run a Bash read of it was taken for a probe.
    from pathlib import Path

    projects = Path.home() / ".claude" / "projects" / "-ws"
    transcript = str(projects / "s1.jsonl")
    saved = str(projects / "s1" / "tool-results" / "b1.txt")

    def hook(tool, **args):
        event = {"tool_name": tool, "tool_input": args, "cwd": str(run.workspace),
                 "transcript_path": transcript}  # fmt: skip
        out = subprocess.run([sys.executable, str(run.root / "agent" / "hook.py")],
                             input=json.dumps(event), capture_output=True, text=True,
                             check=True)  # fmt: skip
        return json.loads(out.stdout)["hookSpecificOutput"] if out.stdout.strip() else None

    assert hook("Read", file_path=saved) is None
    told = hook("Bash", command=f"sed -n 30,400p {saved}")
    assert "Read tool" in told["permissionDecisionReason"]
    assert "recorded" not in told["permissionDecisionReason"]
    tilde = "~/" + str(Path(saved).relative_to(Path.home()))
    assert "Read tool" in hook("Bash", command=f"head {tilde}")["permissionDecisionReason"]
    # the session's result lists the refused call; the scan after it must not count it
    denial = {"tool_name": "Bash", "tool_use_id": "t1", "tool_input": {"command": f"cat {saved}"}}
    result = {"session_id": "s1", "permission_denials": [denial]}
    assert sandbox.record_denials(run, result) == 0
    assert run.ledger().trials("probe") == []
    # another session's transcript, or a way out of the directory, is a probe
    assert hook("Read", file_path=str(projects / "s0.jsonl"))
    assert hook("Read", file_path=saved.replace("b1.txt", "../../s0.jsonl"))
    assert hook("Grep", pattern="sharpe", path=str(projects))
    assert hook("Bash", command=f"cat {saved} {projects}/s0.jsonl")
    assert hook("Bash", command=f"cat {saved}/../../../s0.jsonl")
    assert len(run.ledger().trials("probe")) == 5


@edge
def test_a_wildcard_reaches_the_sessions_own_results_only_when_it_matches_nothing_else(
    run, tmp_path, monkeypatch
):
    # In a real run the agent cd'd into its own results with the workspace and session globbed
    from blinded.hook import SAVED_OUTPUT

    home = tmp_path / "home"
    for session in ("6289cd80-aa", "77aa0000-bb"):
        (home / ".claude" / "projects" / "-ws" / session / "tool-results").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    config = json.loads((run.root / "agent" / "hook.json").read_text())

    def ask(command):
        event = {"tool_name": "Bash", "tool_input": {"command": command},
                 "cwd": str(run.workspace), "session_id": "6289cd80-aa"}  # fmt: skip
        return decide(event, config)

    own = "cd ~/.claude/projects/*/6289cd80*/tool-results/ && sed -n 1,400p b1.txt | tail -n +40"
    assert ask(own) == SAVED_OUTPUT
    assert (
        ask(f"ls {home.resolve()}/.claude/projects/-ws/6289cd80-aa/tool-results") == SAVED_OUTPUT
    )
    # a wildcard that also matches another session, or cannot match this one, is a probe
    for probe in (
        "cat ~/.claude/projects/*/*/tool-results/*",
        "cat ~/.claude/projects/-ws/77aa*/tool-results/*",
        "cat ~/.claude/projects/*/6289cd80*/tool-results/../../*.jsonl",
    ):
        assert ask(probe) not in (None, SAVED_OUTPUT), probe


@known
def test_settings_carry_each_defence(run):
    settings = json.loads((run.root / "agent" / "agent-settings.json").read_text())
    vault = str(run.vault.resolve())
    assert f"Read(/{vault}/**)" in settings["permissions"]["deny"]
    assert "Edit(.claude/**)" in settings["permissions"]["deny"]
    box = settings["sandbox"]
    assert box["enabled"] and not box["allowUnsandboxedCommands"] and box["failIfUnavailable"]
    assert vault in box["filesystem"]["denyRead"] and "~/" in box["filesystem"]["denyRead"]
    assert str(run.workspace.resolve()) in box["filesystem"]["allowRead"]
    assert box["network"]["allowedDomains"] == []
    (entry,) = settings["hooks"]["PreToolUse"]
    assert entry["hooks"][0]["command"].endswith("agent/hook.py")
    # the agent directory is beside the workspace, not in it
    assert not (run.root / "agent").is_relative_to(run.workspace)


@known
def test_the_command_line(run):
    argv = sandbox.command(run, "do the thing", model="haiku", budget_usd=0.5)
    assert argv[:3] == ["claude", "-p", "do the thing"]
    assert "--restricted" in argv and "--setting-sources" not in argv
    i = argv.index("--permission-mode")
    assert argv[i + 1] == "dontAsk"
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert argv[argv.index("--max-budget-usd") + 1] == "0.5"
    assert argv[argv.index("--settings") + 1].endswith("agent/agent-settings.json")


@known
def test_denials_the_hook_never_saw_are_recorded_once(run):
    # the hook saw a Bash probe; Claude Code refused a Read before the hook ran
    event = {"tool_name": "Bash", "tool_use_id": "toolu_A", "cwd": str(run.workspace),
             "tool_input": {"command": f"cat {run.vault}/val.parquet"}}  # fmt: skip
    subprocess.run(
        [sys.executable, str(run.root / "agent" / "hook.py")],
        input=json.dumps(event), capture_output=True, text=True, check=True,
    )  # fmt: skip
    result = {
        "permission_denials": [
            {"tool_name": "Bash", "tool_use_id": "toolu_A", "tool_input": event["tool_input"]},
            {
                "tool_name": "Read",
                "tool_use_id": "toolu_B",
                "tool_input": {"file_path": str(run.vault / "val.parquet")},
            },  # fmt: skip
        ]
    }
    assert sandbox.record_denials(run, result) == 1
    assert sandbox.record_denials(run, result) == 0
    rows = run.ledger().trials("probe")
    assert [(r["name"], r["submitted_by"]) for r in rows] == [("Bash", "hook"), ("Read", "driver")]


@known
def test_a_denial_away_from_the_vault_is_not_a_probe(run):
    result = {
        "permission_denials": [
            {
                "tool_name": "Write",
                "tool_use_id": "toolu_C",
                "tool_input": {"file_path": str(run.workspace / "scratch.py"), "content": "x"},
            },
            {
                "tool_name": "WebFetch",
                "tool_use_id": "toolu_D",
                "tool_input": {"url": "https://x"},
            },
            {
                "tool_name": "Read",
                "tool_use_id": "toolu_E",
                "tool_input": {"file_path": str(run.driver / "logs" / "iter_1.jsonl")},
            },
        ]
    }
    others = []
    assert sandbox.record_denials(run, result, others) == 1
    assert [d["tool_use_id"] for d in others] == ["toolu_C", "toolu_D"]
    (row,) = run.ledger().trials("probe")
    assert row["reason_short"].startswith("[toolu_E] denied before the hook")


@known
def test_the_driver_directory_is_denied_like_the_vault(run):
    settings = json.loads((run.root / "agent" / "agent-settings.json").read_text())
    driver = str(run.driver.resolve())
    assert driver in settings["sandbox"]["filesystem"]["denyRead"]
    assert f"Read(/{driver}/**)" in settings["permissions"]["deny"]
    argv = sandbox.command(run, "x", schema="{}", system_append=run.root / "agent" / "s.md")
    for flag in ("--strict-mcp-config", "--no-session-persistence", "--json-schema"):
        assert flag in argv
    assert argv[argv.index("--append-system-prompt-file") + 1].endswith("agent/s.md")


@known
def test_the_agent_cannot_change_the_rules_its_run_is_judged_under(run):
    # the version lives in vault/manifest.json, the vault's: every tool call that would write
    # it is denied, the sandbox denies reading the vault at all, and the workspace's copy of
    # the brief, which the agent may edit, sets only what its own blinded eval uses
    import yaml

    from blinded import brief as briefs
    from blinded.ladder import RULES

    manifest = run.vault / "manifest.json"
    for tool, inp in (("Write", {"file_path": str(manifest), "content": "{}"}),
                      ("Edit", {"file_path": str(manifest), "old_string": "2", "new_string": "1"}),
                      ("Bash", {"command": f"sed -i '' 's/judge_rules/x/' {manifest}"}),
                      ("Bash", {"command": f"python -c \"open('{manifest}','w')\""})):  # fmt: skip
        out = call_hook(run, tool, **inp)
        assert out and out["hookSpecificOutput"]["permissionDecision"] == "deny", (tool, inp)
    settings = json.loads((run.root / "agent" / "agent-settings.json").read_text())
    vault = str(run.vault.resolve())
    assert vault in settings["sandbox"]["filesystem"]["denyRead"]
    assert f"Edit(/{vault}/**)" in settings["permissions"]["deny"]
    ws = run.workspace / "brief.yaml"
    raw = yaml.safe_load(ws.read_text())
    raw["judge_rules"] = 1
    ws.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert briefs.load(ws).judge_rules == 1
    assert run.judge_rules == run.brief().judge_rules == RULES
    # and a manifest changed behind the ledger's back is caught when the daemon starts
    m = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(m | {"judge_rules": 1}))
    with pytest.raises(ValueError, match=f"judge_rules 1, but the run was started under {RULES}"):
        run.verify()


@known
@pytest.mark.parametrize("command", [
    'cat ../vau"l"t/menu.json',
    "cat ../va''ult/brief.yaml",
    "cat ..\\/vault/manifest.json",
    'X=../"vault"/ledger.sqlite; cat $X',
    "python -c 'print(1)' && head ../v'a'ult/menu.json",
])  # fmt: skip
def test_the_hook_reads_quoted_and_split_words_as_the_shell_does(run_dir, command):
    # the judge audit: the hook matched raw text, and a quote inside the word kept the
    # vault's name out of it. Each command here resolves into the vault once the shell has
    # removed its quotes
    from blinded import hook

    config = {"protected": sandbox.protected(run_dir)}
    event = {"tool_name": "Bash", "tool_input": {"command": command},
             "cwd": str(run_dir.workspace)}  # fmt: skip
    assert hook.decide(event, config) == "the command names the vault"
