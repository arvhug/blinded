import hashlib
import json
import shutil
import subprocess

import pandas as pd
import polars as pl
import pytest
import yaml
from conftest import brief_dict, make_synthetic, write_brief
from polars.testing import assert_frame_equal

from blinded import brief as briefs
from blinded import vault
from blinded.cli import main
from blinded.evaluate import scoring_window
from blinded.features import roll_mean
from blinded.ladder import RULES
from blinded.panel import Panel

known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case


def ts(text):
    return pd.Timestamp(text, tz="UTC")


@known
def test_layout(run_dir):
    ws, vt = run_dir.workspace, run_dir.vault
    for name in ("brief.yaml", "DATA.md", "RULES.md", "LEDGER.md", "data/train.parquet"):
        assert (ws / name).is_file(), name
    for name in ("features", "strategies", ".judge/inbox", ".judge/outbox"):
        assert (ws / name).is_dir(), name
    vault_files = ("train.parquet", "val.parquet", "test.parquet", "manifest.json", "brief.yaml")
    for name in (*vault_files, "ledger.sqlite"):
        assert (vt / name).is_file(), name
    assert (vt / "trials").is_dir()
    # the workspace holds train and nothing else
    assert [p.name for p in ws.rglob("*.parquet")] == ["train.parquet"]


@known
def test_windows_for_two_hours(run_dir):
    # grid 02:00:00.x .. 03:59:59.x: split points round to 03:00 and 03:30, embargo 5 min
    w = run_dir.windows()
    assert w["train"].end == ts("2024-11-28 03:00")
    assert (w["val"].start, w["val"].end) == (ts("2024-11-28 03:05"), ts("2024-11-28 03:30"))
    assert w["test"].start == ts("2024-11-28 03:35")


@cross
def test_segments_partition_the_rows_by_time(run_dir, long_brief):
    brief = briefs.load(long_brief)
    source = pl.read_parquet(brief.data.path)
    parts = {n: pl.read_parquet(run_dir.segment_path(n)) for n in vault.SEGMENTS}
    assert sum(p.height for p in parts.values()) == source.height
    assert_frame_equal(pl.concat(list(parts.values())), source)
    w = run_dir.windows()
    t = {n: p["ts"].dt.replace_time_zone("UTC") for n, p in parts.items()}
    assert t["train"].max() < w["train"].end <= t["val"].min()
    assert t["val"].max() < w["val"].end <= t["test"].min()


@cross
def test_manifest_hashes_recompute(run_dir):
    manifest = run_dir.manifest
    for name, seg in manifest["segments"].items():
        digest = hashlib.sha256(run_dir.segment_path(name).read_bytes()).hexdigest()
        assert seg["sha256"] == digest
        assert seg["rows"] == pl.read_parquet(run_dir.segment_path(name)).height
    workspace_copy = (run_dir.workspace / "data" / "train.parquet").read_bytes()
    assert hashlib.sha256(workspace_copy).hexdigest() == manifest["segments"]["train"]["sha256"]
    run_dir.verify()


@known
def test_manifest_holds_boundaries_and_hashes_only(run_dir):
    manifest = json.loads((run_dir.vault / "manifest.json").read_text())
    assert set(manifest) == {
        "run_id", "blinded", "mode", "judge_rules", "created_at", "brief", "source", "embargo",
        "windows", "segments",
    }  # fmt: skip
    assert manifest["mode"] == "protected"
    # a new run is judged under the current rules, and the workspace's brief names them
    assert manifest["judge_rules"] == RULES == run_dir.judge_rules == run_dir.brief().judge_rules
    assert briefs.load(run_dir.workspace / "brief.yaml").judge_rules == RULES
    assert set(manifest["segments"]["val"]) == {"file", "rows", "sha256"}
    assert "/" not in manifest["source"]["file"]  # a name, not the path


@known
def test_workspace_brief_points_at_train_and_keeps_the_windows(run_dir):
    ws = briefs.load(run_dir.workspace / "brief.yaml")
    assert ws.data.path == run_dir.workspace / "data" / "train.parquet"
    panel = Panel.from_brief(ws)
    # the agent's "train" is the run's train, though its file ends there
    assert scoring_window(ws, panel, "train") == scoring_window(run_dir.brief(), panel, "train")
    assert scoring_window(ws, panel, "train").end == ts("2024-11-28 03:00")
    text = (run_dir.workspace / "DATA.md").read_text()
    assert "## Splits" in text and "03:05:00" in text


@cross
def test_warm_panels_are_prefixes_of_the_full_panel(run_dir, long_brief):
    full = Panel.from_brief(briefs.load(long_brief))
    for segment in ("train", "val"):
        warm = run_dir.panel(segment)
        assert_frame_equal(warm.frame, full.frame.slice(0, len(warm)))
    assert_frame_equal(run_dir.panel("test").frame, full.frame)


@known
def test_validation_features_are_warm_at_the_first_validation_point(run_dir):
    panel = run_dir.panel("val")
    start = run_dir.windows()["val"].start
    x = panel.eval(roll_mean(pl.col("XRPUSDT.mid"), "5m"))
    first = int((panel.frame["ts"] < start).sum())
    # a 5 min window at 03:05 reaches back to 03:00, into the embargo and train
    assert x[first] is not None
    assert panel.time(first) == start


@edge
def test_refuses_data_inside_a_git_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    data = make_synthetic(repo / "events.parquet", minutes=12)
    brief = write_brief(brief_dict(data), tmp_path / "brief.yaml")
    with pytest.raises(vault.InsideGitRepo, match="data file .* inside the git repository"):
        vault.init(brief, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@edge
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_allows_data_its_repo_ignores_and_refuses_data_it_tracks(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("ignored.parquet\n")
    for name in ("ignored.parquet", "tracked.parquet"):
        make_synthetic(repo / name, minutes=12)
    ignored = write_brief(brief_dict(repo / "ignored.parquet"), tmp_path / "a.yaml")
    tracked = write_brief(brief_dict(repo / "tracked.parquet"), tmp_path / "b.yaml")
    assert vault.init(ignored, tmp_path / "run").manifest["segments"]
    with pytest.raises(vault.InsideGitRepo, match="or have that repository ignore it"):
        vault.init(tracked, tmp_path / "run2")


@edge
def test_refuses_a_run_dir_inside_a_git_repo(tmp_path, long_brief):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(vault.InsideGitRepo, match="run directory"):
        vault.init(long_brief, tmp_path / "repo" / "runs" / "one")
    assert main(["init", str(long_brief), "--run-dir", str(tmp_path / "repo" / "r")]) == 2


@edge
def test_refuses_a_non_empty_run_dir(tmp_path, long_brief):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        vault.init(long_brief, tmp_path / "run")


@edge
def test_verify_catches_a_changed_segment(tmp_path, long_brief):
    run = vault.init(long_brief, tmp_path / "run")
    path = run.segment_path("val")
    pl.read_parquet(path).head(10).write_parquet(path)
    with pytest.raises(ValueError, match="val.parquet"):
        run.verify()


@edge
def test_refuses_a_brief_that_names_an_instrument_the_data_lacks(tmp_path, long_brief):
    raw = yaml.safe_load(long_brief.read_text())
    raw["families"] = {"pairs": {"pairs": [["XRPUSDT_PERP", "BTCUSDT"]]}}
    brief = write_brief(raw, tmp_path / "brief.yaml")
    with pytest.raises(ValueError, match="does not have: BTCUSDT; it has XRPUSDT, XRPUSDT_PERP"):
        vault.init(brief, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@known
def test_rules_count_sessions_and_describe_families_only_when_there_are_some(run_dir, pairs_run):
    plain = (run_dir.workspace / "RULES.md").read_text()
    assert "3 sessions in a row without a\n   feature release" in plain
    assert "## Families" not in plain and "submissions in a row" not in plain
    rules = (pairs_run.workspace / "RULES.md").read_text()
    assert "This run studies 2 kinds of strategy" in rules
    assert "### directional" in rules and "### pairs" in rules and "### lead_lag" not in rules
    assert 'pair=("XRPUSDT_PERP", "XRPUSDT")' in rules


@known
@pytest.mark.parametrize("mode", ["protected", "naive"])
def test_init_appends_the_raw_file_notes_a_run_may_see(tmp_path, long_brief, mode):
    data = tmp_path / "data"
    data.mkdir()
    raw = yaml.safe_load(long_brief.read_text())
    shutil.copyfile(raw["data"]["path"], data / "events.parquet")
    raw["data"]["path"] = "events.parquet"
    (data / "brief.yaml").write_text(yaml.safe_dump(raw))
    # the file runs 02:00 to 04:00 on 28 Nov, train to 03:00: the 27th ends before train
    # does, the 28th after it
    (data / "DATA-raw.md").write_text(
        "# The raw files behind events.parquet\n\nchecked\n\n"
        "## 2024-11-27\n\nday one\n\n## 2024-11-28\n\nday two\n"
    )
    run = vault.init(data / "brief.yaml", tmp_path / "run", mode=mode)
    text = (run.workspace / "DATA.md").read_text()
    # appended one heading level down, after the splits
    assert "## The raw files behind events.parquet" in text and "### 2024-11-27" in text
    assert text.index("The raw files") > text.index("## Splits" if mode == "protected"
                                                    else "## Windows")  # fmt: skip
    # a protected run sees no day that reaches into validation; a naive one sees them all
    assert ("day two" in text) == (mode == "naive")


@known
def test_rules_give_the_lead_lag_built_ins_on_the_briefs_own_instruments(long_brief):
    from blinded import prompts
    from blinded.driver import Budget

    raw = yaml.safe_load(long_brief.read_text())
    raw["families"] = {"lead_lag": {"leaders": ["XRPUSDT_PERP"], "followers": ["XRPUSDT"]}}
    rules = prompts.rules("protected", briefs.parse(raw), Budget())
    for name in ("F.lead_ret", "F.lead_gap(leader, follower, k)", "F.lead_gap_z",
                 "F.lagged_corr"):  # fmt: skip
        assert name in rules, name
    assert 'F.lead_gap("XRPUSDT_PERP", "XRPUSDT", "1s")' in rules
