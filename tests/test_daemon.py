import json
import shutil
import textwrap

import numpy as np
import pandas as pd
import polars as pl
import pytest

from blinded import contract
from blinded.cli import main
from blinded.client import find_judge, submit
from blinded.daemon import Daemon, Limits, score_test
from blinded.ledger import AlreadyScored
from blinded.vault import RunDir, sha256

known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case

BASIS = """
import blinded as bl
from blinded import features as F


@bl.feature("perp leads spot; the basis mean-reverts")
def basis(panel):
    b = F.basis("XRPUSDT_PERP", "XRPUSDT")
    return b - F.roll_mean(b, "5m")
"""


@pytest.fixture
def run(run_dir, tmp_path):
    """A private copy of the session's run directory."""
    root = tmp_path / "run"
    shutil.copytree(run_dir.root, root)
    return RunDir(root)


def drop(run, source, rid="r1", name=None, kind="eval"):
    """What ``blinded submit`` writes: the code, then the request."""
    (run.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
    request = {"kind": kind, "file": "f.py", "name": name}
    (run.inbox / f"{rid}.json").write_text(json.dumps(request))
    return run.inbox / f"{rid}.json"


DEFAULT = Limits()


def judge(run, source, rid="r1", limits=DEFAULT, **kw):
    daemon = Daemon(run, limits)
    response = daemon.handle(drop(run, source, rid, **kw))
    return response, daemon


@known
def test_an_accepted_feature_gets_train_metrics_and_the_ladder_only(run):
    response, daemon = judge(run, BASIS)
    (trial,) = response["trials"]
    assert trial["status"] == "accepted"
    assert set(trial["validation"]) == {"truncation", "released", "best"}
    assert set(trial["train"]["horizons"]) == {"100ms", "1s", "10s", "60s"}
    # the validation score itself stays in the vault
    row = daemon.ledger.trial(trial["trial_id"])
    assert "value" in json.loads(row["val_metrics_json"])
    assert (run.outbox / "r1.json").is_file() and not (run.inbox / "r1.json").exists()
    assert (run.vault / "requests" / "r1.py").is_file()
    assert "basis" in (run.workspace / "LEDGER.md").read_text()


def perturbed_copy(run, tmp_path):
    """The same run with validation prices changed: returns scaled by a random factor."""
    root = tmp_path / "other"
    shutil.copytree(run.root, root)
    other = RunDir(root)
    path = other.segment_path("val")
    frame = pl.read_parquet(path)
    rng = np.random.default_rng(0)
    wobble = pl.Series(np.exp(np.cumsum(rng.normal(0, 3e-5, frame.height))))
    spot = pl.col("symbol") == "XRPUSDT"
    frame = frame.with_columns(
        [pl.when(spot).then(pl.col(c) * wobble).otherwise(pl.col(c)).alias(c)
         for c in ("bid0_px", "ask0_px", "price", "mid")]
    )  # fmt: skip
    frame.write_parquet(path)
    manifest = other.manifest
    manifest["segments"]["val"]["sha256"] = sha256(path)
    (other.vault / "manifest.json").write_text(json.dumps(manifest))
    return other


def masked(response):
    text = json.dumps(response, sort_keys=True)
    out = json.loads(text)
    for trial in out["trials"]:
        trial.get("validation", {}).update(released=None, best=None)
    return json.dumps(out, sort_keys=True).encode()


@known
def test_hygiene_changing_validation_data_changes_only_released_and_best(run, tmp_path):
    other = perturbed_copy(run, tmp_path)
    two = """
    @bl.feature("spot mid momentum")
    def momentum(panel):
        return F.ret(F.mid("XRPUSDT"), "10s")
    """
    source = BASIS + textwrap.dedent(two)
    first, d1 = judge(run, source)
    second, d2 = judge(other, source)
    assert masked(first) == masked(second)
    # and the validation scores behind them really did differ
    vals = [
        [json.loads(r["val_metrics_json"])["value"] for r in d.ledger.trials("feature")]
        for d in (d1, d2)
    ]
    assert vals[0] != vals[1]
    raw1 = (run.outbox / "r1.json").read_bytes()
    raw2 = (other.outbox / "r1.json").read_bytes()
    assert masked(json.loads(raw1)) == masked(json.loads(raw2))


@known
def test_errors_return_the_type_and_line_only(run):
    source = """
    import blinded as bl


    @bl.feature("fails with data in the message")
    def broken(panel):
        raise ValueError(f"mean mid {panel.frame['XRPUSDT.mid'].mean():.6f}")
    """
    response, _ = judge(run, source)
    (trial,) = response["trials"]
    assert trial["status"] == "error"
    assert trial["reason"] == "ValueError at line 7"


@known
@pytest.mark.parametrize(
    "body, found",
    [
        ("import os", "imports os"),
        ("import subprocess", "imports subprocess"),
        ("import socket", "imports socket"),
        ("import json", "imports json"),
        ("from pathlib import Path", "imports pathlib"),
        ("open('/etc/hosts')", "calls open()"),
        ("exec('x = 1')", "calls exec()"),
        ("eval('1')", "calls eval()"),
        ("__import__('os')", "calls __import__()"),
        ("getattr(__builtins__, 'op' + 'en')", "calls getattr()"),
        ("pl.read_parquet('vault/val.parquet')", "read_parquet reads or writes files"),
        ("np.load('x.npy')", ".load reads or writes files"),
        ("().__class__.__base__.__subclasses__()", "uses __subclasses__"),
    ],
)
def test_protected_mode_rejects_before_running(run, body, found):
    marker = run.root / "ran"
    source = f"""
    import numpy as np
    import polars as pl

    import blinded as bl
    from blinded import features as F

    {body}


    @bl.feature("would leave a mark")
    def f(panel):
        return F.mid("XRPUSDT")
    """
    response, daemon = judge(run, source.replace("{marker}", str(marker)))
    (trial,) = response["trials"]
    assert trial["status"] == "rejected_static"
    assert trial["reason"].count(found) == 1
    assert daemon.ledger.trials()[-1]["status"] == "rejected_static"


@known
def test_local_mode_keeps_the_m1_scan():
    source = "import json\nimport polars as pl\nx = pl.read_parquet('a.parquet')\n"
    assert contract.scan(source).errors == []
    assert len(contract.scan(source, protected=True).errors) == 2


@known
def test_a_slow_evaluation_times_out(run):
    source = """
    import blinded as bl


    @bl.feature("never returns")
    def spin(panel):
        while True:
            pass
    """
    response, daemon = judge(run, source, limits=Limits(evaluation_seconds=2))
    (trial,) = response["trials"]
    assert trial["status"] == "timeout"
    assert "2 s" in trial["reason"]
    assert daemon.ledger.trials()[-1]["status"] == "timeout"


@known
def test_a_strategy_that_times_out_is_counted_as_a_strategy(run):
    # the trial count the strategy Sharpe is deflated for must hold it
    source = """
    import blinded as bl


    @bl.strategy()
    def spin(panel, features):
        while True:
            pass
    """
    response, daemon = judge(run, source, limits=Limits(evaluation_seconds=2))
    (trial,) = response["trials"]
    assert (trial["status"], trial["kind"]) == ("timeout", "strategy")
    assert daemon.ledger.trials()[-1]["kind"] == "strategy"


@known
def test_slow_module_level_code_times_out_too(run):
    source = """
    import blinded as bl

    while True:
        pass
    """
    response, _ = judge(run, source, limits=Limits(evaluation_seconds=2))
    assert response["trials"][0]["status"] == "timeout"


@known
def test_runaway_memory_is_oom(run):
    source = """
    import numpy as np

    import blinded as bl


    @bl.feature("keeps allocating")
    def hog(panel):
        chunks = []
        while True:
            chunks.append(np.ones(12_500_000))
    """
    limits = Limits(evaluation_seconds=30, rss_bytes=1024**3)
    response, daemon = judge(run, source, limits=limits)
    (trial,) = response["trials"]
    assert trial["status"] == "oom"
    assert daemon.ledger.trials()[-1]["status"] == "oom"


@known
def test_the_test_segment_is_scored_once(run):
    response, daemon = judge(run, BASIS)
    tid = response["trials"][0]["trial_id"]
    metrics = score_test(run, tid)
    assert "score" in metrics
    with pytest.raises(AlreadyScored):
        score_test(run, tid)
    assert main(["test", str(run.root), "--trial", str(tid)]) == 3
    rows = daemon.ledger.db.execute("SELECT COUNT(*) AS n FROM test_scores").fetchone()
    assert rows["n"] == 1


@known
def test_asking_the_inbox_for_test_is_a_probe(run):
    daemon = Daemon(run)
    response = daemon.handle(drop(run, BASIS, kind="test"))
    assert "error" in response and "trials" not in response
    (row,) = daemon.ledger.trials()
    assert (row["status"], row["kind"]) == ("probe", "probe")
    assert daemon.ledger.test_score(run.run_id) is None


@known
def test_the_ladder_survives_a_restart(run):
    response, first = judge(run, BASIS)
    assert response["trials"][0]["validation"]["released"]
    best = first.ladders["feature"].best
    again = Daemon(run)
    assert again.ladders["feature"].best == best
    assert again.ladders["feature"].best_trial == response["trials"][0]["trial_id"]
    # the same code again is not an improvement on itself
    again_response = again.handle(drop(run, BASIS, rid="r2"))
    assert again_response["trials"][0]["validation"] == {
        "truncation": "passed",
        "released": False,
        "best": best,
    }


@edge
def test_the_client_finds_the_workspace_and_gives_up_without_a_daemon(run):
    sub = run.workspace / "features"
    assert find_judge(sub) == run.workspace / ".judge"
    (sub / "f.py").write_text(BASIS)
    with pytest.raises(TimeoutError, match="daemon"):
        submit(sub / "f.py", timeout=0.3)
    with pytest.raises(FileNotFoundError):
        find_judge(run.root)


@edge
def test_a_family_the_brief_does_not_have_is_an_error_and_nothing_runs(run):
    source = """
    import blinded as bl


    @bl.feature("fails if it runs", family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
    def z(panel):
        return 1 / 0
    """
    response, daemon = judge(run, source)
    (trial,) = response["trials"]
    # had it run, the reason would have been "ZeroDivisionError at line 7"
    assert (trial["status"], trial["reason"]) == ("error", "family pairs is not in the brief")
    row = daemon.ledger.trial(trial["trial_id"])
    assert (row["family"], row["legs"]) == ("pairs", None)
    assert "family" not in trial  # a run without families says nothing about them


@pytest.fixture
def families_run(pairs_run, tmp_path):
    root = tmp_path / "pairs"
    shutil.copytree(pairs_run.root, root)
    return RunDir(root)


@known
def test_each_family_has_its_own_ladder(families_run):
    daemon = Daemon(families_run)
    assert set(daemon.ladders) == {
        ("feature", "directional"), ("strategy", "directional"),
        ("feature", "pairs"), ("strategy", "pairs"),
    }  # fmt: skip
    with pytest.raises(KeyError):
        _ = daemon.ladders["feature"]  # two families: the kind alone does not say which
    # the first step of the Ladder's table: replicates 0.0123 -+ 0.002, sd 0.002 against no
    # skill, floored to 0.005; 0.0123 > 0.005, released at round(0.0123 / 0.005) * 0.005
    reps = 0.0123 + np.array([-0.002, 0.002, -0.002, 0.002])
    item = {"kind": "feature", "name": "f", "status": "accepted", "family": "directional",
            "legs": "XRPUSDT", "val": {"truncation": "passed", "value": 0.0123,
                                       "replicates": reps, "blocks": [None] * 4}}  # fmt: skip
    out = daemon._record(item, "x = 1", "f.py", 0.0, {"submitted_by": "agent", "iteration": 1})
    assert out["validation"] == {"truncation": "passed", "released": True, "best": 0.01}
    assert (out["family"], out["legs"]) == ("directional", "XRPUSDT")
    assert daemon.ladders[("feature", "directional")].best == 0.01
    assert daemon.ladders[("feature", "pairs")].best is None
    assert [r["family"] for r in daemon.ledger.releases("feature")] == ["directional"]
    again = Daemon(families_run)  # each Ladder is restored from its own family's releases
    assert again.ladders[("feature", "directional")].best == 0.01
    assert again.ladders[("feature", "pairs")].best is None


FAMILY_FILE = """
import blinded as bl
from blinded import features as F


@bl.feature("the perp-spot spread reverts", family="pairs", pair=("XRPUSDT", "XRPUSDT_PERP"))
def spot_over_perp(panel):
    return F.spread_z("XRPUSDT_PERP", "XRPUSDT", "10m", "10m")


@bl.feature("the perp's own momentum", target="XRPUSDT_PERP")
def on_the_perp(panel):
    return F.ret(F.mid("XRPUSDT_PERP"), "10s")


@bl.strategy(family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
def pair_trade(panel, features):
    return bl.pairs("XRPUSDT_PERP", "XRPUSDT", window="10m", every="10m")


@bl.strategy(family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
def one_leg(panel, features):
    return bl.pairs("XRPUSDT_PERP", "XRPUSDT")["XRPUSDT_PERP"]


@bl.strategy(trades="BTCUSDT")
def elsewhere(panel, features):
    return F.mid("XRPUSDT") * 0
"""


@known
def test_a_run_with_families_judges_each_in_its_family(families_run):
    response, daemon = judge(families_run, FAMILY_FILE)
    by = {t["name"]: t for t in response["trials"]}
    # the pair written spot over perp is put in the brief's order
    assert (by["spot_over_perp"]["family"], by["spot_over_perp"]["legs"]) == (
        "pairs", "XRPUSDT_PERP/XRPUSDT",
    )  # fmt: skip
    assert by["spot_over_perp"]["status"] == "accepted"
    assert (by["on_the_perp"]["family"], by["on_the_perp"]["legs"]) == (
        "directional", "XRPUSDT_PERP",
    )  # fmt: skip
    pair = by["pair_trade"]
    assert pair["status"] == "accepted", pair["reason"]
    assert set(pair["validation"]) == {"truncation", "released", "best"}
    brief_row = pair["train"]["rows"]["brief"]
    assert brief_row["fills"] == brief_row["fills_XRPUSDT_PERP"] + brief_row["fills_XRPUSDT"]
    assert 0 < brief_row["gross_exposure"] <= 2
    # the shape and instrument checks, worded from the decorator, never from the code's output
    assert by["one_leg"]["status"] == "error"
    assert by["one_leg"]["reason"] == (
        "a pairs strategy returns a mapping with a position for each leg, "
        "XRPUSDT_PERP, XRPUSDT, only"
    )
    assert by["elsewhere"]["reason"].startswith("BTCUSDT is not one of the directional family's")
    rows = {r["name"]: r for r in daemon.ledger.trials()}
    assert rows["pair_trade"]["family"] == "pairs"
    assert rows["pair_trade"]["legs"] == "XRPUSDT_PERP/XRPUSDT"
    assert json.loads(rows["pair_trade"]["val_metrics_json"])["metrics"]["rows"]["brief"]
    text = (families_run.workspace / "LEDGER.md").read_text()
    assert "| family | legs |" in text and "| pairs | XRPUSDT_PERP/XRPUSDT |" in text


# -- archetypes -------------------------------------------------------------


@pytest.fixture
def archetypes_copy(archetypes_run, tmp_path):
    root = tmp_path / "archetypes"
    shutil.copytree(archetypes_run.root, root)
    return RunDir(root)


REFUSED = """
import blinded as bl


@bl.trend_following(instrument="XRPUSDT_PERP")
def trend(panel, features):
    return 1 / 0


@bl.strategy()
def bare(panel, features):
    return 1 / 0
"""


@known
def test_a_refused_archetype_and_a_bare_strategy_are_error_rows_that_never_run(
    archetypes_copy,
):
    response, daemon = judge(archetypes_copy, REFUSED)
    by = {t["name"]: t for t in response["trials"]}
    # had either run, its reason would be "ZeroDivisionError at line 7" (or 12)
    assert by["trend"]["status"] == by["bare"]["status"] == "error"
    assert by["trend"]["reason"] == (
        "trend_following (underreaction): not runnable on synthetic-14h: needs 40 weeks of "
        "validation (20 holding periods of 2 weeks); validation is 3 h 25 min."
    )
    assert by["bare"]["reason"] == (
        "this run studies archetypes; decorate with one of the archetypes in ARCHETYPES.md"
    )
    rows = {r["name"]: r for r in daemon.ledger.trials()}
    assert (rows["trend"]["archetype"], rows["trend"]["edge"]) == ("trend_following",
                                                                   "underreaction")  # fmt: skip
    assert (rows["bare"]["archetype"], rows["bare"]["edge"]) == (None, None)
    assert rows["trend"]["reason_short"] == by["trend"]["reason"][:200]
    # both error rows count as strategy trials for the deflation, like every error row
    assert daemon.ledger.n_trials("strategy") == 2
    assert daemon.ledger.n_trials("strategy", edge="underreaction") == 1
    assert "| edge | archetype |" in (archetypes_copy.workspace / "LEDGER.md").read_text()


@known
def test_each_edge_has_its_own_ladder_and_the_briefs_floor(archetypes_copy):
    daemon = Daemon(archetypes_copy)
    assert set(daemon.ladders) == {
        ("feature", "speed"), ("strategy", "speed"),
        ("feature", "relative_value"), ("strategy", "relative_value"),
    }  # fmt: skip
    # 1-minute blocks: the strategy floor is 0.05 sqrt(5 min / 1 min) = 0.05 sqrt(5)
    assert daemon.ladders[("strategy", "speed")].floor == pytest.approx(0.05 * 5**0.5, abs=1e-15)
    assert daemon.ladders[("feature", "speed")].floor == 0.005
    # the Ladder's table, step 1: 0.0123 -+ 0.002 against no skill, floored to 0.005, released
    # at 0.01 on the speed Ladder
    reps = 0.0123 + np.array([-0.002, 0.002, -0.002, 0.002])

    def item(value, offsets, archetype):
        return {"kind": "feature", "name": archetype, "status": "accepted", "family": "lead_lag",
                "legs": "XRPUSDT", "edge": "speed", "archetype": archetype, "group": "speed",
                "val": {"truncation": "passed", "value": value, "blocks": [None] * 4,
                        "replicates": value + np.asarray(offsets)}}  # fmt: skip

    common = {"submitted_by": "agent", "iteration": 1}
    out = daemon._record(item(0.0123, reps - 0.0123, "lead_lag"), "x = 1", "f.py", 0.0, common)
    assert out["validation"] == {"truncation": "passed", "released": True, "best": 0.01}
    assert (out["edge"], out["archetype"]) == ("speed", "lead_lag")
    assert daemon.ladders[("feature", "relative_value")].best is None
    # step 2, from another archetype of the same edge: it moves with step 1 (paired sd 0), so
    # the step is the 0.005 floor and 0.0140 - 0.010 = 0.004 does not clear it
    out = daemon._record(item(0.0140, reps - 0.0123, "order_book_imbalance"), "x = 2", "g.py",
                         0.0, common)  # fmt: skip
    assert out["validation"] == {"truncation": "passed", "released": False, "best": 0.01}
    (row,) = daemon.ledger.releases("feature")
    assert (row["family"], row["group"]) == ("lead_lag", "speed")
    again = Daemon(archetypes_copy)  # restored from the edge's own releases
    assert again.ladders[("feature", "speed")].best == 0.01
    assert again.ladders[("feature", "relative_value")].best is None


@known
def test_the_archetype_and_kind_of_a_file_that_never_finished():
    from blinded.daemon import _declared_archetype, _declared_kind

    two_obi = """
import blinded as bl

imb = bl.order_book_imbalance(instrument="XRPUSDT")


@imb.feature("the bigger queue leans the next move")
def queue(panel):
    return panel


@bl.order_book_imbalance(instrument="XRPUSDT", requires=["queue"])
def lean(panel, features):
    return features["queue"]
"""
    assert _declared_archetype(two_obi) == "order_book_imbalance"
    assert _declared_kind(two_obi, "lean") == "strategy"
    assert _declared_kind(two_obi, "queue") == "feature"
    mixed = """
import blinded as bl


@bl.lead_lag(leader="A", follower="B")
def follow(panel, features):
    return 0


@bl.pairs_trading(a="A", b="B")
def pair(panel, features):
    return 0
"""
    assert _declared_archetype(mixed) is None
    assert _declared_kind(mixed, None) == "strategy"
    bound = """
import blinded as bl

flow = bl.trade_flow_persistence()


@flow
def ride(panel, features):
    return 0
"""
    assert (_declared_archetype(bound), _declared_kind(bound, "ride")) == (
        "trade_flow_persistence", "strategy",
    )  # fmt: skip


HOLD = """
import blinded as bl
import polars as pl


@bl.strategy(requires=["basis"])
def always_long(panel, features):
    # one position from the first bar to the last: one bet, however many blocks it spans
    return pl.lit(1.0) + 0.0 * features["basis"].fill_null(0.0)
"""

FAST = """
import blinded as bl


@bl.strategy(requires=["basis"])
def basis_threshold(panel, features):
    return bl.threshold(features["basis"], z_in=2.0, z_out=0.5, every="2m")
"""


@known
def test_a_strategy_bootstraps_in_blocks_as_long_as_its_holds(run):
    daemon = Daemon(run)
    daemon.handle(drop(run, BASIS, rid="f"))
    rows = {}
    for rid, source in (("hold", HOLD), ("fast", FAST)):
        (trial,) = daemon.handle(drop(run, source, rid=rid))["trials"]
        assert trial["status"] == "accepted", trial
        rows[rid] = json.loads(daemon.ledger.trial(trial["trial_id"])["val_metrics_json"])
        assert set(trial["validation"]) == {"truncation", "released", "best"}
    brief = run.brief()
    step = brief.grid.step
    block = int(brief.stats.block / step)
    hold, fast = rows["hold"], rows["fast"]
    # held the whole window: its own block is the window, so validation holds one of them
    assert hold["hold_steps"] == hold["block_steps"] > block
    assert hold["own_blocks"] == 1 and hold["releasable"] is False
    assert hold["hold_hours"] == pytest.approx(hold["hold_steps"] * step / pd.Timedelta("1h"))
    # a strategy that trades within a block keeps the brief's block and its Sharpe
    assert fast["block_steps"] == block and fast["releasable"] is True
    assert fast["hold_steps"] < block
    assert fast["value"] == pytest.approx(fast["metrics"]["rows"]["brief"]["sharpe_per_sqrt_hour"])


@known
def test_too_few_holds_is_never_released_whatever_the_score(run):
    # the guard, apart from the data: a score far above any step, tight replicates
    daemon = Daemon(run)
    val = {"truncation": "passed", "value": 5.0, "replicates": 5.0 + 0.01 * np.arange(-5, 6),
           "blocks": [1.0, 2.0, 3.0]}  # fmt: skip
    common = dict(submitted_by="agent", iteration=1)

    def record(name, **extra):
        item = {"kind": "strategy", "name": name, "status": "accepted",
                "val": val | extra}  # fmt: skip
        return daemon._record(item, "", "p.py", 0.0, common)["validation"]

    assert record("held", releasable=False) == {"truncation": "passed", "released": False,
                                                "best": None}  # fmt: skip
    assert record("ok", releasable=True)["released"] is True
