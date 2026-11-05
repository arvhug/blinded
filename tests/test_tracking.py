import json
import shutil

import pytest

from blinded.cli import main
from blinded.vault import RunDir

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient  # noqa: E402

from blinded import tracking  # noqa: E402

known = pytest.mark.known_answer
edge = pytest.mark.edge_case

TRAIN = {
    "horizons": {
        "100ms": {"rank_ic": 0.41, "rank_ic_se_nw": 0.004},
        "10s": {"rank_ic": 0.15, "rank_ic_se_nw": 0.010},
    },
    "turnover": 0.42,
}


@pytest.fixture
def run(run_dir, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run_dir.root, root)
    run = RunDir(root)
    led = run.ledger()
    code = run.trials / "abc.py"
    code.write_text("x = 1\n")
    led.add_trial(
        run.run_id, kind="feature", name="basis", hypothesis="perp leads spot",
        code_hash="abc", code_path=str(code), status="accepted", train_metrics_json=TRAIN,
        val_metrics_json={"value": 0.123}, released_bool=True, ladder_best_after=0.1,
    )  # fmt: skip
    led.add_trial(run.run_id, kind="feature", name="peek", status="rejected_lookahead")
    led.probe(run.run_id, "Bash", "cat vault/val.parquet")
    led.close()
    return run


@known
def test_one_experiment_one_run_per_trial(run, tmp_path):
    uri = f"sqlite:///{tmp_path / 'store.db'}"
    exp = tracking.mirror(run, uri)
    client = MlflowClient(tracking_uri=uri)
    assert client.get_experiment(exp).name == f"blinded/{run.run_id}"
    runs = {r.data.tags["blinded.trial_id"]: r for r in client.search_runs([exp])}
    assert sorted(runs) == ["1", "2", "3"]
    basis = runs["1"]
    assert basis.data.params["kind"] == "feature" and basis.data.params["code_hash"] == "abc"
    assert basis.data.tags["hypothesis"] == "perp leads spot"
    # the horizon in milliseconds is the step, so the history is the decay curve
    history = client.get_metric_history(basis.info.run_id, "train.rank_ic")
    assert sorted((m.step, m.value) for m in history) == [(100, 0.41), (10_000, 0.15)]
    assert basis.data.metrics["ladder.best"] == 0.1 and basis.data.metrics["val.score"] == 0.123
    assert [a.path for a in client.list_artifacts(basis.info.run_id, "code")] == ["code/abc.py"]
    assert runs["3"].data.params["status"] == "probe"


@known
def test_mirroring_twice_adds_only_new_trials(run, tmp_path):
    uri = f"sqlite:///{tmp_path / 'store.db'}"
    exp = tracking.mirror(run, uri)
    led = run.ledger()
    led.add_trial(run.run_id, kind="strategy", name="s", status="timeout")
    led.close()
    assert tracking.mirror(run, uri) == exp
    runs = MlflowClient(tracking_uri=uri).search_runs([exp])
    assert len(runs) == 4
    (late,) = [r for r in runs if r.data.tags["blinded.trial_id"] == "4"]
    assert late.info.status == "FAILED"


@edge
def test_the_default_store_is_in_the_vault(run):
    uri = tracking.default_uri(run)
    assert uri == f"sqlite:///{run.vault / 'mlflow.db'}"
    assert str(run.workspace) not in uri
    assert tracking.ui_command(run) == f"mlflow ui --backend-store-uri {uri}"
    assert main(["mlflow", str(run.root)]) == 0
    assert (run.vault / "mlflow.db").is_file()
    assert not any(p.name == "mlflow.db" for p in run.workspace.rglob("*"))
    assert json.loads((run.vault / "manifest.json").read_text())["run_id"] == run.run_id
