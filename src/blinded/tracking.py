"""Mirror the ledger into MLflow, to browse a run's trials in the MLflow UI.

Optional: ``pip install "blinded[mlflow]"``. SQLite in the vault stays the
source of truth; the mirror is a view of it and can be rebuilt at any time.
The tracking store also lives in the vault, never in the workspace, because it
holds validation numbers.

The mapping: one MLflow experiment per run directory, one MLflow run per
trial, the trial's identity as params, train metrics with the horizon in
milliseconds as the step, the Ladder's output and the validation score as
metrics, and the submitted code as an artifact.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from blinded.durations import to_ns
from blinded.vault import RunDir

PARAMS = ("kind", "name", "status", "code_hash", "submitted_by", "iteration", "parent_trial_id")
FAILED = {"error", "timeout", "oom"}


def default_uri(run: RunDir) -> str:
    """``sqlite:///DIR/vault/mlflow.db``."""
    return f"sqlite:///{run.vault / 'mlflow.db'}"


def ui_command(run: RunDir, uri: str | None = None) -> str:
    """The command that serves the MLflow UI for this run's store."""
    return f"mlflow ui --backend-store-uri {uri or default_uri(run)}"


def mirror(run: RunDir, uri: str | None = None) -> str:
    """Copy every trial not yet mirrored into MLflow; returns the experiment id.

    Trials are matched on the ``blinded.trial_id`` tag, so calling it again after
    more trials adds only the new ones.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise ImportError('MLflow is optional: pip install "blinded[mlflow]"') from None
    uri = uri or default_uri(run)
    client = MlflowClient(tracking_uri=uri)
    name = f"blinded/{run.run_id}"
    exp = client.get_experiment_by_name(name)
    if exp is None:
        artifacts = (run.vault / "mlartifacts").resolve().as_uri()
        exp_id = client.create_experiment(
            name, artifact_location=artifacts, tags={"blinded.brief": run.brief().name}
        )
    else:
        exp_id = exp.experiment_id
    done = {r.data.tags.get("blinded.trial_id") for r in _all_runs(client, exp_id)}
    ledger = run.ledger()
    for row in ledger.trials():
        if str(row["trial_id"]) not in done:
            _mirror_trial(client, exp_id, row)
    ledger.close()
    return exp_id


def _mirror_trial(client, exp_id: str, row) -> None:
    from mlflow.entities import Metric, Param

    started = _ms(row["submitted_at"])
    tags = {"blinded.trial_id": str(row["trial_id"]), "mlflow.runName": _run_name(row)}
    if row["hypothesis"]:
        tags["hypothesis"] = row["hypothesis"]
    if row["reason_short"]:
        tags["reason"] = row["reason_short"]
    mrun = client.create_run(exp_id, start_time=started, tags=tags)
    rid = mrun.info.run_id
    params = [Param(k, str(row[k])) for k in PARAMS if row[k] is not None]
    metrics = []
    train = json.loads(row["train_metrics_json"] or "{}")
    for horizon, values in (train.get("horizons") or {}).items():
        step = to_ns(horizon) // 1_000_000
        for key, value in values.items():
            if value is not None:
                metrics.append(Metric(f"train.{key}", float(value), started, step))
    for key in ("turnover", "autocorr"):
        if train.get(key) is not None:
            metrics.append(Metric(f"train.{key}", float(train[key]), started, 0))
    for variant, values in (train.get("rows") or {}).items():
        for key, value in values.items():
            if value is not None:
                label = variant.replace(", ", "_").replace(" ", "_")
                metrics.append(Metric(f"train.{label}.{key}", float(value), started, 0))
    val = json.loads(row["val_metrics_json"] or "{}")
    if isinstance(val.get("value"), int | float):
        metrics.append(Metric("val.score", float(val["value"]), started, 0))
    if row["released_bool"] is not None:
        metrics.append(Metric("ladder.released", float(row["released_bool"]), started, 0))
    if row["ladder_best_after"] is not None:
        metrics.append(Metric("ladder.best", float(row["ladder_best_after"]), started, 0))
    if row["wall_seconds"] is not None:
        metrics.append(Metric("wall_seconds", float(row["wall_seconds"]), started, 0))
    client.log_batch(rid, metrics=metrics, params=params)
    if row["code_path"] and Path(row["code_path"]).is_file():
        client.log_artifact(rid, row["code_path"], artifact_path="code")
    status = "FAILED" if row["status"] in FAILED else "FINISHED"
    client.set_terminated(rid, status=status, end_time=started)


def _all_runs(client, exp_id: str) -> list:
    out, token = [], None
    while True:
        page = client.search_runs([exp_id], max_results=1000, page_token=token)
        out += list(page)
        token = page.token
        if not token:
            return out


def _run_name(row) -> str:
    return f"{row['trial_id']:04d} {row['kind']} {row['name'] or ''}".strip()


def _ms(stamp: str) -> int:
    return int(datetime.fromisoformat(stamp).timestamp() * 1000)
