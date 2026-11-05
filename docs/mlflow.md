# Browsing a run in MLflow

The ledger is a SQLite file, which is the right place for the truth and an awkward place to look at forty trials. MLflow is a tracking server with a web UI built for exactly this: lists of runs you can sort and filter, side-by-side comparisons, metric plots. blinded can copy a run's ledger into an MLflow store and leave you to browse it.

```console
$ pip install "blinded[mlflow]"          # or: uv sync --extra mlflow
$ blinded mlflow ~/runs/xrp-1
experiment 1; browse with:
  mlflow ui --backend-store-uri sqlite:////Users/you/runs/xrp-1/vault/mlflow.db
$ mlflow ui --backend-store-uri sqlite:////Users/you/runs/xrp-1/vault/mlflow.db
```

and open <http://127.0.0.1:5000>. Running `blinded mlflow` again later copies only the trials added since; each copied trial carries its ledger id as a tag, and those are skipped.

## The pieces of MLflow

MLflow stores six kinds of thing, nested like this:

```text
tracking store (a database: here, one SQLite file)
  experiment            a named group of runs
    run                 one attempt at something, with a start time and a status
      params            inputs, written once: name -> string
      metrics           outputs: name -> a history of (value, step, timestamp)
      tags              free-form labels: name -> string, can be changed later
      artifacts         files: kept in an artifact store, listed under the run
```

**Tracking store.** Where the metadata lives. `--backend-store-uri` names it. A URI starting `sqlite:///` is a SQLite file, and the path after the three slashes is the file; an absolute path adds its own leading slash, which is why the command has four. MLflow creates the tables the first time it opens the file.

**Experiment.** A named container of runs, with an id and an artifact location, the directory where its runs' files go. Experiments are what the UI's left-hand list shows.

**Run.** One execution: a start and end time, a status (`FINISHED`, `FAILED`, `RUNNING`, ...), and everything below. In ML use a run is one training job; here it is one trial.

**Param.** A key and a string value, logged once per run and never changed: the configuration that went in. Logging the same param twice with a different value is an error, which is the point. Params are what you filter on: `params.kind = "strategy"`.

**Metric.** A key with a history of numbers. Each entry is `(value, step, timestamp)`; `step` is an integer you choose, usually the training epoch. The runs table shows a metric's latest value; the run's page plots its history against the step. You can log a metric once with step 0 or many times with increasing steps.

**Tag.** A key and a string, like a param, except that it can be overwritten and is meant for labels and notes. MLflow uses a few itself, such as `mlflow.runName`, the name shown in the UI.

**Artifact.** Any file, stored in the experiment's artifact location under the run's id and listed on the run's page: a model, a plot, or here the code.

## How blinded maps onto it

| ledger | MLflow |
|---|---|
| a run directory | the experiment `blinded/<run id>`, artifacts in `vault/mlartifacts/` |
| a trial | a run named `0007 feature basis`, tagged `blinded.trial_id = 7` |
| kind, name, status, code hash, submitted by, iteration | params |
| hypothesis, the short reason for a rejection | tags |
| train rank IC, IC, their SEs and markout at each horizon | metrics `train.rank_ic`, `train.ic`, ..., **the step is the horizon in milliseconds** |
| train turnover and autocorrelation; a strategy's sim rows | metrics at step 0: `train.turnover`, `train.brief.net_bps_per_hour`, `train.latency_0_fees_0.gross_bps_per_hour`, ... |
| the validation score, the Ladder's output | metrics `val.score`, `ladder.released`, `ladder.best` |
| the submitted file | artifact `code/<hash>.py` |
| status error, timeout or oom | run status `FAILED`; every other status is `FINISHED`, with the real status as a param |

The one non-obvious choice is using the horizon as the step: a trial's rank IC at 100 ms, 1 s, 10 s and 60 s is logged as one metric with four steps, 100, 1000, 10000 and 60000, so the metric's chart on the run page is the feature's decay curve, and comparing runs overlays their decay curves. The runs table shows the value at the largest step, the 60 s horizon.

In the UI, sort the runs table by `ladder.best` to see the leaderboard climb, filter with `params.status = "accepted"`, select a handful of runs and press Compare to put their params and metrics side by side.

## Where the store lives, and why

The store and its artifacts go in the vault: `vault/mlflow.db` and `vault/mlartifacts/`. They hold `val.score`, the validation score the Ladder exists to hide, so they must never be readable from the workspace; the agent's sandbox denies the vault, and it has no network to reach a UI. Serve the UI yourself, outside the run.

The ledger stays the source of truth. The mirror is one-way and can be thrown away: delete `vault/mlflow.db` and `vault/mlartifacts/` and run `blinded mlflow` again to rebuild it. Nothing blinded computes reads from MLflow.
