# The judge daemon

The agent and the judge never share a process. The agent works in `workspace/`; the judge daemon, started outside the agent's sandbox, is the only process that reads `vault/`. They talk by dropping files.

```console
$ blinded daemon ~/runs/xrp-1           # the judge, in its own terminal or under the driver
$ cd ~/runs/xrp-1/workspace
$ blinded submit features/basis.py      # the agent's side
```

## The file drop

`blinded submit FILE [--name NAME]` finds the `.judge/` directory of the workspace it is run from, copies the file to `.judge/inbox/<id>.py`, writes the request `<id>.json` next to it, and waits for `.judge/outbox/<id>.json`. The request is written last, through a rename, so the daemon never picks up half of one. The id is random, so two submissions of the same file are two requests.

There is no socket and no port, which means it works inside any sandbox, and every request and response is a file you can read afterwards: the daemon moves each request into `vault/requests/` once it has answered. The client uses only the standard library. The command is `blinded submit` rather than a separate `judge` program, so the tool keeps one name.

The inbox takes evaluation requests only. Anything else, such as a hand-written request for the test segment, is refused and recorded in the ledger as a probe.

## What happens to a submission

1. **Static scan, protected.** On top of the local scan, protected mode rejects (`rejected_static`) any import outside a short allowlist (`numpy`, `pandas`, `polars`, `blinded`, `math`, `statistics`, `functools`, `itertools`, `collections`, `dataclasses`, `typing`, `__future__`), calls to `open`, `exec`, `eval`, `compile`, `__import__`, `getattr` and similar, library calls that read or write files (`read_*`, `scan_*`, `np.load`, `tofile`, ...), dunder attributes such as `__builtins__` and `__subclasses__`, and attributes that reach a refused module through a library that holds it (`pd.io.common.os`, `np.ctypeslib.ctypes`; also `as_strided`, which reads past an array's end). A call named like a write, `to_*`, `write*`, `save*` or `dump*`, is allowed only when it is a conversion that never takes a path ({data}`blinded.contract.SAFE_CALLS`: `to_numpy`, `to_list`, `to_series`, `to_pandas`, `to_timedelta` and their kind). Every published trial file, 628 of them, uses only those five; `to_json`, `to_html` and the rest of the `to_*` writers scanned clean before 20 Oct 2026, and `to_json(path)` wrote 42,387 panel rows, validation included, into the workspace. Rejected code is never executed. `blinded eval` in local mode keeps the static scan, with the same rule for calls named like a write.
2. **Workers, never the judge itself.** The code is copied to `vault/trials/<hash>.py`. A freshly spawned judge process builds the panels from the vault's own copies of the data, but never imports the code: one worker reads which features and strategies the file defines, and every evaluation after it, the full panel or one cut, is a fresh worker of its own ({mod}`blinded.isolate`), sent one panel and nothing after it. Before the worker imports any of the code it installs an audit hook ({func}`blinded.isolate.guard`) that refuses `open` for writing outside the worker's own temporary directory, reads and directory listings anywhere but Python's installation, the environment, blinded's own source, the worker's directory and the trial files ({func}`blinded.isolate.readable`; a missing file is refused as a present one is, before the OS is asked), any change to the file system outside that directory, links, `os.chdir`, sockets, new processes and signals, ctypes, databases, and an import of ctypes, socket or subprocess from the code. A refusal is an `error` row that says what was refused and on which line, even when the code catches the exception. Each worker also runs under Anthropic's sandbox runtime `srt` (on `PATH`, or its path in `BLINDED_SRT`), with writes only in its directory, no reads of the vault and no network: that stops what Python's audit events cannot see, a write from a library's C or Rust code such as polars' `write_csv`. Without `srt` the judge does not start ({func}`blinded.isolate.fence`) unless `BLINDED_UNFENCED=1` says to run the code behind the hook and the scan alone, as the tests do; each judge start is written to `vault/fence.jsonl`, and the report of a run judged without `srt` says so. A library's own read of a file outside the run is not refused by `srt`, whose read rules cover the run's directories: the static scan names the calls that take a path, however they are imported, and refuses a library's private names. When an evaluation on validation or test fails, the agent hears only that it raised or was refused on that panel: an exception's name and line are the code's to choose, so there they could carry the panel's data. The vault keeps the words. Each feature and strategy is first checked against the brief's families ({func}`blinded.contract.bind`): its family is in the brief, its `target` or `trades` is one of the family's instruments, its `pair` one of the family's pairs, in either order. One that fails is an `error` row with the reason and is not evaluated. A pairs feature is scored against the spread return of its pair and a pairs strategy is simulated on both legs; a strategy that returns the wrong legs is an `error` whose reason is the rule, never the output.
3. **Pre-check, then the truncation test on train.** One full run and the 0.50 cut come first: a blatant leak is found after those two evaluations, and a third, a second full run, then says whether it was look-ahead or noise. A clean feature goes on to the determinism run and the other six cuts, nine evaluations in all, as in local mode.
4. **Train metrics.** Exactly what `blinded eval --window train` gives in the workspace, from the same bytes.
5. **The truncation test again, on the warm validation panel.** Only its pass or fail comes back.
6. **The validation score and the Ladder.** Rank IC at the primary horizon for a feature, net Sharpe per $\sqrt{\text{hour}}$ for a strategy, with paired bootstrap replicates; the Ladder of the item's family and kind decides. The score, the replicates and the per-block numbers go into the ledger. A market-making strategy is scored on the worse of two queue models ({doc}`ladder`). A score that measures nothing, a rank correlation of a constant or a Sharpe of fewer than two blocks, is stored as null with a `score_note` and never given to the Ladder; the agent hears `released: false` and the best, as of any loss.
7. **Answer.** One ledger row per feature or strategy in the file, `LEDGER.md` regenerated, and the response written.

A strategy's `requires` resolves to the latest accepted feature of each name, loaded from its copy in the vault, not from the workspace, so editing a feature file after acceptance does not change what a strategy is scored with.

## Limits

Each evaluation of the code may take 120 s of wall time, a whole submission 15 minutes, and each process 4 GB of resident memory. Past either time limit the status is `timeout`; past the memory limit it is `oom`. The judge process watches each worker's clock and reads its resident set size with `ps` every twentieth of a second, and kills it when a limit is passed; the daemon does the same for the judge process, which tells it it is alive while a worker runs. Memory is watched from outside rather than capped with `RLIMIT_AS` because macOS does not enforce that limit. Killing from outside also stops a computation inside polars' Rust code, which a signal handler in the worker could not interrupt. Code at the top level of the file counts against the evaluation that imports it.

## What the agent is told

A response looks like this, for a file with a causal feature and a leak (the synthetic file in the test suite):

```json
{
  "id": "5f0c...",
  "trials": [
    {
      "kind": "feature", "name": "basis", "status": "accepted", "trial_id": 5,
      "reason": "passed: deterministic, matches the full run at all 7 cuts",
      "train": {"horizons": {"10s": {"rank_ic": 0.1484, "rank_ic_se_nw": 0.0102, "...": "..."}}},
      "validation": {"truncation": "passed", "released": true, "best": 0.14}
    },
    {
      "kind": "feature", "name": "peek", "status": "rejected_lookahead", "trial_id": 6,
      "reason": "cut at 2024-11-28T02:32:35.637000+00:00: first mismatch at ..., 1/19556 points differ"
    }
  ]
}
```

A response never contains a validation or test value, a per-block number from either, a traceback, an exception message, or a feature's values. Errors come back as the exception type and the line in the submitted file, `ValueError at line 7`, because a message can carry data. Train details come back in full: train is the agent's data anyway. The response has no timestamps or timings, so it is a function of the train data, the code and the Ladder's output.

The suite tests that last property directly: it builds two run directories that differ only in the validation prices, submits the same file to each, and checks that the two response files are byte for byte identical once `released` and `best` are masked, while the validation scores stored in the two vaults differ.

## Scoring test, once

```console
$ blinded test ~/runs/xrp-1 --trial 17
$ blinded test ~/runs/xrp-1 --trial 17
blinded test: run 3b9e... has already been scored on test; the test segment is scored once
```

Test scoring is for the driver or a person, never the inbox. It runs the trial on the warm test panel in a child under the same limits and records the result in `test_scores`, whose primary key is the run id, so a second score is refused by the database; see {doc}`ledger`.
