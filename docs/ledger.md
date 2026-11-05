# The ledger

Every evaluation in a run is a row in `vault/ledger.sqlite`: accepted, rejected, errored, timed out, out of memory, and every attempt to reach the vault (`probe`). Nothing is filtered on the way in. The ledger is what the deflated Sharpe ratio and PBO are computed from, and those are only honest if the trial count includes the failures, since a rejected strategy was still an attempt to find something that scores on validation.

## Tables

| table | one row per | what it is for |
|---|---|---|
| `runs` | run | brief and data hashes, the split manifest, mode, the final trial and its test score |
| `trials` | evaluation | kind (`feature`, `strategy`, `probe`), family, legs, edge and archetype, name, hypothesis, `code_hash`, status, a short reason, train metrics, validation metrics, the Ladder's output, wall time |
| `blocks` | trial, segment and 5 minute block | per-block P&L (bps) and IC, for the bootstrap, PBO and stability tables |
| `releases` | Ladder release | phase, family, group (the Ladder's key), trial, best before and after, and the $\eta$ that was used |
| `iterations` | agent session | session id, cost, turns (filled by the driver) |
| `test_scores` | run | the one test score; `run_id` is the primary key |

Statuses are `accepted`, `rejected_lookahead`, `rejected_static`, `rejected_nondeterministic`, `error`, `timeout`, `oom` and `probe`. A `CHECK` constraint refuses anything else.

`family` is `directional`, `lead_lag` or `pairs`, `directional` by default, and `legs` the instrument a trial predicts or trades, or its pair as `A/B` in the brief's order. A feature or strategy that names a family, instrument or pair the brief does not have is an `error` row with the reason, written before it is evaluated; there is no status for it. The report's table per family reads these columns: what each family submitted, had accepted and had released of each kind, and, in the naive comparison, each family's judge calls in both runs. The `family` columns came after the first ledgers: opening one of those reads `PRAGMA table_info` and adds the columns it lacks, with every existing row `directional`, so an old run directory opens, reports and replays as it did.

`edge` and `archetype` are the tag an archetype's decorator puts on a trial ({doc}`archetypes`), empty for a bare `bl.feature` or `bl.strategy`; a refused archetype's `error` row carries them too, so the report can count it. `releases.group` is the key of the Ladder a release was on: the edge in a run with archetypes, the family otherwise. These three came after the families columns, and opening an older ledger adds them the same way, filling `group` from each release's `family`, which is what its Ladder was keyed by. {meth}`~blinded.ledger.Ledger.n_trials` takes an `edge` as well as a `family`, and counts every distinct hash of the kind, error rows included, when given neither.

## The test set is single-use by construction

`test_scores` has `run_id` as its primary key. Scoring test inserts a row there inside one transaction with the update to `runs`; a second insert for the same run violates the key, the transaction rolls back, and the judge raises `AlreadyScored`. It does not matter whether the second call comes from a retry, a bug in the driver or a person at the command line: the database refuses it, and a crashed scoring step can be rerun safely because it either committed or did not.

## What counts as a new trial

`code_hash` is the SHA-256 of the file's abstract syntax tree, dumped without line numbers, with docstrings removed, plus the name of the feature or strategy. Parsing throws away comments, blank lines, spacing, line breaks and redundant parentheses, so

```python
basis - F.lag(F.roll_mean(basis, 3000), 1)  # trailing 5 min mean
```

and

```python
(basis - F.lag(F.roll_mean(basis, 3000), 1))
```

hash the same, while changing `3000` to `3001`, `-` to `+` or `1e4` to `1e3` gives a new hash. The deflation count $N$ is the number of distinct hashes among strategy trials, of every family: two families with three distinct strategies each are six trials. A cosmetic edit therefore cannot inflate $N$ to make the best result look more selective than it was, and a real edit cannot hide inside an old hash.

The hash covers the whole file, so editing one feature in a file of three changes the hash of all three. That over-counts, which errs on the side of more deflation. Renaming a variable is a real edit as far as the hash is concerned; that over-counts too.

## What the agent sees

After every trial the daemon rewrites `workspace/LEDGER.md` from the database: one row per trial with its id, kind, family and legs (in a run with families; with the edge and archetype after the family in a run with archetypes), name, status, train rank IC at each horizon with its Newey-West SE, train net Sharpe for strategies, whether it was released and the released best, and the hypothesis (or the reason for a rejection). That is the loop's memory between sessions. Validation numbers do not appear in it beyond the Ladder's `released` and `best`, which the agent was told anyway. When a market-making strategy has been judged, a second table follows: each maker trial's net bps/hour on train under each fill model and at each maker fee, the scored row first, and the model the judge's rule took on train; the model it took on validation stays in the vault ({doc}`ladder`). The file is regenerated each time, so editing it achieves nothing, and the database itself is in the vault.

The ledger module uses only the standard library, so the sandbox hook can append a probe row without importing polars. SQLite runs in WAL mode, so the hook and the daemon can both write while the other has the file open.
