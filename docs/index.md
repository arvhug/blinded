# blinded

blinded runs an AI agent on a search where it can fool itself and measures how far its claims sit from what holds up. The agent works in a sandbox with the training data only, a judge outside it decides what gets released, and the final pick is scored once on a locked test. The test bed is crypto market data, replayed in a simulator; {doc}`runs` has the reports.

The judge works on trading features and strategies. It computes a feature on your data, computes it again on copies of the data cut off at several points in time, and rejects it if any past value changes once the future is gone. Features that pass are scored against forward returns, with standard errors that allow for overlapping horizons.

It never reads your code to decide. A centred rolling window, a full-sample z-score and a forward as-of join look nothing alike in source, but each of them changes its past values when you delete the future, and that change is what the judge looks for.

```{toctree}
:maxdepth: 2

getting-started
concepts
features
runs
api
```

```{toctree}
:maxdepth: 2
:caption: Protected mode

splits
run-directory
ledger
ladder
sim
families
archetypes
deflation
daemon
sandbox
harness
driver
report
forward
mlflow
demo-data
```
