# The loop

`blinded run BRIEF --run-dir DIR` is the whole loop. It builds the run directory, starts the judge daemon, and runs one fresh agent session at a time until a phase rule or a hard stop ends the run, then scores the final strategy on test once.

```console
$ blinded run brief.yaml --run-dir ~/runs/xrp-1 --model sonnet --max-iterations 4 --max-usd 3
run 7c3f1a2b9d04 (protected) in /Users/you/runs/xrp-1
stopped: finalised; final trial: 12 (the released strategy you expect to do best)
```

The driver owns everything the agent cannot: the daemon, the budget, the phase, the one shot at test. The agent sees only its workspace.

## The phases

1. **Features.** The agent proposes features. The phase ends once 60% of `max_iterations` have run, or when every family is closed (below). A family closes after `budget.feature_streak_sessions` (3) of its sessions in a row without a Ladder release of one of its features, or when the agent returns `phase_done` in two of its sessions in a row.
2. **Strategies.** The agent proposes strategies over the accepted features. The phase ends when every family has had `budget.strategy_streak_sessions` (3) of its sessions in a row without a strategy release, or at the budget.
3. **Finalise.** One session names the final strategy by trial id; no new code. The last of `max_iterations` is kept for it.
4. **Test.** The driver scores the final strategy on test, through {func}`blinded.daemon.score_test`, which the ledger's single-row constraint makes idempotent.

The streaks count sessions, not submissions. A session counts against its family when no trial of the phase's kind and of that family was released in it, whether it submitted seven rejected ideas or nothing (an empty session also counts towards the `stuck` stop below), and a release resets the count. Three sessions are three to nine attempts at the one to three ideas a session the prompt asks for, and the number does not depend on how the agent batches. Earlier runs counted submissions, 8 features and 6 strategies in a row, the same order of attempts, and an agent that batched seven ideas into a session could end a phase in two sessions: the protected Tardis run used 7 of its 20. In naive mode there is no Ladder, so the streak rules are off and the phases turn on `phase_done` and the budget alone.

## Families

A brief with a `families` block studies several kinds of strategy in one run: one vault, one ledger with a `family` column, one test opened once. Each session of a phase is for one family. Among the families still open in the phase, the driver picks the one with the fewest sessions so far for its share, $\text{sessions}(f) / \text{share}(f)$, ties to the first in the brief; with shares 0.5, 0.25 and 0.25, twelve sessions go $D, L, P, D, D, L, P, D, D, L, P, D$, six, three and three. When a family closes, its remaining sessions go to the others by the same rule. The assignment is a function of the ledger and `driver/state.json`, which records each iteration's family, so `--resume` reproduces it.

With the Binance demo's shares of 0.3, 0.3 and 0.4 the order is $D, L, P, P, D, L, P, D, L, P, D, L$ while all three are open: the first session goes to directional by brief order, and pairs takes the fourth because its count over its share, $1/0.4 = 2.5$, is then the smallest. Twenty sessions leave at most twelve for features (60%) and one for the finalise. A scripted run of twelve sessions on that file, with the fake harness of the tests (`scripts/binance_fake.py`), went $D, L, P, P, D, L$ in features, each family closing on its second `phase_done`, and through the strategies phase to the finalise, with every family's charts in its report and `blinded replay` matching every number.

In a brief with `archetypes` the same rules run over edges instead of families ({doc}`archetypes`): the groups are the edges that have an archetype the file can run, each session is for one edge, named in the prompt with its archetypes ("This session is for the relative value edge: pairs_trading. Submit ideas with those decorators."), an edge closes on its own streak, and `driver/state.json` keeps the edge of each iteration under the same `families` key. With speed and relative value at 0.5 each, eight feature sessions go $S, R, S, R, S, R, S, R$; if relative value releases nothing, its third session, the sixth, closes it and the seventh and eighth go to speed. `BUDGET.md` then has a row per edge with a `members` column, its archetypes, and the line that says which shares were dropped for an edge with nothing to run. On the hourly trend file three edges run at a third each, and a scripted run there (`scripts/trend_fake.py`) went $S, R, U$ five times, six feature sessions and nine strategy sessions, each edge closing on its own streak, then the finalise. If the second feature session of each edge has nothing to submit, those three empty sessions in a row are the `stuck` stop, and the run ends there before any strategy.

Submissions of another family inside a session are judged and recorded and count towards that family's trials and the deflation count, and towards nobody's streak. The session's family is named in the prompt and in `BUDGET.md`, which has a row per family: its sessions this phase, its sessions in a row without a release against the limit, its released best, and whether it is open or why it closed. A brief without `families` has one directional family, every session is its, and nothing about families appears in the agent's files.

## The final choice

The agent names a trial. In protected mode its pick stands only if that trial was released, whatever its family; otherwise, and when the agent names nothing or names a non-strategy, the driver overrides with the released strategy whose released best is highest across families, ties to the later trial, and records why. Within a family the Ladder releases only improvements, so with one family this is the last released strategy. The released best is a number the agent already saw, so the override leaks nothing new. With nothing released the final strategy is flat, and the report says there is no strategy. In naive mode the pick stands if it is an accepted strategy, else the driver takes the one with the best claimed Sharpe; that pick is then rescored by the protected judge on the locked test with the brief's real latency and fees ({func}`blinded.driver.rescore`), on both legs for a pairs strategy.

## Hard stops

Checked before every session:

- **any probe** halts the run at once; nothing is scored on test, because a probe means the sandbox needs attention before any number can be trusted. This outranks every other stop.
- **`max_usd`**, from the sessions' own `total_cost_usd`.
- **`wall_hours`**.
- **`max_iterations`**.
- **stuck**: three sessions in a row that add no ledger row.

After any hard stop but a probe, the driver finalises without the agent: the last released strategy, or flat.

## What the agent reads, and where the driver writes

The driver writes the workspace files the agent reads: `RULES.md` and the iteration prompts (from {mod}`blinded.prompts`), and `BUDGET.md` before every session, with the iteration, the phase, the spend and wall time against their caps, the current streak against its limit (a row per family in a run with families), and any phase change and its reason. `NOTES.md` is the agent's, created empty and never read by the driver except for the report. The JSON schema each session returns is in `agent/iteration_schema.json`, the appended system prompt in `agent/system_append.md`.

Everything the driver keeps for itself is under `DIR/driver/`, which the sandbox denies to the agent like the vault: `state.json` (the phase of each iteration, the budget, the final choice), `logs/iter_NNN.jsonl` (one session transcript each), `events.jsonl` (sandbox refusals that were not probes), `daemon.log` and `daemon.pid`.

## Resume

Every count the rules use is recomputed from the ledger, and all state is on disk, so `blinded run BRIEF --run-dir DIR --resume RUN_ID` continues from where a crash left off. The crashed session's transcript is read to close its iteration row; its submissions already reached the daemon and are in the ledger, so they count. The driver writes the current iteration number to `vault/iteration` before each session, where only the daemon can read it, so a submission is tagged with its iteration even though the request cannot set one. A crashed test-scoring step is safe to rerun because the score is written once.

## The budget and the model

`budget` in the brief sets `wall_hours`, `max_iterations`, `max_usd`, `model`, `effort`, `max_turns`, the per-session `iteration_usd` and `iteration_minutes`, and the streaks `feature_streak_sessions` and `strategy_streak_sessions`; the command-line flags override it, apart from the streaks, which have none. Each session is launched with its own `--max-budget-usd`, the smaller of `iteration_usd` and what is left of `max_usd`, so a session whose cost never comes back (a kill) cannot overspend the run. The defaults for a real run are `--model opus --effort high`; the smoke runs used `--model sonnet` to keep their spend small.
