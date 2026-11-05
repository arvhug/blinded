# blinded

blinded runs an AI agent on a search where it can fool itself, then measures how far its claims sit from what holds up. The agent works in an OS sandbox with the training data only. Validation and test stay in a vault it can't read. A judge outside the sandbox recomputes everything the agent submits, scores validation through a rule that leaks a bounded amount, and scores the final pick once on a locked test. The same agent can also run on the same budget with those protections off, so each experiment is a pair of runs.

The test bed is crypto market data, where look-ahead bugs and overfit strategies are easy to produce and a locked test is cheap to score. Every result below comes from a simulator replaying recorded public data. None of it is live trading: no live order has ever been sent, and the code has no way to place one.

## Naive and protected runs

`blinded run brief.yaml --run-dir DIR` runs the loop. Headless `claude -p` sessions write features and strategies, the daemon judges them, and the final strategy is scored once on the locked test. `--mode naive` runs the same agent on the same budget with the protections off. It sees the whole file, test included, and scores on any window it likes, with no truncation test, no Ladder and no latency. The protected judge then scores its final pick on the same locked test.

There have been eight pairs of runs so far, each with Claude Sonnet at medium effort and up to 20 sessions per run: six of taker strategies here, and two of making further down.

| data | taker fee | naive loop's claim, on train | naive pick on the locked test | protected loop | judge calls, naive / protected |
|---|---|---|---|---|---|
| 1 day, Binance XRPUSDT spot and perp ([Tardis](docs/demo-data.md), 1 Oct 2026) | 7.5 bps | −3.75 bps/h, Sharpe −0.53 | −4.59 bps/h (90% CI −4.6 to +0.85), 2 fills | 7 features released, no strategy | 43 / 37 |
| same | 1 bps | +1.66, Sharpe +0.61 | −0.19 (−1.1 to +0.31), 4 fills | 2 features, no strategy | 43 / 17 |
| 3 days, Binance BTC, ETH, SOL and XRP perps (27 to 29 Mar 2024) | 7.5 bps | −1.48, Sharpe −0.06 | −2.66 (too few blocks for an interval), 9 fills | 7 features, no strategy; stopped after 19 sessions, so test was not scored | 55 / 49 |
| same | 1 bps | +9.01, Sharpe +0.50 | −3.60 (−11.4 to +0.38), 28 fills | 13 features, no strategy | 60 / 55 |
| 450 days, Binance spot and perps of eight coins, 15-minute bars (1 Apr 2024 to 25 Jun 2025) | 5 bps perp, 10 spot | +0.071, Sharpe +0.11 | +0.014 (too few blocks for an interval), 10 fills | 4 features and 2 strategies; its pick never traded on test, and under the hold rule neither strategy is released | 61 / 60 |
| 450 days, Binance perps of twenty coins, 15-minute bars (same dates) | 5 bps | +1.50, Sharpe +0.03 | +1.09 (too few blocks for an interval), 39 fills | 3 features, no strategy | 63 / 51 |

Net bps/hour is per unit of notional, per leg for a pair. Sharpe is per square-root hour, and test is scored at the brief's latency: 50 ms on the tick files, one 15-minute bar on the 450-day ones. Each interval is the one the run's report prints, the fee verdict's on the hold rule's blocks (studentised); where the pick's own blocks number fewer than the 20 the Ladder needs, the report gives none.

Market making has three files of its own, all from 1 Oct 2026 ([demo data](docs/demo-data.md)): Binance's SOLUSDT and LTCUSDT perpetuals, with every top-of-book row and every trade for the fill engine, and AT spot quoted on MEXC and hedged on Binance, with both venues' books and depth. The SOL file has a scripted pair, with the fake harness and no agent (the touch joined on both sides at the brief's parameters, the same quotes leaned towards the side the touch imbalance says gets hit next, and a trade-flow taker strategy), and a Sonnet pair. The AT file has a Sonnet pair, at the withdrawal fees of the BSC route between the venues. The LTC file has no published run.

| data | fees | runs | naive loop's claim | naive pick on the locked test | protected loop | judge calls, naive / protected |
|---|---|---|---|---|---|---|
| 1 day, Binance SOLUSDT perp, every top-of-book row (Tardis, 1 Oct 2026) | maker 2 bps | scripted | | the leaned quote: −601.1 bps/h (90% CI −678.7 to −538.2) | the touch-imbalance feature released (validation rank IC 0.130), no strategy: on validation the joined quote scored a Sharpe of −2.61, the leaned one −5.24, the taker strategy −0.76 | |
| same | same | Sonnet | +12.97 bps/h on the whole file, test included, Sharpe +0.24, 115 fills | −12.96 bps/h (too few blocks for an interval), 17 fills | 2 features released, no strategy | 61 / 30 |
| 1 day, AT spot quoted on MEXC and hedged on Binance, both books (Tardis, 1 Oct 2026) | maker 0, hedge 10 bps | Sonnet | +10.0 bps a unit on train, Sharpe +1.00, 46 fills | +6.6 bps a unit, 49 fills, +0.29 USD at 10.2 USD a quote (90% CI +0.08 to +0.52 USD) | 2 features and 2 strategies released; the pick made +17.9 bps a unit, 54 fills, +0.49 USD at 5.12 USD a quote (+0.30 to +0.77) | 50 / 43 |

Net bps/hour on the SOL file is per unit of the coin quoted, after the 2 bps maker fee; before it the scripted pick lost 244.9 and the Sonnet pick 10.1. On the AT file a unit is one AT quoted and hedged, net of the hedge fee; the report's verdict line weights by USD and prints +6.5 and +18.3 bps a fill. Every test row is the worse of the two queue models by the hold rule's Sharpe, at 50 ms.

How to read it:

- At 7.5 bps nothing paid, even in-sample, so the naive loop had nothing to inflate: the fee rules out every taker strategy before overfitting can matter.
- At 1 bps on three days, the naive loop found an ETH/BTC pairs trade that made +9 bps/hour on train. It named that trade as its pick, and the pick lost 3.6 bps/hour on test. The protected loop found the same trade, at +6.8 to +11.8 bps/hour on train across five versions. On validation its own pick made −0.05, and the Ladder released nothing.
- The naive claims are at zero latency, because that is how naive mode scores. Re-scored on train at 50 ms, none of them moves by more than 0.3 bps/hour (the ETH/BTC pick makes +9.29), so latency explains almost none of the gap.
- Every taker test loss with an interval has zero inside it, and three picks hold too few of their own blocks for one. The result is the gap between a claim and its test, on one dataset with one pair of runs per fee. The 1 bps fee was chosen after the 7.5 bps runs found nothing, which is why the runs at both fees are listed.
- The protected run at 7.5 bps on three days stopped after 19 sessions and never scored test. The agent had read its own saved tool output with `sed`, and the hook counted that as a sandbox probe. The 1 bps run on the same days took three attempts for the same reason: the first two stopped after 8 and 11 sessions, with no strategy and no test score, and the row is the third. The hook now refuses those reads with a pointer to the Read tool and does not count them.
- Ending flat is the protected loop's right answer here: at these costs none of the candidates works as a taker strategy.
- On the 450-day files both naive picks made money on test, but each holds too few of its own blocks for an interval, so the test can't tell either from no edge. The carry pick, long ETH spot and short the perp while the 3-day mean funding rate is above 0.3 bp, earned +0.014 bps/hour (about 1.3% a year on notional) against a train claim five times that. The lead-lag pick on twenty perps earned +1.09 with a 32% drawdown.
- The protected carry run released a basis trade at 0.10 Sharpe on validation that rested on a single position held for 28 days, and on test it never opened one. That run is why the judge now bootstraps a strategy in blocks as long as its holds and wants 20 of them in validation ([the Ladder](docs/ladder.md)). Re-judged that way, neither of its strategy releases happens. The row shows the run as it was judged at the time.
- On the SOL file the naive claim and its test have opposite signs, +13.0 and −13.0 bps/hour. The report's verdict line says why no quote there pays: the 2 bps maker fee is 4.7 times the half spread of 0.42 bps, and the fills gave back 4.3 times the half spread within 10 s. The protected loop released no strategy, the right answer at that fee.
- On the AT file both loops found the edge the scripted pair had, quoting MEXC wide of Binance's touch and hedging there, and the protected pick made more on test than the naive one, +17.9 against +6.6 bps a unit. Its interval excludes zero, but every fill comes from one flow of 3,582 prints a day, on one day. The scripted quote 30 bps off Binance's touch made +18.8 bps a fill on the same window.

Each report draws the markout fans and the P&L before and after costs, and `blinded replay DIR` rebuilds every number in it from the data. Three of the pairs are online:

| run | protected | naive |
|---|---|---|
| 3 days of perps at 1 bps | [report](https://arvhug.github.io/blinded/reports/taker-1bps-protected.html) | [report](https://arvhug.github.io/blinded/reports/taker-1bps-naive.html) |
| SOL perp, Sonnet | [report](https://arvhug.github.io/blinded/reports/sol-maker-protected.html) | [report](https://arvhug.github.io/blinded/reports/sol-maker-naive.html) |
| AT arbitrage, Sonnet | [report](https://arvhug.github.io/blinded/reports/at-arb-protected.html) | [report](https://arvhug.github.io/blinded/reports/at-arb-naive.html) |

## How the judge works

The judge rejects a feature whose past values change when the data is cut, and scores the rest against forward returns that start after your latency. It recomputes the feature on copies of the data cut at seven points in the scoring window (the 0.30, 0.50, 0.70, 0.85 and 0.95 quantiles, a seeded random point and the middle of the last block), each run in a fresh process, and it never reads the code to decide. A leak that only shows after the last cut, at most half a block at the window's end, can pass.

```python
import blinded as bl
from blinded import features as F
import polars as pl


@bl.feature("perp leads spot; the basis mean-reverts")
def basis(panel):
    b = F.basis("XRPUSDT_PERP", "XRPUSDT")  # a polars expression, nothing computed yet
    return F.ewm_zscore(b, halflife="5m")


@bl.feature("spot against a centred average")
def peek(panel):
    mid = pl.col("XRPUSDT.mid")
    return mid / mid.rolling_mean(50, center=True) - 1


result = basis.evaluate("brief.yaml", window="train")
print(result)  # metrics by horizon, decay, turnover, truncation verdict
peek.evaluate("brief.yaml").status  # 'rejected_lookahead'
```

A feature returns a polars expression over `panel.frame`, a wide frame with a `ts` column and one column per instrument and field (`XRPUSDT.mid`, `XRPUSDT.bid_qty`, `XRPUSDT.cum_signed_qty`, ...). The library in `bl.features` builds the usual ones. pandas works too: return a Series built from `panel.to_pandas()` and the judge treats it the same way.

From the shell it is `blinded eval features.py --brief brief.yaml`, and `blinded data --brief brief.yaml` writes `DATA.md` with event rates, staleness and the lag between streams.

Protected mode is for when something else does the searching. `blinded init brief.yaml --run-dir DIR` splits the data by time into a workspace holding train only and a vault holding validation and test; `blinded daemon DIR` judges whatever is dropped in with `blinded submit`, scores validation only through a release rule that leaks a bounded amount (the Ladder), simulates strategies after latency and fees, writes every attempt to a ledger, and scores test once. The sandbox settings for running an agent against it are written alongside.

## Why

Backtests on high-frequency data fool people in the same few ways: a centred window, a z-score over the whole sample, a forward or nearest as-of join, returns measured from the moment of decision instead of the moment an order could fill. These are hard to spot in code review and easy to produce by accident, and anything that tunes a feature against a score (a person late at night, or an agent) will find them. blinded checks the one property they all break, that the value at time t depends only on data available at t, and it checks it by running the code. An agent proposing features and strategies gets the same treatment: the judge decides, and the agent cannot influence it.

## Install

```console
$ pip install .
```

from a clone, with Python 3.11 or newer. [Getting started](docs/getting-started.md) runs the example above on a free day of Tardis data. The docs are at <https://arvhug.github.io/blinded/>, or build them from `docs/` with `uv run make -C docs html`. Licensed under Apache-2.0.
