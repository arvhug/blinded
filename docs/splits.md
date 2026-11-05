# Splits and the embargo

Protected mode cuts the data by time into three segments. Train is the agent's to read and score as often as it likes. Validation is scored only through the {doc}`Ladder <ladder>`, and test is scored once, at the end.

## Where the cuts go

With `fractions` $(f_1, f_2, f_3)$ and the grid running from $g_0$ to $g_{\text{last}}$, the span is $T = g_{\text{last}} - g_0$ and the two split points are

$$
s_1 = \operatorname{round}(g_0 + f_1 T), \qquad s_2 = \operatorname{round}(g_0 + (f_1 + f_2) T),
$$

rounded to the nearest whole minute, or to the grid step if that is longer. Then

| segment | window |
|---|---|
| train | $[g_0,\ s_1)$ |
| embargo | $[s_1,\ s_1 + e)$ |
| validation | $[s_1 + e,\ s_2)$ |
| embargo | $[s_2,\ s_2 + e)$ |
| test | $[s_2 + e,\ g_{\text{last}} + \Delta)$ |

where $e$ is the embargo and $\Delta$ the grid step. Every window is half-open, and test ends one step after the last grid point so that point is inside.

The rounding is there so the boundaries do not depend on when the first and last events happened to arrive. On the example file the grid runs from 02:00:00.1 to 07:59:59.8, so $T$ = 5:59:59.7. Half of that past 02:00:00.1 is 04:59:59.95 and three quarters is 06:29:59.875; to the minute they are 05:00 and 06:30. With a 5 minute embargo:

| segment | window (UTC) |
|---|---|
| train | 02:00 to 05:00 |
| validation | 05:05 to 06:30 |
| test | 06:35 to 07:59:59.9, which holds the same grid points as 06:35 to 08:00 |

Rounding moves each fraction by at most 30 seconds in several hours. The grid warm-up (`grid.warmup`) is dropped from the start of train when scoring, as in local mode.

## Why there is an embargo

The target at grid point $g$ is the return from $g + L$ to $g + L + h$. At the last train point, just before $s_1$, the longest-horizon target ends at about $s_1 + L + h_{\max}$. If validation started at $s_1$, its first targets and the last train targets would share price moves: a feature fitted to the end of train would be partly fitted to the start of validation. The gap must therefore exceed $L + h_{\max}$, and the brief is rejected if it does not.

The default is

$$
e = \max(5\, h_{\max},\ 5\ \text{min}).
$$

The factor of five is margin. Dependence between returns does not stop exactly at the horizon (volatility clusters, and a slow feature carries information about the level it was computed from), and the 5 minute floor keeps the gap at least one bootstrap block long, so no block straddles the cut. For horizons up to 60 s the default is 5 minutes, and for a 2 minute horizon it is 10.

The embargo is not there for feature look-back. Looking back from the start of validation into train is looking at the past, and that is allowed.

## One warm panel for validation and for test

A feature with a 5 minute rolling window computed on validation alone would be null for its first 5 minutes, and an exponential average would start cold. So validation is scored on one contiguous panel, train plus the embargo plus validation, and only grid points inside validation count. Test is scored the same way on everything up to the end of test.

This does not leak validation into train, nor test into validation, for features that passed the {doc}`truncation test <concepts>`: a causal feature's value at $g$ depends only on events available by $g$, so the train part of the warm panel is the same whether validation is attached or not. The panel is built from the vault's own copies of the segments (see {doc}`run-directory`), never from the workspace.

## The execution block

The splits are the only part of the brief that changes in protected mode. The simulator reads the `execution` block:

```yaml
execution:
  latency_ms: 50
  taker_fee_bps: 7.5
  maker_fee_bps: 0.0
  max_position: 1.0     # positions are in units of notional, within +-max_position
  min_rebalance: 0.1    # position changes smaller than this are not traded
```

`latency_ms` defaults to 0, the fees to 0, `max_position` to 1 and `min_rebalance` to 0. `fee_tier: <venue>/<product>/<tier>` (for example `deribit/inverse-perpetual/standard`) takes the maker and taker fees from the package's fee table ({mod}`blinded.fees`, `tables/fee_tiers.yaml`, one row per venue tier with its source page and the date it was read), so a run can search at a member or market-maker rate; the parser refuses a tier the table does not have, and a brief that names a tier and also gives a fee the tier sets. A market maker programme publishes only its maker fee, so with one of those the brief gives `taker_fee_bps` too. Borrow on a short spot position is not set here: it is a `borrow` stream in the data, the rate as published at each change ({doc}`sim`). A file without one lets a short go free, so a file that trades spot should have one.
