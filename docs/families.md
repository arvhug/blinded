# Families

A run can study more than one kind of strategy on the same data. Each kind is a family, with its own instruments, built-in features, judge target, Ladder and report charts:

| family | what it bets on | the judge scores a feature against | a strategy returns |
|---|---|---|---|
| `directional` | an instrument's own next move | that instrument's forward return | one position |
| `lead_lag` | a leader's recent move predicts a follower's later one | the follower's forward return | one position, in the follower |
| `pairs` | a spread between two legs comes back to its mean | the forward return of A less B, equal notional | a position in each leg |
| `basket` | some names of a cross-section beat the rest | each leg's forward return, by the cross-sectional rank IC | a position in every leg |

It stays one run: one vault, one ledger with a `family` column, one test window opened once for one final strategy picked across every family, and one trial count for the deflation that spans every family. Three runs on the same data would open the test three times, and the second and third look would make it another validation set.

## The brief

```yaml
families:
  directional:
    instruments: [BTCUSDT_PERP, ETHUSDT_PERP, SOLUSDT_PERP, XRPUSDT_PERP]
    share: 0.3
  lead_lag:
    leaders: [BTCUSDT_PERP, ETHUSDT_PERP]
    followers: [SOLUSDT_PERP, XRPUSDT_PERP]
    share: 0.3
  pairs:
    pairs: [[ETHUSDT_PERP, BTCUSDT_PERP], [XRPUSDT_PERP, SOLUSDT_PERP]]   # [A, B]: A over B
    share: 0.4
    defaults: {window: 2h, every: 30m, z_in: 2.0, z_out: 0.5}
```

`blinded demo binance` writes this block. A `basket` family takes `instruments: [...]`, at least three, and its legs are all of them in that order; an archetypes brief implies one over every instrument when there are at least 20 ({doc}`archetypes`). Every leader is paired with every follower that is not itself. `share` is the family's share of each phase's sessions; give all or none (equal), and they sum to 1. `defaults` are the pairs parameters the report and DATA.md use when no pairs strategy says otherwise. A brief without the block is one directional family on `target.instrument`, and nothing about families appears in the agent's files. `execution.fees` sets a taker fee per instrument where it differs from `taker_fee_bps`.

## Writing for a family

```python
import blinded as bl
from blinded import features as F


@bl.feature("BTC's last second, not yet in SOL", family="lead_lag", target="SOLUSDT_PERP")
def btc_sol_gap(panel):
    return F.lead_gap("BTCUSDT_PERP", "SOLUSDT_PERP", "1s")


@bl.strategy(requires=["btc_sol_gap"], family="lead_lag", trades="SOLUSDT_PERP")
def sol_follows_btc(panel, features):
    return bl.threshold(features["btc_sol_gap"], z_in=3.0, z_out=0.5)


@bl.strategy(family="pairs", pair=("ETHUSDT_PERP", "BTCUSDT_PERP"))
def eth_btc_pairs(panel, features):
    return bl.pairs("ETHUSDT_PERP", "BTCUSDT_PERP", z_in=2.0, z_out=0.5, window="2h", every="30m")
```

`family` defaults to `directional`, `target` and `trades` to the brief's target instrument, and `pair` is required for a pairs item and refused for any other. The {doc}`archetype <archetypes>` decorators are sugar over these: `@bl.pairs_trading(a="ETHUSDT_PERP", b="BTCUSDT_PERP")` is the pairs strategy above with a tag in the ledger, and in a families brief it judges the same to the last digit. The judge checks all three against the brief before running anything, and a mismatch is an `error` row whose reason is built from the decorator and the brief, never from what the code printed. A pair written the other way round, `(B, A)`, is judged as `(A, B)` with the feature's sign flipped, so a high value always means the spread of A over B is expected to rise.

The built-ins are in the {doc}`feature library <features>`: `F.lead_ret`, `F.lead_gap`, `F.lead_gap_z` and `F.lagged_corr` for lead-lag; `F.hedge_ratio`, `F.pair_spread`, `F.spread_z` and `F.half_life` for pairs, with `bl.pairs` the built-in pairs strategy ({doc}`sim`). All of them are run through the truncation test with the rest of the library.

## What the judge does with them

A directional or lead-lag feature is scored against its instrument's forward return from $g + L$ to $g + L + h$. A pairs feature is scored against the spread return at equal notional,

$$
r^s_h(g) = \ln\frac{m_A(g + L + h)}{m_A(g + L)} - \ln\frac{m_B(g + L + h)}{m_B(g + L)},
$$

which has no hedge ratio in it: a target with a fitted parameter would make the score depend on a knob, and the ratio belongs to the strategy. For a pair whose hedge ratio is far from 1 this understates the IC of a hedged spread, and the report says so beside its table. A pairs strategy is simulated on both legs ({func}`blinded.sim.simulate_legs`), each on its own book and at its own fee, on one unit of notional per leg.

A basket feature returns one value per leg, the mapping `F.each(panel.legs, ...)` builds and the `F.xs_` functions combine ({doc}`features`); one that leaves out a leg, or adds one, is an `error` row with the rule "a basket feature returns a mapping with one value per leg, {legs}, only". It is scored by its cross-sectional rank IC: at each grid point $t$, over the legs $i$ where both are finite,

$$
c_t = \operatorname{corr}\big(\operatorname{rank}_i x_{i,t},\ \operatorname{rank}_i r_{i,t}\big),
$$

with $r_{i,t}$ leg $i$'s forward return from $g + L$, NaN where fewer than three legs are finite ({func}`blinded.metrics.xs_rank_ic`). The score is the mean of $c_t$ over the window, under the same keys as a single-leg feature's ({func}`blinded.evaluate.score_xs`): `ic` and `rank_ic` both hold it, its Newey-West error uses the series of $c_t$ with the horizon's lags and its bootstrap resamples blocks of it, `n` counts the finite $c_t$, and `markout_bps` is that of a basket long the top fifth of the legs and short the bottom fifth, one notional a leg: the mean over $t$ of half the top's mean return less the bottom's. The Ladder's score is the same mean, with replicates the means of $c_t$ over the drawn blocks ({func}`blinded.ladder.xs_rank_ic_score`). Three legs with features (1, 2, 3) at two points and returns (0.01, 0.02, 0.03) then (0.03, 0.01, 0.02) have $c = 1$ and $1 - 6 \cdot 6 / (3 \cdot 8) = -0.5$, a score of 0.25, and a markout of $(100 - 50) / 2 = 25$ bps; `tests/test_basket.py` checks it and the replicates against a pandas groupby on twenty seeded legs. A basket strategy returns a position for every leg and is simulated on all of them ({doc}`sim`).

Each family has its own Ladder for each kind, so a family can release what the others never would, and the validation score is the same metric in every family (rank IC for features, Sharpe per square-root hour for strategies). The final pick is the agent's named trial if it was released, whatever its family, else the released strategy with the highest released best across families. Sessions are shared out by `share`, and a family's phase closes after three of its sessions in a row without a release ({doc}`driver`).

## Lead-lag: the cross-correlation

The report and DATA.md draw, for every leader and follower,

$$
C(\tau) = \operatorname{corr}\big(r_\ell(g),\ r_f(g + \tau)\big), \qquad r_X(t) = \ln m_X(t) - \ln m_X(t - \Delta),
$$

over the grid points of a window ({func}`blinded.diagnostics.lead_lag_curve`). $r_\ell(g)$ is the leader's last step as the decision at $g$ saw it, which is `F.lead_ret(leader, 1)`; $r_f(g + \tau)$ is the follower's step that ends $\tau$ later, its mid read as-of each time from the events, so a lag need not be a whole step. Both returns are trailing and of the same length, so a follower that copies the leader $c$ later peaks at $\tau = c$, a peak left of zero means the follower moved first, and two prices that move together give a curve symmetric about zero.

A taker who decides at $g$ fills $L + d$ later, the latency plus the follower's feed delay. A follower step that ends before that is over before the order lands; the chart shades $[0, L + d)$. The follower's first whole step after the fill ends at $\tau = L + d + \Delta$, and that point is the judge's own object: the correlation of the leader's last step with the follower's forward return over one step from the fill. The table gives, per leader and follower, the lag at the peak, $C$ at the peak, $C$ on that first step after the fill, the best $C$ from there on, and `usable_by_taker`, which says yes when that best is positive and at least half the peak. It is a rule of thumb, not a test.

The lags are whole steps from $-10\Delta$ to $+10\Delta$ and the first step after the fill. On a file that holds the raw events there are also 10 ms steps across one step either side of zero, and the mirror of the fill point as a control. A thinned file keeps only the stamps the judge reads ({doc}`demo-data`), which serve whole steps and the fill point, so the fine steps are left out; lead-lag inside a step is then measured once by the builder on the raw books and written to `DATA-raw.md`.

The known answers, in `tests/test_leadlag.py`, on a 100 ms grid with a 50 ms latency:

- A follower that is the leader two steps late peaks at 200 ms with $C = 1$, and $C(-200\,\text{ms})$ is the correlation of the leader's eight returns with themselves four steps back, $2/\sqrt 5$ for the sequence the test uses. The first step after the fill is $C(150\,\text{ms})$, the lag-one correlation, and the best from there on is the peak: usable.
- A follower that copies a single jump 30 ms later, 30 ms before a grid point, has its jump in the same step as the leader's: $C(0) = 1$, every other whole step is $-1/(n - 1)$ for two single spikes in $n$ points, and everything from 150 ms on is negative: not usable. On the raw events the fine lags show a plateau of $C = 1$ from $-30$ to $60$ ms, since one jump a step cannot place a lag finer than that, and the tie goes to the lag nearest zero.

On the three Binance days the 500 ms curves on train peak at 0 for every leader and follower ($C$ 0.42 to 0.55) and keep 0.07 to 0.11 on the first step after the fill. The raw books, at 10 ms steps, put the peak at 0 to 10 ms ($C$ 0.26 to 0.35) with 0.06 to 0.08 left on the first 100 ms step after a 100 ms fill. Any lead here is shorter than a taker's round trip.

Writing the follower's return as starting at $g + \tau$ rather than ending there would put the peak of a $c$-late copy at $c - \Delta$ and the peak of two prices moving together at $-\Delta$, against the known answers (a peak at 200 ms for a two-step copy, a curve symmetric about zero for no lead). The trailing form keeps those answers and puts the fill point at $L + d + \Delta$, where the "$+L$" step ends.

## Pairs: what is drawn

For the pair of the final strategy, else of the last released pairs strategy, else of the pairs strategy with the best validation Sharpe, else for every configured pair at the brief's defaults, the report draws four charts:

1. **The spread and its bands.** $s_t - \tilde\mu_t$ in bps, the spread less its walk-forward mean, with $\pm z_\text{in}\tilde\sigma$ dashed and $\pm z_\text{out}\tilde\sigma$ dotted, and under it the position held: the share of each plotted interval spent long the spread less the share spent short. The level of $s_t$ itself jumps by $\Delta\tilde\beta \ln m_B$ at every refit, about 1,100 bps for a 0.01 change on BTC, so the mean is taken off.
2. **The hedge ratio** $\tilde\beta$, one step per clock window of `every`.
3. **The half-life** of an AR(1) fit inside each clock window, on a log axis; an open circle where $\hat\varphi \ge 1$ and the spread did not come back. Read it against the fan. An AR(1) slope with an intercept, fitted on $n$ points of a series that does not revert at all, comes out about $5.4/n$ below 1 on average (the Dickey-Fuller bias): in 400 simulated random walks of 3,600 half-second steps, one 30-minute window each, the median half-life is 290 s and only 4% of windows show no reversion. The chart draws that line, the median half-life of a random walk in a window as long ({func}`blinded.markouts.random_walk_half_life`, seeded so replay can check it), and only points well under it are evidence of reversion. Tick noise in the two mids adds reversion within a step on top. On the Binance validation days ETH over BTC shows a median half-life of about two minutes while its fan shows no move back within a minute.
4. **The markout fan of the spread**: each time $|z|$ went past the strategy's entry band $z_\text{in}$, a trade back towards the mean (short the spread when $z$ is high), events at least 70 s apart, and the mean of $\pm\{[\ln m_A(g + \tau) - \ln m_A(g)] - \tilde\beta_g[\ln m_B(g + \tau) - \ln m_B(g)]\}$ in bps with two standard errors either side, the hedge held at the decision's ratio. A spread that reverts rises after zero. Breaking even from the fill takes $2(\text{half spread}_A + |\tilde\beta|\,\text{half spread}_B)$, at the events, and the fees $2(\text{fee}_A + |\tilde\beta|\,\text{fee}_B)$ on top.

The fit's `window` and `every` and the bands' `z_in` and `z_out` are read from the strategy's own `bl.pairs` call when its function makes exactly one, with literal arguments, on the pair's two instruments; otherwise they are the brief's defaults, and the page says which. `tests/test_report.py` recomputes the fan at 10 s with a pandas groupby and checks that the top bucket falls and the bottom one rises on the synthetic perp over spot.

DATA.md gets a row per pair on train at the defaults: the hedge ratio and residual sd in bps in force at the end of train, the median half-life over clock windows and the share of windows with no reversion. On the Binance train window, ETH over BTC has a ratio of 0.60, a residual sd of 7.0 bps and a median half-life of 116 s; XRP over SOL 0.59, 13 bps and 148 s, with 3 of 71 windows not reverting.

## In the report

Section 2 has a table per family: features and strategies submitted, accepted and released, the released best of each kind, sessions in each phase, and which family holds the final strategy. Section 4 shows the directional family's feature and strategy, then for lead-lag a chart and table per leader and follower and the family's own feature and strategy by the same rule, then the pairs charts. Every number behind them is in `report.json` under `recompute.diagnostics.lead_lag` and `recompute.diagnostics.pairs`, so `blinded replay` checks them. The naive comparison counts each family's judge calls in both runs. A run with {doc}`archetypes` is drawn by edge and archetype instead, with a death-check panel for each, and its naive comparison counts calls per edge and per archetype ({doc}`report`).
