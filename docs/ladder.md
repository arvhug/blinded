# The Ladder

The agent may score itself on train as often as it likes, because the claim is made on validation. The danger there is adaptivity: an agent that sees its exact validation score after every submission can keep the changes that went up and drop the ones that went down, and after a few dozen rounds the best score reflects the noise in validation as much as the signal. The Ladder limits what each submission reveals so that this hill-climbing has little to climb on. It is due to Blum and Hardt, *The Ladder: A Reliable Leaderboard for Machine Learning Competitions* (2015, arXiv:1502.04585).

## The rule

Each phase, features and then strategies, keeps a released best $b$, which starts empty; in a run with families each family keeps its own for each kind, so one family's good strategy does not stop another's from being released, and in a run with {doc}`archetypes` each edge keeps its own, shared by the archetypes of the edge. For a submission with validation score $v$ (rank IC at the primary horizon for a feature, net Sharpe per $\sqrt{\text{hour}}$ for a strategy):

1. $\eta_{\text{boot}}$ is the block-bootstrap standard deviation of $v - v_{\text{best}}$, where $v_{\text{best}}$ is the score of the submission that set $b$. Before anything is released the comparison is with a no-skill baseline whose score is 0 in every replicate.
2. $\eta = \operatorname{step}(\max(\eta_{\text{boot}}, \eta_{\min}))$, with $\eta_{\min}$ = 0.005 for rank IC and $0.05\sqrt{5\ \text{min} / \text{block}}$ for Sharpe (below), and step() raising its argument to the next value of the form 1, 2 or 5 times a power of ten.
3. If $v - b > \eta$ (with $b = 0$ before the first release): release, $b \leftarrow \eta \cdot \operatorname{round}(v / \eta)$, and remember this submission's bootstrap replicates.
4. Return `{"released": ..., "best": b}`.

The agent never sees $v$, $\eta$ or the replicates. They go into the ledger in the vault.

### Paired replicates

Every submission in a phase is resampled with the same circular block draws: the same block starts, blocks of the brief's `stats.block`, `n_boot` of them, from the brief's seed. A strategy whose holds outlast that block uses its own (below). Replicate $k$ of one submission and replicate $k$ of another therefore use the same grid points, and $v^{*}_k - v^{*}_{\text{best},k}$ is a replicate of the difference. When two submissions are near copies, most of the sampling noise is shared and cancels, so the step is small and a real improvement can be released; the test suite checks that pairing cuts the step by more than ten times for near copies. This is the paired $t$-test of Blum and Hardt's parameter-free variant, with the block bootstrap in place of $s/\sqrt{n}$ because neighbouring grid points are not independent.

For rank IC, the ranks are computed once over the window and a replicate is the correlation of the ranks over the drawn blocks, dropping points where the feature is missing. For Sharpe, a replicate is the mean over the standard deviation of the drawn blocks' P&L, times the square root of blocks per hour ($\sqrt{12}$ for 5-minute blocks).

### A worked sequence

Floor 0.005, four submissions, each with four replicates (in the test suite as `test_a_feature_phase_by_hand`):

| | $v$ | replicates of $v - v_{\text{best}}$ | $\eta_{\text{boot}}$ | $\eta$ | $v - b$ | released | $b$ |
|---|---|---|---|---|---|---|---|
| 1 | 0.0123 | $0.0123 \pm 0.002$ (against 0) | 0.002 | 0.005 | 0.0123 | yes | $0.005 \cdot \operatorname{round}(2.46) = 0.010$ |
| 2 | 0.0140 | 0.0017 in every replicate | 0 | 0.005 | 0.004 | no | 0.010 |
| 3 | 0.0420 | $0.0297 + (-0.013, 0.009, -0.013, 0.009)$ | 0.011 | 0.02 | 0.032 | yes | $0.02 \cdot \operatorname{round}(2.1) = 0.04$ |
| 4 | 0.0300 | | | | $-0.010$ | no | 0.04 |

The agent sees `true, 0.01`, `false, 0.01`, `true, 0.04`, `false, 0.04`. Submission 2 was better than submission 1, and the agent cannot tell.

### The Sharpe floor and the block

The Sharpe floor is $0.05\sqrt{5\ \text{min} / \text{block}}$ per square-root hour, with the brief's `stats.block` ({func}`blinded.ladder.floor`); the daemon builds every strategy Ladder with it. With 5-minute blocks, which the tick files use, it is exactly 0.05. A fixed 0.05 per square-root hour is an annualised Sharpe of $0.05\sqrt{8760} = 4.7$, which no slow strategy could clear however long its validation. The bootstrap error of a Sharpe near zero is about $1/\sqrt{\text{validation hours}}$ whatever the block, so a floor that shrinks with longer blocks lets that error set the step where it should: on hourly bars with 2-week blocks the floor is 0.0008 against an error near 0.012. The rank IC floor is unitless and stays at 0.005.

What the floor and the error mean for a first strategy release on each file is printed under the archetype menu ({func}`blinded.archetypes.power`):

| validation | $1/\sqrt{\text{hours}}$ | block | floor | step: under rules 1 and 2 a first release needs a Sharpe per $\sqrt{\text{h}}$ above | under rules 3, also above twice its error |
|---|---|---|---|---|---|
| 3 h 25 min (the Tardis day) | 0.541 | 5 min | 0.05 | 1.0 (93.6 annualised) | 1.08 (101) |
| 17 h 55 min (three Binance days) | 0.236 | 5 min | 0.05 | 0.5 (46.8) | 0.5 (46.8), the step |
| 3 h 25 min (the XRP perpetual on Binance and Bybit) | 0.541 | 5 min | 0.05 | 1.0 (93.6) | 1.08 (101) |
| 110 d 11 h 45 min (twenty perpetuals on 15-minute bars over 450 days) | 0.0194 | 4 h | 0.0072 | 0.02 or 0.05 (1.9 or 4.7), as its own bootstrap falls either side of 0.02 | 0.039 or 0.05 (3.6 or 4.7) |
| 287 d 5 h (four perpetuals on hourly bars over 1,233 days) | 0.0120 | 2 w | 0.0008 | 0.02 (1.9) | 0.024 (2.2) |

The last column takes the error as $1/\sqrt{\text{hours}}$; the Ladder uses the submission's own bootstrap standard deviation. The menu's power line prints the step alone.

On the hourly bars the floor does not bind and the error sets the step, as intended: a scripted run there (`scripts/trend_fake.py`) built three strategy Ladders, one per edge, each with a floor of 0.000787, and its best strategy on validation, a BTC trend at 0.0041 Sharpe per square-root hour, stayed under the 0.02 step. A feature step is set the same way by the rank IC's own bootstrap: the four-week BTC return had a validation rank IC of 0.153 at one week and was not released, so its first step was at least 0.153, which the 1-2-5 grid makes 0.2 or more: the expected 0.1 to 0.2 for about 42 independent weeks per instrument.

A basket feature's validation score is its mean cross-sectional rank IC ({doc}`families`), and its replicates are the means of the per-point ICs over the drawn blocks, paired like a single-leg feature's ({func}`blinded.ladder.xs_rank_ic_score`). Within the relative value edge a pairs feature's rank IC and a basket feature's cross-sectional rank IC share one Ladder: both are rank correlations in $[-1, 1]$ of a feature against what its archetype trades. On the 15-minute bars a scripted run (`scripts/xs_fake.py`) released a one-day residual z of statistical arbitrage at 0.01 and a one-day reversal at 0.03, each a mean over 10,600 grid points of which about 660 are independent at 4 hours.

### A strategy's own holds

A Sharpe ratio's error shrinks with the number of independent bets, and the block bootstrap counts one bet per block. That works while a strategy's positions are shorter than a block. A strategy that opens one position and keeps it for four weeks has made one bet, and cutting validation into 8-hour blocks counts it 84 times. The first Sonnet run on the carry file released such a strategy: one ETH basis position, held for 27.9 of the 110 days of validation, at 0.0999 Sharpe per square-root hour on 8-hour blocks. On the locked test it never opened a position.

So a strategy is bootstrapped in its own blocks:

1. Its hold $\bar L$ is the time-weighted mean length of the runs of one sign of each leg's position on validation, $\sum L^2 / \sum L$ over every run of every leg ({func}`blinded.sim.hold_steps`). It is the expected length of the hold that a random step in position belongs to, so a long hold counts for as much as the P&L it carries: ten 1-hour holds and one 28-day hold give 27.6 days, where the median would give 1 hour.
2. Its block is the brief's block or $\lceil \bar L \rceil$ grid steps, whichever is longer. The point Sharpe, the replicates and the blocks per hour all use it. The window holds $\lfloor n / \text{block} \rfloor$ blocks from its start, and the steps left over, fewer than a block, are folded into the last one ({func}`blinded.ladder.fold`): they hold the end of the window, a market maker's flatten among them, so dropping them would score a strategy without its last loss. A replicate draws as many blocks, the last as long as the window's last, so it too has every step's worth of P&L. Blocks per hour are the block's, not the longer last one's. This is rules 2; a run judged under rules 1, every run before 20 Oct 2026, dropped the remainder, and is still scored and replayed that way ({func}`blinded.ladder.sharpe_blocks`, and the judging rules in {doc}`report`).
3. When its own block is the longer and validation holds fewer than `PERIODS` (20) of them, the Ladder is given no score and releases nothing. This is the span rule of {doc}`archetypes` applied to the strategy's own holds instead of an archetype's typical one.

A strategy that trades within the brief's block keeps that block, and its score and replicates are unchanged. One with longer holds is not paired with the released best's draws, since its blocks differ, and the difference of two unpaired replicates carries the variance of both, so its step can only grow. The agent learns the rule from RULES.md, and on an archetypes run ARCHETYPES.md states the longest mean hold that can release on that file: 5 d 12 h 30 min on the 450-day files, 14 d 8 h on the hourly one. It never learns any strategy's hold on validation. The ledger keeps `hold_steps`, `hold_hours`, `block_steps`, `own_blocks` and `releasable` with each strategy's validation score, and the report names every strategy held back this way.

Re-judged under the rule in a fresh run of the same brief, the carry strategy's own block is 2,680 bars, validation holds 3 of them, and its Sharpe at that block is 0.0195. The run's other release, a fade of ETH trade flow, held 9.9 days on average, so validation holds 11 of its blocks. Neither is released.

The rule is conservative for a constant exposure to returns that are independent from step to step: that P&L is independent whatever the hold, and the rule still counts the hold once. A judge meant not to fool itself should err that way. The deflated Sharpe ({doc}`deflation`) keeps the brief's blocks, the grid every trial shares. A final strategy is always a released one, but a release only needs 20 of its own blocks in validation, so its holds can be as long as a twentieth of validation and span many of the brief's blocks. The deflation counts the effective number of blocks for that ({doc}`deflation`).

### A market maker's score: the worse of two queue models

A market-making strategy's fills depend on where its orders sit in the queue, which the data never shows ({doc}`sim`). The judge runs four fill models, each its own closed loop, and scores the worse of the two queue models, RA and PR:

$$
v = \min\big(\text{Sharpe}_{\text{RA}},\ \text{Sharpe}_{\text{PR}}\big),
$$

each the net Sharpe per square-root hour on validation under the hold rule above: blocks as long as that model's own time-weighted mean hold, or the brief's, whichever is longer ({func}`blinded.sim_maker.held`). The two models share every assumption but the one each is optimistic about: RA counts too few fills and none of the informed cancels ahead of the order, PR counts too many when the cancels are uninformed. A gain that one shows and the other does not is not released. The replicates are those of the model that attained the minimum, under the phase's shared draws when its blocks are the brief's, so the paired comparison with the released best works as for any strategy. T and F bracket the fill count and never score. A model whose Sharpe measures nothing (fewer than two of its blocks, or a P&L that never moved) is the worse; and when the scored model's holds give validation fewer than twenty of its blocks the Ladder is given no score, as for any strategy.

The minimum is taken in the score's own blocks, not the brief's. The two can disagree: on the hand example at a 2 bps maker fee, RA holds one step and PR two (it is long from 2.1 s), so on the brief's 1 s blocks PR's Sharpe is the lower, -38.14 against -38.08, while on its own 2 s blocks, -10 and -110 with the fifth step's flatten folded into the second, PR's is -36.0 and RA's -38.08 is the lower. The judge scores RA there, and so do the sim table's `brief` row, the report's bracket and the locked-test verdict, which all take the model by {func}`blinded.sim_maker.worse`; the report prints both columns and marks the model the rule took (`tests/test_maker.py`).

The ledger keeps all four models' validation numbers with each one's Sharpe, hold and block under the rule (`models`, with `held`) and which model scored (`fill_model`); the agent is told `released` and `best`, never which model scored or what any model made on validation.

A score that measures nothing is never a number anywhere: the ledger keeps `value` as null with a `score_note` saying why, the Ladder is not asked, the agent hears `released: false` and the best as it would of a loss, and a locked-test summary keeps `score` as null with the same note. On the AT arbitrage file, quoting 64 AT at skew 0 on MEXC's 124 bps book, the risk-averse pass of the test window fills 15 times, 740 units, and after its first fill at step 17,782 is never flat again: one inventory run of 195,219 of the window's 213,000 steps, so its own block is that long and the window holds one. Its Sharpe would be NaN, so it is null, with the note "its holds give 1 block of 195219 steps on the window, and a Sharpe needs two".

## Why it leaks only $O(\log N)$ bits

Think of the agent as a deterministic function of everything it has been told. Its $t$-th submission is then fixed by the transcript of Ladder outputs before it, so the set of submissions it could possibly make in $N$ rounds is no larger than the set of possible transcripts. Count those.

**Exact scores.** If each round returned $v$ to four decimals, a rank IC in $[-1, 1]$ has $20{,}001$ possible values and the transcript of $N$ rounds can be any of $20{,}001^N$ sequences: $14.3\,N$ bits, linear in $N$. Every round tells the agent something new about validation.

**The Ladder with a fixed step $\eta$** (Blum and Hardt's Algorithm 1). A transcript is fixed once you know in which rounds a release happened and what was released; in every other round the output repeats the previous one. Each release raises $b$ by at least $\eta$, so with scores in a range of width 1 there are at most $\lceil 1/\eta \rceil$ releases, and each released value is one of $\lceil 1/\eta \rceil$ multiples of $\eta$. Writing down the round of each release takes $\log_2(2N)$ bits and its value $\log_2(2/\eta)$, so (their Claim 3.2) a transcript needs at most

$$
B = \left(\frac{1}{\eta} + 2\right) \log\frac{4N}{\eta}
$$

bits. With $\eta$ fixed, that grows like $\log N$, not like $N$.

**From bits to overfitting.** Let $F$ be every submission the agent could have made, at most $2^B$ of them. For one fixed function, Hoeffding's inequality bounds the chance that its validation score is more than $\varepsilon$ from its true score by $2 e^{-2\varepsilon^2 n}$ for scores in $[0, 1]$. A union bound over $F$ gives

$$
\Pr\left(\exists f \in F: |v(f) - \mu(f)| > \varepsilon\right) \le 2^{B+1} e^{-2\varepsilon^2 n},
$$

which is their Theorem 3.1: with $\eta$ of order $n^{-1/3} \log^{1/3}(Nn)$ the released best is within $O\!\left(\log^{1/3}(Nn) / n^{1/3}\right)$ of the truth with high probability, however adaptively the submissions were chosen. Releasing exact scores gives the agent $N$ chances to pick up noise, and the error of the best one grows like $\sqrt{N}$ in the worst case.

### What changes in blinded's version, and what that costs the argument

The parameter-free variant in Blum and Hardt's Figure 2 uses a threshold $s/\sqrt{n}$ that adapts to each submission and rounds the release to $1/n$. They say themselves that the variant's significance test is heuristic: with a step that can shrink without limit, the number of releases is no longer bounded by $1/\eta$. blinded keeps the adaptive, paired step and adds two things so that the count above goes through:

- **A floor.** Each release raises $b$ by more than $\eta/2 \ge \eta_{\min}/2$: a release needs $v > b + \eta$, and rounding moves $v$ by at most $\eta/2$. So with scores above 0 and below 1 there are at most $2/\eta_{\min}$ = 400 releases of rank IC in a phase, however small the bootstrap error gets.
- **A coarse step.** If the release were rounded to the raw bootstrap error, the digits of every released best would carry that error, a continuous number computed from validation, and the set of possible transcripts would be infinite. Raising $\eta$ to the next 1, 2 or 5 times a power of ten keeps the released values on a finite grid. For rank IC every possible step, 0.005, 0.01, 0.02, 0.05 and so on, is a multiple of 0.005, so a released best is one of the 200 multiples of 0.005 in $(0, 1]$, and the step itself takes one of a handful of values. Raising, never lowering, means the step is never smaller than the bootstrap error.

With $r$ releases in $N$ submissions, the transcript then takes at most $r \left(\log_2 N + \log_2 200\right)$ bits plus $\log_2 N$ for its length. Forty submissions with five releases is at most $5 \times (5.3 + 7.6) + 5.3 \approx 70$ bits, against $40 \times 14.3 = 572$ for exact scores to four decimals.

On 85 minutes of validation at a 10 s horizon the effective sample is about 510 points and the 5 minute blocks number 17, so the Hoeffding step above, with $n$ replaced by an effective sample of that size, gives a loose guarantee. The Ladder stops the agent from extracting more than this from validation; it cannot make 85 minutes of data say more than it does. That is why the test segment is still scored once, at the end, and why the deflated Sharpe ratio counts every trial.

## The first submission

Blum and Hardt start the best loss at $+\infty$, so their first submission is always released. Here the first release has to beat a score of 0 by a step, against a baseline that is 0 in every replicate. How often a null submission is released first, measured with the real Ladder (2,000 replications a cell for strategies, 600 for features; `scripts/mc_ladder_null.py`): a strategy with independent N(0, 1) step P&L on 5-minute blocks, 1.6 to 2.4% under rules 3 at 20 to 200 validation blocks, against 2.3 to 11.1% under rules 1 and 2, where the step alone, one standard deviation raised to the 1-2-5 grid, decided and the rate swung with where the grid fell (10.5% at 20 blocks, 2.9% at 41, 11.1% at 70, 2.3% at 200). A persistent feature against an overlapping forward return, 8 to 20 blocks: 3.8 to 4.2% under rules 3, 10.7 to 13.8% before; above the normal 2.3% because the bootstrap understates the error of a slow signal on few blocks. `best` stays `null` until something is released. Orient a feature so that higher values predict higher returns: a strongly negative rank IC is not released.
