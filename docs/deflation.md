# Deflation: DSR and PBO

The Ladder controls what the agent learns from validation while it searches. It does not change the fact that the final strategy is the best of many. Pick the best of $N$ skill-free strategies and its Sharpe ratio is positive in expectation, and larger the more you tried. The deflated Sharpe ratio asks whether the chosen strategy beats that expected maximum. The probability of backtest overfitting asks a different question of the same ledger: how often the strategy that looks best on one half of the data is below the median on the other half. Both are computed on validation at the end of a run, from the per-block P&L of every strategy trial in the ledger.

## The deflated Sharpe ratio

Bailey and López de Prado, *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*, Journal of Portfolio Management, 2014.

### The threshold: the expected maximum of $N$ trials

Suppose the trials' estimated Sharpe ratios $\widehat{SR}_n$ are normal with mean $E[\widehat{SR}_n]$ and variance $V[\widehat{SR}_n]$. The expected maximum of $N$ of them is approximately (their equation 1, from extreme value theory)

$$
E\left[\max_n \widehat{SR}_n\right] \approx E[\widehat{SR}_n] + \sqrt{V[\widehat{SR}_n]}\left((1 - \gamma)\, Z^{-1}\!\left[1 - \frac{1}{N}\right] + \gamma\, Z^{-1}\!\left[1 - \frac{1}{N e}\right]\right),
$$

with $\gamma \approx 0.5772$ the Euler-Mascheroni constant and $Z^{-1}$ the standard normal quantile function. The bracket is roughly where the largest of $N$ standard normals lands: $Z^{-1}[1 - 1/N]$ is the level exceeded by one draw in $N$, and the mixture with $Z^{-1}[1 - 1/(Ne)]$ corrects for the extreme value distribution's skew. Under the null of no skill, $E[\widehat{SR}_n] = 0$, and this is the threshold $\widehat{SR}_0$. It grows with $N$, and with the spread of the trials' Sharpe ratios, which is how the ledger's honesty about failures feeds into the result. The test suite checks the approximation against a Monte Carlo maximum for $N$ = 10, 100 and 1,000, as the paper's appendix 2 does.

### The probability that the true Sharpe beats it

The probabilistic Sharpe ratio of Bailey and López de Prado (2012) is the probability that the true Sharpe ratio exceeds a threshold $SR^*$, given $T$ observations with skewness $\hat\gamma_3$ and kurtosis $\hat\gamma_4$:

$$
\widehat{PSR}(SR^*) = Z\left[\frac{(\widehat{SR} - SR^*)\sqrt{T - 1}}{\sqrt{1 - \hat\gamma_3\,\widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^2}}\right].
$$

The denominator is the standard error of $\widehat{SR}$ under non-normal returns: negative skew and fat tails widen it, so the same Sharpe ratio is less convincing. $\hat\gamma_4$ is the kurtosis itself, 3 for normal returns, not the excess. The deflated Sharpe ratio is the PSR at the deflated threshold (their equation 2):

$$
\widehat{DSR} = \widehat{PSR}(\widehat{SR}_0).
$$

### The paper's example, step by step

A strategist reports an annualised Sharpe ratio of 2.5 over five years of daily returns, after $N = 100$ trials whose annualised Sharpe ratios have variance $1/2$, with skewness $-3$ and kurtosis 10. Per day, with 250 days a year:

| quantity | value |
|---|---|
| $\widehat{SR} = 2.5 / \sqrt{250}$ | 0.158114 |
| $\sqrt{V} = \sqrt{0.5 / 250}$ | 0.044721 |
| $Z^{-1}[1 - 1/100]$, $Z^{-1}[1 - 1/(100e)]$ | 2.3263, 2.6802 |
| $(1-\gamma) \cdot 2.3263 + \gamma \cdot 2.6802$ | 2.5306 |
| $\widehat{SR}_0 = 0.044721 \times 2.5306$ | **0.1132** |
| numerator $(0.158114 - 0.1132)\sqrt{1249}$ | 1.5883 |
| denominator $\sqrt{1 + 3 \times 0.158114 + \frac{9}{4} \times 0.158114^2}$ | 1.2372 |
| $\widehat{DSR} = Z[1.5883 / 1.2372] = Z[1.2838]$ | **0.9004** |

The paper reports $\widehat{SR}_0 \approx 0.1132$ and $\widehat{DSR} = 0.9004 < 0.95$: not a discovery at 95%. It also reports $\widehat{DSR} = 0.9505$ had there been only 46 trials, and 0.9505 after 88 trials had the returns been normal. `tests/test_deflate.py` reproduces all four numbers to four decimals.

### How blinded applies it

The observations are the 5 minute blocks of validation P&L, so $T$ is the number of validation blocks, 17 for the 85 minute validation window of the example brief. $\widehat{SR}$, $\hat\gamma_3$ and $\hat\gamma_4$ are those of the chosen strategy's block P&L; $V$ is the variance of the validation Sharpe ratios of every strategy trial with validation blocks; $N$ is the number of distinct strategy code hashes in the ledger, rejected trials included. Sharpe ratios are reported per $\sqrt{\text{hour}}$ as well, multiplied by $\sqrt{12}$.

The formula assumes independent trials, and an agent's trials are not: many are near copies of each other. Counting every hash overstates the independent $N$, which raises $\widehat{SR}_0$, but near copies also shrink $V$, the variance of the trials' Sharpe ratios, which lowers it, so whether the count errs the safe way depends on which wins; the paper's appendix 3 discusses estimating an effective $N$ from the correlation of the trials, which is not done here. $V$ is taken over distinct code hashes, each once, so a resubmission of the same code adds no duplicate column and does not shrink $V$ further.

The formula also assumes independent blocks, and a strategy that holds across several of the brief's blocks has fewer: on the carry run the final's validation blocks are autocorrelated 0.10 to 0.37 at lags 1 to 5. So $T$ is the number of validation blocks divided by the final's Newey-West long-run variance factor, $f = 1 + 2\sum_{k \le L}(1 - k/(L+1))\hat\rho_k$ with $L = \lfloor 4 (T/100)^{2/9} \rfloor$ lags and $f$ floored at 1 ({func}`blinded.deflate.nw_factor`): the brief's blocks stay the grid every trial shares, which $V$ and the PBO need, and the final's own dependence, measured rather than assumed from its holds, sets how many of them count. The report gives both, `blocks` and `blocks_effective`.

With $T = 17$ the factor $\sqrt{T - 1} = 4$ leaves the test little power: on one 85 minute window the DSR can rule a strategy out, but it can rarely confirm one. Whenever the number is printed, state these limits with it.

## The probability of backtest overfitting

Bailey, Borwein, López de Prado and Zhu, *The Probability of Backtest Overfitting*, Journal of Computational Finance, 2015.

### Combinatorially symmetric cross-validation

Put the trials' P&L side by side, a $T \times N$ matrix $M$ with one column per trial and synchronous rows, here the 5 minute validation blocks. Then (their Algorithm 2.3):

1. Cut the rows into $S$ consecutive submatrices of equal size ($S = 16$; rows left over at the end are dropped).
2. For each of the $\binom{S}{S/2}$ ways to choose half of the submatrices as a training set $J$, with the other half $\bar J$ as the test set:
   1. compute every trial's metric on $J$ and find the best, $n^*$;
   2. compute every trial's metric on $\bar J$ and the rank $\bar r$ of $n^*$ there, from 1 (worst) to $N$ (best);
   3. take $\bar\omega = \bar r / (N + 1)$ and the logit $\lambda = \ln\left(\bar\omega / (1 - \bar\omega)\right)$.
3. PBO is the share of combinations with $\lambda \le 0$: the strategy that won in sample was at or below the median out of sample.

Every block is in the training set of half the combinations and in the test set of the other half, and the two halves are always the same size, which is the symmetry in the name. For $S = 16$ there are $\binom{16}{8} = 12{,}870$ combinations; the paper's text says 12,780, a transposition. With no skill, the out-of-sample rank of the in-sample winner is uniform on $1, \dots, N$ and PBO is about one half; a value near 0 says the in-sample ranking carries over.

### A worked example

Four blocks A to D ($S = 4$, six combinations, the paper's Figure 1), three trials, and the mean as the metric so the arithmetic is visible:

| | A | B | C | D |
|---|---|---|---|---|
| trial 1 | +3 | +3 | -3 | -3 |
| trial 2 | 1 | 1 | 1 | 1 |
| trial 3 | -2 | -2 | +2 | +2 |

- Train on {A, B}: trial 1 wins with 3. On {C, D} it is the worst, $-3$: $\bar r = 1$, $\bar\omega = 1/4$, $\lambda = \ln(1/3) < 0$.
- Train on {C, D}: trial 3 wins with 2 and is the worst on {A, B}: $\lambda < 0$.
- Train on {A, C}, {A, D}, {B, C} or {B, D}: trials 1 and 3 average 0 and trial 2 wins with 1; it is also best out of sample, $\bar r = 3$, $\bar\omega = 3/4$, $\lambda = \ln 3 > 0$.

Two of six logits are at or below zero, so PBO = 1/3. Trials 1 and 3 look brilliant on half the data and are the worst on the other half; trial 2 is dull and consistent. In both of the bad combinations the in-sample winner also lost money out of sample, which is the paper's "probability of loss", 1/3 here.

### How blinded applies it, and when it refuses

The columns are the strategy trials with validation blocks and the rows are the validation blocks, so on the example brief 17 blocks give 16 submatrices of one block each, with one dropped, and each half is 8 blocks. The metric is the Sharpe ratio, mean over sample standard deviation; a trial that never traded has Sharpe 0. With fewer than 8 trials or fewer than 16 blocks the judge does not print a number; it says "not enough trials/blocks for PBO", because with a handful of trials every rank is coarse and a PBO of 0 or 1/3 is noise.

### What the tests check

The paper's own numerical results for PBO (55% and 13% on two simulated seasonal strategies) come from unseeded random walks and cannot be reproduced, so the tests check the definitions instead: the hand example above, the combination counts, a brute-force implementation of Algorithm 2.3 written with loops, a Monte Carlo check that pure noise gives a PBO of about one half, and a matrix with one real edge, which gives a PBO near zero.
