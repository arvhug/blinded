# Feature library

`bl.features` holds the standard transforms so you do not have to rewrite rolling and exponential boilerplate, or get its edge cases wrong. Every function only looks backwards, and the test suite runs each one through the truncation test on synthetic data and, when it is present, on real data. A library function that fails is a bug.

Every function builds a polars expression and computes nothing. The book, trade and cross-instrument functions take symbols and read the panel's `SYM.field` columns: `F.mid("XRPUSDT")` is `pl.col("XRPUSDT.mid")`. The series transforms take an expression and return one, so they chain, and they accept any expression, not only panel columns:

```python
import blinded as bl
from blinded import features as F


@bl.feature("a stretched basis pulls spot toward the perp")
def basis_z(panel):
    return F.ewm_zscore(F.basis("XRPUSDT_PERP", "XRPUSDT"), halflife="5m")
```

Everything is exported flat (`F.basis`, `F.microprice`, `F.roll_mean`) and also through its submodule (`F.cross.basis`).

Windows are a number of rows (`3000`) or a time span. Spans can be written the polars way (`"5m"`, `"100ms"`, `"1h30m"`) or the pandas way (`"5min"`); `blinded.durations.to_ns` reads both, and every function goes through it. An integer window is `rolling_mean(n)` and friends: null until it is full. A span is `rolling_mean_by("ts", w)`: the window is $(t - w, t]$ of the frame's `ts` column and it emits from the first row, as pandas does. Time lags (`lag(x, "500ms")`) take the last row at or before $t - h$ and assume `ts` strictly increases, as the grid does.

Halflives are a number of rows, giving $\alpha = 1 - 2^{-1/h}$ at every step, or a span, which uses `ewm_mean_by("ts", half_life=h)` and decays by the time since the previous row, $\alpha_t = 1 - 2^{-\Delta t / h}$. On a regular grid the two agree. RSI periods are rows only.

Missing values are null rather than NaN. A ratio with a zero denominator, a z-score of a constant window and a trade ratio over a window with no trades are null; the judge reads null as NaN.

Trade windows are sums over $(t - w, t]$, each the difference of one of the panel's `cum_*` columns at $t$ and at $t - w$ (`F.window_sum`). They count exactly the trades available in the window when $w$ is a whole number of grid steps, which 1 s, 10 s and 5 min are on a 100 ms grid; other windows start at the grid point below $t - w$ and run up to one step long.

Every example in the docstrings below runs as a doctest. The panel examples use {func}`~blinded.features.example.example_panel`, a four-point panel whose values you can check by hand.

## Catalogue

$W_t$ is the trailing window ending at $t$, $\mu_W$ and $\sigma_W$ its mean and sample standard deviation, $b, a, q_b, q_a$ the best bid, ask and their sizes.

### Series (`bl.features`, also `bl.features.series`)

| Function | Formula | Causal note |
|---|---|---|
| `lag(x, n)` | $x_{t-n}$ | a negative lag raises |
| `diff(x, n)` | $x_t - x_{t-n}$ | |
| `ret(x, h, log=True)` | $\ln(x_t / x_{t-h})$ or $x_t/x_{t-h} - 1$ | backward only; there is no forward return |
| `roll_mean(x, w)` | mean over $W_t$ | window includes $t$ |
| `roll_std(x, w)` | sample sd over $W_t$ | |
| `roll_sum(x, w)` | sum over $W_t$ | |
| `roll_min(x, w)`, `roll_max(x, w)` | min, max over $W_t$ | |
| `roll_median(x, w)` | median over $W_t$ | |
| `roll_quantile(x, w, q)` | $Q_q$ over $W_t$, linear interpolation | trailing quantile, not full-sample |
| `roll_rank(x, w)` | rank of $x_t$ in $W_t$ over the window length | ties share their average rank |
| `roll_skew(x, w)` | adjusted sample skewness | |
| `roll_kurt(x, w)` | bias-corrected excess kurtosis | |
| `roll_corr(x, y, w)` | correlation over $W_t$ | |
| `roll_beta(y, x, w)` | $\operatorname{cov}_W(y, x) / \operatorname{var}_W(x)$ | null where $x$ is constant |
| `zscore(x, w)` | $(x_t - \mu_W) / \sigma_W$ | trailing statistics; null where constant |
| `meanrev(x, w)` | $-(x_t - \mu_W) / \sigma_W$ | sign is the reversion forecast |
| `expanding_zscore(x)` | $(x_t - \bar x_{\le t}) / s_{\le t}$ | depends on the start, never the end |
| `ewm_mean(x, hl)` | $m_t = (1-\alpha_t) m_{t-1} + \alpha_t x_t$, $\alpha_t = 1 - 2^{-\Delta t/h}$ | seeded at the first observation |
| `ewm_std(x, hl)` | $\sqrt{v_t}$, $v_t = (1-\alpha_t)(v_{t-1} + \alpha_t (x_t - m_{t-1})^2)$ | $v_0 = 0$ |
| `ewm_zscore(x, hl)` | $(x_t - m_t) / \sqrt{v_t}$ | null at the first point |
| `ewm_cross(x, fast, slow)` | $m^{\text{fast}}_t - m^{\text{slow}}_t$ | |
| `realized_vol(x, w)` | $\sqrt{\sum_{W_t} (\Delta \ln x)^2}$ | not annualised |
| `momentum(x, w)` | $\ln(x_t / x_{t-w}) / \text{RV}_w(t)$ | |
| `rsi(x, n)` | $100\, G_t / (G_t + L_t)$, Wilder averages | seed is the mean of the first $n$ changes; $n$ in rows |
| `bollinger_b(x, w, k)` | $(x_t - \mu_W + k\sigma_W) / (2k\sigma_W)$ | |
| `roll_winsorize(x, w, q)` | clip to the trailing $[Q_q, Q_{1-q}]$ | bounds from the window, not the sample |
| `time_of_day(ts=None)` | hours since midnight | reads `ts` unless given another column |

### Book (`bl.features`, also `bl.features.book`)

| Function | Formula | Causal note |
|---|---|---|
| `mid(sym)` | $(b + a)/2$ | as-of the grid point |
| `spread(sym, bps=False)` | $a - b$, or $10^4 (a - b)/m$ | |
| `spread_ticks(sym, tick)` | $\operatorname{round}((a - b)/\text{tick})$ | tick is passed in, never inferred from the data |
| `microprice(sym)` | $(b\, q_a + a\, q_b)/(q_b + q_a)$ | |
| `imbalance(sym)` | $(q_b - q_a)/(q_b + q_a)$ | |
| `depth_imbalance(sym, levels=5)` | $(\sum_k q^b_k - \sum_k q^a_k)/(\sum_k q^b_k + \sum_k q^a_k)$ over the first `levels` levels | reads `SYM.bid{k}_qty` and `SYM.ask{k}_qty` behind the touch, which a file built with five levels carries |
| `micro_minus_mid(sym, bps=False)` | microprice minus mid | |
| `quote_rate(sym, window)` | book rows per second in $(t - w, t]$ | from `SYM.cum_book` |

### Trades (`bl.features`, also `bl.features.trades`)

Sums and counts are 0 over a window with no trades; ratios are null.

| Function | Formula | Causal note |
|---|---|---|
| `signed_volume(sym, window)` | $\sum_{W_t} s_i q_i$ | trades available in $(t - w, t]$ |
| `trade_count(sym, window)` | number of trades | |
| `trade_imbalance(sym, window)` | $(V_B - V_S)/(V_B + V_S)$ | |
| `vwap(sym, window)` | $\sum p_i q_i / \sum q_i$ | |
| `vwap_dev(sym, window)` | $10^4 (m_t / \text{VWAP}_W - 1)$ | |
| `avg_trade_size(sym, window)` | $\sum q_i / N$ | |
| `kyle_lambda(sym, window)` | trailing OLS slope of step mid return on step signed volume | regression over past steps only |

### Cross-instrument (`bl.features`, also `bl.features.cross`)

| Function | Formula | Causal note |
|---|---|---|
| `basis(a, b, log=False)` | $m_a / m_b - 1$ or $\ln(m_a/m_b)$ | |
| `basis_z(a, b, window= or halflife=)` | trailing or EW z-score of the basis | |
| `lead_ret(leader, h)` | $\ln(m_t / m_{t-h})$ of the leader | the leader's past return, used against another instrument |
| `lead_gap(leader, follower, h)` | $r_\ell(t, h) - r_f(t, h)$, trailing log returns | the move the follower has not made yet |
| `lead_gap_z(leader, follower, h, window)` | the gap over its trailing sd | no mean taken off: the gap is centred on zero |
| `lagged_corr(leader, follower, k, window)` | trailing correlation of $r_f(t, 1)$ with $r_\ell(t - k, 1)$ | how strongly the leader's move $k$ ago has been followed lately |
| `rel_spread(a, b)` | $s_a / s_b$, spreads in bps | |
| `ret_corr(a, b, window)` | trailing correlation of step log returns | |
| `ret_beta(a, b, window)` | trailing beta of $a$'s step returns on $b$'s | |
| `hedge_ratio(a, b, window, every)` | OLS slope $\tilde\beta$ of $\ln m_a$ on $\ln m_b$ | fitted on the window before each clock window of `every`, held through it |
| `pair_spread(a, b, window, every)` | $\ln m_a - \tilde\beta \ln m_b$ | the current mids, the held ratio |
| `spread_z(a, b, window, every)` | $(s_t - \tilde\mu) / \tilde\sigma$, mean and residual sd from the same fit | null for an exact fit |
| `half_life(x, window)` | $-\ln 2 / \ln \hat\varphi$ in steps, $\hat\varphi$ the AR(1) slope with an intercept | infinite when $\hat\varphi \ge 1$ |

### Cross-section (`bl.features`, also `bl.features.xs`)

A basket feature returns one value per leg, a mapping over `panel.legs`, the instruments of the brief's basket family in order. These build one and combine its legs row by row: each row's values across the legs, and nothing else, so they are as causal as their inputs. Nulls and NaN are left out of a row's mean, spread and ranks.

```python
rev = bl.short_term_reversal()


@rev.feature("the names that fell most over the last four hours bounce")
def rev_score(panel):
    return F.xs_zscore(F.each(panel.legs, lambda s: -F.ret(F.mid(s), "4h")))
```

| Function | Formula | Note |
|---|---|---|
| `each(legs, fn)` | $\{i: f(i)\}$ | a function of a symbol on every leg |
| `xs_mean(d)` | $\bar x_t$ on every leg | |
| `xs_demean(d)` | $x_{i,t} - \bar x_t$ | |
| `xs_zscore(d)` | $(x_{i,t} - \bar x_t) / s_t$, $s_t$ the sample standard deviation across the legs | null with fewer than two legs or all equal |
| `xs_rank(d)` | rank of $x_{i,t}$ across the legs, 1 the smallest, ties averaged | null where the leg has none |

The `xs_` functions return a {class}`~blinded.features.xs.Cross`: the legs kept as one polars list expression, each leg an element of it. The judge evaluates a `Cross` once and splits it, where it evaluates any other mapping leg by leg; on twenty legs that is about 0.1 s against 20 times that, and the functions nest (`F.xs_zscore(F.xs_demean(...))`) without multiplying the work. A mapping you build from a `Cross` by hand, such as `{s: -F.zscore(x, "1d") for s, x in resid.items()}`, is evaluated leg by leg and costs each leg the whole cross-section.

### Consolidated (`bl.features`, also `bl.features.consolidated`)

| Function | Formula | Causal note |
|---|---|---|
| `theo(coin, venues=None, field="theo")` | the Kalman filter's consolidated price of `coin` over the brief's `data.theo` legs, as-of each grid point; `field` also `sd_bps`, `innovation_z` or `basis:<venue>` | computed once at `blinded init`, one event at a time, with parameters fitted on train only ({mod}`blinded.theo`); a prefix recursion, so a truncation leaves every earlier value unchanged |

A brief that declares `data.theo: {coin, legs, anchor}` gets the columns `THEO.<coin>.theo`, `.sd_bps`, `.innovation_z` and `.basis_<venue>`. The model is one latent log price and one basis per venue, each quote's log mid a reading of the two with variance $\sigma_v^2 + \sigma^2 a$, $a$ the quote's age on receipt (`age_ms`); the anchor fixes the origin, one venue's basis at zero (the theo in that venue's terms) or a weighted sum of bases at zero (a consensus). The price's volatility, each venue's noise and the event-clock weight are fitted by the prediction-error likelihood, Nelder-Mead, on train's stream conflated to one event per venue per second; the basis wander is held at 0.02 bps per square-root second. `vault/theo.json` keeps the parameters, the fit and the innovation report by segment; DATA.md prints the parameters and train's report.

## Reference

### Series

```{eval-rst}
.. automodule:: blinded.features.series
   :members:
```

### Book

```{eval-rst}
.. automodule:: blinded.features.book
   :members:
```

### Trades

```{eval-rst}
.. automodule:: blinded.features.trades
   :members:
```

### Cross-instrument

```{eval-rst}
.. automodule:: blinded.features.cross
   :members:
```

### Cross-section

```{eval-rst}
.. automodule:: blinded.features.xs
   :members: each, xs_mean, xs_demean, xs_zscore, xs_rank, Cross
```

### Window sums

```{eval-rst}
.. autofunction:: blinded.features.window_sum
```

### Example panel

```{eval-rst}
.. automodule:: blinded.features.example
   :members:
```
