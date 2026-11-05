# Concepts

## The panel: events, availability and the grid

Every row of the data file is an event, a book update or a trade, stamped with `ts`. The brief gives each stream of each instrument a delay, and the event becomes usable at

$$
\text{available\_at} = ts + \text{delay}.
$$

With a receive clock the delay is zero, since you had the row when you stamped it. With an exchange clock it should be your feed latency: a row is no use to you before it arrives.

Decisions happen on a regular grid $g_0, g_0 + \Delta, g_0 + 2\Delta, \dots$, 100 ms in the example brief. A field's value at grid time $g$ is the value carried by the last event available by then:

$$
x(g) = x_{e^*}, \qquad e^* = \max\{\, e : \text{available\_at}_e \le g \,\}.
$$

This is a backward as-of join, polars' `join_asof(strategy="backward")` of each stream's events onto the grid, and the panel offers no other. An event available exactly at $g$ counts. Rows that share a timestamp are resolved in file order and the last one wins, which is what a feed handler would see. A grid point before the first event of a stream is null.

The joined values live in `panel.frame`, one polars DataFrame with a row per grid point: `ts`, then a column per instrument and field, named `SYM.field`.

| Columns | What they hold at $g$ |
|---|---|
| `bid`, `ask`, `bid_qty`, `ask_qty`, `mid`, `microprice` | the book as-of $g$ |
| `price`, `qty`, `side` | the last trade as-of $g$ (`side` is +1 buy, -1 sell) |
| `cum_book`, `cum_trades` | how many events of the stream have arrived by $g$; a stream with an `n` field, the raw events each row of a thinned file stands for, adds up `n` instead |
| `cum_qty`, `cum_buy_qty`, `cum_sell_qty`, `cum_signed_qty`, `cum_notional` | running totals over the trades that have arrived by $g$ |

A sum over the trailing interval $(g - w, g]$ is the difference of two running totals,

$$
S_w(g) = C(g) - C(g - w),
$$

where $C(g - w)$ is the total at the last grid point at or before $g - w$, and 0 before the first one. When $w$ is a whole number of grid steps that point is $g - w$ itself, and $S_w(g)$ counts exactly the events with $g - w < \text{available\_at} \le g$. The trade features in the library are built this way. A window that is not a whole number of steps starts at the grid point below $g - w$, so it runs up to one step long. `panel.book(sym)` and `panel.trades(sym)` give the raw events with their `available_at` column if you need them, and `panel.to_pandas()` gives the frame indexed by `ts`.

`panel.cut(t)` returns the same panel with every event available after $t$ removed and every grid point after $t$ dropped. Nothing on the grid at or before $t$ depends on what was removed, so a causal feature computed on `panel.cut(t)` agrees with the full computation at every grid point up to $t$. The truncation test is built on that. Because every column at $g \le t$ comes from events available by $g$, the cut frame is just the first rows of the full frame, a slice that copies nothing; `tests/test_panel.py` checks that it equals a frame rebuilt from the kept events, for every column type.

(lazy)=
## Expressions and lazy evaluation

What a feature returns describes a computation, which polars runs later. `F.basis("XRPUSDT_PERP", "XRPUSDT")` is a small tree: divide one column by another, subtract one. The judge evaluates it as `panel.frame.lazy().select(expr)`, which hands the whole tree to polars before any of it runs, and polars plans it. For the reference feature in the test suite, `b - F.lag(F.roll_mean(b, 3000), 1)` with `b` the log basis in bps, `explain()` prints

```text
SELECT [(col("__POLARS_CSER_0x100000007") - col("__POLARS_CSER_0x100000007").rolling_mean().shift(1))]
   WITH_COLUMNS: [((col("XRPUSDT_PERP.mid") / col("XRPUSDT.mid")).log() * 10000.0).alias("__POLARS_CSER_0x100000007")]
    DF [...]; PROJECT["XRPUSDT_PERP.mid", "XRPUSDT.mid"] 2/65 COLUMNS
```

(trimmed). Before computing anything, polars has dropped every column the feature does not name: the frame for a four-instrument file has 65 columns and the plan reads 2. It has also noticed that `b` appears twice and computes it once, as `__POLARS_CSER_...`. That only works for the outermost repeated pieces of a tree; a repeat nested inside another repeat is computed each time. Then it runs the plan in one pass, and the log basis, its rolling mean and the shifted mean each live only until the next step has used them.

The pandas version of the same feature builds `spot`, `perp`, `b`, the rolling mean and the result as separate full-length Series, each bound to a name for as long as the function, or the notebook cell, keeps it. On the 100 ms grid that is small: a day is 864,000 rows, 6.9 MB per float column. Notebooks run out of memory on the data upstream of the grid. A day of top-of-book and trades for a few dozen instruments is tens of millions of event rows, and in pandas every filter, merge and sort returns a full copy of every column. Each copy that is bound to a name stays alive, and IPython keeps a reference to every displayed result in `Out[n]` and `_`, so memory climbs with every cell until you delete names or restart the kernel. A lazy polars query over the same events (filter by symbol, join, aggregate) is planned as a whole: filters run before joins, columns nothing uses are never read, and intermediates are freed as soon as the next step has consumed them. The peak is the inputs plus whatever one step of the plan needs, where pandas pays for every intermediate you kept.

Measured on a real file (6 hours of 4 instruments, 933,000 event rows, 216,000 grid points), in seven fresh processes: the wide frame is 108 MB, built once and shared by every feature and every cut, since cuts are slices. Resident memory after loading is 443 MB, against about 408 MB for the earlier pandas implementation. Judging the reference feature (two full runs, six cuts, targets and metrics) then adds at most 3 MB to the process's peak, where that implementation added 15 to 38 MB building its intermediates. The peak, about 494 MB, is set by reading the parquet file, which blinded does eagerly.

## Latency and the return being scored

A decision at $g$ uses the state as-of $g$. An order sent then is live at $g + L$, with $L$ = `execution.latency_ms`. The judge scores the feature against

$$
r_h(g) = \frac{p(g + L + h)}{p(g + L)} - 1,
$$

where $p$ is the target instrument's mid (or microprice) read as-of each of the two times straight from the event stream. Those times are not rounded to the grid, so $L = 50$ ms means 50 ms on a 100 ms grid. A return whose end falls after the last grid point is NaN.

The two times are stamps. With a feed delay $d$, the decision at $g$ has seen the rows stamped by $g - d$, and the order meets the book stamped $g + L$, which is the book available by $g + L + d$; the judge reads prices there, each instrument through its own delay. From what the decision knew to the start of the return is then $L + d$. With a receive clock $d$ is 0.

A feature of the pairs family, with `pair=(A, B)`, is scored against the spread return instead, the forward log return of A less that of B from the same times,

$$
r^s_h(g) = \ln\frac{m_A(g + L + h)}{m_A(g + L)} - \ln\frac{m_B(g + L + h)}{m_B(g + L)},
$$

which is the return of one notional long A and one short B. There is no hedge ratio in it: a ratio is fitted, it belongs to a strategy, and a target with a fitted parameter would make the score depend on a knob. For a spot and its perpetual this is the change of the log basis. A directional feature with `target=` is scored against that instrument's own forward return.

Starting at $g + L$ instead of $g$ removes the part of the move that happens before you could have traded. For fast signals that part is often most of the signal. `--latency-ms 0` shows the IC without the haircut; read it as a diagnostic, never as the result. The gap between the two is what your latency costs.

On the synthetic file in the test suite, where the spot copies the perp 300 ms late, the basis feature's 1 s rank IC is 0.550 with returns from $g$, 0.466 from $g$ + 50 ms and 0.141 from $g$ + 300 ms. Two judges that disagree on this convention will disagree on the number, so the latency is always printed next to it.

## What a maker's score is

A taker's fills follow from its orders; a maker's do not, because where its order sits in the queue is never observed. The simulator runs each quote sequence under four models of the queue ({doc}`sim`), and the judge scores the worse of the two that err in opposite directions, risk-averse (RA) and pro rata (PR), by the hold rule's Sharpe: the net P&L per square-root hour over blocks as long as the strategy's own holds ({doc}`ladder`). Net is in bps of one unit of the quote's size, after the maker fee, funding and the flatten at the window's end. The report splits the gross into what the fills captured against the mid at their stamps, what the mid did over the next 1, 10 and 60 s, and the rest, the inventory. Where the maker fee is larger than the half spread, a quote pays only if the mid after its fills moves its way by more than the difference, and the verdict line says so when it fails. An arbitrage quote adds its hedge: the hedge's taker fee, the walk of the hedge venue's book and the delay to it are costs of the same fill, and its net is given in bps a unit hedged and in USD at the brief's size.

## The truncation test

Look-ahead bugs are short and look harmless: `rolling_mean(50, center=True)`, `(x - x.mean()) / x.std()`, an exponential average seeded at the sample mean, a `join_asof` with `strategy="nearest"` (`merge_asof` with `direction="nearest"` in pandas). Reading code to find them depends on the reader knowing every way polars or pandas can see the future, and an optimiser (a person tuning, or an agent) is very good at finding the ways the reader does not know. What these bugs have in common is that some past value changes when the future is deleted. The truncation test checks only that, by running the code.

### Definition

Let $f$ be the feature on the full panel and $f^{(t)}$ the feature on `panel.cut(t)`. The feature passes if for every cut $t_i$ and every grid point $g \le t_i$

$$
f^{(t_i)}(g) = f(g)
$$

under the equality rule below. A failure reports the first cut that failed, the first grid point that changed, how many points changed and the largest difference, but not the values.

The comparison runs over every grid point up to the cut, from the start of the panel and not only inside the scoring window. That is how an exponential average seeded at the full-sample mean gets caught: its first values depend on the mean of everything, and the mean changes with the cut.

### Equality

Two values are equal when `isclose(a, b, rtol=1e-7, atol=1e-10 * scale)`, with `scale` the median absolute value of the full run (1 if that is 0). Summing in a different order moves values by about $10^{-13}$ relative, while look-ahead moves them on the scale of the feature, so the threshold has plenty of room on both sides. NaN equals NaN. NaN against a number fails in either direction: a causal feature sees the same history in both runs, so its warm-up NaNs are in the same places.

### Determinism

Before any cut the feature runs twice on the full panel. If the runs differ the verdict is `rejected_nondeterministic`. Unseeded random numbers would otherwise look like look-ahead at the first cut.

### One fresh process per evaluation

The judge never runs the submitted code itself. Each evaluation, the full panel or one cut, is a worker of its own (`blinded.isolate`): a fresh interpreter that is sent one panel, imports the trial's file, calls the function once, sends the values back and exits. A module-level cache cannot carry from one evaluation to the next. Run in one process, a dict keyed on the panel's length handed every cut a slice of the full run, and a forward return passed with validation rank IC 0.993 (the red-team audit of 20 Oct 2026). A cut's worker is sent a panel pickled from the cut's rows alone, so it holds no buffer that reaches past the cut, and nothing the code changes in the libraries it imports reaches the process that computes the verdict. The judge reads a worker's answer as JSON and an `.npz` without pickles, and checks every array's length and keys again. Each worker costs a fresh interpreter and the panel's trip through a pipe. On three published briefs, timed in one session on a quiet machine, the workers take a judgement from 9.1 to 18.1 s (taker), 21.5 to 28.0 s (market making) and 50.6 to 56.7 s (arbitrage).

### Where the cuts go

Seven cuts go in the scoring window: at its 0.30, 0.50, 0.70, 0.85 and 0.95 quantiles, at one seeded random point, and in the middle of the window's last block (`stats.block`). The last one is there because the 0.95 cut alone leaves the last 5% of the window unchecked; a leak confined to it would pass. The 0.50 cut runs first, so a blatant leak is rejected after three evaluations. Every cut sits 0.37 of a grid step after a grid point. It never coincides with one, so off-by-one boundary handling gets exercised, and events that arrive between the last grid point and the cut belong to the truncated panel.

A forward or nearest as-of join leaks only across one inter-event gap. If the joined stream's next event arrives before the cut, the full and truncated runs pick the same row and nothing shows. So each cut moves, within the next 100 grid points, to the one where the most streams have their next event after the cut and nearer than their previous event. There a nearest join in the full run reaches past the cut, and in the truncated run it cannot.

Among points that tie on that count, the cut goes to the one where the most columns of the frame change between it and the next grid point. A leak that reads later grid rows, such as a one-second bar labelled on the left, is invisible when the rows after the cut repeat the row before it: on a sticky 100 ms mid most rows do. On the six-hour example file a cut at an arbitrary point exposes the one-second bar leak about 54% of the time, and a cut placed by these two rules about 68%; over seven cuts that is a miss rate under 0.1%.

### What it catches and what it misses

It catches anything that makes a past value depend on later data: negative shifts, centred windows, full-sample z-scores, ranks and scalers, backfill, forward and nearest joins, bars labelled on the left (`resample` in pandas, `group_by_dynamic` in polars), cumulative statistics divided by a sample total, reversed cumulative sums, and any group aggregate broadcast back with `groupby().transform()` or `.over()`. The repository's `tests/planted/` holds eight such leaks, each written once in polars and once in pandas, and six causal controls; the suite checks that all sixteen leaks are rejected and every control passes, on synthetic data and on real data.

The static scan that runs before the cuts warns about the patterns it can read in the source, in both libraries: negative `shift`, `center=True`, forward or nearest as-of joins, backward fills, `cum_sum(reverse=True)`, interpolation, and a whole-column aggregate such as `x.mean()` combined with `x` row by row. It never rejects. Two of the planted leaks, the sample-mean seed and the left-labelled bars, contain none of these patterns, which is why the cuts decide.

The truncation test misses:

- A leak that reaches only from the warm-up into the first few points, such as `bfill()` of leading NaNs. The cuts sit after the warm-up, so both runs fill the same way.
- Some event-time leaks, some of the time. A nearest join is caught only when the value it would grab after the cut differs from the one before. Over seven cuts the chance of missing one is small, but it is not zero.
- A leak confined to the time after the last cut, at most half a block at the window's end.
- Clock problems in the data itself. If the file has an exchange clock and you leave the delay at zero, the panel is too early and every feature inherits that. Fix it with `availability` in the brief; `DATA.md` helps you spot it.

### A worked example

Take ten grid points at 1, 2, ..., 10 s and one quote at $k - 0.5$ s with mid $k^2$, so the mid on the grid is $1, 4, 9, \dots, 100$. The feature is `pl.col("A.mid").rolling_mean(3, center=True)`, or in pandas `panel.to_pandas()["A.mid"].rolling(3, center=True).mean()`; both give the same verdict.

On the full panel, the value at 5 s is $(16 + 25 + 36)/3 = 25.67$.

The 0.50 cut sits after grid point `int(0.5 * 9) = 4`, the fifth point, which is 5 s, so $t = 5.37$ s. The truncated panel keeps the quotes at 0.5 to 4.5 s and the grid points 1 to 5 s. Its mids are $1, 4, 9, 16, 25$. The centred window at 5 s needs the point at 6 s, which is gone, so the value is NaN.

Comparing the two runs over 1 to 5 s: at 1 s both are NaN, because the window needs a point at 0 s; 2, 3 and 4 s agree; at 5 s the truncated run has NaN against 25.67. The verdict is `rejected_lookahead: cut at 5.37 s, first mismatch at 5 s, 1/5 points differ (1 NaN-vs-number)`. The same feature with a trailing window, `rolling_mean(3)`, passes every cut. `tests/test_truncation.py` runs this example in both libraries.

## Overlapping returns and standard errors

On a 100 ms grid with a 10 s horizon, neighbouring targets share 99 of their 100 steps. The scored sample has $n$ points and far fewer independent ones, so the judge prints

$$
n_{\text{eff}} = \frac{n}{h / \Delta}
$$

beside every IC, and two standard errors that allow for the dependence.

### Newey-West on the influence function

For standardised $z_x, z_y$ the sample correlation is $\hat r = \frac{1}{n} \sum_t z_{x,t} z_{y,t}$. Its influence function is

$$
\psi_t = z_{x,t} z_{y,t} - \frac{\hat r}{2}\left(z_{x,t}^2 + z_{y,t}^2\right),
$$

and $\hat r - r \approx \frac{1}{n}\sum_t \psi_t$, so the variance of $\hat r$ is the long-run variance of $\psi$ divided by $n$. With a Bartlett kernel,

$$
\widehat{\operatorname{Var}}(\hat r) = \frac{1}{n^2}\left(\sum_t \psi_t^2 + 2\sum_{k=1}^{K}\left(1 - \frac{k}{K+1}\right)\sum_t \psi_t\, \psi_{t-k}\right),
\qquad K = \left\lceil \frac{h}{\Delta} \right\rceil + \left\lceil 1.5\, n_{\text{eff}}^{1/3} \right\rceil.
$$

The bandwidth is at least the overlap length, plus the usual growth term. The sum goes through `statsmodels`. Rank IC uses the same formula on ranks.

### Circular block bootstrap

The second estimate resamples blocks of consecutive grid points, `stats.block` long (5 minutes by default), wrapping around the end, drawing the same blocks for the feature and the target so that dependence inside a block survives. Each replicate is computed from block sums of $x$, $y$, $x^2$, $y^2$ and $xy$, so it costs one addition per block. The SE is the standard deviation of `n_boot` replicates and the 90% interval is their 5th to 95th percentile. For rank IC the ranks are computed once on the whole window and their correlation is bootstrapped.

The two estimates should be close. When they are not, the usual cause is too few blocks: three hours of data in 5-minute blocks is only 36.

### When overlap matters

Overlapping targets alone do not inflate the error. If the feature is serially independent, $\psi_t$ stays uncorrelated even though the targets overlap, and $1/\sqrt{n}$ is about right. The error grows when both series are persistent. With no true correlation, Bartlett's formula gives

$$
\operatorname{Var}(\hat r) \approx \frac{1}{n}\sum_{k=-\infty}^{\infty} \rho_f(k)\, \rho_r(k),
$$

where $\rho_f$ and $\rho_r$ are the autocorrelations of the feature and of the target. A slow feature against a 10 s target on a 100 ms grid makes that sum large. The crude bound $1/\sqrt{n_{\text{eff}}}$ is much closer to the truth than $1/\sqrt{n}$ in that case, but it is still a rough bound, and the judge never reports it as the SE. `tests/test_metrics.py` has both cases.
