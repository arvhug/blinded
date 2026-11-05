# Getting started

## Install

blinded needs Python 3.11 or newer. From a clone of the repository:

```console
$ pip install .
```

For work on blinded itself, `uv sync` installs it with the test and docs dependencies.

## Get some data

The quickest start is a free day of Binance spot and perpetual data from Tardis, which needs no API key:

```console
$ pip install '.[demo]'
$ blinded demo tardis --date 2026-10-01 --symbols XRPUSDT --out ~/data/tardis --hours 14
```

That downloads the raw files, builds one parquet and writes a brief that points at it, and every command below takes that brief. {doc}`demo-data` says what is in the file and why it is built the way it is.

## Describe the data

The judge reads one parquet file of events plus a brief that says how to read it. This brief is for top-of-book quotes and trades in one long table, one row per event (the full file is `examples/demo.brief.yaml`):

```yaml
data:
  path: ${BLINDED_DATA_PATH}
  time: { column: ts, unit: ns, tz: UTC, clock: receive }
  instrument_column: symbol
  streams:
    book:   { filter: "is_trade == false", fields: { bid: bid0_px, ask: ask0_px, bid_qty: bid0_qty, ask_qty: ask0_qty } }
    trades: { filter: "is_trade == true",  fields: { price: price, qty: qty, side: side }, side_values: { buy: B, sell: S } }
  availability:
    default_delay_ms: 0

target:
  instrument: XRPUSDT
  price: mid
  horizons: [100ms, 1s, 10s, 60s]
  primary_horizon: 10s

grid: { step: 100ms, warmup: 5min }
execution: { latency_ms: 50 }
splits: { fractions: [0.50, 0.25, 0.25], embargo: 5min }
stats: { block: 5min, n_boot: 1000, seed: 20241128 }
```

`streams` splits the rows into book updates and trades, and `fields` maps the names the panel uses (`bid`, `ask`, `bid_qty`, `ask_qty`, `price`, `qty`, `side`) onto your columns. `availability` says how long after its timestamp a row becomes usable. With a receive clock that is zero. With an exchange clock it should be your feed delay; fills and returns are then read at their stamps, so a decision waits the delay plus the latency for its fill. `${VAR}` in the path is read from the environment, so the brief can live in a repository without the location of the data.

A run can also study more than one kind of strategy on the same data. Without a `families` block it has one, directional, on `target.instrument`. With one, it can add pairs and lead-lag:

```yaml
families:
  directional: { instruments: [XRPUSDT, XRPUSDT_PERP], share: 0.5 }
  pairs:       { pairs: [[XRPUSDT_PERP, XRPUSDT]], share: 0.5 }   # [A, B]: the spread of A over B
execution:
  taker_fee_bps: 7.5
  fees: { XRPUSDT: 10.0 }        # an instrument whose taker fee is not the default
```

Each family has its instruments, `instruments` for directional, `leaders` and `followers` for lead-lag, `pairs` for pairs, and a share of each phase's sessions; the shares sum to 1, and are equal when none is given. A feature or strategy says which family it belongs to in its decorator, `@bl.feature("...", family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))`, and `blinded init` refuses a brief that names an instrument the data does not have.

Before writing features, run

```console
$ blinded data --brief brief.yaml
```

It writes `DATA.md` with event rates, gaps between updates, how often the grid sees a stale book, and the lag between streams. If trades arrive late relative to the book, this is where you find out.

## Write a feature

A feature takes a panel and returns a polars expression. Register it with `@bl.feature` and a one-line hypothesis:

```python
import blinded as bl
from blinded import features as F


@bl.feature("perp leads spot; the basis mean-reverts")
def basis(panel):
    b = F.basis("XRPUSDT_PERP", "XRPUSDT")
    return b - F.roll_mean(b, "5m")
```

`F.basis` returns `pl.col("XRPUSDT_PERP.mid") / pl.col("XRPUSDT.mid") - 1`, an expression, and nothing is computed until the judge evaluates it on `panel.frame`. That frame is a wide polars DataFrame on the decision grid: a `ts` column plus one column per instrument and field, such as `XRPUSDT.mid`, `XRPUSDT.bid_qty` or `XRPUSDT.cum_signed_qty`. `panel.col("XRPUSDT", "mid")` is `pl.col("XRPUSDT.mid")` with a check that the column exists. Plain polars works too:

```python
@bl.feature("book pressure")
def pressure(panel):
    return F.imbalance("XRPUSDT").rolling_mean_by("ts", "10s")
```

A file can define several features, and a decorated function is still a plain function. To look at one in a notebook:

```python
panel = bl.Panel.from_brief("brief.yaml")
panel.eval(basis(panel))  # a polars Series, one value per grid point
panel.frame.select("ts", basis(panel).alias("basis"))
```

The {doc}`feature library <features>` covers the usual transforms, and every function in it is causal by construction. Time spans can be written the polars way (`"5m"`, `"100ms"`) or the pandas way (`"5min"`).

If you would rather write pandas, return a pandas Series indexed like `panel.to_pandas()`:

```python
@bl.feature("the same basis, in pandas")
def basis_pd(panel):
    df = panel.to_pandas("XRPUSDT_PERP.mid", "XRPUSDT.mid")
    b = df["XRPUSDT_PERP.mid"] / df["XRPUSDT.mid"] - 1
    return b - b.rolling("5min").mean()
```

A polars Series with one value per grid point is also accepted. The judge treats all three the same way; {ref}`lazy` explains why the expression is the cheaper one.

## Evaluate it

```console
$ blinded eval features.py --brief brief.yaml
$ blinded eval features.py --brief brief.yaml --name basis --window test
$ blinded eval features.py --brief brief.yaml --window 2024-11-28T05:00/2024-11-28T08:00 --latency-ms 0
```

`--window` takes `train`, `val`, `test`, `all` or an explicit `START/END`; the named windows come from `splits` in the brief, minus the grid warm-up. Local mode does not stop you scoring the test window. Protected mode does: `blinded init` puts validation and test where only the judge can read them; see {doc}`run-directory`. The command exits with status 1 when any feature is rejected, which makes it usable in CI.

The same from Python:

```python
result = basis.evaluate("brief.yaml", window="train")
result.ok  # True if it passed
result.reason  # the truncation verdict, or why it failed
result.to_frame()  # metrics by horizon, a polars DataFrame
result.to_pandas()  # the same, indexed by horizon
print(result)  # the text report below
```

## Read the output

Here is `blinded eval --window all` on the synthetic file the test suite builds, where the perp quotes a random walk and the spot copies it 300 ms late:

```text
basis  [accepted]  perp leads spot; the basis mean-reverts
window all [2024-11-28T02:05:00.200000+00:00, 2024-11-28T02:30:00+00:00)  latency 50ms
truncation test: passed: deterministic, matches the full run at all 7 cuts

by horizon (IC: Pearson, rankIC: Spearman; SEs allow for overlapping returns)
           n  n_eff      IC   se_nw  se_boot  rankIC   se_nw  se_boot     t_nw  mkout_bps  tails_bps
100ms  14996  14996  0.4602  0.0049   0.0056  0.4016  0.0055   0.0060  73.0390     0.2303     0.5683
1s     14987   1499  0.4927  0.0104   0.0097  0.4661  0.0101   0.0099  45.9360     0.7578     1.7975
10s    14897    149  0.1641  0.0139   0.0090  0.1626  0.0142   0.0086  11.4138     0.8160     2.0343
60s    14397     24  0.0480  0.0171   0.0114  0.0594  0.0176   0.0131   3.3764     0.8883     1.4855

decay
         100ms   200ms   500ms      1s      2s      5s     10s     30s     60s
IC      0.4602  0.6195  0.6507  0.4927  0.3495  0.2158  0.1641  0.0997  0.0480
rankIC  0.4016  0.5716  0.6209  0.4661  0.3350  0.2108  0.1626  0.1106  0.0594

turnover 0.4224 sign changes per step, lag-1 autocorrelation 0.7119

stability at the primary horizon, 1/1 windows share the sign
        02:00
rankIC  0.163
```

The header gives the verdict, the scoring window and the latency. The window starts 5 minutes in because the brief's warm-up is never scored.

In the table, each row is a horizon from the brief:

- `n` is the number of grid points scored, `n_eff` the same divided by the horizon in grid steps. At 10 s on a 100 ms grid, 14,897 points are worth about 149 independent ones.
- `IC` is the Pearson correlation of the feature with the forward return and `rankIC` the Spearman one. Each has two standard errors, Newey-West (`se_nw`) and a circular block bootstrap (`se_boot`); {doc}`concepts` explains both. `t_nw` is the rank IC over its Newey-West error.
- `mkout_bps` is the mean of the feature's sign times the forward return, in basis points. `tails_bps` is the same over the top and bottom deciles of the feature only.

`decay` is the IC at a fixed ladder of horizons. Its shape tells you how fast you have to be: here the information peaks around 500 ms, which is what a 300 ms copy delay should produce. `turnover` is how often the sign of the feature flips per grid step, and `stability` is the rank IC in each 30-minute clock window.

The forward return starts at `g + latency`, not at `g`. Rerun with `--latency-ms 0` and the 1 s rank IC on this file goes from 0.466 to 0.550; at 300 ms it falls to 0.141. That difference is the part of the move that happens before an order could reach the market.

A rejected feature reads like this:

```text
local_gap  [rejected_lookahead]  spot mid against its 5-second local average
window train [2024-11-28T02:05:00.200000+00:00, 2024-11-28T02:15:00+00:00)  latency 50ms
truncation test: rejected_lookahead: cut at 2024-11-28T02:10:00.137000+00:00: first mismatch at 2024-11-28T02:09:57.800000+00:00, 24/6000 points differ (no finite diff, 24 NaN-vs-number)
warning: line 9: center=True puts future points in the window
```

The feature used `rolling_mean(50, center=True)`. Cutting the data at 02:10:00.137 turned the last 24 values before the cut into NaN, because the centred window needed points that were no longer there. A rejected feature gets no metrics. The warning comes from a static scan of the source and only helps you find the line; the cut is what decides.
