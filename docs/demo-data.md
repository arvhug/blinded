# Demo data

The loop is built and shown on a day anyone can download. Tardis publishes its historical CSV datasets for the first day of every month without an API key, which is enough for Binance spot and the Binance USD-M perpetual, top of book and trades, on a receive clock:

```console
$ pip install 'blinded[demo]'
$ blinded demo tardis --date 2026-10-01 --symbols XRPUSDT --out ~/data/tardis --hours 14
```

This downloads four files into `~/data/tardis/raw/` with the `tardis-dev` client (the dataset endpoint sits behind Cloudflare and refuses a bare `curl`), builds one parquet beside them and writes a brief that points at it. Keep the directory outside every repository: the files are public, but the terms on redistributing them are not clear, and `blinded init` refuses a run directory inside a work tree anyway.

## What is downloaded

For Binance spot and the perpetual, Tardis lists `trades`, `incremental_book_L2`, `quotes`, `book_snapshot_5`, `book_snapshot_25` and `book_ticker` (the perpetual also has `derivative_ticker` and `liquidations`). Two of these are a top of book: `quotes` and `book_ticker`, with the same columns (`timestamp, local_timestamp, ask_amount, ask_price, bid_price, bid_amount`). They are not the same data. On 2026-10-01 (gaps between distinct stamps; the `book_ticker` gaps are over the first 14 hours):

| file | rows | median gap |
|---|---|---|
| spot `quotes` | 369,120 | 100.2 ms |
| spot `book_ticker` | 3,025,260 | 2.0 ms |
| perp `quotes` | 1,622,601 | 28 ms |
| perp `book_ticker` | 9,309,458 | 1.2 ms |

The spot `quotes` arrive on a 100 ms cadence, which looks like a book rebuilt from Binance's 100 ms depth stream. On that book the spot trades lead the spot quotes by 81 ms and the perp leads spot by 68 ms, both of which are artefacts of the sampling: a feature that reads trades to predict the next quote would be scoring the batching, not the market. `book_ticker` is Binance's own best-bid-and-offer stream, one row per change, and on it both lags are 0 to 1 ms. So the demo uses `book_ticker` and `trades`; `--book quotes` builds the coarser version if you want to see the difference.

`local_timestamp` (when Tardis received the message, in microseconds) becomes `ts`. The exchange's own `timestamp` is dropped, so the brief says `clock: receive` and every delay is 0.

## The layout

The parquet's columns are `ts`, `symbol`, `side`, `price`, `qty`, `bid0_px`, `ask0_px`, `bid0_qty`, `ask0_qty`, `mid` and `is_trade`, one row per event. Spot keeps its symbol (`XRPUSDT`) and the perpetual gets `_PERP`. A book row has the four touch columns and their mid, with `side`, `price` and `qty` empty. A trade row has `side` as `B` or `S`, its price and quantity, and the book of its own instrument as received at or before it (a backward as-of join on `ts`, ties included), so trade rows carry a book just as they do in the XRP file. Book rows with an empty side or a bid at or above the ask are dropped (none on this day), and so are trades that arrive before the first book row of their instrument.

Rows are sorted by `ts` with a stable sort. Inside an instrument the rows keep the order of their files, book before trades, so rows that share a receive time resolve as the files list them; there are many such ties (463,000 duplicate book stamps on spot in the first 14 hours), and the panel resolves them last-wins.

The brief matches `examples/demo.brief.yaml`: target `XRPUSDT` mid, horizons 100 ms, 1 s, 10 s and 60 s with 10 s primary, a 100 ms grid with 5 minutes of warm-up, 50 ms latency, 7.5 bps taker fee, fractions 0.50/0.25/0.25 with a 5 minute embargo, 5 minute bootstrap blocks.

## Why 14 hours

A longer day gives validation more blocks, and the Ladder's step is the bootstrap standard error of a validation score, so more blocks mean a finer step. The limit is the judge's: each evaluation runs in a child process that the daemon kills past 120 s or 4 GB. The window was chosen as the longest span from midnight for which one judgement of the reference basis feature (pre-check, truncation test on train, train metrics, truncation test on the warm panel, the score) stays under 60 s and 3 GB, measured the way the daemon measures it, with `ps` on the child every 0.1 s, on the test segment, whose warm panel is the whole file:

| hours | wall per judgement | peak RSS (repeats) |
|---|---|---|
| 24 | 3.3 to 3.4 s | 3.33, 3.78 GB |
| 18 | 2.5 to 3.0 s | 3.34, 3.43 GB |
| 16 | 1.9 to 2.1 s | 2.62, 3.08, 2.84 GB |
| 15 | 1.7 to 1.8 s | 2.73, 2.97, 3.10, 2.92 GB |
| 14 | 1.5 to 1.9 s | 2.45, 2.43, 2.49, 2.85, 2.48, 2.37 GB |

The whole day is 14,967,881 rows; the first 14 hours are 7,835,956. Time is nowhere near binding; memory is, and it is noisy from run to run by a few hundred MB, which is why 15 hours, under 3 GB in three runs of four, is not the choice. On 14 hours the splits are train 00:00 to 07:00, validation 07:05 to 10:30 and test 10:35 to 14:00, so validation has 41 five-minute blocks against XRP's 17. For the reference basis feature the bootstrap standard error on validation is 0.018, which the Ladder raises to a step of 0.02 rank IC; on XRP the same feature's step was 0.05. Its train rank IC at 10 s is 0.204 (Newey-West SE 0.013).

## What DATA.md says

`blinded data --brief binance-xrpusdt-2026-10-01-14h.brief.yaml` over the 14 hours: the perp book updates 73 times a second and spot 21, and the spot book is older than one grid step at 56% of grid points against 14% for the perp. The cross-stream lags are all 0 or 1 ms with clear peaks, so on this receive clock the perp does not visibly lead spot the way it does in the XRP file. Spot trades match the touch best when the book is read 1 ms before the trade's stamp (87% against 51% at 0 ms): the book update a trade causes tends to arrive a millisecond ahead of the trade message.

## Several days of perpetuals: `blinded demo binance`

The Tardis day has a few fills per strategy, too few for the locked test to tell anything from zero. Binance publishes its own archive of USD-M perpetuals at `data.binance.vision`, one zip per symbol and day with a `.CHECKSUM` beside it, and that is enough to build several contiguous days:

```console
$ blinded demo binance --from 2024-03-27 --days 3 --out ~/data/binance
```

This fetches `bookTicker` (every change of the best bid and offer) and `trades` for BTCUSDT, ETHUSDT, SOLUSDT and XRPUSDT, 24 zips and 1.6 GB, with the standard library's `urllib`, and checks each against its SHA-256. The archive's `bookTicker` stops after 2024-03-30, so these days cannot be moved much later. Everything is written under `--out`: `raw/` (deleted at the end unless `--keep-raw`), the parquet, its brief and `DATA-raw.md`. Keep it outside every repository, as with the Tardis day.

### The clock

Both files are stamped by the matching engine: `transaction_time` in the book and `time` in the trades, in ms. The book's `event_time`, when the message left, trails it by a median 6 to 7 ms and is not used; stamping the book with it would put the book 6 ms behind the trades it reflects and invent a trades-lead-book effect. So the brief says `clock: exchange`, and the delay a live feed adds has to come from the brief: `default_delay_ms: 50` with `latency_ms: 50`. A decision at $g$ sees stamps up to $g - 50$ ms, and its order fills at stamp $g + 50$ ms, 100 ms from information to fill ({doc}`sim`).

### The checks, per symbol-day on the raw rows

The build stops with the reason if any of these fail:

1. Each zip matches its `.CHECKSUM`, and its CSV header is the archive's layout exactly (a changed header fails with the header printed).
2. Stamps never go backwards in file order.
3. Neither stream is batched. A stream rebuilt on a cadence piles its gaps into one bin: the Tardis spot `quotes` had a median gap of 100.2 ms. The book must have a median gap between distinct stamps under 50 ms, and neither the book nor the trades may hold more than 20% of their gaps in one 10 ms bin from 20 to 1000 ms. Trades are sparse, so their median says nothing and only the bin test applies.
4. The streams share a clock: on up to 200,000 trades spread over the day, trades print at the touch most often when the book is read within 5 ms of their stamp ({func}`blinded.diagnostics.touch_alignment`), and the mid and the last trade price move together within 5 ms ({func}`blinded.diagnostics.xcorr_lag`).

On 27 to 29 March 2024 every one passed: book median gaps of 2 to 4 ms, the largest 10 ms bin 4 to 6% of gaps (at 20 ms, the start of the range), the touch best at 0 ms for all twelve symbol-days, and mid against trade price at 0 or -1 ms.

### Thinning: keep the states the judge reads

A raw day of the four is about 42 million book rows and 14 million trades. The Tardis file costs 310 to 370 bytes per row in a judgement, so one raw day would need well over the daemon's 4 GB. But the judge reads the book only at a few stamps around each grid point, so the builder keeps exactly those rows and drops the rest. The stamps come from the code, read by read, with $d = 50$ ms of feed delay and $L = 50$ ms of latency on a 500 ms grid:

| what reads the book | as-of `available_at` | stamp | kept |
|---|---|---|---|
| the panel at $g$, what every feature sees | $g$ | $g - 50$ | yes |
| fills and marks of the `brief`, `fees 0` and maker-proxy rows; the start of every forward return | $g + L + d$ | $g + 50$ | yes |
| fills and marks of the `latency 0` rows; the naive judge's targets | $g + d$ | $g$ | yes |
| the next mark, the end of a forward return at a whole number of steps, the maker proxy's mark at +10 s | the same plus whole steps | the same plus whole steps | yes |
| the report's markouts at whole steps, at 0, at the fill (100 ms) | $g + \tau$ | $g - 50 + \tau$ | yes |
| lead-lag $C(\tau)$ at whole steps and at the first step after the fill (600 ms) | $g + \tau$ | $g - 50 + \tau$ | yes |
| decay horizons of 100, 200 and 250 ms | $g + L + d + h$ | $g + 150$, $g + 250$, $g + 300$ | no: left out |
| markout offsets of $-100$ and 250 ms | $g + \tau$ | $g - 150$, $g + 200$ | no: left out |
| lead-lag at 10 ms steps | $g + \tau$ | | no: measured on the raw books instead |
| `blinded eval --latency-ms` at another latency | $g + L' + d$ | $g + L'$ | only at a kept stamp; else a warning |

So for every grid point the file keeps the last book row at or before $g - 50$, $g$ and $g + 50$ ms ({func}`blinded.demo.read_points`, {func}`blinded.demo.thin_book`), each with `n`, the raw rows it stands for, so `cum_book` counts them all. The brief records the kept stamps as `data.thinned: {step: 500ms, offsets_ms: [-50, 0, 50]}`, and a brief whose grid, latency, delays or horizons would read anything else is refused when it is loaded. The judge's decay table and the report's charts leave out the horizons and offsets in the "no" rows, so every number they show is exact; DATA.md says so. The truncation test's cuts fall 0.37 of a step after a grid point, 185 ms, which is after $g + 50$ and before the next $g - 50$, so a forward or nearest join still has a later row to take.

Trades are summed per side between the panel's reads ({func}`blinded.demo.thin_trades`): one row per side and step at the stamp of that side's last trade, `qty` the side's total, `price` its volume-weighted price, `n` the number of trades, and a trade stamped on a read itself in a row of its own. `cum_qty`, `cum_signed_qty`, `cum_buy_qty`, `cum_sell_qty`, `cum_notional` and `cum_trades` are then exact at every grid point, the side of the last trade is too, and the maker proxy's trade rate inside a window that starts and ends on grid points counts the raw trades. What changes is the last trade's `price` and `qty` in the panel: the step's per-side VWAP and total rather than one print, so a feature on the last print moves by up to the step's range.

A read just after midnight returns the previous day's last row when nothing has printed since, so a symbol's last raw row is carried into the next day whenever no read kept it, with the count of the rows it stands for; and on the last day the very last row is kept, so the file ends where the raw one does and `blinded init` builds the same grid. The brief pins `grid.start` to the first midnight for the same reason.

The builder checks the identity on real data. On the first hour of the first symbol-day it builds a panel from the raw rows and one from the kept rows on the same grid and compares them column by column ({func}`blinded.demo.identity_check`): every column but the last trade's price and size equal, the summed volumes and notionals equal to a relative $10^{-9}$ (the same trades added in another order), and the bid and ask equal at every fill read, at the brief's latency and at zero. On BTCUSDT on 27 March: 695,257 raw rows against 30,135 kept, 7,199 grid points, 15 columns, 28,796 fill reads, all equal. `tests/test_demo.py` checks the same on ten synthetic minutes, and there also that `forward_returns`, the sim's net P&L and the maker proxy agree to the bit.

### The file

| symbol | raw book rows (27, 28, 29 March) | raw trades | kept book rows | kept trade rows |
|---|---|---|---|---|
| BTCUSDT | 15.8M, 13.6M, 10.3M | 5.5M, 3.7M, 2.9M | 454k, 444k, 432k | 284k, 275k, 253k |
| ETHUSDT | 11.6M, 9.7M, 8.0M | 4.6M, 3.4M, 2.9M | 452k, 441k, 433k | 238k, 226k, 217k |
| SOLUSDT | 8.2M, 7.2M, 6.5M | 2.6M, 2.1M, 1.9M | 392k, 385k, 381k | 194k, 188k, 190k |
| XRPUSDT | 5.9M, 6.2M, 5.9M | 1.2M, 1.2M, 1.4M | 403k, 408k, 405k | 135k, 145k, 150k |

`binance-um-btc-eth-sol-xrp-2024-03-27-3d-500ms.parquet` has 7,526,342 rows in 113 MB, about the size of the Tardis 14-hour file, from 142 million raw rows; the build took 2 min 43 s and peaked at 5.4 GB, most of it parsing BTC's 1.4 GB of book CSV for one day. The file has 2.4 book rows and 1.2 trade rows per symbol per step: two kept stamps 50 ms apart often return the same book row, and a quiet half-second has one side's trades or none. The builder estimates the rows from those rates before it downloads anything, prints the estimate against an 8.5 million guideline, and refuses above 10 million unless `--force`: `--days 4` estimates 10.0 million and is refused.

The brief: the three families of {doc}`families` (directional on all four, BTC and ETH leading SOL and XRP, ETH over BTC and XRP over SOL), shares 0.3, 0.3 and 0.4, a 500 ms grid from the first midnight, horizons 500 ms, 1 s, 10 s and 60 s with 10 s primary, 7.5 bps a side, fractions 0.5, 0.25, 0.25 and a 5 minute embargo. Train is 27 March and the morning of the 28th, validation 12:05 on the 28th to 06:00 on the 29th, test from 06:05 to midnight: 215 five-minute blocks each for validation and test, against 41 on the Tardis day.

### One judgement: memory and time

Measured as the Tardis window was: the job `score_test` sends, the reference pairs strategy (`bl.pairs` on ETH over BTC, 2 h fit, 30 min refit, z 2 in and 0.5 out) judged on the test segment, whose warm panel is the whole file, in a spawned child whose resident memory is read with `ps` every 0.1 s:

| repeat | wall | peak RSS |
|---|---|---|
| 1 | 32.4 s | 2.88 GB |
| 2 | 31.5 s | 2.85 GB |
| 3 | 32.4 s | 3.07 GB |
| 4 | 31.3 s | 3.15 GB |
| 5 | 31.6 s | 2.73 GB |
| 6 | 32.0 s | 2.96 GB |

All six are under the 3.3 GB and 60 s set for three days, so three days stand and the two-day fallback is not needed. Memory is 390 to 450 bytes per row of the file (GB here is $2^{30}$ bytes, as in the daemon's limit), against 325 to 390 for the Tardis file measured the same way, with twice the instruments and a panel of 65 columns; the headroom to the daemon's 4 GB is 0.85 GB at the worst repeat, so a feature that holds many wide intermediates can still be stopped as `oom`. Time is about 32 s against a first guess of 2 s: the pairs fit runs on up to 518,400 grid points for each of about a dozen truncation cuts and full runs over the two panels, and it stays well inside the 120 s per evaluation. The strategy loses on test, -43 bps per hour net at a Sharpe of -1.46 per square-root hour. The script is `scripts/measure_judgement.py`.

### DATA-raw.md

The builder writes the numbers of every check per symbol-day, the SHA-256 of each zip, the identity and, for the first day, the lead-lag curve of every leader and follower at 10 ms steps over $\pm 500$ ms on the raw books: 100 ms returns on a 100 ms grid, which the half-second grid cannot measure. On 27 March BTC leads SOL and XRP by about 10 ms (a peak correlation of 0.34 and 0.26), and the follower's first whole 100 ms step after a 100 ms fill keeps 0.06 to 0.08 of it, so a taker cannot use it.

`blinded init` appends `DATA-raw.md`, from beside the data file, to the workspace's DATA.md, one heading level down. A protected run gets the header and only the days that end by the end of train, here the 27th, so nothing it reads was measured on validation or test; a naive run, whose workspace holds all the data anyway, gets every day.

## Four perpetuals on hourly bars: `blinded demo trend`

Trend following holds for weeks, and twenty of its shortest holdings, two weeks each, need 40 weeks of validation. Top-of-book history that long does not exist for free, so this file is klines, hourly bars of four USD-M perpetuals, with the bid and ask made from a half spread measured on one day of real books before train:

```console
$ blinded demo trend --out ~/data/binance/trend4 --raw ~/data/binance/raw --costs ~/data/binance/costs
```

### What is downloaded

From `data.binance.vision`: the monthly 1h klines of BTCUSDT, ETHUSDT, XRPUSDT and LTCUSDT from 2023-05 to 2026-09, 164 zips of about 37 KB, and the futures `bookTicker` of 2023-05-16 for the four, 54, 47, 17 and 15 MB. The archive has no futures `bookTicker` before 2023-05-16, which is why the window starts the day after: the day the half spread is measured on must lie before train. The builder sends an HTTP HEAD for all 168 files first and stops, naming every missing file, before a byte is fetched ({func}`~blinded.demo.head_sizes`); it prints what it is about to fetch against a 10 GB cap; every zip is checked against its `.CHECKSUM`; and the zips are deleted after the build unless `--keep-raw`. The half spreads are cached as `costs/bookTicker-2023-05-16.json`, so a rebuild fetches only the klines, 6 MB. The first build fetched 139.7 MB in about four minutes and peaked at 1.5 GB, reading BTC's day of 4.5 million book rows.

### The layout

Each bar becomes three rows at its close stamp, `open_time + 1 h - 1 ms` ({func}`~blinded.demo.bar_rows`), with a `kind` column the streams filter on (`kind == 'book'`, `kind == 'trades'`):

- a book row: `mid` the close, `bid0_px` and `ask0_px` the close times $1 \mp hs/10^4$, no sizes, so `order_book_imbalance` is refused ("needs bid_qty; the book has bid, ask") and nothing pretends to a queue;
- a sell row and a buy row: `qty` the volume less the taker buys, and the taker buys; `price` the bar's VWAP, `quote_volume / volume`, or the close when nothing traded; `n` the bar's trade count split by volume, `round(count x taker_buy_volume / volume)` buys and the rest sells. The panel's cumulative volumes and notionals are exact at every grid point; `n` is an estimate, since the archive gives the count and not its split.

The reader handles two archive quirks, a file with no header row (its first field parses as an integer) and stamps in microseconds (above $10^{14}$), and fails on any other header ({func}`~blinded.demo.read_klines`). None of the 164 futures files had either. A bar that opens off the hour, a duplicate, a close time that is not the open plus an hour less 1 ms, or more than 0.1% of a symbol-month's bars missing fails the build with the symbol and month named ({func}`~blinded.demo.bar_gaps`, {func}`~blinded.demo.missing_check`). None was missing, and no bar had zero volume.

### The fill model

| instrument | median half spread on 2023-05-16, bps | in the brief |
|---|---|---|
| BTCUSDT_PERP | 0.0185086 (one tick of 0.1 on about 27,000) | 0.01851 |
| ETHUSDT_PERP | 0.0274901 | 0.02749 |
| XRPUSDT_PERP | 1.18469 | 1.185 |
| LTCUSDT_PERP | 0.55919 | 0.5592 |

The median is over every two-sided `bookTicker` row of the day of $(ask - bid)/(2\,\text{mid})$ ({func}`~blinded.demo.half_spreads`); the brief keeps four figures, and the file's bid and ask are built from exactly those. The brief's `data.cost_model` block records the kind, the fill (the next bar's close), the source day and the four numbers; `blinded init` checks every book row against them ({doc}`sim`). DATA.md, RULES.md, ARCHETYPES.md and the report's verdict all carry the line "fills are a model: the next bar's close plus or minus a half spread of 0.01851 to 1.185 bps per symbol, measured on data.binance.vision bookTicker 2023-05-16, before train; no book existed at the fill".

### The file and the brief

`binance-um-trend4-2023-05-17-1233d-1h.parquet` has 355,104 rows in 5.2 MB: 29,592 hourly bars a symbol from 2023-05-17 00:00 to the bar opening at 23:00 on 2026-09-30, three rows each. A rebuild with the costs cached writes the parquet and DATA-raw.md byte for byte. The brief: a 1 h grid from 2023-05-17 00:00, latency one bar (3,600,000 ms), no feed delay, horizons 1 day, 1 week and 2 weeks with 1 week primary, a 4-week warm-up, 5 bps a side (Binance VIP0 taker), fractions 0.5, 0.25, 0.25 with a 3-week embargo, 2-week blocks, and an `archetypes` block over the four with `trend_following` at a 4-week lookback refitted weekly and `pairs_trading` at a 1-week fit refitted daily. The builder computes the windows `blinded init` will fix and stops before writing the brief if train or validation is under 40 weeks:

| segment | from | to | scored span |
|---|---|---|---|
| train | 2023-05-17 00:00 | 2025-01-22 12:00 | 588 d 12 h after the warm-up |
| validation | 2025-02-12 12:00 | 2025-11-26 17:00 | 287 d 5 h, 20 whole two-week blocks |
| test | 2025-12-17 17:00 | 2026-10-01 00:00 | 287 d 7 h |

Validation clears the 40 weeks by 7 d 5 h. Four archetypes run, in three edges ({doc}`archetypes`). The power line: $1/\sqrt{6893}$ is 0.012, which the 1-2-5 grid takes to a step of 0.02 Sharpe per square-root hour, 1.9 annualised, so a trend strategy releases only if its validation Sharpe beats that, above the archetype's realistic range; nothing released is the likely outcome, and the menu says so before the run.

DATA.md on a bar file leaves out the millisecond tables, cross-stream lag and trade stamps against the book, with one line saying why, and adds a section on the fills. `DATA-raw.md` beside the file has the cost day as a `## 2023-05-16` section and one `## YYYY-MM` section per month with each symbol's bars expected, present and missing, the bars with no volume, any file quirk and each zip's SHA-256. `blinded init` appends it to a protected run's DATA.md without the months that end after train ({func}`~blinded.demo.notes_until`), so validation and test are not described there.

### One judgement

The reference pairs strategy, here `bl.pairs_trading` on ETH over BTC at the brief's 1-week fit and daily refit, judged six times on the test segment as `score_test` does: 1.2 to 1.3 s and 0.36 to 0.38 GB at the peak each time, against the 60 s and 3.3 GB the files are held to. The file is small: 355,104 rows at about 1,100 bytes a row of peak, most of it the process itself.

## The XRP perpetual on two venues: `blinded demo crossvenue`

Cross-venue arbitrage needs one product on two venues. The Tardis free day has the Binance perpetual already; this builder adds a second venue's perpetual from the same day and builds both into one file on the receive clock:

```console
$ blinded demo crossvenue --out ~/data/tardis/crossvenue --raw ~/data/tardis/raw
```

### The second venue

Before a byte is fetched, the builder asks Tardis for the exchange's details (`tardis_dev.get_exchange_details`) and checks that the symbol is a listed perpetual with `quotes` and `trades` covering the day, and that the day is a free first of the month ({func}`~blinded.demo.venue_problem`). It tries, in order, Bybit `XRPUSDT` on 2026-10-01, OKX `XRP-USDT-SWAP` on the same day, and Bybit on 2026-09-01 (fetching Binance's files for that day too), prints each one it passes over with its reason, and stops when none is left. Bybit comes first because its linear perpetual is the most liquid XRP perpetual outside Binance. On 2026-10-01 Bybit lists XRPUSDT from 2021-05-13 with `quotes`, `trades` and `book_ticker` among its types, so it is the venue; its `quotes` and `trades` for the day are 13.6 and 16.4 MB of gz, fetched with `tardis-dev` and kept in `raw/` beside Binance's, which were there already. Tardis records both venues in Tokyo (GCP asia-northeast1).

Tardis derives Bybit's `quotes` from its depth-50 book stream, which Bybit pushes every 20 ms; its `book_ticker` comes from the depth-1 stream at 10 ms. The builder uses `quotes`, at about 1 million rows a day.

### The layout and the hours

The rows are the Tardis layout of {func}`~blinded.demo.build` ({func}`~blinded.demo.venue_frame`): Binance's `book_ticker` and `trades` as `XRPUSDT_PERP`, the name the Tardis runs use, and Bybit's `quotes` and `trades` as `XRPUSDT_PERP_BYBIT`, each trade carrying its own venue's book as received. Spot is left out, to keep the file inside the row budget. The file keeps the largest whole number of hours from midnight, at most 14, whose rows fit in 7.8 million, the rows of the 14-hour Tardis file whose judgement was measured at 2.4 to 2.9 GB: here all 14 fit, 6,871,453 rows (Binance 4,914,738 book and 1,075,659 trade rows, Bybit 567,437 and 313,619). The brief is the Tardis brief on the perpetual (100 ms grid, 50 ms latency, 7.5 bps on both venues, horizons 100 ms to 60 s with 10 s primary, 5-minute blocks and embargo) with

```yaml
archetypes:
  instruments: [XRPUSDT_PERP, XRPUSDT_PERP_BYBIT]
  cross_venue: [[XRPUSDT_PERP, XRPUSDT_PERP_BYBIT]]
```

and the windows are the Tardis day's: train to 07:00, validation 07:05 to 10:30, test 10:35 to 14:00. Five archetypes run, in two edges, speed and liquidity ({doc}`archetypes`).

### The second venue's cadence

A book rebuilt on a cadence manufactures a lead, as the spot `quotes` above do, so the builder runs the gap check on Bybit's receive stamps and refuses a median gap between distinct stamps at or above 50 ms ({func}`~blinded.demo.venue_cadence`). Below that the venue is accepted and its cadence printed, since a lead shorter than its schedule cannot be told from the schedule. The 20% bin rule of the Binance builder does not apply to it: a 20 ms push piles the gaps into one bin by design. Over the 14 hours the median gap is 40.2 ms and the largest 10 ms bin, 20 to 30 ms, holds 21% of the gaps: the 20 ms push, with the top of book unchanged at many pushes. The median is a rate of change of the top as much as a cadence. On the night hours of train it is 59.9 ms (the same bin, 18%), and on validation 39.8 ms; DATA.md prints the train figure as the "venue feed cadence" line, and D11 of the report prints it per segment. DATA-raw.md beside the file has the files' sizes and SHA-256 (Tardis has no checksums) and, in a `## 2026-10-01` section that a protected run's DATA.md leaves out, the rows at each hour cut and the whole-file cadence.

### One judgement

`bl.cross_venue_arbitrage` with `bl.pairs` on Binance over Bybit at a 10-minute fit, judged six times on the test segment: 10.0 to 12.9 s and 1.91 to 2.15 GB at the peak, against 60 s and 3.3 GB. The build itself took 4 s and 2.5 GB.

## Twenty perpetuals on 15-minute bars: `blinded demo crosssection`

Statistical arbitrage and short-term reversal trade a cross-section, and the menu runs them from twenty names. This file is the trend file's layout ({func}`~blinded.demo.bars`, shared with `blinded demo trend`) on twenty USD-M perpetuals and 15-minute bars over 450 days:

```console
$ blinded demo crosssection --out ~/data/binance/xs20 --raw ~/data/binance/raw --costs ~/data/binance/costs
```

### What is downloaded

BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, DOTUSDT, ARBUSDT, LTCUSDT, TRXUSDT, ATOMUSDT, UNIUSDT, NEARUSDT, APTUSDT, FILUSDT, ETCUSDT and BCHUSDT: liquid perpetuals listed before 2024 and alive through June 2025 (MATICUSDT is left out, renamed POLUSDT in September 2024, and ARBUSDT takes its place). From `data.binance.vision`: their monthly 15m klines from 2024-04 to 2025-06, 300 zips of about 0.13 MB, and the futures `bookTicker` of 2024-03-27, five days before train, for the twenty, 1,252 MB (BTC 178, DOGE 147, ETH 145). The builder sent 320 HEAD requests before fetching any, fetched 1,289.7 MB in 643 s with every `.CHECKSUM` matching, and deleted the zips. The half spreads went to `costs/bookTicker-2024-03-27.json`. Only the two price columns of each `bookTicker` day are parsed ({func}`~blinded.demo.touch_prices`); BTC's day has 15.8 million rows, and the build peaked at 4.3 GB.

### The file and the brief

`binance-um-xs20-2024-04-01-450d-15m.parquet` has 2,592,000 rows in 32.5 MB: 43,200 bars a symbol from 2024-04-01 00:00 to the bar opening at 23:45 on 2025-06-24, three rows each. No bar was missing and no file was headerless or in microseconds; every symbol has four bars with no volume, 20:00 to 21:00 on 2024-10-28, the same hour on all twenty, inside train. The half spreads on 2024-03-27 run from 0.007166 bps (BTC) to 0.8076 bps (XRP), and the brief keeps them to four figures. The brief: a 15-minute grid from 2024-04-01 00:00, latency one bar, horizons 15 min, 1 h, 4 h and 1 day with 4 h primary, a 1-day warm-up, 5 bps a side, fractions 0.5, 0.25, 0.25 with a 2-day embargo, 4-hour blocks, and an `archetypes` block over the twenty with these parameters for the two baskets (`top: 4, every: 4h`, lookbacks 1 day and 4 h) and pairs trading (`window: 2d, every: 4h`). The target is the last name, BCH, as the other builders' targets are.

| segment | from | to | scored span |
|---|---|---|---|
| train | 2024-04-01 00:00 | 2024-11-12 00:00 | 32 weeks after the warm-up |
| validation | 2024-11-14 00:00 | 2025-03-04 11:45 | 110 d 11 h 45 min |
| test | 2025-03-06 11:45 | 2025-06-25 00:00 | 110 d 12 h 15 min |

The builder stops before writing the brief if train or validation is under 80 hours, 20 holding periods of short-term reversal's 4 hours. Five archetypes run, in three edges. The power line: $1/\sqrt{2651.75} = 0.0194$, within a tenth of 0.02, so a strategy releases above 0.02 to 0.05 Sharpe per square-root hour (1.9 to 4.7 annualised), depending on its own bootstrap. DATA.md's pairs table lists the 10 of the 190 implied pairs with the shortest median half-life on train; its lead-lag table has all 380 leader-follower combinations.

### One judgement

The heaviest judgement on this file is a basket: `bl.short_term_reversal` on minus each name's four-hour return, z-scored across the twenty, through `bl.basket(top=4, every="4h")`, with the feature computed inside the judgement. Six repeats on the test segment: 4.1 to 4.2 s and 1.17 to 1.38 GB at the peak. `blinded init` on the file takes 18 s and 1.3 GB, most of it DATA.md's tables.

## Eight coins' spot and perpetual with funding: `blinded demo carry`

The basis trade holds spot against its perpetual and collects the funding rate, so its file needs both legs of each coin and every funding settlement. This file is the cross-section file's layout on eight coins, spot and USD-M perpetual, over the same 450 days and 15-minute bars, with a third stream ({func}`~blinded.demo.carry`):

```console
$ blinded demo carry --out ~/data/binance/carry8 --raw ~/data/binance/raw --costs ~/data/binance/costs --tardis-raw ~/data/tardis/raw
```

### What is downloaded

BTC, ETH, BNB, SOL, XRP, DOGE, ADA and LINK, all among the cross-section's twenty, each as spot `XUSDT` and perpetual `XUSDT_PERP`. From `data.binance.vision`: monthly 15m klines from `spot/monthly/klines` and `futures/um/monthly/klines`, and the perpetuals' monthly `fundingRate`, 2024-04 to 2025-06, 360 zips and 32.9 MB, each checked by a HEAD request before any is fetched and against its `.CHECKSUM` after. Spot and futures kline zips have the same names, so each market is fetched into its own folder, and the build stops if a spot and a perpetual month are one file. The spot half spreads come from Tardis: `binance` `book_ticker` on 2024-03-01, a free first-of-month day a month before train. Before fetching, the builder reads `tardis_dev.get_exchange_details("binance")` and checks that each of the eight is listed as spot with `book_ticker` covering the day ({func}`~blinded.demo.spot_book_problem`), and would fall back to 2024-02-01, printing why. The eight gz files are 285.1 MB (BTC 72.6, SOL 52.5, DOGE 51.1, ETH 49.2), kept beside the other Tardis files with their sizes and SHA-256 in `DATA-raw.md`; their half spreads go to `costs/tardis-binance-book_ticker-2024-03-01.json` ({func}`~blinded.demo.tardis_half_spreads`). The perpetuals' half spreads are the cross-section's table of 2024-03-27, read from its cache. The Binance zips are deleted after the build.

### Funding settlements

`fundingRate` has one row per settlement, `calc_time,funding_interval_hours,last_funding_rate`, and `calc_time` lies a few ms after the settlement it records: 0 to 16 ms on all eight. Each settlement becomes one row with `kind == 'funding'` and `rate`, stamped on the 15-minute boundary nearest its `calc_time` ({func}`~blinded.demo.funding_rows`), so every settlement lands on a grid point and the sim books it to the position held there. The brief makes the funding stream available 1 s after its stamp (`availability.overrides`), so no decision sees a rate before its `calc_time`: the 00:00 rate is first read at the 00:15 decision. A `calc_time` a second or more from every boundary, or two settlements on one boundary, stops the build. Each perpetual has 1,350 settlements, at 00:00, 08:00 and 16:00 UTC, all 8-hour intervals.

### The file and the brief

`binance-carry8-2024-04-01-450d-15m.parquet` has 2,111,684 rows in 27.4 MB: 43,200 bars for each of the 16 instruments, three rows each, 10,800 settlements and 27,284 borrow rates. No bar is missing. The perpetuals have the cross-section's four empty bars of 2024-10-28; the spot bars have none. Every spot kline file is headerless, and those from 2025-01 stamp in microseconds; no futures file is either.

| | spot, Tardis 2024-03-01 | perpetual, `bookTicker` 2024-03-27 |
|---|---|---|
| BTC | 0.000809 (one tick of 0.01) | 0.007166 |
| ETH | 0.01468 | 0.01406 |
| BNB | 1.232 (one tick of 0.1) | 0.08702 |
| SOL | 0.3762 | 0.02701 |
| XRP | 0.8444 | 0.8076 |
| DOGE | 0.4103 | 0.2704 |
| ADA | 0.7337 | 0.7692 |
| LINK | 0.2529 | 0.257 |

Half spreads in bps, as the brief keeps them. The brief's `cost_model.source` names both days; the later one, 2024-03-27, is the one `blinded init` checks against train. A one-tick spread on a large price is close to the rounding of doubles, so the check allows 1e-10 bps on top of its 1e-9 relative tolerance ({func}`~blinded.vault.check_cost_model`).

The brief: the cross-section's grid, latency, split and 2-day embargo, horizons 1 h, 8 h and 1 day with 8 h primary, 8-hour blocks, spot fees of 10 bps and perpetual fees of 5 (Binance VIP0 takers, `execution.fees`), the `funding` and `borrow` streams, and an `archetypes` block over the sixteen with the eight `carry` pairs, `basis_trade: {rate_in: 0.0001, rate_out: 0.0}` and pairs trading at `window: 2d, every: 4h`. The windows are the cross-section's: train to 2024-11-12, validation 2024-11-14 00:00 to 2025-03-04 11:45 (110 d 11 h 45 min), test from 2025-03-06 11:45. The builder stops before writing the brief if train or validation is under 160 hours, 20 holding periods of the basis trade's 8 hours. Four archetypes run, in three edges: lead-lag, trade-flow persistence, pairs trading on the cross-coin pairs, and the basis trade. `ARCHETYPES.md` adds to the basis trade the mean funding rate per settlement on train (672 settlements a perpetual after the warm-up): from 0.614 bps (BTC) to 0.743 bps (XRP) on seven, about 7 to 8% a year, and -0.404 bps on BNB, where shorts paid longs; and a round trip in both legs, 2 x (10 + 5) = 30 bps of fees before any spread or basis move.

A short spot leg pays to borrow the coin. Each coin's VIP 0 daily rate comes from Binance's public margin interest history, which has a row at 00:00 UTC every day and one at each change ({func}`~blinded.demo.borrow_history`, cached in `costs/borrow-2024-04-01-2025-06-25.json`), and each published rate is a `kind == 'borrow'` row of the spot instrument, from the rate in force at the start ({func}`~blinded.demo.borrow_rows`); the brief maps `rate` to `borrow_rate` and makes the stream available 1 s after each stamp. The build stops if a coin goes more than 2 days without a rate. The eight coins have 27,284 rows, from 497 (BNB, which changed least) to 5,167 (LINK), with gaps of at most 24 hours. `ARCHETYPES.md` gives each spot leg's mean and highest rate in force on train, in bps a day: BTC 0.34 (0.408 at most), ETH 0.529 (0.945), BNB 18.2 (41.1), SOL 1.23 (1.8), XRP 1.7 (2.21), DOGE 0.976 (1.2), ADA 1.28 (1.58), LINK 0.466 (0.589).

BNB's borrow rate stood at 25 to 41 bps a day in April 2024 and between 11 and 22 for most of the rest, about 40 to 150% a year. BNB was also the only coin whose perpetual paid shorts on train, -0.404 bps a settlement, so `bl.carry` at the brief's parameters held the reverse trade on it, short spot and long the perpetual, for 97 days of train and collected 495 bps of funding. Without borrow that netted +353 bps; the borrow is 2,088 bps, and it nets -1,735. The same rule on ETH, BTC and SOL was short spot for 1 to 3 days and paid 0.3 to 1.1 bps.

### One judgement

`bl.basis_trade` on BTC: `bl.carry` at the brief's parameters, requiring a feature on the perpetual's last settled rate, so the judgement reads the funding stream, scores both legs and accrues funding. Six repeats on the test segment: 0.7 to 0.8 s and 1.00 to 1.14 GB at the peak. `blinded init` on the file takes 13 s and 1.1 GB.

## A perpetual's day for the fill engine: `blinded demo mm`

```console
$ blinded demo mm --symbol SOLUSDT --date 2026-10-01 --out data/tardis/mm/solusdt --raw data/tardis/raw
```

The fill engine needs the top of book at every change with its sizes and every trade at its own price and stamp; the panel needs only the book at the stamps it reads. A day of the SOL perpetual's top of book is 11.3 million rows, so one file holding both would cost a judgement most of its memory for rows only the engine reads. `blinded demo mm` ({func}`~blinded.demo_mm.mm`) writes them apart, from the Tardis free day of a Binance USD-M perpetual: `book_ticker`, `trades`, `book_snapshot_5` and `derivative_ticker`, verified against `get_exchange_details` before anything is fetched.

- The data file: the book rows a read at a grid point or at a grid point plus the latency returns (a 100 ms grid from midnight read at 0 and 50 ms, thinned as `blinded demo binance` thins), each with `n` and levels 1 to 4 of the snapshot as-of its stamp (`bid1_px` .. `ask4_qty`, so `F.depth_imbalance` has them); every trade at its own stamp with the touch as received; and the day's funding settlements. A settlement's rate is the `funding_rate` of the last `derivative_ticker` row that names it in `funding_timestamp`: Binance's mark-price stream shows the next settlement's rate and moves to the one after a moment past it, so that row carries the settled rate. It must arrive within a second of the settlement, the delay the brief gives the stream; on SOL the three arrived 322, 70 and 995 ms after it, on LTC 66 and 565 ms after and 206 ms before. Rows have `kind` `book`, `trades` or `funding`.
- The touch file: every two-sided `book_ticker` row, `ts`, `bid` and `ask` in ticks, `bid_qty`, `ask_qty`, declared in the brief as `data.touch`. Only the engine reads it ({doc}`run-directory`).
- The brief: 100 ms grid, `latency_ms: 50`, maker 2 bps and taker 5 (Binance VIP0), tick 0.01 and lot 0.01 (SOL) or 0.001 (LTC) from the exchange's `exchangeInfo`, `max_position: 2`, horizons 100 ms to 60 s with 10 s primary, 5-minute blocks, 12, 6 and 6 hours, and `market_making: {size: 1, skew: 0.25, max_inventory: 2}`. The skew is the largest power of two with `skew x (max_inventory + size) < 1` tick, so that at any inventory the cap allows, with one more order in flight, the shedding quote rests at or behind the touch: on a one-tick book a quote a tick inside is on the far side, and post-only rejects it at every decision while the cap withdraws the other side. A whole tick a unit did that after one fill (on the SOL test window, 211,457 rejections and 13 fills under RA); half a tick did it at an inventory of -2.47, the cap and a partial fill (90% of RA's orders on the SOL validation window were rejections).

The checks, each stopping the build: every price a whole tick; the panel from every raw book row and from the kept rows equal in every column over the first hour, and the book equal at every grid point and fill stamp ({func}`~blinded.demo_mm.identity`); the touch file equal to the data file's book at every one of its stamps ({func}`~blinded.vault.check_touch`, which `blinded init` repeats); and the snapshot the book of the touch stream, the touch as-of each snapshot row showing its level 0 prices on at least 90% of them ({func}`~blinded.demo_mm.snapshot_check`).

| 2026-10-01 | SOLUSDT_PERP | LTCUSDT_PERP |
|---|---|---|
| raw `book_ticker` rows; kept on the lattice | 11,281,918; 1,222,332 | 2,473,907; 539,167 |
| trades; funding rows; data rows | 1,953,688; 3; 3,176,023 | 335,530; 3; 874,700 |
| data file; touch file | 51.7 MB; 84.7 MB | 15.3 MB; 18.0 MB |
| snapshot rows; the touch as-of each shows its level 0 prices | 2,206,739; 99.8% | 861,180; 99.9% |
| falls of the best bid: traded out; pulled; partly | 85.7%; 0.1%; 14.1% | 42.9%; 35.1%; 22.1% |
| build | 8.3 s, 3.6 GB | 2.8 s |
| one judgement of the default quote on test, six repeats | 16.4 to 17.1 s, 2.31 to 2.94 GB | 7.3 to 7.4 s, 1.74 to 1.86 GB (two) |

`DATA-raw.md` holds the files' sizes and SHA-256, the checks, and a table of the day (on SOL 21,886 falls, 327,459 aggressor orders, a feed delay of 2.44 ms at the median; on LTC 12,399, 67,413 and 2.87); `DATA.md` prints the same table over the workspace's rows ({func}`~blinded.bookstats.touch_facts`).

## The arbitrage pair: `blinded demo arb`

`blinded demo arb --coin ATUSDT` ({func}`~blinded.demo_mm.arb`) builds a coin on a thin spot venue, the quote venue (MEXC), and a deep one, the hedge venue (Binance), from `book_ticker`, `trades` and `book_snapshot_25` of each. Nothing is thinned: the quote venue changes its touch about once a second, and the hedge leg reads the hedge venue's touch at fill stamps on no lattice. The data file keeps every book row of both venues with `age_ms`, how long after the exchange's stamp the recorder received it (0 where the venue sends no stamp, as Binance spot does not), the hedge venue's levels 1 to 4 as-of each of its rows, and every trade with its own venue's touch; the touch file is the quote venue's; the depth file (`data.depth`) the hedge venue's snapshot at every row, 25 levels a side. The brief adds `hedge_latency_ms`, `hedge_fee_bps: 10`, the quote venue's maker 0 and taker 8 bps, the pair under `archetypes.cross_venue`, and the stock block with `withdrawal_fee_coin` unset. The build refuses a day whose quote venue's spread, time-weighted median, is under three hedge fees, and a snapshot that is not its venue's book.

The brief's hedge fields are read by the arbitrage sim ({doc}`sim`): `execution.hedge_latency_ms` (the hedge's order latency on the hedge venue; with the quote venue's response latency it is how long after a fill its hedge arrives, which must be at least `data.depth.lag_ms`, how far the snapshot trails the touch, or the brief is refused, since the hedge would read a book older than its fill), `execution.hedge_fee_bps`, and `stock`: each venue's `coin` and `usdt`, `rebalance_at`, `transfer_minutes`, and `withdrawal_fee_coin`, which must be named and stays null until it is looked up, while the judge refuses an arbitrage. A brief may add `data.theo: {coin, legs, anchor}` for the consolidated price of {mod}`blinded.theo`; the AT brief as built has none.

On 2026-10-01: 145,157 rows (51,653 and 75,788 book rows, 3,374 and 14,342 trades), 51,653 touch rows and 51,942 depth rows, 5.7 MB in all, built in seconds. MEXC's spread was 124.3 bps against Binance's 12.6, and on the other free days in the raw directory 112.0 (09-01), 11.3 (08-01) and 8.0 (07-01). `DATA-raw.md` and `DATA.md` print the venue tables ({func}`~blinded.bookstats.venue_facts`, {func}`~blinded.bookstats.cross_facts`, {func}`~blinded.bookstats.flow_facts`, {func}`~blinded.bookstats.snapshot_agreement`). A judgement of `bl.market_making` on the quote venue took 3.7 s and 1.55 GB.

## One coin on four spot venues: `blinded demo venues`

`blinded demo venues --coin SOLUSDT` ({func}`~blinded.demo_mm.venues`) puts SOL spot on Binance, OKX, Bybit and MEXC in one file, each venue an instrument (`SOLUSDT_BINANCE` and so on, Binance the target), every book row with `age_ms` and every trade on the one receive clock, nothing thinned: the consolidated price reads every quote. 5,328,099 rows, 65.1 MB; `blinded init` takes 33 s and 2.2 GB. `DATA-raw.md` holds a table of the venues, which the build reproduces (2,803,070, 537,879, 520,583 and 190,413 book rows; collector delays none, 4.1, 36.8 and 6.6 ms at the median), and the snapshot check's MEXC and Binance columns.
