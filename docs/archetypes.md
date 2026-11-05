# Archetypes

An archetype is a strategy known by two things: who pays it, its *edge*, and how the judge scores and simulates it, its *mechanism*. Order-book imbalance and trend following are both directional: one position in one instrument, scored against that instrument's forward return. They are paid by different people, a slower quote in one case and a slow crowd in the other, and they die in different ways. So blinded keeps the two axes apart. The mechanism is one of the {doc}`families <families>` and decides what the judge computes; the edge decides the Ladder, the sessions and how the report groups the run.

blinded has eleven archetypes ({mod}`blinded.archetypes`). Ten are the ones of the reference page *Thirty Quant Strategies* that some free crypto dataset can test, and the eleventh is `cross_venue_making`, described below. The other twenty on that page need options, corporate events, text, rate curves, fundamentals or an exchange session, or hold for months to years, and are not in the registry: an entry for them would be a refusal that can never change.

| n | decorator | edge | mechanism / template | typical holding | needs |
|---|---|---|---|---|---|
| 1 | `bl.lead_lag(leader, follower)` | speed | lead_lag / threshold, linear | 10 µs to 100 ms | 2 instruments |
| 2 | `bl.order_book_imbalance(instrument)` | speed | directional / threshold, linear | 1 ms to 10 s | `bid_qty` and `ask_qty` in the book |
| 3 | `bl.trade_flow_persistence(instrument)` | speed | directional / threshold, linear | 1 s to 6 h | trades with `side` |
| 4 | `bl.pairs_trading(a, b)` | relative value | pairs / pairs | 10 min to 30 days | a pair that is not one product on two venues or a carry pair |
| 5 | `bl.cross_venue_arbitrage(a, b)` | speed | pairs / pairs | 1 µs to 1 s | one product on two venues, under `archetypes.cross_venue` |
| 6 | `bl.statistical_arbitrage()` | relative value | basket / basket | 1 h to 14 days | a cross-section of 20 |
| 7 | `bl.short_term_reversal()` | liquidity | basket / basket | 4 h to 30 days | a cross-section of 20, trades |
| 8 | `bl.basis_trade(spot, perp)` | carry | pairs / carry | 8 h to 90 days | a `funding` stream and a pair under `archetypes.carry` |
| 9 | `bl.trend_following(instrument)` | underreaction | directional / linear, threshold | 14 days to 365 days | 20 holding periods of validation: 40 weeks |
| 10 | `bl.market_making(instrument)` | liquidity | directional / maker | 1 ms to 10 min | a touch stream: the book at every change with its sizes, and each trade at its own price |

Each registry entry carries the page's sentences (the edge, the other side, how it dies, what blinded saw, what to read), the band, the capacity, the built-ins, the template's default parameters and the report's death checks. Short-term reversal's band is 4 hours to 30 days rather than the page's 1 day to 1 month: perpetuals trade around the clock, and the cross-sectional reversal documented in crypto is strongest at horizons of hours. Market making runs only where the file keeps a touch stream: its strategy returns quotes, `bl.quote(...)`, and a fill engine ({mod}`blinded.sim_maker`) rests them on the recorded book and fills them against the recorded trades under four models of the queue. A file thinned to the read lattice, or of bars, keeps no queue, and there it is refused with the line "needs the book at every change with its sizes and each trade at its own price and stamp (a touch stream); the file has ...".

## Writing one

```python
import blinded as bl
from blinded import features as F

lag = bl.lead_lag(leader="BTCUSDT_PERP", follower="SOLUSDT_PERP")


@lag.feature("what BTC did in the last second and SOL has not")
def btc_sol_gap(panel):
    return F.lead_gap("BTCUSDT_PERP", "SOLUSDT_PERP", "1s")


@lag.strategy(requires=["btc_sol_gap"])
def sol_follows_btc(panel, features):
    return bl.threshold(features["btc_sol_gap"], z_in=3.0, z_out=0.5)


@bl.pairs_trading(a="ETHUSDT_PERP", b="BTCUSDT_PERP")
def eth_btc_pairs(panel, features):
    return bl.pairs("ETHUSDT_PERP", "BTCUSDT_PERP", z_in=2.0, z_out=0.5, window="2h", every="30m")
```

A decorator call takes the instruments and returns a {class}`~blinded.archetypes.Bound`. Used on a function, it registers a strategy; `.strategy(requires=...)` does the same with requirements, and `.feature(hypothesis)` registers a feature. What it registers is exactly what `bl.feature` or `bl.strategy` would, with two tags added, `archetype` and `roles`:

| archetype call | is |
|---|---|
| `bl.pairs_trading(a=A, b=B, requires=R)(fn)` | `bl.strategy(requires=R, family="pairs", pair=(A, B))(fn)`, tagged `pairs_trading` |
| `bl.cross_venue_arbitrage(a=A, b=B)(fn)` | `bl.strategy(family="pairs", pair=(A, B))(fn)`, tagged |
| `bl.basis_trade(spot=S, perp=P)(fn)` | `bl.strategy(family="pairs", pair=(S, P))(fn)`, tagged |
| `bl.lead_lag(leader=L, follower=F)(fn)` | `bl.strategy(family="lead_lag", trades=F)(fn)` with `roles={"leader": L}` |
| `bl.order_book_imbalance(instrument=I)(fn)`, and the trade-flow and trend decorators | `bl.strategy(family="directional", trades=I)(fn)`, tagged |
| `bl.statistical_arbitrage()(fn)`, `bl.short_term_reversal()(fn)` | `bl.strategy(family="basket")(fn)`, legs from the brief |
| `bl.market_making(instrument=I)(fn)` | `bl.strategy(family="directional", trades=I)(fn)`, tagged, whose function returns `bl.quote(...)`; on a file without a touch stream every brief refuses it |

`instrument=None` means the brief's target, as `trades=None` does. The basket archetypes take no legs: their legs are every instrument of the brief, since the cross-section is what they trade and a subset of it would be one more knob.

In a brief with `families:`, or with neither block, a decorator is sugar and nothing else: `@bl.pairs_trading(a, b)` binds as `@bl.strategy(family="pairs", pair=(a, b))`, scores the same to the last digit (`tests/test_archetypes.py` checks a twin of the families run's pairs strategy on train and validation), and the ledger records the tag.

## A brief that studies archetypes

```yaml
archetypes:                         # a brief has this block or families:, not both
  instruments: [BTCUSDT_PERP, ETHUSDT_PERP, SOLUSDT_PERP, XRPUSDT_PERP]   # default: the file's
  only: [lead_lag, pairs_trading]   # optional; default every archetype the file can run
  shares: {speed: 0.5, relative_value: 0.5}    # optional; default equal over the edges that run
  cross_venue: [[XRPUSDT_PERP, XRPUSDT_PERP_BYBIT]]   # pairs that are one product on two venues
  carry: [[BTCUSDT, BTCUSDT_PERP]]                     # spot and perpetual pairs, [spot, perp]
  params:                           # template defaults the report draws with
    pairs_trading: {window: 2h, every: 30m, z_in: 2.0, z_out: 0.5}
```

The block implies the mechanism families, so binding, the judge's targets and the report's family machinery run unchanged: `directional` over every instrument, `lead_lag` with every instrument both leader and follower, `pairs` over every unordered pair in list order plus the declared cross-venue and carry pairs, and `basket` over every instrument when there are at least 20. A pair written the other way round binds in the brief's order with its sign flipped, as in a families run. When the block leaves `instruments` out, `blinded init` writes every instrument in the file into the run's brief.

Parsing refuses a name in `only` or `params` that is not an archetype, a share for something that is not one of the five edges (`volatility` is not: no archetype of the volatility edge survived), shares that are not positive or do not sum to 1, a declared pair with one instrument twice, and a brief with both blocks. `blinded init` refuses a declared pair whose legs are not both in the file, an archetype in `only` that the file cannot run (with its refusal line), a file on which nothing runs, and shares that leave out an edge that has something to run. All of that happens before anything is written.

## The menu

`blinded init` decides once what runs, so the agent never discovers a refusal by submitting. {func}`~blinded.archetypes.facts` reads what the file has: the instruments, the fields of each, the streams, two venues of one product, the carry pairs, the lengths of the train, validation and test scoring windows (the grid warm-up taken off train), the grid step, the fill delay and any declared fill model. {func}`~blinded.archetypes.check` then tests one archetype, in this order, and the first test that fails is the reason:

| order | class | refused when | reason |
|---|---|---|---|
| 1 | `sim` | the archetype needs a touch stream, or the hedge venue's depth, and the file lacks it | needs the book at every change with its sizes and each trade at its own price and stamp (a touch stream); the file has {what it has}. Or: needs the hedge venue's depth |
| 2 | `stream` | a stream it needs is not in the brief | needs {stream} data; the file has {streams} |
| 3 | `venues` | it needs one product on two venues and none is declared | needs the same instrument on 2 venues; the file has one |
| 4 | `instruments` | too few instruments; for `pairs_trading` no pair left that is not declared; for `basis_trade` no carry pair | needs {k} instruments; the file has {n}: {list}, and the two pair forms |
| 5 | `cross_section` | fewer than 20 instruments | needs a cross-section of at least 20 names; the file has {n} |
| 6 | `fields` | a field it needs is on no instrument | needs {field}; the book has {fields} |
| 7 | `span` | validation, or train, is shorter than 20 holding periods of the band's low end | needs {20 lo} of validation (20 holding periods of {lo}); validation is {val} |
| 8 | `only` | the brief's `only` leaves it out | left out by archetypes.only |

Two caveats leave an archetype runnable and both print when both apply: `grid`, when the grid step is above the band's high end ("what the judge measures is the first whole step after the fill"), and `cost`, on a file whose fills are a declared model. A refusal prints as one line, the same string in `ARCHETYPES.md`, the `error` row's reason, `LEDGER.md` and the report:

```text
trend_following (underreaction): not runnable on binance-um-btc-eth-sol-xrp-2024-03-27-3d-500ms: needs 40 weeks of validation (20 holding periods of 2 weeks); validation is 17 h 55 min.
```

A refusal that a change to the brief would turn around with the same file, an `only` that leaves the archetype out or an instrument list shorter than the file's, adds "Change the brief: ...". Durations in these lines are written out ({func}`blinded.durations.words`): "40 weeks", "3 h 25 min", "500 ms".

The span rule is one constant, `archetypes.PERIODS = 20`, and it is a floor on sample size, not a promise of power. On the three Binance days pairs trading passes it five times over, 17 h 55 min against 3 h 20 min, while ETH over BTC reverted with a half-life of one to two minutes; the archetype's typical holding and its behaviour on the data are different things, and the report measures the second. The power the Ladder has is printed beside the menu ({func}`~blinded.archetypes.power`): the bootstrap standard error of a Sharpe near zero is about $1/\sqrt{\text{validation hours}}$ per square-root hour, raised to the floor and to the Ladder's 1-2-5 grid, so on the Tardis day a first strategy needs a validation Sharpe above 1.0, and on the Binance days above 0.5.

{func}`~blinded.archetypes.menu` holds a verdict for each archetype, the edges with something to run and their shares, the cost-model line when there is one, the power line, and the facts. `blinded init` writes it to `vault/menu.json`, checks that the training panel gives the verdicts the brief alone gives (the driver's groups are computed from the brief), and renders `workspace/ARCHETYPES.md`: one line with the count, the shares, the power line, then per edge with something to run its tag and sentence and per archetype the decorator filled in with the brief's instruments, the page's sentences, the band and the span rule against this validation, the built-ins, what the judge scores a feature against and what a strategy returns, its caveats, and last the refused archetypes with their lines. The daemon binds every submission against `menu.json`; the report shows it; `blinded replay` recomputes it from the vault's brief and training rows and compares every verdict, line and number.

The first ten on the six files blinded has, as `tests/test_archetypes.py` pins them (R runnable, C runnable with a caveat, X refused, with the class). `cross_venue_making` needs the hedge venue's depth and is refused on all six, class `sim`:

| archetype | Tardis day | three Binance days | XRP perp on Binance and Bybit | twenty perps on 15-minute bars | eight spot and perp pairs on 15-minute bars | four perps on hourly bars |
|---|---|---|---|---|---|---|
| lead_lag | R | C grid: the 500 ms step is above 100 ms | R | C grid, cost | C grid, cost | C grid, cost |
| order_book_imbalance | R | R | R | X fields: the bars have no sizes | X fields | X fields |
| trade_flow_persistence | R | R | R | C cost | C cost | C cost |
| pairs_trading | R: 200 min against 205 | R | X instruments: the only pair is one product on two venues | C cost | C cost: the cross-coin pairs | C cost |
| cross_venue_arbitrage | X venues | X venues | R | X venues | X venues | X venues |
| statistical_arbitrage | X cross_section | X cross_section | X cross_section | C cost | X cross_section: 16 names | X cross_section |
| short_term_reversal | X cross_section | X cross_section | X cross_section | C cost | X cross_section | X cross_section |
| basis_trade | X stream | X stream | X stream | X stream | C cost | X stream |
| trend_following | X span | X span | X span | X span: 40 weeks against 110 d 11 h 45 min | X span: the same | C cost: 40 weeks against 287 d 5 h |
| market_making | R | X sim: a book thinned to the read lattice | R | X sim: bars | X sim: bars | X sim: bars |

Five run on the Tardis day, in three edges (speed, liquidity and relative value); four on the Binance days, in two (speed and relative value); five on the cross-venue day, in two (speed and liquidity); five on the twenty perpetuals, in three (speed, liquidity and relative value); four on the eight spot and perpetual pairs, in three (speed, relative value and carry); and four on the hourly bars, in three, which add underreaction. All ten run on some file. The runs made before market making could run on the Tardis and cross-venue files keep the menu `blinded init` gave them, which their reports show. On a bar file every runnable archetype carries the `cost` caveat and `ARCHETYPES.md` opens with the fill-model line ({doc}`sim`).

## Groups, Ladders and shares

The daemon, the driver and the report loop over groups, {attr}`Brief.groups <blinded.brief.Brief.groups>`: in an archetypes brief the edges with something to run, in the order speed, liquidity, relative value, underreaction, carry; in a families brief the families; with neither block the one group `directional`. Each group has a Ladder per kind, shared by its archetypes: once an order-book-imbalance feature is released at 0.16, a lead-lag feature releases only above 0.16 plus its step. Sessions go to the open group with the fewest sessions for its share, a group closes on its own streak, and the final pick is the released strategy with the highest released best across groups, as in {doc}`driver`. The strategy floor scales with the brief's block ({doc}`ladder`).

Shares default to equal over the edges that run. A share given to an edge with nothing to run is dropped and the rest renormalised, with a line in `ARCHETYPES.md` and `BUDGET.md`: on the Binance days `{speed: 0.5, relative_value: 0.3, carry: 0.2}` becomes speed 0.625 and relative value 0.375.

A submission without an archetype, or of an archetype the menu refuses, is an `error` row with the menu's pointer or line before any of its code runs. It counts as a trial for the deflation, as every error row does, and resets no streak. Its row carries the archetype and edge the decorator named, so the report can count it. `bl.pairs_trading` on a declared cross-venue or carry pair is refused with a pointer to `bl.cross_venue_arbitrage` or `bl.basis_trade`, and those two refuse a pair that is not declared.

The ledger records the mechanism in `trials.family`, the tag in `trials.edge` and `trials.archetype`, and the Ladder's key in `releases.group`; `LEDGER.md` shows the edge and archetype after the family ({doc}`ledger`). `RULES.md` replaces its families section with one that points at `ARCHETYPES.md`, the prompt names the session's edge and its archetypes, and `BUDGET.md` has a row per edge with its members.

## How it dies

Each archetype names the death checks the report runs on it, and the report's section 4 prints them in a "How it dies" panel under the archetype's charts, with the page's sentence on how it dies above the numbers and a column per segment: train, validation, and test when the final strategy was scored there. The checks run on the archetype's own pick, the rule of a families run kept to its trials: the final strategy, else the last released, else the best on validation. Every number is computed by {mod}`blinded.dies` inside `report.recompute`, stored under `recompute.diagnostics.archetypes[edge][archetype].dies`, and checked by `blinded replay`.

| check | the question | the numbers |
|---|---|---|
| D1 `costs` | is the edge smaller than the spread plus fees | the gross P&L over the units traded, the break-even fee; the five sim rows; net bps/hour at fees of 0, 1, 2.5, 5 bps and the brief's, at latency 0 and the brief's, and at half and twice it where the file keeps those stamps (one bar only on bars) |
| D2 `race` | is the move over before the fill | the feature's rank IC on returns from the decision and from the fill at each horizon, the peak, and the share of it left to a taker; the order's entry cost less the half spread, what moved in flight; for lead-lag the C table; on bars, "not measurable below the bar" |
| D3 `capacity` | how much of the touch one unit takes | per leg, each fill's size over the far touch's size as-of the fill, the median and the share above 1, and the units traded over the window's volume; on bars no touch share, since the book has no sizes |
| D4 `worst_window` | when it lost | net P&L per clock window of the brief's block: the worst, the best window's share of the total, the longest stretch under water in hours, the drawdown |
| D5 `half_life` | how long the signal or the spread lasts | the feature's half-life from its autocorrelation at one step, and for a pair the median window half-life of its spread against a random walk's, and the share of windows that did not revert |
| D6 `mae` | does it diverge before it converges | each round trip's adverse excursion, median and worst, and the share under water by more than its own cost that still closed positive |
| D7 `funding` | what carry paid or cost | the perpetual leg's funding over the window, split into what accrued while it was short and while it was long; the mid P&L of both legs, the basis change; both as cumulative lines; the settlements in the window and their mean rate; "no funding stream on this file" elsewhere |
| D8 `crowding` | is everyone in the same book | "not measurable from this data" |
| D9 `breadth` | did one name carry it | for a basket, the share of legs with a positive net, the best leg's share of the total when that is positive, the mean number of legs held, the rebalances and the units traded at each |
| D10 `whipsaw` | did it get chopped | the share of round trips closed within the band's low end, the hit rate, the mean win over the mean loss, the share of the net made by the best tenth |
| D11 `venue_feed` | is the lead a feed schedule | for cross-venue arbitrage, the second venue's median gap between distinct receive stamps over the segment and its largest 10 ms bin, and for each direction across the venues $C$ at its peak against $C$ at the lag nearest that median |
| D12 `toxicity` | are the fills the ones nobody wanted | for a market maker, per fill model: what the resting fills captured against the mid at their stamps, the mid's move in the fill's direction 1, 10 and 60 s after, and the rest of the gross, each per unit filled; the share of fills by a trade through the price, by the book crossing the price with no trade, alone inside the spread, and by the queue at the touch traded out; the 10 s move over the capture |
| D13 `queue` | where in the queue you were | for a market maker, per fill model: the median and 90th percentile wait from an order's arrival to its fills; the share of the orders that rested and were cancelled unfilled; post-only rejections over orders sent; the mean queue ahead at arrival in order sizes; the share placed deeper than the touch; the share of decisions with a bid, and with an ask, resting; the mean and largest absolute inventory |
| D14 `picked_off` | did the deep venue move through the quote first | for an arbitrage, the share of fills picked off (captured below zero: the hedge venue's mid had already passed the price), the share with the hedge venue's touch already through it, the mean captured of the picked-off fills, and both shares by hour |
| D15 `hedge` | what the hedge cost | the hedge delay, slippage and fees a fill in bps; the share of hedges past the hedge venue's 25 levels; the visible crosses at the fill (the hedge venue's touch better than the price by more than both fees) and the share still there for the hedge's walk; the `hedge 1 s` row's net a fill and an hour against the brief's |
| D16 `stock` | where the coin piles up | the buy share, the drift (USD of coin a day moved onto the quote venue), the days the stock it drains would last at that rate, the share of decisions with each side withdrawn for stock, the rebalances booked and their fees |
| D17 `counterparty` | who the fills come from | the quote venue's prints a day and USD, the size distribution's p10, p50 and p90 and the largest 1 USD bucket's share, the gap's coefficient of variation, the share printed at a price never shown as the touch within a second, and the line "the fills are this flow's prints landing through your quote; the replay assumes it does not change when you are there" |

The arithmetic is checked by hand in `tests/test_dies.py`: the five-point book's +6 bps gross over four units is a break-even fee of 1.5 bps and -24 net at 7.5; a net path of +2, -3, -1, +5 inside one round trip has an adverse excursion of -2 and closes at +3; fills of 1 and 3 against touches of 4 and 2 have shares 0.25 and 1.5, median 0.875, half above 1; window nets of +10, -4 and +7 give a worst of -4 and a concentration of 10/13; an autocorrelation of 0.5 on a 100 ms grid is a half-life of 0.1 s; round trips of 1, 3 and 20 steps against a low end of 2 steps are a whipsaw share of 1/3. The D1 grid's cell at the brief's fee and latency is the sim's `brief` row, and its cell at no fee and no latency the `latency 0, fees 0` row.

A market maker's checks run on the passes of the report's bracket ({doc}`report`), each segment on the panel the judge scored it on. D1 is on the scored model's fills: the gross over the units filled while resting is the break-even maker fee, the rows are the bracket's (the scored model, the four models, latency 0, the fee rows), and the grid is the net per hour at maker fees of -0.5, 0, 1 and 2 bps and the brief's, at latency 0 and the brief's. D3 compares each resting fill's order with the size displayed on its side of the touch at its arrival (median, and the share over 1), the median order with the median aggressor order (trades on one stamp and side summed), counts partial fills, and divides each side's units filled by the other side's aggressor volume; when the median order is over a tenth of the median displayed size it prints the caveat that the engine assumes the market does not react to the quote. D4 and D6 are a position strategy's, on the scored model's net and its inventory runs on the grid; D7 needs the file's funding stream. D12 and D13 have a column per fill model. A position strategy on a file with a touch stream adds the `maker-taker` row to D1's rows ({doc}`sim`).

A pairs archetype with no accepted strategy is drawn and checked at its template's parameters (`archetypes.params`), on its declared pair or, for pairs trading, on the implied pair with the shortest median half-life on train; the search over the implied pairs runs only then, and once per process, since on twenty names it fits 190 pairs and takes 18 s. A basket archetype with no accepted strategy is drawn and checked as `bl.basket` at its parameters on a default score: for short-term reversal minus each leg's return over `lookback`, z-scored across the legs; for statistical arbitrage minus the trailing `lookback` z-score of each leg's log mid less the legs' mean, its residual on an equal-weight market.

For a basket, D6 pools each leg's own round trips, since its legs open and close at different times, and D9 counts them. The basket's charts are its feature's cross-sectional rank IC by horizon and by window, the markout fan of the basket the feature implies at the template's parameters, its strategy's net P&L leg by leg as thin lines with the sum bold, and the fan of its own rebalances: at each, the move of every leg weighted by its new position over the gross. Cross-venue arbitrage adds to the pairs charts the cross-correlation across the two venues, both ways round, which is its D2 C table and which D11 reads against the second venue's cadence. On the cross-venue day Binance leads Bybit on the receive clock, $C$ peaking at 0.54 at 80 ms on train and 0.58 at 70 ms on validation, past the cadence (60 and 40 ms), and falling to 0.27 and 0.28 on the step a taker holds.

## Cross-venue making

`bl.cross_venue_making(quote, hedge)` is the eleventh archetype, liquidity's, with the `arb` template: a strategy returns `bl.quote(...)` on the quote venue with `fair` a price off the hedge venue, and the sim hedges every fill there at its depth ({doc}`sim`). It runs only on a file that has the hedge venue's depth (`data.depth`), the two venues declared as one product under `archetypes.cross_venue`, the quote venue's touch stream, and a stock block whose `withdrawal_fee_coin` is set; the refusals read "needs the hedge venue's depth at every change (a depth file); the file has none", "needs the quote and hedge instruments declared as one product under archetypes.cross_venue", "needs the stock block" and "stock.withdrawal_fee_coin is unset: look it up on both venues' withdrawal pages for ATUSDT and its network". Its binding names the hedge (`Binding.hedge`), and on its file market making binds to the quote venue only: the hedge venue's fills belong to the hedge. The judge scores the lower of RA's and PR's Sharpe on net per block, both legs, under the hold rule, as for market making. Its menu entry prints, from one risk-averse pass of the archetype's own quote (30 bps off the hedge venue's touch at the brief's size) on train, the fills, what each captured and paid in slippage and fee, the picked-off share and the quote venue's flow. Its death checks are D1, D3, D4 and D6 in an arbitrage's terms and D14 to D17; the fee verdict under D1 is on the hedge's taker fee, the venue and product the hedge instrument's ({doc}`report`).

On the arbitrage file the market-making entry words its "This book" line for a wide book: where the spread is one tick under half the time it gives the spread's median in ticks and bps, and says that an offset of 0 rests half that from the mid.

## Carry

`bl.basis_trade(spot, perp)` binds a pair the brief declares under `archetypes.carry`, in the pairs mechanism with the `carry` template. Its feature is scored against {func}`~blinded.target.carry_returns`, the spread return of spot over the perpetual plus the funding a short perpetual collects over the hold, so a high value means "long spot, short the perpetual pays"; its strategy is simulated on both legs with funding accrued at each settlement, and `bl.carry` is the built-in ({doc}`sim`). On the carry file its section of `ARCHETYPES.md` adds the mean funding rate per settlement of each perpetual on train, and its sum as a rate a year, and the fees of a round trip in both legs, 2 x (10 + 5) = 30 bps there.

The report draws a basis trade with the four pairs charts on its pair, fitted at the brief's pairs-trading window, shaded by its strategy's position or, with no accepted strategy, by `bl.carry` at its parameters; then D7's two cumulative lines, the funding received against the basis change. D6 and D4 include the funding, since it is part of each step's net.
