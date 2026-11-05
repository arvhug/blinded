# Strategies and the taker sim

A feature that predicts returns is not yet a way to make money. A strategy turns features into positions, and the simulator turns position changes into fills at prices you could have traded at, after your latency, paying the spread and the fee. Most fast signals that survive the truncation test die here, and the sim is built to show which of latency and fees kills them.

## Writing a strategy

```python
import blinded as bl


@bl.strategy(requires=["basis"])
def basis_threshold(panel, features):
    return bl.threshold(features["basis"], z_in=2.0, z_out=0.5)
```

`features` maps each required name to a polars expression for that feature's values, computed on the same panel. The function returns the position to hold after deciding at each grid point, as an expression, a polars Series or a pandas Series, in units of notional; NaN means flat. The judge looks for the required features in the strategy's own file first, then in a `features/` directory next to the strategy's directory, which is the workspace layout.

The strategy goes through the truncation test as one composed function, panel to position: on each cut panel the features are recomputed and handed to the strategy. A leak in a feature or in the strategy shows up the same way.

## Built-ins

Most of the time the question is whether a signal is worth anything after costs, which does not take a clever position function. Two built-ins cover it.

`bl.linear(f)` holds $\operatorname{clip}(f / s,\ \pm\text{max\_position})$, and `bl.threshold(f, z_in, z_out)` goes long one unit when $f/s > z_{\text{in}}$, short when $f/s < -z_{\text{in}}$, flat when $|f/s| < z_{\text{out}}$, and otherwise keeps what it has. The band between $z_{\text{out}}$ and $z_{\text{in}}$ is hysteresis: without it, a signal hovering at the threshold would trade on every grid step.

The scale $s$ is fitted walk-forward. The day is cut into clock-aligned windows (`every`, 30 minutes by default), and at the start of each window $s$ is set to the standard deviation of $f$ over every earlier row, then held for the window. No row's scale depends on its own window or anything after it, so the built-ins pass the truncation test, and the first window has no scale and stays flat. Three hours of train in 30 minute windows give five windows with a fitted scale.

## The sim

Positions are clipped to $\pm$`max_position` and the run starts flat at the start of the window.

1. **Rebalance threshold.** A change smaller than `min_rebalance` from the position held is not traded. The position that is held moves only when the target has moved far enough from it, which is a dead band, not a rounding.
2. **Fills.** A change $\Delta_i$ decided at $g_i$ fills at $t_i = g_i + L$: buying at the ask, selling at the bid, both read from the book events as-of $t_i$, not from the grid, so 50 ms means 50 ms. Each unit traded pays the instrument's taker fee, `taker_fee_bps` unless `execution.fees` names one for it.
3. **Holding.** The new position is held from $t_i$ to $t_{i+1}$ and marked to the target price (mid by default).
4. **Flips.** Going from long to short, or back, is two fills, a close and an open. A sim that let a reversal cost one half-spread would show an edge that is not there.

Under rules 3, the current rules ({data}`blinded.ladder.RULES`), a fee is charged on the trade's own notional, $|\Delta_i|\, p_i / P_{\text{ref}}$ units at the fill price $p_i$, and a position still open when the window ends pays to leave: it is crossed at the far touch as-of the end, $g_{\text{last}} + \text{step} + L$, with its half spread and taker fee charged to the last step. Under rules 1 and 2 the fee was on the units traded at $P_{\text{ref}}$, and the last position was marked to the mid and left. The formulas and hand examples on this page are in rules 2's terms, because the tests that pin them run under rules 2 (`tests/test_sim.py`, `tests/test_maker.py`, `tests/test_arb.py`).

### Fills are read at their stamps

The decision at $g$ sees events with $\text{available\_at} \le g$, which with a feed delay $d$ are those stamped by $g - d$. The order reaches the book as it stood at stamp $g + L$, and that is the book as-of $\text{available\_at} \le g + L + d$, so the sim reads fills and marks there, with each instrument's own $d$. The information-to-fill gap is then $L + d$, as it is live. Forward returns start and end on the same clock. Read as-of $g + L$ instead, a delay would move the decision's information and the fill by the same $d$, and the gap would stay $L$ whatever the brief said. With a receive clock $d$ is 0 and nothing changes; `tests/test_sim.py` runs the example below with every row arriving 0.9 s late and gets the same fills.

### On a thinned file

A file built by `blinded demo binance` keeps only the book rows these reads return ({doc}`demo-data`). With a grid point $g$, latency $L$ and feed delay $d$, every read the sim makes is at one of three stamps:

| read | as-of `available_at` | stamp |
|---|---|---|
| the decision's information, the panel | $g$ | $g - d$ |
| fills and marks of the `brief`, `fees 0` and maker-proxy rows | $g + L + d$ | $g + L$ |
| fills and marks of the two `latency 0` rows, and the naive judge | $g + d$ | $g$ |
| the next mark, the end of a forward return at a whole number of steps | the same, a whole number of steps on | the same |

So the five rows of a strategy are exactly what the raw file would give, and the builder checks it on the first hour of real data. The maker proxy's trade rate counts the trades each kept row stands for (`n`), and a trade stamped on a read has a row of its own, so the count inside a window that starts and ends on grid points is exact too. A latency the file was not thinned for, `blinded eval --latency-ms 100` on the 50 ms Binance file, reads the kept row before the fill; the result carries a warning saying its numbers are approximate. The brief itself is refused at load if its latency, delays or horizons ask for stamps the file does not keep.

### On bars

A file of klines, built by `blinded demo trend` or `blinded demo crosssection` ({doc}`demo-data`), has no bid or ask, so its brief declares a fill model (`data.cost_model`, {class}`~blinded.brief.CostModel`): each bar is a book row at its close stamp, `open_time + interval - 1 ms`, whose bid and ask are the close times $1 \mp hs/10^4$, with $hs$ the instrument's half spread in bps measured on a top-of-book day before train. The grid step is the bar, the grid starts on a bar boundary and the latency is one bar. A decision at boundary $g$ sees the bar that closed 1 ms before it and fills at $g + L$, which reads the next bar's close plus or minus $hs$: a taker gets the next close and pays the half spread, longs and shorts alike. That is a declared model, neither the best nor the worst price of the bar, with no impact, and every page that shows a number from such a file says so. Nothing in the sim changes for it: it makes the reads above.

The worked example, in `tests/test_demo.py`: four 1-minute bars closing at 100, 100, 120 and 100 with $hs = 10$ give books of 99.9/100.1, 99.9/100.1, 119.88/120.12 and 99.9/100.1. Long one unit decided at the boundary between the second and third bars and flat one bar later, at a latency of one bar: the buy fills at 120.12 against a mark of 120, which is the reference, 10 bps of spread; the sell at 99.9 against 100, $0.1/120 = 8.3333$ bps; the mid P&L is $(100 - 120)/120 \times 10^4 = -1666.6667$, the gross $-1685.0$, the fees 15 at 7.5 bps a fill, the net $-1700.0$. At latency 0 the same decisions fill at 100.1 and 119.88 against a reference of 100: mid $+2000$, spreads 10 and 12, gross 1978, net 1963. The latency-0 row on bars is the close a decision has already seen, as unrealistic as anywhere.

`blinded init` checks the model against the file: on every book row $(ask - bid)/(2\,\text{mid})$ must be the brief's half spread to a relative $10^{-9}$, and the day it was measured on must end by the start of train ({func}`blinded.vault.check_cost_model`); otherwise it refuses before writing anything.

### Units

A position of 1 is one unit of reference notional, the quantity worth the mark price $P_{\text{ref}}$ at $t_0$. Every P&L figure is in basis points of that notional:

$$
\text{gross}_i = \frac{10^4}{P_{\text{ref}}}\left(h_i\,(m_{i+1} - m_i) - |\Delta_i|\,|X_i - m_i|\right),
\qquad
\text{net}_i = \text{gross}_i - |\Delta_i| \cdot \text{fee},
$$

where $h_i$ is the position held after step $i$, $m_i$ the mark at $t_i$ and $X_i$ the fill price. Because the quantity is fixed, the P&L of a round trip telescopes to exit price minus entry price, exactly. The test suite checks the gross P&L against cash accounting: pay the ask or receive the bid on every change, mark what is left at the end.

### Reported per hour

Six hours of one instrument cannot be annualised honestly, so every rate is per hour.

| column | meaning |
|---|---|
| `net/h`, `gross/h` | P&L in bps of notional per hour, after and before fees |
| `SR/sqrt(h)` | net Sharpe per $\sqrt{\text{hour}}$: mean over standard deviation of 5 minute block P&L, times $\sqrt{12}$ |
| `maxDD` | largest fall of cumulative net P&L from its running peak, in bps |
| `fills`, `fees` | number of fills (a flip is two) and total fees in bps |
| `turn/h` | units of notional traded per hour |
| `net/fill`, `gross/fill` | P&L per fill, the number that says whether a taker can ever work |
| `hit` | share of round trips (runs of one sign, entry and exit costs included) that made money |

The Sharpe uses 5 minute blocks rather than grid steps because positions persist: per-step P&L is autocorrelated, and multiplying a per-step Sharpe by $\sqrt{36{,}000}$ steps per hour would overstate it. To annualise anyway, multiply by $\sqrt{24 \times 365} \approx 93.6$, and remember that six hours of one day is not a year. The judge goes further for a strategy whose holds outlast the block: it bootstraps in blocks as long as the holds ({doc}`ladder`).

### The diagnostic rows

Every strategy gets five rows, and a sixth where the file keeps a touch stream of its instrument:

| row | latency | fee | |
|---|---|---|---|
| `brief` | the brief's | the brief's taker fee | the result |
| `latency 0` | 0 | the brief's | unrealistic, diagnostic only |
| `fees 0` | the brief's | 0 | unrealistic, diagnostic only |
| `latency 0, fees 0` | 0 | 0 | unrealistic, diagnostic only |
| `maker proxy` | the brief's | maker fee | upper bound, never in the Ladder |
| `maker-taker` | the brief's | maker fee resting, taker fee crossed | the same positions posted at the touch and crossed at a deadline, on the fill engine; diagnostic, never in the Ladder |

Read across them: if `latency 0, fees 0` is positive and `fees 0` is not, latency is what kills the strategy; if `fees 0` is positive and `brief` is not, it is the fee.

### The maker proxy

The same position changes, posted passively instead: a buy rests at the bid as-of $t_i$, a sell at the ask. It fills with probability $1 - e^{-\lambda h}$, where $\lambda$ is the window's rate of trades on the other side (sells, for a resting buy; each row counted `n` times on a thinned file) and $h$ the primary horizon, and it is marked at $t_i + h$. That assumes the order sits at the front of the queue and that a fill is no more likely when the price is about to move through it. Both are false in the direction that flatters the result, so the row is an upper bound. It exists so a report can say "not viable as a taker; any value is as a quote skew", and it never enters the Ladder or a headline number.

## A worked example

From `tests/test_sim.py`. Grid 1 to 5 s, latency 0.5 s, fee 7.5 bps. The book:

| time (s) | bid | ask | mid |
|---|---|---|---|
| 0.0 | 99.98 | 100.02 | 100.00 |
| 1.2 | 99.99 | 100.01 | 100.00 |
| 2.2 | 100.19 | 100.21 | 100.20 |
| 3.2 | 100.09 | 100.11 | 100.10 |

The strategy asks for $+1, +1, -1, -1, 0$. $P_{\text{ref}} = 100.00$, so 0.01 of price is 1 bps.

- At 1 s it decides to buy; the order fills at 1.5 s, at the ask then, 100.01. The ask at the decision was 100.02; the quote moved in between, and the sim uses the later one.
- At 3 s it reverses: sell 2 at the bid as-of 3.5 s, 100.09. That is two fills, closing the long and opening the short.
- At 5 s it buys back 1 at 100.11.

The long made $100.09 - 100.01 = 8$ bps, the short $100.09 - 100.11 = -2$: 6 bps gross. Four fills at 7.5 bps cost 30, so net is $-24$ bps. Over five seconds, $1/720$ of an hour, that is $-17{,}280$ bps per hour. At latency 0 the same positions fill at 100.02, 100.19 and 100.11, and the gross is 25 bps: the 0.5 s delay cost 19 of it.

## Two legs

A strategy of the pairs family returns a position for each leg, `{A: h_A, B: h_B}`, in units of one notional of each leg. {func}`blinded.sim.simulate_legs` runs the sim above once per leg, on that leg's own book, marks, reference price and fee, and adds the legs up:

$$
\text{pnl}_t = \sum_{i \in \{A, B\}} \frac{10^4}{P^i_{\text{ref}}}\Big( h^i_t\,(m^i_{t+1} - m^i_t) - |\Delta^i_t|\,|X^i_t - m^i_t| \Big) - \sum_i |\Delta^i_t|\, \text{fee}_i .
$$

The {class}`~blinded.sim.MultiSim` it returns has the summed per-step P&L, fees and fills, the block P&L of the summed net (so Sharpe is of the pair, not of either leg), each leg's own {class}`~blinded.sim.Sim`, and the mean gross exposure $\sum_i |h^i_t|$, which a pair holding one notional in each leg puts at 2 while in position. A round trip is a run of one sign of leg A's position, charged with both legs' P&L and costs. The five rows of the report are the same, each leg at its own fee, and the maker proxy is each leg's at its own trade rate, added up.

`bl.pairs(A, B, z_in, z_out, window, every)` is the built-in. It holds $u = -\text{state}(z)$ in A and $-\tilde\beta u$ in B, where $z$ is `F.spread_z` and $\tilde\beta$ is `F.hedge_ratio`, both fitted on the trailing `window` at the row before each clock window of `every` and held through it; the state is `bl.threshold`'s hysteresis. B also trades when $\tilde\beta$ is refitted while a position is held, and pays for it.

### The example with a second leg

Leg A is the book above, $+1, +1, -1, -1, 0$. Leg B is short when A is long:

| time (s) | bid | ask | mid |
|---|---|---|---|
| 0.0 | 49.98 | 50.02 | 50.00 |
| 1.2 | 49.99 | 50.01 | 50.00 |
| 2.2 | 50.09 | 50.11 | 50.10 |
| 3.2 | 50.04 | 50.06 | 50.05 |

with positions $-1, -1, +1, +1, 0$. $P^B_{\text{ref}} = 50.00$, so 0.01 of B's price is 2 bps. B sells at 49.99 at 1.5 s, buys two at 50.06 at 3.5 s and sells at 50.04 at 5.5 s. The short made $49.99 - 50.06 = -0.07$, $-14$ bps; the long $50.04 - 50.06$, $-4$ bps: $-18$ gross. As cash: $+49.99 - 2 \times 50.06 + 50.04 = -0.09$, which is $-18$ bps of 50. The pair is $6 - 18 = -12$ gross on eight fills, $8 \times 7.5 = 60$ of fees, $-72$ net over $1/720$ of an hour. Its round trips are $-36$ each: A's long run nets $-7$ and B's part of it $-20 + 10$ of mid, $2 + 7.5$ to open and half of the flip, $2 + 7.5$, so $-29$; the short run is $-17$ and $-19$. `tests/test_pairs.py` checks every figure here.

## Baskets

A basket strategy returns a position for every leg of the brief's basket family, and {func}`blinded.sim.simulate_legs` trades it as it trades two: each leg on its own book, marks, reference notional and fee, the P&L added up, the Sharpe of the sum. A round trip is still a run of one sign of the first leg, charged with every leg; the report's D6 for a basket pools each leg's own runs instead, since a basket's legs open and close at different times ({doc}`archetypes`).

`bl.basket(score, top=4, every="4h", max_position=1.0)` is the built-in. At the start of each clock window of `every`, aligned as `bl.pairs` aligns its refits, each leg's score is read at the row before the window and the legs with a finite score are ranked, ties in leg order; the top `top` hold $+\text{max\_position}/\text{top}$ and the bottom `top` hold $-\text{max\_position}/\text{top}$ through the window, the rest 0. Gross exposure is $2\,\text{max\_position}$ while in position. With fewer than $2\,\text{top}$ finite scores the window is flat, and so is the first, which has no earlier row. Everything is a read of earlier rows at a window start, so it passes the truncation test with the feature library. With three legs, `top = 1` and 4-row windows, scores of (3, 1, 2) then (1, 2, 3) at the rows before the second and third windows give $(+1, -1, 0)$ then $(-1, 0, +1)$: the first leg flips, two fills, the second closes and the third opens.

### The example with a third leg

Add a leg C to the two-leg example, its book a copy of A's. Held flat, it changes nothing: $-12$ gross, eight fills, 60 of fees, $-72$ net, round trips $-36$ and $-36$, gross exposure 1.6, and `fills_C = 0`. Given A's positions, it adds A's $+6$ gross on four fills: $-6$ gross, twelve fills, 90 of fees, $-96$ net, $-96 \times 720 = -69{,}120$ bps per hour, gross exposure $(3 + 3 + 3 + 3 + 0)/5 = 2.4$, and round trips of A's runs, $-7 - 29 - 7 = -43$ and $-17 - 19 - 17 = -53$, which sum to $-96$. D9 on it: no leg nets positive (A and C $-24$, B $-48$), 2.4 legs held on average, and three rebalances (the open, the flip, the close) of four units each. `tests/test_basket.py` checks every figure here.


## Carry: funding at each settlement

A perpetual swap has no expiry; its price is held near spot by a funding rate that longs pay shorts (or shorts pay longs, when it is negative) at each settlement, every 8 hours on Binance. On a file with a `funding` stream, one row per settlement of each perpetual with its `rate`, {func}`blinded.sim.simulate` pays it. At a settlement stamped $f$ the position held is the one whose fill was the last at or before $f$, $h_i$ for the largest $i$ with $t_i = g_i + L \le f$, so a stamp exactly on a fill counts the position after it, and the leg accrues

$$
a_f = -\,h_i \cdot \text{rate}_f \cdot 10^4 \cdot \frac{m(f)}{P_{\text{ref}}}
$$

bps of its reference notional, with $m(f)$ its mark as-of $f$, booked at step $i$: a long pays a positive rate, a short collects it. Settlements from the first fill to the end of the last step count; before the first fill the run is flat. The stamp is the settlement's, not when the rate became available: who pays is decided by the position held at the settlement. `Sim.funding` holds the accrual per step and `net` is gross less fees plus funding, so the Sharpe, the drawdown and the blocks include it; the metrics gain `funding_bps`. A round trip books funding like mid P&L, to the run open at that step. Every leg with a funding stream accrues it, in a pair, a basket or alone, so a pairs strategy on two perpetuals pays its funding too. The diagnostic rows all carry it, since funding is neither a fee nor a latency; the maker proxy, an edge per passive fill at the horizon rather than a holding P&L, does not. On a file without a funding stream `funding` is None and no number moves.

`bl.carry(spot, perp, rate_in=0.0001, rate_out=0.0, max_position=1.0)` is the built-in of `bl.basis_trade`. It reads `pl.col(f"{perp}.rate")`, the last settled rate as-of the decision, and holds long spot and short the perpetual from a rate above `rate_in` until one below `rate_out`, short spot and long the perpetual from a rate below `-rate_in` until one above `-rate_out`, and otherwise what it held, flat before any settlement. On rates of 0.00005, 0.0002, 0.00015, -0.00001 and 0.0 the spot leg holds 0, +1, +1, 0, 0 and the perpetual 0, -1, -1, 0, 0.

A basis trade's feature is scored against {func}`blinded.target.carry_returns`, the spread return of spot over the perpetual plus the funding a short perpetual collects over the hold:

$$
r^c_h(g) = r^s_h(g) + \sum_{f \in [g + L,\ g + L + h)} \text{rate}_f \cdot \frac{m_{\text{perp}}(f)}{m_{\text{perp}}(g + L)},
$$

with $r^s_h$ the pairs target of {doc}`concepts` for spot over the perpetual. The interval is half-open the same way as the sim's rule: a settlement on the entry fill counts, one on the exit fill does not.

### The example with funding

Take the two-leg example with B as the perpetual, settling $+0.0001$ at 2.0 s and $-0.0002$ at 4.0 s. At 2.0 s the live fill is the one at 1.5 s, B short one unit; its mark as-of 2.0 s is 50.00 (the 1.2 s row) against $P^B_{\text{ref}} = 50.00$, so B collects $+1.0$ bps. At 4.0 s the live fill is the one at 3.5 s, B long one; the mark is 50.05 (the 3.2 s row), so B accrues $-(+1)(-0.0002)(10^4)(50.05/50.00) = +2.002$ bps. `funding_bps` is 3.002 and the pair nets $-72 + 3.002 = -68.998$; the round trips are $-36 + 1.0 = -35.0$ and $-36 + 2.002 = -33.998$. A settlement stamped exactly at 3.5 s is B's long, $-1.001$ at a rate of $+0.0001$. At latency 0 the fills move to 1, 2, 3, 4 and 5 s and both stamps fall on a fill, which counts the position after it: B short at 2 s and long at 4 s, so every diagnostic row carries the same 3.002. On a spot that rises 10 bps from entry to exit against a flat perpetual, one settlement of $+0.0001$ inside $[g + L, g + L + h)$ makes the carry target 11 bps, and one just before the entry or on the exit leaves it at 10. `tests/test_carry.py` checks every figure here, and `tests/test_pairs.py` runs each figure of the two-leg example again with a funding stream that settles at 0.

### Borrow on a short spot leg

Shorting spot means borrowing the coin, and the loan pays interest at a rate the venue changes as demand for the coin moves. A file with a `borrow` stream, one row per published rate of each spot instrument with its `borrow_rate`, a day as a fraction, charges it ({func}`blinded.sim.simulate`). A short position pays the rate $b_i$ in force at the stamp $t_i$ that starts its step, as-of like a mark, for as long as the step lasts:

$$
\text{borrow}_i = -\max(-h_i, 0)\; b_i \cdot 10^4 \;\frac{t_{i+1} - t_i}{1\ \text{day}}\;\frac{m_i}{P_{\text{ref}}}
$$

bps of its reference notional at step $i$, on the coins owed valued at the mark when the step starts. A long or a flat step pays nothing, and an instrument without a borrow stream pays nothing either. `Sim.borrow` holds it per step; `net` is gross less fees plus funding plus borrow, so the Sharpe, the drawdown, the blocks and the Ladder all include it, and the metrics gain `borrow_bps`. A round trip books it to the run open at that step. It is neither a fee nor a latency, so every diagnostic row carries the borrow of its own fills. The panel has `SYM.borrow_rate` like any stream field, RULES.md a Borrow section, and ARCHETYPES.md, under the basis trade, each spot leg's mean and highest rate in force on train. `bl.carry`'s reverse trade, short spot and long the perpetual, is where it bites.

### The example with borrow

Take the two-leg example with a borrow stream on A of one rate, 8.64 a day (86,400 bps, a bps a second). A is short from the 3.5 s fill to the 5.5 s one, two steps of a second, each valued at A's mark when it starts, 100.10 (the 3.2 s row), against $P^A_{\text{ref}} = 100.00$: $-(1)(8.64)(10^4)(1/86{,}400)(100.10/100.00) = -1.001$ bps a step. B has no stream, so its short pays nothing. `borrow_bps` is $-2.002$, gross and fees are unchanged at $-12$ and 60, and the pair nets $-72 - 2.002 = -74.002$; the round trips are $-36$ and $-36 - 2.002 = -38.002$. A rate published exactly at 3.5 s is in force for the step that fill starts; one published at 3.6 s first applies to the 4.5 s step, and halving the rate at 4.0 s halves that step's, $-0.5005$. At latency 0 A is short from the 3 s fill to the 5 s one, valued at 100.20 and 100.10 against a reference of 100.00, so those rows carry $-1.002 - 1.001 = -2.003$. `tests/test_carry.py` checks every figure here.

## Quotes and the fill engine

A market-making strategy, one written with `bl.market_making`, returns quotes rather than a position: `bl.quote(bid, ask, size, skew, max_inventory, lean, fair)`, offsets in ticks from each side of the touch, a size a side in units, and the one feedback the sim applies, `skew` ticks per unit of the inventory the strategy knows of. {mod}`blinded.sim_maker` rests them, one order a side, on the recorded top of book and walks every book row and every trade in stamp order. An order reaches the book `latency_ms` after the decision, a cancel `cancel_latency_ms` after it and a fill comes back `response_latency_ms` after it happens, each the order latency unless the brief says otherwise; an order that would cross the book on arrival is rejected, post-only; a side is quoted only while its full fill keeps the known inventory within `max_inventory`; and the window ends with the inventory crossed at the far touch as a taker. On one stamp the order is: cancels, then arrivals, then trades, then book rows, then the decision. On a stamp with no trade, book rows come before the cancels and arrivals. Under rules 1 and 2, book rows came first on every stamp. The engine needs the book at every change with its sizes and each trade at its own price: on a file thinned to the read lattice, or of bars, market making is refused ({doc}`archetypes`).

### Four models of the queue

Binance publishes no order identifiers, so where an order sits in the queue is never observed. Four models each run their own closed loop, since the quotes depend on the fills the strategy knows of:

- **T**, trade-through: the queue ahead is never known, so only a trade strictly through the price fills.
- **RA**, risk-averse: the order joins the back of the displayed size, each trade at its price uses up the queue ahead, and a cancel at the level is taken as behind it, so the queue ahead only ever falls to the displayed size (hftbacktest's `RiskAdverseQueueModel`).
- **PR**, pro rata: as RA, but a cancel comes from ahead of the order in proportion to the queue ahead (hftbacktest's `ProbQueueModel` with the power function of exponent 1).
- **F**, front: the order is always at the front; any trade at or through its price fills it.

Under every model a book row that brings the far touch to a resting order's price fills its whole remainder there, a maker fill with no trade, flagged `crossed` and counted by D12: an ask at or below a live bid, a bid at or above a live ask, and under T only a book strictly through the price, as only a trade strictly through it fills T. The simulated order is not in the recorded book, so when the recorded ask comes down to the bid's price a real seller there would have hit it first; no trade printed only because no recorded bid was at that price. hftbacktest fills these too (`on_best_bid_update`). A row on a cancel's stamp comes before the cancel unless a trade shares the stamp, so it can still cross the order. This is rules 2: a run judged under rules 1, every run before 20 Oct 2026, has no crossing, and its engine fills only on trades ({data}`blinded.ladder.RULES`).

For a fixed quote sequence the fill sets are nested, T in RA in PR in F. No model bounds the P&L: RA undercounts fills and misses the informed cancels ahead of the order, PR overcounts fills when the cancels are uninformed. So the judge scores the worse of RA and PR ({doc}`ladder`), and T and F are printed as the bracket of the fill count.

### The hand example

From `tests/test_maker.py`. Grid 1 to 5 s, every latency 0.5 s, tick 0.01, `bl.quote(bid=0, ask=0, size=10, skew=0, max_inventory=10)`: join the touch on both sides while a full fill keeps the known inventory within ten. Maker fee 2 bps a unit, taker 5. $P_{\text{ref}}$ is the mid as-of 1.5 s, 100.00, so 0.01 of price on one unit is 1 bps.

| stamp | bid / size | ask / size | trades |
|---|---|---|---|
| 0.0 s | 99.99 / 50 | 100.01 / 40 | |
| 1.2 s | 99.99 / 30 | 100.01 / 40 | 20 cancelled before the quote arrives |
| 1.6 s | 99.99 / 50 | 100.01 / 40 | 20 joined behind the quote |
| 1.8 s | 99.99 / 40 | 100.01 / 40 | 10 cancelled; then a sell of 25 at 99.99 at 2.1 s |
| 2.2 s | 99.99 / 15 | 100.01 / 40 | a sell of 20 at 99.99 at 2.6 s |
| 2.7 s | 99.98 / 60 | 100.00 / 25 | a sell of 5 at 99.98 at 2.9 s |
| 3.3 s | 99.99 / 20 | 100.01 / 40 | a buy of 5 at 100.01 at 3.8 s |
| 4.4 s | 100.01 / 10 | 100.02 / 40 | |
| 4.6 s | 99.97 / 80 | 99.99 / 30 | a sell of 10 at 99.97 at 4.7 s, of 10 at 99.96 at 5.7 s |
| 5.8 s | 99.95 / 50 | 99.97 / 30 | |

The bid arrives at 1.5 s behind 30 units. RA keeps 30 ahead through the 1.6 s and 1.8 s rows (the 10 cancelled are taken as behind it), and the 2.1 s sell of 25 leaves 5: the 2.6 s sell of 20 fills it. PR takes the 1.8 s cancel of 10 as $10 \times 30 / 50 = 6$ from ahead, 24 left, so the 2.1 s sell of 25 fills one unit and the 2.6 s sell the other nine. F fills all ten at 2.1 s. T fills only when the 2.9 s sell goes through 99.99, after the level was gone. Every model sells ten at 100.00 at 3.8 s, when a buy goes through its ask; each has one post-only rejection, the ask at 100.01 arriving at 4.5 s against a best bid of 100.01; every model buys ten at 99.97 at 5.7 s and the flatten sells them at 99.95 at 6.5 s.

| model | fills (stamp, side, price, units) | gross | captured | fees | net |
|---|---|---|---|---|---|
| T | 2.9 s buy 99.99 x 10, 3.8 s sell 100.00 x 10, 5.7 s buy 99.97 x 10 | -10 | +10 | 110 | -120 |
| RA | 2.6 s buy 99.99 x 10, then as T | -10 | +20 | 110 | -120 |
| PR | 2.1 s buy 99.99 x 1, 2.6 s buy 99.99 x 9, then as T | -10 | +20 | 110 | -120 |
| F | 2.1 s buy 99.99 x 10, then as T | -10 | +20 | 110 | -120 |

The prices are the same in every model, so the P&L is; what differs is when and how often a fill happens, and on a real day that is what makes the P&L differ.

### Accounting and the rows

{class}`~blinded.sim_maker.MakerSim` keeps the per-step arrays of {class}`~blinded.sim.Sim`, with every fill in its step, so the gross is the cash of every fill, the flatten's included, and a maker's `spread_bps` is negative: earned. Funding accrues on the inventory held at each settlement and borrow on the inventory at each step's start. The decomposition of {func}`~blinded.sim_maker.decompose` splits the gross into what the fills captured against the mid at their stamps, what the mid did over the next 1, 10 and 60 s (adverse selection), and the rest, the inventory held longer and the flatten. The rows are `brief`, the scored model; `RA`, `PR`, `T` and `F`; `latency 0`, the scored model with every latency 0; and the scored model's own fills at maker fees of 0 (`fees 0`, which waives the flatten's taker fee too), a 0.5 bps rebate, 1 and 2 bps: fees change no fill, so those rows are arithmetic on one pass.

### The bracket on the SOL day

`scripts/mm_sol_bracket.py` replays the hand example's policy at size 10 on the SOL perpetual's 2026-10-01, 50 ms orders and cancels, fills known at once, maker 2 bps, taker 5. From 12:00 to 13:00, P&L in bps of one unit of the quote's size:

| | T | RA | PR | F |
|---|---|---|---|---|
| fills (units / 10) | 248 (248) | 1,011 (280) | 1,829 (455) | 8,739 (1,733) |
| gross; net at 2 bps; net at a 0.5 bps rebate | -391; -887; -267 | -366; -928; -229 | -344; -1,259; -121 | +240; -3,229; +1,104 |
| Sharpe per square-root hour, 12 blocks | -6.20 | -7.55 | -7.37 | -13.05 |

The fill count runs from 248 to 8,739 across the models, a factor of 35. Gross rises from T to F while net falls, because the fee is per unit filled; at a rebate the order reverses.

### Maker-taker execution of a position strategy

Where the file keeps a touch stream of the strategy's instrument, a position strategy's sim table has a sixth row, `maker-taker` ({func}`~blinded.sim_maker.simulate_maker_taker`): the same positions, executed by posting each change at the touch and crossing what is left at a deadline. A decision whose target differs from the one its live order was sent for cancels that order and posts the difference between the target and the inventory it knows of, at the best bid for a buy and the best ask for a sell as-of the order's arrival $g + L$, where the taker sim fills at the far touch then. The order rests in the queue; what is unfilled `execution.deadline_ms` (2 s by default) after its arrival is crossed at the far touch as-of that stamp at the taker fee, and what filled while resting pays the maker fee. A decision that keeps the target keeps the order and its place in the queue; one that reverses or cancels the change before the deadline cancels it. Every deadline falls by the end of the window's last step, and the inventory left is marked at the mid there, as the taker sim marks its last position. The row runs under RA and PR and shows the worse by the judge's rule, with `fills` counting every fill, resting or crossed, and `maker_share` the units that filled while resting over all units traded. It is a diagnostic, never in the Ladder: a position strategy is judged on the taker sim, the one fill model every archetype of its edge shares.

On the five-point book above, given its sizes (one unit a side at every row) and its four trades (one unit each), with a 1 s deadline: the 1 s buy rests at 99.99 from 1.5 s; the 2.6 s sell at 100.19 is above it and nothing trades at or through 99.99 by 2.5 s, so the deadline buys at the ask as-of 2.5 s, 100.21, where the taker row bought at 100.01. At 3 s the strategy knows it is long one and wants to be short one, so it posts two units at the ask as-of 3.5 s, 100.11, behind the one unit displayed there; the 3.7 s buy of one at 100.11 uses up that unit, and at 4.5 s the deadline sells both at the bid, 100.09. At 5 s it posts a buy of one at 100.09 from 5.5 s, which no trade reaches; its deadline is the end of the last step, 6.5 s, where it buys at 100.11. Long one from 100.21 to 100.09 and short one from 100.09 to 100.11 is $-14$ bps gross against the taker row's $+6$; four crossed units at 7.5 bps are 30; net $-44$, and `maker_share` is 0 under RA and PR. Under F the 3.7 s buy fills one of the two resting units at 100.11 and the share is 0.25. A change reversed before its deadline costs nothing: with a 3 s deadline the 1 s buy is still resting when the 3 s target turns short, it is cancelled, and nothing trades all window.

## Maker-taker arbitrage: the hedge leg

`bl.cross_venue_making(quote, hedge)` rests quotes on a thin venue A, priced off a deep venue B, and hedges every fill on B as a taker the moment it is known ({func}`~blinded.sim_maker.simulate_arb`). The maker leg is the fill engine above, unchanged but for two things: a trade through the quote fills no more than its own print (`through="print"` on {func}`~blinded.sim_maker.run_engine`; the whole-order rule, `"full"`, is the default and market making keeps it), and the stock rule below may withdraw a side at a decision. `fair` prices the offsets: one price, or a pair `fair=(bid_fair, ask_fair)` that gives each side its own, as a quote off B's touch needs (the bid counted down from `floor(bid_B / tick)`, the ask up from `ceil(ask_B / tick)`).

For each fill at $t_f$, side $s$, price $p$ and quantity $q$:

1. the hedge for $q$ on the other side is sent once the fill is known, at $t_f + L_r$, or at the window's end decision if that is sooner, and reaches B at $t_h = t_f + L_r + L_B$, $L_B$ the brief's `hedge_latency_ms`;
2. it walks B's depth (`data.depth`, the snapshot as-of $t_h$, never the fill's stamp) level by level, a sale into the bids and a purchase from the asks; what the levels cannot fill is marked at the last level's price and the hedge is flagged `unreachable`;
3. the hedge pays `hedge_fee_bps` and the maker fill the quote venue's `maker_fee_bps`, both on the units at $P_\text{ref}$, B's mid as-of the first decision plus $L$;
4. the stock moves: coin $+sq$ and USDT $-sqp$ on A at $t_f$, coin $-sq$ and USDT $+sq\,\text{vwap}$ on B at $t_h$, each less its fee.

Fills are hedged one by one, never netted. {class}`~blinded.sim_maker.ArbSim` is a {class}`~blinded.sim_maker.MakerSim` whose trades are both legs, each marked to B's mid at its own stamp, so a maker fill's capture $s q (m_B(t_f) - p)$ and a hedge's slippage $s q (\text{vwap} - m_B(t_h))$ are in `spread_bps`, the position held between them, $sq$ from $t_f$ to $t_h$, earns the delay $s q (m_B(t_h) - m_B(t_f))$ in `mid_pnl`, and the gross telescopes to the cash of both legs. The last step ends at the later of the end decision plus $L$ and plus $L_B$, so every hedge is inside it.

### The stock on two venues

The brief declares each venue's coin and USDT (`stock`), a share `rebalance_at`, a `transfer_minutes` and a `withdrawal_fee_coin`, which has no default: written null by the builder, it must be looked up on both venues' withdrawal pages, and the judge refuses an arbitrage until it is set. A decision knows the fills stamped by $g - L_r$ and counts both legs of each, since its hedge is sent. A bid rests only while B holds the coin its hedges would sell and A the USDT it would pay, for every unit of that side that could still fill unknown to the decision: the size it wants plus every order of that side it sent whose end it cannot yet know. An ask likewise needs A's coin and B's USDT at B's ask. So the stock never goes short: a side is withdrawn instead, and `stock_withdrawn_share` counts the decisions with one off. When one venue holds more than `rebalance_at` of the two venues' coin, half the difference leaves it at once and arrives after the transfer time less the fee in coin, charged to `fee_paid` at B's mid; the USDT goes the same way under the same rule and the same fee, at B's mid. One transfer of each is in flight at a time, and coin in flight is still owned, so nothing is unhedged in transit.

### The hand example

`tests/test_arb.py` holds a hand example to the last digit: grid 1 to 5 s, every latency of A 0.5 s and B's 0.5 s, ticks 0.0001, size 100, B's taker fee 10 bps, the quote 30 ticks off B's touch. The buy of 100 at 0.9965 at 2.3 s is hedged at 3.3 s by a sale into the 3.0 s bids, 40 at 0.9988, 50 at 0.9987 and 10 at 0.9986, 99.873; the sale of 100 at 1.0028 at 4.2 s by a purchase at 5.2 s from the 5.0 s asks, 60 at 1.0052 and 40 at 1.0053, 100.524.

| round trip | captured | delay | slippage | fee | net | picked off |
|---|---|---|---|---|---|---|
| bought 2.3 s, hedged 3.3 s | +0.30 | -0.02 | -0.057 | 0.100 | +0.123 | no |
| sold 4.2 s, hedged 5.2 s | -0.17 | -0.02 | -0.054 | 0.100 | -0.344 | yes, and B's touch was through the price |

USD, of 100 USD at 1.0000; the total is $-0.221$ USD, $-2{,}210$ bps of one unit. The stock after each leg: A +100 coin and $-99.65$ USDT, then B $-100$ and $+99.773$, then A $-100$ and $+100.28$, then B $+100$ and $-100.624$. No book row of A reaches a resting price in it, so the book-crossing rule fills nothing and the numbers hold under rules 2. With `hedge_latency_ms: 1500` the first hedge arrives at 4.3 s and walks the 4.1 s bids at 1.0040, a delay of +0.50 and a slippage of $-0.05$, and the second, known at 4.7 s, is sent at the window's end decision and reaches B at 6.2 s, still the 5.0 s row: the same $-0.344$, and a total of $+0.306$.

### The rows

`arb_report` gives `brief`, the worse of RA and PR by the hold rule's Sharpe; `RA`, `PR`, `T` and `F`, each capped at the print; `RA uncapped`, RA with a trade through filling the whole order, the capacity row; `latency 0`; `hedge 1 s`, the scored model with each hedge reaching B a second after its fill (B's order latency 1 s less the response latency); and `hedge fee 10`, `7.5`, `5` and `0`, the scored model's fills and hedges at those hedge fees with A's maker fee 0, arithmetic on one pass ({meth}`~blinded.sim_maker.ArbSim.refee`). On the AT file's validation window (12:05 to 18:00 on 2026-10-01, the quote 30 bps off Binance's touch at 64 AT, about 10 USD), RA fills 190 times; an independent probe, rerun on the same window with the same engine and the print cap, fills 201, and so does `simulate_arb` with a stock too large to bind: the 11 fills between are the stock rule's, which withdrew a side at 3.6% of the decisions and booked 4 rebalances, and the probe has no stock.
