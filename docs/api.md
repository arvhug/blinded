# API reference

`import blinded as bl` exposes `bl.feature`, `bl.strategy`, `bl.load`, `bl.evaluate`, `bl.evaluate_strategy`, `bl.linear`, `bl.threshold`, `bl.pairs`, `bl.basket`, `bl.quote`, `bl.Panel`, `bl.Window`, `bl.load_brief`, the eleven archetype decorators ({doc}`archetypes`) and the `bl.features` library, whose functions are documented on the {doc}`feature library <features>` page. The rest is documented below under the modules that define it. A feature returns a polars expression over `panel.frame`, a polars Series or a pandas Series; {func}`blinded.contract.values` is where the three become one array, and where a strategy's mapping of instrument to position becomes {class}`~blinded.contract.Legs`.

## Command line

```text
blinded eval FILE --brief BRIEF [--window train|val|test|all|START/END] [--name NAME] [--latency-ms N]
blinded data --brief BRIEF [-o DATA.md]
blinded init BRIEF --run-dir DIR
blinded run BRIEF --run-dir DIR [--mode protected|naive] [--resume RUN_ID] [--model M] [--effort E] [--hours H] [--max-iterations N] [--max-usd U]
blinded daemon DIR [--once]
blinded report DIR
blinded replay DIR
blinded submit FILE [--name NAME] [--timeout SECONDS]
blinded test DIR --trial ID
blinded mlflow DIR [--uri URI]
blinded demo tardis --date YYYY-MM-01 --symbols SYM[,SYM] --out DIR [--hours H] [--book book_ticker|quotes]
blinded demo binance --from YYYY-MM-DD [--days N] [--symbols SYM,...] --out DIR [--step 500ms] [--latency-ms 50] [--delay-ms 50] [--keep-raw] [--force]
blinded demo trend --out DIR [--raw DIR] [--costs DIR] [--keep-raw]
blinded demo crosssection --out DIR [--raw DIR] [--costs DIR] [--keep-raw]
blinded demo crossvenue --out DIR [--raw DIR]
blinded demo carry --out DIR [--raw DIR] [--costs DIR] [--tardis-raw DIR] [--keep-raw]
blinded demo mm --symbol SYM --date YYYY-MM-01 --out DIR [--raw DIR] [--levels 5] [--latency-ms 50] [--hours H]
blinded demo arb --out DIR [--coin ATUSDT] [--date YYYY-MM-01] [--quote mexc] [--hedge binance] [--raw DIR]
blinded demo venues --out DIR [--coin SOLUSDT] [--date YYYY-MM-01] [--raw DIR]
blinded record --out DIR [--coin ATUSDT] [--hours H]
blinded forward RUN_DIR (--live-data DIR | --segment test) [--latency-ms N] [--watch]
blinded forward serve RUN_DIR [--port 8766]
```

## Features and strategies

```{eval-rst}
.. automodule:: blinded.contract
   :members: feature, strategy, Feature, Strategy, bind, Binding, Legs, Quotes, ContractError, output_rule, load, Module, scan, scan_feature, Scan, run, values, extend, ALLOWED_IMPORTS
```

## Archetypes

```{eval-rst}
.. automodule:: blinded.archetypes
   :members: lead_lag, order_book_imbalance, trade_flow_persistence, pairs_trading, cross_venue_arbitrage, statistical_arbitrage, short_term_reversal, basis_trade, trend_following, market_making, Bound, Archetype, Edge, Needs, Facts, Verdict, Menu, facts, check, line, menu, power, render, example, touch_lack, touch_line, maker_facts
```

## Evaluation

```{eval-rst}
.. automodule:: blinded.evaluate
   :members: evaluate, Result, evaluate_strategy, StrategyResult, positions, find_features, scoring_window, score, score_xs, score_for, panel_for
```

## Panel

```{eval-rst}
.. automodule:: blinded.panel
   :members: Panel, Events, load_events, make_grid
```

## Run directory

```{eval-rst}
.. automodule:: blinded.vault
   :members: init, RunDir, split_rows, git_root, InsideGitRepo, check_cost_model, check_touch, check_depth
```

## Sim

```{eval-rst}
.. automodule:: blinded.sim
   :members: simulate, Sim, simulate_legs, MultiSim, held, report, maker_proxy, linear, threshold, pairs, basket, carry, accrue, borrowed, funding_of, borrow_of, walk_forward_scale
```

## Fill engine

```{eval-rst}
.. automodule:: blinded.sim_maker
   :members: quote, simulate_maker, MakerSim, report, models, worse, decompose, run_engine, Decisions, Book, account, Touch, TouchSource, touch_of, touch_from, infer_tick, to_ticks, simulate_arb, ArbSim, account_arb, arb_models, arb_report, arb_fills, arb_table, StockGate, stock_path, hedge_stamps, Depth, DepthSource, depth_of, Mark, mark_of
```

## Fees

```{eval-rst}
.. automodule:: blinded.fees
   :members: tiers, tier, Tier, venue_of, per_step, verdict, judge, judge_arb, words
```

## Theo

```{eval-rst}
.. automodule:: blinded.theo
   :members: TheoParams, TheoSpec, run, nll, fit, stream, innovations, build, refilter, Live
```

## Daemon and client

```{eval-rst}
.. automodule:: blinded.daemon
   :members: Daemon, Ladders, Limits, run_isolated, score_test

.. automodule:: blinded.isolate
   :members: Pool, Worker, Isolated, Spec, Failed, Limit, describe, declared, payload

.. automodule:: blinded.client
   :members: submit, find_judge
```

## Sandbox

```{eval-rst}
.. automodule:: blinded.sandbox
   :members: write, command, protected, deny_rules, record_denials, environment, aliases

.. automodule:: blinded.hook
   :members: decide, deny, main

.. automodule:: blinded.transcript
   :members: record, refusals, reaches, tool_calls, events, Refusal
```

## Demo data

```{eval-rst}
.. automodule:: blinded.demo
   :members: tardis, download, build, book_rows, trades, brief_for, summary, binance, Built, BuildError, fetch, fetch_url, read_archive, raw_checks, gap_check, read_points, thin_book, thin_trades, thin_symbol_day, Carry, identity_check, binance_brief, estimate_rows, fine_lead_lag, raw_notes, notes_until, trend, crosssection, bars, BarFile, BuiltBars, trend_brief, xs_brief, bar_notes, read_klines, bar_rows, bar_gaps, missing_check, half_spreads, touch_prices, head_sizes, kline_url, book_ticker_url, months, four_figures, crossvenue, BuiltVenues, Venue, venue_problem, venue_frame, venue_cadence, cadence_line, crossvenue_brief, venue_notes, carry, BuiltCarry, carry_brief, carry_notes, funding_url, read_funding, funding_rows, spot_book_problem, tardis_half_spreads, borrow_history, borrow_rows
```

```{eval-rst}
.. automodule:: blinded.demo_mm
   :members: mm, BuiltMM, mm_brief, arb, venues, BuiltVenueFile, Leg, listing_problem, read_ticker, read_trades, read_snapshot, funding_settlements, lattice, levels_asof, touch_layout, depth_layout, identity, snapshot_check, day_facts
```

```{eval-rst}
.. automodule:: blinded.bookstats
   :members: touch_facts, falls, aggressors, delay_ms, snapshot_agreement, venue_facts, cross_facts, flow_facts, tw_quantile, time_share
```

## Driver

```{eval-rst}
.. automodule:: blinded.driver
   :members: Driver, Budget, Status, Standing, start, rescore
```

## Prompts

```{eval-rst}
.. automodule:: blinded.prompts
   :members: rules, families_section, archetypes_section, iteration, SCHEMA, SYSTEM_APPEND
```

## Harness

```{eval-rst}
.. automodule:: blinded.harness.base
   :members: Harness, SessionLimits, IterationResult

.. automodule:: blinded.harness.claude_cli
   :members: ClaudeCLI, parse, stop
```

## Report

```{eval-rst}
.. automodule:: blinded.report
   :members: report, replay, gather, recompute, render
```

## Death checks

```{eval-rst}
.. automodule:: blinded.dies
   :members: breakeven_fee_bps, touch_shares, window_nets, worst_window, underwater_steps, max_drawdown, signal_half_life, run_paths, leg_paths, mae, breadth, breadths, whipsaw, parts, latencies, costs, race, capacity, worst_windows, half_life, funding, whipsaws, venue_feed
```

## Markouts

```{eval-rst}
.. automodule:: blinded.markouts
   :members: events, clusters, fan, mid_moves, spread_moves, entry_cost, window_ics, half_spread_bps, rank_ic, downsample, window_half_lives, random_walk_half_life
```

## MLflow mirror

```{eval-rst}
.. automodule:: blinded.tracking
   :members: mirror, default_uri, ui_command
```

## Deflation

```{eval-rst}
.. automodule:: blinded.deflate
   :members: deflated_sharpe, expected_max_sharpe, probabilistic_sharpe, moments, pbo, PBO, NotEnough, summary
```

## Ladder

```{eval-rst}
.. automodule:: blinded.ladder
   :members: Ladder, Decision, Score, BlockDraw, rank_ic_score, xs_rank_ic_score, sharpe_score, step, round_to, floor
```

## Ledger

```{eval-rst}
.. automodule:: blinded.ledger
   :members: Ledger, code_hash, render, AlreadyScored
```

## Time spans

```{eval-rst}
.. automodule:: blinded.durations
   :members: to_ns, polars_span, timedelta, label, words
```

## Brief

```{eval-rst}
.. automodule:: blinded.brief
   :members: load, parse, windows, window, default_embargo, Brief, Family, Archetypes, Group, Window, Data, Stream, Execution, Thinned, CostModel, Grid
```

## Truncation test

```{eval-rst}
.. automodule:: blinded.truncation
   :members: check, precheck, cut_times, compare, Verdict, Violation
```

## Metrics

```{eval-rst}
.. automodule:: blinded.metrics
   :members:
```

## Targets

```{eval-rst}
.. automodule:: blinded.target
   :members:
```

## Clock diagnostics

```{eval-rst}
.. automodule:: blinded.diagnostics
   :members: stream_table, staleness, xcorr_lag, touch_alignment, lag_table, write_data_md, lead_lag_curve, lead_lag_lags, lead_lag_summary, lead_lag, pairs_table
```

## Forward paper

```{eval-rst}
.. automodule:: blinded.recorder
   :members: Recording, Columns, Mexc, Binance, LocalBook, Gap, Recorder, record, build_files, read_parts, hour_statuses, clock_sample, clock_summary, check_clock, ClockError, day_latencies

.. automodule:: blinded.proto
   :members: schema, decode, encode, wrapper, DecodeError

.. automodule:: blinded.forward
   :members: run_forward, watch, judge, Judgement, Day, segment_day, live_day, Final, final_of, forward_brief, panel_of, quotes_of, hourly, reference, cusum, false_alarms, threshold, verdict, own_interval, own_line, judged_totals, calibration_line, ForwardLedger, ForwardError, page, serve
```
