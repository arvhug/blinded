# Run reports

Each pair is the same agent, Claude Sonnet at medium effort, on the same budget of up to 20 sessions: one run behind the protections and one with them off. Everything in these reports comes from a simulator replaying recorded public data. The reports are as the runs wrote them, and each should be read against the judging rules its run was held to ({doc}`report`). The protected taker run is the third attempt: the first two stopped early on false sandbox probes, before any test score.

| run | data | protected | naive |
|---|---|---|---|
| Taker strategies at 1 bps | 3 days of Binance BTC, ETH, SOL and XRP perps, 27 to 29 Mar 2024 | [report](https://arvhug.github.io/blinded/reports/taker-1bps-protected.html) | [report](https://arvhug.github.io/blinded/reports/taker-1bps-naive.html) |
| Market making at a 2 bps maker fee | 1 day of the Binance SOLUSDT perp, 1 Oct 2026 | [report](https://arvhug.github.io/blinded/reports/sol-maker-protected.html) | [report](https://arvhug.github.io/blinded/reports/sol-maker-naive.html) |
| Quoting on one venue, hedging on another | 1 day of AT spot on MEXC and Binance, 1 Oct 2026 | [report](https://arvhug.github.io/blinded/reports/at-arb-protected.html) | [report](https://arvhug.github.io/blinded/reports/at-arb-naive.html) |
