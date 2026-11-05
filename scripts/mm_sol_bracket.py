"""The SOL bracket in docs/sim.md, recomputed with blinded's fill engine.

    uv run python scripts/mm_sol_bracket.py [--built] [--hour12]
    uv run python scripts/mm_sol_bracket.py --run RUN_DIR

with BLINDED_LOCAL_DATA set. Reads the Tardis raw book_ticker and trades of the SOL perpetual on
2026-10-01 from data/tardis/raw on the receive clock (with --built, the touch file and the
trade rows of the market-making file `blinded demo mm` wrote under data/tardis/mm/solusdt
instead; with --hour12, only 12:00 to 13:00; with --run, through a run directory built from
that file, as the judge reads it, asserting the hour-12 counts), and runs the hand
example's policy at
size 10 (join the touch, no bid while a full fill would take the inventory past ten) on the
100 ms grid: 50 ms orders and cancels, fills known at once, maker 2 bps, taker 5. P&L is in
bps of one unit of the quote's size.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from blinded import sim_maker as mm

LOCAL = Path(os.environ.get("BLINDED_LOCAL_DATA", "."))
RAW = LOCAL / "data/tardis/raw"


def touch() -> mm.Touch:
    b = pl.read_csv(RAW / "binance-futures_book_ticker_2026-10-01_SOLUSDT.csv.gz",
                    columns=["local_timestamp", "ask_amount", "ask_price", "bid_price",
                             "bid_amount"])  # fmt: skip
    x = pl.read_csv(RAW / "binance-futures_trades_2026-10-01_SOLUSDT.csv.gz",
                    columns=["local_timestamp", "side", "price", "amount"])  # fmt: skip
    b = b.filter((pl.col("ask_price") > pl.col("bid_price")) & (pl.col("bid_price") > 0))
    b = b.sort("local_timestamp", maintain_order=True)
    x = x.sort("local_timestamp", maintain_order=True)
    return mm.Touch(
        0.01, b["local_timestamp"].to_numpy() * 1000,
        mm.to_ticks(b["bid_price"].to_numpy(), 0.01), mm.to_ticks(b["ask_price"].to_numpy(), 0.01),
        b["bid_amount"].to_numpy().astype(float), b["ask_amount"].to_numpy().astype(float),
        x["local_timestamp"].to_numpy() * 1000, mm.to_ticks(x["price"].to_numpy(), 0.01),
        x["amount"].to_numpy().astype(float), np.where(x["side"].to_numpy() == "buy", 1, -1))


BUILT = LOCAL / "data/tardis/mm/solusdt"
STEM = "tardis-binance-solusdt-perp-mm-2026-10-01"


def built() -> mm.Touch:
    """The same stream from the built files: the touch file's rows and the data file's trades."""
    b = pl.read_parquet(BUILT / f"{STEM}.touch.parquet")
    x = pl.read_parquet(BUILT / f"{STEM}.parquet", columns=["ts", "kind", "side", "price", "qty"])
    x = x.filter(pl.col("kind") == "trades")
    return mm.Touch(
        0.01, b["ts"].dt.epoch("ns").to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy(),
        b["bid_qty"].to_numpy(), b["ask_qty"].to_numpy(), x["ts"].dt.epoch("ns").to_numpy(),
        mm.to_ticks(x["price"].to_numpy(), 0.01), x["qty"].to_numpy(),
        np.where(x["side"].cast(pl.String).to_numpy() == "B", 1, -1))


# the hour-12 fill counts the engine gives from the raw files since the book-crossing fill;
# before it they were T 248, RA 1,011, PR 1,829, F 8,739
HOUR12 = {"T": 248, "RA": 851, "PR": 1750, "F": 8677}


def through_run(root: str) -> None:
    """The hour-12 bracket as the judge computes it: a run directory's validation panel, its
    touch read from the vault's touch files for the window, sim_maker.models at the run's
    latency with fills known at once; asserts the counts of HOUR12."""
    import dataclasses

    from blinded import sim_maker, vault
    from blinded.brief import Window

    run = vault.RunDir.open(root)
    brief = run.brief()
    ex = dataclasses.replace(brief.execution, response_latency=pd.Timedelta(0),
                             maker_fee_bps=2.0, taker_fee_bps=5.0)  # fmt: skip
    brief = dataclasses.replace(brief, execution=ex)
    panel = run.panel("val")
    start = pd.Timestamp("2026-10-01 12:00", tz="UTC")
    win = Window("hour 12", start, start + pd.Timedelta("1h"))
    n = len(panel)
    quotes = mm.Quotes({"bid": np.zeros(n), "ask": np.zeros(n), "size": np.full(n, 10.0),
                        "skew": np.zeros(n), "max_inventory": np.full(n, 10.0)})  # fmt: skip
    sims = sim_maker.models(quotes, panel, win, brief, brief.target.instrument)
    got = {m: sims[m].metrics()["fills"] for m in sim_maker.MODELS}
    print("through", root, got)
    assert got == HOUR12, got


def main() -> None:
    import sys

    if "--run" in sys.argv:
        through_run(sys.argv[sys.argv.index("--run") + 1])
        return
    t = built() if "--built" in sys.argv else touch()
    day = pd.Timestamp("2026-10-01", tz="UTC").value
    step = 100_000_000
    spans = ((12, 13, mm.MODELS), (2, 3, mm.MODELS), (0, 24, ("T", "RA")))
    for h0, h1, models in spans[:1] if "--hour12" in sys.argv else spans:
        grid = np.arange(day + h0 * 3600 * 10**9, day + h1 * 3600 * 10**9, step, dtype=np.int64)
        n = len(grid)
        quotes = mm.Quotes({"bid": np.zeros(n), "ask": np.zeros(n), "size": np.full(n, 10.0),
                            "skew": np.zeros(n), "max_inventory": np.full(n, 10.0),
                            "lean": np.zeros(n)})  # fmt: skip
        dec = mm.Decisions.make(quotes, grid, t, 0.01)
        end = int(grid[-1] + step)
        for model in models:
            started = time.time()
            book = mm.run_engine(t, dec, end, model, 50_000_000, 50_000_000, 0)
            seconds = time.time() - started
            s = mm.account(book, t, grid, end, step, pd.Timedelta("50ms"), model, 2.0, 5.0,
                           pd.Timedelta("5min"))  # fmt: skip
            k, d = s.metrics(), mm.decompose(s)
            rebate = s.refee(-0.5).metrics()
            print(f"{h0:02d}-{h1:02d} {model}: fills {k['fills']} units/10 {k['units'] / 10:.0f}"
                  f" captured/fill {k['captured_bps'] / 10 / k['fills']:+.2f}"
                  f" markout 10 s/fill {d['adverse_bps']['10s'] / 10 / k['fills']:+.2f}"
                  f" gross/h {k['gross_bps_per_hour'] / 10:.0f}"
                  f" net/h {k['net_bps_per_hour'] / 10:.0f}"
                  f" rebate 0.5 {rebate['net_bps_per_hour'] / 10:.0f}"
                  f" Sharpe {k['sharpe_per_sqrt_hour']:.2f} ({len(s.blocks)} blocks), "
                  f"{seconds:.1f} s")  # fmt: skip


if __name__ == "__main__":
    main()
