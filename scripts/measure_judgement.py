"""Peak RSS and wall time of one judgement on the test segment, measured as the daemon does.

Usage: python measure_judgement.py BRIEF RUN_DIR [REPEATS]

Builds RUN_DIR from BRIEF if it is not there, writes the reference pairs strategy
(bl.pairs on ETH over BTC, 2h fit, 30m refit, z 2 / 0.5) beside it, and judges it
REPEATS times (default 6) on the test segment in a spawned child, exactly the job
score_test sends: train panel, truncation test, train metrics, the warm test panel
(the whole file), its truncation test and the score. The parent reads the child's
RSS with `ps` every 0.1 s, as run_isolated does, and keeps the peak.

On a brief with archetypes the strategy is the same pair as `bl.pairs_trading`, at the
brief's own pairs_trading window and refit (on the hourly trend file, 1w and 1d), since
a bare `bl.strategy` is an error row there and never runs. On the cross-venue file, where
pairs trading is refused, it is `bl.cross_venue_arbitrage` on the declared pair with
`bl.pairs` at that archetype's parameters (10m, 10m); on a file with a basket (twenty
names), the heaviest judgement there is: `bl.short_term_reversal` on minus each name's
four-hour return z-scored across the legs, through `bl.basket(top=4, every="4h")`, the
feature computed inside the judgement as the daemon computes a required feature. On a file
with carry pairs, `bl.basis_trade` on the first declared pair: `bl.carry` at the brief's
basis_trade parameters, requiring a feature on the perpetual's last settled funding rate,
so the judgement reads the funding stream, scores the two legs and accrues funding. On a
file with a touch stream (`data.touch`, the market-making files of `blinded demo mm` and
`blinded demo arb`), `bl.market_making` on the touch's instrument at the brief's
market_making parameters (join the touch, size 1, skew 1, max inventory 2): the judgement
reads the vault's touch files and runs the fill engine under its four models and the
latency-0 pass on train and on test. On an arbitrage file whose brief lets the arbitrage
run (a depth file, the pair declared, the stock's withdrawal fee set), `bl.cross_venue_making`
at 30 bps off the hedge venue's touch at the brief's size: the judgement reads the vault's
touch and depth files and runs the hedge leg under the four models, uncapped, at latency 0
and with a 1 s hedge, on train and on test.
"""

import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

from blinded import vault
from blinded.daemon import _worker

STRATEGY = '''
import blinded as bl


@bl.strategy(family="pairs", pair=("ETHUSDT_PERP", "BTCUSDT_PERP"))
def eth_btc_pairs(panel, features):
    return bl.pairs("ETHUSDT_PERP", "BTCUSDT_PERP", z_in=2.0, z_out=0.5, window="2h", every="30m")
'''


ARCHETYPE_STRATEGY = '''
import blinded as bl


@bl.pairs_trading(a="ETHUSDT_PERP", b="BTCUSDT_PERP")
def eth_btc_pairs(panel, features):
    return bl.pairs("ETHUSDT_PERP", "BTCUSDT_PERP", z_in=2.0, z_out=0.5, window="{window}",
                    every="{every}")
'''


VENUE_STRATEGY = '''
import blinded as bl


@bl.cross_venue_arbitrage(a="{a}", b="{b}")
def venue_pairs(panel, features):
    return bl.pairs("{a}", "{b}", z_in=2.0, z_out=0.5, window="{window}", every="{every}")
'''

BASKET_STRATEGY = '''
import polars as pl

import blinded as bl
from blinded import features as F

rev = bl.short_term_reversal()


@rev.feature("the names that fell most over the last four hours bounce")
def rev_score(panel):
    return F.xs_zscore(F.each(panel.legs, lambda s: -F.ret(pl.col(f"{s}.mid"), "4h")))


@rev.strategy(requires=["rev_score"])
def rev_basket(panel, features):
    return bl.basket(features["rev_score"], top=4, every="4h")
'''

CARRY_STRATEGY = '''
import polars as pl

import blinded as bl

carry = bl.basis_trade(spot="{spot}", perp="{perp}")


@carry.feature("a high last funding rate pays a short perpetual over the next 8 hours")
def last_rate(panel):
    return pl.col("{perp}.rate")


@carry.strategy(requires=["last_rate"])
def basis_carry(panel, features):
    return bl.carry("{spot}", "{perp}", rate_in={rate_in}, rate_out={rate_out})
'''


MAKER_STRATEGY = '''
import blinded as bl

mm = bl.market_making(instrument="{inst}")


@mm.strategy()
def join_touch(panel, features):
    return bl.quote(bid=0, ask=0, size={size}, skew={skew}, max_inventory={max_inventory})
'''


ARB_STRATEGY = '''
import polars as pl

import blinded as bl
from blinded import features as F

arb = bl.cross_venue_making(quote="{quote}", hedge="{hedge}")


@arb.strategy()
def thirty_bps(panel, features):
    bid, ask = pl.col("{hedge}.bid"), pl.col("{hedge}.ask")
    k = (30e-4 * F.mid("{hedge}") / {tick} - 1e-9).ceil()
    return bl.quote(bid=k, ask=k, size={size}, skew=0, max_inventory=1e9, fair=(bid, ask))
'''


def rss(pid: int) -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    text = out.stdout.strip()
    return int(text) * 1024 if text.isdigit() else 0


def judge_once(run: vault.RunDir, code: Path, name: str) -> tuple[float, int, dict]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    job = {"code_path": str(code), "name": name, "accepted": {}, "segment": "test"}
    proc = ctx.Process(target=_worker, args=(child, str(run.root), job), daemon=True)
    started = time.monotonic()
    proc.start()
    child.close()
    peak, result = 0, None
    while result is None:
        if parent.poll(0.1):
            kind, value = parent.recv()
            if kind == "done":
                result = value
        peak = max(peak, rss(proc.pid))
        if not proc.is_alive() and not parent.poll():
            result = {"crash": proc.exitcode}
    wall = time.monotonic() - started
    proc.join(timeout=10)
    return wall, peak, result


def main() -> None:
    brief, root = Path(sys.argv[1]), Path(sys.argv[2])
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    run = vault.RunDir.open(root) if (root / "vault").is_dir() else vault.init(brief, root)
    brief = run.brief()
    spec = brief.archetypes
    name = "eth_btc_pairs"
    from blinded import archetypes

    if brief.data.depth is not None and spec is not None and archetypes.arb_lack(brief) is None:
        name = "thirty_bps"
        quote, hedge = archetypes.arb_pair(brief)
        params = archetypes.REGISTRY["cross_venue_making"].params | spec.params_for(
            "cross_venue_making")
        source = ARB_STRATEGY.format(quote=quote, hedge=hedge, tick=brief.execution.tick_size,
                                     size=params["size"])
    elif brief.data.touch is not None:
        name = "join_touch"
        params = spec.params_for("market_making") if spec is not None else {
            "size": 1.0, "skew": 1.0, "max_inventory": 2.0}
        source = MAKER_STRATEGY.format(inst=brief.data.touch.instrument, **params)
    elif spec is not None and spec.cross_venue:
        name = "venue_pairs"
        (a, b), params = spec.cross_venue[0], spec.params_for("cross_venue_arbitrage")
        source = VENUE_STRATEGY.format(a=a, b=b, **params)
    elif spec is not None and spec.carry:
        name = "basis_carry"
        (spot, perp), params = spec.carry[0], spec.params_for("basis_trade")
        source = CARRY_STRATEGY.format(spot=spot, perp=perp, **params)
    elif brief.family("basket") is not None:
        name, source = "rev_basket", BASKET_STRATEGY
    elif spec is not None:
        source = ARCHETYPE_STRATEGY.format(**spec.params_for("pairs_trading"))
    else:
        source = STRATEGY
    code = root / f"{name}.py"
    code.write_text(source)
    for i in range(repeats):
        wall, peak, result = judge_once(run, code, name)
        item = next((x for x in result.get("items") or [] if x.get("name") == name), {})
        test = item.get("test_summary") or {}
        print(f"repeat {i + 1}: {wall:5.1f} s, peak RSS {peak / 1024**3:.2f} GB, "
              f"{item.get('status', result)}, test Sharpe {test.get('score')}, "
              f"net {test.get('net_bps_per_hour')} bps/h", flush=True)


if __name__ == "__main__":
    main()
