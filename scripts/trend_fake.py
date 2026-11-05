"""A protected and a naive run on the hourly trend file, grouped by edge, with the fake harness.

Usage: python trend_fake.py BRIEF RUNS_DIR

BRIEF is the brief `blinded demo trend` wrote beside the trend file
(binance-um-trend4-2023-05-17-1233d-1h.brief.yaml). Writes RUNS_DIR/trend-fake (protected)
and RUNS_DIR/trend-fake-naive (naive), drives each through the real driver and daemon with
the fake harness from blinded's tests (no model), keeps each session's BUDGET.md and prompt
in driver/budget/, checks the menu against the expected lines and the fill-model line,
then writes both reports and replays them. The ideas: `bl.trend_following` on BTC's
four-week return through `bl.linear(every="1w")`, `bl.trade_flow_persistence` on LTC's
taker imbalance over a day, `bl.pairs_trading` on ETH over BTC, and a refused
`bl.order_book_imbalance`. Proves the bar layout, the cost model, three Ladders by edge and
the death checks end to end; says nothing about any strategy, since the ideas are fixed.
"""

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fake_harness import FakeHarness, Step  # noqa: E402

from blinded import driver  # noqa: E402
from blinded.daemon import Daemon  # noqa: E402
from blinded.driver import Budget, Driver  # noqa: E402
from blinded.report import replay, report  # noqa: E402
from blinded.vault import RunDir  # noqa: E402

TREND = '''
import polars as pl

import blinded as bl
from blinded import features as F

tf = bl.trend_following(instrument="BTCUSDT_PERP")


@tf.feature("BTC's return over the last four weeks carries on into the next week")
def btc_trend_4w(panel):
    return F.ret(pl.col("BTCUSDT_PERP.mid"), "4w")
'''

FLOW = '''
import blinded as bl
from blinded import features as F

flow = bl.trade_flow_persistence(instrument="LTCUSDT_PERP")


@flow.feature("a day of net taker buying in LTC is a large order still being worked")
def ltc_flow_1d(panel):
    return F.trade_imbalance("LTCUSDT_PERP", "1d")
'''

PAIRS = '''
import blinded as bl
from blinded import features as F

pt = bl.pairs_trading(a="ETHUSDT_PERP", b="BTCUSDT_PERP")


@pt.feature("ETH over BTC far above its week's fit falls back")
def eth_btc_z(panel):
    return -F.spread_z("ETHUSDT_PERP", "BTCUSDT_PERP", "1w", "1d")
'''

LEAD_LAG = '''
import blinded as bl
from blinded import features as F

lag = bl.lead_lag(leader="BTCUSDT_PERP", follower="LTCUSDT_PERP")


@lag.feature("what BTC did over the last day and LTC has not")
def btc_ltc_gap(panel):
    return F.lead_gap("BTCUSDT_PERP", "LTCUSDT_PERP", "1d")
'''

PAIRS_SLOW = '''
import blinded as bl
from blinded import features as F

pt = bl.pairs_trading(a="ETHUSDT_PERP", b="BTCUSDT_PERP")


@pt.feature("ETH over BTC far above its two weeks' fit falls back")
def eth_btc_z_2w(panel):
    return -F.spread_z("ETHUSDT_PERP", "BTCUSDT_PERP", "2w", "1d")
'''

TREND_SLOW = '''
import polars as pl

import blinded as bl
from blinded import features as F

tf = bl.trend_following(instrument="BTCUSDT_PERP")


@tf.feature("BTC's return over the last twelve weeks carries on into the next week")
def btc_trend_12w(panel):
    return F.ret(pl.col("BTCUSDT_PERP.mid"), "12w")
'''

REFUSED = '''
import blinded as bl


@bl.order_book_imbalance(instrument="BTCUSDT_PERP")
def btc_imbalance(panel, features):
    """Refused on bars: the book has no sizes. Had it run, 1 / 0 would be its reason."""
    return 1 / 0
'''

TREND_STRATEGY = '''
import blinded as bl


@bl.trend_following(instrument="BTCUSDT_PERP", requires=["btc_trend_4w"])
def btc_trend(panel, features):
    return bl.linear(features["btc_trend_4w"], every="1w")
'''

FLOW_STRATEGY = '''
import blinded as bl


@bl.trade_flow_persistence(instrument="LTCUSDT_PERP", requires=["ltc_flow_1d"])
def ltc_flow(panel, features):
    return bl.threshold(features["ltc_flow_1d"], z_in=2.0, z_out=0.5, every="1w")
'''

PAIRS_STRATEGY = '''
import blinded as bl


@bl.pairs_trading(a="ETHUSDT_PERP", b="BTCUSDT_PERP")
def eth_btc_pairs(panel, features):
    return bl.pairs("ETHUSDT_PERP", "BTCUSDT_PERP", z_in=2.0, z_out=0.5, window="1w", every="1d")
'''

NAME = "binance-um-trend4-2023-05-17-1233d-1h"
# the expected lines, in registry order
REFUSED_LINES = [
    f"order_book_imbalance (speed & information): not runnable on {NAME}: needs bid_qty; the "
    "book has bid, ask.",
    f"cross_venue_arbitrage (speed & information): not runnable on {NAME}: needs the same "
    "instrument on 2 venues; the file has one.",
    f"statistical_arbitrage (relative value): not runnable on {NAME}: needs a cross-section of "
    "at least 20 names; the file has 4.",
    f"short_term_reversal (liquidity & flow): not runnable on {NAME}: needs a cross-section of "
    "at least 20 names; the file has 4.",
    f"basis_trade (carry): not runnable on {NAME}: needs funding data; the file has book, "
    "trades.",
    f"market_making (liquidity & flow): not runnable on {NAME}: needs a maker sim; blinded "
    "fills as a taker and the maker proxy is an upper bound that never enters the Ladder.",
]
COST = ("Fills are a model: the next bar's close plus or minus a half spread of 0.01851 to "
        "1.185 bps per symbol, measured on data.binance.vision bookTicker 2023-05-16, before "
        "train; no book existed at the fill.")
CAVEAT = ("fills are a model on this file: the next bar's close plus or minus a measured half "
          "spread (DATA.md)")
GRID = ("the grid step 1 h is above the typical holding (100 ms); what the judge measures is "
        "the first whole step after the fill")


class Recording(FakeHarness):
    """The fake harness, keeping each session's BUDGET.md and prompt in driver/budget/."""

    def run(self, prompt, workspace, limits, log=None):
        keep = self.rundir.driver / "budget"
        keep.mkdir(exist_ok=True)
        n = self.calls + 1
        shutil.copyfile(Path(workspace) / "BUDGET.md", keep / f"BUDGET-{n:02d}.md")
        (keep / f"prompt-{n:02d}.txt").write_text(prompt + "\n")
        return super().run(prompt, workspace, limits, log)


def summary(names, **extra):
    return {"ideas": [{"name": n, "trial_ids": [], "rationale": n} for n in names],
            "next": "more"} | extra


def step(*files, window=None, **extra):
    names = [Path(p).stem for p, _ in files]
    return Step(files=list(files), submit=[(p, None) for p, _ in files], window=window,
                summary=summary(names, **extra))


FINAL = {"ideas": [], "next": "", "phase_done": True}


def protected(brief: Path, root: Path) -> dict:
    run = driver.start(brief, root, "protected")
    # Features, by the equal shares of three edges: speed (the flow feature, and the refused
    # imbalance strategy), relative value (the ETH/BTC z), underreaction (the trend feature),
    # then a second idea for each (the BTC/LTC lead-lag gap, a two-week ETH/BTC fit, a
    # twelve-week trend): each edge returns phase_done in both of its sessions, which closes
    # it. Three sessions in a row without a submission would stop the run as stuck.
    # Strategies: each edge sends its strategy until it has three sessions without a release;
    # then the finalise
    steps = [
        step(("features/flow.py", FLOW), ("strategies/refused.py", REFUSED), phase_done=True),
        step(("features/eth_btc.py", PAIRS), phase_done=True),
        step(("features/trend.py", TREND), phase_done=True),
        step(("features/btc_ltc.py", LEAD_LAG), phase_done=True),
        step(("features/eth_btc_2w.py", PAIRS_SLOW), phase_done=True),
        step(("features/trend_12w.py", TREND_SLOW), phase_done=True),
    ]
    sessions = [step(("strategies/flow.py", FLOW_STRATEGY)),
                step(("strategies/eth_btc.py", PAIRS_STRATEGY)),
                step(("strategies/trend.py", TREND_STRATEGY))]
    steps += sessions * 3
    steps.append(Step(summary=FINAL | {"final_trial_id": None}))
    return Driver(run, Budget(max_iterations=18, max_usd=1.0, wall_hours=2.0),
                  harness=Recording(run, steps, submit_timeout=1800)).run()


def naive(brief: Path, root: Path) -> dict:
    run = driver.start(brief, root, "naive")
    # 8 sessions: four of features (60%: speed, relative value, underreaction, speed), three
    # of strategies (one per edge) and the finalise, which names the trend strategy, trial 6
    steps = [step(("features/flow.py", FLOW), window="train"),
             step(("features/eth_btc.py", PAIRS), window="train"),
             step(("features/trend.py", TREND), window="train"),
             Step(summary=summary([])),
             step(("strategies/flow.py", FLOW_STRATEGY), window="train"),
             step(("strategies/eth_btc.py", PAIRS_STRATEGY), window="train"),
             step(("strategies/trend.py", TREND_STRATEGY), window="train"),
             Step(summary=FINAL | {"final_trial_id": 6})]
    return Driver(run, Budget(max_iterations=8, max_usd=1.0, wall_hours=2.0),
                  harness=Recording(run, steps, submit_timeout=1800)).run()


def check_menu(run: RunDir) -> None:
    """ARCHETYPES.md lists 4 runnable and the six refused lines, exactly, with the
    fill-model line and the cost caveat on every runnable archetype."""
    text = (run.workspace / "ARCHETYPES.md").read_text()
    head = ("4 of 10 archetypes run on this data, in 3 of 5 edges: speed & information, "
            "relative value, underreaction.")
    refused = [line[2:] for line in text.split("## Not runnable on this data", 1)[1].splitlines()
               if line.startswith("- ")]
    assert head in text, text[:400]
    assert COST in text, text[:800]
    assert refused == REFUSED_LINES, refused
    assert text.count(f"- Caveat: {CAVEAT}.") == 4
    assert text.count(f"- Caveat: {GRID}.") == 1
    rules = (run.workspace / "RULES.md").read_text()
    assert "## Bars" in rules and "There is no book inside a bar." in rules
    data = (run.workspace / "DATA.md").read_text()
    assert "## Fills on bars" in data and "The millisecond tables" in data
    menu = run.menu()
    print(f"  menu: {menu.headline} runnable {list(menu.runnable)}", flush=True)
    print(f"  power: {menu.power}", flush=True)


def describe(run: RunDir, state: dict) -> None:
    print(f"  edges by session: {state.get('families')}", flush=True)
    ledger = run.ledger()
    for t in ledger.trials():
        print(f"  trial {t['trial_id']} {t['kind']} {t['name']}: {t['status']}, {t['edge']}, "
              f"{t['archetype']}, released {t['released_bool']}, best {t['ladder_best_after']}"
              + (f", {t['reason_short']}" if t["status"] != "accepted" else ""), flush=True)
    print(f"  releases by group: {[(r['phase'], r['group']) for r in ledger.releases()]}",
          flush=True)
    print(f"  N strategies for deflation: {ledger.n_trials('strategy')}", flush=True)
    ledger.close()
    if run.mode == "protected":
        ladders = Daemon(run).ladders
        print("  ladders: " + ", ".join(f"{k}/{g} floor {lad.floor:g} best {lad.best}"
                                        for (k, g), lad in sorted(ladders.items())), flush=True)


def main() -> None:
    brief, runs = Path(sys.argv[1]), Path(sys.argv[2])
    for name, go in (("trend-fake-naive", naive), ("trend-fake", protected)):
        started = time.monotonic()
        state = go(brief, runs / name)
        print(f"{name}: {state.get('status')}, stop {state.get('stop')}, final "
              f"{json.dumps(state.get('final'))}, {time.monotonic() - started:.0f} s", flush=True)
        run = RunDir.open(runs / name)
        check_menu(run)
        describe(run, state)
    for name in ("trend-fake-naive", "trend-fake"):
        started = time.monotonic()
        report(runs / name)
        checked = replay(runs / name)
        bad = [c["name"] for c in checked["checks"] if not c["ok"]]
        print(f"{name}: report and replay in {time.monotonic() - started:.0f} s, "
              f"{len(checked['checks'])} numbers, {'all match' if checked['ok'] else bad[:5]}",
              flush=True)


if __name__ == "__main__":
    main()
