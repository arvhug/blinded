"""A protected run and a naive run on the 15-minute cross-section file, grouped by edge, with the
fake harness.

Usage: python xs_fake.py BRIEF RUNS_DIR

BRIEF is the brief `blinded demo crosssection` wrote beside the cross-section file
(binance-um-xs20-2024-04-01-450d-15m.brief.yaml). Writes RUNS_DIR/xs-fake (protected) and
RUNS_DIR/xs-fake-naive (naive), drives each through the real driver and daemon with the fake
harness from blinded's tests (no model), keeps each session's BUDGET.md and prompt in
driver/budget/, checks the menu against the expected lines and the fill-model line, then
writes both reports and replays them. The ideas: `bl.short_term_reversal` on minus each
name's four-hour return, z-scored across the twenty, through `bl.basket(top=4, every="4h")`;
`bl.statistical_arbitrage` on minus the one-day z-score of each name's log mid less the
basket's mean, through the same basket; `bl.pairs_trading` on ETH over BTC; in the speed
edge `bl.trade_flow_persistence` on BCH's day of taker imbalance and a BTC to BCH lead-lag
gap; and a refused `bl.order_book_imbalance`. Proves the basket mechanism, its Ladder score,
D9 and the basket charts end to end; says nothing about any strategy, since the ideas are
fixed.
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

REV = '''
import polars as pl

import blinded as bl
from blinded import features as F

rev = bl.short_term_reversal()


@rev.feature("the names that fell most over the last four hours bounce")
def rev_score(panel):
    return F.xs_zscore(F.each(panel.legs, lambda s: -F.ret(pl.col(f"{s}.mid"), "4h")))
'''

REV_1D = '''
import polars as pl

import blinded as bl
from blinded import features as F

rev = bl.short_term_reversal()


@rev.feature("the names that fell most over the last day bounce")
def rev_score_1d(panel):
    return F.xs_zscore(F.each(panel.legs, lambda s: -F.ret(pl.col(f"{s}.mid"), "1d")))
'''

RESID = '''
import polars as pl

import blinded as bl
from blinded import features as F

sa = bl.statistical_arbitrage()


@sa.feature("a name far below its day's place in the basket comes back")
def resid_z_1d(panel):
    resid = F.xs_demean(F.each(panel.legs, lambda s: pl.col(f"{s}.mid").log()))
    return {s: -F.zscore(x, "1d") for s, x in resid.items()}
'''

RESID_4H = '''
import polars as pl

import blinded as bl
from blinded import features as F

sa = bl.statistical_arbitrage()


@sa.feature("a name far below its four hours' place in the basket comes back")
def resid_z_4h(panel):
    resid = F.xs_demean(F.each(panel.legs, lambda s: pl.col(f"{s}.mid").log()))
    return {s: -F.zscore(x, "4h") for s, x in resid.items()}
'''

PAIRS = '''
import blinded as bl
from blinded import features as F

pt = bl.pairs_trading(a="ETHUSDT_PERP", b="BTCUSDT_PERP")


@pt.feature("ETH over BTC far above its two days' fit falls back")
def eth_btc_z(panel):
    return -F.spread_z("ETHUSDT_PERP", "BTCUSDT_PERP", "2d", "4h")
'''

FLOW = '''
import blinded as bl
from blinded import features as F

flow = bl.trade_flow_persistence(instrument="BCHUSDT_PERP")


@flow.feature("a day of net taker buying in BCH is a large order still being worked")
def bch_flow_1d(panel):
    return F.trade_imbalance("BCHUSDT_PERP", "1d")
'''

LEAD_LAG = '''
import blinded as bl
from blinded import features as F

lag = bl.lead_lag(leader="BTCUSDT_PERP", follower="BCHUSDT_PERP")


@lag.feature("what BTC did over the last hour and BCH has not")
def btc_bch_gap(panel):
    return F.lead_gap("BTCUSDT_PERP", "BCHUSDT_PERP", "1h")
'''

REFUSED = '''
import blinded as bl


@bl.order_book_imbalance(instrument="BCHUSDT_PERP")
def bch_imbalance(panel, features):
    """Refused on bars: the book has no sizes. Had it run, 1 / 0 would be its reason."""
    return 1 / 0
'''

SHORT = '''
import blinded as bl


@bl.short_term_reversal()
def nineteen(panel, features):
    """A basket that leaves out one leg: an error row with the output rule."""
    return {s: 0.0 * panel.col(s, "mid") for s in panel.legs[:19]}
'''

REV_STRATEGY = '''
import blinded as bl


@bl.short_term_reversal(requires=["rev_score"])
def rev_basket(panel, features):
    return bl.basket(features["rev_score"], top=4, every="4h")
'''

STAT_STRATEGY = '''
import blinded as bl


@bl.statistical_arbitrage(requires=["resid_z_1d"])
def stat_basket(panel, features):
    return bl.basket(features["resid_z_1d"], top=4, every="4h")
'''

PAIRS_STRATEGY = '''
import blinded as bl


@bl.pairs_trading(a="ETHUSDT_PERP", b="BTCUSDT_PERP")
def eth_btc_pairs(panel, features):
    return bl.pairs("ETHUSDT_PERP", "BTCUSDT_PERP", z_in=2.0, z_out=0.5, window="2d", every="4h")
'''

FLOW_STRATEGY = '''
import blinded as bl


@bl.trade_flow_persistence(instrument="BCHUSDT_PERP", requires=["bch_flow_1d"])
def bch_flow(panel, features):
    return bl.threshold(features["bch_flow_1d"], z_in=2.0, z_out=0.5, every="1d")
'''

NAME = "binance-um-xs20-2024-04-01-450d-15m"
# the expected lines, in registry order, with the validation the built file
# gives (110 d 11 h 45 min)
REFUSED_LINES = [
    f"order_book_imbalance (speed & information): not runnable on {NAME}: needs bid_qty; the "
    "book has bid, ask.",
    f"cross_venue_arbitrage (speed & information): not runnable on {NAME}: needs the same "
    "instrument on 2 venues; the file has one.",
    f"basis_trade (carry): not runnable on {NAME}: needs funding data; the file has book, "
    "trades.",
    f"trend_following (underreaction): not runnable on {NAME}: needs 40 weeks of validation "
    "(20 holding periods of 2 weeks); validation is 110 d 11 h 45 min.",
    f"market_making (liquidity & flow): not runnable on {NAME}: needs a maker sim; blinded "
    "fills as a taker and the maker proxy is an upper bound that never enters the Ladder.",
]
COST = ("Fills are a model: the next bar's close plus or minus a half spread of 0.007166 to "
        "0.8076 bps per symbol, measured on data.binance.vision bookTicker 2024-03-27, before "
        "train; no book existed at the fill.")
CAVEAT = ("fills are a model on this file: the next bar's close plus or minus a measured half "
          "spread (DATA.md)")
GRID = ("the grid step 15 min is above the typical holding (100 ms); what the judge measures "
        "is the first whole step after the fill")


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
    # Features, by the equal shares of three edges in their order: speed (BCH's flow, and the
    # refused imbalance strategy), liquidity (the four-hour reversal score), relative value
    # (the one-day residual z and the ETH/BTC z), then a second idea for each (the BTC to BCH
    # lead-lag gap, a one-day reversal, a four-hour residual and a basket that leaves out a
    # leg): each edge returns phase_done in both of its sessions, which closes it. Three
    # sessions in a row without a submission would stop the run as stuck. Strategies: each
    # edge sends its strategies until it has three sessions without a release; then the
    # finalise
    steps = [
        step(("features/flow.py", FLOW), ("strategies/refused.py", REFUSED), phase_done=True),
        step(("features/rev.py", REV), phase_done=True),
        step(("features/resid.py", RESID), ("features/eth_btc.py", PAIRS), phase_done=True),
        step(("features/btc_bch.py", LEAD_LAG), phase_done=True),
        step(("features/rev_1d.py", REV_1D), ("strategies/short.py", SHORT), phase_done=True),
        step(("features/resid_4h.py", RESID_4H), phase_done=True),
    ]
    sessions = [step(("strategies/flow.py", FLOW_STRATEGY)),
                step(("strategies/rev.py", REV_STRATEGY)),
                step(("strategies/stat.py", STAT_STRATEGY),
                     ("strategies/eth_btc.py", PAIRS_STRATEGY))]  # fmt: skip
    steps += sessions * 3
    steps.append(Step(summary=FINAL | {"final_trial_id": None}))
    return Driver(run, Budget(max_iterations=18, max_usd=1.0, wall_hours=2.0),
                  harness=Recording(run, steps, submit_timeout=1800)).run()


def naive(brief: Path, root: Path) -> dict:
    run = driver.start(brief, root, "naive")
    # 8 sessions: four of features (60%: speed, liquidity, relative value, speed), three of
    # strategies (one per edge) and the finalise, which names the reversal basket, trial 5
    steps = [step(("features/flow.py", FLOW), window="train"),
             step(("features/rev.py", REV), window="train"),
             step(("features/resid.py", RESID), window="train"),
             Step(summary=summary([])),
             step(("strategies/flow.py", FLOW_STRATEGY), window="train"),
             step(("strategies/rev.py", REV_STRATEGY), window="train"),
             step(("strategies/stat.py", STAT_STRATEGY), window="train"),
             Step(summary=FINAL | {"final_trial_id": 5})]
    return Driver(run, Budget(max_iterations=8, max_usd=1.0, wall_hours=2.0),
                  harness=Recording(run, steps, submit_timeout=1800)).run()


def check_menu(run: RunDir) -> None:
    """ARCHETYPES.md lists 5 runnable and the five refused lines, exactly, with the
    fill-model line and the cost caveat on every runnable archetype."""
    text = (run.workspace / "ARCHETYPES.md").read_text()
    head = ("5 of 10 archetypes run on this data, in 3 of 5 edges: speed & information, "
            "liquidity & flow, relative value.")
    refused = [line[2:] for line in text.split("## Not runnable on this data", 1)[1].splitlines()
               if line.startswith("- ")]
    assert head in text, text[:400]
    assert COST in text, text[:800]
    assert refused == REFUSED_LINES, refused
    assert text.count(f"- Caveat: {CAVEAT}.") == 5
    assert text.count(f"- Caveat: {GRID}.") == 1
    rules = (run.workspace / "RULES.md").read_text()
    assert "## Bars" in rules and "There is no book inside a bar." in rules
    data = (run.workspace / "DATA.md").read_text()
    assert "## Fills on bars" in data and "The millisecond tables" in data
    assert "The 10 of the 190 pairs with the shortest median half-life are shown." in data
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
    for name, go in (("xs-fake-naive", naive), ("xs-fake", protected)):
        started = time.monotonic()
        state = go(brief, runs / name)
        print(f"{name}: {state.get('status')}, stop {state.get('stop')}, final "
              f"{json.dumps(state.get('final'))}, {time.monotonic() - started:.0f} s", flush=True)
        run = RunDir.open(runs / name)
        check_menu(run)
        describe(run, state)
    for name in ("xs-fake-naive", "xs-fake"):
        started = time.monotonic()
        report(runs / name)
        checked = replay(runs / name)
        bad = [c["name"] for c in checked["checks"] if not c["ok"]]
        print(f"{name}: report and replay in {time.monotonic() - started:.0f} s, "
              f"{len(checked['checks'])} numbers, {'all match' if checked['ok'] else bad[:5]}",
              flush=True)


if __name__ == "__main__":
    main()
