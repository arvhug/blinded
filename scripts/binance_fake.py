"""A protected run and a naive run on the Binance file with the scripted fake harness.

Usage: python binance_fake.py BRIEF RUNS_DIR

Writes RUNS_DIR/binance-fake (protected) and RUNS_DIR/binance-fake-naive (naive),
drives each through the real driver and daemon with the fake harness from
blinded's tests (no model), then writes both reports and replays them. Proves the
file, the brief and the three families run end to end; says nothing about any
strategy, since the ideas are fixed.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fake_harness import FakeHarness, Step  # noqa: E402

from blinded import driver  # noqa: E402
from blinded.driver import Budget, Driver  # noqa: E402
from blinded.report import replay, report  # noqa: E402

DIRECTIONAL = '''
import blinded as bl
from blinded import features as F


@bl.feature("book imbalance on the XRP perp leans the next move its way")
def xrp_imbalance(panel):
    return F.imbalance("XRPUSDT_PERP")
'''

LEAD_LAG = '''
import blinded as bl
from blinded import features as F


@bl.feature("what BTC did in the last second and SOL has not", family="lead_lag",
            target="SOLUSDT_PERP")
def btc_sol_gap(panel):
    return F.lead_gap("BTCUSDT_PERP", "SOLUSDT_PERP", "1s")
'''

PAIRS = '''
import blinded as bl
from blinded import features as F


@bl.feature("ETH over BTC far above its walk-forward mean falls back", family="pairs",
            pair=("ETHUSDT_PERP", "BTCUSDT_PERP"))
def eth_btc_z(panel):
    return -F.spread_z("ETHUSDT_PERP", "BTCUSDT_PERP", "2h", "30m")
'''

DIRECTIONAL_STRATEGY = '''
import blinded as bl


@bl.strategy(requires=["xrp_imbalance"])
def xrp_imbalance_threshold(panel, features):
    return bl.threshold(features["xrp_imbalance"], z_in=2.0, z_out=0.5)
'''

LEAD_LAG_STRATEGY = '''
import blinded as bl


@bl.strategy(requires=["btc_sol_gap"], family="lead_lag", trades="SOLUSDT_PERP")
def sol_follows_btc(panel, features):
    return bl.threshold(features["btc_sol_gap"], z_in=3.0, z_out=0.5)
'''

PAIRS_STRATEGY = '''
import blinded as bl


@bl.strategy(family="pairs", pair=("ETHUSDT_PERP", "BTCUSDT_PERP"))
def eth_btc_pairs(panel, features):
    return bl.pairs("ETHUSDT_PERP", "BTCUSDT_PERP", z_in=2.0, z_out=0.5, window="2h", every="30m")
'''

FEATURES = {"directional": ("features/xrp.py", DIRECTIONAL),
            "lead_lag": ("features/btc_sol.py", LEAD_LAG),
            "pairs": ("features/eth_btc.py", PAIRS)}
STRATEGIES = {"directional": ("strategies/xrp.py", DIRECTIONAL_STRATEGY),
              "lead_lag": ("strategies/sol.py", LEAD_LAG_STRATEGY),
              "pairs": ("strategies/eth_btc.py", PAIRS_STRATEGY)}


def summary(names, **extra):
    return {"ideas": [{"name": n, "trial_ids": [], "rationale": n} for n in names],
            "next": "more"} | extra


def step(files, window=None, **extra):
    names = [Path(p).stem for p, _ in files]
    return Step(files=list(files), submit=[(p, None) for p, _ in files], window=window,
                summary=summary(names, **extra))


def protected(brief: Path, root: Path) -> dict:
    run = driver.start(brief, root, "protected")
    # 12 sessions: the features phase gives each family two sessions (D, L, P, P, D, L by
    # the shares 0.3/0.3/0.4), each sending its feature and phase_done; the strategies phase
    # sends all three strategies every session; the last session finalises on the pair
    feats = [FEATURES[f] for f in ("directional", "lead_lag", "pairs", "pairs", "directional",
                                   "lead_lag")]
    steps = [step([f], phase_done=True) for f in feats]
    steps += [step(STRATEGIES.values()) for _ in range(5)]
    steps.append(Step(summary={"ideas": [], "next": "", "phase_done": True,
                               "final_trial_id": None}))
    return Driver(run, Budget(max_iterations=12, max_usd=1.0, wall_hours=2.0),
                  harness=FakeHarness(run, steps, submit_timeout=1800)).run()


def naive(brief: Path, root: Path) -> dict:
    run = driver.start(brief, root, "naive")
    # 6 sessions: three of features (60%), two of strategies, and the finalise, which names
    # the pairs strategy, trial 4, as its own pick
    steps = [step([FEATURES[f]], window="train") for f in ("directional", "lead_lag", "pairs")]
    steps += [step([STRATEGIES["pairs"]], window="train"),
              step([STRATEGIES["directional"], STRATEGIES["lead_lag"]], window="val")]
    steps.append(Step(summary={"ideas": [], "next": "", "phase_done": True, "final_trial_id": 4}))
    return Driver(run, Budget(max_iterations=6, max_usd=1.0, wall_hours=2.0),
                  harness=FakeHarness(run, steps, submit_timeout=1800)).run()


def main() -> None:
    brief, runs = Path(sys.argv[1]), Path(sys.argv[2])
    for name, go in (("binance-fake-naive", naive), ("binance-fake", protected)):
        started = time.monotonic()
        state = go(brief, runs / name)
        print(f"{name}: {state.get('status')}, stop {state.get('stop')}, final "
              f"{json.dumps(state.get('final'))}, {time.monotonic() - started:.0f} s", flush=True)
        print(f"  families by session: {state.get('families')}", flush=True)
    for name in ("binance-fake-naive", "binance-fake"):
        started = time.monotonic()
        report(runs / name)
        checked = replay(runs / name)
        bad = [c["name"] for c in checked["checks"] if not c["ok"]]
        print(f"{name}: report and replay in {time.monotonic() - started:.0f} s, "
              f"{len(checked['checks'])} numbers, {'all match' if checked['ok'] else bad[:5]}",
              flush=True)


if __name__ == "__main__":
    main()
