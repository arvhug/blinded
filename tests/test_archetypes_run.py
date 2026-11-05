"""A run grouped by edge, end to end: the planted case of each runnable edge, through the real
driver and daemon with the fake harness, on fourteen synthetic hours where the spot copies the
perp 300 ms late.

Speed is ``bl.lead_lag`` on ``F.lead_gap``; relative value is ``bl.pairs_trading`` with
``bl.pairs``. Both must bind, pass the truncation test and release on their own edge's
Ladder. Beside them go a refused archetype, a bare ``bl.strategy`` and a pairs strategy that
fits its hedge ratio on the whole sample: two error rows that never run and a rejection.
"""

import json
import shutil

import pytest

import blinded as bl
from blinded import vault
from blinded.driver import Budget, Driver
from blinded.report import _naive_table, gather, replay, report
from blinded.vault import RunDir
from tests.fake_harness import FakeHarness, Step, ideas

known = pytest.mark.known_answer

GAP = """
import blinded as bl
from blinded import features as F

lag = bl.lead_lag(leader="XRPUSDT_PERP", follower="XRPUSDT")


@lag.feature("the perp's last second, not yet in the spot")
def gap(panel):
    return F.lead_gap("XRPUSDT_PERP", "XRPUSDT", 1)
"""

REFUSED = """
import blinded as bl


@bl.trend_following(instrument="XRPUSDT_PERP")
def trend(panel, features):
    return 1 / 0


@bl.strategy()
def bare(panel, features):
    return 1 / 0
"""

SPREAD = """
import blinded as bl
from blinded import features as F

pt = bl.pairs_trading(a="XRPUSDT_PERP", b="XRPUSDT")


@pt.feature("the perp-spot log spread far from its walk-forward mean comes back")
def spread_reverts(panel):
    return -F.spread_z("XRPUSDT_PERP", "XRPUSDT", "2m", "2m")
"""

LEAKY = """
import polars as pl

import blinded as bl


@bl.pairs_trading(a="XRPUSDT_PERP", b="XRPUSDT")
def stable_hedge(panel, features):
    \"\"\"The hedge ratio of two legs of one coin hardly moves, so fit it once on the lot.\"\"\"
    y, x = pl.col("XRPUSDT_PERP.mid").log(), pl.col("XRPUSDT.mid").log()
    beta = pl.cov(y, x) / x.var()
    u = -bl.threshold(y - beta * x, z_in=2.0, z_out=0.5, every="2m")
    return {"XRPUSDT_PERP": u, "XRPUSDT": -beta * u}
"""

FOLLOW = """
import blinded as bl


@bl.lead_lag(leader="XRPUSDT_PERP", follower="XRPUSDT", requires=["gap"])
def follow(panel, features):
    return bl.threshold(features["gap"], z_in=2.0, z_out=0.5, every="2m")
"""

PAIRS = """
import blinded as bl


@bl.pairs_trading(a="XRPUSDT_PERP", b="XRPUSDT")
def perp_spot_pairs(panel, features):
    return bl.pairs("XRPUSDT_PERP", "XRPUSDT", z_in=2.0, z_out=0.5, window="2m", every="2m")
"""

TREND_LINE = (
    "trend_following (underreaction): not runnable on synthetic-14h: needs 40 weeks of "
    "validation (20 holding periods of 2 weeks); validation is 3 h 25 min."
)


def files(*pairs, window=None, done=False, names=()):
    return Step(files=list(pairs), submit=[(p, None) for p, _ in pairs], window=window,
                summary={**ideas(*names), "phase_done": done})  # fmt: skip


@pytest.fixture
def run(archetypes_run, tmp_path):
    return RunDir(shutil.copytree(archetypes_run.root, tmp_path / "archetypes"))


@known
def test_a_protected_run_by_edge_finalises_and_replays(run):
    # sessions alternate speed and relative value from equal shares. Each edge returns
    # phase_done in both of its feature sessions, which closes it; 4 is also 60% of 8. Trials:
    # 1 gap, 2 trend, 3 bare, 4 spread_reverts, 5 stable_hedge, 6 follow, 7 perp_spot_pairs;
    # the finalise session names 7.
    steps = [
        files(("features/gap.py", GAP), ("strategies/refused.py", REFUSED), done=True),
        files(("features/spread.py", SPREAD), ("strategies/leaky.py", LEAKY), done=True),
        Step(summary={**ideas(), "phase_done": True}),
        Step(summary={**ideas(), "phase_done": True}),
        files(("strategies/follow.py", FOLLOW)),
        files(("strategies/pairs.py", PAIRS)),
        Step(summary=ideas()),
        Step(summary={"ideas": [], "next": "", "phase_done": True, "final_trial_id": 7}),
    ]
    state = Driver(run, Budget(max_iterations=8, max_usd=1.0),
                   harness=FakeHarness(run, steps)).run()  # fmt: skip
    assert (state["status"], state["stop"]) == ("finished", "finalised")
    sp, rv = "speed", "relative_value"
    assert state["families"] == {str(n): g for n, g in enumerate([sp, rv, sp, rv, sp, rv, sp], 1)}
    assert state["final"] == {"agent_pick": 7, "finalised_by": "agent", "trial_id": 7,
                              "override": None}  # fmt: skip

    ledger = run.ledger()
    rows = {t["name"]: t for t in ledger.trials()}
    assert {n: (t["status"], t["family"], t["edge"], t["archetype"]) for n, t in rows.items()} == {
        "gap": ("accepted", "lead_lag", sp, "lead_lag"),
        "trend": ("error", "directional", "underreaction", "trend_following"),
        "bare": ("error", "directional", None, None),
        "spread_reverts": ("accepted", "pairs", rv, "pairs_trading"),
        "stable_hedge": ("rejected_lookahead", "pairs", rv, "pairs_trading"),
        "follow": ("accepted", "lead_lag", sp, "lead_lag"),
        "perp_spot_pairs": ("accepted", "pairs", rv, "pairs_trading"),
    }
    # the refused archetype never ran: its reason is the menu's line, not ZeroDivisionError
    assert rows["trend"]["reason_short"] == TREND_LINE[:200]
    assert rows["bare"]["reason_short"].startswith("this run studies archetypes")
    assert all(rows[n]["released_bool"] for n in ("gap", "spread_reverts", "follow",
                                                  "perp_spot_pairs"))  # fmt: skip
    assert [(r["phase"], r["group"]) for r in ledger.releases()] == [
        ("feature", sp), ("feature", rv), ("strategy", sp), ("strategy", rv),
    ]  # fmt: skip
    (scored,) = ledger.db.execute("SELECT * FROM test_scores").fetchall()
    tested = json.loads(scored["metrics_json"])["rows"]["brief"]
    assert scored["trial_id"] == 7 and tested["gross_exposure"] > 0
    ledger.close()

    menu = json.loads((run.vault / "menu.json").read_text())
    assert (menu["runnable"], menu["edges"]) == (4, [sp, rv])
    text = (run.workspace / "ARCHETYPES.md").read_text()
    assert ("4 of 11 archetypes run on this data, in 2 of 5 edges: speed & information, "
            "relative value.") in text  # fmt: skip
    refused = [line for line in text.splitlines() if line.startswith("- ") and "not runnable"
               in line]  # fmt: skip
    assert len(refused) == 7 and f"- {TREND_LINE}" in refused
    rules = (run.workspace / "RULES.md").read_text()
    assert "## Archetypes" in rules and "## Families" not in rules

    report(run.root)
    saved = json.loads((run.root / "report.json").read_text())
    assert saved["archetypes"]["menu"]["runnable"] == 4
    by = {f["family"]: f for f in saved["families"]}
    assert set(by) == {sp, rv}
    assert by[sp]["members"] == ["lead_lag", "order_book_imbalance", "trade_flow_persistence"]
    assert (by[sp]["feature"]["submitted"], by[sp]["strategy"]["submitted"]) == (1, 1)
    assert by[rv]["strategy"] == {"submitted": 2, "accepted": 1, "released": 1, "best": 10.0}
    assert by[sp]["sessions"] == {"features": 2, "strategies": 2}
    assert by[rv]["sessions"] == {"features": 2, "strategies": 1}
    assert (by[sp]["final"], by[rv]["final"]) == (False, True)
    line = saved["verdict"]["line"]
    assert (
        "pairs_trading strategy perp_spot_pairs on XRPUSDT/XRPUSDT_PERP (relative value)" in line
    )
    page = (run.root / "report.html").read_text()
    assert "<th>edge</th><th>members</th>" in page

    # per edge a row for each runnable archetype, with what was planted in it
    rows = {r["archetype"]: r for e in saved["archetypes"]["edges"] for r in e["archetypes"]}
    assert list(rows) == ["lead_lag", "order_book_imbalance", "trade_flow_persistence",
                          "pairs_trading"]  # fmt: skip

    def count(name, kind):
        return tuple(rows[name][kind][k] for k in ("submitted", "accepted", "released"))

    assert count("lead_lag", "feature") == count("lead_lag", "strategy") == (1, 1, 1)
    assert count("order_book_imbalance", "feature") == (0, 0, 0)
    # the leaky stable_hedge and the released perp_spot_pairs, which holds the final
    assert count("pairs_trading", "strategy") == (2, 1, 1) and rows["pairs_trading"]["final"]
    # N is the five strategies' hashes: trend, bare, stable_hedge, follow, perp_spot_pairs;
    # trend is the one refused archetype, booked to underreaction, and bare has no edge
    trials = saved["archetypes"]["trials"]
    assert (trials["n"], trials["refused_errors"], trials["groups"]) == (5, 1, 2)
    assert trials["by_edge"] == {sp: 1, rv: 2}
    assert trials["releases"] == {sp: {"feature": 1, "strategy": 1},
                                  rv: {"feature": 1, "strategy": 1}}  # fmt: skip
    folded = page.split("What this data cannot test (7)</summary><ul>", 1)[1].split("</ul>")[0]
    assert folded.count("<li>") == 7 and "trend_following (underreaction)" in folded

    # section 4 by edge then archetype, each with its death checks on every segment
    diag = saved["recompute"]["diagnostics"]
    assert diag["segments"] == ["train", "val", "test"]
    assert {e: list(m) for e, m in diag["archetypes"].items()} == {
        sp: ["lead_lag", "order_book_imbalance", "trade_flow_persistence"],
        rv: ["pairs_trading"],
    }
    pairs = diag["archetypes"][rv]["pairs_trading"]
    assert list(pairs["dies"]) == ["mae", "half_life", "costs", "capacity", "worst_window"]
    assert set(pairs["dies"]["mae"]) == {"train", "val", "test"}
    # the panel's D1 grid at the brief's latency and fee is the verdict's test row
    grid = pairs["dies"]["costs"]["test"]["grid"]
    cell = grid["net_bps_per_hour"][grid["latencies_ms"].index(50.0)][grid["fees"].index("brief")]
    assert cell == pytest.approx(saved["test"]["rows"]["brief"]["net_bps_per_hour"])
    lead = diag["archetypes"][sp]["lead_lag"]
    assert lead["dies"]["race"]["c_table"][0]["leader"] == "XRPUSDT_PERP"
    assert lead["dies"]["race"]["val"]["peak_rank_ic"] > 0
    nothing = diag["archetypes"][sp]["order_book_imbalance"]["dies"]
    assert nothing["costs"] == {"note": "no accepted strategy of this archetype"}
    assert page.count("How it dies.") == 4

    checked = replay(run.root)
    assert checked["ok"], [c for c in checked["checks"] if not c["ok"]][:5]
    names = {c["name"] for c in checked["checks"]}
    assert {"archetypes.menu.trend_following.line", "archetypes.menu.power"} <= names
    assert "diagnostics.archetypes.relative_value.pairs_trading.dies.costs.test.traded" in names


@known
def test_a_naive_twin_picks_from_either_edge_and_is_rescored(archetypes_brief, tmp_path):
    run = vault.init(archetypes_brief, tmp_path / "naive", mode="naive")
    # 60% of 4 iterations is 2: sessions 1 (speed) and 2 (relative value) are features; 3 is
    # strategies, for speed, where the agent submits a pairs strategy anyway (judged, and
    # booked to its own edge); 4 finalises on it, trial 3
    steps = [
        files(("features/gap.py", GAP), window="train"),
        files(("features/spread.py", SPREAD), window="train"),
        files(("strategies/pairs.py", PAIRS), window="all"),
        Step(summary={"ideas": [], "next": "", "phase_done": True, "final_trial_id": 3}),
    ]
    state = Driver(run, Budget(max_iterations=4), harness=FakeHarness(run, steps)).run()
    assert state["status"] == "finished" and state["final"]["trial_id"] == 3
    assert state["families"] == {"1": "speed", "2": "relative_value", "3": "speed"}
    ledger = run.ledger()
    pick = ledger.trial(3)
    assert (pick["name"], pick["edge"], pick["archetype"]) == ("perp_spot_pairs",
                                                               "relative_value",
                                                               "pairs_trading")  # fmt: skip
    # claimed at zero latency on the whole file; rescored by the protected judge on the test
    claimed = json.loads(pick["train_metrics_json"])["rows"]["brief"]
    assert claimed["latency_ms"] == 0
    tested = json.loads(ledger.test_score(run.run_id)["metrics_json"])["rows"]["brief"]
    assert tested["latency_ms"] == 50 and tested["fills"] > 0
    ledger.close()
    assert "| edge | archetype |" in (run.workspace / "LEDGER.md").read_text()
    model = gather(run)
    line = "pairs_trading strategy perp_spot_pairs on XRPUSDT/XRPUSDT_PERP (relative value)"
    assert line in model["verdict"]["line"]
    # the calls of each archetype: the gap feature, then the spread feature and the strategy
    assert _naive_table(run)["calls_by_archetype"] == {"lead_lag": 1, "pairs_trading": 2}


@known
def test_local_eval_honours_the_menu_and_carries_the_tags(archetypes_run, tmp_path):
    brief = archetypes_run.workspace / "brief.yaml"
    (tmp_path / "gap.py").write_text(GAP)
    (tmp_path / "refused.py").write_text(REFUSED)
    (gap,) = bl.load(tmp_path / "gap.py").features
    result = bl.evaluate(gap, brief, window="train")
    assert result.ok and (result.archetype, result.edge, result.legs) == ("lead_lag", "speed",
                                                                         ("XRPUSDT",))  # fmt: skip
    assert "lead_lag (speed & information), lead_lag on XRPUSDT" in repr(result)
    trend, bare = bl.load(tmp_path / "refused.py").strategies
    assert bl.evaluate_strategy(trend, brief).reason == TREND_LINE
    assert bl.evaluate_strategy(bare, brief).reason.startswith("this run studies archetypes")
