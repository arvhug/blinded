"""Market making end to end, on the two-hour synthetic file where the spot copies the perp
300 ms late: a planted maker strategy through the real driver and daemon with the fake
harness, protected and naive, then the report's bracket of fill models and the replay.

The brief runs market making alone (``only``), on the spot, at the example brief's 50 ms and
no maker fee, with 1-minute blocks so that validation's 25 minutes hold 25 of them. The
planted strategy joins the touch and pulls the ask while the perp is above the spot's touch,
the bid while it is below: the quote the spot's book is about to move through, which the
book crossing would fill at the old price. Joining the touch on both sides loses under
both queue models; pulling the quote in the way releases under both. (The first plant leaned the
quotes a tick towards the perp instead; under the book-crossing fill its validation Sharpe
fell from 3.19 to -6.33.)
"""

import functools
import json
import re
import shutil

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from blinded import archetypes, sim_maker, vault
from blinded import report as report_mod
from blinded.driver import Budget, Driver
from blinded.report import gather, replay, report
from blinded.vault import RunDir
from tests.fake_harness import FakeHarness, Step, ideas

known = pytest.mark.known_answer
edge = pytest.mark.edge_case

LEAD = """
import blinded as bl
from blinded import features as F

mm = bl.market_making(instrument="XRPUSDT")


@mm.feature("the perp's mid ahead of the spot's says where the spot goes next")
def lead(panel):
    return (F.mid("XRPUSDT_PERP") / F.mid("XRPUSDT") - 1) * 1e4
"""

WRONG = """
import blinded as bl

mm = bl.market_making(instrument="XRPUSDT")


@mm.strategy()
def position(panel, features):
    \"\"\"A position where quotes belong.\"\"\"
    return bl.threshold(panel.col("XRPUSDT", "mid"), z_in=2.0, z_out=0.5)
"""

QUOTES = """
import polars as pl

import blinded as bl
from blinded import features as F

mm = bl.market_making(instrument="XRPUSDT")


@mm.strategy()
def join(panel, features):
    return bl.quote(bid=0, ask=0, size=1.0, skew=0.5, max_inventory=5.0)


@mm.strategy(requires=["lead"])
def lean(panel, features):
    # no ask while the perp is above the spot's touch, no bid while it is below: pull the quote
    # the spot's book is about to move through
    half = F.spread("XRPUSDT", bps=True) / 2
    off = pl.lit(None, dtype=pl.Float64)
    bid = pl.when(features["lead"] < -half).then(off).otherwise(0.0)
    ask = pl.when(features["lead"] > half).then(off).otherwise(0.0)
    return bl.quote(bid=bid, ask=ask, size=1.0, skew=0.5, max_inventory=5.0)
"""

RULE = ("a market-making strategy returns bl.quote(...), a mapping with the keys bid, ask, "
        "size, skew, max_inventory and, optionally, lean and fair")  # fmt: skip


@pytest.fixture(scope="module")
def maker_brief(long_brief, tmp_path_factory):
    raw = yaml.safe_load(long_brief.read_text())
    raw["name"] = "synthetic-2h-mm"
    raw["stats"]["block"] = "1min"
    raw["archetypes"] = {"instruments": ["XRPUSDT", "XRPUSDT_PERP"], "only": ["market_making"]}
    path = tmp_path_factory.mktemp("maker") / "brief.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture(scope="module")
def maker_run(maker_brief, tmp_path_factory):
    return vault.init(maker_brief, tmp_path_factory.mktemp("runs") / "maker")


def files(*pairs, window=None, done=False, final=None):
    summary = {**ideas(), "phase_done": done}
    if final is not None:
        summary |= {"final_trial_id": final}
    return Step(files=list(pairs), submit=[(p, None) for p, _ in pairs], window=window,
                summary=summary)  # fmt: skip


@known
def test_the_menu_runs_market_making_with_the_books_facts(maker_run):
    text = (maker_run.workspace / "ARCHETYPES.md").read_text()
    assert "1 of 11 archetypes run on this data, in 1 of 5 edges: liquidity & flow." in text
    assert "### market_making: Market making" in text
    facts = json.loads((maker_run.vault / "menu.json").read_text())["facts"]["maker"]
    # one tick of 0.0001, read from the book, against the spot's median mid on the train
    # grid, and the share of train grid points one tick wide (the generator draws one tick
    # three times in four), recomputed with pandas from the training rows
    frame = pd.read_parquet(maker_run.workspace / "data" / "train.parquet")
    assert facts["tick"] == 0.0001 and not facts["tick_declared"]
    brief = maker_run.brief()
    panel = maker_run.panel("train")
    start = maker_run.windows()["train"].start + brief.grid.warmup
    rows = panel.frame.filter(pl.col("ts") >= start).select("XRPUSDT.bid", "XRPUSDT.ask")
    bid, ask = rows["XRPUSDT.bid"].to_numpy(), rows["XRPUSDT.ask"].to_numpy()
    assert facts["tick_bps"] == pytest.approx(1e-4 / np.median((bid + ask) / 2) * 1e4)
    assert facts["one_tick_share"] == pytest.approx(np.mean(np.rint((ask - bid) / 1e-4) == 1))
    assert 0.7 < facts["one_tick_share"] < 0.8
    spot = frame[(frame.symbol == "XRPUSDT") & ~frame.is_trade]
    assert spot.bid0_px.round(4).diff().abs().replace(0, np.nan).min() == pytest.approx(1e-4)
    assert f"one tick (0.0001, read from the book) is {facts['tick_bps']:.3g} bps" in text
    # RULES.md says what a maker strategy returns and how it is scored, where market making
    # runs, and only there
    rules = " ".join((maker_run.workspace / "RULES.md").read_text().split())
    assert ("A market-making strategy returns quotes, not positions: `bl.quote(...)` on the "
            "panel.") in rules  # fmt: skip
    assert "the Ladder scores the worse of the two queue models" in rules


@known
def test_a_planted_maker_strategy_releases_and_is_scored_on_test(maker_run, tmp_path):
    run = RunDir(shutil.copytree(maker_run.root, tmp_path / "maker"))
    # one group, liquidity. 60% of 6 is 3 feature sessions; the second phase_done in a row
    # closes features after session 2. Trials: 1 lead, 2 position, 3 join, 4 lean; sessions 4
    # and 5 add nothing, and session 6 names trial 4.
    steps = [
        files(("features/lead.py", LEAD), ("strategies/wrong.py", WRONG), done=True),
        Step(summary={**ideas(), "phase_done": True}),
        files(("strategies/quotes.py", QUOTES)),
        Step(summary=ideas()),
        Step(summary=ideas()),
        Step(summary={"ideas": [], "next": "", "phase_done": True, "final_trial_id": 4}),
    ]
    harness = FakeHarness(run, steps)
    state = Driver(run, Budget(max_iterations=6, max_usd=1.0), harness=harness).run()
    assert (state["status"], state["stop"]) == ("finished", "finalised")
    assert state["final"]["trial_id"] == 4

    ledger = run.ledger()
    rows = {t["name"]: t for t in ledger.trials()}
    assert {n: (t["status"], t["edge"], t["archetype"]) for n, t in rows.items()} == {
        "lead": ("accepted", "liquidity", "market_making"),
        "position": ("error", "liquidity", "market_making"),
        "join": ("accepted", "liquidity", "market_making"),
        "lean": ("accepted", "liquidity", "market_making"),
    }
    # a maker strategy that returns a position never reaches the engine
    assert rows["position"]["reason_short"] == RULE[:200]
    assert [bool(rows[n]["released_bool"]) for n in ("lead", "join", "lean")] == [True, False,
                                                                                 True]  # fmt: skip
    # the Ladder scored the worse of RA and PR; the ledger keeps all four models
    for name in ("join", "lean"):
        val = json.loads(rows[name]["val_metrics_json"])
        models = val["models"]
        assert set(models) == set(sim_maker.MODELS)
        # by the hold rule's Sharpe: join's holds outlast the 1-minute block under the book
        # crossing, so its blocks are its own and not the brief's
        ra, pr = (models[m]["held"]["sharpe"] for m in ("RA", "PR"))
        assert val["fill_model"] == ("PR" if pr < ra else "RA")
        assert val["metrics"]["fill_model"] == val["fill_model"]
        assert val["metrics"]["rows"]["brief"]["fills"] == models[val["fill_model"]]["fills"]
        # each model is its own closed loop, so the nesting of a fixed quote sequence need not
        # hold: under the book crossing join's T pass fills more than RA (1,274 against 1,222)
        # on validation; F fills most
        fills = {m: models[m]["fills"] for m in sim_maker.MODELS}
        assert max(fills.values()) == fills["F"], name
    lean = json.loads(rows["lean"]["val_metrics_json"])
    assert lean["value"] == pytest.approx(min(
        lean["models"][m]["held"]["sharpe"] for m in ("RA", "PR")))  # fmt: skip
    assert lean["value"] > 2.0
    (scored,) = ledger.db.execute("SELECT * FROM test_scores").fetchall()
    tested = json.loads(scored["metrics_json"])
    assert scored["trial_id"] == 4 and tested["fill_model"] in ("RA", "PR")
    assert tested["rows"]["brief"]["latency_ms"] == 50.0
    ledger.close()
    # of validation the agent saw the truncation verdict, released and best, never a model;
    # its train rows show all four, as RULES.md says
    public = [t for r in harness.submitted for t in r.get("trials", [])
              if t["name"] in ("join", "lean")]  # fmt: skip
    assert [sorted(t["validation"]) for t in public] == [["best", "released", "truncation"]] * 2
    assert all(set(t["train"]["rows"]) >= set(sim_maker.MODELS) for t in public)

    report(run.root)
    # the fee verdict is the report's alone: on the page and in report.json, never in a file the
    # agent's sandbox reads (the workspace: RULES.md, LEDGER.md, ARCHETYPES.md, DATA.md, the
    # menu) nor in any daemon response, protected or naive
    assert '"fee_verdict"' in (run.root / "report.json").read_text()
    assert "Fee verdict:" in (run.root / "report.html").read_text()
    words = ("fee_verdict", "Fee verdict", "fee-limited", "edge-limited",
             "consistently profitable", "f_star", "a strategy built for a rebate")  # fmt: skip
    seen = [json.dumps(r, default=str) for r in harness.submitted]
    for path in run.workspace.rglob("*"):
        if path.is_file() and path.suffix in (".md", ".json", ".txt", ".py", ".yaml"):
            seen.append(path.read_text(errors="ignore"))
    assert seen and not [w for w in words for text in seen if w in text]
    saved = json.loads((run.root / "report.json").read_text())
    # judged under the current rules, the page says so and nothing more
    assert saved["judge_rules"] == 3 and saved["verdict"]["rules"] == "Judged under rules v3."
    assert not [k for k in saved["recompute"] if k.startswith("rules_v")]
    assert "rules_current" not in saved["verdict"]
    assert "<p class=plain>Judged under rules v3.</p>" in (run.root / "report.html").read_text()
    # the same run as one judged under rules 1 (its manifest without judge_rules), and as one
    # judged under rules 2: each page adds what the current rules show for the final, numbers
    # equal to the v3 run's own judgement of it, and replay recomputes them; the v1 page also
    # keeps what rules 2 show, as published
    final = saved["recompute"]["final"]
    want = {"judge_rules": 3, "fill_model": final["fill_model"],
            "net_bps_per_hour": saved["recompute"]["test_rows"]["brief"]["net_bps_per_hour"],
            "fills": saved["recompute"]["test_rows"]["brief"]["fills"],
            "sharpe_per_sqrt_hour": final["held"]["sharpe"]}  # fmt: skip
    queue = {"RA": "risk-averse", "PR": "pro-rata"}[want["fill_model"]]
    net, sr = report_mod._bps_text(want["net_bps_per_hour"]), want["sharpe_per_sqrt_hour"]
    current = (
        f"Under the current rules (v3: {report_mod.RULES_V3}) this final's locked-test "
        f"net is {net} bps/hour (Sharpe {sr:+.2f}, under the {queue} queue model)."
    )
    for was in (1, 2):
        old = RunDir(shutil.copytree(run.root, tmp_path / f"maker-v{was}"))
        manifest = json.loads((old.vault / "manifest.json").read_text())
        if was == 1:
            del manifest["judge_rules"]
        else:
            manifest["judge_rules"] = 2
        (old.vault / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # the ledger's copy of the manifest names the rules too: replay checks one against
        # the other (RunDir.verify) before it recomputes
        import sqlite3

        db = sqlite3.connect(old.ledger_path)
        for rid, text in db.execute("SELECT run_id, split_manifest_json FROM runs").fetchall():
            m = json.loads(text) | {"judge_rules": was}
            if was == 1:
                del m["judge_rules"]
            db.execute("UPDATE runs SET split_manifest_json = ? WHERE run_id = ?",
                       (json.dumps(m), rid))  # fmt: skip
        db.commit()
        db.close()
        assert old.judge_rules == old.brief().judge_rules == was
        report(old.root)
        v = json.loads((old.root / "report.json").read_text())
        assert v["judge_rules"] == was and v["recompute"]["rules_v3"] == pytest.approx(want)
        assert v["verdict"]["rules_current"] == current
        page = (old.root / "report.html").read_text()
        assert f"<p class=plain>{report_mod.esc(current)}</p>" in page
        assert ("rules_v2" in v["recompute"]) is (was == 1)
        if was == 1:
            assert v["verdict"]["rules"].startswith(
                "Judged under rules v1. Under rules v2 (20 Oct 2026: a book crossing fills a "
                "resting quote, and every step counts in a block Sharpe) this final's "
                "locked-test net is ")  # fmt: skip
            assert v["recompute"]["rules_v2"]["judge_rules"] == 2
        else:
            assert v["verdict"]["rules"] == "Judged under rules v2."
        names = {c["name"] for c in replay(old.root)["checks"] if c["name"].startswith("rules_v")}
        keys = ("judge_rules", "net_bps_per_hour", "fills", "sharpe_per_sqrt_hour", "fill_model")
        want_names = {f"rules_v3.{k}" for k in keys}
        if was == 1:
            want_names |= {f"rules_v2.{k}" for k in keys}
        assert names == want_names
        checks = {c["name"]: c for c in replay(old.root)["checks"]
                  if c["name"].startswith("rules_v")}  # fmt: skip
        assert all(c["ok"] for c in checks.values())
        if was == 1:
            v1 = v
    # under rules 1 no book row fills, so D12 has no crossed share
    path = ("recompute", "diagnostics", "archetypes", "liquidity", "market_making")
    d12 = [json.dumps(functools.reduce(dict.get, path, r)) for r in (saved, v1)]
    assert '"crossed_share"' in d12[0] and '"crossed_share"' not in d12[1]
    line = saved["verdict"]["line"]
    assert "market_making strategy lean on XRPUSDT (liquidity & flow)" in line
    test_rows = saved["test"]["rows"]
    assert {"brief", "RA", "PR", "T", "F", "latency 0", "rebate 0.5"} <= set(test_rows)
    assert saved["recompute"]["final"]["fill_model"] == tested["fill_model"]
    assert test_rows["brief"] == test_rows[tested["fill_model"]]
    entry = saved["recompute"]["diagnostics"]["archetypes"]["liquidity"]["market_making"]
    bracket = entry["maker"]
    assert bracket["name"] == "lean" and list(bracket["segments"]) == ["train", "val", "test"]
    val = bracket["segments"]["val"]
    assert val["scored"] == lean["fill_model"]
    assert val["models"]["RA"]["fills"] == lean["models"]["RA"]["fills"]
    # the page's bracket and the score agree: the scored model, the Sharpe the score used and
    # its block, each model's by the same rule
    for m in ("RA", "PR"):
        assert val["models"][m]["held"]["sharpe"] == lean["models"][m]["held"]["sharpe"]
        assert val["models"][m]["held"]["block_steps"] == lean["models"][m]["held"]["block_steps"]
    assert val["models"][val["scored"]]["held"]["sharpe"] == pytest.approx(lean["value"])
    assert entry["feature"]["name"] == "lead"
    page = (run.root / "report.html").read_text()
    assert page.count("(scored)") == 3 and "T and F bracket the fill count" in page
    marked = f"<td>{val['scored']} (scored)</td>"
    row = page.split("<p class=plain>validation</p>")[1].split(marked)[1].split("</tr>")[0]
    assert f"<b>{lean['value']:.2f}</b>" in row and "25 of 1 min" in row
    assert f"On validation the rule took {val['scored']}, the lower of" in page
    # the three charts and the maker's death checks, D12 and D13 a column per fill model
    maker = page.split("The strategy: <code>lean</code>")[1].split("5. Failed ideas")[0]
    for title in ("the half hour with the most fills", "After the fill: the markout of each",
                  "Where the P&amp;L of lean goes"):  # fmt: skip
        assert title in maker, title
    assert maker.count("<svg") == 3
    for title in ("D1: is the edge", "D3: how much of the touch", "D4: when it lost",
                  "D6: does it diverge", "D7: what carry paid or cost: no funding",
                  "D12: are the fills the ones nobody wanted: validation",
                  "D13: where in the queue you were: test"):  # fmt: skip
        assert title in maker, title
    dies = entry["dies"]
    assert set(dies["toxicity"]["val"]) == set(sim_maker.MODELS)
    tox = dies["toxicity"]["val"]["RA"]
    shares = tox["through_share"] + tox["crossed_share"] + tox["inside_share"] + tox["queue_share"]
    assert shares == pytest.approx(1.0)
    assert tox["crossed_share"] > 0  # the synthetic spot's book moves through resting quotes
    assert dies["queue"]["val"]["T"]["mean_ahead_sizes"] is None  # T never knows its queue
    assert dies["queue"]["val"]["F"]["mean_ahead_sizes"] == 0.0
    assert dies["costs"]["val"]["breakeven_fee_bps"] == pytest.approx(
        dies["costs"]["val"]["gross_bps"] / dies["costs"]["val"]["traded"])  # fmt: skip

    checked = replay(run.root)
    assert checked["ok"], [c for c in checked["checks"] if not c["ok"]][:5]
    names = {c["name"] for c in checked["checks"]}
    at = "diagnostics.archetypes.liquidity.market_making."
    for name in ("maker.segments.val.models.PR.fills", "maker.segments.val.models.RA.held.sharpe",
                 "maker.segments.test.fan.F.mean_bps[19]", "maker.excerpt.inventory[0]",
                 "maker.segments.val.pnl.net[0]", "maker.segments.val.decomposition.T.units",
                 "dies.toxicity.val.RA.adverse_10s_bps_per_unit", "dies.queue.test.PR.p90_wait_s",
                 "dies.costs.val.grid.net_bps_per_hour[1][4]",
                 "dies.capacity.train.median_touch_share"):  # fmt: skip
        assert at + name in names, name
    assert "archetypes.menu.facts.maker.captured_bps_per_unit" in names


@known
def test_a_naive_twin_claims_at_zero_latency_and_is_rescored(maker_brief, maker_run, tmp_path,
                                                            monkeypatch):  # fmt: skip
    # the menu's train pass reads the training panel in either mode, as the replay does: the
    # naive workspace holds the whole file, where the 10 s markouts of train's last fills
    # read past train's end (on the SOL perpetual at skew 0.25, -1.26100 bps a unit at init
    # against -1.26051 in the replay)
    seen = []
    real = archetypes.maker_facts

    def spy(brief, panel, start, end):
        seen.append(int(panel.grid_ns[-1]))
        return real(brief, panel, start, end)

    monkeypatch.setattr(archetypes, "maker_facts", spy)
    run = vault.init(maker_brief, tmp_path / "naive", mode="naive")
    assert seen == [int(maker_run.panel("train").grid_ns[-1])]
    menus = [json.loads((r.vault / "menu.json").read_text()) for r in (run, maker_run)]
    assert menus[0]["facts"]["maker"] == menus[1]["facts"]["maker"]
    # 60% of 3 is 1 feature session; then strategies; the third names trial 3, lean
    steps = [
        files(("features/lead.py", LEAD), window="train"),
        files(("strategies/quotes.py", QUOTES), window="all"),
        Step(summary={"ideas": [], "next": "", "phase_done": True, "final_trial_id": 3}),
    ]
    harness = FakeHarness(run, steps)
    state = Driver(run, Budget(max_iterations=3), harness=harness).run()
    assert state["status"] == "finished" and state["final"]["trial_id"] == 3
    ledger = run.ledger()
    pick = ledger.trial(3)
    assert (pick["name"], pick["archetype"]) == ("lean", "market_making")
    claimed = json.loads(pick["train_metrics_json"])
    assert claimed["rows"]["brief"]["latency_ms"] == 0 and claimed["fill_model"] in ("RA", "PR")
    tested = json.loads(ledger.test_score(run.run_id)["metrics_json"])
    assert tested["rows"]["brief"]["latency_ms"] == 50.0 and tested["rows"]["RA"]["fills"] > 0
    ledger.close()
    model = gather(run)
    assert "market_making strategy lean on XRPUSDT" in model["verdict"]["line"]
    # the naive final has its fee verdict on the page, and the agent saw none of it
    assert model["recompute"]["final"]["fee_verdict"]["kind"] is not None
    seen = [json.dumps(r, default=str) for r in harness.submitted]
    seen += [p.read_text(errors="ignore") for p in run.workspace.rglob("*")
             if p.is_file() and p.suffix in (".md", ".json", ".txt", ".py", ".yaml")]  # fmt: skip
    assert not [t for t in seen if "fee_verdict" in t or "Fee verdict" in t or "-limited" in t]


@known
def test_local_eval_prints_the_fill_engines_rows(maker_run, tmp_path):
    import blinded as bl

    (tmp_path / "lead.py").write_text(LEAD)
    (tmp_path / "quotes.py").write_text(QUOTES)
    (lead,) = bl.load(tmp_path / "lead.py").features
    join, lean = bl.load(tmp_path / "quotes.py").strategies
    brief = maker_run.workspace / "brief.yaml"
    result = bl.evaluate_strategy(lean, brief, features={"lead": lead})
    assert result.ok and result.rows["row"].to_list()[:5] == ["brief", "RA", "PR", "T", "F"]
    text = repr(result)
    assert "fill engine (bps of one unit; per hour, per fill)" in text and "captured" in text
    assert bl.evaluate_strategy(join, brief).rows["model"][0] in ("RA", "PR")
    # an override of the latency takes a slower cancel down with it: 20 ms against 10
    raw = yaml.safe_load(brief.read_text())
    raw["data"]["path"] = str(maker_run.workspace / "data" / "train.parquet")
    raw["execution"]["cancel_latency_ms"] = 20
    (tmp_path / "brief.yaml").write_text(yaml.safe_dump(raw))
    fast = bl.evaluate_strategy(join, tmp_path / "brief.yaml", latency_ms=10)
    assert fast.ok and fast.row("brief")["latency_ms"] == 10.0


@edge
def test_the_position_charts_leave_a_maker_strategy_to_its_bracket(maker_run, tmp_path):
    # in a families brief a maker strategy can be a family's pick; its quotes are no position,
    # so the position charts return nothing for it
    from blinded.report import _strategy_diagnostics

    (tmp_path / "quotes.py").write_text(QUOTES)
    row = {"code_path": str(tmp_path / "quotes.py"), "name": "join", "legs": "XRPUSDT",
           "trial_id": 1}  # fmt: skip
    assert _strategy_diagnostics(row, {}, "why", None, {}, {}, maker_run.brief()) is None


@known
def test_the_daemon_stores_both_queue_models_and_tells_the_agent_neither(
    maker_run, tmp_path, monkeypatch
):
    # the hand example as validation (tests/test_maker.py): with no maker fee
    # PR's Sharpe on its own blocks (the flatten folded into the second) is the lower and it
    # is scored, at 2 bps and at a 5 bps rebate RA's. PR's holds make its blocks 2 s, of which
    # the 5 s window holds two, fewer than twenty: its score is kept and the Ladder is not
    # asked. RA's blocks are the brief's, so it is.
    from blinded.daemon import Daemon
    from blinded.ladder import Ladder
    from tests.test_maker import judged

    run = RunDir(shutil.copytree(maker_run.root, tmp_path / "maker"))
    daemon = Daemon(run)
    asked = []
    submit = Ladder.submit

    def spy(self, score, trial_id=None):
        assert np.isfinite(score.value)  # a NaN is never a Ladder entry
        asked.append(score.value)
        return submit(self, score, trial_id)

    monkeypatch.setattr(Ladder, "submit", spy)
    common = dict(submitted_by="agent", iteration=1)
    for fee, scored, releasable in ((0.0, "PR", False), (2.0, "RA", True), (-5.0, "RA", True)):
        before = len(asked)
        item = judged(fee)
        public = daemon._record(item, f"# fee {fee}", "x.py", 1.0, common)
        assert sorted(public["validation"]) == ["best", "released", "truncation"]
        assert "RA" not in json.dumps(public["validation"])
        assert "fill_model" not in json.dumps(public["validation"])
        stored = json.loads(daemon.ledger.trial(public["trial_id"])["val_metrics_json"])
        assert stored["fill_model"] == scored and set(stored["models"]) == set(sim_maker.MODELS)
        rule = {m: stored["models"][m]["held"]["sharpe"] for m in ("RA", "PR")}
        assert stored["value"] == min(rule.values()) == rule[scored]
        assert stored["releasable"] is releasable
        assert asked[before:] == ([stored["value"]] if releasable else [])
    # a score that measures nothing: one block of its own holds. The Ladder is not asked, the
    # agent hears what it hears of a loss, and the ledger holds null with the reason
    before = len(asked)
    item = judged(2.0, dict(bid=0, ask=None, size=10, skew=0, max_inventory=10))
    public = daemon._record(item, "# bid only", "x.py", 1.0, common)
    assert len(asked) == before
    assert public["validation"] == {
        "truncation": "passed",
        "released": False,
        "best": daemon.ladders[("strategy", "liquidity")].best,
    }
    row = daemon.ledger.trial(public["trial_id"])
    stored = json.loads(row["val_metrics_json"])
    assert stored["value"] is None and "NaN" not in row["val_metrics_json"]
    assert stored["score_note"].startswith("no Sharpe: its holds give 1 block")
    assert row["released_bool"] == 0
    run.write_ledger_md(daemon.ledger)
    ledger_md = (run.workspace / "LEDGER.md").read_text()
    assert re.search(r"\bnan\b", ledger_md, re.IGNORECASE) is None
    # the maker table shows the train rows and the model the rule took on train, nothing else
    assert "## Market-making strategies on train" in ledger_md
    table = ledger_md.split("## Market-making strategies on train")[1]
    assert "| id | name | scored | brief | RA | PR | T | F | latency 0 |" in table


@edge
def test_a_verdict_never_prints_a_sharpe_that_measures_nothing():
    # the AT file's quote at skew 0 on test: fifteen fills, held for 195,219 of 213,000 steps,
    # one block of its own holds, so the Sharpe the judge's rule scores is null
    from blinded.report import _verdict

    rows = {"brief": {"net_bps_per_hour": -2592.3, "gross_bps_per_hour": -2550.0,
                      "sharpe_per_sqrt_hour": -0.047, "fills": 15, "units": 740.0,
                      "adverse_bps_10s": -900.0, "hours": 5.92}}  # fmt: skip
    final = {"name": "join_touch", "archetype": "market_making", "edge": "liquidity",
             "family": "directional", "legs": ["ATUSDT_MEXC"], "fill_model": "RA",
             "held": {"sharpe": None, "hold_steps": 195219.0, "block_steps": 195219,
                      "blocks": 1, "releasable": False},
             "half_spread_bps": 62.0, "maker_fee_bps": 0.0, "step_s": 0.1,
             "brief_block_steps": 3000}  # fmt: skip
    model = {
        "final_id": 3,
        "test": {"rows": rows},
        "recompute": {"final": final, "curves": {}},
        "windows": {"test": ["2026-10-01T18:05:00+00:00", "2026-10-02T00:00:00+00:00"]},
    }
    line = _verdict(model)["line"]
    assert re.search(r"\bnan\b", line, re.IGNORECASE) is None
    assert "under the risk-averse queue model, the worse of the two" in line
    assert "at a Sharpe its holds leave too few blocks to measure" in line
    # 900 bps over 740 units is 1.22 bps a unit, 0.020 of the 62 bps half spread
    assert line.endswith("not viable as a maker: the fills gave back 0.02 times the half spread "
                         "within 10 s.")  # fmt: skip
    # a position strategy whose window holds fewer than two of the brief's blocks
    rows["brief"]["sharpe_per_sqrt_hour"] = None
    final.pop("half_spread_bps")
    line = _verdict(model)["line"]
    assert "net, with too few blocks to measure a Sharpe;" in line


@known
def test_the_rules_line_says_what_rules_2_would_show_and_nothing_under_rules_2():
    from blinded.report import RULES_V2, _rules_line

    v2 = {"net_bps_per_hour": -822.38, "sharpe_per_sqrt_hour": -4.676, "fills": 9,
          "fill_model": "RA", "judge_rules": 2}  # fmt: skip
    assert (
        _rules_line({"judge_rules": 2, "recompute": {"rules_v2": v2}}) == "Judged under rules v2."
    )
    assert _rules_line({"judge_rules": 1, "recompute": {}}) == "Judged under rules v1."
    head = f"Judged under rules v1. Under rules v2 ({RULES_V2}) this final"
    assert _rules_line({"judge_rules": 1, "recompute": {"rules_v2": v2}}) == (
        f"{head}'s locked-test net is -822.4 bps/hour (Sharpe -4.68, under the risk-averse "
        "queue model).")  # fmt: skip
    taker = {"net_bps_per_hour": 0.17, "sharpe_per_sqrt_hour": None, "fills": 3}
    assert _rules_line({"judge_rules": 1, "recompute": {"rules_v2": taker}}) == (
        f"{head}'s locked-test net is +0.17 bps/hour (too few blocks for a Sharpe)."
    )
    idle = taker | {"fills": 0, "net_bps_per_hour": 0.0}
    assert _rules_line({"judge_rules": 1, "recompute": {"rules_v2": idle}}) == (
        f"{head} does not trade on the locked test."
    )
