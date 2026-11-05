import json
import textwrap

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from blinded import vault
from blinded.cli import main
from blinded.daemon import Daemon, score_test
from blinded.report import _bps_text, _window_text, gather, render, replay, report

known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case

FEATURE = """
import blinded as bl
from blinded import features as F


@bl.feature("perp leads spot; the basis mean-reverts")
def basis(panel):
    b = F.basis("XRPUSDT_PERP", "XRPUSDT")
    return b - F.roll_mean(b, "5m")
"""

LEAK = """
import blinded as bl
import polars as pl


@bl.feature("centred mean of the mid")
def leaky(panel):
    return pl.col("XRPUSDT.mid").rolling_mean(50, center=True)
"""

STRATEGY = """
import blinded as bl


@bl.strategy(requires=["basis"])
def basis_threshold(panel, features):
    return bl.threshold(features["basis"], z_in=2.0, z_out=0.5, every="2m")
"""


def seed(root, long_brief, mode="protected"):
    """Build a run, submit a feature, a leak and a strategy, and score the strategy on test."""
    run = vault.init(long_brief, root, mode=mode)
    daemon = Daemon(run)

    def submit(source, rid, window=None):
        (run.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
        request = {"kind": "eval", "file": "f.py", "name": None}
        if window:
            request["window"] = window
        (run.inbox / f"{rid}.json").write_text(json.dumps(request))
        return daemon.handle(run.inbox / f"{rid}.json")

    window = "all" if mode == "naive" else None
    submit(FEATURE, "f1", window)
    if mode == "protected":
        submit(LEAK, "f2")
    strat = submit(STRATEGY, "s1", window)["trials"][0]
    daemon.ledger.update_run(
        run.run_id, final_trial_id=strat["trial_id"], claude_version="2.1.301"
    )
    score_test(run, strat["trial_id"])
    return run, strat["trial_id"]


@pytest.fixture(scope="module")
def scored(long_brief, tmp_path_factory):
    return seed(tmp_path_factory.mktemp("prot") / "run", long_brief)


@known
def test_every_section_is_present_and_no_rows_leak(scored):
    run, _ = scored
    path = report(run.root)
    doc = path.read_text()
    for heading in (
        "1. Verdict",
        "2. What was tried",
        "3. Deflation",
        "4. The signal",
        "5. Failed ideas and why",
        "7. Reproduce",
        "8. Appendix",
    ):
        assert heading in doc, heading
    # self-contained: inline style, no script, no external fetch (the SVG namespace aside)
    assert "<style>" in doc and "<script" not in doc
    assert "http://" not in doc.replace('xmlns="http://www.w3.org/2000/svg"', "").replace(
        'xmlns:xlink="http://www.w3.org/1999/xlink"', "")  # fmt: skip
    assert "cdn" not in doc.lower()
    # the charts: test P&L, the Ladder, train against validation, the strategies, and the
    # feature's fan, decay and windows, and the strategy's P&L and fill markout
    for title in ("The locked test", "The Ladder", "Every accepted feature",
                  "Strategies on validation", "Markout fan of basis", "Decay of basis",
                  "Rank IC of basis in each", "Where the P&amp;L of basis_threshold goes",
                  "Order markout fan of basis_threshold"):  # fmt: skip
        assert title in doc, title
    assert doc.count("<svg") == 9
    # the verdict names the test window and states viability
    assert "locked test window" in doc and "not viable as a taker" in doc
    # a data row would be an event; the ledger, DATA.md summary and code are not rows
    assert "<td>1.4" not in doc  # no mid/price column of events


@cross
def test_replay_recomputes_every_number_from_the_data(scored):
    run, _ = scored
    report(run.root)  # writes report.json
    result = replay(run.root)
    assert result["ok"], [c for c in result["checks"] if not c["ok"]][:5]
    assert len(result["checks"]) > 30  # test rows, decay, windows and the P&L endpoints
    # the stored numbers equal a fresh recompute to within tolerance
    assert all(c["ok"] for c in result["checks"])


@known
def test_the_verdict_matches_the_stored_test_score(scored):
    run, trial_id = scored
    model = gather(run)
    stored = model["test"]["rows"]["brief"]
    assert model["verdict"]["net"] == stored["net_bps_per_hour"]
    assert _bps_text(stored["net_bps_per_hour"]) in model["verdict"]["line"]
    # the strategy loses money net on the locked test (taker, 50 ms, 7.5 bps)
    assert model["verdict"]["net"] < 0
    assert model["final_id"] == trial_id


@known
def test_rejected_ideas_are_grouped_by_reason(scored):
    run, _ = scored
    model = gather(run)
    statuses = {r["status"] for r in model["rejected"]}
    assert "rejected_lookahead" in statuses  # the planted centred mean
    doc = render(model)
    assert "rejected_lookahead" in doc and "leaky" in doc


@known
def test_a_flat_run_still_renders_every_section(long_brief, tmp_path, capsys):
    # a run where nothing was released: no final trial, no test score
    run = vault.init(long_brief, tmp_path / "flat")
    ledger = run.ledger()
    ledger.add_trial(run.run_id, kind="feature", name="weak", status="accepted",
                     iteration=1, code_hash="h1", released_bool=False)  # fmt: skip
    ledger.start_iteration(run.run_id, 1, "i.jsonl")
    ledger.finish_iteration(run.run_id, 1, cost_usd_est=0.2, turns=3)
    ledger.close()
    doc = report(run.root).read_text()
    assert "No strategy was released" in doc or "no locked-test result" in doc
    assert "4. The signal" in doc and "8. Appendix" in doc
    # replay has nothing data-derived to check, and says so rather than claiming a match
    assert replay(run.root) == {"ok": True, "checks": []}
    assert main(["replay", str(run.root)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("verified: the segment") and "\nnothing to recompute" in out


@known
def test_a_probe_leads_the_report(long_brief, tmp_path):
    run = vault.init(long_brief, tmp_path / "probed")
    ledger = run.ledger()
    ledger.probe(run.run_id, "Bash", "reached the vault")
    ledger.close()
    model = gather(run)
    doc = render(model)
    assert len(model["probes"]) == 1
    assert "halted on 1 sandbox probe" in doc


@cross
def test_the_naive_comparison_appears_beside_a_naive_run(long_brief, tmp_path):
    prot, _ = seed(tmp_path / "run", long_brief)
    seed(tmp_path / "run-naive", long_brief, mode="naive")  # the sibling the report looks for
    model = gather(prot)
    assert "naive" in model
    doc = render(model)
    assert "6. Naive comparison" in doc
    assert "naive loop claimed" in doc and "protected judge, same code, locked test" in doc
    assert f"The naive pick is trial {model['naive']['final_id']}" in doc
    assert "claimed on the all window" in doc


@cross
def test_replay_checks_the_naive_numbers_the_protected_report_shows(long_brief, tmp_path):
    prot, _ = seed(tmp_path / "run", long_brief)
    seed(tmp_path / "run-naive", long_brief, mode="naive")
    report(prot.root)
    naive = [c for c in replay(prot.root)["checks"] if c["name"].startswith("naive.")]
    assert naive and all(c["ok"] for c in naive)
    saved = json.loads((prot.root / "report.json").read_text())
    saved["naive"]["protected_test"]["rows"]["brief"]["net_bps_per_hour"] += 1.0
    (prot.root / "report.json").write_text(json.dumps(saved))
    bad = [c["name"] for c in replay(prot.root)["checks"] if not c["ok"]]
    assert bad == ["naive.test_rows.brief.net_bps_per_hour"]


@known
def test_diagnostics_never_open_test_without_a_scored_final(long_brief, tmp_path):
    run = vault.init(long_brief, tmp_path / "open")
    daemon = Daemon(run)
    for rid, source in (("f1", FEATURE), ("s1", STRATEGY)):
        (run.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
        (run.inbox / f"{rid}.json").write_text(json.dumps({"kind": "eval", "file": "f.py"}))
        daemon.handle(run.inbox / f"{rid}.json")
    diag = gather(run)["recompute"]["diagnostics"]
    assert diag["segments"] == ["train", "val"]
    assert "test" not in json.dumps(diag)
    assert diag["feature"]["name"] == "basis" and diag["strategy"]["name"] == "basis_threshold"


def first_past_the_edges(xs, ts, lo, hi, gap_ns):
    """The event rule written as a plain loop: past an edge, and the gap since the last."""
    picked, last = [], None
    for i, (v, t) in enumerate(zip(xs, ts, strict=True)):
        if (v > hi or v < lo) and (last is None or t >= last + gap_ns):
            picked.append(i)
            last = t
    return np.array(picked, dtype=int)


@cross
def test_the_fan_matches_a_plain_loop_over_events(scored):
    import pandas as pd

    from blinded import contract
    from blinded.evaluate import scoring_window
    from blinded.target import forward_returns

    run, _ = scored
    feat = gather(run)["recompute"]["diagnostics"]["feature"]
    brief, panel = run.brief(), run.panel("test")
    ledger = run.ledger()
    path = next(t["code_path"] for t in ledger.trials("feature") if t["name"] == "basis")
    ledger.close()
    fn = contract.load(path).features[0]
    x = np.asarray(contract.values(fn.fn(panel), panel), dtype=float)
    inst, price, g = brief.target.instrument, brief.target.price, panel.grid_ns

    def rows(segment):
        w = scoring_window(brief, panel, segment)
        return slice(*np.searchsorted(g, [w.start.value, w.end.value]))

    # the edges are the 5th and 95th percentiles on train; the direction is the train IC's sign
    lo, hi = pd.Series(x[rows("train")]).dropna().quantile([0.05, 0.95])
    assert (lo, hi) == pytest.approx((feat["events"]["lo"], feat["events"]["hi"]))
    h = feat["primary"]
    fwd = forward_returns(panel, inst, price, (pd.Timedelta(h),), brief.latency)[h].to_numpy()
    ic = pd.Series(x[rows("train")]).corr(pd.Series(fwd[rows("train")]), method="spearman")
    direction = 1.0 if ic >= 0 else -1.0
    assert feat["events"]["direction"] == direction
    # test events by the plain loop, each one's mid move 10 s on in the traded direction
    xs, ts = x[rows("test")], g[rows("test")]
    picked = first_past_the_edges(xs, ts, lo, hi, pd.Timedelta("70s").value)
    sign = np.where(xs[picked] > hi, 1.0, -1.0) * direction
    t = ts[picked]
    marks = panel.stream_of(inst, price)
    ahead = t + pd.Timedelta("10s").value
    move = (marks.asof(price, ahead) / marks.asof(price, t) - 1) * 1e4
    move = np.where(ahead <= g[-1], move, np.nan) * sign
    ok = np.isfinite(move)
    fan = feat["fan"]["test"]
    k = feat["offsets"].index("10s")
    assert len(picked) >= 5 and fan["n"] == len(picked)
    assert fan["mean_bps"][k] == pytest.approx(move[ok].mean())
    assert fan["se_bps"][k] == pytest.approx(move[ok].std(ddof=1) / np.sqrt(ok.sum()))
    assert fan["long_bps"][k] == pytest.approx(move[ok & (sign > 0)].mean())
    assert fan["short_bps"][k] == pytest.approx(move[ok & (sign < 0)].mean())
    # break-even: the move at the fill and twice the median half spread at the events
    book = panel.stream_of(inst, "bid")
    bid, ask = book.asof("bid", t), book.asof("ask", t)
    half = np.median((ask - bid) / ((ask + bid) / 2) * 1e4 / 2)
    fill = feat["offsets"].index(f"{feat['latency_ms']:g}ms")
    assert fan["at_fill_bps"] == fan["mean_bps"][fill]
    assert fan["breakeven_bps"] == pytest.approx(fan["at_fill_bps"] + 2 * half)


@known
def test_the_per_family_table_counts_what_each_family_did(pairs_run, tmp_path):
    import shutil

    from blinded.vault import RunDir

    run = RunDir(shutil.copytree(pairs_run.root, tmp_path / "pairs"))
    final = tmp_path / "t8.py"
    final.write_text(textwrap.dedent("""
        import blinded as bl
        from blinded import features as F


        @bl.strategy()
        def t8(panel, features):
            return bl.linear(F.basis("XRPUSDT_PERP", "XRPUSDT"), every="2m")
        """))  # fmt: skip
    ledger = run.ledger()
    planted = [  # kind, family, status, released, best after
        ("feature", "directional", "accepted", True, 0.1),
        ("feature", "directional", "accepted", False, 0.1),
        ("feature", "directional", "rejected_lookahead", None, None),
        ("feature", "pairs", "accepted", True, 0.03),
        ("feature", "pairs", "accepted", True, 0.06),
        ("strategy", "pairs", "accepted", False, None),
        ("strategy", "pairs", "error", None, None),
        ("strategy", "directional", "accepted", True, 0.5),
    ]
    for i, (kind, family, status, released, best) in enumerate(planted, 1):
        ledger.add_trial(run.run_id, kind=kind, name=f"t{i}", status=status, family=family,
                         released_bool=released, ladder_best_after=best,
                         code_path=str(final) if i == 8 else None)  # fmt: skip
        if released:
            ledger.add_release(run.run_id, kind, i, None, best, 0.01, family)
    ledger.update_run(run.run_id, final_trial_id=8)
    ledger.close()
    sessions = {"1": ("features", "directional"), "2": ("features", "pairs"),
                "3": ("features", "directional"), "4": ("strategies", "pairs"),
                "5": ("strategies", "directional")}  # fmt: skip
    state = {"phases": {n: p for n, (p, _) in sessions.items()},
             "families": {n: f for n, (_, f) in sessions.items()}}  # fmt: skip
    run.driver.mkdir(exist_ok=True)
    (run.driver / "state.json").write_text(json.dumps(state))
    model = gather(run)
    directional, pairs = model["families"]
    assert directional == {
        "family": "directional", "share": 0.5, "final": True,
        "feature": {"submitted": 3, "accepted": 2, "released": 1, "best": 0.1},
        "strategy": {"submitted": 1, "accepted": 1, "released": 1, "best": 0.5},
        "sessions": {"features": 2, "strategies": 1},
    }  # fmt: skip
    # the released best is the last release of the family's kind
    assert pairs["feature"] == {"submitted": 2, "accepted": 2, "released": 2, "best": 0.06}
    assert pairs["strategy"] == {"submitted": 2, "accepted": 1, "released": 0, "best": None}
    assert (pairs["sessions"], pairs["final"]) == ({"features": 1, "strategies": 1}, False)
    doc = render(model)
    assert ("<td>directional</td><td>0.5</td><td>3 / 2 / 1</td><td>0.1000</td>"
            "<td>1 / 1 / 1</td><td>0.5000</td><td>2, 1</td><td>holds the final strategy</td>"
            ) in doc  # fmt: skip
    assert "<td>pairs</td><td>0.5</td><td>2 / 2 / 2</td><td>0.0600</td><td>2 / 1 / 0</td>" in doc
    assert "equal notional" in doc


PAIRS_STRATEGY = """
import blinded as bl


@bl.strategy(family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
def perp_spot_pairs(panel, features):
    return bl.pairs("XRPUSDT_PERP", "XRPUSDT", z_in=2.0, z_out=0.5, window="2m", every="2m")
"""

LEAD_FEATURE = """
import blinded as bl
from blinded import features as F


@bl.feature("the perp's last second, not yet in spot", family="lead_lag", target="XRPUSDT")
def perp_gap(panel):
    return F.lead_gap("XRPUSDT_PERP", "XRPUSDT", "1s")
"""


@pytest.fixture(scope="module")
def families(long_brief, tmp_path_factory):
    """Three families on the two-hour file, with no fee and one-minute blocks so the pairs
    strategy clears its Ladder; the pairs strategy is the final one, scored on test."""
    import yaml

    root = tmp_path_factory.mktemp("families")
    raw = yaml.safe_load(long_brief.read_text())
    raw["families"] = {"directional": {"instruments": ["XRPUSDT"]},
                       "lead_lag": {"leaders": ["XRPUSDT_PERP"], "followers": ["XRPUSDT"]},
                       "pairs": {"pairs": [["XRPUSDT_PERP", "XRPUSDT"]]}}  # fmt: skip
    raw["execution"]["taker_fee_bps"] = 0.0
    raw["stats"]["block"] = "1min"
    (root / "brief.yaml").write_text(yaml.safe_dump(raw))
    run = vault.init(root / "brief.yaml", root / "run")
    daemon = Daemon(run)
    out = {}
    for rid, source in (("f1", FEATURE), ("f2", LEAD_FEATURE), ("s1", STRATEGY),
                        ("s2", PAIRS_STRATEGY)):  # fmt: skip
        (run.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
        (run.inbox / f"{rid}.json").write_text(json.dumps({"kind": "eval", "file": "f.py"}))
        out[rid] = daemon.handle(run.inbox / f"{rid}.json")["trials"][0]
    final = out["s2"]["trial_id"]
    daemon.ledger.update_run(run.run_id, final_trial_id=final)
    score_test(run, final)
    return run, out


@known
def test_a_run_with_three_families_draws_every_family(families):
    run, trials = families
    assert trials["s2"]["validation"]["released"] and trials["s1"]["validation"]["released"]
    doc = report(run.root).read_text()
    saved = json.loads((run.root / "report.json").read_text())
    diag = saved["recompute"]["diagnostics"]
    # the directional pick, the lead-lag curve and pick, and the final strategy's pair
    assert diag["feature"]["name"] == "basis" and diag["strategy"]["name"] == "basis_threshold"
    (curve,) = diag["lead_lag"]["curves"]
    assert (curve["leader"], curve["follower"]) == ("XRPUSDT_PERP", "XRPUSDT")
    assert set(curve["segments"]) == {"train", "val", "test"}
    assert diag["lead_lag"]["feature"]["name"] == "perp_gap"
    (pair,) = diag["pairs"]
    assert pair["pair"] == ["XRPUSDT_PERP", "XRPUSDT"] and pair["name"] == "perp_spot_pairs"
    # the strategy's own window and thresholds were read from its bl.pairs call
    assert pair["params"] == {"window": "2m", "every": "2m", "z_in": 2.0, "z_out": 0.5}
    assert pair["params_from"] == "the strategy's own bl.pairs call"
    for title in ("Lead-lag: XRPUSDT_PERP then XRPUSDT", "The spread of XRPUSDT_PERP over",
                  "Hedge ratio of XRPUSDT_PERP on XRPUSDT", "Half-life of the XRPUSDT_PERP",
                  "Markout fan of the XRPUSDT_PERP over XRPUSDT spread",
                  "Markout fan of perp_gap"):  # fmt: skip
        assert title in doc, title
    # nine charts as before, a lead-lag chart, the lead-lag pick's three feature charts, and
    # four for the pair; no strategy of the lead-lag family was submitted
    assert doc.count("<svg") == 9 + 1 + 3 + 4
    checked = replay(run.root)
    assert checked["ok"], [c for c in checked["checks"] if not c["ok"]][:5]
    assert any(c["name"].startswith("diagnostics.pairs[0].fan.test") for c in checked["checks"])
    assert any(c["name"].startswith("diagnostics.lead_lag.curves[0]") for c in checked["checks"])


@cross
def test_the_spread_fan_matches_a_plain_loop_over_events(families):
    import pandas as pd

    from blinded.evaluate import scoring_window
    from blinded.features import cross as fx

    run, _ = families
    pair = gather(run)["recompute"]["diagnostics"]["pairs"][0]
    brief, panel = run.brief(), run.panel("test")
    a, b = "XRPUSDT_PERP", "XRPUSDT"
    frame = panel.frame.select(z=fx.spread_z(a, b, "2m", "2m"),
                               beta=fx.hedge_ratio(a, b, "2m", "2m")).to_pandas()  # fmt: skip
    g = panel.grid_ns

    def rows(segment):
        w = scoring_window(brief, panel, segment)
        return slice(*np.searchsorted(g, [w.start.value, w.end.value]))

    def log_mid(sym, at):
        return np.log(panel.stream_of(sym, "mid").asof("mid", at))

    # past the strategy's entry band of z 2, short the spread when z is high, long when low
    z_in = pair["params"]["z_in"]
    zs, ts = frame["z"].to_numpy()[rows("test")], g[rows("test")]
    picked = first_past_the_edges(zs, ts, -z_in, z_in, pd.Timedelta("70s").value)
    sign = np.where(zs[picked] > z_in, -1.0, 1.0)
    t, beta = ts[picked], frame["beta"].to_numpy()[rows("test")][picked]
    ahead = t + pd.Timedelta("10s").value
    move = (log_mid(a, ahead) - log_mid(a, t)) - beta * (log_mid(b, ahead) - log_mid(b, t))
    move = np.where(ahead <= g[-1], move * 1e4, np.nan) * sign
    ok = np.isfinite(move)
    fan = pair["fan"]["test"]
    k = pair["offsets"].index("10s")
    assert len(picked) >= 5 and fan["n"] == len(picked)
    assert fan["mean_bps"][k] == pytest.approx(move[ok].mean())
    assert fan["se_bps"][k] == pytest.approx(move[ok].std(ddof=1) / np.sqrt(ok.sum()))
    # a spread that reverts: the trade back towards the mean makes money over the next 10 s
    assert fan["mean_bps"][k] > 0


@known
def test_the_naive_comparison_counts_the_calls_of_each_family(families, tmp_path):
    import shutil

    from blinded.vault import RunDir

    run, _ = families
    prot = RunDir(shutil.copytree(run.root, tmp_path / "run"))
    naive = vault.init(run.root.parent / "brief.yaml", tmp_path / "run-naive", mode="naive")
    daemon = Daemon(naive)
    ids = []
    for rid, source in (("f1", FEATURE), ("f2", LEAD_FEATURE), ("f3", LEAD_FEATURE),
                        ("s1", PAIRS_STRATEGY)):  # fmt: skip
        (naive.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
        request = {"kind": "eval", "file": "f.py", "window": "all"}
        (naive.inbox / f"{rid}.json").write_text(json.dumps(request))
        ids.append(daemon.handle(naive.inbox / f"{rid}.json")["trials"][0]["trial_id"])
    daemon.ledger.update_run(naive.run_id, final_trial_id=ids[-1])
    score_test(naive, ids[-1])
    model = gather(prot)
    # the naive run judged one directional feature, the lead-lag feature twice and one pair;
    # the protected run a directional feature and strategy, one lead-lag feature, one pair
    assert model["naive"]["calls_by_family"] == {"directional": 1, "lead_lag": 2, "pairs": 1}
    doc = render(model)
    for family, theirs, ours in (("directional", 1, 2), ("lead_lag", 2, 1), ("pairs", 1, 1)):
        assert f"<tr><td>judge calls, {family}</td><td>{theirs}</td><td>{ours}</td></tr>" in doc
    assert "of the pairs family" in doc


@known
def test_a_pairs_strategy_with_its_own_feature_is_charted_on_its_own_positions(families):
    from blinded.report import _pair_diagnostics

    run, trials = families
    source = run.root / "own.py"
    source.write_text(textwrap.dedent("""
        import blinded as bl
        from blinded import features as F


        @bl.feature("the spread's z", family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
        def own_z(panel):
            return F.spread_z("XRPUSDT_PERP", "XRPUSDT", "2m", "2m")


        @bl.strategy(requires=["own_z"], family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
        def own_pairs(panel, features):
            u = -bl.threshold(features["own_z"], z_in=2.0, z_out=0.5, every="2m")
            return {"XRPUSDT_PERP": u, "XRPUSDT": -u}
        """))  # fmt: skip
    row = {"code_path": str(source), "name": "own_pairs", "trial_id": 99}
    brief, panel = run.brief(), run.panel("val")
    from blinded.evaluate import scoring_window

    wins = {s: scoring_window(brief, panel, s) for s in ("train", "val")}
    idx = {s: tuple(int(i) for i in np.searchsorted(panel.grid_ns, [w.start.value, w.end.value]))
           for s, w in wins.items()}  # fmt: skip
    pair = ("XRPUSDT_PERP", "XRPUSDT")
    out = _pair_diagnostics(pair, row, "test", {}, panel, wins, idx, brief,
                            brief.family("pairs"))  # fmt: skip
    # its code makes no bl.pairs call, so the bands are the brief's defaults
    assert out["params_from"].startswith("the brief's pairs defaults")
    # its feature is in its own file, not the ledger, and the shading is still its own
    # position, refitted every 2 minutes, not bl.pairs at the defaults' 30 minutes
    fallback = _pair_diagnostics(pair, None, "test", {}, panel, wins, idx, brief,
                                 brief.family("pairs"))  # fmt: skip
    held = np.asarray(out["spread"]["val"]["position"], dtype=float)
    other = np.asarray(fallback["spread"]["val"]["position"], dtype=float)
    assert np.nanmax(np.abs(held)) > 0 and not np.allclose(held, other, equal_nan=True)


@known
def test_markouts_end_the_strip_at_the_fill_and_keep_to_a_thinned_files_stamps():
    from blinded import brief as briefs
    from blinded.durations import to_ns
    from blinded.report import _fan_offsets, _send_offsets, _signed

    raw = demo_binance_brief()
    thin = briefs.parse(raw)
    ms = 1_000_000
    # the fill is the 50 ms latency plus the 50 ms feed delay after the decision. An offset
    # tau reads the mark available at g + tau, stamp g + tau - 50 ms: -100 ms reads g - 150
    # and 250 ms reads g + 200, which a file keeping g - 50, g and g + 50 does not have
    asked = [to_ns(o) for o in ("-500ms", "-100ms", "0s", "250ms", "500ms", "1s")]
    got = [_signed(o) for o in _send_offsets(asked, 100 * ms, 50 * ms, thin)]
    assert got == ["-500ms", "0ms", "100ms", "500ms", "1s"]
    # the fan's own offsets on that file: every 500 ms step from -10 s to 60 s, and the fill
    fan = _fan_offsets(500 * ms, 100 * ms, 50 * ms, thin)
    assert fan == sorted({*range(-10_000 * ms, 60_000 * ms + 1, 500 * ms), 100 * ms})
    # the same brief unthinned keeps every offset, and with no delay the fill is the latency
    del raw["data"]["thinned"]
    raw["data"]["availability"]["default_delay_ms"] = 0
    full = briefs.parse(raw)
    assert _send_offsets(asked, 50 * ms, 0, full) == sorted({*asked, 50 * ms})


def demo_binance_brief():
    from datetime import date

    from blinded import demo

    return demo.binance_brief("x.parquet", ["X", "Y"], date(2024, 3, 27), 1, "500ms", 50, 50)


def ar1(rng, n, half_life_steps):
    """An AR(1) path whose coefficient halves a shock in ``half_life_steps`` steps."""
    phi = 0.5 ** (1 / half_life_steps)
    out = np.zeros(n)
    noise = rng.normal(size=n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


@known
def test_an_archetypes_run_draws_the_pair_with_the_shortest_half_life():
    from blinded import brief as briefs
    from blinded.brief import Window
    from blinded.panel import Panel
    from blinded.report import _archetype_pair

    # A is a random walk; C is A plus noise that halves in 10 s, B plus noise that halves in
    # 120 s, both 2 bps a step. The spread of A and C reverts at 10 s, A and B at 120 s, and
    # B and C, the difference of the two noises, mostly at the slow one: A over C is drawn
    rng = np.random.default_rng(11)
    n = 6 * 3600
    grid = pd.date_range("2024-01-01", periods=n, freq="1s", tz="UTC")
    a = 100 * np.exp(np.cumsum(rng.normal(scale=1e-4, size=n)))
    legs = {"A": a, "B": a * np.exp(2e-4 * ar1(rng, n, 120)),
            "C": a * np.exp(2e-4 * ar1(rng, n, 10))}  # fmt: skip
    book = {
        k: pl.DataFrame({"ts": grid, "bid": v - 0.01, "ask": v + 0.01}) for k, v in legs.items()
    }
    panel = Panel.from_frames(grid, book=book)
    raw = {
        "data": {"path": "x.parquet", "instrument_column": "s", "streams": {}},
        "target": {"instrument": "A", "horizons": ["1s"]},
        "grid": {"step": "1s", "warmup": "0s"},
        "splits": {"embargo": "1min"},
        "archetypes": {"instruments": ["A", "B", "C"],
                       "params": {"pairs_trading": {"window": "30min", "every": "30min"}}},
    }  # fmt: skip
    brief = briefs.parse(raw)
    assert brief.family("pairs").pairs == (("A", "B"), ("A", "C"), ("B", "C"))
    train = Window("train", grid[0], grid[-1])
    assert _archetype_pair("pairs_trading", brief, panel, train) == ("A", "C")


SILENT = """
import blinded as bl


@bl.strategy(requires=["basis"])
def basis_never(panel, features):
    return bl.threshold(features["basis"], z_in=1e9, z_out=0.5, every="2m")
"""


@edge
def test_a_final_strategy_that_never_trades_on_test_still_reports(long_brief, tmp_path):
    # a carry rule waits for a rich basis; if test never gives one there are no fills,
    # the Sharpe is null, and the verdict must say so rather than format None
    run = vault.init(long_brief, tmp_path / "silent")
    daemon = Daemon(run)
    for rid, source in (("f1", FEATURE), ("s1", SILENT)):
        (run.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
        (run.inbox / f"{rid}.json").write_text(json.dumps({"kind": "eval", "file": "f.py"}))
        out = daemon.handle(run.inbox / f"{rid}.json")
    trial_id = out["trials"][0]["trial_id"]
    daemon.ledger.update_run(run.run_id, final_trial_id=trial_id, claude_version="2.1.301")
    score_test(run, trial_id)
    model = gather(run)
    assert model["test"]["rows"]["brief"]["fills"] == 0
    assert "never traded" in model["verdict"]["line"]
    assert model["verdict"]["sharpe"] is None
    report(run.root)
    assert replay(run.root)["ok"]


@cross
def test_the_deflated_sharpe_uses_the_briefs_block(long_brief, tmp_path):
    # with 1-minute blocks there are 60 an hour, not the 12 of the 5-minute default;
    # the deflation's Sharpe per sqrt-hour must then equal the judge's own validation one
    raw = yaml.safe_load(long_brief.read_text())
    raw["stats"]["block"] = "1min"
    brief = tmp_path / "brief.yaml"
    brief.write_text(yaml.safe_dump(raw))
    run, trial_id = seed(tmp_path / "run", brief)
    model = gather(run)
    d = model["deflation"]
    assert "error" not in d, d
    val = json.loads(run.ledger().trial(trial_id)["val_metrics_json"])
    judged = val["metrics"]["rows"]["brief"]["sharpe_per_sqrt_hour"]
    assert d["sharpe_per_sqrt_hour"] == pytest.approx(judged, rel=1e-9)


@edge
def test_the_verdict_keeps_small_numbers_and_long_windows_readable():
    # a carry trade earns hundredths of a bp an hour over months of test
    assert _bps_text(0.014347) == "+0.014"
    assert _bps_text(-0.00042616) == "-0.00043"
    assert _bps_text(1.0936) == "+1.1"
    assert _bps_text(0.0) == "+0.0"
    assert (
        _window_text(["2025-03-06T11:45:00+00:00", "2025-06-25T00:00:00+00:00"])
        == "2025-03-06 to 2025-06-25"
    )
    assert (
        _window_text(["2024-01-01T03:35:00+00:00", "2024-01-01T04:00:00+00:00"])
        == "03:35:00-04:00:00"
    )


@edge
def test_a_positive_net_whose_interval_reaches_zero_is_not_called_viable(scored, monkeypatch):
    import blinded.report as rep

    run, _ = scored
    monkeypatch.setattr(rep, "_ci", lambda model: ((-0.6, 2.9), False))
    model = gather(run)
    model["test"]["rows"]["brief"]["net_bps_per_hour"] = 1.09
    line = rep._verdict(model)["line"]
    assert "cannot tell it from no edge" in line and "viable" not in line
    monkeypatch.setattr(rep, "_ci", lambda model: ((0.2, 2.9), False))
    assert rep._verdict(model)["line"].endswith("viable before costs are reconsidered.")
    # too few blocks for an interval: no interval, so never viable either
    monkeypatch.setattr(rep, "_ci", lambda model: (None, True))
    line = rep._verdict(model)["line"]
    assert "(too few blocks to say for an interval)" in line and "viable" not in line
    assert line.endswith("without an interval the test cannot tell it from no edge.")


HOLD = """
import blinded as bl
import polars as pl


@bl.strategy(requires=["basis"])
def always_long(panel, features):
    return pl.lit(1.0) + 0.0 * features["basis"].fill_null(0.0)
"""


@known
def test_a_strategy_held_back_for_its_holds_is_named_in_the_report(long_brief, tmp_path):
    run = vault.init(long_brief, tmp_path / "held")
    daemon = Daemon(run)
    for rid, source in (("f1", FEATURE), ("s1", HOLD)):
        (run.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
        (run.inbox / f"{rid}.json").write_text(json.dumps({"kind": "eval", "file": "f.py"}))
        daemon.handle(run.inbox / f"{rid}.json")
    model = gather(run)
    (held,) = model["held_back"]
    assert held["name"] == "always_long" and held["own_blocks"] == 1
    # 03:05 to 03:30 held throughout: 25 minutes
    assert "always_long (trial 2), holds of 25 min, 1 in validation" in render(model)
