import json
import shutil

import pytest
import yaml

from blinded.driver import Budget, Driver
from blinded.ledger import now
from blinded.vault import RunDir
from tests.fake_harness import (
    BASIS,
    PAIRS,
    SPREAD_Z,
    STRATEGY,
    WEAK,
    CrashInjected,
    FakeHarness,
    Step,
    ideas,
)

known = pytest.mark.known_answer
edge = pytest.mark.edge_case


@pytest.fixture
def run(run_dir, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run_dir.root, root)
    return RunDir(root)


@pytest.fixture
def naive_run(long_brief, tmp_path):
    from blinded import vault

    return vault.init(long_brief, tmp_path / "naive", mode="naive")


# -- the counting rules, by hand on a built ledger --------------------------


def build(
    run,
    phases: dict[int, str],
    trials: list[dict],
    summaries: dict[int, dict] = None,
    families: dict[int, str] = None,
    budget: Budget = None,
):
    """Populate a fresh ledger with iteration rows and trials, and set the driver state."""
    summaries = summaries or {}
    ledger = run.ledger()
    for table in ("trials", "iterations", "test_scores"):
        ledger.db.execute(f"DELETE FROM {table}")
    for n, phase in phases.items():
        ledger.start_iteration(run.run_id, n, f"iter_{n}.jsonl")
        summary = {"phase": phase, **summaries.get(n, {})}
        ledger.finish_iteration(run.run_id, n, cost_usd_est=1.0, ended_at=now(),
                                summary_json=summary)  # fmt: skip
    for t in trials:
        ledger.add_trial(
            run.run_id,
            kind=t["kind"],
            name=t.get("name", "f"),
            iteration=t["iteration"],
            status=t.get("status", "accepted"),
            released_bool=t.get("released"),
            code_hash=t.get("name") or "h",
            family=t.get("family", "directional"),
            edge=t.get("edge"),
            archetype=t.get("archetype"),
            ladder_best_after=t.get("best"),
        )
    ledger.close()
    d = Driver(run, budget or Budget(max_iterations=20), harness=FakeHarness(run, []))
    d.state = {"phases": {str(n): p for n, p in phases.items()}}
    if families:
        d.state["families"] = {str(n): f for n, f in families.items()}
    return d


def feature(n, released=False, status="accepted", name="f", family="directional"):
    return {
        "kind": "feature",
        "iteration": n,
        "released": released,
        "status": status,
        "name": name,
        "family": family,
    }


@known
def test_three_sessions_without_a_feature_release_end_the_features_phase(run):
    # iterations 1..3 each submit a feature that is not released (max_iterations 20, 60% =
    # 12, so the 60% rule would only bite at iteration 12). After the third session without
    # a release the status names the strategies phase for iteration 4.
    phases = {n: "features" for n in range(1, 4)}
    trials = [feature(n, name=f"f{n}") for n in range(1, 4)]
    two = build(run, {1: "features", 2: "features"}, trials[:2])
    assert (two.status().phase, two.status().non_releases) == ("features", 2)
    three = build(run, phases, trials)
    assert three.status().phase == "strategies"
    assert three.status().phase_end == "3 sessions in a row without a feature release"


@known
def test_seven_rejected_submissions_in_one_session_do_not_close_the_phase(run):
    # the rule counts sessions, not submissions: seven non-releases in one session are one
    trials = [feature(1, status="rejected_lookahead", name=f"f{k}") for k in range(7)]
    d = build(run, {1: "features"}, trials)
    assert (d.status().phase, d.status().non_releases) == ("features", 1)


@known
def test_an_empty_session_counts_as_a_session_without_a_release(run):
    # iteration 2 submits nothing: still a session without a release (and an empty one)
    d = build(run, {1: "features", 2: "features", 3: "features"}, [feature(1), feature(3)])
    status = d.status()
    assert status.phase == "strategies" and status.empty == 0


@known
def test_a_release_resets_the_streak(run):
    # a release in session 3, none in 4 and 5: two sessions since it, so the phase goes on;
    # a sixth without one is the third, and ends it
    phases = {n: "features" for n in range(1, 6)}
    trials = [feature(1), feature(2), feature(3, released=True, name="basis")]
    trials += [feature(n, name=f"f{n}") for n in (4, 5)]
    d = build(run, phases, trials)
    assert (d.status().phase, d.status().non_releases) == ("features", 2)
    six = build(run, {**phases, 6: "features"}, [*trials, feature(6, name="f6")])
    assert six.status().phase == "strategies"


@known
def test_sixty_percent_of_iterations_ends_the_features_phase(run):
    # no streak (releases keep resetting) but iteration 12 reaches 60% of 20
    phases = {n: "features" for n in range(1, 13)}
    trials = [feature(n, released=(n % 2 == 1), name=f"f{n}") for n in range(1, 13)]
    d = build(run, phases, trials)
    assert d.status().phase == "strategies"
    assert "60%" in d.status().phase_end


@known
def test_phase_done_twice_ends_the_features_phase(run):
    phases = {1: "features", 2: "features", 3: "features"}
    trials = [feature(1, released=True, name="basis"), feature(2), feature(3)]
    once = build(run, phases, trials, summaries={3: {"summary": {"phase_done": True}}})
    assert once.status().phase == "features"
    twice = build(run, phases, trials,
                  summaries={2: {"summary": {"phase_done": True}},
                             3: {"summary": {"phase_done": True}}})  # fmt: skip
    assert twice.status().phase == "strategies"
    assert "phase_done" in twice.status().phase_end


@known
def test_three_sessions_without_a_strategy_release_end_the_strategies_phase(run):
    # features end on phase_done twice after iteration 2; strategies run 3, 4, 5 without a
    # release, so the status names finalise for iteration 6
    phases = {1: "features", 2: "features"} | {n: "strategies" for n in (3, 4, 5)}
    trials = [feature(1, released=True, name="basis")]
    trials += [{"kind": "strategy", "iteration": n, "released": False, "name": f"s{n}"}
               for n in (3, 4, 5)]  # fmt: skip
    done = {n: {"summary": {"phase_done": True}} for n in (1, 2)}
    two = build(run, {n: p for n, p in phases.items() if n < 5}, trials[:3], done)
    assert (two.status().phase, two.status().non_releases) == ("strategies", 2)
    d = build(run, phases, trials, done)
    assert d.status().phase == "finalise"
    assert d.status().phase_end == "3 sessions in a row without a strategy release"


@known
def test_the_streak_lengths_come_from_the_budget(run):
    phases = {n: "features" for n in range(1, 4)}
    trials = [feature(n, name=f"f{n}") for n in range(1, 4)]
    budget = Budget(max_iterations=20, feature_streak_sessions=4)
    assert build(run, phases, trials, budget=budget).status().phase == "features"
    with pytest.raises(ValueError, match="feature_streak_sessions"):
        Budget(feature_streak_sessions=0)


# -- families: sessions by share, a streak per family ------------------------


def three_families(run, shares=(0.5, 0.25, 0.25)):
    """The run with directional, lead-lag and pairs families in its vault brief."""
    path = run.vault / "brief.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["families"] = {
        "directional": {"share": shares[0]},
        "lead_lag": {"share": shares[1], "leaders": ["XRPUSDT_PERP"], "followers": ["XRPUSDT"]},
        "pairs": {"share": shares[2], "pairs": [["XRPUSDT_PERP", "XRPUSDT"]]},
    }
    path.write_text(yaml.safe_dump(raw))
    return run


def assigned(run, n, released=lambda family, n: True, budget=None):
    """The families of n feature sessions in a row, each released or not as ``released`` says."""
    order, trials = [], []
    for i in range(1, n + 1):
        phases = {k: "features" for k in range(1, i)}
        fams = dict(enumerate(order, start=1))
        family = build(run, phases, trials, families=fams, budget=budget).status().family
        order.append(family)
        trials.append(feature(i, released=released(family, i), name=f"f{i}", family=family))
    return order


@known
def test_sessions_are_shared_out_by_share(run):
    # directional 0.5, lead_lag 0.25, pairs 0.25: each session goes to the open family with
    # the smallest sessions / share, ties to the first in the brief:
    #  1: 0, 0, 0 -> D; 2: 2, 0, 0 -> L; 3: 2, 4, 0 -> P; 4: 2, 4, 4 -> D; 5: 4, 4, 4 -> D;
    #  6: 6, 4, 4 -> L; 7: 6, 8, 4 -> P; 8: 6, 8, 8 -> D; 9: 8, 8, 8 -> D; 10: 10, 8, 8 -> L;
    # 11: 10, 12, 8 -> P; 12: 10, 12, 12 -> D. Every session releases, so none closes.
    order = assigned(three_families(run), 12)
    d, ll, pp = "directional", "lead_lag", "pairs"
    assert order == [d, ll, pp, d, d, ll, pp, d, d, ll, pp, d]
    assert [order.count(f) for f in (d, ll, pp)] == [6, 3, 3]


@known
def test_a_closed_family_hands_its_sessions_to_the_others(run):
    # as above, but pairs never releases: its third session, the 11th, closes it. Then, with
    # directional on 5 sessions and lead_lag on 3: 12: 5/0.5 = 10, 3/0.25 = 12 -> D;
    # 13: 12, 12 -> D; 14: 14, 12 -> L; 15: 14, 16 -> D; 16: 16, 16 -> D
    order = assigned(three_families(run), 16, lambda family, n: family != "pairs",
                     Budget(max_iterations=40))  # fmt: skip
    d, ll, pp = "directional", "lead_lag", "pairs"
    assert order[:11] == [d, ll, pp, d, d, ll, pp, d, d, ll, pp]
    assert order[11:] == [d, d, ll, d, d]


@known
def test_a_family_closes_on_three_of_its_own_sessions_without_a_release(pairs_run, tmp_path):
    run = RunDir(shutil.copytree(pairs_run.root, tmp_path / "pairs"))
    # sessions alternate pairs, directional, pairs, directional, pairs. No pairs feature is
    # released in sessions 1, 3 or 5; session 3 releases a directional feature, which counts
    # for nothing in either streak, and session 2 releases one in its own family.
    fams = {1: "pairs", 2: "directional", 3: "pairs", 4: "directional", 5: "pairs"}
    phases = dict.fromkeys(fams, "features")
    trials = [feature(1, family="pairs"), feature(2, released=True, name="d2"),
              feature(3, released=True, name="d3"), feature(4, name="d4"),
              feature(5, family="pairs", name="p5")]  # fmt: skip
    status = build(run, phases, trials, families=fams).status()
    by = {st.family: st for st in status.standings}
    assert (by["pairs"].streak, by["pairs"].closed) == (3, "3 sessions in a row without a "
                                                            "feature release")  # fmt: skip
    assert (by["directional"].streak, by["directional"].closed) == (1, None)
    assert (status.phase, status.family) == ("features", "directional")
    d = build(run, phases, trials, families=fams)
    d._write_budget(6, d.status())
    budget = (run.workspace / "BUDGET.md").read_text()
    assert "This session is for the directional family." in budget
    assert "| directional | 0.5 | 2 | 1 of 3 | none yet | open |" in budget
    assert "| pairs | 0.5 | 3 | 3 of 3 | none yet | closed: 3 sessions" in budget
    # a release of a pairs feature in session 5 resets its streak to zero
    trials[-1] = feature(5, released=True, family="pairs", name="p5")
    status = build(run, phases, trials, families=fams).status()
    assert {st.family: st.streak for st in status.standings} == {"pairs": 0, "directional": 1}
    # with both closed the phase ends, and says why for each
    more = {6: "directional", 7: "directional"}
    trials += [feature(6, name="d6"), feature(7, name="d7")]
    trials[-3] = feature(5, family="pairs", name="p5")
    end = build(run, phases | dict.fromkeys(more, "features"), trials, families=fams | more)
    status = end.status()
    assert status.phase == "strategies"
    assert status.phase_end == (
        "every family is closed: directional, 3 sessions in a row without a feature release; "
        "pairs, 3 sessions in a row without a feature release"
    )


# -- archetypes: sessions, streaks and BUDGET.md by edge ----------------------


def edge_feature(n, edge, released=False):
    archetype = "lead_lag" if edge == "speed" else "pairs_trading"
    family = "lead_lag" if edge == "speed" else "pairs"
    return {"kind": "feature", "iteration": n, "released": released, "name": f"f{n}",
            "family": family, "edge": edge, "archetype": archetype,
            "best": 0.1 if released else None}  # fmt: skip


@known
def test_sessions_go_by_edge_and_an_edge_closes_on_its_own_streak(archetypes_run, tmp_path):
    run = RunDir(shutil.copytree(archetypes_run.root, tmp_path / "archetypes"))
    sp, rv = "speed", "relative_value"
    # speed 0.5 and relative value 0.5, every session releasing: the smallest sessions / share
    # goes next, ties to speed (first of the five edges): 0/0 S, 2/0 R, 2/2 S, 4/2 R, ...
    order, trials = [], []
    for i in range(1, 9):
        phases = {k: "features" for k in range(1, i)}
        d = build(run, phases, trials, families=dict(enumerate(order, start=1)))
        order.append(d.status().family)
        trials.append(edge_feature(i, order[-1], released=True))
    assert order == [sp, rv, sp, rv, sp, rv, sp, rv]
    # now relative value never releases: its third session, the 6th, closes it, and with
    # speed alone open the 7th and 8th go to speed
    order, trials = [], []
    for i in range(1, 9):
        phases = {k: "features" for k in range(1, i)}
        d = build(run, phases, trials, families=dict(enumerate(order, start=1)))
        order.append(d.status().family)
        trials.append(edge_feature(i, order[-1], released=order[-1] == sp))
    assert order == [sp, rv, sp, rv, sp, rv, sp, sp]
    by = {st.family: st for st in d.status().standings}
    assert (by[rv].streak, by[rv].closed) == (3, "3 sessions in a row without a feature release")
    # BUDGET.md before the 8th session: one row per edge, with the edge's members
    d._write_budget(8, d.status())
    budget = (run.workspace / "BUDGET.md").read_text()
    assert "This session is for the speed & information edge." in budget
    assert ("| edge | members | share | sessions this phase | without a release in a row | "
            "released best | open |") in budget  # fmt: skip
    assert ("| speed & information | lead_lag, order_book_imbalance, trade_flow_persistence | "
            "0.5 | 4 | 0 of 3 | 0.1 | open |") in budget  # fmt: skip
    assert ("| relative value | pairs_trading | 0.5 | 3 | 3 of 3 | none yet | closed: 3 "
            "sessions in a row without a feature release |") in budget  # fmt: skip
    assert "or every edge is closed; an edge closes after `phase_done` in two" in budget


# -- hard stops -------------------------------------------------------------


@known
def test_each_hard_stop_fires_on_its_own_count(run):
    d = build(run, {1: "features"}, [feature(1)])
    base = d.status()
    assert d._hard_stop(base) is None
    import dataclasses

    assert d._hard_stop(dataclasses.replace(base, probes=1)) == "probe"
    assert d._hard_stop(dataclasses.replace(base, spent_usd=60.0)) == "max_usd"
    assert d._hard_stop(dataclasses.replace(base, wall_seconds=8 * 3600)) == "wall_hours"
    assert d._hard_stop(dataclasses.replace(base, iterations=20)) == "max_iterations"
    assert d._hard_stop(dataclasses.replace(base, empty=3)) == "stuck"
    # a probe outranks everything
    assert d._hard_stop(dataclasses.replace(base, probes=1, spent_usd=60.0)) == "probe"


@known
def test_three_empty_iterations_in_a_row_are_stuck(run):
    phases = {1: "features", 2: "features", 3: "features"}
    trials = [feature(1, released=True, name="basis")]  # iterations 2 and 3 submit nothing
    two = build(run, {1: "features", 2: "features"}, trials)
    assert two.status().empty == 1
    d = build(run, phases, trials)
    assert d.status().empty == 2  # iterations 2, 3 empty; iteration 1 had a trial
    assert d._hard_stop(d.status()) is None
    four = build(run, {**phases, 4: "features"}, trials)
    assert four.status().empty == 3
    assert four._hard_stop(four.status()) == "stuck"


# -- the final choice -------------------------------------------------------


def with_strategies(run, rows, final_summary=None):
    ledger = run.ledger()
    ledger.start_iteration(run.run_id, 1, "i.jsonl")
    ledger.finish_iteration(
        run.run_id, 1, summary_json={"phase": "finalise", **(final_summary or {})}
    )
    for r in rows:
        ledger.add_trial(run.run_id, kind="strategy", name=r["name"], iteration=2,
                         status=r.get("status", "accepted"), released_bool=r.get("released"),
                         code_hash=r["name"], family=r.get("family", "directional"),
                         ladder_best_after=r.get("best"))  # fmt: skip
    ledger.close()
    d = Driver(run, Budget(), harness=FakeHarness(run, []))
    d.state = {"phases": {"1": "finalise"}}
    return d


@known
def test_a_released_pick_the_agent_names_stands(run):
    d = with_strategies(run, [{"name": "s1", "released": True}],
                        final_summary={"summary": {"final_trial_id": 1}})  # fmt: skip
    choice = d._final_choice("finalised")
    assert choice == {"agent_pick": 1, "finalised_by": "agent", "trial_id": 1, "override": None}


@known
def test_an_unreleased_pick_falls_back_to_the_last_released(run):
    # trial 1 released, trial 2 accepted but never released; the agent names 2
    d = with_strategies(run, [{"name": "s1", "released": True}, {"name": "s2", "released": False}],
                        final_summary={"summary": {"final_trial_id": 2}})  # fmt: skip
    choice = d._final_choice("finalised")
    assert choice["trial_id"] == 1
    assert "never released" in choice["override"]
    assert choice["agent_pick"] == 2


@known
def test_the_fallback_is_the_best_released_score_across_families(run):
    # released bests 0.10 (directional), 0.20 (pairs), 0.15 (directional); the agent names
    # trial 4, never released: the pairs strategy, trial 2, has the highest released best
    rows = [{"name": "d1", "released": True, "best": 0.10},
            {"name": "p2", "released": True, "best": 0.20, "family": "pairs"},
            {"name": "d3", "released": True, "best": 0.15},
            {"name": "d4", "released": False}]  # fmt: skip
    d = with_strategies(run, rows, final_summary={"summary": {"final_trial_id": 4}})
    choice = d._final_choice("finalised")
    assert choice["trial_id"] == 2 and "trial 2, is final" in choice["override"]


@known
def test_a_tie_in_released_score_goes_to_the_later_trial(run):
    tied = [{"name": "p1", "released": True, "best": 0.2, "family": "pairs"},
            {"name": "d2", "released": True, "best": 0.2}]  # fmt: skip
    assert with_strategies(run, tied)._final_choice("max_usd")["trial_id"] == 2


@known
def test_with_nothing_released_the_final_strategy_is_flat(run):
    d = with_strategies(run, [{"name": "s1", "released": False}],
                        final_summary={"summary": {"final_trial_id": 1}})  # fmt: skip
    choice = d._final_choice("finalised")
    assert choice["trial_id"] is None
    assert "flat" in choice["override"]


# -- end to end, with the real daemon on synthetic data ---------------------


@known
def test_a_protected_run_finalises_flat_when_nothing_releases(run):
    # iter1 submits basis (releases a feature) and returns phase_done; iter2 phase_done again
    # ends the features phase; iter3 (strategies) submits the basis threshold, which on this
    # synthetic signal is accepted but does not release; iter4 finalises. With no released
    # strategy the driver's final pick is flat and the test segment is not scored.
    steps = [
        Step(
            files=[("features/basis.py", BASIS)],
            submit=[("features/basis.py", None)],
            summary={**ideas("basis"), "phase_done": True},
        ),  # fmt: skip
        Step(summary={**ideas(), "phase_done": True}),
        Step(
            files=[("strategies/s.py", STRATEGY)],
            submit=[("strategies/s.py", None)],
            summary=ideas("basis_threshold"),
        ),  # fmt: skip
        Step(summary={"ideas": [], "next": "", "phase_done": False, "final_trial_id": None}),
    ]
    harness = FakeHarness(run, steps)
    state = Driver(run, Budget(max_iterations=4, max_usd=1.0), harness=harness).run()
    assert state["status"] == "finished" and state["stop"] == "finalised"
    assert harness.calls == 4
    ledger = run.ledger()
    # the feature released, the strategy did not
    assert any(t["released_bool"] for t in ledger.trials("feature"))
    assert not any(t["released_bool"] for t in ledger.trials("strategy"))
    # nothing released, so the final strategy is flat and test is not scored
    assert state["final"]["trial_id"] is None
    assert "flat" in state["final"]["override"]
    assert ledger.test_score(run.run_id) is None
    phases = state["phases"]
    assert phases["1"] == "features" and phases["3"] == "strategies" and phases["4"] == "finalise"
    assert state["phase_ends"][0]["to"] == "strategies"
    ledger.close()


@edge
def test_a_probe_halts_the_run_and_nothing_is_scored(run):
    steps = [
        Step(
            files=[("features/basis.py", BASIS)],
            submit=[("features/basis.py", None)],
            summary=ideas("basis"),
        ),  # fmt: skip
        Step(probe=True, summary=ideas()),
        Step(summary=ideas()),  # never reached
    ]
    harness = FakeHarness(run, steps)
    state = Driver(run, Budget(max_iterations=10), harness=harness).run()
    assert state["status"] == "halted" and state["stop"] == "probe"
    assert harness.calls == 2  # stopped before a third session
    ledger = run.ledger()
    assert [t["kind"] for t in ledger.trials("probe")] == ["probe"]
    assert ledger.test_score(run.run_id) is None
    ledger.close()


@known
def test_a_bash_read_of_the_sessions_own_saved_result_does_not_halt_the_run(run):
    # Claude Code refused it before the hook, so only the driver's scan sees it,
    # and that scan needs the session to tell the agent's own results from another's
    saved = "~/.claude/projects/-ws/{session}/tool-results/b1.txt"
    other = "~/.claude/projects/-ws/fake-1/tool-results/b1.txt"

    def read(path):
        return {
            "tool_name": "Bash",
            "tool_use_id": "tu-read",
            "tool_input": {"command": f"sed -n 1,400p {path}"},
        }

    steps = [
        Step(denials=[read(saved)], summary=ideas()),
        Step(denials=[{**read(other), "tool_use_id": "tu-other"}], summary=ideas()),
        Step(summary=ideas()),  # never reached
    ]
    harness = FakeHarness(run, steps)
    state = Driver(run, Budget(max_iterations=10), harness=harness).run()
    assert harness.calls == 2
    assert state["stop"] == "probe"
    rows = run.ledger().trials("probe")
    assert ["tu-other" in r["reason_short"] for r in rows] == [True]
    events = [json.loads(line) for line in (run.driver / "events.jsonl").open()]
    assert [(e["iteration"], e["tool_use_id"]) for e in events if e["source"] == "denial"] == [
        (1, "tu-read")
    ]


@edge
def test_resume_after_a_crash_mid_iteration(run):
    # iter1 submits basis and says phase_done; iter2 submits weak, says phase_done, then the
    # driver dies before settling the session (the session's transcript is on disk)
    first = [
        Step(
            files=[("features/basis.py", BASIS)],
            submit=[("features/basis.py", None)],
            summary={**ideas("basis"), "phase_done": True},
        ),  # fmt: skip
        Step(
            files=[("features/weak.py", WEAK)],
            submit=[("features/weak.py", None)],
            summary={**ideas("weak"), "phase_done": True},
            crash=True,
        ),  # fmt: skip
    ]
    harness = FakeHarness(run, first)
    with pytest.raises(CrashInjected):
        Driver(run, Budget(max_iterations=4, max_usd=1.0), harness=harness).run()
    ledger = run.ledger()
    rows = ledger.iterations(run.run_id)
    assert len(rows) == 2 and rows[1]["ended_at"] is None  # the crash left iteration 2 open
    assert run.vault.joinpath("iteration").read_text().strip() == "2"
    ledger.close()

    # resume: iter2 is settled from its transcript, the two phase_done end features, iter3 is
    # strategies, iter4 finalises
    rest = [
        Step(
            files=[("strategies/s.py", STRATEGY)],
            submit=[("strategies/s.py", None)],
            summary=ideas("basis_threshold"),
        ),  # fmt: skip
        Step(summary={"ideas": [], "next": "", "phase_done": False, "final_trial_id": None}),
    ]
    resumed = FakeHarness(run, rest)
    state = Driver(run, harness=resumed).run(resume=True)
    assert state["status"] == "finished" and resumed.calls == 2
    ledger = run.ledger()
    closed = ledger.iterations(run.run_id)
    assert len(closed) == 4 and all(r["ended_at"] is not None for r in closed)
    # the crashed session's submission (weak) is in the ledger, settled from its transcript
    assert any(t["name"] == "weak" for t in ledger.trials("feature"))
    assert state["phases"]["3"] == "strategies"
    ledger.close()


@known
def test_naive_final_pick_is_rescored_on_the_locked_test(naive_run):
    run = naive_run
    steps = [
        Step(
            files=[("features/basis.py", BASIS)],
            submit=[("features/basis.py", None)],
            window="all",
            summary={**ideas("basis"), "phase_done": True},
        ),  # fmt: skip
        Step(summary={**ideas(), "phase_done": True}),
        Step(
            files=[("strategies/s.py", STRATEGY)],
            submit=[("strategies/s.py", None)],
            window="all",
            summary=ideas("basis_threshold"),
        ),  # fmt: skip
        Step(summary={"ideas": [], "next": "", "phase_done": False, "final_trial_id": None}),
    ]
    state = Driver(run, Budget(max_iterations=4), harness=FakeHarness(run, steps)).run()
    assert state["status"] == "finished"
    ledger = run.ledger()
    scored = ledger.test_score(run.run_id)
    assert scored is not None
    metrics = json.loads(scored["metrics_json"])
    # rescoring runs the protected judge: a score, or a rejection if it does not pass
    assert "score" in metrics or "status" in metrics
    # the ledger keeps the window each claim was made on, and LEDGER.md shows it
    basis = next(t for t in ledger.trials("feature") if t["name"] == "basis")
    assert json.loads(basis["train_metrics_json"])["window"] == "all"
    text = (run.workspace / "LEDGER.md").read_text()
    assert "| window |" in text and "| all |" in text
    assert "Ladder" not in text and "released" not in text
    ledger.close()


@pytest.fixture
def families_run(long_brief, tmp_path):
    """Directional and pairs families on the two-hour file, with no fee and 1 minute blocks,
    so that a strategy of each family can clear the Ladder on 25 minutes of validation."""
    from blinded import vault

    raw = yaml.safe_load(long_brief.read_text())
    raw["families"] = {"directional": {}, "pairs": {"pairs": [["XRPUSDT_PERP", "XRPUSDT"]]}}
    raw["execution"]["taker_fee_bps"] = 0.0
    raw["stats"]["block"] = "1min"
    (tmp_path / "brief.yaml").write_text(yaml.safe_dump(raw))
    return vault.init(tmp_path / "brief.yaml", tmp_path / "run")


def submits(path, source, *names, done=False):
    return Step(files=[(path, source)], submit=[(path, None)],
                summary={**ideas(*names), "phase_done": done})  # fmt: skip


@known
def test_a_protected_run_with_families_finalises_and_replays(families_run):
    from blinded.report import replay, report

    run = families_run
    # sessions alternate directional and pairs, from equal shares. Each family returns
    # phase_done in both of its feature sessions, which closes it; 4 is also 60% of 8. In the
    # strategies phase each family's strategy clears its own Ladder, and the finalise
    # session names the pairs strategy, trial 4.
    steps = [
        submits("features/basis.py", BASIS, "basis", done=True),
        submits("features/spread.py", SPREAD_Z, "spread_reverts", done=True),
        Step(summary={**ideas(), "phase_done": True}),
        Step(summary={**ideas(), "phase_done": True}),
        submits("strategies/s.py", STRATEGY, "basis_threshold"),
        submits("strategies/pairs.py", PAIRS, "perp_spot_pairs"),
        Step(summary=ideas()),
        Step(summary={"ideas": [], "next": "", "phase_done": True, "final_trial_id": 4}),
    ]
    state = Driver(
        run, Budget(max_iterations=8, max_usd=1.0), harness=FakeHarness(run, steps)
    ).run()
    assert (state["status"], state["stop"]) == ("finished", "finalised")
    d, p = "directional", "pairs"
    assert state["families"] == {str(n): f for n, f in enumerate([d, p, d, p, d, p, d], 1)}
    assert state["final"] == {"agent_pick": 4, "finalised_by": "agent", "trial_id": 4,
                              "override": None}  # fmt: skip
    ledger = run.ledger()
    rows = {t["name"]: t for t in ledger.trials()}
    assert {n: (t["family"], t["legs"]) for n, t in rows.items()} == {
        "basis": (d, "XRPUSDT"),
        "spread_reverts": (p, "XRPUSDT_PERP/XRPUSDT"),
        "basis_threshold": (d, "XRPUSDT"),
        "perp_spot_pairs": (p, "XRPUSDT_PERP/XRPUSDT"),
    }
    assert rows["basis_threshold"]["released_bool"] and rows["perp_spot_pairs"]["released_bool"]
    assert [r["family"] for r in ledger.releases("strategy")] == [d, p]
    (scored,) = ledger.db.execute("SELECT * FROM test_scores").fetchall()
    tested = json.loads(scored["metrics_json"])["rows"]["brief"]
    assert scored["trial_id"] == 4 and tested["gross_exposure"] > 0
    assert tested["fills"] == tested["fills_XRPUSDT_PERP"] + tested["fills_XRPUSDT"]
    ledger.close()
    assert "## Families" in (run.workspace / "RULES.md").read_text()

    report(run.root)
    saved = json.loads((run.root / "report.json").read_text())
    by = {f["family"]: f for f in saved["families"]}
    for fam in (d, p):
        assert by[fam]["feature"]["submitted"] == by[fam]["strategy"]["submitted"] == 1
        assert by[fam]["strategy"]["released"] == 1
    assert by[d]["sessions"] == {"features": 2, "strategies": 2}
    assert by[p]["sessions"] == {"features": 2, "strategies": 1}
    assert (by[d]["final"], by[p]["final"]) == (False, True)
    assert "pairs strategy perp_spot_pairs on XRPUSDT_PERP/XRPUSDT" in saved["verdict"]["line"]
    assert "gross exposure" in saved["verdict"]["line"]
    checked = replay(run.root)
    assert checked["ok"] and len(checked["checks"]) > 30


@known
def test_a_naive_pairs_pick_is_rescored_with_two_legs(long_brief, tmp_path):
    from blinded import vault
    from blinded.report import gather

    raw = yaml.safe_load(long_brief.read_text())
    raw["families"] = {"directional": {}, "pairs": {"pairs": [["XRPUSDT_PERP", "XRPUSDT"]]}}
    (tmp_path / "brief.yaml").write_text(yaml.safe_dump(raw))
    run = vault.init(tmp_path / "brief.yaml", tmp_path / "naive", mode="naive")
    # 60% of 4 iterations is 2: sessions 1 (directional) and 2 (pairs) are features, 3 is
    # strategies and 4 finalises, naming the pairs strategy, trial 2
    steps = [
        Step(files=[("features/basis.py", BASIS)], submit=[("features/basis.py", None)],
             window="all", summary=ideas("basis")),
        Step(summary=ideas()),
        Step(files=[("strategies/pairs.py", PAIRS)], submit=[("strategies/pairs.py", None)],
             window="all", summary=ideas("perp_spot_pairs")),
        Step(summary={"ideas": [], "next": "", "phase_done": True, "final_trial_id": 2}),
    ]  # fmt: skip
    state = Driver(run, Budget(max_iterations=4), harness=FakeHarness(run, steps)).run()
    assert state["status"] == "finished" and state["final"]["trial_id"] == 2
    ledger = run.ledger()
    pick = ledger.trial(2)
    assert (pick["name"], pick["family"]) == ("perp_spot_pairs", "pairs")
    # the claim was made at zero latency on the whole file; the rescoring is the protected
    # judge's, two legs at the brief's latency and fees on the locked test
    claimed = json.loads(pick["train_metrics_json"])["rows"]["brief"]
    assert claimed["latency_ms"] == 0 and claimed["gross_exposure"] > 0
    tested = json.loads(ledger.test_score(run.run_id)["metrics_json"])["rows"]["brief"]
    assert tested["latency_ms"] == 50 and tested["fees_bps"] > 0
    assert tested["fills"] == tested["fills_XRPUSDT_PERP"] + tested["fills_XRPUSDT"]
    ledger.close()
    line = gather(run)["verdict"]["line"]
    assert "pairs strategy perp_spot_pairs on XRPUSDT_PERP/XRPUSDT" in line
