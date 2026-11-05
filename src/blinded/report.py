"""The run report: one self-contained HTML file, and ``blinded replay`` to check it.

``blinded report DIR`` writes ``DIR/report.html`` and, beside it,
``DIR/report.json``, the numbers the page shows. The page is self-contained:
inline CSS, inline SVG charts drawn by :mod:`blinded.plots`, no JavaScript and
no network, so it opens offline and survives email. It never embeds a data
row, only aggregates of them; the data stays in the vault and in the user's
own copy.

The sections, in order: the verdict on the locked test, what
was tried, the deflation, the signal, the failed ideas, the naive comparison when
there is a naive run beside it, how to reproduce, and an appendix.

Everything the page derives from the data, the test verdict, the P&L paths and
the diagnostics behind the charts, is computed once by :func:`recompute` from
the vault. ``blinded replay DIR`` runs :func:`recompute` again and checks every
number against ``report.json``; a test asserts they agree. In a run with
archetypes it also recomputes the menu from the vault's brief and training
rows and checks it against the one the agent was shown.
"""

from __future__ import annotations

import dataclasses
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from blinded import __version__, archetypes, contract, deflate, fees, plots, sim, sim_maker
from blinded import markouts as mk
from blinded.brief import Family
from blinded.durations import label, to_ns, words
from blinded.evaluate import evaluate_strategy, positions, scoring_window
from blinded.ladder import RULES
from blinded.target import forward_returns
from blinded.vault import RunDir

TOL = 1e-6
DIAGNOSTIC = ("latency 0", "fees 0", "latency 0, fees 0")


# -- the numbers ------------------------------------------------------------


def recompute(run: RunDir, trial_id: int | None, diagnostics: bool = True) -> dict:
    """Recompute the data-derived numbers of the report from the vault.

    Deterministic: the final strategy and its features are read from their
    vault copies and scored on the test segment with the brief's latency and
    fees, giving the test sim rows and the cumulative net P&L path of the
    brief variant and the two unrealistic ones. With no final strategy
    (nothing released) there is no test part. ``diagnostics`` adds the
    numbers behind the charts, from :func:`_diagnostics`.
    """
    out: dict = {"final": None}
    if diagnostics:
        out["diagnostics"] = _diagnostics(run, trial_id)
    if trial_id is None:
        return out
    brief = run.brief()
    ledger = run.ledger()
    row = ledger.trial(trial_id)
    accepted = {
        r["name"]: r["code_path"] for r in ledger.trials("feature") if r["status"] == "accepted"
    }
    ledger.close()
    strat = next(s for s in contract.load(row["code_path"]).strategies if s.name == row["name"])
    features = {}
    for need in strat.requires:
        path = accepted.get(need)
        if path:
            features[need] = next(f for f in contract.load(path).features if f.name == need)
    panel = run.panel("test")
    result = evaluate_strategy(strat, brief, "test", panel=panel, features=features)
    win = scoring_window(brief, panel, "test")
    final = {"trial_id": trial_id, "name": row["name"], "requires": strat.requires}
    if brief.has_families or brief.has_archetypes:
        final |= {"family": result.family, "legs": list(result.legs)}
    if brief.has_archetypes:
        final |= {"archetype": result.archetype, "edge": result.edge}
    if isinstance(result.position, contract.Quotes) and result.archetype == "cross_venue_making":
        curves, extra = _arb_test(result.position, panel, win, brief, result.legs)
        final |= extra
    elif isinstance(result.position, contract.Quotes):
        curves, extra = _maker_test(result.position, panel, win, brief, result.legs)
        final |= extra
    else:
        kept: dict = {}
        curves = _curves(result.position, panel, win, brief, result.legs, kept)
        if "brief" in kept:
            # the fee verdict, on the brief row's blocks under the hold rule: report-only
            final["fee_verdict"] = fees.judge(
                kept["brief"], brief, panel.step_ns, _traded(result.position, result.legs, brief)
            )
    out |= {"final": final, "test_rows": _rows(result.rows), "curves": curves}
    # what the final shows under each later set of rules: rules_v2 for a run judged under
    # rules 1, as published, and rules_v<RULES> for any run judged under older ones
    arb = result.archetype == "cross_venue_making"
    for rules in range(brief.judge_rules + 1, RULES + 1):
        if rules == 2 or rules == RULES:
            out[f"rules_v{rules}"] = _under_rules(result.position, panel, win, brief,
                                                  result.legs, rules, arb)  # fmt: skip
    return out


def _under_rules(position, panel, win, brief, legs, rules: int, arb: bool = False) -> dict:
    """What the judge would show for the final on test under other judging ``rules``
    (:data:`blinded.ladder.RULES`): the net and the Sharpe its verdict line prints, a market
    maker's under the worse of RA and PR by the hold rule (:func:`blinded.sim_maker.held`), an
    arbitrage's likewise over both legs, a position strategy's brief row. Report-only: the run
    stays judged under its own rules. Where those rules refuse to score the final (an
    arbitrage whose brief sets no USDT withdrawal fee, under rules 3) ``refused`` says why."""
    other = dataclasses.replace(brief, judge_rules=rules)
    out: dict = {"judge_rules": rules}
    if arb:
        (quote,) = legs or (brief.target.instrument,)
        _, hedge = archetypes.arb_pair(brief)
        why = sim_maker.unset_fees(brief.stock, sim_maker.coin_of(quote), rules)
        if why is not None:
            return out | {"refused": why}
        sims = sim_maker.arb_models(position, panel, win, other, quote, hedge,
                                    which=sim_maker.QUEUE_MODELS)  # fmt: skip
        scored = sim_maker.worse(sims["RA"], sims["PR"])
        sharpe = sim_maker.held(scored)["sharpe"]
        out["fill_model"] = scored.model
    elif isinstance(position, contract.Quotes):
        ex = brief.execution
        (inst,) = legs or (brief.target.instrument,)
        touch = sim_maker.touch_of(panel, inst, ex.tick_size, win)
        sims = sim_maker.models(position, panel, win, other, inst, touch, sim_maker.QUEUE_MODELS)
        scored = sim_maker.worse(sims["RA"], sims["PR"])
        sharpe = sim_maker.held(scored)["sharpe"]
        out["fill_model"] = scored.model
    else:
        (inst, *_) = legs or (brief.target.instrument,)
        scored = _run(position, panel, win, other, inst)(None, None)
        sharpe = scored.metrics()["sharpe_per_sqrt_hour"]
    metrics = scored.metrics()
    out["fills"] = int(metrics["fills"])
    out["net_bps_per_hour"] = float(metrics["net_bps_per_hour"])
    out["sharpe_per_sqrt_hour"] = None if sharpe is None or not np.isfinite(sharpe) else float(
        sharpe)  # fmt: skip
    return out


def _traded(position, legs, brief) -> tuple[str, ...]:
    """The instruments a position strategy trades: its legs', else ``legs``, else the
    brief's target."""
    if isinstance(position, contract.Legs):
        return tuple(position)
    return tuple(i for i in legs if i) or (brief.target.instrument,)


def _maker_test(quotes, panel, win, brief, legs=()) -> tuple[dict, dict]:
    """A market-making final on test: the cumulative net of the scored model at the brief's
    latencies, at every latency 0 and that without fees; and for the verdict the scored model,
    its hold rule's Sharpe and block (:func:`blinded.sim_maker.held`) and the median half
    spread of the window in bps of the mid."""
    ex = brief.execution
    (inst,) = legs or (brief.target.instrument,)
    touch = sim_maker.touch_of(panel, inst, ex.tick_size, win)
    sims = sim_maker.models(quotes, panel, win, brief, inst, touch, sim_maker.QUEUE_MODELS)
    scored = sim_maker.worse(sims["RA"], sims["PR"])
    (fast,) = sim_maker.models(quotes, panel, win, brief, inst, touch, (scored.model,),
                               zero=True).values()  # fmt: skip
    variants = {"brief": scored, "latency 0": fast, "latency 0, fees 0": fast.refee(0, 0)}
    curves = {name: np.cumsum(s.net).tolist() for name, s in variants.items()}
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [win.start.value, win.end.value])
    j = touch.row(g[i0:i1])
    j = j[j >= 0]
    half = mk.half_spread_bps(touch.bid[j] * touch.tick, touch.ask[j] * touch.tick)
    return curves, {"fill_model": scored.model, "held": _held_numbers(sim_maker.held(scored)),
                    "half_spread_bps": half, "maker_fee_bps": ex.maker_fee_bps,
                    "step_s": panel.step_ns / 1e9,
                    "brief_block_steps": scored.per_block,
                    "fee_verdict": fees.judge(scored, brief, panel.step_ns, (inst,))}  # fmt: skip


def _arb_test(quotes, panel, win, brief, legs=()) -> tuple[dict, dict]:
    """An arbitrage final on test: the cumulative net of the scored model at the brief's
    latencies, at every latency 0 and that without fees; and for the verdict the scored model,
    its hold rule's Sharpe, the probe's terms of the pass (:func:`blinded.sim_maker.arb_table`),
    the quote venue's prints a day, and the fee verdict on the hedge fee
    (:func:`blinded.fees.judge`)."""
    from blinded import dies

    (quote,) = legs or (brief.target.instrument,)
    _, hedge = archetypes.arb_pair(brief)
    sims = sim_maker.arb_models(quotes, panel, win, brief, quote, hedge,
                                which=sim_maker.QUEUE_MODELS)  # fmt: skip
    scored = sim_maker.worse(sims["RA"], sims["PR"])
    (fast,) = sim_maker.arb_models(quotes, panel, win, brief, quote, hedge, which=(scored.model,),
                                   zero=True).values()  # fmt: skip
    variants = {"brief": scored, "latency 0": fast, "latency 0, fees 0": fast.refee(0, 0)}
    curves = {name: np.cumsum(s.net).tolist() for name, s in variants.items()}
    flow = dies.counterparty(scored, panel, win.start.value, win.end.value)
    return curves, {
        "fill_model": scored.model,
        "held": _held_numbers(sim_maker.held(scored)),
        "step_s": panel.step_ns / 1e9,
        "brief_block_steps": scored.per_block,
        "quote": quote,
        "hedge": hedge,
        "ref": scored.ref,
        "hedge_fee_bps": scored.hedge_fee_bps,
        "arb": sim_maker.arb_table(scored),
        "transfers": sim_maker.transfer_terms(scored),
        "prints_a_day": flow.get("prints_a_day"),
        "fee_verdict": fees.judge_arb(scored, brief, panel.step_ns, hedge),
    }


def _diagnostics(run: RunDir, final_id: int | None) -> dict:
    """The numbers behind the signal charts: one feature and one strategy, by segment.

    The feature is the final strategy's, else the last one the Ladder
    released, else the accepted one with the largest train rank IC. The
    strategy is the final one, else the accepted one with the best validation
    Sharpe. Both are shown on train and validation, and on test only when the
    final strategy was scored there: the test window is opened once, for it.
    The top level holds the directional family's pick, each on its own
    instrument. A run with a lead-lag family adds ``lead_lag``: the
    cross-correlation of every leader and follower, and the family's own pick
    by the same rule; one with a pairs family adds ``pairs``, the four pairs
    charts (:func:`_pair_diagnostics`).
    """
    ledger = run.ledger()
    trials = ledger.trials()
    tested = final_id is not None and ledger.test_score(run.run_id) is not None
    ledger.close()
    brief = run.brief()
    if brief.has_archetypes:
        return _archetype_diagnostics(run, trials, final_id, tested, brief)

    def usable(kind, family):
        return [t for t in trials if t["kind"] == kind and t["status"] == "accepted"
                and t["family"] == family
                and t["code_path"] and Path(t["code_path"]).is_file()]  # fmt: skip

    grouped = brief.has_families or brief.has_archetypes
    lead_lag = brief.family("lead_lag") if grouped else None
    pairs = brief.family("pairs") if grouped else None
    picks = {"directional": _pick(usable, "directional", final_id, run.mode, brief)}
    if lead_lag is not None:
        picks["lead_lag"] = _pick(usable, "lead_lag", final_id, run.mode, brief)
    if not any(p[0] or p[2] for p in picks.values()) and lead_lag is None and pairs is None:
        return {}

    segs = ["train", "val", "test"] if tested else ["train", "val"]
    panel = run.panel(segs[-1])
    g = panel.grid_ns
    wins = {seg: scoring_window(brief, panel, seg) for seg in segs}
    idx = {seg: tuple(int(i) for i in np.searchsorted(g, [w.start.value, w.end.value]))
           for seg, w in wins.items()}  # fmt: skip
    every_feature = {t["name"]: t for t in trials if t["kind"] == "feature"
                     and t["status"] == "accepted" and t["code_path"]
                     and Path(t["code_path"]).is_file()}  # fmt: skip
    out: dict = {"segments": segs}
    out |= _single_leg(picks["directional"], every_feature, panel, wins, idx, brief)
    if lead_lag is not None:
        curves = _lead_lag_diagnostics(lead_lag, panel, wins, brief)
        out["lead_lag"] = {"curves": curves,
                           **_single_leg(picks["lead_lag"], every_feature, panel, wins, idx,
                                         brief)}  # fmt: skip
    if pairs is not None:
        out["pairs"] = [
            _pair_diagnostics(pair, row, why, every_feature, panel, wins, idx, brief, pairs)
            for pair, row, why in _pair_picks(usable("strategy", "pairs"), final_id, pairs)
        ]
    return out


def _archetype_diagnostics(run: RunDir, trials, final_id, tested: bool, brief) -> dict:
    """The signal charts and the death checks of every runnable archetype, by edge.

    Each archetype gets the pick of :func:`_pick` among its own accepted
    trials, the charts of its mechanism (one leg's feature and strategy, the
    lead-lag curve of its leader and follower, or the pairs charts) and
    ``dies``, its death checks on each segment (:func:`_dies`). A pairs
    archetype with no accepted strategy is drawn at its template's
    parameters, on its declared pair or, for pairs trading, the implied pair
    with the shortest median half-life on train.
    """
    menu = run.menu()
    segs = ["train", "val", "test"] if tested else ["train", "val"]
    panel = run.panel(segs[-1])
    g = panel.grid_ns
    wins = {seg: scoring_window(brief, panel, seg) for seg in segs}
    idx = {seg: tuple(int(i) for i in np.searchsorted(g, [w.start.value, w.end.value]))
           for seg, w in wins.items()}  # fmt: skip
    every_feature = {t["name"]: t for t in trials if t["kind"] == "feature"
                     and t["status"] == "accepted" and t["code_path"]
                     and Path(t["code_path"]).is_file()}  # fmt: skip
    out: dict = {"segments": segs, "archetypes": {}}

    def segment_panel(seg: str):
        # the panel the judge scored a segment on: train up to the end of that segment
        return panel if seg == segs[-1] else run.panel(seg)

    for edge in menu.edges:
        out["archetypes"][edge] = {
            name: _archetype_entry(name, trials, final_id, run.mode, every_feature, panel, wins,
                                   idx, brief, segment_panel)
            for name in menu.members(edge)
        }  # fmt: skip
    return out


def _archetype_entry(name, trials, final_id, mode, every_feature, panel, wins, idx, brief,
                     segment_panel=None):  # fmt: skip
    arch = archetypes.REGISTRY[name]

    def usable(kind, _archetype=None):
        return [t for t in trials if t["kind"] == kind and t["status"] == "accepted"
                and t["archetype"] == name and t["code_path"]
                and Path(t["code_path"]).is_file()]  # fmt: skip

    entry: dict = {"title": arch.title, "mechanism": arch.mechanism, "template": arch.template,
                   "checks": list(arch.checks)}  # fmt: skip
    stash: dict = {}
    pick = _pick(usable, name, final_id, mode, brief)
    if arch.template == "arb":
        feat, why, strat, swhy = pick
        if feat is not None:
            entry["feature"] = _feature_diagnostics(feat, why, panel, wins, idx, brief, stash)
        arb, dies = None, {}
        if strat is not None:
            arb, dies = _arb_panel(strat, every_feature, swhy, segment_panel or
                                   (lambda seg: panel), wins, brief)  # fmt: skip
        if arb:
            entry["arb"] = arb
        else:
            dies = {c: {"note": "no accepted strategy of this archetype"} for c in arch.checks}
        entry["dies"] = dies
        return entry
    if arch.template == "maker":
        # the feature's charts as for any directional feature; the strategy's bracket of the
        # four fill models, its charts and its death checks from the same passes
        feat, why, strat, swhy = pick
        if feat is not None:
            entry["feature"] = _feature_diagnostics(feat, why, panel, wins, idx, brief, stash)
        maker, dies = None, {}
        if strat is not None:
            maker, dies = _maker_panel(strat, every_feature, swhy, segment_panel or
                                       (lambda seg: panel), wins, brief)  # fmt: skip
        if maker:
            entry["maker"] = maker
        else:
            dies = {c: {"note": "no accepted strategy of this archetype"} for c in arch.checks}
        entry["dies"] = dies
        return entry
    if arch.mechanism == "basket":
        entry |= _basket(name, pick, every_feature, panel, wins, idx, brief, stash)
    elif arch.mechanism in ("directional", "lead_lag"):
        entry |= _single_leg(pick, every_feature, panel, wins, idx, brief, stash)
        if arch.mechanism == "lead_lag":
            entry["curves"] = _lead_lag_diagnostics(_leader_follower(pick, brief), panel, wins,
                                                    brief)  # fmt: skip
    else:
        params = brief.archetypes.params_for(name)
        fallback, carry = None, arch.template == "carry"
        if carry:
            # the spread is drawn at the pairs fit; the position, with no strategy, is bl.carry
            fit = brief.archetypes.params_for("pairs_trading")
            params = {k: fit[k] for k in ("window", "every", "z_in", "z_out")} | params
        unknown = Family("pairs", 1.0, pairs=((),), defaults=params)
        ((legs, row, why),) = _pair_picks(usable("strategy"), final_id, unknown)
        if row is None:
            # only now the implied pairs are searched: 190 of them on twenty names
            legs = _archetype_pair(name, brief, panel, wins["train"])
            why = f"no {name} strategy was accepted; " + (
                "the implied pair with the shortest median half-life on train"
                if name == "pairs_trading" else "the declared pair")  # fmt: skip
        pair = tuple(legs)
        if carry:
            rates = {k: params[k] for k in ("rate_in", "rate_out")}
            fallback = sim.carry(*pair, **rates)
            if row is None:
                why += (f", held by bl.carry(rate_in={rates['rate_in']:g}, "
                        f"rate_out={rates['rate_out']:g}), the brief's parameters")  # fmt: skip
        family = Family("pairs", 1.0, pairs=(pair,), defaults=params)
        entry["pairs"] = [_pair_diagnostics(legs, row, why, every_feature, panel, wins, idx,
                                            brief, family, stash, fallback)]  # fmt: skip
        if name == "cross_venue_arbitrage":
            a, b = pair
            venues = Family("lead_lag", 1.0, leaders=(a, b), followers=(a, b))
            entry["curves"] = _lead_lag_diagnostics(venues, panel, wins, brief)
        if pick[0] is not None:
            entry["feature"] = {"trial_id": pick[0]["trial_id"], "name": pick[0]["name"],
                                "why": pick[1]}  # fmt: skip
            stash |= _feature_values(pick[0], panel)
    entry["dies"] = _dies(arch, entry, stash, panel, wins, idx, brief)
    return entry


# the bracket's columns: what each fill model did on a segment
BRACKET = (("fills", "fills"), ("units", "units"), ("captured_bps", "captured, bps"),
           ("adverse_bps_10s", "markout at 10 s, bps"), ("inventory_bps", "inventory, bps"),
           ("fees_bps", "fees, bps"), ("net_bps_per_hour", "net bps/h"),
           ("sharpe_per_sqrt_hour", "Sharpe/sqrt h"))  # fmt: skip


def _maker_panel(strat_row, by_name, why, panels, wins, brief) -> tuple[dict | None, dict]:
    """A market-making strategy on each segment, and its death checks, from one set of passes.

    Each segment is run on ``panels(segment)``, the panel the judge scored it
    on (train up to the end of that segment), so that every number here is the
    judge's: the train rows of the ledger, validation's models and score, and
    the test rows.

    For each segment: the four fill models' fills, units, captured half
    spread, markout at 10 s, inventory beyond it, fees, net and Sharpe on the
    brief's blocks, with ``held``, the hold rule's Sharpe and block of each
    (:func:`blinded.sim_maker.held`); ``scored``, the model the judge's rule
    takes, the worse of RA and PR by that Sharpe; the fee rows on its fills;
    each model's fill fan (:func:`blinded.sim_maker.fill_fan`) and
    decomposition (:func:`blinded.sim_maker.decompose`); and the scored
    model's three P&L lines (:func:`blinded.sim_maker.pnl_lines`). Then the
    excerpt of the last segment (:func:`blinded.sim_maker.excerpt`). The death
    checks are the archetype's, each by segment, on the scored model. None
    and no checks when one of its features was never accepted.
    """
    from blinded import dies

    obj = next(s for s in contract.load(strat_row["code_path"]).strategies
               if s.name == strat_row["name"])  # fmt: skip
    if any(n not in by_name for n in obj.requires):
        return None, {}
    required = {n: next(f for f in contract.load(by_name[n]["code_path"]).features
                        if f.name == n) for n in obj.requires}  # fmt: skip
    inst = strat_row["legs"] or brief.target.instrument
    composed = positions(obj, required, (inst,), "maker")
    ex = brief.execution
    out = {"trial_id": strat_row["trial_id"], "name": strat_row["name"], "why": why,
           "instrument": inst, "maker_fee_bps": ex.maker_fee_bps,
           "block_s": brief.stats.block / pd.Timedelta("1s"),
           "step_s": brief.grid.step / pd.Timedelta("1s"),
           "offsets": list(sim_maker.FAN_OFFSETS), "segments": {}}  # fmt: skip
    keys = [k for k, _ in BRACKET]
    fee_rows = ["brief", "latency 0", *(name for name, _ in sim_maker.FEE_ROWS)]
    checks: dict = {c: {} for c in archetypes.REGISTRY["market_making"].checks}
    last = None
    for seg, win in wins.items():
        panel = panels(seg)
        quotes = contract.Quotes({k: np.asarray(v, dtype=float) for k, v in
                                  contract.values(composed(panel), panel).items()})  # fmt: skip
        touch = sim_maker.touch_of(panel, inst, ex.tick_size, win)
        sims = sim_maker.models(quotes, panel, win, brief, inst, touch)
        scored = sim_maker.worse(sims["RA"], sims["PR"])
        (zero,) = sim_maker.models(quotes, panel, win, brief, inst, touch, (scored.model,),
                                   zero=True).values()  # fmt: skip
        rows = sim_maker.report(quotes, panel, win, brief, inst, touch, sims=sims,
                                scored=scored.model, zero=zero)  # fmt: skip
        by = {r["row"]: r for r in rows.iter_rows(named=True)}
        out["segments"][seg] = {
            "scored": scored.model,
            "models": {m: {k: _num(by[m][k]) for k in keys}
                       | {"held": _held_numbers(sim_maker.held(sims[m]))}
                       for m in sim_maker.MODELS},
            "rows": {r: {k: _num(by[r][k]) for k in keys} for r in fee_rows},
            "fan": {m: sim_maker.fill_fan(sims[m], win.end.value) for m in sim_maker.MODELS},
            "decomposition": {m: sim_maker.decompose(sims[m]) for m in sim_maker.MODELS},
            "pnl": sim_maker.pnl_lines(scored),
        }  # fmt: skip
        lat = ex.latency / pd.Timedelta("1ms")
        a, b = win.start.value, win.end.value
        for check in checks:
            if check == "costs":
                checks[check][seg] = dies.maker_costs(scored, zero, _rows(rows), lat)
                checks[check][seg]["fee_verdict"] = fees.judge(scored, brief, panel.step_ns,
                                                               (inst,))  # fmt: skip
            elif check == "capacity":
                checks[check][seg] = dies.maker_capacity(scored, a, b)
            elif check == "worst_window":
                checks[check][seg] = dies.worst_windows(scored, panel, win, brief)
            elif check == "mae":
                checks[check][seg] = dies.mae(dies.leg_paths(scored))
            elif check == "funding":
                checks[check][seg] = (dies.funding(scored, panel, win, inst)
                                      if scored.funding is not None
                                      else {"note": dies.NO_FUNDING})  # fmt: skip
            elif check == "toxicity":
                checks[check][seg] = dies.toxicity(sims)
            elif check == "queue":
                checks[check][seg] = dies.queue(sims)
        last = scored
    out["excerpt"] = sim_maker.excerpt(last) | {"segment": list(wins)[-1]}
    if any("note" in v for v in checks.get("funding", {}).values()):
        checks["funding"] = {"note": dies.NO_FUNDING}
    return out, checks


# the arbitrage's model table: what each fill model's pass did on a segment, in the probe's
# terms (sim_maker.arb_table), and its Sharpe
ARB_BRACKET = (("fills", "fills"), ("usd_traded", "USD traded"),
               ("captured_bps", "captured, bps a fill"), ("delay_bps", "delay"),
               ("slippage_bps", "slippage"), ("fee_bps", "fees"), ("net_bps", "net"),
               ("picked_off_share", "picked off"), ("net_usd_a_day", "net USD a day"),
               ("buy_share", "buys"), ("drift_usd_a_day", "drift USD a day"))  # fmt: skip
ARB_FILL_KEYS = ("captured_bps", "delay_bps", "slippage_bps", "fee_bps", "net_bps")
ARB_FILLS_SHOWN = 20


def _arb_panel(strat_row, by_name, why, panels, wins, brief) -> tuple[dict | None, dict]:
    """An arbitrage strategy on each segment, and its death checks, from one set of passes.

    Each segment runs on ``panels(segment)``, the panel the judge scored it
    on. For each: the four fill models in the probe's terms
    (:func:`blinded.sim_maker.arb_table`) with the hold rule's Sharpe of each;
    ``scored``, the worse of RA and PR by it; the sim rows' net
    (:func:`blinded.sim_maker.arb_report`: ``RA uncapped``, ``latency 0``,
    ``hedge 1 s`` and the hedge fee rows); the scored model's first
    :data:`ARB_FILLS_SHOWN` fills, each decomposed; and its P&L lines
    (:func:`blinded.sim_maker.arb_pnl_lines`). Then the excerpt of the last
    segment with both venues' touches (:func:`blinded.sim_maker.arb_excerpt`).
    The death checks are the archetype's, by segment, on the scored model.
    """
    from blinded import dies

    obj = next(s for s in contract.load(strat_row["code_path"]).strategies
               if s.name == strat_row["name"])  # fmt: skip
    if any(n not in by_name for n in obj.requires):
        return None, {}
    required = {n: next(f for f in contract.load(by_name[n]["code_path"]).features
                        if f.name == n) for n in obj.requires}  # fmt: skip
    quote = strat_row["legs"] or brief.target.instrument
    _, hedge = archetypes.arb_pair(brief)
    composed = positions(obj, required, (quote,), "arb")
    out = {"trial_id": strat_row["trial_id"], "name": strat_row["name"], "why": why,
           "quote": quote, "hedge": hedge,
           "block_s": brief.stats.block / pd.Timedelta("1s"),
           "step_s": brief.grid.step / pd.Timedelta("1s"), "segments": {}}  # fmt: skip
    rows_shown = ["brief", "RA uncapped", "latency 0", "hedge 1 s",
                  *(name for name, _ in sim_maker.HEDGE_FEE_ROWS)]  # fmt: skip
    keys = ("net_bps_per_hour", "sharpe_per_sqrt_hour", "fills", "units", "net_bps_per_unit")
    checks: dict = {c: {} for c in archetypes.REGISTRY["cross_venue_making"].checks}
    last = None
    for seg, win in wins.items():
        panel = panels(seg)
        quotes = contract.Quotes({k: np.asarray(v, dtype=float) for k, v in
                                  contract.values(composed(panel), panel).items()})  # fmt: skip
        sims = sim_maker.arb_models(quotes, panel, win, brief, quote, hedge)
        scored = sim_maker.worse(sims["RA"], sims["PR"])
        extra = {}
        common = dict(touch=scored.touch, mark=scored.mark)
        (extra["latency 0"],) = sim_maker.arb_models(quotes, panel, win, brief, quote, hedge,
                                                     which=(scored.model,), zero=True,
                                                     **common).values()  # fmt: skip
        (extra["hedge 1 s"],) = sim_maker.arb_models(quotes, panel, win, brief, quote, hedge,
                                                     which=(scored.model,), slow=True,
                                                     **common).values()  # fmt: skip
        (extra["RA uncapped"],) = sim_maker.arb_models(quotes, panel, win, brief, quote, hedge,
                                                       which=("RA",), through="full",
                                                       **common).values()  # fmt: skip
        rows = sim_maker.arb_report(quotes, panel, win, brief, quote, hedge, sims=sims,
                                    scored=scored.model, extra=extra, **common)  # fmt: skip
        by = {r["row"]: r for r in rows.iter_rows(named=True)}
        f = sim_maker.arb_fills(scored)
        k = min(ARB_FILLS_SHOWN, len(f["side"]))
        fills = [
            {
                "seconds": (int(f["ns"][i]) - win.start.value) / 1e9,
                "side": int(f["side"][i]),
                "price": float(f["price"][i]),
                "units": float(f["qty"][i]),
                "vwap": _num(f["vwap"][i]),
                **{key: _num(f[key][i]) for key in ARB_FILL_KEYS},
                "picked_off": bool(f["picked_off"][i]),
                "touch_through": bool(f["touch_through"][i]),
            }
            for i in range(k)
        ]
        out["segments"][seg] = {
            "scored": scored.model,
            "models": {m: {key: _num(v) for key, v in sim_maker.arb_table(sims[m]).items()}
                       | {"sharpe_per_sqrt_hour": _num(by[m]["sharpe_per_sqrt_hour"]),
                          "net_bps_per_hour": _num(by[m]["net_bps_per_hour"]),
                          "held": _held_numbers(sim_maker.held(sims[m]))}
                       for m in sim_maker.MODELS},
            "rows": {r: {key: _num(by[r][key]) for key in keys} for r in rows_shown},
            "fills": fills,
            "pnl": sim_maker.arb_pnl_lines(scored),
        }  # fmt: skip
        a, b = win.start.value, win.end.value
        for check in checks:
            if check == "costs":
                checks[check][seg] = dies.arb_costs(scored, _rows(rows))
                checks[check][seg]["fee_verdict"] = fees.judge_arb(scored, brief, panel.step_ns,
                                                                   hedge)  # fmt: skip
            elif check == "capacity":
                checks[check][seg] = dies.arb_capacity(scored, extra["RA uncapped"], a, b)
            elif check == "worst_window":
                checks[check][seg] = dies.worst_windows(scored, panel, win, brief)
            elif check == "mae":
                checks[check][seg] = dies.mae(dies.leg_paths(scored))
            elif check == "picked_off":
                checks[check][seg] = dies.picked_off(scored)
            elif check == "hedge":
                checks[check][seg] = dies.hedge(scored, extra["hedge 1 s"], by["brief"],
                                                by["hedge 1 s"])  # fmt: skip
            elif check == "stock":
                checks[check][seg] = dies.stock(scored)
            elif check == "counterparty":
                checks[check][seg] = dies.counterparty(scored, panel, a, b)
        last = scored
    out["excerpt"] = sim_maker.arb_excerpt(last) | {"segment": list(wins)[-1]}
    return out, checks


def _held_numbers(rule: dict) -> dict:
    return {"sharpe": _num(rule["sharpe"]), "hold_steps": float(rule["hold_steps"]),
            "block_steps": int(rule["block_steps"]), "blocks": int(rule["blocks"]),
            "releasable": bool(rule["releasable"])}  # fmt: skip


def _basket(name, pick, features, panel, wins, idx, brief, stash: dict) -> dict:
    """A basket archetype's charts: its feature's cross-sectional rank IC by horizon and by
    window, with the fan of the basket it implies at the template's parameters, and its
    strategy's net P&L leg by leg with the fan of its own rebalances. With no accepted
    strategy, ``bl.basket`` on the template's default score at the brief's parameters
    (:func:`_basket_default`), as a pairs archetype is drawn at its defaults."""
    feat, why, strat, swhy = pick
    legs = brief.family("basket").instruments
    params = brief.archetypes.params_for(name)
    out: dict = {"params": params}
    if feat is not None:
        out["feature"] = _basket_feature(feat, why, panel, wins, idx, brief, legs, params, stash)
    position, head = None, {}
    if strat is not None:
        position = _strategy_legs(strat, features, panel, legs)
        head = {"trial_id": strat["trial_id"], "name": strat["name"], "why": swhy}
    if position is None:
        score, words_ = _basket_default(name, legs, params)
        top, every = int(params["top"]), params["every"]
        position = contract.values(sim.basket(score, top, every), panel, legs)
        head = {"trial_id": None, "name": None,
                "why": f"no {name} strategy was accepted; bl.basket(top={top}, every={every}) "
                       f"on {words_}, the brief's parameters"}  # fmt: skip
    out["strategy"] = head | _basket_strategy(position, panel, wins, idx, brief, stash)
    stash |= {"position": position, "inst": legs[0]}
    return out


def _basket_default(name: str, legs, params: dict) -> tuple[dict, str]:
    """The score a basket archetype is drawn with when it has no strategy, and its words.

    Short-term reversal: minus each leg's return over ``lookback``, z-scored
    across the legs, so the names that fell most are bought. Statistical
    arbitrage: minus the trailing ``lookback`` z-score of each leg's log mid
    less the legs' mean log mid, its residual on an equal-weight market.
    """
    from blinded import features as F

    look = params["lookback"]
    if name == "short_term_reversal":
        score = F.xs_zscore(F.each(legs, lambda s: -F.ret(F.mid(s), look)))
        return score, f"minus each leg's {look} return, z-scored across the legs"
    resid = F.xs_demean(F.each(legs, lambda s: F.mid(s).log()))
    score = {s: -F.zscore(v, look) for s, v in resid.items()}
    return score, f"minus the {look} z-score of each leg's log mid less the legs' mean"


def _strategy_legs(strat_row, by_name, panel, legs) -> contract.Legs | None:
    """A basket strategy's positions on the panel, its features the accepted ones; None when one
    of them was never accepted."""
    obj = next(s for s in contract.load(strat_row["code_path"]).strategies
               if s.name == strat_row["name"])  # fmt: skip
    if any(n not in by_name for n in obj.requires):
        return None
    required = {n: next(f for f in contract.load(by_name[n]["code_path"]).features
                        if f.name == n) for n in obj.requires}  # fmt: skip
    return contract.values(positions(obj, required, legs)(panel), panel, legs)


def _basket_feature(feat, why, panel, wins, idx, brief, legs, params, stash) -> dict:
    """A basket feature's cross-sectional rank IC at each decay horizon and in each window of
    the block (30 minutes off bars), and the fan of ``bl.basket`` on it at ``params``."""
    from blinded.metrics import xs_rank_ic

    fn = next(f for f in contract.load(feat["code_path"]).features if f.name == feat["name"])
    x = contract.values(fn.fn(panel), panel, legs)
    stash["x"] = dict(x)
    big_x = np.column_stack([x[leg] for leg in legs])
    hs = _decay_horizons(brief)
    labels = [label(h) for h in hs]
    price = brief.target.price
    fwd = {leg: forward_returns(panel, leg, price, hs, brief.latency) for leg in legs}
    r = {lab: np.column_stack([fwd[leg][lab].to_numpy() for leg in legs]) for lab in labels}
    primary = label(brief.target.primary_horizon)
    every = label(brief.stats.block) if brief.grid.bars else "30min"
    g = panel.grid_ns
    wide = panel.with_columns({f"_score:{leg}": x[leg] for leg in legs})
    implied = sim.basket({leg: pl.col(f"_score:{leg}") for leg in legs}, int(params["top"]),
                         params["every"])  # fmt: skip
    pos = contract.values(implied, wide, legs)
    out = {"trial_id": feat["trial_id"], "name": feat["name"], "why": why, "horizons": labels,
           "primary": primary, "every": every, "params": params, "decay": {}, "windows": [],
           **_basket_offsets(panel, brief, legs), "fan": {}}  # fmt: skip
    for seg in wins:
        a, b = idx[seg]
        out["decay"][seg] = [_mean(xs_rank_ic(big_x[a:b], r[lab][a:b])) for lab in labels]
        c = xs_rank_ic(big_x[a:b], r[primary][a:b])
        key = g[a:b] - g[a:b] % to_ns(every)
        starts, first = np.unique(key, return_index=True)
        for start, lo, hi in zip(starts, first, np.r_[first[1:], b - a], strict=True):
            out["windows"].append({"start": str(pd.Timestamp(int(start), tz="UTC")),
                                   "segment": seg, "rank_ic": _mean(c[lo:hi])})  # fmt: skip
        if seg != "train":
            out["fan"][seg] = _basket_fan(pos, panel, a, b, out["offsets_ns"], brief)
    return out


def _basket_offsets(panel, brief, legs) -> dict:
    marks = panel.stream_of(legs[0], brief.target.price)
    flight = brief.latency.value + marks.delay_ns
    offsets = _fan_offsets(panel.step_ns, flight, marks.delay_ns, brief)
    out = {"offsets": [_signed(o) for o in offsets], "offsets_ns": offsets,
           "latency_ms": flight / 1e6}  # fmt: skip
    if brief.grid.bars:
        out["fan_scale"] = _fan_scale(brief)
    return out


def _basket_fan(position, panel, a: int, b: int, offsets: list[int], brief) -> dict:
    """The markout fan of a basket's rebalances in ``[a, b)``: at each step where a leg's
    position changes and the basket is not flat, the move of every leg weighted by its new
    position over the gross, so one notional across the legs; with weights of plus and minus
    one over ``top``, half the long legs' mean return less the short legs'. Rebalances
    closer than the fan's gap share prices, so they count as one draw in the error."""
    g = panel.grid_ns
    legs = list(position)
    held = np.column_stack([np.nan_to_num(np.asarray(position[leg], dtype=float)[a:b])
                            for leg in legs])  # fmt: skip
    changed = np.any(np.diff(np.vstack([np.zeros((1, len(legs))), held]), axis=0) != 0, axis=1)
    gross = np.abs(held).sum(axis=1)
    hit = np.flatnonzero(changed & (gross > 0))
    t = g[a:b][hit]
    w = held[hit] / gross[hit][:, None]
    price = brief.target.price
    moves = [np.zeros(len(t)) for _ in offsets]
    for i, leg in enumerate(legs):
        marks = panel.stream_of(leg, price)
        for k, m in enumerate(mk.mid_moves(lambda at, ev=marks: ev.asof(price, at), t, offsets,
                                           int(g[-1]))):  # fmt: skip
            moves[k] = moves[k] + w[:, i] * m
    return mk.fan(moves, np.ones(len(t)), cluster=mk.clusters(t, _fan_gap(brief)))


def _basket_strategy(position, panel, wins, idx, brief, stash: dict) -> dict:
    """A basket's net P&L per leg and in all on each segment, and the fan of its rebalances."""
    out = {"pnl": {}, "fan": {}, **_basket_offsets(panel, brief, list(position))}
    step_h = panel.step_ns / 3.6e12
    for seg, win in wins.items():
        s = sim.simulate_legs(position, panel, win, brief)
        keep = mk.downsample(len(s.net))
        net = np.cumsum(s.net)
        out["pnl"][seg] = {"hours": (keep * step_h).tolist(), "net": net[keep].tolist(),
                           "legs": {leg: np.cumsum(x.net)[keep].tolist()
                                    for leg, x in s.legs.items()},
                           "fills": int(s.fills.sum()), "net_bps": float(net[-1])}  # fmt: skip
        stash.setdefault("sims", {})[seg] = s
        if seg != "train":
            out["fan"][seg] = _basket_fan(position, panel, *idx[seg], out["offsets_ns"], brief)
    return out


def _mean(v: np.ndarray) -> float | None:
    ok = np.isfinite(v)
    return float(np.mean(v[ok])) if ok.any() else None


def _leader_follower(pick, brief) -> Family:
    """The lead-lag family of a pick: its own leader and follower, else every implied one."""
    feat, _, strat, _ = pick
    for row in (strat, feat):
        if row is None:
            continue
        module = contract.load(row["code_path"])
        obj = next(o for o in (*module.features, *module.strategies) if o.name == row["name"])
        leader = obj.roles.get("leader")
        if leader and row["legs"]:
            return Family("lead_lag", 1.0, leaders=(leader,), followers=(row["legs"],))
    return brief.family("lead_lag")


_PAIR_CACHE: dict = {}  # the implied pair, by brief, panel and train window: report then replay


def _archetype_pair(name, brief, panel, train) -> tuple[str, str]:
    """The pair a pairs archetype is drawn on when it has no strategy: the declared pair for
    cross-venue arbitrage and the basis trade; for pairs trading, of the implied pairs that
    are not declared, the one with the shortest median half-life on train (ties to the
    brief's order, and the first when none reverts). The search over every implied pair (18
    s for the 190 of twenty names on 15-minute bars) is kept for the process, so a report
    and its replay make it once."""
    mids = [c for c in panel.frame.columns if c.endswith(".mid")]
    key = (name, brief.name, train.start.value, train.end.value, len(panel),
           int(panel.grid_ns[-1]), *panel.frame.select(pl.col(mids).sum()).row(0))  # fmt: skip
    if key not in _PAIR_CACHE:
        _PAIR_CACHE[key] = _search_pair(name, brief, panel, train)
    return _PAIR_CACHE[key]


def _search_pair(name, brief, panel, train) -> tuple[str, str]:
    from blinded.diagnostics import pairs_table

    spec = brief.archetypes
    if name == "cross_venue_arbitrage" and spec.cross_venue:
        return tuple(spec.cross_venue[0])
    if name == "basis_trade" and spec.carry:
        return tuple(spec.carry[0])
    taken = {frozenset(p) for p in (*spec.cross_venue, *spec.carry)}
    free = tuple(p for p in brief.family("pairs").pairs if frozenset(p) not in taken)
    if len(free) == 1:
        return tuple(free[0])
    fam = Family("pairs", 1.0, pairs=free, defaults=spec.params_for(name))
    rows = pairs_table(dataclasses.replace(brief, families=(fam,)), panel, train)
    lives = [r["median_half_life_s"] for r in rows]
    finite = [i for i, v in enumerate(lives) if np.isfinite(v)]
    best = min(finite, key=lambda i: lives[i]) if finite else 0
    return tuple(free[best])


def _feature_values(row, panel) -> dict:
    """A pairs feature's values on the panel, for its half-life."""
    fn = next(f for f in contract.load(row["code_path"]).features if f.name == row["name"])
    return {"x": np.asarray(contract.values(fn.fn(panel), panel), dtype=float)}


def _dies(arch, entry: dict, stash: dict, panel, wins, idx, brief) -> dict:
    """The archetype's death checks (:mod:`blinded.dies`), each by segment, in its order."""
    from blinded import dies

    position, x = stash.get("position"), stash.get("x")
    inst = stash.get("inst")
    no_strategy = {"note": "no accepted strategy of this archetype"}
    sims = {}
    if position is not None:
        for seg, win in wins.items():
            s = (stash.get("sims") or {}).get(seg)
            if s is None:
                s = _run(position, panel, win, brief, inst)(None, None)
            sims[seg] = s
    out: dict = {}
    for check in arch.checks:
        if check == "crowding":
            out[check] = {"note": dies.NOT_MEASURABLE}
            continue
        if check == "funding" and (position is None or not entry.get("pairs")):
            out[check] = {"note": dies.NO_FUNDING if position is not None else
                          no_strategy["note"]}  # fmt: skip
            continue
        if check == "venue_feed":
            b = entry["pairs"][0]["pair"][1]
            out[check] = {seg: dies.venue_feed(panel, win, b, entry.get("curves") or [], seg)
                          for seg, win in wins.items()}  # fmt: skip
            continue
        if check == "race":
            out[check] = _race(entry, stash, panel, wins, idx, brief)
            continue
        if check == "half_life":
            pair = (entry.get("pairs") or [None])[0]
            out[check] = {seg: dies.half_life(x, *idx[seg], panel.step_ns / 1e9, pair, seg)
                          for seg in wins}  # fmt: skip
            continue
        if position is None:
            out[check] = no_strategy
            continue
        per = {}
        for seg, win in wins.items():
            s = sims[seg]
            if check == "costs":
                rows = _rows(sim.report(position, panel, win, brief, inst))
                per[seg] = dies.costs(_run(position, panel, win, brief, inst), s, rows, brief)
                per[seg]["fee_verdict"] = fees.judge(
                    s, brief, panel.step_ns, _traded(position, (inst,), brief)
                )
            elif check == "capacity":
                per[seg] = dies.capacity(s, panel, win, inst)
            elif check == "worst_window":
                per[seg] = dies.worst_windows(s, panel, win, brief)
            elif check == "mae":
                per[seg] = dies.mae(dies.leg_paths(s))
            elif check == "breadth":
                per[seg] = dies.breadths(s)
            elif check == "whipsaw":
                per[seg] = dies.whipsaws(s, arch.holding[0], panel.step_ns)
            elif check == "funding":
                per[seg] = dies.funding(s, panel, win, entry["pairs"][0]["pair"][1])
        if check == "funding" and any("note" in v for v in per.values()):
            per = {"note": dies.NO_FUNDING}
        out[check] = per
    return out


def _run(position, panel, win, brief, inst):
    """``run(latency, fee)``: the position simulated at a latency and a fee for every leg,
    each ``None`` for the brief's own."""
    ex = brief.execution

    def run(latency, fee):
        lat = ex.latency if latency is None else latency
        if isinstance(position, contract.Legs):
            fees = {leg: ex.fee(leg) if fee is None else fee for leg in position}
            return sim.simulate_legs(position, panel, win, brief, lat, fees)
        return sim.simulate(position, panel, win, inst, brief.target.price, lat,
                            ex.fee(inst) if fee is None else fee, ex.min_rebalance,
                            ex.max_position, brief.stats.block,
                            rules=brief.judge_rules)  # fmt: skip

    return run


def _race(entry: dict, stash: dict, panel, wins, idx, brief) -> dict:
    """D2: the feature's rank IC from the decision against from the fill, the order's cost in
    flight, and for lead-lag the C table; on bars, only that it cannot be measured."""
    from blinded import dies

    out: dict = {}
    curves = entry.get("curves") or []
    if curves:
        keys = ("lag_at_peak_ms", "c_at_peak", "c_after_fill", "best_after_fill",
                "usable_by_taker")  # fmt: skip
        out["c_table"] = [{"leader": c["leader"], "follower": c["follower"],
                           "segments": {seg: {k: c["segments"][seg][k] for k in keys}
                                        for seg in c["segments"]}} for c in curves]  # fmt: skip
    if brief.grid.bars:
        return out | {"note": dies.BELOW_THE_BAR}
    x = stash.get("x")
    if x is None:
        return out | {"note": "no accepted feature of this archetype"}
    if "x_inst" not in stash:
        return out | {"note": "a pair's race is the C table across its two venues; its feature "
                              "has no one leg to measure from the fill"}  # fmt: skip
    inst, price = stash["x_inst"], brief.target.price
    hs = stash["horizons"]
    hs = tuple(h for h in hs if brief.data.exact(h))
    lab = [label(h) for h in hs]
    zero = forward_returns(panel, inst, price, hs, pd.Timedelta(0))
    fill = forward_returns(panel, inst, price, hs, brief.latency)
    r0 = {h: zero[h].to_numpy() * 1e4 for h in lab}
    r1 = {h: fill[h].to_numpy() * 1e4 for h in lab}
    book = panel.stream_of(inst, "bid")
    strat = entry.get("strategy") or {}
    g = panel.grid_ns
    for seg in wins:
        a, b = idx[seg]
        half = mk.half_spread_bps(book.asof("bid", g[a:b]), book.asof("ask", g[a:b]))
        entry_bps = ((strat.get("markout") or {}).get(seg) or {}).get("entry_bps")
        out[seg] = dies.race(x, r0, r1, a, b, stash["direction"], entry_bps, half)
    return out


def _decay_horizons(brief) -> tuple[pd.Timedelta, ...]:
    """The report's decay horizons: its ladder and the brief's, less those whose returns end
    at a stamp a thinned file does not keep and, on bars, those shorter than a bar."""
    ladder = {to_ns(h) for h in mk.DECAY_HORIZONS
              if not brief.grid.bars or to_ns(h) >= brief.grid.step.value}  # fmt: skip
    spans = ladder | {h.value for h in brief.target.horizons}
    return tuple(pd.Timedelta(ns, unit="ns") for ns in sorted(spans)
                 if brief.data.exact(brief.latency + pd.Timedelta(ns, unit="ns")))  # fmt: skip


def _pick(usable, family: str, final_id, mode: str, brief) -> tuple:
    """The family's feature and strategy for the charts, each with why it was chosen."""
    feats, strats = usable("feature", family), usable("strategy", family)
    by_name = {t["name"]: t for t in feats}
    final = next((t for t in strats if t["trial_id"] == final_id), None)
    primary = label(brief.target.primary_horizon)
    feat, why = None, ""
    if final is not None:
        need = _requires(final)
        feat, why = (by_name.get(need[0]) if need else None), "the final strategy's feature"
    if feat is None:
        released = [t for t in feats if t["released_bool"]]
        feat, why = (released[-1], "the last feature the Ladder released") if released else (
            None, "")  # fmt: skip
    if feat is None and feats:
        scored_on = "the window it was scored on" if mode == "naive" else "train"
        feat = max(feats, key=lambda t: abs(_train_ic(t, primary) or 0.0))
        why = f"the accepted feature with the largest rank IC on {scored_on}"
    strat, swhy = final, "the final strategy"
    if strat is None and strats:
        vals = [(_val_value(t), t) for t in strats]
        vals = [(v, t) for v, t in vals if v is not None]
        if vals:
            strat = max(vals, key=lambda vt: vt[0])[1]
            swhy = "the accepted strategy with the best validation Sharpe"
            swhy += "" if strat["released_bool"] else "; it was not released"
    return feat, why, strat, swhy


def _single_leg(pick, features, panel, wins, idx, brief, stash=None) -> dict:
    """``feature`` and ``strategy`` of one family's pick, whichever exist. ``stash``, when
    given, receives the feature's values and the strategy's position and sims."""
    feat, why, strat, swhy = pick
    out = {}
    if feat is not None:
        out["feature"] = _feature_diagnostics(feat, why, panel, wins, idx, brief, stash)
    if strat is not None:
        s_out = _strategy_diagnostics(strat, features, swhy, panel, wins, idx, brief, stash)
        if s_out:
            out["strategy"] = s_out
    return out


def _feature_diagnostics(feat, why, panel, wins, idx, brief, stash=None) -> dict:
    """The fan, decay, costs and IC per window of one feature on its own instrument."""
    segs = list(wins)
    g = panel.grid_ns
    price = brief.target.price
    ex = brief.execution
    primary = label(brief.target.primary_horizon)
    inst = feat["legs"] or brief.target.instrument
    book = panel.stream_of(inst, "bid")
    fn = next(f for f in contract.load(feat["code_path"]).features if f.name == feat["name"])
    x = np.asarray(contract.values(fn.fn(panel), panel), dtype=float)
    hs = _decay_horizons(brief)
    labels = [label(h) for h in hs]
    fwd = forward_returns(panel, inst, price, hs, brief.latency)
    r = [fwd[lab].to_numpy() * 1e4 for lab in labels]
    marks = panel.stream_of(inst, price)
    flight = brief.latency.value + marks.delay_ns
    offsets = _fan_offsets(panel.step_ns, flight, marks.delay_ns, brief)
    a, b = idx["train"]
    every = label(brief.stats.block) if brief.grid.bars else "30min"
    rp = r[labels.index(primary)]
    train_ic = mk.rank_ic(x[a:b], rp[a:b])
    direction = -1.0 if train_ic is not None and train_ic < 0 else 1.0
    if stash is not None:
        stash |= {"x": x, "x_inst": inst, "direction": direction, "horizons": hs}
    lo, hi = _tails(x[a:b])
    f_out = {"trial_id": feat["trial_id"], "name": feat["name"], "why": why,
             "horizons": labels, "offsets": [_signed(o) for o in offsets],
             "latency_ms": flight / 1e6, "primary": primary, "every": every,
             "events": {"lo": lo, "hi": hi, "direction": direction, "train_ic": train_ic},
             "decay": {}, "fan": {}, "costs": {}, "windows": []}  # fmt: skip
    if brief.grid.bars:
        f_out["fan_scale"] = _fan_scale(brief)
    for seg in segs:
        a, b = idx[seg]
        f_out["decay"][seg] = [mk.rank_ic(x[a:b], ri[a:b]) for ri in r]
        half = mk.half_spread_bps(book.asof("bid", g[a:b]), book.asof("ask", g[a:b]))
        f_out["costs"][seg] = {"half_spread_bps": half, "fee_bps": ex.fee(inst),
                               "round_trip_bps": None if half is None
                               else 2 * (ex.fee(inst) + half)}  # fmt: skip
        if seg != "train":
            edge = (-np.inf if lo is None else lo, np.inf if hi is None else hi)
            hit, side = mk.events(x[a:b], g[a:b], *edge, _fan_gap(brief))
            t = g[a:b][hit]
            moves = mk.mid_moves(lambda at: marks.asof(price, at), t, offsets, int(g[-1]))
            fan = mk.fan(moves, side * direction)
            spread = mk.half_spread_bps(book.asof("bid", t), book.asof("ask", t))
            f_out["fan"][seg] = fan | _breakeven(fan, offsets.index(flight), spread,
                                                 2 * ex.fee(inst))  # fmt: skip
        for start, ic in mk.window_ics(x[a:b], rp[a:b], g[a:b], to_ns(every)):
            f_out["windows"].append({"start": str(pd.Timestamp(start, tz="UTC")),
                                     "segment": seg, "rank_ic": ic})  # fmt: skip
    return f_out


def _lead_lag_diagnostics(family, panel, wins, brief) -> list[dict]:
    """:math:`C(\\tau)` of every leader and follower on each segment, with its summary."""
    from blinded.diagnostics import lead_lag

    out = []
    for seg, win in wins.items():
        for i, row in enumerate(lead_lag(brief, panel, win, family)):
            if len(out) <= i:
                out.append({k: row[k] for k in ("leader", "follower", "lags_ms", "fill_ms",
                                                "step_ms")} | {"segments": {}})  # fmt: skip
            out[i]["segments"][seg] = {k: v for k, v in row.items() if k not in out[i]}
    return out


def _pair_picks(strats, final_id, family) -> list[tuple]:
    """Which pairs to chart: the final strategy's pair, else that of the last released pairs
    strategy, else of the best one on validation, else every configured pair at the brief's
    defaults."""
    final = next((t for t in strats if t["trial_id"] == final_id), None)
    if final is not None:
        return [(tuple(final["legs"].split("/")), final, "the final strategy")]
    released = [t for t in strats if t["released_bool"]]
    if released:
        return [(tuple(released[-1]["legs"].split("/")), released[-1],
                 "the last pairs strategy the Ladder released")]  # fmt: skip
    vals = [(v, t) for t in strats if (v := _val_value(t)) is not None]
    if vals:
        best = max(vals, key=lambda vt: vt[0])[1]
        why = "the pairs strategy with the best validation Sharpe; it was not released"
        return [(tuple(best["legs"].split("/")), best, why)]
    return [(tuple(pair), None, "no pairs strategy was accepted") for pair in family.pairs]


def _pair_diagnostics(pair, strat_row, why, features, panel, wins, idx, brief, family,
                      stash=None, fallback=None) -> dict:  # fmt: skip
    """The spread against its bands, the hedge ratio and half-life per clock window, and the
    markout fan of the spread by z quintile, for one pair.

    The fit's ``window`` and ``every`` and the bands' ``z_in`` and ``z_out``
    come from the strategy's own ``bl.pairs`` call when its code makes exactly
    one with literal arguments, else from the family's ``defaults``. The
    shading is the strategy's own position in A, its features found as the
    daemon finds them (accepted, else in its own file), or with no strategy
    ``fallback`` (``bl.carry`` for a basis trade), else ``bl.pairs`` at those
    parameters.
    """
    from blinded.features import cross

    a, b = pair
    params, source = _pairs_params(strat_row, pair, family.defaults)
    window, every = params["window"], params["every"]
    beta, mu, var, y, x = cross._fit(a, b, window, every)
    fit = panel.frame.lazy().select(beta=beta, dev=y - beta * x - mu, sd=var.sqrt()).collect()
    beta = fit["beta"].fill_null(np.nan).to_numpy()
    dev = fit["dev"].fill_null(np.nan).to_numpy() * 1e4
    sd = fit["sd"].fill_null(np.nan).to_numpy() * 1e4
    z = dev / sd
    required = None
    if strat_row is not None:
        module = contract.load(strat_row["code_path"])
        obj = next(s for s in module.strategies if s.name == strat_row["name"])
        local = {f.name: f for f in module.features}
        required = {n: next(f for f in contract.load(features[n]["code_path"]).features
                            if f.name == n) if n in features else local.get(n)
                    for n in obj.requires}  # fmt: skip
    if required is not None and None not in required.values():
        legs = contract.values(positions(obj, required, pair)(panel), panel, pair)
    else:
        if fallback is None:
            fallback = sim.pairs(a, b, params["z_in"], params["z_out"], window, every)
        legs = contract.values(fallback, panel, pair)
    held = np.sign(np.nan_to_num(np.asarray(legs[a], dtype=float)))
    if stash is not None:
        stash |= {"position": legs, "inst": a}
    g = panel.grid_ns
    step_h, step_s = panel.step_ns / 3.6e12, panel.step_ns / 1e9
    price = brief.target.price
    ma, mb = panel.stream_of(a, price), panel.stream_of(b, price)
    flight = brief.latency.value + ma.delay_ns
    offsets = _fan_offsets(panel.step_ns, flight, ma.delay_ns, brief)
    every_ns = to_ns(every)
    ex = brief.execution
    z_in = float(params["z_in"])
    out = {"pair": list(pair), "why": why, "params": params, "params_from": source,
           "trial_id": strat_row["trial_id"] if strat_row is not None else None,
           "name": strat_row["name"] if strat_row is not None else None,
           "every": _signed(every_ns), "offsets": [_signed(o) for o in offsets],
           "latency_ms": flight / 1e6, "spread": {}, "windows": [], "fan": {},
           "costs": {}}  # fmt: skip
    if brief.grid.bars:
        out["fan_scale"] = _fan_scale(brief)
    out["random_walk_half_life_s"] = mk.random_walk_half_life(every_ns // panel.step_ns, step_s)
    for seg in wins:
        lo, hi = idx[seg]
        keep = mk.downsample(hi - lo)
        spans = np.diff(np.r_[keep, hi - lo])
        share = np.add.reduceat(held[lo:hi], keep) / spans
        out["spread"][seg] = {"hours": (keep * step_h).tolist(),
                              "dev_bps": _list(dev[lo:hi][keep]),
                              "sd_bps": _list(sd[lo:hi][keep]),
                              "position": _list(share)}  # fmt: skip
        starts = mk.window_half_lives(dev[lo:hi], g[lo:hi], every_ns, step_s)
        first = np.searchsorted(g[lo:hi], [w["start"] for w in starts])
        for w, i in zip(starts, first, strict=True):
            i = min(max(i, 0), hi - lo - 1)
            out["windows"].append({"start": str(pd.Timestamp(w["start"], tz="UTC")),
                                   "segment": seg, "beta": _num(beta[lo + i]),
                                   "phi": w["phi"], "half_life_s": w["half_life_s"],
                                   "reverts": w["reverts"]})  # fmt: skip
        book_a, book_b = panel.stream_of(a, "bid"), panel.stream_of(b, "bid")
        half_a = mk.half_spread_bps(book_a.asof("bid", g[lo:hi]), book_a.asof("ask", g[lo:hi]))
        half_b = mk.half_spread_bps(book_b.asof("bid", g[lo:hi]), book_b.asof("ask", g[lo:hi]))
        fitted = np.abs(beta[lo:hi][np.isfinite(beta[lo:hi])])
        hedge = float(np.median(fitted)) if len(fitted) else None
        trip = None
        if None not in (half_a, half_b, hedge):
            trip = 2 * (ex.fee(a) + half_a) + 2 * hedge * (ex.fee(b) + half_b)
        out["costs"][seg] = {"half_spread_bps": half_a, "half_spread_b_bps": half_b,
                             "hedge": hedge, "round_trip_bps": trip}  # fmt: skip
        if seg != "train":
            # past the entry band, a trade back towards the mean: short the spread when z is high
            hit, side = mk.events(z[lo:hi], g[lo:hi], -z_in, z_in, _fan_gap(brief))
            t, bt = g[lo:hi][hit], beta[lo:hi][hit]
            moves = mk.spread_moves(lambda at: ma.asof(price, at), lambda at: mb.asof(price, at),
                                    bt, t, offsets, int(g[-1]))  # fmt: skip
            fan = mk.fan(moves, -side)
            h = np.abs(bt[np.isfinite(bt)])
            h = float(np.median(h)) if len(h) else None
            sa = mk.half_spread_bps(book_a.asof("bid", t), book_a.asof("ask", t))
            sb = mk.half_spread_bps(book_b.asof("bid", t), book_b.asof("ask", t))
            spread = None if None in (sa, sb, h) else sa + h * sb
            fees = None if h is None else 2 * (ex.fee(a) + h * ex.fee(b))
            out["fan"][seg] = fan | {"hedge": h} | _breakeven(fan, offsets.index(flight),
                                                              spread, fees)  # fmt: skip
    return out


def _pairs_params(strat_row, pair, defaults: dict) -> tuple[dict, str]:
    """``window``, ``every``, ``z_in`` and ``z_out`` of a pairs strategy's one ``pairs`` call.

    Read from the strategy's source with :mod:`ast` when its function makes
    exactly one call to ``pairs`` on the pair's two instruments and every
    argument given is a literal; arguments left out take ``bl.pairs``'s own
    defaults. Otherwise, the family's ``defaults``.
    """
    import ast
    import inspect

    base = {k: defaults[k] for k in ("window", "every", "z_in", "z_out")}
    if strat_row is None:
        return base, "the brief's pairs defaults"
    try:
        tree = ast.parse(Path(strat_row["code_path"]).read_text())
    except (OSError, SyntaxError):
        return base, "the brief's pairs defaults"
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == strat_row["name"]), None)  # fmt: skip
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", getattr(c.func, "id", None)) == "pairs"
             ] if fn else []  # fmt: skip
    names = list(inspect.signature(sim.pairs).parameters)
    found = {}
    if len(calls) == 1:
        try:
            found = {names[i]: ast.literal_eval(v) for i, v in enumerate(calls[0].args)}
            found |= {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
        except (ValueError, IndexError, TypeError):
            found = {}
    if not found or {found.get("a"), found.get("b")} != set(pair):
        return base, "the brief's pairs defaults; the strategy's own could not be read"
    own = {k: v.default for k, v in inspect.signature(sim.pairs).parameters.items()
           if k in base} | {k: found[k] for k in base if k in found}  # fmt: skip
    return own, "the strategy's own bl.pairs call"


def _list(values: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in values]


def _strategy_diagnostics(strat_row, by_name, why, panel, wins, idx, brief, stash=None
                          ) -> dict | None:  # fmt: skip
    """A position strategy's P&L lines and order markout fan by segment; None for one whose
    features were never accepted, and for a market-making strategy, which has quotes rather
    than positions (its bracket is :func:`_maker_bracket`)."""
    obj = next(s for s in contract.load(strat_row["code_path"]).strategies
               if s.name == strat_row["name"])  # fmt: skip
    if any(n not in by_name for n in obj.requires) or obj.archetype == "market_making":
        return None
    required = {n: next(f for f in contract.load(by_name[n]["code_path"]).features
                        if f.name == n) for n in obj.requires}  # fmt: skip
    inst, price = strat_row["legs"] or brief.target.instrument, brief.target.price
    composed = positions(obj, required, (inst,))
    pos = np.asarray(contract.values(composed(panel), panel), dtype=float)
    g = panel.grid_ns
    ex = brief.execution
    book, marks = panel.stream_of(inst, "bid"), panel.stream_of(inst, price)
    flight = brief.latency.value + book.delay_ns
    offsets = _fan_offsets(panel.step_ns, flight, marks.delay_ns, brief)
    step_h = panel.step_ns / 3.6e12
    out = {"trial_id": strat_row["trial_id"], "name": strat_row["name"], "why": why,
           "fee_bps": ex.fee(inst), "offsets": [_signed(o) for o in offsets],
           "latency_ms": flight / 1e6, "pnl": {}, "markout": {}}  # fmt: skip
    if brief.grid.bars:
        out["fan_scale"] = _fan_scale(brief)
    for seg, win in wins.items():
        s = sim.simulate(pos, panel, win, inst, price, brief.latency, ex.fee(inst),
                         ex.min_rebalance, ex.max_position, brief.stats.block,
                         rules=brief.judge_rules)  # fmt: skip
        keep = mk.downsample(len(s.net))
        mid = np.cumsum(s.mid_pnl)
        after = np.cumsum(s.mid_pnl - s.spread_bps)
        net = np.cumsum(s.net)
        fills = int(s.fills.sum())
        out["pnl"][seg] = {"hours": (keep * step_h).tolist(), "mid": mid[keep].tolist(),
                           "after_spread": after[keep].tolist(), "net": net[keep].tolist(),
                           "fills": fills, "net_bps": float(net[-1])}  # fmt: skip
        a = idx[seg][0]
        traded = np.nonzero(s.trade)[0]
        sent = g[a + traded]
        t_fill = sent + brief.latency.value + book.delay_ns
        side, size = np.sign(s.trade[traded]), np.abs(s.trade[traded])
        fill = np.where(side > 0, book.asof("ask", t_fill), book.asof("bid", t_fill))
        moves = mk.mid_moves(lambda at: marks.asof(price, at), sent, offsets, int(g[-1]))
        groups = mk.clusters(sent, _fan_gap(brief))
        fan = mk.fan(moves, side, size, groups)
        entry = mk.entry_cost(side, size, fill, marks.asof(price, sent))
        runs = int(groups[-1]) + 1 if len(groups) else 0
        out["markout"][seg] = fan | {"fills": fills, "entry_bps": entry, "clusters": runs}
        if stash is not None:
            stash.setdefault("sims", {})[seg] = s
    if stash is not None:
        stash |= {"position": pos, "inst": inst}
    return out


def _fan_scale(brief) -> int:
    """How much longer the fan's window is on bars: one bar for each second of it (on hourly
    bars, 10 hours before the send to 60 after, events 70 hours apart); 1 elsewhere."""
    return int(brief.grid.step.value // 10**9) if brief.grid.bars else 1


def _fan_gap(brief) -> int:
    return to_ns(mk.FAN_GAP) * _fan_scale(brief)


def _fan_offsets(step_ns: int, flight_ns: int, delay_ns: int, brief) -> list[int]:
    """Every grid step from :data:`~blinded.markouts.FAN_BEFORE` before the send to
    :data:`~blinded.markouts.FAN_AFTER` after it, at 100 ms or coarser, and the fill; on
    bars the window is :func:`_fan_scale` times as long."""
    step = max(int(step_ns), to_ns("100ms"))
    scale = _fan_scale(brief)
    before, after = to_ns(mk.FAN_BEFORE) * scale, to_ns(mk.FAN_AFTER) * scale
    return _send_offsets(range(-(before // step) * step, after + 1, step), flight_ns, delay_ns,
                         brief)  # fmt: skip


def _tails(x: np.ndarray) -> tuple[float | None, float | None]:
    """The train quantiles :data:`~blinded.markouts.FAN_TAILS` past which a value is an event."""
    ok = x[np.isfinite(x)]
    if len(ok) < 20:
        return None, None
    lo, hi = np.quantile(ok, mk.FAN_TAILS)
    return float(lo), float(hi)


def _breakeven(fan: dict, k_fill: int, half_spread: float | None, fees: float | None) -> dict:
    """Where a taker who fills at offset ``k_fill`` and crosses back out breaks even.

    The move at the fill plus the spread crossed twice, ``2 * half_spread``,
    and with ``fees``, the round trip's fees, on top.
    """
    at_fill = fan["mean_bps"][k_fill]
    trip = None if half_spread is None else 2 * half_spread
    even = None if None in (at_fill, trip) else at_fill + trip
    return {"at_fill_bps": at_fill, "spread_round_trip_bps": trip, "fees_round_trip_bps": fees,
            "breakeven_bps": even,
            "breakeven_fees_bps": None if None in (even, fees) else even + fees}  # fmt: skip


def _send_offsets(offsets_ns, flight_ns: int, delay_ns: int, brief) -> list[int]:
    """Offsets from the send, in ns, at which to read the mark, with the fill among them.

    The send is the decision at :math:`g`, which saw what was available by
    then, so an offset :math:`\tau` reads the mark available at :math:`g +
    \tau`, stamp :math:`g + \tau - d`. The fill is ``flight_ns`` after the
    send, the latency plus the feed delay. On a thinned file the offsets whose
    stamp it does not keep are left out.
    """
    offsets = set(offsets_ns) | {flight_ns}
    keep = (o for o in offsets if brief.data.exact(pd.Timedelta(o - delay_ns, unit="ns")))
    return sorted(keep)


def _signed(ns: int) -> str:
    """A label for an offset that may be negative: ``-60s``, ``0ms``, ``50ms``, ``5min``."""
    return ("-" if ns < 0 else "") + label(pd.Timedelta(abs(ns), unit="ns"))


def _train_ic(row, horizon: str) -> float | None:
    m = json.loads(row["train_metrics_json"] or "{}").get("horizons", {}).get(horizon) or {}
    return _num(m.get("rank_ic"))


def _val_value(row) -> float | None:
    return _num(json.loads(row["val_metrics_json"] or "{}").get("value"))


def _rows(rows) -> dict:
    return {r["row"]: {k: _num(v) for k, v in r.items() if k not in ("row", "note", "model")
                       and not (k in sim.ONE_ROW and v is None)}
            for r in rows.iter_rows(named=True)}  # fmt: skip


def _curves(position, panel, win, brief, legs=(), kept: dict | None = None) -> dict:
    """Cumulative net P&L on test for the brief variant and the two unrealistic ones; with
    ``kept``, the brief variant's sim is left in it under ``brief``."""
    if position is None:
        return {}
    ex = brief.execution
    zero = pd.Timedelta(0)
    if isinstance(position, contract.Quotes):
        return _maker_test(position, panel, win, brief, legs)[0]
    if isinstance(position, contract.Legs):
        fees = {leg: ex.fee(leg) for leg in position}
        variants = {"brief": (ex.latency, fees), "latency 0": (zero, fees),
                    "latency 0, fees 0": (zero, dict.fromkeys(position, 0.0))}  # fmt: skip
        out = {}
        for name, (lat, fee) in variants.items():
            s = sim.simulate_legs(position, panel, win, brief, lat, fee)
            out[name] = np.cumsum(s.net).tolist()
            if kept is not None and name == "brief":
                kept["brief"] = s
        return out
    (inst,) = legs or (brief.target.instrument,)
    variants = {
        "brief": (ex.latency, ex.fee(inst)),
        "latency 0": (zero, ex.fee(inst)),
        "latency 0, fees 0": (zero, 0.0),
    }
    curves = {}
    for name, (latency, fee) in variants.items():
        s = sim.simulate(
            position,
            panel,
            win,
            inst,
            brief.target.price,
            latency,
            fee,
            ex.min_rebalance,
            ex.max_position,
            brief.stats.block,
            rules=brief.judge_rules,
        )
        curves[name] = np.cumsum(s.net).tolist()
        if kept is not None and name == "brief":
            kept["brief"] = s
    return curves


def gather(run: RunDir) -> dict:
    """The whole report as data: ledger numbers plus :func:`recompute`."""
    ledger = run.ledger()
    manifest = run.manifest
    run_row = ledger.run(run.run_id)
    trials = ledger.trials()
    iterations = ledger.iterations(run.run_id)
    releases = {"feature": ledger.releases("feature"), "strategy": ledger.releases("strategy")}
    scored = ledger.test_score(run.run_id)
    final_id = run_row["final_trial_id"] if run_row else None
    protected = run.mode == "protected"
    primary = label(run.brief().target.primary_horizon)
    model = {
        "run_id": run.run_id,
        "mode": run.mode,
        "blinded": manifest.get("blinded", __version__),
        "judge_rules": run.judge_rules,
        "claude_version": run_row["claude_version"] if run_row else None,
        "budget": json.loads(run_row["budget_json"]) if run_row and run_row["budget_json"] else {},
        "windows": manifest["windows"],
        "segments": {
            n: {"rows": s["rows"], "sha256": s["sha256"]} for n, s in manifest["segments"].items()
        },  # fmt: skip
        "source_name": manifest["source"]["file"],
        "counts": _counts(trials),
        "iterations": _iteration_stats(iterations),
        "ladder": [_release(r) for r in releases["feature"] + releases["strategy"]],
        "rejected": _rejected(trials),
        "ladder_points": _ladder_points(trials) if protected else [],
        "train_vs_val": _train_vs_val(trials, primary) if protected else [],
        "strategy_points": _strategy_points(trials, run.mode),
        "held_back": _held_back(trials),
        "probes": [dict(r) for r in trials if r["kind"] == "probe"],
        "final_id": final_id,
        "test": json.loads(scored["metrics_json"]) if scored else None,
        "deflation": _deflation(ledger, final_id, run.brief()),
        "reproduce": _reproduce(run, final_id),
        "appendix": _appendix(run, trials),
    }
    ledger.close()
    brief = run.brief()
    if brief.has_families:
        model["families"] = _families(run, trials, final_id)
    if brief.has_archetypes:
        model["families"] = _edges(run, trials, final_id)
        model["archetypes"] = _archetypes_model(run, trials, final_id)
    model["recompute"] = recompute(run, final_id)
    model["verdict"] = _verdict(model)
    model["verdict"]["rules"] = _rules_line(model)
    current = _current_rules_line(model)
    if current:
        model["verdict"]["rules_current"] = current
    unfenced = _fence_line(run)
    if unfenced:
        model["verdict"]["fence"] = unfenced
    if brief.data.cost_model is not None:
        line = brief.data.cost_model.line()
        model["verdict"]["cost_model"] = line[0].upper() + line[1:] + "."
    naive = _naive_sibling(run)
    if naive is not None:
        model["naive"] = naive
    return model


def _counts(trials) -> dict:
    out = {"feature": {"accepted": 0, "rejected": 0}, "strategy": {"accepted": 0, "rejected": 0}}
    for t in trials:
        if t["kind"] in out:
            out[t["kind"]]["accepted" if t["status"] == "accepted" else "rejected"] += 1
    hashes = {k: {r["code_hash"] for r in trials if r["kind"] == k and r["code_hash"]}
              for k in ("feature", "strategy")}  # fmt: skip
    out["distinct"] = {k: len(v) for k, v in hashes.items()}
    return out


def _families(run: RunDir, trials, final_id) -> list[dict]:
    """Per family: what was submitted, accepted and released of each kind, the released bests,
    the sessions of each phase, and whether the final strategy is the family's."""
    ledger = run.ledger()
    bests = {(r["phase"], r["family"]): r["best_after"] for r in ledger.releases()}
    ledger.close()
    state = run.driver / "state.json"
    state = json.loads(state.read_text()) if state.is_file() else {}
    phases, assigned = state.get("phases", {}), state.get("families", {})
    final = next((t for t in trials if t["trial_id"] == final_id), None)
    out = []
    for fam in run.brief().families:
        row = {"family": fam.name, "share": fam.share,
               "final": final is not None and final["family"] == fam.name}  # fmt: skip
        for kind in ("feature", "strategy"):
            mine = [t for t in trials if t["kind"] == kind and t["family"] == fam.name]
            row[kind] = {"submitted": len(mine),
                         "accepted": sum(t["status"] == "accepted" for t in mine),
                         "released": sum(bool(t["released_bool"]) for t in mine),
                         "best": bests.get((kind, fam.name))}  # fmt: skip
        row["sessions"] = {
            phase: sum(1 for n, p in phases.items() if p == phase and assigned.get(n) == fam.name)
            for phase in ("features", "strategies")
        }
        out.append(row)
    return out


def _edges(run: RunDir, trials, final_id) -> list[dict]:
    """Per edge, the rows :func:`_families` gives per family, with the edge's members."""
    ledger = run.ledger()
    bests = {(r["phase"], r["group"]): r["best_after"] for r in ledger.releases()}
    ledger.close()
    state = run.driver / "state.json"
    state = json.loads(state.read_text()) if state.is_file() else {}
    phases, assigned = state.get("phases", {}), state.get("families", {})
    final = next((t for t in trials if t["trial_id"] == final_id), None)
    out = []
    for group in run.groups():
        row = {"family": group.name, "edge": group.title, "members": list(group.members),
               "share": group.share,
               "final": final is not None and final["edge"] == group.name}  # fmt: skip
        for kind in ("feature", "strategy"):
            mine = [t for t in trials if t["kind"] == kind and t["edge"] == group.name]
            row[kind] = {"submitted": len(mine),
                         "accepted": sum(t["status"] == "accepted" for t in mine),
                         "released": sum(bool(t["released_bool"]) for t in mine),
                         "best": bests.get((kind, group.name))}  # fmt: skip
        mine = {n for n, g in assigned.items() if g == group.name}
        row["sessions"] = {
            phase: sum(1 for n, p in phases.items() if p == phase and n in mine)
            for phase in ("features", "strategies")
        }
        out.append(row)
    return out


def _archetypes_model(run: RunDir, trials, final_id) -> dict:
    """``report.json``'s ``archetypes``: the menu, the fill model, per edge a row for each
    runnable archetype, and the trial counts behind the deflation."""
    brief, menu = run.brief(), run.menu()
    cost = brief.data.cost_model
    final = next((t for t in trials if t["trial_id"] == final_id), None)
    edges = []
    for edge in menu.edges:
        rows = []
        for name in menu.members(edge):
            row = {"archetype": name, "status": menu.verdicts[name].status,
                   "caveats": list(menu.verdicts[name].caveats),
                   "final": final is not None and final["archetype"] == name}  # fmt: skip
            for kind in ("feature", "strategy"):
                mine = [t for t in trials if t["kind"] == kind and t["archetype"] == name]
                vals = [v for t in mine if t["status"] == "accepted"
                        and (v := _val_value(t)) is not None]  # fmt: skip
                row[kind] = {"submitted": len(mine),
                             "accepted": sum(t["status"] == "accepted" for t in mine),
                             "released": sum(bool(t["released_bool"]) for t in mine),
                             "best_val": max(vals) if vals else None}  # fmt: skip
            rows.append(row)
        edges.append({"edge": edge, "title": archetypes.EDGES[edge].title, "archetypes": rows})
    calls = [t for t in trials if t["kind"] in ("feature", "strategy") and t["archetype"]]
    by_archetype = {n: sum(t["archetype"] == n for t in calls) for n in archetypes.REGISTRY
                    if any(t["archetype"] == n for t in calls)}  # fmt: skip
    ledger = run.ledger()
    releases = ledger.releases()
    counts = {"n": ledger.n_trials("strategy"), "groups": len(menu.edges),
              "by_edge": {e: ledger.n_trials("strategy", edge=e) for e in menu.edges},
              "refused_errors": sum(1 for t in trials if t["kind"] == "strategy"
                                    and t["status"] == "error" and t["archetype"]
                                    and menu.line(t["archetype"]) is not None),
              "releases": {e: {k: sum(1 for r in releases if r["group"] == e and r["phase"] == k)
                               for k in ("feature", "strategy")} for e in menu.edges}}  # fmt: skip
    ledger.close()
    return {"menu": menu.to_dict(), "cost_model": None if cost is None else {
        "kind": cost.kind, "fill": cost.fill, "source": cost.source,
        "half_spread_bps": dict(cost.half_spread_bps), "line": cost.line()},
        "edges": edges, "trials": counts, "calls_by_archetype": by_archetype}  # fmt: skip


def _iteration_stats(iterations) -> dict:
    from blinded.driver import _seconds

    done = [r for r in iterations if r["ended_at"]]
    cost = sum(r["cost_usd_est"] or 0.0 for r in done)
    wall = sum(_seconds(r["started_at"], r["ended_at"]) for r in done)
    return {"n": len(iterations), "cost_usd": cost, "wall_seconds": wall,
            "turns": sum(r["turns"] or 0 for r in done)}  # fmt: skip


def _release(row) -> dict:
    return {"trial_id": row["trial_id"], "phase": row["phase"], "best_after": row["best_after"],
            "eta": row["eta"]}  # fmt: skip


def _ladder_points(trials) -> list[dict]:
    """Each feature the Ladder scored, in order: its validation value and the best after it."""
    rows = [t for t in trials if t["kind"] == "feature" and t["released_bool"] is not None]
    return [{"trial_id": t["trial_id"], "value": _val_value(t), "released": t["released_bool"],
             "best_after": t["ladder_best_after"]} for t in rows]  # fmt: skip


def _held_back(trials) -> list[dict]:
    """Accepted strategies the Ladder could not release because validation held fewer than
    ``PERIODS`` of their own holds."""
    out = []
    for t in trials:
        if t["kind"] != "strategy" or t["status"] != "accepted":
            continue
        v = json.loads(t["val_metrics_json"] or "{}")
        if v.get("releasable") is False:
            out.append(
                {
                    "trial_id": t["trial_id"],
                    "name": t["name"],
                    "hold_hours": v.get("hold_hours"),
                    "own_blocks": v.get("own_blocks"),
                }
            )
    return out


def _strategy_points(trials, mode: str) -> list[dict]:
    """Each accepted strategy in order: its validation Sharpe, or in naive mode its claimed one."""
    out = []
    for t in trials:
        if t["kind"] != "strategy" or t["status"] != "accepted":
            continue
        if mode == "naive":
            value = _num(json.loads(t["train_metrics_json"] or "{}").get("sharpe_per_sqrt_hour"))
        else:
            value = _val_value(t)
        out.append({"trial_id": t["trial_id"], "value": value, "released": t["released_bool"]})
    return out


def _train_vs_val(trials, horizon: str) -> list[dict]:
    return [{"trial_id": t["trial_id"], "name": t["name"], "train": _train_ic(t, horizon),
             "val": _val_value(t), "released": t["released_bool"]}
            for t in trials if t["kind"] == "feature" and t["status"] == "accepted"
            and t["released_bool"] is not None]  # fmt: skip


def _rejected(trials) -> list[dict]:
    out = []
    for t in trials:
        if t["kind"] in ("feature", "strategy") and t["status"] != "accepted":
            out.append({"trial_id": t["trial_id"], "kind": t["kind"], "name": t["name"],
                        "status": t["status"], "reason": t["reason_short"],
                        "hypothesis": t["hypothesis"]})  # fmt: skip
    return out


def _deflation(ledger, trial_id, brief) -> dict | None:
    if trial_id is None:
        return None
    # the judge's blocks are the brief's block rounded down to whole grid steps
    step = brief.grid.step
    per_block = max(int(brief.stats.block / step), 1)
    try:
        return deflate.summary(ledger, trial_id, pd.Timedelta("1h") / (per_block * step))
    except (KeyError, ValueError, deflate.NotEnough) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _reproduce(run: RunDir, final_id) -> dict:
    brief = (run.vault / "brief.yaml").read_text()
    code = {}
    if final_id is not None:
        ledger = run.ledger()
        strat = ledger.trial(final_id)
        code[strat["name"]] = Path(strat["code_path"]).read_text()
        for r in ledger.trials("feature"):
            if r["status"] == "accepted" and r["name"] in (strat["name"], *_requires(strat)):
                code[r["name"]] = Path(r["code_path"]).read_text()
        ledger.close()
    return {"brief": brief, "code": code}


def _requires(strat_row) -> list[str]:
    try:
        mod = contract.load(strat_row["code_path"])
        s = next(s for s in mod.strategies if s.name == strat_row["name"])
        return list(s.requires)
    except Exception:
        return []


def _appendix(run: RunDir, trials) -> dict:
    def read(name: str) -> str:
        path = run.workspace / name
        return path.read_text() if path.is_file() else ""

    data_md, notes = read("DATA.md"), read("NOTES.md")
    ledger_rows = [
        {"id": t["trial_id"], "kind": t["kind"], "name": t["name"], "status": t["status"],
         "iteration": t["iteration"], "released": t["released_bool"],
         "note": t["hypothesis"] if t["status"] == "accepted" else t["reason_short"]}
        for t in trials
    ]  # fmt: skip
    return {"data_md": data_md, "notes": notes, "ledger": ledger_rows}


def _verdict(model: dict) -> dict:
    test = model["test"]
    windows = model["windows"]["test"]
    if model["final_id"] is None or test is None:
        return {"line": "No strategy was released on validation, so there is no locked-test "
                "result; the report shows what was tried and why nothing cleared the bar.",
                "net": None}  # fmt: skip
    brief = (test.get("rows") or {}).get("brief", {})
    net = brief.get("net_bps_per_hour")
    sharpe = brief.get("sharpe_per_sqrt_hour")
    ci, few = _ci(model)
    final = model["recompute"].get("final", {})
    name = final.get("name", f"trial {model['final_id']}")
    verb = "made" if (net or 0) >= 0 else "lost"
    ci_text = f" (90% CI {_bps_text(ci[0])} to {_bps_text(ci[1])})" if ci else ""
    if few:
        ci_text = " (too few blocks to say for an interval)"
    span = _window_text(windows)
    what, unit = f"strategy {name}", ""
    if final.get("family"):
        what = f"{final['family']} strategy {name} on {'/'.join(final['legs'])}"
    if final.get("archetype"):
        edge = archetypes.EDGES[final["edge"]].title
        what = f"{final['archetype']} strategy {name} on {'/'.join(final['legs'])} ({edge})"
    if len(final.get("legs") or ()) > 1:
        unit = f" on one unit of notional per leg, gross exposure {brief['gross_exposure']:.2f}"
    if not brief.get("fills"):
        # a rule that waits for a signal the test window never gave: no P&L, no Sharpe
        line = (f"On the locked test window ({span}), {what} "
                "never traded: its entry signal did not fire, so the test says nothing "
                "about whether it works.")  # fmt: skip
        return {"line": line, "net": net, "sharpe": None, "ci": None}
    if final.get("archetype") == "cross_venue_making" and "arb" in final:
        return _arb_verdict(final, span, ci, few)
    if final.get("archetype") == "market_making" and "half_spread_bps" in final:
        # one source under the run's rules: the recompute's row, model, hold rule and path
        rows = (model["recompute"].get("test_rows") or {}).get("brief") or brief
        verb = "made" if (rows.get("net_bps_per_hour") or 0) >= 0 else "lost"
        return _maker_verdict(final, rows, span, what, verb, ci, ci_text)
    sharpe_text = ("with too few blocks to measure a Sharpe" if sharpe is None
                   else f"at a Sharpe per sqrt-hour of {sharpe:+.2f}")  # fmt: skip
    line = (
        f"On the locked test window ({span}), {what} "
        f"{verb} {_bps_text(net)} bps/hour net{ci_text}{unit}, {sharpe_text}; "
    )
    if (net or 0) < 0:
        line += ("not viable as a maker." if final.get("archetype") == "market_making"
                 else "not viable as a taker.")  # fmt: skip
    elif not ci:
        line += "without an interval the test cannot tell it from no edge."
    elif ci[0] <= 0:
        line += "the 90% interval reaches zero, so the test cannot tell it from no edge."
    else:
        line += "viable before costs are reconsidered."
    return {"line": line, "net": net, "sharpe": sharpe, "ci": ci}


RULES_V2 = ("20 Oct 2026: a book crossing fills a resting quote, and every step counts in a "
            "block Sharpe")  # fmt: skip
RULES_V3 = ("20 Oct 2026: a trade before the book row on its stamp, a taker pays to leave its "
            "last position, fees on each trade's notional, a first release needs two standard "
            "errors, no hedge at the mid, and each transfer pays its own fee")  # fmt: skip


def _current_rules_line(model: dict) -> str | None:
    """For a run judged under older rules, what its final's locked test shows under the
    current ones (:data:`blinded.ladder.RULES`), from ``recompute.rules_v<RULES>``; None for a
    run judged under them, or without a final."""
    rules = model.get("judge_rules", 1)
    other = (model.get("recompute") or {}).get(f"rules_v{RULES}")
    if rules >= RULES or not other:
        return None
    head = f"Under the current rules (v{RULES}: {RULES_V3})"
    if other.get("refused"):
        return f"{head} this final is not scored: {other['refused']}."
    if not other.get("fills"):
        return f"{head} this final does not trade on the locked test."
    sharpe = other.get("sharpe_per_sqrt_hour")
    sr = "too few blocks for a Sharpe" if sharpe is None else f"Sharpe {sharpe:+.2f}"
    m = other.get("fill_model")
    if m:
        sr += f", under the {_QUEUE_NAMES.get(m, m)} queue model"
    return (
        f"{head} this final's locked-test net is {_bps_text(other['net_bps_per_hour'])} "
        f"bps/hour ({sr})."
    )


def _rules_line(model: dict) -> str:
    """The judging rules the run was judged under (:data:`blinded.ladder.RULES`), and for a
    run judged under rules 1 what its final's locked test shows under rules 2, from
    ``recompute.rules_v2``."""
    rules = model.get("judge_rules", 1)
    line = f"Judged under rules v{rules}."
    other = (model.get("recompute") or {}).get("rules_v2")
    if rules >= 2 or not other:
        return line
    if not other.get("fills"):
        return f"{line} Under rules v2 ({RULES_V2}) this final does not trade on the locked test."
    sharpe = other.get("sharpe_per_sqrt_hour")
    sr = "too few blocks for a Sharpe" if sharpe is None else f"Sharpe {sharpe:+.2f}"
    m = other.get("fill_model")
    if m:
        sr += f", under the {_QUEUE_NAMES.get(m, m)} queue model"
    return (
        f"{line} Under rules v2 ({RULES_V2}) this final's locked-test net is "
        f"{_bps_text(other['net_bps_per_hour'])} bps/hour ({sr})."
    )


_QUEUE_NAMES = {"RA": "risk-averse", "PR": "pro-rata"}


def _maker_verdict(final, brief, span, what, verb, ci, ci_text) -> dict:
    """The verdict on a market-making final: the scored model, the gross before the maker
    fee, the Sharpe the judge's rule scores, and why it is not viable when it is not: the fee
    and the 10 s giveback each as a multiple of the window's median half spread."""
    net, sharpe = brief.get("net_bps_per_hour"), brief.get("sharpe_per_sqrt_hour")
    m = final.get("fill_model") or "RA"
    queue = _QUEUE_NAMES.get(m, m)
    fee = final.get("maker_fee_bps") or 0.0
    gross = brief.get("gross_bps_per_hour")
    before = (f"{_bps_text(gross)} before the {fee:g} bps maker fee" if fee
              else f"{_bps_text(gross)} before fees")  # fmt: skip
    held = final.get("held") or {}
    if held.get("sharpe") is None:
        sr = "a Sharpe its holds leave too few blocks to measure"
    else:
        sr = f"a Sharpe per sqrt-hour of {held['sharpe']:+.2f}"
        if held["block_steps"] != final.get("brief_block_steps"):
            sr += f" on blocks of {_block_words(held['block_steps'], final['step_s'])}, its holds'"
    amount = _bps_text(net) if verb == "made" else _bps_text(net).lstrip("-")
    line = (f"On the locked test window ({span}), {what} {verb} {amount} bps/hour net "
            f"under the {queue} queue model, the worse of the two{ci_text}, at {sr}; "
            f"{before}; ")  # fmt: skip
    half = final.get("half_spread_bps")
    if (net or 0) < 0:
        why = []
        units = brief.get("units") or 0.0
        if half and fee > 0:
            why.append(f"the fee is {_times(fee / half)} times the half spread of {half:.2f} bps")
        adverse = brief.get("adverse_bps_10s")
        if half and units and adverse is not None and adverse < 0:
            times = _times(-adverse / units / half)
            why.append(f"the fills gave back {times} times the half spread within 10 s")
        line += "not viable as a maker" + (": " + " and ".join(why) if why else "") + "."
    elif not ci:
        line += "without an interval the test cannot tell it from no edge."
    elif ci[0] <= 0:
        line += "the 90% interval reaches zero, so the test cannot tell it from no edge."
    else:
        line += "viable before costs are reconsidered."
    return {"line": line, "net": net, "sharpe": sharpe, "ci": ci}


def _arb_verdict(final: dict, span: str, ci, few: bool = False) -> dict:
    """The verdict on an arbitrage final: the pair, the model
    scored, the net a fill after the hedge fee and in USD at the quote's size with the
    interval, the picked-off share, the drift and the counterparty's prints a day."""
    a, held = final["arb"], final.get("held") or {}
    m = final.get("fill_model") or "RA"
    usd = a["net_usd"]
    verb = "made" if usd >= 0 else "lost"
    scale = final["ref"] * 1e-4 * a["hours"]  # bps of one unit an hour to USD over the window
    ci_text = ""
    if ci:
        ci_text = f" (90% CI {ci[0] * scale:+.2f} to {ci[1] * scale:+.2f} USD)"
    elif few:
        ci_text = " (too few blocks to say for an interval)"
    where = final["quote"] if a["drift_usd_a_day"] >= 0 else final["hedge"]
    drift = abs(a["drift_usd_a_day"]) * a["hours"] / 24
    sr = ("a Sharpe its holds leave too few blocks to measure" if held.get("sharpe") is None
          else f"a Sharpe per sqrt-hour of {held['sharpe']:+.2f}")  # fmt: skip
    prints = final.get("prints_a_day")
    line = (f"On the locked test window ({span}), arb strategy {final['name']} on "
            f"{final['quote']} hedged on {final['hedge']}, under the "
            f"{_QUEUE_NAMES.get(m, m)} queue model, the worse of the two, {verb} "
            f"{a['net_bps']:+.1f} bps a fill net of a {final['hedge_fee_bps']:g} bps hedge fee "
            f"over {a['fills']:,} fills, {usd:+.2f} USD at a {a['size_usd']:.3g} USD size"
            f"{ci_text}, at {sr}; {_share_words(a['picked_off_share'])} picked off, drifting "
            f"{drift:,.0f} USD of coin to {where}")  # fmt: skip
    if prints is not None:
        line += f"; the counterparty is one flow of {prints:,.0f} prints a day"
    line += "."
    moved = final.get("transfers")
    if moved:
        lo = None if not ci else ci[0] * scale
        line += " " + _transfer_words(moved, usd, lo)
    return {"line": line, "net": final.get("net"), "sharpe": held.get("sharpe"), "ci": ci}


def _transfer_words(t: dict, usd: float, lo: float | None) -> str:
    """The rebalancing transfers of an arbitrage's test pass, what they were charged, and the
    fee a transfer would have to cost for the net, and the interval's lower end, to reach 0."""
    n = t["count"]
    if not n:
        return "It made no transfer between the venues on test."
    each = t["fee_usd"] / n
    out = (f"It made {n} transfer{'s' if n != 1 else ''} between the venues on test, "
           f"{t['a_day']:.2g} a day, charged {each:.2f} USD each on average")  # fmt: skip
    even = (usd + t["fee_usd"]) / n
    if lo is None:
        return out + f"; the net reaches zero at {even:.2f} USD a transfer."
    low = (lo + t["fee_usd"]) / n
    return (
        out + f"; the net reaches zero at {even:.2f} USD a transfer, and the interval's "
        f"lower end at {low:.2f}."
    )


def _share_words(x) -> str:
    return "none" if x is None else f"{100 * x:.0f}%"


def _times(x: float) -> str:
    """A multiple in words: one decimal from 1 up, two significant figures below."""
    return f"{x:.1f}" if x >= 1 else f"{x:.2g}"


_bps_text = fees.bps_text


def _window_text(window) -> str:
    """Times of day for a window inside one day, dates for a longer one."""
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    if end - start > pd.Timedelta("1D"):
        return f"{window[0][:10]} to {window[1][:10]}"
    return f"{window[0][11:19]}-{window[1][11:19]}"


def _ci(model: dict) -> tuple[tuple[float, float] | None, bool]:
    """The page's one interval on the final's net bps/hour: the fee verdict's, on the hold
    rule's blocks and the score's draws (:func:`blinded.fees.verdict`), and whether the
    fee verdict says there are too few blocks to say. It replaced an interval of 40 equal
    pieces of the test path drawn independently with a seed of their own, a different
    statistic from the one the verdict's fee words use."""
    fv = (model["recompute"].get("final") or {}).get("fee_verdict") or {}
    if fv.get("interval"):
        lo, hi = fv["interval"]
        return (float(lo), float(hi)), False
    return None, fv.get("kind") is None and fv.get("f0_bps") is not None


def _fence_line(run: RunDir) -> str | None:
    """A sentence when the judge ran judged code without the OS sandbox, from
    ``vault/fence.jsonl`` (:func:`blinded.daemon.record_fence`); None when every start
    ran it inside ``srt``, or the run predates the record."""
    path = run.vault / "fence.jsonl"
    if not path.is_file():
        return None
    starts = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    bare = sum(not s.get("srt") for s in starts)
    if not bare:
        return None
    return (f"The judge ran judged code without the OS sandbox (BLINDED_UNFENCED=1) in "
            f"{bare} of its {len(starts)} starts, behind the audit hook and the static scan "
            "alone, which do not see a library's own reads and writes.")  # fmt: skip


def _naive_dir(run: RunDir) -> Path | None:
    """A naive run beside this protected one (``NAME-naive`` or a ``naive/`` sibling)."""
    if run.mode == "naive":
        return None
    for cand in (run.root.with_name(run.root.name + "-naive"), run.root.parent / "naive"):
        if (cand / "vault" / "manifest.json").is_file():
            return cand
    return None


def _naive_sibling(run: RunDir) -> dict | None:
    found = _naive_dir(run)
    return _naive_table(RunDir.open(found)) if found else None


def _naive_table(naive: RunDir) -> dict:
    ledger = naive.ledger()
    run_row = ledger.run(naive.run_id)
    scored = ledger.test_score(naive.run_id)
    final = run_row["final_trial_id"] if run_row else None
    claimed, name, family = None, None, None
    if final is not None:
        row = ledger.trial(final)
        claimed = json.loads(row["train_metrics_json"] or "{}") if row else None
        name = row["name"] if row else None
        if row and naive.brief().has_archetypes:
            family = f"{row['archetype']} ({archetypes.EDGES[row['edge']].title})"
        elif row and naive.brief().has_families:
            family = row["family"]
    calls = [t for t in ledger.trials() if t["kind"] in ("feature", "strategy")]
    n_calls = len(calls)
    ledger.close()
    protected = json.loads(scored["metrics_json"]) if scored else None
    state = naive.driver / "state.json"
    pick = (json.loads(state.read_text()).get("final") or {}) if state.is_file() else {}
    out = {"run_id": naive.run_id, "final_id": final, "final_name": name, "claimed": claimed,
           "protected_test": protected, "n_calls": n_calls,
           "agent_pick": pick.get("agent_pick"), "override": pick.get("override")}  # fmt: skip
    if naive.brief().has_families:
        out["calls_by_family"] = {f.name: sum(t["family"] == f.name for t in calls)
                                  for f in naive.brief().families}  # fmt: skip
    if naive.brief().has_archetypes:
        out["calls_by_family"] = {g.name: sum(t["edge"] == g.name for t in calls)
                                  for g in naive.groups()}  # fmt: skip
        names = [n for n in archetypes.REGISTRY if any(t["archetype"] == n for t in calls)]
        out["calls_by_archetype"] = {n: sum(t["archetype"] == n for t in calls)
                                     for n in names}  # fmt: skip
    return out | ({"family": family} if family else {})


def _num(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# -- the page ---------------------------------------------------------------


def report(run_dir: str | Path) -> Path:
    """Write ``report.html`` and ``report.json`` to the run directory; return the HTML path."""
    run = RunDir.open(run_dir)
    model = gather(run)
    (run.root / "report.json").write_text(json.dumps(model, indent=1, default=str) + "\n")
    path = run.root / "report.html"
    path.write_text(render(model))
    return path


def render(model: dict) -> str:
    body = "\n".join(
        [
            _head(model),
            _verdict_section(model),
            _tried_section(model),
            _deflation_section(model),
            _signal_section(model),
            _rejected_section(model),
            _naive_section(model),
            _reproduce_section(model),
            _appendix_section(model),
        ]
    )
    return _PAGE.format(title=f"blinded report {esc(model['run_id'])}", body=body, css=_CSS)


def _head(model: dict) -> str:
    w = model["windows"]
    return (
        f"<header><h1>blinded report</h1>"
        f"<p class=sub>run <code>{esc(model['run_id'])}</code>, {esc(model['mode'])} mode, "
        f"blinded {esc(model['blinded'])}"
        + (f", {esc(model['claude_version'])}" if model["claude_version"] else "")
        + f". train {esc(w['train'][0][:19])}, validation {esc(w['val'][0][:19])}, "
        f"test {esc(w['test'][0][:19])} to {esc(w['test'][1][:19])}.</p></header>"
    )


def _verdict_section(model: dict) -> str:
    v = model["verdict"]
    lead = "verdict halted" if model.get("halted") else "verdict"
    out = [f"<section><h2>1. Verdict</h2><p class=verdict>{esc(v['line'])}</p>"]
    if v.get("rules"):
        out.append(f"<p class=plain>{esc(v['rules'])}</p>")
    if v.get("rules_current"):
        out.append(f"<p class=plain>{esc(v['rules_current'])}</p>")
    if v.get("fence"):
        out.append(f"<p class=plain>{esc(v['fence'])}</p>")
    if v.get("cost_model"):
        out.append(f"<p class=plain>{esc(v['cost_model'])}</p>")
    if model["probes"]:
        out.append(
            f"<p class=alert>The run halted on {len(model['probes'])} sandbox probe(s). "
            "A probe means the sandbox needs attention before any number can be trusted; "
            "the test segment was not scored.</p>"
        )
    final = (model.get("recompute") or {}).get("final") or {}
    if final.get("fee_verdict"):
        out.append(f"<p class=plain>{esc(fees.words(final['fee_verdict']))}</p>")
    if final.get("fee_verdict") or '"fee_verdict"' in json.dumps(
        model.get("recompute") or {}, default=str
    ):
        out.append(f"<p class=note>{esc(fees.CAVEAT)}</p>")
    test = model["test"]
    if test and test.get("rows"):
        out.append(_test_table(test["rows"]))
        curves = model["recompute"].get("curves") or {}
        if curves:
            hours = (test["rows"].get("brief") or {}).get("hours") or 1.0
            out.append(_chart(plots.locked_test_pnl(curves, hours)))
    else:
        out.append("<p>No strategy was scored on test.</p>")
    out.append("</section>")
    _ = lead
    return "\n".join(out)


def _test_table(rows: dict) -> str:
    cols = [("net_bps_per_hour", "net bps/h"), ("gross_bps_per_hour", "gross bps/h"),
            ("sharpe_per_sqrt_hour", "Sharpe/sqrt h"), ("net_bps_per_fill", "net bps/fill"),
            ("fills", "fills"), ("hit_rate", "hit rate")]  # fmt: skip
    order = ["brief", "latency 0", "fees 0", "latency 0, fees 0", "maker proxy", "maker-taker"]
    maker = "RA" in rows
    if maker and "hedge 1 s" in rows:
        order = ["brief", *sim_maker.MODELS, "RA uncapped", "latency 0", "hedge 1 s",
                 *(name for name, _ in sim_maker.HEDGE_FEE_ROWS)]  # fmt: skip
        # an arbitrage's bps are of one unit of the coin at P_ref, both legs together
        unit = " of one unit at P_ref"
        cols = [(k, f"{label}{unit}" if "bps" in label else label) for k, label in cols]
        cols.insert(4, ("units", "units"))
    elif maker:
        order = ["brief", *sim_maker.MODELS, "latency 0",
                 *(name for name, _ in sim_maker.FEE_ROWS)]  # fmt: skip
        cols.insert(4, ("units", "units"))
    head = "<tr><th>variant</th>" + "".join(f"<th>{esc(c[1])}</th>" for c in cols) + "</tr>"
    body = []
    for name in order:
        if name not in rows:
            continue
        r = rows[name]
        note = " (unrealistic)" if name in DIAGNOSTIC else ""
        note = " (upper bound)" if name == "maker proxy" else note
        note = " (diagnostic)" if name == "maker-taker" else note
        if maker:
            note = {"latency 0": " (unrealistic)", "T": " (bracket)", "F": " (bracket)"}.get(
                name, "")  # fmt: skip
        cells = "".join(f"<td>{_fmt(r.get(c[0]))}</td>" for c in cols)
        klass = " class=diag" if note else ""
        body.append(f"<tr{klass}><td>{esc(name)}{note}</td>{cells}</tr>")
    return f"<table class=metrics>{head}{''.join(body)}</table>"


def _tried_section(model: dict) -> str:
    c, it = model["counts"], model["iterations"]
    out = [
        "<section><h2>2. What was tried</h2>",
        f"<p>{c['distinct']['feature']} distinct features "
        f"({c['feature']['accepted']} accepted, {c['feature']['rejected']} rejected) and "
        f"{c['distinct']['strategy']} distinct strategies "
        f"({c['strategy']['accepted']} accepted, {c['strategy']['rejected']} rejected) over "
        f"{it['n']} iterations, {_hm(it['wall_seconds'])} and "
        f"an estimated ${it['cost_usd']:.2f}.</p>",
    ]
    if model.get("archetypes"):
        out.append(f"<p>{esc(model['archetypes']['menu']['headline'])}</p>")
    if model.get("families"):
        out.append(_families_table(model["families"], model["mode"]))
    if model.get("archetypes"):
        out += _archetype_tables(model["archetypes"], model["mode"])
    if model["ladder_points"]:
        out.append("<p>Each dot is a feature's validation rank IC, which the agent never saw; "
                   "the step is the released best, which is all it was told.</p>")  # fmt: skip
        out.append(_chart(plots.ladder(model["ladder_points"])))
    if any(p["train"] is not None and p["val"] is not None for p in model["train_vs_val"]):
        out.append("<p>A feature below the dotted line did worse on validation than on train. "
                   "Points drifting further below it as the run goes on is the overfitting "
                   "the Ladder is there to limit.</p>")  # fmt: skip
        out.append(_chart(plots.train_vs_val(model["train_vs_val"]), narrow=True))
    if not model["ladder_points"] and model["mode"] == "protected":
        out.append("<p class=note>The Ladder scored nothing.</p>")
    points = model.get("strategy_points") or []
    if any(p["value"] is not None for p in points):
        if model["mode"] == "naive":
            out.append("<p>Each strategy's Sharpe as the loop saw it, on the window it asked "
                       "for. A loop without a judge ships the best so far, the top of many "
                       "noisy tries.</p>")  # fmt: skip
            ylabel, title = "claimed Sharpe per sqrt-hour", "Strategies as the naive loop saw them"
        else:
            out.append("<p>Each strategy's validation Sharpe, which the agent never saw. A loop "
                       "without a judge would ship the best so far; the Ladder releases one "
                       "only when it clears the best by more than noise.</p>")  # fmt: skip
            ylabel, title = "validation Sharpe per sqrt-hour", "Strategies on validation"
        out.append(_chart(plots.attempts(points, ylabel, title)))
    if model.get("held_back"):
        out.append(_held_back_text(model["held_back"]))
    out.append("</section>")
    return "\n".join(out)


def _held_back_text(held: list[dict]) -> str:
    from blinded.archetypes import PERIODS

    each = "; ".join(
        f"{esc(h['name'])} (trial {h['trial_id']}), holds of "
        f"{words(pd.Timedelta(hours=h['hold_hours'] or 0).round('1min'))}, "
        f"{h['own_blocks']} in validation"
        for h in held
    )
    who = "One strategy was" if len(held) == 1 else f"{len(held)} strategies were"
    return (
        f"<p>{who} not released for holding positions too long for validation to judge. "
        "One hold is one bet however many blocks it spans, so a strategy's noise is "
        "measured in blocks as long as its time-weighted mean hold, and validation must "
        f"hold {PERIODS} of them: {each}.</p>"
    )


def _archetype_tables(arch: dict, mode: str) -> list[str]:
    """A table per edge, one row per runnable archetype, and the folded list of what the data
    cannot test."""
    best = "best on validation" if mode == "protected" else "best on validation (not scored)"
    head = ["archetype", "features: submitted / accepted / released", f"feature {best}",
            "strategies: submitted / accepted / released", f"strategy {best}", ""]  # fmt: skip
    out = ["<p>Per archetype, what was submitted, accepted and released, and its best "
           "validation score of each kind: rank IC for a feature, Sharpe per square-root hour "
           "for a strategy. The agent saw only the edge's released best; these it never "
           "saw.</p>"]  # fmt: skip
    for e in arch["edges"]:
        rows = []
        for r in e["archetypes"]:
            cells = [f"<code>{esc(r['archetype'])}</code>"]
            for kind in ("feature", "strategy"):
                k = r[kind]
                cells += [f"{k['submitted']} / {k['accepted']} / {k['released']}",
                          _fmt(k["best_val"])]  # fmt: skip
            cells.append("holds the final strategy" if r["final"] else "")
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        out.append(f"<h3>{esc(e['title'][0].upper() + e['title'][1:])}</h3>"
                   "<table class=metrics><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head)
                   + "</tr>" + "".join(rows) + "</table>")  # fmt: skip
    lines = [a["line"] for a in arch["menu"]["archetypes"] if a["line"]]
    items = "".join(f"<li>{esc(line)}</li>" for line in lines)
    out.append(f"<details><summary>What this data cannot test ({len(lines)})</summary>"
               f"<ul>{items}</ul></details>")  # fmt: skip
    return out


def _families_table(families: list[dict], mode: str) -> str:
    edges = any("edge" in f for f in families)
    unit = "edge" if edges else "family"
    head = ["edge", "members"] if edges else ["family"]
    head += ["share", "features: submitted / accepted / released", "best feature",
             "strategies: submitted / accepted / released", "best strategy",
             "sessions (features, strategies)", ""]  # fmt: skip
    rows = []
    for f in families:
        cells = [esc(f["edge"]), esc(", ".join(f["members"]))] if edges else [esc(f["family"])]
        cells.append(f"{f['share']:g}")
        for kind in ("feature", "strategy"):
            k = f[kind]
            cells += [f"{k['submitted']} / {k['accepted']} / {k['released']}", _fmt(k["best"])]
        cells += [f"{f['sessions']['features']}, {f['sessions']['strategies']}",
                  "holds the final strategy" if f["final"] else ""]  # fmt: skip
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    table = ("<table class=metrics><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head)
             + "</tr>" + "".join(rows) + "</table>")  # fmt: skip
    intro = (f"<p>Each session was for one {unit}, and each {unit} had its own Ladder for "
             "each kind; the best is the released best, which is all the agent was told.</p>"
             if mode == "protected" else f"<p>Each session was for one {unit}.</p>")  # fmt: skip
    note = ""
    if any(f["family"] == "pairs" for f in families):
        note = ("<p class=plain>A pairs feature is scored against the spread return at equal "
                "notional, with no hedge ratio in it; for a pair whose hedge ratio is far "
                "from 1 that understates the IC of a hedged spread.</p>")  # fmt: skip
    return intro + table + note


def _deflation_section(model: dict) -> str:
    d = model["deflation"]
    out = ["<section><h2>3. Deflation</h2>"]
    if model.get("archetypes") and model["mode"] == "protected":
        out.append(_trial_counts(model["archetypes"]))
    if model["mode"] == "naive":
        out.append("<p>A naive run scores whatever window the agent asks for, so it has no "
                   "validation blocks to deflate with. The protected run beside it does the "
                   "deflation.</p></section>")  # fmt: skip
        return "\n".join(out)
    if not d:
        out.append("<p>No strategy was released, so there is nothing to deflate.</p></section>")
        return "\n".join(out)
    if "error" in d:
        out.append(f"<p>Not enough trials or blocks to deflate: {esc(d['error'])}.</p></section>")
        return "\n".join(out)
    sr = d["sharpe_per_sqrt_hour"]
    sr0 = d["sr0_per_sqrt_hour"]
    unit = "edges" if model.get("archetypes") else "families"
    every = f", of {len(model['families'])} {unit}," if model.get("families") else ""
    out.append(
        f"<p>The chosen strategy's validation Sharpe is {sr:+.2f} per sqrt-hour. After "
        f"{d['n_trials']} trials{every} the expected maximum under the null is {sr0:+.2f}; "
        f"the deflated Sharpe probability is {d['dsr']:.3f}.</p>"
    )
    if d["dsr"] < 0.95:
        out.append("<p class=plain>After this many trials, a validation Sharpe of "
                   f"{sr:+.2f} is within what noise would produce.</p>")  # fmt: skip
    if d.get("pbo") is not None:
        out.append(f"<p>Probability of backtest overfitting (CSCV): {d['pbo']:.2f}.</p>")
    elif d.get("pbo_reason"):
        out.append(f"<p>PBO: {esc(d['pbo_reason'])}.</p>")
    out.append("</section>")
    return "\n".join(out)


def _trial_counts(arch: dict) -> str:
    """N, the K edges and what each edge tried and released, beside the deflation."""
    c = arch["trials"]
    title = {e["edge"]: e["title"] for e in arch["edges"]}
    by = ", ".join(f"{title[e]} {n}" for e, n in c["by_edge"].items())
    rel = "; ".join(f"{title[e]} {r['feature']} feature{'s' if r['feature'] != 1 else ''} and "
                    f"{r['strategy']} strateg{'ies' if r['strategy'] != 1 else 'y'}"
                    for e, r in c["releases"].items())  # fmt: skip
    return (f"<p>N = {c['n']} distinct strategies over the whole run, error rows included "
            f"({c['refused_errors']} of them refused archetypes), across K = {c['groups']} "
            f"edges: {by}. Each edge has its own Ladder, so the run leaks up to K times the "
            f"bits of one. Released: {rel}.</p>")  # fmt: skip


def _signal_section(model: dict) -> str:
    diag = model["recompute"].get("diagnostics") or {}
    out = ["<section><h2>4. The signal</h2>"]
    if "archetypes" in diag:
        return "\n".join(out + _archetype_signal(diag, model) + ["</section>"])
    feat, strat = diag.get("feature"), diag.get("strategy")
    families = "lead_lag" in diag or "pairs" in diag
    if not feat and not strat and not families:
        out.append("<p>No feature or strategy was accepted, so there is no signal to show.</p>")
        out.append("</section>")
        return "\n".join(out)
    names = [plots.SEGMENT_NAME[s] for s in diag["segments"]]
    segs = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
    out.append(f"<p>Shown on {segs}. Test appears only when the final strategy was scored "
               "there; it is never opened for diagnostics alone.</p>")  # fmt: skip
    if families:
        out.append("<h3>Directional</h3>")
        if not feat and not strat:
            out.append("<p>No directional feature or strategy was accepted.</p>")
    out += _single_leg_charts(feat, strat)
    if "lead_lag" in diag:
        out += _lead_lag_charts(diag["lead_lag"])
    for item in diag.get("pairs") or []:
        out += _pair_charts(item)
    maker = (model["test"].get("rows") or {}).get("maker proxy") if model["test"] else None
    if maker:
        out.append(f"<p>Maker proxy, an upper bound (front of the queue, no adverse selection): "
                   f"{_fmt(maker.get('net_bps_per_hour'))} net bps/hour.</p>")  # fmt: skip
    out.append("</section>")
    return "\n".join(out)


def _archetype_signal(diag: dict, model: dict) -> list[str]:
    """Section 4 of a run with archetypes: by edge, then by archetype, the mechanism's charts
    and the "How it dies" panel."""
    names = [plots.SEGMENT_NAME[s] for s in diag["segments"]]
    segs = ", ".join(names[:-1]) + f" and {names[-1]}"
    out = [f"<p>Shown on {segs}, by edge and then by archetype. Each archetype's pick is the "
           "rule of a families run kept to its own trials: the final strategy's, else the last "
           "released, else the best on validation. Test appears only when the final strategy "
           "was scored there.</p>"]  # fmt: skip
    for edge, members in diag["archetypes"].items():
        e = archetypes.EDGES[edge]
        title = e.title[0].upper() + e.title[1:]
        out.append(f"<h3 style='color:var(--fg);font-size:17px'>{esc(title)}: "
                   f"{esc(e.tag.lower())}</h3><p>{esc(e.text)}</p>")  # fmt: skip
        for name, entry in members.items():
            out += _archetype_block(name, entry, diag["segments"])
    maker = (model["test"].get("rows") or {}).get("maker proxy") if model["test"] else None
    if maker:
        out.append(f"<p>Maker proxy, an upper bound (front of the queue, no adverse selection): "
                   f"{_fmt(maker.get('net_bps_per_hour'))} net bps/hour.</p>")  # fmt: skip
    return out


def _archetype_block(name: str, entry: dict, segs: list[str]) -> list[str]:
    arch = archetypes.REGISTRY[name]
    out = [f"<h3 style='color:var(--fg)'>{esc(name)}: {esc(arch.title)}</h3>"]
    if entry.get("note"):
        return out + [f"<p class=note>{esc(entry['note'][0].upper() + entry['note'][1:])}.</p>"]
    feat, strat = entry.get("feature"), entry.get("strategy")
    if arch.template == "arb":
        if not feat and not entry.get("arb"):
            out.append("<p>No feature or strategy of this archetype was accepted.</p>")
        out += _single_leg_charts(feat, None)
        if entry.get("arb"):
            out += _arb_tables(entry["arb"], segs)
    elif arch.template == "maker":
        if not feat and not entry.get("maker"):
            out.append("<p>No feature or strategy of this archetype was accepted.</p>")
        out += _single_leg_charts(feat, None)
        if entry.get("maker"):
            out += _bracket_tables(entry["maker"], segs)
    elif arch.mechanism == "basket":
        out += _basket_charts(entry)
    elif arch.mechanism == "pairs":
        for item in entry.get("pairs") or []:
            out += _pair_charts(item)
        if entry.get("curves"):
            out.append("<p>Across the two venues, the cross-correlation of each venue's last "
                       "step with the other's step ending a lag later, both ways round: a "
                       "venue that copies the other with a delay peaks at that delay. On the "
                       "receive clock a lead can be market lead or feed path, and one no "
                       "longer than the second venue's feed cadence (D11 below) cannot be "
                       "told from its schedule.</p>")  # fmt: skip
        for c in entry.get("curves") or []:
            out += [_chart(plots.lead_lag(c)), _lead_lag_table(c)]
    else:
        if not feat and not strat:
            out.append("<p>No feature or strategy of this archetype was accepted.</p>")
        out += _single_leg_charts(feat, strat)
        for c in entry.get("curves") or []:
            out += [_chart(plots.lead_lag(c)), _lead_lag_table(c)]
    out += _dies_panel(arch, entry.get("dies") or {}, segs)
    return out


def _bracket_tables(maker: dict, segs: list[str]) -> list[str]:
    """The four fill models by segment with the score's Sharpe and its block, the scored model
    marked; the fee rows; and the three charts."""
    from blinded.archetypes import PERIODS

    n = f"<code>{esc(maker['name'])}</code> (trial {maker['trial_id']})"
    block = words(pd.Timedelta(seconds=maker["block_s"]))
    out = [f"<h3>The strategy: {n}, {esc(maker['why'])}</h3>",
           "<p>Four fill models, each its own closed loop on the recorded book and trades: T "
           "fills only when a trade goes through the quote, RA joins the back of the "
           "displayed size and takes every cancel as behind it, PR takes cancels from ahead "
           "of it in proportion, and F is always at the front. Every number is in bps of one "
           "unit at the first mid of the segment. The judge scores the lower of RA's and PR's "
           "Sharpe in the score's column, each measured on blocks at least as long as that "
           "model's time-weighted mean hold, since one hold is one bet however many blocks "
           f"it spans; the column before it is the same P&amp;L on the brief's {block} "
           "blocks.</p>"]  # fmt: skip
    cols = [
        *(t for _, t in BRACKET[:-1]),
        f"Sharpe/sqrt h, {block} blocks",
        "Sharpe/sqrt h, the score's",
        "the score's blocks",
    ]
    head = "<tr><th>model</th>" + "".join(f"<th>{esc(t)}</th>" for t in cols) + "</tr>"
    step = maker["step_s"]
    lines = []
    for seg in segs:
        part = (maker.get("segments") or {}).get(seg)
        if not part:
            continue
        body = []
        for m in sim_maker.MODELS:
            row = part["models"][m]
            held = row.get("held") or {}
            mark = " (scored)" if m == part["scored"] else (" (bracket)" if m in "TF" else "")
            cells = "".join(f"<td>{_fmt(row.get(k))}</td>" for k, _ in BRACKET)
            if m in sim_maker.QUEUE_MODELS:
                value = _fmt(held.get("sharpe")) or "none"
                value = f"<b>{value}</b>" if m == part["scored"] else value
                span = _block_words(held["block_steps"], step)
                cells += f"<td>{value}</td><td>{held['blocks']} of {esc(span)}</td>"
            else:
                cells += "<td></td><td></td>"
            klass = " class=diag" if m in "TF" else ""
            body.append(f"<tr{klass}><td>{m}{mark}</td>{cells}</tr>")
        out.append(f"<p class=plain>{plots.SEGMENT_NAME[seg]}</p><table class=metrics>{head}"
                   f"{''.join(body)}</table>")  # fmt: skip
        lines.append(_scored_words(seg, part, step, block, PERIODS))
    out.append("<p>" + " ".join(lines) + " T and F bracket the fill count, not the P&amp;L: "
               "reality's fill count lies between T and F, and its P&amp;L is not bounded by "
               "any row.</p>")  # fmt: skip
    names = ["brief", "latency 0", *(name for name, _ in sim_maker.FEE_ROWS)]
    shown = [s for s in segs if s in maker["segments"]]
    head = "<tr><th>row</th>" + "".join(f"<th>{plots.SEGMENT_NAME[s]}</th>" for s in shown)
    body = "".join(
        f"<tr><td>{esc(name)}</td>" + "".join(
            f"<td>{_fmt(maker['segments'][s]['rows'][name].get('net_bps_per_hour'))}</td>"
            for s in shown) + "</tr>" for name in names)  # fmt: skip
    out.append("<p class=plain>Net bps/hour on the scored model's fills at other maker fees "
               "(fees change no fill; <code>fees 0</code> waives the flatten's taker fee too), "
               "and with every latency 0 (unrealistic).</p>"
               f"<table class=metrics>{head}</tr>{body}</table>")  # fmt: skip
    out += _maker_charts(maker, shown)
    return out


def _arb_tables(arb: dict, segs: list[str]) -> list[str]:
    """An arbitrage strategy's four fill models by segment in the probe's terms, with the
    score's Sharpe and its blocks, the scored model marked; the sim rows; the first fills
    decomposed; and the two charts."""
    from blinded.archetypes import PERIODS

    n = f"<code>{esc(arb['name'])}</code> (trial {arb['trial_id']})"
    block = words(pd.Timedelta(seconds=arb["block_s"]))
    out = [
        f"<h3>The strategy: {n}, {esc(arb['why'])}</h3>",
        f"<p>Quoted on {esc(arb['quote'])} and every fill hedged on {esc(arb['hedge'])} as "
        "a taker at its depth as-of the hedge's arrival. Four fill models, each its own "
        "closed loop, with a trade through the quote filling no more than its print. Per "
        "fill, in bps of the fill at the hedge venue's mid: captured, the maker leg against "
        "that mid; delay, its move until the hedge arrived; slippage, the hedge against the "
        "mid then; fees, both legs'; net, their sum. Picked off is a fill whose price that "
        "mid had already passed; drift is the USD of coin a day moved onto the quote venue. "
        "The judge scores the lower of RA's and PR's Sharpe on net per block, both legs, "
        f"each on blocks at least as long as its holds (the brief's {block}).</p>",
    ]
    cols = [
        *(t for _, t in ARB_BRACKET),
        "net bps/h",
        "Sharpe/sqrt h, the score's",
        "the score's blocks",
    ]
    head = "<tr><th>model</th>" + "".join(f"<th>{esc(t)}</th>" for t in cols) + "</tr>"
    step = arb["step_s"]
    lines = []
    for seg in segs:
        part = (arb.get("segments") or {}).get(seg)
        if not part:
            continue
        body = []
        for m in sim_maker.MODELS:
            row = part["models"][m]
            held = row.get("held") or {}
            mark = " (scored)" if m == part["scored"] else (" (bracket)" if m in "TF" else "")
            cells = "".join(f"<td>{_fmt(row.get(k))}</td>" for k, _ in ARB_BRACKET)
            cells += f"<td>{_fmt(row.get('net_bps_per_hour'))}</td>"
            if m in sim_maker.QUEUE_MODELS:
                value = _fmt(held.get("sharpe")) or "none"
                value = f"<b>{value}</b>" if m == part["scored"] else value
                span = _block_words(held["block_steps"], step)
                cells += f"<td>{value}</td><td>{held['blocks']} of {esc(span)}</td>"
            else:
                cells += "<td></td><td></td>"
            klass = " class=diag" if m in "TF" else ""
            body.append(f"<tr{klass}><td>{m}{mark}</td>{cells}</tr>")
        out.append(f"<p class=plain>{plots.SEGMENT_NAME[seg]}</p><table class=metrics>{head}"
                   f"{''.join(body)}</table>")  # fmt: skip
        lines.append(_scored_words(seg, part, step, block, PERIODS))
    out.append(
        "<p>" + " ".join(lines) + " T and F bracket the fill count; inside a spread "
        "this wide a quote is alone at its price, so the four rarely differ.</p>"
    )
    shown = [s for s in segs if s in arb["segments"]]
    names = list(next(iter(arb["segments"].values()))["rows"]) if shown else []
    head = "<tr><th>row</th>" + "".join(f"<th>{plots.SEGMENT_NAME[s]}</th>" for s in shown)
    body = "".join(
        f"<tr><td>{esc(name)}</td>" + "".join(
            f"<td>{_fmt(arb['segments'][s]['rows'][name].get('net_bps_per_hour'))}</td>"
            for s in shown) + "</tr>" for name in names)  # fmt: skip
    out.append("<p class=plain>Net bps/hour of one unit on the scored model's fills: with a "
               "trade through filling the whole order (<code>RA uncapped</code>, the capacity "
               "row), every latency 0 (unrealistic), each hedge reaching the hedge venue 1 s "
               "after its fill, and at hedge fees of 10, 7.5, 5 and 0 bps with the quote "
               "venue's maker fee 0 (fees change no fill).</p>"
               f"<table class=metrics>{head}</tr>{body}</table>")  # fmt: skip
    if shown:
        last = shown[-1]
        fills = arb["segments"][last]["fills"]
        if fills:
            cols = (
                "s into the window",
                "side",
                "price",
                "units",
                "hedge vwap",
                "captured",
                "delay",
                "slippage",
                "fees",
                "net",
                "picked off",
            )
            head = "<tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>"
            body = "".join(_fill_row(f) for f in fills)
            out.append(f"<p class=plain>The first {len(fills)} fills of the scored model on "
                       f"{plots.SEGMENT_NAME[last]}, in bps of each fill.</p>"
                       f"<table class=metrics>{head}{body}</table>")  # fmt: skip
    ex = arb.get("excerpt")
    if ex and ex.get("seconds"):
        out += [
            "<p>The half hour of the last segment with the most fills of the scored "
            "model: both venues' touches, the quotes resting on the quote venue, each fill "
            "and the hedge it sent.</p>",
            _chart(plots.arb_excerpt(ex, arb["name"], arb["quote"], arb["hedge"])),
        ]
    pnl = {s: arb["segments"][s]["pnl"] for s in shown}
    scored = {s: arb["segments"][s]["scored"] for s in shown}
    if pnl:
        out.append(_chart(plots.arb_pnl(pnl, arb["name"], scored)))
    return out


def _fill_row(f: dict) -> str:
    """One fill of the per-fill table: its stamp, side, price, units, the hedge's price and
    the decomposition in bps of the fill."""
    vwap = "" if f["vwap"] is None else f"{f['vwap']:.6g}"
    side = "buy" if f["side"] > 0 else "sell"
    cells = "".join(f"<td>{_fmt(f[k])}</td>" for k in ARB_FILL_KEYS)
    return (f"<tr><td>{_fmt(f['seconds'])}</td><td>{side}</td><td>{f['price']:.6g}</td>"
            f"<td>{_fmt(f['units'])}</td><td>{vwap}</td>{cells}"
            f"<td>{'yes' if f['picked_off'] else ''}</td></tr>")  # fmt: skip


def _scored_words(seg: str, part: dict, step_s: float, block: str, periods: int) -> str:
    """Which model scored on a segment and on what, in words: the lower of RA's and PR's Sharpe
    on its own blocks; whether the brief's blocks would have picked the other; and on
    validation, whether the hold rule's twenty blocks refused a score."""
    name = plots.SEGMENT_NAME[seg]
    m = part["scored"]
    other = "PR" if m == "RA" else "RA"
    held = {q: part["models"][q]["held"] for q in sim_maker.QUEUE_MODELS}
    mine, theirs = held[m]["sharpe"], held[other]["sharpe"]
    if mine is None:
        text = (f"On {name} the rule took {m}: its Sharpe measures nothing (fewer than two of "
                "its blocks, or a P&amp;L that never moved), and a pass with no number is the "
                "worse.")  # fmt: skip
    else:
        text = (f"On {name} the rule took {m}, the lower of {m}'s {_fmt(mine)} and "
                f"{other}'s {_fmt(theirs) or 'none'}.")  # fmt: skip
    plain = {q: part["models"][q].get("sharpe_per_sqrt_hour") for q in sim_maker.QUEUE_MODELS}
    if None not in plain.values() and plain[other] < plain[m]:
        text += (f" On the brief's {block} blocks {other} is the lower ({_fmt(plain[other])} "
                 f"against {_fmt(plain[m])}); the score does not use those, because {m}'s "
                 "holds outlast them.")  # fmt: skip
    if seg == "val" and not held[m]["releasable"]:
        span = _block_words(held[m]["hold_steps"], step_s)
        text += (f" {m} held its inventory {esc(span)} on average, so validation holds "
                 f"{held[m]['blocks']} of its blocks, fewer than {periods}: the Ladder was "
                 "given no score, and the strategy could not be released.")  # fmt: skip
    return text


def _block_words(steps: float, step_s: float) -> str:
    """A span of grid steps in words, to the second from a minute up, else to the
    millisecond."""
    span = pd.Timedelta(seconds=steps * step_s)
    return words(span.round("1s" if span >= pd.Timedelta("1min") else "1ms"))


def _maker_charts(maker: dict, segs: list[str]) -> list[str]:
    """The excerpt, the fill markouts by model and where the P&L goes."""
    parts = maker["segments"]
    ex = maker.get("excerpt")
    out = []
    if ex and ex.get("seconds"):
        out += ["<p>The half hour of the last segment with the most fills of the scored "
                "model: the touch, the quotes resting in it, each fill, and the inventory "
                "below.</p>", _chart(plots.quotes(ex, maker["name"]))]  # fmt: skip
    fans = {s: parts[s]["fan"] for s in segs}
    scored = {s: parts[s]["scored"] for s in segs}
    out += ["<p>After each fill: the mid's move in the fill's direction, weighted by units, "
            "for each model. A maker's fill sits below the mid at 0 by what it captured, and "
            "pays the fee; the line has to stay above the break-even for the fill to pay. "
            "Lines that fall after 0 are adverse selection.</p>",
            _chart(plots.fill_markouts(fans, maker["offsets"], maker["maker_fee_bps"],
                                       scored))]  # fmt: skip
    pnl = {s: parts[s]["pnl"] for s in segs}
    out.append(_chart(plots.maker_pnl(pnl, maker["name"], scored)))
    return out


# the rows of each check in the "How it dies" panel: label, key, and how to print the value
_DIES_ROWS = {
    "costs": [("gross P&L, bps", "gross_bps"), ("units traded", "traded"),
              ("break-even fee, bps a unit", "breakeven_fee_bps"),
              ("net bps/h at the brief's latency and fee", "rows.brief.net_bps_per_hour"),
              ("net bps/h, latency 0 (unrealistic)", "rows.latency 0.net_bps_per_hour"),
              ("net bps/h, fees 0 (unrealistic)", "rows.fees 0.net_bps_per_hour"),
              ("net bps/h, maker proxy (upper bound)", "rows.maker proxy.net_bps_per_hour")],
    "race": [("peak rank IC, returns from the decision", "peak_rank_ic"),
             ("at horizon", "peak_horizon"), ("rank IC there, returns from the fill",
                                              "rank_ic_left"),
             ("share left to a taker", "ic_left_share"), ("entry cost, bps", "entry_bps"),
             ("median half spread, bps", "half_spread_bps"),
             ("moved in flight, bps", "in_flight_bps")],
    "worst_window": [("worst window, bps", "worst_bps"), ("best window, bps", "best_bps"),
                     ("total, bps", "total_bps"),
                     ("best window's share of the total", "concentration"),
                     ("losing windows", "losing_windows"), ("windows", "windows"),
                     ("longest under water, hours", "underwater_hours"),
                     ("max drawdown, bps", "max_drawdown_bps")],
    "half_life": [("signal autocorrelation at one step", "autocorr"),
                  ("signal half-life", "signal_half_life_s:span"),
                  ("spread's median half-life in a window", "spread_median_half_life_s:span"),
                  ("a random walk's, same window", "random_walk_half_life_s:span"),
                  ("windows where the spread did not revert", "no_reversion_share")],
    "mae": [("round trips", "round_trips"), ("median adverse excursion, bps", "median_bps"),
            ("worst adverse excursion, bps", "worst_bps"),
            ("median close, bps", "median_close_bps"),
            ("under water past their cost, closed positive", "recovered_share")],
    "breadth": [("legs", "legs"), ("legs with a positive net", "positive_share"),
                ("the best leg", "top_leg"), ("its share of the total", "top_share"),
                ("mean legs held", "mean_legs_held"), ("rebalances", "rebalances"),
                ("units traded a rebalance", "turnover_per_rebalance")],
    "venue_feed": [("instrument", "instrument"), ("median gap, ms", "median_gap_ms"),
                   ("largest 10 ms bin starts at, ms", "bin_max_ms"),
                   ("its share of gaps", "share_max"), ("under 50 ms", "ok")],
    "funding": [("funding received, bps", "funding_bps"),
                ("while short the perpetual", "short_bps"),
                ("while long the perpetual", "long_bps"),
                ("basis change (mid P&L of both legs), bps", "mid_bps"),
                ("settlements in the window", "settlements"),
                ("mean rate a settlement, bps", "mean_rate_bps")],
    "whipsaw": [("round trips", "round_trips"),
                ("closed within the band's low end", "short_share"),
                ("hit rate", "hit_rate"), ("mean win over mean loss", "win_loss"),
                ("share of the net from the best tenth", "best_tenth_share")],
}  # fmt: skip


# a market maker's rows, where they differ from a position strategy's
_MAKER_ROWS = {
    "costs": [("gross P&L, bps", "gross_bps"), ("units filled while resting", "traded"),
              ("break-even maker fee, bps a unit", "breakeven_fee_bps"),
              ("net bps/h, the scored model", "rows.brief.net_bps_per_hour"),
              ("net bps/h, latency 0 (unrealistic)", "rows.latency 0.net_bps_per_hour"),
              ("net bps/h, no fees", "rows.fees 0.net_bps_per_hour"),
              ("net bps/h, a 0.5 bps rebate", "rows.rebate 0.5.net_bps_per_hour"),
              ("net bps/h, T (bracket)", "rows.T.net_bps_per_hour"),
              ("net bps/h, F (bracket)", "rows.F.net_bps_per_hour")],
    "capacity": [("median order over the size displayed at arrival", "median_touch_share"),
                 ("fills whose order was over the size displayed", "touch_share_above_one"),
                 ("median order, units", "median_order"),
                 ("median aggressor order, units", "median_aggressor"),
                 ("order over aggressor", "order_over_aggressor"),
                 ("partial fills", "partial_share"),
                 ("bids filled over sells traded", "participation_bid"),
                 ("asks filled over buys traded", "participation_ask"),
                 ("median size displayed at the touch", "median_displayed")],
    "toxicity": [("units filled while resting", "units"),
                 ("captured at the fill, bps a unit", "captured_bps_per_unit"),
                 ("mid move 1 s after, bps a unit", "adverse_1s_bps_per_unit"),
                 ("mid move 10 s after, bps a unit", "adverse_10s_bps_per_unit"),
                 ("mid move 60 s after, bps a unit", "adverse_60s_bps_per_unit"),
                 ("inventory past 10 s and the flatten, bps a unit", "inventory_bps_per_unit"),
                 ("10 s move over capture", "adverse_over_captured"),
                 ("fills by a trade through the price", "through_share"),
                 ("fills by the book crossing the price, no trade", "crossed_share"),
                 ("fills alone inside the spread", "inside_share"),
                 ("fills by the queue traded out", "queue_share")],
    "queue": [("orders sent", "orders"), ("median wait to a fill, s", "median_wait_s"),
              ("90th percentile wait, s", "p90_wait_s"),
              ("rested and cancelled unfilled", "cancelled_unfilled_share"),
              ("post-only rejections over orders sent", "rejected_share"),
              ("queue ahead at arrival, in order sizes", "mean_ahead_sizes"),
              ("orders placed deeper than the touch", "deeper_share"),
              ("decisions with a bid resting", "bid_quoted_share"),
              ("decisions with an ask resting", "ask_quoted_share"),
              ("mean absolute inventory, units", "mean_abs_inventory"),
              ("largest absolute inventory, units", "max_abs_inventory")],
}  # fmt: skip


# an arbitrage's rows
_ARB_ROWS = {
    "costs": [("gross P&L of both legs, bps of one unit", "gross_bps"),
              ("units hedged", "hedged_units"),
              ("break-even hedge fee, bps a unit", "breakeven_hedge_fee_bps"),
              ("the brief's hedge fee, bps", "hedge_fee_bps"),
              ("net bps/h, the scored model", "rows.brief.net_bps_per_hour"),
              ("net bps/h, RA uncapped (capacity)", "rows.RA uncapped.net_bps_per_hour"),
              ("net bps/h, latency 0 (unrealistic)", "rows.latency 0.net_bps_per_hour"),
              ("net bps/h, hedge 1 s", "rows.hedge 1 s.net_bps_per_hour"),
              ("net bps/h, hedge fee 10", "rows.hedge fee 10.net_bps_per_hour"),
              ("net bps/h, hedge fee 7.5", "rows.hedge fee 7.5.net_bps_per_hour"),
              ("net bps/h, hedge fee 5", "rows.hedge fee 5.net_bps_per_hour"),
              ("net bps/h, hedge fee 0", "rows.hedge fee 0.net_bps_per_hour")],
    "capacity": [("median fill, USD", "size_usd"),
                 ("median print on the quote venue, USD", "print_usd_median"),
                 ("fill over print", "size_over_print"),
                 ("units capped at the print over uncapped", "capped_over_uncapped"),
                 ("bids filled over sells printed", "bid_share_of_sells"),
                 ("asks filled over buys printed", "ask_share_of_buys"),
                 ("hedged units past the hedge venue's level 0", "beyond_level0_share")],
    "picked_off": [("fills", "fills"), ("picked off", "picked_off_share"),
                   ("hedge venue's touch already through the price", "touch_through_share"),
                   ("captured of those picked off, bps", "picked_captured_bps")],
    "hedge": [("delay, bps a fill", "delay_bps"), ("slippage, bps a fill", "slippage_bps"),
              ("fees, bps a fill", "fee_bps"), ("net, bps a fill", "net_bps"),
              ("hedges past the depth", "unreachable_share"),
              ("visible crosses at the fill", "visible_share"),
              ("of those, still there for the hedge", "reached_share"),
              ("hedge 1 s: net, bps a fill", "slow_net_bps"),
              ("hedge 1 s: delay, bps a fill", "slow_delay_bps"),
              ("net bps/h at the brief's hedge delay", "net_bps_per_hour"),
              ("net bps/h, hedge 1 s", "slow_net_bps_per_hour")],
    "stock": [("buys", "buy_share"), ("drift, USD of coin a day onto the quote venue",
                                      "drift_usd_a_day"),
              ("days to exhaust the stock it drains", "days_to_exhaust"),
              ("decisions with the bid withdrawn for stock", "bid_withdrawn_share"),
              ("decisions with the ask withdrawn for stock", "ask_withdrawn_share"),
              ("rebalances booked", "rebalances"), ("of them, coin", "coin_rebalances"),
              ("their fees, USD", "rebalance_fee_usd")],
    "counterparty": [("prints a day on the quote venue", "prints_a_day"),
                     ("USD a day", "usd_a_day"), ("USD a print, p10", "usd_p10"),
                     ("p50", "usd_p50"), ("p90", "usd_p90"),
                     ("largest 1 USD bucket's share", "bucket_max"),
                     ("gap's coefficient of variation", "gap_cv"),
                     ("printed at a price never shown within 1 s", "never_shown")],
}  # fmt: skip


def dies_caveat() -> str:
    from blinded import dies

    return dies.CAVEAT[0].upper() + dies.CAVEAT[1:]


def _model_tables(title: str, per: dict, rows: list) -> list[str]:
    """D12 or D13: one table per segment, a column per fill model."""
    out = []
    for seg, by in per.items():
        models = list(by)
        head = "<tr><th></th>" + "".join(f"<th>{m}</th>" for m in models) + "</tr>"
        body = "".join("<tr><td>" + esc(lab) + "</td>"
                       + "".join(f"<td>{_cell(by[m], key)}</td>" for m in models) + "</tr>"
                       for lab, key in rows)  # fmt: skip
        out.append(f"<p class=plain>{title}: {plots.SEGMENT_NAME[seg]}</p>"
                   f"<table class=metrics>{head}{body}</table>")  # fmt: skip
    return out


def _dies_panel(arch, checks: dict, segs: list[str]) -> list[str]:
    """The archetype's death checks, one table per check with a column per segment; for a
    market maker D12 and D13 have a table per segment with a column per fill model."""
    out = [f"<p><b>How it dies.</b> {esc(arch.dies)}</p>"]
    maker = arch.template == "maker"
    arb = arch.template == "arb"
    for name, value in checks.items():
        title = esc(archetypes.DEATHS[name])
        note = value.get("note")
        per = {seg: value[seg] for seg in segs if seg in value}
        if arb:
            out += _arb_dies(name, title, per, note)
            continue
        rows = list((_MAKER_ROWS if maker else _DIES_ROWS).get(name)
                    or _DIES_ROWS.get(name, []))  # fmt: skip
        if maker and name in ("toxicity", "queue"):
            # a pass judged under rules 1 has no book crossing, so no crossed share
            rows = [(lab, key) for lab, key in rows if key != "crossed_share"
                    or any(key in m for by in per.values() for m in by.values()
                           if isinstance(m, dict))]  # fmt: skip
            out += _model_tables(title, per, rows)
            continue
        if maker and name == "capacity" and any(v.get("caveat") for v in per.values()):
            out.append(f"<p class=alert>{esc(dies_caveat())}.</p>")
        if name == "costs":
            carried = [(what, key) for what, key in (
                ("of which funding, which the gross includes", "funding_bps"),
                ("of which borrow on short spot, which the gross includes", "borrow_bps"),
            ) if any(key in v for v in per.values())]  # fmt: skip
            rows[1:1] = carried
            if not maker and any("maker-taker" in (v.get("rows") or {}) for v in per.values()):
                rows.append(("net bps/h, maker-taker (diagnostic)",
                             "rows.maker-taker.net_bps_per_hour"))  # fmt: skip
        if name == "capacity" and per and not maker:
            legs = list(next(iter(per.values())))
            rows = [(f"{leg}: {what}", f"{leg}.{key}") for leg in legs
                    for what, key in (("median touch share", "median"),
                                      ("fills above the touch", "above_one"),
                                      ("participation", "participation"))]  # fmt: skip
        if per and rows:
            head = "<tr><th></th>" + "".join(f"<th>{plots.SEGMENT_NAME[s]}</th>" for s in per)
            body = "".join(
                "<tr><td>" + esc(lab) + "</td>"
                + "".join(f"<td>{_cell(per[s], key)}</td>" for s in per) + "</tr>"
                for lab, key in rows
            )  # fmt: skip
            out.append(f"<p class=plain>{title}</p><table class=metrics>{head}</tr>{body}"
                       "</table>")  # fmt: skip
        if name == "costs" and per:
            out.append(_fee_grid(per))
            for seg, v in per.items():
                if v.get("fee_verdict"):
                    out.append(f"<p class=plain>{plots.SEGMENT_NAME[seg]}: "
                               f"{esc(fees.words(v['fee_verdict']))}</p>")  # fmt: skip
        if name == "funding" and per:
            out += ["<p class=plain>Funding the perpetual leg received, positive, or paid, "
                    "against what the position made on the marks, both cumulative: a basis "
                    "trade is paid by the first line and dies when the second falls further "
                    "than the first rises.</p>",
                    _chart(plots.funding(per, arch.name))]  # fmt: skip
        if name == "venue_feed" and per:
            for k, c in enumerate(next(iter(per.values()))["curves"]):
                cells = "; ".join(_venue_cells(per[s]["curves"][k], s) for s in per)
                out.append(f"<p class=plain>{title}: {esc(c['leader'])} then "
                           f"{esc(c['follower'])}: {cells}.</p>")  # fmt: skip
        for c in value.get("c_table") or []:
            cells = "; ".join(f"{plots.SEGMENT_NAME[s]} peak C {_fmt(r['c_at_peak'])} at "
                              f"{_span((r['lag_at_peak_ms'] or 0) / 1000)}, "
                              f"{_fmt(r['c_after_fill'])} on the first step after the fill"
                              for s, r in c["segments"].items())  # fmt: skip
            out.append(f"<p class=plain>{title}: {esc(c['leader'])} then "
                       f"{esc(c['follower'])}: {cells}.</p>")  # fmt: skip
        if note:
            out.append(f"<p class=plain>{title}: {esc(note)}.</p>")
    return out


def _arb_dies(name: str, title: str, per: dict, note) -> list[str]:
    """One death check of an arbitrage: a table with a column per segment, the fee verdict
    under D1, and the counterparty's line under D17."""
    rows = _ARB_ROWS.get(name) or _DIES_ROWS.get(name, [])
    out = []
    if per and rows:
        head = "<tr><th></th>" + "".join(f"<th>{plots.SEGMENT_NAME[s]}</th>" for s in per)
        body = "".join("<tr><td>" + esc(lab) + "</td>"
                       + "".join(f"<td>{_cell(per[s], key)}</td>" for s in per) + "</tr>"
                       for lab, key in rows)  # fmt: skip
        out.append(f"<p class=plain>{title}</p><table class=metrics>{head}</tr>{body}</table>")
    if name == "costs":
        for seg, v in per.items():
            if v.get("fee_verdict"):
                out.append(f"<p class=plain>{plots.SEGMENT_NAME[seg]}: "
                           f"{esc(fees.words(v['fee_verdict']))}</p>")  # fmt: skip
    if name == "counterparty" and per:
        line = next((v.get("line") for v in per.values() if v.get("line")), None)
        if line:
            out.append(f"<p class=plain>{title}: {esc(line[0].upper() + line[1:])}.</p>")
    if note:
        out.append(f"<p class=plain>{title}: {esc(note)}.</p>")
    return out


def _venue_cells(c: dict, seg: str) -> str:
    """D11's words for one curve on one segment: C at the peak against C at the cadence."""
    where = "" if c["lag_at_peak_ms"] is None else f" at {_span(c['lag_at_peak_ms'] / 1000)}"
    near = "" if c["cadence_lag_ms"] is None else f" at {_span(c['cadence_lag_ms'] / 1000)}"
    return (f"{plots.SEGMENT_NAME[seg]} peak C {_fmt(c['c_at_peak'])}{where}, C "
            f"{_fmt(c['c_at_cadence'])} at the cadence{near}")  # fmt: skip


def _basket_charts(entry: dict) -> list[str]:
    """A basket archetype's charts: the feature's cross-sectional rank IC by horizon and by
    window and the fan of the basket it implies; the strategy's P&L leg by leg and the fan of
    its rebalances."""
    out = []
    feat, strat = entry.get("feature"), entry.get("strategy")
    if feat:
        n = f"<code>{esc(feat['name'])}</code> (trial {feat['trial_id']})"
        p = feat["params"]
        out += [f"<h3>The feature: {n}, {esc(feat['why'])}</h3>",
                "<p>A basket feature is scored by its cross-sectional rank IC: at each grid "
                "point, the rank correlation across the legs of the feature with each leg's "
                "forward return, averaged over the window. The decay is that average at each "
                "horizon.</p>",
                _chart(plots.decay(feat["decay"], feat["horizons"], feat["name"]), narrow=True),
                f"<p>The same at {esc(feat['primary'])}, inside each window of "
                f"{esc(feat['every'])}. A feature that ranks the legs well in only a few is "
                "fragile.</p>",
                _chart(plots.windows(feat["windows"], feat["name"], feat["every"]))]  # fmt: skip
        out += _basket_fan_chart(feat, f"The basket of {feat['name']}",
                                 f"bl.basket(top={p['top']}, every={p['every']}) on the "
                                 "feature, rebuilt at each window")  # fmt: skip
    if strat:
        who = esc(strat["why"])
        if strat.get("name"):
            who = f"<code>{esc(strat['name'])}</code> (trial {strat['trial_id']}), {who}"
        out += [f"<h3>The strategy: {who}</h3>",
                "<p>Net P&L leg by leg, the thin lines, and the basket's, the bold one: a "
                "basket carried by one or two names shows one thin line far from the rest. "
                "D9 below counts it.</p>",
                _chart(plots.basket_pnl(strat["pnl"], strat.get("name") or "the basket"))
                ]  # fmt: skip
        out += _basket_fan_chart(strat, f"Rebalances of {strat.get('name') or 'the basket'}",
                                 "the strategy's own positions")  # fmt: skip
    return out


def _basket_fan_chart(item: dict, title: str, what: str) -> list[str]:
    if not item.get("fan"):
        return []
    scale = item.get("fan_scale", 1)
    xlabel, unit_s = _fan_axis("from the rebalance", scale)
    text = _bar_text(f"<p>The markout fan of {esc(what)}: at every rebalance, the move of each "
                     "leg weighted by its new position over the gross, so one notional across "
                     "the legs, from 10 s before the decision to 60 s after; with weights of "
                     "plus and minus one over the legs a side, half the long legs' mean return "
                     "less the short legs'. Rebalances less than 70 s apart share prices and "
                     "count as one draw in the standard error. The dotted line is the fill; "
                     "a basket pays where the line, after the fill, clears what it costs to "
                     "turn over.</p>", scale)  # fmt: skip
    unit = {"x_unit_s": unit_s} if scale != 1 else {}
    fan = plots.event_fan(item["fan"], item["offsets"], item["latency_ms"], title,
                          "basket move, bps a notional", xlabel, unit="rebalances",
                          **unit)  # fmt: skip
    table = _fan_table(item["fan"], item["offsets"], item["latency_ms"], "rebalances", scale)
    return [text, _chart(fan), table]


def _fee_grid(per: dict) -> str:
    """D1's net bps/hour at each fee and latency, folded, one table per segment."""
    parts = []
    for seg, c in per.items():
        g = c["grid"]
        fees = "".join(f"<th>fee {esc(f)}{'' if f == 'brief' else ' bps'}</th>" for f in g["fees"])
        head = f"<tr><th>latency</th>{fees}</tr>"
        body = "".join(f"<tr><td>{lat:g} ms</td>" + "".join(f"<td>{_fmt(v)}</td>" for v in row)
                       + "</tr>" for lat, row in zip(g["latencies_ms"], g["net_bps_per_hour"],
                                                     strict=True))  # fmt: skip
        parts.append(f"<p class=plain>{plots.SEGMENT_NAME[seg]}</p><table class=metrics>{head}"
                     f"{body}</table>")  # fmt: skip
    return ("<details><summary>Net bps/hour at each fee and latency</summary>" + "".join(parts)
            + "</details>")  # fmt: skip


def _dig(d, path: str):
    """The value at a dotted ``path`` into nested dicts, a key itself allowed to hold a dot
    (``rows.rebate 0.5.net_bps_per_hour``): the longest key that starts the rest is taken.
    None when the path leads nowhere."""
    while path:
        if not isinstance(d, dict):
            return None
        for k in sorted(d, key=len, reverse=True):
            if path == k or path.startswith(f"{k}."):
                d, path = d[k], path[len(k) + 1 :]
                break
        else:
            return None
    return d


def _cell(d: dict, key: str) -> str:
    key, _, how = key.partition(":")
    value = _dig(d, key)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return esc(value)
    if how == "span":
        if value is not None and not math.isfinite(value):
            return "never"  # an AR(1) slope at or above 1: no reversion at all
        return "" if value is None else esc(_span(value))
    if isinstance(value, float) and value and abs(value) < 1e-3:
        return f"{value:.3g}"
    return _fmt(value)


def _span(seconds: float) -> str:
    """A duration in words, to the minute from an hour up, to the second from a minute up.

    Examples
    --------
    >>> _span(63451.838), _span(75.4), _span(0.1)
    ('17 h 38 min', '1 min 15 s', '100 ms')
    """
    unit = 60 if seconds >= 3600 else 1 if seconds >= 60 else 0.001
    return words(pd.Timedelta(seconds=round(seconds / unit) * unit))


def _single_leg_charts(feat: dict | None, strat: dict | None) -> list[str]:
    """The feature's fan, decay and IC per window, and the strategy's P&L and order markout."""
    out = []
    if feat:
        n = f"<code>{esc(feat['name'])}</code> (trial {feat['trial_id']})"
        out.append(f"<h3>The feature: {n}, {esc(feat['why'])}</h3>")
        scale = feat.get("fan_scale", 1)
        out.append(_bar_text(_feature_fan_words(feat), scale))
        fee = next(iter(feat["costs"].values()))["fee_bps"]
        levels = {seg: [("break-even, spread only", f["breakeven_bps"], "--"),
                        (f"with {fee:g} bps fees", f["breakeven_fees_bps"], "-.")]
                  for seg, f in feat["fan"].items()}  # fmt: skip
        xlabel, unit_s = _fan_axis("from the decision (order sent at 0)", scale)
        out.append(_chart(plots.event_fan(
            feat["fan"], feat["offsets"], feat["latency_ms"], f"Markout fan of {feat['name']}",
            "move in the traded direction, bps", xlabel, levels=levels,
            **({"x_unit_s": unit_s} if scale != 1 else {}))))  # fmt: skip
        out.append(_fan_table(feat["fan"], feat["offsets"], feat["latency_ms"], "events", scale))
        out.append(_cost_line(feat["costs"]))
        out.append(_chart(plots.decay(feat["decay"], feat["horizons"], feat["name"]),
                          narrow=True))  # fmt: skip
        inside = f"{esc(feat['primary'])} inside each {esc(feat['every'])} window"
        if scale != 1:
            at, every = (words(pd.Timedelta(feat[k])) for k in ("primary", "every"))
            inside = f"{esc(at)} inside each window of {esc(every)}"
        out.append(f"<p>Rank IC at {inside}. A signal that works in only a few is "
                   "fragile.</p>")  # fmt: skip
        out.append(_chart(plots.windows(feat["windows"], feat["name"], feat["every"])))
    if strat:
        n = f"<code>{esc(strat['name'])}</code> (trial {strat['trial_id']})"
        out.append(f"<h3>The strategy: {n}, {esc(strat['why'])}</h3>")
        out.append("<p>The same positions three ways: the mid move alone, after paying the "
                   "spread, and after the fee as well. The gap between two lines is what "
                   "that cost took.</p>")  # fmt: skip
        out.append(_chart(plots.pnl(strat["pnl"], strat["name"])))
        scale = strat.get("fan_scale", 1)
        out.append(_bar_text("<p>This is every order the strategy sent, on the clock of the "
                             "send. The "
                   "line is the size-weighted mid move in the order's direction from 10 s "
                   "before to 60 s after, with two standard errors either side. Orders less "
                   "than 70 s apart share prices, so the standard error counts each such run "
                   "as one draw. A line rising into zero means the strategy buys after the "
                   "price has gone up. The diamond is the average fill, the half spread plus "
                   "whatever moved in flight. An order pays only where the line clears the "
                   "dashed line, the fill plus the fee.</p>", scale))  # fmt: skip
        fee = strat["fee_bps"]
        fill_s = strat["latency_ms"] / 1000
        mks = strat["markout"]
        levels = {seg: [(f"fill + {fee:g} bps fee",
                         None if m["entry_bps"] is None else m["entry_bps"] + fee, "--")]
                  for seg, m in mks.items()}  # fmt: skip
        points = {seg: [("average fill", fill_s, m["entry_bps"])] for seg, m in mks.items()}
        xlabel, unit_s = _fan_axis("from the order's send", scale)
        out.append(_chart(plots.event_fan(
            mks, strat["offsets"], strat["latency_ms"], f"Order markout fan of {strat['name']}",
            "mid move in the order's direction, bps", xlabel, unit="orders", levels=levels,
            points=points, **({"x_unit_s": unit_s} if scale != 1 else {}))))  # fmt: skip
        out.append(_fan_table(mks, strat["offsets"], strat["latency_ms"], "orders", scale))
    return out


def _lead_lag_charts(lead: dict) -> list[str]:
    """One cross-correlation chart and table per leader and follower, then the family's pick."""
    out = ["<h3>Lead-lag</h3>"]
    curves = lead.get("curves") or []
    if curves:
        c = curves[0]
        out.append(
            "<p>Each point is the correlation, over the window, of the leader's last "
            f"{c['step_ms']:g} ms return at a decision with the follower's {c['step_ms']:g} ms "
            "return ending a lag later. A follower that copies the leader with a delay peaks "
            "at that delay; a peak left of zero means the follower moved first, and a curve "
            "symmetric about zero is two prices moving together. The shaded strip is the "
            f"{c['fill_ms']:g} ms from the decision to a taker's fill: a follower move that "
            "ends inside it is over before the order lands. The dotted line is the follower's "
            "first whole step after the fill, the step a taker holds; what is right of it is "
            "what a taker can still trade.</p>")  # fmt: skip
    for c in curves:
        out.append(_chart(plots.lead_lag(c)))
        out.append(_lead_lag_table(c))
    feat, strat = lead.get("feature"), lead.get("strategy")
    if feat or strat:
        out.append("<p>The lead-lag family's own feature and strategy, on the follower they "
                   "trade:</p>")  # fmt: skip
        out += _single_leg_charts(feat, strat)
    return out


def _lead_lag_table(c: dict) -> str:
    head = ["segment", "lag at the peak, ms", "C at the peak",
            "C on the first step after the fill", "best C from there on",
            "usable by a taker"]  # fmt: skip
    rows = []
    for seg, r in c["segments"].items():
        lag = r["lag_at_peak_ms"]
        cells = [plots.SEGMENT_NAME[seg], "" if lag is None else f"{lag:g}", _fmt(r["c_at_peak"]),
                 _fmt(r["c_after_fill"]), _fmt(r["best_after_fill"]),
                 "yes" if r["usable_by_taker"] else "no"]  # fmt: skip
        rows.append("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in cells) + "</tr>")
    return ("<table class=metrics><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head)
            + "</tr>" + "".join(rows) + "</table>")  # fmt: skip


def _pair_charts(item: dict) -> list[str]:
    """The four pairs charts of one pair, with what each shows in words."""
    a, b = item["pair"]
    p = item["params"]
    who = esc(item["why"])
    if item.get("name"):
        who = f"<code>{esc(item['name'])}</code> (trial {item['trial_id']}), {who}"
    out = [f"<h3>Pairs: {esc(a)} over {esc(b)}, {who}</h3>",
           f"<p>Fitted on the trailing {esc(p['window'])} before each {esc(p['every'])} clock "
           f"window and held through it; bands at z {p['z_in']:g} in and {p['z_out']:g} out, "
           f"from {esc(item['params_from'])}.</p>",
           "<p>The spread is the log mid of A less the hedge ratio times that of B, here "
           "less its walk-forward mean, in bps. Its level jumps at every refit, so the mean "
           "is taken off. The dashed lines are where the strategy goes in, the dotted ones "
           "where it comes out, and the shading is where it held the spread.</p>",
           _chart(plots.pair_spread(item)),
           "<p>The hedge ratio, the units of B the spread holds against one of A, fitted on "
           "the window before each clock window. A ratio that wanders makes the B leg trade "
           "at every refit.</p>",
           _chart(plots.hedge_ratio(item)),
           "<p>How fast the spread came back inside each clock window: the half-life of an "
           "AR(1) fit. An open circle is a window where it did not come back at all. The dashed "
           "line is what the same fit reads on a pure random walk in a window this long, since "
           "the fitted slope sits below 1 even with no reversion: only points well under it "
           "show the spread coming back.</p>",
           _chart(plots.half_life(item)),
           _bar_text(f"<p>In the markout fan of the spread, every time its z-score went past "
           "the entry "
           f"band of &plusmn;{p['z_in']:g} counts as a trade back towards the mean: short the "
           "spread when z is high, long when it is low. Events are at least 70 s apart. The "
           "line is the average move of the spread in that direction, at the hedge "
           "ratio of the moment, with two standard errors either side. A spread that "
           "reverts rises after zero. Breaking even takes crossing the spread of both "
           "legs twice, the B leg scaled by the hedge ratio, and the fees of both on "
           "top.</p>", item.get("fan_scale", 1))]  # fmt: skip
    levels = {seg: [("break-even, spread only", f["breakeven_bps"], "--"),
                    ("with fees", f["breakeven_fees_bps"], "-.")]
              for seg, f in item["fan"].items()}  # fmt: skip
    scale = item.get("fan_scale", 1)
    xlabel, unit_s = _fan_axis("from the decision (order sent at 0)", scale)
    out.append(_chart(plots.event_fan(
        item["fan"], item["offsets"], item["latency_ms"],
        f"Markout fan of the {a} over {b} spread", "move of the spread, traded direction, bps",
        xlabel, levels=levels, **({"x_unit_s": unit_s} if scale != 1 else {}))))  # fmt: skip
    out.append(_fan_table(item["fan"], item["offsets"], item["latency_ms"], "events", scale))
    return out


FAN_TABLE = (("-5s", "-1s"), ("1s", "5s", "10s", "30s", "60s"))  # either side of the fill


def _bar_text(text: str, scale: int) -> str:
    """A fan's words on bars, where each of its seconds is one bar (:func:`_fan_scale`)."""
    if scale == 1:
        return text
    for n in (70, 10, 60):
        text = text.replace(f"{n} s", words(pd.Timedelta(seconds=n * scale)))
    return text.replace("a second of price", "a bar of price")


def _fan_axis(where: str, scale: int) -> tuple[str, float]:
    """The fan's axis label and unit: seconds, or on bars hours (minutes under an hour)."""
    if scale == 1:
        return f"seconds {where}", 1.0
    return (f"hours {where}", 3600.0) if scale >= 3600 else (f"minutes {where}", 60.0)


def _feature_fan_words(feat: dict) -> str:
    ev = feat["events"]
    flipped = ""
    if ev["direction"] < 0:
        flipped = (" Its rank IC on train is negative, so a high value is taken as a short and "
                   "a low one as a long.")  # fmt: skip
    return ("<p>Every time the feature went past its 5th or 95th percentile on train counts "
            f"as a trade in the direction it predicts.{flipped} Events are at least 70 s "
            "apart, so no two paths share a second of price and the standard error is a fair "
            "one. The thick line is the average mid move in the trade's direction, from 10 s "
            "before the decision to 60 s after, and the band is two standard errors either "
            "side; it widens as single paths drift apart. Longs and shorts are drawn "
            "separately, to show a signal that works in one direction only. Left of zero is "
            "how the price got there. The dotted line is the fill, the latency plus the feed "
            "delay after the decision. A taker who fills there and crosses the spread again "
            "to get out needs the line above the dashed one, and above the dash-dot one once "
            "fees are paid.</p>")  # fmt: skip


def _fan_table(fan: dict, offsets: list[str], fill_ms: float, unit: str, scale: int = 1) -> str:
    """The fan's mean and two standard errors at a few offsets, one column per segment; on
    bars each offset is ``scale`` times as long."""
    fill = _signed(round(fill_ms * 1e6))
    table = (*FAN_TABLE[0], fill, *FAN_TABLE[1])
    if scale != 1:
        table = tuple(o if o == fill else _signed(pd.Timedelta(o).value * scale) for o in table)
    rows = list(dict.fromkeys(o for o in table if o in offsets))  # on bars 1 s is the fill
    head = ["seconds from the send" if scale == 1 else "time from the send"] + [
        f"{plots.SEGMENT_NAME[seg]}, {f['n']:,} {unit}" for seg, f in fan.items()
    ]
    body = []
    for o in rows:
        k = offsets.index(o)
        shown = o
        if scale != 1:
            ns = pd.Timedelta(o).value
            shown = ("-" if ns < 0 else "") + words(pd.Timedelta(abs(ns), unit="ns"))
        cells = [f"{shown} (the fill)" if o == fill else shown]
        for f in fan.values():
            m, se = f["mean_bps"][k], f["se_bps"][k]
            band = "" if se is None else f" &plusmn; {2 * se:.2f}"
            cells.append("" if m is None else f"{m:+.2f}{band}")
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return ("<table class=metrics><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head)
            + "</tr>" + "".join(body) + "</table>"
            "<p class=plain>Mean move in bps &plusmn; two standard errors.</p>")  # fmt: skip


def _cost_line(costs: dict) -> str:
    parts = []
    for seg, c in costs.items():
        if c.get("half_spread_bps") is None:
            continue
        parts.append(f"{plots.SEGMENT_NAME[seg]} {c['half_spread_bps']:.2f} bps half spread, "
                     f"{c['round_trip_bps']:.1f} bps round trip")  # fmt: skip
    if not parts:
        return ""
    fee = next(iter(costs.values()))["fee_bps"]
    return f"<p class=plain>Taker fee {fee:g} bps; " + "; ".join(parts) + ".</p>"


def _chart(svg: str, narrow: bool = False) -> str:
    klass = "chart narrow" if narrow else "chart"
    return f"<figure class='{klass}'>{svg}</figure>"


def _rejected_section(model: dict) -> str:
    groups: dict[str, list] = {}
    for r in model["rejected"]:
        groups.setdefault(r["status"], []).append(r)
    out = ["<section><h2>5. Failed ideas and why</h2>"]
    if not groups:
        out.append("<p>Nothing was rejected.</p></section>")
        return "\n".join(out)
    for status, rows in sorted(groups.items()):
        out.append(f"<h3>{esc(status)} ({len(rows)})</h3><table class=rej>"
                   "<tr><th>id</th><th>kind</th><th>name</th><th>hypothesis</th>"
                   "<th>reason</th></tr>")  # fmt: skip
        for r in rows:
            out.append(f"<tr><td>{r['trial_id']}</td><td>{esc(r['kind'])}</td>"
                       f"<td>{esc(r['name'])}</td><td>{esc(r['hypothesis'] or '')}</td>"
                       f"<td>{esc(r['reason'] or '')}</td></tr>")  # fmt: skip
        out.append("</table>")
    out.append("</section>")
    return "\n".join(out)


def _naive_section(model: dict) -> str:
    naive = model.get("naive")
    if not naive:
        return ""
    claimed = naive.get("claimed") or {}
    protected = naive.get("protected_test") or {}
    pbrief = (protected.get("rows") or {}).get("brief", {})
    out = ["<section><h2>6. Naive comparison</h2>",
           "<p>The same agent and budget with the protections removed, then its final pick "
           "re-scored by the protected judge on the same locked test.</p>",
           "<table class=metrics><tr><th></th><th>naive loop claimed</th>"
           "<th>protected judge, same code, locked test</th></tr>"]  # fmt: skip
    out.append(f"<tr><td>Sharpe/sqrt h</td><td>{_fmt(claimed.get('sharpe_per_sqrt_hour'))}</td>"
               f"<td>{_fmt(pbrief.get('sharpe_per_sqrt_hour'))}</td></tr>")  # fmt: skip
    out.append(f"<tr><td>net bps/hour</td><td>{_fmt(claimed.get('net_bps_per_hour'))}</td>"
               f"<td>{_fmt(pbrief.get('net_bps_per_hour'))}</td></tr>")  # fmt: skip
    c = model["counts"]
    calls = sum(c[k][s] for k in ("feature", "strategy") for s in ("accepted", "rejected"))
    out.append(f"<tr><td>judge calls</td><td>{naive['n_calls']}</td><td>{calls}</td></tr>")
    ours = {f["family"]: f["feature"]["submitted"] + f["strategy"]["submitted"]
            for f in model.get("families") or []}  # fmt: skip
    for family, n in (naive.get("calls_by_family") or {}).items():
        mine = ours.get(family, "")
        out.append(f"<tr><td>judge calls, {esc(family)}</td><td>{n}</td><td>{mine}</td></tr>")
    if naive.get("calls_by_archetype") is not None:
        mine = (model.get("archetypes") or {}).get("calls_by_archetype") or {}
        for name in archetypes.REGISTRY:
            if name in naive["calls_by_archetype"] or name in mine:
                n = naive["calls_by_archetype"].get(name, 0)
                out.append(f"<tr><td>judge calls, {esc(name)}</td><td>{n}</td>"
                           f"<td>{mine.get(name, 0)}</td></tr>")  # fmt: skip
    out.append("</table>")
    if naive.get("final_id") is not None:
        line = f"The naive pick is trial {naive['final_id']} ({esc(naive.get('final_name'))})"
        if naive.get("family"):
            line += f", of the {esc(naive['family'])} family"
        if claimed.get("window"):
            line += f", claimed on the {esc(claimed['window'])} window"
        if naive.get("override") and naive.get("agent_pick") is None:
            line += (f". The agent named no final trial, so the driver took "
                     f"{esc(naive['override'])}, as a loop without a judge would")  # fmt: skip
        out.append(f"<p>{line}.</p>")
    out.append("</section>")
    return "\n".join(out)


def _reproduce_section(model: dict) -> str:
    rep = model["reproduce"]
    seg = model["segments"]
    out = [
        "<section><h2>7. Reproduce</h2>",
        f"<p>Recompute every number above from your own copy of the data with "
        f"<code>blinded replay {esc(model['run_id'])}/</code>. The data is never embedded; "
        "the split boundaries and the SHA-256 of each segment are below.</p>",
        "<table class=metrics><tr><th>segment</th><th>rows</th><th>sha256</th></tr>",
    ]
    for name, s in seg.items():
        out.append(f"<tr><td>{esc(name)}</td><td>{s['rows']:,}</td>"
                   f"<td><code>{esc(s['sha256'][:16])}</code></td></tr>")  # fmt: skip
    out.append("</table>")
    out.append(f"<details><summary>brief.yaml</summary><pre>{esc(rep['brief'])}</pre></details>")
    for name, code in rep["code"].items():
        out.append(f"<details><summary>{esc(name)}</summary><pre>{esc(code)}</pre></details>")
    out.append("</section>")
    return "\n".join(out)


def _appendix_section(model: dict) -> str:
    a = model["appendix"]
    out = ["<section><h2>8. Appendix</h2>",
           f"<p>Sandbox probes: {len(model['probes'])} (expected 0).</p>",
           "<h3>Ledger</h3><table class=rej>"
           "<tr><th>id</th><th>kind</th><th>name</th><th>status</th><th>iter</th>"
           "<th>released</th><th>note</th></tr>"]  # fmt: skip
    for r in a["ledger"]:
        rel = "" if r["released"] is None else ("yes" if r["released"] else "no")
        it = r["iteration"] if r["iteration"] else ""
        cells = (f"<td>{r['id']}</td><td>{esc(r['kind'])}</td><td>{esc(r['name'] or '')}</td>"
                 f"<td>{esc(r['status'])}</td><td>{it}</td><td>{rel}</td>"
                 f"<td>{esc(r['note'] or '')}</td>")  # fmt: skip
        out.append(f"<tr>{cells}</tr>")
    out.append("</table>")
    if a["notes"].strip():
        out.append(f"<details><summary>NOTES.md</summary><pre>{esc(a['notes'])}</pre></details>")
    if a["data_md"].strip():
        out.append(f"<details><summary>DATA.md</summary><pre>{esc(a['data_md'])}</pre></details>")
    out.append("</section>")
    return "\n".join(out)


# -- replay -----------------------------------------------------------------


def replay(run_dir: str | Path) -> dict:
    """Recompute the report's data-derived numbers and check them against ``report.json``.

    Returns ``{"ok": bool, "checks": [{name, reported, recomputed, ok}, ...]}``. A
    mismatch beyond :data:`TOL` fails. The report must have been written first. The run's
    segment, touch and depth files and its manifest are checked against the hashes written
    at init first (:meth:`blinded.vault.RunDir.verify`), so a replay recomputes from the data
    the run was judged on or not at all.
    """
    run = RunDir.open(run_dir)
    run.verify()
    saved = json.loads((run.root / "report.json").read_text())
    fresh = recompute(run, saved.get("final_id"))
    pairs = _flat_pairs(saved.get("recompute", {}), fresh)
    extra = []
    if saved.get("archetypes"):
        extra = _menu_checks(run, saved["archetypes"]["menu"])
    naive = saved.get("naive") or {}
    found = _naive_dir(run)
    if found is not None and naive.get("final_id") is not None:
        theirs = recompute(RunDir.open(found), naive["final_id"], diagnostics=False)
        theirs = theirs.get("test_rows", {})
        shown = (naive.get("protected_test") or {}).get("rows", {})
        for row, metrics in shown.items():
            for k, v in metrics.items():
                pairs.append((f"naive.test_rows.{row}.{k}", v, theirs.get(row, {}).get(k)))
    checks = [{"name": n, "reported": a, "recomputed": b, "ok": _close(a, b)} for n, a, b in pairs]
    for key in sorted(k for k in (saved.get("recompute") or {}) if k.startswith("rules_v")):
        shown = saved["recompute"][key] or {}
        for word in ("fill_model", "refused"):
            if word in shown:
                theirs = (fresh.get(key) or {}).get(word)
                checks.append({"name": f"{key}.{word}", "reported": shown[word],
                               "recomputed": theirs, "ok": shown[word] == theirs})  # fmt: skip
    checks += extra
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def _menu_checks(run: RunDir, shown: dict) -> list[dict]:
    """The menu the agent was shown against the one the vault's brief and training rows give:
    every number in it, and every archetype's verdict and line as text."""
    brief = run.brief()
    panel = run.panel("train")
    fresh = archetypes.menu(brief, archetypes.facts(brief, panel, maker=True)).to_dict()
    fresh = json.loads(json.dumps(fresh))  # as it was stored
    checks = [{"name": n, "reported": a, "recomputed": b, "ok": _close(a, b)}
              for n, a, b in _leaf_pairs("archetypes.menu", shown, fresh)]  # fmt: skip
    theirs = {a["name"]: a for a in fresh["archetypes"]}
    for mine in shown["archetypes"]:
        other = theirs.get(mine["name"], {})
        for key in ("status", "line"):
            name = f"archetypes.menu.{mine['name']}.{key}"
            checks.append({"name": name, "reported": mine[key], "recomputed": other.get(key),
                           "ok": mine[key] == other.get(key)})  # fmt: skip
    for key in ("edges", "headline", "power", "cost_model", "dropped_line"):
        checks.append({"name": f"archetypes.menu.{key}", "reported": shown[key],
                       "recomputed": fresh[key], "ok": shown[key] == fresh[key]})  # fmt: skip
    return checks


def _flat_pairs(saved: dict, fresh: dict):
    def at(d, key, i=None):
        v = d.get(key)
        if i is None:
            return v or {}
        return (v or [])[i] if v and i < len(v) else {}

    pairs = []
    fresh_rows = at(fresh, "test_rows")
    for row, metrics in at(saved, "test_rows").items():
        for k, v in metrics.items():
            pairs.append((f"test_rows.{row}.{k}", v, fresh_rows.get(row, {}).get(k)))
    pairs += _leaf_pairs("diagnostics", saved.get("diagnostics"), fresh.get("diagnostics"))
    pairs += _leaf_pairs("final.fee_verdict", at(saved, "final").get("fee_verdict"),
                         at(fresh, "final").get("fee_verdict"))  # fmt: skip
    for key in sorted(k for k in saved if k.startswith("rules_v")):
        pairs += _leaf_pairs(key, saved.get(key), fresh.get(key))
    fresh_curves = at(fresh, "curves")
    for name, curve in at(saved, "curves").items():
        a = curve[-1] if curve else None
        other = fresh_curves.get(name) or []
        pairs.append((f"curves.{name}.last", a, other[-1] if other else None))
    return pairs


def _leaf_pairs(path: str, saved, fresh) -> list[tuple]:
    """Every number under ``saved``, paired with the number at the same place in ``fresh``."""
    if isinstance(saved, dict):
        fresh = fresh if isinstance(fresh, dict) else {}
        return [p for k, v in saved.items() for p in _leaf_pairs(f"{path}.{k}", v, fresh.get(k))]
    if isinstance(saved, list):
        fresh = fresh if isinstance(fresh, list) else []
        return [
            p
            for i, v in enumerate(saved)
            for p in _leaf_pairs(f"{path}[{i}]", v, fresh[i] if i < len(fresh) else None)
        ]
    if isinstance(saved, bool) or not isinstance(saved, int | float):
        return []
    return [(path, saved, fresh)]


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= TOL + TOL * abs(float(a))


# -- formatting -------------------------------------------------------------


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value) -> str:
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _hm(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h{minutes % 60:02d}m"


def naive_table(*args, **kwargs):  # pragma: no cover - kept for the API surface
    return _naive_table(*args, **kwargs)


_CSS = """
:root { --fg:#1a1a1a; --mut:#666; --line:#d9d9d9; --accent:#2a4d7a; --bad:#9a2a2a; --bg:#fff; }
* { box-sizing:border-box; }
body { font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--fg);
       background:var(--bg); max-width:860px; margin:0 auto; padding:24px 16px; }
h1 { font-size:24px; margin:0; }
h2 { font-size:19px; margin:28px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
h3 { font-size:15px; margin:16px 0 6px; color:var(--mut); }
.sub { color:var(--mut); margin:4px 0 0; }
.verdict { font-size:17px; font-weight:600; }
.alert { color:var(--bad); font-weight:600; }
.plain { color:var(--mut); }
.note { color:var(--mut); font-style:italic; }
table { border-collapse:collapse; width:100%; margin:8px 0; font-size:13px; }
th,td { text-align:right; padding:3px 8px; border-bottom:1px solid var(--line); }
th:first-child,td:first-child { text-align:left; }
tr.diag td { color:var(--mut); }
table.rej th, table.rej td { text-align:left; }
table.rej { display:block; overflow-x:auto; }
code { font-family:ui-monospace,Menlo,monospace; font-size:0.92em; }
pre { background:#f6f6f6; padding:10px; overflow:auto; font-size:12px; border-radius:4px; }
details { margin:6px 0; }
summary { cursor:pointer; color:var(--accent); }
figure.chart { margin:12px 0 18px; }
figure.chart.narrow { max-width:560px; margin-left:auto; margin-right:auto; }
figure.chart svg { width:100%; height:auto; display:block; }
"""

_PAGE = """<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style></head>
<body>
{body}
</body></html>
"""
