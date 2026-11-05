"""Acceptance (c): planted leaks are rejected, causal controls pass, noise is non-deterministic.

Each leak is written twice, once as polars expressions and once in pandas
through ``panel.to_pandas()``; the judge has to catch both.
"""

from pathlib import Path

import pytest

import blinded as bl

PLANTED = Path(__file__).parent / "planted"
LEAKS = sorted(PLANTED.glob("leak_*.py"))
CONTROLS = sorted(PLANTED.glob("control_*.py"))
PAIRS = PLANTED / "pairs"


def judge(path, brief, panel):
    (feature,) = bl.load(path).features
    return bl.evaluate(feature, brief, window="train", panel=panel)


def test_the_planted_set_is_complete():
    polars = {p.stem for p in LEAKS if not p.stem.endswith("_pandas")}
    pandas = {p.stem.removesuffix("_pandas") for p in LEAKS if p.stem.endswith("_pandas")}
    assert len(polars) == 8
    assert pandas == polars
    assert len(CONTROLS) == 6


@pytest.mark.parametrize("path", LEAKS, ids=lambda p: p.stem)
def test_leak_is_rejected_with_its_first_cut(path, dataset):
    _, brief, panel = dataset
    result = judge(path, brief, panel)
    assert result.status == "rejected_lookahead", result.reason
    violation = result.verdict.violation
    assert violation.cut in result.verdict.cuts
    assert violation.first <= violation.cut
    assert violation.n_bad > 0


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.stem)
def test_causal_control_passes(path, dataset):
    _, brief, panel = dataset
    result = judge(path, brief, panel)
    assert result.status == "accepted", result.reason
    assert len(result.verdict.cuts) == 7


def test_unseeded_noise_is_nondeterministic(dataset):
    _, brief, panel = dataset
    result = judge(PLANTED / "noise.py", brief, panel)
    assert result.status == "rejected_nondeterministic"
    assert result.verdict.violation.cut is None


@pytest.mark.parametrize("path", sorted(PAIRS.glob("leak_*.py")), ids=lambda p: p.stem)
def test_pairs_leak_is_rejected_with_its_first_cut(path, pairs_brief):
    # a hedge ratio, and a mean and scale of the spread, fitted on the whole column
    result = judge(path, pairs_brief, None)
    assert result.family == "pairs" and result.legs == ("XRPUSDT_PERP", "XRPUSDT")
    assert result.status == "rejected_lookahead", result.reason
    violation = result.verdict.violation
    assert violation.cut in result.verdict.cuts
    assert violation.first <= violation.cut


def test_the_walk_forward_pairs_control_passes(pairs_brief):
    (strat,) = bl.load(PAIRS / "control_walk_forward.py").strategies
    result = bl.evaluate_strategy(strat, pairs_brief, window="train")
    assert result.ok, result.reason
    assert len(result.verdict.cuts) == 7 and list(result.position) == ["XRPUSDT_PERP", "XRPUSDT"]
    assert result.row()["fills"] > 0  # it traded, so passing is not passing by being flat
