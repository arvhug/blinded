import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import write_brief

import blinded as bl
from blinded.cli import main

known = pytest.mark.known_answer
cross_check = pytest.mark.cross_check
edge = pytest.mark.edge_case
PLANTED = Path(__file__).parent / "planted"

FEATURES = """
import blinded as bl
from blinded import features as F


@bl.feature("perp leads spot; the basis mean-reverts")
def basis(panel):
    b = F.basis("XRPUSDT_PERP", "XRPUSDT")
    return b - F.roll_mean(b, "5m")


@bl.feature("one step ahead")
def peek(panel):
    return F.mid("XRPUSDT").shift(-1)


@bl.feature("broken")
def broken(panel):
    return F.mid("XRPUSDT") / undefined_name


@bl.feature("the same basis, written in pandas")
def basis_pandas(panel):
    df = panel.to_pandas()
    b = df["XRPUSDT_PERP.mid"] / df["XRPUSDT.mid"] - 1
    return b - b.rolling("5min").mean()
"""


@pytest.fixture
def features(tmp_path):
    path = tmp_path / "features.py"
    path.write_text(FEATURES)
    return path


@pytest.fixture
def module(features):
    return bl.load(features)


@known
def test_accepted_feature_has_metrics_by_horizon(module, synthetic_brief):
    basis = module.features[0]
    result = basis.evaluate(synthetic_brief, window="train")
    frame = result.to_pandas()
    assert result.ok and result.status == "accepted"
    assert list(frame.index) == ["100ms", "1s", "10s", "60s"]
    assert {"ic", "rank_ic", "ic_se_nw", "rank_ic_se_boot", "n_eff", "markout_bps"} <= set(frame)
    assert frame.loc["10s", "n_eff"] == pytest.approx(frame.loc["10s", "n"] / 100)
    # the synthetic spot quotes the fair price 300 ms late, so the basis predicts spot
    assert frame.loc["1s", "rank_ic"] > 0.1
    text = repr(result)
    assert "basis  [accepted]" in text and "decay" in text and "passed" in text


@known
def test_train_window_starts_after_the_warm_up(module, synthetic_brief, synthetic_panel):
    result = module.features[0].evaluate(synthetic_brief)
    assert result.window.start == synthetic_panel.time(0) + pd.Timedelta("5min")


@known
def test_rejected_feature_reports_the_cut(module, synthetic_brief):
    result = module.features[1].evaluate(synthetic_brief)
    assert not result.ok
    assert result.status == "rejected_lookahead"
    assert "cut at" in result.reason
    assert result.to_frame().is_empty()


@known
def test_error_names_the_line(module, synthetic_brief):
    result = module.features[2].evaluate(synthetic_brief)
    assert result.status == "error"
    assert result.reason.startswith("NameError at features.py:")


@known
def test_static_rejection(tmp_path, synthetic_brief):
    path = tmp_path / "f.py"
    path.write_text(
        textwrap.dedent(
            """
            import os
            import blinded as bl

            @bl.feature("reads files")
            def f(panel):
                return bl.features.mid("XRPUSDT")
            """
        )
    )
    result = bl.load(path).features[0].evaluate(synthetic_brief)
    assert result.status == "rejected_static"
    assert "imports os" in result.reason


@known
def test_latency_shifts_the_return_window(module, synthetic_brief):
    basis = module.features[0]
    fast = basis.evaluate(synthetic_brief, latency_ms=0).to_pandas()
    slow = basis.evaluate(synthetic_brief, latency_ms=500).to_pandas()
    # spot catches up within ~300 ms; starting the return 500 ms later loses most of it
    assert slow.loc["1s", "rank_ic"] < fast.loc["1s", "rank_ic"]


@edge
def test_explicit_window(module, synthetic_brief):
    spec = "2024-11-28T02:10:00/2024-11-28T02:20:00"
    result = module.features[0].evaluate(synthetic_brief, window=spec)
    assert result.ok
    assert result.to_pandas().loc["100ms", "n"] <= 6000


@known
def test_a_pandas_feature_scores_the_same_as_its_polars_twin(module, synthetic_brief):
    polars_basis, pandas_basis = module.features[0], module.features[3]
    a = polars_basis.evaluate(synthetic_brief).to_pandas()
    b = pandas_basis.evaluate(synthetic_brief).to_pandas()
    assert (a["n"] == b["n"]).all()
    assert a["rank_ic"].to_numpy() == pytest.approx(b["rank_ic"].to_numpy(), abs=1e-12)


def test_cli_eval_prints_every_feature(features, synthetic_brief, capsys):
    code = main(["eval", str(features), "--brief", str(synthetic_brief)])
    out = capsys.readouterr().out
    assert code == 1  # one feature is rejected
    assert "basis  [accepted]" in out
    assert "peek  [rejected_lookahead]" in out
    assert "broken  [error]" in out


def test_cli_eval_by_name(features, synthetic_brief, capsys):
    assert main(["eval", str(features), "--brief", str(synthetic_brief), "--name", "basis"]) == 0
    assert "peek" not in capsys.readouterr().out
    assert main(["eval", str(features), "--brief", str(synthetic_brief), "--name", "nope"]) == 2


def test_cli_data_writes_data_md(synthetic_brief, tmp_path):
    out = tmp_path / "DATA.md"
    assert main(["data", "--brief", str(synthetic_brief), "-o", str(out)]) == 0
    text = out.read_text()
    assert "## Streams" in text and "## Cross-stream lag" in text


FAMILIES = """
import blinded as bl
from blinded import features as F


@bl.feature("the perp-spot spread reverts", family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
def perp_over_spot(panel):
    return -F.spread_z("XRPUSDT_PERP", "XRPUSDT", "5m", "5m")


@bl.feature("the same, written spot over perp", family="pairs", pair=("XRPUSDT", "XRPUSDT_PERP"))
def spot_over_perp(panel):
    return F.spread_z("XRPUSDT_PERP", "XRPUSDT", "5m", "5m")


@bl.feature("the basis, against the perp's own return", target="XRPUSDT_PERP")
def on_perp(panel):
    b = F.basis("XRPUSDT_PERP", "XRPUSDT")
    return b - F.roll_mean(b, "5m")


@bl.strategy(family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
def pair_trade(panel, features):
    return bl.pairs("XRPUSDT_PERP", "XRPUSDT", window="5m", every="5m")
"""


@pytest.fixture(scope="module")
def families(tmp_path_factory, synthetic_brief):
    root = tmp_path_factory.mktemp("families")
    raw = yaml.safe_load(synthetic_brief.read_text())
    raw["families"] = {"directional": {"instruments": ["XRPUSDT", "XRPUSDT_PERP"]},
                       "pairs": {"pairs": [["XRPUSDT_PERP", "XRPUSDT"]]}}  # fmt: skip
    (root / "f.py").write_text(FAMILIES)
    module = bl.load(root / "f.py")
    found = {o.name: o for o in [*module.features, *module.strategies]}
    return write_brief(raw, root / "brief.yaml"), found


@cross_check
def test_a_pairs_feature_is_scored_against_the_spread_return(families, synthetic_panel):
    from blinded.metrics import correlation, rank

    brief, found = families
    result = bl.evaluate(found["perp_over_spot"], brief, panel=synthetic_panel)
    assert result.ok and (result.family, result.legs) == ("pairs", ("XRPUSDT_PERP", "XRPUSDT"))
    # independently: ln m(g + L + 10 s) - ln m(g + L) for each leg, from its own events
    g = synthetic_panel.grid_ns
    latency, h = pd.Timedelta("50ms").value, pd.Timedelta("10s").value

    def move(sym):
        ev = synthetic_panel.stream_of(sym, "mid")
        return np.log(ev.asof("mid", g + latency + h) / ev.asof("mid", g + latency))

    target = move("XRPUSDT_PERP") - move("XRPUSDT")
    target[g + latency + h > g[-1]] = np.nan
    x = result.verdict.full
    inside = (g >= result.window.start.value) & (g < result.window.end.value)
    ok = inside[: len(x)] & np.isfinite(x) & np.isfinite(target[: len(x)])
    want = correlation(rank(x[ok]), rank(target[: len(x)][ok]))[0]
    assert result.to_pandas().loc["10s", "rank_ic"] == pytest.approx(want, abs=1e-12)


@known
def test_a_pair_written_the_other_way_round_scores_the_same(families, synthetic_panel):
    # spot_over_perp is -perp_over_spot declared for (spot, perp); the judge puts the pair in
    # the brief's order and negates the feature, so the two are the same claim
    brief, found = families
    a = bl.evaluate(found["perp_over_spot"], brief, panel=synthetic_panel)
    b = bl.evaluate(found["spot_over_perp"], brief, panel=synthetic_panel)
    assert b.legs == ("XRPUSDT_PERP", "XRPUSDT")
    assert a.to_pandas()["rank_ic"].to_numpy() == pytest.approx(
        b.to_pandas()["rank_ic"].to_numpy(), abs=1e-12
    )


@cross_check
def test_a_target_is_scored_as_if_it_were_the_brief_target(families, synthetic_brief):
    brief, found = families
    on_perp = found["on_perp"].evaluate(brief)
    raw = yaml.safe_load(synthetic_brief.read_text())
    raw["target"]["instrument"] = "XRPUSDT_PERP"
    plain = write_brief(raw, Path(brief).parent / "perp.yaml")
    alone = found["on_perp"].evaluate(plain)
    assert on_perp.legs == ("XRPUSDT_PERP",)
    assert on_perp.to_pandas()["rank_ic"].to_numpy() == pytest.approx(
        alone.to_pandas()["rank_ic"].to_numpy(), abs=1e-12
    )


@edge
def test_a_family_the_brief_does_not_have_is_an_error(families, synthetic_brief):
    _, found = families
    result = found["perp_over_spot"].evaluate(synthetic_brief)
    assert (result.status, result.reason) == ("error", "family pairs is not in the brief")


@known
def test_a_pairs_strategy_has_two_legs(families, synthetic_panel):
    brief, found = families
    result = found["pair_trade"].evaluate(brief, panel=synthetic_panel)
    assert result.ok, result.reason
    assert list(result.position) == ["XRPUSDT_PERP", "XRPUSDT"]
    row = result.row()
    assert row["fills"] == row["fills_XRPUSDT_PERP"] + row["fills_XRPUSDT"] > 0
    text = repr(result)
    assert "pairs on XRPUSDT_PERP/XRPUSDT" in text and "exposure" in text
