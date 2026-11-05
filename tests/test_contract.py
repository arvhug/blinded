import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml
from conftest import EXAMPLE_BRIEF

import blinded as bl
from blinded import brief as briefs
from blinded.contract import run, scan, scan_feature

known = pytest.mark.known_answer
edge = pytest.mark.edge_case
PLANTED = Path(__file__).parent / "planted"
LEAKS = [p.stem for p in PLANTED.glob("leak_*.py") if not p.stem.endswith("_pandas")]


def write(tmp_path, body, name="f.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


@known
def test_decorated_function_stays_a_plain_function():
    @bl.feature("doubles")
    def double(x):
        return 2 * x

    assert double(3) == 6
    assert (double.name, double.hypothesis, double.__name__) == ("double", "doubles", "double")


@known
def test_load_collects_every_feature_and_strategy(tmp_path):
    path = write(
        tmp_path,
        """
        import blinded as bl

        @bl.feature("one")
        def a(panel):
            return bl.features.mid("X")

        @bl.feature("two")
        def b(panel):
            return panel.col("X", "mid")

        @bl.strategy(requires=["a"])
        def follow(panel, features):
            return features["a"]
        """,
    )
    module = bl.load(path)
    assert [f.name for f in module.features] == ["a", "b"]
    assert [s.name for s in module.strategies] == ["follow"]
    assert module.strategies[0].requires == ["a"]


@known
@pytest.mark.parametrize(
    "line, message",
    [
        ("import os", "imports os"),
        ("from subprocess import run", "imports subprocess"),
        ("open('x')", "calls open()"),
        ("df.to_csv('x')", "can write data"),
        ("df.write_parquet('x')", "can write data"),
        ("time.time()", "reads the clock"),
        ("datetime.now()", "reads the clock"),
        ("eval('1')", "calls eval()"),
    ],
)
def test_scan_rejects(line, message):
    errors = scan(line).errors
    assert len(errors) == 1 and message in errors[0]


@known
@pytest.mark.parametrize(
    "line, message",
    [
        ("x.shift(-1)", "negative"),
        ("x.rolling(5, center=True)", "center=True"),
        ("pd.merge_asof(a, b, on='t', direction='forward')", "forward"),
        ("pd.merge_asof(a, b, on='t', direction='nearest')", "nearest"),
        ("x.bfill()", "bfill"),
        ("x.shift(n=-1)", "negative"),
        ("x.rolling_mean(5, center=True)", "center=True"),
        ("a.join_asof(b, on='ts', strategy='forward')", "forward"),
        ("a.join_asof(b, on='ts', strategy='nearest')", "nearest"),
        ("x.fill_null(strategy='backward')", "backward"),
        ("x.backward_fill()", "backward"),
        ("x.fillna(method='bfill')", "backward"),
        ("x - x.mean()", "whole column"),
        ("x / pl.col('y').max()", "whole column"),
        ("x.fill_null(x.median())", "whole column"),
        ("x > np.quantile(x, 0.9)", "whole column"),
        ("x.cum_sum(reverse=True)", "reverse"),
        ("x.interpolate()", "next value"),
    ],
)
def test_scan_warns_but_does_not_reject(line, message):
    result = scan(line)
    assert result.errors == []
    assert len(result.warnings) == 1 and message in result.warnings[0]


@edge
def test_harmless_code_scans_clean():
    result = scan("x.shift(1)\nx.rolling(5).mean()\npd.merge_asof(a, b, on='t')\n")
    assert (result.errors, result.warnings) == ([], [])
    polars = """
x.rolling_mean(5) - x.shift(1)
x - x.rolling(5).mean()
x - x.ewm(alpha=0.1).mean()
y = x.mean().rolling(index_column="ts", period="1m")
a.join_asof(b, on="ts")
x.fill_null(strategy="forward")
x.cum_sum()
x - df.groupby("k")["x"].mean()
df.group_by_dynamic("ts", every="1s").agg(pl.col("x").mean())
"""
    result = scan(polars)
    assert (result.errors, result.warnings) == ([], [])


@edge
@pytest.mark.parametrize("flavour", ["", "_pandas"])
@pytest.mark.parametrize("stem", sorted(set(LEAKS) - {"leak_ewma_init", "leak_resample_left"}))
def test_scan_points_at_the_planted_leaks_it_can_read(stem, flavour):
    (feat,) = bl.load(PLANTED / f"{stem}{flavour}.py").features
    assert scan_feature(feat).warnings


@edge
@pytest.mark.parametrize("flavour", ["", "_pandas"])
@pytest.mark.parametrize("stem", ["leak_ewma_init", "leak_resample_left"])
def test_scan_cannot_read_every_leak(stem, flavour):
    # a sample-mean seed and left-labelled bars look like ordinary code; the cuts decide
    (feat,) = bl.load(PLANTED / f"{stem}{flavour}.py").features
    assert scan_feature(feat).warnings == []


@edge
def test_warnings_are_scoped_to_the_feature(tmp_path):
    path = write(
        tmp_path,
        """
        import blinded as bl

        @bl.feature("clean")
        def clean(panel):
            return panel.col("X", "mid").shift(1)

        @bl.feature("leaky")
        def leaky(panel):
            return panel.col("X", "mid").shift(-1)
        """,
    )
    clean, leaky = bl.load(path).features
    from blinded.contract import scan_feature

    assert scan_feature(clean).warnings == []
    assert len(scan_feature(leaky).warnings) == 1


@edge
def test_hypothesis_is_required():
    with pytest.raises(TypeError, match="hypothesis"):

        @bl.feature
        def nameless(panel):
            return panel


@known
def test_expressions_and_polars_and_pandas_series_give_the_same_values():
    panel = bl.features.example_panel()
    expected = [10.5, 11.0, 11.5, 12.5]
    assert run(lambda p: bl.features.mid("SPOT"), panel).tolist() == expected
    assert run(lambda p: p.frame["SPOT.mid"], panel).tolist() == expected
    assert run(lambda p: p.to_pandas()["SPOT.mid"], panel).tolist() == expected
    out = run(lambda p: pd.Series(1, index=p.to_pandas().index), panel)
    assert out.dtype == "float64"


@edge
def test_output_must_be_one_value_per_grid_point():
    panel = bl.features.example_panel()
    with pytest.raises(TypeError, match="polars expression"):
        run(lambda p: p.frame["SPOT.mid"].to_numpy(), panel)
    with pytest.raises(ValueError, match="one value per grid point"):
        run(lambda p: pl.col("SPOT.mid").mean(), panel)
    with pytest.raises(ValueError, match="one value per grid point"):
        run(lambda p: p.frame["SPOT.mid"][1:], panel)
    with pytest.raises(ValueError, match="indexed by the grid"):
        run(lambda p: p.to_pandas()["SPOT.mid"].iloc[1:], panel)
    assert run(lambda p: pl.col("SPOT.mid").is_null(), panel).tolist() == [0, 0, 0, 0]


def families_brief(families=None):
    raw = yaml.safe_load(EXAMPLE_BRIEF.read_text())
    raw["data"]["path"] = "x.parquet"
    if families is not None:
        raw["families"] = families
    return briefs.parse(raw)


PAIRS = {
    "directional": {"instruments": ["XRPUSDT", "XRPUSDT_PERP"]},
    "pairs": {"pairs": [["XRPUSDT_PERP", "XRPUSDT"]]},
}


@known
def test_decorators_carry_family_target_and_pair():
    @bl.feature("spread reverts", family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
    def z(panel):
        return panel.col("XRPUSDT", "mid")

    @bl.strategy(requires=["z"], family="directional", trades="XRPUSDT_PERP")
    def follow(panel, features):
        return features["z"]

    @bl.feature("plain")
    def plain(panel):
        return panel.col("XRPUSDT", "mid")

    assert (z.family, z.target, z.pair, z.legs) == (
        "pairs", None, ("XRPUSDT_PERP", "XRPUSDT"), ("XRPUSDT_PERP", "XRPUSDT"),
    )  # fmt: skip
    assert (follow.family, follow.trades, follow.legs) == (
        "directional", "XRPUSDT_PERP", ("XRPUSDT_PERP",),
    )  # fmt: skip
    assert (plain.family, plain.legs) == ("directional", ())


@known
def test_bind_finds_the_legs_and_puts_a_pair_in_the_brief_order():
    brief = families_brief(PAIRS)

    @bl.feature("x")
    def to_target(panel):
        return None

    @bl.feature("x", target="XRPUSDT_PERP")
    def to_perp(panel):
        return None

    @bl.feature("x", family="pairs", pair=("XRPUSDT", "XRPUSDT_PERP"))
    def reversed_pair(panel):
        return None

    assert bl.contract.bind(to_target, brief) == bl.contract.Binding("directional", ("XRPUSDT",))
    assert bl.contract.bind(to_perp, brief).legs == ("XRPUSDT_PERP",)
    # declared spot over perp; the brief's pair is perp over spot, so the sign flips
    bound = bl.contract.bind(reversed_pair, brief)
    assert (bound.legs, bound.sign, bound.label) == (
        ("XRPUSDT_PERP", "XRPUSDT"), -1.0, "XRPUSDT_PERP/XRPUSDT",
    )  # fmt: skip


@edge
@pytest.mark.parametrize(
    "kwargs, families, message",
    [
        ({"family": "pairs", "pair": ("XRPUSDT_PERP", "XRPUSDT")}, None,
         "family pairs is not in the brief"),
        ({"target": "BTCUSDT"}, PAIRS, "BTCUSDT is not one of the directional family's"),
        ({"family": "pairs"}, PAIRS, "needs pair="),
        ({"pair": ("XRPUSDT_PERP", "XRPUSDT")}, PAIRS, "pair= is for the pairs family"),
        ({"family": "pairs", "pair": ("XRPUSDT", "BTCUSDT")}, PAIRS, "not one of the brief's"),
    ],
)  # fmt: skip
def test_bind_refuses_what_the_brief_does_not_have(kwargs, families, message):
    @bl.feature("x", **kwargs)
    def f(panel):
        return None

    with pytest.raises(bl.contract.ContractError, match=message):
        bl.contract.bind(f, families_brief(families))


@known
def test_a_mapping_output_is_legs_checked_against_the_strategy_legs():
    from blinded.contract import ContractError, Legs, values
    from blinded.features import example_panel

    p = example_panel()
    spot, perp = pl.col("SPOT.mid"), pl.col("PERP.mid")
    legs = values({"PERP": perp, "SPOT": -spot}, p, ("SPOT", "PERP"))
    # in the order of the legs, whatever the order of the mapping
    assert isinstance(legs, Legs) and list(legs) == ["SPOT", "PERP"]
    assert legs["SPOT"].tolist() == [-10.5, -11.0, -11.5, -12.5]
    # one leg: a one-key mapping and a single series give the same array
    one = values({"SPOT": spot}, p, ("SPOT",))
    assert isinstance(one, np.ndarray) and one.tolist() == values(spot, p, ("SPOT",)).tolist()
    for out, legs in (({"SPOT": spot}, ("SPOT", "PERP")), (spot, ("SPOT", "PERP")),
                      ({"PERP": perp}, ("SPOT",))):  # fmt: skip
        with pytest.raises(ContractError, match="returns"):
            values(out, p, legs)
    assert isinstance(values({"A": spot}, p), Legs)  # unchecked without legs
