"""Cross-sectional features, by hand on three legs, and one evaluation for a cross-section."""

import numpy as np
import polars as pl
import pytest

from blinded import contract
from blinded import features as F
from blinded.features.xs import Cross

known = pytest.mark.known_answer
cross_check = pytest.mark.cross_check
edge = pytest.mark.edge_case

ROWS = pl.DataFrame({"a": [1.0, 3.0, 1.0], "b": [2.0, 1.0, None], "c": [3.0, 2.0, float("nan")]})
LEGS = {k: pl.col(k) for k in "abc"}


def rows(d):
    return ROWS.select(**d).rows()


@known
def test_zscore_rank_mean_and_demean_by_hand():
    # row 0 is (1, 2, 3): mean 2, sample SD 1, so z (-1, 0, 1); row 1 is (3, 1, 2): ranks
    # (3, 1, 2); row 2 has one value, a, as b is null and c NaN
    z = rows(F.xs_zscore(LEGS))
    assert z[0] == (-1.0, 0.0, 1.0)
    assert z[1] == (1.0, -1.0, 0.0)
    assert z[2] == (None, None, None)  # one value: no spread
    assert rows(F.xs_rank(LEGS))[:2] == [(1.0, 2.0, 3.0), (3.0, 1.0, 2.0)]
    assert rows(F.xs_rank(LEGS))[2] == (1.0, None, None)
    assert rows(F.xs_mean(LEGS)) == [(2.0, 2.0, 2.0), (2.0, 2.0, 2.0), (1.0, 1.0, 1.0)]
    assert rows(F.xs_demean(LEGS))[:2] == [(-1.0, 0.0, 1.0), (1.0, -1.0, 0.0)]


@edge
def test_equal_values_have_no_zscore_and_share_a_rank():
    flat = pl.DataFrame({"a": [5.0], "b": [5.0], "c": [5.0]})
    assert flat.select(**F.xs_zscore(LEGS)).row(0) == (None, None, None)
    assert flat.select(**F.xs_rank(LEGS)).row(0) == (2.0, 2.0, 2.0)


@cross_check
def test_nested_cross_sections_equal_their_numpy_arithmetic():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 6))
    frame = pl.DataFrame({f"s{i}": x[:, i] for i in range(6)})
    d = {f"s{i}": pl.col(f"s{i}") for i in range(6)}
    nested = frame.select(**F.xs_zscore(F.xs_demean(d))).to_numpy()
    expected = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)
    np.testing.assert_allclose(nested, expected, rtol=1e-12, atol=1e-12)


@cross_check
def test_a_cross_section_is_evaluated_once_and_split_as_leg_by_leg(synthetic_panel):
    legs = ["XRPUSDT", "XRPUSDT_PERP"]
    d = F.each(legs, lambda s: F.ret(F.mid(s), "10s"))
    d["LAGGED"] = F.lag(F.mid("XRPUSDT"), "1s").log()
    z = F.xs_zscore(d)
    assert isinstance(z, Cross)
    once = contract.values(z, synthetic_panel)
    for leg, expr in z.items():
        np.testing.assert_array_equal(once[leg], contract.values(expr, synthetic_panel))


@known
def test_each_applies_a_function_of_a_symbol_to_every_leg():
    d = F.each(["a", "b"], lambda s: pl.col(s) * 2)
    assert list(d) == ["a", "b"]
    assert ROWS.select(**d).row(0) == (2.0, 4.0)
