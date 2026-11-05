import numpy as np
import pandas as pd
import polars as pl
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import stats

from blinded import metrics as M

known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case


def bivariate(n, rho, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)


@known
def test_correlation_and_its_influence_function():
    r, infl = M.correlation(np.array([1.0, 2, 3, 4]), np.array([1.0, 3, 2, 4]))
    assert r == pytest.approx(0.8)  # sum xy 4 over sqrt(5 * 5)
    assert infl.mean() == pytest.approx(0, abs=1e-15)


@known
def test_rank_averages_ties():
    assert M.rank(np.array([10.0, 20, 20, 30])).tolist() == [1, 2.5, 2.5, 4]


@known
def test_nw_lags_is_overlap_plus_cube_root_term():
    assert M.nw_lags(100, 107_800) == 116  # 100 + ceil(1.5 * 1078 ** (1/3)) = 100 + 16
    assert M.nw_lags(1, 1000) == 16  # 1 + ceil(1.5 * 10)
    assert M.nw_lags(0.5, 1000) == 16  # a horizon shorter than a step still overlaps once


@known
def test_nw_se_by_hand():
    # S = sum u^2 + 2 * (1 - 1/2) * sum u_t u_{t-1} = 4 + (-3) = 1; se = sqrt(S) / n
    assert M.nw_se(np.array([1.0, -1, 1, -1]), lags=1) == pytest.approx(0.25)


@known
def test_markout_overall_and_in_the_tails():
    f = np.array([1.0, -1, 2, -2])
    r = np.array([0.001, 0.002, -0.001, -0.003])
    out = M.markout(f, r)
    # sign(f) * r = .001, -.002, -.001, .003 -> mean .00025
    assert out["markout_bps"] == pytest.approx(2.5)
    # top decile is {2} with r = -.001; bottom decile is {-2} with -r = .003
    assert out["top_bps"] == pytest.approx(-10)
    assert out["bottom_bps"] == pytest.approx(30)
    assert out["tails_bps"] == pytest.approx(10)


@known
def test_turnover_and_autocorr():
    assert M.turnover(np.array([1.0, -1, -1, 1])) == pytest.approx(4 / 3)  # |dsign| = 2, 0, 2
    assert M.autocorr(np.array([1.0, 2, 3, 4])) == pytest.approx(1)


@known
def test_stability_splits_on_the_clock():
    idx = pd.date_range("2024-01-01 00:00", periods=8, freq="10min", tz="UTC")
    f = np.array([1.0, 2, 3, 1, 2, 3, 9, 9])
    r = np.array([1.0, 2, 3, 3, 2, 1, 0, 0])
    out = M.stability(pl.Series(idx), f, r, every="30min", min_n=3)
    # 00:00-00:20 rises together, 00:30-00:50 moves opposite, 01:00-01:10 has only 2 points
    assert out["start"].to_list() == [idx[0], idx[3]]
    assert out["rank_ic"].to_list() == pytest.approx([1.0, -1.0])


@cross
def test_ic_and_rank_ic_match_scipy():
    x, y = bivariate(2000, 0.3, seed=1)
    y = y + 0.5 * y**3
    out = M.ic_summary(x, y, h_steps=1, block=50, n_boot=100, seed=0)
    assert out["ic"] == pytest.approx(stats.pearsonr(x, y)[0], rel=1e-10)
    assert out["rank_ic"] == pytest.approx(stats.spearmanr(x, y)[0], rel=1e-10)


@cross
def test_nw_se_matches_statsmodels_hac_and_a_loop():
    rng = np.random.default_rng(2)
    u = np.convolve(rng.normal(size=3000), np.ones(10), mode="valid")
    u = u - u.mean()
    lags = 15
    fit = sm.OLS(u, np.ones(len(u))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )
    lrv = np.mean(u * u)
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.mean(u[k:] * u[:-k]) * (len(u) - k) / len(u)
    assert M.nw_se(u, lags) == pytest.approx(fit.bse[0], rel=1e-10)
    assert M.nw_se(u, lags) == pytest.approx(np.sqrt(lrv / len(u)), rel=1e-10)


@cross
def test_iid_standard_errors_match_the_analytic_one():
    n, rho = 5000, 0.3
    x, y = bivariate(n, rho, seed=3)
    analytic = (1 - rho**2) / np.sqrt(n)  # asymptotic sd of r for bivariate normal data
    out = M.ic_summary(x, y, h_steps=1, block=50, n_boot=2000, seed=4)
    assert out["ic_se_nw"] == pytest.approx(analytic, rel=0.10)
    assert out["ic_se_boot"] == pytest.approx(analytic, rel=0.15)


@cross
def test_overlap_with_a_persistent_signal_inflates_both_errors_alike():
    # trailing 100-step mean as the signal, the next 100 steps summed as the target:
    # independent, but both persistent, so var(r) is about sum_k rho_f(k) rho_r(k) / n
    rng = np.random.default_rng(5)
    z = rng.normal(size=20_200)
    signal = np.convolve(z, np.ones(100), mode="valid")[:20_000]
    target = np.convolve(z, np.ones(100), mode="valid")[101:20_101]
    out = M.ic_summary(signal, target, h_steps=100, block=1000, n_boot=1000, seed=6)
    naive = 1 / np.sqrt(len(signal))
    assert out["ic_se_nw"] > 3 * naive
    assert out["ic_se_boot"] == pytest.approx(out["ic_se_nw"], rel=0.3)


@cross
def test_overlap_alone_does_not_inflate_the_error():
    # an iid signal against overlapping targets: the influence function stays uncorrelated
    rng = np.random.default_rng(5)
    e = rng.normal(size=20_100)
    target = np.convolve(e, np.ones(100), mode="valid")[:20_000]
    signal = e[:20_000] + rng.normal(scale=3, size=20_000)
    out = M.ic_summary(signal, target, h_steps=100, block=1000, n_boot=500, seed=6)
    assert out["ic_se_nw"] == pytest.approx(1 / np.sqrt(len(signal)), rel=0.15)


@cross
def test_bootstrap_is_reproducible_with_a_seed():
    x, y = bivariate(1000, 0.2, seed=7)
    a = M.bootstrap_corr(x, y, block=20, n_boot=50, seed=11)
    b = M.bootstrap_corr(x, y, block=20, n_boot=50, seed=11)
    c = M.bootstrap_corr(x, y, block=20, n_boot=50, seed=12)
    assert_allclose(a, b)
    assert not np.allclose(a, c)


@cross
def test_bootstrap_with_whole_sample_block_resamples_rotations():
    # one block of length n is a rotation of the sample, so every replicate equals r
    x, y = bivariate(300, 0.5, seed=8)
    reps = M.bootstrap_corr(x, y, block=300, n_boot=20, seed=0)
    assert_allclose(reps, np.corrcoef(x, y)[0, 1], rtol=1e-10)


@edge
def test_constant_signal_gives_nan_without_raising():
    out = M.ic_summary(np.ones(500), np.arange(500.0), h_steps=1, block=10, n_boot=20, seed=0)
    assert np.isnan(out["ic"]) and np.isnan(out["ic_se_nw"]) and np.isnan(out["ic_se_boot"])


@edge
def test_block_longer_than_the_sample_is_clipped():
    x, y = bivariate(50, 0.5, seed=9)
    assert np.isfinite(M.bootstrap_corr(x, y, block=500, n_boot=5, seed=0)).all()
