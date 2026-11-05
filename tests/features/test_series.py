import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from blinded.features import series as S

nan = np.nan
known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case
X, Y = pl.col("x"), pl.col("y")


def s(*values, freq="1s"):
    """A series on a regular 1 s index, so time windows and halflives work."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def at(seconds, values):
    """A series on an irregular index, ``seconds`` after midnight."""
    idx = pd.Timestamp("2024-01-01") + pd.to_timedelta(seconds, unit="s")
    return pd.Series(values, index=idx, dtype=float)


def ev(expr, x, y=None):
    """Evaluate ``expr`` on a frame holding ``x`` (and ``y``) with their index as ``ts``.

    NaN in the inputs becomes null, which is how the panel marks a missing value.
    """
    idx = x.index
    ts = pl.Series("ts", idx.as_unit("ns").asi8).cast(pl.Datetime("ns"))
    if idx.tz is not None:
        ts = ts.dt.replace_time_zone("UTC").dt.convert_time_zone(str(idx.tz))
    data = {"ts": ts.set_sorted(), "x": pl.Series(x.to_numpy(), nan_to_null=True)}
    if y is not None:
        data["y"] = pl.Series(y.to_numpy(), nan_to_null=True)
    return pl.DataFrame(data).select(expr.cast(pl.Float64)).to_series().to_numpy()


def same(result, expected, rtol=1e-9, atol=1e-12):
    assert_allclose(
        np.asarray(result, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


def random_walk(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return s(*(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))))


# known answers


@known
def test_lag_counts_points():
    same(ev(S.lag(X, 2), s(5, 7, 9, 11)), [nan, nan, 5, 7])


@known
def test_lag_by_time_takes_last_value_at_or_before():
    x = at([0, 0.5, 1.2, 2.0], [1, 2, 3, 4])
    # t=1.2 looks up 0.2 -> the value at 0; t=2.0 looks up 1.0 -> the value at 0.5
    same(ev(S.lag(X, "1s"), x), [nan, nan, 1, 2])


@known
def test_diff():
    same(ev(S.diff(X, 2), s(2, 3, 7, 6)), [nan, nan, 5, 3])  # 7-2, 6-3


@known
def test_ret_simple_and_log():
    same(ev(S.ret(X, log=False), s(50, 100, 75)), [nan, 1.0, -0.25])
    same(ev(S.ret(X), s(1, np.e, np.e**3)), [nan, 1.0, 2.0])


@known
def test_roll_mean_counts_and_time():
    same(ev(S.roll_mean(X, 3), s(1, 2, 3, 4, 5)), [nan, nan, 2, 3, 4])
    x = at([0, 1, 1.5, 4], [10, 20, 30, 40])
    # (t-2, t]: {10}, {10,20}, {10,20,30}, {40}
    same(ev(S.roll_mean(X, "2s"), x), [10, 15, 20, 40])


@known
def test_roll_std_sample():
    # mean 5, squared deviations sum to 32, divided by n-1 = 7
    out = ev(S.roll_std(X, 8), s(2, 4, 4, 4, 5, 5, 7, 9))
    same(out, [nan] * 7 + [np.sqrt(32 / 7)])


@known
def test_roll_sum():
    same(ev(S.roll_sum(X, 2), s(1, -1, 2, 5)), [nan, 0, 1, 7])


@known
def test_roll_min_max():
    x = s(4, 2, 6, 1, 3)
    same(ev(S.roll_min(X, 3), x), [nan, nan, 2, 1, 1])
    same(ev(S.roll_max(X, 3), x), [nan, nan, 6, 6, 6])


@known
def test_roll_median_even_window():
    # (2+4)/2 and (2+3)/2
    same(ev(S.roll_median(X, 4), s(5, 1, 4, 2, 3)), [nan, nan, nan, 3.0, 2.5])


@known
def test_roll_quantile_interpolates():
    x = s(10, 20, 30, 40)
    same(ev(S.roll_quantile(X, 4, 0.5), x), [nan, nan, nan, 25])  # position 1.5
    same(ev(S.roll_quantile(X, 4, 0.9), x), [nan, nan, nan, 37])  # position 2.7: 30 + 0.7 * 10


@known
def test_roll_rank_ties_share_average_rank():
    # [4,1,3]: 3 ranks 2 of 3; [1,3,3]: the 3s share rank 2.5 of 3
    same(ev(S.roll_rank(X, 3), s(4, 1, 3, 3)), [nan, nan, 2 / 3, 2.5 / 3])


@known
def test_roll_skew():
    # [0,0,3]: m2 = 2, m3 = 2, g1 = 2 / 2**1.5, G1 = sqrt(3*2)/1 * g1 = sqrt(3)
    same(ev(S.roll_skew(X, 3), s(0, 0, 3)), [nan, nan, np.sqrt(3)])


@known
def test_roll_kurt():
    # [0,0,0,4]: m2 = 3, m4 = 21, g2 = 21/9 - 3, G2 = ((n+1) g2 + 6)(n-1)/((n-2)(n-3)) = 4
    same(ev(S.roll_kurt(X, 4), s(0, 0, 0, 4)), [nan, nan, nan, 4.0])


@known
def test_roll_corr():
    # deviations (-1.5,-.5,.5,1.5) and (-1.5,.5,-.5,1.5): sum xy 4, sum xx = sum yy = 5
    same(ev(S.roll_corr(X, Y, 4), s(1, 2, 3, 4), s(1, 3, 2, 4)), [nan, nan, nan, 0.8])


@known
def test_roll_beta():
    x = s(1, 2, 3, 4)
    same(ev(S.roll_beta(X, Y, 4), s(1, 3, 2, 4), x), [nan, nan, nan, 0.8])  # cov 4/3 over var 5/3
    same(ev(S.roll_beta(X, Y, 2), -3 * x, x), [nan, -3, -3, -3])


@known
def test_zscore():
    same(ev(S.zscore(X, 3), s(2, 4, 6)), [nan, nan, 1.0])  # mean 4, sd 2


@known
def test_meanrev_is_negated_zscore():
    same(ev(S.meanrev(X, 3), s(6, 4, 2)), [nan, nan, 1.0])


@known
def test_expanding_zscore():
    # [2,4]: (4-3)/sqrt(2); [2,4,6]: (6-4)/2; [2,4,6,8]: (8-5)/sqrt(20/3)
    same(ev(S.expanding_zscore(X), s(2, 4, 6, 8)), [nan, 1 / np.sqrt(2), 1.0, 3 / np.sqrt(20 / 3)])


@known
def test_ewm_mean_halflife_one_point():
    # alpha = 0.5: 4, (4+0)/2, (2+0)/2, (1+8)/2
    same(ev(S.ewm_mean(X, 1), s(4, 0, 0, 8)), [4, 2, 1, 4.5])
    same(ev(S.ewm_mean(X, "1s"), s(4, 0, 0, 8)), [4, 2, 1, 4.5])


@known
def test_ewm_std():
    # alpha 0.5: v1 = 0.5 * (0 + 0.5 * 2**2) = 1; v2 = 0.5 * (1 + 0.5 * (2 - 1)**2) = 0.75
    same(ev(S.ewm_std(X, 1), s(0, 2, 2)), [0, 1, np.sqrt(0.75)])


@known
def test_ewm_zscore():
    # means 0, 1, 1.5 and deviations 0, 1, sqrt(.75)
    same(ev(S.ewm_zscore(X, 1), s(0, 2, 2)), [nan, 1.0, 0.5 / np.sqrt(0.75)])


@known
def test_ewm_cross():
    # fast alpha 0.5 -> 1; slow alpha 1 - 2**-0.5 -> 2 * (1 - 2**-0.5); difference sqrt(2) - 1
    same(ev(S.ewm_cross(X, 1, 2), s(0, 2)), [0, np.sqrt(2) - 1])


@known
def test_realized_vol():
    x = s(*np.exp(np.cumsum([0, 0.1, -0.2, 0.2])))
    same(ev(S.realized_vol(X, 3), x), [nan, nan, nan, 0.3])  # sqrt(.01 + .04 + .04)


@known
def test_momentum():
    x = s(*np.exp(np.cumsum([0, 0.1, -0.2, 0.2])))
    same(ev(S.momentum(X, 3), x), [nan, nan, nan, 0.1 / 0.3])


@known
def test_rsi_wilder():
    # gains 1,1,0,2 and losses 0,0,1,0; seed G=1, L=0 -> 100; then G=.5, L=.5 -> 50;
    # then G = .5 + (2 - .5)/2 = 1.25, L = .25 -> 100 * 1.25 / 1.5
    same(ev(S.rsi(X, 2), s(10, 11, 12, 11, 13)), [nan, nan, 100, 50, 250 / 3])


@known
def test_bollinger_b():
    same(ev(S.bollinger_b(X, 3), s(2, 4, 6)), [nan, nan, 0.75])  # (6 - 4 + 2*2) / (2*2*2)


@known
def test_roll_winsorize_uses_trailing_quantiles():
    # window [1..5]: q.2 = 1.8, q.8 = 4.2 -> 5 clips to 4.2
    # window [2,3,4,5,50]: q.8 at position 3.2 = 5 + .2 * 45 = 14 -> 50 clips to 14
    same(ev(S.roll_winsorize(X, 5, 0.2), s(1, 2, 3, 4, 5, 50)), [nan] * 4 + [4.2, 14])


@known
def test_time_of_day():
    idx = pd.DatetimeIndex(["2024-01-01 23:59:30", "2024-01-02 00:15"], tz="UTC")
    same(ev(S.time_of_day(), pd.Series(0.0, index=idx)), [23 + 59.5 / 60, 0.25])


# cross-checks against independent implementations


@cross
def test_ewm_mean_and_std_match_recursion():
    x = random_walk(400)
    a = 1 - 2 ** (-1 / 7.3)
    m, v = x.iloc[0], 0.0
    means, stds = [m], [0.0]
    for value in x.iloc[1:]:
        d = value - m
        m += a * d
        v = (1 - a) * (v + a * d * d)
        means.append(m)
        stds.append(np.sqrt(v))
    same(ev(S.ewm_mean(X, 7.3), x), means, rtol=1e-10)
    same(ev(S.ewm_std(X, 7.3), x), stds, rtol=1e-8)
    expected = (x - means) / np.where(np.array(stds) > 0, stds, nan)
    same(ev(S.ewm_zscore(X, 7.3), x), expected, rtol=1e-8)


@cross
def test_time_halflife_on_a_regular_grid_equals_the_same_number_of_rows():
    # ewm_std by time runs West's recursion through ewm_mean_by; by rows it is polars' ewm_std
    x = random_walk(400, seed=12)
    for fn in (S.ewm_mean, S.ewm_std, S.ewm_zscore):
        same(ev(fn(X, "7s"), x), ev(fn(X, 7), x), rtol=1e-12)


@cross
def test_time_window_skew_and_kurt_match_scipy_on_irregular_times():
    rng = np.random.default_rng(13)
    t = np.cumsum(rng.exponential(0.3, 150))
    x = at(t, rng.standard_t(4, 150))
    v = x.to_numpy()
    windows = [v[(t > ti - 3.0) & (t <= ti)] for ti in t]
    skew = [stats.skew(w, bias=False) if len(w) >= 3 else nan for w in windows]
    kurt = [stats.kurtosis(w, bias=False) if len(w) >= 4 else nan for w in windows]
    same(ev(S.roll_skew(X, "3s"), x), skew, rtol=1e-9)
    same(ev(S.roll_kurt(X, "3s"), x), kurt, rtol=1e-9)


@cross
def test_rsi_matches_textbook_loop():
    x = random_walk(200, seed=3)
    n = 14
    d = np.diff(x.to_numpy())
    gain, loss = np.clip(d, 0, None), np.clip(-d, 0, None)
    g, lo = gain[:n].mean(), loss[:n].mean()
    expected = [nan] * n + [100 * g / (g + lo)]
    for k in range(n, len(d)):
        g = (g * (n - 1) + gain[k]) / n
        lo = (lo * (n - 1) + loss[k]) / n
        expected.append(100 * g / (g + lo))
    same(ev(S.rsi(X, n), x), expected, rtol=1e-9)


@cross
def test_realized_vol_and_momentum_match_numpy():
    x = random_walk(120, seed=4)
    lr = np.diff(np.log(x.to_numpy()))
    w = 20
    vol = [nan] * w + [np.sqrt(np.sum(lr[k - w : k] ** 2)) for k in range(w, len(lr) + 1)]
    same(ev(S.realized_vol(X, w), x), vol)
    logx = np.log(x.to_numpy())
    mom = [nan] * w + [(logx[k] - logx[k - w]) / vol[k] for k in range(w, len(x))]
    same(ev(S.momentum(X, w), x), mom)


def window_loop(x, w, fn):
    v = x.to_numpy()
    return [nan] * (w - 1) + [fn(v[k - w + 1 : k + 1]) for k in range(w - 1, len(v))]


@cross
def test_roll_rank_matches_loop():
    x = s(*np.random.default_rng(5).integers(0, 6, 80))
    expected = window_loop(x, 9, lambda win: stats.rankdata(win)[-1] / len(win))
    same(ev(S.roll_rank(X, 9), x), expected)


@cross
def test_roll_quantile_matches_numpy():
    x = random_walk(150, seed=6)
    for q in (0.1, 0.5, 0.93):
        same(
            ev(S.roll_quantile(X, 17, q), x),
            window_loop(x, 17, lambda win, q=q: np.quantile(win, q)),
        )


@cross
def test_roll_skew_and_kurt_match_scipy():
    x = s(*np.random.default_rng(7).standard_t(4, 150))
    same(
        ev(S.roll_skew(X, 25), x),
        window_loop(x, 25, lambda w: stats.skew(w, bias=False)),
        rtol=1e-6,
    )
    expected = window_loop(x, 25, lambda w: stats.kurtosis(w, bias=False))
    same(ev(S.roll_kurt(X, 25), x), expected, rtol=1e-6)


@cross
def test_roll_beta_matches_lstsq():
    rng = np.random.default_rng(8)
    x = s(*rng.normal(size=100))
    y = 0.7 * x + rng.normal(scale=0.5, size=100)
    v, u = x.to_numpy(), y.to_numpy()
    expected = [nan] * 29
    for k in range(29, 100):
        design = np.column_stack([np.ones(30), v[k - 29 : k + 1]])
        expected.append(np.linalg.lstsq(design, u[k - 29 : k + 1], rcond=None)[0][1])
    same(ev(S.roll_beta(X, Y, 30), y, x), expected, rtol=1e-8)
    corr = [nan] * 29 + [
        np.corrcoef(v[k - 29 : k + 1], u[k - 29 : k + 1])[0, 1] for k in range(29, 100)
    ]
    same(ev(S.roll_corr(X, Y, 30), x, y), corr, rtol=1e-8)


@cross
def test_zscore_and_bollinger_match_loop():
    x = random_walk(90, seed=9)
    z = window_loop(x, 12, lambda w: (w[-1] - w.mean()) / w.std(ddof=1))
    same(ev(S.zscore(X, 12), x), z, rtol=1e-8)
    b = window_loop(x, 12, lambda w: (w[-1] - w.mean() + 2 * w.std(ddof=1)) / (4 * w.std(ddof=1)))
    same(ev(S.bollinger_b(X, 12), x), b, rtol=1e-8)


@cross
def test_time_window_mean_matches_loop_on_irregular_times():
    rng = np.random.default_rng(10)
    t = np.cumsum(rng.exponential(0.3, 200))
    x = at(t, rng.normal(size=200))
    v = x.to_numpy()
    expected = [v[(t > ti - 2.0) & (t <= ti)].mean() for ti in t]
    same(ev(S.roll_mean(X, "2s"), x), expected, rtol=1e-9)


@cross
def test_time_lag_matches_loop_on_irregular_times():
    rng = np.random.default_rng(11)
    t = np.cumsum(rng.exponential(0.3, 100))
    x = at(t, rng.normal(size=100))
    v = x.to_numpy()
    expected = []
    for ti in t:
        before = np.nonzero(t <= ti - 1.0)[0]
        expected.append(v[before[-1]] if len(before) else nan)
    same(ev(S.lag(X, "1s"), x), expected)


# edge cases


@edge
@pytest.mark.parametrize(
    "fn", [S.roll_mean, S.roll_std, S.roll_sum, S.roll_min, S.roll_max, S.roll_median, S.zscore]
)
def test_integer_window_warm_up_is_window_minus_one(fn):
    out = ev(fn(X, 7), random_walk(30))
    assert np.isnan(out[:6]).all()
    assert not np.isnan(out[6:]).any()


@edge
def test_window_longer_than_data_is_all_nan():
    x = s(1, 2, 3)
    assert np.isnan(ev(S.roll_mean(X, 5), x)).all()
    assert np.isnan(ev(S.zscore(X, 5), x)).all()
    assert np.isnan(ev(S.realized_vol(X, 5), x)).all()


@edge
def test_time_window_emits_from_the_first_point():
    same(ev(S.roll_mean(X, "10s"), s(1, 2, 3)), [1, 1.5, 2])


@edge
def test_constant_series_gives_nan_not_inf():
    x = s(3, 3, 3, 3, 3)
    for out in (
        ev(S.zscore(X, 3), x),
        ev(S.meanrev(X, 3), x),
        ev(S.bollinger_b(X, 3), x),
        ev(S.expanding_zscore(X), x),
        ev(S.ewm_zscore(X, 2), x),
        ev(S.roll_beta(X, Y, 3), s(1, 2, 3, 4, 5), x),
        ev(S.rsi(X, 2), x),
    ):
        assert not np.isinf(out).any()
        assert np.isnan(out).all()


@edge
def test_negative_lag_and_empty_window_raise():
    with pytest.raises(ValueError, match="future"):
        S.lag(X, -1)
    with pytest.raises(ValueError, match="future"):
        S.ret(X, "-1s")
    with pytest.raises(ValueError, match="at least 1"):
        S.roll_mean(X, 0)
    with pytest.raises(TypeError, match="number of rows"):
        S.rsi(X, "14s")


@edge
def test_time_halflife_decays_by_elapsed_time_on_irregular_times():
    # alpha is 1 - 2**-dt: 0.5 over the 1 s step, then 0.75 over the 2 s step
    same(ev(S.ewm_mean(X, "1s"), at([0, 1, 3], [0, 4, 4])), [0, 2, 3.5])
    # a number of rows ignores the times: alpha 0.5 at every step
    same(ev(S.ewm_mean(X, 1), at([0, 1, 3], [0, 4, 4])), [0, 2, 3])


@edge
def test_ewm_starts_at_first_valid_point():
    same(ev(S.ewm_mean(X, 1), s(nan, 4, 0)), [nan, 4, 2])


@edge
def test_rsi_with_no_moves_is_nan():
    assert np.isnan(ev(S.rsi(X, 2), s(5, 5, 5, 5))).all()


@edge
def test_winsorize_is_nan_during_warm_up():
    assert np.isnan(ev(S.roll_winsorize(X, 5), s(1, 2, 3))).all()


@edge
def test_tied_timestamps_lose_a_lag_but_never_read_ahead():
    # polars time windows count a tied later row, so the first of the tie at 1 s sees null
    # rather than the value at 0 s; nothing ever comes from a later row
    same(ev(S.lag(X, "1s"), at([0, 1, 1, 2], [1, 2, 3, 4])), [nan, nan, 1, 3])
