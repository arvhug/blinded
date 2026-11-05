import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_allclose

from blinded import Panel
from blinded.features import cross

nan = np.nan
known = pytest.mark.known_answer
cross_check = pytest.mark.cross_check
edge = pytest.mark.edge_case
T0 = pd.Timestamp("2024-01-01", tz="UTC")


def same(result, expected, rtol=1e-12):
    assert_allclose(np.asarray(result, dtype=float), expected, rtol=rtol, equal_nan=True)


def quotes(bids, asks, seconds):
    idx = pd.DatetimeIndex([T0 + pd.Timedelta(seconds=x) for x in seconds])
    n = len(bids)
    return pd.DataFrame(
        {"bid": bids, "ask": asks, "bid_qty": [1.0] * n, "ask_qty": [1.0] * n}, index=idx
    )


@pytest.fixture
def panel():
    """Grid 1..4 s. A mids 101, 102, 102, 104 (spread 2); B mids 100, 100, 102, 104 (spread 1)."""
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=4, freq="1s")
    a = quotes([100.0, 101.0, 103.0], [102.0, 103.0, 105.0], [0.5, 1.5, 3.5])
    b = quotes([99.5, 101.5, 103.5], [100.5, 102.5, 104.5], [0.5, 2.5, 3.5])
    return Panel.from_pandas(grid, book={"A": a, "B": b})


@known
def test_basis(panel):
    same(panel.eval(cross.basis("A", "B")), [0.01, 0.02, 0, 0])  # 101/100 - 1, 102/100 - 1
    same(panel.eval(cross.basis("A", "B", log=True)), np.log([1.01, 1.02, 1, 1]))


@known
def test_basis_z_trailing_window(panel):
    # basis 0.01, 0.02, 0: window 3 at t=3 has mean 0.01, sd 0.01 -> (0 - 0.01)/0.01
    # at t=4: 0.02, 0, 0 has mean 0.02/3 and sd 0.02/sqrt(3)
    expected = [nan, nan, -1.0, (0 - 0.02 / 3) / (0.02 / np.sqrt(3))]
    same(panel.eval(cross.basis_z("A", "B", window=3)), expected, rtol=1e-9)


@known
def test_basis_z_ewm(panel):
    # halflife 1: means 0.01, 0.015; deviation sqrt(0.5 * 0.5 * 0.01**2) = 0.005
    out = panel.eval(cross.basis_z("A", "B", halflife=1))
    same(out[:2], [nan, (0.02 - 0.015) / 0.005], rtol=1e-9)


@known
def test_lead_ret(panel):
    same(panel.eval(cross.lead_ret("B")), [nan, 0, np.log(1.02), np.log(104 / 102)])
    same(panel.eval(cross.lead_ret("B", 2, log=False)), [nan, nan, 0.02, 0.04])


@known
def test_lead_gap_on_the_example_panel():
    from blinded.features import example_panel

    # PERP mids 11, 11, 12.25, 12.25 lead SPOT mids 10.5, 11, 11.5, 12.5
    perp = [nan, 0.0, np.log(12.25 / 11), 0.0]
    spot = [nan, np.log(11 / 10.5), np.log(11.5 / 11), np.log(12.5 / 11.5)]
    gap = np.subtract(perp, spot)
    same(example_panel().eval(cross.lead_gap("PERP", "SPOT")), gap)
    # over two rows: row 2 is ln(12.25/11) - ln(11.5/10.5), row 3 ln(12.25/11) - ln(12.5/11)
    two = [nan, nan, np.log(12.25 / 11) - np.log(11.5 / 10.5), np.log(12.25 / 12.5)]
    same(example_panel().eval(cross.lead_gap("PERP", "SPOT", 2)), two)
    # the z divides by the sample sd of the last two gaps, |g1 - g0| / sqrt(2)
    sd = [nan, nan, abs(gap[2] - gap[1]) / np.sqrt(2), abs(gap[3] - gap[2]) / np.sqrt(2)]
    same(example_panel().eval(cross.lead_gap_z("PERP", "SPOT", 1, 2)), gap / np.array(sd))


def logged(returns, start=100.0):
    """Mids whose one-step log returns are ``returns``, quoted 0.01 either side."""
    mids = start * np.exp(np.cumsum(np.r_[0.0, returns]))
    seconds = np.arange(len(mids)) + 0.5
    return quotes(mids - 0.01, mids + 0.01, seconds)


@known
def test_lagged_corr_over_three_rows():
    c = 1e-3
    # grid 1..5 s. The leader returns c, 2c, 3c, 0 at rows 1-4, so a row later it reads
    # c, 2c, 3c at rows 2-4. The follower returns 5c, c, 3c, 2c at rows 1-4.
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=5, freq="1s")
    lead = logged([c, 2 * c, 3 * c, 0.0])
    follow = logged([5 * c, c, 3 * c, 2 * c])
    p = Panel.from_pandas(grid, book={"L": lead, "F": follow})
    # at row 4 the window holds rows 2-4: x = (1, 2, 3)c, y = (1, 3, 2)c. Deviations from the
    # means (2c each) are (-1, 0, 1) and (-1, 1, 0): products sum to 1, squares to 2 and 2,
    # so the correlation is 1 / sqrt(2 * 2) = 0.5. Row 3 has only two pairs: null.
    same(p.eval(cross.lagged_corr("L", "F", 1, 3)), [nan, nan, nan, nan, 0.5], rtol=1e-9)


@known
def test_rel_spread(panel):
    # A spread 2 on mids 101, 102, 102, 104; B spread 1 on mids 100, 100, 102, 104
    same(panel.eval(cross.rel_spread("A", "B")), [2 / 101 * 100, 2 / 102 * 100, 2, 2])


@known
def test_ret_corr_and_beta_two_point_windows(panel):
    # A steps: ln(102/101), 0, ln(104/102); B steps: 0, ln(1.02), ln(104/102)
    ra = [np.log(102 / 101), 0.0, np.log(104 / 102)]
    rb = [0.0, np.log(1.02), np.log(104 / 102)]
    beta_3 = (ra[1] - ra[0]) / (rb[1] - rb[0])
    beta_4 = (ra[2] - ra[1]) / (rb[2] - rb[1])
    same(panel.eval(cross.ret_beta("A", "B", 2)), [nan, nan, beta_3, beta_4], rtol=1e-9)
    # t=3: A's return falls (ln(102/101) -> 0) while B's rises (0 -> ln 1.02).
    # t=4: A's rises (0 -> ln(104/102)) while B's falls, since 104/102 < 1.02.
    same(panel.eval(cross.ret_corr("A", "B", 2)), [nan, nan, -1, -1], rtol=1e-9)


@cross_check
def test_ret_beta_matches_lstsq():
    rng = np.random.default_rng(3)
    n = 400
    t = np.sort(rng.uniform(0, 100, n))
    b_mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    a_mid = b_mid * np.exp(rng.normal(0, 2e-4, n))
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=99, freq="1s")
    p = Panel.from_pandas(
        grid,
        book={
            "A": quotes(a_mid - 0.01, a_mid + 0.01, t),
            "B": quotes(b_mid - 0.01, b_mid + 0.01, t),
        },
    )
    ra = np.diff(np.log(p.frame["A.mid"].to_numpy()), prepend=nan)
    rb = np.diff(np.log(p.frame["B.mid"].to_numpy()), prepend=nan)
    w = 20
    expected = [nan] * w
    for k in range(w, len(ra)):
        design = np.column_stack([np.ones(w), rb[k - w + 1 : k + 1]])
        expected.append(np.linalg.lstsq(design, ra[k - w + 1 : k + 1], rcond=None)[0][1])
    same(p.eval(cross.ret_beta("A", "B", w)), expected, rtol=1e-8)


@edge
def test_basis_z_needs_exactly_one_of_window_or_halflife(panel):
    with pytest.raises(ValueError, match="exactly one"):
        cross.basis_z("A", "B")
    with pytest.raises(ValueError, match="exactly one"):
        cross.basis_z("A", "B", window=3, halflife=1)


@edge
def test_leg_without_a_quote_yet_gives_nan():
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=2, freq="1s")
    p = Panel.from_pandas(
        grid, book={"A": quotes([1.0], [1.2], [0.5]), "B": quotes([1.0], [1.2], [1.5])}
    )
    same(p.eval(cross.basis("A", "B")), [nan, 0])


def log_mids(y, x):
    """A 1 s grid from midnight whose log mids are ``y`` for A and ``x`` for B."""
    grid = pd.date_range(T0, periods=len(y), freq="1s")

    def book(v):
        m = np.exp(np.asarray(v, dtype=float))
        idx = grid - pd.Timedelta("500ms")
        return pd.DataFrame({"bid": m, "ask": m, "bid_qty": 1.0, "ask_qty": 1.0}, index=idx)

    return Panel.from_pandas(grid, book={"A": book(y), "B": book(x)})


@known
def test_an_exact_fit_has_a_ratio_and_a_mean_but_no_scale():
    # y = 2x + 1 on 12 rows; clock windows of 4 s hold rows 0-3, 4-7 and 8-11, and each is
    # fitted on the 4 rows before it
    x = 0.01 * np.arange(12.0)
    p = log_mids(2 * x + 1, x)
    beta = p.eval(cross.hedge_ratio("A", "B", 4, "4s"))
    same(beta, [nan] * 4 + [2.0] * 8, rtol=1e-9)
    # rows 0-3: mean y - 2 mean x = (1 + 0.02 * 1.5) - 2 * (0.01 * 1.5) = 1, and the spread
    # y - 2x is 1 on every row, so it sits on its mean, with a residual sd of 0: z is null
    same(p.eval(cross.pair_spread("A", "B", 4, "4s")), [nan] * 4 + [1.0] * 8, rtol=1e-9)
    assert p.eval(cross.spread_z("A", "B", 4, "4s")).null_count() == 12


@known
def test_a_unit_hedge_with_noise_puts_the_spread_in_bps():
    # rows 0-3, the fit: x = (-3, -1, 1, 3) bps and y = x + e with e = (+1, -1, -1, +1) bps.
    # sum(x e) = -3 + 1 - 1 + 3 = 0 and mean(e) = 0, so beta = 1, mu = 0 and the residual
    # variance is mean(e^2) = 1 bps^2. In rows 4-7, z = (y - x - 0) / 1 bps = e.
    xb = np.array([-3.0, -1, 1, 3, 0, 2, -2, 5])
    eb = np.array([1.0, -1, -1, 1, 3, -2, 0.5, 1])
    c = np.log(100.0)
    p = log_mids(c + (xb + eb) * 1e-4, c + xb * 1e-4)
    same(p.eval(cross.hedge_ratio("A", "B", 4, "4s"))[4:], [1.0] * 4, rtol=1e-7)
    same(p.eval(cross.spread_z("A", "B", 4, "4s")), [nan] * 4 + [3, -2, 0.5, 1], rtol=1e-6)


def frame(values):
    return pl.DataFrame({"s": np.asarray(values, dtype=float)})


@known
def test_half_life_of_a_geometric_decay():
    # 8, 4, 2, 1, 0.5: each value is half the last, slope 0.5, -ln 2 / ln 0.5 = 1 step
    got = frame([8, 4, 2, 1, 0.5]).select(cross.half_life(pl.col("s"), 4)).to_series()
    same(got, [nan] * 4 + [1.0], rtol=1e-12)
    # 2^(-t/3): slope 2^(-1/3), -ln 2 / (-ln 2 / 3) = 3 steps on every full window
    decay = 2.0 ** (-np.arange(10) / 3)
    got = frame(decay).select(cross.half_life(pl.col("s"), 5)).to_series()
    same(got, [nan] * 5 + [3.0] * 5, rtol=1e-9)


@edge
def test_half_life_without_reversion_is_infinite():
    # 1..5 adds 1 a step (slope 1), 2^t doubles (slope 2): neither comes back
    for values in ([1, 2, 3, 4, 5], 2.0 ** np.arange(5)):
        got = frame(values).select(cross.half_life(pl.col("s"), 4)).to_series().to_list()
        assert got == [None] * 4 + [np.inf]
    # 1, -1, 1, ... has slope -1: it overshoots every step, and has no half-life
    got = frame([1, -1, 1, -1, 1]).select(cross.half_life(pl.col("s"), 4)).to_series()
    assert got.to_list() == [None] * 5


@cross_check
@pytest.mark.parametrize("window", [120, "2min"])
def test_spread_z_matches_a_polyfit_before_each_clock_window(window):
    # 20 minutes at 1 s, clock windows of 5 minutes; each is fitted with np.polyfit on the
    # 120 rows before it (a 2 min span on this grid is the same rows)
    rng = np.random.default_rng(5)
    n, w, every = 1200, 120, 300
    x = np.log(100.0) + np.cumsum(rng.normal(0, 1e-4, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.95 * noise[t - 1] + rng.normal(0, 2e-5)
    y = 0.8 * x + 0.9 + noise
    p = log_mids(y, x)
    beta, z = np.full(n, nan), np.full(n, nan)
    for start in range(every, n, every):
        slope, icpt = np.polyfit(x[start - w : start], y[start - w : start], 1)
        resid = y[start - w : start] - slope * x[start - w : start] - icpt
        rows = slice(start, start + every)
        beta[rows] = slope
        z[rows] = (y[rows] - slope * x[rows] - icpt) / resid.std()
    same(p.eval(cross.hedge_ratio("A", "B", window, "5min")), beta, rtol=1e-6)
    same(p.eval(cross.pair_spread("A", "B", window, "5min")), y - beta * x, rtol=1e-9)
    same(p.eval(cross.spread_z("A", "B", window, "5min")), z, rtol=1e-5)
