"""The lead-lag cross-correlation C(tau), its lags and its summary, by hand."""

import numpy as np
import polars as pl
import pytest

from blinded.diagnostics import lead_lag_curve, lead_lag_lags, lead_lag_summary
from blinded.panel import Events

known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case

MS = 1_000_000
STEP = 100 * MS
LATENCY = 50 * MS


def events(at_ns, mids) -> Events:
    t = pl.Series(np.asarray(at_ns, dtype=np.int64)).cast(pl.Datetime("ns"))
    mids = np.asarray(mids, dtype=float)
    return Events(pl.DataFrame({"ts": t, "available_at": t, "bid": mids, "ask": mids}))


def at(lags, curve, lag_ms):
    return curve[list(lags).index(lag_ms * MS)]


@known
def test_a_follower_two_steps_behind_peaks_at_200ms_and_a_taker_can_use_it():
    # the leader's one-step returns, in units of c, at grid points 1..8 of a 100 ms grid
    x = np.array([1, -1, 1, -1, 2, 0, 1, -1], dtype=float)
    c = 1e-4
    level = np.r_[0.0, np.cumsum(x)]  # S_0 .. S_8
    g = np.arange(9) * STEP
    lead = events(g, 100 * np.exp(c * level))
    # the follower starts at grid point 2 and is the leader two steps late: f(g_t) = S_{t-2}
    follow = events(g[2:], 50 * np.exp(c * level[:-2]))
    lags = lead_lag_lags(STEP, LATENCY, raw=True)
    curve = lead_lag_curve(lead, follow, g, STEP, lags)
    # tau = 200 ms: the follower's step ending 200 ms after g_t is r_f(t + 2) = c x_t, for
    # t = 1..6 (t + 2 stays on the grid): the same six numbers, C = 1
    assert at(lags, curve, 200) == pytest.approx(1.0)
    # tau = -200 ms: r_f(t - 2) = c x_{t-4}, defined from t = 5 (the follower needs g_{t-3} >=
    # g_2). Pairs (x5..x8) = (2, 0, 1, -1) with (x1..x4) = (1, -1, 1, -1). Deviations from the
    # means 0.5 and 0: (1.5, -0.5, 0.5, -1.5) and (1, -1, 1, -1). Sum of products 4, sums of
    # squares 5 and 4: the lag-4 correlation is 4 / sqrt(20) = 2 / sqrt(5).
    assert at(lags, curve, -200) == pytest.approx(2 / np.sqrt(5))
    # tau = 150 ms, the first whole step after a fill at 50 ms: the follower read at g_t + 150
    # is its g_{t+1} value and at g_t + 50 its g_t value, so r_f = c x_{t-1}, t = 2..6:
    # pairs (x_t, x_{t-1}) = (-1, 1), (1, -1), (-1, 1), (2, -1), (0, 2). Deviations from the
    # means 0.2 and 0.4: (-1.2, 0.8, -1.2, 1.8, -0.2) and (0.6, -1.4, 0.6, -1.4, 1.6); the
    # products sum to -5.4 and the squares to 6.8 and 7.2.
    lag1 = -5.4 / np.sqrt(6.8 * 7.2)
    assert at(lags, curve, 150) == pytest.approx(lag1)
    table = lead_lag_summary(lags, curve, LATENCY, STEP)
    assert table["lag_at_peak_ms"] == 200
    assert table["c_at_peak"] == pytest.approx(1.0)
    assert table["c_after_fill"] == pytest.approx(lag1)
    # from 150 ms on the best is the peak itself, at 200 ms: usable
    assert table["best_after_fill"] == pytest.approx(1.0)
    assert table["usable_by_taker"] is True


@known
def test_a_follower_30ms_behind_peaks_at_0_and_is_gone_before_the_fill():
    # 11 grid points, 100 ms apart. The leader jumps once, 60 ms before g_5; the follower
    # copies it 30 ms later, still 30 ms before g_5, inside one step and inside the latency.
    g = np.arange(11) * STEP
    lead = events([0, g[5] - 60 * MS], [100.0, 101.0])
    follow = events([0, g[5] - 30 * MS], [50.0, 50.5])
    lags = lead_lag_lags(STEP, LATENCY, raw=False)  # whole steps and 150 ms, as a thinned file
    curve = lead_lag_curve(lead, follow, g, STEP, lags)
    # The leader's returns are a single spike at t = 5 among t = 1..10 (g_0 has no step behind
    # it). At tau = 0 the follower's spike is at t = 5 too: C = 1. At a whole step k the
    # follower's spike moves to t = 5 - k among the 10 - |k| pairs that stay on the grid, and
    # two single spikes at different places in n points correlate at -1/(n - 1).
    assert at(lags, curve, 0) == pytest.approx(1.0)
    assert at(lags, curve, 100) == pytest.approx(-1 / 8)
    assert at(lags, curve, -100) == pytest.approx(-1 / 8)
    assert at(lags, curve, 300) == pytest.approx(-1 / 6)
    # At 150 ms the follower's step is (g_t + 50, g_t + 150], which holds its jump at
    # g_4 + 70 for t = 4; follower reads stay on the grid for t <= 8, so n = 8.
    assert at(lags, curve, 150) == pytest.approx(-1 / 7)
    # past 5 steps the follower's spike leaves the window: one side is constant, no number
    assert at(lags, curve, 500) is None
    table = lead_lag_summary(lags, curve, LATENCY, STEP)
    assert (table["lag_at_peak_ms"], table["c_at_peak"]) == (0, pytest.approx(1.0))
    # everything a taker could hold, from 150 ms on, is negative: not usable
    assert table["best_after_fill"] == pytest.approx(-1 / 7)
    assert table["usable_by_taker"] is False


@known
def test_one_jump_a_step_is_a_plateau_on_the_fine_lags_and_ties_go_to_zero():
    # the same jump with the raw events: every lag whose follower step (g_5 + tau - 100,
    # g_5 + tau] holds g_5 - 30 sees the spike at t = 5, so C = 1 from -30 ms to 60 ms
    g = np.arange(11) * STEP
    lead = events([0, g[5] - 60 * MS], [100.0, 101.0])
    follow = events([0, g[5] - 30 * MS], [50.0, 50.5])
    lags = lead_lag_lags(STEP, LATENCY, raw=True)
    curve = lead_lag_curve(lead, follow, g, STEP, lags)
    ones = [lag // MS for lag, v in zip(lags, curve, strict=True) if v == pytest.approx(1.0)]
    assert ones == list(range(-30, 70, 10))
    assert lead_lag_summary(lags, curve, LATENCY, STEP)["lag_at_peak_ms"] == 0


@known
def test_the_lags_are_whole_steps_and_the_first_step_after_the_fill():
    coarse = [lag // MS for lag in lead_lag_lags(500 * MS, 100 * MS, raw=False)]
    assert coarse == [*range(-5000, 1, 500), 500, 600, *range(1000, 5001, 500)]
    fine = [lag // MS for lag in lead_lag_lags(100 * MS, 50 * MS, raw=True)]
    # 10 ms steps over one step either side of zero, and the fill point mirrored
    assert fine[:10] == [-1000, -900, -800, -700, -600, -500, -400, -300, -200, -150]
    assert [f for f in fine if -100 <= f <= 100] == list(range(-100, 101, 10))
    assert 150 in fine and len(fine) == 21 + 21 - 3 + 2


@cross
def test_whole_step_lags_match_pandas_shift_and_corr(synthetic_panel):
    p = synthetic_panel
    lead, follow = p.events("book", "XRPUSDT_PERP"), p.events("book", "XRPUSDT")
    frame = p.to_pandas("XRPUSDT_PERP.mid", "XRPUSDT.mid")
    rl = np.log(frame["XRPUSDT_PERP.mid"]).diff()
    rf = np.log(frame["XRPUSDT.mid"]).diff()
    skip = 11  # past the first ten steps either way, both sides read from the grid's own rows
    for k in (-3, -1, 0, 1, 3, 5):
        want = rl.iloc[skip:].corr(rf.shift(-k).iloc[skip:])
        (got,) = lead_lag_curve(lead, follow, p.grid_ns[skip:], p.step_ns, [k * p.step_ns])
        assert got == pytest.approx(want, rel=1e-10), k
    # the spot quotes the perp 300 ms late, so the curve peaks three steps on
    lags = [k * p.step_ns for k in range(-5, 6)]
    curve = lead_lag_curve(lead, follow, p.grid_ns, p.step_ns, lags)
    assert lead_lag_summary(lags, curve, LATENCY, p.step_ns)["lag_at_peak_ms"] == 300


@edge
def test_no_finite_pairs_gives_no_peak():
    lead = events([0], [1.0])
    g = np.arange(5) * STEP
    curve = lead_lag_curve(lead, lead, g, STEP, [0, STEP])
    assert curve == [None, None]
    summary = lead_lag_summary([0, STEP], curve, LATENCY, STEP)
    assert summary["lag_at_peak_ms"] is None and summary["usable_by_taker"] is False
