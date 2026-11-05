import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from blinded import Panel
from blinded.features import trades

nan = np.nan
known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case
T0 = pd.Timestamp("2024-01-01", tz="UTC")


def same(result, expected, rtol=1e-12):
    assert_allclose(np.asarray(result, dtype=float), expected, rtol=rtol, equal_nan=True)


def times(*seconds):
    return pd.DatetimeIndex([T0 + pd.Timedelta(seconds=x) for x in seconds])


def make_panel(prints, quotes=None, periods=4, step="100ms"):
    grid = pd.date_range(T0 + pd.Timedelta(step), periods=periods, freq=step)
    if quotes is None:
        quotes = pd.DataFrame(
            {"bid": [99.0], "ask": [101.0], "bid_qty": [1.0], "ask_qty": [1.0]}, index=times(0)
        )
    return Panel.from_pandas(grid, book={"A": quotes}, trades={"A": prints})


@pytest.fixture
def panel():
    """Grid 0.1..0.4 s. Trades: buy 1@100 (0.05), buy 2@101 and sell 1@100 (both 0.15),
    sell 3@99 (0.35)."""
    prints = pd.DataFrame(
        {
            "price": [100.0, 101.0, 100.0, 99.0],
            "qty": [1.0, 2.0, 1.0, 3.0],
            "side": [1.0, 1.0, -1.0, -1.0],
        },
        index=times(0.05, 0.15, 0.15, 0.35),
    )
    return make_panel(prints)


# windows of 200 ms: (-0.1, 0.1], (0, 0.2], (0.1, 0.3], (0.2, 0.4]


@known
def test_signed_volume(panel):
    same(panel.eval(trades.signed_volume("A", "200ms")), [1, 2, 1, -3])  # 1; 1+2-1; 2-1; -3


@known
def test_trade_count(panel):
    same(panel.eval(trades.trade_count("A", "200ms")), [1, 3, 2, 1])


@known
def test_trade_imbalance(panel):
    same(panel.eval(trades.trade_imbalance("A", "200ms")), [1, 0.5, 1 / 3, -1])  # (3-1)/4, (2-1)/3


@known
def test_vwap(panel):
    # (100 + 202 + 100) / 4 and (202 + 100) / 3
    same(panel.eval(trades.vwap("A", "200ms")), [100, 100.5, 302 / 3, 99])


@known
def test_vwap_dev_against_mid_100(panel):
    same(
        panel.eval(trades.vwap_dev("A", "200ms")),
        [0, (100 / 100.5 - 1) * 1e4, (100 / (302 / 3) - 1) * 1e4, (100 / 99 - 1) * 1e4],
    )


@known
def test_avg_trade_size(panel):
    same(panel.eval(trades.avg_trade_size("A", "200ms")), [1, 4 / 3, 1.5, 3])


@known
def test_kyle_lambda_recovers_an_exact_impact():
    # mid moves by exactly 0.001 log points per unit of signed volume in each step
    flow = np.array([2.0, -1.0, 3.0, 0.5, -2.0, 1.0])
    log_mid = np.concatenate([[0.0], np.cumsum(0.001 * flow)])
    quotes = pd.DataFrame(
        {
            "bid": np.exp(log_mid) - 0.0001,
            "ask": np.exp(log_mid) + 0.0001,
            "bid_qty": 1.0,
            "ask_qty": 1.0,
        },
        index=times(*np.arange(7) + 0.5),
    )
    prints = pd.DataFrame(
        {"price": 1.0, "qty": np.abs(flow), "side": np.sign(flow)},
        index=times(*np.arange(6) + 1.2),
    )
    # grid at 1..7 s: quotes at k+0.5 and trades at k+0.2, so step k carries flow[k-1]
    p = make_panel(prints, quotes, periods=7, step="1s")
    lam = p.eval(trades.kyle_lambda("A", 4))
    same(lam[5:], [0.001, 0.001], rtol=1e-6)


@cross
def test_flow_features_match_a_loop_over_trades():
    rng = np.random.default_rng(1)
    n = 400
    t = np.sort(rng.uniform(0, 20, n))
    prints = pd.DataFrame(
        {
            "price": 100 + rng.normal(0, 0.5, n),
            "qty": rng.exponential(2, n),
            "side": rng.choice([-1.0, 1.0], n),
        },
        index=times(*t),
    )
    p = make_panel(prints, periods=200)
    sv, vw, imb = [], [], []
    for g in pd.DatetimeIndex(p.grid_ns, tz="UTC"):
        win = prints[(prints.index > g - pd.Timedelta("1s")) & (prints.index <= g)]
        buys, sells = win.qty[win.side > 0].sum(), win.qty[win.side < 0].sum()
        sv.append(buys - sells)
        vw.append((win.price * win.qty).sum() / win.qty.sum() if len(win) else nan)
        imb.append((buys - sells) / (buys + sells) if len(win) else nan)
    same(p.eval(trades.signed_volume("A", "1s")), sv, rtol=1e-9)
    same(p.eval(trades.vwap("A", "1s")), vw, rtol=1e-9)
    same(p.eval(trades.trade_imbalance("A", "1s")), imb, rtol=1e-9)


@cross
def test_kyle_lambda_matches_lstsq_per_window():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 60, 500))
    prints = pd.DataFrame(
        {"price": 100.0, "qty": rng.exponential(1, 500), "side": rng.choice([-1.0, 1.0], 500)},
        index=times(*t),
    )
    qt = np.sort(rng.uniform(0, 60, 300))
    mids = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 300)))
    quotes = pd.DataFrame(
        {"bid": mids - 0.01, "ask": mids + 0.01, "bid_qty": 1.0, "ask_qty": 1.0}, index=times(*qt)
    )
    p = make_panel(prints, quotes, periods=550)
    r = np.diff(np.log(p.frame["A.mid"].to_numpy()), prepend=nan)
    v = p.eval(trades.signed_volume("A", "100ms")).to_numpy()
    w = 50
    expected = [nan] * w
    for k in range(w, len(r)):
        design = np.column_stack([np.ones(w), v[k - w + 1 : k + 1]])
        expected.append(np.linalg.lstsq(design, r[k - w + 1 : k + 1], rcond=None)[0][1])
    same(p.eval(trades.kyle_lambda("A", w)), expected, rtol=1e-6)


@edge
def test_empty_window_gives_zero_sums_and_nan_ratios(panel):
    # (0.2, 0.3] holds no trades in a 100 ms window
    assert panel.eval(trades.signed_volume("A", "100ms"))[2] == 0
    assert panel.eval(trades.trade_count("A", "100ms"))[2] == 0
    for fn in (trades.trade_imbalance, trades.vwap, trades.vwap_dev, trades.avg_trade_size):
        assert np.isnan(panel.eval(fn("A", "100ms")).to_numpy()[2])


@edge
def test_window_is_open_on_the_left_and_closed_on_the_right():
    prints = pd.DataFrame({"price": [1.0], "qty": [5.0], "side": [1.0]}, index=times(0.2))
    p = make_panel(prints)
    # a trade stamped exactly 0.2 is in (0.1, 0.2] and (0, 0.2], not in (0.2, 0.3]
    same(p.eval(trades.trade_count("A", "100ms")), [0, 1, 0, 0])
    same(p.eval(trades.trade_count("A", "200ms")), [0, 1, 1, 0])
