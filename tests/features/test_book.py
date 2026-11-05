import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from blinded import Panel
from blinded.features import book, example_panel

nan = np.nan
known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case
T0 = pd.Timestamp("2024-01-01", tz="UTC")


def same(result, expected, rtol=1e-12):
    assert_allclose(np.asarray(result, dtype=float), expected, rtol=rtol, equal_nan=True)


def times(*seconds):
    return pd.DatetimeIndex([T0 + pd.Timedelta(seconds=x) for x in seconds])


@pytest.fixture
def panel():
    """Grid at 0.1, 0.2, 0.3 s. Two rows share 0.25 s; the second one wins."""
    quotes = pd.DataFrame(
        {
            "bid": [100.0, 101.0, 99.0],
            "ask": [102.0, 102.0, 101.0],
            "bid_qty": [1.0, 2.0, 4.0],
            "ask_qty": [3.0, 2.0, 1.0],
        },
        index=times(0.1, 0.25, 0.25),
    )
    grid = pd.date_range(T0 + pd.Timedelta("100ms"), periods=3, freq="100ms")
    return Panel.from_pandas(grid, book={"A": quotes})


@known
def test_mid_reads_inclusive_asof_and_last_duplicate(panel):
    # 0.1 sees the row stamped 0.1 itself; 0.3 sees the second 0.25 row
    same(panel.eval(book.mid("A")), [101, 101, 100])


@known
def test_spread_in_price_bps_and_ticks(panel):
    same(panel.eval(book.spread("A")), [2, 2, 2])
    same(panel.eval(book.spread("A", bps=True)), [2 / 101 * 1e4, 2 / 101 * 1e4, 2 / 100 * 1e4])
    same(panel.eval(book.spread_ticks("A", tick=0.5)), [4, 4, 4])


@known
def test_microprice(panel):
    # (100*3 + 102*1) / 4 and (99*1 + 101*4) / 5
    same(panel.eval(book.microprice("A")), [100.5, 100.5, 100.6])


@known
def test_imbalance(panel):
    same(panel.eval(book.imbalance("A")), [-0.5, -0.5, 0.6])  # (1-3)/4, (4-1)/5


@known
def test_depth_imbalance_adds_the_levels_behind_the_touch():
    # the touch 1 / 3 and the next level 5 / 2, then 2 / 4 behind it: (1 + 5 - 3 - 2) / 11
    # = 1/11 over two levels, (1 + 5 + 2 - 3 - 2 - 4) / 17 = -1/17 over three; one level is
    # the touch's imbalance, (1 - 3) / 4
    quotes = pd.DataFrame({"bid": [100.0], "ask": [102.0], "bid_qty": [1.0], "ask_qty": [3.0],
                           "bid1_qty": [5.0], "ask1_qty": [2.0], "bid2_qty": [2.0],
                           "ask2_qty": [4.0]}, index=times(0.05))  # fmt: skip
    grid = pd.date_range(T0 + pd.Timedelta("100ms"), periods=2, freq="100ms")
    p = Panel.from_pandas(grid, book={"A": quotes})
    same(p.eval(book.depth_imbalance("A", 1)), [-0.5, -0.5])
    same(p.eval(book.depth_imbalance("A", 2)), [1 / 11, 1 / 11])
    same(p.eval(book.depth_imbalance("A", 3)), [-1 / 17, -1 / 17])
    with pytest.raises(ValueError, match="at least 1"):
        book.depth_imbalance("A", 0)


@known
def test_micro_minus_mid(panel):
    same(panel.eval(book.micro_minus_mid("A")), [-0.5, -0.5, 0.6])
    same(panel.eval(book.micro_minus_mid("A", bps=True)), [-0.5 / 101 * 1e4, -0.5 / 101 * 1e4, 60])


@known
def test_quote_rate_counts_rows(panel):
    # 200 ms window: (-0.1, 0.1] has 1 row, (0, 0.2] 1, (0.1, 0.3] the two 0.25 rows
    same(panel.eval(book.quote_rate("A", "200ms")), [5, 5, 10])


@known
def test_example_panel_values():
    p = example_panel()
    same(p.eval(book.microprice("SPOT")), [10.75, 11, 11.5, 12.25])
    same(p.eval(book.imbalance("SPOT")), [0.5, 0, 0, -0.5])


@cross
def test_microprice_and_imbalance_match_a_loop_over_events():
    rng = np.random.default_rng(0)
    n = 300
    t = np.sort(rng.uniform(0, 30, n))
    bid = 100 + np.round(rng.normal(0, 1, n), 1)
    quotes = pd.DataFrame(
        {
            "bid": bid,
            "ask": bid + 0.1,
            "bid_qty": rng.integers(1, 9, n) * 1.0,
            "ask_qty": rng.integers(1, 9, n) * 1.0,
        },
        index=times(*t),
    )
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=29, freq="1s")
    p = Panel.from_pandas(grid, book={"A": quotes})
    micro, imb = [], []
    for g in grid:
        row = quotes[quotes.index <= g].iloc[-1]
        total = row.bid_qty + row.ask_qty
        micro.append((row.bid * row.ask_qty + row.ask * row.bid_qty) / total)
        imb.append((row.bid_qty - row.ask_qty) / total)
    same(p.eval(book.microprice("A")), micro, rtol=1e-12)
    same(p.eval(book.imbalance("A")), imb, rtol=1e-12)


@edge
def test_grid_point_before_the_first_quote_is_nan():
    quotes = pd.DataFrame(
        {"bid": [1.0], "ask": [2.0], "bid_qty": [1.0], "ask_qty": [1.0]}, index=times(1.5)
    )
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=2, freq="1s")
    p = Panel.from_pandas(grid, book={"A": quotes})
    same(p.eval(book.mid("A")), [nan, 1.5])
    same(p.eval(book.quote_rate("A", "1s")), [0, 1])


@edge
def test_quote_between_grid_points_appears_at_the_next_one(panel):
    # the 0.25 rows sit between grid points 0.2 and 0.3
    assert panel.eval(book.mid("A"))[1] == 101
    assert panel.eval(book.mid("A"))[2] == 100
