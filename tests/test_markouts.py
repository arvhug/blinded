import math

import numpy as np
import pytest

from blinded import markouts as mk

known = pytest.mark.known_answer
edge = pytest.mark.edge_case

NAN = float("nan")
S = 1_000_000_000  # one second in ns


@known
def test_events_take_the_first_point_past_an_edge_then_skip_the_gap():
    # one point a second, edges at -1 and +1, a gap of 3 s
    x = np.array([0.0, 2.0, 3.0, -2.0, 0.0, -3.0, 5.0, 0.0, 0.0, 2.0])
    idx, sign = mk.events(x, np.arange(10) * S, -1.0, 1.0, 3 * S)
    # 1 s is taken; 2 s and 3 s fall inside its gap; 5 s is the first past an edge from 4 s on;
    # 6 s and 7 s are inside that gap; 9 s is next
    assert idx.tolist() == [1, 5, 9]
    assert sign.tolist() == [1.0, -1.0, 1.0]


@edge
def test_events_skip_nan_and_points_on_the_edge():
    x = np.array([NAN, 1.0, -1.0, 1.5, NAN])
    idx, sign = mk.events(x, np.arange(5) * S, -1.0, 1.0, S)
    assert idx.tolist() == [3] and sign.tolist() == [1.0]  # an edge itself is not past it
    none, _ = mk.events(np.array([0.0, 0.5]), np.arange(2) * S, -1.0, 1.0, S)
    assert none.tolist() == []


@known
def test_clusters_start_a_new_run_at_the_gap():
    assert mk.clusters(np.array([0, 2, 3, 9, 10, 30]) * S, 5 * S).tolist() == [0, 0, 0, 1, 1, 2]
    assert mk.clusters(np.array([], dtype=np.int64), S).tolist() == []


@known
def test_fan_is_the_mean_with_the_sample_standard_error():
    # four events at two offsets; signs flip the shorts into the traded direction
    moves = [np.array([1.0, -2.0, 3.0, -1.0]), np.array([2.0, -4.0, NAN, 0.0])]
    sign = np.array([1.0, -1.0, 1.0, -1.0])
    got = mk.fan(moves, sign)
    # offset 1: traded moves 1, 2, 3, 1; offset 2: 2, 4, 0 (the NaN event is out)
    first, second = np.array([1.0, 2.0, 3.0, 1.0]), np.array([2.0, 4.0, 0.0])
    assert got["mean_bps"] == pytest.approx([first.mean(), second.mean()])
    assert got["se_bps"] == pytest.approx([first.std(ddof=1) / 2, second.std(ddof=1) / 3**0.5])
    # longs are events 1 and 3: (1 + 3) / 2, then 2 alone; shorts 2 and 4: (2 + 1) / 2, (4 + 0) / 2
    assert got["long_bps"] == pytest.approx([2.0, 2.0])
    assert got["short_bps"] == pytest.approx([1.5, 2.0])
    assert (got["n"], got["n_long"], got["n_short"]) == (4, 2, 2)


@known
def test_fan_weights_the_mean_and_clusters_the_error_by_hand():
    # three orders, sizes 1, 2, 1; the first two are one cluster
    v = np.array([4.0, 1.0, -2.0])
    w = np.array([1.0, 2.0, 1.0])
    got = mk.fan([v], np.ones(3), w, np.array([0, 0, 1]))
    mean = (4 + 2 + -2) / 4  # 1.0
    # cluster sums of w (v - mean): 1 * 3 + 2 * 0 = 3, and 1 * -3 = -3
    se = math.sqrt(2 / 1 * (3**2 + 3**2)) / 4
    assert got["mean_bps"] == [pytest.approx(mean)]
    assert got["se_bps"] == [pytest.approx(se)]


@edge
def test_fan_with_one_draw_has_a_mean_but_no_error():
    got = mk.fan([np.array([2.0, 3.0])], np.ones(2), cluster=np.array([0, 0]))
    assert got["mean_bps"] == [2.5] and got["se_bps"] == [None]
    empty = mk.fan([np.array([])], np.array([]))
    assert empty["mean_bps"] == [None] and empty["n"] == 0


@known
def test_window_ics_follow_the_clock():
    # 30 s windows; the grid starts 10 s into the first one
    g = np.array([10, 15, 20, 25, 30, 35, 40, 45]) * S
    x = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    got = mk.window_ics(x, y, g, 30 * S)
    # window [0, 30) holds 10-25 s, ranks agree: +1; window [30, 60) holds 30-45 s, reversed: -1
    assert [s for s, _ in got] == [0, 30 * S]
    assert [ic for _, ic in got] == pytest.approx([1.0, -1.0])


@edge
def test_rank_ic_needs_three_points_and_spread():
    assert mk.rank_ic(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None
    assert mk.rank_ic(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) is None
    assert mk.rank_ic(np.array([1.0, NAN, 2.0, 3.0]), np.array([3.0, 0.0, 2.0, 1.0])) == -1.0


def step_mid(at):
    """100.00 until 3 s, 100.10 from then on."""
    return np.where(at < 3 * S, 100.00, 100.10)


@known
def test_mid_moves_on_the_decision_clock():
    # decisions at 2 s and 4 s, offsets -2 s, 0 and +2 s, data to 5 s
    got = mk.mid_moves(step_mid, np.array([2, 4]) * S, [-2 * S, 0, 2 * S], last_ns=5 * S)
    up = (100.10 / 100.00 - 1) * 1e4  # +10 bps
    down = (100.00 / 100.10 - 1) * 1e4  # -9.99 bps
    # 2 s before: 0 s is flat against 2 s; 2 s is before the step, against 100.10 at 4 s
    assert got[0].tolist() == pytest.approx([0.0, down])
    assert got[1].tolist() == [0.0, 0.0]  # zero at the decision by construction
    assert got[2][0] == pytest.approx(up)  # 2 s after the 2 s decision: past the step
    assert np.isnan(got[2][1])  # 6 s is past the data


@known
def test_an_order_fan_by_hand():
    # a buy of 1 sent at 1 s and a sell of 2 sent at 4 s
    t = np.array([1, 4]) * S
    side, size = np.array([1.0, -1.0]), np.array([1.0, 2.0])
    moves = mk.mid_moves(step_mid, t, [-2 * S, 0, 3 * S], last_ns=10 * S)
    got = mk.fan(moves, side, size)["mean_bps"]
    # -2 s: the buy sees 100.00 against 100.00 (0); the sell sees 100.00 against 100.10, in its
    # direction +9.99 bps: it sold after the price had gone up. Weighted (0 + 2 * 9.99) / 3.
    sold_after_rise = -(100.00 - 100.10) / 100.10 * 1e4
    assert got[0] == pytest.approx(2 * sold_after_rise / 3)
    assert got[1] == 0.0
    # +3 s: the buy is past the step, +10 bps; the sell is flat at 100.10, 0
    assert got[2] == pytest.approx((100.10 - 100.00) / 100.00 * 1e4 / 3)


@known
def test_entry_cost_is_spread_plus_the_move_in_flight():
    # sent at a mid of 100; the buy fills at the ask 100.03 after the mid moved up 1 bps,
    # the sell at the bid 99.99: 3 bps and 1 bps of cost, weighted 1:3
    got = mk.entry_cost(np.array([1.0, -1.0]), np.array([1.0, 3.0]),
                        np.array([100.03, 99.99]), np.array([100.0, 100.0]))  # fmt: skip
    assert got == pytest.approx((3.0 + 3 * 1.0) / 4)


@edge
def test_an_order_fan_leaves_out_marks_past_the_data():
    flat = lambda at: np.full(len(at), 100.0)  # noqa: E731
    moves = mk.mid_moves(flat, np.array([0, 5 * S]), [2 * S], last_ns=6 * S)
    got = mk.fan(moves, np.ones(2), np.ones(2))
    assert got["mean_bps"] == [0.0]  # only the first order's mark at 2 s is inside the data
    assert got["se_bps"] == [None]  # one order left: no spread to measure
    assert mk.entry_cost(np.array([]), np.array([]), np.array([]), np.array([])) is None


@known
def test_half_spread_skips_one_sided_points():
    bid = np.array([99.0, NAN, 99.5])
    ask = np.array([101.0, 100.0, 100.5])
    # (2 / 100) * 1e4 / 2 = 100 bps and (1 / 100) * 1e4 / 2 = 50 bps; the median of two is 75
    assert mk.half_spread_bps(bid, ask) == pytest.approx(75.0)


@known
def test_downsample_keeps_both_ends():
    idx = mk.downsample(1001, 11)
    assert idx[0] == 0 and idx[-1] == 1000 and len(idx) == 11
    assert np.all(np.diff(idx) == 100)
    assert mk.downsample(5, 600).tolist() == [0, 1, 2, 3, 4]
    assert math.isclose(len(mk.downsample(10**6)), 600)
