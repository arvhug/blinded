"""Forward returns on the hand book of ``test_sim.py``: mids 100.00 from 0.0 s, 100.00 from
1.2 s, 100.20 from 2.2 s and 100.10 from 3.2 s; grid 1 to 5 s."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from test_sim import at, hand_panel

from blinded import Panel
from blinded.target import forward_returns, spread_returns

nan = np.nan
known = pytest.mark.known_answer
LATENCY = pd.Timedelta("500ms")
H = (pd.Timedelta("1s"),)


@known
def test_returns_start_after_the_latency():
    r = forward_returns(hand_panel(), "A", "mid", H, LATENCY)["1s"].to_list()
    # from 1.5 to 2.5 s: 100.00 -> 100.20; 2.5 -> 3.5 s: 100.20 -> 100.10; then flat. The
    # returns from 4 and 5 s end at 5.5 and 6.5 s, after the last grid point.
    assert r == pytest.approx([0.002, 100.10 / 100.20 - 1, 0, nan, nan], nan_ok=True)


@known
def test_returns_are_read_at_their_stamps_under_a_feed_delay():
    # Rows arrive 0.9 s after their stamps. The return from stamp 1.5 to 2.5 s is read as-of
    # available_at 2.4 and 3.4 s and is the same 100.00 -> 100.20; reading as-of 1.5 and
    # 2.5 s would give stamps 0.6 and 1.6 s, 100.00 -> 100.00. The return from 3 s ends at
    # available_at 3 + 0.5 + 0.9 + 1 = 5.4 s, after the last grid point, so it is NaN.
    r = forward_returns(hand_panel(delay_s=0.9), "A", "mid", H, LATENCY)["1s"].to_list()
    assert r == pytest.approx([0.002, 100.10 / 100.20 - 1, nan, nan, nan], nan_ok=True)


@known
def test_the_spread_return_is_a_less_b_in_logs():
    # A as above; B has mids 50.00 from 0.0 s, 50.05 from 1.2 s and 50.15 from 3.2 s
    a = hand_panel().book("A").drop("available_at")
    b = pl.DataFrame({"ts": at(0.0, 1.2, 3.2), "bid": [49.99, 50.04, 50.14],
                      "ask": [50.01, 50.06, 50.16]})  # fmt: skip
    p = Panel.from_frames(at(1, 2, 3, 4, 5), book={"A": a, "B": b})
    r = spread_returns(p, "A", "B", H, LATENCY)["1s"].to_list()
    # 1.5 -> 2.5 s: A 100.00 -> 100.20, B flat at 50.05. 2.5 -> 3.5 s: A 100.20 -> 100.10,
    # B 50.05 -> 50.15. 3.5 -> 4.5 s: both flat.
    expected = [np.log(100.20 / 100.00), np.log(100.10 / 100.20) - np.log(50.15 / 50.05), 0.0]
    assert r == pytest.approx([*expected, nan, nan], nan_ok=True)
