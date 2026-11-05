import numpy as np
import pandas as pd
import polars as pl
import pytest

from blinded import Panel, Window, truncation

nan = np.nan
known = pytest.mark.known_answer
cross = pytest.mark.cross_check
edge = pytest.mark.edge_case
T0 = pd.Timestamp("2024-01-01", tz="UTC")
MID = pl.col("A.mid")


@pytest.fixture
def squares():
    """Grid 1..10 s; a quote at k - 0.5 s with mid k**2, so the mid on the grid is k**2."""
    k = np.arange(1, 11, dtype=float)
    quotes = pd.DataFrame(
        {"bid": k**2 - 0.5, "ask": k**2 + 0.5, "bid_qty": 1.0, "ask_qty": 1.0},
        index=pd.DatetimeIndex([T0 + pd.Timedelta(seconds=x - 0.5) for x in k]),
    )
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=10, freq="1s")
    return Panel.from_pandas(grid, book={"A": quotes})


def whole(panel):
    return Window("all", panel.time(0), panel.time(-1) + pd.Timedelta("1s"))


def centered_polars(p):
    return MID.rolling_mean(3, center=True)


def centered_pandas(p):
    return p.to_pandas()["A.mid"].rolling(3, center=True).mean()


@known
@pytest.mark.parametrize("centered", [centered_polars, centered_pandas])
def test_centered_mean_is_caught_at_the_last_point_before_the_first_cut(squares, centered):
    verdict = truncation.check(centered, squares, whole(squares))
    v = verdict.violation
    assert verdict.status == "rejected_lookahead"
    # 10 points, the 0.50 cut sits after point int(0.5 * 9) = 4, i.e. 5 s, plus 0.37 s
    assert v.cut == T0 + pd.Timedelta("5.37s")
    # full run at 5 s: (16 + 25 + 36) / 3; the cut run has no 6 s point, so NaN
    assert v.first == T0 + pd.Timedelta("5s")
    assert (v.n_bad, v.nan_mismatch, v.n_compared) == (1, 1, 5)


@known
def test_trailing_mean_passes_every_cut(squares):
    def trailing(p):
        return MID.rolling_mean(3)

    verdict = truncation.check(trailing, squares, whole(squares))
    assert verdict.ok
    assert len(verdict.cuts) == 7


@known
def test_a_leak_in_one_leg_names_the_leg(squares):
    def two_legs(p):
        return {"A": MID.rolling_mean(3), "B": MID.rolling_mean(3, center=True)}

    verdict = truncation.check(two_legs, squares, whole(squares))
    v = verdict.violation
    # leg A is the trailing mean and agrees everywhere; leg B is the centred mean above,
    # caught at the same cut and point
    assert verdict.status == "rejected_lookahead" and v.leg == "B"
    assert (v.cut, v.first, v.n_bad) == (T0 + pd.Timedelta("5.37s"), T0 + pd.Timedelta("5s"), 1)
    assert "leg B" in str(v)
    clean = truncation.check(lambda p: {"A": MID, "B": -MID}, squares, whole(squares))
    assert clean.ok and list(clean.full) == ["A", "B"]
    assert clean.full["B"].tolist() == [-(k**2) for k in range(1, 11)]


@known
def test_full_sample_mean_differs_everywhere(squares):
    def demeaned(p):
        return MID - MID.mean()

    v = truncation.check(demeaned, squares, whole(squares)).violation
    # full mean of 1..100 squares is 38.5; the 5 s cut's mean of 1, 4, 9, 16, 25 is 11
    assert v.first == T0 + pd.Timedelta("1s")
    assert v.max_diff == pytest.approx(38.5 - 11)


@known
def test_compare_rules():
    grid = pd.date_range(T0, periods=4, freq="1s").as_unit("ns").asi8
    full = np.array([nan, 1.0, 2.0, 3.0])
    assert truncation.compare(full, full.copy(), grid, 1.0, cut=None) is None
    assert truncation.compare(full, full + 1e-13, grid, 1.0, cut=None) is None
    bad = truncation.compare(full, np.array([nan, 1.0, 2.0 + 1e-5, nan]), grid, 1.0, cut=None)
    assert bad.first == T0 + pd.Timedelta("2s")
    assert (bad.n_bad, bad.nan_mismatch) == (2, 1)
    assert bad.max_diff == pytest.approx(1e-5)


@known
def test_cut_times(squares):
    cuts = truncation.cut_times(squares, whole(squares), seed=0)
    offsets = {c - c.floor("1s") for c in cuts}
    assert len(cuts) == 7
    assert offsets == {pd.Timedelta("370ms")}
    assert cuts[0] == T0 + pd.Timedelta("5.37s")
    assert all(squares.time(0) < c < squares.time(-1) + pd.Timedelta("1s") for c in cuts)


@known
def test_cut_moves_to_where_the_next_event_is_close():
    # quotes at k + 0.1 s, except none at 11.1 or 12.1 and one at 12.6 instead
    stamps = [k + 0.1 for k in range(20) if k not in (11, 12)] + [12.6]
    quotes = pd.DataFrame(
        {"bid": 1.0, "ask": 2.0, "bid_qty": 1.0, "ask_qty": 1.0},
        index=pd.DatetimeIndex(sorted(T0 + pd.Timedelta(seconds=x) for x in stamps)),
    )
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=20, freq="1s")
    panel = Panel.from_pandas(grid, book={"A": quotes})
    # window 10..19 s: the 0.50 point is 14 s, and from there on every next quote is
    # 0.1 s ahead, inside the 0.37 s offset, so the cut stays at 14.37 s
    late = truncation.cut_times(panel, Window("w", grid[9], grid[-1]), seed=0)[0]
    assert late == T0 + pd.Timedelta("14.37s")
    # window 2..19 s: the 0.50 point is int(0.5 * 17) = 8 -> 10 s. At 12 s the next quote
    # (12.6) is 0.6 s ahead and the last (10.1) 1.9 s behind, so the cut moves to 12.37 s
    early = truncation.cut_times(panel, Window("w", grid[1], grid[-1]), seed=0)[0]
    assert early == T0 + pd.Timedelta("12.37s")


@cross
def test_noise_is_nondeterministic(squares):
    def noise(p):
        return pl.Series(np.random.default_rng().normal(size=len(p)))

    verdict = truncation.check(noise, squares, whole(squares))
    assert verdict.status == "rejected_nondeterministic"
    assert verdict.violation.cut is None


@edge
def test_tiny_float_noise_from_summation_order_is_tolerated(squares):
    def wobble(p):
        return MID * (1 + 1e-12 * (len(p) % 2))

    assert truncation.check(wobble, squares, whole(squares)).ok


@edge
def test_nan_against_a_number_fails(squares):
    def blank_before_a_gap(p):
        return pl.when(MID.shift(-1).is_not_null()).then(MID)

    v = truncation.check(blank_before_a_gap, squares, whole(squares)).violation
    assert (v.first, v.nan_mismatch) == (T0 + pd.Timedelta("5s"), 1)


@edge
def test_bfill_of_the_warm_up_is_invisible_to_later_cuts(squares):
    # a known blind spot: the cuts sit in the scoring window, after the warm-up, so a
    # leak that only reaches from the warm-up into the first few points is not seen
    def filled_from_later(p):
        return MID.rolling_mean(3).fill_null(strategy="backward")

    assert truncation.check(filled_from_later, squares, whole(squares)).ok


@edge
def test_window_without_grid_points_raises(squares):
    empty = Window("e", T0 + pd.Timedelta("100s"), T0 + pd.Timedelta("200s"))
    with pytest.raises(ValueError, match="no grid points"):
        truncation.cut_times(squares, empty, seed=0)


def counted(fn):
    calls = []

    def wrapped(panel):
        calls.append(len(panel))
        return fn(panel)

    return wrapped, calls


@known
def test_precheck_counts_its_evaluations(synthetic_brief, synthetic_panel):
    from blinded import brief as briefs
    from blinded.evaluate import scoring_window
    from blinded.truncation import check, precheck

    win = scoring_window(briefs.load(synthetic_brief), synthetic_panel, "train")
    short = synthetic_panel.cut(win.end)
    leak, calls = counted(lambda p: pl.col("XRPUSDT.mid").rolling_mean(50, center=True))
    bad, _ = precheck(leak, short, win)
    # full run, the 0.50 cut, then one more full run to rule out noise: 3 evaluations
    assert bad.status == "rejected_lookahead" and len(calls) == 3
    clean, calls = counted(lambda p: pl.col("XRPUSDT.mid").rolling_mean(50))
    bad, full = precheck(clean, short, win)
    assert bad is None and len(calls) == 2  # full run and the 0.50 cut
    assert check(clean, short, win, full=full, skip=1).ok
    # then the determinism run and the other six cuts: 2 + 7 = 9, k + 2 in all
    assert len(calls) == 9
