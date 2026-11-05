import numpy as np
import pandas as pd
import pytest

from blinded import Panel
from blinded import diagnostics as D

known = pytest.mark.known_answer
edge = pytest.mark.edge_case
T0 = pd.Timestamp("2024-01-01", tz="UTC")


def stamps(seconds):
    return pd.DatetimeIndex([T0 + pd.Timedelta(seconds=float(x)) for x in seconds])


def quotes(seconds, bid, ask):
    n = len(seconds)
    return pd.DataFrame(
        {"bid": bid, "ask": ask, "bid_qty": np.ones(n), "ask_qty": np.ones(n)},
        index=stamps(seconds),
    )


@known
def test_stream_table_gaps_duplicates_and_staleness():
    book = quotes([0.0, 0.1, 0.1, 0.4], [1.0] * 4, [2.0] * 4)
    grid = pd.date_range(T0 + pd.Timedelta("100ms"), periods=5, freq="100ms")
    row = D.stream_table(Panel.from_pandas(grid, book={"A": book})).row(0, named=True)
    assert (row["rows"], row["duplicate_rows"]) == (4, 1)
    assert row["median_gap_ms"] == pytest.approx(200)  # distinct times 0, 0.1, 0.4
    assert row["p90_gap_ms"] == pytest.approx(280)  # 100 + 0.9 * (300 - 100)
    assert row["events_per_s"] == pytest.approx(3 / 0.4)
    assert row["stale_share"] == pytest.approx(1 / 5)  # only 0.3 s is more than a step old


@known
def test_touch_alignment_finds_late_trade_stamps():
    k = np.arange(60)
    book = quotes(k * 0.010, 100.0 + k, 101.0 + k)
    # each buy prints at book k's ask but is stamped 12 ms after book k, i.e. after book k+1
    prints = pd.DataFrame(
        {"price": 101.0 + k[1:50], "qty": 1.0, "side": 1.0}, index=stamps(k[1:50] * 0.010 + 0.012)
    )
    grid = pd.date_range(T0 + pd.Timedelta("100ms"), periods=5, freq="100ms")
    panel = Panel.from_pandas(grid, book={"A": book}, trades={"A": prints})
    fit = D.touch_alignment(panel.events("book", "A"), panel.events("trades", "A"))
    # offsets -11..-2 ms all read book k; the one nearest zero is -2
    assert fit == {"best_offset_ms": -2, "share_at_best": 1.0, "share_at_0": 0.0}


@known
def test_xcorr_lag_recovers_a_delay():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 120, 3000))
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 3000)))
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=1180, freq="100ms")
    panel = Panel.from_pandas(
        grid,
        book={
            "A": quotes(t, mid - 0.01, mid + 0.01),
            "B": quotes(t + 0.030, mid - 0.01, mid + 0.01),
        },
    )
    lag = D.xcorr_lag(
        panel.events("book", "A"), "mid", panel.events("book", "B"), "mid", panel.grid_ns
    )
    assert lag["lag_ms"] == 30
    assert lag["corr_at_lag"] > 0.95


@edge
def test_flat_stream_has_no_lag():
    grid = pd.date_range(T0 + pd.Timedelta("1s"), periods=50, freq="100ms")
    flat = quotes([0.5, 1.5], [1.0, 1.0], [2.0, 2.0])
    panel = Panel.from_pandas(grid, book={"A": flat, "B": flat})
    lag = D.xcorr_lag(
        panel.events("book", "A"), "mid", panel.events("book", "B"), "mid", panel.grid_ns
    )
    assert np.isnan(lag["lag_ms"])


@pytest.fixture(scope="module")
def three_families(long_brief, tmp_path_factory):
    import yaml

    from blinded import brief as briefs

    raw = yaml.safe_load(long_brief.read_text())
    raw["families"] = {"directional": {},
                       "lead_lag": {"leaders": ["XRPUSDT_PERP"], "followers": ["XRPUSDT"]},
                       "pairs": {"pairs": [["XRPUSDT_PERP", "XRPUSDT"]],
                                 "defaults": {"window": "20min", "every": "10min"}}}  # fmt: skip
    path = tmp_path_factory.mktemp("families") / "brief.yaml"
    path.write_text(yaml.safe_dump(raw))
    brief = briefs.load(path)
    return brief, Panel.from_brief(brief)


@known
def test_data_md_has_the_lead_lag_and_pairs_tables_on_train(three_families):
    from blinded.evaluate import scoring_window

    brief, panel = three_families
    text = D.render(brief, panel)
    assert "## Lead-lag on train" in text and "## Pairs on train" in text
    train = scoring_window(brief, panel, "train")
    (lead,) = D.lead_lag(brief, panel, train)
    # the synthetic spot quotes the perp 300 ms late: the peak is three 100 ms steps on, past
    # the first whole step after a 50 ms fill (150 ms), so a taker can use it
    assert lead["lag_at_peak_ms"] == 300 and lead["usable_by_taker"]
    assert "| XRPUSDT_PERP | XRPUSDT | 300 |" in text
    (pair,) = D.pairs_table(brief, panel, train)
    # the hedge ratio in force at the end of train (02:50 to 03:00) is the OLS slope of the
    # perp's log mid on spot's over the 20 minutes before 02:50
    frame = panel.to_pandas("XRPUSDT_PERP.mid", "XRPUSDT.mid")
    fit = (frame.index >= pd.Timestamp("2024-11-28 02:30", tz="UTC")) & (
        frame.index < pd.Timestamp("2024-11-28 02:50", tz="UTC"))  # fmt: skip
    y, x = (np.log(frame[c].to_numpy()[fit]) for c in ("XRPUSDT_PERP.mid", "XRPUSDT.mid"))
    slope, intercept = np.polyfit(x, y, 1)
    assert pair["hedge_ratio"] == pytest.approx(slope, rel=1e-6)
    resid = y - (slope * x + intercept)
    assert pair["residual_sd_bps"] == pytest.approx(resid.std() * 1e4, rel=1e-5)
    # train runs 02:05 to 03:00: the windows from 02:10 have a fit, the one at 02:00 has none
    assert pair["windows"] == 5 and 0 <= pair["no_reversion"] <= 1
    assert pair["median_half_life_s"] > 0
