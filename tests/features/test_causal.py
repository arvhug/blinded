"""Every library function passes the truncation test, on synthetic data and on XRP when present."""

import numpy as np
import pandas as pd
import pytest

import blinded as bl
from blinded import truncation
from blinded.features import book, cross, series, trades, xs

F = bl.features
SPOT, PERP = "XRPUSDT", "XRPUSDT_PERP"


MID = book.mid(SPOT)


def r(sym=SPOT):
    return F.ret(book.mid(sym))


LEGS3 = {SPOT: F.ret(MID, "10s"), PERP: F.ret(book.mid(PERP), "10s"),
         "LAGGED": F.lag(F.ret(MID, "10s"), "1s")}  # fmt: skip

CASES = {
    "series.bollinger_b": F.bollinger_b(MID, "1min"),
    "series.diff": F.diff(MID, "1s"),
    "series.ewm_cross": F.ewm_cross(MID, "10s", "1min"),
    "series.ewm_mean": F.ewm_mean(MID, "30s"),
    "series.ewm_std": F.ewm_std(r(), "30s"),
    "series.ewm_zscore": F.ewm_zscore(MID, "30s"),
    "series.expanding_zscore": F.expanding_zscore(MID),
    "series.lag": F.lag(MID, "500ms"),
    "series.meanrev": F.meanrev(MID, 600),
    "series.momentum": F.momentum(MID, "1min"),
    "series.realized_vol": F.realized_vol(MID, "1min"),
    "series.ret": F.ret(MID, 10),
    "series.roll_beta": F.roll_beta(r(), r(PERP), "1min"),
    "series.roll_corr": F.roll_corr(r(), r(PERP), "1min"),
    "series.roll_kurt": F.roll_kurt(r(), 600),
    "series.roll_max": F.roll_max(MID, "30s"),
    "series.roll_mean": F.roll_mean(MID, "5min"),
    "series.roll_median": F.roll_median(MID, 300),
    "series.roll_min": F.roll_min(MID, "30s"),
    "series.roll_quantile": F.roll_quantile(MID, "1min", 0.9),
    "series.roll_rank": F.roll_rank(MID, 600),
    "series.roll_skew": F.roll_skew(r(), 600),
    "series.roll_std": F.roll_std(MID, "1min"),
    "series.roll_sum": F.roll_sum(r(), "30s"),
    "series.roll_winsorize": F.roll_winsorize(r(), "1min", 0.05),
    "series.rsi": F.rsi(MID, 140),
    "series.time_of_day": F.time_of_day(),
    "series.zscore": F.zscore(MID, "1min"),
    "book.depth_imbalance": book.depth_imbalance(SPOT, levels=1),
    "book.imbalance": book.imbalance(SPOT),
    "book.micro_minus_mid": book.micro_minus_mid(SPOT, bps=True),
    "book.microprice": book.microprice(SPOT),
    "book.mid": book.mid(SPOT),
    "book.quote_rate": book.quote_rate(SPOT, "10s"),
    "book.spread": book.spread(SPOT, bps=True),
    "book.spread_ticks": book.spread_ticks(SPOT, tick=1e-4),
    "trades.avg_trade_size": trades.avg_trade_size(SPOT, "10s"),
    "trades.kyle_lambda": trades.kyle_lambda(SPOT, "2min"),
    "trades.signed_volume": trades.signed_volume(SPOT, "10s"),
    "trades.trade_count": trades.trade_count(SPOT, "10s"),
    "trades.trade_imbalance": trades.trade_imbalance(SPOT, "10s"),
    "trades.vwap": trades.vwap(SPOT, "30s"),
    "trades.vwap_dev": trades.vwap_dev(SPOT, "30s"),
    "cross.basis": cross.basis(PERP, SPOT),
    "cross.basis_z": cross.basis_z(PERP, SPOT, halflife="1min"),
    "cross.half_life": cross.half_life(cross.basis(PERP, SPOT), 600),
    "cross.hedge_ratio": cross.hedge_ratio(PERP, SPOT, "1min", "1min"),
    "cross.lagged_corr": cross.lagged_corr(PERP, SPOT, "300ms", "1min"),
    "cross.lead_gap": cross.lead_gap(PERP, SPOT, "1s"),
    "cross.lead_gap_z": cross.lead_gap_z(PERP, SPOT, 10, "1min"),
    "cross.lead_ret": cross.lead_ret(PERP, "1s"),
    "cross.pair_spread": cross.pair_spread(PERP, SPOT, "1min", "1min"),
    "cross.rel_spread": cross.rel_spread(SPOT, PERP),
    "cross.ret_beta": cross.ret_beta(SPOT, PERP, "1min"),
    "cross.ret_corr": cross.ret_corr(SPOT, PERP, "1min"),
    "cross.spread_z": cross.spread_z(PERP, SPOT, "1min", "1min"),
    # the cross-sectional ones on three legs: each instrument's 10 s return and the spot's
    # lagged a second
    "xs.each": xs.each([SPOT, PERP], lambda s: F.ret(book.mid(s), "10s")),
    "xs.xs_demean": xs.xs_demean(LEGS3),
    "xs.xs_mean": xs.xs_mean(LEGS3),
    "xs.xs_rank": xs.xs_rank(LEGS3),
    "xs.xs_zscore": xs.xs_zscore(LEGS3),
}


def _finite_share(full) -> float:
    """The share of finite values, the smallest over the legs of a cross-section."""
    if isinstance(full, dict):
        return min(np.isfinite(v).mean() for v in full.values())
    return np.isfinite(full).mean()


def feature(name):
    """The case as a feature function: it ignores the panel and returns the expression."""
    expr = CASES[name]
    return lambda panel: expr


def test_every_library_function_has_a_case():
    public = {
        f"{mod.__name__.rsplit('.', 1)[1]}.{name}"
        for mod in (series, book, trades, cross, xs)
        for name in mod.__all__
    }
    assert set(CASES) == public
    assert len(public) == 61  # 60 functions and F.depth_imbalance


@pytest.mark.parametrize("name", sorted(CASES))
def test_library_function_is_causal_on_synthetic(name, synthetic_panel):
    p = synthetic_panel
    window = bl.Window("synthetic", p.time(0) + pd.Timedelta("5min"), p.time(-1))
    verdict = truncation.check(feature(name), p, window, seed=1)
    assert verdict.ok, str(verdict)
    assert _finite_share(verdict.full) > 0.9  # not passing by being empty


@pytest.mark.parametrize("name", ["basket"])
def test_basket_is_causal_on_synthetic(name, synthetic_panel):
    p = synthetic_panel
    window = bl.Window("synthetic", p.time(0) + pd.Timedelta("5min"), p.time(-1))
    verdict = truncation.check(lambda panel: bl.basket(LEGS3, top=1, every="30s"), p, window,
                               seed=1)  # fmt: skip
    assert verdict.ok, str(verdict)
    assert _finite_share(verdict.full) > 0.9
