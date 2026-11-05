"""Judge trading features by recomputation."""

from blinded import features
from blinded.archetypes import (
    basis_trade,
    cross_venue_arbitrage,
    cross_venue_making,
    lead_lag,
    market_making,
    order_book_imbalance,
    pairs_trading,
    short_term_reversal,
    statistical_arbitrage,
    trade_flow_persistence,
    trend_following,
)
from blinded.brief import Brief, Window
from blinded.brief import load as load_brief
from blinded.contract import Feature, Strategy, feature, load, strategy
from blinded.evaluate import Result, StrategyResult, evaluate, evaluate_strategy
from blinded.panel import Panel
from blinded.sim import basket, carry, linear, pairs, threshold
from blinded.sim_maker import quote

__all__ = [
    "Brief",
    "Feature",
    "Panel",
    "Result",
    "Strategy",
    "StrategyResult",
    "Window",
    "basket",
    "carry",
    "evaluate",
    "evaluate_strategy",
    "feature",
    "features",
    "linear",
    "load",
    "load_brief",
    "pairs",
    "quote",
    "strategy",
    "threshold",
    # the archetypes' decorators (blinded.archetypes)
    "basis_trade",
    "cross_venue_arbitrage",
    "cross_venue_making",
    "lead_lag",
    "market_making",
    "order_book_imbalance",
    "pairs_trading",
    "short_term_reversal",
    "statistical_arbitrage",
    "trade_flow_persistence",
    "trend_following",
]
__version__ = "0.1.0"
