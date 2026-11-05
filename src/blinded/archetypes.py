"""Ten strategy archetypes: who pays each one, what data it needs, and whether a file has it.

An archetype is a strategy known by two things: who pays it, its *edge*
(speed, liquidity, relative value, underreaction, carry), and how the judge
scores and simulates it, its *mechanism* (one of the families of
:mod:`blinded.brief`). ``@bl.pairs_trading(a="A", b="B")`` is
``@bl.strategy(family="pairs", pair=("A", "B"))`` with a tag, so in a brief
with ``families:``, or with neither block, it changes nothing but the tag in
the ledger. In a brief with ``archetypes:`` the run is grouped by edge: a
Ladder per edge and kind, sessions shared by edge, and every submission must
use an archetype that the menu says this file can run.

The menu is decided once, at ``blinded init``, from the brief and the file:
:func:`facts` reads what the file has, :func:`check` tests one archetype
against it and :func:`menu` all of them. ``blinded init`` writes the menu to
``vault/menu.json`` and renders it as the agent's ``ARCHETYPES.md``
(:func:`render`), so the agent never finds a refusal by trial. The text of
each entry is copied from the reference page *Thirty Quant Strategies*.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from blinded.durations import timedelta, words

if TYPE_CHECKING:
    from blinded.brief import Brief
    from blinded.contract import Feature, Strategy
    from blinded.panel import Panel

PERIODS = 20
CROSS_SECTION = 20
_STREAMS = ("book", "trades", "funding")
_FIELDS = ("bid", "ask", "bid_qty", "ask_qty", "price", "qty", "side", "n", "rate")
_TRADE_FIELDS = {"price", "qty", "side"}
_HOURS_PER_YEAR = 24 * 365
_td = timedelta


@dataclass(frozen=True)
class Edge:
    """Who pays: a key, the title the refusal lines use, the page's tag and its sentence."""

    key: str
    title: str
    tag: str
    text: str


EDGES: dict[str, Edge] = {
    e.key: e
    for e in (
        Edge("speed", "speed & information", "Paid for reaching a price first",
             "Paid for reaching a price first. The other side is a stale quote or a slower "
             "reader, and the race only gets faster."),
        Edge("liquidity", "liquidity & flow", "Paid by people who must trade now",
             "Paid by people who have to trade now, or on a schedule everyone can see. You "
             "rent them your balance sheet."),
        Edge("relative_value", "relative value", "Paid to wait for two prices to meet",
             "Paid to hold two prices that should match until they do. The risk is that "
             "they move further apart first."),
        Edge("underreaction", "underreaction", "Paid because the crowd updates slowly",
             "Paid because the crowd takes weeks to change its mind. Gains come in trends, "
             "and some go back at the turns."),
        Edge("carry", "carry", "Paid to hold what others won't",
             "Paid to hold what others would rather not, for months or years. The slowest "
             "edges here, and the largest."),
    )
}  # fmt: skip

# the death checks of the report, by name; which an archetype runs is its ``checks``
DEATHS = {
    "costs": "D1: is the edge smaller than the spread plus fees",
    "race": "D2: is the move over before the fill",
    "capacity": "D3: how much of the touch one unit takes",
    "worst_window": "D4: when it lost",
    "half_life": "D5: how long the signal or the spread lasts",
    "mae": "D6: does it diverge before it converges",
    "funding": "D7: what carry paid or cost",
    "crowding": "D8: is everyone in the same book",
    "breadth": "D9: did one name carry it",
    "whipsaw": "D10: did it get chopped",
    "venue_feed": "D11: is the lead a feed schedule",
    "toxicity": "D12: are the fills the ones nobody wanted",
    "queue": "D13: where in the queue you were",
    "picked_off": "D14: did the deep venue move through the quote first",
    "hedge": "D15: what the hedge cost",
    "stock": "D16: where the coin piles up",
    "counterparty": "D17: who the fills come from",
}


@dataclass(frozen=True)
class Needs:
    """What an archetype needs from the file before it can run at all.

    ``touch`` is the fill engine's: the book at every change with its sizes
    and each trade at its own price and stamp (:func:`touch_lack`). ``depth``
    is an arbitrage's hedge leg's: the hedge venue's levels at every snapshot
    (``data.depth``).
    """

    instruments: int = 1
    streams: frozenset[str] = frozenset({"book"})
    fields: frozenset[str] = frozenset()
    venues: int = 1
    cross_section: int = 1
    carry_pairs: bool = False
    sim: str = "taker"
    touch: bool = False
    depth: bool = False


@dataclass(frozen=True)
class Archetype:
    """One registry entry: the edge and mechanism, the holding band, the page's text, the
    instrument roles its decorator takes, what it needs and the report's checks."""

    n: int
    name: str
    title: str
    edge: str
    mechanism: str | None
    template: str
    holding: tuple[pd.Timedelta, pd.Timedelta]
    capacity: str
    edge_text: str
    other_side: str
    dies: str
    seen: str | None
    read: str
    legs: tuple[str, ...]
    needs: Needs
    checks: tuple[str, ...]
    params: dict = field(default_factory=dict)
    builtins: tuple[str, ...] = ()


_THRESHOLD = {"z_in": 2.0, "z_out": 0.5, "every": "30m"}
_PAIRS = {"window": "2h", "every": "30m", "z_in": 2.0, "z_out": 0.5}

REGISTRY: dict[str, Archetype] = {
    a.name: a
    for a in (
        Archetype(
            1, "lead_lag", "Lead-lag", "speed", "lead_lag", "threshold/linear",
            (_td("10us"), _td("100ms")), "tiny",
            "A deep market moves first and thinner ones follow: index futures before the ETF, "
            "the biggest BTC perpetual before alt-coins and smaller venues. Trade the follower "
            "on the leader's move.",
            "Makers whose quotes haven't caught up yet.",
            "The lag is shorter than your round trip. It's a pure speed race, and the lags "
            "keep shrinking.",
            "On three Binance days BTC and ETH led SOL and XRP by about 10 ms. The correlation "
            "fell from about 0.3 to under 0.1 by the time a 100 ms order would fill.",
            "Budish, Cramton & Shim (2015); Aquilina, Budish & O'Neill (2022)",
            ("leader", "follower"), Needs(instruments=2),
            ("race", "costs", "capacity", "worst_window"), _THRESHOLD,
            ("F.lead_ret", "F.lead_gap", "F.lead_gap_z", "F.lagged_corr", "bl.threshold",
             "bl.linear"),
        ),
        Archetype(
            2, "order_book_imbalance", "Order-book imbalance", "speed", "directional",
            "threshold/linear", (_td("1ms"), _td("10s")), "tiny",
            "When the bid queue dwarfs the ask, the next mid move is more likely up. The "
            "microprice weights the mid by the queue sizes to get a better fair value.",
            "Whoever trades against you just before the thin side empties.",
            "The predicted move is usually smaller than the spread, so taking it loses money. "
            "It pays as a quoting signal for a maker, or for timing execution.",
            "Microprice features were the strongest signals found on three Binance days, with "
            "validation IC around 0.1. No strategy built on them beat a 7.5 bp fee.",
            "Cont, Kukanov & Stoikov (2014); Stoikov (2018)",
            ("instrument",), Needs(fields=frozenset({"bid_qty", "ask_qty"})),
            ("costs", "race", "capacity", "worst_window", "half_life"), _THRESHOLD,
            ("F.imbalance", "F.microprice", "F.micro_minus_mid", "bl.threshold", "bl.linear"),
        ),
        Archetype(
            3, "trade_flow_persistence", "Trade-flow persistence", "speed", "directional",
            "threshold/linear", (_td("1s"), _td("6h")), "small",
            "Large orders are cut into thousands of child orders, so the sign of trades stays "
            "correlated for hours. Spot a metaorder early and trade ahead of the rest of it.",
            "Institutions working big orders slowly.",
            "Executors randomise their slicing and hide in dark pools. Impact is concave too, "
            "so the later child orders move the price less than you'd hope.",
            None,
            "Lillo & Farmer (2004); Bouchaud, Farmer & Lillo (2009)",
            ("instrument",),
            Needs(streams=frozenset({"book", "trades"}), fields=frozenset({"side"})),
            ("costs", "capacity", "worst_window", "half_life", "mae"), _THRESHOLD,
            ("F.signed_volume", "F.trade_imbalance", "F.trade_count", "F.vwap_dev",
             "bl.threshold", "bl.linear"),
        ),
        Archetype(
            4, "pairs_trading", "Pairs trading", "relative_value", "pairs", "pairs",
            (_td("10min"), _td("30d")), "small to medium",
            "Two close substitutes drift apart and come back. Short the rich one, buy the "
            "cheap one, and size the legs with a hedge ratio.",
            "Liquidity shocks that hit one leg and not the other.",
            "The relationship breaks for good after a merger, a new product or a regulatory "
            "split. Returns also fell sharply after the early 2000s as more people ran it.",
            "ETH/BTC on Binance reverted with a half-life of one to two minutes. The moves were "
            "under a basis point, against a 29 bp round trip on two legs.",
            "Gatev, Goetzmann & Rouwenhorst (2006); Do & Faff (2010)",
            ("a", "b"), Needs(instruments=2, sim="legs"),
            ("mae", "half_life", "costs", "capacity", "worst_window"), _PAIRS,
            ("F.hedge_ratio", "F.pair_spread", "F.spread_z", "F.half_life", "bl.pairs"),
        ),
        Archetype(
            5, "cross_venue_arbitrage", "Cross-venue arbitrage", "speed", "pairs", "pairs",
            (_td("1us"), _td("1s")), "tiny",
            "The same asset quotes at two prices on two venues, or three currency pairs imply "
            "a cross rate the third doesn't match. Buy the cheap side and sell the rich side "
            "at the same moment.",
            "Stale quotes, and capital that can't move between venues quickly.",
            "You lose the race or run out of inventory on one side. Big gaps only last where "
            "money can't cross, like Korea's kimchi premium.",
            None,
            "Makarov & Schoar (2020)",
            ("a", "b"), Needs(instruments=2, venues=2, sim="legs"),
            ("race", "costs", "capacity", "worst_window", "half_life", "venue_feed"),
            {"window": "10m", "every": "10m", "z_in": 2.0, "z_out": 0.5},
            ("F.basis", "F.spread_z", "F.lead_gap", "bl.pairs"),
        ),
        Archetype(
            6, "statistical_arbitrage", "Statistical arbitrage", "relative_value", "basket",
            "basket", (_td("1h"), _td("14d")), "medium to large",
            "Strip each stock's return of a few factors, from PCA or sector ETFs, and trade "
            "the mean reversion of what's left across hundreds of names, factor-neutral.",
            "Idiosyncratic flow in single names.",
            "Crowding. In August 2007 one large book deleveraged and every similar book lost "
            "at once.",
            None,
            "Avellaneda & Lee (2010); Khandani & Lo (2007)",
            (), Needs(cross_section=CROSS_SECTION, sim="basket"),
            ("mae", "half_life", "costs", "worst_window", "breadth"),
            {"every": "4h", "top": 4, "lookback": "1d"},
            ("F.each", "F.xs_demean", "F.xs_zscore", "F.xs_rank", "bl.basket"),
        ),
        Archetype(
            7, "short_term_reversal", "Short-term reversal", "liquidity", "basket", "basket",
            (_td("4h"), _td("30d")), "medium",
            "The stocks that fell hardest last week tend to bounce. The selling usually came "
            "from someone who needed cash, and the price recovers once they stop.",
            "Investors who had to sell in a hurry.",
            "Costs eat most of it in small caps. It pays best when liquidity is scarce, which "
            "is also when it's hardest to fund.",
            None,
            "Lehmann (1990); Nagel (2012)",
            (), Needs(streams=frozenset({"book", "trades"}), cross_section=CROSS_SECTION,
                      sim="basket"),
            ("mae", "worst_window", "costs", "breadth"),
            {"every": "4h", "top": 4, "lookback": "4h"},
            ("F.each", "F.xs_zscore", "F.xs_rank", "F.ret", "bl.basket"),
        ),
        Archetype(
            8, "basis_trade", "Basis trade", "carry", "pairs", "carry",
            (_td("8h"), _td("90d")), "large",
            "When a future trades rich to its cash asset, buy the cash, sell the future, and "
            "earn the gap as they converge. Crypto perpetuals pay it continuously, as a "
            "funding rate from longs to shorts.",
            "Leveraged longs who want the exposure without holding the cash.",
            "Funding dries up, as in the March 2020 Treasury basis unwind, or the venue fails, "
            "as FTX did in 2022.",
            None,
            "Barth & Kahn (2021); He, Manela, Ross & von Wachter (2022)",
            ("spot", "perp"),
            Needs(instruments=2, streams=frozenset({"book", "funding"}), carry_pairs=True,
                  sim="carry"),
            ("funding", "mae", "worst_window", "costs"),
            {"rate_in": 0.0001, "rate_out": 0.0},
            ('pl.col("PERP.rate")', "F.basis", "bl.carry"),
        ),
        Archetype(
            9, "trend_following", "Trend following", "underreaction", "directional",
            "threshold/linear", (_td("14d"), _td("365d")), "large",
            "An asset's own return over the past year predicts its next month, across equity, "
            "bond, currency and commodity futures. It tends to pay in long crises like 2008 "
            "and 2022.",
            "Slow-moving investors, hedgers and central banks leaning against moves.",
            "Markets chop sideways or reverse hard, as they did through much of the 2010s.",
            None,
            "Moskowitz, Ooi & Pedersen (2012); Hurst, Ooi & Pedersen (2017)",
            ("instrument",), Needs(),
            ("worst_window", "costs", "whipsaw"), {"lookback": "4w", "every": "1w"},
            ("F.momentum", "F.ret", "F.ewm_cross", "bl.linear", "bl.threshold"),
        ),
        Archetype(
            10, "market_making", "Market making", "liquidity", "directional", "maker",
            (_td("1ms"), _td("10min")), "small",
            "Quote both sides, earn the spread and any rebate on each round trip, and skew "
            "your quotes to shed inventory.",
            "Impatient traders who cross the spread.",
            "Adverse selection. You get filled most just before the price moves against you. "
            "Faster makers take the front of the queue, and a fee change can erase the margin.",
            None,
            "Glosten & Milgrom (1985); Avellaneda & Stoikov (2008)",
            ("instrument",),
            Needs(streams=frozenset({"book", "trades"}), sim="maker", touch=True),
            ("costs", "capacity", "worst_window", "mae", "funding", "toxicity", "queue"),
            {"size": 1.0, "skew": 1.0, "max_inventory": 2.0},
            ("bl.quote", "F.imbalance", "F.microprice", "F.spread", "F.trade_count",
             "F.depth_imbalance"),
        ),
        Archetype(
            11, "cross_venue_making", "Quote the thin venue, hedge on the deep one",
            "liquidity", "directional", "arb", (_td("100ms"), _td("1min")), "small",
            "Rest inside the thin venue's spread at a distance from the deep venue's price, and "
            "when a taker there fills you, cross on the deep venue at once; you earn the "
            "distance less the hedge's half spread and fee, and hold nothing but the hedge "
            "delay.",
            "Takers on the thin venue who pay its spread rather than go to the deep one.",
            "The deep venue moves through your quote before you can cancel (picked off); the "
            "hedge fee and half spread eat the distance; the thin venue's flow stops, or "
            "re-prices around you; the coin piles up on one venue and must be moved.",
            None,
            "Hasbrouck (1995) on where price discovery happens; Budish, Cramton & Shim (2015) "
            "on who wins the race",
            ("quote", "hedge"),
            Needs(streams=frozenset({"book", "trades"}), sim="arb", touch=True, depth=True,
                  venues=2),
            ("costs", "capacity", "worst_window", "mae", "picked_off", "hedge", "stock",
             "counterparty"),
            {"distance_bps": 30.0, "size": 64.0},
            ("bl.quote", "F.mid", "F.theo", "F.basis", "F.spread", "F.trade_count",
             "F.imbalance"),
        ),
    )
}  # fmt: skip


# -- facts, the check and the menu ------------------------------------------


@dataclass(frozen=True)
class Facts:
    """What the file and the brief give the check, computed once at ``blinded init``.

    ``instruments`` are the run's (``archetypes.instruments``, or every one in
    the file), ``available`` every one in the file. ``spans`` are the lengths
    of the train, validation and test scoring windows, the grid warm-up taken
    off the start of train. ``fill`` is the latency plus the largest feed
    delay. ``cost_model`` is the kind of a declared fill model, on bar files.
    ``funding`` holds, for the perpetual of each carry pair, the settlements
    stamped in the train scoring window and their mean rate, when the facts
    come from a panel with a funding stream: what carry there was to collect.
    ``borrow`` holds, for the spot leg of each carry pair with a borrow
    stream, the mean and the highest daily rate in force over the train
    scoring window: what the reverse trade paid to be short spot.
    ``touch`` is whether the fill engine can run on the file
    (:func:`touch_lack`), and ``maker``, when the facts come from a training
    panel and were asked for, what market making's entry says of the book:
    the tick in bps, how often the spread is one tick, the touch's median
    size, and what one risk-averse pass of the archetype's own quote on train
    captured and gave back (:func:`maker_facts`).
    """

    instruments: tuple[str, ...]
    fields: dict[str, frozenset[str]]
    streams: frozenset[str]
    venues: int
    carry_pairs: tuple[tuple[str, str], ...]
    spans: dict[str, pd.Timedelta]
    step: pd.Timedelta
    fill: pd.Timedelta
    cost_model: str | None = None
    available: tuple[str, ...] = ()
    funding: dict = field(default_factory=dict)
    borrow: dict = field(default_factory=dict)
    touch: bool = False
    maker: dict = field(default_factory=dict)
    arb: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        extra = {"funding": self.funding} if self.funding else {}
        extra |= {"borrow": self.borrow} if self.borrow else {}
        extra |= {"touch": True} if self.touch else {}
        extra |= {"maker": self.maker} if self.maker else {}
        extra |= {"arb": self.arb} if self.arb else {}
        return extra | {
            "instruments": list(self.instruments),
            "available": list(self.available),
            "streams": [s for s in _ordered(self.streams, _STREAMS)],
            "venues": self.venues,
            "carry_pairs": [list(p) for p in self.carry_pairs],
            "spans_seconds": {k: v.total_seconds() for k, v in self.spans.items()},
            "spans": {k: words(v) for k, v in self.spans.items()},
            "step_seconds": self.step.total_seconds(),
            "fill_seconds": self.fill.total_seconds(),
            "cost_model": self.cost_model,
        }


@dataclass(frozen=True)
class Verdict:
    """One archetype on one file: ``runnable``, ``caveat`` or ``refused``.

    A refusal has the class of the first test that failed and its reason, and
    a ``fix`` when a change to the brief would turn it around with the same
    file. A caveat lists every caveat that applies, in ``caveats``.
    """

    status: str
    cls: str | None = None
    reason: str | None = None
    fix: str | None = None
    caveats: tuple[str, ...] = ()

    @property
    def runs(self) -> bool:
        return self.status != "refused"


def facts(brief: Brief, panel: Panel | None = None, windows: dict | None = None,
          maker: bool = False) -> Facts:  # fmt: skip
    """What the check needs to know, from the brief and, when given, the panel.

    With a ``panel``, the instruments in the file and the fields of each come
    from its events; without one, from the brief, which after ``blinded
    init`` lists the instruments and holds the windows it fixed. ``windows``
    default to the brief's fixed windows, else to those of the panel's span.
    With ``maker`` and a panel, the facts market making's entry prints are
    measured on the train scoring window when the file has a touch stream
    (:func:`maker_facts`): one engine pass, so only ``blinded init`` and the
    replay of its menu ask for them.
    """
    from blinded import brief as briefs

    spec = brief.archetypes
    in_file = tuple(panel.symbols) if panel is not None else ()
    listed = tuple(spec.instruments) if spec is not None and spec.instruments else ()
    instruments = listed or in_file or (brief.target.instrument,)
    available = in_file or instruments
    if panel is not None:
        fields = {
            sym: frozenset(f for stream, s in panel.streams if s == sym
                           for f in panel.events(stream, sym).fields)
            for sym in instruments
        }  # fmt: skip
    else:
        declared = frozenset(f for st in brief.data.streams.values() for f in st.fields)
        fields = dict.fromkeys(instruments, declared)
    have = set(instruments)
    cross = [p for p in (spec.cross_venue if spec else ()) if set(p) <= have]
    carry = tuple(p for p in (spec.carry if spec else ()) if set(p) <= have)
    if windows is None:
        if brief.splits.fixed:
            windows = {k: briefs.Window(k, a, b) for k, (a, b) in brief.splits.fixed.items()}
        elif panel is not None:
            windows = briefs.windows(brief, panel.time(0), panel.time(-1))
        else:
            raise ValueError(
                "the brief has no fixed windows; run blinded init first, or pass the panel"
            )
    first = windows["train"].start + brief.grid.warmup
    spans = {seg: windows[seg].end - max(windows[seg].start, first)
             for seg in ("train", "val", "test")}  # fmt: skip
    delays = [brief.data.default_delay_ms, *(float(r["delay_ms"]) for r in
                                             brief.data.delay_overrides)]  # fmt: skip
    cost = brief.data.cost_model.kind if brief.data.cost_model is not None else None
    funding = _funding(panel, carry, first, windows["train"].end) if panel is not None else {}
    borrow = _borrow(panel, carry, first, windows["train"].end) if panel is not None else {}
    touch = touch_lack(brief) is None
    measured = {}
    if maker and touch and panel is not None and (spec is None or not spec.only
                                                   or "market_making" in spec.only):  # fmt: skip
        measured = maker_facts(brief, panel, first, windows["train"].end)
    arb = {}
    if maker and touch and panel is not None and arb_lack(brief) is None and (
            spec is None or not spec.only or "cross_venue_making" in spec.only):  # fmt: skip
        arb = arb_facts(brief, panel, first, windows["train"].end)
    return Facts(
        instruments=instruments,
        fields=fields,
        streams=frozenset(brief.data.streams),
        venues=2 if cross else 1,
        carry_pairs=carry,
        spans=spans,
        step=brief.grid.step,
        fill=brief.latency + pd.Timedelta(milliseconds=max(delays)),
        cost_model=cost,
        available=available,
        funding=funding,
        borrow=borrow,
        touch=touch,
        maker=measured,
        arb=arb,
    )


def arb_quote(brief: Brief, quote: str, hedge: str, distance_bps: float, size: float):
    """The archetype's own quote: each side ``distance_bps`` off the hedge venue's touch, in
    the quote venue's ticks (the bid under the hedge venue's bid, the ask over its ask), at
    ``size``; the menu's facts and the report's fallback rest it."""
    import polars as pl

    from blinded import sim_maker

    tick = brief.execution.tick_size or 1.0
    bid, ask = pl.col(f"{hedge}.bid"), pl.col(f"{hedge}.ask")
    k = (distance_bps * 1e-4 * (bid + ask) / 2 / tick - 1e-9).ceil()
    return sim_maker.quote(bid=k, ask=k, size=size, skew=0.0, max_inventory=1e9, fair=(bid, ask))


def arb_facts(brief: Brief, panel: Panel, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """What the arbitrage's entry says of the file, measured on ``[start, end)`` of train: the
    two venues' median spreads, one risk-averse pass of the archetype's own quote at its
    parameters (:func:`arb_quote`) with its fills, what each fill captured, paid in slippage
    and in the hedge fee, and was picked off, and who prints on the quote venue
    (:func:`blinded.bookstats.flow_facts`)."""
    import numpy as np

    from blinded import bookstats, sim_maker
    from blinded.brief import Window
    from blinded.contract import values

    quote, hedge = arb_pair(brief)
    params = REGISTRY["cross_venue_making"].params | brief.archetypes.params_for(
        "cross_venue_making")  # fmt: skip
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [start.value, end.value])
    rows = panel.frame.slice(int(i0), int(i1 - i0))

    def spread(inst):
        bid, ask = (rows[f"{inst}.{f}"].to_numpy().astype(float) for f in ("bid", "ask"))
        ok = np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
        return float(np.median((ask[ok] - bid[ok]) / (ask[ok] + bid[ok]) * 2e4)) if ok.any() \
            else None  # fmt: skip

    win = Window("train", start, end)
    q = values(arb_quote(brief, quote, hedge, params["distance_bps"], params["size"]), panel)
    (ra,) = sim_maker.arb_models(q, panel, win, brief, quote, hedge, which=("RA",)).values()
    f = sim_maker.arb_fills(ra)
    n = len(f["side"])
    trades = panel.events("trades", quote)
    tt = trades.frame["ts"].dt.epoch("ns").to_numpy()
    keep = (tt >= start.value) & (tt < end.value)
    touch = ra.touch
    flow = None
    if keep.sum() > 1:
        flow = bookstats.flow_facts(
            touch.book_ns,
            touch.bid * touch.tick,
            touch.ask * touch.tick,
            tt[keep],
            trades.column("side")[keep],
            trades.column("price")[keep],
            trades.column("qty")[keep],
        )
    hours = (end - start) / pd.Timedelta("1h")

    def mean(key):
        return float(np.nanmean(f[key])) if n else None

    return {
        "quote": quote, "hedge": hedge,
        "quote_spread_bps": spread(quote), "hedge_spread_bps": spread(hedge),
        "distance_bps": float(params["distance_bps"]), "size": float(params["size"]),
        "hours": float(hours), "fills": n,
        "captured_bps": mean("captured_bps"), "slippage_bps": mean("slippage_bps"),
        "delay_bps": mean("delay_bps"), "hedge_fee_bps": ra.hedge_fee_bps,
        "net_bps": mean("net_bps"),
        "picked_off_share": float(np.mean(f["picked_off"])) if n else None,
        "prints_a_day": None if flow is None else flow["trades"] * 24 / hours,
        "print_usd_p50": None if flow is None else flow["usd_p50"],
        "never_shown": None if flow is None else flow["never_shown"],
    }  # fmt: skip


def touch_lack(brief: Brief) -> str | None:
    """What a file lacks for the fill engine, in the menu's words, or None when it has it.

    The engine reads the book at every change with its sizes and each trade
    at its own price and stamp: a declared touch stream (``data.touch``), or
    a file that keeps every book row with ``bid_qty`` and ``ask_qty`` and
    every trade with its ``price``, ``qty`` and ``side``, as the Tardis layout
    does. A file thinned to the read lattice, or of bars, keeps neither.
    """
    data = brief.data
    if data.touch is not None:
        return None
    if data.thinned is not None:
        return "a book thinned to the read lattice and trades summed per step"
    if data.cost_model is not None:
        return "bars"
    book, trades = data.streams.get("book"), data.streams.get("trades")
    for name in ("bid_qty", "ask_qty"):
        if book is None or name not in book.fields:
            return f"no {name}"
    if trades is None or not {"price", "qty", "side"} <= set(trades.fields):
        return "no trades with a price, qty and side"
    return None


def maker_facts(brief: Brief, panel: Panel, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """What market making's entry says of the book, measured on ``[start, end)`` of train.

    On the instrument the entry's example quotes: the tick (the brief's, else
    read from the book) in bps of the median mid; the share of grid points
    whose spread is one tick; the median size at the touch, both sides
    pooled; and one risk-averse pass of ``bl.quote`` at the archetype's
    parameters at the brief's latencies and maker fee, its units filled and,
    per unit, what it captured against the mid at the fill and what the mid
    took back within 10 seconds.
    """
    import numpy as np

    from blinded import sim_maker
    from blinded.brief import Window

    _, legs = example(REGISTRY["market_making"], brief)
    inst = legs["instrument"]
    ex = brief.execution
    touch = sim_maker.touch_of(panel, inst, ex.tick_size)
    g = panel.grid_ns
    i0, i1 = np.searchsorted(g, [start.value, end.value])
    rows = panel.frame.slice(int(i0), int(i1 - i0))
    bid, ask = (rows[f"{inst}.{f}"].to_numpy().astype(float) for f in ("bid", "ask"))
    sizes = np.concatenate([rows[f"{inst}.{f}"].to_numpy().astype(float)
                            for f in ("bid_qty", "ask_qty")])  # fmt: skip
    ok = np.isfinite(bid) & np.isfinite(ask)
    mid = (bid[ok] + ask[ok]) / 2
    wide = np.rint((ask[ok] - bid[ok]) / touch.tick)
    params = brief.archetypes.params_for("market_making") if brief.archetypes else {}
    params = REGISTRY["market_making"].params | params
    quote = sim_maker.quote(bid=0, ask=0, size=params["size"], skew=params["skew"],
                            max_inventory=params["max_inventory"])  # fmt: skip
    (ra,) = sim_maker.models(quote, panel, Window("train", start, end), brief, inst, touch,
                             ("RA",)).values()  # fmt: skip
    d = sim_maker.decompose(ra, (sim_maker.ADVERSE,))
    units, per = d["units"], d["per_unit"]
    return {
        "instrument": inst,
        "tick": touch.tick,
        "tick_declared": ex.tick_size is not None,
        "tick_bps": float(touch.tick / np.median(mid) * 1e4) if len(mid) else None,
        "one_tick_share": float(np.mean(wide == 1)) if len(wide) else None,
        "spread_ticks_median": float(np.median(wide)) if len(wide) else None,
        "spread_bps_median": float(np.median((ask[ok] - bid[ok]) / mid * 1e4))
        if len(mid)
        else None,
        "touch_median": float(np.nanmedian(sizes)) if np.isfinite(sizes).any() else None,
        "maker_fee_bps": ex.maker_fee_bps,
        "size": float(params["size"]),
        "units": units,
        "captured_bps_per_unit": per["captured_bps"] if units else None,
        "adverse_bps_10s_per_unit": per["adverse_bps"]["10s"] if units else None,
    }


_YEAR = pd.Timedelta("365D")


def _funding(panel: Panel, carry, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Each carry pair's perpetual: its settlements stamped in ``[start, end)``, their mean
    rate in bps and their sum over the span as a rate a year, in percent."""
    import numpy as np

    out = {}
    for _, perp in carry:
        if ("funding", perp) not in panel.streams:
            continue
        ev = panel.events("funding", perp)
        f = ev.frame["ts"].dt.epoch("ns").to_numpy()
        rate = ev.column("rate")[(f >= start.value) & (f < end.value)]
        rate = rate[np.isfinite(rate)]
        out[perp] = {"settlements": int(len(rate)),
                     "mean_rate_bps": float(np.mean(rate) * 1e4) if len(rate) else None,
                     "yearly_pct": float(rate.sum() * 100 * (_YEAR / (end - start)))}  # fmt: skip
    return out


def _borrow(panel: Panel, carry, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Each carry pair's spot leg: the mean and the highest daily borrow rate in force over
    ``[start, end)``, in bps, the mean weighted by how long each rate held."""
    import numpy as np

    out = {}
    for spot, _ in carry:
        if ("borrow", spot) not in panel.streams:
            continue
        ev = panel.events("borrow", spot)
        f = ev.frame["ts"].dt.epoch("ns").to_numpy()
        rate = ev.column("borrow_rate")
        # the rate in force at start, then each one published inside the span, each held until
        # the next or the end
        first = max(int(np.searchsorted(f, start.value, side="right")) - 1, 0)
        last = int(np.searchsorted(f, end.value, side="left"))
        if last <= first:
            continue
        stamps = np.clip(f[first:last], start.value, end.value)
        held = np.diff(np.append(stamps, end.value)).astype(float)
        r = rate[first:last]
        keep = np.isfinite(r) & (held > 0)
        if not keep.any():
            continue
        out[spot] = {"mean_bps_per_day": float(np.average(r[keep], weights=held[keep]) * 1e4),
                     "max_bps_per_day": float(np.max(r[keep]) * 1e4)}  # fmt: skip
    return out


def check(arch: Archetype, facts: Facts, brief: Brief) -> Verdict:
    """Test one archetype against the file. The first failing test is the reason.

    In order: the sim's data (for an arbitrage, the hedge venue's depth, the
    pair, the stock and its withdrawal fee: :func:`arb_lack`; for market making
    and an arbitrage, a touch stream: :func:`touch_lack`); the streams; two
    venues of one product; the instruments (for ``pairs_trading`` a pair that
    is not one product on two venues or a carry pair; for ``basis_trade`` a
    declared carry pair); the cross-section; the fields; train and validation each at least
    :data:`PERIODS` holding periods of the band's low end; and the brief's
    ``only:``. Then two caveats, both reported when both apply: a grid step
    above the band's high end, and a declared fill model.
    """
    needs, spec = arch.needs, brief.archetypes

    def refused(cls: str, reason: str, fix: str | None = None) -> Verdict:
        return Verdict("refused", cls, reason, fix)

    if needs.depth:
        lack = arb_lack(brief)
        if lack is not None:
            return refused(*lack)
    if needs.touch and not facts.touch:
        return refused("sim", touch_line(brief))
    for stream in _ordered(needs.streams, _STREAMS):
        if stream not in facts.streams:
            have = ", ".join(_ordered(facts.streams, _STREAMS))
            return refused("stream", f"needs {stream} data; the file has {have}")
    if facts.venues < needs.venues:
        return refused("venues", f"needs the same instrument on {needs.venues} venues; the file "
                                 "has one")  # fmt: skip
    k, inst = needs.instruments, facts.instruments
    if needs.carry_pairs and not facts.carry_pairs:
        return refused("instruments", "needs a spot and perpetual pair under archetypes.carry; "
                                      "the brief declares none")  # fmt: skip
    if len(inst) < k:
        return refused("instruments", f"needs {k} instruments; the file has {len(inst)}: "
                                      + ", ".join(inst), _more(facts, k, spec))  # fmt: skip
    if arch.name == "pairs_trading":
        taken = _declared_pairs(spec)
        free = [p for p in itertools.combinations(inst, 2) if frozenset(p) not in taken]
        if not free:
            a, b = next(iter(itertools.combinations(inst, 2)))
            if frozenset((a, b)) in {frozenset(p) for p in (spec.carry if spec else ())}:
                return refused("instruments", f"needs a pair that is not a carry pair; {a} and "
                                              f"{b} are (basis_trade)")  # fmt: skip
            return refused("instruments", "needs a pair that is not one product on two venues; "
                                          f"{a} and {b} are (cross_venue_arbitrage)")  # fmt: skip
    if len(inst) < needs.cross_section:
        return refused("cross_section", f"needs a cross-section of at least "
                                        f"{needs.cross_section} names; the file has {len(inst)}",
                       _more(facts, needs.cross_section, spec))  # fmt: skip
    for name in _ordered(needs.fields, _FIELDS):
        if not any(name in facts.fields.get(sym, ()) for sym in inst):
            return refused("fields", f"needs {name}; {_have(brief, name)}")
    lo, hi = arch.holding
    need = PERIODS * lo
    for seg, word in (("val", "validation"), ("train", "train")):
        span = facts.spans[seg]
        if span < need:
            return refused("span", f"needs {words(need)} of {word} ({PERIODS} holding periods "
                                   f"of {words(lo)}); {word} is {words(span)}")  # fmt: skip
    if spec is not None and spec.only and arch.name not in spec.only:
        fix = f"add {arch.name} to archetypes.only"
        return refused("only", "left out by archetypes.only", fix)
    caveats, classes = [], []
    if facts.step > hi:
        classes.append("grid")
        caveats.append(f"the grid step {words(facts.step)} is above the typical holding "
                       f"({words(hi)}); what the judge measures is the first whole step after "
                       "the fill")  # fmt: skip
    if facts.cost_model:
        classes.append("cost")
        caveats.append("fills are a model on this file: the next bar's close plus or minus a "
                       "measured half spread (DATA.md)")  # fmt: skip
    if caveats:
        return Verdict("caveat", ", ".join(classes), None, None, tuple(caveats))
    return Verdict("runnable")


def arb_pair(brief: Brief) -> tuple[str, str] | None:
    """An arbitrage's quote and hedge venues: the declared ``archetypes.cross_venue`` pair that
    holds the depth file's venue, the other one quoted; None when there is none."""
    depth, spec = brief.data.depth, brief.archetypes
    if depth is None or spec is None:
        return None
    for a, b in spec.cross_venue:
        if depth.venue in (a, b):
            return (b, a) if a == depth.venue else (a, b)
    return None


def arb_lack(brief: Brief) -> tuple[str, str] | None:
    """Why an arbitrage cannot be judged on a file, as ``(class, line)``, or None when it can:
    the hedge venue's depth, the quote and hedge declared as one product, the quote venue's
    touch stream, the stock block, and its withdrawal fee."""
    from blinded.sim_maker import coin_of, unset_fees

    depth = brief.data.depth
    if depth is None:
        return ("sim", "needs the hedge venue's depth at every change (a depth file); the file "
                       "has none")  # fmt: skip
    pair = arb_pair(brief)
    if pair is None:
        return ("venues", "needs the quote and hedge instruments declared as one product under "
                          f"archetypes.cross_venue; the depth file's venue {depth.venue} is "
                          "under none")  # fmt: skip
    touch = brief.data.touch
    if touch is not None and touch.instrument != pair[0]:
        return ("sim", f"needs the quote venue's touch stream; the file's is {touch.instrument}'s")
    stock = brief.stock
    if stock is None or set(stock.venues) != set(pair):
        return ("stock", "needs the stock block: each venue's coin and USDT, for "
                         f"{pair[0]} and {pair[1]}")  # fmt: skip
    why = unset_fees(stock, coin_of(pair[0]), brief.judge_rules)
    if why is not None:
        return ("stock", why)
    return None


def touch_line(brief: Brief) -> str:
    """The reason market making is refused on a file without a touch stream."""
    return ("needs the book at every change with its sizes and each trade at its own price and "
            f"stamp (a touch stream); the file has {touch_lack(brief)}")  # fmt: skip


def line(arch: Archetype, brief_name: str, verdict: Verdict) -> str:
    """The one-line refusal, the same in ARCHETYPES.md, the ledger and the report."""
    head = f"{arch.name} ({EDGES[arch.edge].title}): not runnable on {brief_name}"
    out = f"{head}: {verdict.reason}."
    return out + (f" Change the brief: {verdict.fix}." if verdict.fix else "")


@dataclass(frozen=True)
class Menu:
    """The verdicts on one file, the edges that can run, their session shares, and the
    lines the agent reads.

    ``dropped`` are the edges ``archetypes.shares`` named that have nothing
    to run; the other shares were renormalised over what is left.
    """

    brief: str
    verdicts: dict[str, Verdict]
    edges: tuple[str, ...]
    shares: dict[str, float]
    dropped: tuple[str, ...] = ()
    cost_model: str | None = None
    power: str = ""
    facts: dict = field(default_factory=dict)

    @property
    def runnable(self) -> tuple[str, ...]:
        return tuple(n for n, v in self.verdicts.items() if v.runs)

    @property
    def refused(self) -> tuple[str, ...]:
        return tuple(n for n, v in self.verdicts.items() if not v.runs)

    def members(self, edge: str) -> tuple[str, ...]:
        """The runnable archetypes of one edge, in registry order."""
        return tuple(n for n in self.runnable if REGISTRY[n].edge == edge)

    def line(self, name: str) -> str | None:
        """The refusal line of ``name``, or None when it runs."""
        verdict = self.verdicts[name]
        return None if verdict.runs else line(REGISTRY[name], self.brief, verdict)

    @property
    def headline(self) -> str:
        titles = ", ".join(EDGES[e].title for e in self.edges)
        return (f"{len(self.runnable)} of {len(self.verdicts)} archetypes run on this data, in "
                f"{len(self.edges)} of {len(EDGES)} edges: {titles or 'none'}.")  # fmt: skip

    @property
    def dropped_line(self) -> str | None:
        if not self.dropped:
            return None
        names = ", ".join(EDGES[e].title for e in self.dropped)
        rest = ", ".join(f"{EDGES[e].title} {self.shares[e]:g}" for e in self.edges)
        return (f"shares for {names} dropped: no runnable archetype; the rest renormalised "
                f"to {rest}")  # fmt: skip

    def to_dict(self) -> dict:
        return {
            "brief": self.brief,
            "headline": self.headline,
            "runnable": len(self.runnable),
            "edges": list(self.edges),
            "shares": dict(self.shares),
            "dropped": list(self.dropped),
            "dropped_line": self.dropped_line,
            "cost_model": self.cost_model,
            "power": self.power,
            "facts": self.facts,
            "archetypes": [
                {"name": n, "edge": REGISTRY[n].edge, "status": v.status, "cls": v.cls,
                 "reason": v.reason, "fix": v.fix, "caveats": list(v.caveats),
                 "line": self.line(n)}
                for n, v in self.verdicts.items()
            ],
        }  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict) -> Menu:
        verdicts = {
            a["name"]: Verdict(a["status"], a["cls"], a["reason"], a["fix"], tuple(a["caveats"]))
            for a in d["archetypes"]
        }
        return cls(d["brief"], verdicts, tuple(d["edges"]), dict(d["shares"]),
                   tuple(d["dropped"]), d["cost_model"], d["power"], d["facts"])  # fmt: skip


def menu(brief: Brief, facts: Facts) -> Menu:
    """All the verdicts, the edges with something to run, their shares and the power line.

    Shares default to equal over the edges that can run. Given shares that
    name an edge with nothing to run are dropped and the rest renormalised;
    an edge that can run must have a share when any are given.
    """
    verdicts = {name: check(arch, facts, brief) for name, arch in REGISTRY.items()}
    running = {REGISTRY[n].edge for n, v in verdicts.items() if v.runs}
    edges = tuple(e for e in EDGES if e in running)
    given = dict(brief.archetypes.shares) if brief.archetypes is not None else {}
    dropped: tuple[str, ...] = ()
    if given:
        dropped = tuple(e for e in given if e not in edges)
        missing = [e for e in edges if e not in given]
        if missing:
            raise ValueError(
                f"archetypes.shares gives no share to {', '.join(missing)}, which has "
                "archetypes this file can run; give it a share, or leave it out with only:"
            )
        total = sum(given[e] for e in edges)
        shares = {e: round(given[e] / total, 12) for e in edges}
    else:
        shares = {e: round(1 / len(edges), 12) for e in edges}
    return Menu(
        brief=brief.name,
        verdicts=verdicts,
        edges=edges,
        shares=shares,
        dropped=dropped,
        cost_model=_cost_line(brief) if facts.cost_model else None,
        power=power(facts, brief),
        facts=facts.to_dict(),
    )


def power(facts: Facts, brief: Brief) -> str:
    """What validation can tell apart: the Ladder's first strategy step on this file.

    The bootstrap standard error of a Sharpe near zero is about
    :math:`1/\\sqrt{\\text{validation hours}}` per square-root hour; the
    Ladder raises it to its floor and then to the 1-2-5 grid. When the error
    is within a tenth of the next grid value, the strategy's own bootstrap
    decides which, and both are printed.
    """
    from blinded.ladder import floor, step

    hours = facts.spans["val"] / pd.Timedelta("1h")
    se = 1 / math.sqrt(hours)
    eta = step(max(se, floor("strategy", brief.stats.block)))
    steps = [eta]
    if se > 0.9 * eta:
        steps.append(step(eta * 1.0001))
    sharpe = " to ".join(_sharpe(s) for s in steps)
    yearly = " to ".join(f"{s * math.sqrt(_HOURS_PER_YEAR):.1f}" for s in steps)
    which = ", depending on its P&L's own bootstrap" if len(steps) > 1 else ""
    return (
        f"{words(facts.spans['val'])} of validation; a strategy releases only above {sharpe} "
        f"Sharpe per square-root hour ({yearly} annualised){which}: the bootstrap standard "
        f"error of a Sharpe near zero is about 1/sqrt(validation hours) = {se:.3g}, rounded "
        "up to the Ladder's 1-2-5 grid"
    )


def _times(ratio: float) -> str:
    if ratio >= 1000:
        return "over a thousand"
    return f"{ratio:,.0f}" if ratio >= 100 else f"{ratio:.1f}"


def _sharpe(s: float) -> str:
    return f"{s:.1f}" if s >= 0.1 else f"{s:g}"


def _cost_line(brief: Brief) -> str:
    return brief.data.cost_model.line()


def _ordered(names, order: tuple[str, ...]) -> list[str]:
    known = [n for n in order if n in names]
    return known + sorted(set(names) - set(order))


def _declared_pairs(spec) -> set[frozenset]:
    if spec is None:
        return set()
    return {frozenset(p) for p in (*spec.cross_venue, *spec.carry)}


def _more(facts: Facts, k: int, spec) -> str | None:
    """The fix when the brief's own instrument list is what falls short."""
    if spec is None or not spec.instruments or len(facts.available) < k:
        return None
    return (f"list at least {k} of the file's {len(facts.available)} instruments under "
            "archetypes.instruments, or leave the list out to use them all")  # fmt: skip


def _have(brief: Brief, name: str) -> str:
    """What the stream that would carry ``name`` has, for the fields refusal."""
    stream = "trades" if name in _TRADE_FIELDS else "book"
    spec = brief.data.streams.get(stream)
    have = ", ".join(f for f in (spec.fields if spec else {}) if f != "n") or "nothing"
    return f"the book has {have}" if stream == "book" else f"the trades have {have}"


# -- the decorators ---------------------------------------------------------


class Bound:
    """An archetype with its instruments: a decorator for a strategy, and a maker of both.

    ``Bound(fn)`` registers ``fn`` as a strategy, as ``bl.strategy`` does,
    with the family, legs and tag the archetype implies;
    ``.strategy(requires)`` does the same with its own requirements, and
    ``.feature(hypothesis)`` registers a feature.
    """

    def __init__(self, archetype: Archetype, legs: dict[str, str | None],
                 requires: Sequence[str] = ()):  # fmt: skip
        for role, value in legs.items():
            if callable(value):
                raise TypeError(
                    f"bl.{archetype.name} takes instruments, not a function: decorate with "
                    f"@bl.{archetype.name}(...), with the parentheses"
                )
            if value is not None and not isinstance(value, str):
                raise TypeError(f"bl.{archetype.name}: {role} must be an instrument name")
        self.archetype = archetype
        self.legs = dict(legs)
        self.requires = tuple(requires)

    def __call__(self, fn: Callable) -> Strategy:
        return self.strategy(self.requires)(fn)

    def strategy(self, requires: Sequence[str] = ()) -> Callable[[Callable], Strategy]:
        """Register ``(panel, features) -> position`` as a strategy of this archetype."""
        from blinded.contract import Strategy

        family, leg, pair, roles = self._expansion()

        def register(fn: Callable) -> Strategy:
            return Strategy(fn, list(requires), family, trades=leg, pair=pair,
                            archetype=self.archetype.name, roles=roles)  # fmt: skip

        return register

    def feature(self, hypothesis: str) -> Callable[[Callable], Feature]:
        """Register ``panel -> expression`` as a feature of this archetype."""
        from blinded.contract import Feature

        if callable(hypothesis):
            raise TypeError(
                f"{self.archetype.name}.feature needs a hypothesis: "
                '@x.feature("what this captures")'
            )
        family, leg, pair, roles = self._expansion()

        def register(fn: Callable) -> Feature:
            return Feature(fn, hypothesis, family, target=leg, pair=pair,
                           archetype=self.archetype.name, roles=roles)  # fmt: skip

        return register

    def _expansion(self) -> tuple[str | None, str | None, tuple[str, str] | None, dict]:
        """The ``family``, single leg, pair and roles that ``bl.feature``/``bl.strategy`` take."""
        arch, legs = self.archetype, self.legs
        if arch.mechanism == "pairs":
            return "pairs", None, (legs[arch.legs[0]], legs[arch.legs[1]]), {}
        if arch.mechanism == "lead_lag":
            return "lead_lag", legs["follower"], None, {"leader": legs["leader"]}
        if arch.template == "arb":
            return "directional", legs["quote"], None, {"hedge": legs["hedge"]}
        if arch.mechanism == "directional":
            return "directional", legs["instrument"], None, {}
        return arch.mechanism, None, None, {}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.legs.items())
        return f"<archetype {self.archetype.name}({args})>"


def lead_lag(leader: str, follower: str, requires: Sequence[str] = ()) -> Bound:
    """Lead-lag (speed): trade the follower on the leader's move.

    Expands to ``family="lead_lag"``, ``trades=follower`` (``target`` for a
    feature) and ``roles={"leader": leader}``.

    Examples
    --------
    >>> import blinded as bl
    >>> @bl.lead_lag(leader="PERP", follower="SPOT")
    ... def follow(panel, features):
    ...     return bl.threshold(features["gap"], z_in=2.0, z_out=0.5)
    >>> follow.family, follow.trades, follow.roles, follow.archetype
    ('lead_lag', 'SPOT', {'leader': 'PERP'}, 'lead_lag')
    """
    return Bound(REGISTRY["lead_lag"], {"leader": leader, "follower": follower}, requires)


def order_book_imbalance(instrument: str | None = None, requires: Sequence[str] = ()) -> Bound:
    """Order-book imbalance (speed): the bigger queue leans the next mid move its way.

    Directional on ``instrument``, the brief's target when None.
    """
    return Bound(REGISTRY["order_book_imbalance"], {"instrument": instrument}, requires)


def trade_flow_persistence(instrument: str | None = None, requires: Sequence[str] = ()) -> Bound:
    """Trade-flow persistence (speed): the sign of trades stays correlated while a large order
    is worked. Directional on ``instrument``, the brief's target when None."""
    return Bound(REGISTRY["trade_flow_persistence"], {"instrument": instrument}, requires)


def pairs_trading(a: str, b: str, requires: Sequence[str] = ()) -> Bound:
    """Pairs trading (relative value): two close substitutes drift apart and come back.

    Expands to ``family="pairs"``, ``pair=(a, b)``. In a brief with
    ``archetypes:``, a pair declared as one product on two venues or as a
    carry pair belongs to ``cross_venue_arbitrage`` or ``basis_trade``.

    Examples
    --------
    >>> import blinded as bl
    >>> @bl.pairs_trading(a="PERP", b="SPOT")
    ... def perp_spot(panel, features):
    ...     return bl.pairs("PERP", "SPOT", z_in=2.0, z_out=0.5)
    >>> perp_spot.family, perp_spot.pair, perp_spot.archetype
    ('pairs', ('PERP', 'SPOT'), 'pairs_trading')
    """
    return Bound(REGISTRY["pairs_trading"], {"a": a, "b": b}, requires)


def cross_venue_arbitrage(a: str, b: str, requires: Sequence[str] = ()) -> Bound:
    """Cross-venue arbitrage (speed): one product quoted at two prices on two venues.

    A pairs strategy on a pair the brief declares under
    ``archetypes.cross_venue``.
    """
    return Bound(REGISTRY["cross_venue_arbitrage"], {"a": a, "b": b}, requires)


def statistical_arbitrage(requires: Sequence[str] = ()) -> Bound:
    """Statistical arbitrage (relative value): the mean reversion of what is left of each name
    once common moves are taken out. A basket over every instrument of the brief."""
    return Bound(REGISTRY["statistical_arbitrage"], {}, requires)


def short_term_reversal(requires: Sequence[str] = ()) -> Bound:
    """Short-term reversal (liquidity): the names that fell hardest bounce. A basket over every
    instrument of the brief."""
    return Bound(REGISTRY["short_term_reversal"], {}, requires)


def basis_trade(spot: str, perp: str, requires: Sequence[str] = ()) -> Bound:
    """Basis trade (carry): long spot, short the perpetual, paid the funding rate.

    A pairs strategy, ``pair=(spot, perp)``, on a pair the brief declares
    under ``archetypes.carry``.
    """
    return Bound(REGISTRY["basis_trade"], {"spot": spot, "perp": perp}, requires)


def trend_following(instrument: str | None = None, requires: Sequence[str] = ()) -> Bound:
    """Trend following (underreaction): an asset's own past return predicts its next.

    Directional on ``instrument``, the brief's target when None.
    """
    return Bound(REGISTRY["trend_following"], {"instrument": instrument}, requires)


def market_making(instrument: str | None = None, requires: Sequence[str] = ()) -> Bound:
    """Market making (liquidity): quote both sides of ``instrument`` and earn the spread.

    A feature is directional on ``instrument`` (the brief's target when
    None), scored against its forward return. A strategy returns
    :func:`blinded.quote` rather than a position, and the fill engine
    (:mod:`blinded.sim_maker`) rests the quotes and fills them against the
    recorded book and trades. It runs only on a file with a touch stream
    (:func:`touch_lack`); on any other, a strategy of it is an ``error`` row
    with the refusal line.

    Examples
    --------
    >>> import blinded as bl
    >>> mm = bl.market_making(instrument="PERP")
    >>> @mm.strategy()
    ... def join(panel, features):
    ...     return bl.quote(bid=0, ask=0, size=1.0, skew=1.0, max_inventory=2.0)
    >>> join.family, join.trades, join.archetype
    ('directional', 'PERP', 'market_making')
    """
    return Bound(REGISTRY["market_making"], {"instrument": instrument}, requires)


def cross_venue_making(quote: str, hedge: str, requires: Sequence[str] = ()) -> Bound:
    """Quote the thin venue, hedge on the deep one (liquidity): a maker-taker arbitrage.

    A strategy returns :func:`blinded.quote` on ``quote`` (venue A), its
    offsets counted from ``fair``, a price off ``hedge`` (venue B); the sim
    (:func:`blinded.sim_maker.simulate_arb`) rests the quotes on A's touch
    stream, hedges every fill on B as a taker at its depth, and moves the coin
    and USDT between the two venues' declared stocks. A feature is directional
    on ``quote``, scored against its forward return. Runs only on a file with
    a depth file for ``hedge``, the two declared under
    ``archetypes.cross_venue``, and a stock block (:func:`arb_lack`).

    Examples
    --------
    >>> import blinded as bl
    >>> arb = bl.cross_venue_making(quote="AT_MEXC", hedge="AT_BINANCE")
    >>> @arb.strategy()
    ... def thirty(panel, features):
    ...     return bl.quote(bid=200, ask=200, size=64, skew=0, max_inventory=1e9)
    >>> thirty.family, thirty.trades, thirty.roles, thirty.archetype
    ('directional', 'AT_MEXC', {'hedge': 'AT_BINANCE'}, 'cross_venue_making')
    """
    return Bound(REGISTRY["cross_venue_making"], {"quote": quote, "hedge": hedge}, requires)


DECORATORS = {
    f.__name__: f
    for f in (lead_lag, order_book_imbalance, trade_flow_persistence, pairs_trading,
              cross_venue_arbitrage, statistical_arbitrage, short_term_reversal, basis_trade,
              trend_following, market_making, cross_venue_making)
}  # fmt: skip


# -- ARCHETYPES.md ----------------------------------------------------------

_SCORED = {
    "directional": "A feature is scored against {leg}'s forward return from the decision plus "
                   "the latency; a strategy returns one position in it.",
    "lead_lag": "A feature is scored against the follower's forward return; a strategy holds "
                "one position, in the follower.",
    "pairs": "A feature is scored against the forward log return of A less that of B, one "
             "notional each, so a high value means the spread of A over B is expected to rise; "
             "a strategy returns a position for each leg, in units of one notional of that "
             "leg, and each leg pays its own spread and fee.",
    "basket": "A feature returns one value per leg and is scored by its cross-sectional rank IC "
              "against the legs' forward returns; a strategy returns a position for each leg.",
    "carry": "A feature is scored against the spread return of spot over the perpetual plus "
             "the funding a short perpetual collects; a strategy returns a position for each "
             "leg, and the sim accrues funding at each settlement.",
}  # fmt: skip


def render(menu: Menu, brief: Brief) -> str:
    """``ARCHETYPES.md``: what runs on this data, how to write each, and what does not run."""
    primary = words(brief.target.primary_horizon)
    out = ["# ARCHETYPES", "", menu.headline]
    if menu.cost_model:
        out += ["", menu.cost_model[0].upper() + menu.cost_model[1:] + "."]
    shares = ", ".join(f"{EDGES[e].title} {menu.shares[e]:g}" for e in menu.edges)
    out += ["", f"Sessions are shared by edge: {shares}."]
    if menu.dropped_line:
        out += ["", menu.dropped_line[0].upper() + menu.dropped_line[1:] + "."]
    val = pd.Timedelta(seconds=menu.facts["spans_seconds"]["val"])
    longest = brief.grid.step * (val // brief.grid.step // PERIODS)
    out += [
        "",
        f"Power: {menu.power}. The span rule below, {PERIODS} of an archetype's shortest "
        "typical holdings in train and in validation, is a floor on sample size, not a "
        "promise of power.",
        "",
        "A strategy's noise is measured in blocks at least as long as its time-weighted "
        "mean hold on validation, since one hold is one bet however many blocks it spans. "
        f"A strategy whose holds average longer than {words(longest)} is never released: "
        f"validation would hold fewer than {PERIODS} of them.",
    ]
    for edge in menu.edges:
        e = EDGES[edge]
        out += ["", f"## {e.title[0].upper() + e.title[1:]}: {e.tag.lower()}", "", e.text]
        for name in menu.members(edge):
            out += ["", *_entry(REGISTRY[name], menu.verdicts[name], brief, primary, val,
                                menu.facts)]  # fmt: skip
    out += ["", "## Not runnable on this data", ""]
    refused = [f"- {menu.line(n)}" for n in menu.refused]
    out += refused or ["Every archetype runs on this data."]
    return "\n".join(out) + "\n"


def _entry(arch: Archetype, verdict: Verdict, brief: Brief, primary: str, val,
           facts: dict | None = None) -> list[str]:  # fmt: skip
    if arch.template == "maker":
        return _maker_entry(arch, verdict, brief, primary, (facts or {}).get("maker") or {})
    if arch.template == "arb":
        return _arb_entry(arch, verdict, brief, primary, (facts or {}).get("arb") or {})
    lo, hi = arch.holding
    need = PERIODS * lo
    call, legs = example(arch, brief)
    mechanism = "carry" if arch.template == "carry" else arch.mechanism
    scored = _SCORED[mechanism].format(leg=legs.get("instrument") or "the instrument")
    out = [
        f"### {arch.name}: {arch.title}",
        "",
        f"    x = bl.{call}",
        '    @x.feature("...")              # a feature',
        '    @x.strategy(requires=["..."])  # a strategy that uses it',
        "",
        f"- Edge: {arch.edge_text}",
        f"- Other side: {arch.other_side}",
        f"- How it dies: {arch.dies}",
    ]
    if arch.seen:
        out.append(f"- Seen in blinded: {arch.seen}")
    out += [
        f"- Typical holding: {words(lo)} to {words(hi)}. This run scores at {primary}; the sim "
        "holds as long as your strategy says.",
        f"- Span rule: {PERIODS} holding periods of {words(lo)} is {words(need)}; validation "
        f"is {words(val)}, {_times(val / need)} times that. It counts holding periods of the "
        "band's low end; how long a position lives on this data is yours and the report's to "
        "measure" + (" (DATA.md's pairs table has the half-life on train)."
                     if arch.mechanism == "pairs" else "."),
        f"- Built-ins: {', '.join(f'`{b}`' for b in arch.builtins)}.",
        f"- {scored}",
    ]  # fmt: skip
    if arch.template == "carry":
        out += _carry_lines(brief, (facts or {}).get("funding") or {},
                            (facts or {}).get("borrow") or {})  # fmt: skip
    out += [f"- Caveat: {c}." for c in verdict.caveats]
    out.append(f"- Read: {arch.read}.")
    return out


def _maker_entry(arch: Archetype, verdict: Verdict, brief: Brief, primary: str,
                 book: dict) -> list[str]:  # fmt: skip
    """Market making's entry: a strategy returns quotes, the book's facts on train, and what the
    judge scores."""
    lo, hi = arch.holding
    call, _ = example(arch, brief)
    ex = brief.execution
    fee = f"{ex.maker_fee_bps:g} bps a unit"
    known = words(ex.response)
    out = [
        f"### {arch.name}: {arch.title}",
        "",
        f"    x = bl.{call}",
        '    @x.feature("...")              # a feature: scored against the instrument\'s '
        "forward return",
        '    @x.strategy(requires=["..."])  # a strategy: returns bl.quote(...)',
        "",
        f"- Edge: {arch.edge_text}",
        f"- Other side: {arch.other_side}",
        f"- How it dies: {arch.dies}",
        f"- Typical holding: {words(lo)} to {words(hi)}. This run scores features at "
        f"{primary}; the sim holds inventory as long as your quotes leave it.",
        f"- Built-ins: {', '.join(f'`{b}`' for b in arch.builtins)}.",
        "- A strategy returns bl.quote(bid, ask, size, skew, max_inventory, lean, fair): offsets "
        "in ticks from the touch (0 joins, 1 rests behind, -1 improves and is rejected when it "
        "would cross), a size per side in units, and a skew in ticks per unit of inventory that "
        "the sim applies; a side is withdrawn while a full fill of it would take your "
        f"inventory past max_inventory. The sim knows your fills {known} after they happen, "
        f"and your orders reach the book {words(ex.latency)} after you decide.",
    ]  # fmt: skip
    wide = bool(book) and book.get("spread_ticks_median") is not None and (
        book["one_tick_share"] < 0.5)  # fmt: skip
    if wide:
        # a book many ticks wide (the arbitrage file's quote venue): the spread, not the tick,
        # is what a quote at the touch earns, and joining it is a choice among many prices
        tick = f"{book['tick']:g}" + ("" if book["tick_declared"] else ", read from the book")
        out.append(
            f"- This book, on train: wide. The spread's median is "
            f"{book['spread_ticks_median']:,.0f} ticks ({book['spread_bps_median']:.3g} bps; "
            f"one tick, {tick}, is {book['tick_bps']:.3g} bps), so an offset of 0 rests half "
            "that from the mid and any offset inside it is a price nobody else shows; the "
            f"touch holds about {_size(book['touch_median'])} units; the maker fee is {fee}. "
            + _joined(book)
        )  # fmt: skip
    elif book:
        tick = f"{book['tick']:g}" + ("" if book["tick_declared"] else ", read from the book")
        out.append(
            f"- This book, on train: one tick ({tick}) is {book['tick_bps']:.3g} bps and the "
            f"spread is one tick {_share(book['one_tick_share'])} of the time; the touch holds "
            f"about {_size(book['touch_median'])} units; the maker fee is {fee}. "
            + _joined(book)
        )  # fmt: skip
    else:
        out.append(f"- The maker fee is {fee}.")
    out.append(
        "- The judge runs four fill models and scores the worse of the two queue models "
        "(cancels behind you, or shared pro rata); the report prints all four. The engine "
        "assumes the market does not react to your quote: keep your size well under the "
        "displayed size (the capacity check prints the ratio)."
    )
    if wide:
        out.append("- Fees: on a book this wide the spread, not the maker fee, sets what a "
                   "fill earns; the report prints the rows at a rebate of 0.5 bps and at 0, 1 "
                   "and 2 bps, beside the brief's fee.")  # fmt: skip
    else:
        out.append("- Fees: on a book one tick wide the maker fee is the question, so the "
                   "report prints the rows at a rebate of 0.5 bps and at 0, 1 and 2 bps, beside "
                   "the brief's fee.")  # fmt: skip
    out += [f"- Caveat: {c}." for c in verdict.caveats]
    out.append(f"- Read: {arch.read}.")
    return out


def _arb_entry(arch: Archetype, verdict: Verdict, brief: Brief, primary: str,
               facts: dict) -> list[str]:  # fmt: skip
    """The arbitrage's entry: a strategy returns quotes priced
    off the hedge venue, the file's facts on train, and what the judge scores."""
    lo, hi = arch.holding
    call, legs = example(arch, brief)
    ex = brief.execution
    from blinded.sim_maker import hedge_terms

    lat, fee = hedge_terms(brief, legs["hedge"])
    delay = words(ex.response + lat)
    out = [
        f"### {arch.name}: {arch.title}",
        "",
        f"    x = bl.{call}",
        '    @x.feature("...")              # a feature: scored against the quote venue\'s '
        "forward return",
        '    @x.strategy(requires=["..."])  # a strategy: returns bl.quote(...) with fair= a '
        "price off the hedge venue",
        "",
        f"- Edge: {arch.edge_text}",
        f"- Other side: {arch.other_side}",
        f"- How it dies: {arch.dies}",
        f"- Typical holding: {words(lo)} to {words(hi)}, the hedge delay. This run scores "
        f"features at {primary}.",
        f"- Built-ins: {', '.join(f'`{b}`' for b in arch.builtins)}.",
        "- A strategy returns bl.quote(bid, ask, size, skew, max_inventory, fair=...): offsets "
        "in ticks from `fair` (the hedge venue's touch as fair=(bid, ask), its mid, or "
        "F.theo), a size per side in units; every fill is hedged by the sim on the hedge "
        f"venue {delay} later at its depth and taker fee of {fee:g} bps; the stock on each "
        "venue is declared and a side is withdrawn when its hedge cannot be funded.",
    ]  # fmt: skip
    if facts:
        n, hours = facts["fills"], facts["hours"]
        line = (f"- This file, on train: {facts['quote']}'s spread is "
                f"{facts['quote_spread_bps']:.3g} bps and {facts['hedge']}'s "
                f"{facts['hedge_spread_bps']:.3g} (medians over the grid); a quote "
                f"{facts['distance_bps']:g} bps off {facts['hedge']}'s touch at size "
                f"{facts['size']:g} filled {n:,} times in {hours:.3g} hours")  # fmt: skip
        if n:
            line += (f", captured {facts['captured_bps']:.3g} bps a fill, cost "
                     f"{-facts['slippage_bps']:.2g} in hedge slippage and "
                     f"{facts['hedge_fee_bps']:g} in hedge fee, and was picked off "
                     f"{_share(facts['picked_off_share'])} of the time")  # fmt: skip
        if facts.get("prints_a_day") is not None:
            line += (
                f"; the flow on {facts['quote']} is {facts['prints_a_day']:,.0f} prints a "
                f"day of {facts['print_usd_p50']:.2g} USD median, "
                f"{_share(facts['never_shown'])} of them at a price nobody displayed"
            )
        out.append(line + ".")
    out += [
        "- The judge runs four fill models and scores the worse of the two queue models; "
        "through fills are capped at the print; the report prints picked-off share, hedge "
        "cost, stock drift and the counterparty's signature. The engine assumes that flow "
        "does not change when you are there.",
        f"- Fees: the hedge's taker fee is {fee:g} bps; the report prints the rows at hedge "
        "fees of 10, 7.5, 5 and 0.",
    ]  # fmt: skip
    out += [f"- Caveat: {c}." for c in verdict.caveats]
    out.append(f"- Read: {arch.read}.")
    return out


def _share(x: float) -> str:
    """A share of the time in percent: whole numbers, two decimals above 99%, and never 100%
    for less than all of it."""
    if x >= 1:
        return "100%"
    if x > 0.99:
        return f"{100 * x:.2f}%" if 100 * x < 99.995 else "over 99.99%"
    return f"{100 * x:.0f}%"


def _size(x: float) -> str:
    return f"{x:,.0f}" if x >= 10 else f"{x:.2g}"


def _joined(book: dict) -> str:
    """What joining the touch at the archetype's own size did on train, before the fee."""
    size = f"{book['size']:g}"
    if not book["units"]:
        return f"Joining the touch at size {size} filled nothing on train."
    adverse = book["adverse_bps_10s_per_unit"]
    moved = (f"gave back {-adverse:.2g}" if adverse <= 0 else
             f"gained {adverse:.2g} more")  # fmt: skip
    return (f"Joining the touch at size {size} on train captured "
            f"{book['captured_bps_per_unit']:.2g} bps a unit at the fill and {moved} bps a unit "
            "within 10 s, before the fee.")  # fmt: skip


def _carry_lines(brief: Brief, funding: dict, borrow: dict | None = None) -> list[str]:
    """What carry there was on train, what being short spot cost there, and what a round trip
    in both legs costs in fees."""
    out = []
    if funding:
        each = "; ".join(
            f"{perp} {f['mean_rate_bps']:.3g} bps ({f['yearly_pct']:.3g}% a year)"
            for perp, f in funding.items()
            if f["mean_rate_bps"] is not None
        )
        n = sorted({f["settlements"] for f in funding.values()})
        count = f"{n[0]:,}" if len(n) == 1 else f"{n[0]:,} to {n[-1]:,}"
        out.append(f"- Funding on train, the mean rate per settlement and its sum as a rate a "
                   f"year ({count} settlements each): {each}. A short perpetual collects a "
                   "positive rate.")  # fmt: skip
    ex = brief.execution
    fees = sorted({(ex.fee(s), ex.fee(p)) for s, p in brief.archetypes.carry})
    if len(fees) == 1:
        ((fs, fp),) = fees
        out.append(f"- Fees: {fs:g} bps a unit on spot and {fp:g} on the perpetual, so a round "
                   f"trip in both legs costs 2 x ({fs:g} + {fp:g}) = {2 * (fs + fp):g} bps "
                   "before any spread or basis move.")  # fmt: skip
    if borrow:
        each = "; ".join(f"{s} {b['mean_bps_per_day']:.3g} bps ({b['max_bps_per_day']:.3g} at "
                         "most)" for s, b in borrow.items())  # fmt: skip
        out.append(f"- Borrow: the reverse trade, short spot and long the perpetual, pays to "
                   f"borrow the coin, `SYM.borrow_rate` as-of each step, for as long as it is "
                   f"held. On train, the mean rate a day in force: {each}. Funding settles "
                   "three times a day, so the reverse trade gains only where three settlements "
                   "paid shorts more than this.")  # fmt: skip
    return out


def example(arch: Archetype, brief: Brief) -> tuple[str, dict]:
    """The decorator call with instruments from the brief, and the legs it used."""
    spec = brief.archetypes
    inst = list(spec.instruments) if spec is not None and spec.instruments else []
    inst = inst or [brief.target.instrument]
    target = brief.target.instrument if brief.target.instrument in inst else inst[-1]
    legs: dict[str, str] = {}
    if arch.legs == ("instrument",):
        legs = {"instrument": target}
    elif arch.legs == ("leader", "follower"):
        leader = next((s for s in inst if s != target), target)
        legs = {"leader": leader, "follower": target}
    elif arch.template == "arb":
        pair = arb_pair(brief) or (spec.cross_venue[0] if spec is not None and spec.cross_venue
                                    else (target, target))  # fmt: skip
        legs = dict(zip(arch.legs, pair, strict=True))
    elif arch.name == "cross_venue_arbitrage" and spec is not None and spec.cross_venue:
        legs = dict(zip(arch.legs, spec.cross_venue[0], strict=True))
    elif arch.name == "basis_trade" and spec is not None and spec.carry:
        legs = dict(zip(arch.legs, spec.carry[0], strict=True))
    elif arch.legs:
        taken = _declared_pairs(spec)
        free = [p for p in itertools.combinations(inst, 2) if frozenset(p) not in taken]
        legs = dict(zip(arch.legs, free[0], strict=True)) if free else {}
    args = ", ".join(f'{k}="{v}"' for k, v in legs.items())
    return f"{arch.name}({args})", legs
