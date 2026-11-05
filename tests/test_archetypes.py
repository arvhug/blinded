"""The archetype registry, its decorators, and the menu on six datasets (Table B).

The six demo datasets are built here as briefs only, with the windows
`blinded init` fixes, their instruments, streams and fields: no data is read,
which is what the check promises. All six exist: the Tardis day and the three
Binance days, windows as their runs fixed; the cross-venue, cross-section,
carry and trend files, windows as `blinded init` fixes them on the built file.
"""

import copy
import dataclasses
import math
import re

import pandas as pd
import pytest
import yaml

import blinded as bl
from blinded import archetypes, vault
from blinded import brief as briefs
from blinded.archetypes import EDGES, REGISTRY, Bound
from blinded.contract import ContractError, bind

known = pytest.mark.known_answer
cross_check = pytest.mark.cross_check
edge = pytest.mark.edge_case

# bl.__all__ before the archetypes: no decorator may shadow one of these
BEFORE = {
    "Brief", "Feature", "Panel", "Result", "Strategy", "StrategyResult", "Window", "evaluate",
    "evaluate_strategy", "feature", "features", "linear", "load", "load_brief", "pairs",
    "strategy", "threshold",
}  # fmt: skip


# -- the registry and the decorators ----------------------------------------


@known
def test_eleven_archetypes_in_five_edges_with_the_pages_text():
    assert len(REGISTRY) == 11
    assert sorted(a.n for a in REGISTRY.values()) == list(range(1, 12))
    assert list(EDGES) == ["speed", "liquidity", "relative_value", "underreaction", "carry"]
    assert {a.edge for a in REGISTRY.values()} == set(EDGES)  # each edge has an archetype
    for name, a in REGISTRY.items():
        assert name == a.name and name.isidentifier() and name not in BEFORE
        assert name in bl.__all__ and getattr(bl, name) is archetypes.DECORATORS[name]
        for text in (a.title, a.edge_text, a.other_side, a.dies, a.read, a.capacity):
            assert text.strip()
        assert a.seen is None or a.seen.strip()
        assert a.holding[0] < a.holding[1]
        assert set(a.checks) <= set(archetypes.DEATHS)
    # three of the page's entries say what blinded saw
    assert [n for n, a in REGISTRY.items() if a.seen] == [
        "lead_lag", "order_book_imbalance", "pairs_trading",
    ]  # fmt: skip
    # reversal's band is the revised 4 h to 30 days; the other bands are the page's
    assert REGISTRY["short_term_reversal"].holding == (pd.Timedelta("4h"), pd.Timedelta("30D"))
    assert REGISTRY["trend_following"].holding == (pd.Timedelta("14D"), pd.Timedelta("365D"))
    # market making is directional on its instrument; its strategies go to the fill engine
    assert (REGISTRY["market_making"].mechanism, REGISTRY["market_making"].template) == (
        "directional", "maker")  # fmt: skip
    assert REGISTRY["market_making"].needs.touch
    # the arbitrage quotes one venue and hedges on another: directional on the quote venue,
    # its strategies to the arbitrage sim, which needs the hedge venue's depth
    arb = REGISTRY["cross_venue_making"]
    assert (arb.edge, arb.mechanism, arb.template, arb.legs) == (
        "liquidity", "directional", "arb", ("quote", "hedge"))  # fmt: skip
    assert arb.needs.touch and arb.needs.depth and arb.needs.venues == 2
    assert arb.params == {"distance_bps": 30.0, "size": 64.0}
    assert arb.checks == ("costs", "capacity", "worst_window", "mae", "picked_off", "hedge",
                          "stock", "counterparty")  # fmt: skip
    s = bl.cross_venue_making(quote="A", hedge="B")(body)
    assert (s.family, s.trades, s.roles, s.archetype) == ("directional", "A", {"hedge": "B"},
                                                          "cross_venue_making")  # fmt: skip


def body(panel, features):
    return None


@known
def test_each_decorator_is_the_strategy_or_feature_it_stands_for():
    s = bl.pairs_trading(a="A", b="B", requires=["z"])(body)
    assert isinstance(s, bl.Strategy)
    assert (s.family, s.pair, s.archetype, s.requires) == ("pairs", ("A", "B"),
                                                           "pairs_trading", ["z"])  # fmt: skip
    s = bl.lead_lag(leader="L", follower="F")(body)
    assert (s.family, s.trades, s.roles) == ("lead_lag", "F", {"leader": "L"})
    f = bl.lead_lag(leader="L", follower="F").feature("L's move, not yet in F")(body)
    assert isinstance(f, bl.Feature)
    assert (f.family, f.target, f.roles, f.archetype) == ("lead_lag", "F", {"leader": "L"},
                                                          "lead_lag")  # fmt: skip
    s = bl.statistical_arbitrage()(body)
    assert (s.family, s.legs, s.archetype) == ("basket", (), "statistical_arbitrage")
    s = bl.basis_trade(spot="S", perp="P")(body)
    assert (s.family, s.pair) == ("pairs", ("S", "P"))
    obi = bl.order_book_imbalance()
    s = obi.strategy(requires=["imb"])(body)
    assert (s.family, s.trades, s.requires) == ("directional", None, ["imb"])
    assert bl.trend_following(instrument="X").feature("h")(body).target == "X"
    s = bl.market_making(instrument="X")(body)
    assert (s.family, s.trades, s.archetype) == ("directional", "X", "market_making")
    f = bl.market_making(instrument="X").feature("h")(body)
    assert (f.family, f.target, f.archetype) == ("directional", "X", "market_making")
    # bl.feature and bl.strategy set neither tag
    assert (bl.strategy()(body).archetype, bl.strategy()(body).roles) == (None, {})


@edge
def test_a_decorator_without_its_parentheses_or_a_hypothesis_is_refused():
    with pytest.raises(TypeError, match="with the parentheses"):
        bl.order_book_imbalance(body)
    with pytest.raises(TypeError, match="needs a hypothesis"):
        bl.order_book_imbalance().feature(body)
    assert isinstance(bl.order_book_imbalance("X"), Bound)


TWINS = """
import blinded as bl


@bl.strategy(family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
def bare(panel, features):
    return bl.pairs("XRPUSDT_PERP", "XRPUSDT", z_in=2.0, z_out=0.5, window="2m", every="2m")


@bl.pairs_trading(a="XRPUSDT_PERP", b="XRPUSDT")
def twin(panel, features):
    return bl.pairs("XRPUSDT_PERP", "XRPUSDT", z_in=2.0, z_out=0.5, window="2m", every="2m")
"""


@cross_check
def test_a_pairs_trading_twin_scores_as_the_bare_pairs_strategy(pairs_brief, tmp_path):
    # a legacy families brief: the archetype decorator is sugar, so the sim rows of the two
    # spellings agree on train and validation
    path = tmp_path / "twins.py"
    path.write_text(TWINS)
    bare, twin = bl.load(path).strategies
    for window in ("train", "val"):
        a = bl.evaluate_strategy(bare, pairs_brief, window=window, features={})
        b = bl.evaluate_strategy(twin, pairs_brief, window=window, features={})
        assert a.ok and b.ok and (b.archetype, b.edge) == ("pairs_trading", "relative_value")
        assert a.rows.height == b.rows.height == 5
        for row_a, row_b in zip(a.rows.iter_rows(named=True), b.rows.iter_rows(named=True),
                                strict=True):  # fmt: skip
            for key, value in row_a.items():
                if isinstance(value, float) and math.isfinite(value):
                    assert row_b[key] == pytest.approx(value, abs=1e-12, rel=0), key
                else:
                    assert row_b[key] == value or (value != value and row_b[key] != row_b[key])
        assert "pairs_trading (relative value), pairs on XRPUSDT_PERP/XRPUSDT" in repr(b)


# -- six datasets as briefs -------------------------------------------------

XS20 = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
        "AVAXUSDT", "LINKUSDT", "DOTUSDT", "ARBUSDT", "LTCUSDT", "TRXUSDT", "ATOMUSDT",
        "UNIUSDT", "NEARUSDT", "APTUSDT", "FILUSDT", "ETCUSDT", "BCHUSDT"]  # fmt: skip
CARRY8 = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LINK"]
L1_BOOK = {"bid": "bid0_px", "ask": "ask0_px", "bid_qty": "bid0_qty", "ask_qty": "ask0_qty"}
L1_TRADES = {"price": "price", "qty": "qty", "side": "side"}
SIDES = {"buy": "B", "sell": "S"}


def kline_streams(funding=False):
    book = {"bid": "bid0_px", "ask": "ask0_px"}
    trades = {"price": "price", "qty": "qty", "side": "side", "n": "n"}
    out = {
        "book": {"filter": "kind == 'book'", "fields": book},
        "trades": {"filter": "kind == 'trades'", "side_values": SIDES, "fields": trades},
    }
    if funding:
        out["funding"] = {"filter": "kind == 'funding'", "fields": {"rate": "rate"}}
    return out


COST = {"kind": "half_spread", "fill": "next bar close",
        "source": "data.binance.vision bookTicker 2024-03-27"}  # fmt: skip


def windows(train, val, test):
    return {"train": "/".join(train), "val": "/".join(val), "test": "/".join(test)}


DATASETS = {
    # the Tardis day: windows as runs/tardis-sonnet fixed them
    "a": dict(name="tardis-binance-xrpusdt-2026-10-01", instruments=["XRPUSDT", "XRPUSDT_PERP"],
              streams={"book": {"filter": "is_trade == false", "fields": L1_BOOK},
                       "trades": {"filter": "is_trade == true", "fields": L1_TRADES,
                                  "side_values": SIDES}},
              step="100ms", latency=50, delay=0, horizons=["100ms", "1s", "10s", "60s"],
              primary="10s", embargo="5min", block="5min", warmup="5min",
              windows=windows(("2026-10-01T00:00:00.1Z", "2026-10-01T07:00:00Z"),
                              ("2026-10-01T07:05:00Z", "2026-10-01T10:30:00Z"),
                              ("2026-10-01T10:35:00Z", "2026-10-01T14:00:00Z"))),
    # the three Binance days: windows as runs/binance-fake fixed them
    "b": dict(name="binance-um-btc-eth-sol-xrp-2024-03-27-3d-500ms",
              instruments=["BTCUSDT_PERP", "ETHUSDT_PERP", "SOLUSDT_PERP", "XRPUSDT_PERP"],
              streams={"book": {"filter": "is_trade == false",
                                "fields": L1_BOOK | {"n": "n"}},
                       "trades": {"filter": "is_trade == true", "fields": L1_TRADES | {"n": "n"},
                                  "side_values": SIDES}},
              step="500ms", latency=50, delay=50, horizons=["500ms", "1s", "10s", "60s"],
              primary="10s", embargo="5min", block="5min", warmup="5min",
              thinned={"step": "500ms", "offsets_ms": [-50, 0, 50]},
              windows=windows(("2024-03-27T00:00:00Z", "2024-03-28T12:00:00Z"),
                              ("2024-03-28T12:05:00Z", "2024-03-29T06:00:00Z"),
                              ("2024-03-29T06:05:00Z", "2024-03-30T00:00:00.5Z"))),
    # the XRP perpetual on Binance and Bybit, the Tardis day's windows
    "c": dict(name="tardis-binance-bybit-xrpusdt-perp-2026-10-01-14h",
              instruments=["XRPUSDT_PERP", "XRPUSDT_PERP_BYBIT"],
              cross_venue=[["XRPUSDT_PERP", "XRPUSDT_PERP_BYBIT"]],
              streams={"book": {"filter": "is_trade == false", "fields": L1_BOOK},
                       "trades": {"filter": "is_trade == true", "fields": L1_TRADES,
                                  "side_values": SIDES}},
              step="100ms", latency=50, delay=0, horizons=["100ms", "1s", "10s", "60s"],
              primary="10s", embargo="5min", block="5min", warmup="5min",
              windows=windows(("2026-10-01T00:00:00.1Z", "2026-10-01T07:00:00Z"),
                              ("2026-10-01T07:05:00Z", "2026-10-01T10:30:00Z"),
                              ("2026-10-01T10:35:00Z", "2026-10-01T14:00:00Z"))),
    # twenty perpetuals on 15-minute bars: the windows blinded init fixes on the built
    # file, grid 2024-04-01 00:00 to 2025-06-24 23:45. Section 5.5's 110.5 days of validation
    # end at 12:00; brief.windows rounds the end of validation to 11:45
    "d": dict(name="binance-um-xs20-2024-04-01-450d-15m",
              instruments=[f"{s}_PERP" for s in XS20], streams=kline_streams(),
              step="15min", latency=900_000, delay=0, horizons=["15m", "1h", "4h", "1d"],
              primary="4h", embargo="2d", block="4h", warmup="1d", cost_model=COST,
              windows=windows(("2024-04-01T00:00:00Z", "2024-11-12T00:00:00Z"),
                              ("2024-11-14T00:00:00Z", "2025-03-04T11:45:00Z"),
                              ("2025-03-06T11:45:00Z", "2025-06-25T00:00:00Z"))),
    # eight coins' spot and perpetual with funding, the same 450 days on the same grid:
    # blinded init on the built file fixes these windows too
    "e": dict(name="binance-carry8-2024-04-01-450d-15m",
              instruments=[s for c in CARRY8 for s in (f"{c}USDT", f"{c}USDT_PERP")],
              carry=[[f"{c}USDT", f"{c}USDT_PERP"] for c in CARRY8],
              streams=kline_streams(funding=True), step="15min", latency=900_000, delay=0,
              horizons=["1h", "8h", "1d"], primary="8h", embargo="2d", block="8h",
              warmup="1d", cost_model=COST,
              windows=windows(("2024-04-01T00:00:00Z", "2024-11-12T00:00:00Z"),
                              ("2024-11-14T00:00:00Z", "2025-03-04T11:45:00Z"),
                              ("2025-03-06T11:45:00Z", "2025-06-25T00:00:00Z"))),
    # four perpetuals on hourly bars: windows from brief.windows on 2023-05-17 00:00 to
    # 2026-09-30 23:00 with a 3-week embargo (validation 287 d 5 h, not an early 297.75 d)
    "f": dict(name="binance-um-trend4-2023-05-17-1233d-1h",
              instruments=["BTCUSDT_PERP", "ETHUSDT_PERP", "XRPUSDT_PERP", "LTCUSDT_PERP"],
              streams=kline_streams(), step="1h", latency=3_600_000, delay=0,
              horizons=["1d", "1w", "2w"], primary="1w", embargo="3w", block="2w",
              warmup="4w", cost_model=COST, windows=None,
              span=("2023-05-17T00:00:00Z", "2026-09-30T23:00:00Z")),
}  # fmt: skip


def dataset_raw(key: str, **archetypes_block) -> dict:
    d = DATASETS[key]
    block = {"instruments": d["instruments"]}
    for name in ("cross_venue", "carry"):
        if name in d:
            block[name] = d[name]
    data = {"path": "x.parquet", "instrument_column": "symbol", "streams": d["streams"],
            "availability": {"default_delay_ms": d["delay"]}}  # fmt: skip
    if "thinned" in d:
        data["thinned"] = d["thinned"]
    if "cost_model" in d:
        data["cost_model"] = d["cost_model"]
    raw = {
        "name": d["name"],
        "data": data,
        "target": {"instrument": d["instruments"][-1], "horizons": d["horizons"],
                   "primary_horizon": d["primary"]},
        "grid": {"step": d["step"], "warmup": d["warmup"]},
        "execution": {"latency_ms": d["latency"], "taker_fee_bps": 5.0},
        "splits": {"embargo": d["embargo"]},
        "stats": {"block": d["block"]},
        "archetypes": block | archetypes_block,
    }  # fmt: skip
    if d["windows"]:
        raw["splits"]["windows"] = d["windows"]
    return copy.deepcopy(raw)


def dataset(key: str, **archetypes_block):
    """The brief and facts of a dataset, from the brief alone."""
    brief = briefs.parse(dataset_raw(key, **archetypes_block))
    wins = None
    if DATASETS[key]["windows"] is None:
        first, last = (pd.Timestamp(t) for t in DATASETS[key]["span"])
        wins = briefs.windows(brief, first, last)
    return brief, archetypes.facts(brief, windows=wins)


# Table B: R runnable, C runnable with a caveat (its classes), X refused (its class)
TABLE_B = {
    "lead_lag":               ("R", "C grid", "R", "C grid, cost", "C grid, cost", "C grid, cost"),
    "order_book_imbalance":   ("R", "R", "R", "X fields", "X fields", "X fields"),
    "trade_flow_persistence": ("R", "R", "R", "C cost", "C cost", "C cost"),
    "pairs_trading":          ("R", "R", "X instruments", "C cost", "C cost", "C cost"),
    "cross_venue_arbitrage":  ("X venues", "X venues", "R", "X venues", "X venues", "X venues"),
    "statistical_arbitrage":  ("X cross_section",) * 3 + ("C cost",) + ("X cross_section",) * 2,
    "short_term_reversal":    ("X cross_section",) * 3 + ("C cost",) + ("X cross_section",) * 2,
    "basis_trade":            ("X stream",) * 4 + ("C cost", "X stream"),
    "trend_following":        ("X span",) * 5 + ("C cost",),
    "market_making":          ("R", "X sim", "R", "X sim", "X sim", "X sim"),
    "cross_venue_making":     ("X sim",) * 6,
}  # fmt: skip
RUNNABLE = {"a": (5, 3), "b": (4, 2), "c": (5, 2), "d": (5, 3), "e": (4, 3), "f": (4, 3)}


def cell(verdict) -> str:
    if verdict.status == "runnable":
        return "R"
    return f"{'C' if verdict.status == 'caveat' else 'X'} {verdict.cls}"


@known
@pytest.mark.parametrize("key", list(DATASETS))
def test_the_menu_on_each_dataset_is_its_column_of_table_b(key):
    brief, facts = dataset(key)
    menu = archetypes.menu(brief, facts)
    column = "abcdef".index(key)
    assert {n: cell(v) for n, v in menu.verdicts.items()} == {
        n: row[column] for n, row in TABLE_B.items()
    }
    assert (len(menu.runnable), len(menu.edges)) == RUNNABLE[key]


@known
@pytest.mark.parametrize("key", list(DATASETS))
def test_a_touch_stream_is_a_file_that_keeps_every_book_row_and_trade(key):
    # the Tardis day and the cross-venue day keep every book row with its sizes and every
    # trade; the Binance days are thinned to the read lattice and the other three are bars
    _, facts = dataset(key)
    assert facts.touch is (key in ("a", "c"))
    # a declared touch stream is one whatever the data file is
    raw = dataset_raw(key)
    raw["data"]["touch"] = {"path": "x.touch.parquet", "tick_size": 0.01}
    brief = briefs.parse(raw)
    wins = None
    if not brief.splits.fixed:
        first, last = (pd.Timestamp(t) for t in DATASETS[key]["span"])
        wins = briefs.windows(brief, first, last)
    assert archetypes.facts(brief, windows=wins).touch


TOUCH = ("market_making (liquidity & flow): not runnable on @: needs the book at every change "
         "with its sizes and each trade at its own price and stamp (a touch stream); the file "
         "has ")  # fmt: skip
THINNED = TOUCH + "a book thinned to the read lattice and trades summed per step."
BARS = TOUCH + "bars."
VENUES = ("cross_venue_arbitrage (speed & information): not runnable on @: needs the same "
          "instrument on 2 venues; the file has one.")  # fmt: skip
FUNDING = "basis_trade (carry): not runnable on @: needs funding data; the file has book, trades."
DEPTH = ("cross_venue_making (liquidity & flow): not runnable on @: needs the hedge venue's depth "
         "at every change (a depth file); the file has none.")  # fmt: skip
QTY = "order_book_imbalance (speed & information): not runnable on @: needs bid_qty; the book has bid, ask."  # noqa: E501


def xs(n):
    return [f"{name} ({title}): not runnable on @: needs a cross-section of at least 20 names; "
            f"the file has {n}."
            for name, title in (("statistical_arbitrage", "relative value"),
                                ("short_term_reversal", "liquidity & flow"))]  # fmt: skip


def trend(val):
    return (f"trend_following (underreaction): not runnable on @: needs 40 weeks of validation "
            f"(20 holding periods of 2 weeks); validation is {val}.")  # fmt: skip


REFUSED = {
    "a": [VENUES, *xs(2), FUNDING, trend("3 h 25 min"), DEPTH],
    "b": [VENUES, *xs(4), FUNDING, trend("17 h 55 min"), THINNED, DEPTH],
    "c": ["pairs_trading (relative value): not runnable on @: needs a pair that is not one "
          "product on two venues; XRPUSDT_PERP and XRPUSDT_PERP_BYBIT are "
          "(cross_venue_arbitrage).", *xs(2), FUNDING, trend("3 h 25 min"), DEPTH],
    "d": [QTY, VENUES, FUNDING, trend("110 d 11 h 45 min"), BARS, DEPTH],
    "e": [QTY, VENUES, *xs(16), trend("110 d 11 h 45 min"), BARS, DEPTH],
    "f": [QTY, VENUES, *xs(4), FUNDING, BARS, DEPTH],
}  # fmt: skip


@known
@pytest.mark.parametrize("key", list(DATASETS))
def test_each_refused_archetype_prints_the_spec_line(key):
    brief, facts = dataset(key)
    menu = archetypes.menu(brief, facts)
    expected = [line.replace("@", brief.name) for line in REFUSED[key]]
    # registry order: the n of each refused archetype, which the lists above follow
    assert [menu.line(n) for n in menu.refused] == sorted(
        expected, key=lambda text: REGISTRY[text.split(" ", 1)[0]].n
    )


@known
def test_the_grid_caveat_on_the_binance_days():
    brief, facts = dataset("b")
    verdict = archetypes.menu(brief, facts).verdicts["lead_lag"]
    assert verdict.caveats == (
        "the grid step 500 ms is above the typical holding (100 ms); what the judge measures "
        "is the first whole step after the fill",
    )


@known
def test_pairs_passes_the_span_rule_on_the_tardis_day_by_five_minutes():
    # 20 x 10 min = 200 min <= 205 min of validation; with it 6 minutes shorter, 199 < 200
    brief, facts = dataset("a")
    assert archetypes.check(REGISTRY["pairs_trading"], facts, brief).status == "runnable"
    raw = dataset_raw("a")
    raw["splits"]["windows"]["val"] = "2026-10-01T07:05:00Z/2026-10-01T10:24:00Z"
    short = briefs.parse(raw)
    verdict = archetypes.check(REGISTRY["pairs_trading"], archetypes.facts(short), short)
    assert archetypes.line(REGISTRY["pairs_trading"], short.name, verdict) == (
        "pairs_trading (relative value): not runnable on tardis-binance-xrpusdt-2026-10-01: "
        "needs 3 h 20 min of validation (20 holding periods of 10 min); validation is 3 h 19 min."
    )


@known
def test_basis_trade_without_carry_pairs_declares_none():
    raw = dataset_raw("e")
    del raw["archetypes"]["carry"]
    brief = briefs.parse(raw)
    verdict = archetypes.check(REGISTRY["basis_trade"], archetypes.facts(brief), brief)
    assert (verdict.cls, verdict.reason) == (
        "instruments",
        "needs a spot and perpetual pair under archetypes.carry; the brief declares none",
    )


@known
@pytest.mark.parametrize("key", ["d", "e", "f"])
def test_every_runnable_archetype_on_a_bar_file_carries_the_cost_caveat(key):
    brief, facts = dataset(key)
    menu = archetypes.menu(brief, facts)
    cost = ("fills are a model on this file: the next bar's close plus or minus a measured "
            "half spread (DATA.md)")  # fmt: skip
    assert menu.runnable and all(cost in menu.verdicts[n].caveats for n in menu.runnable)
    assert menu.cost_model.startswith("fills are a model: the next bar's close")
    # the L1 files have no fill model, so no cost caveat
    brief, facts = dataset("a")
    assert archetypes.menu(brief, facts).cost_model is None


@known
def test_the_power_line_on_four_files():
    # 1/sqrt(hours) of validation against the strategy floor 0.05 sqrt(5 min / block), then
    # the 1-2-5 grid. (a) 1/sqrt(3.4167) = 0.541 -> 1.0, x sqrt(8760) = 93.6 annualised.
    # (b) 1/sqrt(17.917) = 0.236 -> 0.5, 46.8. (d) 1/sqrt(2651.75) = 0.0194, above the floor
    # 0.05 sqrt(5/240) = 0.0072 and within a tenth of 0.02, so 0.02 to 0.05, 1.9 to 4.7.
    # (f) 287 d 5 h = 6893 h: 1/sqrt(6893) = 0.0120 -> 0.02, 1.9 annualised.
    expect = {
        "a": ("3 h 25 min of validation; a strategy releases only above 1.0 Sharpe per "
              "square-root hour (93.6 annualised): ", "= 0.541,"),
        "b": ("17 h 55 min of validation; a strategy releases only above 0.5 Sharpe per "
              "square-root hour (46.8 annualised): ", "= 0.236,"),
        "d": ("110 d 11 h 45 min of validation; a strategy releases only above 0.02 to 0.05 "
              "Sharpe per square-root hour (1.9 to 4.7 annualised), depending on its P&L's own "
              "bootstrap: ", "= 0.0194,"),
        "f": ("287 d 5 h of validation; a strategy releases only above 0.02 Sharpe per "
              "square-root hour (1.9 annualised): ", "= 0.012,"),
    }  # fmt: skip
    for key, (head, se) in expect.items():
        brief, facts = dataset(key)
        text = archetypes.power(facts, brief)
        assert text.startswith(head) and se in text, (key, text)


# -- shares, parsing and init ----------------------------------------------


@known
def test_a_share_for_an_edge_with_nothing_to_run_is_dropped_and_the_rest_renormalised():
    # carry has nothing on the Binance days: 0.5 / 0.8 = 0.625 and 0.3 / 0.8 = 0.375
    brief, facts = dataset("b", shares={"speed": 0.5, "relative_value": 0.3, "carry": 0.2})
    menu = archetypes.menu(brief, facts)
    assert menu.shares == {"speed": 0.625, "relative_value": 0.375}
    assert menu.dropped == ("carry",)
    assert menu.dropped_line == (
        "shares for carry dropped: no runnable archetype; the rest renormalised to "
        "speed & information 0.625, relative value 0.375"
    )
    assert "Shares for carry dropped" in archetypes.render(menu, brief)
    # without shares the two edges split equally
    brief, facts = dataset("b")
    assert archetypes.menu(brief, facts).shares == {"speed": 0.5, "relative_value": 0.5}


@edge
@pytest.mark.parametrize(
    "change, message",
    [
        (
            {"archetypes": {"shares": {"speed": 0.5, "relative_value": 0.3, "volatility": 0.2}}},
            "volatility is not an edge",
        ),
        ({"archetypes": {"shares": {"speed": 0.5, "relative_value": 0.6}}}, "sum to 1"),
        ({"archetypes": {"only": ["market_maker"]}}, "not archetypes"),
        ({"archetypes": {"params": {"pairs": {}}}}, "not archetypes"),
        ({"archetypes": {"carry": [["BTCUSDT", "BTCUSDT"]]}}, "two different instruments"),
        ({"archetypes": {}, "families": {"directional": {}}}, "not both"),
    ],
)
def test_bad_archetypes_blocks_fail_at_parse(change, message):
    raw = dataset_raw("b")
    raw.update(change)
    with pytest.raises(ValueError, match=message):
        briefs.parse(raw)


@known
def test_an_archetypes_brief_implies_the_mechanism_families():
    brief, _ = dataset("e")
    by = {f.name: f for f in brief.families}
    assert list(by) == ["directional", "lead_lag", "pairs"]  # 16 names: no basket
    assert by["directional"].instruments == by["lead_lag"].leaders == by["lead_lag"].followers
    # every unordered pair of the 16 in list order, the carry pairs among them
    assert len(by["pairs"].pairs) == 16 * 15 // 2
    assert by["pairs"].pairs[0] == ("BTCUSDT", "BTCUSDT_PERP")
    assert by["pairs"].defaults == {"window": "2h", "every": "30m", "z_in": 2.0, "z_out": 0.5}
    brief, _ = dataset("d")
    assert [f.name for f in brief.families][-1] == "basket"  # twenty names make a basket
    assert brief.has_archetypes and not brief.has_families


@edge
def test_only_market_making_fails_init_and_writes_nothing(long_brief, tmp_path):
    # the two-hour file keeps every book row and trade, but without the book's sizes there is
    # no queue to stand in
    raw = yaml.safe_load(long_brief.read_text())
    raw["archetypes"] = {"only": ["market_making"]}
    raw["data"]["streams"]["book"]["fields"] = {"bid": "bid0_px", "ask": "ask0_px"}
    (tmp_path / "brief.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=re.escape(
            "archetypes.only names what this file cannot run: market_making (liquidity & "
            "flow): not runnable on xrp-l1-trades: needs the book at every change with its "
            "sizes and each trade at its own price and stamp (a touch stream); the file has "
            "no bid_qty.")):  # fmt: skip
        vault.init(tmp_path / "brief.yaml", tmp_path / "run")
    assert not (tmp_path / "run").exists()


@known
def test_only_leaves_the_rest_refused_with_a_fix(long_brief, tmp_path):
    raw = yaml.safe_load(long_brief.read_text())
    raw["archetypes"] = {"only": ["order_book_imbalance"]}
    (tmp_path / "brief.yaml").write_text(yaml.safe_dump(raw))
    run = vault.init(tmp_path / "brief.yaml", tmp_path / "run")
    menu = run.menu()
    assert menu.runnable == ("order_book_imbalance",)
    assert menu.line("lead_lag") == (
        "lead_lag (speed & information): not runnable on xrp-l1-trades: left out by "
        "archetypes.only. Change the brief: add lead_lag to archetypes.only."
    )
    # init wrote the file's instruments into the run's brief
    assert run.brief().archetypes.instruments == ("XRPUSDT", "XRPUSDT_PERP")


# -- binding -------------------------------------------------------------------


def three_venues():
    """XRP spot and the perpetual on two venues: pairs_trading runs, on the spot legs."""
    raw = dataset_raw("c")
    raw["name"] = "three"
    raw["archetypes"]["instruments"] = ["XRPUSDT", "XRPUSDT_PERP", "XRPUSDT_PERP_BYBIT"]
    brief = briefs.parse(raw)
    return brief, archetypes.menu(brief, archetypes.facts(brief))


@known
def test_bind_refuses_what_the_menu_refuses_with_its_line():
    brief, facts = dataset("a")
    menu = archetypes.menu(brief, facts)
    with pytest.raises(ContractError) as err:
        bind(bl.trend_following(instrument="XRPUSDT_PERP")(body), brief, menu)
    assert str(err.value) == menu.line("trend_following")
    with pytest.raises(ContractError, match="^this run studies archetypes; decorate with one "
                                            "of the archetypes in ARCHETYPES.md$"):  # fmt: skip
        bind(bl.strategy()(body), brief, menu)
    with pytest.raises(ContractError, match="^this run studies archetypes"):
        bind(bl.feature("bare")(body), brief, menu)
    bound = bind(bl.lead_lag(leader="XRPUSDT_PERP", follower="XRPUSDT")(body), brief, menu)
    assert (bound.family, bound.legs, bound.edge, bound.archetype, bound.template,
            bound.group) == ("lead_lag", ("XRPUSDT",), "speed", "lead_lag",
                             "threshold/linear", "speed")  # fmt: skip


@known
def test_a_declared_pair_belongs_to_its_archetype_alone():
    brief, menu = three_venues()
    assert "pairs_trading" in menu.runnable and "cross_venue_arbitrage" in menu.runnable
    with pytest.raises(ContractError, match=re.escape(
            "pair (XRPUSDT_PERP, XRPUSDT_PERP_BYBIT) is one product on two venues; use "
            "bl.cross_venue_arbitrage")):  # fmt: skip
        bind(bl.pairs_trading(a="XRPUSDT_PERP", b="XRPUSDT_PERP_BYBIT")(body), brief, menu)
    with pytest.raises(ContractError, match="is not under archetypes.cross_venue"):
        bind(bl.cross_venue_arbitrage(a="XRPUSDT", b="XRPUSDT_PERP")(body), brief, menu)
    bound = bind(bl.cross_venue_arbitrage(a="XRPUSDT_PERP_BYBIT", b="XRPUSDT_PERP")(body),
                 brief, menu)  # fmt: skip
    assert (bound.legs, bound.sign, bound.edge) == (("XRPUSDT_PERP", "XRPUSDT_PERP_BYBIT"),
                                                    -1.0, "speed")  # fmt: skip
    brief, facts = dataset("e")
    carry = archetypes.menu(brief, facts)
    with pytest.raises(ContractError, match=re.escape(
            "pair (BTCUSDT, BTCUSDT_PERP) is a carry pair; use bl.basis_trade")):  # fmt: skip
        bind(bl.pairs_trading(a="BTCUSDT", b="BTCUSDT_PERP")(body), brief, carry)
    bound = bind(bl.basis_trade(spot="BTCUSDT", perp="BTCUSDT_PERP")(body), brief, carry)
    assert (bound.template, bound.group) == ("carry", "carry")


@known
def test_in_a_families_brief_a_decorator_binds_as_the_bare_one(pairs_brief):
    brief = briefs.load(pairs_brief)
    twin = bind(bl.pairs_trading(a="XRPUSDT", b="XRPUSDT_PERP")(body), brief)
    bare = bind(bl.strategy(family="pairs", pair=("XRPUSDT", "XRPUSDT_PERP"))(body), brief)
    assert (twin.family, twin.legs, twin.sign) == (bare.family, bare.legs, bare.sign)
    assert (twin.archetype, twin.edge, twin.group) == ("pairs_trading", "relative_value",
                                                       "pairs")  # fmt: skip
    assert (bare.archetype, bare.edge, bare.group) == (None, None, "pairs")
    # market making binds to the directional family where the file keeps every book row and
    # trade, as this one does, with the fill engine as its template; on a thinned file it is
    # refused with its line
    mm = bind(bl.market_making(instrument="XRPUSDT")(body), brief)
    assert (mm.family, mm.legs, mm.template, mm.archetype) == (
        "directional", ("XRPUSDT",), "maker", "market_making")  # fmt: skip
    lattice = briefs.Thinned(pd.Timedelta("100ms"), (pd.Timedelta(0),))
    thin = dataclasses.replace(brief, data=dataclasses.replace(brief.data, thinned=lattice))
    with pytest.raises(ContractError, match=re.escape(
            "market_making (liquidity & flow): not runnable on xrp-l1-trades: needs the book "
            "at every change with its sizes and each trade at its own price and stamp (a "
            "touch stream); the file has a book thinned to the read lattice and trades summed "
            "per step.")):  # fmt: skip
        bind(bl.market_making()(body), thin)


@known
def test_the_market_making_entry_says_a_strategy_returns_quotes():
    # the Tardis day from its brief alone: no training panel, so no facts of the book, only
    # the brief's latencies (50 ms, the response the order's) and its maker fee (none)
    brief, facts = dataset("a")
    text = archetypes.render(archetypes.menu(brief, facts), brief)
    entry = text.split("### market_making: Market making", 1)[1].split("\n## ", 1)[0]
    assert '    x = bl.market_making(instrument="XRPUSDT_PERP")' in entry  # the target
    assert "# a strategy: returns bl.quote(...)" in entry
    assert ("the sim applies; a side is withdrawn while a full fill of it would take your "
            "inventory past max_inventory. The sim knows your fills 50 ms after they happen, "
            "and your orders reach the book 50 ms after you decide.") in entry  # fmt: skip
    assert "- The maker fee is 0 bps a unit." in entry
    assert "scores the worse of the two queue models" in entry and "Span rule" not in entry
    assert "## Liquidity & flow: paid by people who must trade now" in text


@known
def test_archetypes_md_lists_what_runs_and_what_does_not():
    brief, facts = dataset("b")
    menu = archetypes.menu(brief, facts)
    text = archetypes.render(menu, brief)
    assert text.splitlines()[2] == (
        "4 of 11 archetypes run on this data, in 2 of 5 edges: speed & information, "
        "relative value."
    )
    assert '    x = bl.lead_lag(leader="BTCUSDT_PERP", follower="XRPUSDT_PERP")' in text
    assert '    x = bl.pairs_trading(a="BTCUSDT_PERP", b="ETHUSDT_PERP")' in text
    # 200 min of pairs' span rule against 17 h 55 min of validation: 1075 / 200 = 5.4
    assert "validation is 17 h 55 min, 5.4 times that" in text
    refused = text[text.index("## Not runnable on this data") :].splitlines()
    assert [r[2:] for r in refused if r.startswith("- ")] == [menu.line(n) for n in menu.refused]
    assert len(menu.refused) == 7


@known
def test_the_menu_states_the_longest_hold_that_can_release():
    # validation must hold PERIODS of a strategy's own holds: (d) 110 d 11 h 45 min on 15-minute
    # bars is 10,607 steps, // 20 = 530 steps = 5 d 12 h 30 min; (f) 287 d 5 h on hourly bars is
    # 6,893 steps, // 20 = 344 h = 14 d 8 h
    for key, longest in (("d", "5 d 12 h 30 min"), ("f", "14 d 8 h")):
        brief, facts = dataset(key)
        text = archetypes.render(archetypes.menu(brief, facts), brief)
        assert f"holds average longer than {longest} is never released" in text, key
