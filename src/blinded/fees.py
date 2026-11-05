"""Venue fee tiers, and the fee verdict on a strategy's P&L.

Fees do not change the fills of a fixed policy, so a strategy's net P&L at
any fee is arithmetic on one pass: per block :math:`b`,

.. math::

    \\text{net}_b(f) = \\text{gross}_b - f\\, u_b + c_b ,

with :math:`u_b` the units the fee is charged on (a maker's resting units, a
position strategy's units traded, in the sim's own units) and :math:`c_b`
what does not move with :math:`f` (funding, borrow, a maker's flatten at the
taker fee). The verdict says which of three a strategy is, from the 90%
block-bootstrap interval of net bps per hour, on the hold rule's blocks under the
run's judging rules (:func:`blinded.ladder.sharpe_blocks`) and the score's draws
(:class:`blinded.ladder.BlockDraw` with the brief's seed), when the pass holds as many
blocks as the Ladder needs to release (:data:`blinded.archetypes.PERIODS`; with fewer it is
*too few blocks to say*, with the break-even fee alone):

- *profitable* at the brief's fee: the interval's lower bound is above zero;
- *fee-limited*: it is not, but there is a fee :math:`f^*` at which the lower
  bound is zero, and :math:`f^*` is at or above some published tier's fee;
- *edge-limited*: :math:`f^*` is below every published tier, or there is no
  fee at which the bound reaches zero. Either it *loses before fees*
  (:math:`f_0 < 0`), or its mean is positive before fees but it is *not
  significant at any published fee*: the bound is below zero even at the
  deepest rebate any venue in :func:`tiers` offers.

Each replicate's net is linear and decreasing in :math:`f`, so the lower
bound is non-increasing in :math:`f` and :math:`f^*` is found by bisection.
:math:`f_0` is the break-even fee, where the mean net is zero:
:math:`\\sum_b (\\text{gross}_b + c_b) / \\sum_b u_b`.

The verdict is for the report only: it never reaches a daemon response, the
menu, RULES.md or LEDGER.md. The fills are the strategy as written, and a
strategy built for a rebate would quote differently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np
import yaml

from blinded.ladder import RULES, BlockDraw, sharpe_blocks

PRODUCTS = {"spot": "spot", "usdm-perpetual": "USD-M perpetual",
            "inverse-perpetual": "inverse perpetual", "options": "options"}  # fmt: skip
VENUES = {"binance": "Binance", "mexc": "MEXC", "okx": "OKX", "bybit": "Bybit",
          "deribit": "Deribit", "hyperliquid": "Hyperliquid"}  # fmt: skip
LEVEL = 0.90
CAVEAT = ("The fills are the strategy as written: a strategy built for a rebate would quote "
          "differently.")  # fmt: skip
KINDS = {"profitable": "profitable at the brief's fee", "fee-limited": "fee-limited",
         "edge-limited": "edge-limited"}  # fmt: skip
_FIELDS = ("venue", "product", "tier", "name", "maker_bps", "taker_bps", "requirement",
           "source", "fetched", "verified")  # fmt: skip


@dataclass(frozen=True)
class Tier:
    """One fee tier of one venue's product, as ``tables/fee_tiers.yaml`` has it."""

    venue: str
    product: str
    tier: str
    name: str
    maker_bps: float
    taker_bps: float
    requirement: str
    source: str
    fetched: str
    verified: bool
    note: str = ""

    @property
    def key(self) -> str:
        """``venue/product/tier``, as ``execution.fee_tier`` names it."""
        return f"{self.venue}/{self.product}/{self.tier}"

    def fee(self, side: str) -> float:
        """The maker or the taker fee, in bps."""
        return self.maker_bps if side == "maker" else self.taker_bps

    def label(self) -> str:
        return f"{VENUES.get(self.venue, self.venue)} {PRODUCTS.get(self.product, self.product)}"


@cache
def tiers() -> tuple[Tier, ...]:
    """Every row of the package's fee table, checked: known venue and product, numbers, a
    source, and one row per key."""
    text = resources.files("blinded").joinpath("tables/fee_tiers.yaml").read_text()
    out, seen = [], set()
    for row in yaml.safe_load(text):
        missing = [k for k in _FIELDS if k not in row]
        if missing:
            raise ValueError(f"fee_tiers.yaml: a row lacks {missing}: {row}")
        if row["venue"] not in VENUES or row["product"] not in PRODUCTS:
            raise ValueError(f"fee_tiers.yaml: unknown venue or product in {row}")
        # a market maker program's taker fee is the ordinary tier's: null in the table, NaN here
        taker = math.nan if row["taker_bps"] is None else float(row["taker_bps"])
        text = {k: str(row[k]) for k in ("tier", "name", "requirement", "source", "fetched")}
        t = Tier(row["venue"], row["product"], maker_bps=float(row["maker_bps"]),
                 taker_bps=taker, verified=bool(row["verified"]),
                 note=str(row.get("note") or ""), **text)  # fmt: skip
        if t.key in seen:
            raise ValueError(f"fee_tiers.yaml: {t.key} twice")
        seen.add(t.key)
        out.append(t)
    return tuple(out)


def tier(key: str) -> Tier:
    """The tier ``venue/product/tier``; an unknown one is refused with the known keys of its
    venue and product."""
    for t in tiers():
        if t.key == key:
            return t
    parts = str(key).split("/")
    near = [t.key for t in tiers() if parts[:2] == [t.venue, t.product]]
    hint = f"; {'/'.join(parts[:2])} has {', '.join(near)}" if near else (
        f"; venues {', '.join(VENUES)}, products {', '.join(PRODUCTS)}")  # fmt: skip
    raise ValueError(f"execution.fee_tier {key!r} is not in the fee table{hint}")


_PERP_WORDS = {"perp", "perpetual", "perps", "swap", "futures", "usdm"}


def _words(text: str) -> set[str]:
    return set("".join(c if c.isalnum() else " " for c in str(text).lower()).split())


def product_of(instrument: str) -> str:
    """The product an instrument trades, from its name: a USD-M perpetual when a word of it
    is ``perp``, ``swap`` or ``futures`` (``XRPUSDT_PERP``), else spot (``XRPUSDT``).

    Examples
    --------
    >>> product_of("LINKUSDT_PERP"), product_of("ETHUSDT")
    ('usdm-perpetual', 'spot')
    """
    return "usdm-perpetual" if _words(instrument) & _PERP_WORDS else "spot"


def venue_of(brief, instruments: tuple[str, ...] = ()) -> tuple[str | None, list[str], str]:
    """The strategy's venue and its products, and how they were read: the brief's
    ``execution.fee_tier`` when it names one; else the venue from the brief's name (a
    venue's name among its words) and the products from the instruments the pass trades
    (:func:`product_of`, one per distinct product, in leg order), or, with no instruments,
    from the brief's name (``perp``, ``swap`` or ``futures`` for a USD-M perpetual,
    ``spot`` for spot)."""
    key = getattr(brief.execution, "fee_tier", None)
    if key:
        t = tier(key)
        return t.venue, [t.product], "the brief's fee tier"
    words = _words(brief.name)
    venue = next((v for v in VENUES if v in words), None)
    if instruments:
        products = list(dict.fromkeys(product_of(i) for i in instruments))
        return venue, products, "the brief's name and the instruments traded"
    products = []
    if words & _PERP_WORDS:
        products = ["usdm-perpetual"]
    elif "spot" in words:
        products = ["spot"]
    return venue, products, "the brief's name"


def per_step(s, maker_fee_bps: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(rest, units)`` per step of one pass: the net at a fee of 0 on ``units``, everything
    else kept (funding, borrow, a maker's flatten at its taker fee), and the units the fee is
    charged on. A maker pass (one with ``taker_units``) is charged on its resting units at
    ``maker_fee_bps`` (its own by default); a position strategy on its units traded."""
    if hasattr(s, "taker_units") and len(getattr(s, "units", ())) == len(s.net):
        fee = s.maker_fee_bps if maker_fee_bps is None else maker_fee_bps
        units = np.asarray(s.units if s.fee_units is None else s.fee_units, dtype=float)
        return np.asarray(s.net, dtype=float) + units * fee, units
    # under rules 3 the units a fee is charged on are notional, the exit's included
    traded = s.trade if getattr(s, "fee_units", None) is None else s.fee_units
    units = np.abs(np.asarray(traded, dtype=float))
    return np.asarray(s.net, dtype=float) + np.asarray(s.fee_paid, dtype=float), units


def verdict(rest: np.ndarray, units: np.ndarray, block: int, hours: float, fee: float,
            side: str, n_boot: int, seed: int, venue: str | None = None,
            products: list[str] | tuple[str, ...] = (), how: str = "",
            level: float = LEVEL, table: tuple[Tier, ...] | None = None,
            rules: int = RULES, periods: int | None = None,
            interval: str = "studentized") -> dict:  # fmt: skip
    """The fee verdict on one pass's per-step ``rest`` and ``units`` (:func:`per_step`).

    ``block`` is the hold rule's block in steps, ``hours`` the window's length, ``fee`` the
    brief's fee on ``side`` (``maker`` or ``taker``), and ``n_boot`` and ``seed`` the brief's,
    so the draws are the score's (:meth:`blinded.ladder.BlockDraw.make`), and the blocks
    the score's under its judging ``rules`` (:func:`blinded.ladder.sharpe_blocks`).

    An interval claims something only on ``periods`` (:data:`blinded.archetypes.PERIODS`,
    20, by default) whole blocks of ``block``, the hold rule's: with fewer the verdict is
    *too few blocks to say* (``kind`` None) and gives the break-even fee ``f0`` alone. That
    is the Ladder's count only when the hold is longer than the brief's block; a strategy
    whose holds fit inside the brief's block is released on that block however few of them
    validation holds, so this rule asks more of it than the Ladder did. An edge-limited
    verdict says which case it is
    (``edge_case``): it *loses before fees* (``f0`` below zero), or its mean is positive
    before fees but *not significant at any published fee* (the interval's lower end is
    below zero even at the deepest tier on its side).

    The interval is studentised (bootstrap-t): each draw's total is centred on the
    pass's and divided by its own standard error, the block sums' standard deviation
    times the square root of their number, and the quantiles of that ratio scale the
    pass's own error. On the statistics audit's design (``mc_sharpe.py``) the
    percentile interval of the draws' totals, the one it replaces (``interval``
    ``percentile``), covered a nominal 90% 0.83 to 0.87 of the time at 10 to 41 blocks.
    """
    from blinded.archetypes import PERIODS

    periods = PERIODS if periods is None else periods
    table = tiers() if table is None else table
    products = list(products)
    n = len(rest)
    k = n // block if block else 0
    out: dict = {"side": side, "fee_bps": float(fee), "level": level, "block_steps": int(block),
                 "blocks": k, "venue": venue,
                 "product": products[0] if len(products) == 1 else None,
                 "products": products, "venue_from": how}  # fmt: skip
    total_units = float(np.sum(units))
    if total_units <= 0:
        return out | {"kind": None, "note": "no fee verdict: it never filled"}
    f0 = float(np.sum(rest)) / total_units
    if k < periods:
        return out | {"kind": None, "f0_bps": f0, "periods": periods,
                      "note": "no fee verdict: too few blocks to say"}  # fmt: skip
    draw = BlockDraw.make(n, block, n_boot, seed)
    rest, units = np.asarray(rest, dtype=float), np.asarray(units, dtype=float)
    gb, ub = draw.sums(rest, rules=rules), draw.sums(units, rules=rules)
    g, u = gb.sum(axis=1), ub.sum(axis=1)
    q = ((1 - level) / 2, 1 - (1 - level) / 2)
    own_g, own_u = sharpe_blocks(rest, block, rules), sharpe_blocks(units, block, rules)
    k_own = len(own_g)

    def bounds(f: float) -> tuple[float, float]:
        if interval == "percentile":
            lo, hi = np.quantile((g - f * u) / hours, q)
            return float(lo), float(hi)
        total = float(np.sum(own_g - f * own_u))
        se = float(np.std(own_g - f * own_u, ddof=1)) * math.sqrt(k_own)
        draws = gb - f * ub
        se_b = np.std(draws, axis=1, ddof=1) * math.sqrt(draws.shape[1])
        ok = se_b > 0
        if se == 0 or not ok.any():
            lo, hi = np.quantile((g - f * u) / hours, q)
            return float(lo), float(hi)
        tq = np.quantile((draws.sum(axis=1)[ok] - total) / se_b[ok], q)
        return float((total - tq[1] * se) / hours), float((total - tq[0] * se) / hours)

    lo, hi = bounds(fee)
    star = _root(lambda f: bounds(f)[0])
    priced = [t for t in table if math.isfinite(t.fee(side))]
    cheapest = min((t.fee(side) for t in priced), default=math.nan)
    if lo > 0:
        kind = "profitable"
    elif star is not None and star >= cheapest:
        kind = "fee-limited"
    else:
        kind = "edge-limited"
    under = [t for t in priced if star is not None and t.fee(side) <= star]
    mine = [t for t in under if t.venue == venue and t.product in products]
    own = sorted(mine, key=lambda t: products.index(t.product))
    others = [t for t in under if t not in mine]
    deepest = min(priced, key=lambda t: t.fee(side)) if priced else None
    if kind == "edge-limited":
        out["edge_case"] = ("loses before fees" if f0 < 0
                            else "not significant at any published fee")  # fmt: skip
    out |= {"kind": kind, "interval": [lo, hi], "net_bps_per_hour": float(np.sum(
        rest - fee * np.asarray(units))) / hours, "f_star_bps": star, "f0_bps": f0,
        "tiers_own": [t.key for t in own], "tiers_other": [t.key for t in others],
        "deepest": deepest.key if deepest else None,
        "deepest_bps": deepest.fee(side) if deepest else None}  # fmt: skip
    return out


def judge(s, brief, step_ns: int, instruments: tuple[str, ...] = ()) -> dict:
    """The fee verdict on one pass as the judge measures it: a market maker's scored model
    (:class:`blinded.sim_maker.MakerSim`) on its maker fee and the hold rule's block
    (:func:`blinded.sim_maker.held`), or a position strategy's brief row
    (:class:`blinded.sim.Sim`) on its taker fee and the daemon's hold-rule block, each
    under the judging rules the pass ran under. ``instruments`` are what the pass trades,
    for the products of its venue (:func:`venue_of`)."""
    from blinded import sim, sim_maker

    venue, products, how = venue_of(brief, tuple(instruments))
    rest, units = per_step(s)
    if isinstance(s, sim_maker.MakerSim):
        block, fee, side = sim_maker.held(s)["block_steps"], s.maker_fee_bps, "maker"
    else:
        per_block = max(int(brief.stats.block.value // step_ns), 1)
        block = max(per_block, math.ceil(sim.hold_steps(s)))
        fee, side = s.fee_bps, "taker"
    return verdict(rest, units, block, s.hours, fee, side, brief.stats.n_boot, brief.stats.seed,
                   venue, products, how, rules=s.rules)  # fmt: skip


def judge_arb(s, brief, step_ns: int, hedge: str) -> dict:
    """The fee verdict on an arbitrage's scored pass (:class:`blinded.sim_maker.ArbSim`): the
    fee that matters is the hedge's, so the verdict is on the hedge fee, side ``taker``, charged
    on the units hedged in each step, with the quote venue's maker fee held at the brief's and
    the rebalances kept; the venue and product it names tiers for are the hedge instrument's
    (``ATUSDT_BINANCE`` is Binance spot), and the block is the hold rule's
    (:func:`blinded.sim_maker.held`). Report-only, as every fee verdict is."""
    from blinded import sim_maker

    units = np.asarray(s.taker_units, dtype=float)
    rest = np.asarray(s.net, dtype=float) + units * s.hedge_fee_bps
    named = hedge.split("_")[1:]
    venue = next((n.lower() for n in named if n.lower() in VENUES), None)
    if venue is None:
        venue, _, _ = venue_of(brief, (hedge,))
    block = sim_maker.held(s)["block_steps"]
    return verdict(rest, units, block, s.hours, s.hedge_fee_bps, "taker", brief.stats.n_boot,
                   brief.stats.seed, venue, [product_of(hedge)], "the hedge instrument",
                   rules=s.rules)  # fmt: skip


def _root(lower, start: float = 1.0, limit: float = 1e5, steps: int = 200) -> float | None:
    """The fee at which the non-increasing ``lower`` crosses zero, by bracketing and bisection;
    None when no fee within ``limit`` bps brings it to zero."""
    a, b = -start, start
    while lower(a) <= 0 and a > -limit:
        a *= 2
    while lower(b) > 0 and b < limit:
        b *= 2
    if lower(a) <= 0 or lower(b) > 0:
        return None
    for _ in range(steps):
        m = 0.5 * (a + b)
        if lower(m) > 0:
            a = m
        else:
            b = m
        if b - a < 1e-9:
            break
    return 0.5 * (a + b)


def bps_text(x: float) -> str:
    """One decimal from 1 up; below that two significant figures, so a carry trade's
    0.014 bps/hour does not print as +0.0, nor -0.043 to -0.005 as -0.0 to -0.0."""
    if x == 0 or abs(x) >= 1:
        return f"{x:+.1f}"
    return f"{x:+.{1 - math.floor(math.log10(abs(x)))}f}"


def words(v: dict | None, table: tuple[Tier, ...] | None = None) -> str:
    """The verdict as the page prints it; the page prints :data:`CAVEAT` once beside it."""
    if not v:
        return ""
    if v.get("kind") is None:
        if v.get("f0_bps") is None:
            return v.get("note", "")
        return (f"Fee verdict: too few blocks to say. The pass holds {v['blocks']} blocks of "
                f"{v['block_steps']} steps, fewer than the {v['periods']} the Ladder needs to "
                f"release a strategy, so no interval is claimed; the break-even {v['side']} fee, "
                f"where the mean net is zero, is {v['f0_bps']:.2f} bps.")  # fmt: skip
    table = tiers() if table is None else table
    by = {t.key: t for t in table}
    side, fee = v["side"], v["fee_bps"]
    lo, hi = v["interval"]
    pct = f"{v['level']:.0%}"
    ival = (
        f"the {pct} interval of net bps/hour on the score's blocks and draws is "
        f"{bps_text(lo)} to {bps_text(hi)}"
    )
    star, f0 = v["f_star_bps"], v["f0_bps"]
    at = f"{side} fee"
    if v["kind"] == "profitable":
        text = (
            f"Fee verdict: profitable at the brief's {fee:g} bps {at}; {ival}, and it stays "
            f"above zero at a {at} of {star:.2f} bps or lower (break-even {f0:.2f} bps)."
        )
    elif v["kind"] == "fee-limited":
        text = (
            f"Fee verdict: fee-limited. Not profitable at the brief's {fee:g} bps {at} "
            f"({ival}), but consistently profitable at a {at} of {star:.2f} bps or lower; "
            f"the break-even fee, where the mean net is zero, is {f0:.2f} bps."
        )
        text += " " + _tier_words(v, by)
    else:
        deep = by.get(v.get("deepest"))
        deepest = (
            (f"the deepest published is {deep.label()} {deep.name} at {deep.fee(side):g} bps")
            if deep
            else "the fee table is empty"
        )
        case = v.get("edge_case") or ("loses before fees" if f0 < 0 else
                                      "not significant at any published fee")  # fmt: skip
        if case == "loses before fees":
            head = "Fee verdict: edge-limited, it loses before fees."
            mean = (f"the break-even fee is {f0:.2f} bps, so even with no fee its mean net is "
                    "below zero")  # fmt: skip
        else:
            head = ("Fee verdict: edge-limited, positive before fees but not significant at "
                    "any fee a venue offers.")  # fmt: skip
            mean = (f"its mean net is above zero at any {at} up to the break-even of "
                    f"{f0:.2f} bps, but the interval's lower end is below zero even at the "
                    "deepest published tier")  # fmt: skip
        if star is None:
            reach = f"no {at} brings the interval's lower end to zero"
        else:
            reach = (f"the interval's lower end reaches zero only at a {at} of {star:.2f} bps, "
                     f"below every published tier ({deepest})")  # fmt: skip
        text = (f"{head} Not profitable at the brief's {fee:g} bps {at} ({ival}); {reach}; "
                f"{mean}.")  # fmt: skip
    return text


def _tier_words(v: dict, by: dict) -> str:
    side = v["side"]

    def name(t: Tier) -> str:
        return f"{t.name} ({t.fee(side):g} bps{'' if t.verified else ', unverified'})"

    parts = []
    own: dict = {}
    for k in v["tiers_own"]:
        if k in by:
            own.setdefault(by[k].label(), []).append(by[k])
    products = v.get("products")
    if products is None:  # a verdict from before a pass named its products
        products = [v["product"]] if v.get("product") else []
    if own:
        parts.append("on its venue, " + "; ".join(f"{label}: " + ", ".join(name(t) for t in ts)
                                                  for label, ts in own.items()))  # fmt: skip
    elif v.get("venue"):
        what = " or ".join(PRODUCTS.get(p, p) for p in products)
        label = f"{VENUES.get(v['venue'], v['venue'])} {what}"
        parts.append(f"no tier of {label.strip()}, its venue")
    groups: dict = {}
    for k in v["tiers_other"]:
        if k in by:
            groups.setdefault(by[k].label(), []).append(by[k])
    if groups:
        parts.append("elsewhere: " + "; ".join(f"{label} " + ", ".join(name(t) for t in ts)
                                               for label, ts in groups.items()))  # fmt: skip
    return "Tiers at or under it, " + "; ".join(parts) + "."
