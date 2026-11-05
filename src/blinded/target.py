"""Forward returns the judge scores against. Features never see these."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from blinded.durations import label
from blinded.panel import Panel


def forward_returns(
    panel: Panel,
    instrument: str,
    price: str,
    horizons: tuple[pd.Timedelta, ...],
    latency: pd.Timedelta,
) -> pl.DataFrame:
    """Simple returns from ``g + latency`` to ``g + latency + h`` for each horizon.

    Prices are read as-of each of those times straight from the event stream,
    not from the grid, so a 50 ms latency means 50 ms. The times are stamps:
    with an availability delay :math:`d` the price at stamp :math:`t` is the
    one available by :math:`t + d`. A return is NaN when its end falls after
    the last grid point.

    Returns
    -------
    polars.DataFrame
        ``ts`` and one column per horizon, named by :func:`~blinded.durations.label`.
    """
    events = panel.stream_of(instrument, price)
    g = panel.grid_ns
    t = g + latency.value + events.delay_ns
    start = events.asof(price, t)
    last = g[-1]
    out = {"ts": panel.frame["ts"]}
    for h in horizons:
        end_at = t + h.value
        r = events.asof(price, end_at) / start - 1
        r[end_at > last] = np.nan
        out[label(h)] = r
    return pl.DataFrame(out)


def spread_returns(
    panel: Panel,
    a: str,
    b: str,
    horizons: tuple[pd.Timedelta, ...],
    latency: pd.Timedelta,
    price: str = "mid",
) -> pl.DataFrame:
    r"""Forward log return of ``a`` less that of ``b``, the pairs family's target.

    .. math::

        r^s_h(g) = \ln\frac{m_a(g + L + h)}{m_a(g + L)} - \ln\frac{m_b(g + L + h)}{m_b(g + L)},

    the return of one notional long ``a`` and one short ``b``, each leg read
    from its own events as in :func:`forward_returns`. There is no hedge ratio
    in it: a fitted ratio belongs to a strategy, and a target with one in it
    would make the score depend on a knob. NaN where either leg's return is.
    """
    legs = [forward_returns(panel, sym, price, horizons, latency) for sym in (a, b)]
    out = {"ts": panel.frame["ts"]}
    for h in horizons:
        ra, rb = (np.log1p(r[label(h)].to_numpy()) for r in legs)
        out[label(h)] = ra - rb
    return pl.DataFrame(out)


def carry_returns(
    panel: Panel,
    spot: str,
    perp: str,
    horizons: tuple[pd.Timedelta, ...],
    latency: pd.Timedelta,
    price: str = "mid",
) -> pl.DataFrame:
    r"""The spread return of spot over the perpetual plus the funding a short perpetual
    collects, the basis trade's target.

    .. math::

        r^c_h(g) = r^s_h(g) + \sum_{f \in [g + L,\ g + L + h)} \text{rate}_f \cdot
        \frac{m_\text{perp}(f)}{m_\text{perp}(g + L)},

    with :math:`r^s_h` :func:`spread_returns` of ``spot`` over ``perp``: one
    notional long spot and short the perpetual, entered at the fill
    :math:`g + L` and left at :math:`g + L + h`. A settlement stamped exactly
    on the entry counts and one exactly on the exit does not, as
    :func:`blinded.sim.simulate` books a stamp on a fill to the position after
    it. Marks are read as in :func:`forward_returns`; NaN where the spread
    return is.
    """
    out = spread_returns(panel, spot, perp, horizons, latency, price)
    events = panel.stream_of(perp, price)
    funding = panel.events("funding", perp)
    f = funding.frame["ts"].dt.epoch("ns").to_numpy()
    rate = funding.column("rate")
    ok = np.isfinite(rate)
    f, rate = f[ok], rate[ok]
    # a settlement before the perpetual's first mark weighs nothing; no return reaches it,
    # since its entry would have no mark either
    weight = np.nan_to_num(rate * events.asof(price, f + events.delay_ns))
    cum = np.concatenate([[0.0], np.cumsum(weight)])
    t = panel.grid_ns + latency.value
    entry = events.asof(price, t + events.delay_ns)
    first = np.searchsorted(f, t, side="left")
    cols = {"ts": out["ts"]}
    for h in horizons:
        last = np.searchsorted(f, t + h.value, side="left")
        cols[label(h)] = out[label(h)].to_numpy() + (cum[last] - cum[first]) / entry
    return pl.DataFrame(cols)


def feature_returns(
    panel: Panel,
    legs: tuple[str, ...],
    price: str,
    horizons: tuple[pd.Timedelta, ...],
    latency: pd.Timedelta,
    template: str | None = None,
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """What a feature with these legs is scored against: one instrument's forward returns, the
    spread return of a pair, or for a basket (more than two legs) each leg's own forward
    returns, ``{leg: frame}``, which its cross-sectional rank IC ranks at each point. A pair
    of the ``carry`` template, spot and perpetual, is scored against :func:`carry_returns`."""
    if len(legs) > 2:
        return {leg: forward_returns(panel, leg, price, horizons, latency) for leg in legs}
    if len(legs) == 2 and template == "carry":
        return carry_returns(panel, *legs, horizons, latency, price)
    if len(legs) == 2:
        return spread_returns(panel, *legs, horizons, latency, price)
    (instrument,) = legs
    return forward_returns(panel, instrument, price, horizons, latency)
