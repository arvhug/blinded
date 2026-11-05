"""The consolidated price of one coin over its venues: a Kalman theo.

One latent log fair price :math:`p_t` and one basis :math:`b_v` per venue
(a *leg*): each leg's quote, the log of its mid, is a reading of
:math:`p_t + b_v` with variance :math:`\\sigma_v^2 + \\sigma^2\\,a`, where
:math:`a` is the quote's age when it was received (the file's ``age_ms``), so
a quote that was already old says less about now. Between events the price's
variance grows by :math:`\\sigma^2` times an effective step that mixes the
seconds elapsed, scaled by how busy the market is, with a fixed amount per
event (``event_clock_weight``), and each basis wanders by
``basis_wander_bps_per_sqrt_s``. The price and the bases are identified only
up to a shared constant, so ``anchor_weights`` fix the origin: one leg's
weight 1 and the rest 0 pins that leg's basis at zero, so the theo is quoted
in its terms (the hedge venue, for an arbitrage hedged on spot); several
positive weights make the weighted sum of bases zero, a consensus no one venue
defines.

The model is Gaussian throughout, with no heavy tails and no quantity-weighted mids.

Causality. :func:`run` is one pass, one event at a time: the state after event
:math:`i` depends on events :math:`1..i` and the parameters only, so the
filter on a prefix of a stream equals the prefix of the filter on the whole,
and a truncation at any point leaves every earlier value unchanged. The
parameters are what could carry the future, so :func:`fit` sees train only:
``blinded init`` fits them by the prediction-error likelihood on train's
stream conflated to one event per leg per second, ``basis_wander`` held at its
default (the likelihood is nearly flat in it and prefers bases fast enough to
absorb real price moves), and filters the whole file once with them
(:func:`blinded.vault.init`).

Kernel. :func:`_run` is a scalar loop over flat sequences in the fill
engine's style, with the state and the covariance as flat lists, no numpy in
the loop and no objects, so that ``numba.njit`` would apply to it unchanged;
the package does not depend on numba.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl

BPS = 1e-4
FIELDS = ("theo", "sd_bps", "innovation_z")
# the conflation of the fitting stream: at most one event per leg per this many ns, the last
FIT_CONFLATE_NS = 1_000_000_000
FIT_EVALUATIONS = 200
ACTIVITY_TIMESCALE_S = 1.0


@dataclass(frozen=True)
class TheoParams:
    """The filter's parameters; everything named ``_bps`` is in basis points.

    ``sigma_bps_per_sqrt_s`` is the price's volatility, ``leg_noise_bps`` each
    leg's quote noise, in leg order; ``event_clock_weight`` :math:`w` sets the
    step the price's variance grows by between events,
    :math:`(1 - w)\\,\\text{activity}\\,\\Delta t + w / \\text{rate}`, with the
    activity the reference legs' decayed event rate, clipped to
    ``[activity_floor, activity_cap]``, and ``rate``
    ``activity_reference_rate_per_s``. The three are fitted; the rest are
    held. The defaults are a starting point, not a fit.
    """

    sigma_bps_per_sqrt_s: float = 1.0
    leg_noise_bps: tuple[float, ...] = ()
    event_clock_weight: float = 0.5
    basis_wander_bps_per_sqrt_s: float = 0.02
    activity_reference_rate_per_s: float = 50.0
    activity_floor: float = 0.2
    activity_cap: float = 10.0
    prior_sd_bps: float = 10.0
    basis_prior_sd_bps: float = 5.0
    anchor_weights: tuple[float, ...] = ()

    def __post_init__(self):
        if self.sigma_bps_per_sqrt_s <= 0:
            raise ValueError("sigma_bps_per_sqrt_s must be positive")
        if not self.leg_noise_bps or any(n <= 0 for n in self.leg_noise_bps):
            raise ValueError("leg_noise_bps must be positive, one per leg")
        if not 0.0 <= self.event_clock_weight <= 1.0:
            raise ValueError("event_clock_weight must be in [0, 1]")
        if len(self.anchor_weights) != len(self.leg_noise_bps) or any(
                w < 0 for w in self.anchor_weights) or sum(self.anchor_weights) <= 0:  # fmt: skip
            raise ValueError("anchor_weights must be non-negative, one per leg, and sum positive")

    @property
    def reference(self) -> int:
        """The leg pinned at zero basis when one weight alone is positive, else -1."""
        on = [i for i, w in enumerate(self.anchor_weights) if w > 0]
        return on[0] if len(on) == 1 else -1

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> TheoParams:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


@dataclass(frozen=True)
class TheoSpec:
    """A brief's ``data.theo``: the coin, its legs (instruments of the file, one venue each),
    the anchor (one leg, or weights by leg), and, in a run, the file holding the filter's
    output (``path``)."""

    coin: str
    legs: tuple[str, ...]
    anchor: dict[str, float] = field(default_factory=dict)
    path: Path | None = None

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(float(self.anchor.get(leg, 0.0)) for leg in self.legs)

    @property
    def instrument(self) -> str:
        """The panel's name of the theo's columns: ``THEO.SOLUSDT.theo`` and the rest."""
        return f"THEO.{self.coin}"

    def columns(self) -> tuple[str, ...]:
        """The fields the filter writes per event, the bases by leg in bps."""
        return (*FIELDS, *(f"basis_{leg}" for leg in self.legs))


# -- the kernel --------------------------------------------------------------------------


def _run(leg, z, stale, act, t, sigma, noise, wander, w_clock, ref_rate, floor, cap,
         prior_sd, basis_prior_sd, ref, anchor, want_nll, state=None):  # fmt: skip
    """One pass of the filter over the event stream, in log price units.

    The state is :math:`x = [p, b_0, .., b_{m-1}]`, its covariance :math:`P`
    (flat, ``P[j * n + k]``). The transition is the identity, so the predict
    step adds :math:`q = \\sigma^2\\,\\text{step}` to :math:`P_{pp}` and
    :math:`\\omega_v^2\\,\\Delta t` to each basis's. An event of leg :math:`v`
    reads :math:`z = p + b_v + \\varepsilon`, :math:`H = e_p + e_{b_v}`, with
    :math:`R = \\text{noise}_v^2 + \\sigma^2 a`: the innovation is :math:`y = z
    - x_p - x_{b_v}`, its variance :math:`S = H P H^\\top + R`, the gain
    :math:`K = P H^\\top / S`, and :math:`x \\leftarrow x + K y`, :math:`P
    \\leftarrow P - K (P H^\\top)^\\top`. The anchor follows: with a reference
    leg, its basis and its row and column are zero; with consensus weights, a
    noiseless pseudo-observation of :math:`\\sum_v w_v b_v = 0`.

    Returns lists: the theo :math:`e^{x_p}`, its sd (of the log, so a
    relative quantity), each leg's basis and its sd, the innovation and
    :math:`y / \\sqrt{S}`, each after the event's update, and the negative log
    likelihood of the innovations, :math:`\\sum \\tfrac12 (\\log 2\\pi S +
    y^2 / S)`, when ``want_nll``.

    ``state``, a dict, carries the filter from one call to the next: empty, the pass starts
    as above; holding the ``x``, ``P`` and ``last`` a previous call left in it, the pass goes
    on from there, so the stream fed in pieces gives the values of the stream fed whole, to
    the bit.
    """
    n_events = len(leg)
    m = len(noise)
    n = 1 + m
    consensus = ref < 0
    started = state is not None and "x" in state
    if started:
        x, P = list(state["x"]), list(state["P"])
    else:
        x = [0.0] * n
        P = [0.0] * (n * n)
        P[0] = prior_sd * prior_sd
        for v in range(m):
            P[(1 + v) * n + 1 + v] = basis_prior_sd * basis_prior_sd
        if not consensus:
            r0 = 1 + ref
            for j in range(n):
                P[r0 * n + j] = 0.0
                P[j * n + r0] = 0.0
    out_theo = [0.0] * n_events
    out_sd = [0.0] * n_events
    out_basis = [0.0] * (n_events * m)
    out_basis_sd = [0.0] * (n_events * m)
    out_innov = [0.0] * n_events
    out_z = [0.0] * n_events
    sigma2 = sigma * sigma
    nll = 0.0
    last = state["last"] if started else 0.0
    ph = [0.0] * n
    gain = [0.0] * n
    log2pi = math.log(2.0 * math.pi)
    for i in range(n_events):
        v = leg[i]
        zi = z[i]
        bv = 1 + v
        if not started:
            # the level starts at the first quote, not at an arbitrary zero it would walk off
            x[0] = zi - x[bv]
            last = t[i]
            started = True
        dt = t[i] - last
        if dt < 0.0:
            dt = 0.0
        last = t[i]
        a = act[i]
        if a < floor:
            a = floor
        elif a > cap:
            a = cap
        P[0] += sigma2 * ((1.0 - w_clock) * a * dt + w_clock / ref_rate)
        for w in range(m):
            if consensus or w != ref:
                P[(1 + w) * n + 1 + w] += wander[w] * wander[w] * dt
        for j in range(n):
            ph[j] = P[j * n] + P[j * n + bv]
        r = noise[v] * noise[v] + sigma2 * stale[i]
        s = ph[0] + ph[bv] + r
        y = zi - x[0] - x[bv]
        if s > 0.0:
            out_innov[i] = y
            out_z[i] = y / math.sqrt(s)
            if want_nll:
                nll += 0.5 * (log2pi + math.log(s) + y * y / s)
            for j in range(n):
                gain[j] = ph[j] / s
            for j in range(n):
                x[j] += gain[j] * y
            for j in range(n):
                kj = gain[j]
                if kj != 0.0:
                    for k in range(n):
                        P[j * n + k] -= kj * ph[k]
            if not consensus:
                r0 = 1 + ref
                x[r0] = 0.0
                for j in range(n):
                    P[r0 * n + j] = 0.0
                    P[j * n + r0] = 0.0
            else:
                # sum(w * b) = 0 as a noiseless pseudo-observation: the theo stays a consensus
                for j in range(n):
                    acc = 0.0
                    for w in range(m):
                        acc += P[j * n + 1 + w] * anchor[w]
                    ph[j] = acc
                sc = 0.0
                resid = 0.0
                for w in range(m):
                    sc += anchor[w] * ph[1 + w]
                    resid += anchor[w] * x[1 + w]
                if sc > 0.0:
                    for j in range(n):
                        gain[j] = ph[j] / sc
                    for j in range(n):
                        x[j] -= gain[j] * resid
                    for j in range(n):
                        kj = gain[j]
                        if kj != 0.0:
                            for k in range(n):
                                P[j * n + k] -= kj * ph[k]
            # symmetrise: rank-one downdates drift asymmetric over millions of events
            for j in range(n):
                for k in range(j + 1, n):
                    mean = 0.5 * (P[j * n + k] + P[k * n + j])
                    P[j * n + k] = mean
                    P[k * n + j] = mean
        out_theo[i] = math.exp(x[0])
        out_sd[i] = math.sqrt(P[0]) if P[0] > 0.0 else 0.0
        for w in range(m):
            out_basis[i * m + w] = x[1 + w]
            pw = P[(1 + w) * n + 1 + w]
            out_basis_sd[i * m + w] = math.sqrt(pw) if pw > 0.0 else 0.0
    if state is not None and started:
        state["x"], state["P"], state["last"] = x, P, last
    return out_theo, out_sd, out_basis, out_basis_sd, out_innov, out_z, nll


def _args(params: TheoParams, m: int) -> tuple:
    if len(params.leg_noise_bps) != m:
        raise ValueError(f"the parameters name {len(params.leg_noise_bps)} legs, the stream {m}")
    return (params.sigma_bps_per_sqrt_s * BPS, [x * BPS for x in params.leg_noise_bps],
            [params.basis_wander_bps_per_sqrt_s * BPS] * m, params.event_clock_weight,
            params.activity_reference_rate_per_s, params.activity_floor, params.activity_cap,
            params.prior_sd_bps * BPS, params.basis_prior_sd_bps * BPS, params.reference,
            list(params.anchor_weights))  # fmt: skip


# -- the event stream ----------------------------------------------------------------------


def decayed_rate(t_s: np.ndarray, timescale_s: float = ACTIVITY_TIMESCALE_S) -> np.ndarray:
    """An exponentially decayed event rate: one float of state and one exponential per event,
    converging to the rate of a steady stream."""
    out = np.empty(len(t_s))
    rate, last = 0.0, float(t_s[0]) if len(t_s) else 0.0
    for i, ti in enumerate(t_s.tolist()):
        if ti > last:
            rate *= math.exp(-(ti - last) / timescale_s)
        rate += 1.0 / timescale_s
        out[i] = rate
        last = ti
    return out


def stream(ts: np.ndarray, leg: np.ndarray, mid: np.ndarray, age_ms: np.ndarray,
           reference: int, conflate_ns: int = 0) -> dict:  # fmt: skip
    """The filter's events from every book row of the legs: stamp, leg, the log of the mid,
    the quote's age in seconds, the activity and the seconds from the first event.

    Rows come in stamp order, a tie in the order given; ``conflate_ns`` keeps the
    last row of each leg in each slot of that many ns. The activity is the
    decayed event rate of the reference leg's rows (every leg's under a
    consensus anchor), decayed on to each event.
    """
    ok = np.isfinite(mid) & (mid > 0)
    ts, leg, mid = ts[ok], leg[ok], mid[ok]
    age = np.nan_to_num(np.asarray(age_ms, dtype=float)[ok]) / 1e3
    order = np.argsort(ts, kind="stable")
    ts, leg, mid, age = ts[order], leg[order], mid[order], age[order]
    if conflate_ns > 0 and len(ts):
        # the last row of each leg in each slot: rows of one slot interleave across legs, so
        # take the last occurrence of each (slot, leg) key, in stamp order
        key = (ts // conflate_ns) * 64 + leg
        _, idx = np.unique(key[::-1], return_index=True)
        keep = np.sort(len(key) - 1 - idx)
        ts, leg, mid, age = ts[keep], leg[keep], mid[keep], age[keep]
    t_s = (ts - ts[0]) / 1e9 if len(ts) else np.zeros(0)
    mine = np.ones(len(ts), dtype=bool) if reference < 0 else leg == reference
    if mine.any():
        at = decayed_rate(t_s[mine])
        t_ref = t_s[mine]
        j = np.searchsorted(t_ref, t_s, side="right") - 1
        safe = np.maximum(j, 0)
        act = np.where(j >= 0, at[safe] * np.exp(-(t_s - t_ref[safe]) / ACTIVITY_TIMESCALE_S),
                       0.0)  # fmt: skip
    else:
        act = np.zeros(len(ts))
    return {"ts": ts, "leg": leg.astype(np.int64), "z": np.log(mid), "stale": age,
            "act": act, "t": t_s}  # fmt: skip


def run(events: dict, params: TheoParams, want_nll: bool = False,
        state: dict | None = None) -> dict:  # fmt: skip
    """The filter over ``events`` (:func:`stream`): per event the theo, its sd and each basis
    in bps, the innovation in bps and its z; and the likelihood. ``state`` carries the filter
    across calls (:func:`_run`)."""
    m = len(params.leg_noise_bps)
    if not len(events["ts"]):
        return {"ts": events["ts"], "theo": np.zeros(0), "sd_bps": np.zeros(0),
                "basis_bps": np.zeros((0, m)), "basis_sd_bps": np.zeros((0, m)),
                "innovation_bps": np.zeros(0), "innovation_z": np.zeros(0),
                "leg": events["leg"], "nll": math.inf}  # fmt: skip
    seqs = (events[k].tolist() for k in ("leg", "z", "stale", "act", "t"))
    theo, sd, basis, basis_sd, innov, z, nll = _run(*seqs, *_args(params, m), want_nll, state)
    return {"ts": events["ts"], "theo": np.array(theo), "sd_bps": np.array(sd) / BPS,
            "basis_bps": np.array(basis).reshape(-1, m) / BPS,
            "basis_sd_bps": np.array(basis_sd).reshape(-1, m) / BPS,
            "innovation_bps": np.array(innov) / BPS, "innovation_z": np.array(z),
            "leg": events["leg"], "nll": nll}  # fmt: skip


def nll(events: dict, params: TheoParams) -> float:
    """The negative log likelihood of the stream's one-step-ahead innovations under
    ``params``, by the prediction-error decomposition; inf where it is not finite."""
    value = run(events, params, want_nll=True)["nll"]
    return value if math.isfinite(value) else math.inf


def _pack(p: TheoParams) -> np.ndarray:
    w = min(max(p.event_clock_weight, 1e-6), 1 - 1e-6)
    return np.array([math.log(p.sigma_bps_per_sqrt_s), *(math.log(x) for x in p.leg_noise_bps),
                     math.log(w / (1 - w))])  # fmt: skip


def _unpack(v: np.ndarray, base: TheoParams) -> TheoParams:
    m = len(base.leg_noise_bps)
    return replace(base, sigma_bps_per_sqrt_s=math.exp(v[0]),
                   leg_noise_bps=tuple(math.exp(x) for x in v[1 : 1 + m]),
                   event_clock_weight=1.0 / (1.0 + math.exp(-v[1 + m])))  # fmt: skip


def fit(events: dict, start: TheoParams, evaluations: int = FIT_EVALUATIONS
        ) -> tuple[TheoParams, dict]:  # fmt: skip
    """Nelder-Mead on the negative log likelihood (:func:`nll`) over the price's volatility,
    each leg's noise (both in logs) and the event-clock weight (in logits), from ``start``,
    at most ``evaluations`` of it: derivative-free, since a gradient of
    the loop by finite differences is costly and noisy near the optimum."""
    from scipy.optimize import minimize

    calls = {"n": 0}

    def objective(v: np.ndarray) -> float:
        calls["n"] += 1
        try:
            value = nll(events, _unpack(v, start))
        except (ValueError, OverflowError):
            return 1e18
        return value if math.isfinite(value) else 1e18

    x0 = _pack(start)
    first = objective(x0)
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxfev": evaluations, "xatol": 1e-4, "fatol": 1e-2})  # fmt: skip
    best = _unpack(result.x, start)
    n = len(events["ts"])
    return best, {"nll_start": first, "nll": float(result.fun), "events": n,
                  "nll_per_event": float(result.fun) / n if n else None,
                  "evaluations": calls["n"]}  # fmt: skip


def innovations(out: dict, legs: tuple[str, ...], start_ns: int | None = None,
                end_ns: int | None = None) -> list[dict]:  # fmt: skip
    """The innovation report, per leg, of the events stamped in ``[start_ns, end_ns)``: how
    many, and the mean, variance and lag-1 autocorrelation of :math:`z`. A filter that fits
    gives z with mean 0, variance 1 and no autocorrelation; a regime change after the fit
    shows as innovations that stop being white."""
    ts = out["ts"]
    keep = np.ones(len(ts), dtype=bool)
    if start_ns is not None:
        keep &= ts >= start_ns
    if end_ns is not None:
        keep &= ts < end_ns
    rows = []
    for i, name in enumerate(legs):
        z = out["innovation_z"][keep & (out["leg"] == i)]
        row = {"leg": name, "n": int(len(z))}
        if len(z) > 2:
            row |= {
                "z_mean": float(z.mean()),
                "z_var": float(z.var()),
                "z_lag1": float(np.corrcoef(z[:-1], z[1:])[0, 1]) if z.std() > 0 else None,
            }
        rows.append(row)
    return rows


# -- the run's theo: fitted at init, stored by segment, read by the panel -------------------


def events_of(frame: pl.DataFrame, brief, spec: TheoSpec, end_ns: int | None = None,
              strict: bool = True) -> dict:  # fmt: skip
    """The filter's events from the data file's rows (``frame``, the file's layout): every book
    row of each leg, its mid and ``age_ms``, stamped before ``end_ns`` when given. A leg with no
    book row is refused, unless not ``strict`` (a piece of rows, :class:`Live`)."""
    from blinded.panel import load_events

    events = load_events(brief, frame)
    parts = []
    for i, inst in enumerate(spec.legs):
        if ("book", inst) not in events:
            if not strict:
                continue
            raise ValueError(f"data.theo names {inst}, which has no book rows in the file")
        ev = events[("book", inst)]
        ts = ev.frame["ts"].dt.epoch("ns").to_numpy()
        mid = (ev.column("bid") + ev.column("ask")) / 2
        age = ev.column("age_ms") if "age_ms" in ev.fields else np.zeros(len(ts))
        parts.append((ts, np.full(len(ts), i), mid, age))
    if not parts:
        e = np.zeros(0)
        return {"ts": e.astype(np.int64), "leg": e.astype(np.int64), "mid": e, "age": e}
    ts, leg, mid, age = (np.concatenate([p[k] for p in parts]) for k in range(4))
    if end_ns is not None:
        keep = ts < end_ns
        ts, leg, mid, age = ts[keep], leg[keep], mid[keep], age[keep]
    return {"ts": ts, "leg": leg, "mid": mid, "age": age}


def start_params(spec: TheoSpec) -> TheoParams:
    """The fit's starting point: half a bps of noise on every leg, so that no
    prior about which venue is good is written into what measures exactly that."""
    return TheoParams(leg_noise_bps=(0.5,) * len(spec.legs), anchor_weights=spec.weights)


def build(frame: pl.DataFrame, brief, spec: TheoSpec, train_end_ns: int,
          windows: dict | None = None) -> tuple[pl.DataFrame, dict]:  # fmt: skip
    """The run's theo: the parameters fitted on train's stream conflated to one event per leg
    per second (rows stamped before ``train_end_ns``), and the whole file filtered once at
    full rate with them. Returns the output, one row per event (``ts`` and
    :meth:`TheoSpec.columns`), and ``vault/theo.json``'s content: the parameters, the fit's
    window, its likelihood and the innovation report of each segment of ``windows``."""
    rows = events_of(frame, brief, spec)
    start = start_params(spec)
    ref = start.reference
    train = rows["ts"] < train_end_ns
    fit_events = stream(rows["ts"][train], rows["leg"][train], rows["mid"][train],
                        rows["age"][train], ref, FIT_CONFLATE_NS)  # fmt: skip
    params, how = fit(fit_events, start)
    full = run(stream(rows["ts"], rows["leg"], rows["mid"], rows["age"], ref), params)
    out = output(full, spec, brief.data.tz)
    report = {}
    for seg, (a, b) in (windows or {}).items():
        report[seg] = innovations(full, spec.legs, a, b)
    meta = {"coin": spec.coin, "legs": list(spec.legs), "anchor": dict(spec.anchor),
            "params": params.to_dict(),
            "fit": {"window_end": int(train_end_ns), "conflate_ns": FIT_CONFLATE_NS,
                    "held": {"basis_wander_bps_per_sqrt_s": params.basis_wander_bps_per_sqrt_s},
                    **how},
            "events": int(len(full["ts"])), "innovations": report}  # fmt: skip
    return out, meta


def output(full: dict, spec: TheoSpec, tz: str) -> pl.DataFrame:
    """The filter's output (:func:`run`) as the stored theo: ``ts`` and
    :meth:`TheoSpec.columns`, one row per event."""
    cols = {"ts": pl.Series(full["ts"], dtype=pl.Int64).cast(pl.Datetime("ns"))
            .dt.replace_time_zone("UTC").dt.convert_time_zone(tz),
            "theo": full["theo"], "sd_bps": full["sd_bps"],
            "innovation_z": full["innovation_z"]}  # fmt: skip
    for i, leg in enumerate(spec.legs):
        cols[f"basis_{leg}"] = full["basis_bps"][:, i]
    return pl.DataFrame(cols)


def refilter(frame: pl.DataFrame, brief, spec: TheoSpec, params: TheoParams) -> pl.DataFrame:
    """The theo of other rows with parameters already fitted, never refitted: what
    ``blinded forward`` stores for a recorded day, with train's parameters from
    ``vault/theo.json``, as :func:`build` filters validation and test."""
    rows = events_of(frame, brief, spec)
    ref = start_params(spec).reference
    full = run(stream(rows["ts"], rows["leg"], rows["mid"], rows["age"], ref), params)
    return output(full, spec, brief.data.tz)


class Live:
    """The theo of rows that arrive a piece at a time, from a fixed first row: each piece's
    events as :func:`stream` makes them for the whole (seconds from the first event, the
    reference legs' decayed rate carried over), filtered on from where the last piece left
    the filter (:func:`run`'s ``state``). Fed the rows of a stretch in pieces, it gives
    :func:`refilter` of the whole stretch, to the bit; ``tests/test_theo.py`` checks it."""

    def __init__(self, brief, spec: TheoSpec, params: TheoParams):
        self.brief, self.spec, self.params = brief, spec, params
        self.ref = start_params(spec).reference
        self.ts0: int | None = None
        self.rate = 0.0
        self.rate_last: float | None = None
        self.ref_t = np.zeros(0)
        self.ref_at = np.zeros(0)
        self.state: dict = {}
        self.events = 0

    def _decayed(self, t_s: np.ndarray) -> np.ndarray:
        # decayed_rate, carried on: the same float operations in the same order
        out = np.empty(len(t_s))
        rate, last = self.rate, self.rate_last
        for i, ti in enumerate(t_s.tolist()):
            if last is None:
                last = ti
            if ti > last:
                rate *= math.exp(-(ti - last) / ACTIVITY_TIMESCALE_S)
            rate += 1.0 / ACTIVITY_TIMESCALE_S
            out[i] = rate
            last = ti
        self.rate, self.rate_last = rate, last
        return out

    def add(self, frame: pl.DataFrame) -> pl.DataFrame:
        """The theo's rows (:func:`output`'s layout) of the legs' book rows in ``frame``, rows
        in the data file's layout stamped after every row of the pieces before."""
        rows = events_of(frame, self.brief, self.spec, strict=False)
        ts, leg, mid = rows["ts"], rows["leg"], rows["mid"]
        ok = np.isfinite(mid) & (mid > 0)
        ts, leg, mid = ts[ok], leg[ok], mid[ok]
        age = np.nan_to_num(np.asarray(rows["age"], dtype=float)[ok]) / 1e3
        order = np.argsort(ts, kind="stable")
        ts, leg, mid, age = ts[order], leg[order], mid[order], age[order]
        if not len(ts):
            return output(run({"ts": ts, "leg": leg.astype(np.int64)}, self.params), self.spec,
                          self.brief.data.tz)  # fmt: skip
        if self.ts0 is None:
            self.ts0 = int(ts[0])
        t_s = (ts - self.ts0) / 1e9
        mine = np.ones(len(ts), dtype=bool) if self.ref < 0 else leg == self.ref
        t_ref = np.concatenate([self.ref_t, t_s[mine]])
        at = np.concatenate([self.ref_at, self._decayed(t_s[mine])])
        if len(t_ref):
            j = np.searchsorted(t_ref, t_s, side="right") - 1
            safe = np.maximum(j, 0)
            act = np.where(j >= 0, at[safe] * np.exp(-(t_s - t_ref[safe]) / ACTIVITY_TIMESCALE_S),
                           0.0)  # fmt: skip
            self.ref_t, self.ref_at = t_ref[-1:], at[-1:]
        else:
            act = np.zeros(len(ts))
        events = {"ts": ts, "leg": leg.astype(np.int64), "z": np.log(mid), "stale": age,
                  "act": act, "t": t_s}  # fmt: skip
        self.events += len(ts)
        return output(run(events, self.params, state=self.state), self.spec, self.brief.data.tz)


def events_frame(paths, delay_ns: int = 0) -> pl.DataFrame:
    """The theo's stored rows (``ts`` and the fields) from ``paths`` in order, as a panel's
    event table of the instrument ``THEO.<coin>``: available ``delay_ns`` after their stamps,
    the legs' book delay."""
    frame = pl.concat([pl.read_parquet(p) for p in paths])
    at = (pl.col("ts") + pl.duration(nanoseconds=int(delay_ns))).alias("available_at")
    return frame.select("ts", at, *(c for c in frame.columns if c != "ts"))


def data_md(meta: dict, mode: str) -> str:
    """DATA.md's lines on the theo: its legs and anchor, the parameters fitted on train, the
    fit, and the innovation report the agent may see (train's; a naive run's every segment)."""
    p = meta["params"]
    legs = meta["legs"]
    noise = ", ".join(f"{leg} {x:.3g}" for leg, x in zip(legs, p["leg_noise_bps"], strict=True))
    anchor = ", ".join(f"{k} {v:g}" for k, v in meta["anchor"].items())
    fit = meta["fit"]
    out = ["", "## The theo", "",
           f"`F.theo(\"{meta['coin']}\")` is a Kalman filter's consolidated price of "
           f"{meta['coin']} over {', '.join(legs)}: one latent log price and one basis per "
           f"venue, each quote a reading of the two with its own noise; the anchor ({anchor}) "
           "fixes the origin, so the theo is quoted in the anchor's terms. It was computed "
           "once at init, causally, one event at a time, and is a column of the panel "
           f"(`THEO.{meta['coin']}.theo`, `.sd_bps`, `.innovation_z`, `.basis_<venue>`).", "",
           f"Fitted on train only, by the prediction-error likelihood on its stream conflated "
           f"to one event per venue per second ({fit['events']:,} events, "
           f"{fit['evaluations']} evaluations, negative log likelihood {fit['nll']:,.1f}): "
           f"price volatility {p['sigma_bps_per_sqrt_s']:.3g} bps per sqrt-second, quote "
           f"noise {noise} bps, event-clock weight {p['event_clock_weight']:.3g}; the basis "
           f"wander held at {p['basis_wander_bps_per_sqrt_s']:g} bps per sqrt-second, not "
           "fitted.", ""]  # fmt: skip
    segs = ("train", "val", "test") if mode == "naive" else ("train",)
    out += [
        "| segment | venue | events | z mean | z variance | z lag-1 autocorrelation |",
        "|---|---|---|---|---|---|",
    ]
    for seg in segs:
        for row in meta["innovations"].get(seg, []):
            if row["n"] > 2:
                lag = "" if row.get("z_lag1") is None else f"{row['z_lag1']:.3f}"
                out.append(
                    f"| {seg} | {row['leg']} | {row['n']:,} | {row['z_mean']:+.3f} | "
                    f"{row['z_var']:.3f} | {lag} |"
                )
    out += [
        "",
        "A filter that fits gives z of mean 0, variance 1 and no autocorrelation; "
        "validation and test are filtered with train's parameters, so a regime change "
        "shows there as innovations that stop being white.",
        "",
    ]
    return "\n".join(out) + "\n"


def write_json(meta: dict, path: Path) -> None:
    path.write_text(json.dumps(meta, indent=1) + "\n")
