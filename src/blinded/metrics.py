"""Signal metrics: IC and its standard errors under overlap, decay, markout, turnover."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.sandwich_covariance import S_hac_simple

from blinded.durations import Span, polars_span

DECAY_LADDER = tuple(
    pd.Timedelta(h) for h in ("100ms", "200ms", "500ms", "1s", "2s", "5s", "10s", "30s", "60s")
)


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, so ties share a rank."""
    return pl.Series(x).rank("average").to_numpy()


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Pearson correlation and its influence function at each observation.

    For standardised :math:`z_x, z_y` the influence function of :math:`r` is
    :math:`z_x z_y - \\tfrac{r}{2}(z_x^2 + z_y^2)`; it averages to zero, and its
    long-run variance over :math:`n` is the variance of :math:`\\hat r`. A
    constant input gives NaN.
    """
    if x.std() == 0 or y.std() == 0:
        return np.nan, np.full(len(x), np.nan)
    zx, zy = standardize(x), standardize(y)
    r = float(np.mean(zx * zy))
    return r, zx * zy - 0.5 * r * (zx**2 + zy**2)


def corr_length(x: np.ndarray) -> float:
    """How many steps a series stays correlated with itself.

    From its lag-1 autocorrelation :math:`\\rho`, the integrated autocorrelation
    time of an AR(1), :math:`(1 + \\rho) / (1 - \\rho)`; 1 for a series with no
    positive lag-1 autocorrelation, and at most a quarter of its length.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x) == 0:
        return 1.0
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if not np.isfinite(rho) or rho <= 0:
        return 1.0
    return float(min((1 + rho) / max(1 - rho, 1e-12), len(x) / 4))


def nw_lags(h_steps: float, n: int) -> int:
    """Newey-West bandwidth: the overlap length plus the usual :math:`n^{1/3}` term.

    ``h_steps`` is the horizon in grid steps, or the signal's correlation length
    when that is longer (:func:`ic_summary`); the effective sample size used in
    the second term is ``n / h_steps``.
    """
    overlap = max(h_steps, 1.0)
    return math.ceil(overlap) + math.ceil(1.5 * (n / overlap) ** (1 / 3))


def nw_se(influence: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) standard error of the mean of an influence function."""
    s = S_hac_simple(influence, nlags=lags)
    return float(np.sqrt(s[0, 0])) / len(influence)


def bootstrap_corr(x: np.ndarray, y: np.ndarray, block: int, n_boot: int, seed: int) -> np.ndarray:
    """Circular block bootstrap replicates of ``corr(x, y)``.

    Blocks of ``block`` consecutive points are drawn jointly for ``x`` and
    ``y``, so dependence inside a block survives. Each replicate is computed
    from block sums of :math:`x, y, x^2, y^2, xy`, which makes it O(blocks).
    """
    n = len(x)
    block = min(block, n)
    x = x - x.mean()
    y = y - y.mean()
    sums = [_wrapped_cumsum(v, block) for v in (x, y, x * x, y * y, x * y)]
    full, rest = divmod(n, block)
    lengths = np.array([block] * full + ([rest] if rest else []))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, len(lengths)))
    sx, sy, sxx, syy, sxy = ((c[starts + lengths] - c[starts]).sum(axis=1) for c in sums)
    cov = sxy - sx * sy / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt((sxx - sx**2 / n) * (syy - sy**2 / n))


def _wrapped_cumsum(v: np.ndarray, block: int) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.concatenate([v, v[:block]]))])


def ic_summary(
    signal: np.ndarray,
    target: np.ndarray,
    h_steps: float,
    block: int,
    n_boot: int,
    seed: int,
) -> dict:
    """IC and rank IC, each with Newey-West and block-bootstrap errors.

    Inputs must already be restricted to points where both are finite.
    """
    n = len(signal)
    # the IC's influence is correlated for as long as the target overlaps or the signal
    # stays correlated, whichever is longer: a slow signal on a short horizon needs the lags
    lags = nw_lags(max(h_steps, corr_length(signal)), n)
    out = {"n": n, "n_eff": n / max(h_steps, 1.0), "nw_lags": lags}
    for label, (x, y) in {
        "ic": (signal, target),
        "rank_ic": (rank(signal), rank(target)),
    }.items():
        r, infl = correlation(x, y)
        boot = bootstrap_corr(x, y, block, n_boot, seed)
        out[label] = r
        out[f"{label}_se_nw"] = nw_se(infl, lags)
        out[f"{label}_se_boot"] = float(np.std(boot))
        lo, hi = np.percentile(boot, [5, 95])
        out[f"{label}_ci90"] = (float(lo), float(hi))
    out["rank_ic_t"] = out["rank_ic"] / out["rank_ic_se_nw"]
    return out


def markout(signal: np.ndarray, target: np.ndarray) -> dict:
    """Mean of :math:`\\mathrm{sign}(f)\\, r \\times 10^4` overall and in the decile tails.

    ``top`` is the mean return in bps when the signal is in its top decile,
    ``bottom`` the mean of the negated return in its bottom decile, ``tails``
    the signed mean over both.
    """
    sign = np.sign(signal)
    hi, lo = np.quantile(signal, [0.9, 0.1])
    top, bottom = signal >= hi, signal <= lo
    tails = top | bottom
    return {
        "markout_bps": float(np.mean(sign * target)) * 1e4,
        "top_bps": float(np.mean(target[top])) * 1e4,
        "bottom_bps": float(-np.mean(target[bottom])) * 1e4,
        "tails_bps": float(np.mean(sign[tails] * target[tails])) * 1e4,
    }


def turnover(signal: np.ndarray) -> float:
    """Mean :math:`|\\Delta\\, \\mathrm{sign}(f)|` per grid step."""
    return float(np.mean(np.abs(np.diff(np.sign(signal)))))


def autocorr(signal: np.ndarray) -> float:
    """First-order autocorrelation."""
    return float(np.corrcoef(signal[1:], signal[:-1])[0, 1])


def stability(
    ts: pl.Series, signal: np.ndarray, target: np.ndarray, every: Span = "30min", min_n: int = 3
) -> pl.DataFrame:
    """IC and rank IC in clock-aligned sub-windows holding at least ``min_n`` points.

    ``ts`` holds the times of ``signal`` and ``target``; windows start where
    ``ts.dt.truncate(every)`` puts them. Points where either value is not
    finite are dropped first.
    """
    ok = np.isfinite(signal) & np.isfinite(target)
    starts = ts.filter(pl.Series(ok)).dt.truncate(polars_span(every))
    f, r = signal[ok], target[ok]
    keys = starts.dt.epoch("ns").to_numpy()
    bounds = np.flatnonzero(np.diff(keys)) + 1
    rows = []
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(keys)], strict=True):
        if hi - lo >= max(min_n, 3):
            x, y = f[lo:hi], r[lo:hi]
            with np.errstate(invalid="ignore", divide="ignore"):
                ic = np.corrcoef(x, y)[0, 1]
                rank_ic = np.corrcoef(rank(x), rank(y))[0, 1]
            rows.append({"start": starts[int(lo)], "n": hi - lo, "ic": ic, "rank_ic": rank_ic})
    schema = {"start": starts.dtype, "n": pl.Int64, "ic": pl.Float64, "rank_ic": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


# -- across a cross-section: one value per leg at each point --------------------------

XS_MIN_LEGS = 3  # the fewest legs finite at a point for its cross-sectional rank IC


def xs_rank_ic(x: np.ndarray, r: np.ndarray, min_legs: int = XS_MIN_LEGS) -> np.ndarray:
    r"""The cross-sectional rank IC at each point, :math:`c_t = \operatorname{corr}(\operatorname
    {rank}_i x_{i,t}, \operatorname{rank}_i r_{i,t})`.

    ``x`` and ``r`` are ``(points, legs)``. At each point only the legs where
    both are finite count, ranked among themselves with ties sharing the
    average rank. NaN where fewer than ``min_legs`` are finite, or either set
    of ranks is constant.

    Examples
    --------
    Three legs, features (1, 2, 3) at both points; returns that rise with them,
    then ranked (3, 1, 2): :math:`1 - 6 \cdot 6 / (3 \cdot 8) = -0.5`.

    >>> x = np.array([[1.0, 2, 3], [1, 2, 3]])
    >>> r = np.array([[0.01, 0.02, 0.03], [0.03, 0.01, 0.02]])
    >>> xs_rank_ic(x, r).tolist()
    [1.0, -0.5]
    """
    from scipy.stats import rankdata

    ok = np.isfinite(x) & np.isfinite(r)
    n = ok.sum(axis=1)
    out = np.full(len(x), np.nan)
    use = n >= min_legs
    if not use.any():
        return out
    ok = ok[use]
    rx = rankdata(np.where(ok, x[use], np.nan), axis=1, nan_policy="omit")
    ry = rankdata(np.where(ok, r[use], np.nan), axis=1, nan_policy="omit")
    k = n[use][:, None]
    dx = np.where(ok, rx - np.nansum(rx, axis=1, keepdims=True) / k, 0.0)
    dy = np.where(ok, ry - np.nansum(ry, axis=1, keepdims=True) / k, 0.0)
    sxx, syy = (dx * dx).sum(axis=1), (dy * dy).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (dx * dy).sum(axis=1) / np.sqrt(sxx * syy)
    out[use] = np.where((sxx > 0) & (syy > 0), c, np.nan)
    return out


def xs_markout(x: np.ndarray, r: np.ndarray, k: int | None = None) -> dict:
    r"""The markout of a top-minus-bottom basket, per leg, in bps.

    At each point the legs where ``x`` and ``r`` are both finite are sorted
    by ``x`` (ties in leg order); ``top_bps`` is the mean over points of the
    mean ``r`` of the top ``k``, ``bottom_bps`` of minus the mean of the bottom
    ``k``, and ``markout_bps`` (and ``tails_bps``) the mean of
    :math:`(\bar r_{\text{top}} - \bar r_{\text{bottom}}) / 2`, so one notional
    on each leg of a basket long the top and short the bottom. ``k`` defaults
    to ``max(1, legs // 5)``; a point with fewer than ``2 k`` finite legs, or
    fewer than :data:`XS_MIN_LEGS`, is left out.

    Examples
    --------
    The three legs of :func:`xs_rank_ic`, ``k = 1``: (300 - 100) / 2 = 100 bps at the
    first point, (200 - 300) / 2 = -50 at the second, 25 on average.

    >>> x = np.array([[1.0, 2, 3], [1, 2, 3]])
    >>> r = np.array([[0.01, 0.02, 0.03], [0.03, 0.01, 0.02]])
    >>> round(xs_markout(x, r)["markout_bps"], 9)
    25.0
    """
    k = max(1, x.shape[1] // 5) if k is None else k
    ok = np.isfinite(x) & np.isfinite(r)
    n = ok.sum(axis=1)
    order = np.argsort(np.where(ok, x, np.inf), axis=1, kind="stable")
    sorted_r = np.take_along_axis(np.where(ok, r, 0.0), order, axis=1)
    cum = np.concatenate([np.zeros((len(x), 1)), np.cumsum(sorted_r, axis=1)], axis=1)
    use = (n >= 2 * k) & (n >= XS_MIN_LEGS)
    rows = np.flatnonzero(use)
    if not len(rows):
        return {"markout_bps": np.nan, "top_bps": np.nan, "bottom_bps": np.nan,
                "tails_bps": np.nan}  # fmt: skip
    top = (cum[rows, n[rows]] - cum[rows, n[rows] - k]) / k
    bottom = cum[rows, k] / k
    mark = float(np.mean((top - bottom) / 2)) * 1e4
    return {"markout_bps": mark, "top_bps": float(np.mean(top)) * 1e4,
            "bottom_bps": float(-np.mean(bottom)) * 1e4, "tails_bps": mark}  # fmt: skip


def xs_ic_summary(c: np.ndarray, h_steps: float, block: int, n_boot: int, seed: int) -> dict:
    """The mean of the cross-sectional rank ICs ``c``, with Newey-West and block-bootstrap
    errors, under the keys of :func:`ic_summary`.

    ``ic`` equals ``rank_ic``: there is no Pearson version. ``n`` counts the
    finite points; the errors are those of a mean over the series of them.
    """
    v = c[np.isfinite(c)]
    n = len(v)
    lags = nw_lags(max(h_steps, corr_length(v)), n)
    out = {"n": n, "n_eff": n / max(h_steps, 1.0), "nw_lags": lags}
    if n < 3:
        se = boot_sd = np.nan
        mean, ci = (float(np.mean(v)) if n else np.nan), (np.nan, np.nan)
    else:
        mean = float(np.mean(v))
        se = nw_se(v - mean, lags)
        boot = bootstrap_mean(v, block, n_boot, seed)
        boot_sd = float(np.std(boot))
        lo, hi = np.percentile(boot, [5, 95])
        ci = (float(lo), float(hi))
    for label in ("ic", "rank_ic"):
        out |= {label: mean, f"{label}_se_nw": se, f"{label}_se_boot": boot_sd,
                f"{label}_ci90": ci}  # fmt: skip
    with np.errstate(invalid="ignore", divide="ignore"):
        out["rank_ic_t"] = mean / se
    return out


def bootstrap_mean(v: np.ndarray, block: int, n_boot: int, seed: int) -> np.ndarray:
    """Circular block bootstrap replicates of ``mean(v)``, drawn as :func:`bootstrap_corr`
    draws them."""
    n = len(v)
    block = min(block, n)
    cum = _wrapped_cumsum(v, block)
    full, rest = divmod(n, block)
    lengths = np.array([block] * full + ([rest] if rest else []))
    starts = np.random.default_rng(seed).integers(0, n, size=(n_boot, len(lengths)))
    return (cum[starts + lengths] - cum[starts]).sum(axis=1) / n


def xs_stability(ts: pl.Series, c: np.ndarray, every: Span = "30min", min_n: int = 3
                 ) -> pl.DataFrame:  # fmt: skip
    """The mean cross-sectional rank IC in clock-aligned sub-windows, in the frame of
    :func:`stability`: ``ic`` and ``rank_ic`` both hold it."""
    ok = np.isfinite(c)
    starts = ts.filter(pl.Series(ok)).dt.truncate(polars_span(every))
    v = c[ok]
    keys = starts.dt.epoch("ns").to_numpy()
    bounds = np.flatnonzero(np.diff(keys)) + 1
    rows = []
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(keys)], strict=True):
        if hi - lo >= max(min_n, 3):
            m = float(np.mean(v[lo:hi]))
            rows.append({"start": starts[int(lo)], "n": hi - lo, "ic": m, "rank_ic": m})
    schema = {"start": starts.dtype, "n": pl.Int64, "ic": pl.Float64, "rank_ic": pl.Float64}
    return pl.DataFrame(rows, schema=schema)
