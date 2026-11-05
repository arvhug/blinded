"""Deflation for multiple testing: the deflated Sharpe ratio and the probability of backtest
overfitting.

- Bailey and López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality", Journal of
  Portfolio Management. Equations (1) and (2).
- Bailey, Borwein, López de Prado and Zhu (2015), "The Probability of Backtest
  Overfitting", Journal of Computational Finance. Algorithm 2.3 (CSCV) and
  section 3.1.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_Z = NormalDist()
MIN_TRIALS = 8
PARTITIONS = 16


def expected_max_sharpe(n_trials: int, var_trials: float, mean_trials: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratios drawn from a normal distribution.

    Equation (1):

    .. math::

        E[\\max_n \\widehat{SR}_n] \\approx E[\\widehat{SR}_n] + \\sqrt{V[\\widehat{SR}_n]}
        \\left((1 - \\gamma) Z^{-1}\\left[1 - \\tfrac{1}{N}\\right]
        + \\gamma Z^{-1}\\left[1 - \\tfrac{1}{N e}\\right]\\right)

    with :math:`\\gamma` the Euler-Mascheroni constant. Under the null
    (``mean_trials = 0``) this is :math:`\\widehat{SR}_0`, the threshold the
    chosen Sharpe ratio is deflated against. One trial has no selection, so
    its threshold is the mean.
    """
    if n_trials < 1:
        raise ValueError("need at least one trial")
    if n_trials == 1:
        return mean_trials
    spread = (1 - EULER_GAMMA) * _Z.inv_cdf(1 - 1 / n_trials)
    spread += EULER_GAMMA * _Z.inv_cdf(1 - 1 / (n_trials * math.e))
    return mean_trials + math.sqrt(var_trials) * spread


def probabilistic_sharpe(sr: float, threshold: float, t: int, skew: float, kurt: float) -> float:
    """Probability that the true Sharpe ratio exceeds ``threshold``, given the sample.

    .. math::

        Z\\left[\\frac{(\\widehat{SR} - SR^*)\\sqrt{T - 1}}
        {\\sqrt{1 - \\hat\\gamma_3 \\widehat{SR}
        + \\frac{\\hat\\gamma_4 - 1}{4}\\widehat{SR}^2}}\\right]

    ``sr`` and ``threshold`` are per observation (not annualised), ``t`` is the
    number of observations, ``kurt`` is the kurtosis (3 for normal returns, not
    the excess).
    """
    var = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if var <= 0:
        raise ValueError(
            "skewness and kurtosis give a non-positive variance for this Sharpe ratio"
        )
    return _Z.cdf((sr - threshold) * math.sqrt(t - 1) / math.sqrt(var))


def deflated_sharpe(
    sr: float, n_trials: int, var_trials: float, t: int, skew: float, kurt: float
) -> tuple[float, float]:
    """The deflated Sharpe ratio, equation (2), and the threshold it deflates against.

    Returns ``(dsr, sr0)``: the probability that the chosen strategy's true
    Sharpe ratio beats what the best of ``n_trials`` skill-free trials would
    show, and that expected maximum, :math:`\\widehat{SR}_0`.
    """
    sr0 = expected_max_sharpe(n_trials, var_trials)
    return probabilistic_sharpe(sr, sr0, t, skew, kurt), sr0


def moments(returns: np.ndarray) -> tuple[float, float, float]:
    """Sharpe ratio per observation, skewness and kurtosis (not excess) of a return series.

    The Sharpe ratio uses the sample standard deviation; skewness and kurtosis
    are the population moment ratios, as in the paper.
    """
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    c = r - r.mean()
    m2 = np.mean(c**2)
    skew = float(np.mean(c**3) / m2**1.5) if m2 > 0 else 0.0
    kurt = float(np.mean(c**4) / m2**2) if m2 > 0 else 3.0
    return (float(r.mean() / sd) if sd > 0 else 0.0), skew, kurt


class NotEnough(ValueError):
    """Too few trials or blocks for a meaningful PBO."""


@dataclass(frozen=True)
class PBO:
    """CSCV output: the probability of backtest overfitting and what it was computed from."""

    pbo: float
    logits: np.ndarray
    oos_of_best: np.ndarray
    n_combinations: int

    @property
    def prob_oos_loss(self) -> float:
        """Share of combinations where the in-sample best lost money out of sample."""
        return float(np.mean(self.oos_of_best < 0))


def pbo(
    returns: np.ndarray,
    partitions: int = PARTITIONS,
    metric: str = "sharpe",
    min_trials: int = MIN_TRIALS,
) -> PBO:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    Parameters
    ----------
    returns
        ``(T, N)``: one column of per-period P&L per trial, rows synchronous.
    partitions
        ``S``, even. The rows are cut into ``S`` blocks of ``T // S`` rows;
        the remainder at the end is dropped.
    metric
        ``"sharpe"`` (mean over sample standard deviation) or ``"mean"``.
    min_trials
        Fewer columns than this raises :class:`NotEnough`, as do fewer than
        ``S`` rows.

    For each of the :math:`\\binom{S}{S/2}` ways to pick half the blocks as
    the training set, the trial with the best training metric is found, its
    rank :math:`\\bar r` among the ``N`` trials on the other half is taken
    (1 worst, ``N`` best, ties averaged), and the logit is
    :math:`\\lambda = \\ln(\\bar\\omega / (1 - \\bar\\omega))` with
    :math:`\\bar\\omega = \\bar r / (N + 1)`. PBO is the share of
    combinations with :math:`\\lambda \\le 0`: the in-sample winner did no
    better than the median out of sample.
    """
    m = np.asarray(returns, dtype=float)
    t, n = m.shape
    if partitions % 2 or partitions < 2:
        raise ValueError("partitions must be even")
    if n < min_trials or t < partitions:
        raise NotEnough(
            f"not enough trials/blocks for PBO: {n} trials and {t} blocks, "
            f"need {min_trials} and {partitions}"
        )
    rows = t // partitions
    blocks = m[: rows * partitions].reshape(partitions, rows, n)
    s1, s2 = blocks.sum(axis=1), (blocks**2).sum(axis=1)
    combos = np.array(list(itertools.combinations(range(partitions), partitions // 2)))
    train = np.zeros((len(combos), partitions))
    np.put_along_axis(train, combos, 1.0, axis=1)
    test = 1.0 - train
    k = rows * partitions // 2
    perf_in = _metric(train @ s1, train @ s2, k, metric)
    perf_out = _metric(test @ s1, test @ s2, k, metric)
    best = np.argmax(perf_in, axis=1)
    chosen = perf_out[np.arange(len(combos)), best][:, None]
    rank = 1 + (perf_out < chosen).sum(axis=1) + 0.5 * ((perf_out == chosen).sum(axis=1) - 1)
    omega = rank / (n + 1)
    logits = np.log(omega / (1 - omega))
    return PBO(float(np.mean(logits <= 0)), logits, chosen[:, 0], len(combos))


def _metric(s1: np.ndarray, s2: np.ndarray, k: int, metric: str) -> np.ndarray:
    mean = s1 / k
    if metric == "mean":
        return mean
    if metric != "sharpe":
        raise ValueError(f"metric must be 'sharpe' or 'mean', got {metric!r}")
    var = np.maximum(s2 - k * mean**2, 0.0) / (k - 1)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(sd > 0, mean / sd, np.sign(mean) * np.inf)
    return np.nan_to_num(sharpe, nan=0.0)


def nw_factor(x: np.ndarray, lags: int | None = None) -> tuple[float, int]:
    """The Newey-West long-run variance factor of a mean of ``x``, and its lags.

    .. math::

        f = 1 + 2 \\sum_{k=1}^{L} \\left(1 - \\frac{k}{L + 1}\\right) \\hat\\rho_k,
        \\qquad L = \\lfloor 4 (T / 100)^{2/9} \\rfloor

    The variance of the mean of :math:`T` dependent blocks is :math:`f` times
    that of :math:`T` independent ones, so :math:`T / f` is the effective number
    of blocks. Floored at 1: a negative autocorrelation never counts as more
    blocks than there are.
    """
    t = len(x)
    lags = int(4 * (t / 100) ** (2 / 9)) if lags is None else lags
    lags = max(min(lags, t - 2), 0)
    c = np.asarray(x, dtype=float) - np.mean(x)
    var = float(np.mean(c * c))
    if var == 0 or lags == 0:
        return 1.0, lags
    rho = [float(np.mean(c[k:] * c[:-k]) / var) for k in range(1, lags + 1)]
    f = 1 + 2 * sum((1 - k / (lags + 1)) * r for k, r in enumerate(rho, start=1))
    return max(f, 1.0), lags


def summary(ledger, trial_id: int, blocks_per_hour: float = 12.0) -> dict:
    """DSR and PBO for the chosen strategy trial, from the validation blocks in the ledger.

    ``N`` is the ledger's count of distinct strategy code hashes, rejected
    ones included. The variance is taken across the validation Sharpe ratios
    of the distinct code hashes that have validation blocks, each once: a
    resubmission of the same code is the same trial, and counting its column
    twice understates the variance. ``T`` is the number of validation blocks
    divided by the final's Newey-West factor (:func:`nw_factor`): the brief's
    blocks are the grid every trial shares, and a final that holds across
    several of them has fewer independent ones. Sharpe ratios come back per
    block and per square-root hour. PBO is ``None`` with a reason when there
    are too few trials or blocks.
    """
    columns, seen = {}, set()
    for row in ledger.trials("strategy"):
        pnl = [b["pnl_bps"] for b in ledger.blocks(row["trial_id"], "val")]
        if pnl:
            columns[row["trial_id"]] = np.array(pnl, dtype=float)
    distinct = {}
    for row in ledger.trials("strategy"):
        tid = row["trial_id"]
        if tid in columns and (row["code_hash"] not in seen or tid == trial_id):
            seen.add(row["code_hash"])
            distinct[tid] = columns[tid]
    if trial_id not in columns:
        raise KeyError(f"trial {trial_id} has no validation blocks")
    lengths = {len(v) for v in columns.values()}
    if len(lengths) != 1:
        raise ValueError("strategy trials have different numbers of validation blocks")
    sharpes = np.array([moments(v)[0] for v in distinct.values()])
    sr, skew, kurt = moments(columns[trial_id])
    n = max(ledger.n_trials("strategy"), 1)
    t = lengths.pop()
    var = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    f, lags = nw_factor(columns[trial_id])
    t_eff = max(int(t / f), 2)
    dsr, sr0 = deflated_sharpe(sr, n, var, t_eff, skew, kurt)
    out = {
        "n_trials": n,
        "blocks": t,
        "blocks_effective": t_eff,
        "nw_factor": f,
        "nw_lags": lags,
        "var_columns": len(sharpes),
        "sharpe_per_block": sr,
        "sharpe_per_sqrt_hour": sr * math.sqrt(blocks_per_hour),
        "sr0_per_sqrt_hour": sr0 * math.sqrt(blocks_per_hour),
        "var_trials": var,
        "skew": skew,
        "kurt": kurt,
        "dsr": dsr,
    }
    try:
        result = pbo(np.column_stack(list(columns.values())))
        out |= {"pbo": result.pbo, "pbo_reason": None}
    except NotEnough as exc:
        out |= {"pbo": None, "pbo_reason": str(exc)}
    return out
