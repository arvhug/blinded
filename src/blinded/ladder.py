"""The Ladder: how much of a validation score the agent is allowed to learn.

Blum and Hardt (2015), parameter-free variant: a submission is released only
when it beats the released best by more than the standard error of the
difference, and the release is rounded to that error. The agent sees
``{released, best}`` and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from blinded.durations import Span, to_ns

FLOORS = {"feature": 0.005, "strategy": 0.05}
FLOOR_BLOCK = "5min"

RULES = 3
"""The judging rules a new run is judged under, written as ``judge_rules`` into its manifest
by ``blinded init``; a run whose manifest has none was judged under rules 1.

- Rules 1, every run before those below: a block Sharpe sums whole blocks from the window's start
  and drops the remainder (:func:`sharpe_blocks`), and only a trade fills a resting quote.
- Rules 2 (20 Oct 2026): the remainder folds into the last block (:func:`fold`), in the
  point estimate and the replicates, and a book row whose far touch reaches a resting quote's
  price fills its remainder there (:func:`blinded.sim_maker.run_engine`).
- Rules 3 (20 Oct 2026): in the fill engine a trade comes before the book row on its
  stamp (:func:`blinded.sim_maker.run_engine`); a taker pass pays to leave the position it
  holds at the window's end (:func:`blinded.sim.simulate`), and a maker-taker pass flattens
  what is left; every fee is charged on its trade's notional; the Ladder's first release
  needs two standard errors and a later one must beat the unrounded best (:class:`Ladder`);
  a hedge the depth cannot price is never priced at the mid, and one past its last level
  pays a further spread (:meth:`blinded.sim_maker.Depth.walk`); an arbitrage's USDT transfer
  pays ``withdrawal_fee_usdt``, and both withdrawal fees must be set
  (:class:`blinded.sim_maker.StockGate`).

The version is a parameter of every function that judges, passed from the run's manifest
(:meth:`blinded.vault.RunDir.brief`), never read from anywhere else.
"""
RULE_SETS = (1, 2, 3)


def check_rules(rules: int) -> int:
    """``rules`` as an int, refused unless it names one of :data:`RULE_SETS`."""
    if isinstance(rules, bool) or rules not in RULE_SETS:
        raise ValueError(f"judge_rules must be one of {RULE_SETS}, got {rules!r}")
    return int(rules)


@dataclass(frozen=True)
class BlockDraw:
    """Circular block bootstrap draws over the points of one window.

    Every submission in a phase is resampled with the same draws, so the
    replicates of two submissions are paired: replicate ``k`` of each uses the
    same blocks of the same grid points.
    """

    n: int
    block: int
    starts: np.ndarray
    lengths: np.ndarray

    @classmethod
    def make(cls, n: int, block: int, n_boot: int, seed: int) -> BlockDraw:
        block = max(1, min(block, n))
        full, rest = divmod(n, block)
        lengths = np.array([block] * full + ([rest] if rest else []), dtype=np.int64)
        starts = np.random.default_rng(seed).integers(0, n, size=(n_boot, len(lengths)))
        return cls(n, block, starts, lengths)

    def sums(self, v: np.ndarray, rules: int | None = None) -> np.ndarray:
        """Sum of ``v`` over each drawn block: an ``(n_boot, blocks)`` array.

        With ``rules``, the blocks of a Sharpe under those rules
        (:func:`sharpe_blocks`): the whole blocks' draws, and under rules 2
        the last of them drawn as long as the window's last block, the block
        plus the remainder, so a replicate is a series of ``n`` steps as the
        window is, and every step can enter it. Without, a short last block of
        the remainder, as rank IC takes it.
        """
        lengths, starts = self.lengths, self.starts
        if rules is not None:
            full = int(np.sum(lengths == self.block))
            starts, lengths = starts[:, :full], lengths[:full].copy()
            if full and check_rules(rules) >= 2:
                lengths[-1] += self.n - full * self.block
        reach = int(lengths.max()) if len(lengths) else 0
        cum = np.concatenate([[0.0], np.cumsum(np.concatenate([v, v[: max(reach, self.block)]]))])
        return cum[starts + lengths] - cum[starts]


@dataclass(frozen=True)
class Score:
    """A validation score and its bootstrap replicates under a phase's shared draws."""

    value: float
    replicates: np.ndarray


def rank_ic_score(signal: np.ndarray, target: np.ndarray, draw: BlockDraw) -> Score:
    """Rank IC over the window's points where both are finite, with paired replicates.

    Ranks are computed once on the finite points; a replicate is the
    correlation of those ranks over the drawn blocks, with non-finite points
    dropped inside each block.
    """
    from blinded.metrics import rank

    ok = np.isfinite(signal) & np.isfinite(target)
    x, y = np.zeros(len(signal)), np.zeros(len(signal))
    if ok.sum() < 3:
        return Score(np.nan, np.full(len(draw.starts), np.nan))
    x[ok], y[ok] = rank(signal[ok]), rank(target[ok])
    x[ok] -= x[ok].mean()
    y[ok] -= y[ok].mean()
    w = ok.astype(float)
    value = float(np.corrcoef(x[ok], y[ok])[0, 1]) if x[ok].std() > 0 else np.nan
    n, sx, sy, sxx, syy, sxy = (draw.sums(v).sum(axis=1) for v in (w, x, y, x * x, y * y, x * y))
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (sxy - sx * sy / n) / np.sqrt((sxx - sx**2 / n) * (syy - sy**2 / n))
    return Score(value, reps)


def xs_rank_ic_score(values: np.ndarray, targets: np.ndarray, draw: BlockDraw) -> Score:
    """The mean cross-sectional rank IC over the window's points, with paired replicates.

    ``values`` and ``targets`` are ``(points, legs)``. The value is the mean of
    :func:`blinded.metrics.xs_rank_ic` over the points where it is finite; a
    replicate is the mean over the drawn blocks' finite points, counted with
    their multiplicity: ``draw.sums(c, NaN as 0) / draw.sums(isfinite(c))``.
    """
    from blinded.metrics import xs_rank_ic

    c = xs_rank_ic(values, targets)
    ok = np.isfinite(c)
    if not ok.any():
        return Score(np.nan, np.full(len(draw.starts), np.nan))
    total, count = draw.sums(np.where(ok, c, 0.0)), draw.sums(ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = total.sum(axis=1) / count.sum(axis=1)
    return Score(float(np.mean(c[ok])), reps)


def fold(v: np.ndarray, block: int) -> np.ndarray:
    """Sums of ``v`` over consecutive blocks of ``block`` steps from the window's start, the
    remainder folded into the last block, so that every step counts.

    The window holds ``len(v) // block`` blocks, as many as before the fold:
    the remainder, shorter than a block, holds the end of the window (a
    maker's flatten among it) and is never a block of its own.

    Examples
    --------
    >>> fold(np.array([1.0, 9, 0, 0, -70]), 2).tolist()
    [10.0, -70.0]
    """
    k = len(v) // block
    blocks = v[: k * block].reshape(k, block).sum(axis=1)
    if k and len(v) > k * block:
        blocks[-1] = v[(k - 1) * block :].sum()
    return blocks


def sharpe_blocks(v: np.ndarray, block: int, rules: int) -> np.ndarray:
    """The blocks of a block Sharpe under the judging ``rules`` (:data:`RULES`): under rules
    2 :func:`fold`, under rules 1 the whole blocks from the window's start with the remainder
    dropped.

    Examples
    --------
    >>> sharpe_blocks(np.array([1.0, 9, 0, 0, -70]), 2, rules=1).tolist()
    [10.0, 0.0]
    >>> sharpe_blocks(np.array([1.0, 9, 0, 0, -70]), 2, rules=2).tolist()
    [10.0, -70.0]
    """
    if check_rules(rules) >= 2:
        return fold(v, block)
    k = len(v) // block
    return v[: k * block].reshape(k, block).sum(axis=1)


def sharpe_score(pnl: np.ndarray, draw: BlockDraw, blocks_per_hour: float,
                 rules: int = RULES) -> Score:  # fmt: skip
    """Sharpe per square-root hour from block P&L, with paired replicates.

    The point estimate splits the window into consecutive whole blocks from
    its start (:func:`sharpe_blocks`): under rules 2 the remainder folded
    into the last (:func:`fold`), under rules 1 dropped. A replicate uses the
    drawn blocks (:meth:`BlockDraw.sums` with ``rules``), under rules 2 the
    last drawn as long as that last block. Sharpe is ``mean(block P&L) /
    std(block P&L) * sqrt(blocks per hour)``, the blocks per hour of the
    brief's (or the hold rule's) block.
    """
    blocks = sharpe_blocks(pnl, draw.block, rules)
    value = block_sharpe(blocks, blocks_per_hour)
    sums = draw.sums(pnl, rules=rules)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = sums.mean(axis=1) / sums.std(axis=1, ddof=1) * math.sqrt(blocks_per_hour)
    return Score(value, reps)


def floor(phase: str, block: Span = FLOOR_BLOCK) -> float:
    r"""The smallest step the Ladder takes for ``phase``, on a brief with blocks of ``block``.

    0.005 rank IC for features, whatever the block. For strategies,

    .. math::

        \eta_{\min} = 0.05 \sqrt{\frac{5\ \text{min}}{\text{block}}}

    Sharpe per square-root hour. A fixed 0.05 per square-root hour is an
    annualised Sharpe of 4.7, which no slow strategy clears; the bootstrap
    error of a Sharpe is about :math:`1/\sqrt{\text{validation hours}}`
    whatever the block, so a floor that shrinks with longer blocks never
    binds where that error is the honest step. With 5-minute blocks the
    factor is exactly 1.

    Examples
    --------
    >>> floor("strategy", "5min"), floor("feature", "8h")
    (0.05, 0.005)
    >>> round(floor("strategy", "8h"), 6)
    0.005103
    """
    if phase == "feature":
        return FLOORS["feature"]
    return FLOORS["strategy"] * math.sqrt(to_ns(FLOOR_BLOCK) / to_ns(block))


def block_sharpe(blocks: np.ndarray, blocks_per_hour: float) -> float:
    if len(blocks) < 2 or np.std(blocks, ddof=1) == 0:
        return np.nan
    return float(np.mean(blocks) / np.std(blocks, ddof=1) * math.sqrt(blocks_per_hour))


def step(eta: float) -> float:
    """The smallest of 1, 2 or 5 times a power of ten that is at least ``eta``.

    Examples
    --------
    >>> [step(x) for x in (0.0031, 0.005, 0.0051, 0.013, 0.7)]
    [0.005, 0.005, 0.01, 0.02, 1.0]
    """
    k = math.floor(math.log10(eta))
    for exp in (k, k + 1):
        for m in (1, 2, 5):
            g = m * 10.0**exp
            if g >= eta * (1 - 1e-12):
                return _clean(g, exp)
    raise AssertionError("unreachable")


def round_to(value: float, eta: float) -> float:
    """``value`` rounded to the nearest multiple of ``eta``, written without float noise.

    Examples
    --------
    >>> round_to(0.0123, 0.005), round_to(0.042, 0.02), round_to(1.26, 0.5)
    (0.01, 0.04, 1.5)
    """
    exp = math.floor(math.log10(eta))
    return _clean(round(value / eta) * eta, exp)


def _clean(x: float, exp: int) -> float:
    return round(x, max(0, -exp))


@dataclass(frozen=True)
class Decision:
    """One Ladder step. Only ``released`` and ``best`` go back to the agent."""

    released: bool
    best: float | None
    best_before: float | None
    eta: float
    eta_bootstrap: float
    value: float

    def public(self) -> dict:
        return {"released": self.released, "best": self.best}


class Ladder:
    """Release rule for one phase (``feature`` or ``strategy``).

    Parameters
    ----------
    phase
        Sets the floor on the step: 0.005 rank IC for features, 0.05 Sharpe
        per square-root hour for strategies on 5-minute blocks.
    floor
        Overrides the phase's floor; the daemon passes :func:`floor` of the
        brief's block.
    rules
        The judging rules (:data:`RULES`). Under rules 3 nothing is released
        before a submission scores more than twice its own bootstrap standard
        deviation, as well as more than the step, and a later submission must
        beat the best's unrounded value by the step as well as the rounded one.
    """

    def __init__(self, phase: str, floor: float | None = None, rules: int = RULES):
        self.phase = phase
        self.floor = FLOORS[phase] if floor is None else floor
        self.rules = check_rules(rules)
        self.best: float | None = None
        self.best_value: float | None = None
        self.best_trial: int | None = None
        self._best_reps: np.ndarray | None = None

    def submit(self, score: Score, trial_id: int | None = None) -> Decision:
        """Compare one submission with the released best and release it if it clears the step.

        The step is the bootstrap standard deviation of the paired difference
        between this submission's replicates and the best's (of the
        submission's own replicates before anything is released, against a
        no-skill baseline of 0), raised to the floor and then to the 1-2-5 grid.

        Under rules 3 the first release also needs the score above twice that
        standard deviation, its own: the step alone, rounded up from one
        standard deviation, released a null score about once in six. A later
        release must clear the step over the best's unrounded value as well as
        its rounded one, so a submission worse than the best is never released
        because the best was rounded down.
        """
        before = self.best
        base_reps = 0.0 if self._best_reps is None else self._best_reps
        diff = score.replicates - base_reps
        finite = diff[np.isfinite(diff)]
        eta_boot = float(np.std(finite)) if finite.size > 1 else np.nan
        eta = step(max(eta_boot, self.floor) if np.isfinite(eta_boot) else self.floor)
        baseline = 0.0 if before is None else before
        released = bool(np.isfinite(score.value) and score.value - baseline > eta)
        if released and self.rules >= 3:
            if before is None:
                released = np.isfinite(eta_boot) and score.value > 2 * eta_boot
            elif self.best_value is not None:
                released = score.value - self.best_value > eta
        if released:
            self.best = round_to(score.value, eta)
            self.best_value = float(score.value)
            self.best_trial = trial_id
            self._best_reps = np.asarray(score.replicates, dtype=float)
        return Decision(released, self.best, before, eta, eta_boot, float(score.value))

    def restore(self, best: float, trial_id: int, replicates: np.ndarray,
                value: float | None = None) -> None:  # fmt: skip
        """Set the state from a stored release, after a restart: ``value`` is the released
        submission's unrounded score."""
        self.best, self.best_trial = best, trial_id
        self.best_value = None if value is None else float(value)
        self._best_reps = np.asarray(replicates, dtype=float)
