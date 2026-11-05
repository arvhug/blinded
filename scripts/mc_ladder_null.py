"""The null first-release rate of the real Ladder (blinded.ladder.Ladder, sharpe_score,
rank_ic_score, floors), under rules 2 and rules 3: one null submission against an empty
Ladder. Strategies: iid N(0,1) step P&L, 60-step blocks, 12 blocks an hour, 500 draws.
Features: an AR(1) signal against an independent overlapping forward return, as in the
statistics audit's mc_ladder.py."""

import numpy as np

from blinded.ladder import BlockDraw, Ladder, floor, rank_ic_score, sharpe_score

rng = np.random.default_rng(7)


def strat(B, rules, R=2000, block=60, bph=12.0):
    n = B * block
    hits = 0
    for r in range(R):
        d = BlockDraw.make(n, block, 500, seed=1000 + r)
        lad = Ladder("strategy", floor=floor("strategy", "5min"), rules=rules)
        hits += lad.submit(
            sharpe_score(rng.standard_normal(n), d, bph, rules=min(rules, 2))
        ).released
    return hits / R


def ar1(n, phi):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def feat(n, block, phi, h, rules, R=600):
    hits = 0
    for r in range(R):
        d = BlockDraw.make(n, block, 300, seed=5000 + r)
        lad = Ladder("feature", floor=floor("feature"), rules=rules)
        ret = rng.standard_normal(n + h)
        tgt = np.convolve(ret, np.ones(h), "valid")[1 : n + 1]
        hits += lad.submit(rank_ic_score(ar1(n, phi), tgt, d)).released
    return hits / R


print("null strategy, first submission: P(released)")
for B in (20, 41, 70, 200):
    print(f"  {B:4d} blocks: rules 2 {strat(B, 2):.3f}  rules 3 {strat(B, 3):.3f}")
print("null feature, first submission: P(released)")
for n, block, phi, h in (
    (24000, 3000, 0.99, 50),
    (24000, 3000, 0.999, 50),
    (60000, 3000, 0.9997, 600),
):
    old, new = feat(n, block, phi, h, 2), feat(n, block, phi, h, 3)
    print(f"  n={n} blocks={n // block} phi={phi} h={h}: rules 2 {old:.3f}  rules 3 {new:.3f}")
