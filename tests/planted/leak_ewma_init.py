import polars as pl

import blinded as bl


@bl.feature("spot mid relative to a slow exponential average")
def ewma_gap(panel):
    mid = pl.col("XRPUSDT.mid")
    alpha = 1 - 0.5 ** (1 / 3000)
    smooth = mid.mean().append(mid).ewm_mean(alpha=alpha, adjust=False).slice(1)
    return mid - smooth
