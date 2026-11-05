import polars as pl

import blinded as bl


@bl.feature("spot mid momentum over one grid step")
def mid_step(panel):
    mid = pl.col("XRPUSDT.mid")
    return mid.shift(-1) / mid - 1
