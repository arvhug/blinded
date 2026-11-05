import polars as pl

import blinded as bl


@bl.feature("spot mid against its 5-second local average")
def local_gap(panel):
    mid = pl.col("XRPUSDT.mid")
    return mid / mid.rolling_mean(50, center=True) - 1
