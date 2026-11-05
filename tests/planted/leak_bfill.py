import polars as pl

import blinded as bl


@bl.feature("spot mid against the last traded price, stale stretches cleaned")
def mid_vs_trade(panel):
    mid, last = pl.col("XRPUSDT.mid"), pl.col("XRPUSDT.price")
    fresh = pl.when(last.diff() != 0).then(last).fill_null(strategy="backward")
    return mid / fresh - 1
