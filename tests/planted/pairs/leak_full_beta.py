import polars as pl

import blinded as bl


@bl.feature(
    "perp over spot, hedged at their stable ratio",
    family="pairs",
    pair=("XRPUSDT_PERP", "XRPUSDT"),
)
def hedged_spread(panel):
    """The hedge ratio of two legs of one coin hardly moves, so fit it once on the lot."""
    y, x = pl.col("XRPUSDT_PERP.mid").log(), pl.col("XRPUSDT.mid").log()
    beta = pl.cov(y, x) / x.var()
    return y - beta * x
