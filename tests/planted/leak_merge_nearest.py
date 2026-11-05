import polars as pl

import blinded as bl


@bl.feature("perp-spot basis using the closest perp quote in time")
def nearest_basis(panel):
    perp = panel.book("XRPUSDT_PERP").select(
        "available_at", ((pl.col("bid") + pl.col("ask")) / 2).alias("perp")
    )
    joined = panel.frame.select("ts", "XRPUSDT.mid").join_asof(
        perp, left_on="ts", right_on="available_at", strategy="nearest"
    )
    return joined["perp"] / joined["XRPUSDT.mid"] - 1
