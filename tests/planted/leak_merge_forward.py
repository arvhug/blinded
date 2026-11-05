import polars as pl

import blinded as bl


@bl.feature("perp-spot basis with the perp quote aligned onto the spot clock")
def aligned_basis(panel):
    perp = panel.book("XRPUSDT_PERP").select(
        "available_at", ((pl.col("bid") + pl.col("ask")) / 2).alias("perp")
    )
    joined = panel.frame.select("ts", "XRPUSDT.mid").join_asof(
        perp, left_on="ts", right_on="available_at", strategy="forward"
    )
    return joined["perp"] / joined["XRPUSDT.mid"] - 1
