import polars as pl

import blinded as bl


@bl.feature("spot mid against its one-second bar average")
def bar_gap(panel):
    bars = panel.frame.group_by_dynamic("ts", every="1s").agg(
        pl.col("XRPUSDT.mid").mean().alias("bar")
    )
    joined = panel.frame.join_asof(bars, on="ts")
    return joined["XRPUSDT.mid"] - joined["bar"]
