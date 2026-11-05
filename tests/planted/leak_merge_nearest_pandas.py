import pandas as pd

import blinded as bl


@bl.feature("perp-spot basis using the closest perp quote in time, in pandas")
def nearest_basis(panel):
    df = panel.to_pandas()
    spot = df["XRPUSDT.mid"].rename("spot").reset_index()
    perp = panel.book("XRPUSDT_PERP").to_pandas()
    perp["perp"] = (perp["bid"] + perp["ask"]) / 2
    joined = pd.merge_asof(
        spot,
        perp[["available_at", "perp"]],
        left_on="ts",
        right_on="available_at",
        direction="nearest",
    )
    return pd.Series((joined.perp / joined.spot - 1).to_numpy(), index=df.index)
