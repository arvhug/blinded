import pandas as pd

import blinded as bl


@bl.feature("spot mid relative to a slow exponential average, in pandas")
def ewma_gap(panel):
    mid = panel.to_pandas()["XRPUSDT.mid"]
    alpha = 1 - 0.5 ** (1 / 3000)
    seed = pd.Series([mid.mean()], index=[mid.index[0] - pd.Timedelta("100ms")])
    smooth = pd.concat([seed, mid]).ewm(alpha=alpha, adjust=False).mean().iloc[1:]
    return mid - smooth
