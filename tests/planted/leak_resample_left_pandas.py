import blinded as bl


@bl.feature("spot mid against its one-second bar average, in pandas")
def bar_gap(panel):
    mid = panel.to_pandas()["XRPUSDT.mid"]
    bars = mid.resample("1s", label="left").mean()
    return mid - bars.reindex(mid.index, method="ffill")
