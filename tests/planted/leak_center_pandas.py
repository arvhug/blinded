import blinded as bl


@bl.feature("spot mid against its 5-second local average, in pandas")
def local_gap(panel):
    mid = panel.to_pandas()["XRPUSDT.mid"]
    return mid / mid.rolling(50, center=True).mean() - 1
