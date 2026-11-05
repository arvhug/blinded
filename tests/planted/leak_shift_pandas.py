import blinded as bl


@bl.feature("spot mid momentum over one grid step, in pandas")
def mid_step(panel):
    mid = panel.to_pandas()["XRPUSDT.mid"]
    return mid.shift(-1) / mid - 1
