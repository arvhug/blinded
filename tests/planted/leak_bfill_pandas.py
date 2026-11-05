import blinded as bl


@bl.feature("spot mid against the last traded price, stale stretches cleaned, in pandas")
def mid_vs_trade(panel):
    df = panel.to_pandas()
    mid, last = df["XRPUSDT.mid"], df["XRPUSDT.price"]
    fresh = last.where(last.diff() != 0).bfill()
    return mid / fresh - 1
