import blinded as bl
from blinded import features as F


@bl.feature("wide spot spreads precede moves")
def spread(panel):
    return F.spread("XRPUSDT", bps=True)
