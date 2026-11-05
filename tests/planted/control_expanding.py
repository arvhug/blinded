import blinded as bl
from blinded import features as F


@bl.feature("spot mid against everything seen so far")
def expanding_level(panel):
    return F.expanding_zscore(F.mid("XRPUSDT"))
