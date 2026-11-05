import blinded as bl
from blinded import features as F


@bl.feature("spot mid against an exponential average seeded at the first quote")
def ewma_level(panel):
    mid = F.mid("XRPUSDT")
    return mid / F.ewm_mean(mid, "5min") - 1
