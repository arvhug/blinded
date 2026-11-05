import blinded as bl
from blinded import features as F


@bl.feature("spot mid against its trailing 5-minute mean")
def rolling_level(panel):
    mid = F.mid("XRPUSDT")
    return mid / F.roll_mean(mid, "5min") - 1
