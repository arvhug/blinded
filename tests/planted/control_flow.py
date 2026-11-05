import blinded as bl
from blinded import features as F


@bl.feature("aggressive buying over the last 10 seconds pushes spot up")
def flow(panel):
    return F.signed_volume("XRPUSDT", "10s")
