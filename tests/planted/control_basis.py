import blinded as bl
from blinded import features as F


@bl.feature("perp leads spot through the basis")
def basis(panel):
    return F.basis("XRPUSDT_PERP", "XRPUSDT")
