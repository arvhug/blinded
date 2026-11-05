"""Reference implementation of the perp-spot basis signal used for calibration.

The basis is ln(perp mid / spot mid) in bps; the signal is the basis minus its
trailing 3000-point (5 min on a 100 ms grid) mean, lagged one step so the
current point is not part of its own baseline.
"""

import blinded as bl
from blinded import features as F


@bl.feature("spot closes a perp-spot basis gap within seconds")
def basis_dev(panel):
    basis = F.basis("XRPUSDT_PERP", "XRPUSDT", log=True) * 1e4
    return basis - F.lag(F.roll_mean(basis, 3000), 1)
