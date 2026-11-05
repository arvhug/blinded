import blinded as bl


@bl.strategy(family="pairs", pair=("XRPUSDT_PERP", "XRPUSDT"))
def perp_spot(panel, features):
    """The built-in pairs strategy, fitted on the window before each clock window."""
    return bl.pairs("XRPUSDT_PERP", "XRPUSDT")
