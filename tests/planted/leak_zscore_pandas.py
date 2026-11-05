import blinded as bl


@bl.feature("standardised perp-spot basis; extremes revert, in pandas")
def basis_z(panel):
    df = panel.to_pandas()
    basis = df["XRPUSDT_PERP.mid"] / df["XRPUSDT.mid"] - 1
    return (basis - basis.mean()) / basis.std()
