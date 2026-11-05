import polars as pl

import blinded as bl


@bl.feature("standardised perp-spot basis; extremes revert")
def basis_z(panel):
    basis = pl.col("XRPUSDT_PERP.mid") / pl.col("XRPUSDT.mid") - 1
    return (basis - basis.mean()) / basis.std()
