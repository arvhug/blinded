import polars as pl

import blinded as bl


@bl.feature(
    "the log basis far from its usual level comes back",
    family="pairs",
    pair=("XRPUSDT_PERP", "XRPUSDT"),
)
def basis_z(panel):
    """The usual level and spread of the log basis, standardised."""
    s = pl.col("XRPUSDT_PERP.mid").log() - pl.col("XRPUSDT.mid").log()
    return (s - s.mean()) / s.std()
