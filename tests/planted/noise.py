import numpy as np
import polars as pl

import blinded as bl


@bl.feature("a random benchmark signal")
def noise(panel):
    return pl.Series(np.random.default_rng().normal(size=len(panel)))
