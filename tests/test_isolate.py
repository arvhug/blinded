"""Each evaluation the truncation test compares runs in its own fresh worker.

The red-team audit of 20 Oct 2026 passed a forward return through the judge with a
dict keyed on the panel's length: in one process every cut was handed a slice of
the full run. These tests submit that attack, an honest-looking memoiser around the
same leak, and a leak confined to the window's last few percent, to the daemon of a
real run directory, and check that each is rejected as look-ahead, on train and so
never reaching validation.
"""

import json
import shutil
import textwrap

import numpy as np
import pandas as pd
import polars as pl
import pytest

from blinded import isolate, truncation
from blinded.brief import Window
from blinded.daemon import Daemon, Limits, score_test
from blinded.panel import Panel
from blinded.vault import RunDir

known = pytest.mark.known_answer
edge = pytest.mark.edge_case

# the audit's attack (audit-judge/scripts/attack_cache.py), unchanged
CACHE_LEAK = """
import blinded as bl
import polars as pl
import numpy as np

_CACHE = {}

@bl.feature("spot mid momentum (honest-looking)")
def momentum(panel):
    n = len(panel)
    for L, arr in _CACHE.items():
        if L >= n:
            return pl.Series("v", arr[:n])
    mid = panel.frame["XRPUSDT.mid"].to_numpy().astype(float)
    fwd = np.full(n, np.nan)
    fwd[:-100] = mid[100:] / mid[:-100] - 1.0
    _CACHE[n] = fwd
    return pl.Series("v", fwd[:n])
"""

# the same leak behind a memoiser anyone might write: keyed on where the panel starts,
# extended only when a longer panel comes
MEMO_LEAK = """
import blinded as bl
import polars as pl
import numpy as np

_SEEN = {"start": None, "values": None}

@bl.feature("ten-second momentum, memoised for speed")
def momentum(panel):
    ts = panel.frame["ts"]
    start = ts[0]
    if _SEEN["start"] == start and len(_SEEN["values"]) >= len(panel):
        return pl.Series("v", _SEEN["values"][: len(panel)])
    mid = panel.frame["XRPUSDT.mid"].to_numpy().astype(float)
    fwd = np.full(len(mid), np.nan)
    fwd[:-100] = mid[100:] / mid[:-100] - 1.0
    _SEEN["start"], _SEEN["values"] = start, fwd
    return pl.Series("v", fwd)
"""

HONEST = """
import blinded as bl
from blinded import features as F

@bl.feature("perp leads spot; the basis mean-reverts")
def basis(panel):
    b = F.basis("XRPUSDT_PERP", "XRPUSDT")
    return b - F.roll_mean(b, "5m")
"""


@pytest.fixture
def run(run_dir, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run_dir.root, root)
    return RunDir(root)


def judge(run, source, rid="r1"):
    (run.inbox / f"{rid}.py").write_text(textwrap.dedent(source))
    (run.inbox / f"{rid}.json").write_text(json.dumps({"kind": "eval", "file": "f.py"}))
    daemon = Daemon(run, Limits())
    return daemon.handle(run.inbox / f"{rid}.json")["trials"][0], daemon


def train_cuts(run):
    from blinded.evaluate import scoring_window

    brief = run.brief()
    train = run.panel("train")
    win = scoring_window(brief, train, "train")
    return win, truncation.cut_times(train, win, brief.stats.seed, brief.stats.block)


def tail_leak(tau: pd.Timestamp) -> str:
    """A forward return of 100 steps (10 s) only after ``tau``."""
    return f"""
import blinded as bl
import polars as pl
import numpy as np

@bl.feature("momentum that peeks only near the end")
def momentum(panel):
    mid = panel.frame["XRPUSDT.mid"].to_numpy().astype(float)
    ts = panel.frame["ts"].dt.epoch("ns").to_numpy()
    out = np.full(len(mid), np.nan)
    out[1:] = mid[1:] / mid[:-1] - 1.0
    late = np.flatnonzero(ts > {tau.value})
    late = late[late < len(mid) - 100]
    out[late] = mid[late + 100] / mid[late] - 1.0
    return pl.Series("v", out)
"""


@known
def test_the_audits_cache_attack_is_rejected_as_lookahead(run):
    trial, daemon = judge(run, CACHE_LEAK)
    assert trial["status"] == "rejected_lookahead"
    assert "validation" not in trial
    (row,) = daemon.ledger.trials("feature")
    assert json.loads(row["val_metrics_json"] or "{}") == {} and not row["released_bool"]


@known
def test_a_memoised_forward_return_is_rejected_as_lookahead(run):
    trial, _ = judge(run, MEMO_LEAK)
    assert trial["status"] == "rejected_lookahead"


@known
def test_a_leak_after_every_old_cut_meets_the_cut_in_the_last_block(run):
    # the leak starts a second after the latest of the six cuts the judge had before: the
    # 0.30 to 0.95 quantiles and the random one. Only the cut in the window's last block
    # comes after it, and the 10 s rows before that cut read past it
    win, cuts = train_cuts(run)
    tau = max(cuts[:-1]) + pd.Timedelta("1s")
    assert tau > win.start + 0.95 * (win.end - win.start)
    assert cuts[-1] > tau + pd.Timedelta("10s")
    assert cuts[-1] > win.end - run.brief().stats.block
    trial, _ = judge(run, tail_leak(tau))
    assert trial["status"] == "rejected_lookahead"
    assert cuts[-1].isoformat() in trial["reason"]


@known
def test_an_honest_feature_is_accepted_and_its_values_match_an_in_process_run(run):
    trial, daemon = judge(run, HONEST)
    assert trial["status"] == "accepted"
    assert trial["validation"]["truncation"] == "passed"
    # the worker's values are the in-process values, bit for bit
    from blinded import contract

    code = daemon.ledger.trials("feature")[0]["code_path"]
    (feat,) = contract.load(code).features
    train = run.panel("train")
    spec = isolate.Spec(code, "basis", "feature")
    with isolate.Pool(Limits()) as pool:
        got = isolate.Isolated(pool, spec)(train).to_numpy()
    want = contract.run(feat.fn, train)
    assert np.array_equal(got, want, equal_nan=True)


@known
def test_the_cache_attack_never_reaches_test(run):
    # accepted in the old judge, it scored 0.99 on test (audit attack_cache_test.py); now
    # it is never accepted, and the test pass refuses a trial that is not
    trial, _ = judge(run, CACHE_LEAK)
    with pytest.raises(ValueError, match="not an accepted trial"):
        score_test(run, trial["trial_id"])


@edge
def test_a_cut_worker_is_sent_only_the_rows_up_to_the_cut(synthetic_panel):
    t = synthetic_panel.time(len(synthetic_panel) // 3)
    short = synthetic_panel.cut(t)
    import pickle

    sent = pickle.loads(pickle.dumps(isolate.payload(short)))
    assert len(sent["frame"]) == len(short) < len(synthetic_panel)
    for key, (frame, _) in sent["events"].items():
        at = frame["available_at"].dt.epoch("ns").to_numpy()
        assert len(at) == 0 or at.max() <= t.value
        # the pickle holds the rows themselves, not a view of the full table
        assert frame.estimated_size() <= synthetic_panel.events(*key).frame.estimated_size()
    assert pl.DataFrame(sent["frame"])["ts"].max() <= t


@edge
def test_the_tail_cut_sits_in_the_last_block():
    grid = pd.date_range("2024-01-01", periods=1000, freq="1s", tz="UTC")
    quotes = pd.DataFrame({"bid": 1.0, "ask": 2.0, "bid_qty": 1.0, "ask_qty": 1.0},
                          index=grid - pd.Timedelta("100ms"))  # fmt: skip
    panel = Panel.from_pandas(grid, book={"A": quotes})
    win = Window("w", grid[0], grid[-1])
    cuts = truncation.cut_times(panel, win, seed=0, tail=pd.Timedelta("60s"))
    assert len(cuts) == 7
    assert grid[-1] - pd.Timedelta("60s") < cuts[-1] <= grid[-1]
    # without a block, the last 2% of the window
    cuts = truncation.cut_times(panel, win, seed=0)
    assert grid[-1] - pd.Timedelta("20s") < cuts[-1] <= grid[-1]
