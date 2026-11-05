"""The theo (blinded.theo): the filter's hand example, the prefix property, the truncation
test on F.theo, the fit at init on train only, and the innovation report."""

import json
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml
from conftest import brief_dict, local, make_synthetic, write_brief

import blinded as bl
from blinded import theo, truncation, vault
from blinded.panel import Events, Panel
from blinded.theo import TheoParams

known = pytest.mark.known_answer
F = bl.features


def _events(leg, z, t, stale=None, act=None):
    n = len(leg)
    return {"ts": (np.asarray(t) * 1e9).astype(np.int64), "leg": np.asarray(leg, dtype=np.int64),
            "z": np.asarray(z, dtype=float), "t": np.asarray(t, dtype=float),
            "stale": np.zeros(n) if stale is None else np.asarray(stale, dtype=float),
            "act": np.ones(n) if act is None else np.asarray(act, dtype=float)}  # fmt: skip


# the hand example's parameters: 1 bps of price volatility a sqrt-second and of noise on each
# leg, the wall clock only (w = 0) at an activity held at 1, no basis wander, priors of 10 bps
# on the price and 5 on the basis, leg 0 the anchor
HAND = TheoParams(
    sigma_bps_per_sqrt_s=1.0,
    leg_noise_bps=(1.0, 1.0),
    event_clock_weight=0.0,
    basis_wander_bps_per_sqrt_s=0.0,
    activity_floor=1.0,
    activity_cap=1.0,
    prior_sd_bps=10.0,
    basis_prior_sd_bps=5.0,
    anchor_weights=(1.0, 0.0),
)


@known
def test_two_legs_one_update_each_by_hand():
    # Event 1, leg 0 at t = 0, z1 = ln 100. The level starts at the quote, x_p = z1, so the
    # innovation is 0 and x does not move. The state is [p, b0, b1] with b0 the anchor's,
    # zeroed with its row and column: P = diag(p0^2, 0, bp^2), p0 = 1e-3, bp = 5e-4. H = (1,
    # 1, 0), so P H' = (p0^2, 0, 0), S = p0^2 + n^2 = 1e-6 + 1e-8, the gain K = (p0^2 / S, 0,
    # 0) = (0.990099, 0, 0), and P_pp falls to p0^2 n^2 / S = 1e-14 / 1.01e-6 = 9.90099e-9.
    # Event 2, leg 1 one second later, z2 = ln 100.05. Predict: P_pp += sigma^2 x 1 s = 1e-8,
    # 1.990099e-8; the basis does not wander. H = (1, 0, 1), so P H' = (P_pp, 0, bp^2) and
    # S = P_pp + bp^2 + n^2 = 1.990099e-8 + 2.5e-7 + 1e-8 = 2.7990099e-7. The innovation is
    # y = ln(100.05 / 100) = 4.99875e-4, and the gain splits it between the price and leg 1's
    # basis in proportion to their variances: K_p = P_pp / S = 0.0711002, K_b1 = bp^2 / S =
    # 0.893173. So the theo moves 0.355 bps, to 100.00355, and the basis takes 4.465 bps:
    # one quote of a leg with an uncertain basis is read mostly as that leg's offset.
    z1, z2 = math.log(100.0), math.log(100.05)
    out = theo.run(_events([0, 1], [z1, z2], [0.0, 1.0]), HAND)
    p0, bp, n2 = 1e-3**2, 5e-4**2, 1e-4**2
    p1 = p0 * n2 / (p0 + n2)
    p2 = p1 + 1e-8
    s = p2 + bp + n2
    y = z2 - z1
    kp, kb = p2 / s, bp / s
    assert out["theo"][0] == pytest.approx(100.0, rel=1e-15)
    assert out["innovation_z"][0] == 0.0
    assert out["sd_bps"][0] == pytest.approx(math.sqrt(p1) / 1e-4, rel=1e-12)
    assert out["theo"][1] == pytest.approx(100.0 * math.exp(kp * y), rel=1e-14)
    assert out["basis_bps"][1, 1] == pytest.approx(kb * y / 1e-4, rel=1e-12)
    assert out["basis_bps"][1, 0] == 0.0  # the anchor's basis is zero by definition
    assert out["innovation_z"][1] == pytest.approx(y / math.sqrt(s), rel=1e-12)
    # after the update P_pp = P_pp - K_p P_pp, so the sd is sqrt(P_pp (1 - K_p))
    assert out["sd_bps"][1] == pytest.approx(math.sqrt(p2 * (1 - kp)) / 1e-4, rel=1e-12)
    assert (round(kp, 6), round(kb, 6)) == (0.0711, 0.893173)
    assert round(out["theo"][1], 5) == 100.00355 and round(out["basis_bps"][1, 1], 3) == 4.465
    # the likelihood is the two innovations' Gaussian log densities
    nll = 0.5 * (math.log(2 * math.pi * (p0 + n2)) + math.log(2 * math.pi * s) + y * y / s)
    assert theo.nll(_events([0, 1], [z1, z2], [0.0, 1.0]), HAND) == pytest.approx(nll, rel=1e-12)


def _random_stream(n=4000, legs=3, seed=3):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(0.05, n))
    leg = rng.integers(0, legs, n)
    price = np.cumsum(rng.normal(0, 1e-4, n)) + np.array([0.0, 2e-4, -3e-4])[leg]
    return _events(leg, math.log(118.0) + price + rng.normal(0, 5e-5, n), t,
                   stale=rng.uniform(0, 0.04, n), act=rng.uniform(0.1, 20, n))  # fmt: skip


@known
@pytest.mark.parametrize("anchor", [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
def test_the_filter_on_a_prefix_is_the_prefix_of_the_filter(anchor):
    params = TheoParams(sigma_bps_per_sqrt_s=2.0, leg_noise_bps=(0.4, 0.8, 1.5),
                        event_clock_weight=0.3, anchor_weights=anchor)  # fmt: skip
    ev = _random_stream()
    whole = theo.run(ev, params)
    half = theo.run({k: v[:1500] for k, v in ev.items()}, params)
    for key in ("theo", "sd_bps", "innovation_z", "basis_bps", "basis_sd_bps"):
        assert np.array_equal(half[key], whole[key][:1500]), key
    if anchor == (1.0, 1.0, 1.0):  # a consensus: the bases sum to zero after every event
        assert np.abs(whole["basis_bps"].sum(axis=1)).max() < 1e-9


def _theo_panel(n_grid=1200):
    """A panel of one coin on two venues with the theo's stored output as THEO.X."""
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    ev = _random_stream(6000, 2, seed=5)
    params = TheoParams(sigma_bps_per_sqrt_s=2.0, leg_noise_bps=(0.4, 0.8),
                        anchor_weights=(1.0, 0.0))  # fmt: skip
    out = theo.run(ev, params)
    ts = pl.Series([t0 + timedelta(seconds=float(s)) for s in ev["t"]]).cast(
        pl.Datetime("ns", "UTC")
    )
    frame = pl.DataFrame(
        {
            "ts": ts,
            "available_at": ts,
            "theo": out["theo"],
            "sd_bps": out["sd_bps"],
            "innovation_z": out["innovation_z"],
        }
    )
    grid = np.array([int(pd.Timestamp(t0).value) + int(i * 2e8) for i in range(n_grid)])
    return Panel.build({("theo", "THEO.X"): Events(frame, 0)}, grid, "UTC", int(2e8))


@known
def test_f_theo_passes_the_truncation_test_and_a_centred_version_fails():
    p = _theo_panel()
    window = bl.Window("w", p.time(0) + pd.Timedelta("10s"), p.time(-1))
    ok = truncation.check(lambda panel: F.theo("X"), p, window, seed=1)
    assert ok.ok, str(ok)
    assert np.isfinite(ok.full).mean() > 0.9
    # the same price centred on each point, the mean of the 2 s before and after, reads ahead
    centred = (F.theo("X").shift(10) + F.theo("X").shift(-10)) / 2
    bad = truncation.check(lambda panel: centred, p, window, seed=1)
    assert not bad.ok


@pytest.fixture(scope="module")
def theo_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("theo")
    data = make_synthetic(root / "events.parquet", minutes=60)
    raw = brief_dict(data)
    raw["stats"]["n_boot"] = 50
    raw["data"]["theo"] = {"coin": "XRPUSDT", "legs": ["XRPUSDT_PERP", "XRPUSDT"],
                           "anchor": "XRPUSDT_PERP"}  # fmt: skip
    brief = write_brief(raw, root / "brief.yaml")
    return vault.init(brief, root / "run"), root


@known
def test_init_fits_on_train_and_the_theo_columns_stand_when_rows_are_appended(theo_run):
    run, root = theo_run
    meta = json.loads((run.vault / "theo.json").read_text())
    wins = run.windows()
    assert meta["fit"]["window_end"] == wins["train"].end.value
    assert meta["fit"]["conflate_ns"] == 1_000_000_000
    assert meta["params"]["basis_wander_bps_per_sqrt_s"] == 0.02  # held, not fitted
    assert meta["fit"]["nll"] <= meta["fit"]["nll_start"]
    # the agent's copy holds train's output only, and its brief names it
    ws = yaml.safe_load((run.workspace / "brief.yaml").read_text())
    assert ws["data"]["theo"]["path"] == "data/train.theo.parquet"
    stored = pl.read_parquet(run.workspace / "data/train.theo.parquet")
    assert stored["ts"].max() < wins["train"].end
    # validation's panel and test's agree on every theo column over validation's rows
    val, test = run.panel("val"), run.panel("test")
    cols = [c for c in val.frame.columns if c.startswith("THEO.XRPUSDT.")]
    assert {"THEO.XRPUSDT.theo", "THEO.XRPUSDT.sd_bps", "THEO.XRPUSDT.innovation_z",
            "THEO.XRPUSDT.basis_XRPUSDT"} <= set(cols)  # fmt: skip
    n = len(val.frame)
    for c in cols:
        assert val.frame[c].equals(test.frame[c].head(n)), c
    # the theo sits on the perpetual's mid, its anchor, within a few bps
    gap = (val.frame["THEO.XRPUSDT.theo"] / val.frame["XRPUSDT_PERP.mid"] - 1).abs() * 1e4
    assert gap.drop_nulls().median() < 2.0


@known
def test_the_innovation_report_has_its_columns(theo_run):
    run, _ = theo_run
    meta = json.loads((run.vault / "theo.json").read_text())
    for seg in ("train", "val", "test"):
        rows = meta["innovations"][seg]
        assert [r["leg"] for r in rows] == ["XRPUSDT_PERP", "XRPUSDT"]
        for r in rows:
            assert set(r) == {"leg", "n", "z_mean", "z_var", "z_lag1"} and r["n"] > 100
    text = (run.workspace / "DATA.md").read_text()
    assert "## The theo" in text and "| train | XRPUSDT |" in text
    assert "| val |" not in text  # a protected run's agent sees train's report only


RUN = local("runs/arb-atusdt-fake-bsc")


@pytest.mark.known_answer
@pytest.mark.skipif(not RUN.is_dir(), reason="the published AT pair is not here")
def test_the_theo_fed_in_pieces_is_the_theo_filtered_whole(tmp_path):
    run = vault.RunDir(RUN)
    brief = run.brief()
    params = theo.TheoParams.from_dict(json.loads((run.vault / "theo.json").read_text())["params"])
    frame = pl.read_parquet(run.segment_path("test"))
    frame = frame.filter(pl.col("ts").dt.epoch("ns") < int(frame["ts"].dt.epoch("ns")[0]) + 1800e9)
    whole = theo.refilter(frame, brief, brief.data.theo, params)
    live = theo.Live(brief, brief.data.theo, params)
    rng = np.random.default_rng(1)
    # pieces end between two stamps (every row at or before g)
    ts = frame["ts"].dt.epoch("ns").to_numpy()
    starts = np.flatnonzero(np.diff(ts) > 0) + 1
    edges = np.sort(rng.choice(starts, 300, replace=False))
    pieces = [live.add(frame.slice(int(a), int(b - a)))
              for a, b in zip([0, *edges], [*edges, frame.height], strict=True)]  # fmt: skip
    assert pl.concat(pieces).equals(whole)
    assert whole.height > 1000
