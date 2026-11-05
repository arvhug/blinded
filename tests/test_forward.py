"""Forward paper: the CUSUM and its bootstrap, the forward brief, the fence, the known answer
on the published AT pair's own test segment, and a recorded day with a planted gap."""

import json
import math
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from conftest import local

from blinded import forward, isolate, recorder
from blinded.vault import RunDir

known = pytest.mark.known_answer
edge = pytest.mark.edge_case

# the AT day's 22 whole hours (1 to 22) of the whole-day pass an independent script
# bootstraps (RA, an unbinding stock, the brief's 50 ms): net USD and fills
AT_NET = [0.080857600, 0.359920550, 0.985774924, 0.602164284, 0.853927714, 0.571051101,
          0.273357046, 0.383664949, 0.776369766, 0.517052137, 0.625910800, 0.529162386,
          0.861949761, 0.746185658, 0.269670202, 0.634124330, 0.486440427, 0.565423715,
          0.572794028, 1.021099194, 0.301214797, 0.382698000]  # fmt: skip
AT_FILLS = [5, 20, 51, 37, 55, 32, 21, 24, 40, 35, 35, 35, 52, 37, 21, 35, 24, 36, 29, 58, 20, 25]

RUN = local("runs/arb-atusdt-fake-bsc")


@known
def test_the_cusum_alarms_at_the_hour_derived_by_hand():
    # m 0.5 and s 0.25: z is 0, -1, -2, -2, -2, -2; C_t = max(0, C - z - 0.5) runs 0, 0.5, 2,
    # 3.5, 5 and 6.5, and the alarm needs C over h = 5, so the sixth hour (index 5), not the
    # fifth, which only reaches it
    c, at = forward.cusum(np.array([0.5, 0.25, 0.0, 0.0, 0.0, 0.0]), m=0.5, s=0.25)
    assert c.tolist() == [0.0, 0.5, 2.0, 3.5, 5.0, 6.5]
    assert at == 5
    # an hour above the mean pulls it back down, never under 0
    c, at = forward.cusum(np.array([0.0, 1.5, 1.5, 0.0]), m=0.5, s=0.25)
    assert c.tolist() == [1.5, 0.0, 0.0, 1.5] and at is None


@known
def test_the_bootstrap_gives_the_specs_false_alarm_rate_within_one_point():
    # the independent script at h = 5: 9.5% of weeks on the net, 9.9% on fills
    # (20,000 weeks of 168 hours)
    net = forward.false_alarms(np.array(AT_NET), seed=1)
    fills = forward.false_alarms(np.array(AT_FILLS, dtype=float), seed=1)
    assert abs(net["share"] - 0.095) < 0.01, net
    assert abs(fills["share"] - 0.099) < 0.01, fills
    assert round(net["siegmund_arl_hours"]) == 285
    halved = forward.false_alarms(np.array(AT_NET), scale=0.5, weeks=5000, seed=2)
    assert halved["share"] > 0.99 and 6 <= halved["median_hours"] <= 10


@known
def test_a_threshold_holds_its_rate_once_the_references_own_error_is_counted():
    # the 9.5% above takes m and s as known; the page's come from the locked test's whole
    # hours, five for the AT pair, and a mean from five hours is off by 0.45 sd on average,
    # which the CUSUM reads as drift. Recomputed by hand (a separate script, 20,000 weeks,
    # each week's m and s from 5 or 22 hours drawn from these 22): h = 5 alarms in 43.7% of
    # weeks with 5 reference hours and 27.8% with 22; 10% needs h = 38 and h = 8.
    x = np.array(AT_NET)
    assert 0.41 < forward.false_alarms(x, n=5, seed=7)["share"] < 0.47
    assert 0.25 < forward.false_alarms(x, n=22, seed=7)["share"] < 0.31
    h5, h22, h_known = (forward.threshold(x, n) for n in (5, 22, None))
    assert 36 <= h5 <= 39 and 7 <= h22 <= 9 and 4.5 <= h_known <= 5.2
    # and each holds 10% on weeks drawn afresh
    for n, h in ((5, h5), (22, h22), (None, h_known)):
        assert abs(forward.false_alarms(x, h=h, n=n, seed=8)["share"] - 0.10) < 0.01


@known
def test_the_forward_hours_own_interval_covers_and_says_where_zero_is():
    # under 20 hours it claims nothing; hours with no spread are their own mean
    assert forward.own_interval(np.array(AT_NET[:19]))["side"] is None
    flat = forward.own_interval(np.full(20, 0.25))
    assert (flat["lo"], flat["hi"], flat["side"]) == (0.25, 0.25, "above")
    # on symmetric hours the bootstrap-t is the t-interval, t(0.95, 23) = 1.714, to a few %
    x = np.linspace(-0.2, 1.0, 24)
    r = forward.own_interval(x)
    half = 1.714 * x.std(ddof=1) / np.sqrt(24)
    assert r["lo"] == pytest.approx(x.mean() - half, rel=0.05)
    assert r["hi"] == pytest.approx(x.mean() + half, rel=0.05)
    assert r["side"] == "above"
    # and it covers the AT hours' mean at its 90%, 20 hours drawn at a time
    rng = np.random.default_rng(11)
    pop = np.array(AT_NET)
    hits = [
        (lambda o: o["lo"] <= pop.mean() <= o["hi"])(
            forward.own_interval(rng.choice(pop, 20), n_boot=1999, seed=i)
        )
        for i in range(600)
    ]
    assert 0.87 < np.mean(hits) < 0.93
    # a halved strategy is still above zero inside a week, which the CUSUM cannot tell apart
    # from an unchanged one
    week = forward.own_interval(rng.choice(pop, 168) * 0.5)
    assert week["side"] == "above"
    assert "made money" in forward.own_line(week)


def clone(src: Path, dst: Path) -> RunDir:
    """An APFS clone where cp can make one, else a copy: the published run is never written."""
    if sys.platform == "darwin":
        subprocess.run(["cp", "-cR", str(src), str(dst)], check=True)
    else:
        shutil.copytree(src, dst)
    return RunDir(dst)


needs_run = pytest.mark.skipif(not RUN.is_dir(), reason="the published AT pair is not here")


@edge
@needs_run
def test_without_srt_forward_refuses_and_writes_nothing(tmp_path, monkeypatch):
    run = clone(RUN, tmp_path / "run")
    monkeypatch.delenv("BLINDED_UNFENCED", raising=False)
    monkeypatch.delenv("BLINDED_SRT", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(isolate.Unfenced, match="only inside the OS sandbox runtime srt"):
        forward.run_forward(run.root, segment="test", latency_ms=50)
    assert not (run.vault / "forward").exists()


@edge
@needs_run
def test_the_forward_brief_is_the_vaults_but_for_the_day(tmp_path):
    run = clone(RUN, tmp_path / "run")
    final = forward.final_of(run)
    assert (final.name, final.trial_id, final.archetype) == ("thirty_bps", 3, "cross_venue_making")
    assert Path(final.code_path).parent == run.trials
    day = forward.segment_day(run)
    raw, brief = forward.forward_brief(run, day, {"latency_ms": 400, "hedge_latency_ms": 500},
                                       final)  # fmt: skip
    vault = yaml.safe_load((run.vault / "brief.yaml").read_text())
    changed = {k for k in raw if raw[k] != vault.get(k)}
    assert changed == {"data", "grid", "execution", "splits", "forward"}
    assert raw["splits"]["windows"] == {
        "forward": "2026-10-01T18:05:00+00:00/2026-10-02T00:00:00+00:00"}  # fmt: skip
    assert {k for k in raw["execution"] if raw["execution"][k] != vault["execution"][k]} == {
        "latency_ms", "hedge_latency_ms"}  # fmt: skip
    assert brief.execution.hedge_latency.value == 500_000_000
    assert brief.judge_rules == run.judge_rules == 3
    assert raw["forward"]["final"]["source_sha256"] == final.source_sha
    written = yaml.safe_load((run.vault / "forward" / "test.brief.yaml").read_text())
    assert written == raw


@known
@needs_run
def test_on_its_own_test_segment_forward_gives_the_locked_tests_numbers(tmp_path):
    # the forward judge (the final's code in a fenced worker, both queue models, the worse
    # scored) on the vault's test segment at the vault's 50 ms is the report's pass exactly
    run = clone(RUN, tmp_path / "run")
    report = json.loads((run.root / "report.json").read_text())["recompute"]["final"]
    (j,) = forward.run_forward(run.root, segment="test", latency_ms=50, log=lambda m: None)
    assert j.arb == report["arb"]
    assert j.transfers == report["transfers"]
    assert (j.arb["fills"], j.arb["net_usd"], j.transfers["count"]) == (161, 2.4411550499999444,
                                                                         4)  # fmt: skip
    assert sum(h["fills"] for h in j.hours) == 161
    assert sum(h["net_usd"] for h in j.hours) == pytest.approx(j.arb["net_usd"], abs=1e-12)
    db = sqlite3.connect(run.vault / "forward" / "ledger.sqlite")
    assert db.execute("SELECT COUNT(*) FROM hours").fetchone()[0] == 6
    (arb,) = db.execute("SELECT arb_json FROM days").fetchone()
    assert json.loads(arb) == report["arb"]
    assert (run.vault / "forward" / "forward.html").read_text().count("<svg") == 5


def recorded_day(run: RunDir, live: Path, gap_hour: int, missed: tuple[int, ...] = ()) -> Path:
    """The AT day, whole, laid out as the recorder writes a built day (its files from the
    vault's three segments), every hour but ``missed`` covered, one gap planted in
    ``gap_hour``."""
    from datetime import date

    day = date(2026, 10, 1)
    d = live / day.isoformat()
    d.mkdir(parents=True)
    stem = recorder.stem("ATUSDT", day)
    for kind, suffix in (("", ".parquet"), ("touch", ".touch.parquet"),
                         ("depth", ".depth.parquet")):  # fmt: skip
        parts = [run.segment_path(s) if not kind else run.side_path(s, kind)
                 for s in ("train", "val", "test")]  # fmt: skip
        pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(d / f"{stem}{suffix}")
    start = recorder.day_start(day)
    with open(d / "hours.jsonl", "w") as fh:
        for k in (k for k in range(24) if k not in missed):
            h = start + k * recorder.HOUR_NS
            fh.write(json.dumps({"hour": recorder.iso(h), "hour_ns": h, "part": 0, "from_ns": h,
                                 "to_ns": h + recorder.HOUR_NS, "rows": {}, "frames": {},
                                 "gaps": [], "dropped": 0, "clock": {}, "parts": {},
                                 "host": "a test"}) + "\n")  # fmt: skip
    g = start + gap_hour * recorder.HOUR_NS + 600 * 10**9
    recorder.Recording(live, "ATUSDT", log=lambda m: None).gap(
        recorder.Gap("binance", "trade", "sequence", 10, 13, g, g + 2 * 10**9)
    )
    return d


@known
@needs_run
def test_a_recorded_day_is_one_pass_and_its_gapped_hour_counts_for_nothing(tmp_path):
    run = clone(RUN, tmp_path / "run")
    live = tmp_path / "live"
    recorded_day(run, live, gap_hour=20, missed=(0,))
    said = []
    (j,) = forward.run_forward(run.root, live, latency_ms=50, log=said.append)
    status = {h["hour"][11:13]: (h["status"], h["reason"]) for h in j.hours}
    # 01:00 has no day before it was recorded; 00:00 was not recorded at all, and says so
    assert status["00"] == ("unobserved", "the recorder was not running")
    assert status["01"][0] == "warmup"
    assert status["20"] == ("gapped", "binance trade sequence")
    assert sum(s == "judged" for s, _ in status.values()) == 21
    # the day is one pass: its hours add up to the day's pass, judged or not
    assert sum(h["fills"] for h in j.hours) == j.arb["fills"]
    # and the stock is carried from 00:05: the day's 18:05 on is not the test's restart
    assert j.window.start.isoformat() == "2026-10-01T00:05:00+00:00"
    ledger = forward.ForwardLedger(run.vault / "forward" / "ledger.sqlite")
    hours = ledger.latest_hours()
    (day,) = ledger.days()
    ref = ledger.reference(j.latencies)
    ledger.close()
    assert (day["hours_judged"], day["hours_excluded"], day["note"]) == (21, 3, None)
    v = forward.verdict(hours, ref)
    judged = [h for h in hours if h["status"] == "judged"]
    assert v["n"] == 21 and v["net"] == pytest.approx(sum(h["net_usd"] for h in judged))
    assert v["fills"] == sum(h["fills"] for h in judged) < j.arb["fills"]
    page = (run.vault / "forward" / "forward.html").read_text()
    assert v["line"] in page.replace("&#x27;", "'") and "binance trade sequence" in page
    # the command and the days' table count the judged hours alone, as the verdict does
    assert said[0] == (f"2026-10-01: 21 of 24 hours judged, {v['fills']} fills and net "
                       f"{v['net']:+.4f} USD in them")  # fmt: skip
    assert f"<td>{v['fills']}</td>" in page and f"<td>{v['net']:+.4f}</td>" in page
    assert f"<td>{j.arb['fills']}</td>" not in page
    # the thresholds are the reference's, set for 10% a week with its own error counted
    assert v["h_net"] == ref["h_net"] > forward.H and v["h_fills"] == ref["h_fills"] > forward.H
    cal = json.loads(ref["calibration_json"])
    assert cal["from"] == "the vault day" and cal["hours"] > ref["n"]
    assert abs(cal["net"]["false_alarm"] - 0.10) < 0.015 and cal["net"]["estimated_at_5"] > 0.2
    assert forward.calibration_line(ref) in page.replace("&#x27;", "'")
    # 21 judged hours are past the 20 the forward hours' own interval waits for
    assert v["own"]["n"] == 21 and v["own"]["mean"] == pytest.approx(v["net"] / 21)
    assert v["own"]["side"] is not None and v["own_line"] in page.replace("&#x27;", "'")
    # a stop is the one change derived by hand: each hour's z is -m/s, so C grows by m/s - k
    # an hour and crosses h at the first whole hour past h / (m/s - k)
    grow = ref["m_net"] / ref["s_net"] - forward.K
    assert cal["net"]["stop"]["median_hours"] == math.floor(ref["h_net"] / grow) + 1
