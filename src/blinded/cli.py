"""The ``blinded`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.run(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blinded", description="Judge trading features by recomputation."
    )
    sub = parser.add_subparsers(required=True, metavar="command")

    ev = sub.add_parser("eval", help="score the features in a file")
    ev.add_argument("file", type=Path, help="python file with @bl.feature functions")
    ev.add_argument("--brief", required=True, type=Path)
    ev.add_argument("--window", default="train", help="train, val, test, all or START/END")
    ev.add_argument("--name", help="evaluate only the feature with this name")
    ev.add_argument("--latency-ms", type=float, help="override execution.latency_ms")
    ev.set_defaults(run=_eval)

    data = sub.add_parser("data", help="write DATA.md with clock diagnostics")
    data.add_argument("--brief", required=True, type=Path)
    data.add_argument("-o", "--out", type=Path, default=Path("DATA.md"))
    data.set_defaults(run=_data)

    init = sub.add_parser("init", help="split the data into a workspace and a vault")
    init.add_argument("brief", type=Path)
    init.add_argument("--run-dir", required=True, type=Path)
    init.set_defaults(run=_init)

    submit = sub.add_parser("submit", help="send a file to the judge from inside a workspace")
    submit.add_argument("file", type=Path)
    submit.add_argument("--name", help="judge only the feature or strategy with this name")
    submit.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait")
    submit.add_argument("--window", help="naive runs only: the window to score")
    submit.set_defaults(run=_submit)

    daemon = sub.add_parser("daemon", help="run the judge for a run directory")
    daemon.add_argument("run_dir", type=Path)
    daemon.add_argument("--once", action="store_true", help="answer what is waiting, then exit")
    daemon.set_defaults(run=_daemon)

    test = sub.add_parser("test", help="score one trial on the test segment, once per run")
    test.add_argument("run_dir", type=Path)
    test.add_argument("--trial", required=True, type=int)
    test.set_defaults(run=_test)

    demo = sub.add_parser("demo", help="build public demo data")
    source = demo.add_subparsers(required=True, metavar="source")
    tardis = source.add_parser("tardis", help="a free Tardis day of Binance spot and perp L1")
    tardis.add_argument("--date", required=True, help="a first of the month, YYYY-MM-01")
    tardis.add_argument("--symbols", required=True, help="comma-separated, e.g. XRPUSDT")
    tardis.add_argument("--out", required=True, type=Path)
    tardis.add_argument("--hours", type=float, help="keep the first HOURS hours of the day")
    tardis.add_argument("--book", default="book_ticker", choices=["book_ticker", "quotes"])
    tardis.set_defaults(run=_demo_tardis)
    bn = source.add_parser("binance", help="days of Binance USD-M perpetuals, thinned")
    bn.add_argument("--from", dest="start", required=True, help="the first day, YYYY-MM-DD")
    bn.add_argument("--days", type=int, default=3)
    bn.add_argument("--symbols", default=",".join(demo_symbols()),
                    help="comma-separated, default %(default)s")  # fmt: skip
    bn.add_argument("--out", required=True, type=Path)
    bn.add_argument("--step", default="500ms", help="grid step, default %(default)s")
    bn.add_argument("--latency-ms", type=float, default=50.0, help="order latency")
    bn.add_argument("--delay-ms", type=float, default=50.0, help="feed delay")
    bn.add_argument("--keep-raw", action="store_true", help="keep the downloaded zips")
    bn.add_argument("--force", action="store_true", help="build past the 10M-row limit")
    bn.set_defaults(run=_demo_binance)
    tr = source.add_parser("trend", help="four USD-M perpetuals on hourly bars, 2023-05-17 on")
    tr.add_argument("--out", required=True, type=Path, help="for the parquet, brief, DATA-raw.md")
    tr.add_argument("--raw", type=Path, help="for the downloaded zips; default OUT/raw")
    tr.add_argument("--costs", type=Path, help="for the half-spread cache; default OUT/costs")
    tr.add_argument("--keep-raw", action="store_true", help="keep the downloaded zips")
    tr.set_defaults(run=_demo_trend)
    xs = source.add_parser("crosssection", help="twenty USD-M perpetuals on 15-minute bars")
    xs.add_argument("--out", required=True, type=Path, help="for the parquet, brief, DATA-raw.md")
    xs.add_argument("--raw", type=Path, help="for the downloaded zips; default OUT/raw")
    xs.add_argument("--costs", type=Path, help="for the half-spread cache; default OUT/costs")
    xs.add_argument("--keep-raw", action="store_true", help="keep the downloaded zips")
    xs.set_defaults(run=_demo_crosssection)
    ca = source.add_parser("carry", help="eight coins' spot and perpetual with funding, 15m bars")
    ca.add_argument("--out", required=True, type=Path, help="for the parquet, brief, DATA-raw.md")
    ca.add_argument("--raw", type=Path, help="for the downloaded zips; default OUT/raw")
    ca.add_argument("--costs", type=Path, help="for the half-spread caches; default OUT/costs")
    ca.add_argument(
        "--tardis-raw",
        type=Path,
        help="for the Tardis spot book_ticker gz files, kept; default RAW",
    )
    ca.add_argument("--keep-raw", action="store_true", help="keep the downloaded zips")
    ca.set_defaults(run=_demo_carry)
    cv = source.add_parser("crossvenue", help="the XRP perpetual on Binance and a second venue")
    cv.add_argument("--out", required=True, type=Path, help="for the parquet, brief, DATA-raw.md")
    cv.add_argument("--raw", type=Path, help="for the Tardis gz files, kept; default OUT/raw")
    cv.set_defaults(run=_demo_crossvenue)
    mk = source.add_parser("mm", help="a USD-M perpetual's day for the fill engine, from Tardis")
    mk.add_argument("--symbol", required=True, help="SOLUSDT or LTCUSDT (any listed perpetual)")
    mk.add_argument("--date", required=True, help="a first of the month, YYYY-MM-DD")
    mk.add_argument("--out", required=True, type=Path, help="for the files and DATA-raw.md")
    mk.add_argument("--raw", type=Path, help="for the Tardis gz files, kept; default OUT/raw")
    mk.add_argument("--levels", type=int, default=5, help="book levels kept, the touch included")
    mk.add_argument("--latency-ms", type=float, default=50.0, help="the brief's latency")
    mk.add_argument("--hours", type=float, help="keep the first HOURS hours of the day")
    mk.set_defaults(run=_demo_mm)
    ab = source.add_parser("arb", help="a coin on a thin spot venue and a deep one, from Tardis")
    ab.add_argument("--coin", default="ATUSDT", help="default ATUSDT")
    ab.add_argument("--date", default="2026-10-01", help="a first of the month, YYYY-MM-DD")
    ab.add_argument("--quote", default="mexc", help="the thin venue, default mexc")
    ab.add_argument("--hedge", default="binance", help="the deep venue, default binance")
    ab.add_argument("--out", required=True, type=Path, help="for the files and DATA-raw.md")
    ab.add_argument("--raw", type=Path, help="for the Tardis gz files, kept; default OUT/raw")
    ab.set_defaults(run=_demo_arb)
    vn = source.add_parser("venues", help="a coin's spot touch on four venues on one clock")
    vn.add_argument("--coin", default="SOLUSDT", help="default SOLUSDT")
    vn.add_argument("--date", default="2026-10-01", help="a first of the month, YYYY-MM-DD")
    vn.add_argument("--out", required=True, type=Path, help="for the files and DATA-raw.md")
    vn.add_argument("--raw", type=Path, help="for the Tardis gz files, kept; default OUT/raw")
    vn.set_defaults(run=_demo_venues)

    rec = sub.add_parser("record", help="record a coin's public MEXC and Binance spot streams")
    rec.add_argument("--out", required=True, type=Path, help="a directory a UTC day goes under")
    rec.add_argument("--coin", default="ATUSDT", help="default ATUSDT")
    rec.add_argument("--hours", type=float, help="stop after this many hours; default never")
    rec.set_defaults(run=_record)

    fw = sub.add_parser("forward", help="judge a run's final on recorded days; or serve the page",
                        description="blinded forward RUN_DIR --live-data DIR judges the final on "
                        "every recorded day; blinded forward serve RUN_DIR [--port] serves "
                        "vault/forward/forward.html on 127.0.0.1")  # fmt: skip
    fw.add_argument("first", help="the run directory, or serve")
    fw.add_argument("run_dir", nargs="?", type=Path, help="with serve: the run directory")
    fw.add_argument("--live-data", type=Path, help="the recorder's directory of days")
    fw.add_argument(
        "--latency-ms",
        type=float,
        help="every latency, the hedge's too, in place of the measured ones",
    )
    fw.add_argument(
        "--segment",
        choices=["test"],
        help="judge the vault's own test segment instead (the known answer)",
    )
    fw.add_argument(
        "--watch",
        action="store_true",
        help="judge again ten minutes after every hour, until stopped",
    )
    fw.add_argument("--port", type=int, default=8766, help="with serve; default 8766")
    fw.set_defaults(run=_forward)

    run = sub.add_parser("run", help="run the agent loop on a brief")
    run.add_argument("brief", type=Path)
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument(
        "--mode", choices=["protected", "naive"], help="default: the brief's, else protected"
    )
    run.add_argument("--resume", metavar="RUN_ID", help="continue the run in RUN_DIR")
    run.add_argument("--model", help="default: the brief's budget.model, else opus")
    run.add_argument("--effort", help="default: the brief's budget.effort, else high")
    run.add_argument("--hours", type=float, dest="wall_hours", help="budget.wall_hours")
    run.add_argument("--max-iterations", type=int, help="budget.max_iterations")
    run.add_argument("--max-usd", type=float, help="budget.max_usd")
    run.add_argument("--iteration-usd", type=float, help="budget.iteration_usd")
    run.add_argument("--max-turns", type=int, help="budget.max_turns")
    run.set_defaults(run=_run)

    rep = sub.add_parser("report", help="write report.html and report.json for a run")
    rep.add_argument("run_dir", type=Path)
    rep.set_defaults(run=_report)

    rp = sub.add_parser("replay", help="recompute the report's numbers from the data and check")
    rp.add_argument("run_dir", type=Path)
    rp.set_defaults(run=_replay)

    track = sub.add_parser("mlflow", help="mirror a run's ledger into MLflow (optional extra)")
    track.add_argument("run_dir", type=Path)
    track.add_argument("--uri", help="tracking URI; default sqlite:///RUN_DIR/vault/mlflow.db")
    track.set_defaults(run=_mlflow)
    return parser


def _eval(args) -> int:
    from blinded import brief as briefs
    from blinded.contract import load
    from blinded.evaluate import evaluate, evaluate_strategy, panel_for

    brief = briefs.load(args.brief)
    module = load(args.file)
    features = [f for f in module.features if args.name in (None, f.name)]
    strategies = [s for s in module.strategies if args.name in (None, s.name)]
    if not features and not strategies:
        have = ", ".join(f.name for f in [*module.features, *module.strategies]) or "none"
        print(f"no feature named {args.name!r} in {args.file} (have: {have})", file=sys.stderr)
        return 2
    panel = panel_for(brief)
    results = []
    for feat in features:
        result = evaluate(feat, brief, args.window, args.latency_ms, panel=panel)
        results.append(result)
        print(result, end="\n\n")
    for strat in module.strategies:
        if args.name in (None, strat.name):
            result = evaluate_strategy(strat, brief, args.window, args.latency_ms, panel=panel)
            results.append(result)
            print(result, end="\n\n")
    return 0 if all(r.ok for r in results) else 1


def _data(args) -> int:
    from blinded import brief as briefs
    from blinded.diagnostics import write_data_md
    from blinded.evaluate import panel_for

    brief = briefs.load(args.brief)
    write_data_md(brief, panel_for(brief), args.out)
    print(f"wrote {args.out}")
    return 0


def _init(args) -> int:
    from blinded import vault

    try:
        run = vault.init(args.brief, args.run_dir)
    except (vault.InsideGitRepo, FileExistsError) as exc:
        print(f"blinded init: {exc}", file=sys.stderr)
        return 2
    manifest = run.manifest
    print(f"workspace {run.workspace}")
    print(f"vault     {run.vault}")
    for name, (start, end) in manifest["windows"].items():
        seg = manifest["segments"][name]
        print(f"{name:<5}  {start} to {end}  {seg['rows']:>9,} rows  {seg['sha256'][:12]}")
    if run.brief().has_archetypes:
        print(f"{run.menu().headline} See workspace/ARCHETYPES.md.")
    return 0


def _submit(args) -> int:
    import json

    from blinded.client import submit

    try:
        response = submit(args.file, args.name, args.timeout, window=args.window)
    except (FileNotFoundError, TimeoutError) as exc:
        print(f"blinded submit: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(response, indent=2))
    trials = response.get("trials", [])
    return 0 if trials and all(t["status"] == "accepted" for t in trials) else 1


def _daemon(args) -> int:
    from blinded.daemon import Daemon
    from blinded.vault import RunDir

    Daemon(RunDir.open(args.run_dir)).serve(once=args.once)
    return 0


def _test(args) -> int:
    import json

    from blinded.daemon import score_test
    from blinded.ledger import AlreadyScored
    from blinded.vault import RunDir

    try:
        metrics = score_test(RunDir.open(args.run_dir), args.trial)
    except AlreadyScored as exc:
        print(f"blinded test: {exc}; the test segment is scored once", file=sys.stderr)
        return 3
    print(json.dumps(metrics, indent=2))
    return 0


def _demo_tardis(args) -> int:
    import polars as pl

    from blinded import brief as briefs
    from blinded import demo

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        brief = demo.tardis(args.date, symbols, args.out, args.hours, args.book)
    except (ImportError, ValueError) as exc:
        print(f"blinded demo: {exc}", file=sys.stderr)
        return 2
    data = briefs.load(brief).data.path
    print(demo.summary(pl.read_parquet(data)))
    print(f"wrote {data}\nwrote {brief}")
    return 0


def demo_symbols() -> tuple[str, ...]:
    from blinded.demo import SYMBOLS

    return SYMBOLS


def _demo_binance(args) -> int:
    from blinded import demo

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        built = demo.binance(
            args.start, args.days, args.out, symbols, args.step, args.latency_ms,
            args.delay_ms, args.keep_raw, args.force,
        )  # fmt: skip
    except ValueError as exc:
        print(f"blinded demo: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {built.data}\nwrote {built.brief}\nwrote {built.notes}")
    return 0


def _demo_trend(args) -> int:
    from blinded import demo

    try:
        built = demo.trend(args.out, args.raw, args.costs, keep_raw=args.keep_raw)
    except ValueError as exc:
        print(f"blinded demo: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {built.data}\nwrote {built.brief}\nwrote {built.notes}\n"
          f"downloaded {built.downloaded / 1e6:,.1f} MB")  # fmt: skip
    return 0


def _demo_crosssection(args) -> int:
    from blinded import demo

    try:
        built = demo.crosssection(args.out, args.raw, args.costs, keep_raw=args.keep_raw)
    except ValueError as exc:
        print(f"blinded demo: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {built.data}\nwrote {built.brief}\nwrote {built.notes}\n"
          f"downloaded {built.downloaded / 1e6:,.1f} MB")  # fmt: skip
    return 0


def _demo_carry(args) -> int:
    from blinded import demo

    try:
        built = demo.carry(args.out, args.raw, args.costs, args.tardis_raw, keep_raw=args.keep_raw)
    except (ImportError, ValueError) as exc:
        print(f"blinded demo: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {built.data}\nwrote {built.brief}\nwrote {built.notes}\n"
          f"downloaded {built.downloaded / 1e6:,.1f} MB, {built.tardis_bytes / 1e6:,.1f} MB of "
          "it from Tardis")  # fmt: skip
    return 0


def _demo_crossvenue(args) -> int:
    from blinded import demo

    try:
        built = demo.crossvenue(args.out, args.raw)
    except (ImportError, ValueError) as exc:
        print(f"blinded demo: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {built.data}\nwrote {built.brief}\nwrote {built.notes}")
    return 0


def _demo_mm(args) -> int:
    from blinded import demo_mm

    try:
        built = demo_mm.mm(args.symbol, args.date, args.out, args.raw, args.levels,
                           args.latency_ms, args.hours)  # fmt: skip
    except (ImportError, ValueError) as exc:
        print(f"blinded demo: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {built.data}\nwrote {built.touch}\nwrote {built.brief}\nwrote {built.notes}")
    return 0


def _demo_arb(args) -> int:
    from blinded import demo_mm

    try:
        built = demo_mm.arb(args.coin, args.date, args.out, args.raw, args.quote, args.hedge)
    except (ImportError, ValueError) as exc:
        print(f"blinded demo: {exc}", file=sys.stderr)
        return 2
    for path in (built.data, built.touch, built.depth, built.brief, built.notes):
        print(f"wrote {path}")
    return 0


def _record(args) -> int:
    from datetime import UTC, datetime

    from blinded import recorder

    def log(message: str) -> None:
        print(f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}", flush=True)

    try:
        recorder.record(args.out, args.coin, args.hours, log)
    except ImportError:
        print("blinded record needs aiohttp: pip install 'blinded[live]'", file=sys.stderr)
        return 2
    except recorder.ClockError as exc:
        print(f"blinded record: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


def _forward(args) -> int:
    from blinded import forward, isolate

    if args.first == "serve":
        if args.run_dir is None:
            print("blinded forward serve: name the run directory", file=sys.stderr)
            return 2
        forward.serve(args.run_dir, args.port)
        return 0
    if args.run_dir is not None:
        print("blinded forward: one run directory, or serve and one", file=sys.stderr)
        return 2
    try:
        if args.watch:
            forward.watch(Path(args.first), args.live_data, args.latency_ms)
        forward.run_forward(Path(args.first), args.live_data, args.latency_ms, args.segment)
    except (forward.ForwardError, isolate.Unfenced) as exc:
        print(f"blinded forward: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {forward.forward_dir(_rundir(args.first)) / 'forward.html'}")
    return 0


def _rundir(path):
    from blinded.vault import RunDir

    return RunDir(Path(path).expanduser().resolve())


def _demo_venues(args) -> int:
    from blinded import demo_mm

    try:
        built = demo_mm.venues(args.coin, args.date, args.out, args.raw)
    except (ImportError, ValueError) as exc:
        print(f"blinded demo: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {built.data}\nwrote {built.brief}\nwrote {built.notes}")
    return 0


def _run(args) -> int:
    import yaml

    from blinded import driver, vault
    from blinded.vault import RunDir

    names = ("model", "effort", "wall_hours", "max_iterations", "max_usd", "iteration_usd")
    flags = {k: getattr(args, k) for k in (*names, "max_turns")}
    if args.resume:
        run = RunDir.open(args.run_dir)
        if run.run_id != args.resume:
            message = f"{args.run_dir} holds run {run.run_id}, not {args.resume}"
            print(f"blinded run: {message}", file=sys.stderr)
            return 2
        state = json.loads((run.driver / "state.json").read_text())
        budget = driver.Budget.from_raw(state.get("budget"), **flags)
    else:
        raw = yaml.safe_load(args.brief.read_text())
        mode = args.mode or raw.get("mode") or "protected"
        try:
            run = driver.start(args.brief, args.run_dir, mode)
        except (vault.InsideGitRepo, FileExistsError, ValueError) as exc:
            print(f"blinded run: {exc}", file=sys.stderr)
            return 2
        budget = driver.Budget.from_raw(raw.get("budget"), **flags)
    print(f"run {run.run_id} ({run.mode}) in {run.root}")
    state = driver.Driver(run, budget).run(resume=bool(args.resume))
    final = state.get("final") or {}
    note = f" ({final['override']})" if final.get("override") else ""
    print(f"stopped: {state.get('stop')}; final trial: {final.get('trial_id')}{note}")
    return 0 if state.get("status") == "finished" else 1


def _report(args) -> int:
    from blinded.report import report

    path = report(args.run_dir)
    print(f"wrote {path}")
    return 0


def _replay(args) -> int:
    from blinded.report import replay

    result = replay(args.run_dir)
    print("verified: the segment, touch and depth files and the manifest match the hashes "
          "written at init")  # fmt: skip
    for c in result["checks"]:
        if not c["ok"]:
            print(f"MISMATCH {c['name']}: report {c['reported']} vs data {c['recomputed']}",
                  file=sys.stderr)  # fmt: skip
    if not result["checks"]:
        print("nothing to recompute: no strategy was scored on test")
        return 0
    print(f"{len(result['checks'])} numbers checked; "
          + ("all match the data" if result["ok"] else "MISMATCHES above"))  # fmt: skip
    return 0 if result["ok"] else 1


def _mlflow(args) -> int:
    from blinded import tracking
    from blinded.vault import RunDir

    run = RunDir.open(args.run_dir)
    try:
        exp = tracking.mirror(run, args.uri)
    except ImportError as exc:
        print(f"blinded mlflow: {exc}", file=sys.stderr)
        return 2
    print(f"experiment {exp}; browse with:\n  {tracking.ui_command(run, args.uri)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
