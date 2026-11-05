"""The run directory: a workspace the agent works in and a vault it cannot read.

``blinded init brief.yaml --run-dir DIR`` cuts the data by time and writes

- ``DIR/workspace/``: the training rows (``data/train.parquet``), a brief that
  points at them, ``DATA.md``, ``RULES.md``, ``LEDGER.md``, ``features/``,
  ``strategies/`` and the file-drop directories ``.judge/inbox`` and
  ``.judge/outbox``; with an ``archetypes`` brief, ``ARCHETYPES.md`` too;
- ``DIR/vault/``: its own copy of the training rows, the validation and test
  rows, the same brief, ``manifest.json`` (split boundaries and SHA-256 hashes,
  never rows), ``ledger.sqlite`` and ``trials/``; with an ``archetypes`` brief,
  ``menu.json``, the verdict on every archetype;
- ``DIR/agent/``: the agent's sandbox settings and PreToolUse hook.

Only the judge daemon reads the vault.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from blinded import archetypes
from blinded import brief as briefs
from blinded.brief import Brief, Window
from blinded.durations import label
from blinded.ladder import RULES, check_rules
from blinded.ledger import Ledger
from blinded.ledger import render as render_ledger
from blinded.panel import Panel, event_times, load_events, make_grid

SEGMENTS = ("train", "val", "test")
# files a run splits beside the data file, which only the judge's sims read
SIDE_FILES = ("touch", "depth", "theo")


class InsideGitRepo(ValueError):
    """A data file or run directory sits inside a git work tree."""


@dataclass(frozen=True)
class RunDir:
    """Paths of one run directory, and the vault's view of the data."""

    root: Path

    @classmethod
    def open(cls, root: str | Path) -> RunDir:
        run = cls(Path(root).expanduser().resolve())
        if not (run.vault / "manifest.json").is_file():
            raise FileNotFoundError(f"{run.root} is not a run directory (no vault/manifest.json)")
        return run

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def vault(self) -> Path:
        return self.root / "vault"

    @property
    def driver(self) -> Path:
        """The driver's state and session transcripts; denied to the agent like the vault."""
        return self.root / "driver"

    @property
    def inbox(self) -> Path:
        return self.workspace / ".judge" / "inbox"

    @property
    def outbox(self) -> Path:
        return self.workspace / ".judge" / "outbox"

    @property
    def ledger_path(self) -> Path:
        return self.vault / "ledger.sqlite"

    @property
    def trials(self) -> Path:
        return self.vault / "trials"

    def segment_path(self, name: str) -> Path:
        return self.vault / f"{name}.parquet"

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def mode(self) -> str:
        """``protected``, or ``naive`` when the workspace holds all of the data."""
        return self.manifest.get("mode", "protected")

    def ledger(self) -> Ledger:
        return Ledger(self.ledger_path)

    def write_ledger_md(self, ledger: Ledger) -> None:
        """Regenerate ``workspace/LEDGER.md``; edits the agent made to it are overwritten."""
        brief = self.brief()
        horizons = [label(h) for h in brief.target.horizons]
        text = render_ledger(ledger, horizons, self.mode, families=brief.has_families,
                             archetypes=brief.has_archetypes)  # fmt: skip
        (self.workspace / "LEDGER.md").write_text(text)

    def menu(self) -> archetypes.Menu:
        """The menu ``blinded init`` wrote: what every archetype is on this data."""
        return archetypes.Menu.from_dict(json.loads((self.vault / "menu.json").read_text()))

    def groups(self) -> tuple[briefs.Group, ...]:
        """The run's groups as ``blinded init`` fixed them: from ``vault/menu.json`` in a run with
        archetypes, so a run reads the menu it was given even after the code would give
        another; else :attr:`Brief.groups <blinded.brief.Brief.groups>`."""
        brief = self.brief()
        if not brief.has_archetypes:
            return brief.groups
        menu = self.menu()
        return tuple(briefs.Group(e, archetypes.EDGES[e].title, menu.shares[e], menu.members(e))
                     for e in menu.edges)  # fmt: skip

    @property
    def manifest(self) -> dict:
        return json.loads((self.vault / "manifest.json").read_text())

    @property
    def judge_rules(self) -> int:
        """The judging rules this run is judged under (:data:`blinded.ladder.RULES`): the
        manifest's ``judge_rules``, written by ``blinded init``; 1 for a run whose manifest has
        none, every run before the rules had a version."""
        return check_rules(self.manifest.get("judge_rules", 1))

    def brief(self) -> Brief:
        """The vault's brief, under the manifest's judging rules (:attr:`judge_rules`). The
        workspace copy is the agent's and is never trusted."""
        brief = briefs.load(self.vault / "brief.yaml")
        return dataclasses.replace(brief, judge_rules=self.judge_rules)

    def windows(self) -> dict[str, Window]:
        """The windows fixed at init, warm-up not yet removed."""
        tz = self.brief().data.tz
        return {
            name: Window(name, pd.Timestamp(a).tz_convert(tz), pd.Timestamp(b).tz_convert(tz))
            for name, (a, b) in self.manifest["windows"].items()
        }

    def verify(self) -> None:
        """Check every segment file, and every touch and depth file, against the hash written
        at init, and the manifest's judging rules against the manifest the ledger recorded
        at init."""
        manifest = self.manifest
        if self.ledger_path.is_file():
            ledger = self.ledger()
            row = ledger.run(self.run_id)
            ledger.close()
            if row is not None and row["split_manifest_json"]:
                was = json.loads(row["split_manifest_json"]).get("judge_rules", 1)
                if was != self.judge_rules:
                    raise ValueError(f"vault/manifest.json names judge_rules {self.judge_rules}, "
                                     f"but the run was started under {was}")  # fmt: skip
        for name, seg in manifest["segments"].items():
            if sha256(self.segment_path(name)) != seg["sha256"]:
                raise ValueError(f"vault/{name}.parquet does not match its hash in the manifest")
        for kind in SIDE_FILES:
            for name, seg in (manifest.get(kind) or {}).items():
                if sha256(self.side_path(name, kind)) != seg["sha256"]:
                    raise ValueError(f"vault/{name}.{kind}.parquet does not match its hash in "
                                     "the manifest")  # fmt: skip

    def frame(self, segment: str) -> pl.DataFrame:
        """The raw rows from the start of the data to the end of ``segment``, in file order."""
        names = SEGMENTS[: SEGMENTS.index(segment) + 1]
        return pl.concat([pl.read_parquet(self.segment_path(n)) for n in names])

    def panel(self, segment: str) -> Panel:
        """One contiguous panel from the start of train to the end of ``segment``.

        Validation is scored on train, embargo and validation together, so rolling
        windows are warm at the first validation point; test likewise.
        """
        brief = self.brief()
        theo = None
        if brief.data.theo is not None and (self.manifest.get("theo") or {}):
            names = SEGMENTS[: SEGMENTS.index(segment) + 1]
            theo = tuple(self.side_path(n, "theo") for n in names)
        panel = Panel.from_brief(brief, self.frame(segment), theo=theo)
        spec = brief.data.touch
        if spec is not None:
            from blinded.sim_maker import TouchSource

            names = SEGMENTS[: SEGMENTS.index(segment) + 1]
            paths = tuple(self.side_path(n, "touch") for n in names)
            panel.touch_source = TouchSource(paths, spec.instrument, spec.tick_size)
        depth = brief.data.depth
        if depth is not None and (self.manifest.get("depth") or {}):
            from blinded.sim_maker import DepthSource

            names = SEGMENTS[: SEGMENTS.index(segment) + 1]
            paths = tuple(self.side_path(n, "depth") for n in names)
            panel.depth_source = DepthSource(paths, depth.venue, depth.levels)
        return panel

    def side_path(self, segment: str, kind: str) -> Path:
        """The vault's ``touch`` or ``depth`` file of one segment."""
        return self.vault / f"{segment}.{kind}.parquet"

    def touch(self, segment: str, instrument: str, panel: Panel | None = None,
              window: Window | None = None):  # fmt: skip
        """The fill engine's stream of ``instrument`` from the start of train to the end of
        ``segment`` (:class:`~blinded.sim_maker.Touch`), or only what a pass on ``window``
        reads.

        With ``data.touch`` in the brief, the book rows are the vault's touch files,
        ``{segment}.touch.parquet`` for train up to ``segment``, and the trades the
        panel's; without, both are the panel's own rows, as on a Tardis file that keeps
        every book row and trade. ``panel`` is :meth:`panel` of ``segment`` when not
        given; the judge reads the stream through it.
        """
        from blinded import sim_maker

        brief = self.brief()
        panel = self.panel(segment) if panel is None else panel
        return sim_maker.touch_of(panel, instrument, brief.execution.tick_size, window)


def git_root(path: str | Path) -> Path | None:
    """The work tree containing ``path``, or None. ``path`` need not exist."""
    p = Path(path).expanduser().resolve()
    for parent in (p, *p.parents):
        if (parent / ".git").exists():
            return parent
    return None


def git_ignores(root: Path, path: str | Path) -> bool:
    """Whether the repository at ``root`` ignores ``path``. False if git cannot say."""
    if shutil.which("git") is None:
        return False
    rel = Path(path).expanduser().resolve().relative_to(root)
    done = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", "--", str(rel)], capture_output=True
    )
    return done.returncode == 0


def refuse_inside_git(path: str | Path, what: str, allow_ignored: bool = False) -> None:
    root = git_root(path)
    if root is None or (allow_ignored and git_ignores(root, path)):
        return
    raise InsideGitRepo(
        f"{what} {Path(path).expanduser()} is inside the git repository {root}. "
        "A run holds market data, and anything in a work tree is one `git add` away "
        "from a commit; put it outside every repository"
        + (", or have that repository ignore it." if allow_ignored else ".")
    )


def split_rows(brief: Brief, frame: pl.DataFrame) -> tuple[dict[str, Window], dict]:
    """Windows for the data and the raw rows of each segment.

    Rows stamped before the end of train go to train, rows up to the end of
    validation (the first embargo included) to validation, the rest (the second
    embargo included) to test. File order is kept inside each segment.
    """
    grid = make_grid(brief, load_events(brief, frame))
    tz = brief.data.tz
    first, last = (pd.Timestamp(int(grid[i]), tz="UTC").tz_convert(tz) for i in (0, -1))
    wins = briefs.windows(brief, first, last)
    t = event_times(frame, brief)
    train_end, val_end = wins["train"].end, wins["val"].end
    rows = {
        "train": frame.filter(t < train_end),
        "val": frame.filter((t >= train_end) & (t < val_end)),
        "test": frame.filter(t >= val_end),
    }
    return wins, rows


def init(brief_path: str | Path, run_dir: str | Path, mode: str = "protected") -> RunDir:
    """Build ``run_dir/workspace`` and ``run_dir/vault`` from a brief.

    Refuses a data file that a git work tree would track, a run directory
    inside any work tree, and a run directory that already has files in it.
    In ``naive`` mode the vault is built the same way, so the final pick can
    be scored on the same locked test, but the workspace holds all of the
    data as ``data/all.parquet``.
    """
    from blinded import prompts

    if mode not in ("protected", "naive"):
        raise ValueError(f"mode must be 'protected' or 'naive', got {mode!r}")
    from blinded import __version__
    from blinded.diagnostics import write_data_md

    brief_path = Path(brief_path)
    brief = briefs.load(brief_path)
    if brief.judge_rules != RULES:
        raise ValueError(
            f"judge_rules: {brief.judge_rules} in the brief; a new run is judged "
            f"under the current rules, {RULES}, which blinded init writes"
        )
    refuse_inside_git(brief.data.path, "the data file", allow_ignored=True)
    refuse_inside_git(run_dir, "the run directory")
    run = RunDir(Path(run_dir).expanduser().resolve())
    if run.root.exists() and any(run.root.iterdir()):
        raise FileExistsError(f"{run.root} is not empty")

    source = pl.read_parquet(brief.data.path)
    raw = yaml.safe_load(brief_path.read_text())
    if brief.has_archetypes:
        brief, raw = _with_instruments(brief, raw, source)
    _check_instruments(brief, source)
    wins, rows = split_rows(brief, source)
    if brief.data.cost_model is not None:
        check_cost_model(brief, source, wins["train"].start)
    sides = _side_files(brief, source, wins)
    theo = _theo_files(brief, source, wins) if brief.data.theo is not None else None
    menu = _menu(brief, wins, source) if brief.has_archetypes else None
    for path in (run.vault, run.trials, run.inbox, run.outbox):
        path.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "features", "strategies"):
        (run.workspace / sub).mkdir()
    segments = {}
    for name in SEGMENTS:
        path = run.segment_path(name)
        rows[name].write_parquet(path)
        segments[name] = {"file": path.name, "rows": rows[name].height, "sha256": sha256(path)}
    workspace_data = "data/all.parquet" if mode == "naive" else "data/train.parquet"
    if mode == "naive":
        source.write_parquet(run.workspace / workspace_data)
    else:
        shutil.copyfile(run.segment_path("train"), run.workspace / workspace_data)
    side_manifest, side_paths = {}, {}
    for kind, (spec, parts) in sides.items():
        side_manifest[kind] = {}
        for name in SEGMENTS:
            path = run.side_path(name, kind)
            parts[name].write_parquet(path)
            side_manifest[kind][name] = {"file": path.name, "rows": parts[name].height,
                                         "sha256": sha256(path)}  # fmt: skip
        side_paths[kind] = f"data/{'all' if mode == 'naive' else 'train'}.{kind}.parquet"
        if mode == "naive":
            shutil.copyfile(spec.path, run.workspace / side_paths[kind])
        else:
            shutil.copyfile(run.side_path("train", kind), run.workspace / side_paths[kind])
        del parts
    if theo is not None:
        # the theo's output, cut at the segments' ends as the data is; theo.json holds its
        # parameters, fitted on train, its fit and its innovation report by segment
        frames, meta = theo
        side_manifest["theo"] = {}
        for name in SEGMENTS:
            path = run.side_path(name, "theo")
            frames[name].write_parquet(path)
            side_manifest["theo"][name] = {"file": path.name, "rows": frames[name].height,
                                           "sha256": sha256(path)}  # fmt: skip
        side_paths["theo"] = f"data/{'all' if mode == 'naive' else 'train'}.theo.parquet"
        whole = pl.concat([frames[n] for n in SEGMENTS]) if mode == "naive" else frames["train"]
        whole.write_parquet(run.workspace / side_paths["theo"])
        from blinded import theo as theos

        theos.write_json(meta, run.vault / "theo.json")

    # the workspace's copy names the rules, so blinded eval there judges as the daemon does;
    # the run's own are the manifest's (RunDir.brief)
    raw["judge_rules"] = RULES
    raw.setdefault("splits", {})
    raw["splits"]["embargo"] = label(brief.splits.embargo)
    raw["splits"]["windows"] = {
        name: f"{wins[name].start.isoformat()}/{wins[name].end.isoformat()}" for name in SEGMENTS
    }
    for where, path in ((run.vault, "train.parquet"), (run.workspace, workspace_data)):
        raw["data"]["path"] = path
        for kind in (*sides, *(("theo",) if theo is not None else ())):
            raw["data"][kind] = dict(raw["data"][kind]) | {
                "path": f"train.{kind}.parquet" if where == run.vault else side_paths[kind]
            }
        (where / "brief.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))

    manifest = {
        "run_id": uuid.uuid4().hex[:12],
        "blinded": __version__,
        "mode": mode,
        "judge_rules": RULES,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "brief": {"name": brief.name, "sha256": sha256(brief_path)},
        "source": {
            "file": brief.data.path.name,
            "sha256": sha256(brief.data.path),
            "rows": source.height,
        },
        "embargo": label(brief.splits.embargo),
        "windows": {n: [wins[n].start.isoformat(), wins[n].end.isoformat()] for n in SEGMENTS},
        "segments": segments,
        **side_manifest,
    }
    (run.vault / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    ws_brief = briefs.load(run.workspace / "brief.yaml")
    ws_panel = Panel.from_brief(ws_brief)
    text = write_data_md(ws_brief, ws_panel, run.workspace / "DATA.md")
    text += _splits_md(manifest) + _raw_notes(brief, wins["train"].end, mode)
    if theo is not None:
        from blinded import theo as theos

        text += theos.data_md(theo[1], mode)
    (run.workspace / "DATA.md").write_text(text)
    if menu is not None:
        # the training panel, as the replay reads it: a naive workspace holds the whole file,
        # where the markouts of train's last fills would read past train's end
        _write_menu(run, ws_brief, ws_panel if mode == "protected" else run.panel("train"),
                    wins, menu)  # fmt: skip
    from blinded.driver import Budget

    budget = Budget.from_raw(brief.raw.get("budget"))
    (run.workspace / "RULES.md").write_text(prompts.rules(mode, ws_brief, budget, menu))
    ledger = run.ledger()
    ledger.start_run(
        manifest["run_id"],
        brief_hash=manifest["brief"]["sha256"],
        data_sha256=manifest["source"]["sha256"],
        split_manifest_json=manifest,
        mode=mode,
    )
    run.write_ledger_md(ledger)
    ledger.close()
    from blinded import sandbox

    sandbox.write(run)
    return run


def _theo_files(brief: Brief, source: pl.DataFrame, wins: dict[str, Window]) -> tuple:
    """The theo of a brief with ``data.theo`` (:func:`blinded.theo.build`): its output cut at
    the segments' ends as the data is, by segment, and ``theo.json``'s content."""
    from blinded import theo as theos

    spans = {seg: (wins[seg].start.value, wins[seg].end.value) for seg in SEGMENTS}
    out, meta = theos.build(source, brief, brief.data.theo, wins["train"].end.value, spans)
    at = out["ts"].dt.epoch("ns")
    a, b = wins["train"].end.value, wins["val"].end.value
    frames = {"train": out.filter(at < a), "val": out.filter((at >= a) & (at < b)),
              "test": out.filter(at >= b)}  # fmt: skip
    return frames, meta


def _side_files(brief: Brief, source: pl.DataFrame, wins: dict[str, Window]) -> dict:
    """The touch and depth files a brief declares, checked and cut at the segments' ends:
    ``{kind: (spec, {segment: rows})}``. Raises ``ValueError`` before anything is written.

    A row goes to the segment its stamp falls in, as a data row does
    (:func:`split_rows`). The touch file must pass :func:`check_touch`; the depth
    file must have its levels' columns and stamps in order.
    """
    out = {}
    data = brief.data
    for kind, spec in (("touch", data.touch), ("depth", data.depth)):
        if spec is None:
            continue
        refuse_inside_git(spec.path, f"the {kind} file", allow_ignored=True)
        rows = pl.read_parquet(spec.path)
        if kind == "touch":
            check_touch(brief, source, rows)
        else:
            check_depth(spec, rows)
        at = rows["ts"].cast(pl.Datetime("ns")).dt.epoch("ns")
        a, b = wins["train"].end.value, wins["val"].end.value
        parts = {"train": rows.filter(at < a), "val": rows.filter((at >= a) & (at < b)),
                 "test": rows.filter(at >= b)}  # fmt: skip
        out[kind] = (spec, parts)
    return out


TOUCH_COLUMNS = ("ts", "bid", "ask", "bid_qty", "ask_qty")


def check_touch(brief: Brief, frame: pl.DataFrame, touch: pl.DataFrame) -> dict:
    """A touch file must be the data file's book, every row of it: refuses (``ValueError``) a
    file whose stamps are not in order, whose prices are not whole ticks or not two-sided,
    or whose touch at any stamp of the data file's book rows of its instrument differs from
    those rows' bid, ask and sizes. Returns what was checked.

    The data file's rows are the touch as the panel reads it; on a thinned file they are
    the lattice's rows, so this is the identity of the two streams at every stamp the
    panel can read.
    """
    spec = brief.data.touch
    missing = [c for c in TOUCH_COLUMNS if c not in touch.columns]
    if missing:
        raise ValueError(f"the touch file {spec.path.name} has no {', '.join(missing)}")
    ts = touch["ts"].cast(pl.Datetime("ns")).dt.epoch("ns").to_numpy()
    if len(ts) and (np.diff(ts) < 0).any():
        bad = int(np.flatnonzero(np.diff(ts) < 0)[0]) + 1
        raise ValueError(f"the touch file's stamps go backwards at row {bad:,}")
    ticks = {}
    for side in ("bid", "ask"):
        x = touch[side].cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(x)) or np.any(x != np.rint(x)):
            raise ValueError(f"the touch file's {side} is not a whole number of ticks of "
                             f"{spec.tick_size:g} on every row")  # fmt: skip
        ticks[side] = x.astype(np.int64)
    if not np.all((ticks["ask"] > ticks["bid"]) & (ticks["bid"] > 0)):
        raise ValueError("the touch file has a row whose ask is not above its bid, or a bid "
                         "at or under zero")  # fmt: skip
    mine = frame.filter(pl.col(brief.data.instrument_column).cast(pl.String) == spec.instrument)
    book = load_events(brief, mine).get(("book", spec.instrument))
    if book is None or not len(book):
        raise ValueError(f"the data file has no book rows of {spec.instrument}, the touch "
                         "file's instrument")  # fmt: skip
    at = book.frame["ts"].dt.epoch("ns").to_numpy()
    j = np.searchsorted(ts, at, side="right") - 1
    if (j < 0).any():
        raise ValueError(f"the data file has a book row of {spec.instrument} before the touch "
                         "file's first")  # fmt: skip
    for side in ("bid", "ask", "bid_qty", "ask_qty"):
        want = book.column(side)
        if side in ticks:
            # the data file's price in ticks, which must be whole as the touch file's are
            got, want = ticks[side][j].astype(float), want / spec.tick_size
            same = np.abs(got - want) <= 1e-6
        else:
            got = touch[side].cast(pl.Float64).to_numpy()[j]
            same = got == want
        if not same.all():
            k = int(np.flatnonzero(~same)[0])
            when, a, b = pd.Timestamp(int(at[k]), tz="UTC"), float(got[k]), float(want[k])
            raise ValueError(f"the touch file and the data file disagree on {spec.instrument}'s "
                             f"{side} at {when}: {a:.10g} against {b:.10g}"
                             + (" ticks" if side in ticks else ""))  # fmt: skip
    return {"rows": len(ts), "stamps_checked": len(at)}


def check_depth(spec, depth: pl.DataFrame) -> None:
    """A depth file must hold every level it declares and its stamps in order."""
    need = ["ts"] + [f"{s}{k}_{f}" for s in ("bid", "ask") for k in range(spec.levels)
                     for f in ("px", "qty")]  # fmt: skip
    missing = [c for c in need if c not in depth.columns]
    if missing:
        raise ValueError(f"the depth file {spec.path.name} lacks {', '.join(missing[:4])}"
                         + (" and more" if len(missing) > 4 else ""))  # fmt: skip
    ts = depth["ts"].cast(pl.Datetime("ns")).dt.epoch("ns").to_numpy()
    if len(ts) and (np.diff(ts) < 0).any():
        raise ValueError("the depth file's stamps go backwards")


def _raw_notes(brief: Brief, train_end: pd.Timestamp, mode: str) -> str:
    """``DATA-raw.md`` from beside the data file, which ``blinded demo binance`` writes: the
    checks of the raw files before thinning. A protected run keeps only the days that end
    by the end of train, so nothing in it comes from validation or test."""
    from blinded.demo import notes_until

    path = brief.data.path.parent / "DATA-raw.md"
    if not path.is_file():
        return ""
    text = notes_until(path.read_text(), None if mode == "naive" else train_end)
    # one heading level down, so its sections sit under DATA.md's own
    lines = ["#" + line if line.startswith("#") else line for line in text.splitlines()]
    return "\n" + "\n".join(lines) + "\n"


def _with_instruments(brief: Brief, raw: dict, frame: pl.DataFrame) -> tuple[Brief, dict]:
    """An archetypes block without instruments gets every instrument in the file, in the
    panel's order, so the brief the run keeps says what the run studies."""
    if brief.archetypes.instruments:
        return brief, raw
    raw["archetypes"] = dict(raw.get("archetypes") or {}) | {"instruments": _symbols(brief, frame)}
    base = Path(brief.source).parent if brief.source is not None else None
    return briefs.parse(copy.deepcopy(raw), base=base, source=brief.source), raw


def _symbols(brief: Brief, frame: pl.DataFrame) -> list[str]:
    return sorted(frame[brief.data.instrument_column].cast(pl.String).unique().to_list())


def _menu(brief: Brief, wins: dict[str, Window], frame: pl.DataFrame) -> archetypes.Menu:
    """The menu from the brief and the windows init fixed; refuses what cannot run.

    Called before anything is written, so a brief whose ``only:`` names an
    archetype this file cannot run, or that leaves nothing to run, fails
    with nothing left behind.
    """
    facts = archetypes.facts(brief, windows=wins)
    facts = dataclasses.replace(facts, available=tuple(_symbols(brief, frame)))
    menu = archetypes.menu(brief, facts)
    spec = brief.archetypes
    bad = [menu.line(n) for n in spec.only if menu.line(n) is not None]
    if bad:
        raise ValueError("archetypes.only names what this file cannot run: " + " ".join(bad))
    if not menu.edges:
        raise ValueError(f"no archetype runs on {brief.name}: " + " ".join(
            menu.line(n) for n in menu.refused))  # fmt: skip
    return menu


def _write_menu(run: RunDir, brief: Brief, panel: Panel, wins, menu: archetypes.Menu) -> None:
    """``vault/menu.json`` and ``workspace/ARCHETYPES.md`` from the training panel, after
    checking that it gives the verdicts the brief gave, which the driver's groups use."""
    found = archetypes.menu(brief, archetypes.facts(brief, panel, wins, maker=True))
    differ = [n for n in menu.verdicts if found.verdicts[n] != menu.verdicts[n]]
    if differ:
        raise ValueError(f"the training rows do not give the menu the brief does: {differ}")
    (run.vault / "menu.json").write_text(json.dumps(found.to_dict(), indent=1) + "\n")
    (run.workspace / "ARCHETYPES.md").write_text(archetypes.render(found, brief))


def _check_instruments(brief: Brief, frame: pl.DataFrame) -> None:
    """Every instrument the brief's families or archetypes name must be in the data."""
    have = set(frame[brief.data.instrument_column].cast(pl.String).unique().to_list())
    spec = brief.archetypes
    for key in ("cross_venue", "carry") if spec is not None else ():
        for pair in getattr(spec, key):
            lost = [s for s in pair if s not in have]
            if lost:
                raise ValueError(f"archetypes.{key} pair ({pair[0]}, {pair[1]}) names "
                                 f"{', '.join(lost)}, which the data does not have")  # fmt: skip
    named = {sym for family in brief.families for sym in family.symbols}
    missing = sorted(named - have)
    if missing:
        raise ValueError(
            f"the brief names instruments the data does not have: {', '.join(missing)}; "
            f"it has {', '.join(sorted(have))}"
        )


COST_TOLERANCE = 1e-9
# the rounding of (ask - bid) / (ask + bid) in doubles is about 1e-12 bps whatever the price,
# which is more than 1e-9 of a one-tick half spread on BTC spot (0.000809 bps); this floor is
# a hundred times that rounding and far below any spread a file could be built with
COST_FLOAT_BPS = 1e-10


def check_cost_model(brief: Brief, frame: pl.DataFrame, train_start: pd.Timestamp) -> dict:
    """A bar file's bid and ask must be the half spreads its brief declares, measured on a day
    before train; raises ``ValueError`` otherwise, before anything is written.

    On every book row, :math:`(ask - bid) / (2\\,\\text{mid})` with :math:`\\text{mid} =
    (bid + ask) / 2` must equal the instrument's ``half_spread_bps`` to a relative
    :data:`COST_TOLERANCE`, plus :data:`COST_FLOAT_BPS` for the rounding of doubles.
    Returns the largest relative difference per instrument.
    """
    cost = brief.data.cost_model
    if cost.day is None:
        raise ValueError(f"data.cost_model.source names no day: {cost.source!r}")
    if cost.day + pd.Timedelta("1D") > train_start:
        raise ValueError(f"data.cost_model.source day {cost.day.date()} is not before train, "
                         f"which starts {train_start}: the half spread must be measured on "
                         "data the run does not score")  # fmt: skip
    events = load_events(brief, frame)
    worst = {}
    for (stream, sym), ev in events.items():
        if stream != "book":
            continue
        if sym not in cost.half_spread_bps:
            raise ValueError(f"data.cost_model.half_spread_bps has no value for {sym}")
        bid, ask = ev.column("bid"), ev.column("ask")
        half = (ask - bid) / (bid + ask) * 1e4
        want = cost.half_spread_bps[sym]
        off = np.abs(half - want)
        worst[sym] = float(np.max(off) / want) if len(half) else 0.0
        if not np.all(off <= COST_TOLERANCE * want + COST_FLOAT_BPS):
            raise ValueError(f"the file's bid and ask for {sym} are not its brief's half "
                             f"spread of {want:g} bps: off by up to {worst[sym]:.3g} "
                             "relative")  # fmt: skip
    return worst


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _splits_md(manifest: dict) -> str:
    w = manifest["windows"]
    train = manifest["segments"]["train"]
    if manifest.get("mode") == "naive":
        return "\n".join(
            [
                "",
                "## Windows",
                "",
                f"- train: {w['train'][0]} to {w['train'][1]}",
                f"- val: {w['val'][0]} to {w['val'][1]}",
                f"- test: {w['test'][0]} to {w['test'][1]}",
                "",
                "This workspace holds all of the data; `--window` takes any of these names, "
                "`all` or `START/END`.",
                "",
            ]
        )
    return "\n".join(
        [
            "",
            "## Splits",
            "",
            f"- train: {w['train'][0]} to {w['train'][1]}; this workspace holds these "
            f"{train['rows']:,} rows (sha256 `{train['sha256'][:16]}`)",
            f"- validation: {w['val'][0]} to {w['val'][1]}; in the vault, scored through "
            "the Ladder only",
            f"- test: {w['test'][0]} to {w['test'][1]}; in the vault, scored once at the end",
            f"- embargo of {manifest['embargo']} before validation and before test",
            "",
        ]
    )
