"""The trial ledger: one SQLite row per evaluation, and the LEDGER.md the agent reads.

Standard library only, so the PreToolUse hook can record probes without
importing polars.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

STATUSES = (
    "accepted",
    "rejected_lookahead",
    "rejected_static",
    "rejected_nondeterministic",
    "error",
    "timeout",
    "oom",
    "probe",
)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    brief_hash TEXT,
    data_sha256 TEXT,
    split_manifest_json TEXT,
    mode TEXT NOT NULL DEFAULT 'protected' CHECK (mode IN ('protected', 'naive')),
    budget_json TEXT,
    harness TEXT,
    claude_version TEXT,
    model TEXT,
    final_trial_id INTEGER,
    test_metrics_json TEXT,
    test_scored_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    submitted_at TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('feature', 'strategy', 'probe')),
    name TEXT,
    hypothesis TEXT,
    code_hash TEXT,
    code_path TEXT,
    submitted_by TEXT NOT NULL DEFAULT 'agent'
        CHECK (submitted_by IN ('agent', 'user', 'driver', 'hook')),
    iteration INTEGER,
    status TEXT NOT NULL CHECK (status IN ({", ".join(repr(s) for s in STATUSES)})),
    reason_short TEXT,
    train_metrics_json TEXT,
    val_metrics_json TEXT,
    released_json TEXT,
    ladder_best_before REAL,
    ladder_best_after REAL,
    released_bool INTEGER,
    wall_seconds REAL,
    parent_trial_id INTEGER REFERENCES trials (trial_id),
    family TEXT NOT NULL DEFAULT 'directional',
    legs TEXT,
    edge TEXT,
    archetype TEXT
);
CREATE TABLE IF NOT EXISTS blocks (
    trial_id INTEGER NOT NULL REFERENCES trials (trial_id),
    segment TEXT NOT NULL CHECK (segment IN ('train', 'val', 'test')),
    block_idx INTEGER NOT NULL,
    pnl_bps REAL,
    ic REAL,
    PRIMARY KEY (trial_id, segment, block_idx)
);
CREATE TABLE IF NOT EXISTS iterations (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    iteration INTEGER NOT NULL,
    session_id TEXT,
    started_at TEXT,
    ended_at TEXT,
    cost_usd_est REAL,
    turns INTEGER,
    summary_json TEXT,
    transcript_path TEXT,
    PRIMARY KEY (run_id, iteration)
);
CREATE TABLE IF NOT EXISTS releases (
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    phase TEXT NOT NULL CHECK (phase IN ('feature', 'strategy')),
    trial_id INTEGER NOT NULL REFERENCES trials (trial_id),
    best_before REAL,
    best_after REAL NOT NULL,
    eta REAL NOT NULL,
    family TEXT NOT NULL DEFAULT 'directional',
    "group" TEXT NOT NULL DEFAULT 'directional'
);
-- one row per run, ever: the primary key is what makes the test set single-use
CREATE TABLE IF NOT EXISTS test_scores (
    run_id TEXT PRIMARY KEY REFERENCES runs (run_id),
    trial_id INTEGER NOT NULL REFERENCES trials (trial_id),
    scored_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);
"""

# columns added since the first ledgers; opening an older one adds them
ADDED = {
    "trials": {"family": "TEXT NOT NULL DEFAULT 'directional'", "legs": "TEXT",
               "edge": "TEXT", "archetype": "TEXT"},
    "releases": {"family": "TEXT NOT NULL DEFAULT 'directional'",
                 "group": "TEXT NOT NULL DEFAULT 'directional'"},
}  # fmt: skip
# a release written before releases had a group was keyed by its family
_BACKFILL = {("releases", "group"): 'UPDATE releases SET "group" = family'}
_TRIAL_FIELDS = {
    "kind", "name", "hypothesis", "code_hash", "code_path", "submitted_by", "iteration",
    "status", "reason_short", "train_metrics_json", "val_metrics_json", "released_json",
    "ladder_best_before", "ladder_best_after", "released_bool", "wall_seconds",
    "parent_trial_id", "family", "legs", "edge", "archetype",
}  # fmt: skip
_DOCUMENTED = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class AlreadyScored(RuntimeError):
    """The run's test segment has been scored; it is never scored twice."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def code_hash(source: str, name: str | None = None) -> str:
    """SHA-256 of the normalised source, plus the name of the feature or strategy.

    The source is parsed and dumped as an AST without line numbers, so comments,
    blank lines, spacing, line breaks and docstrings do not change the hash;
    any change to what the code does, a number, a name or an operator, does.

    Examples
    --------
    >>> a = code_hash("x = 1  # one")
    >>> a == code_hash("x=1")
    True
    >>> a == code_hash("x = 2")
    False
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, _DOCUMENTED) or not node.body:
            continue
        first = node.body[0]
        value = first.value if isinstance(first, ast.Expr) else None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    dump = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(f"{dump}\n{name or ''}".encode()).hexdigest()


class Ledger:
    """The ledger in ``vault/ledger.sqlite``.

    WAL journaling lets the hook append probe rows while the daemon holds
    the database open.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.db = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        for table, columns in ADDED.items():
            have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns.items():
                if name in have:
                    continue
                try:
                    self.db.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" {decl}')
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc):  # another process got there first
                        raise
                    continue
                if (table, name) in _BACKFILL:
                    self.db.execute(_BACKFILL[(table, name)])

    def close(self) -> None:
        self.db.close()

    def start_run(self, run_id: str, **fields) -> None:
        cols = ["run_id", "started_at", *fields]
        values = [run_id, now(), *(_sql(v) for v in fields.values())]
        marks = ", ".join("?" for _ in cols)
        self.db.execute(f"INSERT INTO runs ({', '.join(cols)}) VALUES ({marks})", values)

    def update_run(self, run_id: str, **fields) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        values = [*(_sql(v) for v in fields.values()), run_id]
        self.db.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", values)

    def run(self, run_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def start_iteration(self, run_id: str, iteration: int, transcript_path: str) -> None:
        self.db.execute(
            "INSERT INTO iterations (run_id, iteration, started_at, transcript_path)"
            " VALUES (?, ?, ?, ?)",
            (run_id, iteration, now(), transcript_path),
        )

    def finish_iteration(self, run_id: str, iteration: int, **fields) -> None:
        """Fill in an iteration row: ``session_id``, ``ended_at``, ``cost_usd_est``, ``turns``,
        ``summary_json``."""
        fields.setdefault("ended_at", now())
        sets = ", ".join(f"{k} = ?" for k in fields)
        values = [*(_sql(v) for v in fields.values()), run_id, iteration]
        self.db.execute(f"UPDATE iterations SET {sets} WHERE run_id = ? AND iteration = ?", values)

    def iterations(self, run_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM iterations WHERE run_id = ? ORDER BY iteration", (run_id,)
        ).fetchall()

    def run_id(self) -> str:
        row = self.db.execute("SELECT run_id FROM runs ORDER BY started_at LIMIT 1").fetchone()
        return row["run_id"]

    def add_trial(self, run_id: str, **fields) -> int:
        """Insert one trial row and return its id. Fields are the ``trials`` columns."""
        unknown = set(fields) - set(_TRIAL_FIELDS)
        if unknown:
            raise KeyError(f"not trial columns: {sorted(unknown)}")
        cols = ["run_id", "submitted_at", *fields]
        values = [run_id, now(), *(_sql(v) for v in fields.values())]
        marks = ", ".join("?" for _ in cols)
        cur = self.db.execute(f"INSERT INTO trials ({', '.join(cols)}) VALUES ({marks})", values)
        return int(cur.lastrowid)

    def update_trial(self, trial_id: int, **fields) -> None:
        unknown = set(fields) - set(_TRIAL_FIELDS)
        if unknown:
            raise KeyError(f"not trial columns: {sorted(unknown)}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        values = [*(_sql(v) for v in fields.values()), trial_id]
        self.db.execute(f"UPDATE trials SET {sets} WHERE trial_id = ?", values)

    def probe(self, run_id: str, tool: str, detail: str) -> int:
        """Record an attempt to reach the vault, denied by the hook."""
        return self.add_trial(
            run_id,
            kind="probe",
            name=tool,
            submitted_by="hook",
            status="probe",
            reason_short=detail[:200],
        )

    def add_blocks(self, trial_id: int, segment: str, pnl_bps, ic=None) -> None:
        ic = [None] * len(pnl_bps) if ic is None else ic
        rows = [
            (trial_id, segment, i, _sql(p), _sql(c))
            for i, (p, c) in enumerate(zip(pnl_bps, ic, strict=True))
        ]
        self.db.executemany("INSERT INTO blocks VALUES (?, ?, ?, ?, ?)", rows)

    def blocks(self, trial_id: int, segment: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM blocks WHERE trial_id = ? AND segment = ? ORDER BY block_idx",
            (trial_id, segment),
        ).fetchall()

    def add_release(
        self,
        run_id: str,
        phase: str,
        trial_id: int,
        before,
        after,
        eta,
        family="directional",
        group: str | None = None,
    ) -> None:
        """Record a release on the Ladder of ``group``: the edge in a run with archetypes,
        else the family, which is the default."""
        self.db.execute(
            "INSERT INTO releases"
            ' (run_id, phase, trial_id, best_before, best_after, eta, family, "group")'
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, phase, trial_id, _sql(before), _sql(after), _sql(eta), family,
             group or family),
        )  # fmt: skip

    def releases(
        self, phase: str | None = None, family: str | None = None, group: str | None = None
    ) -> list[sqlite3.Row]:
        where, args = _where(phase=phase, family=family, group=group)
        return self.db.execute(f"SELECT * FROM releases {where} ORDER BY rowid", args).fetchall()

    def trial(self, trial_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()

    def trials(self, kind: str | None = None, family: str | None = None) -> list[sqlite3.Row]:
        where, args = _where(kind=kind, family=family)
        return self.db.execute(f"SELECT * FROM trials {where} ORDER BY trial_id", args).fetchall()

    def n_trials(
        self, kind: str = "strategy", family: str | None = None, edge: str | None = None
    ) -> int:
        """Distinct code hashes of one kind, accepted or not: the N used for deflation.

        Without ``family`` or ``edge`` the count spans every family and edge,
        as deflation needs; an ``error`` row counts like any other.
        """
        where, args = _where(kind=kind, family=family, edge=edge)
        (n,) = self.db.execute(
            f"SELECT COUNT(DISTINCT code_hash) FROM trials {where} AND code_hash IS NOT NULL",
            args,
        ).fetchone()
        return int(n)

    def record_test(self, run_id: str, trial_id: int, metrics: dict) -> None:
        """Store the one test score of the run; a second call raises :class:`AlreadyScored`."""
        stamp = now()
        try:
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.execute(
                    "INSERT INTO test_scores VALUES (?, ?, ?, ?)",
                    (run_id, trial_id, stamp, json.dumps(metrics)),
                )
                self.db.execute(
                    "UPDATE runs SET final_trial_id = ?, test_metrics_json = ?, test_scored_at = ?"
                    " WHERE run_id = ?",
                    (trial_id, json.dumps(metrics), stamp, run_id),
                )
        except sqlite3.IntegrityError:
            raise AlreadyScored(f"run {run_id} has already been scored on test") from None

    def test_score(self, run_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM test_scores WHERE run_id = ?", (run_id,)).fetchone()


def render(
    ledger: Ledger,
    horizons: list[str],
    mode: str = "protected",
    families: bool = False,
    archetypes: bool = False,
) -> str:
    """LEDGER.md: each trial's train metrics and Ladder output, and nothing else from the vault.

    Validation numbers never appear: a row shows whether the submission was
    released and the released best after it, which is what the agent was told.
    A naive run has no Ladder, so its rows show the window each was scored on.
    With ``families``, each row also shows its family and legs; with
    ``archetypes``, its edge and archetype as well.
    """
    naive = mode == "naive"
    tags = ["family", "edge", "archetype", "legs"] if archetypes else ["family", "legs"]
    head = ["id", "kind", *(tags if families or archetypes else []), "name", "status"]
    head += [f"rankIC {h}" for h in horizons]
    head += ["net SR/sqrt(h)", *(["window"] if naive else ["released", "best"])]
    head += ["hypothesis or reason"]
    intro = (
        "Every evaluation in this run, newest last, scored exactly on the window it "
        "asked for; rankIC is shown with its Newey-West SE."
        if naive
        else "Every evaluation in this run, newest last. Train numbers are exact; "
        "rankIC is shown with its Newey-West SE. `released` and `best` are the "
        "Ladder's output, all there is to know about validation."
    )
    lines = ["# LEDGER", "", intro, ""]
    rows = ledger.trials()
    if not rows:
        return "\n".join([*lines, "No trials yet."]) + "\n"
    lines += ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for row in rows:
        train = json.loads(row["train_metrics_json"] or "{}")
        by_h = train.get("horizons", {})
        cells = [str(row["trial_id"]), row["kind"]]
        if families or archetypes:
            cells.append(_text(row["family"] if row["kind"] != "probe" else ""))
            cells += [_text(row["edge"]), _text(row["archetype"])] if archetypes else []
            cells.append(_text(row["legs"]))
        cells += [_text(row["name"]), row["status"]]
        for h in horizons:
            m = by_h.get(h)
            cells.append(_ic_cell(m))
        sr = train.get("sharpe_per_sqrt_hour")
        cells.append(f"{sr:.3f}" if sr is not None else "")
        if naive:
            cells.append(_text(train.get("window")))
        else:
            released = row["released_bool"]
            cells.append("" if released is None else ("yes" if released else "no"))
            best = row["ladder_best_after"]
            cells.append("" if best is None else f"{best:g}")
        note = row["hypothesis"] if row["status"] == "accepted" else row["reason_short"]
        cells.append(_text(note))
        lines.append("| " + " | ".join(cells) + " |")
    lines += _maker_rows(rows, naive)
    lines += _arb_rows(rows, naive)
    return "\n".join(lines) + "\n"


def _ic_cell(m: dict | None) -> str:
    """A horizon's rank IC with its Newey-West SE, blank when it has none: a feature that
    is constant on train has no rank IC, and must not stop the ledger being written."""
    if not m or m.get("rank_ic") is None:
        return ""
    se = m.get("rank_ic_se_nw")
    return f"{m['rank_ic']:.4f}" + ("" if se is None else f" ({se:.4f})")


# a market-making strategy's train rows in LEDGER.md: the scored row, the four fill models,
# latency 0 and the fee rows
MAKER_ROWS = ("brief", "RA", "PR", "T", "F", "latency 0", "fees 0", "rebate 0.5", "fee 1",
              "fee 2 (Binance VIP0)")  # fmt: skip


def _maker_rows(rows, naive: bool = False) -> list[str]:
    """LEDGER.md's market-making table: each maker trial's net bps/hour on train (in a naive
    run, on the window it asked for) under each fill model and at each fee, and the model the
    judge's rule took there; nothing of validation. Empty when no trial is a maker's."""
    makers = []
    for row in rows:
        train = json.loads(row["train_metrics_json"] or "{}")
        if train.get("fill_model") and "RA" in (train.get("rows") or {}) and (
                "hedge 1 s" not in train["rows"]):  # fmt: skip
            makers.append((row, train))
    if not makers:
        return []
    head = ["id", "name", "scored", *MAKER_ROWS]
    where = "on the window each asked for" if naive else "on train"
    out = ["", f"## Market-making strategies {where}", "",
           f"Net bps/hour {where} under each fill model and at each maker fee. `scored` is "
           "the worse of the two queue models there, RA or PR, by the judge's rule"
           + ("." if naive else "; the validation model is never shown."), "",
           "| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]  # fmt: skip
    for row, train in makers:
        cells = [str(row["trial_id"]), _text(row["name"]), _text(train["fill_model"])]
        for name in MAKER_ROWS:
            v = (train["rows"].get(name) or {}).get("net_bps_per_hour")
            cells.append("" if v is None else f"{v:.1f}")
        out.append("| " + " | ".join(cells) + " |")
    return out


ARB_ROWS = ("RA", "PR", "T", "F", "RA uncapped", "hedge 1 s", "hedge fee 10", "hedge fee 7.5",
            "hedge fee 5", "hedge fee 0")  # fmt: skip


def _arb_rows(rows, naive: bool = False) -> list[str]:
    """LEDGER.md's arbitrage table: each arbitrage trial's net bps/hour on train under each fill
    model, uncapped, with a 1 s hedge and at each hedge fee, and the model the judge's rule
    took there. Empty when no trial is an arbitrage's."""
    arbs = []
    for row in rows:
        train = json.loads(row["train_metrics_json"] or "{}")
        if train.get("fill_model") and "hedge 1 s" in (train.get("rows") or {}):
            arbs.append((row, train))
    if not arbs:
        return []
    head = ["id", "name", "scored", *ARB_ROWS]
    where = "on the window each asked for" if naive else "on train"
    out = ["", f"## Arbitrage strategies {where}", "",
           f"Net bps/hour {where}, both legs, under each fill model and at each hedge fee. "
           "`scored` is the worse of the two queue models there by the judge's rule"
           + ("." if naive else "; the validation model is never shown."), "",
           "| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]  # fmt: skip
    for row, train in arbs:
        cells = [str(row["trial_id"]), _text(row["name"]), _text(train["fill_model"])]
        for name in ARB_ROWS:
            v = (train["rows"].get(name) or {}).get("net_bps_per_hour")
            cells.append("" if v is None else f"{v:.1f}")
        out.append("| " + " | ".join(cells) + " |")
    return out


def _where(**filters) -> tuple[str, tuple]:
    given = {k: v for k, v in filters.items() if v is not None}
    clause = " AND ".join(f'"{k}" = ?' for k in given) or "1 = 1"
    return f"WHERE {clause}", tuple(given.values())


def _text(value) -> str:
    text = "" if value is None else str(value)
    text = text.replace("|", "/").replace("\n", " ")
    return text if len(text) <= 120 else text[:117] + "..."


def _sql(value):
    if isinstance(value, dict | list):
        return json.dumps(value)
    if isinstance(value, bool):
        return int(value)
    if hasattr(value, "item"):
        return value.item()
    return value
