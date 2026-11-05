import json
import sqlite3
from pathlib import Path

import pytest

from blinded import ledger as ledgers
from blinded.ledger import AlreadyScored, Ledger, code_hash

known = pytest.mark.known_answer
edge = pytest.mark.edge_case
REFERENCE = (Path(__file__).parent / "reference_features" / "basis_dev.py").read_text()


@pytest.fixture
def book(tmp_path):
    led = Ledger(tmp_path / "ledger.sqlite")
    led.start_run("r1", mode="protected")
    yield led
    led.close()


@known
def test_comment_and_layout_edits_keep_the_hash():
    base = code_hash(REFERENCE, "basis_dev")
    commented = REFERENCE.replace(
        "    basis = F.basis(", "    # log basis in bps\n    basis = F.basis("
    )
    reflowed = REFERENCE.replace("return basis - F.lag(", "return (basis\n        - F.lag(")
    reflowed = reflowed.replace("3000), 1)", "3000), 1))")
    redocumented = REFERENCE.replace("used for calibration", "kept for the calibration test")
    assert commented != REFERENCE and reflowed != REFERENCE and redocumented != REFERENCE
    assert code_hash(commented, "basis_dev") == base
    assert code_hash(reflowed, "basis_dev") == base
    assert code_hash(redocumented, "basis_dev") == base


@known
def test_real_edits_change_the_hash():
    base = code_hash(REFERENCE, "basis_dev")
    assert code_hash(REFERENCE.replace("3000", "3001"), "basis_dev") != base
    assert code_hash(REFERENCE.replace("* 1e4", "* 1e3"), "basis_dev") != base
    assert code_hash(REFERENCE.replace("basis - F.lag", "basis + F.lag"), "basis_dev") != base
    assert code_hash(REFERENCE, "other_name") != base


@edge
def test_a_syntax_error_raises():
    with pytest.raises(SyntaxError):
        code_hash("def f(:\n    pass")


@known
def test_every_status_is_a_row(book):
    for status in ledgers.STATUSES:
        kind = "probe" if status == "probe" else "feature"
        book.add_trial("r1", kind=kind, name=status, status=status, code_hash=status)
    assert [r["status"] for r in book.trials()] == list(ledgers.STATUSES)


@edge
def test_unknown_status_and_columns_are_refused(book):
    with pytest.raises(sqlite3.IntegrityError):
        book.add_trial("r1", kind="feature", status="great")
    with pytest.raises(KeyError, match="val_score"):
        book.add_trial("r1", kind="feature", status="accepted", val_score=0.1)


@known
def test_n_trials_counts_distinct_strategy_hashes(book):
    for h in ("a", "a", "b"):
        book.add_trial("r1", kind="strategy", status="accepted", code_hash=h)
    book.add_trial("r1", kind="strategy", status="rejected_lookahead", code_hash="c")
    book.add_trial("r1", kind="feature", status="accepted", code_hash="d")
    book.probe("r1", "Bash", "cat vault/val.parquet")
    # a, b and the rejected c: rejected trials were still attempts
    assert book.n_trials("strategy") == 3
    assert book.n_trials("feature") == 1


@known
def test_test_segment_is_scored_once(book):
    t1 = book.add_trial("r1", kind="strategy", status="accepted")
    t2 = book.add_trial("r1", kind="strategy", status="accepted")
    book.record_test("r1", t1, {"net_bps_per_hour": -0.4})
    with pytest.raises(AlreadyScored):
        book.record_test("r1", t1, {"net_bps_per_hour": 9.9})
    with pytest.raises(AlreadyScored):
        book.record_test("r1", t2, {"net_bps_per_hour": 9.9})
    rows = book.db.execute("SELECT * FROM test_scores").fetchall()
    assert len(rows) == 1 and json.loads(rows[0]["metrics_json"]) == {"net_bps_per_hour": -0.4}
    run = book.db.execute("SELECT final_trial_id, test_scored_at FROM runs").fetchone()
    assert run["final_trial_id"] == t1 and run["test_scored_at"] is not None


@known
def test_blocks_round_trip(book):
    t = book.add_trial("r1", kind="strategy", status="accepted")
    book.add_blocks(t, "val", [1.5, -0.5, 2.0])
    assert [b["pnl_bps"] for b in book.blocks(t, "val")] == [1.5, -0.5, 2.0]
    assert book.blocks(t, "train") == []


@known
def test_a_second_connection_can_write_a_probe(book, tmp_path):
    other = Ledger(tmp_path / "ledger.sqlite")
    other.probe("r1", "Read", "vault/val.parquet")
    other.close()
    assert [r["status"] for r in book.trials()] == ["probe"]


@known
def test_ledger_md_shows_no_validation_numbers(book):
    train = {"horizons": {"10s": {"rank_ic": 0.0707, "rank_ic_se_nw": 0.0183}}}
    t = book.add_trial(
        "r1",
        kind="feature",
        name="basis",
        hypothesis="perp leads spot",
        status="accepted",
        train_metrics_json=train,
        val_metrics_json={"rank_ic": 0.0123456789},
        released_json={"released": True, "best": 0.012},
        released_bool=True,
        ladder_best_after=0.012,
    )
    book.add_trial(
        "r1", kind="feature", name="peek", status="rejected_lookahead", reason_short="cut"
    )
    book.add_blocks(t, "val", [0.987654321])
    text = ledgers.render(book, ["10s"])
    assert "0.0707 (0.0183)" in text and "| yes | 0.012 |" in text
    assert "0123456789" not in text and "987654321" not in text
    assert "rejected_lookahead" in text and "cut" in text


@edge
def test_empty_ledger_md(book):
    assert ledgers.render(book, ["10s"]).endswith("No trials yet.\n")


# the two tables as the first ledgers had them, before families
OLD_SCHEMA = """
CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, mode TEXT);
CREATE TABLE trials (
    trial_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, submitted_at TEXT NOT NULL,
    kind TEXT NOT NULL, name TEXT, hypothesis TEXT, code_hash TEXT, code_path TEXT,
    submitted_by TEXT NOT NULL DEFAULT 'agent', iteration INTEGER, status TEXT NOT NULL,
    reason_short TEXT, train_metrics_json TEXT, val_metrics_json TEXT, released_json TEXT,
    ladder_best_before REAL, ladder_best_after REAL, released_bool INTEGER,
    wall_seconds REAL, parent_trial_id INTEGER
);
CREATE TABLE releases (
    run_id TEXT NOT NULL, phase TEXT NOT NULL, trial_id INTEGER NOT NULL,
    best_before REAL, best_after REAL NOT NULL, eta REAL NOT NULL
);
INSERT INTO runs VALUES ('r0', '2026-10-17', 'protected');
INSERT INTO trials (run_id, submitted_at, kind, name, status, released_bool, ladder_best_after)
    VALUES ('r0', '2026-10-17', 'feature', 'basis', 'accepted', 1, 0.1);
INSERT INTO releases VALUES ('r0', 'feature', 1, NULL, 0.1, 0.02);
"""


@edge
def test_an_old_ledger_gains_the_family_columns_when_opened(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.executescript(OLD_SCHEMA)
    db.close()
    led = Ledger(path)
    (row,) = led.trials()
    # the rows already there were all directional
    assert (row["family"], row["legs"]) == ("directional", None)
    assert [r["family"] for r in led.releases("feature")] == ["directional"]
    led.add_release("r0", "feature", 1, 0.1, 0.2, 0.05, family="pairs")
    assert [r["best_after"] for r in led.releases("feature", "pairs")] == [0.2]
    text = ledgers.render(led, ["10s"], families=True)
    assert "| id | kind | family | legs | name | status |" in text
    assert "| 1 | feature | directional |  | basis | accepted |" in text
    led.close()
    Ledger(path).close()  # opening it again finds nothing to add


@known
def test_n_trials_spans_families_unless_asked_for_one(book):
    # three distinct strategy hashes in each family: N = 6 for deflation, 3 per family
    for family in ("directional", "pairs"):
        for h in ("a", "b", "c", "a"):
            book.add_trial("r1", kind="strategy", status="accepted", code_hash=f"{family}-{h}",
                           family=family)  # fmt: skip
    assert book.n_trials("strategy") == 6
    assert book.n_trials("strategy", "pairs") == book.n_trials("strategy", "directional") == 3
    assert len(book.trials("strategy", "pairs")) == 4


@known
def test_ledger_md_shows_family_and_legs_only_for_a_run_with_families(book):
    book.add_trial("r1", kind="strategy", name="pp", status="accepted", family="pairs",
                   legs="ETHUSDT_PERP/BTCUSDT_PERP", hypothesis="spread reverts")  # fmt: skip
    plain = ledgers.render(book, ["10s"])
    assert "family" not in plain and "| 1 | strategy | pp | accepted |" in plain
    text = ledgers.render(book, ["10s"], families=True)
    assert "| 1 | strategy | pairs | ETHUSDT_PERP/BTCUSDT_PERP | pp | accepted |" in text


F2_SCHEMA = """
CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
                   mode TEXT NOT NULL DEFAULT 'protected');
CREATE TABLE trials (
    trial_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, submitted_at TEXT NOT NULL,
    kind TEXT NOT NULL, name TEXT, hypothesis TEXT, code_hash TEXT, code_path TEXT,
    submitted_by TEXT NOT NULL DEFAULT 'agent', iteration INTEGER, status TEXT NOT NULL,
    reason_short TEXT, train_metrics_json TEXT, val_metrics_json TEXT, released_json TEXT,
    ladder_best_before REAL, ladder_best_after REAL, released_bool INTEGER,
    wall_seconds REAL, parent_trial_id INTEGER,
    family TEXT NOT NULL DEFAULT 'directional', legs TEXT
);
CREATE TABLE releases (
    run_id TEXT NOT NULL, phase TEXT NOT NULL, trial_id INTEGER NOT NULL,
    best_before REAL, best_after REAL NOT NULL, eta REAL NOT NULL,
    family TEXT NOT NULL DEFAULT 'directional'
);
INSERT INTO runs VALUES ('r0', '2026-10-17', 'protected');
INSERT INTO trials (run_id, submitted_at, kind, name, status, released_bool, family, legs)
    VALUES ('r0', '2026-10-17', 'feature', 'z', 'accepted', 1, 'pairs', 'A/B');
INSERT INTO releases VALUES ('r0', 'feature', 1, NULL, 0.1, 0.02, 'pairs');
"""


@edge
def test_an_f2_ledger_gains_edge_archetype_and_group_when_opened(tmp_path):
    path = tmp_path / "f2.sqlite"
    db = sqlite3.connect(path)
    db.executescript(F2_SCHEMA)
    db.close()
    led = Ledger(path)
    (row,) = led.trials()
    assert (row["family"], row["edge"], row["archetype"]) == ("pairs", None, None)
    # a release written before groups was keyed by its family, and keeps that key
    (rel,) = led.releases("feature")
    assert (rel["family"], rel["group"]) == ("pairs", "pairs")
    assert len(led.releases("feature", group="pairs")) == 1
    led.add_release("r0", "feature", 1, 0.1, 0.2, 0.05, family="pairs", group="relative_value")
    assert [r["best_after"] for r in led.releases(group="relative_value")] == [0.2]
    led.close()
    Ledger(path).close()  # opening it again finds nothing to add


@known
def test_ledger_md_shows_edge_and_archetype_in_a_run_with_archetypes(book):
    book.add_trial("r1", kind="strategy", name="pp", status="accepted", family="pairs",
                   legs="A/B", edge="relative_value", archetype="pairs_trading",
                   hypothesis="spread reverts")  # fmt: skip
    text = ledgers.render(book, ["10s"], archetypes=True)
    assert "| id | kind | family | edge | archetype | legs | name | status |" in text
    row = "| 1 | strategy | pairs | relative_value | pairs_trading | A/B | pp | accepted |"
    assert row in text


def test_a_horizon_without_a_rank_ic_leaves_its_cell_blank():
    # a feature constant on train has no rank IC; the ledger must still be written
    from blinded.ledger import _ic_cell

    assert _ic_cell(None) == ""
    assert _ic_cell({"rank_ic": None, "rank_ic_se_nw": None}) == ""
    assert _ic_cell({"rank_ic": 0.25, "rank_ic_se_nw": 0.0625}) == "0.2500 (0.0625)"
