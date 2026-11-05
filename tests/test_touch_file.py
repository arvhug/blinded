"""A declared touch stream through a run directory: the vault cuts it at the segments' ends
beside the data file, the workspace gets train's, ``blinded init`` refuses one that is not
the data file's book, and the judge's engine reads it in place of the panel's own rows.

The touch file here is every book row of the spot in the two-hour synthetic file, so a pass
of the engine through the vault's touch files must equal, fill for fill, a pass on the
panel's own rows; read for a window only, it must equal a pass on the whole stream.
"""

import json
import shutil

import numpy as np
import polars as pl
import pytest
import yaml

from blinded import sim_maker, vault
from blinded.brief import load
from blinded.evaluate import scoring_window
from blinded.panel import Panel

known = pytest.mark.known_answer
edge = pytest.mark.edge_case
INST = "XRPUSDT"


def touch_rows(data, tick) -> pl.DataFrame:
    """The spot's book rows as a touch file: stamps, prices in ticks, sizes."""
    book = pl.read_parquet(data).filter(~pl.col("is_trade") & (pl.col("symbol") == INST))
    return book.select(
        pl.col("ts").cast(pl.Datetime("ns")),
        (pl.col("bid0_px") / tick).round().cast(pl.Int64).alias("bid"),
        (pl.col("ask0_px") / tick).round().cast(pl.Int64).alias("ask"),
        pl.col("bid0_qty").alias("bid_qty"),
        pl.col("ask0_qty").alias("ask_qty"),
    )


def with_touch(long_brief, root, rows=None, **extra) -> tuple:
    raw = yaml.safe_load(long_brief.read_text())
    data = raw["data"]["path"]
    tick = sim_maker.infer_tick(*(pl.read_parquet(data)[c].drop_nulls().to_numpy()
                                  for c in ("bid0_px", "ask0_px")))  # fmt: skip
    rows = touch_rows(data, tick) if rows is None else rows
    rows.write_parquet(root / "x.touch.parquet")
    raw["data"]["touch"] = {"path": str(root / "x.touch.parquet"), "tick_size": tick,
                            "instrument": INST, **extra}  # fmt: skip
    raw["name"] = "synthetic-2h-touch"
    path = root / "brief.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, tick, rows


@pytest.fixture(scope="module")
def touch_run(long_brief, tmp_path_factory):
    root = tmp_path_factory.mktemp("touchfile")
    brief, tick, rows = with_touch(long_brief, root)
    return vault.init(brief, root / "run"), tick, rows


def join(n):
    one, zero = np.ones(n), np.zeros(n)
    return sim_maker.Quotes({"bid": zero, "ask": zero, "size": one, "skew": 0.5 * one,
                             "max_inventory": 3 * one})  # fmt: skip


@known
def test_the_vault_cuts_the_touch_file_where_it_cuts_the_data(touch_run):
    run, tick, rows = touch_run
    manifest = run.manifest
    wins = run.windows()
    got = {s: pl.read_parquet(run.side_path(s, "touch")) for s in vault.SEGMENTS}
    # every row once, in file order, each in the segment its stamp falls in
    assert pl.concat(list(got.values())).equals(rows)
    at = {s: f["ts"].dt.epoch("ns").to_numpy() for s, f in got.items()}
    assert at["train"].max() < wins["train"].end.value <= at["val"].min()
    assert at["val"].max() < wins["val"].end.value <= at["test"].min()
    assert {s: manifest["touch"][s]["rows"] for s in got} == {s: f.height for s, f in got.items()}
    run.verify()
    # the workspace holds train's, and its brief points there; the vault's brief at train's
    assert pl.read_parquet(run.workspace / "data/train.touch.parquet").equals(got["train"])
    ws = load(run.workspace / "brief.yaml")
    assert ws.data.touch.path == run.workspace / "data" / "train.touch.parquet"
    assert run.brief().data.touch.path == run.vault / "train.touch.parquet"
    assert ws.data.touch.instrument == INST and ws.data.touch.tick_size == tick


@edge
def test_a_changed_touch_segment_fails_the_vaults_hash(touch_run, tmp_path):
    run, _, _ = touch_run
    copy = vault.RunDir(tmp_path / "run")
    shutil.copytree(run.root, copy.root)
    part = pl.read_parquet(copy.side_path("val", "touch"))
    part.with_columns(pl.col("bid_qty") + 1).write_parquet(copy.side_path("val", "touch"))
    with pytest.raises(ValueError, match="val.touch.parquet does not match its hash"):
        copy.verify()


@known
def test_the_engine_reads_the_vaults_touch_and_it_is_the_panels_own(touch_run):
    run, tick, _ = touch_run
    brief = run.brief()
    panel = run.panel("val")
    assert panel.touch_source.paths == (run.side_path("train", "touch"),
                                        run.side_path("val", "touch"))  # fmt: skip
    win = scoring_window(brief, panel, "val")
    q = join(len(panel))
    via_file = sim_maker.models(q, panel, win, brief, INST)
    own = sim_maker.touch_of(panel.cut(panel.time(-1)), INST, tick)  # cut keeps the source
    assert panel.cut(panel.time(-1)).touch_source == panel.touch_source
    bare = Panel.from_brief(brief, run.frame("val"))
    assert bare.touch_source is None
    via_rows = sim_maker.models(q, bare, win, brief, INST)
    for m in sim_maker.MODELS:
        a, b = via_file[m], via_rows[m]
        assert np.array_equal(a.book.fill_ns, b.book.fill_ns), m
        assert np.array_equal(a.book.fill_qty, b.book.fill_qty), m
        assert np.allclose(a.net, b.net, rtol=0, atol=1e-12), m
    assert via_file["RA"].metrics()["fills"] > 20
    # the vault's whole stream is the panel's own book, row for row
    assert np.array_equal(own.book_ns, sim_maker.touch_of(bare, INST, tick).book_ns)
    # RunDir.touch for a window reads only what a pass on it reads, and gives the same pass
    part = run.touch("val", INST, panel, win)
    assert part.book_ns[0] <= win.start.value < part.book_ns[1]
    assert len(part.book_ns) < len(own.book_ns)
    (ra,) = sim_maker.models(q, panel, win, brief, INST, part, ("RA",)).values()
    assert np.array_equal(ra.book.fill_ns, via_file["RA"].book.fill_ns)
    assert np.allclose(ra.net, via_file["RA"].net, rtol=0, atol=1e-12)


@known
def test_the_workspace_panel_reads_trains_touch(touch_run):
    run, tick, _ = touch_run
    ws = Panel.from_brief(run.workspace / "brief.yaml")
    assert ws.touch_source.paths == (run.workspace / "data" / "train.touch.parquet",)
    t = sim_maker.touch_of(ws, INST)
    train = pl.read_parquet(run.side_path("train", "touch"))
    assert np.array_equal(t.book_ns, train["ts"].dt.epoch("ns").to_numpy())
    assert t.tick == tick


@known
def test_a_naive_workspace_gets_the_whole_touch_file(long_brief, tmp_path):
    brief, _, rows = with_touch(long_brief, tmp_path)
    run = vault.init(brief, tmp_path / "naive", mode="naive")
    assert pl.read_parquet(run.workspace / "data/all.touch.parquet").equals(rows)
    assert json.loads((run.vault / "manifest.json").read_text())["touch"]["test"]["rows"] > 0


@edge
@pytest.mark.parametrize("bad, message", [
    ("backwards", "stamps go backwards at row"),
    ("half tick", "is not a whole number of ticks"),
    ("crossed", "ask is not above its bid"),
    ("size", "disagree on XRPUSDT's bid_qty"),
    ("missing row", "disagree on XRPUSDT's"),
    ("no column", "has no ask_qty"),
])  # fmt: skip
def test_init_refuses_a_touch_file_that_is_not_the_books(long_brief, tmp_path, bad, message):
    raw = yaml.safe_load(long_brief.read_text())
    tick = 1e-4
    rows = touch_rows(raw["data"]["path"], tick)
    k = rows.height // 2
    if bad == "backwards":
        rows = pl.concat([rows.slice(0, k), rows.slice(k + 1, 1), rows.slice(k, 1),
                          rows.slice(k + 2)])  # fmt: skip
    elif bad == "half tick":
        rows = rows.with_columns(pl.col("bid").cast(pl.Float64) + 0.5 * (pl.int_range(
            pl.len()) == k))  # fmt: skip
    elif bad == "crossed":
        rows = rows.with_columns(pl.when(pl.int_range(pl.len()) == k).then(pl.col("bid"))
                                 .otherwise(pl.col("ask")).alias("ask"))  # fmt: skip
    elif bad == "size":
        rows = rows.with_columns(pl.when(pl.int_range(pl.len()) == k).then(-1.0)
                                 .otherwise(pl.col("bid_qty")).alias("bid_qty"))  # fmt: skip
    elif bad == "missing row":
        rows = pl.concat([rows.slice(0, k), rows.slice(k + 1)])
    else:
        rows = rows.drop("ask_qty")
    brief, _, _ = with_touch(long_brief, tmp_path, rows)
    with pytest.raises(ValueError, match=message):
        vault.init(brief, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@known
def test_the_depth_file_is_cut_and_checked_like_the_touch_file(long_brief, tmp_path):
    raw = yaml.safe_load(long_brief.read_text())
    frame = pl.read_parquet(raw["data"]["path"]).filter(
        ~pl.col("is_trade") & (pl.col("symbol") == "XRPUSDT_PERP")
    )
    levels = {f"{s}{k}_{f}": pl.col(f"{s}0_{f}") + (k if f == "qty" else 0)
              for s in ("bid", "ask") for k in range(2) for f in ("px", "qty")}  # fmt: skip
    depth = frame.select(pl.col("ts").cast(pl.Datetime("ns")), **levels)
    depth.write_parquet(tmp_path / "x.depth.parquet")
    raw["data"]["depth"] = {"path": str(tmp_path / "x.depth.parquet"), "venue": "XRPUSDT_PERP",
                            "levels": 2, "lag_ms": 10}  # fmt: skip
    (tmp_path / "brief.yaml").write_text(yaml.safe_dump(raw))
    run = vault.init(tmp_path / "brief.yaml", tmp_path / "run")
    parts = [pl.read_parquet(run.side_path(s, "depth")) for s in vault.SEGMENTS]
    assert pl.concat(parts).equals(depth) and all(p.height for p in parts)
    assert load(run.workspace / "brief.yaml").data.depth.path == (
        run.workspace / "data" / "train.depth.parquet"
    )
    run.verify()
    # DATA.md reads the workspace's depth against the venue's book: the depth here is that
    # book row for row, so its level 0 is the touch at every snapshot
    text = (run.workspace / "DATA.md").read_text()
    assert "## The depth file" in text
    assert "the touch as-of each snapshot row shows its level 0 prices on 100.0%" in text
    raw["data"]["depth"]["levels"] = 3
    (tmp_path / "brief.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="lacks bid2_px"):
        vault.init(tmp_path / "brief.yaml", tmp_path / "run2")


@edge
def test_market_making_on_a_thinned_file_quotes_the_touch_files_instrument(long_brief, tmp_path):
    from blinded import contract
    from blinded.brief import parse

    brief, _, _ = with_touch(long_brief, tmp_path)
    raw = yaml.safe_load(brief.read_text())
    raw["data"]["thinned"] = {"step": raw["grid"]["step"], "offsets_ms": [0, 50]}
    parsed = parse(raw)
    ns: dict = {}
    exec(
        "import blinded as bl\n"
        "mm = bl.market_making(instrument='XRPUSDT_PERP')\n"
        "ok = bl.market_making(instrument='XRPUSDT')\n"
        "@mm.strategy()\n"
        "def perp(panel, features):\n"
        "    return bl.quote()\n"
        "@ok.strategy()\n"
        "def spot(panel, features):\n"
        "    return bl.quote()\n",
        ns,
    )
    with pytest.raises(contract.ContractError, match="quotes XRPUSDT, the instrument"):
        contract.bind(ns["perp"], parsed)
    assert contract.bind(ns["spot"], parsed).template == "maker"


@known
def test_data_md_measures_the_touch_stream_on_train(touch_run):
    from blinded import bookstats

    run, tick, _ = touch_run
    text = (run.workspace / "DATA.md").read_text()
    assert "## The touch stream" in text
    train = pl.read_parquet(run.side_path("train", "touch"))
    ws = Panel.from_brief(run.workspace / "brief.yaml")
    t = sim_maker.touch_of(ws, INST)
    facts = bookstats.touch_facts(t.book_ns, t.bid, t.ask, t.bid_qty, t.ask_qty, tick,
                                  int(ws.grid_ns[-1]) + ws.step_ns, t.trade_ns, t.side, t.price,
                                  t.qty)  # fmt: skip
    assert facts["rows"] == train.height
    assert f"| book rows (a second); distinct stamps | {train.height:,} (" in text
    line = next(x for x in text.splitlines() if x.startswith("| falls of the best bid"))
    assert line.split("| ")[2].startswith(f"{facts['falls']:,}; ")
