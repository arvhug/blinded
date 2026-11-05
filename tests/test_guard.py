"""Judged code cannot write outside its own directory, read the vault, or reach out.

The red-team audit of 20 Oct 2026 wrote 42,387 panel rows, validation included, into
the workspace with ``panel.to_pandas(...).to_json(path)``, and on the test pass the
whole file with the locked test, because the static scan blocked only ``to_parquet``,
``to_csv``, ``to_pickle`` and ``write_*``. Two fences now: the scan allows only the
``to_*`` calls that never take a path (:data:`blinded.contract.SAFE_CALLS`), and the
worker that runs the code installs an audit hook first (:func:`blinded.isolate.guard`).
The hook is tested here event by event, on files the scan never sees, so each refusal
is the hook's own.
"""

import json
import os
import shutil
import textwrap
from pathlib import Path

import pytest

from blinded import contract, isolate
from blinded.daemon import Daemon, Limits, score_test
from blinded.vault import RunDir

known = pytest.mark.known_answer
edge = pytest.mark.edge_case

HEAD = """
import blinded as bl
import polars as pl

@bl.feature("a probe")
def probe(panel):
{body}
    return pl.col("XRPUSDT.mid").diff()
"""


@pytest.fixture
def outside(tmp_path):
    """A directory outside every worker's own, where nothing may appear."""
    d = tmp_path / "outside"
    d.mkdir()
    return d


def run_probe(tmp_path, panel, body: str, deny=(), allow=()) -> tuple[str, str | None]:
    """Evaluate a feature whose body is ``body`` in a guarded worker: ("ok", None) or
    ("failed", reason)."""
    code = tmp_path / "probe.py"
    code.write_text(HEAD.format(body=textwrap.indent(textwrap.dedent(body), "    ")))
    spec = isolate.Spec(str(code), "probe", "feature")
    allow = list(allow) or [str(tmp_path)]
    with isolate.Pool(Limits(evaluation_seconds=60), deny=list(deny), allow=allow) as pool:
        try:
            isolate.Isolated(pool, spec)(panel)
        except isolate.Failed as exc:
            return "failed", exc.reason
    return "ok", None


# each probe, the words its refusal names, and what it would leave outside if let through
PROBES = {
    "open for writing": ("open('{out}/a.txt', 'w').write('x')", "write"),
    "os.open for writing": ("import os\nos.close(os.open('{out}/a', os.O_WRONLY | os.O_CREAT))",
                            "write"),
    "Path.write_text": ("from pathlib import Path\nPath('{out}/a.txt').write_text('x')", "write"),
    "np.save": ("import numpy as np\nnp.save('{out}/a.npy', np.zeros(3))", "write"),
    "pickle.dump": ("import pickle\npickle.dump([1], open('{out}/a.pkl', 'wb'))", "write"),
    "to_json (the audit's)": ("panel.to_pandas('XRPUSDT.mid').reset_index().to_json('{out}/a')",
                              "write"),
    "to_html (the audit's)": ("panel.to_pandas('XRPUSDT.mid').to_html('{out}/a.html')", "write"),
    "open_memmap": ("import numpy as np\nnp.lib.format.open_memmap('{out}/a', mode='w+', "
                    "shape=(3,))", "write"),
    "os.mkdir": ("import os\nos.mkdir('{out}/d')", "files"),
    "os.rename": ("import os\nos.rename('{out}/../x', '{out}/y')", "files"),
    "os.remove": ("import os\nos.remove('{out}/z')", "files"),
    "shutil.copyfile": ("import shutil\nshutil.copyfile(__file__, '{out}/copy.py')", "files"),
    "os.symlink": ("import os\nos.symlink('{out}', 'link')", "link"),
    "os.chdir": ("import os\nos.chdir('{out}')", "chdir"),
    # the modules a library already holds, reached without an import
    "socket connect": ("import email.utils as u\nu.socket.create_connection(('127.0.0.1', 9), "
                       "timeout=1)", "socket"),
    "subprocess": ("import pandas._config.localization as m\nm.subprocess.run(['touch', "
                   "'{out}/p'])", "process"),
    "os.system": ("import os\nos.system('touch {out}/s')", "process"),
    "os.kill": ("import os\nos.kill(os.getppid(), 0)", "process"),
    "ctypes": ("import numpy._core._internal as m\nm.ctypes.CDLL(None)", "ctypes"),
    "import socket": ("import socket", "import"),
    "import subprocess": ("import subprocess", "import"),
    "import ctypes": ("import ctypes", "import"),
    "sqlite3": ("import sqlite3\nsqlite3.connect('{out}/db')", "database"),
    "caught and ignored": ("try:\n    open('{out}/a.txt', 'w')\nexcept OSError:\n    pass",
                           "write"),
}  # fmt: skip


@known
@pytest.mark.parametrize("name", sorted(PROBES))
def test_the_guard_refuses(name, tmp_path, outside, synthetic_panel):
    body, word = PROBES[name]
    status, reason = run_probe(tmp_path, synthetic_panel, body.format(out=outside))
    assert status == "failed"
    assert reason.startswith("the judge refused it: judged code " + isolate.REFUSALS[word])
    assert list(outside.iterdir()) == []


@known
def test_the_guard_lets_code_write_in_its_own_directory(tmp_path, synthetic_panel):
    # the worker's temporary directory is its working directory and TMPDIR
    body = "import tempfile\nopen('mine.txt', 'w').write('x')\ntempfile.mkstemp()"
    assert run_probe(tmp_path, synthetic_panel, body) == ("ok", None)


@known
def test_the_guard_refuses_reads_of_the_vault_but_not_the_trial(tmp_path, synthetic_panel):
    vault = tmp_path / "vault"
    (vault / "trials").mkdir(parents=True)
    (vault / "ledger.sqlite").write_text("validation scores")
    body = f"open('{vault}/ledger.sqlite').read()"
    status, reason = run_probe(tmp_path, synthetic_panel, body, deny=[str(vault)])
    assert status == "failed" and isolate.REFUSALS["read"] in reason
    # the trial's own file, in the vault's trials directory, still loads
    code = vault / "trials" / "probe.py"
    code.write_text(HEAD.format(body="    pass"))
    spec = isolate.Spec(str(code), "probe", "feature")
    with isolate.Pool(Limits(), deny=[str(vault)], allow=[str(vault / "trials")]) as pool:
        assert len(isolate.Isolated(pool, spec)(synthetic_panel)) == len(synthetic_panel)


@known
def test_the_refusal_names_the_line(tmp_path, outside, synthetic_panel):
    status, reason = run_probe(tmp_path, synthetic_panel, f"x = 1\nopen('{outside}/a', 'w')")
    # HEAD puts the body's first line at line 7 of the file
    assert reason.endswith("at line 8")


@known
def test_every_audit_event_named_is_a_real_312_event():
    # the names the hook matches, each raised by the operation the probes run (above); a name
    # Python does not raise would match nothing and refuse nothing
    import sys

    assert sys.version_info[:2] >= (3, 12)
    named = set(isolate._FS_EVENTS) | isolate._PROCESS_EVENTS
    seen = set(_events_raised())
    for name in ("os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.chmod",
                 "os.utime", "shutil.copyfile", "shutil.copytree", "shutil.move",
                 "shutil.rmtree", "shutil.make_archive", "subprocess.Popen", "os.system",
                 "os.posix_spawn", "os.fork", "os.kill"):  # fmt: skip
        assert name in named and name in seen, name
    for prefix in ("socket.", "ctypes.", "sqlite3."):
        assert any(e.startswith(prefix) for e in seen), prefix
    assert {"open", "os.symlink", "os.link", "os.chdir", "import", "os.listdir",
            "os.scandir"} <= seen  # fmt: skip


def _events_raised() -> list[str]:
    """The audit events a fresh interpreter raises for one of each operation."""
    import subprocess
    import sys

    script = textwrap.dedent("""
        import ctypes, os, shutil, socket, sqlite3, subprocess, sys, tempfile
        seen = []
        d = tempfile.mkdtemp()
        sys.addaudithook(lambda e, a: seen.append(e))
        p = d + "/f"
        open(p, "w").close()
        os.truncate(p, 0); os.chmod(p, 0o644); os.utime(p)
        os.mkdir(d + "/s"); os.rmdir(d + "/s"); os.rename(p, p + "r"); os.rename(p + "r", p)
        os.symlink(p, p + "l"); os.link(p, p + "h"); os.chdir(d)
        shutil.copyfile(p, p + "c"); os.mkdir(d + "/t"); shutil.copytree(d + "/t", d + "/u")
        shutil.move(p + "c", p + "m"); shutil.rmtree(d + "/u")
        shutil.make_archive(d + "/z", "zip", d + "/t"); os.remove(p + "m")
        socket.socket().close()
        subprocess.run(["true"]); os.system("true")
        os.waitpid(os.posix_spawn("/usr/bin/true", ["true"], {}), 0)
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0); os.kill(os.getpid(), 0)
        ctypes.CDLL(None); sqlite3.connect(":memory:").close()
        os.listdir(d); list(os.scandir(d))
        import json
        print("\\n".join(sorted(set(seen))))
    """)
    out = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True,
                         check=True)  # fmt: skip
    return out.stdout.split()


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


@known
def test_the_audits_exfil_is_refused_by_the_scan_and_writes_nothing(run):
    dump = run.workspace / "features" / "val_dump.json"
    source = f"""
    import blinded as bl
    import polars as pl

    @bl.feature("looks like a plain mid feature")
    def sneaky(panel):
        panel.to_pandas("XRPUSDT.mid").reset_index().to_json("{dump}")
        return pl.col("XRPUSDT.mid").diff()
    """
    trial, _ = judge(run, source)
    assert trial["status"] == "rejected_static"
    assert ".to_json()" in trial["reason"]
    assert not dump.exists()


@known
def test_a_write_the_scan_misses_is_an_error_row_from_the_hook(run, monkeypatch):
    out = run.workspace / "features"
    before = sorted(os.listdir(out))
    source = f"""
    import blinded as bl
    import numpy as np
    import polars as pl

    @bl.feature("a memory-mapped dump")
    def sneaky(panel):
        x = panel.frame["XRPUSDT.mid"].to_numpy()
        np.lib.format.open_memmap("{out}/dump.npy", mode="w+", shape=x.shape)[:] = x
        return pl.col("XRPUSDT.mid").diff()
    """
    # the scan names open_memmap now; were it to miss a call like it, the hook is next
    assert contract.scan(textwrap.dedent(source), protected=True).errors
    monkeypatch.setattr(contract, "scan", lambda *a, **k: contract.Scan())
    trial, daemon = judge(run, source)
    assert trial["status"] == "error"
    assert trial["reason"].startswith("the judge refused it: judged code opened a file for "
                                      "writing")  # fmt: skip
    assert sorted(os.listdir(out)) == before
    # never accepted, so never scored on test either
    with pytest.raises(ValueError, match="not an accepted trial"):
        score_test(run, trial["trial_id"])


@known
def test_every_published_kind_of_call_still_scans_clean():
    # the to_* calls the 628 published trial files make (survey of 20 Oct 2026)
    for call in ("to_list", "to_series", "to_numpy", "to_pandas", "to_timedelta"):
        assert call in contract.SAFE_CALLS
        assert not contract.scan(f"x = y.{call}()", protected=True).errors
    for call in ("to_json", "to_html", "to_xml", "to_markdown", "to_latex", "to_string",
                 "to_feather", "to_hdf", "to_excel", "to_stata", "to_parquet", "to_csv",
                 "to_pickle", "to_sql", "to_clipboard", "write_csv", "write", "writelines",
                 "save", "savefig", "dump", "dumps", "to_records_file"):  # fmt: skip
        errors = contract.scan(f"y.{call}('p')", protected=True).errors
        assert any(f".{call}" in e for e in errors), call


@known
def test_module_attributes_that_reach_the_os_are_refused_by_the_scan():
    for name in ("os", "sys", "subprocess", "socket", "ctypes", "builtins", "shutil", "io",
                 "pathlib", "importlib", "as_strided"):  # fmt: skip
        errors = contract.scan(f"import pandas as pd\nx = pd.io.common.{name}", protected=True)
        assert errors.errors, name


SRT = os.environ.get("BLINDED_SRT") or shutil.which("srt")


@pytest.mark.os_sandbox
@pytest.mark.skipif(SRT is None, reason="srt is not installed")
def test_under_srt_a_native_write_is_refused_too(tmp_path, outside, synthetic_panel):
    # polars writes from Rust, which no audit hook sees; the OS sandbox does
    assert isolate.sandbox_runtime() is not None
    body = f"panel.frame.write_csv('{outside}/rows.csv')"
    status, reason = run_probe(tmp_path, synthetic_panel, body)
    assert status == "failed"
    assert list(outside.iterdir()) == []
    assert Path(isolate.sandbox_runtime()).exists()


@known
def test_the_guard_refuses_reads_and_listings_outside_its_allowlist(tmp_path, outside,
                                                                   synthetic_panel):  # fmt: skip
    # a present file and a missing one alike: the refusal comes before the OS is asked
    (outside / "marker.txt").write_text("x")
    for body in (f"open('{outside}/marker.txt').read()", f"open('{outside}/missing.txt')",
                 f"import os\nos.listdir('{outside}')"):  # fmt: skip
        status, reason = run_probe(tmp_path, synthetic_panel, body)
        assert status == "failed", body
        assert reason.startswith("the judge refused it: judged code " + isolate.REFUSALS["read"])


@known
def test_the_scan_names_file_io_however_it_is_reached():
    for source in ("from polars import read_parquet", "from numpy import fromregex",
                   "import polars.io", "import polars as pl\npl.DataFrame().serialize('p')",
                   "import numpy as np\nnp.lib._datasource",
                   "from polars import sql"):  # fmt: skip
        assert contract.scan(source, protected=True).errors, source
    for source in ("from polars import col, lit", "from pandas import to_datetime",
                   "import numpy as np\nnp.log(2.0)"):  # fmt: skip
        assert not contract.scan(source, protected=True).errors, source


@known
def test_without_srt_the_judge_refuses_unless_told(monkeypatch):
    monkeypatch.delenv("BLINDED_SRT", raising=False)
    monkeypatch.delenv("BLINDED_UNFENCED", raising=False)
    monkeypatch.setattr(isolate.shutil, "which", lambda name: None)
    with pytest.raises(isolate.Unfenced, match="BLINDED_SRT"):
        isolate.fence()
    monkeypatch.setenv("BLINDED_UNFENCED", "1")
    assert isolate.fence() is None


@known
def test_a_run_judged_without_srt_says_so_on_its_report(run, monkeypatch):
    from blinded.report import _fence_line

    monkeypatch.delenv("BLINDED_SRT", raising=False)
    monkeypatch.setattr(isolate.shutil, "which", lambda name: None)
    monkeypatch.setenv("BLINDED_UNFENCED", "1")
    assert _fence_line(run) is None
    Daemon(run, Limits())
    starts = [json.loads(s) for s in (run.vault / "fence.jsonl").read_text().splitlines()]
    assert starts[-1]["srt"] is False
    assert "without the OS sandbox (BLINDED_UNFENCED=1)" in _fence_line(run)


@known
def test_an_error_off_the_train_panel_tells_the_agent_only_that_it_failed(run):
    train_end = run.manifest["windows"]["train"][1]
    source = f"""
    import blinded as bl
    import polars as pl

    @bl.feature("fails after the train window")
    def late(panel):
        if panel.to_pandas("XRPUSDT.mid").index.max().isoformat() > "{train_end}":
            raise KeyError("words the judged code chose")
        return pl.col("XRPUSDT.mid").diff()
    """
    trial, daemon = judge(run, source)
    assert trial["status"] == "accepted"
    assert trial["validation"]["reason"] == "it raised an error on the validation panel"
    stored = json.loads(daemon.ledger.trial(trial["trial_id"])["val_metrics_json"])
    assert stored["detail"].startswith("KeyError at line")
