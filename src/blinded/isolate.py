"""Judged code in fresh processes: every evaluation the judge compares runs alone.

The truncation test compares a feature's values on the full panel with its values
on panels cut at several times. Run in one process with one import of the trial's
file, module-level state carries from one call to the next: a dict keyed on the
panel's length hands every cut a slice of the full run, and a forward return
passes. A cut that is a view of the full panel carries the full panel's buffers
too (``.base`` of an array, ``.buffers()`` of an Arrow array), so even a fresh
import could read past the cut. And code imported where the judge computes its
verdicts can replace the judge's own functions.

So the judge never runs judged code. Each evaluation, the full panel or one cut,
is one worker: a fresh interpreter, ``python -s -P -B -m blinded.isolate``, that is sent
one panel pickled from that panel's rows alone, imports the trial's file, calls
the function once, sends the values back and exits. Nothing of one evaluation
reaches another, and nothing after a cut reaches the cut's worker, in memory or
on disk. The judge reads a worker's answer as JSON and an ``.npz`` loaded with
``allow_pickle=False``, never as a pickle, and puts every array through
:func:`blinded.contract.values` again, so a worker can send numbers of the
panel's shape and nothing else. Why an evaluation failed comes back as an
exception's type and its line in the trial's file, the same words the judge
gave when it ran the code itself.

One spare worker starts ahead, so its interpreter and imports overlap the
evaluation before it.

Each worker has its own temporary directory as its
working directory and ``TMPDIR``, removed when it exits.

Judged code is fenced in before any of it is imported (:func:`guard`): an
audit hook refuses writes outside the worker's directory, reads and listings
outside Python, blinded, the worker's directory and the trial files
(:func:`readable`), sockets, new processes, ctypes and the imports that reach
them, and a refusal comes back to the judge as an error row whatever the code
does with the exception. Each worker also runs inside the OS sandbox runtime
(``srt``, the one under Claude Code's Bash sandbox; ``BLINDED_SRT`` or on
``PATH``, :func:`sandbox_runtime`), with writes only in its directory, no
reads of the vault and no network: that also stops what the hook cannot see, a
write from a library's C or Rust code. Without ``srt`` the pool does not start
(:func:`fence`) unless ``BLINDED_UNFENCED=1`` says to run the code behind the
hook and the static scan alone. A library's own read of a file outside the run
is the scan's to stop: ``srt``'s read rules cover only the run's directories.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import select
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_LEN = struct.Struct("<Q")
_NAME = 200  # longest name, family, archetype or leg a worker may report
_TEXT = 4000  # longest hypothesis
# what a worker reads beyond Python, blinded and its own directory, found by tracing the
# opens of every worker the test suite starts: the time-zone database, which pandas opens
# as it is imported, and macOS's version file, read as Python starts
SYSTEM_READS = ("/usr/share/zoneinfo", "/System/Library/CoreServices/SystemVersion.plist")
UNFENCED = "BLINDED_UNFENCED"


class Limit(BaseException):
    """A worker ran past one evaluation's wall time or memory: the submission stops here.

    A ``BaseException``, so the judge's ``except Exception`` around one evaluation
    lets it through to the daemon, which records ``outcome`` (``timeout`` or ``oom``).
    """

    def __init__(self, outcome: str, why: str):
        super().__init__(outcome, why)
        self.outcome, self.why = outcome, why


def sandbox_runtime() -> str | None:
    """The OS sandbox runtime ``srt``: ``BLINDED_SRT``, else ``srt`` on ``PATH``, else None."""
    return os.environ.get("BLINDED_SRT") or shutil.which("srt")


class Unfenced(RuntimeError):
    """No OS sandbox to run judged code in, and no one said to run it without one."""


def fence() -> str | None:
    """The ``srt`` judged code runs under, or None when :data:`UNFENCED` is ``1``.

    The audit hook sees only what Python opens; a library's own C or Rust code reads
    and writes past it, and the static scan stops only the calls it can name. So
    without ``srt`` the judge does not run judged code at all, unless the environment
    says so explicitly (``BLINDED_UNFENCED=1``, for tests and local work), and the
    run records that it did (:func:`blinded.daemon.record_fence`).
    """
    srt = sandbox_runtime()
    if srt is not None:
        return srt
    if os.environ.get(UNFENCED) == "1":
        return None
    raise Unfenced(
        "the judge runs judged code only inside the OS sandbox runtime srt, and none was "
        "found: install it (npm install @anthropic-ai/sandbox-runtime@0.0.80 in a directory "
        "of your choice) and set BLINDED_SRT to its node_modules/.bin/srt, or put srt on "
        f"PATH. {UNFENCED}=1 runs judged code without it, for tests only."
    )


def readable(tmp: str) -> list[str]:
    """Where a worker may read, besides its trial files: one allowlist for the audit hook
    and for ``srt``.

    Python's installation (``sys.base_prefix``), the environment's (``sys.prefix``
    and its site-packages), the directory blinded is imported from (``src/`` of a
    checkout, which the import system lists, or site-packages), the worker's own
    directory, and :data:`SYSTEM_READS`; each as given and resolved, with and without
    macOS's ``/private``. Everything else is refused: the home directory, every other
    temporary directory, raw data wherever it sits, and the run directory but for the
    trial files.
    """
    import site

    import blinded
    from blinded.sandbox import aliases

    package = Path(blinded.__file__).resolve().parent
    roots = {sys.base_prefix, sys.prefix, *site.getsitepackages(), str(package.parent), tmp,
             *SYSTEM_READS}  # fmt: skip
    return sorted({a for p in roots for a in aliases(Path(p))})


class Failed(Exception):
    """The judged code raised, or its worker died; ``reason`` is what the agent may hear."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Spec:
    """What a worker evaluates: one feature or strategy of the file at ``code_path``.

    ``required`` maps each feature a strategy requires to the file it is taken
    from, ``None`` for the strategy's own file. ``legs`` and ``template`` are the
    binding's, for :func:`blinded.evaluate.positions`; ``values_legs`` what the
    output is checked against, a basket feature's legs or None.
    """

    code_path: str
    name: str
    kind: str
    required: dict[str, str | None] = field(default_factory=dict)
    legs: tuple[str, ...] | None = None
    template: str | None = None
    values_legs: tuple[str, ...] | None = None


class Pool:
    """Single-use workers for one judgement, one spare kept warm.

    ``limits`` are the daemon's (:class:`blinded.daemon.Limits`): one
    evaluation's wall time and resident memory. ``tick`` is called while a
    worker runs, so the daemon's watchdog sees the judge alive. ``deny`` are the
    run's vault, agent and driver directories and ``allow`` the trial directory
    inside them: a worker reads nothing of the run but trial files. The pool takes
    its ``srt`` from :func:`fence`, so it raises :class:`Unfenced` without one.
    """

    def __init__(self, limits, tick: Callable[[], None] | None = None,
                 deny: list[str] | None = None, allow: list[str] | None = None):  # fmt: skip
        self.limits = limits
        self.tick = tick or (lambda: None)
        self.deny, self.allow = list(deny or []), list(allow or [])
        self.srt = fence()
        self.spare: Worker | None = None
        self.started = 0

    def take(self) -> Worker:
        alive = self.spare is not None and self.spare.alive()
        worker = self.spare if alive else Worker(self.deny, self.allow, self.srt)
        self.spare = Worker(self.deny, self.allow, self.srt)
        self.started += 1
        return worker

    def run(self, job: dict) -> tuple[dict, dict[str, np.ndarray]]:
        """One job in one fresh worker: its JSON answer and its arrays."""
        worker = self.take()
        try:
            return worker.run(job, self.limits, self.tick)
        finally:
            worker.close()

    def close(self) -> None:
        if self.spare is not None:
            self.spare.close()
            self.spare = None

    def __enter__(self) -> Pool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Worker:
    """One fresh interpreter, started now, that takes one job and exits.

    What Python opens it may read only in :func:`readable` and its trial files, and
    write only in its own directory: the audit hook's rules (:func:`guard`). ``deny``
    are the run's directories, which neither the hook nor ``srt`` lets it read but for
    ``allow``, the trial directory; under ``srt`` every write outside its directory,
    a library's own included, is refused too, and the network.
    """

    def __init__(self, deny: list[str] | None = None, allow: list[str] | None = None,
                 srt: str | None = None):  # fmt: skip
        self.tmp = tempfile.mkdtemp(prefix="blinded-eval-")
        self.deny = list(deny or [])
        self.reads = readable(self.tmp)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env |= {"TMPDIR": self.tmp, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        # -s -P rather than -I, which would also drop the hash seed and the no-bytecode flag
        argv = [sys.executable, "-s", "-P", "-B", "-m", "blinded.isolate"]
        self.box = None
        if srt is not None:
            self.box = tempfile.mkdtemp(prefix="blinded-box-")
            settings = Path(self.box) / "srt-settings.json"
            settings.write_text(json.dumps(_box(self.tmp, self.deny, list(allow or []))))
            argv = [srt, "--settings", str(settings), "-c", shlex.join(argv)]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, cwd=self.tmp, env=env)  # fmt: skip
        self.buf = bytearray()
        self.pid = None

    def alive(self) -> bool:
        return self.proc.poll() is None

    def run(self, job: dict, limits, tick) -> tuple[dict, dict[str, np.ndarray]]:
        started = time.monotonic()
        panel = job.pop("panel", None)
        try:
            # pickled straight into the pipe, the job and then the panel: the judge never
            # holds a second copy of a panel, and the worker reads it as it comes
            pickle.dump(job | {"tmp": self.tmp, "deny": self.deny, "reads": self.reads},
                        self.proc.stdin, protocol=5)  # fmt: skip
            if panel is not None:
                pickle.dump(payload(panel), self.proc.stdin, protocol=5)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise Failed("the evaluation process died") from exc
        header = None
        while True:
            message = self._next(started, limits, tick)
            if message is None:
                raise Failed("the evaluation process died")
            if header is None:
                header = _json(message)
                if "refused" in header:
                    raise Failed(_refusal(header))
                if "hello" in header:
                    self.pid = header["hello"] if isinstance(header["hello"], int) else None
                    header = None
                    continue
                if not header.get("arrays"):
                    return header, {}
                continue
            return header, _arrays(message)

    def _next(self, started: float, limits, tick) -> bytes | None:
        """The next whole frame from the worker, waiting no longer than the limits allow."""
        fd = self.proc.stdout.fileno()
        while True:
            if len(self.buf) >= _LEN.size:
                (n,) = _LEN.unpack_from(self.buf)
                if len(self.buf) >= _LEN.size + n:
                    message = bytes(self.buf[_LEN.size : _LEN.size + n])
                    del self.buf[: _LEN.size + n]
                    return message
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                chunk = os.read(fd, 1 << 22)
                if not chunk:
                    return None
                self.buf += chunk
                continue
            tick()
            if time.monotonic() - started > limits.evaluation_seconds:
                raise Limit("timeout", f"one evaluation ran past {limits.evaluation_seconds:g} s")
            if self.pid is not None and _rss(self.pid) > limits.rss_bytes:
                raise Limit("oom", f"resident memory passed {limits.rss_bytes / 1024**3:g} GB")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        if self.pid is not None and self.pid != self.proc.pid:
            # under srt the interpreter is a grandchild: kill it too, not only the runtime
            with contextlib.suppress(OSError):
                os.kill(self.pid, 9)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout):
            with contextlib.suppress(OSError):
                stream.close()
        shutil.rmtree(self.tmp, ignore_errors=True)
        if self.box is not None:
            shutil.rmtree(self.box, ignore_errors=True)


def _box(tmp: str, deny: list[str], allow: list[str]) -> dict:
    """``srt``'s settings for one worker: writes only in ``tmp``, no reads of ``deny``
    but ``allow``, no network.

    The allowlist of reads (:func:`readable`) is the audit hook's: ``srt`` refuses
    reads only of the run's own directories, so a library's own read of a file
    elsewhere is the static scan's to stop.
    """
    from blinded.sandbox import aliases

    def every(paths):
        return sorted({a for p in paths for a in aliases(Path(p))})

    return {"filesystem": {"denyRead": every(deny), "allowRead": every(allow),
                           "allowWrite": every([tmp]), "denyWrite": []},
            "network": {"allowedDomains": [], "deniedDomains": []}}  # fmt: skip


def payload(panel) -> dict:
    """What a worker rebuilds a panel from: its frame and event tables, nothing else.

    Pickling a polars frame writes the rows it holds, not the buffers it shares
    with a longer frame, so a cut's worker never receives a row after the cut.
    The touch and depth sources (the fill engine's files) stay with the judge.
    """
    events = {key: (ev.frame, ev._delay) for key, ev in panel._events.items()}
    return {"frame": panel.frame, "events": events, "step_ns": panel.step_ns,
            "legs": panel._legs}  # fmt: skip


def describe(pool: Pool, code_path: str) -> list[dict]:
    """The features and strategies a trial file defines, each as plain fields, loaded in a
    worker; raises :class:`Failed` when the file does not import."""
    header, _ = pool.run({"op": "describe", "code_path": code_path})
    if "refused" in header:
        raise Failed(_refusal(header))
    if "error" in header:
        raise Failed(_reason(header, None))
    items = header.get("items")
    if not isinstance(items, list):
        raise Failed("the evaluation process died")
    return [_checked(item) for item in items]


def declared(item: dict):
    """A :class:`~blinded.contract.Feature` or :class:`~blinded.contract.Strategy` with the
    fields a worker reported, for :func:`blinded.contract.bind`; calling it fails."""
    from blinded.contract import Feature, Strategy

    def fn(*args, **kwargs):
        raise RuntimeError("a declared feature or strategy runs only in a worker")

    fn.__name__ = fn.__qualname__ = item["name"]
    if item["kind"] == "feature":
        return Feature(fn, item.get("hypothesis") or "", item["family"], item.get("target"),
                       item.get("pair"), item.get("archetype"), item.get("roles"))  # fmt: skip
    obj = Strategy(fn, item.get("requires") or [], item["family"], item.get("trades"),
                   item.get("pair"), item.get("archetype"), item.get("roles"))  # fmt: skip
    obj.hypothesis = item.get("hypothesis")
    return obj


class Isolated:
    """``panel -> output`` of one feature or strategy, each call in a fresh worker.

    Returns what the judge's own call returned: a ``pl.Series`` for one value per
    point, :class:`~blinded.contract.Legs` or :class:`~blinded.contract.Quotes`
    otherwise, each checked again by :func:`blinded.contract.values`.
    """

    def __init__(self, pool: Pool, spec: Spec, before: Callable[[], None] | None = None):
        self.pool, self.spec = pool, spec
        self.before = before or (lambda: None)

    def __call__(self, panel):
        import polars as pl

        from blinded.contract import ContractError, Legs, Quotes, values

        self.before()
        job = {"op": "evaluate", "spec": self.spec, "panel": panel}
        header, arrays = self.pool.run(job)
        if "refused" in header:
            raise Failed(_refusal(header))
        if "error" in header:
            if header["error"] == "contract":
                raise ContractError("the output does not fit the contract")
            raise Failed(_reason(header, self.spec.code_path))
        kind, keys = header.get("type"), header.get("keys")
        n = len(panel)
        malformed = Failed("the evaluation process sent a malformed answer")
        for v in arrays.values():
            if v.dtype != np.float64 or v.ndim != 1 or len(v) != n:
                raise malformed
        if kind == "array" and set(arrays) == {"v"}:
            out = pl.Series("v", arrays["v"])
        elif kind in ("legs", "quotes") and _names(keys) and len(keys) == len(arrays):
            named = {k: arrays[f"k{i}"] for i, k in enumerate(keys) if f"k{i}" in arrays}
            if len(named) != len(keys):
                raise malformed
            out = Quotes(named) if kind == "quotes" else Legs(named)
        else:
            raise malformed
        # the checks the judge's own call made, made again here on what came back
        spec = self.spec
        if spec.kind == "strategy":
            res = values(out, panel, spec.legs, spec.template)
        else:
            res = values(out, panel, spec.values_legs)
            if isinstance(res, Legs) and spec.values_legs is None:
                raise ContractError("a feature gives one value per point")
        return res if isinstance(res, Legs) else pl.Series("v", res)


def _checked(item) -> dict:
    """A worker's description of one feature or strategy, refused unless every field has
    the type and length a decorator gives it."""
    bad = Failed("the evaluation process sent a malformed answer")
    if not isinstance(item, dict) or item.get("kind") not in ("feature", "strategy"):
        raise bad
    out = {"kind": item["kind"]}
    for key in ("name", "family", "target", "trades", "archetype"):
        v = item.get(key)
        if v is not None and not (isinstance(v, str) and 0 < len(v) <= _NAME):
            raise bad
        out[key] = v
    if out["name"] is None:
        raise bad
    hyp = item.get("hypothesis")
    if hyp is not None and not (isinstance(hyp, str) and len(hyp) <= _TEXT):
        raise bad
    out["hypothesis"] = hyp
    pair = item.get("pair")
    if pair is not None and not _names(pair):
        raise bad
    out["pair"] = None if pair is None else tuple(pair)
    roles = item.get("roles") or {}
    if not isinstance(roles, dict) or not (_names(list(roles)) or not roles):
        raise bad
    if not all(isinstance(v, str) and len(v) <= _NAME for v in roles.values()):
        raise bad
    out["roles"] = dict(roles)
    requires = item.get("requires") or []
    if not (_names(requires) or not requires):
        raise bad
    out["requires"] = list(requires)
    return out


def _names(v) -> bool:
    return (isinstance(v, list | tuple) and len(v) > 0
            and all(isinstance(s, str) and 0 < len(s) <= _NAME for s in v))  # fmt: skip


def _refusal(header: dict) -> str:
    """The words for a refusal, built here from the hook's word and line."""
    word, line = header.get("refused"), header.get("line")
    if word not in REFUSALS:
        return "the evaluation process sent a malformed answer"
    where = ""
    if isinstance(line, int) and not isinstance(line, bool) and 0 < line < 10**7:
        where = f" at line {line}"
    return f"the judge refused it: judged code {REFUSALS[word]}{where}"


def _reason(header: dict, code_path: str | None) -> str:
    """The words for a failed evaluation, built here from the worker's type and line."""
    name, line = header.get("type"), header.get("line")
    if not (isinstance(name, str) and name.isidentifier() and len(name) <= 80):
        return "the evaluation process died"
    if isinstance(line, int) and not isinstance(line, bool) and 0 < line < 10**7:
        return f"{name} at line {line}"
    return name


def _json(message: bytes) -> dict:
    try:
        out = json.loads(message.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise Failed("the evaluation process sent a malformed answer") from exc
    if not isinstance(out, dict):
        raise Failed("the evaluation process sent a malformed answer")
    return out


def _arrays(message: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(message), allow_pickle=False) as found:
            return {k: found[k] for k in found.files}
    except Exception as exc:
        raise Failed("the evaluation process sent a malformed answer") from exc


def _frame(data: bytes) -> bytes:
    return _LEN.pack(len(data)) + data


def _rss(pid: int) -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    text = out.stdout.strip()
    return int(text) * 1024 if text.isdigit() else 0


# the guard ----------------------------------------------------------------------------------

# what a refusal names, by the word the hook sends; the judge prints only these words
REFUSALS = {
    "write": "opened a file for writing outside its own temporary directory",
    "read": "read or listed a file outside Python, blinded and its own directory",
    "files": "changed the file system outside its own temporary directory",
    "link": "made a link",
    "chdir": "changed directory",
    "socket": "used a socket",
    "process": "started or signalled a process",
    "ctypes": "used ctypes",
    "import": "imported ctypes, socket or subprocess",
    "database": "opened a database",
}
_FS_EVENTS = {
    # event: (indices of the paths it changes, index of its dir_fd or None)
    "os.remove": ((0,), 1), "os.rmdir": ((0,), 1), "os.mkdir": ((0,), 2),
    "os.rename": ((0, 1), 2), "os.truncate": ((0,), None), "os.chmod": ((0,), 2),
    "os.chown": ((0,), 3), "os.utime": ((0,), 3), "os.chflags": ((0,), None),
    "os.lchflags": ((0,), None), "os.lchmod": ((0,), None), "os.setxattr": ((0,), None),
    "os.removexattr": ((0,), None), "shutil.copyfile": ((1,), None),
    "shutil.copymode": ((1,), None), "shutil.copystat": ((1,), None),
    "shutil.copytree": ((1,), None), "shutil.move": ((0, 1), None),
    "shutil.rmtree": ((0,), 1), "shutil.chown": ((0,), None),
    "shutil.make_archive": ((0,), None), "shutil.unpack_archive": ((1,), None),
}  # fmt: skip
_PROCESS_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
                   "os.fork", "os.forkpty", "os.kill", "os.killpg", "pty.spawn"}  # fmt: skip
_BLOCKED_IMPORTS = {"ctypes", "socket", "subprocess", "_socket", "_ctypes", "_posixsubprocess"}


def guard(tmp: str, deny: list[str], trial_files: list[str], notify: int,
          reads: list[str] | None = None) -> None:  # fmt: skip
    """Install the audit hook that fences judged code in, before any of it is imported.

    Refused, by Python's audit events (checked against 3.12 in
    ``tests/test_isolate.py``): ``open`` for writing anywhere but ``tmp``; ``open``
    for reading, and listing a directory (``os.listdir``, ``os.scandir``), anywhere
    but the ``trial_files`` and the allowlist ``reads`` (:func:`readable`), and
    anywhere under ``deny`` (the run's vault, agent and driver directories) but the
    ``trial_files``; every file-system change outside
    ``tmp`` (``os.remove``, ``os.rename``, ``os.mkdir``, ``shutil.*`` and the
    rest of :data:`_FS_EVENTS`); links, ``os.chdir``; ``socket.*``;
    ``subprocess.Popen``, ``os.system``, ``os.exec*``, ``os.spawn*``,
    ``os.posix_spawn``, ``os.fork``, ``os.kill``; ``ctypes.*``;
    ``sqlite3.connect``; and an ``import`` of ctypes, socket or subprocess
    while the trial's code is on the stack. A refusal raises
    ``PermissionError`` and, before that, writes a frame on ``notify`` that the
    judge reads first, so code that catches the error is still refused.

    The hook takes everything it calls now, as C functions bound in its
    closure, so judged code that rebinds ``os.path`` or this module's names
    later changes nothing it does. Paths are made absolute and normalised
    lexically; a link could carry a path outside ``tmp`` past that, so links
    are refused outright. What Python does not see, a write by a library's own
    C or Rust code, is the static scan's and the OS sandbox's to stop.
    """
    import posix

    write, fspath, getcwd, getframe = posix.write, posix.fspath, posix.getcwd, sys._getframe
    normpath = posix._path_normpath
    pack = _LEN.pack
    flags_w = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    inside = tuple({normpath(tmp) + "/", normpath(os.path.realpath(tmp)) + "/"})
    denied = tuple({normpath(d) + "/" for d in deny})
    # None: no allowlist, only ``deny`` (a caller that predates it)
    allowed = None if reads is None else tuple({normpath(r) + "/" for r in reads})
    trials = frozenset(normpath(f) for f in trial_files)
    # the import system looks for a trial's cached bytecode beside it
    caches = tuple(normpath(os.path.dirname(f)) + "/__pycache__/"
                   + os.path.basename(f).rsplit(".", 1)[0] + "." for f in trials)  # fmt: skip
    # copies, so judged code that changes this module's tables changes nothing here
    fs_events, process_events = dict(_FS_EVENTS), frozenset(_PROCESS_EVENTS)
    blocked, words = frozenset(_BLOCKED_IMPORTS), dict(REFUSALS)
    # a module already imported raises no import event, so these leave sys.modules: the
    # libraries keep the references they hold, and an import of one from judged code is seen
    for name in [n for n in sys.modules if n.split(".")[0] in blocked]:
        del sys.modules[name]

    def absolute(path) -> str | None:
        if isinstance(path, int):
            return None
        path = fspath(path)
        if isinstance(path, bytes):
            path = path.decode("utf-8", "surrogateescape")
        if not path.startswith("/"):
            path = getcwd() + "/" + path
        return normpath(path)

    def within(path) -> bool:
        a = absolute(path)
        return a is not None and (a + "/").startswith(inside)

    def line() -> int:
        f = getframe(2)
        while f is not None:
            if f.f_code.co_filename in trials:
                return f.f_lineno
            f = f.f_back
        return 0

    def refuse(word: str):
        data = f'{{"refused": "{word}", "line": {line()}}}'.encode()
        data = pack(len(data)) + data
        while data:
            data = data[write(notify, data) :]
        raise PermissionError(f"the judge refused this: judged code {words[word]}")

    def may_read(path) -> bool:
        # decided from the path alone, before the OS is asked: a missing file and a
        # present one are refused alike
        a = absolute(path)
        if a is None or a in trials or a.startswith(caches):
            return True
        if (a + "/").startswith(denied):
            return False
        return allowed is None or (a + "/").startswith(allowed)

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = args
            writing = (isinstance(flags, int) and flags & flags_w) or (
                isinstance(mode, str) and ("w" in mode or "a" in mode or "x" in mode
                                           or "+" in mode))  # fmt: skip
            if writing:
                if not within(path):
                    refuse("write")
            elif not may_read(path):
                refuse("read")
        elif event in ("os.listdir", "os.scandir"):
            if not may_read(getcwd() if args[0] is None else args[0]):
                refuse("read")
        elif event in fs_events:
            paths, dir_fd = fs_events[event]
            if dir_fd is not None and len(args) > dir_fd and args[dir_fd] not in (None, -1):
                refuse("files")
            for k in paths:
                if len(args) > k and not (isinstance(args[k], int) or within(args[k])):
                    refuse("files")
        elif event in ("os.symlink", "os.link"):
            refuse("link")
        elif event == "os.chdir":
            refuse("chdir")
        elif event.startswith("socket."):
            refuse("socket")
        elif event in process_events:
            refuse("process")
        elif event.startswith("ctypes."):
            refuse("ctypes")
        elif event.startswith("sqlite3."):
            refuse("database")
        elif event == "import":
            root = str(args[0]).split(".")[0]
            if root in blocked and line():
                refuse("import")

    sys.addaudithook(hook)


# the worker's side ------------------------------------------------------------------------


def _send(fd: int, header: dict, arrays: dict[str, np.ndarray] | None = None) -> None:
    data = _frame(json.dumps(header | {"arrays": bool(arrays)}).encode())
    if arrays:
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        data += _frame(buf.getvalue())
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _error(exc: BaseException, code_path: str | None) -> dict:
    from blinded.contract import ContractError

    if isinstance(exc, ContractError):
        return {"error": "contract"}
    frames = traceback.extract_tb(exc.__traceback__)
    mine = [f for f in frames if code_path is not None and Path(f.filename) == Path(code_path)]
    return {"error": "exception", "type": type(exc).__name__,
            "line": mine[-1].lineno if mine else None}  # fmt: skip


def _describe_items(module) -> list[dict]:
    out = []
    for f in module.features:
        out.append({"kind": "feature", "name": f.name, "hypothesis": f.hypothesis,
                    "family": f.family, "target": f.target, "pair": f.pair,
                    "archetype": f.archetype, "roles": f.roles})  # fmt: skip
    for s in module.strategies:
        out.append({"kind": "strategy", "name": s.name, "requires": s.requires,
                    "family": s.family, "trades": s.trades, "pair": s.pair,
                    "archetype": s.archetype, "roles": s.roles,
                    "hypothesis": getattr(s, "hypothesis", None)})  # fmt: skip
    return out


def _evaluate(spec: Spec, panel):
    from blinded import contract
    from blinded.evaluate import positions

    module = contract.load(spec.code_path)
    if spec.kind == "feature":
        (obj,) = [f for f in module.features if f.name == spec.name]
        fn = obj.fn
    else:
        (obj,) = [s for s in module.strategies if s.name == spec.name]
        required = {}
        for need, path in spec.required.items():
            source = module if path is None else contract.load(path)
            (required[need],) = [f for f in source.features if f.name == need]
        fn = positions(obj, required, spec.legs, spec.template)
    return contract.values(fn(panel), panel, spec.values_legs)


def _encode(out) -> tuple[dict, dict[str, np.ndarray]]:
    from blinded.contract import Legs, Quotes

    if isinstance(out, Legs):
        kind = "quotes" if isinstance(out, Quotes) else "legs"
        keys = [str(k) for k in out]
        arrays = {f"k{i}": np.asarray(out[k], dtype=np.float64) for i, k in enumerate(out)}
        return {"type": kind, "keys": keys}, arrays
    return {"type": "array"}, {"v": np.asarray(out, dtype=np.float64)}


def _rebuild(payload: dict):
    from blinded.panel import Events, Panel

    events = {key: Events(frame, delay) for key, (frame, delay) in payload["events"].items()}
    return Panel(payload["frame"], events, payload["step_ns"], payload["legs"])


def _orphaned() -> None:
    try:
        while os.read(0, 1 << 16):
            pass
    finally:
        os._exit(1)


def main() -> None:
    """A worker: one job from stdin, one answer on stdout, then exit."""
    out = os.dup(1)
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 1)
    # the imports a judgement needs, before the job arrives, so a spare is warm
    import polars  # noqa: F401

    from blinded import contract, evaluate, features  # noqa: F401

    _send(out, {"hello": os.getpid()})
    code_path = None
    stdin = os.fdopen(0, "rb", buffering=1 << 20, closefd=False)
    try:
        job = pickle.load(stdin)
        code_path = job.get("code_path") or job["spec"].code_path
        trials = [code_path]
        panel = None if job["op"] == "describe" else _rebuild(pickle.load(stdin))
        if job["op"] != "describe":
            trials += [p for p in job["spec"].required.values() if p is not None]
        # the judge holds this pipe open until it has the answer: when it closes, the judge
        # has stopped waiting (or died), and the worker goes too
        threading.Thread(target=_orphaned, daemon=True).start()
        # temporary files go to the worker's own directory, whatever TMPDIR the sandbox set
        os.environ["TMPDIR"] = tempfile.tempdir = job["tmp"]
        files = [str(Path(p).resolve()) for p in trials]
        guard(job["tmp"], job.get("deny") or [], files, out, job.get("reads"))
        if panel is None:
            _send(out, {"items": _describe_items(contract.load(code_path))})
        else:
            header, arrays = _encode(_evaluate(job["spec"], panel))
            _send(out, header, arrays)
    except EOFError:
        os._exit(1)
    except BaseException as exc:  # noqa: BLE001  the judged code may raise anything
        _send(out, _error(exc, code_path))
    os._exit(0)


if __name__ == "__main__":
    main()
