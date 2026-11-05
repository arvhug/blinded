"""The agent's side of the file drop: ``blinded submit``. Standard library only.

It copies the file to ``.judge/inbox/<id>.py``, writes ``<id>.json`` beside it
(last, so the daemon never sees half a request), and waits for
``.judge/outbox/<id>.json``.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


def find_judge(start: Path | None = None) -> Path:
    """The ``.judge`` directory of the workspace containing ``start`` (default: the cwd)."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / ".judge" / "inbox").is_dir():
            return folder / ".judge"
    raise FileNotFoundError("not inside a workspace: no .judge/inbox here or above")


def submit(
    file: Path,
    name: str | None = None,
    timeout: float = 1800.0,
    judge: Path | None = None,
    window: str | None = None,
) -> dict:
    """Drop one evaluation request and wait for the response.

    ``window`` is read only by a naive run's judge; a protected judge ignores it.
    """
    judge = judge or find_judge(Path(file).parent)
    rid = uuid.uuid4().hex
    inbox = judge / "inbox"
    (inbox / f"{rid}.py").write_text(Path(file).read_text())
    request = {"kind": "eval", "file": Path(file).name, "name": name}
    if window:
        request["window"] = window
    tmp = inbox / f"{rid}.json.tmp"
    tmp.write_text(json.dumps(request))
    tmp.replace(inbox / f"{rid}.json")
    answer = judge / "outbox" / f"{rid}.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if answer.is_file():
            return json.loads(answer.read_text())
        time.sleep(0.2)
    raise TimeoutError(f"no answer from the judge within {timeout:g} s; is the daemon running?")
