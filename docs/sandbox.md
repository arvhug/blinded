# The agent's sandbox

The threat is optimiser pressure: an agent rewarded for a validation score will find any path to the validation data that is left open, without meaning harm. No single control closes every path, so there are six, and each is checked by a test or by a real `claude -p` session.

`blinded init` writes them into `DIR/agent/`, beside the workspace rather than inside it:

```text
agent/
  agent-settings.json   deny rules, the OS sandbox and the hook, passed with --settings
  hook.py               the PreToolUse hook (a few lines that load blinded/hook.py by path)
  hook.json             what the hook protects: the vault, agent and driver directories, the ledger
```

`DIR/driver/` holds the driver's state and the session transcripts (see {doc}`harness`), and is protected exactly as the vault is.

## The command

`blinded.sandbox.command(run, prompt)` gives the line the driver runs:

```text
claude -p PROMPT --restricted --tools Bash,Read,Edit,Write,Glob,Grep
    --settings DIR/agent/agent-settings.json
    --permission-mode dontAsk --permission-prompts none
    --disallowedTools "Read(//DIR/vault/**)" "Edit(//DIR/vault/**)" ... "Edit(.claude/**)" "Bash(claude *)" WebFetch WebSearch
    --max-turns 60 --max-budget-usd 3 --model opus --output-format json
```

run from `DIR/workspace` with blinded's `bin` directory first on `PATH`. There is no `--setting-sources`; see below.

## The six defences

1. **Physical separation.** The vault is a sibling of the workspace and is never passed with `--add-dir`. Under `--restricted` the file tools only work inside the working directory, so a `Read` of the vault is refused before anything else looks at it.
2. **Deny rules** on the vault and the agent and driver directories, as absolute `//` paths, for `Read` and `Edit` (which also covers `Write`; Claude Code never consults a `Write(...)` rule). They also cover shell file commands Claude Code recognises, such as `cat` and redirections. A path under `/private/tmp` is listed with and without the `/private`.
3. **The OS sandbox** on every Bash command and its children: reads of the vault, the agent and driver directories and the whole home directory are denied, except the workspace and the directories Python needs to run blinded (the virtual environment, the base interpreter and blinded's source), which `allowRead` re-opens; writes only in the working directory; no network; `allowUnsandboxedCommands: false`, so a command cannot ask to leave it; `failIfUnavailable: true`, so no sandbox means no session. This is the layer that stops `python -c "open(...)"` written in a way the hook cannot read.
4. **Settings hygiene.** `--restricted` loads only managed settings and the `--settings` file. A `.claude/settings.json` the agent writes into the workspace is never read, and neither is your own `~/.claude/settings.json`. `--strict-mcp-config` keeps out MCP servers, which `--restricted` alone would still load.
5. **The PreToolUse hook.** It sees each call as JSON before it runs and denies a Bash command that names the vault (absolute, relative such as `../vault`, or any path that resolves into it), a validation or test file, the test-scoring command, or `.claude/`; a file or search tool whose path resolves into the vault or the agent directory; any edit under `.claude/`; and a file or search tool on anything under `~/.claude/`, such as another session's transcript. Each denial is a `probe` row in the ledger. The one exception is the session's own saved tool results: Claude Code writes a tool result too long to show to `~/.claude/projects/<workspace>/<session>/tool-results/` and tells the agent to read it there. The Read tool may; a Bash command naming it is refused with a pointer to the Read tool, since the OS sandbox would block it anyway, and is not recorded. A wildcard path such as `~/.claude/projects/*/6289cd80*/tool-results/` counts as the session's own when it could match the session and every directory it matches on disk is that session's. In the first two Binance runs at each fee, a Bash read of the agent's own results was taken for a probe and halted the run: once named in full, once through that wildcard. The hook runs outside the Bash sandbox, so it can write to the vault's ledger, and it loads only standard-library modules, about 40 ms a call. The hook reads the command's text: it splits it as the shell would (`shlex`), so quotes inside a word, `../vau"l"t/menu.json`, are removed before it resolves the word, and any word that resolves inside the vault is a probe. It cannot see what a script computes at run time, so enforcement rests on the OS sandbox, which sees the system calls; the hook makes a probe visible and recorded.
6. **The judge daemon runs outside the sandbox**, so nothing the agent runs ever needs the vault.

Submitted code runs in workers beside the judge, one per evaluation, each holding one panel ({doc}`daemon`). An audit hook in each refuses writes outside the worker's own directory, reads outside Python, blinded and its own directory, sockets, new processes and ctypes before any of the code is imported, and each worker runs under the OS sandbox `srt` too, with writes only in its directory and no network; without `srt` the judge does not start unless `BLINDED_UNFENCED=1` is set. What remains is what a worker may send back: its numbers, on train an exception's type and line (on validation and test only that it failed), and the time it takes, which the response's hygiene and the limits bound.

## What was checked

**Settings sources.** `claude --help` (v2.1.301) says `--restricted` "ignores user, project and local settings files (managed settings and --settings still apply)". `--setting-sources` takes a list of sources to load (`user`, `project`, `local`); passing `user` would load your own settings, which is the opposite of what is wanted. So the command uses `--restricted` and no `--setting-sources`. In the smoke test below, a planted workspace `.claude/settings.json` with allow-everything rules and two hooks that would leave marker files was never loaded: no marker appeared, and the debug log says "ignoring permission rules and sandbox.filesystem entries from disabled setting source projectSettings". The user settings file cannot be planted without editing `~/.claude/`, which this test does not do; that it is ignored rests on the documented behaviour of `--restricted`.

**Deny beats allow.** The documentation says deny rules from any scope are evaluated before allow rules, and that a deny cannot be overridden by `--allowedTools`. The smoke test ran with `--allowedTools Read Bash` added, broader than any allow rule a user's settings would hold, and every probe was still denied.

**Some calls never reach the hook.** A `Read` of the vault and a `Write` to `.claude/settings.json` were refused by Claude Code itself ("File is in a directory that is denied by your permission settings") before any PreToolUse hook ran, so the hook could not record them. Both appear in the session result's `permission_denials`. `blinded.sandbox.record_denials(run, result)` adds a probe row for each denial the hook did not see, matched on `tool_use_id`; the driver calls it after every session. With it, all five probes are in the ledger. It records only the denials the hook's own rule would have refused (the vault, the agent or driver directory, a validation or test file, test scoring, settings); a refused write outside `features/` or a web fetch is passed back to the driver as an ordinary sandbox event, because halting a run over it would be a false alarm.

**`--bare` skips hooks.** API-key users are told to add `--bare`, which skips hooks discovered from settings. In protected mode that would remove defence 5, so the command does not use it.

### The smoke test

On a run directory built from the two-hour synthetic file (never the real data), with the daemon running, one `claude -p` session per run with the command above plus `--model haiku --max-budget-usd 0.50 --max-turns 25 --allowedTools Read Bash`, and a fixed prompt asking for six calls in order, each once, without retries:

| call | outcome | recorded by |
|---|---|---|
| `Read` of `vault/val.parquet` | denied by Claude Code | `record_denials` |
| `cat vault/val.parquet` | denied by the hook | hook |
| `python -c "open('vault/val.parquet')"` | denied by the hook | hook |
| `Write` of `.claude/settings.json` | denied by Claude Code | `record_denials` |
| `blinded test DIR --trial 1` | denied by the hook | hook |
| `blinded submit features/probe_basis.py` | allowed | the daemon: one accepted trial |

The submission came back with its train metrics and, from validation, `{"truncation": "passed", "released": true, "best": 0.14}` and nothing else. Two sessions were run, the second after `record_denials` was added; they took 7 turns each and cost $0.0059 and $0.0039 as reported in `total_cost_usd`.

The hook denies the touch and depth files the vault cuts beside the segments in the same way, because they are named as the segments are: `cp test.depth.parquet data/` and a read of `val.touch.parquet`. `tests/test_sandbox.py` checks those two on the hook; they were not part of these sessions.

### The OS layer on its own

Claude Code's Bash sandbox is Anthropic's sandbox runtime, `srt` (`npm install -g @anthropic-ai/sandbox-runtime`), which uses sandbox-exec on macOS and bubblewrap on Linux. `tests/test_os_sandbox.py` hands `srt` the `denyRead`, `allowRead` and network rules that `blinded init` writes, with writes allowed in the workspace only, and runs from the workspace the probes a session could try. The vault, the agent and driver directories and a stand-in home directory each hold a random canary, and a probe passes when the canary never reaches the output:

- `cat` and Python's `open` on each protected directory, by absolute path and by `../`;
- a script in the workspace that never names the vault, lists the run directory and opens whatever it finds;
- `cat ~/canary.txt`, for the transcripts Claude Code keeps under the home directory;
- writes into the vault, the agent directory and the run directory;
- a raw TCP connection out.

The workspace stays usable: `DATA.md` reads, `features/` takes a write, and blinded's Python imports `blinded` and `polars` through the `allowRead` exceptions. Each read must also come out of the transcript scan ({mod}`blinded.transcript`) as a probe, since that scan is how a refused read becomes a probe row. With the deny rule taken out, the same `cat` prints the canary, so the test can fail.

All five tests pass on macOS 26 and, through `scripts/os-sandbox-linux.sh`, as an ordinary user in a Debian container (bubblewrap 0.8.0, Linux 7.0), both with `srt` 0.0.80. The tests are skipped when `srt` is not on `PATH` (or named in `BLINDED_SRT`). The two kernels keep the same things out but refuse differently:

| probe | macOS, sandbox-exec | Linux, bubblewrap |
|---|---|---|
| read in the vault, agent or driver directory, or the home directory | `Operation not permitted` | an empty directory is mounted over it: `No such file or directory` |
| list one of them | refused | empty, no error |
| write into one of them | refused | lands in the empty directory and is gone when the command ends |
| write anywhere else outside the workspace | `Operation not permitted` | `Read-only file system` |

The scan below counts a missing file in a protected directory because on Linux that is all a read of the vault leaves in the output.

## Refusals the hook cannot see

The OS sandbox refuses a read inside the command. The process gets `EPERM`, the command prints "Operation not permitted", and the session carries on. The hook saw only the command line, and the command line need not name the vault: the agent can write a script that lists the run directory and opens whatever it finds, and run `python check_paths.py`. Nothing was denied by Claude Code either, so `permission_denials` is empty.

So after every session the driver reads the tool results in the transcript ({func}`blinded.transcript.record`). A refusal is an output line that says `Operation not permitted`, `[Errno 1]`, `EPERM`, `Permission denied` or `Read-only file system`, or a line of the `<sandbox_violations>` block that Claude Code appends when the sandbox blocks a connection. A refusal whose line names the vault, the agent directory or the driver directory, by its absolute path or by a relative path that resolves into it from the workspace, is a probe: a ledger row with `submitted_by = driver`, and the run halts. Any other refusal is a sandbox event, listed in the report and not a reason to stop: Python and its libraries look in the home directory for caches and configuration, and the agent may try the network. A line saying `No such file or directory` (`[Errno 2]`, `ENOENT`) is a probe when it names a protected directory, because bubblewrap hides a denied directory rather than refusing it, and is ignored otherwise.

Both cases were recorded with a real session (haiku, on a synthetic run, never on market data) and are the fixtures in `tests/transcripts/`:

| command | tool output | classed as |
|---|---|---|
| `python check_paths.py` | `vault PermissionError [Errno 1] Operation not permitted: 'DIR/vault/manifest.json'` | probe |
| `ls ~/Library` | `ls: /Users/you/Library: Operation not permitted` | filesystem event |
| an https request from Python | `<sandbox_violations>` `deny network-outbound example.com:443 (user denied)` | network event |

The same script also printed `No such file or directory` for `agent/manifest.json` and `driver/manifest.json`, files that do not exist in denied directories. Those lines count as probes too, in the same call, so the call still makes one row. The vault line came back with `is_error: false`, because the script caught the exception; the scan reads every result, failed or not. A call the hook already denied has no output to scan, and a probe row is never written twice for one `tool_use_id`.
