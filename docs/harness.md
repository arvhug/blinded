# The harness

The driver runs the agent one fresh session at a time and needs four things back from each: what it cost, how many turns it took, what the agent says it did, and which calls were refused. The harness is the piece that runs the session. The judge never talks to it, so swapping `claude -p` for the Agent SDK later changes nothing on the judge's side.

```python
class Harness:
    def run(self, prompt, workspace, limits, log=None) -> IterationResult: ...
```

`limits` is a {class}`~blinded.harness.SessionLimits`: model, effort, maximum turns, the session's spend cap in USD and its wall time. The result carries the session's own outcome (`success`, `error_max_turns`, `error_max_budget_usd`, ..., or `timeout` and `no_result` when the session never finished), its `session_id`, `total_cost_usd`, `num_turns`, the structured summary, the `permission_denials` and the path of the transcript.

## `claude -p`

{class}`~blinded.harness.claude_cli.ClaudeCLI` runs this line from the workspace, with blinded's `bin` directory first on `PATH`:

```text
claude -p PROMPT --restricted --tools Bash,Read,Edit,Write,Glob,Grep
    --settings DIR/agent/agent-settings.json
    --permission-mode dontAsk --permission-prompts none
    --disallowedTools "Read(//DIR/vault/**)" ... "Read(//DIR/driver/**)" ... "Edit(.claude/**)" "Bash(claude *)" WebFetch WebSearch
    --strict-mcp-config --disable-slash-commands --no-session-persistence
    --max-turns 60 --max-budget-usd 3 --model opus --effort high
    --append-system-prompt-file DIR/agent/system_append.md
    --json-schema '{...the iteration schema...}'
    --output-format stream-json --verbose
```

stdout goes to `DIR/driver/logs/iter_<n>.jsonl` and stderr beside it. The process starts in a new session, so when it outlives its wall time the harness can send `SIGTERM` and then `SIGKILL` to the whole process group: the `claude` process and every command its Bash tool started.

The transcript is one JSON event per line. The last is `{"type": "result", ...}`, which holds `total_cost_usd`, `num_turns`, `session_id`, `permission_denials`, `is_error`, the outcome as `subtype`, and, because of `--json-schema`, the agent's summary as `structured_output`. A session that was killed has no result line; its cost is then unknown and recorded as 0, which is the one way the spend guard can be short, and why each session also carries its own `--max-budget-usd`.

The transcripts are in `DIR/driver/`, which the agent's settings deny exactly as they deny the vault and `DIR/agent/`: a `Read` and `Edit` deny rule, the OS sandbox's `denyRead`, and the hook. They hold everything the agent saw and did, including the judge's answers, and a later session that could read them would have a second memory beside `LEDGER.md` and `NOTES.md`.

## The flags, checked against Claude Code 2.1.301

`--max-turns` and `--append-system-prompt-file` are not in `claude --help` on 2.1.301. Both are accepted: the option parser rejects an unknown flag before anything else runs (`error: unknown option '--bogus-flag'`), and it accepts `--max-turns 1` and reports a missing `--append-system-prompt-file` as "Append system prompt file not found". Whether a session stops at its turn limit is checked in the driver's smoke runs. `--max-budget-usd` is listed and is the hard guard either way.

Three flags matter here:

- `--strict-mcp-config`. `--restricted` ignores settings files but, as its help says, still loads MCP servers unless this flag is added. Without it, servers configured for your own account would be tools in the agent's session.
- `--no-session-persistence`. Sessions are not saved under `~/.claude/projects/`, and the only transcript is the driver's. One thing still lands there: a tool result too long to show, which Claude Code saves to `~/.claude/projects/<workspace>/<session>/tool-results/` for the agent to read back. It holds whatever the agent printed, so on private data, delete that directory after the run.
- `--disable-slash-commands`. No skills from your configuration in the agent's context.

`--json-schema` takes the schema as text. With it, Claude Code finishes the session by producing an object that validates against the schema; the agent sees it as one more tool call, which is why a session that does nothing else still reports two turns.
