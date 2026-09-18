# agent-shepherd

[![CI](https://github.com/CallMeHFK/agent-shepherd/actions/workflows/ci.yml/badge.svg)](https://github.com/CallMeHFK/agent-shepherd/actions)

A supervisor plugin that observes AI agents (**QwenPaw**, **Claude Code**, **Codex**) while they reason and execute tools, and injects concise corrective guidance when they drift off course.

The supervisor is a two-tier policy engine:

1. **Tier 0 — deterministic detectors** (zero cost, always on): loop detection (exact and near-duplicate calls), regression detection, off-spec edits, context rot, and sustained **drift** — a CUSUM alarm over a cheap failure signal whose threshold is calibrated by Monte-Carlo simulation to a 5% per-window false-alarm budget.
2. **Tier 1 — LLM step scorer** (PRM-style, sliding window): scores each recent step on on-goal / justified / verified, names the step where the trajectory drifted, and returns actionable guidance. The judge wakes at natural checkpoints (prompt start, iteration end, session stop) and — mid-iteration — only when the CUSUM statistic climbs toward its alarm line, so the expensive reviewer stays asleep while the agent is healthy.

The design decisions above, and the papers that motivate them, are recorded in [RESEARCH_NOTES.md](RESEARCH_NOTES.md).

The judge model backend is pluggable: **Agnes** (`agnes-3.0-flash` via the Agnes AI Hub, OpenAI-compatible) is the default, and any OpenAI-compatible endpoint (local **vLLM**, cc-switch, etc.) works by setting `SHEPHERD_JUDGE_BASE_URL` / `SHEPHERD_JUDGE_MODEL`.

## Install

```bash
uv pip install -e .
```

Create a starter config (the daemon does this automatically on first start):

```yaml
# ~/.shepherd/config.yaml
judge:
  backend: agnes
  base_url: https://apihub.agnes-ai.com/v1
  model: agnes-3.0-flash
  api_key: ${SHEPHERD_JUDGE_API_KEY}
  timeout: 30
policy:
  nudge_threshold: 0.6
  block_threshold: 0.85
  max_guidance_tokens: 500
  fail_open: true
  drift_enabled: true
  drift_target_fpr: 0.05
  drift_watch_fraction: 0.6
agents:
  qwenpaw: {enabled: true, block_enabled: false}
  claude: {enabled: true, block_enabled: false}
  codex: {enabled: true, block_enabled: false}
port: 4890
```

Start the daemon:

```bash
shepherd start
```

## Adapters

### QwenPaw (in-process plugin, highest fidelity)

```bash
shepherd install qwenpaw
```

Installs `~/.qwenpaw/plugins/agent-shepherd` and restart QwenPaw. The plugin registers:

- an agentscope `MiddlewareBase` that observes every reasoning delta and tool call in-process, and
- a `StopGate` that asks the daemon for a verdict at the end of each ReAct iteration and injects the guidance as a new user turn (`INTERRUPT_AND_CONTINUE`).

A zero-install REST/SSE fallback client (`agent_shepherd.adapters.qwenpaw.rest_client.QwenPawRestClient`) observes `127.0.0.1:19999` and resolves Tool Guard approvals.

### Claude Code

```bash
shepherd install claude
```

Writes `PreToolUse` / `PostToolUse` / `Stop` / `UserPromptSubmit` hooks into `~/.claude/settings.json`. The hook thin-shell (`shepherd-hook claude`) translates the daemon verdict into Claude Code's `hookSpecificOutput.additionalContext` / `decision`.

### Codex CLI

```bash
shepherd install codex
```

Writes the same events into `~/.codex/hooks.json` (the Codex-specific schema: a
top-level `hooks` map of MatcherGroups with `{"type": "command", ...}` handlers).
The thin-shell (`shepherd-hook codex`) translates verdicts into Codex's
`hookSpecificOutput.additionalContext` / top-level `decision: block`.

**Hook trust:** Codex only runs hooks that have been *persisted as trusted*.
- Interactive TUI: the first time Codex sees the new hooks it prompts to trust
  them; approve once and they run on every later session.
- Non-interactive `codex exec`: untrusted hooks are **silently skipped** (no
  warning). Pass `--dangerously-bypass-hook-trust` so automation runs the hooks
  without the persisted trust, e.g. `codex exec --skip-git-repo-check
  --dangerously-bypass-hook-trust "..."`.

## Audit and replay

Every observation and verdict is appended to `~/.shepherd/sessions/<agent>/<session>.jsonl` (append-only, never rewritten):

```bash
shepherd replay qwenpaw <session-id>
shepherd tail qwenpaw <session-id>
```

## Tests

```bash
uv run pytest -q
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, adapter/detector conventions, commit and PR guidelines, and the
strict no-real-secrets rule. Quick start:

```bash
uv sync --all-extras && uv run pytest -q && uv run ruff check .
```

Interactions in this project are governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT