# agent-shepherd

[![CI](https://github.com/CallMeHFK/agent-shepherd/actions/workflows/ci.yml/badge.svg)](https://github.com/CallMeHFK/agent-shepherd/actions)

A supervisor plugin that observes AI agents (**QwenPaw**, **Claude Code**, **Codex**) while they reason and execute tools, and injects concise corrective guidance when they drift off course.

The supervisor is a two-tier policy engine:

1. **Tier 0 — deterministic detectors** (zero cost, always on): loop detection (exact and near-duplicate calls), regression detection, off-spec edits judged against the session's *delegation contract*, **binding drift** (acting on a near-miss sibling of the file you actually resolved), context rot (requires a sustained wall-clock quiet stretch, so quick streaming output never trips it), and sustained **drift** — a CUSUM alarm whose threshold is calibrated by Monte-Carlo simulation to a **session-length** false-alarm budget, with a self-starting per-session baseline. Tool outcomes are classified three ways — *failed / ok / unknown* — from structured evidence (exit codes, failure hooks) before falling back to anchored error grammars, so `grep error logs/error.log` is not a failure and a lost response is not a success. Repeat NUDGEs from the same detector within a session are suppressed until `nudge_cooldown_seconds` has elapsed (hysteresis), so a detected condition does not nag the agent on every subsequent event.
2. **Tier 1 — LLM step scorer** (PRM-style, sliding window): scores each recent step on on-goal / justified / verified, names the step where the trajectory drifted, and returns actionable guidance. The judge wakes at natural checkpoints (prompt start, iteration end, session stop) and — mid-iteration — only when the CUSUM statistic climbs toward its alarm line, so the expensive reviewer stays asleep while the agent is healthy.

A judge's self-reported confidence is *not* taken at face value: `nudge_threshold` / `block_threshold` gate admission (and are re-fitted from observed outcomes by conformal risk control), `block` is only ever emitted for an agent that opted in, and guidance is truncated to `max_guidance_tokens`. Every session's own cost — events seen, judge calls, judge tokens, nudges, suppressions — is counted and readable with `shepherd stats`.

The design decisions above, and the papers that motivate them, are recorded in [RESEARCH_NOTES.md](RESEARCH_NOTES.md).

Recurring guidance the agent demonstrably followed is promoted into a short
"house rules" header injected at the start of each prompt instead of re-nagged
mid-run (`policy.rulebook_enabled`).

**Does it work?** `shepherd eval` runs an offline counterfactual benchmark: scripted healthy sessions plus a fault injector that breaks exactly one thing at a known step, scored for per-detector precision/recall, detection delay in steps, false-alarm rate, and supervision cost. It is deterministic, offline, and runs in CI.

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
  nudge_threshold: 0.6      # admission prior; replaced by CRC once outcomes accumulate
  block_threshold: 0.85
  max_guidance_tokens: 500
  fail_open: true
  nudge_cooldown_seconds: 300
  drift_enabled: true
  drift_target_fpr: 0.05    # budget over drift_horizon re-checks, not one window
  drift_horizon: 120
  drift_adaptive_baseline: true
  drift_watch_fraction: 0.6
  drift_unknown_weight: 0.75   # a lost response counts 75% as much as a failure
  review_clean_iterations: false  # don't spend the judge on a clean boundary
  binding_enabled: true
  scope_allow_globs: []
  scope_deny_globs: ["*.env", "*.pem", ".git/*"]
  risk_target_fpr: 0.05
  rulebook_enabled: true
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
shepherd stats claude <session-id>   # what it did, and what it cost
```

Installing an adapter never destroys your configuration: the previous
`settings.json` / `hooks.json` is copied to a timestamped `.bak` first, an
unparseable existing file is refused rather than overwritten, and a previous
QwenPaw plugin bundle is moved aside as `agent-shepherd.prev-<stamp>` instead of
being deleted.

## Measuring it

```bash
shepherd eval --seed 7 --sessions 8 --steps 60 --fail-under-f1 0.6
```

Injects known faults into scripted sessions and reports per-fault recall,
detection delay in steps, and the false-alarm rate on clean sessions. The
benchmark is the answer to "is the supervisor worth its own cost?" — run it
before and after changing any threshold.

## Tests

```bash
uv run pytest -q
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, adapter/detector conventions, commit and PR guidelines, and the
strict no-real-secrets rule. Quick start:

```bash
uv sync --all-extras && uv run python -m pytest -q && uv run ruff check .
```

Use `python -m pytest` (not a bare `pytest`): if the dev extras are not
installed, a bare `pytest` can resolve from outside this project's environment
and import a *stale installed copy* of the package — reporting green tests
against code that is no longer on disk.

Interactions in this project are governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT
## Running the daemon

```bash
shepherd start              # foreground
shepherd start-bg           # detached, writes ~/.shepherd/daemon.pid and daemon.log
shepherd status             # up/down -- down means supervision is silently failing open
shepherd stop               # SIGTERM the start-bg process
shepherd risk               # which admission thresholds are in force, prior or calibrated
```

For a machine that should always be supervising, install the shipped systemd
user unit (no root required). It is deliberately not enabled by the installer:
turning on a background process that injects text into your agents is your
decision, not a side effect of `pip install`.

```bash
install -Dm644 packaging/shepherd.service ~/.config/systemd/user/shepherd.service
systemctl --user daemon-reload && systemctl --user enable --now shepherd
```

The first start pays the drift detector's Monte-Carlo threshold fit (a few
seconds) before it binds; the result is cached in
`~/.shepherd/cusum_thresholds.json`, so later starts bind immediately. While the
daemon is down, adapters fail open -- `shepherd status` is the command that says so.

## Judge backends

The judge is any OpenAI-compatible chat endpoint. A local gateway needs no key
at all -- leave `SHEPHERD_JUDGE_API_KEY` unset and no `Authorization` header is
sent, and loopback/LAN endpoints bypass proxy environment:

```bash
export SHEPHERD_JUDGE_BASE_URL=http://127.0.0.1:19991/v1
export SHEPHERD_JUDGE_MODEL=<name from that endpoint's /v1/models>
```
