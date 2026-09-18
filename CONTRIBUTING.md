# Contributing to agent-shepherd

Thanks for your interest in agent-shepherd! This document covers the dev setup,
the conventions this codebase follows, and the rules for submitting contributions.

## Legal

By submitting a contribution you agree that it is licensed under the project's
[MIT License](LICENSE), the same as the rest of the codebase.

## Development setup

Prerequisites: Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/CallMeHFK/agent-shepherd.git
cd agent-shepherd
uv sync --all-extras          # creates .venv, installs runtime + dev deps (pytest, ruff)
```

Run the same checks as CI:

```bash
uv run pytest -q
uv run ruff check .
```

Optional: run a local daemon for live testing:

```bash
uv run shepherd start
```

The daemon listens on `127.0.0.1:4890` and auto-generates a starter config at
`~/.shepherd/config.yaml` on first start (see the README for the full format).
The judge API key is read from the `SHEPHERD_JUDGE_API_KEY` environment variable
or the `api_key` field of that config.

## Repository layout

- `agent_shepherd/core/` — the supervisor daemon and policy engine
  - `rules/detectors.py` — Tier 0 deterministic detectors (loop, near-duplicate loop, regression, off-spec, context rot, CUSUM drift)
  - `judge/` — Tier 1 LLM step scorer (OpenAI-compatible client, prompts)
  - `server.py` — HTTP API and the ingest → policy → verdict loop
  - `ledger.py` — append-only per-session JSONL ledger
  - `config.py` — config loading and the `${VAR}` env expansion
- `agent_shepherd/adapters/` — one package per supported agent
  - `claude/` — Claude Code `~/.claude/settings.json` hooks + `shepherd-hook claude` thin shell
  - `codex/` — Codex `~/.codex/hooks.json` (MatcherGroup schema) + `shepherd-hook codex` thin shell
  - `qwenpaw/` — in-process plugin (middleware + stop gate) + REST/SSE fallback client
  - `hook.py` — shared `shepherd-hook` CLI entry point
- `tests/` — pytest suite (offline; no real network calls or API keys)
- `.github/workflows/ci.yml` — CI: `uv sync --all-extras`, `pytest`, `ruff check`

## Adding a Tier 0 detector

Detectors are zero-cost deterministic rules. Most simply return a verdict; the
CUSUM drift detector additionally exposes a `watch_level` the policy engine
uses as a *soft* trigger to wake the LLM judge early, without raising a full
verdict. Conventions:

- Put the class in `agent_shepherd/core/rules/detectors.py` and implement
  `evaluate(event, history) -> Verdict | None`.
- **`history` excludes the current event** — the server snapshots the state
  *before* pushing the event. Count the current event yourself (e.g.
  `count = 1 + sum(...)`); a detector that relies on the current event being in
  `history` will fire one step too late.
- Keep detectors synchronous, pure, and stateless. Per-session state lives in the
  server, never in the detector. (The CUSUM detector's alarm threshold is
  calibrated once in `__init__` from seeded Monte-Carlo simulation and memoized —
  evaluation itself stays a pure function of `(event, history)`.)
- If a detector exposes a soft trigger, wire its `watch_level` into
  `PolicyEngine._wake` and gate it behind a `PolicyConfig` flag so it can be
  disabled.
- Add unit tests in `tests/test_detectors.py`; at minimum cover "fires on the Nth
  occurrence and not on the N-1th".

## Adding a new agent adapter

An adapter has two halves:

1. **Normalization** — map the agent's raw payload to shepherd events
   (`prompt_submit`, `tool_call`, `tool_result`, `iteration_end`, …). Verify field
   names against the actual agent version before assuming: both Claude Code and
   Codex send `tool_response` (not `tool_output`), for example.
2. **Response envelope** — map verdicts to the agent's stdout contract. For
   hook-based agents: nudges go out as `{"hookSpecificOutput": {"hookEventName": …,
   "additionalContext": …}}` (include `hookEventName` where the agent requires it),
   blocks as a top-level `{"decision": "block", "reason": …}`. Check the agent's
   own validation code for the exact allowed shape before shipping.

If the agent's hooks come from a config file, provide an `install.py` that is
**idempotent** (re-running must not duplicate entries) and removes legacy keys so a
stale file cannot keep failing to parse.

## Tests

- Every bug fix must ship with a regression test (reproduce the failure first, then fix it).
- The suite must run offline: mock or fake HTTP, never a real endpoint or API key.
- CI must be green before requesting review.

## Commit & pull request guidelines

- Branch names: `fix/<topic>`, `feat/<topic>`, `docs/<topic>`.
- Commit messages: imperative mood, one-line summary, body for the why. Keep one
  logical change per commit.
- PRs: explain what and why, name the affected adapter(s), and describe how you
  tested. Use the provided template.
- Keep diffs small. For large refactors or new architecture, open an issue first
  and agree on the direction before writing code.

## Secrets (strict rule)

- **Never commit real API keys** — not in code, config samples, docs, tests, or
  commit messages.
- Use the `${SHEPHERD_JUDGE_API_KEY}` placeholder in config and dummy values (e.g.
  `api_key="k"`) in tests.
- All runtime state lives outside the repository and must never be committed:
  `~/.shepherd/`, `~/.codex/`, `~/.claude/`, `.env`, etc. (see `.gitignore`).

## Code of conduct

All contributions and interactions must follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Where to ask

- Questions / bugs: GitHub Issues (`bug` label)
- Ideas / design: GitHub Issues (`enhancement` label) — or open a draft PR
