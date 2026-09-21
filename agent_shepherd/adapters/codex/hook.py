"""Codex hook thin-shell.

Reads a Codex hook payload from stdin, normalizes it, POSTs it to the
supervisor daemon, and writes the verdict back as Codex hook JSON on stdout.

Usage (from ``~/.codex/hooks.json``):
    { "command": "shepherd-hook codex", "matcher": "*" }
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import httpx


def _daemon_url() -> str:
    return os.environ.get("SHEPHERD_DAEMON_URL", "http://127.0.0.1:4890").rstrip("/")


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{_daemon_url()}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError):
        return None


def _flatten_result(raw: Any) -> tuple[Any, dict[str, Any]]:
    """Split a Codex tool response into display text + structured evidence.

    Same reasoning as the Claude adapter: an exit code that arrives as a field
    is evidence, and flattening it into text before the daemon sees it forces
    the detectors back to substring guessing.
    """
    extra: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in ("exit_code", "returncode", "is_error", "success"):
            if key in raw:
                extra[key] = raw[key]
        parts = [str(raw[k]) for k in ("stdout", "stderr", "output", "content") if raw.get(k) is not None]
        text = "\n".join(parts) if parts else json.dumps(raw, ensure_ascii=False)[:4000]
        return text, extra
    return raw, extra


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Codex hook payload into the canonical event shape.

    Codex sends ``tool_response`` (not ``tool_output``) for PostToolUse and has
    no numeric ``iteration`` field (it uses an opaque ``turn_id`` instead).
    """
    event_name = str(payload.get("hook_event_name", "notification"))
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if event_name == "PreToolUse":
        event = "tool_call"
    elif event_name == "PostToolUse":
        event = "tool_result"
    elif event_name == "Stop":
        event = "iteration_end"
    elif event_name == "UserPromptSubmit":
        event = "prompt_submit"
    else:
        event = "notification"

    result, evidence = _flatten_result(payload.get("tool_response") or payload.get("tool_output"))
    return {
        "agent": "codex",
        "session_id": str(payload.get("session_id", "unknown")),
        "event": event,
        "ts": time.time(),
        "iteration": int(payload.get("iteration", 0)),
        "tool": {"name": str(tool_name), "input": tool_input} if tool_name else None,
        "tool_result": result,
        "reasoning": payload.get("reasoning"),
        "prompt": payload.get("prompt"),
        "metadata": {"hook_event_name": event_name, "turn_id": payload.get("turn_id"), **evidence},
    }


def _response(verdict: dict[str, Any] | None, event_name: str = "") -> dict[str, Any]:
    """Translate a daemon verdict into the Codex hook stdout envelope.

    Nudges are injected via ``hookSpecificOutput.additionalContext`` (only
    accepted on tool/prompt events — ``Stop`` has no such field). Blocks use the
    top-level ``decision``/``reason`` pair.
    """
    if not verdict:
        return {}
    action = verdict.get("action", "pass")
    guidance = verdict.get("guidance")
    if action == "nudge" and guidance:
        if event_name in ("PreToolUse", "PostToolUse", "UserPromptSubmit"):
            return {
                "hookSpecificOutput": {
                    "hookEventName": event_name,
                    "additionalContext": guidance,
                }
            }
        return {}
    if action in ("block", "escalate"):
        return {"decision": "block", "reason": verdict.get("reason", "blocked by supervisor")}
    return {}


def main() -> int:
    payload = json.loads(sys.stdin.read())
    event_name = str(payload.get("hook_event_name", ""))
    verdict = _post("/ingest/codex", _normalize(payload))
    sys.stdout.write(json.dumps(_response(verdict, event_name), ensure_ascii=False))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())