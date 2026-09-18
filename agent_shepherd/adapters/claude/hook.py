"""Claude Code hook thin-shell.

Reads a Claude Code hook payload from stdin, normalizes it, POSTs it to the
supervisor daemon, and writes the daemon's verdict back as Claude Code's
``hookSpecificOutput`` JSON on stdout.

Usage (from a Claude Code hook config):
    command: "shepherd-hook claude"
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


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Claude Code hook payload into the canonical event shape.

    Claude Code sends ``tool_response`` (not ``tool_output``) for the
    PostToolUse/PostToolUseFailure events.
    """
    event_name = str(payload.get("hook_event_name", "notification"))
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    event = "notification"
    if event_name == "PreToolUse":
        event = "tool_call"
    elif event_name in ("PostToolUse", "PostToolUseFailure"):
        event = "tool_result"
    elif event_name == "Stop":
        event = "iteration_end"
    elif event_name == "UserPromptSubmit":
        event = "prompt_submit"

    normalized: dict[str, Any] = {
        "agent": "claude",
        "session_id": str(payload.get("session_id", "unknown")),
        "event": event,
        "ts": time.time(),
        "iteration": int(payload.get("iteration", 0)),
        "tool": {"name": str(tool_name), "input": tool_input} if tool_name else None,
        "tool_result": payload.get("tool_response") or payload.get("tool_output"),
        "reasoning": payload.get("reasoning"),
        "prompt": payload.get("prompt"),
        "metadata": {"hook_event_name": event_name},
    }
    return normalized


def _response(verdict: dict[str, Any] | None, event_name: str = "") -> dict[str, Any]:
    """Translate a daemon verdict into Claude Code hook output.

    Nudges go through ``hookSpecificOutput.additionalContext`` (with the
    ``hookEventName`` Claude Code expects). Blocks use the top-level
    ``decision``/``reason`` pair.
    """
    if not verdict:
        return {}
    action = verdict.get("action", "pass")
    guidance = verdict.get("guidance")
    if action == "nudge" and guidance:
        hook_specific: dict[str, Any] = {"additionalContext": guidance}
        if event_name:
            hook_specific = {"hookEventName": event_name, **hook_specific}
        return {"hookSpecificOutput": hook_specific}
    if action in ("block", "escalate"):
        return {"decision": "block", "reason": verdict.get("reason", "blocked by supervisor")}
    return {}


def main() -> int:
    payload = json.loads(sys.stdin.read())
    event_name = str(payload.get("hook_event_name", ""))
    verdict = _post("/ingest/claude", _normalize(payload))
    sys.stdout.write(json.dumps(_response(verdict, event_name), ensure_ascii=False))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())