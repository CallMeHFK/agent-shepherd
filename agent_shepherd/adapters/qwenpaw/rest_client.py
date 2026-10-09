"""QwenPaw REST polling adapter (zero-install fallback).

When the in-process plugin is not available, this client observes QwenPaw via
its local REST API (``127.0.0.1:19999``) and injects guidance through the Tool
Guard approval machinery:

- ``GET /api/tool-calls/{session_id}`` — live tool calls
- ``GET /api/approval/list`` — pending approvals
- ``POST /api/approval/{approve,deny}`` — resolve them with a verdict

The client is intentionally conservative: it only intervenes on explicit
approvals (STRICT/SMART), so it never wedges an idle agent.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlencode

import httpx

logger = logging.getLogger("agent_shepherd.qwenpaw.rest")


def _tool_call_event(
    session_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    tool_result: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The canonical wire shape. Built by hand (not from ``AgentEvent.__dict__``)
    so the payload is JSON-serializable: dataclasses and enums are not."""
    return {
        "agent": "qwenpaw",
        "session_id": session_id,
        "event": "tool_call",
        "ts": time.time(),
        "iteration": 0,
        "tool": {"name": tool_name, "input": tool_input},
        "tool_result": tool_result,
        "reasoning": None,
        "prompt": None,
        "metadata": metadata or {},
    }


class QwenPawRestClient:
    def __init__(self, base_url: str = "http://127.0.0.1:19999", daemon_url: str = "http://127.0.0.1:4890"):
        self.base_url = base_url.rstrip("/")
        self.daemon_url = daemon_url.rstrip("/")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                resp = client.post(f"{self.daemon_url}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError, ImportError):
            return None

    def _qwen_get(self, path: str) -> dict[str, Any] | None:
        try:
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                resp = client.get(f"{self.base_url}{path}")
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError, ImportError):
            return None

    def poll(self, session_id: str) -> None:
        """One poll cycle: observe tool calls, resolve pending approvals."""
        calls = self._qwen_get(f"/api/tool-calls/{session_id}") or {}
        approvals = self._qwen_get(f"/api/approval/list?session_id={session_id}") or {}

        for call in calls.get("tool_calls", []):
            verdict = self._post(
                "/ingest/qwenpaw",
                _tool_call_event(
                    session_id,
                    str(call.get("tool_name", "unknown")),
                    dict(call.get("input", {})),
                ),
            )
            if verdict and verdict.get("action") == "block":
                self._qwen_get(
                    "/api/approval/deny?"
                    + urlencode({"request_id": call.get("request_id"), "session_id": session_id})
                )

        for approval in approvals.get("pending_approvals", []):
            request_id = approval.get("request_id")
            if not request_id:
                continue
            verdict = self._post(
                "/ingest/qwenpaw",
                _tool_call_event(
                    session_id,
                    str(approval.get("tool_name", "unknown")),
                    {},
                    tool_result=approval.get("reasoning"),
                    metadata={"phase": "approval"},
                ),
            )
            if not verdict:
                continue
            action = verdict.get("action", "pass")
            if action == "block":
                self._qwen_get(
                    "/api/approval/deny?" + urlencode({"request_id": request_id, "session_id": session_id})
                )
            elif action == "nudge" and verdict.get("guidance"):
                self._qwen_get(
                    "/api/approval/approve?"
                    + urlencode(
                        {
                            "request_id": request_id,
                            "session_id": session_id,
                            "reason": verdict["guidance"],
                        }
                    )
                )

    def run_forever(self, session_id: str, interval: float = 2.0) -> None:
        while True:
            try:
                self.poll(session_id)
            except (httpx.HTTPError, ValueError, ImportError) as exc:
                logger.warning("QwenPaw REST poll failed: %s", exc)
            time.sleep(interval)