"""QwenPaw REST/SSE adapter (zero-install fallback).

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

import httpx

from ...core.types import AgentEvent, EventType, ToolCall

logger = logging.getLogger("agent_shepherd.qwenpaw.rest")


class QwenPawRestClient:
    def __init__(self, base_url: str = "http://127.0.0.1:19999", daemon_url: str = "http://127.0.0.1:4890"):
        self.base_url = base_url.rstrip("/")
        self.daemon_url = daemon_url.rstrip("/")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(f"{self.daemon_url}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError):
            return None

    def _qwen_get(self, path: str) -> dict[str, Any] | None:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}{path}")
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError):
            return None

    def poll(self, agent_id: str, session_id: str) -> None:
        """One poll cycle: observe tool calls, resolve pending approvals."""
        calls = self._qwen_get(f"/api/tool-calls/{session_id}") or {}
        approvals = self._qwen_get(f"/api/approval/list?session_id={session_id}") or {}

        for call in calls.get("tool_calls", []):
            event = AgentEvent(
                agent="qwenpaw",  # type: ignore[arg-type]
                session_id=session_id,
                event=EventType.TOOL_CALL,
                ts=time.time(),
                tool=ToolCall(name=str(call.get("tool_name", "unknown")), input=dict(call.get("input", {}))),
            )
            verdict = self._post("/ingest/qwenpaw", event.__dict__)
            if verdict and verdict.get("action") == "block":
                self._qwen_get(f"/api/approval/deny?request_id={call.get('request_id')}&session_id={session_id}")

        for approval in approvals.get("pending_approvals", []):
            request_id = approval.get("request_id")
            if not request_id:
                continue
            verdict = self._post(
                "/ingest/qwenpaw",
                {
                    "agent": "qwenpaw",
                    "session_id": session_id,
                    "event": "tool_call",
                    "ts": time.time(),
                    "iteration": 0,
                    "tool": {"name": approval.get("tool_name", "unknown"), "input": {}},
                    "tool_result": approval.get("reasoning"),
                    "reasoning": None,
                    "prompt": None,
                    "metadata": {"phase": "approval"},
                },
            )
            if not verdict:
                continue
            action = verdict.get("action", "pass")
            if action == "block":
                self._qwen_get(f"/api/approval/deny?request_id={request_id}&session_id={session_id}")
            elif action == "nudge" and verdict.get("guidance"):
                self._qwen_get(
                    f"/api/approval/approve?request_id={request_id}&session_id={session_id}&reason={verdict['guidance']}"
                )

    def run_forever(self, agent_id: str, session_id: str, interval: float = 2.0) -> None:
        while True:
            try:
                self.poll(agent_id, session_id)
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("QwenPaw REST poll failed: %s", exc)
            time.sleep(interval)