"""Tests for the QwenPaw zero-install REST client."""

from __future__ import annotations

import json

from agent_shepherd.adapters.qwenpaw.rest_client import QwenPawRestClient


def test_poll_posts_json_serializable_events(monkeypatch):
    """event.__dict__ carried the ToolCall dataclass into json.dumps, which
    raised TypeError outside _post's exception list and crashed poll()."""
    client = QwenPawRestClient()

    def fake_get(path: str):
        if "tool-calls" in path:
            return {"tool_calls": [{"tool_name": "shell", "input": {"cmd": "ls"}, "request_id": "r1"}]}
        return {}

    sent: list[dict] = []

    def fake_post(path: str, payload: dict):
        json.dumps(payload)  # the wire contract: payload must serialize
        sent.append(payload)
        return {"action": "pass"}

    monkeypatch.setattr(client, "_qwen_get", fake_get)
    monkeypatch.setattr(client, "_post", fake_post)

    client.poll("s1")
    assert sent, "the observed tool call never reached the daemon"


def test_poll_urlencodes_guidance_in_approval_reason(monkeypatch):
    """Guidance interpolated raw into a query string breaks on '&', '=' and
    spaces — the approval reason must be percent-encoded."""
    client = QwenPawRestClient()
    gets: list[str] = []

    def fake_get(path: str):
        gets.append(path)
        if "approval/list" in path:
            return {"pending_approvals": [{"request_id": "r9", "tool_name": "shell", "reasoning": "x"}]}
        return {}

    def fake_post(path: str, payload: dict):
        return {"action": "nudge", "guidance": "stop & read the error = first"}

    monkeypatch.setattr(client, "_qwen_get", fake_get)
    monkeypatch.setattr(client, "_post", fake_post)

    client.poll("s1")
    approve = [p for p in gets if "approval/approve" in p]
    assert approve, "a nudge with guidance should approve-with-reason"
    reason = approve[0].split("reason=", 1)[1]
    assert "&" not in reason and " " not in reason
