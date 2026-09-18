"""Tests for the append-only audit ledger."""

from __future__ import annotations

import time

from agent_shepherd.core.ledger import Ledger
from agent_shepherd.core.types import Agent, AgentEvent, EventType, ToolCall, Verdict, VerdictAction


def test_ledger_appends_and_reads_records(tmp_path):
    ledger = Ledger(root=tmp_path)
    event = AgentEvent(
        agent=Agent.QWENPAW,
        session_id="s1",
        event=EventType.TOOL_CALL,
        ts=time.time(),
        tool=ToolCall(name="shell", input={"cmd": "echo hi"}),
    )
    ledger.record_event(event)
    records = ledger.recent(Agent.QWENPAW, "s1")
    assert len(records) == 1
    assert records[0]["type"] == "event"
    assert records[0]["tool"]["name"] == "shell"


def test_ledger_records_verdicts(tmp_path):
    ledger = Ledger(root=tmp_path)
    verdict = Verdict(
        action=VerdictAction.NUDGE,
        reason="loop detected",
        guidance="stop looping",
        confidence=0.9,
        detector="loop",
    )
    ledger.record_verdict(Agent.QWENPAW, "s1", verdict, {"window": 3})
    records = ledger.recent(Agent.QWENPAW, "s1")
    assert records[0]["type"] == "verdict"
    assert records[0]["action"] == "nudge"
    assert records[0]["context"]["window"] == 3