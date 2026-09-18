"""Tests for the Tiered policy engine."""

from __future__ import annotations

import time

from agent_shepherd.core.config import JudgeConfig, PolicyConfig, ShepherdConfig
from agent_shepherd.core.ledger import Ledger
from agent_shepherd.core.server import PolicyEngine
from agent_shepherd.core.types import Agent, AgentEvent, EventType, VerdictAction


class FakeScorer:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    def score(self, event, history):
        self.calls += 1
        return self.verdict


def test_policy_engine_wakes_scorer_on_iteration_end():
    from agent_shepherd.core.types import Verdict

    cfg = ShepherdConfig(
        judge=JudgeConfig(),
        policy=PolicyConfig(wake_every_n_steps=4),
        agents={},
    )
    engine = PolicyEngine(cfg, Ledger(root="/tmp/shepherd-test-ledger"))
    scorer = FakeScorer(
        Verdict(
            action=VerdictAction.NUDGE,
            reason="drift",
            guidance="fix it",
            confidence=0.8,
            detector="judge",
        )
    )
    engine.scorer = scorer
    event = AgentEvent(
        agent=Agent.QWENPAW,
        session_id="s1",
        event=EventType.ITERATION_END,
        ts=time.time(),
        iteration=1,
    )
    verdict = engine.process(event)
    assert verdict.action == VerdictAction.NUDGE
    assert scorer.calls == 1


def test_policy_engine_passes_when_detectors_are_silent_and_no_wake():
    from agent_shepherd.core.types import Verdict

    cfg = ShepherdConfig(
        judge=JudgeConfig(),
        policy=PolicyConfig(wake_every_n_steps=4),
        agents={},
    )
    engine = PolicyEngine(cfg, Ledger(root="/tmp/shepherd-test-ledger-2"))
    scorer = FakeScorer(
        Verdict(
            action=VerdictAction.NUDGE,
            reason="drift",
            guidance="fix it",
            confidence=0.8,
            detector="judge",
        )
    )
    engine.scorer = scorer
    event = AgentEvent(
        agent=Agent.QWENPAW,
        session_id="s2",
        event=EventType.REASONING,
        ts=time.time(),
        iteration=1,
        reasoning="thinking",
    )
    verdict = engine.process(event)
    assert verdict.action == VerdictAction.PASS
    assert scorer.calls == 0


def test_loop_detector_fires_on_third_identical_call_not_second():
    """Regression: the engine pushes the event before evaluating, so the
    pre-event window must exclude the current call — otherwise the loop
    detector double-counts and fires on the 2nd identical call instead of
    the 3rd (its documented threshold)."""
    from agent_shepherd.core.types import ToolCall

    cfg = ShepherdConfig(
        judge=JudgeConfig(),
        policy=PolicyConfig(wake_every_n_steps=4),
        agents={},
    )
    engine = PolicyEngine(cfg, Ledger(root="/tmp/shepherd-test-ledger-loop"))
    engine.scorer = FakeScorer(None)

    tool = ToolCall(name="shell", input={"cmd": "echo hi"})
    calls = []
    for i in range(4):
        event = AgentEvent(
            agent=Agent.QWENPAW,
            session_id="s4",
            event=EventType.TOOL_CALL,
            ts=time.time(),
            iteration=i,
            tool=tool,
        )
        calls.append(engine.process(event).action)

    # 1st and 2nd identical calls are below the threshold; 3rd and 4th fire.
    assert calls[0] == VerdictAction.PASS
    assert calls[1] == VerdictAction.PASS
    assert calls[2] == VerdictAction.NUDGE
    assert calls[3] == VerdictAction.NUDGE


def test_inline_nudge_is_parked_and_drained_at_gate_boundary():
    """A loop verdict that fires on a tool_call event must surface at the
    next iteration_end, not be silently returned to an adapter that can't
    act on it inline."""
    from agent_shepherd.core.types import ToolCall

    cfg = ShepherdConfig(
        judge=JudgeConfig(),
        policy=PolicyConfig(wake_every_n_steps=4),
        agents={},
    )
    engine = PolicyEngine(cfg, Ledger(root="/tmp/shepherd-test-ledger-3"))
    engine.scorer = FakeScorer(None)

    tool = ToolCall(name="shell", input={"cmd": "echo hi"})
    for i in range(3):
        event = AgentEvent(
            agent=Agent.QWENPAW,
            session_id="s3",
            event=EventType.TOOL_CALL,
            ts=time.time(),
            iteration=i,
            tool=tool,
        )
        verdict = engine.process(event)
        if i == 2:
            # Third identical call trips the loop detector inline.
            assert verdict.action == VerdictAction.NUDGE
            assert verdict.detector == "loop"

    # The gate boundary (iteration_end) drains the parked verdict without
    # calling the LLM scorer.
    gate = engine.process(
        AgentEvent(
            agent=Agent.QWENPAW,
            session_id="s3",
            event=EventType.ITERATION_END,
            ts=time.time(),
            iteration=3,
        )
    )
    assert gate.action == VerdictAction.NUDGE
    assert gate.detector == "loop"
    # Drained: the next gate is clean.
    assert engine.take_pending_verdict(Agent.QWENPAW, "s3") is None