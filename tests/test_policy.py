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
        policy=PolicyConfig(),
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
        policy=PolicyConfig(),
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
        policy=PolicyConfig(),
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
        policy=PolicyConfig(),
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


def _engine_and_scorer(root: str, policy=None):
    from agent_shepherd.core.types import Verdict

    cfg = ShepherdConfig(
        judge=JudgeConfig(),
        policy=policy or PolicyConfig(),
        agents={},
    )
    engine = PolicyEngine(cfg, Ledger(root=root))
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
    return engine, scorer


def _result(session_id: str, result: str, iteration: int = 1):
    return AgentEvent(
        agent=Agent.CLAUDE,
        session_id=session_id,
        event=EventType.TOOL_RESULT,
        ts=time.time(),
        iteration=iteration,
        tool=None,
        tool_result=result,
    )


def test_tool_call_does_not_wake_scorer():
    """Anti-saturation gate: a bare tool call no longer wakes the judge."""
    engine, scorer = _engine_and_scorer("/tmp/shepherd-test-wake-call")
    event = AgentEvent(
        agent=Agent.CLAUDE,
        session_id="wk1",
        event=EventType.TOOL_CALL,
        ts=time.time(),
        iteration=1,
        tool=None,
    )
    verdict = engine.process(event)
    assert verdict.action == VerdictAction.PASS
    assert scorer.calls == 0


def test_tool_result_wakes_scorer_only_when_drift_is_rising():
    """One isolated failure stays below the watch fraction (silent); two
    consecutive failures push the CUSUM to the alarm line and softly wake the
    judge before a hard Tier 0 alarm."""
    engine, scorer = _engine_and_scorer("/tmp/shepherd-test-wake-drift")

    assert engine.process(_result("wd1", "ok")).action == VerdictAction.PASS
    assert scorer.calls == 0
    # First failure: watch level ~0.5, below the 0.6 soft trigger.
    assert engine.process(_result("wd1", "error: boom")).action == VerdictAction.PASS
    assert scorer.calls == 0
    # Second consecutive failure: watch level ~1.0, judge wakes.
    verdict = engine.process(_result("wd1", "error: again"))
    assert verdict.action == VerdictAction.NUDGE
    assert verdict.detector == "judge"
    assert scorer.calls == 1


def test_drift_hard_alarm_preempts_scorer():
    """A sustained run of failures triggers the Tier 0 drift detector directly,
    with no LLM call at all (the soft trigger is disabled so the hard alarm
    is the only drift signal)."""
    engine, scorer = _engine_and_scorer(
        "/tmp/shepherd-test-wake-drift2",
        policy=PolicyConfig(drift_watch_fraction=1.0),
    )
    verdict = None
    for i in range(4):
        verdict = engine.process(_result("wd2", "error: boom", iteration=i))
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE
    assert verdict.detector == "drift"
    assert scorer.calls == 0


def test_soft_wake_disabled_when_drift_detector_is_off():
    """With drift_enabled=False there is no CUSUM statistic, so a failing tool
    result no longer wakes the judge (only natural checkpoints do)."""
    engine, scorer = _engine_and_scorer(
        "/tmp/shepherd-test-wake-nodrift",
        policy=PolicyConfig(drift_enabled=False),
    )
    assert all(d.name != "drift" for d in engine.detectors)
    assert engine.process(_result("wn1", "error: boom")).action == VerdictAction.PASS
    assert scorer.calls == 0
    # The natural checkpoint still wakes the judge.
    gate = engine.process(
        AgentEvent(
            agent=Agent.CLAUDE,
            session_id="wn1",
            event=EventType.ITERATION_END,
            ts=time.time(),
            iteration=1,
        )
    )
    assert gate.action == VerdictAction.NUDGE
    assert scorer.calls == 1