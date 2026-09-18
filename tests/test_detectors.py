"""Unit tests for the Tier 0 deterministic detectors."""

from __future__ import annotations

import time

from agent_shepherd.core.rules.detectors import (
    ContextRotDetector,
    CUSUMDriftDetector,
    LoopDetector,
    OffSpecDetector,
    RegressionDetector,
)
from agent_shepherd.core.types import Agent, AgentEvent, EventType, ToolCall, VerdictAction


def ev(
    *,
    event: EventType,
    tool: ToolCall | None = None,
    result: str | None = None,
    reasoning: str | None = None,
    prompt: str | None = None,
    iteration: int = 1,
    ts: float | None = None,
) -> AgentEvent:
    return AgentEvent(
        agent=Agent.QWENPAW,
        session_id="s1",
        event=event,
        ts=ts if ts is not None else time.time(),
        iteration=iteration,
        tool=tool,
        tool_result=result,
        reasoning=reasoning,
        prompt=prompt,
    )


def test_loop_detector_fires_on_repeated_tool_calls():
    detector = LoopDetector(window=8, threshold=3)
    tool = ToolCall(name="shell", input={"cmd": "echo hi"})
    history = [
        ev(event=EventType.TOOL_CALL, tool=tool, iteration=i)
        for i in range(1, 4)
    ]
    verdict = detector.evaluate(history[-1], history[:-1])
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE
    assert verdict.detector == "loop"


def test_loop_detector_ignores_different_tools():
    detector = LoopDetector(window=8, threshold=3)
    history = [
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="shell", input={"cmd": "echo a"}), iteration=1),
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="shell", input={"cmd": "echo b"}), iteration=2),
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="read", input={"path": "a.txt"}), iteration=3),
    ]
    assert detector.evaluate(history[-1], history[:-1]) is None


def test_regression_detector_fires_when_test_fails_after_passing():
    detector = RegressionDetector()
    history = [
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="pytest"), iteration=1),
        ev(event=EventType.TOOL_RESULT, tool=ToolCall(name="pytest"), result="passed", iteration=2),
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="pytest"), iteration=3),
    ]
    verdict = detector.evaluate(
        ev(event=EventType.TOOL_RESULT, tool=ToolCall(name="pytest"), result="failed", iteration=4),
        history,
    )
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE
    assert "regression" in verdict.reason


def test_offspec_detector_fires_on_unrelated_edits():
    detector = OffSpecDetector()
    current = ev(
        event=EventType.TOOL_CALL,
        tool=ToolCall(name="write_file", input={"path": "src/unrelated.py"}),
        prompt="Fix the bug in src/main.py",
        iteration=1,
    )
    verdict = detector.evaluate(current, [])
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE
    assert "off-spec" in verdict.reason


def test_context_rot_detector_fires_on_long_reasoning():
    detector = ContextRotDetector(max_reasoning_chars=100, max_consecutive_reasoning=4)
    base = 1_000_000.0
    history = [
        ev(event=EventType.REASONING, reasoning="x" * 30, iteration=i, ts=base + 30.0 * i)
        for i in range(1, 5)
    ]
    verdict = detector.evaluate(history[-1], history[:-1])
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE


def test_context_rot_stays_silent_on_quick_streaming():
    """Fast streaming deltas (seconds apart) are normal thinking, not rot:
    the quiet stretch has to be sustained in wall-clock time."""
    detector = ContextRotDetector(max_reasoning_chars=100, max_consecutive_reasoning=4)
    base = 1_000_000.0
    history = [
        ev(event=EventType.REASONING, reasoning="x" * 30, iteration=i, ts=base + 2.0 * i)
        for i in range(1, 5)
    ]
    assert detector.evaluate(history[-1], history[:-1]) is None


def test_context_rot_stays_silent_while_tools_are_making_progress():
    """Consecutive reasoning deltas are normal between tool calls; only flag
    true spinning (quiet agent), not streamed output interleaved with work."""
    detector = ContextRotDetector(max_reasoning_chars=100, max_consecutive_reasoning=4)
    history = [
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="shell", input={"cmd": "echo 1"}), iteration=1),
        ev(event=EventType.TOOL_RESULT, tool=ToolCall(name="shell", input={"cmd": "echo 1"}), result="ok", iteration=1),
        ev(event=EventType.REASONING, reasoning="x" * 30, iteration=2),
    ]
    verdict = detector.evaluate(history[-1], history[:-1])
    assert verdict is None


def test_loop_detector_fires_on_near_duplicate_calls():
    """Retrying the same command with slightly tweaked arguments is the more
    common real-world loop; near-duplicate matching catches it."""
    detector = LoopDetector(window=8, threshold=3)
    base = "pytest tests/test_foo.py -k bar"
    history = [
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="shell", input={"cmd": base}), iteration=1),
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="shell", input={"cmd": base + " --tb=short"}), iteration=2),
    ]
    current = ev(
        event=EventType.TOOL_CALL, tool=ToolCall(name="shell", input={"cmd": base + " --tb=short"}), iteration=3
    )
    verdict = detector.evaluate(current, history)
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE
    assert "near-identical" in verdict.reason


def test_drift_detector_stays_silent_on_healthy_stream_with_isolated_failures():
    """A healthy session with an occasional failing call must not alarm."""
    detector = CUSUMDriftDetector()
    history = []
    for i in range(40):
        result = "error: boom" if i in (5, 17) else "ok"
        event = ev(event=EventType.TOOL_RESULT, tool=ToolCall(name="shell"), result=result, iteration=i)
        assert detector.evaluate(event, history) is None
        history.append(event)


def test_drift_detector_fires_on_sustained_failures():
    detector = CUSUMDriftDetector()
    history = []
    fired_at = None
    for i in range(10):
        event = ev(event=EventType.TOOL_RESULT, tool=ToolCall(name="shell"), result="error: boom", iteration=i)
        if detector.evaluate(event, history) is not None:
            fired_at = i
            break
        history.append(event)
    # A sustained run is caught within one window, well before it fills up.
    assert fired_at is not None
    assert fired_at < detector.window


def test_drift_detector_watch_level_tracks_consecutive_failures():
    """The soft trigger: one isolated failure stays below the watch fraction
    (default 0.6); two consecutive push the statistic to the alarm line."""
    detector = CUSUMDriftDetector()
    bad = ev(event=EventType.TOOL_RESULT, tool=ToolCall(name="shell"), result="error: x")
    one = detector.watch_level(bad, [])
    two = detector.watch_level(bad, [bad])
    assert 0 < one < 0.6 <= two
    # The calibrated threshold keeps false alarms within the 5% budget.
    assert detector.threshold > 0.1