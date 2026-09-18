"""Unit tests for the Tier 0 deterministic detectors."""

from __future__ import annotations

import time

from agent_shepherd.core.rules.detectors import (
    ContextRotDetector,
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
) -> AgentEvent:
    return AgentEvent(
        agent=Agent.QWENPAW,
        session_id="s1",
        event=event,
        ts=time.time(),
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
    history = [
        ev(event=EventType.REASONING, reasoning="x" * 30, iteration=i)
        for i in range(1, 5)
    ]
    verdict = detector.evaluate(history[-1], history[:-1])
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE


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