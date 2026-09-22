"""Unit tests for the Tier 0 deterministic detectors."""

from __future__ import annotations

import time

from agent_shepherd.core.rules.detectors import (
    BindingDriftDetector,
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
        ev(
            event=EventType.TOOL_RESULT,
            tool=ToolCall(name="pytest"),
            result="42 passed in 0.87s",
            iteration=2,
        ),
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="pytest"), iteration=3),
    ]
    verdict = detector.evaluate(
        ev(
            event=EventType.TOOL_RESULT,
            tool=ToolCall(name="pytest"),
            result="===== FAILURES =====\nFAILED tests/test_x.py::test_y - assert 0",
            iteration=4,
        ),
        history,
    )
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE
    assert "regression" in verdict.reason


def test_regression_detector_needs_real_evidence_in_both_directions():
    """The old substring signal called ``grep error logs/error.log`` a failure.

    Fixture text below is genuine tool output: a passing run, then a run whose
    only "error" mention is a filename.
    """
    detector = RegressionDetector()
    history = [
        ev(
            event=EventType.TOOL_RESULT,
            tool=ToolCall(name="pytest"),
            result="42 passed in 0.87s",
            iteration=1,
        ),
    ]
    quiet = ev(
        event=EventType.TOOL_RESULT,
        tool=ToolCall(name="pytest"),
        result="tests/test_error_handling.py:12: note: uses the word error liberally",
        iteration=2,
    )
    assert detector.evaluate(quiet, history) is None


def test_regression_detector_reads_structured_exit_code():
    """An exit code stated as a field outranks whatever the text says."""
    detector = RegressionDetector()
    history = [
        ev(
            event=EventType.TOOL_RESULT,
            tool=ToolCall(name="pytest"),
            result="42 passed in 0.87s",
            iteration=1,
        ),
    ]
    current = AgentEvent(
        agent=Agent.QWENPAW,
        session_id="s1",
        event=EventType.TOOL_RESULT,
        ts=time.time(),
        tool=ToolCall(name="pytest"),
        tool_result="no failure grammar in this text at all",
        metadata={"exit_code": 1},
    )
    assert detector.evaluate(current, history) is not None


def test_offspec_detector_fires_on_unrelated_edits():
    # The unobserved-path nudge is opt-in: measured at ~1 false positive per
    # healthy session, so the default is deny-globs only.
    detector = OffSpecDetector(nudge_unobserved=True)
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
    """The soft trigger is onset-shaped, not state-shaped.

    The alarm line is a calibrated output (a Monte-Carlo fit against a
    session-length false-alarm budget over a graded null model), so the exact
    number of failures that crosses it moves whenever the calibration changes.
    Pinning the constant would only make this test churn; what must not change
    is the shape: monotone in the length of the failing run, silent at one
    isolated failure, and reaching the alarm within a single window.
    """
    detector = CUSUMDriftDetector()
    bad = ev(event=EventType.TOOL_RESULT, tool=ToolCall(name="shell"), result="error: x")
    levels = [detector.watch_level(bad, [bad] * n) for n in range(detector.window)]
    assert levels == sorted(levels), "monotone in the length of the failing run"
    assert 0 < levels[0] < 0.6, "one failure stays below the watch fraction"
    assert levels[-1] >= 1.0, "a window full of failures reaches the hard alarm"
    crossing = next((n for n, level in enumerate(levels) if level >= 0.6), None)
    assert crossing is not None and crossing >= 1, "the soft trigger sits above a single failure"
    # The calibrated threshold keeps false alarms within the session budget.
    assert detector.threshold > 1.0


def test_drift_detector_baseline_comes_from_the_pre_alarm_reference_period():
    """Self-normalization trap: if the reference mean is taken from the window
    that is alarming, a run of failures pushes the baseline to 1.0 and the
    detector can never alarm again."""
    detector = CUSUMDriftDetector()
    healthy = [
        ev(event=EventType.TOOL_RESULT, tool=ToolCall(name="shell"), result="done", iteration=i)
        for i in range(10)
    ]
    failing = [
        ev(event=EventType.TOOL_RESULT, tool=ToolCall(name="shell"), result="error: boom", iteration=i)
        for i in range(10, 20)
    ]
    assert detector.evaluate(failing[-1], healthy + failing[:-1]) is not None

# --- BindingDriftDetector: the near-miss sibling of a resolved path ------


def _binding_case(inspected: str, target: str):
    detector = BindingDriftDetector()
    history = [
        ev(event=EventType.TOOL_CALL, tool=ToolCall(name="read", input={"path": inspected}), iteration=1)
    ]
    current = ev(
        event=EventType.TOOL_CALL, tool=ToolCall(name="edit_file", input={"path": target}), iteration=2
    )
    return detector.evaluate(current, history)


def test_binding_drift_quotes_paths_with_their_original_case():
    """Regression (seen live 2026-09-22): the string used for comparison was
    also the string displayed, so the guidance named a lowercased path that
    does not exist on a case-sensitive filesystem -- an agent told to re-check
    a file it cannot find stops trusting the nudge."""
    inspected = "/home/leo/out/patent-disclosure-MTS-ND-v2/专利交底书_MTS-ND_v2.0.md"
    target = "/home/leo/out/patent-disclosure-MTS-ND-v2/专利交底书_MTS-ND_v2.0.bak"
    verdict = _binding_case(inspected, target)
    assert verdict is not None and verdict.detector == "binding"
    assert inspected in verdict.reason
    assert inspected in verdict.guidance


def test_binding_drift_treats_a_case_only_difference_as_the_same_file():
    """Matching stays case-insensitive: a path differing only in case is the
    file already read, not a near-miss sibling."""
    assert _binding_case("/srv/app/config/Settings.yaml", "/srv/app/config/settings.yaml") is None


def test_binding_drift_stays_silent_on_different_names_in_one_directory():
    """The directory is shared by every file a session touches, so folding it
    into the similarity made ordinary multi-file work look like a mis-binding:
    reading detectors.py and editing signals.py scored 0.71 and nudged. Seen
    live 2026-09-22, and baked into the benchmark's own binding_drift case."""
    assert (
        _binding_case("/repo/core/rules/detectors.py", "/repo/core/rules/signals.py") is None
    )


def test_binding_drift_stays_silent_on_unrelated_non_ascii_names():
    """The old tokenizer split on ``[^a-z0-9]+``, which deleted every CJK
    character: two unrelated Chinese-named documents in one directory compared
    as identical (similarity 1.0) and always nudged. Any project with
    non-ASCII filenames was guaranteed false alarms."""
    assert _binding_case("/out/专利交底书_v2.md", "/out/客户需求说明书_v2.md") is None


def test_binding_drift_fires_on_the_same_name_under_a_different_directory():
    """The other reachable form of right-file-wrong-place: reading the judge's
    risk.py and editing the rules one."""
    verdict = _binding_case(
        "agent_shepherd/core/judge/risk.py", "agent_shepherd/core/rules/risk.py"
    )
    assert verdict is not None and verdict.detector == "binding"


def test_binding_drift_fires_on_the_canonical_sibling():
    """arXiv 2607.18316's own example: ``cat config.py`` then edit
    ``config.py.bak``."""
    verdict = _binding_case("/srv/app/config.py", "/srv/app/config.py.bak")
    assert verdict is not None and verdict.detector == "binding"


def test_binding_drift_stays_silent_between_a_module_and_its_own_tests():
    """A filename overlap is only a mis-binding risk between *interchangeable*
    artifacts. Reading the implementation and editing its test is ordinary work
    -- and the two names share every token but the ``test_`` marker, so a pure
    name-similarity rule flags nearly every TDD session."""
    assert _binding_case("/repo/core/rules/detectors.py", "/repo/tests/test_detectors.py") is None
