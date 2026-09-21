"""Tests for the structured outcome classifier and the contract-based detectors.

These exist because the old failure signal was ``"error" in result.lower()``:
it could not tell a failing command from a successful one that mentioned the
word, and it read a lost response as a success.
"""

from __future__ import annotations

import time

from agent_shepherd.core.rules import signals
from agent_shepherd.core.rules.detectors import BindingDriftDetector, OffSpecDetector
from agent_shepherd.core.types import Agent, AgentEvent, EventType, ToolCall, VerdictAction


def result(text=None, **meta) -> AgentEvent:
    return AgentEvent(
        agent=Agent.CLAUDE,
        session_id="sig",
        event=EventType.TOOL_RESULT,
        ts=time.time(),
        tool=ToolCall(name="shell", input={"cmd": "x"}),
        tool_result=text,
        metadata=meta,
    )


def call(tool: str, **inputs) -> AgentEvent:
    return AgentEvent(
        agent=Agent.CLAUDE,
        session_id="sig",
        event=EventType.TOOL_CALL,
        ts=time.time(),
        tool=ToolCall(name=tool, input=inputs),
        prompt="Fix the parser in src/parser.py",
    )


def test_clean_output_that_merely_mentions_error_is_not_a_failure():
    assert signals.classify(result("grep -rn error logs/error.log -> 0 hits")).state == signals.OK
    assert signals.classify(result("tests/test_error_handling.py 12 passed in 0.30s")).state == signals.OK


def test_real_failure_grammars_are_caught():
    for text in (
        "error: could not compile `crate`\n",
        "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nZeroDivisionError",
        "2 failed, 40 passed in 1.10s",
        "Permission denied: '/etc/shadow'",
        "npm ERR! code ELIFECYCLE",
    ):
        assert signals.classify(result(text)).state == signals.FAILED, text


def test_structured_evidence_outranks_text():
    assert signals.classify(result("all good", exit_code=1)).state == signals.FAILED
    assert signals.classify(result("error: see above", exit_code=0)).state == signals.OK
    assert signals.classify(result("whatever", is_error=True)).state == signals.FAILED
    assert signals.classify(result("whatever", hook_event_name="PostToolUseFailure")).state == signals.FAILED


def test_lost_response_is_unknown_not_ok():
    """The non-atomic-failure case: no output means nothing was observed, which
    is neither success nor failure (arXiv 2608.02645)."""
    outcome = signals.classify(result(""))
    assert outcome.state == signals.UNKNOWN
    assert signals.signal_value(outcome) == 0.5
    assert signals.signal_value(signals.classify(result("nope", exit_code=2))) == 1.0
    assert signals.signal_value(signals.classify(result("fine"))) == 0.0


def test_offspec_uses_the_session_contract_not_just_prompt_words():
    detector = OffSpecDetector(nudge_unobserved=True)
    history = [
        call("read_file", path="src/lexer.py"),
        result("def tokenize(): ...", exit_code=0),
    ]
    history[0].prompt = "Fix the parser in src/parser.py"
    # A file the agent already read is in play, even though the goal never named it.
    assert detector.evaluate(call("edit_file", path="src/lexer.py"), history) is None
    # A file nobody named and nobody looked at is not.
    verdict = detector.evaluate(call("edit_file", path="vendor/other.py"), history)
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE
    assert "vendor/other.py" in verdict.reason


def test_offspec_deny_glob_blocks():
    detector = OffSpecDetector(deny_globs=["*.env"])
    verdict = detector.evaluate(call("write_file", path="prod.env"), [])
    assert verdict is not None
    assert verdict.action == VerdictAction.BLOCK


def test_offspec_is_silent_without_an_edit():
    detector = OffSpecDetector()
    assert detector.evaluate(call("read_file", path="anything.py"), []) is None


def test_binding_drift_catches_the_near_miss_sibling():
    """Right tool, wrong entity: the call succeeds, so nothing else sees it
    (arXiv 2607.18316)."""
    detector = BindingDriftDetector()
    history = [call("read_file", path="config.py")]
    verdict = detector.evaluate(call("edit_file", path="config.prod.py"), history)
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE
    assert "binding drift" in verdict.reason
    # Editing the file that was actually inspected is fine.
    assert detector.evaluate(call("edit_file", path="config.py"), history) is None
    # So is a genuinely different target.
    assert detector.evaluate(call("edit_file", path="docs/readme.md"), history) is None


def test_drift_guidance_adapts_to_unobservable_outcomes():
    """Telling the agent to "read the error output" is useless when nothing came
    back; that case needs a side-effect check instead."""
    from agent_shepherd.core.rules.detectors import CUSUMDriftDetector

    lost = [result("") for _ in range(20)]
    verdict = CUSUMDriftDetector().evaluate(lost[-1], lost[:-1])
    assert verdict is not None
    assert "nothing observable" in verdict.guidance
    assert "read the actual error output" not in verdict.guidance

    hard = [result("error: boom") for _ in range(20)]
    verdict = CUSUMDriftDetector().evaluate(hard[-1], hard[:-1])
    assert verdict is not None
    assert "read the actual error output" in verdict.guidance
