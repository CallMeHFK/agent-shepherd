"""Tests for the judge client and verdict parsing."""

from __future__ import annotations

from agent_shepherd.core.judge.client import JudgeClient
from agent_shepherd.core.types import VerdictAction


def test_parse_verdict_parses_strict_json():
    client = JudgeClient(base_url="http://example.invalid", model="m", api_key="k")
    verdict = client.parse_verdict(
        '{"action": "nudge", "reason": "drift", "guidance": "fix it", "confidence": 0.8}'
    )
    assert verdict.action == VerdictAction.NUDGE
    assert verdict.guidance == "fix it"
    assert verdict.confidence == 0.8


def test_parse_verdict_wraps_code_fences():
    client = JudgeClient(base_url="http://example.invalid", model="m", api_key="k")
    verdict = client.parse_verdict(
        '```json\n{"action": "block", "reason": "danger", "confidence": 1.0}\n```'
    )
    assert verdict.action == VerdictAction.BLOCK
    assert verdict.confidence == 1.0


def test_parse_verdict_falls_back_to_conservative_nudge():
    client = JudgeClient(base_url="http://example.invalid", model="m", api_key="k")
    verdict = client.parse_verdict("not json at all")
    assert verdict.action == VerdictAction.NUDGE
    assert verdict.detector == "judge"

def test_parse_verdict_folds_verbal_critiques_into_the_reason():
    """Verbal process supervision (arXiv 2503.18494): the useful output of a
    step judge is a sentence about what the step assumed, not a scalar."""
    client = JudgeClient(base_url="http://x", model="m", api_key="k")
    verdict = client.parse_verdict(
        '{"action": "nudge", "reason": "step 4 drifted", "guidance": "check the diff", '
        '"confidence": 0.8, "critique": ['
        '{"step": 4, "says": "assumed the file was the one it read"}, '
        '{"step": 6, "says": "retried without checking the side effect"}, '
        '{"step": 7, "says": "third worst"}, '
        '{"step": 9, "says": "fourth worst, never reaches the agent"}]}'
    )
    assert "step 4: assumed the file was the one it read" in verdict.reason
    assert "step 6: retried without checking the side effect" in verdict.reason
    assert "step 7: third worst" in verdict.reason
    assert "step 9" not in verdict.reason, "capped at the three worst steps"


def test_parse_verdict_tolerates_missing_or_malformed_critique():
    client = JudgeClient(base_url="http://x", model="m", api_key="k")
    for payload in (
        '{"action": "nudge", "reason": "drift", "guidance": "g", "confidence": 0.7}',
        '{"action": "nudge", "reason": "drift", "critique": "not a list", "confidence": 0.7}',
        '{"action": "nudge", "reason": "drift", "critique": [{"nope": 1}], "confidence": 0.7}',
    ):
        assert client.parse_verdict(payload).reason.startswith("drift")
