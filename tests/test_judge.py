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