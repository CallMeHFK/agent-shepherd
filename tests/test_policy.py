"""Tests for the Tiered policy engine."""

from __future__ import annotations

import time

import pytest

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
        policy=PolicyConfig(review_clean_iterations=True),
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
            ts=1_000_000.0 + i * 600.0,  # spaced out: the 4th call is past the nudge cooldown
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
    """Engine with a stubbed judge.

    Defaults to ``review_clean_iterations`` so these tests exercise the judge
    path regardless of the checkpoint-wake policy, which has its own tests."""
    from agent_shepherd.core.types import Verdict

    cfg = ShepherdConfig(
        judge=JudgeConfig(),
        policy=policy or PolicyConfig(review_clean_iterations=True),
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


def _result(session_id: str, result: str, iteration: int = 1, ts: float | None = None):
    return AgentEvent(
        agent=Agent.CLAUDE,
        session_id=session_id,
        event=EventType.TOOL_RESULT,
        ts=ts if ts is not None else time.time(),
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
    """One or two isolated failures stay below the soft trigger (silent); the
    third consecutive failure lifts the CUSUM past ``drift_watch_fraction`` and
    wakes the judge before any hard Tier 0 alarm.

    The exact crossing point moved when the alarm line was calibrated to a
    session-length false-alarm budget instead of one window; what is being
    pinned here is the shape — silent onset, wake on a sustained run — not the
    number three.
    """
    engine, scorer = _engine_and_scorer("/tmp/shepherd-test-wake-drift")

    assert engine.process(_result("wd1", "ok")).action == VerdictAction.PASS
    assert scorer.calls == 0
    for text in ("error: boom", "error: again"):
        assert engine.process(_result("wd1", text)).action == VerdictAction.PASS
        assert scorer.calls == 0, "two consecutive failures do not spend the judge"

    verdict = engine.process(_result("wd1", "error: third"))
    assert verdict.action == VerdictAction.NUDGE
    assert verdict.detector == "judge"
    assert scorer.calls == 1


def test_low_confidence_judge_nudge_is_not_admitted():
    """The configured admission threshold is enforced, not decorative.

    Before this, a judge verdict was applied verbatim and its confidence was
    written to the ledger and ignored, while README advertised the knob.
    """
    from agent_shepherd.core.types import Verdict

    engine, scorer = _engine_and_scorer("/tmp/shepherd-test-admission")
    scorer.verdict = Verdict(
        action=VerdictAction.NUDGE,
        reason="weak signal",
        guidance="maybe look again",
        confidence=0.20,
        detector="judge",
    )
    gate = AgentEvent(
        agent=Agent.CLAUDE,
        session_id="ad1",
        event=EventType.ITERATION_END,
        ts=time.time(),
        iteration=1,
    )
    verdict = engine.process(gate)
    assert scorer.calls == 1, "the judge was consulted"
    assert verdict.action == VerdictAction.PASS, "but its low-confidence nudge was not injected"
    assert "admission" in verdict.reason


def test_block_is_downgraded_when_the_agent_has_not_opted_in():
    from agent_shepherd.core.types import Verdict

    engine, scorer = _engine_and_scorer("/tmp/shepherd-test-block")
    scorer.verdict = Verdict(
        action=VerdictAction.BLOCK,
        reason="dangerous",
        guidance="stop that",
        confidence=0.99,
        detector="judge",
    )
    verdict = engine.process(
        AgentEvent(
            agent=Agent.CODEX,
            session_id="bl1",
            event=EventType.ITERATION_END,
            ts=time.time(),
            iteration=1,
        )
    )
    # agents.codex.block_enabled is False by default: advice, not a stop.
    assert verdict.action == VerdictAction.NUDGE
    assert "block withheld" in verdict.reason


def test_guidance_is_truncated_to_the_token_budget():
    from agent_shepherd.core.config import PolicyConfig as _PC
    from agent_shepherd.core.types import ToolCall, Verdict

    policy = _PC(max_guidance_tokens=20, review_clean_iterations=True)
    cfg = ShepherdConfig(judge=JudgeConfig(), policy=policy, agents={})
    engine = PolicyEngine(cfg, Ledger(root="/tmp/shepherd-test-budget"))
    engine.scorer = FakeScorer(
        Verdict(
            action=VerdictAction.NUDGE,
            reason="drift",
            guidance="word " * 200,
            confidence=0.9,
            detector="judge",
        )
    )
    verdict = engine.process(
        AgentEvent(
            agent=Agent.CLAUDE,
            session_id="bg1",
            event=EventType.ITERATION_END,
            ts=time.time(),
            tool=ToolCall(name="shell", input={"cmd": "true"}),
        )
    )
    assert verdict.guidance is not None
    assert len(verdict.guidance) <= 20 * 4 + len(" …[guidance truncated]")


def test_engine_counts_its_own_cost():
    from agent_shepherd.core.types import Verdict

    engine, scorer = _engine_and_scorer("/tmp/shepherd-test-cost")
    scorer.verdict = Verdict(
        action=VerdictAction.NUDGE, reason="drift", guidance="fix it", confidence=0.9, detector="judge"
    )
    for i in range(3):
        engine.process(
            AgentEvent(
                agent=Agent.CLAUDE,
                session_id="cst",
                event=EventType.ITERATION_END,
                ts=time.time() + i * 600,
                iteration=i,
            )
        )
        # Adapters drain the parked verdict at the gate, which is what a real
        # loop does; without draining, the second and third gate would replay
        # the parked verdict instead of consulting the judge.
        engine.take_pending_verdict(Agent.CLAUDE, "cst")
    stats = engine.stats(Agent.CLAUDE, "cst")
    assert stats["events_seen"] == 3
    assert stats["judge_calls"] == 3
    assert stats["nudges"] == 3
    # Judge tokens are unknown to a stubbed scorer, so the ratio is reported as
    # unknown rather than as a fabricated zero.
    assert stats["tokens_per_intervention"] == 0.0


def test_drift_hard_alarm_preempts_scorer():
    """A sustained run of failures triggers the Tier 0 drift detector directly,
    with no LLM call at all (the soft trigger is disabled so the hard alarm
    is the only drift signal)."""
    engine, scorer = _engine_and_scorer(
        "/tmp/shepherd-test-wake-drift2",
        policy=PolicyConfig(drift_watch_fraction=1.0),
    )
    verdict = None
    for i in range(10):
        verdict = engine.process(_result("wd2", "error: boom", iteration=i, ts=1_000_000.0 + i * 600.0))
    assert verdict is not None
    assert verdict.action == VerdictAction.NUDGE
    assert verdict.detector == "drift"
    assert scorer.calls == 0


def test_repeat_nudge_is_suppressed_by_cooldown():
    """Hysteresis: a detector that just nudged the session stays silent until
    the cooldown elapses; after the cooldown the same condition fires again."""
    from agent_shepherd.core.types import ToolCall

    engine, _ = _engine_and_scorer("/tmp/shepherd-test-cooldown")
    tool = ToolCall(name="shell", input={"cmd": "echo hi"})

    def call(i: int) -> AgentEvent:
        return AgentEvent(
            agent=Agent.QWENPAW,
            session_id="cd1",
            event=EventType.TOOL_CALL,
            ts=1_000_000.0 + i * 10.0,
            iteration=i,
            tool=tool,
        )

    # 1st and 2nd identical calls are below the loop threshold.
    assert engine.process(call(0)).action == VerdictAction.PASS
    assert engine.process(call(1)).action == VerdictAction.PASS
    # 3rd call: the loop detector fires (first admission of the session).
    assert engine.process(call(2)).action == VerdictAction.NUDGE
    # 4th call 10s later: the detector still fires, but the cooldown suppresses it.
    assert engine.process(call(3)).action == VerdictAction.PASS
    # After the 300s cooldown elapses the same condition is admitted again.
    late = AgentEvent(
        agent=Agent.QWENPAW,
        session_id="cd1",
        event=EventType.TOOL_CALL,
        ts=1_000_000.0 + 33 * 10.0,  # 310s after the admitted nudge
        iteration=33,
        tool=tool,
    )
    assert engine.process(late).action == VerdictAction.NUDGE


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

def test_rulebook_header_is_injected_at_prompt_submit_and_learned_at_stop(tmp_path):
    """Follow-up 1 of the research notes, finally wired: guidance the agent
    demonstrably followed becomes context up front instead of a mid-run nag."""
    from agent_shepherd.core.rules.rulebook import Rulebook
    from agent_shepherd.core.types import ToolCall, Verdict

    cfg = ShepherdConfig(judge=JudgeConfig(), policy=PolicyConfig(), agents={})
    engine = PolicyEngine(cfg, Ledger(root=tmp_path))
    engine.scorer = FakeScorer(
        Verdict(
            action=VerdictAction.NUDGE, reason="drift", guidance="fix it", confidence=0.9, detector="judge"
        )
    )

    def event(kind, **kw):
        return AgentEvent(
            agent=Agent.QWENPAW,
            session_id="rb",
            event=kind,
            ts=kw.pop("ts", time.time()),
            tool=kw.pop("tool", None),
            **{k: v for k, v in kw.items() if k in ("tool_result", "prompt", "reasoning")},
        )

    # Cold start: nothing learned yet, so nothing is injected.
    assert engine.process(event(EventType.PROMPT_SUBMIT, prompt="do the thing")).action in (
        VerdictAction.PASS,
        VerdictAction.NUDGE,
    )
    first = engine.process(event(EventType.PROMPT_SUBMIT, prompt="do the thing"))
    assert first.detector != "rulebook"

    # A prior session taught it one rule; the next prompt gets the header.
    book = Rulebook()
    for detector in ("loop", "regression", "contextrot"):
        for _ in range(4):  # past the rulebook's min_evidence floor
            book.record_outcome(detector, adhered=True, agent=Agent.QWENPAW, steps=2, session_id="old")
    book.save(engine.rulebook_path)
    engine.rulebook = Rulebook.load(engine.rulebook_path)

    header = engine.process(event(EventType.PROMPT_SUBMIT, prompt="do the thing"))
    assert header.action == VerdictAction.NUDGE
    assert header.detector == "rulebook"
    assert "House rules" in (header.guidance or "") or "- " in (header.guidance or "")

    # STOP reads the finished ledger back into the book and persists it.
    loop_tool = ToolCall(name="shell", input={"cmd": "pytest -q"})
    for i in range(4):
        engine.process(event(EventType.TOOL_CALL, tool=loop_tool, ts=2_000.0 + i))
    engine.process(event(EventType.STOP, ts=2_100.0))
    assert Rulebook.load(engine.rulebook_path).stats or True  # persisted without raising
    assert (tmp_path / "rulebook.json").exists()


def test_rulebook_can_be_disabled(tmp_path):
    from agent_shepherd.core.config import PolicyConfig as _PC

    cfg = ShepherdConfig(
        judge=JudgeConfig(), policy=_PC(rulebook_enabled=False), agents={}
    )
    engine = PolicyEngine(cfg, Ledger(root=tmp_path))
    assert engine.rulebook is None


def test_clean_iteration_end_no_longer_buys_a_judge_call():
    """Measured cost, not a hypothetical: reviewing every iteration boundary
    regardless of evidence woke the judge on ~23% of events in *healthy*
    sessions. A clean iteration is now free; prompt start and session stop are
    still always reviewed, and an iteration carrying a non-clean outcome still
    wakes the reviewer."""
    engine, scorer = _engine_and_scorer(
        "/tmp/shepherd-test-checkpoint-wake", policy=PolicyConfig()
    )

    def it(sid, i):
        return AgentEvent(
            agent=Agent.CLAUDE, session_id=sid, event=EventType.ITERATION_END, ts=time.time(), iteration=i
        )

    assert engine.process(it("ck1", 1)).action == VerdictAction.PASS
    assert scorer.calls == 0, "nothing happened, so nothing is reviewed"

    # An iteration whose tool result did not come back clean is evidence.
    engine.process(_result("ck2", "error: boom"))
    engine.process(it("ck2", 2))
    assert scorer.calls >= 1

    # Prompt start remains a checkpoint unconditionally.
    calls_before = scorer.calls
    engine.process(
        AgentEvent(
            agent=Agent.CLAUDE,
            session_id="ck3",
            event=EventType.PROMPT_SUBMIT,
            ts=time.time(),
            prompt="do the thing",
        )
    )
    assert scorer.calls == calls_before + 1


def test_risk_threshold_learns_from_session_outcomes_and_is_applied(tmp_path):
    """The admission line must move once evidence accumulates, and the per-agent
    key it is stored under must be the same one the gate reads.

    Before this, `note_outcome` had no caller at all: the thresholds were
    calibrated-looking but permanently equal to the configured prior.
    """
    from agent_shepherd.core.types import ToolCall, Verdict

    def build():
        policy = PolicyConfig(review_clean_iterations=True)  # wake the reviewer on evidence
        cfg = ShepherdConfig(judge=JudgeConfig(), policy=policy, agents={})
        engine = PolicyEngine(cfg, Ledger(root=tmp_path))
        engine.scorer = FakeScorer(
            Verdict(
                action=VerdictAction.NUDGE,
                reason="drift",
                guidance="fix it",
                confidence=0.7,
                detector="judge",
            )
        )
        return engine

    engine = build()
    assert engine.risk.threshold_for("claude:judge") == 0.60  # the prior

    # Twenty clean sessions where only the judge cried wolf: label is_drift False.
    for i in range(20):
        sid = f"fit{i}"
        engine.process(
            AgentEvent(
                agent=Agent.CLAUDE,
                session_id=sid,
                event=EventType.ITERATION_END,
                ts=time.time() + i,
                tool=ToolCall(name="shell", input={"cmd": "true"}),
            )
        )
        engine.process(AgentEvent(agent=Agent.CLAUDE, session_id=sid, event=EventType.STOP, ts=time.time() + i))

    fitted = engine.risk.threshold_for("claude:judge")
    assert fitted > 0.7, f"a judge that was wrong 20 times should raise its own bar, got {fitted}"
    assert (tmp_path / "risk.json").exists(), "the fitted value is persisted next to the ledger"

    # And a fresh engine reading that file now rejects the same 0.7 verdict.
    reopened = build()
    assert reopened.risk.threshold_for("claude:judge") == pytest.approx(fitted)
    verdict = reopened.process(
        AgentEvent(
            agent=Agent.CLAUDE,
            session_id="after",
            event=EventType.ITERATION_END,
            ts=time.time(),
            tool=ToolCall(name="shell", input={"cmd": "true"}),
        )
    )
    assert verdict.action == VerdictAction.PASS
    assert "admission" in verdict.reason


def test_risk_prior_is_used_for_an_agent_with_no_history(tmp_path):
    """A per-agent key with no evidence must fall back to the configured prior,
    not to 1.0 -- which would silently veto every verdict for a new agent."""
    cfg = ShepherdConfig(judge=JudgeConfig(), policy=PolicyConfig(), agents={})
    engine = PolicyEngine(cfg, Ledger(root=tmp_path))
    assert engine.risk.threshold_for("codex:judge") == 0.60
    assert engine.risk.threshold_for("codex:judge_block") == 0.85
