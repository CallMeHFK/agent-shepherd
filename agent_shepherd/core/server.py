"""Supervisor daemon: HTTP ingest + policy engine + guidance egress.

The daemon is deliberately framework-free (stdlib ``http.server``) so it can
run anywhere the user can run Python. Adapters POST normalized ``AgentEvent``s
to ``POST /ingest/<agent>`` and receive a ``Verdict`` back. The daemon keeps a
per-session state (sliding window of events) and writes every observation and
verdict to the append-only ledger.

The policy engine is a two-tier pipeline:

1. **Tier 0 — deterministic detectors** (zero cost, always on): loop detection
   (exact and near-duplicate), regression detection, off-spec edits judged
   against the session's delegation contract, binding drift, context rot, and
   sustained-drift detection (a CUSUM alarm calibrated to a false-alarm budget
   over the whole session rather than one window). The drift detector's rising
   statistic also soft-triggers the judge between checkpoints.
2. **Tier 1 — LLM step scorer** (PRM-style, sliding window): the judge scores
   the recent steps on on-goal / justified / verified and returns a drift score
   plus actionable guidance.

Between the two tiers and the agent sit the controls that make the supervisor
worth running: a per-detector NUDGE cooldown (hysteresis), risk-controlled
admission of judge verdicts instead of taking its self-reported confidence at
face value, a guidance token budget, and cost counters that say what the
supervision actually cost.

Verdicts: PASS (silent) / NUDGE (guidance text) / BLOCK (stop the action) /
ESCALATE (hand control to the human).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import ShepherdConfig
from .judge.risk import RiskModel, truncate_guidance
from .judge.tier1 import StepScorer
from .ledger import Ledger
from .rules import signals
from .rules.detectors import (
    BindingDriftDetector,
    ContextRotDetector,
    CUSUMDriftDetector,
    Detector,
    LoopDetector,
    OffSpecDetector,
    RegressionDetector,
)
from .rules.rulebook import Rulebook
from .types import Agent, AgentEvent, EventType, Verdict, VerdictAction


@dataclass
class SessionState:
    """Sliding window of recent events for one agent session, plus its costs."""

    events: list[AgentEvent] = field(default_factory=list)
    max_events: int = 60
    # Per-detector timestamp of the last admitted NUDGE; drives the cooldown
    # (hysteresis) so a detector does not re-nag the agent with the same
    # verdict while the situation is unchanged.
    last_nudge: dict[str, float] = field(default_factory=dict)
    events_seen: int = 0
    judge_calls: int = 0
    judge_tokens: int = 0
    nudges: int = 0
    blocked: int = 0
    escalated: int = 0
    suppressed: int = 0

    def push(self, event: AgentEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_seen": self.events_seen,
            "judge_calls": self.judge_calls,
            "judge_tokens": self.judge_tokens,
            "nudges": self.nudges,
            "blocked": self.blocked,
            "escalated": self.escalated,
            "suppressed": self.suppressed,
            # The number that answers "is the supervisor paying for itself":
            # judge tokens spent per intervention that reached the agent.
            "tokens_per_intervention": (
                round(self.judge_tokens / (self.nudges + self.blocked + self.escalated), 1)
                if (self.nudges + self.blocked + self.escalated)
                else None
            ),
        }


class PolicyEngine:
    """Tiered policy engine: deterministic detectors + LLM step scorer."""

    def __init__(self, config: ShepherdConfig, ledger: Ledger):
        self.config = config
        self.ledger = ledger
        # Specific pattern detectors first; the CUSUM drift detector last, so a
        # sharp local pattern (a loop, a regression) is reported with its
        # precise diagnosis rather than the coarser "sustained drift" message.
        self.detectors: list[Detector] = [
            LoopDetector(),
            RegressionDetector(),
            OffSpecDetector(
                allow_globs=config.policy.scope_allow_globs,
                deny_globs=config.policy.scope_deny_globs,
                nudge_unobserved=config.policy.scope_nudge_unobserved,
            ),
            ContextRotDetector(),
        ]
        if config.policy.binding_enabled:
            self.detectors.append(
                BindingDriftDetector(similarity=config.policy.binding_similarity)
            )
        self._drift: CUSUMDriftDetector | None = None
        if config.policy.drift_enabled:
            self._drift = CUSUMDriftDetector(
                target_fpr=config.policy.drift_target_fpr,
                horizon=config.policy.drift_horizon,
                adaptive_baseline=config.policy.drift_adaptive_baseline,
                unknown_weight=config.policy.drift_unknown_weight,
            )
            self.detectors.append(self._drift)
        self.scorer = StepScorer(config.judge)
        # The rulebook lives beside the ledger rather than at $SHEPHERD_HOME so
        # that an engine pointed at a temp ledger cannot write into the user's
        # real config directory.
        self.rulebook_path = self.ledger.root / "rulebook.json"
        self.rulebook: Rulebook | None = None
        if config.policy.rulebook_enabled:
            self.rulebook = Rulebook.load(self.rulebook_path)
        self.risk = RiskModel.load(
            defaults={
                "judge": config.policy.nudge_threshold,
                "judge_block": config.policy.block_threshold,
            },
            target_fpr=config.policy.risk_target_fpr,
            min_samples=config.policy.risk_min_samples,
        )
        self._sessions: dict[tuple[str, str], SessionState] = {}
        # Verdicts that fired inline on an event the adapter can't act on
        # (e.g. a loop nudge during a streaming reasoning POST) are parked
        # here so the next gate check (iteration boundary) can drain them.
        self._pending: dict[tuple[str, str], Verdict] = {}
        self._lock = threading.RLock()

    def _state(self, agent: Agent, session_id: str) -> SessionState:
        key = (agent.value, session_id)
        with self._lock:
            return self._sessions.setdefault(key, SessionState())

    def stats(self, agent: Agent, session_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._sessions.get((agent.value, session_id))
        return state.to_dict() if state else {}

    def take_pending_verdict(self, agent: Agent, session_id: str) -> Verdict | None:
        """Return and clear the most recent non-PASS verdict for a session."""
        with self._lock:
            return self._pending.pop((agent.value, session_id), None)

    def _park(self, key: tuple[str, str], verdict: Verdict) -> None:
        with self._lock:
            if verdict.action != VerdictAction.PASS:
                self._pending[key] = verdict

    def _admit(self, state: SessionState, event: AgentEvent, verdict: Verdict) -> Verdict | None:
        """Apply the per-detector NUDGE cooldown (hysteresis).

        A repeat NUDGE from the same detector for the same session, within
        ``nudge_cooldown_seconds``, is suppressed — the agent already has that
        guidance in its context, and re-nagging is the "saturation trap" in
        action. ``None`` means "suppressed, stay silent"; otherwise the
        verdict is admitted and its cooldown clock is (re)started.
        BLOCK/ESCALATE/PASS are never suppressed.
        """
        if verdict.action != VerdictAction.NUDGE:
            return verdict
        last = state.last_nudge.get(verdict.detector)
        if last is not None and event.ts - last < self.config.policy.nudge_cooldown_seconds:
            return None
        state.last_nudge[verdict.detector] = event.ts
        return verdict

    def _control(self, agent: Agent, verdict: Verdict) -> Verdict:
        """Apply the calibrated admission thresholds and the guidance budget.

        A judge's verbalized confidence is a score, not a guarantee, and this
        project's own literature says thresholding that score is the wrong
        control (arXiv 2606.21399, 2412.14737). So the cut-points start at the
        configured priors and are replaced by conformal-risk-control values once
        enough labeled outcomes have accumulated (see :func:`note_outcome`), and
        BLOCK is only ever emitted for an agent that opted in.
        """
        if verdict.detector != "judge":
            return self._budget(verdict)
        if verdict.action == VerdictAction.NUDGE:
            limit = self.risk.threshold_for("judge")
            if verdict.confidence < limit:
                return Verdict(
                    action=VerdictAction.PASS,
                    reason=(
                        f"judge nudge below risk-controlled admission line "
                        f"(confidence {verdict.confidence:.2f} < {limit:.2f})"
                    ),
                    confidence=0.0,
                    detector="judge",
                )
        if verdict.action in (VerdictAction.BLOCK, VerdictAction.ESCALATE):
            limit = self.risk.threshold_for("judge_block")
            allowed = self.config.agent_config(agent.value).block_enabled
            if verdict.confidence < limit or not allowed:
                # Losing the authority to stop, keeping the advice.
                return self._budget(
                    Verdict(
                        action=VerdictAction.NUDGE,
                        reason=f"{verdict.reason} (block withheld)",
                        guidance=verdict.guidance,
                        confidence=verdict.confidence,
                        detector="judge",
                    )
                )
        return self._budget(verdict)

    def _budget(self, verdict: Verdict) -> Verdict:
        text = truncate_guidance(verdict.guidance, self.config.policy.max_guidance_tokens)
        return verdict if text == verdict.guidance else Verdict(
            action=verdict.action,
            reason=verdict.reason,
            guidance=text,
            confidence=verdict.confidence,
            detector=verdict.detector,
        )

    def note_outcome(self, agent: Agent, session_id: str, key: str, score: float, is_drift: bool) -> None:
        """Feed one labeled outcome to the risk model and refit that threshold.

        The labels come from observed adherence in the ledger (did the agent
        change behavior the way the nudge asked) and from the offline benchmark,
        not from a human reading transcripts.
        """
        self.risk.observe(f"{agent.value}:{key}" if ":" not in key else key, score, is_drift)
        self.risk.refit(f"{agent.value}:{key}" if ":" not in key else key)
        self.risk.save()

    def _wake(self, agent: Agent, event: AgentEvent, history: list[AgentEvent]) -> bool:
        """Deterministic gate: should the LLM judge wake up for this event?

        The gate is onset-based rather than a fixed threshold on state. The
        judge wakes:

        * at natural checkpoints (prompt start, iteration end, session stop),
          where a full review of the window is cheap and meaningful;
        * mid-iteration, only when the cheap CUSUM drift statistic climbs to
          ``drift_watch_fraction`` of its alarm line on a tool result — i.e.
          on evidence of *onset*, not on state that stays high.

        Waking on every tool event is the "state-saturation trap": a judge that
        reviews a large constant fraction of actions costs more than the agent
        it supervises, so the expensive reviewer must stay asleep except when
        something actually starts to happen.
        """
        if event.event in (EventType.PROMPT_SUBMIT, EventType.STOP):
            return True
        if event.event == EventType.ITERATION_END:
            if self.config.policy.review_clean_iterations:
                return True
            # An iteration whose every outcome came back clean is not evidence
            # of anything; spending a reviewer on it is the cost side of the
            # saturation trap measured by `shepherd eval`.
            return any(
                signals.classify(e).state != signals.OK for e in history if e.event == EventType.TOOL_RESULT
            )
        if event.event == EventType.TOOL_RESULT and self._drift is not None:
            return self._drift.watch_level(event, history) >= self.config.policy.drift_watch_fraction
        return False

    def _run_detectors(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        """Run all Tier 0 detectors; return the first (highest priority) hit.

        ``history`` is the window *before* the current event; every detector
        follows the contract ``evaluate(event, history)`` where ``history``
        excludes the current event (each accounts for the current event itself).
        """
        for detector in self.detectors:
            verdict = detector.evaluate(event, history)
            if verdict is not None:
                return verdict
        return None

    def _score(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        """Tier 1: ask the LLM step scorer to score the sliding window."""
        if self.scorer is None:
            return None
        try:
            return self.scorer.score(event, history)
        except Exception as exc:  # noqa: BLE001 - fail open, never wedge the agent
            if self.config.policy.fail_open:
                return None
            return Verdict(
                action=VerdictAction.ESCALATE,
                reason=f"judge failed: {exc}",
                confidence=1.0,
                detector="judge",
            )

    def _cost(self, state: SessionState) -> dict[str, Any]:
        return {
            "judge_calls": state.judge_calls,
            "judge_tokens": state.judge_tokens,
            "events_seen": state.events_seen,
        }

    def process(self, event: AgentEvent) -> Verdict:
        """Consume one event and return a verdict for the adapter to apply."""
        self.ledger.record_event(event)
        state = self._state(event.agent, event.session_id)
        # Capture the window *before* this event. Both the detectors and the
        # scorer expect ``history`` to exclude the current event (each accounts
        # for it itself); pushing first would double-count it.
        history = list(state.events)
        state.push(event)
        state.events_seen += 1
        key = (event.agent.value, event.session_id)

        # Gate boundary: drain any verdict parked by an earlier inline event
        # before evaluating the gate event itself.
        if event.event in (EventType.ITERATION_END, EventType.STOP, EventType.PROMPT_SUBMIT):
            pending = self.take_pending_verdict(event.agent, event.session_id)
            if pending is not None:
                return pending

        # Session start: promote the guidance this agent demonstrably followed
        # in past sessions into context it sees before drifting, instead of
        # re-nagging it mid-run (arXiv 2509.03990).
        if event.event == EventType.PROMPT_SUBMIT and self.rulebook is not None:
            header = self.rulebook.render_header(self.config.policy.rulebook_budget_tokens)
            if header:
                verdict = Verdict(
                    action=VerdictAction.NUDGE,
                    reason="house rules from past sessions",
                    guidance=header,
                    confidence=0.99,
                    detector="rulebook",
                )
                self.ledger.record_verdict(
                    event.agent, event.session_id, verdict, context=self._cost(state)
                )
                return verdict

        # Session end: the whole trajectory is in the ledger, which is the only
        # place "did the agent actually follow the nudge" can be answered.
        if event.event == EventType.STOP and self.rulebook is not None:
            self.rulebook.observe(
                self.ledger.iter_records(event.agent, event.session_id),
                agent=event.agent,
                session_id=event.session_id,
            )
            self.rulebook.save(self.rulebook_path)

        # Tier 0 first: deterministic signals are cheap and always on.
        verdict = self._run_detectors(event, history)
        if verdict is not None:
            if self._admit(state, event, verdict) is None:
                return self._silent(state, event, f"{verdict.detector} nudge suppressed (cooldown)")
            verdict = self._control(event.agent, verdict)
            if verdict.action == VerdictAction.PASS:
                return self._silent(state, event, verdict.reason)
            self._tally(state, verdict)
            self.ledger.record_verdict(
                event.agent, event.session_id, verdict, context=self._cost(state)
            )
            self._park(key, verdict)
            return verdict

        # Tier 1 only when the deterministic gate says "wake up".
        if self._wake(event.agent, event, history):
            state.judge_calls += 1
            verdict = self._score(event, history)
            usage = getattr(self.scorer, "last_usage", None) or {}
            state.judge_tokens += int(usage.get("total_tokens") or 0)
            if verdict is not None:
                if self._admit(state, event, verdict) is None:
                    return self._silent(state, event, "judge nudge suppressed (cooldown)")
                verdict = self._control(event.agent, verdict)
                if verdict.action == VerdictAction.PASS:
                    self.ledger.record_verdict(
                        event.agent,
                        event.session_id,
                        verdict,
                        context={**self._cost(state), "admission": "risk_threshold"},
                    )
                    return verdict
                self._tally(state, verdict)
                self.ledger.record_verdict(
                    event.agent, event.session_id, verdict, context=self._cost(state)
                )
                self._park(key, verdict)
                return verdict

        return Verdict(action=VerdictAction.PASS, reason="no drift detected", confidence=0.0)

    def _silent(self, state: SessionState, event: AgentEvent, reason: str) -> Verdict:
        """Record a suppression so the ledger shows the hysteresis worked."""
        state.suppressed += 1
        verdict = Verdict(action=VerdictAction.PASS, reason=reason, confidence=0.0)
        self.ledger.record_verdict(event.agent, event.session_id, verdict, context=self._cost(state))
        return verdict

    def _tally(self, state: SessionState, verdict: Verdict) -> None:
        if verdict.action == VerdictAction.NUDGE:
            state.nudges += 1
        elif verdict.action == VerdictAction.BLOCK:
            state.blocked += 1
        elif verdict.action == VerdictAction.ESCALATE:
            state.escalated += 1


class ShepherdDaemon:
    """HTTP daemon that serves the policy engine."""

    def __init__(self, config: ShepherdConfig):
        self.config = config
        self.ledger = Ledger()
        self.engine = PolicyEngine(config, self.ledger)
        self._server: ThreadingHTTPServer | None = None

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        engine = self.engine

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                # Keep the daemon quiet unless the user asks for logs.
                return

            def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _pending(self, parts: list[str]) -> None:
                try:
                    agent = Agent(parts[1])
                except ValueError:
                    self._send_json({"error": f"unknown agent: {parts[1]}"}, 400)
                    return
                verdict = engine.take_pending_verdict(agent, parts[2])
                self._send_json(verdict.to_dict() if verdict is not None else {})

            def _stats(self, parts: list[str]) -> None:
                try:
                    agent = Agent(parts[1])
                except ValueError:
                    self._send_json({"error": f"unknown agent: {parts[1]}"}, 400)
                    return
                self._send_json(engine.stats(agent, parts[2]))

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    payload = json.loads(self.rfile.read(length))
                except json.JSONDecodeError:
                    self._send_json({"error": "invalid json"}, 400)
                    return

                if self.path == "/health":
                    self._send_json({"ok": True})
                    return

                parts = self.path.strip("/").split("/")
                if len(parts) == 2 and parts[0] == "ingest":
                    self._ingest(parts[1], payload)
                    return
                if len(parts) == 3 and parts[0] == "pending":
                    self._pending(parts)
                    return

                self._send_json({"error": "not found"}, 404)

            def _ingest(self, agent_name: str, payload: dict[str, Any]) -> None:
                from .types import ToolCall

                try:
                    agent = Agent(agent_name)
                except ValueError:
                    self._send_json({"error": f"unknown agent: {agent_name}"}, 400)
                    return

                tool_raw = payload.get("tool")
                tool = None
                if isinstance(tool_raw, dict):
                    tool = ToolCall(
                        name=str(tool_raw.get("name", "unknown")),
                        input=dict(tool_raw.get("input") or {}),
                        tool_use_id=tool_raw.get("tool_use_id"),
                    )
                event = AgentEvent(
                    agent=agent,
                    session_id=str(payload.get("session_id", "")),
                    event=EventType(payload.get("event", "")),
                    ts=float(payload.get("ts", time.time())),
                    iteration=int(payload.get("iteration", 0)),
                    tool=tool,
                    tool_result=payload.get("tool_result"),
                    reasoning=payload.get("reasoning"),
                    prompt=payload.get("prompt"),
                    metadata=payload.get("metadata") or {},
                )
                verdict = engine.process(event)
                self._send_json(verdict.to_dict())

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._send_json({"ok": True})
                    return
                parts = self.path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "pending":
                    self._pending(parts)
                    return
                if len(parts) == 3 and parts[0] == "stats":
                    self._stats(parts)
                    return
                self._send_json({"error": "not found"}, 404)

        return Handler

    def serve_forever(self, host: str = "127.0.0.1", port: int | None = None) -> None:
        """Run the daemon until interrupted."""
        port = port or self.config.port
        self._server = ThreadingHTTPServer((host, port), self._handler())
        print(f"shepherd daemon listening on http://{host}:{port}", flush=True)
        try:
            self._server.serve_forever()
        finally:
            self.engine.risk.save()
            self._server.server_close()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def create_daemon(config: ShepherdConfig | None = None) -> ShepherdDaemon:
    """Factory for the daemon (easy to wire into tests / embedders)."""
    return ShepherdDaemon(config or ShepherdConfig.load())
