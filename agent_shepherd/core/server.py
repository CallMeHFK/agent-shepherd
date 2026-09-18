"""Supervisor daemon: HTTP ingest + policy engine + guidance egress.

The daemon is deliberately framework-free (stdlib ``http.server``) so it can
run anywhere the user can run Python. Adapters POST normalized ``AgentEvent``s
to ``POST /ingest/<agent>`` and receive a ``Verdict`` back. The daemon keeps a
per-session state (sliding window of events) and writes every observation and
verdict to the append-only ledger.

The policy engine is a two-tier pipeline:

1. **Tier 0 — deterministic detectors** (zero cost, always on): loop detection
   (exact and near-duplicate), regression detection, off-spec edits, context
   rot, and sustained-drift detection (a CUSUM alarm with a calibrated
   false-alarm budget). The drift detector's rising statistic also soft-triggers
   the judge between checkpoints.
2. **Tier 1 — LLM step scorer** (PRM-style, sliding window): the judge scores
   the recent steps on on-goal / justified / verified and returns a drift score
   plus actionable guidance.

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
from .judge.tier1 import StepScorer
from .ledger import Ledger
from .rules.detectors import (
    ContextRotDetector,
    CUSUMDriftDetector,
    Detector,
    LoopDetector,
    OffSpecDetector,
    RegressionDetector,
)
from .types import Agent, AgentEvent, EventType, Verdict, VerdictAction


@dataclass
class SessionState:
    """Sliding window of recent events for one agent session."""

    events: list[AgentEvent] = field(default_factory=list)
    max_events: int = 60

    def push(self, event: AgentEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]


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
            OffSpecDetector(),
            ContextRotDetector(),
        ]
        self._drift: CUSUMDriftDetector | None = None
        if config.policy.drift_enabled:
            self._drift = CUSUMDriftDetector(target_fpr=config.policy.drift_target_fpr)
            self.detectors.append(self._drift)
        self.scorer = StepScorer(config.judge)
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

    def take_pending_verdict(self, agent: Agent, session_id: str) -> Verdict | None:
        """Return and clear the most recent non-PASS verdict for a session."""
        with self._lock:
            return self._pending.pop((agent.value, session_id), None)

    def _park(self, key: tuple[str, str], verdict: Verdict) -> None:
        with self._lock:
            if verdict.action != VerdictAction.PASS:
                self._pending[key] = verdict

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
        if event.event in (EventType.PROMPT_SUBMIT, EventType.ITERATION_END, EventType.STOP):
            return True
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

    def process(self, event: AgentEvent) -> Verdict:
        """Consume one event and return a verdict for the adapter to apply."""
        self.ledger.record_event(event)
        state = self._state(event.agent, event.session_id)
        # Capture the window *before* this event. Both the detectors and the
        # scorer expect ``history`` to exclude the current event (each accounts
        # for it itself); pushing first would double-count it.
        history = list(state.events)
        state.push(event)
        key = (event.agent.value, event.session_id)

        # Gate boundary: drain any verdict parked by an earlier inline event
        # before evaluating the gate event itself.
        if event.event in (EventType.ITERATION_END, EventType.STOP, EventType.PROMPT_SUBMIT):
            pending = self.take_pending_verdict(event.agent, event.session_id)
            if pending is not None:
                return pending

        # Tier 0 first: deterministic signals are cheap and always on.
        verdict = self._run_detectors(event, history)
        if verdict is not None:
            self.ledger.record_verdict(event.agent, event.session_id, verdict)
            self._park(key, verdict)
            return verdict

        # Tier 1 only when the deterministic gate says "wake up".
        if self._wake(event.agent, event, history):
            verdict = self._score(event, history)
            if verdict is not None:
                self.ledger.record_verdict(event.agent, event.session_id, verdict)
                self._park(key, verdict)
                return verdict

        return Verdict(action=VerdictAction.PASS, reason="no drift detected", confidence=0.0)


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
                    agent_name = parts[1]
                    try:
                        agent = Agent(agent_name)
                    except ValueError:
                        self._send_json({"error": f"unknown agent: {agent_name}"}, 400)
                        return

                    from .types import ToolCall

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
                    return

                if len(parts) == 3 and parts[0] == "pending":
                    # GET /pending/<agent>/<session>: drain the verdict parked
                    # by an inline event the adapter couldn't act on yet.
                    try:
                        agent = Agent(parts[1])
                    except ValueError:
                        self._send_json({"error": f"unknown agent: {parts[1]}"}, 400)
                        return
                    verdict = engine.take_pending_verdict(agent, parts[2])
                    self._send_json(verdict.to_dict() if verdict is not None else {})
                    return

                self._send_json({"error": "not found"}, 404)

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._send_json({"ok": True})
                    return
                parts = self.path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "pending":
                    try:
                        agent = Agent(parts[1])
                    except ValueError:
                        self._send_json({"error": f"unknown agent: {parts[1]}"}, 400)
                        return
                    verdict = engine.take_pending_verdict(agent, parts[2])
                    self._send_json(verdict.to_dict() if verdict is not None else {})
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
            self._server.server_close()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def create_daemon(config: ShepherdConfig | None = None) -> ShepherdDaemon:
    """Factory for the daemon (easy to wire into tests / embedders)."""
    return ShepherdDaemon(config or ShepherdConfig.load())