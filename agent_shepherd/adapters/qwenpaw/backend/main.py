"""QwenPaw in-process supervisor plugin.

This plugin is loaded by QwenPaw at startup (``~/.qwenpaw/plugins/agent-shepherd``)
and plugs into two native seams:

1. **middleware** — observes every reasoning step and tool execution in-process
   (highest fidelity, no polling).
2. **stop handler gate** — at the end of each ReAct iteration, asks the daemon
   for a verdict and, on drift, returns ``INTERRUPT_AND_CONTINUE`` with the
   corrective message as a new user turn. This is the true mid-loop nudge.

The daemon is a plain HTTP service on ``127.0.0.1:4890``. If it is down, the
plugin fails open: it observes silently and never blocks the agent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any

import httpx
from agentscope.middleware import MiddlewareBase
from qwenpaw.loop.gates import StopAction, StopGate, StopHandlerResult
from qwenpaw.plugins.api import PluginApi

logger = logging.getLogger("agent_shepherd.qwenpaw")
logger.debug("plugin module imported pid=%s file=%s", os.getpid(), __file__)


def _daemon_url() -> str:
    return os.environ.get("SHEPHERD_DAEMON_URL", "http://127.0.0.1:4890").rstrip("/")


def _post(path: str, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any] | None:
    """POST to the daemon; fail open on any error."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{_daemon_url()}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - fail open, never wedge the agent
        logger.warning("shepherd daemon unavailable: %s", exc)
        return None


def _normalize_event(
    *,
    session_id: str,
    event: str,
    iteration: int = 0,
    tool_name: str | None = None,
    tool_input: dict[str, Any] | None = None,
    tool_result: str | None = None,
    reasoning: str | None = None,
    prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical event shape the daemon expects."""
    return {
        "agent": "qwenpaw",
        "session_id": session_id,
        "event": event,
        "ts": time.time(),
        "iteration": iteration,
        "tool": {"name": tool_name, "input": tool_input or {}} if tool_name else None,
        "tool_result": tool_result,
        "reasoning": reasoning,
        "prompt": prompt,
        "metadata": metadata or {},
    }


def _ctx_value(ctx: Any, *keys: str, default: str = "unknown") -> str:
    """Read a value from QwenPaw's hook ctx (dict-like or object-like)."""
    for key in keys:
        if isinstance(ctx, dict) and ctx.get(key):
            return str(ctx[key])
        value = getattr(ctx, key, None)
        if value:
            return str(value)
    return default


def _session_id(ctx: Any) -> str:
    return _ctx_value(ctx, "session_id", "chat_id", "session")


def _iteration(ctx: Any) -> int:
    value = _ctx_value(ctx, "iteration", default="0")
    try:
        return int(value)
    except ValueError:
        return 0


def _agent_session_id(agent: Any) -> str:
    """Resolve session id from QwenPaw's agent object for stop gates."""
    sid = _ctx_value(agent, "session_id", "chat_id", "session")
    if sid != "unknown":
        return sid
    state = getattr(agent, "state", None)
    if state is not None:
        sid = _ctx_value(state, "session_id", "chat_id", "session")
        if sid != "unknown":
            return sid
    request_context = getattr(agent, "_request_context", None) or {}
    if isinstance(request_context, dict):
        for key in ("session_id", "chat_id", "session"):
            if request_context.get(key):
                return str(request_context[key])
    return "unknown"


def _get(path: str, timeout: float = 2.0) -> dict[str, Any] | None:
    """GET from the daemon; fail open on any error."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{_daemon_url()}{path}")
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001 - fail open, never wedge the agent
        return None


class SupervisorMiddleware(MiddlewareBase):
    """Observe reasoning and tool calls via the agentscope onion hooks.

    The factory resolves the session id once per request (from the build
    ctx); the agent object may not carry it yet during assembly, so the
    middleware and the gate share the same resolved id — otherwise the
    daemon's per-session sliding window splits across two session ids and
    gate verdicts never see what the middleware observed.
    """

    def __init__(self, session_id: str = "unknown"):
        self.session_id = session_id

    def _resolve_session(self, agent: Any) -> str:
        """Prefer the live agent's session id; fall back to the factory's."""
        if agent is not None:
            sid = _agent_session_id(agent)
            if sid != "unknown":
                return sid
        return self.session_id

    async def on_reasoning(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[Any, None]],
    ) -> AsyncGenerator[Any, None]:
        # Stream the consolidated reasoning text once the delta generator
        # exhausts, instead of one HTTP POST per token — per-delta POSTs
        # flood the daemon and interleave with the sliding window.
        chunks: list[str] = []
        async for item in next_handler():
            delta = getattr(item, "delta", None)
            if delta:
                chunks.append(str(delta))
            yield item
        text = "".join(chunks)
        if text:
            _post(
                "/ingest/qwenpaw",
                _normalize_event(
                    session_id=self._resolve_session(agent),
                    event="reasoning",
                    reasoning=text[:2000],
                ),
            )

    async def on_acting(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[Any, None]],
    ) -> AsyncGenerator[Any, None]:
        tool_call = input_kwargs.get("tool_call")
        tool_name = getattr(tool_call, "name", str(tool_call))
        raw_input = getattr(tool_call, "input", {})
        tool_input: dict[str, Any]
        if isinstance(raw_input, dict):
            tool_input = raw_input
        else:
            try:
                parsed = json.loads(str(raw_input))
                tool_input = parsed if isinstance(parsed, dict) else {"raw": str(raw_input)}
            except (ValueError, TypeError):
                tool_input = {"raw": str(raw_input)}
        _post(
            "/ingest/qwenpaw",
            _normalize_event(
                session_id=self._resolve_session(agent),
                event="tool_call",
                tool_name=str(tool_name),
                tool_input=tool_input,
            ),
        )
        async for item in next_handler():
            yield item


class SupervisorGate(StopGate):
    """Ask the daemon for a verdict at the end of each iteration."""

    def __init__(self, session_id: str = "unknown"):
        self.session_id = session_id
        self._continuation = ""

    @property
    def name(self) -> str:
        return "agent-shepherd"

    @property
    def priority(self) -> int:
        return 20

    async def check(self, ctx: Any) -> StopHandlerResult | None:
        agent = ctx.get("agent") if isinstance(ctx, dict) else None
        session_id = _agent_session_id(agent) if agent is not None else self.session_id
        if session_id == "unknown":
            session_id = self.session_id
        has_tools = bool(ctx.get("has_tool_calls")) if isinstance(ctx, dict) else False
        logger.debug(
            "gate check sid=%s has_tool_calls=%s it=%s",
            session_id,
            has_tools,
            _iteration(ctx),
        )

        if isinstance(ctx, dict) and ctx.get("has_tool_calls"):
            # Tool-call iteration: the runner ignores INTERRUPT_AND_CONTINUE
            # here (only TERMINATE gets deferred via _gate_pending_stop), so
            # draining a parked verdict now would lose it silently. Hold the
            # verdict for the next text/stop boundary where injection works.
            return None

        # Drain a verdict that fired on an inline event (loop nudge during a
        # tool POST, contextrot during reasoning) — the gate boundary is the
        # only place the plugin can actually inject guidance.
        pending = _get(f"/pending/qwenpaw/{session_id}")
        verdict = pending or {}
        if not verdict:
            verdict = _post(
                "/ingest/qwenpaw",
                _normalize_event(
                    session_id=session_id,
                    event="iteration_end",
                    iteration=_iteration(ctx),
                ),
            ) or {}
        action = verdict.get("action", "pass")
        guidance = verdict.get("guidance")
        reason = verdict.get("reason", "")
        logger.debug(
            "gate verdict action=%s detector=%s has_guidance=%s",
            action,
            verdict.get("detector"),
            bool(guidance),
        )
        if action == "nudge" and guidance:
            self._continuation = guidance
            return StopHandlerResult(
                action=StopAction.INTERRUPT_AND_CONTINUE,
                continuation_message=guidance,
                reason=reason,
            )
        if action in ("block", "escalate"):
            return StopHandlerResult(
                action=StopAction.TERMINATE,
                reason=reason or guidance or "supervisor stopped the agent",
            )
        return None

    def build_continuation(self) -> str:
        return self._continuation


class SupervisorStopHandler:
    """Single-gate stop handler that composes with QwenPaw's mode handlers.

    QwenPaw's built-in ``StopHandler`` returns ``TERMINATE`` when none of its
    gates produce a continuation — correct for a complete stop-policy bundle,
    wrong for one advisory gate. Two things make the gate actually reachable:

    * ``BYPASS`` when the gate is idle (not ``TERMINATE``), so the chain
      falls through to the default-mode handler, which is what cleanly stops
      the agent on a normal answer.
    * registered with ``priority < 0`` so it runs *before* the default-mode
      handler (priority 0), whose empty gate set would otherwise terminate
      the chain first and silently swallow a drift verdict.
    """

    def __init__(self, gate: SupervisorGate):
        self._gate = gate

    async def __call__(self, ctx: Any) -> StopHandlerResult:
        result = await self._gate.check(ctx)
        if result is None:
            return StopHandlerResult(action=StopAction.BYPASS)
        return result


def _supervisor_factory(ctx: Any, agent_config: Any) -> SupervisorMiddleware | None:
    return SupervisorMiddleware(session_id=_session_id(ctx))


class SupervisorPlugin:
    def register(self, api: PluginApi) -> None:
        api.register_middleware(_supervisor_factory, priority=50)
        api.register_agent_stop_handler(
            handler=SupervisorStopHandler(SupervisorGate()),
            priority=-100,
            name="agent-shepherd",
        )
        logger.debug(
            "middleware + stop handler registered pid=%s",
            os.getpid(),
        )


plugin = SupervisorPlugin()