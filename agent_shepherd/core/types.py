"""Canonical event / verdict / guidance types shared by every adapter.

All adapters normalize their agent's native events into :class:`AgentEvent`
before pushing to the supervisor daemon. The daemon never sees a
Claude Code hook payload, a Codex rollout line, or a QwenPaw middleware
context — it only ever sees this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Agent(str, Enum):
    """The agents the supervisor currently knows about."""

    CLAUDE = "claude"
    CODEX = "codex"
    QWENPAW = "qwenpaw"


class EventType(str, Enum):
    """Normalized event kinds. The daemon's policy engine only reasons over these."""

    PROMPT_SUBMIT = "prompt_submit"  # user prompt entered
    REASONING = "reasoning"  # a model reasoning chunk (may be partial)
    TOOL_CALL = "tool_call"  # a tool invocation (pre-execution)
    TOOL_RESULT = "tool_result"  # the result of a tool invocation
    ITERATION_END = "iteration_end"  # end of one ReAct iteration
    STOP = "stop"  # agent finished / about to finish
    ERROR = "error"  # agent surfaced an error
    USER_INTERVENTION = "user_intervention"  # the human injected a message


class VerdictAction(str, Enum):
    """What the supervisor wants to do about the current situation."""

    PASS = "pass"  # silent, no intervention
    NUDGE = "nudge"  # inject advisory guidance into the agent's context
    BLOCK = "block"  # stop the offending action (opt-in, high-risk only)
    ESCALATE = "escalate"  # pause and hand control back to the human


@dataclass
class ToolCall:
    """A normalized tool invocation."""

    name: str
    input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str | None = None


@dataclass
class AgentEvent:
    """One normalized observation from an agent's loop."""

    agent: Agent
    session_id: str
    event: EventType
    ts: float  # unix seconds
    iteration: int = 0
    tool: ToolCall | None = None
    tool_result: str | None = None
    reasoning: str | None = None  # raw reasoning text, if available
    prompt: str | None = None  # the user prompt, for prompt_submit
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_str(self) -> str:
        return self.agent.value


@dataclass
class Verdict:
    """The supervisor's decision about the current situation."""

    action: VerdictAction
    reason: str  # why this verdict
    guidance: str | None = None  # the corrective text (for NUDGE / BLOCK)
    confidence: float = 0.0  # 0..1
    detector: str | None = None  # which detector fired (e.g. "loop", "judge")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "guidance": self.guidance,
            "confidence": self.confidence,
            "detector": self.detector,
        }


@dataclass
class GuidanceMsg:
    """A guidance message addressed at a specific agent session."""

    agent: Agent
    session_id: str
    text: str
    priority: str = "normal"  # "low" | "normal" | "high"
    verdict: Verdict | None = None