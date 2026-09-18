"""Tier 0 deterministic detectors.

Each detector is a pure function of the current event plus the recent event
history. Detectors run on every event and return the first Verdict that fires,
or ``None`` if the event looks healthy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import AgentEvent, EventType, Verdict, VerdictAction


class Detector(ABC):
    """Base class for a deterministic detector."""

    name: str

    @abstractmethod
    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        """Evaluate one event against the recent history."""
        raise NotImplementedError


def _tool_signature(event: AgentEvent) -> tuple[str, str] | None:
    """Stable signature of a tool call (name + canonicalized input)."""
    if event.event != EventType.TOOL_CALL or event.tool is None:
        return None
    import json

    try:
        input_json = json.dumps(event.tool.input, sort_keys=True, separators=(",", ":"))
    except TypeError:
        input_json = repr(event.tool.input)
    return (event.tool.name, input_json)


class LoopDetector(Detector):
    """Detect repeated identical tool calls within a sliding window.

    A loop is 3+ identical tool calls (same name + same input) in the last
    8 events. This catches the classic "agent retries the same broken command"
    pattern before it burns budget.
    """

    name = "loop"

    def __init__(self, window: int = 8, threshold: int = 3):
        self.window = window
        self.threshold = threshold

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        sig = _tool_signature(event)
        if sig is None:
            return None
        recent = history[-self.window :]
        count = 1 + sum(1 for e in recent if _tool_signature(e) == sig)
        if count >= self.threshold:
            return Verdict(
                action=VerdictAction.NUDGE,
                reason=(
                    f"loop detected: tool '{sig[0]}' repeated {count} times "
                    "with identical input in the recent window"
                ),
                guidance=(
                    "You appear to be repeating the same tool call without new "
                    "information. Stop and reassess: what did the last result "
                    "actually tell you, and what is the next genuinely different "
                    "step toward the goal?"
                ),
                confidence=0.95,
                detector=self.name,
            )
        return None


class RegressionDetector(Detector):
    """Detect a test/lint/build command that now fails after previously passing.

    A regression is flagged when a tool named like a test/lint/build command
    returns a non-zero exit code (or contains failure markers) and the same
    command previously returned success in the same session.
    """

    name = "regression"
    _TEST_NAMES = ("pytest", "test", "lint", "ruff", "eslint", "mypy", "build", "tsc", "check")

    def _is_test_command(self, event: AgentEvent) -> bool:
        if event.event != EventType.TOOL_RESULT or event.tool is None:
            return False
        name = event.tool.name.lower()
        return any(t in name for t in self._TEST_NAMES)

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        if not self._is_test_command(event):
            return None
        result = (event.tool_result or "").lower()
        failed = any(marker in result for marker in ("failed", "error", "exit code: 1", "non-zero"))
        if not failed:
            return None
        # Was this same tool successful earlier in the session?
        earlier_success = any(
            e.event == EventType.TOOL_RESULT
            and e.tool is not None
            and e.tool.name == event.tool.name
            and not any(marker in (e.tool_result or "").lower() for marker in ("failed", "error", "exit code: 1", "non-zero"))
            for e in history
        )
        if earlier_success:
            return Verdict(
                action=VerdictAction.NUDGE,
                reason="regression: a previously passing test/lint command now fails",
                guidance=(
                    "This command passed earlier in the session and now fails. "
                    "Treat this as a regression from your own changes: inspect the "
                    "diff since the last passing run before touching anything else."
                ),
                confidence=0.9,
                detector=self.name,
            )
        return None


class OffSpecDetector(Detector):
    """Detect file edits that drift outside the user's stated goal.

    Heuristic: if the user prompt mentions specific files/directories (or the
    session has a declared plan in the prompt), edits to unrelated paths are
    flagged. If no plan is available, this detector stays silent.
    """

    name = "offspec"

    def _mentioned_paths(self, event: AgentEvent) -> set[str]:
        prompt = (event.prompt or "").lower()
        paths: set[str] = set()
        # Very lightweight path extraction from the prompt.
        import re

        for match in re.findall(r"(?:^|\s)([./]?[\w.-]+/(?:[\w.-]+/)*[\w.-]+)", prompt):
            paths.add(match.strip("./").lower())
        return paths

    def _edited_paths(self, event: AgentEvent) -> set[str]:
        if event.event != EventType.TOOL_CALL or event.tool is None:
            return set()
        if event.tool.name not in {"write_file", "edit_file", "str_replace", "patch"}:
            return set()
        path = event.tool.input.get("path") or event.tool.input.get("file_path")
        if not path:
            return set()
        return {str(path).strip("./").lower()}

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        mentioned = self._mentioned_paths(event)
        edited = self._edited_paths(event)
        if not mentioned or not edited:
            return None
        # Flag edits to paths not mentioned in the goal.
        unrelated = edited - mentioned
        if unrelated:
            return Verdict(
                action=VerdictAction.NUDGE,
                reason="off-spec edit: files outside the user's stated goal are being modified",
                guidance=(
                    "You are editing files that are not part of the user's stated "
                    "goal. Re-read the original request and confirm the scope "
                    "before continuing. If the edit is necessary, explain why in "
                    "the next step."
                ),
                confidence=0.75,
                detector=self.name,
            )
        return None


class ContextRotDetector(Detector):
    """Detect context bloat / rot from long reasoning without progress.

    If the agent produces a long reasoning chunk (or a long sequence of
    reasoning chunks) without a tool call or a user-visible result, the session
    is likely spinning. Flag it so the judge can step in.
    """

    name = "contextrot"

    def __init__(self, max_reasoning_chars: int = 4000, max_consecutive_reasoning: int = 4, window: int = 8):
        self.max_reasoning_chars = max_reasoning_chars
        self.max_consecutive_reasoning = max_consecutive_reasoning
        self.window = window

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        if event.event != EventType.REASONING or not event.reasoning:
            return None
        recent = history[-self.window :]
        # Streaming model output produces consecutive reasoning deltas with
        # nothing between them. Only flag "context rot" when the agent has
        # actually gone quiet: no tool activity anywhere in the window.
        if any(e.event in (EventType.TOOL_CALL, EventType.TOOL_RESULT) for e in recent):
            return None
        consecutive = 1  # the current event is itself a reasoning step
        for e in reversed(recent):
            if e.event == EventType.REASONING and e.reasoning:
                consecutive += 1
            else:
                break
        total_chars = sum(len(e.reasoning or "") for e in recent if e.event == EventType.REASONING)
        total_chars += len(event.reasoning or "")
        if consecutive >= self.max_consecutive_reasoning or total_chars > self.max_reasoning_chars:
            return Verdict(
                action=VerdictAction.NUDGE,
                reason="context rot: long reasoning without tool progress",
                guidance=(
                    "You have been reasoning for a while without taking a concrete "
                    "step. Summarize the current state in one sentence, then make "
                    "one small, verifiable tool call to unblock progress."
                ),
                confidence=0.7,
                detector=self.name,
            )
        return None