"""Structured outcome classification for tool results.

The original failure signal was a substring test over the raw result text
(``"error" in result``), which fails in both directions: it calls a successful
``grep error logs/error.log`` a failure, and it calls a timed-out side-effecting
command a success. Literature on non-atomic tool failures
(Verified Tool Calls, arXiv 2608.02645; Did It Happen?) makes the case for a
three-state signal instead of a binary one: when a call's outcome cannot be
observed, that is its own state, not "ok".

So classification prefers *structured* evidence that the adapters carry and
only falls back to text patterns, and the text fallback is anchored to line
starts and real error grammars rather than bare substrings.

States:
  FAILED   — definite failure (non-zero exit, explicit failure hook event)
  OK       — definite success (zero exit, clean structured success)
  UNKNOWN  — neither can be established (timeout with lost response, opaque tool)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..types import AgentEvent, EventType

FAILED = "failed"
OK = "ok"
UNKNOWN = "unknown"

# Structured exit-code keys seen across the supported harnesses.
_EXIT_KEYS = ("exit_code", "exitcode", "returncode", "return_code", "exitCode", "status_code")

# Failure hooks the adapters surface verbatim in metadata.
_FAILURE_HOOKS = ("PostToolUseFailure", "PostToolUseError")

# Line-anchored error grammars. Bare substrings are deliberately excluded:
# "error" appears in perfectly normal output (file names, test names, greps).
_ERROR_PATTERNS = (
    re.compile(r"^traceback \(most recent call last\)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*E\s+\S", re.MULTILINE),  # pytest's E-continuation lines
    re.compile(r"^\s*(?:FAILED|ERROR)\s+\S", re.MULTILINE),  # pytest summary lines
    re.compile(r"\berror TS\d+\b", re.IGNORECASE),
    re.compile(r"\berror\[[A-Za-z0-9_]+\]", re.IGNORECASE),  # rustc
    re.compile(r"^npm ERR!", re.MULTILINE),
    re.compile(r"^error: ", re.MULTILINE),  # git / gcc / cargo
    re.compile(r"\bnon-zero exit status \d+", re.IGNORECASE),
    re.compile(r"\bexit code:?[^\w\n]?[1-9]\d*\b", re.IGNORECASE),
    re.compile(r"\bcommand not found\b", re.IGNORECASE),
    re.compile(r"\bpermission denied\b", re.IGNORECASE),
    re.compile(r"\bno such file or directory\b", re.IGNORECASE),
    re.compile(r"\bis not a directory\b", re.IGNORECASE),
    re.compile(r"^\S+\.py:\d+:\s+\S+Error\b", re.MULTILINE),
    re.compile(r"\bpanic\b.*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bsegmentation fault\b", re.IGNORECASE),
    re.compile(r"\btimed? ?out\b", re.IGNORECASE),
)

# "N failed" is a real failure count; it has to win over "M passed".
_FAILED_COUNT = re.compile(r"\b(\d+) failed\b", re.IGNORECASE)


@dataclass
class Outcome:
    """What we can claim about one tool result."""

    state: str  # FAILED | OK | UNKNOWN
    source: str  # which evidence decided: hook | exit_code | text | none
    detail: str = ""

    @property
    def is_failure(self) -> bool:
        return self.state == FAILED


def _find_exit_code(obj: Any) -> int | None:
    """Pull an exit code out of a structured tool response, if one is present."""
    if isinstance(obj, dict):
        for key in _EXIT_KEYS:
            if key in obj:
                try:
                    return int(obj[key])
                except (TypeError, ValueError):
                    pass
        for value in obj.values():
            code = _find_exit_code(value)
            if code is not None:
                return code
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            code = _find_exit_code(value)
            if code is not None:
                return code
    return None


def _explicit_flag(event: AgentEvent) -> bool | None:
    """A boolean success/failure flag the harness stated outright."""
    meta = event.metadata or {}
    for key in ("is_error", "tool_error", "failed", "error"):
        if key in meta and isinstance(meta[key], bool):
            return bool(meta[key])
    return None


def classify(event: AgentEvent) -> Outcome:
    """Classify one tool-result event into FAILED / OK / UNKNOWN."""
    if event.event != EventType.TOOL_RESULT:
        return Outcome(OK, "not_a_result")

    meta = event.metadata or {}
    hook = str(meta.get("hook_event_name") or "")
    if hook in _FAILURE_HOOKS:
        return Outcome(FAILED, "hook", hook)

    flag = _explicit_flag(event)
    if flag is not None:
        return Outcome(FAILED if flag else OK, "flag", f"is_error={flag}")

    raw = event.tool_result
    # Structured evidence arrives in two places depending on the harness: the
    # adapters lift fields like ``exit_code`` into metadata and leave the text in
    # tool_result, but a response passed through untouched is a dict itself. Both
    # are checked, because reading only one silently reverts this module to
    # guessing from prose.
    code = _find_exit_code(raw)
    if code is None:
        code = _find_exit_code(meta)
    if code is None and isinstance(raw, str):
        # Adapters flatten some responses to text; recover a stated exit code.
        m = re.search(r"['\"]?(?:exit_code|exit code|returncode)['\"]?\s*[:=]\s*(-?\d+)", raw, re.IGNORECASE)
        if m:
            code = int(m.group(1))
    if code is not None:
        return Outcome(FAILED if code != 0 else OK, "exit_code", f"exit={code}")

    text = raw if isinstance(raw, str) else _stringify(raw)
    if not text.strip():
        # An empty result is not evidence of success — it is the case where the
        # response was lost (timeout, dropped side-effecting call).
        return Outcome(UNKNOWN, "empty", "no observable output")
    if any(p.search(text) for p in _ERROR_PATTERNS):
        return Outcome(FAILED, "text", "error grammar matched")
    counts = _FAILED_COUNT.search(text)
    if counts and int(counts.group(1)) > 0:
        return Outcome(FAILED, "text", f"{counts.group(1)} failed")
    # A response that arrived and carries no failure grammar counts as arrived
    # clean. UNKNOWN is reserved for the lost-response case: treating every
    # opaque tool output as unobservable would let the drift statistic accumulate
    # on healthy sessions, which is the false-alarm failure this module exists
    # to prevent.
    return Outcome(OK, "none", "no failure grammar present")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def signal_value(outcome: Outcome) -> float:
    """The 0/0.5/1 value the drift statistic accumulates.

    UNKNOWN contributes a half-step: an agent that keeps getting back nothing
    observable is drifting, but we do not want an opaque tool to alarm as
    loudly as a command that reported a real failure.
    """
    if outcome.state == FAILED:
        return 1.0
    if outcome.state == UNKNOWN:
        return 0.5
    return 0.0
