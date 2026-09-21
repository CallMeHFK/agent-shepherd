"""Shepherd rulebook: promote the nudges that demonstrably work.

Every event and every verdict is already in the ledger, so the project can
answer a question nothing currently answers: *which nudges actually work?* This
module reads raw ledger records, decides adherence deterministically (no LLM,
no network), and keeps a per-(agent, detector) scorecard of nudges issued,
nudges followed, steps-to-recovery and sessions observed.

Meta-Policy Reflexion (<https://arxiv.org/abs/2509.03990>) argues that
recurring, *effective* guidance belongs in reflective memory — short rules the
agent sees up front — rather than being re-nagged mid-run. The project's own
caveats (<https://arxiv.org/abs/2607.28576>,
<https://arxiv.org/abs/2601.00828>) are that models detect errors far better
than they fix them, so a rule that fired twice, or that fired and was ignored,
is noise. Hence the two floors in :meth:`Rulebook.rules` (minimum evidence,
minimum adherence) and the token budget in :meth:`Rulebook.render_header`: the
book is deliberately short and evidence-ranked, never a dump of everything a
detector has ever said.

Attribution caveat, stated up front: *adherence* here means "the agent changed
behaviour in the way the nudge asked for within ``window`` events". It is a
correlation, not a causal effect — the ledger has no counterfactual. That is
exactly why promotion needs an evidence floor, and why a nudge whose effect the
ledger cannot judge at all is recorded as *inconclusive* and excluded from the
adherence rate instead of being counted for or against.

Two seams are all the policy engine needs, and neither is wired up here (this
module adds analysis only): :func:`adherence_after_nudge`, a pure read of a
closed window that is straight callable on ``Ledger.recent`` output and feeds
:meth:`Rulebook.record_outcome`; and :meth:`Rulebook.render_header`, the block to
prepend at session start (``PROMPT_SUBMIT`` is where the engine already parks
verdicts, so it is the natural gate for it).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..ledger import Ledger, default_root
from ..types import Agent, AgentEvent, EventType, ToolCall, Verdict
from .detectors import (
    ContextRotDetector,
    CUSUMDriftDetector,
    LoopDetector,
    OffSpecDetector,
    RegressionDetector,
)

#: Events the supervisor watches after a nudge before calling it. Six covers a
#: call/result pair plus the next step or two: long enough to see whether the
#: agent actually moved, short enough that the answer still belongs to the
#: nudge rather than to something that happened minutes later.
DEFAULT_WINDOW = 6
#: Default budget for the injected header, in the same rough tokens the policy
#: config counts ``max_guidance_tokens`` in.
DEFAULT_HEADER_TOKENS = 160
#: Characters per token for the dependency-free estimate (no tokenizer is
#: allowed here; 4 chars/token is the standard English approximation).
TOKEN_CHARS = 4
#: A rule line shorter than this is not worth the header space it would take.
MIN_LINE_TOKENS = 8
#: Used when the ledger file's agent is not passed explicitly; the agent lives
#: in the session *path*, not in the record.
UNKNOWN_AGENT = "unknown"
#: Bumped when the persisted shape changes; a stale file is ignored, not migrated.
SCHEMA_VERSION = 1
RULEBOOK_FILENAME = "rulebook.json"

# Mirrors the inline literal in ``OffSpecDetector._edited_paths``; it is not a
# module constant there, and this module must not refactor ``detectors.py``.
_EDIT_TOOLS = frozenset({"write_file", "edit_file", "str_replace", "patch"})

# Read off ``LoopDetector`` rather than restated, so "did the call actually
# change?" and "is this a loop?" use one and the same similarity bar.
_NEAR_DUP_JACCARD = LoopDetector().near_dup_threshold

# How each detector's nudge is answered, operationally:
#   "change" — the next tool call must differ from the offending one, and the
#              offending one must not come back inside the window;
#   "stop"   — the offending tool (or edited path) must not be touched again
#              for the rest of the window, *and* some other call must happen;
#   "act"    — any tool call at all must happen (the agent was spinning).
ADHERENCE_FAMILIES: dict[str, str] = {
    "loop": "change",
    "near_duplicate": "change",
    "drift": "change",
    "regression": "stop",
    "offspec": "stop",
    "binding": "stop",
    "contextrot": "act",
}

# Aliases so a name the ledger carries still resolves to a family/text. The loop
# detector's second mode reports itself as "loop", not "near_duplicate".
DETECTOR_ALIASES: dict[str, str] = {"near_duplicate": "loop", "near-identical": "loop"}

# One-line policy statements. These are *not* the detectors' guidance strings:
# those are two-to-three-sentence mid-run nag texts addressed at a live
# situation, and they are pulled out of ``detectors.py`` unmodified into
# ``Rule.text`` (see :func:`_guidance_for`). A whole rulebook shares ~160
# tokens, so each entry here is a compressed imperative of the same rule.
_LOOP_RULE = "Do not re-run a near-identical call; take a genuinely different step"
RULE_TEXT_BY_DETECTOR: dict[str, str] = {
    "loop": _LOOP_RULE,
    "near_duplicate": _LOOP_RULE,
    "regression": "When a command you already passed starts failing, diff since the last pass",
    "offspec": "Confirm the request's scope before editing a path it never mentions",
    "contextrot": "When reasoning runs long, make one small verifiable tool call",
    "drift": "After repeated failures, read the error and test one new hypothesis",
    "binding": "Tie any claim about the code to a check you actually ran",
}

# Fallback guidance for detector names a ledger may carry that no Tier 0
# detector in this commit emits, so there is nothing to import them from:
# "binding" is the follow-up detector, "judge" is the Tier 1 scorer (whose
# guidance is free-form per call and therefore not promotable as-is).
GUIDANCE_BY_DETECTOR: dict[str, str] = {
    "binding": (
        "You asserted something about the code without binding it to evidence. "
        "Read the file or run the check before describing what the code does."
    ),
    "judge": (
        "The supervisor's scorer flagged this step as off-goal. Re-read the request, "
        "name the evidence for your next action, and take one verifiable step."
    ),
}

HEADER_TITLE = "House rules learned from your past sessions (follow these first):"


# --------------------------------------------------------------------------
# raw-record helpers
#
# The rulebook reads *ledger dicts* (what ``Ledger.recent`` / ``iter_records``
# return), not ``AgentEvent``s, because the ledger is its source of truth and
# replaying through dataclasses would drop records it cannot type. The
# canonicalization below deliberately matches ``detectors._tool_signature`` so
# the two modules agree on what "the same call" means.
# --------------------------------------------------------------------------


def _norm_name(detector: Any) -> str:
    """Canonical detector name; unknown or missing names are kept as-is."""
    name = str(detector or "")
    return DETECTOR_ALIASES.get(name, name)


def _canonical(raw_input: Any) -> str:
    if isinstance(raw_input, Mapping):
        with contextlib.suppress(TypeError, ValueError):
            return json.dumps(raw_input, sort_keys=True, separators=(",", ":"))
    return repr(raw_input)


def _is_event(rec: Mapping[str, Any]) -> bool:
    return rec.get("type") == "event"


def _is_nudge(rec: Mapping[str, Any]) -> bool:
    return rec.get("type") == "verdict" and rec.get("action") == "nudge"


def _is_tool_call(rec: Mapping[str, Any]) -> bool:
    return _is_event(rec) and rec.get("event") == EventType.TOOL_CALL.value


def _is_stop(rec: Mapping[str, Any]) -> bool:
    return _is_event(rec) and rec.get("event") == EventType.STOP.value


def _tool_of(rec: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tool = rec.get("tool") if _is_event(rec) else None
    if isinstance(tool, Mapping) and tool.get("name"):
        return tool
    return None


def _tool_identity(rec: Mapping[str, Any]) -> tuple[str, str] | None:
    """(tool name, canonical input) for any event carrying a tool.

    A tool *result* records carry the same tool + input as the call they answer
    (the adapters do populate it), which is what lets a drift nudge — raised on
    a result — be compared against the agent's next step.
    """
    tool = _tool_of(rec)
    if tool is None:
        return None
    return (str(tool["name"]), _canonical(tool.get("input") or {}))


def _signature(rec: Mapping[str, Any]) -> tuple[str, str] | None:
    """The identity of a *tool call*; ``None`` for anything else."""
    return _tool_identity(rec) if _is_tool_call(rec) else None


def _tokens(canonical: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_./-]+", canonical.lower()))


def _same_call(left: tuple[str, str], right: tuple[str, str]) -> bool:
    """True if two calls are identical or near-identical (tweaked arguments)."""
    if left == right:
        return True
    if left[0] != right[0]:
        return False
    a, b = _tokens(left[1]), _tokens(right[1])
    if len(a) < 2 or len(b) < 2:
        return False
    union = len(a | b)
    return bool(union) and len(a & b) / union >= _NEAR_DUP_JACCARD


def _edited_path(rec: Mapping[str, Any]) -> str | None:
    tool = _tool_of(rec)
    if tool is None or str(tool.get("name")) not in _EDIT_TOOLS:
        return None
    raw = tool.get("input") or {}
    if not isinstance(raw, Mapping):
        return None
    path = raw.get("path") or raw.get("file_path")
    return str(path).strip("./").lower() if path else None


def _stop_subject(rec: Mapping[str, Any]) -> tuple[str, str] | None:
    """What a "stop" nudge asks the agent to leave alone: a path, or a tool."""
    path = _edited_path(rec)
    if path:
        return ("path", path)
    tool = _tool_of(rec)
    return ("tool", str(tool["name"]).lower()) if tool else None


def _touches_subject(rec: Mapping[str, Any], subject: tuple[str, str]) -> bool:
    """Does this record go back to the thing the nudge said to leave alone?

    A path subject matches *edits* only — re-reading the flagged file is not the
    violation the detector named. A tool subject matches the call and its
    result, because in real ledgers a tool result is the proof the tool ran
    again (the call may sit outside the window).
    """
    if subject[0] == "path":
        return _edited_path(rec) == subject[1]
    tool = _tool_of(rec)
    return bool(tool) and str(tool["name"]).lower() == subject[1]


def _in_order(records: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Normalize a record stream to ledger append order.

    Needed because ``Ledger.iter_records`` says "oldest first" but returns
    ``reversed(recent(...))`` — newest first — while ``Ledger.recent`` returns
    append order. Rather than trust either name, take the majority direction of
    the timestamps the ledger stamps on every record and flip once if that
    majority is descending. A single reversal (never a sort) is important:
    verdict records carry wall-clock ``ts`` while events carry the agent's own,
    so sorting by time would split a verdict away from the event it judged and
    break the "offending event = the one right before the verdict" rule.
    """
    if len(records) < 2:
        return records
    stamps = [_float(rec.get("ts")) for rec in records]
    down = sum(1 for a, b in pairwise(stamps) if a > b)
    up = sum(1 for a, b in pairwise(stamps) if a < b)
    return list(reversed(records)) if down > up else records


def _last_event(records: Sequence[Any]) -> Mapping[str, Any] | None:
    """The event that produced a verdict (the ledger appends it immediately before)."""
    for rec in reversed(list(records)):
        if isinstance(rec, Mapping) and _is_event(rec):
            return rec
    return None


def _follow_window(
    records: Iterable[Any], window: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """The next ``window`` *event* records, plus the verdicts interleaved with them.

    Verdicts do not consume the budget (the ledger writes one per event), but a
    repeat nudge inside the window is evidence of non-adherence, so it has to
    travel with the events.
    """
    events: list[Mapping[str, Any]] = []
    verdicts: list[Mapping[str, Any]] = []
    for rec in records:
        if len(events) >= window:
            break
        if not isinstance(rec, Mapping):
            continue
        if _is_event(rec):
            events.append(rec)
        elif rec.get("type") == "verdict":
            verdicts.append(rec)
    return events, verdicts


def _refires(verdicts: Sequence[Mapping[str, Any]], detector: str | None) -> bool:
    """Did the same detector have to say it again inside the window?"""
    wanted = _norm_name(detector)
    return any(_is_nudge(rec) and _norm_name(rec.get("detector")) == wanted for rec in verdicts)


@dataclass(frozen=True)
class Outcome:
    """One judged nudge: did the agent follow it, and after how many events?

    ``decided`` is False when the ledger does not carry enough to judge the
    nudge at all (unknown detector, no offending tool on the triggering event).
    Undecidable samples must not be scored as failures — that would punish a
    detector for the adapter's missing fields.
    """

    adhered: bool = False
    steps: int | None = None
    decided: bool = True


def _inconclusive() -> Outcome:
    return Outcome(adhered=False, steps=None, decided=False)


def _first_real_call(events: Sequence[Mapping[str, Any]]) -> int | None:
    """Index of the first tool call that names a tool.

    ``AgentEvent.tool`` is optional and the ingest path happily accepts a
    tool-less call record, so an empty one is not evidence of work.
    """
    for i, rec in enumerate(events):
        if _is_tool_call(rec) and _tool_of(rec) is not None:
            return i
    return None


def _judge_act(events: Sequence[Mapping[str, Any]]) -> Outcome:
    """Context rot: adherence is simply *doing* something."""
    moved = _first_real_call(events)
    return Outcome(adhered=moved is not None, steps=None if moved is None else moved + 1)


def _judge_stop(offending: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> Outcome:
    subject = _stop_subject(offending)
    if subject is None:
        return _inconclusive()
    moved_on = _first_real_call(events)
    if any(_touches_subject(rec, subject) for rec in events):
        # The offending tool or path reappears anywhere in the window: the rule
        # was not followed, however the agent behaved in between.
        return Outcome(adhered=False)
    if moved_on is None:
        # Nothing happened at all: silence is not evidence the agent moved on.
        return Outcome(adhered=False)
    return Outcome(adhered=True, steps=moved_on + 1)


def _later_recurrence(sig: tuple[str, str], events: Sequence[Mapping[str, Any]]) -> bool:
    """Did the offending call come back after the agent had already moved on?"""
    return any(_same_call(sig, later) for later in map(_signature, events) if later)


def _judge_change(offending: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> Outcome:
    # The offending side may be a tool *result* (that is where the drift
    # detector raises), but the answer must be a tool *call*: the result of the
    # flagged call is still in flight and would otherwise look like a repeat.
    sig = _tool_identity(offending)
    if sig is None:
        # No offending signature means "changed behaviour" is undefined.
        return _inconclusive()
    for i, rec in enumerate(events):
        other = _signature(rec)
        if other is None:
            continue
        if _same_call(sig, other) or _later_recurrence(sig, events[i + 1 :]):
            return Outcome(adhered=False)
        return Outcome(adhered=True, steps=i + 1)
    return Outcome(adhered=False)


def _family_for(nudge_detector: str | None) -> str | None:
    return ADHERENCE_FAMILIES.get(_norm_name(nudge_detector))


def _judge(
    nudge_detector: str | None,
    events_before: Sequence[Any],
    events_after: Sequence[Any],
    window: int = DEFAULT_WINDOW,
) -> Outcome:
    family = _family_for(nudge_detector)
    events, verdicts = _follow_window(events_after, window)
    if _refires(verdicts, nudge_detector):
        # The same detector had to say it again: the first nudge did not land.
        return Outcome(adhered=False) if family else _inconclusive()
    if family is None:
        return _inconclusive()
    offending = _last_event(events_before)
    if family == "act":
        return _judge_act(events)
    if offending is None:
        return _inconclusive()
    if family == "stop":
        return _judge_stop(offending, events)
    return _judge_change(offending, events)


def adherence_after_nudge(
    nudge_detector: str | None,
    events_before: Sequence[Any],
    events_after: Sequence[Any],
    window: int = DEFAULT_WINDOW,
) -> bool:
    """Did the agent follow the nudge that fired at the end of ``events_before``?

    Pure, deterministic, and the piece the policy engine calls: inputs are raw
    ledger records (``Ledger.recent`` output), where the *last* event record in
    ``events_before`` is the offending event the verdict was written for, and
    ``events_after`` is everything observed since (only the next ``window``
    events are read; verdict records may be mixed in).

    Adherence is defined per detector family:

    * ``loop`` / ``near_duplicate`` / ``drift`` — the next tool call's signature
      (name + canonical input) differs from the offending one, the offending
      call does not come back for the rest of the window, and a tweaked-argument
      variant counts as *the same* call (same Jaccard bar as ``LoopDetector``).
    * ``regression`` / ``offspec`` / ``binding`` — the offending tool name (or,
      for an off-spec edit, the offending path) is not touched again for the
      rest of the window *and* at least one other tool call happens.
    * ``contextrot`` — a tool call actually happens inside the window.

    Any nudge from the same detector inside the window means non-adherence, whatever
    the agent did in between. Returns False for samples the ledger cannot judge
    (unknown detector, no tool on the offending event); use :meth:`Rulebook.observe`
    if you need to distinguish "ignored" from "undecidable".
    """
    outcome = _judge(nudge_detector, events_before, events_after, window)
    return outcome.decided and outcome.adhered


# --------------------------------------------------------------------------
# guidance lookup
# --------------------------------------------------------------------------


def _mk_event(
    event: EventType,
    tool: ToolCall | None = None,
    *,
    result: str | None = None,
    reasoning: str | None = None,
    prompt: str | None = None,
    ts: float = 1_000_000.0,
) -> AgentEvent:
    return AgentEvent(
        agent=Agent.QWENPAW,
        session_id="rulebook-fixture",
        event=event,
        ts=ts,
        iteration=1,
        tool=tool,
        tool_result=result,
        reasoning=reasoning,
        prompt=prompt,
    )


# Each detector's guidance is a literal buried in its ``evaluate`` body, so the
# way to read it without duplicating it (or refactoring detectors.py) is to make
# the detector fire on a minimal fixture and take the text off the verdict. If a
# detector's fixture ever stops firing, the lookup degrades to the local maps.
def _fire_loop() -> Verdict | None:
    call = _mk_event(EventType.TOOL_CALL, ToolCall("shell", {"cmd": "pytest -q"}))
    return LoopDetector().evaluate(call, [call, call])


def _fire_regression() -> Verdict | None:
    passed = _mk_event(EventType.TOOL_RESULT, ToolCall("pytest"), result="42 passed")
    failed = _mk_event(EventType.TOOL_RESULT, ToolCall("pytest"), result="2 failed")
    return RegressionDetector().evaluate(failed, [passed])


def _fire_offspec() -> Verdict | None:
    current = _mk_event(
        EventType.TOOL_CALL,
        ToolCall("write_file", {"path": "src/unrelated.py"}),
        prompt="Fix the bug in src/main.py",
    )
    return OffSpecDetector().evaluate(current, [])


def _fire_contextrot() -> Verdict | None:
    detector = ContextRotDetector(max_reasoning_chars=100, max_consecutive_reasoning=4)
    history = [
        _mk_event(EventType.REASONING, reasoning="x" * 30, ts=1_000_000.0 + 30.0 * i)
        for i in range(1, 5)
    ]
    return detector.evaluate(history[-1], history[:-1])


def _fire_drift() -> Verdict | None:
    detector = CUSUMDriftDetector()
    history: list[AgentEvent] = []
    for i in range(4):
        event = _mk_event(
            EventType.TOOL_RESULT,
            ToolCall("shell"),
            result="error: boom",
            ts=1_000_000.0 + 60.0 * i,
        )
        verdict = detector.evaluate(event, history)
        if verdict is not None:
            return verdict
        history.append(event)
    return None


_FIRE_FIXTURES = {
    "loop": _fire_loop,
    "regression": _fire_regression,
    "offspec": _fire_offspec,
    "contextrot": _fire_contextrot,
    "drift": _fire_drift,
}


@cache
def _guidance_for(detector: str) -> str:
    """The detector's own guidance text, read off a verdict it produced.

    ``detectors.py`` keeps its guidance strings inside ``evaluate`` bodies, so
    the only way to reuse them verbatim without refactoring that module is to
    make the detector fire on a fixture and take the text off the verdict. A
    detector whose fixture ever stops firing degrades to the local maps rather
    than raising. Names with no Tier 0 detector here fall back to
    :data:`GUIDANCE_BY_DETECTOR`.
    """
    name = _norm_name(detector)
    fixture = _FIRE_FIXTURES.get(name)
    if fixture is not None:
        with contextlib.suppress(Exception):
            verdict = fixture()
            if verdict is not None and verdict.guidance:
                return " ".join(verdict.guidance.split())
    return GUIDANCE_BY_DETECTOR.get(name) or _rule_text_for(name)


def _rule_text_for(detector: str) -> str:
    """The one-line policy statement; unmapped names fall back to the name itself."""
    return RULE_TEXT_BY_DETECTOR.get(_norm_name(detector), detector)


def _estimate_tokens(text: str) -> int:
    """Rough token count. A real tokenizer is a dependency this module may not take."""
    if not text:
        return 0
    return max(1, round(len(text) / TOKEN_CHARS))


def _block(lines: Sequence[str]) -> str:
    """The header as it would actually be injected, so the budget is measured on
    the bytes the agent sees rather than on a sum of parts."""
    return "\n".join([HEADER_TITLE, *lines])


def _fit_line(line: str, budget_tokens: int) -> str:
    """Word-boundary truncation of a rule line to ``budget_tokens``."""
    chars = budget_tokens * TOKEN_CHARS
    if len(line) <= chars:
        return line
    head = line[: max(0, chars - 3)]
    cut = head.rfind(" ")
    if cut > 16:
        head = head[:cut]
    return head.rstrip(" ,;:-") + "..."


def _agent_value(agent: Agent | str | None) -> str:
    if agent is None:
        return UNKNOWN_AGENT
    if isinstance(agent, Agent):
        return agent.value
    return str(agent)


@dataclass(frozen=True)
class _AgentHandle:
    """Stand-in for an ``Agent`` whose name this build does not know.

    ``Ledger`` only ever reads ``agent.value`` to build the session path, so a
    rulebook can still replay sessions written by a newer build.
    """

    value: str


def _agent_handle(agent: Agent | str) -> Agent | _AgentHandle:
    if isinstance(agent, Agent):
        return agent
    try:
        return Agent(str(agent))
    except ValueError:
        return _AgentHandle(str(agent))


# --------------------------------------------------------------------------
# scorecard
# --------------------------------------------------------------------------


@dataclass
class RuleStat:
    """Running scorecard for one (agent, detector) pair.

    ``nudges`` counts every closed-window sample seen, ``inconclusive`` the ones
    the ledger could not judge; the adherence rate is over ``nudges -
    inconclusive`` so missing adapter fields neither help nor hurt a detector.
    ``recovered_steps`` is the sum of steps-to-recovery over adhered samples
    (divide by ``adhered``; see :attr:`mean_steps_to_recovery`).
    """

    detector: str
    agent: str = UNKNOWN_AGENT
    nudges: int = 0
    adhered: int = 0
    recovered_steps: int = 0
    last_seen_ts: float = 0.0
    inconclusive: int = 0
    sessions: list[str] = field(default_factory=list)

    @property
    def evidence(self) -> int:
        return self.nudges - self.inconclusive

    @property
    def adherence_rate(self) -> float:
        return self.adhered / self.evidence if self.evidence else 0.0

    @property
    def mean_steps_to_recovery(self) -> float:
        return self.recovered_steps / self.adhered if self.adhered else 0.0

    @property
    def sessions_observed(self) -> int:
        return len(self.sessions)

    def record(
        self,
        *,
        adhered: bool,
        steps: int | None = None,
        inconclusive: bool = False,
        session_id: str | None = None,
        ts: float = 0.0,
    ) -> None:
        self.nudges += 1
        if inconclusive:
            self.inconclusive += 1
        elif adhered:
            self.adhered += 1
            if steps:
                self.recovered_steps += int(steps)
        self.last_seen_ts = max(self.last_seen_ts, float(ts or 0.0))
        if session_id and session_id not in self.sessions:
            self.sessions.append(session_id)
            self.sessions.sort()

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "agent": self.agent,
            "nudges": self.nudges,
            "adhered": self.adhered,
            "recovered_steps": self.recovered_steps,
            "last_seen_ts": self.last_seen_ts,
            "inconclusive": self.inconclusive,
            "sessions": list(self.sessions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuleStat | None:
        detector = data.get("detector")
        if not isinstance(detector, str):
            return None
        sessions = data.get("sessions")
        return cls(
            detector=detector,
            agent=_agent_value(data.get("agent")),
            nudges=_nonneg_int(data.get("nudges")),
            adhered=_nonneg_int(data.get("adhered")),
            recovered_steps=_nonneg_int(data.get("recovered_steps")),
            last_seen_ts=_float(data.get("last_seen_ts")),
            inconclusive=_nonneg_int(data.get("inconclusive")),
            sessions=_str_list(sessions),
        )


def _str_list(raw: Any) -> list[str]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        return []
    return sorted({str(item) for item in raw})


def _nonneg_int(raw: Any) -> int:
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _float(raw: Any) -> float:
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class Rule:
    """A promoted, evidence-backed rule. ``headline`` is the injected one-liner,
    ``text`` the detector's own (longer) guidance for callers that want it."""

    detector: str
    agent: str
    headline: str
    text: str
    evidence: int
    adherence: float
    mean_steps_to_recovery: float
    sessions_observed: int
    score: float

    def render_line(self) -> str:
        return (
            f"- {self.headline} (observed {self.evidence}x, "
            f"{round(self.adherence * 100)}% followed)."
        )


def _discover_sessions(ledger: Ledger, name: str) -> list[str]:
    """Session ids recorded for an agent — the ledger encodes them in filenames."""
    directory = ledger.root / "sessions" / name
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.jsonl"))


@dataclass
class Rulebook:
    """Evidence-ranked rulebook built from the ledger."""

    window: int = DEFAULT_WINDOW
    max_rules: int = 3
    stats: dict[tuple[str, str], RuleStat] = field(default_factory=dict)
    # Per-session record counts already consumed, so repeated observe() calls on
    # a growing ledger are incremental and never double-count a nudge.
    cursors: dict[str, int] = field(default_factory=dict)

    # -- scoring ----------------------------------------------------------

    def _stat(self, detector: str, agent: str) -> RuleStat:
        key = (agent, detector)
        stat = self.stats.get(key)
        if stat is None:
            stat = RuleStat(detector=detector, agent=agent)
            self.stats[key] = stat
        return stat

    def record_outcome(
        self,
        detector: str,
        *,
        adhered: bool,
        agent: Agent | str = UNKNOWN_AGENT,
        steps: int | None = None,
        inconclusive: bool = False,
        session_id: str | None = None,
        ts: float | None = None,
    ) -> None:
        """Persist one adherence judgement.

        This is the engine-facing hook: it calls :func:`adherence_after_nudge` on
        a closed window and reports the result here, so the scorecard survives
        restarts via :meth:`save` without the engine re-reading old sessions.
        """
        self._stat(detector, _agent_value(agent)).record(
            adhered=adhered,
            steps=steps,
            inconclusive=inconclusive,
            session_id=session_id,
            ts=time.time() if ts is None else float(ts),
        )

    def _closed(self, stream: Sequence[Mapping[str, Any]], index: int) -> bool:
        """Is the sample at ``index`` finished — a full window, or the run ended?"""
        events, _ = _follow_window(stream[index + 1 :], self.window)
        return len(events) >= self.window or any(_is_stop(rec) for rec in events)

    def observe(
        self,
        ledger_records: Iterable[Any],
        agent: Agent | str = UNKNOWN_AGENT,
        session_id: str | None = None,
    ) -> int:
        """Update the scorecard from one session's raw ledger records.

        ``agent`` has to be passed in because the ledger stores it in the *path*,
        never in the record. Records may arrive in either ledger order — see
        :func:`_in_order` — and are normalized to append order here. Pass the
        session's whole stream (what ``Ledger.iter_records`` returns), not a
        bounded tail slice: the cursor indexes into the stream it was given, so
        a sliding slice would make it skip records. Nudges whose window has not
        closed yet are left unconsumed and judged on the next call, so feeding
        this the same growing stream twice counts each nudge exactly once.
        Returns the number of samples newly recorded.
        """
        stream = _in_order([rec for rec in ledger_records if isinstance(rec, Mapping)])
        name = _agent_value(agent)
        cursor_key = f"{name}:{session_id or '-'}"
        judged = 0
        index = self.cursors.get(cursor_key, 0)
        while index < len(stream):
            rec = stream[index]
            if not _is_nudge(rec):
                index += 1
                continue
            if not self._closed(stream, index):
                break
            outcome = _judge(rec.get("detector"), stream[:index], stream[index + 1 :], self.window)
            self._stat(_norm_name(rec.get("detector")) or "unknown", name).record(
                adhered=outcome.adhered,
                steps=outcome.steps,
                inconclusive=not outcome.decided,
                session_id=session_id,
                ts=_float(rec.get("ts")),
            )
            judged += 1
            index += 1
        self.cursors[cursor_key] = min(index, len(stream))
        return judged

    @classmethod
    def from_ledger(
        cls,
        ledger: Ledger,
        agent: Agent | str,
        session_ids: Iterable[str] | None = None,
        *,
        window: int = DEFAULT_WINDOW,
    ) -> Rulebook:
        """Build a rulebook by replaying ledger sessions.

        ``session_ids=None`` discovers every session recorded for that agent,
        which is what the daemon wants at startup: the book is only as good as
        the evidence behind it, and evidence is spread over sessions.
        """
        book = cls(window=window)
        name = _agent_value(agent)
        handle = _agent_handle(agent)
        ids = _discover_sessions(ledger, name) if session_ids is None else session_ids
        for session_id in ids:
            records = ledger.iter_records(handle, str(session_id))
            book.observe(records, agent=name, session_id=str(session_id))
        return book

    # -- promotion --------------------------------------------------------

    def _rule(self, stat: RuleStat) -> Rule:
        # adherence x log(evidence): a rule needs to work *and* recur. log1p
        # keeps the ranking monotone and non-zero even if a caller drops
        # ``min_evidence`` below 1, so an almost-empty book still orders sanely.
        score = stat.adherence_rate * math.log1p(stat.evidence)
        return Rule(
            detector=stat.detector,
            agent=stat.agent,
            headline=_rule_text_for(stat.detector),
            text=_guidance_for(stat.detector),
            evidence=stat.evidence,
            adherence=stat.adherence_rate,
            mean_steps_to_recovery=stat.mean_steps_to_recovery,
            sessions_observed=stat.sessions_observed,
            score=round(score, 6),
        )

    def rules(
        self,
        min_evidence: int = 3,
        min_adherence: float = 0.5,
        *,
        agent: Agent | str | None = None,
    ) -> list[Rule]:
        """Promoted rules, best first: enough evidence, good enough adherence.

        The floors are the admissibility check Meta-Policy Reflexion insists on —
        an agent's up-front context is scarce, so a rarely-observed or
        poorly-followed detector never gets in. When ``agent`` is None the best
        entry per detector wins, so a shared header never says the same rule
        twice.
        """
        wanted = _agent_value(agent) if agent is not None else None
        best: dict[str, Rule] = {}
        for stat in self.stats.values():
            if wanted is not None and stat.agent != wanted:
                continue
            if stat.evidence < min_evidence or stat.adherence_rate < min_adherence:
                continue
            rule = self._rule(stat)
            current = best.get(rule.detector)
            if current is None or (rule.score, rule.agent) > (current.score, current.agent):
                best[rule.detector] = rule
        return sorted(best.values(), key=lambda r: (-r.score, r.detector, r.agent))

    def render_header(self, budget_tokens: int = DEFAULT_HEADER_TOKENS) -> str | None:
        """The injectable "house rules" block, cut to ``budget_tokens``.

        Prepended at session start so the agent sees the rule before it drifts,
        which is the whole point: a nudge that has to be re-issued every session
        is a rule that was never read. ``None`` when nothing qualifies — the
        engine then injects nothing rather than an empty heading.
        """
        rules = self.rules()[: max(0, self.max_rules)]
        if not rules:
            return None
        lines: list[str] = []
        for rule in rules:
            line = rule.render_line()
            left = budget_tokens - _estimate_tokens(_block(lines))
            if _estimate_tokens(line) > left:
                if left < MIN_LINE_TOKENS:
                    break
                lines.append(_fit_line(line, left))
                break
            lines.append(line)
        if not lines:
            return None
        return _block(lines)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "window": self.window,
            "max_rules": self.max_rules,
            "stats": [self.stats[key].to_dict() for key in sorted(self.stats)],
            "cursors": {key: self.cursors[key] for key in sorted(self.cursors)},
        }

    @classmethod
    def from_dict(cls, data: Any) -> Rulebook:
        """Rebuild from :meth:`to_dict` output; anything malformed yields an empty book."""
        book = cls()
        if not isinstance(data, Mapping) or data.get("version") != SCHEMA_VERSION:
            return book
        book.window = _nonneg_int(data.get("window")) or DEFAULT_WINDOW
        book.max_rules = _nonneg_int(data.get("max_rules")) or book.max_rules
        raw_stats = data.get("stats")
        if isinstance(raw_stats, Iterable) and not isinstance(raw_stats, (str, bytes)):
            for raw in raw_stats:
                if not isinstance(raw, Mapping):
                    continue
                stat = RuleStat.from_dict(raw)
                if stat is not None:
                    book.stats[(stat.agent, stat.detector)] = stat
        cursors = data.get("cursors")
        if isinstance(cursors, Mapping):
            book.cursors = {str(k): _nonneg_int(v) for k, v in cursors.items()}
        return book

    def save(self, path: Path | str | None = None) -> bool:
        """Write the book atomically to ``$SHEPHERD_HOME/rulebook.json``.

        Returns whether the write happened. Failures are swallowed on purpose:
        the ledger is the source of truth and the book is a recomputable cache,
        so a read-only home directory must never wedge a supervised session.
        """
        target = Path(path) if path else rulebook_path()
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        tmp: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target.parent), prefix=".rulebook-", suffix=".json"
            )
            tmp = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
            os.replace(tmp, target)
            tmp = None
            return True
        except OSError:
            return False
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path | str | None = None, *, window: int | None = None) -> Rulebook:
        """Read the persisted book; missing or corrupt yields a fresh one.

        Corrupt is the normal state of this file in the field — it is written by
        whatever build the user last ran — so parsing never raises: an empty book
        simply rebuilds its evidence from the next sessions it observes.
        """
        target = Path(path) if path else rulebook_path()
        try:
            with open(target, encoding="utf-8") as fh:
                book = cls.from_dict(json.load(fh))
        except (OSError, ValueError):
            return cls(window=window or DEFAULT_WINDOW)
        if window:
            book.window = window
        return book


def rulebook_path() -> Path:
    """``$SHEPHERD_HOME/rulebook.json`` — same home the ledger and config use."""
    return default_root() / RULEBOOK_FILENAME
