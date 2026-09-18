"""Tier 0 deterministic detectors.

Each detector is a pure function of the current event plus the recent event
history. Detectors run on every event and return the first Verdict that fires,
or ``None`` if the event looks healthy.

The detectors are deliberately stateless (a function of ``event`` +
``history``); any running statistic (e.g. the CUSUM sum) is recomputed from the
bounded history window so the daemon keeps no per-detector mutable state.
"""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod

from ..types import AgentEvent, EventType, Verdict, VerdictAction

# Markers that indicate a tool result was a failure. Shared by the regression
# and drift detectors so they agree on what "failed" means.
_FAILURE_MARKERS = ("failed", "error", "exception", "traceback", "exit code: 1", "non-zero", "no such file")


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
    """Detect repeated tool calls within a sliding window.

    Two modes, checked in order:

    1. **Exact loop** — 3+ *identical* tool calls (same name + same canonical
       input) in the last ``window`` events. The classic "agent retries the same
       broken command" pattern.
    2. **Near-duplicate loop** — 3+ calls to the *same tool* whose inputs are
       highly similar (token Jaccard >= ``near_dup_threshold``). Catches the
       more common real-world failure: retrying with slightly different
       arguments (a trailing flag, a tweaked path) instead of changing strategy.

    Near-duplicate matching is conservative (high threshold, same tool name
    required, and at least 2 tokens) so legitimately different commands —
    ``echo a`` vs ``echo b`` — are never conflated.
    """

    name = "loop"

    def __init__(self, window: int = 8, threshold: int = 3, near_dup_threshold: float = 0.7):
        self.window = window
        self.threshold = threshold
        # 0.0 disables near-duplicate matching (exact-match only).
        self.near_dup_threshold = near_dup_threshold

    @staticmethod
    def _input_tokens(event: AgentEvent) -> set[str] | None:
        if event.event != EventType.TOOL_CALL or event.tool is None:
            return None
        try:
            s = json.dumps(event.tool.input, sort_keys=True, separators=(",", ":"))
        except TypeError:
            s = repr(event.tool.input)
        return set(re.findall(r"[a-z0-9_./-]+", s.lower()))

    def _loop_verdict(self, tool_name: str, count: int, kind: str) -> Verdict:
        return Verdict(
            action=VerdictAction.NUDGE,
            reason=f"loop detected: tool '{tool_name}' repeated {count} times {kind} in the recent window",
            guidance=(
                "You appear to be repeating the same tool call without new "
                "information. Stop and reassess: what did the last result "
                "actually tell you, and what is the next genuinely different "
                "step toward the goal?"
            ),
            confidence=0.95,
            detector=self.name,
        )

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        sig = _tool_signature(event)
        if sig is None:
            return None
        recent = history[-self.window :]
        # Fast path: exact-match loop.
        exact = 1 + sum(1 for e in recent if _tool_signature(e) == sig)
        if exact >= self.threshold:
            return self._loop_verdict(sig[0], exact, "identical")
        # Near-duplicate loop: same tool, highly similar input.
        if self.near_dup_threshold > 0:
            cur_tokens = self._input_tokens(event)
            if cur_tokens and len(cur_tokens) >= 2:
                near = 1
                for e in recent:
                    other = self._input_tokens(e)
                    if not other or len(other) < 2:
                        continue
                    if e.tool is None or e.tool.name != sig[0]:
                        continue
                    inter = len(cur_tokens & other)
                    union = len(cur_tokens | other)
                    if union and inter / union >= self.near_dup_threshold:
                        near += 1
                if near >= self.threshold:
                    return self._loop_verdict(sig[0], near, "near-identical")
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
        failed = any(marker in result for marker in _FAILURE_MARKERS)
        if not failed:
            return None
        # Was this same tool successful earlier in the session?
        earlier_success = any(
            e.event == EventType.TOOL_RESULT
            and e.tool is not None
            and e.tool.name == event.tool.name
            and not any(marker in (e.tool_result or "").lower() for marker in _FAILURE_MARKERS)
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


# Memoized CUSUM alarm thresholds, keyed by the parameters that determine
# them. The calibration is deterministic (seeded), so the memo is exact.
_CUSUM_THRESHOLD_CACHE: dict[tuple, float] = {}


class CUSUMDriftDetector(Detector):
    """Detect *sustained* drift with a one-sided CUSUM over a cheap failure signal.

    The existing detectors catch sharp, local patterns (an identical call, a
    single regression). They miss the insidious failure the project is named
    for: an agent that slowly accumulates small failures and keeps going. A
    one-sided CUSUM statistic

        S_t = max(0, S_{t-1} + (x_t - baseline - slack))

    where ``x_t`` is a 0/1 "this step failed" signal, accumulates when the
    failure rate stays above baseline and decays when it does not — so it
    alarms on a *run* of failures rather than a single unlucky one.

    The alarm threshold is not a hand-picked constant: it is calibrated by
    Monte-Carlo simulation so that, under the null (a healthy session drawing
    steps from a Bernoulli(``baseline``) failure process), the probability of
    a false alarm within a ``window``-step monitoring window stays at or below
    ``target_fpr``. This is the "calibrate the threshold to a false-alarm
    budget" idea from the online change-point detection literature, which notes
    that a fixed threshold over a moving window is a multiple-testing problem.

    The statistic is recomputed from the bounded history each call (the
    detector keeps no mutable state), so it is a pure function of
    ``(event, history)`` like the other detectors.
    """

    name = "drift"

    def __init__(
        self,
        window: int = 16,
        baseline: float = 0.1,
        slack: float = 0.25,
        target_fpr: float = 0.05,
        seed: int = 1234,
        n_sim: int = 20000,
    ):
        self.window = window
        self.baseline = baseline
        self.slack = slack
        self.target_fpr = target_fpr
        self.seed = seed
        self.n_sim = n_sim
        # The Monte-Carlo calibration is pure and seeded, so identical
        # parameters always yield the same threshold. Cache it so that
        # repeated construction (e.g. in tests) does not re-run the
        # simulation.
        key = (window, baseline, slack, target_fpr, seed, n_sim)
        cached = _CUSUM_THRESHOLD_CACHE.get(key)
        if cached is None:
            cached = self._calibrate(window, baseline, slack, target_fpr, seed, n_sim)
            _CUSUM_THRESHOLD_CACHE[key] = cached
        self.threshold = cached

    def _signal(self, event: AgentEvent) -> float:
        """Cheap 0/1 failure signal: 1.0 if a tool result looks like a failure."""
        if event.event != EventType.TOOL_RESULT:
            return 0.0
        result = (event.tool_result or "").lower()
        return 1.0 if any(m in result for m in _FAILURE_MARKERS) else 0.0

    def _statistic(self, event: AgentEvent, history: list[AgentEvent]) -> float:
        """Recompute the one-sided CUSUM sum over the window ending at ``event``."""
        s = 0.0
        for e in history[-self.window :]:
            s = max(0.0, s + (self._signal(e) - self.baseline - self.slack))
        s = max(0.0, s + (self._signal(event) - self.baseline - self.slack))
        return s

    def _calibrate(
        self, window: int, baseline: float, slack: float, target_fpr: float, seed: int, n_sim: int
    ) -> float:
        """Find the most sensitive alarm threshold that keeps the empirical
        false-alarm rate within ``target_fpr`` over a ``window``-step horizon.

        Under the null each step is an independent Bernoulli(``baseline``)
        failure. We binary-search the smallest threshold whose simulated
        false-alarm rate is at or below the target — i.e. the most sensitive
        detector the budget allows. The RNG is seeded so the threshold is
        deterministic and testable.
        """
        rng = random.Random(seed)

        def false_alarm_rate(h: float) -> float:
            alarms = 0
            for _ in range(n_sim):
                s = 0.0
                for _ in range(window):
                    x = 1.0 if rng.random() < baseline else 0.0
                    s = max(0.0, s + (x - baseline - slack))
                    if s >= h:
                        alarms += 1
                        break
            return alarms / n_sim

        # FPR is monotone decreasing in h, so the "within budget" region is
        # h >= h* for some crossing point h*. Binary-search for the *smallest*
        # h that stays within the budget (the most sensitive detector the
        # budget allows): if mid is within budget, the crossing is at or below
        # mid (search down); otherwise it is above (search up). Return the
        # budget side (hi) so the empirical rate never exceeds the target.
        lo, hi = 0.1, 5.0
        for _ in range(20):
            mid = (lo + hi) / 2.0
            if false_alarm_rate(mid) <= target_fpr:
                hi = mid  # within budget: try a more sensitive (lower) threshold
            else:
                lo = mid  # over budget: must raise the threshold
        return round(hi, 6)

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        s = self._statistic(event, history)
        if s >= self.threshold:
            return Verdict(
                action=VerdictAction.NUDGE,
                reason=(
                    f"drift detected: sustained elevated failure rate "
                    f"(CUSUM statistic {s:.2f} >= threshold {self.threshold:.2f})"
                ),
                guidance=(
                    "Your recent steps have been failing more often than normal. "
                    "Stop making the same kind of call: read the actual error output, "
                    "form one hypothesis about the root cause, and test it with a "
                    "different command before retrying."
                ),
                confidence=0.8,
                detector=self.name,
            )
        return None

    def watch_level(self, event: AgentEvent, history: list[AgentEvent]) -> float:
        """Normalized distance of the CUSUM sum toward the alarm line (0..~1+).

        Exposed so the policy engine can use a *soft* trigger: when the cheap
        statistic is climbing toward (but not yet over) the alarm, wake the
        expensive LLM judge to look. This is a change/onset-based trigger rather
        than an absolute-state threshold, which is what avoids the
        "state-saturation trap" where a fixed-state trigger fires on a constant
        fraction of actions.
        """
        if self.threshold <= 0:
            return 0.0
        return self._statistic(event, history) / self.threshold