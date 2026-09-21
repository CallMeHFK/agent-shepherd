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
from fnmatch import fnmatch
from pathlib import Path

from ..types import AgentEvent, EventType, Verdict, VerdictAction
from . import signals

_EDIT_TOOLS = {"write_file", "edit_file", "str_replace", "patch", "Write", "Edit", "MultiEdit", "NotebookEdit"}
_PATH_KEYS = ("path", "file_path", "notebook_path", "filename", "file")



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
        if signals.classify(event).state != signals.FAILED:
            return None
        # Was this same tool successful earlier in the session?
        earlier_success = any(
            e.event == EventType.TOOL_RESULT
            and e.tool is not None
            and e.tool.name == event.tool.name
            and signals.classify(e).state == signals.OK
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
    """Detect edits outside the delegation contract for this session.

    The earlier version regexed paths out of the user prompt and flagged any
    edit not in that set — which both missed real scope violations (the user
    rarely types every file) and fired on legitimate work (you cannot fix a
    module without touching a file nobody named).

    This version models the contract explicitly, following the delegation-contract
    framing (arXiv 2606.17099) and the deterministic-compliance-check result that
    an SMT/solver-side admissibility check beats asking a model (arXiv
    2603.20449): deny-patterns are checked first and hard-BLOCKed.

    ``nudge_unobserved`` additionally flags edits to a path nothing in the
    session named, allowed or read. It defaults to **off** because the offline
    benchmark measured it at one false nudge per healthy session (the offline
    benchmark, ``shepherd eval``): creating a new test file, or editing a file
    found by a repo-wide grep, is normal competent work. Novelty is not the
    same thing as exceeding one's authority, and a detector that costs the
    agent a nudge per healthy session is the saturation failure the whole
    design exists to avoid.
    """

    name = "offspec"

    def __init__(
        self,
        allow_globs: list[str] | None = None,
        deny_globs: list[str] | None = None,
        nudge_unobserved: bool = False,
    ):
        self.allow_globs = allow_globs or []
        self.deny_globs = deny_globs or []
        self.nudge_unobserved = nudge_unobserved

    def _mentioned_paths(self, event: AgentEvent) -> set[str]:
        prompt = event.prompt or ""
        paths: set[str] = set()
        for match in re.findall(r"(?:^|\s)([./]?[\w.\-]+/(?:[\w.\-]+/)*[\w.\-]+)", prompt):
            paths.add(match.strip("./").lower())
        # Also accept bare filenames ("update config.py"), which prompts use often.
        for match in re.findall(r"\b([\w.\-]+\.[a-zA-Z]{1,5})\b", prompt):
            paths.add(match.lower())
        return paths

    def _edited_path(self, event: AgentEvent) -> str | None:
        if event.event != EventType.TOOL_CALL or event.tool is None:
            return None
        if event.tool.name not in _EDIT_TOOLS:
            return None
        for key in _PATH_KEYS:
            value = event.tool.input.get(key)
            if value:
                return _clean_path(value)
        return None

    def _observed_paths(self, history: list[AgentEvent]) -> set[str]:
        """Paths the agent has already seen results for — in-play, not out of scope."""
        seen: set[str] = set()
        for e in history:
            if e.tool is None:
                continue
            for key in _PATH_KEYS:
                value = e.tool.input.get(key)
                if value:
                    seen.add(_clean_path(value))
            for token in re.findall(r"[\w.\-]+/[\w./\-]+", e.tool_result or ""):
                seen.add(_clean_path(token))
        return seen

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        edited = self._edited_path(event)
        if not edited:
            return None
        if any(fnmatch(edited, g) or fnmatch(f"/{edited}", g) for g in self.deny_globs):
            return Verdict(
                action=VerdictAction.BLOCK,
                reason=f"out-of-contract edit: '{edited}' matches a configured deny-pattern",
                guidance=(
                    f"'{edited}' is explicitly outside the permitted scope for this "
                    "session. Do not edit it; ask the user if the scope needs to change."
                ),
                confidence=0.95,
                detector=self.name,
            )
        in_scope = (
            edited in self._mentioned_paths(event)
            or edited in self._observed_paths(history)
            or any(fnmatch(edited, g) or fnmatch(f"/{edited}", g) for g in self.allow_globs)
        )
        if in_scope or not self.nudge_unobserved:
            return None
        # Nothing named it, nothing read it, nothing allows it.
        return Verdict(
            action=VerdictAction.NUDGE,
            reason=(
                f"off-spec edit: '{edited}' is outside the stated goal and was never "
                "read or referenced this session"
            ),
            guidance=(
                f"You are about to edit '{edited}', which the request never named and "
                "which you have not read or seen referenced in this session. Re-read the "
                "original request: either confirm in one line why this file is required, "
                "or go back to the file the goal actually names."
            ),
            confidence=0.75,
            detector=self.name,
        )


class BindingDriftDetector(Detector):
    """Detect acting on a near-miss sibling of the entity that was resolved.

    *Binding Drift in Multi-Step Tool-Augmented Agents* (arXiv 2607.18316)
    reports that agents pick the right tool but bind it to the wrong entity a
    large fraction of the time, and that the error is *silent*: the call succeeds.
    A loop detector cannot see it (the arguments genuinely differ) and neither
    can an off-spec check (the path may be in scope).

    The cheap deterministic signature: an edit targeting a path that is highly
    similar to a path the agent actually inspected, but not that path, with no
    new observation in between establishing the switch. ``cat config.py`` then
    ``edit config.py.bak`` is exactly this.
    """

    name = "binding"

    def __init__(self, similarity: float = 0.6, lookback: int = 24):
        self.similarity = similarity
        self.lookback = lookback

    @staticmethod
    def _tokens(path: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", path.lower()) if len(t) > 1}

    def _sim(self, a: str, b: str) -> float:
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        if event.event != EventType.TOOL_CALL or event.tool is None:
            return None
        if event.tool.name not in _EDIT_TOOLS:
            return None
        target = next((_clean_path(event.tool.input[k]) for k in _PATH_KEYS if event.tool.input.get(k)), None)
        if not target:
            return None
        recent = history[-self.lookback :]
        inspected: set[str] = set()
        for e in recent:
            if e.tool is None:
                continue
            if e.event == EventType.TOOL_RESULT or e.tool.name not in _EDIT_TOOLS:
                for k in _PATH_KEYS:
                    if e.tool.input.get(k):
                        inspected.add(_clean_path(e.tool.input[k]))
            if e.event == EventType.TOOL_RESULT and e.tool_result:
                for token in re.findall(r"[\w.\-]+/[\w./\-]+|[\w.\-]+\.[a-zA-Z]{1,5}", e.tool_result):
                    inspected.add(_clean_path(token))
        if target in inspected:
            return None
        for seen in inspected:
            if seen == target or self._sim(seen, target) < self.similarity:
                continue
            return Verdict(
                action=VerdictAction.NUDGE,
                reason=(
                    f"binding drift: editing '{target}' but the session established '{seen}' "
                    "(similar name, never inspected)"
                ),
                guidance=(
                    f"You inspected '{seen}' but are now editing '{target}', a different file "
                    "with a very similar name. Confirm which one the task means, and read the "
                    "one you are about to change before changing it."
                ),
                confidence=0.8,
                detector=self.name,
            )
        return None


def _clean_path(value: object) -> str:
    path = str(value).strip().strip("'\"").lower()
    return path.removeprefix("./")


class ContextRotDetector(Detector):
    """Detect context bloat / rot from long reasoning without progress.

    If the agent produces a long reasoning chunk (or a long sequence of
    reasoning chunks) without a tool call or a user-visible result, the session
    is likely spinning. Flag it so the judge can step in.

    "Rot" is a property of wall-clock time, not of event count: a model can
    stream several quick reasoning deltas in a few seconds while thinking
    normally. The detector therefore requires the agent to have been *quiet*
    (no tool activity observed) for at least ``min_quiet_seconds`` before it
    will flag. Without this, a state-threshold detector saturates — it fires
    on a constant fraction of events for as long as the quiet stretch lasts
    (the "saturation trap"), which in live operation produced tens of
    thousands of identical nudges per session.
    """

    name = "contextrot"

    def __init__(
        self,
        max_reasoning_chars: int = 4000,
        max_consecutive_reasoning: int = 4,
        window: int = 8,
        min_quiet_seconds: float = 60.0,
        quiet_lookback: int = 64,
    ):
        self.max_reasoning_chars = max_reasoning_chars
        self.max_consecutive_reasoning = max_consecutive_reasoning
        self.window = window
        self.min_quiet_seconds = min_quiet_seconds
        self.quiet_lookback = quiet_lookback

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        if event.event != EventType.REASONING or not event.reasoning:
            return None
        recent = history[-self.window :]
        # Streaming model output produces consecutive reasoning deltas with
        # nothing between them. Only flag "context rot" when the agent has
        # actually gone quiet: no tool activity anywhere in the window.
        if any(e.event in (EventType.TOOL_CALL, EventType.TOOL_RESULT) for e in recent):
            return None
        # The quiet stretch must be sustained in wall-clock time. Measure from
        # the most recent tool activity within the lookback (or the start of
        # the lookback if there was none there — the span is then a lower
        # bound on the true quiet time).
        lookback = history[-self.quiet_lookback :]
        tool_ts = [e.ts for e in lookback if e.event in (EventType.TOOL_CALL, EventType.TOOL_RESULT)]
        quiet_start = max(tool_ts) if tool_ts else (lookback[0].ts if lookback else event.ts)
        if event.ts - quiet_start < self.min_quiet_seconds:
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


def _threshold_cache_path() -> Path:
    from ..ledger import default_root

    return default_root() / "cusum_thresholds.json"


def _read_threshold_cache() -> dict[str, float]:
    """Persisted mirror of the in-process memo.

    The calibration is ~24 Monte-Carlo passes x 20k sessions x a 120-step
    horizon. That is fine once and unacceptable every time the daemon starts:
    for the ~15s it took, every hook call failed open and the supervisor was
    silently absent. Because the fit is seeded and deterministic, the answer can
    be cached on disk and reused verbatim.
    """
    path = _threshold_cache_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _write_threshold_cache(cache: dict[str, float]) -> None:
    path = _threshold_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass  # a cold start is allowed to be slow; it must not be fatal


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
        horizon: int = 120,
        unknown_weight: float = 0.75,
        adaptive_baseline: bool = True,
        min_baseline_samples: int = 8,
    ):
        self.window = window
        self.baseline = baseline
        self.slack = slack
        self.target_fpr = target_fpr
        self.seed = seed
        self.n_sim = n_sim
        self.unknown_weight = unknown_weight
        # The alarm line is calibrated for a monitoring window, but the engine
        # re-tests it on every step, so the *session-level* false-alarm rate is
        # what an agent actually experiences. Calibrating over a horizon of
        # overlapping windows is the fix for the caveat the earlier design
        # accepted (see "When Drift Detectors cry Wolf", arXiv 2607.17336).
        self.horizon = max(horizon, window)
        self.adaptive_baseline = adaptive_baseline
        self.min_baseline_samples = min_baseline_samples
        key = (self.horizon, baseline, slack, target_fpr, seed, n_sim, unknown_weight)
        disk_key = json.dumps([str(x) for x in key])
        cached = _CUSUM_THRESHOLD_CACHE.get(key)
        if cached is None:
            cached = _read_threshold_cache().get(disk_key)
            if cached is not None:
                _CUSUM_THRESHOLD_CACHE[key] = cached
        if cached is None:
            cached = self._calibrate(
                self.horizon, window, baseline, slack, target_fpr, seed, n_sim, unknown_weight
            )
            _CUSUM_THRESHOLD_CACHE[key] = cached
            disk = _read_threshold_cache()
            disk[disk_key] = cached
            _write_threshold_cache(disk)
        self.threshold = cached

    def _baseline_for(self, reference: list[AgentEvent]) -> float:
        """Self-starting baseline: the session's own pre-alarm failure rate.

        A fixed ``baseline`` assumes every agent runs in an equally flaky
        environment, which is false — one agent on a clean repo and one fighting
        a broken sandbox get different CUSUMs from the same behavior. Once
        enough results have been seen, the running mean replaces the prior (the
        self-starting CUSUM idea, arXiv 2410.12736 and 2509.07112).

        The mean is taken over the *reference* history only — everything the
        session showed before the current monitoring window. Estimating it from
        the window under alarm is the classic self-normalization trap: a run of
        pure failures drives the baseline to 1.0, the reference drift to
        ``1.0 + slack``, and the statistic can never alarm again.
        """
        if not self.adaptive_baseline:
            return self.baseline
        observed = [self._signal(e) for e in reference if e.event == EventType.TOOL_RESULT]
        if len(observed) < self.min_baseline_samples:
            return self.baseline
        mean = sum(observed) / len(observed)
        # Never let the baseline collapse to 0: a stretch of clean results would
        # make the detector scream at the first failure.
        return max(mean, self.baseline / 2.0)

    def _signal(self, event: AgentEvent) -> float:
        """Graded failure signal from the structured outcome classifier."""
        outcome = signals.classify(event)
        if outcome.state == signals.FAILED:
            return 1.0
        if outcome.state == signals.UNKNOWN:
            return self.unknown_weight
        return 0.0

    def _statistic(self, event: AgentEvent, history: list[AgentEvent]) -> float:
        """Recompute the one-sided CUSUM sum over the *results* in the window.

        Only tool results carry signal, so only they may be summed. Folding the
        interleaved calls and reasoning chunks too — each contributing ``0.0 -
        slack`` — makes a session that never gets a response back undetectable:
        the decay from the calls outruns the +0.5 credit from the empty results
        and the statistic can never reach its alarm line. That is exactly the
        non-atomic-failure case this signal was widened to cover.
        """
        window = history[-self.window :]
        reference = history[: -self.window] if len(history) > self.window else []
        base = self._baseline_for(reference)
        slack = self.slack + base
        s = 0.0
        for e in window:
            if e.event != EventType.TOOL_RESULT:
                continue
            s = max(0.0, s + (self._signal(e) - slack))
        if event.event == EventType.TOOL_RESULT:
            s = max(0.0, s + (self._signal(event) - slack))
        return s

    def _calibrate(
        self,
        horizon: int,
        window: int,
        baseline: float,
        slack: float,
        target_fpr: float,
        seed: int,
        n_sim: int,
        unknown_weight: float,
    ) -> float:
        """Find the most sensitive alarm threshold whose false-alarm rate over a
        whole ``horizon``-step session stays at or below ``target_fpr``.

        The null model is the graded one the detector actually sees: each step is
        a failure with probability ``baseline``, an unobservable outcome with
        probability ``unknown_rate``, and otherwise clean. Calibration runs over
        the full horizon with the statistic re-tested at every step, which is
        why the resulting threshold is higher than a single-window calibration
        would give — and why the long-run alarm budget finally means something.
        """
        rng = random.Random(seed)
        unknown_rate = max(0.02, min(0.12, baseline))
        drift = baseline + slack

        def false_alarm_rate(h: float) -> float:
            alarms = 0
            for _ in range(n_sim):
                s = 0.0
                for _ in range(horizon):
                    u = rng.random()
                    if u < baseline:
                        x = 1.0
                    elif u < baseline + unknown_rate:
                        x = unknown_weight
                    else:
                        x = 0.0
                    s = max(0.0, s + (x - drift))
                    if s >= h:
                        alarms += 1
                        break
            return alarms / n_sim

        # FPR is monotone decreasing in h, so the "within budget" region is
        # h >= h* for some crossing point h*. Binary-search the *smallest* h that
        # fits the budget (the most sensitive detector the budget allows) and
        # return the budget side, so the empirical rate never exceeds the target.
        lo, hi = 0.1, 20.0
        for _ in range(24):
            mid = (lo + hi) / 2.0
            if false_alarm_rate(mid) <= target_fpr:
                hi = mid
            else:
                lo = mid
        return round(hi, 6)

    def evaluate(self, event: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        s = self._statistic(event, history)
        if s >= self.threshold:
            window = history[-self.window :] + [event]
            outcomes = [signals.classify(e) for e in window if e.event == EventType.TOOL_RESULT]
            failing = [o for o in outcomes if o.state != signals.OK]
            # "Read the error output" is worthless advice when nothing came back;
            # an unobservable outcome needs a side-effect check, not a rerun.
            mostly_unknown = bool(failing) and sum(
                1 for o in failing if o.state == signals.UNKNOWN
            ) > len(failing) / 2
            guidance = (
                "Recent calls returned nothing observable, so you do not know whether "
                "their effects happened. Before retrying anything that writes or sends, "
                "check the world state (read the file, list the directory, query the "
                "record) and act on what you find."
                if mostly_unknown
                else (
                    "Your recent steps have been failing more often than normal. "
                    "Stop making the same kind of call: read the actual error output, "
                    "form one hypothesis about the root cause, and test it with a "
                    "different command before retrying."
                )
            )
            return Verdict(
                action=VerdictAction.NUDGE,
                reason=(
                    f"drift detected: sustained elevated failure rate "
                    f"(CUSUM statistic {s:.2f} >= threshold {self.threshold:.2f})"
                ),
                guidance=guidance,
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