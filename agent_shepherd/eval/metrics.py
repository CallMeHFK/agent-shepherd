"""Scoring the supervisor's verdicts against step-level ground truth.

The metrics are deliberately unforgiving about attribution, in the spirit of the
benchmarks this repo's design borrows: a monitor that detects the right *thing*
at the wrong *time*, or blames the wrong component, is not yet worth its cost
(trajectory-judge arXiv 2609.00038; ToolRobustBench arXiv 2608.23635 both score
monitors against known step indices rather than session-level outcomes).

Definitions used here, stated once so a reader of the report cannot be misled:

*step* — one ingested event. **Delay** is
``first qualifying verdict index - injected onset``, in steps, which is the
only quantity that distinguishes a supervisor from a post-mortem.

**First detection wins** attribution: the *first* non-PASS verdict of each
detector in a session is judged — it is a true positive for a fault case iff it
either names the detector that owns the fault (any time at or after the onset),
or it names *some* detector within ``tolerance`` steps of the onset (drift was
flagged in time, but credited to the wrong component). Every other first
verdict — before the onset, late and misattributed, or any verdict on a clean
session — is a false positive. Later verdicts from an already-judged detector
are **repeats**: not new alarms, but the nagging the cooldown exists to bound, so
they are counted in their own column rather than folded into the false-alarm
rate.

**Recall** is per case (did this fault get a true positive at all); **precision**
is per verdict (of the verdicts this detector/fault produced, how many were true
positives). Mixing the two units is how monitor evaluations quietly inflate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any

from ..core.types import VerdictAction

PASS = VerdictAction.PASS.value


@dataclass(frozen=True)
class StepVerdict:
    """One engine answer, pinned to the event index that provoked it."""

    index: int
    event: str
    action: str
    detector: str | None = None
    reason: str = ""

    @property
    def fired(self) -> bool:
        return self.action != PASS


@dataclass
class Cost:
    """What one supervised session cost the agent.

    Tokens are estimated at ~4 characters per token — the same convention
    ``judge.risk.truncate_guidance`` uses for its guidance budget — because the
    point is a comparable budget, not an invoice.
    """

    events_ingested: int = 0
    judge_wakes: int = 0
    judge_tokens: int = 0
    nudges_emitted: int = 0
    nudges_suppressed: int = 0

    def merge(self, other: Cost) -> Cost:
        return Cost(
            self.events_ingested + other.events_ingested,
            self.judge_wakes + other.judge_wakes,
            self.judge_tokens + other.judge_tokens,
            self.nudges_emitted + other.nudges_emitted,
            self.nudges_suppressed + other.nudges_suppressed,
        )


@dataclass
class CaseScore:
    """Ground truth, verdicts and cost for one generated session."""

    fault: str
    session_id: str
    owner: str | None
    onset: int | None
    is_drift: bool
    answerable: bool
    requires: tuple[str, ...] = ()
    steps: int = 0
    verdicts: list[StepVerdict] = field(default_factory=list)
    true_positives: list[StepVerdict] = field(default_factory=list)
    false_positives: list[StepVerdict] = field(default_factory=list)
    repeats: list[StepVerdict] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)

    @property
    def detected(self) -> bool:
        return bool(self.true_positives)

    @property
    def delay(self) -> int | None:
        if not self.true_positives or self.onset is None:
            return None
        return self.true_positives[0].index - self.onset


def attribute(
    verdicts: list[StepVerdict],
    *,
    onset: int | None,
    owner: str | None,
    is_drift: bool,
    tolerance: int = 8,
) -> tuple[list[StepVerdict], list[StepVerdict], list[StepVerdict]]:
    """Split a session's non-PASS verdicts into true, false and repeats."""
    tp: list[StepVerdict] = []
    fp: list[StepVerdict] = []
    repeats: list[StepVerdict] = []
    judged: set[str | None] = set()
    for verdict in (v for v in verdicts if v.fired):
        if verdict.detector in judged:
            repeats.append(verdict)
            continue
        judged.add(verdict.detector)
        if not is_drift or onset is None:
            fp.append(verdict)  # nothing to credit: every intervention is a cost
        elif verdict.index < onset:
            fp.append(verdict)  # the drift did not exist yet
        elif verdict.detector == owner or verdict.index <= onset + tolerance:
            tp.append(verdict)
        else:
            fp.append(verdict)  # right idea, far too late, wrong component
    return tp, fp, repeats


def score_case(
    *,
    fault: str,
    session_id: str,
    owner: str | None,
    onset: int | None,
    is_drift: bool,
    requires: tuple[str, ...],
    capabilities: frozenset[str],
    verdicts: list[StepVerdict],
    steps: int,
    cost: Cost,
    tolerance: int = 8,
) -> CaseScore:
    """Attribute one session's verdicts and bundle it with its answer key."""
    tp, fp, repeats = attribute(
        verdicts,
        onset=onset,
        owner=owner,
        is_drift=is_drift,
        tolerance=tolerance,
    )
    return CaseScore(
        fault=fault,
        session_id=session_id,
        owner=owner,
        onset=onset,
        is_drift=is_drift,
        answerable=set(requires) <= set(capabilities),
        requires=requires,
        steps=steps,
        verdicts=verdicts,
        true_positives=tp,
        false_positives=fp,
        repeats=repeats,
        cost=cost,
    )


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision / recall / F1 from confusion counts (zero-safe)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _delays(cases: list[CaseScore]) -> dict[str, float | int | None]:
    delays = [c.delay for c in cases if c.delay is not None]
    return {
        "n": len(delays),
        "mean": round(fmean(delays), 2) if delays else None,
        "max": max(delays) if delays else None,
        "p95": sorted(delays)[min(len(delays) - 1, int(0.95 * len(delays)))] if delays else None,
    }


def _fault_block(kind: str, cases: list[CaseScore]) -> dict[str, Any]:
    drift = [c for c in cases if c.is_drift]
    tp = sum(len(c.true_positives) for c in cases)
    fp = sum(len(c.false_positives) for c in cases)
    fn = sum(1 for c in drift if not c.detected)
    return {
        "sessions": len(cases),
        "drift_sessions": len(drift),
        "owner": cases[0].owner,
        "answerable": cases[0].answerable,
        "requires": list(cases[0].requires),
        "detected_sessions": sum(1 for c in drift if c.detected),
        "true_positive_verdicts": tp,
        "false_positive_verdicts": fp,
        "false_alarms_per_session": round(fp / len(cases), 3) if cases else 0.0,
        "repeats_per_session": round(
            sum(len(c.repeats) for c in cases) / len(cases), 3
        )
        if cases
        else 0.0,
        "delay": _delays(cases),
        **prf(tp, fp, fn),
    }


def _detector_block(cases: list[CaseScore]) -> dict[str, Any]:
    """Per-detector honesty: what it claimed, and how much of it was right."""
    out: dict[str, Any] = {}
    for name in sorted({v.detector for c in cases for v in c.verdicts if v.fired and v.detector}):
        fired = [v for c in cases for v in c.verdicts if v.fired and v.detector == name]
        tp = [v for c in cases for v in c.true_positives if v.detector == name]
        fp = [v for c in cases for v in c.false_positives if v.detector == name]
        owned = [c for c in cases if c.is_drift and c.owner == name]
        missed = sum(1 for c in owned if not any(v.detector == name for v in c.true_positives))
        out[name] = {
            "verdicts": len(fired),
            "first_verdicts": len(tp) + len(fp),
            "true_positives": len(tp),
            "false_positives": len(fp),
            "repeats": len(fired) - len(tp) - len(fp),
            "precision": round(len(tp) / (len(tp) + len(fp)), 4) if tp or fp else 0.0,
            # Recall over the faults this detector actually owns; a fault caught
            # by someone else is not this detector's credit.
            "owned_faults": len(owned),
            "owner_recall": round((len(owned) - missed) / len(owned), 4) if owned else None,
        }
    return out


def summarize(cases: list[CaseScore], *, tolerance: int = 8) -> dict[str, Any]:
    """Aggregate case scores into the report's ``summary`` block."""
    drift = [c for c in cases if c.is_drift]
    clean = [c for c in cases if not c.is_drift]
    tp = sum(len(c.true_positives) for c in cases)
    fp = sum(len(c.false_positives) for c in cases)
    fn = sum(1 for c in drift if not c.detected)
    cost = Cost()
    for c in cases:
        cost = cost.merge(c.cost)
    n_clean = len(clean) or 1
    return {
        "cases": len(cases),
        "tolerance_steps": tolerance,
        "overall": {**prf(tp, fp, fn), "true_positives": tp, "false_positives": fp, "missed": fn},
        "false_alarm_rate_per_clean_session": round(fp / n_clean, 3),
        "false_alarms_on_clean_sessions": sum(len(c.false_positives) for c in clean),
        "repeat_verdicts": sum(len(c.repeats) for c in cases),
        "missed_faults": sorted({c.fault for c in drift if not c.detected}),
        "unanswerable_faults": sorted({c.fault for c in cases if not c.answerable}),
        "delay": _delays(cases),
        "by_fault": {kind: _fault_block(kind, group) for kind, group in _by_kind(cases).items()},
        "by_detector": _detector_block(cases),
        "cost": asdict(cost),
        "cost_per_session": {
            k: round(v / (len(cases) or 1), 2)
            for k, v in asdict(cost).items()
        },
    }


def _by_kind(cases: list[CaseScore]) -> dict[str, list[CaseScore]]:
    groups: dict[str, list[CaseScore]] = {}
    for c in cases:
        groups.setdefault(c.fault, []).append(c)
    return groups
