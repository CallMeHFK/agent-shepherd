"""Benchmark driver: scenarios -> real policy engine -> scored report.

Everything runs in-process against :class:`agent_shepherd.core.server.PolicyEngine`
— no daemon, no network, no writes to ``~/.shepherd`` (the ledger is pointed at a
temporary root, and ``SHEPHERD_HOME`` is redirected for the duration of the run
so nothing constructed downstream can touch the real one).

``detectors_only=True`` (the CI mode) stubs Tier 1: the run is deterministic,
offline and free, and the judge's *cost* is still accounted for — the wake gate
and the prompt that would have been sent are both measured, so the report can
answer "what would Tier 1 have cost on this corpus" without paying for it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import sys
import tempfile
import time
from typing import Any

from ..core.config import JudgeConfig, PolicyConfig, ShepherdConfig
from ..core.judge.prompts import SYSTEM_PROMPT, build_user_prompt
from ..core.judge.tier1 import StepScorer
from ..core.ledger import Ledger
from ..core.rules.detectors import CUSUMDriftDetector, OffSpecDetector
from ..core.server import PolicyEngine
from ..core.types import Agent, AgentEvent, EventType, ToolCall, VerdictAction
from . import metrics, scenarios

# ~4 characters per token: the same convention ``judge.risk`` uses for its
# guidance budget. A comparable ceiling, not an invoice.
CHARS_PER_TOKEN = 4
DEFAULT_TOLERANCE = 8


def default_config() -> ShepherdConfig:
    """In-memory config: benchmark defaults, and never reads ``~/.shepherd``."""
    return ShepherdConfig(
        judge=JudgeConfig(api_key=""),
        # The rulebook is cross-session memory; leaving it on would let the
        # benchmark learn from the cases it has already replayed and break the
        # independence of every case after it.
        policy=PolicyConfig(rulebook_enabled=False),
        agents={},
    )


def estimate_tokens(text: str) -> int:
    return (len(SYSTEM_PROMPT) + len(text)) // CHARS_PER_TOKEN


class RecordingScorer:
    """Tier 1 stand-in that measures wakes and prompt size.

    ``inner=None`` (the CI mode) means "the judge stayed asleep and said pass";
    the accounting still happens, which is what makes the cost columns honest.
    """

    def __init__(self, inner: Any = None):
        self.inner = inner
        self.calls = 0
        self.tokens = 0

    def score(self, current: AgentEvent, history: list[AgentEvent]) -> Any:
        self.calls += 1
        self.tokens += estimate_tokens(build_user_prompt(history, current))
        return None if self.inner is None else self.inner.score(current, history)


def _event(kind: EventType, *, tool: ToolCall | None = None, result: str | None = None) -> AgentEvent:
    return AgentEvent(
        agent=Agent.QWENPAW,
        session_id="capability-probe",
        event=kind,
        ts=0.0,
        tool=tool,
        tool_result=result,
    )


def _contract_scope() -> bool:
    """Can the contract check see an out-of-scope edit when the prompt is not on
    the same event? Real adapters only send the prompt at ``UserPromptSubmit``,
    so a detector that needs it on the tool call is dead in production.

    The history holds one *in-contract* read, so the probe cannot be satisfied
    by a detector that simply flags every edit whatsoever.
    """
    seen = _event(EventType.TOOL_RESULT, tool=ToolCall(name="read_file", input={"path": "core/server.py"}))
    edit = _event(EventType.TOOL_CALL, tool=ToolCall(name="edit_file", input={"path": "deploy/prod.sh"}))
    return OffSpecDetector().evaluate(edit, [seen]) is not None


def _unobservable_alarm() -> bool:
    """Can the drift alarm be reached at all by a run of *lost* tool responses?

    Probed in the shape the benchmark injects — a healthy prefix, then calls
    whose results never arrive, each call followed by its result — because the
    statistic's own baseline and the interleaved non-result steps both decide
    the answer.
    """
    detector = CUSUMDriftDetector(target_fpr=0.05)
    shell = ToolCall(name="shell")
    reader = ToolCall(name="read_file", input={"path": "session.jsonl"})
    history: list[AgentEvent] = []
    for _ in range(16):
        history.extend(
            (
                _event(EventType.TOOL_CALL, tool=shell),
                _event(EventType.TOOL_RESULT, tool=shell, result="3 passed in 0.21s"),
            )
        )
    for _ in range(30):
        pair = (
            _event(EventType.TOOL_CALL, tool=reader),
            _event(EventType.TOOL_RESULT, tool=reader, result=""),
        )
        for made in pair:
            if detector.evaluate(made, history) is not None:
                return True
            history.append(made)
    return False


def capabilities(config: ShepherdConfig | None = None, ledger_root: str | None = None) -> frozenset[str]:
    """What this build of the detectors can, in principle, see.

    Faults declare what they require; unanswerable faults are still run and
    still scored, they are just not silently counted as detector failures.
    """
    root = ledger_root or tempfile.mkdtemp(prefix="shepherd-cap-")
    try:
        engine = PolicyEngine(config or default_config(), Ledger(root=root))
        caps = {d.name for d in engine.detectors}
    finally:
        if ledger_root is None:
            shutil.rmtree(root, ignore_errors=True)
    if _has_signals():
        caps.add("signals")
    if _contract_scope():
        caps.add("contract_scope")
    if _unobservable_alarm():
        caps.add("unobservable_alarm")
    return frozenset(caps)


def _has_signals() -> bool:
    try:
        from ..core.rules import signals  # noqa: F401
    except ImportError:
        return False
    return True


def _run_session(
    scenario: scenarios.Scenario,
    config: ShepherdConfig,
    ledger_root: str,
    *,
    detectors_only: bool,
) -> tuple[list[metrics.StepVerdict], metrics.Cost]:
    """Feed one session through a fresh engine and record every answer."""
    engine = PolicyEngine(config, Ledger(root=ledger_root))
    scorer = RecordingScorer(None if detectors_only else StepScorer(config.judge))
    engine.scorer = scorer
    verdicts: list[metrics.StepVerdict] = []
    cost = metrics.Cost()
    for index, event in enumerate(scenario.events):
        verdict = engine.process(event)
        fired = verdict.action != VerdictAction.PASS
        verdicts.append(
            metrics.StepVerdict(
                index=index,
                event=event.event.value,
                action=verdict.action.value,
                detector=verdict.detector,
                reason=verdict.reason,
            )
        )
        cost.events_ingested += 1
        cost.nudges_emitted += 1 if fired else 0
        cost.nudges_suppressed += 1 if "suppressed (cooldown)" in verdict.reason else 0
    cost.judge_wakes = scorer.calls
    cost.judge_tokens = scorer.tokens
    return verdicts, cost


def _case(
    fault: str,
    run: int,
    seed: int,
    steps: int,
    config: ShepherdConfig,
    ledger_root: str,
    caps: frozenset[str],
    *,
    detectors_only: bool,
    tolerance: int,
) -> metrics.CaseScore:
    session_seed = seed + run
    rng = random.Random(f"{fault}:{session_seed}")
    at_step = max(6, scenarios.default_onset(fault, steps) + rng.choice((-2, 0, 2, 4)))
    scenario = scenarios.build_session(
        fault,
        steps=steps,
        seed=session_seed,
        at_step=at_step,
        session_id=f"{fault}-{run}-{session_seed}",
    )
    verdicts, cost = _run_session(scenario, config, ledger_root, detectors_only=detectors_only)
    truth = scenario.truth
    return metrics.score_case(
        fault=fault,
        session_id=scenario.session_id,
        owner=truth.owner,
        onset=truth.onset,
        is_drift=truth.is_drift,
        requires=truth.requires,
        capabilities=caps,
        verdicts=verdicts,
        steps=len(scenario),
        cost=cost,
        tolerance=tolerance,
    )


def run_benchmark(
    *,
    detectors_only: bool = True,
    seed: int = 7,
    sessions_per_fault: int = 8,
    steps: int = 60,
    config: ShepherdConfig | None = None,
    faults: list[str] | None = None,
    tolerance: int = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Run the corpus and return a JSON-serializable report.

    The report is a ``summary`` (per-fault, per-detector, cost) plus the
    ``cases`` it was computed from, so a surprising number can always be traced
    back to the session that produced it.
    """
    cfg = config or default_config()
    chosen = faults or scenarios.fault_kinds()
    root = tempfile.mkdtemp(prefix="shepherd-eval-")
    previous_home = os.environ.get("SHEPHERD_HOME")
    os.environ["SHEPHERD_HOME"] = root
    started = time.perf_counter()
    try:
        caps = capabilities(cfg, root)
        cases = [
            _case(
                fault,
                run,
                seed,
                steps,
                cfg,
                root,
                caps,
                detectors_only=detectors_only,
                tolerance=tolerance,
            )
            for fault in chosen
            for run in range(sessions_per_fault)
        ]
    finally:
        os.environ.pop("SHEPHERD_HOME", None)
        if previous_home is not None:
            os.environ["SHEPHERD_HOME"] = previous_home
        shutil.rmtree(root, ignore_errors=True)
    return {
        "meta": {
            "generated_at": round(time.time(), 3),
            "seed": seed,
            "sessions_per_fault": sessions_per_fault,
            "steps": steps,
            "detectors_only": detectors_only,
            "tolerance_steps": tolerance,
            "faults": chosen,
            "capabilities": sorted(caps),
            "runtime_seconds": round(time.perf_counter() - started, 2),
        },
        "summary": metrics.summarize(cases, tolerance=tolerance),
        "cases": [_case_dict(c) for c in cases],
    }


def _case_dict(case: metrics.CaseScore) -> dict[str, Any]:
    data = dataclasses.asdict(case)
    data["detected"] = case.detected
    data["delay"] = case.delay
    return data


def format_table(report: dict[str, Any]) -> str:
    """Compact fault -> recall / delay / false-alarm table."""
    summary = report["summary"]
    by_fault: dict[str, Any] = summary["by_fault"]
    lines = [
        f"capabilities: {', '.join(report['meta']['capabilities'])}",
        f"{'fault':24s} {'recall':>7s} {'f1':>6s} {'delay':>11s} {'FA/sess':>8s} {'rep/s':>6s} {'nudges':>7s}  owner",
    ]
    for kind, block in by_fault.items():
        delay = block["delay"]
        span = "-" if delay["mean"] is None else f"{delay['mean']:.1f}/{delay['max']}"
        nudges = sum(c["cost"]["nudges_emitted"] for c in report["cases"] if c["fault"] == kind)
        missing = sorted(set(block["requires"]) - set(report["meta"]["capabilities"]))
        flag = f"  (needs {'+'.join(missing)})" if missing else ""
        lines.append(
            f"{kind:24s} {block['recall']:7.2f} {block['f1']:6.2f} {span:>11s} "
            f"{block['false_alarms_per_session']:8.2f} {block['repeats_per_session']:6.2f} "
            f"{nudges:7d}  {block['owner'] or '-'}{flag}"
        )
    overall = summary["overall"]
    cost = summary["cost"]
    lines.append(
        f"{'OVERALL':24s} {overall['recall']:7.2f} {overall['f1']:6.2f} "
        f"{summary['delay']['mean'] or 0:>11.1f} {summary['false_alarm_rate_per_clean_session']:8.2f}"
    )
    lines.append(
        "cost: "
        f"{cost['events_ingested']} events, {cost['judge_wakes']} judge wakes "
        f"(~{cost['judge_tokens']} tokens), {cost['nudges_emitted']} nudges, "
        f"{cost['nudges_suppressed']} suppressed"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m agent_shepherd.eval.harness``."""
    parser = argparse.ArgumentParser(description="Evaluate the supervisor against known-answer sessions.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sessions", type=int, default=8, help="sessions per fault")
    parser.add_argument("--steps", type=int, default=60, help="target events per session")
    parser.add_argument("--fault", action="append", help="only run these faults (repeatable)")
    parser.add_argument("--json", metavar="OUT", help="write the full report to OUT")
    parser.add_argument("--fail-under-f1", type=float, default=None, help="exit 1 if overall F1 is lower")
    parser.add_argument("--judge", action="store_true", help="enable Tier 1 (needs a judge endpoint)")
    args = parser.parse_args(argv)

    try:
        report = run_benchmark(
            detectors_only=not args.judge,
            seed=args.seed,
            sessions_per_fault=args.sessions,
            steps=args.steps,
            faults=args.fault,
        )
    except KeyError as exc:
        print(f"shepherd eval: {exc}", file=sys.stderr)
        return 2
    print(format_table(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print(f"report written to {args.json}")
    f1 = report["summary"]["overall"]["f1"]
    if args.fail_under_f1 is not None and f1 < args.fail_under_f1:
        print(f"FAIL: overall F1 {f1:.3f} < {args.fail_under_f1:.3f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
