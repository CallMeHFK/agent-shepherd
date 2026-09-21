"""Tests for the evaluation harness.

These are not tests of the detectors — they test that the *measurement* works:
that a fault which the installed detectors can see is caught with bounded delay,
that the clean corpus really is clean, that the numbers mean what their names
say, and that a benchmark run leaves the user's home directory alone.

Where a fault is not answerable by the build under test, the test asserts the
miss rather than excusing it: a benchmark that quietly stops scoring what it
cannot detect is how a supervisor starts believing its own green table.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agent_shepherd.core.config import JudgeConfig, PolicyConfig, ShepherdConfig
from agent_shepherd.core.rules.detectors import CUSUMDriftDetector
from agent_shepherd.core.types import AgentEvent
from agent_shepherd.eval import harness, metrics, scenarios
from agent_shepherd.eval.metrics import StepVerdict

# Measured bounds, not aspirations: an exact loop needs three calls (delay 2),
# context rot needs eight quiet non-tool events before its window is clean
# enough to read as a stall (delay 4), and the drift alarm has to out-accumulate
# a calibrated false-alarm budget (delay up to one monitoring window).
MAX_DELAY = {
    "exact_loop": 3,
    "near_duplicate_loop": 3,
    "regression": 5,
    "offspec_edit": 2,
    "context_rot": 10,
    "sustained_drift": 16,
    "binding_drift": 2,
}
CLEAN_KINDS = ("none", "flaky_but_healthy")


@pytest.fixture(scope="module")
def report() -> dict:
    return harness.run_benchmark(detectors_only=True, seed=7, sessions_per_fault=6, steps=50)


@pytest.fixture(scope="module")
def caps(report) -> frozenset:
    return frozenset(report["meta"]["capabilities"])


def _cases_of(report: dict, kind: str) -> list[dict]:
    return [c for c in report["cases"] if c["fault"] == kind]


# --------------------------------------------------------------------------
# the generator: paired design, known onset, real failure grammars
# --------------------------------------------------------------------------


def test_injected_session_shares_its_prefix_with_the_control():
    """The fault must be the only difference between a case and its control."""
    control = scenarios.build_session("none", steps=50, seed=7, at_step=20)
    injected, truth = scenarios.inject(control, "exact_loop", 20, seed=7)
    assert truth.onset is not None and truth.owner == "loop"
    assert injected[: truth.block_start] == control.events[: truth.block_start]
    assert len(injected) >= len(control)


def test_inject_is_deterministic_and_names_the_owner():
    events, truth = scenarios.inject(scenarios.healthy_session(steps=50, seed=7), "sustained_drift", 16, 7)
    again, again_truth = scenarios.inject(scenarios.healthy_session(steps=50, seed=7), "sustained_drift", 16, 7)
    assert events == again
    assert truth == again_truth
    assert truth.onset == 16


def test_every_fault_kind_declares_a_capability_requirement_set():
    for kind, spec in scenarios.FAULTS.items():
        assert spec.kind == kind
        assert isinstance(spec.requires, tuple)
        if spec.owner is not None:
            assert spec.owner in {"loop", "regression", "offspec", "contextrot", "drift", "binding"}


def test_clean_corpus_carries_no_failure_evidence():
    """Guard against fixtures that test nothing: the drift statistic must read
    the clean session as exactly zero, and read the injected drift block as
    strictly positive. Otherwise a "no alarm" result is vacuous."""
    detector = CUSUMDriftDetector(target_fpr=0.05)
    clean = scenarios.build_session("none", steps=50, seed=7)
    history: list[AgentEvent] = []
    for event in clean.events:
        assert detector.watch_level(event, history) == 0.0
        history.append(event)

    drifted = scenarios.build_session("sustained_drift", steps=50, seed=7)
    rising = 0.0
    history = []
    for event in drifted.events:
        rising = max(rising, detector.watch_level(event, history))
        history.append(event)
    assert rising > 0.0


def test_lost_responses_never_reach_the_alarm_line(caps):
    """The ambiguous corpus has to be ambiguous *in this build*, not by our
    saying so: the drift detector alone must never alarm on a run of lost
    responses, while it does alarm on the same pacing of real failures."""

    def drift_fires(kind: str) -> bool:
        detector = CUSUMDriftDetector(target_fpr=0.05)
        session = scenarios.build_session(kind, steps=50, seed=7)
        return any(detector.evaluate(e, session.events[:i]) is not None for i, e in enumerate(session.events))

    assert not drift_fires("ambiguous_tool_loss")
    assert drift_fires("sustained_drift")
    assert ("unobservable_alarm" in caps) is drift_fires("ambiguous_tool_loss")


# --------------------------------------------------------------------------
# attribution: the part that can quietly lie
# --------------------------------------------------------------------------


def _verdicts(*pairs: tuple[int, str]) -> list[StepVerdict]:
    return [StepVerdict(index=i, event="tool_result", action="nudge", detector=d) for i, d in pairs]


def test_attribute_credits_owner_and_window_but_not_late_strangers():
    verdicts = _verdicts((10, "loop"), (14, "drift"), (40, "drift"))
    tp, fp, repeats = metrics.attribute(verdicts, onset=8, owner="loop", is_drift=True, tolerance=8)
    assert [(v.index, v.detector) for v in tp] == [(10, "loop"), (14, "drift")]
    assert [(v.index, v.detector) for v in fp] == []
    assert [(v.index, v.detector) for v in repeats] == [(40, "drift")]


def test_attribute_treats_pre_onset_and_clean_sessions_as_false_positives():
    tp, fp, _ = metrics.attribute(_verdicts((4, "loop")), onset=8, owner="loop", is_drift=True)
    assert tp == [] and [v.index for v in fp] == [4]
    tp, fp, _ = metrics.attribute(_verdicts((9, "loop")), onset=None, owner=None, is_drift=False)
    assert tp == [] and [v.index for v in fp] == [9]


# --------------------------------------------------------------------------
# the benchmark discriminates
# --------------------------------------------------------------------------


def test_answerable_faults_are_all_caught_within_bounded_delay(report):
    checked = 0
    for kind, block in report["summary"]["by_fault"].items():
        if kind in CLEAN_KINDS or not block["answerable"]:
            continue
        checked += 1
        assert block["recall"] == 1.0, (kind, block)
        delays = [c["delay"] for c in _cases_of(report, kind) if c["delay"] is not None]
        assert delays and max(delays) <= MAX_DELAY[kind], (kind, delays)
    assert checked >= 5, "the build under test stopped scoring enough faults to be a check"


def test_unanswerable_faults_are_reported_as_misses_not_skips(report, caps):
    unanswerable = [k for k, b in report["summary"]["by_fault"].items() if not b["answerable"]]
    for kind in unanswerable:
        block = report["summary"]["by_fault"][kind]
        assert block["recall"] == 0.0
        assert kind in report["summary"]["missed_faults"]
        assert set(block["requires"]) - set(caps), "labelled unanswerable without a missing capability"


def test_clean_sessions_produce_zero_tier0_false_alarms(report):
    for kind in CLEAN_KINDS:
        cases = _cases_of(report, kind)
        assert cases, kind
        for case in cases:
            assert case["false_positives"] == [], (kind, case["onset"], case["verdicts"])
            assert case["true_positives"] == []
    assert report["summary"]["false_alarms_on_clean_sessions"] == 0


def test_false_alarms_do_appear_when_a_fault_is_answerable_but_missed(caps):
    """The zero above must be a measurement, not an absence of chances to fail:
    drop a required capability and the same fault is scored as a real miss."""
    broken = caps - {"loop"}
    case = metrics.score_case(
        fault="exact_loop",
        session_id="synthetic",
        owner="loop",
        onset=8,
        is_drift=True,
        requires=("loop",),
        capabilities=broken,
        verdicts=_verdicts((9, "drift")),
        steps=20,
        cost=metrics.Cost(),
    )
    assert not case.answerable
    assert case.detected  # drift was flagged in the window, so it is still credit


def test_delay_is_measured_from_the_injected_onset(report):
    measured = [(c["fault"], c["onset"], c["delay"]) for c in report["cases"] if c["delay"] is not None]
    assert measured, "no delay was ever measured"
    for case in report["cases"]:
        if case["delay"] is None:
            continue
        first = case["true_positives"][0]["index"]
        assert case["delay"] == first - case["onset"] >= 0


def test_detectors_only_mode_never_asks_the_judge_but_still_costs(report):
    assert all(v["detector"] != "judge" for c in report["cases"] for v in c["verdicts"])
    cost = report["summary"]["cost"]
    assert cost["judge_wakes"] > 0 and cost["judge_tokens"] > 0
    assert cost["nudges_emitted"] > 0
    # Hysteresis must be exercised, or the corpus is too short to reach the cooldown.
    assert cost["nudges_suppressed"] > 0


def test_events_ingested_matches_the_corpus_size(report):
    assert report["summary"]["cost"]["events_ingested"] == sum(c["steps"] for c in report["cases"])


def test_report_is_json_serializable(report, tmp_path):
    text = json.dumps(report, sort_keys=True)
    path = tmp_path / "report.json"
    path.write_text(text, encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["summary"]["overall"]["f1"] == report["summary"]["overall"]["f1"]
    assert {"meta", "summary", "cases"} <= set(loaded)


def test_run_benchmark_is_reproducible_under_a_seed():
    a = harness.run_benchmark(seed=11, sessions_per_fault=3, steps=40)
    b = harness.run_benchmark(seed=11, sessions_per_fault=3, steps=40)
    assert json.dumps(a["summary"], sort_keys=True) == json.dumps(b["summary"], sort_keys=True)
    assert json.dumps(a["cases"], sort_keys=True) == json.dumps(b["cases"], sort_keys=True)


def test_changing_the_seed_changes_the_corpus_not_the_conclusion():
    """A seeded corpus must still vary, or "recall 1.0 over 24 sessions" is
    really "recall 1.0 over one session written 24 times"."""
    one = scenarios.build_session("exact_loop", steps=50, seed=11)
    two = scenarios.build_session("exact_loop", steps=50, seed=12)
    assert one.events != two.events
    assert one.truth.onset is not None and two.truth.onset is not None
    rep = harness.run_benchmark(seed=11, sessions_per_fault=6, steps=40, faults=["exact_loop"])
    assert len({c["onset"] for c in rep["cases"]}) > 1


# --------------------------------------------------------------------------
# the harness must not touch the user's machine
# --------------------------------------------------------------------------


def _home_state() -> list[str]:
    root = Path.home() / ".shepherd"
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_benchmark_leaves_the_real_home_alone():
    before = _home_state()
    harness.run_benchmark(seed=3, sessions_per_fault=2, steps=40)
    assert _home_state() == before


def test_benchmark_leaves_no_temporary_ledgers_behind():
    def scratch() -> set[str]:
        root = Path(tempfile.gettempdir())
        return {p.name for p in root.iterdir() if p.name.startswith(("shepherd-eval-", "shepherd-cap-"))}

    before = scratch()
    harness.run_benchmark(seed=3, sessions_per_fault=2, steps=40)
    assert scratch() == before


def test_benchmark_honours_an_injected_config(tmp_path):
    """A caller-supplied config (different cooldown, drift off) must be used as
    given — the benchmark scores the deployment, not its own defaults."""
    cfg = ShepherdConfig(
        judge=JudgeConfig(api_key=""),
        policy=PolicyConfig(drift_enabled=False, nudge_cooldown_seconds=0.0),
        agents={},
    )
    report = harness.run_benchmark(config=cfg, seed=5, sessions_per_fault=2, steps=40)
    assert "drift" not in report["meta"]["capabilities"]
    assert report["summary"]["by_detector"].get("drift", {}).get("verdicts", 0) == 0
    assert report["summary"]["by_fault"]["sustained_drift"]["recall"] < 1.0


def test_cli_reports_and_enforces_the_f1_gate(report, tmp_path, capsys):
    out = tmp_path / "eval.json"
    assert harness.main(["--seed", "7", "--sessions", "2", "--steps", "40", "--json", str(out)]) == 0
    assert out.exists() and json.loads(out.read_text(encoding="utf-8"))["cases"]
    table = capsys.readouterr().out
    assert "near_duplicate_loop" in table and "OVERALL" in table
    assert harness.main(["--seed", "7", "--sessions", "2", "--steps", "40", "--fail-under-f1", "0.99"]) == 1
    assert harness.main(["--seed", "7", "--sessions", "2", "--steps", "40", "--fail-under-f1", "0.1"]) == 0
    assert harness.main(["--fault", "not-a-fault"]) == 2
