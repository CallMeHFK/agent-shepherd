"""Tests for the shepherd rulebook (which nudges actually work?)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_shepherd.core.ledger import Ledger, default_root
from agent_shepherd.core.rules import rulebook as rb
from agent_shepherd.core.rules.detectors import LoopDetector, RegressionDetector
from agent_shepherd.core.rules.rulebook import (
    DEFAULT_WINDOW,
    Rulebook,
    RuleStat,
    adherence_after_nudge,
    rulebook_path,
)
from agent_shepherd.core.types import Agent, AgentEvent, EventType, ToolCall, Verdict, VerdictAction

# ---------------------------------------------------------------------------
# raw ledger-record builders
#
# The rulebook consumes what ``Ledger`` writes, so these fixtures speak JSON,
# not dataclasses. Every record gets a fresh increasing timestamp at
# construction time, which keeps a hand-built list in ledger append order.
# ---------------------------------------------------------------------------

LOOP_CMD = "pytest tests/test_foo.py -k bar"
TWEAK_CMD = LOOP_CMD + " --tb=short"
OTHER_CMD = "ruff check agent_shepherd/core"

_CLOCK = [1000.0]


def _ts(ts: float | None) -> float:
    if ts is not None:
        return float(ts)
    _CLOCK[0] += 1.0
    return _CLOCK[0]


def call(
    name: str, inp: dict | None = None, *, ts: float | None = None, kind: str = "tool_call"
) -> dict:
    return {
        "type": "event",
        "event": kind,
        "iteration": 1,
        "ts": _ts(ts),
        "tool": {"name": name, "input": dict(inp or {}), "tool_use_id": None},
        "tool_result": None,
        "reasoning": None,
        "prompt": None,
        "metadata": {},
    }


def result(name: str, text: str, inp: dict | None = None) -> dict:
    rec = call(name, inp, kind="tool_result")
    rec["tool_result"] = text
    return rec


def bare(kind: str, *, tool_result: str | None = None, reasoning: str | None = None) -> dict:
    """An event record with no tool attached (adapters do emit these)."""
    return {
        "type": "event",
        "event": kind,
        "iteration": 1,
        "ts": _ts(None),
        "tool": None,
        "tool_result": tool_result,
        "reasoning": reasoning,
        "prompt": None,
        "metadata": {},
    }


def say(detector: str | None, *, action: str = "nudge") -> dict:
    return {
        "type": "verdict",
        "action": action,
        "reason": "because",
        "guidance": "guidance",
        "confidence": 0.9,
        "detector": detector,
        "context": {},
        "ts": _ts(None),
    }


def pad(records: list[dict], k: int = DEFAULT_WINDOW) -> list[dict]:
    """Top a post-nudge stream up to ``k`` events, so its window is closed."""
    return records + [bare("iteration_end") for _ in range(max(0, k - len(records)))]


def offender() -> dict:
    return call("shell", {"cmd": LOOP_CMD})


def different() -> dict:
    return call("shell", {"cmd": OTHER_CMD})


def tweaked() -> dict:
    return call("shell", {"cmd": TWEAK_CMD})


def read() -> dict:
    return call("read", {"path": "src/a.py"})


def looped(n: int = 3) -> list[dict]:
    return [offender() for _ in range(n)]


# ---------------------------------------------------------------------------
# adherence: the "change" family (loop / near_duplicate / drift)
# ---------------------------------------------------------------------------


def test_loop_nudge_is_followed_when_the_next_call_genuinely_differs():
    assert adherence_after_nudge("loop", looped(), pad([different(), read()])) is True


def test_loop_nudge_is_ignored_when_the_identical_call_comes_back():
    assert adherence_after_nudge("loop", looped(), pad([offender()])) is False


def test_loop_nudge_is_ignored_when_only_the_arguments_get_tweaked():
    """Retrying with a new flag is the classic non-fix. The judgment uses
    ``LoopDetector``'s own near-duplicate bar, so the two never disagree."""
    assert adherence_after_nudge("loop", looped(), pad([tweaked()] * 3)) is False


def test_near_duplicate_resolves_to_the_loop_detector_that_reports_it():
    assert adherence_after_nudge("near_duplicate", looped(), pad([different(), read()])) is True


def test_loop_nudge_is_ignored_when_the_offending_call_returns_later_in_the_window():
    after = [different(), offender()] + pad([])[:4]
    assert adherence_after_nudge("loop", looped(), after) is False


def test_only_the_next_k_events_are_examined():
    """A recurrence after the window is a new problem, not this nudge's failure."""
    after = pad([different()], k=3) + [offender()]
    assert adherence_after_nudge("loop", looped(), after, window=3) is True


def test_silence_after_a_change_nudge_is_not_adherence():
    after = pad([bare("reasoning", reasoning="hmm")])
    assert adherence_after_nudge("loop", looped(), after) is False


def test_repeat_nudge_from_the_same_detector_inside_the_window_is_not_followed():
    after = [read(), say("loop")] + pad([])[:4]
    assert adherence_after_nudge("loop", looped(), after) is False


def test_a_different_detector_firing_is_not_a_repeat_of_this_one():
    after = [read(), say("drift")] + pad([different()])[:4]
    assert adherence_after_nudge("loop", looped(), after) is True


def test_drift_nudge_is_judged_like_a_loop_on_the_last_failing_call():
    before = [result("shell", "error: boom", {"cmd": LOOP_CMD})]
    assert adherence_after_nudge("drift", before, pad([read(), different()])) is True


# ---------------------------------------------------------------------------
# adherence: the "stop" family (regression / offspec / binding)
# ---------------------------------------------------------------------------


def test_regression_nudge_is_followed_when_the_failing_command_is_left_alone():
    before = [call("pytest"), result("pytest", "2 failed")]
    assert adherence_after_nudge("regression", before, pad([read(), different()])) is True


def test_regression_nudge_is_ignored_when_the_command_runs_again():
    """The rerun shows up as a *result* record, not only as a call."""
    before = [result("pytest", "2 failed")]
    after = [read(), result("pytest", "1 failed"), result("pytest", "3 failed")] + pad([])[:3]
    assert adherence_after_nudge("regression", before, after) is False


def test_offspec_nudge_tracks_the_path_not_the_tool():
    before = [call("write_file", {"path": "src/unrelated.py"})]
    after = pad([call("edit_file", {"path": "src/main.py"})])
    assert adherence_after_nudge("offspec", before, after) is True


def test_offspec_nudge_is_ignored_when_the_same_path_is_edited_again():
    before = [call("write_file", {"path": "src/unrelated.py"})]
    after = [read(), call("str_replace", {"file_path": "./src/Unrelated.py"})] + pad([])[:4]
    assert adherence_after_nudge("offspec", before, after) is False


def test_binding_nudge_belongs_to_the_stop_family():
    before = [call("shell", {"cmd": "git status"})]
    assert adherence_after_nudge("binding", before, pad([read()])) is True
    again = call("shell", {"cmd": "git status"})
    assert adherence_after_nudge("binding", before, pad([again])) is False


def test_a_stop_nudge_needs_to_see_the_agent_move_somewhere():
    """Never touching the offending tool again, but doing nothing at all, is not
    evidence the rule worked."""
    before = [result("pytest", "2 failed")]
    after = pad([bare("reasoning", reasoning="hmm")])
    assert adherence_after_nudge("regression", before, after) is False


# ---------------------------------------------------------------------------
# adherence: the "act" family (contextrot)
# ---------------------------------------------------------------------------


def test_contextrot_nudge_is_followed_by_any_concrete_call():
    before = [bare("reasoning", reasoning="thinking hard")]
    assert adherence_after_nudge("contextrot", before, pad([read()])) is True


def test_contextrot_nudge_is_ignored_when_the_agent_keeps_reasoning():
    before = [bare("reasoning", reasoning="thinking hard")]
    after = [bare("reasoning", reasoning="still thinking") for _ in range(DEFAULT_WINDOW)]
    assert adherence_after_nudge("contextrot", before, after) is False


def test_contextrot_needs_a_real_tool_not_an_empty_event():
    before = [bare("reasoning", reasoning="thinking hard")]
    assert adherence_after_nudge("contextrot", before, pad([bare("tool_call")])) is False


# ---------------------------------------------------------------------------
# inconclusive samples
# ---------------------------------------------------------------------------


def test_an_unknown_detector_cannot_be_judged():
    """Tier 1 guidance is free-form, so "did it work?" has no deterministic
    answer; guessing either way would poison the ranking."""
    after = pad([different()])
    assert adherence_after_nudge("judge", looped(), after) is False
    assert adherence_after_nudge(None, looped(), after) is False
    assert rb._judge("judge", looped(), after).decided is False


def test_a_change_nudge_without_an_offending_signature_is_inconclusive():
    before = [bare("tool_result", tool_result="error: boom")]
    assert rb._judge("drift", before, pad([read()])).decided is False


def test_inconclusive_samples_are_excluded_from_the_adherence_rate():
    book = Rulebook()
    records = [bare("tool_result", tool_result="error"), say("drift"), *pad([read()])]
    assert book.observe(records, agent="qwenpaw", session_id="s1") == 1
    stat = book.stats[("qwenpaw", "drift")]
    assert (stat.nudges, stat.inconclusive, stat.evidence) == (1, 1, 0)
    assert stat.adherence_rate == 0.0
    assert book.rules() == []


# ---------------------------------------------------------------------------
# Rulebook.observe
# ---------------------------------------------------------------------------


def _loop_session(adhered: bool) -> list[dict]:
    response = different() if adhered else offender()
    return [*looped(), say("loop"), *pad([response])]


def test_observe_tallies_evidence_adherence_and_sessions():
    book = Rulebook()
    for i in range(3):
        assert book.observe(_loop_session(True), agent="claude", session_id=f"s{i}") == 1
    assert book.observe(_loop_session(False), agent="claude", session_id="s9") == 1
    stat = book.stats[("claude", "loop")]
    assert stat.detector == "loop" and stat.agent == "claude"
    assert stat.nudges == 4
    assert stat.adhered == 3
    assert stat.adherence_rate == 0.75
    assert stat.sessions_observed == 4
    assert stat.mean_steps_to_recovery == 1.0
    assert stat.last_seen_ts > 0


def test_observe_defaults_the_agent_to_the_unknown_bucket():
    book = Rulebook()
    book.observe(_loop_session(True))
    assert list(book.stats) == [(rb.UNKNOWN_AGENT, "loop")]


def test_observe_ignores_suppressed_and_pass_verdicts():
    """The engine records a PASS for every cooldown-suppressed repeat; only
    nudges that actually reached the agent are evidence."""
    book = Rulebook()
    records = [offender(), offender(), say("loop", action="pass"), *pad([different()])]
    assert book.observe(records, agent="claude", session_id="s1") == 0
    assert book.stats == {}


def test_observe_defers_an_open_window_and_never_double_counts():
    book = Rulebook()
    partial = [*looped(), say("loop"), read(), different()]
    assert book.observe(partial, agent="qwenpaw", session_id="s1") == 0
    assert book.stats == {}
    assert book.observe(partial, agent="qwenpaw", session_id="s1") == 0
    grown = partial + pad([read()])
    assert book.observe(grown, agent="qwenpaw", session_id="s1") == 1
    assert book.observe(grown, agent="qwenpaw", session_id="s1") == 0
    assert book.stats[("qwenpaw", "loop")].nudges == 1


def test_a_run_that_ended_closes_its_last_window():
    book = Rulebook()
    records = [*looped(), say("loop"), read(), bare("stop")]
    assert book.observe(records, agent="qwenpaw", session_id="s1") == 1
    assert book.stats[("qwenpaw", "loop")].adhered == 1


def test_observe_judges_every_nudge_in_one_stream():
    book = Rulebook()
    records = [
        *looped(),
        say("loop"),
        *pad([different()]),
        offender(),
        say("loop"),
        *pad([offender()]),
    ]
    assert book.observe(records, agent="qwenpaw", session_id="multi") == 2
    stat = book.stats[("qwenpaw", "loop")]
    assert stat.nudges == 2
    assert stat.adhered == 1


def test_a_second_nudge_inside_the_first_window_voids_the_first_sample():
    """The repeat is itself evidence the first one did not land."""
    book = Rulebook()
    records = [*looped(), say("loop"), read(), say("loop"), *pad([different()])]
    assert book.observe(records, agent="qwenpaw", session_id="tight") == 2
    stat = book.stats[("qwenpaw", "loop")]
    assert (stat.nudges, stat.adhered) == (2, 1)


def test_observe_swallows_junk_records():
    book = Rulebook()
    records: list = [None, "junk", 3, *looped(), say("loop"), *pad([different()])]
    assert book.observe(records, agent="qwenpaw", session_id="s1") == 1


# ---------------------------------------------------------------------------
# Rulebook.from_ledger
# ---------------------------------------------------------------------------


def _write_session(root: Path, session_id: str, records: list[dict]) -> None:
    path = root / "sessions" / Agent.QWENPAW.value / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec) + "\n" for rec in records)


def test_from_ledger_replays_sessions_in_the_right_direction(tmp_path):
    """``Ledger.iter_records`` hands records over newest-first, so the replay
    path is exactly where a wrong order would show up as an empty book."""
    _write_session(tmp_path, "s1", _loop_session(True))
    _write_session(tmp_path, "s2", _loop_session(False))
    book = Rulebook.from_ledger(Ledger(root=tmp_path), Agent.QWENPAW, ["s1", "s2"])
    stat = book.stats[(Agent.QWENPAW.value, "loop")]
    assert stat.nudges == 2
    assert stat.adhered == 1
    assert stat.sessions_observed == 2


def test_from_ledger_discovers_sessions_when_none_is_given(tmp_path):
    for sid in ("a", "b", "c"):
        _write_session(tmp_path, sid, _loop_session(True))
    book = Rulebook.from_ledger(Ledger(root=tmp_path), "qwenpaw")
    assert book.stats[(("qwenpaw"), "loop")].sessions_observed == 3


def test_from_ledger_survives_an_unknown_agent_name(tmp_path):
    book = Rulebook.from_ledger(Ledger(root=tmp_path), "future-agent")
    assert book.stats == {}


def test_from_ledger_reads_a_real_ledger(tmp_path):
    ledger = Ledger(root=tmp_path)
    offending = ToolCall(name="shell", input={"cmd": LOOP_CMD})
    ledger.record_event(AgentEvent(Agent.CLAUDE, "s1", EventType.TOOL_CALL, 1.0, tool=offending))
    verdict = Verdict(VerdictAction.NUDGE, "loop", "g", 0.95, "loop")
    ledger.record_verdict(Agent.CLAUDE, "s1", verdict)
    for i in range(DEFAULT_WINDOW):
        tool = ToolCall(name="read", input={"p": i})
        event = AgentEvent(Agent.CLAUDE, "s1", EventType.TOOL_CALL, 2.0 + i, tool=tool)
        ledger.record_event(event)
    book = Rulebook.from_ledger(ledger, Agent.CLAUDE, ["s1"])
    assert book.stats[(Agent.CLAUDE.value, "loop")].adhered == 1


# ---------------------------------------------------------------------------
# promotion and ranking
# ---------------------------------------------------------------------------


def _book_with(**counts: tuple[int, int]) -> Rulebook:
    """Build a book from detector -> (adhered, ignored) outcome counts."""
    book = Rulebook()
    for detector, (adhered, ignored) in counts.items():
        for i in range(adhered):
            book.record_outcome(
                detector, adhered=True, steps=2, agent="qwenpaw", session_id=f"a{i}"
            )
        for i in range(ignored):
            book.record_outcome(detector, adhered=False, agent="qwenpaw", session_id=f"b{i}")
    return book


def test_min_evidence_floor_keeps_thin_detectors_out_of_the_book():
    book = _book_with(loop=(2, 0), drift=(9, 1))
    assert [r.detector for r in book.rules(min_evidence=3)] == ["drift"]
    assert [r.detector for r in book.rules(min_evidence=10)] == ["drift"]
    assert [r.detector for r in book.rules(min_evidence=11)] == []


def test_min_adherence_floor_keeps_ignored_nudges_out_of_the_book():
    book = _book_with(loop=(1, 4), drift=(4, 1))
    rules = book.rules(min_evidence=3, min_adherence=0.5)
    assert [r.detector for r in rules] == ["drift"]
    assert rules[0].adherence == pytest.approx(0.8)


def test_ranking_is_adherence_times_log_evidence():
    """Equally reliable, but the recurring one is the rule worth the context."""
    book = _book_with(loop=(4, 1), contextrot=(8, 2))
    rules = book.rules(min_evidence=3, min_adherence=0.5)
    assert [r.detector for r in rules] == ["contextrot", "loop"]
    assert rules[0].evidence == 10 and rules[1].evidence == 5
    assert rules[0].score > rules[1].score
    assert rules[0].adherence == rules[1].adherence


def test_a_rarely_observed_perfect_rule_loses_to_a_reliable_recurring_one():
    book = _book_with(loop=(3, 0), drift=(16, 4))
    assert [r.detector for r in book.rules()] == ["drift", "loop"]


def test_ranking_is_deterministic_for_ties():
    book = _book_with(loop=(3, 1), regression=(3, 1))
    assert [r.detector for r in book.rules()] == ["loop", "regression"]


def test_rules_can_be_filtered_by_agent_and_never_repeat_a_detector():
    book = Rulebook()
    book.record_outcome("loop", adhered=True, agent="claude")
    for i in range(3):
        book.record_outcome("loop", adhered=True, agent="qwenpaw", session_id=f"s{i}")
    assert [r.agent for r in book.rules(min_evidence=1, agent="claude")] == ["claude"]
    shared = book.rules(min_evidence=1)
    assert len(shared) == 1
    assert shared[0].agent == "qwenpaw"  # the better-evidenced agent wins


def test_rule_carries_the_detectors_own_guidance_text():
    """No copy-paste: the promoted rule's text must be what the detector itself
    puts on a live verdict."""
    live = RegressionDetector().evaluate(
        AgentEvent(
            agent=Agent.CLAUDE,
            session_id="x",
            event=EventType.TOOL_RESULT,
            ts=1.0,
            tool=ToolCall("pytest"),
            tool_result="1 failed",
        ),
        [
            AgentEvent(
                agent=Agent.CLAUDE,
                session_id="x",
                event=EventType.TOOL_RESULT,
                ts=0.5,
                tool=ToolCall("pytest"),
                tool_result="7 passed",
            )
        ],
    )
    assert live is not None and live.guidance
    rule = _book_with(regression=(5, 0)).rules()[0]
    assert rule.text == " ".join(live.guidance.split())
    assert rule.headline != rule.text  # the header uses the compressed form


def test_every_known_detector_resolves_to_a_non_empty_rule():
    for detector in rb.ADHERENCE_FAMILIES:
        rule = _book_with(**{detector: (4, 0)}).rules()[0]
        assert rule.headline and rule.headline != detector
        assert rule.text and len(rule.text.split()) > 5


def test_rule_text_falls_back_for_a_detector_that_does_not_exist_yet():
    rule = _book_with(binding=(4, 1)).rules()[0]
    assert rule.headline == rb.RULE_TEXT_BY_DETECTOR["binding"]
    assert rule.text == rb.GUIDANCE_BY_DETECTOR["binding"]


def test_record_outcome_accepts_an_enum_agent_and_tracks_steps():
    book = Rulebook()
    book.record_outcome("loop", adhered=True, steps=3, agent=Agent.CODEX, session_id="s1")
    book.record_outcome("loop", adhered=True, steps=5, agent="codex", session_id="s1")
    stat = book.stats[(Agent.CODEX.value, "loop")]
    assert stat.adhered == 2
    assert stat.mean_steps_to_recovery == 4.0
    assert stat.sessions_observed == 1


def test_record_outcome_marks_inconclusive_samples():
    book = Rulebook()
    book.record_outcome("drift", adhered=False, inconclusive=True, agent="codex")
    stat = book.stats[("codex", "drift")]
    assert (stat.nudges, stat.inconclusive, stat.evidence) == (1, 1, 0)


# ---------------------------------------------------------------------------
# rendered header
# ---------------------------------------------------------------------------


def test_render_header_lists_rules_with_their_evidence():
    book = _book_with(regression=(12, 0))
    header = book.render_header()
    assert header is not None
    lines = header.splitlines()
    assert lines[0] == rb.HEADER_TITLE
    assert lines[1] == (
        "- When a command you already passed starts failing, diff since the last pass "
        "(observed 12x, 100% followed)."
    )
    assert len(lines) == 2


def test_render_header_is_none_when_nothing_qualifies():
    assert Rulebook().render_header() is None
    assert _book_with(loop=(1, 1)).render_header() is None
    assert _book_with(loop=(0, 6)).render_header() is None


def test_render_header_respects_the_token_budget():
    book = _book_with(regression=(12, 0), loop=(9, 1), offspec=(6, 2), contextrot=(5, 2))
    full = book.render_header(budget_tokens=400)
    assert full is not None and len(full.splitlines()) == 4  # capped by max_rules
    tight = book.render_header(budget_tokens=60)
    assert tight is not None and rb._estimate_tokens(tight) <= 60
    assert len(tight.splitlines()) < len(full.splitlines())


def test_render_header_truncates_the_last_line_instead_of_dropping_it():
    book = _book_with(regression=(12, 0), loop=(9, 1))
    header = book.render_header(budget_tokens=60)
    assert header is not None
    assert header.splitlines()[-1].endswith("...")
    assert rb._estimate_tokens(header) <= 60


def test_render_header_gives_up_when_the_budget_cannot_hold_a_rule():
    book = _book_with(regression=(12, 0))
    assert book.render_header(budget_tokens=rb._estimate_tokens(rb.HEADER_TITLE) + 2) is None


def test_render_header_caps_at_max_rules():
    book = _book_with(regression=(20, 0), loop=(19, 0), offspec=(18, 0), contextrot=(17, 0))
    book.max_rules = 2
    assert len(book.render_header(budget_tokens=400).splitlines()) == 3


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _populated_book() -> Rulebook:
    book = Rulebook(window=5, max_rules=2)
    book.observe(_loop_session(True), agent="qwenpaw", session_id="s1")
    book.observe(_loop_session(False), agent="claude", session_id="s2")
    spinning = [bare("reasoning", reasoning="x"), say("contextrot"), *pad([read()])]
    book.observe(spinning, agent="codex")
    return book


def test_json_round_trip_preserves_stats_and_cursors():
    book = _populated_book()
    restored = Rulebook.from_dict(json.loads(json.dumps(book.to_dict())))
    assert restored.to_dict() == book.to_dict()
    assert restored.window == 5 and restored.max_rules == 2
    assert restored.rules(min_evidence=1) == book.rules(min_evidence=1)


def test_save_and_load_use_the_shepherd_home_env_var(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("SHEPHERD_HOME", str(home))
    assert default_root() == home
    assert rulebook_path() == home / "rulebook.json"
    book = _populated_book()
    assert book.save() is True
    assert (home / "rulebook.json").exists()
    assert Rulebook.load().to_dict() == book.to_dict()


def test_default_path_follows_the_home_directory(monkeypatch, tmp_path):
    """The fallback when SHEPHERD_HOME is unset, proven without touching a real
    ~/.shepherd."""
    monkeypatch.delenv("SHEPHERD_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake-home")
    assert rulebook_path() == tmp_path / "fake-home" / ".shepherd" / "rulebook.json"
    assert Rulebook().save() is True
    assert list((tmp_path / "fake-home" / ".shepherd").iterdir()) == [rulebook_path()]


def test_save_leaves_no_temp_files_behind(monkeypatch, tmp_path):
    monkeypatch.setenv("SHEPHERD_HOME", str(tmp_path))
    assert _populated_book().save() is True
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rulebook.json"]
    payload = json.loads((tmp_path / "rulebook.json").read_text(encoding="utf-8"))
    assert payload["version"] == rb.SCHEMA_VERSION


def test_save_failure_is_reported_not_raised(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("still not a directory", encoding="utf-8")
    assert Rulebook().save(blocker / "rulebook.json") is False


def test_load_returns_an_empty_book_for_a_missing_file(tmp_path):
    book = Rulebook.load(tmp_path / "nope" / "rulebook.json")
    assert isinstance(book, Rulebook)
    assert book.stats == {} and book.rules() == [] and book.render_header() is None


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not json",
        "[]",
        "{}",
        '{"version": 999}',
        '{"version": 1, "stats": 3}',
        '{"stats": [{}]}',
        "null",
        '{"version": 1, "stats": [{"nudges": 5}]}',
    ],
)
def test_load_never_crashes_on_a_corrupt_file(tmp_path, payload):
    path = tmp_path / "rulebook.json"
    path.write_text(payload, encoding="utf-8")
    book = Rulebook.load(path)
    assert book.rules() == []


def test_from_dict_skips_junk_entries_but_keeps_the_rest():
    data = {
        "version": rb.SCHEMA_VERSION,
        "stats": [
            {"detector": "loop", "agent": "qwenpaw", "nudges": 4, "adhered": 4},
            {"agent": "qwenpaw"},
            "junk",
            {"detector": "drift", "nudges": "many", "adhered": None, "sessions": "s1"},
        ],
        "cursors": {"qwenpaw:s1": "7"},
    }
    book = Rulebook.from_dict(data)
    # The junk drift entry survives but is demoted: unreadable counters become 0
    # and the missing agent becomes the unknown bucket.
    assert sorted(book.stats) == [("qwenpaw", "loop"), ("unknown", "drift")]
    assert book.stats[("unknown", "drift")].nudges == 0
    assert book.stats[("unknown", "drift")].sessions == []
    assert book.cursors == {"qwenpaw:s1": 7}


def test_rules_from_a_stat_survive_a_round_trip():
    book = _populated_book()
    restored = Rulebook.from_dict(book.to_dict())
    assert set(restored.stats) == set(book.stats)
    assert all(isinstance(stat, RuleStat) for stat in restored.stats.values())


def test_reobserving_a_saved_book_adds_nothing():
    """Why the cursors are persisted: a restart must not recount last week's
    evidence and make a flimsy rule look proven."""
    book = _populated_book()
    restored = Rulebook.from_dict(json.loads(json.dumps(book.to_dict())))
    restored.observe(_loop_session(True), agent="qwenpaw", session_id="s1")
    assert restored.stats[("qwenpaw", "loop")].nudges == book.stats[("qwenpaw", "loop")].nudges


# ---------------------------------------------------------------------------
# end to end: detectors + ledger + rulebook
# ---------------------------------------------------------------------------


def test_a_live_detector_loop_becomes_a_rule_the_engine_can_inject(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_HOME", str(tmp_path))
    ledger = Ledger(root=tmp_path)
    detector = LoopDetector()
    offending = ToolCall(name="shell", input={"cmd": LOOP_CMD})
    compliant = ToolCall(name="read", input={"path": "src/a.py"})

    def session(session_id: str, follows: bool) -> None:
        clock = 5_000.0
        history: list[AgentEvent] = []
        fired = False
        for i in range(3):
            event = AgentEvent(
                Agent.QWENPAW, session_id, EventType.TOOL_CALL, clock + i, tool=offending
            )
            ledger.record_event(event)
            verdict = detector.evaluate(event, list(history))
            history.append(event)
            if verdict is not None and not fired:
                ledger.record_verdict(Agent.QWENPAW, session_id, verdict)
                fired = True
        tool = compliant if follows else offending
        for i in range(DEFAULT_WINDOW):
            event = AgentEvent(
                Agent.QWENPAW, session_id, EventType.TOOL_CALL, clock + 10 + i, tool=tool
            )
            ledger.record_event(event)

    for sid in ("c1", "c2", "c3", "c4"):
        session(sid, follows=True)
    session("d1", follows=False)

    book = Rulebook.from_ledger(ledger, Agent.QWENPAW)
    stat = book.stats[(Agent.QWENPAW.value, "loop")]
    assert stat.evidence == 5
    assert stat.adherence_rate == pytest.approx(0.8)
    assert stat.mean_steps_to_recovery == 1.0
    rule = book.rules()[0]
    # The promoted text is the detector's own live guidance, verbatim.
    assert rule.text == " ".join(detector.evaluate(
        AgentEvent(Agent.QWENPAW, "z", EventType.TOOL_CALL, 1.0, tool=offending),
        [
            AgentEvent(Agent.QWENPAW, "z", EventType.TOOL_CALL, 0.5, tool=offending),
            AgentEvent(Agent.QWENPAW, "z", EventType.TOOL_CALL, 0.6, tool=offending),
        ],
    ).guidance.split())
    header = book.render_header()
    assert header is not None
    assert header.splitlines()[1] == (
        "- Do not re-run a near-identical call; take a genuinely different step "
        "(observed 5x, 80% followed)."
    )
    assert book.save() is True
    assert Rulebook.load().render_header() == header


def test_observe_accepts_a_lazy_iterable_full_of_junk():
    """The engine will hand this ``Ledger.recent()``, but any iterable of
    records is the same contract: skip what is not a record."""

    def stream():
        yield None
        yield "not a record"
        yield from looped()
        yield say("loop")
        yield from pad([different()])

    book = Rulebook()
    assert book.observe(stream(), agent="codex", session_id="s1") == 1
    assert book.stats[("codex", "loop")].adhered == 1


def test_cursors_are_keyed_per_session_not_globally():
    """Two sessions that happen to be the same length must not consume each
    other's records."""
    book = Rulebook()
    partial = [*looped(), say("loop"), read()]
    assert book.observe(partial, agent="qwenpaw", session_id="a") == 0
    assert book.observe(partial, agent="qwenpaw", session_id="b") == 0
    assert book.observe(partial + pad([read()]), agent="qwenpaw", session_id="a") == 1
    assert book.observe(partial + pad([read()]), agent="qwenpaw", session_id="b") == 1
    assert book.stats[("qwenpaw", "loop")].nudges == 2
    assert len(book.cursors) == 2


def test_the_engines_own_ledger_output_is_what_the_rulebook_reads(tmp_path):
    """End to end through the real ingest path: the events and verdicts
    ``PolicyEngine.process`` appends are sufficient to score its own nudge."""
    from agent_shepherd.core.config import JudgeConfig, PolicyConfig, ShepherdConfig
    from agent_shepherd.core.server import PolicyEngine

    ledger = Ledger(root=tmp_path)
    config = ShepherdConfig(judge=JudgeConfig(), policy=PolicyConfig(), agents={})
    engine = PolicyEngine(config, ledger)
    engine.scorer = None  # Tier 0 only; the loop detector is what we are testing

    tool = ToolCall(name="shell", input={"cmd": LOOP_CMD})

    def loop_call(i: int) -> AgentEvent:
        return AgentEvent(
            agent=Agent.QWENPAW,
            session_id="live",
            event=EventType.TOOL_CALL,
            ts=1_000.0 + i * 600.0,  # spaced past the nudge cooldown
            iteration=i,
            tool=tool,
        )

    assert engine.process(loop_call(0)).action == VerdictAction.PASS
    assert engine.process(loop_call(1)).action == VerdictAction.PASS
    assert engine.process(loop_call(2)).action == VerdictAction.NUDGE
    # The agent then does something genuinely different.
    for i in range(3, 9):
        engine.process(
            AgentEvent(
                agent=Agent.QWENPAW,
                session_id="live",
                event=EventType.TOOL_CALL,
                ts=1_000.0 + i * 600.0,
                iteration=i,
                # Genuinely different steps: re-running one identical read would
                # itself be a loop, and the rulebook would be right to say so.
                tool=ToolCall(name="read", input={"path": f"src/a{i}.py"}),
            )
        )
    records = ledger.recent(Agent.QWENPAW, "live", limit=200)
    assert any(r.get("type") == "verdict" and r.get("detector") == "loop" for r in records)
    book = Rulebook.from_ledger(ledger, Agent.QWENPAW, ["live"])
    stat = book.stats[(Agent.QWENPAW.value, "loop")]
    assert (stat.nudges, stat.adhered, stat.evidence) == (1, 1, 1)
    assert book.rules(min_evidence=1)[0].detector == "loop"


def test_a_near_duplicate_nudge_shares_the_loop_scorecard():
    """The loop detector's second mode must not split into its own rule."""
    book = Rulebook()
    for sid in ("n1", "n2", "n3"):
        book.observe([*looped(), say("near_duplicate"), *pad([different()])], agent="qwenpaw", session_id=sid)
    assert list(book.stats) == [("qwenpaw", "loop")]
    assert book.stats[("qwenpaw", "loop")].adhered == 3
    assert book.rules()[0].detector == "loop"
