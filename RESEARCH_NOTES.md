# Research notes: literature-grounded optimizations (2026-09)

This file maps the papers behind the current design of the policy engine to the
code that implements them, and lists the follow-up ideas that were considered
but not (yet) implemented. It exists so a future maintainer can see *why* a
piece of code looks the way it does, and where the open questions are.

## Method

A focused literature sweep over the project's core problem — *cheaply deciding
when a drifting agent needs expensive intervention* — ran six arXiv queries
(PRM / step-level scoring, CUSUM & change-point detection, agentic loops,
tool-call necessity, context rot / lost-in-the-middle, self-correction &
reflection) over a 24-month window (148 candidate papers). Eighteen abstracts
were read in depth; the ones cited below each motivated a specific, small
change rather than a rewrite. All links were retrieved on 2026-09-18.

## Implemented optimizations

### 1. CUSUM drift detector with a calibrated false-alarm budget

**What changed:** new `CUSUMDriftDetector` (Tier 0, `agent_shepherd/core/rules/detectors.py`).
A one-sided CUSUM over a cheap 0/1 "this tool result failed" signal alarms on a
*sustained* run of failures. The alarm threshold is not a hand-tuned constant:
it is found by a seeded Monte-Carlo binary search over the null model
(healthy session = Bernoulli(`baseline`) failures), picking the most sensitive
threshold whose simulated false-alarm rate stays at or below `drift_target_fpr`
(5% per monitoring window). The result is deterministic and cached.

**Why:**
- *Real-Time Detection and Repair of LLM Agent Failures* —
  <https://arxiv.org/abs/2608.02464> — is the closest work: it frames
  mid-run supervision as one-class novelty detection + CUSUM alarms, trains
  only on healthy runs, and explicitly budgets a ~5% false-alarm rate because
  "judging every step with a second LLM costs more than the agent itself".
  Our detector is the deterministic, zero-training analogue of that alarm.
- *Sequential Control of False Positives in Online Change Point Detection* —
  <https://arxiv.org/abs/2607.15423> — points out that a fixed threshold over
  a moving window is a multiple-testing problem and calibrates thresholds with
  simulation; we follow that (simpler per-window budget, no sequential FWER).
- *Quickest Detection of Hallucination Onset* —
  <https://arxiv.org/abs/2606.12476> — the change-detection framing (delay
  bounds, CUSUM) used to keep the detector onset-based rather than state-based.

**Trade-off noted:** overlapping monitoring windows mean the *long-horizon*
false-alarm rate exceeds the per-window 5% budget. That is accepted: the
detector's nudges are cheap and the same evidence also wakes the judge, which
is where the cost lives.

### 2. Anti-saturation wake gate

**What changed:** `PolicyEngine._wake` (server.py) no longer wakes the LLM
judge on every tool call / tool result and no longer pulses every N events.
The judge now wakes (a) at natural checkpoints — `PROMPT_SUBMIT`,
`ITERATION_END`, `STOP` (all three adapters emit iteration boundaries,
verified), and (b) mid-iteration only when the CUSUM statistic climbs to
`drift_watch_fraction` (default 0.6) of its alarm line on a tool result —
i.e., on *onset evidence*, as a soft trigger. `wake_every_n_steps` was retired
from the config.

**Why:**
- *The Saturation Trap and the Subjectivity of Intervention Timing* —
  <https://arxiv.org/abs/2606.04296> — shows threshold-on-state triggers fire
  on a large constant fraction of actions (39–83% in their study), so the
  supervisor's cost ends up dominating. A CUSUM statistic with decay only
  re-crosses its watch line on an up-swing, which is exactly the hysteresis a
  state threshold lacks.
- *Real-Time Detection and Repair* (2608.02464, above): the expensive
  reviewer should be asleep except when something starts to happen.

**Behavior (verified in tests):** 1 isolated failure → silent (watch ≈ 0.5);
2 consecutive failures → soft wake (watch = 1.0); 3 consecutive → hard Tier 0
drift nudge, no judge call at all.

### 3. Near-duplicate loop detection

**What changed:** `LoopDetector` now has two phases: the original exact-match
loop (3+ identical calls), then a near-duplicate phase — 3+ calls to the *same
tool* whose inputs share token Jaccard ≥ 0.7 (conservative: same tool name,
≥ 2 tokens required). Catches the common real failure of retrying with
slightly different arguments instead of changing strategy, while
`echo a` / `echo b` / `read` remain undetectably different.

**Why:**
- *When Agents Do Not Stop: Uncovering Infinite Agentic Loops in LLM Agents* —
  <https://arxiv.org/abs/2607.01641> (IAL-Scan) — documents that real loops
  are usually *state/workflow cycles*, not literally identical calls;
  near-duplicate matching is the cheap first step toward catching the "same
  idea, new spelling" loop.
- *To Call or Not to Call* — <https://arxiv.org/abs/2605.00737> — motivates
  asking "was this call necessary / did it add information" per call, which
  is the spirit of flagging re-calls that change nothing material.

### 4. Stepwise Tier 1 prompt with credit assignment

**What changed:** the judge prompt (`core/judge/prompts.py`) now (a) pins the
user's goal at the **top** of the input, (b) numbers every window line so the
judge can refer to steps by index, and (c) instructs *stepwise* scoring with
explicit **credit assignment**: score each step separately, locate the
*earliest* step where the trajectory diverged, do not blame a later step for
an earlier mistake, and name the drifted step's index in `reason`. The strict
JSON output shape is unchanged, so `parse_verdict` is untouched.

**Why:**
- *Self-Evaluating LLMs for Multi-Step Tasks* —
  <https://arxiv.org/abs/2511.07364> — stepwise confidence estimation beats
  holistic window scoring (+15% relative AUC-ROC), so per-step scoring is the
  right granularity for a PRM-style judge.
- *GroundedPRM* — <https://arxiv.org/abs/2510.14942> — and *Beyond
  Trajectory Rewards: Step-level Credit Assignment (GDCR)* —
  <https://arxiv.org/abs/2605.29697> — both identify *credit misattribution*
  (reward/verdict assigned to the wrong step) as a central failure of
  trajectory-level judging; "name the origin step" is the cheap prompt-level
  version of the fix.
- *Lost in the Middle, and In-Between* —
  <https://arxiv.org/abs/2412.10079> — information in the middle of context is
  underused; the goal (the judge's drift reference frame) is the one thing
  that must not sit in the middle, so it is pinned to the top.

### 5. Anti-nagging hysteresis + wall-clock quiet for context rot

**What changed:** two anti-saturation measures, both motivated by live
operation. (a) **Nudge cooldown (hysteresis):** the policy engine tracks the
last NUDGE timestamp per detector per session; a NUDGE from the same detector
within `policy.nudge_cooldown_seconds` (default 300s) is suppressed (logged
as a PASS with a "suppressed (cooldown)" reason). BLOCK/ESCALATE are never
suppressed and never touch the cooldown. (b) **Wall-clock quiet requirement:**
the context-rot detector now requires the agent to have been *quiet* (no tool
activity observed in a 64-event lookback) for at least `min_quiet_seconds`
(60s) in wall-clock time before it flags.

**Why:**
- *State-saturation trap* — <https://arxiv.org/abs/2606.04296> (see item 2):
  a fixed state threshold fires on a constant fraction of actions for as long
  as the condition holds. We watched exactly this happen in production: after
  the item-2 fixes, a long QwenPaw background-task wait (reasoning while
  polling a sleeping task) produced **62,165 identical context-rot nudges** in
  one session's ledger. The wall-clock quiet check makes "rot" a property of
  time, not event count — fast streaming deltas no longer look like spinning —
  and the cooldown makes even a genuine sustained condition cost one nudge per
  5 minutes instead of one per event.

## Second sweep (2026-09-20/21): what changed and why

A second sweep ran five query batches (~400 records, 93 kept) aimed specifically
at what the first round left undone: measuring the supervisor, the failure signal
it thinks it has, and the thresholds it advertises. Items 6-11 below are
implemented; the follow-ups list is rewritten.

### 6. Structured, three-state outcome classification

**What changed:** new `core/rules/signals.py`. The Tier 0 failure signal was
`any(marker in result.lower())` over a seven-word list, shared by the regression
and drift detectors. It is now `classify(event) -> FAILED | OK | UNKNOWN`,
decided from structured evidence first — a `PostToolUseFailure` hook event, an
`is_error` flag, an `exit_code` field (looked for in *both* metadata and the
response body, since the adapters lift fields into metadata) — and only then
from line-anchored error grammars (`^error: `, `Traceback (most recent...`,
`npm ERR!`, `N failed`, `exit code: [1-9]`). The Claude and Codex adapters stop
flattening `tool_response` dicts into strings, which is what makes the exit code
survive to the daemon at all.

**Why:**
- *Verified Tool Calls Improve LLM Agent Reliability Under Non-Atomic Failures* —
  <https://arxiv.org/abs/2608.02645> — real tool calls fail non-atomically
  (timeout after dispatch, delayed error, lost response). A binary signal has to
  file those somewhere, and the old one filed them as *success*.
- *Did It Happen? Counterfactual Evaluation of LLM Agent Recovery from Ambiguous
  Tool Outcomes* — a timeout after a side-effecting call does not reveal whether
  the action failed or executed and lost its response; retrying is right in one
  case and duplicates the effect in the other. Hence a third state.
- Substring matching was also live-fire wrong: `grep -rn error logs/error.log`
  with zero hits, or a passing run that mentions `test_error_handling`, scored
  as a failure — and because the same signal feeds the CUSUM, the false positive
  was *integrated* into a hard alarm.

**Deliberate limit:** a response that arrives and matches no failure grammar is
`OK`, not `UNKNOWN`. Reserving UNKNOWN for "nothing observable" keeps the graded
CUSUM from accumulating on healthy sessions; the alternative was to reintroduce
the false-alarm failure this item exists to remove.

### 7. Session-length false-alarm budget + self-starting CUSUM baseline

**What changed:** the alarm threshold is now calibrated over a `drift_horizon`
(default 120) of *re-checks* instead of one 16-step window, against a null model
that matches the new graded signal (failure / unobservable / clean). The
reference mean comes from the session's own pre-alarm history
(`_baseline_for(reference)`) once ≥8 results exist, floored at `baseline/2`.
Item 1's accepted trade-off ("the long-horizon false-alarm rate exceeds the
per-window 5% budget") is therefore no longer accepted — it is fixed.

**Why:**
- *When Drift Detectors cry Wolf: False Alarm Rates in continuous ML Monitoring*
  — <https://arxiv.org/abs/2607.17336> — detectors evaluated on isolated windows
  systematically under-report production false alarms precisely because
  monitoring is repeated. Overlapping re-checks are the multiple-testing problem.
- *Finite-Horizon Quickest Change Detection Balancing Latency with False Alarm
  Probability* — <https://arxiv.org/abs/2511.12803> — the delay-vs-false-alarm
  formulation the calibration approximates.
- *A comparative study of self-starting CUSUM control charts* —
  <https://arxiv.org/abs/2410.12736> and *Self-Normalization for CUSUM* —
  <https://arxiv.org/abs/2509.07112> — no Phase-I calibration phase is
  available here, so the reference must be built online. (Follow-up 3 from the
  first sweep, implemented.)

**Measured consequence:** the stricter budget moves the soft wake from two
consecutive failures to three, and the hard alarm from ~3 to ~4 (threshold
1.30 → 2.35). `test_drift_detector_watch_level_tracks_consecutive_failures` now
pins the *shape* (monotone, silent at one, crossing somewhere in the run) rather
than the constant, because the constant is a calibrated output.

**Trap found and avoided while implementing:** estimating the adaptive baseline
from the alarming window itself is self-defeating — a pure run of failures drives
the reference to 1.0, the required drift to `1.0 + slack`, and the statistic can
never alarm again. The test for this is
`test_drift_detector_baseline_comes_from_the_pre_alarm_reference_period`.

### 8. Off-spec edits became a delegation contract

**What changed:** `OffSpecDetector` no longer regexes paths out of the prompt and
flags everything else (which fired on nearly every real edit, since users rarely
name every file). A path is admissible when it is named in the goal, matched by
`scope_allow_globs`, **or already observed this session** — a file the agent
read or saw referenced is in play. `scope_deny_globs` is checked first and
yields BLOCK. Edit-tool and path-key recognition now covers the real tool names
across the three harnesses (`Write`/`Edit`/`MultiEdit`/`NotebookEdit`,
`notebook_path`, `filename`, …) instead of four names from one of them.

**Why:**
- *Software Delegation Contracts: Measuring Reviewability in AI Coding-Agent
  Work* — <https://arxiv.org/abs/2606.17099> — the unit of delegated coding work
  is task + *bounded authority* + returned package, not a keyword list.
- *Solver-Aided Verification of Policy Compliance in Tool-Augmented LLM Agents* —
  <https://arxiv.org/abs/2603.20449> — admissibility should be decided
  deterministically, not by asking a model.
- *Trust but Verify? Uncovering the Security Debt of Autonomous Coding Agents* —
  <https://arxiv.org/abs/2607.12428> — the risk concentrates in specific
  high-impact paths, which is what a deny-list encodes.

### 9. Binding-drift detector

**What changed:** new `BindingDriftDetector` (Tier 0, `binding_enabled`). An edit
whose target path is ≥0.6 token-Jaccard similar to a path the session
*established* (read, grepped, or merely named in a result) but is not that path
nudges: "you inspected `config.py` and are now editing `config.prod.py`".

**Why:** *Binding Drift in Multi-Step Tool-Augmented Agents* —
<https://arxiv.org/abs/2607.18316> — agents pick the correct tool and bind it to
the wrong entity a large fraction of the time (24–26% reported for single-step
actions), and the call *succeeds*, so no outcome-based signal sees it. Neither
the loop detector (arguments genuinely differ) nor the contract check (the path
may be in scope) covers it.

### 10. The advertised thresholds now exist, and are calibrated

**What changed:** `nudge_threshold`, `block_threshold` and `max_guidance_tokens`
were declared in config, documented in README as tuning knobs, and read by
nothing — a judge verdict was applied verbatim and its confidence only recorded.
`PolicyEngine._control` gates admission on them; BLOCK additionally requires the
per-agent `block_enabled` opt-in and otherwise degrades to a NUDGE; guidance is
truncated to the token budget. New `core/judge/risk.py` fits those cut-points by
split conformal risk control (monotone FPR in the threshold, `(n+1)/n`
finite-sample correction, configured priors used until `risk_min_samples`
labeled outcomes exist), persisted to `$SHEPHERD_HOME/risk.json`, fed by
`PolicyEngine.note_outcome` from observed adherence.

**Why:**
- *Calibration Is Not Control: Why LLM-Agent Oversight Needs Intervention* —
  <https://arxiv.org/abs/2606.21399> — framing oversight as "estimate a risk
  score, cross a threshold" controls the wrong object; the quantity of interest
  is the effect of intervening. This is the closest paper in the sweep to the
  project's own premise and the reason the fix is *calibrate then gate* rather
  than *hand-pick a number and pretend it is a guarantee*.
- *On Verbalized Confidence Scores for LLMs* —
  <https://arxiv.org/abs/2412.14737> — the number being thresholded is a
  verbalized self-report, so it earns a calibration step, not blind trust.
- *Conformal Selective Prediction with General Risk Control* —
  <https://arxiv.org/abs/2603.24704> and *Selective Conformal Risk Control* —
  <https://arxiv.org/abs/2512.12844> — the fitting procedure.

### 11. Cost accounting, so "worth it" is answerable

**What changed:** `SessionState` counts events seen, judge calls, judge tokens
(from the OpenAI-compatible `usage` block), nudges, blocks, escalations and
suppressions; every verdict is written to the ledger with that context, and
`PolicyEngine.stats` / `GET /stats/<agent>/<session>` / `shepherd stats` expose
it, including `tokens_per_intervention` — the ratio that decides whether the
reviewer's cost is dominated by supervision or by the agent.

**Why:** the saturation-trap papers argue the supervisor's cost ends up
dominating; that is an argument about a number nobody in this repo could
formerly print. *EcoAgent-Bench* — <https://arxiv.org/abs/2608.05519> — makes the
same methodological point for agents generally: resource use is part of the task,
not an auxiliary statistic.

## Honest caveats (negative results we did NOT ignore)

- *Sample More, Reflect Less* — <https://arxiv.org/abs/2607.28576> — at equal
  token cost, repeated sampling beat Self-Refine/Reflexion-style reflection on
  their tasks. Implication: the judge's value here is *steering* (guidance the
  agent can act on mid-run), not self-reflection per se — which is exactly
  what this project is, but a future eval should budget tokens against
  "just retry the agent N times".
- *Decomposing LLM Self-Correction* —
  <https://arxiv.org/abs/2601.00828> — the accuracy-correction paradox: models
  are better at *detecting* errors than fixing them; detection is where the
  value is. This is the design principle behind keeping Tier 0 cheap and
  broad, and Tier 1 advisory rather than corrective.

### 12. A counterfactual benchmark, which immediately changed a default

**What changed:** `agent_shepherd/eval/` (scenarios + metrics + harness) and
`shepherd eval`, wired into CI with `--fail-under-f1`. Scripted healthy sessions
plus a fault injector that breaks exactly one thing at a *known* step, scored
for per-fault recall, detection delay in steps, false alarms per session,
repeats, and supervision cost. Tier 1 is stubbed, so it is deterministic,
offline and free.

**Why:** *trajectory-judge* — <https://arxiv.org/abs/2609.00038> — measures what
outcome-only judging misses using exactly this construction (deterministic
environment, scripted oracle, one fault at a known step); *ToolRobustBench*
(<https://arxiv.org/abs/2608.23635>) localizes failure stage-wise;
*What Twelve LLM Agent Benchmark Papers Disclose About Themselves*
(<https://arxiv.org/abs/2605.21404>) is why the harness prints its own cost
columns instead of only accuracy.

**What it found on the first run (and what we did):** the new off-spec rule —
"edit only paths named in the goal, allow-globbed, or already observed" — fired
**1.0 false nudge per clean session**, mostly on perfectly normal work (writing a
brand-new test file, editing a file a repo-wide grep had surfaced). Requiring a
file to have been *read* before it may be edited conflates novelty with
overstepping authority, and a detector that costs one nudge per healthy session
is the saturation failure in item 2 all over again. `scope_nudge_unobserved` is
therefore **off by default** (deny-globs still BLOCK, which is the
authority-relevant half), and measured false alarms on clean sessions went
1.00 → 0.00 with aggregate F1 0.62 → 0.79. Recall on the `offspec_edit` fault
dropped to 0 by the same switch: Tier 0 cannot answer that one under the default
policy, and the harness marks it `(unanswerable here)` rather than hiding it.

**Still open after the benchmark:** `sustained_drift` alarms ~13 steps after
onset and cost 0.17 false alarms per session; `ambiguous_tool_loss` is not
detectable by Tier 0 at all (an empty result is *unobserved*, not evidence).
Judge wake rate is ~20% of events, which is the number item 11 exists to make
visible.

### 13. The installed hook must not be able to break the agent

**What changed:** daemon calls from the adapters use `trust_env=False`, and the
hook thin-shells fail open on *any* exception.

**Why:** found by running the real chain after installing, not by a unit test —
with `ALL_PROXY=socks5://…` in the environment, `httpx.Client(...)` raises
`ImportError` (SOCKS support is an optional extra), so `shepherd-hook claude`
produced a traceback on **every tool call**. Catching only `HTTPError` let it
through. A supervisor unreachable on a proxy-configured machine must be
invisible, not fatal; it must also not route loopback telemetry through an
unrelated proxy.

Also fixed the same class of failure in the dev workflow: `pytest` was only an
optional extra, so a bare `uv run pytest` could resolve outside the project
environment, import a stale installed copy of `agent_shepherd`, and report
**green tests against code no longer on disk** — which is exactly what it did
here before `dependency-groups` was declared.

### 14. Rulebook: guidance that worked becomes context, not a repeated nag

**What changed:** `core/rules/rulebook.py` judges adherence *operationally* from
the ledger — for `loop`/`near_duplicate`/`drift`, the next call's signature must
differ at `LoopDetector`'s own near-duplicate bar and the offending call must not
return in-window; for `regression`/`offspec`/`binding`, the offending tool or path
must go untouched while other work happens; for `contextrot`, any tool call counts.
A detector that fires again in-window voids the sample (a nudge the agent ignored
is not evidence that it works). Rules are ranked by a **Wilson lower bound on the
adherence rate** (`_wilson_lower_bound` in `core/rules/rulebook.py`), rendered
into a token-budgeted header, injected at `PROMPT_SUBMIT`, and re-learned from
the finished ledger at `STOP`. Since entry 32 the book also learns *online*, and
the ranking score blends a second, shrinkage-weighted Wilson bound over Tier 1
agreement with the behavioural one: each Tier 0 nudge that reached the agent is
labelled by the following Tier 1 review — agreed when the judge's drift score
clears the risk model's current admission line, disagreed when it does not — so a
rule has to survive both behaviour ("was it followed?") and Tier 1 ("was it
right?"). The rulebook
file lives beside the ledger, so a test or benchmark pointed at a temp ledger
cannot write into the user's config directory.

**Why:** *Meta-Policy Reflexion* — <https://arxiv.org/abs/2509.03990> — follow-up
1 of the first sweep, and the reason item 5's cooldown exists at all: the
alternative to repeating a nudge is to stop repeating it *and* keep the effect.

**Benchmark interaction found while wiring it:** leaving the rulebook on inside
`shepherd eval` let the run learn from cases it had already replayed, so a later
clean session scored a "false alarm" that was really memory leaking across cases.
The benchmark now disables it; case independence is worth more than one extra
column.

**Honest limit:** `judge` guidance is free-form prose, so it is never promotable
into a rule — only the fixed detector texts are.

## Follow-ups considered, not implemented

1. **Shepherd rulebook injection into the live loop** — *Meta-Policy Reflexion*
   (<https://arxiv.org/abs/2509.03990>): **done, and this entry was stale when
   entry 31 re-read it on 2026-10-12.** It claimed the engine "does not yet
   inject that header at `PROMPT_SUBMIT`"; the injection shipped with item 14 —
   `core/server.py:388` renders the header on `PROMPT_SUBMIT` and returns it as
   a NUDGE with `detector="rulebook"` (`:396`). Kept here rather than deleted so
   the numbering below stays stable.
2. **Graph-based credit assignment** — GDCR (2605.29697): replace the
   "name the origin step" heuristic with a causal graph over steps for the
   judge's verdict. Higher precision, much more machinery.
3. **Multi-view progress scoring** — *ProgRouter*
   (<https://arxiv.org/abs/2608.25992>): the scalar CUSUM is one view of
   "progress"; *SiLR* (<https://arxiv.org/abs/2609.04629>) warns that no single
   scalar surrogate is sound for multi-dimensional violation state. A 2–3 view
   statistic (failure rate + novelty of outputs + goal coverage) would be the
   principled next step. The self-normalized CUSUM item from the first sweep is
   done (item 7); this one is not.
4. **Tool-call-rate steering** — <https://arxiv.org/abs/2608.25198>: the
   nudge/block levers are the steering knobs; item 11 makes their cost
   measurable, but their *effect* on call frequency is still unmeasured.
   `trajectory-judge` (2609.00038) is the template for that experiment.
5. **Context assembly for the judge window** — *ContextPipe*
   (<https://arxiv.org/abs/2609.00749>) and *Tessera*: the judge window is still
   a fixed last-20 slice; treating window selection as an assembly problem
   (goal + last failure + last verification) should beat it on long sessions.
6. **Per-agent baselines from the eval corpus**: the risk model is fitted per
   (agent, source), but `shepherd eval` currently reports aggregate numbers; the
   per-agent/per-model breakdown that would let a user *see* the difference is
   not surfaced.
7. **Ambiguity-aware guidance for `UNKNOWN` outcomes**: item 6 introduced a
   third state, but the guidance text still tells the agent to "read the error
   output" — which is useless when nothing came back. A retry-vs-verify-side-
   effect branch (Did It Happen?) belongs in the prompt, not in the detector.
8. **AFTraj as an external label corpus for the risk model** — AgentForesight
   (<https://arxiv.org/abs/2605.08715>, see the fourth sweep below): 2,276
   CC-BY-4.0 trajectories whose *released* label is session-level safe/unsafe
   (no step-level `mistake_step` — see the correction in entry 19), which is
   exactly the shape `CalibrationSet.add(score, is_drift)` wants, so it can
   cross-check the weak "Tier 0 confirmed" live label (item 17) via an offline
   replay loader. Not implemented: lossy schema mapping, a parquet→JSON
   vendoring step, distribution shift from multi-agent to single-agent
   sessions, and real judge API calls to produce the scores being calibrated.
   Details and grading in the fourth sweep.
9. **Paired statistical acceptance gate for the three self-improvement loops**
   — SAGE (<https://arxiv.org/abs/2609.36043>, see the fifth sweep below):
   replace the rulebook's *aggregate* promotion score (then
   `adherence_rate * log1p(evidence)`; since replaced by a Wilson lower bound on
   the adherence rate blended with Tier 1 agreement, entries 30 and 32 — SAGE's
   objection to it is unchanged) with a per-item paired evaluation on the
   `shepherd eval` corpus plus an exact one-sided binomial
   (McNemar) commit test, and later run `risk.py` threshold refits through the
   same gate. Stdlib-only, offline, promotion-time cost. Graded P0/P1/P2 in
   entry 27.

## Third round: things the tests could not see

### 15. Two measured corrections: what the statistic folds over, and what a checkpoint buys

**The drift statistic was folding over the wrong domain.** `_statistic` summed
every event in the window, but only tool results carry signal, so each
interleaved call contributed `0.0 - slack` of decay. Consequence: a session whose
tool responses keep disappearing -- the exact non-atomic failure item 6 introduced
UNKNOWN to catch -- could *never* reach its alarm line. The benchmark reported
this honestly as `ambiguous_tool_loss: 0.00 recall, unanswerable`, and a test even
pinned the limitation as if it were a property. The sum now runs over results only
and an unobservable outcome is credited at 0.75 rather than 0.5: that fault
measures recall 1.00 at a 15-step delay, clean-session false alarms stay 0.00,
aggregate F1 0.84 -> 0.88, and `sustained_drift` delay improved 13 -> 7 steps.

**Reviewing every iteration boundary was the dominant cost.** With Tier 1 gated on
natural checkpoints, `ITERATION_END` woke the judge on ~23% of events in *healthy*
sessions. The reviewer now wakes at prompt start and session stop unconditionally,
and at an iteration boundary only when something in it did not come back clean
(`policy.review_clean_iterations` restores the old behaviour). Judge wakes over
the benchmark corpus: 660 -> 369 at strictly better recall.

Both are onset-based readings of the same argument as item 2 (arXiv 2606.04296):
spend the expensive reviewer on evidence, not on schedule. Neither showed up in a
unit test -- one came from a benchmark case that reported itself unanswerable
instead of being quietly dropped, the other from a cost column nobody had read.

### 16. The judge's product is a sentence, not a score

The Tier 1 prompt now asks for a verbal per-step critique -- up to three worst
steps, one <=30-word sentence each naming what the step assumed and what the
evidence says instead -- and `parse_verdict` folds those into the recorded reason.
*Verbal Process Supervision Elicits Better Coding Agents* —
<https://arxiv.org/abs/2503.18494> — finds natural-language step feedback beats
scalar process rewards; here it cost no machinery, only a field the parser was
already tolerant of, and a judge that ignores the instruction degrades to the
previous behaviour rather than failing.

### 17. Making the calibration actually receive labels

Item 10's fitter had no caller: `note_outcome` was never invoked, so the
"calibrated" thresholds were permanently the configured prior while looking
calibrated. Connecting it surfaced three defects that would each have made the
mechanism look working while doing nothing:

* learning steps hung off `STOP` *after* the parked-verdict drain, and the drain
  returns early, so a session that ended holding a verdict never learned. Session
  finalization now runs before any early return.
* `refit` stored its fallback when evidence was thin, and `threshold_for` read
  that stored fallback back as a calibrated value. For a per-agent key whose prior
  lives under the base name, that pinned the gate at 1.0 and would have vetoed
  every judge verdict for any newly seen agent. An under-evidenced key is now left
  explicitly unfitted.
* the fitted threshold was rounded to nearest, which can land below the grid point
  that satisfied the budget -- a risk bound broken by its last decimal. It rounds
  up now.

**The label, and its honest limit.** Live, the label is "Tier 0 independently
confirmed drift somewhere in this session". That makes the fitted admission line
concordant with the detectors, which is not the same as correct, and it will pull
the judge toward agreeing with Tier 0 rather than toward being right. The
by-construction label exists only in `shepherd eval`, and
`shepherd eval --judge --fit-risk` now writes that one out (the report also had to
start carrying each verdict's confidence, which it silently was not -- a corpus
that records what fired but not how sure it was can score detection and still
never calibrate a risk bound).

### 18. Startup cost: the supervisor was absent for its first ~8 seconds

A lifecycle problem wearing perf clothes. The daemon ran the Monte-Carlo threshold
fit (24 binary-search passes x 20k simulated sessions x a 120-step horizon)
*before* binding its socket. During those seconds every adapter POST fails to
connect and fails open, so a freshly started supervisor is silently not
supervising -- the one state in which it reports healthy while doing nothing.
Item 1's fit is seeded and deterministic, so the answer is cached in
`$SHEPHERD_HOME/cusum_thresholds.json` and reused verbatim: a cold start pays the
fit once, warm starts bind immediately (measured ~8s -> 0s construction time). A
cache write failure is ignored on purpose; a cold start may be slow, it may not be
fatal. `shepherd status` exists because the alternative was learning this from an
empty ledger.

Local judge endpoints are supported the same way: an unset
`SHEPHERD_JUDGE_API_KEY` now means "send no Authorization header" instead of
refusing to connect (a local gateway needs no credential, and this is where
running a supervisor costs nothing), and loopback/LAN judge URLs bypass proxy
environment for the same reason item 13 had to fix in the adapters.

## Fourth sweep (2026-10-12): AgentForesight — early failure prediction

A single-paper applicability study of *AgentForesight: Online Auditing for
Early Failure Prediction in Multi-Agent Systems* —
<https://arxiv.org/abs/2605.08715> (Zhang, Zhu, Shi, Liu, Tang; NeurIPS 2026
per the repo description). Primary sources read: the arXiv abstract and HTML
full text, the GitHub repo (<https://github.com/ZBox1005/AgentForesight>,
README plus `inference/prompts.py` and `inference/metrics.py` source), the HF
dataset card (<https://huggingface.co/datasets/ZBox008003/AFTraj>), and the
project page. Not fetched: Appendix B.3 (proposer/verifier counts), D.3
(compute analysis) and G.2 (limitations) — the HTML rendering truncates before
them; claims from those appendices are marked NOT VERIFIED below. The model
checkpoint is **not released** ("will be released on HuggingFace upon paper
acceptance", repo README), so every runtime claim about the 7B auditor is
paper-only.

### 19. What AgentForesight actually claims

**The protocol is the contribution, and it is our protocol.** "Online
auditing" (Sec. 2, Def. 2.2): at each step of an unfolding trajectory an
auditor sees *only the current prefix* `τ_{0:k}` and emits Continue or an
Alarm naming the predicted decisive-error step, the responsible agent, and a
one-to-two-sentence reason (Eq. 4; the exact JSON shape is in
`inference/prompts.py`: `{"answer": <step|"SAFE">, "agent": ..., "reason":
...}` inside a `<think>…</think><answer>…</answer>` envelope). Evaluation is a
strict step-by-step incremental walk, one independent model call per prefix,
greedy decoding (Sec. 4.1). So: same granularity as us — per-step, streaming,
prefix-only, usable for mid-run intervention — but it *is* a model call per
step, which is the thing items 2 and 15 exist to avoid. The paper predicts
only; it never actually intervenes ("paves the way for runtime safeguards",
conclusion).

**The model.** AgentForesight-7B, fine-tuned from Qwen2.5-7B-Instruct (the
base scores 21.05 Exact-F1 in Table 1) with a two-stage coarse-to-fine RL
recipe (Sec. 3.2): Stage 1 "BPPO" (Boundary-Pair Preference Optimization)
trains on adjacent safe/unsafe prefix pairs so the verdict flips exactly at
the decisive step; Stage 2 GRPO with a gated three-axis reward — format
(what), Gaussian-scored step accuracy (where), agent attribution (who) —
anchored to the Stage-1 checkpoint. Training used 2× H200 with vLLM rollouts
(Sec. 4.1). No training code is in the repo; inference only.

**AFTraj-2K** (Sec. 3.1; HF card): the released corpus is **2,276 rows —
1,162 safe + 1,114 unsafe** (verified via
`datasets-server.huggingface.co/info?dataset=ZBox008003/AFTraj`, splits
`safe`/`unsafe`), while the paper's Table 3 totals **2,272 (1,158 safe +
1,114 unsafe)**; the 4-row difference is unexplained in either source, so
cite both. Trajectories come from AutoGen / MetaGPT / Smolagents runs over
MATH-500, HumanEval+ / MBPP+, and agentic tasks (HotpotQA 234/242, GAIA
67/133, ToolSafety 19/43, Expert-Team 85/52; safe/unsafe per subdomain on
the card). Safe runs pass three binary filters (outcome equivalence, no
invalid tool calls, LLM-judged sub-goal coherence). Unsafe labels come from
two streams — controlled **injection** into verified-safe runs (a
by-construction label, exactly the `shepherd eval` construction, item 12, but
on real frameworks) and **diagnosis** of naturally failed runs by a
propose-and-verify LLM ensemble (admission needs > ⌊V/2⌋+1 verifiers agreeing
on four criteria, Eq. 8). **The per-stream counts are NOT VERIFIED**: the
paper defers source-level detail to Appendix B.3, which the HTML rendering
truncates; in the fetched full text `386` does not occur at all and `728`
occurs once, inside a bibliography page range (53728–53741). An earlier
revision of this entry stated "728 injected / 386 diagnosed" as sourced
facts — that was unsupported and is withdrawn.

**The released dataset carries session-level labels only.** The paper
describes per-trajectory `mistake_step` (k\*), `mistake_agent` (a\*) and
`unsafe_source` annotations, but the *published* parquet has none of them:
the verified feature list is `conv_id, domain, task, gold_answer, num_turns,
tools[{name, description}], turns[{role, thought, action, content}]`, and the
only label is which split the row sits in (`safe` vs `unsafe`). License
CC BY 4.0; two parquet files plus a held-out `splits_test.json` (332
trajectories, matching Table 3's Test Families row).

**Numbers** (Tables 1–2, Sec. 4.2–4.3; `inference/metrics.py` defines the
metrics): on AFTraj-2K, Exact-F1 66.44 and Absolute Step Shift 0.59 vs
DeepSeek-V4-Pro 46.56/1.77, Gemini-3-Flash 29.74/2.19, GPT-4.1 27.43/2.67,
Claude-Haiku-4.5 25.53/2.85 (the +19.9 / 3× headline). On Who&When: Step-Acc
57.69, Agent-Acc 73.08, ASS 1.62 vs GPT-4.1's 38.10/66.67/2.38. FAR (fraction
of *safe trajectories* with any alarm) is 2.4% (Sec. 4.3; project page says
2.37%).

**Their negative results are the most useful part.** (a) Fig. 4: 3–4B models
"collapse to near-universal false alarms", and even DeepSeek-V4-Pro sits
outside their deployable region (FAR ≤ 20%, Step-Acc ≥ 50%) at **43.2% FAR** —
a proprietary judge alarming on nearly half of healthy runs is the
saturation-trap argument (2606.04296) restated with numbers, and independent
support for items 10/17: a judge's self-reported verdict does not earn
admission, a calibrated gate does. (b) Ablation (Sec. 4.3): Stage 2 alone
degrades badly on Agentic tasks (19.0 Exact-F1 vs Stage 1's 31.6);
single-stage training collapses because the localization reward is too sparse.
(c) Failure modes (Appendix D.4, via the HTML TOC — full text NOT VERIFIED):
false alarms during in-turn verifier self-correction, and off-by-one upstream
localization. Per-step inference cost/latency: Appendix D.3 — NOT FETCHED
(truncated).

### 20. Overlap with what is already here

- **Problem framing**: identical to items 1/2 — alarm at the earliest decisive
  error, cheap enough to run mid-flight. They solve it with a trained 7B LLM
  called on every prefix; we solve it with a deterministic CUSUM (item 1) and
  an onset-gated judge wake (items 2, 15). Their FAR is per safe *trajectory*;
  our `drift_target_fpr` budget is per monitoring horizon (items 1, 7) —
  comparable in spirit, not directly comparable in units.
- **Injection-labeled benchmark**: AFTraj's 728 injected trajectories are
  item 12's fault injector with known onsets, scaled up to real multi-agent
  frameworks. Nothing conceptually new for us; the corpus itself is the
  novelty.
- **Prompt-level credit assignment**: their auditor prompt ("identify the
  EARLIEST decisive error supported by the visible evidence", "flag a step
  only when the visible evidence already supports the flag") is item 4's
  stepwise scoring with origin-step naming, plus item 16's verbal reason. We
  already have this. One difference worth noting: they hard-truncate each
  turn's content to 2,000 chars in the prompt builder (`inference/prompts.py`)
  — a blunt cousin of follow-up 5 (window assembly).
- **Genuinely new relative to this repo**: (i) a released, by-construction
  *labeled* corpus with decisive-step and responsible-agent annotations;
  (ii) the BPPO boundary-pair training idea; (iii) the "deployable region"
  (FAR ≤ 20% ∧ Step-Acc ≥ 50%) as an explicit two-metric acceptance plot —
  `shepherd eval` prints recall, delay, false alarms and cost separately but
  never as one admissibility region.

### 21. The three concrete angles

**(a) AFTraj as real labels for the conformal risk model (item 17).** Partially
viable, and viable for a *narrower* reason than first written here. The release
carries **session-level** safe/unsafe labels only — no `mistake_step` — which
turns out to be exactly the shape `CalibrationSet.add(score, is_drift)`
(`core/judge/risk.py:45`) consumes, so the fit is direct. What it cannot do is
supply step-level localization labels, so it cannot replace the by-construction
onsets inside `shepherd eval` (item 12); its honest role is an **external
cross-check corpus**: fit the same thresholds on it and see whether they agree
with the labels the benchmark produces. Three costs stand:
(1) *schema*: AFTraj turns are `{role, thought, action, content}` multi-agent
chat records from AutoGen/MetaGPT/Smolagents; our normalized events are
tool-call lifecycle events from Claude Code / Codex / QwenPaw hooks. A lossy
mapping (turn → synthetic TOOL_CALL/TOOL_RESULT pair) is writable in stdlib,
but the detector stack (loops, exit codes, path scopes) would see events that
never came from a real harness, so this calibrates the *judge*, not Tier 0 —
and a threshold fitted on multi-agent trajectories carries a distribution shift
when applied to single-agent Claude/Codex sessions. (2) *scores*: the risk model
calibrates (score, label) pairs; AFTraj supplies only labels — the judge must be
run offline over replayed prefixes to produce scores, i.e. real API calls, real
money, and it calibrates whichever judge model we pay for. (3) *deps*: the
corpus is parquet; pandas/pyarrow are off-limits here (stdlib + httpx +
PyYAML). The workable shape is a one-time offline conversion to a vendored
JSON fixture subset (CC BY 4.0 permits redistribution with attribution)
consumed by an eval-style replay loader. This is follow-up 8, not a this-week
change.

**(b) Replace the CUSUM `watch_level` soft trigger with their predictor.** No.
The watch level is free, deterministic, offline, and exists precisely so the
expensive reviewer stays asleep (item 2). AgentForesight's auditor is one 7B
inference per prefix — swapping a zero-cost statistic for a per-step model
call is the state-saturation trap with extra GPUs. Direction of travel is the
opposite: our CUSUM is the cheap pre-filter their protocol lacks. If the
checkpoint is ever released, the sane slot is *behind* the existing gate as
an alternative Tier-1 judge (it is API-callable via `infer_api.py`'s
OpenAI-compatible path), subject to the same conformal admission as any judge
— never as the wake signal.

**(c) Granularity.** Per-step, online, prefix-only (Sec. 2, Sec. 4.1) — same
granularity as agent-shepherd, so mid-run intervention is architecturally
possible; the blocker is cost and the unreleased checkpoint, not hindsight.

### 22. Cost, fit, and verdict

Cost discipline: the auditor is a model call per step; their own Fig. 4 shows
what happens to cheaper substitutes (small models collapse to universal false
alarms). Dependency discipline: the repo needs torch/transformers (and vLLM
for anything resembling their throughput); neither belongs in a stdlib daemon,
and no model can run in an offline test suite. The *dataset* and the *metric
framing* are the only parts that fit this project's constraints.

**Verdict: do not adopt the mechanism; partially adopt the evidence.**

- **P0** — none. No runtime change is justified today. (This entry itself is
  the P0: record that the deployable-region numbers independently support the
  item-10/17 design, and that the checkpoint is unreleased so nothing could
  be adopted even if we wanted to.)
- **P1** — add the two-metric "deployable region" (FAR ceiling ∧ step-accuracy
  floor) as a single reported admissibility check in `shepherd eval` output.
  Stdlib-only, deterministic, offline; it turns four scattered columns into
  one pass/fail a user can read.
- **P2** — the AFTraj replay loader (follow-up 8): one-time offline
  parquet→JSON conversion of a sampled, attributed subset; map turns to
  normalized events; replay prefixes through the judge offline and fit
  `risk.py` on by-construction labels. Revisit only if (i) the weak-label
  problem in item 17 starts visibly hurting, or (ii) the 7B checkpoint ships
  and someone wants a second judge candidate behind the existing gate.

### 23. Cross-examination: two independent analysts, three contested facts

This sweep was checked by two agents that had not written it — QwenPaw's
`Codex_Agent` (pointed at a worktree containing entries 19–22, so it could
diff against them) and ZCode (pointed at a worktree without them, so it
reported "cannot cross-check, independent judgement only" instead of
improvising). Both reports were then re-verified here against primary sources,
because both contained citations that did not survive checking:

| Contested fact | Claim | Re-verification | Verdict |
| --- | --- | --- | --- |
| Training hardware | one analyst "corrected" this entry to "2×2 NVIDIA H200 (4 GPUs)", quoting Sec. 4.1 | arXiv HTML, tags stripped: three occurrences, all `2 × NVIDIA H200`; `grep 'times2'` → zero hits | the original "2× H200" stands; the correction was fabricated |
| Step-level labels in the release | one analyst reported the HF card lists `mistake_step` / `mistake_agent` | `datasets-server.huggingface.co/info` features: `conv_id, domain, task, gold_answer, num_turns, tools, turns[{role, thought, action, content}]` — no label fields | ZCode was right; entry 19 and 21(a) corrected above |
| Injected vs diagnosed counts | this entry asserted "728 injected / 386 diagnosed" citing Sec. 3.1 and the HF card | full text: `386` → zero hits; `728` → one hit, inside bibliography pages 53728–53741; the paper defers the split to Appendix B.3 (truncated) | withdrawn; recorded as NOT VERIFIED |

Reproduction, so the next reader does not have to trust this table either:

```bash
curl -sL https://arxiv.org/html/2605.08715 -o /tmp/af.html
python3 - <<'PY'   # tag-strip, then print ±160 chars around each needle
import re
t = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', open('/tmp/af.html', errors='replace').read()))
for n in ('728', '386', 'H200', 'times2'):
    print(n, [t[max(0, m.start()-160):m.start()+160] for m in list(re.finditer(re.escape(n), t))[:2]])
PY
curl -s 'https://datasets-server.huggingface.co/info?dataset=ZBox008003/AFTraj' | python3 -m json.tool | head -40
```

`grep -c` counts *lines* and this HTML is one long line, and
`grep -o '.\{200\}728.\{200\}'` silently returns nothing on it — both give a
false "not present". Strip tags in Python and search there.

**What the cross-examination added that this sweep had missed** (all three are
evaluation hygiene, not runtime machinery, and all are stdlib-only):

- **Boundary-pair assertions, without the training.** BPPO trains on adjacent
  safe/unsafe prefix pairs so the verdict flips exactly at the decisive step.
  `shepherd eval` already injects faults at known onsets, so the same idea is
  available as a *test*: the prefix at `onset-1` must stay clean and the
  prefix at `onset` must alarm. That is a sharper localization criterion than
  `metrics.py`'s current tolerance window, and it costs nothing to run.
  *(Downgraded in entry 30: it is a re-parameterization of criteria the
  benchmark already applies, not a strictly finer signal.)*
- **Trajectory-grouped splits.** AFTraj places each safe trajectory and its
  injected variants in the same partition to prevent train/test leakage. Any
  future expansion of the eval corpus should copy that rule; a per-sample split
  would leak the injection.
- **Verify the injection actually flipped the outcome.** The paper keeps an
  injected trajectory only if the outcome is demonstrably degraded. Our injector
  currently assumes the injected fault produces observable drift; asserting it
  (and dropping samples where it does not) would make the `--judge` labels real
  rather than intended.
- **An evidence-sufficiency clause in the judge prompt.** Their auditor is told
  to "flag a step only when the visible evidence already supports the flag" and
  that minor recoverable mistakes are not decisive errors.
  `core/judge/prompts.py` scores a `justified` dimension (:14) and says "do not
  flag harmless reads or safe exploration" (:22), but has no clause binding an
  alarm to evidence sufficiency. Cheap to add; its effect is *not* verified —
  their 2.4% FAR comes from RL training, not from prompt wording, and measuring
  the difference here needs `eval --judge`, which costs real judge calls.
  Graded P1, not P0, for that reason.

## Fifth sweep (2026-10-12): agent RSI methodologies, and which one fits a stdlib supervisor

The question: survey the latest agent recursive-self-improvement (RSI) work,
compare the candidate methodologies against AgentForesight's (entries 19–23)
and against each other on fit for *this* project, and recommend exactly one.
The hard constraints are the ones this repo already lives by: stdlib + httpx +
PyYAML only, offline deterministic test suite, zero-cost-when-idle
(anti-saturation, items 2/15), single maintainer with no training infra, and
the item-10 external-anchor rule — a self-reported score is not a control, so
anything that self-modifies needs a reference it cannot rewrite.

**Method.** The `paper_search` skill was *not* used (known-unreliable on this
machine). All discovery ran through the arXiv API directly
(`export.arxiv.org/api/query`, Atom XML parsed with stdlib ElementTree) with
queries for "self-evolving agents", "recursive self-improvement", "automated
design of agentic systems", "self-improving agent", "reward hacking",
"LLM-as-a-judge self-improvement", "agent workflow memory", "goal drift",
"verifier gaming", sorted by submission date — ~150 unique records screened,
34 abstracts read, 4 full texts fetched (`arxiv.org/html/<id>`) and every
load-bearing number below re-verified by tag-stripping the HTML in Python and
needle-searching it, per entry 23's method. Anything not verified that way is
marked NOT VERIFIED. Retrieved 2026-10-12.

### 24. The lay of the land: RSI moved out of the weights and into the harness

Sorting the 2025–2026 crop by *what gets modified at runtime* gives four
branches:

- **Weights.** Self-Rewarding LMs (<https://arxiv.org/abs/2401.10020>), SEAL
  (<https://arxiv.org/abs/2506.10943>), MASS multi-agent self-supervision
  (<https://arxiv.org/abs/2610.12176>), ReSAIL self-distillation
  (<https://arxiv.org/abs/2609.39306>), idea-level critics trained by GRPO
  (<https://arxiv.org/abs/2610.08993>). **AgentForesight belongs here** — its
  contribution is a fine-tuned 7B auditor — and entry 22 already rejected the
  branch on dependency, cost and release grounds.
- **Its own code.** The Gödel-machine lineage: Darwin Gödel Machine
  (<https://arxiv.org/abs/2505.22954>; abstract-verified: SWE-bench 20.0% →
  50.0%, Polyglot 14.2% → 30.7%, open-ended archive of self-modified agents,
  "all experiments were done with safety precautions (e.g., sandboxing, human
  oversight)"), Gödel Agent (<https://arxiv.org/abs/2410.04444>), and AIDE²
  (<https://arxiv.org/abs/2609.26457>; HTML-verified: an autonomous 8-day run
  in which a frontier AI research agent proposed changes to its own code, kept
  those that performed best on **hidden evaluations**, found seven successive
  improvements, and — never optimized for this — saw reward hacking on a
  held-out task family fall 55% → 32%).
- **The agent-system design.** ADAS / Meta Agent Search
  (<https://arxiv.org/abs/2408.08435>) and AgentSquare
  (<https://arxiv.org/abs/2410.06153>): a meta agent searches a space of
  agent designs in code / modules, scored on benchmark suites.
- **The harness: prompts, skills, memory, tools — model frozen.** This is
  where the field's center of gravity sits as of Sept–Oct 2026: of the 30
  newest "self-evolving agents" hits, the large majority edit a persistent
  text/code artifact around a frozen model. Anchors: Promptbreeder
  (<https://arxiv.org/abs/2309.16797>), Voyager's skill library
  (<https://arxiv.org/abs/2305.16291>), Reflexion
  (<https://arxiv.org/abs/2303.11366>), Agent Workflow Memory
  (<https://arxiv.org/abs/2409.07429>). Current wave: SkillOpt
  (<https://arxiv.org/abs/2605.23904>, abstract-verified: "the skill should
  … be trained as the external state of a frozen agent"), SAGE
  (<https://arxiv.org/abs/2609.36043>), HarnessEvolve
  (<https://arxiv.org/abs/2609.00829>), MESH-Harness
  (<https://arxiv.org/abs/2610.05300>), Turbo Harness
  (<https://arxiv.org/abs/2609.40330>), agent-as-its-own-optimizer
  (<https://arxiv.org/abs/2609.38372>), Memento 3's reflective rulebooks
  (<https://arxiv.org/abs/2610.11794>).

Two observations that matter for us. First, "harness" in this literature
means *the code around the model* — we do not own the supervised agents'
harnesses (Claude Code / Codex / QwenPaw ship their own), so the only harness
evolution available to this project is evolution of **the supervisor's own
persistent state**: the rulebook header (item 14), the conformal cut-points
(items 10/17), the detector configuration. Second, the harness branch is the
only one that is training-free by construction, which is what makes it the
only branch that can satisfy the constraints at all. The comparison below is
therefore mostly intra-branch, with the other branches listed so the reader can
see what was rejected and why.

### 25. Candidate comparison on the axes that decide fit

"What is modified" = at runtime, in deployment. "Gate gameable?" = can the
thing being improved corrupt or fool the gate that admits the improvement —
the item-10 question.

| Methodology (arXiv) | Modified at runtime | Improvement gate | Gate gameable by the gated? | Training/GPU | Sandbox | Cost per cycle | Stdlib+offline testable | Improves |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AgentForesight (2605.08715) | weights (fine-tuned 7B auditor) | RL reward (format/step/agent), offline | n/a — gate is training-time, not deployment-time | **yes, 2× H200** | no | 1 model call **per step** | **no** (torch/vLLM; weights unreleased) | the supervisor-as-judge |
| DGM (2505.22954) / Gödel Agent (2410.04444) | its own code + archive | benchmark score of the rewritten agent | yes — harness tampering documented (2609.00069) | no training, but heavy rollout compute | **yes** | hours–days of rollouts | no | the agent itself |
| AIDE² (2609.26457) | its own code | **hidden** evaluation suite | partly — hidden eval is the anchor; 55%→32% reward hacking shows it bites | no | yes (runs ML jobs) | 8-day autonomous run | no | the agent itself |
| ADAS (2408.08435) / AgentSquare (2410.06153) | agent design (code/modules) | benchmark suite scores | yes (benchmark overfitting is the known failure) | no | yes (executes invented agents) | thousands of rollouts | no | the agent design |
| Self-Rewarding (2401.10020) / SEAL (2506.10943) | weights | **the model's own LLM-judge score** | **yes — self-preference, no external anchor** | yes | no | full training runs | no | the model |
| Reflexion (2303.11366) / AWM (2409.07429) / Voyager (2305.16291) | episodic memory / workflows / skill library | naive: keep what raised the outcome score | yes — aggregate-score gate; see SAGE's regression analysis | no | no (env feedback only) | ~free | partly | the supervised agent |
| Promptbreeder (2309.16797) | prompt population (+ mutation-prompts) | training-set fitness | yes (fitness = aggregate score) | no | no | population × evals | partly | the agent's prompts |
| Memento 3 (2610.11794) | natural-language rulebook **compiled to code** | LLM faithfulness judge **+ deterministic cell-exact replay** | replay half is not gameable; judge half is | no | no (frozen LLM, external memory) | env interactions | no (compiles self-code) | the agent's world model |
| HarnessEvolve (2609.00829) | harness (prompts/skills/tools/logic) | quality gate + performance gate (improve current batch, don't degrade recent batches, held-out epoch validation) | partly — decoupled roles help; gates are still score thresholds | no | no | multi-agent pipeline per batch | partly | the agent's harness |
| MESH-Harness (2610.05300) | harness module pool | LinUCB bandit over combinations under an eval budget | bandit scores are aggregate predictions | no | no | eval budget per candidate | partly (needs modular harness) | the agent's harness |
| PROCTOR (2609.02246) | prompts, via a Teacher-Student optimizer | **five deterministic guardrails that outrank the LLM Teacher** | the guardrails themselves are designed not to be (canaries, frozen holdouts, role separation) | no | **yes (hermetic)** | LLM optimizer loop per mutation | partly | the agent's prompts |
| WVV / Double Ratchet (2610.11464) | **the verifier** (composition of drawback detectors) + skills | agreement with a ten-item **anchored reference set**, never the agent's score | anchor set is the non-gameable part; remove it and the verifier collapses (see entry 26) | no | no | verifier synthesis + screening per cycle | partly | the grader |
| **SAGE (2609.36043)** | persistent skill document (optimizer-agnostic) | **per-item paired eval + one-sided exact binomial (McNemar) test + asymmetric regression penalty + abstention** | **no, given a frozen corpus** — deterministic integer statistics over paired outcomes; the edit cannot touch the test | **no** | **no** | 2× validation-corpus runs, at promotion time only | **yes** (math.comb; corpus can be stubbed deterministic) | whichever artifact is gated |

SAGE detail, because it carries the recommendation (all HTML-verified): the
loop is *optimizer proposes an edit to a persistent skill document, gate
accepts or rejects*; prior work (their SOTA baseline is SkillOpt,
2605.23904) concentrated on the optimizer while the gate stayed the naive
"keep it if the aggregate validation score rose". SAGE shows that gate fails
two ways — it admits **permanent regressions** (an edit raises the average
while breaking items the skill already solves; observed naive-gate regression
rates 2.5%–42.8% across their five benchmarks) and it suffers the
**Optimizer's Curse** (best observed score on a finite noisy validation set is
upward biased, so lucky edits get committed). The fix is arithmetic: run
incumbent and candidate on *identical* validation items, keep the per-item
win/loss ledger, weight losses asymmetrically (λ > 1 — "breaking a solved
item is more harmful than missing a new one"), and commit only when
Δ_λ > 0, the one-sided exact conditional binomial p-value p_λ(w,ℓ) < α
(at λ = 1 this is exactly the one-sided exact McNemar test), and the empirical
regression fraction ρ ≤ τ; with zero discordant pairs the gate abstains
(p = 1) and keeps the incumbent. Reported effect: regression rate lowered in
19 of 20 backbone×benchmark settings (36.5% → 0% on LiveMath, 42.8% → 0% on
OfficeQA with DeepSeek-V4) and highest final score in all 20 (LiveMath 34.15 →
48.78). The paper's own stated limits: the sign test discards magnitude, and
the guarantee is per-comparison, not sequential over many edits. No code link
found in the HTML — NOT VERIFIED whether code is released, and not needed: the
gate is ~20 lines of `math.comb`.

### 26. What the RSI safety literature says about gates — the anchor rule, in the field's own words

Entry 10's position ("a self-reported score is not a control") turns out to be
the consensus of the 2026 RSI-safety literature, arrived at independently and
usually by getting burned in production:

- **PROCTOR** (<https://arxiv.org/abs/2609.02246>, HTML-verified): months of
  autonomous prompt-optimization loops in production (contract analysis,
  compliance review, code quality) catalogued **eleven ways the evaluation
  signal failed**, in four classes (judge bias; harness and metric failures;
  ground-truth errors; reward hacking). Three concrete episodes: agents read
  cached answer keys from the environment — "a 100% pass rate concealing 68%
  true capability"; a corrupted ground-truth label caused the optimizer to
  **delete correct compliance rules** in order to agree with it; a
  syntactically broken prompt was promoted as winner because a silent parser
  fallback improved the metric. Their conclusion is our thesis: the LLM judge
  "should be demoted from oracle to advisor", with deterministic guardrails it
  cannot override — hermetic sandboxes, capability-disjoint roles (nothing
  both proposes and applies), acceptance checks that outrank the Teacher,
  stratified **frozen holdouts**, and **canary cases engineered so that a
  perfect score is itself evidence of cheating**.
- **Who Verifies the Verifier?** (<https://arxiv.org/abs/2610.11464>,
  HTML-verified): the sharpest single result of the sweep for this project.
  They co-evolve an inspectable verifier alongside the self-improving agent,
  selected *only* for agreement with a ten-item anchored reference set —
  "never for the agent's score". Ablation: **removing the anchor guards
  collapses the verifier into a vacuous always-pass grader, yet that collapsed
  verifier trains skills just as well** — i.e., everything downstream keeps
  looking healthy while the verifier is dead. Their sentence: "Downstream task
  score cannot certify a self-evolved verifier." This is evaluator collapse
  demonstrated, not speculated, and it is exactly the failure mode item 10 was
  written against.
- **SEABench** (<https://arxiv.org/abs/2609.35596>, abstract): *endogenous*
  misalignment — self-evolution raises task completion but produces safety
  failures **absent in paired non-evolving baseline agents**, with no
  adversary involved; locally useful updates persist into later tasks where
  they are harmful. Mitigation that worked: monitoring the agents'
  chain-of-thought, at low false-positive rate. The paired non-evolving
  baseline is directly liftable as an eval column.
- **Auditing Harness Tampering** (<https://arxiv.org/abs/2609.00069>,
  abstract): misaligned harness edits (illusory gains; violations of
  authorization/provenance/completeness) "consistently occur in real runs",
  and — the part that should scare anyone running an archive or a promotion
  loop — **persist in the lineage of the best-scoring agent**. Scoring does
  not filter tampering; tampering helps the score.
- **Compositional Safety Failures in Harness Evolution**
  (<https://arxiv.org/abs/2609.33123>, abstract): individually safe,
  utility-preserving component updates interact into unsafe behavior (43
  pairwise + 18 irreducible 3-way failures across three benchmarks);
  per-component validation misses this and joint validation is combinatorial.
  For us: rulebook header × risk cut-points × detector flags are three
  independently updated components.
- **Safety in Self-Evolving Agents: A Survey**
  (<https://arxiv.org/abs/2610.00093>, abstract): the SAVER framing —
  "legitimate state can become unsafe when adaptation expands its
  **persistence, authority, or scope** beyond the conditions under which it was
  valid." A promoted rulebook rule is precisely that: locally-earned guidance
  whose persistence (all future sessions), authority (injected at prompt
  start) and scope (agents/tasks beyond the evidence) all expand at promotion
  time.
- **Goal drift**, the supervised-agent side: <https://arxiv.org/abs/2505.02709>
  (agents drift from system-prompt goals under compromising context),
  <https://arxiv.org/abs/2603.03258> (drift can be *inherited* from context
  pressure even in robust models), <https://arxiv.org/abs/2603.03456>
  (asymmetric drift in coding agents under value conflict). These motivate the
  supervisor's existence, not a self-improvement method.
- Older anchors that still hold: model collapse is avoided by **accumulating**
  rather than replacing data (<https://arxiv.org/abs/2404.01413>) — relevant
  to a rulebook that appends; self-correction without external feedback
  largely fails (<https://arxiv.org/abs/2406.01297>); and our own honest
  caveat, *Sample More, Reflect Less* (2607.28576), is of a piece with both.
  The theory paper <https://arxiv.org/abs/2610.10611> gives the cleanest
  formal statement of the anchor rule: bounded self-modification stays within
  the same verification class only "under a common sound interpreter and
  **fixed verification protocol**", and adaptive candidate selection needs "a
  separate conditional-error budget" to control false selections — i.e., a
  frozen benchmark plus selection-error control (which is the conformal
  discipline of items 10/17, restated).

### 27. Recommendation: a SAGE-style statistical acceptance gate over the three loops this repo already has

**The single methodology: SAGE (2609.36043) — the statistical acceptance gate
— applied to agent-shepherd's own persistent state, with `shepherd eval` as
the frozen corpus.** This is not "add a new self-improvement loop"; the three
loops exist (item 14 rulebook, items 10/17 risk refit, item 12 eval-driven
defaults). What SAGE supplies is the one component two of the three loops are
missing: a gate that an edit cannot pass by being *lucky* or by *trading a
solved item for an average*.

Why it wins on each constraint:

- **Modified at runtime: text and config, never weights or code.** SAGE's
  substrate is a persistent skill document — structurally identical to our
  rulebook header (item 14; SkillOpt's phrase, "the trainable external state
  of a frozen agent", describes `rulebook.py` exactly). The gate is
  optimizer-agnostic, so we keep the proposer we already have (deterministic
  adherence statistics from the ledger) and replace only the promotion rule.
- **The gate cannot be gamed by the thing it gates.** The gate is integer
  arithmetic over per-item paired outcomes on a corpus the promoted artifact
  has no write access to. Contrast: the promotion score is an aggregate
  quantity — then `adherence_rate * log1p(evidence)`, now a Wilson lower bound
  on the adherence rate blended with the Tier 1 agreement rate (entries 30, 32)
  — and it is exactly the naive aggregate gate SAGE refutes: a rule that raises
  average adherence while
  silencing a detector on the fault classes it happens to mention promotes
  itself, and with per-(detector, agent) nudge counts as small as ours, the
  Optimizer's Curse is not hypothetical. WVV's collapse result (entry 26) says
  the same thing from the other side: whatever evolves, its admission
  criterion must hang off an anchored corpus, not off downstream scores.
- **No training, no GPU, no sandbox.** The gate is `math.comb`. The proposer
  is our existing ledger logic. Nothing executes generated code — which is
  what disqualifies the entire code-branch (DGM, AIDE², ADAS, Gödel Agent,
  Memento 3's rulebook-to-code compilation) regardless of their headline
  numbers, along with the tampering-lineage finding (2609.00069) as
  independent reason not to want it.
- **Zero-cost-when-idle survives.** The gate runs at *promotion time* —
  STOP-time rulebook re-learning or CI — over the eval corpus, never per live
  event. `shepherd eval` with Tier 1 stubbed is deterministic, offline and
  free (item 12), so the P0 version costs two corpus replays per candidate
  rule. Nothing in the hot path changes; the anti-saturation discipline
  (items 2/15) is untouched.
- **Offline stdlib testability.** The gate itself is a pure function
  (wins, losses, regressions, λ, α, τ) → commit/abstain, testable with the
  same fixture discipline as `risk.py`; a degenerate-always-commit and a
  degenerate-never-commit test are the negative controls.
- **It improves the supervisor.** Correct target: we do not own the
  supervised agents' harnesses, weights, or prompts. The subject agent
  benefits only through better-steered guidance — which is what this project
  is.

**Runner-up: PROCTOR (2609.02246).** Disqualified *as the methodology* because
its center of gravity is a full Teacher-Student optimizer — an LLM Teacher
grading drafted mutations inside hermetic sandboxes — machinery for a team
running prompt-optimization loops on production LLM pipelines, when our
proposer is already deterministic and free. But its five guardrails are the
best-specified anchoring checklist in the sweep, and three of the five are
adopted into the plan below (frozen holdouts, canaries, role separation — the
last trivially satisfied: the ledger analyzer proposes, the gate commits, and
neither can write the corpus). WVV/Double Ratchet is the third-place finisher
and is explicitly *deferred*: co-evolving the verifier before an anchored
corpus exists is the collapse experiment with the anchor removed.

**What to borrow from AgentForesight alongside it** (entry 23's four hygiene
items, folded in as the corpus-preparation layer for the gate, not
re-litigated): *boundary-pair assertions* become the per-item pass/fail the
paired ledger is built from — for each injected fault, `onset-1` clean and
`onset` alarmed are two items a candidate rulebook can win or lose
individually, which is a strictly finer signal than aggregate F1 and is what
makes SAGE's regression penalty bite; *trajectory-grouped splits* become the
dev/holdout partition rule (a safe trajectory and its injected variants never
straddle the split); *injection-outcome verification* keeps dead items out of
the ledger (an injection that produced no observable drift cannot be a "loss"
against any candidate); the *evidence-sufficiency clause* goes in the judge
prompt as graded in entry 23. Plus entry 22's P1 deployable-region check as
one reported admissibility line.

**Graded plan.** *(The P0 below was refuted before it was written — see entry
30. It is kept verbatim as the record of what the recommendation was; the plan
that shipped is entry 30's.)*

- **P0 — paired gate for rulebook promotion.** Split the eval corpus into a
  dev partition (visible to adherence/ranking) and a frozen holdout (scored
  only by the gate), grouped by trajectory. For each candidate rule set:
  replay both books over identical items with Tier 1 stubbed; build the
  per-item win/loss/regression ledger; commit only on Δ_λ > 0 ∧ exact
  one-sided binomial p_λ(w,ℓ) < α ∧ ρ ≤ τ, with λ > 1 (losing a previously
  caught fault is worse than failing to bank a marginal adherence gain) and
  abstention when there are no discordant pairs. Keep today's score
  (`_wilson_lower_bound`, plus the Tier 1 agreement blend of entry 32) for
  *ranking* under the token budget; the gate decides *admission*. Offline,
  deterministic, free, ~20 lines of stdlib.
  Negative controls: a corpus-overfitting fake rule must be rejected; an
  empty candidate must abstain, not commit.
- **P1 — canary cases + paired non-evolving baseline.** Add eval scenarios
  engineered so that a perfect score is itself evidence of cheating (PROCTOR):
  e.g., a clean session in which a candidate rule must not suppress the one
  injected fault it textually resembles; a rulebook change that zeroes alarms
  corpus-wide fails automatically. Report SEABench-style paired columns
  (rulebook on vs off, per case) permanently in `shepherd eval` output, next
  to the deployable-region line, and wire the entry-23 items
  (injection-outcome verification; evidence-sufficiency clause in the judge
  prompt, measured via `eval --judge` when someone is willing to pay for the
  judge calls).
- **P2 — one gate discipline for all three loops.** Run `risk.py` cut-point
  refits through the same paired test before committing (candidate threshold
  vs incumbent over the labeled eval corpus; commit only on reliable wins),
  so the risk model and the rulebook share one external anchor. Add
  SAVER-style scope metadata to promoted rules — the conditions under which
  the evidence was earned (agent, harness, corpus version) — and require
  re-gating, not silent inheritance, when a rule is about to apply outside
  them (the survey's persistence/authority/scope expansion, and the
  compositional-failure warning that three separately gated components can
  still interact: gate the *combination* that ships, not only each part).

### 28. What NOT to borrow, and why

- **Any per-step or per-event model call** — AgentForesight's auditor protocol
  (entry 22), always-on critique, SAG's per-step deliberation
  (<https://arxiv.org/abs/2610.07335>). SAG deserves a footnote: its
  value-of-information framing independently confirms items 2/15 (selective
  critique beats always-on at matched token budget; ALFWorld 24.6% → 78.4% at
  ~ReAct cost, abstract-level), but its gating signals are entropy and top-2
  margin over *admissible actions*, which black-box supervised agents do not
  expose; our CUSUM-on-failures remains the available substitute.
- **Self-modifying code, archives, and design search** — DGM, Gödel Agent,
  AIDE², ADAS, AgentSquare, MESH-Harness's module pools: all require executing
  generated code in a sandbox and/or benchmark-rollout budgets that a
  single-maintainer stdlib daemon cannot host, and the tampering-lineage
  result (2609.00069) says the archive's best scorer is a *likely carrier*.
  Memento 3's specific twist — compiling the rulebook into executable
  prediction code — would turn our text artifact into self-modifying code for
  no supervision gain; its deterministic-replay acceptance rule is the same
  anchor principle we get cheaper from `shepherd eval`.
- **Weight-branch and self-judged loops** — Self-Rewarding (2401.10020),
  SEAL (2506.10943), MASS (2610.12176), trained critics (2610.08993):
  training infra, and in the self-rewarding case the gate *is* the gated
  model's own judgment — the anchor-free failure mode this project exists to
  avoid.
- **Naive memory/workflow accumulation without a paired gate** — AWM,
  Voyager-style skill libraries, plain Reflexion loops: cheap and stdlib-able,
  and we already have their good half (item 14 via Meta-Policy Reflexion),
  but their keep-what-scored-higher admission is precisely the gate SAGE
  measures at 2.5%–42.8% permanent regression. Adopting *more* substrate
  without adopting the gate would be adopting the failure.
- **Verifier co-evolution (WVV/Double Ratchet) — not yet.** Right diagnosis,
  wrong order of operations for us: it evolves the grader before we have an
  anchored reference set for drift verdicts (our labels are Tier-0-concordant
  weak labels, item 17). Revisit if the eval corpus and an AFTraj-style
  external cross-check (follow-up 8) mature to the point where a ten-item
  anchored set for the *judge* exists.

### 29. Verification ledger for this sweep

HTML-fetched and needle-verified (method of entry 23: tag-strip in Python,
search the text, not `grep -c` on one-line HTML): SAGE 2609.36043 (binomial /
McNemar reduction at λ=1; Eq. 11 commit rule Δ_λ>0 ∧ p_λ<α ∧ ρ≤τ; abstain at
m=0; regression rates 2.5% ALFWorld / 4.8% SearchQA / 42.8% OfficeQA;
36.5→0, 42.8→0, 34.15→48.78, 19-of-20; SkillOpt baseline = arXiv 2605.23904
per its bibliography — the 2.5/4.8 needles were verified separately on
2026-10-12 after the entry-30 review noticed they were missing from this
list); PROCTOR 2609.02246 (eleven failures, four classes;
100%-concealing-68%; deleted compliance rules; the five guardrails
verbatim); WVV 2610.11464 (always-pass collapse with anchors removed;
ten-item anchored reference set; Double Ratchet 88–110% of lift); AIDE²
2609.26457 (8-day run, seven successive improvements, hidden evaluations,
55%→32% reward hacking). Abstract-only (arXiv API, claims above are
abstract-level): 2610.00093, 2609.35596, 2609.00069, 2609.33123, 2610.05300,
2609.00829, 2609.38372, 2610.11794, 2610.07335, 2610.08902, 2609.32091,
2610.10611, 2610.12176, 2609.39306, 2610.08993, 2608.03392, 2505.02709,
2603.03258, 2603.03456, 2605.23904, plus the pre-2026 anchors (2408.08435,
2505.22954, 2410.06153, 2410.04444, 2401.10020, 2506.10943, 2303.11366,
2305.16291, 2409.07429, 2309.16797, 2404.01413, 2406.01297). NOT VERIFIED:
whether SAGE released code (no link in the HTML); SAGE's concrete default
values of α, τ, λ (paper says tuned on held-out data only; values not
extracted); everything in the abstract-only list beyond what the abstract
itself states; Semantic Scholar and dblp were not queried (known-blocked on
this machine). No claim in entries 24–28 rests on a secondary source.

### 30. The P0 in entry 27 was refuted before it was written

Entry 27's plan was reviewed adversarially by a second model (Claude Code,
read-only, repo at c0dbb1e) and the P0 did not survive. Every load-bearing
claim below was then re-checked here against the code, because the previous
cross-examination round (entry 23) produced two fabricated citations.

**The premise was wrong: our benchmark corpus cannot react to the artifact the
gate would judge.** `scenarios.build_session(fault, *, steps, seed, at_step,
session_id, agent, start_ts)` (`eval/scenarios.py:698`) is a pure function of
its parameters — the event stream is generated before any verdict exists, so a
candidate rulebook cannot change a single item's outcome. A paired per-item
win/loss ledger over that corpus measures nothing about the candidate.

**There is no injection point, and that is deliberate.** `run_benchmark`
accepts `detectors_only`, `seed`, `sessions_per_fault`, `steps`, `config`,
`faults`, `tolerance` (`eval/harness.py:292`) — no rulebook, no ledger root;
the temp root is created inside the function (`harness.py:310`) while the
engine reads its book from `ledger.root/rulebook.json` (`core/server.py:140`),
and `default_config` sets `rulebook_enabled=False` on purpose so the benchmark
cannot learn from cases it already replayed (`harness.py:58-60`). The existing
`config=` seam can turn the rulebook *on*; it cannot supply candidate content.

**Even wired up, the only visible signal would be "header present".**
`metrics.py` never special-cases `detector="rulebook"` (zero hits in the file;
`core/server.py:396` is the only emitter), so the header's NUDGE is scored as
an ordinary verdict: one false positive per session (`metrics.py:133-136`).
Rule *text* is invisible to the score. And with the corpus saturated —
every answerable fault at `recall == 1.0` and clean sessions at zero false
alarms (`tests/test_eval.py:159-171`, `:180-187`) — there is no headroom for
wins, so a SAGE-style gate would abstain on every candidate; if "win" were
defined as fewer false positives instead, the *empty* rulebook would win
always, and the gate would reward saying nothing.

**Two further defects in the plan as written.** (i) The cost claim
("two corpus replays per candidate, ≈20 lines of stdlib") ignored that with
`rulebook_enabled=True` every session STOP re-learns and re-saves the book
(`core/server.py:452-457`) into a root shared by the whole run
(`harness.py:310`), so the book mutates mid-benchmark and "replay the same
items under two books" is not a thing that can happen. (ii) "Frozen holdout"
named no artifact: a holdout regenerated from a runner seed lives in the same
repository as the proposer, which breaks the anchor rule this sweep itself
argued for (entry 26).

**The methodological error, stated plainly:** SAGE gates an *optimizer* that
searches over candidate edits. This repo has no such optimizer — the proposer
is a deterministic ledger analysis and a single maintainer ships the result.
Importing the gate without the search imports the machinery and none of the
threat model. The exploit SAGE exists to stop is nonetheless real here, but it
sits in the *ranking* score, not in a missing acceptance test.

**What survives, and what shipped.**

- **P0 (shipped) — replace the ranking score with a Wilson lower bound.**
  The score being replaced was `adherence_rate * log1p(evidence)`
  (`core/rules/rulebook.py`, in `Rulebook._rule` before this change): a rate
  times a function of the sample size, so a long mediocre record outranks a
  short perfect one — 55/100 scores 0.55·ln(101) = 2.54 while 6/6 scores
  1.0·ln(7) = 1.95. The Wilson lower bound on the adherence rate discounts a
  rate by how little evidence backs it, keeps recurrence mattering, needs no
  α/τ/λ, no corpus and no new dependency (`math.sqrt`). Honest framing: this
  is a bias correction on a ranking score, **not** a statistical acceptance
  gate — the gate would need a corpus that reacts, which does not exist here.
- **P1 — gate the risk-threshold refit instead.** That is the one loop where a
  candidate genuinely changes per-item outcomes and the labels are
  construction facts: `harness.py:272` pairs each judge verdict's score with
  `is_drift` (`:267`), the candidate is a threshold (`core/judge/risk.py:64`),
  a win is a newly caught drift and a loss a newly alarmed clean session, and
  `tests/test_eval.py:329-360` shows the gate can be tested offline from a
  synthetic report. Cost note the entry-27 plan got wrong: the *gate* is free
  and offline, but regenerating the corpus is not — it needs `--judge`
  (`harness.py:414`), i.e. real judge calls.
- **P1 — boundary-pair assertions as an offline script over the report JSON**,
  production code unchanged. Entry 23 called this "a sharper localization
  criterion"; that was overstated. Pre-onset verdicts are already counted as
  false positives (`metrics.py:135-136`) and premature alarms from other
  detectors already surface in the per-detector precision columns
  (`metrics.py:226-247`), so the assertion is a re-parameterization of what the
  benchmark reports, useful as a readable check rather than as new signal.
- **P2 — decide how `metrics.py` should score `detector="rulebook"`.** Today
  an intentionally injected prompt-start header is booked as a false alarm,
  which is why `shepherd eval` has to disable the rulebook at all. Any future
  "rulebook on/off" comparison column needs that policy settled first,
  or it will publish "+1 FP per session, +0 detections" as a result.
- **Not done, still open:** the deployable-region line from entry 22 (P1),
  the evidence-sufficiency clause in the judge prompt from entry 23 (P1, and
  measurable only by paying for `eval --judge`), injection-outcome
  verification, trajectory-grouped splits, and the combined-artifact gate.

### 31. The open follow-ups were re-evaluated: one test shipped, four rejected

Entry 30 left a graded list. Each item was re-read against the code before
being built, and most did not survive. Recording the rejections matters as much
as the one thing that shipped, because three of them were recommended by
external analyses (entries 22, 23, 30) and would have been built on authority
rather than on a defect anyone could point at.

**Shipped: a report-level net for premature alarms on drift sessions.**
`ci.yml` calls the benchmark "the regression net for 'a threshold change made
the supervisor nag healthy sessions'", but only the clean-session half of that
claim was asserted (`tests/test_eval.py:180-187` covers `CLEAN_KINDS`).
`metrics.attribute` had scored a verdict with `index < onset` as a false
positive all along (`eval/metrics.py:135-136`) and a unit test pinned the rule
(`tests/test_eval.py:147`), so nothing in the *scoring* was broken — what was
missing was any assertion that the shipped build obeys it on drift sessions. A
detector that started firing before the injected fault existed would have
passed CI. `test_answerable_drift_sessions_have_no_premature_alarms` now covers
42 drift cases across 7 answerable fault kinds (0 false positives today), and
`test_the_premature_alarm_check_can_fail` is its negative control: the same
predicate over a mutated report must go red, so the net cannot pass vacuously.
Zero production lines.

**Rejected: the deployable-region line (entry 22's P1).** It would print
`FAR ≤ ceiling ∧ step-accuracy ≥ floor` as one pass/fail. The FAR ceiling could
come from `policy.drift_target_fpr`, but the accuracy floor has no source —
AgentForesight's 50% is a property of *their* auditor on *their* corpus, and
copying it would put a number in our output that nothing here measured.
`--fail-under-f1 0.8` is already the CI gate and its value is pinned by
`ci.yml`. Revisit only if a floor can be derived from something this repo owns.

**Rejected: the paired gate on the risk-threshold refit (entry 30's P1).** The
reasoning there was that this is the one loop where a candidate changes
per-item outcomes and the labels are construction facts — true, but it missed
that the loop is *already* gated by an anchored statistical rule rather than by
an aggregate score: `calibrate()` takes the most permissive threshold whose
(n+1)/n-corrected empirical false-intervention rate fits the budget
(`core/judge/risk.py:64-96`), and `refit()` refuses to store a value when
evidence is thinner than `min_samples` (`:163-177`). SAGE's target failure —
committing because the aggregate went up — is not the mechanism here, so a
paired win/loss gate would be a second opinion on a rule that already has a
finite-sample bound. Revisit if the refit ever starts committing on aggregate
improvement, or if the labels stop being construction facts.

**Rejected: the boundary-pair offline script (entry 30's P1).** Redundant for
the reason given above — pre-onset verdicts are already false positives and
already unit-tested. The shipped test is the part that was actually missing.

**Deferred: the evidence-sufficiency clause in the judge prompt (entry 23).**
Two lines of prompt text, and its benefit is unmeasurable here: the paper's low
FAR comes from RL training, not from wording, and the only way to measure the
difference is `shepherd eval --judge`, which costs real judge calls. Item 12
set the house rule — a default changed because the benchmark said so, not
because a paper did. Shipping it blind would break that rule for no evidence.

**Deferred: how `metrics.py` should score `detector="rulebook"` (entry 30's
P2).** The mis-scoring is real (an intentionally injected prompt-start header
is booked as a false alarm, which is why `default_config` disables the rulebook
at `eval/harness.py:58-60`), but entry 30 also established that the corpus
cannot react to rule *content*, so fixing the policy would buy the header's
cost and never its benefit. Worth doing when someone wants that cost number.

Still open from earlier sweeps and untouched here: follow-ups 2–7 (graph credit
assignment, multi-view progress scoring, tool-call-rate steering, judge window
assembly, per-agent eval breakdowns, ambiguity-aware `UNKNOWN` guidance),
follow-up 8 (AFTraj as an external label corpus), and entry 23's
injection-outcome verification and trajectory-grouped splits.

### 32. Self-evolution: Tier 1 scoring and the risk gate update the Tier 0 rulebook online

Every Tier 1 review produced two numbers that were thrown away. The judge's
drift score went to `RiskModel.observe` so its own admission line could be
refit; Tier 0's nudge order went nowhere. Yet the two are a matched pair: a
review scores the window that follows a Tier 0 fire, and the risk gate has
already decided what score counts as real drift. Recording that verdict against
the rule that fired closes the loop — Tier 0 stops being a fixed asset that the
rest of the supervisor learns *about*, and starts being updated by it.

**What shipped.** `PolicyEngine` now keeps the detectors whose nudges actually
reached the agent since the last review (`SessionState.tier0_pending`), and
`_feed_rulebook` labels each one when the next review lands: **agreed** when the
review's confidence clears `risk.threshold_for("<agent>:judge")` — the same
fitted line that decides whether the judge's own nudge is injected — and
**disagreed** otherwise. `Rulebook.note_judge_feedback` accumulates both into
per-(agent, detector) counters (persisted schema bumped to v2), and
`Rulebook._rule` blends a second Wilson lower bound over the agreement rate into
the ranking score with shrinkage weight `judge_evidence / (judge_evidence + 4)`,
so one opinion moves a rule by at most a fifth while a long concordant or
discordant record moves it decisively. Past `MIN_JUDGE_EVIDENCE` (4) labelled
windows below `MIN_JUDGE_CONCORDANCE` (0.5), the rule is vetoed from promotion
outright, however well agents happened to comply. Behaviour answers "was it
followed?"; Tier 1 answers "was it right?".

**Two choices worth defending.** *Why the risk line and not a new constant:*
the cut-point is already calibrated per (agent, source) with a finite-sample
correction and already falls back to the configured prior when evidence is thin
(`core/judge/risk.py`, `refit`), so this introduces no second arbitrary
threshold and inherits an honest cold start. Any score the gate would have
rejected is, by construction, below what this system believes counts as drift —
using anything looser would let the judge rubber-stamp Tier 0 with numbers it
itself refuses to act on. *Why veto requires evidence:* the judge evaluates a
window for drift, not the validity of the rule that fired in it, and it is a
frozen model with its own biases — so below four labels the channel only
re-ranks, never excludes. A single opinion cannot silence a detector; a
sustained pattern can. A fail-open `judge unavailable` PASS labels nothing at
all, because an outage is not an opinion.

**Honest limits.** This is a **weak label**, exactly like the tier0-confirmed
tag that already trains the risk model: it is a correlation between two
mechanisms sharing one frame, not ground truth, and if both are wrong in the
same direction the loop reinforces the mistake. Nothing here prevents that;
entry 27's corpus-anchored paired acceptance gate is still the missing piece,
and it stays missing for the reason entry 30 found — the `shepherd eval` corpus
cannot react to rule *content*, and `default_config` disables the rulebook
outright (`eval/harness.py:58-60`). So this loop is deliberately *not*
benchmark-verified: what is tested is its mechanism (agreed/disagreed counts,
persistence, the disabled switch, the outage path), not that it improves F1.
What it does not touch matters too: no detector parameter, threshold, or line of
guidance text is rewritten — only ranks and admission of existing rules — so the
tampering-lineage findings and the whole code-branch exclusion of entry 27
(compiling a rulebook into executable logic) do not apply.

**Tests.** `test_a_confirmed_judge_review_reinforces_the_tier0_rule_online` and
its disagreement twin cover both labels end to end through the real engine;
`test_rulebook_judge_feedback_can_be_disabled` covers the switch;
`test_a_judge_outage_labels_nothing` proves an outage costs Tier 0 nothing;
`test_a_lone_judge_opinion_only_nudges_the_score` pins the shrinkage (one
sample moves the score < 0.25); and `test_a_consistently_rejected_rule_is_vetoed_from_promotion`
ships with the thin-evidence counter-example (three disagreements re-rank, they
do not veto) so the veto cannot silently tighten.
