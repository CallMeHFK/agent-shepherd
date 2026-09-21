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

## Follow-ups considered, not implemented

1. **Shepherd rulebook injection into the live loop** — *Meta-Policy Reflexion*
   (<https://arxiv.org/abs/2509.03990>): the adherence analysis and the rendered
   "house rules" header exist (`core/rules/rulebook.py`), but the engine does not
   yet inject that header at `PROMPT_SUBMIT`. The ledger already records
   everything needed; this is wiring, not research.
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
