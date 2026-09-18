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

1. **Shepherd rulebook** — *Meta-Policy Reflexion*
   (<https://arxiv.org/abs/2509.03990>): persist the recurring nudge texts
   that actually work (per-agent "reflective memory") plus hard admissibility
   checks, so repeated failures become rules the agent sees up front. The
   ledger already records everything needed; this is an analysis + injection
   layer on top.
2. **Graph-based credit assignment** — GDCR (2605.29697): replace the
   "name the origin step" heuristic with a causal graph over steps for the
   judge's verdict. Higher precision, much more machinery.
3. **Self-normalized CUSUM** — <https://arxiv.org/abs/2509.07112>: the
   current baseline/slack are fixed; a self-normalized version would adapt to
   per-session failure baselines (agents vary a lot in how flaky their
   environment is).
4. **Multi-view progress scoring** — *ProgRouter*
   (<https://arxiv.org/abs/2608.25992>): the scalar CUSUM is one view of
   "progress"; *SiLR* (<https://arxiv.org/abs/2609.04629>) warns that no
   single scalar surrogate is sound for multi-dimensional violation state.
   A 2–3 view statistic (failure rate + novelty of outputs + goal coverage)
   would be the principled next step.
5. **Tool-call-rate steering** — <https://arxiv.org/abs/2608.25198>: the
   nudge/block levers are the steering knobs; their effect on call frequency
   is currently unmeasured and would be a natural experiment.
6. **Context assembly** — *ContextPipe*
   (<https://arxiv.org/abs/2609.00749>): the judge window is a fixed last-20
   slice; treating window selection as a retrieval/assembly problem (include
   the goal, the last failure, the last verification) is likely to beat the
   fixed slice on long sessions.
