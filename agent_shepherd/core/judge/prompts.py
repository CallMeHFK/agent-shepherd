"""System/user prompts for the Tier 1 step scorer."""

from __future__ import annotations

from ..rules.signals import UNKNOWN, classify
from ..types import AgentEvent

SYSTEM_PROMPT = """You are the supervisor for an AI coding agent. You observe the agent's reasoning and tool calls in real time and detect when the agent is drifting away from the user's goal.

Method (stepwise scoring with credit assignment):
1. Read the GOAL pinned at the top of the input; it is the reference for "on_goal".
2. Score each numbered step in the window separately on three dimensions:
   - on_goal: does this step move toward the stated goal?
   - justified: is the step justified by the reasoning and evidence that produced it?
   - verified: does this step (or the step that follows it) check its own effect — read output, run tests, inspect a diff?
3. Locate the drift origin: the earliest step where the trajectory first diverged from the goal. Do not blame a later step that merely repeated the consequences of an earlier mistake; the guidance must point at the origin step.
4. Let the overall action follow from the worst scored step, not from a holistic impression of the window.

Be skeptical but fair. Do not flag harmless reads or safe exploration. Flag real drift: loops, regressions, edits outside the goal, unverified claims, reasoning that ignores evidence.

Tool results are tagged [FAILED], [OK] or [UNKNOWN]. [UNKNOWN] means nothing observable came back — a lost response, an empty result, a timeout after the action may already have run. Treat that as an unverified effect, not as a success: the right guidance is to check whether the side effect happened before retrying, never to retry blindly. Do not tell the agent to "read the error output" when there was none.

Respond with strict JSON only, in this exact shape:
{"action": "pass" | "nudge" | "block" | "escalate", "reason": "...", "guidance": "...", "confidence": 0.0}

- action "pass": no intervention needed
- action "nudge": inject advisory guidance into the agent's next context
- action "block": stop the offending action (high-risk only)
- action "escalate": pause and hand control to the human

If the action is not "pass", name the drifted step by its number in the reason (e.g. "step 3 drifted: ..."). Keep guidance short (under 500 tokens), concrete, and actionable: if the agent loops, tell it to stop repeating the same call and reassess; if it made a regression, tell it to inspect the diff since the last passing state; if it is editing files outside the goal, tell it to re-read the request and confirm scope.

Never output markdown, never output prose outside the JSON."""


def _goal_line(events: list[AgentEvent], current: AgentEvent) -> str:
    """The user's most recent stated goal, or a fallback marker.

    Pinned to the top of the judge prompt on purpose: with the goal in the
    middle of a long window, models underuse it ("lost in the middle"), and
    the drift reference frame is exactly the part of the context the judge
    must not lose.
    """
    for e in reversed(events):
        if e.event.value == "prompt_submit" and e.prompt:
            return e.prompt.strip()[:1000]
    if current.prompt:
        return current.prompt.strip()[:1000]
    return "(no explicit goal recorded; infer the intent from the steps themselves)"


def build_user_prompt(events: list[AgentEvent], current: AgentEvent) -> str:
    """Render the recent session window for the judge.

    The goal is pinned at the top and every window line is numbered so the
    judge can score step by step and name the drifted step by index. Tool
    results are labelled with their classified outcome, because "no output" is
    evidence of a different kind from output that says nothing went wrong — and
    an unlabelled empty line reads as a silent success.
    """
    window = events[-20:]
    lines: list[str] = [f"GOAL (pin this as your reference):\n{_goal_line(events, current)}", ""]
    for i, e in enumerate(window, start=1):
        ts = f"[iter={e.iteration}]"
        if e.event.value == "tool_call" and e.tool:
            lines.append(f"STEP {i} {ts} TOOL_CALL {e.tool.name} input={e.tool.input!r}")
        elif e.event.value == "tool_result":
            outcome = classify(e)
            result = (e.tool_result or "")[:500]
            if outcome.state == UNKNOWN:
                result = "(no observable output — response lost or empty)"
            lines.append(f"STEP {i} {ts} TOOL_RESULT [{outcome.state.upper()}] {result}")
        elif e.event.value == "reasoning" and e.reasoning:
            lines.append(f"STEP {i} {ts} REASONING {e.reasoning[:500]}")
        elif e.event.value == "prompt_submit" and e.prompt:
            lines.append(f"STEP {i} {ts} USER_PROMPT {e.prompt[:500]}")
        elif e.event.value == "iteration_end":
            lines.append(f"STEP {i} {ts} ITERATION_END")
        else:
            lines.append(f"STEP {i} {ts} {e.event.value}")
    lines.append(f"[current] {current.event.value} iteration={current.iteration}")
    return "\n".join(lines)