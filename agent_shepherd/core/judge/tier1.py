"""Tier 1: PRM-style step scorer backed by the LLM judge."""

from __future__ import annotations

from ..config import JudgeConfig
from ..types import AgentEvent, Verdict
from .client import JudgeClient
from .prompts import SYSTEM_PROMPT, build_user_prompt


class StepScorer:
    """Score a sliding window of recent steps and return a verdict."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        self.client = JudgeClient(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout=config.timeout,
        )

    def score(self, current: AgentEvent, history: list[AgentEvent]) -> Verdict | None:
        """Ask the judge to score the recent window.

        Returns ``None`` only if the judge says "pass". Any other verdict is
        returned verbatim so the daemon can record it.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(history, current)},
        ]
        text = self.client.complete(messages, max_tokens=512, temperature=0.0)
        return self.client.parse_verdict(text)