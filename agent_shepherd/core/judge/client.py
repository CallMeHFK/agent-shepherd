"""OpenAI-compatible judge client.

Works with any OpenAI-compatible endpoint: Agnes AI Hub (default), a local
vLLM server, cc-switch, etc. The client is intentionally thin — the judge
prompt and scoring logic live in :mod:`agent_shepherd.core.judge.tier1`.
"""

from __future__ import annotations

import json

import httpx

from ..types import Verdict, VerdictAction


class JudgeClient:
    """Small OpenAI-compatible chat-completions client for the judge."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        # Token usage of the most recent call. The supervisor's own cost is the
        # number that decides whether it is worth running, so it is recorded in
        # the ledger rather than left as a guess (the reviewer's cost is what the
        # saturation-trap literature is really about).
        self.last_usage: dict[str, int] = {}

    def usage_totals(self) -> dict[str, int]:
        return dict(self.last_usage)

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Call the judge model and return its raw text response."""
        if not self.api_key:
            raise RuntimeError("judge api_key is empty")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"judge request failed ({resp.status_code}): {resp.text[:300]}") from exc
        data = resp.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        self.last_usage = {
            str(k): int(v) for k, v in usage.items() if isinstance(v, (int, float))
        } if isinstance(usage, dict) else {}
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"judge returned malformed response: {data!r}") from exc

    def parse_verdict(self, text: str) -> Verdict:
        """Parse a judge response into a :class:`Verdict`.

        The judge is instructed to emit strict JSON. If parsing fails, fall
        back to a conservative NUDGE so a broken judge never silently passes.
        """
        text = text.strip()
        try:
            # Some models wrap JSON in a code fence.
            if text.startswith("```"):
                text = text.strip("` \n")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            obj = json.loads(text)
            action = VerdictAction(str(obj.get("action", "pass")))
            return Verdict(
                action=action,
                reason=str(obj.get("reason", "judge found drift")),
                guidance=obj.get("guidance"),
                confidence=float(obj.get("confidence", 0.5)),
                detector="judge",
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            # Conservative fallback: a broken judge should never be silent.
            return Verdict(
                action=VerdictAction.NUDGE,
                reason="judge response could not be parsed; applying conservative guidance",
                guidance="The supervisor judge could not parse its own response. Please pause, summarize the current state, and take one small verifiable step.",
                confidence=0.5,
                detector="judge",
            )