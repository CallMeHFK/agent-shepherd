"""Append-only audit ledger.

Every observation and every verdict is written to
``~/.shepherd/sessions/<agent>/<session>.jsonl`` — one JSON object per line.
Append-only by construction: we never rewrite a line, only append. This makes
verdicts replayable and lets a user audit why the supervisor blocked (or nudged)
something.

The ledger is the source of truth for the deterministic detectors (they read
the recent lines of the *current* session) and for the human-facing ``shepherd
replay`` command.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .types import Agent, AgentEvent, Verdict


def default_root() -> Path:
    """Where the ledger lives by default (overridable via SHEPHERD_HOME)."""
    return Path(os.environ.get("SHEPHERD_HOME", str(Path.home() / ".shepherd")))


class Ledger:
    """Append-only JSONL session ledger."""

    def __init__(self, root: Path | str | None = None, agent: Agent | None = None):
        self.root = Path(root) if root else default_root()
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _session_dir(self, agent: Agent) -> Path:
        return self.root / "sessions" / agent.value

    def _session_path(self, agent: Agent, session_id: str) -> Path:
        return self._session_dir(agent) / f"{session_id}.jsonl"

    def _lock_for(self, key: str) -> threading.Lock:
        with self._global_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def append(self, agent: Agent, session_id: str, record: dict[str, Any]) -> None:
        """Append one JSON record to the session's ledger file."""
        path = self._session_path(agent, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record.setdefault("ts", time.time())
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        lock = self._lock_for(f"{agent.value}:{session_id}")
        with lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def record_event(self, event: AgentEvent) -> None:
        """Persist an observed event."""
        self.append(
            event.agent,
            event.session_id,
            {
                "type": "event",
                "event": event.event.value,
                "iteration": event.iteration,
                "tool": (
                    {
                        "name": event.tool.name,
                        "input": event.tool.input,
                        "tool_use_id": event.tool.tool_use_id,
                    }
                    if event.tool
                    else None
                ),
                "tool_result": event.tool_result,
                "reasoning": event.reasoning,
                "prompt": event.prompt,
                "metadata": event.metadata,
                "ts": event.ts,
            },
        )

    def record_verdict(
        self,
        agent: Agent,
        session_id: str,
        verdict: Verdict,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Persist a verdict (and the context that produced it)."""
        self.append(
            agent,
            session_id,
            {
                "type": "verdict",
                "action": verdict.action.value,
                "reason": verdict.reason,
                "guidance": verdict.guidance,
                "confidence": verdict.confidence,
                "detector": verdict.detector,
                "context": context or {},
            },
        )

    def recent(self, agent: Agent, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent `limit` ledger records for a session."""
        path = self._session_path(agent, session_id)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-limit:]

    def iter_records(self, agent: Agent, session_id: str) -> list[dict[str, Any]]:
        """All records for a session, oldest first (for replay)."""
        return list(reversed(self.recent(agent, session_id, limit=10**9)))