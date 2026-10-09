"""Shared helpers for the hook thin-shells (Claude Code, Codex).

Both shells do the same three things: find the daemon, POST a normalized event
to it without ever raising into the host harness, and split a harness tool
response into display text plus structured evidence. The QwenPaw plugin keeps
its own copies on purpose: the plugin bundle must stay self-contained (see
``packaging/build_plugin_zip.py``).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


def daemon_url() -> str:
    return os.environ.get("SHEPHERD_DAEMON_URL", "http://127.0.0.1:4890").rstrip("/")


def post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    # trust_env=False: the daemon is a loopback service, so honoring
    # HTTP(S)_PROXY / ALL_PROXY here would both fail (a SOCKS proxy needs an
    # optional extra) and send agent telemetry through an unrelated proxy.
    try:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            resp = client.post(f"{daemon_url()}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001
        # A supervisor that raises on every tool call when its daemon is
        # unreachable is worse than no supervisor: the host harness treats a
        # crashing hook as an error on the action, not as a silent pass.
        return None


def flatten_result(raw: Any) -> tuple[Any, dict[str, Any]]:
    """Split a harness tool response into display text + structured evidence.

    Claude Code and Codex hand PostToolUse a ``tool_response`` that is often a
    *dict* (Bash gives ``{stdout, stderr, exit_code, interrupted}``). Keeping
    the exit code and the error flags as structured metadata is what lets the
    daemon tell a real failure from output that merely contains the word
    "error" — flattening everything into a string here would throw that
    evidence away.
    """
    extra: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in ("exit_code", "returncode", "is_error", "interrupted", "success"):
            if key in raw:
                extra[key] = raw[key]
        parts = [str(raw[k]) for k in ("stdout", "stderr", "content", "output") if raw.get(k) is not None]
        text = "\n".join(parts) if parts else json.dumps(raw, ensure_ascii=False)[:4000]
        return text, extra
    return raw, extra
