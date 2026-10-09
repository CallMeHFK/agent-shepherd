"""Install the Claude Code hook config."""

from __future__ import annotations

import json
from pathlib import Path

from ..backup import load_json_or_raise, write_json


def install_claude(dry_run: bool = False) -> None:
    """Write the Claude Code hook config into ``~/.claude/settings.json``."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    hook = {
        "type": "command",
        "command": "shepherd-hook claude",
    }
    events = ["PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop", "UserPromptSubmit"]

    if dry_run:
        print(f"dry-run: would append hooks to {settings_path}")
        return

    existing = load_json_or_raise(settings_path)

    hooks = existing.setdefault("hooks", {})
    for event in events:
        # Drop any shepherd entry whose shape predates the current one:
        # idempotency by exact dict equality would leave both behind, and the
        # hook would fire twice per event.
        bucket = [e for e in hooks.get(event, []) if "shepherd-hook" not in json.dumps(e)]
        bucket.append({"matcher": "*", "hooks": [hook]})
        hooks[event] = bucket

    saved = write_json(settings_path, existing)
    print(f"installed Claude Code supervisor hooks -> {settings_path}")
    if saved:
        print(f"previous settings preserved at {saved}")