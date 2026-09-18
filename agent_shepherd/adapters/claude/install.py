"""Install the Claude Code hook config."""

from __future__ import annotations

import json
from pathlib import Path

from ...core.config import ShepherdConfig


def install_claude(cfg: ShepherdConfig, dry_run: bool = False) -> None:
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

    existing = {}
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    hooks = existing.setdefault("hooks", {})
    for event in events:
        hooks[event] = hooks.get(event, [])
        entry = {"matcher": "*", "hooks": [hook]}
        if entry not in hooks[event]:
            hooks[event].append(entry)

    settings_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"installed Claude Code supervisor hooks -> {settings_path}")