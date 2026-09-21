"""Install the Claude Code hook config."""

from __future__ import annotations

from pathlib import Path

from ...core.config import ShepherdConfig
from ..backup import load_json_or_raise, write_json


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

    existing = load_json_or_raise(settings_path)

    hooks = existing.setdefault("hooks", {})
    for event in events:
        hooks[event] = hooks.get(event, [])
        entry = {"matcher": "*", "hooks": [hook]}
        if entry not in hooks[event]:
            hooks[event].append(entry)

    saved = write_json(settings_path, existing)
    print(f"installed Claude Code supervisor hooks -> {settings_path}")
    if saved:
        print(f"previous settings preserved at {saved}")