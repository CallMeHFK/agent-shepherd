"""Install the Codex hook config.

Codex (codex-cli >= 0.150) expects ``~/.codex/hooks.json`` in a specific
shape (see ``codex-rs/config/src/hook_config.rs``):

* the top level has only ``description`` and ``hooks`` (deny_unknown_fields);
* each event maps to a list of *MatcherGroups* ``{matcher?, hooks: [...]}``;
* each handler is tagged ``{"type": "command", "command": ..., "timeout": N}``.

Only these events exist: PreToolUse, PostToolUse, Stop, UserPromptSubmit,
SessionStart, SessionEnd, PreCompact, PostCompact, PermissionRequest,
SubagentStart, SubagentStop, Interrupt. (There is no ``PostToolUseFailure``
or ``Notification``.)
"""

from __future__ import annotations

from pathlib import Path

from ...core.config import ShepherdConfig
from ..backup import load_json_or_raise, write_json

# Events this adapter subscribes to. ``matcher`` is honored for tool events but
# ignored for Stop/UserPromptSubmit, so it is omitted there.
_EVENTS = ["PreToolUse", "PostToolUse", "Stop", "UserPromptSubmit"]
_MATCHED = {"PreToolUse", "PostToolUse"}
# Legacy top-level keys written by the pre-0.1.1 adapter (invalid schema) that
# we clean up so a stale file cannot keep failing to parse.
_LEGACY_TOP_LEVEL = _EVENTS + ["PostToolUseFailure"]


def install_codex(cfg: ShepherdConfig, dry_run: bool = False) -> None:
    """Write the Codex hook config into ``~/.codex/hooks.json``."""
    hooks_path = Path.home() / ".codex" / "hooks.json"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"dry-run: would write hooks to {hooks_path}")
        return

    existing: dict = load_json_or_raise(hooks_path)

    # Remove any legacy top-level event keys (old broken format) so the file
    # parses under the current schema.
    for key in _LEGACY_TOP_LEVEL:
        existing.pop(key, None)

    existing.setdefault("description", "Agent Shepherd supervisor hooks for Codex")
    events = existing.setdefault("hooks", {})

    command_hook = {"type": "command", "command": "shepherd-hook codex", "timeout": 5}
    for event in _EVENTS:
        group = (
            {"matcher": "*", "hooks": [command_hook]}
            if event in _MATCHED
            else {"hooks": [command_hook]}
        )
        bucket = events.setdefault(event, [])
        if group not in bucket:
            bucket.append(group)

    saved = write_json(hooks_path, existing)
    print(f"installed Codex supervisor hooks -> {hooks_path}")
    if saved:
        print(f"previous hooks file preserved at {saved}")
    print(
        "note: Codex runs hooks only after they are *trusted*. Approve the hooks once\n"
        "      in the interactive Codex TUI, or pass --dangerously-bypass-hook-trust\n"
        "      to `codex exec` for automation (untrusted hooks are silently skipped)."
    )
