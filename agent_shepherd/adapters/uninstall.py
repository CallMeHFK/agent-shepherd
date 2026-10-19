"""Remove what ``shepherd install`` wrote — and nothing else.

The installers are deliberately conservative because they edit files the user
cares about more than this plugin (``~/.claude/settings.json``,
``~/.codex/hooks.json``, an installed QwenPaw plugin). Uninstall keeps that
posture in reverse:

* only entries this package wrote are removed — foreign hooks, settings and
  plugins survive untouched;
* nothing is deleted without a timestamped copy beside it;
* a file that carried nothing but ours is deleted rather than left as an
  empty husk, so "uninstalled" does not mean "a trace of me remains".
"""

from __future__ import annotations

import time
from pathlib import Path

from .backup import backup, load_json_or_raise, strip_hook_entries, write_json

# The command every hook entry this package writes contains; the installers
# already use it as their identity, and it is what an uninstall matches on.
_MARKER = "shepherd-hook"
_CODEX_DESCRIPTION = "Agent Shepherd supervisor hooks for Codex"
# Top-level keys the pre-0.1.1 Codex adapter wrote (invalid under the current
# schema). install_codex drops them; uninstall finishes the job.
_LEGACY_TOP_LEVEL = [
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
    "UserPromptSubmit",
]


def uninstall_claude(dry_run: bool = False) -> bool:
    """Strip shepherd's hook entries from ``~/.claude/settings.json``.

    Returns whether anything was removed, so a caller can tell "uninstalled"
    from "was never installed" without re-reading the file.
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    if dry_run:
        print(f"dry-run: would remove shepherd hooks from {settings_path}")
        return False
    if not settings_path.exists():
        print(f"nothing to remove: {settings_path} does not exist")
        return False

    existing = load_json_or_raise(settings_path)
    hooks = existing.get("hooks")
    if not isinstance(hooks, dict) or not strip_hook_entries(hooks, _MARKER):
        print(f"no shepherd hooks found in {settings_path} — nothing to do")
        return False
    if not hooks:
        existing.pop("hooks", None)
    return _finish(settings_path, existing, "removed the Claude Code supervisor hooks")


def uninstall_codex(dry_run: bool = False) -> bool:
    """Strip shepherd's hook groups from ``~/.codex/hooks.json``."""
    hooks_path = Path.home() / ".codex" / "hooks.json"
    if dry_run:
        print(f"dry-run: would remove shepherd hooks from {hooks_path}")
        return False
    if not hooks_path.exists():
        print(f"nothing to remove: {hooks_path} does not exist")
        return False

    existing = load_json_or_raise(hooks_path)
    events = existing.get("hooks")
    changed = strip_hook_entries(events, _MARKER) if isinstance(events, dict) else False
    for key in _LEGACY_TOP_LEVEL:
        if key in existing:
            del existing[key]
            changed = True
    if not changed:
        print(f"no shepherd hooks found in {hooks_path} — nothing to do")
        return False
    if isinstance(events, dict) and not events:
        existing.pop("hooks", None)
    if existing.get("description") == _CODEX_DESCRIPTION:
        # Ours only if it is still exactly what the installer writes; a
        # hand-edited description belongs to the user.
        existing.pop("description", None)
    return _finish(hooks_path, existing, "removed the Codex supervisor hooks")


def uninstall_qwenpaw(dry_run: bool = False) -> bool:
    """Move the plugin bundle out of QwenPaw's scan path — never delete it.

    An aside copy left in ``plugins/`` is not a backup, it is a second plugin:
    every copy carries the same manifest id. So the bundle goes to
    ``plugin-archive/`` under a ``.removed-`` stamp, matching where install
    already keeps previous versions.
    """
    plugins_dir = Path.home() / ".qwenpaw" / "plugins"
    archive_dir = Path.home() / ".qwenpaw" / "plugin-archive"
    if dry_run:
        print(f"dry-run: would move agent-shepherd* out of {plugins_dir} -> {archive_dir}")
        return False
    if not plugins_dir.exists():
        print(f"nothing to remove: {plugins_dir} does not exist")
        return False

    mine = sorted(p for p in plugins_dir.iterdir() if p.is_dir() and p.name.startswith("agent-shepherd"))
    if not mine:
        print(f"no agent-shepherd bundle in {plugins_dir} — nothing to do")
        return False
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    for bundle in mine:
        target = archive_dir / f"{bundle.name}.removed-{stamp}"
        bundle.rename(target)
        print(f"moved {bundle.name} out of the plugin scan path -> {target}")
    print("restart QwenPaw to unload the plugin")
    return True


def _finish(path: Path, data: dict, message: str) -> bool:
    """Write the stripped document back, or delete it if nothing is left."""
    if not data:
        saved = backup(path)
        path.unlink()
        print(f"{message}; {path} held nothing else and was deleted")
        if saved:
            print(f"previous file preserved at {saved}")
        return True
    saved = write_json(path, data)
    print(f"{message} -> {path}")
    if saved:
        print(f"previous file preserved at {saved}")
    return True
