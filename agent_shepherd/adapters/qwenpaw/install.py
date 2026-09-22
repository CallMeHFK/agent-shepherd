"""Install the QwenPaw supervisor plugin into ``~/.qwenpaw/plugins/``."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ...core.config import ShepherdConfig


def install_qwenpaw(cfg: ShepherdConfig, dry_run: bool = False) -> None:
    """Copy the plugin bundle into QwenPaw's plugin directory."""
    qwenpaw_root = Path.home() / ".qwenpaw"
    plugins_dir = qwenpaw_root / "plugins" / "agent-shepherd"
    archive_dir = qwenpaw_root / "plugin-archive"
    bundle = Path(__file__).resolve().parent

    if dry_run:
        print(f"dry-run: would install {bundle} -> {plugins_dir}")
        return

    plugins_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # QwenPaw loads every subdirectory of plugins/, and each copy carries the
    # same manifest id. An aside copy left in there is therefore not a backup,
    # it is a second plugin: the stale bundle can win the race and the agent
    # runs last release's middleware while `doctor` points at the new path.
    stale = sorted(p for p in plugins_dir.parent.iterdir() if p.is_dir() and _is_prev_copy(p.name))
    for aside in stale:
        target = archive_dir / aside.name
        aside.rename(target)
        print(f"stale duplicate bundle moved out of the plugin scan path -> {target}")

    if plugins_dir.exists():
        # Move the previous bundle aside rather than deleting it: an older plugin
        # version may be the only copy the user has of a local edit.
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aside = archive_dir / f"agent-shepherd.prev-{stamp}"
        plugins_dir.rename(aside)
        print(f"previous plugin bundle archived to {aside}")
    shutil.copytree(bundle, plugins_dir, ignore=shutil.ignore_patterns("__pycache__"))

    # Keep the daemon config in sync with the user's local model setup.
    cfg.ensure_defaults()
    print(f"installed QwenPaw supervisor plugin -> {plugins_dir}")
    print("restart QwenPaw to load the plugin (middleware + stop handler)")


def _is_prev_copy(name: str) -> bool:
    return name.startswith("agent-shepherd.prev-")
