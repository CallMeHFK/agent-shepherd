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
    bundle = Path(__file__).resolve().parent

    if dry_run:
        print(f"dry-run: would install {bundle} -> {plugins_dir}")
        return

    plugins_dir.parent.mkdir(parents=True, exist_ok=True)
    if plugins_dir.exists():
        # Move the previous bundle aside rather than deleting it: an older plugin
        # version may be the only copy the user has of a local edit.
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aside = plugins_dir.with_name(f"agent-shepherd.prev-{stamp}")
        plugins_dir.rename(aside)
        print(f"previous plugin bundle moved to {aside}")
    shutil.copytree(bundle, plugins_dir, ignore=shutil.ignore_patterns("__pycache__"))

    # Keep the daemon config in sync with the user's local model setup.
    cfg.ensure_defaults()
    print(f"installed QwenPaw supervisor plugin -> {plugins_dir}")
    print("restart QwenPaw to load the plugin (middleware + stop handler)")