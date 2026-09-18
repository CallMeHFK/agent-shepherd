"""Install the QwenPaw supervisor plugin into ``~/.qwenpaw/plugins/``."""

from __future__ import annotations

import shutil
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
        shutil.rmtree(plugins_dir)
    shutil.copytree(bundle, plugins_dir)

    # Keep the daemon config in sync with the user's local model setup.
    cfg.ensure_defaults()
    print(f"installed QwenPaw supervisor plugin -> {plugins_dir}")
    print("restart QwenPaw to load the plugin (middleware + stop handler)")