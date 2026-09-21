"""Safety wrappers for writing into another tool's configuration.

These installers modify files the user cares about more than they care about
this plugin (``~/.claude/settings.json``, ``~/.codex/hooks.json``, an installed
QwenPaw plugin). Two rules follow from that, and both are cheap:

* keep a timestamped copy of whatever was there before changing it, and
* refuse to overwrite a file we could not parse — a JSON syntax error in the
  existing file means the write would replace hand-edited configuration with
  ours and the user would only find out later.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


class UnparseableConfig(Exception):
    """The existing file is not valid JSON; a write would destroy it."""


def backup(path: Path) -> Path | None:
    """Copy ``path`` aside with a timestamp suffix. Returns the backup path."""
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    target = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 0
    while target.exists():
        counter += 1
        target = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
    shutil.copy2(path, target)
    return target


def load_json_or_raise(path: Path) -> dict:
    """Read a JSON object config file, or refuse to proceed.

    A missing file is an empty config. A file that exists but does not parse is
    an error the caller must surface, never a licence to start over.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UnparseableConfig(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise UnparseableConfig(f"{path}: expected a JSON object, found {type(data).__name__}")
    return data


def write_json(path: Path, data: dict) -> Path | None:
    """Back up, then atomically replace ``path`` with ``data`` as JSON."""
    saved = backup(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return saved
