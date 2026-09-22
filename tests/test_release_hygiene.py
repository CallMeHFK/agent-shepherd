"""Tests for release hygiene: the version is one fact, not three copies."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)


def _plugin_manifest_version() -> str:
    manifest = ROOT / "agent_shepherd" / "adapters" / "qwenpaw" / "plugin.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["version"]


def test_packaging_and_plugin_manifest_agree_on_the_version():
    """The QwenPaw manifest is copied into ~/.qwenpaw/plugins and is the only
    version the host displays, so a manifest left behind advertises an old
    plugin from new code -- which is exactly what makes a stale bundle
    indistinguishable from a current one."""
    from agent_shepherd import __version__

    assert _pyproject_version() == __version__
    assert _plugin_manifest_version() == __version__
