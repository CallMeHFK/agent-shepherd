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


def test_the_releases_named_in_the_docs_are_the_current_one():
    """An install URL has the version baked into its filename (`.../download/vX/`
    and `agent_shepherd-X-py3-none-any.whl`), so a bump that forgets the docs
    sends readers to the *previous* release's artifact -- which still downloads
    and still installs, just not the code the page claims."""
    from agent_shepherd import __version__

    docs = [ROOT / "README.md", ROOT / "agent_shepherd" / "adapters" / "qwenpaw" / "README.md"]
    pattern = re.compile(r"(?:@v|/download/v|agent_shepherd-)(\d+\.\d+\.\d+)")
    named = {(doc.relative_to(ROOT).as_posix(), v) for doc in docs for v in pattern.findall(doc.read_text(encoding="utf-8"))}
    stale = sorted(item for item in named if item[1] != __version__)
    assert not stale, f"docs name a release other than {__version__}: {stale}"
