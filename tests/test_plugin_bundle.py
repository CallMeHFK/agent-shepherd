"""Tests for the release plugin bundle: the ZIP a stranger installs must be
what ``shepherd install qwenpaw`` writes, in the shape QwenPaw's installers
accept."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The directory is named `packaging`, which the PyPI `packaging` distribution
# already owns on sys.path, so it is loaded by file rather than imported.
_spec = importlib.util.spec_from_file_location("build_plugin_zip", ROOT / "packaging" / "build_plugin_zip.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the installer at a temp HOME so a real ~/.qwenpaw is never touched."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return tmp_path


def _tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def test_the_zip_is_one_top_level_directory_holding_the_manifest(tmp_path):
    """QwenPaw's in-app installer takes plugin.json from the archive root or one
    top-level directory; its offline CLI installer refuses an archive with more
    than one. A single directory named after the manifest id satisfies both, and
    matches where the host puts the bundle on disk."""
    archive = builder.build(tmp_path / "dist")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert archive.name == "agent-shepherd.zip", "named after the plugin, not the release"
    tops = {name.split("/")[0] for name in names}
    assert tops == {"agent-shepherd"}, f"expected exactly one top-level directory, got {tops}"
    assert "agent-shepherd/plugin.json" in names
    assert "agent-shepherd/backend/main.py" in names, "the manifest's entry.backend must be in the archive"
    assert "agent-shepherd/README.md" in names, "whoever imports the zip inherits its prerequisites with it"
    assert not any(n.endswith(".pyc") or "__pycache__" in n for n in names), "tool caches are not payload"


def test_the_zip_carries_the_same_tree_the_cli_installer_writes(tmp_path, home):
    """Two install paths, one bundle: a release-asset install that quietly ships
    a different file set than `shepherd install qwenpaw` is indistinguishable at
    runtime until the two diverge in behaviour."""
    from agent_shepherd.adapters.qwenpaw.install import install_qwenpaw
    from agent_shepherd.core.config import ShepherdConfig

    archive = builder.build(tmp_path / "dist")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp_path / "extracted")
    zipped = _tree(tmp_path / "extracted" / "agent-shepherd")

    install_qwenpaw(ShepherdConfig.load(home / "missing.yaml"), dry_run=False)
    installed = _tree(home / ".qwenpaw" / "plugins" / "agent-shepherd")

    assert zipped == installed


def test_rebuilding_the_zip_from_the_same_commit_is_byte_identical(tmp_path):
    """The asset is stamped, not dated, so a re-upload of a release can be
    compared against the one already attached instead of differing by mtimes."""
    first = builder.build(tmp_path / "a").read_bytes()
    second = builder.build(tmp_path / "b").read_bytes()
    assert first == second
