"""Uninstall must remove exactly what install wrote, and nothing more.

The posture mirrors the installers: foreign configuration survives, nothing is
deleted without a timestamped copy beside it, and a file that carried only ours
is removed instead of left behind as an empty husk.
"""

from __future__ import annotations

import json

import pytest

from agent_shepherd.adapters.claude.install import install_claude
from agent_shepherd.adapters.codex.install import install_codex
from agent_shepherd.adapters.uninstall import uninstall_claude, uninstall_codex, uninstall_qwenpaw


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point every installer/uninstaller at a temp HOME."""
    for dot in (".claude", ".codex", ".qwenpaw"):
        (tmp_path / dot).mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return tmp_path


def test_claude_uninstall_removes_only_shepherd_entries(home):
    settings = home / ".claude" / "settings.json"
    foreign = {"matcher": "Write", "hooks": [{"type": "command", "command": "other-tool"}]}
    settings.write_text(json.dumps({"theme": "dark", "hooks": {"PreToolUse": [foreign]}}), encoding="utf-8")
    install_claude(dry_run=False)
    assert len(json.loads(settings.read_text())["hooks"]) == 5

    assert uninstall_claude(dry_run=False) is True

    after = json.loads(settings.read_text())
    assert after["theme"] == "dark", "the user's own keys survive"
    assert after["hooks"] == {"PreToolUse": [foreign]}, "ours gone, the foreign group kept"
    assert uninstall_claude(dry_run=False) is False, "a second run finds nothing to do"


def test_claude_uninstall_deletes_a_file_that_held_only_ours(home):
    settings = home / ".claude" / "settings.json"
    install_claude(dry_run=False)
    assert settings.exists()

    assert uninstall_claude(dry_run=False) is True

    assert not settings.exists(), "an empty settings.json is a husk, not a keepsake"
    backups = [p.name for p in (home / ".claude").iterdir() if p.name.startswith("settings.json.bak-")]
    assert backups, "the pre-edit file is preserved beside it"


def test_claude_uninstall_without_a_file_is_a_no_op(home):
    assert uninstall_claude(dry_run=False) is False


def test_claude_uninstall_dry_run_touches_nothing(home):
    settings = home / ".claude" / "settings.json"
    install_claude(dry_run=False)
    before = settings.read_text()

    assert uninstall_claude(dry_run=True) is False

    assert settings.read_text() == before


def test_uninstall_refuses_an_unparseable_config(home):
    """Same rule as install: a file nobody parsed is never rewritten."""
    from agent_shepherd.adapters.backup import UnparseableConfig

    settings = home / ".claude" / "settings.json"
    settings.write_text("{ model: oops", encoding="utf-8")
    with pytest.raises(UnparseableConfig):
        uninstall_claude(dry_run=False)
    assert settings.read_text() == "{ model: oops", "the bytes nobody parsed are still there"


def test_codex_uninstall_keeps_foreign_groups_and_drops_ours(home):
    hooks = home / ".codex" / "hooks.json"
    mine = {"type": "command", "command": "other-tool", "timeout": 5}
    hooks.write_text(
        json.dumps({"description": "user", "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [mine]}]}}),
        encoding="utf-8",
    )
    install_codex(dry_run=False)

    assert uninstall_codex(dry_run=False) is True

    after = json.loads(hooks.read_text())
    assert after["description"] == "user", "a hand-edited description is the user's"
    assert after["hooks"] == {"PreToolUse": [{"matcher": "Bash", "hooks": [mine]}]}
    assert set(after["hooks"]) == {"PreToolUse"}, "our other events are pruned, not left empty"


def test_codex_uninstall_deletes_a_file_that_held_only_ours(home):
    hooks = home / ".codex" / "hooks.json"
    install_codex(dry_run=False)

    assert uninstall_codex(dry_run=False) is True

    assert not hooks.exists(), "hooks.json existed only to carry ours"
    assert any(p.name.startswith("hooks.json.bak-") for p in (home / ".codex").iterdir())


def test_codex_uninstall_drops_legacy_top_level_keys(home):
    """The pre-0.1.1 shape wrote event keys at the top level, which Codex's
    schema rejects; uninstall finishes what install started."""
    hooks = home / ".codex" / "hooks.json"
    hooks.write_text(
        json.dumps({"PreToolUse": [{"matcher": "*", "hooks": [{"command": "shepherd-hook codex"}]}]}),
        encoding="utf-8",
    )

    assert uninstall_codex(dry_run=False) is True

    assert not hooks.exists(), "the legacy shape was ours and invalid; nothing else was in the file"


def test_qwenpaw_uninstall_moves_the_bundle_out_of_the_scan_path(home):
    """A copy left in plugins/ is not a backup, it is a second plugin: every
    copy carries the same manifest id. It moves to the archive instead."""
    plugins = home / ".qwenpaw" / "plugins"
    (plugins / "agent-shepherd" / "backend").mkdir(parents=True)
    (plugins / "agent-shepherd" / "backend" / "main.py").write_text("live\n", encoding="utf-8")
    (plugins / "agent-shepherd.prev-20260921-091308").mkdir()
    (plugins / "cad").mkdir()
    (plugins / "cad" / "keep.txt").write_text("untouched\n", encoding="utf-8")

    assert uninstall_qwenpaw(dry_run=False) is True

    assert sorted(p.name for p in plugins.iterdir()) == ["cad"], "no shepherd-named dir left in the scan path"
    archive = home / ".qwenpaw" / "plugin-archive"
    archived = sorted(p.name for p in archive.iterdir())
    assert any(n.startswith("agent-shepherd.removed-") for n in archived)
    assert any(n.startswith("agent-shepherd.prev-20260921-091308.removed-") for n in archived)
    moved = archive / next(n for n in archived if n.startswith("agent-shepherd.removed-"))
    assert (moved / "backend" / "main.py").read_text() == "live\n", "moved, not deleted"
    assert (plugins / "cad" / "keep.txt").read_text() == "untouched\n", "foreign plugins untouched"


def test_qwenpaw_uninstall_without_a_bundle_is_a_no_op(home):
    plugins = home / ".qwenpaw" / "plugins"
    plugins.mkdir()
    assert uninstall_qwenpaw(dry_run=False) is False


def test_qwenpaw_uninstall_dry_run_touches_nothing(home):
    plugins = home / ".qwenpaw" / "plugins"
    (plugins / "agent-shepherd").mkdir(parents=True)

    assert uninstall_qwenpaw(dry_run=True) is False

    assert (plugins / "agent-shepherd").exists()
