"""Tests for the adapter installers' safety around the user's own config."""

from __future__ import annotations

import json

import pytest

from agent_shepherd.adapters.backup import UnparseableConfig, load_json_or_raise, write_json
from agent_shepherd.adapters.claude.install import install_claude
from agent_shepherd.adapters.codex.install import install_codex
from agent_shepherd.core.config import ShepherdConfig


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the installers at a temp HOME so real settings are never touched."""
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".codex").mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return tmp_path


def test_write_json_backs_up_the_previous_file(home):
    target = home / ".claude" / "settings.json"
    target.write_text(json.dumps({"model": "keep-me"}), encoding="utf-8")
    saved = write_json(target, {"model": "keep-me", "hooks": {}})
    assert saved is not None and saved.exists()
    assert json.loads(saved.read_text())["model"] == "keep-me"
    assert json.loads(target.read_text())["hooks"] == {}


def test_unparseable_config_is_refused_not_replaced(home):
    broken = home / ".claude" / "settings.json"
    broken.write_text("{ model: oops", encoding="utf-8")
    with pytest.raises(UnparseableConfig):
        load_json_or_raise(broken)
    # The bytes nobody parsed are still there.
    assert broken.read_text() == "{ model: oops"


def test_claude_install_preserves_unrelated_settings(home):
    settings = home / ".claude" / "settings.json"
    settings.write_text(json.dumps({"theme": "dark", "hooks": {}}), encoding="utf-8")
    install_claude(ShepherdConfig.load(home / "missing.yaml"), dry_run=False)
    after = json.loads(settings.read_text())
    assert after["theme"] == "dark", "the user's own keys survive"
    assert "PreToolUse" in after["hooks"] and "UserPromptSubmit" in after["hooks"]


def test_claude_install_is_idempotent(home):
    settings = home / ".claude" / "settings.json"
    cfg = ShepherdConfig.load(home / "missing.yaml")
    install_claude(cfg, dry_run=False)
    first = json.loads(settings.read_text())
    install_claude(cfg, dry_run=False)
    assert json.loads(settings.read_text()) == first, "re-running does not duplicate hook groups"


def test_codex_install_keeps_existing_groups(home):
    hooks = home / ".codex" / "hooks.json"
    mine = {"type": "command", "command": "other-tool", "timeout": 5}
    hooks.write_text(
        json.dumps({"description": "user", "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [mine]}]}}),
        encoding="utf-8",
    )
    install_codex(ShepherdConfig.load(home / "missing.yaml"), dry_run=False)
    after = json.loads(hooks.read_text())
    group = after["hooks"]["PreToolUse"][0]
    assert group["matcher"] == "Bash" and group["hooks"] == [mine], "foreign group untouched"
    assert any(g.get("matcher") == "*" for g in after["hooks"]["PreToolUse"][1:]), "ours appended"
    assert after["description"] == "user"


def test_qwenpaw_install_keeps_one_bundle_in_the_plugin_scan_path(home):
    """Regression: the installer used to move the previous bundle to
    ``plugins/agent-shepherd.prev-<stamp>``, i.e. still inside the directory
    QwenPaw scans. Every copy carries the same manifest id, so the stale bundle
    was loaded as a second plugin and could win — the agent ran last release's
    middleware while every check pointed at the freshly written path."""
    from agent_shepherd.adapters.qwenpaw.install import install_qwenpaw

    plugins = home / ".qwenpaw" / "plugins"
    (plugins / "agent-shepherd").mkdir(parents=True)
    (plugins / "agent-shepherd" / "backend").mkdir()
    (plugins / "agent-shepherd" / "backend" / "main.py").write_text("old\n", encoding="utf-8")
    (plugins / "agent-shepherd.prev-20260921-091308").mkdir()
    (plugins / "agent-shepherd.prev-20260921-091308" / "backend").mkdir()
    (plugins / "agent-shepherd.prev-20260921-091308" / "backend" / "main.py").write_text("older\n", encoding="utf-8")
    (plugins / "cad").mkdir()
    (plugins / "cad" / "keep.txt").write_text("untouched\n", encoding="utf-8")

    install_qwenpaw(ShepherdConfig.load(home / "missing.yaml"), dry_run=False)

    copies = sorted(p.name for p in plugins.iterdir() if p.name.startswith("agent-shepherd"))
    assert copies == ["agent-shepherd"], "no duplicate-id bundle left in the scan path"
    assert (plugins / "cad" / "keep.txt").read_text() == "untouched\n", "foreign plugins untouched"
    assert "trust_env=False" in (plugins / "agent-shepherd" / "backend" / "main.py").read_text()
    archived = sorted(p.name for p in (home / ".qwenpaw" / "plugin-archive").iterdir())
    assert "agent-shepherd.prev-20260921-091308" in archived, "the old copy is kept, just out of the scan path"


def test_hook_survives_a_socks_proxy_environment(monkeypatch):
    """Regression: with ALL_PROXY=socks5:// every hook call raised ImportError
    (httpx needs an optional extra for SOCKS), so the installed hook crashed on
    every tool call instead of failing open. Daemon calls are loopback and must
    not inherit proxy environment."""
    from agent_shepherd.adapters.claude import hook as chook

    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")
    monkeypatch.setenv("HTTPS_PROXY", "socks5://127.0.0.1:1080")
    monkeypatch.setenv("SHEPHERD_DAEMON_URL", "http://127.0.0.1:1")
    # Pre-fix this raised ImportError out of httpx.Client(...) instead of
    # returning None, which surfaced as a traceback in the agent's hook output.
    assert chook._post("/ingest/claude", {"a": 1}) is None


def test_hook_never_raises_when_the_daemon_is_down(monkeypatch):
    """A supervisor the agent cannot reach must be invisible, not fatal."""
    from agent_shepherd.adapters.claude import hook as chook

    monkeypatch.setenv("SHEPHERD_DAEMON_URL", "http://127.0.0.1:1")  # nothing listening
    assert chook._post("/ingest/claude", {"a": 1}) is None
    assert chook._response(None) == {}
