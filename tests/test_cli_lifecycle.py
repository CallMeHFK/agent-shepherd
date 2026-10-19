"""The lifecycle commands a user actually types: install, stop, uninstall.

Everything here runs against a temp HOME and a dead daemon URL, so no test can
touch a real ``~/.claude``, a real ``~/.shepherd``, or a real running daemon.
"""

from __future__ import annotations

import os

import pytest

from agent_shepherd.cli import main
from agent_shepherd.core.server import _managed_by


@pytest.mark.parametrize(
    "cgroup,expected",
    [
        # The shipped unit: systemd owns this process and will restart it.
        ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/shepherd.service\n", "systemd"),
        # Started from a shell (or start-bg): the shell's scope, nobody restarts it.
        ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-t3code.scope\n", "manual"),
        ("", "manual"),
        ("0::/\n", "manual"),
    ],
)
def test_managed_by_reads_the_cgroup_leaf(cgroup, expected):
    """INVOCATION_ID would mislabel an ordinary `shepherd start` as unit-managed
    when the shell itself already runs under systemd; the cgroup leaf does not."""
    assert _managed_by(cgroup) == expected


@pytest.fixture
def home(tmp_path, monkeypatch):
    for dot in (".claude", ".codex", ".qwenpaw"):
        (tmp_path / dot).mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    # Nothing is ever listening here, so every daemon probe fails fast instead
    # of reaching a daemon the developer happens to have running.
    monkeypatch.setenv("SHEPHERD_DAEMON_URL", "http://127.0.0.1:1")
    return tmp_path


def test_stop_without_a_pidfile_reports_already_stopped(home, capsys):
    """A daemon started any way but start-bg used to be unstoppable through the
    CLI; "no pidfile" is now a question for /health, not the end of the road."""
    assert main(["stop"]) == 0
    assert "no daemon running" in capsys.readouterr().out


def test_stop_follows_the_health_payload_to_a_manual_daemon(home, monkeypatch):
    killed: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))

    class _Resp:
        def json(self):
            return {"ok": True, "pid": 4242, "managed_by": "manual"}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())

    assert main(["stop"]) == 0
    assert killed == [4242]


def test_stop_defers_to_systemd_when_the_unit_owns_the_daemon(home, monkeypatch):
    """The shipped unit has Restart=always: a bare SIGTERM comes back two
    seconds later with the user believing supervision is off."""
    import subprocess

    class _Resp:
        def json(self):
            return {"ok": True, "pid": 4242, "managed_by": "systemd"}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    def _no_kill(*args):
        raise AssertionError("a systemd-owned daemon must not be signalled directly")

    monkeypatch.setattr(os, "kill", _no_kill)

    assert main(["stop"]) == 0
    assert calls == [["systemctl", "--user", "stop", "shepherd"]]


def test_stop_falls_through_a_stale_pidfile_to_the_health_payload(home, monkeypatch):
    """The unit outlives the start-bg process it replaced, so a dead pidfile is
    not proof that nothing is supervising."""
    import subprocess

    from agent_shepherd.core.ledger import default_root

    root = default_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "daemon.pid").write_text("999999", encoding="utf-8")

    class _Resp:
        def json(self):
            return {"ok": True, "pid": 4242, "managed_by": "systemd"}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert main(["stop"]) == 0
    assert not (root / "daemon.pid").exists(), "the stale pidfile is cleaned up either way"
    assert calls == [["systemctl", "--user", "stop", "shepherd"]]


def test_install_warns_while_the_daemon_is_down(home, capsys):
    """Hooks with no daemon behind them fail open and look like "nothing
    happens"; the install says so instead of leaving the user to guess."""
    assert main(["install", "claude"]) == 0
    assert "daemon DOWN" in capsys.readouterr().err


def test_uninstall_all_purges_every_trace(home, capsys):
    from agent_shepherd.adapters.claude.install import install_claude
    from agent_shepherd.adapters.codex.install import install_codex
    from agent_shepherd.core.config import ShepherdConfig

    install_claude(dry_run=False)
    install_codex(dry_run=False)
    (home / ".qwenpaw" / "plugins" / "agent-shepherd").mkdir(parents=True)
    (home / ".qwenpaw" / "plugins" / "agent-shepherd" / "plugin.json").write_text("{}", encoding="utf-8")
    ShepherdConfig.load(home / "missing.yaml").ensure_defaults()
    (home / ".shepherd" / "sessions").mkdir(parents=True)
    (home / ".shepherd" / "sessions" / "ledger.jsonl").write_text("{}\n", encoding="utf-8")

    assert main(["uninstall", "all", "--purge"]) == 0

    assert not (home / ".claude" / "settings.json").exists()
    assert not (home / ".codex" / "hooks.json").exists()
    assert not (home / ".qwenpaw" / "plugins" / "agent-shepherd").exists()
    assert not (home / ".shepherd").exists(), "purge removes the state root, not just the hooks"
    assert "purged" in capsys.readouterr().out


def test_uninstall_without_purge_keeps_the_state_root(home, capsys):
    from agent_shepherd.adapters.claude.install import install_claude
    from agent_shepherd.core.config import ShepherdConfig

    install_claude(dry_run=False)
    ShepherdConfig.load(home / "missing.yaml").ensure_defaults()

    assert main(["uninstall", "claude"]) == 0

    assert not (home / ".claude" / "settings.json").exists()
    assert (home / ".shepherd" / "config.yaml").exists(), "the ledger and config survive an adapter uninstall"
    assert "re-run with --purge" in capsys.readouterr().out


def test_uninstall_one_agent_leaves_the_daemon_for_the_others(home, capsys):
    from agent_shepherd.adapters.claude.install import install_claude
    from agent_shepherd.adapters.codex.install import install_codex

    install_claude(dry_run=False)
    install_codex(dry_run=False)

    assert main(["uninstall", "claude"]) == 0

    assert (home / ".codex" / "hooks.json").exists()
    assert "daemon left running" in capsys.readouterr().out


def test_uninstall_dry_run_changes_nothing(home, capsys):
    from agent_shepherd.adapters.claude.install import install_claude

    install_claude(dry_run=False)
    before = (home / ".claude" / "settings.json").read_text()

    assert main(["uninstall", "all", "--dry-run"]) == 0

    assert (home / ".claude" / "settings.json").read_text() == before
    assert "nothing was changed" in capsys.readouterr().out


def test_uninstall_refuses_an_unparseable_config_without_touching_it(home, capsys):
    settings = home / ".claude" / "settings.json"
    settings.write_text("{ model: oops", encoding="utf-8")

    assert main(["uninstall", "all"]) == 2

    assert settings.read_text() == "{ model: oops"
    assert "refusing to edit" in capsys.readouterr().err
