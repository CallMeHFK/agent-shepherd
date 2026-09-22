"""Tests for the config migration surface (`config merge` / `config prune`)."""

from __future__ import annotations

from agent_shepherd.core import config as cfgmod


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_prune_drops_retired_keys_and_keeps_live_ones(tmp_path):
    """A key the release no longer reads still looks like a live control in the
    file, so someone tunes it for nothing. `merge` only adds; this is the other
    half."""
    path = _write(
        tmp_path,
        "policy:\n  wake_every_n_steps: true\n  nudge_threshold: 0.55\n"
        "judge:\n  model: kept-me\n  retired_thing: 1\n",
    )
    assert cfgmod.stale_keys(path) == ["judge.retired_thing", "policy.wake_every_n_steps"]

    removed, written = cfgmod.prune_stale_keys(path)
    assert written == path
    assert removed == ["policy.wake_every_n_steps", "judge.retired_thing"]
    assert cfgmod.stale_keys(path) == []

    import yaml

    after = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert after["policy"] == {"nudge_threshold": 0.55}, "live values untouched"
    assert after["judge"] == {"model": "kept-me"}


def test_prune_backs_up_before_writing(tmp_path):
    original = "policy:\n  retired_thing: 1\n"
    path = _write(tmp_path, original)
    _, written = cfgmod.prune_stale_keys(path)
    backups = sorted(p for p in tmp_path.iterdir() if p.name.startswith("config.yaml.bak-"))
    assert written and backups, "the previous file is preserved"
    assert backups[-1].read_text(encoding="utf-8") == original


def test_prune_is_a_no_op_without_a_config_file(tmp_path):
    assert cfgmod.prune_stale_keys(tmp_path / "absent.yaml") == ([], None)


def test_merge_and_prune_together_converge_on_the_current_schema(tmp_path):
    """An old file needs both directions: retirements removed, new controls
    added. Running the pair twice must be stable."""
    path = _write(tmp_path, "policy:\n  retired_thing: 1\n  nudge_threshold: 0.55\n")
    for _ in range(2):
        cfgmod.prune_stale_keys(path)
        added, _ = cfgmod.merge_missing_keys(path)
        assert cfgmod.stale_keys(path) == []
    assert added == [], "second pass adds nothing"

    import yaml

    assert yaml.safe_load(path.read_text(encoding="utf-8"))["policy"]["nudge_threshold"] == 0.55
