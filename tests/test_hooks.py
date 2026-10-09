"""Tests for the helpers both hook thin-shells share."""

from __future__ import annotations

from agent_shepherd.adapters.daemon import daemon_url, flatten_result


def test_flatten_result_lifts_structured_evidence():
    text, extra = flatten_result({"stdout": "hi", "exit_code": 0, "interrupted": False})
    assert text == "hi"
    assert extra == {"exit_code": 0, "interrupted": False}


def test_flatten_result_falls_back_to_json_for_opaque_dicts():
    text, extra = flatten_result({"unexpected": 1})
    assert "unexpected" in text
    assert extra == {}


def test_flatten_result_passes_plain_strings_through():
    assert flatten_result("plain") == ("plain", {})


def test_daemon_url_defaults_to_loopback_and_strips_trailing_slash(monkeypatch):
    monkeypatch.delenv("SHEPHERD_DAEMON_URL", raising=False)
    assert daemon_url() == "http://127.0.0.1:4890"
    monkeypatch.setenv("SHEPHERD_DAEMON_URL", "http://127.0.0.1:5090/")
    assert daemon_url() == "http://127.0.0.1:5090"
