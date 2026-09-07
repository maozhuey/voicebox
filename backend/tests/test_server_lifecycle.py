"""Regression coverage for durable server lifecycle classification."""

import json

from backend import config
from backend.services import server_lifecycle


def test_running_marker_is_classified_as_unexpected_on_next_start(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    monkeypatch.setattr(server_lifecycle, "_current_instance_id", None)

    first = server_lifecycle.start_server_lifecycle()
    assert first.previous_exit == "normal"

    monkeypatch.setattr(server_lifecycle, "_current_instance_id", None)
    second = server_lifecycle.start_server_lifecycle()

    assert second.previous_exit == "unexpected"


def test_clean_shutdown_is_not_classified_as_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    monkeypatch.setattr(server_lifecycle, "_current_instance_id", None)

    server_lifecycle.start_server_lifecycle()
    server_lifecycle.mark_server_clean_shutdown()
    monkeypatch.setattr(server_lifecycle, "_current_instance_id", None)

    assert server_lifecycle.start_server_lifecycle().previous_exit == "normal"


def test_corrupt_marker_does_not_prevent_start(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    monkeypatch.setattr(server_lifecycle, "_current_instance_id", None)
    (tmp_path / "server-lifecycle.json").write_text("not-json", encoding="utf-8")

    lifecycle = server_lifecycle.start_server_lifecycle()

    assert lifecycle.previous_exit == "unknown"
    assert json.loads((tmp_path / "server-lifecycle.json").read_text())["state"] == "running"


def test_unknown_marker_state_is_not_mistaken_for_clean_shutdown(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    monkeypatch.setattr(server_lifecycle, "_current_instance_id", None)
    (tmp_path / "server-lifecycle.json").write_text('{"state":"partial"}', encoding="utf-8")

    assert server_lifecycle.start_server_lifecycle().previous_exit == "unknown"
