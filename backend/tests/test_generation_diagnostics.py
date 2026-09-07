"""Regression coverage for safe generation failure diagnostics."""

import json

from backend import config
from backend.services.generation_diagnostics import (
    TERMINAL_STATUS_MISSING,
    write_generation_diagnostic,
)


def test_generation_diagnostic_is_actionable_without_user_content(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_data_dir", tmp_path)

    diagnostic = write_generation_diagnostic(
        kind="terminal_status_missing",
        generation_id="generation-1",
        engine="qwen",
        model_size="1.7B",
        progress_current=1,
        progress_total=3,
        lifecycle="normal",
    )

    assert diagnostic.error_code.startswith(f"{TERMINAL_STATUS_MISSING}:{diagnostic.diagnostic_id}:1/3")
    record = json.loads((tmp_path / "logs" / "generation-diagnostics.jsonl").read_text().strip())
    assert record == {
        "id": diagnostic.diagnostic_id,
        "created_at": record["created_at"],
        "generation_id": "generation-1",
        "kind": "terminal_status_missing",
        "engine": "qwen",
        "model_size": "1.7B",
        "progress_current": 1,
        "progress_total": 3,
        "lifecycle": "normal",
    }
    serialized = json.dumps(record, ensure_ascii=False)
    assert "用户正文" not in serialized
    assert "/Users/" not in serialized
    assert "Traceback" not in serialized
