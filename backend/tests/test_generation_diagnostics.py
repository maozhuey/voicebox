"""Regression coverage for safe generation failure diagnostics."""

import json
import zlib

import pytest

from backend import config
from backend.services.generation_diagnostics import (
    BINARY_MODULE_EXTRACTION_FAILED,
    TERMINAL_STATUS_MISSING,
    classify_binary_extraction_failure,
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


def test_binary_module_extraction_failed_records_subtype(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_data_dir", tmp_path)

    diagnostic = write_generation_diagnostic(
        kind="binary_module_extraction_failed",
        generation_id="generation-2",
        engine="qwen_custom_voice",
        model_size="1.7B",
        progress_current=None,
        progress_total=None,
        lifecycle="unexpected",
        failure_subtype="zlib_corruption",
    )

    assert diagnostic.error_code == f"{BINARY_MODULE_EXTRACTION_FAILED}:{diagnostic.diagnostic_id}"
    record = json.loads((tmp_path / "logs" / "generation-diagnostics.jsonl").read_text().strip())
    assert record["kind"] == "binary_module_extraction_failed"
    assert record["failure_subtype"] == "zlib_corruption"
    # Subtype is internal support metadata and must never leak the underlying
    # zlib message or any path the user controls.
    assert "incorrect header" not in json.dumps(record, ensure_ascii=False)
    assert "Traceback" not in json.dumps(record, ensure_ascii=False)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (zlib.error("Error -3 while decompressing data: incorrect header check"), "zlib_corruption"),
        (ModuleNotFoundError("No module named 'backend.backends.qwen_custom_voice_backend'"), "module_not_found"),
        (ImportError("import of backend.backends.qwen_custom_voice_backend halted"), "import_error"),
    ],
)
def test_classify_binary_extraction_failure_matches_top_level(exc, expected):
    assert classify_binary_extraction_failure(exc) == expected


def test_classify_binary_extraction_failure_walks_cause_chain():
    # Real failure path: ``pyimod02_importers`` wraps a ``zlib.error`` in an
    # ``ImportError`` while the module loader extracts a corrupt PYZ entry.
    # The classifier must still recognise the underlying zlib failure.
    inner = zlib.error("Error -3 while decompressing data: incorrect header check")
    outer = ImportError("Loader failed")
    outer.__cause__ = inner
    assert classify_binary_extraction_failure(outer) == "zlib_corruption"


def test_classify_binary_extraction_failure_returns_none_for_unrelated():
    # RuntimeError with no zlib/ImportError cause must not be misclassified;
    # otherwise the generation path would hide legitimate failures behind a
    # reinstall hint.
    class _UnrelatedExtractionError(RuntimeError):
        pass

    assert classify_binary_extraction_failure(_UnrelatedExtractionError("boom")) is None
    assert classify_binary_extraction_failure(ValueError("boom")) is None
    assert classify_binary_extraction_failure(None) is None  # type: ignore[arg-type]


def test_classify_binary_extraction_failure_handles_cycles():
    # A pathological cycle in the cause chain must terminate instead of
    # looping forever; returning ``None`` keeps the failure path safe.
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert classify_binary_extraction_failure(a) is None
