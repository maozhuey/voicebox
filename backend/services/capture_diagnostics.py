"""Privacy-safe, durable diagnostics for failed capture transcription."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import config

_MAX_LOG_BYTES = 1_000_000


def record_capture_failure(*, stage: str, source: str, filename: str, model_size: str, error: Exception) -> str:
    """Persist a minimal local record without retaining audio or transcript content."""
    diagnostic_id = f"cap-{uuid.uuid4().hex[:10]}"
    entry = {
        "id": diagnostic_id,
        "at": datetime.now(timezone.utc).isoformat(),
        "kind": "transcription",
        "stage": stage,
        "source": source,
        "extension": Path(filename).suffix.lower() or "unknown",
        "model_size": model_size,
        "error_type": type(error).__name__,
        # The exception class is sufficient to group failures. Never preserve
        # the original exception text: decoder and provider errors can echo a
        # filename, a filesystem path, or user-spoken content.
        "summary": _sanitize_summary(str(error)),
    }
    _append_entry(entry)
    return diagnostic_id


def record_capture_refinement_failure(
    *, stage: str, source: str, model_size: str, error: Exception
) -> str:
    """Record optional capture refinement failures without storing user content.

    Refinement already operates on a persisted capture, so filenames and audio
    paths add no diagnostic value and must not be copied into the local log.
    """
    diagnostic_id = f"cap-{uuid.uuid4().hex[:10]}"
    entry = {
        "id": diagnostic_id,
        "at": datetime.now(timezone.utc).isoformat(),
        "kind": "refinement",
        "stage": stage,
        "source": source,
        "model_size": model_size,
        "error_type": type(error).__name__,
        "summary": _sanitize_summary(str(error)),
    }
    _append_entry(entry)
    return diagnostic_id


def _append_entry(entry: dict) -> None:
    """Append a diagnostic best-effort; logging must never alter request outcome."""
    try:
        log_dir = config.get_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "capture-diagnostics.jsonl"
        if log_path.exists() and log_path.stat().st_size >= _MAX_LOG_BYTES:
            rotated = log_path.with_suffix(".jsonl.1")
            os.replace(log_path, rotated)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        # Diagnostics must not turn a recoverable request failure into a crash.
        pass


def _sanitize_summary(message: str) -> str:
    """Return only a fixed, privacy-safe marker for the original exception."""
    del message
    return "原始异常信息已脱敏"
