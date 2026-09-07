"""Safe, local diagnostics for generations that cannot publish a terminal state."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .. import config

logger = logging.getLogger(__name__)

TERMINAL_STATUS_MISSING = "GENERATION_TERMINAL_STATUS_MISSING"
WORKER_EXITED = "GENERATION_WORKER_EXITED"
SERVER_EXITED = "GENERATION_SERVER_EXITED"

DiagnosticKind = Literal[
    "terminal_status_missing",
    "worker_exited",
    "server_exited",
]
LifecycleKind = Literal["normal", "unexpected", "unknown"]


@dataclass(frozen=True)
class GenerationDiagnostic:
    """A bounded support record that deliberately excludes user-owned content."""

    diagnostic_id: str
    error_code: str


def _diagnostics_path() -> Path:
    path = config.get_data_dir() / "logs" / "generation-diagnostics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _error_code(kind: DiagnosticKind) -> str:
    return {
        "terminal_status_missing": TERMINAL_STATUS_MISSING,
        "worker_exited": WORKER_EXITED,
        "server_exited": SERVER_EXITED,
    }[kind]


def write_generation_diagnostic(
    *,
    kind: DiagnosticKind,
    generation_id: str,
    engine: str | None,
    model_size: str | None,
    progress_current: int | None,
    progress_total: int | None,
    lifecycle: LifecycleKind,
) -> GenerationDiagnostic:
    """Append a safe diagnostic and return the stable error stored in history.

    Business rule: failure diagnostics are for local support correlation, not a
    second copy of a user's text or voice assets.  Keep the schema intentionally
    small so a failed task remains actionable without widening local retention.
    """
    diagnostic_id = str(uuid.uuid4())
    code = _error_code(kind)
    record = {
        "id": diagnostic_id,
        "created_at": datetime.now(UTC).isoformat(),
        "generation_id": generation_id,
        "kind": kind,
        "engine": engine or "unknown",
        "model_size": model_size or "unknown",
        "progress_current": progress_current,
        "progress_total": progress_total,
        "lifecycle": lifecycle,
    }
    try:
        with _diagnostics_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        # Failure-state persistence must not prevent the queue from concluding
        # the generation row.  The error code remains useful even if logging is
        # unavailable (for example, a nearly-full local disk).
        logger.exception("Could not write generation diagnostic %s", diagnostic_id)

    progress = (
        f":{progress_current}/{progress_total}"
        if progress_current is not None and progress_total is not None
        else ""
    )
    return GenerationDiagnostic(
        diagnostic_id=diagnostic_id,
        error_code=f"{code}:{diagnostic_id}{progress}",
    )
