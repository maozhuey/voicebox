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
# The PyInstaller-packaged sidecar binary embeds every backend module in a
# single zlib-compressed PYZ archive. When that archive is corrupted (most
# often by an interrupted prior build or a copy that truncated the file),
# ``import backend.backends.<x>`` raises ``zlib.error`` before the module is
# even loaded. The same user-visible outcome happens when an import fails
# for any other reason rooted in the bundled binary. These three error
# classes share a single user-facing message: "the Voicebox install is
# broken, reinstall to recover". Mapping them to one stable code keeps the
# client copy short and avoids leaking zlib internals into the UI.
BINARY_MODULE_EXTRACTION_FAILED = "BINARY_MODULE_EXTRACTION_FAILED"

DiagnosticKind = Literal[
    "terminal_status_missing",
    "worker_exited",
    "server_exited",
    "binary_module_extraction_failed",
]
LifecycleKind = Literal["normal", "unexpected", "unknown"]
# Sub-classes of failures that fall under ``binary_module_extraction_failed``.
# Recorded for support correlation only; never shown to end users.
BinaryFailureSubtype = Literal["zlib_corruption", "import_error", "module_not_found"]


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
        "binary_module_extraction_failed": BINARY_MODULE_EXTRACTION_FAILED,
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
    failure_subtype: str | None = None,
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
    # ``failure_subtype`` is opt-in: existing kinds never emit it, only the
    # binary-extraction kind passes through here. Keeping it as an explicit
    # kwarg avoids widening the schema for unrelated failure paths.
    if failure_subtype is not None:
        record["failure_subtype"] = failure_subtype
    try:
        with _diagnostics_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        # Failure-state persistence must not prevent the queue from concluding
        # the generation row.  The error code remains useful even if logging is
        # unavailable (for example, a nearly-full local disk).
        logger.exception("Could not write generation diagnostic %s", diagnostic_id)

    progress = (
        f":{progress_current}/{progress_total}" if progress_current is not None and progress_total is not None else ""
    )
    return GenerationDiagnostic(
        diagnostic_id=diagnostic_id,
        error_code=f"{code}:{diagnostic_id}{progress}",
    )


def classify_binary_extraction_failure(exc: BaseException) -> BinaryFailureSubtype | None:
    """Return a stable subtype for a binary-extraction failure, or ``None``.

    Walks ``__cause__`` / ``__context__`` so a wrapped ``zlib.error`` raised
    inside ``pyimod02_importers`` still matches the most specific cause.
    Returning ``None`` means the exception does not belong to the
    binary-extraction family and should be surfaced as a normal failure.

    Order matters: ``ModuleNotFoundError`` is the most specific import failure
    and must be reported as such, even when an outer ``ImportError`` would
    match first by isinstance. ``zlib.error`` likewise takes precedence over
    a wrapping ``ImportError`` so the support team can tell a corrupt PYZ
    apart from a missing module without re-reading the traceback.
    """
    import zlib as _zlib

    saw_import = False
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ModuleNotFoundError):
            return "module_not_found"
        if isinstance(current, _zlib.error):
            return "zlib_corruption"
        if isinstance(current, ImportError):
            saw_import = True
        current = current.__cause__ or current.__context__
    return "import_error" if saw_import else None
