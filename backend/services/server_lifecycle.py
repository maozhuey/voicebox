"""Durable lifecycle evidence for distinguishing clean and abnormal exits."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .. import config

logger = logging.getLogger(__name__)

LifecycleExitKind = Literal["normal", "unexpected", "unknown"]


@dataclass(frozen=True)
class ServerLifecycle:
    """State owned by the current server instance."""

    instance_id: str
    previous_exit: LifecycleExitKind


_current_instance_id: str | None = None


def _marker_path() -> Path:
    return config.get_data_dir() / "server-lifecycle.json"


def _read_previous_exit(path: Path) -> LifecycleExitKind:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return "unknown"
    state = payload.get("state")
    if state == "running":
        return "unexpected"
    if state == "clean_shutdown":
        return "normal"
    return "unknown"


def _write_marker(payload: dict) -> None:
    """Atomically replace the marker so power loss cannot create partial JSON."""
    path = _marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        with suppress(OSError):
            Path(temporary_name).unlink(missing_ok=True)


def start_server_lifecycle() -> ServerLifecycle:
    """Record this running process and return how the previous one ended."""
    global _current_instance_id
    path = _marker_path()
    previous_exit = _read_previous_exit(path) if path.exists() else "normal"
    _current_instance_id = str(uuid.uuid4())
    try:
        _write_marker(
            {
                "instance_id": _current_instance_id,
                "state": "running",
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
    except OSError:
        # Lifecycle evidence improves recovery but must never stop local speech
        # when a data directory is temporarily read-only or the disk is full.
        logger.exception("Could not write server lifecycle start marker")
        previous_exit = "unknown"
    return ServerLifecycle(instance_id=_current_instance_id, previous_exit=previous_exit)


def mark_server_clean_shutdown() -> None:
    """Mark a graceful lifespan exit without claiming a native crash was clean."""
    if _current_instance_id is None:
        return
    try:
        _write_marker(
            {
                "instance_id": _current_instance_id,
                "state": "clean_shutdown",
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
    except OSError:
        logger.exception("Could not write server lifecycle shutdown marker")
