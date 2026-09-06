"""
Serial generation queue — ensures only one TTS inference runs at a time
to avoid GPU contention.
"""

import asyncio
import logging
import traceback
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Literal

from .. import config

logger = logging.getLogger(__name__)

_INTERRUPTED_GENERATION_ERROR = "Generation interrupted before completion"

# Keep references to fire-and-forget background tasks to prevent GC
_background_tasks: set = set()


@dataclass
class GenerationJob:
    """Queued generation work plus the generation ID it belongs to."""

    generation_id: str
    coro: Coroutine


# Generation queue — serializes TTS inference to avoid GPU contention
_generation_queue: asyncio.Queue = None  # type: ignore  # initialized at startup
_generation_worker_task: asyncio.Task | None = None
_queued_generation_ids: set[str] = set()
_running_generation_tasks: dict[str, asyncio.Task] = {}
_cancelled_generation_ids: set[str] = set()
_queue_stopping = False


def create_background_task(coro) -> asyncio.Task:
    """Create a background task and prevent it from being garbage collected."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _generation_worker():
    """Worker that processes generation tasks one at a time."""
    while True:
        job = await _generation_queue.get()
        job_finished_cleanly = False
        try:
            if job.generation_id in _cancelled_generation_ids:
                _cancelled_generation_ids.discard(job.generation_id)
                job.coro.close()
                continue

            task = asyncio.create_task(job.coro)
            _running_generation_tasks[job.generation_id] = task
            _queued_generation_ids.discard(job.generation_id)
            try:
                await asyncio.wait_for(
                    task,
                    timeout=config.get_generation_execution_timeout_seconds(),
                )
                job_finished_cleanly = True
            except TimeoutError:
                # A stuck model call blocks every later request because this is
                # the only inference worker. Persist a retryable terminal state
                # even when the cancelled generation already wrote its generic
                # cancellation error, then allow the next queued job to start.
                await _force_fail_if_active(
                    job.generation_id,
                    "Generation timed out. Please retry.",
                    force=True,
                )
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
        except Exception:
            traceback.print_exc()
            await _force_fail_if_active(
                job.generation_id,
                "Worker exited without writing terminal status",
            )
        finally:
            _running_generation_tasks.pop(job.generation_id, None)
            _queued_generation_ids.discard(job.generation_id)
            # A generation coroutine can return after its own error/status
            # write failed (for example, during a transient SQLite lock). The
            # database is what drives the desktop UI, so never leave an active
            # row behind once this process no longer owns its execution.
            if job_finished_cleanly:
                await _mark_generation_failed_if_active(
                    job.generation_id,
                    _INTERRUPTED_GENERATION_ERROR,
                )
            _generation_queue.task_done()


async def _mark_generation_failed_if_active(
    generation_id: str,
    error: str,
    *,
    force: bool = False,
) -> None:
    """Best-effort terminal write for one unowned active generation record."""
    try:
        from ..database import Generation as DBGeneration, get_db
        from ..database.session import SessionLocal
        from . import history

        # Queue-only tests intentionally run without initializing SQLite.
        # There is no persisted row to recover in that environment.
        if SessionLocal is None:
            return

        db = next(get_db())
        try:
            gen = db.query(DBGeneration).filter_by(id=generation_id).first()
            if gen is None:
                return
            active_statuses = ("loading_model", "generating")
            status = gen.status or "completed"
            if status not in active_statuses and not (force and status == "failed"):
                return
            await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=db,
                error=error,
            )
        finally:
            db.close()
    except Exception:
        traceback.print_exc()


async def _force_fail_if_active(
    generation_id: str,
    error: str,
    *,
    force: bool = False,
) -> None:
    """Compatibility wrapper used by timeout and worker-exception paths."""
    await _mark_generation_failed_if_active(generation_id, error, force=force)


def enqueue_generation(generation_id: str, coro):
    """Add a generation coroutine to the serial queue."""
    if _generation_queue is None:
        raise RuntimeError("Generation queue has not been initialized")

    _queued_generation_ids.add(generation_id)
    _generation_queue.put_nowait(GenerationJob(generation_id=generation_id, coro=coro))


def cancel_generation(generation_id: str) -> Literal["queued", "running"] | None:
    """Cancel a queued or running generation if it is still active."""
    running_task = _running_generation_tasks.get(generation_id)
    if running_task is not None:
        running_task.cancel()
        return "running"

    if generation_id in _queued_generation_ids:
        _queued_generation_ids.discard(generation_id)
        _cancelled_generation_ids.add(generation_id)
        return "queued"

    return None


def get_tracked_generation_ids() -> set[str]:
    """Return generation IDs still owned by this server process.

    The recovery path uses this exact in-memory ownership signal instead of a
    duration threshold: a long CosyVoice inference remains protected while a
    database row with no queued or running owner can be safely concluded.
    """
    return set(_queued_generation_ids) | set(_running_generation_tasks)


async def recover_orphaned_generations(*, error: str = _INTERRUPTED_GENERATION_ERROR) -> int:
    """Fail active history rows that are not owned by the current queue.

    Recovery never replays work and never deletes user content. A previous
    process has no in-memory ownership after startup, while runtime recovery
    preserves rows that are still queued or executing in this process.
    """
    try:
        from ..database import Generation as DBGeneration, get_db
        from . import history

        tracked_ids = get_tracked_generation_ids()
        db = next(get_db())
        try:
            active_rows = (
                db.query(DBGeneration)
                .filter(DBGeneration.status.in_(("loading_model", "generating")))
                .all()
            )
            recovered = 0
            for generation in active_rows:
                if generation.id in tracked_ids:
                    continue
                await history.update_generation_status(
                    generation.id,
                    status="failed",
                    db=db,
                    error=error,
                )
                recovered += 1
            if recovered:
                logger.warning("Recovered %d orphaned generation(s)", recovered)
            return recovered
        finally:
            db.close()
    except Exception:
        logger.exception("Could not recover orphaned generations")
        return 0


def _start_generation_worker() -> None:
    """Start one supervised worker for the current queue instance."""
    global _generation_worker_task
    _generation_worker_task = create_background_task(_generation_worker())
    _generation_worker_task.add_done_callback(_handle_generation_worker_exit)


def _handle_generation_worker_exit(task: asyncio.Task) -> None:
    """Restore an unexpectedly ended idle worker and close lost active rows."""
    if task is not _generation_worker_task or _queue_stopping or _generation_queue is None:
        return

    try:
        if not task.cancelled():
            exception = task.exception()
            if exception is not None:
                logger.error("Generation worker exited unexpectedly: %s", exception)
            else:
                logger.warning("Generation worker exited unexpectedly")
        else:
            # Cancellation is the expected shutdown/force-reinitialization
            # signal. A crashed worker reports an exception or a normal return
            # and is restarted below; do not revive it during event-loop exit.
            return
    except asyncio.CancelledError:
        return

    # This callback runs on the worker's event loop after it has released any
    # queue item. Start a fresh consumer first, then close only rows that no
    # longer have queued/running ownership in this process.
    _start_generation_worker()
    create_background_task(recover_orphaned_generations())


def shutdown_queue() -> None:
    """Stop queue supervision during server shutdown without triggering a restart."""
    global _queue_stopping
    _queue_stopping = True
    if _generation_worker_task is not None and not _generation_worker_task.done():
        if not _generation_worker_task.get_loop().is_closed():
            _generation_worker_task.cancel()
    for task in list(_running_generation_tasks.values()):
        task.cancel()


def init_queue(force: bool = False):
    """Initialize the generation queue and start the worker.

    Must be called once during application startup (inside a running event loop).
    """
    global _generation_queue, _generation_worker_task
    global _queued_generation_ids, _running_generation_tasks, _cancelled_generation_ids, _queue_stopping

    if _generation_worker_task is not None and not _generation_worker_task.done():
        if not force:
            return
        _generation_worker_task.cancel()
        for task in list(_running_generation_tasks.values()):
            task.cancel()

    _generation_queue = asyncio.Queue()
    _queued_generation_ids = set()
    _running_generation_tasks = {}
    _cancelled_generation_ids = set()
    _queue_stopping = False
    _start_generation_worker()
