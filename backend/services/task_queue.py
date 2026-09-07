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
from .generation_diagnostics import write_generation_diagnostic

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
    # A force-reinitialization replaces the module-level queue while an old
    # worker may still be unwinding cancellation. Keep this worker bound to
    # the queue it consumed from: its ``task_done`` must never decrement a new
    # queue's unfinished-task counter or make a later generation disappear.
    worker_queue = _generation_queue
    while True:
        job = await worker_queue.get()
        job_finished_cleanly = False
        runner_result = None
        try:
            if job.generation_id in _cancelled_generation_ids:
                _cancelled_generation_ids.discard(job.generation_id)
                job.coro.close()
                continue

            task = asyncio.create_task(job.coro)
            _running_generation_tasks[job.generation_id] = task
            _queued_generation_ids.discard(job.generation_id)
            try:
                runner_result = await asyncio.wait_for(
                    task,
                    timeout=config.get_generation_execution_timeout_seconds(),
                )
                if getattr(runner_result, "status", None) not in {"completed", "failed"}:
                    # Legacy or third-party runners cannot prove that they
                    # published a terminal record. The database verification
                    # in ``finally`` remains authoritative, but this log makes
                    # an accidental bare ``return`` diagnosable during a
                    # backend integration without exposing request content.
                    logger.warning(
                        "Generation runner returned without a terminal receipt: %s",
                        job.generation_id,
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
                # A user cancellation targets the child generation task and
                # is an expected terminal path. A queue shutdown or forced
                # reinitialization targets this worker instead; swallowing it
                # would leave a zombie consumer waiting on the old queue.
                if _queue_stopping or asyncio.current_task().cancelling() or not task.cancelled():
                    raise
        except Exception:
            traceback.print_exc()
            await _mark_generation_failed_if_active(
                job.generation_id,
                failure_kind="worker_exited",
            )
        finally:
            _running_generation_tasks.pop(job.generation_id, None)
            _queued_generation_ids.discard(job.generation_id)
            # A coroutine returning is not proof that the user-visible result
            # was committed. The queue retains ownership until a fresh database
            # read confirms a terminal state, preventing a 1/3 task from being
            # silently treated as a normal completion.
            if job_finished_cleanly:
                await _mark_generation_failed_if_active(
                    job.generation_id,
                    failure_kind="terminal_status_missing",
                )
            worker_queue.task_done()


async def _mark_generation_failed_if_active(
    generation_id: str,
    error: str | None = None,
    *,
    force: bool = False,
    failure_kind: Literal["terminal_status_missing", "worker_exited"] | None = None,
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
            if failure_kind is not None:
                diagnostic = write_generation_diagnostic(
                    kind=failure_kind,
                    generation_id=generation_id,
                    engine=gen.engine,
                    model_size=gen.model_size,
                    progress_current=gen.progress_current,
                    progress_total=gen.progress_total,
                    lifecycle="normal",
                )
                error = diagnostic.error_code
            if error is None:
                error = _INTERRUPTED_GENERATION_ERROR
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


async def recover_orphaned_generations(
    *,
    error: str = _INTERRUPTED_GENERATION_ERROR,
    lifecycle: Literal["normal", "unexpected", "unknown"] | None = None,
) -> int:
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
                resolved_error = error
                if lifecycle is not None:
                    diagnostic = write_generation_diagnostic(
                        kind="server_exited",
                        generation_id=generation.id,
                        engine=generation.engine,
                        model_size=generation.model_size,
                        progress_current=generation.progress_current,
                        progress_total=generation.progress_total,
                        lifecycle=lifecycle,
                    )
                    resolved_error = diagnostic.error_code
                await history.update_generation_status(
                    generation.id,
                    status="failed",
                    db=db,
                    error=resolved_error,
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
    _cancel_task_if_loop_open(_generation_worker_task)
    for task in list(_running_generation_tasks.values()):
        task.cancel()


def _cancel_task_if_loop_open(task: asyncio.Task | None) -> None:
    """Cancel a task only while its owner loop can receive cancellation.

    Test and desktop reload paths can replace a queue after its previous event
    loop closed. Calling ``Task.cancel`` on that task raises and prevents the
    new worker from starting, even though the stale task cannot execute again.
    """
    if task is not None and not task.done() and not task.get_loop().is_closed():
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
        _cancel_task_if_loop_open(_generation_worker_task)
        for task in list(_running_generation_tasks.values()):
            task.cancel()

    _generation_queue = asyncio.Queue()
    _queued_generation_ids = set()
    _running_generation_tasks = {}
    _cancelled_generation_ids = set()
    _queue_stopping = False
    _start_generation_worker()
