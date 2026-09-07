import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base, Generation
from backend.services import task_queue


def _configure_queue_test_database(monkeypatch, tmp_path):
    """Bind queue recovery to an isolated SQLite database for one test."""
    monkeypatch.setattr(config, "_data_dir", tmp_path / "data")
    engine = create_engine(f"sqlite:///{tmp_path / 'queue-recovery.db'}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr("backend.database.session.SessionLocal", session_factory)
    return engine, session_factory


def _active_generation(
    generation_id: str = "orphan",
    *,
    progress_current: int | None = None,
    progress_total: int | None = None,
) -> Generation:
    return Generation(
        id=generation_id,
        profile_id="profile",
        text="保留原始文本",
        language="zh",
        audio_path="generations/existing.wav",
        status="generating",
        engine="cosyvoice",
        model_size="rl",
        progress_current=progress_current,
        progress_total=progress_total,
    )


@pytest.mark.asyncio
async def test_cancel_queued_generation_skips_execution():
    task_queue.init_queue(force=True)

    running_started = asyncio.Event()
    release_running = asyncio.Event()
    queued_ran = asyncio.Event()

    async def running_job():
        running_started.set()
        await release_running.wait()

    async def queued_job():
        queued_ran.set()

    task_queue.enqueue_generation("gen-running", running_job())
    await asyncio.wait_for(running_started.wait(), timeout=1)

    task_queue.enqueue_generation("gen-queued", queued_job())
    assert task_queue.cancel_generation("gen-queued") == "queued"

    release_running.set()
    await asyncio.sleep(0.1)

    assert not queued_ran.is_set()


@pytest.mark.asyncio
async def test_cancel_running_generation_cancels_task():
    task_queue.init_queue(force=True)

    running_started = asyncio.Event()
    running_cancelled = asyncio.Event()

    async def running_job():
        running_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            running_cancelled.set()
            raise

    task_queue.enqueue_generation("gen-running", running_job())
    await asyncio.wait_for(running_started.wait(), timeout=1)

    assert task_queue.cancel_generation("gen-running") == "running"
    await asyncio.wait_for(running_cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_timed_out_generation_releases_queue_for_next_job(monkeypatch):
    task_queue.init_queue(force=True)
    monkeypatch.setattr(
        "backend.config.get_generation_execution_timeout_seconds",
        lambda: 0.01,
        raising=False,
    )

    timed_out_cancelled = asyncio.Event()
    cancelled_job_ran = asyncio.Event()
    next_job_ran = asyncio.Event()
    timeout_failures = []

    async def stalled_job():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            timed_out_cancelled.set()
            raise

    async def next_job():
        next_job_ran.set()

    async def cancelled_job():
        cancelled_job_ran.set()

    async def record_timeout_failure(generation_id, error, *, force=False):
        timeout_failures.append((generation_id, error, force))

    monkeypatch.setattr(task_queue, "_force_fail_if_active", record_timeout_failure)

    task_queue.enqueue_generation("timed-out", stalled_job())
    task_queue.enqueue_generation("cancelled", cancelled_job())
    task_queue.enqueue_generation("next", next_job())
    assert task_queue.cancel_generation("cancelled") == "queued"

    await asyncio.wait_for(timed_out_cancelled.wait(), timeout=1)
    await asyncio.wait_for(next_job_ran.wait(), timeout=1)

    assert timeout_failures == [
        ("timed-out", "Generation timed out. Please retry.", True)
    ]
    assert not cancelled_job_ran.is_set()


@pytest.mark.asyncio
async def test_generation_that_finishes_before_timeout_is_not_cancelled(monkeypatch):
    task_queue.init_queue(force=True)
    monkeypatch.setattr(
        "backend.config.get_generation_execution_timeout_seconds",
        lambda: 1,
        raising=False,
    )

    completed = asyncio.Event()

    async def quick_job():
        completed.set()

    task_queue.enqueue_generation("quick", quick_job())

    await asyncio.wait_for(completed.wait(), timeout=1)
    await asyncio.sleep(0)

    assert task_queue.cancel_generation("quick") is None


@pytest.mark.asyncio
async def test_unexpected_idle_worker_exit_is_replaced_and_consumes_waiting_job(monkeypatch):
    original_worker = task_queue._generation_worker
    worker_starts = 0
    ran = asyncio.Event()

    async def exit_once_then_work():
        nonlocal worker_starts
        worker_starts += 1
        if worker_starts == 1:
            return
        await original_worker()

    monkeypatch.setattr(task_queue, "_generation_worker", exit_once_then_work)
    task_queue.init_queue(force=True)
    try:
        for _ in range(20):
            if worker_starts >= 2:
                break
            await asyncio.sleep(0.01)
        assert worker_starts >= 2

        async def queued_job():
            ran.set()

        task_queue.enqueue_generation("after-worker-restart", queued_job())
        await asyncio.wait_for(ran.wait(), timeout=1)
    finally:
        task_queue.shutdown_queue()


@pytest.mark.asyncio
async def test_recovery_marks_unowned_active_generation_failed_without_deleting_content(monkeypatch, tmp_path):
    engine, session_factory = _configure_queue_test_database(monkeypatch, tmp_path)
    session = session_factory()
    session.add(_active_generation())
    session.commit()

    task_queue.init_queue(force=True)
    try:
        recovered = await task_queue.recover_orphaned_generations()
        session.expire_all()
        generation = session.query(Generation).filter_by(id="orphan").one()

        assert recovered == 1
        assert generation.status == "failed"
        assert generation.error == "Generation interrupted before completion"
        assert generation.text == "保留原始文本"
        assert generation.audio_path == "generations/existing.wav"
    finally:
        task_queue.shutdown_queue()
        session.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_recovery_keeps_currently_running_generation_active(monkeypatch, tmp_path):
    engine, session_factory = _configure_queue_test_database(monkeypatch, tmp_path)
    session = session_factory()
    session.add(_active_generation("running"))
    session.commit()

    started = asyncio.Event()
    release = asyncio.Event()

    async def long_running_job():
        started.set()
        await release.wait()

    task_queue.init_queue(force=True)
    try:
        task_queue.enqueue_generation("running", long_running_job())
        await asyncio.wait_for(started.wait(), timeout=1)

        recovered = await task_queue.recover_orphaned_generations()
        session.expire_all()

        assert recovered == 0
        assert session.query(Generation).filter_by(id="running").one().status == "generating"
    finally:
        release.set()
        task_queue.shutdown_queue()
        session.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_runner_returning_without_terminal_status_is_diagnosed_and_next_job_runs(
    monkeypatch, tmp_path
):
    engine, session_factory = _configure_queue_test_database(monkeypatch, tmp_path)
    session = session_factory()
    session.add(_active_generation("unfinished", progress_current=1, progress_total=3))
    session.commit()
    next_job_ran = asyncio.Event()

    async def unfinished_job():
        # This emulates the historical failure: inference got through one
        # semantic segment but the runner returned before publishing a status.
        return None

    async def next_job():
        next_job_ran.set()

    task_queue.init_queue(force=True)
    try:
        task_queue.enqueue_generation("unfinished", unfinished_job())
        task_queue.enqueue_generation("next", next_job())
        await asyncio.wait_for(next_job_ran.wait(), timeout=1)

        session.expire_all()
        generation = session.query(Generation).filter_by(id="unfinished").one()
        assert generation.status == "failed"
        assert generation.progress_current == 1
        assert generation.progress_total == 3
        assert generation.error.startswith("GENERATION_TERMINAL_STATUS_MISSING:")
        assert generation.error.endswith(":1/3")
    finally:
        task_queue.shutdown_queue()
        session.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_worker_exception_is_diagnosed_and_next_job_runs(monkeypatch, tmp_path):
    engine, session_factory = _configure_queue_test_database(monkeypatch, tmp_path)
    session = session_factory()
    session.add(_active_generation("worker-error", progress_current=2, progress_total=3))
    session.commit()
    next_job_ran = asyncio.Event()

    async def failing_job():
        raise RuntimeError("do not expose this worker message")

    async def next_job():
        next_job_ran.set()

    task_queue.init_queue(force=True)
    try:
        task_queue.enqueue_generation("worker-error", failing_job())
        task_queue.enqueue_generation("next", next_job())
        await asyncio.wait_for(next_job_ran.wait(), timeout=1)

        session.expire_all()
        generation = session.query(Generation).filter_by(id="worker-error").one()
        assert generation.status == "failed"
        assert generation.error.startswith("GENERATION_WORKER_EXITED:")
        assert generation.error.endswith(":2/3")
        assert "do not expose" not in generation.error
    finally:
        task_queue.shutdown_queue()
        session.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_abnormal_server_recovery_uses_lifecycle_diagnostic(monkeypatch, tmp_path):
    engine, session_factory = _configure_queue_test_database(monkeypatch, tmp_path)
    session = session_factory()
    session.add(_active_generation("server-exit", progress_current=1, progress_total=3))
    session.commit()

    task_queue.init_queue(force=True)
    try:
        assert await task_queue.recover_orphaned_generations(lifecycle="unexpected") == 1
        session.expire_all()
        generation = session.query(Generation).filter_by(id="server-exit").one()
        assert generation.status == "failed"
        assert generation.error.startswith("GENERATION_SERVER_EXITED:")
        assert generation.error.endswith(":1/3")
    finally:
        task_queue.shutdown_queue()
        session.close()
        engine.dispose()
