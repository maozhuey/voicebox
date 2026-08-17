"""Regression coverage for model downloads on restricted HF networks."""

import asyncio
import ssl

import pytest

from backend import models
from backend.routes import models as models_route
from backend.utils.progress import ProgressManager
from backend.utils.tasks import TaskManager


class _ModelConfig:
    model_name = "qwen3-0.6b"


@pytest.mark.asyncio
async def test_download_retries_through_mirror_after_ssl_failure(monkeypatch):
    """An SSL EOF from the official Hub gets one mirror retry."""
    from backend.utils import hf_download

    attempts = 0
    activated: list[str] = []

    async def load_model():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ssl.SSLEOFError(8, "EOF occurred in violation of protocol")

    monkeypatch.setattr(hf_download, "get_current_hf_endpoint", lambda: hf_download.DEFAULT_HF_ENDPOINT)
    monkeypatch.setattr(hf_download, "activate_hf_endpoint", activated.append)

    used_endpoint = await hf_download.run_with_hf_fallback(load_model)

    assert attempts == 2
    assert activated == [hf_download.DEFAULT_HF_FALLBACK_ENDPOINT]
    assert used_endpoint == hf_download.DEFAULT_HF_FALLBACK_ENDPOINT


@pytest.mark.asyncio
async def test_download_does_not_retry_auth_or_model_errors(monkeypatch):
    """Only connectivity failures switch source; application errors remain exact."""
    from backend.utils import hf_download

    attempts = 0

    async def load_model():
        nonlocal attempts
        attempts += 1
        raise ValueError("Unknown model architecture")

    monkeypatch.setattr(hf_download, "get_current_hf_endpoint", lambda: hf_download.DEFAULT_HF_ENDPOINT)

    with pytest.raises(ValueError, match="Unknown model architecture"):
        await hf_download.run_with_hf_fallback(load_model)

    assert attempts == 1


@pytest.mark.asyncio
async def test_terminal_download_error_reaches_task_and_progress_managers(monkeypatch):
    """A background failure must remain visible to both polling and SSE clients."""
    task_manager = TaskManager()
    progress_manager = ProgressManager()
    background_tasks = []

    async def fail_download():
        raise RuntimeError("mirror unavailable")

    monkeypatch.setattr(models_route, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(models_route, "get_progress_manager", lambda: progress_manager)
    monkeypatch.setattr(
        "backend.backends.get_model_config",
        lambda _name: _ModelConfig(),
    )
    monkeypatch.setattr(
        "backend.backends.get_model_load_func",
        lambda _config: fail_download,
    )
    monkeypatch.setattr(
        models_route,
        "create_background_task",
        lambda coroutine: background_tasks.append(coroutine),
    )

    await models_route.trigger_model_download(models.ModelDownloadRequest(model_name="qwen3-0.6b"))
    await asyncio.gather(*background_tasks)

    task = task_manager.get_active_downloads()[0]
    progress = progress_manager.get_progress("qwen3-0.6b")
    assert task.status == "error"
    assert task.error == "mirror unavailable"
    assert progress is not None
    assert progress["status"] == "error"
    assert progress["error"] == "mirror unavailable"
