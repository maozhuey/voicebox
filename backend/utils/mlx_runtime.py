"""One process-wide execution lane for MLX/Metal native work."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")


class MLXRuntimeCoordinator:
    """Serialize MLX calls on one stream-owning OS thread.

    MLX binds Metal streams to the calling thread. Voicebox can otherwise run
    Qwen TTS, capture refinement, and transcription through separate executor
    threads. Those calls share the same Metal runtime and have caused native
    command-encoder crashes.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="voicebox-mlx-runtime",
        )
        self._thread_id: int | None = None
        self._thread_id_lock = threading.Lock()

    def _invoke(self, func: Callable[..., T], args: tuple[Any, ...]) -> T:
        with self._thread_id_lock:
            self._thread_id = threading.get_ident()
        return func(*args)

    async def run(self, func: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(self._invoke, func, args))

    def run_sync(self, func: Callable[..., T], *args: Any) -> T:
        """Run lifecycle cleanup on the owner thread from synchronous APIs."""
        with self._thread_id_lock:
            on_runtime_thread = self._thread_id == threading.get_ident()
        if on_runtime_thread:
            return func(*args)
        return self._executor.submit(self._invoke, func, args).result()


_mlx_runtime = MLXRuntimeCoordinator()


def get_mlx_runtime() -> MLXRuntimeCoordinator:
    """Return the shared MLX execution lane for TTS, LLM, and STT."""
    return _mlx_runtime
