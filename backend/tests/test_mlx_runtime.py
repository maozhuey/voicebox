"""Regression coverage for one shared MLX/Metal execution lane."""

import asyncio
import threading
import time

import pytest

from backend.backends.mlx_backend import MLXSTTBackend, MLXTTSBackend
from backend.backends.qwen_llm_backend import MLXQwenLLMBackend
from backend.utils.mlx_runtime import MLXRuntimeCoordinator


@pytest.mark.asyncio
async def test_mlx_runtime_serializes_concurrent_native_callbacks_on_one_thread():
    runtime = MLXRuntimeCoordinator()
    running = 0
    max_running = 0
    thread_ids: list[int] = []
    lock = threading.Lock()

    def native_callback() -> None:
        nonlocal running, max_running
        with lock:
            running += 1
            max_running = max(max_running, running)
            thread_ids.append(threading.get_ident())
        time.sleep(0.02)
        with lock:
            running -= 1

    await asyncio.gather(*(runtime.run(native_callback) for _ in range(4)))

    assert max_running == 1
    assert len(set(thread_ids)) == 1


def test_tts_llm_and_stt_backends_share_the_process_wide_mlx_runtime():
    """Model families must not create independent Metal executor threads."""
    tts = MLXTTSBackend()
    llm = MLXQwenLLMBackend()
    stt = MLXSTTBackend()

    assert tts._mlx_runtime is llm._mlx_runtime is stt._mlx_runtime
