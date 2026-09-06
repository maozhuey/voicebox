"""Regression coverage for MLX TTS GPU stream thread affinity."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from backend.backends.mlx_backend import MLXTTSBackend


@pytest.mark.asyncio
async def test_mlx_tts_load_and_generate_share_a_dedicated_thread(monkeypatch):
    """Loading and inference must not be split across default worker threads."""
    backend = MLXTTSBackend()
    thread_ids: list[int] = []

    class FakeModel:
        def generate(self, _text, *, lang_code):
            assert lang_code == "chinese"
            thread_ids.append(threading.get_ident())
            yield SimpleNamespace(audio=np.array([0.1], dtype=np.float32), sample_rate=24_000)

    def fake_load(model_size: str) -> None:
        assert model_size == "1.7B"
        thread_ids.append(threading.get_ident())
        backend.model = FakeModel()
        backend._current_model_size = model_size

    monkeypatch.setattr(backend, "_load_model_sync", fake_load)

    # A default executor may choose a different worker for the second call;
    # MLX streams are thread-local, so this backend must own its executor.
    with patch(
        "backend.backends.mlx_backend.asyncio.to_thread",
        side_effect=AssertionError("MLX TTS must not use the default executor"),
    ):
        await backend.load_model_async("1.7B")
        audio, sample_rate = await backend.generate("测试", {}, language="zh")

    assert sample_rate == 24_000
    assert audio.tolist() == pytest.approx([0.1])
    assert len(thread_ids) == 2
    assert thread_ids[0] == thread_ids[1]
