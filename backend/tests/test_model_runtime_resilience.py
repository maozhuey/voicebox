"""Regression coverage for GPU-first device selection and load timeouts."""

import asyncio
from unittest.mock import patch

import pytest

from backend.backends.base import get_torch_device


def test_torch_device_prefers_mps_when_available():
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=True
    ):
        assert get_torch_device() == "mps"


def test_torch_device_falls_back_to_cpu_without_accelerator():
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=False
    ):
        assert get_torch_device() == "cpu"


def test_torch_device_keeps_cuda_as_highest_priority():
    with patch("torch.cuda.is_available", return_value=True):
        assert get_torch_device() == "cuda"


def test_model_load_timeout_configuration_uses_a_positive_number(monkeypatch):
    from backend import config

    monkeypatch.setenv("VOICEBOX_MODEL_LOAD_TIMEOUT_SECONDS", "45.5")
    assert config.get_model_load_timeout_seconds() == 45.5

    monkeypatch.setenv("VOICEBOX_MODEL_LOAD_TIMEOUT_SECONDS", "0")
    assert config.get_model_load_timeout_seconds() == 180.0

    monkeypatch.setenv("VOICEBOX_MODEL_LOAD_TIMEOUT_SECONDS", "not-a-number")
    assert config.get_model_load_timeout_seconds() == 180.0


@pytest.mark.parametrize(
    ("backend_module", "backend_class", "args"),
    [
        ("qwen_custom_voice_backend", "QwenCustomVoiceBackend", ()),
        ("cosyvoice_backend", "CosyVoiceTTSBackend", ()),
        ("chatterbox_backend", "ChatterboxTTSBackend", ()),
        ("chatterbox_turbo_backend", "ChatterboxTurboTTSBackend", ()),
        ("hume_backend", "HumeTadaBackend", ()),
        ("kokoro_backend", "KokoroTTSBackend", ()),
        ("luxtts_backend", "LuxTTSBackend", ()),
        ("pytorch_backend", "PyTorchTTSBackend", ()),
    ],
)
def test_tts_backends_prefer_available_mps(monkeypatch, backend_module, backend_class, args):
    module = __import__(f"backend.backends.{backend_module}", fromlist=[backend_class])
    monkeypatch.setattr(module, "get_torch_device", lambda **_kwargs: "mps")

    backend = getattr(module, backend_class)(*args)

    assert backend._get_device() == "mps"


@pytest.mark.asyncio
async def test_load_engine_model_times_out_with_user_safe_error(monkeypatch):
    from backend.backends import ModelLoadTimeoutError, load_engine_model

    class SlowBackend:
        async def load_model_async(self, _size):
            await asyncio.sleep(1)

    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: SlowBackend())
    monkeypatch.setattr("backend.config.get_model_load_timeout_seconds", lambda: 0.01)

    with pytest.raises(ModelLoadTimeoutError, match="模型加载超时"):
        await load_engine_model("qwen_custom_voice", "1.7B")


@pytest.mark.asyncio
async def test_load_engine_model_retries_cpu_after_accelerator_failure(monkeypatch):
    from backend.backends import load_engine_model

    class FallbackBackend:
        device = "mps"

        def __init__(self):
            self.attempts = 0
            self.unloaded = False

        def _is_model_cached(self, _size):
            return False

        def _get_device(self):
            return self.device

        def unload_model(self):
            self.unloaded = True

        async def load_model_async(self, _size):
            self.attempts += 1
            if self.device == "mps":
                raise RuntimeError("unsupported MPS operator")

    backend = FallbackBackend()
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: backend)

    await load_engine_model("qwen_custom_voice", "1.7B")

    assert backend.unloaded
    assert backend.device == "cpu"
    assert backend.attempts == 2
