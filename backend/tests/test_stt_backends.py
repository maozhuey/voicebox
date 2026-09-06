"""Speech-to-text backend regression tests."""

import numpy as np
import pytest

from backend.backends import pytorch_backend


class _Inputs(dict):
    def to(self, _device):
        return self


class _PromptIds:
    def to(self, _device):
        return "prompt-ids"


class _Processor:
    def __init__(self):
        self.call_kwargs = None

    def __call__(self, _audio, **kwargs):
        self.call_kwargs = kwargs
        return _Inputs(input_features="features", attention_mask="mask")

    def get_prompt_ids(self, prompt, return_tensors):
        assert prompt == "OpenAI, GenAI.mil"
        assert return_tensors == "pt"
        return _PromptIds()

    def batch_decode(self, _ids, skip_special_tokens):
        assert skip_special_tokens is True
        return ["完整长音频转录"]


class _Model:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return {
            "sequences": "all-sequences",
            "segments": [[{"start": 0.0, "end": 31.0, "tokens": [1, 2, 3]}]],
        }


@pytest.mark.asyncio
async def test_pytorch_whisper_uses_long_form_generation_and_prompt(monkeypatch):
    """Audio after Whisper's 30-second window must not be silently truncated."""
    backend = pytorch_backend.PyTorchSTTBackend(model_size="large")
    backend.processor = _Processor()
    backend.model = _Model()
    backend.device = "cpu"

    async def already_loaded(_model_size=None):
        return None

    backend.load_model_async = already_loaded
    monkeypatch.setattr(
        pytorch_backend,
        "load_audio",
        lambda _path, sample_rate: (np.zeros(sample_rate * 31, dtype=np.float32), sample_rate),
    )

    result = await backend.transcribe(
        "long.wav",
        language="zh",
        model_size="large",
        initial_prompt="OpenAI, GenAI.mil",
    )

    assert result.text == "完整长音频转录"
    assert result.segments[0].end_ms == 31_000
    assert backend.processor.call_kwargs["truncation"] is False
    assert backend.processor.call_kwargs["return_attention_mask"] is True
    assert backend.model.generate_kwargs["return_timestamps"] is True
    assert backend.model.generate_kwargs["return_segments"] is True
    assert backend.model.generate_kwargs["language"] == "zh"
    assert backend.model.generate_kwargs["task"] == "transcribe"
    assert backend.model.generate_kwargs["prompt_ids"] == "prompt-ids"
