"""Contracts for timestamped Whisper transcription results."""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.backends import mlx_backend, pytorch_backend
from backend.transcription import TranscriptionResult, build_transcription_result, format_timestamped_transcript


def test_build_transcription_result_normalizes_and_formats_segments():
    result = build_transcription_result(
        "第二句 第一句",
        [
            {"start": 2.0004, "end": 3.25, "text": "第二句"},
            {"start": 0.1255, "end": 1.0054, "text": "  第一句  "},
            {"start": 1.1, "end": 1.2, "text": "  "},
        ],
    )

    assert isinstance(result, TranscriptionResult)
    assert [segment.model_dump() for segment in result.segments] == [
        {"start_ms": 126, "end_ms": 1005, "text": "第一句"},
        {"start_ms": 2000, "end_ms": 3250, "text": "第二句"},
    ]
    assert result.timestamped_text == ("[00:00:00.126 --> 00:00:01.005] 第一句\n[00:00:02.000 --> 00:00:03.250] 第二句")


def test_timestamp_formatter_supports_hour_boundaries():
    result = build_transcription_result(
        "跨小时",
        [{"start": 3599.999, "end": 3601.004, "text": "跨小时"}],
    )

    assert format_timestamped_transcript(result.segments) == ("[00:59:59.999 --> 01:00:01.004] 跨小时")


@pytest.mark.parametrize(
    "segment",
    [
        {"start": -0.1, "end": 1.0, "text": "负数"},
        {"start": 2.0, "end": 1.0, "text": "倒序"},
        {"start": float("nan"), "end": 1.0, "text": "非有限"},
    ],
)
def test_invalid_timestamp_boundaries_are_rejected(segment):
    with pytest.raises(ValueError, match="timestamp"):
        build_transcription_result("文本", [segment])


def test_no_non_empty_timestamp_segments_is_rejected():
    with pytest.raises(ValueError, match="timestamped segments"):
        build_transcription_result("文本", [{"start": 0, "end": 1, "text": ""}])


def test_segments_are_clamped_to_real_audio_duration():
    result = build_transcription_result(
        "短音频",
        [
            {"start": 0, "end": 29.98, "text": "短音频"},
            {"start": 25, "end": 29.98, "text": "推理窗口填充"},
        ],
        audio_duration_ms=4020,
    )

    assert [segment.model_dump() for segment in result.segments] == [
        {"start_ms": 0, "end_ms": 4020, "text": "短音频"},
    ]


@pytest.mark.asyncio
async def test_mlx_backend_returns_original_segments(monkeypatch):
    backend = mlx_backend.MLXSTTBackend(model_size="large")
    backend.model = SimpleNamespace(
        generate=lambda *_args, **_kwargs: SimpleNamespace(
            text="第一句 第二句",
            segments=[
                {"start": 0.0, "end": 1.25, "text": "第一句"},
                {"start": 1.25, "end": 2.5, "text": "第二句"},
            ],
        )
    )

    async def already_loaded(_model_size=None):
        return None

    backend.load_model_async = already_loaded
    monkeypatch.setattr(
        mlx_backend,
        "load_audio",
        lambda _path, sample_rate: (np.zeros(sample_rate * 3, dtype=np.float32), sample_rate),
    )
    result = await backend.transcribe("sample.wav", language="zh", model_size="large")

    assert result.text == "第一句 第二句"
    assert [segment.model_dump() for segment in result.segments] == [
        {"start_ms": 0, "end_ms": 1250, "text": "第一句"},
        {"start_ms": 1250, "end_ms": 2500, "text": "第二句"},
    ]


class _Inputs(dict):
    def to(self, _device):
        return self


class _Processor:
    def __call__(self, _audio, **_kwargs):
        return _Inputs(input_features="features", attention_mask="mask")

    def batch_decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is True
        if ids == "all-sequences":
            return ["第一句 第二句"]
        token = ids[0][0]
        return [{11: "第一句", 22: "第二句"}[token]]


class _Model:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return {
            "sequences": "all-sequences",
            "segments": [
                [
                    {"start": 0.0, "end": 1.25, "tokens": [11]},
                    {"start": 1.25, "end": 2.5, "tokens": [22]},
                ]
            ],
        }


@pytest.mark.asyncio
async def test_pytorch_backend_returns_original_segments(monkeypatch):
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
        lambda _path, sample_rate: (np.zeros(sample_rate * 3, dtype=np.float32), sample_rate),
    )

    result = await backend.transcribe("sample.wav", language="zh", model_size="large")

    assert result.text == "第一句 第二句"
    assert [segment.model_dump() for segment in result.segments] == [
        {"start_ms": 0, "end_ms": 1250, "text": "第一句"},
        {"start_ms": 1250, "end_ms": 2500, "text": "第二句"},
    ]
    assert backend.model.generate_kwargs["return_timestamps"] is True
    assert backend.model.generate_kwargs["return_segments"] is True
