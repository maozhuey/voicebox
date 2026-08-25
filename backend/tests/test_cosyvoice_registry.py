"""Regression coverage for CosyVoice 3 registry wiring.

These tests deliberately do not download the 5.4 GB model. Model load and
generation are covered by the local integration checklist because the upstream
runtime is source-distributed rather than a lightweight pip package.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from pydantic import ValidationError

from backend.backends import get_model_config, get_tts_model_configs
from backend.backends.cosyvoice_backend import (
    CosyVoiceTTSBackend,
    _install_qwen2_compat,
    _prepare_imports,
    _validate_generated_audio,
)
from backend.models import GenerationRequest
from backend.services.profiles import CLONING_ENGINES, validate_profile_engine


def test_cosyvoice3_variants_are_registered_with_rl_as_default_candidate():
    configs = {config.model_name: config for config in get_tts_model_configs()}

    rl = configs["cosyvoice3-0.5b-rl"]
    base = configs["cosyvoice3-0.5b"]

    assert rl.engine == base.engine == "cosyvoice"
    assert rl.model_size == "rl"
    assert base.model_size == "base"
    assert rl.supports_instruct is base.supports_instruct is True
    # Upstream publishes the RL checkpoint as llm.rl.pt inside this shared
    # public repository; the backend maps it to llm.pt in an isolated view.
    assert rl.hf_repo_id == base.hf_repo_id
    assert "zh" in rl.languages


def test_cosyvoice3_model_names_resolve_from_registry():
    assert get_model_config("cosyvoice3-0.5b-rl").model_size == "rl"
    assert get_model_config("cosyvoice3-0.5b").model_size == "base"


def test_generation_request_accepts_cosyvoice_variants():
    request = GenerationRequest(profile_id="profile-1", text="测试", engine="cosyvoice", model_size="rl")

    assert request.engine == "cosyvoice"
    assert request.model_size == "rl"


def test_generation_request_rejects_unknown_cosyvoice_variant():
    with pytest.raises(ValidationError):
        GenerationRequest(profile_id="profile-1", text="测试", engine="cosyvoice", model_size="v3")


def test_generation_request_rejects_unknown_dialect():
    with pytest.raises(ValidationError):
        GenerationRequest(profile_id="profile-1", text="测试", engine="cosyvoice", dialect="cantonese")


def test_generation_request_rejects_unknown_cosyvoice_mode():
    with pytest.raises(ValidationError):
        GenerationRequest(
            profile_id="profile-1",
            text="测试",
            engine="cosyvoice",
            cosyvoice_mode="voiceprint",
        )


def test_cosyvoice_accepts_cloned_voice_profiles():
    """CosyVoice uses reference audio for zero-shot cloning like the other cloning engines."""
    assert "cosyvoice" in CLONING_ENGINES
    validate_profile_engine(SimpleNamespace(voice_type="cloned"), "cosyvoice")


def test_rl_variant_creates_an_isolated_checkpoint_view(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "llm.pt").write_bytes(b"base")
    rl_checkpoint = source / "llm.rl.pt"
    rl_checkpoint.write_bytes(b"rl")
    (source / "cosyvoice3.yaml").write_text("sample_rate: 24000")

    backend = CosyVoiceTTSBackend()
    view_dir = backend._create_rl_model_view(source)
    try:
        assert (view_dir / "llm.pt").resolve() == rl_checkpoint.resolve()
        assert (source / "llm.pt").read_bytes() == b"base"
    finally:
        backend.unload_model()


def test_prepare_imports_exposes_training_free_yaml_symbols():
    """CosyVoice YAML resolution needs parent-package attributes, not just modules."""
    _prepare_imports()

    dataset = sys.modules["cosyvoice.dataset"]
    processor = sys.modules["cosyvoice.dataset.processor"]
    assert dataset.processor is processor
    assert callable(processor.parquet_opener)

    import cosyvoice
    import matcha

    assert cosyvoice.dataset is dataset
    assert matcha.utils is sys.modules["matcha.utils"]


def test_cosyvoice_reference_audio_loader_does_not_require_torchcodec(tmp_path):
    reference = tmp_path / "reference.wav"
    samples = np.array([[0.25, -0.25], [0.5, -0.5]], dtype=np.float32)
    sf.write(reference, samples, 16_000, subtype="FLOAT")

    _prepare_imports()
    import torchaudio

    waveform, sample_rate = torchaudio.load(reference, channels_first=True)

    assert sample_rate == 16_000
    assert tuple(waveform.shape) == (2, 2)
    np.testing.assert_allclose(waveform.numpy(), samples.T)


def test_cosyvoice3_minimum_length_masks_every_stop_token():
    """RL must not terminate on any CosyVoice 3 control token before min length."""
    _prepare_imports()
    from cosyvoice.llm.llm import CosyVoice3LM

    model = object.__new__(CosyVoice3LM)
    model.speech_token_size = 4
    model.stop_token_ids = [4, 5, 6]
    model.sampling = lambda scores, _decoded, _sampling: int(torch.argmax(scores))
    scores = torch.tensor([9.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0])

    assert model.sampling_ids(scores, [], 25, ignore_eos=True) == 0


def test_cosyvoice3_installs_checkpoint_compatible_qwen2_runtime():
    """CosyVoice must use the Qwen2 implementation recorded by its weights."""
    _prepare_imports()
    compat_class = _install_qwen2_compat()

    import cosyvoice.llm.llm as cosyvoice_llm

    assert compat_class.__module__.endswith("modeling_qwen2_440")
    assert cosyvoice_llm.Qwen2ForCausalLM is compat_class


def test_cosyvoice_rejects_known_early_stop_audio():
    with pytest.raises(RuntimeError, match="stopped before producing usable speech"):
        _validate_generated_audio(
            np.ones(960, dtype=np.float32),
            24_000,
            "这是一段本应该生成正常语音的文本。",
        )


def test_cosyvoice_rejects_broadband_codec_noise():
    noise = np.tile(np.array([-0.5, 0.5], dtype=np.float32), 24_000)

    with pytest.raises(RuntimeError, match="unintelligible codec noise"):
        _validate_generated_audio(
            noise,
            24_000,
            "这是一段本应该生成清晰人声的文本。",
        )


@pytest.mark.asyncio
async def test_cosyvoice_reference_mode_uses_zero_shot_prompt(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(16_000, dtype=np.float32), 16_000)
    captured: dict[str, str] = {}

    class FakeCosyVoice:
        sample_rate = 24_000

        def inference_zero_shot(self, **kwargs):
            captured["prompt_text"] = kwargs["prompt_text"]
            captured["prompt_wav"] = kwargs["prompt_wav"]
            yield {"tts_speech": torch.zeros(1, 6_000)}

        def inference_instruct2(self, **_kwargs):
            raise AssertionError("reference mode must not use instruct2")

    backend = CosyVoiceTTSBackend()
    backend.model = FakeCosyVoice()
    backend._current_model_size = "rl"

    async def keep_fake_model(_model_size):
        return None

    monkeypatch.setattr(backend, "load_model", keep_fake_model)

    await backend.generate(
        "测试",
        {"ref_audio": str(reference), "ref_text": "今儿天气不错。"},
        language="zh",
        instruct=None,
    )

    assert captured["prompt_text"] == "You are a helpful assistant.<|endofprompt|>今儿天气不错。"
    assert captured["prompt_wav"] == str(reference)


@pytest.mark.asyncio
async def test_cosyvoice_dialect_instruction_stays_outside_spoken_text(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(16_000, dtype=np.float32), 16_000)
    captured: dict[str, str] = {}

    class FakeCosyVoice:
        sample_rate = 24_000

        def inference_instruct2(self, **kwargs):
            captured["instruct_text"] = kwargs["instruct_text"]
            yield {"tts_speech": torch.zeros(1, 6_000)}

    backend = CosyVoiceTTSBackend()
    backend.model = FakeCosyVoice()
    backend._current_model_size = "rl"

    async def keep_fake_model(_model_size):
        return None

    monkeypatch.setattr(backend, "load_model", keep_fake_model)

    await backend.generate(
        "测试",
        {"ref_audio": str(reference), "ref_text": "参考文本。"},
        language="zh",
        instruct="请用河南话表达。",
    )

    assert captured["instruct_text"] == (
        "You are a helpful assistant. 请用河南话表达。<|endofprompt|>"
    )
    assert captured["instruct_text"].count("<|endofprompt|>") == 1
