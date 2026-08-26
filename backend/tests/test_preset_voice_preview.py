"""Tests for the non-persistent built-in voice preview path."""

import pytest

from backend.services import profiles, tts


def test_preset_voice_preview_uses_the_voice_configured_language():
    assert profiles._get_preset_voice_language("kokoro", "af_alloy") == "en"
    assert profiles._get_preset_voice_language("kokoro", "jf_alpha") == "ja"
    assert profiles._get_preset_voice_language("qwen_custom_voice", "Ryan") == "en"
    assert profiles._get_preset_voice_language("qwen_custom_voice", "Aiden") == "zh"
    assert profiles._get_preset_voice_language("qwen_custom_voice", "Ono_Anna") == "zh"
    assert profiles._get_preset_voice_language("qwen_custom_voice", "Sohee") == "zh"


@pytest.mark.asyncio
async def test_preset_voice_preview_synthesizes_without_a_profile(tmp_path, monkeypatch):
    calls = []

    class FakeBackend:
        async def generate(self, text, voice_prompt, *, language):
            calls.append((text, voice_prompt, language))
            return [0.0], 24_000

    async def fake_ensure_model_cached(engine, model_size):
        calls.append(("cached", engine, model_size))

    async def fake_load_model(engine, model_size):
        calls.append(("loaded", engine, model_size))

    monkeypatch.setattr(profiles, "_get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("backend.backends.ensure_model_cached_or_raise", fake_ensure_model_cached)
    monkeypatch.setattr("backend.backends.load_engine_model", fake_load_model)
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: FakeBackend())
    monkeypatch.setattr(tts, "audio_to_wav_bytes", lambda audio, sample_rate: b"preview-wav")

    result = await profiles.generate_preset_voice_preview("kokoro", "zf_xiaobei")

    assert result == b"preview-wav"
    assert calls == [
        ("cached", "kokoro", "default"),
        ("loaded", "kokoro", "default"),
        (
            "你好，这是我的声音预览。",  # noqa: RUF001
            {
                "voice_type": "preset",
                "preset_engine": "kokoro",
                "preset_voice_id": "zf_xiaobei",
            },
            "zh",
        ),
    ]


@pytest.mark.asyncio
async def test_preset_voice_preview_reuses_local_wav_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "preset_previews" / "kokoro" / "zf_xiaobei-v2.wav"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"cached-preview-wav")

    monkeypatch.setattr(profiles, "_get_cache_dir", lambda: tmp_path)

    async def should_not_generate(*_args, **_kwargs):
        raise AssertionError("Cached preview must not load a model or synthesize again")

    monkeypatch.setattr("backend.backends.ensure_model_cached_or_raise", should_not_generate)

    result = await profiles.generate_preset_voice_preview("kokoro", "zf_xiaobei")

    assert result == b"cached-preview-wav"


@pytest.mark.asyncio
async def test_preset_voice_preview_writes_generated_wav_to_local_cache(tmp_path, monkeypatch):
    class FakeBackend:
        async def generate(self, *_args, **_kwargs):
            return [0.0], 24_000

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(profiles, "_get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("backend.backends.ensure_model_cached_or_raise", noop)
    monkeypatch.setattr("backend.backends.load_engine_model", noop)
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: FakeBackend())
    monkeypatch.setattr(tts, "audio_to_wav_bytes", lambda *_args: b"generated-preview-wav")

    result = await profiles.generate_preset_voice_preview("kokoro", "zf_xiaobei")

    assert result == b"generated-preview-wav"
    assert (tmp_path / "preset_previews" / "kokoro" / "zf_xiaobei-v2.wav").read_bytes() == result


@pytest.mark.asyncio
async def test_preset_voice_preview_rejects_unknown_voice():
    with pytest.raises(ValueError, match="not valid"):
        await profiles.generate_preset_voice_preview("kokoro", "not-a-voice")
