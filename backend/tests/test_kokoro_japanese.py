"""Regression coverage for Kokoro Japanese voice prerequisites."""

from backend.backends.kokoro_backend import KokoroTTSBackend


def test_kokoro_japanese_tokenizer_uses_bundled_unidic_lite(monkeypatch):
    import misaki.cutlet

    with monkeypatch.context() as patch:
        patch.setattr(misaki.cutlet, "Tagger", misaki.cutlet.Tagger)
        patch.setattr(misaki.cutlet, "_voicebox_uses_unidic_lite", False, raising=False)
        KokoroTTSBackend._configure_japanese_tokenizer()

        tokens = list(misaki.cutlet.Tagger()("こんにちは"))

    assert tokens
