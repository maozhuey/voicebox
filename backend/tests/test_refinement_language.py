"""Regression coverage for language-preserving capture refinement."""
# ruff: noqa: RUF001 -- Chinese inputs and expected outputs verify the user-visible guarantee.

from types import SimpleNamespace

import pytest

from backend.services import captures, refinement


def test_auto_detects_chinese_transcript_for_refinement():
    """Chinese STT text must get a Chinese output constraint in auto mode."""
    assert refinement.infer_refinement_language("把接口地址发给我，我来调用 JSON 接口。") == "zh"


def test_chinese_prompt_explicitly_forbids_translation():
    """The model prompt makes Chinese preservation an unambiguous requirement."""
    prompt = refinement.build_refinement_prompt(refinement.RefinementFlags(), language="zh")

    assert "Chinese" in prompt
    assert "Never translate" in prompt
    assert "same language" in prompt


def test_chinese_prompt_requires_punctuation_for_unpunctuated_transcripts():
    """Long Chinese ASR text cannot be returned unchanged without punctuation."""
    prompt = refinement.build_refinement_prompt(refinement.RefinementFlags(), language="zh")

    assert "must add punctuation" in prompt
    assert "must not return it unchanged" in prompt


def test_chinese_refinement_uses_only_chinese_examples():
    """English demonstrations must not outweigh the Chinese output constraint."""
    examples = refinement.build_refinement_examples("zh")

    assert len(examples) >= 2
    assert all(any("\u4e00" <= char <= "\u9fff" for char in input_text) for input_text, _ in examples)


def test_chinese_punctuation_fallback_preserves_words_and_adds_boundaries():
    """A no-op long-form result is still readable when the small model skips punctuation."""
    raw = "用户点按钮之后给我的接口发请求把数据带上就行如果这个事件有用我们再监听没用就放过就行了"

    refined = refinement.add_fallback_chinese_punctuation(raw)

    assert refined.replace("，", "").replace("。", "") == raw
    assert "，" in refined
    assert refined.endswith("。")


@pytest.mark.asyncio
async def test_refine_transcript_falls_back_when_chinese_model_omits_punctuation(monkeypatch):
    """The user never receives an unchanged long Chinese run solely due to model failure."""

    class FakeBackend:
        model_size = "1.7B"

        async def generate(self, **_kwargs):
            return "用户点按钮之后给我的接口发请求把数据带上就行如果这个事件有用我们再监听没用就放过就行了"

    monkeypatch.setattr(refinement.llm_service, "get_llm_model", lambda: FakeBackend())

    text, _ = await refinement.refine_transcript(
        "用户点按钮之后给我的接口发请求把数据带上就行如果这个事件有用我们再监听没用就放过就行了",
        refinement.RefinementFlags(),
    )

    assert "，" in text
    assert text.endswith("。")


@pytest.mark.asyncio
async def test_refine_transcript_uses_chinese_constraint_and_example(monkeypatch):
    """Chinese text in auto mode receives a Chinese few-shot refinement example."""
    calls: list[dict] = []

    class FakeBackend:
        model_size = "1.7B"

        async def generate(self, **kwargs):
            calls.append(kwargs)
            return "请把 JSON 接口地址发给我。"

    monkeypatch.setattr(refinement.llm_service, "get_llm_model", lambda: FakeBackend())

    text, model_size = await refinement.refine_transcript(
        "请把 json 接口地址发给我",
        refinement.RefinementFlags(),
    )

    assert text == "请把 JSON 接口地址发给我。"
    assert model_size == "1.7B"
    assert "Chinese" in calls[0]["system"]
    assert any("接口" in input_text for input_text, _ in calls[0]["examples"])


@pytest.mark.asyncio
async def test_refine_capture_passes_saved_capture_language_to_refinement(monkeypatch):
    """An explicit Chinese capture keeps its language through the service layer."""
    row = SimpleNamespace(
        id="capture-1",
        transcript_raw="把接口地址发给我",
        language="zh",
        transcript_refined=None,
        llm_model=None,
        refinement_flags=None,
    )
    received: dict[str, object] = {}

    class FakeQuery:
        def filter(self, *_args):
            return self

        def first(self):
            return row

    class FakeDb:
        def query(self, _model):
            return FakeQuery()

        def commit(self):
            return None

        def refresh(self, _row):
            return None

    async def fake_refine(transcript, flags, model_size=None, language=None):
        received.update(
            transcript=transcript,
            flags=flags,
            model_size=model_size,
            language=language,
        )
        return "把接口地址发给我。", "1.7B"

    monkeypatch.setattr(captures, "refine_transcript", fake_refine)
    monkeypatch.setattr(captures, "_to_response", lambda value: value)

    result = await captures.refine_capture(
        "capture-1",
        refinement.RefinementFlags(),
        model_size="1.7B",
        db=FakeDb(),
    )

    assert received["language"] == "zh"
    assert result.transcript_refined == "把接口地址发给我。"
