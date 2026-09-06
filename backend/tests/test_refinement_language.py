"""Regression coverage for language-preserving capture refinement."""
# ruff: noqa: RUF001 -- Chinese inputs and expected outputs verify the user-visible guarantee.

from types import SimpleNamespace

import pytest

from backend.services import captures, refinement


def test_auto_detects_chinese_transcript_for_refinement():
    """Chinese STT text must get a Chinese output constraint in auto mode."""
    assert refinement.infer_refinement_language("把接口地址发给我，我来调用 JSON 接口。") == "zh"


def test_refinement_language_follows_current_transcript_over_stale_hint():
    """The text produced by the latest STT pass is the refinement source of truth."""
    assert refinement.resolve_refinement_language("请保留 OpenAI 这个专有名词。", "en") == "zh"
    assert refinement.resolve_refinement_language("Keep the OpenAI product name.", "zh") is None
    assert refinement.resolve_refinement_language("Keep the OpenAI product name.", "en") == "en"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "stale_hint", "required_rule", "forbidden_rule"),
    [
        ("请保留 OpenAI 这个专有名词。", "en", "transcript language is Chinese", "transcript language is English"),
        (
            "Keep the OpenAI product name.",
            "zh",
            "same language or languages as the source transcript",
            "transcript language is Chinese",
        ),
    ],
)
async def test_refine_transcript_builds_prompt_from_current_transcript_language(
    monkeypatch,
    transcript,
    stale_hint,
    required_rule,
    forbidden_rule,
):
    """The LLM instruction itself must follow the current STT text."""
    calls: list[dict] = []

    class FakeBackend:
        model_size = "1.7B"

        async def load_model(self, _model_size):
            return None

        async def generate(self, **kwargs):
            calls.append(kwargs)
            return transcript

    monkeypatch.setattr(refinement.llm_service, "get_llm_model", lambda: FakeBackend())

    result, _ = await refinement.refine_transcript(
        transcript,
        refinement.RefinementFlags(),
        language=stale_hint,
    )

    assert result == transcript
    assert required_rule in calls[0]["system"]
    assert forbidden_rule not in calls[0]["system"]


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
    assert "full-width Chinese punctuation" in prompt
    assert "GenAI.mil" in prompt


@pytest.mark.parametrize(
    ("language", "candidate", "expected"),
    [
        (
            "zh",
            "请保留 GenAI.mil, 版本 1.2, 共 1,000 次!",
            "请保留 GenAI.mil， 版本 1.2， 共 1,000 次！",
        ),
        ("en", "Keep OpenAI，DeepMind。", "Keep OpenAI,DeepMind."),
        ("ja", "OpenAI,次に進む.", "OpenAI、次に進む。"),
    ],
)
def test_refinement_uses_language_specific_punctuation_without_breaking_terms(
    language,
    candidate,
    expected,
):
    assert refinement.normalize_refinement_punctuation(candidate, language) == expected


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


def test_long_refinement_cannot_drop_repeated_but_intentional_content():
    """A small LLM may not treat a headline plus bullet recap as duplicate noise."""
    source = (
        "OpenAI买下数万台Mac训Agent苹果高配售罄发货排到16到18周"
        "DeepMind净流失80人5家实验室确认流动274次"
        "OpenAI买数万台Mac训Agent高配售罄"
        "5实验室人才流动DeepMind净流失80人"
    )
    shortened = "OpenAI买数万台Mac训Agent，高配售罄。DeepMind净流失80人。"

    guarded = refinement.apply_refinement_quality_guard(source, shortened, "zh")

    assert guarded.replace("，", "").replace("。", "") == source


def test_long_chinese_refinement_cannot_translate_or_drop_technical_terms():
    source = (
        "OpenAI买下数万台Mac训计算机使用Agent苹果高配售罄发货排到16到18周"
        "DeepMind净流失80人五家实验室确认流动274次"
    )

    guarded = refinement.apply_refinement_quality_guard(
        source,
        "OpenAI buys many Mac computers. DeepMind lost 80 people.",
        "zh",
    )

    assert "买下数万台" in guarded
    assert "274" in guarded


def test_safe_long_refinement_with_punctuation_is_kept():
    source = "这是一段需要恢复标点的中文听写内容" * 6
    candidate = f"{source[:40]}，{source[40:]}。"

    assert refinement.apply_refinement_quality_guard(source, candidate, "zh") == candidate


@pytest.mark.asyncio
async def test_refine_transcript_falls_back_when_chinese_model_omits_punctuation(monkeypatch):
    """The user never receives an unchanged long Chinese run solely due to model failure."""

    class FakeBackend:
        model_size = "1.7B"

        async def load_model(self, _model_size):
            return None

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

        async def load_model(self, _model_size):
            return None

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


@pytest.mark.asyncio
async def test_manual_refine_uses_raw_language_and_matching_punctuation(monkeypatch):
    """The Captures page's manual Refine/Re-refine action preserves raw language."""
    row = SimpleNamespace(
        id="capture-manual",
        transcript_raw="请保留 OpenAI 然后继续",
        # Simulate an old or stale STT hint; transcript_raw remains authoritative.
        language="en",
        transcript_refined=None,
        llm_model=None,
        refinement_flags=None,
    )
    calls: list[dict] = []

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

    class FakeBackend:
        model_size = "1.7B"

        async def load_model(self, _model_size):
            return None

        async def generate(self, **kwargs):
            calls.append(kwargs)
            return "请保留 OpenAI,然后继续!"

    monkeypatch.setattr(refinement.llm_service, "get_llm_model", lambda: FakeBackend())
    monkeypatch.setattr(captures, "_to_response", lambda value: value)

    result = await captures.refine_capture(
        "capture-manual",
        refinement.RefinementFlags(),
        model_size="1.7B",
        db=FakeDb(),
    )

    assert result.transcript_refined == "请保留 OpenAI，然后继续！"
    assert "transcript language is Chinese" in calls[0]["system"]
    assert "full-width Chinese punctuation" in calls[0]["system"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["model_load", "llm_generate"])
async def test_refine_transcript_tags_model_failures_without_exposing_original_error(monkeypatch, stage):
    """A failed optional refinement must be diagnosable without leaking text."""

    class FakeBackend:
        model_size = "1.7B"

        async def load_model(self, _model_size):
            if stage == "model_load":
                raise RuntimeError("sensitive transcript /private/capture.wav")

        async def generate(self, **_kwargs):
            if stage == "llm_generate":
                raise RuntimeError("sensitive transcript /private/capture.wav")
            return "不会执行到这里"

    monkeypatch.setattr(refinement.llm_service, "get_llm_model", lambda: FakeBackend())

    with pytest.raises(refinement.CaptureRefinementError) as error:
        await refinement.refine_transcript("敏感听写内容", refinement.RefinementFlags())

    assert error.value.stage == stage
    assert "sensitive transcript" not in str(error.value)
    assert "/private" not in str(error.value)


@pytest.mark.asyncio
async def test_refine_capture_rolls_back_failed_persistence(monkeypatch):
    """A manual re-refine cannot overwrite the previously usable result."""
    row = SimpleNamespace(
        id="capture-persist",
        transcript_raw="原始转录",
        language="zh",
        transcript_refined="已有精修",
        llm_model="0.6B",
        refinement_flags='{"smart_cleanup": false}',
    )

    class FakeQuery:
        def filter(self, *_args):
            return self

        def first(self):
            return row

    class FailingDb:
        rolled_back = False

        def query(self, _model):
            return FakeQuery()

        def commit(self):
            raise RuntimeError("write failed for sensitive transcript")

        def refresh(self, _row):
            return None

        def rollback(self):
            self.rolled_back = True

    async def fake_refine(*_args, **_kwargs):
        return "新的精修", "1.7B"

    db = FailingDb()
    monkeypatch.setattr(captures, "refine_transcript", fake_refine)

    with pytest.raises(refinement.CaptureRefinementError) as error:
        await captures.refine_capture("capture-persist", refinement.RefinementFlags(), "1.7B", db)

    assert error.value.stage == "persist"
    assert db.rolled_back is True
    assert row.transcript_refined == "已有精修"
    assert row.llm_model == "0.6B"
    assert row.refinement_flags == '{"smart_cleanup": false}'


@pytest.mark.asyncio
async def test_auto_retranscription_clears_stale_capture_language(monkeypatch, tmp_path):
    """Switching STT back to auto cannot retain a language from an older pass."""
    audio_path = tmp_path / "capture.wav"
    audio_path.write_bytes(b"wav")
    row = SimpleNamespace(
        id="capture-1",
        audio_path="captures/capture.wav",
        language="zh",
        transcript_raw="旧的中文转录",
        transcript_refined="旧的中文转录。",
        stt_model="large",
        llm_model="1.7B",
        refinement_flags="{}",
    )

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

    class FakeWhisper:
        model_size = "large"

        async def transcribe(self, _path, language, _model_size, initial_prompt=None):
            assert language is None
            assert initial_prompt == "OpenAI"
            from backend.transcription import build_transcription_result

            return build_transcription_result(
                "Keep the OpenAI product name.",
                [{"start": 0.0, "end": 1.0, "text": "Keep the OpenAI product name."}],
            )

    monkeypatch.setattr(captures.config, "resolve_storage_path", lambda _path: audio_path)
    monkeypatch.setattr(captures, "get_whisper_model", lambda: FakeWhisper())
    monkeypatch.setattr(captures, "_to_response", lambda value: value)

    result = await captures.retranscribe_capture(
        "capture-1",
        stt_model="large",
        language=None,
        initial_prompt="OpenAI",
        db=FakeDb(),
    )

    assert result.language is None
    assert result.transcript_raw == "Keep the OpenAI product name."
    assert result.transcript_refined is None
