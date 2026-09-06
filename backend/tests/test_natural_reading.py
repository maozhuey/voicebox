"""Natural-reading prosody planning and audio assembly tests."""

# Chinese full-width punctuation is intentional: recognizing it is the behavior
# these regression cases verify.
# ruff: noqa: RUF001

import numpy as np
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from backend.database import Base, Generation as DBGeneration, Story, StoryItem, VoiceProfile
from backend.database.migrations import run_migrations
from backend.models import GenerationRequest, GenerationSettingsUpdate
from backend.services import history, stories
from backend.utils.chunked_tts import (
    generate_chunked,
    merge_short_prosody_chunks,
    plan_semantic_boundary_chunks,
    plan_natural_reading,
)

SAMPLE_RATE = 1000


def test_planner_preserves_line_and_paragraph_breaks():
    chunks = plan_natural_reading(
        "第一行。\n第二行。\n\n最后一段！",
        max_chars=70,
    )

    assert [(chunk.text, chunk.pause_after_ms) for chunk in chunks] == [
        ("第一行。", 650),
        ("第二行。", 1000),
        ("最后一段！", 0),
    ]


def test_planner_preserves_blank_lines_that_contain_spaces():
    chunks = plan_natural_reading("第一段。\n   \n第二段。", max_chars=70)

    assert [(chunk.text, chunk.pause_after_ms) for chunk in chunks] == [
        ("第一段。", 1000),
        ("第二段。", 0),
    ]


def test_planner_uses_sentence_pauses_without_newlines():
    chunks = plan_natural_reading("第一句。第二句？最后一句！", max_chars=70)

    assert [(chunk.text, chunk.pause_after_ms) for chunk in chunks] == [
        ("第一句。", 450),
        ("第二句？", 450),
        ("最后一句！", 0),
    ]


def test_planner_splits_long_sentence_at_clause_boundary():
    text = "这是第一部分内容，需要先介绍清楚，这是第二部分内容，也需要保持自然停顿。"

    chunks = plan_natural_reading(text, max_chars=24)

    assert all(len(chunk.text) <= 24 for chunk in chunks)
    assert any(chunk.pause_after_ms == 220 for chunk in chunks[:-1])
    assert "".join(chunk.text for chunk in chunks) == text


def test_planner_supports_explicit_pause_tag():
    chunks = plan_natural_reading("这是重点。[停顿0.8秒]请仔细听。", max_chars=70)

    assert [(chunk.text, chunk.pause_after_ms) for chunk in chunks] == [
        ("这是重点。", 800),
        ("请仔细听。", 0),
    ]


def test_planner_treats_semicolon_as_a_breath_boundary():
    chunks = plan_natural_reading(
        "请把新录音和文案发给我；或者直接告诉我第二段的准确内容。",
        max_chars=70,
    )

    assert [(chunk.text, chunk.pause_after_ms) for chunk in chunks] == [
        ("请把新录音和文案发给我；", 220),
        ("或者直接告诉我第二段的准确内容。", 0),
    ]


def test_planner_splits_long_comma_clause_even_below_hard_limit():
    chunks = plan_natural_reading(
        "这是一段需要稳定朗读的较长内容，需要在逗号处自然换气后再继续完成后半句。",
        max_chars=70,
    )

    assert [(chunk.text, chunk.pause_after_ms) for chunk in chunks] == [
        ("这是一段需要稳定朗读的较长内容，", 220),
        ("需要在逗号处自然换气后再继续完成后半句。", 0),
    ]


def test_cosyvoice_style_merging_avoids_tiny_natural_reading_units():
    text = (
        "今天介绍油蟠桃。\n\n它的果形很规整。\n\n果面比较光滑。\n\n"
        "果肉细脆多汁。\n\n甜味也很明显。\n\n成熟时间比较早。\n\n"
        "种植前建议先试种。\n\n表现稳定后再扩大面积。"
    )
    planned = plan_natural_reading(text, max_chars=100)

    merged = merge_short_prosody_chunks(planned, min_chars=45, max_chars=100)

    assert len(merged) < len(planned)
    assert "".join(chunk.text for chunk in merged) == "".join(chunk.text for chunk in planned)
    assert all(len(chunk.text) <= 100 for chunk in merged)
    assert all(len(chunk.text) >= 45 for chunk in merged[:-1])


def test_semantic_boundary_plan_never_uses_a_fixed_character_cut():
    unpunctuated = "无标点" * 200

    chunks = plan_semantic_boundary_chunks(unpunctuated, natural_reading=False)

    assert [(chunk.text, chunk.pause_after_ms) for chunk in chunks] == [(unpunctuated, 0)]


def test_semantic_boundary_plan_uses_only_sentence_semicolon_and_paragraph_boundaries():
    chunks = plan_semantic_boundary_chunks(
        "第一句，仍在同一句。第二句；第三段\n第四句",
        natural_reading=True,
    )

    assert [(chunk.text, chunk.pause_after_ms) for chunk in chunks] == [
        ("第一句，仍在同一句。", 450),
        ("第二句；", 220),
        ("第三段", 650),
        ("第四句", 0),
    ]


@pytest.mark.asyncio
async def test_natural_reading_inserts_real_silence_and_reuses_stable_seed():
    class FakeBackend:
        def __init__(self):
            self.calls = []

        async def generate(self, text, _voice_prompt, _language, seed, _instruct):
            self.calls.append((text, seed))
            return np.ones(100, dtype=np.float32), SAMPLE_RATE

    backend = FakeBackend()

    audio, sample_rate = await generate_chunked(
        backend,
        "第一句。第二句。",
        {},
        natural_reading=True,
        max_chunk_chars=100,
        crossfade_ms=50,
    )

    assert sample_rate == SAMPLE_RATE
    assert backend.calls[0][0] == "第一句。"
    assert backend.calls[1][0] == "第二句。"
    assert backend.calls[0][1] is not None
    assert backend.calls[0][1] == backend.calls[1][1]
    assert len(audio) == 650
    assert np.all(audio[:100] == 1)
    assert np.all(audio[100:550] == 0)
    assert np.all(audio[550:] == 1)


@pytest.mark.asyncio
async def test_separate_natural_generations_can_produce_new_takes():
    class FakeBackend:
        def __init__(self):
            self.seeds = []

        async def generate(self, _text, _voice_prompt, _language, seed, _instruct):
            self.seeds.append(seed)
            return np.ones(100, dtype=np.float32), SAMPLE_RATE

    backend = FakeBackend()
    await generate_chunked(backend, "短句。", {}, natural_reading=True)
    await generate_chunked(backend, "短句。", {}, natural_reading=True)

    assert backend.seeds[0] is not None
    assert backend.seeds[1] is not None
    assert backend.seeds[0] != backend.seeds[1]


@pytest.mark.asyncio
async def test_standard_mode_keeps_single_shot_behavior():
    class FakeBackend:
        def __init__(self):
            self.calls = []

        async def generate(self, text, _voice_prompt, _language, seed, _instruct):
            self.calls.append((text, seed))
            return np.ones(100, dtype=np.float32), SAMPLE_RATE

    backend = FakeBackend()

    audio, _ = await generate_chunked(
        backend,
        "第一句。第二句。",
        {},
        natural_reading=False,
        max_chunk_chars=100,
        crossfade_ms=50,
    )

    assert backend.calls == [("第一句。第二句。", None)]
    assert len(audio) == 100


@pytest.mark.asyncio
async def test_chunk_progress_reports_the_segment_currently_being_generated():
    class FakeBackend:
        async def generate(self, _text, _voice_prompt, _language, _seed, _instruct):
            return np.ones(100, dtype=np.float32), SAMPLE_RATE

    progress = []

    async def report_progress(current: int, total: int):
        progress.append((current, total))

    await generate_chunked(
        FakeBackend(),
        "第一句。第二句。第三句。",
        {},
        max_chunk_chars=5,
        progress_callback=report_progress,
    )

    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_natural_reading_api_models_default_off_and_accept_opt_in():
    request = GenerationRequest(profile_id="voice-id", text="测试")
    settings_patch = GenerationSettingsUpdate(natural_reading=True)

    assert request.natural_reading is False
    assert settings_patch.natural_reading is True


def test_migration_adds_natural_reading_without_enabling_existing_data(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE generations (
                id VARCHAR PRIMARY KEY,
                profile_id VARCHAR NOT NULL,
                text TEXT NOT NULL,
                language VARCHAR,
                audio_path VARCHAR,
                duration FLOAT,
                seed INTEGER,
                instruct TEXT,
                created_at DATETIME
            )
        """))
        connection.execute(text("""
            INSERT INTO generations (id, profile_id, text, audio_path)
            VALUES ('existing', 'voice-id', '旧记录', '')
        """))
        connection.execute(text("""
            CREATE TABLE generation_settings (
                id INTEGER PRIMARY KEY,
                max_chunk_chars INTEGER NOT NULL DEFAULT 800,
                crossfade_ms INTEGER NOT NULL DEFAULT 50,
                normalize_audio BOOLEAN NOT NULL DEFAULT 1,
                autoplay_on_generate BOOLEAN NOT NULL DEFAULT 1
            )
        """))
        connection.execute(text("INSERT INTO generation_settings (id) VALUES (1)"))

    run_migrations(engine)

    assert "natural_reading" in {
        column["name"] for column in inspect(engine).get_columns("generations")
    }
    assert "progress_current" in {
        column["name"] for column in inspect(engine).get_columns("generations")
    }
    assert "progress_total" in {
        column["name"] for column in inspect(engine).get_columns("generations")
    }
    assert "cosyvoice_phase" in {
        column["name"] for column in inspect(engine).get_columns("generations")
    }
    assert "cosyvoice_phase_durations" in {
        column["name"] for column in inspect(engine).get_columns("generations")
    }
    assert "cosyvoice_mode" in {
        column["name"] for column in inspect(engine).get_columns("generations")
    }
    assert "dialect" in {
        column["name"] for column in inspect(engine).get_columns("generations")
    }
    assert "natural_reading" in {
        column["name"] for column in inspect(engine).get_columns("generation_settings")
    }
    assert "target_story_id" in {
        column["name"] for column in inspect(engine).get_columns("generations")
    }
    with engine.connect() as connection:
        generation_value = connection.execute(
            text("SELECT natural_reading FROM generations WHERE id = 'existing'")
        ).scalar_one()
        setting_value = connection.execute(
            text("SELECT natural_reading FROM generation_settings WHERE id = 1")
        ).scalar_one()

    assert generation_value == 0
    assert setting_value == 0


@pytest.mark.asyncio
async def test_history_remembers_natural_reading_for_retry(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'history.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(VoiceProfile(id="voice-id", name="测试声音", language="zh"))
    session.commit()

    response = await history.create_generation(
        profile_id="voice-id",
        text="第一段。\n\n第二段。",
        language="zh",
        audio_path="",
        duration=0,
        seed=None,
        db=session,
        natural_reading=True,
    )

    stored = session.query(DBGeneration).filter_by(id=response.id).one()
    assert response.natural_reading is True
    assert stored.natural_reading is True
    session.close()


@pytest.mark.asyncio
async def test_history_persists_target_story_for_generation_recovery(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'story-target.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(VoiceProfile(id="voice-id", name="测试声音", language="zh"))
    session.add(Story(id="story-id", name="测试故事"))
    session.commit()

    response = await history.create_generation(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        audio_path="",
        duration=0,
        seed=None,
        db=session,
        status="generating",
        target_story_id="story-id",
    )

    stored = session.query(DBGeneration).filter_by(id=response.id).one()
    assert response.target_story_id == "story-id"
    assert stored.target_story_id == "story-id"
    session.close()


@pytest.mark.asyncio
async def test_history_remembers_cosyvoice_configuration_for_story_cards(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'cosyvoice-history.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(VoiceProfile(id="voice-id", name="测试声音", language="zh"))
    session.commit()

    response = await history.create_generation(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        audio_path="",
        duration=0,
        seed=None,
        db=session,
        engine="cosyvoice",
        model_size="rl",
        cosyvoice_mode="instruct",
        dialect="henan",
        natural_reading=True,
    )

    stored = session.query(DBGeneration).filter_by(id=response.id).one()
    assert response.cosyvoice_mode == "instruct"
    assert response.dialect == "henan"
    assert stored.cosyvoice_mode == "instruct"
    assert stored.dialect == "henan"

    story_item = StoryItem(
        id="story-item-id",
        story_id="story-id",
        generation_id=response.id,
        start_time_ms=0,
    )
    session.add(story_item)
    session.commit()

    detail = stories._build_item_detail(story_item, stored, "测试声音", session)
    assert detail.model_size == "rl"
    assert detail.cosyvoice_mode == "instruct"
    assert detail.dialect == "henan"
    assert detail.natural_reading is True
    session.close()


@pytest.mark.asyncio
async def test_history_exposes_live_chunk_progress(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'progress.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(VoiceProfile(id="voice-id", name="测试声音", language="zh"))
    session.commit()

    generation = await history.create_generation(
        profile_id="voice-id",
        text="第一段。第二段。",
        language="zh",
        audio_path="",
        duration=0,
        seed=None,
        db=session,
        status="generating",
        engine="cosyvoice",
    )
    updated = await history.update_generation_status(
        generation.id,
        "generating",
        session,
        progress_current=2,
        progress_total=4,
        cosyvoice_phase="flow",
        cosyvoice_phase_durations={"preprocessing": 0.2, "llm_decoding": 1.1},
    )

    assert updated is not None
    assert updated.progress_current == 2
    assert updated.progress_total == 4
    assert updated.cosyvoice_phase == "flow"
    assert updated.cosyvoice_phase_durations == {"preprocessing": 0.2, "llm_decoding": 1.1}
    session.close()
