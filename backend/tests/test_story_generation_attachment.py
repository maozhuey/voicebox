"""Regression coverage for durable story-bound generation tasks."""

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import backends, config
from backend.database import Base, Generation, Story, StoryItem, VoiceProfile
from backend.services import generation as generation_service
from backend.services import profiles
from backend.utils import audio as audio_utils
from backend.utils import chunked_tts


@pytest.mark.asyncio
async def test_completed_generation_is_attached_to_persisted_target_story(
    tmp_path,
    monkeypatch,
):
    engine = create_engine(f"sqlite:///{tmp_path / 'story-generation.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        session.add(VoiceProfile(id="voice-id", name="测试声音", language="zh"))
        session.add(Story(id="story-id", name="测试故事"))
        session.add(
            Generation(
                id="generation-id",
                profile_id="voice-id",
                target_story_id="story-id",
                text="这是需要自动加入故事的正文。",
                language="zh",
                audio_path="",
                duration=0,
                status="generating",
                engine="qwen",
            )
        )
        session.commit()

    class FakeBackend:
        @staticmethod
        def is_loaded():
            return True

    class FakeTaskManager:
        @staticmethod
        def complete_generation(_generation_id):
            return None

    async def fake_load_engine_model(_engine, _model_size):
        return None

    async def fake_create_voice_prompt(*_args, **_kwargs):
        return object()

    async def fake_generate_chunked(*_args, **_kwargs):
        return np.ones(250, dtype=np.float32), 1000

    def fake_save_audio(_audio, path, _sample_rate):
        Path(path).write_bytes(b"test-wav")

    generations_dir = tmp_path / "generations"
    generations_dir.mkdir()
    monkeypatch.setattr(generation_service, "get_db", lambda: iter([session_factory()]))
    monkeypatch.setattr(generation_service, "get_task_manager", lambda: FakeTaskManager())
    monkeypatch.setattr(generation_service, "_notify_speak_end", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda _engine: FakeBackend())
    monkeypatch.setattr(backends, "load_engine_model", fake_load_engine_model)
    monkeypatch.setattr(profiles, "create_voice_prompt_for_profile", fake_create_voice_prompt)
    monkeypatch.setattr(chunked_tts, "generate_chunked", fake_generate_chunked)
    monkeypatch.setattr(audio_utils, "save_audio", fake_save_audio)
    monkeypatch.setattr(config, "get_generations_dir", lambda: generations_dir)
    monkeypatch.setattr(config, "to_storage_path", lambda path: str(path))

    result = await generation_service.run_generation(
        generation_id="generation-id",
        profile_id="voice-id",
        text="这是需要自动加入故事的正文。",
        language="zh",
        engine="qwen",
        model_size="1.7B",
        seed=None,
        mode="generate",
    )

    with session_factory() as session:
        completed = session.query(Generation).filter_by(id="generation-id").one()
        story_item = session.query(StoryItem).filter_by(generation_id="generation-id").one()

        assert completed.status == "completed"
        assert completed.duration == 0.25
        assert story_item.story_id == "story-id"
        assert story_item.start_time_ms == 0
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_model_load_timeout_marks_generation_failed_and_releases_queue(tmp_path, monkeypatch):
    """A timed-out loader must never leave the history item in loading_model."""
    engine = create_engine(f"sqlite:///{tmp_path / 'load-timeout.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        session.add(VoiceProfile(id="voice-id", name="测试声音", language="zh"))
        session.add(
            Generation(
                id="generation-id",
                profile_id="voice-id",
                text="超时测试",
                language="zh",
                audio_path="",
                duration=0,
                status="loading_model",
                engine="qwen_custom_voice",
            )
        )
        session.commit()

    class FakeBackend:
        @staticmethod
        def is_loaded():
            return False

    class FakeTaskManager:
        completed = []

        @classmethod
        def complete_generation(cls, generation_id):
            cls.completed.append(generation_id)

    async def fail_loading(*_args, **_kwargs):
        from backend.backends import ModelLoadTimeoutError

        raise ModelLoadTimeoutError("模型加载超时，请重试或检查模型与设备")

    monkeypatch.setattr(generation_service, "get_db", lambda: iter([session_factory()]))
    monkeypatch.setattr(generation_service, "get_task_manager", lambda: FakeTaskManager())
    monkeypatch.setattr(generation_service, "_notify_speak_end", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda _engine: FakeBackend())
    monkeypatch.setattr(backends, "load_engine_model", fail_loading)

    result = await generation_service.run_generation(
        generation_id="generation-id",
        profile_id="voice-id",
        text="超时测试",
        language="zh",
        engine="qwen_custom_voice",
        model_size="1.7B",
        seed=None,
        mode="generate",
    )

    with session_factory() as session:
        failed = session.query(Generation).filter_by(id="generation-id").one()
        assert failed.status == "failed"
        assert failed.error == "模型加载超时，请重试或检查模型与设备"
    assert result.status == "failed"
    assert FakeTaskManager.completed == ["generation-id"]
