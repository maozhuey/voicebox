"""Regression coverage for importing video files into Captures."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database import Base
from backend.services import captures


class _FakeWhisper:
    """Small STT double that verifies captures receive a canonical WAV."""

    model_size = "turbo"

    async def transcribe(self, path: str, language: str | None, model_size: str) -> str:
        assert Path(path).suffix == ".wav"
        assert language == "zh"
        assert model_size == "turbo"
        return "视频中的语音"


@pytest.fixture
def capture_db(tmp_path):
    """Create an isolated captures database."""
    engine = create_engine(f"sqlite:///{tmp_path / 'captures.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_create_capture_extracts_audio_from_mp4_before_transcribing(monkeypatch, tmp_path, capture_db):
    """An MP4 upload is decoded before Whisper receives the audio."""
    captures_dir = tmp_path / "captures"
    decoded_paths: list[Path] = []

    def fake_load_audio(path: str):
        decoded_paths.append(Path(path))
        return np.zeros(24_000, dtype=np.float32), 24_000

    captures_dir.mkdir()
    monkeypatch.setattr(captures.config, "get_captures_dir", lambda: captures_dir)
    monkeypatch.setattr(captures, "load_audio", fake_load_audio)
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _FakeWhisper())

    capture = await captures.create_capture(
        audio_bytes=b"minimal mp4 payload",
        filename="meeting.mp4",
        source="file",
        language="zh",
        stt_model="turbo",
        db=capture_db,
    )

    assert decoded_paths[0].suffix == ".mp4"
    assert capture.audio_path.endswith(".wav")
    assert capture.transcript_raw == "视频中的语音"
