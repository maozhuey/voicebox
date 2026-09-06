"""Regression coverage for safe capture failure handling."""

from io import BytesIO
import json

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import UploadFile

from backend.database import Base
from backend.routes import captures as capture_routes
from backend.services import capture_diagnostics, captures


class _FailingWhisper:
    model_size = "turbo"

    async def transcribe(self, *_args, **_kwargs):
        raise RuntimeError("runtime failed for /private/user-sensitive.wav")


@pytest.fixture
def capture_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'captures.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_corrupt_webm_is_a_400_and_leaves_no_audio(monkeypatch, tmp_path, capture_db):
    captures_dir = tmp_path / "captures"
    captures_dir.mkdir()
    monkeypatch.setattr(captures.config, "get_captures_dir", lambda: captures_dir)
    monkeypatch.setattr(captures, "load_audio", lambda _path: (_ for _ in ()).throw(ValueError("bad webm")))

    with pytest.raises(ValueError, match="Could not decode .webm audio"):
        await captures.create_capture(
            audio_bytes=b"not a recording",
            filename="dictation.webm",
            source="dictation",
            language="zh",
            stt_model="turbo",
            db=capture_db,
        )

    assert list(captures_dir.iterdir()) == []
    assert capture_db.query(captures.DBCapture).count() == 0


@pytest.mark.asyncio
async def test_internal_transcription_error_returns_safe_diagnostic(monkeypatch, tmp_path, capture_db):
    captures_dir = tmp_path / "captures"
    captures_dir.mkdir()
    monkeypatch.setattr(captures.config, "get_captures_dir", lambda: captures_dir)
    monkeypatch.setattr(captures, "load_audio", lambda _path: (np.zeros(16_000), 16_000))
    monkeypatch.setattr(captures, "get_whisper_model", _FailingWhisper)
    monkeypatch.setattr(capture_diagnostics.config, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(capture_routes.settings_service, "get_capture_settings", lambda _db: type("Settings", (), {
        "stt_model": "turbo", "language": "zh", "transcription_prompt": ""
    })())

    with pytest.raises(HTTPException) as error:
        await capture_routes.create_capture_endpoint(
            file=UploadFile(file=BytesIO(b"wav"), filename="safe.wav"),
            source="dictation",
            language="zh",
            stt_model="turbo",
            db=capture_db,
        )

    assert error.value.status_code == 500
    assert error.value.detail.startswith("CAPTURE_DIAGNOSTIC:cap-")
    entry = json.loads((tmp_path / "logs" / "capture-diagnostics.jsonl").read_text().strip())
    assert entry["stage"] == "transcribe"
    assert entry["error_type"] == "CaptureTranscriptionError"
    assert "safe.wav" not in json.dumps(entry)
    assert "/private" not in json.dumps(entry)
    assert list(captures_dir.iterdir()) == []
