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
from backend.services.refinement import CaptureRefinementError


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


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["model_load", "llm_generate", "persist"])
async def test_refinement_error_returns_safe_stage_diagnostic(monkeypatch, tmp_path, capture_db, stage):
    """A successful capture remains private and recoverable when refine fails."""
    row = captures.DBCapture(
        id=f"capture-{stage}",
        audio_path="captures/private-audio.wav",
        source="dictation",
        transcript_raw="绝不能写入日志的听写正文",
        stt_model="turbo",
    )
    capture_db.add(row)
    capture_db.commit()
    monkeypatch.setattr(capture_diagnostics.config, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(capture_routes.settings_service, "get_capture_settings", lambda _db: type("Settings", (), {
        "llm_model": "1.7B",
        "smart_cleanup": True,
        "self_correction": True,
        "preserve_technical": True,
    })())

    async def fail_refinement(**_kwargs):
        try:
            raise RuntimeError("/private/private-audio.wav 绝不能写入日志的听写正文")
        except RuntimeError as cause:
            raise CaptureRefinementError(stage) from cause

    monkeypatch.setattr(capture_routes.captures_service, "refine_capture", fail_refinement)

    with pytest.raises(HTTPException) as error:
        await capture_routes.refine_capture_endpoint(
            row.id,
            capture_routes.models.CaptureRefineRequest(),
            capture_db,
        )

    assert error.value.status_code == 500
    assert error.value.detail.startswith("CAPTURE_REFINEMENT_DIAGNOSTIC:cap-")
    entry = json.loads((tmp_path / "logs" / "capture-diagnostics.jsonl").read_text().strip())
    assert entry["kind"] == "refinement"
    assert entry["stage"] == stage
    assert entry["source"] == "dictation"
    assert entry["model_size"] == "1.7B"
    serialized = json.dumps(entry, ensure_ascii=False)
    assert "绝不能写入日志" not in serialized
    assert "/private" not in serialized
    assert "private-audio.wav" not in serialized
