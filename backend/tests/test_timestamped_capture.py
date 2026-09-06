"""Persistence and API contracts for timestamped capture transcripts."""

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import UploadFile

from backend.database import Base, Capture
from backend.database.migrations import run_migrations
from backend.routes import transcription as transcription_route
from backend.services import captures
from backend.services.refinement import RefinementFlags
from backend.transcription import build_transcription_result


class _FakeWhisper:
    model_size = "large"

    def is_loaded(self):
        return True

    def _is_model_cached(self, _model_size):
        return True

    async def transcribe(self, *_args, **_kwargs):
        return build_transcription_result(
            "原始第一句 原始第二句",
            [
                {"start": 0, "end": 1.2, "text": "原始第一句"},
                {"start": 1.2, "end": 2.4, "text": "原始第二句"},
            ],
        )


class _FailingWhisper(_FakeWhisper):
    async def transcribe(self, *_args, **_kwargs):
        raise ValueError("Whisper did not return valid timestamped segments")


@pytest.fixture
def capture_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'captures.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_transcribe_route_keeps_old_fields_and_adds_timestamped_fields(monkeypatch):
    monkeypatch.setattr(transcription_route.transcribe, "get_whisper_model", lambda: _FakeWhisper())
    from backend.utils import audio as audio_utils

    monkeypatch.setattr(
        audio_utils,
        "load_audio",
        lambda _path: (np.zeros(38_400, dtype=np.float32), 16_000),
    )
    upload = UploadFile(file=BytesIO(b"fake wav"), filename="sample.wav")

    response = await transcription_route.transcribe_audio(
        upload,
        language="zh",
        model="large",
        initial_prompt=None,
    )

    assert response.text == "原始第一句 原始第二句"
    assert response.duration == 2.4
    assert response.timestamped_text.startswith("[00:00:00.000 --> 00:00:01.200]")
    assert response.segments[1].start_ms == 1200


@pytest.mark.parametrize("model_size", ["large", "turbo"])
async def test_large_and_turbo_share_the_timestamped_response_contract(monkeypatch, model_size):
    """Both public Whisper model values must preserve the additive API response."""
    monkeypatch.setattr(transcription_route.transcribe, "get_whisper_model", lambda: _FakeWhisper())
    from backend.utils import audio as audio_utils

    monkeypatch.setattr(
        audio_utils,
        "load_audio",
        lambda _path: (np.zeros(38_400, dtype=np.float32), 16_000),
    )
    upload = UploadFile(file=BytesIO(b"fake wav"), filename="sample.wav")

    response = await transcription_route.transcribe_audio(
        upload,
        language="zh",
        model=model_size,
        initial_prompt=None,
    )

    assert response.text == "原始第一句 原始第二句"
    assert response.duration == 2.4
    assert response.timestamped_text
    assert response.segments


def test_transcription_documentation_and_generated_client_match_runtime_contract():
    """Public artifacts must not teach the obsolete `audio`/`whisper-turbo` contract."""
    root = Path(__file__).resolve().parents[2]
    openapi = json.loads((root / "docs/openapi.json").read_text(encoding="utf-8"))
    response = openapi["components"]["schemas"]["TranscriptionResponse"]
    request = openapi["components"]["schemas"]["Body_transcribe_audio_transcribe_post"]

    assert {"text", "duration", "timestamped_text", "segments"} <= set(response["properties"])
    assert {"text", "duration", "timestamped_text", "segments"} <= set(response["required"])
    assert {"file", "model"} <= set(request["properties"])

    generated_client = (root / "app/src/lib/api/models/TranscriptionResponse.ts").read_text(encoding="utf-8")
    assert "timestamped_text: string;" in generated_client
    assert "segments: Array<TranscriptSegment>;" in generated_client

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert '-F "file=@recording.wav"' in readme
    assert '-F "model=turbo"' in readme


@pytest.mark.asyncio
async def test_transcribe_route_rejects_a_result_without_timestamps(monkeypatch):
    monkeypatch.setattr(
        transcription_route.transcribe,
        "get_whisper_model",
        lambda: _FailingWhisper(),
    )
    from backend.utils import audio as audio_utils

    monkeypatch.setattr(audio_utils, "load_audio", lambda _path: (np.zeros(16_000), 16_000))
    upload = UploadFile(file=BytesIO(b"fake wav"), filename="sample.wav")

    with pytest.raises(HTTPException) as error:
        await transcription_route.transcribe_audio(
            upload,
            language="zh",
            model="large",
            initial_prompt=None,
        )

    assert getattr(error.value, "status_code", None) == 500
    assert "timestamped segments" in getattr(error.value, "detail", "")


@pytest.mark.asyncio
async def test_capture_persists_segments_and_refinement_does_not_change_them(monkeypatch, tmp_path, capture_db):
    captures_dir = tmp_path / "captures"
    captures_dir.mkdir()
    monkeypatch.setattr(captures.config, "get_captures_dir", lambda: captures_dir)
    monkeypatch.setattr(captures, "load_audio", lambda _path: (np.zeros(38_400), 16_000))
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _FakeWhisper())

    created = await captures.create_capture(
        audio_bytes=b"fake wav",
        filename="sample.wav",
        source="file",
        language="zh",
        stt_model="large",
        db=capture_db,
    )
    original_segments = [segment.model_dump() for segment in created.transcript_segments]

    async def fake_refine(*_args, **_kwargs):
        return "精修后的内容", "1.7B"

    monkeypatch.setattr(captures, "refine_transcript", fake_refine)
    refined = await captures.refine_capture(
        created.id,
        RefinementFlags(),
        "1.7B",
        capture_db,
    )

    assert refined.transcript_refined == "精修后的内容"
    assert [segment.model_dump() for segment in refined.transcript_segments] == original_segments
    assert "原始第一句" in refined.transcript_timestamped
    assert "精修后的内容" not in refined.transcript_timestamped


def test_legacy_and_corrupt_capture_segments_are_read_as_missing(capture_db):
    row = Capture(
        id="legacy",
        audio_path="captures/legacy.wav",
        source="file",
        transcript_raw="旧文本",
        transcript_segments="{broken json",
    )
    capture_db.add(row)
    capture_db.commit()

    response = captures.get_capture("legacy", capture_db)

    assert response.transcript_raw == "旧文本"
    assert response.transcript_segments == []
    assert response.transcript_timestamped is None


@pytest.mark.asyncio
async def test_retranscribe_replaces_raw_timeline_atomically_and_failure_keeps_it(monkeypatch, tmp_path, capture_db):
    audio_path = tmp_path / "capture.wav"
    audio_path.write_bytes(b"fake wav")
    row = Capture(
        id="existing",
        audio_path=str(audio_path),
        source="file",
        transcript_raw="旧原文",
        transcript_segments='[{"start_ms":0,"end_ms":500,"text":"旧原文"}]',
        transcript_refined="旧精修",
        llm_model="1.7B",
    )
    capture_db.add(row)
    capture_db.commit()
    monkeypatch.setattr(captures.config, "resolve_storage_path", lambda _path: audio_path)
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _FakeWhisper())

    updated = await captures.retranscribe_capture(
        "existing",
        stt_model="large",
        language="zh",
        db=capture_db,
    )
    assert updated.transcript_raw == "原始第一句 原始第二句"
    assert updated.transcript_segments[1].start_ms == 1200
    assert updated.transcript_refined is None

    snapshot = (
        updated.transcript_raw,
        [segment.model_dump() for segment in updated.transcript_segments],
        updated.transcript_refined,
    )
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _FailingWhisper())
    with pytest.raises(ValueError, match="timestamped segments"):
        await captures.retranscribe_capture(
            "existing",
            stt_model="large",
            language="zh",
            db=capture_db,
        )

    unchanged = captures.get_capture("existing", capture_db)
    assert (
        unchanged.transcript_raw,
        [segment.model_dump() for segment in unchanged.transcript_segments],
        unchanged.transcript_refined,
    ) == snapshot


def test_capture_segment_migration_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE captures (id VARCHAR PRIMARY KEY, transcript_raw TEXT)"))

    run_migrations(engine)
    run_migrations(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("captures")}
    assert "transcript_segments" in columns
    engine.dispose()


def test_new_capture_database_migration_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'new.db'}")
    Base.metadata.create_all(bind=engine)

    run_migrations(engine)
    run_migrations(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("captures")}
    assert "transcript_segments" in columns
    engine.dispose()
