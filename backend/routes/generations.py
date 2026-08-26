"""TTS generation endpoints."""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile, get_db
from ..services import history, profiles, tts
from ..services.generation import run_generation
from ..services.instruction_summary import (
    sanitize_cosyvoice_style_summary,
    summarize_cosyvoice_style_instruction,
)
from ..services.task_queue import cancel_generation as cancel_generation_job, enqueue_generation
from ..utils.audio import load_audio
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter()

IMPORTED_AUDIO_PROFILE_NAME = "Imported Audio"
IMPORT_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
IMPORT_AUDIO_MAX_BYTES = 200 * 1024 * 1024  # 200 MB
DIALECT_INSTRUCTIONS = {
    "mandarin": "请用普通话表达。",
    "henan": "请用河南话表达。",
    "sichuan": "请用四川话表达。",
}
ENGINE_DEFAULT_MODEL_SIZES = {
    "qwen": "1.7B",
    "qwen_custom_voice": "1.7B",
    "tada": "1B",
    "cosyvoice": "rl",
}


def _get_or_create_import_profile(db: Session) -> DBVoiceProfile:
    """Singleton profile every imported audio clip points at — keeps the
    Generation FK happy without making profile_id nullable across the schema."""
    row = (
        db.query(DBVoiceProfile)
        .filter(DBVoiceProfile.name == IMPORTED_AUDIO_PROFILE_NAME)
        .first()
    )
    if row is not None:
        return row
    row = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name=IMPORTED_AUDIO_PROFILE_NAME,
        description="External audio imported into a story timeline.",
        language="en",
        voice_type="import",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _resolve_generation_engine(data: models.GenerationRequest, profile) -> str:
    return data.engine or getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen"


def _resolve_generation_model_size(
    data: models.GenerationRequest,
    profile,
    engine: str,
) -> str | None:
    from ..backends import engine_has_model_sizes

    if not engine_has_model_sizes(engine):
        return None
    if data.model_size:
        return data.model_size

    # 业务规则：只有实际使用声音档案默认引擎时，才继承该档案的具体型号；
    # 用户临时切换到其他引擎时不能误用 CosyVoice 的 rl/base 值。
    profile_model_size = None
    if engine == getattr(profile, "default_engine", None):
        profile_model_size = getattr(profile, "default_model_size", None)
    return profile_model_size or ENGINE_DEFAULT_MODEL_SIZES.get(engine, "1.7B")


def build_cosyvoice_instruction(dialect: str, style_instruction: str | None) -> str:
    """Build one non-conflicting, bounded CosyVoice instruction prompt."""
    dialect_instruction = DIALECT_INSTRUCTIONS[dialect]
    style_instruction = sanitize_cosyvoice_style_summary(style_instruction)
    # 业务规则：方言是本次生成的硬约束，必须位于提示末尾并紧邻
    # <|endofprompt|>，避免前面的人物风格提示削弱河南话/四川话要求。
    return "\n".join(part for part in (style_instruction, dialect_instruction) if part)


def snapshot_cosyvoice_configuration(
    data: models.GenerationRequest,
    engine: str,
) -> tuple[str | None, str | None]:
    """Return only the CosyVoice controls that affect this generated take.

    The form keeps a dialect selection while users switch engines, but it is
    meaningful only for Chinese CosyVoice instruct generation. Saving the
    normalized snapshot prevents story cards from mislabelling older takes.
    """
    if engine != "cosyvoice":
        return None, None
    if data.cosyvoice_mode != "instruct" or data.language != "zh":
        return data.cosyvoice_mode, None
    return data.cosyvoice_mode, data.dialect


async def prepare_generation_content(
    data: models.GenerationRequest,
    profile: DBVoiceProfile,
) -> tuple[str, str | None, str]:
    """准备最终朗读内容，并保证用户正文在整个生成链路中保持不变。

    人物设定和朗读指令只能改变表达方式，不能把正文替换成设定文本。
    不支持指令通道的声音也必须逐字朗读用户输入的正文。
    """
    source = "manual"
    if data.personality and getattr(profile, "personality", None):
        source = "personality_reading"

    engine = _resolve_generation_engine(data, profile)
    instruct = data.instruct
    if engine == "cosyvoice":
        if data.cosyvoice_mode == "reference":
            # 业务规则：参考音频跟随模式必须保持 instruct 为空，后端才会走
            # inference_zero_shot，并同时使用参考录音及其逐字文本来跟随口音、
            # 节奏和语气。即使表单里残留旧朗读指令，也不能静默切回 instruct2。
            instruct = None
        elif data.language == "zh":
            # 业务规则：人物设定仍可编辑 500 字，但长设定必须先由本地
            # AI 提炼为“可听的朗读特征”，禁止直接截取或原样传给 CosyVoice。
            # 方言下拉项是硬约束，由后端去除设定中的冲突语句后统一追加。
            style_instruction = await summarize_cosyvoice_style_instruction(data.instruct)
            instruct = build_cosyvoice_instruction(data.dialect, style_instruction)

    return data.text, instruct, source


@router.post("/generate", response_model=models.GenerationResponse)
async def generate_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech from text using a voice profile."""
    task_manager = get_task_manager()
    generation_id = str(uuid.uuid4())

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_size = _resolve_generation_model_size(data, profile, engine)
    cosyvoice_mode, dialect = snapshot_cosyvoice_configuration(data, engine)

    text, instruct, source = await prepare_generation_content(data, profile)

    generation = await history.create_generation(
        profile_id=data.profile_id,
        text=text,
        language=data.language,
        audio_path="",
        duration=0,
        seed=data.seed,
        db=db,
        instruct=instruct,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=model_size,
        cosyvoice_mode=cosyvoice_mode,
        dialect=dialect,
        natural_reading=data.natural_reading,
        source=source,
    )

    task_manager.start_generation(
        task_id=generation_id,
        profile_id=data.profile_id,
        text=text,
    )

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [e.model_dump() for e in data.effects_chain]
    else:
        import json as _json

        profile_obj = db.query(DBVoiceProfile).filter_by(id=data.profile_id).first()
        if profile_obj and profile_obj.effects_chain:
            try:
                effects_chain_config = _json.loads(profile_obj.effects_chain)
            except Exception:
                pass

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=data.profile_id,
            text=text,
            language=data.language,
            engine=engine,
            model_size=model_size,
            seed=data.seed,
            normalize=data.normalize,
            effects_chain=effects_chain_config,
            instruct=instruct,
            mode="generate",
            max_chunk_chars=data.max_chunk_chars,
            crossfade_ms=data.crossfade_ms,
            natural_reading=data.natural_reading,
        )
    )

    return generation


@router.post("/generate/{generation_id}/retry", response_model=models.GenerationResponse)
async def retry_generation(generation_id: str, db: Session = Depends(get_db)):
    """Retry a failed generation using the same parameters."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") != "failed":
        raise HTTPException(status_code=400, detail="Only failed generations can be retried")

    gen.status = "generating"
    gen.error = None
    gen.audio_path = ""
    gen.duration = 0
    gen.progress_current = None
    gen.progress_total = None
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=gen.profile_id,
            text=gen.text,
            language=gen.language,
            engine=gen.engine or "qwen",
            model_size=gen.model_size or ENGINE_DEFAULT_MODEL_SIZES.get(gen.engine or "qwen", "1.7B"),
            seed=gen.seed,
            instruct=gen.instruct,
            natural_reading=bool(gen.natural_reading),
            mode="retry",
        )
    )

    return models.GenerationResponse.model_validate(gen)


@router.post(
    "/generate/{generation_id}/regenerate",
    response_model=models.GenerationResponse,
)
async def regenerate_generation(generation_id: str, db: Session = Depends(get_db)):
    """Re-run TTS with the same parameters and save the result as a new version."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation must be completed to regenerate")

    gen.status = "generating"
    gen.error = None
    gen.progress_current = None
    gen.progress_total = None
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    version_id = str(uuid.uuid4())

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=gen.profile_id,
            text=gen.text,
            language=gen.language,
            engine=gen.engine or "qwen",
            model_size=gen.model_size or ENGINE_DEFAULT_MODEL_SIZES.get(gen.engine or "qwen", "1.7B"),
            seed=gen.seed,
            instruct=gen.instruct,
            natural_reading=bool(gen.natural_reading),
            mode="regenerate",
            version_id=version_id,
        )
    )

    return models.GenerationResponse.model_validate(gen)


@router.post("/generate/{generation_id}/cancel")
async def cancel_generation(generation_id: str, db: Session = Depends(get_db)):
    """Cancel a queued or running generation."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") not in ("loading_model", "generating"):
        raise HTTPException(status_code=400, detail="Only active generations can be cancelled")

    cancellation_state = cancel_generation_job(generation_id)
    if cancellation_state is None:
        # Row says active but the worker is no longer tracking it — the gen
        # coroutine exited without writing a terminal status (most often a
        # SQLite lock racing with the failed-status write inside the worker's
        # exception handler). Fail the row here so the user can move on.
        task_manager = get_task_manager()
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=db,
            error="Generation orphaned by worker",
        )
        return {"message": "Orphaned generation cleared"}

    if cancellation_state == "queued":
        task_manager = get_task_manager()
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=db,
            error="Generation cancelled",
        )
        return {"message": "Queued generation cancelled"}

    return {"message": "Generation cancellation requested"}


@router.get("/generate/{generation_id}/status")
async def get_generation_status(generation_id: str, db: Session = Depends(get_db)):
    """SSE endpoint that streams generation status updates."""
    import json

    async def event_stream():
        try:
            while True:
                db.expire_all()
                gen = db.query(DBGeneration).filter_by(id=generation_id).first()
                if not gen:
                    yield f"data: {json.dumps({'status': 'not_found', 'id': generation_id})}\n\n"
                    return

                payload = {
                    "id": gen.id,
                    "status": gen.status or "completed",
                    "duration": gen.duration,
                    "error": gen.error,
                    "progress_current": gen.progress_current,
                    "progress_total": gen.progress_total,
                    # Agent-originated sources ("mcp", "rest") skip main-window
                    # autoplay — the floating pill plays those directly.
                    "source": gen.source,
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if (gen.status or "completed") in ("completed", "failed"):
                    return

                await asyncio.sleep(1)
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("SSE client disconnected for generation %s", generation_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/generate/stream")
async def stream_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech and stream the WAV audio directly without saving to disk."""
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        ensure_model_cached_or_raise,
        get_tts_backend_for_engine,
        load_engine_model,
    )

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    text, instruct, _ = await prepare_generation_content(data, profile)
    tts_model = get_tts_backend_for_engine(engine)
    model_size = _resolve_generation_model_size(data, profile, engine) or "default"

    await ensure_model_cached_or_raise(engine, model_size)
    await load_engine_model(engine, model_size)

    voice_prompt = await profiles.create_voice_prompt_for_profile(
        data.profile_id,
        db,
        engine=engine,
    )

    from ..utils.chunked_tts import generate_chunked

    trim_fn = None
    runaway_detector = None
    if engine_needs_trim(engine):
        from ..utils.audio import trim_tts_output

        trim_fn = trim_tts_output
    if engine_retries_runaway(engine):
        from ..utils.audio import has_tts_runaway

        runaway_detector = has_tts_runaway

    audio, sample_rate = await generate_chunked(
        tts_model,
        text,
        voice_prompt,
        language=data.language,
        seed=data.seed,
        instruct=instruct,
        max_chunk_chars=data.max_chunk_chars,
        crossfade_ms=data.crossfade_ms,
        natural_reading=data.natural_reading,
        trim_fn=trim_fn,
        runaway_detector=runaway_detector,
    )

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [e.model_dump() for e in data.effects_chain]
    elif profile.effects_chain:
        import json as _json

        try:
            effects_chain_config = _json.loads(profile.effects_chain)
        except Exception:
            effects_chain_config = None

    if effects_chain_config:
        from ..utils.effects import apply_effects

        audio = apply_effects(audio, sample_rate, effects_chain_config)

    if data.normalize:
        from ..utils.audio import normalize_audio

        audio = normalize_audio(audio)

    wav_bytes = tts.audio_to_wav_bytes(audio, sample_rate)

    async def _wav_stream():
        try:
            chunk_size = 64 * 1024
            for i in range(0, len(wav_bytes), chunk_size):
                yield wav_bytes[i : i + chunk_size]
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("Client disconnected during audio stream")

    return StreamingResponse(
        _wav_stream(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@router.post("/generate/import", response_model=models.GenerationResponse)
async def import_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Register an external audio file as a generation row.

    Designed for the story timeline so users can drop in music or other
    non-TTS audio. The row points at a singleton "Imported Audio" profile
    so the existing generation/story plumbing keeps working unchanged."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMPORT_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(IMPORT_AUDIO_EXTENSIONS)}",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > IMPORT_AUDIO_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {IMPORT_AUDIO_MAX_BYTES // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    generation_id = str(uuid.uuid4())
    target = config.get_generations_dir() / f"{generation_id}{suffix}"
    target.write_bytes(audio_bytes)

    try:
        audio, sr = load_audio(str(target))
        duration = float(len(audio) / sr) if sr else 0.0
    except Exception as decode_err:
        try:
            target.unlink()
        except OSError:
            pass
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio: {decode_err}",
        ) from decode_err

    profile = _get_or_create_import_profile(db)
    display_name = Path(file.filename or "Imported audio").stem or "Imported audio"

    return await history.create_generation(
        profile_id=profile.id,
        text=display_name,
        language="en",
        audio_path=config.to_storage_path(target),
        duration=duration,
        seed=None,
        db=db,
        generation_id=generation_id,
        status="completed",
        engine="import",
        model_size=None,
        source="import",
    )
