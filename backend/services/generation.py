"""
Unified TTS generation orchestration.

Replaces the three near-identical closures (_run_generation, _run_retry,
_run_regenerate) that lived in main.py with a single ``run_generation()``
function parameterized by *mode*.

Mode differences:
  - "generate"   : full pipeline -- save clean version, optionally apply
                    effects and create a processed version.
  - "retry"      : re-runs a failed generation with the same seed.
                    No effects, no version creation.
  - "regenerate" : re-runs with seed=None for variation.  Creates a new
                    version with an auto-incremented "take-N" label.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass
from typing import Literal, Optional

from .. import config
from ..database import get_db
from ..utils.tasks import get_task_manager
from . import history, profiles

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationRunResult:
    """The in-process proof that a runner has published its terminal state."""

    status: Literal["completed", "failed"]


def _chunking_options(engine: str, max_chunk_chars: Optional[int]) -> dict:
    """Return engine-specific chunk limits without changing public settings.

    CosyVoice runs one independently terminable worker per semantic unit. It
    may only split at author-supplied boundaries; a numeric ceiling would
    alter an unpunctuated sentence, so it is intentionally not used here.
    Other engines retain their existing chunk behavior.
    """
    if engine != "cosyvoice":
        return {"max_chunk_chars": max_chunk_chars} if max_chunk_chars is not None else {}

    return {"semantic_boundaries_only": True}


async def run_generation(
    *,
    generation_id: str,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: Optional[int],
    normalize: bool = False,
    effects_chain: Optional[list] = None,
    instruct: Optional[str] = None,
    mode: Literal["generate", "retry", "regenerate"],
    max_chunk_chars: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
    natural_reading: bool = False,
    version_id: Optional[str] = None,
) -> GenerationRunResult:
    """Execute TTS inference and persist the result.

    This is the single entry point for all background generation work.
    It is designed to be enqueued via ``services.task_queue.enqueue_generation``.
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        acquire_tts_runtime,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..utils.chunked_tts import generate_chunked
    from ..utils.audio import has_tts_runaway, normalize_audio, save_audio, trim_tts_output

    task_manager = get_task_manager()
    bg_db = next(get_db())
    cosyvoice_phase: str | None = None
    cosyvoice_phase_durations: dict[str, float] = {}

    try:
        # The lease spans model load, prompt construction and audio inference.
        # This is what prevents a CosyVoice request from evicting an already
        # running preview or streaming request belonging to another engine.
        async with acquire_tts_runtime(engine):
            tts_model = get_tts_backend_for_engine(engine)

            if not tts_model.is_loaded():
                await history.update_generation_status(generation_id, "loading_model", bg_db)

            await load_engine_model(engine, model_size)

            voice_prompt = await profiles.create_voice_prompt_for_profile(
                profile_id,
                bg_db,
                use_cache=True,
                engine=engine,
            )

            await history.update_generation_status(generation_id, "generating", bg_db)
            trim_fn = trim_tts_output if engine_needs_trim(engine) else None
            runaway_detector = has_tts_runaway if engine_retries_runaway(engine) else None

            gen_kwargs: dict = dict(
                language=language,
                seed=seed if mode != "regenerate" else None,
                instruct=instruct,
                trim_fn=trim_fn,
                runaway_detector=runaway_detector,
            )
            gen_kwargs.update(_chunking_options(engine, max_chunk_chars))
            if crossfade_ms is not None:
                gen_kwargs["crossfade_ms"] = crossfade_ms
            gen_kwargs["natural_reading"] = natural_reading

            async def report_chunk_progress(current: int, total: int) -> None:
                # Persist progress because the desktop can reload while generation
                # continues; the SSE endpoint and history API then agree on the
                # exact segment currently being synthesized.
                await history.update_generation_status(
                    generation_id,
                    "generating",
                    bg_db,
                    progress_current=current,
                    progress_total=total,
                )

            gen_kwargs["progress_callback"] = report_chunk_progress
            phase_tasks: list[asyncio.Task] = []
            phase_lock = asyncio.Lock()
            phase_started_at: float | None = None
            request_loop = asyncio.get_running_loop()

            def report_cosyvoice_phase(next_phase: str) -> None:
                """Move worker-thread events onto the request event loop."""
                nonlocal cosyvoice_phase, phase_started_at
                async def persist_phase() -> None:
                    nonlocal cosyvoice_phase, phase_started_at
                    async with phase_lock:
                        now = request_loop.time()
                        if cosyvoice_phase is not None and phase_started_at is not None:
                            cosyvoice_phase_durations[cosyvoice_phase] = round(
                                cosyvoice_phase_durations.get(cosyvoice_phase, 0.0)
                                + now - phase_started_at,
                                3,
                            )
                        cosyvoice_phase = next_phase
                        phase_started_at = now
                        await history.update_generation_status(
                            generation_id,
                            "generating",
                            bg_db,
                            cosyvoice_phase=cosyvoice_phase,
                            cosyvoice_phase_durations=dict(cosyvoice_phase_durations),
                        )

                request_loop.call_soon_threadsafe(
                    lambda: phase_tasks.append(asyncio.create_task(persist_phase()))
                )

            if engine == "cosyvoice":
                tts_model.set_phase_callback(report_cosyvoice_phase)
            try:
                audio, sample_rate = await generate_chunked(tts_model, text, voice_prompt, **gen_kwargs)
            finally:
                if engine == "cosyvoice":
                    tts_model.set_phase_callback(None)
                    # The worker emits phase events from a background thread;
                    # drain scheduled history writes before a final status is
                    # committed or the request database session is closed.
                    await asyncio.sleep(0)
                    if phase_tasks:
                        await asyncio.gather(*phase_tasks)
                    if cosyvoice_phase is not None and phase_started_at is not None:
                        cosyvoice_phase_durations[cosyvoice_phase] = round(
                            cosyvoice_phase_durations.get(cosyvoice_phase, 0.0)
                            + asyncio.get_running_loop().time() - phase_started_at,
                            3,
                        )
                        await history.update_generation_status(
                            generation_id,
                            "generating",
                            bg_db,
                            cosyvoice_phase=cosyvoice_phase,
                            cosyvoice_phase_durations=dict(cosyvoice_phase_durations),
                        )

        # --- Normalize (generate and regenerate always; retry skips) -----
        if normalize or mode == "regenerate":
            audio = normalize_audio(audio)

        duration = len(audio) / sample_rate

        # --- Persist audio and update status -----------------------------
        if mode == "generate":
            final_path = _save_generate(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                effects_chain=effects_chain,
                save_audio=save_audio,
                db=bg_db,
            )
        elif mode == "retry":
            final_path = _save_retry(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
            )
        elif mode == "regenerate":
            final_path = _save_regenerate(
                generation_id=generation_id,
                version_id=version_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
                db=bg_db,
            )

        # Persist the finished media first while the task is still marked as
        # generating. If it originated from a story, attach it before exposing
        # the completed state so SSE clients can never observe completion while
        # the story relationship is still missing.
        completed_generation = await history.update_generation_status(
            generation_id=generation_id,
            status="generating",
            db=bg_db,
            audio_path=final_path,
            duration=duration,
        )

        target_story_id = (
            completed_generation.target_story_id if completed_generation is not None else None
        )
        if target_story_id:
            from ..models import StoryItemCreate
            from . import stories

            # add_item_to_story is idempotent for a story/generation pair. This
            # also makes retries safe if a process exits between attaching the
            # item and publishing the final completed status.
            await stories.add_item_to_story(
                target_story_id,
                StoryItemCreate(generation_id=generation_id),
                bg_db,
            )

        final_status = await history.update_generation_status(
            generation_id=generation_id,
            status="completed",
            db=bg_db,
        )
        if final_status is None:
            raise RuntimeError("Generation record disappeared before completion")

    except asyncio.CancelledError:
        # Cancellation must still publish a user-visible failed state so the
        # queue worker never has to fall back to the "Generation interrupted
        # before completion" diagnostic. ``bg_db`` is already opened for this
        # worker and reused in the finally block.
        try:
            failed_status = await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=bg_db,
                error="Generation cancelled",
            )
        except Exception:
            logger.exception("Failed to persist cancelled status for %s", generation_id)
            failed_status = None
        if failed_status is None:
            # The cancellation record itself should never raise; the task
            # queue's recovery path will retry as orphaned cleanup.
            logger.warning(
                "Cancellation record missing for generation %s; queue recovery will retry",
                generation_id,
            )
        _notify_speak_end(generation_id, status="cancelled")
        return GenerationRunResult(status="failed")
    except Exception as e:
        traceback.print_exc()
        error = str(e)
        if engine == "cosyvoice":
            phase = getattr(e, "phase", None) or cosyvoice_phase or "preprocessing"
            safe_labels = {
                "preprocessing": "前处理",
                "llm_decoding": "LLM 解码",
                "flow": "Flow",
                "vocoder": "声码器",
            }
            error = f"CosyVoice 在{safe_labels.get(phase, '前处理')}阶段失败，可重试。"  # noqa: RUF001
        # MLX / Metal native crashes are surfaced as ``RuntimeError`` from
        # ``MLXTTSBackend`` after the runtime-reset retry already failed.
        # Replace the raw native message with a user-facing summary so the
        # desktop shows an actionable hint instead of an opaque stack trace.
        elif "MLX 推理失败" in error or isinstance(e, RuntimeError) and getattr(e, "__mlx_native__", False):
            error = "模型推理失败(MLX runtime),请重试或重启 Voicebox"
        else:
            # The PyInstaller sidecar bundles every backend module in a single
            # zlib-compressed PYZ. When that archive is corrupted, the import
            # call inside ``get_tts_backend_for_engine`` raises ``zlib.error``
            # before inference ever starts. The raw message ("Error -3 while
            # decompressing data: incorrect header check") is not actionable
            # for users and changes between Python versions, so map the whole
            # family to a single stable diagnostic. The descriptor carried in
            # ``failure_subtype`` lets support tell zlib corruption apart from
            # a genuinely missing dependency without widening the user copy.
            from .generation_diagnostics import (
                classify_binary_extraction_failure,
                write_generation_diagnostic,
            )

            failure_subtype = classify_binary_extraction_failure(e)
            if failure_subtype is not None:
                diagnostic = write_generation_diagnostic(
                    kind="binary_module_extraction_failed",
                    generation_id=generation_id,
                    engine=engine,
                    model_size=model_size,
                    progress_current=None,
                    progress_total=None,
                    lifecycle="unexpected",
                    failure_subtype=failure_subtype,
                )
                error = diagnostic.error_code
        try:
            failed_status = await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=bg_db,
                error=error,
                cosyvoice_phase=cosyvoice_phase if engine == "cosyvoice" else None,
                cosyvoice_phase_durations=(cosyvoice_phase_durations if engine == "cosyvoice" else None),
            )
        except Exception:
            logger.exception("Failed to persist failed status for %s", generation_id)
            failed_status = None
        if failed_status is None:
            # Surface the failure but never re-raise: the task queue worker
            # must be able to advance without falling back to its generic
            # "Generation interrupted before completion" diagnostic.
            logger.error(
                "Generation row missing while recording failure for %s: %s",
                generation_id,
                error,
            )
        _notify_speak_end(generation_id, status="failed")
        return GenerationRunResult(status="failed")
    else:
        _notify_speak_end(generation_id, status="completed")
        return GenerationRunResult(status="completed")
    finally:
        task_manager.complete_generation(generation_id)
        bg_db.close()


def _notify_speak_end(generation_id: str, *, status: str) -> None:
    """Publish a speak-end event; the frontend ignores unknown ids."""
    try:
        from ..mcp_server import events as mcp_events

        mcp_events.publish(
            "speak-end",
            {"generation_id": generation_id, "status": status},
        )
    except Exception:
        # Never let event pub/sub break generation completion.
        pass


def _save_generate(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    effects_chain: Optional[list],
    save_audio,
    db,
) -> str:
    """Save clean version and optionally an effects-processed version.

    Returns the final audio path (processed if effects were applied,
    otherwise clean).
    """
    from . import versions as versions_mod

    clean_audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(clean_audio_path), sample_rate)

    has_effects = effects_chain and any(e.get("enabled", True) for e in effects_chain)

    versions_mod.create_version(
        generation_id=generation_id,
        label="original",
        audio_path=config.to_storage_path(clean_audio_path),
        db=db,
        effects_chain=None,
        is_default=not has_effects,
    )

    final_audio_path = str(clean_audio_path)

    if has_effects:
        from ..utils.effects import apply_effects, validate_effects_chain

        assert effects_chain is not None

        error_msg = validate_effects_chain(effects_chain)
        if error_msg:
            import logging
            logging.getLogger(__name__).warning("invalid effects chain, skipping: %s", error_msg)
            versions_mod.set_default_version(
                versions_mod.list_versions(generation_id, db)[0].id, db
            )
        else:
            processed_audio = apply_effects(audio, sample_rate, effects_chain)
            processed_path = config.get_generations_dir() / f"{generation_id}_processed.wav"
            save_audio(processed_audio, str(processed_path), sample_rate)
            final_audio_path = str(processed_path)
            versions_mod.create_version(
                generation_id=generation_id,
                label="version-2",
                audio_path=config.to_storage_path(processed_path),
                db=db,
                effects_chain=effects_chain,
                is_default=True,
            )

    return config.to_storage_path(final_audio_path)


def _save_retry(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    save_audio,
) -> str:
    """Save retry output -- single file, no versions.

    Returns the audio path.
    """
    audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(audio_path), sample_rate)
    return config.to_storage_path(audio_path)


async def generate_audio_sync(
    *,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: Optional[int] = None,
    instruct: Optional[str] = None,
    normalize: bool = True,
    max_chunk_chars: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
    natural_reading: bool = False,
) -> bytes:
    """Run a TTS generation synchronously and return the resulting wav bytes.

    Unlike :func:`run_generation`, this path does not touch the
    ``generations`` table, enqueue work, or write anything to the
    generations directory. It's used by ``POST /profiles/{id}/speak``
    when the caller passes ``persist=false`` — they just want the audio
    back in the HTTP response without polluting their history.

    Loads the engine model on demand, runs ``generate_chunked``, optional
    normalize, then encodes in-memory via :func:`tts.audio_to_wav_bytes`
    (same helper ``/generate/stream`` uses).
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        acquire_tts_runtime,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..utils.chunked_tts import generate_chunked
    from ..utils.audio import has_tts_runaway, normalize_audio, trim_tts_output
    from . import tts

    bg_db = next(get_db())
    try:
        async with acquire_tts_runtime(engine):
            tts_model = get_tts_backend_for_engine(engine)
            await load_engine_model(engine, model_size)

            voice_prompt = await profiles.create_voice_prompt_for_profile(
                profile_id,
                bg_db,
                use_cache=True,
                engine=engine,
            )

            trim_fn = trim_tts_output if engine_needs_trim(engine) else None
            runaway_detector = has_tts_runaway if engine_retries_runaway(engine) else None

            gen_kwargs: dict = dict(
                language=language,
                seed=seed,
                instruct=instruct,
                trim_fn=trim_fn,
                runaway_detector=runaway_detector,
            )
            gen_kwargs.update(_chunking_options(engine, max_chunk_chars))
            if crossfade_ms is not None:
                gen_kwargs["crossfade_ms"] = crossfade_ms
            gen_kwargs["natural_reading"] = natural_reading

            audio, sample_rate = await generate_chunked(
                tts_model, text, voice_prompt, **gen_kwargs
            )
    finally:
        bg_db.close()

    if normalize:
        audio = normalize_audio(audio)

    return tts.audio_to_wav_bytes(audio, sample_rate)


def _save_regenerate(
    *,
    generation_id: str,
    version_id: Optional[str],
    audio,
    sample_rate: int,
    save_audio,
    db,
) -> str:
    """Save regeneration output as a new version with auto-label.

    Returns the audio path.
    """
    from . import versions as versions_mod

    import uuid as _uuid

    suffix = _uuid.uuid4().hex[:8]
    audio_path = config.get_generations_dir() / f"{generation_id}_{suffix}.wav"
    save_audio(audio, str(audio_path), sample_rate)

    # Count via DB query rather than list length to avoid TOCTOU race
    from ..database import GenerationVersion as DBGenerationVersion

    count = db.query(DBGenerationVersion).filter_by(generation_id=generation_id).count()
    label = f"take-{count + 1}"

    versions_mod.create_version(
        generation_id=generation_id,
        label=label,
        audio_path=config.to_storage_path(audio_path),
        db=db,
        effects_chain=None,
        is_default=True,
    )

    return config.to_storage_path(audio_path)
