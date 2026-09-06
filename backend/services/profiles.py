"""Voice profile management module."""

import asyncio
import json as _json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import config
from ..database import Generation as DBGeneration, ProfileSample as DBProfileSample, VoiceProfile as DBVoiceProfile
from ..models import (
    EffectConfig,
    ProfileSampleResponse,
    VoiceProfileCreate,
    VoiceProfileResponse,
)
from ..utils.audio import save_audio, validate_and_load_reference_audio
from ..utils.cache import _get_cache_dir, clear_profile_cache
from ..utils.images import process_avatar, validate_image

logger = logging.getLogger(__name__)

# These engines accept reference-audio profiles. Keep this in sync with the
# generation selector: cloned profiles must reach CosyVoice's zero-shot prompt
# path instead of being rejected before model generation begins.
CLONING_ENGINES = {"qwen", "luxtts", "chatterbox", "chatterbox_turbo", "tada", "cosyvoice"}
PROFILE_MODEL_SIZES = {"cosyvoice": {"rl", "base"}}

_PRESET_PREVIEW_TEXTS = {
    "zh": "你好，这是我的声音预览。",  # noqa: RUF001
    "en": "Hello, this is a preview of my voice.",
    "ja": "こんにちは、これは私の声のプレビューです。",
    "ko": "안녕하세요, 제 목소리 미리 듣기입니다.",
    "es": "Hola, esta es una muestra de mi voz.",
    "fr": "Bonjour, voici un aperçu de ma voix.",
    "hi": "नमस्ते, यह मेरी आवाज़ का नमूना है।",
    "it": "Ciao, questa è un'anteprima della mia voce.",
    "pt": "Olá, esta é uma prévia da minha voz.",
}
_PRESET_PREVIEW_CACHE_VERSION = "v2"


def _profile_to_response(
    profile: DBVoiceProfile,
    generation_count: int = 0,
    sample_count: int = 0,
) -> VoiceProfileResponse:
    """Convert a DB profile to a VoiceProfileResponse, deserializing effects_chain."""
    effects_chain = None
    if profile.effects_chain:
        try:
            raw = _json.loads(profile.effects_chain)
            effects_chain = [EffectConfig(**e) for e in raw]
        except Exception as e:
            import logging

            logging.warning(f"Failed to parse effects_chain for profile {profile.id}: {e}")
    return VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        language=profile.language,
        avatar_path=profile.avatar_path,
        effects_chain=effects_chain,
        voice_type=getattr(profile, "voice_type", None) or "cloned",
        preset_engine=getattr(profile, "preset_engine", None),
        preset_voice_id=getattr(profile, "preset_voice_id", None),
        design_prompt=getattr(profile, "design_prompt", None),
        default_engine=getattr(profile, "default_engine", None),
        default_model_size=getattr(profile, "default_model_size", None),
        personality=getattr(profile, "personality", None),
        generation_count=generation_count,
        sample_count=sample_count,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _get_preset_voice_ids(engine: str) -> set[str]:
    if engine == "kokoro":
        from ..backends.kokoro_backend import KOKORO_VOICES

        return {voice_id for voice_id, _name, _gender, _lang in KOKORO_VOICES}

    if engine == "qwen_custom_voice":
        from ..backends.qwen_custom_voice_backend import QWEN_CUSTOM_VOICES

        return {voice_id for voice_id, _name, _gender, _lang, _desc in QWEN_CUSTOM_VOICES}

    return set()


def _get_preset_voice_language(engine: str, voice_id: str) -> str | None:
    """Return a preset voice's language after verifying its identifier."""
    if engine == "kokoro":
        from ..backends.kokoro_backend import KOKORO_VOICES

        return next(
            (language for identifier, _name, _gender, language in KOKORO_VOICES if identifier == voice_id), None
        )

    if engine == "qwen_custom_voice":
        from ..backends.qwen_custom_voice_backend import QWEN_CUSTOM_VOICES

        return next(
            (
                language
                for identifier, _name, _gender, language, _description in QWEN_CUSTOM_VOICES
                if identifier == voice_id
            ),
            None,
        )

    return None


def _get_preset_preview_cache_path(engine: str, voice_id: str) -> Path:
    """Return the local audio cache path for a versioned preset preview."""
    return _get_cache_dir() / "preset_previews" / engine / f"{voice_id}-{_PRESET_PREVIEW_CACHE_VERSION}.wav"


def _read_preset_preview_cache(cache_path: Path) -> bytes | None:
    """Read a non-empty cached preview, treating failed reads as cache misses."""
    try:
        wav_bytes = cache_path.read_bytes()
        return wav_bytes or None
    except OSError:
        return None


def _write_preset_preview_cache(cache_path: Path, wav_bytes: bytes) -> None:
    """Atomically persist a generated preview so browsers can replay it locally."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False) as temp_file:
        temp_file.write(wav_bytes)
        temp_path = Path(temp_file.name)

    try:
        os.replace(temp_path, cache_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


async def generate_preset_voice_preview(engine: str, voice_id: str) -> bytes:
    """Return a cached native-language preview, generating it once when missing."""
    language = _get_preset_voice_language(engine, voice_id)
    if language is None:
        raise ValueError(f"Preset voice '{voice_id}' is not valid for engine '{engine}'")

    from ..backends import (
        acquire_tts_runtime,
        ensure_model_cached_or_raise,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from . import tts

    cache_path = _get_preset_preview_cache_path(engine, voice_id)
    cached_preview = await asyncio.to_thread(_read_preset_preview_cache, cache_path)
    if cached_preview is not None:
        return cached_preview

    # 业务规则：每个内置声音仅在本地首次试听时生成，之后浏览器取得的是缓存 WAV，  # noqa: RUF003
    # 不再触发模型推理；升级试听文案或生成规则时递增缓存版本以自动重新生成。  # noqa: RUF003
    # 试听不会创建档案或生成历史。Qwen CustomVoice 档案默认使用 1.7B，试听保持一致。  # noqa: RUF003
    model_size = "1.7B" if engine == "qwen_custom_voice" else "default"
    await ensure_model_cached_or_raise(engine, model_size)
    async with acquire_tts_runtime(engine):
        await load_engine_model(engine, model_size)

        audio, sample_rate = await get_tts_backend_for_engine(engine).generate(
            _PRESET_PREVIEW_TEXTS.get(language, _PRESET_PREVIEW_TEXTS["en"]),
            {
                "voice_type": "preset",
                "preset_engine": engine,
                "preset_voice_id": voice_id,
            },
            language=language,
        )
    wav_bytes = tts.audio_to_wav_bytes(audio, sample_rate)
    await asyncio.to_thread(_write_preset_preview_cache, cache_path, wav_bytes)
    return wav_bytes


def _validate_profile_fields(
    *,
    voice_type: str,
    preset_engine: str | None,
    preset_voice_id: str | None,
    design_prompt: str | None,
    default_engine: str | None,
    default_model_size: str | None,
) -> str | None:
    if default_model_size is not None:
        allowed_model_sizes = PROFILE_MODEL_SIZES.get(default_engine or "")
        if not allowed_model_sizes or default_model_size not in allowed_model_sizes:
            return (
                f"Default model '{default_model_size}' is not valid for engine "
                f"'{default_engine or 'none'}'"
            )

    if voice_type == "preset":
        if not preset_engine or not preset_voice_id:
            return "Preset profiles require both preset_engine and preset_voice_id"
        if default_engine and default_engine != preset_engine:
            return "Preset profiles must use their preset_engine as default_engine"

        available_voice_ids = _get_preset_voice_ids(preset_engine)
        if available_voice_ids and preset_voice_id not in available_voice_ids:
            return f"Preset voice '{preset_voice_id}' is not valid for engine '{preset_engine}'"
        return None

    if voice_type == "designed":
        if not design_prompt or not design_prompt.strip():
            return "Designed profiles require a design_prompt"
        if preset_engine or preset_voice_id:
            return "Designed profiles cannot set preset_engine or preset_voice_id"
        return None

    if preset_engine or preset_voice_id:
        return "Cloned profiles cannot set preset_engine or preset_voice_id"
    if design_prompt:
        return "Cloned profiles cannot set design_prompt"
    if default_engine and default_engine not in CLONING_ENGINES:
        return f"Cloned profiles cannot use default engine '{default_engine}'"
    return None


def validate_profile_engine(profile, engine: str) -> None:
    voice_type = getattr(profile, "voice_type", None) or "cloned"

    if voice_type == "preset":
        preset_engine = getattr(profile, "preset_engine", None)
        preset_voice_id = getattr(profile, "preset_voice_id", None)
        if not preset_engine or not preset_voice_id:
            raise ValueError(f"Preset profile {profile.id} is missing preset engine metadata")
        if preset_engine != engine:
            raise ValueError(
                f"Preset profile {profile.id} only supports engine '{preset_engine}', not '{engine}'"
            )
        return

    if voice_type == "designed":
        design_prompt = getattr(profile, "design_prompt", None)
        if not design_prompt or not design_prompt.strip():
            raise ValueError(f"Designed profile {profile.id} is missing design_prompt")
        return

    if engine not in CLONING_ENGINES:
        raise ValueError(f"Engine '{engine}' does not support cloned voice profiles")


async def create_profile(
    data: VoiceProfileCreate,
    db: Session,
) -> VoiceProfileResponse:
    """
    Create a new voice profile.

    Args:
        data: Profile creation data
        db: Database session

    Returns:
        Created profile

    Raises:
        ValueError: If a profile with the same name already exists
    """
    existing_profile = db.query(DBVoiceProfile).filter_by(name=data.name).first()
    if existing_profile:
        raise ValueError(f"A profile with the name '{data.name}' already exists. Please choose a different name.")

    # Auto-set default_engine for preset profiles
    default_engine = data.default_engine
    voice_type = data.voice_type or "cloned"
    if voice_type == "preset" and data.preset_engine and not default_engine:
        default_engine = data.preset_engine

    validation_error = _validate_profile_fields(
        voice_type=voice_type,
        preset_engine=data.preset_engine,
        preset_voice_id=data.preset_voice_id,
        design_prompt=data.design_prompt,
        default_engine=default_engine,
        default_model_size=data.default_model_size,
    )
    if validation_error:
        raise ValueError(validation_error)

    db_profile = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        language=data.language,
        voice_type=voice_type,
        preset_engine=data.preset_engine,
        preset_voice_id=data.preset_voice_id,
        design_prompt=data.design_prompt,
        default_engine=default_engine,
        default_model_size=data.default_model_size,
        personality=data.personality,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)

    profile_dir = config.get_profiles_dir() / db_profile.id
    profile_dir.mkdir(parents=True, exist_ok=True)

    return _profile_to_response(db_profile)


async def add_profile_sample(
    profile_id: str,
    audio_path: str,
    reference_text: str,
    db: Session,
) -> ProfileSampleResponse:
    """
    Add a sample to a voice profile.

    Args:
        profile_id: Profile ID
        audio_path: Path to temporary audio file
        reference_text: Transcript of audio
        db: Database session

    Returns:
        Created sample
    """
    import asyncio

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    # Validate and load audio in a single pass, off the event loop
    is_valid, error_msg, audio, sr = await asyncio.to_thread(
        validate_and_load_reference_audio, audio_path
    )
    if not is_valid:
        raise ValueError(f"Invalid reference audio: {error_msg}")

    sample_id = str(uuid.uuid4())
    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    dest_path = profile_dir / f"{sample_id}.wav"
    await asyncio.to_thread(save_audio, audio, str(dest_path), sr)

    db_sample = DBProfileSample(
        id=sample_id,
        profile_id=profile_id,
        audio_path=config.to_storage_path(dest_path),
        reference_text=reference_text,
    )

    db.add(db_sample)

    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(db_sample)

    # Invalidate combined audio cache for this profile
    # Since a new sample was added, any cached combined audio is now stale
    clear_profile_cache(profile_id)

    return ProfileSampleResponse.model_validate(db_sample)


async def get_profile(
    profile_id: str,
    db: Session,
) -> VoiceProfileResponse | None:
    """
    Get a voice profile by ID.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        Profile or None if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None

    return _profile_to_response(profile)


def get_profile_orm_by_name_or_id(
    name_or_id: str,
    db: Session,
) -> DBVoiceProfile | None:
    """Resolve a profile from a user-supplied string that may be either id or name.

    Id is tried first (fast path, matches UUIDs). Name fallback is
    case-insensitive so agents can say "Morgan" regardless of casing.
    """
    if not name_or_id:
        return None
    row = db.query(DBVoiceProfile).filter(DBVoiceProfile.id == name_or_id).first()
    if row is not None:
        return row
    return (
        db.query(DBVoiceProfile)
        .filter(func.lower(DBVoiceProfile.name) == name_or_id.lower())
        .first()
    )


async def get_profile_samples(
    profile_id: str,
    db: Session,
) -> list[ProfileSampleResponse]:
    """
    Get all samples for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        List of samples
    """
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
    return [ProfileSampleResponse.model_validate(s) for s in samples]


async def list_profiles(db: Session) -> list[VoiceProfileResponse]:
    """
    List all voice profiles with generation and sample counts.

    Args:
        db: Database session

    Returns:
        List of profiles
    """
    profiles = db.query(DBVoiceProfile).order_by(DBVoiceProfile.created_at.desc()).all()

    if not profiles:
        return []

    # Batch-fetch generation counts
    gen_counts_rows = (
        db.query(DBGeneration.profile_id, func.count(DBGeneration.id)).group_by(DBGeneration.profile_id).all()
    )
    gen_counts = {row[0]: row[1] for row in gen_counts_rows}

    # Batch-fetch sample counts
    sample_counts_rows = (
        db.query(DBProfileSample.profile_id, func.count(DBProfileSample.id)).group_by(DBProfileSample.profile_id).all()
    )
    sample_counts = {row[0]: row[1] for row in sample_counts_rows}

    return [
        _profile_to_response(
            p,
            generation_count=gen_counts.get(p.id, 0),
            sample_count=sample_counts.get(p.id, 0),
        )
        for p in profiles
    ]


async def update_profile(
    profile_id: str,
    data: VoiceProfileCreate,
    db: Session,
) -> VoiceProfileResponse | None:
    """
    Update a voice profile.

    Args:
        profile_id: Profile ID
        data: Updated profile data
        db: Database session

    Returns:
        Updated profile or None if not found

    Raises:
        ValueError: If a profile with the same name already exists (different profile)
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None

    if profile.name != data.name:
        existing_profile = db.query(DBVoiceProfile).filter_by(name=data.name).first()
        if existing_profile:
            raise ValueError(f"A profile with the name '{data.name}' already exists. Please choose a different name.")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    preset_engine = getattr(profile, "preset_engine", None)
    preset_voice_id = getattr(profile, "preset_voice_id", None)
    design_prompt = getattr(profile, "design_prompt", None)
    default_engine = data.default_engine if data.default_engine is not None else getattr(profile, "default_engine", None)
    # 业务规则：提交 default_engine 表示用户明确重选默认项，此时具体型号也
    # 以本次 payload 为准；旧客户端完全不提交引擎字段时保留已有型号。
    default_model_size = (
        data.default_model_size
        if data.default_engine is not None
        else getattr(profile, "default_model_size", None)
    )

    validation_error = _validate_profile_fields(
        voice_type=voice_type,
        preset_engine=preset_engine,
        preset_voice_id=preset_voice_id,
        design_prompt=design_prompt,
        default_engine=default_engine,
        default_model_size=default_model_size,
    )
    if validation_error:
        raise ValueError(validation_error)

    profile.name = data.name
    profile.description = data.description
    profile.language = data.language
    profile.personality = data.personality
    if data.default_engine is not None:
        profile.default_engine = data.default_engine or None  # empty string → NULL
        profile.default_model_size = default_model_size if data.default_engine else None
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


async def delete_profile(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete a voice profile and all associated data.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return False

    db.query(DBProfileSample).filter_by(profile_id=profile_id).delete()

    db.delete(profile)
    db.commit()

    profile_dir = config.get_profiles_dir() / profile_id
    if profile_dir.exists():
        shutil.rmtree(profile_dir)

    # Clean up combined audio cache files for this profile
    clear_profile_cache(profile_id)

    return True


async def delete_profile_sample(
    sample_id: str,
    db: Session,
) -> bool:
    """
    Delete a profile sample.

    Args:
        sample_id: Sample ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return False

    # Store profile_id before deleting
    profile_id = sample.profile_id

    audio_path = config.resolve_storage_path(sample.audio_path)
    if audio_path is not None and audio_path.exists():
        audio_path.unlink()

    db.delete(sample)
    db.commit()

    # Invalidate combined audio cache for this profile
    # Since the sample set changed, any cached combined audio is now stale
    clear_profile_cache(profile_id)

    return True


async def update_profile_sample(
    sample_id: str,
    reference_text: str,
    db: Session,
) -> ProfileSampleResponse | None:
    """
    Update a profile sample's reference text.

    Args:
        sample_id: Sample ID
        reference_text: Updated reference text
        db: Database session

    Returns:
        Updated sample or None if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return None

    # Store profile_id before updating
    profile_id = sample.profile_id

    sample.reference_text = reference_text
    db.commit()
    db.refresh(sample)

    # Invalidate combined audio cache for this profile
    # Since the reference text changed, cache keys and combined text are now stale
    clear_profile_cache(profile_id)

    return ProfileSampleResponse.model_validate(sample)


async def create_voice_prompt_for_profile(
    profile_id: str,
    db: Session,
    use_cache: bool = True,
    engine: str = "qwen",
) -> dict:
    """
    Create a voice prompt from a profile.

    For cloned profiles: combines all audio samples into a voice prompt.
    For preset profiles: returns the engine-specific preset voice reference.
    For designed profiles: returns the text design prompt (future).

    Args:
        profile_id: Profile ID
        db: Database session
        use_cache: Whether to use cached prompts
        engine: TTS engine to create prompt for

    Returns:
        Voice prompt dictionary
    """
    from ..backends import get_tts_backend_for_engine

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile not found: {profile_id}")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    validate_profile_engine(profile, engine)

    # ── Preset profiles: return engine-specific voice reference ──
    if voice_type == "preset":
        if not profile.preset_engine or not profile.preset_voice_id:
            raise ValueError(f"Preset profile {profile_id} is missing preset engine metadata")
        if profile.preset_engine != engine:
            raise ValueError(
                f"Preset profile {profile_id} only supports engine '{profile.preset_engine}', not '{engine}'"
            )
        return {
            "voice_type": "preset",
            "preset_engine": profile.preset_engine,
            "preset_voice_id": profile.preset_voice_id,
        }

    # ── Designed profiles: return text description (future) ──
    if voice_type == "designed":
        if not profile.design_prompt or not profile.design_prompt.strip():
            raise ValueError(f"Designed profile {profile_id} is missing design_prompt")
        return {
            "voice_type": "designed",
            "design_prompt": profile.design_prompt,
        }

    if engine not in CLONING_ENGINES:
        raise ValueError(f"Engine '{engine}' does not support cloned voice profiles")

    # ── Cloned profiles: create from audio samples ──
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()

    if not samples:
        raise ValueError(f"No samples found for profile {profile_id}")

    tts_model = get_tts_backend_for_engine(engine)

    if len(samples) == 1:
        sample = samples[0]
        sample_audio_path = config.resolve_storage_path(sample.audio_path)
        if sample_audio_path is None:
            raise ValueError(f"Sample audio not found for profile {profile_id}")
        voice_prompt, _ = await tts_model.create_voice_prompt(
            str(sample_audio_path),
            sample.reference_text,
            use_cache=use_cache,
        )
        return voice_prompt

    audio_paths = []
    for sample in samples:
        sample_audio_path = config.resolve_storage_path(sample.audio_path)
        if sample_audio_path is None:
            raise ValueError(f"Sample audio not found for profile {profile_id}")
        audio_paths.append(str(sample_audio_path))
    reference_texts = [s.reference_text for s in samples]

    combined_audio, combined_text = await tts_model.combine_voice_prompts(
        audio_paths,
        reference_texts,
    )

    # Save combined audio to cache directory (persistent)
    # Create a hash of sample IDs to identify this specific combination
    import hashlib

    sample_ids_str = "-".join(sorted([s.id for s in samples]))
    combination_hash = hashlib.md5(sample_ids_str.encode()).hexdigest()[:12]

    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined_path = cache_dir / f"combined_{profile_id}_{combination_hash}.wav"

    save_audio(combined_audio, str(combined_path), 24000)

    voice_prompt, _ = await tts_model.create_voice_prompt(
        str(combined_path),
        combined_text,
        use_cache=use_cache,
    )
    return voice_prompt


async def upload_avatar(
    profile_id: str,
    image_path: str,
    db: Session,
) -> VoiceProfileResponse:
    """
    Upload and process avatar image for a profile.

    Args:
        profile_id: Profile ID
        image_path: Path to uploaded image file
        db: Database session

    Returns:
        Updated profile
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    is_valid, error_msg = validate_image(image_path)
    if not is_valid:
        raise ValueError(error_msg)

    if profile.avatar_path:
        old_avatar = config.resolve_storage_path(profile.avatar_path)
        if old_avatar is not None and old_avatar.exists():
            old_avatar.unlink()

    # Determine file extension from uploaded file
    from PIL import Image

    with Image.open(image_path) as img:
        # Normalize JPEG variants (MPO is multi-picture format from some cameras)
        img_format = img.format
        if img_format in ("MPO", "JPG"):
            img_format = "JPEG"

        ext_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
        ext = ext_map.get(img_format, ".png")

    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_path = profile_dir / f"avatar{ext}"

    process_avatar(image_path, str(output_path))

    profile.avatar_path = config.to_storage_path(output_path)
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


async def delete_avatar(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete avatar image for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        True if deleted, False if not found or no avatar
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile or not profile.avatar_path:
        return False

    avatar_path = config.resolve_storage_path(profile.avatar_path)
    if avatar_path is not None and avatar_path.exists():
        avatar_path.unlink()

    profile.avatar_path = None
    profile.updated_at = datetime.utcnow()

    db.commit()

    return True
