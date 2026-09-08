"""Regression coverage for binary-extraction failures during generation.

The PyInstaller sidecar bundles every backend module in a single zlib-compressed
PYZ archive. When that archive is corrupted, ``get_tts_backend_for_engine``
raises ``zlib.error`` before inference ever starts. Without dedicated handling,
that raw zlib message would land in ``Generation.error`` and the client would
fall back to the generic "生成过程中发生错误" toast.

These tests pin the contract: a binary-extraction failure must produce a
stable ``BINARY_MODULE_EXTRACTION_FAILED`` error code with a diagnostic id,
and an unrelated exception must keep its original error message.
"""

from __future__ import annotations

import json
import zlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import backends, config
from backend.database import Base, Generation, VoiceProfile
from backend.services import generation as generation_service
from backend.services.generation_diagnostics import BINARY_MODULE_EXTRACTION_FAILED


def _build_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'binary-extract.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        session.add(VoiceProfile(id="voice-id", name="预设-Vivian", language="zh"))
        session.add(
            Generation(
                id="generation-id",
                profile_id="voice-id",
                text="采集规范的内容传给 AI 进行分析",
                language="zh",
                audio_path="",
                duration=0,
                status="loading_model",
                engine="qwen_custom_voice",
                model_size="1.7B",
            )
        )
        session.commit()
    return session_factory


class _IdleBackend:
    @staticmethod
    def is_loaded() -> bool:
        return True


class _TaskManager:
    def __init__(self) -> None:
        self.completed: list[str] = []

    def complete_generation(self, generation_id: str) -> None:
        self.completed.append(generation_id)


async def _fake_acquire_runtime(_engine: str):
    raise NotImplementedError("acquire_tts_runtime is patched at module level")


# ``acquire_tts_runtime`` is called as ``async with acquire_tts_runtime(engine):``
# so the patched object must already be an async context manager; a coroutine
# is rejected by the ``async with`` protocol and would mask the real failure
# we are trying to exercise.
class _IdleLease:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *_args):
        return None


def _fake_acquire_runtime_factory(_engine: str):
    return _IdleLease()


async def _fake_load_engine_model(_engine: str, _model_size: str) -> None:
    return None


async def _fake_create_voice_prompt(*_args, **_kwargs):
    return object()


def _install_patches(monkeypatch, session_factory, *, backends_patch):
    monkeypatch.setattr(generation_service, "get_db", lambda: iter([session_factory()]))
    monkeypatch.setattr(generation_service, "get_task_manager", lambda: _TaskManager())
    monkeypatch.setattr(generation_service, "_notify_speak_end", lambda *_a, **_k: None)
    # ``acquire_tts_runtime`` is re-exported into ``generation`` via
    # ``from ..backends import ...``; patch the canonical location so the
    # import lookup inside the production module picks up our stub.
    monkeypatch.setattr(backends, "acquire_tts_runtime", _fake_acquire_runtime_factory)
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", backends_patch)
    monkeypatch.setattr(backends, "load_engine_model", _fake_load_engine_model)
    from backend.services import profiles

    monkeypatch.setattr(profiles, "create_voice_prompt_for_profile", _fake_create_voice_prompt)


@pytest.mark.asyncio
async def test_zlib_corruption_is_mapped_to_stable_diagnostic_code(tmp_path, monkeypatch):
    session_factory = _build_db(tmp_path)
    monkeypatch.setattr(config, "_data_dir", tmp_path)

    def corrupt_backend(_engine: str):
        # The real failure path lives inside the module loader; raising
        # ``zlib.error`` here exercises the same except branch.
        raise zlib.error("Error -3 while decompressing data: incorrect header check")

    _install_patches(monkeypatch, session_factory, backends_patch=corrupt_backend)

    result = await generation_service.run_generation(
        generation_id="generation-id",
        profile_id="voice-id",
        text="采集规范的内容传给 AI 进行分析",
        language="zh",
        engine="qwen_custom_voice",
        model_size="1.7B",
        seed=None,
        mode="generate",
    )

    assert result.status == "failed"

    with session_factory() as session:
        failed = session.query(Generation).filter_by(id="generation-id").one()
        # Stable code + diagnostic id; raw zlib message never reaches history.
        assert failed.error.startswith(f"{BINARY_MODULE_EXTRACTION_FAILED}:")
        assert "incorrect header" not in (failed.error or "")
        assert "Traceback" not in (failed.error or "")

    # Diagnostic record carries the subtype for support correlation and
    # deliberately omits the user-supplied text.
    record = json.loads((tmp_path / "logs" / "generation-diagnostics.jsonl").read_text().strip().splitlines()[-1])
    assert record["kind"] == "binary_module_extraction_failed"
    assert record["failure_subtype"] == "zlib_corruption"
    assert record["engine"] == "qwen_custom_voice"
    assert record["model_size"] == "1.7B"
    assert "采集规范" not in json.dumps(record, ensure_ascii=False)


@pytest.mark.asyncio
async def test_module_not_found_inside_cause_chain_is_still_mapped(tmp_path, monkeypatch):
    session_factory = _build_db(tmp_path)
    monkeypatch.setattr(config, "_data_dir", tmp_path)

    def missing_module(_engine: str):
        inner = ModuleNotFoundError("No module named 'backend.backends.qwen_custom_voice_backend'")
        outer = ImportError("Loader failed")
        outer.__cause__ = inner
        raise outer

    _install_patches(monkeypatch, session_factory, backends_patch=missing_module)

    result = await generation_service.run_generation(
        generation_id="generation-id",
        profile_id="voice-id",
        text="缺失依赖测试",
        language="zh",
        engine="qwen_custom_voice",
        model_size="1.7B",
        seed=None,
        mode="generate",
    )

    assert result.status == "failed"

    with session_factory() as session:
        failed = session.query(Generation).filter_by(id="generation-id").one()
        assert failed.error.startswith(f"{BINARY_MODULE_EXTRACTION_FAILED}:")

    record = json.loads((tmp_path / "logs" / "generation-diagnostics.jsonl").read_text().strip().splitlines()[-1])
    assert record["failure_subtype"] == "module_not_found"


@pytest.mark.asyncio
async def test_unrelated_runtime_error_keeps_original_message(tmp_path, monkeypatch):
    session_factory = _build_db(tmp_path)
    monkeypatch.setattr(config, "_data_dir", tmp_path)

    def unrelated_failure(_engine: str):
        raise RuntimeError("model produced empty audio")

    _install_patches(monkeypatch, session_factory, backends_patch=unrelated_failure)

    result = await generation_service.run_generation(
        generation_id="generation-id",
        profile_id="voice-id",
        text="无关错误测试",
        language="zh",
        engine="qwen_custom_voice",
        model_size="1.7B",
        seed=None,
        mode="generate",
    )

    assert result.status == "failed"

    with session_factory() as session:
        failed = session.query(Generation).filter_by(id="generation-id").one()
        # The original message is preserved; the binary-extraction mapper only
        # fires for the zlib / import family, never for arbitrary errors.
        assert failed.error == "model produced empty audio"
        assert BINARY_MODULE_EXTRACTION_FAILED not in (failed.error or "")

    # No diagnostic is written for unrelated failures.
    diagnostics_path = tmp_path / "logs" / "generation-diagnostics.jsonl"
    assert not diagnostics_path.exists()
