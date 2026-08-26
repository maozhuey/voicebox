"""Regression tests for the task-setting reading switch."""

# Chinese punctuation is intentional in the user-facing regression fixture.
# ruff: noqa: RUF001

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.models import GenerationRequest, VoiceProfileCreate
from backend.routes.generations import (
    build_cosyvoice_instruction,
    prepare_generation_content,
    snapshot_cosyvoice_configuration,
)
from backend.services.instruction_summary import COSYVOICE_STYLE_MAX_CHARS


def test_profile_task_setting_is_limited_to_500_characters():
    valid = VoiceProfileCreate(name="voice", personality="a" * 500)
    assert len(valid.personality or "") == 500

    with pytest.raises(ValidationError):
        VoiceProfileCreate(name="voice", personality="a" * 501)


def test_cosyvoice_configuration_snapshot_keeps_only_effective_controls():
    instruct_request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="cosyvoice",
        cosyvoice_mode="instruct",
        dialect="henan",
    )
    reference_request = instruct_request.model_copy(update={"cosyvoice_mode": "reference"})

    assert snapshot_cosyvoice_configuration(instruct_request, "cosyvoice") == ("instruct", "henan")
    assert snapshot_cosyvoice_configuration(reference_request, "cosyvoice") == ("reference", None)
    assert snapshot_cosyvoice_configuration(instruct_request, "qwen") == (None, None)


@pytest.mark.asyncio
async def test_task_setting_keeps_long_script_verbatim_for_cloned_qwen_voice():
    script = "第一段详细介绍油蟠桃的果形和果重。\n\n" * 80
    request = GenerationRequest(
        profile_id="voice-id",
        text=script,
        language="zh",
        engine="qwen",
        personality=True,
    )
    profile = SimpleNamespace(personality="一位说话朴实的老果农。")

    text, instruct, source = await prepare_generation_content(request, profile)

    assert text == script
    assert instruct is None
    assert source == "personality_reading"


@pytest.mark.asyncio
async def test_task_setting_keeps_explicit_custom_voice_instruction():
    request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="qwen_custom_voice",
        instruct="语速稍慢，语气自然。",
        personality=True,
    )
    profile = SimpleNamespace(personality="一位说话朴实的老果农。")

    text, instruct, source = await prepare_generation_content(request, profile)

    assert text == request.text
    assert instruct == request.instruct
    assert source == "personality_reading"


@pytest.mark.asyncio
async def test_cosyvoice_henan_dialect_is_combined_with_delivery_instruction():
    request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="cosyvoice",
        cosyvoice_mode="instruct",
        dialect="henan",
        instruct="语速稍慢，语气自然。",
    )
    profile = SimpleNamespace(personality=None)

    text, instruct, source = await prepare_generation_content(request, profile)

    assert text == request.text
    assert instruct == "语速稍慢，语气自然。\n请用河南话表达。"
    assert source == "manual"


@pytest.mark.asyncio
async def test_cosyvoice_long_character_setting_cannot_replace_spoken_script(monkeypatch):
    script = "这是唯一应该被朗读的正文。"
    character_setting = (
        "使用自然、沉稳、可信的成年男声，音高偏中低，普通话清晰自然。"
        "采用农产品种植知识型短视频口播风格，耐心介绍，不要播音腔或叫卖腔。"
        "每讲完一个果实特点、数字或结论，都自然停顿一下。"
    )
    request = GenerationRequest(
        profile_id="voice-id",
        text=script,
        language="zh",
        engine="cosyvoice",
        cosyvoice_mode="instruct",
        dialect="henan",
        instruct=character_setting,
    )
    profile = SimpleNamespace(personality=character_setting)

    async def summarize(_instruct):
        return "成年男声，低沉可信，语速稍慢，自然停顿"

    monkeypatch.setattr(
        "backend.routes.generations.summarize_cosyvoice_style_instruction",
        summarize,
    )

    text, instruct, _ = await prepare_generation_content(request, profile)

    assert text == script
    assert instruct != character_setting
    assert instruct.endswith("请用河南话表达。")
    assert "普通话" not in instruct
    assert len(instruct) <= len("请用河南话表达。\n") + COSYVOICE_STYLE_MAX_CHARS


def test_cosyvoice_selected_dialect_removes_conflicting_language_clauses():
    instruct = build_cosyvoice_instruction(
        "henan",
        "普通话发音清楚自然。请用四川话表达。语速稍慢，语气亲切。",
    )

    assert instruct == "语速稍慢，语气亲切。\n请用河南话表达。"
    assert "普通话" not in instruct
    assert "四川话" not in instruct


@pytest.mark.asyncio
async def test_cosyvoice_reference_following_is_default_and_ignores_stale_instruction():
    request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="cosyvoice",
        dialect="henan",
        instruct="语速稍慢，语气自然。",
    )
    profile = SimpleNamespace(personality=None)

    _, instruct, _ = await prepare_generation_content(request, profile)

    assert request.cosyvoice_mode == "reference"
    assert instruct is None


@pytest.mark.asyncio
async def test_cosyvoice_instruct_mode_defaults_to_mandarin():
    request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="cosyvoice",
        cosyvoice_mode="instruct",
    )
    profile = SimpleNamespace(personality=None)

    _, instruct, _ = await prepare_generation_content(request, profile)

    assert instruct == "请用普通话表达。"


@pytest.mark.asyncio
async def test_dialect_is_ignored_for_non_cosyvoice_engines():
    request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="qwen",
        dialect="sichuan",
    )
    profile = SimpleNamespace(personality=None)

    _, instruct, _ = await prepare_generation_content(request, profile)

    assert instruct is None
