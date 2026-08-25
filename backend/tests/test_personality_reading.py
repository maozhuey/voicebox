"""Regression tests for the task-setting reading switch."""

# Chinese punctuation is intentional in the user-facing regression fixture.
# ruff: noqa: RUF001

from types import SimpleNamespace

from backend.models import GenerationRequest
from backend.routes.generations import prepare_generation_content


def test_task_setting_keeps_long_script_verbatim_for_cloned_qwen_voice():
    script = "第一段详细介绍油蟠桃的果形和果重。\n\n" * 80
    request = GenerationRequest(
        profile_id="voice-id",
        text=script,
        language="zh",
        engine="qwen",
        personality=True,
    )
    profile = SimpleNamespace(personality="一位说话朴实的老果农。")

    text, instruct, source = prepare_generation_content(request, profile)

    assert text == script
    assert instruct is None
    assert source == "personality_reading"


def test_task_setting_keeps_explicit_custom_voice_instruction():
    request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="qwen_custom_voice",
        instruct="语速稍慢，语气自然。",
        personality=True,
    )
    profile = SimpleNamespace(personality="一位说话朴实的老果农。")

    text, instruct, source = prepare_generation_content(request, profile)

    assert text == request.text
    assert instruct == request.instruct
    assert source == "personality_reading"


def test_cosyvoice_henan_dialect_is_combined_with_delivery_instruction():
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

    text, instruct, source = prepare_generation_content(request, profile)

    assert text == request.text
    assert instruct == "请用河南话表达。\n语速稍慢，语气自然。"
    assert source == "manual"


def test_cosyvoice_reference_following_is_default_and_ignores_stale_instruction():
    request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="cosyvoice",
        dialect="henan",
        instruct="语速稍慢，语气自然。",
    )
    profile = SimpleNamespace(personality=None)

    _, instruct, _ = prepare_generation_content(request, profile)

    assert request.cosyvoice_mode == "reference"
    assert instruct is None


def test_cosyvoice_instruct_mode_defaults_to_mandarin():
    request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="cosyvoice",
        cosyvoice_mode="instruct",
    )
    profile = SimpleNamespace(personality=None)

    _, instruct, _ = prepare_generation_content(request, profile)

    assert instruct == "请用普通话表达。"


def test_dialect_is_ignored_for_non_cosyvoice_engines():
    request = GenerationRequest(
        profile_id="voice-id",
        text="请完整朗读这段文案。",
        language="zh",
        engine="qwen",
        dialect="sichuan",
    )
    profile = SimpleNamespace(personality=None)

    _, instruct, _ = prepare_generation_content(request, profile)

    assert instruct is None
