"""Tests for the local-AI CosyVoice instruction compiler."""

# Chinese punctuation is intentional in the user-facing regression fixtures.
# ruff: noqa: RUF001

from types import SimpleNamespace

import pytest

from backend.services import instruction_summary


@pytest.mark.asyncio
async def test_short_instruction_is_preserved_without_loading_llm(monkeypatch):
    def unexpected_backend():
        raise AssertionError("short instructions must not load the LLM")

    monkeypatch.setattr(instruction_summary.llm_service, "get_llm_model", unexpected_backend)

    result = await instruction_summary.summarize_cosyvoice_style_instruction(
        "语速稍慢，语气自然。"
    )

    assert result == "语速稍慢，语气自然。"


@pytest.mark.asyncio
async def test_long_instruction_is_semantically_summarized_by_local_llm(monkeypatch):
    captured = {}

    async def generate(**kwargs):
        captured.update(kwargs)
        return (
            "朗读指令：成年男声，低沉可信，语速稍慢，重点数据加重，"
            "自然停顿，内容完整。请用四川话表达。"
        )

    monkeypatch.setattr(
        instruction_summary.llm_service,
        "get_llm_model",
        lambda: SimpleNamespace(generate=generate),
    )
    source = (
        "请用河南话表达。40岁左右的成年男性，声音沉稳可信，音高偏低，语速稍慢。"
        "介绍油蟠桃时像有经验的种植户，不要广告腔；重点数字适度加重，句间自然停顿。"
    )

    result = await instruction_summary.summarize_cosyvoice_style_instruction(source)

    assert result == "成年男声，低沉可信，语速稍慢，重点适度加重，自然停顿"
    assert captured["model_size"] == "1.7B"
    assert captured["temperature"] == 0.0
    assert "河南话" not in captured["prompt"]
    assert len(result) <= instruction_summary.COSYVOICE_STYLE_MAX_CHARS


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_bounded_instruction(monkeypatch):
    async def generate(**_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        instruction_summary.llm_service,
        "get_llm_model",
        lambda: SimpleNamespace(generate=generate),
    )

    result = await instruction_summary.summarize_cosyvoice_style_instruction(
        "使用自然、沉稳、可信的成年男声，音高偏中低，语速稍慢，语气亲切。" * 5
    )

    assert result == instruction_summary.DEFAULT_COSYVOICE_STYLE
