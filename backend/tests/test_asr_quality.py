"""Optional real-audio regression for mixed Chinese/English dictation."""
# ruff: noqa: RUF001 -- Chinese punctuation is part of the accuracy fixture.

import hashlib
import os
import re
import unicodedata
from pathlib import Path

import pytest

from backend.services.transcribe import get_whisper_model

REFERENCE_TEXT = """OpenAI 买下数万台 Mac 训计算机使用 Agent，苹果高配售罄，发货排到 16-18 周。DeepMind 净流失 80 人，五家实验室确认流动 274 次。

1. OpenAI 买数万台 Mac 训 Agent，高配售罄
2. 五实验室人才流动：DeepMind 净流失 80 人
3. 美军 GenAI.mil 接入 Grok，获 IL5 认证"""

MATCHED_FIXTURE_SHA256 = "06e0af026db2672679925270bbc02d99b22805e6a4e3c4473db2bb59bca55264"
CONTEXT_PROMPT = (
    "本文讨论 OpenAI、Mac、计算机使用 Agent、苹果、DeepMind、"
    "GenAI.mil、Grok、IL5；保留数字、英文专名和中文标点。"
)


def normalize_spoken_content(text: str) -> str:
    """Normalize ASR content without scoring document-only list markers."""
    without_list_markers = re.sub(r"(?m)^\s*[1-9][.\u3001]\s*", "", text)
    normalized = unicodedata.normalize("NFKC", without_list_markers).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def normalized_character_accuracy(reference: str, transcript: str) -> float:
    normalized_reference = normalize_spoken_content(reference)
    normalized_transcript = normalize_spoken_content(transcript)
    denominator = max(len(normalized_reference), len(normalized_transcript), 1)
    return max(0.0, 1.0 - (_edit_distance(normalized_reference, normalized_transcript) / denominator))


def test_character_accuracy_metric_ignores_punctuation_case_and_list_markers():
    reference = "1. OpenAI 接入 Grok，获 IL5 认证。"
    transcript = "openai接入grok获il5认证"

    assert normalized_character_accuracy(reference, transcript) == 1.0


@pytest.mark.asr_e2e
@pytest.mark.asyncio
async def test_mixed_chinese_english_fixture_reaches_ninety_percent():
    """Run explicitly with VOICEBOX_ASR_FIXTURE; personal audio is not committed."""
    fixture_value = os.getenv("VOICEBOX_ASR_FIXTURE")
    if not fixture_value:
        pytest.skip("set VOICEBOX_ASR_FIXTURE to the matched 38.867-second capture")

    fixture = Path(fixture_value)
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == MATCHED_FIXTURE_SHA256

    whisper = get_whisper_model()
    result = await whisper.transcribe(
        str(fixture),
        language="zh",
        model_size="large",
        initial_prompt=CONTEXT_PROMPT,
    )

    transcript = result.text
    assert normalized_character_accuracy(REFERENCE_TEXT, transcript) >= 0.90
    # The source contains two summary sentences plus three list items; losing all
    # boundaries would recreate the user's original unreadable no-punctuation result.
    assert transcript.count("。") >= 4
    assert any(separator in transcript for separator in ("，", ",", "、"))
