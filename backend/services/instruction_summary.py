"""Compile detailed voice-profile prose into safe CosyVoice controls."""

# Chinese punctuation is part of the prompts sent to the local model.
# ruff: noqa: RUF001

import logging
import re

from . import llm as llm_service

logger = logging.getLogger(__name__)

INSTRUCTION_SUMMARY_MODEL_SIZE = "1.7B"
COSYVOICE_STYLE_MAX_CHARS = 48
DEFAULT_COSYVOICE_STYLE = "自然、清晰地朗读"
_EMPTY_SUMMARIES = {"", "无", "无可用指令", "没有", "null", "none"}
_DIALECT_CLAUSE_PATTERN = re.compile(
    r"(?:请)?(?:用|使用|采用|改用)?(?:标准|地道|纯正)?"
    r"(?:普通话|河南话|四川话)[^\uFF0C,\u3002.!\uFF01?\uFF1F\uFF1B;\n]{0,20}"
    r"[\uFF0C,\u3002.!\uFF01?\uFF1F\uFF1B;]?"
)
_OUTPUT_PREFIX_PATTERN = re.compile(
    r"^(?:总结|输出|指令|朗读指令|语音指令|风格指令)\s*[:：]\s*",
    re.IGNORECASE,
)
_NON_AUDIBLE_CLAUSE_PATTERN = re.compile(
    r"(?:内容|文案|信息|事实|数字|产品|主题|受众)"
    r".*(?:完整|准确|保留|遗漏)|(?:不要|不)遗漏|(?:逐字|完整)朗读"
)

_SUMMARY_SYSTEM_PROMPT = """你是本地语音合成系统的“朗读风格指令编译器”。
将用户输入的详细人物设定压缩为一句简短的中文语音合成控制指令。

只保留能直接听出来的特征：大致年龄与性别感、音色、音高、语速、情绪强度、节奏、停顿、重音和口播风格。
必须删除主题、产品、事实、数字、任务目标、受众、营销要求和任何可能被当成正文的句子。
不要输出普通话、河南话或四川话要求，方言会由系统另行添加。
不要改写、续写或朗读输入内容。
只输出最终指令，不要标题、引号、解释、列表或 Markdown。
最多 40 个汉字；如果没有可听的风格信息，输出“自然、清晰地朗读”。"""

_SUMMARY_EXAMPLES = [
    (
        "40岁左右的河南成年男性，声音沉稳可信，音高偏低，语速稍慢。"
        "介绍农产品时像有经验的种植户，不要广告腔，重点数据稍微加重，句间自然停顿。",
        "成年男声，低沉可信，语速稍慢，自然停顿，重点适度加重",
    ),
    (
        "她性格开朗，喜欢旅行和音乐。用年轻女声，轻快亲切，语速略快，不要过度夸张。",
        "年轻女声，轻快亲切，语速略快，表达克制",
    ),
    (
        "介绍油蟠桃的果重、甜度和成熟期，面向种植户做短视频。",
        "自然、清晰地朗读",
    ),
]


def clean_cosyvoice_style_source(instruct: str | None) -> str:
    """Remove control markers and dialect clauses owned by the dropdown."""
    if not instruct:
        return ""
    cleaned = instruct.replace("<|endofprompt|>", " ")
    cleaned = _DIALECT_CLAUSE_PATTERN.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" \uFF0C,\uFF1B;")


def _truncate_at_phrase_boundary(text: str) -> str:
    if len(text) <= COSYVOICE_STYLE_MAX_CHARS:
        return text

    shortened = text[:COSYVOICE_STYLE_MAX_CHARS]
    boundary = max(shortened.rfind(mark) for mark in "\uFF0C,\u3002.!\uFF01?\uFF1F\uFF1B;")
    if boundary >= COSYVOICE_STYLE_MAX_CHARS // 2:
        shortened = shortened[: boundary + 1]
    return shortened.strip(" \uFF0C,\uFF1B;")


def sanitize_cosyvoice_style_summary(summary: str | None) -> str:
    """Normalize untrusted LLM output before it reaches CosyVoice."""
    if not summary:
        return ""
    cleaned = summary.replace("```", " ").replace("<|endofprompt|>", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" “”\"'`")
    cleaned = _OUTPUT_PREFIX_PATTERN.sub("", cleaned)
    cleaned = _DIALECT_CLAUSE_PATTERN.sub(" ", cleaned)
    # Convert object-specific emphasis into a reusable delivery control, then
    # drop clauses about factual completeness: those describe the writing task,
    # not an audible voice characteristic, and are prone to prompt recitation.
    cleaned = re.sub(
        r"重点(?:数据|数字|结论|信息)?(?:稍微|适度)?加重",
        "重点适度加重",
        cleaned,
    )
    clauses = re.split(r"([\uFF0C,\u3002.!\uFF01?\uFF1F\uFF1B;])", cleaned)
    audible_parts: list[str] = []
    for index in range(0, len(clauses), 2):
        clause = clauses[index].strip()
        if not clause or _NON_AUDIBLE_CLAUSE_PATTERN.search(clause):
            continue
        separator = clauses[index + 1] if index + 1 < len(clauses) else ""
        audible_parts.append(clause + separator)
    cleaned = "".join(audible_parts)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" “”\"'`，,；;")
    if cleaned.lower() in _EMPTY_SUMMARIES:
        return ""
    return _truncate_at_phrase_boundary(cleaned)


async def summarize_cosyvoice_style_instruction(instruct: str | None) -> str:
    """Summarize long profile prose with local Qwen3 and fail closed safely.

    Business rule: the editable 500-character personality remains the source
    of truth, but CosyVoice must receive only an audible delivery description.
    Concise instructions are preserved verbatim; long prose is semantically
    compressed by the local LLM instead of being cut at an arbitrary offset.
    """
    cleaned = clean_cosyvoice_style_source(instruct)
    if len(cleaned) <= COSYVOICE_STYLE_MAX_CHARS:
        return cleaned

    backend = llm_service.get_llm_model()
    try:
        output = await backend.generate(
            prompt=cleaned,
            system=_SUMMARY_SYSTEM_PROMPT,
            max_tokens=96,
            temperature=0.0,
            model_size=INSTRUCTION_SUMMARY_MODEL_SIZE,
            examples=_SUMMARY_EXAMPLES,
        )
        summary = sanitize_cosyvoice_style_summary(output)
        if summary:
            return summary
        logger.warning("CosyVoice instruction summarizer returned an empty result")
    except Exception:
        # AI enhancement must never prevent speech generation. Do not fall back
        # to truncating the original prose: even a bounded fragment can still
        # look like body text and recreate the prompt-recitation bug.
        logger.exception("CosyVoice instruction summarization failed; using safe fallback")

    return DEFAULT_COSYVOICE_STYLE
