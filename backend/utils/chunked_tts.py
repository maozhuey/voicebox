"""
Chunked TTS generation utilities.

Splits long text into sentence-boundary chunks, generates audio per-chunk
via any TTSBackend, and concatenates with crossfade.  All logic is
engine-agnostic — it wraps the standard ``TTSBackend.generate()`` interface.

Short text (≤ max_chunk_chars) uses the single-shot fast path with zero
overhead.
"""

import logging
import re
import secrets
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("voicebox.chunked-tts")

# Default chunk size in characters.  Can be overridden per-request via
# the ``max_chunk_chars`` field on GenerationRequest.
DEFAULT_MAX_CHUNK_CHARS = 800
MAX_RUNAWAY_RETRIES = 2
MIN_RUNAWAY_RETRY_CHARS = 100

# Common abbreviations that should NOT be treated as sentence endings.
# Lowercase for case-insensitive matching.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "blvd",
        "inc",
        "ltd",
        "corp",
        "dept",
        "est",
        "approx",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.s.a",
        "u.k",
    }
)

# Paralinguistic tags used by Chatterbox Turbo.  The splitter must never
# cut inside one of these.
_PARA_TAG_RE = re.compile(r"\[[^\]]*\]")

# Natural-reading mode treats these tags as editorial timing instructions.
# They are removed before synthesis so the model never tries to pronounce them.
_EXPLICIT_PAUSE_RE = re.compile(
    r"\[停顿\s*(\d+(?:\.\d+)?)\s*(秒|毫秒)\]",
    re.IGNORECASE,
)
_SENTENCE_ENDINGS = frozenset("。！？.!?")
_CLAUSE_ENDINGS = frozenset("，,；;：:、—")
_STRONG_CLAUSE_ENDINGS = frozenset("；;")
_NATURAL_BREATH_CHARS = 32


@dataclass
class ProsodyChunk:
    """A synthesis unit plus the intentional silence that follows it."""

    text: str
    pause_after_ms: int = 0


def _boundary_pause_ms(text: str) -> int:
    """Return the default pause implied by the final visible character."""
    stripped = text.rstrip()
    if not stripped:
        return 0
    if stripped[-1] in _SENTENCE_ENDINGS:
        return 450
    if stripped[-1] in _CLAUSE_ENDINGS:
        return 220
    return 120


def _split_natural_segment(text: str, max_chars: int) -> List[ProsodyChunk]:
    """Split text at every sentence end, then cap long units by clauses."""
    if not text:
        return []

    sentence_units: List[str] = []
    start = 0
    for index, char in enumerate(text):
        if char not in _SENTENCE_ENDINGS:
            continue
        end = index + 1
        while end < len(text) and text[end] in "”’\"』」":
            end += 1
        sentence_units.append(text[start:end])
        start = end
    if start < len(text):
        sentence_units.append(text[start:])

    # A semicolon marks an intentional change of thought even when the whole
    # sentence fits under the character cap. Treat it as a breathing boundary
    # rather than asking one model call to rush across both clauses.
    breath_units: List[str] = []
    for sentence in sentence_units:
        start = 0
        for index, char in enumerate(sentence):
            if char in _STRONG_CLAUSE_ENDINGS:
                breath_units.append(sentence[start : index + 1])
                start = index + 1
        if start < len(sentence):
            breath_units.append(sentence[start:])

    short_breath_units: List[str] = []
    for unit in breath_units:
        if len(unit) <= _NATURAL_BREATH_CHARS:
            short_breath_units.append(unit)
            continue
        start = 0
        for index, char in enumerate(unit):
            if char in "，,":
                short_breath_units.append(unit[start : index + 1])
                start = index + 1
        if start < len(unit):
            short_breath_units.append(unit[start:])

    chunks: List[ProsodyChunk] = []
    for unit in short_breath_units:
        remaining = unit
        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            split_at = max((window.rfind(mark) for mark in _CLAUSE_ENDINGS), default=-1)
            if split_at < 0:
                split_at = window.rfind(" ")
            if split_at < 0:
                split_at = max_chars - 1
            part = remaining[: split_at + 1]
            chunks.append(ProsodyChunk(part, _boundary_pause_ms(part)))
            remaining = remaining[split_at + 1 :]
        if remaining:
            chunks.append(ProsodyChunk(remaining, _boundary_pause_ms(remaining)))

    return chunks


def plan_natural_reading(
    text: str,
    max_chars: int = 70,
) -> List[ProsodyChunk]:
    """Build a rhythm plan while preserving the author's paragraph structure.

    The source text's punctuation, line breaks, blank lines, and explicit
    ``[停顿0.8秒]`` tags are business-level timing signals. Single line breaks
    receive a medium pause, blank lines a paragraph pause, and long sentences
    are split at clauses so cloned voices do not drift during one long breath.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: List[ProsodyChunk] = []

    for part in re.split(r"(\n+)", normalized):
        if not part:
            continue
        if part.startswith("\n"):
            if chunks:
                newline_pause = 1000 if len(part) >= 2 else 650
                chunks[-1].pause_after_ms = max(
                    chunks[-1].pause_after_ms,
                    newline_pause,
                )
            continue
        if not part.strip():
            if chunks:
                chunks[-1].pause_after_ms = max(chunks[-1].pause_after_ms, 1000)
            continue

        cursor = 0
        for match in _EXPLICIT_PAUSE_RE.finditer(part):
            chunks.extend(_split_natural_segment(part[cursor : match.start()], max_chars))
            if chunks:
                value = float(match.group(1))
                pause_ms = int(round(value if match.group(2) == "毫秒" else value * 1000))
                chunks[-1].pause_after_ms = pause_ms
            cursor = match.end()
        chunks.extend(_split_natural_segment(part[cursor:], max_chars))

    if chunks:
        chunks[-1].pause_after_ms = 0
    return chunks


def split_text_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> List[str]:
    """Split *text* at natural boundaries into chunks of at most *max_chars*.

    Priority: sentence-end (``.!?`` not preceded by an abbreviation and not
    inside brackets) → clause boundary (``;:,—``) → whitespace → hard cut.

    Paralinguistic tags like ``[laugh]`` are treated as atomic and will not
    be split across chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]

        # Try to split at the last real sentence ending
        split_pos = _find_last_sentence_end(segment)
        if split_pos == -1:
            split_pos = _find_last_clause_boundary(segment)
        if split_pos == -1:
            split_pos = segment.rfind(" ")
        if split_pos == -1:
            # Absolute fallback: hard cut but avoid splitting inside a tag
            split_pos = _safe_hard_cut(segment, max_chars)

        chunk = remaining[: split_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos + 1 :]

    return chunks


def _find_last_sentence_end(text: str) -> int:
    """Return the index of the last sentence-ending punctuation in *text*.

    Skips periods that follow common abbreviations (``Dr.``, ``Mr.``, etc.)
    and periods inside bracket tags (``[laugh]``).  Also handles CJK
    sentence-ending punctuation (``。！？``).
    """
    best = -1
    # ASCII sentence ends
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        pos = m.start()
        char = text[pos]
        # Skip periods after abbreviations
        if char == ".":
            # Walk backwards to find the preceding word
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1 : pos].lower()
            if word in _ABBREVIATIONS:
                continue
            # Skip decimal numbers (digit immediately before the period)
            if word_start >= 0 and text[word_start].isdigit():
                continue
        # Skip if we're inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    # CJK sentence-ending punctuation
    for m in re.finditer(r"[\u3002\uff01\uff1f]", text):
        if m.start() > best:
            best = m.start()
    return best


def _find_last_clause_boundary(text: str) -> int:
    """Return the index of the last clause-boundary punctuation."""
    best = -1
    for m in re.finditer(r"[;:,\u2014](?:\s|$)", text):
        pos = m.start()
        # Skip if inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    return best


def _inside_bracket_tag(text: str, pos: int) -> bool:
    """Return True if *pos* falls inside a ``[...]`` tag."""
    for m in _PARA_TAG_RE.finditer(text):
        if m.start() < pos < m.end():
            return True
    return False


def _safe_hard_cut(segment: str, max_chars: int) -> int:
    """Find a hard-cut position that doesn't split a ``[tag]``."""
    cut = max_chars - 1
    # Check if the cut falls inside a bracket tag; if so, move before it
    for m in _PARA_TAG_RE.finditer(segment):
        if m.start() < cut < m.end():
            return m.start() - 1 if m.start() > 0 else cut
    return cut


def concatenate_audio_chunks(
    chunks: List[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Concatenate audio arrays with a short crossfade to eliminate clicks.

    Each chunk is expected to be a 1-D float32 ndarray at *sample_rate* Hz.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = np.array(chunks[0], dtype=np.float32, copy=True)

    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        overlap = min(crossfade_samples, len(result), len(chunk))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
            result = np.concatenate([result, chunk[overlap:]])
        else:
            result = np.concatenate([result, chunk])

    return result


def concatenate_audio_with_pauses(
    chunks: List[np.ndarray],
    pauses_ms: List[int],
    sample_rate: int,
) -> np.ndarray:
    """Concatenate generated units with deliberate zero-valued silence."""
    if not chunks:
        return np.array([], dtype=np.float32)

    pieces: List[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        pieces.append(np.asarray(chunk, dtype=np.float32))
        pause_ms = pauses_ms[index] if index < len(pauses_ms) else 0
        pause_samples = int(sample_rate * pause_ms / 1000)
        if pause_samples > 0:
            pieces.append(np.zeros(pause_samples, dtype=np.float32))
    return np.concatenate(pieces)


async def generate_chunked(
    backend,
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    crossfade_ms: int = 50,
    natural_reading: bool = False,
    trim_fn=None,
    runaway_detector=None,
) -> Tuple[np.ndarray, int]:
    """Generate audio with automatic chunking for long text.

    For text shorter than *max_chunk_chars* this is a thin wrapper around
    ``backend.generate()`` with zero overhead.

    For longer text the input is split at natural sentence boundaries,
    each chunk is generated independently, optionally trimmed (useful for
    Chatterbox engines that hallucinate trailing noise), and the results
    are concatenated with a crossfade (or hard cut if *crossfade_ms* is 0).

    Parameters
    ----------
    backend : TTSBackend
        Any backend implementing the ``generate()`` protocol.
    text : str
        Input text (may be arbitrarily long).
    voice_prompt, language, seed, instruct
        Forwarded to ``backend.generate()`` verbatim.
    max_chunk_chars : int
        Maximum characters per chunk (default 800).
    crossfade_ms : int
        Crossfade duration in milliseconds between chunks.  0 for a hard
        cut with no overlap (default 50).
    trim_fn : callable | None
        Optional ``(audio, sample_rate) -> audio`` post-processing
        function applied to each chunk before concatenation (e.g.
        ``trim_tts_output`` for Chatterbox engines).
    runaway_detector : callable | None
        Optional ``(audio, sample_rate) -> bool`` detector. When it flags
        unstable output, the affected text is split in half and retried.

    Returns
    -------
    (audio, sample_rate) : Tuple[np.ndarray, int]
    """
    async def generate_one(
        chunk_text: str,
        chunk_seed: int | None,
        retry_depth: int = 0,
    ) -> tuple[np.ndarray, int]:
        chunk_audio, chunk_sr = await backend.generate(
            chunk_text,
            voice_prompt,
            language,
            chunk_seed,
            instruct,
        )

        if runaway_detector is not None and runaway_detector(chunk_audio, chunk_sr):
            if retry_depth >= MAX_RUNAWAY_RETRIES or len(chunk_text) <= MIN_RUNAWAY_RETRY_CHARS:
                raise RuntimeError(
                    "TTS output remained unstable after retrying smaller text chunks"
                )

            retry_max_chars = max(MIN_RUNAWAY_RETRY_CHARS, len(chunk_text) // 2)
            retry_chunks = split_text_into_chunks(chunk_text, retry_max_chars)
            if len(retry_chunks) <= 1:
                raise RuntimeError("Unable to split unstable TTS output for retry")

            logger.warning(
                "Detected unstable TTS output for %d chars; retrying as %d smaller chunks",
                len(chunk_text),
                len(retry_chunks),
            )
            retry_audio: list[np.ndarray] = []
            for i, retry_text in enumerate(retry_chunks):
                retry_seed = (
                    chunk_seed + ((retry_depth + 1) * 1000) + i
                    if chunk_seed is not None
                    else None
                )
                audio, sample_rate = await generate_one(
                    retry_text,
                    retry_seed,
                    retry_depth + 1,
                )
                retry_audio.append(np.asarray(audio, dtype=np.float32))

            return (
                concatenate_audio_chunks(
                    retry_audio,
                    sample_rate,
                    crossfade_ms=crossfade_ms,
                ),
                sample_rate,
            )

        if trim_fn is not None:
            chunk_audio = trim_fn(chunk_audio, chunk_sr)
        return np.asarray(chunk_audio, dtype=np.float32), chunk_sr

    prosody_chunks = (
        plan_natural_reading(text, min(max_chunk_chars, 70))
        if natural_reading
        else [ProsodyChunk(chunk) for chunk in split_text_into_chunks(text, max_chunk_chars)]
    )
    chunks = [chunk.text for chunk in prosody_chunks]
    natural_seed = seed
    if natural_reading and natural_seed is None:
        # One take gets one unpredictable seed shared by every chunk. This
        # keeps the voice stable inside the take while allowing Regenerate to
        # produce a genuinely different performance.
        natural_seed = secrets.randbelow(2**31)

    if len(chunks) <= 1:
        # Short text — single-shot fast path
        stable_seed = natural_seed if natural_reading else seed
        return await generate_one(chunks[0] if chunks else text, stable_seed)

    # Long text — chunked generation
    logger.info(
        "Splitting %d chars into %d chunks (max %d chars each)",
        len(text),
        len(chunks),
        max_chunk_chars,
    )
    audio_chunks: List[np.ndarray] = []
    sample_rate: int | None = None

    for i, chunk_text in enumerate(chunks):
        logger.info(
            "Generating chunk %d/%d (%d chars)",
            i + 1,
            len(chunks),
            len(chunk_text),
        )
        # Natural-reading chunks reuse one seed because voice identity must stay
        # stable across paragraph boundaries. Standard mode retains its legacy
        # varying-seed behavior for backward-compatible output.
        if natural_reading:
            chunk_seed = natural_seed
        else:
            chunk_seed = (seed + i) if seed is not None else None

        chunk_audio, chunk_sr = await generate_one(
            chunk_text,
            chunk_seed,
        )

        audio_chunks.append(chunk_audio)
        if sample_rate is None:
            sample_rate = chunk_sr

    if natural_reading:
        audio = concatenate_audio_with_pauses(
            audio_chunks,
            [chunk.pause_after_ms for chunk in prosody_chunks],
            sample_rate,
        )
    else:
        audio = concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=crossfade_ms)
    return audio, sample_rate
