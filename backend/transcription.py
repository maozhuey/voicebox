"""Shared timestamped transcription result contracts."""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    """One Whisper-native text segment on the source audio timeline."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def validate_timeline(self):
        if self.end_ms < self.start_ms:
            raise ValueError("segment end timestamp precedes start timestamp")
        return self


class TranscriptionResult(BaseModel):
    """Raw Whisper text and timestamp segments produced by the same pass."""

    text: str
    segments: list[TranscriptSegment]

    @property
    def timestamped_text(self) -> str:
        return format_timestamped_transcript(self.segments)


def _read_value(segment: Any, key: str) -> Any:
    if isinstance(segment, Mapping):
        return segment.get(key)
    return getattr(segment, key, None)


def build_transcription_result(
    text: str,
    raw_segments: Iterable[Any],
    *,
    audio_duration_ms: int | None = None,
) -> TranscriptionResult:
    """Validate Whisper segments and normalize them to the source timeline."""

    if audio_duration_ms is not None and audio_duration_ms <= 0:
        raise ValueError("Audio duration must be positive")

    normalized: list[TranscriptSegment] = []
    for raw in raw_segments or []:
        segment_text = str(_read_value(raw, "text") or "").strip()
        if not segment_text:
            continue

        start = _read_value(raw, "start")
        end = _read_value(raw, "end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
        ):
            raise ValueError("Invalid Whisper timestamp segment")

        start_ms = round(start * 1000)
        end_ms = round(end * 1000)
        if audio_duration_ms is not None:
            # Whisper decodes short files inside a padded 30-second window and
            # may report 29.98 seconds as the last segment end. The exported
            # source timeline must never point beyond the actual recording.
            if start_ms >= audio_duration_ms:
                continue
            end_ms = min(end_ms, audio_duration_ms)

        normalized.append(TranscriptSegment(start_ms=start_ms, end_ms=end_ms, text=segment_text))

    normalized.sort(key=lambda segment: (segment.start_ms, segment.end_ms))
    if not normalized:
        raise ValueError("Whisper did not return valid timestamped segments")

    raw_text = str(text or "").strip()
    if not raw_text:
        raise ValueError("Whisper did not return transcript text")
    return TranscriptionResult(text=raw_text, segments=normalized)


def parse_stored_segments(value: str | None) -> list[TranscriptSegment]:
    """Read persisted segments without allowing a corrupt legacy row to break lists."""

    if not value:
        return []
    try:
        payload = json.loads(value)
        if not isinstance(payload, list):
            return []
        segments = [TranscriptSegment.model_validate(item) for item in payload]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return sorted(segments, key=lambda segment: (segment.start_ms, segment.end_ms))


def serialize_segments(segments: Sequence[TranscriptSegment]) -> str:
    return json.dumps(
        [segment.model_dump() for segment in segments],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _format_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_timestamped_transcript(segments: Sequence[TranscriptSegment]) -> str:
    """Render the portable timestamped raw-text format used by API and export."""

    return "\n".join(
        f"[{_format_timestamp(segment.start_ms)} --> {_format_timestamp(segment.end_ms)}] {segment.text.strip()}"
        for segment in sorted(segments, key=lambda item: (item.start_ms, item.end_ms))
        if segment.text.strip()
    )
