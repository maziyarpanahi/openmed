"""Deterministic, content-free audio-format distribution summaries.

Aggregates validated per-asset audio descriptors into deployment-planning
counts: channel, sample-rate, bit-depth, duration-bucket, and container
format categories, with small-cell suppression and deterministic sorting.
The summary never carries raw samples, filenames, paths, identifiers,
transcripts, or free-form metadata; every record is a validated
:class:`AudioFormatRecord` whose fields are closed or bounded values only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .asset_manifest import MAX_MANIFEST_DURATION_SECONDS

__all__ = [
    "AUDIO_FORMAT_SUMMARY_SCHEMA_VERSION",
    "DEFAULT_MIN_CELL_COUNT",
    "SUPPRESSED_CATEGORY",
    "AudioFormat",
    "AudioFormatRecord",
    "AudioFormatSummary",
    "AudioFormatSummaryError",
    "summarize_audio_formats",
]

AUDIO_FORMAT_SUMMARY_SCHEMA_VERSION: Final = 1
DEFAULT_MIN_CELL_COUNT: Final = 3
SUPPRESSED_CATEGORY: Final = "suppressed"
_MAX_CHANNELS: Final = 64
_MAX_BIT_DEPTH: Final = 64
_MAX_SAMPLE_RATE_HZ: Final = (1 << 32) - 1

# Inclusive lower bounds, in seconds, of the fixed duration buckets.
_DURATION_BUCKET_EDGES: Final = (
    (0.0, "under_1s"),
    (1.0, "1s_to_10s"),
    (10.0, "10s_to_1m"),
    (60.0, "1m_to_10m"),
    (600.0, "10m_or_more"),
)


class AudioFormatSummaryError(ValueError):
    """Value-free failure for invalid audio records or summary arguments."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class AudioFormat(str, Enum):
    """Closed set of supported audio container categories."""

    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"
    M4A = "m4a"
    OGG = "ogg"


def _bounded_int(value: Any, category: str, maximum: int, *, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise AudioFormatSummaryError(category)


@dataclass(frozen=True, slots=True)
class AudioFormatRecord:
    """One asset's validated audio facts, with no content or identifiers.

    No field accepts raw samples, filenames, paths, identifiers, transcripts,
    or free-form metadata; the dataclass has exactly these five closed or
    bounded fields.
    """

    format: AudioFormat
    channels: int
    sample_rate_hz: int
    bit_depth: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.format, AudioFormat):
            raise AudioFormatSummaryError("audio_format_format_invalid")
        _bounded_int(self.channels, "audio_format_channels_invalid", _MAX_CHANNELS)
        _bounded_int(
            self.sample_rate_hz,
            "audio_format_sample_rate_invalid",
            _MAX_SAMPLE_RATE_HZ,
        )
        _bounded_int(self.bit_depth, "audio_format_bit_depth_invalid", _MAX_BIT_DEPTH)
        if (
            type(self.duration_seconds) not in (int, float)
            or not math.isfinite(self.duration_seconds)
            or not 0 <= self.duration_seconds <= MAX_MANIFEST_DURATION_SECONDS
        ):
            raise AudioFormatSummaryError("audio_format_duration_invalid")
        object.__setattr__(self, "format", AudioFormat(self.format))


@dataclass(frozen=True, slots=True)
class CategoryCount:
    """One aggregated category with its count, or the suppressed pool."""

    category: int | str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "count": self.count}


def _duration_bucket(duration_seconds: float) -> str:
    label = _DURATION_BUCKET_EDGES[-1][1]
    for edge, bucket in _DURATION_BUCKET_EDGES:
        if duration_seconds >= edge:
            label = bucket
        else:
            break
    return label


_SUMMARY_FIELDS: Final = (
    "schema_version",
    "total_assets",
    "by_format",
    "by_channels",
    "by_sample_rate_hz",
    "by_bit_depth",
    "by_duration_bucket",
)


@dataclass(frozen=True, slots=True)
class AudioFormatSummary:
    """Deterministic, content-free distribution artifact for a run.

    Each breakdown is sorted by category and hides cells below the small-cell
    threshold in a single trailing ``suppressed`` pool whose count is the sum
    of the suppressed cells. ``to_dict()``/``to_json()`` are byte-identical
    for reordered inputs.
    """

    schema_version: int
    total_assets: int
    by_format: tuple[CategoryCount, ...]
    by_channels: tuple[CategoryCount, ...]
    by_sample_rate_hz: tuple[CategoryCount, ...]
    by_bit_depth: tuple[CategoryCount, ...]
    by_duration_bucket: tuple[CategoryCount, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "schema_version": self.schema_version,
            "total_assets": self.total_assets,
            "by_format": [entry.to_dict() for entry in self.by_format],
            "by_channels": [entry.to_dict() for entry in self.by_channels],
            "by_sample_rate_hz": [entry.to_dict() for entry in self.by_sample_rate_hz],
            "by_bit_depth": [entry.to_dict() for entry in self.by_bit_depth],
            "by_duration_bucket": [
                entry.to_dict() for entry in self.by_duration_bucket
            ],
        }

    def to_json(self) -> str:
        """Serialize with deterministic key order and no insignificant space."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=False,
            separators=(",", ":"),
        )


def _breakdown(keys: list[Any], min_cell_count: int) -> tuple[CategoryCount, ...]:
    counts: dict[Any, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    visible = [
        CategoryCount(category=key, count=count)
        for key, count in sorted(counts.items())
        if count >= min_cell_count
    ]
    suppressed_total = sum(count for count in counts.values() if count < min_cell_count)
    if suppressed_total:
        visible.append(
            CategoryCount(category=SUPPRESSED_CATEGORY, count=suppressed_total)
        )
    return tuple(visible)


def summarize_audio_formats(
    records: Iterable[AudioFormatRecord],
    *,
    min_cell_count: Any = DEFAULT_MIN_CELL_COUNT,
) -> AudioFormatSummary:
    """Aggregate validated records into a deterministic distribution summary.

    Every record is validated before anything is aggregated: an invalid
    record raises :class:`AudioFormatSummaryError` and no summary -- partial
    or otherwise -- is returned. Categories with fewer than
    ``min_cell_count`` assets are pooled into one ``suppressed`` entry per
    breakdown. No audio is decoded and no model is run.
    """
    if type(min_cell_count) is not int or not 1 <= min_cell_count <= 1000:
        raise ValueError("min_cell_count must be an integer between 1 and 1000")
    materialized = list(records)
    for record in materialized:
        if not isinstance(record, AudioFormatRecord):
            raise AudioFormatSummaryError("audio_format_record_type")

    return AudioFormatSummary(
        schema_version=AUDIO_FORMAT_SUMMARY_SCHEMA_VERSION,
        total_assets=len(materialized),
        by_format=_breakdown(
            [record.format.value for record in materialized], min_cell_count
        ),
        by_channels=_breakdown(
            [record.channels for record in materialized], min_cell_count
        ),
        by_sample_rate_hz=_breakdown(
            [record.sample_rate_hz for record in materialized], min_cell_count
        ),
        by_bit_depth=_breakdown(
            [record.bit_depth for record in materialized], min_cell_count
        ),
        by_duration_bucket=_breakdown(
            [_duration_bucket(record.duration_seconds) for record in materialized],
            min_cell_count,
        ),
    )
