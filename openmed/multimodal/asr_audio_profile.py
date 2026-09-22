"""Compare privacy-safe WAV metadata with immutable local ASR input profiles.

A file can be a valid WAV and still violate the channel, rate, depth, format
or duration limits a local ASR provider accepts. This module answers that
question from declared metadata alone. It never decodes, resamples, downmixes,
or transcribes, and a report carries categorical and numeric fields only.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .wav_metadata import WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM, WavMetadata

ASR_AUDIO_PROFILE_SCHEMA_VERSION: Final[str] = "openmed.multimodal.asr_audio_profile.v1"
MAX_ASR_SAMPLE_RATE_HZ: Final[int] = 768_000
MAX_ASR_CHANNEL_COUNT: Final[int] = 64
MAX_ASR_DURATION_SECONDS: Final[float] = 86_400.0

SUPPORTED_BIT_DEPTHS: Final[tuple[int, ...]] = (8, 16, 24, 32, 64)
SUPPORTED_FORMAT_CODES: Final[tuple[int, ...]] = (
    WAVE_FORMAT_PCM,
    WAVE_FORMAT_IEEE_FLOAT,
)

ASR_REASON_CODES: Final[tuple[str, ...]] = (
    "empty_audio",
    "format_unsupported",
    "bit_depth_unsupported",
    "channel_count_unsupported",
    "sample_rate_unsupported",
    "duration_below_minimum",
    "duration_above_maximum",
    "downmix_required",
    "resample_required",
)

_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?$")
_REPORT_FIELDS = (
    "schema_version",
    "profile_id",
    "compatibility",
    "reason_codes",
    "format_code",
    "channels",
    "sample_rate_hz",
    "bit_depth",
    "frame_count",
    "duration_seconds",
)


class AsrCompatibility(str, Enum):
    """Closed verdict vocabulary for one profile comparison.

    Values:
        COMPATIBLE: The declared metadata already matches the profile.
        RESAMPLE: Only the sample rate has to change.
        DOWNMIX: Channels have to be mixed down, possibly with a resample.
        REVIEW: A human decision is needed before the audio is sent.
        INCOMPATIBLE: The profile cannot accept this audio at all.
    """

    COMPATIBLE = "compatible"
    RESAMPLE = "resample"
    DOWNMIX = "downmix"
    REVIEW = "review"
    INCOMPATIBLE = "incompatible"


_VERDICT_RANK: Final[dict[AsrCompatibility, int]] = {
    AsrCompatibility.COMPATIBLE: 0,
    AsrCompatibility.RESAMPLE: 1,
    AsrCompatibility.DOWNMIX: 2,
    AsrCompatibility.REVIEW: 3,
    AsrCompatibility.INCOMPATIBLE: 4,
}


class AsrAudioProfileError(ValueError):
    """Value-free failure raised for an unusable profile or comparison."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class AsrAudioProfile:
    """Immutable declaration of what one local ASR provider accepts.

    Attributes:
        profile_id: Lowercase bounded identifier for the profile.
        format_codes: Accepted WAVE format codes, sorted and unique.
        sample_rates_hz: Accepted sample rates, sorted and unique.
        channel_counts: Accepted channel counts, sorted and unique.
        bit_depths: Accepted sample depths, sorted and unique.
        min_duration_seconds: Shortest accepted duration.
        max_duration_seconds: Longest accepted duration.
        allow_resample: Whether a rate mismatch may be resampled.
        allow_downmix: Whether extra channels may be mixed down.
    """

    profile_id: str
    format_codes: tuple[int, ...]
    sample_rates_hz: tuple[int, ...]
    channel_counts: tuple[int, ...]
    bit_depths: tuple[int, ...]
    min_duration_seconds: float = 0.0
    max_duration_seconds: float = MAX_ASR_DURATION_SECONDS
    allow_resample: bool = True
    allow_downmix: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.profile_id) is not str
            or _IDENTIFIER_RE.fullmatch(self.profile_id) is None
        ):
            raise AsrAudioProfileError("asr_profile_id_invalid")
        _validate_values(self.format_codes, "format", allowed=SUPPORTED_FORMAT_CODES)
        _validate_values(self.sample_rates_hz, "rate", maximum=MAX_ASR_SAMPLE_RATE_HZ)
        _validate_values(self.channel_counts, "channel", maximum=MAX_ASR_CHANNEL_COUNT)
        _validate_values(self.bit_depths, "depth", allowed=SUPPORTED_BIT_DEPTHS)
        minimum = _validate_duration(self.min_duration_seconds)
        maximum = _validate_duration(self.max_duration_seconds)
        if minimum > maximum:
            raise AsrAudioProfileError("asr_profile_duration_range_invalid")
        if type(self.allow_resample) is not bool:
            raise AsrAudioProfileError("asr_profile_flag_invalid")
        if type(self.allow_downmix) is not bool:
            raise AsrAudioProfileError("asr_profile_flag_invalid")
        object.__setattr__(self, "min_duration_seconds", minimum)
        object.__setattr__(self, "max_duration_seconds", maximum)


@dataclass(frozen=True, slots=True)
class AsrCompatibilityReport:
    """Deterministic verdict for one profile and one WAV header."""

    profile_id: str
    compatibility: AsrCompatibility
    reason_codes: tuple[str, ...]
    format_code: int
    channels: int
    sample_rate_hz: int
    bit_depth: int
    frame_count: int
    duration_seconds: float
    schema_version: str = ASR_AUDIO_PROFILE_SCHEMA_VERSION

    @property
    def is_compatible(self) -> bool:
        """Return whether the audio needs no change and no review."""

        return self.compatibility is AsrCompatibility.COMPATIBLE

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "compatibility": self.compatibility.value,
            "reason_codes": list(self.reason_codes),
            "format_code": self.format_code,
            "channels": self.channels,
            "sample_rate_hz": self.sample_rate_hz,
            "bit_depth": self.bit_depth,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
        }
        return {field: values[field] for field in _REPORT_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def check_asr_compatibility(
    metadata: WavMetadata, profile: AsrAudioProfile
) -> AsrCompatibilityReport:
    """Compare declared WAV metadata with one ASR input profile.

    Findings are collected in a single pass and the worst verdict wins, ranked
    ``compatible < resample < downmix < review < incompatible``. A profile that
    forbids a transform turns the corresponding mismatch into an incompatible
    verdict instead of a transform request.

    Args:
        metadata: Privacy-safe WAV metadata, as returned by
            :func:`~openmed.multimodal.wav_metadata.read_wav_metadata`.
        profile: The immutable profile to compare against.

    Returns:
        An :class:`AsrCompatibilityReport` whose reason codes follow the fixed
        order in :data:`ASR_REASON_CODES`.

    Raises:
        AsrAudioProfileError: If either argument is not of the expected type.
    """

    if not isinstance(metadata, WavMetadata):
        raise AsrAudioProfileError("asr_metadata_type_invalid")
    if not isinstance(profile, AsrAudioProfile):
        raise AsrAudioProfileError("asr_profile_type_invalid")

    findings: list[tuple[str, AsrCompatibility]] = []
    if metadata.frame_count == 0:
        findings.append(("empty_audio", AsrCompatibility.INCOMPATIBLE))
    if metadata.format_code not in profile.format_codes:
        findings.append(("format_unsupported", AsrCompatibility.INCOMPATIBLE))
    if metadata.bit_depth not in profile.bit_depths:
        findings.append(("bit_depth_unsupported", AsrCompatibility.INCOMPATIBLE))

    if metadata.channels not in profile.channel_counts:
        if profile.allow_downmix and metadata.channels > max(profile.channel_counts):
            findings.append(("downmix_required", AsrCompatibility.DOWNMIX))
        else:
            findings.append(
                ("channel_count_unsupported", AsrCompatibility.INCOMPATIBLE)
            )

    if metadata.sample_rate_hz not in profile.sample_rates_hz:
        if profile.allow_resample:
            findings.append(("resample_required", AsrCompatibility.RESAMPLE))
        else:
            findings.append(("sample_rate_unsupported", AsrCompatibility.INCOMPATIBLE))

    if metadata.duration_seconds < profile.min_duration_seconds:
        findings.append(("duration_below_minimum", AsrCompatibility.REVIEW))
    if metadata.duration_seconds > profile.max_duration_seconds:
        findings.append(("duration_above_maximum", AsrCompatibility.REVIEW))

    verdict = AsrCompatibility.COMPATIBLE
    for _, candidate in findings:
        if _VERDICT_RANK[candidate] > _VERDICT_RANK[verdict]:
            verdict = candidate
    reason_codes = tuple(
        sorted((code for code, _ in findings), key=ASR_REASON_CODES.index)
    )
    return AsrCompatibilityReport(
        profile_id=profile.profile_id,
        compatibility=verdict,
        reason_codes=reason_codes,
        format_code=metadata.format_code,
        channels=metadata.channels,
        sample_rate_hz=metadata.sample_rate_hz,
        bit_depth=metadata.bit_depth,
        frame_count=metadata.frame_count,
        duration_seconds=metadata.duration_seconds,
    )


def _validate_values(
    values: Any,
    name: str,
    *,
    maximum: int | None = None,
    allowed: tuple[int, ...] | None = None,
) -> None:
    if type(values) is not tuple:
        raise AsrAudioProfileError(f"asr_profile_{name}_values_invalid")
    if not values:
        raise AsrAudioProfileError(f"asr_profile_{name}_values_empty")
    for value in values:
        if type(value) is not int:
            raise AsrAudioProfileError(f"asr_profile_{name}_values_invalid")
        if allowed is not None and value not in allowed:
            raise AsrAudioProfileError(f"asr_profile_{name}_values_unsupported")
        if allowed is None and (value < 1 or (maximum is not None and value > maximum)):
            raise AsrAudioProfileError(f"asr_profile_{name}_values_out_of_range")
    if tuple(sorted(set(values))) != values:
        raise AsrAudioProfileError(f"asr_profile_{name}_values_unsorted")


def _validate_duration(value: Any) -> float:
    if type(value) not in (int, float):
        raise AsrAudioProfileError("asr_profile_duration_invalid")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise AsrAudioProfileError("asr_profile_duration_invalid")
    if not 0.0 <= normalized <= MAX_ASR_DURATION_SECONDS:
        raise AsrAudioProfileError("asr_profile_duration_out_of_range")
    return normalized


MONO_16K_PCM_PROFILE: Final[AsrAudioProfile] = AsrAudioProfile(
    profile_id="mono-16k-pcm",
    format_codes=(WAVE_FORMAT_PCM,),
    sample_rates_hz=(16_000,),
    channel_counts=(1,),
    bit_depths=(16,),
    min_duration_seconds=0.1,
    max_duration_seconds=3_600.0,
)
"""Generic mono 16 kHz PCM profile; it names no provider and bundles no model."""


__all__ = [
    "ASR_AUDIO_PROFILE_SCHEMA_VERSION",
    "ASR_REASON_CODES",
    "MAX_ASR_CHANNEL_COUNT",
    "MAX_ASR_DURATION_SECONDS",
    "MAX_ASR_SAMPLE_RATE_HZ",
    "MONO_16K_PCM_PROFILE",
    "SUPPORTED_BIT_DEPTHS",
    "SUPPORTED_FORMAT_CODES",
    "AsrAudioProfile",
    "AsrAudioProfileError",
    "AsrCompatibility",
    "AsrCompatibilityReport",
    "check_asr_compatibility",
]
