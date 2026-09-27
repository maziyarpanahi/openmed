"""Deterministic, allocation-free resampling plans for ASR adapters.

Converts declared sample rates and frame counts into an exact plan: the
reduced rational rate, the target frame count under an explicit rounding
policy, the resulting duration error, and an exact or rounded status. The
planner reads no audio, allocates no samples, performs no filtering, and
makes no claim about resampling quality; it only plans integer arithmetic
that a decoder or model adapter would otherwise repeat ad hoc.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .asset_manifest import MAX_MANIFEST_BYTE_SIZE

__all__ = [
    "AUDIO_RESAMPLE_PLAN_SCHEMA_VERSION",
    "MAX_AUDIO_RATE_HZ",
    "MAX_AUDIO_FRAMES",
    "AudioResamplePlan",
    "AudioResamplePlanError",
    "ResampleStatus",
    "RoundingPolicy",
    "plan_audio_resampling",
]

MAX_AUDIO_RATE_HZ: Final[int] = (1 << 32) - 1
MAX_AUDIO_FRAMES: Final[int] = (1 << 32) - 1
AUDIO_RESAMPLE_PLAN_SCHEMA_VERSION: Final = 1


class AudioResamplePlanError(ValueError):
    """Value-free failure for invalid rates, frames, or plan arithmetic."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class RoundingPolicy(str, Enum):
    """Closed set of deterministic target-frame rounding policies."""

    FLOOR = "floor"
    NEAREST_EVEN = "nearest_even"
    CEILING = "ceiling"


class ResampleStatus(str, Enum):
    """Whether the target frame count is exact or the result of rounding."""

    EXACT = "exact"
    ROUNDED = "rounded"


def _count(value: Any, name: str, *, allow_zero: bool) -> int:
    if value is None:
        raise AudioResamplePlanError(f"audio_resample_{name}_missing")
    if type(value) is bool:
        raise AudioResamplePlanError(f"audio_resample_{name}_boolean")
    if type(value) is float and not math.isfinite(value):
        raise AudioResamplePlanError(f"audio_resample_{name}_not_finite")
    if type(value) is not int:
        raise AudioResamplePlanError(f"audio_resample_{name}_not_integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise AudioResamplePlanError(f"audio_resample_{name}_not_positive")
    if value > MAX_AUDIO_FRAMES:
        raise AudioResamplePlanError(f"audio_resample_{name}_overflow")
    return value


def _nearest_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = 2 * remainder
    if doubled < denominator:
        return quotient
    if doubled > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


@dataclass(frozen=True, slots=True)
class AudioResamplePlan:
    """Integer-exact resampling plan with no audio and no allocation."""

    schema_version: int
    source_rate_hz: int
    target_rate_hz: int
    source_frames: int
    target_frames: int
    rate_ratio_numerator: int
    rate_ratio_denominator: int
    rounding: RoundingPolicy
    status: ResampleStatus
    duration_error_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "schema_version": self.schema_version,
            "source_rate_hz": self.source_rate_hz,
            "target_rate_hz": self.target_rate_hz,
            "source_frames": self.source_frames,
            "target_frames": self.target_frames,
            "rate_ratio_numerator": self.rate_ratio_numerator,
            "rate_ratio_denominator": self.rate_ratio_denominator,
            "rounding": self.rounding.value,
            "status": self.status.value,
            "duration_error_seconds": self.duration_error_seconds,
        }

    def to_json(self) -> str:
        """Serialize with deterministic key order and no insignificant space."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=False,
            separators=(",", ":"),
        )


def plan_audio_resampling(
    source_rate_hz: Any,
    target_rate_hz: Any,
    source_frames: Any,
    *,
    rounding: Any = RoundingPolicy.NEAREST_EVEN,
    max_duration_error_seconds: Any = None,
) -> AudioResamplePlan:
    """Plan integer-exact frame conversion under an explicit rounding policy.

    ``source_rate_hz`` and ``target_rate_hz`` must be positive integers of at
    most ``MAX_AUDIO_RATE_HZ``; ``source_frames`` must be a non-negative
    integer of at most ``MAX_AUDIO_FRAMES``. The target frame count is
    ``source_frames * target_rate_hz / source_rate_hz`` rounded by the chosen
    policy. The duration error is the signed difference between the resampled
    duration and the declared duration, in seconds. No audio is read and no
    sample buffer is allocated.
    """
    if not isinstance(rounding, RoundingPolicy):
        raise ValueError("rounding must be a RoundingPolicy")
    checked_source_rate = _count(source_rate_hz, "source_rate", allow_zero=False)
    checked_target_rate = _count(target_rate_hz, "target_rate", allow_zero=False)
    checked_frames = _count(source_frames, "frames", allow_zero=True)
    if checked_frames > MAX_MANIFEST_BYTE_SIZE // checked_target_rate:
        raise AudioResamplePlanError("audio_resample_product_overflow")

    scaled_numerator = checked_frames * checked_target_rate
    scaled_denominator = checked_source_rate
    exact = scaled_numerator % scaled_denominator == 0
    if rounding is RoundingPolicy.FLOOR:
        target_frames = scaled_numerator // scaled_denominator
    elif rounding is RoundingPolicy.CEILING:
        target_frames = -(-scaled_numerator // scaled_denominator)
    else:
        target_frames = _nearest_even(scaled_numerator, scaled_denominator)

    error_numerator = target_frames * checked_source_rate - scaled_numerator
    error_denominator = checked_source_rate * checked_target_rate
    duration_error = (
        0.0 if error_numerator == 0 else error_numerator / error_denominator
    )
    if max_duration_error_seconds is not None:
        if (
            type(max_duration_error_seconds) not in (int, float)
            or not math.isfinite(max_duration_error_seconds)
            or max_duration_error_seconds < 0
        ):
            raise ValueError(
                "max_duration_error_seconds must be a finite non-negative number"
            )
        if abs(duration_error) > float(max_duration_error_seconds):
            raise AudioResamplePlanError("audio_resample_error_exceeded")

    divisor = math.gcd(checked_source_rate, checked_target_rate)
    return AudioResamplePlan(
        schema_version=AUDIO_RESAMPLE_PLAN_SCHEMA_VERSION,
        source_rate_hz=checked_source_rate,
        target_rate_hz=checked_target_rate,
        source_frames=checked_frames,
        target_frames=target_frames,
        rate_ratio_numerator=checked_source_rate // divisor,
        rate_ratio_denominator=checked_target_rate // divisor,
        rounding=rounding,
        status=ResampleStatus.EXACT if exact else ResampleStatus.ROUNDED,
        duration_error_seconds=duration_error,
    )
