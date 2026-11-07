"""Strict, content-free manifests for reproducible clinical-video sampling.

A frame-sampling manifest records which frames a deterministic sampler
selected from a clip, without carrying any frames, audio, captions, or
identifiers: only integer tick positions, a rational time base, a sampling
mode, the declared strategy fields, and a coverage digest over the
canonical form. Validation enforces monotonicity, uniqueness, bounds, and
that the timestamps actually satisfy the declared strategy. It is a
reproducibility contract, not a video decoder and not a sampling-policy
recommendation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .asset_manifest import MAX_MANIFEST_COUNT

__all__ = [
    "FRAME_SAMPLING_SCHEMA_VERSION",
    "SamplingMode",
    "FrameSamplingManifest",
    "FrameSamplingManifestError",
    "canonical_coverage_digest",
]

FRAME_SAMPLING_SCHEMA_VERSION: Final = 1
_HEX64_LENGTH: Final = 64


class FrameSamplingManifestError(ValueError):
    """Value-free failure for an invalid frame-sampling manifest."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class SamplingMode(str, Enum):
    """Closed set of deterministic frame-selection modes."""

    UNIFORM = "uniform"
    KEYFRAME = "keyframe"
    BOUNDED_WINDOW = "bounded_window"


def _tick(value: Any, category: str) -> int:
    if type(value) is not int:
        raise FrameSamplingManifestError(category)
    if not 0 < value <= MAX_MANIFEST_COUNT:
        raise FrameSamplingManifestError(category)
    return value


def canonical_coverage_digest(
    sampling_mode: SamplingMode,
    time_base_num: int,
    time_base_den: int,
    timestamps: Any,
) -> str:
    """Return the SHA-256 coverage digest over a manifest's canonical form.

    The digested payload is the deterministic JSON encoding of the schema
    version, the sampling mode, the reduced time base, and the selected
    tick positions. Two manifests that cover the same frames in the same
    time base produce the same digest regardless of how they were built.
    """
    if not isinstance(sampling_mode, SamplingMode):
        raise FrameSamplingManifestError("frame_sampling_mode_unsupported")
    _tick(time_base_num, "frame_sampling_time_base_invalid")
    _tick(time_base_den, "frame_sampling_time_base_invalid")
    materialized = tuple(timestamps)
    for tick in materialized:
        if type(tick) is not int or not 0 <= tick <= MAX_MANIFEST_COUNT:
            raise FrameSamplingManifestError("frame_sampling_timestamps_invalid")
    payload = json.dumps(
        {
            "schema_version": FRAME_SAMPLING_SCHEMA_VERSION,
            "mode": sampling_mode.value,
            "time_base": [time_base_num, time_base_den],
            "timestamps": list(materialized),
        },
        ensure_ascii=True,
        sort_keys=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class FrameSamplingManifest:
    """Validated, content-free description of one frame selection.

    No field on this type carries frames, audio, captions, pixel data, or
    identifiers. ``timestamps`` are integer tick positions in the manifest's
    rational time base, ``duration_ticks`` bounds them, and the per-mode
    strategy fields (``step_ticks`` for uniform, ``window_start_ticks`` and
    ``window_end_ticks`` for bounded windows) are validated against the
    selected positions so a misdeclared strategy fails closed.
    """

    duration_ticks: int
    time_base_num: int
    time_base_den: int
    sampling_mode: SamplingMode
    timestamps: tuple[int, ...]
    coverage_digest: str
    step_ticks: int | None = None
    window_start_ticks: int | None = None
    window_end_ticks: int | None = None
    schema_version: int = FRAME_SAMPLING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FRAME_SAMPLING_SCHEMA_VERSION:
            raise FrameSamplingManifestError("frame_sampling_schema_unsupported")
        num = _tick(self.time_base_num, "frame_sampling_time_base_invalid")
        den = _tick(self.time_base_den, "frame_sampling_time_base_invalid")
        if math.gcd(num, den) != 1:
            raise FrameSamplingManifestError("frame_sampling_time_base_not_reduced")
        duration = _tick(self.duration_ticks, "frame_sampling_duration_invalid")
        if not isinstance(self.sampling_mode, SamplingMode):
            raise FrameSamplingManifestError("frame_sampling_mode_unsupported")
        materialized = tuple(self.timestamps)
        previous: int | None = None
        for tick in materialized:
            if type(tick) is not int:
                raise FrameSamplingManifestError("frame_sampling_timestamps_invalid")
            if not 0 <= tick < duration:
                raise FrameSamplingManifestError(
                    "frame_sampling_timestamps_out_of_range"
                )
            if previous is not None and tick <= previous:
                raise FrameSamplingManifestError(
                    "frame_sampling_timestamps_not_increasing"
                )
            previous = tick
        object.__setattr__(self, "timestamps", materialized)

        if self.sampling_mode is SamplingMode.UNIFORM:
            if self.window_start_ticks is not None or self.window_end_ticks is not None:
                raise FrameSamplingManifestError(
                    "frame_sampling_strategy_fields_invalid"
                )
            step = _tick(self.step_ticks, "frame_sampling_step_invalid")
            for index in range(1, len(materialized)):
                if materialized[index] - materialized[index - 1] != step:
                    raise FrameSamplingManifestError("frame_sampling_step_mismatch")
        elif self.sampling_mode is SamplingMode.KEYFRAME:
            if (
                self.step_ticks is not None
                or self.window_start_ticks is not None
                or self.window_end_ticks is not None
            ):
                raise FrameSamplingManifestError(
                    "frame_sampling_strategy_fields_invalid"
                )
        else:
            if self.step_ticks is not None:
                raise FrameSamplingManifestError(
                    "frame_sampling_strategy_fields_invalid"
                )
            start = self.window_start_ticks
            end = self.window_end_ticks
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= duration
            ):
                raise FrameSamplingManifestError("frame_sampling_window_invalid")
            for tick in materialized:
                if not start <= tick < end:
                    raise FrameSamplingManifestError("frame_sampling_window_mismatch")

        if (
            type(self.coverage_digest) is not str
            or len(self.coverage_digest) != _HEX64_LENGTH
            or any(char not in "0123456789abcdef" for char in self.coverage_digest)
        ):
            raise FrameSamplingManifestError("frame_sampling_coverage_digest_invalid")
        expected = canonical_coverage_digest(self.sampling_mode, num, den, materialized)
        if self.coverage_digest != expected:
            raise FrameSamplingManifestError("frame_sampling_coverage_digest_mismatch")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "duration_ticks": self.duration_ticks,
            "time_base_num": self.time_base_num,
            "time_base_den": self.time_base_den,
            "sampling_mode": self.sampling_mode.value,
            "timestamps": list(self.timestamps),
            "coverage_digest": self.coverage_digest,
        }
        if self.step_ticks is not None:
            data["step_ticks"] = self.step_ticks
        if self.window_start_ticks is not None:
            data["window_start_ticks"] = self.window_start_ticks
        if self.window_end_ticks is not None:
            data["window_end_ticks"] = self.window_end_ticks
        return data

    def to_json(self) -> str:
        """Serialize with deterministic key order and no insignificant space."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=False,
            separators=(",", ":"),
        )
