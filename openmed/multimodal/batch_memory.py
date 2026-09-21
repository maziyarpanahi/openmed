"""Bounded decoded-memory plans for multimodal batches.

Before a batch is decoded, an edge runtime needs to know whether its decoded
images, DICOM frames, and audio waveforms fit a memory budget. This module
estimates that from validated :class:`AssetManifest` metadata plus an explicit,
caller-supplied :class:`MemoryEstimationPolicy`, and returns one of three plans:
accept the batch as given, split it into ordered batches that each fit, or
reject it.

An estimate holds only under the policy's documented assumptions; it is not a
guarantee that decoding or inference fits. The planner never reads headers,
decodes media, measures live memory, or guesses a factor the manifest and the
policy do not supply: a missing factor or missing geometry leaves the asset
unevaluable, and an unevaluable asset rejects the batch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Final

from .asset_batch import MAX_BATCH_ASSETS
from .asset_manifest import MAX_MANIFEST_BYTE_SIZE, AssetManifest

__all__ = [
    "BATCH_MEMORY_PLAN_VERSION",
    "BATCH_MEMORY_REASON_CODES",
    "MAX_AUDIO_CHANNELS",
    "MAX_AUDIO_SAMPLE_RATE_HZ",
    "MAX_BYTES_PER_UNIT",
    "SATURATION_CEILING",
    "AssetMemoryEstimate",
    "BatchMemoryError",
    "BatchMemoryPlan",
    "BatchOutcome",
    "MemoryEstimationPolicy",
    "PlannedBatch",
    "plan_batch_memory",
]

BATCH_MEMORY_PLAN_VERSION: Final = 1

# Every product, sum, budget, and overhead is held at or below this signed
# 64-bit ceiling, so a plan means the same thing on every platform. A product
# that would pass it saturates: the asset is reported ``estimate_saturated``
# with no estimate, and the batch is rejected.
SATURATION_CEILING: Final = MAX_MANIFEST_BYTE_SIZE

# Inclusive bounds on policy factors.
MAX_BYTES_PER_UNIT: Final = 1024
MAX_AUDIO_CHANNELS: Final = 1024
MAX_AUDIO_SAMPLE_RATE_HZ: Final = (1 << 32) - 1

BATCH_MEMORY_REASON_CODES: Final = frozenset(
    {"insufficient_metadata", "estimate_saturated", "budget_exceeded"}
)


class BatchMemoryError(ValueError):
    """Raised when a policy, budget, or batch input is invalid."""


class BatchOutcome(str, Enum):
    """Closed set of plan outcomes."""

    ACCEPT = "accept"
    SPLIT = "split"
    REJECT = "reject"


def _require_int(value: Any, name: str, *, minimum: int, maximum: int) -> None:
    # ``type(...) is int`` rejects booleans, floats (including NaN and
    # infinities), and int subclasses in one test.
    if type(value) is not int or not minimum <= value <= maximum:
        raise BatchMemoryError(f"{name} must be a bounded integer")


@dataclass(frozen=True, slots=True)
class MemoryEstimationPolicy:
    """Caller-supplied factors the manifest cannot carry.

    Each factor is ``None`` (not supplied) or a bounded positive integer.
    There are no defaults: a modality whose factor is ``None`` is unevaluable.

    - ``image_bytes_per_pixel``: decoded bytes per image pixel.
    - ``dicom_bytes_per_pixel``: decoded bytes per DICOM pixel, per frame.
    - ``audio_sample_rate_hz``: decoded waveform sample rate.
    - ``audio_channels``: decoded waveform channel count.
    - ``audio_bytes_per_sample``: decoded bytes per sample, per channel.
    """

    image_bytes_per_pixel: int | None = None
    dicom_bytes_per_pixel: int | None = None
    audio_sample_rate_hz: int | None = None
    audio_channels: int | None = None
    audio_bytes_per_sample: int | None = None

    def __post_init__(self) -> None:
        bounds = {
            "image_bytes_per_pixel": MAX_BYTES_PER_UNIT,
            "dicom_bytes_per_pixel": MAX_BYTES_PER_UNIT,
            "audio_sample_rate_hz": MAX_AUDIO_SAMPLE_RATE_HZ,
            "audio_channels": MAX_AUDIO_CHANNELS,
            "audio_bytes_per_sample": MAX_BYTES_PER_UNIT,
        }
        for name, maximum in bounds.items():
            value = getattr(self, name)
            if value is not None:
                _require_int(value, name, minimum=1, maximum=maximum)

    def to_dict(self) -> dict[str, Any]:
        """Return the policy as a JSON-ready mapping with a fixed key order."""
        return {
            "image_bytes_per_pixel": self.image_bytes_per_pixel,
            "dicom_bytes_per_pixel": self.dicom_bytes_per_pixel,
            "audio_sample_rate_hz": self.audio_sample_rate_hz,
            "audio_channels": self.audio_channels,
            "audio_bytes_per_sample": self.audio_bytes_per_sample,
        }


@dataclass(frozen=True, slots=True)
class AssetMemoryEstimate:
    """Content-free estimate for one asset, identified only by position.

    ``estimated_bytes`` is ``None`` whenever ``reason_code`` is
    ``insufficient_metadata`` or ``estimate_saturated``. For
    ``insufficient_metadata``, ``field_name`` names the first missing input.
    """

    position: int
    modality: str | None
    estimated_bytes: int | None
    reason_code: str | None = None
    field_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the estimate as a JSON-ready mapping with a fixed key order."""
        return {
            "position": self.position,
            "modality": self.modality,
            "estimated_bytes": self.estimated_bytes,
            "reason_code": self.reason_code,
            "field_name": self.field_name,
        }


@dataclass(frozen=True, slots=True)
class PlannedBatch:
    """Input positions for one planned batch and its estimate with overhead."""

    positions: tuple[int, ...]
    estimated_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return the batch as a JSON-ready mapping with a fixed key order."""
        return {
            "positions": list(self.positions),
            "estimated_bytes": self.estimated_bytes,
        }


@dataclass(frozen=True, slots=True)
class BatchMemoryPlan:
    """Deterministic accept/split/reject plan. ``batches`` is empty on reject."""

    outcome: BatchOutcome
    budget_bytes: int
    overhead_bytes: int
    assets: tuple[AssetMemoryEstimate, ...]
    batches: tuple[PlannedBatch, ...]
    version: int = BATCH_MEMORY_PLAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as a JSON-ready mapping with a fixed key order."""
        return {
            "version": self.version,
            "outcome": self.outcome.value,
            "budget_bytes": self.budget_bytes,
            "overhead_bytes": self.overhead_bytes,
            "assets": [asset.to_dict() for asset in self.assets],
            "batches": [batch.to_dict() for batch in self.batches],
        }

    def to_json(self) -> str:
        """Serialize with a fixed key order and no insignificant whitespace."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


def plan_batch_memory(
    assets: Sequence[AssetManifest],
    policy: MemoryEstimationPolicy,
    *,
    budget_bytes: int,
    overhead_bytes: int,
) -> BatchMemoryPlan:
    """Plan decoded memory for ``assets`` in the order given.

    ``budget_bytes`` is an inclusive ceiling (equal passes) and
    ``overhead_bytes`` is charged once per planned batch. The plan rejects when
    any asset is unevaluable, saturates, or does not fit alone with the
    overhead, because splitting cannot cure any of those. Otherwise it accepts
    when everything fits in one batch, and splits into the fewest contiguous
    batches that each fit when it does not.
    """
    if not isinstance(policy, MemoryEstimationPolicy):
        raise TypeError("policy must be a MemoryEstimationPolicy")
    _require_int(budget_bytes, "budget_bytes", minimum=0, maximum=SATURATION_CEILING)
    _require_int(
        overhead_bytes, "overhead_bytes", minimum=0, maximum=SATURATION_CEILING
    )
    if isinstance(assets, (str, bytes)) or not isinstance(assets, Sequence):
        raise TypeError("assets must be a sequence of AssetManifest")
    if not 0 < len(assets) <= MAX_BATCH_ASSETS:
        raise BatchMemoryError("assets must hold between 1 and MAX_BATCH_ASSETS")
    if not all(isinstance(asset, AssetManifest) for asset in assets):
        raise TypeError("assets must be a sequence of AssetManifest")

    estimates = tuple(
        _estimate(position, asset, policy, budget_bytes, overhead_bytes)
        for position, asset in enumerate(assets)
    )
    if any(estimate.reason_code is not None for estimate in estimates):
        return BatchMemoryPlan(
            BatchOutcome.REJECT, budget_bytes, overhead_bytes, estimates, ()
        )

    # Each asset fits alone with the overhead, so greedy contiguous packing
    # always terminates, and it yields the fewest batches that keep order.
    # Running totals stay at or below the budget, so no sum can saturate.
    batches: list[PlannedBatch] = []
    positions: list[int] = []
    total = overhead_bytes
    for estimate in estimates:
        size = estimate.estimated_bytes
        assert size is not None
        if size > budget_bytes - total:
            batches.append(PlannedBatch(tuple(positions), total))
            positions, total = [], overhead_bytes
        positions.append(estimate.position)
        total += size
    batches.append(PlannedBatch(tuple(positions), total))

    outcome = BatchOutcome.ACCEPT if len(batches) == 1 else BatchOutcome.SPLIT
    return BatchMemoryPlan(
        outcome, budget_bytes, overhead_bytes, estimates, tuple(batches)
    )


def _estimate(
    position: int,
    manifest: AssetManifest,
    policy: MemoryEstimationPolicy,
    budget_bytes: int,
    overhead_bytes: int,
) -> AssetMemoryEstimate:
    modality = _modality_for(manifest.media_type)
    if modality is None:
        return _unevaluable(position, None, "media_type")
    if modality == "pdf":
        # PDF_V1 carries no raster geometry, and a page count alone cannot
        # stand in for it, so a PDF is always unevaluable here.
        return _unevaluable(position, modality, "width")

    inputs: tuple[tuple[str, Any], ...]
    if modality == "image":
        inputs = (
            ("width", manifest.width),
            ("height", manifest.height),
            ("image_bytes_per_pixel", policy.image_bytes_per_pixel),
        )
    elif modality == "dicom":
        inputs = (
            ("width", manifest.width),
            ("height", manifest.height),
            ("frames", manifest.frames),
            ("dicom_bytes_per_pixel", policy.dicom_bytes_per_pixel),
        )
    else:
        inputs = (
            ("duration_seconds", manifest.duration_seconds),
            ("audio_sample_rate_hz", policy.audio_sample_rate_hz),
            ("audio_channels", policy.audio_channels),
            ("audio_bytes_per_sample", policy.audio_bytes_per_sample),
        )
    for name, value in inputs:
        if value is None:
            return _unevaluable(position, modality, name)

    factors: list[Any] = [value for _, value in inputs]
    if modality == "audio":
        # Exact rational ceiling of duration * rate: a float duration is an
        # exact binary fraction, so no float product decides the frame count.
        duration, rate = factors[0], factors[1]
        factors[:2] = [math.ceil(Fraction(duration) * rate)]

    size = _saturating_product(factors)
    if size is None:
        return AssetMemoryEstimate(position, modality, None, "estimate_saturated")
    if size > budget_bytes - overhead_bytes:
        return AssetMemoryEstimate(position, modality, size, "budget_exceeded")
    return AssetMemoryEstimate(position, modality, size)


def _unevaluable(
    position: int, modality: str | None, field_name: str
) -> AssetMemoryEstimate:
    return AssetMemoryEstimate(
        position, modality, None, "insufficient_metadata", field_name
    )


def _saturating_product(factors: Sequence[int]) -> int | None:
    product = 1
    for factor in factors:
        product *= factor
        if product > SATURATION_CEILING:
            return None
    return product


def _modality_for(media_type: str) -> str | None:
    # Mirrors ``preflight._modality_for``; application/dicom+json has no
    # geometry profile, so it is unevaluable.
    if media_type == "application/pdf":
        return "pdf"
    if media_type == "application/dicom":
        return "dicom"
    if media_type.startswith("image/"):
        return "image"
    if media_type.startswith("audio/"):
        return "audio"
    return None
