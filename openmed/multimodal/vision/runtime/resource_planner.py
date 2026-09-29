"""Deterministic memory planning for on-device vision-language execution.

Before an edge runtime allocates tensors for a vision-language model it needs a
declared answer to one question: does this model and preprocessing plan fit a
registered device tier, or must the runtime abstain? This module answers that
from caller-supplied factors and a declared preprocessing plan, using integer
arithmetic only.

The planner never allocates, decodes, measures live memory, inspects a device,
or contacts a network. Every figure is an estimate that holds only under the
documented accounting assumptions, so a plan is a guardrail for allocation
order, not a measurement. When an input factor is missing the estimate is
unevaluable, when a product would exceed the signed 64-bit ceiling the estimate
saturates, and when no registered tier fits the estimated total the plan
abstains. Abstention is final: this contract has no cloud fallback, and a plan
never names a provider, endpoint, credential, or region.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

__all__ = [
    "CLOUD_FALLBACK_ALLOWED",
    "MAX_ACTIVATION_BYTES_PER_PIXEL",
    "MAX_CACHE_BYTES_PER_TILE",
    "MAX_DEVICE_TIERS",
    "MAX_FRAMES",
    "MAX_MEMORY_BYTES",
    "MAX_MODEL_PARAMETERS",
    "MAX_OUTPUT_BYTES",
    "MAX_PIXEL_SIDE",
    "MAX_TILE_SIZE",
    "MAX_WEIGHT_BYTES_PER_PARAMETER",
    "RESOURCE_PLAN_REASON_CODES",
    "RESOURCE_PLAN_VERSION",
    "VLM_TIER_DESKTOP_V1",
    "VLM_TIER_MOBILE_V1",
    "DeviceTier",
    "PreprocessingPlan",
    "ResourcePlanOutcome",
    "TierCandidate",
    "VLMExecutionPlan",
    "VLMExecutionPolicy",
    "VLMResourceEstimate",
    "VLMResourcePlannerError",
    "plan_vlm_execution",
]

RESOURCE_PLAN_VERSION: Final = 1

# This contract is local-only by construction. The constant exists so a caller
# can assert the invariant instead of trusting a docstring.
CLOUD_FALLBACK_ALLOWED: Final = False

# Every product, sum, and budget is held at or below this signed 64-bit
# ceiling, so a plan means the same thing on every platform. A product that
# would pass it saturates: the component is dropped from the estimate and the
# plan abstains with ``estimate_saturated``.
MAX_MEMORY_BYTES: Final = (1 << 63) - 1

# Inclusive bounds on declared inputs and policy factors.
MAX_DEVICE_TIERS: Final = 64
MAX_MODEL_PARAMETERS: Final = 1 << 62
MAX_WEIGHT_BYTES_PER_PARAMETER: Final = 64
MAX_ACTIVATION_BYTES_PER_PIXEL: Final = 4096
MAX_CACHE_BYTES_PER_TILE: Final = 1 << 32
MAX_OUTPUT_BYTES: Final = 1 << 40
MAX_PIXEL_SIDE: Final = 1 << 16
MAX_TILE_SIZE: Final = 1 << 14
MAX_FRAMES: Final = 1 << 20

RESOURCE_PLAN_REASON_CODES: Final = frozenset(
    {"insufficient_metadata", "estimate_saturated", "memory_budget_exceeded"}
)

# A tier name is an opaque lower-case label. It never carries a hostname, a
# serial number, a site, an endpoint, or any other identifying string.
_TIER_NAME = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z")


class VLMResourcePlannerError(ValueError):
    """Raised when a policy, tier, or preprocessing plan is invalid."""


class ResourcePlanOutcome(str, Enum):
    """Closed set of plan outcomes."""

    SAFE = "safe"
    ABSTAIN = "abstain"


def _require_int(value: Any, name: str, *, minimum: int, maximum: int) -> None:
    # ``type(...) is int`` rejects booleans, floats (including NaN and
    # infinities), and int subclasses in one test.
    if type(value) is not int or not minimum <= value <= maximum:
        raise VLMResourcePlannerError(f"{name} must be a bounded integer")


def _saturating_add(total: int, value: int) -> int | None:
    total += value
    return total if total <= MAX_MEMORY_BYTES else None


def _saturating_product(factors: Sequence[int]) -> int | None:
    product = 1
    for factor in factors:
        product *= factor
        if product > MAX_MEMORY_BYTES:
            return None
    return product


@dataclass(frozen=True, slots=True)
class DeviceTier:
    """A registered device memory tier, identified by an opaque label.

    ``memory_budget_bytes`` is the total budget for the process and
    ``reserved_bytes`` is the part the runtime keeps for itself, so
    :attr:`available_bytes` is what a plan may use.
    """

    name: str
    memory_budget_bytes: int
    reserved_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _TIER_NAME.match(self.name):
            raise VLMResourcePlannerError("device tier name must be a bounded label")
        _require_int(
            self.memory_budget_bytes,
            "memory_budget_bytes",
            minimum=1,
            maximum=MAX_MEMORY_BYTES,
        )
        _require_int(
            self.reserved_bytes,
            "reserved_bytes",
            minimum=0,
            maximum=MAX_MEMORY_BYTES,
        )
        if self.reserved_bytes >= self.memory_budget_bytes:
            raise VLMResourcePlannerError(
                "reserved_bytes must be lower than memory_budget_bytes"
            )

    @property
    def available_bytes(self) -> int:
        """Return the budget a plan may use."""
        return self.memory_budget_bytes - self.reserved_bytes

    def to_dict(self) -> dict[str, Any]:
        """Return the tier as a JSON-ready mapping with a fixed key order."""
        return {
            "name": self.name,
            "memory_budget_bytes": self.memory_budget_bytes,
            "reserved_bytes": self.reserved_bytes,
            "available_bytes": self.available_bytes,
        }


# Declared, conservative reference tiers. Callers may register their own.
VLM_TIER_MOBILE_V1: Final = DeviceTier(
    name="mobile.v1",
    memory_budget_bytes=6 * (1 << 30),
    reserved_bytes=1 << 30,
)
VLM_TIER_DESKTOP_V1: Final = DeviceTier(
    name="desktop.v1",
    memory_budget_bytes=32 * (1 << 30),
    reserved_bytes=4 * (1 << 30),
)


@dataclass(frozen=True, slots=True)
class PreprocessingPlan:
    """Declared image geometry and tiling, before any tensor is allocated.

    ``tile_size`` is the side of one square tile and ``tile_overlap`` the
    overlapping margin between neighbouring tiles; overlapping tiles repeat
    activation work, so the planner charges every tile it will process.
    """

    width: int
    height: int
    tile_size: int
    tile_overlap: int = 0
    frames: int = 1

    def __post_init__(self) -> None:
        _require_int(self.width, "width", minimum=1, maximum=MAX_PIXEL_SIDE)
        _require_int(self.height, "height", minimum=1, maximum=MAX_PIXEL_SIDE)
        _require_int(self.tile_size, "tile_size", minimum=1, maximum=MAX_TILE_SIZE)
        _require_int(
            self.tile_overlap,
            "tile_overlap",
            minimum=0,
            maximum=MAX_TILE_SIZE - 1,
        )
        _require_int(self.frames, "frames", minimum=1, maximum=MAX_FRAMES)
        if self.tile_overlap >= self.tile_size:
            raise VLMResourcePlannerError("tile_overlap must be lower than tile_size")

    @property
    def stride(self) -> int:
        """Return the tile stride in pixels."""
        return self.tile_size - self.tile_overlap

    @property
    def tiles_x(self) -> int:
        """Return the number of tile columns covering ``width``."""
        return -(-self.width // self.stride)

    @property
    def tiles_y(self) -> int:
        """Return the number of tile rows covering ``height``."""
        return -(-self.height // self.stride)

    @property
    def tile_count(self) -> int:
        """Return the number of tiles covering the declared image."""
        return self.tiles_x * self.tiles_y

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as a JSON-ready mapping with a fixed key order."""
        return {
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "frames": self.frames,
            "tiles_x": self.tiles_x,
            "tiles_y": self.tiles_y,
            "tile_count": self.tile_count,
        }


@dataclass(frozen=True, slots=True)
class VLMExecutionPolicy:
    """Caller-supplied factors and the device tiers a plan may choose.

    Each factor is ``None`` (not supplied) or a bounded integer. There are no
    defaults: a factor that is ``None`` leaves the estimate unevaluable, and an
    unevaluable estimate abstains rather than guessing. ``runtime_overhead_bytes``
    is charged once for the runtime's own working set.
    """

    tiers: tuple[DeviceTier, ...]
    parameter_count: int | None = None
    weight_bytes_per_parameter: int | None = None
    activation_bytes_per_pixel: int | None = None
    cache_bytes_per_tile: int | None = None
    output_bytes: int | None = None
    runtime_overhead_bytes: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.tiers, (str, bytes)) or not isinstance(self.tiers, tuple):
            raise VLMResourcePlannerError("tiers must be a tuple of DeviceTier")
        if not 0 < len(self.tiers) <= MAX_DEVICE_TIERS:
            raise VLMResourcePlannerError(
                "tiers must hold between 1 and MAX_DEVICE_TIERS device tiers"
            )
        names: set[str] = set()
        for tier in self.tiers:
            if not isinstance(tier, DeviceTier):
                raise VLMResourcePlannerError("tiers must be a tuple of DeviceTier")
            if tier.name in names:
                raise VLMResourcePlannerError("device tier names must be unique")
            names.add(tier.name)
        bounds = {
            "parameter_count": (0, MAX_MODEL_PARAMETERS),
            "weight_bytes_per_parameter": (1, MAX_WEIGHT_BYTES_PER_PARAMETER),
            "activation_bytes_per_pixel": (1, MAX_ACTIVATION_BYTES_PER_PIXEL),
            "cache_bytes_per_tile": (0, MAX_CACHE_BYTES_PER_TILE),
            "output_bytes": (0, MAX_OUTPUT_BYTES),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if value is not None:
                _require_int(value, name, minimum=minimum, maximum=maximum)
        _require_int(
            self.runtime_overhead_bytes,
            "runtime_overhead_bytes",
            minimum=0,
            maximum=MAX_MEMORY_BYTES,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the policy as a JSON-ready mapping with a fixed key order."""
        return {
            "tiers": [tier.to_dict() for tier in self.tiers],
            "parameter_count": self.parameter_count,
            "weight_bytes_per_parameter": self.weight_bytes_per_parameter,
            "activation_bytes_per_pixel": self.activation_bytes_per_pixel,
            "cache_bytes_per_tile": self.cache_bytes_per_tile,
            "output_bytes": self.output_bytes,
            "runtime_overhead_bytes": self.runtime_overhead_bytes,
        }


@dataclass(frozen=True, slots=True)
class VLMResourceEstimate:
    """Content-free memory estimate for one declared workload.

    Every component is ``None`` whenever the estimate is unevaluable or
    saturated. For ``insufficient_metadata``, ``field_name`` names the first
    missing factor; for ``estimate_saturated`` it names the first component that
    passed the ceiling.
    """

    model_bytes: int | None
    peak_activation_bytes: int | None
    cache_bytes: int | None
    output_bytes: int | None
    runtime_overhead_bytes: int
    total_bytes: int | None
    reason_code: str | None = None
    field_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the estimate as a JSON-ready mapping with a fixed key order."""
        return {
            "model_bytes": self.model_bytes,
            "peak_activation_bytes": self.peak_activation_bytes,
            "cache_bytes": self.cache_bytes,
            "output_bytes": self.output_bytes,
            "runtime_overhead_bytes": self.runtime_overhead_bytes,
            "total_bytes": self.total_bytes,
            "reason_code": self.reason_code,
            "field_name": self.field_name,
        }


@dataclass(frozen=True, slots=True)
class TierCandidate:
    """One registered tier evaluated against the estimated total."""

    name: str
    available_bytes: int
    fits: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the candidate as a JSON-ready mapping with a fixed key order."""
        return {
            "name": self.name,
            "available_bytes": self.available_bytes,
            "fits": self.fits,
        }


@dataclass(frozen=True, slots=True)
class VLMExecutionPlan:
    """Deterministic safe-or-abstain plan.

    ``tier_name`` names the selected tier for a safe plan and is ``None`` for an
    abstained plan; ``candidates`` is ordered by available bytes then name, so
    the report does not depend on tier declaration order.
    """

    outcome: ResourcePlanOutcome
    tier_name: str | None
    tier_available_bytes: int | None
    estimate: VLMResourceEstimate
    candidates: tuple[TierCandidate, ...]
    preprocessing: PreprocessingPlan
    cloud_fallback_allowed: bool = CLOUD_FALLBACK_ALLOWED
    version: int = RESOURCE_PLAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as a JSON-ready mapping with a fixed key order."""
        return {
            "version": self.version,
            "outcome": self.outcome.value,
            "tier_name": self.tier_name,
            "tier_available_bytes": self.tier_available_bytes,
            "estimate": self.estimate.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "preprocessing": self.preprocessing.to_dict(),
            "cloud_fallback_allowed": self.cloud_fallback_allowed,
        }

    def to_json(self) -> str:
        """Serialize with a fixed key order and no insignificant whitespace."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


def plan_vlm_execution(
    policy: VLMExecutionPolicy,
    preprocessing: PreprocessingPlan,
) -> VLMExecutionPlan:
    """Plan memory for one workload against the registered device tiers.

    The estimate charges the model weights, one decoded frame stack plus one
    tile working buffer, the declared per-tile cache for every tile, the
    declared output, and the runtime overhead once. A safe plan selects the
    smallest registered tier that fits, breaking ties by tier name; an abstained
    plan always reports ``tier_name`` as ``None`` and never falls back to a
    remote provider.
    """
    if not isinstance(policy, VLMExecutionPolicy):
        raise TypeError("policy must be a VLMExecutionPolicy")
    if not isinstance(preprocessing, PreprocessingPlan):
        raise TypeError("preprocessing must be a PreprocessingPlan")

    components = (
        ("parameter_count", policy.parameter_count),
        ("weight_bytes_per_parameter", policy.weight_bytes_per_parameter),
        ("activation_bytes_per_pixel", policy.activation_bytes_per_pixel),
        ("cache_bytes_per_tile", policy.cache_bytes_per_tile),
        ("output_bytes", policy.output_bytes),
    )
    for name, value in components:
        if value is None:
            return _abstain(
                policy,
                preprocessing,
                _unevaluable(policy, "insufficient_metadata", name),
            )

    assert policy.parameter_count is not None
    assert policy.weight_bytes_per_parameter is not None
    assert policy.activation_bytes_per_pixel is not None
    assert policy.cache_bytes_per_tile is not None
    assert policy.output_bytes is not None

    model_bytes = _saturating_product(
        (policy.parameter_count, policy.weight_bytes_per_parameter)
    )
    if model_bytes is None:
        return _abstain(
            policy, preprocessing, _unevaluable(policy, "estimate_saturated", "model")
        )

    frame_stack = _saturating_product(
        (
            preprocessing.width,
            preprocessing.height,
            preprocessing.frames,
            policy.activation_bytes_per_pixel,
        )
    )
    tile_buffer = _saturating_product(
        (
            preprocessing.tile_size,
            preprocessing.tile_size,
            preprocessing.frames,
            policy.activation_bytes_per_pixel,
        )
    )
    if frame_stack is None:
        return _abstain(
            policy,
            preprocessing,
            _unevaluable(policy, "estimate_saturated", "peak_activation"),
        )
    if tile_buffer is None:
        return _abstain(
            policy,
            preprocessing,
            _unevaluable(policy, "estimate_saturated", "peak_activation"),
        )
    peak_activation_bytes = _saturating_add(frame_stack, tile_buffer)
    if peak_activation_bytes is None:
        return _abstain(
            policy,
            preprocessing,
            _unevaluable(policy, "estimate_saturated", "peak_activation"),
        )

    cache_bytes = _saturating_product(
        (preprocessing.tile_count, policy.cache_bytes_per_tile)
    )
    if cache_bytes is None:
        return _abstain(
            policy, preprocessing, _unevaluable(policy, "estimate_saturated", "cache")
        )

    output_bytes = policy.output_bytes
    runtime_overhead_bytes = policy.runtime_overhead_bytes
    total = 0
    for value in (
        model_bytes,
        peak_activation_bytes,
        cache_bytes,
        output_bytes,
        runtime_overhead_bytes,
    ):
        summed = _saturating_add(total, value)
        if summed is None:
            return _abstain(
                policy,
                preprocessing,
                _unevaluable(policy, "estimate_saturated", "total"),
            )
        total = summed

    estimate = VLMResourceEstimate(
        model_bytes=model_bytes,
        peak_activation_bytes=peak_activation_bytes,
        cache_bytes=cache_bytes,
        output_bytes=output_bytes,
        runtime_overhead_bytes=runtime_overhead_bytes,
        total_bytes=total,
    )
    candidates = _candidates(policy, total)
    selected = next((candidate for candidate in candidates if candidate.fits), None)
    if selected is None:
        return VLMExecutionPlan(
            outcome=ResourcePlanOutcome.ABSTAIN,
            tier_name=None,
            tier_available_bytes=None,
            estimate=VLMResourceEstimate(
                model_bytes=estimate.model_bytes,
                peak_activation_bytes=estimate.peak_activation_bytes,
                cache_bytes=estimate.cache_bytes,
                output_bytes=estimate.output_bytes,
                runtime_overhead_bytes=estimate.runtime_overhead_bytes,
                total_bytes=estimate.total_bytes,
                reason_code="memory_budget_exceeded",
                field_name=None,
            ),
            candidates=candidates,
            preprocessing=preprocessing,
        )
    return VLMExecutionPlan(
        outcome=ResourcePlanOutcome.SAFE,
        tier_name=selected.name,
        tier_available_bytes=selected.available_bytes,
        estimate=estimate,
        candidates=candidates,
        preprocessing=preprocessing,
    )


def _abstain(
    policy: VLMExecutionPolicy,
    preprocessing: PreprocessingPlan,
    estimate: VLMResourceEstimate,
) -> VLMExecutionPlan:
    return VLMExecutionPlan(
        outcome=ResourcePlanOutcome.ABSTAIN,
        tier_name=None,
        tier_available_bytes=None,
        estimate=estimate,
        candidates=(),
        preprocessing=preprocessing,
    )


def _unevaluable(
    policy: VLMExecutionPolicy, reason_code: str, field_name: str
) -> VLMResourceEstimate:
    assert reason_code in RESOURCE_PLAN_REASON_CODES
    return VLMResourceEstimate(
        model_bytes=None,
        peak_activation_bytes=None,
        cache_bytes=None,
        output_bytes=None,
        runtime_overhead_bytes=policy.runtime_overhead_bytes,
        total_bytes=None,
        reason_code=reason_code,
        field_name=field_name,
    )


def _candidate_key(tier: DeviceTier) -> tuple[int, str]:
    return (tier.available_bytes, tier.name)


def _candidates(
    policy: VLMExecutionPolicy, total_bytes: int
) -> tuple[TierCandidate, ...]:
    return tuple(
        TierCandidate(
            name=tier.name,
            available_bytes=tier.available_bytes,
            fits=tier.available_bytes >= total_bytes,
        )
        for tier in sorted(policy.tiers, key=_candidate_key)
    )
