"""Typed, bounded edge-VLM contracts without model loading or network access.

Image bytes, prompts and generated text are protected in-memory payloads, never
audit fields. Only :meth:`VLMResult.to_json` is intended for reporting. Providers
are trusted local code: capability declarations are admission checks, not a
network sandbox or model qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from openmed.multimodal.abstention import AbstentionRecord, AbstentionStage

__all__ = [
    "NON_DIAGNOSTIC_NOTICE",
    "VLM_SCHEMA_VERSION",
    "Confidence",
    "EdgeVLMProvider",
    "ImageInput",
    "ImageRegion",
    "LocalProviderMetadata",
    "Prompt",
    "RuntimeBackend",
    "RuntimeMetadata",
    "VisualEvidence",
    "VLMContractError",
    "VLMRequest",
    "VLMResult",
    "run_local_vlm",
]

VLM_SCHEMA_VERSION = "openmed.vision.provider.v1"
NON_DIAGNOSTIC_NOTICE = (
    "Non-diagnostic model output. Explicit human review is required; "
    "do not use this output to trigger clinical decisions automatically."
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_TEXT_BYTES = 64 * 1024
_MAX_EVIDENCE = 256


class VLMContractError(ValueError):
    """Content-free failure at the local provider boundary."""


class RuntimeBackend(str, Enum):
    """Closed backend vocabulary; no hosted execution backend exists."""

    MLX = "mlx"
    COREML = "coreml"
    ONNX = "onnx"
    TORCH = "torch"
    SYNTHETIC = "synthetic"


def _require(condition: bool) -> None:
    if not condition:
        raise VLMContractError("invalid VLM contract")


def _digest(value: object) -> None:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None)


def _integer(value: object, minimum: int, maximum: int) -> None:
    _require(type(value) is int and minimum <= value <= maximum)


def _text(value: object, maximum: int, *, empty: bool = False) -> None:
    _require(type(value) is str)
    try:
        length = len(value.encode("utf-8"))
    except UnicodeError:
        length = maximum + 1
    _require((empty or length > 0) and length <= maximum)


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInput:
    """Caller-supplied encoded image without filenames or patient metadata.

    Args:
        data: Protected encoded bytes, never written by this contract.
        media_type: Supported encoded media type.
        width: Declared source width in pixels; adapters must verify it.
        height: Declared source height in pixels; adapters must verify it.
    """

    data: bytes = field(repr=False)
    media_type: str
    width: int
    height: int

    def __post_init__(self) -> None:
        _require(type(self.data) is bytes and 0 < len(self.data) <= _MAX_IMAGE_BYTES)
        _require(
            type(self.media_type) is str
            and self.media_type in ("image/png", "image/jpeg", "image/webp")
        )
        _integer(self.width, 1, 65536)
        _integer(self.height, 1, 65536)
        _require(self.width * self.height <= 100_000_000)

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the exact encoded input bytes."""
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageRegion:
    """Half-open pixel rectangle in the declared source image coordinates.

    Args:
        image_digest: Digest binding the region to one encoded image.
        x0: Inclusive left edge.
        y0: Inclusive top edge.
        x1: Exclusive right edge.
        y1: Exclusive bottom edge.
    """

    image_digest: str
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        _digest(self.image_digest)
        for coordinate in (self.x0, self.y0, self.x1, self.y1):
            _integer(coordinate, 0, 65536)
        _require(self.x0 < self.x1 and self.y0 < self.y1)


@dataclass(frozen=True, slots=True, kw_only=True)
class Prompt:
    """Protected caller prompt and reproducible generation limits.

    Args:
        text: In-memory prompt, excluded from representations and reports.
        max_output_tokens: Hard generation token ceiling for the adapter.
        max_output_bytes: Hard UTF-8 output ceiling checked at the boundary.
        seed: Explicit deterministic generation seed; sampling must be disabled.
    """

    text: str = field(repr=False)
    max_output_tokens: int = 512
    max_output_bytes: int = 4096
    seed: int = 0

    def __post_init__(self) -> None:
        _text(self.text, _MAX_TEXT_BYTES)
        _integer(self.max_output_tokens, 1, 4096)
        _integer(self.max_output_bytes, 1, _MAX_TEXT_BYTES)
        _integer(self.seed, 0, (1 << 32) - 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class VLMRequest:
    """Single image, prompt and optional source-space region selection.

    Args:
        image: Protected encoded image and its declared geometry.
        prompt: Protected prompt and generation ceilings.
        regions: Immutable tuple of bounded regions in the source image.
    """

    image: ImageInput
    prompt: Prompt
    regions: tuple[ImageRegion, ...] = ()

    def __post_init__(self) -> None:
        _require(type(self.image) is ImageInput and type(self.prompt) is Prompt)
        _require(type(self.regions) is tuple and len(self.regions) <= _MAX_EVIDENCE)
        for region in self.regions:
            self.validate_region(region)

    def validate_region(self, region: ImageRegion) -> None:
        """Reject regions from another image or outside source geometry."""
        _require(type(region) is ImageRegion)
        _require(region.image_digest == self.image.digest)
        _require(region.x1 <= self.image.width and region.y1 <= self.image.height)

    @property
    def digest(self) -> str:
        """Bind image bytes, geometry, prompt and all generation parameters."""
        payload = [
            VLM_SCHEMA_VERSION,
            self.image.digest,
            self.image.media_type,
            self.image.width,
            self.image.height,
            hashlib.sha256(self.prompt.text.encode("utf-8")).hexdigest(),
            self.prompt.max_output_tokens,
            self.prompt.max_output_bytes,
            self.prompt.seed,
            [[r.x0, r.y0, r.x1, r.y1] for r in self.regions],
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class Confidence:
    """Uncalibrated provider score, or explicit unknown confidence.

    Args:
        score: Finite score in [0, 1], or None when unavailable. This is not a
            calibrated probability or evidence of clinical validity.
    """

    score: float | None = None

    def __post_init__(self) -> None:
        if self.score is not None:
            _require(type(self.score) in (float, int))
            _require(math.isfinite(self.score) and 0 <= self.score <= 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualEvidence:
    """A generated character span linked to a source region, without text.

    Args:
        region: Source-space evidence coordinates and image digest.
        start: Inclusive Unicode code-point offset in generated text.
        end: Exclusive Unicode code-point offset in generated text.
        confidence: Uncalibrated score or explicit unknown.
    """

    region: ImageRegion
    start: int
    end: int
    confidence: Confidence = field(default_factory=Confidence)

    def __post_init__(self) -> None:
        _require(
            type(self.region) is ImageRegion and type(self.confidence) is Confidence
        )
        _integer(self.start, 0, _MAX_TEXT_BYTES)
        _integer(self.end, 1, _MAX_TEXT_BYTES)
        _require(self.start < self.end)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalProviderMetadata:
    """Pinned local provider declaration, never a qualification certificate.

    Args:
        backend: Local runtime backend.
        model_digest: SHA-256 identifying the exact caller-supplied model assets.
        preprocessing_digest: SHA-256 of the exact preprocessing configuration.
        local_only: Must be True; no telemetry or cloud fallback is permitted.
        requires_network: Must be False, including initialization dependencies.
        deterministic: Must be True; adapters must implement seeded, non-sampling
            generation and document device reproducibility limitations.
    """

    backend: RuntimeBackend
    model_digest: str
    preprocessing_digest: str
    local_only: bool = True
    requires_network: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        _require(type(self.backend) is RuntimeBackend)
        _digest(self.model_digest)
        _digest(self.preprocessing_digest)
        _require(self.local_only is True and self.requires_network is False)
        _require(self.deterministic is True)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeMetadata:
    """Per-result provenance and bounded measured runtime counters.

    Args:
        provider: Exact local model and preprocessing declaration for this run.
        duration_ms: Measured duration, independent of content determinism.
        output_tokens: Actual adapter-reported generation token count.
    """

    provider: LocalProviderMetadata
    duration_ms: float
    output_tokens: int

    def __post_init__(self) -> None:
        _require(type(self.provider) is LocalProviderMetadata)
        _require(type(self.duration_ms) in (int, float))
        _require(
            math.isfinite(self.duration_ms) and 0 <= self.duration_ms <= 86_400_000
        )
        _integer(self.output_tokens, 0, 4096)


@dataclass(frozen=True, slots=True, kw_only=True)
class VLMResult:
    """Protected draft or metadata-only abstention with mandatory provenance.

    Args:
        request_digest: Digest of the complete request.
        runtime: Mandatory model/preprocessing digests and execution metadata.
        text: Protected generated draft; absent for abstentions.
        evidence: Immutable generated-span/source-region links.
        confidence: Uncalibrated overall confidence or explicit unknown.
        abstention: Shared inference-stage abstention with no free-text reason.
    """

    request_digest: str
    runtime: RuntimeMetadata
    text: str = field(default="", repr=False)
    evidence: tuple[VisualEvidence, ...] = ()
    confidence: Confidence = field(default_factory=Confidence)
    abstention: AbstentionRecord | None = None

    @property
    def notice(self) -> str:
        """Return the mandatory non-diagnostic notice bound to every result."""
        return NON_DIAGNOSTIC_NOTICE

    def __post_init__(self) -> None:
        _digest(self.request_digest)
        _require(
            type(self.runtime) is RuntimeMetadata
            and type(self.confidence) is Confidence
        )
        _text(self.text, _MAX_TEXT_BYTES, empty=True)
        _require(type(self.evidence) is tuple and len(self.evidence) <= _MAX_EVIDENCE)
        if self.abstention is not None:
            _require(type(self.abstention) is AbstentionRecord)
            _require(self.abstention.stage is AbstentionStage.INFERENCE)
            _require(
                not self.text and not self.evidence and self.confidence.score is None
            )
            _require(self.runtime.output_tokens == 0)
        else:
            _require(bool(self.text) and self.runtime.output_tokens > 0)
        for item in self.evidence:
            _require(type(item) is VisualEvidence and item.end <= len(self.text))

    def reviewed_text(self, *, reviewer_confirmed: bool) -> str:
        """Return the protected draft only after explicit caller confirmation.

        Args:
            reviewer_confirmed: Must be exactly True after a human review.

        Raises:
            VLMContractError: Confirmation is absent or the result abstained.
        """
        _require(reviewer_confirmed is True and self.abstention is None)
        return self.text

    def to_json(self) -> str:
        """Return deterministic metadata only; never serialize protected text."""
        provider = self.runtime.provider
        payload = {
            "schema_version": VLM_SCHEMA_VERSION,
            "request_digest": self.request_digest,
            "backend": provider.backend.value,
            "model_digest": provider.model_digest,
            "preprocessing_digest": provider.preprocessing_digest,
            "local_only": True,
            "requires_network": False,
            "deterministic": True,
            "duration_ms": self.runtime.duration_ms,
            "output_tokens": self.runtime.output_tokens,
            "output_digest": hashlib.sha256(self.text.encode()).hexdigest()
            if self.text
            else None,
            "confidence": self.confidence.score,
            "abstention": self.abstention.to_dict() if self.abstention else None,
            "evidence": [
                {
                    "image_digest": e.region.image_digest,
                    "box": [e.region.x0, e.region.y0, e.region.x1, e.region.y1],
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence.score,
                }
                for e in self.evidence
            ],
            "notice": self.notice,
            "review_required": True,
        }
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )


@runtime_checkable
class EdgeVLMProvider(Protocol):
    """Narrow adapter seam; all assets must be provisioned locally in advance."""

    @property
    def metadata(self) -> LocalProviderMetadata:
        """Return the pinned local-only provider declaration without I/O."""
        ...

    def infer(self, request: VLMRequest) -> VLMResult:
        """Generate a bounded deterministic draft or a provenance-bound abstention.

        Args:
            request: Protected input; never log, cache or transmit its contents.
        """
        ...


def run_local_vlm(provider: EdgeVLMProvider, request: VLMRequest) -> VLMResult:
    """Admit a local provider and validate the complete result before consumption.

    Args:
        provider: Trusted offline adapter with pinned metadata.
        request: Bounded, validated image/prompt request.

    Returns:
        Validated protected draft or abstention; human review remains mandatory.

    Raises:
        VLMContractError: Admission, provider execution or result validation fails.
            Errors contain no input or provider exception text.
    """
    _require(type(request) is VLMRequest)
    failed = False
    try:
        metadata = provider.metadata
        _require(type(metadata) is LocalProviderMetadata)
        # Revalidate declarations before inference even if an adapter mutated one.
        metadata.__post_init__()
        result = provider.infer(request)
        _require(type(result) is VLMResult)
        result.__post_init__()
        result.runtime.__post_init__()
        _require(result.runtime.provider == metadata)
        _require(result.request_digest == request.digest)
        _require(len(result.text.encode()) <= request.prompt.max_output_bytes)
        _require(result.runtime.output_tokens <= request.prompt.max_output_tokens)
        for evidence in result.evidence:
            request.validate_region(evidence.region)
            if request.regions:
                _require(evidence.region in request.regions)
    except Exception:
        failed = True
    # Raise outside the handler so even the exception context is content-free.
    if failed:
        raise VLMContractError("local VLM execution rejected")
    return result
