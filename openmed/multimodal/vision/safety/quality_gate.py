"""Bounded image measurements and task-specific abstention before inference."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from io import BytesIO
from typing import Any, ClassVar, Iterator

from openmed.multimodal.image_header import (
    ImageHeaderError,
    read_image_header,
)
from openmed.multimodal.notices import NOTICE_CATALOG, MultimodalNotice, NoticeKind
from openmed.multimodal.vision.providers.base import (
    EdgeVLMProvider,
    ImageInput,
    VLMRequest,
    VLMResult,
    run_local_vlm,
)
from openmed.multimodal.webp_dimensions import (
    WebpDimensionsError,
    read_webp_dimensions,
)

QUALITY_SCHEMA_VERSION = "openmed.vision.quality.v1"
MAX_DECODE_PIXELS = 16_000_000
MAX_SAMPLE_SIDE = 512
_DECODER_LOCK = threading.RLock()
_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


class ImageQualityError(ValueError):
    """Value-free error for invalid quality policy or caller contract."""


class QualityReason(str, Enum):
    """Deterministic reasons to withhold visual inference."""

    CORRUPT_IMAGE = "corrupt_or_unsupported_header"
    RESOURCE_LIMIT = "decode_pixel_limit"
    GEOMETRY_MISMATCH = "declared_geometry_mismatch"
    FORMAT_MISMATCH = "declared_format_mismatch"
    DECODER_UNAVAILABLE = "decoder_unavailable_or_permissive"
    DECODE_FAILED = "image_decode_failed"
    UNSUPPORTED_IMAGE = "unsupported_animation_mode_or_transparency"
    RESOLUTION = "insufficient_resolution"
    SAMPLE_TOO_SMALL = "insufficient_analysis_geometry"
    DARK = "underexposure_or_dark_clipping"
    BRIGHT = "overexposure_or_bright_clipping"
    CONTRAST = "insufficient_contrast"
    SHARPNESS = "insufficient_sharpness"
    BOUNDARY = "content_at_expected_blank_boundary"
    UNASSESSED_REGION = "region_quality_requires_separate_assessment"


class QualityAction(str, Enum):
    """Next input-quality action; passing is not clinical approval."""

    CONTINUE = "continue_to_reviewed_extraction"
    REACQUIRE = "reacquire_image"
    REVIEW = "human_review_input"


def _require(condition: bool) -> None:
    if not condition:
        raise ImageQualityError("invalid image quality contract")


@dataclass(frozen=True, slots=True, kw_only=True)
class QualityPolicy:
    """Explicit task thresholds; no clinically qualified defaults are supplied.

    Args:
        min_short_side: Minimum source short side, in pixels (at least three).
        min_long_side: Minimum source long side, at least min_short_side.
        min_luminance: Minimum normalized mean grayscale luminance.
        max_luminance: Maximum normalized mean grayscale luminance.
        min_contrast: Minimum normalized grayscale standard deviation.
        min_sharpness: Minimum normalized four-neighbor Laplacian variance.
        max_dark_fraction: Maximum fraction of samples at grayscale 0..12.
        max_bright_fraction: Maximum fraction of samples at grayscale 243..255.
        max_boundary_fraction: Optional maximum foreground fraction on any edge.
        background_luminance: Expected blank-margin luminance, required with the
            boundary threshold. Samples differing by at least 32/255 are ink.
    """

    min_short_side: int
    min_long_side: int
    min_luminance: float
    max_luminance: float
    min_contrast: float
    min_sharpness: float
    max_dark_fraction: float
    max_bright_fraction: float
    max_boundary_fraction: float | None = None
    background_luminance: float | None = None

    def __post_init__(self) -> None:
        for dimension in (self.min_short_side, self.min_long_side):
            _require(type(dimension) is int and 3 <= dimension <= 65536)
        _require(self.min_short_side <= self.min_long_side)
        for threshold in (
            self.min_luminance,
            self.max_luminance,
            self.min_contrast,
            self.min_sharpness,
            self.max_dark_fraction,
            self.max_bright_fraction,
        ):
            _require(type(threshold) in (int, float) and math.isfinite(threshold))
            _require(0 <= threshold <= 1)
        _require(self.min_luminance <= self.max_luminance)
        _require(self.min_contrast <= 0.5)
        _require(
            (self.max_boundary_fraction is None) == (self.background_luminance is None)
        )
        for optional_threshold in (
            self.max_boundary_fraction,
            self.background_luminance,
        ):
            if optional_threshold is not None:
                _require(
                    type(optional_threshold) in (int, float)
                    and math.isfinite(optional_threshold)
                )
                _require(0 <= optional_threshold <= 1)

    @property
    def digest(self) -> str:
        """Bind reports to all supplied thresholds without a free-text task name."""
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Aggregate image measurements only; no pixels, text or embedded metadata."""

    width: int
    height: int
    sample_width: int
    sample_height: int
    luminance: float
    contrast: float
    sharpness: float
    dark_fraction: float
    bright_fraction: float
    boundary_fraction: float | None


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Content-free assessment bound to exact encoded bytes and task thresholds."""

    review_notice: ClassVar[MultimodalNotice] = NOTICE_CATALOG[
        NoticeKind.VISUAL_DESCRIPTION
    ]

    image_digest: str
    policy: QualityPolicy
    reasons: tuple[QualityReason, ...]
    action: QualityAction
    metrics: QualityMetrics | None = None

    @property
    def accepted(self) -> bool:
        """Whether these input checks permit inference, never clinical use."""
        return not self.reasons

    def to_json(self) -> str:
        """Serialize aggregates and controlled reasons without protected payloads."""
        return json.dumps(
            {
                "schema_version": QUALITY_SCHEMA_VERSION,
                "image_digest": self.image_digest,
                "policy_digest": self.policy.digest,
                "thresholds": asdict(self.policy),
                "metrics": asdict(self.metrics) if self.metrics else None,
                "accepted": self.accepted,
                "reasons": [reason.value for reason in self.reasons],
                "action": self.action.value,
                "clinical_review_required": True,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@contextmanager
def _decoder_diagnostics() -> Iterator[None]:
    # Pillow's PNG parser logs raw ICC profile names at DEBUG. Filter only this
    # decoding thread, without changing logger levels or other threads' records.
    thread = threading.get_ident()

    class DecoderFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.thread != thread

    filter_ = DecoderFilter()
    loggers = [
        logging.getLogger(name)
        for name in (
            "PIL.Image",
            "PIL.ImageFile",
            "PIL.PngImagePlugin",
            "PIL.JpegImagePlugin",
            "PIL.WebPImagePlugin",
        )
    ]
    # Python 3.11/3.12 warning contexts share process state. Serialize this
    # decoder section so overlapping assessments restore their original filters.
    with _DECODER_LOCK:
        for logger in loggers:
            logger.addFilter(filter_)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", module=r"^PIL(?:\.|$)")
                yield
        finally:
            for logger in loggers:
                logger.removeFilter(filter_)


def _measure(gray: Any, policy: QualityPolicy, box_resampling: int) -> QualityMetrics:
    width, height = gray.size
    gray.thumbnail((MAX_SAMPLE_SIDE, MAX_SAMPLE_SIDE), resample=box_resampling)
    sw, sh = gray.size
    if min(sw, sh) < 3:
        raise ImageQualityError("invalid image quality contract")
    pixels = gray.tobytes()
    count = len(pixels)
    histogram = gray.histogram()
    total = sum(value * frequency for value, frequency in enumerate(histogram))
    squares = sum(
        value * value * frequency for value, frequency in enumerate(histogram)
    )
    variance = max(0, squares * count - total * total) / (count * count)
    lap_sum = lap_squares = 0
    for row in range(1, sh - 1):
        for column in range(1, sw - 1):
            i = row * sw + column
            lap = (
                pixels[i - 1]
                + pixels[i + 1]
                + pixels[i - sw]
                + pixels[i + sw]
                - 4 * pixels[i]
            )
            lap_sum += lap
            lap_squares += lap * lap
    interior = (sw - 2) * (sh - 2)
    sharpness = max(0, lap_squares * interior - lap_sum * lap_sum) / (
        interior * interior * 1020**2
    )
    boundary = None
    if policy.background_luminance is not None:
        background = policy.background_luminance * 255
        edges = (pixels[:sw], pixels[-sw:], pixels[::sw], pixels[sw - 1 :: sw])
        boundary = max(
            sum(abs(p - background) >= 32 for p in edge) / len(edge) for edge in edges
        )
    return QualityMetrics(
        width,
        height,
        sw,
        sh,
        total / (count * 255),
        math.sqrt(variance) / 255,
        sharpness,
        sum(histogram[:13]) / count,
        sum(histogram[243:]) / count,
        boundary,
    )


def assess_image_quality(image: ImageInput, policy: QualityPolicy) -> QualityReport:
    """Decode bounded PNG/JPEG/WebP locally and compare measured quality to policy.

    Args:
        image: Encoded input, protected in memory and never written to disk.
        policy: Task-specific, caller-qualified measurement thresholds.

    Returns:
        Aggregate quality report. Corruption, unsupported content, missing Pillow
        or an unmet threshold abstains. Passing is not diagnostic interpretation.

    Raises:
        ImageQualityError: Caller supplied the wrong contract or invalid policy.
    """
    _require(type(image) is ImageInput and type(policy) is QualityPolicy)
    policy.__post_init__()

    def reject(reason: QualityReason) -> QualityReport:
        return QualityReport(image.digest, policy, (reason,), QualityAction.REVIEW)

    try:
        header = (
            read_webp_dimensions(image.data, max_pixels=MAX_DECODE_PIXELS)
            if image.media_type == "image/webp"
            else read_image_header(image.data, max_pixels=MAX_DECODE_PIXELS)
        )
    except (ImageHeaderError, WebpDimensionsError) as exc:
        reason = (
            QualityReason.RESOURCE_LIMIT
            if "pixel_limit" in exc.category
            else QualityReason.CORRUPT_IMAGE
        )
        return reject(reason)
    if (header.width, header.height) != (image.width, image.height):
        return reject(QualityReason.GEOMETRY_MISMATCH)
    try:
        from PIL import Image, ImageFile
    except Exception:
        return reject(QualityReason.DECODER_UNAVAILABLE)
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        return reject(QualityReason.DECODER_UNAVAILABLE)
    try:
        with _decoder_diagnostics():
            with Image.open(
                BytesIO(image.data), formats=list(_FORMATS.values())
            ) as decoded:
                if decoded.format != _FORMATS[image.media_type]:
                    return reject(QualityReason.FORMAT_MISMATCH)
                if decoded.size != (header.width, header.height):
                    return reject(QualityReason.GEOMETRY_MISMATCH)
                if getattr(decoded, "n_frames", 1) != 1 or decoded.mode not in (
                    "L",
                    "RGB",
                    "RGBA",
                ):
                    return reject(QualityReason.UNSUPPORTED_IMAGE)
                decoded.verify()
            with Image.open(
                BytesIO(image.data), formats=list(_FORMATS.values())
            ) as decoded:
                decoded.load()
                if "transparency" in decoded.info or (
                    decoded.mode == "RGBA"
                    and decoded.getchannel("A").getextrema() != (255, 255)
                ):
                    return reject(QualityReason.UNSUPPORTED_IMAGE)
                with decoded.convert("L") as gray:
                    if (
                        min(gray.size) < 3
                        or min(gray.size) * MAX_SAMPLE_SIDE // max(gray.size) < 3
                    ):
                        return reject(QualityReason.SAMPLE_TOO_SMALL)
                    metrics = _measure(gray, policy, Image.Resampling.BOX)
    except Exception:
        return reject(QualityReason.DECODE_FAILED)
    reasons = []
    if (
        min(metrics.width, metrics.height) < policy.min_short_side
        or max(metrics.width, metrics.height) < policy.min_long_side
    ):
        reasons.append(QualityReason.RESOLUTION)
    if (
        metrics.luminance < policy.min_luminance
        or metrics.dark_fraction > policy.max_dark_fraction
    ):
        reasons.append(QualityReason.DARK)
    if (
        metrics.luminance > policy.max_luminance
        or metrics.bright_fraction > policy.max_bright_fraction
    ):
        reasons.append(QualityReason.BRIGHT)
    if metrics.contrast < policy.min_contrast:
        reasons.append(QualityReason.CONTRAST)
    if metrics.sharpness < policy.min_sharpness:
        reasons.append(QualityReason.SHARPNESS)
    if (
        policy.max_boundary_fraction is not None
        and metrics.boundary_fraction is not None
        and metrics.boundary_fraction > policy.max_boundary_fraction
    ):
        reasons.append(QualityReason.BOUNDARY)
    return QualityReport(
        image.digest,
        policy,
        tuple(reasons),
        QualityAction.REACQUIRE if reasons else QualityAction.CONTINUE,
        metrics,
    )


def run_quality_gated_vlm(
    provider: EdgeVLMProvider, request: VLMRequest, policy: QualityPolicy
) -> tuple[QualityReport, VLMResult | None]:
    """Assess exact request bytes before even accessing provider metadata.

    Args:
        provider: Trusted local adapter admitted by run_local_vlm after quality.
        request: Protected visual extraction request.
        policy: Explicit thresholds qualified by the caller for this task.

    Returns:
        Quality report and validated draft, or None when input checks abstain.
        All successful drafts still require the existing explicit human review.
    """
    _require(type(request) is VLMRequest)
    report = assess_image_quality(request.image, policy)
    if report.accepted and request.regions:
        # A global average cannot qualify an arbitrary crop. Do not silently
        # admit region-specific extraction using only whole-image measurements.
        report = QualityReport(
            request.image.digest,
            policy,
            (QualityReason.UNASSESSED_REGION,),
            QualityAction.REVIEW,
            report.metrics,
        )
    return report, run_local_vlm(provider, request) if report.accepted else None
