"""Source-bound annotation redaction with an in-memory, verified export gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any, TypeVar

from openmed.core.custom_recognizer import (
    MAX_CUSTOM_RECOGNIZER_SCAN_WORK,
    CustomRecognizer,
    CustomRecognizerLimitError,
)
from openmed.core.safety_sweep import SAFETY_SWEEP_PATTERNS_VERSION, safety_sweep
from openmed.core.schemas.span import OpenMedSpan
from openmed.core.script_detect import normalize_for_pii_detection

from ..ocr import OcrResult, OcrWord
from ..ocr_tesseract_stream import StreamingTesseractEngine
from ..render_raster import RasterRedactionPage, render_redacted_raster_pages
from .lead_layout import (
    AcquisitionKind,
    LeadFindingCode,
    LeadLayout,
    LeadObservation,
    assess_lead_layout,
)
from .waveform import EcgWaveform

ANNOTATION_VERSION = "openmed.ecg.annotation-privacy.v1"
ANNOTATION_NOTICE = (
    "Protected ECG review export, not anonymous data or clinical approval. "
    "Coverage and detection depend on qualified local providers and source-bound "
    "human review. Preserve the waveform, calibration and provenance together; "
    "never automatically trigger clinical decisions."
)
_MASK = "[REDACTED]"
_T = TypeVar("_T")
Rectangle = tuple[int, int, int, int]


class PrivacyReason(str, Enum):
    """Controlled export refusal codes, never source values or backend messages."""

    DECLARATION = "invalid-annotation-declaration"
    LIMIT = "annotation-resource-limit"
    COVERAGE = "annotation-coverage-unestablished"
    LAYOUT = "lead-identity-requires-review"
    REVIEW = "source-bound-review-required"
    STALE = "annotation-source-or-plan-changed"
    RESIDUAL = "residual-identifiers"
    BACKEND = "local-annotation-backend-failed"
    OFFLINE = "local-provider-required"


class AnnotationPrivacyError(ValueError):
    """Fixed value-free error with a controlled reason."""

    def __init__(self, reason: PrivacyReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _require(
    condition: bool, reason: PrivacyReason = PrivacyReason.DECLARATION
) -> None:
    if not condition:
        raise AnnotationPrivacyError(reason)


def _integer(value: object, lower: int, upper: int) -> None:
    _require(type(value) is int and lower <= value <= upper)  # type: ignore[operator]


def _digest(value: object) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _text(value: object) -> None:
    _require(type(value) is str and len(value) <= 100_000)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _protected_call(call: Callable[[], _T]) -> _T:
    # Raise outside the handler: even __context__ must not retain raw provider errors.
    failure = None
    try:
        return call()
    except AnnotationPrivacyError as error:
        failure = error.reason
    except CustomRecognizerLimitError:
        failure = PrivacyReason.LIMIT
    except Exception:
        failure = PrivacyReason.BACKEND
    raise AnnotationPrivacyError(failure)


class SurfaceKind(str, Enum):
    """Structured or rendered annotation surfaces included in the inventory."""

    HEADER = "header"
    ANNOTATION = "annotation"
    CALIBRATION = "calibration"
    LEAD_LABEL = "lead-label"
    FOOTER = "footer"
    METADATA = "metadata"
    FILENAME = "filename"


class RegionKind(str, Enum):
    """Exhaustive page partition; trace/background may contain no OCR text."""

    TRACE = "waveform-trace"
    ANNOTATION = "annotation"
    BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class AnnotationField:
    """One protected source field; names and values are both examined."""

    kind: SurfaceKind
    name: str = field(repr=False)
    value: str = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.kind) is SurfaceKind)
        _text(self.name)
        _text(self.value)


@dataclass(frozen=True, slots=True, kw_only=True)
class PageRegion:
    """Half-open original-pixel region with a stable host reference.

    Args:
        region_ref: Opaque 32-hex identity, never a patient or device name.
        kind: Trace, annotation surface or background.
        bounds: Integer top-left x0, y0, x1, y1 on the unchanged source raster.
        surface: Required only for an annotation region.
        slot: Required only for a trace; row/column in the actual LeadLayout.
    """

    region_ref: str
    kind: RegionKind
    bounds: Rectangle = field(repr=False)
    surface: SurfaceKind | None = None
    slot: tuple[int, int] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require(
            type(self.region_ref) is str
            and re.fullmatch(r"[0-9a-f]{32}", self.region_ref) is not None
        )
        _require(type(self.kind) is RegionKind)
        _require(type(self.bounds) is tuple and len(self.bounds) == 4)
        for value in self.bounds:
            _integer(value, 0, 100_000)
        x0, y0, x1, y1 = self.bounds
        _require(x0 < x1 and y0 < y1)
        _require(
            (type(self.surface) is SurfaceKind)
            if self.kind is RegionKind.ANNOTATION
            else self.surface is None
        )
        if self.kind is RegionKind.TRACE:
            _require(type(self.slot) is tuple and len(self.slot) == 2)
            assert self.slot is not None
            for value in self.slot:
                _integer(value, 0, 63)
        else:
            _require(self.slot is None)


@dataclass(frozen=True, slots=True)
class AnnotationPage:
    """One already selected, orientation-normalized Pillow raster and its partition."""

    image: Any = field(repr=False)
    regions: tuple[PageRegion, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.regions) is tuple and 1 <= len(self.regions) <= 512)
        _require(all(type(item) is PageRegion for item in self.regions))
        _require(len({item.region_ref for item in self.regions}) == len(self.regions))


@dataclass(frozen=True, slots=True, kw_only=True)
class EcgAnnotationSource:
    """Complete protected extraction supplied by a native reader or host intake.

    Args:
        source_digest: Digest of the original source, retained as provenance only.
        extraction_digest: Host evidence for complete field/frame extraction.
        filename: Original source name; always replaced in the exported bundle.
        fields: Every extracted header, annotation, calibration and other text field.
        pages: Every rendered page/frame, with exhaustive pixel partitions.
        layout: Independent expected lead layout.
        observations: Original lead labels and positions, before normalization.
        acquisition: Digital or image-derived, matching the supplied surfaces.
        waveform: Optional actual normalized waveform; samples are never changed.

    This is not a native EDF/WFDB/DICOM parser or rewriter. The host must establish
    complete extraction; supplied digests cannot authenticate that assertion.
    """

    source_digest: str
    extraction_digest: str
    filename: str = field(repr=False)
    fields: tuple[AnnotationField, ...] = field(repr=False)
    pages: tuple[AnnotationPage, ...] = field(repr=False)
    layout: LeadLayout = field(repr=False)
    observations: tuple[LeadObservation, ...] = field(repr=False)
    acquisition: AcquisitionKind
    waveform: EcgWaveform | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _digest(self.source_digest)
        _digest(self.extraction_digest)
        _text(self.filename)
        _require(type(self.fields) is tuple and len(self.fields) <= 1024)
        _require(all(type(item) is AnnotationField for item in self.fields))
        _require(
            all(
                item.kind not in (SurfaceKind.FILENAME, SurfaceKind.LEAD_LABEL)
                for item in self.fields
            )
        )
        _require(type(self.pages) is tuple and len(self.pages) <= 8)
        _require(all(type(item) is AnnotationPage for item in self.pages))
        _require(type(self.layout) is LeadLayout)
        _require(type(self.observations) is tuple and len(self.observations) <= 64)
        _require(all(type(item) is LeadObservation for item in self.observations))
        _require(type(self.acquisition) is AcquisitionKind)
        _require(self.waveform is None or type(self.waveform) is EcgWaveform)
        _require(bool(self.pages) or self.waveform is not None)
        _require(not self.pages or self.acquisition is AcquisitionKind.IMAGE_DERIVED)
        if self.waveform is not None:
            _require(self.waveform.provenance.source_digest == self.source_digest)
            _require(
                {item.lead_id for item in self.waveform.leads}
                == set(self.layout.expected_leads)
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class AnnotationPolicy:
    """Explicit qualified language and resource limits; no automatic model choice."""

    version: str
    evidence_digest: str
    language: str
    ocr_padding_pixels: int = 2
    maximum_text_chars: int = 1_000_000
    maximum_page_pixels: int = 12_000_000
    maximum_total_pixels: int = 72_000_000
    maximum_output_bytes: int = 32_000_000

    def __post_init__(self) -> None:
        _require(
            type(self.version) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.version)
            is not None
        )
        _digest(self.evidence_digest)
        _require(
            type(self.language) is str
            and re.fullmatch(r"[a-z]{2}", self.language) is not None
        )
        _integer(self.ocr_padding_pixels, 0, 32)
        _integer(self.maximum_text_chars, 1, 1_000_000)
        _integer(self.maximum_page_pixels, 1, 12_000_000)
        _integer(self.maximum_total_pixels, 1, 72_000_000)
        _integer(self.maximum_output_bytes, 1, 128_000_000)

    @property
    def digest(self) -> str:
        """Bind policy and the deterministic structured-identifier sweep version."""
        return _hash(
            {
                "algorithm": ANNOTATION_VERSION,
                "sweep": SAFETY_SWEEP_PATTERNS_VERSION,
                "policy": asdict(self),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalAnnotationDetector:
    """Host-owned, already loaded local detector over exact original code points.

    Args:
        detect: Bounded function returning OpenMedSpan records, without logging,
            network, file writes, caching raw values or implicit model downloads.
        model_digest: Exact loaded model/rules/configuration identity.
        qualification_digest: Host evaluation for this language and ECG surface.
        local_only: Must explicitly be True. This contract is not a sandbox.

    Every returned span is redacted irrespective of score/action/replacement.
    Only offsets are consumed; provider metadata is never copied into reports.
    """

    detect: Callable[[str], Sequence[OpenMedSpan]] = field(repr=False, compare=False)
    model_digest: str
    qualification_digest: str
    local_only: bool

    def __post_init__(self) -> None:
        _require(callable(self.detect))
        _digest(self.model_digest)
        _digest(self.qualification_digest)
        _require(self.local_only is True, PrivacyReason.OFFLINE)
        _require(
            getattr(self.detect, "network_egress", False) is False
            and getattr(self.detect, "allows_network", False) is False,
            PrivacyReason.OFFLINE,
        )

    @property
    def digest(self) -> str:
        """Bind declared loaded detector identity, not an authentication claim."""
        return _hash([self.model_digest, self.qualification_digest, self.local_only])


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalAnnotationOcr:
    """Bounded native stdin/stdout OCR; installed language data must be supplied."""

    engine: StreamingTesseractEngine = field(repr=False, compare=False)
    runtime_digest: str
    qualification_digest: str

    def __post_init__(self) -> None:
        _require(type(self.engine) is StreamingTesseractEngine, PrivacyReason.OFFLINE)
        _digest(self.runtime_digest)
        _digest(self.qualification_digest)

    @property
    def digest(self) -> str:
        """Bind host runtime evidence and actual execution limits/configuration."""
        engine = self.engine
        return _hash(
            [
                self.runtime_digest,
                self.qualification_digest,
                str(engine.executable),
                str(engine.tessdata_dir),
                engine.timeout_seconds,
                engine.max_pixels,
                engine.max_output_bytes,
                engine.max_words,
                engine.max_text_chars,
            ]
        )


@dataclass(frozen=True, slots=True)
class FieldRedaction:
    """Protected original combined-name/newline/value offsets and stable identity."""

    field_ref: str
    kind: SurfaceKind
    spans: tuple[tuple[int, int], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class PixelRedaction:
    """Protected original word-box mask and source span, before image rewriting."""

    redaction_ref: str
    page: int
    region_refs: tuple[str, ...]
    span: tuple[int, int] = field(repr=False)
    bounds: Rectangle = field(repr=False)


@dataclass(frozen=True, slots=True)
class AnnotationPlan:
    """Reviewable source-bound edits, with no source text, pixels or export bytes."""

    source_fingerprint: str
    policy_digest: str
    detector_digest: str
    ocr_digest: str | None
    fields: tuple[FieldRedaction, ...] = field(repr=False)
    pixels: tuple[PixelRedaction, ...] = field(repr=False)
    region_refs: tuple[str, ...] = field(repr=False)
    findings: tuple[PrivacyReason, ...]
    lead_findings: tuple[LeadFindingCode, ...]

    @property
    def digest(self) -> str:
        """Bind every edit and source/policy reference for an exact review."""
        return _hash({"algorithm": ANNOTATION_VERSION, "plan": asdict(self)})

    def to_report_json(self) -> str:
        """Counts, controlled findings and digests only; no raw offsets or values."""
        return _json(
            {
                "algorithm": ANNOTATION_VERSION,
                "plan_digest": self.digest,
                "source_fingerprint": self.source_fingerprint,
                "field_count": len(self.fields),
                "pixel_mask_count": len(self.pixels),
                "region_count": len(self.region_refs),
                "findings": self.findings,
                "lead_findings": self.lead_findings,
                "notice": ANNOTATION_NOTICE,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AnnotationReview:
    """Host review of every exact inventoried field and image region.

    The host must review extraction completeness, OCR omissions/handwriting,
    retained traces and preservation of clinically meaningful calibration and
    labels. Enumerating these references records that decision, not a signature
    or independent proof. A missing reference prevents export.
    """

    plan_digest: str
    reviewer_ref: str
    evidence_digest: str
    reviewed_field_refs: tuple[str, ...] = field(repr=False)
    reviewed_region_refs: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        for value in (self.plan_digest, self.evidence_digest):
            _digest(value)
        _require(
            type(self.reviewer_ref) is str
            and re.fullmatch(r"[0-9a-f]{32}", self.reviewer_ref) is not None
        )
        for refs in (self.reviewed_field_refs, self.reviewed_region_refs):
            _require(type(refs) is tuple and len(refs) <= 4096)
            for value in refs:
                _digest(value)
            _require(len(set(refs)) == len(refs))


@dataclass(frozen=True, slots=True)
class RedactedEcgAnnotations:
    """Verified protected bundle; unflagged clinical text and traces remain private."""

    structured_json: str = field(repr=False)
    png_pages: tuple[bytes, ...] = field(repr=False)
    waveform: EcgWaveform | None = field(repr=False)
    provenance_json: str = field(repr=False)
    plan_digest: str
    review_digest: str

    def to_report_json(self) -> str:
        """Content-free audit, distinct from protected bundle/provenance payloads."""
        return _json(
            {
                "algorithm": ANNOTATION_VERSION,
                "plan_digest": self.plan_digest,
                "review_digest": self.review_digest,
                "structured_digest": hashlib.sha256(
                    self.structured_json.encode()
                ).hexdigest(),
                "page_digests": [
                    hashlib.sha256(item).hexdigest() for item in self.png_pages
                ],
                "provenance_digest": hashlib.sha256(
                    self.provenance_json.encode()
                ).hexdigest(),
                "has_waveform": self.waveform is not None,
                "notice": ANNOTATION_NOTICE,
            }
        )


@dataclass(frozen=True, slots=True)
class _Raster:
    size: tuple[int, int]
    pixels: bytes = field(repr=False)
    regions: tuple[PageRegion, ...] = field(repr=False)
    fingerprint: str


def _area(bounds: Rectangle) -> int:
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _intersection(first: Rectangle, second: Rectangle) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _fields(source: EcgAnnotationSource) -> tuple[AnnotationField, ...]:
    return (
        (AnnotationField(SurfaceKind.FILENAME, "filename", source.filename),)
        + source.fields
        + tuple(
            AnnotationField(SurfaceKind.LEAD_LABEL, f"lead-{index}", item.label)
            for index, item in enumerate(source.observations)
        )
    )


def _capture(
    source: EcgAnnotationSource, policy: AnnotationPolicy
) -> tuple[str, tuple[_Raster, ...]]:
    _require(type(source) is EcgAnnotationSource and type(policy) is AnnotationPolicy)
    fields = _fields(source)
    _require(
        sum(len(item.name) + len(item.value) + 1 for item in fields)
        <= policy.maximum_text_chars,
        PrivacyReason.LIMIT,
    )
    rasters = []
    total_pixels = 0
    trace_slots = []
    region_refs = []
    for page in source.pages:
        from PIL import Image

        _require(isinstance(page.image, Image.Image))
        width, height = page.image.size
        _require(
            width > 0 and height > 0 and width * height <= policy.maximum_page_pixels,
            PrivacyReason.LIMIT,
        )
        total_pixels += width * height
        _require(total_pixels <= policy.maximum_total_pixels, PrivacyReason.LIMIT)
        _require(
            getattr(page.image, "n_frames", 1) == 1
            and page.image.getexif().get(274, 1) == 1
        )
        for index, region in enumerate(page.regions):
            _require(
                region.bounds[2] <= width and region.bounds[3] <= height,
                PrivacyReason.COVERAGE,
            )
            _require(
                all(
                    _intersection(region.bounds, other.bounds) == 0
                    for other in page.regions[:index]
                ),
                PrivacyReason.COVERAGE,
            )
            if region.kind is RegionKind.TRACE:
                trace_slots.append(region.slot)
            region_refs.append(region.region_ref)
        _require(
            sum(_area(region.bounds) for region in page.regions) == width * height,
            PrivacyReason.COVERAGE,
        )
        with (
            page.image.convert("RGBA") as rgba,
            Image.new("RGBA", page.image.size, "white") as white,
        ):
            with (
                Image.alpha_composite(white, rgba) as flattened,
                flattened.convert("RGB") as rgb,
            ):
                pixels = rgb.tobytes()
        fingerprint = _hash(
            {
                "size": [width, height],
                "pixels": hashlib.sha256(pixels).hexdigest(),
                "regions": [asdict(item) for item in page.regions],
            }
        )
        rasters.append(_Raster((width, height), pixels, page.regions, fingerprint))
    _require(len(set(region_refs)) == len(region_refs), PrivacyReason.COVERAGE)
    if source.pages:
        expected = {(item.row, item.column) for item in source.layout.slots}
        _require(
            len(trace_slots) == len(expected) and set(trace_slots) == expected,
            PrivacyReason.COVERAGE,
        )
    fingerprint = _hash(
        {
            "algorithm": ANNOTATION_VERSION,
            "source_digest": source.source_digest,
            "extraction_digest": source.extraction_digest,
            "fields": [asdict(item) for item in fields],
            "layout": asdict(source.layout),
            "observations": [asdict(item) for item in source.observations],
            "acquisition": source.acquisition,
            "waveform": asdict(source.waveform)
            if source.waveform is not None
            else None,
            "pages": [item.fingerprint for item in rasters],
        }
    )
    return fingerprint, tuple(rasters)


def _merge(spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(set(spans)):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return tuple(result)


def _detect(
    text: str, detector: LocalAnnotationDetector, policy: AnnotationPolicy
) -> tuple[tuple[int, int], ...]:
    _require(len(text) <= policy.maximum_text_chars, PrivacyReason.LIMIT)
    _require(
        detector.local_only is True
        and getattr(detector.detect, "network_egress", False) is False
        and getattr(detector.detect, "allows_network", False) is False,
        PrivacyReason.OFFLINE,
    )
    detected = detector.detect(text)
    _require(type(detected) in (tuple, list) and len(detected) <= 25_000)
    _require(all(type(item) is OpenMedSpan for item in detected))
    # Sweep independently: a short ML span must not suppress a whole structured ID.
    sweep = safety_sweep(text, (), lang=policy.language)
    spans = []
    for item in (*detected, *sweep):
        _require(
            type(item.start) is int
            and type(item.end) is int
            and 0 <= item.start < item.end <= len(text)
        )
        spans.append((item.start, item.end))
    _require(len(spans) <= 25_000, PrivacyReason.LIMIT)
    return _merge(spans)


def _recognize(
    raster: _Raster, ocr: LocalAnnotationOcr, policy: AnnotationPolicy
) -> OcrResult:
    from PIL import Image

    with Image.frombytes("RGB", raster.size, raster.pixels) as image:
        result = ocr.engine.recognize(image, languages=(policy.language,))
    _require(
        type(result) is OcrResult
        and type(result.words) is tuple
        and len(result.words) <= 25_000
    )
    _require(
        sum(
            len(word.text) + 1
            for word in result.words
            if type(word) is OcrWord and type(word.text) is str
        )
        <= policy.maximum_text_chars,
        PrivacyReason.LIMIT,
    )
    for word in result.words:
        _require(
            type(word) is OcrWord and type(word.text) is str and bool(word.text.strip())
        )
        _require(type(word.page) is int and word.page == 0)
        _require(type(word.bbox) is tuple and len(word.bbox) == 4)
        _require(all(type(v) in (int, float) and math.isfinite(v) for v in word.bbox))
        x0, y0, x1, y1 = word.bbox
        _require(0 <= x0 < x1 <= raster.size[0] and 0 <= y0 < y1 <= raster.size[1])
        _require(
            type(word.confidence) in (int, float)
            and math.isfinite(word.confidence)
            and 0 <= word.confidence <= 1
        )
    return result


def _contained(bounds: Rectangle, regions: tuple[PageRegion, ...]) -> tuple[str, ...]:
    overlaps = tuple(
        item
        for item in regions
        if item.kind is RegionKind.ANNOTATION and _intersection(bounds, item.bounds)
    )
    if sum(_intersection(bounds, item.bounds) for item in overlaps) != _area(bounds):
        return ()
    return tuple(item.region_ref for item in overlaps)


def _known_spans(
    text: str, known: CustomRecognizer, policy: AnnotationPolicy
) -> tuple[tuple[int, int], ...]:
    # Literal tokens with flexible whitespace and case. Do not turn a short name into a
    # substring rule that silently removes unrelated clinical vocabulary.
    normalized = normalize_for_pii_detection(
        text, lang=policy.language, preserve_script_letters=True
    )
    text = normalized.text
    spans = []
    for match in known.deny_matches(text):
        start, end = match.start, match.end
        if start and re.match(r"\w", text[start - 1]) and re.match(r"\w", text[start]):
            continue
        if (
            end < len(text)
            and re.match(r"\w", text[end - 1])
            and re.match(r"\w", text[end])
        ):
            continue
        spans.append(normalized.remap_span(start, end))
    _require(len(spans) <= 25_000, PrivacyReason.LIMIT)
    return _merge(spans)


def _prepare(
    source: EcgAnnotationSource,
    policy: AnnotationPolicy,
    detector: LocalAnnotationDetector,
    ocr: LocalAnnotationOcr | None,
) -> tuple[AnnotationPlan, CustomRecognizer]:
    _require(type(detector) is LocalAnnotationDetector)
    _require(ocr is None or type(ocr) is LocalAnnotationOcr)
    _require(not source.pages or ocr is not None, PrivacyReason.COVERAGE)
    detector_digest = detector.digest
    ocr_digest = ocr.digest if ocr is not None else None
    fingerprint, rasters = _capture(source, policy)
    inventory = _fields(source)
    field_texts = tuple(item.name + "\n" + item.value for item in inventory)
    field_hits = tuple(_detect(text, detector, policy) for text in field_texts)
    recognized_pages = []
    page_hits = []
    text_chars = sum(map(len, field_texts))
    known_values = set()
    for item, text, spans in zip(inventory, field_texts, field_hits, strict=True):
        boundary = len(item.name) + 1
        for start, end in spans:
            # A detector may include the header key in a span. Propagate its
            # value, or the matched name when the hit is wholly in that name.
            value = text[
                max(start, boundary) if end > boundary else start : end
            ].strip()
            if value:
                known_values.add(value)
    for raster in rasters:
        assert ocr is not None
        recognized = _recognize(raster, ocr, policy)
        recognized_pages.append(recognized)
        text_chars += len(recognized.text)
        _require(text_chars <= policy.maximum_text_chars, PrivacyReason.LIMIT)
        hits = _detect(recognized.text, detector, policy)
        page_hits.append(hits)
        known_values.update(
            value
            for start, end in hits
            if (value := recognized.text[start:end].strip())
        )
    normalized_values = set()
    for value in known_values:
        value = normalize_for_pii_detection(
            value, lang=policy.language, preserve_script_letters=True
        ).text.strip()
        if value:
            normalized_values.add(value)
            # A sweep may include a context label such as "MRN:" in its hit.
            # Retain long numeric tokens from confirmed identifier spans too;
            # this does not classify unrelated unlabelled numbers as identifiers.
            normalized_values.update(re.findall(r"(?<!\w)\d{6,}(?!\w)", value))
    _require(
        text_chars * len(normalized_values) <= MAX_CUSTOM_RECOGNIZER_SCAN_WORK,
        PrivacyReason.LIMIT,
    )
    known = CustomRecognizer(
        deny_patterns=[
            {
                "pattern": r"\s+".join(re.escape(token) for token in value.split()),
                "label": "ID_NUM",
            }
            for value in sorted(normalized_values)
        ],
        case_sensitive=False,
    )
    fields = tuple(
        FieldRedaction(
            _hash([fingerprint, "field", index]),
            item.kind,
            ((0, len(item.name) + 1 + len(item.value)),)
            if item.kind is SurfaceKind.FILENAME
            else _merge((*hits, *_known_spans(text, known, policy))),
        )
        for index, (item, text, hits) in enumerate(
            zip(inventory, field_texts, field_hits, strict=True)
        )
    )
    findings: list[PrivacyReason] = []
    if any(item.kind is SurfaceKind.LEAD_LABEL and item.spans for item in fields):
        findings.append(PrivacyReason.LAYOUT)
    report = assess_lead_layout(
        source.observations,
        source.layout,
        source_digest=source.source_digest,
        acquisition=source.acquisition,
        image_labels_confirmed=False,
    )
    identity_codes = {
        LeadFindingCode.UNKNOWN_LEAD,
        LeadFindingCode.MISSING_SLOT,
        LeadFindingCode.UNDECLARED_SLOT,
        LeadFindingCode.DUPLICATE_SLOT,
        LeadFindingCode.WRONG_LEAD,
        LeadFindingCode.DUPLICATE_LEAD,
    }
    if any(item.code in identity_codes for item in report.findings):
        findings.append(PrivacyReason.LAYOUT)
    pixels = []
    for index, (raster, recognized, hits) in enumerate(
        zip(rasters, recognized_pages, page_hits, strict=True)
    ):
        text = recognized.text
        offsets = []
        cursor = 0
        for word in recognized.words:
            end = cursor + len(word.text)
            bounds = (
                math.floor(word.bbox[0]),
                math.floor(word.bbox[1]),
                math.ceil(word.bbox[2]),
                math.ceil(word.bbox[3]),
            )
            if not _contained(bounds, raster.regions):
                findings.append(PrivacyReason.COVERAGE)
            offsets.append((cursor, end, bounds))
            cursor = end + 1
        detected_spans = _merge((*hits, *_known_spans(text, known, policy)))
        word_hits: set[int] = set()
        word_cursor = 0
        for span in detected_spans:
            while word_cursor < len(offsets) and offsets[word_cursor][1] <= span[0]:
                word_cursor += 1
            matched = False
            match_cursor = word_cursor
            while match_cursor < len(offsets) and offsets[match_cursor][0] < span[1]:
                word_hits.add(match_cursor)
                matched = True
                match_cursor += 1
            _require(matched)
        retained_words = [
            box
            for word_index, (_a, _b, box) in enumerate(offsets)
            if word_index not in word_hits
        ]
        _require(len(word_hits) * len(retained_words) <= 2_000_000, PrivacyReason.LIMIT)
        for word_index in sorted(word_hits):
            start, end, box = offsets[word_index]
            pad = policy.ocr_padding_pixels
            bounds = (
                max(0, box[0] - pad),
                max(0, box[1] - pad),
                min(raster.size[0], box[2] + pad),
                min(raster.size[1], box[3] + pad),
            )
            regions = _contained(bounds, raster.regions)
            if not regions:
                findings.append(PrivacyReason.COVERAGE)
            if any(
                part.surface is SurfaceKind.LEAD_LABEL
                and _intersection(bounds, part.bounds)
                for part in raster.regions
            ):
                findings.append(PrivacyReason.LAYOUT)
            if any(_intersection(bounds, box) for box in retained_words):
                # Padding must not silently remove adjacent calibration or text.
                findings.append(PrivacyReason.COVERAGE)
            pixels.append(
                PixelRedaction(
                    _hash(
                        [fingerprint, "pixel", index, start, end, word_index, bounds]
                    ),
                    index,
                    regions,
                    (start, end),
                    bounds,
                )
            )
    _require(_capture(source, policy)[0] == fingerprint, PrivacyReason.STALE)
    _require(
        detector.digest == detector_digest
        and (ocr.digest if ocr is not None else None) == ocr_digest,
        PrivacyReason.STALE,
    )
    return AnnotationPlan(
        fingerprint,
        policy.digest,
        detector_digest,
        ocr_digest,
        fields,
        tuple(pixels),
        tuple(
            _hash([fingerprint, "region", index, item.region_ref])
            for index, raster in enumerate(rasters)
            for item in raster.regions
        ),
        tuple(dict.fromkeys(findings)),
        tuple(dict.fromkeys(item.code for item in report.findings)),
    ), known


def prepare_annotation_privacy(
    source: EcgAnnotationSource,
    policy: AnnotationPolicy,
    detector: LocalAnnotationDetector,
    ocr: LocalAnnotationOcr | None = None,
) -> AnnotationPlan:
    """Inventory actual fields/rasters and plan coordinated local redaction.

    Args:
        source: Complete protected extraction, actual waveform and rendered pages.
        policy: Qualified language and explicit resource/projection limits.
        detector: Already loaded local identifier detector over original code points.
        ocr: Native streaming OCR runtime; mandatory when source pages exist.

    Returns:
        Source-bound offsets/masks and controlled coverage/lead-identity findings.
        No clinical values, image bytes or raw text are returned in this plan.

    Raises:
        AnnotationPrivacyError: Fixed declaration/resource/backend refusal.
    """
    return _protected_call(lambda: _prepare(source, policy, detector, ocr)[0])


def _rewrite(text: str, spans: tuple[tuple[int, int], ...], origin: int = 0) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        left, right = max(0, start - origin), min(len(text), end - origin)
        if left < right:
            parts.extend((text[cursor:left], _MASK))
            cursor = right
    parts.append(text[cursor:])
    return "".join(parts)


def _export(
    source: EcgAnnotationSource,
    policy: AnnotationPolicy,
    detector: LocalAnnotationDetector,
    ocr: LocalAnnotationOcr | None,
    plan: AnnotationPlan,
    review: AnnotationReview,
) -> RedactedEcgAnnotations:
    _require(type(plan) is AnnotationPlan and type(review) is AnnotationReview)
    fresh, known = _prepare(source, policy, detector, ocr)
    _require(fresh == plan, PrivacyReason.STALE)
    _require(
        not plan.findings, plan.findings[0] if plan.findings else PrivacyReason.COVERAGE
    )
    _require(
        review.plan_digest == plan.digest
        and set(review.reviewed_field_refs) == {item.field_ref for item in plan.fields}
        and set(review.reviewed_region_refs) == set(plan.region_refs),
        PrivacyReason.REVIEW,
    )
    fingerprint, rasters = _capture(source, policy)
    rewritten = []
    for item, edits in zip(_fields(source), plan.fields, strict=True):
        if item.kind is SurfaceKind.FILENAME:
            name, value = "filename", "ecg-redacted"
        else:
            name = _rewrite(item.name, edits.spans)
            value = _rewrite(item.value, edits.spans, len(item.name) + 1)
        _require(
            not _detect(name + "\n" + value, detector, policy)
            and not _known_spans(name + "\n" + value, known, policy),
            PrivacyReason.RESIDUAL,
        )
        rewritten.append(
            {
                "field_ref": edits.field_ref,
                "kind": item.kind,
                "name": name,
                "value": value,
            }
        )
    encoded = []
    total_bytes = 0
    for index, raster in enumerate(rasters):
        from PIL import Image

        assert ocr is not None
        masks = tuple(item.bounds for item in plan.pixels if item.page == index)
        with Image.frombytes("RGB", raster.size, raster.pixels) as image:
            result = render_redacted_raster_pages(
                (RasterRedactionPage(image, masks),),
                output_format="png",
                padding_pixels=0,
                max_pages=1,
                max_page_pixels=policy.maximum_page_pixels,
                max_total_pixels=policy.maximum_total_pixels,
                max_regions=25_000,
                max_output_bytes=policy.maximum_output_bytes,
            )
        total_bytes += len(result.content)
        _require(total_bytes <= policy.maximum_output_bytes, PrivacyReason.LIMIT)
        with Image.open(BytesIO(result.content)) as image:
            output = _Raster(image.size, image.tobytes(), raster.regions, "")
        verified = _recognize(output, ocr, policy)
        _require(
            not _detect(verified.text, detector, policy)
            and not _known_spans(verified.text, known, policy),
            PrivacyReason.RESIDUAL,
        )
        encoded.append(result.content)
    _require(
        _capture(source, policy)[0] == fingerprint == plan.source_fingerprint,
        PrivacyReason.STALE,
    )
    _require(
        policy.digest == plan.policy_digest
        and detector.digest == plan.detector_digest
        and (ocr.digest if ocr is not None else None) == plan.ocr_digest,
        PrivacyReason.STALE,
    )
    structured = _json(
        {
            "schema": ANNOTATION_VERSION,
            "filename": "ecg-redacted",
            "fields": rewritten,
            "page_names": [
                f"ecg-page-{index:04d}.png" for index in range(len(encoded))
            ],
            "lead_layout": asdict(source.layout),
            "notice": ANNOTATION_NOTICE,
        }
    )
    _require(len(structured) <= policy.maximum_text_chars * 4, PrivacyReason.LIMIT)
    provenance = _json(
        {
            "schema": ANNOTATION_VERSION,
            "plan": asdict(plan),
            "review": asdict(review),
            "source_digest": source.source_digest,
            "extraction_digest": source.extraction_digest,
            "regions": [
                [asdict(item) for item in raster.regions] for raster in rasters
            ],
            "pixel_verification": "exact-decoded-raster",
            "source_metadata": "not-copied",
            "source_text_layer": "not-copied",
            "waveform_modified": False,
            "notice": ANNOTATION_NOTICE,
        }
    )
    _require(
        total_bytes + len(structured.encode()) + len(provenance.encode())
        <= policy.maximum_output_bytes,
        PrivacyReason.LIMIT,
    )
    return RedactedEcgAnnotations(
        structured,
        tuple(encoded),
        source.waveform,
        provenance,
        plan.digest,
        _hash(asdict(review)),
    )


def export_annotation_privacy(
    source: EcgAnnotationSource,
    policy: AnnotationPolicy,
    detector: LocalAnnotationDetector,
    plan: AnnotationPlan,
    review: AnnotationReview,
    ocr: LocalAnnotationOcr | None = None,
) -> RedactedEcgAnnotations:
    """Export only after source-bound review and fresh text/pixel residual checks.

    Args:
        source: Original complete source; no in-place writes are performed.
        policy: The exact policy used to prepare the reviewed plan.
        detector: Same qualified local detector, called again on actual outputs.
        plan: Exact prepared edit/coverage inventory.
        review: Explicit host decision covering every current field and region.
        ocr: Same streaming runtime, called on each exact encoded output raster.

    Returns:
        Protected rewritten structured values, verified PNG pages, unchanged
        normalized waveform and stable field/region provenance, all in memory.
        No bytes are returned if any page, residual or coverage check fails.

    Raises:
        AnnotationPrivacyError: Missing review, changed source/plan, unknown
            coverage, residual identifier, resource or local backend failure.

    No clinical approval or native source-container rewrite is implied. The host
    must retain the bundle's waveform, calibration and provenance together.
    """
    return _protected_call(lambda: _export(source, policy, detector, ocr, plan, review))
