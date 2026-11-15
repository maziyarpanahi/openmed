"""Shared evidence coordinates for native OpenMedKit document intake.

Swift and Android OpenMedKit render PDF pages and images on device, hand each
page to a local OCR engine as an upright image and receive text with pixel
boxes. This module is the reference mapping from that OCR image back to the
original page. Swift and Kotlin implement the same steps and are checked
against ``tests/fixtures/parity/document_intake_v1.json``:

1. undo the render scale between the OCR image and the upright crop;
2. undo the clockwise quarter-turn rotation that made the crop upright;
3. undo the crop, giving top-left coordinates on the original page.

Every mapped page, declared crop and recognized region becomes a ``visual``
packet from :mod:`openmed.multimodal.apple_contract`, so its source box,
Unicode scalar span and display rotation are revalidated by that contract.
Pages that cannot be mapped are held with a closed reason code, and the report
outcome is ``complete``, ``partial`` or ``refused``. Records contain
coordinates, offsets, opaque references and digests. The extracted text is the
only protected value; it is excluded from ``repr`` and from
:meth:`IntakeReport.to_dict`, and no error interpolates a rejected value.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .apple_contract import AppleContractError, MultimodalPacket, project_visual
from .evidence.fused_graph import (
    EvidenceNode,
    FusedEvidenceError,
    NodeKind,
    PageGeometry,
    SpatialUnit,
)

__all__ = [
    "BOX_TOLERANCE",
    "PAGE_SEPARATOR",
    "SCHEMA_VERSION",
    "IntakeLimits",
    "IntakeOutcome",
    "IntakePage",
    "IntakePageGeometry",
    "IntakePageInput",
    "IntakeReason",
    "IntakeRegion",
    "IntakeReport",
    "NativeIntakeError",
    "PageStatus",
    "Recognition",
    "RecognizedRegion",
    "intake_node_ref",
    "map_document",
    "map_recognized_box",
    "refuse_document",
]

SCHEMA_VERSION = "openmed.multimodal.document-intake.v1"
PAGE_SEPARATOR = "\n\n"
# OCR engines report boxes that overshoot the image edge by a few pixels. A
# coordinate up to this fraction of the image dimension outside the image is
# clamped to the edge; anything further suggests a different coordinate space.
BOX_TOLERANCE = 0.01
_MAX_COORDINATE = 2**31 - 1
_QUARTER_TURNS = (0, 90, 180, 270)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")

Box = tuple[float, float, float, float]


class NativeIntakeError(ValueError):
    """Invalid reference-mapper input, with no rejected values attached.

    Attributes:
        reason: The hold reason when a box could not be mapped, else ``None``.
    """

    def __init__(self, reason: IntakeReason | None = None) -> None:
        self.reason = reason
        super().__init__(
            "invalid native document intake input" if reason is None else reason.value
        )


class IntakeOutcome(str, Enum):
    """Document-level result shared by Python, Swift and Android."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    REFUSED = "refused"


class IntakeReason(str, Enum):
    """Closed, value-free reason vocabulary, in report order."""

    UNSUPPORTED_DOCUMENT = "unsupported_document"
    DOCUMENT_LOCKED = "document_locked"
    PERMISSION_DENIED = "permission_denied"
    SIZE_LIMIT = "size_limit"
    EMPTY_DOCUMENT = "empty_document"
    PAGE_LIMIT = "page_limit"
    PAGE_MISSING = "page_missing"
    PERMISSION_LOST = "permission_lost"
    RECOGNITION_FAILED = "recognition_failed"
    RESOURCE_LIMIT = "resource_limit"
    AMBIGUOUS_TRANSFORM = "ambiguous_transform"
    REGION_OUT_OF_BOUNDS = "region_out_of_bounds"
    INVALID_RECOGNITION = "invalid_recognition"


_ORDER = {reason: index for index, reason in enumerate(IntakeReason)}
_DOCUMENT_REFUSALS = frozenset(
    {
        IntakeReason.UNSUPPORTED_DOCUMENT,
        IntakeReason.DOCUMENT_LOCKED,
        IntakeReason.PERMISSION_DENIED,
        IntakeReason.SIZE_LIMIT,
    }
)
_PAGE_UNAVAILABLE = frozenset(
    {
        IntakeReason.PAGE_LIMIT,
        IntakeReason.PAGE_MISSING,
        IntakeReason.PERMISSION_LOST,
        IntakeReason.RECOGNITION_FAILED,
        IntakeReason.RESOURCE_LIMIT,
        IntakeReason.AMBIGUOUS_TRANSFORM,
    }
)


class PageStatus(str, Enum):
    """Whether a page contributed text and evidence, or was held."""

    MAPPED = "mapped"
    HELD = "held"


def _require(condition: bool) -> None:
    if not condition:
        raise NativeIntakeError()


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_number(value: object) -> bool:
    return type(value) in (int, float)


@dataclass(frozen=True, slots=True, kw_only=True)
class IntakeLimits:
    """Bounds applied before any page contributes text.

    Args:
        max_pages: Pages read before the remainder is reported as ``page_limit``.
        max_regions_per_page: Recognized regions accepted on one page.
        max_page_scalars: Unicode scalars accepted in one page's text.
    """

    max_pages: int = 2048
    max_regions_per_page: int = 4096
    max_page_scalars: int = 65536

    def __post_init__(self) -> None:
        for value in (self.max_pages, self.max_regions_per_page, self.max_page_scalars):
            _require(_is_int(value) and 1 <= value <= _MAX_COORDINATE)


@dataclass(frozen=True, slots=True, kw_only=True)
class IntakePageGeometry:
    """Original page extent plus the crop and rotation handed to OCR.

    Values are checked when the page is mapped, so an inconsistent native
    declaration becomes an ``ambiguous_transform`` hold instead of an error.

    Args:
        width: Original page width with a top-left origin, in ``unit``.
        height: Original page height.
        unit: Pixels for images, PDF points for PDF pages.
        rotation: Clockwise quarter turn that made the crop upright for OCR.
        crop: Half-open ``(left, top, right, bottom)`` original-page region
            handed to OCR. ``None`` means the whole page.
    """

    width: float
    height: float
    unit: SpatialUnit = SpatialUnit.PIXEL
    rotation: int = 0
    crop: Box | None = None

    def __post_init__(self) -> None:
        _require(_is_number(self.width) and _is_number(self.height))
        _require(type(self.unit) is SpatialUnit and _is_int(self.rotation))
        if self.crop is not None:
            _require(type(self.crop) is tuple and len(self.crop) == 4)
            _require(all(_is_number(value) for value in self.crop))


@dataclass(frozen=True, slots=True, kw_only=True)
class RecognizedRegion:
    """One OCR region in the upright OCR image.

    Args:
        span: Half-open Unicode scalar offsets into the page text.
        box: Half-open ``(left, top, right, bottom)`` OCR-image pixels.
    """

    span: tuple[int, int]
    box: Box

    def __post_init__(self) -> None:
        _require(type(self.span) is tuple and len(self.span) == 2)
        _require(all(_is_int(value) for value in self.span))
        _require(type(self.box) is tuple and len(self.box) == 4)
        _require(all(_is_number(value) for value in self.box))


@dataclass(frozen=True, slots=True, kw_only=True)
class Recognition:
    """OCR output for one upright page image.

    Args:
        width: OCR image width in pixels.
        height: OCR image height in pixels.
        text: Recognized page text; protected and never repeated in ``repr``.
        regions: Regions in engine reading order.
    """

    width: int
    height: int
    text: str = field(repr=False)
    regions: tuple[RecognizedRegion, ...] = ()

    def __post_init__(self) -> None:
        _require(_is_int(self.width) and _is_int(self.height))
        _require(type(self.text) is str and type(self.regions) is tuple)
        _require(all(type(region) is RecognizedRegion for region in self.regions))


@dataclass(frozen=True, slots=True, kw_only=True)
class IntakePageInput:
    """What a native source produced for one page, in page order.

    A page is either rendered and recognized, or unavailable with a reason.
    ``permission_lost`` ends the document: later pages are never read.
    ``page_limit`` marks a source that stopped at :class:`IntakeLimits`.

    Args:
        geometry: Original page geometry for a rendered page.
        recognition: OCR output for a rendered page.
        unavailable: Reason a page could not be rendered or recognized.
    """

    geometry: IntakePageGeometry | None = None
    recognition: Recognition | None = field(default=None, repr=False)
    unavailable: IntakeReason | None = None

    def __post_init__(self) -> None:
        if self.unavailable is None:
            _require(type(self.geometry) is IntakePageGeometry)
            _require(type(self.recognition) is Recognition)
        else:
            _require(type(self.unavailable) is IntakeReason)
            _require(self.unavailable in _PAGE_UNAVAILABLE)
            _require(self.geometry is None and self.recognition is None)


@dataclass(frozen=True, slots=True)
class IntakePage:
    """Page record: mapped with text and evidence, or held with a reason."""

    page: int
    status: PageStatus
    reason: IntakeReason | None = None
    text_span: tuple[int, int] | None = None
    evidence: MultimodalPacket | None = None
    crop_evidence: MultimodalPacket | None = None

    @property
    def node_ref(self) -> str | None:
        """Return the opaque page reference shared with Swift and Android."""
        return None if self.evidence is None else self.evidence.payload()["node_ref"]


@dataclass(frozen=True, slots=True)
class IntakeRegion:
    """Recognized region mapped to original-page coordinates.

    Attributes:
        page: Zero-based page index.
        span: Half-open Unicode scalar offsets into :attr:`IntakeReport.text`.
        box: Original-page ``(left, top, right, bottom)`` with a top-left origin.
        node_ref: Opaque reference shared with Swift and Android.
        evidence: Validated ``ocr_span`` visual packet.
    """

    page: int
    span: tuple[int, int]
    box: Box
    node_ref: str
    evidence: MultimodalPacket


@dataclass(frozen=True, slots=True)
class IntakeReport:
    """Bounded intake result with explicit partial and refused outcomes."""

    outcome: IntakeOutcome
    reasons: tuple[IntakeReason, ...]
    text: str = field(repr=False)
    pages: tuple[IntakePage, ...] = ()
    regions: tuple[IntakeRegion, ...] = ()

    def regions_for_span(self, start: int, end: int) -> tuple[IntakeRegion, ...]:
        """Return regions whose scalar span overlaps ``[start, end)``.

        Args:
            start: First Unicode scalar offset into :attr:`text`.
            end: Exclusive end offset.

        Raises:
            NativeIntakeError: If the span is not inside the report text.
        """
        _require(_is_int(start) and _is_int(end) and 0 <= start <= end)
        _require(end <= len(self.text))
        return tuple(
            region
            for region in self.regions
            if region.span[0] < end and start < region.span[1]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return value-free metadata: codes, offsets, references and digests."""
        return {
            "schema_version": SCHEMA_VERSION,
            "outcome": self.outcome.value,
            "reasons": [reason.value for reason in self.reasons],
            "text_scalars": len(self.text),
            "pages": [
                {
                    "page": page.page,
                    "status": page.status.value,
                    "reason": None if page.reason is None else page.reason.value,
                    "text_span": None
                    if page.text_span is None
                    else list(page.text_span),
                    "node_ref": page.node_ref,
                    "evidence_digest": None
                    if page.evidence is None
                    else page.evidence.evidence_digest,
                    "crop_evidence_digest": None
                    if page.crop_evidence is None
                    else page.crop_evidence.evidence_digest,
                }
                for page in self.pages
            ],
            "regions": [
                {
                    "page": region.page,
                    "span": list(region.span),
                    "node_ref": region.node_ref,
                    "evidence_digest": region.evidence.evidence_digest,
                }
                for region in self.regions
            ],
        }


class _Hold(Exception):
    def __init__(self, reason: IntakeReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


def intake_node_ref(source_digest: str, page: int, label: str) -> str:
    """Return the opaque 128-bit reference for one page, crop or region.

    Args:
        source_digest: Lowercase SHA-256 of the source document bytes.
        page: Zero-based page index.
        label: ``"page"``, ``"crop"`` or ``"region:<index>"``.
    """
    _require(type(source_digest) is str and _DIGEST.fullmatch(source_digest))
    _require(_is_int(page) and page >= 0 and type(label) is str)
    identity = f"{SCHEMA_VERSION}|{source_digest}|{page}|{label}"
    return hashlib.sha256(identity.encode("ascii")).hexdigest()[:32]


def _finite(value: float, maximum: float = _MAX_COORDINATE) -> bool:
    return math.isfinite(value) and 0 <= value <= maximum


def _float(value: float) -> float:
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _checked_crop(geometry: IntakePageGeometry, width: int, height: int) -> Box:
    page_width, page_height = _float(geometry.width), _float(geometry.height)
    if not (_finite(page_width) and _finite(page_height)):
        raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM)
    if page_width <= 0 or page_height <= 0:
        raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM)
    if geometry.rotation not in _QUARTER_TURNS:
        raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM)
    if not (1 <= width <= _MAX_COORDINATE and 1 <= height <= _MAX_COORDINATE):
        raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM)
    if geometry.crop is None:
        left, top, right, bottom = 0.0, 0.0, page_width, page_height
    else:
        left, top, right, bottom = (_float(value) for value in geometry.crop)
        if not all(_finite(value) for value in (left, top, right, bottom)):
            raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM)
        if not (left < right <= page_width and top < bottom <= page_height):
            raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM)
    crop_width, crop_height = right - left, bottom - top
    upright_width, upright_height = (
        (crop_height, crop_width)
        if geometry.rotation in (90, 270)
        else (crop_width, crop_height)
    )
    # Rendering rounds each OCR dimension to a whole pixel, so the image may
    # differ from the upright crop's aspect by at most half a pixel per side.
    skew = abs(width * upright_height - height * upright_width)
    if not skew <= (upright_width + upright_height) * 0.5:
        raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM)
    return left, top, right, bottom


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def map_recognized_box(
    geometry: IntakePageGeometry, width: int, height: int, box: Box
) -> Box:
    """Map one OCR-image box back to original-page coordinates.

    The arithmetic order is part of the shared contract so that Swift and
    Kotlin produce the same binary64 values.

    Args:
        geometry: Original page geometry, crop and rotation.
        width: OCR image width in pixels.
        height: OCR image height in pixels.
        box: OCR-image ``(left, top, right, bottom)`` with a top-left origin.

    Returns:
        The box on the original page, top-left origin, in ``geometry.unit``.

    Raises:
        NativeIntakeError: With the hold reason and no values attached, when
            the geometry or box cannot be mapped unambiguously.
    """
    _require(type(geometry) is IntakePageGeometry)
    _require(_is_int(width) and _is_int(height))
    _require(type(box) is tuple and len(box) == 4)
    _require(all(_is_number(value) for value in box))
    try:
        return _map_box(geometry, width, height, box)
    except _Hold as hold:
        raise NativeIntakeError(hold.reason) from None


def _map_box(geometry: IntakePageGeometry, width: int, height: int, box: Box) -> Box:
    left, top, right, bottom = _checked_crop(geometry, width, height)
    crop_width, crop_height = right - left, bottom - top
    rotation = geometry.rotation
    upright_width, upright_height = (
        (crop_height, crop_width)
        if rotation in (90, 270)
        else (crop_width, crop_height)
    )
    x0, y0, x1, y1 = (_float(value) for value in box)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise _Hold(IntakeReason.REGION_OUT_OF_BOUNDS)
    tolerance_x, tolerance_y = width * BOX_TOLERANCE, height * BOX_TOLERANCE
    if x0 < -tolerance_x or y0 < -tolerance_y:
        raise _Hold(IntakeReason.REGION_OUT_OF_BOUNDS)
    if x1 > width + tolerance_x or y1 > height + tolerance_y:
        raise _Hold(IntakeReason.REGION_OUT_OF_BOUNDS)
    x0, x1 = _clamp(x0, 0.0, width), _clamp(x1, 0.0, width)
    y0, y1 = _clamp(y0, 0.0, height), _clamp(y1, 0.0, height)
    if not (x0 < x1 and y0 < y1):
        raise _Hold(IntakeReason.REGION_OUT_OF_BOUNDS)
    u0 = _clamp(x0 * upright_width / width, 0.0, upright_width)
    u1 = _clamp(x1 * upright_width / width, 0.0, upright_width)
    v0 = _clamp(y0 * upright_height / height, 0.0, upright_height)
    v1 = _clamp(y1 * upright_height / height, 0.0, upright_height)
    if rotation == 0:
        local = (u0, v0, u1, v1)
    elif rotation == 90:
        local = (v0, crop_height - u1, v1, crop_height - u0)
    elif rotation == 180:
        local = (crop_width - u1, crop_height - v1, crop_width - u0, crop_height - v0)
    else:
        local = (crop_width - v1, u0, crop_width - v0, u1)
    mapped = (
        _clamp(left + local[0], left, right) + 0.0,
        _clamp(top + local[1], top, bottom) + 0.0,
        _clamp(left + local[2], left, right) + 0.0,
        _clamp(top + local[3], top, bottom) + 0.0,
    )
    if not (mapped[0] < mapped[2] and mapped[1] < mapped[3]):
        raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM)
    return mapped


def _visual(node: EvidenceNode, rotation: int) -> MultimodalPacket:
    try:
        return project_visual(node, rotation=rotation)
    except AppleContractError:
        raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM) from None


def _map_page(
    digest: str,
    index: int,
    geometry: IntakePageGeometry,
    recognition: Recognition,
    limits: IntakeLimits,
    offset: int,
) -> tuple[IntakePage, tuple[IntakeRegion, ...]]:
    _checked_crop(geometry, recognition.width, recognition.height)
    text = recognition.text
    if len(recognition.regions) > limits.max_regions_per_page:
        raise _Hold(IntakeReason.RESOURCE_LIMIT)
    if len(text) > limits.max_page_scalars:
        raise _Hold(IntakeReason.RESOURCE_LIMIT)
    mapped = []
    for region in recognition.regions:
        start, end = region.span
        if not 0 <= start < end <= len(text):
            raise _Hold(IntakeReason.INVALID_RECOGNITION)
        box = _map_box(geometry, recognition.width, recognition.height, region.box)
        mapped.append(((offset + start, offset + end), box))
    try:
        extent = PageGeometry(
            page=index,
            width=float(geometry.width),
            height=float(geometry.height),
            unit=geometry.unit,
        )
        page_packet = _visual(
            EvidenceNode(
                node_id=intake_node_ref(digest, index, "page"),
                kind=NodeKind.PAGE,
                source_digest=digest,
                geometry=extent,
            ),
            geometry.rotation,
        )
        crop_packet = None
        if geometry.crop is not None:
            crop_packet = _visual(
                EvidenceNode(
                    node_id=intake_node_ref(digest, index, "crop"),
                    kind=NodeKind.PIXELS,
                    source_digest=digest,
                    geometry=extent,
                    region=_checked_crop(
                        geometry, recognition.width, recognition.height
                    ),
                ),
                geometry.rotation,
            )
        regions = []
        for ordinal, (span, box) in enumerate(mapped):
            ref = intake_node_ref(digest, index, f"region:{ordinal}")
            packet = _visual(
                EvidenceNode(
                    node_id=ref,
                    kind=NodeKind.OCR,
                    source_digest=digest,
                    geometry=extent,
                    region=box,
                    span=span,
                ),
                geometry.rotation,
            )
            regions.append(IntakeRegion(index, span, box, ref, packet))
    except FusedEvidenceError:
        raise _Hold(IntakeReason.AMBIGUOUS_TRANSFORM) from None
    page = IntakePage(
        page=index,
        status=PageStatus.MAPPED,
        text_span=(offset, offset + len(text)),
        evidence=page_packet,
        crop_evidence=crop_packet,
    )
    return page, tuple(regions)


def _ordered(reasons: set[IntakeReason]) -> tuple[IntakeReason, ...]:
    return tuple(sorted(reasons, key=_ORDER.__getitem__))


def refuse_document(reason: IntakeReason) -> IntakeReport:
    """Return the refused report for a document that could not be opened.

    Args:
        reason: ``unsupported_document``, ``document_locked``,
            ``permission_denied`` or ``size_limit``.
    """
    _require(reason in _DOCUMENT_REFUSALS)
    return IntakeReport(IntakeOutcome.REFUSED, (reason,), "")


def map_document(
    source_digest: str,
    pages: Sequence[IntakePageInput],
    *,
    limits: IntakeLimits | None = None,
    page_separator: str = PAGE_SEPARATOR,
) -> IntakeReport:
    """Map native page inputs to page, crop and region evidence.

    Args:
        source_digest: Lowercase SHA-256 of the source document bytes.
        pages: One input per page, in page order; the index is the position.
        limits: Bounds; defaults to :class:`IntakeLimits`.
        page_separator: Text inserted between consecutive mapped pages.

    Returns:
        A report whose every region span indexes the report text. Held pages
        contribute no text. ``partial`` means at least one page was mapped and
        at least one reason applies; ``refused`` means no page was mapped.
    """
    limits = IntakeLimits() if limits is None else limits
    _require(type(limits) is IntakeLimits and type(page_separator) is str)
    _require(type(source_digest) is str and _DIGEST.fullmatch(source_digest))
    _require(all(type(page) is IntakePageInput for page in pages))
    if not pages:
        return IntakeReport(IntakeOutcome.REFUSED, (IntakeReason.EMPTY_DOCUMENT,), "")
    reasons: set[IntakeReason] = set()
    if len(pages) > limits.max_pages:
        reasons.add(IntakeReason.PAGE_LIMIT)
    text: list[str] = []
    length = 0
    records: list[IntakePage] = []
    regions: list[IntakeRegion] = []
    for index, item in enumerate(pages[: limits.max_pages]):
        reason = item.unavailable
        if reason is IntakeReason.PAGE_LIMIT:
            reasons.add(reason)
            break
        if reason is None:
            assert item.geometry is not None and item.recognition is not None
            separator = page_separator if text else ""
            try:
                page, mapped = _map_page(
                    source_digest,
                    index,
                    item.geometry,
                    item.recognition,
                    limits,
                    length + len(separator),
                )
            except _Hold as hold:
                reason = hold.reason
            else:
                text.extend((separator, item.recognition.text))
                length += len(separator) + len(item.recognition.text)
                records.append(page)
                regions.extend(mapped)
                continue
        reasons.add(reason)
        records.append(IntakePage(page=index, status=PageStatus.HELD, reason=reason))
        if reason is IntakeReason.PERMISSION_LOST:
            break
    mapped_pages = sum(page.status is PageStatus.MAPPED for page in records)
    if mapped_pages == 0:
        outcome = IntakeOutcome.REFUSED
    elif reasons:
        outcome = IntakeOutcome.PARTIAL
    else:
        outcome = IntakeOutcome.COMPLETE
    return IntakeReport(
        outcome, _ordered(reasons), "".join(text), tuple(records), tuple(regions)
    )
