"""Versioned PDF, OCR, image, and DICOM evidence-coordinate adapters.

The visual adapter records an invertible source-to-observed transform chain and
always persists the resulting locator in the original source coordinate space.
The DICOM adapter records explicit Study/Series/Instance UIDs and element tags.
Neither adapter parses or stores clinical values; callers retain the source
bytes and may resolve only the evidence coordinates after a digest check.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import (
    EvidenceLocator,
    canonical_json,
)
from openmed.multimodal.base import ExtractedDocument
from openmed.multimodal.ocr import OcrResult
from openmed.structured.store import StoreResult, StoreState

from .evidence_adapters import (
    DEFAULT_MAX_SOURCE_BYTES,
    EvidenceAdapterContext,
    EvidenceAdapterOutcome,
    _limit_quarantine,
    _locator_matches_source,
    _quarantine,
    _success,
)

MULTIMODAL_COORDINATE_SCHEMA_VERSION = "1.0.0"
MULTIMODAL_COORDINATE_COMPATIBILITY_POLICY = "same_major"
MULTIMODAL_COORDINATE_SCHEMA_PACKAGE = "openmed.core.schemas.json"
MULTIMODAL_COORDINATE_SCHEMA_NAME = "multimodal_coordinate"

VISUAL_SOURCE_FORMATS = frozenset({"pdf", "ocr", "image"})
VISUAL_COORDINATE_SPACES = frozenset({"pixels", "points"})
VISUAL_TRANSFORM_OPERATIONS = frozenset(
    {"identity", "rotate_clockwise", "crop", "scale", "ocr_scale"}
)
VISUAL_COORDINATE_CONVENTIONS = MappingProxyType(
    {
        "image": "image_top_left_v1",
        "ocr": "ocr_top_left_v1",
        "pdf": "pdf_top_left_v1",
    }
)
DICOM_COORDINATE_CONVENTION = "dicom_element_v1"

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DICOM_PATH_RE = re.compile(r"^[1-9][0-9]*(?:\.[1-9][0-9]*)*$")
_MEDIA_TYPE_RE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")

Box = tuple[float, float, float, float]


class MultimodalCoordinateError(ValueError):
    """Value-safe validation error for multimodal coordinates."""


@dataclass(frozen=True, slots=True)
class CoordinateTransformStep:
    """One invertible, dimension-bound step in a visual transform chain."""

    operation: str
    input_width: float
    input_height: float
    output_width: float
    output_height: float
    parameters: Mapping[str, Any] = field(default_factory=dict, repr=False)
    schema_version: str = MULTIMODAL_COORDINATE_SCHEMA_VERSION
    compatibility_policy: str = MULTIMODAL_COORDINATE_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.operation not in VISUAL_TRANSFORM_OPERATIONS:
            raise MultimodalCoordinateError("visual transform operation is unsupported")
        input_width = _positive_number(self.input_width, "input_width")
        input_height = _positive_number(self.input_height, "input_height")
        output_width = _positive_number(self.output_width, "output_width")
        output_height = _positive_number(self.output_height, "output_height")
        parameters = _parameters(self.parameters)
        if self.operation == "identity":
            _exact_parameter_keys(parameters, frozenset())
            _require_dimensions(
                (output_width, output_height),
                (input_width, input_height),
                "identity transform dimensions do not match",
            )
        elif self.operation == "rotate_clockwise":
            _exact_parameter_keys(parameters, frozenset({"degrees"}))
            degrees = parameters["degrees"]
            if type(degrees) is not int or degrees not in {90, 180, 270}:
                raise MultimodalCoordinateError(
                    "clockwise rotation must be 90, 180, or 270 degrees"
                )
            expected = (
                (input_height, input_width)
                if degrees in {90, 270}
                else (input_width, input_height)
            )
            _require_dimensions(
                (output_width, output_height),
                expected,
                "rotation output dimensions do not match",
            )
        elif self.operation == "crop":
            _exact_parameter_keys(parameters, frozenset({"box"}))
            crop = _box(parameters["box"], "crop box")
            _require_box_inside(crop, input_width, input_height, tolerance=0.0)
            _require_dimensions(
                (output_width, output_height),
                (crop[2] - crop[0], crop[3] - crop[1]),
                "crop output dimensions do not match",
            )
            parameters = MappingProxyType({"box": crop})
        else:
            _exact_parameter_keys(parameters, frozenset())
        _require_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(self, "input_width", input_width)
        object.__setattr__(self, "input_height", input_height)
        object.__setattr__(self, "output_width", output_width)
        object.__setattr__(self, "output_height", output_height)
        object.__setattr__(self, "parameters", parameters)

    def forward_box(self, box: Sequence[float]) -> Box:
        """Project one box from this step's input into its output space."""

        source = _box(box, "visual box")
        _require_box_inside(
            source,
            self.input_width,
            self.input_height,
            tolerance=0.0,
        )
        projected = _project_corners(source, self._forward_point)
        _require_box_inside(
            projected,
            self.output_width,
            self.output_height,
            tolerance=1e-9,
        )
        return projected

    def inverse_box(self, box: Sequence[float]) -> Box:
        """Project one box from this step's output back into its input space."""

        observed = _box(box, "visual box")
        _require_box_inside(
            observed,
            self.output_width,
            self.output_height,
            tolerance=0.0,
        )
        projected = _project_corners(observed, self._inverse_point)
        _require_box_inside(
            projected,
            self.input_width,
            self.input_height,
            tolerance=1e-9,
        )
        return projected

    def _forward_point(self, x: float, y: float) -> tuple[float, float]:
        if self.operation == "identity":
            return x, y
        if self.operation == "rotate_clockwise":
            degrees = int(self.parameters["degrees"])
            if degrees == 90:
                return self.input_height - y, x
            if degrees == 180:
                return self.input_width - x, self.input_height - y
            return y, self.input_width - x
        if self.operation == "crop":
            x0, y0, _, _ = self.parameters["box"]
            return x - x0, y - y0
        return (
            x * self.output_width / self.input_width,
            y * self.output_height / self.input_height,
        )

    def _inverse_point(self, x: float, y: float) -> tuple[float, float]:
        if self.operation == "identity":
            return x, y
        if self.operation == "rotate_clockwise":
            degrees = int(self.parameters["degrees"])
            if degrees == 90:
                return y, self.input_height - x
            if degrees == 180:
                return self.input_width - x, self.input_height - y
            return self.input_width - y, x
        if self.operation == "crop":
            x0, y0, _, _ = self.parameters["box"]
            return x + x0, y + y0
        return (
            x * self.input_width / self.output_width,
            y * self.input_height / self.output_height,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic public transform-step record."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "input_dimensions": [self.input_width, self.input_height],
            "operation": self.operation,
            "output_dimensions": [self.output_width, self.output_height],
            "parameters": _plain(self.parameters),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class CoordinateTransformChain:
    """Ordered invertible transform chain from source to observed geometry."""

    source_width: float
    source_height: float
    steps: tuple[CoordinateTransformStep, ...] = ()
    schema_version: str = MULTIMODAL_COORDINATE_SCHEMA_VERSION
    compatibility_policy: str = MULTIMODAL_COORDINATE_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        source_width = _positive_number(self.source_width, "source_width")
        source_height = _positive_number(self.source_height, "source_height")
        steps = tuple(self.steps)
        if any(not isinstance(item, CoordinateTransformStep) for item in steps):
            raise TypeError(
                "transform chain steps must be CoordinateTransformStep values"
            )
        expected = (source_width, source_height)
        for step in steps:
            _require_dimensions(
                (step.input_width, step.input_height),
                expected,
                "transform chain dimensions are discontinuous",
            )
            expected = (step.output_width, step.output_height)
        _require_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(self, "source_width", source_width)
        object.__setattr__(self, "source_height", source_height)
        object.__setattr__(self, "steps", steps)

    @property
    def observed_width(self) -> float:
        """Return the width of the final observed coordinate space."""

        return self.steps[-1].output_width if self.steps else self.source_width

    @property
    def observed_height(self) -> float:
        """Return the height of the final observed coordinate space."""

        return self.steps[-1].output_height if self.steps else self.source_height

    def forward_box(self, box: Sequence[float]) -> Box:
        """Project a source box through every transform step."""

        projected = _box(box, "visual box")
        _require_box_inside(
            projected,
            self.source_width,
            self.source_height,
            tolerance=0.0,
        )
        for step in self.steps:
            projected = step.forward_box(projected)
        return projected

    def inverse_box(self, box: Sequence[float]) -> Box:
        """Project an observed box back to original source coordinates."""

        projected = _box(box, "visual box")
        _require_box_inside(
            projected,
            self.observed_width,
            self.observed_height,
            tolerance=0.0,
        )
        for step in reversed(self.steps):
            projected = step.inverse_box(projected)
        return projected

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic public transform-chain record."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "observed_dimensions": [self.observed_width, self.observed_height],
            "schema_version": self.schema_version,
            "source_dimensions": [self.source_width, self.source_height],
            "steps": [item.to_dict() for item in self.steps],
        }


@dataclass(frozen=True, slots=True)
class VisualEvidenceFrame:
    """One source page or frame and its source-to-observed transform chain."""

    frame_id: str
    page: int
    coordinate_space: str
    transform: CoordinateTransformChain
    tolerance: float = 0.5

    def __post_init__(self) -> None:
        _controlled(self.frame_id, "frame_id")
        if type(self.page) is not int or self.page < 1:
            raise MultimodalCoordinateError("visual frame page must be positive")
        if self.coordinate_space not in VISUAL_COORDINATE_SPACES:
            raise MultimodalCoordinateError("visual coordinate space is unsupported")
        if not isinstance(self.transform, CoordinateTransformChain):
            raise TypeError("visual frame transform must be a CoordinateTransformChain")
        tolerance = _non_negative_number(self.tolerance, "tolerance")
        if tolerance > 10:
            raise MultimodalCoordinateError("visual coordinate tolerance is too large")
        object.__setattr__(self, "tolerance", tolerance)

    def coordinate_record(self, *, convention: str) -> dict[str, Any]:
        """Return the versioned source/frame/transform provenance record."""

        if convention not in VISUAL_COORDINATE_CONVENTIONS.values():
            raise MultimodalCoordinateError(
                "visual coordinate convention is unsupported"
            )
        return {
            "compatibility_policy": MULTIMODAL_COORDINATE_COMPATIBILITY_POLICY,
            "coordinate_convention": convention,
            "frame_id": self.frame_id,
            "page": self.page,
            "schema_version": MULTIMODAL_COORDINATE_SCHEMA_VERSION,
            "source_coordinate_space": self.coordinate_space,
            "tolerance": self.tolerance,
            "transform_chain": self.transform.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class VisualEvidenceRegion:
    """One observed visual region linked to an optional normalized text span."""

    frame_id: str
    box: Box
    region_kind: str
    normalized_start: int | None = None
    normalized_end: int | None = None

    def __post_init__(self) -> None:
        _controlled(self.frame_id, "frame_id")
        _controlled(self.region_kind, "region_kind")
        object.__setattr__(self, "box", _box(self.box, "visual region box"))
        if (self.normalized_start is None) != (self.normalized_end is None):
            raise MultimodalCoordinateError(
                "normalized text span bounds must be supplied together"
            )
        if self.normalized_start is not None:
            if (
                type(self.normalized_start) is not int
                or type(self.normalized_end) is not int
                or self.normalized_start < 0
                or self.normalized_end <= self.normalized_start
            ):
                raise MultimodalCoordinateError("normalized text span is invalid")


@dataclass(frozen=True, slots=True)
class VisualEvidenceInput:
    """Caller-held bytes plus visual frames and observed evidence regions."""

    source_bytes: bytes = field(repr=False)
    source_format: str
    source_version: str
    media_type: str
    frames: tuple[VisualEvidenceFrame, ...]
    regions: tuple[VisualEvidenceRegion, ...]
    unmapped_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.source_bytes, bytes):
            raise TypeError("visual source_bytes must be bytes")
        if self.source_format not in VISUAL_SOURCE_FORMATS:
            raise MultimodalCoordinateError("visual source format is unsupported")
        _version(self.source_version, "source_version")
        if (
            not isinstance(self.media_type, str)
            or _MEDIA_TYPE_RE.fullmatch(self.media_type) is None
        ):
            raise MultimodalCoordinateError("visual media type is invalid")
        frames = tuple(self.frames)
        regions = tuple(self.regions)
        if any(not isinstance(item, VisualEvidenceFrame) for item in frames):
            raise TypeError("visual frames must be VisualEvidenceFrame values")
        if any(not isinstance(item, VisualEvidenceRegion) for item in regions):
            raise TypeError("visual regions must be VisualEvidenceRegion values")
        frame_ids = tuple(item.frame_id for item in frames)
        pages = tuple(item.page for item in frames)
        if len(frame_ids) != len(set(frame_ids)) or len(pages) != len(set(pages)):
            raise MultimodalCoordinateError("visual frames must be uniquely identified")
        if any(item.frame_id not in set(frame_ids) for item in regions):
            raise MultimodalCoordinateError("visual region references an unknown frame")
        if type(self.unmapped_count) is not int or self.unmapped_count < 0:
            raise MultimodalCoordinateError(
                "visual unmapped_count must be non-negative"
            )
        object.__setattr__(
            self, "frames", tuple(sorted(frames, key=lambda item: item.page))
        )
        object.__setattr__(
            self,
            "regions",
            tuple(
                sorted(
                    regions,
                    key=lambda item: (
                        item.frame_id,
                        item.box,
                        item.normalized_start
                        if item.normalized_start is not None
                        else -1,
                        item.region_kind,
                    ),
                )
            ),
        )

    @classmethod
    def from_document(
        cls,
        document: ExtractedDocument,
        *,
        source_bytes: bytes,
        source_format: str,
        source_version: str,
        media_type: str,
        frames: Sequence[VisualEvidenceFrame],
    ) -> "VisualEvidenceInput":
        """Bridge coordinate-bearing PDF/OCR documents into visual evidence."""

        if not isinstance(document, ExtractedDocument):
            raise TypeError("document must be an ExtractedDocument")
        frame_by_page = {item.page: item for item in frames}
        regions: list[VisualEvidenceRegion] = []
        unmapped = 0
        for span in document.spans:
            frame = frame_by_page.get(span.page + 1)
            if span.bbox is None or frame is None:
                unmapped += 1
                continue
            regions.append(
                VisualEvidenceRegion(
                    frame_id=frame.frame_id,
                    box=span.bbox,
                    region_kind=("ocr_word" if source_format == "ocr" else "pdf_text"),
                    normalized_start=span.start,
                    normalized_end=span.end,
                )
            )
        return cls(
            source_bytes=source_bytes,
            source_format=source_format,
            source_version=source_version,
            media_type=media_type,
            frames=tuple(frames),
            regions=tuple(regions),
            unmapped_count=unmapped,
        )

    @classmethod
    def from_ocr(
        cls,
        result: OcrResult,
        *,
        source_bytes: bytes,
        source_version: str,
        media_type: str,
        frames: Sequence[VisualEvidenceFrame],
    ) -> "VisualEvidenceInput":
        """Bridge the common OCR result while preserving every word box."""

        if not isinstance(result, OcrResult):
            raise TypeError("result must be an OcrResult")
        return cls.from_document(
            result.to_document(),
            source_bytes=source_bytes,
            source_format="ocr",
            source_version=source_version,
            media_type=media_type,
            frames=frames,
        )


class VisualEvidenceAdapter:
    """Map PDF, OCR, and image regions back to original source coordinates."""

    parser_id = "openmed.evidence.visual"
    parser_version = "1.0.0"

    def __init__(self, *, max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES) -> None:
        if type(max_source_bytes) is not int or max_source_bytes < 1:
            raise ValueError("max_source_bytes must be positive")
        self.max_source_bytes = max_source_bytes

    def adapt(
        self,
        source: Any,
        context: EvidenceAdapterContext,
    ) -> StoreResult[EvidenceAdapterOutcome]:
        """Return exact source-space boxes or a value-free typed quarantine."""

        if not isinstance(source, VisualEvidenceInput):
            payload = source if isinstance(source, bytes) else b""
            return _quarantine(
                payload,
                context,
                source_format="image",
                source_version="unknown",
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                state=StoreState.UNSUPPORTED,
                classification="unsupported",
                reason_code="visual_source_unsupported",
            )
        limited = _limit_quarantine(
            source.source_bytes,
            context,
            source_format=source.source_format,
            source_version=source.source_version,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            max_source_bytes=self.max_source_bytes,
        )
        if limited is not None:
            return limited
        if source.unmapped_count:
            return _quarantine(
                source.source_bytes,
                context,
                source_format=source.source_format,
                source_version=source.source_version,
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                state=StoreState.PARTIAL,
                classification="partial",
                reason_code="visual_coordinates_incomplete",
                candidate_count=len(source.regions),
            )
        if not source.regions:
            return _quarantine(
                source.source_bytes,
                context,
                source_format=source.source_format,
                source_version=source.source_version,
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                state=StoreState.UNKNOWN,
                classification="partial",
                reason_code="visual_evidence_missing",
            )

        convention = VISUAL_COORDINATE_CONVENTIONS[source.source_format]
        frames = {item.frame_id: item for item in source.frames}
        specs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
        try:
            for region in source.regions:
                frame = frames[region.frame_id]
                _require_box_inside(
                    region.box,
                    frame.transform.observed_width,
                    frame.transform.observed_height,
                    tolerance=frame.tolerance,
                )
                source_box = frame.transform.inverse_box(region.box)
                replayed = frame.transform.forward_box(source_box)
                if _box_error(replayed, region.box) > frame.tolerance:
                    raise MultimodalCoordinateError(
                        "visual coordinate round trip exceeds tolerance"
                    )
                coordinate_record = frame.coordinate_record(convention=convention)
                transform: dict[str, Any] = {
                    "coordinate_record": coordinate_record,
                    "observed_box": list(region.box),
                    "region_kind": region.region_kind,
                }
                if region.normalized_start is not None:
                    transform["normalized_start"] = region.normalized_start
                    transform["normalized_end"] = region.normalized_end
                specs.append(
                    (
                        "page_box",
                        {
                            "box": list(source_box),
                            "coordinate_space": frame.coordinate_space,
                            "page": frame.page,
                            "page_height": frame.transform.source_height,
                            "page_width": frame.transform.source_width,
                        },
                        transform,
                    )
                )
        except (KeyError, MultimodalCoordinateError, ValueError):
            return _quarantine(
                source.source_bytes,
                context,
                source_format=source.source_format,
                source_version=source.source_version,
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                state=StoreState.CONFLICT,
                classification="ambiguous",
                reason_code="coordinate_projection_invalid",
                candidate_count=len(specs),
            )

        return _success(
            source_bytes=source.source_bytes,
            artifact_bytes=source.source_bytes,
            context=context,
            source_format=source.source_format,
            source_version=source.source_version,
            media_type=source.media_type,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            coordinate_convention=convention,
            normalization_transform={
                "frame_count": len(source.frames),
                "operation": "visual_source_projection",
                "region_count": len(source.regions),
                "version": MULTIMODAL_COORDINATE_SCHEMA_VERSION,
            },
            locator_specs=tuple(specs),
        )

    def resolve(
        self,
        source: Any,
        locator: EvidenceLocator,
    ) -> StoreResult[Any]:
        """Resolve one locator to verified source-space coordinate metadata."""

        if (
            not isinstance(source, VisualEvidenceInput)
            or locator.location_type != "page_box"
        ):
            return StoreResult.outcome(StoreState.UNSUPPORTED, "locator_unsupported")
        if not _locator_matches_source(locator, source.source_bytes):
            return StoreResult.outcome(StoreState.CONFLICT, "source_digest_mismatch")
        page = int(locator.location["page"])
        if page not in {item.page for item in source.frames}:
            return StoreResult.outcome(StoreState.CONFLICT, "locator_unresolvable")
        return StoreResult.success(
            {
                "box": list(locator.location["box"]),
                "coordinate_space": locator.location["coordinate_space"],
                "page": page,
                "page_height": locator.location.get("page_height"),
                "page_width": locator.location.get("page_width"),
            }
        )


@dataclass(frozen=True, slots=True)
class DICOMElementReference:
    """One explicit DICOM tag reference with optional SR node provenance."""

    tag: str
    node_path: str | None = None
    normalized_start: int | None = None
    normalized_end: int | None = None

    def __post_init__(self) -> None:
        # EvidenceLocator performs the normative DICOM-tag validation. This
        # constructor validates only the additional SR path/span relationship.
        if not isinstance(self.tag, str) or not self.tag:
            raise MultimodalCoordinateError("DICOM element tag is invalid")
        if (
            self.node_path is not None
            and _DICOM_PATH_RE.fullmatch(self.node_path) is None
        ):
            raise MultimodalCoordinateError("DICOM SR node path is invalid")
        if (self.normalized_start is None) != (self.normalized_end is None):
            raise MultimodalCoordinateError(
                "DICOM normalized span bounds must be supplied together"
            )
        if self.normalized_start is not None and (
            type(self.normalized_start) is not int
            or type(self.normalized_end) is not int
            or self.normalized_start < 0
            or self.normalized_end <= self.normalized_start
        ):
            raise MultimodalCoordinateError("DICOM normalized span is invalid")


@dataclass(frozen=True, slots=True)
class DICOMEvidenceInput:
    """Caller-held DICOM bytes and explicit element references."""

    source_bytes: bytes = field(repr=False)
    study_uid: str = field(repr=False)
    series_uid: str = field(repr=False)
    instance_uid: str = field(repr=False)
    elements: tuple[DICOMElementReference, ...] = field(repr=False)
    source_version: str = "dicom_sr"

    def __post_init__(self) -> None:
        if not isinstance(self.source_bytes, bytes):
            raise TypeError("DICOM source_bytes must be bytes")
        for name in ("study_uid", "series_uid", "instance_uid"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise MultimodalCoordinateError("DICOM UID is invalid")
        elements = tuple(self.elements)
        if any(not isinstance(item, DICOMElementReference) for item in elements):
            raise TypeError("DICOM elements must be DICOMElementReference values")
        _version(self.source_version, "source_version")
        object.__setattr__(
            self,
            "elements",
            tuple(
                sorted(
                    elements,
                    key=lambda item: (
                        item.tag,
                        item.node_path or "",
                        item.normalized_start
                        if item.normalized_start is not None
                        else -1,
                    ),
                )
            ),
        )

    @classmethod
    def from_sr_document(
        cls,
        document: ExtractedDocument,
        *,
        source_bytes: bytes,
        study_uid: str,
        series_uid: str,
        instance_uid: str,
    ) -> "DICOMEvidenceInput":
        """Bridge existing DICOM-SR spans to the Content Sequence tag."""

        if not isinstance(document, ExtractedDocument):
            raise TypeError("document must be an ExtractedDocument")
        elements: list[DICOMElementReference] = []
        for span in document.spans:
            node_path = span.metadata.get("node_path")
            if not isinstance(node_path, str):
                raise MultimodalCoordinateError(
                    "DICOM SR span is missing explicit node provenance"
                )
            elements.append(
                DICOMElementReference(
                    tag="0040,A730",
                    node_path=node_path,
                    normalized_start=span.start,
                    normalized_end=span.end,
                )
            )
        return cls(
            source_bytes=source_bytes,
            study_uid=study_uid,
            series_uid=series_uid,
            instance_uid=instance_uid,
            elements=tuple(elements),
        )


class DICOMEvidenceAdapter:
    """Preserve explicit DICOM tag and DICOM-SR node provenance."""

    parser_id = "openmed.evidence.dicom"
    parser_version = "1.0.0"

    def __init__(self, *, max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES) -> None:
        if type(max_source_bytes) is not int or max_source_bytes < 1:
            raise ValueError("max_source_bytes must be positive")
        self.max_source_bytes = max_source_bytes

    def adapt(
        self,
        source: Any,
        context: EvidenceAdapterContext,
    ) -> StoreResult[EvidenceAdapterOutcome]:
        """Return DICOM element locators or a value-free typed quarantine."""

        if not isinstance(source, DICOMEvidenceInput):
            payload = source if isinstance(source, bytes) else b""
            return _quarantine(
                payload,
                context,
                source_format="dicom_sr",
                source_version="unknown",
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                state=StoreState.UNSUPPORTED,
                classification="unsupported",
                reason_code="dicom_source_unsupported",
            )
        limited = _limit_quarantine(
            source.source_bytes,
            context,
            source_format="dicom_sr",
            source_version=source.source_version,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            max_source_bytes=self.max_source_bytes,
        )
        if limited is not None:
            return limited
        if not source.elements:
            return _quarantine(
                source.source_bytes,
                context,
                source_format="dicom_sr",
                source_version=source.source_version,
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                state=StoreState.PARTIAL,
                classification="partial",
                reason_code="dicom_evidence_missing",
            )

        specs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
        try:
            for element in source.elements:
                record = _dicom_coordinate_record(element)
                specs.append(
                    (
                        "dicom_element",
                        {
                            "instance_uid": source.instance_uid,
                            "series_uid": source.series_uid,
                            "study_uid": source.study_uid,
                            "tag": element.tag,
                        },
                        {"coordinate_record": record},
                    )
                )
            # Constructing locators below performs the normative UID/tag checks.
            return _success(
                source_bytes=source.source_bytes,
                artifact_bytes=source.source_bytes,
                context=context,
                source_format="dicom_sr",
                source_version=source.source_version,
                media_type="application/dicom",
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                coordinate_convention=DICOM_COORDINATE_CONVENTION,
                normalization_transform={
                    "element_count": len(source.elements),
                    "operation": "dicom_element_projection",
                    "version": MULTIMODAL_COORDINATE_SCHEMA_VERSION,
                },
                locator_specs=tuple(specs),
            )
        except ValueError:
            return _quarantine(
                source.source_bytes,
                context,
                source_format="dicom_sr",
                source_version=source.source_version,
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                state=StoreState.FAILURE,
                classification="malformed",
                reason_code="dicom_coordinates_invalid",
                candidate_count=len(specs),
            )

    def resolve(
        self,
        source: Any,
        locator: EvidenceLocator,
    ) -> StoreResult[Any]:
        """Resolve one verified DICOM element coordinate without its value."""

        if (
            not isinstance(source, DICOMEvidenceInput)
            or locator.location_type != "dicom_element"
        ):
            return StoreResult.outcome(StoreState.UNSUPPORTED, "locator_unsupported")
        if not _locator_matches_source(locator, source.source_bytes):
            return StoreResult.outcome(StoreState.CONFLICT, "source_digest_mismatch")
        expected = {
            "instance_uid": source.instance_uid,
            "series_uid": source.series_uid,
            "study_uid": source.study_uid,
        }
        if any(locator.location.get(key) != value for key, value in expected.items()):
            return StoreResult.outcome(StoreState.CONFLICT, "locator_unresolvable")
        return StoreResult.success(
            {
                "instance_uid": locator.location["instance_uid"],
                "series_uid": locator.location["series_uid"],
                "study_uid": locator.location["study_uid"],
                "tag": locator.location["tag"],
            }
        )


def load_multimodal_coordinate_schema() -> dict[str, Any]:
    """Load the bundled multimodal coordinate-record JSON Schema."""

    resource = resources.files(MULTIMODAL_COORDINATE_SCHEMA_PACKAGE).joinpath(
        f"{MULTIMODAL_COORDINATE_SCHEMA_NAME}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dicom_coordinate_record(element: DICOMElementReference) -> dict[str, Any]:
    return {
        "compatibility_policy": MULTIMODAL_COORDINATE_COMPATIBILITY_POLICY,
        "coordinate_convention": DICOM_COORDINATE_CONVENTION,
        "node_path": element.node_path,
        "normalized_span": (
            [element.normalized_start, element.normalized_end]
            if element.normalized_start is not None
            else None
        ),
        "schema_version": MULTIMODAL_COORDINATE_SCHEMA_VERSION,
    }


def _parameters(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("visual transform parameters must be a mapping")
    return MappingProxyType(dict(value))


def _exact_parameter_keys(
    parameters: Mapping[str, Any], expected: frozenset[str]
) -> None:
    if set(parameters) != set(expected):
        raise MultimodalCoordinateError("visual transform parameters are invalid")


def _positive_number(value: Any, name: str) -> float:
    number = _non_negative_number(value, name)
    if number <= 0:
        raise MultimodalCoordinateError(f"{name} must be positive")
    return number


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultimodalCoordinateError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise MultimodalCoordinateError(f"{name} must be a finite number")
    return number


def _box(value: Sequence[float], name: str) -> Box:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MultimodalCoordinateError(f"{name} must contain four numbers")
    if len(value) != 4:
        raise MultimodalCoordinateError(f"{name} must contain four numbers")
    coordinates = tuple(_non_negative_number(item, name) for item in value)
    x0, y0, x1, y1 = coordinates
    if x1 <= x0 or y1 <= y0:
        raise MultimodalCoordinateError(f"{name} must have positive area")
    return coordinates  # type: ignore[return-value]


def _require_box_inside(
    box: Box,
    width: float,
    height: float,
    *,
    tolerance: float,
) -> None:
    if (
        box[0] < -tolerance
        or box[1] < -tolerance
        or box[2] > width + tolerance
        or box[3] > height + tolerance
    ):
        raise MultimodalCoordinateError("visual box exceeds declared dimensions")


def _project_corners(
    box: Box,
    operation: Any,
) -> Box:
    points = (
        operation(box[0], box[1]),
        operation(box[2], box[1]),
        operation(box[0], box[3]),
        operation(box[2], box[3]),
    )
    xs = tuple(item[0] for item in points)
    ys = tuple(item[1] for item in points)
    return min(xs), min(ys), max(xs), max(ys)


def _box_error(first: Box, second: Box) -> float:
    return max(abs(left - right) for left, right in zip(first, second))


def _require_dimensions(
    actual: tuple[float, float],
    expected: tuple[float, float],
    message: str,
) -> None:
    if any(abs(left - right) > 1e-9 for left, right in zip(actual, expected)):
        raise MultimodalCoordinateError(message)


def _controlled(value: Any, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise MultimodalCoordinateError(f"{name} must be controlled")
    return value


def _version(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value) is None
    ):
        raise MultimodalCoordinateError(f"{name} is invalid")
    return value


def _require_version(version: str, compatibility_policy: str) -> None:
    if version != MULTIMODAL_COORDINATE_SCHEMA_VERSION:
        raise MultimodalCoordinateError("multimodal coordinate schema is unsupported")
    if compatibility_policy != MULTIMODAL_COORDINATE_COMPATIBILITY_POLICY:
        raise MultimodalCoordinateError(
            "multimodal coordinate compatibility policy is unsupported"
        )


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


__all__ = [
    "DICOM_COORDINATE_CONVENTION",
    "MULTIMODAL_COORDINATE_COMPATIBILITY_POLICY",
    "MULTIMODAL_COORDINATE_SCHEMA_NAME",
    "MULTIMODAL_COORDINATE_SCHEMA_PACKAGE",
    "MULTIMODAL_COORDINATE_SCHEMA_VERSION",
    "VISUAL_COORDINATE_CONVENTIONS",
    "VISUAL_COORDINATE_SPACES",
    "VISUAL_SOURCE_FORMATS",
    "VISUAL_TRANSFORM_OPERATIONS",
    "CoordinateTransformChain",
    "CoordinateTransformStep",
    "DICOMEvidenceAdapter",
    "DICOMEvidenceInput",
    "DICOMElementReference",
    "MultimodalCoordinateError",
    "VisualEvidenceAdapter",
    "VisualEvidenceFrame",
    "VisualEvidenceInput",
    "VisualEvidenceRegion",
    "load_multimodal_coordinate_schema",
]
