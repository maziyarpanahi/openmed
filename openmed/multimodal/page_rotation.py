"""Deterministic, lossless transforms for OCR page-space geometry.

OCR coordinates are represented in a top-left-origin page coordinate system
whose outer edges are ``[0, width]`` and ``[0, height]``.  Rotations are
clockwise and limited to quarter turns.  The module validates geometry before
transforming it, never clips coordinates, and keeps the source order and
metadata of OCR/document records unchanged.

The implementation is local-only and uses only the Python standard library.
It deliberately keeps error messages to stable field names and reason codes;
coordinates, OCR text, and source references are never interpolated into
exceptions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from math import isfinite
from typing import Any, TypeAlias

from .base import ExtractedDocument, SourceSpan
from .ocr import OcrResult, OcrWord

Point: TypeAlias = tuple[float, float]
BBox: TypeAlias = tuple[float, float, float, float]


class PageRotationError(ValueError):
    """Base error for invalid page-rotation input."""

    def __init__(self, reason: str, *, field_name: str | None = None) -> None:
        location = f" ({field_name})" if field_name else ""
        super().__init__(f"invalid page rotation{location}: {reason}")
        self.reason = reason
        self.field_name = field_name


class AmbiguousOrientationError(PageRotationError):
    """Raised when a rotation is missing, conflicting, or unsupported."""


class InvalidPageDimensionsError(PageRotationError):
    """Raised when page dimensions are missing or not strictly positive."""


class GeometryValidationError(PageRotationError):
    """Raised when a point or box is malformed or outside the page."""


class OutOfBoundsError(GeometryValidationError):
    """Raised when geometry extends outside the supplied page bounds."""


class PageRotation(IntEnum):
    """Supported clockwise page rotations in degrees."""

    ROTATE_0 = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270

    # Descriptive aliases keep the enum convenient without adding new
    # interpretations of the orientation value.
    DEGREES_0 = 0
    DEGREES_90 = 90
    DEGREES_180 = 180
    DEGREES_270 = 270
    CLOCKWISE_0 = 0
    CLOCKWISE_90 = 90
    CLOCKWISE_180 = 180
    CLOCKWISE_270 = 270


Rotation = PageRotation
PageRotationLike: TypeAlias = PageRotation | int | float | str


def _finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise GeometryValidationError("numeric value required", field_name=field_name)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise GeometryValidationError(
            "numeric value required", field_name=field_name
        ) from None
    if not isfinite(converted):
        raise GeometryValidationError(
            "finite numeric value required", field_name=field_name
        )
    return converted


def _dimension(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise InvalidPageDimensionsError(
            "positive finite dimension required", field_name=field_name
        )
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise InvalidPageDimensionsError(
            "positive finite dimension required", field_name=field_name
        ) from None
    if not isfinite(converted) or converted <= 0:
        raise InvalidPageDimensionsError(
            "positive finite dimension required", field_name=field_name
        )
    return converted


@dataclass(frozen=True)
class PageSize:
    """Validated width and height for one source page."""

    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _dimension(self.width, field_name="width"))
        object.__setattr__(self, "height", _dimension(self.height, field_name="height"))

    def as_tuple(self) -> tuple[float, float]:
        """Return the validated ``(width, height)`` pair."""

        return self.width, self.height


PageDimensions = PageSize
PageSizeLike: TypeAlias = PageSize | Sequence[Any] | Mapping[str, Any]
PointLike: TypeAlias = Sequence[Any] | Mapping[str, Any]
BBoxLike: TypeAlias = Sequence[Any] | Mapping[str, Any]


def _coerce_page_size(value: PageSizeLike | None) -> PageSize:
    if isinstance(value, PageSize):
        return value
    if value is None:
        raise InvalidPageDimensionsError(
            "page size is required", field_name="page_size"
        )
    if isinstance(value, Mapping):
        if "width" not in value or "height" not in value:
            raise InvalidPageDimensionsError(
                "width and height are required", field_name="page_size"
            )
        return PageSize(value["width"], value["height"])
    if isinstance(value, (str, bytes, bytearray)):
        raise InvalidPageDimensionsError(
            "width and height are required", field_name="page_size"
        )
    try:
        dimensions = tuple(value)
    except TypeError:
        raise InvalidPageDimensionsError(
            "width and height are required", field_name="page_size"
        ) from None
    if len(dimensions) != 2:
        raise InvalidPageDimensionsError(
            "width and height are required", field_name="page_size"
        )
    return PageSize(*dimensions)


def _coerce_rotation(value: PageRotationLike | None) -> PageRotation:
    if value is None:
        raise AmbiguousOrientationError(
            "an explicit quarter-turn rotation is required", field_name="rotation"
        )
    if isinstance(value, bool):
        raise AmbiguousOrientationError(
            "rotation must be 0, 90, 180, or 270 degrees", field_name="rotation"
        )
    if isinstance(value, str):
        normalized = value.strip().lower().replace("°", "")
        if normalized not in {"0", "90", "180", "270"}:
            raise AmbiguousOrientationError(
                "rotation must be 0, 90, 180, or 270 degrees",
                field_name="rotation",
            )
        value = int(normalized)
    elif isinstance(value, float):
        if not isfinite(value) or not value.is_integer():
            raise AmbiguousOrientationError(
                "rotation must be 0, 90, 180, or 270 degrees",
                field_name="rotation",
            )
        value = int(value)
    elif not isinstance(value, (PageRotation, int)):
        raise AmbiguousOrientationError(
            "rotation must be 0, 90, 180, or 270 degrees", field_name="rotation"
        )
    try:
        return PageRotation(int(value))
    except ValueError:
        raise AmbiguousOrientationError(
            "rotation must be 0, 90, 180, or 270 degrees", field_name="rotation"
        ) from None


def _resolve_rotation(
    rotation: PageRotationLike | None,
    orientation: PageRotationLike | None,
) -> PageRotation:
    if rotation is not None and orientation is not None:
        raise AmbiguousOrientationError(
            "rotation and orientation cannot both be supplied", field_name="rotation"
        )
    return _coerce_rotation(rotation if rotation is not None else orientation)


def _resolve_page_size(
    page_size: PageSizeLike | None,
    width: Any,
    height: Any,
) -> PageSize:
    if page_size is not None and (width is not None or height is not None):
        raise InvalidPageDimensionsError(
            "multiple page size representations supplied", field_name="page_size"
        )
    if page_size is not None:
        return _coerce_page_size(page_size)
    if width is None or height is None:
        return _coerce_page_size(None)
    return PageSize(width, height)


def _canonical(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _coordinates(value: Any, *, field_name: str, count: int) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        if field_name == "point":
            keys = ("x", "y")
            if "point" in value:
                if "x" in value or "y" in value:
                    raise GeometryValidationError(
                        "multiple point representations supplied", field_name=field_name
                    )
                value = value["point"]
            elif not all(key in value for key in keys):
                raise GeometryValidationError(
                    "x and y are required", field_name=field_name
                )
            else:
                value = tuple(value[key] for key in keys)
        else:
            representations = (
                ("bbox", ("x0", "y0", "x1", "y1")),
                ("edges", ("left", "top", "right", "bottom")),
            )
            present = [
                keys for _, keys in representations if all(key in value for key in keys)
            ]
            if "bbox" in value:
                if present:
                    raise GeometryValidationError(
                        "multiple box representations supplied", field_name=field_name
                    )
                value = value["bbox"]
            elif len(present) == 1:
                value = tuple(value[key] for key in present[0])
            elif not present:
                raise GeometryValidationError(
                    "four box coordinates are required", field_name=field_name
                )
            else:
                raise GeometryValidationError(
                    "multiple box representations supplied", field_name=field_name
                )
    if isinstance(value, (str, bytes, bytearray)):
        raise GeometryValidationError(
            f"{count} coordinates are required", field_name=field_name
        )
    try:
        raw = tuple(value)
    except TypeError:
        raise GeometryValidationError(
            f"{count} coordinates are required", field_name=field_name
        ) from None
    if len(raw) != count:
        raise GeometryValidationError(
            f"{count} coordinates are required", field_name=field_name
        )
    return tuple(_finite_number(item, field_name=field_name) for item in raw)


def _validate_point(point: PointLike, page_size: PageSize) -> Point:
    x, y = _coordinates(point, field_name="point", count=2)
    if x < 0 or x > page_size.width or y < 0 or y > page_size.height:
        raise OutOfBoundsError("point is outside page bounds", field_name="point")
    return _canonical(x), _canonical(y)


def _validate_bbox(box: BBoxLike, page_size: PageSize) -> BBox:
    x0, y0, x1, y1 = _coordinates(box, field_name="bbox", count=4)
    if x0 >= x1 or y0 >= y1:
        raise GeometryValidationError("box must have positive area", field_name="bbox")
    if x0 < 0 or y0 < 0 or x1 > page_size.width or y1 > page_size.height:
        raise OutOfBoundsError("box is outside page bounds", field_name="bbox")
    return (
        _canonical(x0),
        _canonical(y0),
        _canonical(x1),
        _canonical(y1),
    )


def _rotated_point(point: Point, page_size: PageSize, rotation: PageRotation) -> Point:
    x, y = point
    if rotation is PageRotation.ROTATE_0:
        return x, y
    if rotation is PageRotation.ROTATE_90:
        return _canonical(page_size.height - y), x
    if rotation is PageRotation.ROTATE_180:
        return _canonical(page_size.width - x), _canonical(page_size.height - y)
    return y, _canonical(page_size.width - x)


def _rotated_bbox(box: BBox, page_size: PageSize, rotation: PageRotation) -> BBox:
    x0, y0, x1, y1 = box
    corners = (
        _rotated_point((x0, y0), page_size, rotation),
        _rotated_point((x0, y1), page_size, rotation),
        _rotated_point((x1, y0), page_size, rotation),
        _rotated_point((x1, y1), page_size, rotation),
    )
    return (
        _canonical(min(point[0] for point in corners)),
        _canonical(min(point[1] for point in corners)),
        _canonical(max(point[0] for point in corners)),
        _canonical(max(point[1] for point in corners)),
    )


@dataclass(frozen=True)
class PageTransform:
    """A validated, reversible clockwise transform for one page."""

    page_size: PageSizeLike
    rotation: PageRotationLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "page_size", _coerce_page_size(self.page_size))
        object.__setattr__(self, "rotation", _coerce_rotation(self.rotation))

    @property
    def source_size(self) -> PageSize:
        """Return the dimensions before the transform."""

        return self.page_size

    @property
    def target_size(self) -> PageSize:
        """Return the dimensions after the transform."""

        if self.rotation in (PageRotation.ROTATE_90, PageRotation.ROTATE_270):
            return PageSize(self.page_size.height, self.page_size.width)
        return self.page_size

    @property
    def output_size(self) -> PageSize:
        """Alias for :attr:`target_size`."""

        return self.target_size

    @property
    def inverse_rotation(self) -> PageRotation:
        """Return the quarter turn that reverses this transform."""

        return PageRotation((360 - int(self.rotation)) % 360)

    def inverse(self) -> "PageTransform":
        """Return a transform that maps target geometry back to the source."""

        return PageTransform(self.target_size, self.inverse_rotation)

    def point(self, point: PointLike) -> Point:
        """Transform one validated page-space point."""

        return _rotated_point(
            _validate_point(point, self.page_size),
            self.page_size,
            self.rotation,
        )

    def bbox(self, box: BBoxLike) -> BBox:
        """Transform one validated axis-aligned OCR bounding box."""

        return _rotated_bbox(
            _validate_bbox(box, self.page_size),
            self.page_size,
            self.rotation,
        )

    def box(self, box: BBoxLike) -> BBox:
        """Alias for :meth:`bbox`."""

        return self.bbox(box)


def _make_transform(
    page_size: PageSizeLike | None,
    rotation: PageRotationLike | None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> PageTransform:
    return PageTransform(
        _resolve_page_size(page_size, width, height),
        _resolve_rotation(rotation, orientation),
    )


def transform_point(
    point: PointLike,
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> Point:
    """Transform a point through an explicit clockwise quarter turn.

    ``page_size`` is ``(width, height)`` in the point's source coordinate
    system.  ``width`` and ``height`` are equivalent keyword shorthands.
    Coordinates on the page boundary are valid; values outside it are not
    clipped.
    """

    return _make_transform(
        page_size,
        rotation,
        width=width,
        height=height,
        orientation=orientation,
    ).point(point)


def transform_bbox(
    box: BBoxLike,
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> BBox:
    """Transform an axis-aligned OCR box without clipping or normalization."""

    return _make_transform(
        page_size,
        rotation,
        width=width,
        height=height,
        orientation=orientation,
    ).bbox(box)


def transform_box(
    box: BBoxLike,
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> BBox:
    """Alias for :func:`transform_bbox`."""

    return transform_bbox(
        box,
        page_size,
        rotation,
        width=width,
        height=height,
        orientation=orientation,
    )


def rotate_point(
    point: PointLike,
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> Point:
    """Alias for :func:`transform_point`."""

    return transform_point(
        point,
        page_size,
        rotation,
        width=width,
        height=height,
        orientation=orientation,
    )


def rotate_bbox(
    box: BBoxLike,
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> BBox:
    """Alias for :func:`transform_bbox`."""

    return transform_bbox(
        box,
        page_size,
        rotation,
        width=width,
        height=height,
        orientation=orientation,
    )


def rotate_box(
    box: BBoxLike,
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> BBox:
    """Alias for :func:`transform_bbox`."""

    return rotate_bbox(
        box,
        page_size,
        rotation,
        width=width,
        height=height,
        orientation=orientation,
    )


def transform_ocr_words(
    words: Iterable[OcrWord],
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> tuple[OcrWord, ...]:
    """Transform OCR boxes while preserving text, pages, confidence, and order."""

    transformer = _make_transform(
        page_size,
        rotation,
        width=width,
        height=height,
        orientation=orientation,
    )
    return tuple(replace(word, bbox=transformer.bbox(word.bbox)) for word in words)


def transform_ocr_result(
    result: OcrResult,
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> OcrResult:
    """Transform an :class:`OcrResult` without changing source references.

    The ordered word sequence and result metadata are copied unchanged.  Call
    ``transformed.to_layout()`` when the transformed geometry should drive a
    fresh reading-order reconstruction.
    """

    if not isinstance(result, OcrResult):
        raise TypeError("result must be an OcrResult")
    words = transform_ocr_words(
        result.words,
        page_size,
        rotation,
        width=width,
        height=height,
        orientation=orientation,
    )
    return OcrResult(words=words, metadata=dict(result.metadata))


def transform_source_spans(
    spans: Iterable[SourceSpan],
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> tuple[SourceSpan, ...]:
    """Transform span boxes while preserving offsets, pages, and metadata."""

    transformer = _make_transform(
        page_size,
        rotation,
        width=width,
        height=height,
        orientation=orientation,
    )
    return tuple(
        span if span.bbox is None else replace(span, bbox=transformer.bbox(span.bbox))
        for span in spans
    )


def transform_document(
    document: ExtractedDocument,
    page_size: PageSizeLike | None = None,
    rotation: PageRotationLike | None = None,
    *,
    width: Any = None,
    height: Any = None,
    orientation: PageRotationLike | None = None,
) -> ExtractedDocument:
    """Transform document source boxes while preserving text and references."""

    if not isinstance(document, ExtractedDocument):
        raise TypeError("document must be an ExtractedDocument")
    return replace(
        document,
        spans=transform_source_spans(
            document.spans,
            page_size,
            rotation,
            width=width,
            height=height,
            orientation=orientation,
        ),
    )


rotate_ocr_words = transform_ocr_words
rotate_ocr_result = transform_ocr_result
rotate_source_spans = transform_source_spans
rotate_document = transform_document


__all__ = [
    "Point",
    "BBox",
    "PageSize",
    "PageDimensions",
    "PageRotation",
    "Rotation",
    "PageTransform",
    "PageRotationError",
    "AmbiguousOrientationError",
    "InvalidPageDimensionsError",
    "GeometryValidationError",
    "OutOfBoundsError",
    "transform_point",
    "transform_bbox",
    "transform_box",
    "rotate_point",
    "rotate_bbox",
    "rotate_box",
    "transform_ocr_words",
    "transform_ocr_result",
    "transform_source_spans",
    "transform_document",
    "rotate_ocr_words",
    "rotate_ocr_result",
    "rotate_source_spans",
    "rotate_document",
]
