"""Deterministic, bounded normalization for OCR page boxes.

OCR engines commonly report rectangles in pixels, PDF points, or relative
fractions.  This module converts those representations into one page-space
contract: a top-left-origin ``(x0, y0, x1, y1)`` rectangle with every value in
the closed interval ``[0, 1]``.  Pixel and point page dimensions are supplied
in the same unit as the input rectangle.

The normalizer is deliberately strict.  It does not clip malformed input,
guess a missing unit, or infer whether a four-value sequence is ``xyxy`` or
``xywh``.  Rejecting those cases keeps source-span provenance deterministic.
Only opaque caller-supplied ``source_ref`` values are carried through; OCR
text is not accepted or included in errors and reports.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

BBox: TypeAlias = tuple[float, float, float, float]
BoxLike: TypeAlias = Sequence[Any] | Mapping[str, Any]
PageSizeLike: TypeAlias = "PageSize | Sequence[Any] | Mapping[str, Any]"


class BoxNormalizationError(ValueError):
    """Base error for malformed or ambiguous OCR box input.

    Error messages contain only stable field names and reason codes.  They do
    not interpolate coordinates, source references, or OCR text.
    """

    def __init__(self, reason: str, *, field_name: str | None = None) -> None:
        self.reason = reason
        self.field_name = field_name
        location = f" ({field_name})" if field_name else ""
        super().__init__(f"invalid OCR box{location}: {reason}")


class AmbiguousBoxError(BoxNormalizationError):
    """Raised when a box has more than one valid interpretation."""

    def __init__(
        self, reason: str = "box interpretation is ambiguous", **kwargs: Any
    ) -> None:
        if "ambiguous" not in reason.lower():
            reason = f"ambiguous {reason}"
        super().__init__(reason, **kwargs)


class BoxValidationError(BoxNormalizationError):
    """Raised when a box or page geometry violates the contract."""


# Descriptive aliases keep the public error surface easy to discover.
InvalidBoxError = BoxValidationError
CoordinateNormalizationError = BoxNormalizationError


class CoordinateUnit(str, Enum):
    """Supported source units for OCR coordinates."""

    PIXEL = "pixel"
    POINT = "point"
    NORMALIZED = "normalized"

    @classmethod
    def coerce(cls, value: Any) -> "CoordinateUnit":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise BoxValidationError("unsupported coordinate unit", field_name="unit")
        normalized = value.strip().lower().replace("_", "-")
        aliases = {
            "pixel": cls.PIXEL,
            "pixels": cls.PIXEL,
            "px": cls.PIXEL,
            "point": cls.POINT,
            "points": cls.POINT,
            "pt": cls.POINT,
            "normalized": cls.NORMALIZED,
            "normalised": cls.NORMALIZED,
            "fraction": cls.NORMALIZED,
            "fractions": cls.NORMALIZED,
            "relative": cls.NORMALIZED,
        }
        try:
            return aliases[normalized]
        except KeyError:
            raise BoxValidationError(
                "unsupported coordinate unit", field_name="unit"
            ) from None


class CoordinateOrigin(str, Enum):
    """Supported source origins; output always uses ``top-left``."""

    TOP_LEFT = "top-left"
    BOTTOM_LEFT = "bottom-left"

    @classmethod
    def coerce(cls, value: Any) -> "CoordinateOrigin":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise BoxValidationError(
                "unsupported coordinate origin", field_name="origin"
            )
        normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
        aliases = {
            "top-left": cls.TOP_LEFT,
            "topleft": cls.TOP_LEFT,
            "top": cls.TOP_LEFT,
            "bottom-left": cls.BOTTOM_LEFT,
            "bottomleft": cls.BOTTOM_LEFT,
            "bottom": cls.BOTTOM_LEFT,
        }
        try:
            return aliases[normalized]
        except KeyError:
            raise BoxValidationError(
                "unsupported coordinate origin", field_name="origin"
            ) from None


# Short aliases are useful for callers that think in terms of a box rather
# than a generic coordinate system.
BoxUnit = CoordinateUnit
BoxOrigin = CoordinateOrigin


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise BoxValidationError("numeric value required", field_name=field_name)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise BoxValidationError(
            "numeric value required", field_name=field_name
        ) from None
    if math.isnan(converted):
        raise BoxValidationError("NaN coordinate", field_name=field_name)
    if math.isinf(converted):
        raise BoxValidationError("infinite coordinate", field_name=field_name)
    return converted


def _positive_dimension(value: Any, *, field_name: str) -> float:
    converted = _finite_float(value, field_name=field_name)
    if converted <= 0:
        raise BoxValidationError(
            "page dimension must be positive", field_name=field_name
        )
    return converted


@dataclass(frozen=True)
class PageSize:
    """Width and height of a source page in the source coordinate unit."""

    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "width",
            _positive_dimension(self.width, field_name="page_width"),
        )
        object.__setattr__(
            self,
            "height",
            _positive_dimension(self.height, field_name="page_height"),
        )

    def as_tuple(self) -> tuple[float, float]:
        """Return ``(width, height)`` as validated floats."""

        return self.width, self.height


def _validate_page(page: Any) -> int:
    if isinstance(page, bool):
        raise BoxValidationError(
            "page must be a non-negative integer", field_name="page"
        )
    try:
        converted = int(page)
    except (TypeError, ValueError, OverflowError):
        raise BoxValidationError(
            "page must be a non-negative integer", field_name="page"
        ) from None
    try:
        is_integral = float(page) == converted
    except (TypeError, ValueError, OverflowError):
        is_integral = False
    if not is_integral or converted < 0:
        raise BoxValidationError(
            "page must be a non-negative integer", field_name="page"
        )
    return converted


def _validate_source_ref(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BoxValidationError(
            "source_ref must be a non-empty opaque string", field_name="source_ref"
        )
    return value


def _validate_bbox(values: Sequence[Any], *, maximum: tuple[float, float]) -> BBox:
    if len(values) != 4:
        raise BoxValidationError("box must contain four coordinates", field_name="bbox")
    coordinates = tuple(_finite_float(value, field_name="bbox") for value in values)
    x0, y0, x1, y1 = coordinates
    if x0 >= x1 or y0 >= y1:
        raise BoxValidationError("inverted or zero-area box", field_name="bbox")
    max_x, max_y = maximum
    if x0 < 0 or y0 < 0 or x1 > max_x or y1 > max_y:
        raise BoxValidationError("out-of-bounds coordinate", field_name="bbox")
    return coordinates  # type: ignore[return-value]


def _canonical_float(value: float) -> float:
    # Avoid leaking signed zero into serialized provenance and keep exact page
    # edges stable after origin conversion.
    if value == 0.0:
        return 0.0
    if value == 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class NormalizedBox:
    """A validated top-left-origin rectangle in normalized page space.

    ``bbox`` is always ``(x0, y0, x1, y1)`` with all values in ``[0, 1]``.
    ``source_ref`` is an opaque, caller-owned stable reference such as a word
    index.  It is intentionally excluded from ``repr`` so routine diagnostics
    cannot accidentally print it.
    """

    bbox: BBox
    page: int = 0
    source_unit: CoordinateUnit = CoordinateUnit.NORMALIZED
    source_origin: CoordinateOrigin = CoordinateOrigin.TOP_LEFT
    source_ref: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bbox",
            _validate_bbox(self.bbox, maximum=(1.0, 1.0)),
        )
        object.__setattr__(self, "page", _validate_page(self.page))
        object.__setattr__(self, "source_unit", CoordinateUnit.coerce(self.source_unit))
        object.__setattr__(
            self,
            "source_origin",
            CoordinateOrigin.coerce(self.source_origin),
        )
        object.__setattr__(self, "source_ref", _validate_source_ref(self.source_ref))

    @property
    def coordinates(self) -> BBox:
        """Alias for the canonical normalized bounding box."""

        return self.bbox

    @property
    def page_bbox(self) -> BBox:
        """Alias for :attr:`bbox` used by page projection callers."""

        return self.bbox

    @property
    def unit(self) -> CoordinateUnit:
        """Return the unit used by the source OCR adapter."""

        return self.source_unit

    @property
    def origin(self) -> CoordinateOrigin:
        """Return the origin used by the source OCR adapter."""

        return self.source_origin

    def to_tuple(self) -> BBox:
        """Return only the canonical coordinates."""

        return self.bbox

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready, provenance-preserving representation.

        Callers should provide an opaque ``source_ref`` rather than OCR text
        or another sensitive value.  The normalizer never creates such a value
        and never emits source text.
        """

        payload: dict[str, Any] = {
            "bbox": list(self.bbox),
            "coordinate_system": "normalized-page",
            "page": self.page,
            "source_origin": self.source_origin.value,
            "source_unit": self.source_unit.value,
        }
        if self.source_ref is not None:
            payload["source_ref"] = self.source_ref
        return payload

    def __iter__(self):
        return iter(self.bbox)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int | slice) -> float | BBox:
        return self.bbox[index]  # type: ignore[return-value]


@dataclass(frozen=True)
class _ParsedInput:
    coordinates: BBox
    unit: Any = None
    origin: Any = None
    page_size: Any = None
    page: Any = None
    source_ref: Any = None


def _single_mapping_value(
    mapping: Mapping[str, Any], keys: Sequence[str], *, field_name: str
) -> Any:
    present = tuple(key for key in keys if key in mapping)
    if len(present) > 1:
        raise AmbiguousBoxError(f"multiple {field_name} fields", field_name=field_name)
    return mapping[present[0]] if present else None


def _coordinate_payload(raw: Any, *, format_hint: Any = None) -> BBox:
    if isinstance(raw, Mapping):
        return _coordinates_from_mapping(raw)
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise BoxValidationError("box must contain four coordinates", field_name="bbox")
    if len(raw) != 4:
        if len(raw) > 4:
            raise AmbiguousBoxError("ambiguous coordinate sequence", field_name="bbox")
        raise BoxValidationError("box must contain four coordinates", field_name="bbox")
    if any(
        isinstance(value, (Mapping, Sequence))
        and not isinstance(value, (str, bytes, bytearray))
        for value in raw
    ):
        raise AmbiguousBoxError("ambiguous coordinate sequence", field_name="bbox")
    coordinates = tuple(_finite_float(value, field_name="bbox") for value in raw)
    normalized_format = "xyxy" if format_hint is None else str(format_hint).lower()
    if normalized_format in {"xyxy", "ltrb", "left-top-right-bottom"}:
        return coordinates  # type: ignore[return-value]
    if normalized_format in {"xywh", "left-top-width-height"}:
        x, y, width, height = coordinates
        return x, y, x + width, y + height
    raise AmbiguousBoxError("unsupported box coordinate format", field_name="format")


def _coordinates_from_mapping(mapping: Mapping[str, Any]) -> BBox:
    styles = (
        ("xyxy", ("x0", "y0", "x1", "y1")),
        ("edges", ("left", "top", "right", "bottom")),
        ("xywh", ("x", "y", "width", "height")),
    )
    direct_styles: list[tuple[str, tuple[str, ...]]] = []
    for style, keys in styles:
        present = tuple(key for key in keys if key in mapping)
        if present:
            if len(present) != len(keys):
                raise BoxValidationError(
                    "incomplete box coordinate fields", field_name="bbox"
                )
            direct_styles.append((style, keys))

    nested_keys = tuple(key for key in ("bbox", "box", "coordinates") if key in mapping)
    if len(direct_styles) + len(nested_keys) != 1:
        if len(direct_styles) + len(nested_keys) == 0:
            raise BoxValidationError("box coordinates are required", field_name="bbox")
        raise AmbiguousBoxError(
            "multiple box coordinate representations", field_name="bbox"
        )

    if nested_keys:
        return _coordinate_payload(
            mapping[nested_keys[0]], format_hint=mapping.get("format")
        )

    style, keys = direct_styles[0]
    coordinates = tuple(_finite_float(mapping[key], field_name="bbox") for key in keys)
    if style == "xywh":
        x, y, width, height = coordinates
        return x, y, x + width, y + height
    return coordinates  # type: ignore[return-value]


def _parse_mapping(mapping: Mapping[str, Any]) -> _ParsedInput:
    return _ParsedInput(
        coordinates=_coordinates_from_mapping(mapping),
        unit=_single_mapping_value(
            mapping,
            ("unit", "source_unit", "coordinate_unit"),
            field_name="unit",
        ),
        origin=_single_mapping_value(
            mapping,
            ("origin", "source_origin", "coordinate_origin"),
            field_name="origin",
        ),
        page_size=_mapping_page_size(mapping),
        page=_single_mapping_value(mapping, ("page", "page_index"), field_name="page"),
        source_ref=_single_mapping_value(
            mapping,
            ("source_ref", "source_reference", "source_id"),
            field_name="source_ref",
        ),
    )


def _mapping_page_size(mapping: Mapping[str, Any]) -> Any:
    nested = _single_mapping_value(
        mapping, ("page_size", "page_dimensions"), field_name="page_size"
    )
    width_present = "page_width" in mapping
    height_present = "page_height" in mapping
    if nested is not None and (width_present or height_present):
        raise AmbiguousBoxError(
            "multiple page size representations", field_name="page_size"
        )
    if width_present != height_present:
        raise BoxValidationError(
            "page width and height are both required", field_name="page_size"
        )
    if nested is not None:
        return nested
    if width_present:
        return mapping["page_width"], mapping["page_height"]
    return None


def _parse_input(box: BoxLike) -> _ParsedInput:
    if isinstance(box, NormalizedBox):
        return _ParsedInput(
            coordinates=box.bbox,
            unit=box.source_unit,
            origin=box.source_origin,
            page=box.page,
            source_ref=box.source_ref,
        )
    if isinstance(box, Mapping):
        return _parse_mapping(box)
    return _ParsedInput(coordinates=_coordinate_payload(box))


def _coerce_page_size(value: Any) -> PageSize | None:
    if value is None:
        return None
    if isinstance(value, PageSize):
        return value
    if isinstance(value, Mapping):
        width = value.get("width", value.get("page_width"))
        height = value.get("height", value.get("page_height"))
        if width is None or height is None:
            raise BoxValidationError(
                "page width and height are both required", field_name="page_size"
            )
        return PageSize(width, height)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BoxValidationError(
            "page size must contain width and height", field_name="page_size"
        )
    if len(value) != 2:
        raise BoxValidationError(
            "page size must contain width and height", field_name="page_size"
        )
    return PageSize(value[0], value[1])


def _resolve_page_size(
    explicit_page_size: PageSizeLike | None,
    page_width: Any,
    page_height: Any,
    embedded_page_size: Any,
) -> PageSize | None:
    if explicit_page_size is not None and (
        page_width is not None or page_height is not None
    ):
        raise AmbiguousBoxError(
            "multiple page size representations", field_name="page_size"
        )
    if page_width is not None or page_height is not None:
        if page_width is None or page_height is None:
            raise BoxValidationError(
                "page width and height are both required", field_name="page_size"
            )
        explicit_page_size = (page_width, page_height)
    explicit = _coerce_page_size(explicit_page_size)
    embedded = _coerce_page_size(embedded_page_size)
    if explicit is not None and embedded is not None and explicit != embedded:
        raise AmbiguousBoxError("conflicting page_size values", field_name="page_size")
    return explicit if explicit is not None else embedded


def _resolve_unit(explicit: Any, embedded: Any) -> CoordinateUnit:
    explicit_unit = CoordinateUnit.coerce(explicit) if explicit is not None else None
    embedded_unit = CoordinateUnit.coerce(embedded) if embedded is not None else None
    if explicit_unit is not None and embedded_unit is not None:
        if explicit_unit != embedded_unit:
            raise AmbiguousBoxError("conflicting unit values", field_name="unit")
        return explicit_unit
    resolved = explicit_unit or embedded_unit
    if resolved is None:
        raise AmbiguousBoxError("coordinate unit is required", field_name="unit")
    return resolved


def _resolve_origin(explicit: Any, embedded: Any) -> CoordinateOrigin:
    explicit_origin = (
        CoordinateOrigin.coerce(explicit) if explicit is not None else None
    )
    embedded_origin = (
        CoordinateOrigin.coerce(embedded) if embedded is not None else None
    )
    if explicit_origin is not None and embedded_origin is not None:
        if explicit_origin != embedded_origin:
            raise AmbiguousBoxError("conflicting origin values", field_name="origin")
        return explicit_origin
    return explicit_origin or embedded_origin or CoordinateOrigin.TOP_LEFT


def _resolve_page(explicit: Any, embedded: Any) -> int:
    explicit_page = _validate_page(explicit) if explicit is not None else None
    embedded_page = _validate_page(embedded) if embedded is not None else None
    if explicit_page is not None and embedded_page is not None:
        if explicit_page != embedded_page:
            raise AmbiguousBoxError("conflicting page values", field_name="page")
        return explicit_page
    return explicit_page if explicit_page is not None else embedded_page or 0


def _resolve_source_ref(explicit: Any, embedded: Any) -> str | None:
    explicit_ref = _validate_source_ref(explicit)
    embedded_ref = _validate_source_ref(embedded)
    if explicit_ref is not None and embedded_ref is not None:
        if explicit_ref != embedded_ref:
            raise AmbiguousBoxError(
                "conflicting source_ref values", field_name="source_ref"
            )
        return explicit_ref
    return explicit_ref if explicit_ref is not None else embedded_ref


def normalize_box(
    box: BoxLike,
    *,
    unit: CoordinateUnit | str | None = None,
    page_size: PageSizeLike | None = None,
    page_width: Any = None,
    page_height: Any = None,
    origin: CoordinateOrigin | str | None = None,
    page: int | None = None,
    source_ref: str | None = None,
) -> NormalizedBox:
    """Normalize one OCR rectangle into bounded top-left page space.

    Args:
        box: A four-value ``(x0, y0, x1, y1)`` sequence or a mapping using
            ``bbox``, ``x0/y0/x1/y1``, ``left/top/right/bottom``, or
            ``x/y/width/height``.  A mapping may carry the other options.
        unit: Explicit source unit: ``"pixel"``, ``"point"``, or
            ``"normalized"``.  It is required unless the mapping carries it.
        page_size: Source page ``(width, height)``.  Required for pixel and
            point input; normalized input uses a unit page.
        page_width: Width shorthand for ``page_size``.
        page_height: Height shorthand for ``page_size``.
        origin: Source origin, either ``"top-left"`` or ``"bottom-left"``.
            The default is top-left.
        page: Non-negative zero-based page index.
        source_ref: Opaque stable source reference preserved in the result.

    Returns:
        A :class:`NormalizedBox` whose ``bbox`` is a finite top-left-origin
        rectangle bounded by ``[0, 1]``.

    Raises:
        AmbiguousBoxError: If the unit, representation, or provenance options
            have conflicting or missing interpretations.
        BoxValidationError: If coordinates or page geometry are malformed.
    """

    if (
        isinstance(box, NormalizedBox)
        and unit is None
        and page_size is None
        and page_width is None
        and page_height is None
        and origin is None
        and page is None
        and source_ref is None
    ):
        return box

    parsed = _parse_input(box)
    resolved_unit = _resolve_unit(unit, parsed.unit)
    resolved_origin = _resolve_origin(origin, parsed.origin)
    resolved_page = _resolve_page(page, parsed.page)
    resolved_ref = _resolve_source_ref(source_ref, parsed.source_ref)
    resolved_size = _resolve_page_size(
        page_size, page_width, page_height, parsed.page_size
    )

    if resolved_unit is CoordinateUnit.NORMALIZED:
        maximum = (1.0, 1.0)
        # Normalized input already describes a unit square.  A supplied page
        # size is still validated for callers that attach page metadata, but
        # it must not rescale relative coordinates a second time.
        source_size = PageSize(1.0, 1.0)
    else:
        if resolved_size is None:
            raise AmbiguousBoxError(
                "page_size is required for pixel or point coordinates",
                field_name="page_size",
            )
        maximum = resolved_size.as_tuple()
        source_size = resolved_size

    validated = _validate_bbox(parsed.coordinates, maximum=maximum)
    x0, y0, x1, y1 = validated
    if resolved_origin is CoordinateOrigin.BOTTOM_LEFT:
        y0, y1 = source_size.height - y1, source_size.height - y0

    normalized = (
        _canonical_float(x0 / source_size.width),
        _canonical_float(y0 / source_size.height),
        _canonical_float(x1 / source_size.width),
        _canonical_float(y1 / source_size.height),
    )
    return NormalizedBox(
        bbox=normalized,
        page=resolved_page,
        source_unit=resolved_unit,
        source_origin=resolved_origin,
        source_ref=resolved_ref,
    )


def normalize_boxes(
    boxes: Iterable[BoxLike],
    *,
    unit: CoordinateUnit | str | None = None,
    page_size: PageSizeLike | None = None,
    page_width: Any = None,
    page_height: Any = None,
    origin: CoordinateOrigin | str | None = None,
    page: int | None = None,
    source_ref: str | None = None,
) -> tuple[NormalizedBox, ...]:
    """Normalize an ordered collection while preserving source references."""

    return tuple(
        normalize_box(
            box,
            unit=unit,
            page_size=page_size,
            page_width=page_width,
            page_height=page_height,
            origin=origin,
            page=page,
            source_ref=source_ref,
        )
        for box in boxes
    )


def normalize_coordinates(box: BoxLike, **kwargs: Any) -> BBox:
    """Return only the canonical coordinates for one OCR rectangle."""

    return normalize_box(box, **kwargs).bbox


# Common naming variants share the same strict implementation.
normalize_bbox = normalize_box
normalize_ocr_box = normalize_box


__all__ = [
    "BBox",
    "BoxLike",
    "PageSizeLike",
    "BoxNormalizationError",
    "CoordinateNormalizationError",
    "AmbiguousBoxError",
    "BoxValidationError",
    "InvalidBoxError",
    "CoordinateUnit",
    "CoordinateOrigin",
    "BoxUnit",
    "BoxOrigin",
    "PageSize",
    "NormalizedBox",
    "normalize_box",
    "normalize_bbox",
    "normalize_ocr_box",
    "normalize_boxes",
    "normalize_coordinates",
]
