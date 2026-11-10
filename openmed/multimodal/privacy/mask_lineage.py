"""Exact, bounded propagation of protected source regions through image geometry."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, TypeAlias

from ..evidence.fused_graph import EvidenceNode, NodeKind, PageGeometry, SpatialUnit
from ..page_rotation import PageRotation, PageSize, PageTransform

MASK_LINEAGE_SCHEMA_VERSION = "openmed.multimodal.mask-lineage.v1"
MAX_MASKS = 256
MAX_TRANSFORMS = 32
MAX_COVERAGE_RECTS = 64
MAX_COVERAGE_CELLS = 1024
MAX_CLIP_TESTS = 100_000
MAX_ARITHMETIC_BITS = 512
Number: TypeAlias = int | float | Fraction
Point: TypeAlias = tuple[Fraction, Fraction]
Polygon: TypeAlias = tuple[Point, ...]
_REF = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class MaskLineageError(ValueError):
    """Fixed-message invalid geometry contract; never includes caller values."""


class MaskFailure(str, Enum):
    """Controlled fail-closed reasons; no partial masks accompany a failure."""

    SOURCE_MISMATCH = "source_mismatch"
    SPACE_MISMATCH = "coordinate_space_mismatch"
    SINGULAR = "noninvertible_transform"
    COVERAGE_LOSS = "protected_coverage_loss"
    RESOURCE_LIMIT = "resource_limit"
    ARITHMETIC_LIMIT = "arithmetic_limit"


class _ArithmeticLimit(MaskLineageError):
    def __init__(self) -> None:
        super().__init__("redaction mask arithmetic limit")


class _ResourceLimit(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise MaskLineageError("invalid redaction mask lineage")


def _bound(value: Fraction) -> Fraction:
    if (
        max(value.numerator.bit_length(), value.denominator.bit_length())
        > MAX_ARITHMETIC_BITS
    ):
        raise _ArithmeticLimit
    return value


def _number(value: Number) -> Fraction:
    _require(type(value) in (int, float, Fraction))
    if type(value) is float:
        _require(math.isfinite(value))
    if type(value) is int:
        _require(value.bit_length() <= MAX_ARITHMETIC_BITS)
    result = Fraction(value)
    _require(
        max(result.numerator.bit_length(), result.denominator.bit_length())
        <= MAX_ARITHMETIC_BITS
    )
    return result


def _ratio(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


@dataclass(frozen=True, slots=True)
class Rectangle:
    """Positive half-open rectangle in a declared top-left coordinate space."""

    left: Number
    top: Number
    right: Number
    bottom: Number

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            object.__setattr__(self, name, _number(getattr(self, name)))
        _require(0 <= self.left < self.right and 0 <= self.top < self.bottom)

    @property
    def polygon(self) -> Polygon:
        """Four ordered corners in the declared top-left space, without rounding."""
        left, top, right, bottom = map(
            _number, (self.left, self.top, self.right, self.bottom)
        )
        return ((left, top), (right, top), (right, bottom), (left, bottom))

    def to_dict(self) -> dict[str, Any]:
        """Return exact rational bounds as numerator/denominator pairs."""
        return {
            name: _ratio(_number(getattr(self, name)))
            for name in ("left", "top", "right", "bottom")
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CoordinateSpace:
    """One immutable geometry identity in the lineage of an original source.

    Args:
        space_ref: Opaque identity unique to this stage, not a path or patient ID.
        source_digest: Original source asset SHA-256, retained through derivatives.
            This does not claim to hash rendered derivative pixels.
        geometry: Page, extents and units from the existing evidence contract.
            Pixel extents must be integral; point extents may be fractional.
    """

    space_ref: str
    source_digest: str
    geometry: PageGeometry

    def __post_init__(self) -> None:
        _require(
            type(self.space_ref) is str and _REF.fullmatch(self.space_ref) is not None
        )
        _require(
            type(self.source_digest) is str
            and _DIGEST.fullmatch(self.source_digest) is not None
        )
        _require(type(self.geometry) is PageGeometry)
        if self.geometry.unit is SpatialUnit.PIXEL:
            _require(
                self.geometry.width.is_integer() and self.geometry.height.is_integer()
            )

    @property
    def extent(self) -> Rectangle:
        """Entire declared page, without extrapolated or padded coverage."""
        return Rectangle(0, 0, self.geometry.width, self.geometry.height)

    def to_dict(self) -> dict[str, Any]:
        """Return opaque identity, original digest and declared geometry."""
        return {
            "space_ref": self.space_ref,
            "source_digest": self.source_digest,
            "page": self.geometry.page,
            "unit": self.geometry.unit.value,
            "extent": self.extent.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class Affine:
    """Exact 2D map: x'=a*x+b*y+c; y'=d*x+e*y+f.

    Integers and Fraction values preserve declared rational intent. Floats are
    interpreted as their exact binary values, without an acceptance tolerance.
    """

    a: Number = 1
    b: Number = 0
    c: Number = 0
    d: Number = 0
    e: Number = 1
    f: Number = 0

    def __post_init__(self) -> None:
        for name in "abcdef":
            object.__setattr__(self, name, _number(getattr(self, name)))

    @property
    def values(self) -> tuple[Fraction, ...]:
        """Bounded exact coefficients in a,b,c,d,e,f order."""
        return tuple(_number(getattr(self, name)) for name in "abcdef")

    @property
    def determinant(self) -> Fraction:
        """Exact determinant; zero is never hidden by floating-point tolerance."""
        a, b, _, d, e, _ = self.values
        return _bound(a * e - b * d)

    def point(self, point: tuple[Number, Number]) -> Point:
        """Map one point without clipping or rounding."""
        _require(type(point) is tuple and len(point) == 2)
        x, y = map(_number, point)
        a, b, c, d, e, f = self.values
        return _bound(a * x + b * y + c), _bound(d * x + e * y + f)

    def after(self, previous: Affine) -> Affine:
        """Compose self after previous, preserving the original source mapping."""
        _require(type(previous) is Affine)
        a, b, c, d, e, f = self.values
        g, h, i, j, k, l = previous.values
        return Affine(
            *(
                _bound(value)
                for value in (
                    a * g + b * j,
                    a * h + b * k,
                    a * i + b * l + c,
                    d * g + e * j,
                    d * h + e * k,
                    d * i + e * l + f,
                )
            )
        )

    def inverse(self) -> Affine:
        """Invert the coordinate map; crop coverage is not restored by inversion."""
        a, b, c, d, e, f = self.values
        det = self.determinant
        _require(det != 0)
        return Affine(
            *(
                _bound(value / det)
                for value in (e, -b, b * f - e * c, -d, a, d * c - a * f)
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformStep:
    """One affine/crop stage with explicit output coverage and sampling support.

    Args:
        source: Exact previous coordinate-space declaration.
        target: New coordinate-space declaration for the same original asset/page.
        affine: Actual source-to-target transform used by the renderer.
        resampling_margin: Conservative filter support radius in target pixels.
            Explicit zero is suitable only for lossless or qualified zero-margin
            sampling. Point-space outputs require zero. Nonzero margins expand
            protected support conservatively before subsequent stages.
        coverage: Union of covered target rectangles. None declares the complete
            target extent; an empty tuple declares no covered output.
        rasterize: Whether this stage materializes target pixels. True expands
            support to whole pixel cells before subsequent transforms. False is
            only for coordinate composition without an intermediate raster.
    """

    source: CoordinateSpace
    target: CoordinateSpace
    affine: Affine
    resampling_margin: int
    coverage: tuple[Rectangle, ...] | None = None
    rasterize: bool = True

    def __post_init__(self) -> None:
        _require(
            type(self.source) is CoordinateSpace
            and type(self.target) is CoordinateSpace
        )
        _require(type(self.affine) is Affine)
        _require(
            type(self.resampling_margin) is int and 0 <= self.resampling_margin <= 128
        )
        _require(
            self.target.geometry.unit is SpatialUnit.PIXEL
            or self.resampling_margin == 0
        )
        _require(type(self.rasterize) is bool)
        _require(not self.rasterize or self.target.geometry.unit is SpatialUnit.PIXEL)
        _require(self.resampling_margin == 0 or self.rasterize)
        if self.coverage is not None:
            _require(
                type(self.coverage) is tuple
                and len(self.coverage) <= MAX_COVERAGE_RECTS
            )
            for rect in self.coverage:
                _require(type(rect) is Rectangle)
                _require(
                    rect.right <= self.target.geometry.width
                    and rect.bottom <= self.target.geometry.height
                )

    @classmethod
    def crop(
        cls, source: CoordinateSpace, target: CoordinateSpace, region: Rectangle
    ) -> TransformStep:
        """Declare a lossless crop without accepting protected coverage loss."""
        _require(type(region) is Rectangle)
        _require(
            region.right <= source.geometry.width
            and region.bottom <= source.geometry.height
        )
        _require(target.geometry.unit is source.geometry.unit)
        if source.geometry.unit is SpatialUnit.PIXEL:
            _require(
                all(
                    _number(value).denominator == 1
                    for value in (region.left, region.top, region.right, region.bottom)
                )
            )
        _require(_number(target.geometry.width) == region.right - region.left)
        _require(_number(target.geometry.height) == region.bottom - region.top)
        return cls(
            source=source,
            target=target,
            affine=Affine(c=-region.left, f=-region.top),
            resampling_margin=0,
            rasterize=target.geometry.unit is SpatialUnit.PIXEL,
        )

    @classmethod
    def scale(
        cls, source: CoordinateSpace, target: CoordinateSpace, *, resampling_margin: int
    ) -> TransformStep:
        """Scale declared extents, including explicit point-to-pixel page rendering."""
        return cls(
            source=source,
            target=target,
            affine=Affine(
                a=_number(target.geometry.width) / _number(source.geometry.width),
                e=_number(target.geometry.height) / _number(source.geometry.height),
            ),
            resampling_margin=resampling_margin,
            rasterize=target.geometry.unit is SpatialUnit.PIXEL,
        )

    @classmethod
    def rotate(
        cls, source: CoordinateSpace, target: CoordinateSpace, rotation: PageRotation
    ) -> TransformStep:
        """Reuse the existing lossless clockwise quarter-turn geometry contract."""
        _require(
            type(rotation) is PageRotation
            and target.geometry.unit is source.geometry.unit
        )
        transform = PageTransform(
            PageSize(source.geometry.width, source.geometry.height), rotation
        )
        _require(
            (target.geometry.width, target.geometry.height)
            == (transform.target_size.width, transform.target_size.height)
        )
        width, height = source.geometry.width, source.geometry.height
        affine = {
            PageRotation.ROTATE_0: Affine(),
            PageRotation.ROTATE_90: Affine(0, -1, height, 1, 0, 0),
            PageRotation.ROTATE_180: Affine(-1, 0, width, 0, -1, height),
            PageRotation.ROTATE_270: Affine(0, 1, 0, -1, 0, width),
        }[rotation]
        return cls(
            source=source,
            target=target,
            affine=affine,
            resampling_margin=0,
            rasterize=target.geometry.unit is SpatialUnit.PIXEL,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the declared transform and coverage, without media payloads."""
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "affine": [_ratio(value) for value in self.affine.values],
            "resampling_margin": self.resampling_margin,
            "rasterize": self.rasterize,
            "coverage": [rect.to_dict() for rect in self.coverage]
            if self.coverage is not None
            else None,
        }


@dataclass(frozen=True, slots=True)
class MaskProjection:
    """Protected support in the target space, linked to immutable source evidence."""

    node_id: str
    polygon: Polygon
    raster_box: tuple[int, int, int, int] | None

    def to_dict(self) -> dict[str, Any]:
        """Return exact support coordinates and outward-rounded raster bounds."""
        return {
            "node_id": self.node_id,
            "polygon": [[_ratio(x), _ratio(y)] for x, y in self.polygon],
            "raster_box": self.raster_box,
        }


@dataclass(frozen=True, slots=True)
class MaskLineageResult:
    """All-or-nothing mask geometry; rejection never contains usable partial masks."""

    source: CoordinateSpace
    target: CoordinateSpace
    source_nodes: tuple[EvidenceNode, ...]
    steps: tuple[TransformStep, ...]
    masks: tuple[MaskProjection, ...]
    composed: Affine | None
    failure: MaskFailure | None
    failed_step: int | None

    @property
    def accepted(self) -> bool:
        """Whether every declared protected region retained complete coverage."""
        return self.failure is None

    def to_json(self) -> str:
        """Serialize coordinates, opaque references and controlled failures only."""
        return json.dumps(
            {
                "schema_version": MASK_LINEAGE_SCHEMA_VERSION,
                "accepted": self.accepted,
                "failure": self.failure.value if self.failure else None,
                "failed_step": self.failed_step,
                "source": self.source.to_dict(),
                "target": self.target.to_dict(),
                "sources": [
                    {
                        "node_id": node.node_id,
                        "kind": node.kind.value,
                        "region": Rectangle(*node.region).to_dict(),
                        "span": node.span,
                    }
                    for node in self.source_nodes
                    if node.region is not None
                ],
                "steps": [step.to_dict() for step in self.steps],
                "masks": [mask.to_dict() for mask in self.masks],
                "composed": [_ratio(value) for value in self.composed.values]
                if self.composed
                else None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def _area(polygon: Polygon) -> Fraction:
    if not polygon:
        return Fraction(0)
    return _bound(
        abs(
            sum(
                (
                    _bound(x * v - y * u)
                    for (x, y), (u, v) in zip(polygon, polygon[1:] + polygon[:1])
                ),
                Fraction(0),
            )
        )
        / 2
    )


def _clip(polygon: Polygon, rect: Rectangle) -> Polygon:
    for axis, edge, lower in (
        (0, rect.left, True),
        (0, rect.right, False),
        (1, rect.top, True),
        (1, rect.bottom, False),
    ):
        if not polygon:
            break
        output = []
        boundary = _number(edge)
        previous = polygon[-1]
        previous_inside = (
            previous[axis] >= boundary if lower else previous[axis] <= boundary
        )
        for current in polygon:
            inside = current[axis] >= boundary if lower else current[axis] <= boundary
            if inside != previous_inside:
                ratio = _bound(
                    (boundary - previous[axis]) / (current[axis] - previous[axis])
                )
                output.append(
                    (
                        _bound(previous[0] + ratio * (current[0] - previous[0])),
                        _bound(previous[1] + ratio * (current[1] - previous[1])),
                    )
                )
            if inside:
                output.append(current)
            previous, previous_inside = current, inside
        polygon = tuple(output)
    return polygon


def _cells(rectangles: tuple[Rectangle, ...]) -> tuple[Rectangle, ...]:
    """Disjoint vertical cells of a rectangle union, so overlaps never double count."""
    xs = sorted({_number(x) for rect in rectangles for x in (rect.left, rect.right)})
    cells: list[Rectangle] = []
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (_number(rect.top), _number(rect.bottom))
            for rect in rectangles
            if rect.left <= left and right <= rect.right
        )
        merged: list[tuple[Fraction, Fraction]] = []
        for top, bottom in intervals:
            if merged and top <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], bottom))
            else:
                merged.append((top, bottom))
        cells.extend(Rectangle(left, top, right, bottom) for top, bottom in merged)
        if len(cells) > MAX_COVERAGE_CELLS:
            raise _ResourceLimit
    return tuple(cells)


def _bounds(polygon: Polygon) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return (
        min(x for x, _ in polygon),
        min(y for _, y in polygon),
        max(x for x, _ in polygon),
        max(y for _, y in polygon),
    )


def propagate_masks(
    source: CoordinateSpace,
    masks: tuple[EvidenceNode, ...],
    steps: tuple[TransformStep, ...],
    *,
    max_clip_tests: int = MAX_CLIP_TESTS,
) -> MaskLineageResult:
    """Compose declared geometry and require complete protected coverage at every stage.

    Args:
        source: Coordinate identity matching every original evidence node.
        masks: Spatial PIXELS/OCR evidence; original regions are never overwritten.
        steps: Ordered transforms with unique target identities and explicit support.
        max_clip_tests: Deterministic work ceiling for coverage intersection tests.

    Returns:
        Exact coordinate lineage and conservative raster boxes, or a controlled
        rejection with no partial masks. No media is decoded or redacted here.
        A successful geometry plan does not detect PHI or verify a renderer.
    """
    _require(type(source) is CoordinateSpace)
    _require(type(masks) is tuple and 0 < len(masks) <= MAX_MASKS)
    _require(type(steps) is tuple and len(steps) <= MAX_TRANSFORMS)
    _require(
        all(
            type(node) is EvidenceNode and node.kind in (NodeKind.PIXELS, NodeKind.OCR)
            for node in masks
        )
    )
    _require(len({node.node_id for node in masks}) == len(masks))
    _require(all(type(step) is TransformStep for step in steps))
    _require(type(max_clip_tests) is int and 1 <= max_clip_tests <= MAX_CLIP_TESTS)
    current, composed, index = source, Affine(), None

    def reject(reason: MaskFailure) -> MaskLineageResult:
        return MaskLineageResult(source, current, masks, steps, (), None, reason, index)

    if any(
        node.source_digest != source.source_digest or node.geometry != source.geometry
        for node in masks
    ):
        return reject(MaskFailure.SOURCE_MISMATCH)
    polygons = tuple(
        Rectangle(*node.region).polygon for node in masks if node.region is not None
    )
    identities = {source.space_ref}
    work = 0
    try:
        for index, step in enumerate(steps):
            if (
                step.source != current
                or step.target.space_ref in identities
                or step.target.source_digest != source.source_digest
                or step.target.geometry.page != source.geometry.page
            ):
                return reject(MaskFailure.SPACE_MISMATCH)
            if step.affine.determinant == 0:
                return reject(MaskFailure.SINGULAR)
            composed = step.affine.after(composed)
            current = step.target
            identities.add(current.space_ref)
            cells = _cells(
                step.coverage if step.coverage is not None else (current.extent,)
            )
            work += len(polygons) * len(cells)
            if work > max_clip_tests:
                return reject(MaskFailure.RESOURCE_LIMIT)
            projected = []
            for polygon in polygons:
                mapped = tuple(step.affine.point(point) for point in polygon)
                left, top, right, bottom = _bounds(mapped)
                margin = step.resampling_margin
                if (
                    left - margin < 0
                    or top - margin < 0
                    or right + margin > current.geometry.width
                    or bottom + margin > current.geometry.height
                ):
                    return reject(MaskFailure.COVERAGE_LOSS)
                if step.resampling_margin:
                    mapped = Rectangle(
                        left - margin, top - margin, right + margin, bottom + margin
                    ).polygon
                if step.rasterize:
                    left, top, right, bottom = _bounds(mapped)
                    if left < 0 or top < 0:
                        return reject(MaskFailure.COVERAGE_LOSS)
                    mapped = Rectangle(
                        math.floor(left),
                        math.floor(top),
                        math.ceil(right),
                        math.ceil(bottom),
                    ).polygon
                area = _area(mapped)
                covered = _bound(
                    sum((_area(_clip(mapped, cell)) for cell in cells), Fraction(0))
                )
                if area <= 0 or area != covered:
                    return reject(MaskFailure.COVERAGE_LOSS)
                projected.append(mapped)
            polygons = tuple(projected)
        output = []
        for node, polygon in zip(masks, polygons):
            left, top, right, bottom = _bounds(polygon)
            raster = None
            if current.geometry.unit is SpatialUnit.PIXEL and (
                not steps or steps[-1].rasterize
            ):
                raster = (
                    math.floor(left),
                    math.floor(top),
                    math.ceil(right),
                    math.ceil(bottom),
                )
            output.append(MaskProjection(node.node_id, polygon, raster))
        return MaskLineageResult(
            source, current, masks, steps, tuple(output), composed, None, None
        )
    except _ArithmeticLimit:
        return reject(MaskFailure.ARITHMETIC_LIMIT)
    except _ResourceLimit:
        return reject(MaskFailure.RESOURCE_LIMIT)
