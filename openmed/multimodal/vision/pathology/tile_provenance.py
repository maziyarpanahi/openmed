"""Validate complete whole-slide tile lineage before aggregating protected features."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, ClassVar

from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.notices import NOTICE_CATALOG, MultimodalNotice, NoticeKind
from openmed.multimodal.vision.pathology.slide_tiler import (
    MAX_SLIDE_SIDE,
    MAX_TILES,
    SlideTileError,
    SlideTilePlan,
    TileDecision,
    TissueMask,
    plan_slide_tiles,
)

PROVENANCE_VERSION = "openmed.slide-aggregation-provenance.v1"
PROVENANCE_NOTICE = "Tile-feature aggregation with declared source lineage only; not pixel-coverage, model-accuracy or clinical verification."
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_DIMENSIONS = 4096
_MAX_TOTAL_FEATURES = 262_144
_MAX_TRANSFORMS = 16
_MAX_FEATURE_MAGNITUDE = 1e100
_Matrix = tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]


class TileProvenanceError(ValueError):
    """Fixed invalid-lineage or incomplete-aggregation failure without payload values."""


class TransformKind(str, Enum):
    """Closed geometry operations; actual pixel processing remains with the adapter."""

    CROP = "crop"
    RESIZE = "resize"
    ROTATE = "clockwise_quarter_turns"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"


def _require(condition: bool) -> None:
    if not condition:
        raise TileProvenanceError("invalid tile aggregation provenance")


def _digest(value: str) -> None:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None)


def _size(value: tuple[int, int]) -> None:
    _require(type(value) is tuple and len(value) == 2)
    _require(all(type(side) is int and 1 <= side <= 8192 for side in value))


def _bounds(value: tuple[int, int, int, int]) -> None:
    _require(type(value) is tuple and len(value) == 4)
    _require(
        all(
            type(coordinate) is int and 0 <= coordinate <= MAX_SLIDE_SIDE
            for coordinate in value
        )
    )
    _require(value[0] < value[2] and value[1] < value[3])


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _matrix(*values: int | Fraction) -> _Matrix:
    a, b, c, d, e, f = values
    return Fraction(a), Fraction(b), Fraction(c), Fraction(d), Fraction(e), Fraction(f)


def _compose(parent: _Matrix, child: _Matrix) -> _Matrix:
    a, b, c, d, e, f = parent
    g, h, i, j, k, l = child
    return (
        a * g + b * j,
        a * h + b * k,
        a * i + b * l + c,
        d * g + e * j,
        d * h + e * k,
        d * i + e * l + f,
    )


def _rational(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


@dataclass(frozen=True, slots=True, kw_only=True)
class TileTransform:
    """Validated output-to-input geometry for one declared preprocessing operation.

    Args:
        kind: Crop, resize, clockwise rotation or axis flip.
        input_size: Width and height of the preceding tile image, in pixels.
        output_size: Width and height after the declared operation.
        crop_bounds: Required only for crop, within the input extent.
        quarter_turns: One, two or three clockwise turns; required only for rotation.

    Coordinates describe pixel edges, not decoder-specific interpolation centers.
    This declaration records geometry; it does not execute or verify pixel processing.
    """

    kind: TransformKind
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    crop_bounds: tuple[int, int, int, int] | None = None
    quarter_turns: int | None = None

    def __post_init__(self) -> None:
        _require(type(self.kind) is TransformKind)
        _size(self.input_size)
        _size(self.output_size)
        width, height = self.input_size
        if self.kind is TransformKind.CROP:
            _require(self.crop_bounds is not None and self.quarter_turns is None)
            assert self.crop_bounds is not None
            _bounds(self.crop_bounds)
            left, top, right, bottom = self.crop_bounds
            _require(right <= width and bottom <= height)
            _require(self.output_size == (right - left, bottom - top))
        elif self.kind is TransformKind.ROTATE:
            _require(
                self.crop_bounds is None
                and type(self.quarter_turns) is int
                and self.quarter_turns in (1, 2, 3)
            )
            assert self.quarter_turns is not None
            _require(
                self.output_size
                == ((height, width) if self.quarter_turns % 2 else (width, height))
            )
        else:
            _require(self.crop_bounds is None and self.quarter_turns is None)
            if self.kind is not TransformKind.RESIZE:
                _require(self.output_size == self.input_size)

    @property
    def output_to_input(self) -> _Matrix:
        """Return the exact affine mapping of output pixel edges into input space."""
        width, height = self.input_size
        if self.kind is TransformKind.CROP:
            assert self.crop_bounds is not None
            return _matrix(1, 0, self.crop_bounds[0], 0, 1, self.crop_bounds[1])
        if self.kind is TransformKind.RESIZE:
            return _matrix(
                Fraction(width, self.output_size[0]),
                0,
                0,
                0,
                Fraction(height, self.output_size[1]),
                0,
            )
        if self.kind is TransformKind.FLIP_HORIZONTAL:
            return _matrix(-1, 0, width, 0, 1, 0)
        if self.kind is TransformKind.FLIP_VERTICAL:
            return _matrix(1, 0, 0, 0, -1, height)
        if self.quarter_turns == 1:
            return _matrix(0, 1, 0, -1, 0, height)
        if self.quarter_turns == 2:
            return _matrix(-1, 0, width, 0, -1, height)
        return _matrix(0, -1, width, 1, 0, 0)


@dataclass(frozen=True, slots=True, kw_only=True)
class TileFeatures:
    """Protected feature vector bound to one planned source tile and transform history.

    Args:
        tile_ref: Stable identity from the current tile planner.
        plan_digest: Complete current plan, including selection policy and mask binding.
        source_digest: Original slide digest, never a filename or patient identifier.
        level: Original native pyramid level.
        native_bounds: Original tile bounds in native-level coordinates.
        source_bounds: Planner's original clipped level-zero citation box.
        processed_input_digest: Digest of the exact processed input consumed upstream.
        provider_digest: Pinned provider/weights/configuration provenance digest.
        feature_space_digest: Feature layer, schema and normalization specification digest.
        transforms: Complete ordered geometric preprocessing history.
        features: Protected finite numeric feature vector, excluded from reports and repr.
    """

    tile_ref: str
    plan_digest: str
    source_digest: str
    level: int
    native_bounds: tuple[int, int, int, int]
    source_bounds: tuple[int, int, int, int]
    processed_input_digest: str
    provider_digest: str
    feature_space_digest: str
    transforms: tuple[TileTransform, ...]
    features: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        for value in (
            self.tile_ref,
            self.plan_digest,
            self.source_digest,
            self.processed_input_digest,
            self.provider_digest,
            self.feature_space_digest,
        ):
            _digest(value)
        _require(type(self.level) is int and 0 <= self.level < 32)
        _bounds(self.native_bounds)
        _bounds(self.source_bounds)
        _size(
            (
                self.native_bounds[2] - self.native_bounds[0],
                self.native_bounds[3] - self.native_bounds[1],
            )
        )
        _require(
            type(self.transforms) is tuple and len(self.transforms) <= _MAX_TRANSFORMS
        )
        _require(all(type(item) is TileTransform for item in self.transforms))
        _require(
            type(self.features) is tuple and 1 <= len(self.features) <= _MAX_DIMENSIONS
        )
        _require(
            all(
                type(value) in (int, float)
                and -_MAX_FEATURE_MAGNITUDE <= value <= _MAX_FEATURE_MAGNITUDE
                for value in self.features
            )
        )
        object.__setattr__(
            self, "features", tuple(float(value) for value in self.features)
        )

    @property
    def feature_digest(self) -> str:
        """Bind the exact protected feature vector without serializing its values publicly."""
        return _hash(self.features)

    def to_report(self) -> dict[str, Any]:
        """Return declared provenance, dimensions and hashes without feature values."""
        return {
            "tile_ref": self.tile_ref,
            "plan_digest": self.plan_digest,
            "source_digest": self.source_digest,
            "level": self.level,
            "native_bounds": self.native_bounds,
            "source_bounds": self.source_bounds,
            "processed_input_digest": self.processed_input_digest,
            "provider_digest": self.provider_digest,
            "feature_space_digest": self.feature_space_digest,
            "transforms": [asdict(item) for item in self.transforms],
            "feature_dimensions": len(self.features),
            "feature_digest": self.feature_digest,
        }

    @property
    def digest(self) -> str:
        """Bind all source, transform, provider and feature lineage for this result."""
        return _hash({"version": PROVENANCE_VERSION, **self.to_report()})


def _fresh_plan(plan: SlideTilePlan, mask: TissueMask | None) -> SlideTilePlan:
    _require(type(plan) is SlideTilePlan)
    try:
        current = plan_slide_tiles(
            plan.source_digest,
            plan.source_width,
            plan.source_height,
            plan.levels,
            plan.policy,
            tissue_mask=mask,
        )
    except SlideTileError:
        pass
    else:
        if current == plan:
            return current
    raise TileProvenanceError("tile plan and selection evidence do not match")


def _lineage(result: TileFeatures, plan: SlideTilePlan) -> dict[str, Any]:
    level = next(item for item in plan.levels if item.index == result.level)
    scale = Fraction(level.scale_numerator, level.scale_denominator)
    left, top, right, bottom = result.native_bounds
    size = (right - left, bottom - top)
    mapping = _matrix(scale, 0, left * scale, 0, scale, top * scale)
    for transform in result.transforms:
        if transform.input_size != size:
            raise TileProvenanceError(
                "tile transform chain has inconsistent dimensions"
            )
        mapping = _compose(mapping, transform.output_to_input)
        size = transform.output_size
    a, b, c, d, e, f = mapping
    corners = tuple(
        (a * x + b * y + c, d * x + e * y + f)
        for x, y in ((0, 0), (size[0], 0), (size[0], size[1]), (0, size[1]))
    )
    source_edge_clipped = any(
        not (0 <= x <= plan.source_width and 0 <= y <= plan.source_height)
        for x, y in corners
    )
    x0, y0, x1, y1 = result.source_bounds
    evidence = EvidenceNode(
        node_id=_hash(["tile-source", result.tile_ref])[:32],
        kind=NodeKind.PIXELS,
        source_digest=plan.source_digest,
        geometry=PageGeometry(
            page=0, width=plan.source_width, height=plan.source_height
        ),
        region=(float(x0), float(y0), float(x1), float(y1)),
    )
    return {
        **result.to_report(),
        "result_digest": result.digest,
        "output_size": size,
        "output_to_level_zero": [_rational(value) for value in mapping],
        "projected_source_corners": [[_rational(x), _rational(y)] for x, y in corners],
        "source_edge_clipped": source_edge_clipped,
        "evidence": asdict(evidence),
    }


@dataclass(frozen=True, slots=True)
class SlideAggregation:
    """Protected computed feature mean with complete content-free source lineage."""

    review_notice: ClassVar[MultimodalNotice] = NOTICE_CATALOG[NoticeKind.MEASUREMENT]

    features: tuple[float, ...] = field(repr=False)
    report_json: str

    def to_json(self) -> str:
        """Return lineage and aggregate digests without feature values."""
        return self.report_json


def aggregate_tile_features(
    plan: SlideTilePlan,
    results: tuple[TileFeatures, ...],
    *,
    tissue_mask: TissueMask | None = None,
) -> SlideAggregation:
    """Validate complete tile provenance before computing an unweighted feature mean.

    Args:
        plan: Original complete plan, freshly reconstructed before trusting its tiles.
        results: Exactly one feature observation for every selected tile.
        tissue_mask: Original selection mask, required when the plan used a mask.

    Returns:
        Protected mean features and original selection/source/transform lineage.
        This is an equal-weight mean over tiles, not a spatially unbiased or
        clinically qualified summary; overlapping tiles remain separate inputs.

    Raises:
        TileProvenanceError: Invalid, stale, missing, duplicate or inconsistent
            inputs. No feature result or partial lineage is returned on failure.
    """
    current = _fresh_plan(plan, tissue_mask)
    selected = current.selected_tiles
    _require(type(results) is tuple and 1 <= len(results) <= MAX_TILES)
    _require(all(type(item) is TileFeatures for item in results))
    _require(len({item.tile_ref for item in results}) == len(results))
    if {item.tile_ref for item in results} != {tile.tile_ref for tile in selected}:
        raise TileProvenanceError("aggregation requires exactly the selected tile set")
    _require(sum(len(item.features) for item in results) <= _MAX_TOTAL_FEATURES)
    _require(len({len(item.features) for item in results}) == 1)
    _require(len({item.provider_digest for item in results}) == 1)
    _require(len({item.feature_space_digest for item in results}) == 1)
    observations = {item.tile_ref: item for item in results}
    ordered = tuple(observations[tile.tile_ref] for tile in selected)
    plan_digest = current.digest
    lineages = []
    for tile, result in zip(selected, ordered):
        if (
            result.plan_digest != plan_digest
            or result.source_digest != current.source_digest
            or result.level != tile.level
            or result.native_bounds != tile.bounds
            or result.source_bounds != tile.source_bounds
        ):
            raise TileProvenanceError(
                "tile result does not match current source geometry and plan"
            )
        lineages.append({**_lineage(result, current), "selection": asdict(tile)})
    # The feature bounds keep every sum finite; stable plan order also makes
    # floating-point reduction independent of result arrival order.
    features = tuple(
        math.fsum(item.features[index] for item in ordered) / len(ordered)
        for index in range(len(ordered[0].features))
    )
    input_digest = _hash(
        {
            "version": PROVENANCE_VERSION,
            "plan_digest": plan_digest,
            "results": [item.digest for item in ordered],
            "method": "equal_weight_tile_mean",
        }
    )
    report = {
        "schema_version": PROVENANCE_VERSION,
        "notice": PROVENANCE_NOTICE,
        "input_digest": input_digest,
        "aggregate_digest": _hash(features),
        "method": "equal_weight_tile_mean",
        "feature_dimensions": len(features),
        "review_required": True,
        "uncertain_selection_count": sum(
            tile.decision is TileDecision.REVIEW for tile in selected
        ),
        "plan": current.to_dict(),
        "tiles": lineages,
    }
    return SlideAggregation(
        features,
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )
