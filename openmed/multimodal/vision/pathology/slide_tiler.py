"""Deterministic multiresolution tile plans with bounded mask and buffer work."""

from __future__ import annotations

import hashlib
import json
import math
import re
from array import array
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterator

SLIDE_TILE_SCHEMA_VERSION = "openmed.vision.slide_tiles.v1"
MAX_LEVELS = 32
MAX_TILES = 4096
MAX_MASK_SIDE = 512
MAX_SLIDE_SIDE = 1_000_000_000
MAX_MEMORY_BYTES = (1 << 63) - 1
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class SlideTileError(ValueError):
    """Value-free rejection before a partial plan or pixel allocation is returned."""


class TileDecision(str, Enum):
    """Coarse-mask routing, never a clinical tissue classification."""

    KEEP = "keep_tissue_candidate"
    REVIEW = "keep_uncertain_coverage"
    SKIP = "skip_background_candidate"


def _require(condition: bool) -> None:
    if not condition:
        raise SlideTileError("invalid slide tile plan")


def _integer(value: object, minimum: int, maximum: int) -> None:
    _require(type(value) is int and minimum <= value <= maximum)


def _digest(value: object) -> None:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None)


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


@dataclass(frozen=True, slots=True, kw_only=True)
class PyramidLevel:
    """Declared native level geometry with an explicit rational downsample.

    Args:
        index: Native pyramid level index, not a filename or slide identifier.
        width: Native level width in pixels.
        height: Native level height in pixels.
        scale_numerator: Numerator of the native-to-level-zero pixel scale.
        scale_denominator: Denominator, coprime to the numerator.

    Both axes use the declared scale. Rounded level dimensions do not redefine
    that scale. The planner reports source edge gaps caused by rounded metadata.
    """

    index: int
    width: int
    height: int
    scale_numerator: int
    scale_denominator: int = 1

    def __post_init__(self) -> None:
        _integer(self.index, 0, MAX_LEVELS - 1)
        for side in (self.width, self.height):
            _integer(side, 1, MAX_SLIDE_SIDE)
        _integer(self.scale_numerator, 1, MAX_SLIDE_SIDE)
        _integer(self.scale_denominator, 1, MAX_SLIDE_SIDE)
        _require(self.scale_numerator >= self.scale_denominator)
        _require(math.gcd(self.scale_numerator, self.scale_denominator) == 1)
        _require((self.index == 0) == (self.scale_numerator == self.scale_denominator))


@dataclass(frozen=True, slots=True, kw_only=True)
class TilePolicy:
    """Explicit geometry and declared concurrent working-buffer ceiling.

    Args:
        width: Maximum native tile width, at most 8192 pixels.
        height: Maximum native tile height, at most 8192 pixels.
        overlap_x: Horizontal overlap; must be smaller than width.
        overlap_y: Vertical overlap; must be smaller than height.
        working_bytes_per_pixel: All simultaneous decoded/input/output/scratch
            bytes per pixel required by the caller's tile adapter.
        max_inflight_tiles: Maximum number of concurrently resident tile buffers.
        budget_bytes: Ceiling for reserved bytes plus admitted working buffers.
        reserved_bytes: Model, decoder and other caller-accounted memory reserve.
        min_tissue_percent: Minimum known tissue-cell percentage to retain a tile.
            Unknown or unsampled coverage is always retained for review.
    """

    width: int
    height: int
    overlap_x: int
    overlap_y: int
    working_bytes_per_pixel: int
    max_inflight_tiles: int
    budget_bytes: int
    reserved_bytes: int
    min_tissue_percent: int

    def __post_init__(self) -> None:
        _integer(self.width, 1, 8192)
        _integer(self.height, 1, 8192)
        _integer(self.overlap_x, 0, self.width - 1)
        _integer(self.overlap_y, 0, self.height - 1)
        _integer(self.working_bytes_per_pixel, 1, 4096)
        _integer(self.max_inflight_tiles, 1, MAX_TILES)
        _integer(self.budget_bytes, 1, MAX_MEMORY_BYTES)
        _integer(self.reserved_bytes, 0, self.budget_bytes - 1)
        _integer(self.min_tissue_percent, 0, 100)


@dataclass(frozen=True, slots=True, kw_only=True)
class TissueMask:
    """Protected coarse full-slide mask; 0=background, 1=tissue candidate, 2=unknown.

    Args:
        source_digest: Digest of the caller-owned slide this mask represents.
        width: Mask columns, at most 512.
        height: Mask rows, at most 512.
        cells: Row-major bytes. No RGB image or mask is included in plan reports.

    Cells uniformly partition the full level-zero extent. Caller-provided masks
    must follow that convention and preserve unknown classifications as 2.
    """

    source_digest: str
    width: int
    height: int
    cells: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _digest(self.source_digest)
        _integer(self.width, 1, MAX_MASK_SIDE)
        _integer(self.height, 1, MAX_MASK_SIDE)
        _require(
            type(self.cells) is bytes and len(self.cells) == self.width * self.height
        )
        _require(all(value in (0, 1, 2) for value in self.cells))

    @property
    def digest(self) -> str:
        """Bind coverage to mask geometry, classes and the caller's slide digest."""
        return _hash(
            [
                self.source_digest,
                self.width,
                self.height,
                hashlib.sha256(self.cells).hexdigest(),
            ]
        )


def mask_from_rgb(
    source_digest: str,
    width: int,
    height: int,
    data: bytes,
    *,
    white_floor: int,
    min_chroma: int,
) -> TissueMask:
    """Classify a bounded caller-owned RGB thumbnail without decoding a slide.

    Args:
        source_digest: Digest of the source slide, assigned by its owner.
        width: Thumbnail width, at most 512.
        height: Thumbnail height, at most 512.
        data: Protected RGB bytes, row-major with no padding or metadata.
        white_floor: All three channels must meet this threshold for background.
        min_chroma: Minimum max-minus-min channel difference for a tissue candidate.

    Returns:
        A coarse mask. Nonwhite, low-chroma pixels remain unknown, so dark ink or
        uncertain regions are not silently discarded. Thresholds require task
        qualification; this is a color heuristic, not tissue identification.
    """
    _digest(source_digest)
    _integer(width, 1, MAX_MASK_SIDE)
    _integer(height, 1, MAX_MASK_SIDE)
    _integer(white_floor, 1, 255)
    _integer(min_chroma, 1, 255)
    _require(type(data) is bytes and len(data) == width * height * 3)
    cells = bytearray()
    for offset in range(0, len(data), 3):
        channels = data[offset : offset + 3]
        cells.append(
            0
            if min(channels) >= white_floor
            else 1
            if max(channels) - min(channels) >= min_chroma
            else 2
        )
    return TissueMask(
        source_digest=source_digest, width=width, height=height, cells=bytes(cells)
    )


@dataclass(frozen=True, slots=True)
class SlideTile:
    """One immutable tile, including auditable exclusions and mask uncertainty."""

    tile_ref: str
    level: int
    bounds: tuple[int, int, int, int]
    source_bounds: tuple[int, int, int, int]
    working_bytes: int
    sampled_cells: int
    tissue_cells: int
    unknown_cells: int
    decision: TileDecision


@dataclass(frozen=True, slots=True)
class SlideTilePlan:
    """Coordinates and metadata only; no slide, thumbnail, mask or tile payload."""

    source_digest: str
    source_width: int
    source_height: int
    levels: tuple[PyramidLevel, ...]
    policy: TilePolicy
    mask_digest: str | None
    tiles: tuple[SlideTile, ...]
    peak_working_bytes: int

    @property
    def selected_tiles(self) -> tuple[SlideTile, ...]:
        """Return kept and review-required tiles in native level/row/column order."""
        return tuple(
            tile for tile in self.tiles if tile.decision is not TileDecision.SKIP
        )

    def iter_batches(self) -> Iterator[tuple[SlideTile, ...]]:
        """Yield ordered batches within the declared concurrent tile limit.

        Adapters must release a batch's buffers before allocating the next one.
        This iterator contains metadata only and cannot enforce adapter behavior.
        """
        selected = self.selected_tiles
        count = self.policy.max_inflight_tiles
        for start in range(0, len(selected), count):
            yield selected[start : start + count]

    def to_dict(self) -> dict[str, Any]:
        """Export stable coordinates, digests, declared budgets and mask counts."""
        return {
            "schema_version": SLIDE_TILE_SCHEMA_VERSION,
            "source_digest": self.source_digest,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "levels": [asdict(level) for level in self.levels],
            "source_edge_gaps": [
                {
                    "level": level.index,
                    "right_pixels": max(
                        0,
                        self.source_width
                        - _ceil_div(
                            level.width * level.scale_numerator, level.scale_denominator
                        ),
                    ),
                    "bottom_pixels": max(
                        0,
                        self.source_height
                        - _ceil_div(
                            level.height * level.scale_numerator,
                            level.scale_denominator,
                        ),
                    ),
                }
                for level in self.levels
            ],
            "policy": asdict(self.policy),
            "mask_digest": self.mask_digest,
            "tiles": [asdict(tile) for tile in self.tiles],
            "selected_count": len(self.selected_tiles),
            "peak_working_bytes": self.peak_working_bytes,
            "review_required": True,
        }

    def to_json(self) -> str:
        """Return deterministic JSON without writing any source or tile to disk."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        """Digest the complete coordinate/quality plan, not decoded image pixels."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


class _MaskIndex:
    def __init__(self, mask: TissueMask) -> None:
        self.width, self.height = mask.width, mask.height
        self.stride = mask.width + 1
        size = self.stride * (mask.height + 1)
        # At most two bounded (513 x 513) native-integer prefix arrays.
        self.tissue = array("I", [0]) * size
        self.unknown = array("I", [0]) * size
        for y in range(mask.height):
            tissue = unknown = 0
            for x in range(mask.width):
                value = mask.cells[y * mask.width + x]
                tissue += value == 1
                unknown += value == 2
                i = (y + 1) * self.stride + x + 1
                self.tissue[i] = self.tissue[i - self.stride] + tissue
                self.unknown[i] = self.unknown[i - self.stride] + unknown

    def counts(
        self, bounds: tuple[int, int, int, int], width: int, height: int
    ) -> tuple[int, int, int]:
        x0, y0, x1, y1 = bounds
        # Select cell centers in the half-open source rectangle using integers.
        left = max(0, _ceil_div(2 * x0 * self.width - width, 2 * width))
        right = min(self.width, _ceil_div(2 * x1 * self.width - width, 2 * width))
        top = max(0, _ceil_div(2 * y0 * self.height - height, 2 * height))
        bottom = min(self.height, _ceil_div(2 * y1 * self.height - height, 2 * height))

        def total(values: array[int]) -> int:
            a, b = top * self.stride, bottom * self.stride
            return (
                values[b + right]
                - values[b + left]
                - values[a + right]
                + values[a + left]
            )

        return (right - left) * (bottom - top), total(self.tissue), total(self.unknown)


def plan_slide_tiles(
    source_digest: str,
    source_width: int,
    source_height: int,
    levels: tuple[PyramidLevel, ...],
    policy: TilePolicy,
    *,
    tissue_mask: TissueMask | None = None,
) -> SlideTilePlan:
    """Plan bounded native-level tiles without opening or decoding a source slide.

    Args:
        source_digest: Digest binding the caller-owned slide to this plan.
        source_width: Full level-zero width.
        source_height: Full level-zero height.
        levels: Requested native levels with explicit scale and geometry.
        policy: Overlap, filtering and declared working-buffer limits.
        tissue_mask: Optional coarse full-slide mask; missing or unknown coverage
            retains tiles for review rather than assuming background.

    Returns:
        A complete plan, including skipped tiles and any source edge gaps.

    Raises:
        SlideTileError: Invalid metadata, more than 4096 candidate tiles, a mask
            for another source, or declared concurrent buffers exceeding budget.
    """
    _digest(source_digest)
    _integer(source_width, 1, MAX_SLIDE_SIDE)
    _integer(source_height, 1, MAX_SLIDE_SIDE)
    _require(type(policy) is TilePolicy)
    _require(type(levels) is tuple and 0 < len(levels) <= MAX_LEVELS)
    _require(all(type(level) is PyramidLevel for level in levels))
    _require(len({level.index for level in levels}) == len(levels))
    ordered = tuple(sorted(levels, key=lambda level: level.index))
    grids = []
    candidate_count = 0
    sx, sy = policy.width - policy.overlap_x, policy.height - policy.overlap_y
    for level in ordered:
        n, d = level.scale_numerator, level.scale_denominator
        _require(abs(level.width * n - source_width * d) < n)
        _require(abs(level.height * n - source_height * d) < n)
        columns = 1 + _ceil_div(max(0, level.width - policy.width), sx)
        rows = 1 + _ceil_div(max(0, level.height - policy.height), sy)
        candidate_count += columns * rows
        _require(candidate_count <= MAX_TILES)
        grids.append((level, columns, rows))
    for left, right in zip(ordered, ordered[1:]):
        _require(left.width >= right.width and left.height >= right.height)
        _require(
            left.scale_numerator * right.scale_denominator
            < right.scale_numerator * left.scale_denominator
        )
    if tissue_mask is not None:
        _require(
            type(tissue_mask) is TissueMask
            and tissue_mask.source_digest == source_digest
        )
    index = _MaskIndex(tissue_mask) if tissue_mask is not None else None
    tiles = []
    max_working = selected_count = 0
    for level, columns, rows in grids:
        n, d = level.scale_numerator, level.scale_denominator
        for row in range(rows):
            for column in range(columns):
                x, y = column * sx, row * sy
                end_x, end_y = (
                    min(level.width, x + policy.width),
                    min(level.height, y + policy.height),
                )
                bounds = (x, y, end_x, end_y)
                source_bounds = (
                    x * n // d,
                    y * n // d,
                    min(source_width, _ceil_div(end_x * n, d)),
                    min(source_height, _ceil_div(end_y * n, d)),
                )
                sampled, tissue, unknown = (
                    index.counts(source_bounds, source_width, source_height)
                    if index
                    else (0, 0, 0)
                )
                decision = (
                    TileDecision.REVIEW
                    if unknown or not sampled
                    else TileDecision.KEEP
                    if tissue * 100 >= sampled * policy.min_tissue_percent
                    else TileDecision.SKIP
                )
                working = (end_x - x) * (end_y - y) * policy.working_bytes_per_pixel
                if decision is not TileDecision.SKIP:
                    max_working = max(max_working, working)
                    selected_count += 1
                tiles.append(
                    SlideTile(
                        _hash([source_digest, asdict(level), bounds]),
                        level.index,
                        bounds,
                        source_bounds,
                        working,
                        sampled,
                        tissue,
                        unknown,
                        decision,
                    )
                )
    peak = policy.reserved_bytes + max_working * min(
        policy.max_inflight_tiles, selected_count
    )
    _require(peak <= policy.budget_bytes)
    return SlideTilePlan(
        source_digest,
        source_width,
        source_height,
        ordered,
        policy,
        tissue_mask.digest if tissue_mask else None,
        tuple(tiles),
        peak,
    )
