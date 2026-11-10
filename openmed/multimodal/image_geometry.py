"""Allocation-safe geometry derived from validated image dimensions.

Preflight code repeatedly needs the same derived numbers after a bounded
header read: a checked pixel count, a reduced aspect ratio, a portrait or
landscape class, and sometimes an allocation-sized memory estimate. This
module computes exactly those numbers with overflow checks before any
allocation-relevant value is produced, and nothing else. It never reads,
decodes, resizes, or allocates an image, and it carries no paths, formats,
or metadata -- only dimensions and derived arithmetic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .asset_manifest import MAX_MANIFEST_BYTE_SIZE, MAX_MANIFEST_COUNT

__all__ = [
    "DEFAULT_MAX_IMAGE_PIXELS",
    "IMAGE_GEOMETRY_SCHEMA_VERSION",
    "MAX_IMAGE_DIMENSION",
    "ImageGeometry",
    "ImageGeometryError",
    "Orientation",
    "derive_image_geometry",
]

MAX_IMAGE_DIMENSION: Final[int] = MAX_MANIFEST_COUNT
DEFAULT_MAX_IMAGE_PIXELS: Final[int] = 100_000_000
IMAGE_GEOMETRY_SCHEMA_VERSION: Final = 1
_MAX_BYTES_PER_PIXEL: Final[int] = 1024


class ImageGeometryError(ValueError):
    """Value-free failure for invalid dimensions or geometry arithmetic."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class Orientation(str, Enum):
    """Closed set of orientation classes for validated dimensions."""

    SQUARE = "square"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


def _dimension(value: Any, name: str) -> int:
    if value is None:
        raise ImageGeometryError(f"image_geometry_{name}_missing")
    if type(value) is bool:
        raise ImageGeometryError(f"image_geometry_{name}_boolean")
    if type(value) is float and not math.isfinite(value):
        raise ImageGeometryError(f"image_geometry_{name}_not_finite")
    if type(value) is not int:
        raise ImageGeometryError(f"image_geometry_{name}_not_integer")
    if value <= 0:
        raise ImageGeometryError(f"image_geometry_{name}_not_positive")
    if value > MAX_IMAGE_DIMENSION:
        raise ImageGeometryError(f"image_geometry_{name}_overflow")
    return value


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    """Dimensions plus derived numbers, with no content or metadata.

    Every field is a validated dimension or a number derived from them by
    overflow-checked arithmetic; results never carry paths, formats,
    identifiers, or pixel values.
    """

    schema_version: int
    width: int
    height: int
    pixel_count: int
    aspect_ratio_width: int
    aspect_ratio_height: int
    orientation: Orientation
    bytes_per_pixel: int | None = None
    estimated_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "width": self.width,
            "height": self.height,
            "pixel_count": self.pixel_count,
            "aspect_ratio_width": self.aspect_ratio_width,
            "aspect_ratio_height": self.aspect_ratio_height,
            "orientation": self.orientation.value,
        }
        if self.bytes_per_pixel is not None:
            data["bytes_per_pixel"] = self.bytes_per_pixel
        if self.estimated_bytes is not None:
            data["estimated_bytes"] = self.estimated_bytes
        return data

    def to_json(self) -> str:
        """Serialize with deterministic key order and no insignificant space."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=False,
            separators=(",", ":"),
        )


def derive_image_geometry(
    width: Any,
    height: Any,
    *,
    bytes_per_pixel: Any = None,
    max_pixels: Any = DEFAULT_MAX_IMAGE_PIXELS,
) -> ImageGeometry:
    """Derive checked pixel count, reduced ratio, class, and memory estimate.

    Inputs are validated dimension values (as produced by the bounded header
    preflight helpers) or anything the caller wishes to have fail closed. The
    pixel product and the byte estimate are both checked before the
    allocation-sized values are produced, so a caller can never receive a
    number that exceeds the declared limits. No image is read, decoded, or
    allocated, and EXIF orientation is out of scope.
    """
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")
    if max_pixels > MAX_MANIFEST_BYTE_SIZE:
        raise ValueError("max_pixels is out of range")
    checked_width = _dimension(width, "width")
    checked_height = _dimension(height, "height")
    checked_bytes_per_pixel: int | None
    if bytes_per_pixel is not None:
        if (
            type(bytes_per_pixel) is not int
            or not 0 < bytes_per_pixel <= _MAX_BYTES_PER_PIXEL
        ):
            raise ValueError("bytes_per_pixel must be a positive bounded integer")
        checked_bytes_per_pixel = bytes_per_pixel
    else:
        checked_bytes_per_pixel = None

    # Division check instead of multiplication: width*height can exceed any
    # machine-relevant range, and the product must never be materialized
    # before the limit is known to hold.
    if checked_width > max_pixels // checked_height:
        raise ImageGeometryError("image_geometry_pixel_limit_exceeded")
    pixel_count = checked_width * checked_height

    estimated_bytes: int | None = None
    if checked_bytes_per_pixel is not None:
        if checked_bytes_per_pixel > MAX_MANIFEST_BYTE_SIZE // pixel_count:
            raise ImageGeometryError("image_geometry_byte_estimate_overflow")
        estimated_bytes = pixel_count * checked_bytes_per_pixel

    divisor = math.gcd(checked_width, checked_height)
    if checked_width == checked_height:
        orientation = Orientation.SQUARE
    elif checked_width < checked_height:
        orientation = Orientation.PORTRAIT
    else:
        orientation = Orientation.LANDSCAPE
    return ImageGeometry(
        schema_version=IMAGE_GEOMETRY_SCHEMA_VERSION,
        width=checked_width,
        height=checked_height,
        pixel_count=pixel_count,
        aspect_ratio_width=checked_width // divisor,
        aspect_ratio_height=checked_height // divisor,
        orientation=orientation,
        bytes_per_pixel=checked_bytes_per_pixel,
        estimated_bytes=estimated_bytes,
    )
