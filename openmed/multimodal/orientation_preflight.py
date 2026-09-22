"""Detect conflicts between declared image transforms and orientation metadata.

An image manifest can record a rotation or mirror that disagrees with the
EXIF/TIFF ``Orientation`` tag of the stored pixels. When that happens, region
coordinates silently point at the wrong place after a transform. This module
compares numeric orientation values with the declared transform and optional
dimensions. It never decodes pixels, reads metadata strings, or rotates
anything, and a report carries categorical and numeric fields only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Sequence

ORIENTATION_PREFLIGHT_SCHEMA_VERSION: Final[str] = (
    "openmed.multimodal.orientation_preflight.v1"
)
MAX_ORIENTATION_VALUES: Final[int] = 8
MAX_IMAGE_DIMENSION: Final[int] = 1_000_000

ORIENTATION_REASON_CODES: Final[tuple[str, ...]] = (
    "orientation_value_limit",
    "orientation_value_invalid",
    "dimensions_invalid",
    "orientation_conflict",
    "transform_conflict",
    "transform_without_orientation",
    "dimensions_inconsistent",
    "transform_not_applied",
    "transform_required",
    "orientation_duplicate",
    "orientation_missing",
)

_REPORT_FIELDS = (
    "schema_version",
    "status",
    "reason_codes",
    "orientation",
    "required_transform",
    "declared_transform",
    "oriented_width",
    "oriented_height",
)


class OrientationStatus(str, Enum):
    """Closed verdict vocabulary for one orientation comparison.

    Values:
        ALIGNED: The declared transform agrees with the orientation metadata.
        TRANSFORM_REQUIRED: Metadata requires a transform nobody declared.
        AMBIGUOUS: The inputs disagree and a reviewer must pick one source.
        INVALID: Orientation values or dimensions are malformed.
    """

    ALIGNED = "aligned"
    TRANSFORM_REQUIRED = "transform_required"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


_STATUS_RANK: Final[dict[OrientationStatus, int]] = {
    OrientationStatus.ALIGNED: 0,
    OrientationStatus.TRANSFORM_REQUIRED: 1,
    OrientationStatus.AMBIGUOUS: 2,
    OrientationStatus.INVALID: 3,
}

_REASON_STATUS: Final[dict[str, OrientationStatus]] = {
    "orientation_value_limit": OrientationStatus.INVALID,
    "orientation_value_invalid": OrientationStatus.INVALID,
    "dimensions_invalid": OrientationStatus.INVALID,
    "orientation_conflict": OrientationStatus.AMBIGUOUS,
    "transform_conflict": OrientationStatus.AMBIGUOUS,
    "transform_without_orientation": OrientationStatus.AMBIGUOUS,
    "dimensions_inconsistent": OrientationStatus.AMBIGUOUS,
    "transform_not_applied": OrientationStatus.TRANSFORM_REQUIRED,
    "transform_required": OrientationStatus.TRANSFORM_REQUIRED,
    "orientation_duplicate": OrientationStatus.ALIGNED,
    "orientation_missing": OrientationStatus.ALIGNED,
}


class OrientationPreflightError(ValueError):
    """Value-free failure raised for unusable arguments."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ImageTransform:
    """A mirror-then-rotate transform that displays stored pixels upright.

    The optional horizontal mirror is applied first, then a clockwise rotation.
    This covers exactly the eight EXIF orientations.

    Attributes:
        rotation_degrees: Clockwise rotation, one of 0, 90, 180, or 270.
        mirrored: Whether stored pixels are mirrored horizontally first.
    """

    rotation_degrees: int = 0
    mirrored: bool = False

    def __post_init__(self) -> None:
        if type(self.rotation_degrees) is not int or self.rotation_degrees not in (
            0,
            90,
            180,
            270,
        ):
            raise OrientationPreflightError("transform_rotation_invalid")
        if type(self.mirrored) is not bool:
            raise OrientationPreflightError("transform_mirror_invalid")

    @classmethod
    def from_exif_orientation(cls, orientation: int) -> "ImageTransform":
        """Return the transform that displays an EXIF orientation upright."""
        if type(orientation) is not int or orientation not in _EXIF_TRANSFORMS:
            raise OrientationPreflightError("orientation_value_invalid")
        rotation, mirrored = _EXIF_TRANSFORMS[orientation]
        return cls(rotation_degrees=rotation, mirrored=mirrored)

    @property
    def is_identity(self) -> bool:
        """Return whether the transform leaves pixels unchanged."""
        return self.rotation_degrees == 0 and not self.mirrored

    @property
    def swaps_dimensions(self) -> bool:
        """Return whether width and height trade places."""
        return self.rotation_degrees in (90, 270)

    @property
    def exif_orientation(self) -> int:
        """Return the EXIF orientation value this transform corrects."""
        return _TRANSFORM_EXIF[(self.rotation_degrees, self.mirrored)]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible mapping."""
        return {
            "rotation_degrees": self.rotation_degrees,
            "mirrored": self.mirrored,
        }


_EXIF_TRANSFORMS: Final[dict[int, tuple[int, bool]]] = {
    1: (0, False),
    2: (0, True),
    3: (180, False),
    4: (180, True),
    5: (270, True),
    6: (90, False),
    7: (90, True),
    8: (270, False),
}
_TRANSFORM_EXIF: Final[dict[tuple[int, bool], int]] = {
    value: key for key, value in _EXIF_TRANSFORMS.items()
}


@dataclass(frozen=True, slots=True)
class OrientationReport:
    """Categorical and numeric result of one orientation comparison.

    Attributes:
        status: Most severe verdict implied by ``reason_codes``.
        reason_codes: Findings in :data:`ORIENTATION_REASON_CODES` order.
        orientation: The single agreed EXIF orientation, or ``None`` when it is
            missing, conflicting, or invalid.
        required_transform: Transform implied by ``orientation``, if known.
        declared_transform: Transform the manifest declared, if any.
        oriented_width: Expected upright width, when dimensions were supplied
            and the effective transform is known.
        oriented_height: Expected upright height under the same conditions.
    """

    status: OrientationStatus
    reason_codes: tuple[str, ...]
    orientation: int | None
    required_transform: ImageTransform | None
    declared_transform: ImageTransform | None
    oriented_width: int | None
    oriented_height: int | None

    @property
    def is_aligned(self) -> bool:
        """Return whether the manifest can be used without a transform change."""
        return self.status is OrientationStatus.ALIGNED

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping in declared field order."""
        values = {
            "schema_version": ORIENTATION_PREFLIGHT_SCHEMA_VERSION,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "orientation": self.orientation,
            "required_transform": (
                None
                if self.required_transform is None
                else self.required_transform.to_dict()
            ),
            "declared_transform": (
                None
                if self.declared_transform is None
                else self.declared_transform.to_dict()
            ),
            "oriented_width": self.oriented_width,
            "oriented_height": self.oriented_height,
        }
        return {field: values[field] for field in _REPORT_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def check_image_orientation(
    orientation_values: Sequence[int],
    *,
    declared_transform: ImageTransform | None = None,
    stored_size: tuple[int, int] | None = None,
    declared_size: tuple[int, int] | None = None,
) -> OrientationReport:
    """Compare orientation metadata with a manifest's declared transform.

    Args:
        orientation_values: Every numeric orientation value found in bounded
            metadata, such as EXIF IFD0 and TIFF tags. Empty means missing.
        declared_transform: Transform the manifest says makes the stored
            pixels upright, or ``None`` when the manifest declares nothing.
        stored_size: Optional ``(width, height)`` of the stored pixels.
        declared_size: Optional ``(width, height)`` the manifest records for
            the upright image. Compared only when ``stored_size`` is present.

    Returns:
        An :class:`OrientationReport`.

    Raises:
        OrientationPreflightError: If an argument has the wrong type.
    """
    values = _orientation_values(orientation_values)
    if declared_transform is not None and not isinstance(
        declared_transform, ImageTransform
    ):
        raise OrientationPreflightError("declared_transform_invalid")
    stored = _size(stored_size, "stored_size_invalid")
    declared = _size(declared_size, "declared_size_invalid")

    reasons: set[str] = set()
    orientation: int | None = None
    known_orientation = False

    if values is None:
        reasons.add("orientation_value_limit")
    elif any(value not in _EXIF_TRANSFORMS for value in values):
        reasons.add("orientation_value_invalid")
    elif not values:
        reasons.add("orientation_missing")
    elif len(set(values)) > 1:
        reasons.add("orientation_conflict")
    else:
        orientation = values[0]
        known_orientation = True
        if len(values) > 1:
            reasons.add("orientation_duplicate")

    required = (
        ImageTransform.from_exif_orientation(orientation)
        if orientation is not None
        else None
    )

    if "orientation_missing" in reasons:
        if declared_transform is not None and not declared_transform.is_identity:
            reasons.add("transform_without_orientation")
    elif known_orientation and required is not None:
        if declared_transform is None:
            if not required.is_identity:
                reasons.add("transform_required")
        elif declared_transform != required:
            if declared_transform.is_identity:
                reasons.add("transform_not_applied")
            else:
                reasons.add("transform_conflict")

    effective: ImageTransform | None
    if declared_transform is not None:
        effective = declared_transform
    elif required is not None:
        effective = required
    elif "orientation_missing" in reasons:
        effective = ImageTransform()
    else:
        effective = None

    oriented_width: int | None = None
    oriented_height: int | None = None
    if stored_size is not None and stored is None:
        reasons.add("dimensions_invalid")
    if declared_size is not None and declared is None:
        reasons.add("dimensions_invalid")
    if stored is not None and effective is not None:
        width, height = stored
        if effective.swaps_dimensions:
            width, height = height, width
        oriented_width, oriented_height = width, height
        if declared is not None and declared != (width, height):
            reasons.add("dimensions_inconsistent")

    reason_codes = tuple(code for code in ORIENTATION_REASON_CODES if code in reasons)
    status = OrientationStatus.ALIGNED
    for code in reason_codes:
        candidate = _REASON_STATUS[code]
        if _STATUS_RANK[candidate] > _STATUS_RANK[status]:
            status = candidate

    return OrientationReport(
        status=status,
        reason_codes=reason_codes,
        orientation=orientation,
        required_transform=required,
        declared_transform=declared_transform,
        oriented_width=oriented_width,
        oriented_height=oriented_height,
    )


def _orientation_values(value: Any) -> tuple[int, ...] | None:
    """Return validated values, or ``None`` when the count limit is exceeded."""
    if type(value) not in (list, tuple):
        raise OrientationPreflightError("orientation_values_invalid")
    if len(value) > MAX_ORIENTATION_VALUES:
        return None
    items = tuple(value)
    for item in items:
        if type(item) is not int:
            raise OrientationPreflightError("orientation_values_invalid")
    return items


def _size(value: Any, category: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if type(value) is not tuple or len(value) != 2:
        raise OrientationPreflightError(category)
    width, height = value
    if type(width) is not int or type(height) is not int:
        raise OrientationPreflightError(category)
    if not (0 < width <= MAX_IMAGE_DIMENSION and 0 < height <= MAX_IMAGE_DIMENSION):
        return None
    return width, height


__all__ = [
    "MAX_IMAGE_DIMENSION",
    "MAX_ORIENTATION_VALUES",
    "ORIENTATION_PREFLIGHT_SCHEMA_VERSION",
    "ORIENTATION_REASON_CODES",
    "ImageTransform",
    "OrientationPreflightError",
    "OrientationReport",
    "OrientationStatus",
    "check_image_orientation",
]
