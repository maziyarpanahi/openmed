"""Coordinate lineage for protected multimodal regions."""

from .mask_lineage import (
    MASK_LINEAGE_SCHEMA_VERSION,
    Affine,
    CoordinateSpace,
    MaskFailure,
    MaskLineageError,
    MaskLineageResult,
    MaskProjection,
    Rectangle,
    TransformStep,
    propagate_masks,
)
from .pixel_coverage import (
    LocalPixelDetector,
    PixelCoverageError,
    PixelCoverageLimits,
    PixelCoverageResult,
    PixelFrame,
    PixelFramePlan,
    PixelInspection,
    audit_pixel_redaction,
    inventory_pixel_frames,
)

__all__ = [
    "MASK_LINEAGE_SCHEMA_VERSION",
    "Affine",
    "CoordinateSpace",
    "MaskFailure",
    "MaskLineageError",
    "MaskLineageResult",
    "MaskProjection",
    "Rectangle",
    "TransformStep",
    "propagate_masks",
    "LocalPixelDetector",
    "PixelCoverageError",
    "PixelCoverageLimits",
    "PixelCoverageResult",
    "PixelFrame",
    "PixelFramePlan",
    "PixelInspection",
    "audit_pixel_redaction",
    "inventory_pixel_frames",
]
