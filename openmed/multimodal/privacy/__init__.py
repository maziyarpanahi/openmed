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
]
