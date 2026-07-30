"""Declarative generalization hierarchies for tabular quasi-identifiers.

Exposes a monotone, versioned family of generalization levels per column type
(age, ZIP/postcode, date) plus the :func:`generalize_value` leaf transform and
level introspection helpers. The ordering is the stable typed contract a
sibling lattice search composes to reach a target k-anonymity.
"""

from .functions import (
    AGE_BAND_WIDTHS,
    AGE_MAX,
    AGE_MIN_BAND_WIDTH,
    AGE_TOP_BAND,
    AGE_TOP_THRESHOLD,
    COLUMN_TYPE_AGE,
    COLUMN_TYPE_DATE,
    COLUMN_TYPE_ZIP,
    HIERARCHY_SCHEMA_VERSION,
    SUPPORTED_COLUMN_TYPES,
    SUPPRESSED,
    ZIP_MAX_TRUNCATION,
    GeneralizationLevel,
    Hierarchy,
    HierarchyError,
    describe_level,
    generalize_value,
    get_hierarchy,
    max_level,
)

__all__ = [
    "AGE_BAND_WIDTHS",
    "AGE_MAX",
    "AGE_MIN_BAND_WIDTH",
    "AGE_TOP_BAND",
    "AGE_TOP_THRESHOLD",
    "COLUMN_TYPE_AGE",
    "COLUMN_TYPE_DATE",
    "COLUMN_TYPE_ZIP",
    "GeneralizationLevel",
    "HIERARCHY_SCHEMA_VERSION",
    "Hierarchy",
    "HierarchyError",
    "SUPPORTED_COLUMN_TYPES",
    "SUPPRESSED",
    "ZIP_MAX_TRUNCATION",
    "describe_level",
    "generalize_value",
    "get_hierarchy",
    "max_level",
]
