"""Declarative generalization hierarchies for tabular quasi-identifiers.

Exposes a monotone, versioned family of generalization levels per column type
(age, ZIP/postcode, date, and caller-populated clinical codes) plus the
:func:`generalize_value` leaf transform and level introspection helpers. The
ordering is the stable typed contract a lattice search composes to reach a
target privacy policy.
"""

from .functions import (
    AGE_BAND_WIDTHS,
    AGE_MAX,
    AGE_MIN_BAND_WIDTH,
    AGE_TOP_BAND,
    AGE_TOP_THRESHOLD,
    COLUMN_TYPE_AGE,
    COLUMN_TYPE_CLINICAL_CODE,
    COLUMN_TYPE_DATE,
    COLUMN_TYPE_ZIP,
    HIERARCHY_SCHEMA_VERSION,
    SUPPORTED_COLUMN_TYPES,
    SUPPRESSED,
    ZIP_MAX_TRUNCATION,
    GeneralizationLevel,
    Hierarchy,
    HierarchyError,
    build_enforcement_hierarchies,
    describe_level,
    generalize_value,
    get_hierarchy,
    max_level,
    to_enforce_kanon_hierarchy,
)

__all__ = [
    "AGE_BAND_WIDTHS",
    "AGE_MAX",
    "AGE_MIN_BAND_WIDTH",
    "AGE_TOP_BAND",
    "AGE_TOP_THRESHOLD",
    "COLUMN_TYPE_AGE",
    "COLUMN_TYPE_CLINICAL_CODE",
    "COLUMN_TYPE_DATE",
    "COLUMN_TYPE_ZIP",
    "GeneralizationLevel",
    "HIERARCHY_SCHEMA_VERSION",
    "Hierarchy",
    "HierarchyError",
    "SUPPORTED_COLUMN_TYPES",
    "SUPPRESSED",
    "ZIP_MAX_TRUNCATION",
    "build_enforcement_hierarchies",
    "describe_level",
    "generalize_value",
    "get_hierarchy",
    "max_level",
    "to_enforce_kanon_hierarchy",
]
