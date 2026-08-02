"""Local-first, version-aware FHIR exchange helpers."""

from __future__ import annotations

from .profiles import (
    PROFILE_MATRIX,
    PROFILE_MATRIX_PATH,
    SUPPORTED_PROFILE_MATRIX,
    get_profile,
    profile_matrix,
    validate_profile_matrix,
)
from .validation import (
    FHIRValidationResult,
    validate,
    validate_bundle,
    validate_document,
    validate_resource,
    validation_result,
)
from .versions import (
    FHIR_R4,
    FHIR_R5,
    SUPPORTED_RESOURCE_TYPES,
    FHIRConversionError,
    FHIRVersion,
    FHIRVersionAdapter,
    FHIRVersionError,
    UnsupportedFHIRField,
    UnsupportedFHIRFieldError,
    VersionAdapter,
    convert_bundle,
    convert_resource,
    parse_fhir_version,
    r4_to_r5,
    r5_to_r4,
)

__all__ = [
    "FHIRConversionError",
    "FHIRValidationResult",
    "FHIRVersion",
    "FHIR_R4",
    "FHIR_R5",
    "FHIRVersionAdapter",
    "FHIRVersionError",
    "PROFILE_MATRIX_PATH",
    "PROFILE_MATRIX",
    "SUPPORTED_PROFILE_MATRIX",
    "SUPPORTED_RESOURCE_TYPES",
    "UnsupportedFHIRFieldError",
    "UnsupportedFHIRField",
    "VersionAdapter",
    "convert_bundle",
    "convert_resource",
    "get_profile",
    "parse_fhir_version",
    "r4_to_r5",
    "r5_to_r4",
    "profile_matrix",
    "validate",
    "validate_bundle",
    "validate_document",
    "validate_profile_matrix",
    "validate_resource",
    "validation_result",
]
