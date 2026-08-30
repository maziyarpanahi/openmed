"""Modality-specific validation for privacy-safe asset manifest metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "AUDIO_V1",
    "DICOM_V1",
    "IMAGE_V1",
    "PDF_V1",
    "ManifestProfile",
    "ManifestProfileError",
    "ValidationFinding",
    "validate_manifest_metadata",
]

_MANIFEST_FIELDS: Final = frozenset(
    {"pages", "width", "height", "frames", "duration_seconds"}
)
_INTEGER_FIELDS: Final = frozenset({"pages", "width", "height", "frames"})
_MODALITIES: Final = frozenset({"image", "pdf", "dicom", "audio"})
_REASON_CODES: Final = frozenset(
    {
        "inapplicable_present",
        "invalid_boolean",
        "invalid_type",
        "invalid_zero",
        "missing_required",
        "non_finite_numeric",
        "out_of_range",
    }
)
_MAX_COUNT: Final = (1 << 31) - 1
_MAX_DURATION_SECONDS: Final = float(_MAX_COUNT)


class ManifestProfileError(ValueError):
    """Raised when a manifest profile or metadata mapping is invalid."""


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """A deterministic, privacy-safe finding from manifest validation."""

    field_name: str
    reason_code: str

    def __post_init__(self) -> None:
        if type(self.field_name) is not str or self.field_name not in _MANIFEST_FIELDS:
            raise ManifestProfileError("finding field_name is unsupported")
        if type(self.reason_code) is not str or self.reason_code not in _REASON_CODES:
            raise ManifestProfileError("finding reason_code is unsupported")


@dataclass(frozen=True, slots=True)
class ManifestProfile:
    """A versioned metadata profile for a specific modality."""

    modality: str
    version: str
    required_fields: frozenset[str] = frozenset()
    optional_fields: frozenset[str] = frozenset()
    inapplicable_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if type(self.modality) is not str or self.modality not in _MODALITIES:
            raise ManifestProfileError("profile modality is unsupported")
        if type(self.version) is not str or self.version != "1.0":
            raise ManifestProfileError("profile version is unsupported")
        groups = (
            self.required_fields,
            self.optional_fields,
            self.inapplicable_fields,
        )
        if any(type(group) is not frozenset for group in groups):
            raise ManifestProfileError("profile fields must be frozen sets")
        declared = (
            self.required_fields | self.optional_fields | self.inapplicable_fields
        )
        if declared != _MANIFEST_FIELDS:
            raise ManifestProfileError("profile must classify every manifest field")
        pairs = ((groups[0], groups[1]), (groups[0], groups[2]), (groups[1], groups[2]))
        if any(left & right for left, right in pairs):
            raise ManifestProfileError("profile field groups must be disjoint")


IMAGE_V1 = ManifestProfile(
    modality="image",
    version="1.0",
    required_fields=frozenset({"width", "height"}),
    inapplicable_fields=frozenset({"pages", "frames", "duration_seconds"}),
)

PDF_V1 = ManifestProfile(
    modality="pdf",
    version="1.0",
    required_fields=frozenset({"pages"}),
    inapplicable_fields=frozenset({"width", "height", "frames", "duration_seconds"}),
)

DICOM_V1 = ManifestProfile(
    modality="dicom",
    version="1.0",
    required_fields=frozenset({"frames", "width", "height"}),
    inapplicable_fields=frozenset({"pages", "duration_seconds"}),
)

AUDIO_V1 = ManifestProfile(
    modality="audio",
    version="1.0",
    required_fields=frozenset({"duration_seconds"}),
    inapplicable_fields=frozenset({"width", "height", "pages", "frames"}),
)


def validate_manifest_metadata(
    profile: ManifestProfile, manifest: Mapping[str, Any]
) -> list[ValidationFinding]:
    """Validate metadata deterministically without opening or decoding an asset.

    The mapping is copied before validation. Unknown structural manifest fields
    are intentionally left to the asset-manifest validator; this function reads
    only the five fixed, metadata-only profile fields.
    """

    if not isinstance(profile, ManifestProfile):
        raise TypeError("profile must be a ManifestProfile")
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest metadata must be a mapping")
    try:
        fields = dict(manifest)
    except Exception:
        raise ManifestProfileError("manifest metadata could not be read") from None

    findings: list[ValidationFinding] = []
    declared_fields = sorted(
        profile.required_fields | profile.optional_fields | profile.inapplicable_fields
    )
    for field_name in declared_fields:
        if field_name in profile.inapplicable_fields:
            if field_name in fields:
                findings.append(ValidationFinding(field_name, "inapplicable_present"))
            continue
        if field_name not in fields:
            if field_name in profile.required_fields:
                findings.append(ValidationFinding(field_name, "missing_required"))
            continue
        reason = _numeric_reason(field_name, fields[field_name])
        if reason is not None:
            findings.append(ValidationFinding(field_name, reason))
    return findings


def _numeric_reason(field_name: str, value: Any) -> str | None:
    if type(value) is bool:
        return "invalid_boolean"
    if field_name in _INTEGER_FIELDS:
        if type(value) is float and not math.isfinite(value):
            return "non_finite_numeric"
        if type(value) is not int:
            return "invalid_type"
        if value == 0:
            return "invalid_zero"
        if not 0 < value <= _MAX_COUNT:
            return "out_of_range"
        return None
    if type(value) not in (int, float):
        return "invalid_type"
    if not math.isfinite(value):
        return "non_finite_numeric"
    if value == 0:
        return "invalid_zero"
    if not 0 < value <= _MAX_DURATION_SECONDS:
        return "out_of_range"
    return None
