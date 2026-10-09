"""Modality-specific validation for privacy-safe asset manifest metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final

from .asset_manifest import (
    MAX_MANIFEST_COUNT,
    MAX_MANIFEST_DURATION_SECONDS,
    AssetManifest,
)

__all__ = [
    "AUDIO_V1",
    "DICOM_V1",
    "DICOM_NON_PIXEL_V1",
    "IMAGE_V1",
    "PDF_V1",
    "WAVEFORM_V1",
    "ManifestProfile",
    "ManifestProfileError",
    "ValidationFinding",
    "validate_manifest_metadata",
]

_LEGACY_FIELDS: Final = frozenset(
    {"pages", "width", "height", "frames", "duration_seconds"}
)
_WAVEFORM_FIELDS: Final = frozenset(
    {"channel_count", "sample_rate_hz", "samples_per_channel"}
)
_MANIFEST_FIELDS: Final = _LEGACY_FIELDS | _WAVEFORM_FIELDS
_INTEGER_FIELDS: Final = frozenset(
    {"pages", "width", "height", "frames", "channel_count", "samples_per_channel"}
)
_MODALITIES: Final = frozenset(
    {"image", "pdf", "dicom", "audio", "waveform", "dicom_non_pixel"}
)
_REASON_CODES: Final = frozenset(
    {
        "inapplicable_present",
        "invalid_boolean",
        "invalid_type",
        "invalid_zero",
        "missing_required",
        "non_finite_numeric",
        "out_of_range",
        "sample_count_mismatch",
    }
)


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
        if type(self.version) is not str or self.version not in {"1.0", "1.1"}:
            raise ManifestProfileError("profile version is unsupported")
        if self.modality in {"waveform", "dicom_non_pixel"} and self.version != "1.1":
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
        expected = _LEGACY_FIELDS if self.version == "1.0" else _MANIFEST_FIELDS
        if declared != expected:
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

WAVEFORM_V1 = ManifestProfile(
    modality="waveform",
    version="1.1",
    required_fields=_WAVEFORM_FIELDS | {"duration_seconds"},
    inapplicable_fields=frozenset({"pages", "width", "height", "frames"}),
)

DICOM_NON_PIXEL_V1 = ManifestProfile(
    modality="dicom_non_pixel",
    version="1.1",
    inapplicable_fields=_MANIFEST_FIELDS,
)


def validate_manifest_metadata(
    profile: ManifestProfile, manifest: Mapping[str, Any] | AssetManifest
) -> list[ValidationFinding]:
    """Validate metadata deterministically without opening or decoding an asset.

    The mapping is copied before validation. Unknown structural manifest fields
    are intentionally left to the asset-manifest validator; this function reads
    only fixed, metadata-only profile fields.
    """

    if not isinstance(profile, ManifestProfile):
        raise TypeError("profile must be a ManifestProfile")
    if isinstance(manifest, AssetManifest):
        fields = manifest.to_dict()
    elif not isinstance(manifest, Mapping):
        raise TypeError("manifest metadata must be a mapping or AssetManifest")
    else:
        try:
            fields = dict(manifest)
        except Exception:
            raise ManifestProfileError("manifest metadata could not be read") from None

    findings: list[ValidationFinding] = []
    declared_fields = sorted(_MANIFEST_FIELDS)
    for field_name in declared_fields:
        if field_name in profile.inapplicable_fields or (
            profile.version == "1.0" and field_name in _WAVEFORM_FIELDS
        ):
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
    if (
        profile.modality == "waveform"
        and not findings
        and {"duration_seconds", "sample_rate_hz", "samples_per_channel"}
        <= fields.keys()
    ):
        expected_samples = Fraction(fields["duration_seconds"]) * Fraction(
            fields["sample_rate_hz"]
        )
        # One sample of tolerance covers fractional endpoint conventions;
        # larger differences cannot silently understate the declared workload.
        if abs(Fraction(fields["samples_per_channel"]) - expected_samples) > 1:
            findings.append(
                ValidationFinding("samples_per_channel", "sample_count_mismatch")
            )
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
        if not 0 < value <= MAX_MANIFEST_COUNT:
            return "out_of_range"
        return None
    if type(value) not in (int, float):
        return "invalid_type"
    if type(value) is float and not math.isfinite(value):
        return "non_finite_numeric"
    if value == 0:
        return "invalid_zero"
    if not 0 < value <= MAX_MANIFEST_DURATION_SECONDS:
        return "out_of_range"
    return None
