from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ValidationFinding:
    """A deterministic, privacy-safe finding from manifest validation."""

    field_name: str
    reason_code: str


@dataclass(frozen=True)
class ManifestProfile:
    """A versioned metadata profile for a specific modality."""

    modality: str
    version: str
    required_fields: frozenset[str] = frozenset()
    optional_fields: frozenset[str] = frozenset()
    inapplicable_fields: frozenset[str] = frozenset()


IMAGE_V1 = ManifestProfile(
    modality="image",
    version="1.0",
    required_fields=frozenset(["width", "height"]),
    optional_fields=frozenset(),
    inapplicable_fields=frozenset(["page_count", "frame_count", "duration"]),
)

PDF_V1 = ManifestProfile(
    modality="pdf",
    version="1.0",
    required_fields=frozenset(["page_count"]),
    optional_fields=frozenset(),
    inapplicable_fields=frozenset(["width", "height", "frame_count", "duration"]),
)

DICOM_V1 = ManifestProfile(
    modality="dicom",
    version="1.0",
    required_fields=frozenset(["frame_count", "width", "height"]),
    optional_fields=frozenset(),
    inapplicable_fields=frozenset(["page_count", "duration"]),
)

AUDIO_V1 = ManifestProfile(
    modality="audio",
    version="1.0",
    required_fields=frozenset(["duration"]),
    optional_fields=frozenset(),
    inapplicable_fields=frozenset(["width", "height", "page_count", "frame_count"]),
)


def validate_manifest_metadata(
    profile: ManifestProfile, manifest: Mapping[str, Any]
) -> Sequence[ValidationFinding]:
    """Validate manifest metadata deterministically against a profile.

    This operates ONLY on the dictionary of metadata fields.
    It never opens files or inspects source paths.
    """
    findings: list[ValidationFinding] = []

    # Sort fields for deterministic output order
    all_fields = sorted(
        profile.required_fields | profile.optional_fields | profile.inapplicable_fields
    )

    for field_name in all_fields:
        if field_name in profile.inapplicable_fields:
            if field_name in manifest:
                findings.append(ValidationFinding(field_name, "inapplicable_present"))
            continue

        if field_name not in manifest:
            if field_name in profile.required_fields:
                findings.append(ValidationFinding(field_name, "missing_required"))
            continue

        value = manifest[field_name]

        # Numeric and boolean checks
        if isinstance(value, bool):
            findings.append(ValidationFinding(field_name, "invalid_boolean"))
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                findings.append(ValidationFinding(field_name, "non_finite_numeric"))
            elif value == 0:
                findings.append(ValidationFinding(field_name, "invalid_zero"))
        else:
            findings.append(ValidationFinding(field_name, "invalid_type"))

    return findings
