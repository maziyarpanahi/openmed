"""Versioned, local-only contracts for typed clinical summary templates.

Summary profiles describe bounded output fields for a local generator.  They
contain stable field names, field types, cardinality limits, and source-section
metadata; they do not contain a free-form system prompt, source text, model
configuration, or extracted values.  The built-in catalog is deliberately
closed-world so a profile name and version can be audited before generation.

Loading and validation are deterministic and local.  Reports and exceptions
retain only profile metadata, field names, fixed reason codes, and counts, so a
caller can validate a generated payload without copying its sensitive values
into an audit artifact.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

SUMMARY_PROFILE_SCHEMA_VERSION: Final[int] = 1
SUMMARY_TEMPLATE_PROFILE_SCHEMA_VERSION: Final[int] = SUMMARY_PROFILE_SCHEMA_VERSION
SUMMARY_PROFILE_VERSION: Final[str] = "1.0"
CURRENT_SUMMARY_PROFILE_VERSION: Final[str] = SUMMARY_PROFILE_VERSION
SUMMARY_PROFILE_FORMAT: Final[str] = "typed_json"
SUMMARY_PROFILE_DISCLAIMER: Final[str] = (
    "Clinical summary profiles are bounded assistive templates, not clinical "
    "decisions or autonomous care recommendations. Qualified clinical review "
    "is required."
)

MAX_SUMMARY_PROFILE_JSON_BYTES: Final[int] = 64 * 1024
MAX_SUMMARY_PROFILE_FIELDS: Final[int] = 16
MAX_SUMMARY_OUTPUT_FIELDS: Final[int] = 32
MAX_SUMMARY_FIELD_ITEMS: Final[int] = 32
MAX_SUMMARY_FIELD_CHARACTERS: Final[int] = 4_096

SUMMARY_PROFILE_NAMES: Final[tuple[str, ...]] = (
    "brief_hospital_course",
    "clinical_handoff",
    "discharge_summary",
    "problem_oriented",
)
SUMMARY_SECTION_NAMES: Final[tuple[str, ...]] = (
    "admission",
    "assessment",
    "discharge",
    "follow_up",
    "handoff",
    "history",
    "hospital_course",
    "medications",
    "pending",
    "plan",
    "procedures",
    "results",
)
SUMMARY_FIELD_NAMES: Final[tuple[str, ...]] = (
    "active_problems",
    "admission_reason",
    "assessment",
    "background",
    "current_situation",
    "discharge_condition",
    "discharge_diagnoses",
    "discharge_medications",
    "follow_up",
    "hospital_course",
    "key_findings",
    "pending_items",
    "plan",
    "procedures",
    "safety_concerns",
)

_PROFILE_ALIASES: Final[Mapping[str, str]] = {
    "bhc": "brief_hospital_course",
    "brief-hospital-course": "brief_hospital_course",
    "clinical-handoff": "clinical_handoff",
    "discharge": "discharge_summary",
    "discharge-summary": "discharge_summary",
    "problem-list": "problem_oriented",
    "problem_list": "problem_oriented",
}
_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_PROFILE_REFERENCE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}(?:@[0-9]+\.[0-9]+)?$")
_MISSING = object()


class SummaryFieldType(str, Enum):
    """Closed set of values a local summary field may contain."""

    TEXT = "text"
    TEXT_LIST = "text_list"
    PROBLEM_LIST = "problem_list"
    MEDICATION_LIST = "medication_list"
    PROCEDURE_LIST = "procedure_list"
    FOLLOW_UP_LIST = "follow_up_list"

    @property
    def repeated(self) -> bool:
        """Return whether this field contains a bounded list of strings."""

        return self is not SummaryFieldType.TEXT


class SummaryProfileError(ValueError):
    """Raised when a summary profile or typed summary payload is unsafe."""


class UnknownSummaryProfileError(SummaryProfileError):
    """Raised when a closed-world summary profile is not registered."""


class UnknownSummaryProfileVersionError(SummaryProfileError):
    """Raised when a registered profile is requested at an unknown version."""


_SUMMARY_FINDING_REASONS: Final[frozenset[str]] = frozenset(
    {
        "missing_required",
        "invalid_type",
        "empty_value",
        "too_many_items",
        "value_too_long",
        "invalid_item_type",
        "empty_item",
        "item_too_long",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize profile metadata with stable ordering and no non-finite data."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot_mapping(
    value: Any,
    label: str,
    *,
    maximum_items: int,
) -> dict[str, Any]:
    """Copy a bounded mapping without allowing hostile values to escape."""

    if not isinstance(value, Mapping):
        raise SummaryProfileError(f"{label} must be an object")
    try:
        items = list(itertools.islice(value.items(), maximum_items + 1))
    except Exception:
        raise SummaryProfileError(f"{label} could not be read") from None
    if len(items) > maximum_items:
        raise SummaryProfileError(f"{label} exceeds the supported item limit")

    snapshot: dict[str, Any] = {}
    for item in items:
        if type(item) not in (tuple, list) or len(item) != 2:
            raise SummaryProfileError(f"{label} could not be read")
        key, item_value = item
        if type(key) is not str:
            raise SummaryProfileError(f"{label} keys must be strings")
        if key in snapshot:
            raise SummaryProfileError(f"{label} contains duplicate fields")
        snapshot[key] = item_value
    return snapshot


def _unknown_fields(
    value: Mapping[str, Any], allowed: frozenset[str], label: str
) -> None:
    if any(key not in allowed for key in value):
        raise SummaryProfileError(f"{label} contains unsupported fields")


def _aliased_value(
    value: Mapping[str, Any],
    aliases: tuple[str, ...],
    label: str,
    *,
    default: Any = _MISSING,
) -> Any:
    present = tuple(alias for alias in aliases if alias in value)
    if len(present) > 1:
        raise SummaryProfileError(f"{label} contains conflicting aliases")
    if present:
        return value[present[0]]
    if default is not _MISSING:
        return default
    raise SummaryProfileError(f"{label} is required")


def _canonical_profile_name(value: Any) -> str:
    if type(value) is not str:
        raise UnknownSummaryProfileError("unsupported summary profile")
    normalized = value.strip().lower()
    normalized = _PROFILE_ALIASES.get(normalized, normalized)
    if (
        not _PROFILE_NAME_RE.fullmatch(normalized)
        or normalized not in SUMMARY_PROFILE_NAMES
    ):
        raise UnknownSummaryProfileError("unsupported summary profile")
    return normalized


def _profile_version(value: Any) -> str:
    if type(value) is not str or value != SUMMARY_PROFILE_VERSION:
        raise UnknownSummaryProfileVersionError("unsupported summary profile version")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise SummaryProfileError(f"{label} must be a boolean")
    return value


def _bounded_integer(value: Any, label: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise SummaryProfileError(f"{label} is outside the supported bound")
    return value


def _nonnegative_integer(value: Any, label: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise SummaryProfileError(f"{label} is outside the supported bound")
    return value


def _normalize_sections(value: Any) -> tuple[str, ...]:
    if type(value) not in (tuple, list):
        raise SummaryProfileError("field source_sections must be a sequence")
    try:
        sections = tuple(value)
    except Exception:
        raise SummaryProfileError("field source_sections could not be read") from None
    normalized: set[str] = set()
    for section in sections:
        if type(section) is not str or section not in SUMMARY_SECTION_NAMES:
            raise SummaryProfileError(
                "field source_sections contains an unsupported section"
            )
        normalized.add(section)
    return tuple(sorted(normalized))


def _normalize_field_type(value: Any) -> SummaryFieldType:
    if type(value) is SummaryFieldType:
        return value
    if type(value) is not str:
        raise SummaryProfileError("field_type is unsupported")
    try:
        return SummaryFieldType(value)
    except ValueError:
        raise SummaryProfileError("field_type is unsupported") from None


@dataclass(frozen=True, slots=True)
class SummaryTemplateField:
    """One bounded, typed field in a local summary template."""

    name: str
    field_type: SummaryFieldType | str
    required: bool = False
    max_items: int | None = None
    max_characters: int = MAX_SUMMARY_FIELD_CHARACTERS
    source_sections: tuple[str, ...] = ()
    evidence_required: bool = True

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in SUMMARY_FIELD_NAMES:
            raise SummaryProfileError("field name is unsupported")
        field_type = _normalize_field_type(self.field_type)
        required = _strict_bool(self.required, "field required")
        evidence_required = _strict_bool(
            self.evidence_required,
            "field evidence_required",
        )
        if self.max_items is not None:
            if type(self.max_items) is not int:
                raise SummaryProfileError("field max_items must be an integer or null")
            _bounded_integer(
                self.max_items,
                "field max_items",
                maximum=MAX_SUMMARY_FIELD_ITEMS,
            )
        if field_type.repeated and self.max_items is None:
            raise SummaryProfileError("repeated fields require max_items")
        if not field_type.repeated and self.max_items is not None:
            raise SummaryProfileError("scalar fields cannot declare max_items")
        max_characters = _bounded_integer(
            self.max_characters,
            "field max_characters",
            maximum=MAX_SUMMARY_FIELD_CHARACTERS,
        )
        sections = _normalize_sections(self.source_sections)
        object.__setattr__(self, "field_type", field_type)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "evidence_required", evidence_required)
        object.__setattr__(self, "max_characters", max_characters)
        object.__setattr__(self, "source_sections", sections)

    @property
    def value_type(self) -> str:
        """Return the serialized type name under a descriptive alias."""

        return cast(SummaryFieldType, self.field_type).value

    @property
    def type(self) -> str:
        """Return the serialized type name for schema-oriented callers."""

        return self.value_type

    @property
    def repeated(self) -> bool:
        """Return whether the field accepts a bounded list."""

        return cast(SummaryFieldType, self.field_type).repeated

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready field definition without source values."""

        return {
            "name": self.name,
            "field_type": self.value_type,
            "required": self.required,
            "max_items": self.max_items,
            "max_characters": self.max_characters,
            "source_sections": list(self.source_sections),
            "evidence_required": self.evidence_required,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SummaryTemplateField":
        """Build a field from its closed, typed metadata mapping."""

        data = _snapshot_mapping(
            value,
            "summary field",
            maximum_items=16,
        )
        _unknown_fields(
            data,
            frozenset(
                {
                    "name",
                    "field_type",
                    "value_type",
                    "type",
                    "required",
                    "max_items",
                    "max_characters",
                    "source_sections",
                    "evidence_required",
                }
            ),
            "summary field",
        )
        field_type = _aliased_value(
            data,
            ("field_type", "value_type", "type"),
            "summary field type",
        )
        return cls(
            name=_aliased_value(data, ("name",), "summary field name"),
            field_type=field_type,
            required=data.get("required", False),
            max_items=data.get("max_items"),
            max_characters=data.get(
                "max_characters",
                MAX_SUMMARY_FIELD_CHARACTERS,
            ),
            source_sections=data.get("source_sections", ()),
            evidence_required=data.get("evidence_required", True),
        )

    from_dict = from_mapping


@dataclass(frozen=True, slots=True)
class SummaryTemplateProfile:
    """An immutable, versioned typed contract for one summary form."""

    name: str
    version: str
    fields: tuple[SummaryTemplateField, ...]
    schema_version: int = SUMMARY_PROFILE_SCHEMA_VERSION
    format: str = SUMMARY_PROFILE_FORMAT
    requires_clinician_review: bool = True
    autonomous_decision: bool = False
    disclaimer: str = SUMMARY_PROFILE_DISCLAIMER

    def __post_init__(self) -> None:
        name = _canonical_profile_name(self.name)
        version = _profile_version(self.version)
        if type(self.schema_version) is not int or (
            self.schema_version != SUMMARY_PROFILE_SCHEMA_VERSION
        ):
            raise SummaryProfileError("unsupported summary profile schema version")
        if type(self.format) is not str or self.format != SUMMARY_PROFILE_FORMAT:
            raise SummaryProfileError("summary profile format is unsupported")
        if type(self.disclaimer) is not str or (
            self.disclaimer != SUMMARY_PROFILE_DISCLAIMER
        ):
            raise SummaryProfileError("summary profile disclaimer is unsupported")
        requires_review = _strict_bool(
            self.requires_clinician_review,
            "summary profile requires_clinician_review",
        )
        autonomous_decision = _strict_bool(
            self.autonomous_decision,
            "summary profile autonomous_decision",
        )
        if not requires_review or autonomous_decision:
            raise SummaryProfileError("summary profile review guardrails are required")
        if type(self.fields) not in (tuple, list):
            raise SummaryProfileError("summary profile fields must be a sequence")
        try:
            fields = tuple(self.fields)
        except Exception:
            raise SummaryProfileError(
                "summary profile fields could not be read"
            ) from None
        if not 0 < len(fields) <= MAX_SUMMARY_PROFILE_FIELDS:
            raise SummaryProfileError("summary profile field count is unsupported")
        if any(type(field) is not SummaryTemplateField for field in fields):
            raise SummaryProfileError("summary profile fields are invalid")
        names = tuple(field.name for field in fields)
        if len(set(names)) != len(names):
            raise SummaryProfileError("summary profile fields must be unique")
        if not any(field.required for field in fields):
            raise SummaryProfileError("summary profile requires a required field")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "requires_clinician_review", requires_review)
        object.__setattr__(self, "autonomous_decision", autonomous_decision)

    @property
    def profile_name(self) -> str:
        """Return the stable profile name."""

        return self.name

    @property
    def profile_version(self) -> str:
        """Return the stable version under the explicit profile terminology."""

        return self.version

    @property
    def profile_id(self) -> str:
        """Return the stable ``name@version`` catalog identifier."""

        return f"{self.name}@{self.version}"

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return fields in their canonical template order."""

        return tuple(field.name for field in self.fields)

    @property
    def field_map(self) -> Mapping[str, SummaryTemplateField]:
        """Return an immutable field lookup."""

        return MappingProxyType({field.name: field for field in self.fields})

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic profile metadata without prompts or values."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "format": self.format,
            "fields": [field.to_dict() for field in self.fields],
            "requires_clinician_review": self.requires_clinician_review,
            "autonomous_decision": self.autonomous_decision,
            "disclaimer": self.disclaimer,
        }

    def canonical_json(self) -> str:
        """Return the compact JSON used to calculate the profile digest."""

        return _canonical_json(self.to_dict())

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the profile as stable, human-readable JSON."""

        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=indent,
                sort_keys=True,
            )
            + "\n"
        )

    @property
    def digest(self) -> str:
        """Return a SHA-256 digest of the canonical profile definition."""

        return _digest(self.canonical_json())

    @property
    def profile_digest(self) -> str:
        """Return :attr:`digest` under the explicit profile terminology."""

        return self.digest

    @property
    def fingerprint(self) -> str:
        """Compatibility alias for the profile digest."""

        return self.digest

    def validate(self, output: Mapping[str, Any]) -> "SummaryValidationReport":
        """Validate a generated typed payload without retaining its values."""

        return validate_summary_output(self, output)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SummaryTemplateProfile":
        """Load a known profile version from a closed metadata mapping."""

        return _load_profile_mapping(value)

    from_dict = from_mapping

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "SummaryTemplateProfile":
        """Load a known profile from local JSON text."""

        return _load_profile_json(value)


@dataclass(frozen=True, slots=True)
class SummaryValidationFinding:
    """A value-free typed-output validation finding."""

    field_name: str
    reason_code: str
    item_index: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.field_name) is not str
            or self.field_name not in SUMMARY_FIELD_NAMES
        ):
            raise SummaryProfileError("validation field_name is unsupported")
        if (
            type(self.reason_code) is not str
            or self.reason_code not in _SUMMARY_FINDING_REASONS
        ):
            raise SummaryProfileError("validation reason_code is unsupported")
        if self.item_index is not None:
            _nonnegative_integer(
                self.item_index,
                "validation item_index",
                maximum=MAX_SUMMARY_FIELD_ITEMS,
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready finding with no submitted value."""

        payload: dict[str, Any] = {
            "field_name": self.field_name,
            "reason_code": self.reason_code,
        }
        if self.item_index is not None:
            payload["item_index"] = self.item_index
        return payload


@dataclass(frozen=True, slots=True)
class SummaryValidationReport:
    """Deterministic, value-free validation output for a typed summary."""

    profile_name: str
    profile_version: str
    valid: bool
    findings: tuple[SummaryValidationFinding, ...] = ()
    unknown_field_count: int = 0
    schema_version: int = SUMMARY_PROFILE_SCHEMA_VERSION
    requires_clinician_review: bool = True

    def __post_init__(self) -> None:
        _canonical_profile_name(self.profile_name)
        _profile_version(self.profile_version)
        _strict_bool(self.valid, "validation valid")
        _strict_bool(
            self.requires_clinician_review,
            "validation requires_clinician_review",
        )
        if not self.requires_clinician_review:
            raise SummaryProfileError("validation review guardrail is required")
        if type(self.schema_version) is not int or (
            self.schema_version != SUMMARY_PROFILE_SCHEMA_VERSION
        ):
            raise SummaryProfileError("unsupported validation schema version")
        if type(self.findings) not in (tuple, list):
            raise SummaryProfileError("validation findings must be a sequence")
        findings = tuple(self.findings)
        if any(type(finding) is not SummaryValidationFinding for finding in findings):
            raise SummaryProfileError("validation findings are invalid")
        unknown_count = _nonnegative_integer(
            self.unknown_field_count,
            "validation unknown_field_count",
            maximum=MAX_SUMMARY_OUTPUT_FIELDS,
        )
        if self.valid != (not findings and unknown_count == 0):
            raise SummaryProfileError("validation status is inconsistent")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "unknown_field_count", unknown_count)

    @property
    def error_codes(self) -> tuple[str, ...]:
        """Return sorted distinct reason codes for safe aggregation."""

        return tuple(sorted({finding.reason_code for finding in self.findings}))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report with metadata and counts only."""

        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "profile_digest": summary_profile_digest(
                self.profile_name,
                version=self.profile_version,
            ),
            "valid": self.valid,
            "unknown_field_count": self.unknown_field_count,
            "findings": [finding.to_dict() for finding in self.findings],
            "requires_clinician_review": self.requires_clinician_review,
            "disclaimer": SUMMARY_PROFILE_DISCLAIMER,
        }

    def to_json(self) -> str:
        """Serialize the report as compact deterministic JSON."""

        return _canonical_json(self.to_dict())


SummaryProfile = SummaryTemplateProfile
SummaryField = SummaryTemplateField
SummaryProfileValidation = SummaryValidationReport
SummaryProfileValidationFinding = SummaryValidationFinding


def _field(
    name: str,
    field_type: SummaryFieldType,
    *,
    required: bool = False,
    max_items: int | None = None,
    max_characters: int,
    source_sections: tuple[str, ...],
) -> SummaryTemplateField:
    return SummaryTemplateField(
        name=name,
        field_type=field_type,
        required=required,
        max_items=max_items,
        max_characters=max_characters,
        source_sections=source_sections,
    )


BRIEF_HOSPITAL_COURSE_V1 = SummaryTemplateProfile(
    name="brief_hospital_course",
    version=SUMMARY_PROFILE_VERSION,
    fields=(
        _field(
            "admission_reason",
            SummaryFieldType.TEXT,
            required=True,
            max_characters=512,
            source_sections=("admission", "history"),
        ),
        _field(
            "discharge_diagnoses",
            SummaryFieldType.PROBLEM_LIST,
            required=True,
            max_items=8,
            max_characters=512,
            source_sections=("assessment", "discharge"),
        ),
        _field(
            "hospital_course",
            SummaryFieldType.TEXT,
            required=True,
            max_characters=4_096,
            source_sections=("assessment", "hospital_course"),
        ),
        _field(
            "key_findings",
            SummaryFieldType.TEXT_LIST,
            max_items=8,
            max_characters=512,
            source_sections=("assessment", "results"),
        ),
        _field(
            "procedures",
            SummaryFieldType.PROCEDURE_LIST,
            max_items=8,
            max_characters=512,
            source_sections=("procedures",),
        ),
        _field(
            "discharge_condition",
            SummaryFieldType.TEXT,
            max_characters=256,
            source_sections=("discharge",),
        ),
        _field(
            "follow_up",
            SummaryFieldType.FOLLOW_UP_LIST,
            max_items=8,
            max_characters=512,
            source_sections=("follow_up",),
        ),
    ),
)

DISCHARGE_SUMMARY_V1 = SummaryTemplateProfile(
    name="discharge_summary",
    version=SUMMARY_PROFILE_VERSION,
    fields=(
        _field(
            "admission_reason",
            SummaryFieldType.TEXT,
            required=True,
            max_characters=512,
            source_sections=("admission", "history"),
        ),
        _field(
            "discharge_diagnoses",
            SummaryFieldType.PROBLEM_LIST,
            required=True,
            max_items=16,
            max_characters=512,
            source_sections=("assessment", "discharge"),
        ),
        _field(
            "hospital_course",
            SummaryFieldType.TEXT,
            required=True,
            max_characters=4_096,
            source_sections=("assessment", "hospital_course"),
        ),
        _field(
            "procedures",
            SummaryFieldType.PROCEDURE_LIST,
            max_items=16,
            max_characters=512,
            source_sections=("procedures",),
        ),
        _field(
            "discharge_medications",
            SummaryFieldType.MEDICATION_LIST,
            max_items=24,
            max_characters=512,
            source_sections=("discharge", "medications"),
        ),
        _field(
            "discharge_condition",
            SummaryFieldType.TEXT,
            max_characters=256,
            source_sections=("discharge",),
        ),
        _field(
            "follow_up",
            SummaryFieldType.FOLLOW_UP_LIST,
            max_items=16,
            max_characters=512,
            source_sections=("follow_up",),
        ),
        _field(
            "pending_items",
            SummaryFieldType.TEXT_LIST,
            max_items=8,
            max_characters=512,
            source_sections=("pending", "discharge"),
        ),
    ),
)

PROBLEM_ORIENTED_V1 = SummaryTemplateProfile(
    name="problem_oriented",
    version=SUMMARY_PROFILE_VERSION,
    fields=(
        _field(
            "active_problems",
            SummaryFieldType.PROBLEM_LIST,
            required=True,
            max_items=16,
            max_characters=512,
            source_sections=("assessment", "history"),
        ),
        _field(
            "assessment",
            SummaryFieldType.TEXT,
            required=True,
            max_characters=2_048,
            source_sections=("assessment",),
        ),
        _field(
            "plan",
            SummaryFieldType.TEXT_LIST,
            required=True,
            max_items=16,
            max_characters=512,
            source_sections=("plan",),
        ),
        _field(
            "pending_items",
            SummaryFieldType.TEXT_LIST,
            max_items=8,
            max_characters=512,
            source_sections=("pending",),
        ),
        _field(
            "follow_up",
            SummaryFieldType.FOLLOW_UP_LIST,
            max_items=8,
            max_characters=512,
            source_sections=("follow_up",),
        ),
    ),
)

CLINICAL_HANDOFF_V1 = SummaryTemplateProfile(
    name="clinical_handoff",
    version=SUMMARY_PROFILE_VERSION,
    fields=(
        _field(
            "current_situation",
            SummaryFieldType.TEXT,
            required=True,
            max_characters=1_024,
            source_sections=("handoff",),
        ),
        _field(
            "background",
            SummaryFieldType.TEXT,
            required=True,
            max_characters=2_048,
            source_sections=("history", "handoff"),
        ),
        _field(
            "assessment",
            SummaryFieldType.TEXT,
            required=True,
            max_characters=2_048,
            source_sections=("assessment", "handoff"),
        ),
        _field(
            "plan",
            SummaryFieldType.TEXT_LIST,
            required=True,
            max_items=12,
            max_characters=512,
            source_sections=("handoff", "plan"),
        ),
        _field(
            "safety_concerns",
            SummaryFieldType.TEXT_LIST,
            max_items=8,
            max_characters=512,
            source_sections=("assessment", "handoff"),
        ),
        _field(
            "pending_items",
            SummaryFieldType.TEXT_LIST,
            max_items=8,
            max_characters=512,
            source_sections=("handoff", "pending"),
        ),
    ),
)

BHC_V1 = BRIEF_HOSPITAL_COURSE_V1
DISCHARGE_V1 = DISCHARGE_SUMMARY_V1
PROBLEM_LIST_V1 = PROBLEM_ORIENTED_V1
HANDOFF_V1 = CLINICAL_HANDOFF_V1

SUMMARY_PROFILES: Mapping[str, SummaryTemplateProfile] = MappingProxyType(
    {
        profile.name: profile
        for profile in (
            BRIEF_HOSPITAL_COURSE_V1,
            CLINICAL_HANDOFF_V1,
            DISCHARGE_SUMMARY_V1,
            PROBLEM_ORIENTED_V1,
        )
    }
)
SUMMARY_TEMPLATE_PROFILES = SUMMARY_PROFILES
BUILTIN_SUMMARY_PROFILES = SUMMARY_PROFILES


def _registered_profile(name: Any, version: Any) -> SummaryTemplateProfile:
    canonical_name = _canonical_profile_name(name)
    canonical_version = _profile_version(version)
    try:
        profile = SUMMARY_PROFILES[canonical_name]
    except KeyError:
        raise UnknownSummaryProfileError("unsupported summary profile") from None
    if profile.version != canonical_version:
        raise UnknownSummaryProfileVersionError("unsupported summary profile version")
    return profile


def _field_sequence(value: Any) -> tuple[SummaryTemplateField, ...]:
    if type(value) not in (tuple, list):
        raise SummaryProfileError("summary profile fields must be a sequence")
    try:
        entries = tuple(value)
    except Exception:
        raise SummaryProfileError("summary profile fields could not be read") from None
    if not 0 < len(entries) <= MAX_SUMMARY_PROFILE_FIELDS:
        raise SummaryProfileError("summary profile field count is unsupported")
    fields: list[SummaryTemplateField] = []
    for entry in entries:
        if type(entry) is SummaryTemplateField:
            fields.append(entry)
        elif isinstance(entry, Mapping):
            fields.append(SummaryTemplateField.from_mapping(entry))
        else:
            raise SummaryProfileError("summary profile field is invalid")
    return tuple(fields)


def _field_signature(
    fields: Sequence[SummaryTemplateField],
) -> tuple[tuple[str, str, bool, int | None, int, tuple[str, ...], bool], ...]:
    return tuple(
        (
            field.name,
            field.value_type,
            field.required,
            field.max_items,
            field.max_characters,
            field.source_sections,
            field.evidence_required,
        )
        for field in sorted(fields, key=lambda item: item.name)
    )


def _load_profile_mapping(value: Mapping[str, Any]) -> SummaryTemplateProfile:
    data = _snapshot_mapping(
        value,
        "summary profile",
        maximum_items=16,
    )
    _unknown_fields(
        data,
        frozenset(
            {
                "schema_version",
                "name",
                "profile_name",
                "id",
                "version",
                "profile_version",
                "format",
                "fields",
                "requires_clinician_review",
                "autonomous_decision",
                "disclaimer",
            }
        ),
        "summary profile",
    )
    name = _aliased_value(
        data,
        ("name", "profile_name", "id"),
        "summary profile name",
    )
    version = _aliased_value(
        data,
        ("version", "profile_version"),
        "summary profile version",
    )
    expected = _registered_profile(name, version)
    fields = _field_sequence(data.get("fields", _MISSING))
    candidate = SummaryTemplateProfile(
        name=name,
        version=version,
        fields=fields,
        schema_version=data.get("schema_version", SUMMARY_PROFILE_SCHEMA_VERSION),
        format=data.get("format", SUMMARY_PROFILE_FORMAT),
        requires_clinician_review=data.get("requires_clinician_review", True),
        autonomous_decision=data.get("autonomous_decision", False),
        disclaimer=data.get("disclaimer", SUMMARY_PROFILE_DISCLAIMER),
    )
    if _field_signature(candidate.fields) != _field_signature(expected.fields):
        raise SummaryProfileError(
            "summary profile definition does not match its known version"
        )
    if candidate.to_dict() == expected.to_dict():
        return expected
    # Field order is presentation metadata, not a version escape hatch.  The
    # loaded object is always the canonical catalog object, even if a local
    # JSON document listed the same fields in another order.
    if (
        candidate.schema_version != expected.schema_version
        or candidate.format != expected.format
        or candidate.requires_clinician_review != expected.requires_clinician_review
        or candidate.autonomous_decision != expected.autonomous_decision
        or candidate.disclaimer != expected.disclaimer
    ):
        raise SummaryProfileError(
            "summary profile metadata does not match its known version"
        )
    return expected


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _load_profile_json(value: str | bytes | bytearray) -> SummaryTemplateProfile:
    if type(value) not in (str, bytes, bytearray):
        raise SummaryProfileError("summary profile JSON must be text or bytes")
    try:
        payload_size = len(value.encode("utf-8")) if type(value) is str else len(value)
    except UnicodeError:
        raise SummaryProfileError("invalid summary profile JSON") from None
    if payload_size > MAX_SUMMARY_PROFILE_JSON_BYTES:
        raise SummaryProfileError(
            "summary profile JSON exceeds the supported size limit"
        )
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise SummaryProfileError("invalid summary profile JSON") from None
    if not isinstance(payload, Mapping):
        raise SummaryProfileError("summary profile JSON must contain an object")
    return _load_profile_mapping(payload)


def _read_local_profile(path: Path) -> SummaryTemplateProfile:
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_SUMMARY_PROFILE_JSON_BYTES + 1)
    except OSError:
        raise SummaryProfileError("could not read local summary profile") from None
    return _load_profile_json(payload)


def _profile_reference(
    value: str, version: str | None
) -> SummaryTemplateProfile | None:
    reference = value.strip().lower()
    if not _SAFE_PROFILE_REFERENCE_RE.fullmatch(reference):
        return None
    if "@" in reference:
        name, reference_version = reference.rsplit("@", 1)
        if version is not None:
            raise SummaryProfileError("summary profile reference contains two versions")
        version = reference_version
    try:
        name = _canonical_profile_name(name if "@" in reference else reference)
    except UnknownSummaryProfileError:
        return None
    return _registered_profile(
        name, SUMMARY_PROFILE_VERSION if version is None else version
    )


def load_summary_profile(
    source: object = None,
    *,
    version: str | None = None,
) -> SummaryTemplateProfile:
    """Load a known profile by name, JSON, mapping, or local path.

    A bare profile name selects the current catalog version.  JSON and local
    mappings must carry an explicit ``version`` (or ``profile_version``).
    URLs, remote references, unknown names, and unknown versions are rejected;
    no network or model access is performed.
    """

    if version is not None and type(version) is not str:
        raise UnknownSummaryProfileVersionError("unsupported summary profile version")
    if source is None:
        return _registered_profile(
            "brief_hospital_course",
            SUMMARY_PROFILE_VERSION if version is None else version,
        )
    if type(source) is SummaryTemplateProfile:
        profile = _registered_profile(source.name, source.version)
        if version is not None and version != profile.version:
            raise UnknownSummaryProfileVersionError(
                "unsupported summary profile version"
            )
        if source.to_dict() != profile.to_dict():
            raise SummaryProfileError(
                "summary profile is not the registered definition"
            )
        return profile
    if isinstance(source, Mapping):
        if version is not None:
            raise SummaryProfileError("mapping profile sources cannot override version")
        return _load_profile_mapping(source)
    if type(source) in (bytes, bytearray):
        if version is not None:
            raise SummaryProfileError("JSON profile sources cannot override version")
        return _load_profile_json(cast(bytes | bytearray, source))
    if type(source) is str:
        if version is not None and source.lstrip().startswith(("{", "[")):
            raise SummaryProfileError("JSON profile sources cannot override version")
        if source.lstrip().startswith(("{", "[")):
            return _load_profile_json(source)
        if "://" in source:
            raise SummaryProfileError("summary profile sources must be local")
        referenced_profile = _profile_reference(source, version)
        if referenced_profile is not None:
            return referenced_profile
        return _read_local_profile(Path(source))
    if isinstance(source, Path):
        if version is not None:
            raise SummaryProfileError("local profile paths cannot override version")
        return _read_local_profile(source)
    raise TypeError(
        "summary profile source must be a name, mapping, JSON, or local path"
    )


def get_summary_profile(
    name: str = "brief_hospital_course",
    *,
    version: str | None = None,
) -> SummaryTemplateProfile:
    """Return a registered profile by name and optional explicit version."""

    return load_summary_profile(name, version=version)


def default_summary_profile() -> SummaryTemplateProfile:
    """Return the default brief-hospital-course profile."""

    return BRIEF_HOSPITAL_COURSE_V1


def available_summary_profiles() -> tuple[str, ...]:
    """Return registered profile names in stable order."""

    return SUMMARY_PROFILE_NAMES


def available_summary_profile_versions(name: str) -> tuple[str, ...]:
    """Return the registered versions for one profile name."""

    _canonical_profile_name(name)
    return (SUMMARY_PROFILE_VERSION,)


def summary_profile_digest(
    name: str = "brief_hospital_course",
    *,
    version: str | None = None,
) -> str:
    """Return the digest for a registered profile definition."""

    return get_summary_profile(name, version=version).digest


def _validate_field_value(
    field: SummaryTemplateField,
    value: Any,
) -> list[SummaryValidationFinding]:
    findings: list[SummaryValidationFinding] = []
    if field.repeated:
        if type(value) not in (list, tuple):
            return [SummaryValidationFinding(field.name, "invalid_type")]
        if field.max_items is not None and len(value) > field.max_items:
            findings.append(SummaryValidationFinding(field.name, "too_many_items"))
        for index, item in enumerate(value[:MAX_SUMMARY_FIELD_ITEMS]):
            if type(item) is not str:
                findings.append(
                    SummaryValidationFinding(field.name, "invalid_item_type", index)
                )
            elif not item.strip():
                findings.append(
                    SummaryValidationFinding(field.name, "empty_item", index)
                )
            elif len(item) > field.max_characters:
                findings.append(
                    SummaryValidationFinding(field.name, "item_too_long", index)
                )
        return findings
    if type(value) is not str:
        return [SummaryValidationFinding(field.name, "invalid_type")]
    if not value.strip():
        findings.append(SummaryValidationFinding(field.name, "empty_value"))
    if len(value) > field.max_characters:
        findings.append(SummaryValidationFinding(field.name, "value_too_long"))
    return findings


def validate_summary_output(
    profile: SummaryTemplateProfile,
    output: Mapping[str, Any],
) -> SummaryValidationReport:
    """Validate typed summary output while returning only safe metadata.

    The function never stores, formats, hashes, or echoes submitted field
    values.  Unknown output fields are counted rather than named, and every
    finding uses a fixed field name and reason code from the profile contract.
    """

    if type(profile) is not SummaryTemplateProfile:
        raise TypeError("profile must be a SummaryTemplateProfile")
    data = _snapshot_mapping(
        output,
        "summary output",
        maximum_items=MAX_SUMMARY_OUTPUT_FIELDS,
    )
    known_fields = profile.field_map
    unknown_field_count = sum(key not in known_fields for key in data)
    findings: list[SummaryValidationFinding] = []
    for field in profile.fields:
        if field.name not in data:
            if field.required:
                findings.append(
                    SummaryValidationFinding(field.name, "missing_required")
                )
            continue
        findings.extend(_validate_field_value(field, data[field.name]))
    return SummaryValidationReport(
        profile_name=profile.name,
        profile_version=profile.version,
        valid=not findings and unknown_field_count == 0,
        findings=tuple(findings),
        unknown_field_count=unknown_field_count,
        requires_clinician_review=profile.requires_clinician_review,
    )


validate_summary = validate_summary_output
load_summary_template_profile = load_summary_profile
get_summary_template_profile = get_summary_profile


__all__ = [
    "BHC_V1",
    "BRIEF_HOSPITAL_COURSE_V1",
    "BUILTIN_SUMMARY_PROFILES",
    "CLINICAL_HANDOFF_V1",
    "CURRENT_SUMMARY_PROFILE_VERSION",
    "DISCHARGE_SUMMARY_V1",
    "DISCHARGE_V1",
    "HANDOFF_V1",
    "MAX_SUMMARY_FIELD_CHARACTERS",
    "MAX_SUMMARY_FIELD_ITEMS",
    "MAX_SUMMARY_OUTPUT_FIELDS",
    "MAX_SUMMARY_PROFILE_FIELDS",
    "MAX_SUMMARY_PROFILE_JSON_BYTES",
    "PROBLEM_LIST_V1",
    "PROBLEM_ORIENTED_V1",
    "SUMMARY_FIELD_NAMES",
    "SUMMARY_PROFILE_DISCLAIMER",
    "SUMMARY_PROFILE_FORMAT",
    "SUMMARY_PROFILE_NAMES",
    "SUMMARY_PROFILE_SCHEMA_VERSION",
    "SUMMARY_PROFILE_VERSION",
    "SUMMARY_PROFILES",
    "SUMMARY_SECTION_NAMES",
    "SUMMARY_TEMPLATE_PROFILE_SCHEMA_VERSION",
    "SUMMARY_TEMPLATE_PROFILES",
    "SummaryField",
    "SummaryFieldType",
    "SummaryProfile",
    "SummaryProfileError",
    "SummaryProfileValidation",
    "SummaryProfileValidationFinding",
    "SummaryTemplateField",
    "SummaryTemplateProfile",
    "SummaryValidationFinding",
    "SummaryValidationReport",
    "UnknownSummaryProfileError",
    "UnknownSummaryProfileVersionError",
    "available_summary_profile_versions",
    "available_summary_profiles",
    "default_summary_profile",
    "get_summary_profile",
    "get_summary_template_profile",
    "load_summary_profile",
    "load_summary_template_profile",
    "summary_profile_digest",
    "validate_summary",
    "validate_summary_output",
]
