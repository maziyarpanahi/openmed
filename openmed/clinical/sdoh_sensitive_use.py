"""Machine-readable sensitive-use labels for SDOH output fields."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

SDOH_SENSITIVE_USE_SCHEMA_VERSION: Final = 1

_SAFE_FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


class SDOHPurpose(str, Enum):
    """Permitted purposes for labeled SDOH fields."""

    CLINICAL_REVIEW = "clinical_review"
    CARE_COORDINATION = "care_coordination"
    PATIENT_REQUESTED_SUMMARY = "patient_requested_summary"


class ProhibitedAutomatedUse(str, Enum):
    """Automated decisions that must not consume SDOH fields."""

    ELIGIBILITY_DECISION = "eligibility_decision"
    INSURANCE_UNDERWRITING = "insurance_underwriting"
    EMPLOYMENT_DECISION = "employment_decision"
    CARE_DENIAL = "care_denial"
    AUTONOMOUS_DIAGNOSIS = "autonomous_diagnosis"


@dataclass(frozen=True, slots=True)
class SDOHSensitiveUseLabel:
    """Purpose restrictions attached to one exported field."""

    field_name: str
    allowed_purposes: tuple[SDOHPurpose, ...] = (
        SDOHPurpose.CLINICAL_REVIEW,
        SDOHPurpose.CARE_COORDINATION,
        SDOHPurpose.PATIENT_REQUESTED_SUMMARY,
    )
    prohibited_automated_uses: tuple[ProhibitedAutomatedUse, ...] = tuple(
        ProhibitedAutomatedUse
    )
    human_review_required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_name", _field_name(self.field_name))
        if any(not isinstance(item, SDOHPurpose) for item in self.allowed_purposes):
            raise TypeError("allowed_purposes must contain SDOHPurpose values")
        if any(
            not isinstance(item, ProhibitedAutomatedUse)
            for item in self.prohibited_automated_uses
        ):
            raise TypeError(
                "prohibited_automated_uses must contain ProhibitedAutomatedUse values"
            )
        allowed = tuple(sorted(set(self.allowed_purposes), key=lambda item: item.value))
        prohibited = tuple(
            sorted(set(self.prohibited_automated_uses), key=lambda item: item.value)
        )
        if not allowed:
            raise ValueError("allowed_purposes must not be empty")
        if not prohibited:
            raise ValueError("prohibited_automated_uses must not be empty")
        if type(self.human_review_required) is not bool:
            raise TypeError("human_review_required must be a boolean")
        object.__setattr__(self, "allowed_purposes", allowed)
        object.__setattr__(self, "prohibited_automated_uses", prohibited)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic machine-readable label."""

        return {
            "field_name": self.field_name,
            "sensitivity": "social_determinants_of_health",
            "allowed_purposes": [item.value for item in self.allowed_purposes],
            "prohibited_automated_uses": [
                item.value for item in self.prohibited_automated_uses
            ],
            "human_review_required": self.human_review_required,
        }


@dataclass(frozen=True, slots=True)
class LabeledSDOHExport:
    """An export that cannot serialize without complete field labels."""

    fields: Mapping[str, Any] = field(repr=False)
    labels: tuple[SDOHSensitiveUseLabel, ...]
    schema_version: int = SDOH_SENSITIVE_USE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SDOH_SENSITIVE_USE_SCHEMA_VERSION:
            raise ValueError("unsupported SDOH sensitive-use schema version")
        if not isinstance(self.fields, Mapping):
            raise TypeError("fields must be a mapping")
        copied_fields = {_field_name(key): value for key, value in self.fields.items()}
        if any(not isinstance(label, SDOHSensitiveUseLabel) for label in self.labels):
            raise TypeError("labels must contain SDOHSensitiveUseLabel values")
        labels = tuple(sorted(self.labels, key=lambda label: label.field_name))
        if len({label.field_name for label in labels}) != len(labels):
            raise ValueError("sensitive-use labels must be unique per field")
        if set(copied_fields) != {label.field_name for label in labels}:
            raise ValueError("every exported SDOH field requires exactly one label")
        try:
            json.dumps(copied_fields, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError, OverflowError):
            raise TypeError("SDOH export fields must be JSON-compatible") from None
        object.__setattr__(self, "fields", MappingProxyType(copied_fields))
        object.__setattr__(self, "labels", labels)

    def to_dict(self) -> dict[str, Any]:
        """Return fields and mandatory labels in one envelope."""

        return {
            "schema_version": self.schema_version,
            "fields": dict(self.fields),
            "sensitive_use_labels": [label.to_dict() for label in self.labels],
        }

    def to_json(self) -> str:
        """Serialize fields only inside the label-preserving envelope."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def label_sdoh_output(
    fields: Mapping[str, Any],
    *,
    labels: Iterable[SDOHSensitiveUseLabel] | None = None,
) -> LabeledSDOHExport:
    """Build a fail-closed export with one label for every field.

    When labels are omitted, the standard restrictive label is applied to each
    field. Supplying a partial custom label set is rejected.
    """

    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping")
    selected = (
        tuple(SDOHSensitiveUseLabel(field_name=key) for key in fields)
        if labels is None
        else tuple(labels)
    )
    return LabeledSDOHExport(fields=fields, labels=selected)


def serialize_labeled_sdoh_output(export: LabeledSDOHExport) -> str:
    """Serialize only a validated label-preserving SDOH export."""

    if not isinstance(export, LabeledSDOHExport):
        raise TypeError("export must be a LabeledSDOHExport")
    return export.to_json()


def _field_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("field names must be strings")
    if _SAFE_FIELD_RE.fullmatch(value) is None:
        raise ValueError("field names must use the safe schema-field format")
    return value


__all__ = [
    "SDOH_SENSITIVE_USE_SCHEMA_VERSION",
    "LabeledSDOHExport",
    "ProhibitedAutomatedUse",
    "SDOHPurpose",
    "SDOHSensitiveUseLabel",
    "label_sdoh_output",
    "serialize_labeled_sdoh_output",
]
