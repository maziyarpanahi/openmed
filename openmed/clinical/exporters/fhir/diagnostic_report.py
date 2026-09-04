"""Conservative FHIR R4/R5 DiagnosticReport projection.

Deterministic, offline, privacy-safe projection for synthetic report
mappings. Status is explicit with ``"unknown"`` for missing/empty values;
invalid status values are rejected fail-closed. The emitted resource is
validated against the R4/R5 union allowlist (field-name allowlisting) so
only known fields are emitted, and no inference is performed from
``conclusion`` to ``conclusionCode``. Scalar fields are type-checked
fail-closed (e.g. ``conclusion`` must be string, ``effectivePeriod`` must
be mapping) but no semantic profile validation is performed.

No network, telemetry, or wall-clock dependency is introduced; ``issued``
is caller-supplied only when present.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

__all__ = [
    "to_diagnostic_report",
    "DIAGNOSTIC_REPORT_STATUSES",
    "DIAGNOSTIC_REPORT_STATUS_UNKNOWN",
]

DIAGNOSTIC_REPORT_STATUS_UNKNOWN: str = "unknown"

DIAGNOSTIC_REPORT_STATUSES: frozenset[str] = frozenset(
    {
        "registered",
        "partial",
        "preliminary",
        "final",
        "amended",
        "corrected",
        "appended",
        "cancelled",
        "entered-in-error",
        "unknown",
    }
)

_BASE_FIELDS: frozenset[str] = frozenset(
    {
        "resourceType",
        "id",
        "meta",
        "implicitRules",
        "language",
        "text",
        "contained",
        "extension",
        "modifierExtension",
    }
)

_RESOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "identifier",
        "basedOn",
        "status",
        "category",
        "code",
        "subject",
        "encounter",
        "effectiveDateTime",
        "effectivePeriod",
        "issued",
        "performer",
        "resultsInterpreter",
        "specimen",
        "result",
        "imagingStudy",
        "study",
        "media",
        "composition",
        "conclusion",
        "conclusionCode",
        "presentedForm",
        "note",
        "supportingInfo",
    }
)

# Public alias matching the plan's R4/R5 union name.
DIAGNOSTIC_REPORT_FIELDS_R4R5: frozenset[str] = _RESOURCE_FIELDS

_ALLOWED_FIELDS: frozenset[str] = _BASE_FIELDS | _RESOURCE_FIELDS

# Input keys that the conservative exporter copies verbatim when present.
# This is exactly the optional-keys API documented for ``report`` plus the
# FHIR ``subject`` reference (supplied via arg or mapping) and base ``text``.
_DIRECT_COPY_KEYS: tuple[str, ...] = (
    "meta",
    "implicitRules",
    "language",
    "contained",
    "extension",
    "modifierExtension",
    "identifier",
    "basedOn",
    "category",
    "encounter",
    "effectiveDateTime",
    "effectivePeriod",
    "issued",
    "performer",
    "resultsInterpreter",
    "specimen",
    "result",
    "imagingStudy",
    "study",
    "media",
    "composition",
    "conclusion",
    "conclusionCode",
    "presentedForm",
    "note",
    "supportingInfo",
    "text",
)

_LIST_FIELDS: frozenset[str] = frozenset(
    {
        "identifier",
        "contained",
        "extension",
        "modifierExtension",
        "basedOn",
        "category",
        "performer",
        "resultsInterpreter",
        "specimen",
        "result",
        "imagingStudy",
        "study",
        "media",
        "conclusionCode",
        "presentedForm",
        "note",
        "supportingInfo",
    }
)

# Scalar fields that must be string when present (conservative type gate).
_STRING_FIELDS: frozenset[str] = frozenset(
    {"conclusion", "effectiveDateTime", "implicitRules", "issued", "language"}
)

# Scalar fields that must be mapping when present.
_MAPPING_FIELDS: frozenset[str] = frozenset({"effectivePeriod", "meta", "text"})

# Reference fields that accept string (normalized to {"reference": ...}) or mapping.
_REFERENCE_FIELDS: frozenset[str] = frozenset({"encounter", "composition"})


def _normalize_status(value: Any) -> str:
    """Normalize ``status`` with explicit unknown for missing/empty.

    Args:
        value: Raw ``status`` from the report mapping.

    Returns:
        A lower-cased FHIR DiagnosticReport status code. Missing or empty
        string returns ``"unknown"``.

    Raises:
        ValueError: If ``value`` is not a string-like status in the
            allowlist. The message references only the field name.
    """
    if value is None:
        return DIAGNOSTIC_REPORT_STATUS_UNKNOWN
    if not isinstance(value, str):
        raise ValueError("invalid value for field 'status'")
    stripped = value.strip()
    if not stripped:
        return DIAGNOSTIC_REPORT_STATUS_UNKNOWN
    normalized = stripped.casefold()
    if normalized not in DIAGNOSTIC_REPORT_STATUSES:
        raise ValueError("invalid value for field 'status'")
    return normalized


def _build_code(value: Any) -> dict[str, Any]:
    """Build the DiagnosticReport ``code`` element.

    Args:
        value: Raw ``code`` value from the report mapping.

    Returns:
        A CodeableConcept dict. Missing or empty mapping returns the
        synthetic default.

    Raises:
        ValueError: If ``value`` is present but not a mapping.
    """
    if value is None:
        return {"text": "synthetic diagnostic report"}
    if not isinstance(value, Mapping):
        raise ValueError("invalid value for field 'code'")
    if not value:
        return {"text": "synthetic diagnostic report"}
    return copy.deepcopy(dict(value))


def _validate_shape(resource: Mapping[str, Any]) -> None:
    """Validate that ``resource`` contains only allowlisted fields.

    Checks the 32-key union (23 resource + 9 base) and rejects any extra
    key fail-closed.

    Args:
        resource: The assembled DiagnosticReport mapping.

    Raises:
        ValueError: If any unsupported field is present. The message is a
            fixed category and never includes the rejected key.
    """
    for key in resource:
        if key not in _ALLOWED_FIELDS:
            raise ValueError("DiagnosticReport contains an unsupported field")


def _validate_input_fields(report: Mapping[Any, Any]) -> None:
    """Reject unknown input keys before inspecting or copying their values."""

    try:
        keys = tuple(report.keys())
    except Exception:
        raise ValueError("DiagnosticReport fields could not be read") from None
    if any(type(key) is not str or key not in _ALLOWED_FIELDS for key in keys):
        raise ValueError("DiagnosticReport contains an unsupported field")


def _deep_copy_value(value: Any) -> Any:
    """Deterministic deep copy for evidence preservation."""
    return copy.deepcopy(value)


def to_diagnostic_report(
    report: Mapping[str, Any],
    *,
    report_id: str | None = None,
    subject_reference: str | None = None,
    doc_id: str = "openmed-document",
) -> dict[str, Any]:
    """Build a conservative FHIR R4/R5 DiagnosticReport.

    Deterministic and offline. No network call, no nondeterminism, no wall-clock.
    ``issued`` is emitted only when supplied by the caller.

    Args:
        report: Synthetic report mapping with optional keys: ``status``,
            ``category``, ``code``, ``encounter``, ``effectiveDateTime``,
            ``effectivePeriod``, ``issued``, ``performer``,
            ``resultsInterpreter``, ``specimen``, ``result``,
            ``imagingStudy``, ``study``, ``media``, ``conclusion``,
            ``conclusionCode``, ``presentedForm``, ``note``,
            ``supportingInfo``, ``composition``, ``text``,
            ``identifier``, ``basedOn``, and ``subject``.
            ``result`` and ``supportingInfo`` are list[Reference] preserved
            as list[dict] with stable ordering; ``presentedForm`` is
            list[dict] Attachment preserved verbatim with deep copy.
        report_id: Optional resource ``id``. When provided, emitted as
            ``DiagnosticReport.id``. If ``report`` contains ``"id"`` and
            ``report_id`` is not provided, ``report["id"]`` is used when
            non-empty string.
        subject_reference: Optional ``DiagnosticReport.subject`` reference
            (e.g. ``"Patient/synthetic"``). Takes precedence over
            ``report["subject"]`` when both are present.
        doc_id: Stable document identifier. Accepted for deterministic
            Bundle compatibility; not otherwise used in the projection.

    Returns:
        A ``resourceType="DiagnosticReport"`` mapping validated against the
        R4+R5 union allowlist. Union allowlisting means a resource with
        ``study`` (R5) or ``imagingStudy`` (R4) or both passes the field-name
        gate; per-version profile validation is an opt-in layer outside this
        helper.

    Raises:
        TypeError: If ``report`` is not a mapping.
        ValueError: If ``status`` is invalid, ``code`` is not a mapping,
            a list field is not a list, a string field is not a string,
            a mapping field is not a mapping, or the shape contains an
            unsupported field. Messages reference only field names.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    _validate_input_fields(report)
    if "resourceType" in report and report["resourceType"] != "DiagnosticReport":
        raise ValueError("invalid value for field 'resourceType'")
    # doc_id reserved for deterministic Bundle fullUrl seeding; not used in
    # standalone projection to keep output byte-stable (see bundle.py).
    del doc_id  # noqa: F841

    status = _normalize_status(report.get("status"))
    code = _build_code(report.get("code"))

    resource: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "status": status,
        "code": code,
    }

    # report_id takes precedence over report["id"]; otherwise honor report["id"].
    # C9: id fields are untrusted — trim, length-bound (≤200), fail closed.
    # report["id"] whitespace is silently ignored (preserves existing test),
    # while explicit report_id/subject_reference whitespace is rejected below.
    effective_report_id = report_id
    if effective_report_id is None and "id" in report and report["id"] is not None:
        effective_report_id = report["id"]
    if effective_report_id is not None:
        if isinstance(effective_report_id, bool) or not isinstance(
            effective_report_id, str
        ):
            raise ValueError("invalid value for field 'report_id'")
        stripped_id = effective_report_id.strip()
        if stripped_id:
            if len(stripped_id) > 200:
                raise ValueError("invalid value for field 'report_id'")
            resource["id"] = stripped_id
        elif report_id is not None and effective_report_id.strip() == "":
            # Explicit report_id param that is whitespace-only → fail closed
            # (mirrors subject_reference parity). report["id"] whitespace
            # remains silent per existing contract.
            raise ValueError("invalid value for field 'report_id'")

    if subject_reference is not None:
        if isinstance(subject_reference, bool) or not isinstance(
            subject_reference, str
        ):
            raise ValueError("invalid value for field 'subject_reference'")
        stripped_ref = subject_reference.strip()
        if not stripped_ref or len(stripped_ref) > 512:
            raise ValueError("invalid value for field 'subject_reference'")
        resource["subject"] = {"reference": stripped_ref}
    elif "subject" in report and report["subject"] is not None:
        subject_val = report["subject"]
        if isinstance(subject_val, bool):
            raise ValueError("invalid value for field 'subject'")
        if isinstance(subject_val, Mapping):
            resource["subject"] = _deep_copy_value(dict(subject_val))
        elif isinstance(subject_val, str):
            stripped = subject_val.strip()
            if not stripped or len(stripped) > 512:
                raise ValueError("invalid value for field 'subject'")
            resource["subject"] = {"reference": stripped}
        else:
            raise ValueError("invalid value for field 'subject'")

    for key in _DIRECT_COPY_KEYS:
        if key not in report:
            continue
        value = report[key]
        if value is None:
            continue
        if key in _LIST_FIELDS:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"invalid value for field '{key}'")
            if any(not isinstance(item, Mapping) for item in value):
                raise ValueError(f"invalid value for field '{key}'")
            # FHIR repeating elements are complex records, never raw scalars.
            resource[key] = [_deep_copy_value(dict(item)) for item in value]
        elif key in _REFERENCE_FIELDS:
            if isinstance(value, bool):
                raise ValueError(f"invalid value for field '{key}'")
            if isinstance(value, Mapping):
                resource[key] = _deep_copy_value(dict(value))
            elif isinstance(value, str):
                stripped = value.strip()
                if not stripped or len(stripped) > 512:
                    raise ValueError(f"invalid value for field '{key}'")
                resource[key] = {"reference": stripped}
            else:
                raise ValueError(f"invalid value for field '{key}'")
        elif key in _MAPPING_FIELDS:
            if not isinstance(value, Mapping):
                raise ValueError(f"invalid value for field '{key}'")
            resource[key] = _deep_copy_value(dict(value))
        elif key in _STRING_FIELDS:
            if not isinstance(value, str):
                raise ValueError(f"invalid value for field '{key}'")
            resource[key] = value
        else:
            # Fallback for other scalar/dict fields: shallow type gate.
            if isinstance(value, Mapping):
                resource[key] = _deep_copy_value(dict(value))
            elif isinstance(value, (list, tuple)):
                # Defensive: caller passed list for scalar field.
                raise ValueError(f"invalid value for field '{key}'")
            else:
                resource[key] = _deep_copy_value(value)

    if "effectiveDateTime" in resource and "effectivePeriod" in resource:
        raise ValueError("invalid value for field 'effectivePeriod'")

    _validate_shape(resource)
    return resource
