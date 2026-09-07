"""Dependency-free structural validation for OpenMed's FHIR R4 exports.

This validator checks the small base-R4 subset bundled with OpenMed. It is
deliberately separate from :mod:`profile_check`, which evaluates profiles from
a caller-supplied implementation-guide package. Both validators share only
generic traversal and coding primitives.

The public functions never raise because a resource is malformed. Findings use
FHIRPath-style locations and fixed, value-free messages so validation reports
do not echo clinical content or identifiers.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from importlib import resources
from typing import Any

from ._validation_primitives import _extract_codes, _Occurrence, _occurrence_groups

__all__ = [
    "BASE_R4_RESOURCE_TYPES",
    "ValidationFinding",
    "ValidationResult",
    "validate_bundle",
    "validate_resource",
]

BASE_R4_RESOURCE_TYPES = frozenset(
    {
        "AllergyIntolerance",
        "Condition",
        "DiagnosticReport",
        "Encounter",
        "Immunization",
        "MedicationStatement",
        "Observation",
        "Procedure",
    }
)


@dataclass(frozen=True)
class ValidationFinding:
    """One sanitized base-R4 validation finding.

    Attributes:
        severity: Either ``"error"`` or ``"warning"``.
        location: FHIRPath-style location of the malformed element.
        message: Value-free explanation safe to include in audit output.
        code: FHIR R4 ``issue-type`` code for OperationOutcome adaptation.
    """

    severity: str
    location: str
    message: str
    code: str = "invalid"

    @property
    def expression(self) -> str:
        """Return the location under the OperationOutcome-compatible name."""

        return self.location

    @property
    def diagnostics(self) -> str:
        """Return the message under the OperationOutcome-compatible name."""

        return self.message


@dataclass(frozen=True)
class ValidationResult:
    """Immutable errors and warnings produced by base-R4 validation."""

    errors: tuple[ValidationFinding, ...] = ()
    warnings: tuple[ValidationFinding, ...] = ()

    @property
    def findings(self) -> tuple[ValidationFinding, ...]:
        """Return every finding, with errors before warnings."""

        return self.errors + self.warnings

    @property
    def issues(self) -> tuple[ValidationFinding, ...]:
        """Return findings for the shared OperationOutcome adapter."""

        return self.findings

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when validation found no errors."""

        return not self.errors

    @property
    def valid(self) -> bool:
        """Alias for :attr:`is_valid` used by other validation results."""

        return self.is_valid


def validate_resource(resource: Mapping[str, Any]) -> ValidationResult:
    """Validate one resource against OpenMed's bundled FHIR R4 base subset.

    Supported resource types are Condition, Observation, MedicationStatement,
    Procedure, DiagnosticReport, AllergyIntolerance, Immunization, and
    Encounter. Other resource types produce a warning because no claim of base
    conformance can be made for a type outside the bundled subset.

    Args:
        resource: Resource-like input. Non-mappings and malformed mappings are
            surfaced as structured errors rather than raised exceptions.

    Returns:
        An immutable :class:`ValidationResult`. Messages never contain values
        read from ``resource``.
    """

    try:
        findings = _validate_resource_at(resource, root=None)
    except Exception:
        findings = [
            _error(
                "Resource",
                "Resource structure could not be inspected.",
                code="structure",
            )
        ]
    return _to_result(findings)


def validate_bundle(bundle: Mapping[str, Any]) -> ValidationResult:
    """Validate every resource contained in an exported FHIR R4 Bundle.

    The Bundle container is checked for its required type and entry shape. Each
    ``entry.resource`` is then checked independently and its findings retain a
    Bundle-qualified FHIRPath-style location.

    Args:
        bundle: Bundle-like input. Malformed input is always returned as
            findings and is never raised to the caller.

    Returns:
        Aggregated errors and warnings in deterministic entry order.
    """

    try:
        findings = _validate_bundle(bundle)
    except Exception:
        findings = [
            _error(
                "Bundle",
                "Bundle structure could not be inspected.",
                code="structure",
            )
        ]
    return _to_result(findings)


def _validate_bundle(bundle: Any) -> list[ValidationFinding]:
    if not isinstance(bundle, Mapping):
        return [_error("Bundle", "Bundle must be a JSON object.", code="structure")]

    findings: list[ValidationFinding] = []
    if bundle.get("resourceType") != "Bundle":
        findings.append(
            _error(
                "Bundle.resourceType",
                "resourceType must identify a FHIR Bundle.",
                code="value",
            )
        )

    bundle_type = bundle.get("type")
    if bundle_type is None or bundle_type == "":
        findings.append(
            _error("Bundle.type", "Required element is missing or empty.", "required")
        )
    elif not _matches_primitive(bundle_type, "code"):
        findings.append(
            _error("Bundle.type", "Element has an invalid FHIR R4 datatype.")
        )
    elif bundle_type not in _definitions()["bundleTypes"]:
        findings.append(
            _error(
                "Bundle.type",
                "Code is outside the required base R4 binding.",
                "code-invalid",
            )
        )

    entries = bundle.get("entry")
    if entries is None:
        return findings
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        findings.append(
            _error("Bundle.entry", "Bundle.entry must be an array.", "structure")
        )
        return findings

    for index, entry in enumerate(entries):
        root = f"Bundle.entry[{index}].resource"
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("resource"), Mapping
        ):
            findings.append(
                _error(
                    root,
                    "Bundle entry does not contain a FHIR resource.",
                    "required",
                )
            )
            continue
        findings.extend(_validate_resource_at(entry["resource"], root=root))
    return findings


def _validate_resource_at(
    resource: Any,
    *,
    root: str | None,
) -> list[ValidationFinding]:
    fallback_root = root or "Resource"
    if not isinstance(resource, Mapping):
        return [
            _error(
                fallback_root,
                "FHIR resource must be a JSON object.",
                code="structure",
            )
        ]

    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str) or not resource_type:
        return [
            _error(
                f"{fallback_root}.resourceType",
                "Required resourceType is missing or invalid.",
                code="required",
            )
        ]

    location_root = root or resource_type
    definition = _definitions()["resources"].get(resource_type)
    if definition is None:
        return [
            _warning(
                f"{location_root}.resourceType",
                "Resource type is outside the bundled base R4 validation subset.",
                code="not-supported",
            )
        ]

    elements = (*_definitions()["commonElements"], *definition["elements"])
    findings: list[ValidationFinding] = []
    for element in elements:
        findings.extend(_validate_element(resource, location_root, element))
    return findings


def _validate_element(
    resource: Mapping[str, Any],
    root: str,
    element: Mapping[str, Any],
) -> list[ValidationFinding]:
    path = element["path"]
    segments = tuple(path.split("."))
    groups = _occurrence_groups(resource, segments, root)
    minimum = element.get("min", 0)
    raw_maximum = element.get("max", "*")
    maximum = None if raw_maximum == "*" else int(raw_maximum)
    repeating = maximum is None or maximum > 1
    findings: list[ValidationFinding] = []

    for group in groups:
        count = len(group.occurrences)
        if count < minimum:
            findings.append(
                _error(
                    group.expression,
                    "Required element is missing or empty.",
                    "required",
                )
            )
        if maximum is not None and count > maximum:
            findings.append(
                _error(
                    group.expression,
                    "Maximum element cardinality is exceeded.",
                    "structure",
                )
            )

        if group.occurrences:
            represented_as_array = any(item.repeated for item in group.occurrences)
            if repeating and not represented_as_array:
                findings.append(
                    _error(
                        group.expression,
                        "Repeating element must use a JSON array.",
                        "structure",
                    )
                )
            elif not repeating and represented_as_array:
                findings.append(
                    _error(
                        group.expression,
                        "Single-valued element must not use a JSON array.",
                        "structure",
                    )
                )

        for occurrence in group.occurrences:
            if not _occurrence_matches_types(occurrence, path, element["types"]):
                findings.append(
                    _error(
                        occurrence.expression,
                        "Element has an invalid FHIR R4 datatype.",
                    )
                )
                continue
            binding_name = element.get("binding")
            if binding_name is not None and not _matches_binding(
                occurrence.value,
                _definitions()["valueSets"][binding_name],
            ):
                findings.append(
                    _error(
                        occurrence.expression,
                        "Code is outside the required base R4 binding.",
                        "code-invalid",
                    )
                )
    return findings


def _occurrence_matches_types(
    occurrence: _Occurrence,
    path: str,
    allowed_types: Sequence[str],
) -> bool:
    if path.endswith("[x]"):
        prefix = path.rsplit(".", 1)[-1][:-3]
        key = occurrence.key or ""
        if not key.startswith(prefix) or len(key) <= len(prefix):
            return False
        suffix = key[len(prefix) :]
        selected_type = suffix[0].lower() + suffix[1:]
        matching_type = next(
            (
                allowed
                for allowed in allowed_types
                if allowed.casefold() == selected_type.casefold()
            ),
            None,
        )
        return matching_type is not None and _matches_type(
            occurrence.value, matching_type
        )
    return any(_matches_type(occurrence.value, item) for item in allowed_types)


def _matches_type(value: Any, fhir_type: str) -> bool:
    if fhir_type in _COMPLEX_TYPES:
        return isinstance(value, Mapping)
    return _matches_primitive(value, fhir_type)


def _matches_primitive(value: Any, fhir_type: str) -> bool:
    if fhir_type == "boolean":
        return type(value) is bool
    if fhir_type == "integer":
        return type(value) is int and -(2**31) <= value < 2**31
    if fhir_type == "unsignedInt":
        return type(value) is int and 0 <= value < 2**31
    if fhir_type == "positiveInt":
        return type(value) is int and 0 < value < 2**31
    if fhir_type == "decimal":
        return type(value) in (int, float) and math.isfinite(value)
    if not isinstance(value, str) or not value:
        return False
    if fhir_type in {"string", "markdown", "xhtml"}:
        return bool(value)
    if fhir_type == "code":
        return _CODE_RE.fullmatch(value) is not None
    if fhir_type == "id":
        return _ID_RE.fullmatch(value) is not None
    if fhir_type == "date":
        return _valid_date(value)
    if fhir_type == "dateTime":
        return _valid_datetime(value, require_time=False)
    if fhir_type == "instant":
        return _valid_datetime(value, require_time=True)
    if fhir_type == "time":
        return _TIME_RE.fullmatch(value) is not None
    if fhir_type in {"uri", "url", "canonical"}:
        return not any(character.isspace() for character in value)
    if fhir_type == "oid":
        return _OID_RE.fullmatch(value) is not None
    if fhir_type == "uuid":
        return _UUID_RE.fullmatch(value) is not None
    if fhir_type == "base64Binary":
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            return False
        return True
    return False


def _valid_date(value: str) -> bool:
    match = _DATE_RE.fullmatch(value)
    if match is None:
        return False
    month = match.group("month")
    day = match.group("day")
    if month is None:
        return True
    if day is None:
        return True
    try:
        date(int(match.group("year")), int(month), int(day))
    except ValueError:
        return False
    return True


def _valid_datetime(value: str, *, require_time: bool) -> bool:
    if "T" not in value:
        return not require_time and _valid_date(value)
    if _DATETIME_RE.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _matches_binding(value: Any, value_set: Mapping[str, Any]) -> bool:
    allowed_codes = frozenset(value_set["codes"])
    expected_system = value_set.get("system", "")
    primitive = isinstance(value, str)
    for system, code in _extract_codes(value):
        if code not in allowed_codes:
            continue
        if primitive or not expected_system or system == expected_system:
            return True
    return False


def _to_result(findings: Sequence[ValidationFinding]) -> ValidationResult:
    return ValidationResult(
        errors=tuple(item for item in findings if item.severity == "error"),
        warnings=tuple(item for item in findings if item.severity == "warning"),
    )


def _error(
    location: str,
    message: str,
    code: str = "invalid",
) -> ValidationFinding:
    return ValidationFinding("error", location, message, code)


def _warning(
    location: str,
    message: str,
    code: str = "invalid",
) -> ValidationFinding:
    return ValidationFinding("warning", location, message, code)


@lru_cache(maxsize=1)
def _definitions() -> Mapping[str, Any]:
    definition_path = resources.files(__package__).joinpath(
        "definitions", "r4_base.json"
    )
    payload = json.loads(definition_path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != 1 or payload.get("fhirVersion") != "4.0.1":
        raise RuntimeError("bundled FHIR R4 constraint table is incompatible")
    return payload


_COMPLEX_TYPES = frozenset(
    {
        "Address",
        "Age",
        "Annotation",
        "Attachment",
        "BackboneElement",
        "CodeableConcept",
        "Coding",
        "ContactPoint",
        "Dosage",
        "Duration",
        "Extension",
        "HumanName",
        "Identifier",
        "Meta",
        "Narrative",
        "Period",
        "Quantity",
        "Range",
        "Ratio",
        "Reference",
        "Resource",
        "SampledData",
        "Signature",
        "Timing",
    }
)

_CODE_RE = re.compile(r"[^\s]+(?: [^\s]+)*")
_ID_RE = re.compile(r"[A-Za-z0-9\-.]{1,64}")
_DATE_RE = re.compile(
    r"(?P<year>[1-9]\d{3})(?:-(?P<month>0[1-9]|1[0-2])"
    r"(?:-(?P<day>0[1-9]|[12]\d|3[01]))?)?"
)
_DATETIME_RE = re.compile(
    r"[1-9]\d{3}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"T(?:[01]\d|2[0-3]):[0-5]\d:(?:[0-5]\d|60)(?:\.\d+)?"
    r"(?:Z|[+-](?:(?:0\d|1[0-3]):[0-5]\d|14:00))"
)
_TIME_RE = re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d:(?:[0-5]\d|60)(?:\.\d+)?")
_OID_RE = re.compile(r"urn:oid:[0-2](?:\.(?:0|[1-9]\d*))+", re.IGNORECASE)
_UUID_RE = re.compile(
    r"urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
    re.IGNORECASE,
)
