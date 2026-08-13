"""Offline validation of FHIR reference target types.

FHIR validates the shape of a relative reference separately from the type of
resource that a particular element is allowed to reference.  This module
provides a small, explicit target-type check for locally supplied R4 and R5
resources.  It never dereferences a URL, contacts a terminology server, or
attempts to discover resources outside the input collection.

Only fields listed in :data:`REFERENCE_TARGET_ALLOWLISTS` are checked.  That
intentional boundary keeps unknown profiles and resource fields non-blocking;
this helper is a conservative export check, not a complete FHIR validator.
Diagnostics contain a structural expression and a generic explanation only.
They never include a reference string, resource id, identifier, or other
resource value.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .operation_outcome import OperationOutcomeIssue, to_operation_outcome

__all__ = [
    "FHIR_R4_REFERENCE_TARGETS",
    "FHIR_R5_REFERENCE_TARGETS",
    "REFERENCE_TARGET_ALLOWLISTS",
    "ReferenceTargetIssue",
    "check_reference_targets",
    "find_reference_target_issues",
    "validate_fhir_reference_types",
    "validate_reference_targets",
    "validate_reference_types",
]


def _freeze_allowlist(
    values: Mapping[str, Mapping[str, Sequence[str]]],
) -> Mapping[str, Mapping[str, frozenset[str]]]:
    """Return a read-only, deterministic copy of a target-type map."""

    return MappingProxyType(
        {
            resource_type: MappingProxyType(
                {
                    field_path: frozenset(target_types)
                    for field_path, target_types in fields.items()
                }
            )
            for resource_type, fields in values.items()
        }
    )


# These maps deliberately use JSON element paths, not FHIRPath expressions.
# Array indexes are removed while walking a resource, so e.g.
# ``performer[0].actor`` matches ``performer.actor``.
FHIR_R4_REFERENCE_TARGETS = _freeze_allowlist(
    {
        "CarePlan": {
            "subject": ("Patient", "Group"),
            "encounter": ("Encounter",),
            "author": (
                "Patient",
                "Practitioner",
                "PractitionerRole",
                "Device",
                "RelatedPerson",
                "Organization",
            ),
            "careTeam": ("CareTeam",),
            "addresses": (
                "Condition",
                "Procedure",
                "MedicationStatement",
                "NutritionOrder",
                "ServiceRequest",
            ),
        },
        "Condition": {
            "subject": ("Patient", "Group"),
            "encounter": ("Encounter",),
            "recorder": ("Practitioner", "PractitionerRole"),
            "asserter": (
                "Patient",
                "RelatedPerson",
                "Practitioner",
                "PractitionerRole",
            ),
            "evidence.detail": (
                "Condition",
                "Observation",
                "AllergyIntolerance",
                "FamilyMemberHistory",
                "DiagnosticReport",
                "Procedure",
                "DocumentReference",
            ),
        },
        "DiagnosticReport": {
            "basedOn": (
                "CarePlan",
                "ImmunizationRecommendation",
                "MedicationRequest",
                "NutritionOrder",
                "ServiceRequest",
            ),
            "subject": ("Patient", "Group", "Device", "Location"),
            "encounter": ("Encounter",),
            "performer": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
                "CareTeam",
            ),
            "resultsInterpreter": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
                "CareTeam",
            ),
            "specimen": ("Specimen",),
            "result": ("Observation", "ImagingStudy", "DiagnosticReport"),
            "imagingStudy": ("ImagingStudy",),
        },
        "Device": {
            "patient": ("Patient",),
            "owner": ("Organization", "Patient", "RelatedPerson"),
            "location": ("Location",),
        },
        "DeviceRequest": {
            "subject": ("Patient", "Group"),
            "encounter": ("Encounter",),
            "requester": ("Device", "Practitioner", "PractitionerRole"),
            "performer": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
                "CareTeam",
                "HealthcareService",
                "Device",
            ),
            "reasonReference": (
                "Condition",
                "Observation",
                "DiagnosticReport",
                "DocumentReference",
            ),
            "priorRequest": ("DeviceRequest",),
            "insurance": ("Coverage",),
        },
        "DocumentReference": {
            "subject": ("Patient", "Group", "Practitioner", "Device"),
            "author": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
                "Device",
                "Patient",
                "RelatedPerson",
            ),
            "custodian": ("Organization",),
            "relatesTo.target": ("DocumentReference",),
        },
        "Encounter": {
            "subject": ("Patient", "Group"),
            "episodeOfCare": ("EpisodeOfCare",),
            "basedOn": ("CarePlan", "DeviceRequest", "ServiceRequest"),
            "participant.individual": (
                "Practitioner",
                "PractitionerRole",
                "RelatedPerson",
            ),
            "appointment": ("Appointment",),
            "reasonReference": (
                "Condition",
                "Procedure",
                "MedicationStatement",
                "ImmunizationRecommendation",
            ),
            "diagnosis.condition": ("Condition", "Procedure"),
            "location.location": ("Location",),
            "serviceProvider": ("Organization",),
            "partOf": ("Encounter",),
        },
        "EpisodeOfCare": {
            "patient": ("Patient",),
            "managingOrganization": ("Organization",),
            "careManager": ("Practitioner", "PractitionerRole"),
            "referralRequest": ("ServiceRequest",),
        },
        "FamilyMemberHistory": {
            "patient": ("Patient",),
        },
        "Group": {
            "member.entity": (
                "Patient",
                "Practitioner",
                "PractitionerRole",
                "Device",
                "Medication",
                "RelatedPerson",
            ),
            "managingEntity": (
                "Organization",
                "Practitioner",
                "PractitionerRole",
            ),
        },
        "ImagingStudy": {
            "subject": ("Patient", "Group", "Device"),
            "encounter": ("Encounter",),
            "referrer": ("Practitioner", "PractitionerRole"),
            "interpreter": ("Practitioner", "PractitionerRole"),
            "procedureReference": ("Procedure",),
            "endpoint": ("Endpoint",),
        },
        "MedicationAdministration": {
            "subject": ("Patient", "Group"),
            "context": ("Encounter", "EpisodeOfCare"),
            "performer.actor": (
                "Practitioner",
                "PractitionerRole",
                "Patient",
                "RelatedPerson",
                "Device",
            ),
            "medicationReference": ("Medication",),
            "request": ("MedicationRequest",),
            "reasonReference": (
                "Condition",
                "Observation",
                "DiagnosticReport",
            ),
            "eventHistory": ("Provenance",),
        },
        "MedicationDispense": {
            "subject": ("Patient", "Group"),
            "context": ("Encounter", "EpisodeOfCare"),
            "performer.actor": ("Practitioner", "PractitionerRole"),
            "authorizingPrescription": ("MedicationRequest",),
            "receiver": (
                "Patient",
                "Practitioner",
                "RelatedPerson",
            ),
        },
        "MedicationRequest": {
            "subject": ("Patient", "Group"),
            "encounter": ("Encounter",),
            "requester": (
                "Practitioner",
                "PractitionerRole",
                "Patient",
                "RelatedPerson",
                "Device",
            ),
            "performer": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
                "Device",
            ),
            "reasonReference": (
                "Condition",
                "Observation",
                "DiagnosticReport",
            ),
            "basedOn": ("CarePlan", "MedicationRequest", "ServiceRequest"),
            "priorPrescription": ("MedicationRequest",),
            "medicationReference": ("Medication",),
            "insurance": ("Coverage",),
            "detectedIssue": ("DetectedIssue",),
            "eventHistory": ("Provenance",),
        },
        "MedicationStatement": {
            "subject": ("Patient", "Group"),
            "context": ("Encounter", "EpisodeOfCare"),
            "informationSource": (
                "Patient",
                "Practitioner",
                "PractitionerRole",
                "RelatedPerson",
            ),
            "derivedFrom": (
                "MedicationRequest",
                "MedicationAdministration",
                "MedicationDispense",
            ),
            "basedOn": ("MedicationRequest", "CarePlan"),
            "partOf": (
                "MedicationAdministration",
                "MedicationDispense",
                "MedicationStatement",
            ),
            "reasonReference": (
                "Condition",
                "Observation",
                "DiagnosticReport",
            ),
        },
        "Observation": {
            "basedOn": (
                "CarePlan",
                "DeviceRequest",
                "ImmunizationRecommendation",
                "MedicationRequest",
                "NutritionOrder",
                "ServiceRequest",
            ),
            "partOf": (
                "MedicationAdministration",
                "MedicationDispense",
                "MedicationStatement",
                "Procedure",
                "Immunization",
                "ImagingStudy",
            ),
            "subject": (
                "Patient",
                "Group",
                "Device",
                "Location",
                "Practitioner",
                "Organization",
                "Procedure",
                "PlanDefinition",
                "Medication",
                "Substance",
            ),
            "encounter": ("Encounter",),
            "focus": ("Resource",),
            "performer": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
                "CareTeam",
                "Patient",
                "RelatedPerson",
            ),
            "specimen": ("Specimen",),
            "device": ("Device",),
            "derivedFrom": (
                "DocumentReference",
                "ImagingStudy",
                "Media",
                "QuestionnaireResponse",
                "Observation",
                "MolecularSequence",
            ),
            "hasMember": ("Observation", "QuestionnaireResponse", "MolecularSequence"),
        },
        "Patient": {
            "generalPractitioner": (
                "Organization",
                "Practitioner",
                "PractitionerRole",
            ),
            "managingOrganization": ("Organization",),
            "link.other": ("Patient",),
        },
        "Procedure": {
            "subject": ("Patient", "Group", "Device"),
            "encounter": ("Encounter",),
            "basedOn": ("CarePlan", "ServiceRequest"),
            "partOf": ("Procedure",),
            "performer.actor": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
                "Patient",
                "RelatedPerson",
                "Device",
            ),
            "location": ("Location",),
            "reasonReference": (
                "Condition",
                "Observation",
                "Procedure",
                "DiagnosticReport",
                "DocumentReference",
            ),
            "report": ("DiagnosticReport",),
            "focalDevice.manipulated": ("Device",),
            "usedReference": ("Device", "Medication", "Substance"),
        },
        "Provenance": {
            "target": (
                "Patient",
                "Practitioner",
                "PractitionerRole",
                "RelatedPerson",
                "Organization",
                "Device",
                "Location",
                "Encounter",
                "Condition",
                "Observation",
                "Procedure",
                "DiagnosticReport",
                "DocumentReference",
            ),
            "patient": ("Patient",),
            "agent.who": (
                "Practitioner",
                "PractitionerRole",
                "RelatedPerson",
                "Patient",
                "Device",
                "Organization",
            ),
            "agent.onBehalfOf": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
            ),
            "entity.what": (
                "Patient",
                "Practitioner",
                "Organization",
                "Device",
                "Location",
                "Encounter",
                "Condition",
                "Observation",
                "Procedure",
                "DiagnosticReport",
                "DocumentReference",
            ),
        },
        "ServiceRequest": {
            "subject": ("Patient", "Group", "Device", "Location"),
            "encounter": ("Encounter",),
            "requester": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
                "Patient",
                "RelatedPerson",
                "Device",
            ),
            "performer": (
                "Practitioner",
                "PractitionerRole",
                "Organization",
                "CareTeam",
                "HealthcareService",
                "Device",
            ),
            "reasonReference": (
                "Condition",
                "Observation",
                "DiagnosticReport",
                "DocumentReference",
            ),
            "basedOn": ("CarePlan", "ServiceRequest", "MedicationRequest"),
            "insurance": ("Coverage",),
            "supportingInfo": ("Resource",),
            "specimen": ("Specimen",),
        },
        "Specimen": {
            "subject": ("Patient", "Group", "Device"),
            "parent": ("Specimen",),
            "request": ("ServiceRequest",),
            "collection.collector": ("Practitioner", "PractitionerRole"),
            "collection.procedure": ("Procedure",),
        },
    }
)


def _r5_field_map(
    fields: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Adapt R4 field targets whose medication resource was renamed in R5."""

    return {
        field_path: tuple(
            sorted(
                "MedicationUsage" if target == "MedicationStatement" else target
                for target in target_types
            )
        )
        for field_path, target_types in fields.items()
    }


# R5 renamed MedicationStatement to MedicationUsage and changed several
# choice-element JSON names.  Keep those additions explicit rather than
# silently treating every R4 field as valid in R5.
FHIR_R5_REFERENCE_TARGETS = _freeze_allowlist(
    {
        **{
            resource_type: _r5_field_map(fields)
            for resource_type, fields in FHIR_R4_REFERENCE_TARGETS.items()
            if resource_type not in {"MedicationStatement"}
        },
        "MedicationRequest": {
            **dict(FHIR_R4_REFERENCE_TARGETS["MedicationRequest"]),
            "medication": ("Medication",),
        },
        "MedicationUsage": {
            "subject": ("Patient", "Group"),
            "context": ("Encounter", "EpisodeOfCare"),
            "informationSource": (
                "Patient",
                "Practitioner",
                "PractitionerRole",
                "RelatedPerson",
            ),
            "derivedFrom": (
                "MedicationRequest",
                "MedicationAdministration",
                "MedicationDispense",
            ),
            "basedOn": ("MedicationRequest", "CarePlan"),
            "partOf": (
                "MedicationAdministration",
                "MedicationDispense",
                "MedicationUsage",
            ),
            "reason": ("Condition", "Observation", "DiagnosticReport"),
        },
        "Observation": {
            **_r5_field_map(FHIR_R4_REFERENCE_TARGETS["Observation"]),
            "focus": ("Resource",),
        },
        "Procedure": {
            **dict(FHIR_R4_REFERENCE_TARGETS["Procedure"]),
            "reason": (
                "Condition",
                "Observation",
                "Procedure",
                "DiagnosticReport",
                "DocumentReference",
            ),
            "used": ("Device", "Medication", "Substance"),
        },
        "ServiceRequest": {
            **dict(FHIR_R4_REFERENCE_TARGETS["ServiceRequest"]),
            "reason": (
                "Condition",
                "Observation",
                "DiagnosticReport",
                "DocumentReference",
            ),
        },
    }
)

REFERENCE_TARGET_ALLOWLISTS = MappingProxyType(
    {"R4": FHIR_R4_REFERENCE_TARGETS, "R5": FHIR_R5_REFERENCE_TARGETS}
)

_INDEX_PATTERN = re.compile(r"\[\d+\]")
_RESOURCE_TYPE_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_URI_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

_FINDING_SPECS = {
    "missing": (
        "error",
        "not-found",
        "Referenced resource is missing from the local validation set.",
    ),
    "ambiguous": (
        "error",
        "multiple-matches",
        "Reference target is ambiguous in the local validation set.",
    ),
    "contained": (
        "information",
        "not-supported",
        "Contained references are not resolved by this local target check.",
    ),
    "disallowed": (
        "error",
        "structure",
        "Reference target type is not allowed for this field.",
    ),
}


@dataclass(frozen=True)
class ReferenceTargetIssue:
    """A value-free finding from the local reference target check."""

    kind: str
    expression: str

    @property
    def severity(self) -> str:
        """Return the FHIR ``OperationOutcome.issue`` severity."""

        return _FINDING_SPECS[self.kind][0]

    @property
    def code(self) -> str:
        """Return the FHIR ``OperationOutcome.issue`` code."""

        return _FINDING_SPECS[self.kind][1]

    @property
    def diagnostics(self) -> str:
        """Return a generic diagnostic that contains no resource values."""

        return _FINDING_SPECS[self.kind][2]

    @property
    def reason(self) -> str:
        """Compatibility alias for callers that use reason terminology."""

        return self.kind

    @property
    def path(self) -> str:
        """Compatibility alias for the structural expression."""

        return self.expression

    def to_operation_outcome_issue(self) -> OperationOutcomeIssue:
        """Convert this finding to the shared FHIR issue shape."""

        return OperationOutcomeIssue(
            severity=self.severity,
            code=self.code,
            diagnostics=self.diagnostics,
            expression=self.expression,
        )


@dataclass(frozen=True)
class _ResourceEntry:
    resource: Mapping[str, Any]
    expression: str
    resource_type: str
    resource_id: str | None
    full_url: str | None


def find_reference_target_issues(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    fhir_version: str = "R4",
    version: str | None = None,
) -> tuple[ReferenceTargetIssue, ...]:
    """Find local FHIR reference target-type issues.

    Args:
        resources: A FHIR resource, a FHIR ``Bundle``, or an ordered sequence
            of resources. Bundle entries may include ``fullUrl`` values, which
            lets the checker validate deterministic ``urn:uuid`` references
            emitted by :func:`to_bundle` without a server call.
        fhir_version: FHIR release to use, ``"R4"`` or ``"R5"``. The default
            is R4.
        version: Optional spelling alias for ``fhir_version``.

    Returns:
        A deterministic tuple of value-free findings. Only reference fields
        present in the explicit release map are inspected. Absolute external
        references and logical references using ``identifier`` are left
        untouched because resolving them would require outside knowledge.

    Raises:
        TypeError: If the input is not resource-shaped.
        ValueError: If the input has an invalid Bundle/resource shape or an
            unsupported FHIR release.
    """

    release = _resolve_version(fhir_version, version)
    entries = _resource_entries(resources)
    by_typed, by_id, by_full_url = _build_indexes(entries)
    findings: list[ReferenceTargetIssue] = []

    for entry in entries:
        field_targets = REFERENCE_TARGET_ALLOWLISTS[release].get(entry.resource_type)
        if field_targets is None:
            continue
        _walk_resource(
            entry.resource,
            root_expression=entry.expression,
            relative_path="",
            field_targets=field_targets,
            findings=findings,
            by_typed=by_typed,
            by_id=by_id,
            by_full_url=by_full_url,
        )
    return tuple(findings)


def validate_reference_targets(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    fhir_version: str = "R4",
    version: str | None = None,
) -> dict[str, Any]:
    """Return local reference target findings as a FHIR R4 OperationOutcome.

    The outcome is deterministic and contains no raw reference or resource
    values. ``information/not-supported`` findings identify contained
    references; missing, ambiguous, and disallowed targets are errors.
    """

    issues = find_reference_target_issues(
        resources,
        fhir_version=fhir_version,
        version=version,
    )
    return to_operation_outcome(issue.to_operation_outcome_issue() for issue in issues)


def check_reference_targets(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    fhir_version: str = "R4",
    version: str | None = None,
) -> dict[str, Any]:
    """Alias for :func:`validate_reference_targets`."""

    return validate_reference_targets(
        resources,
        fhir_version=fhir_version,
        version=version,
    )


def validate_reference_types(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    fhir_version: str = "R4",
    version: str | None = None,
) -> dict[str, Any]:
    """Compatibility alias for :func:`validate_reference_targets`."""

    return validate_reference_targets(
        resources,
        fhir_version=fhir_version,
        version=version,
    )


def validate_fhir_reference_types(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    fhir_version: str = "R4",
    version: str | None = None,
) -> dict[str, Any]:
    """Compatibility alias for :func:`validate_reference_targets`."""

    return validate_reference_targets(
        resources,
        fhir_version=fhir_version,
        version=version,
    )


def _resolve_version(fhir_version: str, version: str | None) -> str:
    primary = _normalise_version(fhir_version)
    if version is None:
        return primary
    alias = _normalise_version(version)
    if primary != "R4" and primary != alias:
        raise ValueError("fhir_version and version must select the same FHIR release")
    return alias


def _normalise_version(version: str) -> str:
    if not isinstance(version, str):
        raise ValueError("FHIR release must be R4 or R5")
    normalized = version.strip().upper()
    if normalized in {"R4", "4", "4.0", "4.0.1", "R4B", "4.3.0"}:
        return "R4"
    if normalized in {"R5", "5", "5.0", "5.0.0"}:
        return "R5"
    raise ValueError("FHIR release must be R4 or R5")


def _resource_entries(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[_ResourceEntry, ...]:
    if isinstance(resources, Mapping):
        if resources.get("resourceType") == "Bundle":
            return _bundle_entries(resources)
        return (
            _make_resource_entry(
                resources, expression=_resource_expression(resources, 0)
            ),
        )

    if isinstance(resources, (str, bytes)) or not isinstance(resources, Sequence):
        raise TypeError(
            "resources must be a FHIR resource, Bundle, or resource sequence"
        )

    entries = []
    for index, resource in enumerate(resources):
        if not isinstance(resource, Mapping):
            raise TypeError("resource sequence entries must be mappings")
        entries.append(_make_resource_entry(resource, expression=f"resources[{index}]"))
    return tuple(entries)


def _bundle_entries(bundle: Mapping[str, Any]) -> tuple[_ResourceEntry, ...]:
    raw_entries = bundle.get("entry", [])
    if raw_entries is None:
        return ()
    if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Sequence):
        raise ValueError("FHIR Bundle.entry must be a sequence")

    entries: list[_ResourceEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise TypeError("FHIR Bundle.entry items must be mappings")
        resource = raw_entry.get("resource")
        if resource is None:
            continue
        if not isinstance(resource, Mapping):
            raise TypeError("FHIR Bundle.entry.resource must be a mapping")
        entries.append(
            _make_resource_entry(
                resource,
                expression=f"Bundle.entry[{index}].resource",
                full_url=raw_entry.get("fullUrl"),
            )
        )
    return tuple(entries)


def _resource_expression(resource: Mapping[str, Any], index: int) -> str:
    resource_type = resource.get("resourceType")
    if isinstance(resource_type, str) and resource_type:
        return resource_type
    return f"resources[{index}]"


def _make_resource_entry(
    resource: Mapping[str, Any],
    *,
    expression: str,
    full_url: Any = None,
) -> _ResourceEntry:
    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str) or not resource_type.strip():
        raise ValueError("each FHIR resource must declare resourceType")
    resource_id = resource.get("id")
    if not isinstance(resource_id, str) or not resource_id:
        resource_id = None
    normalized_full_url = full_url if isinstance(full_url, str) and full_url else None
    return _ResourceEntry(
        resource=resource,
        expression=expression,
        resource_type=resource_type,
        resource_id=resource_id,
        full_url=normalized_full_url,
    )


def _build_indexes(
    entries: Sequence[_ResourceEntry],
) -> tuple[
    dict[tuple[str, str], list[_ResourceEntry]],
    dict[str, list[_ResourceEntry]],
    dict[str, list[_ResourceEntry]],
]:
    by_typed: dict[tuple[str, str], list[_ResourceEntry]] = {}
    by_id: dict[str, list[_ResourceEntry]] = {}
    by_full_url: dict[str, list[_ResourceEntry]] = {}
    for entry in entries:
        if entry.resource_id is not None:
            by_typed.setdefault((entry.resource_type, entry.resource_id), []).append(
                entry
            )
            by_id.setdefault(entry.resource_id, []).append(entry)
        if entry.full_url is not None:
            by_full_url.setdefault(entry.full_url, []).append(entry)
    return by_typed, by_id, by_full_url


def _walk_resource(
    node: Any,
    *,
    root_expression: str,
    relative_path: str,
    field_targets: Mapping[str, frozenset[str]],
    findings: list[ReferenceTargetIssue],
    by_typed: Mapping[tuple[str, str], list[_ResourceEntry]],
    by_id: Mapping[str, list[_ResourceEntry]],
    by_full_url: Mapping[str, list[_ResourceEntry]],
) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _walk_resource(
                item,
                root_expression=root_expression,
                relative_path=f"{relative_path}[{index}]",
                field_targets=field_targets,
                findings=findings,
                by_typed=by_typed,
                by_id=by_id,
                by_full_url=by_full_url,
            )
        return
    if not isinstance(node, Mapping):
        return

    for key, value in node.items():
        if key == "resourceType":
            continue
        child_path = f"{relative_path}.{key}" if relative_path else key
        field_path = _INDEX_PATTERN.sub("", child_path)
        target_types = field_targets.get(field_path)
        expression = f"{root_expression}.{child_path}"
        if target_types is not None:
            _inspect_reference(
                value,
                expression=expression,
                allowed_types=target_types,
                findings=findings,
                by_typed=by_typed,
                by_id=by_id,
                by_full_url=by_full_url,
            )
            continue
        _walk_resource(
            value,
            root_expression=root_expression,
            relative_path=child_path,
            field_targets=field_targets,
            findings=findings,
            by_typed=by_typed,
            by_id=by_id,
            by_full_url=by_full_url,
        )


def _inspect_reference(
    value: Any,
    *,
    expression: str,
    allowed_types: frozenset[str],
    findings: list[ReferenceTargetIssue],
    by_typed: Mapping[tuple[str, str], list[_ResourceEntry]],
    by_id: Mapping[str, list[_ResourceEntry]],
    by_full_url: Mapping[str, list[_ResourceEntry]],
) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _inspect_reference(
                item,
                expression=f"{expression}[{index}]",
                allowed_types=allowed_types,
                findings=findings,
                by_typed=by_typed,
                by_id=by_id,
                by_full_url=by_full_url,
            )
        return

    if value is None:
        findings.append(ReferenceTargetIssue("missing", expression))
        return
    if not isinstance(value, Mapping):
        findings.append(ReferenceTargetIssue("missing", expression))
        return

    if "reference" not in value:
        # A logical Reference is intentionally outside local target resolution.
        # R5 CodeableReference fields can carry a concept instead of a
        # reference; terminology validation is outside this helper as well.
        if "identifier" in value or "concept" in value:
            return
        findings.append(ReferenceTargetIssue("missing", expression))
        return

    reference = value.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        findings.append(ReferenceTargetIssue("missing", expression))
        return
    reference = reference.strip()

    if reference.startswith("#"):
        findings.append(ReferenceTargetIssue("contained", expression))
        return

    type_hint = value.get("type")
    if not isinstance(type_hint, str) or not _RESOURCE_TYPE_PATTERN.fullmatch(
        type_hint
    ):
        type_hint = None

    if reference.startswith("urn:uuid:"):
        candidates = by_full_url.get(reference, [])
        _check_candidates(
            candidates,
            allowed_types=allowed_types,
            type_hint=type_hint,
            expression=expression,
            findings=findings,
        )
        return

    # Absolute references are valid FHIR references, but resolving them would
    # require external state. This check is intentionally local-only.
    if _URI_SCHEME_PATTERN.match(reference):
        return

    target_type, target_id = _split_relative_reference(reference)
    target_type = target_type or type_hint
    if (
        target_type is not None
        and "Resource" not in allowed_types
        and target_type not in allowed_types
    ):
        findings.append(ReferenceTargetIssue("disallowed", expression))
        return
    if target_id is None:
        findings.append(ReferenceTargetIssue("missing", expression))
        return

    candidates = (
        by_typed.get((target_type, target_id), [])
        if target_type is not None
        else by_id.get(target_id, [])
    )
    _check_candidates(
        candidates,
        allowed_types=allowed_types,
        type_hint=target_type,
        expression=expression,
        findings=findings,
    )


def _check_candidates(
    candidates: Sequence[_ResourceEntry],
    *,
    allowed_types: frozenset[str],
    type_hint: str | None,
    expression: str,
    findings: list[ReferenceTargetIssue],
) -> None:
    if not candidates:
        findings.append(ReferenceTargetIssue("missing", expression))
        return
    if len(candidates) != 1:
        findings.append(ReferenceTargetIssue("ambiguous", expression))
        return

    target_type = candidates[0].resource_type
    if ("Resource" not in allowed_types and target_type not in allowed_types) or (
        type_hint is not None and type_hint != "Resource" and type_hint != target_type
    ):
        findings.append(ReferenceTargetIssue("disallowed", expression))


def _split_relative_reference(reference: str) -> tuple[str | None, str | None]:
    parts = reference.split("/")
    if len(parts) >= 2 and _RESOURCE_TYPE_PATTERN.fullmatch(parts[0]) and parts[1]:
        return parts[0], parts[1]
    if len(parts) == 1 and parts[0]:
        return None, parts[0]
    return None, None
