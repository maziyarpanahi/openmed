"""Explicit, loss-aware FHIR R4/R5 conversion for OpenMed resources.

The exchange workbench deliberately does not infer a FHIR release from JSON.
FHIR JSON has no required release marker, and guessing one makes a conversion
boundary impossible to audit. Callers therefore provide both the source and
target release explicitly.

Only the resource subset used by OpenMed's clinical exchange helpers is
accepted. Fields that have no safe representation in the target release are
either carried in a documented preservation extension or rejected with the
resource path. No input mapping is mutated.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

__all__ = [
    "FHIRConversionError",
    "FHIRVersion",
    "FHIR_R4",
    "FHIR_R5",
    "FHIRVersionAdapter",
    "FHIRVersionError",
    "SUPPORTED_RESOURCE_TYPES",
    "UnsupportedFHIRFieldError",
    "UnsupportedFHIRField",
    "VersionAdapter",
    "convert_bundle",
    "convert_resource",
    "parse_fhir_version",
    "r4_to_r5",
    "r5_to_r4",
]


class FHIRVersion(str, Enum):
    """FHIR releases supported by the exchange workbench."""

    R4 = "R4"
    R5 = "R5"

    @property
    def specification_version(self) -> str:
        """Return the exact published core specification version."""

        return {self.R4: "4.0.1", self.R5: "5.0.0"}[self]


FHIR_R4 = FHIRVersion.R4
FHIR_R5 = FHIRVersion.R5


class FHIRConversionError(ValueError):
    """Base error for a rejected FHIR version conversion."""


class FHIRVersionError(FHIRConversionError):
    """Raised when a caller supplies an unsupported FHIR release."""


class UnsupportedFHIRFieldError(FHIRConversionError):
    """Raised when a field cannot be represented across the release boundary."""

    def __init__(
        self,
        path: str,
        *,
        source_version: FHIRVersion,
        target_version: FHIRVersion,
        reason: str = "field is not supported by the target release",
    ) -> None:
        self.path = path
        self.source_version = source_version
        self.target_version = target_version
        self.reason = reason
        super().__init__(
            f"Unsupported cross-version field at {path} "
            f"({source_version.value}->{target_version.value}): {reason}"
        )


# The set is intentionally finite. Related resources are included when they
# are needed to preserve references, narrative, or provenance in a document.
SUPPORTED_RESOURCE_TYPES = frozenset(
    {
        "AllergyIntolerance",
        "AuditEvent",
        "Bundle",
        "CarePlan",
        "Composition",
        "Condition",
        "Device",
        "DiagnosticReport",
        "DocumentReference",
        "Encounter",
        "Goal",
        "Immunization",
        "Medication",
        "MedicationRequest",
        "MedicationStatement",
        "Observation",
        "Organization",
        "Patient",
        "Procedure",
        "Provenance",
        "Practitioner",
        "PractitionerRole",
        "RelatedPerson",
        "ServiceRequest",
        "Specimen",
    }
)

_BASE_RESOURCE_FIELDS = frozenset(
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

# These are the union of the R4 and R5 top-level fields for the supported
# resource subset. The union is used only to detect an unknown field at the
# conversion boundary; resource cardinality and terminology are handled by
# the local validator.
_RESOURCE_FIELDS: dict[str, frozenset[str]] = {
    "Patient": frozenset(
        {
            "identifier",
            "active",
            "name",
            "telecom",
            "gender",
            "birthDate",
            "deceasedBoolean",
            "deceasedDateTime",
            "address",
            "maritalStatus",
            "multipleBirthBoolean",
            "multipleBirthInteger",
            "photo",
            "contact",
            "communication",
            "generalPractitioner",
            "managingOrganization",
            "link",
        }
    ),
    "Composition": frozenset(
        {
            "identifier",
            "status",
            "type",
            "category",
            "subject",
            "encounter",
            "date",
            "author",
            "name",
            "title",
            "confidentiality",
            "attester",
            "custodian",
            "relatesTo",
            "event",
            "section",
        }
    ),
    "Bundle": frozenset(
        {
            "identifier",
            "type",
            "timestamp",
            "total",
            "link",
            "entry",
            "signature",
            "issues",
        }
    ),
    "DocumentReference": frozenset(
        {
            "masterIdentifier",
            "identifier",
            "status",
            "docStatus",
            "type",
            "category",
            "subject",
            "date",
            "author",
            "authenticator",
            "custodian",
            "relatesTo",
            "description",
            "securityLabel",
            "content",
            "context",
        }
    ),
    "Condition": frozenset(
        {
            "identifier",
            "clinicalStatus",
            "verificationStatus",
            "category",
            "severity",
            "code",
            "bodySite",
            "subject",
            "encounter",
            "onsetDateTime",
            "onsetAge",
            "onsetPeriod",
            "onsetRange",
            "onsetString",
            "abatementDateTime",
            "abatementAge",
            "abatementPeriod",
            "abatementRange",
            "abatementString",
            "recordedDate",
            "recorder",
            "asserter",
            "stage",
            "evidence",
            "note",
        }
    ),
    "Observation": frozenset(
        {
            "identifier",
            "basedOn",
            "partOf",
            "status",
            "category",
            "code",
            "subject",
            "focus",
            "encounter",
            "effectiveDateTime",
            "effectivePeriod",
            "effectiveTiming",
            "effectiveInstant",
            "issued",
            "performer",
            "valueInteger",
            "valueDecimal",
            "valueBoolean",
            "valueQuantity",
            "valueRange",
            "valueRatio",
            "valueSampledData",
            "valueTime",
            "valueDateTime",
            "valuePeriod",
            "valueString",
            "valueCodeableConcept",
            "valueAttachment",
            "valueReference",
            "dataAbsentReason",
            "interpretation",
            "note",
            "bodySite",
            "method",
            "specimen",
            "device",
            "referenceRange",
            "hasMember",
            "derivedFrom",
            "component",
        }
    ),
    "MedicationStatement": frozenset(
        {
            "identifier",
            "basedOn",
            "partOf",
            "status",
            "statusReason",
            "category",
            "medicationCodeableConcept",
            "medicationReference",
            "medication",
            "subject",
            "context",
            "encounter",
            "effectiveDateTime",
            "effectivePeriod",
            "effectiveTiming",
            "dateAsserted",
            "informationSource",
            "derivedFrom",
            "reasonCode",
            "reasonReference",
            "reason",
            "note",
            "dosage",
            "renderedDosageInstruction",
            "relatedClinicalInformation",
            "adherence",
        }
    ),
    "AllergyIntolerance": frozenset(
        {
            "identifier",
            "clinicalStatus",
            "verificationStatus",
            "type",
            "category",
            "criticality",
            "code",
            "patient",
            "encounter",
            "onsetDateTime",
            "onsetAge",
            "onsetPeriod",
            "onsetRange",
            "onsetString",
            "recordedDate",
            "recorder",
            "asserter",
            "lastOccurrence",
            "note",
            "reaction",
        }
    ),
    "Procedure": frozenset(
        {
            "identifier",
            "instantiatesCanonical",
            "instantiatesUri",
            "basedOn",
            "partOf",
            "status",
            "statusReason",
            "category",
            "code",
            "subject",
            "encounter",
            "performedDateTime",
            "performedPeriod",
            "performedString",
            "performedAge",
            "recorder",
            "asserter",
            "performer",
            "location",
            "reasonCode",
            "reasonReference",
            "bodySite",
            "outcome",
            "report",
            "complication",
            "followUp",
            "note",
            "focalDevice",
            "usedReference",
            "usedCodeableConcept",
            "used",
        }
    ),
    "Provenance": frozenset(
        {
            "target",
            "occurredPeriod",
            "occurredDateTime",
            "recorded",
            "policy",
            "location",
            "authorization",
            "activity",
            "basedOn",
            "encounter",
            "agent",
            "entity",
            "signature",
        }
    ),
    "AuditEvent": frozenset(
        {
            "type",
            "subtype",
            "action",
            "period",
            "recorded",
            "outcome",
            "outcomeDesc",
            "purposeOfEvent",
            "agent",
            "source",
            "entity",
        }
    ),
    "DiagnosticReport": frozenset(
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
            "media",
            "conclusion",
            "conclusionCode",
            "presentedForm",
        }
    ),
    "Encounter": frozenset(
        {
            "identifier",
            "status",
            "statusHistory",
            "class",
            "classHistory",
            "type",
            "serviceType",
            "priority",
            "subject",
            "episodeOfCare",
            "basedOn",
            "participant",
            "appointment",
            "period",
            "length",
            "reasonCode",
            "reasonReference",
            "diagnosis",
            "account",
            "hospitalization",
            "location",
        }
    ),
    "Practitioner": frozenset(
        {
            "identifier",
            "active",
            "name",
            "telecom",
            "address",
            "gender",
            "birthDate",
            "photo",
            "qualification",
            "communication",
        }
    ),
    "PractitionerRole": frozenset(
        {
            "identifier",
            "active",
            "period",
            "practitioner",
            "organization",
            "code",
            "specialty",
            "location",
            "healthcareService",
            "telecom",
            "availableTime",
            "notAvailable",
            "availabilityExceptions",
            "endpoint",
        }
    ),
    "Organization": frozenset(
        {
            "identifier",
            "active",
            "type",
            "name",
            "alias",
            "telecom",
            "address",
            "partOf",
            "contact",
            "endpoint",
        }
    ),
    "RelatedPerson": frozenset(
        {
            "identifier",
            "active",
            "patient",
            "relationship",
            "name",
            "telecom",
            "gender",
            "birthDate",
            "address",
            "photo",
            "period",
            "communication",
        }
    ),
    "Device": frozenset(
        {
            "identifier",
            "definition",
            "udiCarrier",
            "status",
            "statusReason",
            "distinctIdentifier",
            "manufacturer",
            "manufactureDate",
            "expirationDate",
            "lotNumber",
            "serialNumber",
            "name",
            "modelNumber",
            "partNumber",
            "type",
            "specialization",
            "version",
            "property",
            "patient",
            "owner",
            "contact",
            "location",
            "url",
            "note",
            "safety",
            "parent",
        }
    ),
    "Medication": frozenset(
        {
            "identifier",
            "code",
            "status",
            "marketingAuthorizationHolder",
            "doseForm",
            "amount",
            "ingredient",
            "batch",
            "package",
            "manufacturer",
        }
    ),
    "MedicationRequest": frozenset(
        {
            "identifier",
            "status",
            "statusReason",
            "intent",
            "category",
            "priority",
            "doNotPerform",
            "reportedBoolean",
            "reportedReference",
            "medicationCodeableConcept",
            "medicationReference",
            "medication",
            "subject",
            "encounter",
            "supportingInformation",
            "authoredOn",
            "requester",
            "recorder",
            "reasonCode",
            "reasonReference",
            "instantiatesCanonical",
            "instantiatesUri",
            "basedOn",
            "groupIdentifier",
            "courseOfTherapyType",
            "insurance",
            "note",
            "dosageInstruction",
            "dispenseRequest",
            "substitution",
            "priorPrescription",
            "detectedIssue",
            "eventHistory",
        }
    ),
    "Immunization": frozenset(
        {
            "identifier",
            "status",
            "statusReason",
            "vaccineCode",
            "patient",
            "encounter",
            "occurrenceDateTime",
            "occurrenceString",
            "recorded",
            "primarySource",
            "reportOrigin",
            "location",
            "manufacturer",
            "lotNumber",
            "expirationDate",
            "performer",
            "note",
            "reasonCode",
            "reasonReference",
            "isSubpotent",
            "subpotentReason",
            "education",
            "programEligibility",
            "fundingSource",
            "reaction",
            "protocolApplied",
        }
    ),
    "CarePlan": frozenset(
        {
            "identifier",
            "instantiatesCanonical",
            "instantiatesUri",
            "basedOn",
            "replaces",
            "partOf",
            "status",
            "intent",
            "category",
            "title",
            "description",
            "subject",
            "encounter",
            "period",
            "author",
            "contributor",
            "careTeam",
            "addresses",
            "supportingInfo",
            "goal",
            "activity",
            "note",
        }
    ),
    "Goal": frozenset(
        {
            "identifier",
            "lifecycleStatus",
            "achievementStatus",
            "category",
            "priority",
            "description",
            "subject",
            "startDate",
            "startCodeableConcept",
            "target",
            "statusDate",
            "statusReason",
            "expressedBy",
            "addresses",
            "note",
        }
    ),
    "ServiceRequest": frozenset(
        {
            "identifier",
            "instantiatesCanonical",
            "instantiatesUri",
            "basedOn",
            "replaces",
            "requisition",
            "status",
            "intent",
            "category",
            "priority",
            "doNotPerform",
            "code",
            "orderDetail",
            "quantityQuantity",
            "quantityRatio",
            "quantityRange",
            "subject",
            "focus",
            "encounter",
            "occurrenceDateTime",
            "occurrencePeriod",
            "occurrenceTiming",
            "asNeededBoolean",
            "asNeededCodeableConcept",
            "authoredOn",
            "requester",
            "performerType",
            "performer",
            "locationCode",
            "locationReference",
            "reasonCode",
            "reasonReference",
            "insurance",
            "supportingInfo",
            "specimen",
            "bodySite",
            "note",
            "patientInstruction",
            "relevantHistory",
            "orderDetail",
        }
    ),
    "Specimen": frozenset(
        {
            "identifier",
            "accessionIdentifier",
            "status",
            "type",
            "subject",
            "receivedTime",
            "parent",
            "request",
            "collection",
            "processing",
            "container",
            "condition",
            "note",
        }
    ),
}

_PRESERVATION_EXTENSION_URL = (
    "https://openmed.dev/fhir/StructureDefinition/version-preserved-field"
)
_PRESERVED_PATH_URL = "path"
_PRESERVED_VALUE_URL = "value-json"
_PRESERVED_SOURCE_URL = "source-version"

_R4_TO_R5_MEDICATION_STATUS = {
    "active": "recorded",
    "completed": "recorded",
    "intended": "draft",
    "stopped": "recorded",
    "on-hold": "draft",
    "unknown": "draft",
    "not-taken": "entered-in-error",
    "entered-in-error": "entered-in-error",
}
_R5_TO_R4_MEDICATION_STATUS = {
    "entered-in-error": "entered-in-error",
}


def parse_fhir_version(value: FHIRVersion | str) -> FHIRVersion:
    """Normalize a FHIR release label or exact core version."""

    if isinstance(value, FHIRVersion):
        return value
    normalized = str(value or "").strip().upper()
    aliases = {
        "R4": FHIRVersion.R4,
        "4": FHIRVersion.R4,
        "4.0.1": FHIRVersion.R4,
        "FHIR R4": FHIRVersion.R4,
        "R5": FHIRVersion.R5,
        "5": FHIRVersion.R5,
        "5.0.0": FHIRVersion.R5,
        "FHIR R5": FHIRVersion.R5,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise FHIRVersionError(
            f"unsupported FHIR version {value!r}; expected R4/4.0.1 or R5/5.0.0"
        ) from exc


def convert_resource(
    resource: Mapping[str, Any],
    source_version: FHIRVersion | str = FHIRVersion.R4,
    target_version: FHIRVersion | str = FHIRVersion.R4,
) -> dict[str, Any]:
    """Convert one supported FHIR resource without silently dropping fields.

    Args:
        resource: A FHIR JSON resource mapping.
        source_version: Explicit release of ``resource``.
        target_version: Release to emit.

    Returns:
        A new JSON-compatible mapping.

    Raises:
        TypeError: If ``resource`` is not mapping-shaped.
        ValueError: If the resource type or a cross-version field is unsupported.
        UnsupportedFHIRFieldError: For a path-specific non-representable field.
    """

    source = parse_fhir_version(source_version)
    target = parse_fhir_version(target_version)
    if not isinstance(resource, Mapping):
        raise TypeError("FHIR resource must be a mapping")
    converted = copy.deepcopy(dict(resource))
    resource_type = converted.get("resourceType")
    _validate_resource_shape(converted, "", source, target)
    if source == target:
        return converted

    if resource_type == "Bundle":
        _convert_bundle_entries(converted, source, target)
    elif resource_type == "MedicationStatement":
        _convert_medication_statement(converted, source, target)
    elif resource_type == "Procedure":
        _convert_procedure(converted, source, target)

    return converted


def convert_bundle(
    bundle: Mapping[str, Any],
    source_version: FHIRVersion | str = FHIRVersion.R4,
    target_version: FHIRVersion | str = FHIRVersion.R4,
) -> dict[str, Any]:
    """Convert a Bundle and every contained entry resource."""

    if not isinstance(bundle, Mapping) or bundle.get("resourceType") != "Bundle":
        raise ValueError("FHIR exchange input must be a Bundle")
    return convert_resource(bundle, source_version, target_version)


class FHIRVersionAdapter:
    """Small explicit adapter object for callers that retain release state."""

    def __init__(
        self,
        source_version: FHIRVersion | str = FHIRVersion.R4,
        target_version: FHIRVersion | str = FHIRVersion.R4,
    ) -> None:
        self.source_version = parse_fhir_version(source_version)
        self.target_version = parse_fhir_version(target_version)

    def convert(
        self,
        resource: Mapping[str, Any],
        source_version: FHIRVersion | str | None = None,
        target_version: FHIRVersion | str | None = None,
        *,
        from_version: FHIRVersion | str | None = None,
        to_version: FHIRVersion | str | None = None,
    ) -> dict[str, Any]:
        """Convert using configured releases, with optional per-call overrides."""

        source = source_version or from_version or self.source_version
        target = target_version or to_version or self.target_version
        return convert_resource(resource, source, target)

    def convert_bundle(
        self,
        bundle: Mapping[str, Any],
        **kwargs: FHIRVersion | str,
    ) -> dict[str, Any]:
        """Convert a Bundle using the adapter's release boundary."""

        converted = self.convert(bundle, **kwargs)
        if converted.get("resourceType") != "Bundle":
            raise ValueError("FHIR exchange input must be a Bundle")
        return converted

    def r4_to_r5(self, resource: Mapping[str, Any]) -> dict[str, Any]:
        """Convert an explicitly R4 resource to R5."""

        return convert_resource(resource, FHIRVersion.R4, FHIRVersion.R5)

    def r5_to_r4(self, resource: Mapping[str, Any]) -> dict[str, Any]:
        """Convert an explicitly R5 resource to R4."""

        return convert_resource(resource, FHIRVersion.R5, FHIRVersion.R4)


VersionAdapter = FHIRVersionAdapter
UnsupportedFHIRField = UnsupportedFHIRFieldError


def r4_to_r5(resource: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one supported R4 resource to R5."""

    return convert_resource(resource, FHIRVersion.R4, FHIRVersion.R5)


def r5_to_r4(resource: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one supported R5 resource to R4."""

    return convert_resource(resource, FHIRVersion.R5, FHIRVersion.R4)


def _validate_resource_shape(
    resource: Mapping[str, Any],
    path: str,
    source: FHIRVersion,
    target: FHIRVersion,
) -> None:
    resource_type = resource.get("resourceType")
    resource_path = path or str(resource_type or "resource")
    if not isinstance(resource_type, str) or not resource_type:
        raise ValueError(f"{resource_path}.resourceType is required")
    if resource_type not in SUPPORTED_RESOURCE_TYPES:
        raise UnsupportedFHIRFieldError(
            f"{resource_path}.resourceType",
            source_version=source,
            target_version=target,
            reason=f"resource type {resource_type!r} is outside the supported subset",
        )

    allowed = _BASE_RESOURCE_FIELDS | _RESOURCE_FIELDS.get(resource_type, frozenset())
    for key in resource:
        if key not in allowed:
            raise UnsupportedFHIRFieldError(
                f"{resource_path}.{key}",
                source_version=source,
                target_version=target,
            )

    contained = resource.get("contained")
    if contained is not None:
        if not isinstance(contained, Sequence) or isinstance(contained, (str, bytes)):
            raise ValueError(f"{resource_path}.contained must be an array")
        for index, child in enumerate(contained):
            if not isinstance(child, Mapping):
                raise ValueError(
                    f"{resource_path}.contained[{index}] must be a resource"
                )
            _validate_resource_shape(
                child,
                f"{resource_path}.contained[{index}]",
                source,
                target,
            )

    if resource_type == "Bundle":
        entries = resource.get("entry", [])
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"{resource_path}.entry must be an array")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"{resource_path}.entry[{index}] must be an object")
            for entry_key in entry:
                if entry_key not in {
                    "fullUrl",
                    "resource",
                    "search",
                    "request",
                    "response",
                }:
                    raise UnsupportedFHIRFieldError(
                        f"{resource_path}.entry[{index}].{entry_key}",
                        source_version=source,
                        target_version=target,
                    )
            child = entry.get("resource")
            if child is not None:
                if not isinstance(child, Mapping):
                    raise ValueError(
                        f"{resource_path}.entry[{index}].resource must be an object"
                    )
                _validate_resource_shape(
                    child,
                    f"{resource_path}.entry[{index}].resource",
                    source,
                    target,
                )


def _convert_bundle_entries(
    bundle: dict[str, Any], source: FHIRVersion, target: FHIRVersion
) -> None:
    entries = bundle.get("entry")
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        child = entry.get("resource")
        if isinstance(child, Mapping):
            entry["resource"] = convert_resource(child, source, target)


def _convert_medication_statement(
    resource: dict[str, Any], source: FHIRVersion, target: FHIRVersion
) -> None:
    if source == FHIRVersion.R4 and target == FHIRVersion.R5:
        _r4_to_r5_medication(resource, source, target)
        return
    _r5_to_r4_medication(resource, source, target)


def _r4_to_r5_medication(
    resource: dict[str, Any], source: FHIRVersion, target: FHIRVersion
) -> None:
    _move_choice(
        resource,
        "medicationCodeableConcept",
        "medication",
        lambda value: {"concept": value},
        source,
        target,
    )
    _move_choice(
        resource,
        "medicationReference",
        "medication",
        lambda value: {"reference": value},
        source,
        target,
    )
    if "medication" in resource and not _is_codeable_reference(resource["medication"]):
        raise UnsupportedFHIRFieldError(
            "MedicationStatement.medication",
            source_version=source,
            target_version=target,
            reason="expected a CodeableReference with concept or reference",
        )

    if "status" in resource:
        status = resource["status"]
        if not isinstance(status, str) or status not in _R4_TO_R5_MEDICATION_STATUS:
            raise UnsupportedFHIRFieldError(
                "MedicationStatement.status",
                source_version=source,
                target_version=target,
                reason="R4 status code has no supported R5 mapping",
            )
        _preserve_field(resource, "MedicationStatement.status", status, source)
        resource["status"] = _R4_TO_R5_MEDICATION_STATUS[status]

    if "context" in resource:
        context = resource.pop("context")
        if not _reference_target_is(context, "Encounter"):
            raise UnsupportedFHIRFieldError(
                "MedicationStatement.context",
                source_version=source,
                target_version=target,
                reason="R5 encounter no longer accepts an EpisodeOfCare reference",
            )
        resource["encounter"] = context

    if "reasonCode" in resource or "reasonReference" in resource:
        reasons: list[dict[str, Any]] = []
        reasons.extend({"concept": item} for item in resource.pop("reasonCode", []))
        reasons.extend(
            {"reference": item} for item in resource.pop("reasonReference", [])
        )
        resource["reason"] = reasons

    for field in ("statusReason", "basedOn"):
        if field in resource:
            _preserve_field(
                resource, f"MedicationStatement.{field}", resource[field], source
            )
            resource.pop(field)

    for field in (
        "relatedClinicalInformation",
        "renderedDosageInstruction",
        "adherence",
    ):
        if field in resource:
            raise UnsupportedFHIRFieldError(
                f"MedicationStatement.{field}",
                source_version=source,
                target_version=target,
                reason="field is R5-only and cannot be converted from R4",
            )


def _r5_to_r4_medication(
    resource: dict[str, Any], source: FHIRVersion, target: FHIRVersion
) -> None:
    medication = resource.pop("medication", None)
    if medication is not None:
        if not isinstance(medication, Mapping):
            raise UnsupportedFHIRFieldError(
                "MedicationStatement.medication",
                source_version=source,
                target_version=target,
                reason="expected a CodeableReference mapping",
            )
        keys = set(medication)
        if keys - {"concept", "reference"} or not keys:
            raise UnsupportedFHIRFieldError(
                "MedicationStatement.medication",
                source_version=source,
                target_version=target,
                reason="CodeableReference contains an unsupported member",
            )
        if "concept" in medication and "reference" in medication:
            raise UnsupportedFHIRFieldError(
                "MedicationStatement.medication",
                source_version=source,
                target_version=target,
                reason="R4 medication[x] cannot carry concept and reference together",
            )
        if "concept" in medication:
            resource["medicationCodeableConcept"] = copy.deepcopy(medication["concept"])
        else:
            resource["medicationReference"] = copy.deepcopy(medication["reference"])

    if "status" in resource:
        preserved = _take_preserved_field(resource, "MedicationStatement.status")
        if preserved is not None:
            resource["status"] = preserved
        else:
            status = resource["status"]
            if status not in _R5_TO_R4_MEDICATION_STATUS:
                raise UnsupportedFHIRFieldError(
                    "MedicationStatement.status",
                    source_version=source,
                    target_version=target,
                    reason="R5 status requires a preservation extension for lossless R4 conversion",
                )
            resource["status"] = _R5_TO_R4_MEDICATION_STATUS[status]

    if "encounter" in resource:
        resource["context"] = resource.pop("encounter")

    reasons = resource.pop("reason", None)
    if reasons is not None:
        if not isinstance(reasons, list):
            raise UnsupportedFHIRFieldError(
                "MedicationStatement.reason",
                source_version=source,
                target_version=target,
                reason="expected an array of CodeableReference values",
            )
        reason_codes: list[Any] = []
        reason_references: list[Any] = []
        for index, item in enumerate(reasons):
            if not isinstance(item, Mapping):
                raise UnsupportedFHIRFieldError(
                    f"MedicationStatement.reason[{index}]",
                    source_version=source,
                    target_version=target,
                    reason="expected a CodeableReference mapping",
                )
            if "concept" in item and "reference" in item:
                raise UnsupportedFHIRFieldError(
                    f"MedicationStatement.reason[{index}]",
                    source_version=source,
                    target_version=target,
                    reason="R4 reason fields cannot carry concept and reference together",
                )
            if set(item) == {"concept"}:
                reason_codes.append(copy.deepcopy(item["concept"]))
            elif set(item) == {"reference"}:
                reason_references.append(copy.deepcopy(item["reference"]))
            else:
                raise UnsupportedFHIRFieldError(
                    f"MedicationStatement.reason[{index}]",
                    source_version=source,
                    target_version=target,
                    reason="CodeableReference contains an unsupported member",
                )
        if reason_codes:
            resource["reasonCode"] = reason_codes
        if reason_references:
            resource["reasonReference"] = reason_references

    for field in (
        "relatedClinicalInformation",
        "renderedDosageInstruction",
        "adherence",
    ):
        if field in resource:
            raise UnsupportedFHIRFieldError(
                f"MedicationStatement.{field}",
                source_version=source,
                target_version=target,
                reason="field is not representable in R4",
            )

    for field in ("statusReason", "basedOn"):
        if field not in resource:
            preserved = _take_preserved_field(resource, f"MedicationStatement.{field}")
            if preserved is not None:
                resource[field] = preserved


def _convert_procedure(
    resource: dict[str, Any], source: FHIRVersion, target: FHIRVersion
) -> None:
    if source == FHIRVersion.R4 and target == FHIRVersion.R5:
        used_reference = resource.pop("usedReference", None)
        used_concept = resource.pop("usedCodeableConcept", None)
        if used_reference is not None and used_concept is not None:
            raise UnsupportedFHIRFieldError(
                "Procedure.used[x]",
                source_version=source,
                target_version=target,
                reason="R4 choice cannot contain both usedReference and usedCodeableConcept",
            )
        if used_reference is not None:
            resource["used"] = {"reference": used_reference}
        elif used_concept is not None:
            resource["used"] = {"concept": used_concept}
        return

    used = resource.pop("used", None)
    if used is None:
        return
    if not isinstance(used, Mapping) or set(used) not in ({"reference"}, {"concept"}):
        raise UnsupportedFHIRFieldError(
            "Procedure.used",
            source_version=source,
            target_version=target,
            reason="R5 CodeableReference cannot be represented by one R4 choice",
        )
    if "reference" in used:
        resource["usedReference"] = copy.deepcopy(used["reference"])
    else:
        resource["usedCodeableConcept"] = copy.deepcopy(used["concept"])


def _move_choice(
    resource: dict[str, Any],
    source_key: str,
    target_key: str,
    wrapper: Any,
    source: FHIRVersion,
    target: FHIRVersion,
) -> None:
    if source_key not in resource:
        return
    if target_key in resource:
        raise UnsupportedFHIRFieldError(
            f"MedicationStatement.{target_key}",
            source_version=source,
            target_version=target,
            reason="choice has both source and target representations",
        )
    resource[target_key] = wrapper(resource.pop(source_key))


def _is_codeable_reference(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value).issubset({"concept", "reference"})
        and bool(value)
    )


def _reference_target_is(value: Any, resource_type: str) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("reference"), str)
        and value["reference"].startswith(f"{resource_type}/")
    )


def _preserve_field(
    resource: dict[str, Any], path: str, value: Any, source: FHIRVersion
) -> None:
    extensions = resource.setdefault("extension", [])
    if not isinstance(extensions, list):
        raise UnsupportedFHIRFieldError(
            f"{resource.get('resourceType', 'Resource')}.extension",
            source_version=source,
            target_version=FHIRVersion.R5
            if source == FHIRVersion.R4
            else FHIRVersion.R4,
            reason="cannot append a loss-preservation extension to a non-array extension",
        )
    extensions.append(
        {
            "url": _PRESERVATION_EXTENSION_URL,
            "extension": [
                {"url": _PRESERVED_SOURCE_URL, "valueCode": source.value},
                {"url": _PRESERVED_PATH_URL, "valueString": path},
                {
                    "url": _PRESERVED_VALUE_URL,
                    "valueString": json.dumps(
                        value, sort_keys=True, separators=(",", ":")
                    ),
                },
            ],
        }
    )


def _take_preserved_field(resource: dict[str, Any], path: str) -> Any:
    extensions = resource.get("extension")
    if not isinstance(extensions, list):
        return None
    retained: list[Any] = []
    found: Any = None
    for extension in extensions:
        if (
            not isinstance(extension, Mapping)
            or extension.get("url") != _PRESERVATION_EXTENSION_URL
        ):
            retained.append(extension)
            continue
        children = extension.get("extension")
        if not isinstance(children, list):
            retained.append(extension)
            continue
        child_map = {
            child.get("url"): child
            for child in children
            if isinstance(child, Mapping) and isinstance(child.get("url"), str)
        }
        path_value = child_map.get(_PRESERVED_PATH_URL, {}).get("valueString")
        source_value = child_map.get(_PRESERVED_SOURCE_URL, {}).get("valueCode")
        value_json = child_map.get(_PRESERVED_VALUE_URL, {}).get("valueString")
        if (
            path_value == path
            and source_value == FHIRVersion.R4.value
            and value_json is not None
        ):
            try:
                found = json.loads(value_json)
            except (TypeError, ValueError):
                found = None
            continue
        retained.append(extension)
    if retained:
        resource["extension"] = retained
    else:
        resource.pop("extension", None)
    return found
