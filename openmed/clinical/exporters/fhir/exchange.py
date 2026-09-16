"""FHIR clinical exchange workbench built on OpenMed's R4 primitives.

The workbench is intentionally local and mechanical. It assembles synthetic
patient summaries and clinical documents, invokes the existing FHIR narrative
de-identification operation, pseudonymizes logical identities while rewiring
references, and delegates cross-release conversion to the explicit adapter in
``openmed.interop.fhir``.

It does not contact a terminology server, run a remote validator, or claim
legal, regulatory, or certification conformance.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openmed.interop.fhir.profiles import get_profile
from openmed.interop.fhir.validation import validate, validation_result
from openmed.interop.fhir.versions import (
    FHIRVersion,
    FHIRVersionAdapter,
    convert_resource,
    parse_fhir_version,
)

from .bundle import to_bundle
from .privacy import sanitize_india_health_identifiers
from .references import deterministic_fullurl

__all__ = [
    "FHIRExchangeError",
    "FHIRExchangeWorkbench",
    "FHIRClinicalExchangeWorkbench",
    "FHIRExchange",
    "FHIRValidationError",
    "build_clinical_document",
    "build_ipa_patient_access",
    "build_ipa_example",
    "build_ips_patient_summary",
    "build_patient_summary",
    "build_fhir_document",
    "deidentify_bundle",
    "deidentify_fhir",
    "export_fhir",
    "export_bundle",
    "import_fhir",
    "import_bundle",
    "validate_exchange",
]

Deidentifier = Callable[..., Any]

_IPS_COMPOSITION_PROFILE = (
    "http://hl7.org/fhir/uv/ips/StructureDefinition/Composition-uv-ips|2.0.1"
)
_IPS_PATIENT_PROFILE = (
    "http://hl7.org/fhir/uv/ips/StructureDefinition/Patient-uv-ips|2.0.1"
)
_IPA_PATIENT_PROFILE = (
    "http://hl7.org/fhir/uv/ipa/StructureDefinition/ipa-patient|1.1.0"
)
_CLINICAL_DOCUMENT_COMPOSITION_PROFILE = (
    "http://hl7.org/fhir/uv/fhir-clinical-document/StructureDefinition/"
    "clinical-document-composition|1.1.0"
)
_IDENTIFIER_SYSTEM = "https://openmed.dev/fhir/sid/pseudonymous-identifier"
_DEFAULT_DATE = "2026-01-01T00:00:00Z"


class FHIRExchangeError(ValueError):
    """Base error for an exchange workbench operation."""


class FHIRValidationError(FHIRExchangeError):
    """Raised when a selected local profile has blocking issues."""

    def __init__(self, outcome: Mapping[str, Any]) -> None:
        self.outcome = copy.deepcopy(dict(outcome))
        paths = [
            expression
            for issue in self.outcome.get("issue", [])
            for expression in issue.get("expression", [])
            if isinstance(expression, str)
        ]
        suffix = f" at {', '.join(paths[:3])}" if paths else ""
        super().__init__(f"FHIR local validation failed{suffix}")


def import_fhir(
    payload: Mapping[str, Any],
    *,
    version: FHIRVersion | str = FHIRVersion.R4,
    profile: str | None = None,
    validate_input: bool = True,
) -> dict[str, Any]:
    """Import a supported FHIR resource or Bundle without mutating it.

    The FHIR release is explicit because a JSON payload does not carry enough
    information to infer R4 versus R5. Profile validation is local and may be
    disabled when the caller wants to collect an outcome separately.
    """

    resolved = parse_fhir_version(version)
    if not isinstance(payload, Mapping):
        raise TypeError("FHIR import payload must be a mapping")
    imported = copy.deepcopy(dict(payload))
    # Same-release conversion performs the supported resource and field-boundary
    # check without changing the payload.
    imported = convert_resource(imported, resolved, resolved)
    if validate_input:
        _raise_if_invalid(imported, resolved, profile)
    return imported


def deidentify_fhir(
    payload: Mapping[str, Any],
    *,
    policy: str = "hipaa_safe_harbor",
    method: str = "mask",
    deidentifier: Deidentifier | None = None,
    document_id: str = "openmed-document",
    redact_logical_ids: bool = True,
) -> dict[str, Any]:
    """De-identify a resource or Bundle and preserve its reference graph.

    Narrative, free text, and ``Identifier.value`` surfaces are delegated to
    the existing local FHIR operation. Resource logical ids, Bundle fullUrls,
    and references are then pseudonymized deterministically so source ids do
    not leak while internal links and Provenance targets continue to resolve.
    Coding systems and codes are never passed through the free-text walker.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("FHIR de-identification payload must be a mapping")
    from openmed.interop.fhir_operations import (
        de_identify_bundle,
        de_identify_resource,
    )

    if payload.get("resourceType") == "Bundle":
        transformed = de_identify_bundle(
            payload,
            policy=policy,
            method=method,
            deidentifier=deidentifier,
        )
        transformed = sanitize_india_health_identifiers(transformed)
        if redact_logical_ids:
            return _redact_bundle_identities(transformed, document_id=document_id)
        return transformed

    transformed = de_identify_resource(
        payload,
        policy=policy,
        method=method,
        deidentifier=deidentifier,
    )
    transformed = sanitize_india_health_identifiers(transformed)
    if redact_logical_ids:
        transformed = _redact_resource_identity(transformed, document_id=document_id)
    return transformed


def export_fhir(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    source_version: FHIRVersion | str = FHIRVersion.R4,
    target_version: FHIRVersion | str | None = None,
    profile: str | None = None,
    doc_id: str = "openmed-document",
    deidentifier: Deidentifier | None = None,
    policy: str = "hipaa_safe_harbor",
    method: str = "mask",
    validate_output: bool = True,
) -> dict[str, Any]:
    """Export a supported resource list or Bundle at an explicit FHIR release.

    A sequence of standalone resources is assembled as an R4-style Bundle
    before conversion. Pass ``deidentifier`` to apply the local privacy pass;
    otherwise the function remains a pure structural/version export.
    """

    source = parse_fhir_version(source_version)
    target = parse_fhir_version(target_version or source)
    if isinstance(payload, Mapping):
        candidate = copy.deepcopy(dict(payload))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        candidate = to_bundle(payload, doc_id=doc_id, bundle_type="document")
    else:
        raise TypeError("FHIR export payload must be a mapping or resource sequence")

    if deidentifier is not None:
        candidate = deidentify_fhir(
            candidate,
            policy=policy,
            method=method,
            deidentifier=deidentifier,
            document_id=doc_id,
        )
    emitted = convert_resource(candidate, source, target)
    if validate_output:
        _raise_if_invalid(emitted, target, profile)
    return emitted


def validate_exchange(
    payload: Mapping[str, Any],
    *,
    version: FHIRVersion | str = FHIRVersion.R4,
    profile: str | None = None,
) -> dict[str, Any]:
    """Run the offline supported-subset validator and return OperationOutcome."""

    return validate(payload, version, profile=profile)


def build_ips_patient_summary(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    patient: Mapping[str, Any] | None = None,
    document_id: str = "synthetic-ips",
    date: str = _DEFAULT_DATE,
    author_reference: str = "Device/openmed",
    validate_output: bool = True,
) -> dict[str, Any]:
    """Build an IPS 2.0.1-shaped synthetic patient-summary document Bundle."""

    source_resources = _resources_from_payload(resources)
    selected_patient = (
        copy.deepcopy(dict(patient))
        if patient is not None
        else _first_resource(source_resources, "Patient")
    )
    if selected_patient is None:
        raise FHIRExchangeError("IPS patient summary requires a Patient resource")
    selected_patient.setdefault("resourceType", "Patient")
    selected_patient.setdefault("id", _stable_id(document_id, "Patient", "patient"))
    _append_profile(selected_patient, _IPS_PATIENT_PROFILE)

    source_resources = [
        resource
        for resource in source_resources
        if resource.get("resourceType") not in {"Patient", "Composition"}
    ]
    composition = _first_resource(resources, "Composition")
    if composition is None:
        composition = _composition_shell(
            composition_id=f"{document_id}-composition",
            subject_reference=f"Patient/{selected_patient['id']}",
            date=date,
            author_reference=author_reference,
            title="International Patient Summary",
            profile=_IPS_COMPOSITION_PROFILE,
        )
    else:
        composition = copy.deepcopy(composition)
        composition.setdefault("id", f"{document_id}-composition")
        composition.setdefault(
            "subject", {"reference": f"Patient/{selected_patient['id']}"}
        )
        composition.setdefault("date", date)
        composition.setdefault("author", [{"reference": author_reference}])
        composition.setdefault("title", "International Patient Summary")
        composition.setdefault("status", "final")
        composition.setdefault(
            "type",
            {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "60591-5",
                        "display": "Patient summary Document",
                    }
                ]
            },
        )
        _append_profile(composition, _IPS_COMPOSITION_PROFILE)

    composition["section"] = _summary_sections(source_resources)
    ordered = [composition, selected_patient, *source_resources]
    bundle = to_bundle(ordered, doc_id=document_id, bundle_type="document")
    if validate_output:
        _raise_if_invalid(bundle, FHIRVersion.R4, "ips")
    return bundle


build_patient_summary = build_ips_patient_summary


def build_clinical_document(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    patient: Mapping[str, Any] | None = None,
    composition: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    document_id: str = "synthetic-clinical-document",
    date: str = _DEFAULT_DATE,
    author_reference: str = "Device/openmed",
    validate_output: bool = True,
) -> dict[str, Any]:
    """Build a FHIR Clinical Documents 1.1.0-shaped document Bundle."""

    source_resources = _resources_from_payload(resources)
    selected_patient = (
        copy.deepcopy(dict(patient))
        if patient is not None
        else _first_resource(source_resources, "Patient")
    )
    if selected_patient is not None:
        selected_patient.setdefault("id", _stable_id(document_id, "Patient", "patient"))
    source_resources = [
        resource
        for resource in source_resources
        if resource.get("resourceType") not in {"Patient", "Composition", "Provenance"}
    ]
    if provenance is None:
        provenance = _first_resource(resources, "Provenance")
    if composition is None:
        composition = _first_resource(resources, "Composition")
    if composition is None:
        if selected_patient is None:
            raise FHIRExchangeError("clinical document requires a Patient resource")
        composition_dict = _composition_shell(
            composition_id=f"{document_id}-composition",
            subject_reference=f"Patient/{selected_patient['id']}",
            date=date,
            author_reference=author_reference,
            title="Clinical document",
            profile=_CLINICAL_DOCUMENT_COMPOSITION_PROFILE,
        )
    else:
        composition_dict = copy.deepcopy(dict(composition))
        composition_dict.setdefault("id", f"{document_id}-composition")
        composition_dict.setdefault("status", "final")
        composition_dict.setdefault("date", date)
        composition_dict.setdefault("author", [{"reference": author_reference}])
        composition_dict.setdefault("title", "Clinical document")
        if selected_patient is not None:
            composition_dict.setdefault(
                "subject", {"reference": f"Patient/{selected_patient['id']}"}
            )
        _append_profile(composition_dict, _CLINICAL_DOCUMENT_COMPOSITION_PROFILE)

    composition_dict.setdefault("text", _safe_narrative("Clinical document"))
    composition_dict["section"] = _summary_sections(source_resources)

    ordered: list[Mapping[str, Any]] = [composition_dict]
    if selected_patient is not None:
        ordered.append(selected_patient)
    ordered.extend(source_resources)
    if provenance is not None:
        ordered.append(provenance)
    bundle = to_bundle(ordered, doc_id=document_id, bundle_type="document")
    if validate_output:
        _raise_if_invalid(bundle, FHIRVersion.R4, "clinical-document")
    return bundle


def build_ipa_patient_access(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    patient: Mapping[str, Any] | None = None,
    document_id: str = "synthetic-ipa",
    validate_output: bool = True,
) -> dict[str, Any]:
    """Build an IPA 1.1.0-shaped synthetic patient-access search Bundle."""

    source_resources = _resources_from_payload(resources)
    selected_patient = (
        copy.deepcopy(dict(patient))
        if patient is not None
        else _first_resource(source_resources, "Patient")
    )
    if selected_patient is None:
        raise FHIRExchangeError(
            "IPA patient-access example requires a Patient resource"
        )
    selected_patient.setdefault("id", _stable_id(document_id, "Patient", "patient"))
    selected_patient.setdefault(
        "identifier",
        [
            {
                "system": _IDENTIFIER_SYSTEM,
                "value": _stable_identifier(document_id, "patient", "patient"),
            }
        ],
    )
    _append_profile(selected_patient, _IPA_PATIENT_PROFILE)
    source_resources = [
        resource
        for resource in source_resources
        if resource.get("resourceType") not in {"Patient", "Composition"}
    ]
    bundle = to_bundle(
        [selected_patient, *source_resources],
        doc_id=document_id,
        bundle_type="searchset",
    )
    bundle["total"] = len(bundle["entry"])
    bundle["link"] = [{"relation": "self", "url": f"urn:openmed:ipa:{document_id}"}]
    if validate_output:
        _raise_if_invalid(bundle, FHIRVersion.R4, "ipa")
    return bundle


build_ipa_example = build_ipa_patient_access


class FHIRExchangeWorkbench:
    """Stateful facade for one explicit FHIR/profile exchange boundary."""

    def __init__(
        self,
        version: FHIRVersion | str = FHIRVersion.R4,
        *,
        profile: str | None = None,
        target_version: FHIRVersion | str | None = None,
        document_id: str = "openmed-document",
    ) -> None:
        self.version = parse_fhir_version(version)
        self.target_version = parse_fhir_version(target_version or version)
        if profile is not None:
            get_profile(profile)
        self.profile = profile
        self.document_id = document_id
        self.adapter = FHIRVersionAdapter(self.version, self.target_version)

    def import_payload(
        self, payload: Mapping[str, Any], *, validate_input: bool = True
    ) -> dict[str, Any]:
        """Import one resource or Bundle at the configured source release."""

        return import_fhir(
            payload,
            version=self.version,
            profile=self.profile,
            validate_input=validate_input,
        )

    import_resource = import_payload
    import_bundle = import_payload

    def deidentify(
        self,
        payload: Mapping[str, Any],
        *,
        policy: str = "hipaa_safe_harbor",
        method: str = "mask",
        deidentifier: Deidentifier | None = None,
    ) -> dict[str, Any]:
        """Apply the configured local privacy transform."""

        return deidentify_fhir(
            payload,
            policy=policy,
            method=method,
            deidentifier=deidentifier,
            document_id=self.document_id,
        )

    deidentify_resource = deidentify
    deidentify_bundle = deidentify

    def export(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        deidentifier: Deidentifier | None = None,
        policy: str = "hipaa_safe_harbor",
        method: str = "mask",
        validate_output: bool = True,
    ) -> dict[str, Any]:
        """Export at the configured target release."""

        return export_fhir(
            payload,
            source_version=self.version,
            target_version=self.target_version,
            profile=self.profile,
            doc_id=self.document_id,
            deidentifier=deidentifier,
            policy=policy,
            method=method,
            validate_output=validate_output,
        )

    export_resource = export
    export_bundle = export

    def convert(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Convert an already imported payload to the target release."""

        return self.adapter.convert(payload)

    def validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run the configured local validator."""

        return validate_exchange(
            payload, version=self.target_version, profile=self.profile
        )


FHIRClinicalExchangeWorkbench = FHIRExchangeWorkbench
FHIRExchange = FHIRExchangeWorkbench


build_fhir_document = build_clinical_document
deidentify_bundle = deidentify_fhir
export_bundle = export_fhir
import_bundle = import_fhir


def _raise_if_invalid(
    payload: Mapping[str, Any],
    version: FHIRVersion,
    profile: str | None,
) -> None:
    result = validation_result(payload, version, profile=profile)
    if not result.valid:
        raise FHIRValidationError(result.to_operation_outcome())


def _resources_from_payload(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        if payload.get("resourceType") == "Bundle":
            entries = payload.get("entry", [])
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                raise TypeError("FHIR Bundle.entry must be an array")
            resources = [
                dict(entry["resource"])
                for entry in entries
                if isinstance(entry, Mapping)
                and isinstance(entry.get("resource"), Mapping)
            ]
            return resources
        if isinstance(payload.get("resourceType"), str):
            return [dict(payload)]
        raise ValueError("FHIR payload must be a resource or Bundle")
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        resources: list[dict[str, Any]] = []
        for index, resource in enumerate(payload):
            if not isinstance(resource, Mapping):
                raise TypeError(f"FHIR resource at index {index} must be an object")
            if not isinstance(resource.get("resourceType"), str):
                raise ValueError(
                    f"FHIR resource at index {index} is missing resourceType"
                )
            resources.append(dict(resource))
        return resources
    raise TypeError("FHIR resources must be a resource sequence or Bundle")


def _first_resource(
    resources: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    resource_type: str,
) -> dict[str, Any] | None:
    for resource in _resources_from_payload(resources):
        if resource.get("resourceType") == resource_type:
            return copy.deepcopy(resource)
    return None


def _composition_shell(
    *,
    composition_id: str,
    subject_reference: str,
    date: str,
    author_reference: str,
    title: str,
    profile: str,
) -> dict[str, Any]:
    return {
        "resourceType": "Composition",
        "id": composition_id,
        "meta": {"profile": [profile]},
        "status": "final",
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "60591-5" if "Patient" in title else "34133-9",
                    "display": "Patient summary Document"
                    if "Patient" in title
                    else "Summarization of Episode of Care",
                }
            ]
        },
        "subject": {"reference": subject_reference},
        "date": date,
        "author": [{"reference": author_reference}],
        "title": title,
        "confidentiality": "N",
        "text": _safe_narrative(title),
    }


def _summary_sections(resources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    section_definitions = {
        "Condition": ("11348-0", "Problems and Conditions"),
        "AllergyIntolerance": ("48765-2", "Allergies and Intolerances"),
        "MedicationStatement": ("10160-0", "Medication Summary"),
        "Observation": ("30954-2", "Results"),
        "Procedure": ("47519-4", "History of Procedures"),
    }
    grouped: dict[str, tuple[str, str, list[dict[str, str]]]] = {}
    for resource in resources:
        resource_type = resource.get("resourceType")
        if resource_type not in section_definitions or not resource.get("id"):
            continue
        code, display = section_definitions[resource_type]
        group = grouped.setdefault(resource_type, (code, display, []))
        group[2].append({"reference": f"{resource_type}/{resource['id']}"})
    sections = []
    for resource_type in section_definitions:
        if resource_type not in grouped:
            continue
        code, display, references = grouped[resource_type]
        sections.append(
            {
                "title": display,
                "code": {
                    "coding": [
                        {"system": "http://loinc.org", "code": code, "display": display}
                    ]
                },
                "text": _safe_narrative(display),
                "entry": references,
            }
        )
    return sections


def _safe_narrative(title: str) -> dict[str, str]:
    return {
        "status": "generated",
        "div": f'<div xmlns="http://www.w3.org/1999/xhtml"><p>{title}</p></div>',
    }


def _append_profile(resource: dict[str, Any], profile: str) -> None:
    meta = resource.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = resource["meta"] = {}
    profiles = meta.setdefault("profile", [])
    if not isinstance(profiles, list):
        profiles = meta["profile"] = []
    if profile not in profiles:
        profiles.append(profile)


def _redact_bundle_identities(
    bundle: Mapping[str, Any], *, document_id: str
) -> dict[str, Any]:
    transformed = copy.deepcopy(dict(bundle))
    if isinstance(transformed.get("id"), str):
        transformed["id"] = _stable_id(document_id, "Bundle", transformed["id"])
    if isinstance(transformed.get("identifier"), list):
        _redact_identifier_values(
            {"identifier": transformed["identifier"]}, document_id
        )
    entries = transformed.get("entry")
    if not isinstance(entries, list):
        return transformed

    logical_map: dict[str, str] = {}
    full_url_map: dict[str, str] = {}
    fragment_map: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        original_full_url = entry.get("fullUrl")
        new_full_url = deterministic_fullurl(document_id, index)
        if isinstance(original_full_url, str):
            full_url_map[original_full_url] = new_full_url
        entry["fullUrl"] = new_full_url
        resource = entry.get("resource")
        if not isinstance(resource, dict):
            continue
        resource_type = resource.get("resourceType")
        original_id = resource.get("id")
        if isinstance(resource_type, str) and isinstance(original_id, str):
            logical_map[f"{resource_type}/{original_id}"] = (
                f"{resource_type}/{_stable_id(document_id, resource_type, original_id)}"
            )
            fragment_map[f"#{original_id}"] = (
                f"#{_stable_id(document_id, resource_type, original_id)}"
            )

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if isinstance(resource, dict):
            _redact_resource_ids(resource, document_id, logical_map)
        _rewrite_identity_references(entry, logical_map, full_url_map, fragment_map)
    return transformed


def _redact_resource_identity(
    resource: Mapping[str, Any], *, document_id: str
) -> dict[str, Any]:
    transformed = copy.deepcopy(dict(resource))
    resource_type = transformed.get("resourceType")
    logical_map: dict[str, str] = {}
    if isinstance(resource_type, str) and isinstance(transformed.get("id"), str):
        original = transformed["id"]
        logical_map[f"{resource_type}/{original}"] = (
            f"{resource_type}/{_stable_id(document_id, resource_type, original)}"
        )
    _redact_resource_ids(transformed, document_id, logical_map)
    _rewrite_identity_references(transformed, logical_map, {}, {})
    return transformed


def _redact_resource_ids(
    resource: dict[str, Any], document_id: str, logical_map: Mapping[str, str]
) -> None:
    resource_type = resource.get("resourceType")
    original_id = resource.get("id")
    if isinstance(resource_type, str) and isinstance(original_id, str):
        resource["id"] = _stable_id(document_id, resource_type, original_id)
    _redact_identifier_values(resource, document_id)
    contained = resource.get("contained")
    if isinstance(contained, list):
        for child in contained:
            if isinstance(child, dict):
                _redact_resource_ids(child, document_id, logical_map)


def _redact_identifier_values(node: Any, document_id: str, path: str = "") -> None:
    if isinstance(node, Mapping):
        for key, value in list(node.items()):
            child_path = f"{path}.{key}" if path else str(key)
            if key == "identifier" and isinstance(value, list):
                for identifier in value:
                    if isinstance(identifier, dict) and isinstance(
                        identifier.get("value"), str
                    ):
                        system = str(identifier.get("system") or "")
                        identifier["value"] = _stable_identifier(
                            document_id, system, identifier["value"]
                        )
            else:
                _redact_identifier_values(value, document_id, child_path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _redact_identifier_values(value, document_id, f"{path}[{index}]")


def _rewrite_identity_references(
    node: Any,
    logical_map: Mapping[str, str],
    full_url_map: Mapping[str, str],
    fragment_map: Mapping[str, str],
) -> None:
    if isinstance(node, Mapping):
        if isinstance(node, dict) and node.get("reference") in full_url_map:
            node["reference"] = full_url_map[node["reference"]]
        elif isinstance(node, dict) and node.get("reference") in logical_map:
            node["reference"] = logical_map[node["reference"]]
        elif isinstance(node, dict) and node.get("reference") in fragment_map:
            node["reference"] = fragment_map[node["reference"]]
        for key, value in node.items():
            _rewrite_identity_references(value, logical_map, full_url_map, fragment_map)
    elif isinstance(node, list):
        for value in node:
            _rewrite_identity_references(value, logical_map, full_url_map, fragment_map)


def _stable_id(document_id: str, resource_type: str, value: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}\x1f{resource_type}\x1f{value}".encode()
    ).hexdigest()[:24]
    return f"openmed-{digest}"


def _stable_identifier(document_id: str, system: str, value: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}\x1f{system}\x1f{value}".encode()
    ).hexdigest()[:24]
    return f"openmed-id-{digest}"
