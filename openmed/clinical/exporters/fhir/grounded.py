"""Deterministic FHIR R4 resources for grounded clinical spans."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ...context import (
    AFFIRMED,
    CERTAIN,
    PATIENT_EXPERIENCER,
    RECENT,
    ClinicalAssertion,
)
from ...coreference import CoreferenceChain
from ...grounding.assertion_grounding import (
    GROUNDING_HISTORICAL,
    GROUNDING_HYPOTHETICAL,
    GROUNDING_PRESENT,
    GROUNDING_REFUTED,
    AssertedGroundedSpan,
    assertion_grounding_status,
)
from ...grounding.types import GroundedSpan
from ..codeable_concept import to_codeable_concept
from .bundle import to_bundle
from .codeable_concept import POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL
from .condition import to_condition

__all__ = [
    "COREFERENCE_EVIDENCE_EXTENSION_URL",
    "FHIR_RESOURCE_TYPES",
    "to_fhir",
]

COREFERENCE_EVIDENCE_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/clinical-coreference-evidence"
)

FHIR_RESOURCE_TYPES = (
    "Condition",
    "MedicationStatement",
    "Observation",
    "Procedure",
)

_RESOURCE_BY_LABEL = {
    "CONDITION": "Condition",
    "MEDICATION": "MedicationStatement",
    "LAB_TEST": "Observation",
    "PROCEDURE": "Procedure",
}
_RESOURCE_BY_SYSTEM = {
    "HPO": "Condition",
    "ICD10CM": "Condition",
    "ICD11": "Condition",
    "MESH": "Condition",
    "SNOMED": "Condition",
    "UMLS": "Condition",
    "RXNORM": "MedicationStatement",
    "LOINC": "Observation",
}


@dataclass(frozen=True)
class _CoreferenceBinding:
    chain: CoreferenceChain


def to_fhir(
    grounded: GroundedSpan | Iterable[GroundedSpan],
    *,
    resource: str | None = None,
    subject_reference: str = "Patient/openmed-subject",
    document_id: str = "openmed-document",
    value: Any = None,
    unit: str | None = None,
    coreference_chains: Sequence[CoreferenceChain] = (),
) -> dict[str, Any] | None:
    """Export grounded spans as valid deterministic FHIR R4 resources.

    A single span returns one resource (or ``None`` for a non-patient finding).
    An iterable returns a transaction ``Bundle`` containing every retained
    resource. The resource type is inferred from the canonical clinical label,
    then from the selected coding system; callers may override it explicitly.

    Args:
        grounded: One grounded span or an iterable from one document.
        resource: Optional R4 resource type. Supported values are Condition,
            MedicationStatement, Observation, and Procedure.
        subject_reference: Patient reference used by emitted resources.
        document_id: Stable Bundle/fullUrl seed and resource-id namespace.
        value: Optional Observation value.
        unit: Optional UCUM display/code for a numeric Observation value.
        coreference_chains: Optional document-local clinical coreference chains.
            Same-cluster grounded spans collapse to one resource with supporting
            offsets and HMAC hashes in a privacy-safe FHIR extension.

    Returns:
        One FHIR resource, a transaction Bundle, or ``None`` when a single
        non-patient span is deliberately excluded.
    """

    if not isinstance(subject_reference, str) or not subject_reference.strip():
        raise ValueError("subject_reference must be a non-empty FHIR reference")
    coreference_by_offset = _coreference_bindings(coreference_chains)
    if isinstance(grounded, GroundedSpan):
        return _one_resource(
            grounded,
            resource=resource,
            subject_reference=subject_reference,
            document_id=document_id,
            value=value,
            unit=unit,
            coreference=coreference_by_offset.get((grounded.start, grounded.end)),
        )

    spans = tuple(grounded)
    if any(not isinstance(span, GroundedSpan) for span in spans):
        raise TypeError("to_fhir expects GroundedSpan objects")
    collapsed_spans = _collapse_grounded_spans(
        spans,
        resource=resource,
        coreference_by_offset=coreference_by_offset,
    )
    resources = [
        exported
        for span, coreference in collapsed_spans
        if (
            exported := _one_resource(
                span,
                resource=resource,
                subject_reference=subject_reference,
                document_id=document_id,
                value=span.metadata.get("value", value),
                unit=span.metadata.get("unit", unit),
                coreference=coreference,
            )
        )
        is not None
    ]
    return to_bundle(resources, doc_id=document_id)


def _one_resource(
    grounded: GroundedSpan,
    *,
    resource: str | None,
    subject_reference: str,
    document_id: str,
    value: Any,
    unit: str | None,
    coreference: _CoreferenceBinding | None,
) -> dict[str, Any] | None:
    asserted = _asserted_span(grounded)
    if not asserted.status.patient_subject:
        return None
    resource_type = _resource_type(grounded, resource)
    resource_id = _resource_id(
        document_id,
        grounded,
        resource_type,
        cluster_id=coreference.chain.chain_id if coreference else None,
    )

    if resource_type == "Condition":
        condition = to_condition(
            asserted,
            subject_reference=subject_reference,
            condition_id=resource_id,
        )
        if condition is None:
            return None
        return _attach_coreference_evidence(_strict_fhir(condition), coreference)

    concept = _strict_codeable_concept(grounded)
    if resource_type == "Observation":
        result: dict[str, Any] = {
            "resourceType": "Observation",
            "id": resource_id,
            "status": _observation_status(asserted),
            "code": concept,
            "subject": {"reference": subject_reference},
        }
        if value is not None:
            if isinstance(value, bool):
                result["valueBoolean"] = value
            elif isinstance(value, (int, float)):
                quantity: dict[str, Any] = {"value": value}
                if unit:
                    quantity.update(
                        {
                            "unit": unit,
                            "system": "http://unitsofmeasure.org",
                            "code": unit,
                        }
                    )
                result["valueQuantity"] = quantity
            else:
                result["valueString"] = str(value)
        return _attach_coreference_evidence(result, coreference)

    if resource_type == "MedicationStatement":
        result = {
            "resourceType": "MedicationStatement",
            "id": resource_id,
            "status": _medication_status(asserted),
            "medicationCodeableConcept": concept,
            "subject": {"reference": subject_reference},
        }
        return _attach_coreference_evidence(result, coreference)

    result = {
        "resourceType": "Procedure",
        "id": resource_id,
        "status": _procedure_status(asserted),
        "code": concept,
        "subject": {"reference": subject_reference},
    }
    return _attach_coreference_evidence(result, coreference)


def _coreference_bindings(
    chains: Sequence[CoreferenceChain],
) -> dict[tuple[int, int], _CoreferenceBinding]:
    bindings: dict[tuple[int, int], _CoreferenceBinding] = {}
    cluster_ids: set[str] = set()
    document_id: str | None = None
    for chain in chains:
        if not isinstance(chain, CoreferenceChain):
            raise TypeError("coreference_chains must contain CoreferenceChain values")
        if chain.chain_id in cluster_ids:
            raise ValueError("coreference cluster ids must be unique")
        cluster_ids.add(chain.chain_id)
        if chain.representative not in chain.members:
            raise ValueError("coreference representative must be a chain member")
        chain_document_ids = {member.doc_id for member in chain.members}
        if len(chain_document_ids) != 1:
            raise ValueError("coreference chains must be document-local")
        chain_document_id = next(iter(chain_document_ids))
        if document_id is not None and chain_document_id != document_id:
            raise ValueError("coreference chains must belong to one document")
        document_id = chain_document_id
        binding = _CoreferenceBinding(chain=chain)
        for member in chain.members:
            offset = (member.start, member.end)
            if offset in bindings:
                raise ValueError(
                    "one coreference source offset cannot belong to multiple clusters"
                )
            bindings[offset] = binding
    return bindings


def _collapse_grounded_spans(
    spans: tuple[GroundedSpan, ...],
    *,
    resource: str | None,
    coreference_by_offset: Mapping[tuple[int, int], _CoreferenceBinding],
) -> tuple[tuple[GroundedSpan, _CoreferenceBinding | None], ...]:
    grouped: dict[
        tuple[str, str, str], tuple[GroundedSpan, _CoreferenceBinding | None]
    ] = {}
    for index, span in enumerate(spans):
        binding = coreference_by_offset.get((span.start, span.end))
        if binding is None or not _asserted_span(span).status.patient_subject:
            key = ("span", str(index), "")
        else:
            key = (
                "coreference",
                binding.chain.chain_id,
                _resource_type(span, resource),
            )
        current = grouped.get(key)
        if current is None or _is_representative_span(span, binding):
            grouped[key] = (span, binding)
    return tuple(grouped.values())


def _is_representative_span(
    span: GroundedSpan,
    binding: _CoreferenceBinding | None,
) -> bool:
    if binding is None:
        return False
    representative = binding.chain.representative
    return (span.start, span.end) == (representative.start, representative.end)


def _attach_coreference_evidence(
    resource: dict[str, Any],
    binding: _CoreferenceBinding | None,
) -> dict[str, Any]:
    if binding is None:
        return resource
    chain = binding.chain
    representative = chain.representative
    supporting_mentions = [
        {
            "url": "supportingMention",
            "extension": [
                {"url": "start", "valueUnsignedInt": member.start},
                {"url": "end", "valueUnsignedInt": member.end},
                {"url": "textHash", "valueString": member.text_hash},
            ],
        }
        for member in sorted(
            chain.members,
            key=lambda item: (item.start, item.end, item.text_hash),
        )
    ]
    evidence = {
        "url": COREFERENCE_EVIDENCE_EXTENSION_URL,
        "extension": [
            {"url": "clusterId", "valueString": chain.chain_id},
            {
                "url": "representative",
                "extension": [
                    {"url": "start", "valueUnsignedInt": representative.start},
                    {"url": "end", "valueUnsignedInt": representative.end},
                    {
                        "url": "textHash",
                        "valueString": representative.text_hash,
                    },
                ],
            },
            *supporting_mentions,
        ],
    }
    extensions = list(resource.get("extension", ()))
    extensions.append(evidence)
    resource["extension"] = extensions
    return resource


def _resource_type(grounded: GroundedSpan, resource: str | None) -> str:
    if resource is not None:
        normalized = resource.strip().casefold()
        matches = {candidate.casefold(): candidate for candidate in FHIR_RESOURCE_TYPES}
        if normalized not in matches:
            raise ValueError(
                f"resource must be one of {FHIR_RESOURCE_TYPES!r}, got {resource!r}"
            )
        return matches[normalized]
    label = (grounded.canonical_label or "").upper()
    if label in _RESOURCE_BY_LABEL:
        return _RESOURCE_BY_LABEL[label]
    if grounded.candidates:
        system = grounded.candidates[0].system.upper()
        if system in _RESOURCE_BY_SYSTEM:
            return _RESOURCE_BY_SYSTEM[system]
    raise ValueError(
        "cannot infer a FHIR resource type; provide canonical_label or resource="
    )


def _asserted_span(grounded: GroundedSpan) -> AssertedGroundedSpan:
    assertion = grounded.assertion or ClinicalAssertion(
        temporality=RECENT,
        certainty=CERTAIN,
        negation=AFFIRMED,
        experiencer=PATIENT_EXPERIENCER,
    )
    return AssertedGroundedSpan(
        grounded=grounded,
        assertion=assertion,
        status=assertion_grounding_status(assertion),
        provenance={"span_offset": [grounded.start, grounded.end]},
    )


def _strict_codeable_concept(grounded: GroundedSpan) -> dict[str, Any]:
    return _strict_fhir(to_codeable_concept(grounded))


def _strict_fhir(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove implementation-only fields and unresolved custom extensions."""

    result = deepcopy(dict(value))
    _remove_internal_fields(result)
    return result


def _remove_internal_fields(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("_score", None)
        extensions = node.get("extension")
        if isinstance(extensions, list):
            retained = [
                extension
                for extension in extensions
                if not (
                    isinstance(extension, Mapping)
                    and str(extension.get("url") or "").startswith(
                        "https://openmed.ai/fhir/StructureDefinition/"
                    )
                    and extension.get("url")
                    != POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL
                )
            ]
            if retained:
                node["extension"] = retained
            else:
                node.pop("extension")
        for child in node.values():
            _remove_internal_fields(child)
    elif isinstance(node, list):
        for child in node:
            _remove_internal_fields(child)


def _resource_id(
    document_id: str,
    grounded: GroundedSpan,
    resource_type: str,
    *,
    cluster_id: str | None = None,
) -> str:
    codes = "|".join(
        f"{candidate.system}:{candidate.code}" for candidate in grounded.candidates
    )
    payload = (
        f"{document_id}\x1f{resource_type}\x1f"
        f"{cluster_id or f'{grounded.start}:{grounded.end}'}"
        f"\x1f{codes}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"openmed-{digest}"


def _observation_status(asserted: AssertedGroundedSpan) -> str:
    if asserted.status.status in {GROUNDING_REFUTED, GROUNDING_HYPOTHETICAL}:
        return "cancelled"
    return "final"


def _medication_status(asserted: AssertedGroundedSpan) -> str:
    return {
        GROUNDING_PRESENT: "active",
        GROUNDING_HISTORICAL: "completed",
        GROUNDING_REFUTED: "not-taken",
        GROUNDING_HYPOTHETICAL: "intended",
    }.get(asserted.status.status, "unknown")


def _procedure_status(asserted: AssertedGroundedSpan) -> str:
    return {
        GROUNDING_PRESENT: "completed",
        GROUNDING_HISTORICAL: "completed",
        GROUNDING_REFUTED: "not-done",
        GROUNDING_HYPOTHETICAL: "preparation",
    }.get(asserted.status.status, "unknown")
