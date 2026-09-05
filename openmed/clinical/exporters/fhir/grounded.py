"""Deterministic FHIR R4 resources for grounded clinical spans."""

from __future__ import annotations

import hashlib
from collections import Counter
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
    ClinicalContextResult,
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
from .bundle import to_bundle
from .codeable_concept import (
    GROUNDED_CODE_PROVENANCE_EXTENSION_URL,
    MEDICAL_DEVICE_ASSIST_EXTENSION_URL,
    POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL,
    to_codeable_concept,
)
from .condition import to_condition
from .observation import to_observation

__all__ = [
    "COREFERENCE_EVIDENCE_EXTENSION_URL",
    "FHIRBundle",
    "FHIRExportSummary",
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

_DEFAULT_DOCUMENT_ID = "openmed-document"
_UNLABELED = "UNLABELED"


@dataclass(frozen=True)
class FHIRExportSummary:
    """Aggregate, PHI-free facts about one grounded-span export.

    Counts are keyed by normalized canonical label. ``exported_by_label``
    includes mapped labels that produced zero resources, for example because
    assertion context excluded every non-patient span. ``unmapped_by_label``
    contains labels for which the facade has no exporter.

    Attributes:
        exported_by_label: Number of emitted resources for each mapped label.
        unmapped_by_label: Number of skipped spans for each unmapped label.
    """

    exported_by_label: Mapping[str, int]
    unmapped_by_label: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exported_by_label",
            dict(sorted(self.exported_by_label.items())),
        )
        object.__setattr__(
            self,
            "unmapped_by_label",
            dict(sorted(self.unmapped_by_label.items())),
        )

    @property
    def resource_count(self) -> int:
        """Return the total number of emitted resources."""

        return sum(self.exported_by_label.values())

    @property
    def unmapped_count(self) -> int:
        """Return the total number of spans skipped for missing mappings."""

        return sum(self.unmapped_by_label.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe counts-only summary."""

        return {
            "exported_by_label": dict(self.exported_by_label),
            "unmapped_by_label": dict(self.unmapped_by_label),
            "resource_count": self.resource_count,
            "unmapped_count": self.unmapped_count,
        }


class FHIRBundle(dict[str, Any]):
    """FHIR Bundle mapping carrying a counts-only export summary sidecar.

    The mapping contains only the R4 Bundle produced by :func:`to_bundle`, so
    normal dictionary access and JSON serialization remain unchanged. Summary
    data is available through :attr:`summary` and is deliberately kept outside
    the FHIR payload.
    """

    summary: FHIRExportSummary

    def __init__(
        self,
        bundle: Mapping[str, Any],
        *,
        summary: FHIRExportSummary,
    ) -> None:
        super().__init__(bundle)
        self.summary = summary


@dataclass(frozen=True)
class _CoreferenceBinding:
    chain: CoreferenceChain


def to_fhir(
    grounded: GroundedSpan | Iterable[GroundedSpan],
    *,
    doc_id: str = _DEFAULT_DOCUMENT_ID,
    bundle_type: str = "transaction",
    systems: Mapping[str, str] | None = None,
    resource: str | None = None,
    subject_reference: str = "Patient/openmed-subject",
    document_id: str | None = None,
    value: Any = None,
    unit: str | None = None,
    coreference_chains: Sequence[CoreferenceChain] = (),
) -> dict[str, Any] | None:
    """Export grounded spans as valid deterministic FHIR R4 resources.

    A single span returns one resource (or ``None`` for a non-patient finding).
    An iterable returns one ``FHIRBundle`` containing every retained resource.
    Iterable dispatch is driven by canonical clinical labels. Unmapped labels
    are skipped and counted in ``bundle.summary`` instead of aborting export.
    A single-span call retains the legacy resource/system inference behavior.

    Args:
        grounded: One grounded span or an iterable from one document.
        doc_id: Stable Bundle/fullUrl seed and resource-id namespace.
        bundle_type: FHIR Bundle type for iterable input. Defaults to
            ``"transaction"``.
        systems: Optional mapping from grounding system names to supported FHIR
            resource types. It provides explicit routing for spans without a
            canonical label; an unrecognized non-empty canonical label remains
            unmapped and is never inferred from its coding system.
        resource: Optional R4 resource type. Supported values are Condition,
            MedicationStatement, Observation, and Procedure.
        subject_reference: Patient reference used by emitted resources.
        document_id: Compatibility alias for ``doc_id``.
        value: Optional Observation value.
        unit: Optional UCUM display/code for a numeric Observation value.
        coreference_chains: Optional document-local clinical coreference chains.
            Same-cluster grounded spans collapse to one resource with supporting
            offsets and HMAC hashes in a privacy-safe FHIR extension.

    Returns:
        One FHIR resource, a ``FHIRBundle`` with a counts-only ``summary``
        sidecar, or ``None`` when a single non-patient span is deliberately
        excluded. No Patient resource is synthesized.

    Raises:
        ValueError: If ``doc_id`` and ``document_id`` disagree, the resolved
            document id is empty, or a route names an unsupported resource.
        TypeError: If iterable values are not ``GroundedSpan`` objects or
            ``systems`` is not a mapping.
    """

    if not isinstance(subject_reference, str) or not subject_reference.strip():
        raise ValueError("subject_reference must be a non-empty FHIR reference")
    document_id = _resolve_document_id(doc_id, document_id)
    system_routes = _normalize_system_routes(systems)
    coreference_by_offset = _coreference_bindings(coreference_chains)
    if isinstance(grounded, GroundedSpan):
        return _one_resource(
            grounded,
            resource=resource,
            subject_reference=subject_reference,
            document_id=document_id,
            value=grounded.metadata.get("value", value),
            unit=grounded.metadata.get("unit", unit),
            coreference=coreference_by_offset.get((grounded.start, grounded.end)),
            system_routes=system_routes,
            allow_default_system_fallback=systems is None,
        )

    spans = tuple(grounded)
    if any(not isinstance(span, GroundedSpan) for span in spans):
        raise TypeError("to_fhir expects GroundedSpan objects")
    routed_spans: list[tuple[GroundedSpan, str, str]] = []
    exported_by_label: Counter[str] = Counter()
    unmapped_by_label: Counter[str] = Counter()
    for span in spans:
        label = _normalized_label(span)
        resource_type = _resource_type(
            span,
            resource,
            system_routes=system_routes,
            allow_default_system_fallback=False,
        )
        if resource_type is None:
            unmapped_by_label[label] += 1
            continue
        exported_by_label[label] += 0
        routed_spans.append((span, resource_type, label))
    collapsed_spans = _collapse_grounded_spans(
        tuple(routed_spans),
        coreference_by_offset=coreference_by_offset,
    )
    resources: list[dict[str, Any]] = []
    for span, resource_type, label, coreference in collapsed_spans:
        exported = _one_resource(
            span,
            resource=resource_type,
            subject_reference=subject_reference,
            document_id=document_id,
            value=span.metadata.get("value", value),
            unit=span.metadata.get("unit", unit),
            coreference=coreference,
            system_routes=system_routes,
            allow_default_system_fallback=False,
        )
        if exported is not None:
            resources.append(exported)
            exported_by_label[label] += 1
    summary = FHIRExportSummary(
        exported_by_label=exported_by_label,
        unmapped_by_label=unmapped_by_label,
    )
    bundle = to_bundle(
        resources,
        doc_id=document_id,
        bundle_type=bundle_type,
    )
    return FHIRBundle(bundle, summary=summary)


def _one_resource(
    grounded: GroundedSpan,
    *,
    resource: str | None,
    subject_reference: str,
    document_id: str,
    value: Any,
    unit: str | None,
    coreference: _CoreferenceBinding | None,
    system_routes: Mapping[str, str],
    allow_default_system_fallback: bool,
) -> dict[str, Any] | None:
    asserted = _asserted_span(grounded)
    if not asserted.status.patient_subject:
        return None
    resource_type = _resource_type(
        grounded,
        resource,
        system_routes=system_routes,
        allow_default_system_fallback=allow_default_system_fallback,
    )
    if resource_type is None:
        return None
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

    if resource_type == "Observation":
        observation = to_observation(
            asserted,
            subject_reference=subject_reference,
            observation_id=resource_id,
            value=value,
            unit=unit,
        )
        if observation is None:
            return None
        return _attach_coreference_evidence(_strict_fhir(observation), coreference)

    concept = _strict_codeable_concept(grounded)
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
    spans: tuple[tuple[GroundedSpan, str, str], ...],
    *,
    coreference_by_offset: Mapping[tuple[int, int], _CoreferenceBinding],
) -> tuple[
    tuple[GroundedSpan, str, str, _CoreferenceBinding | None],
    ...,
]:
    grouped: dict[
        tuple[str, str, str],
        tuple[GroundedSpan, str, str, _CoreferenceBinding | None],
    ] = {}
    for index, (span, resource_type, label) in enumerate(spans):
        binding = coreference_by_offset.get((span.start, span.end))
        if binding is None or not _asserted_span(span).status.patient_subject:
            key = ("span", str(index), "")
        else:
            key = (
                "coreference",
                binding.chain.chain_id,
                resource_type,
            )
        current = grouped.get(key)
        if current is None or _is_representative_span(span, binding):
            grouped[key] = (span, resource_type, label, binding)
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


def _resource_type(
    grounded: GroundedSpan,
    resource: str | None,
    *,
    system_routes: Mapping[str, str],
    allow_default_system_fallback: bool,
) -> str | None:
    if resource is not None:
        return _normalize_resource_type(resource)
    label = _normalized_label(grounded)
    if label != _UNLABELED:
        resource_type = _RESOURCE_BY_LABEL.get(label)
        if resource_type is not None or not allow_default_system_fallback:
            return resource_type
    if grounded.candidates:
        system = grounded.candidates[0].system.strip().upper()
        if system in system_routes:
            return system_routes[system]
        if allow_default_system_fallback:
            return _RESOURCE_BY_SYSTEM.get(system)
    return None


def _normalized_label(grounded: GroundedSpan) -> str:
    label = grounded.canonical_label
    return label.strip().upper() if label is not None else _UNLABELED


def _normalize_resource_type(resource: str) -> str:
    if not isinstance(resource, str):
        raise TypeError("FHIR resource type must be a string")
    normalized = resource.strip().casefold()
    matches = {candidate.casefold(): candidate for candidate in FHIR_RESOURCE_TYPES}
    if normalized not in matches:
        raise ValueError(
            f"resource must be one of {FHIR_RESOURCE_TYPES!r}, got {resource!r}"
        )
    return matches[normalized]


def _normalize_system_routes(
    systems: Mapping[str, str] | None,
) -> dict[str, str]:
    if systems is None:
        return {}
    if not isinstance(systems, Mapping):
        raise TypeError("systems must be a mapping")
    routes: dict[str, str] = {}
    for system, resource_type in systems.items():
        if not isinstance(system, str) or not system.strip():
            raise ValueError("systems keys must be non-empty strings")
        routes[system.strip().upper()] = _normalize_resource_type(resource_type)
    return routes


def _resolve_document_id(doc_id: str, document_id: str | None) -> str:
    if document_id is not None and doc_id != _DEFAULT_DOCUMENT_ID:
        if document_id != doc_id:
            raise ValueError("doc_id and document_id must match when both are provided")
    resolved = document_id if document_id is not None else doc_id
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError("doc_id must be a non-empty string")
    return resolved


def _asserted_span(grounded: GroundedSpan) -> AssertedGroundedSpan:
    raw_assertion = grounded.assertion
    if isinstance(raw_assertion, ClinicalContextResult):
        assertion = raw_assertion.to_assertion()
    else:
        assertion = raw_assertion or ClinicalAssertion(
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
        for key in tuple(node):
            if str(key).startswith("_"):
                node.pop(key)
        extensions = node.get("extension")
        if isinstance(extensions, list):
            retained_urls = {
                GROUNDED_CODE_PROVENANCE_EXTENSION_URL,
                MEDICAL_DEVICE_ASSIST_EXTENSION_URL,
                POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL,
            }
            retained = [
                extension
                for extension in extensions
                if not (
                    isinstance(extension, Mapping)
                    and str(extension.get("url") or "").startswith(
                        "https://openmed.ai/fhir/StructureDefinition/"
                    )
                    and extension.get("url") not in retained_urls
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
