"""Offline, review-bound FHIR R4 projection of guarded clinical relations."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from openmed.clinical.family_history import FamilyHistoryRecord
from openmed.clinical.relations.evidence_binding import AssertionState, GuardedRelation
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewTransition,
    ReviewTransitionPolicy,
    validate_review_history,
)

from .bundle import to_bundle
from .reference_types import find_reference_target_issues
from .validate import validate_resource

__all__ = [
    "DEFAULT_RELATION_CODE_SYSTEMS",
    "RelationFHIRExportError",
    "ReviewedFHIRRelation",
    "RelationFHIRLoss",
    "RelationFHIRExport",
    "relation_fhir_review_fingerprint",
    "export_reviewed_relations",
]

_BASE = "https://openmed.ai/fhir/StructureDefinition/reviewed-relation"
_ROLE_SYSTEM = "https://openmed.ai/fhir/CodeSystem/family-role"
_KINDS = frozenset(
    {
        "Patient",
        "Device",
        "RelatedPerson",
        "Condition",
        "MedicationStatement",
        "Procedure",
        "Observation",
        "DiagnosticReport",
    }
)
_REFERENCE = re.compile(r"([A-Za-z]+)/([A-Za-z0-9.-]{1,64})\Z")
_ROLES = frozenset(
    {
        "maternal_grandmother",
        "paternal_grandmother",
        "maternal_grandfather",
        "paternal_grandfather",
        "grandmother",
        "grandfather",
        "grandparent",
        "mother",
        "father",
        "parent",
        "sister",
        "brother",
        "sibling",
        "daughter",
        "son",
        "child",
        "aunt",
        "uncle",
        "cousin",
        "niece",
        "nephew",
        "spouse",
        "maternal",
        "paternal",
        "family",
        "relative",
    }
)
DEFAULT_RELATION_CODE_SYSTEMS = (
    "http://hl7.org/fhir/sid/icd-10-cm",
    "http://hl7.org/fhir/sid/icd-10",
    "http://loinc.org",
    "http://snomed.info/sct",
    "http://www.nlm.nih.gov/research/umls/rxnorm",
    "http://terminology.hl7.org/CodeSystem/condition-clinical",
    "http://terminology.hl7.org/CodeSystem/condition-ver-status",
    "https://openmed.ai/fhir/CodeSystem/synthetic-test",
)
_LOSS_CODES = frozenset(
    {
        "unreviewed",
        "rejected",
        "review_not_approved",
        "review_invalid",
        "review_mismatch",
        "assertion_refused",
        "non_patient",
        "unsupported_relation",
        "endpoint_missing",
        "endpoint_invalid",
        "reference_type_refused",
        "family_binding_invalid",
        "attribution_conflict",
        "duplicate_relation",
    }
)


class RelationFHIRExportError(ValueError):
    """Fixed, source-free failure at the relation-export boundary."""


def _invalid() -> RelationFHIRExportError:
    return RelationFHIRExportError("Invalid reviewed FHIR relation input.")


def _reference(value: Any) -> str:
    if type(value) is not str:
        raise _invalid()
    match = _REFERENCE.fullmatch(value)
    if match is None or len(match[1]) > 64:
        raise _invalid()
    return value


def _digest(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _opaque_id(value: str) -> str:
    return (
        "openmed-"
        + hashlib.sha256(("reviewed-fhir-resource\0" + value).encode()).hexdigest()[:48]
    )


@dataclass(frozen=True)
class ReviewedFHIRRelation:
    """A guarded candidate, explicit attribution and existing review history.

    The history is supplied by the caller's review workflow. This record does
    not create approval events or authenticate a clinician's identity.

    Attributes:
        relation: Existing value-free, evidence-bound candidate.
        head_reference: Already exported head resource, as ResourceType/id.
        tail_reference: Already exported tail, or None for a family role.
        experiencer: Explicit patient, family or other attribution.
        review_transitions: Existing transitions bound to the export fingerprint.
        family_record: Required offset-bound family role for condition_to_relative.
    """

    relation: GuardedRelation
    head_reference: str = field(repr=False)
    tail_reference: str | None = field(repr=False)
    experiencer: str
    review_transitions: tuple[ReviewTransition, ...] = ()
    family_record: FamilyHistoryRecord | None = None

    def __post_init__(self) -> None:
        try:
            if type(self.relation) is not GuardedRelation:
                raise _invalid()
            relation = replace(self.relation)
            for span in (relation.head, relation.tail, *relation.evidence_spans):
                if span.end > 2_147_483_647:
                    raise _invalid()
            if type(self.experiencer) is not str or self.experiencer not in (
                "patient",
                "family",
                "other",
            ):
                raise _invalid()
            if (
                type(self.review_transitions) is not tuple
                or len(self.review_transitions) > 128
            ):
                raise _invalid()
            if any(type(t) is not ReviewTransition for t in self.review_transitions):
                raise _invalid()
            object.__setattr__(self, "relation", relation)
            object.__setattr__(self, "head_reference", _reference(self.head_reference))
            if self.tail_reference is not None:
                object.__setattr__(
                    self, "tail_reference", _reference(self.tail_reference)
                )
            object.__setattr__(
                self,
                "review_transitions",
                tuple(replace(t) for t in self.review_transitions),
            )
            if self.family_record is not None:
                if type(self.family_record) is not FamilyHistoryRecord:
                    raise _invalid()
                object.__setattr__(self, "family_record", replace(self.family_record))
        except Exception:
            raise _invalid() from None


@dataclass(frozen=True)
class RelationFHIRLoss:
    """One refused input index with a controlled, source-free reason."""

    index: int
    code: str

    def __post_init__(self) -> None:
        if (
            type(self.index) is not int
            or self.index < 0
            or self.code not in _LOSS_CODES
        ):
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        """Return the index and fixed loss code."""
        return {"index": self.index, "code": self.code}


@dataclass(frozen=True)
class RelationFHIRExport:
    """Collection Bundle and value-free projection losses; no write authority."""

    bundle: dict[str, Any]
    accepted_count: int
    losses: tuple[RelationFHIRLoss, ...]
    omitted_endpoint_field_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return the FHIR collection plus controlled counts and losses."""
        return {
            "bundle": json.loads(json.dumps(self.bundle)),
            "accepted_count": self.accepted_count,
            "losses": [loss.to_dict() for loss in self.losses],
            "loss_counts": dict(
                sorted(Counter(loss.code for loss in self.losses).items())
            ),
            "omitted_endpoint_field_count": self.omitted_endpoint_field_count,
            "review_required": True,
            "write_authority": False,
        }


def _systems(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or len(values) > 64:
        raise _invalid()
    result = []
    for value in values:
        if type(value) is not str or len(value) > 256:
            raise _invalid()
        if (
            re.fullmatch(
                r"(?:https?://[A-Za-z0-9.-]+(?:/[A-Za-z0-9/._-]+)?|urn:[A-Za-z0-9:._-]+)",
                value,
            )
            is None
        ):
            raise _invalid()
        result.append(value)
    return tuple(sorted(set(result)))


def _store(resources: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if isinstance(resources, (str, bytes)) or len(resources) > 1024:
        raise _invalid()
    encoded = json.dumps(resources, ensure_ascii=True, allow_nan=False)
    if len(encoded.encode()) > 4_194_304:
        raise _invalid()
    plain = json.loads(encoded)
    pending = [(plain, 0)]
    count = 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if depth > 32 or count > 65_536:
            raise _invalid()
        if isinstance(value, dict):
            pending.extend((v, depth + 1) for v in value.values())
        elif isinstance(value, list):
            pending.extend((v, depth + 1) for v in value)
    result = {}
    for resource in plain:
        if type(resource) is not dict:
            raise _invalid()
        kind = resource.get("resourceType")
        identity = resource.get("id")
        if type(kind) is not str or type(identity) is not str:
            raise _invalid()
        reference = _reference(kind + "/" + identity)
        if reference in result:
            raise _invalid()
        result[reference] = resource
    return result


def _fingerprint(
    item: ReviewedFHIRRelation,
    store: Mapping[str, Any],
    patient: str,
    systems: tuple[str, ...],
) -> str:
    refs = sorted(
        {
            patient,
            item.head_reference,
            *(() if item.tail_reference is None else (item.tail_reference,)),
        }
    )
    return "sha256:" + _digest(
        {
            "contract": "openmed.reviewed_fhir_relation.v1",
            "relation": item.relation.to_dict(),
            "head": item.head_reference,
            "tail": item.tail_reference,
            "experiencer": item.experiencer,
            "family": item.family_record.to_dict() if item.family_record else None,
            "patient": patient,
            "code_systems": systems,
            "endpoints": {
                ref: _digest(store[ref]) if ref in store else None for ref in refs
            },
        }
    )


def relation_fhir_review_fingerprint(
    item: ReviewedFHIRRelation,
    resources: Sequence[Mapping[str, Any]],
    patient_reference: str,
    *,
    code_systems: Sequence[str] = DEFAULT_RELATION_CODE_SYSTEMS,
) -> str:
    """Bind review to the candidate, attribution, resources and terminology policy.

    Args:
        item: Candidate and endpoint bindings, without newly created approvals.
        resources: Already exported endpoint resources and the active Patient.
        patient_reference: Explicit active Patient reference.
        code_systems: Trusted caller-declared terminology systems permitted in output.

    Returns:
        A sha256-prefixed fingerprint for the existing local review state machine.

    Raises:
        RelationFHIRExportError: If the bounded input contract is invalid.
    """
    try:
        if type(item) is not ReviewedFHIRRelation:
            raise _invalid()
        item = replace(item)
        patient = _reference(patient_reference)
        if not patient.startswith("Patient/"):
            raise _invalid()
        return _fingerprint(item, _store(resources), patient, _systems(code_systems))
    except Exception:
        raise _invalid() from None


def _concept(value: Any, systems: tuple[str, ...]) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("coding")) is not list:
        raise _invalid()
    if not 1 <= len(value["coding"]) <= 16:
        raise _invalid()
    coding = []
    for item in value["coding"]:
        if type(item) is not dict:
            raise _invalid()
        system, code = item.get("system"), item.get("code")
        if system not in systems or type(code) is not str:
            raise _invalid()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", code) is None:
            raise _invalid()
        coding.append({"system": system, "code": code})
    return {"coding": coding}


def _endpoint(
    resource: dict[str, Any], reference: str, patient: str, systems: tuple[str, ...]
) -> tuple[dict[str, Any], int]:
    kind = resource["resourceType"]
    if kind not in _KINDS or resource.get("modifierExtension"):
        raise _invalid()
    projected: dict[str, Any] = {"resourceType": kind, "id": _opaque_id(reference)}
    if kind in ("Patient", "Device"):
        return projected, max(0, len(resource) - 2)
    subject_key = "patient" if kind == "RelatedPerson" else "subject"
    subject = resource.get(subject_key)
    if type(subject) is not dict or subject.get("reference") != patient:
        raise _invalid()
    projected[subject_key] = {"reference": patient}
    retained = {"resourceType", "id", subject_key}
    if kind == "RelatedPerson":
        return projected, max(0, len(resource) - len(retained))
    concept_key = (
        "medicationCodeableConcept" if kind == "MedicationStatement" else "code"
    )
    projected[concept_key] = _concept(resource.get(concept_key), systems)
    retained.add(concept_key)
    if kind != "Condition":
        projected["status"] = resource.get("status")
        retained.add("status")
    else:
        for key in ("clinicalStatus", "verificationStatus"):
            if key in resource:
                projected[key] = _concept(resource[key], systems)
                retained.add(key)
        if any(
            coding["code"] in ("refuted", "entered-in-error")
            for coding in projected.get("verificationStatus", {}).get("coding", [])
        ):
            raise _invalid()
    if projected.get("status") == "entered-in-error":
        raise _invalid()
    result = validate_resource(projected)
    if result.errors:
        raise _invalid()
    return projected, max(0, len(resource) - len(retained))


def _review_loss(
    item: ReviewedFHIRRelation, fingerprint: str, policy: ReviewTransitionPolicy | None
) -> str | None:
    if not item.review_transitions:
        return "unreviewed"
    try:
        report = validate_review_history(item.review_transitions, policy=policy)
    except Exception:
        return "review_invalid"
    if any(t.provenance_fingerprint != fingerprint for t in item.review_transitions):
        return "review_mismatch"
    if report.current_state is ReviewState.REJECTED:
        return "rejected"
    if report.current_state is not ReviewState.APPROVED:
        return "review_not_approved"
    if item.review_transitions[-1].from_state is not ReviewState.IN_REVIEW:
        return "review_invalid"
    return None


def _link_direction(item: ReviewedFHIRRelation) -> tuple[str, str] | None:
    kind = item.relation.relation_type
    if kind == "diagnosis_to_treatment":
        if item.tail_reference is None:
            return None
        return item.tail_reference, item.head_reference
    if kind in (
        "procedure_to_indication",
        "drug_to_reason",
        "drug_to_indication",
        "medication_change",
    ):
        if item.tail_reference is None:
            return None
        return item.head_reference, item.tail_reference
    return None


def _technical_now() -> datetime:
    return datetime.now(timezone.utc)


def _provenance(
    item: ReviewedFHIRRelation,
    target: str,
    fingerprint: str,
    recorded: str,
    software: str,
    index: int,
) -> dict[str, Any]:
    extensions = []
    for role, span in (
        ("head", item.relation.head),
        ("tail", item.relation.tail),
        *(("link", span) for span in item.relation.evidence_spans),
    ):
        extensions.append(
            {
                "url": _BASE + "/evidence-offset",
                "extension": [
                    {"url": "role", "valueCode": role},
                    {"url": "start", "valueUnsignedInt": span.start},
                    {"url": "end", "valueUnsignedInt": span.end},
                ],
            }
        )
    return {
        "resourceType": "Provenance",
        "id": _opaque_id(f"provenance:{fingerprint}:{index}"),
        "recorded": recorded,
        "target": [{"reference": target}],
        "agent": [{"who": {"reference": software}}],
        "entity": [
            {
                "role": "source",
                "what": {
                    "identifier": {
                        "system": "https://openmed.ai/fhir/sid/relation-document-digest",
                        "value": item.relation.document_id,
                    }
                },
                "extension": extensions,
            }
        ],
        "extension": [
            {"url": _BASE + "/review-fingerprint", "valueString": fingerprint},
            {
                "url": _BASE + "/review-history-digest",
                "valueString": _digest([t.to_dict() for t in item.review_transitions]),
            },
        ],
    }


def export_reviewed_relations(
    relations: Sequence[ReviewedFHIRRelation],
    resources: Sequence[Mapping[str, Any]],
    *,
    patient_reference: str,
    code_systems: Sequence[str] = DEFAULT_RELATION_CODE_SYSTEMS,
    review_policy: ReviewTransitionPolicy | None = None,
    clock: Callable[[], datetime] = _technical_now,
) -> RelationFHIRExport:
    """Project reviewed relations into an offline FHIR collection with losses.

    Args:
        relations: Evidence-bound candidates and existing local review histories.
        resources: Already exported endpoint resources, including the active Patient.
        patient_reference: Explicit active Patient identity used for scope checks.
        code_systems: Trusted terminology policy; no dictionaries are bundled.
        review_policy: Existing configured transition policy, or the default policy.
        clock: Technical Provenance creation time; never a clinical event date.

    Returns:
        Sanitized endpoint resources, approved links/family history, Provenance,
        and a controlled loss for every candidate that cannot be projected.

    Raises:
        RelationFHIRExportError: If the bounded input or technical clock is invalid.
    """
    try:
        return _export(
            relations, resources, patient_reference, code_systems, review_policy, clock
        )
    except Exception:
        raise _invalid() from None


def _export(
    relations: Any,
    resources: Any,
    patient: Any,
    code_systems: Any,
    policy: Any,
    clock: Any,
) -> RelationFHIRExport:
    if isinstance(relations, (str, bytes)) or len(relations) > 512:
        raise _invalid()
    if any(type(item) is not ReviewedFHIRRelation for item in relations):
        raise _invalid()
    rows = tuple(replace(item) for item in relations)
    systems = _systems(code_systems)
    patient = _reference(patient)
    if not patient.startswith("Patient/"):
        raise _invalid()
    store = _store(resources)
    if patient not in store:
        raise _invalid()
    if policy is not None and type(policy) is not ReviewTransitionPolicy:
        raise _invalid()
    losses: list[RelationFHIRLoss] = []
    admitted = []
    seen = set()
    for index, item in enumerate(rows):
        fingerprint = _fingerprint(item, store, patient, systems)
        reason = _review_loss(item, fingerprint, policy)
        if reason is None and item.relation.assertion_state not in (
            AssertionState.AFFIRMED,
            AssertionState.CONFIRMED,
            AssertionState.HISTORICAL,
        ):
            reason = "assertion_refused"
        family = item.relation.relation_type == "condition_to_relative"
        if reason is None and item.experiencer != ("family" if family else "patient"):
            reason = "non_patient"
        if reason is None and fingerprint in seen:
            reason = "duplicate_relation"
        if reason is None and any(
            ref not in store
            for ref in (
                item.head_reference,
                *(() if item.tail_reference is None else (item.tail_reference,)),
            )
        ):
            reason = "endpoint_missing"
        if reason is None and family:
            record = item.family_record
            if (
                record is None
                or record.condition_span != item.relation.head.offset
                or record.relative_span != item.relation.tail.offset
                or record.relative_role not in _ROLES
                or not item.head_reference.startswith("Condition/")
                or (
                    item.tail_reference is not None
                    and not item.tail_reference.startswith("RelatedPerson/")
                )
            ):
                reason = "family_binding_invalid"
        if reason is None and not family and _link_direction(item) is None:
            reason = "unsupported_relation"
        if reason is not None:
            losses.append(RelationFHIRLoss(index, reason))
            continue
        seen.add(fingerprint)
        admitted.append((index, item, fingerprint, family))
    family_heads = {item.head_reference for _, item, _, family in admitted if family}
    patient_heads = {
        ref
        for _, item, _, family in admitted
        if not family
        for ref in (item.head_reference, item.tail_reference)
        if ref is not None
    }
    conflicts = family_heads & patient_heads
    output: dict[str, dict[str, Any]] = {}
    accepted = []
    omitted = 0
    for index, item, fingerprint, family in admitted:
        if item.head_reference in conflicts or item.tail_reference in conflicts:
            losses.append(RelationFHIRLoss(index, "attribution_conflict"))
            continue
        needed = {
            patient,
            item.head_reference,
            *(() if item.tail_reference is None else (item.tail_reference,)),
        }
        staged = {}
        try:
            for ref in needed:
                staged[ref] = _endpoint(store[ref], ref, patient, systems)
            if family:
                record = item.family_record
                assert record is not None
                target = "FamilyMemberHistory/" + _opaque_id("family:" + fingerprint)
                new = {
                    "resourceType": "FamilyMemberHistory",
                    "id": target.split("/")[1],
                    "status": "partial",
                    "patient": {"reference": patient},
                    "relationship": {
                        "coding": [
                            {"system": _ROLE_SYSTEM, "code": record.relative_role}
                        ]
                    },
                    "condition": [{"code": staged[item.head_reference][0]["code"]}],
                }
                if validate_resource(new).errors:
                    raise _invalid()
            else:
                direction = _link_direction(item)
                assert direction is not None
                source, target_ref = direction
                source_resource = staged[source][0]
                expected_sources = (
                    ("MedicationStatement", "Procedure")
                    if item.relation.relation_type == "diagnosis_to_treatment"
                    else ("Procedure",)
                    if item.relation.relation_type == "procedure_to_indication"
                    else ("MedicationStatement",)
                )
                if source_resource["resourceType"] not in expected_sources:
                    losses.append(RelationFHIRLoss(index, "reference_type_refused"))
                    continue
                candidate = dict(
                    source_resource, reasonReference=[{"reference": target_ref}]
                )
                endpoints = [
                    dict(resource, id=ref.split("/")[1])
                    for ref, (resource, _) in staged.items()
                    if ref != source
                ]
                candidate["id"] = source.split("/")[1]
                if find_reference_target_issues(
                    [*endpoints, candidate], fhir_version="R4"
                ):
                    losses.append(RelationFHIRLoss(index, "reference_type_refused"))
                    continue
                target = source
                new = None
        except Exception:
            losses.append(RelationFHIRLoss(index, "endpoint_invalid"))
            continue
        for ref, (resource, count) in staged.items():
            if family and ref == item.head_reference:
                continue  # A relative's code template never becomes a patient Condition.
            if ref not in output:
                output[ref] = resource
                omitted += count
        if family:
            assert new is not None
            output[target] = new
        else:
            links = output[target].setdefault("reasonReference", [])
            link = {"reference": target_ref}
            if link not in links:
                links.append(link)
        accepted.append((index, item, fingerprint, target))
    if not accepted:
        return RelationFHIRExport(
            {"resourceType": "Bundle", "type": "collection", "entry": []},
            0,
            tuple(sorted(losses, key=lambda x: x.index)),
            0,
        )
    instant = clock()
    if type(instant) is not datetime or instant.utcoffset() is None:
        raise _invalid()
    recorded = instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    software_id = _opaque_id("software:openmed-reviewed-relations-v1")
    software = "Device/" + software_id
    output[software] = {"resourceType": "Device", "id": software_id}
    for index, item, fingerprint, target in accepted:
        provenance = _provenance(item, target, fingerprint, recorded, software, index)
        output["Provenance/" + provenance["id"]] = provenance
    # Rename every internal identity/reference before Bundle assembly; raw
    # endpoint ids and source-bearing resource fields never enter the output.
    ref_map = {
        ref: resource["resourceType"] + "/" + resource["id"]
        for ref, resource in output.items()
    }

    def rewrite(node):
        if isinstance(node, dict):
            result = {key: rewrite(value) for key, value in node.items()}
            if "reference" in result:
                if result["reference"] not in ref_map:
                    raise _invalid()
                result["reference"] = ref_map[result["reference"]]
            return result
        if isinstance(node, list):
            return [rewrite(value) for value in node]
        return node

    bundle = to_bundle(
        [rewrite(output[key]) for key in sorted(output)],
        doc_id="reviewed-relations:" + _digest(sorted(f for _, _, f, _ in accepted)),
        bundle_type="collection",
    )
    if find_reference_target_issues(bundle, fhir_version="R4"):
        raise _invalid()
    return RelationFHIRExport(
        bundle, len(accepted), tuple(sorted(losses, key=lambda x: x.index)), omitted
    )
