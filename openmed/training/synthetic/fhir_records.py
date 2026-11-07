"""Referentially closed synthetic FHIR fixtures from existing Journey fact graphs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import date, datetime, timezone
from itertools import islice

from openmed.clinical.exporters.fhir.bundle import to_bundle
from openmed.clinical.exporters.fhir.validate import validate_bundle
from openmed.clinical.journey_contracts import ClinicalFact

_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SYSTEM = "urn:openmed:synthetic-facts"
_TYPES = {
    "condition": "Condition",
    "medication": "MedicationStatement",
    "laboratory": "Observation",
    "procedure": "Procedure",
    "outcome": "Observation",
}


def _resource_id(kind: str, source: str, seed: int) -> str:
    return hashlib.sha256(f"{kind}:{source}:{seed}".encode()).hexdigest()[:32]


def _synthetic_facts(facts: tuple[ClinicalFact, ...]) -> tuple[ClinicalFact, ...]:
    facts = tuple(islice(facts, 1001))
    if (
        not facts
        or len(facts) > 1000
        or any(not isinstance(f, ClinicalFact) for f in facts)
    ):
        raise ValueError("invalid synthetic FHIR fact collection")
    if len({f.fact_id for f in facts}) != len(facts):
        raise ValueError("duplicate synthetic FHIR fact identity")
    indexed = {f.fact_id: f for f in facts}
    for fact in facts:
        if (
            fact.attributes.get("synthetic") is not True
            or fact.status != "synthetic"
            or fact.fact_type not in _TYPES.keys() | {"encounter"}
        ):
            raise ValueError("FHIR fixture requires declared synthetic facts")
        if not isinstance(fact.value, Mapping) or set(fact.value) != {"synthetic_code"}:
            raise ValueError("invalid synthetic FHIR fact value")
        code = fact.value["synthetic_code"]
        if not isinstance(code, str) or not _CODE.fullmatch(code):
            raise ValueError("invalid synthetic FHIR terminology placeholder")
        effective = fact.effective_time.get("date")
        observed = fact.attributes.get("observed_at")
        try:
            day = date.fromisoformat(effective)
            stamp = datetime.fromisoformat(observed)
            if stamp.tzinfo is None or day > stamp.date():
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError("invalid synthetic FHIR fact chronology") from None
        for parent_id in fact.parent_fact_ids:
            parent = indexed.get(parent_id)
            if (
                parent is None
                or parent.subject_id != fact.subject_id
                or parent.effective_time.get("date", "") > effective
            ):
                raise ValueError("invalid synthetic FHIR parent reference")
    completed = set()
    remaining = set(indexed)
    while remaining:
        ready = {
            code
            for code in remaining
            if set(indexed[code].parent_fact_ids) <= completed
        }
        if not ready:
            raise ValueError("synthetic graph contains a causal cycle")
        completed |= ready
        remaining -= ready
    return tuple(
        sorted(
            facts,
            key=lambda f: (
                f.subject_id,
                f.effective_time["date"],
                f.fact_type,
                f.fact_id,
            ),
        )
    )


def generate_fhir_records(
    facts: tuple[ClinicalFact, ...], *, seed: int = 0
) -> dict[str, object]:
    """Create an offline synthetic R4 collection Bundle with no write requests.

    Inputs use the existing ClinicalFact graph and explicit synthetic markers.
    Terminology is local placeholder coding; no restricted vocabularies or
    inferred clinical states are introduced.
    """
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("invalid synthetic FHIR seed")
    facts = _synthetic_facts(facts)
    subjects = sorted({f.subject_id for f in facts})
    patient_ids = {s: _resource_id("Patient", s, seed) for s in subjects}
    encounters = {}
    for fact in facts:
        if fact.fact_type == "encounter":
            if fact.encounter_id is None or fact.encounter_id in encounters:
                raise ValueError("invalid synthetic FHIR encounter inventory")
            encounters[fact.encounter_id] = fact
    for fact in facts:
        encounter = encounters.get(fact.encounter_id)
        if (
            encounter is None
            or encounter.subject_id != fact.subject_id
            or encounter.effective_time["date"] > fact.effective_time["date"]
        ):
            raise ValueError("synthetic FHIR fact has no matching encounter")
    encounter_ids = {e: _resource_id("Encounter", e, seed) for e in encounters}
    resources = []
    tag = {"tag": [{"system": _SYSTEM, "code": "synthetic"}]}
    for subject in subjects:
        resources.append(
            {"resourceType": "Patient", "id": patient_ids[subject], "meta": tag}
        )
    for encounter_id, fact in sorted(encounters.items()):
        resources.append(
            {
                "resourceType": "Encounter",
                "id": encounter_ids[encounter_id],
                "meta": tag,
                "status": "unknown",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB",
                },
                "subject": {"reference": "Patient/" + patient_ids[fact.subject_id]},
                "period": {"start": fact.effective_time["date"]},
            }
        )
    targets = {}
    for fact in facts:
        if fact.fact_type == "encounter":
            targets[fact.fact_id] = "Encounter/" + encounter_ids[fact.encounter_id]
            continue
        kind = _TYPES[fact.fact_type]
        resource_id = _resource_id(kind, fact.fact_id, seed)
        resource = {
            "resourceType": kind,
            "id": resource_id,
            "meta": tag,
            "subject": {"reference": "Patient/" + patient_ids[fact.subject_id]},
        }
        code = {"coding": [{"system": _SYSTEM, "code": fact.value["synthetic_code"]}]}
        encounter = {"reference": "Encounter/" + encounter_ids[fact.encounter_id]}
        effective = fact.effective_time["date"]
        if kind == "MedicationStatement":
            resource.update(
                status="unknown",
                medicationCodeableConcept=code,
                context=encounter,
                effectiveDateTime=effective,
            )
        elif kind == "Condition":
            resource.update(code=code, encounter=encounter, onsetDateTime=effective)
        elif kind == "Procedure":
            resource.update(
                status="unknown",
                code=code,
                encounter=encounter,
                performedDateTime=effective,
            )
        else:
            resource.update(
                status="final",
                code=code,
                encounter=encounter,
                effectiveDateTime=effective,
                valueString=fact.value["synthetic_code"],
            )
        resources.append(resource)
        targets[fact.fact_id] = kind + "/" + resource_id
    for fact in facts:
        resources.append(
            {
                "resourceType": "Provenance",
                "id": _resource_id("Provenance", fact.fact_id, seed),
                "meta": tag,
                "target": [{"reference": targets[fact.fact_id]}],
                "recorded": fact.attributes["observed_at"],
                "agent": [
                    {
                        "who": {
                            "identifier": {
                                "system": _SYSTEM,
                                "value": "synthetic_fixture_generator",
                            }
                        }
                    }
                ],
                "entity": [
                    {
                        "role": "source",
                        "what": {
                            "identifier": {"system": _SYSTEM, "value": fact.fact_id}
                        },
                    }
                ],
            }
        )
    bundle = to_bundle(
        resources, doc_id="synthetic-fhir-" + str(seed), bundle_type="collection"
    )
    if validate_synthetic_fhir(bundle):
        raise ValueError("synthetic FHIR generation failed fixture constraints")
    return bundle


def validate_synthetic_fhir(bundle: Mapping[str, object]) -> tuple[str, ...]:
    """Independently check cardinality subset, references, synthetic tags and time.

    OpenMed's bundled validator covers a base R4 subset, not full server or
    implementation-guide conformance. Patient/Provenance receive explicit
    fixture checks here; no broader conformance assurance is inferred.
    """
    if (
        not isinstance(bundle, Mapping)
        or not isinstance(bundle.get("entry"), list)
        or len(bundle["entry"]) > 3000
    ):
        raise ValueError("invalid synthetic FHIR bundle")
    errors = set()
    if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "collection":
        errors.add("fixture_bundle_type")
    if not validate_bundle(bundle).is_valid:
        errors.add("base_cardinality_or_type")
    indexed = {}
    for entry in bundle["entry"]:
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("resource"), Mapping
        ):
            errors.add("entry_structure")
            continue
        if "request" in entry:
            errors.add("unexpected_write_request")
        fullurl = entry.get("fullUrl")
        if not isinstance(fullurl, str) or fullurl in indexed:
            errors.add("duplicate_or_missing_fullurl")
            continue
        resource = entry["resource"]
        indexed[fullurl] = resource
        meta = resource.get("meta")
        if (
            not isinstance(meta, Mapping)
            or not isinstance(meta.get("tag"), list)
            or {"system": _SYSTEM, "code": "synthetic"} not in meta["tag"]
        ):
            errors.add("synthetic_tag_missing")

    inspected = 0

    def references(value):
        nonlocal inspected
        pending = [(value, 0)]
        seen = set()
        found = []
        while pending:
            node, depth = pending.pop()
            inspected += 1
            if inspected > 100000 or depth > 32:
                raise ValueError("synthetic FHIR reference inspection limit exceeded")
            if isinstance(node, (Mapping, list)):
                if id(node) in seen:
                    continue
                seen.add(id(node))
                if len(node) + len(pending) > 100000:
                    raise ValueError(
                        "synthetic FHIR reference inspection limit exceeded"
                    )
                if isinstance(node, Mapping):
                    if "reference" in node:
                        found.append(node["reference"])
                    pending.extend((child, depth + 1) for child in node.values())
                else:
                    pending.extend((child, depth + 1) for child in node)
        return found

    provenance_targets = set()
    provenance_sources = set()

    def instant(value):
        if not isinstance(value, str):
            raise ValueError
        if len(value) == 10:
            return datetime.combine(
                date.fromisoformat(value), datetime.min.time(), timezone.utc
            )
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError
        return parsed

    for resource in indexed.values():
        try:
            refs = references(resource)
        except ValueError:
            errors.add("reference_structure_limit")
            continue
        if any(not isinstance(ref, str) or ref not in indexed for ref in refs):
            errors.add("dangling_reference")
        if resource.get("resourceType") == "Provenance":
            if (
                not resource.get("target")
                or not resource.get("agent")
                or not isinstance(resource.get("recorded"), str)
            ):
                errors.add("provenance_incomplete")
            targets = resource.get("target")
            entities = resource.get("entity")
            if (
                not isinstance(targets, list)
                or len(targets) != 1
                or not isinstance(targets[0], Mapping)
                or not isinstance(targets[0].get("reference"), str)
            ):
                errors.add("provenance_incomplete")
            else:
                target = targets[0]["reference"]
                if (
                    target in provenance_targets
                    or target not in indexed
                    or indexed[target].get("resourceType") in {"Patient", "Provenance"}
                ):
                    errors.add("provenance_target_mismatch")
                provenance_targets.add(target)
                target_resource = indexed.get(target, {})
                target_period = target_resource.get("period")
                effective = target_resource.get(
                    "effectiveDateTime",
                    target_resource.get(
                        "onsetDateTime",
                        target_resource.get(
                            "performedDateTime",
                            target_period.get("start")
                            if isinstance(target_period, Mapping)
                            else None,
                        ),
                    ),
                )
                try:
                    recorded_value = resource.get("recorded")
                    if not isinstance(recorded_value, str):
                        raise ValueError
                    recorded_time = datetime.fromisoformat(recorded_value)
                    if recorded_time.tzinfo is None:
                        raise ValueError
                    if effective is not None and recorded_time < instant(effective):
                        errors.add("provenance_before_target")
                except (TypeError, ValueError):
                    errors.add("provenance_time_invalid")
            if (
                not isinstance(entities, list)
                or len(entities) != 1
                or not isinstance(entities[0], Mapping)
            ):
                errors.add("provenance_source_missing")
            else:
                entity = entities[0]
                what = entity.get("what")
                identifier = (
                    what.get("identifier") if isinstance(what, Mapping) else None
                )
                if (
                    not isinstance(identifier, Mapping)
                    or identifier.get("system") != _SYSTEM
                    or not isinstance(identifier.get("value"), str)
                    or not re.fullmatch(r"fact_[0-9a-f]{16,64}", identifier["value"])
                    or entity.get("role") != "source"
                ):
                    errors.add("provenance_source_missing")
                else:
                    source = identifier["value"]
                    if source in provenance_sources:
                        errors.add("provenance_source_duplicate")
                    provenance_sources.add(source)
            continue
        if resource.get("resourceType") == "Patient":
            continue
        subject = resource.get("subject")
        subject_ref = subject.get("reference") if isinstance(subject, Mapping) else None
        if (
            not isinstance(subject_ref, str)
            or indexed.get(subject_ref, {}).get("resourceType") != "Patient"
        ):
            errors.add("subject_reference_type")
        encounter_node = resource.get("encounter", resource.get("context", {}))
        if not isinstance(encounter_node, Mapping):
            errors.add("encounter_reference_structure")
            continue
        encounter_ref = encounter_node.get("reference")
        if encounter_ref is not None and not isinstance(encounter_ref, str):
            errors.add("encounter_reference_structure")
            continue
        encounter = indexed.get(encounter_ref)
        effective = resource.get(
            "effectiveDateTime",
            resource.get("onsetDateTime", resource.get("performedDateTime")),
        )
        if resource.get("resourceType") != "Encounter":
            if encounter is None or encounter.get("resourceType") != "Encounter":
                errors.add("encounter_reference_type")
            elif encounter.get("subject") != resource.get("subject"):
                errors.add("encounter_subject_mismatch")
        if encounter is not None and effective is not None:
            try:
                period = encounter.get("period")
                if not isinstance(period, Mapping) or instant(
                    period.get("start")
                ) > instant(effective):
                    errors.add("clinical_time_before_encounter")
            except (TypeError, ValueError):
                errors.add("clinical_time_invalid")
    required_targets = {
        ref
        for ref, resource in indexed.items()
        if resource.get("resourceType") not in {"Patient", "Provenance"}
    }
    if provenance_targets != required_targets:
        errors.add("provenance_coverage_missing")
    return tuple(sorted(errors))
