"""FHIR R4 Journey round-trip, custody, and loss tests."""

from __future__ import annotations

import base64
import copy
import json
from dataclasses import replace
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.clinical.journey import JourneyEvent, JourneyEvidencePath
from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    DatasetSnapshot,
    EvidenceLocator,
    ResolutionEvent,
    derived_opaque_id,
)
from openmed.interop.fhir import (
    OPENMED_FACT_EXTENSION,
    export_journey_to_fhir,
    import_journey_from_fhir,
    load_fhir_journey_schema,
    reference_integrity_report,
    validation_result,
)
from openmed.structured.store import StoreState

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "structured"
    / "journey_contracts.json"
)


def _records() -> tuple[list[ClinicalFact], ResolutionEvent, DatasetSnapshot]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    first = ClinicalFact.from_dict(payload["clinical_fact"])
    facts = [
        first,
        replace(
            first,
            fact_id="fact_bbbbbbbbbbbbbbbb",
            fact_type="observation",
            status="final",
            value={
                "code": "synthetic-lab",
                "system": "https://example.invalid/synthetic-codes",
                "value": 7.25,
            },
            unit="mg",
        ),
        replace(
            first,
            fact_id="fact_cccccccccccccccc",
            fact_type="medication",
            status="active",
            value={"code": "synthetic-medication"},
        ),
        replace(
            first,
            fact_id="fact_dddddddddddddddd",
            fact_type="procedure",
            status="completed",
            value={"code": "synthetic-procedure"},
        ),
    ]
    resolution = replace(
        ResolutionEvent.from_dict(payload["resolution_event"]),
        selected_fact_ids=(first.fact_id,),
        rejected_fact_ids=(facts[1].fact_id,),
    )
    snapshot = replace(
        DatasetSnapshot.from_dict(payload["dataset_snapshot"]),
        record_count=len(facts),
        source_fact_ids=tuple(item.fact_id for item in facts),
    )
    return facts, resolution, snapshot


def test_supported_records_round_trip_exactly_with_valid_references() -> None:
    facts, resolution, snapshot = _records()
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    artifact = ClinicalArtifact.from_dict(payload["clinical_artifact"])
    locator = EvidenceLocator.from_dict(payload["evidence_locator"])
    event = JourneyEvent(
        event_id=derived_opaque_id("journeyevent", facts[0].fact_id),
        position=0,
        event_type="condition",
        journey_state="current",
        correction_state="none",
        fact=facts[0],
        evidence_paths=(JourneyEvidencePath(locator=locator, artifact=artifact),),
    )

    exported = export_journey_to_fhir(
        facts,
        source_snapshot=snapshot,
        resolutions=(resolution,),
        events=(event,),
    )

    assert exported.state is StoreState.SUCCESS
    assert exported.value is not None
    assert exported.value.lossless is True
    bundle = exported.value.bundle
    assert reference_integrity_report(bundle, fhir_version="R4").valid is True
    assert validation_result(bundle, "R4").valid is True
    assert {entry["resource"]["resourceType"] for entry in bundle["entry"]} == {
        "Patient",
        "Condition",
        "Observation",
        "MedicationStatement",
        "Procedure",
        "Provenance",
    }

    imported = import_journey_from_fhir(bundle, expected_snapshot=snapshot)

    assert imported.state is StoreState.SUCCESS
    assert imported.value is not None
    assert imported.value.facts == tuple(sorted(facts, key=lambda item: item.fact_id))
    assert imported.value.resolutions == (resolution,)
    assert len(imported.value.events) == 1
    assert imported.value.events[0].payload == event.to_dict()
    assert imported.value.source_snapshot == snapshot
    assert {item.record_id for item in imported.value.evidence_mappings} == {
        *(item.fact_id for item in facts),
        event.event_id,
        resolution.resolution_id,
    }


def test_export_contract_is_byte_stable_and_matches_schema() -> None:
    facts, resolution, snapshot = _records()
    first = export_journey_to_fhir(
        facts, source_snapshot=snapshot, resolutions=(resolution,)
    )
    second = export_journey_to_fhir(
        reversed(facts), source_snapshot=snapshot, resolutions=(resolution,)
    )
    assert first.value is not None and second.value is not None

    assert first.value.to_json() == second.value.to_json()
    restored = type(first.value).from_dict(first.value.to_dict())
    assert restored.to_json() == first.value.to_json()
    schema = load_fhir_journey_schema()
    validator = validator_for(schema)
    validator.check_schema(schema)
    assert not tuple(validator(schema).iter_errors(first.value.to_dict()))


def test_unsupported_fact_and_external_resource_are_visible_losses() -> None:
    facts, _, snapshot = _records()
    unsupported = replace(
        facts[0],
        fact_id="fact_eeeeeeeeeeeeeeee",
        fact_type="synthetic_unsupported",
    )
    snapshot = replace(
        snapshot,
        source_fact_ids=(*snapshot.source_fact_ids, unsupported.fact_id),
        record_count=snapshot.record_count + 1,
    )

    exported = export_journey_to_fhir([facts[0], unsupported], source_snapshot=snapshot)

    assert exported.state is StoreState.PARTIAL
    assert exported.value is not None
    assert [item.reason_code for item in exported.value.losses] == [
        "fact_type_unsupported"
    ]
    strict = export_journey_to_fhir(
        [facts[0], unsupported], source_snapshot=snapshot, strict=True
    )
    assert strict.state is StoreState.UNSUPPORTED
    assert strict.value is not None

    bundle = copy.deepcopy(exported.value.bundle)
    bundle["entry"].append(
        {
            "fullUrl": "Device/synthetic-device",
            "resource": {"resourceType": "Device", "id": "synthetic-device"},
        }
    )
    imported = import_journey_from_fhir(bundle)
    assert imported.state is StoreState.PARTIAL
    assert imported.value is not None
    assert imported.value.losses[0].reason_code == "resource_type_unsupported"


def test_tampered_canonical_payload_and_snapshot_fail_closed() -> None:
    facts, resolution, snapshot = _records()
    exported = export_journey_to_fhir(
        facts, source_snapshot=snapshot, resolutions=(resolution,)
    )
    assert exported.value is not None
    bundle = copy.deepcopy(exported.value.bundle)
    fact_resource = next(
        entry["resource"]
        for entry in bundle["entry"]
        if entry["resource"].get("resourceType") == "Condition"
    )
    extension = next(
        item
        for item in fact_resource["extension"]
        if item["url"] == OPENMED_FACT_EXTENSION
    )
    canonical = next(
        item for item in extension["extension"] if item["url"] == "canonicalJson"
    )
    payload = json.loads(base64.b64decode(canonical["valueBase64Binary"]))
    payload["status"] = "inactive"
    canonical["valueBase64Binary"] = base64.b64encode(
        json.dumps(payload).encode("utf-8")
    ).decode("ascii")

    assert import_journey_from_fhir(bundle).state is StoreState.CONFLICT
    different_snapshot = replace(snapshot, manifest_hash="sha256:" + "9" * 64)
    assert (
        import_journey_from_fhir(
            exported.value.bundle, expected_snapshot=different_snapshot
        ).state
        is StoreState.CONFLICT
    )

    without_snapshot = copy.deepcopy(exported.value.bundle)
    without_snapshot.pop("extension")
    partial = import_journey_from_fhir(without_snapshot)
    assert partial.state is StoreState.PARTIAL
    assert partial.value is not None
    assert partial.value.losses[0].reason_code == "source_snapshot_missing"


def test_snapshot_custody_and_split_leakage_are_conflicts() -> None:
    facts, _, snapshot = _records()
    uncovered = replace(snapshot, source_fact_ids=(facts[0].fact_id,))
    assert (
        export_journey_to_fhir(facts, source_snapshot=uncovered).state
        is StoreState.CONFLICT
    )

    first = replace(facts[0], attributes={"dataset_split": "train"})
    second = replace(
        facts[1],
        attributes={"dataset_split": "holdout"},
        subject_id=first.subject_id,
    )
    split_snapshot = replace(
        snapshot,
        source_fact_ids=(first.fact_id, second.fact_id),
        split_hashes={
            "train": "sha256:" + "3" * 64,
            "holdout": "sha256:" + "4" * 64,
        },
    )
    assert (
        export_journey_to_fhir([first, second], source_snapshot=split_snapshot).state
        is StoreState.CONFLICT
    )


def test_import_rejects_snapshot_that_does_not_cover_bundle_facts() -> None:
    facts, _, snapshot = _records()
    complete = export_journey_to_fhir(facts, source_snapshot=snapshot)
    subset = export_journey_to_fhir(
        facts[:1],
        source_snapshot=replace(
            snapshot, source_fact_ids=(facts[0].fact_id,), record_count=1
        ),
    )
    assert complete.value is not None and subset.value is not None
    bundle = copy.deepcopy(complete.value.bundle)
    bundle["extension"] = copy.deepcopy(subset.value.bundle["extension"])
    result = import_journey_from_fhir(bundle)
    assert result.state is StoreState.CONFLICT
    assert result.code == "fhir_custody_conflict"


@given(
    fact_type=st.sampled_from(
        (
            "condition",
            "diagnosis",
            "problem",
            "observation",
            "laboratory",
            "measurement",
            "vital",
            "social_determinant",
            "medication",
            "drug",
            "procedure",
        )
    )
)
def test_supported_fact_type_property_preserves_status_and_evidence(
    fact_type: str,
) -> None:
    facts, _, snapshot = _records()
    fact = replace(facts[0], fact_type=fact_type)
    snapshot = replace(snapshot, source_fact_ids=(fact.fact_id,), record_count=1)

    exported = export_journey_to_fhir([fact], source_snapshot=snapshot)
    assert exported.state is StoreState.SUCCESS
    assert exported.value is not None
    imported = import_journey_from_fhir(exported.value.bundle)
    assert imported.state is StoreState.SUCCESS
    assert imported.value is not None
    assert imported.value.facts[0].status == fact.status
    assert imported.value.facts[0].evidence_ids == fact.evidence_ids
