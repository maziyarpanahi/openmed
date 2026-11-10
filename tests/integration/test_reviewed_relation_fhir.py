"""Offline synthetic generation, evidence, review and FHIR projection journeys."""

import json
import socket
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from openmed.clinical.exporters.fhir import (
    ReviewedFHIRRelation,
    export_reviewed_relations,
    find_reference_target_issues,
    relation_fhir_review_fingerprint,
)
from openmed.clinical.family_history import FamilyHistoryRecord
from openmed.clinical.relations.diagnosis_treatments import (
    generate_diagnosis_treatment_candidates,
)
from openmed.clinical.relations.evidence_binding import bind_relation_evidence
from openmed.clinical.relations.family_history import extract_family_history_relations
from openmed.clinical.review_state_machine import ReviewState, ReviewStateMachine

pytestmark = pytest.mark.integration
SYSTEM = "https://openmed.ai/fhir/CodeSystem/synthetic-test"


@pytest.mark.parametrize("family", [False, True])
def test_generated_candidate_requires_review_before_offline_projection(
    monkeypatch, family
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network unavailable in offline relation projection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    note = "mother had pneumonia" if family else "Pneumonia treated with ceftriaxone."
    condition = "pneumonia" if family else "Pneumonia"
    start = note.index(condition)
    spans = [{"start": start, "end": start + len(condition), "label": "CONDITION"}]
    if family:
        (generated,) = extract_family_history_relations(note, spans)
        guarded = bind_relation_evidence(
            generated,
            document_id="synthetic-family-note",
            assertion_state="affirmed",
            evidence_spans=[(7, 10)],
            document_text=note,
        )
        item = ReviewedFHIRRelation(
            guarded,
            "Condition/c",
            None,
            "family",
            family_record=FamilyHistoryRecord(
                "mother", guarded.head.offset, relative_span=guarded.tail.offset
            ),
        )
    else:
        offset = note.index("ceftriaxone")
        spans.append({"start": offset, "end": offset + 11, "label": "MEDICATION"})
        (generated,) = generate_diagnosis_treatment_candidates(note, spans)
        guarded = bind_relation_evidence(
            generated.to_dict(),
            document_id="synthetic-treatment-note",
            head=generated.diagnosis.to_dict(),
            tail=generated.treatment.to_dict(),
            evidence_spans=[generated.linking_cue.to_dict()],
            assertion_state="affirmed",
            document_text=note,
        )
        item = ReviewedFHIRRelation(
            guarded, "Condition/c", "MedicationStatement/m", "patient"
        )
    data = [
        {"resourceType": "Patient", "id": "p"},
        {
            "resourceType": "Condition",
            "id": "c",
            "subject": {"reference": "Patient/p"},
            "code": {"coding": [{"system": SYSTEM, "code": "synthetic-condition"}]},
        },
        {
            "resourceType": "MedicationStatement",
            "id": "m",
            "subject": {"reference": "Patient/p"},
            "status": "active",
            "medicationCodeableConcept": {
                "coding": [{"system": SYSTEM, "code": "synthetic-medication"}]
            },
        },
    ]
    assert (
        export_reviewed_relations(
            [item], data, patient_reference="Patient/p"
        ).accepted_count
        == 0
    )
    fingerprint = relation_fhir_review_fingerprint(item, data, "Patient/p")
    machine = ReviewStateMachine()
    # Explicit synthetic decisions stand in for the caller's existing review UI.
    machine.transition(ReviewState.IN_REVIEW, "evt_" + "1" * 16, fingerprint)
    machine.transition(ReviewState.APPROVED, "evt_" + "2" * 16, fingerprint)
    approved = replace(item, review_transitions=machine.transitions)
    report = export_reviewed_relations(
        [approved],
        data,
        patient_reference="Patient/p",
        clock=lambda: datetime(2026, 10, 9, tzinfo=timezone.utc),
    )
    assert report.accepted_count == 1
    assert find_reference_target_issues(report.bundle) == ()
    payload = json.dumps(report.to_dict())
    assert (
        note not in payload
        and condition not in payload
        and "ceftriaxone" not in payload
    )
    entries = report.bundle["entry"]
    urls = {entry["fullUrl"] for entry in entries}

    def references(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "reference":
                    yield child
                else:
                    yield from references(child)
        elif isinstance(value, list):
            for child in value:
                yield from references(child)

    assert set(references(report.bundle)) <= urls
    types = {entry["resource"]["resourceType"] for entry in entries}
    assert ("FamilyMemberHistory" in types) is family
    assert ("Condition" in types) is not family
