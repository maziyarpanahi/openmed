"""Synthetic privacy, review-binding and R4 reference controls for relation export."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from openmed.clinical.exporters.fhir import (
    RelationFHIRExportError,
    ReviewedFHIRRelation,
    export_reviewed_relations,
    find_reference_target_issues,
    relation_fhir_review_fingerprint,
    validate_resource,
)
from openmed.clinical.family_history import FamilyHistoryRecord
from openmed.clinical.relations.evidence_binding import (
    AssertionState,
    bind_relation_evidence,
)
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewStateMachine,
    ReviewTransitionPolicy,
)

SYSTEM = "https://openmed.ai/fhir/CodeSystem/synthetic-test"
PRIVATE = "SOURCE-SYNTHETIC-NAME"
PATIENT = "Patient/private-patient-id"
CONDITION = "Condition/private-condition-id"
MEDICATION = "MedicationStatement/private-medication-id"
PROCEDURE = "Procedure/private-procedure-id"
TIME = datetime(2026, 10, 9, 8, 0, tzinfo=timezone.utc)


def concept(code="synthetic-condition"):
    return {
        "coding": [{"system": SYSTEM, "code": code, "display": PRIVATE}],
        "text": PRIVATE,
    }


def resources():
    return [
        {"resourceType": "Patient", "id": PATIENT.split("/")[1], "name": PRIVATE},
        {
            "resourceType": "Condition",
            "id": CONDITION.split("/")[1],
            "subject": {"reference": PATIENT, "display": PRIVATE},
            "code": concept(),
            "note": [{"text": PRIVATE}],
        },
        {
            "resourceType": "MedicationStatement",
            "id": MEDICATION.split("/")[1],
            "subject": {"reference": PATIENT},
            "status": "active",
            "medicationCodeableConcept": concept("synthetic-medication"),
            "text": {"status": "generated", "div": PRIVATE},
            "effectiveDateTime": "2000-01-01",
            "reasonReference": [{"reference": "Condition/unreviewed"}],
        },
        {
            "resourceType": "Procedure",
            "id": PROCEDURE.split("/")[1],
            "subject": {"reference": PATIENT},
            "status": "completed",
            "code": concept("synthetic-procedure"),
        },
    ]


def candidate(kind="diagnosis_to_treatment", *, assertion="affirmed"):
    relation = bind_relation_evidence(
        {
            "relation_type": kind,
            "document_id": "synthetic-note",
            "head": {"start": 0, "end": 5, "label": "PROBLEM"},
            "tail": {"start": 10, "end": 15, "label": "TREATMENT"},
            "evidence_spans": [{"start": 6, "end": 9}],
            "assertion_state": assertion,
        }
    )
    return ReviewedFHIRRelation(relation, CONDITION, MEDICATION, "patient")


def reviewed(item, data, *, state=ReviewState.APPROVED, policy=None):
    fingerprint = relation_fhir_review_fingerprint(item, data, PATIENT)
    machine = ReviewStateMachine(policy=policy)
    machine.transition(ReviewState.IN_REVIEW, "evt_" + "a" * 16, fingerprint)
    machine.transition(state, "evt_" + "b" * 16, fingerprint)
    return replace(item, review_transitions=machine.transitions)


def project(items, data, **kwargs):
    return export_reviewed_relations(
        items, data, patient_reference=PATIENT, clock=lambda: TIME, **kwargs
    )


def bundle_resources(report):
    return [entry["resource"] for entry in report.bundle["entry"]]


def family_candidate(*, relative=True):
    item = candidate("condition_to_relative")
    return replace(
        item,
        tail_reference="RelatedPerson/private-relative-id" if relative else None,
        experiencer="family",
        family_record=FamilyHistoryRecord(
            "mother", item.relation.head.offset, relative_span=item.relation.tail.offset
        ),
    )


def family_resources():
    return [
        *resources(),
        {
            "resourceType": "RelatedPerson",
            "id": "private-relative-id",
            "patient": {"reference": PATIENT},
            "name": PRIVATE,
        },
    ]


def test_approved_reason_link_is_closed_source_free_and_input_is_unchanged():
    data = resources()
    before = copy.deepcopy(data)
    item = reviewed(candidate(), data)
    report = project([item], data)
    assert report.accepted_count == 1 and report.losses == ()
    assert report.omitted_endpoint_field_count == 5
    assert data == before
    assert report.bundle["type"] == "collection"
    output = bundle_resources(report)
    assert {r["resourceType"] for r in output} == {
        "Patient",
        "Condition",
        "MedicationStatement",
        "Device",
        "Provenance",
    }
    medication = next(r for r in output if r["resourceType"] == "MedicationStatement")
    target = medication["reasonReference"][0]["reference"]
    urls = {entry["fullUrl"]: entry["resource"] for entry in report.bundle["entry"]}
    assert urls[target]["resourceType"] == "Condition"
    assert find_reference_target_issues(report.bundle) == ()
    assert not validate_resource(medication).errors
    payload = json.dumps(report.to_dict())
    for source in (
        PRIVATE,
        "private-patient-id",
        "private-condition-id",
        "private-medication-id",
        "2000-01-01",
        "unreviewed",
    ):
        assert source not in payload
    assert all("request" not in entry for entry in report.bundle["entry"])
    assert report.to_dict()["write_authority"] is False
    provenance = next(r for r in output if r["resourceType"] == "Provenance")
    offsets = provenance["entity"][0]["extension"]
    assert [
        (
            e["extension"][0]["valueCode"],
            e["extension"][1]["valueUnsignedInt"],
            e["extension"][2]["valueUnsignedInt"],
        )
        for e in offsets
    ] == [("head", 0, 5), ("tail", 10, 15), ("link", 6, 9)]
    assert provenance["recorded"] == "2026-10-09T08:00:00Z"


@pytest.mark.parametrize(
    "kind,head,tail,source",
    [
        ("diagnosis_to_treatment", CONDITION, MEDICATION, "MedicationStatement"),
        ("diagnosis_to_treatment", CONDITION, PROCEDURE, "Procedure"),
        ("procedure_to_indication", PROCEDURE, CONDITION, "Procedure"),
        ("drug_to_reason", MEDICATION, CONDITION, "MedicationStatement"),
        ("drug_to_indication", MEDICATION, CONDITION, "MedicationStatement"),
        ("medication_change", MEDICATION, CONDITION, "MedicationStatement"),
    ],
)
def test_supported_direction_uses_reason_reference(kind, head, tail, source):
    data = resources()
    item = replace(candidate(kind), head_reference=head, tail_reference=tail)
    report = project([reviewed(item, data)], data)
    assert report.accepted_count == 1
    linked = next(r for r in bundle_resources(report) if r["resourceType"] == source)
    assert len(linked["reasonReference"]) == 1
    assert find_reference_target_issues(report.bundle) == ()


@pytest.mark.parametrize(
    "kind", ["Condition", "Observation", "DiagnosticReport", "Procedure"]
)
@pytest.mark.parametrize("source", ["MedicationStatement", "Procedure"])
def test_release_reference_allowlist_is_enforced(kind, source):
    data = resources()
    target = f"{kind}/reason"
    endpoint = {
        "resourceType": kind,
        "id": "reason",
        "subject": {"reference": PATIENT},
        "code": concept(),
    }
    if kind != "Condition":
        endpoint["status"] = "completed" if kind == "Procedure" else "final"
    data.append(endpoint)
    item = replace(
        candidate(
            "drug_to_reason"
            if source == "MedicationStatement"
            else "procedure_to_indication"
        ),
        head_reference=MEDICATION if source == "MedicationStatement" else PROCEDURE,
        tail_reference=target,
    )
    report = project([reviewed(item, data)], data)
    if kind == "Procedure" and source == "MedicationStatement":
        assert report.accepted_count == 0
        assert report.losses[0].code == "reference_type_refused"
    else:
        assert report.accepted_count == 1
        assert find_reference_target_issues(report.bundle) == ()


@pytest.mark.parametrize("relative", [True, False])
def test_family_condition_is_inline_and_never_emitted_as_patient_condition(relative):
    data = family_resources()
    report = project([reviewed(family_candidate(relative=relative), data)], data)
    assert report.accepted_count == 1
    output = bundle_resources(report)
    assert not any(
        r["resourceType"] in ("Condition", "MedicationStatement", "Procedure")
        for r in output
    )
    family = next(r for r in output if r["resourceType"] == "FamilyMemberHistory")
    assert family["relationship"]["coding"][0]["code"] == "mother"
    assert family["condition"][0]["code"]["coding"][0]["code"] == "synthetic-condition"
    assert not validate_resource(family).findings
    assert find_reference_target_issues(report.bundle) == ()
    assert PRIVATE not in json.dumps(report.to_dict())


@pytest.mark.parametrize("key", ["status", "patient", "relationship"])
def test_family_required_fields_are_locally_validated(key):
    data = family_resources()
    report = project([reviewed(family_candidate(), data)], data)
    family = next(
        r
        for r in bundle_resources(report)
        if r["resourceType"] == "FamilyMemberHistory"
    )
    del family[key]
    assert any(f.code == "required" for f in validate_resource(family).errors)


def test_family_nested_condition_code_and_status_binding_are_validated():
    data = family_resources()
    report = project([reviewed(family_candidate(), data)], data)
    family = next(
        r
        for r in bundle_resources(report)
        if r["resourceType"] == "FamilyMemberHistory"
    )
    family["condition"] = [{}]
    assert validate_resource(family).errors
    family["condition"] = []
    family["status"] = PRIVATE
    result = validate_resource(family)
    assert result.errors and PRIVATE not in repr(result)


@pytest.mark.parametrize(
    "assertion",
    [
        s.value
        for s in AssertionState
        if s
        not in (
            AssertionState.AFFIRMED,
            AssertionState.CONFIRMED,
            AssertionState.HISTORICAL,
        )
    ],
)
def test_non_affirmative_relations_never_create_links(assertion):
    data = resources()
    report = project([reviewed(candidate(assertion=assertion), data)], data)
    assert report.bundle["entry"] == []
    assert report.losses[0].code == "assertion_refused"


@pytest.mark.parametrize("assertion", ["affirmed", "confirmed", "historical"])
def test_positive_reviewed_assertion_can_be_projected(assertion):
    data = resources()
    assert (
        project([reviewed(candidate(assertion=assertion), data)], data).accepted_count
        == 1
    )


@pytest.mark.parametrize(
    "state,code",
    [(ReviewState.REJECTED, "rejected"), (ReviewState.EXPIRED, "review_not_approved")],
)
def test_review_refusals_are_explicit(state, code):
    data = resources()
    report = project([reviewed(candidate(), data, state=state)], data)
    assert report.bundle["entry"] == [] and report.losses[0].code == code


def test_unreviewed_and_fabricated_approved_history_fail_closed():
    data = resources()
    item = candidate()
    assert project([item], data).losses[0].code == "unreviewed"
    approved = reviewed(item, data)
    assert (
        project(
            [replace(approved, review_transitions=approved.review_transitions[-1:])],
            data,
        )
        .losses[0]
        .code
        == "review_invalid"
    )


def test_reopened_history_needs_new_review():
    data = resources()
    item = candidate()
    fp = relation_fhir_review_fingerprint(item, data, PATIENT)
    machine = ReviewStateMachine()
    for i, state in enumerate(
        (ReviewState.IN_REVIEW, ReviewState.APPROVED, ReviewState.REOPENED)
    ):
        machine.transition(state, "evt_" + f"{i + 1:016x}", fp)
    item = replace(item, review_transitions=machine.transitions)
    assert project([item], data).losses[0].code == "review_not_approved"


@pytest.mark.parametrize(
    "change",
    [
        "code",
        "private_field",
        "subject",
        "attribution",
        "relation",
        "offset",
        "terminology",
    ],
)
def test_review_is_invalidated_by_changed_projection_inputs(change):
    data = resources()
    item = reviewed(candidate(), data)
    kwargs = {}
    if change == "code":
        data[1]["code"] = concept("changed")
    elif change == "private_field":
        data[0]["name"] = "changed-source"
    elif change == "subject":
        data[1]["subject"] = {"reference": "Patient/other"}
    elif change == "attribution":
        item = replace(item, experiencer="family")
    elif change == "relation":
        item = replace(
            item, relation=replace(item.relation, relation_type="medication_change")
        )
    elif change == "offset":
        item = replace(
            item,
            relation=replace(item.relation, head=replace(item.relation.head, end=6)),
        )
    else:
        kwargs["code_systems"] = (SYSTEM,)
    report = project([item], data, **kwargs)
    assert report.accepted_count == 0 and report.losses[0].code == "review_mismatch"


@pytest.mark.parametrize("experiencer", ["family", "other"])
def test_non_patient_reason_relations_are_refused(experiencer):
    data = resources()
    item = replace(candidate(), experiencer=experiencer)
    assert project([reviewed(item, data)], data).losses[0].code == "non_patient"


@pytest.mark.parametrize(
    "change",
    [
        "condition_offset",
        "relative_offset",
        "absent_record",
        "wrong_head",
        "wrong_tail",
        "patient_attribution",
    ],
)
def test_family_attribution_and_offset_bindings_cannot_be_guessed(change):
    data = family_resources()
    item = family_candidate()
    if change == "condition_offset":
        item = replace(
            item, family_record=replace(item.family_record, condition_span=(1, 5))
        )
    elif change == "relative_offset":
        item = replace(
            item, family_record=replace(item.family_record, relative_span=(11, 15))
        )
    elif change == "absent_record":
        item = replace(item, family_record=None)
    elif change == "wrong_head":
        item = replace(item, head_reference=MEDICATION)
    elif change == "wrong_tail":
        item = replace(item, tail_reference=MEDICATION)
    else:
        item = replace(item, experiencer="patient")
    report = project([reviewed(item, data)], data)
    assert report.accepted_count == 0
    assert report.losses[0].code == (
        "non_patient" if change == "patient_attribution" else "family_binding_invalid"
    )


def test_same_condition_cannot_be_attributed_to_both_patient_and_relative():
    data = family_resources()
    items = [reviewed(family_candidate(), data), reviewed(candidate(), data)]
    report = project(items, data)
    assert report.accepted_count == 0
    assert [loss.code for loss in report.losses] == ["attribution_conflict"] * 2


@pytest.mark.parametrize(
    "change",
    [
        "subject",
        "status",
        "modifier",
        "unknown_system",
        "free_text_code",
        "missing_code",
        "wrong_reason_type",
    ],
)
def test_endpoint_contracts_refuse_unsafe_or_unrepresentable_content(change):
    data = resources()
    item = candidate()
    if change == "subject":
        data[1]["subject"]["reference"] = "Patient/other"
    elif change == "status":
        data[2]["status"] = "entered-in-error"
    elif change == "modifier":
        data[1]["modifierExtension"] = [{"url": SYSTEM, "valueBoolean": True}]
    elif change == "unknown_system":
        data[1]["code"]["coding"][0]["system"] = "https://untrusted.example/terms"
    elif change == "free_text_code":
        data[1]["code"]["coding"][0]["code"] = "private source text"
    elif change == "missing_code":
        del data[1]["code"]
    else:
        item = replace(item, head_reference=MEDICATION, tail_reference=PROCEDURE)
    report = project([reviewed(item, data)], data)
    assert report.accepted_count == 0
    assert report.losses[0].code in ("endpoint_invalid", "reference_type_refused")
    assert PRIVATE not in json.dumps(report.to_dict())


def test_unrelated_unsupported_or_invalid_resources_do_not_poison_accepted_link():
    data = resources()
    data.extend(
        [
            {
                "resourceType": "DocumentReference",
                "id": "unrelated",
                "description": PRIVATE,
            },
            {
                "resourceType": "Procedure",
                "id": "unrelated",
                "reasonReference": [{"reference": MEDICATION}],
            },
        ]
    )
    report = project([reviewed(candidate(), data)], data)
    assert report.accepted_count == 1
    assert not any(
        r["resourceType"] == "DocumentReference" for r in bundle_resources(report)
    )


def test_missing_and_unmapped_endpoints_are_losses_without_partial_resources():
    data = resources()
    missing = replace(candidate(), head_reference="Condition/missing")
    unknown = replace(
        candidate(),
        relation=replace(candidate().relation, relation_type="laboratory_result"),
    )
    document = {
        "resourceType": "DocumentReference",
        "id": "document",
        "description": PRIVATE,
    }
    data.append(document)
    unsupported = replace(candidate(), head_reference="DocumentReference/document")
    report = project([reviewed(i, data) for i in (missing, unknown, unsupported)], data)
    assert report.bundle["entry"] == []
    assert [loss.code for loss in report.losses] == [
        "endpoint_missing",
        "unsupported_relation",
        "endpoint_invalid",
    ]


def test_duplicate_candidate_does_not_duplicate_link_or_provenance():
    data = resources()
    item = reviewed(candidate(), data)
    report = project([item, item], data)
    assert report.accepted_count == 1 and report.losses[0].code == "duplicate_relation"
    assert sum(r["resourceType"] == "Provenance" for r in bundle_resources(report)) == 1


def test_custom_policy_cannot_skip_review_before_export():
    data = resources()
    item = candidate()
    policy = ReviewTransitionPolicy(
        policy_id="synthetic-shortcut",
        allowed_transitions={ReviewState.QUEUED: (ReviewState.APPROVED,)},
    )
    fingerprint = relation_fhir_review_fingerprint(item, data, PATIENT)
    machine = ReviewStateMachine(policy=policy)
    machine.transition(ReviewState.APPROVED, "evt_" + "a" * 16, fingerprint)
    approved = replace(item, review_transitions=machine.transitions)
    report = project([approved], data, review_policy=policy)
    assert report.accepted_count == 0 and report.losses[0].code == "review_invalid"


def test_each_review_transition_must_bind_to_the_same_input():
    data = resources()
    item = reviewed(candidate(), data)
    first = replace(
        item.review_transitions[0], provenance_fingerprint="sha256:" + "a" * 64
    )
    item = replace(item, review_transitions=(first, item.review_transitions[1]))
    assert project([item], data).losses[0].code == "review_mismatch"


@pytest.mark.parametrize(
    "kind,wrong_source",
    [
        ("procedure_to_indication", MEDICATION),
        ("drug_to_reason", PROCEDURE),
        ("drug_to_indication", PROCEDURE),
        ("medication_change", PROCEDURE),
    ],
)
def test_relation_direction_requires_the_semantically_correct_source_type(
    kind, wrong_source
):
    data = resources()
    item = replace(
        candidate(kind), head_reference=wrong_source, tail_reference=CONDITION
    )
    report = project([reviewed(item, data)], data)
    assert report.accepted_count == 0
    assert report.losses[0].code == "reference_type_refused"


def test_new_approval_after_reopening_is_accepted():
    data = resources()
    item = candidate()
    fp = relation_fhir_review_fingerprint(item, data, PATIENT)
    machine = ReviewStateMachine()
    states = (
        ReviewState.IN_REVIEW,
        ReviewState.APPROVED,
        ReviewState.REOPENED,
        ReviewState.IN_REVIEW,
        ReviewState.APPROVED,
    )
    for i, state in enumerate(states):
        machine.transition(state, "evt_" + f"{i + 1:016x}", fp)
    assert (
        project(
            [replace(item, review_transitions=machine.transitions)], data
        ).accepted_count
        == 1
    )


def test_multiple_approved_reasons_accumulate_without_copying_old_links():
    data = resources()
    second_condition = copy.deepcopy(data[1])
    second_condition["id"] = "second"
    data.append(second_condition)
    first = candidate()
    second = replace(first, head_reference="Condition/second")
    report = project([reviewed(i, data) for i in (first, second)], data)
    assert report.accepted_count == 2
    medication = next(
        r
        for r in bundle_resources(report)
        if r["resourceType"] == "MedicationStatement"
    )
    assert len(medication["reasonReference"]) == 2
    assert find_reference_target_issues(report.bundle) == ()
    assert "unreviewed" not in json.dumps(report.to_dict())


@pytest.mark.parametrize("verification", ["refuted", "entered-in-error"])
def test_refuted_condition_cannot_be_a_reason_or_family_template(verification):
    data = family_resources()
    data[1]["verificationStatus"] = {
        "coding": [
            {
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": verification,
            }
        ]
    }
    for item in (candidate(), family_candidate()):
        report = project([reviewed(item, data)], data)
        assert (
            report.accepted_count == 0 and report.losses[0].code == "endpoint_invalid"
        )


@pytest.mark.parametrize("condition", [None, {}])
def test_explicit_empty_family_backbone_entry_is_invalid(condition):
    family = {
        "resourceType": "FamilyMemberHistory",
        "status": "partial",
        "patient": {"reference": PATIENT},
        "relationship": concept("mother"),
        "condition": [condition],
    }
    assert validate_resource(family).errors


def test_safe_boundary_discards_private_clock_exception():
    data = resources()

    def fail():
        raise RuntimeError(PRIVATE)

    with pytest.raises(RelationFHIRExportError) as caught:
        export_reviewed_relations(
            [reviewed(candidate(), data)], data, patient_reference=PATIENT, clock=fail
        )
    assert str(caught.value) == "Invalid reviewed FHIR relation input."
    assert caught.value.__suppress_context__


def test_refused_only_run_never_uses_technical_clock():
    def fail():
        raise AssertionError(PRIVATE)

    report = export_reviewed_relations(
        [candidate()], resources(), patient_reference=PATIENT, clock=fail
    )
    assert report.accepted_count == 0


@pytest.mark.parametrize("instant", [datetime(2026, 1, 1), PRIVATE, None])
def test_invalid_technical_clock_fails_with_fixed_diagnostic(instant):
    data = resources()
    with pytest.raises(
        RelationFHIRExportError, match=r"^Invalid reviewed FHIR relation input\.$"
    ) as caught:
        export_reviewed_relations(
            [reviewed(candidate(), data)],
            data,
            patient_reference=PATIENT,
            clock=lambda: instant,
        )
    assert caught.value.__suppress_context__


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "oversized",
        "nonfinite",
        "depth",
        "missing_patient",
        "bad_reference",
        "too_many_relations",
    ],
)
def test_bounded_input_contract_has_source_free_errors(change):
    data = resources()
    items = [candidate()]
    patient = PATIENT
    if change == "duplicate":
        data.append(copy.deepcopy(data[0]))
    elif change == "oversized":
        data[0]["name"] = "x" * 4_194_304
    elif change == "nonfinite":
        data[0]["name"] = float("nan")
    elif change == "depth":
        value = PRIVATE
        for _ in range(35):
            value = {"nested": value}
        data[0]["name"] = value
    elif change == "missing_patient":
        del data[0]
    elif change == "bad_reference":
        patient = "https://private.example/Patient/private-id"
    else:
        items *= 513
    with pytest.raises(RelationFHIRExportError) as caught:
        export_reviewed_relations(items, data, patient_reference=patient)
    assert str(caught.value) == "Invalid reviewed FHIR relation input."
    assert PRIVATE not in str(caught.value)
