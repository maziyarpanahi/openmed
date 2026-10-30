"""Synthetic packet consistency, evidence gaps and distractor controls."""

from dataclasses import replace

import pytest

from openmed.training.synthetic.prior_auth import (
    AuthorizationCaseSpec,
    AuthorizationCriterion,
    AuthorizationEvidence,
    generate_authorization_packet,
    validate_authorization_packet,
)


def spec(evidence=(), **kwargs):
    return AuthorizationCaseSpec(
        "synthetic_diagnosis",
        "synthetic_treatment",
        (AuthorizationCriterion("prior_trial", ("note",)),),
        evidence,
        **kwargs,
    )


def test_reproducible_three_document_packet_and_exact_evidence_truth():
    packet = generate_authorization_packet(
        spec((AuthorizationEvidence("note", "prior_trial", True),)), seed=42
    )
    assert packet == generate_authorization_packet(
        spec((AuthorizationEvidence("note", "prior_trial", True),)), seed=42
    )
    assert {d.kind for d in packet.documents} == {"note", "order", "form"}
    assert validate_authorization_packet(packet) == ()
    row = packet.documents[1].evidence[0]
    assert packet.documents[1].text[row["start"] : row["end"]] == "prior_trial: present"
    assert packet.review_state == "ready_for_review"


@pytest.mark.parametrize(
    "evidence,status,state",
    [
        ((), "missing", "abstain"),
        (
            (AuthorizationEvidence("note", "prior_trial", False),),
            "not_satisfied",
            "requirements_not_satisfied",
        ),
        (
            (
                AuthorizationEvidence("note", "prior_trial", True),
                AuthorizationEvidence("note", "prior_trial", False),
            ),
            "conflicting",
            "abstain",
        ),
        ((AuthorizationEvidence("form", "prior_trial", True),), "missing", "abstain"),
        (
            (AuthorizationEvidence("note", "prior_trial", True, True),),
            "missing",
            "abstain",
        ),
    ],
)
def test_missing_conflicting_disallowed_and_distractor_evidence_never_satisfies(
    evidence, status, state
):
    packet = generate_authorization_packet(spec(evidence))
    assert packet.criteria[0]["status"] == status
    assert packet.review_state == state
    assert validate_authorization_packet(packet) == ()


def test_independent_validator_catches_identity_placeholder_span_and_state_changes():
    packet = generate_authorization_packet(
        spec((AuthorizationEvidence("note", "prior_trial", True),))
    )
    doc = packet.documents[0]
    assert "case_identity_mismatch" in validate_authorization_packet(
        replace(
            packet, documents=(replace(doc, case_id="changed"),) + packet.documents[1:]
        )
    )
    assert "clinical_placeholder_mismatch" in validate_authorization_packet(
        replace(
            packet,
            documents=(replace(doc, diagnosis_code="changed"),) + packet.documents[1:],
        )
    )
    note = packet.documents[1]
    rows = tuple({**r, "start": r["start"] + 1} for r in note.evidence)
    assert "evidence_span_mismatch" in validate_authorization_packet(
        replace(
            packet, documents=(doc, replace(note, evidence=rows), packet.documents[2])
        )
    )
    assert "review_state_mismatch" in validate_authorization_packet(
        replace(packet, review_state="approved")
    )


def test_unknown_or_duplicate_criterion_inputs_reject():
    with pytest.raises(ValueError, match="undeclared"):
        spec((AuthorizationEvidence("note", "other", True),))
    criterion = AuthorizationCriterion("prior_trial", ("note",))
    with pytest.raises(ValueError, match="duplicate"):
        AuthorizationCaseSpec(
            "synthetic_diagnosis", "synthetic_treatment", (criterion, criterion), ()
        )
    with pytest.raises(ValueError):
        AuthorizationEvidence("note", "private raw text", True)


def test_altered_criterion_status_cannot_create_apparently_complete_packet():
    packet = generate_authorization_packet(spec())
    criteria = tuple({**row, "status": "satisfied"} for row in packet.criteria)
    changed = replace(packet, criteria=criteria, review_state="ready_for_review")
    assert "criterion_truth_mismatch" in validate_authorization_packet(changed)


def test_changed_document_header_is_detected_even_without_criterion_spans():
    packet = generate_authorization_packet(spec())
    doc = packet.documents[0]
    bad = replace(
        doc, text=doc.text.replace("synthetic_diagnosis", "changed_diagnosis")
    )
    assert "document_header_mismatch" in validate_authorization_packet(
        replace(packet, documents=(bad,) + packet.documents[1:])
    )


def test_validator_rejects_missing_or_malformed_gold_criteria():
    packet = generate_authorization_packet(spec())
    assert validate_authorization_packet(
        replace(packet, criteria=(), review_state="ready_for_review")
    ) == ("invalid_criteria",)
    for criteria in [
        ({},),
        (dict(packet.criteria[0], code=[]),),
        (dict(packet.criteria[0], evidence_references=[[]]),),
        packet.criteria * 2,
    ]:
        assert validate_authorization_packet(replace(packet, criteria=criteria)) == (
            "invalid_criteria",
        )


def test_validator_returns_controlled_categories_for_malformed_documents_and_rows():
    packet = generate_authorization_packet(
        spec((AuthorizationEvidence("note", "prior_trial", True),))
    )
    assert validate_authorization_packet(
        replace(packet, documents=(None, *packet.documents[1:]))
    ) == ("invalid_document",)
    document = packet.documents[1]
    for row in [
        {},
        {"reference": []},
        dict(document.evidence[0], present="yes"),
        dict(document.evidence[0], code=[]),
    ]:
        altered = replace(document, evidence=(row,))
        assert validate_authorization_packet(
            replace(
                packet, documents=(packet.documents[0], altered, packet.documents[2])
            )
        ) == ("invalid_evidence",)
