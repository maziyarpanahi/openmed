"""Tests for relation-level assertion propagation."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.relations.assertion_filter import (
    ASSERTION_FILTER_ADVISORY,
    RELATION_CONDITIONAL,
    RELATION_CONFIRMED,
    RELATION_POSSIBLE,
    RELATION_REFUTED,
    RelationAssertion,
    filter_asserted_relations,
    propagate_relation_assertion,
)
from openmed.clinical.relations.candidate import RelationCandidate, SpanReference


def _relation(doc: str, drug: str, attribute: str) -> RelationCandidate:
    d = doc.index(drug)
    a = doc.index(attribute)
    return RelationCandidate(
        relation_type="drug_to_dose",
        head=SpanReference(
            text=drug, label="MEDICATION", start=d, end=d + len(drug), score=1.0
        ),
        attribute=SpanReference(
            text=attribute, label="dose", start=a, end=a + len(attribute), score=1.0
        ),
        score=1.0,
        confidence=1.0,
        features={},
        explanation=(),
    )


# --------------------------------------------------------------------------
# Propagation rules
# --------------------------------------------------------------------------


def test_negated_argument_makes_relation_refuted():
    doc = "Denies metformin 500 mg daily"
    rel = _relation(doc, "metformin", "500 mg")

    result = propagate_relation_assertion(rel, doc)

    assert isinstance(result, RelationAssertion)
    assert result.status == RELATION_REFUTED
    assert result.fhir_verification_status == "refuted"
    assert result.asserted is False


def test_hypothetical_argument_makes_relation_conditional():
    doc = "Start metformin 1000 mg if A1c rises"
    rel = _relation(doc, "metformin", "1000 mg")

    result = propagate_relation_assertion(rel, doc)

    assert result.status == RELATION_CONDITIONAL
    assert result.fhir_verification_status == "provisional"
    assert result.asserted is False


def test_uncertain_argument_makes_relation_possible():
    doc = "Possible metformin toxicity, 500 mg"
    rel = _relation(doc, "metformin", "500 mg")

    result = propagate_relation_assertion(rel, doc)

    assert result.status == RELATION_POSSIBLE
    assert result.fhir_verification_status == "provisional"
    assert result.asserted is True


def test_plain_argument_is_confirmed():
    doc = "metformin 500 mg daily"
    rel = _relation(doc, "metformin", "500 mg")

    result = propagate_relation_assertion(rel, doc)

    assert result.status == RELATION_CONFIRMED
    assert result.fhir_verification_status == "confirmed"
    assert result.asserted is True


def test_negation_takes_precedence_over_uncertainty():
    # "if" yields both hypothetical and uncertain; a negated argument still wins.
    doc = "No metformin 500 mg if intolerant"
    rel = _relation(doc, "metformin", "500 mg")

    assert propagate_relation_assertion(rel, doc).status == RELATION_REFUTED


# --------------------------------------------------------------------------
# Filtering: refuted / conditional excluded from the default fact set
# --------------------------------------------------------------------------


def test_filter_excludes_refuted_and_conditional_only():
    confirmed = _relation("metformin 500 mg daily", "metformin", "500 mg")
    refuted = _relation("denies lisinopril 10 mg", "lisinopril", "10 mg")
    conditional = _relation("start aspirin 81 mg if MI", "aspirin", "81 mg")

    asserted, withheld = filter_asserted_relations(
        [confirmed, refuted, conditional],
        {
            id(confirmed): "metformin 500 mg daily",
            id(refuted): "denies lisinopril 10 mg",
            id(conditional): "start aspirin 81 mg if MI",
        },
    )

    assert {r.status for r in asserted} == {RELATION_CONFIRMED}
    assert {r.status for r in withheld} == {RELATION_REFUTED, RELATION_CONDITIONAL}


# --------------------------------------------------------------------------
# Audit trace: offsets + tags + hash, no raw note text
# --------------------------------------------------------------------------


def test_audit_entry_has_offsets_tags_hash_and_no_note_text():
    doc = "Denies metformin 500 mg daily"
    rel = _relation(doc, "metformin", "500 mg")

    entry = propagate_relation_assertion(rel, doc).to_audit_entry()

    assert entry["head_offset"] == [rel.head.start, rel.head.end]
    assert entry["attribute_offset"] == [rel.attribute.start, rel.attribute.end]
    assert entry["status"] == RELATION_REFUTED
    assert "content_hash" in entry
    # The raw note text must never appear in the audit entry.
    assert doc not in json.dumps(entry)


def test_deterministic_hash():
    doc = "metformin 500 mg daily"
    rel = _relation(doc, "metformin", "500 mg")

    a = propagate_relation_assertion(rel, doc).to_audit_entry()["content_hash"]
    b = propagate_relation_assertion(rel, doc).to_audit_entry()["content_hash"]
    assert a == b


def test_advisory_exposed():
    assert isinstance(ASSERTION_FILTER_ADVISORY, str) and ASSERTION_FILTER_ADVISORY
