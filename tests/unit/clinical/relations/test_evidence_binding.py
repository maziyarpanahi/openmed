"""Offline acceptance tests for mandatory guarded-relation evidence binding."""

from __future__ import annotations

import copy
import json
import socket

import pytest

from openmed.clinical.relations.assertion_filter import propagate_relation_assertion
from openmed.clinical.relations.candidate import (
    Relation,
    RelationCandidate,
    SpanReference,
)
from openmed.clinical.relations.evidence_binding import (
    AssertionState,
    EvidenceBindingError,
    EvidenceSpan,
    GuardedRelation,
    bind_relation_evidence,
    bind_relation_records,
    require_guarded_relations,
)

DOCUMENT_ID = "synthetic-relation-document"
DOCUMENT_TEXT = "Synthetic medication change was held after review."
RAW_MARKER = "synthetic-sensitive-value"


def _candidate() -> dict[str, object]:
    return {
        "relation_type": "medication_change",
        "document_id": DOCUMENT_ID,
        "head": {
            "document_id": DOCUMENT_ID,
            "start": 0,
            "end": 8,
            "label": "MEDICATION",
            "text": RAW_MARKER,
        },
        "tail": {
            "document_id": DOCUMENT_ID,
            "start": 9,
            "end": 17,
            "label": "PROBLEM",
            "text": RAW_MARKER,
        },
        "evidence_spans": [
            {
                "document_id": DOCUMENT_ID,
                "start": 18,
                "end": 25,
                "label": "CONTEXT_CUE",
                "text": RAW_MARKER,
            }
        ],
        "assertion_state": "affirmed",
        "score": 0.91,
        "metadata": {"raw_text": RAW_MARKER},
    }


def test_binding_requires_all_review_provenance_and_is_value_free() -> None:
    bound = bind_relation_evidence(
        _candidate(),
        document_text=DOCUMENT_TEXT,
    )

    assert isinstance(bound, GuardedRelation)
    assert bound.assertion_state is AssertionState.AFFIRMED
    assert bound.requires_clinician_review is True
    assert bound.autonomous_decision is False
    assert bound.document_id.startswith("sha256:")
    assert all(
        span.document_id == bound.document_id
        for span in (bound.head, bound.tail, *bound.evidence_spans)
    )
    rendered = bound.to_json()
    assert RAW_MARKER not in rendered
    assert RAW_MARKER not in repr(bound)
    assert "text" not in bound.to_dict()["head"]


def test_identifier_hash_secret_is_applied_before_serialization() -> None:
    candidate = _candidate()
    candidate["head"] = {
        **candidate["head"],  # type: ignore[arg-type]
        "span_id": RAW_MARKER,
    }

    bound = bind_relation_evidence(candidate, hash_secret="synthetic-secret")

    assert bound.document_id.startswith("hmac-sha256:")
    assert bound.head.span_id is not None
    assert bound.head.span_id.startswith("hmac-sha256:")
    assert RAW_MARKER not in bound.to_json()


def test_binding_accepts_existing_relation_api_with_explicit_assertion_and_evidence() -> (
    None
):
    head = SpanReference(
        text="Synthetic medication",
        label="MEDICATION",
        start=0,
        end=8,
        score=0.9,
    )
    tail = SpanReference(
        text="Synthetic problem",
        label="PROBLEM",
        start=9,
        end=17,
        score=0.8,
    )
    relation = Relation(head=head, type="dose", tail=tail, score=0.84)

    bound = bind_relation_evidence(
        relation,
        document_id=DOCUMENT_ID,
        evidence_spans=((18, 25),),
        assertion_state={"negation": "affirmed", "certainty": "certain"},
        document_length=len(DOCUMENT_TEXT),
    )

    assert bound.relation_type == "drug_to_dose"
    assert bound.head.offset == (0, 8)
    assert bound.tail.offset == (9, 17)
    assert bound.evidence_spans[0].offset == (18, 25)
    assert bound.confidence == 0.84


def test_binding_accepts_relation_candidates_and_assertion_wrappers() -> None:
    head = SpanReference(
        text="Synthetic medication",
        label="MEDICATION",
        start=0,
        end=8,
        score=0.9,
    )
    tail = SpanReference(
        text="Synthetic dose",
        label="DOSAGE",
        start=9,
        end=17,
        score=0.8,
    )
    candidate = RelationCandidate(
        relation_type="drug_to_dose",
        head=head,
        attribute=tail,
        score=0.84,
        confidence=0.84,
        features={},
        explanation=(),
    )
    asserted = propagate_relation_assertion(candidate, DOCUMENT_TEXT)

    bound = bind_relation_evidence(
        asserted,
        document_id=DOCUMENT_ID,
        evidence_spans=({"offset": {"start": 18, "end": 25}},),
    )

    assert bound.relation_type == "drug_to_dose"
    assert bound.assertion_state is AssertionState.CONFIRMED
    assert bound.evidence_spans[0].offset == (18, 25)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("head", "relation endpoint spans are required"),
        ("tail", "relation endpoint spans are required"),
        ("evidence_spans", "relation evidence spans are required"),
        ("assertion_state", "assertion_state is required"),
        ("document_id", "document_id is required"),
    ),
)
def test_incomplete_records_fail_closed_without_echoing_values(
    field: str,
    message: str,
) -> None:
    candidate = _candidate()
    if field == "document_id":
        candidate.pop("document_id")
        candidate["head"] = {"start": 0, "end": 8, "label": "MEDICATION"}
        candidate["tail"] = {"start": 9, "end": 17, "label": "PROBLEM"}
        candidate["evidence_spans"] = [{"start": 18, "end": 25}]
    else:
        candidate.pop(field)

    with pytest.raises(EvidenceBindingError, match=message) as exc_info:
        bind_relation_evidence(candidate)

    assert RAW_MARKER not in str(exc_info.value)


def test_conflicting_span_documents_are_rejected_before_workflow_entry() -> None:
    candidate = _candidate()
    candidate["tail"] = {
        **candidate["tail"],  # type: ignore[arg-type]
        "document_id": "another-synthetic-document",
    }

    with pytest.raises(EvidenceBindingError, match="does not match"):
        bind_relation_evidence(candidate)

    bound = bind_relation_evidence(_candidate())
    with pytest.raises(EvidenceBindingError, match="require bound relation records"):
        require_guarded_relations((_candidate(),))  # type: ignore[arg-type]
    assert require_guarded_relations((bound,), workflow="summary") == (bound,)


def test_batch_binding_is_deterministic_and_source_bounds_are_checked() -> None:
    first = _candidate()
    second = _candidate()
    second["relation_type"] = "diagnosis_treatment"
    second["head"] = {**second["head"], "start": 26, "end": 34}  # type: ignore[arg-type]
    second["tail"] = {**second["tail"], "start": 35, "end": 43}  # type: ignore[arg-type]
    second["evidence_spans"] = [{"document_id": DOCUMENT_ID, "start": 44, "end": 49}]

    forward = bind_relation_records(
        (first, second),
        document_text=DOCUMENT_TEXT,
    )
    reverse = bind_relation_records(
        (second, first),
        document_text=DOCUMENT_TEXT,
    )

    assert [record.to_json() for record in forward] == [
        record.to_json() for record in reverse
    ]
    with pytest.raises(EvidenceBindingError, match="exceed"):
        bind_relation_evidence(
            _candidate(),
            document_length=10,
        )


def test_serialization_is_deterministic_and_does_not_use_network(monkeypatch) -> None:
    bound = bind_relation_evidence(_candidate())

    def fail_network(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    assert bound.to_json() == bound.to_json()
    assert json.loads(bound.to_json())["assertion_state"] == "affirmed"


def test_input_order_does_not_change_evidence_order() -> None:
    candidate = _candidate()
    candidate["evidence_spans"] = [
        {"document_id": DOCUMENT_ID, "start": 30, "end": 35},
        {"document_id": DOCUMENT_ID, "start": 18, "end": 25},
    ]
    reversed_candidate = copy.deepcopy(candidate)
    reversed_candidate["evidence_spans"] = list(
        reversed(candidate["evidence_spans"])  # type: ignore[arg-type]
    )

    forward = bind_relation_evidence(candidate)
    reverse = bind_relation_evidence(reversed_candidate)
    assert forward == reverse
    assert [span.offset for span in forward.evidence_spans] == [(18, 25), (30, 35)]
