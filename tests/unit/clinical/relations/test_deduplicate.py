"""Synthetic offline tests for provenance-preserving relation collapse."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.relations import (
    Relation,
    RelationDeduplicationError,
    SpanReference,
    collapse_duplicate_relations,
)


def test_repeated_mentions_count_once_for_confidence_but_keep_locations() -> None:
    candidates = [
        {
            "relation_type": " Treats ",
            "normalized_head": {"system": "SYNTHETIC", "code": "H-001"},
            "normalized_tail": {"system": "SYNTHETIC", "code": "T-001"},
            "source_id": "synthetic-note-a",
            "score": 0.8,
            "evidence": [
                {"start": 10, "end": 20},
                {"start": 30, "end": 40},
            ],
        },
        {
            "relation_type": "treats",
            "normalized_head": {"system": "synthetic", "code": "h-001"},
            "normalized_tail": {"system": "synthetic", "code": "t-001"},
            "source_id": "synthetic-note-a",
            "score": 0.9,
            "evidence": [
                {"start": 10, "end": 20},
                {"start": 50, "end": 60},
            ],
        },
        {
            "relation_type": "treats",
            "normalized_head": {"system": "SYNTHETIC", "code": "H-001"},
            "normalized_tail": {"system": "SYNTHETIC", "code": "T-001"},
            "source_id": "synthetic-note-b",
            "score": 0.7,
            "evidence": [{"start": 5, "end": 15}],
        },
    ]

    result = collapse_duplicate_relations(candidates, hash_secret="synthetic-key")

    assert len(result) == 1
    relation = result[0]
    assert relation.relation_type == "treats"
    assert relation.head == "synthetic:h-001"
    assert relation.tail == "synthetic:t-001"
    assert relation.score == 0.97
    assert relation.candidate_count == 3
    assert relation.mention_count == 4
    assert relation.independent_source_count == 2
    assert {item.offset for item in relation.evidence_locations} == {
        (5, 15),
        (10, 20),
        (30, 40),
        (50, 60),
    }
    assert len({item.source_id for item in relation.evidence_locations}) == 2


def test_result_is_byte_deterministic_and_does_not_export_source_values() -> None:
    candidates = [
        {
            "relation_type": "associated_with",
            "head": "SYNTHETIC_RAW_HEAD",
            "tail": "SYNTHETIC_RAW_TAIL",
            "source_id": "synthetic-document-2",
            "score": 0.61,
            "evidence": {"start": 12, "end": 25},
        },
        {
            "relation_type": "associated_with",
            "head": "synthetic_raw_head",
            "tail": "synthetic_raw_tail",
            "source_id": "synthetic-document-1",
            "score": 0.62,
            "evidence": {"start": 2, "end": 15},
        },
    ]

    forward = collapse_duplicate_relations(candidates, hash_secret="synthetic-key")
    reverse = collapse_duplicate_relations(
        reversed(candidates),
        hash_secret="synthetic-key",
    )
    payload = forward[0].to_json()

    assert forward == reverse
    assert payload == reverse[0].to_json()
    assert "SYNTHETIC_RAW_HEAD" not in payload
    assert "SYNTHETIC_RAW_TAIL" not in payload
    assert all(
        item.source_id.startswith("hmac-sha256:")
        for item in forward[0].evidence_locations
    )


def test_context_axes_keep_assertion_conflicts_separate() -> None:
    candidates = [
        {
            "relation_type": "associated_with",
            "normalized_head": "SYNTHETIC:H-002",
            "normalized_tail": "SYNTHETIC:T-002",
            "assertion_status": "affirmed",
            "source_id": "synthetic-note",
            "score": 0.8,
            "evidence": {"start": 1, "end": 8},
        },
        {
            "relation_type": "associated_with",
            "normalized_head": "synthetic:h-002",
            "normalized_tail": "synthetic:t-002",
            "assertion_status": "negated",
            "source_id": "synthetic-note",
            "score": 0.9,
            "evidence": {"start": 20, "end": 28},
        },
    ]

    result = collapse_duplicate_relations(candidates)

    assert len(result) == 2
    assert {item.context["assertion_status"] for item in result} == {
        "affirmed",
        "negated",
    }
    assert all(item.independent_source_count == 1 for item in result)


def test_existing_relation_adapter_hashes_surfaces_and_preserves_pair_offsets() -> None:
    head = SpanReference(
        text="SYNTHETIC_MEDICATION",
        label="MEDICATION",
        start=4,
        end=24,
        score=1.0,
    )
    tail = SpanReference(
        text="SYNTHETIC_DOSE",
        label="DOSAGE",
        start=25,
        end=39,
        score=1.0,
    )

    result = collapse_duplicate_relations(
        Relation(head=head, type="dose", tail=tail, score=0.75),
        document_id="synthetic-note",
    )

    assert len(result) == 1
    relation = result[0]
    assert relation.relation_type == "drug_to_dose"
    assert relation.mention_count == 1
    assert relation.evidence_locations[0].offset == (4, 39)
    encoded = json.dumps(relation.to_dict(), sort_keys=True)
    assert "SYNTHETIC_MEDICATION" not in encoded
    assert "SYNTHETIC_DOSE" not in encoded


def test_invalid_candidates_fail_without_echoing_submitted_values() -> None:
    sensitive_marker = "SYNTHETIC_PRIVATE_VALUE"

    with pytest.raises(RelationDeduplicationError) as raised:
        collapse_duplicate_relations(
            {
                "relation_type": "treats",
                "normalized_head": sensitive_marker,
                "normalized_tail": "SYNTHETIC:T-003",
                "score": 0.5,
            }
        )

    assert sensitive_marker not in str(raised.value)
