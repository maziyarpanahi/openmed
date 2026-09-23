"""Tests for the value-free clinical evidence coverage matrix."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    COVERAGE_STATUSES,
    EvidenceCoverageError,
    EvidenceCoverageMatrix,
    build_evidence_coverage_matrix,
    fingerprint_source,
)


def test_matrix_counts_statuses_and_excludes_raw_values():
    raw_claim_text = "synthetic claim text must not be serialized"
    raw_evidence_text = "synthetic evidence text must not be serialized"
    matrix = build_evidence_coverage_matrix(
        [
            {
                "claim_id": "claim-z2",
                "claim_text": raw_claim_text,
                "required_evidence": [
                    {
                        "evidence_class": "lab_result",
                        "review_state": "reviewed",
                        "source": "synthetic-source-z2",
                        "evidence_text": raw_evidence_text,
                    },
                    {"evidence_class": "imaging", "review_state": "missing"},
                ],
            },
            {
                "claim_id": "claim-a1",
                "required_evidence": [
                    {
                        "evidence_class": "lab_result",
                        "review_state": "unreviewed",
                        "source_fingerprint": fingerprint_source("synthetic-source-a1"),
                    },
                    {
                        "evidence_class": "review_note",
                        "review_state": "conflicting",
                    },
                ],
            },
        ]
    )

    report = matrix.to_dict()
    assert report["claim_count"] == 2
    assert report["required_evidence_count"] == 4
    assert report["status_counts"] == {
        "present": 1,
        "missing": 1,
        "conflicting": 1,
        "unreviewed": 1,
    }
    assert [claim["claim_id"] for claim in report["claims"]] == [
        "claim-a1",
        "claim-z2",
    ]
    serialized = matrix.to_json()
    rendered = matrix.to_markdown()
    for unsafe_value in (raw_claim_text, raw_evidence_text):
        assert unsafe_value not in serialized
        assert unsafe_value not in rendered
    assert '"claim_text"' not in serialized
    assert '"evidence_text"' not in serialized
    assert all(
        fingerprint.startswith("sha256:")
        for record in matrix.records
        for fingerprint in record.source_fingerprints
    )


def test_input_order_does_not_change_rows_counts_or_hashes():
    claims = [
        {
            "claim_id": "claim-b2",
            "required_evidence": {
                "local_record": {
                    "status": "present",
                    "source_fingerprint": fingerprint_source("source-b2"),
                },
                "second_source": "missing",
            },
        },
        {
            "claim_id": "claim-a1",
            "required_evidence": [
                {
                    "evidence_class": "local_record",
                    "status": "present",
                    "source_fingerprint": fingerprint_source("source-a1"),
                }
            ],
        },
    ]

    forward = build_evidence_coverage_matrix(claims)
    reversed_matrix = build_evidence_coverage_matrix(tuple(reversed(claims)))

    assert forward.to_json() == reversed_matrix.to_json()
    assert forward.matrix_hash == reversed_matrix.matrix_hash
    assert forward.status_counts == {
        "present": 2,
        "missing": 1,
        "conflicting": 0,
        "unreviewed": 0,
    }


def test_claim_level_class_maps_are_normalized_into_cells():
    matrix = build_evidence_coverage_matrix(
        {
            "claim-a1": {
                "required_evidence_classes": ["local_record", "second_source"],
                "review_states": {
                    "local_record": "reviewed",
                    "second_source": "unreviewed",
                },
                "source_fingerprints_by_class": {
                    "local_record": fingerprint_source("source-a"),
                    "second_source": fingerprint_source("source-b"),
                },
            }
        }
    )

    assert [record.status for record in matrix.records] == ["present", "unreviewed"]
    assert matrix.status_counts == {
        "present": 1,
        "missing": 0,
        "conflicting": 0,
        "unreviewed": 1,
    }


def test_duplicate_class_sources_merge_and_disagreement_is_conflicting():
    matrix = build_evidence_coverage_matrix(
        [
            {
                "claim_id": "claim-a1",
                "required_evidence": [
                    {
                        "evidence_class": "local_record",
                        "status": "present",
                        "source_fingerprint": fingerprint_source("source-a"),
                    },
                    {
                        "evidence_class": "local_record",
                        "status": "missing",
                    },
                ],
            }
        ]
    )

    record = matrix.records[0]
    assert record.status == "conflicting"
    assert record.review_state == "conflicting"
    assert len(record.source_fingerprints) == 1
    assert matrix.status_counts["conflicting"] == 1


def test_round_trip_preserves_the_value_free_matrix():
    matrix = build_evidence_coverage_matrix(
        {
            "claim-a1": {
                "local_record": {
                    "review_state": "reviewed",
                    "source_fingerprint": fingerprint_source("source-a"),
                }
            }
        }
    )

    restored = EvidenceCoverageMatrix.from_mapping(json.loads(matrix.to_json()))

    assert restored.to_json() == matrix.to_json()


def test_invalid_input_errors_do_not_echo_raw_values():
    raw_value = "synthetic private value must not appear in an exception"

    with pytest.raises(EvidenceCoverageError) as error:
        build_evidence_coverage_matrix(
            [
                {
                    "claim_id": "claim-a1",
                    "required_evidence": [
                        {
                            "evidence_class": "lab_result",
                            "status": "present",
                            "evidence_text": raw_value,
                        }
                    ],
                }
            ]
        )

    assert raw_value not in str(error.value)
    assert COVERAGE_STATUSES == ("present", "missing", "conflicting", "unreviewed")
