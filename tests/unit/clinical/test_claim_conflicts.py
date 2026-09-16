"""Synthetic offline tests for evidence-claim contradiction review."""

from __future__ import annotations

import json
import socket

from openmed.clinical.claim_conflicts import (
    CLAIM_REVIEW_CLEAR,
    CLAIM_REVIEW_REQUIRED,
    ClaimConflictReport,
    ClaimRecord,
    ClaimReference,
    SourceIntegrityRecord,
    TemporalRecord,
    review_claim_conflicts,
    review_claims,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


def test_claim_review_routes_assertion_temporal_and_integrity_conflicts() -> None:
    report = review_claim_conflicts(
        [
            {
                "claim_id": "claim-1",
                "expected_assertion": "affirmed",
                "references": [
                    {
                        "evidence_id": "evidence-a",
                        "assertion_id": "assertion-a",
                        "temporal_id": "temporal-a",
                        "source_integrity_id": "source-a",
                        "text": "Synthetic private marker A",
                    },
                    {
                        "evidence_id": "evidence-b",
                        "assertion_id": "assertion-b",
                        "temporal_id": "temporal-b",
                        "source_integrity_id": "source-b",
                        "text": "Synthetic private marker B",
                    },
                ],
            }
        ],
        assertion_records=[
            {"record_id": "assertion-a", "assertion": "affirmed"},
            {"record_id": "assertion-b", "assertion": "negated"},
        ],
        temporal_records=[
            {
                "record_id": "temporal-a",
                "interval": {"start": "2026-01-01", "end": "2026-01-02"},
            },
            {
                "record_id": "temporal-b",
                "interval": {"start": "2026-03-01", "end": "2026-03-02"},
            },
        ],
        source_integrity_records=[
            {
                "record_id": "source-a",
                "status": "verified",
                "source_hash": HASH_A,
            },
            {
                "record_id": "source-b",
                "status": "mismatch",
                "expected_hash": HASH_B,
                "actual_hash": HASH_C,
            },
        ],
    )

    assert isinstance(report, ClaimConflictReport)
    assert report.review_state == CLAIM_REVIEW_REQUIRED
    review = report.claims[0]
    assert review.review_state == CLAIM_REVIEW_REQUIRED
    assert set(review.review_routes) == {
        "assertion_conflict",
        "source_integrity_conflict",
        "temporal_conflict",
    }
    assert {conflict.conflict_type for conflict in report.conflicts} == {
        "assertion",
        "source_integrity",
        "temporal",
    }

    payload = report.to_json()
    assert "synthetic private marker" not in payload.casefold()
    assert "2026-01-01" not in payload
    assert "2026-03-02" not in payload
    assert "evidence-a" in payload
    assert HASH_C in payload


def test_clear_review_is_deterministic_for_mapping_and_record_order() -> None:
    claims = {
        "claim-clear": {
            "evidence_ids": ["evidence-b", "evidence-a"],
        }
    }
    assertions = {
        "evidence-b": {"record_id": "evidence-b", "assertion": "confirmed"},
        "evidence-a": {"record_id": "evidence-a", "assertion": "affirmed"},
    }
    temporal = [
        {"record_id": "evidence-b", "interval": "2026-01-02"},
        {"record_id": "evidence-a", "interval": "2026-01-01/2026-01-03"},
    ]
    integrity = [
        {"record_id": "evidence-b", "verified": True, "text": "marker-b"},
        {"record_id": "evidence-a", "status": "valid", "text": "marker-a"},
    ]

    first = review_claim_conflicts(
        claims,
        assertions=assertions,
        temporal=temporal,
        source_integrity=integrity,
    )
    second = review_claim_conflicts(
        [{"claim_id": "claim-clear", "evidence_ids": ["evidence-a", "evidence-b"]}],
        list(reversed(tuple(assertions.values()))),
        list(reversed(temporal)),
        list(reversed(integrity)),
    )

    assert first.to_dict() == second.to_dict()
    assert first.review_state == CLAIM_REVIEW_CLEAR
    assert first.conflicts == ()
    assert first.claims[0].evidence_ids == ("evidence-a", "evidence-b")
    assert first.claims[0].evidence_hashes
    assert "marker-a" not in first.to_json()
    assert "marker-b" not in first.to_json()


def test_typed_records_and_references_serialize_without_source_values() -> None:
    reference = ClaimReference(
        evidence_id="evidence-typed",
        text_hash="Synthetic raw source marker",
        expected_interval={"start": "2026-04-01", "end": "2026-04-02"},
    )
    claim = ClaimRecord(
        claim_id="claim-typed",
        references=(reference,),
        expected_assertion="affirmed",
    )
    temporal = TemporalRecord(
        record_id="evidence-typed",
        interval={"start": "2026-04-01", "end": "2026-04-02"},
        text_hash="Synthetic temporal source marker",
    )
    integrity = SourceIntegrityRecord(
        record_id="evidence-typed",
        verified=True,
        text_hash="Synthetic integrity source marker",
    )

    payload = json.dumps(
        {
            "claim": claim.to_dict(),
            "temporal": temporal.to_dict(),
            "integrity": integrity.to_dict(),
        }
    )
    assert "synthetic raw source marker" not in payload.casefold()
    assert "synthetic temporal source marker" not in payload.casefold()
    assert "synthetic integrity source marker" not in payload.casefold()
    assert "2026-04-01" not in payload


def test_missing_records_route_to_review_without_network_access(
    monkeypatch,
) -> None:
    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("claim review must not access the network")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    report = review_claim_conflicts(
        [{"claim_id": "claim-missing", "evidence_ids": ["evidence-missing"]}],
        assertion_records=[],
        temporal_records=[],
        source_integrity_records=[],
    )

    assert report.review_state == CLAIM_REVIEW_REQUIRED
    assert report.conflicts[0].conflict_type == "missing_evidence"
    assert report.conflicts[0].evidence_ids == ("evidence-missing",)


def test_compatibility_wrapper_accepts_record_collection_aliases() -> None:
    report = review_claims(
        [{"claim_id": "claim-alias", "evidence_ids": ["evidence-alias"]}],
        assertions={
            "evidence-alias": {"assertion": "affirmed"},
        },
        temporal={
            "evidence-alias": {"interval": "2026-05-01"},
        },
        source_integrity={
            "evidence-alias": {"hash": HASH_A, "verified": True},
        },
    )

    assert report.review_state == CLAIM_REVIEW_CLEAR
    assert report.claims[0].evidence_ids == ("evidence-alias",)
