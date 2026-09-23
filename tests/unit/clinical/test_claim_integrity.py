"""Tests for deterministic, privacy-safe claim-packet integrity checks."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.claim_integrity import (
    CLAIM_PACKET_SCHEMA_VERSION,
    DIGEST_MISMATCH_REASON,
    DUPLICATE_RECORD_REASON,
    DUPLICATE_REFERENCE_REASON,
    MISSING_REFERENCE_REASON,
    MUTATED_RECORD_REASON,
    REORDERED_REFERENCE_REASON,
    ClaimIntegrityError,
    canonicalize_claim_packet,
    check_claim_packet_integrity,
    compute_claim_packet_digest,
    verify_claim_packet_digest,
)


def _packet() -> dict[str, object]:
    return {
        "claims": [
            {
                "claim_id": "claim-a",
                "claim_text": "synthetic claim value",
                "citation_ids": ["citation-a", "citation-b"],
                "policy_id": "policy-a",
            }
        ],
        "citations": [
            {
                "citation_id": "citation-b",
                "citation_value": "synthetic citation beta",
            },
            {
                "citation_id": "citation-a",
                "citation_value": "synthetic citation alpha",
            },
        ],
        "reviews": [
            {
                "review_id": "review-a",
                "claim_ids": ["claim-a"],
                "review_value": "synthetic review value",
            }
        ],
        "policy": {
            "policy_id": "policy-a",
            "policy_version": "v1",
            "policy_value": "synthetic policy value",
        },
    }


def test_digest_is_deterministic_for_mapping_and_record_order() -> None:
    packet = _packet()
    reordered = {
        "policy": packet["policy"],
        "reviews": list(reversed(packet["reviews"])),
        "citations": list(reversed(packet["citations"])),
        "claims": list(reversed(packet["claims"])),
    }

    first = compute_claim_packet_digest(packet)
    second = compute_claim_packet_digest(reordered)

    assert first == second
    assert first.startswith("sha256:")
    assert verify_claim_packet_digest(packet, first) is True


def test_public_canonical_form_is_versioned_and_value_free() -> None:
    packet = _packet()
    canonical = canonicalize_claim_packet(packet)
    report = check_claim_packet_integrity(packet)
    serialized = json.dumps(canonical) + report.to_json() + report.to_markdown()

    assert canonical["schema_version"] == CLAIM_PACKET_SCHEMA_VERSION
    assert canonical["digest"] == report.digest
    assert canonical["counts"]["claims"] == 1
    assert "synthetic claim value" not in serialized
    assert "synthetic citation alpha" not in serialized
    assert "synthetic review value" not in serialized
    assert "synthetic policy value" not in serialized
    assert "claim-a" not in serialized
    assert "citation-a" not in serialized
    assert report.passed is True
    assert report.failure_reasons == ()


def test_reordered_reference_is_detected_against_a_baseline() -> None:
    expected = _packet()
    candidate = _packet()
    candidate_claim = candidate["claims"][0]
    assert isinstance(candidate_claim, dict)
    candidate_claim["citation_ids"] = ["citation-b", "citation-a"]

    report = check_claim_packet_integrity(
        candidate,
        expected_packet=expected,
    )

    assert report.passed is False
    assert report.digest_matches is False
    assert report.reordered_reference_count == 1
    assert REORDERED_REFERENCE_REASON in report.issues
    assert DIGEST_MISMATCH_REASON in report.issues
    assert report.mutated_record_count == 0


def test_missing_and_duplicate_references_fail_closed_without_values() -> None:
    packet = _packet()
    claim = packet["claims"][0]
    assert isinstance(claim, dict)
    claim["citation_ids"] = ["citation-a", "citation-a", "unavailable-citation"]
    packet["citations"] = [packet["citations"][1]]

    report = check_claim_packet_integrity(packet)
    serialized = report.to_json() + report.to_markdown()

    assert report.passed is False
    assert report.missing_reference_count == 1
    assert report.duplicate_reference_count == 1
    assert report.issues == (
        MISSING_REFERENCE_REASON,
        DUPLICATE_REFERENCE_REASON,
    )
    assert "unavailable-citation" not in serialized
    assert "synthetic citation alpha" not in serialized


def test_duplicate_record_and_mutated_record_are_reported() -> None:
    expected = _packet()
    candidate = _packet()
    citations = candidate["citations"]
    assert isinstance(citations, list)
    duplicate = dict(citations[0])
    citations.append(duplicate)

    report = check_claim_packet_integrity(
        candidate,
        expected_packet=expected,
    )
    serialized = report.to_json()

    assert report.passed is False
    assert report.duplicate_record_count == 1
    assert report.mutated_record_count == 0
    assert DUPLICATE_RECORD_REASON in report.issues
    assert "citation-b" not in serialized

    mutated_packet = _packet()
    mutated_citations = mutated_packet["citations"]
    assert isinstance(mutated_citations, list)
    mutated_citations[0]["citation_value"] = "synthetic mutated citation"
    mutated_report = check_claim_packet_integrity(
        mutated_packet,
        expected_packet=expected,
    )

    assert mutated_report.mutated_record_count == 1
    assert MUTATED_RECORD_REASON in mutated_report.issues


def test_invalid_input_error_contains_only_a_reason_code() -> None:
    report = check_claim_packet_integrity(
        {
            "claims": [
                {
                    "claim_id": "claim-a",
                    "citation_ids": [{"text": "synthetic unkeyed reference"}],
                }
            ]
        }
    )

    assert report.passed is False
    assert report.issues == ("invalid_reference",)
    assert "object" not in report.to_json()

    with pytest.raises(ClaimIntegrityError) as error:
        compute_claim_packet_digest({"claims": [object()]})

    assert str(error.value) == "claim packet rejected: invalid_packet"
    assert "object" not in str(error.value)


def test_digest_mutation_is_detected_without_a_baseline_packet() -> None:
    expected = compute_claim_packet_digest(_packet())
    candidate = _packet()
    candidate["policy"]["policy_value"] = "synthetic changed policy"

    report = check_claim_packet_integrity(candidate, expected_digest=expected)

    assert report.passed is False
    assert report.digest_matches is False
    assert report.issues == (DIGEST_MISMATCH_REASON,)
    assert "synthetic changed policy" not in report.to_json()
