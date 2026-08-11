"""Synthetic offline tests for the guarded evidence packet boundary."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.evidence_packet import (
    REJECTION_INVALID_SOURCE_OFFSET,
    REJECTION_NOT_SYNTHETIC,
    REJECTION_POLICY_MISMATCH,
    REJECTION_RAW_TEXT,
    REJECTION_UNVERIFIED,
    EvidencePacket,
    EvidencePacketValidationError,
    EvidenceReference,
    build_evidence_packet,
    fingerprint_policy,
    validate_evidence_packet,
)

POLICY_FINGERPRINT = fingerprint_policy({"policy": "synthetic-review", "version": 1})


def _reference(reference_id: str = "synthetic:ref-001", **overrides):
    payload = {
        "reference_id": reference_id,
        "source_id": "synthetic:document-001",
        "start": 8,
        "end": 17,
        "review_state": "reviewed",
        "policy_fingerprint": POLICY_FINGERPRINT,
        "synthetic": True,
        "verified": True,
    }
    payload.update(overrides)
    return payload


def test_valid_references_are_sorted_and_serialized_without_text() -> None:
    packet = build_evidence_packet(
        [
            _reference("synthetic:ref-002", start=28, end=36),
            _reference("synthetic:ref-001", start=4, end=12),
        ],
        policy_fingerprint=POLICY_FINGERPRINT,
    )

    assert isinstance(packet, EvidencePacket)
    assert [reference.reference_id for reference in packet.references] == [
        "synthetic:ref-001",
        "synthetic:ref-002",
    ]
    assert packet.rejection_counts == {}
    assert packet.to_dict()["rejection_report"] == {
        "input_count": 2,
        "accepted_count": 2,
        "rejected_count": 0,
        "rejection_counts": {},
    }
    assert "text" not in packet.to_json()


def test_rejections_are_stable_counts_only_and_do_not_leak_values() -> None:
    sensitive_marker = "synthetic-sensitive-marker"
    candidates = [
        _reference("synthetic:raw", text=sensitive_marker),
        _reference("synthetic:unverified", verified=False),
        _reference("synthetic:external", synthetic=False),
        _reference("synthetic:offset", start=-1),
    ]

    packet = build_evidence_packet(
        candidates,
        policy_fingerprint=POLICY_FINGERPRINT,
    )

    assert packet.references == ()
    assert packet.rejection_counts == {
        REJECTION_RAW_TEXT: 1,
        REJECTION_UNVERIFIED: 1,
        REJECTION_NOT_SYNTHETIC: 1,
        REJECTION_INVALID_SOURCE_OFFSET: 1,
    }
    serialized = json.dumps(packet.to_dict(), sort_keys=True)
    assert sensitive_marker not in serialized
    with pytest.raises(EvidencePacketValidationError) as error:
        EvidenceReference.from_dict(_reference("synthetic:bad", text=sensitive_marker))
    assert error.value.category == REJECTION_RAW_TEXT
    assert sensitive_marker not in str(error.value)


def test_policy_fingerprint_mismatch_is_rejected_without_record_details() -> None:
    other_policy = fingerprint_policy({"policy": "other-synthetic-policy"})
    packet = build_evidence_packet(
        [_reference(policy_fingerprint=other_policy)],
        policy_fingerprint=POLICY_FINGERPRINT,
    )

    assert packet.accepted_count == 0
    assert packet.rejection_counts == {REJECTION_POLICY_MISMATCH: 1}


def test_offsets_and_review_state_are_validated_before_packaging() -> None:
    with pytest.raises(EvidencePacketValidationError) as offset_error:
        EvidenceReference.from_dict(_reference(start=10, end=10))
    assert offset_error.value.category == REJECTION_INVALID_SOURCE_OFFSET

    with pytest.raises(EvidencePacketValidationError) as review_error:
        EvidenceReference.from_dict(_reference(review_state="unreviewed"))
    assert review_error.value.category == "invalid_review_state"


def test_mapping_and_json_round_trip_is_deterministic() -> None:
    packet = build_evidence_packet(
        [_reference()],
        policy_fingerprint=POLICY_FINGERPRINT,
        packet_id="synthetic:packet-001",
    )

    restored = EvidencePacket.from_json(packet.to_json())
    assert restored.to_dict() == packet.to_dict()
    assert validate_evidence_packet(packet).to_dict() == packet.to_dict()
    assert packet.to_json() == restored.to_json()


def test_policy_fingerprint_is_local_and_deterministic() -> None:
    policy = {"review": ["synthetic", "verified"], "version": 1}
    assert fingerprint_policy(policy) == fingerprint_policy(
        {"version": 1, "review": ["synthetic", "verified"]}
    )
    assert POLICY_FINGERPRINT.startswith("sha256:")


def test_empty_packet_requires_an_explicit_policy_fingerprint() -> None:
    with pytest.raises(EvidencePacketValidationError):
        build_evidence_packet([])

    packet = build_evidence_packet([], policy_fingerprint=POLICY_FINGERPRINT)
    assert packet.references == ()
    assert packet.rejection_report.input_count == 0
