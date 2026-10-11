"""Synthetic offline tests for the guarded evidence packet boundary."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.evidence_packet import (
    REJECTION_INVALID_REVIEW_STATE,
    REJECTION_INVALID_SOURCE_OFFSET,
    REJECTION_NOT_SYNTHETIC,
    REJECTION_POLICY_MISMATCH,
    REJECTION_RAW_TEXT,
    REJECTION_UNVERIFIED,
    EvidencePacket,
    EvidencePacketValidationError,
    EvidenceReference,
    build_evidence_packet,
    fingerprint_evidence_review,
    fingerprint_policy,
    validate_evidence_packet,
)
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewStateMachine,
    make_opaque_event_id,
)

POLICY_FINGERPRINT = fingerprint_policy({"policy": "synthetic-review", "version": 1})


@pytest.mark.parametrize(
    ("field", "value", "category"),
    [
        ("verified", False, REJECTION_UNVERIFIED),
        ("synthetic", False, REJECTION_NOT_SYNTHETIC),
        ("review_state", "queued", REJECTION_INVALID_REVIEW_STATE),
        ("start", -1, REJECTION_INVALID_SOURCE_OFFSET),
        ("end", 18, REJECTION_INVALID_REVIEW_STATE),
        ("review_transitions", (), REJECTION_INVALID_REVIEW_STATE),
    ],
)
def test_nested_reference_invariants_are_rechecked(field, value, category):
    packet = build_evidence_packet(
        [_reference()], policy_fingerprint=POLICY_FINGERPRINT
    )
    reference = packet.references[0]
    object.__setattr__(reference, field, value)
    with pytest.raises(EvidencePacketValidationError) as caught:
        validate_evidence_packet(packet)
    assert caught.value.category == category
    filtered = build_evidence_packet([reference], policy_fingerprint=POLICY_FINGERPRINT)
    assert filtered.references == ()
    assert filtered.rejection_counts == {category: 1}


def test_nested_rejection_report_invariants_are_rechecked():
    packet = build_evidence_packet(
        [_reference()], policy_fingerprint=POLICY_FINGERPRINT
    )
    object.__setattr__(packet.rejection_report, "rejected_count", -1)
    with pytest.raises(ValueError, match="non-negative"):
        validate_evidence_packet(packet)


def _reference(reference_id: str = "synthetic:ref-001", **overrides):
    payload = {
        "reference_id": reference_id,
        "source_id": "synthetic:document-001",
        "start": 8,
        "end": 17,
        "review_state": "approved",
        "policy_fingerprint": POLICY_FINGERPRINT,
        "synthetic": True,
        "verified": True,
    }
    payload.update(overrides)
    if "review_transitions" not in overrides:
        start, end = payload["start"], payload["end"]
        if not isinstance(start, int) or start < 0 or end <= start:
            start, end = 8, 17
        provenance = fingerprint_evidence_review(
            reference_id=reference_id,
            source_id=payload["source_id"],
            start=start,
            end=end,
            policy_fingerprint=payload["policy_fingerprint"],
        )
        machine = ReviewStateMachine()
        machine.transition(
            ReviewState.IN_REVIEW,
            make_opaque_event_id((reference_id, "in_review")),
            provenance,
        )
        machine.transition(
            ReviewState.APPROVED,
            make_opaque_event_id((reference_id, "approved")),
            provenance,
        )
        payload["review_transitions"] = [item.to_dict() for item in machine.transitions]
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
    nested = _reference("synthetic:nested")
    nested["review_transitions"][0]["text"] = sensitive_marker
    candidates = [
        _reference("synthetic:raw", text=sensitive_marker),
        nested,
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
        REJECTION_RAW_TEXT: 2,
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


def test_synthetic_flag_cannot_make_patient_values_safe_identifiers() -> None:
    patient_value = "SyntheticPatientName"
    packet = build_evidence_packet(
        [
            _reference(patient_value),
            _reference("synthetic:ref-002", source_id=patient_value),
        ],
        policy_fingerprint=POLICY_FINGERPRINT,
    )

    assert packet.rejection_counts == {REJECTION_NOT_SYNTHETIC: 2}
    assert patient_value not in packet.to_json()
    with pytest.raises(EvidencePacketValidationError) as caught:
        build_evidence_packet(
            [_reference()],
            policy_fingerprint=POLICY_FINGERPRINT,
            packet_id=patient_value,
        )
    assert caught.value.category == REJECTION_NOT_SYNTHETIC
    assert patient_value not in str(caught.value)


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


def test_approval_history_is_required_and_bound_to_source_offsets() -> None:
    missing = _reference(review_transitions=[])
    skipped = _reference()
    skipped["review_transitions"] = [
        {
            **skipped["review_transitions"][-1],
            "sequence": 1,
            "from_state": "queued",
        }
    ]
    moved = _reference()
    moved["start"] = 9

    packet = build_evidence_packet(
        [missing, skipped, moved], policy_fingerprint=POLICY_FINGERPRINT
    )
    assert packet.accepted_count == 0
    assert packet.rejection_counts == {REJECTION_INVALID_REVIEW_STATE: 3}


def test_reopened_approval_does_not_enter_the_packet() -> None:
    candidate = _reference()
    provenance = fingerprint_evidence_review(
        reference_id=candidate["reference_id"],
        source_id=candidate["source_id"],
        start=candidate["start"],
        end=candidate["end"],
        policy_fingerprint=candidate["policy_fingerprint"],
    )
    machine = ReviewStateMachine()
    for state in (ReviewState.IN_REVIEW, ReviewState.APPROVED, ReviewState.REOPENED):
        machine.transition(
            state,
            make_opaque_event_id((candidate["reference_id"], state.value)),
            provenance,
        )
    candidate["review_transitions"] = [item.to_dict() for item in machine.transitions]

    packet = build_evidence_packet([candidate], policy_fingerprint=POLICY_FINGERPRINT)
    assert packet.rejection_counts == {REJECTION_INVALID_REVIEW_STATE: 1}


def test_mapping_and_json_round_trip_is_deterministic() -> None:
    packet = build_evidence_packet(
        [_reference()],
        policy_fingerprint=POLICY_FINGERPRINT,
        packet_id="synthetic:packet-001",
    )

    restored = EvidencePacket.from_json(packet.to_json())
    assert packet.to_dict()["schema_version"] == 2
    assert restored.to_dict() == packet.to_dict()
    assert validate_evidence_packet(packet).to_dict() == packet.to_dict()
    assert packet.to_json() == restored.to_json()
    assert packet.digest == restored.digest
    with pytest.raises(ValueError, match="unsupported evidence packet schema version"):
        EvidencePacket.from_dict({**packet.to_dict(), "schema_version": 1})


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
