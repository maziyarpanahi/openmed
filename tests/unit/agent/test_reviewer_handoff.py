"""Offline tests for strict reviewer handoff packets."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openmed.agent import (
    ArtifactKind,
    ArtifactReference,
    RequestedDecision,
    ReviewerHandoffError,
    ReviewerHandoffPacket,
    RunId,
    WorkflowId,
)
from openmed.agent.reviewer_handoff import REVIEWER_HANDOFF_SCHEMA_VERSION

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def _reference(index: int = 1, **updates: Any) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "artifact_id": "art_" + f"{index:032x}",
        "kind": "evidence",
        "schema_id": "openmed.agent.evidence.v1",
        "sha256": f"{index:064x}",
        "byte_size": index,
    }
    reference.update(updates)
    return reference


def _payload(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": "run_" + "1" * 32,
        "workflow_id": "workflow:org.openmed/reviewer-handoff@1.0.0",
        "reason_code": "conflicting_evidence",
        "requested_decision": "resolve_evidence_conflict",
        "evidence_references": [_reference(1), _reference(2)],
        "issued_at": "2026-09-20T11:00:00Z",
        "expires_at": "2026-09-20T13:00:00Z",
    }
    payload.update(updates)
    return payload


def test_valid_packet_round_trips_deterministically_and_preserves_order() -> None:
    packet = ReviewerHandoffPacket.from_dict(_payload(), now=NOW)

    assert ReviewerHandoffPacket.from_json(packet.to_json(), now=NOW) == packet
    assert [reference.artifact_id for reference in packet.evidence_references] == [
        "art_" + "0" * 31 + "1",
        "art_" + "0" * 31 + "2",
    ]
    assert packet.to_json() == json.dumps(
        packet.to_dict(), sort_keys=True, separators=(",", ":")
    )
    assert list(packet.to_dict()) == [
        "schema_version",
        "run_id",
        "workflow_id",
        "reason_code",
        "requested_decision",
        "evidence_references",
        "issued_at",
        "expires_at",
    ]


def test_valid_packet_only_requests_review_and_grants_no_authority() -> None:
    packet = ReviewerHandoffPacket.from_dict(_payload(), now=NOW)

    assert packet.requires_human_review is True
    assert packet.authorizes_clinical_action is False


@pytest.mark.parametrize(
    "reason_code",
    [
        "insufficient_evidence",
        "out_of_scope",
        "low_confidence",
        "conflicting_evidence",
        "safety_review",
        "human_gate",
    ],
)
def test_abstention_and_review_reason_codes_are_accepted(reason_code: str) -> None:
    packet = ReviewerHandoffPacket.from_dict(_payload(reason_code=reason_code), now=NOW)
    assert packet.reason_code == reason_code


@pytest.mark.parametrize("decision", list(RequestedDecision))
def test_requested_decision_vocabulary_round_trips(decision: RequestedDecision) -> None:
    packet = ReviewerHandoffPacket.from_dict(
        _payload(requested_decision=decision.value), now=NOW
    )
    assert packet.requested_decision is decision


def test_missing_requested_decision_fails_closed() -> None:
    payload = _payload()
    del payload["requested_decision"]

    with pytest.raises(ReviewerHandoffError) as caught:
        ReviewerHandoffPacket.from_dict(payload, now=NOW)

    assert caught.value.code == "missing_field"


def test_malformed_evidence_digest_fails_without_echo() -> None:
    sentinel = "SYNTHETIC-SECRET-DIGEST"
    payload = _payload(evidence_references=[_reference(sha256=sentinel)])

    with pytest.raises(ReviewerHandoffError) as caught:
        ReviewerHandoffPacket.from_dict(payload, now=NOW)

    assert caught.value.code == "invalid_evidence_reference"
    assert sentinel not in "".join(traceback.format_exception(caught.value))


def test_expired_packet_fails_closed() -> None:
    with pytest.raises(ReviewerHandoffError) as caught:
        ReviewerHandoffPacket.from_dict(
            _payload(expires_at="2026-09-20T12:00:00Z"), now=NOW
        )

    assert caught.value.code == "expired"
    assert caught.value.field_name == "expires_at"


def test_duplicate_evidence_reference_fails_closed() -> None:
    reference = _reference()

    with pytest.raises(ReviewerHandoffError) as caught:
        ReviewerHandoffPacket.from_dict(
            _payload(evidence_references=[reference, reference]), now=NOW
        )

    assert caught.value.code == "duplicate_evidence_reference"


def test_unknown_free_text_field_fails_with_phi_safe_error() -> None:
    sentinel = "Jane Synthetic has diagnosis Z99.999; bearer secret"

    with pytest.raises(ReviewerHandoffError) as caught:
        ReviewerHandoffPacket.from_dict(_payload(clinical_summary=sentinel), now=NOW)

    assert caught.value.code == "unknown_field"
    assert sentinel not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        (
            "schema_version",
            "openmed.agent.reviewer_handoff.v2",
            "invalid_schema_version",
        ),
        ("run_id", "Patient-Jane-Doe", "invalid_identifier"),
        ("workflow_id", "patient/jane", "invalid_identifier"),
        ("reason_code", "Patient is unstable", "unknown_reason"),
        ("reason_code", "completed", "unknown_reason"),
        ("requested_decision", "approve_treatment", "unknown_decision"),
        ("issued_at", "2026-09-20 11:00:00", "invalid_timestamp"),
        ("expires_at", "2026-09-20T13:00:00+00:00", "invalid_timestamp"),
    ],
)
def test_invalid_fields_fail_without_echo(field: str, value: Any, code: str) -> None:
    with pytest.raises(ReviewerHandoffError) as caught:
        ReviewerHandoffPacket.from_dict(_payload(**{field: value}), now=NOW)

    assert caught.value.code == code
    assert str(value) not in str(caught.value)


def test_expiry_must_follow_issue_time() -> None:
    with pytest.raises(ReviewerHandoffError) as caught:
        ReviewerHandoffPacket.from_dict(
            _payload(expires_at="2026-09-20T10:59:59Z"), now=NOW
        )

    assert caught.value.code == "invalid_expiry"


def test_empty_evidence_collection_is_explicit_and_valid() -> None:
    packet = ReviewerHandoffPacket.from_dict(_payload(evidence_references=[]), now=NOW)
    assert packet.evidence_references == ()
    assert packet.to_dict()["evidence_references"] == []


def test_direct_construction_enforces_typed_contract() -> None:
    reference = ArtifactReference(
        artifact_id="art_" + "1" * 32,
        kind=ArtifactKind.EVIDENCE,
        schema_id="openmed.agent.evidence.v1",
        sha256="a" * 64,
        byte_size=1,
    )

    with pytest.raises(ReviewerHandoffError, match="run_id: wrong_identifier_kind"):
        ReviewerHandoffPacket(
            run_id="run_" + "1" * 32,  # type: ignore[arg-type]
            workflow_id=WorkflowId("workflow:org.openmed/handoff@1.0.0"),
            reason_code="human_gate",
            requested_decision=RequestedDecision.DECIDE_NEXT_STEP,
            evidence_references=(reference,),
            issued_at=NOW,
            expires_at=NOW + timedelta(hours=1),
            validation_time=NOW,
        )


def test_duplicate_json_fields_and_malformed_json_fail_closed() -> None:
    duplicate = json.dumps(_payload()).replace(
        '"reason_code": "conflicting_evidence"',
        '"reason_code": "human_gate", "reason_code": "conflicting_evidence"',
    )
    with pytest.raises(ReviewerHandoffError, match="malformed_json"):
        ReviewerHandoffPacket.from_json(duplicate, now=NOW)

    sentinel = "Jane Synthetic /private/chart"
    with pytest.raises(ReviewerHandoffError) as caught:
        ReviewerHandoffPacket.from_json(sentinel, now=NOW)
    assert caught.value.code == "malformed_json"
    assert sentinel not in "".join(traceback.format_exception(caught.value))


def test_contract_is_exported_from_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.ReviewerHandoffPacket is ReviewerHandoffPacket
    assert agent.ReviewerHandoffError is ReviewerHandoffError
    assert agent.RequestedDecision is RequestedDecision
    assert agent.REVIEWER_HANDOFF_SCHEMA_VERSION == REVIEWER_HANDOFF_SCHEMA_VERSION
