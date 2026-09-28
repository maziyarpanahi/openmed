from __future__ import annotations

import json
from datetime import datetime, timezone

from jsonschema import Draft202012Validator

from openmed.clinical.journey_contracts import canonical_digest
from openmed.clinical.review_sla import summarize_review_queue
from openmed.clinical.review_transitions import (
    ClinicalReviewPacket,
    load_clinical_review_schema,
    transition_review_packet,
)


def _packet(
    suffix: str,
    *,
    state: str = "queued",
    priority: str = "normal",
    created_at: str = "2026-09-20T10:00:00Z",
    expires_at: str | None = None,
) -> ClinicalReviewPacket:
    packet = ClinicalReviewPacket(
        packet_id=f"reviewpacket_{suffix * 16}",
        conflict_id=f"conflict_{suffix * 16}",
        fact_ids=(f"fact_{suffix * 16}", f"fact_{suffix * 15}b"),
        state="queued",
        priority=priority,
        created_at=created_at,
        expires_at=expires_at,
        policy_id="openmed.fact.reconciliation",
        policy_version="1.0.0",
        provenance_fingerprint=canonical_digest({"suffix": suffix}),
    )
    if state == "queued":
        return packet
    if state != "expired":
        raise ValueError("test helper only supports queued or expired")
    transitioned = transition_review_packet(
        packet,
        to_state="expired",
        occurred_at="2026-09-21T09:00:00Z",
        policy_id="openmed.review.transitions",
        policy_version="1.0.0",
        provenance_fingerprint=canonical_digest({"suffix": suffix, "state": state}),
        reason_code="review_expired",
    )
    assert transitioned.value is not None
    return transitioned.value


def test_queue_summary_uses_injected_clock_and_contains_counts_only() -> None:
    packets = (
        _packet("a", priority="critical", created_at="2026-09-21T09:30:00Z"),
        _packet("c", priority="high", created_at="2026-09-21T00:00:00Z"),
        _packet("d", state="expired", created_at="2026-09-19T10:00:00Z"),
    )
    now = datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc)

    first = summarize_review_queue(packets, now=now)
    second = summarize_review_queue(tuple(reversed(packets)), now=now)

    assert first.to_dict() == second.to_dict()
    assert first.total == 3
    assert first.expired_count == 1
    assert first.overdue_count == 1
    assert first.priority_counts["critical"] == 1
    assert first.age_bucket_counts["under_4h"] == 1
    payload = json.dumps(first.to_dict())
    assert "packet_id" not in payload
    assert "fact_" not in payload
    assert "reviewer" not in payload


def test_queue_summary_schema_validates() -> None:
    summary = summarize_review_queue(
        (_packet("a"),),
        now=datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc),
    )
    schema = load_clinical_review_schema("clinical_review_queue_summary")

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(summary.to_dict())
