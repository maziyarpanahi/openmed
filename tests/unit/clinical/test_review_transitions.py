from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from openmed.clinical.journey_contracts import canonical_digest
from openmed.clinical.review_transitions import (
    ClinicalReviewPacket,
    load_clinical_review_schema,
    persist_review_packet,
    transition_review_packet,
)
from openmed.structured.store import SQLiteJourneyStore, StorePoint, StoreState

T0 = "2026-09-21T10:00:00Z"
T1 = "2026-09-21T11:00:00Z"
T2 = "2026-09-21T12:00:00Z"
T3 = "2026-09-21T13:00:00Z"


def _packet() -> ClinicalReviewPacket:
    return ClinicalReviewPacket(
        packet_id="reviewpacket_aaaaaaaaaaaaaaaa",
        conflict_id="conflict_aaaaaaaaaaaaaaaa",
        fact_ids=("fact_aaaaaaaaaaaaaaaa", "fact_bbbbbbbbbbbbbbbb"),
        state="queued",
        priority="high",
        created_at=T0,
        expires_at="2026-09-22T10:00:00Z",
        policy_id="openmed.fact.reconciliation",
        policy_version="1.0.0",
        provenance_fingerprint=canonical_digest({"synthetic": "packet"}),
    )


def _transition(packet: ClinicalReviewPacket, state: str, at: str):
    return transition_review_packet(
        packet,
        to_state=state,
        occurred_at=at,
        policy_id="openmed.review.transitions",
        policy_version="1.0.0",
        provenance_fingerprint=canonical_digest(
            {"packet_id": packet.packet_id, "state": state}
        ),
        reason_code=f"review_{state}",
    )


def test_guarded_transition_chain_is_deterministic_and_identity_free() -> None:
    first = _transition(_packet(), "in_review", T1)
    assert first.ok and first.value is not None
    approved = _transition(first.value, "approved", T2)
    assert approved.ok and approved.value is not None

    replay_first = _transition(_packet(), "in_review", T1)
    assert replay_first.value is not None
    replay = _transition(replay_first.value, "approved", T2)
    assert replay.value is not None

    assert approved.value.to_dict() == replay.value.to_dict()
    assert approved.value.state == "approved"
    assert len(approved.value.transitions) == 2
    payload = approved.value.to_json()
    assert "reviewer" not in payload
    assert "case_content" not in payload


def test_review_cannot_skip_required_in_review_state() -> None:
    skipped = _transition(_packet(), "approved", T1)

    assert skipped.state is StoreState.CONFLICT
    assert skipped.code == "review_transition_invalid"


def test_completed_review_reopens_with_append_only_provenance() -> None:
    in_review = _transition(_packet(), "in_review", T1).value
    assert in_review is not None
    approved = _transition(in_review, "approved", T2).value
    assert approved is not None
    reopened = _transition(approved, "reopened", T3)

    assert reopened.ok and reopened.value is not None
    assert reopened.value.state == "reopened"
    assert approved.transitions == reopened.value.transitions[:-1]


def test_packet_versions_persist_and_reconstruct_point_in_time(tmp_path: Path) -> None:
    store = SQLiteJourneyStore(tmp_path / "journey.sqlite3")
    original = _packet()
    started = _transition(original, "in_review", T1).value
    assert started is not None

    first = persist_review_packet(original, store, recorded_at=T0)
    assert first.ok and first.revision is not None
    point = StorePoint(first.revision)
    second = persist_review_packet(started, store, recorded_at=T1)
    assert second.ok

    earlier = store.get_job(original.packet_id, as_of=point)
    latest = store.get_job(original.packet_id)
    assert earlier.value is not None and earlier.value.state == "queued"
    assert latest.value is not None and latest.value.state == "in_review"
    assert len(store.list_job_versions(original.packet_id).value) == 2
    store.close()


def test_review_packet_and_transition_schemas_validate() -> None:
    started = _transition(_packet(), "in_review", T1).value
    assert started is not None
    payloads = {
        "clinical_review_packet": started.to_dict(),
        "clinical_review_transition": started.transitions[0].to_dict(),
    }

    for name, payload in payloads.items():
        schema = load_clinical_review_schema(name)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)


def test_packet_rejects_sensitive_extension_keys() -> None:
    payload = _packet().to_dict()
    payload["extensions"] = {"source_text": "SYNTHETIC-PRIVATE-CANARY"}

    try:
        ClinicalReviewPacket.from_dict(payload)
    except ValueError as exc:
        assert "CANARY" not in str(exc)
    else:
        raise AssertionError("sensitive extension was accepted")

    assert "SYNTHETIC-PRIVATE-CANARY" not in json.dumps(_packet().to_dict())
