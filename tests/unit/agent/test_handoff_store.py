"""Synthetic offline concurrency, expiry and privacy controls for handoff storage."""

import hashlib
import json
import sqlite3
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone

import pytest

from openmed.agent import ReviewerHandoffPacket
from openmed.agent.handoff_store import (
    DecisionReceipt,
    HandoffRevision,
    HandoffStoreError,
    SQLiteHandoffStore,
)
from openmed.clinical.review_state_machine import ReviewState
from openmed.structured.store.protocols import StoreState

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
ACTION = "act_" + "1" * 32
ROLE = "role_" + "2" * 32
REVIEWER = "rev_" + "3" * 32
DIGEST = "4" * 64
SENSITIVE = "Synthetic Patient /private/records token=synthetic-secret"


def packet(*, evidence_digest="5" * 64, expiry=3600, issued=-60):
    return ReviewerHandoffPacket.from_dict(
        {
            "run_id": "run_" + "6" * 32,
            "workflow_id": "workflow:org.example/review@1.0.0",
            "reason_code": "conflicting_evidence",
            "requested_decision": "resolve_evidence_conflict",
            "evidence_references": [
                {
                    "artifact_id": "art_" + "7" * 32,
                    "kind": "evidence",
                    "schema_id": "example.evidence.v1",
                    "sha256": evidence_digest,
                    "byte_size": 12,
                }
            ],
            "issued_at": (NOW + timedelta(seconds=issued)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "expires_at": (NOW + timedelta(seconds=expiry)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
        now=NOW,
    )


@pytest.fixture
def clock():
    return [NOW]


@pytest.fixture
def store(tmp_path, clock):
    value = SQLiteHandoffStore(tmp_path / "handoffs.db", clock=lambda: clock[0])
    yield value
    value.close()


def publish(store, *, expected_revision=0, action_digest=DIGEST, **kwargs):
    return store.publish(
        ACTION, action_digest, packet(**kwargs), expected_revision=expected_revision
    )


def acquire(store, *, revision=1, seconds=300, reviewer=REVIEWER):
    return store.acquire(
        ACTION,
        expected_revision=revision,
        reviewer_role_ref=ROLE,
        reviewer_ref=reviewer,
        seconds=seconds,
    )


def test_binding_commits_exact_packet_and_evidence(store):
    source = packet()
    binding = publish(store).value
    assert binding.action_digest == DIGEST
    assert (
        binding.handoff_digest == hashlib.sha256(source.to_json().encode()).hexdigest()
    )
    evidence = json.dumps(
        [ref.to_dict() for ref in source.evidence_references],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert binding.evidence_digest == hashlib.sha256(evidence.encode()).hexdigest()
    assert store.current(ACTION).value == binding


def test_one_decision_for_two_reviewers_and_no_idempotent_approval_reuse(store):
    publish(store)
    first = acquire(store).value
    second = acquire(store, reviewer="rev_" + "8" * 32).value
    assert first.lease_id != second.lease_id
    accepted = store.decide(first, ReviewState.APPROVED)
    assert accepted.ok and accepted.created
    assert accepted.value.authorizes_clinical_action is False
    for lease in (first, second):
        conflict = store.decide(lease, ReviewState.REJECTED)
        assert conflict.state is StoreState.CONFLICT
        assert conflict.code == "already_decided" and conflict.value is None
    assert acquire(store).code == "already_decided"
    assert store.receipts(ACTION) == (accepted.value,)


@pytest.mark.parametrize("change", ["action", "evidence", "packet", "identical"])
def test_correction_invalidates_pending_and_accepted_reviews(store, change):
    original = publish(store).value
    accepted_lease = acquire(store).value
    stale = acquire(store, reviewer="rev_" + "8" * 32).value
    receipt = store.decide(accepted_lease, ReviewState.APPROVED).value
    updates = {"expected_revision": 1}
    if change == "action":
        updates["action_digest"] = "9" * 64
    elif change == "evidence":
        updates["evidence_digest"] = "9" * 64
    elif change == "packet":
        updates["expiry"] = 1800
    corrected = publish(store, **updates).value
    assert corrected.revision == 2
    if change == "action":
        assert corrected.action_digest != original.action_digest
    elif change == "evidence":
        assert corrected.evidence_digest != original.evidence_digest
        assert corrected.handoff_digest != original.handoff_digest
    elif change == "packet":
        assert corrected.handoff_digest != original.handoff_digest
    assert store.decide(stale, ReviewState.APPROVED).code == "superseded"
    assert store.decide(accepted_lease, ReviewState.APPROVED).code == "superseded"
    assert acquire(store).code == "superseded"
    assert store.receipts(ACTION) == (receipt,)
    new_lease = acquire(store, revision=2).value
    assert store.decide(new_lease, ReviewState.REJECTED).ok
    assert [r.decision for r in store.receipts(ACTION)] == [
        ReviewState.APPROVED,
        ReviewState.REJECTED,
    ]


def test_competing_corrections_do_not_overwrite(store):
    publish(store)
    assert publish(store).code == "revision_conflict"
    assert publish(store, expected_revision=1, action_digest="9" * 64).ok
    assert publish(store, expected_revision=1).code == "revision_conflict"
    assert store.current(ACTION).value.action_digest == "9" * 64


@pytest.mark.parametrize("delta", [300, 301])
def test_expired_lease_is_refused_at_boundary_and_can_be_replaced(store, clock, delta):
    publish(store)
    expired = acquire(store).value
    clock[0] = NOW + timedelta(seconds=delta)
    assert store.decide(expired, ReviewState.APPROVED).code == "expired"
    assert store.receipts(ACTION) == ()
    fresh = acquire(store).value
    assert fresh.lease_id != expired.lease_id
    assert store.decide(fresh, ReviewState.REJECTED).ok


def test_lease_capped_by_packet_and_packet_expiry_blocks_every_path(store, clock):
    publish(store, expiry=60)
    lease = acquire(store).value
    assert lease.expires_at == (NOW + timedelta(seconds=60)).timestamp()
    clock[0] += timedelta(seconds=60)
    assert store.current(ACTION).code == "expired"
    assert acquire(store).code == "expired"
    assert store.decide(lease, ReviewState.APPROVED).code == "expired"
    assert (
        store.publish(ACTION, DIGEST, packet(expiry=60), expected_revision=1).code
        == "expired"
    )
    assert store.receipts(ACTION) == ()


def test_expired_accepted_receipt_remains_history_without_approval_reuse(store, clock):
    publish(store, expiry=60)
    lease = acquire(store).value
    receipt = store.decide(lease, ReviewState.APPROVED).value
    clock[0] += timedelta(seconds=60)
    assert store.receipts(ACTION) == (receipt,)
    assert store.current(ACTION).code == "expired"
    assert store.decide(lease, ReviewState.APPROVED).code == "expired"


def test_accepted_decision_does_not_extend_lease_lifetime(store, clock):
    publish(store)
    lease = acquire(store).value
    receipt = store.decide(lease, ReviewState.APPROVED).value
    clock[0] += timedelta(seconds=300)
    assert store.current(ACTION).ok  # Packet is still live, but this lease is not.
    assert store.decide(lease, ReviewState.APPROVED).code == "expired"
    assert store.receipts(ACTION) == (receipt,)
    assert acquire(store).code == "already_decided"


@pytest.mark.parametrize(
    "field", ["action_digest", "handoff_digest", "evidence_digest", "revision"]
)
def test_lease_is_bound_to_exact_revision_and_all_digests(store, field):
    publish(store)
    lease = acquire(store).value
    binding = replace(lease.binding, **{field: 2 if field == "revision" else "f" * 64})
    assert (
        store.decide(replace(lease, binding=binding), ReviewState.APPROVED).code
        == "invalid_lease"
    )
    assert store.receipts(ACTION) == ()
    assert store.decide(lease, ReviewState.APPROVED).ok


@pytest.mark.parametrize(
    "updates",
    [
        {"lease_id": "lease_" + "f" * 32},
        {"reviewer_ref": "rev_" + "f" * 32},
        {"reviewer_role_ref": "role_" + "f" * 32},
        {"expires_at": NOW.timestamp() + 301},
    ],
)
def test_forged_lease_fields_cannot_accept_decision(store, updates):
    publish(store)
    lease = acquire(store).value
    assert (
        store.decide(replace(lease, **updates), ReviewState.APPROVED).code
        == "invalid_lease"
    )
    assert store.receipts(ACTION) == ()


def test_restart_preserves_decisions_leases_and_clock_watermark(tmp_path, clock):
    path = tmp_path / "handoffs.db"
    first = SQLiteHandoffStore(path, clock=lambda: clock[0])
    publish(first)
    lease = acquire(first).value
    receipt = first.decide(lease, ReviewState.REJECTED).value
    first.close()
    second = SQLiteHandoffStore(path, clock=lambda: clock[0])
    try:
        assert second.receipts(ACTION) == (receipt,)
        assert second.decide(lease, ReviewState.APPROVED).code == "already_decided"
        clock[0] -= timedelta(seconds=1)
        with pytest.raises(HandoffStoreError, match="clock_regressed"):
            second.decide(lease, ReviewState.APPROVED)
    finally:
        second.close()


def test_clock_rollback_cannot_resurrect_expired_lease(store, clock):
    publish(store)
    lease = acquire(store).value
    clock[0] += timedelta(seconds=300)
    assert store.decide(lease, ReviewState.APPROVED).code == "expired"
    clock[0] = NOW
    with pytest.raises(HandoffStoreError, match="clock_regressed"):
        store.decide(lease, ReviewState.APPROVED)
    clock[0] = NOW + timedelta(seconds=300)
    assert store.receipts(ACTION) == ()


@pytest.mark.parametrize("seconds", [True, 0, -1, 901, 86401, float("nan"), SENSITIVE])
def test_lease_lifetime_is_bounded_and_errors_are_value_free(store, seconds):
    publish(store)
    with pytest.raises(HandoffStoreError) as error:
        acquire(store, seconds=seconds)
    assert SENSITIVE not in str(error.value)
    assert store.receipts(ACTION) == ()


@pytest.mark.parametrize(
    "argument", ["action_id", "action_digest", "reviewer_ref", "reviewer_role_ref"]
)
def test_reject_source_content_before_persistence(store, tmp_path, argument):
    publish(store)
    with pytest.raises(HandoffStoreError) as error:
        if argument in ("action_id", "action_digest"):
            kwargs = {"action_id": ACTION, "action_digest": DIGEST, argument: SENSITIVE}
            store.publish(**kwargs, packet=packet(), expected_revision=1)
        else:
            kwargs = {
                "reviewer_ref": REVIEWER,
                "reviewer_role_ref": ROLE,
                argument: SENSITIVE,
            }
            store.acquire(ACTION, expected_revision=1, **kwargs)
    assert SENSITIVE not in "".join(traceback.format_exception(error.value))
    assert SENSITIVE.encode() not in (tmp_path / "handoffs.db").read_bytes()


@pytest.mark.parametrize(
    "decision",
    [ReviewState.IN_REVIEW, ReviewState.EXPIRED, "approved", SENSITIVE, None],
)
def test_only_terminal_controlled_review_states_are_accepted(store, decision):
    publish(store)
    lease = acquire(store).value
    with pytest.raises(HandoffStoreError, match="invalid_decision"):
        store.decide(lease, decision)
    assert store.receipts(ACTION) == ()


def test_missing_action_and_future_packet_are_explicit(store):
    assert store.current(ACTION).code == "not_found"
    assert acquire(store).code == "not_found"
    assert publish(store, issued=60).code == "not_yet_valid"
    assert store.current(ACTION).code == "not_found"


@pytest.mark.parametrize("revision", [True, -1, 1.0, SENSITIVE])
def test_invalid_optimistic_revision(store, revision):
    with pytest.raises(HandoffStoreError, match="invalid_revision"):
        publish(store, expected_revision=revision)


def test_sql_receipts_are_append_only_and_contain_metadata_only(store, tmp_path):
    publish(store)
    lease = acquire(store).value
    receipt = store.decide(lease, ReviewState.APPROVED).value
    metadata = json.dumps(asdict(receipt))
    assert SENSITIVE not in metadata
    assert "evidence_references" not in metadata
    assert receipt.lease.reviewer_role_ref == ROLE
    with sqlite3.connect(tmp_path / "handoffs.db") as db:
        for statement in (
            "DELETE FROM handoff_receipts",
            "UPDATE handoff_receipts SET decision='rejected'",
            "DELETE FROM handoff_leases",
            "DELETE FROM handoff_revisions",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append_only"):
                db.execute(statement)
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert store.receipts(ACTION) == (receipt,)


def test_storage_errors_hide_paths_and_closed_connections(tmp_path):
    with pytest.raises(HandoffStoreError) as error:
        SQLiteHandoffStore(tmp_path / "missing" / SENSITIVE)
    assert str(tmp_path) not in str(error.value) and SENSITIVE not in str(error.value)
    store = SQLiteHandoffStore(":memory:", clock=lambda: NOW)
    store.close()
    with pytest.raises(HandoffStoreError, match="store_closed"):
        store.current(ACTION)


def test_unknown_schema_refused_without_resetting_history(store, tmp_path, clock):
    publish(store)
    receipt = store.decide(acquire(store).value, ReviewState.APPROVED).value
    with sqlite3.connect(tmp_path / "handoffs.db") as db:
        db.execute("UPDATE handoff_meta SET version=999")
    with pytest.raises(HandoffStoreError, match="store_unavailable_or_incompatible"):
        SQLiteHandoffStore(tmp_path / "handoffs.db", clock=lambda: clock[0])
    assert store.receipts(ACTION) == (receipt,)


def test_public_receipt_types_reject_free_text_and_invalid_timestamps():
    with pytest.raises(HandoffStoreError):
        HandoffRevision(SENSITIVE, 1, DIGEST, DIGEST, DIGEST, NOW.timestamp())
    binding = HandoffRevision(ACTION, 1, DIGEST, DIGEST, DIGEST, NOW.timestamp())
    for value in (True, float("nan"), float("inf"), SENSITIVE):
        with pytest.raises(HandoffStoreError):
            replace(binding, expires_at=value)
    with pytest.raises(HandoffStoreError):
        DecisionReceipt(SENSITIVE, ReviewState.APPROVED, NOW.timestamp())


def test_clock_provider_failures_are_sanitized_and_rolled_back(store):
    publish(store)

    def fail():
        raise RuntimeError(SENSITIVE)

    store._clock = fail
    with pytest.raises(HandoffStoreError) as error:
        store.current(ACTION)
    assert SENSITIVE not in "".join(traceback.format_exception(error.value))
    store._clock = lambda: NOW
    assert acquire(store).ok
