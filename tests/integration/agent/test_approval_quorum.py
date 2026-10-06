"""Durable synthetic quorum progress across independent local store instances."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from openmed.agent.approvals.quorum import (
    ApprovalQuorumError,
    ApprovalQuorumEvaluator,
    ApprovalQuorumPolicy,
    SQLiteApprovalQuorumStore,
)
from openmed.agent.approvals.tokens import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)

pytestmark = pytest.mark.integration

CLINICIAN = "role:org.example/clinician@1.0.0"
PHARMACIST = "role:org.example/pharmacist@1.0.0"
REQUESTER = "role:org.example/trainee@1.0.0"
ACTION = "sha256:" + "a" * 64
CHANGED = "sha256:" + "b" * 64
SLOT = "sha256:" + "c" * 64


def evaluator(required_count=2):
    return ApprovalQuorumEvaluator(
        [
            ApprovalQuorumPolicy(
                "high-impact-write", required_count, (CLINICIAN, PHARMACIST)
            )
        ]
    )


def verified(role, nonce, action=ACTION, expiry=100):
    key = b"synthetic-local-quorum-key-32-bytes"
    token = ApprovalTokenSigner(key).issue(
        action_digest=action,
        reviewer_role=role,
        expires_at=expiry,
        nonce=f"nonce_{nonce:032x}",
    )
    return ApprovalTokenVerifier(key, InMemoryApprovalNonceStore()).consume(
        token, action_digest=action, reviewer_role=role, now=10
    )


def collect(store, receipts=(), action=ACTION, now=20, slot=SLOT, configured=None):
    return store.collect(
        progress_digest=slot,
        evaluator=configured or evaluator(),
        action_class="high-impact-write",
        action_digest=action,
        requester_role=REQUESTER,
        receipts=receipts,
        now=now,
    )


def test_partial_approvals_survive_restart_and_expire(tmp_path):
    path = tmp_path / "quorum.sqlite"
    first = verified(CLINICIAN, 1)
    assert collect(SQLiteApprovalQuorumStore(path), [first]).approved_count == 1
    restarted = SQLiteApprovalQuorumStore(path)
    assert collect(restarted, [first]).approved_count == 1
    assert collect(restarted, [verified(PHARMACIST, 2)]).satisfied
    assert collect(SQLiteApprovalQuorumStore(path), now=100).approved_count == 0
    with pytest.raises(ApprovalQuorumError, match="clock_rollback"):
        collect(restarted, now=99)


def test_changed_action_resets_and_old_receipts_cannot_return(tmp_path):
    store = SQLiteApprovalQuorumStore(tmp_path / "quorum.sqlite")
    old = [verified(CLINICIAN, 1), verified(PHARMACIST, 2)]
    assert collect(store, old).satisfied
    assert collect(store, old, action=CHANGED).approved_count == 0
    assert collect(store, old).approved_count == 0
    assert collect(store, [verified(CLINICIAN, 3), verified(PHARMACIST, 4)]).satisfied


def test_replay_cannot_move_to_an_independent_action_slot(tmp_path):
    store = SQLiteApprovalQuorumStore(tmp_path / "quorum.sqlite")
    receipt = verified(CLINICIAN, 1)
    assert collect(store, [receipt]).approved_count == 1
    assert collect(store, [receipt], slot="sha256:" + "d" * 64).approved_count == 0


def test_policy_and_requester_changes_reset_progress(tmp_path):
    store = SQLiteApprovalQuorumStore(tmp_path / "quorum.sqlite")
    old = [verified(CLINICIAN, 1), verified(PHARMACIST, 2)]
    assert collect(store, old).satisfied
    assert not collect(store, configured=evaluator(1)).satisfied
    assert collect(store, [verified(CLINICIAN, 3)], configured=evaluator(1)).satisfied
    decision = store.collect(
        progress_digest=SLOT,
        evaluator=evaluator(1),
        action_class="high-impact-write",
        action_digest=ACTION,
        requester_role=PHARMACIST,
        receipts=(),
        now=20,
    )
    assert decision.approved_count == 0


def test_same_role_renewal_is_retained_without_double_counting(tmp_path):
    store = SQLiteApprovalQuorumStore(tmp_path / "quorum.sqlite")
    assert (
        collect(
            store, [verified(CLINICIAN, 1, expiry=30), verified(CLINICIAN, 2)]
        ).approved_count
        == 1
    )
    assert collect(store, [verified(PHARMACIST, 3)], now=30).satisfied


def test_concurrent_independent_connections_preserve_partial_signoffs(tmp_path):
    path = tmp_path / "quorum.sqlite"
    stores = [SQLiteApprovalQuorumStore(path), SQLiteApprovalQuorumStore(path)]
    receipts = [verified(CLINICIAN, 1), verified(PHARMACIST, 2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        decisions = list(
            executor.map(
                lambda pair: collect(pair[0], [pair[1]]), zip(stores, receipts)
            )
        )
    assert sorted(d.approved_count for d in decisions) == [1, 2]
    assert collect(SQLiteApprovalQuorumStore(path)).satisfied


def test_duplicate_and_conflicting_tokens_do_not_create_progress(tmp_path):
    store = SQLiteApprovalQuorumStore(tmp_path / "quorum.sqlite")
    receipt = verified(CLINICIAN, 1)
    assert collect(store, [receipt, receipt]).approved_count == 0
    assert collect(store, [receipt]).approved_count == 0


def test_store_diagnostics_and_rows_are_value_free(tmp_path):
    path = tmp_path / "private-synthetic-path.sqlite"
    store = SQLiteApprovalQuorumStore(path)
    receipt = verified(CLINICIAN, 1)
    collect(store, [receipt])
    assert str(path) not in repr(store)
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT receipt FROM quorum_receipts").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == receipt.to_json()
    assert "nonce_" not in rows[0][0]
    assert "signature" not in rows[0][0]
    with pytest.raises(ApprovalQuorumError, match="store_unavailable") as caught:
        SQLiteApprovalQuorumStore(tmp_path / "private-missing" / "database.sqlite")
    assert "private-missing" not in str(caught.value)


def test_invalid_stored_receipt_fails_closed_and_rolls_back_collection(tmp_path):
    path = tmp_path / "quorum.sqlite"
    store = SQLiteApprovalQuorumStore(path)
    collect(store, [verified(CLINICIAN, 1)])
    with sqlite3.connect(path) as db:
        db.execute("UPDATE quorum_receipts SET receipt = ?", ('{"PHI": "synthetic"}',))
    with pytest.raises(ApprovalQuorumError, match="store_unavailable"):
        collect(store, [verified(PHARMACIST, 2)])
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM quorum_seen").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM quorum_receipts").fetchone()[0] == 1
