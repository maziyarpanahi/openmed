"""Real offline process races against one synthetic handoff database."""

import multiprocessing
from datetime import datetime, timezone

import pytest

from openmed.agent import ReviewerHandoffPacket
from openmed.agent.handoff_store import SQLiteHandoffStore
from openmed.clinical.review_state_machine import ReviewState

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
ACTION = "act_" + "1" * 32


def _race(path, lease, barrier, queue):
    store = SQLiteHandoffStore(path, clock=lambda: NOW)
    try:
        barrier.wait(timeout=30)
        outcome = store.decide(lease, ReviewState.APPROVED)
        queue.put((outcome.state.value, outcome.code))
    finally:
        store.close()


@pytest.mark.integration
def test_two_process_reviewers_race_then_restart_and_correct(tmp_path):
    path = tmp_path / "handoffs.db"
    store = SQLiteHandoffStore(path, clock=lambda: NOW)
    packet = ReviewerHandoffPacket.from_dict(
        {
            "run_id": "run_" + "2" * 32,
            "workflow_id": "workflow:org.example/review@1.0.0",
            "reason_code": "conflicting_evidence",
            "requested_decision": "resolve_evidence_conflict",
            "evidence_references": [],
            "issued_at": "2026-10-06T11:59:00Z",
            "expires_at": "2026-10-06T13:00:00Z",
        },
        now=NOW,
    )
    assert store.publish(ACTION, "3" * 64, packet, expected_revision=0).ok
    leases = [
        store.acquire(
            ACTION,
            expected_revision=1,
            reviewer_role_ref="role_" + f"{index:032x}",
            reviewer_ref="rev_" + f"{index:032x}",
        ).value
        for index in (1, 2)
    ]
    store.close()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    processes = [
        context.Process(target=_race, args=(path, lease, barrier, queue))
        for lease in leases
    ]
    try:
        for process in processes:
            process.start()
        outcomes = [queue.get(timeout=45) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert outcomes.count(("success", None)) == 1
        assert outcomes.count(("conflict", "already_decided")) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        queue.close()
        queue.join_thread()
    restarted = SQLiteHandoffStore(path, clock=lambda: NOW)
    try:
        receipts = restarted.receipts(ACTION)
        assert len(receipts) == 1
        assert receipts[0].lease in leases
        assert receipts[0].authorizes_clinical_action is False
        assert restarted.publish(ACTION, "4" * 64, packet, expected_revision=1).ok
        assert all(
            restarted.decide(lease, ReviewState.APPROVED).code == "superseded"
            for lease in leases
        )
        assert restarted.receipts(ACTION) == receipts
        fresh = restarted.acquire(
            ACTION,
            expected_revision=2,
            reviewer_role_ref="role_" + "5" * 32,
            reviewer_ref="rev_" + "6" * 32,
        ).value
        assert restarted.decide(fresh, ReviewState.REJECTED).ok
    finally:
        restarted.close()
