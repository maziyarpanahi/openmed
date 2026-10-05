"""Synthetic spawn-process tests; no models, services, or network required."""

from __future__ import annotations

import multiprocessing

import pytest

from openmed.agent.approvals import (
    ApprovalReplayError,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    SQLiteApprovalNonceStore,
)

KEY = b"synthetic-local-approval-key-32-bytes"
ACTION = "sha256:" + "a" * 64
ROLE = "role:org.example/clinical-reviewer@1.0.0"


def _consume(path, token, barrier, results):
    try:
        verifier = ApprovalTokenVerifier(KEY, SQLiteApprovalNonceStore(path))
        barrier.wait(timeout=60)
        verifier.consume(token, action_digest=ACTION, reviewer_role=ROLE, now=100)
    except ApprovalReplayError:
        results.put("replayed")
    except Exception:
        results.put("failed")
    else:
        results.put("claimed")


def _run_consumers(path, token, count):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(count)
    results = context.Queue()
    processes = [
        context.Process(target=_consume, args=(path, token, barrier, results))
        for _ in range(count)
    ]
    try:
        for process in processes:
            process.start()
        outcomes = [results.get(timeout=90) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        return outcomes
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()
        results.join_thread()


@pytest.mark.integration
def test_cross_process_claim_has_exactly_one_winner_and_survives_restart(tmp_path):
    path = tmp_path / "claims.db"
    SQLiteApprovalNonceStore(path)
    token = (
        ApprovalTokenSigner(KEY)
        .issue(
            action_digest=ACTION,
            reviewer_role=ROLE,
            expires_at=200,
            nonce="nonce_" + "1" * 32,
        )
        .to_json()
    )
    outcomes = _run_consumers(path, token, 4)
    assert outcomes.count("claimed") == 1
    assert outcomes.count("replayed") == 3
    assert _run_consumers(path, token, 1) == ["replayed"]
