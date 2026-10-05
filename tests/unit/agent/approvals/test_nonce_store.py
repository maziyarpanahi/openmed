"""Offline safety and privacy checks for durable nonce claims."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat

import pytest

from openmed.agent.approvals import (
    ApprovalNonceStoreError,
    ApprovalReplayError,
    ApprovalTokenSigner,
    ApprovalTokenValidationError,
    ApprovalTokenVerifier,
    SQLiteApprovalNonceStore,
    dispatch_with_approval_token,
)

KEY = b"synthetic-local-approval-key-32-bytes"
ACTION = "sha256:" + "a" * 64
ROLE = "role:org.example/clinical-reviewer@1.0.0"
DIGEST = "sha256:" + "b" * 64


def test_reopen_replay_and_database_canary(tmp_path):
    path = tmp_path / "private-canary.db"
    store = SQLiteApprovalNonceStore(path)
    token = ApprovalTokenSigner(KEY).issue(
        action_digest=ACTION,
        reviewer_role=ROLE,
        expires_at=200,
        nonce="nonce_" + "1" * 32,
    )
    ApprovalTokenVerifier(KEY, store).consume(
        token,
        action_digest=ACTION,
        reviewer_role=ROLE,
        now=100,
    )
    reopened = ApprovalTokenVerifier(KEY, SQLiteApprovalNonceStore(path))
    with pytest.raises(ApprovalReplayError):
        reopened.consume(token, action_digest=ACTION, reviewer_role=ROLE, now=100)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM nonce_claims").fetchall() == [
            ("sha256:" + hashlib.sha256(token.nonce.encode()).hexdigest(), 200)
        ]
        assert connection.execute(
            "SELECT typeof(nonce_digest), typeof(expires_at) FROM nonce_claims"
        ).fetchall() == [("text", "integer")]
        assert [
            row[1] for row in connection.execute("PRAGMA table_info(nonce_claims)")
        ] == ["nonce_digest", "expires_at"]
    database = path.read_bytes()
    for canary in (
        token.to_json(),
        token.nonce,
        token.signature,
        ACTION,
        ROLE,
        KEY.decode(),
    ):
        assert canary.encode() not in database
    assert str(path) not in repr(store)
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_expiry_purge_preserves_unexpired_claims(tmp_path):
    path = tmp_path / "claims.db"
    store = SQLiteApprovalNonceStore(path)
    short = "sha256:" + "c" * 64
    assert store.claim(short, expires_at=101, now=100)
    assert store.claim(DIGEST, expires_at=200, now=100)
    assert not store.claim(short, expires_at=101, now=101)
    assert not store.claim(DIGEST, expires_at=200, now=101)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM nonce_claims").fetchall() == [
            (DIGEST, 200)
        ]
    assert store.claim(short, expires_at=201, now=101)


@pytest.mark.parametrize(
    "damage", ["truncated", "empty", "future", "missing", "schema"]
)
def test_damaged_store_refuses_dispatch_and_reopening(tmp_path, damage):
    path = tmp_path / "private-canary.db"
    store = SQLiteApprovalNonceStore(path)
    assert store.claim(DIGEST, expires_at=200, now=100)
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:100])
    elif damage == "empty":
        path.write_bytes(b"")
    elif damage == "missing":
        path.unlink()
    else:
        with sqlite3.connect(path) as connection:
            connection.execute(
                "PRAGMA user_version = 99"
                if damage == "future"
                else "DROP TABLE nonce_claims"
            )
    token = ApprovalTokenSigner(KEY).issue(
        action_digest=ACTION,
        reviewer_role=ROLE,
        expires_at=200,
    )
    calls = []
    with pytest.raises(ApprovalNonceStoreError) as error:
        dispatch_with_approval_token(
            token,
            action_digest=ACTION,
            reviewer_role=ROLE,
            verifier=ApprovalTokenVerifier(KEY, store),
            dispatch=lambda: calls.append(True),
            now=100,
        )
    assert calls == []
    assert "private-canary" not in str(error.value)
    if damage != "missing":
        with pytest.raises(ApprovalNonceStoreError):
            SQLiteApprovalNonceStore(path)


def test_lock_timeout_fails_closed_without_claiming(tmp_path):
    path = tmp_path / "claims.db"
    store = SQLiteApprovalNonceStore(path, timeout=0)
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        with pytest.raises(ApprovalNonceStoreError):
            store.claim(DIGEST, expires_at=200, now=100)
        with pytest.raises(ApprovalNonceStoreError):
            SQLiteApprovalNonceStore(path, timeout=0)
    finally:
        connection.close()
    assert store.claim(DIGEST, expires_at=200, now=100)


def test_failed_commit_rolls_back_and_never_authorizes(tmp_path, monkeypatch):
    path = tmp_path / "claims.db"
    store = SQLiteApprovalNonceStore(path)
    connect = store._connect

    class FailedCommit:
        def __init__(self):
            self.connection = connect()

        def execute(self, *args):
            return self.connection.execute(*args)

        def commit(self):
            raise sqlite3.OperationalError("synthetic private storage failure")

        def close(self):
            self.connection.close()

    monkeypatch.setattr(store, "_connect", FailedCommit)
    with pytest.raises(ApprovalNonceStoreError):
        store.claim(DIGEST, expires_at=200, now=100)
    assert SQLiteApprovalNonceStore(path).claim(DIGEST, expires_at=200, now=100)


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan"), True, "private"])
def test_invalid_timeout_is_value_free(tmp_path, timeout):
    with pytest.raises(ApprovalNonceStoreError) as error:
        SQLiteApprovalNonceStore(tmp_path / "claims.db", timeout=timeout)
    assert error.value.code == "invalid_nonce_store_timeout"


@pytest.mark.parametrize(
    "digest,expiry,now",
    [
        ("private raw nonce", 200, 100),
        (DIGEST, 200.5, 100),
        (DIGEST, 200, True),
        (DIGEST, 2**63, 100),
    ],
)
def test_invalid_claim_inputs_are_not_persisted(tmp_path, digest, expiry, now):
    path = tmp_path / "claims.db"
    store = SQLiteApprovalNonceStore(path)
    with pytest.raises(ApprovalTokenValidationError):
        store.claim(digest, expires_at=expiry, now=now)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM nonce_claims").fetchall() == []


def test_symlink_store_is_refused(tmp_path):
    if os.name != "posix":
        pytest.skip("symlink creation requires platform-specific privileges")
    path = tmp_path / "claims.db"
    SQLiteApprovalNonceStore(path)
    link = tmp_path / "link.db"
    link.symlink_to(path)
    with pytest.raises(ApprovalNonceStoreError):
        SQLiteApprovalNonceStore(link)
