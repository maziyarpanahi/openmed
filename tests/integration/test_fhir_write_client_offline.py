"""Fake FHIR wire checks with a durable SQLite attempt ledger; no live EHR."""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from openmed.agent.approvals.tokens import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.interop.fhir.write_client import (
    FHIRHTTPResponse,
    FHIRWriteOutcome,
    FHIRWriteStatus,
)
from tests.unit.interop.fhir.test_write_client import (
    AUDIENCE,
    HANDLE,
    KEY,
    NOW,
    ROLE,
    SECRET,
    _prepared,
    _receipt,
    _setup,
    _transaction,
)

pytestmark = pytest.mark.integration


class _DurableLedger:
    """Test-only SQLite receipt store proving restart and atomic reservation."""

    def __init__(self, path):
        self.path = path
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS attempts (key TEXT PRIMARY KEY, action TEXT NOT NULL, outcome TEXT)"
            )

    def claim(self, key, action):
        with sqlite3.connect(self.path) as connection:
            return (
                connection.execute(
                    "INSERT OR IGNORE INTO attempts (key, action) VALUES (?, ?)",
                    (key, action),
                ).rowcount
                == 1
            )

    def lookup(self, key):
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT outcome FROM attempts WHERE key = ?", (key,)
            ).fetchone()
        if row is None or row[0] is None:
            return None
        value = json.loads(row[0])
        return FHIRWriteOutcome(
            FHIRWriteStatus(value["status"]),
            value["reason_code"],
            value["action_digest"],
            value["payload_digest"],
            value["resource_count"],
            value["response_digest"],
        )

    def finish(self, key, action, outcome):
        with sqlite3.connect(self.path) as connection:
            return (
                connection.execute(
                    "UPDATE attempts SET outcome = ? WHERE key = ? AND action = ? AND outcome IS NULL",
                    (json.dumps(outcome.to_dict()), key, action),
                ).rowcount
                == 1
            )


class _FakeFHIR:
    """In-process fake validating approved wire bytes before applying a write."""

    def __init__(self, prepared, kind):
        self.prepared, self.kind = prepared, kind
        self.calls = 0
        self.applied = 0
        self.lose_ack = False
        self.lock = Lock()

    def __call__(self, request):
        with self.lock:
            self.calls += 1
            expected_digest = (
                "sha256:"
                + hmac.new(
                    SECRET,
                    b"openmed.fhir.write-client.v1\0payload\0" + request.body,
                    hashlib.sha256,
                ).hexdigest()
            )
            assert expected_digest == self.prepared.payload_digest
            assert request.body == self.prepared._body
            assert request.retries == 0 and request.follow_redirects is False
            headers = dict(request.headers)
            assert headers["Idempotency-Key"] == KEY
            assert headers["Authorization"] == "Bearer SyntheticPrivateToken"
            assert headers["Content-Type"] == "application/fhir+json"
            if self.kind == "transaction":
                assert request.method == "POST" and request.url == AUDIENCE
                response = {
                    "resourceType": "Bundle",
                    "type": "transaction-response",
                    "entry": [
                        {
                            "response": {
                                "status": "200 OK",
                                "location": "Observation/synthetic-1/_history/2",
                                "etag": 'W/"2"',
                            }
                        },
                        {
                            "response": {
                                "status": "201 Created",
                                "location": "Provenance/synthetic-p/_history/1",
                                "etag": 'W/"1"',
                            }
                        },
                    ],
                }
            else:
                predicate = "identifier=urn%3Asynthetic%7C123"
                assert request.method == ("POST" if self.kind == "create" else "PUT")
                assert request.url == AUDIENCE + "/Observation" + (
                    "?" + predicate if self.kind == "update" else ""
                )
                if self.kind == "create":
                    assert headers["If-None-Exist"] == predicate
                else:
                    assert headers["If-Match"] == 'W/"1"'
                response = {
                    "resourceType": "Observation",
                    "id": "synthetic-1",
                    "meta": {"versionId": "2"},
                }
            self.applied += 1
            if self.lose_ack:
                raise TimeoutError("synthetic-private-server-error")
            return FHIRHTTPResponse(
                200 if self.kind == "transaction" else 201,
                json.dumps(response).encode(),
                (("Content-Type", "application/fhir+json"),),
            )


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("live network forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def _signed_receipt(prepared):
    signer = ApprovalTokenSigner(b"synthetic-local-human-approval-key")
    token = signer.issue(
        action_digest=prepared.action_digest,
        reviewer_role=ROLE,
        expires_at=int(NOW.timestamp()) + 60,
    )
    verifier = ApprovalTokenVerifier(
        b"synthetic-local-human-approval-key",
        InMemoryApprovalNonceStore(),
        clock=lambda: int(NOW.timestamp()),
    )
    return verifier.consume(
        token, action_digest=prepared.action_digest, reviewer_role=ROLE
    )


@pytest.mark.parametrize("kind", ["create", "update", "transaction"])
def test_fake_server_exact_approved_wire_with_real_consumed_receipt(tmp_path, kind):
    env = _setup(ledger=_DurableLedger(tmp_path / "attempts.db"))
    if kind == "transaction":
        proposed = env.client.prepare_transaction(
            _transaction(),
            idempotency_key=KEY,
            credential_handle=HANDLE,
            lineage=env.manifest,
        )
        env.broker.dispatch = lambda h, *, audience, required_scopes: env.broker.sender(
            audience, "Bearer SyntheticPrivateToken"
        )
    else:
        proposed = _prepared(env, kind)
    receipt = _signed_receipt(proposed)
    env.client._authorize = lambda p, r: r == receipt
    server = _FakeFHIR(proposed, kind)
    env.client._transport = server
    outcome = env.client.submit(proposed, receipt)
    assert outcome.status is FHIRWriteStatus.COMMITTED
    assert env.client.submit(proposed, receipt) == outcome
    assert env.client.reconcile(proposed) == outcome
    assert server.calls == server.applied == 1


def test_commit_then_timeout_stays_unknown_across_reopened_ledger(tmp_path):
    path = tmp_path / "attempts.db"
    env = _setup(ledger=_DurableLedger(path))
    proposed = _prepared(env)
    server = _FakeFHIR(proposed, "create")
    server.lose_ack = True
    env.client._transport = server
    result = env.client.submit(proposed, _signed_receipt(proposed))
    assert result.status is FHIRWriteStatus.UNKNOWN and server.applied == 1
    restarted = _setup(ledger=_DurableLedger(path), transport=server)
    same = _prepared(restarted)
    assert same.action_digest == proposed.action_digest
    assert restarted.client.submit(same, _receipt(same)) == result
    assert restarted.client.reconcile(same) == result
    assert server.calls == 1


def test_concurrent_clients_cannot_dispatch_the_same_key_twice(tmp_path):
    path = tmp_path / "attempts.db"
    first = _setup(ledger=_DurableLedger(path))
    proposed = _prepared(first)
    server = _FakeFHIR(proposed, "create")
    clients = [_setup(ledger=_DurableLedger(path), transport=server) for _ in range(12)]

    def run(env):
        prepared = _prepared(env)
        return env.client.submit(prepared, _receipt(prepared))

    with ThreadPoolExecutor(max_workers=12) as executor:
        outcomes = list(executor.map(run, clients))
    assert any(o.status is FHIRWriteStatus.COMMITTED for o in outcomes)
    assert all(
        o.status in {FHIRWriteStatus.COMMITTED, FHIRWriteStatus.UNKNOWN}
        for o in outcomes
    )
    assert server.calls == server.applied == 1
    assert all(
        env.client.reconcile(_prepared(env)).status is FHIRWriteStatus.COMMITTED
        for env in clients
    )


def test_reserved_crash_without_outcome_never_replays_after_reopen(tmp_path):
    path = tmp_path / "attempts.db"
    first = _setup(ledger=_DurableLedger(path))
    proposed = _prepared(first)
    assert first.ledger.claim(KEY, proposed.action_digest)
    restarted = _setup(ledger=_DurableLedger(path))
    same = _prepared(restarted)
    assert (
        restarted.client.submit(same, _receipt(same)).reason_code
        == "attempt_in_progress"
    )
    assert restarted.client.reconcile(same).status is FHIRWriteStatus.UNKNOWN
    assert not restarted.requests
