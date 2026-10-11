"""Synthetic dispatch/rotation race checks with every network socket forbidden."""

from __future__ import annotations

import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from openmed.interop.fhir import (
    SMARTCredential,
    SMARTCredentialRefresher,
    SMARTRefreshConfig,
    SMARTRefreshLease,
    SMARTTokenResponse,
)
from openmed.interop.smart_scope_audit import smart_scopes_cover

pytestmark = pytest.mark.integration


class AtomicCustody:
    """Test-only custody with opaque handles, quarantine and permanent tombstones."""

    def __init__(self, config):
        self.handle = "cred_" + "3" * 32
        self.binding = config.binding_digest
        self.lock = threading.Lock()
        self.credential = SMARTCredential(
            "access-old", "refresh-old", 110, config.requested_scopes
        )
        self.version = 0
        self.active = None
        self.revoked = False

    def reserve(self, handle):
        with self.lock:
            if self.revoked or self.active is not None or handle != self.handle:
                return None
            self.active = SMARTRefreshLease(
                handle,
                "lease_" + f"{self.version:032x}",
                self.version,
                self.binding,
                self.credential,
            )
            return self.active

    def replace(self, lease, credential):
        with self.lock:
            if (
                self.revoked
                or lease is not self.active
                or lease.generation != self.version
            ):
                return False
            self.credential = credential
            self.version += 1
            self.active = None
            return True

    def release(self, lease):
        with self.lock:
            if self.revoked or lease is not self.active:
                return False
            self.active = None
            self.version += 1
            return True

    def revoke(self, handle):
        with self.lock:
            assert handle == self.handle
            self.revoked = True
            self.credential = None
            self.active = None
            self.version += 1
            return True

    def simulate_approved_dispatch(self, required, now, approved):
        # Counts-only fake effect; production adapters still enforce approved
        # action binding, target policy and credential custody at dispatch.
        with self.lock:
            return bool(
                approved
                and not self.revoked
                and self.active is None
                and self.credential is not None
                and self.credential.expires_at > now
                and smart_scopes_cover(required, self.credential.granted_scopes)
            )


def test_rotation_narrowing_and_simultaneous_refresh_are_offline(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Unexpected network operation")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    config = SMARTRefreshConfig(
        "https://auth.example.test/token",
        "synthetic-client",
        "refresh_token",
        ("system/Observation.ru",),
    )
    store = AtomicCustody(config)
    entered, finish = threading.Event(), threading.Event()
    forms = []

    def transport(request):
        forms.append(dict(request.form))
        assert not store.simulate_approved_dispatch(
            ("system/Observation.r",), 100, True
        )
        entered.set()
        assert finish.wait(timeout=5)
        return SMARTTokenResponse(
            200,
            b'{"access_token":"access-new","refresh_token":"refresh-new","token_type":"Bearer","expires_in":300,"scope":"system/Observation.r"}',
        )

    refresher = SMARTCredentialRefresher(
        config, custody=store, transport=transport, clock=lambda: 100
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            refresher.ensure_ready,
            store.handle,
            required_scopes=("system/Observation.u",),
        )
        assert entered.wait(timeout=5)
        second = pool.submit(refresher.ensure_ready, store.handle)
        assert second.result(timeout=5).code == "credential_unavailable"
        finish.set()
        result = first.result(timeout=5)
    assert result.code == "insufficient_scope" and result.narrowed
    assert forms[0]["refresh_token"] == "refresh-old" and len(forms) == 1
    assert store.credential.refresh_token == "refresh-new"
    assert not store.simulate_approved_dispatch(("system/Observation.u",), 100, True)
    assert not store.simulate_approved_dispatch(("system/Observation.r",), 100, False)
    assert store.simulate_approved_dispatch(("system/Observation.r",), 100, True)
    evidence = json.dumps(result.to_dict())
    for secret in (
        "refresh-old",
        "refresh-new",
        "access-new",
        config.client_id,
        config.token_endpoint,
    ):
        assert secret not in evidence


def test_revoke_during_exchange_prevents_late_commit(monkeypatch):
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: pytest.fail("Unexpected network operation"),
    )
    config = SMARTRefreshConfig(
        "https://auth.example.test/token",
        "synthetic-client",
        "refresh_token",
        ("system/Observation.r",),
    )
    store = AtomicCustody(config)

    def transport(_):
        assert store.revoke(store.handle)
        return SMARTTokenResponse(
            200,
            b'{"access_token":"access-new","refresh_token":"refresh-new","token_type":"Bearer","expires_in":300,"scope":"system/Observation.r"}',
        )

    refresher = SMARTCredentialRefresher(
        config, custody=store, transport=transport, clock=lambda: 100
    )
    result = refresher.ensure_ready(store.handle)
    assert result.code == "custody_unavailable" and result.revocation_confirmed
    assert store.credential is None and store.reserve(store.handle) is None
    assert not store.simulate_approved_dispatch(("system/Observation.r",), 100, True)
