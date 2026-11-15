"""Idempotency-Key replay for FHIR Bulk Data and SMART ingestion starts."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import hash_api_key
from openmed.service.bulk_data import FHIRBulkJobManager
from openmed.service.idempotency import (
    CREDENTIAL_REQUEST_FIELDS,
    DEFAULT_IDEMPOTENCY_MAX_KEYS,
    DEFAULT_IDEMPOTENCY_TTL_SECONDS,
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENCY_MAX_KEYS_ENV_VAR,
    IDEMPOTENCY_REPLAYED_HEADER,
    IDEMPOTENCY_TTL_ENV_VAR,
    PENDING_LEASE_SECONDS,
    IdempotencyError,
    IdempotencyReservation,
    IdempotentReplay,
    IdempotentStartRequest,
    StartIdempotencyStore,
    parse_idempotency_key,
    request_digest_material,
)
from openmed.service.job_ownership import BULK_ADMIN_SCOPE
from openmed.service.keyed_digest import (
    SERVICE_DIGEST_KEY_ENV_VAR,
    SERVICE_DIGEST_KEY_ID_ENV_VAR,
    ServiceDigestKey,
)
from openmed.service.smart_backend import SMARTBackendJobManager

LOOPBACK_BASE_URL = "http://127.0.0.1"
SYNTHETIC_DIGEST_KEY = "synthetic-digest-key-" + "c" * 40
SYNTHETIC_IDEMPOTENCY_KEY = "synthetic-retry-key-0001"
SYNTHETIC_PRIVATE_KEY = "synthetic-private-key-secret"
SMART_BASE = "/fhir/smart-backend/ingestions"
_BULK_SCOPES = ["bulk:read", "bulk:write"]
_PRINCIPALS = {
    "owner": ("synthetic-owner-key", "synthetic-owner", _BULK_SCOPES),
    "owner-rotated": ("synthetic-owner-key-2", "synthetic-owner", _BULK_SCOPES),
    "other": ("synthetic-other-key", "synthetic-other", _BULK_SCOPES),
    "admin": (
        "synthetic-admin-key",
        "synthetic-admin",
        [*_BULK_SCOPES, BULK_ADMIN_SCOPE],
    ),
}
_SERVICE_ENV_VARS = (
    "OPENMED_PROFILE",
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_MTLS_ENABLED",
    "OPENMED_SERVICE_BULK_INPUT_ROOTS",
    "OPENMED_SERVICE_BULK_OUTPUT_ROOT",
    IDEMPOTENCY_TTL_ENV_VAR,
    IDEMPOTENCY_MAX_KEYS_ENV_VAR,
    SERVICE_DIGEST_KEY_ENV_VAR,
    SERVICE_DIGEST_KEY_ID_ENV_VAR,
)


class FakeLoader:
    """Minimal model loader double; no model weights are loaded."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@dataclass
class _FakeResult:
    deidentified_text: str
    pii_entities: tuple[()] = ()


def _fake_deidentify(text: str, **_: Any) -> _FakeResult:
    return _FakeResult(deidentified_text=text.replace("Jane Roe", "[NAME]"))


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _FakeSMARTServer:
    """Offline SMART token/export endpoints that count every remote call."""

    base_url = "https://fhir.example.test"
    token_url = "https://auth.example.test/token"

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == self.token_url:
            self.calls["token"] += 1
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-access-token",
                    "token_type": "bearer",
                    "expires_in": 300,
                    "scope": "system/*.read",
                },
            )
        if request.url.path == "/$export":
            self.calls["kickoff"] += 1
            return httpx.Response(
                202,
                headers={
                    "Content-Location": f"{self.base_url}/bulk-status/1",
                    "Retry-After": "0",
                },
            )
        if request.url.path == "/bulk-status/1":
            self.calls["status"] += 1
            return httpx.Response(
                200,
                json={
                    "transactionTime": "2026-01-01T00:00:00Z",
                    "request": f"{self.base_url}/$export",
                    "output": [],
                },
            )
        self.calls["unexpected"] += 1
        return httpx.Response(404)

    def payload(self, output_dir: str = "synthetic-smart-out") -> dict[str, Any]:
        return {
            "fhir_base_url": self.base_url,
            "token_url": self.token_url,
            "client_id": "synthetic-client",
            "private_key_pem": SYNTHETIC_PRIVATE_KEY,
            "output_dir": output_dir,
            "poll_interval_seconds": 0,
        }


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setattr(openmed, "deidentify", _fake_deidentify)


@pytest.fixture
def gate() -> Iterator[threading.Event]:
    event = threading.Event()
    try:
        yield event
    finally:
        event.set()


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    auth: bool = False,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("OPENMED_SERVICE_BULK_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("OPENMED_SERVICE_BULK_OUTPUT_ROOT", str(tmp_path))
    if auth:
        monkeypatch.setenv(SERVICE_DIGEST_KEY_ENV_VAR, SYNTHETIC_DIGEST_KEY)
        monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
        monkeypatch.setenv(
            "OPENMED_SERVICE_AUTH_API_KEYS",
            json.dumps(
                [
                    {
                        "id": f"{name}-id",
                        "key_hash": hash_api_key(raw_key),
                        "principal": principal,
                        "scopes": scopes,
                    }
                    for name, (raw_key, principal, scopes) in _PRINCIPALS.items()
                ]
            ),
        )


def _headers(key: str | None = SYNTHETIC_IDEMPOTENCY_KEY, who: str | None = None):
    headers: dict[str, str] = {}
    if key is not None:
        headers[IDEMPOTENCY_KEY_HEADER] = key
    if who is not None:
        headers["X-API-Key"] = _PRINCIPALS[who][0]
    return headers


def _write_export(path: Path, count: int = 1) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "Patient.ndjson").write_text(
        "".join(
            json.dumps(
                {
                    "resourceType": "Patient",
                    "id": f"patient-{index}",
                    "name": [{"text": "Jane Roe"}],
                },
                separators=(",", ":"),
            )
            + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def _tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _wait(predicate: Callable[[], bool]) -> None:
    for _ in range(500):
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def _count_runs(manager: Any) -> list[str]:
    """Record every background task a manager schedules."""
    runs: list[str] = []
    original = manager._run

    async def counting(job_id: str, *args: Any) -> None:
        runs.append(job_id)
        await original(job_id, *args)

    manager._run = counting
    return runs


def _bulk_app(tmp_path: Path, deidentifier: Callable[..., Any] = _fake_deidentify):
    app = create_app()
    app.state.fhir_bulk_deidentifier = deidentifier
    _write_export(tmp_path / "synthetic-source")
    return app


def _local_body(output_dir: str = "synthetic-out") -> dict[str, Any]:
    return {"input_dir": "synthetic-source", "output_dir": output_dir}


def _start_request(
    key: ServiceDigestKey,
    *,
    route: str = "fhir-bulk-export",
    owner: dict[str, str] | None = None,
    idempotency_key: str = SYNTHETIC_IDEMPOTENCY_KEY,
    fields: dict[str, Any] | None = None,
) -> IdempotentStartRequest:
    return IdempotentStartRequest.derive(
        key,
        route=route,
        owner_digest=owner,
        idempotency_key=idempotency_key,
        request_fields=fields if fields is not None else {"output_dir": "a"},
    )


# --- key grammar ----------------------------------------------------------


def test_parse_idempotency_key_accepts_tokens_and_quoted_strings() -> None:
    assert parse_idempotency_key([]) is None
    uuid_key = "8e03978e-40d5-43e8-bc93-6894a57f9324"
    for raw, expected in (
        (uuid_key, uuid_key),
        (f'"{uuid_key}"', uuid_key),
        (f"  {uuid_key}\t", uuid_key),
        ("a", "a"),
        ("K" * 255, "K" * 255),
        ("base64+/=_.:~-key", "base64+/=_.:~-key"),
    ):
        assert parse_idempotency_key([raw]) == expected


def test_parse_idempotency_key_rejects_malformed_values_without_echo(subtests) -> None:
    marker = "SYNTHETIC-KEY-MARKER"
    for values in (
        [""],
        ['""'],
        ['"'],
        ["K" * 256],
        [f"{marker} with spaces"],
        [f"{marker},second"],
        [f"{marker};param"],
        [f'"{marker}'],
        [f"{marker}é"],
        [f"{marker}\x00"],
        [marker, marker],
    ):
        with subtests.test(values=repr(values)[:40]):
            with pytest.raises(IdempotencyError) as error:
                parse_idempotency_key(values)
            assert error.value.code == "idempotency_key_invalid"
            assert error.value.status_code == 400
            assert marker not in str(error.value)


# --- digests ----------------------------------------------------------------


def test_start_identity_is_scoped_and_excludes_credentials() -> None:
    key = ServiceDigestKey.from_secret(SYNTHETIC_DIGEST_KEY.encode())
    owner = {"alg": "hmac-sha256", "key_id": "k1", "value": "1" * 64}
    other = {"alg": "hmac-sha256", "key_id": "k1", "value": "2" * 64}
    fields = {
        "output_dir": "synthetic-out",
        "fhir_base_url": "https://fhir.example.test",
        "private_key_pem": SYNTHETIC_PRIVATE_KEY,
    }
    base = _start_request(key, owner=owner, fields=fields)

    assert base == _start_request(key, owner=owner, fields=dict(fields))
    assert (
        base.scope_digest
        != _start_request(key, owner=other, fields=fields).scope_digest
    )
    assert (
        base.scope_digest != _start_request(key, owner=None, fields=fields).scope_digest
    )
    assert (
        base.scope_digest
        != _start_request(
            key, route="fhir-bulk-import", owner=owner, fields=fields
        ).scope_digest
    )
    assert (
        base.scope_digest
        != _start_request(
            key, owner=owner, idempotency_key="another-key", fields=fields
        ).scope_digest
    )
    changed = _start_request(key, owner=owner, fields={**fields, "output_dir": "b"})
    assert changed.scope_digest == base.scope_digest
    assert changed.request_digest != base.request_digest
    rotated = _start_request(
        key, owner=owner, fields={**fields, "private_key_pem": "rotated-secret"}
    )
    assert rotated == base
    assert "private_key_pem" in CREDENTIAL_REQUEST_FIELDS
    assert b"private_key_pem" not in request_digest_material(fields)
    assert request_digest_material({"b": 1, "a": 2}) == b'{"a":2,"b":1}'

    rendered = repr(base) + base.scope_digest + base.request_digest
    for secret in (
        SYNTHETIC_IDEMPOTENCY_KEY,
        SYNTHETIC_PRIVATE_KEY,
        "synthetic-out",
        "fhir.example.test",
    ):
        assert secret not in rendered
    assert base.request_digest not in repr(base)
    other_key = ServiceDigestKey.from_secret(("x" * 40).encode())
    assert _start_request(other_key, owner=owner, fields=fields) != base


# --- store ------------------------------------------------------------------


def _store_request(scope: str = "s" * 64, digest: str = "d" * 64):
    return IdempotentStartRequest(scope_digest=scope, request_digest=digest)


def test_store_replays_conflicts_and_releases() -> None:
    store = StartIdempotencyStore()
    request = _store_request()
    reservation = store.reserve(request, job_exists=lambda _: True)
    assert isinstance(reservation, IdempotencyReservation)

    for candidate in (request, _store_request(digest="e" * 64)):
        with pytest.raises(IdempotencyError) as pending:
            store.reserve(candidate, job_exists=lambda _: True)
        expected = (
            "idempotency_key_in_progress"
            if candidate is request
            else "idempotency_key_conflict"
        )
        assert pending.value.code == expected
        assert pending.value.status_code == 409

    store.complete(reservation, job_id="job-1", response={"job_id": "job-1"})
    store.release(reservation)  # no-op once completed
    replay = store.reserve(request, job_exists=lambda _: True)
    assert replay == IdempotentReplay(job_id="job-1", response={"job_id": "job-1"})
    assert isinstance(replay, IdempotentReplay)
    replay.response["job_id"] = "mutated"
    again = store.reserve(request, job_exists=lambda _: True)
    assert isinstance(again, IdempotentReplay)
    assert again.response == {"job_id": "job-1"}

    with pytest.raises(IdempotencyError) as conflict:
        store.reserve(_store_request(digest="e" * 64), job_exists=lambda _: True)
    assert conflict.value.code == "idempotency_key_conflict"
    assert conflict.value.retry_after is None

    released = store.reserve(_store_request(scope="t" * 64), job_exists=bool)
    assert isinstance(released, IdempotencyReservation)
    store.release(released)
    assert isinstance(
        store.reserve(_store_request(scope="t" * 64, digest="f" * 64), job_exists=bool),
        IdempotencyReservation,
    )


def test_store_expires_entries_after_ttl_and_pending_lease() -> None:
    clock = _FakeClock()
    store = StartIdempotencyStore(ttl_seconds=600, clock=clock)
    request = _store_request()
    reservation = store.reserve(request, job_exists=lambda _: True)
    assert isinstance(reservation, IdempotencyReservation)
    store.complete(reservation, job_id="job-1", response={"job_id": "job-1"})

    clock.now += 599
    assert isinstance(store.reserve(request, job_exists=bool), IdempotentReplay)
    clock.now += 1
    assert len(store) == 0
    fresh = store.reserve(_store_request(digest="e" * 64), job_exists=bool)
    assert isinstance(fresh, IdempotencyReservation)

    # A start that never completes frees its key after the short pending lease.
    clock.now += PENDING_LEASE_SECONDS
    assert isinstance(store.reserve(request, job_exists=bool), IdempotencyReservation)
    # The late completion of the superseded reservation is ignored.
    store.complete(fresh, job_id="job-late", response={"job_id": "job-late"})
    with pytest.raises(IdempotencyError) as pending:
        store.reserve(request, job_exists=bool)
    assert pending.value.code == "idempotency_key_in_progress"
    assert pending.value.retry_after == 1


def test_store_forgets_entries_for_removed_jobs() -> None:
    store = StartIdempotencyStore()
    request = _store_request()
    reservation = store.reserve(request, job_exists=lambda _: True)
    assert isinstance(reservation, IdempotencyReservation)
    store.complete(reservation, job_id="job-1", response={"job_id": "job-1"})

    outcome = store.reserve(
        _store_request(digest="e" * 64), job_exists=lambda job_id: False
    )
    assert isinstance(outcome, IdempotencyReservation)


def test_store_is_bounded_and_evicts_oldest_completed_entry() -> None:
    store = StartIdempotencyStore(max_entries=2)
    first = store.reserve(_store_request(scope="1" * 64), job_exists=bool)
    second = store.reserve(_store_request(scope="2" * 64), job_exists=bool)
    assert isinstance(first, IdempotencyReservation)
    assert isinstance(second, IdempotencyReservation)
    with pytest.raises(IdempotencyError) as full:
        store.reserve(_store_request(scope="3" * 64), job_exists=bool)
    assert full.value.code == "idempotency_capacity_exhausted"
    assert full.value.status_code == 503

    store.complete(first, job_id="job-1", response={})
    store.complete(second, job_id="job-2", response={})
    third = store.reserve(_store_request(scope="3" * 64), job_exists=bool)
    assert isinstance(third, IdempotencyReservation)
    assert len(store) == 2
    # The oldest completed entry was evicted; the newer one still replays.
    assert isinstance(
        store.reserve(_store_request(scope="2" * 64), job_exists=bool),
        IdempotentReplay,
    )
    assert isinstance(
        store.reserve(_store_request(scope="1" * 64), job_exists=bool),
        IdempotencyReservation,
    )


def test_store_admits_exactly_one_of_concurrent_duplicate_reservations() -> None:
    store = StartIdempotencyStore()
    request = _store_request()
    barrier = threading.Barrier(16)

    def attempt(_: int) -> object:
        barrier.wait()
        try:
            return store.reserve(request, job_exists=bool)
        except IdempotencyError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(attempt, range(16)))

    reservations = [
        item for item in outcomes if isinstance(item, IdempotencyReservation)
    ]
    assert len(reservations) == 1
    assert outcomes.count("idempotency_key_in_progress") == 15


def test_store_settings_come_from_bounded_environment(subtests) -> None:
    default = StartIdempotencyStore.from_env({})
    assert default.ttl_seconds == DEFAULT_IDEMPOTENCY_TTL_SECONDS
    assert default.max_entries == DEFAULT_IDEMPOTENCY_MAX_KEYS
    configured = StartIdempotencyStore.from_env(
        {IDEMPOTENCY_TTL_ENV_VAR: "120", IDEMPOTENCY_MAX_KEYS_ENV_VAR: " 8 "}
    )
    assert (configured.ttl_seconds, configured.max_entries) == (120, 8)
    for name in (IDEMPOTENCY_TTL_ENV_VAR, IDEMPOTENCY_MAX_KEYS_ENV_VAR):
        for raw in ("0", "-1", "1.5", "SYNTHETIC-VALUE", "99999999"):
            with subtests.test(name=name, raw=raw):
                with pytest.raises(ValueError) as error:
                    StartIdempotencyStore.from_env({name: raw})
                assert name in str(error.value)
                assert "SYNTHETIC" not in str(error.value)
    for bad in (0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            StartIdempotencyStore(ttl_seconds=bad)
    with pytest.raises(ValueError):
        StartIdempotencyStore(max_entries=0)


# --- /fhir/bulk/* routes ----------------------------------------------------


@pytest.mark.parametrize("route", ["/fhir/bulk/exports", "/fhir/bulk/imports"])
def test_bulk_retry_returns_first_response_without_a_second_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route: str,
) -> None:
    _configure(monkeypatch, tmp_path)
    calls: list[str] = []

    def counting_deidentify(text: str, **kwargs: Any) -> _FakeResult:
        calls.append("call")
        return _fake_deidentify(text, **kwargs)

    app = _bulk_app(tmp_path, counting_deidentify)

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        manager = app.state.fhir_bulk_jobs
        runs = _count_runs(manager)
        first = client.post(route, json=_local_body(), headers=_headers())
        assert first.status_code == 202
        job_id = first.json()["job_id"]
        _wait(lambda: manager.get(job_id).status == "succeeded")
        tree = _tree(tmp_path)
        replays = [
            client.post(route, json=_local_body(), headers=_headers()) for _ in range(3)
        ]

    assert IDEMPOTENCY_REPLAYED_HEADER.lower() not in first.headers
    for replay in replays:
        assert replay.status_code == first.status_code == 202
        assert replay.headers[IDEMPOTENCY_REPLAYED_HEADER] == "true"
        assert replay.headers["content-location"] == first.headers["content-location"]
        assert replay.headers["content-location"] == f"{route}/{job_id}"
        assert replay.json() == first.json()
    assert runs == [job_id]
    assert list(manager._jobs) == [job_id]
    assert calls == ["call"]
    assert _tree(tmp_path) == tree


def test_bulk_key_reuse_with_a_different_body_conflicts_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _configure(monkeypatch, tmp_path)
    caplog.set_level(logging.DEBUG)
    app = _bulk_app(tmp_path)

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        manager = app.state.fhir_bulk_jobs
        runs = _count_runs(manager)
        first = client.post(
            "/fhir/bulk/exports", json=_local_body(), headers=_headers()
        )
        job_id = first.json()["job_id"]
        _wait(lambda: manager.get(job_id).status == "succeeded")
        tree = _tree(tmp_path)
        conflicts = [
            client.post("/fhir/bulk/exports", json=body, headers=_headers())
            for body in (
                _local_body("synthetic-other-out"),
                {**_local_body(), "method": "mask"},
                {**_local_body(), "max_buffered_resources": 2},
            )
        ]
        other_route = client.post(
            "/fhir/bulk/imports", json=_local_body(), headers=_headers()
        )

    assert first.status_code == 202
    for conflict in conflicts:
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "idempotency_key_conflict"
        assert IDEMPOTENCY_REPLAYED_HEADER.lower() not in conflict.headers
        assert SYNTHETIC_IDEMPOTENCY_KEY not in conflict.text
        assert "synthetic-other-out" not in conflict.text
    assert not (tmp_path / "synthetic-other-out").exists()
    # The same key on another start route has its own scope.
    assert other_route.status_code == 400
    assert other_route.json()["error"]["code"] == "path_exists"
    assert runs == [job_id]
    assert _tree(tmp_path) == tree
    assert SYNTHETIC_IDEMPOTENCY_KEY not in caplog.text


def test_bulk_invalid_idempotency_keys_are_rejected_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    subtests,
) -> None:
    _configure(monkeypatch, tmp_path)
    app = _bulk_app(tmp_path)
    marker = "SYNTHETIC-KEY-MARKER"

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        tree = _tree(tmp_path)
        for headers in (
            [(IDEMPOTENCY_KEY_HEADER, "")],
            [(IDEMPOTENCY_KEY_HEADER, marker + " x")],
            [(IDEMPOTENCY_KEY_HEADER, marker + "," + marker)],
            [(IDEMPOTENCY_KEY_HEADER, marker + "K" * 255)],
            [(IDEMPOTENCY_KEY_HEADER, marker), (IDEMPOTENCY_KEY_HEADER, marker)],
        ):
            with subtests.test(headers=repr(headers)[:40]):
                response = client.post(
                    "/fhir/bulk/exports", json=_local_body(), headers=headers
                )
                assert response.status_code == 400
                assert response.json()["error"]["code"] == "idempotency_key_invalid"
                assert marker not in response.text
                assert _tree(tmp_path) == tree
        assert not app.state.fhir_bulk_jobs._jobs
        assert len(app.state.fhir_bulk_jobs._idempotency) == 0


def test_bulk_rejected_start_does_not_consume_its_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path)
    app = _bulk_app(tmp_path)

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        rejected = client.post(
            "/fhir/bulk/exports",
            json={"input_dir": "synthetic-source", "output_dir": "../escape"},
            headers=_headers(),
        )
        assert len(app.state.fhir_bulk_jobs._idempotency) == 0
        accepted = client.post(
            "/fhir/bulk/exports", json=_local_body(), headers=_headers()
        )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "path_invalid"
    assert accepted.status_code == 202
    assert IDEMPOTENCY_REPLAYED_HEADER.lower() not in accepted.headers


def test_bulk_concurrent_duplicate_submissions_create_one_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate: threading.Event,
) -> None:
    _configure(monkeypatch, tmp_path)

    def gated_deidentify(text: str, **kwargs: Any) -> _FakeResult:
        gate.wait(5)
        return _fake_deidentify(text, **kwargs)

    app = _bulk_app(tmp_path, gated_deidentify)
    barrier = threading.Barrier(8)

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        manager = app.state.fhir_bulk_jobs
        runs = _count_runs(manager)

        def submit(_: int) -> httpx.Response:
            barrier.wait()
            return client.post(
                "/fhir/bulk/exports", json=_local_body(), headers=_headers()
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(submit, range(8)))
        gate.set()

    accepted = [response for response in responses if response.status_code == 202]
    in_progress = [response for response in responses if response.status_code == 409]
    assert len(accepted) + len(in_progress) == 8
    for response in in_progress:
        assert response.json()["error"]["code"] == "idempotency_key_in_progress"
        assert response.headers["retry-after"] == "1"
    assert len({response.json()["job_id"] for response in accepted}) == 1
    originals = [
        response
        for response in accepted
        if IDEMPOTENCY_REPLAYED_HEADER.lower() not in response.headers
    ]
    assert len(originals) == 1
    assert len(runs) == 1
    assert len(manager._jobs) == 1


def test_bulk_key_expires_after_ttl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path)
    app = _bulk_app(tmp_path)
    clock = _FakeClock()
    app.state.fhir_bulk_jobs = FHIRBulkJobManager(
        idempotency=StartIdempotencyStore(ttl_seconds=60, clock=clock)
    )

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        first = client.post(
            "/fhir/bulk/exports", json=_local_body(), headers=_headers()
        )
        clock.now += 59
        replay = client.post(
            "/fhir/bulk/exports", json=_local_body(), headers=_headers()
        )
        clock.now += 1
        # Expired: the same body is processed afresh and hits the reserved output.
        expired_same = client.post(
            "/fhir/bulk/exports", json=_local_body(), headers=_headers()
        )
        expired_other = client.post(
            "/fhir/bulk/exports",
            json=_local_body("synthetic-second-out"),
            headers=_headers(),
        )

    assert first.status_code == 202
    assert replay.headers[IDEMPOTENCY_REPLAYED_HEADER] == "true"
    assert replay.json()["job_id"] == first.json()["job_id"]
    assert expired_same.status_code == 400
    assert expired_same.json()["error"]["code"] == "path_exists"
    assert expired_other.status_code == 202
    assert IDEMPOTENCY_REPLAYED_HEADER.lower() not in expired_other.headers
    assert expired_other.json()["job_id"] != first.json()["job_id"]


def test_bulk_keys_are_isolated_per_principal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path, auth=True)
    app = _bulk_app(tmp_path)

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        manager = app.state.fhir_bulk_jobs
        first = client.post(
            "/fhir/bulk/exports", json=_local_body(), headers=_headers(who="owner")
        )
        job_id = first.json()["job_id"]
        rotated = client.post(
            "/fhir/bulk/exports",
            json=_local_body(),
            headers=_headers(who="owner-rotated"),
        )
        other_same = client.post(
            "/fhir/bulk/exports", json=_local_body(), headers=_headers(who="other")
        )
        other_new = client.post(
            "/fhir/bulk/exports",
            json=_local_body("synthetic-other-out"),
            headers=_headers(who="other"),
        )
        admin_new = client.post(
            "/fhir/bulk/exports",
            json=_local_body("synthetic-admin-out"),
            headers=_headers(who="admin"),
        )
        owner_conflict = client.post(
            "/fhir/bulk/exports",
            json=_local_body("synthetic-owner-second"),
            headers=_headers(who="owner"),
        )

    assert first.status_code == 202
    # A rotated API key for the same principal is the same owner.
    assert rotated.status_code == 202
    assert rotated.headers[IDEMPOTENCY_REPLAYED_HEADER] == "true"
    assert rotated.json() == first.json()
    # Another principal never sees the owner's job through the shared key.
    assert other_same.status_code == 400
    assert other_same.json()["error"]["code"] == "path_exists"
    assert job_id not in other_same.text
    for response in (other_new, admin_new):
        assert response.status_code == 202
        assert IDEMPOTENCY_REPLAYED_HEADER.lower() not in response.headers
        assert response.json()["job_id"] != job_id
    assert owner_conflict.status_code == 409
    assert owner_conflict.json()["error"]["code"] == "idempotency_key_conflict"
    assert manager.owner_digest(other_new.json()["job_id"]) != manager.owner_digest(
        job_id
    )
    assert len(manager._jobs) == 3


def test_stored_idempotency_entries_hold_no_keys_credentials_paths_or_urls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path, auth=True)
    server = _FakeSMARTServer()
    app = _bulk_app(tmp_path)
    app.state.fhir_bulk_jobs = FHIRBulkJobManager(
        transport=server.transport,
        client_assertion_builder=lambda _: "synthetic-client-assertion",
    )
    app.state.smart_backend_jobs = SMARTBackendJobManager(
        transport=server.transport,
        client_assertion_builder=lambda _: "synthetic-client-assertion",
    )

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        local = client.post(
            "/fhir/bulk/exports", json=_local_body(), headers=_headers(who="owner")
        )
        remote = client.post(
            "/fhir/bulk/exports",
            json=server.payload("synthetic-remote-out"),
            headers=_headers("synthetic-retry-key-0002", who="owner"),
        )
        smart = client.post(
            SMART_BASE, json=server.payload(), headers=_headers(who="owner")
        )

    assert {local.status_code, remote.status_code, smart.status_code} == {200, 202}
    stores = (
        app.state.fhir_bulk_jobs._idempotency,
        app.state.smart_backend_jobs._idempotency,
    )
    rendered = json.dumps(
        [
            {
                "scope": scope,
                "request": entry.request_digest,
                "job_id": entry.job_id,
                "response": entry.response,
                "repr": repr(entry),
            }
            for store in stores
            for scope, entry in store._entries.items()
        ]
    )
    assert sum(len(store) for store in stores) == 3
    for forbidden in (
        SYNTHETIC_IDEMPOTENCY_KEY,
        "synthetic-retry-key-0002",
        SYNTHETIC_PRIVATE_KEY,
        "synthetic-client",
        "synthetic-out",
        "synthetic-remote-out",
        "synthetic-smart-out",
        "synthetic-source",
        str(tmp_path),
        "example.test",
        "synthetic-owner",
    ):
        assert forbidden not in rendered


# --- remote calls -----------------------------------------------------------


@pytest.mark.parametrize("route", [SMART_BASE, "/fhir/bulk/exports"])
def test_remote_retry_makes_no_second_remote_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route: str,
) -> None:
    _configure(monkeypatch, tmp_path)
    server = _FakeSMARTServer()
    app = create_app()
    app.state.fhir_bulk_deidentifier = _fake_deidentify
    app.state.fhir_bulk_jobs = FHIRBulkJobManager(
        transport=server.transport,
        client_assertion_builder=lambda _: "synthetic-client-assertion",
    )
    app.state.smart_backend_jobs = SMARTBackendJobManager(
        transport=server.transport,
        client_assertion_builder=lambda _: "synthetic-client-assertion",
    )
    manager = (
        app.state.smart_backend_jobs
        if route == SMART_BASE
        else app.state.fhir_bulk_jobs
    )
    runs = _count_runs(manager)
    payload = server.payload()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        first = client.post(route, json=payload, headers=_headers())
        assert first.status_code in {200, 202}
        job_id = first.json()["job_id"]
        _wait(lambda: manager.get(job_id).status != "running")
        assert manager.get(job_id).status == "succeeded"
        calls = Counter(server.calls)
        replay = client.post(route, json=payload, headers=_headers())
        rotated_credential = client.post(
            route,
            json={**payload, "private_key_pem": "synthetic-rotated-private-key"},
            headers=_headers(),
        )
        conflict = client.post(
            route,
            json={**payload, "client_id": "synthetic-other-client"},
            headers=_headers(),
        )
        time.sleep(0.05)

    assert calls == Counter({"token": 1, "kickoff": 1, "status": 1})
    assert server.calls == calls
    for repeated in (replay, rotated_credential):
        assert repeated.status_code == first.status_code
        assert repeated.headers[IDEMPOTENCY_REPLAYED_HEADER] == "true"
        assert repeated.json() == first.json()
        assert repeated.headers.get("content-location") == first.headers.get(
            "content-location"
        )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_key_conflict"
    assert runs == [job_id]
    assert SYNTHETIC_PRIVATE_KEY not in replay.text


def test_smart_unkeyed_starts_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path)
    server = _FakeSMARTServer()
    app = create_app()
    app.state.smart_backend_jobs = SMARTBackendJobManager(
        transport=server.transport,
        client_assertion_builder=lambda _: "synthetic-client-assertion",
    )

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        first = client.post(SMART_BASE, json=server.payload("synthetic-a"))
        second = client.post(SMART_BASE, json=server.payload("synthetic-b"))

    assert first.status_code == second.status_code == 200
    assert first.json()["job_id"] != second.json()["job_id"]
    for response in (first, second):
        assert IDEMPOTENCY_REPLAYED_HEADER.lower() not in response.headers
    assert len(app.state.smart_backend_jobs._idempotency) == 0


def test_openapi_documents_the_smart_idempotency_header() -> None:
    operation = create_app().openapi()["paths"][SMART_BASE]["post"]
    (parameter,) = [item for item in operation["parameters"] if item["in"] == "header"]
    assert parameter["name"] == IDEMPOTENCY_KEY_HEADER
    assert parameter["required"] is False
    assert parameter["schema"]["maxLength"] == 255
    assert IDEMPOTENCY_REPLAYED_HEADER in operation["responses"]["200"]["headers"]
    assert {"400", "409"} <= set(operation["responses"])
