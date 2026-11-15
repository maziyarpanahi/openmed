"""Object-level ownership for async job, bulk and SMART ingestion records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.service import jobs
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import AuthPrincipal, hash_api_key
from openmed.service.job_ownership import (
    BULK_ADMIN_SCOPE,
    JOB_OWNER_DIGEST_DOMAIN,
    JOBS_ADMIN_SCOPE,
    JobOwnershipPolicy,
    principal_digest_material,
)
from openmed.service.keyed_digest import (
    KEYED_DIGEST_ALGORITHM,
    SERVICE_DIGEST_KEY_ENV_VAR,
    SERVICE_DIGEST_KEY_ID_ENV_VAR,
    DigestKeyError,
    KeyedDigest,
    ServiceDigestKey,
    encode_digest_fields,
    resolve_service_digest_key,
)
from openmed.service.smart_backend import SMARTBackendJobManager

LOOPBACK_BASE_URL = "http://127.0.0.1"
SYNTHETIC_DIGEST_KEY = "synthetic-digest-key-" + "a" * 40
SYNTHETIC_ROTATED_KEY = "synthetic-digest-key-" + "b" * 40
SYNTHETIC_KEY_ID = "synthetic-key-1"
_ALL_SCOPES = ["jobs:read", "jobs:write", "bulk:read", "bulk:write"]
_PRINCIPALS = {
    "owner": ("synthetic-owner-key", "synthetic-owner", _ALL_SCOPES),
    "owner-rotated": ("synthetic-owner-key-2", "synthetic-owner", _ALL_SCOPES),
    "other": ("synthetic-other-key", "synthetic-other", _ALL_SCOPES),
    "admin": (
        "synthetic-admin-key",
        "synthetic-admin",
        [*_ALL_SCOPES, JOBS_ADMIN_SCOPE, BULK_ADMIN_SCOPE],
    ),
    "wildcard": ("synthetic-wildcard-key", "synthetic-wildcard", ["*"]),
}
_SERVICE_ENV_VARS = (
    "OPENMED_PROFILE",
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_MTLS_ENABLED",
    "OPENMED_SERVICE_JOBS_STORE_PATH",
    "OPENMED_SERVICE_JOBS_WORKERS",
    "OPENMED_SERVICE_BULK_INPUT_ROOTS",
    "OPENMED_SERVICE_BULK_OUTPUT_ROOT",
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


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setattr(openmed, "deidentify", _fake_deidentify)


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    profile: str = "test",
    auth: bool = True,
    digest_key: str | None = SYNTHETIC_DIGEST_KEY,
) -> Path:
    store_path = tmp_path / "jobs.json"
    monkeypatch.setenv("OPENMED_PROFILE", profile)
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(store_path))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "1")
    monkeypatch.setenv("OPENMED_SERVICE_BULK_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("OPENMED_SERVICE_BULK_OUTPUT_ROOT", str(tmp_path))
    if digest_key is not None:
        monkeypatch.setenv(SERVICE_DIGEST_KEY_ENV_VAR, digest_key)
        monkeypatch.setenv(SERVICE_DIGEST_KEY_ID_ENV_VAR, SYNTHETIC_KEY_ID)
    if auth:
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
    return store_path


def _as(name: str) -> dict[str, str]:
    return {"X-API-Key": _PRINCIPALS[name][0]}


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


def _wait(predicate: Callable[[], bool]) -> None:
    for _ in range(300):
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def _not_found(client: TestClient, method: str, path: str, who: str) -> None:
    response = client.request(method, path, headers=_as(who))
    assert response.status_code == 404, (method, path, who, response.status_code)


def _error_body(response: httpx.Response) -> dict[str, Any]:
    """Return the error envelope without its per-request correlation id."""
    payload = response.json()
    payload["error"].pop("request_id", None)
    return payload


def _missing_body(client: TestClient, method: str, path: str) -> dict[str, Any]:
    response = client.request(method, path, headers=_as("other"))
    assert response.status_code == 404
    return _error_body(response)


@pytest.fixture
def gate() -> Iterator[threading.Event]:
    event = threading.Event()
    try:
        yield event
    finally:
        event.set()


# --- keyed digest helper -------------------------------------------------


def test_keyed_digest_is_domain_separated_and_carries_markers() -> None:
    key = ServiceDigestKey.from_secret(SYNTHETIC_DIGEST_KEY.encode(), key_id="k1")
    material = encode_digest_fields("api_key", None, "synthetic-owner")
    digest = key.digest(JOB_OWNER_DIGEST_DOMAIN, material)

    assert digest.alg == KEYED_DIGEST_ALGORITHM
    assert digest.key_id == "k1"
    assert digest == key.digest(JOB_OWNER_DIGEST_DOMAIN, material)
    assert digest != key.digest("span-text", material)
    assert key.verify(digest, JOB_OWNER_DIGEST_DOMAIN, material)
    assert not key.verify(digest, "span-text", material)
    assert not key.verify(
        digest,
        JOB_OWNER_DIGEST_DOMAIN,
        encode_digest_fields("api_key", None, "synthetic-other"),
    )
    assert KeyedDigest.from_dict(digest.to_dict()) == digest
    # Unkeyed hashes of the identity never match the stored value.
    assert digest.value != hashlib.sha256(material).hexdigest()
    assert digest.value != hashlib.sha256(b"synthetic-owner").hexdigest()


def test_keyed_digest_fails_closed_across_keys_and_bad_records(subtests) -> None:
    material = encode_digest_fields("api_key", None, "synthetic-owner")
    key = ServiceDigestKey.from_secret(SYNTHETIC_DIGEST_KEY.encode())
    rotated = ServiceDigestKey.from_secret(SYNTHETIC_ROTATED_KEY.encode())
    same_id_other_secret = ServiceDigestKey.from_secret(
        SYNTHETIC_ROTATED_KEY.encode(), key_id=key.key_id
    )
    digest = key.digest(JOB_OWNER_DIGEST_DOMAIN, material)

    assert key.key_id.startswith("k-") and key.key_id != rotated.key_id
    assert not rotated.verify(digest, JOB_OWNER_DIGEST_DOMAIN, material)
    assert not same_id_other_secret.verify(digest, JOB_OWNER_DIGEST_DOMAIN, material)
    assert SYNTHETIC_DIGEST_KEY not in repr(key)

    for raw in (
        None,
        "hmac-sha256:abc",
        {"alg": "sha256", "key_id": "k1", "value": "0" * 64},
        {"alg": KEYED_DIGEST_ALGORITHM, "key_id": "", "value": "0" * 64},
        {"alg": KEYED_DIGEST_ALGORITHM, "key_id": "k1", "value": "Z" * 64},
        {"alg": KEYED_DIGEST_ALGORITHM, "key_id": "k1", "value": "0" * 64, "x": 1},
    ):
        with subtests.test(raw=repr(raw)[:40]):
            with pytest.raises(ValueError):
                KeyedDigest.from_dict(raw)
    with pytest.raises(ValueError):
        key.digest("Bad Domain", material)
    with pytest.raises(TypeError):
        key.digest(JOB_OWNER_DIGEST_DOMAIN, "not-bytes")  # type: ignore[arg-type]
    assert encode_digest_fields("a,b", None) != encode_digest_fields("a", "b,")
    with pytest.raises(TypeError):
        encode_digest_fields("a", 1)  # type: ignore[arg-type]


def test_digest_key_resolution_requires_durable_key_in_prod(subtests) -> None:
    configured = resolve_service_digest_key(
        profile="prod",
        required=True,
        environ={
            SERVICE_DIGEST_KEY_ENV_VAR: SYNTHETIC_DIGEST_KEY + "\n",
            SERVICE_DIGEST_KEY_ID_ENV_VAR: SYNTHETIC_KEY_ID,
        },
    )
    assert configured.key_id == SYNTHETIC_KEY_ID
    assert configured.ephemeral is False
    assert configured.secret == SYNTHETIC_DIGEST_KEY.encode()

    for profile, required in (("dev", True), ("test", True), ("prod", False)):
        with subtests.test(profile=profile, required=required):
            key = resolve_service_digest_key(
                profile=profile, required=required, environ={}
            )
            assert key.ephemeral is True
            assert key.key_id.startswith("ephemeral-")

    for profile in ("prod", "low_resource"):
        with subtests.test(profile=profile):
            with pytest.raises(DigestKeyError) as error:
                resolve_service_digest_key(profile=profile, required=True, environ={})
            assert error.value.code == "digest_key_required"

    with pytest.raises(DigestKeyError) as short:
        resolve_service_digest_key(
            profile="dev",
            required=False,
            environ={SERVICE_DIGEST_KEY_ENV_VAR: "synthetic-short-key"},
        )
    assert short.value.code == "digest_key_too_short"
    assert "synthetic-short-key" not in str(short.value)
    with pytest.raises(DigestKeyError) as bad_id:
        resolve_service_digest_key(
            profile="dev",
            required=False,
            environ={
                SERVICE_DIGEST_KEY_ENV_VAR: SYNTHETIC_DIGEST_KEY,
                SERVICE_DIGEST_KEY_ID_ENV_VAR: "bad key id",
            },
        )
    assert bad_id.value.code == "digest_key_id_invalid"


def test_prod_service_with_auth_refuses_to_start_without_digest_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path, profile="prod", digest_key=None)
    with pytest.raises(DigestKeyError) as error:
        create_app()
    assert error.value.code == "digest_key_required"
    assert "synthetic" not in str(error.value)

    # Auth disabled: the prod service starts exactly as before.
    monkeypatch.delenv("OPENMED_SERVICE_AUTH_ENABLED")
    app = create_app()
    assert app.state.job_ownership.enforced is False


# --- principal binding policy --------------------------------------------


def _principal(
    subject: str,
    *,
    credential_type: str = "api_key",
    scopes: tuple[str, ...] = (),
    key_id: str | None = None,
    claims: dict[str, Any] | None = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        subject=subject,
        credential_type=credential_type,
        scopes=scopes,
        key_id=key_id,
        claims=claims or {},
    )


def test_policy_binds_identity_not_key_id_and_separates_credential_types() -> None:
    key = ServiceDigestKey.from_secret(SYNTHETIC_DIGEST_KEY.encode())
    policy = JobOwnershipPolicy(key=key, enforced=True)
    owner = policy.owner_digest(_principal("synthetic-owner", key_id="a"))
    assert owner is not None
    assert "synthetic-owner" not in json.dumps(owner)

    assert policy.permits(
        _principal("synthetic-owner", key_id="b"), owner, admin_scope=JOBS_ADMIN_SCOPE
    )
    for stranger in (
        _principal("synthetic-other"),
        _principal("synthetic-owner", credential_type="jwt"),
        _principal("synthetic-owner", credential_type="mtls"),
        _principal("synthetic-wildcard", scopes=("*", "jobs:*")),
        _principal("synthetic-other", scopes=(BULK_ADMIN_SCOPE,)),
    ):
        assert not policy.permits(stranger, owner, admin_scope=JOBS_ADMIN_SCOPE)
    assert policy.permits(
        _principal("synthetic-admin", scopes=(JOBS_ADMIN_SCOPE,)),
        owner,
        admin_scope=JOBS_ADMIN_SCOPE,
    )

    first = _principal("sub-1", credential_type="jwt", claims={"iss": "issuer-a"})
    second = _principal("sub-1", credential_type="jwt", claims={"iss": "issuer-b"})
    assert principal_digest_material(first) != principal_digest_material(second)
    jwt_owner = policy.owner_digest(first)
    assert policy.permits(first, jwt_owner, admin_scope=JOBS_ADMIN_SCOPE)
    assert not policy.permits(second, jwt_owner, admin_scope=JOBS_ADMIN_SCOPE)


def test_policy_handles_unowned_rotated_and_anonymous_records() -> None:
    key = ServiceDigestKey.from_secret(SYNTHETIC_DIGEST_KEY.encode())
    owner_principal = _principal("synthetic-owner")
    admin = _principal("synthetic-admin", scopes=(JOBS_ADMIN_SCOPE,))
    prod = JobOwnershipPolicy(key=key, enforced=True, allow_unowned=False)
    dev = JobOwnershipPolicy(key=key, enforced=True, allow_unowned=True)
    rotated = JobOwnershipPolicy(
        key=ServiceDigestKey.from_secret(SYNTHETIC_ROTATED_KEY.encode()),
        enforced=True,
    )
    owner = prod.owner_digest(owner_principal)

    # Legacy records without an owner are admin-only outside dev/test.
    assert not prod.permits(owner_principal, None, admin_scope=JOBS_ADMIN_SCOPE)
    assert prod.permits(admin, None, admin_scope=JOBS_ADMIN_SCOPE)
    assert dev.permits(owner_principal, None, admin_scope=JOBS_ADMIN_SCOPE)
    # A rotated key cannot verify old digests: fail closed, admins still see it.
    assert not rotated.permits(owner_principal, owner, admin_scope=JOBS_ADMIN_SCOPE)
    assert rotated.permits(admin, owner, admin_scope=JOBS_ADMIN_SCOPE)
    # Malformed stored owners never match.
    assert not prod.permits(owner_principal, "corrupt", admin_scope=JOBS_ADMIN_SCOPE)
    # Without a principal: allowed only when authentication is disabled.
    assert not prod.permits(None, owner, admin_scope=JOBS_ADMIN_SCOPE)
    disabled = JobOwnershipPolicy(key=key, enforced=False)
    assert disabled.owner_digest(None) is None
    assert disabled.permits(None, owner, admin_scope=JOBS_ADMIN_SCOPE)


# --- /jobs ---------------------------------------------------------------


def test_jobs_routes_return_404_to_other_principals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store_path = _configure(monkeypatch, tmp_path)
    caplog.set_level(logging.DEBUG)
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        created = client.post(
            "/jobs",
            json={"documents": [{"id": "doc-1", "text": "Patient Jane Roe"}]},
            headers=_as("owner"),
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        path = f"/jobs/{job_id}"
        _wait(
            lambda: (
                client.get(path, headers=_as("owner")).json()["status"]
                in jobs.TERMINAL_STATUSES
            )
        )
        responses = {
            who: client.get(path, headers=_as(who))
            for who in ("owner", "owner-rotated", "admin", "other", "wildcard")
        }
        missing = _missing_body(client, "GET", "/jobs/" + "0" * 32)

    for who in ("owner", "owner-rotated", "admin"):
        assert responses[who].status_code == 200, who
        assert responses[who].json()["id"] == job_id
    for who in ("other", "wildcard"):
        assert responses[who].status_code == 404, who
        assert _error_body(responses[who]) == missing

    stored = json.loads(store_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == jobs.JOB_STORE_SCHEMA_VERSION
    owner = stored["jobs"][job_id][jobs.OWNER_DIGEST_FIELD]
    assert owner["alg"] == KEYED_DIGEST_ALGORITHM
    assert owner["key_id"] == SYNTHETIC_KEY_ID
    rendered = json.dumps(
        [created.json(), *(response.json() for response in responses.values())]
    )
    assert jobs.OWNER_DIGEST_FIELD not in rendered
    assert owner["value"] not in rendered
    assert owner["value"] not in caplog.text
    store_text = store_path.read_text(encoding="utf-8")
    for principal in ("synthetic-owner", "synthetic-other", "synthetic-admin"):
        assert principal not in store_text


def test_job_webhook_and_response_payloads_exclude_owner_digest() -> None:
    record = {
        "id": "synthetic-job",
        "status": "done",
        "progress_percent": 100.0,
        "document_count": 0,
        "processed_count": 0,
        "failed_count": 0,
        "label_histogram": {},
        "spans": [],
        "documents": [],
        "error": None,
        "completed_at": "2026-01-01T00:00:00Z",
        jobs.OWNER_DIGEST_FIELD: {
            "alg": KEYED_DIGEST_ALGORITHM,
            "key_id": SYNTHETIC_KEY_ID,
            "value": "1" * 64,
        },
    }

    for payload in (jobs.webhook_payload(record), jobs.job_response_payload(record)):
        assert jobs.OWNER_DIGEST_FIELD not in payload
        assert "1" * 64 not in json.dumps(payload)
    assert jobs.OWNER_DIGEST_FIELD in record


def test_legacy_job_store_records_are_migrated_as_admin_only_in_prod(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    legacy_id = "1" * 32
    for profile, visible_to_others in (("prod", False), ("test", True)):
        case = tmp_path / profile
        case.mkdir()
        store_path = _configure(monkeypatch, case, profile=profile)
        store_path.write_text(
            json.dumps(
                {
                    "jobs": {
                        legacy_id: {
                            "id": legacy_id,
                            "status": "done",
                            "expires_at": "2999-01-01T00:00:00Z",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        loaded = jobs.LocalJobStore(store_path).get(legacy_id)
        assert loaded is not None and loaded[jobs.OWNER_DIGEST_FIELD] is None

        app = create_app()
        with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
            owner = client.get(f"/jobs/{legacy_id}", headers=_as("owner"))
            admin = client.get(f"/jobs/{legacy_id}", headers=_as("admin"))

        assert admin.status_code == 200
        assert jobs.OWNER_DIGEST_FIELD not in admin.json()
        assert owner.status_code == (200 if visible_to_others else 404)


def test_job_store_writes_schema_version_and_rejects_unknown_versions(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "jobs.json"
    store = jobs.LocalJobStore(store_path)
    store.create({"id": "synthetic", "status": "queued"})
    assert json.loads(store_path.read_text(encoding="utf-8"))["schema_version"] == 2

    for version in (0, 99, "2", True):
        store_path.write_text(
            json.dumps({"jobs": {}, "schema_version": version}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="unsupported schema version"):
            jobs.LocalJobStore(store_path)


def test_auth_disabled_job_routes_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure(monkeypatch, tmp_path, auth=False, digest_key=None)
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        created = client.post("/jobs", json={"documents": [{"text": "synthetic"}]})
        job_id = created.json()["id"]
        status = client.get(f"/jobs/{job_id}")

    assert created.status_code == 202
    assert status.status_code == 200
    assert jobs.OWNER_DIGEST_FIELD not in status.json()
    stored = json.loads(store_path.read_text(encoding="utf-8"))["jobs"][job_id]
    assert stored[jobs.OWNER_DIGEST_FIELD] is None


# --- /fhir/bulk/* ---------------------------------------------------------


def test_bulk_routes_return_404_to_other_principals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate: threading.Event,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _configure(monkeypatch, tmp_path)
    caplog.set_level(logging.DEBUG)
    app = create_app()

    def gated_deidentify(text: str, **kwargs: Any) -> _FakeResult:
        gate.wait(5)
        return _fake_deidentify(text, **kwargs)

    app.state.fhir_bulk_deidentifier = gated_deidentify
    _write_export(tmp_path / "synthetic-source")
    prefixes = ("/fhir/bulk/exports", "/fhir/bulk/imports")

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        started = client.post(
            "/fhir/bulk/exports",
            json={"input_dir": "synthetic-source", "output_dir": "synthetic-out"},
            headers=_as("owner"),
        )
        assert started.status_code == 202
        job_id = started.json()["job_id"]
        manager = app.state.fhir_bulk_jobs

        # While running: non-owners cannot even learn that it is incomplete.
        for prefix in prefixes:
            for suffix in ("/manifest", "/report"):
                path = f"{prefix}/{job_id}{suffix}"
                assert client.get(path, headers=_as("owner")).status_code == 409
                assert client.get(path, headers=_as("admin")).status_code == 409
                for who in ("other", "wildcard"):
                    _not_found(client, "GET", path, who)
            for who in ("other", "wildcard"):
                _not_found(client, "DELETE", f"{prefix}/{job_id}", who)
                _not_found(client, "GET", f"{prefix}/{job_id}", who)
        assert manager.get(job_id).status == "running"

        gate.set()
        _wait(lambda: manager.get(job_id).status == "succeeded")
        missing = {
            suffix: _missing_body(client, "GET", f"{prefixes[0]}/{'0' * 32}{suffix}")
            for suffix in ("", "/manifest", "/report")
        }
        allowed: list[httpx.Response] = []
        for prefix in prefixes:
            for suffix in ("", "/manifest", "/report"):
                path = f"{prefix}/{job_id}{suffix}"
                for who in ("owner", "owner-rotated", "admin"):
                    response = client.get(path, headers=_as(who))
                    assert response.status_code == 200, (path, who)
                    allowed.append(response)
                for who in ("other", "wildcard"):
                    response = client.get(path, headers=_as(who))
                    assert response.status_code == 404
                    assert _error_body(response) == missing[suffix]
        cancelled = client.delete(f"{prefixes[0]}/{job_id}", headers=_as("owner"))
        admin_cancel = client.delete(f"{prefixes[1]}/{job_id}", headers=_as("admin"))

    assert cancelled.status_code == 202
    assert admin_cancel.status_code == 202
    owner = manager.owner_digest(job_id)
    assert owner is not None and owner["key_id"] == SYNTHETIC_KEY_ID
    rendered = json.dumps(
        [started.json(), cancelled.json(), *(item.json() for item in allowed)]
    )
    assert owner["value"] not in rendered
    assert "owner_digest" not in rendered
    assert owner["value"] not in caplog.text
    assert "synthetic-owner" not in rendered


def test_bulk_cancel_by_other_principal_does_not_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate: threading.Event,
) -> None:
    _configure(monkeypatch, tmp_path)
    app = create_app()

    def gated_deidentify(text: str, **kwargs: Any) -> _FakeResult:
        gate.wait(5)
        return _fake_deidentify(text, **kwargs)

    app.state.fhir_bulk_deidentifier = gated_deidentify
    _write_export(tmp_path / "synthetic-source")

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        job_id = client.post(
            "/fhir/bulk/imports",
            json={"source_dir": "synthetic-source", "output_dir": "synthetic-out"},
            headers=_as("owner"),
        ).json()["job_id"]
        foreign = client.delete(f"/fhir/bulk/imports/{job_id}", headers=_as("other"))
        still_running = app.state.fhir_bulk_jobs.get(job_id).status
        owned = client.delete(f"/fhir/bulk/imports/{job_id}", headers=_as("owner"))
        gate.set()

    assert foreign.status_code == 404
    assert still_running == "running"
    assert owned.status_code == 202
    assert owned.json()["status"] == "cancelled"


# --- /fhir/smart-backend/ingestions ---------------------------------------


def test_smart_ingestion_routes_return_404_to_other_principals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate: threading.Event,
) -> None:
    _configure(monkeypatch, tmp_path)
    app = create_app()

    async def blocked(_: httpx.Request) -> httpx.Response:
        while not gate.is_set():
            await asyncio.sleep(0.01)
        return httpx.Response(503)

    manager = SMARTBackendJobManager(
        transport=httpx.MockTransport(blocked),
        client_assertion_builder=lambda _: "synthetic-client-assertion",
    )
    app.state.smart_backend_jobs = manager
    payload = {
        "fhir_base_url": "https://fhir.example.test",
        "token_url": "https://auth.example.test/token",
        "client_id": "synthetic-client",
        "private_key_pem": "synthetic-private-key-secret",
        "output_dir": "synthetic-smart-out",
    }
    base = "/fhir/smart-backend/ingestions"

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        started = client.post(base, json=payload, headers=_as("owner"))
        assert started.status_code == 200
        job_id = started.json()["job_id"]
        missing = {
            suffix: _missing_body(client, "GET", f"{base}/{'0' * 32}{suffix}")
            for suffix in ("", "/summary")
        }
        for who in ("owner", "owner-rotated", "admin"):
            assert client.get(f"{base}/{job_id}", headers=_as(who)).status_code == 200
            summary = client.get(f"{base}/{job_id}/summary", headers=_as(who))
            assert summary.status_code == 409
        for who in ("other", "wildcard"):
            for suffix in ("", "/summary"):
                response = client.get(f"{base}/{job_id}{suffix}", headers=_as(who))
                assert response.status_code == 404
                assert _error_body(response) == missing[suffix]
            _not_found(client, "DELETE", f"{base}/{job_id}", who)
        assert manager.get(job_id).status == "running"
        cancelled = client.delete(f"{base}/{job_id}", headers=_as("owner"))
        gate.set()

    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"
    owner = manager.owner_digest(job_id)
    assert owner is not None
    rendered = json.dumps([started.json(), cancelled.json()])
    assert owner["value"] not in rendered
    assert "synthetic-private-key-secret" not in rendered


def test_auth_disabled_bulk_and_smart_records_have_no_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path, auth=False, digest_key=None)
    app = create_app()
    app.state.fhir_bulk_deidentifier = _fake_deidentify
    manager = SMARTBackendJobManager(
        transport=httpx.MockTransport(lambda _: httpx.Response(503)),
        client_assertion_builder=lambda _: "synthetic-client-assertion",
    )
    app.state.smart_backend_jobs = manager
    _write_export(tmp_path / "synthetic-source")

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        bulk_id = client.post(
            "/fhir/bulk/exports",
            json={"input_dir": "synthetic-source", "output_dir": "synthetic-out"},
        ).json()["job_id"]
        smart_id = client.post(
            "/fhir/smart-backend/ingestions",
            json={
                "fhir_base_url": "https://fhir.example.test",
                "token_url": "https://auth.example.test/token",
                "client_id": "synthetic-client",
                "private_key_pem": "synthetic-private-key-secret",
                "output_dir": "synthetic-smart-out",
            },
        ).json()["job_id"]
        bulk_status = client.get(f"/fhir/bulk/exports/{bulk_id}")
        smart_status = client.get(f"/fhir/smart-backend/ingestions/{smart_id}")

    assert bulk_status.status_code == 200
    assert smart_status.status_code == 200
    assert app.state.fhir_bulk_jobs.owner_digest(bulk_id) is None
    assert manager.owner_digest(smart_id) is None
