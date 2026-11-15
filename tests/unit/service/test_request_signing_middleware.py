"""Tests for opt-in inbound HMAC request-signature enforcement."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from types import ModuleType
from typing import Any, Callable, Optional

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from openmed.service import runtime as service_runtime
from openmed.service.signing import (
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    sign_request,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
NOW = 1_800_000_000
SECRET = b"SYNTHETIC-SIGNING-SECRET-SENTINEL-0123456789"
BODY_SENTINEL = "SYNTHETIC-BODY-SENTINEL"
NONCE_SENTINEL = "SYNTHETIC-NONCE-SENTINEL"
_SIGNING_ENV_VARS = (
    "OPENMED_SERVICE_REQUEST_SIGNING_ROUTES",
    "OPENMED_SERVICE_REQUEST_SIGNING_SECRET_FILE",
    "OPENMED_SERVICE_REQUEST_SIGNING_MAX_SKEW_SECONDS",
    "OPENMED_SERVICE_REQUEST_SIGNING_NONCE_CACHE_SIZE",
)
_SERVICE_ENV_VARS = (
    *_SIGNING_ENV_VARS,
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    "OPENMED_SERVICE_MAX_REQUEST_BYTES",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_MTLS_ENABLED",
    "OPENMED_SERVICE_METRICS_ENABLED",
    "OPENMED_SERVICE_JOBS_STORE_PATH",
)


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


class FakeLoader:
    """Model loader double; signed-route tests never load a model."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@pytest.fixture(autouse=True)
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


@pytest.fixture
def signing() -> ModuleType:
    return importlib.import_module("openmed.service.request_signing")


def _error_response(*args: Any, **kwargs: Any) -> Any:
    from openmed.service.app import _error_response as service_error_response

    return service_error_response(*args, **kwargs)


def _signed(
    method: str,
    target: str,
    body: bytes = b"",
    *,
    timestamp: int = NOW,
    nonce: str = "synthetic-nonce-1",
    secret: bytes = SECRET,
) -> dict[str, str]:
    return sign_request(
        method,
        target,
        body,
        secret=secret,
        timestamp=timestamp,
        nonce=nonce,
    )


class _Clock:
    def __init__(self, value: float = NOW) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _minimal_app(
    signing: ModuleType,
    *,
    routes: frozenset[tuple[str, str]],
    clock: Optional[Callable[[], float]] = None,
    nonce_cache_size: int = 100,
    max_body_bytes: int = 1_000_000,
    install: bool = True,
) -> tuple[FastAPI, list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []
    app = FastAPI()

    @app.post("/notes")
    async def create_note(request: Request) -> dict[str, Any]:
        body = await request.body()
        calls.append({"route": "create", "body": body})
        return {"sha256": hashlib.sha256(body).hexdigest(), "length": len(body)}

    @app.post("/notes/{note_id}/cancel")
    async def cancel_note(note_id: str, request: Request) -> dict[str, Any]:
        body = await request.body()
        calls.append({"route": "cancel", "body": body, "note_id": note_id})
        return {"cancelled": note_id}

    @app.get("/notes/{note_id}")
    async def read_note(note_id: str) -> dict[str, Any]:
        calls.append({"route": "read", "note_id": note_id})
        return {"note_id": note_id}

    @app.get("/open")
    async def open_route(request: Request) -> dict[str, Any]:
        calls.append({"route": "open"})
        return {"status": "ok", "query": str(request.url.query)}

    verifier = None
    if install:
        verifier = signing.RequestSignatureVerifier(
            signing.RequestSigningConfig(
                routes=routes,
                secret=SECRET,
                max_skew_seconds=300,
                nonce_cache_size=nonce_cache_size,
            ),
            error_response=_error_response,
            clock=clock or _Clock(),
        )
        app.add_middleware(
            signing.RequestSigningMiddleware,
            verifier=verifier,
            max_body_bytes=max_body_bytes,
        )
    return app, calls, verifier


def _assert_signature_error(response: Any, code: str) -> dict[str, Any]:
    assert response.status_code == 401
    payload = response.json()
    assert payload["error"]["code"] == code
    assert payload["error"]["details"] is None
    assert response.headers["www-authenticate"] == "OPENMED-HMAC-SHA256"
    return payload


# Middleware behavior with an injected clock.


def test_valid_signature_reaches_handler_with_identical_body(
    signing: ModuleType,
) -> None:
    app, calls, _ = _minimal_app(signing, routes=frozenset({("POST", "/notes")}))
    body = json.dumps({"note": "synthetic"}, separators=(",", ":")).encode()
    client = TestClient(app, base_url=LOOPBACK_BASE_URL)

    response = client.post(
        "/notes?mode=async",
        content=body,
        headers=_signed("POST", "/notes?mode=async", body),
    )

    assert response.status_code == 200
    assert response.json() == {
        "sha256": hashlib.sha256(body).hexdigest(),
        "length": len(body),
    }
    assert calls == [{"route": "create", "body": body}]


def test_chunked_body_is_verified_and_replayed_to_handler(
    signing: ModuleType,
) -> None:
    app, calls, _ = _minimal_app(signing, routes=frozenset({("POST", "/notes")}))
    chunks = [b'{"part":', b'"one",', b'"tail":true}']
    body = b"".join(chunks)
    client = TestClient(app, base_url=LOOPBACK_BASE_URL)

    response = client.post(
        "/notes",
        content=iter(chunks),
        headers=_signed("POST", "/notes", body),
    )

    assert response.status_code == 200
    assert calls == [{"route": "create", "body": body}]


@pytest.mark.parametrize(
    "missing",
    [
        (TIMESTAMP_HEADER,),
        (NONCE_HEADER,),
        (SIGNATURE_HEADER,),
        (TIMESTAMP_HEADER, NONCE_HEADER, SIGNATURE_HEADER),
    ],
)
def test_missing_signature_headers_are_rejected_before_handler(
    signing: ModuleType,
    missing: tuple[str, ...],
) -> None:
    app, calls, _ = _minimal_app(signing, routes=frozenset({("POST", "/notes")}))
    headers = _signed("POST", "/notes", b"{}")
    for name in missing:
        headers.pop(name)

    response = TestClient(app, base_url=LOOPBACK_BASE_URL).post(
        "/notes", content=b"{}", headers=headers
    )

    _assert_signature_error(response, "signature_missing")
    assert calls == []


def _tampered_cases() -> list[tuple[str, str, bytes, dict[str, str]]]:
    body = b'{"note":"synthetic"}'
    base = _signed("POST", "/notes", body)
    with_timestamp = {**base, TIMESTAMP_HEADER: str(NOW + 1)}
    with_nonce = {**base, NONCE_HEADER: "synthetic-nonce-other"}
    bad_digest = {**base, SIGNATURE_HEADER: "sha256=" + "0" * 64}
    bad_prefix = {**base, SIGNATURE_HEADER: base[SIGNATURE_HEADER][7:]}
    bad_timestamp = {**base, TIMESTAMP_HEADER: "not-a-number"}
    return [
        ("body", "/notes", body + b" ", base),
        ("path", "/notes?mode=other", body, base),
        ("method", "/notes", body, _signed("PUT", "/notes", body)),
        ("timestamp", "/notes", body, with_timestamp),
        ("nonce", "/notes", body, with_nonce),
        ("digest", "/notes", body, bad_digest),
        ("prefix", "/notes", body, bad_prefix),
        ("timestamp-format", "/notes", body, bad_timestamp),
        ("wrong-secret", "/notes", body, _signed("POST", "/notes", body, secret=b"x")),
    ]


@pytest.mark.parametrize(
    ("label", "target", "body", "headers"),
    _tampered_cases(),
    ids=[case[0] for case in _tampered_cases()],
)
def test_tampered_requests_are_rejected_before_handler(
    signing: ModuleType,
    label: str,
    target: str,
    body: bytes,
    headers: dict[str, str],
) -> None:
    del label
    app, calls, verifier = _minimal_app(signing, routes=frozenset({("POST", "/notes")}))

    response = TestClient(app, base_url=LOOPBACK_BASE_URL).post(
        target, content=body, headers=headers
    )

    _assert_signature_error(response, "signature_invalid")
    assert calls == []
    assert len(verifier.nonce_cache) == 0


def test_duplicate_signature_header_is_invalid(signing: ModuleType) -> None:
    app, calls, _ = _minimal_app(signing, routes=frozenset({("POST", "/notes")}))
    headers = list(_signed("POST", "/notes", b"{}").items())
    headers.append((SIGNATURE_HEADER, "sha256=" + "1" * 64))

    response = TestClient(app, base_url=LOOPBACK_BASE_URL).post(
        "/notes", content=b"{}", headers=headers
    )

    _assert_signature_error(response, "signature_invalid")
    assert calls == []


@pytest.mark.parametrize("offset", [301, -301])
def test_stale_timestamp_is_rejected_with_injected_clock(
    signing: ModuleType,
    offset: int,
) -> None:
    clock = _Clock(NOW + offset)
    app, calls, verifier = _minimal_app(
        signing, routes=frozenset({("POST", "/notes")}), clock=clock
    )
    headers = _signed("POST", "/notes", b"{}")
    client = TestClient(app, base_url=LOOPBACK_BASE_URL)

    response = client.post("/notes", content=b"{}", headers=headers)

    _assert_signature_error(response, "signature_stale")
    assert calls == []
    assert len(verifier.nonce_cache) == 0

    clock.value = NOW + 300 if offset > 0 else NOW - 300
    accepted = client.post("/notes", content=b"{}", headers=headers)
    assert accepted.status_code == 200
    assert len(calls) == 1


def test_replayed_nonce_is_rejected_before_handler(signing: ModuleType) -> None:
    clock = _Clock()
    app, calls, _ = _minimal_app(
        signing, routes=frozenset({("POST", "/notes")}), clock=clock
    )
    headers = _signed("POST", "/notes", b"{}")
    client = TestClient(app, base_url=LOOPBACK_BASE_URL)

    assert client.post("/notes", content=b"{}", headers=headers).status_code == 200
    clock.value = NOW + 10
    replay = client.post("/notes", content=b"{}", headers=headers)

    _assert_signature_error(replay, "signature_replayed")
    assert len(calls) == 1


def test_failed_attempt_does_not_consume_the_nonce(signing: ModuleType) -> None:
    app, calls, _ = _minimal_app(signing, routes=frozenset({("POST", "/notes")}))
    headers = _signed("POST", "/notes", b"{}")
    client = TestClient(app, base_url=LOOPBACK_BASE_URL)

    tampered = client.post("/notes", content=b"{} ", headers=headers)
    accepted = client.post("/notes", content=b"{}", headers=headers)

    _assert_signature_error(tampered, "signature_invalid")
    assert accepted.status_code == 200
    assert len(calls) == 1


def test_full_nonce_cache_fails_closed_without_running_handler(
    signing: ModuleType,
) -> None:
    app, calls, _ = _minimal_app(
        signing, routes=frozenset({("POST", "/notes")}), nonce_cache_size=1
    )
    client = TestClient(app, base_url=LOOPBACK_BASE_URL)

    first = client.post(
        "/notes", content=b"{}", headers=_signed("POST", "/notes", b"{}")
    )
    second = client.post(
        "/notes",
        content=b"{}",
        headers=_signed("POST", "/notes", b"{}", nonce="synthetic-nonce-2"),
    )

    assert first.status_code == 200
    assert second.status_code == 503
    assert second.json()["error"]["code"] == "service_busy"
    assert len(calls) == 1


def test_path_parameter_templates_require_signatures(signing: ModuleType) -> None:
    app, calls, _ = _minimal_app(
        signing, routes=frozenset({("POST", "/notes/{note_id}/cancel")})
    )
    client = TestClient(app, base_url=LOOPBACK_BASE_URL)

    unsigned = client.post("/notes/synthetic-7/cancel", content=b"")
    signed = client.post(
        "/notes/synthetic-7/cancel",
        content=b"",
        headers=_signed("POST", "/notes/synthetic-7/cancel"),
    )
    wrong_target = client.post(
        "/notes/synthetic-8/cancel",
        content=b"",
        headers=_signed("POST", "/notes/synthetic-7/cancel", nonce="other"),
    )

    _assert_signature_error(unsigned, "signature_missing")
    assert signed.status_code == 200
    assert signed.json() == {"cancelled": "synthetic-7"}
    _assert_signature_error(wrong_target, "signature_invalid")
    assert [call["note_id"] for call in calls] == ["synthetic-7"]


def test_signed_get_template_also_covers_implicit_head(signing: ModuleType) -> None:
    app, calls, _ = _minimal_app(
        signing, routes=frozenset({("GET", "/notes/{note_id}")})
    )
    client = TestClient(app, base_url=LOOPBACK_BASE_URL)

    head = client.head("/notes/synthetic-1")
    get = client.get("/notes/synthetic-1")
    signed_get = client.get(
        "/notes/synthetic-1", headers=_signed("GET", "/notes/synthetic-1")
    )

    assert head.status_code == 401
    assert head.headers["www-authenticate"] == "OPENMED-HMAC-SHA256"
    _assert_signature_error(get, "signature_missing")
    assert signed_get.status_code == 200
    assert len(calls) == 1


def test_unconfigured_routes_are_unchanged(signing: ModuleType) -> None:
    signed_app, signed_calls, _ = _minimal_app(
        signing, routes=frozenset({("POST", "/notes")})
    )
    plain_app, plain_calls, _ = _minimal_app(signing, routes=frozenset(), install=False)
    garbage = {
        TIMESTAMP_HEADER: "garbage",
        NONCE_HEADER: "garbage",
        SIGNATURE_HEADER: "garbage",
    }

    for headers in ({}, garbage, _signed("GET", "/open?x=1")):
        for method, target in (("GET", "/open?x=1"), ("GET", "/notes/synthetic")):
            with_signing = TestClient(signed_app, base_url=LOOPBACK_BASE_URL).request(
                method, target, headers=headers
            )
            without = TestClient(plain_app, base_url=LOOPBACK_BASE_URL).request(
                method, target, headers=headers
            )
            assert with_signing.status_code == without.status_code == 200
            assert with_signing.content == without.content
            assert dict(with_signing.headers) == dict(without.headers)
        unsigned_cancel = TestClient(signed_app, base_url=LOOPBACK_BASE_URL).post(
            "/notes/synthetic/cancel", content=b"raw", headers=headers
        )
        assert unsigned_cancel.status_code == 200

    assert len(signed_calls) == len(plain_calls) + 3


def test_oversized_signed_route_body_is_rejected_by_middleware_bound(
    signing: ModuleType,
) -> None:
    app, calls, _ = _minimal_app(
        signing, routes=frozenset({("POST", "/notes")}), max_body_bytes=8
    )
    body = b"0123456789"

    response = TestClient(app, base_url=LOOPBACK_BASE_URL).post(
        "/notes", content=iter([body]), headers=_signed("POST", "/notes", body)
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert calls == []


def test_config_repr_never_contains_the_secret(signing: ModuleType) -> None:
    config = signing.RequestSigningConfig(
        routes=frozenset({("POST", "/jobs")}), secret=SECRET
    )

    assert SECRET.decode() not in repr(config)
    assert config.enabled
    assert not signing.RequestSigningConfig().enabled


# Service configuration and wiring.


def _write_secret(
    tmp_path, value: bytes = SECRET + b"\n", *, name: str = "signing.key"
) -> str:
    path = tmp_path / f"SYNTHETIC-PATH-SENTINEL-{name}"
    path.write_bytes(value)
    return str(path)


def test_signing_is_disabled_by_default(signing: ModuleType) -> None:
    from openmed.service.app import create_app

    app = create_app()

    assert app.state.request_signing is None
    assert all(
        middleware.cls is not signing.RequestSigningMiddleware
        for middleware in app.user_middleware
    )
    assert not signing.parse_request_signing_config().enabled


def _startup_failure_cases() -> list[tuple[str, dict[str, Optional[str]]]]:
    return [
        ("routes-without-secret", {"ROUTES": "POST /jobs", "SECRET_FILE": None}),
        ("secret-without-routes", {"ROUTES": None, "SECRET_FILE": "<secret>"}),
        ("missing-file", {"ROUTES": "POST /jobs", "SECRET_FILE": "<missing>"}),
        ("empty-secret", {"ROUTES": "POST /jobs", "SECRET_FILE": "<empty>"}),
        ("short-secret", {"ROUTES": "POST /jobs", "SECRET_FILE": "<short>"}),
        ("directory", {"ROUTES": "POST /jobs", "SECRET_FILE": "<directory>"}),
        ("unknown-route", {"ROUTES": "POST /SYNTHETIC-ROUTE-SENTINEL"}),
        ("unknown-method", {"ROUTES": "PATCH /jobs"}),
        ("malformed-entry", {"ROUTES": "/jobs"}),
        ("websocket-route", {"ROUTES": "WEBSOCKET /graphql"}),
        ("bad-skew", {"ROUTES": "POST /jobs", "MAX_SKEW_SECONDS": "0"}),
        ("bad-cache", {"ROUTES": "POST /jobs", "NONCE_CACHE_SIZE": "zero"}),
    ]


@pytest.mark.parametrize(
    ("label", "settings"),
    _startup_failure_cases(),
    ids=[case[0] for case in _startup_failure_cases()],
)
def test_startup_fails_closed_for_bad_configuration(
    signing: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    label: str,
    settings: dict[str, Optional[str]],
) -> None:
    from openmed.service.app import create_app

    del label
    secret_path = _write_secret(tmp_path)
    resolved = {
        "<secret>": secret_path,
        "<missing>": str(tmp_path / "SYNTHETIC-PATH-SENTINEL-missing.key"),
        "<empty>": _write_secret(tmp_path, b" \n", name="empty.key"),
        "<short>": _write_secret(tmp_path, b"short-secret", name="short.key"),
        "<directory>": str(tmp_path),
    }
    values = {"SECRET_FILE": "<secret>", **settings}
    for suffix, value in values.items():
        name = f"OPENMED_SERVICE_REQUEST_SIGNING_{suffix}"
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, resolved.get(value, value))

    with pytest.raises(ValueError) as raised:
        create_app()

    message = str(raised.value)
    assert "SENTINEL" not in message
    assert str(tmp_path) not in message
    assert raised.value.__cause__ is None


def test_unregistered_policy_route_is_rejected_against_live_app(
    signing: ModuleType,
) -> None:
    app = FastAPI()

    @app.post("/jobs")
    async def create_job() -> dict[str, str]:
        return {}

    config = signing.RequestSigningConfig(
        routes=frozenset({("GET", "/jobs/{job_id}")}), secret=SECRET
    )

    with pytest.raises(ValueError, match="not registered"):
        signing.validate_signed_routes(app, config.routes)


class _SpyJobQueue:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.submitted: list[Any] = []
        self.lookups: list[str] = []

    def submit(self, payload: Any) -> dict[str, Any]:
        self.submitted.append(payload)
        return {"id": "synthetic-job", "status": "queued"}

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        self.lookups.append(job_id)
        return None

    def shutdown(self) -> None:
        return None


def _service_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    routes: str = "POST /jobs,GET /jobs/{job_id}",
    **create_kwargs: Any,
) -> tuple[Any, _SpyJobQueue]:
    from openmed.service.app import create_app
    from openmed.service.runtime import ServiceRuntime

    monkeypatch.setenv("OPENMED_SERVICE_REQUEST_SIGNING_ROUTES", routes)
    monkeypatch.setenv(
        "OPENMED_SERVICE_REQUEST_SIGNING_SECRET_FILE", _write_secret(tmp_path)
    )
    app = create_app(**create_kwargs)
    runtime = ServiceRuntime.from_env(metrics=app.state.metrics)
    app.state.runtime = runtime
    queue = _SpyJobQueue(runtime)
    app.state.job_queue = queue
    monkeypatch.setattr(app.state.request_signing, "clock", _Clock())
    return app, queue


def _job_body(text: str = "Synthetic note") -> bytes:
    return json.dumps(
        {"documents": [{"id": "doc-1", "text": text}]},
        separators=(",", ":"),
    ).encode()


def test_service_enforces_signatures_on_configured_routes(
    signing: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app, queue = _service_app(monkeypatch, tmp_path)
    client = TestClient(app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False)
    body = _job_body()
    json_headers = {"content-type": "application/json"}

    unsigned = client.post("/jobs", content=body, headers=json_headers)
    signed = client.post(
        "/jobs",
        content=body,
        headers={**json_headers, **_signed("POST", "/jobs", body)},
    )
    unsigned_get = client.get("/jobs/synthetic-job")
    signed_get = client.get(
        "/jobs/synthetic-job",
        headers=_signed("GET", "/jobs/synthetic-job", nonce="synthetic-nonce-2"),
    )

    payload = _assert_signature_error(unsigned, "signature_missing")
    assert payload["error"]["request_id"] == unsigned.headers["x-request-id"]
    assert signed.status_code == 202
    assert [item.documents[0].text for item in queue.submitted] == ["Synthetic note"]
    _assert_signature_error(unsigned_get, "signature_missing")
    assert signed_get.status_code == 404
    assert queue.lookups == ["synthetic-job"]
    assert isinstance(app.state.request_signing, signing.RequestSignatureVerifier)


def test_service_unconfigured_routes_are_unchanged(
    signing: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app, queue = _service_app(monkeypatch, tmp_path, routes="POST /jobs")
    client = TestClient(app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False)
    garbage = {SIGNATURE_HEADER: "garbage", NONCE_HEADER: "x", TIMESTAMP_HEADER: "1"}

    for headers in ({}, garbage):
        assert client.get("/livez", headers=headers).status_code == 200
        assert client.get("/jobs/unknown", headers=headers).status_code == 404

    assert queue.lookups == ["unknown", "unknown"]


def test_body_limits_run_before_signature_checks(
    signing: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    app, queue = _service_app(monkeypatch, tmp_path, max_request_body_bytes=64)
    client = TestClient(app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False)
    oversized = _job_body("x" * 128)

    response = client.post(
        "/jobs", content=oversized, headers={"content-type": "application/json"}
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert queue.submitted == []


def test_operational_body_limit_runs_before_signature_checks(
    signing: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_MAX_REQUEST_BYTES", "64")
    app, queue = _service_app(monkeypatch, tmp_path)
    client = TestClient(app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False)

    response = client.post(
        "/jobs",
        content=_job_body("x" * 128),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_size_exceeded"
    assert queue.submitted == []


def test_failures_logs_and_metrics_never_contain_signing_material(
    signing: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    app, queue = _service_app(monkeypatch, tmp_path)
    client = TestClient(app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False)
    body = _job_body(BODY_SENTINEL)
    json_headers = {"content-type": "application/json"}
    valid = _signed("POST", "/jobs", body, nonce=NONCE_SENTINEL)
    stale = _signed("POST", "/jobs", body, nonce=NONCE_SENTINEL + "-2", timestamp=1)
    sentinels = [
        SECRET.decode(),
        BODY_SENTINEL,
        NONCE_SENTINEL,
        valid[SIGNATURE_HEADER].removeprefix("sha256="),
        stale[SIGNATURE_HEADER].removeprefix("sha256="),
        str(tmp_path),
    ]

    responses = []
    with caplog.at_level(logging.DEBUG):
        responses.append(client.post("/jobs", content=body, headers=json_headers))
        responses.append(
            client.post(
                "/jobs",
                content=body + b" ",
                headers={**json_headers, **valid},
            )
        )
        responses.append(
            client.post("/jobs", content=body, headers={**json_headers, **stale})
        )
        responses.append(
            client.post("/jobs", content=body, headers={**json_headers, **valid})
        )
        responses.append(
            client.post("/jobs", content=body, headers={**json_headers, **valid})
        )
        metrics = client.get("/metrics")

    codes = [response.json().get("error", {}).get("code") for response in responses]
    assert codes == [
        "signature_missing",
        "signature_invalid",
        "signature_stale",
        None,
        "signature_replayed",
    ]
    assert len(queue.submitted) == 1
    assert metrics.status_code == 200
    rendered = [
        *(response.text for response in responses),
        *(json.dumps(dict(response.headers)) for response in responses),
        caplog.text,
        metrics.text,
    ]
    for sentinel in sentinels:
        assert all(sentinel not in text for text in rendered)


def test_signature_codes_are_stable_error_categories() -> None:
    from openmed.service.api_compatibility import (
        STABLE_ERROR_CATEGORIES,
        discover_service_error_categories,
    )

    expected = {
        "signature_invalid",
        "signature_missing",
        "signature_replayed",
        "signature_stale",
    }
    assert expected <= set(STABLE_ERROR_CATEGORIES)
    assert expected <= set(discover_service_error_categories())
