"""Inbound host, body-size, auth and work-bound checks for the privacy proxy."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openmed.service.auth import APIKeyCredential, ServiceAuthConfig, hash_api_key
from openmed.service.privacy_proxy.app import create_app

SYNTHETIC_NAME = "Avery Example"
SENTINEL = "SYNTHETIC-PROXY-SENTINEL-7731"
LOOPBACK_BASE_URL = "http://127.0.0.1"
REBOUND_HOST = "rebound.attacker.example.test"
VALID_KEY = "synthetic-proxy-key-0001"
UNSCOPED_KEY = "synthetic-proxy-key-0002"
HS_SECRET = b"synthetic-proxy-hs256-secret-0001"
PROXY_ENV_VARS = (
    "OPENMED_PRIVACY_PROXY_TRUSTED_HOSTS",
    "OPENMED_PRIVACY_PROXY_MAX_REQUEST_BYTES",
    "OPENMED_PRIVACY_PROXY_MAX_MESSAGES",
    "OPENMED_PRIVACY_PROXY_MAX_TEXT_CHARS",
    "OPENMED_PRIVACY_PROXY_MAX_STREAM_SECONDS",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_JWKS",
    "OPENMED_SERVICE_AUTH_JWKS_FILE",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_MAX_REQUEST_BYTES",
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class _Recorder:
    """Synthetic extractor and transport that record calls without network."""

    def __init__(self) -> None:
        self.extractor_calls = 0
        self.transport_calls = 0

    def extractor(self, text: str, **_: Any) -> list[dict[str, Any]]:
        self.extractor_calls += 1
        if SYNTHETIC_NAME not in text:
            return []
        start = text.index(SYNTHETIC_NAME)
        return [
            {
                "label": "NAME",
                "start": start,
                "end": start + len(SYNTHETIC_NAME),
                "confidence": 0.99,
            }
        ]

    def transport(self, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.transport_calls += 1
        content = payload["messages"][0]["content"]
        return {"content": content if isinstance(content, str) else "ok"}

    @property
    def untouched(self) -> bool:
        return self.extractor_calls == 0 and self.transport_calls == 0


def _app(recorder: _Recorder, transport: Any = None, **kwargs: Any) -> Any:
    return create_app(
        transport=transport or recorder.transport,
        extractor=recorder.extractor,
        tripwire_extractor=lambda *_args, **_kwargs: [],
        **kwargs,
    )


def _client(app: Any, base_url: str = LOOPBACK_BASE_URL) -> TestClient:
    return TestClient(app, base_url=base_url)


def _request(content: str = f"Contact {SYNTHETIC_NAME}", **extra: Any) -> dict:
    return {
        "model": "local-fixture",
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


def _invalid_json_with_sentinel() -> bytes:
    # A body that would fail JSON parsing proves which check ran first.
    return f'{{"messages": [{{"role": "user", "content": "{SENTINEL}"'.encode()


def _assert_value_free(response: Any, caplog: pytest.LogCaptureFixture) -> None:
    for marker in (SENTINEL, SYNTHETIC_NAME, "OPENMED_PHI", "mapping"):
        assert marker not in response.text
        assert marker not in caplog.text
    for value in response.headers.values():
        assert SENTINEL not in value


def _error(response: Any) -> dict[str, Any]:
    return response.json()["error"]


def _auth_config(**overrides: Any) -> ServiceAuthConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "api_keys": (
            APIKeyCredential(
                key_id="proxy-key",
                key_hash=hash_api_key(VALID_KEY),
                principal="proxy-client",
                scopes=("pii:write",),
            ),
            APIKeyCredential(
                key_id="unscoped-key",
                key_hash=hash_api_key(UNSCOPED_KEY),
                principal="unscoped-client",
                scopes=(),
            ),
        ),
        "failure_rate_limit_rps": 0.0,
    }
    values.update(overrides)
    return ServiceAuthConfig(**values)


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _jwt_hs256(claims: dict[str, Any]) -> str:
    header = {"alg": "HS256", "kid": "proxy-hs", "typ": "JWT"}
    segments = [
        _b64url(json.dumps(part, separators=(",", ":")).encode())
        for part in (header, claims)
    ]
    signing_input = ".".join(segments)
    signature = hmac.new(
        HS_SECRET, signing_input.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64url(signature)}"


def _stream_events(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")
    ]


def _stream_content(text: str) -> str:
    return "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in _stream_events(text)
        if "choices" in event
    )


# Trusted host ----------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        REBOUND_HOST,
        f"{REBOUND_HOST}:8081",
        "127.0.0.1.rebound.attacker.example.test",
        "localhost.attacker.example.test",
    ],
)
def test_untrusted_host_is_rejected_before_parsing_or_transport(
    host: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    recorder = _Recorder()
    with _client(_app(recorder)) as client:
        response = client.post(
            "/v1/chat/completions",
            content=_invalid_json_with_sentinel(),
            headers={"host": host, "content-type": "application/json"},
        )

    assert response.status_code == 400
    assert _error(response)["code"] == "bad_request"
    assert recorder.untouched
    _assert_value_free(response, caplog)


def test_untrusted_host_with_valid_request_never_reaches_transport(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    recorder = _Recorder()
    with _client(_app(recorder), base_url=f"http://{REBOUND_HOST}") as client:
        response = client.post(
            "/v1/chat/completions", json=_request(f"{SENTINEL} {SYNTHETIC_NAME}")
        )
        health = client.get("/health")

    assert response.status_code == 400
    assert health.status_code == 400
    assert recorder.untouched
    _assert_value_free(response, caplog)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1",
        "http://127.0.0.1:8081",
        "http://localhost:8081",
        "http://[::1]:8081",
    ],
)
def test_loopback_hosts_are_trusted_by_default(base_url: str) -> None:
    recorder = _Recorder()
    with _client(_app(recorder), base_url=base_url) as client:
        response = client.post("/v1/chat/completions", json=_request())

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == f"Contact {SYNTHETIC_NAME}"
    assert recorder.transport_calls == 1


def test_trusted_hosts_can_be_set_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENMED_PRIVACY_PROXY_TRUSTED_HOSTS", " proxy.internal.example.test , "
    )
    recorder = _Recorder()
    app = _app(recorder)
    with _client(app, base_url="http://proxy.internal.example.test") as client:
        allowed = client.post("/v1/chat/completions", json=_request())
    with _client(app) as client:
        loopback = client.post("/v1/chat/completions", json=_request())

    assert allowed.status_code == 200
    assert loopback.status_code == 400
    assert recorder.transport_calls == 1


def test_trusted_hosts_argument_overrides_the_default() -> None:
    recorder = _Recorder()
    app = _app(recorder, trusted_hosts=("proxy.internal.example.test",))
    with _client(app, base_url="http://proxy.internal.example.test") as client:
        allowed = client.post("/v1/chat/completions", json=_request())
    with _client(app) as client:
        loopback = client.post("/v1/chat/completions", json=_request())

    assert allowed.status_code == 200
    assert loopback.status_code == 400


def test_global_wildcard_trusted_host_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    with pytest.raises(ValueError, match="OPENMED_PRIVACY_PROXY_TRUSTED_HOSTS"):
        _app(recorder, trusted_hosts=("*",))
    with pytest.raises(ValueError, match="OPENMED_PRIVACY_PROXY_TRUSTED_HOSTS"):
        _app(recorder, trusted_hosts=())
    monkeypatch.setenv("OPENMED_PRIVACY_PROXY_TRUSTED_HOSTS", "localhost,*")
    with pytest.raises(ValueError, match="OPENMED_PRIVACY_PROXY_TRUSTED_HOSTS"):
        _app(recorder)


# Body size -------------------------------------------------------------------


def test_default_body_limit_rejects_twenty_megabytes_before_parsing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    recorder = _Recorder()
    body = _invalid_json_with_sentinel() + b"x" * (20 * 1024 * 1024)
    with _client(_app(recorder)) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert _error(response)["code"] == "request_size_exceeded"
    assert recorder.untouched
    _assert_value_free(response, caplog)


def test_body_limit_from_environment_applies_to_chunked_bodies(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PRIVACY_PROXY_MAX_REQUEST_BYTES", "1024")
    recorder = _Recorder()

    def chunks():
        yield _invalid_json_with_sentinel()
        for _ in range(8):
            yield b"x" * 256

    with _client(_app(recorder)) as client:
        response = client.post(
            "/v1/chat/completions",
            content=chunks(),
            headers={"content-type": "application/json"},
        )
        small = client.post("/v1/chat/completions", json=_request())

    assert response.status_code == 413
    assert _error(response)["code"] == "request_size_exceeded"
    assert small.status_code == 200
    assert recorder.transport_calls == 1
    _assert_value_free(response, caplog)


def test_operational_limits_argument_bounds_the_body() -> None:
    from openmed.guard.operational_limits import OperationalLimits

    recorder = _Recorder()
    app = _app(recorder, operational_limits=OperationalLimits(max_request_bytes=64))
    with _client(app) as client:
        response = client.post(
            "/v1/chat/completions", json=_request(SENTINEL + "x" * 128)
        )

    assert response.status_code == 413
    assert recorder.untouched


# Authentication --------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "status_code", "code"),
    [
        ({}, 401, "authentication_required"),
        ({"x-api-key": "synthetic-wrong-key"}, 401, "invalid_credentials"),
        ({"authorization": "Bearer not.a.jwt"}, 401, "invalid_credentials"),
        ({"authorization": "Basic c3ludGhldGlj"}, 401, "invalid_credentials"),
        ({"x-api-key": UNSCOPED_KEY}, 403, "forbidden"),
    ],
)
def test_configured_auth_rejects_before_parsing_or_transport(
    headers: dict[str, str],
    status_code: int,
    code: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    recorder = _Recorder()
    with _client(_app(recorder, auth_config=_auth_config())) as client:
        response = client.post(
            "/v1/chat/completions",
            content=_invalid_json_with_sentinel(),
            headers={"content-type": "application/json", **headers},
        )
        compatibility = client.post(
            "/chat/completions",
            content=_invalid_json_with_sentinel(),
            headers={"content-type": "application/json", **headers},
        )

    for item in (response, compatibility):
        assert item.status_code == status_code
        assert _error(item)["code"] == code
        _assert_value_free(item, caplog)
    assert recorder.untouched


def test_configured_auth_accepts_valid_api_key_and_keeps_health_open() -> None:
    recorder = _Recorder()
    with _client(_app(recorder, auth_config=_auth_config())) as client:
        health = client.get("/health")
        response = client.post(
            "/v1/chat/completions",
            json=_request(),
            headers={"x-api-key": VALID_KEY},
        )

    assert health.status_code == 200
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == f"Contact {SYNTHETIC_NAME}"
    assert recorder.transport_calls == 1


def test_configured_auth_accepts_valid_jwt_bearer_token() -> None:
    jwk = {"kty": "oct", "kid": "proxy-hs", "alg": "HS256", "k": _b64url(HS_SECRET)}
    now = int(time.time())
    token = _jwt_hs256(
        {
            "sub": "proxy-client",
            "scope": "pii:write",
            "iat": now,
            "exp": now + 300,
        }
    )
    recorder = _Recorder()
    app = _app(recorder, auth_config=_auth_config(api_keys=(), jwks=(jwk,)))
    with _client(app) as client:
        valid = client.post(
            "/v1/chat/completions",
            json=_request(),
            headers={"authorization": f"Bearer {token}"},
        )
        tampered = client.post(
            "/v1/chat/completions",
            json=_request(),
            headers={"authorization": f"Bearer {token[:-2]}AA"},
        )

    assert valid.status_code == 200
    assert tampered.status_code == 401
    assert recorder.transport_calls == 1


def test_shared_service_auth_environment_enables_proxy_auth(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": "proxy-key",
                    "key_hash": hash_api_key(VALID_KEY),
                    "scopes": ["pii:write"],
                }
            ]
        ),
    )
    recorder = _Recorder()
    with _client(_app(recorder)) as client:
        missing = client.post(
            "/v1/chat/completions", json=_request(f"{SENTINEL} {SYNTHETIC_NAME}")
        )
        valid = client.post(
            "/v1/chat/completions",
            json=_request(),
            headers={"x-api-key": VALID_KEY},
        )

    assert missing.status_code == 401
    assert _error(missing)["code"] == "authentication_required"
    assert valid.status_code == 200
    assert recorder.transport_calls == 1
    _assert_value_free(missing, caplog)


def test_proxy_routes_require_auth_even_when_deny_by_default_is_off() -> None:
    recorder = _Recorder()
    app = _app(recorder, auth_config=_auth_config(deny_by_default=False))
    with _client(app) as client:
        response = client.post("/v1/chat/completions", json=_request())

    assert response.status_code == 401
    assert recorder.untouched


def test_auth_is_off_by_default() -> None:
    recorder = _Recorder()
    with _client(_app(recorder)) as client:
        response = client.post("/v1/chat/completions", json=_request())

    assert response.status_code == 200


# Message, text and stream bounds ---------------------------------------------


def test_message_count_bound_returns_controlled_code_without_transport(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PRIVACY_PROXY_MAX_MESSAGES", "2")
    recorder = _Recorder()
    payload = {
        "model": "local-fixture",
        "messages": [
            {"role": "user", "content": f"{SENTINEL} {SYNTHETIC_NAME}"}
            for _ in range(3)
        ],
    }
    with _client(_app(recorder)) as client:
        response = client.post("/v1/chat/completions", json=payload)
        allowed = client.post(
            "/v1/chat/completions",
            json={**payload, "messages": payload["messages"][:2]},
        )

    assert response.status_code == 413
    assert _error(response)["code"] == "privacy_proxy_limit_exceeded"
    assert _error(response)["details"] == {"reason": "message_count_exceeded"}
    assert allowed.status_code == 200
    assert recorder.transport_calls == 1
    assert recorder.extractor_calls == 2
    _assert_value_free(response, caplog)


def test_total_text_character_bound_counts_every_redacted_leaf(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PRIVACY_PROXY_MAX_TEXT_CHARS", "60")
    recorder = _Recorder()
    first = f"{SENTINEL} {SYNTHETIC_NAME}"
    second = "y" * (60 - len(first) + 1)
    payload = {
        "model": "local-fixture",
        "messages": [
            {"role": "user", "content": first},
            {"role": "user", "content": [{"type": "text", "text": second}]},
        ],
    }
    with _client(_app(recorder)) as client:
        response = client.post("/v1/chat/completions", json=payload)
        payload["messages"][1]["content"][0]["text"] = second[:-1]
        at_limit = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 413
    assert _error(response)["code"] == "privacy_proxy_limit_exceeded"
    assert _error(response)["details"] == {"reason": "text_length_exceeded"}
    assert at_limit.status_code == 200
    assert recorder.transport_calls == 1
    _assert_value_free(response, caplog)


def test_limits_argument_bounds_messages_and_text() -> None:
    from openmed.service.privacy_proxy.app import PrivacyProxyLimits

    recorder = _Recorder()
    app = _app(recorder, limits=PrivacyProxyLimits(max_messages=1, max_text_chars=8))
    with _client(app) as client:
        too_many = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                ]
            },
        )
        too_long = client.post("/v1/chat/completions", json=_request("z" * 9))

    assert _error(too_many)["details"]["reason"] == "message_count_exceeded"
    assert _error(too_long)["details"]["reason"] == "text_length_exceeded"
    assert recorder.untouched


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENMED_PRIVACY_PROXY_MAX_MESSAGES", "0"),
        ("OPENMED_PRIVACY_PROXY_MAX_MESSAGES", "many"),
        ("OPENMED_PRIVACY_PROXY_MAX_TEXT_CHARS", "-1"),
        ("OPENMED_PRIVACY_PROXY_MAX_STREAM_SECONDS", "0"),
        ("OPENMED_PRIVACY_PROXY_MAX_STREAM_SECONDS", "nan"),
        ("OPENMED_PRIVACY_PROXY_MAX_REQUEST_BYTES", "large"),
    ],
)
def test_invalid_limit_settings_fail_closed(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name) as raised:
        _app(_Recorder())
    assert f"={value}" not in str(raised.value)


def test_async_stream_duration_bound_stops_a_stalled_transport(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PRIVACY_PROXY_MAX_STREAM_SECONDS", "0.2")
    recorder = _Recorder()
    closed: list[bool] = []

    async def transport(payload: dict[str, Any], **_: Any):
        recorder.transport_calls += 1
        try:
            yield "partial "
            await asyncio.sleep(30)
            yield payload["messages"][0]["content"]
        finally:
            closed.append(True)

    started = time.monotonic()
    with _client(_app(recorder, transport=transport)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_request(f"{SENTINEL} {SYNTHETIC_NAME}", stream=True),
        )
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert elapsed < 10
    assert '"code":"privacy_proxy_timeout"' in response.text
    assert '"reason":"stream_duration_exceeded"' in response.text
    assert "data: [DONE]" not in response.text
    assert _stream_content(response.text) == "partial "
    assert closed == [True]
    _assert_value_free(response, caplog)


def test_sync_stream_duration_bound_is_checked_between_chunks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PRIVACY_PROXY_MAX_STREAM_SECONDS", "0.05")
    recorder = _Recorder()
    produced: list[int] = []

    def transport(payload: dict[str, Any], **_: Any):
        recorder.transport_calls += 1
        for index in range(50):
            produced.append(index)
            time.sleep(0.02)
            yield payload["messages"][0]["content"] if index == 3 else "."

    with _client(_app(recorder, transport=transport)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_request(f"{SENTINEL} {SYNTHETIC_NAME}", stream=True),
        )

    assert response.status_code == 200
    assert '"reason":"stream_duration_exceeded"' in response.text
    assert "data: [DONE]" not in response.text
    assert len(produced) < 50
    _assert_value_free(response, caplog)


# Unchanged loopback behavior -------------------------------------------------


def test_loopback_streaming_restores_placeholders_with_auth_and_limits() -> None:
    from openmed.service.privacy_proxy.app import PrivacyProxyLimits

    recorder = _Recorder()
    seen: list[str] = []

    async def transport(payload: dict[str, Any], **_: Any):
        recorder.transport_calls += 1
        redacted = payload["messages"][0]["content"]
        seen.append(redacted)
        token = redacted.split(" ", 1)[-1]
        yield "The contact is "
        yield token[: len(token) // 2]
        yield token[len(token) // 2 :]

    app = _app(
        recorder,
        transport=transport,
        auth_config=_auth_config(),
        limits=PrivacyProxyLimits(max_messages=4, max_text_chars=256),
    )
    with _client(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_request(stream=True),
            headers={"x-api-key": VALID_KEY},
        )

    assert response.status_code == 200
    assert response.text.rstrip().endswith("data: [DONE]")
    assert _stream_content(response.text) == f"The contact is {SYNTHETIC_NAME}"
    assert SYNTHETIC_NAME not in seen[0]
    assert "OPENMED_PHI" in seen[0]
    assert recorder.transport_calls == 1


def test_proxy_security_settings_are_exposed_without_secrets() -> None:
    recorder = _Recorder()
    app = _app(recorder, auth_config=_auth_config())

    assert app.state.trusted_hosts == ("localhost", "127.0.0.1", "[::1]")
    assert app.state.operational_limits.max_request_bytes == 16 * 1024 * 1024
    assert app.state.auth.enabled is True
    assert VALID_KEY not in repr(app.state.auth.config)
