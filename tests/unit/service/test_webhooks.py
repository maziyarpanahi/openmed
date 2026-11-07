"""Tests for signed async job webhook delivery."""

from __future__ import annotations

import json

import httpx
import pytest

from openmed.service.signing import (
    ATTEMPT_HEADER,
    DELIVERY_ID_HEADER,
    EVENT_HEADER,
    NONCE_HEADER,
    WEBHOOK_SIGNATURE_HEADER,
    NonceCache,
    SignedWebhookHeaders,
    parse_webhook_headers,
    sign_request,
    verify_request_signature,
    webhook_signature_digest,
    webhook_signing_string,
)
from openmed.service.signing_keys import (
    SigningKeyError,
    SigningKeyring,
    derive_key_id,
)
from openmed.service.webhooks import (
    DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    canonical_json_bytes,
    deliver_webhook,
    sign_webhook_payload,
)


def test_sign_webhook_payload_uses_canonical_json_and_timestamp() -> None:
    payload = {"status": "done", "job_id": "abc", "label_histogram": {"NAME": 1}}

    body, timestamp, nonce, signature = sign_webhook_payload(
        payload,
        secret="secret",
        path="/openmed?source=jobs",
        timestamp=1_800_000_000,
        nonce="webhook-nonce",
    )

    assert body == canonical_json_bytes(payload)
    assert timestamp == "1800000000"
    assert nonce == "webhook-nonce"
    expected = sign_request(
        "POST",
        "/openmed?source=jobs",
        body,
        secret="secret",
        timestamp=1_800_000_000,
        nonce="webhook-nonce",
    )
    assert signature == expected[SIGNATURE_HEADER]


def test_deliver_webhook_retries_with_backoff_until_success() -> None:
    attempts = 0
    delays: list[float] = []
    seen_nonces: list[str] = []
    seen_request: httpx.Request | None = None
    nonce_cache = NonceCache()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        nonlocal seen_request
        attempts += 1
        seen_request = request
        seen_nonces.append(request.headers[NONCE_HEADER])
        assert verify_request_signature(
            request.method,
            request.url.raw_path.decode("ascii"),
            request.content,
            request.headers,
            secret="secret",
            nonce_cache=nonce_cache,
            now=int(request.headers[TIMESTAMP_HEADER]),
        )
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = deliver_webhook(
        "https://callbacks.example.test/openmed",
        {"event": "job.done", "job_id": "abc", "status": "done"},
        secret="secret",
        max_attempts=3,
        backoff_seconds=0.25,
        client=client,
        sleeper=delays.append,
    )

    assert result.success is True
    assert result.attempts == 2
    assert result.status_code == 204
    assert delays == [0.25]
    assert seen_request is not None
    assert seen_request.headers[SIGNATURE_HEADER].startswith("sha256=")
    assert seen_request.headers[TIMESTAMP_HEADER]
    assert seen_request.headers[NONCE_HEADER]
    assert len(seen_nonces) == len(set(seen_nonces))


def test_deliver_webhook_reports_failure_after_attempts() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, json={"error": "temporary"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = deliver_webhook(
        "https://callbacks.example.test/openmed",
        {"event": "job.failed", "job_id": "abc", "status": "failed"},
        secret="secret",
        max_attempts=2,
        backoff_seconds=0,
        client=client,
    )

    assert calls == 2
    assert result.success is False
    assert result.attempts == 2
    assert result.status_code == 500
    assert result.error == "HTTP 500"


def test_deliver_webhook_records_value_free_transport_error_codes() -> None:
    sentinel_url = "https://zqtransportsentinel.example.test/hook?token=zq-secret"
    for error, code in [
        (httpx.ConnectError(f"cannot reach {sentinel_url}"), "transport_error"),
        (httpx.ReadTimeout(f"timed out reading {sentinel_url}"), "timeout"),
    ]:

        def handler(_: httpx.Request, error: Exception = error) -> httpx.Response:
            raise error

        client = httpx.Client(transport=httpx.MockTransport(handler))
        result = deliver_webhook(
            sentinel_url,
            {"event": "job.done", "job_id": "abc", "status": "done"},
            secret="secret",
            max_attempts=1,
            backoff_seconds=0,
            client=client,
        )

        assert result.success is False
        assert result.status_code is None
        assert result.error == code
        assert "zq" not in json.dumps(result.to_dict())


def test_canonical_json_bytes_is_stable() -> None:
    payload = {"b": 2, "a": {"d": 4, "c": 3}}

    assert canonical_json_bytes(payload) == json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _timeout_extensions(seconds: float) -> dict[str, float]:
    return {"connect": seconds, "read": seconds, "write": seconds, "pool": seconds}


def _recording_transport(
    seen: list[dict[str, float]],
    statuses: list[int] | None = None,
) -> httpx.MockTransport:
    remaining = list(statuses or [204])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        status = remaining.pop(0) if remaining else 204
        return httpx.Response(status)

    return httpx.MockTransport(handler)


def test_deliver_webhook_applies_timeout_to_a_supplied_client() -> None:
    seen: list[dict[str, float]] = []
    with httpx.Client(
        timeout=None,
        transport=_recording_transport(seen),
    ) as client:
        result = deliver_webhook(
            "https://callbacks.example.test/openmed",
            {"event": "job.done", "job_id": "abc", "status": "done"},
            secret="secret",
            client=client,
            timeout_seconds=0.5,
            max_attempts=1,
        )

        assert result.success is True
        assert seen == [_timeout_extensions(0.5)]
        assert client.is_closed is False


def test_deliver_webhook_timeout_overrides_the_supplied_client_default() -> None:
    seen: list[dict[str, float]] = []
    with httpx.Client(
        timeout=httpx.Timeout(30.0),
        transport=_recording_transport(seen),
    ) as client:
        result = deliver_webhook(
            "https://callbacks.example.test/openmed",
            {"event": "job.done", "job_id": "abc", "status": "done"},
            secret="secret",
            client=client,
            timeout_seconds=1.25,
            max_attempts=1,
        )

        assert result.success is True
        assert seen == [_timeout_extensions(1.25)]
        assert client.timeout.connect == 30.0
        assert client.timeout.read == 30.0


def test_deliver_webhook_applies_timeout_to_every_attempt() -> None:
    seen: list[dict[str, float]] = []
    client = httpx.Client(
        timeout=None,
        transport=_recording_transport(seen, statuses=[503, 204]),
    )
    result = deliver_webhook(
        "https://callbacks.example.test/openmed",
        {"event": "job.done", "job_id": "abc", "status": "done"},
        secret="secret",
        client=client,
        timeout_seconds=0.25,
        max_attempts=2,
        backoff_seconds=0,
    )

    assert result.success is True
    assert result.attempts == 2
    assert seen == [_timeout_extensions(0.25), _timeout_extensions(0.25)]


def test_deliver_webhook_uses_the_default_timeout_when_unspecified() -> None:
    seen: list[dict[str, float]] = []
    with httpx.Client(
        timeout=None,
        transport=_recording_transport(seen),
    ) as client:
        deliver_webhook(
            "https://callbacks.example.test/openmed",
            {"event": "job.done", "job_id": "abc", "status": "done"},
            secret="secret",
            client=client,
            max_attempts=1,
        )

        assert seen == [_timeout_extensions(DEFAULT_WEBHOOK_TIMEOUT_SECONDS)]


def test_deliver_webhook_leaves_the_supplied_client_ownership_alone() -> None:
    seen: list[dict[str, float]] = []
    client = httpx.Client(
        timeout=httpx.Timeout(30.0),
        transport=_recording_transport(seen),
    )
    deliver_webhook(
        "https://callbacks.example.test/openmed",
        {"event": "job.done", "job_id": "abc", "status": "done"},
        secret="secret",
        client=client,
        timeout_seconds=0.5,
        max_attempts=1,
    )

    assert client.is_closed is False
    assert client.timeout.connect == 30.0
    client.close()


# --- Version 2 delivery contract (#3573) -----------------------------------

V2_SECRET_SENTINEL = "zq-v2-webhook-secret-sentinel-0123456789"
V2_URL = "https://zqcallbacksentinel.example.test/openmed?source=jobs"


class _MutableResolver:
    """Key resolver double whose generation can change between attempts."""

    def __init__(self, keyring: SigningKeyring) -> None:
        self.keyring = keyring
        self.calls = 0

    def current(self) -> SigningKeyring:
        self.calls += 1
        return self.keyring


class _BrokenResolver:
    def current(self) -> SigningKeyring:
        raise SigningKeyError("keyring_unreadable")


def _verify_v2(request: httpx.Request, secret: str) -> SignedWebhookHeaders:
    signed = parse_webhook_headers(request.headers)
    expected = webhook_signature_digest(
        secret,
        webhook_signing_string(
            request.method,
            request.url.raw_path.decode("ascii"),
            signed.timestamp,
            signed.nonce,
            request.content,
            key_id=signed.signature.key_id,
            event=signed.event,
            delivery_id=signed.delivery_id,
            attempt=signed.attempt,
        ),
    )
    assert expected == signed.signature.digest
    return signed


def test_retries_keep_the_delivery_id_with_fresh_attempt_identity() -> None:
    seen: list[SignedWebhookHeaders] = []
    statuses = [503, 502, 204]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(_verify_v2(request, V2_SECRET_SENTINEL))
        return httpx.Response(statuses[len(seen) - 1])

    result = deliver_webhook(
        V2_URL,
        {"event": "job.done", "job_id": "abc", "status": "done"},
        secret=V2_SECRET_SENTINEL,
        max_attempts=3,
        backoff_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: 1_800_000_000.75,
    )

    assert result.success is True
    assert {item.delivery_id for item in seen} == {result.delivery_id}
    assert result.delivery_id is not None and result.delivery_id.startswith("dlv_")
    assert [item.attempt for item in seen] == [1, 2, 3]
    assert len({item.nonce for item in seen}) == 3
    assert {item.event for item in seen} == {"job.done"}
    assert {item.timestamp for item in seen} == {1_800_000_000}
    assert {item.signature.version for item in seen} == {"v2"}
    assert {item.signature.key_id for item in seen} == {
        derive_key_id(V2_SECRET_SENTINEL)
    }
    assert [receipt.status_code for receipt in result.attempt_receipts] == [
        503,
        502,
        204,
    ]
    assert [receipt.error for receipt in result.attempt_receipts] == [
        "HTTP 503",
        "HTTP 502",
        None,
    ]
    summary = json.dumps(result.to_dict())
    assert result.to_dict()["signature_version"] == "v2"
    assert V2_SECRET_SENTINEL not in summary
    assert "zqcallbacksentinel" not in summary


def test_an_explicit_delivery_id_and_key_id_are_reused_verbatim() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(500)

    result = deliver_webhook(
        V2_URL,
        {"event": "job.failed", "job_id": "abc", "status": "failed"},
        secret=V2_SECRET_SENTINEL,
        key_id="callbacks-2026-11",
        delivery_id="dlv_synthetic_fixed_0001",
        max_attempts=2,
        backoff_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result.success is False
    assert result.delivery_id == "dlv_synthetic_fixed_0001"
    for request in seen:
        signed = _verify_v2(request, V2_SECRET_SENTINEL)
        assert signed.delivery_id == "dlv_synthetic_fixed_0001"
        assert signed.signature.key_id == "callbacks-2026-11"
        assert request.headers[EVENT_HEADER] == "job.failed"
        assert request.headers[DELIVERY_ID_HEADER] == "dlv_synthetic_fixed_0001"
    assert [request.headers[ATTEMPT_HEADER] for request in seen] == ["1", "2"]


def test_a_retry_after_rotation_signs_with_the_newly_active_key() -> None:
    old_secret = "o" * 32
    new_secret = "n" * 32
    resolver = _MutableResolver(SigningKeyring.single(old_secret, key_id="k-old"))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        signed = parse_webhook_headers(request.headers)
        seen.append(signed.signature.key_id)
        secret = old_secret if signed.signature.key_id == "k-old" else new_secret
        _verify_v2(request, secret)
        # Rotate between attempt 1 and attempt 2.
        resolver.keyring = SigningKeyring.single(
            new_secret, key_id="k-new", generation=2
        )
        return httpx.Response(503 if len(seen) == 1 else 204)

    result = deliver_webhook(
        V2_URL,
        {"event": "job.done", "job_id": "abc", "status": "done"},
        keyring=resolver,
        max_attempts=2,
        backoff_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result.success is True
    assert seen == ["k-old", "k-new"]
    assert [receipt.key_id for receipt in result.attempt_receipts] == [
        "k-old",
        "k-new",
    ]
    assert resolver.calls == 2


def test_an_unavailable_signing_key_sends_nothing_and_records_a_code() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(204)

    result = deliver_webhook(
        V2_URL,
        {"event": "job.done", "job_id": "abc", "status": "done"},
        keyring=_BrokenResolver(),
        max_attempts=2,
        backoff_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert calls == 0
    assert result.success is False
    assert result.attempts == 2
    assert result.error == "signing_key_unavailable"
    assert [receipt.key_id for receipt in result.attempt_receipts] == [None, None]


def test_the_legacy_v1_header_can_be_turned_off() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    deliver_webhook(
        V2_URL,
        {"event": "job.done", "job_id": "abc", "status": "done"},
        secret=V2_SECRET_SENTINEL,
        include_legacy_signature=False,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert SIGNATURE_HEADER not in seen[0].headers
    assert WEBHOOK_SIGNATURE_HEADER in seen[0].headers
    _verify_v2(seen[0], V2_SECRET_SENTINEL)


def test_deliver_webhook_requires_exactly_one_key_source() -> None:
    payload = {"event": "job.done", "job_id": "abc", "status": "done"}
    keyring = SigningKeyring.single(V2_SECRET_SENTINEL)
    with pytest.raises(ValueError):
        deliver_webhook(V2_URL, payload)
    with pytest.raises(ValueError):
        deliver_webhook(V2_URL, payload, secret=V2_SECRET_SENTINEL, keyring=keyring)
    with pytest.raises(ValueError):
        deliver_webhook(V2_URL, payload, keyring=keyring, key_id="k1")
    with pytest.raises(ValueError):
        deliver_webhook(
            V2_URL, payload, secret=V2_SECRET_SENTINEL, delivery_id="bad id"
        )
    with pytest.raises(ValueError):
        deliver_webhook(V2_URL, {"event": "bad event"}, secret=V2_SECRET_SENTINEL)
