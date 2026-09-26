"""Tests for signed async job webhook delivery."""

from __future__ import annotations

import json

import httpx

from openmed.service.signing import (
    NONCE_HEADER,
    NonceCache,
    sign_request,
    verify_request_signature,
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
