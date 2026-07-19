"""Tests for HMAC request signing and local replay protection."""

from __future__ import annotations

import json

import httpx
import pytest

from openmed.service.signing import (
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    InvalidSignatureError,
    NonceCache,
    NonceCacheFullError,
    ReplayDetectedError,
    StaleTimestampError,
    canonical_signing_string,
    sign_request,
    verify_request_signature,
)
from openmed.service.webhooks import deliver_webhook


def test_signed_request_verifies_and_rejects_tampering_and_replay() -> None:
    body = b'{"document_hash":"sha256:abc","offsets":[[4,12]]}'
    headers = sign_request(
        "POST",
        "/jobs?mode=async",
        body,
        secret="shared-secret",
        timestamp=1_800_000_000,
        nonce="synthetic-request-1",
    )
    cache = NonceCache(max_entries=10, window_seconds=300)

    with pytest.raises(InvalidSignatureError):
        verify_request_signature(
            "POST",
            "/jobs?mode=async",
            body + b" ",
            headers,
            secret="shared-secret",
            nonce_cache=cache,
            now=1_800_000_000,
        )

    assert verify_request_signature(
        "POST",
        "/jobs?mode=async",
        body,
        headers,
        secret="shared-secret",
        nonce_cache=cache,
        now=1_800_000_000,
    )

    with pytest.raises(ReplayDetectedError):
        verify_request_signature(
            "POST",
            "/jobs?mode=async",
            body,
            headers,
            secret="shared-secret",
            nonce_cache=cache,
            now=1_800_000_001,
        )


def test_stale_timestamp_is_rejected_without_consuming_nonce() -> None:
    headers = sign_request(
        "GET",
        "/jobs/abc",
        secret="shared-secret",
        timestamp=1_800_000_000,
        nonce="stale-request",
    )
    cache = NonceCache(window_seconds=30)

    with pytest.raises(StaleTimestampError):
        verify_request_signature(
            "GET",
            "/jobs/abc",
            b"",
            headers,
            secret="shared-secret",
            nonce_cache=cache,
            max_skew_seconds=30,
            now=1_800_000_031,
        )

    assert len(cache) == 0


def test_canonical_string_contains_body_hash_not_raw_body() -> None:
    body = b"Patient Maria Garcia"

    canonical = canonical_signing_string(
        "post",
        "/pii/deidentify",
        1_800_000_000,
        "privacy-safe-nonce",
        body,
    )

    assert b"Patient" not in canonical
    assert b"Maria" not in canonical
    assert canonical.startswith(b"OPENMED-HMAC-SHA256-V1\nPOST\n")
    assert canonical.endswith(
        b"sha256:96d954518f3d24df5c23dfe61bb203c80c41561ab480a0b198d3125320deed1e"
    )


def test_nonce_cache_is_bounded_and_evicts_only_past_window() -> None:
    cache = NonceCache(max_entries=2, window_seconds=10)

    assert cache.check_and_store("nonce-a", timestamp=100, now=100)
    assert cache.check_and_store("nonce-b", timestamp=101, now=101)
    with pytest.raises(NonceCacheFullError):
        cache.check_and_store("nonce-c", timestamp=102, now=102)
    assert len(cache) == 2

    assert cache.purge_expired(now=111) == 1
    assert len(cache) == 1
    assert cache.check_and_store("nonce-c", timestamp=111, now=111)
    assert len(cache) == 2


def test_outbound_webhook_carries_a_verifiable_signature() -> None:
    cache = NonceCache()
    verified = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal verified
        verified = verify_request_signature(
            request.method,
            request.url.raw_path.decode("ascii"),
            request.content,
            request.headers,
            secret="webhook-secret",
            nonce_cache=cache,
            now=int(request.headers[TIMESTAMP_HEADER]),
        )
        assert request.headers[NONCE_HEADER]
        assert request.headers[SIGNATURE_HEADER].startswith("sha256=")
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = deliver_webhook(
        "https://callbacks.example.test/openmed/jobs?source=openmed",
        {"event": "job.done", "job_id": "abc", "status": "done"},
        secret="webhook-secret",
        client=client,
    )

    assert result.success is True
    assert verified is True


def test_client_signer_returns_only_the_three_signature_headers() -> None:
    headers = sign_request(
        "POST",
        "/jobs",
        json.dumps({"document_hash": "sha256:abc"}).encode(),
        secret=b"shared-secret",
        timestamp=1_800_000_000,
        nonce="client-nonce",
    )

    assert headers == {
        TIMESTAMP_HEADER: "1800000000",
        NONCE_HEADER: "client-nonce",
        SIGNATURE_HEADER: headers[SIGNATURE_HEADER],
    }
    assert headers[SIGNATURE_HEADER].startswith("sha256=")
