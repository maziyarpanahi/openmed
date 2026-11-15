"""Tests for HMAC request signing and local replay protection."""

from __future__ import annotations

import json

import httpx
import pytest

from openmed.service.signing import (
    ATTEMPT_HEADER,
    DELIVERY_ID_HEADER,
    EVENT_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    WEBHOOK_SIGNATURE_HEADER,
    InvalidSignatureError,
    NonceCache,
    NonceCacheFullError,
    ReplayDetectedError,
    StaleTimestampError,
    canonical_signing_string,
    new_delivery_id,
    normalize_delivery_id,
    parse_webhook_headers,
    parse_webhook_signature,
    sign_request,
    sign_webhook_request,
    verify_request_signature,
    webhook_signing_string,
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


def test_explicit_empty_nonce_is_rejected() -> None:
    with pytest.raises(ValueError):
        sign_request(
            "POST",
            "/jobs",
            secret="shared-secret",
            timestamp=1_800_000_000,
            nonce="",
        )


def test_nonce_cache_rejects_non_finite_window() -> None:
    with pytest.raises(ValueError):
        NonceCache(window_seconds=float("nan"))


# --- Version 2 webhook-delivery contract (#3573) ---------------------------

V2_BODY = b'{"event":"job.done","job_id":"synthetic-job","status":"done"}'
V2_FIELDS = {
    "key_id": "webhooks-2026-11",
    "event": "job.done",
    "delivery_id": "dlv_00000000000000000000000000000001",
    "attempt": 2,
}


def test_v2_signing_string_wraps_the_unchanged_v1_canonical_string() -> None:
    v1 = canonical_signing_string(
        "post",
        "/callbacks/openmed?source=jobs",
        1_800_000_000,
        "synthetic-attempt-nonce",
        V2_BODY,
    )
    v2 = webhook_signing_string(
        "post",
        "/callbacks/openmed?source=jobs",
        1_800_000_000,
        "synthetic-attempt-nonce",
        V2_BODY,
        **V2_FIELDS,
    )

    assert v2 == (
        b"OPENMED-WEBHOOK-HMAC-SHA256-V2\n"
        b"key-id:webhooks-2026-11\n"
        b"event:job.done\n"
        b"delivery-id:dlv_00000000000000000000000000000001\n"
        b"attempt:2\n" + v1
    )
    assert b"synthetic-job" not in v2


def test_v2_signature_matches_the_published_test_vector() -> None:
    headers = sign_webhook_request(
        "POST",
        "/callbacks/openmed?source=jobs",
        V2_BODY,
        secret="synthetic-webhook-secret",
        timestamp=1_800_000_000,
        nonce="synthetic-attempt-nonce",
        **V2_FIELDS,
    )

    assert headers == {
        EVENT_HEADER: "job.done",
        DELIVERY_ID_HEADER: "dlv_00000000000000000000000000000001",
        ATTEMPT_HEADER: "2",
        TIMESTAMP_HEADER: "1800000000",
        NONCE_HEADER: "synthetic-attempt-nonce",
        WEBHOOK_SIGNATURE_HEADER: (
            "v2,kid=webhooks-2026-11,"
            "sha256=bf5c4494b17722c287c69ecf8b1c8ffa5425067eeafbadc80ffc57d08bc529a3"
        ),
    }


def test_v2_legacy_header_is_the_v1_signature_with_the_same_key() -> None:
    headers = sign_webhook_request(
        "POST",
        "/callbacks",
        V2_BODY,
        secret="synthetic-webhook-secret",
        timestamp=1_800_000_000,
        nonce="synthetic-attempt-nonce",
        include_legacy_signature=True,
        **V2_FIELDS,
    )

    assert verify_request_signature(
        "POST",
        "/callbacks",
        V2_BODY,
        headers,
        secret="synthetic-webhook-secret",
        nonce_cache=NonceCache(),
        now=1_800_000_000,
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "sha256=" + "a" * 64,
        "v2,kid=k1,sha256=" + "A" * 64,
        "v2,kid=k1,sha256=" + "a" * 63,
        "v2,kid=-bad,sha256=" + "a" * 64,
        "v2;kid=k1;sha256=" + "a" * 64,
        "v2,sha256=" + "a" * 64 + ",kid=k1",
        "v0,kid=k1,sha256=" + "a" * 64,
    ],
)
def test_malformed_v2_signature_values_are_rejected(value: str) -> None:
    with pytest.raises(InvalidSignatureError):
        parse_webhook_signature(value)


def test_parse_webhook_signature_reports_unsupported_versions() -> None:
    parsed = parse_webhook_signature("v3,kid=k1,sha256=" + "a" * 64)

    assert parsed.version == "v3"
    assert parsed.key_id == "k1"
    assert "a" * 64 not in repr(parsed)


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempt": 0},
        {"attempt": 1001},
        {"attempt": True},
        {"delivery_id": "short"},
        {"delivery_id": "dlv bad"},
        {"event": ""},
        {"event": "job done"},
        {"key_id": "bad/key"},
    ],
)
def test_v2_contract_fields_are_validated(overrides: dict[str, object]) -> None:
    fields = {**V2_FIELDS, **overrides}
    with pytest.raises((TypeError, ValueError)):
        webhook_signing_string(
            "POST",
            "/callbacks",
            1_800_000_000,
            "nonce",
            b"",
            **fields,  # type: ignore[arg-type]
        )


def test_parse_webhook_headers_requires_every_header_exactly_once() -> None:
    headers = sign_webhook_request(
        "POST",
        "/callbacks",
        V2_BODY,
        secret="synthetic-webhook-secret",
        timestamp=1_800_000_000,
        nonce="synthetic-attempt-nonce",
        **V2_FIELDS,
    )
    parsed = parse_webhook_headers({name.lower(): v for name, v in headers.items()})
    assert parsed.delivery_id == V2_FIELDS["delivery_id"]
    assert parsed.attempt == 2

    for missing in headers:
        partial = {name: value for name, value in headers.items() if name != missing}
        with pytest.raises(InvalidSignatureError):
            parse_webhook_headers(partial)
    with pytest.raises(InvalidSignatureError):
        parse_webhook_headers({**headers, ATTEMPT_HEADER: "02"})
    with pytest.raises(InvalidSignatureError):
        parse_webhook_headers(
            {**headers, DELIVERY_ID_HEADER.lower(): "dlv_duplicate_value_0001"}
        )


def test_new_delivery_ids_are_opaque_and_unique() -> None:
    first, second = new_delivery_id(), new_delivery_id()

    assert first != second
    assert normalize_delivery_id(first) == first
    assert first.startswith("dlv_") and len(first) == 36
