"""Receiver-side replay, idempotency, and rotation tests for webhooks (#3573)."""

from __future__ import annotations

import base64
import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest

from openmed.service.signing import (
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    WEBHOOK_SIGNATURE_HEADER,
    sign_request,
    sign_webhook_request,
)
from openmed.service.signing_keys import (
    KEY_STATE_ACTIVE,
    KEY_STATE_RETIRED,
    KEY_STATE_RETIRING,
    SigningKey,
    SigningKeyError,
    SigningKeyring,
    SigningKeyringFile,
)
from openmed.service.webhook_replay import (
    FileReplayLedger,
    WebhookOutcome,
    WebhookReceiver,
    WebhookVerification,
)
from openmed.service.webhooks import deliver_webhook

T0 = 1_800_000_000
PATH = "/callbacks/openmed?source=jobs"
URL = f"https://zqreceiversentinel.example.test{PATH}"
LEDGER_KEY = b"L" * 32
SECRET_OLD = b"zq-receiver-old-secret-0123456789abcdef"
SECRET_NEW = b"zq-receiver-new-secret-0123456789abcdef"
PAYLOAD_SENTINEL = "ZQPAYLOADSENTINEL Synthetic Patient 555-0100"
BODY = json.dumps(
    {"event": "job.done", "job_id": "synthetic", "note": PAYLOAD_SENTINEL},
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
DELIVERY = "dlv_synthetic_delivery_0001"
SENTINELS = (
    SECRET_OLD,
    SECRET_NEW,
    LEDGER_KEY,
    PAYLOAD_SENTINEL.encode(),
    DELIVERY.encode(),
    b"zqreceiversentinel",
)


def _assert_value_free(blob: Any, *, allow_delivery_id: bool = False) -> None:
    text = blob if isinstance(blob, (bytes, str)) else json.dumps(blob)
    data = text.encode("utf-8") if isinstance(text, str) else text
    for sentinel in SENTINELS:
        if allow_delivery_id and sentinel == DELIVERY.encode():
            continue
        assert sentinel not in data
        assert base64.b64encode(sentinel) not in data


def _keyring(**overrides: Any) -> SigningKeyring:
    keys = overrides.pop(
        "keys",
        (
            SigningKey(
                key_id="k-new",
                state=KEY_STATE_ACTIVE,
                secret=SECRET_NEW,
                not_before=T0 - 10,
            ),
            SigningKey(
                key_id="k-old",
                state=KEY_STATE_RETIRING,
                secret=SECRET_OLD,
                verify_until=T0 + 100,
            ),
            SigningKey(key_id="k-gone", state=KEY_STATE_RETIRED),
        ),
    )
    return SigningKeyring(generation=2, keys=keys, rotated_at=T0 - 10, **overrides)


def _ledger(tmp_path: Path, **overrides: Any) -> FileReplayLedger:
    options: dict[str, Any] = {"integrity_key": LEDGER_KEY, "clock": lambda: T0}
    options.update(overrides)
    return FileReplayLedger(tmp_path / "ledger.json", **options)


def _receiver(tmp_path: Path, **overrides: Any) -> WebhookReceiver:
    keyring = overrides.pop("keyring", _keyring())
    ledger = overrides.pop("ledger", None) or _ledger(tmp_path)
    return WebhookReceiver(
        keyring=keyring, ledger=ledger, clock=lambda: T0, **overrides
    )


def _signed(
    *,
    attempt: int = 1,
    delivery_id: str = DELIVERY,
    key_id: str = "k-new",
    secret: bytes = SECRET_NEW,
    timestamp: int = T0,
    nonce: Optional[str] = None,
    body: bytes = BODY,
) -> dict[str, str]:
    return sign_webhook_request(
        "POST",
        PATH,
        body,
        secret=secret,
        key_id=key_id,
        event="job.done",
        delivery_id=delivery_id,
        attempt=attempt,
        timestamp=timestamp,
        nonce=nonce or f"nonce-{delivery_id}-{attempt}-{timestamp}",
    )


def _verify(
    receiver: WebhookReceiver,
    headers: dict[str, str],
    *,
    now: float = T0,
    body: bytes = BODY,
) -> WebhookVerification:
    return receiver.verify("POST", PATH, body, headers, now=now)


def test_a_multi_attempt_delivery_applies_its_effect_exactly_once(
    tmp_path: Path,
) -> None:
    receiver = _receiver(tmp_path)
    keyring_resolver = SigningKeyring.single(SECRET_NEW, key_id="k-new")
    effects: list[str] = []
    outcomes: list[WebhookVerification] = []

    def handler(request: httpx.Request) -> httpx.Response:
        verification = receiver.verify(
            request.method,
            request.url.raw_path.decode("ascii"),
            request.content,
            request.headers,
            now=T0,
        )
        outcomes.append(verification)
        if verification.accepted:
            effects.append(verification.delivery_id or "")
            assert receiver.mark_applied(verification) is True
            # The receiver applied the effect but the response is lost, so
            # the sender retries the same delivery.
            return httpx.Response(500)
        return httpx.Response(verification.http_status)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = deliver_webhook(
        URL,
        json.loads(BODY),
        keyring=keyring_resolver,
        max_attempts=3,
        backoff_seconds=0,
        client=client,
        clock=lambda: T0,
    )

    assert result.success is True
    assert result.attempts == 2
    assert effects == [result.delivery_id]
    assert [item.outcome for item in outcomes] == [
        WebhookOutcome.ACCEPTED,
        WebhookOutcome.DUPLICATE,
    ]
    assert outcomes[1].reason == "delivery_applied"
    assert {item.delivery_id for item in outcomes} == {result.delivery_id}
    assert [item.attempt for item in outcomes] == [1, 2]
    stats = receiver.ledger.stats()
    assert stats["deliveries_applied"] == 1
    assert stats["attempts"] == 2


def test_retransmitting_any_seen_attempt_is_a_replay(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path)
    first, second = _signed(attempt=1), _signed(attempt=2)

    assert _verify(receiver, first).outcome is WebhookOutcome.ACCEPTED
    assert _verify(receiver, second).outcome is WebhookOutcome.DUPLICATE
    replay_one = _verify(receiver, first, now=T0 + 5)
    replay_two = _verify(receiver, second, now=T0 + 5)

    assert replay_one.outcome is WebhookOutcome.REPLAYED
    assert replay_one.reason == "attempt_replayed"
    assert replay_two.outcome is WebhookOutcome.REPLAYED
    assert replay_one.http_status == 409
    # A different delivery that reuses an accepted nonce is also a replay.
    reused_nonce = _signed(
        delivery_id="dlv_synthetic_delivery_0002", nonce=first[NONCE_HEADER]
    )
    reused = _verify(receiver, reused_nonce)
    assert reused.outcome is WebhookOutcome.REPLAYED
    assert reused.reason == "nonce_replayed"


def test_replay_stays_rejected_after_a_receiver_restart(tmp_path: Path) -> None:
    first = _receiver(tmp_path)
    attempt_one = _signed(attempt=1)
    accepted = _verify(first, attempt_one)
    assert accepted.accepted
    assert first.mark_applied(accepted)

    # Restart: fresh objects over the same ledger file.
    restarted = _receiver(tmp_path)

    assert _verify(restarted, attempt_one, now=T0 + 1).outcome is (
        WebhookOutcome.REPLAYED
    )
    retry = _verify(restarted, _signed(attempt=2, timestamp=T0 + 30), now=T0 + 30)
    assert retry.outcome is WebhookOutcome.DUPLICATE
    assert retry.reason == "delivery_applied"


def test_concurrent_receivers_cannot_both_claim_one_delivery(tmp_path: Path) -> None:
    # Separate receiver and ledger objects share one file, as separate worker
    # processes would; the file lock serializes their claims.
    receivers = [_receiver(tmp_path) for _ in range(4)]
    attempts = [_signed(attempt=index + 1) for index in range(8)]
    barrier = threading.Barrier(len(attempts))
    results: list[WebhookVerification] = []
    lock = threading.Lock()

    def claim(index: int) -> None:
        barrier.wait()
        verification = _verify(receivers[index % len(receivers)], attempts[index])
        with lock:
            results.append(verification)

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    outcomes = [item.outcome for item in results]
    assert outcomes.count(WebhookOutcome.ACCEPTED) == 1
    assert outcomes.count(WebhookOutcome.DUPLICATE) == 7

    # The same attempt delivered to two receivers at once: one claim wins.
    same = _signed(delivery_id="dlv_synthetic_delivery_0003")
    barrier_two = threading.Barrier(2)
    racing: list[WebhookVerification] = []

    def race(receiver: WebhookReceiver) -> None:
        barrier_two.wait()
        verification = _verify(receiver, same)
        with lock:
            racing.append(verification)

    pair = [threading.Thread(target=race, args=(r,)) for r in receivers[:2]]
    for thread in pair:
        thread.start()
    for thread in pair:
        thread.join()
    assert sorted(item.outcome.value for item in racing) == ["accepted", "replayed"]


def test_a_released_claim_can_be_retried_but_not_replayed(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path)
    first = _signed(attempt=1)
    accepted = _verify(receiver, first)

    assert receiver.release(accepted) is True
    assert _verify(receiver, first).outcome is WebhookOutcome.REPLAYED
    retry = _verify(receiver, _signed(attempt=2))
    assert retry.outcome is WebhookOutcome.ACCEPTED
    assert receiver.mark_applied(retry) is True
    assert receiver.mark_applied(retry) is False
    assert receiver.release(retry) is False


def test_in_progress_claims_are_reported_as_duplicates(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path)
    assert _verify(receiver, _signed(attempt=1)).accepted

    duplicate = _verify(receiver, _signed(attempt=2))

    assert duplicate.outcome is WebhookOutcome.DUPLICATE
    assert duplicate.reason == "delivery_in_progress"
    assert duplicate.http_status == 200


def test_clock_edges_for_freshness_and_replay_retention(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path)
    headers = _signed(timestamp=T0)

    early = _verify(receiver, headers, now=T0 - 301)
    assert early.outcome is WebhookOutcome.STALE
    assert early.reason == "timestamp_outside_window"
    assert _verify(receiver, headers, now=T0 + 300).accepted
    # At the very edge of the window the attempt record is still held.
    assert _verify(receiver, headers, now=T0 + 300).outcome is (WebhookOutcome.REPLAYED)
    # Past the window the record may expire, but the request is stale anyway.
    late = _verify(receiver, headers, now=T0 + 301)
    assert late.outcome is WebhookOutcome.STALE
    assert receiver.ledger.purge_expired(now=T0 + 301) == 2
    future = _signed(attempt=2, timestamp=T0 + 300)
    assert _verify(receiver, future, now=T0).outcome is WebhookOutcome.DUPLICATE
    too_far = _signed(attempt=3, timestamp=T0 + 301)
    assert _verify(receiver, too_far, now=T0).outcome is WebhookOutcome.STALE


def test_failed_verification_never_consumes_a_delivery_identity(
    tmp_path: Path,
) -> None:
    receiver = _receiver(tmp_path)
    genuine = _signed()

    tampered = _verify(receiver, genuine, body=BODY + b" ")
    wrong_key = _verify(receiver, _signed(secret=SECRET_OLD))
    garbage = _verify(receiver, {**genuine, WEBHOOK_SIGNATURE_HEADER: "v2,kid=k-new,"})
    stale = _verify(receiver, genuine, now=T0 + 10_000)

    assert tampered.outcome is WebhookOutcome.BAD_SIGNATURE
    assert tampered.reason == "signature_mismatch"
    # A wrong secret under a valid key id looks exactly like any other mismatch.
    assert wrong_key.to_dict() == tampered.to_dict()
    assert garbage.outcome is WebhookOutcome.BAD_SIGNATURE
    assert garbage.reason == "headers_invalid"
    assert stale.outcome is WebhookOutcome.STALE
    for rejected in (tampered, wrong_key, garbage, stale):
        assert rejected.delivery_id is None
        assert rejected.key_id is None
        assert rejected.http_status == 401
    assert not (tmp_path / "ledger.json").exists()
    assert _verify(receiver, genuine).accepted


def test_rotation_overlap_accepts_old_and_new_keys_only_inside_the_window(
    tmp_path: Path,
) -> None:
    receiver = _receiver(tmp_path)

    new_key = _verify(receiver, _signed(delivery_id="dlv_rotation_new_key_01"))
    old_key = _verify(
        receiver,
        _signed(
            delivery_id="dlv_rotation_old_key_01",
            key_id="k-old",
            secret=SECRET_OLD,
            timestamp=T0 + 100,
        ),
        now=T0 + 100,
    )
    expired = _verify(
        receiver,
        _signed(
            delivery_id="dlv_rotation_old_key_02",
            key_id="k-old",
            secret=SECRET_OLD,
            timestamp=T0 + 101,
        ),
        now=T0 + 101,
    )
    retired = _verify(
        receiver,
        _signed(delivery_id="dlv_rotation_gone_key_1", key_id="k-gone"),
    )
    unknown = _verify(
        receiver, _signed(delivery_id="dlv_rotation_unknown_1", key_id="k-other")
    )
    premature = _verify(
        receiver,
        _signed(delivery_id="dlv_rotation_premature", timestamp=T0 - 20),
        now=T0 - 20,
    )

    assert new_key.accepted and new_key.key_id == "k-new"
    assert old_key.accepted and old_key.key_id == "k-old"
    assert (expired.outcome, expired.reason) == (
        WebhookOutcome.RETIRED_KEY,
        "key_overlap_expired",
    )
    assert (retired.outcome, retired.reason) == (
        WebhookOutcome.RETIRED_KEY,
        "key_retired",
    )
    assert (unknown.outcome, unknown.reason) == (
        WebhookOutcome.UNKNOWN_KEY,
        "key_unknown",
    )
    assert (premature.outcome, premature.reason) == (
        WebhookOutcome.UNKNOWN_KEY,
        "key_not_yet_active",
    )
    # None of the rejected deliveries were recorded.
    assert receiver.ledger.stats()["deliveries_claimed"] == 2


def test_file_rotation_keeps_in_flight_deliveries_verifiable(tmp_path: Path) -> None:
    keyring_file = SigningKeyringFile(tmp_path / "keys" / "keyring.json")
    keyring_file.rotate(new_secret=SECRET_OLD, new_key_id="k-old", now=T0 - 50)
    receiver = _receiver(tmp_path, keyring=keyring_file)
    in_flight = _signed(key_id="k-old", secret=SECRET_OLD)

    receipt = keyring_file.rotate(
        new_secret=SECRET_NEW, new_key_id="k-new", overlap_seconds=60, now=T0
    )

    assert receipt.retiring_key_ids == ("k-old",)
    assert _verify(receiver, in_flight, now=T0 + 60).accepted
    retry = _signed(attempt=2, timestamp=T0 + 61)
    assert _verify(receiver, retry, now=T0 + 61).outcome is WebhookOutcome.DUPLICATE
    late_old = _signed(
        delivery_id="dlv_after_overlap_0001",
        key_id="k-old",
        secret=SECRET_OLD,
        timestamp=T0 + 61,
    )
    assert _verify(receiver, late_old, now=T0 + 61).outcome is (
        WebhookOutcome.RETIRED_KEY
    )
    _assert_value_free(receipt.to_dict())


def test_a_tampered_ledger_fails_closed(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path)
    assert _verify(receiver, _signed(attempt=1)).accepted
    path = tmp_path / "ledger.json"
    document = json.loads(path.read_text())
    digest, entry = next(iter(document["deliveries"].items()))
    entry["state"] = "applied"
    path.write_text(json.dumps(document))

    tampered = _verify(receiver, _signed(attempt=2))
    assert tampered.outcome is WebhookOutcome.UNAVAILABLE
    assert tampered.reason == "ledger_integrity_failed"
    assert tampered.http_status == 503

    # Dropping an entry (to make a replay look new) is detected the same way.
    del document["deliveries"][digest]
    path.write_text(json.dumps(document))
    assert _verify(_receiver(tmp_path), _signed(attempt=1)).reason == (
        "ledger_integrity_failed"
    )

    # A truncated file is rejected as well.
    path.write_text('{"format":')
    assert _verify(_receiver(tmp_path), _signed(attempt=3)).outcome is (
        WebhookOutcome.UNAVAILABLE
    )


def test_wrong_integrity_key_and_rollback_fail_closed(tmp_path: Path) -> None:
    receiver = _receiver(tmp_path)
    accepted = _verify(receiver, _signed(attempt=1))
    path = tmp_path / "ledger.json"
    snapshot = path.read_bytes()
    assert receiver.mark_applied(accepted)

    other_key = _receiver(tmp_path, ledger=_ledger(tmp_path, integrity_key=b"M" * 32))
    assert _verify(other_key, _signed(attempt=2)).reason == "ledger_integrity_failed"

    # Restoring an older, validly signed copy while running is a rollback.
    path.write_bytes(snapshot)
    rolled_back = _verify(receiver, _signed(attempt=1))
    assert rolled_back.outcome is WebhookOutcome.UNAVAILABLE
    assert rolled_back.reason == "ledger_rollback"

    # Deleting the ledger after use never silently resets replay protection.
    path.unlink()
    assert _verify(receiver, _signed(attempt=1)).reason == "ledger_rollback"


def test_a_full_ledger_fails_closed_until_entries_expire(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, max_entries=3, delivery_ttl_seconds=400)
    receiver = _receiver(tmp_path, ledger=ledger)
    assert _verify(receiver, _signed()).accepted

    full = _verify(receiver, _signed(delivery_id="dlv_synthetic_delivery_0009"))
    assert full.outcome is WebhookOutcome.UNAVAILABLE
    assert full.reason == "ledger_full"
    assert ledger.stats()["entries"] == 3

    later = T0 + 401
    retry = _signed(delivery_id="dlv_synthetic_delivery_0009", timestamp=later)
    assert _verify(receiver, retry, now=later).accepted


def test_legacy_v1_requests_need_the_explicit_compatibility_flag(
    tmp_path: Path,
) -> None:
    legacy = sign_request(
        "POST", PATH, BODY, secret=SECRET_OLD, timestamp=T0, nonce="legacy-nonce-1"
    )
    strict = _receiver(tmp_path)
    rejected = _verify(strict, legacy)
    assert rejected.outcome is WebhookOutcome.BAD_SIGNATURE
    assert rejected.reason == "signature_version_unsupported"

    compatible = _receiver(tmp_path, accept_legacy_v1=True)
    accepted = _verify(compatible, legacy)
    assert accepted.accepted
    assert accepted.signature_version == "v1"
    assert accepted.key_id == "k-old"
    assert accepted.delivery_id is None
    assert compatible.mark_applied(accepted) is False
    assert _verify(compatible, legacy).outcome is WebhookOutcome.REPLAYED
    forged = {**legacy, NONCE_HEADER: "legacy-nonce-2"}
    assert _verify(compatible, forged).reason == "signature_mismatch"
    assert _verify(compatible, {TIMESTAMP_HEADER: str(T0)}).reason == (
        "headers_invalid"
    )
    bad_digest = {**legacy, SIGNATURE_HEADER: "sha256=" + "é" * 64}
    assert _verify(compatible, bad_digest).reason == "headers_invalid"


def test_unsupported_signature_versions_are_rejected(tmp_path: Path) -> None:
    headers = _signed()
    digest = headers[WEBHOOK_SIGNATURE_HEADER].rsplit("=", 1)[1]
    headers[WEBHOOK_SIGNATURE_HEADER] = f"v3,kid=k-new,sha256={digest}"

    rejected = _verify(_receiver(tmp_path), headers)

    assert rejected.outcome is WebhookOutcome.BAD_SIGNATURE
    assert rejected.reason == "signature_version_unsupported"


def test_an_unavailable_keyring_fails_closed(tmp_path: Path) -> None:
    class Broken:
        def current(self) -> SigningKeyring:
            raise SigningKeyError("keyring_unreadable")

    rejected = _verify(_receiver(tmp_path, keyring=Broken()), _signed())

    assert rejected.outcome is WebhookOutcome.UNAVAILABLE
    assert rejected.reason == "keyring_unavailable"


def test_ledger_and_receipts_hold_no_payload_keys_or_raw_identifiers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    receiver = _receiver(tmp_path)
    receipts: list[dict[str, Any]] = []
    with caplog.at_level(logging.DEBUG):
        first = _verify(receiver, _signed(attempt=1))
        receiver.mark_applied(first)
        for headers in (_signed(attempt=1), _signed(attempt=2)):
            receipts.append(_verify(receiver, headers).to_dict())
        receipts.append(_verify(receiver, _signed(), body=b"tampered").to_dict())
        receipts.append(receiver.ledger.stats())

    stored = b"".join(path.read_bytes() for path in tmp_path.iterdir())
    _assert_value_free(stored)
    _assert_value_free(caplog.text)
    _assert_value_free(receipts, allow_delivery_id=True)
    _assert_value_free(repr(receiver.keyring))
    _assert_value_free(repr(first), allow_delivery_id=True)


def test_receiver_and_ledger_validate_their_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FileReplayLedger(tmp_path / "l.json", integrity_key=b"short")
    with pytest.raises(ValueError):
        FileReplayLedger(tmp_path / "l.json", integrity_key=LEDGER_KEY, max_entries=2)
    with pytest.raises(ValueError):
        FileReplayLedger(
            tmp_path / "l.json", integrity_key=LEDGER_KEY, delivery_ttl_seconds=0
        )
    short_ttl = FileReplayLedger(
        tmp_path / "l.json", integrity_key=LEDGER_KEY, delivery_ttl_seconds=60
    )
    with pytest.raises(ValueError):
        WebhookReceiver(keyring=_keyring(), ledger=short_ttl, max_skew_seconds=300)
