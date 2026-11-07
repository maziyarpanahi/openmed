"""Tests for the protected async job payload lifecycle stores."""

from __future__ import annotations

import base64
import errno
import json
import os
import random
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core.aead_envelope import EnvelopeAuthenticationError
from openmed.core.key_provider import (
    DATA_KEY_BYTES,
    EncryptionError,
    InMemoryKeyProvider,
    KeyRevokedError,
    KeyState,
    KeyUnavailableError,
)
from openmed.service import job_payloads
from openmed.service.job_payloads import (
    JOB_PAYLOAD_KEY_PURPOSE,
    PAYLOAD_KIND_INPUT,
    PAYLOAD_KIND_TERMINAL_EVENT,
    EncryptedJobPayloadStore,
    JobPayloadError,
    JobPayloadStore,
    MemoryJobPayloadStore,
)

KEY_ONE = bytes(range(DATA_KEY_BYTES))
KEY_TWO = bytes(reversed(range(DATA_KEY_BYTES)))
TEXT_SENTINEL = "ZQJOBPAYLOADSENTINEL41 Maria Synthetic"
SECRET_SENTINEL = "ZQWEBHOOKSECRETSENTINEL77"
URL_SENTINEL = "https://zqcallbacksentinel.example.test/hook"
SENTINELS = (TEXT_SENTINEL, SECRET_SENTINEL, URL_SENTINEL)
JOB_ID = "0123456789abcdef0123456789abcdef"
OTHER_JOB_ID = "fedcba9876543210fedcba9876543210"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


class Clock:
    """Mutable deterministic clock."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def _provider(material: bytes = KEY_ONE, key_id: str = "jobs-key-1") -> Any:
    provider = InMemoryKeyProvider()
    provider.add_key(
        key_id, material, purpose=JOB_PAYLOAD_KEY_PURPOSE, state=KeyState.ACTIVE
    )
    return provider


def _encrypted(
    root: Path,
    clock: Clock,
    *,
    provider: Any = None,
    max_retention_seconds: int = 3600,
) -> EncryptedJobPayloadStore:
    return EncryptedJobPayloadStore(
        root,
        key_provider=_provider() if provider is None else provider,
        max_retention_seconds=max_retention_seconds,
        clock=clock,
    )


def _body() -> dict[str, Any]:
    return {
        "documents": [{"id": "synthetic-0", "text": TEXT_SENTINEL}],
        "webhook": {"url": URL_SENTINEL, "secret": SECRET_SENTINEL},
        "method": "mask",
    }


def _all_bytes(root: Path) -> bytes:
    return b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())


def _assert_no_sentinels(blob: bytes | str) -> None:
    data = blob.encode("utf-8") if isinstance(blob, str) else blob
    for sentinel in SENTINELS:
        encoded = sentinel.encode("utf-8")
        assert encoded not in data
        assert base64.b64encode(encoded) not in data


def _payload_files(root: Path) -> list[Path]:
    return sorted(root.glob("*.payload.json")) if root.exists() else []


@pytest.fixture(params=["memory", "encrypted"])
def any_store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    clock = Clock()
    if request.param == "memory":
        store: JobPayloadStore = MemoryJobPayloadStore(
            max_retention_seconds=3600, clock=clock
        )
    else:
        store = _encrypted(tmp_path / "payloads", clock)
    return store, clock


def test_round_trip_returns_fresh_copies_of_an_immutable_snapshot(any_store) -> None:
    store, _clock = any_store
    body = _body()

    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, body, retention_seconds=60)
    body["documents"][0]["text"] = "changed after create"
    first = store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    first["documents"].append({"text": "appended by reader"})
    second = store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)

    assert second == _body()
    assert ref.ref.startswith("jp_") and JOB_ID not in ref.ref
    assert ref.expires_at == "2026-01-01T00:01:00Z"
    assert ref.to_record()["state"] == "retained"


def test_binding_to_job_and_kind_is_enforced(any_store) -> None:
    store, _clock = any_store
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)

    for job_id, kind in [
        (OTHER_JOB_ID, PAYLOAD_KIND_INPUT),
        (JOB_ID, PAYLOAD_KIND_TERMINAL_EVENT),
    ]:
        with pytest.raises(JobPayloadError) as caught:
            store.read(ref.ref, job_id=job_id, kind=kind)
        assert caught.value.code == "payload_missing"
    assert store.contains(ref.ref)


@pytest.mark.parametrize("retention", [0, -1, 3601, True, 1.5, "60"])
def test_retention_must_stay_within_the_configured_ceiling(
    any_store, retention: Any
) -> None:
    store, _clock = any_store
    with pytest.raises(JobPayloadError) as caught:
        store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=retention)
    assert caught.value.code == "payload_retention_invalid"
    assert store.retained_count() == 0


@pytest.mark.parametrize(
    ("kind", "job_id", "body"),
    [
        ("other", JOB_ID, {}),
        (PAYLOAD_KIND_INPUT, "../escape", {}),
        (PAYLOAD_KIND_INPUT, JOB_ID, ["not", "a", "mapping"]),
        (PAYLOAD_KIND_INPUT, JOB_ID, {"value": float("nan")}),
        (PAYLOAD_KIND_INPUT, JOB_ID, {"value": object()}),
    ],
)
def test_invalid_create_arguments_fail_without_retaining(
    any_store, kind: str, job_id: str, body: Any
) -> None:
    store, _clock = any_store
    with pytest.raises(JobPayloadError):
        store.create(kind, job_id, body, retention_seconds=60)
    assert store.retained_count() == 0


@pytest.mark.parametrize("ref", ["", "jp_short", "../jp_" + "0" * 32, 7, None])
def test_invalid_references_are_refused(any_store, ref: Any) -> None:
    store, _clock = any_store
    with pytest.raises(JobPayloadError) as caught:
        store.read(ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    assert caught.value.code == "payload_ref_invalid"


def test_expired_payload_is_refused_removed_and_reported(any_store) -> None:
    store, clock = any_store
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)

    clock.advance(59)
    assert store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    clock.advance(1)
    with pytest.raises(JobPayloadError) as caught:
        store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)

    assert caught.value.code == "payload_expired"
    assert not store.contains(ref.ref)
    assert [entry.ref for entry in store.sweep_expired()] == [ref.ref]
    assert store.sweep_expired() == ()
    with pytest.raises(JobPayloadError) as again:
        store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    assert again.value.code == "payload_expired"
    events = [receipt.event for receipt in store.receipts()]
    assert events == ["created", "read", "expired", "read_refused", "read_refused"]


def test_sweep_expires_only_payloads_past_their_deadline(any_store) -> None:
    store, clock = any_store
    short = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=10)
    long = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=100)

    clock.advance(10)
    expired = store.sweep_expired()

    assert [entry.ref for entry in expired] == [short.ref]
    assert store.contains(long.ref)
    assert store.retained_count() == 1


def test_deleted_payload_cannot_be_reopened(any_store) -> None:
    store, _clock = any_store
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)

    store.delete(ref.ref, reason="consumed")
    store.delete(ref.ref, reason="consumed")

    with pytest.raises(JobPayloadError) as caught:
        store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    assert caught.value.code == "payload_deleted"
    deleted = [r for r in store.receipts() if r.event == "deleted"]
    assert len(deleted) == 1 and deleted[0].reason == "consumed"


def test_pinned_payload_survives_cleanup_until_released(any_store) -> None:
    store, clock = any_store
    ref = store.create(
        PAYLOAD_KIND_TERMINAL_EVENT, JOB_ID, _body(), retention_seconds=60
    )

    with store.pinned(ref.ref):
        assert store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_TERMINAL_EVENT)
        clock.advance(61)
        assert store.sweep_expired() == ()
        assert store.retained_count() == 1
        with pytest.raises(JobPayloadError) as caught:
            store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_TERMINAL_EVENT)
        assert caught.value.code == "payload_expired"
        assert store.retained_count() == 1

    assert store.retained_count() == 0
    assert [entry.ref for entry in store.sweep_expired()] == [ref.ref]
    with pytest.raises(JobPayloadError):
        with store.pinned(ref.ref):
            pass


def test_delete_during_pin_is_deferred_and_blocks_new_reads(any_store) -> None:
    store, _clock = any_store
    ref = store.create(
        PAYLOAD_KIND_TERMINAL_EVENT, JOB_ID, _body(), retention_seconds=60
    )

    with store.pinned(ref.ref):
        with store.pinned(ref.ref):
            store.delete(ref.ref, reason="delivered")
            assert store.retained_count() == 1
            with pytest.raises(JobPayloadError) as caught:
                store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_TERMINAL_EVENT)
            assert caught.value.code == "payload_deleted"
        assert store.retained_count() == 1

    assert store.retained_count() == 0
    assert store.receipts()[-1].event == "deleted"
    assert store.receipts()[-1].reason == "delivered"


def test_concurrent_cleanup_and_delivery_never_lose_or_resurrect(any_store) -> None:
    store, clock = any_store
    refs = [
        store.create(PAYLOAD_KIND_TERMINAL_EVENT, JOB_ID, _body(), retention_seconds=30)
        for _ in range(12)
    ]
    start = threading.Barrier(3)
    errors: list[BaseException] = []
    delivered: list[str] = []

    def deliver() -> None:
        start.wait()
        for ref in refs:
            try:
                with store.pinned(ref.ref):
                    store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_TERMINAL_EVENT)
                    delivered.append(ref.ref)
            except JobPayloadError as exc:
                if exc.code not in {"payload_expired", "payload_deleted"}:
                    errors.append(exc)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

    def cleanup() -> None:
        start.wait()
        for index, ref in enumerate(refs):
            if index % 3 == 0:
                store.delete(ref.ref, reason="delivered")
            store.sweep_expired()

    threads = [threading.Thread(target=deliver), threading.Thread(target=cleanup)]
    for thread in threads:
        thread.start()
    start.wait()
    clock.advance(31)
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    store.sweep_expired()
    assert store.retained_count() == 0
    for ref in refs:
        assert not store.contains(ref.ref)
        with pytest.raises(JobPayloadError):
            store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_TERMINAL_EVENT)
    if isinstance(store, EncryptedJobPayloadStore):
        assert _payload_files(store.directory) == []


def test_receipts_and_evidence_are_value_free(any_store, tmp_path: Path) -> None:
    store, clock = any_store
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    with pytest.raises(JobPayloadError):
        store.read(ref.ref, job_id=OTHER_JOB_ID, kind=PAYLOAD_KIND_INPUT)
    other = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=5)
    store.delete(ref.ref, reason="consumed")
    clock.advance(5)
    store.sweep_expired()

    evidence = store.lifecycle_evidence()
    rendered = json.dumps(evidence, sort_keys=True)

    _assert_no_sentinels(rendered)
    assert JOB_ID not in rendered and str(tmp_path) not in rendered
    assert evidence["schema_version"] == 1
    assert evidence["mode"] == store.mode
    assert evidence["retained_count"] == 0
    assert evidence["event_counts"] == {
        "created": 2,
        "deleted": 1,
        "expired": 1,
        "read": 1,
        "read_refused": 1,
    }
    assert {receipt["ref"] for receipt in evidence["receipts"]} == {
        ref.ref,
        other.ref,
    }
    for receipt in evidence["receipts"]:
        assert set(receipt) == {
            "event",
            "ref",
            "kind",
            "at",
            "reason",
            "digest",
            "schema_version",
        }
    plaintext_digest = job_payloads._sha256_digest(json.dumps(_body()).encode())
    assert plaintext_digest not in rendered


def test_errors_are_value_free(tmp_path: Path) -> None:
    clock = Clock()
    store = _encrypted(tmp_path / "payloads", clock)
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    wrong = _encrypted(tmp_path / "payloads", clock, provider=_provider(KEY_TWO))

    with pytest.raises(EncryptionError) as caught:
        wrong.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)

    message = str(caught.value)
    _assert_no_sentinels(message)
    assert str(tmp_path) not in message
    assert ref.ref not in message and "jobs-key-1" not in message
    assert repr(store).startswith("EncryptedJobPayloadStore(retained=1")
    assert str(tmp_path) not in repr(store)


def test_encrypted_files_and_temp_files_never_hold_plaintext(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "payloads"
    store = _encrypted(root, Clock())
    temp_snapshots: list[bytes] = []
    original_replace = os.replace

    def snapshot_replace(source: Any, destination: Any) -> None:
        temp_snapshots.append(Path(source).read_bytes())
        original_replace(source, destination)

    monkeypatch.setattr(job_payloads.os, "replace", snapshot_replace)
    input_ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    store.create(PAYLOAD_KIND_TERMINAL_EVENT, JOB_ID, _body(), retention_seconds=60)
    monkeypatch.undo()

    assert len(temp_snapshots) == 2
    for blob in [*temp_snapshots, _all_bytes(tmp_path)]:
        _assert_no_sentinels(blob)
    document = json.loads(_payload_files(root)[0].read_text(encoding="ascii"))
    assert set(document) == set(job_payloads._FILE_FIELDS)
    assert document["envelope"].startswith("omae:")
    if os.name == "posix":
        assert root.stat().st_mode & 0o777 == 0o700
        for path in _payload_files(root):
            assert path.stat().st_mode & 0o077 == 0
    assert input_ref.digest is not None and input_ref.digest.startswith("sha256:")


def test_restart_recovers_payloads_until_their_declared_expiry(tmp_path: Path) -> None:
    root = tmp_path / "payloads"
    clock = Clock()
    first = _encrypted(root, clock)
    input_ref = first.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    event_ref = first.create(
        PAYLOAD_KIND_TERMINAL_EVENT,
        JOB_ID,
        {"body": {"status": "done"}},
        retention_seconds=120,
    )
    del first

    restarted = _encrypted(root, clock)
    counts = restarted.recover({input_ref.ref, event_ref.ref})

    assert counts["retained"] == 2
    assert restarted.read(input_ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT) == (
        _body()
    )
    clock.advance(60)
    with pytest.raises(JobPayloadError) as caught:
        restarted.read(input_ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    assert caught.value.code == "payload_expired"
    assert restarted.read(
        event_ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_TERMINAL_EVENT
    ) == {"body": {"status": "done"}}

    clock.advance(60)
    again = _encrypted(root, clock)
    assert again.recover({input_ref.ref, event_ref.ref})["expired_removed"] == 1
    assert _payload_files(root) == []


@pytest.mark.parametrize(
    ("provider", "error_type", "code"),
    [
        (
            _provider(KEY_TWO),
            EnvelopeAuthenticationError,
            "envelope_authentication_failed",
        ),
        (
            _provider(KEY_ONE, key_id="jobs-key-2"),
            KeyUnavailableError,
            "key_unavailable",
        ),
    ],
)
def test_wrong_or_missing_key_fails_closed_and_preserves_bytes(
    tmp_path: Path, provider: Any, error_type: type[Exception], code: str
) -> None:
    root = tmp_path / "payloads"
    clock = Clock()
    ref = _encrypted(root, clock).create(
        PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60
    )
    before = _all_bytes(root)
    wrong = _encrypted(root, clock, provider=provider)

    with pytest.raises(error_type) as caught:
        wrong.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)

    assert caught.value.code == code
    assert _all_bytes(root) == before
    assert wrong.contains(ref.ref)
    assert wrong.receipts()[-1].event == "read_refused"
    assert wrong.receipts()[-1].reason == code
    assert (
        _encrypted(root, clock).read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
        == _body()
    )


def test_revoked_key_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "payloads"
    clock = Clock()
    provider = _provider()
    store = _encrypted(root, clock, provider=provider)
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    provider.set_state("jobs-key-1", KeyState.REVOKED, purpose=JOB_PAYLOAD_KEY_PURPOSE)

    with pytest.raises(KeyRevokedError):
        store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    with pytest.raises(KeyUnavailableError):
        store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    assert len(_payload_files(root)) == 1


def _rewrite(path: Path, **changes: Any) -> None:
    document = json.loads(path.read_text(encoding="ascii"))
    document.update(changes)
    path.write_text(json.dumps(document), encoding="ascii")


def _flip_envelope(path: Path) -> None:
    document = json.loads(path.read_text(encoding="ascii"))
    raw = bytearray(base64.b64decode(document["envelope"][len("omae:") :]))
    raw[-1] ^= 0x01
    document["envelope"] = "omae:" + base64.b64encode(bytes(raw)).decode("ascii")
    path.write_text(json.dumps(document), encoding="ascii")


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize(
    "damage",
    ["flip_envelope", "extend_deadline", "move_job", "truncate", "not_json"],
)
def test_damaged_envelopes_fail_closed(
    tmp_path: Path, damage: str, restart: bool
) -> None:
    root = tmp_path / "payloads"
    clock = Clock()
    store = _encrypted(root, clock)
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    path = _payload_files(root)[0]
    job_id = JOB_ID
    if damage == "flip_envelope":
        _flip_envelope(path)
    elif damage == "extend_deadline":
        _rewrite(path, expires_at="2026-01-02T00:00:00Z")
    elif damage == "move_job":
        _rewrite(path, job_id=OTHER_JOB_ID)
        job_id = OTHER_JOB_ID if restart else JOB_ID
    elif damage == "truncate":
        path.write_bytes(path.read_bytes()[:40])
    else:
        path.write_bytes(b"\xff not json")
    damaged_bytes = path.read_bytes()
    reader = _encrypted(root, clock) if restart else store

    with pytest.raises(EncryptionError) as caught:
        reader.read(ref.ref, job_id=job_id, kind=PAYLOAD_KIND_INPUT)

    expected = "payload_damaged"
    if restart and damage in {"extend_deadline", "move_job"}:
        # The rewritten cleartext metadata is self-consistent, but the AEAD
        # associated data no longer matches the sealed identity.
        expected = "envelope_authentication_failed"
    assert caught.value.code == expected
    assert path.read_bytes() == damaged_bytes


def test_failed_writes_leave_no_payload_and_keep_existing_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "payloads"
    store = _encrypted(root, Clock())
    kept = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    before = sorted(path.name for path in root.iterdir())

    def disk_full(_descriptor: int) -> None:
        raise OSError(errno.ENOSPC, "synthetic disk full")

    def interrupted(_source: Any, _destination: Any) -> None:
        raise OSError(errno.EIO, "synthetic interrupted write")

    for attribute, replacement, code in [
        ("fsync", disk_full, "payload_storage_full"),
        ("replace", interrupted, "payload_write_failed"),
    ]:
        with monkeypatch.context() as patched:
            patched.setattr(job_payloads.os, attribute, replacement)
            with pytest.raises(JobPayloadError) as caught:
                store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
        assert caught.value.code == code
        assert "synthetic" not in str(caught.value)
        assert sorted(path.name for path in root.iterdir()) == before
        assert store.retained_count() == 1

    assert store.read(kept.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT) == _body()


def test_failed_cleanup_is_reported_and_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "payloads"
    store = _encrypted(root, Clock())
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    original_unlink = Path.unlink

    def refuse(self: Path, *args: Any, **kwargs: Any) -> None:
        raise OSError(errno.EACCES, "synthetic unlink failure")

    with monkeypatch.context() as patched:
        patched.setattr(Path, "unlink", refuse)
        store.delete(ref.ref, reason="consumed")
        assert store.receipts()[-1].event == "cleanup_failed"
        assert store.lifecycle_evidence()["cleanup_pending_count"] == 1
        with pytest.raises(JobPayloadError) as caught:
            store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
        assert caught.value.code == "payload_deleted"
        assert len(_payload_files(root)) == 1

    assert Path.unlink is original_unlink
    store.sweep_expired()
    assert _payload_files(root) == []
    assert store.receipts()[-1].event == "deleted"
    assert store.lifecycle_evidence()["cleanup_pending_count"] == 0


def test_recover_removes_crash_remnants_and_orphans(tmp_path: Path) -> None:
    root = tmp_path / "payloads"
    clock = Clock()
    store = _encrypted(root, clock)
    live = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    orphan = store.create(
        PAYLOAD_KIND_INPUT, OTHER_JOB_ID, _body(), retention_seconds=60
    )
    remnant = root / f".{orphan.ref}.abc123_x.tmp"
    remnant.write_bytes(b"synthetic ciphertext remnant")
    damaged_live = "jp_" + "a" * 32
    damaged_orphan = "jp_" + "b" * 32
    (root / f"{damaged_live}.payload.json").write_bytes(b"{broken")
    (root / f"{damaged_orphan}.payload.json").write_bytes(b"{broken")
    unrelated = root / "operator-notes.txt"
    unrelated.write_text("left alone", encoding="utf-8")

    restarted = _encrypted(root, clock)
    counts = restarted.recover({live.ref, damaged_live})

    assert counts == {
        "cleanup_failed": 0,
        "damaged_retained": 1,
        "orphans_removed": 2,
        "remnants_removed": 1,
        "retained": 1,
    }
    assert sorted(path.name for path in root.iterdir()) == sorted(
        [f"{live.ref}.payload.json", f"{damaged_live}.payload.json", unrelated.name]
    )
    with pytest.raises(JobPayloadError) as caught:
        restarted.read(damaged_live, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    assert caught.value.code == "payload_damaged"
    with pytest.raises(JobPayloadError) as missing:
        restarted.read(orphan.ref, job_id=OTHER_JOB_ID, kind=PAYLOAD_KIND_INPUT)
    assert missing.value.code == "payload_deleted"
    reasons = {(r.event, r.reason) for r in restarted.receipts()}
    assert ("deleted", "crash_remnant") in reasons
    assert ("deleted", "orphan") in reasons


def test_deleted_payload_restored_on_disk_is_not_resurrected(tmp_path: Path) -> None:
    root = tmp_path / "payloads"
    clock = Clock()
    store = _encrypted(root, clock)
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    path = _payload_files(root)[0]
    saved = path.read_bytes()
    store.delete(ref.ref, reason="consumed")
    path.write_bytes(saved)

    with pytest.raises(JobPayloadError):
        store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)

    restarted = _encrypted(root, clock)
    assert restarted.recover(set())["orphans_removed"] == 1
    assert _payload_files(root) == []


def test_memory_store_writes_nothing_to_disk(tmp_path: Path) -> None:
    store = MemoryJobPayloadStore(clock=Clock())
    before = sorted(tmp_path.rglob("*"))
    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, _body(), retention_seconds=60)
    store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT)
    store.delete(ref.ref, reason="consumed")

    assert sorted(tmp_path.rglob("*")) == before
    assert store.retained_count() == 0
    assert store._blobs == {}
    assert ref.digest is None


def test_seeded_bodies_round_trip_in_both_stores(tmp_path: Path) -> None:
    generator = random.Random(3572)
    stores = [
        MemoryJobPayloadStore(clock=Clock()),
        _encrypted(tmp_path / "payloads", Clock()),
    ]
    for _ in range(40):
        body = {
            f"k{index}": generator.choice(
                [
                    generator.randint(-(10**6), 10**6),
                    "".join(chr(generator.randint(32, 0x2FFF)) for _ in range(8)),
                    [generator.random(), None, True],
                    {"nested": [generator.randint(0, 9)]},
                ]
            )
            for index in range(generator.randint(0, 5))
        }
        for store in stores:
            ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, body, retention_seconds=60)
            assert store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT) == body


_JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=20),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=12,
)


@pytest.mark.fuzz
@settings(max_examples=60, deadline=None)
@given(body=st.dictionaries(st.text(max_size=8), _JSON_VALUES, max_size=5))
def test_property_encrypted_round_trip_and_binding(tmp_path_factory, body) -> None:
    root = tmp_path_factory.mktemp("payload-property")
    store = _encrypted(root, Clock())

    ref = store.create(PAYLOAD_KIND_INPUT, JOB_ID, body, retention_seconds=60)

    assert store.read(ref.ref, job_id=JOB_ID, kind=PAYLOAD_KIND_INPUT) == body
    with pytest.raises(JobPayloadError):
        store.read(ref.ref, job_id=OTHER_JOB_ID, kind=PAYLOAD_KIND_INPUT)
    _rewrite(_payload_files(root)[0], job_id=OTHER_JOB_ID)
    with pytest.raises(EncryptionError):
        _encrypted(root, Clock()).read(
            ref.ref, job_id=OTHER_JOB_ID, kind=PAYLOAD_KIND_INPUT
        )
