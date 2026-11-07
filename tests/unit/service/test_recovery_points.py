"""Encrypted, integrity-checked recovery points for durable service state.

All state, keys, and sentinels are synthetic (``tests/fixtures``).  The
Journey store stays open while backups run, so captures exercise the SQLite
online backup API against a live write-ahead log.
"""

from __future__ import annotations

import errno
import json
import os
import sqlite3
import sys
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from openmed.core.aead_envelope import EnvelopeCipher
from openmed.core.key_provider import KeyState
from openmed.service import recovery
from openmed.service.recovery import (
    BACKUP_KEY_PURPOSE,
    EXCLUDED_STATE,
    JOB_STORE_PATH,
    JOURNEY_DATABASE_PATH,
    RECOVERY_METADATA_DIRECTORY,
    REGISTRY_STATE_FILE,
    ComponentCategory,
    RecoveryEpochContext,
    RecoveryError,
    RecoveryOutcome,
    RecoveryReceipt,
    ServiceStateSources,
    activate_restored_state,
    create_recovery_point,
    discard_restored_state,
    inventory_service_state,
    journey_store_check,
    restore_recovery_point,
    verify_recovery_point,
)
from openmed.structured.store import JourneyStoreEncryption
from openmed.structured.store.migrations import LATEST_MIGRATION_VERSION
from tests.fixtures.service_recovery import (
    BACKUP_KEY,
    JOURNEY_KEY,
    OTHER_BACKUP_KEY,
    ROTATED_BACKUP_KEY,
    SENTINEL,
    SENTINEL_BYTES,
    SyntheticServiceState,
    backup_provider,
    build_service_state,
    journey_encryption,
    journey_provider,
    logical_snapshot,
    tree_bytes,
)

pytest.importorskip("cryptography")

POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="atomic symlink activation is POSIX-only"
)
T0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
STREAM_HEADER_BYTES = 9


@pytest.fixture(autouse=True)
def _development_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)


@pytest.fixture
def state(tmp_path: Path) -> Iterator[SyntheticServiceState]:
    service = build_service_state(tmp_path / "state")
    yield service
    service.close()


def _backup(
    state: SyntheticServiceState,
    destination: Path,
    *,
    provider: Any = None,
    **options: Any,
) -> RecoveryReceipt:
    options.setdefault("writers_stopped", True)
    return create_recovery_point(
        state.sources(),
        destination,
        key_provider=provider or backup_provider(),
        **options,
    )


def _cipher(provider: Any = None) -> EnvelopeCipher:
    return EnvelopeCipher(provider or backup_provider(), purpose=BACKUP_KEY_PURPOSE)


def _point_id(point: Path) -> str:
    return json.loads((point / "recovery-point.json").read_text())["recovery_point_id"]


def _manifest(point: Path, provider: Any = None) -> dict[str, Any]:
    context = {"point": _point_id(point), "record": "recovery_manifest"}
    sealed = (point / "manifest.omae").read_bytes()
    return json.loads(_cipher(provider).open(sealed, context=context))


def _rewrite_manifest(
    point: Path,
    mutate: Callable[[dict[str, Any]], None],
    provider: Any = None,
) -> None:
    """Forge a manifest with the synthetic key, keeping it self-consistent."""

    manifest = _manifest(point, provider)
    mutate(manifest)
    manifest["components"] = [
        item.to_dict()
        for item in recovery._components(manifest["objects"], manifest["boundary"])
    ]
    manifest["content_digest"] = recovery._content_digest(
        manifest["boundary"], manifest["objects"]
    )
    context = {"point": _point_id(point), "record": "recovery_manifest"}
    (point / "manifest.omae").write_bytes(
        _cipher(provider).seal(recovery._canonical_json(manifest), context=context)
    )


def _object_path(point: Path, logical: str, provider: Any = None) -> Path:
    entry = next(
        item
        for item in _manifest(point, provider)["objects"]
        if item["path"] == logical
    )
    return point / "objects" / f"{entry['object']}.omrc"


def _split_stream(data: bytes) -> tuple[bytes, list[bytes]]:
    header = data[:STREAM_HEADER_BYTES]
    records = []
    offset = STREAM_HEADER_BYTES
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        records.append(data[offset : offset + 4 + length])
        offset += 4 + length
    return header, records


def _live_digest(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "journey" not in path.relative_to(root).parts
    }


# ---------------------------------------------------------------------------
# Inventory and layout
# ---------------------------------------------------------------------------


def test_inventory_is_category_only_and_classifies_excluded_state(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    inventory = inventory_service_state(state.sources())
    document = inventory.to_dict()
    encoded = json.dumps(document)

    entries = {entry["category"]: entry for entry in document["entries"]}
    assert document["schema"] == "openmed.service.recovery-inventory.v1"
    assert [category.value for category in ComponentCategory] == [
        name for name in entries if entries[name]["disposition"] == "authoritative"
    ]
    assert entries["journey_metadata"]["schema_version"] == LATEST_MIGRATION_VERSION
    assert entries["journey_artifacts"]["objects"] == 2
    assert entries["journey_encryption"]["schema_version"] == 1
    assert entries["model_registry"]["schema_version"] == 2
    assert entries["audit_chain"]["schema_version"] == 1
    assert entries["job_metadata"]["schema_version"] == 1
    for name, disposition in EXCLUDED_STATE.items():
        assert entries[name]["disposition"] == disposition.value
        assert entries[name]["objects"] == 0
    assert entries["encryption_keys"]["disposition"] == "excluded_secret"
    assert str(tmp_path) not in encoded
    assert SENTINEL not in encoded
    assert "journey-key-1" not in encoded


def test_inventory_refuses_symlinked_sources_without_paths(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    link = tmp_path / "linked-jobs.json"
    link.symlink_to(state.root / JOB_STORE_PATH)

    with pytest.raises(RecoveryError) as excinfo:
        inventory_service_state(ServiceStateSources(job_store=link))

    assert excinfo.value.code == "source_invalid"
    assert str(tmp_path) not in str(excinfo.value)


def test_restore_layout_matches_the_local_journey_store(
    state: SyntheticServiceState,
) -> None:
    from openmed.structured.store import local

    assert local._ENCRYPTION_MARKER == recovery._JOURNEY_MARKER_NAME
    assert (state.root / JOURNEY_DATABASE_PATH).is_file()
    assert (state.root / recovery.JOURNEY_MARKER_PATH).is_file()
    blobs = list((state.root / recovery.JOURNEY_BLOB_ROOT).rglob("*"))
    assert sum(path.is_file() for path in blobs) == 2
    assert repr(state.sources()).startswith("ServiceStateSources(components=")
    assert str(state.root) not in repr(state.sources())


# ---------------------------------------------------------------------------
# Full backup and restore
# ---------------------------------------------------------------------------


def test_full_backup_restores_equivalent_logical_state(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    receipt = _backup(state, tmp_path / "rp-full", chunk_bytes=4096)

    assert receipt.completed, receipt.reason
    assert receipt.kind == "full"
    assert receipt.sequence == 0
    assert receipt.counts["objects_stored"] == receipt.counts["objects_total"]
    assert {item.category for item in receipt.components} == {
        category.value for category in ComponentCategory
    }
    verification = verify_recovery_point(
        tmp_path / "rp-full", key_provider=backup_provider()
    )
    assert verification.completed, verification.reason
    assert verification.content_digest == receipt.content_digest
    assert verification.manifest_digest == receipt.manifest_digest

    restored = restore_recovery_point(
        tmp_path / "rp-full", tmp_path / "restored", key_provider=backup_provider()
    )

    assert restored.completed, restored.reason
    assert restored.recovery_point_id == receipt.recovery_point_id
    assert (tmp_path / "restored" / RECOVERY_METADATA_DIRECTORY).is_dir()
    expected = logical_snapshot(state.root, state.encryption)
    assert logical_snapshot(tmp_path / "restored", state.encryption) == expected


def test_backups_are_deterministic_and_carry_no_plaintext_or_keys(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    first = _backup(state, tmp_path / "rp-one")
    second = _backup(state, tmp_path / "rp-two")

    assert first.completed and second.completed
    assert first.content_digest == second.content_digest
    assert first.components == second.components
    assert first.recovery_point_id != second.recovery_point_id
    assert first.manifest_digest != second.manifest_digest
    for point in (tmp_path / "rp-one", tmp_path / "rp-two"):
        stored = tree_bytes(point)
        assert SENTINEL_BYTES not in stored
        assert JOURNEY_KEY not in stored
        assert BACKUP_KEY not in stored
        assert str(tmp_path).encode() not in stored
        assert b"journey-key-1" not in stored
    for receipt in (first, second):
        encoded = receipt.to_json()
        assert SENTINEL not in encoded
        assert str(tmp_path) not in encoded
        assert "backup-key-1" not in encoded
    assert not list(tmp_path.glob(".recovery-partial-*"))


def test_plaintext_development_store_is_sealed_before_leaving_staging(
    tmp_path: Path,
) -> None:
    service = build_service_state(
        tmp_path / "state", encryption=JourneyStoreEncryption.development_plaintext()
    )
    try:
        assert SENTINEL_BYTES in tree_bytes(service.root)
        receipt = _backup(service, tmp_path / "rp", chunk_bytes=1024)
    finally:
        service.close()

    assert receipt.completed, receipt.reason
    assert SENTINEL_BYTES not in tree_bytes(tmp_path / "rp")


def test_backup_requires_a_quiesce_boundary(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    receipt = _backup(state, tmp_path / "rp", writers_stopped=False)

    assert receipt.outcome is RecoveryOutcome.REFUSED
    assert receipt.reason == "quiesce_required"
    assert not (tmp_path / "rp").exists()


def test_quiesce_hook_wraps_capture_and_is_recorded(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    events: list[str] = []

    class Drain:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, *exc: object) -> None:
            events.append("exit")

    receipt = _backup(
        state, tmp_path / "rp", writers_stopped=False, quiesce=lambda: Drain()
    )

    assert receipt.completed, receipt.reason
    assert events == ["enter", "exit"]
    assert _manifest(tmp_path / "rp")["consistency"] == {"mode": "quiesced"}

    def broken() -> Any:
        raise RuntimeError(SENTINEL)

    refused = _backup(state, tmp_path / "rp-broken", quiesce=broken)
    assert refused.reason == "quiesce_failed"
    assert SENTINEL not in refused.to_json()


def test_writer_during_capture_invalidates_the_backup(
    state: SyntheticServiceState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = recovery._store_objects

    def store_while_writing(*args: Any, **kwargs: Any) -> Any:
        state.add_job()
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery, "_store_objects", store_while_writing)

    receipt = _backup(state, tmp_path / "rp")

    assert receipt.reason == "snapshot_boundary_moved"
    assert not (tmp_path / "rp").exists()
    assert not list(tmp_path.glob(".recovery-partial-*"))


def test_backup_refuses_dangling_artifact_references(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    blob = next(
        path
        for path in (state.root / recovery.JOURNEY_BLOB_ROOT).rglob("*")
        if path.is_file()
    )
    blob.unlink()

    receipt = _backup(state, tmp_path / "rp")

    assert receipt.reason == "cross_reference_missing_artifact"
    assert not (tmp_path / "rp").exists()


def test_backup_refuses_existing_destination_and_bad_sources(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    (tmp_path / "taken").mkdir()
    assert _backup(state, tmp_path / "taken").reason == "destination_exists"

    (state.root / JOB_STORE_PATH).write_text("[1, 2", encoding="utf-8")
    assert _backup(state, tmp_path / "rp").reason == "source_invalid"

    missing = ServiceStateSources(job_store=tmp_path / "absent.json")
    receipt = create_recovery_point(
        missing, tmp_path / "rp2", key_provider=backup_provider(), writers_stopped=True
    )
    assert receipt.reason == "source_unavailable"


# ---------------------------------------------------------------------------
# Tampering, keys, and compatibility
# ---------------------------------------------------------------------------


@pytest.fixture
def point(state: SyntheticServiceState, tmp_path: Path) -> Path:
    destination = tmp_path / "rp"
    receipt = _backup(state, destination, chunk_bytes=4096)
    assert receipt.completed, receipt.reason
    return destination


def _assert_refused(point: Path, tmp_path: Path, reason: str, **kwargs: Any) -> None:
    provider = kwargs.pop("provider", None) or backup_provider()
    verification = verify_recovery_point(point, key_provider=provider, **kwargs)
    assert verification.outcome is RecoveryOutcome.REFUSED
    assert verification.reason == reason
    target = tmp_path / f"restore-{reason}"
    restored = restore_recovery_point(point, target, key_provider=provider, **kwargs)
    assert restored.reason == reason
    assert not target.exists()


def test_tampered_object_byte_is_detected(point: Path, tmp_path: Path) -> None:
    path = _object_path(point, JOURNEY_DATABASE_PATH)
    data = bytearray(path.read_bytes())
    data[-5] ^= 0x01
    path.write_bytes(bytes(data))

    _assert_refused(point, tmp_path, "object_authentication_failed")


def test_truncated_object_is_detected(point: Path, tmp_path: Path) -> None:
    path = _object_path(point, JOURNEY_DATABASE_PATH)
    header, records = _split_stream(path.read_bytes())
    assert len(records) > 2
    path.write_bytes(header + b"".join(records[:-1]))
    _assert_refused(point, tmp_path, "object_corrupt")

    path.write_bytes(header + b"".join(records)[:-3])
    _assert_refused(point, tmp_path, "object_corrupt")


def test_reordered_chunks_are_detected(point: Path, tmp_path: Path) -> None:
    path = _object_path(point, JOURNEY_DATABASE_PATH)
    header, records = _split_stream(path.read_bytes())
    records[0], records[1] = records[1], records[0]
    path.write_bytes(header + b"".join(records))

    _assert_refused(point, tmp_path, "object_authentication_failed")


def test_objects_cannot_move_between_recovery_points(
    state: SyntheticServiceState, point: Path, tmp_path: Path
) -> None:
    other = tmp_path / "rp-other"
    assert _backup(state, other, chunk_bytes=4096).completed
    name = _object_path(point, JOB_STORE_PATH).name
    (point / "objects" / name).write_bytes((other / "objects" / name).read_bytes())

    _assert_refused(point, tmp_path, "object_authentication_failed")


def test_missing_object_is_detected(point: Path, tmp_path: Path) -> None:
    _object_path(point, REGISTRY_STATE_FILE).unlink()

    _assert_refused(point, tmp_path, "object_missing")


def test_tampered_manifest_and_descriptor_are_detected(
    point: Path, tmp_path: Path
) -> None:
    sealed = bytearray((point / "manifest.omae").read_bytes())
    original_manifest = bytes(sealed)
    sealed[-1] ^= 0x01
    (point / "manifest.omae").write_bytes(bytes(sealed))
    _assert_refused(point, tmp_path, "manifest_authentication_failed")
    (point / "manifest.omae").write_bytes(original_manifest)

    descriptor_path = point / "recovery-point.json"
    original = descriptor_path.read_text()
    descriptor = json.loads(original)
    descriptor["recovery_point_id"] = "rp_" + "0" * 32
    descriptor_path.write_text(json.dumps(descriptor))
    _assert_refused(point, tmp_path, "manifest_authentication_failed")

    descriptor = json.loads(original)
    descriptor["kind"] = "incremental"
    descriptor_path.write_text(json.dumps(descriptor))
    _assert_refused(point, tmp_path, "descriptor_mismatch")

    descriptor_path.write_text("{not json")
    _assert_refused(point, tmp_path, "descriptor_invalid")


def test_wrong_unknown_and_revoked_keys_are_rejected(
    point: Path, tmp_path: Path
) -> None:
    _assert_refused(
        point,
        tmp_path,
        "manifest_authentication_failed",
        provider=backup_provider(OTHER_BACKUP_KEY),
    )
    _assert_refused(
        point,
        tmp_path,
        "key_unavailable",
        provider=backup_provider(BACKUP_KEY, key_id="backup-key-9"),
    )
    revoked = backup_provider()
    revoked.set_state("backup-key-1", KeyState.REVOKED, purpose=BACKUP_KEY_PURPOSE)
    receipt = verify_recovery_point(point, key_provider=revoked)
    assert receipt.reason == "key_revoked"


def test_rotated_keys_still_restore_older_points(point: Path, tmp_path: Path) -> None:
    provider = backup_provider()
    provider.rotate("backup-key-2", ROTATED_BACKUP_KEY, purpose=BACKUP_KEY_PURPOSE)

    restored = restore_recovery_point(
        point, tmp_path / "restored", key_provider=provider
    )

    assert restored.completed, restored.reason


def test_incompatible_schemas_are_refused(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    connection = sqlite3.connect(state.root / JOURNEY_DATABASE_PATH)
    try:
        connection.execute(
            "INSERT INTO schema_migrations(version, name, checksum) VALUES (?, ?, ?)",
            (LATEST_MIGRATION_VERSION + 50, "future", "synthetic"),
        )
        connection.commit()
    finally:
        connection.close()
    receipt = _backup(state, tmp_path / "rp-future")
    assert receipt.completed, receipt.reason
    _assert_refused(tmp_path / "rp-future", tmp_path, "schema_incompatible")


def test_retired_registry_schema_is_refused(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    (state.root / REGISTRY_STATE_FILE).write_text(
        json.dumps({"schema_version": 1, "slots": {}}), encoding="utf-8"
    )
    assert _backup(state, tmp_path / "rp-old").completed

    _assert_refused(tmp_path / "rp-old", tmp_path, "schema_incompatible")


def test_forged_cross_reference_and_boundary_mismatches_are_detected(
    point: Path, tmp_path: Path
) -> None:
    original = (point / "manifest.omae").read_bytes()

    def drop_blob(manifest: dict[str, Any]) -> None:
        manifest["objects"] = [
            item
            for item in manifest["objects"]
            if item["category"] != ComponentCategory.JOURNEY_ARTIFACTS.value
        ][:]
        blobs = [
            item
            for item in _manifest(point)["objects"]
            if item["category"] == ComponentCategory.JOURNEY_ARTIFACTS.value
        ]
        manifest["objects"] = sorted(
            [*manifest["objects"], *blobs[1:]], key=lambda item: item["path"]
        )

    _rewrite_manifest(point, drop_blob)
    _assert_refused(point, tmp_path, "cross_reference_missing_artifact")

    (point / "manifest.omae").write_bytes(original)

    def shift_jobs(manifest: dict[str, Any]) -> None:
        manifest["boundary"]["jobs"]["records"] += 1

    _rewrite_manifest(point, shift_jobs)
    _assert_refused(point, tmp_path, "snapshot_boundary_mismatch")

    (point / "manifest.omae").write_bytes(original)

    def shift_revision(manifest: dict[str, Any]) -> None:
        manifest["boundary"]["journey"]["revision"] += 1

    _rewrite_manifest(point, shift_revision)
    _assert_refused(point, tmp_path, "snapshot_boundary_mismatch")


def test_inconsistent_manifest_is_invalid(point: Path, tmp_path: Path) -> None:
    manifest = _manifest(point)
    manifest["objects"][0]["size"] += 1
    context = {"point": _point_id(point), "record": "recovery_manifest"}
    (point / "manifest.omae").write_bytes(
        _cipher().seal(recovery._canonical_json(manifest), context=context)
    )

    _assert_refused(point, tmp_path, "manifest_invalid")


def test_stale_and_expired_points_fail_closed(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    destination = tmp_path / "rp"
    receipt = _backup(state, destination, retention_days=1, clock=lambda: T0)
    assert receipt.completed, receipt.reason

    soon = verify_recovery_point(
        destination,
        key_provider=backup_provider(),
        max_age=timedelta(hours=1),
        clock=lambda: T0 + timedelta(minutes=30),
    )
    assert soon.completed, soon.reason
    stale = verify_recovery_point(
        destination,
        key_provider=backup_provider(),
        max_age=timedelta(hours=1),
        clock=lambda: T0 + timedelta(hours=2),
    )
    assert stale.reason == "recovery_point_stale"
    expired = restore_recovery_point(
        destination,
        tmp_path / "restored",
        key_provider=backup_provider(),
        clock=lambda: T0 + timedelta(days=2),
    )
    assert expired.reason == "recovery_point_expired"
    assert not (tmp_path / "restored").exists()


# ---------------------------------------------------------------------------
# Incremental chains
# ---------------------------------------------------------------------------


def test_incremental_chain_stores_changes_and_restores_equivalent_state(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    full = tmp_path / "rp-0"
    first = tmp_path / "rp-1"
    second = tmp_path / "rp-2"
    assert _backup(state, full).completed
    state.add_note()
    state.add_job()
    delta = _backup(state, first, parent=full)
    state.add_audit_entry()
    state.add_note()
    final = _backup(state, second, parent=first)

    assert delta.completed and final.completed, (delta.reason, final.reason)
    assert delta.kind == "incremental" and delta.sequence == 1
    assert final.sequence == 2
    # Unchanged blobs, the encryption marker, and registry state are reused.
    assert delta.counts["objects_reused"] >= 4
    assert delta.counts["objects_stored"] == 3
    assert final.counts["objects_stored"] == 3

    restored = restore_recovery_point(
        second,
        tmp_path / "restored",
        key_provider=backup_provider(),
        ancestors=(full, first),
    )

    assert restored.completed, restored.reason
    assert restored.chain_length == 3
    expected = logical_snapshot(state.root, state.encryption)
    assert logical_snapshot(tmp_path / "restored", state.encryption) == expected


@pytest.fixture
def chain(state: SyntheticServiceState, tmp_path: Path) -> tuple[Path, Path, Path]:
    full, first, second = tmp_path / "c-0", tmp_path / "c-1", tmp_path / "c-2"
    assert _backup(state, full).completed
    state.add_note()
    assert _backup(state, first, parent=full).completed
    state.add_note()
    assert _backup(state, second, parent=first).completed
    return full, first, second


def test_missing_increment_is_refused(
    chain: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    full, _first, second = chain

    _assert_refused(second, tmp_path, "chain_incomplete", ancestors=(full,))
    _assert_refused(second, tmp_path, "chain_incomplete")


def test_substituted_parent_is_refused(
    chain: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    full, first, second = chain

    def extend_retention(manifest: dict[str, Any]) -> None:
        manifest["retention"]["expires_at"] = "2999-01-01T00:00:00Z"

    _rewrite_manifest(first, extend_retention)

    _assert_refused(second, tmp_path, "chain_invalid", ancestors=(full, first))


def test_duplicate_ancestors_are_refused(
    chain: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    full, first, second = chain

    _assert_refused(second, tmp_path, "chain_invalid", ancestors=(full, full, first))


def test_regressed_sequence_bounds_are_refused(
    chain: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    full, first, second = chain
    parent_revision = _manifest(first)["boundary"]["journey"]["revision"]

    def regress(manifest: dict[str, Any]) -> None:
        manifest["boundary"]["journey"]["revision"] = parent_revision - 1

    _rewrite_manifest(second, regress)

    _assert_refused(
        second, tmp_path, "chain_sequence_regressed", ancestors=(full, first)
    )


def test_chain_length_is_bounded(
    state: SyntheticServiceState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery, "MAX_CHAIN_LENGTH", 2)
    assert _backup(state, tmp_path / "b-0").completed
    assert _backup(state, tmp_path / "b-1", parent=tmp_path / "b-0").completed

    receipt = _backup(state, tmp_path / "b-2", parent=tmp_path / "b-1")

    assert receipt.reason == "chain_too_long"


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def _fail_writes_after(monkeypatch: pytest.MonkeyPatch, allowed: int) -> None:
    original = recovery._write_all
    calls = {"count": 0}

    def write_all(descriptor: int, data: bytes) -> None:
        calls["count"] += 1
        if calls["count"] > allowed:
            raise OSError(errno.ENOSPC, "No space left on device")
        original(descriptor, data)

    monkeypatch.setattr(recovery, "_write_all", write_all)


def test_disk_exhaustion_during_backup_leaves_no_point(
    state: SyntheticServiceState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_writes_after(monkeypatch, 3)

    receipt = _backup(state, tmp_path / "rp")

    assert receipt.outcome is RecoveryOutcome.FAILED
    assert receipt.reason == "backup_write_failed"
    assert not (tmp_path / "rp").exists()
    assert not list(tmp_path.glob(".recovery-partial-*"))


def test_disk_exhaustion_during_restore_leaves_live_state_untouched(
    state: SyntheticServiceState,
    point: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_before = _live_digest(state.root)
    point_before = tree_bytes(point)
    _fail_writes_after(monkeypatch, 2)

    receipt = restore_recovery_point(
        point, tmp_path / "restored", key_provider=backup_provider()
    )

    assert receipt.outcome is RecoveryOutcome.FAILED
    assert receipt.reason == "restore_write_failed"
    assert not (tmp_path / "restored").exists()
    assert _live_digest(state.root) == live_before
    assert tree_bytes(point) == point_before


def test_interrupted_restore_removes_the_partial_target(
    point: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(recovery, "_verify_semantics", interrupt)

    with pytest.raises(KeyboardInterrupt):
        restore_recovery_point(
            point, tmp_path / "restored", key_provider=backup_provider()
        )

    assert not (tmp_path / "restored").exists()


def test_restore_refuses_existing_or_overlapping_targets(
    point: Path, tmp_path: Path
) -> None:
    (tmp_path / "taken").mkdir()
    taken = restore_recovery_point(
        point, tmp_path / "taken", key_provider=backup_provider()
    )
    assert taken.reason == "target_exists"
    assert list((tmp_path / "taken").iterdir()) == []

    live = tmp_path / "live-state"
    live.mkdir()
    nested = restore_recovery_point(
        point, live / "nested", key_provider=backup_provider(), live=live
    )
    assert nested.reason == "target_overlaps_live"
    assert not (live / "nested").exists()


# ---------------------------------------------------------------------------
# Activation and cleanup
# ---------------------------------------------------------------------------


class _EpochRecorder:
    def __init__(self, live: Path, *, refuse: bool = False) -> None:
        self.live = live
        self.refuse = refuse
        self.contexts: list[RecoveryEpochContext] = []
        self.live_targets: list[str | None] = []

    def open_recovery_epoch(self, context: RecoveryEpochContext) -> None:
        self.contexts.append(context)
        self.live_targets.append(
            os.readlink(self.live) if os.path.islink(self.live) else None
        )
        if self.refuse:
            raise RuntimeError(SENTINEL)


@pytest.fixture
def staged(point: Path, tmp_path: Path) -> Path:
    target = tmp_path / "generation-1"
    receipt = restore_recovery_point(point, target, key_provider=backup_provider())
    assert receipt.completed, receipt.reason
    return target


@POSIX_ONLY
def test_activation_swaps_the_live_link_after_checks_and_epoch_hooks(
    state: SyntheticServiceState, point: Path, staged: Path, tmp_path: Path
) -> None:
    live = tmp_path / "live"
    hook = _EpochRecorder(live)
    expected = logical_snapshot(state.root, state.encryption)

    receipt = activate_restored_state(
        staged,
        live,
        key_provider=backup_provider(),
        checks=[journey_store_check(state.encryption)],
        epoch_hooks=[hook],
    )

    assert receipt.completed, (receipt.reason, receipt.details)
    assert live.is_symlink()
    assert live.resolve() == staged.resolve()
    assert receipt.counts == {
        "checks_passed": 1,
        "epoch_hooks": 1,
        "previous_target": 0,
    }
    assert hook.live_targets == [None]
    assert hook.contexts[0].recovery_point_id == _point_id(point)
    assert hook.contexts[0].manifest_digest == receipt.manifest_digest
    assert {item.category for item in receipt.components} == {
        category.value for category in ComponentCategory
    }
    assert logical_snapshot(live, state.encryption) == expected

    second = tmp_path / "generation-2"
    assert restore_recovery_point(
        point, second, key_provider=backup_provider(), live=live
    ).completed
    again = activate_restored_state(second, live, key_provider=backup_provider())
    assert again.completed, again.reason
    assert again.counts["previous_target"] == 1
    assert live.resolve() == second.resolve()
    assert staged.is_dir()


@POSIX_ONLY
def test_activation_refuses_without_touching_live_state(
    state: SyntheticServiceState, staged: Path, tmp_path: Path
) -> None:
    unmanaged = tmp_path / "plain-live"
    unmanaged.mkdir()
    (unmanaged / "keep.txt").write_text("synthetic", encoding="utf-8")
    refused = activate_restored_state(staged, unmanaged, key_provider=backup_provider())
    assert refused.reason == "live_target_unmanaged"
    assert (unmanaged / "keep.txt").read_text(encoding="utf-8") == "synthetic"

    live = tmp_path / "live"
    hook = _EpochRecorder(live, refuse=True)
    refused = activate_restored_state(
        staged, live, key_provider=backup_provider(), epoch_hooks=[hook]
    )
    assert refused.reason == "recovery_epoch_refused"
    assert SENTINEL not in refused.to_json()
    assert not os.path.lexists(live)

    wrong_journey = journey_encryption(
        journey_provider(bytes(range(200, 232)), key_id="journey-key-1")
    )
    refused = activate_restored_state(
        staged,
        live,
        key_provider=backup_provider(),
        checks=[journey_store_check(wrong_journey)],
    )
    assert refused.reason == "activation_check_failed"
    assert refused.details["check"] == "journey_store"
    assert recovery._CODE_RE.fullmatch(refused.details["check_reason"])
    assert not os.path.lexists(live)

    refused = activate_restored_state(
        staged, live, key_provider=backup_provider(), checks=[lambda root: False]
    )
    assert refused.details == {"check": "custom", "check_reason": "check_failed"}
    assert not os.path.lexists(live)


@POSIX_ONLY
def test_activation_refuses_modified_or_incomplete_staged_state(
    staged: Path, tmp_path: Path
) -> None:
    live = tmp_path / "live"
    jobs = staged / JOB_STORE_PATH
    original = jobs.read_bytes()
    jobs.write_bytes(original + b" ")
    refused = activate_restored_state(staged, live, key_provider=backup_provider())
    assert refused.reason == "staged_state_modified"
    jobs.write_bytes(original)

    (staged / "jobs" / "extra.json").write_text("{}", encoding="utf-8")
    refused = activate_restored_state(staged, live, key_provider=backup_provider())
    assert refused.reason == "staged_state_modified"
    (staged / "jobs" / "extra.json").unlink()

    seal = staged / RECOVERY_METADATA_DIRECTORY / "staging.omae"
    sealed = bytearray(seal.read_bytes())
    sealed[-1] ^= 0x01
    seal.write_bytes(bytes(sealed))
    refused = activate_restored_state(staged, live, key_provider=backup_provider())
    assert refused.reason == "staging_authentication_failed"

    seal.unlink()
    refused = activate_restored_state(staged, live, key_provider=backup_provider())
    assert refused.reason == "restore_incomplete"
    assert not os.path.lexists(live)


@POSIX_ONLY
def test_activation_refuses_reactivating_the_live_target(
    staged: Path, tmp_path: Path
) -> None:
    live = tmp_path / "live"
    assert activate_restored_state(
        staged, live, key_provider=backup_provider()
    ).completed

    again = activate_restored_state(staged, live, key_provider=backup_provider())

    assert again.reason == "already_active"


def test_activation_is_refused_on_windows(
    staged: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery.sys, "platform", "win32")

    receipt = activate_restored_state(
        staged, tmp_path / "live", key_provider=backup_provider()
    )

    assert receipt.reason == "atomic_swap_unsupported"


@POSIX_ONLY
def test_discard_removes_only_inactive_restored_targets(
    staged: Path, point: Path, tmp_path: Path
) -> None:
    live = tmp_path / "live"
    other = tmp_path / "generation-2"
    assert restore_recovery_point(
        point, other, key_provider=backup_provider()
    ).completed
    assert activate_restored_state(
        staged, live, key_provider=backup_provider()
    ).completed

    assert discard_restored_state(staged, live=live).reason == "staged_is_live"
    assert staged.is_dir()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert discard_restored_state(plain).reason == "not_a_restored_target"
    assert plain.is_dir()

    cleanup = discard_restored_state(other, live=live)

    assert cleanup.completed
    assert cleanup.operation == "cleanup"
    assert cleanup.recovery_point_id == _point_id(point)
    assert not other.exists()


# ---------------------------------------------------------------------------
# Concurrency and receipts
# ---------------------------------------------------------------------------


def test_backups_under_concurrent_reads_are_deterministic(
    state: SyntheticServiceState, tmp_path: Path
) -> None:
    stop = threading.Event()
    reads = {"count": 0, "failures": 0}

    def read_loop() -> None:
        while not stop.is_set():
            result = state.store.metadata.list_facts("subject_aaaaaaaaaaaaaaaa")
            reads["count"] += 1
            reads["failures"] += 0 if result.ok else 1

    reader = threading.Thread(target=read_loop)
    reader.start()
    try:
        receipts = [_backup(state, tmp_path / f"rp-{index}") for index in range(3)]
    finally:
        stop.set()
        reader.join()

    assert all(receipt.completed for receipt in receipts)
    assert len({receipt.content_digest for receipt in receipts}) == 1
    assert reads["failures"] == 0
    restored = restore_recovery_point(
        tmp_path / "rp-2", tmp_path / "restored", key_provider=backup_provider()
    )
    assert restored.completed, restored.reason
    assert logical_snapshot(tmp_path / "restored", state.encryption) == (
        logical_snapshot(state.root, state.encryption)
    )


def test_receipts_are_versioned_and_value_free(point: Path, tmp_path: Path) -> None:
    receipt = verify_recovery_point(point, key_provider=backup_provider())
    document = receipt.to_dict()

    assert document["schema"] == "openmed.service.recovery-receipt.v1"
    assert document["operation"] == "verify"
    assert document["outcome"] == "completed"
    assert recovery._POINT_ID_RE.fullmatch(document["recovery_point_id"])
    assert set(document["boundary"]) == {"audit", "jobs", "journey", "registry"}
    assert receipt.require_completed() is receipt
    refused = verify_recovery_point(
        tmp_path / "missing", key_provider=backup_provider()
    )
    assert refused.reason == "point_unavailable"
    with pytest.raises(RecoveryError) as excinfo:
        refused.require_completed()
    assert excinfo.value.code == "point_unavailable"
    assert str(tmp_path) not in str(excinfo.value)
    assert RecoveryError("not-a-code").code == "recovery_failed"
