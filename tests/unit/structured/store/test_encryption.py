"""Application-layer encryption tests for local Journey storage.

All keys, records, and sentinels are synthetic.  The raw-byte scans read every
file under the store directory, including the SQLite write-ahead log while the
store is open, and the temporary blob file at the moment it is linked.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    ConflictSet,
    DatasetSnapshot,
    EvidenceLocator,
    ResolutionEvent,
    canonical_digest,
    sha256_digest,
)
from openmed.core.aead_envelope import envelope_from_text, read_envelope_header
from openmed.core.key_provider import (
    DataKey,
    InMemoryKeyProvider,
    KeyState,
    KeyUnavailableError,
)
from openmed.interop.ingest import SQLiteIngestionStore
from openmed.structured.store import (
    JOURNEY_KEY_PURPOSE,
    CanonicalRecord,
    DenyStorageOperations,
    JobMetadata,
    JourneyEncryptionError,
    JourneyStoreEncryption,
    LocalArtifactStore,
    LocalJourneyStore,
    PostgresJourneyStore,
    SQLiteJourneyStore,
    StorePoint,
    StoreState,
    resolve_journey_profile,
)
from openmed.structured.store.local import PAYLOAD_RECORD_KEYS

SENTINEL = "ZQJOURNEYSENTINELPHI7341"
SENTINEL_BYTES = SENTINEL.encode("ascii")
KEY_ONE = bytes(range(32))
KEY_TWO = bytes(range(32, 64))
CONTENT = b"synthetic clinical note for " + SENTINEL_BYTES
SECOND_CONTENT = b"synthetic discharge summary for " + SENTINEL_BYTES
T1 = "2026-01-02T03:04:05Z"
T2 = "2026-01-02T04:04:05Z"
T3 = "2026-01-02T05:04:05Z"
T4 = "2026-01-02T06:04:05Z"
SUBJECT = "subject_aaaaaaaaaaaaaaaa"
PLAINTEXT_DEV = JourneyStoreEncryption.development_plaintext()


@pytest.fixture(autouse=True)
def _development_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)


def _provider(material: bytes = KEY_ONE) -> InMemoryKeyProvider:
    provider = InMemoryKeyProvider()
    provider.add_key(
        "key-one",
        material,
        purpose=JOURNEY_KEY_PURPOSE,
        state=KeyState.ACTIVE,
    )
    return provider


def _encrypted(provider: Any) -> JourneyStoreEncryption:
    return JourneyStoreEncryption.encrypted(provider)


def _artifact(
    content: bytes = CONTENT,
    artifact_id: str = "artifact_aaaaaaaaaaaaaaaa",
) -> ClinicalArtifact:
    return ClinicalArtifact(
        artifact_id=artifact_id,
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest(content),
        byte_size=len(content),
        source_id="source_aaaaaaaaaaaaaaaa",
        recorded_at=T1,
        subject_id=SUBJECT,
    )


def _evidence() -> EvidenceLocator:
    return EvidenceLocator(
        locator_id="evidence_aaaaaaaaaaaaaaaa",
        artifact_id=_artifact().artifact_id,
        location_type="text_span",
        location={"start": 0, "end": 9},
    )


def _fact(
    fact_id: str = "fact_aaaaaaaaaaaaaaaa",
    *,
    state: str = "active",
    parents: tuple[str, ...] = (),
) -> ClinicalFact:
    return ClinicalFact(
        fact_id=fact_id,
        subject_id=SUBJECT,
        fact_type="condition",
        value={"code": "synthetic-condition", "note": SENTINEL, "state": state},
        status=state,
        evidence_ids=(_evidence().locator_id,),
        derivation_hash=canonical_digest({"fact": fact_id, "state": state}),
        parent_fact_ids=parents,
    )


def _second_fact() -> ClinicalFact:
    return _fact(
        "fact_bbbbbbbbbbbbbbbb",
        state="inactive",
        parents=("fact_aaaaaaaaaaaaaaaa",),
    )


def _conflict() -> ConflictSet:
    return ConflictSet(
        conflict_id="conflict_aaaaaaaaaaaaaaaa",
        subject_id=SUBJECT,
        conflict_type="status",
        fact_ids=("fact_aaaaaaaaaaaaaaaa", "fact_bbbbbbbbbbbbbbbb"),
        status="open",
        detected_by="journey.conflict",
        derivation_hash=canonical_digest({"conflict": "synthetic"}),
        evidence_ids=(_evidence().locator_id,),
    )


def _resolution() -> ResolutionEvent:
    return ResolutionEvent(
        resolution_id="resolution_aaaaaaaaaaaaaaaa",
        conflict_id="conflict_aaaaaaaaaaaaaaaa",
        action="select",
        actor_type="human",
        policy_id="journey.review",
        policy_version="1.0.0",
        occurred_at=T3,
        rationale_code="source_priority",
        derivation_hash=canonical_digest({"resolution": "synthetic"}),
        selected_fact_ids=("fact_aaaaaaaaaaaaaaaa",),
        rejected_fact_ids=("fact_bbbbbbbbbbbbbbbb",),
    )


def _snapshot() -> DatasetSnapshot:
    return DatasetSnapshot(
        snapshot_id="snapshot_aaaaaaaaaaaaaaaa",
        dataset_id="dataset_aaaaaaaaaaaaaaaa",
        created_at=T3,
        query_hash=canonical_digest({"query": "synthetic"}),
        schema_hash=canonical_digest({"schema": "synthetic"}),
        manifest_hash=canonical_digest({"manifest": "synthetic"}),
        record_count=2,
        source_artifact_ids=(_artifact().artifact_id,),
        source_fact_ids=("fact_aaaaaaaaaaaaaaaa",),
        file_hashes={"facts/data.jsonl": canonical_digest({"data": "synthetic"})},
        split_hashes={"test": canonical_digest({"split": "synthetic"})},
        component_versions={"journey.store": "1.0.0"},
        license_tags=("synthetic",),
    )


def _canonical(fact_id: str, state: str) -> CanonicalRecord:
    return CanonicalRecord(
        canonical_id="canonical_aaaaaaaaaaaaaaaa",
        subject_id=SUBJECT,
        fact_id=fact_id,
        record_type="condition",
        state=state,
        effective_at=T1,
        reason_code="source_priority",
        metadata={"review": {"required": False}},
    )


def _job() -> JobMetadata:
    return JobMetadata(
        job_id="job_aaaaaaaaaaaaaaaa",
        state="completed",
        recorded_at=T3,
        metadata={"artifact_count": 2, "fact_count": 2},
    )


def _populate(store: LocalJourneyStore) -> None:
    """Write one record into every payload table across three revisions."""

    assert store.ingest_graph(
        _artifact(),
        CONTENT,
        evidence=(_evidence(),),
        facts=(_fact(),),
        committed_at=T1,
    ).ok
    with store.metadata.transaction(committed_at=T2) as transaction:
        assert transaction.put_fact(_second_fact()).ok
        assert transaction.put_conflict(_conflict()).ok
        assert transaction.put_canonical(_canonical(_fact().fact_id, "active")).ok
    second = _artifact(SECOND_CONTENT, "artifact_bbbbbbbbbbbbbbbb")
    assert store.artifacts.put_bytes(second, SECOND_CONTENT).ok
    with store.metadata.transaction(committed_at=T3) as transaction:
        assert transaction.put_artifact(second).ok
        assert transaction.put_resolution(_resolution()).ok
        assert transaction.put_dataset(_snapshot()).ok
        assert transaction.put_canonical(
            _canonical(_second_fact().fact_id, "inactive")
        ).ok
        assert transaction.put_job(_job()).ok


def _reads(store: LocalJourneyStore) -> dict[str, Any]:
    """Return every public read, including point-in-time reads."""

    metadata = store.metadata
    first, second = StorePoint(revision=1), StorePoint(revision=2)
    canonical_id = "canonical_aaaaaaaaaaaaaaaa"
    results = {
        "artifact": metadata.get_artifact(_artifact().artifact_id),
        "artifact_by_hash": metadata.get_artifact_by_hash(_artifact().content_hash),
        "bytes": store.artifacts.get_bytes(_artifact().content_hash),
        "second_bytes": store.artifacts.get_bytes(sha256_digest(SECOND_CONTENT)),
        "evidence": metadata.get_evidence(_evidence().locator_id),
        "fact": metadata.get_fact(_fact().fact_id),
        "facts": metadata.list_facts(SUBJECT),
        "facts_at_1": metadata.list_facts(SUBJECT, as_of=first),
        "conflicts": metadata.list_conflicts(SUBJECT),
        "resolutions": metadata.list_resolutions(_conflict().conflict_id),
        "dataset": metadata.get_dataset(_snapshot().snapshot_id),
        "canonical": metadata.get_canonical(canonical_id),
        "canonical_at_2": metadata.get_canonical(canonical_id, as_of=second),
        "canonical_history": metadata.list_canonical_versions(canonical_id),
        "canonical_records": metadata.list_canonical_records(SUBJECT),
        "job": metadata.get_job(_job().job_id),
        "job_versions": metadata.list_job_versions(_job().job_id),
        "jobs": metadata.list_jobs(),
        "integrity": metadata.integrity_check(),
    }
    failed = {name: result.code for name, result in results.items() if not result.ok}
    assert failed == {}
    values = {name: result.value for name, result in results.items()}
    values["latest_revision"] = metadata.latest_revision
    return values


def _files_containing(root: Path, needle: bytes) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and needle in path.read_bytes()
    )


def _raw_rows(path: Path) -> dict[str, list[tuple[Any, ...]]]:
    connection = sqlite3.connect(path)
    try:
        return {
            table: connection.execute(
                f"SELECT payload_hash, payload_json FROM {table} ORDER BY rowid"  # noqa: S608
            ).fetchall()
            for table in PAYLOAD_RECORD_KEYS
        }
    finally:
        connection.close()


def _table_key_ids(path: Path) -> dict[str, list[str | None]]:
    """Return each row's sealing key id per table; ``None`` means plaintext."""

    return {
        table: [
            None
            if stored.startswith("{")
            else read_envelope_header(envelope_from_text(stored)).key_id
            for _payload_hash, stored in rows
        ]
        for table, rows in _raw_rows(path).items()
    }


def _row_key_ids(path: Path) -> list[str | None]:
    return [key_id for key_ids in _table_key_ids(path).values() for key_id in key_ids]


def _blob_files(root: Path) -> dict[str, bytes]:
    blob_root = root / "artifacts" / "blobs" / "sha256"
    return {
        path.name: path.read_bytes()
        for path in sorted(blob_root.rglob("*"))
        if path.is_file()
    }


class _FlakyProvider:
    """Delegates to a provider, failing one lookup to simulate a crash."""

    def __init__(
        self,
        inner: InMemoryKeyProvider,
        *,
        fail_resolve_of: str | None = None,
        fail_active_after: int | None = None,
        allowed_resolves: int = 0,
    ) -> None:
        self.inner = inner
        self.fail_resolve_of = fail_resolve_of
        self.fail_active_after = fail_active_after
        self.allowed_resolves = allowed_resolves
        self.active_calls = 0

    def active_key(self, purpose: str) -> DataKey:
        self.active_calls += 1
        if self.fail_active_after is not None and (
            self.active_calls > self.fail_active_after
        ):
            raise KeyUnavailableError()
        return self.inner.active_key(purpose)

    def resolve_key(self, key_id: str, *, purpose: str) -> DataKey:
        if key_id == self.fail_resolve_of:
            if self.allowed_resolves <= 0:
                raise KeyUnavailableError()
            self.allowed_resolves -= 1
        return self.inner.resolve_key(key_id, purpose=purpose)


def test_stores_refuse_to_open_without_an_explicit_encryption_choice(
    tmp_path: Path,
) -> None:
    factories = (
        lambda: SQLiteJourneyStore(tmp_path / "metadata.sqlite3"),
        lambda: LocalArtifactStore(tmp_path / "artifacts"),
        lambda: LocalJourneyStore(tmp_path / "journey"),
        lambda: SQLiteIngestionStore(tmp_path / "ingestion.sqlite3"),
    )

    for factory in factories:
        with pytest.raises(JourneyEncryptionError) as error:
            factory()
        assert error.value.code == "encryption_required"
    opened = SQLiteJourneyStore.open(tmp_path / "metadata.sqlite3")

    assert opened.state is StoreState.FAILURE
    assert opened.code == "encryption_required"
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(TypeError):
        SQLiteJourneyStore(tmp_path / "x.sqlite3", encryption="plaintext")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        JourneyStoreEncryption()


def test_production_profile_refuses_plaintext_and_code_cannot_weaken_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "prod")

    for encryption in (
        JourneyStoreEncryption.development_plaintext(),
        JourneyStoreEncryption.development_plaintext(profile="dev"),
    ):
        with pytest.raises(JourneyEncryptionError) as error:
            LocalJourneyStore(tmp_path / "plaintext", encryption=encryption)
        assert error.value.code == "plaintext_refused_in_production"
    assert resolve_journey_profile("dev") == "prod"
    with LocalJourneyStore(
        tmp_path / "encrypted", encryption=_encrypted(_provider())
    ) as store:
        assert store.encryption_state == "encrypted"
        assert store.artifacts.encryption_state == "encrypted"

    monkeypatch.delenv("OPENMED_PROFILE")
    assert resolve_journey_profile() == "dev"
    with pytest.raises(JourneyEncryptionError) as asserted:
        SQLiteJourneyStore(
            tmp_path / "asserted.sqlite3",
            encryption=JourneyStoreEncryption.development_plaintext(
                profile="production"
            ),
        )
    assert asserted.value.code == "plaintext_refused_in_production"


def test_sentinel_phi_is_absent_from_database_wal_temp_and_blob_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_temp_files_with_sentinel: list[bool] = []
    real_link = os.link

    def inspecting_link(source: Any, target: Any, *args: Any, **kwargs: Any) -> None:
        linked_temp_files_with_sentinel.append(
            SENTINEL_BYTES in Path(source).read_bytes()
        )
        real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", inspecting_link)
    root = tmp_path / "encrypted"
    store = LocalJourneyStore(root, encryption=_encrypted(_provider()))
    _populate(store)

    reads = _reads(store)
    assert reads["bytes"] == CONTENT
    assert reads["fact"] == _fact()
    wal = root / "journey.sqlite3-wal"
    assert wal.is_file() and wal.stat().st_size > 0
    assert _files_containing(root, SENTINEL_BYTES) == []
    assert linked_temp_files_with_sentinel == [False, False]
    assert all(key_id == "key-one" for key_id in _row_key_ids(root / "journey.sqlite3"))
    store.close()
    assert _files_containing(root, SENTINEL_BYTES) == []

    # Positive control: the same scan finds the sentinel in plaintext storage.
    control = tmp_path / "plaintext"
    with LocalJourneyStore(control, encryption=PLAINTEXT_DEV) as plaintext:
        _populate(plaintext)
        found = _files_containing(control, SENTINEL_BYTES)
    assert any(name.startswith("artifacts/blobs/") for name in found)
    assert any(name.startswith("journey.sqlite3") for name in found)


def test_ciphertext_is_bound_to_its_record_identity(tmp_path: Path) -> None:
    root = tmp_path / "journey"
    store = LocalJourneyStore(root, encryption=_encrypted(_provider()))
    _populate(store)
    connection = sqlite3.connect(root / "journey.sqlite3")
    first, second = (
        row[0]
        for row in connection.execute(
            "SELECT payload_json FROM clinical_facts ORDER BY fact_id"
        )
    )
    connection.execute(
        "UPDATE clinical_facts SET payload_json = ? WHERE fact_id = ?",
        (second, _fact().fact_id),
    )
    connection.commit()
    connection.close()

    moved = store.metadata.get_fact(_fact().fact_id)
    integrity = store.metadata.integrity_check()

    assert moved.state is StoreState.FAILURE
    assert moved.code == "envelope_authentication_failed"
    assert integrity.code == "envelope_authentication_failed"
    blobs = root / "artifacts" / "blobs" / "sha256"
    first_blob = blobs / _artifact().content_hash[7:9] / _artifact().content_hash[7:]
    second_hash = sha256_digest(SECOND_CONTENT)
    second_blob = blobs / second_hash[7:9] / second_hash[7:]
    first_blob.write_bytes(second_blob.read_bytes())
    swapped = store.artifacts.get_bytes(_artifact().content_hash)
    assert swapped.code == "envelope_authentication_failed"
    store.close()


def test_wrong_unavailable_revoked_and_retired_keys_fail_closed_without_damage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journey"
    with LocalJourneyStore(root, encryption=_encrypted(_provider())) as store:
        _populate(store)
        expected = _reads(store)
    rows_before = _raw_rows(root / "journey.sqlite3")
    blobs_before = _blob_files(root)

    wrong = _encrypted(_provider(KEY_TWO))
    for factory in (
        lambda: SQLiteJourneyStore(root / "journey.sqlite3", encryption=wrong),
        lambda: LocalArtifactStore(root / "artifacts", encryption=wrong),
    ):
        with pytest.raises(JourneyEncryptionError) as error:
            factory()
        assert error.value.code == "key_check_failed"
        assert "key-one" not in str(error.value)
    assert SQLiteJourneyStore.open(root / "journey.sqlite3", encryption=wrong).code == (
        "key_check_failed"
    )

    unavailable = LocalJourneyStore(root, encryption=_encrypted(InMemoryKeyProvider()))
    assert unavailable.metadata.get_fact(_fact().fact_id).code == "key_unavailable"
    assert unavailable.artifacts.get_bytes(_artifact().content_hash).code == (
        "key_unavailable"
    )
    late = JobMetadata(job_id="job_bbbbbbbbbbbbbbbb", state="queued", recorded_at=T4)
    assert unavailable.metadata.put_job(late, committed_at=T4).code == (
        "key_unavailable"
    )
    third = b"synthetic third note " + SENTINEL_BYTES
    third_artifact = _artifact(third, "artifact_cccccccccccccccc")
    assert unavailable.artifacts.put_bytes(third_artifact, third).code == (
        "key_unavailable"
    )
    assert unavailable.metadata.latest_revision == expected["latest_revision"]
    unavailable.close()

    revoked_provider = InMemoryKeyProvider()
    revoked_provider.add_key(
        "key-one", KEY_ONE, purpose=JOURNEY_KEY_PURPOSE, state=KeyState.REVOKED
    )
    with LocalJourneyStore(root, encryption=_encrypted(revoked_provider)) as revoked:
        assert revoked.metadata.get_fact(_fact().fact_id).code == "key_revoked"
        assert revoked.metadata.integrity_check().code == "key_revoked"
        assert revoked.artifacts.get_bytes(_artifact().content_hash).code == (
            "key_revoked"
        )

    retired_provider = _provider()
    retired_provider.rotate("key-two", KEY_TWO, purpose=JOURNEY_KEY_PURPOSE)
    retired_provider.set_state("key-one", KeyState.RETIRED, purpose=JOURNEY_KEY_PURPOSE)
    with LocalJourneyStore(root, encryption=_encrypted(retired_provider)) as retired:
        assert retired.metadata.list_facts(SUBJECT).code == "key_retired"
        assert retired.reencrypt().code == "key_retired"
        retired_provider.set_state(
            "key-one", KeyState.DECRYPT_ONLY, purpose=JOURNEY_KEY_PURPOSE
        )
        assert retired.metadata.get_fact(_fact().fact_id).value == _fact()

    assert _raw_rows(root / "journey.sqlite3") == rows_before
    assert _blob_files(root) == blobs_before
    with LocalJourneyStore(root, encryption=_encrypted(_provider())) as restored:
        assert _reads(restored) == expected


def test_rotation_preserves_hashes_history_and_reads_across_a_crash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journey"
    database = root / "journey.sqlite3"
    provider = _provider()
    store = LocalJourneyStore(root, encryption=_encrypted(provider))
    _populate(store)
    expected = _reads(store)
    hashes_before = {
        table: [payload_hash for payload_hash, _stored in rows]
        for table, rows in _raw_rows(database).items()
    }
    provider.rotate("key-two", KEY_TWO, purpose=JOURNEY_KEY_PURPOSE)

    partial = store.metadata.reencrypt(batch_size=1, max_batches=1)
    assert partial.ok and partial.value is not None
    assert (partial.value.reencrypted, partial.value.complete) == (1, False)
    store.close()
    key_ids_after_partial = _row_key_ids(database)
    assert key_ids_after_partial.count("key-two") == 1

    # The key lookup for the second fact fails after the first fact was
    # re-sealed in the same batch: that whole batch rolls back, while the
    # batches committed before it (artifacts, evidence) are kept.
    flaky = _FlakyProvider(provider, fail_resolve_of="key-one", allowed_resolves=3)
    with SQLiteJourneyStore(database, encryption=_encrypted(flaky)) as crashed:
        interrupted = crashed.reencrypt(batch_size=8)
    assert interrupted.state is StoreState.FAILURE
    assert interrupted.code == "key_unavailable"
    assert interrupted.value is not None and not interrupted.value.complete
    assert interrupted.value.reencrypted == 2
    after_crash = _table_key_ids(database)
    assert after_crash["artifacts"] == ["key-two", "key-two"]
    assert after_crash["evidence_locators"] == ["key-two"]
    assert after_crash["clinical_facts"] == ["key-one", "key-one"]
    key_ids_after_crash = _row_key_ids(database)

    restarted = LocalJourneyStore(root, encryption=_encrypted(provider))
    assert _reads(restarted) == expected
    resumed = restarted.reencrypt(batch_size=2)
    assert resumed.ok and resumed.value is not None and resumed.value.complete
    assert resumed.value.metadata.reencrypted == key_ids_after_crash.count("key-one")
    assert resumed.value.artifacts.reencrypted == 2
    provider.set_state("key-one", KeyState.RETIRED, purpose=JOURNEY_KEY_PURPOSE)

    assert _reads(restarted) == expected
    assert set(_row_key_ids(database)) == {"key-two"}
    assert {
        read_envelope_header(stored).key_id for stored in _blob_files(root).values()
    } == {"key-two"}
    assert {
        table: [payload_hash for payload_hash, _stored in rows]
        for table, rows in _raw_rows(database).items()
    } == hashes_before
    again = restarted.reencrypt()
    assert again.ok and again.value is not None
    assert again.value.metadata.reencrypted == 0
    restarted.close()
    assert _files_containing(root, SENTINEL_BYTES) == []


def test_plaintext_store_migration_resumes_and_leaves_no_plaintext(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journey"
    database = root / "journey.sqlite3"
    with LocalJourneyStore(root, encryption=PLAINTEXT_DEV) as legacy:
        _populate(legacy)
        expected = _reads(legacy)
    assert _files_containing(root, SENTINEL_BYTES)
    # Simulate a store written before encryption state was recorded, plus a
    # stale temporary blob left behind by an interrupted plaintext write.
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM store_encryption")
    connection.commit()
    connection.close()
    (root / "artifacts" / "encryption.json").unlink()
    blob_prefix = next((root / "artifacts" / "blobs" / "sha256").iterdir())
    stale = blob_prefix / ".journey-blob-stale"
    stale.write_bytes(CONTENT)
    hashes_before = {
        table: [payload_hash for payload_hash, _stored in rows]
        for table, rows in _raw_rows(database).items()
    }
    provider = _provider()
    encryption = _encrypted(provider)

    for factory in (
        lambda: SQLiteJourneyStore(database, encryption=encryption),
        lambda: LocalArtifactStore(root / "artifacts", encryption=encryption),
    ):
        with pytest.raises(JourneyEncryptionError) as refused:
            factory()
        assert refused.value.code == "store_migration_required"

    partial = SQLiteJourneyStore.migrate_to_encrypted(
        database,
        encryption=encryption,
        batch_size=1,
        max_batches=2,
    )
    assert partial.ok and partial.value is not None
    assert (partial.value.migrated, partial.value.complete) == (2, False)
    assert partial.value.state == "migrating"
    with pytest.raises(JourneyEncryptionError) as downgrade:
        SQLiteJourneyStore(database, encryption=PLAINTEXT_DEV)
    assert downgrade.value.code == "store_encrypted"
    artifacts = LocalArtifactStore.migrate_to_encrypted(
        root / "artifacts",
        encryption=encryption,
        max_files=1,
    )
    assert artifacts.ok and artifacts.value is not None
    assert (artifacts.value.migrated, artifacts.value.complete) == (1, False)
    assert not stale.exists()
    with LocalJourneyStore(root, encryption=encryption) as mixed:
        assert mixed.encryption_state == "migrating"
        assert _reads(mixed) == expected

    plaintext_rows = _row_key_ids(database).count(None)
    flaky = _FlakyProvider(provider, fail_active_after=4)
    interrupted = SQLiteJourneyStore.migrate_to_encrypted(
        database,
        encryption=_encrypted(flaky),
        batch_size=64,
    )
    assert interrupted.state is StoreState.FAILURE
    assert interrupted.code == "key_unavailable"
    assert interrupted.value is not None
    # Only committed batches are counted; the batch in flight rolled back.
    remaining = _row_key_ids(database).count(None)
    assert 0 < remaining < plaintext_rows
    assert plaintext_rows - remaining == interrupted.value.migrated
    # Finish the metadata half on an open store and scan its live files.
    with SQLiteJourneyStore(database, encryption=encryption) as finishing:
        finished = finishing.reencrypt(batch_size=3)
        assert finished.ok and finished.value is not None
        assert finished.value.complete and finished.value.migrated == remaining
        assert finishing.encryption_state == "encrypted"
        live = _files_containing(root, SENTINEL_BYTES)
        assert [name for name in live if name.startswith("journey.sqlite3")] == []

    completed = LocalJourneyStore.migrate_to_encrypted(root, encryption=encryption)
    assert completed.ok and completed.value is not None and completed.value.complete
    assert completed.value.metadata.migrated == 0
    assert completed.value.artifacts.migrated == 1
    assert completed.value.metadata.state == "encrypted"
    assert completed.value.artifacts.state == "encrypted"
    assert _files_containing(root, SENTINEL_BYTES) == []
    assert None not in _row_key_ids(database)
    assert {
        table: [payload_hash for payload_hash, _stored in rows]
        for table, rows in _raw_rows(database).items()
    } == hashes_before
    with LocalJourneyStore(root, encryption=encryption) as migrated:
        assert migrated.encryption_state == "encrypted"
        assert _reads(migrated) == expected
    repeated = LocalJourneyStore.migrate_to_encrypted(root, encryption=encryption)
    assert repeated.ok and repeated.value is not None
    assert repeated.value.metadata.migrated == 0
    with pytest.raises(JourneyEncryptionError):
        LocalJourneyStore(root, encryption=PLAINTEXT_DEV)


def test_planted_plaintext_and_planted_envelopes_fail_closed(tmp_path: Path) -> None:
    encrypted_root = tmp_path / "encrypted"
    store = LocalJourneyStore(encrypted_root, encryption=_encrypted(_provider()))
    _populate(store)
    connection = sqlite3.connect(encrypted_root / "journey.sqlite3")
    envelope = connection.execute(
        "SELECT payload_json FROM clinical_facts WHERE fact_id = ?",
        (_fact().fact_id,),
    ).fetchone()[0]
    connection.execute(
        "UPDATE clinical_facts SET payload_json = ? WHERE fact_id = ?",
        (_fact().to_json(), _fact().fact_id),
    )
    connection.commit()
    connection.close()

    assert store.metadata.get_fact(_fact().fact_id).code == "payload_not_encrypted"
    assert store.metadata.integrity_check().code == "payload_not_encrypted"
    planted = b"synthetic planted note " + SENTINEL_BYTES
    planted_hash = sha256_digest(planted)
    planted_path = (
        encrypted_root
        / "artifacts"
        / "blobs"
        / "sha256"
        / planted_hash[7:9]
        / planted_hash[7:]
    )
    planted_path.parent.mkdir(parents=True, exist_ok=True)
    planted_path.write_bytes(planted)
    assert store.artifacts.get_bytes(planted_hash).code == "payload_not_encrypted"
    store.close()

    plaintext_root = tmp_path / "plaintext"
    with LocalJourneyStore(plaintext_root, encryption=PLAINTEXT_DEV) as plaintext:
        _populate(plaintext)
        connection = sqlite3.connect(plaintext_root / "journey.sqlite3")
        connection.execute(
            "UPDATE clinical_facts SET payload_json = ? WHERE fact_id = ?",
            (envelope, _fact().fact_id),
        )
        connection.commit()
        connection.close()
        assert plaintext.metadata.get_fact(_fact().fact_id).code == (
            "payload_encrypted_in_plaintext_store"
        )


def test_encryption_state_and_maintenance_outcomes_are_typed(tmp_path: Path) -> None:
    provider = _provider()
    provider.add_key(
        "archive-key",
        KEY_TWO,
        purpose="journey-archive",
        state=KeyState.ACTIVE,
    )
    root = tmp_path / "journey"
    with LocalJourneyStore(root, encryption=_encrypted(provider)) as store:
        _populate(store)
    with pytest.raises(JourneyEncryptionError) as mismatch:
        SQLiteJourneyStore(
            root / "journey.sqlite3",
            encryption=JourneyStoreEncryption.encrypted(
                provider, key_purpose="journey-archive"
            ),
        )
    assert mismatch.value.code == "key_purpose_mismatch"

    marker = root / "artifacts" / "encryption.json"
    marker.write_text("{not json", encoding="utf-8")
    with pytest.raises(JourneyEncryptionError) as invalid:
        LocalArtifactStore(root / "artifacts", encryption=_encrypted(provider))
    assert invalid.value.code == "encryption_state_invalid"

    with SQLiteJourneyStore(
        tmp_path / "plaintext.sqlite3", encryption=PLAINTEXT_DEV
    ) as plaintext:
        assert plaintext.reencrypt().code == "store_not_encrypted"
        with pytest.raises(ValueError):
            plaintext.reencrypt(batch_size=0)
    with SQLiteJourneyStore(
        tmp_path / "denied.sqlite3",
        encryption=_encrypted(provider),
        policy=DenyStorageOperations(frozenset({"reencrypt"})),
    ) as denied:
        assert denied.reencrypt().state is StoreState.DENIED
    unsupported = PostgresJourneyStore.migrate_to_encrypted(
        object(), encryption=_encrypted(provider)
    )
    assert unsupported.state is StoreState.UNSUPPORTED
    assert unsupported.code == "encryption_unsupported"


def test_ingestion_store_seals_journey_payloads_and_reprs_are_value_free(
    tmp_path: Path,
) -> None:
    provider = _provider()
    encryption = _encrypted(provider)
    path = tmp_path / "ingestion.sqlite3"
    with SQLiteIngestionStore(path, encryption=encryption) as store:
        assert store.put_artifact(_artifact(), committed_at=T1).ok
        assert store.get_artifact(_artifact().artifact_id).value == _artifact()
    assert all(key_id == "key-one" for key_id in _row_key_ids(path))

    rendered = " ".join(
        (
            repr(encryption),
            repr(provider),
            repr(provider.active_key(JOURNEY_KEY_PURPOSE)),
            str(JourneyEncryptionError("key_check_failed")),
        )
    )
    assert KEY_ONE.hex() not in rendered
    assert repr(KEY_ONE) not in rendered
    assert "InMemoryKeyProvider(keys=1)" in rendered
    assert SENTINEL not in rendered
