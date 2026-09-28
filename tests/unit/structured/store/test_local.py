"""Tests for durable local Journey storage and point-in-time history."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
from openmed.structured.store import (
    MIGRATIONS,
    ArtifactStore,
    CanonicalRecord,
    CanonicalStore,
    DatasetStore,
    DenyStorageOperations,
    EvidenceStore,
    FactStore,
    JobMetadata,
    JobMetadataStore,
    LocalArtifactStore,
    LocalJourneyStore,
    PointInTimeReader,
    ResolutionStore,
    SQLiteJourneyStore,
    StorePoint,
    StoreResult,
    StoreState,
    TransactionalJourneyStore,
)

CONTENT = b"synthetic local journey artifact"
COMMITTED_1 = "2026-01-02T03:04:05Z"
COMMITTED_2 = "2026-01-02T04:04:05Z"
COMMITTED_3 = "2026-01-02T05:04:05Z"


def _artifact() -> ClinicalArtifact:
    return ClinicalArtifact(
        artifact_id="artifact_aaaaaaaaaaaaaaaa",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest(CONTENT),
        byte_size=len(CONTENT),
        source_id="source_aaaaaaaaaaaaaaaa",
        recorded_at=COMMITTED_1,
        subject_id="subject_aaaaaaaaaaaaaaaa",
        encounter_id="encounter_aaaaaaaaaaaaaaaa",
    )


def _evidence() -> EvidenceLocator:
    return EvidenceLocator(
        locator_id="evidence_aaaaaaaaaaaaaaaa",
        artifact_id=_artifact().artifact_id,
        location_type="text_span",
        location={"start": 0, "end": 9},
    )


def _fact(
    *,
    fact_id: str = "fact_aaaaaaaaaaaaaaaa",
    state: str = "active",
    parent_fact_ids: tuple[str, ...] = (),
) -> ClinicalFact:
    return ClinicalFact(
        fact_id=fact_id,
        subject_id="subject_aaaaaaaaaaaaaaaa",
        fact_type="condition",
        value={"code": "synthetic-condition", "state": state},
        status=state,
        evidence_ids=(_evidence().locator_id,),
        derivation_hash=canonical_digest(
            {"fact_id": fact_id, "state": state, "synthetic": True}
        ),
        encounter_id="encounter_aaaaaaaaaaaaaaaa",
        parent_fact_ids=parent_fact_ids,
    )


def _second_fact() -> ClinicalFact:
    return _fact(
        fact_id="fact_bbbbbbbbbbbbbbbb",
        state="inactive",
        parent_fact_ids=("fact_aaaaaaaaaaaaaaaa",),
    )


def _conflict() -> ConflictSet:
    return ConflictSet(
        conflict_id="conflict_aaaaaaaaaaaaaaaa",
        subject_id="subject_aaaaaaaaaaaaaaaa",
        conflict_type="status",
        fact_ids=("fact_aaaaaaaaaaaaaaaa", "fact_bbbbbbbbbbbbbbbb"),
        status="open",
        detected_by="journey.conflict",
        derivation_hash=canonical_digest({"conflict": "synthetic-status"}),
        evidence_ids=("evidence_aaaaaaaaaaaaaaaa",),
    )


def _resolution() -> ResolutionEvent:
    return ResolutionEvent(
        resolution_id="resolution_aaaaaaaaaaaaaaaa",
        conflict_id="conflict_aaaaaaaaaaaaaaaa",
        action="select",
        actor_type="human",
        policy_id="journey.review",
        policy_version="1.0.0",
        occurred_at=COMMITTED_2,
        rationale_code="source_priority",
        derivation_hash=canonical_digest({"resolution": "synthetic"}),
        selected_fact_ids=("fact_aaaaaaaaaaaaaaaa",),
        rejected_fact_ids=("fact_bbbbbbbbbbbbbbbb",),
    )


def _snapshot() -> DatasetSnapshot:
    return DatasetSnapshot(
        snapshot_id="snapshot_aaaaaaaaaaaaaaaa",
        dataset_id="dataset_aaaaaaaaaaaaaaaa",
        created_at=COMMITTED_2,
        query_hash=canonical_digest({"query": "synthetic"}),
        schema_hash=canonical_digest({"schema": "synthetic"}),
        manifest_hash=canonical_digest({"manifest": "synthetic"}),
        record_count=2,
        source_artifact_ids=("artifact_aaaaaaaaaaaaaaaa",),
        source_fact_ids=("fact_aaaaaaaaaaaaaaaa",),
        file_hashes={"facts/data.jsonl": canonical_digest({"data": "synthetic"})},
        split_hashes={"test": canonical_digest({"split": "synthetic"})},
        component_versions={"journey.store": "1.0.0"},
        license_tags=("synthetic",),
    )


def _canonical(fact_id: str, state: str) -> CanonicalRecord:
    return CanonicalRecord(
        canonical_id="canonical_aaaaaaaaaaaaaaaa",
        subject_id="subject_aaaaaaaaaaaaaaaa",
        fact_id=fact_id,
        record_type="condition",
        state=state,
        effective_at=COMMITTED_1,
        reason_code="source_priority",
        metadata={"review": {"required": False}},
    )


def test_artifact_store_is_content_addressed_idempotent_and_private(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = store.put_bytes(_artifact(), CONTENT)
    second = store.put_bytes(_artifact(), CONTENT)

    assert first.ok and first.created
    assert second.ok and not second.created
    assert store.get_bytes(_artifact().content_hash).value == CONTENT

    digest = _artifact().content_hash.removeprefix("sha256:")
    blob = tmp_path / "artifacts" / "blobs" / "sha256" / digest[:2] / digest
    assert blob.is_file()
    if os.name == "posix":
        assert os.stat(blob).st_mode & 0o077 == 0


def test_artifact_store_rejects_mismatch_and_detects_restart_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    store = LocalArtifactStore(root)
    mismatch = store.put_bytes(_artifact(), b"different")

    assert mismatch.state is StoreState.CONFLICT
    assert mismatch.code == "artifact_size_mismatch"

    assert store.put_bytes(_artifact(), CONTENT).ok
    digest = _artifact().content_hash.removeprefix("sha256:")
    blob = root / "blobs" / "sha256" / digest[:2] / digest
    blob.write_bytes(b"corrupted")

    reopened = LocalArtifactStore(root)
    result = reopened.get_bytes(_artifact().content_hash)
    assert result.state is StoreState.FAILURE
    assert result.code == "artifact_integrity_failed"
    assert "corrupted" not in repr(result)


def test_migrations_are_deterministic_and_restart_safe(tmp_path: Path) -> None:
    path = tmp_path / "journey.sqlite3"
    first = SQLiteJourneyStore(path)
    rows_before = first._connection.execute(  # noqa: SLF001
        "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    first.close()

    second = SQLiteJourneyStore(path)
    rows_after = second._connection.execute(  # noqa: SLF001
        "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()

    assert second.schema_version == MIGRATIONS[-1].version
    assert [tuple(row) for row in rows_after] == [tuple(row) for row in rows_before]
    assert rows_after[0][2] == MIGRATIONS[0].checksum
    second.close()


def test_public_store_protocols_match_local_implementations(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    metadata = SQLiteJourneyStore(tmp_path / "journey.sqlite3")

    assert isinstance(artifacts, ArtifactStore)
    assert isinstance(metadata, EvidenceStore)
    assert isinstance(metadata, FactStore)
    assert isinstance(metadata, CanonicalStore)
    assert isinstance(metadata, ResolutionStore)
    assert isinstance(metadata, DatasetStore)
    assert isinstance(metadata, JobMetadataStore)
    assert isinstance(metadata, PointInTimeReader)
    assert isinstance(metadata, TransactionalJourneyStore)
    metadata.close()


def test_newer_or_drifted_migration_is_typed_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "journey.sqlite3"
    store = SQLiteJourneyStore(path)
    store.close()
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO schema_migrations(version, name, checksum) VALUES (999, ?, ?)",
        ("future", "f" * 64),
    )
    connection.commit()
    connection.close()

    result = SQLiteJourneyStore.open(path)

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "schema_unsupported"
    assert result.value is None


def test_graph_ingest_is_atomic_idempotent_and_restart_safe(tmp_path: Path) -> None:
    root = tmp_path / "journey"
    store = LocalJourneyStore(root)
    first = store.ingest_graph(
        _artifact(),
        CONTENT,
        evidence=(_evidence(),),
        facts=(_fact(),),
        committed_at=COMMITTED_1,
    )
    second = store.ingest_graph(
        _artifact(),
        CONTENT,
        evidence=(_evidence(),),
        facts=(_fact(),),
        committed_at=COMMITTED_2,
    )

    assert first.ok and first.created and first.revision == 1
    assert second.ok and not second.created and second.revision == 1
    assert store.metadata.latest_revision == 1
    counts = store.metadata.integrity_check()
    assert counts.ok
    assert counts.value["artifacts"] == 1
    assert counts.value["evidence_locators"] == 1
    assert counts.value["clinical_facts"] == 1
    store.close()

    reopened = LocalJourneyStore(root)
    assert reopened.artifacts.get_bytes(_artifact().content_hash).value == CONTENT
    assert reopened.metadata.get_fact(_fact().fact_id).value == _fact()
    assert reopened.metadata.latest_revision == 1
    reopened.close()


def test_failed_graph_transaction_leaves_no_partial_graph_or_blob(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "journey")
    result = store.ingest_graph(
        _artifact(),
        CONTENT,
        facts=(_fact(),),
        committed_at=COMMITTED_1,
    )

    assert result.state is StoreState.PARTIAL
    assert result.code == "missing_evidence"
    assert store.metadata.latest_revision is None
    assert (
        store.metadata.get_artifact(_artifact().artifact_id).state is StoreState.UNKNOWN
    )
    assert store.metadata.get_fact(_fact().fact_id).state is StoreState.UNKNOWN
    assert (
        store.artifacts.get_bytes(_artifact().content_hash).state is StoreState.UNKNOWN
    )
    store.close()


def test_exception_rolls_back_transaction(tmp_path: Path) -> None:
    store = SQLiteJourneyStore(tmp_path / "journey.sqlite3")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        with store.transaction(committed_at=COMMITTED_1) as transaction:
            assert transaction.put_artifact(_artifact()).ok
            raise RuntimeError("synthetic interruption")

    assert store.latest_revision is None
    assert store.get_artifact(_artifact().artifact_id).state is StoreState.UNKNOWN
    store.close()


def test_point_in_time_reads_reproduce_state_before_correction(tmp_path: Path) -> None:
    store = LocalJourneyStore(tmp_path / "journey")
    ingested = store.ingest_graph(
        _artifact(),
        CONTENT,
        evidence=(_evidence(),),
        facts=(_fact(),),
        committed_at=COMMITTED_1,
    )
    assert ingested.revision == 1

    first_pointer = store.metadata.put_canonical(
        _canonical(_fact().fact_id, "active"),
        committed_at=COMMITTED_2,
    )
    assert first_pointer.revision == 2

    with store.metadata.transaction(committed_at=COMMITTED_3) as transaction:
        corrected = transaction.put_fact(_second_fact())
        latest_pointer = transaction.put_canonical(
            _canonical(_second_fact().fact_id, "inactive")
        )
    assert corrected.ok and latest_pointer.ok

    earlier = store.metadata.get_canonical(
        "canonical_aaaaaaaaaaaaaaaa",
        as_of=StorePoint(2),
    )
    latest = store.metadata.get_canonical("canonical_aaaaaaaaaaaaaaaa")
    earlier_facts = store.metadata.list_facts(
        "subject_aaaaaaaaaaaaaaaa",
        as_of=StorePoint(2),
    )
    history = store.metadata.list_canonical_versions("canonical_aaaaaaaaaaaaaaaa")

    assert earlier.value.record.fact_id == _fact().fact_id
    assert earlier.value.version == 1
    assert latest.value.record.fact_id == _second_fact().fact_id
    assert latest.value.version == 2
    assert earlier_facts.value == (_fact(),)
    assert tuple(item.record.fact_id for item in history.value) == (
        _fact().fact_id,
        _second_fact().fact_id,
    )
    store.close()


def test_conflicting_fact_version_is_typed_and_preserves_original(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "journey")
    assert store.ingest_graph(
        _artifact(),
        CONTENT,
        evidence=(_evidence(),),
        facts=(_fact(),),
        committed_at=COMMITTED_1,
    ).ok
    changed = replace(
        _fact(),
        value={"code": "synthetic-condition", "state": "changed"},
    )

    result = store.metadata.put_fact(changed, committed_at=COMMITTED_2)

    assert result.state is StoreState.CONFLICT
    assert result.code == "record_conflict"
    assert store.metadata.get_fact(_fact().fact_id).value == _fact()
    assert store.metadata.latest_revision == 1
    store.close()


def test_conflict_resolution_dataset_and_job_metadata_are_append_only(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "journey")
    graph = store.ingest_graph(
        _artifact(),
        CONTENT,
        evidence=(_evidence(),),
        facts=(_fact(), _second_fact()),
        conflicts=(_conflict(),),
        committed_at=COMMITTED_1,
    )
    assert graph.ok

    with store.metadata.transaction(committed_at=COMMITTED_2) as transaction:
        resolution = transaction.put_resolution(_resolution())
        dataset = transaction.put_dataset(_snapshot())
        job = transaction.put_job(
            JobMetadata(
                job_id="job_aaaaaaaaaaaaaaaa",
                state="completed",
                recorded_at=COMMITTED_2,
                metadata={"artifact_count": 1, "fact_count": 2},
            )
        )

    assert resolution.ok and dataset.ok and job.ok
    assert (
        store.metadata.get_resolution(_resolution().resolution_id).value
        == _resolution()
    )
    assert store.metadata.get_dataset(_snapshot().snapshot_id).value == _snapshot()
    assert store.metadata.get_job("job_aaaaaaaaaaaaaaaa").value.state == "completed"
    store.close()


def test_unknown_denied_partial_and_failure_states_are_not_success(
    tmp_path: Path,
) -> None:
    denied_policy = DenyStorageOperations(frozenset({"write"}))
    denied_blobs = LocalArtifactStore(
        tmp_path / "denied-artifacts", policy=denied_policy
    )
    denied = denied_blobs.put_bytes(_artifact(), CONTENT)
    assert denied.state is StoreState.DENIED and not denied.ok

    write_only = LocalArtifactStore(
        tmp_path / "write-only-artifacts",
        policy=DenyStorageOperations(frozenset({"read"})),
    )
    assert write_only.put_bytes(_artifact(), CONTENT).ok
    assert write_only.get_bytes(_artifact().content_hash).state is StoreState.DENIED

    store = SQLiteJourneyStore(tmp_path / "journey.sqlite3")
    unknown = store.get_fact("fact_aaaaaaaaaaaaaaaa")
    partial = store.put_fact(_fact(), committed_at=COMMITTED_1)
    assert unknown.state is StoreState.UNKNOWN and not unknown.ok
    assert partial.state is StoreState.PARTIAL and not partial.ok
    assert store.latest_revision is None
    store.close()


def test_job_metadata_blocks_sensitive_fields_without_echoing_values() -> None:
    canary = "SYNTHETIC-PHI-CANARY"

    with pytest.raises(ValueError) as exc_info:
        JobMetadata(
            job_id="job_aaaaaaaaaaaaaaaa",
            state="queued",
            recorded_at=COMMITTED_1,
            metadata={"raw_text": canary},
        )

    assert canary not in str(exc_info.value)


@settings(max_examples=25, deadline=None)
@given(
    st.dictionaries(
        keys=st.sampled_from(("attempt", "count", "flags", "review")),
        values=st.one_of(
            st.integers(min_value=0, max_value=100),
            st.booleans(),
            st.lists(st.integers(min_value=0, max_value=10), max_size=4),
        ),
        max_size=4,
    )
)
def test_job_metadata_schema_round_trip_and_canonicalization(
    metadata: dict[str, object],
) -> None:
    first = JobMetadata(
        job_id="job_aaaaaaaaaaaaaaaa",
        state="queued",
        recorded_at=COMMITTED_1,
        metadata=metadata,
    )
    reversed_metadata = dict(reversed(tuple(metadata.items())))
    second = JobMetadata(
        job_id="job_aaaaaaaaaaaaaaaa",
        state="queued",
        recorded_at=COMMITTED_1,
        metadata=reversed_metadata,
    )

    assert first.to_json() == second.to_json()
    assert JobMetadata.from_dict(first.to_dict()) == first
    assert _canonical(_fact().fact_id, "active") == CanonicalRecord.from_dict(
        _canonical(_fact().fact_id, "active").to_dict()
    )


def test_store_result_repr_hides_clinical_values() -> None:
    result = StoreResult.success(_fact())

    assert "synthetic-condition" not in repr(result)
    assert result.ok


def test_payload_integrity_check_fails_closed_without_echoing_content(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "journey")
    assert store.ingest_graph(
        _artifact(),
        CONTENT,
        evidence=(_evidence(),),
        facts=(_fact(),),
        committed_at=COMMITTED_1,
    ).ok
    store.metadata._connection.execute(  # noqa: SLF001
        "UPDATE clinical_facts SET payload_json = ?",
        ('{"value":"SYNTHETIC-PHI-CANARY"}',),
    )

    result = store.metadata.integrity_check()

    assert result.state is StoreState.FAILURE
    assert result.code == "payload_hash_mismatch"
    assert "SYNTHETIC-PHI-CANARY" not in repr(result)
    store.close()


def test_invalid_stored_json_returns_typed_failure(tmp_path: Path) -> None:
    store = LocalJourneyStore(tmp_path / "journey")
    assert store.ingest_graph(
        _artifact(),
        CONTENT,
        evidence=(_evidence(),),
        facts=(_fact(),),
        committed_at=COMMITTED_1,
    ).ok
    store.metadata._connection.execute(  # noqa: SLF001
        "UPDATE clinical_facts SET payload_json = ?",
        ('{"value":"SYNTHETIC-PHI-CANARY"',),
    )

    result = store.metadata.integrity_check()

    assert result.state is StoreState.FAILURE
    assert result.code == "stored_payload_invalid"
    assert "SYNTHETIC-PHI-CANARY" not in repr(result)
    store.close()
