"""Tests for bounded fsspec Journey artifact storage."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    EvidenceLocator,
    sha256_digest,
)
from openmed.structured.store import (
    ArtifactStore,
    CommitStatusUnknown,
    ComposedJourneyStore,
    DenyStorageOperations,
    FsspecArtifactStore,
    ObjectStoreNamespace,
    SQLiteJourneyStore,
    StoreState,
)

CONTENT = b"synthetic object journey artifact"
RECORDED_AT = "2026-01-02T03:04:05Z"


def _artifact(content: bytes = CONTENT) -> ClinicalArtifact:
    return ClinicalArtifact(
        artifact_id="artifact_cccccccccccccccc",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest(content),
        byte_size=len(content),
        source_id="source_cccccccccccccccc",
        recorded_at=RECORDED_AT,
        subject_id="subject_cccccccccccccccc",
        encounter_id="encounter_cccccccccccccccc",
    )


def test_local_namespace_is_default_private_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    store = FsspecArtifactStore(root)

    first = store.put_bytes(_artifact(), CONTENT)
    second = store.put_bytes(_artifact(), CONTENT)

    assert isinstance(store, ArtifactStore)
    assert store.protocol == "file"
    assert first.ok and first.created
    assert second.ok and not second.created
    assert store.get_bytes(_artifact().content_hash).value == CONTENT
    digest = _artifact().content_hash.removeprefix("sha256:")
    blob = root / "blobs" / "sha256" / digest[:2] / digest
    assert blob.is_file()
    if os.name == "posix":
        assert os.stat(blob).st_mode & 0o077 == 0
        assert os.stat(blob.parent).st_mode & 0o077 == 0


def test_memory_namespace_requires_explicit_allowlist() -> None:
    with pytest.raises(ValueError, match="not explicitly allowed"):
        FsspecArtifactStore("memory://journey/default-denied")

    namespace = ObjectStoreNamespace(
        "memory://journey/explicit",
        allowed_protocols=frozenset({"memory"}),
    )
    store = FsspecArtifactStore(namespace)
    result = store.put_bytes(_artifact(), CONTENT)

    assert result.ok and result.created
    assert store.protocol == "memory"
    assert store.get_bytes(_artifact().content_hash).value == CONTENT


@pytest.mark.parametrize(
    "root_uri",
    (
        "https://example.invalid/journey",
        "s3://synthetic-bucket/journey",
        "memory://journey/root",
    ),
)
def test_network_and_nonlocal_protocols_are_denied_by_default(root_uri: str) -> None:
    with pytest.raises(ValueError, match="not explicitly allowed"):
        ObjectStoreNamespace(root_uri)


def test_unbounded_local_root_is_rejected() -> None:
    with pytest.raises(ValueError, match="bounded"):
        FsspecArtifactStore("file:///")


@pytest.mark.parametrize(
    "root_uri",
    (
        "memory://journey/../escape",
        "file:///tmp/journey/../../escape",
        "s3://synthetic-bucket/root?token=not-stored",
        "s3://user:credential@synthetic-bucket/root",
    ),
)
def test_unsafe_namespace_forms_are_rejected(root_uri: str) -> None:
    protocol = root_uri.split(":", 1)[0]
    with pytest.raises(ValueError):
        ObjectStoreNamespace(
            root_uri,
            allowed_protocols=frozenset({protocol}),
        )


def test_storage_options_are_not_exposed_in_repr() -> None:
    canary = "credential-canary-do-not-print"
    namespace = ObjectStoreNamespace(
        "memory://journey/private-options",
        allowed_protocols=frozenset({"memory"}),
        storage_options={"credential": canary},
    )

    assert canary not in repr(namespace)


def test_mismatch_corruption_unknown_and_policy_denial(tmp_path: Path) -> None:
    store = FsspecArtifactStore(tmp_path / "objects")
    wrong = store.put_bytes(_artifact(), b"wrong")
    missing = store.get_bytes(sha256_digest(b"missing"))

    assert wrong.state is StoreState.CONFLICT
    assert wrong.code == "artifact_size_mismatch"
    assert missing.state is StoreState.UNKNOWN

    assert store.put_bytes(_artifact(), CONTENT).ok
    key = store._key_for_digest(_artifact().content_hash)
    assert key is not None
    with store._filesystem.open(key, "wb") as stream:
        stream.write(b"corrupt")
    corrupt = store.get_bytes(_artifact().content_hash)
    assert corrupt.state is StoreState.FAILURE
    assert corrupt.code == "artifact_integrity_failed"

    denied = FsspecArtifactStore(
        tmp_path / "denied",
        policy=DenyStorageOperations(frozenset({"read", "write"})),
    )
    assert denied.put_bytes(_artifact(), CONTENT).state is StoreState.DENIED
    assert denied.get_bytes(_artifact().content_hash).state is StoreState.DENIED


def test_discard_is_bounded_to_the_digest_object(tmp_path: Path) -> None:
    store = FsspecArtifactStore(tmp_path / "objects")
    sentinel = tmp_path / "objects" / "sentinel.txt"
    sentinel.write_text("synthetic", encoding="utf-8")
    assert store.put_bytes(_artifact(), CONTENT).ok

    store.discard_if_created(_artifact().content_hash)

    assert sentinel.read_text(encoding="utf-8") == "synthetic"
    assert store.get_bytes(_artifact().content_hash).state is StoreState.UNKNOWN


def test_composed_store_commits_metadata_and_compensates_failure(
    tmp_path: Path,
) -> None:
    artifacts = FsspecArtifactStore(
        "memory://journey/composed",
        allowed_protocols=frozenset({"memory"}),
    )
    metadata = SQLiteJourneyStore(tmp_path / "journey.sqlite3")
    store = ComposedJourneyStore(artifacts, metadata)
    locator = EvidenceLocator(
        locator_id="evidence_cccccccccccccccc",
        artifact_id=_artifact().artifact_id,
        location_type="text_span",
        location={"start": 0, "end": 9},
    )

    committed = store.ingest_graph(
        _artifact(),
        CONTENT,
        evidence=(locator,),
        committed_at=RECORDED_AT,
    )
    assert committed.ok and committed.created
    assert metadata.get_evidence(locator.locator_id).ok

    failed_content = b"synthetic object that must be compensated"
    failed_artifact = ClinicalArtifact(
        artifact_id="artifact_eeeeeeeeeeeeeeee",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest(failed_content),
        byte_size=len(failed_content),
        source_id="source_eeeeeeeeeeeeeeee",
        recorded_at=RECORDED_AT,
        subject_id="subject_eeeeeeeeeeeeeeee",
        encounter_id="encounter_eeeeeeeeeeeeeeee",
    )
    invalid_locator = EvidenceLocator(
        locator_id="evidence_eeeeeeeeeeeeeeee",
        artifact_id="artifact_ffffffffffffffff",
        location_type="text_span",
        location={"start": 0, "end": 9},
    )
    failed = store.ingest_graph(
        failed_artifact,
        failed_content,
        evidence=(invalid_locator,),
        committed_at=RECORDED_AT,
    )

    assert failed.state is StoreState.PARTIAL
    assert (
        metadata.get_artifact(failed_artifact.artifact_id).state is StoreState.UNKNOWN
    )
    assert artifacts.get_bytes(failed_artifact.content_hash).state is StoreState.UNKNOWN
    store.close()


def test_local_object_store_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    outside = tmp_path / "outside"
    outside.mkdir()
    store = FsspecArtifactStore(root)
    digest = _artifact().content_hash.removeprefix("sha256:")
    prefix = root / "blobs" / "sha256" / digest[:2]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.symlink_to(outside, target_is_directory=True)
    (outside / digest).write_bytes(CONTENT)

    result = store.get_bytes(_artifact().content_hash)

    assert result.state is StoreState.FAILURE
    assert result.code == "artifact_path_unsafe"


def test_composed_store_keeps_blob_when_commit_status_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = FsspecArtifactStore(
        "memory://journey/unknown-commit",
        allowed_protocols=frozenset({"memory"}),
    )
    metadata = SQLiteJourneyStore(tmp_path / "journey.sqlite3")
    store = ComposedJourneyStore(artifacts, metadata)
    original_transaction = metadata.transaction

    @contextmanager
    def unknown_transaction(*, committed_at: str) -> Any:
        with original_transaction(committed_at=committed_at) as transaction:
            yield transaction
        raise CommitStatusUnknown("synthetic unknown commit")

    monkeypatch.setattr(metadata, "transaction", unknown_transaction)

    result = store.ingest_graph(
        _artifact(),
        CONTENT,
        committed_at=RECORDED_AT,
    )

    assert result.state is StoreState.UNKNOWN
    assert result.code == "commit_status_unknown"
    assert artifacts.get_bytes(_artifact().content_hash).ok
    store.close()
