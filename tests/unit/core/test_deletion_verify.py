"""Focused tests for deterministic, verified local artifact deletion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openmed.core import deletion_verify
from openmed.core.deletion_verify import (
    AmbiguousPathError,
    DeletionArtifact,
    DeletionTransactionError,
    FingerprintMismatchError,
    UnsafePathError,
    delete_verified_artifacts,
    fingerprint_file,
)


def _write_artifact(path: Path, content: bytes = b"SYNTHETIC_ARTIFACT") -> str:
    path.write_bytes(content)
    return fingerprint_file(path)


def test_success_is_deterministic_and_writes_counts_only_evidence(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "redacted-source.bin"
    fingerprint = _write_artifact(artifact_path)
    evidence_path = tmp_path / "evidence.json"

    result = delete_verified_artifacts(
        tmp_path,
        {artifact_path: fingerprint},
        evidence_path=evidence_path,
    )

    assert result.passed
    assert result.to_dict() == {
        "schema_version": 1,
        "requested_count": 1,
        "verified_count": 1,
        "deleted_count": 1,
        "rolled_back_count": 0,
        "status": "completed",
    }
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence == result.to_dict()
    serialized = evidence_path.read_text(encoding="utf-8")
    assert str(artifact_path) not in serialized
    assert fingerprint not in serialized
    assert not artifact_path.exists()
    assert list(tmp_path.glob(".openmed-deletion-*")) == []


def test_fingerprint_mismatch_rejects_the_whole_request(tmp_path: Path) -> None:
    first = tmp_path / "source-a.bin"
    second = tmp_path / "source-b.bin"
    first_fingerprint = _write_artifact(first, b"SYNTHETIC_A")
    _write_artifact(second, b"SYNTHETIC_B")
    evidence_path = tmp_path / "rejected.json"

    with pytest.raises(FingerprintMismatchError) as error:
        delete_verified_artifacts(
            tmp_path,
            [
                DeletionArtifact(first, first_fingerprint),
                (second, hashlib.sha256(b"SYNTHETIC_WRONG").hexdigest()),
            ],
            evidence_path=evidence_path,
        )

    assert "SYNTHETIC" not in str(error.value)
    assert first.exists()
    assert second.exists()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "rejected"
    assert evidence["requested_count"] == 2
    assert evidence["verified_count"] == 1
    assert evidence["deleted_count"] == 0
    assert "source-a.bin" not in evidence_path.read_text(encoding="utf-8")


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "synthetic-outside.bin"
    _write_artifact(outside, b"SYNTHETIC_OUTSIDE")
    link = tmp_path / "temporary-map.bin"
    link.symlink_to(outside)

    try:
        with pytest.raises(UnsafePathError):
            delete_verified_artifacts(
                tmp_path,
                {link: hashlib.sha256(b"SYNTHETIC_OUTSIDE").hexdigest()},
            )
        assert link.is_symlink()
        assert outside.exists()
    finally:
        outside.unlink(missing_ok=True)


def test_aliases_are_ambiguous_and_do_not_delete_anything(tmp_path: Path) -> None:
    artifact_path = tmp_path / "mapping.json"
    fingerprint = _write_artifact(artifact_path, b"SYNTHETIC_MAPPING")

    with pytest.raises(AmbiguousPathError):
        delete_verified_artifacts(
            tmp_path,
            [
                (artifact_path, fingerprint),
                (Path("mapping.json"), fingerprint),
            ],
        )

    assert artifact_path.exists()


def test_evidence_path_cannot_overwrite_a_verified_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "mapping.json"
    fingerprint = _write_artifact(artifact_path, b"SYNTHETIC_MAPPING")

    with pytest.raises(AmbiguousPathError):
        delete_verified_artifacts(
            tmp_path,
            {artifact_path: fingerprint},
            evidence_path=artifact_path,
        )

    assert artifact_path.read_bytes() == b"SYNTHETIC_MAPPING"


def test_rejection_evidence_cannot_overwrite_a_requested_artifact(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first_fingerprint = _write_artifact(first, b"SYNTHETIC_FIRST")
    _write_artifact(second, b"SYNTHETIC_SECOND")

    with pytest.raises(AmbiguousPathError):
        delete_verified_artifacts(
            tmp_path,
            [
                (first, first_fingerprint),
                (second, hashlib.sha256(b"SYNTHETIC_WRONG").hexdigest()),
            ],
            evidence_path=first,
        )

    assert first.read_bytes() == b"SYNTHETIC_FIRST"
    assert second.read_bytes() == b"SYNTHETIC_SECOND"


def test_partial_deletion_rolls_back_all_staged_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first_fingerprint = _write_artifact(first, b"SYNTHETIC_FIRST")
    second_fingerprint = _write_artifact(second, b"SYNTHETIC_SECOND")
    evidence_path = tmp_path / "rollback.json"
    real_unlink = deletion_verify.os.unlink
    payload_unlinks = 0

    def fail_on_second_payload_unlink(path: str | bytes | Path, *args: object) -> None:
        nonlocal payload_unlinks
        if Path(path).parent.name == "payload":
            payload_unlinks += 1
            if payload_unlinks == 2:
                raise OSError("synthetic deletion failure")
        real_unlink(path, *args)

    monkeypatch.setattr(deletion_verify.os, "unlink", fail_on_second_payload_unlink)

    with pytest.raises(DeletionTransactionError) as error:
        delete_verified_artifacts(
            tmp_path,
            [(first, first_fingerprint), (second, second_fingerprint)],
            evidence_path=evidence_path,
        )

    assert "synthetic" not in str(error.value).lower()
    assert first.read_bytes() == b"SYNTHETIC_FIRST"
    assert second.read_bytes() == b"SYNTHETIC_SECOND"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "rolled_back"
    assert evidence["rolled_back_count"] == 2


def test_stage_failure_restores_the_file_that_was_already_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first_fingerprint = _write_artifact(first, b"SYNTHETIC_FIRST")
    second_fingerprint = _write_artifact(second, b"SYNTHETIC_SECOND")
    real_copy = deletion_verify._copy_descriptor_to_path
    copy_calls = 0

    def fail_on_second_copy(descriptor: int, target: Path, mode: int) -> None:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 2:
            raise OSError("synthetic staging failure")
        real_copy(descriptor, target, mode)

    monkeypatch.setattr(
        deletion_verify,
        "_copy_descriptor_to_path",
        fail_on_second_copy,
    )

    with pytest.raises(DeletionTransactionError):
        delete_verified_artifacts(
            tmp_path,
            [(first, first_fingerprint), (second, second_fingerprint)],
        )

    assert first.read_bytes() == b"SYNTHETIC_FIRST"
    assert second.read_bytes() == b"SYNTHETIC_SECOND"


def test_commit_cleanup_failure_restores_already_unlinked_backups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first_fingerprint = _write_artifact(first, b"SYNTHETIC_FIRST")
    second_fingerprint = _write_artifact(second, b"SYNTHETIC_SECOND")
    evidence_path = tmp_path / "rollback.json"
    real_unlink = deletion_verify.os.unlink
    backup_unlinks = 0

    def fail_on_second_backup_unlink(path: str | bytes | Path, *args: object) -> None:
        nonlocal backup_unlinks
        if Path(path).parent.name == "backup":
            backup_unlinks += 1
            if backup_unlinks == 2:
                raise OSError("synthetic cleanup failure")
        real_unlink(path, *args)

    monkeypatch.setattr(deletion_verify.os, "unlink", fail_on_second_backup_unlink)

    with pytest.raises(DeletionTransactionError):
        delete_verified_artifacts(
            tmp_path,
            [(first, first_fingerprint), (second, second_fingerprint)],
            evidence_path=evidence_path,
        )

    assert first.read_bytes() == b"SYNTHETIC_FIRST"
    assert second.read_bytes() == b"SYNTHETIC_SECOND"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "rolled_back"
    assert evidence["rolled_back_count"] == 2


def test_staged_payload_corruption_rolls_back_from_independent_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    fingerprint = _write_artifact(artifact, b"SYNTHETIC_ORIGINAL")
    real_verify = deletion_verify._verify_staged_before_commit

    def corrupt_then_verify(staged: list[object]) -> None:
        staged_item = staged[0]
        staged_item.payload.write_bytes(b"SYNTHETIC_CORRUPTED")
        real_verify(staged)

    monkeypatch.setattr(
        deletion_verify,
        "_verify_staged_before_commit",
        corrupt_then_verify,
    )

    with pytest.raises(DeletionTransactionError):
        delete_verified_artifacts(tmp_path, [(artifact, fingerprint)])

    assert artifact.read_bytes() == b"SYNTHETIC_ORIGINAL"


def test_artifact_count_is_bounded_before_any_file_is_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first_fingerprint = _write_artifact(first, b"SYNTHETIC_FIRST")
    second_fingerprint = _write_artifact(second, b"SYNTHETIC_SECOND")
    monkeypatch.setattr(deletion_verify, "MAX_ARTIFACTS", 1)

    with pytest.raises(deletion_verify.InvalidDeletionRequest):
        delete_verified_artifacts(
            tmp_path,
            [(first, first_fingerprint), (second, second_fingerprint)],
        )

    assert first.read_bytes() == b"SYNTHETIC_FIRST"
    assert second.read_bytes() == b"SYNTHETIC_SECOND"


def test_hostile_path_and_iterable_errors_are_value_free(tmp_path: Path) -> None:
    marker = "synthetic-sensitive-path"

    class HostilePath:
        def __fspath__(self) -> str:
            raise RuntimeError(marker)

    class HostileArtifacts:
        def __iter__(self):
            raise RuntimeError(marker)

    for root, artifacts in (
        (HostilePath(), []),
        (tmp_path, HostileArtifacts()),
    ):
        with pytest.raises(deletion_verify.InvalidDeletionRequest) as error:
            delete_verified_artifacts(root, artifacts)
        assert marker not in str(error.value)


def test_evidence_publish_failure_after_cleanup_restores_from_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    fingerprint = _write_artifact(artifact, b"SYNTHETIC_ORIGINAL")
    evidence_path = tmp_path / "evidence.json"
    real_replace = deletion_verify.os.replace
    failed_publish = False

    def fail_first_evidence_publish(
        source: str | bytes | Path,
        destination: str | bytes | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal failed_publish
        if Path(destination) == evidence_path and not failed_publish:
            failed_publish = True
            raise OSError("synthetic evidence failure")
        real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(deletion_verify.os, "replace", fail_first_evidence_publish)

    with pytest.raises(deletion_verify.EvidenceWriteError):
        delete_verified_artifacts(
            tmp_path,
            [(artifact, fingerprint)],
            evidence_path=evidence_path,
        )

    assert artifact.read_bytes() == b"SYNTHETIC_ORIGINAL"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "rolled_back"
    assert evidence["rolled_back_count"] == 1
    assert list(tmp_path.glob(".openmed-deletion-*")) == []


@pytest.fixture(autouse=True, params=[False, True])
def recovery_backend(request, monkeypatch):
    if deletion_verify.os.name == "nt" and not request.param:
        pytest.skip("Descriptor recovery requires POSIX unlink semantics")
    monkeypatch.setattr(deletion_verify, "_USE_MEMORY_RECOVERY", request.param)


def test_binary_fingerprints_preserve_crlf_and_control_z(tmp_path):
    content = b"SYNTHETIC\r\n\x1a\x00\xff"
    artifact = tmp_path / "binary.bin"
    fingerprint = _write_artifact(artifact, content)
    assert fingerprint == "sha256:" + hashlib.sha256(content).hexdigest()
    assert delete_verified_artifacts(tmp_path, [(artifact, fingerprint)]).passed


def test_memory_recovery_limit_rolls_back_before_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(deletion_verify, "_USE_MEMORY_RECOVERY", True)
    monkeypatch.setattr(deletion_verify, "MAX_RECOVERY_BYTES", 20)
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    fingerprints = [
        (path, _write_artifact(path, b"SYNTHETIC_123")) for path in (first, second)
    ]
    with pytest.raises(DeletionTransactionError):
        delete_verified_artifacts(tmp_path, fingerprints)
    assert first.read_bytes() == second.read_bytes() == b"SYNTHETIC_123"
    assert list(tmp_path.glob(".openmed-deletion-*")) == []


def test_hash_accepts_windows_path_and_descriptor_creation_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    artifact = tmp_path / "synthetic.bin"
    content = b"SYNTHETIC\r\n\x1a"
    artifact.write_bytes(content)
    descriptor = deletion_verify._open_for_hash(artifact)
    real_fstat = deletion_verify.os.fstat
    state = real_fstat(descriptor)
    fields = {
        name: getattr(state, name)
        for name in (
            "st_mode",
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
    }
    expected = SimpleNamespace(**fields, st_birthtime_ns=123)
    opened = SimpleNamespace(
        **{**fields, "st_ctime_ns": fields["st_ctime_ns"] + 1}, st_birthtime_ns=123
    )
    monkeypatch.setattr(deletion_verify, "_WINDOWS_STAT", True, raising=False)
    monkeypatch.setattr(deletion_verify.os, "fstat", lambda fd: opened)
    try:
        assert deletion_verify._hash_descriptor(descriptor, expected) == (
            "sha256:" + hashlib.sha256(content).hexdigest()
        )
    finally:
        deletion_verify.os.close(descriptor)


@pytest.mark.parametrize(
    "field",
    [
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_nlink",
        "st_birthtime_ns",
    ],
)
def test_windows_hash_rejects_changed_identity_or_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    from types import SimpleNamespace

    artifact = tmp_path / "synthetic.bin"
    artifact.write_bytes(b"SYNTHETIC")
    descriptor = deletion_verify._open_for_hash(artifact)
    state = deletion_verify.os.fstat(descriptor)
    fields = {
        name: getattr(state, name)
        for name in (
            "st_mode",
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
    }
    fields["st_birthtime_ns"] = 123
    expected = SimpleNamespace(**fields)
    opened = SimpleNamespace(**{**fields, field: fields[field] + 1})
    monkeypatch.setattr(deletion_verify, "_WINDOWS_STAT", True, raising=False)
    monkeypatch.setattr(deletion_verify.os, "fstat", lambda fd: opened)
    try:
        with pytest.raises(DeletionTransactionError):
            deletion_verify._hash_descriptor(descriptor, expected)
    finally:
        deletion_verify.os.close(descriptor)


def test_windows_hash_rejects_descriptor_metadata_change_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    artifact = tmp_path / "synthetic.bin"
    artifact.write_bytes(b"SYNTHETIC")
    descriptor = deletion_verify._open_for_hash(artifact)
    state = deletion_verify.os.fstat(descriptor)
    fields = {
        name: getattr(state, name)
        for name in (
            "st_mode",
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
    }
    fields["st_birthtime_ns"] = 123
    expected = SimpleNamespace(**fields)
    changed = SimpleNamespace(**{**fields, "st_ctime_ns": fields["st_ctime_ns"] + 1})
    states = iter([expected, changed])
    monkeypatch.setattr(deletion_verify, "_WINDOWS_STAT", True, raising=False)
    monkeypatch.setattr(deletion_verify.os, "fstat", lambda fd: next(states))
    try:
        with pytest.raises(DeletionTransactionError):
            deletion_verify._hash_descriptor(descriptor, expected)
    finally:
        deletion_verify.os.close(descriptor)
