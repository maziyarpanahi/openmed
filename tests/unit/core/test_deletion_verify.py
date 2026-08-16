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
    real_link = deletion_verify.os.link
    link_calls = 0

    def fail_on_second_link(
        source: str | bytes | Path,
        destination: str | bytes | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("synthetic staging failure")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(deletion_verify.os, "link", fail_on_second_link)

    with pytest.raises(DeletionTransactionError):
        delete_verified_artifacts(
            tmp_path,
            [(first, first_fingerprint), (second, second_fingerprint)],
        )

    assert first.read_bytes() == b"SYNTHETIC_FIRST"
    assert second.read_bytes() == b"SYNTHETIC_SECOND"
