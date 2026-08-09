"""Focused tests for deterministic, privacy-safe release verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openmed.compliance.reproducible_release import (
    MISMATCH_ARTIFACT_HASH,
    MISMATCH_BUILD_INPUTS,
    MISMATCH_CATEGORIES,
    MISMATCH_DEPENDENCY_LOCK,
    MISMATCH_PROVENANCE_SCHEMA,
    MISMATCH_SCHEMA,
    MISMATCH_SOURCE_REVISION,
    PROVENANCE_SCHEMA,
    ReleaseProvenance,
    ReleaseProvenanceError,
    ReleaseProvenanceVerificationError,
    build_release_provenance,
    compute_artifact_digest,
    compute_build_inputs_digest,
    compute_dependency_lock_digest,
    verify_release_provenance,
)


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _record(
    *,
    build_inputs: dict[str, object] | None = None,
    source_revision: str = "a" * 40,
    dependency_lock_digest: str = _digest("b"),
    artifact_hashes: dict[str, str] | None = None,
    provenance_schema: str = PROVENANCE_SCHEMA,
    schema_version: str = "openmed.reproducible_release.v1",
) -> ReleaseProvenance:
    return build_release_provenance(
        source_revision=source_revision,
        build_inputs=build_inputs
        or {
            "python": "3.10",
            "toolchain": {"flags": ["-O2", "-fno-plt"], "cc": "clang"},
        },
        dependency_lock_digest=dependency_lock_digest,
        artifact_hashes=artifact_hashes
        or {"sdist": _digest("c"), "wheel": _digest("d")},
        provenance_schema=provenance_schema,
        schema_version=schema_version,
    )


def test_build_inputs_are_canonical_and_records_are_reproducible() -> None:
    first = _record(
        build_inputs={
            "toolchain": {"flags": {"-fno-plt", "-O2"}, "cc": "clang"},
            "python": "3.10",
        }
    )
    second = _record(
        build_inputs={
            "python": "3.10",
            "toolchain": {"cc": "clang", "flags": {"-O2", "-fno-plt"}},
        }
    )

    assert (
        compute_build_inputs_digest(
            {
                "python": "3.10",
                "toolchain": {"cc": "clang", "flags": {"-O2", "-fno-plt"}},
            }
        )
        == first.build_inputs_digest
    )
    assert first == second
    report = verify_release_provenance(first, second)
    assert report.valid is True
    assert report.verify() is True
    assert bool(report) is True
    assert report.mismatch_categories == ()
    assert report.to_dict() == {
        "mismatch_categories": [],
        "mismatches": [],
        "schema_version": "openmed.reproducible_release.v1",
        "valid": True,
    }


def test_verifier_emits_stable_categories_for_each_claim() -> None:
    expected = _record(
        build_inputs={
            "credential": "synthetic-credential-not-for-evidence",
            "source_path": "/private/synthetic/source-payload",
        }
    )
    actual = _record(
        build_inputs={"python": "3.11"},
        source_revision="e" * 40,
        dependency_lock_digest=_digest("f"),
        artifact_hashes={"wheel": _digest("1")},
        provenance_schema="in-toto/v1",
        schema_version="openmed.reproducible_release.v2",
    )

    report = verify_release_provenance(expected, actual)

    assert report.valid is False
    assert report.mismatch_categories == MISMATCH_CATEGORIES
    assert list(dict.fromkeys(item.category for item in report.mismatches)) == list(
        MISMATCH_CATEGORIES
    )
    evidence = json.dumps(report.to_dict(), sort_keys=True)
    assert "synthetic-credential-not-for-evidence" not in evidence
    assert "/private/synthetic/source-payload" not in evidence
    with pytest.raises(ReleaseProvenanceVerificationError) as raised:
        report.raise_if_invalid()
    assert "synthetic-credential-not-for-evidence" not in str(raised.value)
    assert "/private/synthetic/source-payload" not in str(raised.value)
    assert set(report.mismatch_categories) == {
        MISMATCH_SCHEMA,
        MISMATCH_SOURCE_REVISION,
        MISMATCH_BUILD_INPUTS,
        MISMATCH_DEPENDENCY_LOCK,
        MISMATCH_ARTIFACT_HASH,
        MISMATCH_PROVENANCE_SCHEMA,
    }


def test_mapping_round_trip_retains_only_safe_evidence() -> None:
    record = _record()
    payload = record.to_dict()
    restored = ReleaseProvenance.from_mapping(payload)

    assert restored == record
    assert restored.to_evidence() == payload
    assert "build_inputs" not in payload
    assert "credentials" not in json.dumps(payload)


def test_local_lock_and_artifact_hashing_is_offline_and_path_free(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "synthetic.lock"
    artifact = tmp_path / "synthetic.whl"
    lock.write_bytes(b"synthetic dependency lock")
    artifact.write_bytes(b"synthetic release artifact")

    record = build_release_provenance(
        source_revision="a" * 40,
        build_inputs={"python": "3.10"},
        dependency_lock=lock,
        artifacts={"wheel": artifact},
    )

    assert compute_dependency_lock_digest(lock) == (
        "sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest()
    )
    assert compute_artifact_digest(artifact) == (
        "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    assert str(tmp_path) not in json.dumps(record.to_dict(), sort_keys=True)


def test_invalid_input_errors_do_not_echo_paths_or_raw_values() -> None:
    with pytest.raises(ReleaseProvenanceError) as raised:
        build_release_provenance(
            source_revision="a" * 40,
            build_inputs={"python": "3.10"},
            dependency_lock_digest=_digest("b"),
            artifact_hashes={"/private/synthetic/credential-path.whl": _digest("c")},
        )

    assert "/private/synthetic/credential-path.whl" not in str(raised.value)
    assert "credential" not in str(raised.value)
