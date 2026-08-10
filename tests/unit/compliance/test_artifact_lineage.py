"""Focused tests for privacy-artifact lineage manifests."""

from __future__ import annotations

import json
import socket
from dataclasses import replace

import pytest

from openmed.compliance import (
    ArtifactLineageManifest,
    ArtifactLineageNode,
    ArtifactLineageParent,
    ArtifactLineageValidationError,
    build_artifact_lineage_manifest,
    compute_artifact_hash,
    compute_policy_fingerprint,
    verify_artifact_lineage,
)


def _policy_fingerprint() -> str:
    return compute_policy_fingerprint(
        {
            "name": "synthetic-redaction-policy",
            "rules": ["mask-direct-identifiers", "shift-dates"],
        }
    )


def _lineage_nodes() -> tuple[ArtifactLineageNode, ArtifactLineageNode]:
    policy_fingerprint = _policy_fingerprint()
    source = ArtifactLineageNode.create(
        artifact_type="source-input",
        transformation="ingest",
        policy_fingerprint=policy_fingerprint,
        schema_version=1,
    )
    derived = ArtifactLineageNode.create(
        artifact_type="redacted-output",
        parents=(ArtifactLineageParent("source-input", source.artifact_hash),),
        transformation="redact",
        policy_fingerprint=policy_fingerprint,
        schema_version=1,
    )
    return source, derived


def test_manifest_records_typed_parents_and_is_deterministic() -> None:
    source, derived = _lineage_nodes()

    first = build_artifact_lineage_manifest((derived, source))
    second = build_artifact_lineage_manifest((source, derived))

    assert first.to_json() == second.to_json()
    assert first.verify().to_dict() == {
        "cycle_count": 0,
        "duplicate_hash_count": 0,
        "hash_mismatch_count": 0,
        "missing_parent_count": 0,
        "node_count": 2,
        "parent_reference_count": 1,
        "valid": True,
    }
    assert derived.to_dict()["parents"] == [
        {"hash": source.artifact_hash, "type": "source-input"}
    ]
    assert derived.to_dict()["transformation"] == "redact"
    assert derived.to_dict()["policy_fingerprint"] == _policy_fingerprint()
    assert "synthetic-redaction-policy" not in first.to_json()


def test_manifest_round_trips_without_network_or_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, derived = _lineage_nodes()
    manifest = build_artifact_lineage_manifest((source, derived))

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("lineage manifest attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    restored = ArtifactLineageManifest.from_json(manifest.to_json())

    assert restored.to_json() == manifest.to_json()
    assert verify_artifact_lineage(restored.to_dict()).valid


def test_verification_counts_missing_parents_cycles_and_hash_mismatches() -> None:
    policy_fingerprint = _policy_fingerprint()
    missing = ArtifactLineageNode.create(
        artifact_type="derived-output",
        parents=(
            ArtifactLineageParent(
                "source-input",
                compute_policy_fingerprint("synthetic-missing-parent"),
            ),
        ),
        transformation="redact",
        policy_fingerprint=policy_fingerprint,
        schema_version=1,
    )
    source, _ = _lineage_nodes()
    wrong_type = ArtifactLineageNode.create(
        artifact_type="derived-output",
        parents=(ArtifactLineageParent("wrong-type", source.artifact_hash),),
        transformation="redact",
        policy_fingerprint=policy_fingerprint,
        schema_version=1,
    )
    tampered = replace(source, transformation="unexpected-transform")

    diagnostics = ArtifactLineageManifest(
        nodes=(missing, wrong_type, tampered)
    ).verify()

    assert diagnostics.missing_parent_count == 1
    assert diagnostics.hash_mismatch_count == 2
    assert diagnostics.cycle_count == 0
    assert diagnostics.valid is False
    assert "sha256:" not in json.dumps(diagnostics.to_dict(), sort_keys=True)


def test_verification_counts_one_cycle_without_returning_node_values() -> None:
    policy_fingerprint = _policy_fingerprint()
    first_hash = compute_artifact_hash(
        "cycle-first",
        transformation="record",
        policy_fingerprint=policy_fingerprint,
        schema_version=1,
    )
    second_hash = compute_artifact_hash(
        "cycle-second",
        transformation="record",
        policy_fingerprint=policy_fingerprint,
        schema_version=1,
    )
    first = ArtifactLineageNode(
        artifact_hash=first_hash,
        artifact_type="cycle-first",
        transformation="record",
        policy_fingerprint=policy_fingerprint,
        schema_version=1,
        parents=(ArtifactLineageParent("cycle-second", second_hash),),
    )
    second = ArtifactLineageNode(
        artifact_hash=second_hash,
        artifact_type="cycle-second",
        transformation="record",
        policy_fingerprint=policy_fingerprint,
        schema_version=1,
        parents=(ArtifactLineageParent("cycle-first", first_hash),),
    )

    diagnostics = build_artifact_lineage_manifest((first, second)).verify()

    assert diagnostics.cycle_count == 1
    assert diagnostics.to_dict()["node_count"] == 2
    assert first_hash not in json.dumps(diagnostics.to_dict(), sort_keys=True)


def test_invalid_values_never_appear_in_validation_errors() -> None:
    raw_marker = "synthetic-sensitive-value"

    with pytest.raises(ArtifactLineageValidationError) as error:
        ArtifactLineageNode(
            artifact_hash=raw_marker,
            artifact_type="source-input",
            transformation="ingest",
        )

    assert raw_marker not in str(error.value)

    with pytest.raises(ArtifactLineageValidationError) as error:
        ArtifactLineageNode.from_mapping(
            {
                "artifact_type": "source-input",
                "transformation": "ingest",
                "payload": raw_marker,
            }
        )

    assert raw_marker not in str(error.value)
