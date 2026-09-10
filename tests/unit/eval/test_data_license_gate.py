"""Tests for the training-data license gate (OM-095)."""

from __future__ import annotations

import json
import socket
from pathlib import Path

from openmed.core.manifest_schema import validate_manifest_row
from openmed.eval.data_license_gate import (
    DATA_LICENSE_GATE,
    data_license_gate_errors,
    evaluate_data_license_gate,
)
from openmed.eval.release_gates import _manifest_coherence_check

_REPO_ID = "OpenMed/clinical-ner-v1"


def _base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "repo_id": _REPO_ID,
        "family": "NER",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Small",
        "param_count": 44_000_000,
        "architecture": "distilbert",
        "base_model": "distilbert-base-uncased",
        "formats": ["pytorch"],
        "canonical_labels": ["PROBLEM"],
        "benchmark": {
            "dataset": "synthetic-clinical-ner",
            "micro_f1": 0.9,
            "recall": 0.9,
        },
        "arxiv": None,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:" + "a" * 64,
        "released": "2026-01-01",
    }
    row.update(overrides)
    return row


def _write_manifest(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "models.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_non_redistributable_train_role_source_fails(tmp_path: Path) -> None:
    row = _base_row(
        training_data_licenses=[
            {
                "name": "UMLS Metathesaurus",
                "license": "UMLS-Metathesaurus-License",
                "redistributable": False,
                "role": "train",
            }
        ]
    )
    manifest = _write_manifest(tmp_path, [row])

    check = evaluate_data_license_gate(manifest)

    assert check.gate == DATA_LICENSE_GATE
    assert check.passed is False
    assert check.details["violations"] == [
        {
            "repo_id": _REPO_ID,
            "name": "UMLS Metathesaurus",
            "license": "UMLS-Metathesaurus-License",
        }
    ]

    errors = data_license_gate_errors(manifest)
    assert len(errors) == 1
    assert _REPO_ID in errors[0]
    assert "UMLS Metathesaurus" in errors[0]


def test_same_source_with_eval_role_passes(tmp_path: Path) -> None:
    row = _base_row(
        training_data_licenses=[
            {
                "name": "UMLS Metathesaurus",
                "license": "UMLS-Metathesaurus-License",
                "redistributable": False,
                "role": "eval",
            }
        ]
    )
    manifest = _write_manifest(tmp_path, [row])

    check = evaluate_data_license_gate(manifest)

    assert check.passed is True
    assert data_license_gate_errors(manifest) == []


def test_fully_public_training_data_passes(tmp_path: Path) -> None:
    row = _base_row(
        training_data_licenses=[
            {
                "name": "MedMentions",
                "license": "CC0-1.0",
                "redistributable": True,
                "role": "train",
            }
        ]
    )
    manifest = _write_manifest(tmp_path, [row])

    check = evaluate_data_license_gate(manifest)

    assert check.passed is True
    assert data_license_gate_errors(manifest) == []


def test_manifest_coherence_path_enforces_data_license_gate(tmp_path: Path) -> None:
    row = _base_row(
        training_data_licenses=[
            {
                "name": "UMLS Metathesaurus",
                "license": "UMLS-Metathesaurus-License",
                "redistributable": False,
                "role": "train",
            }
        ]
    )
    manifest = _write_manifest(tmp_path, [row])
    identity = {
        "repo_id": _REPO_ID,
        "family": "NER",
        "tier": "Small",
        "param_count": 44_000_000,
        "format": "pytorch",
        "eval_set_hash": "sha256:" + "b" * 64,
        "leakage_fixture_hash": "sha256:" + "c" * 64,
    }

    check = _manifest_coherence_check(
        identity,
        {"manifest_path": str(manifest), "require_manifest_row": True},
    )

    assert check.passed is False
    mismatch = check.details["mismatches"]["training_data_licenses"]
    assert mismatch["details"]["violations"][0]["repo_id"] == _REPO_ID


def test_manifest_coherence_path_allows_eval_only_restricted_data(
    tmp_path: Path,
) -> None:
    row = _base_row(
        training_data_licenses=[
            {
                "name": "UMLS Metathesaurus",
                "license": "UMLS-Metathesaurus-License",
                "redistributable": False,
                "role": "eval",
            }
        ]
    )
    manifest = _write_manifest(tmp_path, [row])
    identity = {
        "repo_id": _REPO_ID,
        "family": "NER",
        "tier": "Small",
        "param_count": 44_000_000,
        "format": "pytorch",
        "eval_set_hash": "sha256:" + "b" * 64,
        "leakage_fixture_hash": "sha256:" + "c" * 64,
    }

    check = _manifest_coherence_check(
        identity,
        {"manifest_path": str(manifest), "require_manifest_row": True},
    )

    assert check.passed is True


def test_unpublished_checkpoint_is_not_gated(tmp_path: Path) -> None:
    row = _base_row(
        released=None,
        training_data_licenses=[
            {
                "name": "SNOMED CT",
                "license": "SNOMED-CT-Affiliate-License",
                "redistributable": False,
                "role": "train",
            }
        ],
    )
    manifest = _write_manifest(tmp_path, [row])

    check = evaluate_data_license_gate(manifest)

    assert check.passed is True


def test_missing_redistributable_field_fails_schema_validation() -> None:
    row = _base_row(
        training_data_licenses=[
            {
                "name": "CPT",
                "license": "AMA-CPT-License",
                "role": "eval",
            }
        ]
    )

    violations = validate_manifest_row(row, line_number=1)

    assert any(
        "training_data_licenses[0] missing required key: redistributable"
        in str(violation)
        for violation in violations
    )


def test_gate_is_read_only_and_never_touches_the_network(
    tmp_path: Path, monkeypatch
) -> None:
    row = _base_row(
        training_data_licenses=[
            {
                "name": "SNOMED CT",
                "license": "SNOMED-CT-Affiliate-License",
                "redistributable": False,
                "role": "train",
            }
        ]
    )
    manifest = _write_manifest(tmp_path, [row])
    original_bytes = manifest.read_bytes()

    blocked_attempts: list[tuple] = []

    def fail_socket(*args: object, **kwargs: object) -> None:
        blocked_attempts.append((args, kwargs))
        raise AssertionError("network egress attempted from data_license_gate")

    monkeypatch.setattr(socket.socket, "connect", fail_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)

    check = evaluate_data_license_gate(manifest)

    assert check.passed is False
    assert blocked_attempts == []
    assert manifest.read_bytes() == original_bytes
