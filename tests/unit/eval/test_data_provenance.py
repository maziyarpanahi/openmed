"""Tests for PHI-safe training data provenance manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path

from openmed.eval.data_provenance import (
    build_training_data_manifest,
    compute_training_data_manifest_hash,
    write_training_data_manifest,
)


def _fixture() -> dict[str, object]:
    return {
        "fixture_id": "doc-001",
        "language": "en",
        "text": "Patient Alice Smith lives in Paris.",
        "gold_spans": [
            {"start": 8, "end": 19, "label": "PERSON", "text": "Alice Smith"},
            {"start": 29, "end": 34, "label": "LOCATION", "text": "Paris"},
        ],
    }


def test_training_data_manifest_hashes_content_without_persisting_raw_text() -> None:
    manifest = build_training_data_manifest(
        [_fixture()],
        dataset_id="synthetic-pii",
        data_revision="git:abc123",
    )
    serialized = json.dumps(manifest, sort_keys=True)

    assert "Alice" not in serialized
    assert "Paris" not in serialized
    assert "Patient" not in serialized
    assert manifest["data_revision"] == "git:abc123"
    assert manifest["fixture_count"] == 1
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", manifest["manifest_hash"])

    fixture = manifest["fixtures"][0]
    assert fixture["fixture_id"] == "doc-001"
    assert fixture["text_length"] == len(_fixture()["text"])
    assert fixture["spans"] == [
        {
            "end": 19,
            "label": "PERSON",
            "length": 11,
            "start": 8,
            "text_sha256": fixture["spans"][0]["text_sha256"],
        },
        {
            "end": 34,
            "label": "LOCATION",
            "length": 5,
            "start": 29,
            "text_sha256": fixture["spans"][1]["text_sha256"],
        },
    ]
    assert all(
        re.fullmatch(r"sha256:[0-9a-f]{64}", span["text_sha256"])
        for span in fixture["spans"]
    )


def test_training_data_manifest_hash_is_deterministic() -> None:
    first = build_training_data_manifest(
        [_fixture()],
        dataset_id="synthetic-pii",
        data_revision="git:abc123",
    )
    second = build_training_data_manifest(
        [_fixture()],
        data_revision="git:abc123",
        dataset_id="synthetic-pii",
    )

    assert first == second
    assert compute_training_data_manifest_hash(first) == first["manifest_hash"]


def test_write_training_data_manifest_outputs_json(tmp_path: Path) -> None:
    path = write_training_data_manifest(
        tmp_path / "training-data-manifest.json",
        [_fixture()],
        dataset_id="synthetic-pii",
        data_revision="git:abc123",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.exists()
    assert payload["dataset_id"] == "synthetic-pii"
    assert payload["manifest_hash"] == compute_training_data_manifest_hash(payload)
