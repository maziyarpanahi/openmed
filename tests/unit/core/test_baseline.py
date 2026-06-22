"""Tests for last-green baseline storage and reproducibility hashes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.core.baseline import (
    BASELINE_PATH,
    BASELINE_SCHEMA_VERSION,
    BaselineMiss,
    baseline_key,
    get_baseline,
    load_baseline_store,
    require_baseline,
    update_baseline_entry,
    validate_baseline_store,
    write_baseline_store,
)
from openmed.core.hf_publish import publish_artifact
from openmed.core.repro_hash import compute_reproducibility_hash


class RepositoryNotFoundError(Exception):
    pass


class FakeApi:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.uploaded: list[dict[str, object]] = []

    def repo_info(self, **kwargs: object) -> object:
        raise RepositoryNotFoundError("not found")

    def create_repo(self, **kwargs: object) -> None:
        self.created.append(dict(kwargs))

    def upload_folder(self, **kwargs: object) -> None:
        self.uploaded.append(dict(kwargs))


def _store() -> dict[str, object]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "entries": {
            "pii::small::mlx-fp": {
                "key": "pii::small::mlx-fp",
                "family": "PII",
                "tier": "Small",
                "format": "mlx-fp",
                "metrics": {"micro_f1": 0.98, "recall": 0.97},
                "reproducibility_hash": "sha256:" + "a" * 64,
                "repo_id": "OpenMed/pii-small-mlx",
                "released": "2026-06-01",
            },
            "ner::large::pytorch": {
                "key": "ner::large::pytorch",
                "family": "NER",
                "tier": "Large",
                "format": "pytorch",
                "metrics": {"micro_f1": 0.93},
                "reproducibility_hash": "sha256:" + "b" * 64,
                "repo_id": "OpenMed/ner-large",
                "released": "2026-05-20",
            },
        },
    }


def _write_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "_mlx_task": "token-classification",
                "_mlx_model_type": "deberta-v2",
                "id2label": {"0": "O", "1": "B-PERSON", "2": "I-DATE"},
            }
        ),
        encoding="utf-8",
    )
    (artifact / "openmed-mlx.json").write_text(
        json.dumps({"format": "openmed-mlx", "format_version": 2}),
        encoding="utf-8",
    )
    (artifact / "weights.safetensors").write_bytes(b"weights")
    return artifact


def test_committed_baseline_store_validates_schema() -> None:
    store = load_baseline_store(BASELINE_PATH)

    validate_baseline_store(store)
    assert store["schema_version"] == BASELINE_SCHEMA_VERSION
    assert get_baseline("PII", "Small", "mlx-fp", store=store) is not None


def test_update_baseline_entry_preserves_other_keys(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline_store(_store(), path)
    before = load_baseline_store(path)
    untouched = before["entries"]["ner::large::pytorch"]

    updated = update_baseline_entry(
        path,
        family="PII",
        tier="Small",
        format_name="mlx-fp",
        metrics={"micro_f1": 0.991, "recall": 0.988},
        reproducibility_hash="sha256:" + "c" * 64,
        repo_id="OpenMed/pii-small-mlx-v2",
        source_model_id="OpenMed/pii-small",
        released="2026-06-14",
        git_sha="abc123",
    )
    after = load_baseline_store(path)

    assert updated["key"] == "pii::small::mlx-fp"
    assert after["entries"]["pii::small::mlx-fp"]["metrics"]["micro_f1"] == 0.991
    assert after["entries"]["ner::large::pytorch"] == untouched


def test_reader_returns_entry_and_clear_miss() -> None:
    store = _store()

    entry = get_baseline("PII", "Small", "mlx-fp", store=store)
    missing = get_baseline("PII", "Large", "mlx-fp", store=store)

    assert entry is not None
    assert entry["repo_id"] == "OpenMed/pii-small-mlx"
    assert missing is None
    with pytest.raises(BaselineMiss, match="No last-green baseline"):
        require_baseline("PII", "Large", "mlx-fp", store=store)


def test_reproducibility_hash_is_deterministic_for_fixed_inputs() -> None:
    first = compute_reproducibility_hash(
        recipe={"format": "mlx-fp", "steps": ["convert", "publish"]},
        data_manifest={"dataset": "fixture", "revision": "2026-06-14"},
        base_model="OpenMed/base-model",
        git_sha="abc123",
    )
    second = compute_reproducibility_hash(
        recipe={"steps": ["convert", "publish"], "format": "mlx-fp"},
        data_manifest={"revision": "2026-06-14", "dataset": "fixture"},
        base_model="OpenMed/base-model",
        git_sha="abc123",
    )
    changed = compute_reproducibility_hash(
        recipe={"format": "mlx-fp", "steps": ["convert", "publish"]},
        data_manifest={"dataset": "fixture", "revision": "2026-06-14"},
        base_model="OpenMed/base-model",
        git_sha="def456",
    )

    assert first == second
    assert first != changed
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first)


def test_baseline_key_uses_family_tier_format_dimensions() -> None:
    assert baseline_key("PII", "Small", "MLX_FP") == "pii::small::mlx-fp"
    assert baseline_key("General", None, "coreml") == "general::none::coreml"


def test_publish_artifact_updates_manifest_and_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = _write_artifact(tmp_path)
    manifest = tmp_path / "models.jsonl"
    baseline = tmp_path / "baseline.json"
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    result = publish_artifact(
        artifact_dir=artifact,
        source_model_id="OpenMed/test-model",
        format_name="mlx-fp",
        manifest_path=manifest,
        baseline_path=baseline,
        baseline_metrics={"micro_f1": 0.99, "recall": 0.98},
        api=FakeApi(),
        released="2026-06-14",
        git_sha="abc123",
    )

    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    store = load_baseline_store(baseline)
    entry = require_baseline("General", None, "mlx-fp", store=store)

    assert rows == [result.manifest_row]
    assert entry["repo_id"] == "OpenMed/test-model-v1-mlx"
    assert entry["metrics"] == {"micro_f1": 0.99, "recall": 0.98}
    assert entry["reproducibility_hash"] == rows[0]["reproducibility_hash"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", rows[0]["reproducibility_hash"])
