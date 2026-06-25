"""Tests for publishing converted model artifacts."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from openmed.core.hf_publish import (
    HfPublishError,
    append_manifest_row,
    build_manifest_row,
    publish_artifact,
    target_repo_id,
)


class RepositoryNotFoundError(Exception):
    pass


class FakeApi:
    def __init__(self, *, exists: bool = False):
        self.exists = exists
        self.created = []
        self.uploaded = []
        self.uploaded_cards = []
        self.info_calls = []

    def repo_info(self, **kwargs):
        self.info_calls.append(kwargs)
        if not self.exists:
            raise RepositoryNotFoundError("not found")
        return object()

    def create_repo(self, **kwargs):
        self.created.append(kwargs)
        self.exists = True

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)
        self.uploaded_cards.append(
            (Path(kwargs["folder_path"]) / "README.md").read_text(encoding="utf-8")
        )


def _write_mlx_artifact(tmp_path):
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
    (artifact / "weights.safetensors").write_bytes(b"weights")
    return artifact


def test_target_repo_id_keeps_existing_version_and_adds_variant():
    assert (
        target_repo_id(
            "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
            "mlx-fp",
        )
        == "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx"
    )


def test_target_repo_id_adds_version_before_8bit_variant():
    assert (
        target_repo_id("OpenMed/test-model", "mlx-8bit", version=2)
        == "OpenMed/test-model-v2-mlx-8bit"
    )


def test_publish_artifact_creates_repo_uploads_folder_and_writes_manifest(
    tmp_path, monkeypatch
):
    artifact = _write_mlx_artifact(tmp_path)
    manifest = tmp_path / "models.jsonl"
    fake_api = FakeApi()
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    result = publish_artifact(
        artifact_dir=artifact,
        source_model_id="OpenMed/test-model",
        format_name="mlx-fp",
        manifest_path=manifest,
        api=fake_api,
        released="2026-06-14",
    )

    assert result.repo_id == "OpenMed/test-model-v1-mlx"
    assert result.skipped is False
    assert fake_api.created[0]["repo_id"] == result.repo_id
    assert fake_api.created[0]["token"] == "secret-token"
    assert fake_api.uploaded[0]["folder_path"] == str(artifact)
    assert fake_api.uploaded[0]["token"] == "secret-token"
    assert (artifact / "README.md").exists()
    assert "OpenMed/test-model-v1-mlx" in fake_api.uploaded_cards[0]
    assert "sha256:" in fake_api.uploaded_cards[0]

    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [result.manifest_row]
    assert rows[0]["repo_id"] == "OpenMed/test-model-v1-mlx"
    assert rows[0]["formats"] == ["mlx-fp"]
    assert rows[0]["arxiv"] == "2508.01630"
    assert rows[0]["canonical_labels"] == ["PERSON", "DATE"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", rows[0]["reproducibility_hash"])
    assert "secret-token" not in json.dumps(rows[0])


def test_manifest_row_can_record_multiple_runtime_formats(tmp_path):
    artifact = _write_mlx_artifact(tmp_path)

    row = build_manifest_row(
        repo_id="OpenMed/test-model-v1-onnx",
        source_model_id="OpenMed/test-model",
        artifact_dir=artifact,
        format_name="onnx",
        formats=["onnx", "webgpu", "onnx"],
        git_sha="abc123",
    )

    assert row["formats"] == ["onnx", "webgpu"]


def test_publish_artifact_skips_existing_repo_without_upload(tmp_path, monkeypatch):
    artifact = _write_mlx_artifact(tmp_path)
    fake_api = FakeApi(exists=True)
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    result = publish_artifact(
        artifact_dir=artifact,
        source_model_id="OpenMed/test-model",
        format_name="coreml",
        api=fake_api,
    )

    assert result.skipped is True
    assert fake_api.created == []
    assert fake_api.uploaded == []
    assert (artifact / "README.md").exists()
    assert result.repo_id == "OpenMed/test-model-v1-coreml"


def test_publish_artifact_errors_when_token_is_missing(tmp_path, monkeypatch):
    artifact = _write_mlx_artifact(tmp_path)
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)

    with pytest.raises(HfPublishError, match="HF_WRITE_TOKEN is required"):
        publish_artifact(
            artifact_dir=artifact,
            source_model_id="OpenMed/test-model",
            format_name="mlx-fp",
            api=FakeApi(),
        )


def test_manifest_append_replaces_existing_repo_row(tmp_path):
    manifest = tmp_path / "models.jsonl"
    first = {"repo_id": "OpenMed/model", "formats": ["mlx-fp"]}
    second = {"repo_id": "OpenMed/model", "formats": ["coreml"]}

    append_manifest_row(manifest, first)
    append_manifest_row(manifest, second)

    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [second]


def test_conversion_modules_expose_publish_options():
    from openmed.coreml.convert import convert as convert_coreml
    from openmed.mlx.convert import convert as convert_mlx

    assert "publish_to_hub" in inspect.signature(convert_mlx).parameters
    assert "publish_to_hub" in inspect.signature(convert_coreml).parameters
