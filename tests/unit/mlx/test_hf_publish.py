"""Tests for publishing converted model artifacts."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from openmed.core.hf_publish import (
    HfPublishError,
    _parse_json_object_arg,
    append_manifest_row,
    build_manifest_row,
    publish_artifact,
    target_repo_id,
)
from openmed.core.repro_hash import build_training_provenance, write_training_provenance
from openmed.eval.release_gates import RELEASABLE, GateCheck, GateReport


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


def test_publish_artifact_blocks_stale_gate_card_before_hf_write(
    tmp_path,
    monkeypatch,
):
    artifact = _write_mlx_artifact(tmp_path)
    (artifact / "README.md").write_text("# stale card\n", encoding="utf-8")
    fake_api = FakeApi()
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    with pytest.raises(HfPublishError, match="stale model card"):
        publish_artifact(
            artifact_dir=artifact,
            source_model_id="OpenMed/OpenMed-PII-Tiny-44M",
            format_name="mlx-fp",
            api=fake_api,
            gate_report_path=_write_json(tmp_path / "gate-report.json", _gate_report()),
            training_provenance_path=_write_training_provenance(tmp_path),
        )

    assert fake_api.created == []
    assert fake_api.uploaded == []


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


def test_manifest_append_replaces_existing_repo_row_and_merges_formats(tmp_path):
    manifest = tmp_path / "models.jsonl"
    first = {"repo_id": "OpenMed/model", "formats": ["mlx-fp"]}
    second = {"repo_id": "OpenMed/model", "formats": ["coreml"]}

    append_manifest_row(manifest, first)
    append_manifest_row(manifest, second)

    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [{"repo_id": "OpenMed/model", "formats": ["mlx-fp", "coreml"]}]


def test_manifest_append_skips_malformed_json_without_logging_row(tmp_path, caplog):
    manifest = tmp_path / "models.jsonl"
    secret = "patient-name-should-not-be-logged"
    manifest.write_text(
        f'{{"repo_id":"OpenMed/old","formats":["mlx-fp"]}}\n{secret}\n',
        encoding="utf-8",
    )

    append_manifest_row(manifest, {"repo_id": "OpenMed/new", "formats": ["coreml"]})

    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {"repo_id": "OpenMed/old", "formats": ["mlx-fp"]},
        {"repo_id": "OpenMed/new", "formats": ["coreml"]},
    ]
    assert "Skipping malformed JSONL line 2" in caplog.text
    assert secret not in caplog.text


def test_parse_json_object_arg_rejects_invalid_json():
    with pytest.raises(HfPublishError, match="not valid JSON"):
        _parse_json_object_arg('{"micro_f1":')


def test_conversion_modules_expose_publish_options():
    from openmed.coreml.convert import convert as convert_coreml
    from openmed.mlx.convert import convert as convert_mlx

    assert "publish_to_hub" in inspect.signature(convert_mlx).parameters
    assert "publish_to_hub" in inspect.signature(convert_coreml).parameters


def _gate_report() -> dict[str, object]:
    report = GateReport(
        repo_id="OpenMed/OpenMed-PII-Tiny-44M-v1-mlx",
        family="PII",
        tier="Tiny",
        param_count=44_000_000,
        format="mlx-fp",
        per_label_recall={"PERSON": 0.995, "DATE": 0.992},
        per_label_precision={"PERSON": 0.994, "DATE": 0.991},
        critical_leakage_count=0,
        residual_leakage_rate=0.0,
        quant_recall_delta=0.0,
        p50_ms=32.0,
        p95_ms=81.0,
        ram_mb=128.0,
        eval_set_hash="sha256:eval",
        leakage_fixture_hash="sha256:leakage",
        decision=RELEASABLE,
        gate_results=(
            GateCheck(
                "G5",
                True,
                reason="ok",
                details={
                    "tier": "tiny",
                    "budget": {"ram_mb": 350.0, "p50_ms": 60.0, "p95_ms": 150.0},
                    "violations": {},
                },
            ),
        ),
        policy="hipaa_safe_harbor",
        threshold_profile="strict",
        target_leakage_rate=0.005,
    ).sign("unit-publish-key")
    return report.to_dict()


def _write_training_provenance(tmp_path: Path) -> Path:
    provenance = build_training_provenance(
        rng_seeds={"python": 123},
        data_manifest_hash="sha256:" + "2" * 64,
        recipe_config_hash="sha256:" + "3" * 64,
        env_lock_digest="sha256:" + "4" * 64,
        base_model="OpenMed/OpenMed-PII-Tiny-44M",
        base_model_revision="main",
        git_sha="abc123",
        repo_id="OpenMed/OpenMed-PII-Tiny-44M-v1-mlx",
    )
    return write_training_provenance(tmp_path, provenance)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
