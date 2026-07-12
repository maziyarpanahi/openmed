"""Tests for ONNX and quantized Hugging Face publishing."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.core import hf_publish
from openmed.core.hf_publish import publish_artifact
from openmed.core.manifest_schema import validate_manifest_row


class RepositoryNotFoundError(Exception):
    pass


class FakeApi:
    def __init__(self, *, exists: bool = False) -> None:
        self.exists = exists
        self.created: list[dict[str, object]] = []
        self.uploaded: list[dict[str, object]] = []
        self.uploaded_cards: list[str] = []
        self.info_calls: list[dict[str, object]] = []

    def repo_info(self, **kwargs: object) -> object:
        self.info_calls.append(kwargs)
        if not self.exists:
            raise RepositoryNotFoundError("not found")
        return object()

    def create_repo(self, **kwargs: object) -> None:
        self.created.append(kwargs)
        self.exists = True

    def upload_folder(self, **kwargs: object) -> None:
        self.uploaded.append(kwargs)
        self.uploaded_cards.append(
            (Path(str(kwargs["folder_path"])) / "README.md").read_text(encoding="utf-8")
        )


def test_publish_onnx_artifact_targets_onnx_repo_and_updates_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = _write_onnx_artifact(tmp_path)
    manifest = tmp_path / "models.jsonl"
    fake_api = FakeApi()
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    result = publish_artifact(
        artifact_dir=artifact,
        source_model_id="OpenMed/test-model",
        format_name="onnx",
        formats=["onnx", "webgpu"],
        manifest_path=manifest,
        api=fake_api,
        released="2026-06-27",
        git_sha="abc123",
    )

    assert result.repo_id == "OpenMed/test-model-v1-onnx"
    assert result.manifest_row["formats"] == ["onnx", "webgpu"]
    assert fake_api.created[0]["repo_id"] == "OpenMed/test-model-v1-onnx"
    assert fake_api.uploaded[0]["repo_id"] == "OpenMed/test-model-v1-onnx"
    assert "## Artifact Format" in fake_api.uploaded_cards[0]
    assert "| Runtime artifacts | onnx, webgpu |" in fake_api.uploaded_cards[0]
    assert "secret-token" not in fake_api.uploaded_cards[0]

    rows = _manifest_rows(manifest)
    assert rows == [result.manifest_row]
    assert validate_manifest_row(rows[0], line_number=1) == []


def test_publish_quantized_artifact_merges_format_without_duplicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = _write_onnx_artifact(tmp_path)
    manifest = tmp_path / "models.jsonl"
    existing = {
        "repo_id": "OpenMed/test-model-v1-onnx-int8",
        "formats": ["onnx"],
    }
    manifest.write_text(json.dumps(existing) + "\n", encoding="utf-8")
    fake_api = FakeApi()
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    result = publish_artifact(
        artifact_dir=artifact,
        source_model_id="OpenMed/test-model",
        format_name="int8",
        formats=["onnx", "int8", "int8"],
        manifest_path=manifest,
        api=fake_api,
        released="2026-06-27",
        git_sha="abc123",
    )

    assert result.repo_id == "OpenMed/test-model-v1-onnx-int8"
    assert result.manifest_row["formats"] == ["onnx", "int8"]
    assert "| Quantization | int8 |" in fake_api.uploaded_cards[0]
    rows = _manifest_rows(manifest)
    assert rows[0]["formats"] == ["onnx", "int8"]
    assert rows[0]["formats"].count("int8") == 1
    assert validate_manifest_row(rows[0], line_number=1) == []


def test_publish_android_onnx_artifact_renders_runtime_formats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = _write_onnx_artifact(tmp_path)
    manifest = tmp_path / "models.jsonl"
    fake_api = FakeApi()
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    result = publish_artifact(
        artifact_dir=artifact,
        source_model_id="OpenMed/test-model",
        format_name="onnx-android",
        formats=["onnx-android", "onnx-int8", "ort-android"],
        manifest_path=manifest,
        api=fake_api,
        released="2026-07-09",
        git_sha="abc123",
        private=True,
    )

    assert result.repo_id == "OpenMed/test-model-v1-onnx-android"
    assert result.manifest_row["formats"] == [
        "onnx-android",
        "int8",
        "ort-android",
    ]
    assert fake_api.created[0]["private"] is True
    card = fake_api.uploaded_cards[0]
    assert "## Included Artifacts" in card
    assert "`model_int8.onnx`" in card
    assert "`model_fp16.onnx`" in card
    assert "`model.ort`" in card
    assert 'implementation("com.github.maziyarpanahi:openmed:v1.8.2")' in card
    assert "Reproducibility hash" not in card
    rows = _manifest_rows(manifest)
    assert validate_manifest_row(rows[0], line_number=1) == []


def test_publish_onnx_rerun_skips_existing_repo_without_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = _write_onnx_artifact(tmp_path)
    fake_api = FakeApi(exists=True)
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    result = publish_artifact(
        artifact_dir=artifact,
        source_model_id="OpenMed/test-model",
        format_name="onnx",
        api=fake_api,
        released="2026-06-27",
        git_sha="abc123",
    )

    assert result.repo_id == "OpenMed/test-model-v1-onnx"
    assert result.skipped is True
    assert fake_api.info_calls == [
        {
            "repo_id": "OpenMed/test-model-v1-onnx",
            "repo_type": "model",
            "token": "secret-token",
        }
    ]
    assert fake_api.created == []
    assert fake_api.uploaded == []


def test_publish_overwrite_existing_repo_avoids_creation_endpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = _write_onnx_artifact(tmp_path)
    fake_api = FakeApi(exists=True)
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    result = publish_artifact(
        artifact_dir=artifact,
        source_model_id="OpenMed/test-model",
        format_name="onnx",
        api=fake_api,
        skip_existing=False,
        released="2026-06-27",
        git_sha="abc123",
    )

    assert result.skipped is False
    assert fake_api.created == []
    assert len(fake_api.uploaded) == 1


def test_publish_token_never_appears_in_logs_or_cli_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
    caplog,
) -> None:
    artifact = _write_onnx_artifact(tmp_path)
    fake_api = FakeApi()
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")
    monkeypatch.setattr(hf_publish, "_load_hf_api", lambda: fake_api)

    hf_publish.main(
        [
            "--model",
            "OpenMed/test-model",
            "--artifact-dir",
            str(artifact),
            "--format",
            "onnx",
            "--formats",
            "onnx,webgpu",
            "--manifest",
            str(tmp_path / "models.jsonl"),
            "--git-sha",
            "abc123",
        ]
    )

    captured = capsys.readouterr()
    assert "secret-token" not in captured.out
    assert "secret-token" not in captured.err
    assert "secret-token" not in caplog.text


def _write_onnx_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "id2label": {"0": "O", "1": "B-PERSON", "2": "I-DATE"},
            }
        ),
        encoding="utf-8",
    )
    (artifact / "openmed-onnx.json").write_text(
        json.dumps(
            {
                "format": "openmed-onnx",
                "formats": ["onnx", "webgpu"],
                "source_model_id": "OpenMed/test-model",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "model.onnx").write_bytes(b"onnx")
    return artifact


def _manifest_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
