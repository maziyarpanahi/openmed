"""Offline acceptance tests for the clinical SLM artifact manifest."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from openmed.models.clinical_slm_manifest import (
    CLINICAL_SLM_MANIFEST_FILENAME,
    ClinicalSLMArtifact,
    ClinicalSLMArtifactDigestMismatchError,
    ClinicalSLMArtifactManifest,
    ClinicalSLMArtifactMissingError,
    ClinicalSLMManifestError,
    ClinicalSLMQuantization,
    load_clinical_slm_manifest,
    validate_clinical_slm_manifest,
    verify_clinical_slm_package,
)


def _write_package(tmp_path: Path) -> tuple[Path, ClinicalSLMArtifactManifest]:
    package = tmp_path / "synthetic-clinical-slm"
    files = {
        "weights/model.safetensors": b"synthetic weights",
        "tokenizer/tokenizer.json": b"synthetic tokenizer",
        "templates/templates.json": b"synthetic templates",
        "quantization/quantization.json": b"synthetic quantization",
    }
    artifacts: list[ClinicalSLMArtifact] = []
    for relative_path, content in files.items():
        local_path = package / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)
        artifacts.append(
            ClinicalSLMArtifact(
                component=relative_path.split("/", 1)[0],
                path=relative_path,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
            )
        )

    manifest = ClinicalSLMArtifactManifest(
        model_id="OpenMed/Synthetic-Clinical-SLM",
        revision="a" * 40,
        components=artifacts,
        quantization=ClinicalSLMQuantization(scheme="int4", bits=4),
        licenses={
            "weights": "Apache-2.0",
            "tokenizer": "MIT",
            "templates": "BSD-3-Clause",
            "quantization": "Apache-2.0",
        },
        supported_tasks=["clinical-ner", "clinical-summarization"],
    )
    package.mkdir(parents=True, exist_ok=True)
    (package / CLINICAL_SLM_MANIFEST_FILENAME).write_text(
        manifest.to_json() + "\n",
        encoding="utf-8",
    )
    return package, manifest


def test_valid_package_is_deterministic_and_verified_offline(tmp_path, monkeypatch):
    package, expected = _write_package(tmp_path)

    def fail_network(*_args, **_kwargs):
        raise AssertionError("clinical SLM verification attempted network access")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    loaded = load_clinical_slm_manifest(package)
    result = verify_clinical_slm_package(package)

    assert loaded is not expected
    assert loaded.to_json() == expected.to_json()
    assert result.verified is True
    assert result.component_count == 4
    assert result.executable_component_count == 4
    assert result.bytes_checked == sum(item.size_bytes for item in expected.components)
    assert result.to_json() == verify_clinical_slm_package(package).to_json()
    assert str(package) not in result.to_json()
    assert "synthetic weights" not in result.to_json()

    payload = loaded.to_dict()
    payload["components"].append(
        {
            "component": "runtime",
            "path": "runtime/runtime.bin",
            "sha256": "0" * 64,
            "size_bytes": 1,
            "executable": True,
        }
    )
    assert loaded.component_count == 4
    assert len(payload["components"]) == 5


def test_incomplete_and_unknown_license_manifests_fail_closed_without_values(
    tmp_path: Path,
) -> None:
    package, manifest = _write_package(tmp_path)
    payload = manifest.to_dict()
    payload["components"] = [
        item for item in payload["components"] if item["component"] != "templates"
    ]

    with pytest.raises(ClinicalSLMManifestError) as missing:
        validate_clinical_slm_manifest(payload)
    assert missing.value.code == "missing_component"
    assert "templates" not in str(missing.value)

    unknown_license = manifest.to_dict()
    unknown_license["licenses"][0]["spdx_id"] = "synthetic-unknown-license"
    with pytest.raises(ClinicalSLMManifestError) as license_error:
        validate_clinical_slm_manifest(unknown_license)
    assert license_error.value.code == "unknown_license"
    assert "synthetic-unknown-license" not in str(license_error.value)
    assert str(package) not in str(license_error.value)


@pytest.mark.parametrize(
    ("field_name", "value", "code"),
    [
        ("revision", "main", "mutable_revision"),
        ("revision", "feature/branch", "mutable_revision"),
        ("offline", False, "offline_required"),
        ("human_review_required", False, "human_review_required"),
    ],
)
def test_mutable_or_unsafe_policy_metadata_is_rejected(
    tmp_path: Path,
    field_name: str,
    value,
    code: str,
) -> None:
    _, manifest = _write_package(tmp_path)
    payload = manifest.to_dict()
    payload[field_name] = value

    with pytest.raises(ClinicalSLMManifestError) as raised:
        validate_clinical_slm_manifest(payload)
    assert raised.value.code == code
    assert "main" not in str(raised.value)


def test_manifest_digest_is_required_on_disk_and_binds_metadata(tmp_path: Path) -> None:
    package, manifest = _write_package(tmp_path)
    payload = manifest.to_dict()
    payload.pop("manifest_digest")
    (package / CLINICAL_SLM_MANIFEST_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ClinicalSLMManifestError) as missing:
        load_clinical_slm_manifest(package)
    assert missing.value.code == "manifest_digest_required"
    assert str(package) not in str(missing.value)

    payload["manifest_digest"] = manifest.manifest_digest
    payload["supported_tasks"] = ["different-task"]
    with pytest.raises(ClinicalSLMManifestError) as stale:
        validate_clinical_slm_manifest(payload)
    assert stale.value.code == "manifest_digest_mismatch"
    assert "different-task" not in str(stale.value)


def test_tampered_and_missing_artifacts_are_rejected_before_loading(
    tmp_path: Path,
) -> None:
    package, _ = _write_package(tmp_path)
    weights = package / "weights/model.safetensors"
    weights.write_bytes(b"corrupted weights")

    with pytest.raises(ClinicalSLMArtifactDigestMismatchError) as mismatch:
        verify_clinical_slm_package(package)
    assert mismatch.value.code == "component_digest_mismatch"
    assert str(weights) not in str(mismatch.value)
    assert "corrupted weights" not in str(mismatch.value)

    weights.unlink()
    with pytest.raises(ClinicalSLMArtifactMissingError) as missing:
        verify_clinical_slm_package(package)
    assert missing.value.code == "component_missing_on_disk"
    assert str(package) not in str(missing.value)


def test_symlinked_artifacts_are_not_accepted(tmp_path: Path) -> None:
    package, _ = _write_package(tmp_path)
    weights = package / "weights/model.safetensors"
    backup = package / "weights/model-copy.safetensors"
    weights.rename(backup)
    try:
        weights.symlink_to(backup.name)
    except OSError:
        pytest.skip("symlinks are unavailable in this environment")

    with pytest.raises(ClinicalSLMManifestError) as raised:
        verify_clinical_slm_package(package)
    assert raised.value.code == "unsafe_component_path"
    assert str(backup) not in str(raised.value)


def test_duplicate_json_fields_and_hostile_mapping_values_do_not_escape(
    tmp_path: Path,
) -> None:
    _, manifest = _write_package(tmp_path)
    encoded = json.dumps(manifest.to_dict())
    duplicate = encoded[:-1] + ',"model_id":"synthetic-sensitive-value"}'
    with pytest.raises(ClinicalSLMManifestError) as raised:
        ClinicalSLMArtifactManifest.from_json(duplicate)
    assert raised.value.code == "duplicate_field"
    assert "synthetic-sensitive-value" not in str(raised.value)

    marker = "synthetic-pathlike-value"

    class FailingPath:
        def __fspath__(self):
            raise RuntimeError(marker)

    with pytest.raises(ClinicalSLMManifestError) as path_error:
        load_clinical_slm_manifest(FailingPath())  # type: ignore[arg-type]
    assert marker not in str(path_error.value)
