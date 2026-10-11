"""Offline acceptance tests for the clinical SLM artifact manifest."""

from __future__ import annotations

import hashlib
import json
import socket
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

import openmed.models.clinical_slm_manifest as manifest_module
from openmed.models.clinical_slm_manifest import (
    CLINICAL_SLM_MANIFEST_FILENAME,
    ClinicalSLMArtifact,
    ClinicalSLMArtifactDigestMismatchError,
    ClinicalSLMArtifactManifest,
    ClinicalSLMArtifactMissingError,
    ClinicalSLMManifestError,
    ClinicalSLMQuantization,
    ClinicalSLMVerificationResult,
    load_clinical_slm_manifest,
    validate_clinical_slm_manifest,
    verify_clinical_slm_package,
)

requires_secure_reads = pytest.mark.skipif(
    not manifest_module._HAS_SECURE_LOCAL_READ,
    reason="package verification requires POSIX no-follow directory descriptors",
)


def test_typed_manifest_is_revalidated_before_verification(tmp_path):
    package, manifest = _write_package(tmp_path)
    object.__setattr__(manifest, "offline", False)
    with pytest.raises(ClinicalSLMManifestError):
        verify_clinical_slm_package(package, manifest)


def test_nested_typed_license_is_revalidated(tmp_path):
    _, manifest = _write_package(tmp_path)
    object.__setattr__(manifest.licenses[0], "spdx_id", "SYNTHETIC_PRIVATE")
    with pytest.raises(ClinicalSLMManifestError):
        replace(manifest, manifest_digest=None)


def test_quantization_bits_must_agree_with_scheme():
    with pytest.raises(ClinicalSLMManifestError):
        ClinicalSLMQuantization("int4", bits=8)


def test_format_aliases_cannot_disagree():
    with pytest.raises(ClinicalSLMManifestError):
        ClinicalSLMArtifact(
            "weights",
            "model.bin",
            "a" * 64,
            1,
            format="safetensors",
            format_name="pickle",
        )


def test_verification_result_reason_codes_are_fixed():
    with pytest.raises(ClinicalSLMManifestError):
        ClinicalSLMVerificationResult(
            True, 4, 4, 10, "a" * 64, reason_codes=("synthetic_private",)
        )


def test_zero_component_verification_is_not_success():
    with pytest.raises(ClinicalSLMManifestError):
        ClinicalSLMVerificationResult(True, 0, 0, 0, "a" * 64)


def test_json_failure_discards_sensitive_context():
    with pytest.raises(ClinicalSLMManifestError) as caught:
        ClinicalSLMArtifactManifest.from_json('{"SYNTHETIC_PRIVATE":')
    assert caught.value.__context__ is None


def test_grouped_component_cannot_override_its_role(tmp_path):
    _, manifest = _write_package(tmp_path)
    payload = manifest.to_dict()
    payload["components"] = {"weights": [payload["components"][0]]}
    assert payload["components"]["weights"][0]["component"] != "weights"
    with pytest.raises(ClinicalSLMManifestError):
        ClinicalSLMArtifactManifest.from_mapping(payload)


def test_collection_read_stops_at_bound(tmp_path):
    _, manifest = _write_package(tmp_path)
    reads = []

    class UnboundedTasks(Sequence):
        def __len__(self):
            return 10**12

        def __getitem__(self, index):
            reads.append(index)
            assert index <= manifest_module.MAX_COMPONENTS
            return f"task-{index}"

    with pytest.raises(ClinicalSLMManifestError):
        replace(manifest, supported_tasks=UnboundedTasks(), manifest_digest=None)
    assert len(reads) == manifest_module.MAX_COMPONENTS + 1


def test_large_json_is_rejected_before_parsing(monkeypatch):
    monkeypatch.setattr(manifest_module, "MAX_MANIFEST_BYTES", 16)
    with pytest.raises(ClinicalSLMManifestError) as caught:
        ClinicalSLMArtifactManifest.from_json(" " * 17)
    assert caught.value.code == "manifest_unreadable"
    assert caught.value.__context__ is None


def test_success_cannot_include_failure_reason():
    with pytest.raises(ClinicalSLMManifestError):
        ClinicalSLMVerificationResult(
            True, 4, 4, 10, "a" * 64, reason_codes=("component_mutated",)
        )


@requires_secure_reads
def test_symlink_swap_between_check_and_open_is_rejected(tmp_path, monkeypatch):
    package, manifest = _write_package(tmp_path)
    weights = package / "weights/model.safetensors"
    backup = package / "weights/backup.safetensors"
    real_open = manifest_module.os.open

    def swap(path, flags, *args, **kwargs):
        if path == "model.safetensors":
            weights.rename(backup)
            weights.symlink_to(backup.name)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(manifest_module.os, "open", swap)
    with pytest.raises(ClinicalSLMManifestError) as caught:
        verify_clinical_slm_package(package, manifest)
    assert caught.value.__context__ is None


@requires_secure_reads
def test_same_size_mutation_during_read_is_rejected(tmp_path, monkeypatch):
    package, manifest = _write_package(tmp_path)
    weights = package / "weights/model.safetensors"
    original_sha256 = manifest_module.hashlib.sha256

    class MutatingDigest:
        def __init__(self):
            self.digest = original_sha256()

        def update(self, chunk):
            self.digest.update(chunk)
            if chunk == b"synthetic weights":
                weights.write_bytes(b"synthetic weights")

        def hexdigest(self):
            return self.digest.hexdigest()

    def digest_factory(*args):
        return original_sha256(*args) if args else MutatingDigest()

    monkeypatch.setattr(manifest_module.hashlib, "sha256", digest_factory)
    with pytest.raises(ClinicalSLMManifestError):
        verify_clinical_slm_package(package, manifest)


@requires_secure_reads
def test_nested_manifest_symlink_is_rejected(tmp_path):
    package, _ = _write_package(tmp_path)
    (package / "alias").symlink_to(package, target_is_directory=True)
    with pytest.raises(ClinicalSLMManifestError):
        verify_clinical_slm_package(
            package, manifest_filename="alias/clinical-slm-manifest.json"
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


@requires_secure_reads
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


@requires_secure_reads
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


@requires_secure_reads
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


def test_platform_without_secure_reads_can_parse_but_cannot_verify(
    tmp_path, monkeypatch
):
    package, manifest = _write_package(tmp_path)
    monkeypatch.setattr(manifest_module, "_HAS_SECURE_LOCAL_READ", False)
    assert load_clinical_slm_manifest(package).to_json() == manifest.to_json()
    with pytest.raises(ClinicalSLMManifestError) as caught:
        verify_clinical_slm_package(package, manifest)
    assert caught.value.code == "component_unreadable"
    assert caught.value.__context__ is None


def test_optional_runtime_metadata_keeps_existing_manifest_bytes(tmp_path):
    _, manifest = _write_package(tmp_path)
    # Captured from the unmodified v3.0 implementation and synthetic fixture.
    assert (
        manifest.manifest_digest
        == "sha256:ff675c64c675624b9341d03a408bcb7b77a11cd3f042232f600ac7144f84ce6d"
    )
    assert "context_limits" not in manifest.to_dict()
    assert "required_runtime_features" not in manifest.to_dict()
    assert (
        ClinicalSLMArtifactManifest.from_json(manifest.to_json()).to_json()
        == manifest.to_json()
    )


def test_runtime_metadata_is_immutable_digest_bound_and_round_trips(tmp_path):
    _, old = _write_package(tmp_path)
    limits = {
        "max_context_tokens": 4096,
        "max_input_tokens": 3072,
        "max_output_tokens": 1024,
    }
    manifest = replace(
        old,
        context_limits=limits,
        required_runtime_features=("mlx",),
        manifest_digest=None,
    )
    limits["max_context_tokens"] = 8192
    assert manifest.context_limits["max_context_tokens"] == 4096
    with pytest.raises(TypeError):
        manifest.context_limits["max_context_tokens"] = 8192
    assert manifest.manifest_digest != old.manifest_digest
    assert (
        ClinicalSLMArtifactManifest.from_json(manifest.to_json()).to_json()
        == manifest.to_json()
    )
    with pytest.raises(ClinicalSLMManifestError) as caught:
        replace(manifest, required_runtime_features=("cpu",))
    assert caught.value.code == "manifest_digest_mismatch"


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {"max_context_tokens": 4096},
        {
            "max_context_tokens": 4096,
            "max_input_tokens": True,
            "max_output_tokens": 1024,
        },
        {"max_context_tokens": 4096, "max_input_tokens": 4096, "max_output_tokens": 1},
        {"max_context_tokens": 2**100, "max_input_tokens": 1, "max_output_tokens": 1},
    ],
)
def test_invalid_context_metadata_has_value_free_refusal(tmp_path, limits):
    _, old = _write_package(tmp_path)
    with pytest.raises(ClinicalSLMManifestError) as caught:
        replace(old, context_limits=limits, manifest_digest=None)
    assert caught.value.code == "invalid_context_limits"
    assert caught.value.__context__ is None


@requires_secure_reads
@pytest.mark.parametrize(
    "extra", ["extra.json", "nested/extra.json", "empty-directory"]
)
def test_strict_package_inventory_rejects_extra_members(tmp_path, extra):
    root, manifest = _write_package(tmp_path)
    extra_path = root / extra
    if extra == "empty-directory":
        extra_path.mkdir()
    else:
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        extra_path.write_text("synthetic private payload")
    with pytest.raises(ClinicalSLMManifestError) as caught:
        verify_clinical_slm_package(
            root,
            expected_manifest_digest=manifest.manifest_digest,
            reject_undeclared_files=True,
        )
    assert caught.value.code == "undeclared_component"
    assert str(root) not in str(caught.value)


@requires_secure_reads
def test_strict_inventory_and_trusted_pin_admit_nested_package(tmp_path):
    root, manifest = _write_package(tmp_path)
    result = verify_clinical_slm_package(
        root,
        expected_manifest_digest=manifest.manifest_digest,
        reject_undeclared_files=True,
    )
    assert result.verified
    with pytest.raises(ClinicalSLMManifestError) as caught:
        verify_clinical_slm_package(
            root, expected_manifest_digest="0" * 64, reject_undeclared_files=True
        )
    assert caught.value.code == "manifest_digest_mismatch"


@requires_secure_reads
def test_strict_verification_binds_persisted_manifest_with_supplied_object(tmp_path):
    root, manifest = _write_package(tmp_path)
    altered = replace(manifest, supported_tasks=("clinical-ner",), manifest_digest=None)
    (root / CLINICAL_SLM_MANIFEST_FILENAME).write_text(altered.to_json())
    with pytest.raises(ClinicalSLMManifestError) as caught:
        verify_clinical_slm_package(
            root,
            manifest,
            expected_manifest_digest=manifest.manifest_digest,
            reject_undeclared_files=True,
        )
    assert caught.value.code == "manifest_digest_mismatch"
