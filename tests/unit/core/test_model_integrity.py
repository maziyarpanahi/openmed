"""Tests for cached model and signed manifest integrity verification."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from openmed.core.config import OpenMedConfig
from openmed.core.model_integrity import (
    ARTIFACT_MANIFEST_FILENAME,
    ARTIFACT_MANIFEST_SCHEMA,
    ModelIntegrityError,
    prepare_model_reference,
    sha256_file,
    verify_artifact_manifest,
    verify_manifest_signature,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "model_integrity"


def _write_integrity_manifest(
    model_dir: Path,
    *,
    model_id: str = "OpenMed/integrity-fixture",
) -> Path:
    artifact = model_dir / "model.safetensors"
    sidecar = model_dir / ARTIFACT_MANIFEST_FILENAME
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_MANIFEST_SCHEMA,
                "model_id": model_id,
                "reproducibility_hash": "sha256:" + "1" * 64,
                "artifact_root": ".",
                "artifacts": [
                    {
                        "path": artifact.name,
                        "sha256": sha256_file(artifact),
                        "size": artifact.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def test_artifact_manifest_rejects_one_byte_tamper_with_both_hashes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"trusted model bytes")
    sidecar = _write_integrity_manifest(tmp_path)

    clean = verify_artifact_manifest(sidecar)
    assert clean.expected_sha256 == clean.actual_sha256

    expected = sha256_file(artifact)
    artifact.write_bytes(b"Trusted model bytes")
    actual = sha256_file(artifact)
    with pytest.raises(ModelIntegrityError) as raised:
        verify_artifact_manifest(sidecar)

    assert raised.value.expected_sha256 == expected
    assert raised.value.actual_sha256 == actual
    assert expected in str(raised.value)
    assert actual in str(raised.value)


def test_artifact_manifest_rejects_wrong_registry_revision(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"trusted model bytes")
    sidecar = _write_integrity_manifest(tmp_path)

    with pytest.raises(
        ModelIntegrityError,
        match="does not match the registry revision",
    ):
        verify_artifact_manifest(
            sidecar,
            expected_reproducibility_hash="sha256:" + "2" * 64,
        )


def test_artifact_manifest_reports_malformed_sidecar_as_integrity_error(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / ARTIFACT_MANIFEST_FILENAME
    sidecar.write_text("{}", encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="integrity manifest is invalid"):
        verify_artifact_manifest(sidecar)


@patch("openmed.core.models.HF_AVAILABLE", True)
@patch("openmed.core.models.AutoConfig")
def test_load_model_fails_before_pipeline_construction_on_tamper(
    auto_config: Mock,
    tmp_path: Path,
) -> None:
    from openmed.core.models import ModelLoader

    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"trusted model bytes")
    _write_integrity_manifest(tmp_path)
    artifact.write_bytes(b"tampered model bytes")

    loader = ModelLoader(OpenMedConfig(backend="hf"))
    with pytest.raises(ModelIntegrityError):
        loader.load_model(str(tmp_path))

    auto_config.from_pretrained.assert_not_called()


def test_skip_env_logs_prominent_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OPENMED_SKIP_MODEL_VERIFY", "1")
    with caplog.at_level(logging.WARNING):
        result = prepare_model_reference(
            "OpenMed/example",
            registry_info=None,
            cache_dir="unused",
            local_only=True,
        )
    assert result == "OpenMed/example"
    assert "MODEL INTEGRITY VERIFICATION DISABLED" in caplog.text


def test_strict_mode_rejects_registry_model_without_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENMED_MODEL_VERIFY_STRICT", "1")
    registry_info = SimpleNamespace(
        model_id="OpenMed/missing-hash",
        reproducibility_hash=None,
    )
    with pytest.raises(ModelIntegrityError, match="no valid reproducibility_hash"):
        prepare_model_reference(
            registry_info.model_id,
            registry_info=registry_info,
            cache_dir=tmp_path,
            local_only=True,
        )


@pytest.mark.parametrize("weight_storage", ("lfs", "git"))
def test_verified_download_pins_remote_metadata_and_rechecks_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    weight_storage: str,
) -> None:
    model_id = "OpenMed/download-fixture"
    revision = "abc123"
    released = "2026-07-14"
    weights = b"verified model weights"
    config = b'{"model_type":"bert"}\n'
    weight_hash = hashlib.sha256(weights).hexdigest()
    weight_blob = hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(weights)}\0".encode("ascii") + weights,
        usedforsecurity=False,
    ).hexdigest()
    config_blob = hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(config)}\0".encode("ascii") + config,
        usedforsecurity=False,
    ).hexdigest()
    siblings = [
        SimpleNamespace(
            rfilename="config.json",
            blob_id=config_blob,
            lfs=None,
        ),
        SimpleNamespace(
            rfilename="model.safetensors",
            blob_id=weight_blob,
            lfs=(
                SimpleNamespace(sha256=weight_hash) if weight_storage == "lfs" else None
            ),
        ),
    ]
    remote = SimpleNamespace(
        id=model_id,
        sha=revision,
        created_at=None,
        last_modified=datetime(2026, 7, 14, tzinfo=timezone.utc),
        siblings=siblings,
    )
    repro_payload = json.dumps(
        {
            "repo_id": model_id,
            "sha": revision,
            "released": released,
            "siblings": sorted(item.rfilename for item in siblings),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    reproducibility_hash = (
        "sha256:" + hashlib.sha256(repro_payload.encode("utf-8")).hexdigest()
    )
    registry_info = SimpleNamespace(
        model_id=model_id,
        reproducibility_hash=reproducibility_hash,
    )
    snapshot = tmp_path / "snapshot"
    calls: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, *_args, **_kwargs):
            return remote

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        snapshot.mkdir()
        (snapshot / "config.json").write_bytes(config)
        (snapshot / "model.safetensors").write_bytes(weights)
        return str(snapshot)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    prepared = prepare_model_reference(
        model_id,
        registry_info=registry_info,
        cache_dir=tmp_path / "cache",
        local_only=False,
    )
    assert prepared == str(snapshot)
    assert calls[0]["revision"] == revision
    assert set(calls[0]["allow_patterns"]) == {
        "config.json",
        "model.safetensors",
    }

    def fail_api(*_args, **_kwargs):
        raise AssertionError("cached verification attempted network access")

    fake_hub.HfApi = fail_api
    assert prepare_model_reference(
        model_id,
        registry_info=registry_info,
        cache_dir=tmp_path / "cache",
        local_only=True,
    ) == str(snapshot)

    (snapshot / "model.safetensors").write_bytes(b"tampered model weights")
    with pytest.raises(ModelIntegrityError) as raised:
        prepare_model_reference(
            model_id,
            registry_info=registry_info,
            cache_dir=tmp_path / "cache",
            local_only=True,
        )
    assert raised.value.expected_sha256 == f"sha256:{weight_hash}"
    assert raised.value.actual_sha256 == sha256_file(snapshot / "model.safetensors")


@pytest.mark.slow
def test_sha256_streaming_500mb_stays_under_two_seconds(tmp_path: Path) -> None:
    artifact = tmp_path / "synthetic-500mb.bin"
    artifact.touch()
    with artifact.open("r+b") as handle:
        handle.truncate(500 * 1024 * 1024)

    started = time.perf_counter()
    digest = sha256_file(artifact)
    elapsed = time.perf_counter() - started

    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + hashlib.sha256().digest_size * 2
    assert elapsed < 2.0, f"streaming 500MB took {elapsed:.3f}s"


def test_sigstore_bundle_rejects_tampered_manifest(tmp_path: Path) -> None:
    source_manifest = FIXTURES / "models.jsonl"
    bundle = FIXTURES / "models.jsonl.sigstore.json"
    manifest = tmp_path / "models.jsonl"
    # Git for Windows may check out this text fixture with CRLF, while the
    # detached test bundle signs the canonical LF payload.
    manifest.write_bytes(source_manifest.read_bytes().replace(b"\r\n", b"\n"))
    verified = verify_manifest_signature(manifest, bundle)
    assert verified == sha256_file(manifest)

    tampered = tmp_path / "tampered-models.jsonl"
    tampered.write_bytes(manifest.read_bytes().replace(b"fixture", b"tampered"))
    with pytest.raises(ModelIntegrityError) as raised:
        verify_manifest_signature(tampered, bundle)
    assert raised.value.expected_sha256 != raised.value.actual_sha256
