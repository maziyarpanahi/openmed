"""Tests for mapping converter and catalog metadata into portable bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from openmed.core import portable_bundle as pb
from openmed.core import portable_bundle_adapters as adapters
from openmed.coreml.convert import (
    COREML_MANIFEST_FILENAME,
    CoreMLVariantRecord,
    _write_coreml_conversion_manifest,
)
from openmed.mlx.artifact import write_manifest as write_mlx_manifest
from openmed.onnx.convert import ExportArtifact, write_export_manifest
from openmed.processing.tokenization import package_segmenter_resources

ROOT = Path(__file__).resolve().parents[3]
RELEASE = adapters.BundleRelease(
    bundle_id="openmed.synthetic.adapter",
    version="1.0.0",
    sequence=1,
    issued_at="2026-10-01T00:00:00Z",
    expires_at="2027-10-01T00:00:00Z",
)
CATALOG = {
    "architecture": "bert",
    "arxiv": None,
    "base_model": "OpenMed/synthetic-tiny",
    "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
    "canonical_labels": ["NAME"],
    "family": "PII",
    "formats": ["onnx"],
    "languages": ["en"],
    "license": "apache-2.0",
    "param_count": 1,
    "released": "2026-11-01",
    "repo_id": "OpenMed/synthetic-tiny-onnx",
    "reproducibility_hash": "sha256:" + "0" * 64,
    "task": "token-classification",
    "tier": "Tiny",
}
SEED = hashlib.sha256(b"openmed.portable_bundle.TEST-ONLY.adapter-tests").digest()


def _common_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"bert"}\n', encoding="utf-8")
    (root / "id2label.json").write_text(
        '{"0":"O","1":"B-NAME","2":"I-NAME"}\n', encoding="utf-8"
    )
    (root / "tokenizer.json").write_text(
        '{"model":{"type":"WordPiece"}}\n', encoding="utf-8"
    )
    (root / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    (root / "LICENSE").write_text("Synthetic Apache-2.0 fixture.\n", encoding="utf-8")


def _onnx_export(root: Path, *, extra: list[ExportArtifact] = ()) -> Path:
    _common_files(root)
    (root / "model.onnx").write_bytes(b"synthetic fp32 graph")
    (root / "model.onnx_data").write_bytes(b"synthetic external data")
    (root / "model_int8.onnx").write_bytes(b"synthetic int8 graph")
    (root / "model.webgpu.onnx").write_bytes(b"synthetic webgpu graph")
    descriptor = package_segmenter_resources(root, "openmed-han-v1")
    write_export_manifest(
        root,
        source_model_id="OpenMed/synthetic-tiny",
        config={"model_type": "bert", "max_position_embeddings": 128},
        artifacts=[
            ExportArtifact("onnx", root / "model.onnx", "float32"),
            ExportArtifact("onnx-int8", root / "model_int8.onnx", "int8"),
            ExportArtifact("webgpu", root / "model.webgpu.onnx", "float16"),
            *extra,
        ],
        tokenizer_files=["tokenizer.json", "vocab.txt"],
        segmenter=descriptor,
    )
    return root


def _mlx_export(root: Path, bits: int | None = 8) -> Path:
    _common_files(root)
    (root / "weights.safetensors").write_bytes(b"synthetic mlx weights")
    config: dict[str, Any] = {
        "_mlx_family": "bert",
        "_mlx_task": "token-classification",
        "max_position_embeddings": 128,
    }
    if bits is not None:
        config["_mlx_quantization"] = {"bits": bits, "group_size": 64}
    write_mlx_manifest(
        root,
        source_model_id="OpenMed/synthetic-tiny",
        config=config,
        tokenizer_files=["tokenizer.json", "vocab.txt"],
    )
    return root


def _coreml_export(root: Path) -> Path:
    _common_files(root)
    package = root / "OpenMed.mlpackage"
    (package / "Data" / "com.apple.CoreML").mkdir(parents=True)
    (package / "Manifest.json").write_text("{}", encoding="utf-8")
    (package / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(b"graph")
    _write_coreml_conversion_manifest(
        root / COREML_MANIFEST_FILENAME,
        model_id="OpenMed/synthetic-tiny",
        model_type="bert",
        compute_units="all",
        max_seq_length=128,
        optimize_for_ane=True,
        variants=[
            CoreMLVariantRecord(
                name="fp16",
                path="OpenMed.mlpackage",
                precision="float16",
                quantization="none",
                size_bytes=5,
                latency_ms={"p50": 1.5},
                parity={"recall_delta": 0.0},
            )
        ],
    )
    # Add the label map and tokenizer references Core ML manifests omit.
    manifest = json.loads((root / COREML_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["label_map_path"] = "id2label.json"
    manifest["tokenizer"] = {"path": ".", "files": ["tokenizer.json", "vocab.txt"]}
    (root / COREML_MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _sign_and_verify(
    root: Path, draft: adapters.PortableManifestDraft, target: pb.VerificationTarget
) -> pb.PortableBundleVerification:
    manifest = draft.encode()
    envelope = pb.sign_portable_manifest(manifest, [("adapter-test", SEED)])
    pb.write_portable_bundle(root, manifest, envelope)
    policy = pb.TrustPolicy(
        keys=(
            pb.TrustedKey(
                kid="adapter-test",
                public_key=pb.ed25519_public_key(SEED),
                not_before="2026-01-01T00:00:00Z",
                not_after="2028-01-01T00:00:00Z",
            ),
        )
    )
    return pb.verify_portable_bundle_dir(
        root,
        trust_policy=policy,
        context=pb.VerificationContext(now="2026-11-07T00:00:00Z", target=target),
    )


def _losses(draft: adapters.PortableManifestDraft) -> dict[str, str]:
    return {loss.source_field: loss.reason for loss in draft.losses}


def test_onnx_export_maps_signs_and_verifies(tmp_path: Path) -> None:
    root = _onnx_export(tmp_path)
    draft = adapters.manifest_from_onnx(root, release=RELEASE, catalog_entry=CATALOG)
    manifest = draft.manifest
    assert manifest["model"] == {
        "architecture": "bert",
        "family": "pii",
        "id": "OpenMed/synthetic-tiny-onnx",
        "languages": ["en"],
        "revision": None,
        "task": "token-classification",
    }
    runtimes = {item["id"]: item for item in manifest["runtimes"]}
    assert set(runtimes) == {"onnx-fp32", "onnx-int8-int8", "webgpu-fp16"}
    assert runtimes["onnx-fp32"]["files"] == [
        "config.json",
        "model.onnx",
        "model.onnx_data",
    ]
    assert runtimes["webgpu-fp16"]["platforms"] == [
        {"max_os_exclusive": None, "min_os": None, "platform": "web"}
    ]
    assert manifest["labels"] == {"count": 3, "path": "id2label.json", "scheme": "bio"}
    assert manifest["tokenizer"]["kind"] == "wordpiece"
    assert manifest["segmenter"] == {
        "files": ["segmenter/han_words.txt"],
        "id": "openmed-han-v1",
    }
    licenses = {item["id"]: item["files"] for item in manifest["licenses"]}
    assert licenses == {"apache-2.0": ["LICENSE"], "mit": ["segmenter/han_words.txt"]}
    assert manifest["evidence"] == [
        {"kind": "converter-manifest", "path": "openmed-onnx.json"}
    ]
    losses = _losses(draft)
    for name in ("optimization", "validation", "dynamic_shapes", "minimum_opset"):
        assert losses[name] == adapters.LOSS_PRESERVED
    assert losses["catalog.benchmark"] == adapters.LOSS_CATALOG_ONLY
    assert losses["decoder"] == adapters.LOSS_DEFAULTED
    assert losses["model.revision"] == adapters.LOSS_UNKNOWN
    target = pb.VerificationTarget(
        platform="android",
        sdk_name="openmedkit-android",
        sdk_version="4.0.0",
        runtimes=("onnx",),
        os_version="26",
    )
    result = _sign_and_verify(root, draft, target)
    assert result.ok and result.files_verified
    assert [
        manifest["runtimes"][index]["id"] for index in result.eligible_runtimes
    ] == [
        "onnx-fp32",
        "onnx-int8-int8",
    ]


def test_unsupported_onnx_artifacts_fail_or_are_recorded(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    (root / "model.xml").write_bytes(b"synthetic openvino")
    _onnx_export(
        root, extra=[ExportArtifact("openvino-ir", root / "model.xml", "float32")]
    )
    with pytest.raises(adapters.PortableBundleAdapterError) as error:
        adapters.manifest_from_onnx(root, release=RELEASE, catalog_entry=CATALOG)
    assert error.value.reason == "source_format_unsupported"
    draft = adapters.manifest_from_onnx(
        root, release=RELEASE, catalog_entry=CATALOG, skip_unsupported=True
    )
    assert _losses(draft)["artifacts.3.format"] == adapters.LOSS_ARTIFACT_SKIPPED
    assert "openvino" not in json.dumps(draft.manifest["runtimes"])


def test_mlx_artifact_maps_precision_explicitly(tmp_path: Path) -> None:
    quantized = _mlx_export(tmp_path / "q8", bits=8)
    draft = adapters.manifest_from_mlx(
        quantized, release=RELEASE, catalog_entry=CATALOG
    )
    (runtime,) = draft.manifest["runtimes"]
    assert runtime["format"] == "mlx" and runtime["precision"] == "int8"
    assert runtime["entrypoint"] == "weights.safetensors"
    assert _losses(draft)["quantization"] == adapters.LOSS_PRESERVED
    assert _losses(draft)["platforms"] == adapters.LOSS_DEFAULTED
    target = pb.VerificationTarget(
        platform="macos",
        sdk_name="openmedkit-swift",
        sdk_version="4.0.0",
        runtimes=("mlx",),
        os_version="15.0",
    )
    assert _sign_and_verify(quantized, draft, target).ok

    full = _mlx_export(tmp_path / "fp", bits=None)
    with pytest.raises(adapters.PortableBundleAdapterError) as error:
        adapters.manifest_from_mlx(full, release=RELEASE, catalog_entry=CATALOG)
    assert error.value.reason == "precision_ambiguous"
    draft = adapters.manifest_from_mlx(
        full, release=RELEASE, catalog_entry=CATALOG, precision="float32"
    )
    assert draft.manifest["runtimes"][0]["precision"] == "fp32"
    with pytest.raises(adapters.PortableBundleAdapterError) as error:
        adapters.manifest_from_mlx(
            quantized, release=RELEASE, catalog_entry=CATALOG, precision="int4"
        )
    assert error.value.reason == "metadata_conflict"


def test_coreml_conversion_maps_and_preserves_evidence(tmp_path: Path) -> None:
    root = _coreml_export(tmp_path)
    draft = adapters.manifest_from_coreml(root, release=RELEASE, catalog_entry=CATALOG)
    (runtime,) = draft.manifest["runtimes"]
    assert runtime["id"] == "coreml-fp16"
    assert runtime["entrypoint"] == "OpenMed.mlpackage/Manifest.json"
    assert "OpenMed.mlpackage/Data/com.apple.CoreML/model.mlmodel" in runtime["files"]
    assert [item["platform"] for item in runtime["platforms"]] == ["ios", "macos"]
    losses = _losses(draft)
    assert losses["variants.0.latency_ms"] == adapters.LOSS_PRESERVED
    assert losses["variants.0.parity"] == adapters.LOSS_PRESERVED
    assert losses["generated_at"] == adapters.LOSS_PRESERVED
    target = pb.VerificationTarget(
        platform="ios",
        sdk_name="openmedkit-swift",
        sdk_version="4.2.0",
        runtimes=("coreml",),
        os_version="17.4",
    )
    assert _sign_and_verify(root, draft, target).ok


def test_adapters_refuse_to_guess(tmp_path: Path) -> None:
    root = _onnx_export(tmp_path)
    for overrides, reason in (
        ({"license": "other"}, "license_unmapped"),
        ({"license": None}, "license_unmapped"),
        ({"task": "text-generation"}, "metadata_conflict"),
        ({"architecture": "roberta"}, "metadata_conflict"),
        ({"repo_id": "/Users/someone/models/x"}, "source_identity_not_portable"),
        ({"languages": []}, "metadata_missing"),
    ):
        with pytest.raises(adapters.PortableBundleAdapterError) as error:
            adapters.manifest_from_onnx(
                root, release=RELEASE, catalog_entry={**CATALOG, **overrides}
            )
        assert error.value.reason == reason, overrides
        assert str(error.value) == reason
    draft = adapters.manifest_from_onnx(
        root,
        release=RELEASE,
        catalog_entry={**CATALOG, "license": "other"},
        license_id="MIT",
    )
    assert draft.manifest["licenses"][0]["id"] == "mit"
    assert _losses(draft)["catalog.license"] == adapters.LOSS_DEFAULTED
    with pytest.raises(adapters.PortableBundleAdapterError) as error:
        adapters.manifest_from_onnx(root, release=RELEASE)
    assert error.value.reason == "metadata_missing"
    with pytest.raises(adapters.PortableBundleAdapterError) as error:
        adapters.manifest_from_mlx(root, release=RELEASE, catalog_entry=CATALOG)
    assert error.value.reason == "source_manifest_missing"


def test_local_source_paths_never_reach_the_manifest(tmp_path: Path) -> None:
    root = _onnx_export(tmp_path)
    source = json.loads((root / "openmed-onnx.json").read_text(encoding="utf-8"))
    local = str(tmp_path / "private" / "checkpoint")
    source["source_model_id"] = local
    (root / "openmed-onnx.json").write_text(json.dumps(source), encoding="utf-8")
    draft = adapters.manifest_from_onnx(
        root,
        release=RELEASE,
        model={
            "architecture": "bert",
            "family": "pii",
            "id": "local:sha256:" + "a" * 64,
            "languages": ["en"],
            "revision": None,
            "task": "token-classification",
        },
        license_id="apache-2.0",
    )
    assert _losses(draft)["source_model_id"] == adapters.LOSS_UNKNOWN
    assert local not in draft.encode().decode("utf-8")
    assert str(tmp_path) not in draft.encode().decode("utf-8")


def test_export_provenance_records_are_cross_checked(tmp_path: Path) -> None:
    root = _onnx_export(tmp_path)
    files = pb.inventory_files(root)
    provenance = {
        "schema_version": 1,
        "source": {
            "content_digest": "sha256:" + "b" * 64,
            "license": "apache-2.0",
            "model_id": "OpenMed/synthetic-tiny",
            "revision": "c" * 40,
        },
        "files": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
            for item in files
        ],
    }
    draft = adapters.manifest_from_onnx(
        root, release=RELEASE, catalog_entry=CATALOG, export_provenance=provenance
    )
    assert draft.manifest["model"]["revision"] == "c" * 40
    assert draft.manifest["provenance"]["source_digest"] == "sha256:" + "b" * 64
    tampered = json.loads(json.dumps(provenance))
    tampered["files"][0]["sha256"] = "sha256:" + "d" * 64
    with pytest.raises(adapters.PortableBundleAdapterError) as error:
        adapters.manifest_from_onnx(
            root, release=RELEASE, catalog_entry=CATALOG, export_provenance=tampered
        )
    assert error.value.reason == "inventory_invalid"


def test_every_catalog_row_maps_or_fails_with_a_controlled_reason() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "models.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    outcomes: dict[str, int] = {}
    for row in rows:
        try:
            model, license_id, losses = adapters.model_section_from_catalog(row)
        except adapters.PortableBundleAdapterError as error:
            assert error.reason in adapters.ADAPTER_REASONS
            outcomes[error.reason] = outcomes.get(error.reason, 0) + 1
            continue
        outcomes["mapped"] = outcomes.get("mapped", 0) + 1
        assert model["id"] == row["repo_id"]
        assert license_id
        assert {loss.source_field for loss in losses} >= {
            f"catalog.{name}"
            for name in row
            if name
            not in {"repo_id", "family", "task", "languages", "architecture", "license"}
        }
    assert outcomes["mapped"] > len(rows) // 2
    assert set(outcomes) <= {"mapped", *adapters.ADAPTER_REASONS}


def test_exports_without_tokenizer_or_labels_map_to_null_sections(
    tmp_path: Path,
) -> None:
    (tmp_path / "LICENSE").write_text("Synthetic fixture.\n", encoding="utf-8")
    (tmp_path / "model.onnx").write_bytes(b"synthetic graph")
    write_export_manifest(
        tmp_path,
        source_model_id="OpenMed/synthetic-tiny",
        config={"model_type": "bert"},
        artifacts=[ExportArtifact("onnx", tmp_path / "model.onnx", "float32")],
    )
    draft = adapters.manifest_from_onnx(
        tmp_path, release=RELEASE, catalog_entry=CATALOG
    )
    assert draft.manifest["tokenizer"] is None
    assert draft.manifest["labels"] is None
    assert draft.manifest["segmenter"] is None
    assert _losses(draft)["tokenizer"] == adapters.LOSS_UNKNOWN
    pb.PortableBundleManifest.from_bytes(draft.encode())
