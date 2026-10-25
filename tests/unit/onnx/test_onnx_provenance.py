"""Offline producer provenance with synthetic checkpoint and graph bytes."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def _fixture_source(root: Path) -> Path:
    root.mkdir(parents=True)
    config = {
        "model_type": "bert",
        "id2label": {"0": "O"},
        "license": "MIT",
        "_name_or_path": str(root),
        "max_position_embeddings": 16,
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text('{"fixture": true}')
    (root / "model.safetensors").write_bytes(b"synthetic-weights-A")
    return root


def _install_backends(monkeypatch, module):
    loads = []

    class AutoConfig:
        @staticmethod
        def from_pretrained(reference, **kwargs):
            loads.append(("config", reference))
            value = json.loads((Path(reference) / "config.json").read_text())
            return types.SimpleNamespace(to_dict=lambda: value)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(reference, **kwargs):
            loads.append(("tokenizer", reference))

            def save(directory):
                (Path(directory) / "tokenizer.json").write_bytes(
                    (Path(reference) / "tokenizer.json").read_bytes()
                )

            return types.SimpleNamespace(save_pretrained=save)

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = AutoConfig
    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    def export(reference, output_path, **kwargs):
        loads.append(("model", reference))
        path = Path(output_path)
        path.write_bytes(
            b"mock-graph:" + (Path(reference) / "model.safetensors").read_bytes()
        )
        path.with_suffix(".onnx.data").write_bytes(b"mock external tensor")
        return path

    monkeypatch.setattr(module, "export_onnx", export)
    return loads


def test_local_conversion_is_deterministic_private_and_content_addressed(
    tmp_path, monkeypatch
):
    module = importlib.import_module("openmed.onnx.convert")
    source = _fixture_source(tmp_path / "synthetic-user" / "checkpoint")
    _install_backends(monkeypatch, module)

    def convert(name):
        return module.convert(
            str(source),
            tmp_path / name,
            include_webgpu=False,
            include_int8=False,
            optimize_onnx=False,
        )

    first, second = convert("run-a"), convert("run-b")
    first_bytes = first.manifest_path.read_bytes()
    assert first_bytes == second.manifest_path.read_bytes()
    repeated = convert("run-a")
    assert first_bytes == repeated.manifest_path.read_bytes()
    manifest = json.loads(first_bytes)
    assert manifest["source_model_id"].startswith("local:sha256:")
    assert manifest["provenance"]["source"]["license"] == "MIT"
    assert manifest["provenance"]["source"]["content_digest"].startswith("sha256:")
    assert {f["path"] for f in manifest["provenance"]["files"]} >= {
        "model.onnx",
        "model.onnx.data",
        "config.json",
        "tokenizer.json",
    }
    assert str(tmp_path) not in first_bytes.decode()
    assert str(tmp_path) not in (first.output_dir / "config.json").read_text()
    (source / "model.safetensors").write_bytes(b"synthetic-weights-B")
    changed = convert("run-c")
    assert (
        json.loads(changed.manifest_path.read_bytes())["provenance"]["source"][
            "content_digest"
        ]
        != manifest["provenance"]["source"]["content_digest"]
    )


@pytest.mark.parametrize("mutation", ["modify", "add", "delete", "sidecar"])
def test_explicit_verifier_checks_whole_onnx_output(tmp_path, mutation):
    module = importlib.import_module("openmed.onnx.convert")
    (tmp_path / "model.onnx").write_bytes(b"graph")
    (tmp_path / "model.onnx.data").write_bytes(b"tensor")
    module.write_export_manifest(
        tmp_path,
        source_model_id="fixture/tiny",
        config={"model_type": "bert"},
        artifacts=[module.ExportArtifact("onnx", tmp_path / "model.onnx", "float32")],
    )
    from openmed.core.export_provenance import verify_export_provenance

    verify_export_provenance(tmp_path, "openmed-onnx.json")
    if mutation == "modify":
        (tmp_path / "model.onnx").write_bytes(b"Graph")
    elif mutation == "add":
        (tmp_path / "extra.bin").write_bytes(b"extra")
    elif mutation == "delete":
        (tmp_path / "model.onnx").unlink()
    else:
        (tmp_path / "model.onnx.data").write_bytes(b"Tensor")
    with pytest.raises(ValueError):
        verify_export_provenance(tmp_path, "openmed-onnx.json")


def test_variable_latency_is_retained_outside_canonical_manifest(tmp_path):
    module = importlib.import_module("openmed.onnx.convert")
    payloads = []
    for name, latency in [("a", 1.0), ("b", 2.0)]:
        root = tmp_path / name
        root.mkdir()
        (root / "model.onnx").write_bytes(b"graph")
        path = module.write_export_manifest(
            root,
            source_model_id="fixture/tiny",
            config={"model_type": "bert"},
            artifacts=[module.ExportArtifact("onnx", root / "model.onnx", "float32")],
            validation={
                "passed": True,
                "latency": {"baseline_ms": latency * 2, "optimized_ms": latency},
            },
        )
        payloads.append(path.read_bytes())
        observation = json.loads(
            (tmp_path / f"{name}_onnx_observations.json").read_text()
        )
        assert observation["validation"]["latency"]["optimized_ms"] == latency
    assert payloads[0] == payloads[1]


def test_resolved_snapshot_is_shared_by_model_config_and_tokenizer(
    tmp_path, monkeypatch
):
    module = importlib.import_module("openmed.onnx.convert")
    snapshot = _fixture_source(tmp_path / ("a" * 40))
    calls = []
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kw: calls.append(kw) or str(snapshot),
    )
    loads = _install_backends(monkeypatch, module)
    result = module.convert(
        "fixture/tiny",
        tmp_path / "output",
        include_webgpu=False,
        include_int8=False,
        optimize_onnx=False,
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert len(calls) == 1
    assert {ref for _, ref in loads} == {str(snapshot)}
    assert manifest["provenance"]["source"]["revision"] == "a" * 40
    assert manifest["source_model_id"] == "fixture/tiny"


def test_local_source_mutation_during_export_is_rejected(tmp_path, monkeypatch):
    module = importlib.import_module("openmed.onnx.convert")
    source = _fixture_source(tmp_path / "checkpoint")
    _install_backends(monkeypatch, module)
    original_export = module.export_onnx

    def mutating_export(reference, output_path, **kwargs):
        result = original_export(reference, output_path, **kwargs)
        (source / "model.safetensors").write_bytes(b"synthetic-weights-B")
        return result

    monkeypatch.setattr(module, "export_onnx", mutating_export)
    with pytest.raises(ValueError, match="local source changed"):
        module.convert(
            str(source),
            tmp_path / "output",
            include_webgpu=False,
            include_int8=False,
            optimize_onnx=False,
        )
    assert not (tmp_path / "output" / "openmed-onnx.json").exists()


def test_current_directory_output_keeps_observations_outside_inventory(
    tmp_path, monkeypatch
):
    module = importlib.import_module("openmed.onnx.convert")
    root = tmp_path / "artifact"
    root.mkdir()
    monkeypatch.chdir(root)
    Path("model.onnx").write_bytes(b"graph")
    manifest = module.write_export_manifest(
        ".",
        source_model_id="fixture/tiny",
        config={"model_type": "bert"},
        artifacts=[module.ExportArtifact("onnx", Path("model.onnx"), "float32")],
        validation={"passed": True, "latency": {"optimized_ms": 1.0}},
    )
    assert (
        json.loads((tmp_path / "artifact_onnx_observations.json").read_text())[
            "validation"
        ]["latency"]["optimized_ms"]
        == 1.0
    )
    from openmed.core.export_provenance import verify_export_provenance

    assert verify_export_provenance(root, manifest.name)
    assert not list(root.glob("*observations*"))
