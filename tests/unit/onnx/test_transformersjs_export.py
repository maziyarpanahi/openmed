"""Tests for Transformers.js ONNX bundle export."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
NODE_SMOKE = ROOT / "tests" / "fixtures" / "onnx" / "transformersjs_smoke.mjs"


def _module():
    return importlib.import_module("openmed.onnx.transformersjs")


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def test_export_transformersjs_bundle_writes_layout_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    source_dir = _write_source_export(tmp_path / "onnx-export")
    _install_fake_onnx_stack(monkeypatch)

    result = module.export_transformersjs_bundle(source_dir)

    assert result.output_dir == source_dir / "transformersjs"
    assert result.files == module.REQUIRED_BUNDLE_FILES
    assert (result.output_dir / "onnx" / "model.onnx").read_bytes() == b"onnx"
    assert (result.output_dir / "onnx" / "model_quantized.onnx").read_bytes() == (
        b"quantized"
    )
    assert (result.output_dir / "tokenizer.json").exists()
    assert (result.output_dir / "tokenizer_config.json").exists()
    config = json.loads((result.output_dir / "config.json").read_text())
    assert config["id2label"] == {"0": "O", "1": "B-NAME", "2": "I-NAME"}
    assert config["label2id"] == {"O": 0, "B-NAME": 1, "I-NAME": 2}

    assert result.contract["inputs"] == [
        {
            "name": "input_ids",
            "axes": ["batch", "sequence"],
            "shape": [
                {"kind": "dynamic", "name": "batch"},
                {"kind": "dynamic", "name": "sequence"},
            ],
        },
        {
            "name": "attention_mask",
            "axes": ["batch", "sequence"],
            "shape": [
                {"kind": "dynamic", "name": "batch"},
                {"kind": "dynamic", "name": "sequence"},
            ],
        },
    ]
    assert result.contract["outputs"][0]["name"] == "logits"

    manifest = json.loads((source_dir / "openmed-onnx.json").read_text())
    assert manifest["formats"] == ["onnx", "transformersjs"]
    assert manifest["artifacts"][-1] == {
        "format": "transformersjs",
        "path": "transformersjs",
        "precision": "int8",
    }


def test_validate_transformersjs_contract_rejects_static_sequence_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _install_fake_onnx_stack(monkeypatch, sequence_dynamic=False)

    with pytest.raises(ValueError, match="axis sequence must be dynamic"):
        module.validate_transformersjs_contract(model_path)


def test_validate_transformersjs_contract_rejects_old_opset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _install_fake_onnx_stack(monkeypatch, opset=17)

    with pytest.raises(ValueError, match="requires opset >= 18"):
        module.validate_transformersjs_contract(model_path)


def test_convert_can_include_transformersjs_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    publish_calls = []

    def fake_export_onnx(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"onnx")
        return path

    def fake_export_webgpu(onnx_path, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"webgpu")
        return path

    def fake_save_source_assets(model_id, output_dir, **kwargs):
        output_dir = Path(output_dir)
        (output_dir / "config.json").write_text(
            json.dumps({"model_type": "bert", "id2label": {"0": "O"}}),
            encoding="utf-8",
        )
        (output_dir / "id2label.json").write_text('{"0": "O"}', encoding="utf-8")
        (output_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return {"model_type": "bert", "id2label": {"0": "O"}}, [
            "tokenizer.json",
            "tokenizer_config.json",
        ]

    def fake_export_transformersjs_bundle(onnx_export_dir, output_dir, **kwargs):
        bundle_dir = Path(output_dir)
        bundle_dir.mkdir(parents=True)
        return types.SimpleNamespace(output_dir=bundle_dir)

    def fake_publish_artifact(**kwargs):
        publish_calls.append(kwargs)
        return types.SimpleNamespace(
            skipped=False,
            repo_id="OpenMed/test-model-v1-onnx",
        )

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "export_webgpu", fake_export_webgpu)
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(
        module,
        "export_transformersjs_bundle",
        fake_export_transformersjs_bundle,
    )
    monkeypatch.setattr(module, "publish_artifact", fake_publish_artifact)

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        include_transformersjs=True,
        optimize_onnx=False,
        publish_to_hub=True,
    )

    assert result.formats == ["onnx", "webgpu", "transformersjs"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["formats"] == ["onnx", "webgpu", "transformersjs"]
    assert manifest["artifacts"][-1] == {
        "format": "transformersjs",
        "path": "transformersjs",
        "precision": "int8",
    }
    assert publish_calls[0]["formats"] == ["onnx", "webgpu", "transformersjs"]


def test_manifest_schema_accepts_transformersjs_format() -> None:
    from openmed.core.manifest_schema import validate_manifest_row

    row = {
        "repo_id": "OpenMed/test-model-v1-transformersjs",
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Tiny",
        "param_count": 1,
        "architecture": "bert",
        "base_model": "OpenMed/test-model",
        "formats": ["transformersjs"],
        "canonical_labels": ["O"],
        "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
        "arxiv": None,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:" + "0" * 64,
        "released": "2026-06-28",
    }

    assert validate_manifest_row(row, line_number=1) == []


def test_committed_node_smoke_fixture_validates_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")

    module = _module()
    source_dir = _write_source_export(tmp_path / "onnx-export")
    _install_fake_onnx_stack(monkeypatch)
    result = module.export_transformersjs_bundle(source_dir)

    completed = subprocess.run(
        [node, str(NODE_SMOKE), str(result.output_dir)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "contract ok" in completed.stdout


def _write_source_export(source_dir: Path) -> Path:
    source_dir.mkdir(parents=True)
    (source_dir / "model.onnx").write_bytes(b"onnx")
    (source_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "max_position_embeddings": 128,
                "id2label": {"0": "O", "1": "B-NAME", "2": "I-NAME"},
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "openmed-onnx.json").write_text(
        json.dumps(
            {
                "format": "openmed-onnx",
                "format_version": 1,
                "formats": ["onnx"],
                "artifacts": [
                    {"format": "onnx", "path": "model.onnx", "precision": "float32"}
                ],
            }
        ),
        encoding="utf-8",
    )
    return source_dir


def _install_fake_onnx_stack(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sequence_dynamic: bool = True,
    opset: int = 18,
) -> None:
    onnx_mod = types.ModuleType("onnx")
    onnx_mod.load = lambda path: _fake_onnx_model(
        sequence_dynamic=sequence_dynamic,
        opset=opset,
    )

    runtime_mod = types.ModuleType("onnxruntime")
    quantization_mod = types.ModuleType("onnxruntime.quantization")
    quantization_mod.QuantType = types.SimpleNamespace(QInt8="qint8")

    def fake_quantize_dynamic(input_model, output_model, *, weight_type):
        assert weight_type == "qint8"
        assert Path(input_model).name == "model.onnx"
        Path(output_model).write_bytes(b"quantized")

    quantization_mod.quantize_dynamic = fake_quantize_dynamic

    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quantization_mod)


def _fake_onnx_model(*, sequence_dynamic: bool = True, opset: int = 18):
    sequence_dim = _dim("sequence") if sequence_dynamic else _dim(value=512)
    return types.SimpleNamespace(
        opset_import=[types.SimpleNamespace(domain="", version=opset)],
        graph=types.SimpleNamespace(
            initializer=[],
            input=[
                _value_info("input_ids", [_dim("batch"), sequence_dim]),
                _value_info("attention_mask", [_dim("batch"), sequence_dim]),
            ],
            output=[
                _value_info(
                    "logits",
                    [_dim("batch"), sequence_dim, _dim(value=3)],
                )
            ],
        ),
    )


def _value_info(name: str, dims):
    return types.SimpleNamespace(
        name=name,
        type=types.SimpleNamespace(
            tensor_type=types.SimpleNamespace(
                shape=types.SimpleNamespace(dim=dims),
            ),
        ),
    )


def _dim(name: str = "", *, value: int = 0):
    return types.SimpleNamespace(dim_param=name, dim_value=value)
