"""Tests for ONNX and WebGPU conversion helpers."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def test_write_export_manifest_records_onnx_and_webgpu(tmp_path: Path) -> None:
    module = _convert_module()
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    (tmp_path / "model.webgpu.onnx").write_bytes(b"webgpu")
    (tmp_path / "id2label.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    manifest_path = module.write_export_manifest(
        tmp_path,
        source_model_id="OpenMed/test-model",
        config={
            "model_type": "bert",
            "max_position_embeddings": 128,
            "id2label": {"0": "O"},
        },
        artifacts=[
            module.ExportArtifact("onnx", tmp_path / "model.onnx", "float32"),
            module.ExportArtifact(
                "webgpu",
                tmp_path / "model.webgpu.onnx",
                "float16",
            ),
        ],
        tokenizer_files=["tokenizer.json"],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "openmed-onnx"
    assert manifest["formats"] == ["onnx", "webgpu"]
    assert manifest["source_model_id"] == "OpenMed/test-model"
    assert manifest["label_map_path"] == "id2label.json"
    assert manifest["artifacts"] == [
        {"format": "onnx", "path": "model.onnx", "precision": "float32"},
        {
            "format": "webgpu",
            "path": "model.webgpu.onnx",
            "precision": "float16",
        },
    ]
    assert manifest["tokenizer"]["files"] == ["tokenizer.json"]


def test_convert_orchestrates_artifacts_and_multi_format_publish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _convert_module()
    publish_calls = []

    def fake_export_onnx(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"onnx")
        return path

    def fake_export_webgpu(onnx_path, output_path, **kwargs):
        assert Path(onnx_path).name == "model.onnx"
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
        return {"model_type": "bert", "id2label": {"0": "O"}}, []

    def fake_publish_artifact(**kwargs):
        publish_calls.append(kwargs)
        return types.SimpleNamespace(skipped=False, repo_id="OpenMed/test-model-v1-onnx")

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "export_webgpu", fake_export_webgpu)
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(module, "publish_artifact", fake_publish_artifact)

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        publish_to_hub=True,
        publish_manifest_path=tmp_path / "models.jsonl",
    )

    assert result.formats == ["onnx", "webgpu"]
    assert (result.output_dir / "model.onnx").exists()
    assert (result.output_dir / "model.webgpu.onnx").exists()
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))["formats"] == [
        "onnx",
        "webgpu",
    ]
    assert publish_calls[0]["format_name"] == "onnx"
    assert publish_calls[0]["formats"] == ["onnx", "webgpu"]
    assert publish_calls[0]["manifest_path"] == tmp_path / "models.jsonl"


def test_export_webgpu_converts_to_fp16_and_preserves_io_types(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _convert_module()
    input_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.webgpu.onnx"
    input_path.write_bytes(b"onnx")
    saved = {}
    checked = []

    onnx_mod = types.ModuleType("onnx")

    def fake_load(path):
        return {"path": path}

    def fake_save(model, path):
        saved["model"] = model
        saved["path"] = path
        Path(path).write_bytes(b"fp16")

    onnx_mod.load = fake_load
    onnx_mod.save = fake_save
    onnx_mod.checker = types.SimpleNamespace(
        check_model=lambda model: checked.append(model)
    )

    runtime_mod = types.ModuleType("onnxruntime")
    transformers_mod = types.ModuleType("onnxruntime.transformers")
    float16_mod = types.ModuleType("onnxruntime.transformers.float16")

    def fake_convert(model, keep_io_types):
        return {"fp16": model, "keep_io_types": keep_io_types}

    float16_mod.convert_float_to_float16 = fake_convert

    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime.transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime.transformers.float16", float16_mod)

    result = module.export_webgpu(input_path, output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"fp16"
    assert saved["model"]["keep_io_types"] is True
    assert checked
