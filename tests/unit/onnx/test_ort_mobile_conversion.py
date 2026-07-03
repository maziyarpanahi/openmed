"""Tests for Android ONNX Runtime Mobile conversion artifacts."""

from __future__ import annotations

import importlib
import json
import logging
import types
from pathlib import Path

import pytest


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def _ort_module():
    return importlib.import_module("openmed.onnx.ort_mobile")


def test_convert_android_onnx_to_ort_writes_model_and_op_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ort_module()
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    validation = types.SimpleNamespace(operators=("Add", "MatMul"), opset=18)

    def fake_convert_onnx_models_to_ort(
        model_path_or_dir,
        *,
        output_dir=None,
        optimization_styles=None,
        target_platform=None,
        save_optimized_onnx_model=True,
        allow_conversion_failures=True,
        enable_type_reduction=False,
    ):
        assert model_path_or_dir == onnx_path
        assert output_dir == tmp_path
        assert optimization_styles == ["Fixed"]
        assert target_platform == "arm"
        assert save_optimized_onnx_model is False
        assert allow_conversion_failures is False
        assert enable_type_reduction is True
        (tmp_path / "model.ort").write_bytes(b"ort")
        (tmp_path / "model.required_operators_and_types.config").write_text(
            ";18;Add,MatMul\n",
            encoding="utf-8",
        )

    fake_tools = types.SimpleNamespace(
        OptimizationStyle=types.SimpleNamespace(Fixed="Fixed"),
        convert_onnx_models_to_ort=fake_convert_onnx_models_to_ort,
    )
    monkeypatch.setattr(module, "_load_ort_tools", lambda: fake_tools)

    result = module.convert_android_onnx_to_ort(
        onnx_path,
        output_dir=tmp_path,
        validation=validation,
    )

    assert result.skipped is False
    assert result.ort_path == tmp_path / "model.ort"
    assert (
        result.op_config_path == tmp_path / "model.required_operators_and_types.config"
    )
    assert result.ort_path.exists()
    op_config = result.op_config_path.read_text(encoding="utf-8")
    assert "Add" in op_config
    assert "MatMul" in op_config
    assert result.to_metadata(tmp_path) == {
        "format": "ort-android",
        "profile": "android",
        "source_onnx_path": "model.onnx",
        "optimization_style": "Fixed",
        "target_platform": "arm",
        "op_config_type": "required_operators_and_types",
        "enable_type_reduction": True,
        "ort_path": "model.ort",
        "op_config_path": "model.required_operators_and_types.config",
        "operators": ["Add", "MatMul"],
        "opset": 18,
    }


def test_convert_android_profile_records_ort_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    ort_module = _ort_module()

    def fake_export_onnx(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"onnx")
        return path

    def fake_export_android_fp16(onnx_path, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"fp16")
        return path

    validation = types.SimpleNamespace(
        operators=("Add", "MatMul"),
        opset=module.ANDROID_ONNX_OPSET,
        to_metadata=lambda: {
            "profile": "android",
            "opset": module.ANDROID_ONNX_OPSET,
            "inputs": [],
            "outputs": [],
            "operators": ["Add", "MatMul"],
            "unsupported_ops": [],
            "warnings": [],
        },
    )

    def fake_save_source_assets(model_id, output_dir, **kwargs):
        output_dir = Path(output_dir)
        config = {
            "model_type": "bert",
            "id2label": {"0": "O"},
            "max_sequence_length": kwargs["max_seq_length"],
        }
        (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (output_dir / "id2label.json").write_text('{"0": "O"}', encoding="utf-8")
        return config, []

    def fake_convert_onnx_models_to_ort(*args, output_dir=None, **kwargs):
        assert kwargs["enable_type_reduction"] is True
        output_dir = Path(output_dir)
        (output_dir / "model.ort").write_bytes(b"ort")
        (output_dir / "model.required_operators_and_types.config").write_text(
            ";18;Add,MatMul\n",
            encoding="utf-8",
        )

    fake_tools = types.SimpleNamespace(
        OptimizationStyle=types.SimpleNamespace(Fixed="Fixed"),
        convert_onnx_models_to_ort=fake_convert_onnx_models_to_ort,
    )

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "export_android_fp16", fake_export_android_fp16)
    monkeypatch.setattr(module, "validate_android_profile", lambda *args: validation)
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(ort_module, "_load_ort_tools", lambda: fake_tools)

    result = module.convert(
        "OpenMed/test-model", tmp_path / "artifact", profile="android"
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.formats == ["onnx-android", "ort-android"]
    assert [item["path"] for item in manifest["artifacts"]] == [
        "model.onnx",
        "model_fp16.onnx",
        "model.ort",
    ]
    ort_artifact = manifest["artifacts"][2]
    assert ort_artifact["format"] == "ort-android"
    assert ort_artifact["metadata"]["format"] == "ort-android"
    assert ort_artifact["metadata"]["ort_path"] == "model.ort"
    assert (
        ort_artifact["metadata"]["op_config_path"]
        == "model.required_operators_and_types.config"
    )
    assert ort_artifact["metadata"]["operators"] == ["Add", "MatMul"]
    assert (result.output_dir / "model.ort").exists()
    assert "MatMul" in (
        result.output_dir / "model.required_operators_and_types.config"
    ).read_text(encoding="utf-8")


def test_convert_android_profile_skips_ort_when_tooling_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _convert_module()
    ort_module = _ort_module()

    def fake_export_onnx(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"onnx")
        return path

    def fake_export_android_fp16(onnx_path, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"fp16")
        return path

    validation = types.SimpleNamespace(
        operators=("Add",),
        opset=module.ANDROID_ONNX_OPSET,
        to_metadata=lambda: {
            "profile": "android",
            "opset": module.ANDROID_ONNX_OPSET,
            "operators": ["Add"],
            "unsupported_ops": [],
            "warnings": [],
        },
    )

    def fake_save_source_assets(model_id, output_dir, **kwargs):
        output_dir = Path(output_dir)
        config = {
            "model_type": "bert",
            "id2label": {"0": "O"},
            "max_sequence_length": kwargs["max_seq_length"],
        }
        (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (output_dir / "id2label.json").write_text('{"0": "O"}', encoding="utf-8")
        return config, []

    def raise_unavailable():
        raise ort_module.OrtMobileConversionUnavailable(
            ort_module.ORT_CONVERSION_UNAVAILABLE_REASON
        )

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "export_android_fp16", fake_export_android_fp16)
    monkeypatch.setattr(module, "validate_android_profile", lambda *args: validation)
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(ort_module, "_load_ort_tools", raise_unavailable)
    caplog.set_level(logging.INFO, logger="openmed.onnx.ort_mobile")

    result = module.convert(
        "Patient John Doe",
        tmp_path / "artifact",
        profile="android",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert (result.output_dir / "model.onnx").exists()
    assert "ort-android" not in manifest["formats"]
    assert [item["path"] for item in manifest["artifacts"]] == [
        "model.onnx",
        "model_fp16.onnx",
    ]
    assert "Skipping Android ORT mobile conversion" in caplog.text
    assert "pip install openmed[onnx]" in caplog.text
    assert "Patient" not in caplog.text
    assert "John Doe" not in caplog.text
