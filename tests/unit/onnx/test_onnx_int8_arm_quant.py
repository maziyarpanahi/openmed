"""Tests for Android ONNX INT8 quantization and recall certification."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest


def _quant_module():
    return importlib.import_module("openmed.onnx.quantize_int8")


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def test_quantize_dynamic_int8_writes_qint8_arm_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _quant_module()
    source = tmp_path / "model.onnx"
    output = tmp_path / "model_int8.onnx"
    source.write_bytes(b"onnx")
    calls: list[dict[str, Any]] = []

    onnx_mod = types.ModuleType("onnx")
    onnx_mod.load = lambda path: {"path": path}
    onnx_mod.checker = types.SimpleNamespace(check_model=lambda model: None)

    runtime_mod = types.ModuleType("onnxruntime")
    quantization_mod = types.ModuleType("onnxruntime.quantization")
    quantization_mod.QuantType = types.SimpleNamespace(QInt8="qint8")

    def fake_quantize_dynamic(input_model, output_model, **kwargs):
        calls.append(
            {
                "input_model": input_model,
                "output_model": output_model,
                **kwargs,
            }
        )
        Path(output_model).write_bytes(b"int8")

    quantization_mod.quantize_dynamic = fake_quantize_dynamic
    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quantization_mod)

    result = module.quantize_dynamic_int8(source, output)

    assert result == output
    assert output.read_bytes() == b"int8"
    assert calls == [
        {
            "input_model": str(source),
            "output_model": str(output),
            "weight_type": "qint8",
            "op_types_to_quantize": ["MatMul", "Gemm"],
        }
    ]


def test_int8_report_certifies_and_omits_record_text(tmp_path: Path) -> None:
    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")

    report = module.write_int8_recall_delta_report(
        source_model_id="OpenMed/stub-token-classifier",
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        parent_runner=_perfect_runner,
        candidate_runner=_perfect_runner,
    )

    assert report["format"] == "onnx-int8"
    assert report["certified"] is True
    assert report["quant_recall_delta"] == 0.0
    assert report["per_label"]["PERSON"]["fp_recall"] == 1.0
    assert report["per_label"]["PERSON"]["int8_recall"] == 1.0
    assert report["per_label"]["PERSON"]["gold_span_count"] == 1

    raw_report = (artifact / "recall_delta.json").read_text(encoding="utf-8")
    assert "John Doe" not in raw_report
    assert "Patient John Doe arrived today." not in raw_report
    _assert_no_record_text_fields(json.loads(raw_report))

    manifest = json.loads((artifact / "openmed-onnx.json").read_text())
    config = json.loads((artifact / "config.json").read_text())
    int8_artifact = next(
        item for item in manifest["artifacts"] if item["path"] == "model_int8.onnx"
    )
    assert manifest["formats"] == ["onnx-android", "onnx-int8"]
    assert manifest["certified"] is True
    assert manifest["quant_recall_delta"] == 0.0
    assert int8_artifact["format"] == "onnx-int8"
    assert int8_artifact["metadata"]["quant_recall_delta"] == 0.0
    assert int8_artifact["metadata"]["format"] == "onnx-int8"
    assert config["_onnx_quantization"]["format"] == "onnx-int8"
    assert config["certified"] is True


def test_int8_report_marks_over_budget_delta_uncertified(tmp_path: Path) -> None:
    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")

    report = module.write_int8_recall_delta_report(
        source_model_id="OpenMed/stub-token-classifier",
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        parent_runner=_perfect_runner,
        candidate_runner=_miss_runner,
    )

    assert report["certified"] is False
    assert report["quant_recall_delta"] == 1.0
    assert report["delta"]["blocking_format"] == "onnx-int8"

    manifest = json.loads((artifact / "openmed-onnx.json").read_text())
    assert manifest["certified"] is False
    assert manifest["quantization"]["certified"] is False


def test_convert_android_profile_emits_int8_and_runs_certification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    report_path = tmp_path / "reports" / "recall_delta.json"
    report_calls: list[dict[str, Any]] = []

    def fake_export_onnx(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"onnx")
        return path

    def fake_export_android_fp16(onnx_path, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"fp16")
        return path

    def fake_quantize_dynamic_int8(onnx_path, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"int8")
        return path

    def fake_validate_android_profile(model_path, **kwargs):
        return types.SimpleNamespace(
            to_metadata=lambda: {
                "profile": "android",
                "opset": module.ANDROID_ONNX_OPSET,
                "inputs": [],
                "outputs": [],
                "unsupported_ops": [],
                "warnings": [],
            }
        )

    def fake_save_source_assets(model_id, output_dir, **kwargs):
        output_dir = Path(output_dir)
        config = {
            "model_type": "bert",
            "id2label": {"0": "O", "1": "B-NAME"},
            "max_sequence_length": kwargs["max_seq_length"],
        }
        (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (output_dir / "id2label.json").write_text(
            json.dumps(config["id2label"]),
            encoding="utf-8",
        )
        return config, []

    def fake_write_int8_recall_delta_report(**kwargs):
        report_calls.append(kwargs)
        return {
            "format": "onnx-int8",
            "limit": 0.005,
            "certified": True,
            "quant_recall_delta": 0.0,
            "metric": "character_recall",
            "report_path": "recall_delta.json",
        }

    def fake_convert_android_onnx_to_ort(*args, **kwargs):
        return types.SimpleNamespace(skipped=True, ort_path=None)

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "export_android_fp16", fake_export_android_fp16)
    monkeypatch.setattr(module, "quantize_dynamic_int8", fake_quantize_dynamic_int8)
    monkeypatch.setattr(
        module, "validate_android_profile", fake_validate_android_profile
    )
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(
        module,
        "write_int8_recall_delta_report",
        fake_write_int8_recall_delta_report,
    )
    monkeypatch.setattr(
        module,
        "convert_android_onnx_to_ort",
        fake_convert_android_onnx_to_ort,
    )

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        profile="android",
        eval_suite_path=fixture_path,
        recall_delta_report_path=report_path,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.formats == ["onnx-android", "onnx-int8"]
    assert [item["path"] for item in manifest["artifacts"]] == [
        "model.onnx",
        "model_fp16.onnx",
        "model_int8.onnx",
    ]
    assert manifest["artifacts"][2]["format"] == "onnx-int8"
    assert manifest["artifacts"][2]["precision"] == "int8"
    assert manifest["artifacts"][2]["metadata"]["certified"] is True
    assert report_calls[0]["eval_suite_path"] == fixture_path
    assert report_calls[0]["output_path"] == report_path
    assert report_calls[0]["int8_model_path"] == result.output_dir / "model_int8.onnx"


def _write_fixture(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "id": "stub-note",
                        "text": "Patient John Doe arrived today.",
                        "gold_spans": [
                            {
                                "start": 8,
                                "end": 16,
                                "label": "PERSON",
                                "text": "John Doe",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_stub_onnx_artifact(path: Path) -> Path:
    path.mkdir()
    (path / "model.onnx").write_bytes(b"onnx")
    (path / "model_int8.onnx").write_bytes(b"int8")
    config = {
        "model_type": "bert",
        "id2label": {"0": "O", "1": "PERSON"},
        "max_sequence_length": 128,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "id2label.json").write_text(
        json.dumps(config["id2label"]),
        encoding="utf-8",
    )
    (path / "openmed-onnx.json").write_text(
        json.dumps(
            {
                "format": "openmed-onnx",
                "format_version": 1,
                "formats": ["onnx-android"],
                "artifacts": [
                    {
                        "format": "onnx-android",
                        "path": "model.onnx",
                        "precision": "float32",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _perfect_runner(fixture: Any, model_name: str, device: str) -> list[dict[str, Any]]:
    del model_name, device
    span = fixture.gold_spans[0]
    return [
        {
            "entity_group": span.label,
            "score": 0.99,
            "start": span.start,
            "end": span.end,
            "word": fixture.text[span.start : span.end],
        }
    ]


def _miss_runner(fixture: Any, model_name: str, device: str) -> list[dict[str, Any]]:
    del fixture, model_name, device
    return []


def _assert_no_record_text_fields(value: Any) -> None:
    forbidden = {
        "text",
        "note_text",
        "record_text",
        "source_text",
        "span_text",
        "word",
        "predicted_spans",
        "gold_spans",
        "fixture_ids",
        "record_id",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for item in value.values():
            _assert_no_record_text_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_record_text_fields(item)
