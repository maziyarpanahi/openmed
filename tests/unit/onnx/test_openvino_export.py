"""Tests for OpenVINO export, runtime selection, and INT8 gating."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _export_module():
    return importlib.import_module("openmed.onnx.openvino_export")


def _session_module():
    return importlib.import_module("openmed.onnx.openvino_session")


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def test_resolve_openvino_device_prefers_request_then_cpu_fallback() -> None:
    module = _session_module()

    direct = module.resolve_openvino_device("GPU", ("CPU", "GPU"))
    fallback = module.resolve_openvino_device("NPU", ("GPU", "CPU"))

    assert direct.selected_device == "GPU"
    assert direct.fallback_used is False
    assert fallback.selected_device == "CPU"
    assert fallback.fallback_used is True
    assert fallback.to_metadata()["available_devices"] == ["GPU", "CPU"]


def test_openvino_session_compiles_selected_device_and_returns_logits() -> None:
    module = _session_module()
    compile_calls = []
    logits = np.array([[[0.9, 0.1]]], dtype=np.float32)

    class FakeCore:
        available_devices = ("GPU", "CPU")

        def read_model(self, path):
            return {"path": path}

        def compile_model(self, model, device):
            compile_calls.append((model, device))
            return lambda inputs: {"logits": logits}

    session = module.OpenVinoTokenClassificationSession(
        "model.xml",
        device="NPU",
        core=FakeCore(),
    )

    result = session.run(input_ids=[[1]], attention_mask=[[1]])

    assert session.selected_device == "CPU"
    assert compile_calls[0][1] == "CPU"
    assert np.array_equal(result, logits)


def test_certify_openvino_reference_accepts_same_spans_within_tolerance() -> None:
    module = _export_module()
    reference = np.array(
        [
            [
                [0.9, 0.1, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.1, 0.9],
                [0.9, 0.1, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    candidate = reference + 0.0001

    result = module.certify_openvino_reference(
        reference_logits=reference,
        openvino_logits=candidate,
        id2label={0: "O", 1: "B-NAME", 2: "I-NAME"},
        attention_mask=np.array([[1, 1, 1, 1]], dtype=np.int64),
        tolerance=0.001,
    )

    assert result.passed is True
    assert result.max_abs_logit_delta == pytest.approx(0.0001, abs=1e-7)
    assert result.reference_token_spans == (
        {"label": "NAME", "start_token": 1, "end_token": 3},
    )


def test_certify_openvino_reference_rejects_changed_spans() -> None:
    module = _export_module()
    reference = np.array([[[0.1, 0.9], [0.9, 0.1]]], dtype=np.float32)
    candidate = np.array([[[0.9, 0.1], [0.9, 0.1]]], dtype=np.float32)

    with pytest.raises(module.OpenVinoVerificationError, match="decoded token spans"):
        module.certify_openvino_reference(
            reference_logits=reference,
            openvino_logits=candidate,
            id2label={0: "O", 1: "B-NAME"},
            tolerance=1.0,
        )


def test_export_openvino_ir_converts_saves_and_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _export_module()
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    reference = np.array([[[0.9, 0.1], [0.1, 0.9]]], dtype=np.float32)
    calls = []

    openvino_mod = types.ModuleType("openvino")

    def fake_convert_model(path):
        calls.append(("convert", path))
        return {"converted": path}

    def fake_save_model(model, path, compress_to_fp16=False):
        calls.append(("save", model, path, compress_to_fp16))
        Path(path).write_text("<xml />", encoding="utf-8")
        Path(path).with_suffix(".bin").write_bytes(b"weights")

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.model_path = model_path

        def run(self, **kwargs):
            return reference

    openvino_mod.convert_model = fake_convert_model
    openvino_mod.save_model = fake_save_model
    monkeypatch.setitem(sys.modules, "openvino", openvino_mod)
    monkeypatch.setattr(module, "OpenVinoTokenClassificationSession", FakeSession)

    result = module.export_openvino_ir(
        onnx_path,
        tmp_path / "openvino",
        sample_inputs={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]},
        reference_logits=reference,
        id2label={0: "O", 1: "B-NAME"},
    )

    assert result.model_xml_path.read_text(encoding="utf-8") == "<xml />"
    assert result.model_bin_path.read_bytes() == b"weights"
    assert result.verification is not None
    assert calls[0] == ("convert", str(onnx_path))
    assert calls[1][0] == "save"


def test_quantize_openvino_int8_saves_only_after_g4_gate_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _export_module()
    source_xml = tmp_path / "model.xml"
    source_xml.write_text("<xml />", encoding="utf-8")
    saved = {}

    openvino_mod = types.ModuleType("openvino")

    class FakeCore:
        def read_model(self, path):
            return {"source": path}

    def fake_save_model(model, path, compress_to_fp16=False):
        saved["model"] = model
        Path(path).write_text("<int8 />", encoding="utf-8")
        Path(path).with_suffix(".bin").write_bytes(b"int8")

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

    def fake_quantize(model, dataset):
        return {"quantized": model, "calibration_rows": dataset.rows}

    nncf_mod = types.ModuleType("nncf")
    nncf_mod.Dataset = FakeDataset
    nncf_mod.quantize = fake_quantize
    openvino_mod.Core = FakeCore
    openvino_mod.save_model = fake_save_model
    monkeypatch.setitem(sys.modules, "openvino", openvino_mod)
    monkeypatch.setitem(sys.modules, "nncf", nncf_mod)

    result = module.quantize_openvino_int8(
        source_xml,
        tmp_path / "openvino_int8",
        calibration_data=[{"input_ids": [[1]], "attention_mask": [[1]]}],
        family="bert",
        candidate_recall={"PERSON": 0.990},
        parent_recall={"PERSON": 0.992},
    )

    assert result.model_xml_path.read_text(encoding="utf-8") == "<int8 />"
    assert result.recall_delta_gate.passed is True
    assert result.to_metadata(tmp_path)["gate"] == "G4"
    assert saved["model"]["calibration_rows"][0]["input_ids"] == [[1]]


def test_quantize_openvino_int8_rejects_missing_recall_evidence(
    tmp_path: Path,
) -> None:
    module = _export_module()

    with pytest.raises(module.OpenVinoQuantizationRejected) as exc:
        module.quantize_openvino_int8(
            tmp_path / "model.xml",
            tmp_path / "openvino_int8",
            calibration_data=[{"input_ids": [[1]], "attention_mask": [[1]]}],
            family="bert",
        )

    assert exc.value.gate.passed is False
    assert exc.value.gate.source == "missing_evidence"


def test_openvino_benchmark_report_uses_benchmark_schema(tmp_path: Path) -> None:
    module = _export_module()
    public = importlib.import_module("openmed.onnx")
    output = tmp_path / module.OPENVINO_BENCHMARK_REPORT

    path = public.write_openvino_benchmark_report(
        output,
        model_name="OpenMed/test-model",
        generated_at="2026-07-01T00:00:00+00:00",
        records=[
            module.OpenVinoBenchmarkRecord(
                device="CPU",
                precision="float32",
                latency_ms=4.0,
                throughput_items_per_second=250.0,
                sample_count=3,
                sequence_length=32,
            )
        ],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["suite"] == "openvino-runtime"
    assert payload["device"] == "openvino:CPU"
    assert payload["metrics"]["devices"]["CPU"]["latency"]["p50_ms"] == 4.0
    assert payload["metrics"]["devices"]["CPU"]["throughput"] == {
        "items_per_second": 250.0
    }


def test_convert_openvino_profile_records_ir_artifact_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    reference = np.array([[[0.9, 0.1]]], dtype=np.float32)
    export_calls = []

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": np.array([[1]], dtype=np.int64),
                "attention_mask": np.array([[1]], dtype=np.int64),
            }

    def fake_export_onnx(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"onnx")
        return path

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

    def fake_export_openvino_ir(onnx_path, output_dir, **kwargs):
        export_calls.append(kwargs)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_xml = output_dir / "model.xml"
        model_xml.write_text("<xml />", encoding="utf-8")
        model_xml.with_suffix(".bin").write_bytes(b"weights")
        verification = types.SimpleNamespace(
            to_metadata=lambda: {"passed": True, "max_abs_logit_delta": 0.0}
        )
        return types.SimpleNamespace(
            model_xml_path=model_xml,
            to_metadata=lambda root: {
                "profile": "openvino",
                "model_xml_path": "openvino/model.xml",
                "synthetic_verification": verification.to_metadata(),
            },
        )

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(
        module,
        "_transformers_tokenizer_loader",
        lambda **kwargs: lambda *args, **loader_kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(module, "run_onnx_reference_logits", lambda *args: reference)
    monkeypatch.setattr(module, "export_openvino_ir", fake_export_openvino_ir)
    monkeypatch.setattr(
        module,
        "export_webgpu",
        lambda *args, **kwargs: pytest.fail("openvino profile must not emit WebGPU"),
    )

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        profile="openvino",
        include_webgpu=True,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.formats == ["openvino-ir"]
    assert manifest["artifacts"][0]["path"] == "openvino/model.xml"
    assert (
        manifest["artifacts"][0]["metadata"]["synthetic_verification"]["passed"] is True
    )
    assert export_calls[0]["id2label"] == {"0": "O", "1": "B-NAME"}
