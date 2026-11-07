"""Tests for the fp16 WebGPU export parity gate."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from openmed.onnx.webgpu_gate import (
    DEFAULT_FP32_KEEP_OPS,
    FP16_MAX_FINITE,
    WEBGPU_FP16_GATE_KEY,
    Fp16ParityObservation,
    WebGpuFp16GateConfig,
    WebGpuFp16GateError,
    evaluate_webgpu_fp16_gate,
    verify_webgpu_fp16_gate_evidence,
)

ID2LABEL = {
    "0": "O",
    "1": "B-PERSON",
    "2": "I-PERSON",
    "3": "B-DATE",
    "4": "I-DATE",
}
PHI_SENTINEL = "SENTINEL-Zelda-Quixote-1977-04-02"
COMPUTED_PROBE = {
    "status": "computed",
    "lengths": [8],
    "tensors_probed": 3,
    "limit": FP16_MAX_FINITE,
    "max_abs_activation": 12.0,
    "overflow_tensor_count": 0,
    "overflow_op_types": [],
}


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one_hot_rows(labels: list[list[int]], width: int = 5) -> list[list[list[float]]]:
    return [
        [[4.0 if index == label else -1.0 for index in range(width)] for label in row]
        for row in labels
    ]


# A PERSON span (tokens 0-1), an O token and a DATE span (tokens 3-4).
REFERENCE_LABELS = [[1, 2, 0, 3, 4, 0]]


# ---------------------------------------------------------------------------
# Pure scoring: runs without ONNX or ONNX Runtime.
# ---------------------------------------------------------------------------


def test_gate_passes_well_scaled_logits_and_records_deltas() -> None:
    reference = _one_hot_rows(REFERENCE_LABELS)
    candidate = [
        [[value + 0.01 for value in token] for token in row] for row in reference
    ]

    report = evaluate_webgpu_fp16_gate(
        [Fp16ParityObservation("synthetic", reference, candidate, requested_length=6)],
        id2label=ID2LABEL,
        activation_probe=COMPUTED_PROBE,
        evidence={"evaluated_lengths": [6], "artifact_sha256": "0" * 64},
    )

    assert report["passed"] is True
    assert report["reasons"] == []
    assert report["action"] == "emitted"
    assert report["policy"] == "fail"
    assert report["evaluated_lengths"] == [6]
    assert report["tolerances"]["logit_atol"] == pytest.approx(0.05)
    assert report["logits"]["max_abs_diff"] == pytest.approx(0.01)
    assert report["logits"]["non_finite_values"] == 0
    assert report["tokens"] == {"evaluated": 6, "mismatches": 0, "mismatch_rate": 0.0}
    assert report["spans"]["reference"] == 2
    assert report["spans"]["agreement"] == 1.0
    assert report["per_label"]["PERSON"]["critical"] is True
    assert report["per_label"]["DATE"]["recall_delta"] == 0.0
    assert report["max_critical_recall_delta"] == 0.0
    assert report["results"][0]["requested_length"] == 6
    json.dumps(report, allow_nan=False)


def test_gate_fails_non_finite_fp16_logits_without_nan_in_report() -> None:
    reference = _one_hot_rows(REFERENCE_LABELS)
    candidate = json.loads(json.dumps(reference))
    candidate[0][2][0] = float("inf")
    candidate[0][4][1] = float("nan")

    report = evaluate_webgpu_fp16_gate(
        [Fp16ParityObservation("synthetic", reference, candidate, requested_length=6)],
        id2label=ID2LABEL,
        activation_probe=COMPUTED_PROBE,
    )

    assert report["passed"] is False
    assert "non_finite_logits" in report["reasons"]
    assert "logit_tolerance_exceeded" in report["reasons"]
    assert report["logits"]["non_finite_values"] == 2
    assert report["results"][0]["max_abs_diff"] is None
    json.dumps(report, allow_nan=False)


def test_gate_fails_when_fp16_loses_a_critical_span() -> None:
    reference = _one_hot_rows(REFERENCE_LABELS)
    # The fp16 graph labels the PERSON tokens as O; logits stay within tolerance
    # of a near-tie so only token, span and recall evidence can catch it.
    reference[0][0] = [1.00, 1.01, -1.0, -1.0, -1.0]
    reference[0][1] = [1.00, -1.0, 1.01, -1.0, -1.0]
    candidate = json.loads(json.dumps(reference))
    candidate[0][0] = [1.02, 1.0, -1.0, -1.0, -1.0]
    candidate[0][1] = [1.02, -1.0, 1.0, -1.0, -1.0]

    report = evaluate_webgpu_fp16_gate(
        [Fp16ParityObservation("synthetic", reference, candidate, requested_length=6)],
        id2label=ID2LABEL,
        activation_probe=COMPUTED_PROBE,
    )

    assert report["logits"]["within_tolerance"] is True
    assert report["passed"] is False
    assert "token_mismatch_rate_exceeded" in report["reasons"]
    assert "critical_recall_delta_exceeded" in report["reasons"]
    assert report["per_label"]["PERSON"]["recall_delta"] == 1.0
    assert report["max_critical_recall_delta"] == 1.0
    assert report["spans"]["identical"] is False


def test_gate_fails_closed_on_activation_overflow_or_missing_probe() -> None:
    rows = _one_hot_rows(REFERENCE_LABELS)
    observation = Fp16ParityObservation("synthetic", rows, rows, requested_length=6)

    overflow = evaluate_webgpu_fp16_gate(
        [observation],
        id2label=ID2LABEL,
        activation_probe={
            **COMPUTED_PROBE,
            "overflow_tensor_count": 2,
            "overflow_op_types": ["Pow", "ReduceMean"],
        },
    )
    missing = evaluate_webgpu_fp16_gate([observation], id2label=ID2LABEL)
    no_inputs = evaluate_webgpu_fp16_gate([], activation_probe=COMPUTED_PROBE)

    assert overflow["reasons"] == ["fp16_activation_overflow"]
    assert missing["reasons"] == ["activation_probe_failed"]
    assert missing["activation_probe"] == {"status": "missing"}
    assert no_inputs["reasons"] == ["no_evaluation_inputs"]


def test_gate_policy_controls_failure_action() -> None:
    rows = _one_hot_rows(REFERENCE_LABELS)
    observation = Fp16ParityObservation("synthetic", rows, rows)

    rejected = evaluate_webgpu_fp16_gate([observation], id2label=ID2LABEL)
    omitted = evaluate_webgpu_fp16_gate(
        [observation],
        id2label=ID2LABEL,
        gate_config=WebGpuFp16GateConfig(on_failure="omit"),
    )

    assert rejected["action"] == "rejected"
    assert omitted["action"] == "omitted"
    assert omitted["policy"] == "omit"
    error = WebGpuFp16GateError(rejected)
    assert str(error) == "fp16 WebGPU parity gate failed: activation_probe_failed"
    assert error.report["passed"] is False


def test_gate_config_validates_policy_tolerances_and_keep_list() -> None:
    config = WebGpuFp16GateConfig(
        on_failure=" OMIT ",
        keep_fp32_ops=("Softmax", " Softmax ", "", "Erf"),
        keep_fp32_nodes=["node_a", "node_a"],
    )

    assert config.on_failure == "omit"
    assert config.keep_fp32_ops == ("Softmax", "Erf")
    assert config.keep_fp32_nodes == ("node_a",)
    assert WebGpuFp16GateConfig().keep_fp32_ops == DEFAULT_FP32_KEEP_OPS
    assert {"LayerNormalization", "Softmax", "ReduceMean"} <= set(DEFAULT_FP32_KEEP_OPS)
    with pytest.raises(ValueError, match="on_failure"):
        WebGpuFp16GateConfig(on_failure="skip")
    with pytest.raises(ValueError, match="logit_atol"):
        WebGpuFp16GateConfig(logit_atol=float("nan"))
    with pytest.raises(ValueError, match="activation_limit"):
        WebGpuFp16GateConfig(activation_limit=70000.0)
    with pytest.raises(ValueError, match="max_token_mismatch_rate"):
        WebGpuFp16GateConfig(max_token_mismatch_rate=1.5)


def _gated_manifest(root: Path, *, passed: bool = True) -> dict[str, Any]:
    artifact = root / "model.webgpu.onnx"
    artifact.write_bytes(b"fp16-graph")
    return {
        "formats": ["onnx", "webgpu"],
        "artifacts": [
            {"format": "onnx", "path": "model.onnx", "precision": "float32"},
            {
                "format": "webgpu",
                "path": "model.webgpu.onnx",
                "precision": "float16",
            },
        ],
        WEBGPU_FP16_GATE_KEY: {
            "passed": passed,
            "tolerances": WebGpuFp16GateConfig().tolerances(),
            "evaluated_lengths": [8, 16],
            "artifact_sha256": _sha256(artifact),
        },
    }


def test_verify_evidence_requires_a_passing_digest_bound_record(
    tmp_path: Path,
) -> None:
    manifest = _gated_manifest(tmp_path)

    assert verify_webgpu_fp16_gate_evidence(tmp_path, manifest)["passed"] is True
    assert verify_webgpu_fp16_gate_evidence(tmp_path, {"formats": ["onnx"]}) is None

    ungated = {
        key: value for key, value in manifest.items() if key != WEBGPU_FP16_GATE_KEY
    }
    with pytest.raises(ValueError, match="requires an fp16 parity gate record"):
        verify_webgpu_fp16_gate_evidence(tmp_path, ungated)
    with pytest.raises(ValueError, match="did not pass"):
        verify_webgpu_fp16_gate_evidence(
            tmp_path, _gated_manifest(tmp_path, passed=False)
        )
    incomplete = json.loads(json.dumps(manifest))
    del incomplete[WEBGPU_FP16_GATE_KEY]["evaluated_lengths"]
    with pytest.raises(ValueError, match="incomplete"):
        verify_webgpu_fp16_gate_evidence(tmp_path, incomplete)
    with pytest.raises(ValueError, match="without a gated artifact"):
        verify_webgpu_fp16_gate_evidence(
            tmp_path,
            {**manifest, "artifacts": manifest["artifacts"][:1]},
        )
    with pytest.raises(ValueError, match="without a gated artifact"):
        verify_webgpu_fp16_gate_evidence(
            tmp_path,
            {**manifest, "formats": ["onnx"], "artifacts": []},
            require=True,
        )
    escaped = json.loads(json.dumps(manifest))
    escaped["artifacts"][1]["path"] = "../model.webgpu.onnx"
    with pytest.raises(ValueError, match="inside the bundle"):
        verify_webgpu_fp16_gate_evidence(tmp_path, escaped)

    (tmp_path / "model.webgpu.onnx").write_bytes(b"swapped-after-gate")
    with pytest.raises(ValueError, match="digest does not match") as excinfo:
        verify_webgpu_fp16_gate_evidence(tmp_path, manifest)
    assert str(tmp_path) not in str(excinfo.value)


# ---------------------------------------------------------------------------
# convert() orchestration with an injected gate: runs without ONNX Runtime.
# ---------------------------------------------------------------------------


def _fake_export_onnx(model_id, output_path, **kwargs):
    path = Path(output_path)
    path.write_bytes(b"onnx")
    return path


def _fake_export_webgpu(onnx_path, output_path, **kwargs):
    path = Path(output_path)
    path.write_bytes(b"webgpu")
    return path


def _fake_save_source_assets(model_id, output_dir, **kwargs):
    output_dir = Path(output_dir)
    config = {"model_type": "bert", "id2label": ID2LABEL}
    (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return config, []


def _injected_gate(*, passed: bool, calls: list[dict[str, Any]], policy: str = "fail"):
    def fake_gate(onnx_path, webgpu_path, **kwargs):
        calls.append({"reference": Path(onnx_path).name, **kwargs})
        rows = _one_hot_rows(REFERENCE_LABELS)
        probe = dict(COMPUTED_PROBE)
        if not passed:
            probe.update(overflow_tensor_count=1, overflow_op_types=["Pow"])
        return evaluate_webgpu_fp16_gate(
            [Fp16ParityObservation("synthetic", rows, rows, requested_length=8)],
            id2label=ID2LABEL,
            gate_config=kwargs["gate_config"],
            activation_probe=probe,
            evidence={
                "artifact": Path(webgpu_path).name,
                "artifact_sha256": _sha256(Path(webgpu_path)),
                "evaluated_lengths": list(kwargs["lengths"]),
            },
        )

    return fake_gate


def _install_fakes(monkeypatch, module, gate, publish_calls=None) -> None:
    monkeypatch.setattr(module, "export_onnx", _fake_export_onnx)
    monkeypatch.setattr(module, "export_webgpu", _fake_export_webgpu)
    monkeypatch.setattr(module, "save_source_assets", _fake_save_source_assets)
    monkeypatch.setattr(module, "validate_webgpu_fp16_export", gate)

    def fake_publish(**kwargs):
        if publish_calls is not None:
            publish_calls.append(kwargs)
        return types.SimpleNamespace(skipped=False, repo_id="OpenMed/test-v1-onnx")

    monkeypatch.setattr(module, "publish_artifact", fake_publish)


def test_convert_gates_webgpu_even_without_graph_optimization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    calls: list[dict[str, Any]] = []
    publish_calls: list[dict[str, Any]] = []
    _install_fakes(
        monkeypatch, module, _injected_gate(passed=True, calls=calls), publish_calls
    )

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        optimize_onnx=False,
        validation_lengths=(8, 16),
        validation_texts=["synthetic validation note"],
        publish_to_hub=True,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    gate = manifest[WEBGPU_FP16_GATE_KEY]
    assert len(calls) == 1
    assert calls[0]["reference"] == "model.onnx"
    assert calls[0]["lengths"] == (8, 16)
    assert calls[0]["validation_texts"] == ["synthetic validation note"]
    assert calls[0]["sample_texts"] == (module.DEFAULT_SAMPLE_TEXT,)
    assert manifest["optimization"]["enabled"] is False
    assert result.formats == ["onnx", "webgpu"]
    assert gate["passed"] is True
    assert gate["evaluated_lengths"] == [8, 16]
    assert gate["artifact_sha256"] == _sha256(result.output_dir / "model.webgpu.onnx")
    assert set(gate["tolerances"]) >= {"logit_atol", "critical_recall_delta_limit"}
    assert publish_calls[0]["formats"] == ["onnx", "webgpu"]


def test_convert_fail_policy_raises_removes_artifact_and_skips_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    calls: list[dict[str, Any]] = []
    publish_calls: list[dict[str, Any]] = []
    _install_fakes(
        monkeypatch, module, _injected_gate(passed=False, calls=calls), publish_calls
    )
    output_dir = tmp_path / "artifact"

    with pytest.raises(
        WebGpuFp16GateError, match="fp16_activation_overflow"
    ) as excinfo:
        module.convert(
            "OpenMed/test-model",
            output_dir,
            optimize_onnx=False,
            publish_to_hub=True,
        )

    assert excinfo.value.report["action"] == "rejected"
    assert not (output_dir / "model.webgpu.onnx").exists()
    assert not (output_dir / "openmed-onnx.json").exists()
    assert publish_calls == []


def test_convert_omit_policy_drops_webgpu_and_records_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _convert_module()
    calls: list[dict[str, Any]] = []
    publish_calls: list[dict[str, Any]] = []
    _install_fakes(
        monkeypatch, module, _injected_gate(passed=False, calls=calls), publish_calls
    )

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        optimize_onnx=False,
        publish_to_hub=True,
        webgpu_gate_config=WebGpuFp16GateConfig(on_failure="omit"),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.formats == ["onnx"]
    assert manifest["formats"] == ["onnx"]
    assert not (result.output_dir / "model.webgpu.onnx").exists()
    assert manifest[WEBGPU_FP16_GATE_KEY]["passed"] is False
    assert manifest[WEBGPU_FP16_GATE_KEY]["action"] == "omitted"
    assert manifest[WEBGPU_FP16_GATE_KEY]["reasons"] == ["fp16_activation_overflow"]
    assert publish_calls[0]["formats"] == ["onnx"]
    assert "fp16_activation_overflow" in caplog.text
    assert str(tmp_path) not in caplog.text


def test_cli_forwards_gate_policy_and_keep_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        module, "convert", lambda *args, **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "openmed.onnx.convert",
            "--model",
            "OpenMed/test-model",
            "--output",
            "dist/out",
            "--no-onnx-optimization",
            "--webgpu-gate-policy",
            "omit",
            "--webgpu-keep-fp32-op",
            "Erf",
            "--webgpu-keep-fp32-node",
            "/encoder/layer.0/Add",
        ],
    )

    module.main()

    config = captured["webgpu_gate_config"]
    assert captured["optimize_onnx"] is False
    assert config.on_failure == "omit"
    assert config.keep_fp32_ops == (*DEFAULT_FP32_KEEP_OPS, "Erf")
    assert config.keep_fp32_nodes == ("/encoder/layer.0/Add",)


# ---------------------------------------------------------------------------
# Real fp16 conversion and ONNX Runtime CPU evaluation on tiny synthetic graphs.
# ---------------------------------------------------------------------------


def _onnx_stack():
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxruntime.transformers.float16")
    numpy = pytest.importorskip("numpy")
    return onnx, numpy


def _token_inputs(helper, tensor_proto, *, token_types: bool = False):
    names = ["input_ids", "attention_mask"] + (
        ["token_type_ids"] if token_types else []
    )
    return [
        helper.make_tensor_value_info(name, tensor_proto.INT64, ["batch", "sequence"])
        for name in names
    ]


def _save_model(onnx, graph, path: Path) -> Path:
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18)]
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _rms_overflow_graph(onnx, numpy, path: Path, *, scale: float) -> Path:
    """Embedding -> RMS norm -> classifier; Pow/ReduceMean overflow fp16 at scale."""

    helper = onnx.helper
    rng = numpy.random.default_rng(7)
    embedding = (rng.standard_normal((32, 8)) * scale).astype(numpy.float32)
    weights = (rng.standard_normal((8, 5)) * 2.0).astype(numpy.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gather", ["embedding", "input_ids"], ["hidden"], name="gather"
            ),
            helper.make_node("Pow", ["hidden", "two"], ["squared"], name="pow"),
            helper.make_node(
                "ReduceMean",
                ["squared", "axes"],
                ["mean_square"],
                keepdims=1,
                name="mean",
            ),
            helper.make_node("Sqrt", ["mean_square"], ["rms"], name="sqrt"),
            helper.make_node("Div", ["hidden", "rms"], ["normalized"], name="div"),
            helper.make_node(
                "MatMul", ["normalized", "classifier"], ["projected"], name="matmul"
            ),
            helper.make_node("Add", ["projected", "bias"], ["logits"], name="add"),
        ],
        "rms_overflow",
        _token_inputs(helper, onnx.TensorProto),
        [
            helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT, ["batch", "sequence", 5]
            )
        ],
        [
            onnx.numpy_helper.from_array(embedding, "embedding"),
            onnx.numpy_helper.from_array(weights, "classifier"),
            onnx.numpy_helper.from_array(numpy.zeros(5, dtype=numpy.float32), "bias"),
            onnx.numpy_helper.from_array(numpy.array(2.0, dtype=numpy.float32), "two"),
            onnx.numpy_helper.from_array(numpy.array([-1], dtype=numpy.int64), "axes"),
        ],
    )
    return _save_model(onnx, graph, path)


def _tiny_random_bert(
    onnx, numpy, path: Path, *, vocab: int = 48, hidden: int = 16
) -> Path:
    """One-layer, single-head BERT-style token classifier with random weights."""

    helper = onnx.helper
    tensor = onnx.TensorProto
    rng = numpy.random.default_rng(3939)

    def weight(name, *shape, scale=0.3):
        return onnx.numpy_helper.from_array(
            (rng.standard_normal(shape) * scale).astype(numpy.float32), name
        )

    def const(name, value, dtype=numpy.float32):
        return onnx.numpy_helper.from_array(numpy.array(value, dtype=dtype), name)

    initializers = [
        weight("word_embeddings", vocab, hidden, scale=1.0),
        weight("position_embeddings", 64, hidden, scale=0.5),
        weight("token_type_embeddings", 2, hidden, scale=0.5),
        const("zero_i", 0, numpy.int64),
        const("one_i", 1, numpy.int64),
        const("one_f", 1.0),
        const("mask_fill", -10000.0),
        const("scale", 1.0 / numpy.sqrt(hidden)),
        const("sqrt2", numpy.sqrt(2.0)),
        const("half", 0.5),
        const("mask_axes", [1], numpy.int64),
    ]
    for prefix in ("ln0", "ln1", "ln2"):
        initializers.append(const(f"{prefix}_gamma", numpy.ones(hidden)))
        initializers.append(const(f"{prefix}_beta", numpy.zeros(hidden)))
    for name, shape in (
        ("wq", (hidden, hidden)),
        ("wk", (hidden, hidden)),
        ("wv", (hidden, hidden)),
        ("wo", (hidden, hidden)),
        ("w1", (hidden, hidden * 4)),
        ("w2", (hidden * 4, hidden)),
        ("wc", (hidden, len(ID2LABEL))),
    ):
        initializers.append(weight(name, *shape))
        initializers.append(weight(f"b{name[1:]}", shape[1], scale=0.05))

    def node(op, inputs, outputs, **attrs):
        return helper.make_node(op, inputs, outputs, name=outputs[0], **attrs)

    def dense(prefix, source, target):
        return [
            node("MatMul", [source, f"w{prefix}"], [f"{target}_mm"]),
            node("Add", [f"{target}_mm", f"b{prefix}"], [target]),
        ]

    nodes = [
        node("Gather", ["word_embeddings", "input_ids"], ["word"]),
        node("Shape", ["input_ids"], ["ids_shape"]),
        node("Gather", ["ids_shape", "one_i"], ["seq_len"], axis=0),
        node("Range", ["zero_i", "seq_len", "one_i"], ["position_ids"]),
        node("Gather", ["position_embeddings", "position_ids"], ["position"]),
        node("Gather", ["token_type_embeddings", "token_type_ids"], ["token_type"]),
        node("Add", ["word", "position"], ["word_position"]),
        node("Add", ["word_position", "token_type"], ["embeddings"]),
        node(
            "LayerNormalization",
            ["embeddings", "ln0_gamma", "ln0_beta"],
            ["hidden0"],
            axis=-1,
            epsilon=1e-12,
        ),
        *dense("q", "hidden0", "query"),
        *dense("k", "hidden0", "key"),
        *dense("v", "hidden0", "value"),
        node("Transpose", ["key"], ["key_t"], perm=[0, 2, 1]),
        node("MatMul", ["query", "key_t"], ["raw_scores"]),
        node("Mul", ["raw_scores", "scale"], ["scores"]),
        node("Cast", ["attention_mask"], ["mask_f"], to=tensor.FLOAT),
        node("Sub", ["one_f", "mask_f"], ["inverted_mask"]),
        node("Mul", ["inverted_mask", "mask_fill"], ["mask_bias"]),
        node("Unsqueeze", ["mask_bias", "mask_axes"], ["mask_bias_3d"]),
        node("Add", ["scores", "mask_bias_3d"], ["masked_scores"]),
        node("Softmax", ["masked_scores"], ["probs"], axis=-1),
        node("MatMul", ["probs", "value"], ["context"]),
        *dense("o", "context", "attention_out"),
        node("Add", ["attention_out", "hidden0"], ["attention_residual"]),
        node(
            "LayerNormalization",
            ["attention_residual", "ln1_gamma", "ln1_beta"],
            ["hidden1"],
            axis=-1,
            epsilon=1e-12,
        ),
        *dense("1", "hidden1", "ffn_in"),
        node("Div", ["ffn_in", "sqrt2"], ["gelu_scaled"]),
        node("Erf", ["gelu_scaled"], ["gelu_erf"]),
        node("Add", ["gelu_erf", "one_f"], ["gelu_shift"]),
        node("Mul", ["ffn_in", "gelu_shift"], ["gelu_mul"]),
        node("Mul", ["gelu_mul", "half"], ["gelu"]),
        *dense("2", "gelu", "ffn_out"),
        node("Add", ["ffn_out", "hidden1"], ["ffn_residual"]),
        node(
            "LayerNormalization",
            ["ffn_residual", "ln2_gamma", "ln2_beta"],
            ["hidden2"],
            axis=-1,
            epsilon=1e-12,
        ),
        *dense("c", "hidden2", "logits"),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_random_bert",
        _token_inputs(helper, tensor, token_types=True),
        [
            helper.make_tensor_value_info(
                "logits", tensor.FLOAT, ["batch", "sequence", len(ID2LABEL)]
            )
        ],
        initializers,
    )
    return _save_model(onnx, graph, path)


def _near_tie_graph(onnx, numpy, path: Path) -> Path:
    """Logits near 1000 whose argmax depends on detail fp16 cannot store."""

    helper = onnx.helper
    rng = numpy.random.default_rng(11)
    embedding = (rng.standard_normal((32, 2)) * 0.2).astype(numpy.float32)
    weights = numpy.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0]], dtype=numpy.float32)
    bias = numpy.array([0.0, 1000.0, 1000.25], dtype=numpy.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gather", ["embedding", "input_ids"], ["hidden"], name="gather"
            ),
            helper.make_node(
                "MatMul", ["hidden", "classifier"], ["projected"], name="matmul"
            ),
            helper.make_node("Add", ["projected", "bias"], ["logits"], name="add"),
        ],
        "near_tie",
        _token_inputs(helper, onnx.TensorProto),
        [
            helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT, ["batch", "sequence", 3]
            )
        ],
        [
            onnx.numpy_helper.from_array(embedding, "embedding"),
            onnx.numpy_helper.from_array(weights, "classifier"),
            onnx.numpy_helper.from_array(bias, "bias"),
        ],
    )
    return _save_model(onnx, graph, path)


def _gate(
    convert_module, reference: Path, *, keep_fp32_ops=None, config=None, **kwargs
):
    candidate = convert_module.export_webgpu(
        reference,
        reference.with_name("model.webgpu.onnx"),
        keep_fp32_ops=keep_fp32_ops,
    )
    from openmed.onnx.webgpu_gate import validate_webgpu_fp16_export

    return candidate, validate_webgpu_fp16_export(
        reference,
        candidate,
        lengths=kwargs.pop("lengths", (8, 16, 32)),
        config=config or {"id2label": ID2LABEL, "vocab_size": 32},
        **kwargs,
    )


def test_fp16_overflowing_activation_fails_gate_without_keep_list(
    tmp_path: Path,
) -> None:
    onnx, numpy = _onnx_stack()
    module = _convert_module()
    reference = _rms_overflow_graph(onnx, numpy, tmp_path / "model.onnx", scale=1000.0)

    _, unprotected = _gate(module, reference, keep_fp32_ops=())
    candidate, protected = _gate(module, reference)

    assert unprotected["passed"] is False
    assert "fp16_activation_overflow" in unprotected["reasons"]
    assert unprotected["activation_probe"]["overflow_op_types"] == ["Pow", "ReduceMean"]
    assert unprotected["activation_probe"]["max_abs_activation"] > FP16_MAX_FINITE
    assert unprotected["keep_fp32"]["op_types"] == []

    assert protected["passed"] is True, protected["reasons"]
    assert protected["activation_probe"]["overflow_tensor_count"] == 0
    assert protected["activation_probe"]["max_abs_activation"] < FP16_MAX_FINITE
    assert protected["keep_fp32"]["op_types"] == list(DEFAULT_FP32_KEEP_OPS)
    assert protected["keep_fp32"]["kept_node_count"] == 3
    assert protected["artifact_sha256"] == _sha256(candidate)
    assert not list(tmp_path.glob(".openmed-fp16-probe-*"))


def test_well_scaled_graph_passes_gate_with_recorded_deltas(tmp_path: Path) -> None:
    onnx, numpy = _onnx_stack()
    module = _convert_module()
    reference = _rms_overflow_graph(onnx, numpy, tmp_path / "model.onnx", scale=1.0)

    _, report = _gate(module, reference, keep_fp32_ops=())

    assert report["passed"] is True, report["reasons"]
    assert report["evaluated_lengths"] == [8, 16, 32]
    assert 0.0 < report["logits"]["max_abs_diff"] <= report["tolerances"]["logit_atol"]
    assert report["tokens"]["evaluated"] == 2 * (8 + 16 + 32) - (2 + 4 + 8)
    assert report["activation_probe"]["lengths"] == [8, 32]


def test_tiny_random_bert_passes_gate_and_records_per_label_deltas(
    tmp_path: Path,
) -> None:
    onnx, numpy = _onnx_stack()
    module = _convert_module()
    reference = _tiny_random_bert(onnx, numpy, tmp_path / "model.onnx")

    def tokenizer(texts, **kwargs):
        ids = numpy.arange(len(texts) * 12, dtype=numpy.int64).reshape(len(texts), 12)
        return {"input_ids": ids % 48, "attention_mask": numpy.ones_like(ids)}

    candidate, report = _gate(
        module,
        reference,
        config={
            "id2label": ID2LABEL,
            "vocab_size": 48,
            "max_sequence_length": 32,
        },
        lengths=(8, 16, 32, 64),
        validation_texts=[PHI_SENTINEL],
        tokenizer=tokenizer,
    )

    assert report["passed"] is True, report["reasons"]
    assert report["evaluated_lengths"] == [8, 16, 32]
    assert report["skipped_lengths"] == [64]
    assert report["corpus_examples"] == 1
    assert report["corpus_status"] == "evaluated"
    assert report["spans"]["reference"] > 0
    assert report["spans"]["agreement"] == 1.0
    assert set(report["per_label"]) <= {"PERSON", "DATE"}
    assert report["critical_labels"]
    assert report["activation_probe"]["tensors_probed"] > 10
    assert report["activation_probe"]["overflow_tensor_count"] == 0
    assert report["keep_fp32"]["kept_node_count"] == 4
    assert report["runtime"]["execution_provider"] == "CPUExecutionProvider"
    assert report["artifact_sha256"] == _sha256(candidate)
    assert report["reference_sha256"] == _sha256(reference)
    serialized = json.dumps(report, allow_nan=False)
    assert PHI_SENTINEL not in serialized
    assert str(tmp_path) not in serialized

    converted = onnx.load(str(candidate))
    onnx.checker.check_model(converted)
    props = {prop.key: prop.value for prop in converted.metadata_props}
    assert json.loads(props["openmed.webgpu.fp32_keep_ops"]) == list(
        DEFAULT_FP32_KEEP_OPS
    )
    float_types = {
        info.name: info.type.tensor_type.elem_type
        for info in converted.graph.value_info
    }
    kept = [
        node
        for node in converted.graph.node
        if node.op_type in {"LayerNormalization", "Softmax"}
    ]
    assert len(kept) == 4
    assert all(float_types[node.input[0]] == onnx.TensorProto.FLOAT for node in kept)


def test_fp16_precision_loss_fails_span_and_recall_parity(tmp_path: Path) -> None:
    onnx, numpy = _onnx_stack()
    module = _convert_module()
    reference = _near_tie_graph(onnx, numpy, tmp_path / "model.onnx")

    _, report = _gate(
        module,
        reference,
        config={
            "id2label": {"0": "O", "1": "B-PERSON", "2": "B-DATE"},
            "vocab_size": 32,
        },
    )

    assert report["passed"] is False
    assert report["logits"]["within_tolerance"] is True
    assert report["activation_probe"]["overflow_tensor_count"] == 0
    assert "token_mismatch_rate_exceeded" in report["reasons"]
    assert "critical_recall_delta_exceeded" in report["reasons"]
    assert report["max_critical_recall_delta"] > 0.003


def test_gate_fails_closed_when_requested_corpus_cannot_be_tokenized(
    tmp_path: Path,
) -> None:
    onnx, numpy = _onnx_stack()
    module = _convert_module()
    reference = _rms_overflow_graph(onnx, numpy, tmp_path / "model.onnx", scale=1.0)

    def broken_tokenizer(texts, **kwargs):
        raise RuntimeError(f"cannot tokenize {texts[0]}")

    _, sample_only = _gate(module, reference, sample_texts=["synthetic sample"])
    _, required = _gate(module, reference, validation_texts=[PHI_SENTINEL])
    _, broken = _gate(
        module,
        reference,
        validation_texts=[PHI_SENTINEL],
        tokenizer=broken_tokenizer,
    )

    assert sample_only["passed"] is True
    assert sample_only["corpus_status"] == "skipped_no_tokenizer"
    assert required["passed"] is False
    assert required["reasons"] == ["validation_corpus_unavailable"]
    assert broken["reasons"] == ["validation_corpus_unavailable"]
    assert broken["corpus_status"] == "tokenizer_error"
    assert broken["runtime_errors"][0]["error_type"] == "RuntimeError"
    assert PHI_SENTINEL not in json.dumps(required)
    assert PHI_SENTINEL not in json.dumps(broken)


def test_export_webgpu_keeps_blocked_chains_fp32_and_sorts_graph(
    tmp_path: Path,
) -> None:
    onnx, numpy = _onnx_stack()
    module = _convert_module()
    reference = _rms_overflow_graph(onnx, numpy, tmp_path / "model.onnx", scale=1.0)

    candidate = module.export_webgpu(
        reference,
        tmp_path / "model.webgpu.onnx",
        keep_fp32_nodes=("div",),
    )

    converted = onnx.load(str(candidate))
    onnx.checker.check_model(converted)
    types_by_name = {
        info.name: info.type.tensor_type.elem_type
        for info in converted.graph.value_info
    }
    nodes = {node.name: node for node in converted.graph.node if node.name}
    producers = {
        output: node.op_type for node in converted.graph.node for output in node.output
    }
    # Pow -> ReduceMean -> Sqrt -> Div stays fp32: no fp16 round trip in between.
    assert types_by_name.get("squared") != onnx.TensorProto.FLOAT16
    assert types_by_name.get("mean_square") != onnx.TensorProto.FLOAT16
    assert producers[nodes["mean"].input[0]] == "Pow"
    assert producers[nodes["sqrt"].input[0]] == "ReduceMean"
    assert producers[nodes["div"].input[1]] == "Sqrt"
    props = {prop.key: prop.value for prop in converted.metadata_props}
    assert json.loads(props["openmed.webgpu.fp32_keep_nodes"]) == ["div"]
    produced = [output for node in converted.graph.node for output in node.output]
    assert len(produced) == len(set(produced))


def test_convert_end_to_end_gates_real_fp16_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx, numpy = _onnx_stack()
    module = _convert_module()
    source = _tiny_random_bert(onnx, numpy, tmp_path / "source.onnx")
    publish_calls: list[dict[str, Any]] = []

    def export_tiny_bert(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(source.read_bytes())
        return path

    def save_assets(model_id, output_dir, **kwargs):
        config = {
            "model_type": "bert",
            "id2label": ID2LABEL,
            "vocab_size": 48,
            "max_sequence_length": 64,
        }
        (Path(output_dir) / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return config, []

    def publish(**kwargs):
        publish_calls.append(kwargs)
        return types.SimpleNamespace(skipped=False, repo_id="OpenMed/test-v1-onnx")

    monkeypatch.setattr(module, "export_onnx", export_tiny_bert)
    monkeypatch.setattr(module, "save_source_assets", save_assets)
    monkeypatch.setattr(module, "_webgpu_gate_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(module, "publish_artifact", publish)

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        optimize_onnx=False,
        validation_lengths=(8, 32),
        publish_to_hub=True,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    gate = manifest[WEBGPU_FP16_GATE_KEY]
    assert result.formats == ["onnx", "webgpu"]
    assert gate["passed"] is True, gate["reasons"]
    assert gate["evaluated_lengths"] == [8, 32]
    assert gate["corpus_status"] == "skipped_no_tokenizer"
    assert gate["artifact_sha256"] == _sha256(result.output_dir / "model.webgpu.onnx")
    assert module.validate_onnx_bundle(result.output_dir)[WEBGPU_FP16_GATE_KEY]
    assert publish_calls[0]["formats"] == ["onnx", "webgpu"]
    assert not list(result.output_dir.glob(".openmed-fp16-probe-*"))


def test_convert_end_to_end_rejects_overflowing_fp16_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx, numpy = _onnx_stack()
    module = _convert_module()
    source = _rms_overflow_graph(onnx, numpy, tmp_path / "source.onnx", scale=1000.0)
    publish_calls: list[dict[str, Any]] = []

    def export_graph(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(source.read_bytes())
        return path

    def save_assets(model_id, output_dir, **kwargs):
        config = {"model_type": "bert", "id2label": ID2LABEL, "vocab_size": 32}
        (Path(output_dir) / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return config, []

    monkeypatch.setattr(module, "export_onnx", export_graph)
    monkeypatch.setattr(module, "save_source_assets", save_assets)
    monkeypatch.setattr(module, "_webgpu_gate_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(
        module,
        "publish_artifact",
        lambda **kwargs: publish_calls.append(kwargs),
    )
    unprotected = WebGpuFp16GateConfig(keep_fp32_ops=())

    with pytest.raises(WebGpuFp16GateError, match="fp16_activation_overflow"):
        module.convert(
            "OpenMed/test-model",
            tmp_path / "rejected",
            optimize_onnx=False,
            validation_lengths=(8, 16),
            publish_to_hub=True,
            webgpu_gate_config=unprotected,
        )
    omitted = module.convert(
        "OpenMed/test-model",
        tmp_path / "omitted",
        optimize_onnx=False,
        validation_lengths=(8, 16),
        webgpu_gate_config=WebGpuFp16GateConfig(keep_fp32_ops=(), on_failure="omit"),
    )

    assert publish_calls == []
    assert not (tmp_path / "rejected" / "model.webgpu.onnx").exists()
    assert omitted.formats == ["onnx"]
    record = json.loads(omitted.manifest_path.read_text(encoding="utf-8"))[
        WEBGPU_FP16_GATE_KEY
    ]
    assert record["action"] == "omitted"
    assert record["reasons"] == ["fp16_activation_overflow"]
