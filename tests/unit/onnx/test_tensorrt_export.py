"""Tests for TensorRT token-classification engine export and inference."""

from __future__ import annotations

import importlib
import json
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest


def _export_module():
    return importlib.import_module("openmed.onnx.tensorrt_export")


def _session_module():
    return importlib.import_module("openmed.onnx.tensorrt_session")


def _legacy_trt_module() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__="10.9.0",
        BuilderFlag=types.SimpleNamespace(FP16=1, INT8=2),
        IInt8EntropyCalibrator2=object,
    )


def _trt11_module() -> types.SimpleNamespace:
    return types.SimpleNamespace(__version__="11.1.0")


def test_shape_profile_validates_and_serializes_ranges() -> None:
    module = _export_module()
    profile = module.TensorRTShapeProfile(
        min_batch_size=1,
        opt_batch_size=2,
        max_batch_size=4,
        min_sequence_length=16,
        opt_sequence_length=128,
        max_sequence_length=512,
    )

    assert profile.to_dict() == {
        "min": [1, 16],
        "opt": [2, 128],
        "max": [4, 512],
    }
    with pytest.raises(ValueError, match="sequence length"):
        module.TensorRTShapeProfile(
            min_sequence_length=128,
            opt_sequence_length=64,
        )


def test_optimization_profile_covers_dynamic_token_inputs() -> None:
    module = _export_module()
    calls = []

    class FakeOptimizationProfile:
        def set_shape(self, name, minimum, optimum, maximum):
            calls.append((name, minimum, optimum, maximum))
            return True

    class FakeBuilder:
        def create_optimization_profile(self):
            return FakeOptimizationProfile()

    class FakeConfig:
        def add_optimization_profile(self, profile):
            self.profile = profile
            return 0

    inputs = [
        types.SimpleNamespace(
            name="input_ids",
            shape=(-1, -1),
            is_shape_tensor=False,
        ),
        types.SimpleNamespace(
            name="attention_mask",
            shape=(-1, -1),
            is_shape_tensor=False,
        ),
    ]
    network = types.SimpleNamespace(
        num_inputs=len(inputs),
        get_input=lambda index: inputs[index],
    )

    module._add_optimization_profile(
        FakeBuilder(),
        network,
        FakeConfig(),
        module.TensorRTShapeProfile(
            min_sequence_length=8,
            opt_sequence_length=64,
            max_sequence_length=256,
        ),
    )

    assert calls == [
        ("input_ids", (1, 8), (1, 64), (1, 256)),
        ("attention_mask", (1, 8), (1, 64), (1, 256)),
    ]


def test_serializer_parses_onnx_sets_fp16_and_workspace(tmp_path: Path) -> None:
    module = _export_module()
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    state = {}

    class FakeOptimizationProfile:
        def set_shape(self, name, minimum, optimum, maximum):
            state.setdefault("shapes", {})[name] = (minimum, optimum, maximum)
            return True

    class FakeConfig:
        def set_memory_pool_limit(self, pool, size):
            state["workspace"] = (pool, size)

        def add_optimization_profile(self, profile):
            state["profile"] = profile
            return 0

        def set_flag(self, flag):
            state["flag"] = flag

    inputs = [
        types.SimpleNamespace(
            name="input_ids",
            shape=(-1, -1),
            is_shape_tensor=False,
        ),
        types.SimpleNamespace(
            name="attention_mask",
            shape=(-1, -1),
            is_shape_tensor=False,
        ),
    ]
    network = types.SimpleNamespace(
        num_inputs=len(inputs),
        get_input=lambda index: inputs[index],
    )

    class FakeBuilder:
        def __init__(self, logger):
            self.logger = logger

        def create_network(self, flags):
            state["network_flags"] = flags
            return network

        def create_builder_config(self):
            return FakeConfig()

        def create_optimization_profile(self):
            return FakeOptimizationProfile()

        def build_serialized_network(self, parsed_network, config):
            assert parsed_network is network
            return b"serialized-engine"

    class FakeParser:
        num_errors = 0

        def __init__(self, parsed_network, logger):
            assert parsed_network is network

        def parse_from_file(self, path):
            state["parsed_path"] = path
            return True

    class FakeLogger:
        WARNING = 1

        def __init__(self, level):
            self.level = level

    fake_trt = types.SimpleNamespace(
        Logger=FakeLogger,
        Builder=FakeBuilder,
        OnnxParser=FakeParser,
        BuilderFlag=types.SimpleNamespace(FP16="fp16"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0),
    )

    serialized = module._serialize_tensorrt_engine(
        onnx_path,
        precision="fp16",
        shape_profile=module.TensorRTShapeProfile(),
        workspace_size_bytes=4096,
        calibration_spec=None,
        trt=fake_trt,
    )

    assert serialized == b"serialized-engine"
    assert state["parsed_path"] == str(onnx_path)
    assert state["network_flags"] == 1
    assert state["workspace"] == ("workspace", 4096)
    assert state["flag"] == "fp16"


def test_certify_tensorrt_reference_accepts_matching_spans() -> None:
    module = _export_module()
    reference = np.array([[[0.9, 0.1], [0.1, 0.9]]], dtype=np.float32)
    candidate = reference + np.float32(1e-5)

    result = module.certify_tensorrt_reference(
        reference_logits=reference,
        tensorrt_logits=candidate,
        id2label={0: "O", 1: "B-NAME"},
        attention_mask=np.array([[1, 1]], dtype=np.int32),
    )

    assert result.passed is True
    assert result.reference_token_spans == result.tensorrt_token_spans
    assert result.max_abs_logit_delta == pytest.approx(1e-5, abs=1e-7)


def test_certify_tensorrt_reference_rejects_span_change() -> None:
    module = _export_module()
    reference = np.array([[[0.9, 0.1], [0.1, 0.9]]], dtype=np.float32)
    candidate = np.array([[[0.9, 0.1], [0.9, 0.1]]], dtype=np.float32)

    with pytest.raises(module.TensorRTVerificationError, match="decoded token spans"):
        module.certify_tensorrt_reference(
            reference_logits=reference,
            tensorrt_logits=candidate,
            id2label={0: "O", 1: "B-NAME"},
            tolerance=1.0,
        )


def test_build_fp16_engine_writes_hashes_and_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _export_module()
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"pinned-onnx")
    reference = np.array([[[0.9, 0.1], [0.1, 0.9]]], dtype=np.float32)
    serialize_calls = []

    def fake_serialize(path, **kwargs):
        serialize_calls.append((path, kwargs))
        return b"deterministic-engine"

    class FakeSession:
        def __init__(self, engine_path):
            assert Path(engine_path).is_file()

        def run(self, **inputs):
            assert set(inputs) == {"input_ids", "attention_mask"}
            return reference

    monkeypatch.setattr(module, "_serialize_tensorrt_engine", fake_serialize)
    result = module.build_tensorrt_engine(
        onnx_path,
        tmp_path / "model.engine",
        family="bert",
        precision="fp16",
        sample_inputs={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]},
        reference_logits=reference,
        id2label={0: "O", 1: "B-NAME"},
        trt_module=_legacy_trt_module(),
        session_factory=FakeSession,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert result.engine_path.read_bytes() == b"deterministic-engine"
    assert result.engine_sha256 == module.sha256_file(result.engine_path)
    assert metadata["build_input_sha256"] == result.build_input_sha256
    assert metadata["engine_sha256"] == result.engine_sha256
    assert metadata["synthetic_verification"]["passed"] is True
    assert serialize_calls[0][1]["shape_profile"].maximum == (1, 512)


def test_reproducibility_hash_rejects_engine_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _export_module()
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    output_path = tmp_path / "model.engine"
    monkeypatch.setattr(
        module,
        "_serialize_tensorrt_engine",
        lambda *args, **kwargs: b"engine",
    )

    with pytest.raises(module.TensorRTReproducibilityError, match="engine hash"):
        module.build_tensorrt_engine(
            onnx_path,
            output_path,
            family="bert",
            precision="fp32",
            expected_engine_sha256="0" * 64,
            trt_module=_legacy_trt_module(),
        )

    assert not output_path.exists()


def test_int8_build_uses_shared_calibration_and_passing_g4_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _export_module()
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    captured = {}

    def fake_serialize(path, **kwargs):
        captured.update(kwargs)
        return b"int8-engine"

    monkeypatch.setattr(
        module,
        "load_quantization_calibration_texts",
        lambda: [" synthetic one ", "synthetic two"],
    )
    monkeypatch.setattr(module, "_serialize_tensorrt_engine", fake_serialize)
    result = module.build_tensorrt_engine(
        onnx_path,
        tmp_path / "model.engine",
        family="deberta-v2",
        precision="int8",
        calibration_tokenizer=object(),
        candidate_recall={"PERSON": 0.990},
        parent_recall={"PERSON": 0.992},
        trt_module=_legacy_trt_module(),
    )

    assert result.recall_delta_gate is not None
    assert result.recall_delta_gate.passed is True
    assert result.calibration_sha256 == module.calibration_texts_sha256(
        ["synthetic one", "synthetic two"]
    )
    assert captured["calibration_spec"].texts == (
        "synthetic one",
        "synthetic two",
    )


def test_int8_build_rejects_missing_or_over_budget_recall_evidence(
    tmp_path: Path,
) -> None:
    module = _export_module()
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    with pytest.raises(module.TensorRTQuantizationRejected) as missing:
        module.build_tensorrt_engine(
            onnx_path,
            tmp_path / "missing.engine",
            family="bert",
            precision="int8",
            calibration_tokenizer=object(),
            trt_module=_legacy_trt_module(),
        )
    assert missing.value.gate.source == "missing_evidence"

    with pytest.raises(module.TensorRTQuantizationRejected) as regression:
        module.build_tensorrt_engine(
            onnx_path,
            tmp_path / "regression.engine",
            family="bert",
            precision="int8",
            calibration_tokenizer=object(),
            candidate_recall={"PERSON": 0.980},
            parent_recall={"PERSON": 0.990},
            trt_module=_legacy_trt_module(),
        )
    assert regression.value.gate.passed is False
    assert not (tmp_path / "regression.engine").exists()


def test_trt11_int8_uses_modelopt_explicit_quantization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _export_module()
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    calls = []

    def fake_quantize(source, output, **kwargs):
        calls.append((source, output, kwargs))
        output.write_bytes(b"qdq-onnx")
        return output

    def fake_serialize(path, **kwargs):
        assert path.read_bytes() == b"qdq-onnx"
        assert kwargs["calibration_spec"] is None
        return b"trt11-engine"

    monkeypatch.setattr(module, "_quantize_onnx_with_modelopt", fake_quantize)
    monkeypatch.setattr(module, "_serialize_tensorrt_engine", fake_serialize)
    result = module.build_tensorrt_engine(
        onnx_path,
        tmp_path / "model.engine",
        family="bert",
        precision="int8",
        calibration_tokenizer=object(),
        precomputed_delta={"PERSON": 0.001},
        trt_module=_trt11_module(),
    )

    assert calls[0][1] == tmp_path / "model.int8.onnx"
    assert result.build_onnx_path.read_bytes() == b"qdq-onnx"


def test_tensorrt_benchmark_report_records_device_tier(tmp_path: Path) -> None:
    module = _export_module()
    public = importlib.import_module("openmed.onnx")
    output = tmp_path / module.TENSORRT_BENCHMARK_REPORT
    assert public.TensorRTShapeProfile is module.TensorRTShapeProfile
    path = public.write_tensorrt_benchmark_report(
        output,
        model_name="OpenMed/test-model",
        generated_at="2026-07-19T00:00:00+00:00",
        records=[
            module.TensorRTBenchmarkRecord(
                device_tier="jetson-orin",
                device="Orin AGX",
                precision="fp16",
                latency_ms=4.0,
                throughput_items_per_second=250.0,
                sample_count=3,
                sequence_length=128,
            )
        ],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]["devices"]["jetson-orin:Orin AGX"]
    assert payload["suite"] == "tensorrt-runtime"
    assert metrics["device_tier"] == "jetson-orin"
    assert metrics["latency"]["p50_ms"] == 4.0
    assert metrics["throughput"]["items_per_second"] == 250.0


def test_named_io_session_loads_engine_and_returns_logits(tmp_path: Path) -> None:
    module = _session_module()
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"trusted-engine")
    registry = {}

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)
            self.shape = self.array.shape
            self.dtype = self.array.dtype
            self.pointer = id(self)
            registry[self.pointer] = self

        def contiguous(self):
            return self

        def data_ptr(self):
            return self.pointer

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class FakeStream:
        cuda_stream = 17

        def synchronize(self):
            return None

    class FakeContext:
        def __init__(self):
            self.addresses = {}
            self.input_shape = None

        def set_input_shape(self, name, shape):
            self.input_shape = shape
            return True

        def set_tensor_address(self, name, pointer):
            self.addresses[name] = pointer
            return True

        def get_tensor_shape(self, name):
            return (*self.input_shape, 2)

        def execute_async_v3(self, stream_handle):
            output = registry[self.addresses["logits"]]
            output.array.fill(0.5)
            return stream_handle == 17

    context = FakeContext()

    class FakeEngine:
        num_io_tensors = 3

        def get_tensor_name(self, index):
            return ["input_ids", "attention_mask", "logits"][index]

        def get_tensor_mode(self, name):
            return "input" if name != "logits" else "output"

        def get_tensor_dtype(self, name):
            return "float32" if name == "logits" else "int32"

        def create_execution_context(self):
            return context

    class FakeRuntime:
        def __init__(self, logger):
            self.logger = logger

        def deserialize_cuda_engine(self, engine_bytes):
            assert engine_bytes == b"trusted-engine"
            return FakeEngine()

    class FakeLogger:
        WARNING = 1

        def __init__(self, level):
            self.level = level

    fake_trt = types.SimpleNamespace(
        Logger=FakeLogger,
        Runtime=FakeRuntime,
        TensorIOMode=types.SimpleNamespace(INPUT="input"),
        nptype=lambda dtype: np.dtype(dtype),
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            Stream=lambda device: FakeStream(),
            stream=lambda stream: nullcontext(),
        ),
        as_tensor=lambda array, device: FakeTensor(array),
        from_numpy=lambda array: FakeTensor(array),
        empty=lambda shape, dtype, device: FakeTensor(np.empty(shape, dtype=dtype)),
    )

    session = module.TensorRTTokenClassificationSession(
        engine_path,
        trt_module=fake_trt,
        torch_module=fake_torch,
    )
    logits = session.run(
        input_ids=np.array([[1, 2]], dtype=np.int64),
        attention_mask=np.array([[1, 1]], dtype=np.int64),
    )

    assert logits.shape == (1, 2, 2)
    assert np.all(logits == np.float32(0.5))


def test_gpu_built_engine_matches_onnx_reference_spans(tmp_path: Path) -> None:
    """Exercise real ONNX -> TensorRT parity when a CUDA runner is available."""

    trt = pytest.importorskip(
        "tensorrt",
        reason="TensorRT GPU parity requires the TensorRT Python runtime",
    )
    torch = pytest.importorskip(
        "torch",
        reason="TensorRT GPU parity requires CUDA-enabled PyTorch",
    )
    if not torch.cuda.is_available():
        pytest.skip("TensorRT GPU parity requires an available CUDA device")
    onnx = pytest.importorskip(
        "onnx",
        reason="TensorRT GPU parity requires ONNX graph helpers",
    )
    ort = pytest.importorskip(
        "onnxruntime",
        reason="TensorRT GPU parity requires the ONNX Runtime reference",
    )
    module = _export_module()
    onnx_path = tmp_path / "synthetic-token-classifier.onnx"

    input_ids = onnx.helper.make_tensor_value_info(
        "input_ids",
        onnx.TensorProto.INT32,
        ["batch", "sequence"],
    )
    attention_mask = onnx.helper.make_tensor_value_info(
        "attention_mask",
        onnx.TensorProto.INT32,
        ["batch", "sequence"],
    )
    logits = onnx.helper.make_tensor_value_info(
        "logits",
        onnx.TensorProto.FLOAT,
        ["batch", "sequence", 2],
    )
    axes = onnx.helper.make_tensor(
        "axes",
        onnx.TensorProto.INT64,
        [1],
        [2],
    )
    nodes = [
        onnx.helper.make_node("Cast", ["input_ids"], ["ids_float"], to=1),
        onnx.helper.make_node("Unsqueeze", ["ids_float", "axes"], ["positive"]),
        onnx.helper.make_node("Neg", ["positive"], ["negative"]),
        onnx.helper.make_node(
            "Concat",
            ["negative", "positive"],
            ["logits"],
            axis=2,
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "synthetic-token-classifier",
        [input_ids, attention_mask],
        [logits],
        [axes],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    onnx.save(model, onnx_path)

    sample_inputs = {
        "input_ids": np.array([[4, 3, 7, 6, 2, 10, 1, 5]], dtype=np.int32),
        "attention_mask": np.ones((1, 8), dtype=np.int32),
    }
    reference_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    reference = reference_session.run(["logits"], sample_inputs)[0]
    result = module.build_tensorrt_engine(
        onnx_path,
        tmp_path / "synthetic.engine",
        family="synthetic-bert",
        precision="fp32",
        shape_profile=module.TensorRTShapeProfile(
            min_sequence_length=1,
            opt_sequence_length=8,
            max_sequence_length=32,
        ),
        sample_inputs=sample_inputs,
        reference_logits=reference,
        id2label={0: "O", 1: "B-NAME"},
        trt_module=trt,
    )

    assert result.verification is not None
    assert result.verification.passed is True
    assert result.verification.reference_token_spans == (
        {"label": "NAME", "start_token": 0, "end_token": 8},
    )
