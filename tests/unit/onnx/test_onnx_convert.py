"""Tests for ONNX and WebGPU conversion helpers."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def test_check_onnx_model_validates_by_path_for_large_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _convert_module()
    model_path = tmp_path / "large.onnx"
    model_path.write_bytes(b"onnx")
    checked = []

    onnx_mod = types.ModuleType("onnx")
    onnx_mod.checker = types.SimpleNamespace(
        check_model=lambda path: checked.append(path)
    )
    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)

    module._check_onnx_model(model_path)

    assert checked == [str(model_path)]


def test_consolidates_external_onnx_tensors_into_one_sidecar(
    tmp_path: Path,
) -> None:
    onnx = pytest.importorskip("onnx")
    numpy = pytest.importorskip("numpy")
    module = _convert_module()
    model_path = tmp_path / "model.onnx"
    first = onnx.numpy_helper.from_array(
        numpy.asarray([1.0, 2.0], dtype=numpy.float32),
        name="first",
    )
    second = onnx.numpy_helper.from_array(
        numpy.asarray([3.0, 4.0], dtype=numpy.float32),
        name="second",
    )
    graph = onnx.helper.make_graph([], "external", [], [], [first, second])
    model = onnx.helper.make_model(graph)
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=0,
    )
    before = {
        item.value
        for tensor in onnx.load(
            str(model_path), load_external_data=False
        ).graph.initializer
        for item in tensor.external_data
        if item.key == "location"
    }
    assert len(before) == 2

    module._consolidate_external_onnx_data(model_path)

    metadata = onnx.load(str(model_path), load_external_data=False)
    locations = {
        item.value
        for tensor in metadata.graph.initializer
        for item in tensor.external_data
        if item.key == "location"
    }
    assert locations == {"model.onnx.data"}
    assert (tmp_path / "model.onnx.data").is_file()
    assert all(not (tmp_path / name).exists() for name in before)
    onnx.checker.check_model(str(model_path))

    module._consolidate_external_onnx_data(model_path)

    assert (tmp_path / "model.onnx.data").is_file()
    onnx.checker.check_model(str(model_path))


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
    assert manifest["minimum_opset"] == 18
    assert manifest["dynamic_shapes"]["sequence_axis"] == "dynamic"
    assert manifest["dynamic_shapes"]["shape_buckets"]["buckets"][-1] == 2048
    assert manifest["optimization"]["enabled"] is False
    assert manifest["operator_fallbacks"] == []
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
        return types.SimpleNamespace(
            skipped=False, repo_id="OpenMed/test-model-v1-onnx"
        )

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "export_webgpu", fake_export_webgpu)
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(module, "publish_artifact", fake_publish_artifact)

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        optimize_onnx=False,
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

    converted = {}

    def fake_convert(model, keep_io_types):
        converted["model"] = model
        return {"fp16": model, "keep_io_types": keep_io_types}

    float16_mod.convert_float_to_float16 = fake_convert

    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime.transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime.transformers.float16", float16_mod)

    result = module.export_webgpu(input_path, output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"fp16"
    assert converted["model"] == str(input_path)
    assert saved["model"]["keep_io_types"] is True
    assert checked


def test_export_onnx_uses_torch_two_dynamic_shapes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _convert_module()
    output_path = tmp_path / "model.onnx"
    export_call = {}

    torch_mod = types.ModuleType("torch")

    class Module:
        def __init__(self) -> None:
            pass

        def eval(self) -> None:
            pass

        def to(self, *, dtype=None):
            self.dtype = dtype
            return self

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_dim(name, *, min=None, max=None):
        return {"name": name, "min": min, "max": max}

    def fake_export(
        model,
        args,
        f,
        *,
        input_names=None,
        output_names=None,
        opset_version=None,
        dynamo=True,
        dynamic_shapes=None,
        dynamic_axes=None,
        do_constant_folding=True,
    ):
        export_call.update(
            {
                "args": args,
                "input_names": input_names,
                "output_names": output_names,
                "opset_version": opset_version,
                "dynamo": dynamo,
                "dynamic_shapes": dynamic_shapes,
                "dynamic_axes": dynamic_axes,
                "do_constant_folding": do_constant_folding,
            }
        )
        Path(f).write_bytes(b"onnx")

    torch_mod.nn = types.SimpleNamespace(Module=Module)
    torch_mod.no_grad = NoGrad
    torch_mod.float32 = "float32"
    torch_mod.export = types.SimpleNamespace(Dim=fake_dim)
    torch_mod.onnx = types.SimpleNamespace(export=fake_export)

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            assert len(texts) == 2
            return {
                "input_ids": [[101, 102], [101, 102]],
                "attention_mask": [[1, 1], [1, 1]],
            }

    class FakeModel(Module):
        pass

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )
    transformers_mod.AutoModelForTokenClassification = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeModel()
    )

    onnx_mod = types.ModuleType("onnx")
    onnx_mod.load = lambda path, **kwargs: types.SimpleNamespace(
        graph=types.SimpleNamespace(initializer=[])
    )
    onnx_mod.checker = types.SimpleNamespace(check_model=lambda model: None)

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)

    assert module.export_onnx("OpenMed/test-model", output_path, max_seq_length=32)

    assert export_call["dynamo"] is True
    assert export_call["dynamic_axes"] is None
    assert export_call["dynamic_shapes"] == {
        "input_ids": {
            0: {"name": "batch", "min": 1, "max": None},
            1: {"name": "sequence", "min": 1, "max": None},
        },
        "attention_mask": {
            0: {"name": "batch", "min": 1, "max": None},
            1: {"name": "sequence", "min": 1, "max": None},
        },
    }


def test_longformer_export_compatibility_is_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    original_mask = object()
    modeling = types.SimpleNamespace(create_bidirectional_mask=original_mask)
    longformer_module = types.ModuleType("transformers.models.longformer")
    longformer_module.modeling_longformer = modeling
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.longformer",
        longformer_module,
    )
    model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="longformer"))
    torch_module = types.SimpleNamespace()

    with module._onnx_export_compatibility(model, torch_module):
        assert modeling.create_bidirectional_mask is not original_mask

    assert modeling.create_bidirectional_mask is original_mask


def test_convert_optimizes_before_webgpu_and_records_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    calls = {}

    def fake_export_onnx(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"raw")
        calls["export_path"] = path.name
        return path

    def fake_optimize(input_path, output_path, **kwargs):
        calls["optimize_input"] = Path(input_path).name
        path = Path(output_path)
        path.write_bytes(b"optimized")
        return {
            "enabled": True,
            "backend": "fake",
            "passes": {"attention_fusion": True},
            "minimum_opset": 18,
        }

    def fake_validate(unoptimized_path, optimized_path, **kwargs):
        assert Path(unoptimized_path).name == "model.unoptimized.onnx"
        assert Path(optimized_path).name == "model.onnx"
        return {
            "passed": True,
            "dynamic_shapes": {"passed": True, "lengths": []},
            "numeric_parity": {"passed": True, "results": []},
            "latency": {"passed": True, "improvement": 0.25},
            "operator_fallbacks": [
                {
                    "requested_provider": "CUDAExecutionProvider",
                    "execution_provider": "CPUExecutionProvider",
                    "op_type": "LayerNormalization",
                    "node_name": "layernorm",
                    "reason": "profiled_on_fallback_provider",
                }
            ],
        }

    def fake_export_webgpu(onnx_path, output_path, **kwargs):
        assert Path(onnx_path).read_bytes() == b"optimized"
        path = Path(output_path)
        path.write_bytes(b"webgpu")
        return path

    def fake_save_source_assets(model_id, output_dir, **kwargs):
        output_dir = Path(output_dir)
        (output_dir / "config.json").write_text(
            json.dumps({"model_type": "bert", "id2label": {"0": "O"}}),
            encoding="utf-8",
        )
        return {"model_type": "bert", "id2label": {"0": "O"}}, []

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "optimize_onnx_graph", fake_optimize)
    monkeypatch.setattr(module, "validate_optimized_onnx_export", fake_validate)
    monkeypatch.setattr(module, "export_webgpu", fake_export_webgpu)
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        optimization_config=module.OnnxOptimizationConfig(
            providers=("CUDAExecutionProvider", "CPUExecutionProvider")
        ),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert calls["export_path"] == "model.unoptimized.onnx"
    assert calls["optimize_input"] == "model.unoptimized.onnx"
    assert manifest["artifacts"][0]["path"] == "model.onnx"
    assert manifest["optimization"]["enabled"] is True
    assert manifest["validation"]["passed"] is True
    assert manifest["operator_fallbacks"][0]["op_type"] == "LayerNormalization"


def test_export_onnx_rejects_opset_below_token_classification_minimum(
    tmp_path: Path,
) -> None:
    module = _convert_module()

    with pytest.raises(ValueError, match="requires opset >= 18"):
        module.export_onnx("OpenMed/test-model", tmp_path / "model.onnx", opset=17)


def test_shape_bucket_config_uses_next_bucket_and_exact_overflow() -> None:
    module = _convert_module()
    config = module.ShapeBucketConfig(buckets=(8, 16, 32), max_length=64)

    assert config.bucket_for(9) == 16
    assert config.bucket_for(32) == 32
    assert config.bucket_for(99) == 99
    assert config.to_manifest()["overflow"] == "exact_length"


def test_operator_fallback_parser_reports_profiled_provider_change() -> None:
    module = _convert_module()

    fallbacks = module._operator_fallbacks_from_profile_events(
        [
            {
                "name": "node_kernel_time",
                "args": {
                    "provider": "CPUExecutionProvider",
                    "op_name": "Gelu",
                    "node_name": "gelu_1",
                },
            }
        ],
        preferred_provider="CUDAExecutionProvider",
    )

    assert fallbacks == [
        {
            "requested_provider": "CUDAExecutionProvider",
            "execution_provider": "CPUExecutionProvider",
            "op_type": "Gelu",
            "node_name": "gelu_1",
            "reason": "profiled_on_fallback_provider",
        }
    ]


def test_optimize_onnx_graph_uses_runtime_saved_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    input_path = tmp_path / "model.unoptimized.onnx"
    output_path = tmp_path / "model.onnx"
    input_path.write_bytes(b"raw")
    levels = {}

    onnx_mod = types.ModuleType("onnx")
    onnx_mod.load = lambda path: {"path": path}
    onnx_mod.checker = types.SimpleNamespace(check_model=lambda model: None)

    runtime_mod = types.ModuleType("onnxruntime")
    runtime_mod.GraphOptimizationLevel = types.SimpleNamespace(
        ORT_DISABLE_ALL="disable",
        ORT_ENABLE_BASIC="basic",
        ORT_ENABLE_EXTENDED="extended",
    )
    runtime_mod.get_available_providers = lambda: ["CPUExecutionProvider"]

    class SessionOptions:
        graph_optimization_level = None
        optimized_model_filepath = None

    class InferenceSession:
        def __init__(self, path, *, sess_options=None, providers=None):
            levels["level"] = sess_options.graph_optimization_level
            Path(sess_options.optimized_model_filepath).write_bytes(b"optimized")

    runtime_mod.SessionOptions = SessionOptions
    runtime_mod.InferenceSession = InferenceSession

    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime_mod)

    manifest = module.optimize_onnx_graph(input_path, output_path)

    assert output_path.read_bytes() == b"optimized"
    assert levels["level"] == "extended"
    assert manifest["backend"] == "onnxruntime-session"
    assert manifest["passes"]["attention_fusion"] is True
