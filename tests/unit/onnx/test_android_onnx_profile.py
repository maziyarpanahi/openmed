"""Tests for the Android ONNX Runtime Mobile export profile."""

from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from pathlib import Path

import pytest


def _android_module():
    return importlib.import_module("openmed.onnx.android_profile")


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def test_committed_android_contract_fixture_asserts_names_dtypes_and_dynamic_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _android_module()
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    model = _fake_android_model(nodes=("MatMul", "CustomOp"))
    checked = _install_fake_onnx(monkeypatch, model)
    caplog.set_level(logging.WARNING)

    validation = module.validate_android_profile(model_path)

    assert checked == [str(model_path)]
    assert validation.opset == module.ANDROID_ONNX_OPSET
    assert validation.inputs == (
        {
            "name": "input_ids",
            "dtype": "int64",
            "axes": ["batch", "sequence"],
            "shape": [
                {"kind": "dynamic", "name": "batch"},
                {"kind": "dynamic", "name": "sequence"},
            ],
        },
        {
            "name": "attention_mask",
            "dtype": "int64",
            "axes": ["batch", "sequence"],
            "shape": [
                {"kind": "dynamic", "name": "batch"},
                {"kind": "dynamic", "name": "sequence"},
            ],
        },
    )
    assert validation.outputs == (
        {
            "name": "logits",
            "dtype": "float32",
            "axes": ["batch", "sequence", "labels"],
            "shape": [
                {"kind": "dynamic", "name": "batch"},
                {"kind": "dynamic", "name": "sequence"},
                {"kind": "static", "value": 3},
            ],
        },
    )
    assert validation.unsupported_ops == ("CustomOp",)
    assert "CustomOp is not in the Android ONNX Runtime Mobile" in caplog.text
    assert validation.to_metadata()["profile"] == module.ANDROID_PROFILE_NAME


def test_validate_android_profile_rejects_static_sequence_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _android_module()
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _install_fake_onnx(monkeypatch, _fake_android_model(sequence_dynamic=False))

    with pytest.raises(ValueError, match="axis sequence must be dynamic"):
        module.validate_android_profile(model_path)


def test_validate_android_profile_rejects_wrong_opset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _android_module()
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _install_fake_onnx(monkeypatch, _fake_android_model(opset=17))

    with pytest.raises(ValueError, match="requires opset 18"):
        module.validate_android_profile(model_path)


def test_export_android_fp16_converts_weights_and_keeps_io_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _android_module()
    input_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model_fp16.onnx"
    input_path.write_bytes(b"onnx")
    model = _fake_android_model()
    saved = {}
    checked = _install_fake_onnx(monkeypatch, model, saved=saved)

    runtime_mod = types.ModuleType("onnxruntime")
    transformers_mod = types.ModuleType("onnxruntime.transformers")
    float16_mod = types.ModuleType("onnxruntime.transformers.float16")

    converted = {}

    def fake_convert(model_obj, keep_io_types):
        converted["model"] = model_obj
        return {"fp16": model_obj, "keep_io_types": keep_io_types}

    float16_mod.convert_float_to_float16 = fake_convert
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime.transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime.transformers.float16", float16_mod)

    result = module.export_android_fp16(input_path, output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"fp16"
    assert converted["model"] == str(input_path)
    assert saved["model"]["keep_io_types"] is True
    assert checked == [str(output_path)]


def test_convert_android_profile_records_fp32_fp16_artifacts_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    export_calls = []

    def fake_export_onnx(model_id, output_path, **kwargs):
        export_calls.append(kwargs)
        path = Path(output_path)
        path.write_bytes(b"onnx")
        return path

    def fake_export_android_fp16(onnx_path, output_path, **kwargs):
        assert Path(onnx_path).name == "model.onnx"
        path = Path(output_path)
        path.write_bytes(b"fp16")
        return path

    def fake_quantize_dynamic_int8(onnx_path, output_path, **kwargs):
        assert Path(onnx_path).name == "model.onnx"
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
        assert kwargs["require_id2label"] is True
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

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "export_android_fp16", fake_export_android_fp16)
    monkeypatch.setattr(module, "quantize_dynamic_int8", fake_quantize_dynamic_int8)
    monkeypatch.setattr(
        module, "validate_android_profile", fake_validate_android_profile
    )
    monkeypatch.setattr(
        module,
        "convert_android_onnx_to_ort",
        lambda *args, **kwargs: types.SimpleNamespace(
            skipped=True,
            ort_path=None,
            skip_reason="tooling unavailable",
        ),
    )
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(
        module,
        "export_webgpu",
        lambda *args, **kwargs: pytest.fail("android profile must not emit WebGPU"),
    )

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        profile="android",
        include_webgpu=True,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert export_calls[0]["profile"] == "android"
    assert result.formats == ["onnx-android", "onnx-int8"]
    assert manifest["formats"] == ["onnx-android", "onnx-int8"]
    assert [item["path"] for item in manifest["artifacts"]] == [
        "model.onnx",
        "model_fp16.onnx",
        "model_int8.onnx",
    ]
    assert [item["precision"] for item in manifest["artifacts"]] == [
        "float32",
        "float16",
        "int8",
    ]
    assert manifest["artifacts"][0]["format"] == "onnx-android"
    assert manifest["artifacts"][2]["format"] == "onnx-int8"
    assert manifest["artifacts"][0]["metadata"]["opset"] == module.ANDROID_ONNX_OPSET
    config = json.loads((result.output_dir / "config.json").read_text())
    assert config["id2label"] == {"0": "O", "1": "B-NAME"}


def test_export_onnx_android_profile_uses_fixed_opset_and_named_dynamic_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    output_path = tmp_path / "model.onnx"
    export_call = {}

    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = object()

    class Module:
        def eval(self) -> None:
            pass

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

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
    torch_mod.onnx = types.SimpleNamespace(export=fake_export)

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            assert len(texts) == 2
            return {
                "input_ids": [[101, 102], [101, 102]],
                "attention_mask": [[1, 1], [1, 1]],
                "token_type_ids": [[0, 0], [0, 0]],
            }

    class FakeModel(Module):
        dtype = None

        def to(self, *, dtype):
            self.dtype = dtype
            return self

    fake_model = FakeModel()

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )
    transformers_mod.AutoModelForTokenClassification = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: fake_model
    )
    onnx_mod = types.ModuleType("onnx")
    onnx_mod.load = lambda path, **kwargs: types.SimpleNamespace(
        graph=types.SimpleNamespace(initializer=[])
    )
    onnx_mod.checker = types.SimpleNamespace(check_model=lambda model: None)

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)

    module.export_onnx(
        "OpenMed/test-model",
        output_path,
        max_seq_length=32,
        profile="android",
    )

    assert export_call["opset_version"] == module.ANDROID_ONNX_OPSET
    assert fake_model.dtype is torch_mod.float32
    assert export_call["dynamo"] is False
    assert export_call["dynamic_shapes"] is None
    assert export_call["input_names"] == [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]
    assert export_call["output_names"] == ["logits"]
    assert export_call["dynamic_axes"] == {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "token_type_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }


def test_save_source_assets_populates_id2label_for_android(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()

    class FakeConfig:
        def to_dict(self):
            return {"model_type": "bert", "num_labels": 2}

    class FakeTokenizer:
        def save_pretrained(self, output_dir):
            Path(output_dir, "tokenizer.json").write_text("{}", encoding="utf-8")

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoConfig = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeConfig()
    )
    transformers_mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setattr(
        module,
        "get_tokenizer_with_loader",
        lambda model_id, loader, cache_dir=None: FakeTokenizer(),
    )

    config, tokenizer_files = module.save_source_assets(
        "OpenMed/test-model",
        tmp_path,
        require_id2label=True,
    )

    assert config["id2label"] == {"0": "LABEL_0", "1": "LABEL_1"}
    assert json.loads((tmp_path / "config.json").read_text())["id2label"] == {
        "0": "LABEL_0",
        "1": "LABEL_1",
    }
    assert "tokenizer.json" in tokenizer_files


def test_save_source_assets_requires_fast_tokenizer_for_android(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()

    class FakeConfig:
        def to_dict(self):
            return {"model_type": "bert", "num_labels": 1}

    class MissingFastTokenizer:
        def save_pretrained(self, output_dir):
            Path(output_dir, "tokenizer_config.json").write_text(
                "{}",
                encoding="utf-8",
            )

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoConfig = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeConfig()
    )
    transformers_mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: MissingFastTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setattr(
        module,
        "get_tokenizer_with_loader",
        lambda model_id, loader, cache_dir=None: MissingFastTokenizer(),
    )

    with pytest.raises(RuntimeError, match="requires tokenizer.json"):
        module.save_source_assets(
            "OpenMed/test-model",
            tmp_path,
            require_id2label=True,
            require_tokenizer_json=True,
        )


def _install_fake_onnx(
    monkeypatch: pytest.MonkeyPatch,
    model,
    *,
    saved: dict | None = None,
) -> list:
    checked = []
    onnx_mod = types.ModuleType("onnx")
    onnx_mod.load = lambda path, **kwargs: model
    onnx_mod.checker = types.SimpleNamespace(
        check_model=lambda model_obj: checked.append(model_obj)
    )

    def fake_save(model_obj, path):
        if saved is not None:
            saved["model"] = model_obj
            saved["path"] = path
        Path(path).write_bytes(b"fp16")

    onnx_mod.save = fake_save
    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)
    return checked


def _fake_android_model(
    *,
    opset: int = 18,
    nodes: tuple[str, ...] = ("MatMul", "Add"),
    sequence_dynamic: bool = True,
):
    sequence_dim = _dim("sequence") if sequence_dynamic else _dim(value=512)
    return types.SimpleNamespace(
        opset_import=[types.SimpleNamespace(domain="", version=opset)],
        graph=types.SimpleNamespace(
            initializer=[],
            input=[
                _value_info("input_ids", [_dim("batch"), sequence_dim], elem_type=7),
                _value_info(
                    "attention_mask",
                    [_dim("batch"), sequence_dim],
                    elem_type=7,
                ),
            ],
            output=[
                _value_info(
                    "logits",
                    [_dim("batch"), sequence_dim, _dim(value=3)],
                    elem_type=1,
                )
            ],
            node=[types.SimpleNamespace(op_type=node, domain="") for node in nodes],
        ),
    )


def _value_info(name: str, dims, *, elem_type: int):
    return types.SimpleNamespace(
        name=name,
        type=types.SimpleNamespace(
            tensor_type=types.SimpleNamespace(
                elem_type=elem_type,
                shape=types.SimpleNamespace(dim=dims),
            ),
        ),
    )


def _dim(name: str = "", *, value: int = 0):
    return types.SimpleNamespace(dim_param=name, dim_value=value)
