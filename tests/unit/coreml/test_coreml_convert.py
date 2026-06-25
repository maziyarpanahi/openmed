"""Tests for the CoreML conversion script."""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import pytest


class TestCoreMLConvertModule:
    """Verify the coreml.convert module is importable and has expected API."""

    def test_module_importable(self):
        from openmed.coreml import convert

        assert hasattr(convert, "convert")
        assert hasattr(convert, "main")

    def test_convert_signature(self):
        """convert() should accept model_id, output_path, and options."""
        from openmed.coreml.convert import convert

        sig = inspect.signature(convert)
        params = list(sig.parameters.keys())
        assert "model_id" in params
        assert "output_path" in params
        assert "max_seq_length" in params
        assert "compute_precision" in params
        assert "compute_units" in params
        assert "quantize" in params
        assert "quantized_output_path" in params

    def test_main_exists(self):
        from openmed.coreml.convert import main

        assert callable(main)


def test_resolve_supported_model_type_accepts_architecture_hints():
    from openmed.coreml.convert import resolve_supported_model_type

    config = types.SimpleNamespace(
        model_type=None,
        architectures=["DebertaV2ForTokenClassification"],
    )

    assert resolve_supported_model_type(config) == "deberta-v2"


def test_convert_emits_float16_and_int8_packages(monkeypatch, tmp_path):
    from openmed.coreml.convert import convert

    state = _install_conversion_stubs(
        monkeypatch,
        model_type="distilbert",
        architectures=["DistilBertForTokenClassification"],
    )
    output_path = tmp_path / "privacy.mlpackage"

    result = convert(
        "OpenMed/tiny-stub",
        output_path,
        max_seq_length=16,
        compute_units="cpuAndNeuralEngine",
        quantize="int8",
    )

    int8_path = tmp_path / "privacy_int8.mlpackage"
    assert result == output_path
    assert output_path.is_dir()
    assert int8_path.is_dir()
    assert _artifact_size(int8_path) < _artifact_size(output_path)

    assert state.convert_kwargs["compute_precision"] == "FLOAT16"
    assert state.convert_kwargs["compute_units"] == "CPU_AND_NE"
    assert state.palettizer_config.mode == "kmeans"
    assert state.palettizer_config.nbits == 8

    float_metadata = _metadata(output_path)
    int8_metadata = _metadata(int8_path)
    assert json.loads(float_metadata["id2label"]) == {"0": "O", "1": "B-NAME"}
    assert float_metadata["source_model_type"] == "distilbert"
    assert float_metadata["compute_units"] == "cpuAndNeuralEngine"
    assert float_metadata["quantization"] == "none"
    assert int8_metadata["compute_units"] == "cpuAndNeuralEngine"
    assert int8_metadata["quantization"] == "int8"

    assert json.loads((tmp_path / "privacy_id2label.json").read_text()) == {
        "0": "O",
        "1": "B-NAME",
    }
    assert json.loads((tmp_path / "privacy_int8_id2label.json").read_text()) == {
        "0": "O",
        "1": "B-NAME",
    }


def test_convert_rejects_unsupported_architecture(monkeypatch, tmp_path):
    from openmed.coreml.convert import convert

    state = _install_conversion_stubs(
        monkeypatch,
        model_type="longformer",
        architectures=["LongformerForTokenClassification"],
    )
    output_path = tmp_path / "unsupported.mlpackage"

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported CoreML token-classification architecture "
            ".*Supported families: bert, distilbert, electra, roberta, "
            "xlm-roberta, deberta-v2"
        ),
    ):
        convert("OpenMed/unsupported-stub", output_path, quantize="int8")

    assert state.convert_kwargs is None
    assert not output_path.exists()
    assert not (tmp_path / "unsupported_int8.mlpackage").exists()


def _install_conversion_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_type: str,
    architectures: list[str],
):
    state = types.SimpleNamespace(
        convert_kwargs=None,
        palettizer_config=None,
    )

    class FakeTorchModule:
        def eval(self):
            return self

    class FakeModel(FakeTorchModule):
        def __init__(self):
            self.config = types.SimpleNamespace(
                model_type=model_type,
                architectures=architectures,
                num_labels=2,
                id2label={0: "O", 1: "B-NAME"},
            )

        def __call__(self, **_kwargs):
            return types.SimpleNamespace(logits="logits")

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
            }

    torch_module = types.ModuleType("torch")
    torch_module.nn = types.SimpleNamespace(Module=FakeTorchModule)
    torch_module.jit = types.SimpleNamespace(
        trace=lambda wrapper, sample: {
            "wrapper": wrapper,
            "sample": sample,
        }
    )

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: FakeTokenizer()
    )
    transformers_module.AutoModelForTokenClassification = types.SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: FakeModel()
    )

    coremltools_module = _fake_coremltools_module(state)

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "coremltools", coremltools_module)
    return state


def _fake_coremltools_module(state):
    coremltools_module = types.ModuleType("coremltools")
    coremltools_module.precision = types.SimpleNamespace(
        FLOAT16="FLOAT16",
        FLOAT32="FLOAT32",
    )
    coremltools_module.ComputeUnit = types.SimpleNamespace(
        ALL="ALL",
        CPU_AND_NE="CPU_AND_NE",
        CPU_ONLY="CPU_ONLY",
    )
    coremltools_module.target = types.SimpleNamespace(iOS16="iOS16")
    coremltools_module.RangeDim = _FakeRangeDim
    coremltools_module.Shape = _FakeShape
    coremltools_module.TensorType = _FakeTensorType

    def fake_convert(_traced, **kwargs):
        state.convert_kwargs = kwargs
        return _FakeCoreMLModel(kind="float16")

    def fake_palettize_weights(mlmodel, *, config):
        state.palettizer_config = config.global_config
        return _FakeCoreMLModel(kind="int8", source=mlmodel)

    coremltools_module.convert = fake_convert
    coremltools_module.optimize = types.SimpleNamespace(
        coreml=types.SimpleNamespace(
            OpPalettizerConfig=_FakeOpPalettizerConfig,
            OptimizationConfig=_FakeOptimizationConfig,
            palettize_weights=fake_palettize_weights,
        )
    )
    return coremltools_module


class _FakeRangeDim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeShape:
    def __init__(self, *, shape):
        self.shape = shape


class _FakeTensorType:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeCoreMLModel:
    def __init__(self, *, kind: str, source=None):
        self.kind = kind
        self.short_description = getattr(source, "short_description", "")
        self.author = getattr(source, "author", "")
        self.license = getattr(source, "license", "")
        self.user_defined_metadata = dict(getattr(source, "user_defined_metadata", {}))

    def save(self, path: str):
        package = Path(path)
        package.mkdir(parents=True, exist_ok=True)
        payload_size = 4096 if self.kind == "float16" else 1024
        (package / "weights.bin").write_bytes(b"0" * payload_size)
        (package / "metadata.json").write_text(
            json.dumps(
                {
                    "kind": self.kind,
                    "short_description": self.short_description,
                    "author": self.author,
                    "license": self.license,
                    "user_defined_metadata": self.user_defined_metadata,
                },
                sort_keys=True,
            )
        )


class _FakeOpPalettizerConfig:
    def __init__(self, *, mode: str, nbits: int):
        self.mode = mode
        self.nbits = nbits


class _FakeOptimizationConfig:
    def __init__(self, *, global_config):
        self.global_config = global_config


def _metadata(path: Path) -> dict[str, str]:
    payload = json.loads((path / "metadata.json").read_text())
    return payload["user_defined_metadata"]


def _artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
