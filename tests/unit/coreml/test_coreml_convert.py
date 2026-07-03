"""Tests for the CoreML conversion script."""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

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
        assert "conversion_manifest_path" in params
        assert "eval_suite_path" in params
        assert "swift_parity_corpus_path" in params

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


@pytest.mark.parametrize(
    "model_type",
    ["bert", "distilbert", "electra", "roberta", "xlm-roberta", "deberta-v2"],
)
def test_resolve_supported_model_type_accepts_supported_families(model_type: str):
    from openmed.coreml.convert import resolve_supported_model_type

    config = types.SimpleNamespace(model_type=model_type, architectures=[])

    assert resolve_supported_model_type(config) == model_type


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
    assert float_metadata["ane_optimization_profile"] == "static-rank2-fp16"

    assert json.loads((tmp_path / "privacy_id2label.json").read_text()) == {
        "0": "O",
        "1": "B-NAME",
    }
    assert json.loads((tmp_path / "privacy_int8_id2label.json").read_text()) == {
        "0": "O",
        "1": "B-NAME",
    }
    assert state.config_loads == 1
    assert state.tokenizer_loads == 1
    assert state.model_loads == 1

    manifest = json.loads((tmp_path / "privacy_coreml_manifest.json").read_text())
    assert manifest["format"] == "openmed-coreml"
    assert [variant["name"] for variant in manifest["variants"]] == [
        "coreml-fp16",
        "coreml-int8",
    ]
    assert manifest["variants"][0]["precision"] == "float16"
    assert manifest["variants"][0]["latency_ms"]["measured"] is False
    assert manifest["variants"][0]["residency"]["source"] == "missing_compute_plan"


def test_convert_uses_custom_quantized_output_path(monkeypatch, tmp_path):
    from openmed.coreml.convert import convert

    _install_conversion_stubs(
        monkeypatch,
        model_type="bert",
        architectures=["BertForTokenClassification"],
    )
    output_path = tmp_path / "privacy.mlpackage"
    int8_path = tmp_path / "custom-int8.mlpackage"

    convert(
        "OpenMed/tiny-stub",
        output_path,
        max_seq_length=16,
        quantize="int8",
        quantized_output_path=int8_path,
    )

    assert output_path.is_dir()
    assert int8_path.is_dir()
    assert json.loads((tmp_path / "custom-int8_id2label.json").read_text()) == {
        "0": "O",
        "1": "B-NAME",
    }


def test_convert_emits_int8_and_int4_variants(monkeypatch, tmp_path):
    from openmed.coreml.convert import convert

    state = _install_conversion_stubs(
        monkeypatch,
        model_type="bert",
        architectures=["BertForTokenClassification"],
    )
    output_path = tmp_path / "privacy.mlpackage"

    convert(
        "OpenMed/tiny-stub",
        output_path,
        max_seq_length=16,
        quantize="all",
    )

    assert (tmp_path / "privacy_int8.mlpackage").is_dir()
    assert (tmp_path / "privacy_int4.mlpackage").is_dir()
    assert [config.nbits for config in state.palettizer_configs] == [8, 4]

    manifest = json.loads((tmp_path / "privacy_coreml_manifest.json").read_text())
    assert [variant["name"] for variant in manifest["variants"]] == [
        "coreml-fp16",
        "coreml-int8",
        "coreml-int4",
    ]
    assert manifest["variants"][2]["quantization"] == "int4"


def test_analyze_ane_residency_flags_cpu_fallback(tmp_path):
    from openmed.coreml.convert import analyze_ane_residency

    compiled = tmp_path / "compiled.mlmodelc"
    compiled.mkdir()
    (compiled / "compute_plan.json").write_text(
        json.dumps(
            {
                "operations": [
                    {"name": "attention/q", "compute_unit": "ANE", "flops": 45},
                    {
                        "name": "attention/k",
                        "compute_unit": "NeuralEngine",
                        "flops": 45,
                    },
                    {"name": "classifier", "compute_unit": "CPU", "flops": 10},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_ane_residency(compiled)

    assert report.ane_residency_percentage == pytest.approx(0.90)
    assert report.cpu_fallback_layers[0].name == "classifier"
    assert report.passed is False


def test_write_coreml_variant_parity_report_rejects_int4(tmp_path):
    from openmed.coreml.convert import write_coreml_variant_parity_report

    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    paths = {
        "coreml-fp16": tmp_path / "fp16.mlpackage",
        "coreml-int8": tmp_path / "int8.mlpackage",
        "coreml-int4": tmp_path / "int4.mlpackage",
    }
    for path in paths.values():
        path.mkdir()

    report = write_coreml_variant_parity_report(
        source_model_id="OpenMed/stub",
        variant_paths=paths,
        eval_suite_path=fixture_path,
        output_path=tmp_path / "coreml_parity.json",
        swift_corpus_path=tmp_path / "swift_parity.json",
        parent_runner=_perfect_runner,
        candidate_runners={
            "coreml-fp16": _perfect_runner,
            "coreml-int8": _perfect_runner,
            "coreml-int4": _miss_runner,
        },
    )

    by_format = {variant["format"]: variant for variant in report["variants"]}
    assert by_format["coreml-fp16"]["passed"] is True
    assert by_format["coreml-int8"]["passed"] is True
    assert by_format["coreml-int4"]["passed"] is False
    assert by_format["coreml-int4"]["auto_rejected"] is True

    swift_payload = json.loads((tmp_path / "swift_parity.json").read_text())
    assert "text" not in swift_payload["fixtures"][0]
    assert swift_payload["fixtures"][0]["text_sha256"]


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

    assert state.config_loads == 1
    assert state.tokenizer_loads == 0
    assert state.model_loads == 0
    assert state.convert_kwargs is None
    assert not output_path.exists()
    assert not (tmp_path / "unsupported_int8.mlpackage").exists()


def test_convert_rejects_invalid_compute_precision_before_loading(
    monkeypatch,
    tmp_path,
):
    from openmed.coreml.convert import convert

    state = _install_conversion_stubs(
        monkeypatch,
        model_type="bert",
        architectures=["BertForTokenClassification"],
    )

    with pytest.raises(ValueError, match="compute_precision"):
        convert(
            "OpenMed/tiny-stub",
            tmp_path / "privacy.mlpackage",
            compute_precision="bfloat16",
        )

    assert state.config_loads == 0
    assert state.tokenizer_loads == 0
    assert state.model_loads == 0
    assert state.convert_kwargs is None


def test_convert_rejects_invalid_compute_units_before_loading(monkeypatch, tmp_path):
    from openmed.coreml.convert import convert

    state = _install_conversion_stubs(
        monkeypatch,
        model_type="bert",
        architectures=["BertForTokenClassification"],
    )

    with pytest.raises(ValueError, match="compute_units"):
        convert(
            "OpenMed/tiny-stub",
            tmp_path / "privacy.mlpackage",
            compute_units="gpuOnly",
        )

    assert state.config_loads == 0
    assert state.tokenizer_loads == 0
    assert state.model_loads == 0
    assert state.convert_kwargs is None


def test_convert_rejects_quantized_output_without_quantize_before_loading(
    monkeypatch,
    tmp_path,
):
    from openmed.coreml.convert import convert

    state = _install_conversion_stubs(
        monkeypatch,
        model_type="bert",
        architectures=["BertForTokenClassification"],
    )

    with pytest.raises(ValueError, match="quantized_output_path requires"):
        convert(
            "OpenMed/tiny-stub",
            tmp_path / "privacy.mlpackage",
            quantized_output_path=tmp_path / "privacy-int8.mlpackage",
        )

    assert state.config_loads == 0
    assert state.tokenizer_loads == 0
    assert state.model_loads == 0
    assert state.convert_kwargs is None


def test_convert_reports_missing_coreml_optimize_for_int8(monkeypatch, tmp_path):
    from openmed.coreml.convert import convert

    _install_conversion_stubs(
        monkeypatch,
        model_type="bert",
        architectures=["BertForTokenClassification"],
        include_optimizer=False,
    )
    output_path = tmp_path / "privacy.mlpackage"

    with pytest.raises(ImportError, match="coremltools.optimize.coreml"):
        convert("OpenMed/tiny-stub", output_path, quantize="int8")

    assert output_path.is_dir()
    assert not (tmp_path / "privacy_int8.mlpackage").exists()


def _install_conversion_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_type: str,
    architectures: list[str],
    include_optimizer: bool = True,
):
    state = types.SimpleNamespace(
        config_loads=0,
        convert_kwargs=None,
        model_loads=0,
        palettizer_config=None,
        palettizer_configs=[],
        tokenizer_loads=0,
    )

    class FakeConfig:
        def __init__(self):
            self.model_type = model_type
            self.architectures = architectures
            self.num_labels = 2
            self.id2label = {0: "O", 1: "B-NAME"}

    class FakeTorchModule:
        def eval(self):
            return self

    class FakeModel(FakeTorchModule):
        def __init__(self):
            self.config = FakeConfig()

        def __call__(self, **_kwargs):
            return types.SimpleNamespace(logits="logits")

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
            }

    def load_config(*_args, **_kwargs):
        state.config_loads += 1
        return FakeConfig()

    def load_tokenizer(*_args, **_kwargs):
        state.tokenizer_loads += 1
        return FakeTokenizer()

    def load_model(*_args, **_kwargs):
        state.model_loads += 1
        return FakeModel()

    torch_module = types.ModuleType("torch")
    torch_module.nn = types.SimpleNamespace(Module=FakeTorchModule)
    torch_module.jit = types.SimpleNamespace(
        trace=lambda wrapper, sample: {
            "wrapper": wrapper,
            "sample": sample,
        }
    )

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = types.SimpleNamespace(from_pretrained=load_config)
    transformers_module.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=load_tokenizer
    )
    transformers_module.AutoModelForTokenClassification = types.SimpleNamespace(
        from_pretrained=load_model
    )

    coremltools_module = _fake_coremltools_module(
        state,
        include_optimizer=include_optimizer,
    )

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "coremltools", coremltools_module)
    return state


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


def _fake_coremltools_module(state, *, include_optimizer: bool = True):
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
        state.palettizer_configs.append(config.global_config)
        return _FakeCoreMLModel(kind="int8", source=mlmodel)

    coremltools_module.convert = fake_convert
    if include_optimizer:
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
