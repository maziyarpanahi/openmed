"""Tests for family-aware PyTorch export adapters."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.eval.export_qualification import ExportQualificationError, ExportReason
from openmed.torch import export_adapters
from openmed.torch.export_adapters import (
    MODERNBERT_EXPORT_ADAPTER,
    modernbert_sliding_window,
    registered_export_families,
    resolve_export_adapter,
)


@pytest.fixture
def transformers_version(monkeypatch: pytest.MonkeyPatch):
    def install(version: str | None) -> None:
        def fake_version(name: str) -> str:
            if version is None:
                raise metadata.PackageNotFoundError(name)
            return version

        monkeypatch.setattr(
            export_adapters,
            "metadata",
            SimpleNamespace(
                version=fake_version,
                PackageNotFoundError=metadata.PackageNotFoundError,
            ),
        )

    return install


def _config(**overrides):
    values = {"max_position_embeddings": 512, "local_attention": 128}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_modernbert_aliases_resolve_to_one_adapter() -> None:
    for model_type in ("modernbert", "ModernBert", "modern-bert", "modern_bert"):
        assert resolve_export_adapter(model_type) is MODERNBERT_EXPORT_ADAPTER
    for model_type in (None, "", "bert", "longformer"):
        assert resolve_export_adapter(model_type) is None
    assert "modernbert" in registered_export_families()
    description = MODERNBERT_EXPORT_ADAPTER.describe()
    assert description["support"]["coreml"]["shape_modes"] == ["static"]
    assert description["support"]["onnx"]["providers"] == ["CPUExecutionProvider"]


def test_sliding_window_follows_transformers_convention() -> None:
    assert modernbert_sliding_window({"sliding_window": 64}) == 64
    assert modernbert_sliding_window({"local_attention": 128}) == 64


def test_validate_request_rejects_unsupported_combinations(transformers_version):
    transformers_version("5.12.1")
    adapter = MODERNBERT_EXPORT_ADAPTER
    accepted = adapter.validate_request(
        runtime="coreml",
        config=_config(),
        max_sequence_length=512,
        precisions=("float16",),
        quantizations=("none", "int8", "int4"),
        providers=("cpuAndNeuralEngine",),
        shape_mode="static",
    )
    assert accepted.runtime == "coreml"

    cases = (
        ({"runtime": "tflite"}, ExportReason.UNSUPPORTED_RUNTIME),
        ({"runtime": "coreml", "shape_mode": "dynamic"}, "unsupported_sequence_shape"),
        ({"runtime": "onnx", "quantizations": ("int4",)}, "unsupported_quantization"),
        (
            {"runtime": "onnx", "providers": ("CoreMLExecutionProvider",)},
            "unsupported_provider",
        ),
        ({"runtime": "onnx", "precisions": ("bfloat16",)}, "unsupported_precision"),
        (
            {"runtime": "onnx", "max_sequence_length": 513},
            "unsupported_sequence_length",
        ),
    )
    for overrides, reason in cases:
        kwargs = {"config": _config(), "max_sequence_length": 512, **overrides}
        with pytest.raises(ExportQualificationError) as error:
            adapter.validate_request(**kwargs)
        assert error.value.reason_code == ExportReason(reason).value


@pytest.mark.parametrize("version", ["4.57.1", None, "dev"])
def test_validate_request_rejects_unqualified_transformers(
    transformers_version, version
) -> None:
    transformers_version(version)
    with pytest.raises(ExportQualificationError) as error:
        MODERNBERT_EXPORT_ADAPTER.validate_request(
            runtime="onnx", config=_config(), max_sequence_length=128
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT


def _real_dependencies():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    if int(transformers.__version__.split(".", 1)[0]) < 5:
        pytest.skip("ModernBERT export adapters require transformers>=5")
    return torch, transformers


def _stub_model(tmp_path: Path):
    torch, transformers = _real_dependencies()
    from tests.fixtures.modernbert_export_stub import build_tiny_modernbert

    checkpoint = build_tiny_modernbert(tmp_path / "checkpoint")
    reference = transformers.AutoModelForTokenClassification.from_pretrained(
        checkpoint
    ).eval()
    traced = transformers.AutoModelForTokenClassification.from_pretrained(
        checkpoint
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    return torch, reference, traced, tokenizer


def test_wrapper_reproduces_reference_logits_with_padding(tmp_path: Path) -> None:
    torch, reference, model, tokenizer = _stub_model(tmp_path)
    adapter = MODERNBERT_EXPORT_ADAPTER
    adapter.prepare_model(model)
    assert model.config._attn_implementation == "eager"
    wrapper = adapter.build_wrapper(torch, model)
    texts = ["patient ZORVAK QUILLAN, mrn 44172290.", "record 88310 seen."]
    batch = tokenizer(texts, padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        expected = reference(**batch).logits
        actual = wrapper(batch["input_ids"], batch["attention_mask"])
        mask = batch["attention_mask"].bool()
        assert (expected - actual).abs()[mask].max().item() < 1e-4

        wrapper.sliding_window = 10_000
        broken = wrapper(batch["input_ids"], batch["attention_mask"])
        assert (expected - broken).abs()[mask].max().item() > 1e-2


def test_trace_context_is_scoped_and_removes_runtime_int_casts(
    tmp_path: Path,
) -> None:
    torch, _, model, tokenizer = _stub_model(tmp_path)
    from transformers.models.modernbert import modeling_modernbert

    adapter = MODERNBERT_EXPORT_ADAPTER
    adapter.prepare_model(model)
    wrapper = adapter.build_wrapper(torch, model)
    sample = tokenizer(["patient ZORVAK"], padding="max_length", max_length=32)
    inputs = (
        torch.tensor(sample["input_ids"]),
        torch.tensor(sample["attention_mask"]),
    )
    original = modeling_modernbert.rotate_half
    with torch.no_grad():
        with adapter.trace_context(torch, model):
            assert modeling_modernbert.rotate_half is not original
            traced = torch.jit.trace(wrapper, inputs)
        stock = torch.jit.trace(wrapper, inputs)
    assert modeling_modernbert.rotate_half is original
    assert "aten::floor_divide" not in str(traced.inlined_graph)
    assert "aten::floor_divide" in str(stock.inlined_graph)
    with torch.no_grad():
        assert torch.allclose(traced(*inputs), wrapper(*inputs), atol=1e-6)
