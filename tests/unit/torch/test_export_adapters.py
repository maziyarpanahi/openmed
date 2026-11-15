"""Tests for family-aware PyTorch export adapters."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.eval.export_qualification import ExportQualificationError, ExportReason
from openmed.torch import export_adapters
from openmed.torch.export_adapters import (
    LONGFORMER_EXPORT_ADAPTER,
    MODERNBERT_EXPORT_ADAPTER,
    longformer_attention_bytes,
    longformer_attention_windows,
    longformer_position_limit,
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
    for model_type in (None, "", "bert", "gpt2"):
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


# ---------------------------------------------------------------------------
# Longformer
# ---------------------------------------------------------------------------


def _longformer_config(**overrides):
    """A longformer-base-4096-shaped configuration (no weights)."""
    values = {
        "model_type": "longformer",
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "attention_window": 512,
        "max_position_embeddings": 4098,
        "pad_token_id": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_longformer_adapter_declares_inputs_policies_and_support() -> None:
    assert resolve_export_adapter("longformer") is LONGFORMER_EXPORT_ADAPTER
    assert {"modernbert", "longformer"} <= set(registered_export_families())
    adapter = LONGFORMER_EXPORT_ADAPTER
    assert adapter.inputs_for("coreml") == (
        "input_ids",
        "attention_mask",
        "global_attention_mask",
    )
    assert adapter.inputs_for("onnx") == adapter.inputs_for("coreml")
    # The Android profile keeps the two-input mobile contract.
    assert adapter.inputs_for("onnx", "android") == ("input_ids", "attention_mask")
    description = adapter.describe()
    assert description["global_attention_policies"] == ["none", "first_token"]
    assert description["support"]["coreml"]["shape_modes"] == ["static"]
    assert description["support"]["onnx"]["quantizations"] == ["none", "int8"]
    # Graphs that kept the stock data-dependent gathers fail before inference.
    adapter.check_operators(["ai.onnx:MatMul", "ai.onnx:Softmax"])
    with pytest.raises(ExportQualificationError) as error:
        adapter.check_operators(["ai.onnx:MatMul", "ai.onnx:NonZero"])
    assert error.value.reason is ExportReason.UNSUPPORTED_OPERATOR
    MODERNBERT_EXPORT_ADAPTER.check_operators(["ai.onnx:NonZero"])
    assert adapter.windowed_documents is True


def test_longformer_attention_windows_accept_int_and_per_layer_lists() -> None:
    assert longformer_attention_windows(_longformer_config()) == (512,) * 12
    assert longformer_attention_windows(
        _longformer_config(num_hidden_layers=3, attention_window=[8, 16, 8])
    ) == (8, 16, 8)
    for bad in ([8, 16], 7, 0, [8, 9, 8]):
        with pytest.raises(ExportQualificationError) as error:
            longformer_attention_windows(
                _longformer_config(num_hidden_layers=3, attention_window=bad)
            )
        assert error.value.reason is ExportReason.IO_CONTRACT_MISMATCH


def test_longformer_limits_bound_buckets_by_positions_and_memory() -> None:
    config = _longformer_config()
    assert longformer_position_limit(config) == 4096
    coreml_fp16 = LONGFORMER_EXPORT_ADAPTER.limits_for(config, "coreml", "float16")
    assert coreml_fp16.buckets == (512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
    # 4 dense [heads, S, S] tensors per layer: heads * S^2 * 2 bytes * 4.
    assert coreml_fp16.bucket_memory_bytes[2048] == 4 * 12 * 2048**2 * 2
    assert coreml_fp16.memory_ceiling_bytes == 512 * 1024**2
    assert coreml_fp16.supported_buckets == (512, 1024, 1536, 2048)
    coreml_fp32 = LONGFORMER_EXPORT_ADAPTER.limits_for(config, "coreml", "float32")
    assert coreml_fp32.max_supported_length == 1536
    onnx_fp32 = LONGFORMER_EXPORT_ADAPTER.limits_for(config, "onnx", "float32")
    assert onnx_fp32.max_supported_length == 3072
    payload = coreml_fp16.to_dict(selected=1024)
    assert payload["selected_max_sequence_length"] == 1024
    assert payload["selected_estimated_bytes"] == longformer_attention_bytes(
        config, 1024, "float16"
    )
    assert payload["supported_buckets"] == [512, 1024, 1536, 2048]
    assert set(payload["estimated_bytes_by_bucket"]) == {
        str(bucket) for bucket in coreml_fp16.buckets
    }


def test_longformer_validate_request_rejects_unsafe_combinations(
    transformers_version,
) -> None:
    transformers_version("5.5.0")
    adapter = LONGFORMER_EXPORT_ADAPTER
    config = _longformer_config()
    accepted = adapter.validate_request(
        runtime="coreml",
        config=config,
        max_sequence_length=2048,
        precisions=("float16",),
        quantizations=("none", "int8", "int4"),
        providers=("cpuAndNeuralEngine",),
        shape_mode="static",
        global_attention="first_token",
    )
    assert accepted.runtime == "coreml"
    cases = (
        ({"max_sequence_length": 4097}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH),
        ({"max_sequence_length": 1000}, ExportReason.BUCKET_UNAVAILABLE),
        ({"max_sequence_length": 4096}, ExportReason.MEMORY_BUDGET_EXCEEDED),
        (
            {"precisions": ("float32",), "max_sequence_length": 2048},
            ExportReason.MEMORY_BUDGET_EXCEEDED,
        ),
        ({"shape_mode": "dynamic"}, ExportReason.UNSUPPORTED_SEQUENCE_SHAPE),
        ({"global_attention": "all_tokens"}, ExportReason.UNSUPPORTED_GLOBAL_ATTENTION),
        ({"precisions": ("bfloat16",)}, ExportReason.UNSUPPORTED_PRECISION),
        (
            {
                "runtime": "onnx",
                "shape_mode": "dynamic",
                "providers": ("CUDAExecutionProvider",),
            },
            ExportReason.UNSUPPORTED_PROVIDER,
        ),
        (
            {"runtime": "onnx", "shape_mode": "dynamic", "quantizations": ("int4",)},
            ExportReason.UNSUPPORTED_QUANTIZATION,
        ),
        (
            {
                "runtime": "onnx",
                "shape_mode": "dynamic",
                "profile": "android",
                "global_attention": "first_token",
            },
            ExportReason.UNSUPPORTED_GLOBAL_ATTENTION,
        ),
    )
    for overrides, reason in cases:
        kwargs = {
            "runtime": "coreml",
            "config": config,
            "max_sequence_length": 512,
            "precisions": ("float16",),
            "shape_mode": "static",
            **overrides,
        }
        with pytest.raises(ExportQualificationError) as error:
            adapter.validate_request(**kwargs)
        assert error.value.reason is reason, overrides
        assert "512" not in str(error.value)
    # ONNX fp32 allows longer buckets than Core ML fp16.
    adapter.validate_request(
        runtime="onnx", config=config, max_sequence_length=3072, shape_mode="dynamic"
    )


@pytest.mark.parametrize("version", ["3.36.2", None])
def test_longformer_rejects_unqualified_transformers(
    transformers_version, version
) -> None:
    transformers_version(version)
    with pytest.raises(ExportQualificationError) as error:
        LONGFORMER_EXPORT_ADAPTER.validate_request(
            runtime="onnx", config=_longformer_config(), max_sequence_length=512
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT


def test_global_attention_resolution_is_explicit() -> None:
    adapter = LONGFORMER_EXPORT_ADAPTER
    assert adapter.resolve_global_attention(None, runtime="coreml") == "none"
    assert (
        adapter.resolve_global_attention("first_token", runtime="onnx") == "first_token"
    )
    assert (
        adapter.resolve_global_attention(None, runtime="onnx", profile="android")
        == "none"
    )
    assert (
        MODERNBERT_EXPORT_ADAPTER.resolve_global_attention(None, runtime="onnx") is None
    )
    with pytest.raises(ExportQualificationError) as error:
        MODERNBERT_EXPORT_ADAPTER.resolve_global_attention("none", runtime="onnx")
    assert error.value.reason is ExportReason.UNSUPPORTED_GLOBAL_ATTENTION


def _longformer_stub(tmp_path: Path):
    # The Longformer wrapper only reads module weights, so Transformers 4.x
    # is fine here (unlike ModernBERT).
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tests.fixtures.longformer_export_stub import build_tiny_longformer

    checkpoint = build_tiny_longformer(tmp_path / "longformer")
    reference = transformers.AutoModelForTokenClassification.from_pretrained(
        checkpoint
    ).eval()
    traced = transformers.AutoModelForTokenClassification.from_pretrained(
        checkpoint
    ).eval()
    return torch, reference, traced


def _longformer_batch(torch, total: int, real: int, globals_: tuple[int, ...]):
    generator = torch.Generator().manual_seed(total * 101 + real)
    ids = torch.randint(5, 60, (2, total), generator=generator)
    mask = torch.zeros(2, total, dtype=torch.long)
    mask[:, :real] = 1
    ids[:, real:] = 0
    glob = torch.zeros_like(mask)
    for position in globals_:
        glob[:, position] = 1
    return ids, mask, glob


# (bucket, real tokens): minimum, typical, window boundaries (4, 8, 16 tokens),
# bucket boundaries, and the declared maximum.
LONGFORMER_LENGTH_CASES = (
    (16, 3),
    (16, 5),
    (16, 8),
    (16, 9),
    (32, 17),
    (48, 33),
    (48, 47),
    (64, 49),
    (64, 64),
)


def test_longformer_wrapper_matches_reference_across_lengths_and_globals(
    tmp_path: Path,
) -> None:
    torch, reference, model = _longformer_stub(tmp_path)
    wrapper = LONGFORMER_EXPORT_ADAPTER.build_wrapper(torch, model)
    worst = 0.0
    with torch.no_grad():
        for total, real in LONGFORMER_LENGTH_CASES:
            probes = ((), (0,), (0, real // 2, real - 1))
            for globals_ in probes:
                ids, mask, glob = _longformer_batch(torch, total, real, globals_)
                expected = reference(
                    input_ids=ids, attention_mask=mask, global_attention_mask=glob
                ).logits
                actual = wrapper(ids, mask, glob)
                keep = mask.bool()
                worst = max(worst, (expected - actual).abs()[keep].max().item())
            # Without a global mask the graph is the stock local-only model.
            ids, mask, _ = _longformer_batch(torch, total, real, ())
            expected = reference(input_ids=ids, attention_mask=mask).logits
            actual = wrapper(ids, mask)
            worst = max(worst, (expected - actual).abs()[mask.bool()].max().item())
    assert worst < 1e-4


def test_longformer_wrapper_drift_detects_broken_window_and_global_path(
    tmp_path: Path,
) -> None:
    torch, reference, model = _longformer_stub(tmp_path)
    wrapper = LONGFORMER_EXPORT_ADAPTER.build_wrapper(torch, model)
    ids, mask, glob = _longformer_batch(torch, 64, 64, (0, 32))
    with torch.no_grad():
        expected = reference(
            input_ids=ids, attention_mask=mask, global_attention_mask=glob
        ).logits
        assert (expected - wrapper(ids, mask, glob)).abs().max().item() < 1e-4
        # Ignoring the global mask must be visible as drift.
        assert (expected - wrapper(ids, mask)).abs().max().item() > 1e-2
        wrapper.one_sided_windows = (10_000,) * len(wrapper.one_sided_windows)
        broken = wrapper(ids, mask, glob)
        assert (expected - broken).abs().max().item() > 1e-2


def test_longformer_wrapper_traces_without_runtime_shape_reads(
    tmp_path: Path,
) -> None:
    torch, _, model = _longformer_stub(tmp_path)
    wrapper = LONGFORMER_EXPORT_ADAPTER.build_wrapper(torch, model)
    ids, mask, glob = _longformer_batch(torch, 32, 20, (0,))
    with torch.no_grad():
        with LONGFORMER_EXPORT_ADAPTER.trace_context(torch, model):
            traced = torch.jit.trace(wrapper, (ids, mask, glob))
        graph = str(traced.inlined_graph)
        # No nonzero gathers or data-dependent branches survive tracing.
        assert "aten::nonzero" not in graph
        assert "aten::item" not in graph
        assert torch.allclose(traced(ids, mask, glob), wrapper(ids, mask, glob))
        # The traced graph follows a different global mask, not the sample.
        _, _, other = _longformer_batch(torch, 32, 20, (3, 11))
        assert torch.allclose(
            traced(ids, mask, other), wrapper(ids, mask, other), atol=1e-6
        )
