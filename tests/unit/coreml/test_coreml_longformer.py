"""Longformer Core ML export adapter and qualification tests.

The first group runs everywhere with stubbed converters. The last group
converts a tiny Longformer built in-test and runs Core ML predictions, so it
needs macOS with ``coremltools``, PyTorch, and Transformers.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmed.eval.export_qualification import (
    ExportQualificationError,
    ExportReason,
    check_qualified_request,
    verify_qualification_report,
)
from openmed.torch import export_adapters
from tests.fixtures.longformer_export_stub import (
    LABELS,
    SENTINELS,
    NetworkGuard,
    build_tiny_longformer,
    files_containing_sentinels,
    text_contains_sentinels,
    write_rule_fixtures,
)
from tests.unit.coreml.test_coreml_convert import _fake_coremltools_module


def test_allowlist_accepts_longformer_and_architecture_hint() -> None:
    from openmed.coreml.convert import (
        SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES,
        resolve_supported_model_type,
    )

    assert "longformer" in SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES
    config = types.SimpleNamespace(model_type="Longformer", architectures=[])
    assert resolve_supported_model_type(config) == "longformer"
    hinted = types.SimpleNamespace(
        model_type=None, architectures=["LongformerForTokenClassification"]
    )
    assert resolve_supported_model_type(hinted) == "longformer"


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch, *, commit_hash: str | None = "c0ffee"
) -> types.SimpleNamespace:
    state = types.SimpleNamespace(
        config_loads=0,
        tokenizer_loads=0,
        model_loads=0,
        convert_kwargs=None,
        traced_inputs=None,
        palettizer_config=None,
        palettizer_configs=[],
        published=[],
    )

    class FakeConfig:
        def __init__(self) -> None:
            self.model_type = "longformer"
            self.architectures = ["LongformerForTokenClassification"]
            self.num_labels = len(LABELS)
            self.id2label = dict(LABELS)
            self.num_hidden_layers = 3
            self.num_attention_heads = 4
            self.attention_window = [8, 16, 8]
            self.max_position_embeddings = 66
            self.pad_token_id = 0
            self._commit_hash = commit_hash

        def to_dict(self) -> dict[str, Any]:
            return {"model_type": "longformer", "num_labels": len(LABELS)}

    class FakeModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def eval(self) -> "FakeModule":
            return self

    class FakeModel(FakeModule):
        def __init__(self) -> None:
            self.config = FakeConfig()

    def counted(name: str, factory: Any):
        def load(*_args: Any, **_kwargs: Any) -> Any:
            setattr(state, name, getattr(state, name) + 1)
            return factory()

        return load

    def trace(wrapper: Any, sample: Any) -> Any:
        state.traced_inputs = len(sample)
        return wrapper

    def tokenize(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": np.full((1, 32), 5, dtype=np.int64),
            "attention_mask": np.ones((1, 32), dtype=np.int64),
        }

    torch_module = types.ModuleType("torch")
    torch_module.nn = types.SimpleNamespace(Module=FakeModule)
    torch_module.jit = types.SimpleNamespace(trace=trace)
    torch_module.zeros_like = np.zeros_like
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = types.SimpleNamespace(
        from_pretrained=counted("config_loads", FakeConfig)
    )
    transformers_module.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=counted("tokenizer_loads", lambda: tokenize)
    )
    transformers_module.AutoModelForTokenClassification = types.SimpleNamespace(
        from_pretrained=counted("model_loads", FakeModel)
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "coremltools", _fake_coremltools_module(state))
    monkeypatch.setattr(
        export_adapters,
        "metadata",
        types.SimpleNamespace(
            version=lambda name: "5.5.0", PackageNotFoundError=LookupError
        ),
    )
    # The stub torch cannot trace the real wrapper; keep the adapter contract
    # (inputs, bounds, policies, support) and stub only the torch parts.
    adapter = dataclasses.replace(
        export_adapters.LONGFORMER_EXPORT_ADAPTER,
        build_wrapper=lambda torch, model: model,
    )
    monkeypatch.setitem(export_adapters._ADAPTERS, "longformer", adapter)

    from openmed.coreml import convert as convert_module

    def fake_publish(**kwargs: Any) -> Any:
        state.published.append(kwargs)
        return types.SimpleNamespace(skipped=False, repo_id="OpenMed/stub")

    monkeypatch.setattr(convert_module, "publish_artifact", fake_publish)
    return state


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"max_seq_length": 40}, ExportReason.BUCKET_UNAVAILABLE),
        ({"max_seq_length": 80}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH),
        ({"optimize_for_ane": False}, ExportReason.UNSUPPORTED_SEQUENCE_SHAPE),
        ({"global_attention": "all_tokens"}, ExportReason.UNSUPPORTED_GLOBAL_ATTENTION),
        ({"qualify": True}, ExportReason.RUNTIME_UNAVAILABLE),
        (
            {"expected_source_revision": "deadbeef"},
            ExportReason.SOURCE_REVISION_MISMATCH,
        ),
    ],
)
def test_longformer_requests_fail_before_weights_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kwargs: dict[str, Any],
    reason: ExportReason,
) -> None:
    from openmed.coreml import convert as convert_module

    state = _install_stubs(monkeypatch)
    # Keep the RUNTIME_UNAVAILABLE case deterministic on macOS hosts too.
    monkeypatch.setattr(
        convert_module, "_coreml_prediction_available", lambda ct: False
    )
    options = {"max_seq_length": 32, **kwargs}
    with pytest.raises(ExportQualificationError) as error:
        convert_module.convert("OpenMed/stub", tmp_path / "stub.mlpackage", **options)
    assert error.value.reason is reason
    assert (state.config_loads, state.tokenizer_loads, state.model_loads) == (1, 0, 0)
    assert state.convert_kwargs is None


def test_memory_ceiling_rejects_core_ml_buckets_before_weights_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openmed.coreml.convert import convert

    state = _install_stubs(monkeypatch)
    # fp16: 4 tensors * 4 heads * 32^2 * 2 bytes = 32768 bytes for bucket 32.
    monkeypatch.setitem(
        export_adapters.LONGFORMER_MEMORY_CEILING_BYTES, "coreml", 32_767
    )
    with pytest.raises(ExportQualificationError) as error:
        convert("OpenMed/stub", tmp_path / "stub.mlpackage", max_seq_length=32)
    assert error.value.reason is ExportReason.MEMORY_BUDGET_EXCEEDED
    assert state.model_loads == 0
    # The float32 estimate doubles, so a ceiling that fits fp16 rejects fp32.
    monkeypatch.setitem(
        export_adapters.LONGFORMER_MEMORY_CEILING_BYTES, "coreml", 32_768
    )
    with pytest.raises(ExportQualificationError) as error:
        convert(
            "OpenMed/stub",
            tmp_path / "stub.mlpackage",
            max_seq_length=32,
            compute_precision="float32",
        )
    assert error.value.reason is ExportReason.MEMORY_BUDGET_EXCEEDED


def test_unqualified_longformer_packages_declare_inputs_and_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openmed.coreml import convert as convert_module

    state = _install_stubs(monkeypatch)
    monkeypatch.setattr(
        convert_module, "_coreml_prediction_available", lambda ct: False
    )
    output = tmp_path / "stub.mlpackage"
    with pytest.raises(ExportQualificationError) as error:
        convert_module.convert(
            "OpenMed/stub",
            output,
            max_seq_length=32,
            quantize="int8",
            global_attention="first_token",
            publish_to_hub=True,
        )
    assert error.value.reason is ExportReason.MISSING_EVIDENCE
    assert state.published == []
    assert state.traced_inputs == 3

    inputs = state.convert_kwargs["inputs"]
    assert [item.kwargs["name"] for item in inputs] == [
        "input_ids",
        "attention_mask",
        "global_attention_mask",
    ]
    assert all(item.kwargs["shape"].shape == (1, 32) for item in inputs)
    metadata = json.loads((output / "metadata.json").read_text())
    user = metadata["user_defined_metadata"]
    assert user["export_family"] == "longformer"
    assert user["sequence_buckets"] == "[32]"
    assert user["global_attention_policy"] == "first_token"
    assert user["global_attention_input"] == "global_attention_mask"
    assert user["attention_windows"] == "[8, 16, 8]"
    assert user["memory_ceiling_bytes"] == str(512 * 1024**2)
    assert user["estimated_attention_bytes"] == str(4 * 4 * 32 * 32 * 2)
    manifest = json.loads((tmp_path / "stub_coreml_manifest.json").read_text())
    block = manifest["qualification"]
    assert block["status"] == "not_run"
    assert block["reasons"] == ["runtime_unavailable"]
    assert block["global_attention"] == "first_token"
    assert block["input_names"][-1] == "global_attention_mask"
    assert block["limits"]["declared_buckets"] == [16, 32, 48, 64]
    assert block["limits"]["selected_max_sequence_length"] == 32
    assert block["limits"]["precision"] == "float16"
    assert not (tmp_path / "stub_qualification.json").exists()


def _coreml_ready() -> bool:
    if sys.platform != "darwin":
        return False
    for name in ("coremltools", "torch", "transformers", "tokenizers"):
        if importlib.util.find_spec(name) is None:
            return False
    return True


requires_coreml = pytest.mark.skipif(
    not _coreml_ready(),
    reason="Core ML predictions need macOS, coremltools, torch, and transformers",
)


def _convert_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any):
    from openmed.coreml.convert import convert

    checkpoint = build_tiny_longformer(tmp_path / "checkpoint")
    fixtures = write_rule_fixtures(tmp_path / "fixtures.jsonl")
    guard = NetworkGuard().install(monkeypatch)
    output_dir = tmp_path / "coreml"
    output_dir.mkdir()
    options = {
        "max_seq_length": 64,
        "compute_units": "cpuOnly",
        "latency_iterations": 1,
        "qualification_fixtures_path": fixtures,
        **kwargs,
    }
    convert(str(checkpoint), output_dir / "Stub.mlpackage", **options)
    return checkpoint, output_dir, guard


@requires_coreml
def test_longformer_coreml_export_qualifies_offline_without_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    import coremltools as ct
    from transformers import AutoTokenizer

    quantize = "int8" if importlib.util.find_spec("sklearn") else None
    caplog.set_level(logging.DEBUG)
    checkpoint, output_dir, guard = _convert_stub(
        tmp_path,
        monkeypatch,
        compute_precision="float16",
        quantize=quantize,
        global_attention="first_token",
    )

    report = verify_qualification_report(
        output_dir / "Stub_qualification.json",
        artifact_root=output_dir,
        tokenizer=AutoTokenizer.from_pretrained(checkpoint),
        id2label=LABELS,
    )
    expected = ["coreml-fp16"] + (["coreml-int8"] if quantize else [])
    assert report["passed"] is True
    assert report["qualified_variants"] == expected
    contract = report["io_contract"]
    assert contract["sequence_buckets"] == [64]
    assert contract["shape_mode"] == "static"
    assert contract["global_attention"]["policy"] == "first_token"
    assert contract["windowing"]["window_tokens"] == 64
    assert report["limits"]["selected_estimated_bytes"] == 4 * 4 * 64 * 64 * 2
    for variant in report["variants"]:
        assert variant["direct_identifier_recall"]["candidate"] == {
            "ID_NUM": 1.0,
            "PERSON": 1.0,
        }
        assert variant["critical_leakage"]["new_leaks"] == 0
        assert variant["long_document"]["passed"] is True
        assert variant["long_document"]["reference_spans_in_overlaps"] >= 1
    fp16 = report["variants"][0]
    assert fp16["logit_parity"]["within_tolerance"] is True
    assert fp16["span_parity"]["identical"] is True
    sweep = fp16["sequence_sweep"]
    assert sweep["within_tolerance"] is True and sweep["buckets"] == [64]
    assert sweep["lengths"][0] == 3 and sweep["lengths"][-1] == 64
    assert {4, 5, 7, 8, 9, 15, 16, 17, 63} <= set(sweep["lengths"])

    package = ct.models.MLModel(
        str(output_dir / "Stub.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    spec = package.get_spec()
    assert [item.name for item in spec.description.input] == [
        "input_ids",
        "attention_mask",
        "global_attention_mask",
    ]
    assert all(
        list(item.type.multiArrayType.shape) == [1, 64]
        for item in spec.description.input
    )
    assert package.user_defined_metadata["global_attention_policy"] == "first_token"

    rejections = (
        ({"sequence_length": 65}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH),
        (
            {"document_tokens": contract["windowing"]["max_document_tokens"] + 1},
            ExportReason.SEQUENCE_TOO_LONG,
        ),
        ({"global_attention": "none"}, ExportReason.UNSUPPORTED_GLOBAL_ATTENTION),
    )
    for kwargs, reason in rejections:
        with pytest.raises(ExportQualificationError) as error:
            check_qualified_request(
                report, runtime="coreml", variant="coreml-fp16", **kwargs
            )
        assert error.value.reason is reason

    assert guard.attempts == []
    assert files_containing_sentinels(output_dir, SENTINELS) == []
    captured = capfd.readouterr()
    assert not text_contains_sentinels(caplog.text, SENTINELS)
    assert not text_contains_sentinels(captured.out + captured.err, SENTINELS)


@requires_coreml
def test_longformer_coreml_float32_static_bucket_matches_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir, guard = _convert_stub(
        tmp_path, monkeypatch, compute_precision="float32"
    )
    report = verify_qualification_report(
        output_dir / "Stub_qualification.json", artifact_root=output_dir
    )
    variant = report["variants"][0]
    assert report["passed"] is True
    assert report["io_contract"]["global_attention"]["policy"] == "none"
    assert variant["precision"] == "float32"
    assert variant["logit_parity"]["max_abs_diff"] < 1e-4
    assert variant["sequence_sweep"]["max_abs_diff"] < 1e-4
    assert variant["sequence_sweep"]["lengths"][-1] == 64
    assert guard.attempts == []
