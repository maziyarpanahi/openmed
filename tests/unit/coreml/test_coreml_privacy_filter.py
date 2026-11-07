"""Privacy Filter Core ML export adapter and qualification tests.

The first group runs everywhere with stubbed converters. The last group
converts a tiny native Privacy Filter built in-test and runs Core ML
predictions, so it needs macOS with ``coremltools``, PyTorch, and
Transformers 5.6 or newer.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from openmed.eval.export_qualification import (
    ExportQualificationError,
    ExportReason,
    load_qualification_report,
    verify_qualification_report,
)
from openmed.torch import export_adapters
from openmed.torch import privacy_filter_export as pf
from tests.fixtures.privacy_filter_export_stub import (
    LABELS,
    SENTINELS,
    NetworkGuard,
    files_containing_sentinels,
    text_contains_sentinels,
)
from tests.unit.coreml.test_coreml_convert import _fake_coremltools_module

PF_CONFIG: dict[str, Any] = {
    "model_type": "openai_privacy_filter",
    "architectures": ["OpenAIPrivacyFilterForTokenClassification"],
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 16,
    "head_dim": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "sliding_window": 4,
    "max_position_embeddings": 256,
    "rope_parameters": {"rope_type": "yarn", "factor": 32.0},
    "id2label": dict(LABELS),
    "num_labels": len(LABELS),
}


def test_allowlist_and_architecture_hint_accept_privacy_filter() -> None:
    from openmed.coreml.convert import (
        SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES,
        resolve_supported_model_type,
    )

    assert "openai-privacy-filter" in SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES
    for model_type in ("openai_privacy_filter", "openai-privacy-filter"):
        config = types.SimpleNamespace(model_type=model_type, architectures=[])
        assert resolve_supported_model_type(config) == "openai-privacy-filter"
    hinted = types.SimpleNamespace(
        model_type=None, architectures=["OpenAIPrivacyFilterForTokenClassification"]
    )
    assert resolve_supported_model_type(hinted) == "openai-privacy-filter"
    # The original OPF and MLX spellings never reach the exporter.
    with pytest.raises(ExportQualificationError) as error:
        resolve_supported_model_type(
            types.SimpleNamespace(model_type="privacy_filter", architectures=[])
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_FAMILY


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    native: bool = True,
    **config_overrides: Any,
) -> types.SimpleNamespace:
    state = types.SimpleNamespace(
        config_loads=0,
        tokenizer_loads=0,
        model_loads=0,
        convert_kwargs=None,
        palettizer_config=None,
        palettizer_configs=[],
        published=[],
    )
    values = {**PF_CONFIG, "_commit_hash": "c0ffee", **config_overrides}

    class FakeConfig:
        def __init__(self) -> None:
            for key, value in values.items():
                setattr(self, key, value)

        def to_dict(self) -> dict[str, Any]:
            return dict(values)

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

    torch_module = types.ModuleType("torch")
    torch_module.nn = types.SimpleNamespace(Module=FakeModule)
    torch_module.jit = types.SimpleNamespace(trace=lambda wrapper, sample: wrapper)
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = types.SimpleNamespace(
        from_pretrained=counted("config_loads", FakeConfig)
    )
    transformers_module.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=counted(
            "tokenizer_loads",
            lambda: lambda *a, **k: {"input_ids": "ids", "attention_mask": "mask"},
        )
    )
    transformers_module.AutoModelForTokenClassification = types.SimpleNamespace(
        from_pretrained=counted("model_loads", FakeModel)
    )
    coremltools_module = _fake_coremltools_module(state)
    coremltools_module.transform = types.SimpleNamespace(
        FP16ComputePrecision=lambda op_selector: ("FLOAT16_SELECTED", op_selector)
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "coremltools", coremltools_module)
    monkeypatch.setattr(
        pf,
        "metadata",
        types.SimpleNamespace(
            version=lambda name: "5.22.1", PackageNotFoundError=LookupError
        ),
    )
    monkeypatch.setattr(
        pf,
        "importlib_util",
        types.SimpleNamespace(find_spec=lambda name: object() if native else None),
    )
    monkeypatch.setattr(pf, "_cached_hub_file", lambda *args, **kwargs: None)
    # The stub torch cannot trace the real module; keep the adapter contract
    # (eligibility, support table, decoder profile) and stub the torch parts.
    adapter = dataclasses.replace(
        pf.PRIVACY_FILTER_EXPORT_ADAPTER,
        prepare_model=lambda model: None,
        build_wrapper=lambda torch, model: model,
        trace_context=lambda torch, model: nullcontext(),
    )
    for alias in adapter.model_types:
        monkeypatch.setitem(export_adapters._ADAPTERS, alias, adapter)

    from openmed.coreml import convert as convert_module

    def fake_publish(**kwargs: Any) -> Any:
        state.published.append(kwargs)
        return types.SimpleNamespace(skipped=False, repo_id="OpenMed/stub")

    monkeypatch.setattr(convert_module, "publish_artifact", fake_publish)
    return state


@pytest.mark.parametrize(
    ("kwargs", "config", "native", "reason"),
    [
        ({"expected_source_revision": None}, {}, True, "source_revision_unavailable"),
        (
            {"expected_source_revision": "deadbeef"},
            {},
            True,
            "source_revision_mismatch",
        ),
        ({}, {"auto_map": {"AutoModel": "x.Y"}}, True, "unsupported_remote_code"),
        ({}, {"num_experts_per_tok": 9}, True, "unsupported_configuration"),
        ({"model_id": "attacker/privacy-filter"}, {}, True, "unsupported_family"),
        ({}, {}, False, "unsupported_converter_environment"),
        ({"quantize": "int4"}, {}, True, "unsupported_quantization"),
        ({"optimize_for_ane": False}, {}, True, "unsupported_sequence_shape"),
        ({"max_seq_length": 257}, {}, True, "unsupported_sequence_length"),
    ],
)
def test_privacy_filter_requests_fail_before_weights_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kwargs: dict[str, Any],
    config: dict[str, Any],
    native: bool,
    reason: str,
) -> None:
    from openmed.coreml.convert import convert

    state = _install_stubs(monkeypatch, native=native, **config)
    options: dict[str, Any] = {
        "model_id": "openai/privacy-filter",
        "max_seq_length": 32,
        "expected_source_revision": "c0ffee",
        **kwargs,
    }
    model_id = options.pop("model_id")
    with pytest.raises(ExportQualificationError) as error:
        convert(model_id, tmp_path / "stub.mlpackage", **options)
    assert error.value.reason_code == reason
    assert (state.tokenizer_loads, state.model_loads) == (0, 0)
    assert state.convert_kwargs is None


def test_unqualified_privacy_filter_package_ships_profile_and_is_never_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openmed.coreml.convert import convert

    state = _install_stubs(monkeypatch)
    output = tmp_path / "stub.mlpackage"
    with pytest.raises(ExportQualificationError) as error:
        convert(
            "openai/privacy-filter",
            output,
            max_seq_length=32,
            compute_precision="float16",
            quantize="int8",
            publish_to_hub=True,
            expected_source_revision="c0ffee",
        )
    assert error.value.reason is ExportReason.MISSING_EVIDENCE
    assert state.published == []

    precision = state.convert_kwargs["compute_precision"]
    assert precision[0] == "FLOAT16_SELECTED"
    selector = precision[1]
    for op_type in sorted(pf.PRIVACY_FILTER_COREML_FP32_OPS):
        assert selector(types.SimpleNamespace(op_type=op_type)) is False
    assert selector(types.SimpleNamespace(op_type="matmul")) is True
    inputs = state.convert_kwargs["inputs"]
    assert all(item.kwargs["shape"].shape == (1, 32) for item in inputs)

    profile = pf.load_decoder_profile(tmp_path / "stub_decoder_profile.json")
    metadata = json.loads((output / "metadata.json").read_text())
    user = metadata["user_defined_metadata"]
    assert user["export_family"] == "openai-privacy-filter"
    assert user["decoder"] == "bioes-viterbi"
    assert user["decoder_profile_sha256"] == profile["sha256"]
    assert user["label_space"] == "openai-privacy-filter-v1"
    assert user["float32_ops"] == "exp,real_div,reduce_sum"

    manifest = json.loads((tmp_path / "stub_coreml_manifest.json").read_text())
    assert manifest["eligibility"]["declared_checkpoint"] == "openai/privacy-filter"
    assert manifest["decoder_profile"]["path"] == "stub_decoder_profile.json"
    assert manifest["decoder_profile"]["sha256"] == profile["sha256"]
    assert manifest["qualification"]["status"] == "not_run"
    assert manifest["qualification"]["decoder_profile_sha256"] == profile["sha256"]
    assert manifest["qualification"]["reasons"] == ["runtime_unavailable"]


def _coreml_ready() -> bool:
    if sys.platform != "darwin":
        return False
    for name in ("coremltools", "torch", "transformers", "tokenizers"):
        if importlib.util.find_spec(name) is None:
            return False
    try:
        return importlib.util.find_spec(pf.PRIVACY_FILTER_NATIVE_MODULE) is not None
    except (ImportError, ValueError):
        return False


requires_coreml = pytest.mark.skipif(
    not _coreml_ready(),
    reason="Core ML predictions need macOS, coremltools, torch, transformers>=5.6",
)


def _convert_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any):
    from openmed.coreml.convert import convert
    from tests.fixtures.privacy_filter_export_stub import (
        build_tiny_privacy_filter,
        content_revision,
        write_rule_fixtures,
    )

    checkpoint = build_tiny_privacy_filter(tmp_path / "checkpoint")
    fixtures = write_rule_fixtures(tmp_path / "fixtures.jsonl")
    revision = content_revision(checkpoint)
    guard = NetworkGuard().install(monkeypatch)
    output_dir = tmp_path / "coreml"
    output_dir.mkdir()
    options = {
        "max_seq_length": 64,
        "compute_units": "cpuOnly",
        "latency_iterations": 1,
        "qualification_fixtures_path": fixtures,
        "expected_source_revision": revision,
        **kwargs,
    }
    convert(str(checkpoint), output_dir / "Stub.mlpackage", **options)
    return checkpoint, output_dir, guard


@requires_coreml
def test_privacy_filter_coreml_export_qualifies_offline_without_leaks(
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
        tmp_path, monkeypatch, compute_precision="float16", quantize=quantize
    )

    report = verify_qualification_report(
        output_dir / "Stub_qualification.json",
        artifact_root=output_dir,
        tokenizer=AutoTokenizer.from_pretrained(checkpoint),
        id2label=LABELS,
    )
    expected = ["coreml-fp16"] + (["coreml-int8"] if quantize else [])
    assert report["passed"] is True
    assert report["family"] == "openai-privacy-filter"
    assert report["qualified_variants"] == expected
    assert report["io_contract"]["sequence_buckets"] == [64]
    profile = pf.load_decoder_profile(output_dir / "Stub_decoder_profile.json")
    assert report["decoder"] == profile
    assert "Stub_decoder_profile.json" in report["artifacts"]
    for variant in report["variants"]:
        assert variant["direct_identifier_recall"]["candidate"] == {
            "ACCOUNT_NUMBER": 1.0,
            "PERSON": 1.0,
        }
        assert variant["critical_leakage"]["new_leaks"] == 0
        assert variant["span_integrity"]["passed"] is True
        assert variant["multilingual_recall"]["passed"] is True
    fp16 = report["variants"][0]
    assert fp16["logit_parity"]["within_tolerance"] is True
    assert fp16["span_parity"]["identical"] is True

    manifest = json.loads((output_dir / "Stub_coreml_manifest.json").read_text())
    assert manifest["qualification"]["passed"] is True
    assert manifest["qualification"]["decoder_profile_sha256"] == profile["sha256"]

    package = ct.models.MLModel(
        str(output_dir / "Stub.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    spec = package.get_spec()
    assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 64]
    assert package.user_defined_metadata["decoder"] == "bioes-viterbi"

    assert guard.attempts == []
    assert files_containing_sentinels(output_dir, SENTINELS) == []
    captured = capfd.readouterr()
    assert not text_contains_sentinels(caplog.text, SENTINELS)
    assert not text_contains_sentinels(captured.out + captured.err, SENTINELS)


@requires_coreml
def test_privacy_filter_coreml_gate_failure_writes_evidence_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ExportQualificationError) as error:
        _convert_stub(
            tmp_path,
            monkeypatch,
            compute_precision="float32",
            direct_identifier_recall_floor=1.5,
        )
    assert error.value.reason is ExportReason.DIRECT_IDENTIFIER_RECALL_FAILED
    output_dir = tmp_path / "coreml"
    report = load_qualification_report(output_dir / "Stub_qualification.json")
    variant = report["variants"][0]
    assert report["passed"] is False
    assert variant["logit_parity"]["max_abs_diff"] < 1e-4
    assert variant["reasons"] == ["direct_identifier_recall_failed"]
