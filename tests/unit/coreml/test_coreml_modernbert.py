"""ModernBERT Core ML export adapter and qualification tests.

The first group runs everywhere with stubbed converters. The last group
converts a tiny ModernBERT built in-test and runs Core ML predictions, so it
needs macOS with ``coremltools``, PyTorch, and Transformers 5.
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
    check_qualified_request,
    load_qualification_report,
    verify_qualification_report,
)
from openmed.torch import export_adapters
from tests.fixtures.modernbert_export_stub import (
    LABELS,
    SENTINELS,
    NetworkGuard,
    build_tiny_modernbert,
    files_containing_sentinels,
    text_contains_sentinels,
    write_rule_fixtures,
)
from tests.unit.coreml.test_coreml_convert import _fake_coremltools_module


def test_allowlist_accepts_modernbert_spellings_and_architecture_hint() -> None:
    from openmed.coreml.convert import (
        SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES,
        resolve_supported_model_type,
    )

    assert "modernbert" in SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES
    for model_type in ("modernbert", "ModernBert", "modern_bert", "modern-bert"):
        config = types.SimpleNamespace(model_type=model_type, architectures=[])
        assert resolve_supported_model_type(config) == "modernbert"
    hinted = types.SimpleNamespace(
        model_type=None, architectures=["ModernBertForTokenClassification"]
    )
    assert resolve_supported_model_type(hinted) == "modernbert"


def test_unsupported_family_keeps_message_and_gains_reason_code() -> None:
    from openmed.coreml.convert import resolve_supported_model_type

    config = types.SimpleNamespace(model_type="gpt2", architectures=[])
    with pytest.raises(ExportQualificationError) as error:
        resolve_supported_model_type(config)
    assert error.value.reason is ExportReason.UNSUPPORTED_FAMILY
    assert "Supported families: bert, distilbert" in str(error.value)
    assert str(error.value).endswith("deberta-v2, modernbert, longformer.")


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_type: str = "modernbert",
    commit_hash: str | None = "c0ffee",
    max_position_embeddings: int = 64,
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

    class FakeConfig:
        def __init__(self) -> None:
            self.model_type = model_type
            self.architectures = ["ModernBertForTokenClassification"]
            self.num_labels = len(LABELS)
            self.id2label = dict(LABELS)
            self.max_position_embeddings = max_position_embeddings
            self.local_attention = 16
            self._commit_hash = commit_hash

        def to_dict(self) -> dict[str, Any]:
            return {"model_type": model_type, "num_labels": len(LABELS)}

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
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "coremltools", _fake_coremltools_module(state))
    monkeypatch.setattr(
        export_adapters,
        "metadata",
        types.SimpleNamespace(
            version=lambda name: "5.12.1",
            PackageNotFoundError=LookupError,
        ),
    )
    # The stub torch cannot trace the real wrapper; keep the adapter contract
    # (validation, support table, static shapes) and stub the torch parts.
    adapter = dataclasses.replace(
        export_adapters.MODERNBERT_EXPORT_ADAPTER,
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
    ("kwargs", "reason"),
    [
        ({"optimize_for_ane": False}, ExportReason.UNSUPPORTED_SEQUENCE_SHAPE),
        ({"max_seq_length": 65}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH),
        ({"qualify": True}, ExportReason.RUNTIME_UNAVAILABLE),
        (
            {"expected_source_revision": "deadbeef"},
            ExportReason.SOURCE_REVISION_MISMATCH,
        ),
    ],
)
def test_modernbert_requests_fail_before_weights_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kwargs: dict[str, Any],
    reason: ExportReason,
) -> None:
    from openmed.coreml.convert import convert

    state = _install_stubs(monkeypatch)
    options = {"max_seq_length": 32, **kwargs}
    with pytest.raises(ExportQualificationError) as error:
        convert("OpenMed/stub", tmp_path / "stub.mlpackage", **options)
    assert error.value.reason is reason
    assert (state.config_loads, state.tokenizer_loads, state.model_loads) == (1, 0, 0)
    assert state.convert_kwargs is None


def test_unpinned_source_revision_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openmed.coreml.convert import convert

    state = _install_stubs(monkeypatch, commit_hash=None)
    with pytest.raises(ExportQualificationError) as error:
        convert("OpenMed/stub", tmp_path / "stub.mlpackage", max_seq_length=32)
    assert error.value.reason is ExportReason.SOURCE_REVISION_UNAVAILABLE
    assert state.model_loads == 0


def test_qualify_flag_requires_an_adapter_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openmed.coreml.convert import convert

    state = _install_stubs(monkeypatch, model_type="bert")
    with pytest.raises(ExportQualificationError) as error:
        convert("OpenMed/stub", tmp_path / "stub.mlpackage", qualify=True)
    assert error.value.reason is ExportReason.UNSUPPORTED_FAMILY
    assert state.model_loads == 0


def test_unqualified_modernbert_packages_are_static_and_never_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openmed.coreml.convert import convert

    state = _install_stubs(monkeypatch)
    output = tmp_path / "stub.mlpackage"
    with pytest.raises(ExportQualificationError) as error:
        convert(
            "OpenMed/stub",
            output,
            max_seq_length=32,
            compute_precision="float32",
            quantize="int8",
            publish_to_hub=True,
        )
    assert error.value.reason is ExportReason.MISSING_EVIDENCE
    assert state.published == []

    inputs = state.convert_kwargs["inputs"]
    assert [item.kwargs["name"] for item in inputs] == ["input_ids", "attention_mask"]
    assert all(item.kwargs["shape"].shape == (1, 32) for item in inputs)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["user_defined_metadata"]["export_family"] == "modernbert"
    assert metadata["user_defined_metadata"]["sequence_buckets"] == "[32]"
    manifest = json.loads((tmp_path / "stub_coreml_manifest.json").read_text())
    assert manifest["qualification"] == {
        "family": "modernbert",
        "adapter_version": 1,
        "passed": False,
        "status": "not_run",
        "reasons": ["runtime_unavailable"],
        "qualified_variants": [],
        "sequence_buckets": [32],
    }
    assert not (tmp_path / "stub_qualification.json").exists()


def _coreml_ready() -> bool:
    if sys.platform != "darwin":
        return False
    for name in ("coremltools", "torch", "transformers", "tokenizers"):
        if importlib.util.find_spec(name) is None:
            return False
    import transformers

    return int(transformers.__version__.split(".", 1)[0]) >= 5


requires_coreml = pytest.mark.skipif(
    not _coreml_ready(),
    reason="Core ML predictions need macOS, coremltools, torch, and transformers>=5",
)


def _convert_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any):
    from openmed.coreml.convert import convert

    checkpoint = build_tiny_modernbert(tmp_path / "checkpoint")
    fixtures = write_rule_fixtures(tmp_path / "fixtures.jsonl")
    guard = NetworkGuard().install(monkeypatch)
    output_dir = tmp_path / "coreml"
    output_dir.mkdir()
    options = {
        "max_seq_length": 96,
        "compute_units": "cpuOnly",
        "latency_iterations": 1,
        "qualification_fixtures_path": fixtures,
        **kwargs,
    }
    convert(str(checkpoint), output_dir / "Stub.mlpackage", **options)
    return checkpoint, output_dir, guard


@requires_coreml
def test_modernbert_coreml_export_qualifies_offline_without_leaks(
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

    report_path = output_dir / "Stub_qualification.json"
    report = verify_qualification_report(
        report_path,
        artifact_root=output_dir,
        tokenizer=AutoTokenizer.from_pretrained(checkpoint),
        id2label=LABELS,
    )
    expected = ["coreml-fp16"] + (["coreml-int8"] if quantize else [])
    assert report["passed"] is True
    assert report["qualified_variants"] == expected
    assert report["source"]["kind"] == "content_sha256"
    assert report["io_contract"]["sequence_buckets"] == [96]
    assert report["io_contract"]["shape_mode"] == "static"
    assert report["support"]["target"] == "iOS16"
    for variant in report["variants"]:
        assert variant["direct_identifier_recall"]["candidate"] == {
            "ID_NUM": 1.0,
            "PERSON": 1.0,
        }
        assert variant["critical_leakage"]["new_leaks"] == 0
    fp16 = report["variants"][0]
    assert fp16["logit_parity"]["within_tolerance"] is True
    assert fp16["span_parity"]["identical"] is True

    manifest = json.loads((output_dir / "Stub_coreml_manifest.json").read_text())
    assert manifest["qualification"]["passed"] is True
    assert manifest["qualification"]["stable_sha256"] == report["stable_sha256"]

    package = ct.models.MLModel(
        str(output_dir / "Stub.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    spec = package.get_spec()
    assert [item.name for item in spec.description.input] == [
        "input_ids",
        "attention_mask",
    ]
    assert [item.name for item in spec.description.output] == ["logits"]
    assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 96]
    assert package.user_defined_metadata["export_family"] == "modernbert"

    with pytest.raises(ExportQualificationError) as error:
        check_qualified_request(
            report, runtime="coreml", variant="coreml-fp16", sequence_length=97
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH

    assert guard.attempts == []
    assert files_containing_sentinels(output_dir, SENTINELS) == []
    captured = capfd.readouterr()
    assert not text_contains_sentinels(caplog.text, SENTINELS)
    assert not text_contains_sentinels(captured.out + captured.err, SENTINELS)


@requires_coreml
def test_modernbert_coreml_gate_failure_writes_evidence_and_raises(
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
    assert variant["precision"] == "float32"
    assert variant["logit_parity"]["max_abs_diff"] < 1e-4
    assert variant["reasons"] == ["direct_identifier_recall_failed"]
    manifest = json.loads((output_dir / "Stub_coreml_manifest.json").read_text())
    assert manifest["qualification"]["passed"] is False
