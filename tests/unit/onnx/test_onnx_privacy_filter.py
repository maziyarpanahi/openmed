"""Privacy Filter ONNX export, qualification, and consumer-gate tests.

Stubbed tests run everywhere. The real-export tests build a tiny native
Privacy Filter in-test and need PyTorch, Transformers 5.6 or newer, ONNX, and
ONNX Runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmed.eval import export_qualification as eq
from openmed.eval.export_qualification import ExportQualificationError, ExportReason
from openmed.onnx import inference
from openmed.torch import privacy_filter_export as pf
from tests.fixtures.privacy_filter_export_stub import (
    LABELS,
    SENTINELS,
    NetworkGuard,
    files_containing_sentinels,
    text_contains_sentinels,
)

PF_CONFIG: dict[str, Any] = {
    "model_type": "openai_privacy_filter",
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
    "id2label": {str(key): value for key, value in LABELS.items()},
}


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def test_decoder_profile_filename_is_shared_by_exporter_and_loader() -> None:
    assert (
        _convert_module().DECODER_PROFILE_FILENAME
        == inference.DECODER_PROFILE_FILENAME
        == pf.DECODER_PROFILE_FILENAME
    )


@pytest.fixture
def native_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend a Transformers release with the native family is installed."""
    monkeypatch.setattr(
        pf,
        "metadata",
        SimpleNamespace(
            version=lambda name: "5.22.1", PackageNotFoundError=LookupError
        ),
    )
    monkeypatch.setattr(
        pf, "importlib_util", SimpleNamespace(find_spec=lambda name: object())
    )


@pytest.fixture
def stubbed_convert(monkeypatch: pytest.MonkeyPatch, native_environment):
    module = _convert_module()
    calls: dict[str, list[Any]] = {"optimize": [], "publish": [], "export": []}
    state = {"config": dict(PF_CONFIG)}

    def fake_export_onnx(model_id, output_path, **kwargs):
        calls["export"].append(kwargs)
        path = Path(output_path)
        path.write_bytes(b"graph")
        return path

    def fake_export_webgpu(onnx_path, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"fp16 graph")
        return path

    def fake_save_source_assets(model_id, output_dir, **kwargs):
        output_dir = Path(output_dir)
        (output_dir / "config.json").write_text(json.dumps(state["config"]))
        (output_dir / "id2label.json").write_text(json.dumps(LABELS))
        return dict(state["config"]), []

    def fake_optimize(*args, **kwargs):
        calls["optimize"].append(kwargs)
        raise AssertionError("optimization must not run for a rejected request")

    def fake_publish(**kwargs):
        calls["publish"].append(kwargs)
        return SimpleNamespace(skipped=False, repo_id="OpenMed/stub")

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "export_webgpu", fake_export_webgpu)
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(module, "optimize_onnx_graph", fake_optimize)
    monkeypatch.setattr(module, "validate_optimized_onnx_export", fake_optimize)
    monkeypatch.setattr(module, "publish_artifact", fake_publish)
    return module, calls, state


@pytest.mark.parametrize(
    ("kwargs", "config", "reason"),
    [
        (
            {"expected_source_revision": None},
            {},
            ExportReason.SOURCE_REVISION_UNAVAILABLE,
        ),
        ({}, {"auto_map": {"AutoModel": "x.Y"}}, ExportReason.UNSUPPORTED_REMOTE_CODE),
        ({}, {"num_key_value_heads": 3}, ExportReason.UNSUPPORTED_CONFIGURATION),
        ({"model_id": "attacker/privacy-filter"}, {}, ExportReason.UNSUPPORTED_FAMILY),
        ({"optimize_onnx": True}, {}, ExportReason.UNSUPPORTED_OPERATOR),
        ({"profile": "openvino"}, {}, ExportReason.UNSUPPORTED_RUNTIME),
        ({"include_transformersjs": True}, {}, ExportReason.UNSUPPORTED_RUNTIME),
        ({"max_seq_length": 257}, {}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH),
    ],
)
def test_unsupported_privacy_filter_requests_fail_before_optimization(
    stubbed_convert, tmp_path: Path, kwargs: dict[str, Any], config, reason
) -> None:
    module, calls, state = stubbed_convert
    state["config"] = {**PF_CONFIG, **config}
    options: dict[str, Any] = {
        "model_id": "openai/privacy-filter",
        "max_seq_length": 64,
        "optimize_onnx": False,
        "expected_source_revision": "c0ffee",
        **kwargs,
    }
    model_id = options.pop("model_id")
    with pytest.raises(ExportQualificationError) as error:
        module.convert(model_id, tmp_path / "out", **options)
    assert error.value.reason is reason
    assert calls["optimize"] == []
    assert not (tmp_path / "out" / eq.EXPORT_QUALIFICATION_FILENAME).exists()


def test_unqualified_privacy_filter_bundle_records_profile_and_blocks_publish(
    stubbed_convert, tmp_path: Path
) -> None:
    module, calls, _ = stubbed_convert
    with pytest.raises(ExportQualificationError) as error:
        module.convert(
            "openai/privacy-filter",
            tmp_path / "out",
            max_seq_length=64,
            optimize_onnx=False,
            qualify=False,
            publish_to_hub=True,
            expected_source_revision="c0ffee",
        )
    assert error.value.reason is ExportReason.MISSING_EVIDENCE
    assert calls["publish"] == []
    assert calls["export"][0]["expected_source_revision"] == "c0ffee"
    out = tmp_path / "out"
    manifest = json.loads((out / "openmed-onnx.json").read_text())
    assert manifest["qualification"]["family"] == "openai-privacy-filter"
    assert manifest["qualification"]["status"] == "not_run"
    assert manifest["eligibility"]["declared_checkpoint"] == "openai/privacy-filter"
    profile = pf.load_decoder_profile(out / pf.DECODER_PROFILE_FILENAME)
    assert manifest["decoder_profile"]["sha256"] == profile["sha256"]
    assert manifest["qualification"]["decoder_profile_sha256"] == profile["sha256"]
    assert (
        manifest["decoder_profile"]["threshold_profile"] == profile["threshold_profile"]
    )


def _fake_transformers(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]):
    loads: list[str] = []

    def from_pretrained(*args: Any, **kwargs: Any) -> Any:
        loads.append("model")
        raise AssertionError("weights must not load for a rejected request")

    torch_module = types.ModuleType("torch")
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(**config)
    )
    transformers_module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: loads.append("tokenizer")
    )
    transformers_module.AutoModelForTokenClassification = SimpleNamespace(
        from_pretrained=from_pretrained
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    return loads


@pytest.mark.parametrize(
    ("revision", "config", "reason"),
    [
        (None, {}, ExportReason.SOURCE_REVISION_UNAVAILABLE),
        ("deadbeef", {}, ExportReason.SOURCE_REVISION_MISMATCH),
        (
            "c0ffee",
            {"auto_map": {"AutoModel": "x.Y"}},
            ExportReason.UNSUPPORTED_REMOTE_CODE,
        ),
        ("c0ffee", {"head_dim": 7}, ExportReason.UNSUPPORTED_CONFIGURATION),
    ],
)
def test_export_onnx_rejects_privacy_filter_sources_before_weights_load(
    monkeypatch: pytest.MonkeyPatch,
    native_environment,
    tmp_path: Path,
    revision: str | None,
    config: dict[str, Any],
    reason: ExportReason,
) -> None:
    loads = _fake_transformers(
        monkeypatch, {**PF_CONFIG, "_commit_hash": "c0ffee", **config}
    )
    with pytest.raises(ExportQualificationError) as error:
        _convert_module().export_onnx(
            "openai/privacy-filter",
            tmp_path / "model.onnx",
            max_seq_length=64,
            expected_source_revision=revision,
        )
    assert error.value.reason is reason
    assert loads == []


def _qualified_bundle(tmp_path: Path, *, adapter_version: int = 1) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "model.onnx").write_bytes(b"fp32 graph")
    (bundle / "config.json").write_text(json.dumps(PF_CONFIG))
    (bundle / "id2label.json").write_text(json.dumps(LABELS))
    profile = pf.privacy_filter_decoder_profile(PF_CONFIG)
    pf.write_decoder_profile(profile, bundle / pf.DECODER_PROFILE_FILENAME)
    report = {
        "format": eq.EXPORT_QUALIFICATION_FORMAT,
        "schema_version": eq.EXPORT_QUALIFICATION_SCHEMA_VERSION,
        "family": "openai-privacy-filter",
        "adapter_version": adapter_version,
        "runtime": "onnx",
        "source": {"revision": "c0ffee"},
        "tokenizer": {"sha256": "ab" * 32},
        "label_map": {"sha256": eq.label_map_identity(LABELS)["sha256"]},
        "io_contract": eq.ExportIOContract(
            input_names=("input_ids", "attention_mask"),
            output_names=("logits",),
            sequence_buckets=(32, 64),
            max_sequence_length=64,
            shape_mode="dynamic",
            batch_size=None,
            input_dtype="int64",
        ).to_dict(),
        "offset_contract": dict(eq.OFFSET_CONTRACT),
        "environment": {},
        "suite": {},
        "artifacts": {"model.onnx": eq.sha256_file(bundle / "model.onnx")},
        "decoder": profile,
        "variants": [
            {
                "name": "onnx-fp32",
                "precision": "float32",
                "provider": "CPUExecutionProvider",
                "passed": True,
                "claim": "qualified",
                "artifacts": ["model.onnx"],
            }
        ],
    }
    (bundle / eq.EXPORT_QUALIFICATION_FILENAME).write_text(json.dumps(report))
    return bundle


class _FakeSession:
    def __init__(self, model_path, sess_options=None, providers=None):
        self.providers = providers

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids", type="tensor(int64)", shape=[])]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inference,
        "_load_runtime_dependencies",
        lambda: (np, SimpleNamespace(InferenceSession=_FakeSession), None),
    )


def test_onnx_loader_applies_the_bundled_decoder_profile(
    tmp_path: Path, fake_runtime
) -> None:
    bundle = _qualified_bundle(tmp_path)
    model = inference.OnnxModel(bundle, variant="fp32", tokenizer=object())
    assert model.qualification["family"] == "openai-privacy-filter"
    assert model.decoder_profile["decoder"] == "bioes-viterbi"

    text = "Ana Lima"
    logits = np.zeros((2, len(LABELS)), dtype=np.float32)
    logits[0, next(k for k, v in LABELS.items() if v == "B-private_person")] = 9.0
    logits[1, next(k for k, v in LABELS.items() if v == "E-private_person")] = 9.0
    entities = inference._decode_entities(
        np,
        logits,
        [(0, 3), (3, 8)],
        LABELS,
        text,
        threshold=0.5,
        decoder_profile=model.decoder_profile,
    )
    assert [(item.label, item.text) for item in entities] == [
        ("private_person", "Ana Lima")
    ]


def test_onnx_loader_fails_closed_on_profile_and_adapter_drift(
    tmp_path: Path, fake_runtime
) -> None:
    stale = _qualified_bundle(tmp_path / "stale", adapter_version=2)
    with pytest.raises(ExportQualificationError) as error:
        inference.OnnxModel(stale, variant="fp32", tokenizer=object())
    assert error.value.reason is ExportReason.ADAPTER_VERSION_MISMATCH

    unknown = _qualified_bundle(tmp_path / "unknown")
    report_path = unknown / eq.EXPORT_QUALIFICATION_FILENAME
    report = json.loads(report_path.read_text())
    report["family"] = "privacy-filter"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ExportQualificationError) as error:
        inference.OnnxModel(unknown, variant="fp32", tokenizer=object())
    assert error.value.reason is ExportReason.UNSUPPORTED_FAMILY

    missing = _qualified_bundle(tmp_path / "missing")
    (missing / pf.DECODER_PROFILE_FILENAME).unlink()
    with pytest.raises(ExportQualificationError) as error:
        inference.OnnxModel(missing, variant="fp32", tokenizer=object())
    assert error.value.reason is ExportReason.ARTIFACT_MISSING

    swapped = _qualified_bundle(tmp_path / "swapped")
    other = pf.privacy_filter_decoder_profile(PF_CONFIG, confidence_threshold=0.1)
    pf.write_decoder_profile(other, swapped / pf.DECODER_PROFILE_FILENAME)
    with pytest.raises(ExportQualificationError) as error:
        inference.OnnxModel(swapped, variant="fp32", tokenizer=object())
    assert error.value.reason is ExportReason.ARTIFACT_DIGEST_MISMATCH

    relabeled = _qualified_bundle(tmp_path / "relabeled")
    labels = dict(LABELS)
    labels[1], labels[2] = labels[2], labels[1]
    (relabeled / "id2label.json").write_text(json.dumps(labels))
    with pytest.raises(ExportQualificationError) as error:
        inference.OnnxModel(relabeled, variant="fp32", tokenizer=object())
    assert error.value.reason is ExportReason.LABEL_MAP_MISMATCH


# ---------------------------------------------------------------------------
# Real offline exports
# ---------------------------------------------------------------------------


def _onnx_ready() -> bool:
    for name in ("torch", "transformers", "tokenizers", "onnx", "onnxruntime"):
        if importlib.util.find_spec(name) is None:
            return False
    try:
        return importlib.util.find_spec(pf.PRIVACY_FILTER_NATIVE_MODULE) is not None
    except (ImportError, ValueError):
        return False


requires_onnx = pytest.mark.skipif(
    not _onnx_ready(),
    reason="Privacy Filter ONNX export needs torch, transformers>=5.6, onnx, ort",
)


def _export_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any):
    from tests.fixtures.privacy_filter_export_stub import (
        build_tiny_privacy_filter,
        content_revision,
        write_rule_fixtures,
    )

    checkpoint = build_tiny_privacy_filter(tmp_path / "checkpoint")
    fixtures = write_rule_fixtures(tmp_path / "fixtures.jsonl")
    revision = content_revision(checkpoint)
    guard = NetworkGuard().install(monkeypatch)
    options: dict[str, Any] = {
        "max_seq_length": 64,
        "optimize_onnx": False,
        "qualification_fixtures_path": fixtures,
        "expected_source_revision": revision,
        **kwargs,
    }
    result = _convert_module().convert(str(checkpoint), tmp_path / "out", **options)
    return checkpoint, result, guard


@requires_onnx
def test_privacy_filter_onnx_export_qualifies_offline_without_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    caplog.set_level(logging.DEBUG)
    checkpoint, result, guard = _export_stub(tmp_path, monkeypatch)
    out = result.output_dir

    report = eq.verify_qualification_report(
        out / eq.EXPORT_QUALIFICATION_FILENAME,
        artifact_root=out,
        tokenizer=AutoTokenizer.from_pretrained(checkpoint),
        id2label=LABELS,
    )
    assert report["family"] == "openai-privacy-filter"
    assert report["adapter_version"] == pf.PRIVACY_FILTER_ADAPTER_VERSION
    claims = {item["name"]: item["claim"] for item in report["variants"]}
    assert claims == {"onnx-fp32": "qualified", "webgpu-fp16": "cpu_proxy"}
    profile = pf.load_decoder_profile(out / pf.DECODER_PROFILE_FILENAME)
    assert report["decoder"] == profile
    assert profile["viterbi"]["source"] == "viterbi_calibration.json"
    assert pf.DECODER_PROFILE_FILENAME in report["artifacts"]
    assert report["source"]["kind"] == "content_sha256"
    for variant in report["variants"]:
        assert set(variant["evidence"]["operator_types"]) <= (
            pf.PRIVACY_FILTER_ONNX_OPERATORS
        )
        assert "ai.onnx:TopK" in variant["evidence"]["operator_types"]
        assert variant["span_parity"]["identical"] is True
        assert variant["span_integrity"]["passed"] is True
        assert variant["direct_identifier_recall"]["candidate"] == {
            "ACCOUNT_NUMBER": 1.0,
            "PERSON": 1.0,
        }
        multilingual = variant["multilingual_recall"]
        assert multilingual["gated"] is True and multilingual["passed"] is True
        assert {"ar", "de", "es", "fr", "hi", "ja", "zh"} <= set(
            multilingual["languages"]
        )
        assert variant["critical_leakage"]["new_leaks"] == 0
    assert report["variants"][0]["logit_parity"]["max_abs_diff"] < 1e-4

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["qualification"]["passed"] is True
    assert manifest["qualification"]["decoder_profile_sha256"] == profile["sha256"]
    assert manifest["eligibility"]["label_space"] == "openai-privacy-filter-v1"
    assert manifest["eligibility"]["source_kind"] == "local_directory"

    session = ort.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert [item.name for item in session.get_inputs()] == [
        "input_ids",
        "attention_mask",
    ]
    assert all(isinstance(dim, str) for dim in session.get_inputs()[0].shape)

    model = inference.OnnxModel.from_pretrained(out, variant="fp32")
    assert model.decoder_profile["sha256"] == profile["sha256"]
    for text, expected in (
        (
            "patient ZORVAK QUILLAN was seen today, account 44172290.",
            [("private_person", "ZORVAK QUILLAN"), ("account_number", "44172290")],
        ),
        (
            "患者ゾルヴァク・キランさんは本日来院しました。番号88310442。",
            [("private_person", "ゾルヴァク・キラン"), ("account_number", "88310442")],
        ),
    ):
        entities = model.predict(text, threshold=0.5)
        assert [(item.label, item.text) for item in entities] == expected

    assert guard.attempts == []
    assert files_containing_sentinels(out, SENTINELS) == []
    captured = capfd.readouterr()
    assert not text_contains_sentinels(caplog.text, SENTINELS)
    assert not text_contains_sentinels(captured.out + captured.err, SENTINELS)
    assert str(tmp_path) not in (out / eq.EXPORT_QUALIFICATION_FILENAME).read_text()


@requires_onnx
def test_privacy_filter_android_profile_qualifies_every_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result, guard = _export_stub(tmp_path, monkeypatch, profile="android")
    report = eq.verify_qualification_report(
        result.output_dir / eq.EXPORT_QUALIFICATION_FILENAME,
        artifact_root=result.output_dir,
    )
    by_name = {item["name"]: item for item in report["variants"]}
    assert {"onnx-android-fp32", "onnx-android-fp16", "onnx-int8"} <= set(by_name)
    assert all(item["claim"] == "qualified" for item in by_name.values())
    int8 = by_name["onnx-int8"]
    assert int8["quantization_delta"]["limit"] == 0.005
    assert int8["policy"]["gate_logits"] is False
    assert {
        "ai.onnx:DynamicQuantizeLinear",
        "ai.onnx:MatMulInteger",
    } <= set(int8["evidence"]["operator_types"])
    assert guard.attempts == []


@requires_onnx
def test_privacy_filter_gate_failure_writes_evidence_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ExportQualificationError) as error:
        _export_stub(
            tmp_path,
            monkeypatch,
            include_webgpu=False,
            direct_identifier_recall_floor=1.5,
        )
    assert error.value.reason is ExportReason.DIRECT_IDENTIFIER_RECALL_FAILED
    out = tmp_path / "out"
    report = eq.load_qualification_report(out / eq.EXPORT_QUALIFICATION_FILENAME)
    assert report["passed"] is False
    assert report["variants"][0]["reasons"] == ["direct_identifier_recall_failed"]
    manifest = json.loads((out / "openmed-onnx.json").read_text())
    assert manifest["qualification"]["passed"] is False
    with pytest.raises(ExportQualificationError) as error:
        inference.OnnxModel.from_pretrained(out, variant="fp32")
    assert error.value.reason is ExportReason.VARIANT_NOT_QUALIFIED
