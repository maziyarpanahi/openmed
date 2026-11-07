"""ModernBERT ONNX export adapter, qualification, and consumer-gate tests.

Stubbed tests run everywhere. The real-export tests build a tiny ModernBERT
in-test and need PyTorch, Transformers 5, ONNX, and ONNX Runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmed.eval import export_qualification as eq
from openmed.eval.export_qualification import (
    ExportQualificationError,
    ExportReason,
    VariantCandidate,
)
from openmed.onnx import inference
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

MODERNBERT_CONFIG = {
    "model_type": "modernbert",
    "id2label": {str(key): value for key, value in LABELS.items()},
    "max_position_embeddings": 128,
    "local_attention": 16,
}


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


@pytest.fixture
def stubbed_convert(monkeypatch: pytest.MonkeyPatch):
    module = _convert_module()
    calls: dict[str, list[Any]] = {"optimize": [], "publish": [], "export": []}

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
        (output_dir / "config.json").write_text(json.dumps(MODERNBERT_CONFIG))
        (output_dir / "id2label.json").write_text(json.dumps(LABELS))
        return dict(MODERNBERT_CONFIG), []

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
    monkeypatch.setattr(
        export_adapters,
        "metadata",
        SimpleNamespace(
            version=lambda name: "5.12.1", PackageNotFoundError=LookupError
        ),
    )
    return module, calls


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"profile": "openvino"}, ExportReason.UNSUPPORTED_RUNTIME),
        ({"include_transformersjs": True}, ExportReason.UNSUPPORTED_RUNTIME),
        ({"max_seq_length": 129}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH),
        (
            {"providers": ("CUDAExecutionProvider", "CPUExecutionProvider")},
            ExportReason.UNSUPPORTED_PROVIDER,
        ),
    ],
)
def test_unsupported_modernbert_requests_fail_before_inference(
    stubbed_convert, tmp_path: Path, kwargs: dict[str, Any], reason: ExportReason
) -> None:
    module, calls = stubbed_convert
    options = dict(kwargs)
    providers = options.pop("providers", None)
    if providers is not None:
        options["optimization_config"] = module.OnnxOptimizationConfig(
            providers=providers
        )
    with pytest.raises(ExportQualificationError) as error:
        module.convert(
            "OpenMed/stub",
            tmp_path / "out",
            max_seq_length=options.pop("max_seq_length", 64),
            **options,
        )
    assert error.value.reason is reason
    assert calls["optimize"] == []
    assert not (tmp_path / "out" / eq.EXPORT_QUALIFICATION_FILENAME).exists()


def test_skipped_qualification_is_recorded_and_blocks_publishing(
    stubbed_convert, tmp_path: Path
) -> None:
    module, calls = stubbed_convert
    with pytest.raises(ExportQualificationError) as error:
        module.convert(
            "OpenMed/stub",
            tmp_path / "out",
            max_seq_length=64,
            optimize_onnx=False,
            qualify=False,
            publish_to_hub=True,
        )
    assert error.value.reason is ExportReason.MISSING_EVIDENCE
    assert calls["publish"] == []
    manifest = json.loads((tmp_path / "out" / "openmed-onnx.json").read_text())
    assert manifest["family"] == "modernbert"
    assert manifest["qualification"]["status"] == "not_run"
    assert manifest["qualification"]["qualified_variants"] == []


def test_qualify_flag_requires_an_adapter_family(
    stubbed_convert, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = stubbed_convert

    def bert_assets(model_id, output_dir, **kwargs):
        Path(output_dir, "config.json").write_text('{"model_type": "bert"}')
        return {"model_type": "bert", "id2label": {"0": "O"}}, []

    monkeypatch.setattr(module, "save_source_assets", bert_assets)
    with pytest.raises(ExportQualificationError) as error:
        module.convert(
            "OpenMed/stub", tmp_path / "out", optimize_onnx=False, qualify=True
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_FAMILY


class _CharTokenizer:
    pad_token_id = 0
    special_tokens_map = {"pad_token": "[PAD]"}
    backend_tokenizer = SimpleNamespace(
        to_str=lambda: json.dumps({"model": {"type": "WordLevel", "vocab": {}}})
    )

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids, offsets = [2], [(0, 0)]
        for index, char in enumerate(text):
            if not char.isspace():
                ids.append(10 if char.isupper() else 11 if char.isdigit() else 12)
                offsets.append((index, index + 1))
        ids.append(3)
        offsets.append((0, 0))
        return {"input_ids": ids, "offset_mapping": offsets}


def _rule_logits(feed: Any) -> Any:
    ids = np.asarray(feed["input_ids"])[0]
    logits = np.zeros((1, ids.shape[0], len(LABELS)), dtype=np.float32)
    for index, token in enumerate(ids):
        logits[0, index, {10: 1, 11: 2}.get(int(token), 0)] = 5.0
    return logits


def _qualified_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"fp32 graph")
    (bundle / "model_int8.onnx").write_bytes(b"int8 graph")
    (bundle / "config.json").write_text(json.dumps(MODERNBERT_CONFIG))
    (bundle / "id2label.json").write_text(json.dumps(LABELS))

    def dropped(feed: Any) -> Any:
        logits = _rule_logits(feed)
        logits[..., 1] = 0.0
        return logits

    report = eq.qualify_export(
        family="modernbert",
        adapter_version=1,
        runtime="onnx",
        support=export_adapters.MODERNBERT_ONNX_SUPPORT,
        io_contract=eq.ExportIOContract(
            input_names=("input_ids", "attention_mask"),
            output_names=("logits",),
            sequence_buckets=(32, 64),
            max_sequence_length=96,
            shape_mode="dynamic",
            batch_size=None,
            input_dtype="int64",
        ),
        tokenizer=_CharTokenizer(),
        id2label=LABELS,
        reference_runner=_rule_logits,
        candidates=[
            VariantCandidate(
                name="onnx-fp32",
                runner=_rule_logits,
                precision="float32",
                provider="CPUExecutionProvider",
                artifact_paths=(bundle / "model.onnx",),
            ),
            VariantCandidate(
                name="onnx-int8",
                runner=dropped,
                precision="float32",
                quantization="int8",
                provider="CPUExecutionProvider",
                artifact_paths=(bundle / "model_int8.onnx",),
            ),
        ],
        source={"kind": "hub_commit", "revision": "c0ffee"},
        artifact_root=bundle,
        fixtures_path=write_rule_fixtures(tmp_path / "fixtures.jsonl"),
    )
    eq.write_qualification_report(report, bundle / eq.EXPORT_QUALIFICATION_FILENAME)
    return bundle


class _FakeSession:
    def __init__(self, model_path, sess_options=None, providers=None):
        self.providers = providers

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids", type="tensor(int64)", shape=[])]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]


def test_onnx_runtime_loader_enforces_bundled_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _qualified_bundle(tmp_path)
    monkeypatch.setattr(
        inference,
        "_load_runtime_dependencies",
        lambda: (np, SimpleNamespace(InferenceSession=_FakeSession), None),
    )

    model = inference.OnnxModel(bundle, variant="fp32", tokenizer=object())
    assert model.qualification == {
        "family": "modernbert",
        "variant": "onnx-fp32",
        "claim": "qualified",
        "max_sequence_length": 96,
    }
    with pytest.raises(ExportQualificationError) as error:
        model.predict("patient ZORVAK QUILLAN", max_length=128)
    assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH

    for kwargs, reason in (
        ({"variant": "int8"}, ExportReason.VARIANT_NOT_QUALIFIED),
        (
            {"variant": "fp32", "providers": ("CoreMLExecutionProvider",)},
            ExportReason.UNSUPPORTED_PROVIDER,
        ),
    ):
        with pytest.raises(ExportQualificationError) as error:
            inference.OnnxModel(bundle, tokenizer=object(), **kwargs)
        assert error.value.reason is reason


def _onnx_ready() -> bool:
    for name in ("torch", "transformers", "tokenizers", "onnx", "onnxruntime"):
        if importlib.util.find_spec(name) is None:
            return False
    import transformers

    return int(transformers.__version__.split(".", 1)[0]) >= 5


requires_onnx = pytest.mark.skipif(
    not _onnx_ready(),
    reason="ModernBERT ONNX export needs torch, transformers>=5, onnx, onnxruntime",
)


def _export_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[Path, Any, NetworkGuard]:
    checkpoint = build_tiny_modernbert(tmp_path / "checkpoint")
    fixtures = write_rule_fixtures(tmp_path / "fixtures.jsonl")
    guard = NetworkGuard().install(monkeypatch)
    result = _convert_module().convert(
        str(checkpoint),
        tmp_path / "out",
        max_seq_length=96,
        optimize_onnx=False,
        qualification_fixtures_path=fixtures,
        **kwargs,
    )
    return checkpoint, result, guard


@requires_onnx
def test_modernbert_onnx_export_qualifies_offline_without_leaks(
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
    claims = {item["name"]: item["claim"] for item in report["variants"]}
    assert claims == {"onnx-fp32": "qualified", "webgpu-fp16": "cpu_proxy"}
    assert report["qualified_variants"] == ["onnx-fp32"]
    assert report["io_contract"]["sequence_buckets"] == [8, 16, 32, 64]
    assert report["io_contract"]["max_sequence_length"] == 96
    assert {"model.onnx", "config.json", "tokenizer.json"} <= set(report["artifacts"])
    fp32 = report["variants"][0]
    assert fp32["logit_parity"]["within_tolerance"] is True
    assert fp32["span_parity"]["identical"] is True
    assert fp32["direct_identifier_recall"]["candidate"] == {
        "ID_NUM": 1.0,
        "PERSON": 1.0,
    }
    assert "ai.onnx:Softmax" in fp32["evidence"]["operator_types"]
    assert len(fp32["buckets_evaluated"]) >= 2

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["family"] == "modernbert"
    assert manifest["qualification"]["passed"] is True

    session = ort.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert [item.name for item in session.get_inputs()] == [
        "input_ids",
        "attention_mask",
    ]
    assert [item.name for item in session.get_outputs()] == ["logits"]
    assert all(isinstance(dim, str) for dim in session.get_inputs()[0].shape)

    model = inference.OnnxModel.from_pretrained(out, variant="fp32")
    assert model.qualification["variant"] == "onnx-fp32"
    entities = model.predict("follow up with ZORVAK QUILLAN, mrn 44172290.")
    assert [(item.label, item.text) for item in entities] == [
        ("NAME", "ZORVAK QUILLAN"),
        ("ID_NUM", "44172290"),
    ]
    with pytest.raises(ExportQualificationError) as error:
        inference.OnnxModel.from_pretrained(
            out, variant="fp32", providers=("CoreMLExecutionProvider",)
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_PROVIDER

    assert guard.attempts == []
    assert files_containing_sentinels(out, SENTINELS) == []
    captured = capfd.readouterr()
    assert not text_contains_sentinels(caplog.text, SENTINELS)
    assert not text_contains_sentinels(captured.out + captured.err, SENTINELS)


@requires_onnx
def test_modernbert_android_profile_qualifies_int8_against_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result, guard = _export_stub(
        tmp_path,
        monkeypatch,
        profile="android",
        eval_suite_path=tmp_path / "fixtures.jsonl",
    )
    # INT8 recall certification rewrites config.json; the qualification
    # digests must cover that final file.
    report = eq.verify_qualification_report(
        result.output_dir / eq.EXPORT_QUALIFICATION_FILENAME,
        artifact_root=result.output_dir,
    )
    assert "config.json" in report["artifacts"]
    by_name = {item["name"]: item for item in report["variants"]}
    assert {"onnx-android-fp32", "onnx-android-fp16", "onnx-int8"} <= set(by_name)
    int8 = by_name["onnx-int8"]
    assert int8["quantization"] == "int8"
    assert int8["passed"] is True
    assert int8["quantization_delta"]["limit"] == 0.005
    assert int8["policy"]["gate_logits"] is False
    assert guard.attempts == []
