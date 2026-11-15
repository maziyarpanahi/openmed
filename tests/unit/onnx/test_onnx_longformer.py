"""Longformer ONNX export adapter, qualification, and runtime-bound tests.

Stubbed tests run everywhere. The real-export tests build a tiny Longformer
in-test and need PyTorch, Transformers, ONNX, and ONNX Runtime.
"""

from __future__ import annotations

import dataclasses
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
from openmed.eval.export_qualification import (
    ExportQualificationError,
    ExportReason,
    VariantCandidate,
)
from openmed.onnx import inference
from openmed.torch import export_adapters
from tests.fixtures.longformer_export_stub import (
    LABELS,
    SENTINELS,
    NetworkGuard,
    build_tiny_longformer,
    files_containing_sentinels,
    rule_fixture_rows,
    text_contains_sentinels,
    write_rule_fixtures,
)

LONGFORMER_CONFIG = {
    "model_type": "longformer",
    "id2label": {str(key): value for key, value in LABELS.items()},
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "attention_window": [8, 16, 8],
    "max_position_embeddings": 66,
    "pad_token_id": 0,
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
        (output_dir / "config.json").write_text(json.dumps(LONGFORMER_CONFIG))
        (output_dir / "id2label.json").write_text(json.dumps(LABELS))
        return dict(LONGFORMER_CONFIG), []

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
        SimpleNamespace(version=lambda name: "5.5.0", PackageNotFoundError=LookupError),
    )
    return module, calls


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"profile": "openvino"}, ExportReason.UNSUPPORTED_RUNTIME),
        ({"include_transformersjs": True}, ExportReason.UNSUPPORTED_RUNTIME),
        ({"max_seq_length": 80}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH),
        ({"max_seq_length": 40}, ExportReason.BUCKET_UNAVAILABLE),
        ({"global_attention": "all_tokens"}, ExportReason.UNSUPPORTED_GLOBAL_ATTENTION),
        (
            {"profile": "android", "global_attention": "first_token"},
            ExportReason.UNSUPPORTED_GLOBAL_ATTENTION,
        ),
        (
            {"providers": ("CUDAExecutionProvider",)},
            ExportReason.UNSUPPORTED_PROVIDER,
        ),
    ],
)
def test_unsupported_longformer_requests_fail_before_inference(
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


def test_memory_ceiling_rejects_buckets_before_inference(
    stubbed_convert, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, calls = stubbed_convert
    # 4 tensors * 4 heads * 64^2 * 4 bytes = 262144 bytes for the 64 bucket.
    monkeypatch.setitem(
        export_adapters.LONGFORMER_MEMORY_CEILING_BYTES, "onnx", 262_143
    )
    with pytest.raises(ExportQualificationError) as error:
        module.convert("OpenMed/stub", tmp_path / "out", max_seq_length=64)
    assert error.value.reason is ExportReason.MEMORY_BUDGET_EXCEEDED
    assert "262" not in str(error.value)
    assert calls["optimize"] == []


def test_global_attention_requires_a_family_that_declares_it(
    stubbed_convert, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = stubbed_convert

    def bert_assets(model_id, output_dir, **kwargs):
        Path(output_dir, "config.json").write_text('{"model_type": "bert"}')
        return {"model_type": "bert", "id2label": {"0": "O"}}, []

    monkeypatch.setattr(module, "save_source_assets", bert_assets)
    with pytest.raises(ExportQualificationError) as error:
        module.convert(
            "OpenMed/stub",
            tmp_path / "out",
            optimize_onnx=False,
            global_attention="none",
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_GLOBAL_ATTENTION


def test_skipped_qualification_records_bounds_and_blocks_publishing(
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
            global_attention="first_token",
            publish_to_hub=True,
        )
    assert error.value.reason is ExportReason.MISSING_EVIDENCE
    assert calls["publish"] == []
    manifest = json.loads((tmp_path / "out" / "openmed-onnx.json").read_text())
    assert manifest["family"] == "longformer"
    block = manifest["qualification"]
    assert block["status"] == "not_run"
    assert block["global_attention"] == "first_token"
    assert block["input_names"] == [
        "input_ids",
        "attention_mask",
        "global_attention_mask",
    ]
    assert block["limits"]["declared_buckets"] == [16, 32, 48, 64]
    assert block["limits"]["selected_max_sequence_length"] == 64
    assert block["limits"]["memory_ceiling_bytes"] == 2 * 1024**3
    buckets = manifest["dynamic_shapes"]["shape_buckets"]
    assert buckets["buckets"] == [16, 32, 48, 64]
    assert buckets["max_length"] == 64


def test_export_declares_three_inputs_and_a_bounded_sequence_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _convert_module()
    export_call: dict[str, Any] = {}

    class Module:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    def fake_dim(name, *, min=None, max=None):
        return {"name": name, "min": min, "max": max}

    def fake_export(
        wrapper,
        example_inputs,
        output_path,
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
                "inputs": len(example_inputs),
                "input_names": input_names,
                "dynamo": dynamo,
                "dynamic_shapes": dynamic_shapes,
                "dynamic_axes": dynamic_axes,
            }
        )
        Path(output_path).write_bytes(b"graph")

    torch_mod = types.ModuleType("torch")
    torch_mod.nn = types.SimpleNamespace(Module=Module)
    torch_mod.no_grad = lambda: _NullContext()
    torch_mod.float32 = "float32"
    torch_mod.zeros_like = lambda value: np.zeros_like(value)
    torch_mod.export = types.SimpleNamespace(Dim=fake_dim)
    torch_mod.onnx = types.SimpleNamespace(export=fake_export)

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": np.full((2, 64), 5),
                "attention_mask": np.ones((2, 64), dtype=int),
            }

    class FakeModel(Module):
        config = SimpleNamespace(**LONGFORMER_CONFIG)

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )
    transformers_mod.AutoModelForTokenClassification = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeModel()
    )
    onnx_mod = types.ModuleType("onnx")
    onnx_mod.load = lambda path, **kwargs: types.SimpleNamespace(
        graph=types.SimpleNamespace(initializer=[])
    )
    onnx_mod.checker = types.SimpleNamespace(check_model=lambda model: None)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)
    monkeypatch.setattr(
        export_adapters,
        "metadata",
        SimpleNamespace(version=lambda name: "5.5.0", PackageNotFoundError=LookupError),
    )
    adapter = dataclasses.replace(
        export_adapters.LONGFORMER_EXPORT_ADAPTER,
        build_wrapper=lambda torch, model: model,
    )
    monkeypatch.setitem(export_adapters._ADAPTERS, "longformer", adapter)

    module.export_onnx("OpenMed/stub", tmp_path / "model.onnx", max_seq_length=64)
    assert export_call["input_names"] == [
        "input_ids",
        "attention_mask",
        "global_attention_mask",
    ]
    assert export_call["inputs"] == 3
    shapes = export_call["dynamic_shapes"]
    assert set(shapes) == set(export_call["input_names"])
    for name in export_call["input_names"]:
        assert shapes[name][1] == {"name": "sequence", "min": 1, "max": 64}

    android_call = dict(export_call)
    export_call.clear()
    module.export_onnx(
        "OpenMed/stub", tmp_path / "android.onnx", max_seq_length=64, profile="android"
    )
    assert export_call["input_names"] == ["input_ids", "attention_mask"]
    assert export_call["inputs"] == 2
    assert android_call["input_names"] != export_call["input_names"]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _CharTokenizer:
    """Character-level stand-in for the Hugging Face tokenizer protocol."""

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
        encoded: dict[str, Any] = {"input_ids": ids, "offset_mapping": offsets}
        if kwargs.get("return_tensors") == "np":
            # The OnnxModel custom-tokenizer path wants batched arrays.
            encoded = {
                "input_ids": np.asarray([ids]),
                "offset_mapping": np.asarray([offsets]),
                "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            }
        return encoded


def _rule_logits(feed: Any) -> Any:
    ids = np.asarray(feed["input_ids"])
    logits = np.zeros((*ids.shape, len(LABELS)), dtype=np.float32)
    for row in range(ids.shape[0]):
        for index, token in enumerate(ids[row]):
            logits[row, index, {10: 1, 11: 2}.get(int(token), 0)] = 5.0
    return logits


def _qualified_bundle(tmp_path: Path, *, policy: str = "first_token") -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"fp32 graph")
    (bundle / "config.json").write_text(json.dumps(LONGFORMER_CONFIG))
    (bundle / "id2label.json").write_text(json.dumps(LABELS))
    report = eq.qualify_export(
        family="longformer",
        adapter_version=1,
        runtime="onnx",
        support=export_adapters.LONGFORMER_ONNX_SUPPORT,
        io_contract=eq.ExportIOContract(
            input_names=("input_ids", "attention_mask", "global_attention_mask"),
            output_names=("logits",),
            sequence_buckets=(16, 32, 48, 64),
            max_sequence_length=64,
            shape_mode="dynamic",
            batch_size=None,
            input_dtype="int64",
            global_attention=policy,
            windowing=eq.WindowingPolicy(
                window_tokens=64, overlap_tokens=15, max_document_tokens=2048
            ),
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
        ],
        source={"kind": "hub_commit", "revision": "c0ffee"},
        artifact_root=bundle,
        fixtures_path=write_rule_fixtures(tmp_path / "fixtures.jsonl"),
        sequence_sweep_boundaries=(4, 8, 16),
    )
    eq.write_qualification_report(report, bundle / eq.EXPORT_QUALIFICATION_FILENAME)
    return bundle


class _RecordingSession:
    feeds: list[dict[str, Any]] = []

    def __init__(self, model_path, sess_options=None, providers=None):
        self.providers = providers

    def get_inputs(self):
        return [
            SimpleNamespace(
                name=name, type="tensor(int64)", shape=["batch", "sequence"]
            )
            for name in ("input_ids", "attention_mask", "global_attention_mask")
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, names, feed):
        type(self).feeds.append({key: np.asarray(value) for key, value in feed.items()})
        return [_rule_logits(feed)]


def _long_text(repeats: int = 4) -> str:
    return " ".join(row["text"] for row in rule_fixture_rows() * repeats)


def test_onnx_runtime_fills_global_attention_and_windows_long_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _qualified_bundle(tmp_path)
    monkeypatch.setattr(
        inference,
        "_load_runtime_dependencies",
        lambda: (np, SimpleNamespace(InferenceSession=_RecordingSession), None),
    )
    _RecordingSession.feeds = []
    model = inference.OnnxModel(bundle, variant="fp32", tokenizer=_CharTokenizer())
    assert model.qualification["global_attention"] == "first_token"
    assert model.qualification["windowing"]["window_tokens"] == 64

    text = _long_text()
    prediction = model.predict_batch_detailed([text])[0]
    assert prediction.complete is True
    assert prediction.window_count > 2
    assert prediction.processed_tokens == prediction.token_count
    feeds = _RecordingSession.feeds
    assert feeds and all(feed["input_ids"].shape[1] <= 64 for feed in feeds)
    for feed in feeds:
        # One global token per window (the leading special), never padding.
        assert feed["global_attention_mask"][:, 0].tolist() == [1] * len(
            feed["global_attention_mask"]
        )
        assert int(feed["global_attention_mask"][:, 1:].sum()) == 0
    # Document-global offsets: every gold span of every repeat is recovered.
    expected = []
    cursor = 0
    for row in rule_fixture_rows() * 4:
        for span in row["gold_spans"]:
            expected.append((span["start"] + cursor, span["end"] + cursor))
        cursor += len(row["text"]) + 1
    found = {(entity.start, entity.end) for entity in prediction.entities}
    names = [pair for pair in expected if text[pair[0] : pair[1]].isupper()]
    assert set(names) <= found

    with pytest.raises(ExportQualificationError) as error:
        model.predict_batch_detailed([_long_text(repeats=12)])
    assert error.value.reason is ExportReason.SEQUENCE_TOO_LONG
    with pytest.raises(ExportQualificationError) as error:
        model.predict(text, max_length=65)
    assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH


def test_onnx_runtime_defaults_to_local_attention_for_none_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _qualified_bundle(tmp_path, policy="none")
    monkeypatch.setattr(
        inference,
        "_load_runtime_dependencies",
        lambda: (np, SimpleNamespace(InferenceSession=_RecordingSession), None),
    )
    _RecordingSession.feeds = []
    model = inference.OnnxModel(bundle, variant="fp32", tokenizer=_CharTokenizer())
    model.predict("patient ZORVAK QUILLAN, mrn 44172290.")
    assert all(
        int(feed["global_attention_mask"].sum()) == 0
        for feed in _RecordingSession.feeds
    )


def _onnx_ready() -> bool:
    for name in ("torch", "transformers", "tokenizers", "onnx", "onnxruntime"):
        if importlib.util.find_spec(name) is None:
            return False
    return True


requires_onnx = pytest.mark.skipif(
    not _onnx_ready(),
    reason="Longformer ONNX export needs torch, transformers, onnx, onnxruntime",
)


def _export_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[Path, Any, NetworkGuard]:
    checkpoint = build_tiny_longformer(tmp_path / "checkpoint")
    fixtures = write_rule_fixtures(tmp_path / "fixtures.jsonl")
    guard = NetworkGuard().install(monkeypatch)
    result = _convert_module().convert(
        str(checkpoint),
        tmp_path / "out",
        max_seq_length=64,
        optimize_onnx=False,
        qualification_fixtures_path=fixtures,
        **kwargs,
    )
    return checkpoint, result, guard


@requires_onnx
def test_longformer_onnx_export_qualifies_offline_without_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    caplog.set_level(logging.DEBUG)
    checkpoint, result, guard = _export_stub(
        tmp_path, monkeypatch, global_attention="first_token"
    )
    out = result.output_dir

    report = eq.verify_qualification_report(
        out / eq.EXPORT_QUALIFICATION_FILENAME,
        artifact_root=out,
        tokenizer=AutoTokenizer.from_pretrained(checkpoint),
        id2label=LABELS,
    )
    claims = {item["name"]: item["claim"] for item in report["variants"]}
    assert claims == {"onnx-fp32": "qualified", "webgpu-fp16": "cpu_proxy"}
    contract = report["io_contract"]
    assert contract["input_names"] == [
        "input_ids",
        "attention_mask",
        "global_attention_mask",
    ]
    assert contract["sequence_buckets"] == [16, 32, 48, 64]
    assert contract["max_sequence_length"] == 64
    assert contract["global_attention"]["policy"] == "first_token"
    assert contract["windowing"]["window_tokens"] == 64
    assert report["limits"]["attention_windows"] == [8, 16, 8]
    assert report["limits"]["supported_buckets"] == [16, 32, 48, 64]
    fp32 = report["variants"][0]
    assert fp32["logit_parity"]["within_tolerance"] is True
    assert fp32["span_parity"]["identical"] is True
    assert fp32["direct_identifier_recall"]["candidate"] == {
        "ID_NUM": 1.0,
        "PERSON": 1.0,
    }
    sweep = fp32["sequence_sweep"]
    assert sweep["within_tolerance"] is True
    # Minimum, attention-window boundaries (4, 8, 16), buckets, and maximum.
    assert sweep["lengths"][0] == 3 and sweep["lengths"][-1] == 64
    assert {4, 5, 7, 8, 9, 15, 16, 17, 47, 48, 49, 63} <= set(sweep["lengths"])
    assert sweep["probes"] == ["global_probe", "policy"]
    long_document = fp32["long_document"]
    assert long_document["passed"] is True
    assert long_document["windows"] >= 4
    assert long_document["reference_spans_in_overlaps"] >= 1
    operators = set(fp32["evidence"]["operator_types"])
    assert not operators & set(export_adapters.LONGFORMER_FORBIDDEN_ONNX_OPERATORS)
    assert "ai.onnx:Softmax" in operators

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["family"] == "longformer"
    assert manifest["qualification"]["passed"] is True
    assert manifest["qualification"]["global_attention"]["policy"] == "first_token"
    assert manifest["qualification"]["limits"]["selected_max_sequence_length"] == 64
    assert manifest["dynamic_shapes"]["shape_buckets"]["buckets"] == [16, 32, 48, 64]

    session = ort.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert [item.name for item in session.get_inputs()] == contract["input_names"]
    assert all(isinstance(dim, str) for dim in session.get_inputs()[0].shape)

    model = inference.OnnxModel.from_pretrained(out, variant="fp32")
    assert model.qualification["global_attention"] == "first_token"
    text = _long_text()
    prediction = model.predict_batch_detailed([text])[0]
    assert prediction.window_count > 2 and prediction.complete
    labels = {(entity.label, entity.text) for entity in prediction.entities}
    assert ("NAME", "ZORVAK QUILLAN") in labels
    assert ("ID_NUM", "44172290") in labels
    with pytest.raises(ExportQualificationError) as error:
        model.predict(text, max_length=65)
    assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH

    assert guard.attempts == []
    assert files_containing_sentinels(out, SENTINELS) == []
    captured = capfd.readouterr()
    assert not text_contains_sentinels(caplog.text, SENTINELS)
    assert not text_contains_sentinels(captured.out + captured.err, SENTINELS)


@requires_onnx
def test_longformer_android_profile_keeps_two_inputs_and_qualifies_int8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import onnxruntime as ort

    _, result, guard = _export_stub(
        tmp_path,
        monkeypatch,
        profile="android",
        eval_suite_path=tmp_path / "fixtures.jsonl",
    )
    out = result.output_dir
    report = eq.verify_qualification_report(
        out / eq.EXPORT_QUALIFICATION_FILENAME, artifact_root=out
    )
    contract = report["io_contract"]
    assert contract["input_names"] == ["input_ids", "attention_mask"]
    assert contract["global_attention"]["fixed_in_graph"] is True
    assert contract["global_attention"]["policy"] == "none"
    by_name = {item["name"]: item for item in report["variants"]}
    assert {"onnx-android-fp32", "onnx-android-fp16", "onnx-int8"} <= set(by_name)
    int8 = by_name["onnx-int8"]
    assert int8["passed"] is True
    assert int8["quantization_delta"]["limit"] == 0.005
    assert int8["sequence_sweep"]["gated"] is False
    assert int8["sequence_sweep"]["probes"] == ["policy"]
    session = ort.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert [item.name for item in session.get_inputs()] == [
        "input_ids",
        "attention_mask",
    ]
    assert guard.attempts == []
    assert files_containing_sentinels(out, SENTINELS) == []
