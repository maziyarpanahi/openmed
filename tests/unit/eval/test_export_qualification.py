"""Tests for the family-neutral export qualification harness."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from openmed.eval import export_qualification as eq
from openmed.eval.export_qualification import (
    ExportIOContract,
    ExportQualificationError,
    ExportReason,
    QualificationPolicy,
    RuntimeSupport,
    VariantCandidate,
)
from tests.fixtures.modernbert_export_stub import (
    LABELS,
    SENTINELS,
    rule_fixture_rows,
    write_rule_fixtures,
)

_NAME_ID, _NUMBER_ID, _OTHER_ID = 10, 11, 12
SUPPORT = RuntimeSupport(
    runtime="onnx",
    precisions=("float32", "float16"),
    quantizations=("none", "int8"),
    providers=("CPUExecutionProvider",),
    shape_modes=("dynamic",),
)
CONTRACT = ExportIOContract(
    input_names=("input_ids", "attention_mask"),
    output_names=("logits",),
    sequence_buckets=(16, 32, 64),
    max_sequence_length=96,
    shape_mode="dynamic",
    batch_size=None,
    input_dtype="int64",
)
SOURCE = {"kind": "hub_commit", "revision": "0" * 40, "model_id": "OpenMed/stub"}


class CharTokenizer:
    """Character-level fast-tokenizer stand-in with zero-width specials."""

    pad_token_id = 0
    special_tokens_map = {"pad_token": "[PAD]", "cls_token": "[CLS]"}

    def __init__(self, tag: str = "v1", *, offsets: bool = True) -> None:
        self._offsets = offsets
        self._state = {"padding": None}
        self.backend_tokenizer = SimpleNamespace(to_str=self._serialize)

    def _serialize(self) -> str:
        return json.dumps(
            {
                "model": {"type": "WordLevel", "vocab": {"[PAD]": 0, "A": 10}},
                "padding": self._state["padding"],
                "truncation": None,
            }
        )

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids, offsets = [2], [(0, 0)]
        for index, char in enumerate(text):
            if char.isspace():
                continue
            if char.isupper():
                ids.append(_NAME_ID)
            elif char.isdigit():
                ids.append(_NUMBER_ID)
            else:
                ids.append(_OTHER_ID)
            offsets.append((index, index + 1))
        ids.append(3)
        offsets.append((0, 0))
        encoded: dict[str, Any] = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if self._offsets and kwargs.get("return_offsets_mapping"):
            encoded["offset_mapping"] = offsets
        return encoded


def rule_runner(
    *, noise: float = 0.0, drop_names: bool = False, shift: bool = False
) -> Any:
    calls: list[int] = []

    def run(feed: Mapping[str, Any]) -> Any:
        calls.append(1)
        ids = np.asarray(feed["input_ids"])[0]
        logits = np.zeros((1, ids.shape[0], len(LABELS)), dtype=np.float32)
        for index, token in enumerate(ids):
            label = 0
            if token == _NAME_ID and not drop_names:
                label = 1
            elif token == _NUMBER_ID:
                label = 2
            if shift and token == _NAME_ID and index + 1 < ids.shape[0]:
                label = 1 if ids[index + 1] == _NAME_ID else 0
            logits[0, index, label] = 5.0
        if noise:
            logits += np.float32(noise)
            logits[..., 0] -= np.float32(noise)
        return logits

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _artifact_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    root.mkdir(exist_ok=True)
    model = root / "model.onnx"
    model.write_bytes(b"synthetic graph bytes")
    (root / "id2label.json").write_text(json.dumps(LABELS), encoding="utf-8")
    return root, model


def _qualify(tmp_path: Path, *candidates: VariantCandidate, **overrides: Any):
    fixtures = write_rule_fixtures(tmp_path / "fixtures.jsonl")
    root, _ = _artifact_root(tmp_path)
    kwargs: dict[str, Any] = {
        "family": "stub",
        "adapter_version": 1,
        "runtime": "onnx",
        "support": SUPPORT,
        "io_contract": CONTRACT,
        "tokenizer": CharTokenizer(),
        "id2label": LABELS,
        "reference_runner": rule_runner(),
        "candidates": candidates,
        "source": SOURCE,
        "artifact_root": root,
        "artifact_paths": [root / "id2label.json"],
        "fixtures_path": fixtures,
        "environment": {"python": "3.x", "packages": {}},
    }
    kwargs.update(overrides)
    return eq.qualify_export(**kwargs)


def _candidate(tmp_path: Path, name: str = "onnx-fp32", **kwargs: Any):
    _, model = _artifact_root(tmp_path)
    defaults: dict[str, Any] = {
        "runner": rule_runner(),
        "precision": "float32",
        "provider": "CPUExecutionProvider",
        "artifact_paths": (model,),
    }
    defaults.update(kwargs)
    return VariantCandidate(name=name, **defaults)


def test_reason_codes_are_stable_and_value_free() -> None:
    assert {reason.value for reason in ExportReason} == {
        "unsupported_family",
        "unsupported_runtime",
        "unsupported_provider",
        "unsupported_precision",
        "unsupported_quantization",
        "unsupported_sequence_length",
        "unsupported_sequence_shape",
        "bucket_unavailable",
        "sequence_too_long",
        "memory_budget_exceeded",
        "unsupported_global_attention",
        "unsupported_operator",
        "unsupported_converter_environment",
        "runtime_unavailable",
        "missing_label_map",
        "missing_offset_mapping",
        "missing_evidence",
        "source_revision_unavailable",
        "source_revision_mismatch",
        "tokenizer_identity_mismatch",
        "label_map_mismatch",
        "io_contract_mismatch",
        "artifact_missing",
        "artifact_digest_mismatch",
        "report_invalid",
        "variant_not_qualified",
        "inference_failed",
        "logit_parity_failed",
        "span_parity_failed",
        "sequence_parity_failed",
        "long_document_parity_failed",
        "direct_identifier_recall_failed",
        "critical_leakage_detected",
        "quantization_delta_exceeded",
    }
    error = ExportQualificationError(ExportReason.UNSUPPORTED_PROVIDER)
    assert isinstance(error, ValueError)
    assert error.reason_code == "unsupported_provider"
    assert str(error).startswith("unsupported_provider: ")


def test_runtime_support_rejects_each_undeclared_dimension() -> None:
    cases = {
        "precision": ({"precision": "bfloat16"}, ExportReason.UNSUPPORTED_PRECISION),
        "quantization": (
            {"quantization": "int4"},
            ExportReason.UNSUPPORTED_QUANTIZATION,
        ),
        "provider": (
            {"provider": "CUDAExecutionProvider"},
            ExportReason.UNSUPPORTED_PROVIDER,
        ),
        "shape": ({"shape_mode": "static"}, ExportReason.UNSUPPORTED_SEQUENCE_SHAPE),
    }
    for kwargs, reason in cases.values():
        with pytest.raises(ExportQualificationError) as error:
            SUPPORT.validate(**kwargs)
        assert error.value.reason is reason
    SUPPORT.validate(
        precision="float16",
        quantization="int8",
        provider="CPUExecutionProvider",
        shape_mode="dynamic",
    )


def test_io_contract_buckets_bounds_and_round_trip() -> None:
    static = ExportIOContract(
        input_names=("input_ids", "attention_mask"),
        output_names=("logits",),
        sequence_buckets=(128,),
        max_sequence_length=128,
    )
    assert static.bucket_for(1) == 128
    assert static.bucket_for(128) == 128
    assert CONTRACT.bucket_for(20) == 32
    assert CONTRACT.bucket_for(80) == 80
    for length in (0, 97):
        with pytest.raises(ExportQualificationError) as error:
            CONTRACT.bucket_for(length)
        assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH
    assert ExportIOContract.from_dict(CONTRACT.to_dict()) == CONTRACT
    assert CONTRACT.to_dict()["overflow"] == "reject"
    with pytest.raises(ExportQualificationError) as error:
        ExportIOContract(("input_ids",), ("logits",), (256,), 128)
    assert error.value.reason is ExportReason.IO_CONTRACT_MISMATCH


def test_policy_matches_documented_limits() -> None:
    fp32 = QualificationPolicy.for_variant(precision="float32")
    fp16 = QualificationPolicy.for_variant(precision="float16")
    int8 = QualificationPolicy.for_variant(precision="float32", quantization="int8")
    int4 = QualificationPolicy.for_variant(precision="float16", quantization="int4")
    assert (fp32.logit_atol, fp32.gate_logits, fp32.gate_spans) == (1e-4, True, True)
    assert fp16.logit_atol == eq.FLOAT16_LOGIT_ATOL and fp16.gate_spans
    assert (int8.gate_logits, int8.gate_spans, int8.recall_delta_limit) == (
        False,
        False,
        0.005,
    )
    assert int4.recall_delta_limit == 0.010
    with pytest.raises(ExportQualificationError) as error:
        QualificationPolicy.for_variant(precision="bfloat16")
    assert error.value.reason is ExportReason.UNSUPPORTED_PRECISION


def test_identity_digests_are_canonical() -> None:
    first = eq.label_map_identity({0: "O", 1: "I-NAME"})
    second = eq.label_map_identity({"1": "I-NAME", "0": "O"})
    assert first == second and first["num_labels"] == 2
    assert eq.label_map_identity({0: "O", 1: "B-NAME"}) != first
    with pytest.raises(ExportQualificationError) as error:
        eq.label_map_identity({})
    assert error.value.reason is ExportReason.MISSING_LABEL_MAP

    tokenizer = CharTokenizer()
    before = eq.tokenizer_identity(tokenizer)
    tokenizer._state["padding"] = {"length": 64}
    assert eq.tokenizer_identity(tokenizer) == before
    with pytest.raises(ExportQualificationError) as error:
        eq.tokenizer_identity(object())
    assert error.value.reason is ExportReason.MISSING_OFFSET_MAPPING


def test_source_identity_pins_revision_without_recording_local_paths(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "private-checkpoint-dir"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "stub"}')
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    config = {"model_type": "stub"}

    local = eq.source_identity(checkpoint, config=config)
    assert local["kind"] == "content_sha256"
    assert local["revision"].startswith("sha256:")
    assert local["model_id"] is None
    assert "private-checkpoint-dir" not in json.dumps(local)
    assert eq.source_identity(
        checkpoint, config=config, expected_revision=local["revision"]
    )

    hub = eq.source_identity(
        "OpenMed/stub", config={"model_type": "stub", "_commit_hash": "c0ffee"}
    )
    assert (hub["kind"], hub["revision"]) == ("hub_commit", "c0ffee")

    for model_id, cfg, expected, reason in (
        ("OpenMed/stub", config, None, ExportReason.SOURCE_REVISION_UNAVAILABLE),
        (checkpoint, config, "sha256:other", ExportReason.SOURCE_REVISION_MISMATCH),
    ):
        with pytest.raises(ExportQualificationError) as error:
            eq.source_identity(model_id, config=cfg, expected_revision=expected)
        assert error.value.reason is reason


def test_encode_fixtures_pads_to_buckets_and_rejects_long_or_offsetless_inputs(
    tmp_path: Path,
) -> None:
    fixtures, suite = eq.load_qualification_fixtures(
        write_rule_fixtures(tmp_path / "fixtures.jsonl")
    )
    assert suite["default"] is False and suite["fixture_count"] == len(fixtures)

    encoded = eq.encode_fixtures(fixtures, CharTokenizer(), CONTRACT)
    for item in encoded:
        assert item.feed["input_ids"].shape == (1, item.bucket)
        assert item.feed["input_ids"].dtype == np.int64
        assert sum(item.attention) == item.token_count
        assert item.offsets[item.token_count :] == ((0, 0),) * (
            item.bucket - item.token_count
        )
    assert {item.bucket for item in encoded} <= set(CONTRACT.sequence_buckets)

    tight = ExportIOContract(("input_ids",), ("logits",), (16,), 16)
    with pytest.raises(ExportQualificationError) as error:
        eq.encode_fixtures(fixtures, CharTokenizer(), tight)
    assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH
    with pytest.raises(ExportQualificationError) as error:
        eq.encode_fixtures(fixtures, CharTokenizer(offsets=False), CONTRACT)
    assert error.value.reason is ExportReason.MISSING_OFFSET_MAPPING


def test_critical_leaks_use_label_agnostic_character_coverage(tmp_path: Path) -> None:
    fixtures, _ = eq.load_qualification_fixtures(
        write_rule_fixtures(tmp_path / "fixtures.jsonl")
    )
    gold = {
        str(item.fixture_id): [
            {"label": span.label, "start": span.start, "end": span.end}
            for span in item.gold_spans
        ]
        for item in fixtures
    }
    assert eq.critical_leaks(fixtures, gold) == set()
    relabelled = {
        key: [{**span, "label": "ORGANIZATION"} for span in spans]
        for key, spans in gold.items()
    }
    assert eq.critical_leaks(fixtures, relabelled) == set()
    truncated = {
        key: [{**span, "end": span["end"] - 1} for span in spans]
        for key, spans in gold.items()
    }
    total = sum(len(item.gold_spans) for item in fixtures)
    assert len(eq.critical_leaks(fixtures, truncated)) == total


def test_identical_candidate_qualifies_and_report_is_value_free(tmp_path: Path) -> None:
    report = _qualify(tmp_path, _candidate(tmp_path))
    payload = report.to_dict()

    assert report.passed and payload["qualified_variants"] == ["onnx-fp32"]
    variant = payload["variants"][0]
    assert variant["claim"] == "qualified"
    assert variant["logit_parity"]["max_abs_diff"] == 0.0
    assert variant["span_parity"]["identical"] is True
    assert variant["direct_identifier_recall"]["candidate"] == {
        "ID_NUM": 1.0,
        "PERSON": 1.0,
    }
    assert variant["critical_leakage"]["new_leaks"] == 0
    assert set(variant["buckets_evaluated"]) <= {16, 32, 64}
    assert payload["offset_contract"]["unit"] == "unicode_code_point"
    assert payload["io_contract"]["input_names"] == ["input_ids", "attention_mask"]
    assert set(payload["artifacts"]) == {"id2label.json", "model.onnx"}
    serialized = json.dumps(payload)
    assert not any(value in serialized for value in SENTINELS)
    assert str(tmp_path) not in serialized


def test_float_gates_catch_logit_and_span_drift(tmp_path: Path) -> None:
    noisy = _candidate(tmp_path, runner=rule_runner(noise=1e-3))
    assert _qualify(tmp_path, noisy).reasons == (
        ExportReason.LOGIT_PARITY_FAILED.value,
    )

    fp16 = _candidate(
        tmp_path, "webgpu-fp16", runner=rule_runner(noise=1e-3), precision="float16"
    )
    assert _qualify(tmp_path, fp16).passed

    shifted = _candidate(tmp_path, runner=rule_runner(shift=True))
    reasons = _qualify(tmp_path, shifted).reasons
    assert ExportReason.SPAN_PARITY_FAILED.value in reasons
    assert ExportReason.CRITICAL_LEAKAGE_DETECTED.value in reasons


def test_quantized_variant_ships_only_when_recall_and_leakage_hold(
    tmp_path: Path,
) -> None:
    good = _candidate(
        tmp_path, "onnx-int8", runner=rule_runner(noise=0.5), quantization="int8"
    )
    passed = _qualify(tmp_path, good).variants[0]
    assert passed.passed and passed.quantization_delta["passed"] is True
    assert passed.logit_parity["gated"] is False

    degraded = _candidate(
        tmp_path,
        "onnx-int8",
        runner=rule_runner(drop_names=True),
        quantization="int8",
    )
    report = _qualify(tmp_path, good, degraded)
    failed = report.variants[1]
    assert not report.passed
    assert failed.reasons == (
        ExportReason.QUANTIZATION_DELTA_EXCEEDED.value,
        ExportReason.CRITICAL_LEAKAGE_DETECTED.value,
    )
    assert failed.critical_leakage["new_leaks_by_label"] == {"PERSON": 6}
    assert failed.to_dict()["claim"] == "rejected"
    with pytest.raises(ExportQualificationError) as error:
        report.raise_for_failure()
    assert error.value.reason is ExportReason.QUANTIZATION_DELTA_EXCEEDED


def test_runtime_failures_are_recorded_without_echoing_inputs(tmp_path: Path) -> None:
    def crashing(feed: Mapping[str, Any]) -> Any:
        raise RuntimeError(f"kernel failed on {SENTINELS[0]}")

    def wrong_shape(feed: Mapping[str, Any]) -> Any:
        return np.zeros((1, 2, len(LABELS)), dtype=np.float32)

    report = _qualify(
        tmp_path,
        _candidate(tmp_path, "onnx-fp32", runner=crashing),
        _candidate(tmp_path, "webgpu-fp16", runner=wrong_shape, precision="float16"),
    )
    assert [item.reasons for item in report.variants] == [
        ("inference_failed",),
        ("io_contract_mismatch",),
    ]
    assert SENTINELS[0] not in json.dumps(report.to_dict())


def test_unsupported_combinations_fail_before_candidate_inference(
    tmp_path: Path,
) -> None:
    runner = rule_runner()
    cuda = _candidate(tmp_path, runner=runner, provider="CUDAExecutionProvider")
    with pytest.raises(ExportQualificationError) as error:
        _qualify(tmp_path, cuda)
    assert error.value.reason is ExportReason.UNSUPPORTED_PROVIDER
    assert runner.calls == []
    int4 = _candidate(tmp_path, runner=runner, quantization="int4")
    with pytest.raises(ExportQualificationError) as error:
        _qualify(tmp_path, int4)
    assert error.value.reason is ExportReason.UNSUPPORTED_QUANTIZATION
    assert runner.calls == []


def test_report_is_deterministic_and_verifies_offline(tmp_path: Path) -> None:
    first = _qualify(tmp_path, _candidate(tmp_path)).to_dict()
    second = _qualify(tmp_path, _candidate(tmp_path)).to_dict()
    assert first["stable_sha256"] == second["stable_sha256"]

    root = tmp_path / "bundle"
    path = eq.write_qualification_report(first, root / eq.EXPORT_QUALIFICATION_FILENAME)
    verified = eq.verify_qualification_report(
        path, artifact_root=root, tokenizer=CharTokenizer(), id2label=LABELS
    )
    assert verified["passed"] is True

    with pytest.raises(ExportQualificationError) as error:
        eq.verify_qualification_report(
            path, artifact_root=root, id2label={0: "O", 1: "I-PERSON", 2: "I-ID"}
        )
    assert error.value.reason is ExportReason.LABEL_MAP_MISMATCH

    class OtherTokenizer(CharTokenizer):
        def _serialize(self) -> str:
            return json.dumps({"model": {"type": "WordPiece", "vocab": {}}})

    with pytest.raises(ExportQualificationError) as error:
        eq.verify_qualification_report(
            path, artifact_root=root, tokenizer=OtherTokenizer()
        )
    assert error.value.reason is ExportReason.TOKENIZER_IDENTITY_MISMATCH

    tampered = dict(first, passed=False)
    with pytest.raises(ExportQualificationError) as error:
        eq.verify_qualification_report(tampered, artifact_root=root)
    assert error.value.reason is ExportReason.REPORT_INVALID

    (root / "model.onnx").write_bytes(b"swapped graph")
    with pytest.raises(ExportQualificationError) as error:
        eq.verify_qualification_report(path, artifact_root=root)
    assert error.value.reason is ExportReason.ARTIFACT_DIGEST_MISMATCH

    (root / "model.onnx").unlink()
    with pytest.raises(ExportQualificationError) as error:
        eq.verify_qualification_report(path, artifact_root=root)
    assert error.value.reason is ExportReason.ARTIFACT_MISSING


def test_missing_identity_fields_make_the_report_invalid(tmp_path: Path) -> None:
    payload = _qualify(tmp_path, _candidate(tmp_path)).to_dict()
    for key in ("source", "tokenizer", "label_map"):
        broken = json.loads(json.dumps(payload))
        broken[key] = {}
        with pytest.raises(ExportQualificationError) as error:
            eq.check_qualified_request(broken, runtime="onnx", variant="onnx-fp32")
        assert error.value.reason is ExportReason.REPORT_INVALID


def test_check_qualified_request_gates_runtime_variant_provider_and_length(
    tmp_path: Path,
) -> None:
    fp16 = _candidate(
        tmp_path,
        "webgpu-fp16",
        precision="float16",
        target_provider="WebGpuExecutionProvider",
    )
    bad = _candidate(
        tmp_path, "onnx-int8", runner=rule_runner(drop_names=True), quantization="int8"
    )
    payload = _qualify(tmp_path, _candidate(tmp_path), fp16, bad).to_dict()

    entry = eq.check_qualified_request(
        payload,
        runtime="onnx",
        artifact="model.onnx",
        providers=["CPUExecutionProvider"],
        sequence_length=96,
    )
    assert entry["name"] == "onnx-fp32"
    proxy = eq.check_qualified_request(payload, runtime="onnx", variant="webgpu-fp16")
    assert proxy["claim"] == "cpu_proxy"

    rejections = (
        ({"runtime": "coreml", "variant": "onnx-fp32"}, "unsupported_runtime"),
        ({"runtime": "onnx", "variant": "onnx-int8"}, "variant_not_qualified"),
        ({"runtime": "onnx", "variant": "missing"}, "variant_not_qualified"),
        (
            {
                "runtime": "onnx",
                "variant": "webgpu-fp16",
                "providers": ["WebGpuExecutionProvider"],
            },
            "unsupported_provider",
        ),
        (
            {"runtime": "onnx", "variant": "onnx-fp32", "sequence_length": 97},
            "unsupported_sequence_length",
        ),
    )
    for kwargs, reason in rejections:
        with pytest.raises(ExportQualificationError) as error:
            eq.check_qualified_request(payload, **kwargs)
        assert error.value.reason_code == reason


def test_summary_block_is_compact_and_value_free(tmp_path: Path) -> None:
    report = _qualify(tmp_path, _candidate(tmp_path))
    summary = eq.summarize_report(report)
    assert summary["passed"] is True
    assert summary["source_revision"] == SOURCE["revision"]
    assert summary["variants"]["onnx-fp32"]["claim"] == "qualified"
    assert summary["stable_sha256"] == report.to_dict()["stable_sha256"]


def test_packaged_suite_is_synthetic_and_offset_exact() -> None:
    fixtures, suite = eq.load_qualification_fixtures()
    assert suite["default"] is True and suite["synthetic"] is True
    assert suite["fixture_count"] == len(fixtures) >= 10
    assert set(eq.direct_identifier_labels(fixtures)) >= {
        "PERSON",
        "ID_NUM",
        "PHONE",
        "EMAIL",
        "SSN",
    }
    assert set(suite["languages"]) >= {"en", "es", "fr", "de"}
    for fixture in fixtures:
        assert fixture.metadata.get("synthetic") is True
        for span in fixture.gold_spans:
            assert fixture.text[span.start : span.end] == span.text


def test_rule_fixture_rows_match_gold_offsets() -> None:
    for row in rule_fixture_rows():
        for span in row["gold_spans"]:
            assert row["text"][span["start"] : span["end"]] == span["text"]


# ---------------------------------------------------------------------------
# Global attention, sequence sweeps, and long-document windowing (Longformer)
# ---------------------------------------------------------------------------

WINDOWED_CONTRACT = ExportIOContract(
    input_names=("input_ids", "attention_mask", "global_attention_mask"),
    output_names=("logits",),
    sequence_buckets=(16, 32, 48, 64),
    max_sequence_length=64,
    shape_mode="dynamic",
    batch_size=None,
    input_dtype="int64",
    global_attention="none",
    windowing=eq.WindowingPolicy.for_window(64),
)
WINDOWED_LIMITS = {"memory_ceiling_bytes": 1024, "declared_buckets": [16, 32]}


def global_runner(*, use_global: bool = True, **kwargs: Any) -> Any:
    """Rule runner whose logits also read the global attention mask."""
    inner = rule_runner(**kwargs)

    def run(feed: Mapping[str, Any]) -> Any:
        logits = inner(feed)
        if use_global and "global_attention_mask" in feed:
            weight = float(np.asarray(feed["global_attention_mask"]).sum())
            logits[0, :, 0] += np.float32(0.01 * weight)
        return logits

    run.calls = inner.calls  # type: ignore[attr-defined]
    return run


def full_window_runner(feed: Mapping[str, Any]) -> Any:
    """Drops names only in completely filled 64-token windows."""
    logits = rule_runner()(feed)
    mask = np.asarray(feed["attention_mask"])[0]
    if mask.shape[0] == 64 and mask.all():
        logits[0, :, 1] = 0.0
    return logits


def _variant(tmp_path: Path, name: str, runner: Any, **kwargs: Any):
    return _candidate(tmp_path, name, runner=runner, **kwargs)


def _qualify_windowed(tmp_path: Path, *candidates: VariantCandidate, **overrides):
    options: dict[str, Any] = {
        "io_contract": WINDOWED_CONTRACT,
        "reference_runner": global_runner(),
        "limits": WINDOWED_LIMITS,
    }
    options.update(overrides)
    return _qualify(tmp_path, *candidates, **options)


def test_global_attention_rows_follow_declared_policies() -> None:
    assert eq.global_attention_row("none", 3, 5) == [0, 0, 0, 0, 0]
    assert eq.global_attention_row("first_token", 3, 5) == [1, 0, 0, 0, 0]
    assert eq.global_attention_row("first_token", 0, 4) == [0, 0, 0, 0]
    with pytest.raises(ExportQualificationError) as error:
        eq.global_attention_row("all_tokens", 3, 5)
    assert error.value.reason is ExportReason.UNSUPPORTED_GLOBAL_ATTENTION


def test_windowing_policy_defaults_bounds_and_round_trip() -> None:
    policy = eq.WindowingPolicy.for_window(512)
    assert (policy.window_tokens, policy.overlap_tokens) == (512, 96)
    assert policy.max_document_tokens == 512 * eq.DEFAULT_MAX_DOCUMENT_WINDOWS
    assert eq.WindowingPolicy.for_window(64).overlap_tokens == 15
    assert eq.default_window_overlap(64) == 15
    payload = policy.to_dict()
    assert payload["truncation"] == "never"
    assert payload["overflow"] == "sequence_too_long"
    assert payload["offsets"] == "document_global"
    assert eq.WindowingPolicy.from_dict(payload) == policy
    policy.check_document(policy.max_document_tokens)
    with pytest.raises(ExportQualificationError) as error:
        policy.check_document(policy.max_document_tokens + 1)
    assert error.value.reason is ExportReason.SEQUENCE_TOO_LONG
    for kwargs in (
        {"window_tokens": 0, "overlap_tokens": 0, "max_document_tokens": 8},
        {"window_tokens": 8, "overlap_tokens": 8, "max_document_tokens": 8},
        {"window_tokens": 8, "overlap_tokens": 2, "max_document_tokens": 7},
    ):
        with pytest.raises(ExportQualificationError) as error:
            eq.WindowingPolicy(**kwargs)
        assert error.value.reason is ExportReason.IO_CONTRACT_MISMATCH


def test_io_contract_declares_global_attention_and_windowing() -> None:
    payload = WINDOWED_CONTRACT.to_dict()
    assert payload["input_names"][-1] == "global_attention_mask"
    assert payload["global_attention"] == {
        "policy": "none",
        "input": "global_attention_mask",
        "fixed_in_graph": False,
        "values": [0, 1],
        "padding": "never_global",
    }
    assert payload["windowing"]["window_tokens"] == 64
    assert ExportIOContract.from_dict(payload) == WINDOWED_CONTRACT
    fixed = ExportIOContract(
        input_names=("input_ids", "attention_mask"),
        output_names=("logits",),
        sequence_buckets=(64,),
        max_sequence_length=64,
        global_attention="none",
    )
    assert fixed.to_dict()["global_attention"]["fixed_in_graph"] is True
    base: dict[str, Any] = {
        "output_names": ("logits",),
        "sequence_buckets": (64,),
        "max_sequence_length": 64,
    }
    cases = (
        (
            {"input_names": ("input_ids", "attention_mask", "global_attention_mask")},
            ExportReason.IO_CONTRACT_MISMATCH,
        ),
        (
            {"input_names": ("input_ids",), "global_attention": "everything"},
            ExportReason.UNSUPPORTED_GLOBAL_ATTENTION,
        ),
        (
            {
                "input_names": ("input_ids",),
                "windowing": eq.WindowingPolicy.for_window(128),
            },
            ExportReason.IO_CONTRACT_MISMATCH,
        ),
    )
    for overrides, reason in cases:
        with pytest.raises(ExportQualificationError) as error:
            ExportIOContract(**base, **overrides)
        assert error.value.reason is reason


def test_build_feed_fills_global_attention_and_never_marks_padding() -> None:
    contract = ExportIOContract(
        input_names=("input_ids", "attention_mask", "global_attention_mask"),
        output_names=("logits",),
        sequence_buckets=(8,),
        max_sequence_length=8,
        global_attention="first_token",
    )
    feed = eq.build_feed(contract, [2, 10, 3], bucket=8, pad_id=0)
    assert feed["input_ids"].tolist() == [[2, 10, 3, 0, 0, 0, 0, 0]]
    assert feed["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]
    assert feed["global_attention_mask"].tolist() == [[1, 0, 0, 0, 0, 0, 0, 0]]
    assert feed["global_attention_mask"].dtype == np.int32
    explicit = eq.build_feed(
        contract, [2, 10, 3], bucket=8, global_attention=[0, 1, 1, 0, 0, 0, 0, 0]
    )
    assert explicit["global_attention_mask"].tolist()[0][:3] == [0, 1, 1]
    with pytest.raises(ExportQualificationError) as error:
        eq.build_feed(contract, [2, 10, 3], bucket=8, global_attention=[0] * 7 + [1])
    assert error.value.reason is ExportReason.IO_CONTRACT_MISMATCH
    with pytest.raises(ExportQualificationError) as error:
        eq.build_feed(contract, [2] * 9, bucket=8)
    assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH


def test_sweep_lengths_cover_minimum_typical_boundaries_and_maximum() -> None:
    lengths = eq.sequence_sweep_lengths(
        WINDOWED_CONTRACT, minimum=3, typical=30, boundaries=(4, 8, 16)
    )
    assert lengths[0] == 3 and lengths[-1] == 64
    assert {3, 30, 64} <= set(lengths)
    assert {4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63} <= set(lengths)
    assert 65 not in lengths
    static = ExportIOContract(
        input_names=("input_ids",),
        output_names=("logits",),
        sequence_buckets=(64,),
        max_sequence_length=64,
    )
    assert eq.sequence_sweep_lengths(static, minimum=3, typical=3) == (3, 63, 64)


def test_sequence_sweep_catches_a_graph_that_ignores_global_attention(
    tmp_path: Path,
) -> None:
    good = _variant(tmp_path, "onnx-fp32", global_runner())
    deaf = _variant(tmp_path, "onnx-deaf", global_runner(use_global=False))
    report = _qualify_windowed(
        tmp_path, good, deaf, sequence_sweep_boundaries=(4, 8, 16)
    )
    passed, failed = report.variants
    assert passed.passed, passed.reasons
    sweep = passed.sequence_sweep
    assert sweep["within_tolerance"] is True and sweep["passed"] is True
    assert sweep["probes"] == ["global_probe", "policy"]
    assert sweep["global_probe_positions"] == ["first", "middle", "last"]
    assert sweep["lengths"][0] == 3 and sweep["lengths"][-1] == 64
    assert {7, 8, 9, 15, 16, 17} <= set(sweep["lengths"])
    assert sweep["cases"] == 2 * len(sweep["lengths"])
    # The policy fill is "none", so the main gates cannot see the bug; only
    # the global-attention probe in the sweep can.
    assert failed.logit_parity["within_tolerance"] is True
    assert failed.reasons == (ExportReason.SEQUENCE_PARITY_FAILED.value,)
    assert failed.sequence_sweep["failed_lengths"] == sweep["lengths"]
    payload = report.to_dict()
    assert payload["limits"] == WINDOWED_LIMITS
    serialized = json.dumps(payload)
    assert not any(value in serialized for value in SENTINELS)


def test_quantized_sweep_drift_is_recorded_but_not_gated(tmp_path: Path) -> None:
    noisy = _variant(
        tmp_path, "onnx-int8", global_runner(noise=0.5), quantization="int8"
    )
    report = _qualify_windowed(tmp_path, noisy, sequence_sweep_boundaries=())
    variant = report.variants[0]
    assert variant.passed, variant.reasons
    assert variant.sequence_sweep["within_tolerance"] is False
    assert variant.sequence_sweep["gated"] is False


def test_long_documents_compare_document_global_spans_across_overlaps(
    tmp_path: Path,
) -> None:
    good = _variant(tmp_path, "onnx-fp32", global_runner())
    edge = _variant(tmp_path, "onnx-edge", full_window_runner)
    report = _qualify_windowed(tmp_path, good, edge)
    passed, failed = report.variants
    assert passed.passed, passed.reasons
    long_document = passed.long_document
    assert long_document["documents"] == 2
    assert long_document["windows"] >= 2 * 2
    assert long_document["window_tokens"] == 64
    assert long_document["overlap_tokens"] == 15
    assert long_document["tokens"] >= 2 * 150
    assert long_document["tokens_in_overlaps"] > 0
    assert long_document["reference_spans_in_overlaps"] >= 1
    assert long_document["span_parity"]["identical"] is True
    assert long_document["new_critical_leaks"] == 0
    # Single-window gates pass; only windowed documents expose the defect.
    assert failed.span_parity["identical"] is True
    assert failed.reasons == (
        ExportReason.LONG_DOCUMENT_PARITY_FAILED.value,
        ExportReason.CRITICAL_LEAKAGE_DETECTED.value,
    )
    assert failed.long_document["new_critical_leaks_by_label"]["PERSON"] > 0
    serialized = json.dumps(report.to_dict())
    assert not any(value in serialized for value in SENTINELS)


def test_long_document_runtime_failure_is_recorded_not_raised(
    tmp_path: Path,
) -> None:
    def fails_on_full_windows(feed: Mapping[str, Any]) -> Any:
        mask = np.asarray(feed["attention_mask"])[0]
        if mask.shape[0] == 64 and mask.all():
            raise RuntimeError(f"kernel failed near {SENTINELS[1]}")
        return rule_runner()(feed)

    candidate = _variant(tmp_path, "onnx-fp32", fails_on_full_windows)
    variant = _qualify_windowed(tmp_path, candidate).variants[0]
    assert variant.reasons == (ExportReason.INFERENCE_FAILED.value,)
    assert variant.long_document["passed"] is False
    assert SENTINELS[1] not in json.dumps(variant.to_dict())


def test_long_documents_respect_the_declared_document_bound(tmp_path: Path) -> None:
    tight = ExportIOContract(
        input_names=WINDOWED_CONTRACT.input_names,
        output_names=("logits",),
        sequence_buckets=(16, 32, 48, 64),
        max_sequence_length=64,
        shape_mode="dynamic",
        batch_size=None,
        input_dtype="int64",
        global_attention="none",
        windowing=eq.WindowingPolicy(
            window_tokens=64, overlap_tokens=15, max_document_tokens=128
        ),
    )
    with pytest.raises(ExportQualificationError) as error:
        _qualify_windowed(tmp_path, _candidate(tmp_path), io_contract=tight)
    assert error.value.reason is ExportReason.SEQUENCE_TOO_LONG


def test_check_qualified_request_enforces_document_bound_and_policy(
    tmp_path: Path,
) -> None:
    candidate = _variant(tmp_path, "onnx-fp32", global_runner())
    payload = _qualify_windowed(tmp_path, candidate).to_dict()
    bound = payload["io_contract"]["windowing"]["max_document_tokens"]
    entry = eq.check_qualified_request(
        payload,
        runtime="onnx",
        variant="onnx-fp32",
        sequence_length=64,
        document_tokens=bound,
        global_attention="none",
    )
    assert entry["passed"] is True
    rejections = (
        ({"document_tokens": bound + 1}, "sequence_too_long"),
        ({"sequence_length": 65}, "unsupported_sequence_length"),
        ({"global_attention": "first_token"}, "unsupported_global_attention"),
    )
    for kwargs, reason in rejections:
        with pytest.raises(ExportQualificationError) as error:
            eq.check_qualified_request(
                payload, runtime="onnx", variant="onnx-fp32", **kwargs
            )
        assert error.value.reason_code == reason
    # Without windowing a whole document must fit one window.
    plain = _qualify(tmp_path, _candidate(tmp_path)).to_dict()
    with pytest.raises(ExportQualificationError) as error:
        eq.check_qualified_request(
            plain, runtime="onnx", variant="onnx-fp32", document_tokens=97
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH


def test_summary_carries_bounds_only_for_bounded_families(tmp_path: Path) -> None:
    candidate = _variant(tmp_path, "onnx-fp32", global_runner())
    bounded = eq.summarize_report(_qualify_windowed(tmp_path, candidate))
    assert bounded["global_attention"]["policy"] == "none"
    assert bounded["windowing"]["window_tokens"] == 64
    assert bounded["input_names"][-1] == "global_attention_mask"
    assert bounded["max_sequence_length"] == 64
    assert bounded["limits"]["memory_ceiling_bytes"] == 1024
    plain = eq.summarize_report(_qualify(tmp_path, _candidate(tmp_path)))
    assert not {"global_attention", "windowing", "limits"} & set(plain)
