from __future__ import annotations

import pytest

from openmed.eval.quant_delta import (
    evaluate_coreml_span_parity,
    evaluate_export_variant_gate,
    evaluate_onnx_logit_parity,
    evaluate_quant_recall_delta,
    evaluate_speculative_redaction_parity,
    token_frequency_kl,
)

_EXPORT_TIER_BUDGETS = {
    "tiny": {"ram_mb": 350.0, "p50_ms": 60.0, "p95_ms": 150.0},
    "base": {"ram_mb": 900.0, "p50_ms": 150.0, "p95_ms": 400.0},
}
_EXPORT_PARENT_RECALL = {
    "PERSON": 0.995,
    "DATE": 0.995,
    "ID_NUM": 0.995,
}


def _export_variant(
    fmt: str,
    tier: str,
    *,
    recall: dict[str, float] | None = None,
    p50_ms: float = 40.0,
    p95_ms: float = 120.0,
    ram_mb: float = 300.0,
    **overrides: object,
) -> dict[str, object]:
    variant: dict[str, object] = {
        "format": fmt,
        "tier": tier,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "ram_mb": ram_mb,
    }
    if recall is not None:
        variant["recall"] = recall
    variant.update(overrides)
    return variant


def _passing_export_variants() -> list[dict[str, object]]:
    return [
        _export_variant("onnx", "base", p50_ms=90.0, p95_ms=280.0, ram_mb=700.0),
        _export_variant(
            "onnx-int8",
            "tiny",
            recall={"PERSON": 0.992, "DATE": 0.993, "ID_NUM": 0.994},
        ),
        _export_variant("webgpu", "base", p50_ms=90.0, p95_ms=280.0, ram_mb=700.0),
    ]


def test_export_variant_gate_releases_passing_onnx_and_webgpu() -> None:
    result = evaluate_export_variant_gate(
        variants=_passing_export_variants(),
        tier_budgets=_EXPORT_TIER_BUDGETS,
        parent_recall=_EXPORT_PARENT_RECALL,
        required_variants=["onnx", "onnx-int8", "webgpu"],
    )

    assert result.passed is True
    assert result.blocked_formats == ()
    assert result.missing_required == ()
    assert set(result.evaluated_formats) == {"onnx", "onnx-int8", "webgpu"}


def test_export_variant_gate_blocks_degraded_onnx_int8_only() -> None:
    variants = _passing_export_variants()
    variants[1]["recall"] = {"PERSON": 0.985, "DATE": 0.993, "ID_NUM": 0.994}

    result = evaluate_export_variant_gate(
        variants=variants,
        tier_budgets=_EXPORT_TIER_BUDGETS,
        parent_recall=_EXPORT_PARENT_RECALL,
        required_variants=["onnx", "onnx-int8", "webgpu"],
    )

    assert result.passed is False
    assert result.blocked_formats == ("onnx-int8",)
    passing = {r.format for r in result.variant_results if r.passed}
    assert passing == {"onnx", "webgpu"}


def test_export_variant_gate_blocks_webgpu_tier_budget_breach() -> None:
    variants = _passing_export_variants()
    variants[2]["p95_ms"] = 500.0

    result = evaluate_export_variant_gate(
        variants=variants,
        tier_budgets=_EXPORT_TIER_BUDGETS,
        parent_recall=_EXPORT_PARENT_RECALL,
    )

    assert result.passed is False
    assert result.blocked_formats == ("webgpu",)
    webgpu = next(r for r in result.variant_results if r.format == "webgpu")
    assert webgpu.tier_fit_passed is False
    assert "p95_ms" in webgpu.tier_violations


def test_export_variant_gate_fails_closed_on_missing_tier_evidence() -> None:
    variants = [
        {
            "format": "onnx-int8",
            "tier": "tiny",
            "recall": {"PERSON": 0.992},
        }
    ]

    result = evaluate_export_variant_gate(
        variants=variants,
        tier_budgets=_EXPORT_TIER_BUDGETS,
        parent_recall=_EXPORT_PARENT_RECALL,
    )

    assert result.passed is False
    assert result.blocked_formats == ("onnx-int8",)


def test_export_variant_gate_fails_closed_on_missing_required_variant() -> None:
    variants = [
        _export_variant("onnx", "base", p50_ms=90.0, p95_ms=280.0, ram_mb=700.0)
    ]

    result = evaluate_export_variant_gate(
        variants=variants,
        tier_budgets=_EXPORT_TIER_BUDGETS,
        parent_recall=_EXPORT_PARENT_RECALL,
        required_variants=["onnx", "webgpu"],
    )

    assert result.passed is False
    assert result.missing_required == ("webgpu",)


def test_int8_delta_below_half_point_passes() -> None:
    result = evaluate_quant_recall_delta(
        format_name="mlx-8bit",
        candidate_recall={"PERSON": 0.986},
        parent_recall={"PERSON": 0.990},
    )

    assert result.passed is True
    assert result.max_delta == pytest.approx(0.004)
    assert result.blocking_format is None


def test_int8_delta_at_half_point_fails() -> None:
    result = evaluate_quant_recall_delta(
        format_name="mlx-8bit",
        candidate_recall={"PERSON": 0.985},
        parent_recall={"PERSON": 0.990},
    )

    assert result.passed is False
    assert result.blocking_format == "mlx-8bit"
    assert result.offending_labels["PERSON"]["limit"] == 0.005


def test_int4_delta_at_one_point_fails() -> None:
    result = evaluate_quant_recall_delta(
        format_name="mlx-4bit",
        candidate_recall={"DATE": 0.970},
        parent_recall={"DATE": 0.980},
    )

    assert result.passed is False
    assert result.blocking_format == "mlx-4bit"
    assert result.offending_labels["DATE"]["limit"] == 0.010


def test_precomputed_per_format_delta_blocks_only_that_format() -> None:
    int8 = evaluate_quant_recall_delta(
        format_name="mlx-8bit",
        candidate_recall={"PERSON": 0.99},
        precomputed_delta={"mlx-8bit": {"PERSON": 0.006}, "mlx-4bit": 0.0},
    )
    int4 = evaluate_quant_recall_delta(
        format_name="mlx-4bit",
        candidate_recall={"PERSON": 0.99},
        precomputed_delta={"mlx-8bit": {"PERSON": 0.006}, "mlx-4bit": 0.0},
    )

    assert int8.passed is False
    assert int8.blocking_format == "mlx-8bit"
    assert int4.passed is True
    assert int4.blocking_format is None


def test_coreml_span_parity_passes_identical_spans_and_three_tenths_point() -> None:
    result = evaluate_coreml_span_parity(
        format_name="coreml-int8",
        reference_spans={
            "note": [{"label": "PERSON", "start": 8, "end": 16, "text": "John Doe"}]
        },
        candidate_spans={
            "note": [{"label": "PERSON", "start": 8, "end": 16, "text": "John Doe"}]
        },
        reference_recall={"PERSON": 0.990},
        candidate_recall={"PERSON": 0.987},
    )

    assert result.passed is True
    assert result.max_recall_delta == pytest.approx(0.003)


def test_coreml_int4_parity_auto_rejects_with_clear_reason() -> None:
    result = evaluate_coreml_span_parity(
        format_name="coreml-int4",
        reference_spans={
            "note": [{"label": "PERSON", "start": 8, "end": 16, "text": "John Doe"}]
        },
        candidate_spans={"note": []},
        reference_recall={"PERSON": 1.0},
        candidate_recall={"PERSON": 0.0},
        rejectable=True,
    )

    assert result.passed is False
    assert result.auto_rejected is True
    assert "span parity mismatch" in (result.rejection_reason or "")
    assert "recall delta exceeds limit" in (result.rejection_reason or "")


def test_onnx_logit_parity_passes_when_logits_and_spans_match() -> None:
    result = evaluate_onnx_logit_parity(
        baseline_logits=[[[0.99, 0.01, 0.0], [0.01, 0.98, 0.01], [0.01, 0.03, 0.96]]],
        candidate_logits=[
            [[0.99001, 0.00999, 0.0], [0.01, 0.98001, 0.01], [0.01, 0.03, 0.96]]
        ],
        id2label={"0": "O", "1": "B-NAME", "2": "I-NAME"},
        offsets=[[[0, 0], [8, 12], [12, 16]]],
        atol=1e-3,
        rtol=1e-3,
    )

    assert result.passed is True
    assert result.logits_within_tolerance is True
    assert result.spans_identical is True
    assert result.span_count == 1


def test_onnx_logit_parity_fails_when_token_predictions_change() -> None:
    result = evaluate_onnx_logit_parity(
        baseline_logits=[[[0.1, 0.9], [0.8, 0.2]]],
        candidate_logits=[[[0.9, 0.1], [0.8, 0.2]]],
        atol=1.0,
        rtol=1.0,
    )

    assert result.passed is False
    assert result.logits_within_tolerance is True
    assert result.spans_identical is False
    assert result.token_mismatches == 1


def test_speculative_redaction_parity_passes_with_equivalent_evidence() -> None:
    result = evaluate_speculative_redaction_parity(
        reference_greedy_outputs=["Patient: [NAME]", "SSN: [SSN]"],
        speculative_greedy_outputs=["Patient: [NAME]", "SSN: [SSN]"],
        reference_sampling_counts={"NAME": 70, "SSN": 30},
        speculative_sampling_counts={"NAME": 70, "SSN": 30},
        reference_latency_ms=[10.0, 11.0, 12.0],
        speculative_latency_ms=[5.0, 6.0, 7.0],
        reference_recall={"PERSON": 1.0, "SSN": 1.0},
        speculative_recall={"PERSON": 1.0, "SSN": 1.0},
        reference_leak_count=0,
        speculative_leak_count=0,
        tokenizer_fallback_count=1,
        tokenizer_fallback_correct=True,
    )

    assert result.passed is True
    assert result.greedy_mismatch_count == 0
    assert result.sampling_kl == pytest.approx(0.0)
    assert result.median_latency_speedup == pytest.approx(11.0 / 6.0)
    assert result.max_recall_delta == 0.0
    assert result.new_leak_count == 0


def test_speculative_redaction_parity_blocks_mismatches_and_new_leaks() -> None:
    result = evaluate_speculative_redaction_parity(
        reference_greedy_outputs=["Patient: [NAME]"],
        speculative_greedy_outputs=["Patient: Alice"],
        reference_sampling_counts={"NAME": 1},
        speculative_sampling_counts={"NAME": 1},
        reference_latency_ms=[10.0],
        speculative_latency_ms=[5.0],
        reference_recall={"PERSON": 1.0},
        speculative_recall={"PERSON": 0.99},
        reference_leak_count=0,
        speculative_leak_count=1,
    )

    assert result.passed is False
    assert result.greedy_mismatch_count == 1
    assert result.greedy_mismatch_indices == (0,)
    assert result.max_recall_delta == pytest.approx(0.01)
    assert result.new_leak_count == 1


def test_token_frequency_kl_detects_distribution_shift() -> None:
    assert token_frequency_kl({"A": 5}, {"A": 5}) == pytest.approx(0.0)
    assert token_frequency_kl({"A": 5}, {"B": 5}) > 20.0
