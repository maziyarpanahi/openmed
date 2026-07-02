from __future__ import annotations

import pytest

from openmed.eval.quant_delta import (
    evaluate_coreml_span_parity,
    evaluate_quant_recall_delta,
)


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
