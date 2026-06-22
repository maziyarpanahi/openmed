from __future__ import annotations

import pytest

from openmed.eval.quant_delta import evaluate_quant_recall_delta


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
