from __future__ import annotations

import math

import pytest

from openmed.training import load_preset
from openmed.training.distill import (
    build_distillation_report,
    compute_kd_loss,
    decode_repaired_spans,
    soft_label_distributions,
    span_logits_from_repaired_spans,
    student_backbone_from_tiny_distill_preset,
)


def test_kd_loss_collapses_to_ce_when_alpha_zero_temperature_one():
    student_logits = [
        [[2.0, 0.0, -1.0], [0.0, 3.0, -2.0]],
    ]
    teacher_logits = [
        [[-1.0, 0.0, 2.0], [2.0, 0.5, -0.5]],
    ]
    hard_labels = [[0, 1]]

    loss = compute_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        hard_labels=hard_labels,
        student_span_logits=[[[0.0, 2.0]]],
        teacher_span_logits=[[[2.0, 0.0]]],
        temperature=1.0,
        alpha=0.0,
    )

    expected_ce = (
        sum(
            -_log_softmax(row)[label]
            for row, label in zip(student_logits[0], hard_labels[0])
        )
        / 2
    )
    assert loss.hard_ce == pytest.approx(expected_ce)
    assert loss.soft_kl > 0
    assert loss.span_transfer > 0
    assert loss.total == pytest.approx(expected_ce)
    assert loss.to_dict()["total"] == pytest.approx(expected_ce)


def test_span_logit_transfer_term_is_computed_and_weighted():
    loss = compute_kd_loss(
        student_logits=[[[3.0, 0.0], [0.0, 3.0]]],
        teacher_logits=[[[3.0, 0.0], [0.0, 3.0]]],
        hard_labels=[[0, 1]],
        student_span_logits=[[[1.0, 2.0], [3.0, 4.0]]],
        teacher_span_logits=[[[2.0, 0.0], [1.0, 4.0]]],
        temperature=2.0,
        alpha=1.0,
        span_loss_weight=0.25,
    )

    expected_span_mse = ((1.0 - 2.0) ** 2 + (2.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 4
    assert loss.soft_kl == pytest.approx(0.0)
    assert loss.span_transfer == pytest.approx(expected_span_mse)
    assert loss.total == pytest.approx(0.25 * expected_span_mse)


def test_soft_labels_use_temperature_scaled_teacher_distribution():
    soft_labels = soft_label_distributions(
        [[[2.0, 0.0], [0.0, 2.0]]],
        temperature=2.0,
    )

    expected = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
    assert soft_labels[0][0][0] == pytest.approx(expected)
    assert soft_labels[0][1][1] == pytest.approx(expected)


def test_tiny_distill_backbone_is_read_from_preset():
    recipe = load_preset("tiny_distill")

    assert student_backbone_from_tiny_distill_preset() == recipe.backbone.model_ref


def test_repaired_spans_reuse_core_decoding_and_refine_offsets():
    text = "alice@example.test and"
    spans = decode_repaired_spans(
        [[[0.0, 8.0]]],
        {0: "O", 1: "S-EMAIL"},
        texts=[text],
        offset_mapping=[[(0, len(text))]],
    )

    assert len(spans) == 1
    assert len(spans[0]) == 1
    email_span = spans[0][0]
    assert email_span.label == "EMAIL"
    assert email_span.start == 0
    assert email_span.end == len("alice@example.test")
    assert email_span.token_start == 0
    assert email_span.token_end == 1
    assert span_logits_from_repaired_spans(spans) == (((0.0, 8.0),),)


def test_distillation_report_records_recall_deltas_and_critical_drops():
    report = build_distillation_report(
        teacher_id="teacher-local",
        student_backbone="openmed/backbones/tiny-direct-identifier-135m",
        temperature=2.0,
        alpha=0.6,
        teacher_recall_by_label={"EMAIL": 0.99, "PHONE": 0.95},
        student_recall_by_label={"EMAIL": 0.97, "PHONE": 0.96},
        critical_labels=("EMAIL",),
    )

    payload = report.to_dict()
    deltas = {item["label"]: item for item in payload["per_label_recall_delta"]}
    assert deltas["EMAIL"]["delta"] == pytest.approx(-0.02)
    assert deltas["EMAIL"]["critical_drop"] is True
    assert deltas["PHONE"]["delta"] == pytest.approx(0.01)
    assert payload["critical_label_drops"] == ["EMAIL"]
    assert payload["recall_gate_passed"] is False
    assert report.model_card_evidence()["distillation"] == payload


def _log_softmax(values):
    max_value = max(values)
    total = sum(math.exp(value - max_value) for value in values)
    log_total = max_value + math.log(total)
    return [value - log_total for value in values]
