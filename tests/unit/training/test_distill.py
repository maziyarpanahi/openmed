from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from openmed.training import load_preset
from openmed.training.distill import (
    EntityTypeWeights,
    LabeledSpan,
    ModeADistillationPipeline,
    build_distillation_report,
    compute_kd_loss,
    compute_span_agreement_loss,
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


def test_kd_loss_alpha_zero_does_not_require_teacher_outputs():
    student_logits = [
        [[2.0, 0.0, -1.0], [0.0, 3.0, -2.0]],
    ]
    hard_labels = [[0, 1]]

    loss = compute_kd_loss(
        student_logits=student_logits,
        hard_labels=hard_labels,
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
    assert loss.soft_kl == pytest.approx(0.0)
    assert loss.total == pytest.approx(expected_ce)


def test_kd_loss_requires_teacher_outputs_when_teacher_weight_is_active():
    with pytest.raises(ValueError, match="alpha > 0"):
        compute_kd_loss(
            student_logits=[[[2.0, 0.0], [0.0, 2.0]]],
            hard_labels=[[0, 1]],
            temperature=1.0,
            alpha=0.5,
        )


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


def test_entity_type_weights_upweight_rare_label_kl_rows():
    student_logits = [
        [
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
    ]
    teacher_logits = [
        [
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
    ]
    hard_labels = [[1, 2]]
    id2label = {0: "O", 1: "B-EMAIL", 2: "B-NAME"}

    unweighted = compute_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        hard_labels=hard_labels,
        temperature=1.0,
        alpha=1.0,
        id2label=id2label,
    )
    weighted = compute_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        hard_labels=hard_labels,
        temperature=1.0,
        alpha=1.0,
        id2label=id2label,
        entity_type_weights={"EMAIL": 8.0},
    )

    assert weighted.soft_kl > unweighted.soft_kl
    assert (
        weighted.to_dict()
        == compute_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            hard_labels=hard_labels,
            temperature=1.0,
            alpha=1.0,
            id2label=id2label,
            entity_type_weights=EntityTypeWeights({"EMAIL": 8.0}),
        ).to_dict()
    )


def test_span_agreement_penalizes_shifted_and_missing_spans():
    teacher_spans = [
        [
            LabeledSpan("EMAIL", 0, 2, start=0, end=18),
            LabeledSpan("PHONE", 3, 4, start=24, end=36),
        ]
    ]
    shifted_student_spans = [[LabeledSpan("EMAIL", 0, 2, start=1, end=18)]]

    exact = compute_span_agreement_loss(
        teacher_spans=teacher_spans,
        student_spans=teacher_spans,
        entity_type_weights={"PHONE": 3.0},
    )
    shifted = compute_span_agreement_loss(
        teacher_spans=teacher_spans,
        student_spans=shifted_student_spans,
        entity_type_weights={"PHONE": 3.0},
    )

    assert exact.total == pytest.approx(0.0)
    assert shifted.matched == 1
    assert shifted.boundary > 0.0
    assert shifted.missed == pytest.approx(3.0)
    assert shifted.total > 0.0
    assert shifted.to_dict()["teacher_count"] == 2


def test_kd_loss_includes_weighted_span_agreement_when_requested():
    teacher_spans = [[LabeledSpan("EMAIL", 0, 2)]]
    student_spans = [[LabeledSpan("EMAIL", 0, 1)]]

    loss = compute_kd_loss(
        student_logits=[[[4.0, 0.0]]],
        teacher_logits=[[[4.0, 0.0]]],
        hard_labels=[[0]],
        teacher_spans=teacher_spans,
        student_spans=student_spans,
        temperature=1.0,
        alpha=1.0,
        span_loss_weight=0.0,
        span_agreement_weight=2.0,
    )

    assert loss.soft_kl == pytest.approx(0.0)
    assert loss.span_transfer == pytest.approx(0.0)
    assert loss.span_agreement > 0.0
    assert loss.total == pytest.approx(2.0 * loss.span_agreement)


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


def test_repaired_spans_filter_special_tokens_before_viterbi(monkeypatch):
    captured_lengths = []

    def fake_viterbi_decode(token_logprobs, *, label_info, biases):
        captured_lengths.append(len(token_logprobs))
        return [1, 2]

    monkeypatch.setattr(
        "openmed.training.distill.viterbi_decode",
        fake_viterbi_decode,
    )

    spans = decode_repaired_spans(
        [
            [
                [0.0, 99.0, 0.0],
                [0.0, 8.0, 0.0],
                [0.0, 0.0, 8.0],
                [0.0, 99.0, 0.0],
            ]
        ],
        {0: "O", 1: "B-NAME", 2: "E-NAME"},
        texts=["Jane"],
        offset_mapping=[[(0, 0), (0, 2), (2, 4), (0, 0)]],
    )

    assert captured_lengths == [2]
    assert [(span.token_start, span.token_end) for span in spans[0]] == [(1, 3)]
    assert spans[0][0].start == 0
    assert spans[0][0].end == 4


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


def test_distillation_report_writes_deterministic_json(tmp_path):
    report = build_distillation_report(
        teacher_id="teacher-local",
        student_backbone="openmed/backbones/tiny-direct-identifier-135m",
        temperature=2.0,
        alpha=0.6,
        teacher_recall_by_label={"EMAIL": 0.99},
        student_recall_by_label={"EMAIL": 0.98},
        critical_labels=("EMAIL",),
    )

    report_path = report.write_json(tmp_path / "distillation_report.json")

    assert report_path.read_text(encoding="utf-8") == report.to_json()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report.to_dict()


def test_mode_a_pipeline_builds_teacher_targets_and_student_loss():
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["alice@example.test ok"]
            assert kwargs["return_offsets_mapping"] is True
            return {
                "attention_mask": torch.tensor([[1, 1]]),
                "input_ids": torch.tensor([[101, 102]]),
                "offset_mapping": [[(0, len("alice@example.test")), (19, 21)]],
            }

    class FakeTeacher:
        config = SimpleNamespace(id2label={0: "O", 1: "S-EMAIL"})

        def parameters(self):
            yield torch.zeros(())

        def __call__(self, **model_inputs):
            assert set(model_inputs) == {"attention_mask", "input_ids"}
            return SimpleNamespace(
                logits=torch.tensor([[[0.0, 8.0], [8.0, 0.0]]]),
                span_logits=torch.tensor([[[0.0, 8.0]]]),
            )

    class FakeStudent:
        def __call__(self, **model_inputs):
            assert set(model_inputs) == {"attention_mask", "input_ids"}
            return SimpleNamespace(
                logits=torch.tensor([[[0.5, 7.5], [7.0, 0.5]]]),
                span_logits=torch.tensor([[[0.25, 7.75]]]),
            )

    pipeline = ModeADistillationPipeline(
        teacher_id="teacher-local",
        student_backbone="tiny-local",
        teacher_model=FakeTeacher(),
        student_model=FakeStudent(),
        tokenizer=FakeTokenizer(),
        temperature=2.0,
        alpha=0.5,
        span_loss_weight=0.25,
    )

    targets = pipeline.teacher_targets("alice@example.test ok")
    loss = pipeline.student_loss(
        model_inputs={
            "attention_mask": torch.tensor([[1, 1]]),
            "input_ids": torch.tensor([[101, 102]]),
        },
        hard_labels=torch.tensor([[1, 0]]),
        teacher_targets=targets,
    )

    assert targets.teacher_id == "teacher-local"
    assert targets.repaired_spans[0][0].label == "EMAIL"
    assert loss.total.item() >= 0.0
    assert loss.span_transfer.item() > 0.0


def _log_softmax(values):
    max_value = max(values)
    total = sum(math.exp(value - max_value) for value in values)
    log_total = max_value + math.log(total)
    return [value - log_total for value in values]
