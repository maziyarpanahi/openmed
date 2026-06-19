from __future__ import annotations

from openmed.training.adjudication import AdjudicationQueue
from openmed.training.weak_labeling import WeakLabelSpan, weak_label_document


def test_accepts_spans_on_inter_model_agreement():
    text = "Patient Jordan Smith was discharged."
    outputs = {
        "teacher_a": [{"start": 8, "end": 20, "label": "PERSON", "score": 0.91}],
        "teacher_b": [{"start": 8, "end": 20, "label": "PERSON", "score": 0.87}],
    }

    decision = weak_label_document(text, outputs)

    assert len(decision.accepted_spans) == 1
    accepted = decision.accepted_spans[0]
    assert accepted.label == "PERSON"
    assert accepted.metadata["accepted_by"] == "inter_model_agreement"
    assert decision.adjudication_items == ()


def test_never_auto_accepts_single_model_span():
    text = "Patient Jordan Smith was discharged."
    outputs = {
        "teacher_a": [{"start": 8, "end": 20, "label": "PERSON", "score": 0.91}],
    }

    decision = weak_label_document(text, outputs)

    assert decision.accepted_spans == ()
    assert len(decision.rejected_spans) == 1
    assert decision.rejected_spans[0].metadata["rejection_reason"] == "single_model"


def test_human_review_can_accept_single_model_span():
    text = "Patient Jordan Smith was discharged."
    span = {"start": 8, "end": 20, "label": "PERSON", "score": 0.91}

    decision = weak_label_document(text, {"teacher_a": [span]}, human_reviewed_spans=[span])

    assert len(decision.accepted_spans) == 1
    assert decision.accepted_spans[0].metadata["accepted_by"] == "human_review"


def test_routes_overlapping_disagreements_to_adjudication_and_hook():
    text = "Patient Jordan Smith was discharged."
    queued = []
    outputs = {
        "teacher_a": [{"start": 8, "end": 20, "label": "PERSON", "score": 0.91}],
        "teacher_b": [{"start": 16, "end": 20, "label": "LAST_NAME", "score": 0.88}],
    }

    decision = weak_label_document(
        text,
        outputs,
        record_id="note-1",
        active_learning_hook=queued.append,
    )

    assert decision.accepted_spans == ()
    assert len(decision.adjudication_items) == 1
    item = decision.adjudication_items[0]
    assert item.record_id == "note-1"
    assert item.reason == "inter_model_disagreement"
    assert item.metadata["hard_negative_seed"] is True
    assert item.to_hard_negative_seed()["source"] == "weak_labeling_adjudication"
    assert queued == [item]


def test_validators_can_reject_agreed_span():
    text = "Patient Jordan Smith was discharged."
    outputs = {
        "teacher_a": [{"start": 8, "end": 20, "label": "PERSON", "score": 0.91}],
        "teacher_b": [{"start": 8, "end": 20, "label": "PERSON", "score": 0.87}],
    }

    decision = weak_label_document(text, outputs, validators=[lambda span: False])

    assert decision.accepted_spans == ()
    assert len(decision.rejected_spans) == 1
    assert decision.rejected_spans[0].metadata["rejection_reason"] == "validator_rejected"


def test_adjudication_queue_drains_items():
    queue = AdjudicationQueue()
    decision = weak_label_document(
        "Call 555-111-2222 tomorrow.",
        {
            "teacher_a": [{"start": 5, "end": 17, "label": "PHONE", "score": 0.9}],
            "teacher_b": [{"start": 5, "end": 14, "label": "PHONE", "score": 0.8}],
        },
    )
    queue.extend(decision.adjudication_items)

    drained = queue.drain()

    assert len(drained) == 1
    assert queue.items == []


def test_weak_label_span_to_dict_is_json_ready():
    span = WeakLabelSpan(start=0, end=4, label="PERSON", text="John", score=0.5)

    assert span.to_dict()["label"] == "PERSON"
