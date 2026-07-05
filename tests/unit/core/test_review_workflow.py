"""Tests for the human-in-the-loop review workflow API.

All fixtures are synthetic. No real PHI is used, and the PHI-safety tests assert
that no synthetic identifier substring leaks into serialized artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.core.review_workflow import (
    REVIEW_REASON_CRITICAL_LABEL,
    REVIEW_REASON_LOW_CONFIDENCE,
    ReviewFeedback,
    ReviewItem,
    ReviewQueue,
    append_feedback,
    build_review_queue,
    critical_labels,
    record_review_decision,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult

# Synthetic document mixing high-confidence critical IDs, a low-confidence name,
# and confident non-critical clinical concepts that should NOT be queued.
SYNTHETIC_TEXT = (
    "Patient Jane Roe, SSN 123-45-6789, was prescribed aspirin for hypertension."
)
#            offsets: "Jane Roe" -> 8:16, "123-45-6789" -> 22:33,
#                     "aspirin" -> 50:57, "hypertension" -> 62:74


def _entity(
    text: str,
    label: str,
    start: int,
    end: int,
    confidence: float,
    action: str | None = None,
) -> PIIEntity:
    return PIIEntity(
        text=text,
        label=label,
        start=start,
        end=end,
        confidence=confidence,
        entity_type=label,
        original_text=text,
        action=action,
    )


def _deid_result() -> DeidentificationResult:
    return DeidentificationResult(
        original_text=SYNTHETIC_TEXT,
        deidentified_text=(
            "Patient [NAME], SSN [SSN], was prescribed aspirin for hypertension."
        ),
        pii_entities=[
            _entity("Jane Roe", "NAME", 8, 16, 0.40, action="mask"),
            _entity("123-45-6789", "SSN", 22, 33, 0.99, action="mask"),
            _entity("aspirin", "MEDICATION", 50, 57, 0.97),
            _entity("hypertension", "CONDITION", 62, 74, 0.95),
        ],
        method="mask",
        timestamp=datetime(2026, 7, 5, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Queue selection
# ---------------------------------------------------------------------------


def test_queue_selects_low_confidence_and_critical_only():
    queue = build_review_queue(_deid_result(), confidence_threshold=0.7)

    assert isinstance(queue, ReviewQueue)
    assert queue.total_spans == 4
    # NAME (low-conf + critical) and SSN (critical). aspirin/hypertension drop.
    selected = [(item.canonical_label, sorted(item.reasons)) for item in queue.items]
    assert selected == [
        ("PERSON", [REVIEW_REASON_CRITICAL_LABEL, REVIEW_REASON_LOW_CONFIDENCE]),
        ("SSN", [REVIEW_REASON_CRITICAL_LABEL]),
    ]


def test_confident_non_critical_entities_are_not_queued():
    queue = build_review_queue(_deid_result(), confidence_threshold=0.7)
    queued_labels = {item.canonical_label for item in queue.items}
    assert "MEDICATION" not in queued_labels
    assert "CONDITION" not in queued_labels


def test_low_confidence_reason_uses_strict_threshold():
    # A confidence exactly equal to the threshold is NOT low-confidence.
    result = PredictionResult(
        text="temperature reading",
        entities=[
            EntityPrediction(
                text="reading", label="lab_test", start=12, end=19, confidence=0.70
            )
        ],
        model_name="fixture",
        timestamp="2026-07-05T00:00:00",
    )
    queue = build_review_queue(result, confidence_threshold=0.70)
    assert queue.review_count == 0


def test_queue_ordered_by_offsets():
    # Feed entities out of order; queue must sort by (start, end, label).
    result = PredictionResult(
        text=SYNTHETIC_TEXT,
        entities=[
            EntityPrediction(
                text="123-45-6789", label="ssn", start=22, end=33, confidence=0.99
            ),
            EntityPrediction(
                text="Jane Roe", label="name", start=8, end=16, confidence=0.40
            ),
        ],
        model_name="fixture",
        timestamp="2026-07-05T00:00:00",
    )
    queue = build_review_queue(result, confidence_threshold=0.7)
    assert [item.start for item in queue.items] == [8, 22]


def test_custom_critical_labels_override_default():
    queue = build_review_queue(
        _deid_result(),
        confidence_threshold=0.0,  # disable low-confidence flagging
        critical_labels=["condition"],
    )
    # Only the CONDITION span qualifies; SSN/NAME no longer critical here.
    assert [item.canonical_label for item in queue.items] == ["CONDITION"]
    assert queue.items[0].reasons == (REVIEW_REASON_CRITICAL_LABEL,)


def test_default_critical_labels_include_direct_identifiers():
    critical = critical_labels()
    assert "SSN" in critical
    assert "PERSON" in critical
    assert "CREDIT_CARD" in critical
    # Clinical concepts are not critical.
    assert "CONDITION" not in critical
    assert "MEDICATION" not in critical


def test_include_all_keeps_confident_non_critical_spans():
    queue = build_review_queue(
        _deid_result(), confidence_threshold=0.7, include_all=True
    )
    assert queue.review_count == 4
    medication = next(i for i in queue.items if i.canonical_label == "MEDICATION")
    assert medication.reasons == ()


def test_accepts_to_dict_payload_input():
    payload = _deid_result().to_dict()
    queue = build_review_queue(payload, confidence_threshold=0.7)
    assert {item.canonical_label for item in queue.items} == {"PERSON", "SSN"}


def test_invalid_offsets_raise():
    result = PredictionResult(
        text="abc",
        entities=[
            EntityPrediction(text="x", label="ssn", start=5, end=9, confidence=0.9)
        ],
        model_name="fixture",
        timestamp="2026-07-05T00:00:00",
    )
    with pytest.raises(ValueError):
        build_review_queue(result, confidence_threshold=0.7)


# ---------------------------------------------------------------------------
# Decision capture / round-trip
# ---------------------------------------------------------------------------


def _first_item() -> ReviewItem:
    return build_review_queue(_deid_result(), confidence_threshold=0.7).items[0]


def test_accept_decision_round_trips():
    item = _first_item()
    feedback = record_review_decision(item, "accept", reviewer_id="reviewer-1")
    assert feedback.decision == "accept"

    restored = ReviewFeedback.from_dict(json.loads(feedback.to_jsonl_line()))
    assert restored == feedback
    assert restored.reviewer_id == "reviewer-1"
    assert restored.corrected_label is None


def test_reject_decision_round_trips():
    item = _first_item()
    feedback = record_review_decision(item, "reject")
    restored = ReviewFeedback.from_dict(json.loads(feedback.to_jsonl_line()))
    assert restored == feedback
    assert restored.decision == "reject"


def test_correct_decision_captures_label_and_offsets():
    item = _first_item()  # PERSON span at 8:16
    feedback = record_review_decision(
        item,
        "correct",
        corrected_label="first_name",
        corrected_start=8,
        corrected_end=12,
    )
    assert feedback.decision == "correct"
    assert feedback.corrected_label == "first_name"
    assert feedback.corrected_canonical_label == "FIRST_NAME"
    assert (feedback.corrected_start, feedback.corrected_end) == (8, 12)

    restored = ReviewFeedback.from_dict(json.loads(feedback.to_jsonl_line()))
    assert restored == feedback


def test_correct_requires_label_or_offsets():
    with pytest.raises(ValueError):
        record_review_decision(_first_item(), "correct")


def test_correction_fields_rejected_for_non_correct_decision():
    with pytest.raises(ValueError):
        record_review_decision(_first_item(), "accept", corrected_label="ssn")


def test_invalid_decision_raises():
    with pytest.raises(ValueError):
        record_review_decision(_first_item(), "maybe")


def test_partial_corrected_offsets_raise():
    with pytest.raises(ValueError):
        record_review_decision(_first_item(), "correct", corrected_start=8)


def test_append_feedback_writes_jsonl(tmp_path):
    path = tmp_path / "nested" / "feedback.jsonl"
    item = _first_item()
    record_a = record_review_decision(item, "accept")
    record_b = record_review_decision(item, "reject")

    append_feedback(path, record_a)
    written = append_feedback(path, record_b)

    lines = written.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    decoded = [json.loads(line) for line in lines]
    assert [row["decision"] for row in decoded] == ["accept", "reject"]
    # Each line is valid standalone JSON (true JSONL).
    for line in lines:
        json.loads(line)


# ---------------------------------------------------------------------------
# PHI-safety
# ---------------------------------------------------------------------------

_SYNTHETIC_IDENTIFIERS = ("Jane", "Roe", "Jane Roe", "123-45-6789")


def _assert_no_phi(blob: str) -> None:
    for identifier in _SYNTHETIC_IDENTIFIERS:
        assert identifier not in blob, f"leaked synthetic PHI: {identifier!r}"


def test_review_item_serialization_is_phi_safe():
    queue = build_review_queue(_deid_result(), confidence_threshold=0.7)
    blob = json.dumps(queue.to_dict())
    _assert_no_phi(blob)
    # The span text is masked in context, and only a hash represents it.
    for item in queue.items:
        assert item.context["span"] == "[REDACTED]"
        assert item.text_hash.startswith("sha256:")


def test_feedback_record_is_phi_safe_even_with_note():
    item = _first_item()
    # A reviewer note that itself contains synthetic PHI must not be stored.
    feedback = record_review_decision(
        item,
        "correct",
        corrected_label="first_name",
        note="This is really Jane Roe, not an SSN 123-45-6789.",
        reviewer_id="clinician-7",
    )
    blob = feedback.to_jsonl_line()
    _assert_no_phi(blob)
    assert feedback.note_hash is not None
    assert feedback.note_hash.startswith("sha256:")
    assert "clinician-7" in blob  # non-PHI reviewer id is retained


def test_feedback_jsonl_file_is_phi_safe(tmp_path):
    path = tmp_path / "feedback.jsonl"
    for item in build_review_queue(_deid_result(), confidence_threshold=0.7).items:
        append_feedback(path, record_review_decision(item, "accept"))
    _assert_no_phi(path.read_text(encoding="utf-8"))


def test_text_hash_matches_audit_hash_of_span():
    from openmed.core.audit import hash_text

    item = _first_item()  # PERSON span "Jane Roe" at 8:16
    assert item.text_hash == hash_text("Jane Roe")
