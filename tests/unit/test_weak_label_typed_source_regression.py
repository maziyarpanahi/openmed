from openmed.training.weak_labeling import (
    WeakLabelSpan,
    normalize_weak_span,
    weak_label_document,
)


def span():
    return WeakLabelSpan(0, 3, "PERSON", "Ada", 0.9)


def test_typed_teacher_spans_have_same_consensus_as_mapping_spans():
    item = span()
    typed = weak_label_document("Ada", {"teacher-a": [item], "teacher-b": [item]})
    mapping = weak_label_document(
        "Ada", {"teacher-a": [item.to_dict()], "teacher-b": [item.to_dict()]}
    )
    assert typed == mapping
    assert len(typed.accepted_spans) == 1
    assert typed.rejected_spans == ()


def test_source_default_is_filled_without_mutating_the_original():
    original = span()
    normalized = normalize_weak_span(original, source="teacher-a")
    assert normalized.source == "teacher-a"
    assert original.source == ""
    assert normalized is not original


def test_explicit_source_and_identity_are_preserved():
    original = WeakLabelSpan(0, 3, "PERSON", "Ada", 0.9, source="explicit")
    assert normalize_weak_span(original, source="outer") is original
    assert normalize_weak_span(span()).source == ""


def test_two_predictions_from_one_model_are_not_consensus():
    result = weak_label_document("Ada", {"teacher-a": [span(), span()]})
    assert result.accepted_spans == ()
    assert len(result.rejected_spans) == 1


def test_consensus_still_obeys_validator_rejection():
    result = weak_label_document(
        "Ada", {"a": [span()], "b": [span()]}, validators=[lambda _: False]
    )
    assert result.accepted_spans == ()
    assert result.rejected_spans[0].metadata["rejection_reason"] == "validator_rejected"
