"""Synthetic handwriting hypotheses and exact-candidate review/export controls."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.vision.handwriting.uncertainty import (
    HandwritingCandidate,
    HandwritingReview,
    HandwritingReviewError,
    HandwritingState,
    ReviewAction,
    TranscriptionAlternative,
    export_handwriting,
    review_handwriting,
)
from openmed.multimodal.vision.providers.base import Confidence


def alternative(text="0.5 mg", index=0, score=0.7):
    return TranscriptionAlternative(
        alternative_id=f"{index + 1:032x}",
        text=text,
        confidence=Confidence(score=score),
    )


def candidate(index=0, **changes):
    return replace(
        HandwritingCandidate(
            candidate_id=f"{index + 100:032x}",
            evidence=EvidenceNode(
                node_id=f"{index + 1000:032x}",
                kind=NodeKind.PIXELS,
                source_digest="a" * 64,
                geometry=PageGeometry(page=2, width=1000, height=800),
                region=(10, 20, 100, 60),
            ),
            alternatives=(alternative(), alternative("5 mg", 1, 0.95)),
            recognizer_digest="c" * 64,
        ),
        **changes,
    )


def review(item, **changes):
    return replace(
        HandwritingReview(
            review_id=f"{int(item.candidate_id, 16) + 5000:032x}",
            reviewer_id="d" * 32,
            decision_digest="b" * 64,
            candidate_id=item.candidate_id,
            candidate_digest=item.digest,
            action=ReviewAction.SELECT,
            alternative_id=item.alternatives[0].alternative_id
            if item.alternatives
            else "1" * 32,
            reviewer_confirmed=True,
        ),
        **changes,
    )


def test_high_confidence_does_not_verify_or_export_without_review():
    item = candidate(alternatives=(alternative(score=1),))
    assert item.state is HandwritingState.UNVERIFIED
    with pytest.raises(HandwritingReviewError):
        export_handwriting((item,), ())
    with pytest.raises(HandwritingReviewError, match="explicit confirmation"):
        export_handwriting((item,), (review(item, reviewer_confirmed=False),))
    assert item.state is HandwritingState.UNVERIFIED


def test_selection_is_explicit_and_does_not_choose_highest_score():
    item = candidate()
    decision = review(item)
    result = review_handwriting(item, decision)
    assert result.text == "0.5 mg"
    assert result.confidence.score == 0.7
    assert result.state is HandwritingState.REVIEWED_TRANSCRIPTION
    assert result.selected_alternative_id == item.alternatives[0].alternative_id
    assert item.state is HandwritingState.UNVERIFIED
    payload = json.loads(export_handwriting((item,), (decision,)).data_json)
    assert payload["items"][0]["value"] == "0.5 mg"
    assert payload["items"][0]["source_kind"] == "handwriting"


def test_reviewer_can_choose_any_existing_alternative():
    item = candidate()
    chosen = item.alternatives[1]
    result = review_handwriting(
        item, review(item, alternative_id=chosen.alternative_id)
    )
    assert result.text == chosen.text
    assert result.confidence == chosen.confidence


def test_correction_preserves_transcription_without_numeric_or_date_normalization():
    item = candidate()
    corrected = " 0,5 mg · 03/04/26 "
    decision = review(
        item, action=ReviewAction.CORRECT, alternative_id=None, corrected_text=corrected
    )
    result = review_handwriting(item, decision)
    assert result.text == corrected
    assert result.confidence.score is None  # Human review is not a calibrated score.
    assert result.selected_alternative_id is None
    assert (
        json.loads(export_handwriting((item,), (decision,)).data_json)["items"][0][
            "value"
        ]
        == corrected
    )
    assert item.alternatives[0].text == "0.5 mg"


@pytest.mark.parametrize(
    ("action", "state"),
    [
        (ReviewAction.REJECT, HandwritingState.REJECTED),
        (ReviewAction.ABSTAIN, HandwritingState.ABSTAINED),
    ],
)
def test_rejection_and_abstention_are_retained_but_block_export(action, state):
    item = candidate()
    decision = review(item, action=action, alternative_id=None)
    result = review_handwriting(item, decision)
    assert result.state is state
    assert result.text is None
    assert result.confidence.score is None
    with pytest.raises(HandwritingReviewError, match="every candidate to be accepted"):
        export_handwriting((item,), (decision,))


def test_original_visual_coordinates_and_decision_lineage_survive_export():
    item = candidate()
    decision = review(item)
    exported = export_handwriting((item,), (decision,))
    record = json.loads(exported.data_json)["items"][0]
    assert record["evidence"]["source_digest"] == "a" * 64
    assert record["evidence"]["geometry"] == {
        "page": 2,
        "width": 1000.0,
        "height": 800.0,
        "unit": "pixel",
    }
    assert record["evidence"]["region"] == [10, 20, 100, 60]
    assert record["reviewer_id"] == decision.reviewer_id
    assert record["review_digest"] == decision.digest
    assert record["candidate_digest"] == item.digest
    assert record["decision_digest"] == decision.decision_digest
    assert exported.items[0].evidence is item.evidence


@pytest.mark.parametrize(
    "change",
    [
        "text",
        "score",
        "order",
        "extra_alternative",
        "source",
        "bounds",
        "geometry",
        "evidence_id",
        "candidate_id",
        "recognizer",
    ],
)
def test_any_candidate_change_invalidates_prior_review(change):
    original = candidate()
    decision = review(original)
    item = original
    if change == "text":
        item = replace(
            item,
            alternatives=(
                replace(item.alternatives[0], text="0.05 mg"),
                item.alternatives[1],
            ),
        )
    elif change == "score":
        item = replace(
            item,
            alternatives=(
                replace(item.alternatives[0], confidence=Confidence(score=0.8)),
                item.alternatives[1],
            ),
        )
    elif change == "order":
        item = replace(item, alternatives=tuple(reversed(item.alternatives)))
    elif change == "extra_alternative":
        item = replace(
            item, alternatives=item.alternatives + (alternative("50 mg", 2),)
        )
    elif change == "source":
        item = replace(item, evidence=replace(item.evidence, source_digest="e" * 64))
    elif change == "bounds":
        item = replace(item, evidence=replace(item.evidence, region=(11, 20, 100, 60)))
    elif change == "geometry":
        item = replace(
            item,
            evidence=replace(
                item.evidence, geometry=PageGeometry(page=3, width=1000, height=800)
            ),
        )
    elif change == "evidence_id":
        item = replace(item, evidence=replace(item.evidence, node_id="e" * 32))
    elif change == "candidate_id":
        item = replace(item, candidate_id="e" * 32)
    else:
        item = replace(item, recognizer_digest="e" * 64)
    with pytest.raises(HandwritingReviewError, match="does not match"):
        review_handwriting(item, decision)


def test_empty_hypotheses_remain_unknown_until_reviewer_provides_a_transcription():
    item = candidate(alternatives=(), recognizer_digest=None)
    assert item.state is HandwritingState.UNVERIFIED
    assert json.loads(item.to_json())["alternatives"] == []
    with pytest.raises(HandwritingReviewError, match="not available"):
        review_handwriting(item, review(item))
    decision = review(
        item,
        action=ReviewAction.CORRECT,
        alternative_id=None,
        corrected_text="Synthetic reviewed transcription",
    )
    assert (
        export_handwriting((item,), (decision,)).items[0].text
        == decision.corrected_text
    )


def test_unknown_confidence_is_preserved_after_review():
    item = candidate(alternatives=(alternative(score=None),))
    result = review_handwriting(item, review(item))
    assert result.confidence.score is None
    assert json.loads(result.to_json())["confidence"] is None


@pytest.mark.parametrize(
    "defect", ["unconfirmed", "stale", "rejected", "unknown_choice"]
)
def test_mixed_batch_never_returns_a_partial_export(defect):
    first, second = candidate(), candidate(1)
    decisions = (review(first), review(second))
    if defect == "unconfirmed":
        decisions = (decisions[0], replace(decisions[1], reviewer_confirmed=False))
    elif defect == "stale":
        decisions = (decisions[0], replace(decisions[1], candidate_digest="f" * 64))
    elif defect == "rejected":
        decisions = (
            decisions[0],
            replace(decisions[1], action=ReviewAction.REJECT, alternative_id=None),
        )
    else:
        decisions = (decisions[0], replace(decisions[1], alternative_id="f" * 32))
    with pytest.raises(HandwritingReviewError):
        export_handwriting((first, second), decisions)


def test_export_preserves_candidate_order_and_is_independent_of_review_order():
    candidates = (candidate(), candidate(1))
    decisions = tuple(review(item) for item in candidates)
    first = export_handwriting(candidates, decisions)
    second = export_handwriting(candidates, tuple(reversed(decisions)))
    assert first == second
    assert tuple(item.candidate_id for item in first.items) == tuple(
        item.candidate_id for item in candidates
    )
    assert first.to_report_json() == second.to_report_json()


def test_duplicate_or_unrelated_reviews_and_candidate_ids_are_rejected():
    first, second = candidate(), candidate(1)
    with pytest.raises(HandwritingReviewError):
        export_handwriting((first, first), (review(first), review(first)))
    with pytest.raises(HandwritingReviewError):
        export_handwriting((first, second), (review(first), review(first)))
    with pytest.raises(HandwritingReviewError):
        export_handwriting(
            (first, second),
            (review(first), review(second, review_id=review(first).review_id)),
        )
    with pytest.raises(HandwritingReviewError):
        export_handwriting((first,), (review(second),))


def test_text_is_only_in_protected_export_not_representations_reports_or_errors():
    raw = "synthetic-private-hypothesis"
    corrected = "synthetic-private-correction"
    item = candidate(alternatives=(alternative(raw),))
    decision = review(
        item, action=ReviewAction.CORRECT, alternative_id=None, corrected_text=corrected
    )
    reviewed = review_handwriting(item, decision)
    exported = export_handwriting((item,), (decision,))
    diagnostics = (
        repr(item)
        + repr(item.alternatives[0])
        + repr(decision)
        + repr(reviewed)
        + repr(exported)
    )
    diagnostics += item.to_json() + reviewed.to_json() + exported.to_report_json()
    assert raw not in diagnostics and corrected not in diagnostics
    assert corrected in exported.data_json and raw not in exported.data_json
    with pytest.raises(HandwritingReviewError) as raised:
        review_handwriting(item, replace(decision, reviewer_confirmed=False))
    assert raw not in str(raised.value) and corrected not in str(raised.value)
    assert raised.value.__context__ is None


def test_json_values_cannot_create_extra_export_fields():
    value = 'synthetic "}, "state": "verified", "value": "replacement\\n'
    item = candidate(alternatives=(alternative(value),))
    payload = json.loads(export_handwriting((item,), (review(item),)).data_json)
    assert len(payload["items"]) == 1
    assert payload["items"][0]["value"] == value
    assert payload["items"][0]["state"] == "reviewed_transcription"


def test_candidate_and_alternative_are_immutable():
    item = candidate()
    with pytest.raises(FrozenInstanceError):
        item.recognizer_digest = "f" * 64
    with pytest.raises(FrozenInstanceError):
        item.alternatives[0].text = "changed"


@pytest.mark.parametrize("text", ["", "   ", "x" * 4097, "bad\ud800value"])
def test_malformed_text_has_value_free_errors(text):
    with pytest.raises(HandwritingReviewError) as raised:
        alternative(text)
    assert str(raised.value) == "invalid handwriting review declaration"
    assert raised.value.__context__ is None


def test_ambiguous_actions_and_invalid_evidence_fail_closed():
    item = candidate()
    with pytest.raises(HandwritingReviewError):
        review(item, corrected_text="unexpected correction")
    with pytest.raises(HandwritingReviewError):
        review(item, action=ReviewAction.CORRECT, alternative_id=None)
    with pytest.raises(HandwritingReviewError):
        review(item, reviewer_confirmed=1)
    with pytest.raises(HandwritingReviewError):
        replace(item, candidate_id="patient-name")
    with pytest.raises(HandwritingReviewError):
        replace(item, alternatives=(item.alternatives[0], item.alternatives[0]))
    with pytest.raises(HandwritingReviewError):
        replace(item, evidence=EvidenceNode(node_id="e" * 32, kind=NodeKind.FACT))


def test_batch_and_hypothesis_limits_are_enforced_before_export():
    with pytest.raises(HandwritingReviewError):
        candidate(alternatives=tuple(alternative("synthetic", i) for i in range(17)))
    items = tuple(candidate(i) for i in range(129))
    with pytest.raises(HandwritingReviewError):
        export_handwriting(items, tuple(review(item) for item in items))
    large = tuple(
        candidate(i, alternatives=tuple(alternative("x" * 4096, j) for j in range(16)))
        for i in range(5)
    )
    with pytest.raises(HandwritingReviewError):
        export_handwriting(large, tuple(review(item) for item in large))
