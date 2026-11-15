"""Cross-source agreement, independence and disagreement negative controls."""

from itertools import permutations

import pytest

from openmed.training.languages.label_consensus import (
    ConsensusRule,
    LabelVote,
    build_label_consensus,
)


def vote(
    source="native_model", producer=1, *, start=0, end=4, label="CONCEPT", confidence=1
):
    return LabelVote(
        "0" * 64, start, end, label, source, f"{producer:064x}", confidence
    )


def test_independent_native_and_translated_sources_accept_exact_label():
    result = build_label_consensus([vote(), vote("translated_text", 2)])
    assert result[0].decision == "accepted"
    assert result[0].label == "CONCEPT"
    assert result[0].independent_source_count == 2


def test_duplicates_do_not_inflate_votes():
    assert (
        build_label_consensus([vote()] * 10)[0].reason
        == "insufficient_independent_sources"
    )


def test_one_producer_cannot_impersonate_multiple_categories():
    result = build_label_consensus(
        [vote(), vote("translated_text", 1), vote("rule", 1)]
    )
    assert result[0].independent_source_count == 1
    assert result[0].decision == "needs_adjudication"


def test_multiple_models_in_same_category_do_not_replace_missing_sources():
    result = build_label_consensus([vote(producer=n) for n in range(1, 8)])
    assert result[0].independent_source_count == 1


def test_majority_cannot_hide_disagreement():
    result = build_label_consensus(
        [vote(), vote("translated_text", 2), vote("rule", 3, label="OTHER")]
    )
    assert result[0].reason == "label_disagreement"
    assert result[0].label is None


def test_overlapping_span_projection_routes_all_affected_spans():
    result = build_label_consensus(
        [vote(), vote("translated_text", 2), vote("rule", 3, end=5)]
    )
    assert all(item.reason == "overlapping_projection" for item in result)


def test_adjacent_nonoverlapping_spans_are_independently_scored():
    rows = [
        vote(),
        vote("rule", 2),
        vote(start=4, end=8),
        vote("rule", 2, start=4, end=8),
    ]
    assert all(item.decision == "accepted" for item in build_label_consensus(rows))


def test_strong_source_cannot_override_low_confidence_evidence():
    result = build_label_consensus([vote(), vote("rule", 2, confidence=0.1)])
    assert result[0].reason == "low_confidence"


def test_custom_source_quorum_and_exact_threshold_boundary():
    rule = ConsensusRule(minimum_sources=3, minimum_confidence=0.5)
    rows = [vote(confidence=0.5), vote("rule", 2, confidence=0.5)]
    assert build_label_consensus(rows, rule=rule)[0].decision == "needs_adjudication"
    rows.append(vote("terminology", 3, confidence=0.5))
    assert build_label_consensus(rows, rule=rule)[0].decision == "accepted"


def test_input_order_does_not_change_decisions():
    rows = [vote(), vote("rule", 2), vote("terminology", 3)]
    expected = build_label_consensus(rows)
    assert all(build_label_consensus(order) == expected for order in permutations(rows))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -0.1, 1.1])
def test_invalid_confidence_fails(value):
    with pytest.raises(ValueError):
        vote(confidence=value)


def test_empty_input_has_no_accepted_labels():
    assert build_label_consensus([]) == ()


def test_public_decision_constructor_rejects_uncontrolled_export_metadata():
    from openmed.training.languages.label_consensus import ConsensusDecision

    with pytest.raises(ValueError, match="reference"):
        ConsensusDecision(
            "source text",
            0,
            1,
            None,
            "needs_adjudication",
            "low_confidence",
            0,
            ("CONCEPT",),
        )
    with pytest.raises(ValueError, match="accepted"):
        ConsensusDecision(
            "0" * 64,
            0,
            1,
            "CONCEPT",
            "accepted",
            "exact_source_consensus",
            1,
            ("CONCEPT",),
        )


def test_unhashable_disposition_and_source_raise_controlled_errors():
    from dataclasses import replace

    result = build_label_consensus([vote(), vote("translated_text", 2)])[0]
    for field in ("reason", "decision"):
        with pytest.raises(ValueError):
            replace(result, **{field: []})
    with pytest.raises(ValueError):
        replace(vote(), source=[])
    with pytest.raises(ValueError):
        replace(result, candidate_labels=(["CONCEPT"],))
