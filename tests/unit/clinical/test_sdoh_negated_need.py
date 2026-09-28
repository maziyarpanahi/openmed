"""Synthetic offline tests for determinant-scoped SDOH need assertions."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.sdoh import SDOHFinding
from openmed.clinical.sdoh_negated_need import (
    AFFIRMED,
    NEED_ABSENT,
    NEED_PRESENT,
    NEGATED,
    SDOH_NEGATED_NEED_SCHEMA_VERSION,
    UNKNOWN,
    SDOHNegatedNeedEvidence,
    resolve_sdoh_negated_need,
    resolve_sdoh_negated_needs,
)


def _candidate(text: str, value: str, category: str) -> dict[str, object]:
    start = text.index(value)
    return {
        "start": start,
        "end": start + len(value),
        "category": category,
    }


def test_determinant_negation_stays_with_its_finding() -> None:
    text = "No food insecurity, but reports a transportation barrier."
    findings = [
        _candidate(text, "food insecurity", "food_insecurity"),
        _candidate(text, "transportation barrier", "transportation_barrier"),
    ]

    results = resolve_sdoh_negated_needs(text, findings)

    assert [result.assertion for result in results] == [NEGATED, AFFIRMED]
    assert [result.need_status for result in results] == [
        NEED_ABSENT,
        NEED_PRESENT,
    ]
    assert results[0].determinant == "food_insecurity"
    assert results[1].determinant == "transportation_barrier"
    assert results[0].cue_offsets == ((0, 2),)
    assert results[0].review_required is False
    assert results[1].review_required is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Reports food insecurity, denies transportation barrier.",
            [AFFIRMED, NEGATED],
        ),
        (
            "Food insecurity is denied and transportation barrier is present.",
            [NEGATED, AFFIRMED],
        ),
        (
            "No food insecurity and transportation barrier is present.",
            [NEGATED, AFFIRMED],
        ),
        (
            "Reports food insecurity and denies transportation barrier.",
            [AFFIRMED, NEGATED],
        ),
    ],
)
def test_independent_determinant_assertions_do_not_leak(
    text: str,
    expected: list[str],
) -> None:
    lower = text.casefold()
    food_start = lower.index("food insecurity")
    transport_start = lower.index("transportation barrier")
    findings = [
        {
            "start": food_start,
            "end": food_start + len("food insecurity"),
            "category": "food_insecurity",
        },
        {
            "start": transport_start,
            "end": transport_start + len("transportation barrier"),
            "category": "transportation_barrier",
        },
    ]

    results = resolve_sdoh_negated_needs(text, findings)

    assert [result.assertion for result in results] == expected
    assert all(not result.review_required for result in results)


def test_one_denial_can_cover_a_coordinated_need_list() -> None:
    text = "Denies food insecurity and transportation barrier."
    findings = [
        _candidate(text, "food insecurity", "food_insecurity"),
        _candidate(text, "transportation barrier", "transportation_barrier"),
    ]

    results = resolve_sdoh_negated_needs(text, findings)

    assert [result.assertion for result in results] == [NEGATED, NEGATED]
    assert [result.negation_cue_count for result in results] == [1, 1]


def test_backward_negation_is_attached_to_the_matching_determinant() -> None:
    text = "Housing instability is not present."
    result = resolve_sdoh_negated_need(
        text,
        _candidate(text, "Housing instability", "housing_instability"),
    )

    assert result.assertion == NEGATED
    assert result.need_status == NEED_ABSENT
    cue_start = text.index("not present")
    assert result.cue_offsets == ((cue_start, cue_start + len("not present")),)


def test_double_negation_abstains_instead_of_using_parity() -> None:
    text = "Not no food insecurity."
    result = resolve_sdoh_negated_need(
        text,
        _candidate(text, "food insecurity", "food_insecurity"),
    )

    assert result.assertion == UNKNOWN
    assert result.need_status == "unknown"
    assert result.double_negation is True
    assert result.negation_cue_count == 2
    assert result.review_required is True
    assert result.review_reasons == ("double_negation",)
    assert result.conflicting_assertions == (AFFIRMED, NEGATED)


def test_contradictory_local_cues_are_sent_to_review() -> None:
    text = "Food insecurity is denied and present."
    result = resolve_sdoh_negated_need(
        text,
        _candidate(text, "Food insecurity", "food_insecurity"),
    )

    assert result.assertion == UNKNOWN
    assert result.review_required is True
    assert result.review_reasons == ("contradictory_cues",)
    assert result.conflicting_assertions == (AFFIRMED, NEGATED)
    assert len(result.cue_offsets) == 2


def test_contrastive_contradictory_cues_are_sent_to_review() -> None:
    text = "Food insecurity is denied but present."
    result = resolve_sdoh_negated_need(
        text,
        _candidate(text, "Food insecurity", "food_insecurity"),
    )

    assert result.assertion == UNKNOWN
    assert result.review_required is True
    assert result.review_reasons == ("contradictory_cues",)
    assert result.conflicting_assertions == (AFFIRMED, NEGATED)


def test_uncertain_pseudo_negation_is_not_treated_as_a_denial() -> None:
    text = "Food insecurity cannot be excluded."
    result = resolve_sdoh_negated_need(
        text,
        _candidate(text, "Food insecurity", "food_insecurity"),
    )

    assert result.assertion == UNKNOWN
    assert result.negation_cue_count == 0
    assert result.review_required is True
    assert result.review_reasons == ("uncertain_cue",)


def test_leading_pseudo_negation_is_reviewable() -> None:
    text = "Not ruled out food insecurity."
    result = resolve_sdoh_negated_need(
        text,
        _candidate(text, "food insecurity", "food_insecurity"),
    )

    assert result.assertion == UNKNOWN
    assert result.review_required is True
    assert result.review_reasons == ("uncertain_cue",)
    assert result.negation_cue_count == 0


def test_existing_sdoh_findings_and_input_order_are_supported() -> None:
    text = "Reports a transportation barrier and food insecurity."
    later = SDOHFinding(
        category="food_insecurity",
        value="food insecurity",
        status="current",
        extent=None,
        temporality="recent",
        span=(
            text.index("food insecurity"),
            text.index("food insecurity") + len("food insecurity"),
        ),
        score=1.0,
    )
    earlier = _candidate(text, "transportation barrier", "transportation")

    results = resolve_sdoh_negated_needs(text, [later, earlier])

    assert [result.input_index for result in results] == [1, 0]
    assert [result.assertion for result in results] == [AFFIRMED, AFFIRMED]


def test_reports_are_value_free_and_round_trip() -> None:
    text = "No food insecurity."
    result = resolve_sdoh_negated_need(
        text,
        _candidate(text, "food insecurity", "food_insecurity"),
    )

    payload = result.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == SDOH_NEGATED_NEED_SCHEMA_VERSION
    assert SDOHNegatedNeedEvidence.from_dict(payload) == result
    assert "food insecurity" not in serialized
    assert "text" not in payload
    assert "value" not in payload
    assert result.to_json() == result.to_json()


def test_default_assertion_preserves_an_upstream_finding() -> None:
    text = "Food insecurity."
    result = resolve_sdoh_negated_need(
        text,
        _candidate(text, "Food insecurity", "food_insecurity"),
    )

    assert result.assertion == AFFIRMED
    assert result.need_status == NEED_PRESENT
    assert result.source == "default"
    assert result.review_required is False


def test_malformed_offsets_do_not_echo_candidate_text() -> None:
    marker = "synthetic-private-marker"

    with pytest.raises(ValueError, match="within the source text") as exc_info:
        resolve_sdoh_negated_needs(
            "No food insecurity.",
            [{"start": 0, "end": 999, "text": marker}],
        )

    assert marker not in str(exc_info.value)
