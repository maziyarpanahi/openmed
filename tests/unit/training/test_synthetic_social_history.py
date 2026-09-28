"""Synthetic, offline gold-event coverage for social-history generation."""

from __future__ import annotations

import json

import pytest

from openmed.training.synthetic.social_history import (
    SOCIAL_HISTORY_CATEGORIES,
    generate_social_history_examples,
)


def test_all_five_categories_have_aligned_shac_style_events() -> None:
    examples = generate_social_history_examples(3, seed=17)

    assert len(examples) == 3
    for example in examples:
        assert example.text.startswith("Social History:\n")
        assert example.synthetic is True
        assert {event.category for event in example.events} == set(
            SOCIAL_HISTORY_CATEGORIES
        )
        for event in example.events:
            start, end = event.span
            assert 0 <= start < end <= len(example.text)
            assert example.text[start:end]
            finding = event.to_finding()
            assert finding.category == event.category
            assert finding.status == event.status
            assert finding.extent == event.amount
            assert finding.span == event.span
            assert finding.score == 1.0

        record = example.to_dict()
        assert record["metadata"] == {
            "synthetic": True,
            "source": "openmed.synthetic.social_history",
            "restricted_data": False,
        }
        assert len(record["events"]) == 5
        json.dumps(record)


def test_negated_and_never_use_is_negative_gold_not_positive() -> None:
    examples = generate_social_history_examples(30, seed=4)

    for example in examples:
        for event in example.events:
            clause = example.text[slice(*event.span)].lower()
            if event.category in {"alcohol", "drug", "tobacco"} and (
                "denies" in clause or "never" in clause
            ):
                assert event.status == "none"
                assert event.amount is None

    for category in SOCIAL_HISTORY_CATEGORIES:
        assert (
            len(
                {
                    event.status
                    for example in examples
                    for event in example.events
                    if event.category == category
                }
            )
            >= 3
        )


def test_fixed_seed_reproduces_text_annotations_and_provenance() -> None:
    first = generate_social_history_examples(12, seed=91)
    second = generate_social_history_examples(12, seed=91)
    third = generate_social_history_examples(12, seed=92)

    assert first == second
    assert first != third
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_count_fails_closed(count: object) -> None:
    with pytest.raises(ValueError, match="count"):
        generate_social_history_examples(count)  # type: ignore[arg-type]
