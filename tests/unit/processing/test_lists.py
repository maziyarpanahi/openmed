"""Synthetic offline tests for deterministic clinical list parsing."""

from __future__ import annotations

import pytest

from openmed.processing.lists import (
    ListItemSpan,
    list_boundary_f1,
    parse_lists,
    validate_list_items,
)


@pytest.mark.parametrize(
    ("text", "style", "markers"),
    (
        ("1. Hypertension\n2) Asthma\n3. Migraine", "numeric", ["1.", "2)", "3."]),
        ("- Penicillin\n* Latex\n• Adhesive", "bullet", ["-", "*", "•"]),
        (
            "a. Metformin\nb) Lisinopril\n(C) Atorvastatin",
            "lettered",
            ["a.", "b)", "(C)"],
        ),
    ),
)
def test_parse_lists_detects_explicit_enumeration_styles(
    text: str,
    style: str,
    markers: list[str],
) -> None:
    items = parse_lists(text)

    assert len(items) == 3
    assert [item.style for item in items] == [style, style, style]
    assert [item.marker for item in items] == markers
    assert [item.nesting_level for item in items] == [0, 0, 0]
    validate_list_items(text, items)
    assert all(text[item.start : item.end] == item.text for item in items)


def test_nested_items_attach_to_parent_and_parent_contains_continuations() -> None:
    text = (
        "1. Metformin 500 mg\n"
        "   a) Route: oral\n"
        "      Continue with meals\n"
        "   b) Frequency: twice daily\n"
        "2. Lisinopril 10 mg daily"
    )

    items = parse_lists(text)

    assert [item.marker for item in items] == ["1.", "a)", "b)", "2."]
    assert [item.nesting_level for item in items] == [0, 1, 1, 0]
    assert [item.parent_index for item in items] == [None, 0, 0, None]
    assert "Continue with meals" in items[1].text
    assert items[0].start <= items[1].start < items[1].end <= items[0].end
    assert items[0].end == items[3].start


def test_line_per_item_medications_keep_indented_sig_with_logical_item() -> None:
    text = (
        "Metformin 500 mg orally\n"
        "  twice daily with meals\n"
        "Lisinopril 10 mg daily\n"
        "Atorvastatin 20 mg nightly"
    )

    items = parse_lists(text)

    assert len(items) == 3
    assert all(item.style == "line" for item in items)
    assert "twice daily with meals" in items[0].text
    assert items[1].content_text.startswith("Lisinopril")
    assert items[2].content_text == "Atorvastatin 20 mg nightly"


def test_unindented_sig_directive_is_a_continuation_not_a_new_item() -> None:
    text = (
        "Metformin 500 mg orally\nTake one tablet twice daily\nLisinopril 10 mg daily"
    )

    items = parse_lists(text)

    assert len(items) == 2
    assert "Take one tablet" in items[0].text
    assert items[1].content_text == "Lisinopril 10 mg daily"


def test_single_unmarked_line_is_not_misclassified_as_a_list() -> None:
    assert parse_lists("Patient reports one stable chronic condition.") == []


def test_parse_lists_is_deterministic() -> None:
    text = "- Penicillin: rash\n- Latex: hives\n- Shellfish: nausea"

    assert parse_lists(text) == parse_lists(text)


def test_synthetic_list_heavy_boundary_f1_is_at_least_point_nine() -> None:
    cases = (
        ("1. Hypertension\n2. Asthma\n3. Migraine", [0, 16, 26]),
        ("- Penicillin\n- Latex\n- Adhesive", [0, 13, 21]),
        ("a) Metformin\nb) Lisinopril\nc) Atorvastatin", [0, 13, 27]),
        (
            "Metformin 500 mg daily\nLisinopril 10 mg daily\nAtorvastatin nightly",
            [0, 23, 47],
        ),
        ("Diabetes mellitus\nHypertension\nAsthma", [0, 18, 31]),
    )
    scores = []
    for text, starts in cases:
        gold = [
            ListItemSpan(
                text=text[
                    start : starts[index + 1] if index + 1 < len(starts) else len(text)
                ],
                start=start,
                end=starts[index + 1] if index + 1 < len(starts) else len(text),
                nesting_level=0,
                style="line",
                content_start=start,
            )
            for index, start in enumerate(starts)
        ]
        scores.append(list_boundary_f1(gold, parse_lists(text)))

    mean_f1 = sum(scores) / len(scores)
    assert mean_f1 >= 0.90, f"synthetic item-boundary F1 was {mean_f1:.3f}"
