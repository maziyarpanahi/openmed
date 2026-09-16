"""Synthetic offline tests for history-family section segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from openmed.clinical import SECTION_LABEL_ALIASES
from openmed.clinical.lexicons import normalize_section_header
from openmed.clinical.sections import (
    UNSECTIONED_SECTION,
    SectionSpan,
    segment_history_family,
)


@dataclass(frozen=True)
class _HistoryFixture:
    name: str
    text: str
    gold: tuple[SectionSpan, ...]


def _fixture(name: str, *sections: tuple[str, str]) -> _HistoryFixture:
    text = "".join(fragment for _, fragment in sections)
    cursor = 0
    gold: list[SectionSpan] = []
    for label, fragment in sections:
        end = cursor + len(fragment)
        gold.append(SectionSpan(label=label, start=cursor, end=end))
        cursor = end
    return _HistoryFixture(name=name, text=text, gold=tuple(gold))


HISTORY_FIXTURES = (
    _fixture(
        "full_names",
        (UNSECTIONED_SECTION, "Synthetic follow-up note.\n\n"),
        (
            "history_of_present_illness",
            "History of Present Illness: Dry cough began this morning.\n",
        ),
        ("past_medical_history", "Past Medical History: Childhood asthma.\n"),
        ("family_history", "Family History: A relative has seasonal allergies.\n"),
        ("social_history", "Social History: Lives with a housemate."),
    ),
    _fixture(
        "short_aliases",
        ("history_of_present_illness", "HPI: Headache after a late night.\n"),
        ("past_medical_history", "PMH: Synthetic remote ankle sprain.\n"),
        ("family_history", "FH: One relative uses reading glasses.\n"),
        ("social_history", "SH: Walks to work."),
    ),
    _fixture(
        "standalone_blank_line_headers",
        (UNSECTIONED_SECTION, "Synthetic intake narrative.\n\n"),
        ("history_of_present_illness", "HPI\nDry cough for two days.\n\n"),
        ("past_medical_history", "PMH\nNo prior surgery.\n\n"),
        ("family_history", "FH\nNo synthetic conditions recorded.\n\n"),
        ("social_history", "SH\nEnjoys gardening."),
    ),
    _fixture(
        "indented_headers",
        (
            "history_of_present_illness",
            "    History of Present Illness:\n        Mild cough.\n",
        ),
        ("past_medical_history", "\tPast Medical History: None recorded.\n"),
        ("family_history", "  Family Medical History: Noncontributory.\n"),
        ("social_history", "    Social Hx: Shares an apartment."),
    ),
    _fixture(
        "bulleted_headers",
        ("history_of_present_illness", "- HPI: Sore throat today.\n"),
        ("past_medical_history", "* PMH: Prior wrist strain.\n"),
        ("family_history", "• FH: No relevant history recorded.\n"),
        ("social_history", "- SH: Drinks tea."),
    ),
    _fixture(
        "numbered_headers",
        ("history_of_present_illness", "1. HPI: Fatigue this week.\n"),
        ("past_medical_history", "2) PMH: No hospital stays.\n"),
        ("family_history", "3. FH: A relative has hay fever.\n"),
        ("social_history", "4) SH: Cycles on weekends."),
    ),
    _fixture(
        "case_spacing_and_full_width_colons",
        ("history_of_present_illness", "hPi：Transient nausea.\n"),
        ("past_medical_history", "PAST   MEDICAL HISTORY ﹕ None recorded.\n"),
        ("family_history", "family history꞉ Noncontributory.\n"),
        ("social_history", "sOcIaL hIsToRy: Remote office work."),
    ),
    _fixture(
        "generic_history",
        (UNSECTIONED_SECTION, "Synthetic summary.\n"),
        ("history", "History: Occasional seasonal sneezing.\n"),
        ("family_history", "Family History: No relevant history recorded."),
    ),
    _fixture(
        "windows_newlines",
        (UNSECTIONED_SECTION, "Synthetic Windows-style note.\r\n\r\n"),
        ("history_of_present_illness", "HPI: Mild congestion.\r\n"),
        ("past_medical_history", "PMH: No prior admission.\r\n"),
        ("family_history", "FH: Noncontributory.\r\n"),
        ("social_history", "SH: Reads daily.\r\n"),
    ),
    _fixture(
        "unrelated_header_is_content",
        (
            "history_of_present_illness",
            "HPI: Improving.\nMedications: Synthetic tablet daily.\n",
        ),
        ("past_medical_history", "PMH: No prior procedures."),
    ),
    _fixture(
        "header_words_inside_prose",
        (
            UNSECTIONED_SECTION,
            "The synthetic HPI summary mentions PMH, FH, and SH inline only.",
        ),
    ),
    _fixture(
        "leading_blank_lines",
        (UNSECTIONED_SECTION, "\n  \n"),
        ("history_of_present_illness", "HPI: Brief dizziness.\n"),
        ("social_history", "SH: Uses public transit."),
    ),
    _fixture(
        "single_header_at_start",
        (
            "past_medical_history",
            "Medical History: Synthetic childhood ear infection.",
        ),
    ),
)


@pytest.mark.parametrize(
    "case",
    HISTORY_FIXTURES,
    ids=lambda case: case.name,
)
def test_history_family_gold_labels_boundaries_and_coverage(
    case: _HistoryFixture,
) -> None:
    predicted = segment_history_family(case.text)

    assert predicted == case.gold
    assert predicted[0].start == 0
    assert predicted[-1].end == len(case.text)
    assert all(left.end == right.start for left, right in zip(predicted, predicted[1:]))
    assert "".join(case.text[span.start : span.end] for span in predicted) == case.text


def test_history_family_fixture_quality_exceeds_acceptance_thresholds() -> None:
    correct_labels = 0
    gold_headers = 0
    boundary_ious: list[float] = []

    for case in HISTORY_FIXTURES:
        predicted = segment_history_family(case.text)
        predicted_by_start = {span.start: span for span in predicted}
        for gold in case.gold:
            if gold.label != UNSECTIONED_SECTION:
                gold_headers += 1
                predicted_at_boundary = predicted_by_start.get(gold.start)
                correct_labels += int(
                    predicted_at_boundary is not None
                    and predicted_at_boundary.label == gold.label
                )

        for index in range(max(len(case.gold), len(predicted))):
            if index >= len(case.gold) or index >= len(predicted):
                boundary_ious.append(0.0)
                continue
            gold = case.gold[index]
            candidate = predicted[index]
            intersection = max(
                0,
                min(gold.end, candidate.end) - max(gold.start, candidate.start),
            )
            union = max(gold.end, candidate.end) - min(gold.start, candidate.start)
            boundary_ious.append(intersection / union)

    header_label_accuracy = correct_labels / gold_headers
    mean_boundary_character_iou = sum(boundary_ious) / len(boundary_ious)

    assert header_label_accuracy >= 0.90
    assert mean_boundary_character_iou >= 0.85


@pytest.mark.parametrize(
    ("header", "canonical"),
    (
        ("HPI", "history_of_present_illness"),
        ("PMH", "past_medical_history"),
        ("FH", "family_history"),
        ("SH", "social_history"),
    ),
)
def test_history_aliases_use_context_canonical_labels(
    header: str,
    canonical: str,
) -> None:
    text = f"{header}: Synthetic content."

    assert SECTION_LABEL_ALIASES[normalize_section_header(header)] == canonical
    assert segment_history_family(text) == (
        SectionSpan(label=canonical, start=0, end=len(text)),
    )


def test_leading_prose_is_one_unsectioned_span() -> None:
    text = "Synthetic preamble line one.\nLine two.\n\nHPI: New dry cough."
    header_start = text.index("HPI:")

    assert segment_history_family(text)[0] == SectionSpan(
        label=UNSECTIONED_SECTION,
        start=0,
        end=header_start,
    )


def test_empty_text_has_no_section_spans() -> None:
    assert segment_history_family("") == ()
