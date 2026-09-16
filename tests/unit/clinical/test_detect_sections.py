"""Synthetic offline tests for clinical section orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import openmed.clinical.sections.detect as section_detection
from openmed.clinical.sections import (
    UNSECTIONED_SECTION,
    SectionSpan,
    detect_sections,
    validate_sections,
)


@dataclass(frozen=True)
class _GoldSection:
    family: str
    span: SectionSpan


@dataclass(frozen=True)
class _MultiFamilyFixture:
    name: str
    text: str
    gold: tuple[_GoldSection, ...]


def _fixture(
    name: str,
    *sections: tuple[str, str, str],
) -> _MultiFamilyFixture:
    text = "".join(fragment for _, _, fragment in sections)
    cursor = 0
    gold: list[_GoldSection] = []
    for family, label, fragment in sections:
        end = cursor + len(fragment)
        gold.append(
            _GoldSection(
                family=family,
                span=SectionSpan(label=label, start=cursor, end=end),
            )
        )
        cursor = end
    return _MultiFamilyFixture(name=name, text=text, gold=tuple(gold))


MULTI_FAMILY_FIXTURES = (
    _fixture(
        "outpatient_follow_up",
        ("none", UNSECTIONED_SECTION, "Synthetic follow-up note.\n"),
        ("history", "history_of_present_illness", "HPI: Dry cough today.\n"),
        ("list", "medications", "MEDICATIONS: Saline spray as directed.\n"),
        ("assessment", "assessment_and_plan", "A/P: Continue supportive care."),
    ),
    _fixture(
        "mixed_intake_and_imaging",
        ("intake", "chief_complaint", "CC: Mild congestion.\n"),
        ("history", "family_history", "FH: Noncontributory.\n"),
        ("list", "allergies", "ALLERGIES: No known allergies.\n"),
        ("radiology", "findings", "FINDINGS: Clear lungs.\n"),
        ("radiology", "impression", "IMPRESSION: No acute finding."),
    ),
)


@pytest.mark.parametrize(
    "case",
    MULTI_FAMILY_FIXTURES,
    ids=lambda case: case.name,
)
def test_detect_sections_assembles_total_cover_with_high_gold_coverage(
    case: _MultiFamilyFixture,
) -> None:
    predicted = detect_sections(case.text)
    gold_spans = tuple(entry.span for entry in case.gold)
    families = {entry.family for entry in case.gold if entry.family != "none"}

    assert len(families) >= 3
    assert predicted[0].start == 0
    assert predicted[-1].end == len(case.text)
    assert all(left.end == right.start for left, right in zip(predicted, predicted[1:]))
    validate_sections(case.text, predicted)

    correct_characters = 0
    for offset in range(len(case.text)):
        predicted_label = next(
            span.label for span in predicted if span.start <= offset < span.end
        )
        gold_label = next(
            span.label for span in gold_spans if span.start <= offset < span.end
        )
        correct_characters += int(predicted_label == gold_label)
    assert correct_characters / len(case.text) >= 0.88


@pytest.mark.parametrize(
    ("first", "second", "expected_label"),
    (
        (
            SectionSpan(
                label="short_header",
                start=0,
                end=24,
                header_start=0,
                header_end=3,
            ),
            SectionSpan(
                label="long_header",
                start=0,
                end=24,
                header_start=0,
                header_end=4,
            ),
            "long_header",
        ),
        (
            SectionSpan(
                label="later_header",
                start=1,
                end=24,
                header_start=1,
                header_end=5,
            ),
            SectionSpan(
                label="earlier_header",
                start=0,
                end=24,
                header_start=0,
                header_end=4,
            ),
            "earlier_header",
        ),
    ),
)
def test_overlap_resolution_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    first: SectionSpan,
    second: SectionSpan,
    expected_label: str,
) -> None:
    text = "HPI: Synthetic overlap.\n"

    def first_segmenter(
        source: str,
        language: str | None,
    ) -> tuple[SectionSpan, ...]:
        del source, language
        return (first,)

    def second_segmenter(
        source: str,
        language: str | None,
    ) -> tuple[SectionSpan, ...]:
        del source, language
        return (second,)

    monkeypatch.setattr(
        section_detection,
        "_REGISTERED_SECTION_SEGMENTERS",
        (first_segmenter, second_segmenter),
    )

    results = tuple(detect_sections(text) for _ in range(5))

    assert all(result == results[0] for result in results)
    assert [span.label for span in results[0]] == [expected_label]


def test_detect_sections_fills_internal_candidate_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "HPI:x....PLAN:y"
    plan_start = text.index("PLAN")

    def partial_segmenter(
        source: str,
        language: str | None,
    ) -> tuple[SectionSpan, ...]:
        del source, language
        return (
            SectionSpan(
                label="history_of_present_illness",
                start=0,
                end=5,
                header_start=0,
                header_end=3,
            ),
            SectionSpan(
                label="plan",
                start=plan_start,
                end=len(text),
                header_start=plan_start,
                header_end=plan_start + 4,
            ),
        )

    monkeypatch.setattr(
        section_detection,
        "_REGISTERED_SECTION_SEGMENTERS",
        (partial_segmenter,),
    )

    sections = detect_sections(text)

    assert [span.label for span in sections] == [
        "history_of_present_illness",
        UNSECTIONED_SECTION,
        "plan",
    ]
    assert text[sections[1].start : sections[1].end] == "...."
    validate_sections(text, sections)


@pytest.mark.parametrize(
    ("spans", "message"),
    (
        ((SectionSpan(label="history", start=-1, end=2),), "outside"),
        ((SectionSpan(label="history", start=0, end=5),), "outside"),
        (
            (
                SectionSpan(label="history", start=0, end=2),
                SectionSpan(label="plan", start=3, end=4),
            ),
            "gap",
        ),
        (
            (
                SectionSpan(label="history", start=0, end=3),
                SectionSpan(label="plan", start=2, end=4),
            ),
            "overlap",
        ),
    ),
)
def test_validate_sections_rejects_invalid_covers(
    spans: tuple[SectionSpan, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_sections("abcd", spans)
