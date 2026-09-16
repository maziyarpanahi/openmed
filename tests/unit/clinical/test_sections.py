"""Synthetic offline tests for rules-first clinical section detection."""

from __future__ import annotations

import pytest

from openmed.clinical import canonical_section_label
from openmed.clinical.lexicons import load_section_headers
from openmed.clinical.sections import (
    LIST_SECTION_LOINC_CODES,
    UNSECTIONED_SECTION,
    SectionSpan,
    detect_sections,
    is_list_bearing_section,
    list_section_label,
    parse_section_lists,
    validate_section_spans,
)

CANONICAL_SECTION_LABELS = {
    "history_of_present_illness",
    "past_medical_history",
    "problem_list",
    "medications",
    "allergies",
    "social_history",
    "assessment_and_plan",
    "findings",
    "impression",
}


def test_section_header_resource_covers_canonical_labels_and_synonyms() -> None:
    headers = load_section_headers()

    assert CANONICAL_SECTION_LABELS <= headers.keys()
    assert "HPI" in headers["history_of_present_illness"]
    assert "PMH" in headers["past_medical_history"]
    assert "Soc Hx" in headers["social_history"]
    assert "A/P" in headers["assessment_and_plan"]
    assert "CC" in headers["chief_complaint"]
    assert "ROS" in headers["review_of_systems"]


def test_detect_sections_covers_canonical_note_sections_without_offset_gaps() -> None:
    text = (
        "Synthetic note preamble.\n"
        "HPI: Dry cough today.\n"
        "PMH\nChildhood asthma.\n"
        "MEDICATIONS: Albuterol as directed.\n"
        "ALLERGIES\nNo known allergies.\n"
        "Soc Hx: Never smoked.\n"
        "A/P\nSupportive care.\n"
        "FINDINGS: Clear lungs.\n"
        "IMPRESSION\nNo acute finding."
    )

    sections = detect_sections(text)

    assert [section["label"] for section in sections] == [
        UNSECTIONED_SECTION,
        "history_of_present_illness",
        "past_medical_history",
        "medications",
        "allergies",
        "social_history",
        "assessment_and_plan",
        "findings",
        "impression",
    ]
    assert sections[0]["start"] == 0
    assert sections[-1]["end"] == len(text)
    assert all(isinstance(section, SectionSpan) for section in sections)
    assert sections[1].label == "history_of_present_illness"
    assert (sections[1].start, sections[1].end) == (
        sections[1]["start"],
        sections[1]["end"],
    )
    assert all(
        left["end"] == right["start"] for left, right in zip(sections, sections[1:])
    )
    assert "".join(text[span["start"] : span["end"]] for span in sections) == text
    validate_section_spans(text, sections)


def test_section_synonyms_normalize_to_canonical_labels() -> None:
    text = "CC: Cough.\nROS: No fever.\nSoc Hx: Never smoked.\nA/P: Supportive care."

    assert [section["label"] for section in detect_sections(text)] == [
        "chief_complaint",
        "review_of_systems",
        "social_history",
        "assessment_and_plan",
    ]
    assert canonical_section_label("Impression") == "impression"


def test_list_bearing_sections_include_loinc_metadata_and_absolute_items() -> None:
    text = (
        "HPI: Synthetic cough.\n"
        "MEDICATIONS:\n"
        "1. Metformin 500 mg twice daily\n"
        "2. Lisinopril 10 mg daily\n"
        "PROBLEM LIST:\n"
        "- Hypertension\n"
        "- Type 2 diabetes"
    )

    sections = detect_sections(text)
    medication_section = next(
        section for section in sections if section.label == "medications"
    )
    problem_section = next(
        section for section in sections if section.label == "problem_list"
    )
    items = parse_section_lists(text, sections)
    top_level = [item for item in items if item.nesting_level == 0]

    assert medication_section["loinc_code"] == LIST_SECTION_LOINC_CODES["medications"]
    assert problem_section["loinc_code"] == LIST_SECTION_LOINC_CODES["problem_list"]
    assert len(top_level) == 4
    assert [item.marker for item in top_level] == ["1.", "2.", "-", "-"]
    assert all(text[item.start : item.end] == item.text for item in items)


def test_list_bearing_section_can_be_identified_by_loinc_code() -> None:
    text = "Penicillin: rash\nLatex: hives"
    coded_section = SectionSpan(
        label="external_section",
        start=0,
        end=len(text),
        coding={"system": "http://loinc.org", "code": "48765-2"},
    )

    assert is_list_bearing_section(coded_section)
    assert list_section_label(coded_section) == "allergies"
    assert len(parse_section_lists(text, (coded_section,))) == 2


def test_inline_list_header_preserves_first_item_offset() -> None:
    text = "MEDICATIONS: Metformin 500 mg daily\nLisinopril 10 mg daily"

    items = parse_section_lists(text)

    assert len(items) == 2
    assert items[0].content_text.startswith("Metformin")
    assert items[0].start == text.index("Metformin")
    assert text[items[0].start : items[0].end] == items[0].text


def test_headerless_note_returns_one_unsectioned_preamble_span() -> None:
    text = "Synthetic narrative without a recognized clinical header."

    assert detect_sections(text) == (
        SectionSpan(label=UNSECTIONED_SECTION, start=0, end=len(text)),
    )


@pytest.mark.parametrize(
    ("spans", "message"),
    (
        (
            (
                SectionSpan(label="history", start=0, end=4),
                SectionSpan(label="plan", start=5, end=8),
            ),
            "gap",
        ),
        (
            (
                SectionSpan(label="history", start=0, end=5),
                SectionSpan(label="plan", start=4, end=8),
            ),
            "overlap",
        ),
        ((SectionSpan(label="history", start=0, end=4),), "gap"),
    ),
)
def test_section_span_validator_rejects_gaps_and_overlaps(
    spans: tuple[SectionSpan, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_section_spans("abcdefgh", spans)
