"""Tests for deterministic radiology report parsing and stated RADS capture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import (
    RADIOLOGY_REPORT_ADVISORY,
    parse_radiology_report,
)
from openmed.clinical.radiology_report import FINDINGS, IMPRESSION, RECOMMENDATION

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "radiology_report.jsonl"
)


def _load_fixture() -> list[dict]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _norm(value: str) -> str:
    return " ".join(value.split())


# ---------------------------------------------------------------------------
# Fixture-driven gold rows
# ---------------------------------------------------------------------------


def test_fixture_rows_are_synthetic():
    rows = _load_fixture()
    assert rows
    assert all(row["metadata"]["synthetic"] is True for row in rows)


@pytest.mark.parametrize("row", _load_fixture())
def test_parser_matches_gold_row(row):
    parsed = parse_radiology_report(row["text"])
    gold = row["gold"]
    for key in ("findings", "impression", "recommendation"):
        assert _norm(parsed[f"{key}_text"]) == _norm(gold[f"{key}_text"]), key
    assert parsed["assessment_system"] == gold["assessment_system"]
    assert parsed["assessment_category"] == gold["assessment_category"]


# ---------------------------------------------------------------------------
# Segmentation behavior
# ---------------------------------------------------------------------------


def test_own_line_headings_segment_and_strip_labels():
    parsed = parse_radiology_report(
        "FINDINGS:\nClear lungs.\n\nIMPRESSION:\nNormal.\n\n"
        "RECOMMENDATION:\nNo follow-up."
    )
    assert parsed["findings_text"] == "Clear lungs."
    assert parsed["impression_text"] == "Normal."
    assert parsed["recommendation_text"] == "No follow-up."


def test_inline_heading_content_on_same_line():
    parsed = parse_radiology_report(
        "FINDINGS: Stable scarring.\nIMPRESSION: Benign.\n"
        "RECOMMENDATION: Routine screening."
    )
    assert parsed["findings_text"] == "Stable scarring."
    assert parsed["impression_text"] == "Benign."
    assert parsed["recommendation_text"] == "Routine screening."


def test_cue_fallback_when_headings_absent():
    parsed = parse_radiology_report(
        "Solid 8 mm nodule right upper lobe. "
        "Impression: suspicious nodule. "
        "Recommendation: PET-CT."
    )
    assert parsed["findings_text"] == "Solid 8 mm nodule right upper lobe."
    assert parsed["impression_text"] == "suspicious nodule."
    assert parsed["recommendation_text"] == "PET-CT."


def test_inline_cues_complete_flattened_line_start_sections():
    parsed = parse_radiology_report(
        "FINDINGS: Stable scarring. IMPRESSION: Benign. "
        "RECOMMENDATION: Routine screening."
    )
    assert parsed["findings_text"] == "Stable scarring."
    assert parsed["impression_text"] == "Benign."
    assert parsed["recommendation_text"] == "Routine screening."


def test_inline_cue_requires_a_label_boundary():
    text = "This is not an impression: just a sentence."
    parsed = parse_radiology_report(text)
    assert parsed["findings_text"] == text
    assert parsed["impression_text"] == ""


def test_report_without_headings_or_cues_is_all_findings():
    text = "Frontal radiograph shows clear lungs without acute abnormality."
    parsed = parse_radiology_report(text)
    assert parsed["findings_text"] == text
    assert parsed["impression_text"] == ""
    assert parsed["recommendation_text"] == ""


def test_section_spans_index_into_original_text():
    text = "FINDINGS: A mass.\nIMPRESSION: Suspicious."
    parsed = parse_radiology_report(text)
    spans = parsed["section_spans"]
    assert text[slice(*spans[FINDINGS])] == "A mass."
    assert text[slice(*spans[IMPRESSION])] == "Suspicious."
    assert spans[RECOMMENDATION] is None


def test_empty_input_yields_empty_template():
    parsed = parse_radiology_report("")
    assert parsed["findings_text"] == ""
    assert parsed["impression_text"] == ""
    assert parsed["recommendation_text"] == ""
    assert parsed["assessment_system"] is None
    assert parsed["assessment_category"] is None
    assert all(span is None for span in parsed["section_spans"].values())


# ---------------------------------------------------------------------------
# Stated-only RADS capture (never inferred)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,system,category",
    [
        ("IMPRESSION: Suspicious mass, BI-RADS category 4.", "BI-RADS", "4"),
        ("IMPRESSION: Incomplete. BI-RADS 0.", "BI-RADS", "0"),
        ("IMPRESSION: Known malignancy, BI-RADS category 6.", "BI-RADS", "6"),
        ("IMPRESSION: Suspicious nodule, Lung-RADS 4B.", "Lung-RADS", "4B"),
        ("IMPRESSION: Probably suspicious, Lung-RADS 4A.", "Lung-RADS", "4A"),
        ("IMPRESSION: Airway lesion, Lung-RADS 4X.", "Lung-RADS", "4X"),
        ("IMPRESSION: Benign nodule, Lung-RADS 2.", "Lung-RADS", "2"),
        # Punctuation/spelling variants that are still explicit statements.
        ("IMPRESSION: BIRADS 4.", "BI-RADS", "4"),
        ("IMPRESSION: BI RADS 4.", "BI-RADS", "4"),
        ("IMPRESSION: Lung RADS 2.", "Lung-RADS", "2"),
        ("IMPRESSION: Lung-RADS 4b.", "Lung-RADS", "4B"),
        ("IMPRESSION: BI-RADS: 5.", "BI-RADS", "5"),
        ("IMPRESSION: The BI-RADS category is 4.", "BI-RADS", "4"),
        ("IMPRESSION: BI-RADS final assessment category is 5.", "BI-RADS", "5"),
        ("IMPRESSION: Lung-RADS assessment category is 4A.", "Lung-RADS", "4A"),
    ],
)
def test_captures_stated_category(text, system, category):
    parsed = parse_radiology_report(text)
    assert parsed["assessment_system"] == system
    assert parsed["assessment_category"] == category


@pytest.mark.parametrize(
    "text",
    [
        "Findings are category 4 in severity.",  # no RADS token
        "We used the BI-RADS lexicon for reporting.",  # token, no number
        "ACR BI-RADS Atlas, fifth edition.",  # token + ordinal word
        "BI-RADS 2013 edition guidelines applied.",  # token + year
        "Lung-RADS version 1.1 criteria.",  # version number
        "The mass measures 4 cm; highly suspicious.",  # bare number
        "BI-RADS 7.",  # out of range
        "Lung-RADS 5.",  # out of range
        # Counts and scale legends adjacent to the token are NOT categories.
        "BI-RADS 4 lesions were identified.",  # count
        "The BI-RADS 0-6 assessment scale was used.",  # scale legend
        "Legend: BI-RADS 0 = incomplete, 1 = negative.",  # legend
        "The BI-RADS 5 point scale was applied.",  # scale
        "BI-RADS 6 categories exist in the lexicon.",  # count
        "Lung-RADS 3 nodules noted in both lobes.",  # count
        "Legend: BI-RADS 0: incomplete, 1: negative.",  # legend entry
        "Legend: BI-RADS category 0 means incomplete.",  # qualified legend
    ],
)
def test_does_not_capture_non_stated_or_invalid_category(text):
    parsed = parse_radiology_report(text)
    assert parsed["assessment_system"] is None
    assert parsed["assessment_category"] is None


def test_count_adjacent_to_token_does_not_shadow_real_category():
    # A count next to the token ("3 nodules") must not shadow the stated
    # assessment ("category 2") elsewhere in the report.
    parsed = parse_radiology_report(
        "FINDINGS: Two BI-RADS 3 nodules in the right breast.\n"
        "IMPRESSION: Benign, BI-RADS category 2."
    )
    assert parsed["assessment_system"] == "BI-RADS"
    assert parsed["assessment_category"] == "2"


def test_qualifier_form_captures_even_when_not_terminal():
    parsed = parse_radiology_report(
        "IMPRESSION: BI-RADS category 4 in the right breast, suspicious."
    )
    assert parsed["assessment_system"] == "BI-RADS"
    assert parsed["assessment_category"] == "4"


def test_no_category_is_inferred_when_absent():
    parsed = parse_radiology_report(
        "FINDINGS: Spiculated mass with suspicious morphology.\n"
        "IMPRESSION: Findings are concerning for malignancy."
    )
    # A suspicious description must NOT yield a category — only stated ones count.
    assert parsed["assessment_system"] is None
    assert parsed["assessment_category"] is None


def test_earliest_stated_category_wins_when_multiple_present():
    parsed = parse_radiology_report(
        "IMPRESSION: Right breast BI-RADS 4. Left breast BI-RADS 2."
    )
    assert parsed["assessment_system"] == "BI-RADS"
    assert parsed["assessment_category"] == "4"


# ---------------------------------------------------------------------------
# Advisory
# ---------------------------------------------------------------------------


def test_advisory_states_category_is_read_not_computed():
    lowered = RADIOLOGY_REPORT_ADVISORY.lower()
    assert "read verbatim" in lowered
    assert "never computed" in lowered
    assert "makes no assessment decision" in lowered
