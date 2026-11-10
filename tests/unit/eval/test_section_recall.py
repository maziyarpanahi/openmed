"""Unit tests for clinical-note section recall reports."""

from __future__ import annotations

import json

import pytest

from openmed.eval import compute_section_recall as exported_compute_section_recall
from openmed.eval.section_recall import (
    UNSECTIONED_SECTION,
    SectionSpan,
    compute_section_recall,
)


def test_section_recall_matches_hand_computed_values() -> None:
    note = "HPI: John met Jane.\nMedications: Call 555-1212.\n"
    hpi_end = note.index("\nMedications")
    medication_start = note.index("Medications")
    john_start = note.index("John")
    jane_start = note.index("Jane")
    phone_start = note.index("555-1212")

    report = compute_section_recall(
        note,
        [
            SectionSpan("HPI", 0, hpi_end),
            SectionSpan("Medications", medication_start, len(note)),
        ],
        [
            {"start": john_start, "end": john_start + 4, "label": "PERSON"},
            {"start": jane_start, "end": jane_start + 4, "label": "PERSON"},
            {"start": phone_start, "end": phone_start + 8, "label": "PHONE"},
        ],
        [
            {"start": john_start, "end": john_start + 4, "label": "PERSON"},
            {"start": jane_start, "end": jane_start + 2, "label": "PERSON"},
        ],
    )

    assert exported_compute_section_recall is compute_section_recall
    assert report.per_section["HPI"].covered_chars == 6
    assert report.per_section["HPI"].total_chars == 8
    assert report.per_section["HPI"].recall == pytest.approx(6 / 8)
    assert report.per_section["Medications"].covered_chars == 0
    assert report.per_section["Medications"].total_chars == 8
    assert report.per_section["Medications"].recall == 0.0
    assert report.overall.covered_chars == 6
    assert report.overall.total_chars == 16
    assert report.overall.recall == pytest.approx(6 / 16)
    assert report.worst_sections == ("Medications",)


def test_spans_outside_declared_sections_are_unsectioned() -> None:
    note = "HPI: John met Jane.\nMedications: Call 555-1212.\n"
    hpi_end = note.index("\nMedications")
    phone_start = note.index("555-1212")

    report = compute_section_recall(
        note,
        [("HPI", 0, hpi_end)],
        [{"start": phone_start, "end": phone_start + 8, "label": "PHONE"}],
        [{"start": phone_start, "end": phone_start + 4, "label": "PHONE"}],
    )

    assert report.per_section[UNSECTIONED_SECTION].covered_chars == 4
    assert report.per_section[UNSECTIONED_SECTION].total_chars == 8
    assert report.per_section[UNSECTIONED_SECTION].recall == pytest.approx(0.5)
    assert report.worst_sections == (UNSECTIONED_SECTION,)


def test_section_recall_serializes_counts_and_rates_deterministically() -> None:
    note = "HPI: John met Jane.\nMedications: Call 555-1212.\n"
    hpi_end = note.index("\nMedications")
    medication_start = note.index("Medications")
    john_start = note.index("John")
    jane_start = note.index("Jane")
    phone_start = note.index("555-1212")
    report = compute_section_recall(
        note,
        [
            {"name": "HPI", "start": 0, "end": hpi_end},
            {"name": "Medications", "start": medication_start, "end": len(note)},
        ],
        [
            {"start": john_start, "end": john_start + 4, "label": "PERSON"},
            {"start": jane_start, "end": jane_start + 4, "label": "PERSON"},
            {"start": phone_start, "end": phone_start + 8, "label": "PHONE"},
        ],
        [
            {"start": john_start, "end": john_start + 4, "label": "PERSON"},
            {"start": jane_start, "end": jane_start + 2, "label": "PERSON"},
        ],
    )

    json_report = report.to_json()
    markdown_report = report.to_markdown()

    assert json_report == report.to_json()
    assert json.loads(json_report) == {
        "overall": {"covered_chars": 6, "recall": 0.375, "total_chars": 16},
        "per_section": {
            "HPI": {"covered_chars": 6, "recall": 0.75, "total_chars": 8},
            "Medications": {
                "covered_chars": 0,
                "recall": 0.0,
                "total_chars": 8,
            },
            "unsectioned": {"covered_chars": 0, "recall": 1.0, "total_chars": 0},
        },
        "worst_sections": ["Medications"],
    }
    assert markdown_report == report.to_markdown()
    assert markdown_report == (
        "| Section | Covered Chars | Total Chars | Recall |\n"
        "|---|---:|---:|---:|\n"
        "| Medications | 0 | 8 | 0.000000 |\n"
        "| HPI | 6 | 8 | 0.750000 |\n"
        "| unsectioned | 0 | 0 | 1.000000 |\n"
    )
    for forbidden in ("John", "Jane", "555-1212", "start", "end"):
        assert forbidden not in json_report
        assert forbidden not in markdown_report


def test_section_tagged_gold_spans_can_drive_section_recall() -> None:
    note = "HPI: John.\nFamily History: Jane.\n"
    john_start = note.index("John")
    jane_start = note.index("Jane")

    report = compute_section_recall(
        note,
        None,
        [
            {
                "start": john_start,
                "end": john_start + 4,
                "label": "PERSON",
                "section": "HPI",
            },
            {
                "start": jane_start,
                "end": jane_start + 4,
                "label": "PERSON",
                "metadata": {"section": "Family History"},
            },
        ],
        [{"start": john_start, "end": john_start + 4, "label": "PERSON"}],
    )

    assert report.per_section["HPI"].recall == 1.0
    assert report.per_section["Family History"].recall == 0.0
    assert report.worst_sections == ("Family History",)
