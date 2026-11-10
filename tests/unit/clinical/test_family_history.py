"""Focused tests for family-history relation attribution."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import (
    FAMILY_EXPERIENCER,
    FamilyHistoryRecord,
    assert_context,
    extract_family_history,
)
from openmed.clinical.experiencer import ExperiencerAssignment
from openmed.clinical.sections import detect_sections

FIXTURE = Path(__file__).parents[2] / "fixtures/clinical/family_history_sections.json"


def _case() -> dict:
    suite = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert suite["synthetic"] is True
    assert suite["provenance"].startswith("Synthetic offline")
    return suite["cases"][0]


def test_golden_family_history_section_attributes_multiple_relatives() -> None:
    case = _case()
    sections = detect_sections(case["text"])
    condition_spans = [span for span in case["spans"] if span["label"] == "CONDITION"]
    refined = assert_context(case["text"], condition_spans, sections=sections)

    records = extract_family_history(
        case["spans"],
        refined,
        sections,
        text=case["text"],
    )

    fields = ("relative_role", "condition_span", "onset_age", "vital_status")
    assert [
        {field: record.to_dict()[field] for field in fields} for record in records
    ] == case["expected"]
    assert all(record.record_type == "FamilyHistory" for record in records)


def test_patient_experiencer_is_excluded_from_family_history() -> None:
    case = _case()
    sections = detect_sections(case["text"])
    patient = next(span for span in case["spans"] if span["text"] == "asthma")
    family = next(span for span in case["spans"] if span["text"] == "breast cancer")
    assignments = [
        {
            **family,
            "experiencer": FAMILY_EXPERIENCER,
            "relative_role": "mother",
        },
        {**patient, "experiencer": "patient"},
    ]

    records = extract_family_history(
        [family, patient], assignments, sections, text=case["text"]
    )

    assert [record.condition_span for record in records] == [(27, 40)]


def test_refined_assignment_preserves_relative_cue_offset() -> None:
    text = "Family History: The patient's father had an MI at 50."
    condition_start = text.index("MI")
    span = {
        "text": "MI",
        "label": "CONDITION",
        "start": condition_start,
        "end": condition_start + 2,
    }
    assignment = ExperiencerAssignment(
        experiencer=FAMILY_EXPERIENCER,
        cue="father",
        cue_offset=(text.index("father"), text.index("father") + 6),
        source="cue",
    )

    [record] = extract_family_history(
        [span],
        {tuple((span["start"], span["end"])): assignment},
        text=text,
    )

    assert record.relative_role == "father"
    assert record.relative_span == (text.index("father"), text.index("father") + 6)
    assert record.onset_age == 50


def test_family_history_record_serialization_contains_offsets_not_source_text() -> None:
    record = FamilyHistoryRecord(
        relative_role="mother",
        condition_span=(10, 22),
        onset_age=42,
        vital_status="deceased",
    )

    payload = record.to_dict()

    assert payload["condition_span"] == [10, 22]
    assert payload["relative_role"] == "mother"
    assert "breast" not in json.dumps(payload).casefold()
    assert "text" not in payload
