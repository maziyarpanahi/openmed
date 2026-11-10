from openmed.structured import (
    REQUIRED_DISCHARGE_SLOTS,
    canonical_discharge_slot,
    structure_discharge_summary,
)


def _detected_sections(text: str, headers: list[tuple[str, str]]) -> list[dict]:
    sections: list[dict] = []
    for index, (label, header) in enumerate(headers):
        start = text.index(header)
        end = (
            text.index(headers[index + 1][1]) if index + 1 < len(headers) else len(text)
        )
        sections.append(
            {
                "label": label,
                "header": header,
                "start": start,
                "end": end,
                "header_start": start,
                "header_end": start + len(header),
                "content_start": text.index(":", start) + 1,
            }
        )
    return sections


def test_structures_synthetic_discharge_sections_with_content_offsets():
    text = (
        "Admission Diagnosis: Synthetic pneumonia\n"
        "Hospital Course: Improved with supportive care.\n"
        "D/C Meds: Example tablet once daily\n"
        "Follow-up: Primary care in seven days.\n"
    )
    sections = _detected_sections(
        text,
        [
            ("admission_diagnosis", "Admission Diagnosis"),
            ("hospital_course", "Hospital Course"),
            ("medications", "D/C Meds"),
            ("follow_up", "Follow-up"),
        ],
    )

    result = structure_discharge_summary(text, sections)

    expected_content = {
        "admission_diagnosis": "Synthetic pneumonia",
        "hospital_course": "Improved with supportive care.",
        "discharge_medications": "Example tablet once daily",
        "follow_up": "Primary care in seven days.",
    }
    for slot_name, expected in expected_content.items():
        slot = result[slot_name]
        assert slot is not None
        assert slot["slot"] == slot_name
        assert slot["content"] == expected
        assert text[slot["start"] : slot["end"]] == expected
    assert result["missing_required_slots"] == []


def test_normalizes_common_header_aliases_to_canonical_slots():
    assert canonical_discharge_slot("Admission Dx") == "admission_diagnosis"
    assert canonical_discharge_slot("BRIEF HOSPITAL COURSE") == "hospital_course"
    assert canonical_discharge_slot("D/C Meds") == "discharge_medications"
    assert canonical_discharge_slot("Follow-Up Instructions") == "follow_up"
    assert canonical_discharge_slot("Past Medical History") is None


def test_reports_missing_required_slots_in_canonical_order():
    text = "Hospital Course: Stable throughout the synthetic admission."
    result = structure_discharge_summary(
        text,
        [
            {
                "label": "hospital_course",
                "start": 0,
                "end": len(text),
                "header": "Hospital Course",
            }
        ],
    )

    assert result["hospital_course"] == {
        "slot": "hospital_course",
        "header": "Hospital Course",
        "content": "Stable throughout the synthetic admission.",
        "start": text.index("Stable"),
        "end": len(text),
    }
    assert result["missing_required_slots"] == [
        slot for slot in REQUIRED_DISCHARGE_SLOTS if slot != "hospital_course"
    ]
    assert result["admission_diagnosis"] is None
    assert result["discharge_medications"] is None
    assert result["follow_up"] is None


def test_rejects_recognized_sections_with_invalid_offsets():
    text = "D/C Meds: Example tablet"

    try:
        structure_discharge_summary(
            text,
            [
                {
                    "header": "D/C Meds",
                    "start": 0,
                    "end": len(text) + 1,
                }
            ],
        )
    except ValueError as error:
        assert "outside source text" in str(error)
    else:
        raise AssertionError("invalid source offsets must be rejected")
