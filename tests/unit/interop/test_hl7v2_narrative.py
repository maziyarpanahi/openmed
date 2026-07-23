"""Tests for HL7 v2 de-identified narrative extraction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.interop.hl7v2 import parse_hl7v2
from openmed.interop.hl7v2_narrative import extract_hl7v2_narrative

FIXTURES = Path(__file__).parent / "fixtures"


def fake_deidentifier(text: str, **kwargs):
    """Deterministic offline stand-in for the complete narrative pass."""

    assert kwargs["method"] == "mask"
    assert kwargs["lang"] == "en"
    redacted = (
        text.replace("Jane Roe", "[PERSON]")
        .replace("John Doe", "[PERSON]")
        .replace("jane.roe@example.com", "[EMAIL]")
        .replace("555-0199", "[PHONE]")
        .replace("555-0101", "[PHONE]")
        .replace("MRN67890", "[ID_NUM]")
        .replace("MRN12345", "[ID_NUM]")
    )
    return SimpleNamespace(deidentified_text=redacted)


def test_flat_oru_is_coherent_safe_and_maps_results_to_source_fields():
    result = extract_hl7v2_narrative(
        FIXTURES / "synthetic_phi_oru.hl7",
        deidentifier=fake_deidentifier,
    )

    assert result.mode == "flat"
    assert result.message_type == "ORU^R01"
    assert result.text.startswith("Message type: ORU R01.")
    assert "Observation 1: Clinical note (NOTE)." in result.text
    assert (
        "Result 1: Patient [PERSON] called from [PHONE] about [ID_NUM]." in result.text
    )
    assert (
        "Observation 2: Glucose (GLU). Result 2: 7.1. Units 2: mmol/L." in result.text
    )
    assert "Note 1: Follow-up email [EMAIL] belongs to [PERSON]." in result.text
    assert "Jane Roe" not in result.text
    assert "jane.roe@example.com" not in result.text
    assert "MRN67890" not in result.text

    result_span = result.spans_for("OBX", 5, segment_occurrence=1)[0]
    assert result_span.source.segment_index == 3
    assert result_span.source.path == "OBX[1]-5"
    assert result.text_for(result_span) == (
        "Patient [PERSON] called from [PHONE] about [ID_NUM]."
    )

    glucose_offset = result.text.index("7.1")
    provenance = result.provenance_at(glucose_offset)
    assert [(span.source.path, span.label) for span in provenance] == [
        ("OBX[2]-5", "Result 2")
    ]


def test_sectioned_adt_has_stable_patient_and_encounter_sections():
    first = extract_hl7v2_narrative(
        FIXTURES / "synthetic_phi_adt.hl7",
        mode="sectioned",
        deidentifier=fake_deidentifier,
    )
    second = extract_hl7v2_narrative(
        FIXTURES / "synthetic_phi_adt.hl7",
        mode="sectioned",
        deidentifier=fake_deidentifier,
    )

    assert first == second
    assert first.text.startswith("## Message\n")
    assert "\n\n## Patient\n" in first.text
    assert "\n\n## Encounter\n" in first.text
    assert "Administrative sex: Male (M)" in first.text
    assert "Patient class: Inpatient (I)" in first.text
    assert "DOE" not in first.text
    assert "MRN12345" not in first.text
    assert "1980-01-01" not in first.text
    assert "Date of birth: 1980-01-31" in first.text

    assert [section.name for section in first.sections] == [
        "Message",
        "Patient",
        "Encounter",
    ]
    for section in first.sections:
        assert first.text[section.start : section.end].startswith(f"## {section.name}")


def test_deidentifier_receives_complete_narrative_once():
    calls: list[str] = []

    def deidentifier(text: str, **kwargs):
        calls.append(text)
        assert kwargs == {"method": "mask", "lang": "en"}
        return text.replace("Jane Roe", "[PERSON]")

    result = extract_hl7v2_narrative(
        FIXTURES / "synthetic_phi_oru.hl7",
        deidentifier=deidentifier,
    )

    assert len(calls) == 1
    assert calls[0].startswith("Message type:")
    assert "Patient Jane Roe called" in calls[0]
    assert "Patient Jane Roe called" not in result.text


def test_all_field_mappings_index_nonempty_final_text():
    result = extract_hl7v2_narrative(
        FIXTURES / "synthetic_phi_oru.hl7",
        mode="sectioned",
        deidentifier=fake_deidentifier,
    )

    assert result.field_mappings == result.spans
    assert result.spans
    for span in result.spans:
        assert 0 <= span.start < span.end <= len(result.text)
        assert result.text_for(span)
        assert span.source.segment.isupper()
        assert span.source.field_position > 0


def test_parsed_message_and_mapping_result_are_supported():
    message = parse_hl7v2(
        (FIXTURES / "synthetic_phi_oru.hl7").read_text(encoding="utf-8")
    )

    result = extract_hl7v2_narrative(
        message,
        deidentifier=lambda text, **_: {"deidentified_text": text},
    )

    assert result.message_type == "ORU^R01"
    assert result.spans_for("OBX", 3, segment_occurrence=2)


def test_extractor_calls_public_hl7_parser(monkeypatch):
    import openmed.interop.hl7v2_narrative as narrative_module

    calls: list[str] = []
    parser = narrative_module.parse_hl7v2

    def tracking_parser(message: str):
        calls.append(message)
        return parser(message)

    monkeypatch.setattr(narrative_module, "parse_hl7v2", tracking_parser)

    narrative_module.extract_hl7v2_narrative(
        FIXTURES / "synthetic_phi_adt.hl7",
        deidentifier=lambda text, **_: text,
    )

    assert len(calls) == 1
    assert calls[0].startswith("MSH")


def test_invalid_mode_is_rejected_before_processing():
    with pytest.raises(ValueError, match="flat.*sectioned"):
        extract_hl7v2_narrative(
            FIXTURES / "synthetic_phi_oru.hl7",
            mode="html",  # type: ignore[arg-type]
            deidentifier=fake_deidentifier,
        )
