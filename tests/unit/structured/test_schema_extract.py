"""Tests for the schema-guided JSON extraction API."""

from __future__ import annotations

from typing import Any

import pytest

from openmed.structured.schema_extract import (
    SCHEMA_EXTRACT_ADVISORY,
    SchemaDefinitionError,
    extract_to_schema,
    normalize_field_key,
)

# A synthetic clinical note built so every value can be located by ``str.index``
# rather than a hand-counted offset. No real patient data is used.
NOTE = (
    "Chief Complaint: cough\n"
    "Patient Age: 54 years\n"
    "Sex: Female\n"
    "Temperature: 37.8 C\n"
    "Current Smoker: no\n"
    "MRN: 481529\n"
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["patient_age", "sex", "diagnosis"],
    "properties": {
        "patient_age": {"type": "integer", "aliases": ["age"]},
        "sex": {"type": "string", "enum": ["Male", "Female"]},
        "temperature": {"type": "number", "aliases": ["temp"]},
        "smoker": {"type": "boolean", "aliases": ["current smoker"]},
        "mrn": {"type": "string", "pattern": r"\d{6}"},
        "diagnosis": {"type": "string"},
    },
}


def _span(text: str, value: str) -> tuple[int, int]:
    start = text.index(value)
    return start, start + len(value)


# --------------------------------------------------------------------------
# Key-value binding, coercion and offset provenance
# --------------------------------------------------------------------------


def test_returns_schema_valid_object_with_typed_values():
    result = extract_to_schema(NOTE, SCHEMA)

    assert result["data"]["patient_age"] == 54
    assert result["data"]["sex"] == "Female"
    assert result["data"]["temperature"] == 37.8
    assert result["data"]["smoker"] is False
    assert result["data"]["mrn"] == "481529"


def test_each_filled_field_carries_source_offsets():
    result = extract_to_schema(NOTE, SCHEMA)

    for field, raw in (
        ("patient_age", "54 years"),
        ("sex", "Female"),
        ("temperature", "37.8 C"),
        ("mrn", "481529"),
    ):
        binding = result["bindings"][field]
        start, end = _span(NOTE, raw)
        assert (binding["start"], binding["end"]) == (start, end)
        assert NOTE[binding["start"] : binding["end"]] == binding["raw"]
        assert binding["source"] == "key_value"


def test_aliases_match_alternative_labels():
    schema = {"properties": {"age": {"type": "integer", "aliases": ["patient age"]}}}
    result = extract_to_schema(NOTE, schema)

    assert result["data"]["age"] == 54


# --------------------------------------------------------------------------
# Missing required slots and validation failures are reported, not dropped
# --------------------------------------------------------------------------


def test_missing_required_field_is_reported():
    result = extract_to_schema(NOTE, SCHEMA)

    # ``diagnosis`` is required but appears nowhere in the note.
    assert "diagnosis" in result["missing_required"]
    assert "diagnosis" not in result["data"]


def test_coercion_failure_is_reported_and_slot_left_unfilled():
    text = "Patient Age: unknown\nSex: Female\nDiagnosis: asthma\n"
    result = extract_to_schema(text, SCHEMA)

    assert "patient_age" not in result["data"]
    assert "patient_age" in result["missing_required"]
    issue = next(e for e in result["errors"] if e["field"] == "patient_age")
    assert issue["reason"] == "expected an integer value"
    assert issue["raw"] == "unknown"
    assert text[issue["start"] : issue["end"]] == "unknown"


def test_pattern_violation_is_reported():
    text = "MRN: 12AB\n"
    schema = {"properties": {"mrn": {"type": "string", "pattern": r"\d{6}"}}}
    result = extract_to_schema(text, schema)

    assert "mrn" not in result["data"]
    assert result["errors"][0]["reason"] == "value does not match required pattern"


def test_enum_violation_is_reported():
    text = "Sex: Unspecified\n"
    schema = {"properties": {"sex": {"type": "string", "enum": ["Male", "Female"]}}}
    result = extract_to_schema(text, schema)

    assert "sex" not in result["data"]
    assert result["errors"][0]["reason"] == (
        "value is not one of the permitted enum options"
    )


def test_enum_match_is_case_insensitive_and_returns_canonical_form():
    text = "Sex: female\n"
    schema = {"properties": {"sex": {"type": "string", "enum": ["Male", "Female"]}}}
    result = extract_to_schema(text, schema)

    assert result["data"]["sex"] == "Female"


def test_enum_on_non_string_slot_keeps_declared_type():
    # An ``enum`` on a numeric slot must not smuggle the option string back into
    # ``data`` -- the value has to stay an ``int``/``float``.
    text = "Stage: 2\nDose: 2.5\n"
    schema = {
        "properties": {
            "stage": {"type": "integer", "enum": ["1", "2", "3"]},
            "dose": {"type": "number", "enum": ["2.5", "5.0"]},
        }
    }
    result = extract_to_schema(text, schema)

    assert result["data"]["stage"] == 2
    assert isinstance(result["data"]["stage"], int)
    assert result["data"]["dose"] == 2.5
    assert isinstance(result["data"]["dose"], float)


# --------------------------------------------------------------------------
# Entity and table cell binding
# --------------------------------------------------------------------------


def test_detected_entities_bind_to_entity_slots():
    text = "Seen at Mercy General Hospital today."
    start, end = _span(text, "Mercy General Hospital")
    entities = [
        {
            "label": "HOSPITAL",
            "text": "Mercy General Hospital",
            "start": start,
            "end": end,
        }
    ]
    schema = {"properties": {"facility": {"type": "string", "entity": "HOSPITAL"}}}

    result = extract_to_schema(text, schema, entities=entities)

    binding = result["bindings"]["facility"]
    assert result["data"]["facility"] == "Mercy General Hospital"
    assert binding["source"] == "entity"
    assert (binding["start"], binding["end"]) == (start, end)


def test_table_cells_bind_key_value_rows():
    # A synthetic two-column key/value table in ``openmed.structured.Table`` shape.
    text = "Heart Rate 72\nSodium 140\n"
    hr_start, hr_end = _span(text, "72")
    na_start, na_end = _span(text, "140")
    table = {
        "cells": [
            {"row": 0, "column": 0, "text": "Heart Rate", "start": 0, "end": 10},
            {"row": 0, "column": 1, "text": "72", "start": hr_start, "end": hr_end},
            {"row": 1, "column": 0, "text": "Sodium", "start": 14, "end": 20},
            {"row": 1, "column": 1, "text": "140", "start": na_start, "end": na_end},
        ]
    }
    schema = {
        "properties": {
            "heart_rate": {"type": "integer"},
            "sodium": {"type": "integer"},
        }
    }

    result = extract_to_schema(text, schema, tables=[table])

    assert result["data"] == {"heart_rate": 72, "sodium": 140}
    assert result["bindings"]["heart_rate"]["source"] == "table"


def test_entity_source_wins_over_key_value_for_same_slot():
    text = "Patient Age: 54 years\n"
    start, end = _span(text, "54")
    entities = [{"label": "AGE", "text": "54", "start": start, "end": end}]
    schema = {
        "properties": {
            "patient_age": {"type": "integer", "aliases": ["age"], "entity": "AGE"}
        }
    }

    result = extract_to_schema(text, schema, entities=entities)

    binding = result["bindings"]["patient_age"]
    assert binding["source"] == "entity"
    assert (binding["start"], binding["end"]) == (start, end)


# --------------------------------------------------------------------------
# Schema validation (bad schema raises; bad document never does)
# --------------------------------------------------------------------------


def test_non_object_schema_raises():
    with pytest.raises(SchemaDefinitionError):
        extract_to_schema(NOTE, {"type": "array", "properties": {}})


def test_unknown_scalar_type_raises():
    with pytest.raises(SchemaDefinitionError):
        extract_to_schema(NOTE, {"properties": {"x": {"type": "object"}}})


def test_required_referencing_unknown_property_raises():
    with pytest.raises(SchemaDefinitionError):
        extract_to_schema(
            NOTE, {"properties": {"x": {"type": "string"}}, "required": ["y"]}
        )


def test_partial_document_never_raises():
    # Garbage / empty documents yield an empty object, not an exception.
    result = extract_to_schema("", SCHEMA)
    assert result["data"] == {}
    assert set(result["missing_required"]) == {"patient_age", "sex", "diagnosis"}


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_normalize_field_key_collapses_separators_and_case():
    assert normalize_field_key("Patient_Age") == "patient age"
    assert normalize_field_key("PATIENT-AGE") == "patient age"
    assert normalize_field_key(" patient  age ") == "patient age"


def test_deterministic():
    assert extract_to_schema(NOTE, SCHEMA) == extract_to_schema(NOTE, SCHEMA)


def test_advisory_exposed():
    assert isinstance(SCHEMA_EXTRACT_ADVISORY, str) and SCHEMA_EXTRACT_ADVISORY
