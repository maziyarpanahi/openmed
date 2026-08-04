"""Tests for deterministic key-value and form-field extraction."""

from __future__ import annotations

from openmed.structured.key_value import (
    KEY_VALUE_ADVISORY,
    FormField,
    extract_form_fields,
    extract_key_value_pairs,
    structure_form_fields,
)


def test_colon_fields_preserve_value_offsets_and_policy_labels():
    text = (
        "Name: Synthetic Patient\n"
        "DOB: 1980-01-02\n"
        "MRN: SYN-001\n"
        "Diagnosis: synthetic asthma\n"
    )

    fields = extract_form_fields(text)

    assert [field["key"] for field in fields] == [
        "Name",
        "DOB",
        "MRN",
        "Diagnosis",
    ]
    for field in fields:
        assert text[field["key_start"] : field["key_end"]] == field["key"]
        assert text[field["value_start"] : field["value_end"]] == field["value"]
        assert 0.0 <= field["confidence"] <= 1.0

    assert [(field["canonical_label"], field["is_phi"]) for field in fields] == [
        ("PERSON", True),
        ("DATE_OF_BIRTH", True),
        ("ID_NUM", True),
        ("PROBLEM", False),
    ]


def test_whitespace_delimited_and_multiline_fields_are_recovered():
    text = (
        "Name    Synthetic Patient\n"
        "Address:\n"
        "123 Synthetic Street\n"
        "Suite 4\n"
        "DOB\t1980-01-02\n"
    )

    fields = extract_key_value_pairs(text)

    assert [(field["key"], field["value"]) for field in fields] == [
        ("Name", "Synthetic Patient"),
        ("Address", "123 Synthetic Street\nSuite 4"),
        ("DOB", "1980-01-02"),
    ]
    address = fields[1]
    assert text[address["value_start"] : address["value_end"]] == address["value"]
    assert address["delimiter"] == "colon"


def test_full_width_colon_and_indented_continuation_are_supported():
    text = "Phone：+1 555 0100\nNotes: first line\n  second line\nNext: value\n"

    fields = structure_form_fields(text)

    assert [(field["key"], field["value"]) for field in fields] == [
        ("Phone", "+1 555 0100"),
        ("Notes", "first line\n  second line"),
        ("Next", "value"),
    ]
    assert fields[0]["delimiter"] == "colon"
    assert fields[1]["value_end"] == text.index("second line") + len("second line")


def test_unknown_layout_label_is_not_marked_as_phi():
    field = extract_form_fields("Preferred language  Synthetic English\n")[0]

    assert field["key"] == "Preferred language"
    assert field["canonical_label"] is None
    assert field["policy_label"] == "CLINICAL_CONCEPT"
    assert field["is_phi"] is False


def test_empty_input_is_deterministic_and_typed():
    assert extract_form_fields("") == []
    assert extract_form_fields("Name: Synthetic") == extract_form_fields(
        "Name: Synthetic"
    )
    assert FormField is not None
    assert isinstance(KEY_VALUE_ADVISORY, str) and KEY_VALUE_ADVISORY
