from __future__ import annotations

import pytest

from openmed.clinical.normalization import (
    normalize_indian_clinical_abbreviation,
    normalize_indian_clinical_surface,
)
from openmed.core.pii_entity_merger import merge_entities_with_semantic_units


@pytest.mark.parametrize(
    ("surface", "canonical"),
    [
        ("Tab.", "tablet"),
        ("Cap", "capsule"),
        ("OD", "once daily"),
        ("BD", "twice daily"),
        ("TDS", "three times daily"),
        ("HS", "at bedtime"),
        ("SOS", "as needed"),
        ("1-0-1", "morning and evening"),
    ],
)
def test_indian_clinical_abbreviation_map(surface: str, canonical: str) -> None:
    assert normalize_indian_clinical_abbreviation(surface) == canonical


def test_indian_clinical_surface_normalizes_multiple_prescription_tokens() -> None:
    result = normalize_indian_clinical_surface("Tab. Nivora 500 mg BD")

    assert result.text == "tablet Nivora 500 mg twice daily"
    assert result.expansions == ("tab", "bd")


def test_indian_abbreviations_are_normalized_before_entity_merge() -> None:
    text = "Tab. Nivora 500 mg BD"
    merged = merge_entities_with_semantic_units(
        [
            {
                "entity_type": "MEDICATION",
                "score": 0.97,
                "start": 0,
                "end": len(text),
                "word": text,
            }
        ],
        text,
        use_semantic_patterns=False,
        india_clinical=True,
    )

    assert merged[0]["word"] == text
    assert merged[0]["normalized_word"] == "tablet Nivora 500 mg twice daily"
    assert merged[0]["clinical_normalization"]["expansions"] == ["tab", "bd"]


def test_transliterated_person_and_location_fragments_bridge_exact_offsets() -> None:
    text = "Patient काव्या (Kavya) from पुणे Pune."
    person_start = text.index("काव्या")
    person_latin_start = text.index("Kavya")
    location_start = text.index("पुणे")
    location_latin_start = text.index("Pune")
    merged = merge_entities_with_semantic_units(
        [
            {
                "entity_type": "PERSON",
                "score": 0.95,
                "start": person_start,
                "end": person_start + len("काव्या"),
                "word": "काव्या",
            },
            {
                "entity_type": "PERSON",
                "score": 0.94,
                "start": person_latin_start,
                "end": person_latin_start + len("Kavya"),
                "word": "Kavya",
            },
            {
                "entity_type": "LOCATION",
                "score": 0.93,
                "start": location_start,
                "end": location_start + len("पुणे"),
                "word": "पुणे",
            },
            {
                "entity_type": "LOCATION",
                "score": 0.92,
                "start": location_latin_start,
                "end": location_latin_start + len("Pune"),
                "word": "Pune",
            },
        ],
        text,
        use_semantic_patterns=False,
        india_clinical=True,
    )

    assert [(entity["word"], entity["start"], entity["end"]) for entity in merged] == [
        ("काव्या (Kavya)", person_start, person_latin_start + len("Kavya)")),
        ("पुणे Pune", location_start, location_latin_start + len("Pune")),
    ]
    assert all(
        entity["india_clinical_merge"]["transliteration_pair"] is True
        for entity in merged
    )


def test_same_script_person_spans_are_not_widened() -> None:
    text = "Patient Asha Sharma"
    entities = [
        {
            "entity_type": "PERSON",
            "score": 0.9,
            "start": 8,
            "end": 12,
            "word": "Asha",
        },
        {
            "entity_type": "PERSON",
            "score": 0.9,
            "start": 13,
            "end": 19,
            "word": "Sharma",
        },
    ]

    assert (
        merge_entities_with_semantic_units(
            entities,
            text,
            use_semantic_patterns=False,
            india_clinical=True,
        )
        == entities
    )
