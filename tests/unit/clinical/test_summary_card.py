"""Tests for PHI-free clinical summary cards."""

from __future__ import annotations

from dataclasses import dataclass

from openmed.clinical import ClinicalSummaryCard, build_summary_card


@dataclass(frozen=True)
class EntityObject:
    entity_type: str
    system: str | None = None
    code: str | None = None
    text: str | None = None
    start: int | None = None
    end: int | None = None


def test_mixed_entities_count_categories_codes_and_sections_without_phi():
    entities = [
        {
            "category": "problem",
            "system": "snomed",
            "code": "44054006",
            "text": "Jane Doe has diabetes",
            "start": 11,
            "end": 19,
        },
        EntityObject(
            entity_type="medication",
            system="rxnorm",
            code="860975",
            text="metformin for Jane Doe",
            start=30,
            end=39,
        ),
        {
            "label": "lab_result",
            "coding": [{"system": "loinc", "code": "2160-0", "display": "Creatinine"}],
            "value": "creatinine 1.8 on 1974-12-02",
        },
        {
            "type": "procedure",
            "system": "snomed",
            "code": "80146002",
            "word": "appendectomy",
            "offsets": (40, 52),
        },
        {
            "category": "condition",
            "system": "snomed",
            "code": "44054006",
            "text": "repeat diabetes mention",
        },
        {
            "entity_type": "family history",
            "text": "MRN-12345 sibling history",
            "start": 80,
            "end": 95,
        },
    ]
    document = {
        "clinical_entities": entities,
        "sections": [
            {"label": "Assessment", "start": 0, "end": 70},
            {"label": "Plan", "start": 71, "end": 120},
        ],
    }

    card = build_summary_card(document)

    assert card.to_dict() == {
        "entity_counts": {
            "problems": 2,
            "medications": 1,
            "labs": 1,
            "procedures": 1,
            "other": 1,
        },
        "coding_counts": {
            "coded_entities": 5,
            "uncoded_entities": 1,
            "distinct_codes": 4,
        },
        "section_count": 2,
    }

    serialized = card.to_json()
    for unsafe_value in (
        "Jane Doe",
        "MRN-12345",
        "1974-12-02",
        "metformin",
        "appendectomy",
        "44054006",
        "2160-0",
        "80146002",
        "Creatinine",
        "Assessment",
        "Plan",
    ):
        assert unsafe_value not in serialized
    for unsafe_key in ("text", "word", "start", "end", "offsets", "display", "value"):
        assert f'"{unsafe_key}"' not in serialized


def test_empty_document_yields_all_zero_card():
    assert build_summary_card([], sections=[]).to_dict() == {
        "entity_counts": {
            "problems": 0,
            "medications": 0,
            "labs": 0,
            "procedures": 0,
            "other": 0,
        },
        "coding_counts": {
            "coded_entities": 0,
            "uncoded_entities": 0,
            "distinct_codes": 0,
        },
        "section_count": 0,
    }


def test_json_representation_has_fixed_order_and_is_byte_stable():
    card = ClinicalSummaryCard(
        problems=1,
        medications=1,
        labs=0,
        procedures=0,
        other=0,
        coded_entities=1,
        uncoded_entities=1,
        distinct_codes=1,
        section_count=0,
    )

    expected = (
        b'{"entity_counts":{"problems":1,"medications":1,"labs":0,'
        b'"procedures":0,"other":0},"coding_counts":{"coded_entities":1,'
        b'"uncoded_entities":1,"distinct_codes":1},"section_count":0}'
    )

    assert card.to_json().encode("utf-8") == expected
    assert card.to_json().encode("utf-8") == expected


def test_serialization_is_stable_for_equivalent_input_orderings():
    entities = [
        {"category": "problem", "system": "snomed", "code": "44054006"},
        {"category": "medication"},
        {"category": "problem", "system": "snomed", "code": "44054006"},
    ]

    assert (
        build_summary_card(entities).to_json()
        == build_summary_card(tuple(reversed(entities))).to_json()
    )
