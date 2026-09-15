"""Focused tests for condition-to-family-member relation extraction."""

from __future__ import annotations

from openmed.clinical import (
    CERTAIN,
    FAMILY_HISTORY_RELATION_TYPE,
    UNCERTAIN,
    FamilyHistoryRelation,
    extract_family_history_relations,
)
from openmed.clinical.sections import detect_sections


def test_multi_relative_sentence_keeps_conditions_separate() -> None:
    text = "mother had breast cancer, father had myocardial infarction"
    spans = [
        _span(text, "breast cancer", "CONDITION"),
        _span(text, "myocardial infarction", "CONDITION"),
    ]

    relations = extract_family_history_relations(text, spans)

    assert [
        (relation.relative.text, relation.condition.text) for relation in relations
    ] == [
        ("mother", "breast cancer"),
        ("father", "myocardial infarction"),
    ]
    assert all(isinstance(relation, FamilyHistoryRelation) for relation in relations)
    assert all(
        relation.relation_type == FAMILY_HISTORY_RELATION_TYPE for relation in relations
    )
    assert all(0.0 < relation.score <= 1.0 for relation in relations)


def test_patient_experiencer_is_not_emitted_as_family_history() -> None:
    text = "mother had breast cancer, patient has asthma"
    asthma = _span(text, "asthma", "CONDITION")
    asthma["metadata"] = {"clinical_context": {"experiencer": "patient"}}

    relations = extract_family_history_relations(
        text,
        [_span(text, "breast cancer", "CONDITION"), asthma],
    )

    assert [
        (relation.relative.text, relation.condition.text) for relation in relations
    ] == [("mother", "breast cancer")]


def test_other_experiencer_and_surface_cue_are_consumed() -> None:
    text = "mother had breast cancer"
    condition = _span(text, "breast cancer", "CONDITION")
    condition["metadata"] = {"clinical_context": {"experiencer": "other"}}

    [relation] = extract_family_history_relations(text, [condition])

    assert relation.relative.text == "mother"
    assert relation.condition.text == "breast cancer"


def test_uncertainty_axis_is_carried_to_the_relation() -> None:
    text = "mother may have breast cancer"

    [relation] = extract_family_history_relations(
        text,
        [_span(text, "breast cancer", "CONDITION")],
    )

    assert relation.certainty == UNCERTAIN


def test_explicit_uncertainty_and_family_section_are_supported() -> None:
    text = "Family History:\nMother has diabetes.\nAssessment:\nPatient has asthma."
    diabetes = _span(text, "diabetes", "CONDITION")
    diabetes["certainty"] = CERTAIN
    sections = detect_sections(text)

    [relation] = extract_family_history_relations(text, [diabetes], sections=sections)

    assert relation.relative.text == "Mother"
    assert relation.condition.text == "diabetes"
    assert relation.certainty == CERTAIN


def _span(text: str, surface: str, label: str) -> dict[str, object]:
    start = text.index(surface)
    return {
        "text": surface,
        "label": label,
        "start": start,
        "end": start + len(surface),
        "score": 1.0,
    }
