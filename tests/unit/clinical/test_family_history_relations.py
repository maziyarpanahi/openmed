"""Focused tests for condition-to-family-member relation extraction."""

from __future__ import annotations

import pytest

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


def test_uncertainty_is_scoped_to_each_relative_clause() -> None:
    text = "mother may have breast cancer, father has myocardial infarction"

    relations = extract_family_history_relations(
        text,
        [
            _span(text, "breast cancer", "CONDITION"),
            _span(text, "myocardial infarction", "CONDITION"),
        ],
    )

    assert [
        (relation.relative.text, relation.condition.text, relation.certainty)
        for relation in relations
    ] == [
        ("mother", "breast cancer", UNCERTAIN),
        ("father", "myocardial infarction", CERTAIN),
    ]


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


def test_conjoined_relative_clauses_keep_the_governing_subject():
    text = "mother previously had breast cancer and father had MI"
    relations = extract_family_history_relations(
        text,
        [_span(text, "breast cancer", "CONDITION"), _span(text, "MI", "CONDITION")],
    )
    assert [(r.relative.text, r.condition.text) for r in relations] == [
        ("mother", "breast cancer"),
        ("father", "MI"),
    ]


def test_surface_patient_subject_does_not_inherit_a_family_condition():
    text = "mother has diabetes and patient has asthma"
    assert (
        extract_family_history_relations(text, [_span(text, "asthma", "CONDITION")])
        == ()
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_patient_context_always_suppresses_family_relation(reverse):
    text = "mother has diabetes"
    family = dict(_span(text, "diabetes", "CONDITION"), experiencer="family")
    patient = dict(family, experiencer="patient")
    spans = [family, patient]
    if reverse:
        spans.reverse()
    assert extract_family_history_relations(text, spans) == ()


def test_nested_patient_context_cannot_be_hidden_by_top_level_family():
    text = "mother has diabetes"
    span = dict(_span(text, "diabetes", "CONDITION"), experiencer="family")
    span["metadata"] = {"clinical_context": {"experiencer": "patient"}}
    assert extract_family_history_relations(text, [span]) == ()


@pytest.mark.parametrize("certainty", ["unknown", "unreviewed"])
def test_unknown_explicit_certainty_is_not_promoted(certainty):
    text = "mother has diabetes"
    span = dict(_span(text, "diabetes", "CONDITION"), certainty=certainty)
    [relation] = extract_family_history_relations(text, [span])
    assert relation.certainty == UNCERTAIN


def test_conflicting_certainty_axes_preserve_uncertainty():
    text = "mother has diabetes"
    span = dict(
        _span(text, "diabetes", "CONDITION"), certainty=CERTAIN, uncertainty=True
    )
    [relation] = extract_family_history_relations(text, [span])
    assert relation.certainty == UNCERTAIN


@pytest.mark.parametrize("start", [True, 11.5, "11"])
def test_noninteger_offsets_are_not_coerced(start):
    text = "mother has diabetes"
    span = dict(_span(text, "diabetes", "CONDITION"), start=start)
    with pytest.raises(ValueError):
        extract_family_history_relations(text, [span])


def test_input_iterator_errors_do_not_retain_source_text():
    def items():
        raise RuntimeError("SYNTHETIC-PRIVATE-CONTENT")
        yield

    with pytest.raises(ValueError) as caught:
        extract_family_history_relations("mother has diabetes", items())
    assert "SYNTHETIC" not in str(caught.value)
    assert caught.value.__context__ is None


def test_span_collections_are_bounded(monkeypatch):
    from openmed.clinical.relations import family_history as module

    monkeypatch.setattr(module, "_MAX_ITEMS", 2)
    text = "mother has diabetes"
    span = _span(text, "diabetes", "CONDITION")
    with pytest.raises(ValueError):
        extract_family_history_relations(text, (span for _ in range(3)))


def test_postpositive_relative_remains_supported():
    text = "diabetes in mother"
    [relation] = extract_family_history_relations(
        text, [_span(text, "diabetes", "CONDITION")]
    )
    assert relation.relative.text == "mother"


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_certainty_preserves_uncertain_in_both_orders(reverse):
    text = "mother has diabetes"
    a = dict(_span(text, "diabetes", "CONDITION"), certainty=CERTAIN)
    b = dict(a, certainty=UNCERTAIN)
    [relation] = extract_family_history_relations(text, [b, a] if reverse else [a, b])
    assert relation.certainty == UNCERTAIN


def test_nested_context_graph_is_bounded(monkeypatch):
    from openmed.clinical.relations import family_history as module

    monkeypatch.setattr(module, "_MAX_CONTEXT_NODES", 2)
    text = "mother has diabetes"
    span = _span(text, "diabetes", "CONDITION")
    span["metadata"] = {"clinical_context": {"context": {"experiencer": "family"}}}
    with pytest.raises(ValueError):
        extract_family_history_relations(text, [span])


def test_cyclic_context_does_not_loop():
    text = "mother has diabetes"
    span = _span(text, "diabetes", "CONDITION")
    span["metadata"] = span
    assert len(extract_family_history_relations(text, [span])) == 1


def test_text_and_section_inputs_are_bounded(monkeypatch):
    from openmed.clinical.relations import family_history as module

    monkeypatch.setattr(module, "_MAX_TEXT_LENGTH", 5)
    with pytest.raises(ValueError):
        extract_family_history_relations("mother has diabetes", [])
    monkeypatch.setattr(module, "_MAX_TEXT_LENGTH", 100)
    monkeypatch.setattr(module, "_MAX_ITEMS", 2)
    with pytest.raises(ValueError):
        extract_family_history_relations(
            "mother has diabetes", [], sections=({} for _ in range(3))
        )


@pytest.mark.parametrize(
    "field,value",
    [("score", True), ("relative", None), ("advisory", "SYNTHETIC-PRIVATE-CONTENT")],
)
def test_direct_relation_constructor_validates_metadata(field, value):
    from dataclasses import replace

    text = "mother has diabetes"
    [relation] = extract_family_history_relations(
        text, [_span(text, "diabetes", "CONDITION")]
    )
    with pytest.raises(ValueError) as caught:
        replace(relation, **{field: value})
    assert caught.value.__context__ is None
    assert "SYNTHETIC" not in str(caught.value)
