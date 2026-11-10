"""Tests for deterministic problem attribute relation extraction."""

from __future__ import annotations

from openmed.clinical import (
    PROBLEM_RELATION_ADVISORY,
    ClinicalContextResult,
    Relation,
    assert_context,
    extract_problem_relations,
)


def test_severity_and_body_site_bind_to_pneumonia() -> None:
    text = "severe pneumonia, right lower lobe"
    spans = [
        _span(text, "severe", "SEVERITY"),
        _span(text, "pneumonia", "PROBLEM"),
        _span(text, "right lower lobe", "BODY_SITE"),
    ]

    relations = extract_problem_relations(text, spans)

    assert [(item.type, item.head.text, item.tail.text) for item in relations] == [
        ("severity", "pneumonia", "severe"),
        ("body_site", "pneumonia", "right lower lobe"),
    ]
    assert all(isinstance(item, Relation) for item in relations)
    assert [item.relation_type for item in relations] == [
        "problem_to_severity",
        "problem_to_body_site",
    ]
    assert all(0.0 < item.score <= 1.0 for item in relations)
    assert all(
        text[item.tail.start : item.tail.end] == item.tail.text for item in relations
    )


def test_chronic_cue_sets_problem_status() -> None:
    text = "chronic kidney disease"
    spans = [_span(text, "kidney disease", "PROBLEM")]

    relations = extract_problem_relations(text, spans)

    assert [(item.type, item.tail.text) for item in relations] == [
        ("status", "chronic")
    ]
    assert relations[0].tail.derived is False
    assert relations[0].relation_type == "problem_to_status"


def test_context_status_overrides_lexical_status_and_preserves_provenance() -> None:
    text = "Active pneumonia and myocardial infarction"
    pneumonia = _span(text, "pneumonia", "PROBLEM")
    pneumonia["metadata"] = {
        "clinical_context": {"negation": "negated", "temporality": "recent"}
    }
    infarction = _span(text, "myocardial infarction", "PROBLEM")
    infarction["temporality"] = "historical"

    relations = extract_problem_relations(text, [pneumonia, infarction])

    assert [(item.head.text, item.tail.text) for item in relations] == [
        ("pneumonia", "negated"),
        ("myocardial infarction", "historical"),
    ]
    assert all(item.type == "status" for item in relations)
    assert all(item.tail.derived for item in relations)
    assert all(item.tail.offset_key() == item.head.offset_key() for item in relations)
    assert relations[0].tail.to_dict()["derived"] is True


def test_missing_context_degrades_without_inventing_status() -> None:
    text = "mild pneumonia"
    spans = [
        _span(text, "mild", "SEVERITY"),
        _span(text, "pneumonia", "PROBLEM"),
    ]

    relations = extract_problem_relations(text, spans)

    assert [(item.type, item.tail.text) for item in relations] == [("severity", "mild")]


def test_context_result_object_is_consumed_when_supplied() -> None:
    text = "Pneumonia"
    problem = _span(text, "Pneumonia", "PROBLEM")
    problem["clinical_context"] = ClinicalContextResult(
        temporality="historical",
        certainty="certain",
        negation="affirmed",
    )

    relations = extract_problem_relations(text, [problem])

    assert [(item.type, item.tail.text) for item in relations] == [
        ("status", "historical")
    ]


def test_assert_context_output_sets_historical_status() -> None:
    text = "History of pneumonia"
    problem = _span(text, "pneumonia", "PROBLEM")

    enriched = assert_context(text, [problem])
    relations = extract_problem_relations(text, enriched)

    assert [(item.type, item.tail.text) for item in relations] == [
        ("status", "historical")
    ]


def test_each_modifier_binds_to_the_nearest_problem() -> None:
    text = "severe pneumonia and mild asthma"
    spans = [
        _span(text, "severe", "SEVERITY"),
        _span(text, "pneumonia", "PROBLEM"),
        _span(text, "mild", "SEVERITY"),
        _span(text, "asthma", "PROBLEM"),
    ]

    relations = extract_problem_relations(text, spans)

    assert [(item.head.text, item.tail.text) for item in relations] == [
        ("pneumonia", "severe"),
        ("asthma", "mild"),
    ]


def test_free_floating_or_cross_clause_modifier_is_not_bound() -> None:
    distant_text = "Severe symptoms persisted for seven full days before pneumonia"
    distant_spans = [
        _span(distant_text, "Severe", "SEVERITY"),
        _span(distant_text, "pneumonia", "PROBLEM"),
    ]
    cross_clause_text = "Right lower lobe noted. Pneumonia was diagnosed."
    cross_clause_spans = [
        _span(cross_clause_text, "Right lower lobe", "BODY_SITE"),
        _span(cross_clause_text, "Pneumonia", "PROBLEM"),
    ]

    assert extract_problem_relations(distant_text, distant_spans) == ()
    assert extract_problem_relations(cross_clause_text, cross_clause_spans) == ()


def test_known_section_boundary_blocks_relation() -> None:
    text = "pneumonia severe"
    spans = [
        _span(text, "pneumonia", "PROBLEM"),
        _span(text, "severe", "SEVERITY"),
    ]
    sections = (
        {"label": "assessment", "start": 0, "end": 10},
        {"label": "plan", "start": 10, "end": len(text)},
    )

    assert extract_problem_relations(text, spans, sections=sections) == ()


def test_public_api_carries_clinical_review_advisory() -> None:
    docstring = extract_problem_relations.__doc__ or ""

    assert "not an automated diagnosis" in docstring
    assert "clinician review" in PROBLEM_RELATION_ADVISORY


def _span(text: str, surface: str, label: str) -> dict[str, object]:
    start = text.index(surface)
    return {
        "text": surface,
        "label": label,
        "start": start,
        "end": start + len(surface),
        "score": 1.0,
    }
