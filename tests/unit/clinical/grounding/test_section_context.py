"""Focused tests for section-aware grounding context."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from openmed.clinical.grounding import (
    ConceptMatch,
    LexicalConcept,
    LexicalMatcher,
    SectionContextConfig,
    apply_section_context,
)

SYSTEM_URI = "https://example.org/fhir/CodeSystem/synthetic-grounding"


def _match(
    code: str,
    display: str,
    semantic_type: str,
    *,
    score: float = 1.0,
) -> ConceptMatch:
    return ConceptMatch(
        system_uri=SYSTEM_URI,
        code=code,
        display=display,
        score=score,
        match_type="exact",
        matched_term="shared term",
        metadata={"semantic_type": semantic_type},
    )


def _matcher() -> LexicalMatcher:
    return LexicalMatcher(
        {
            "shared term": [
                LexicalConcept(
                    system_uri=SYSTEM_URI,
                    code="ALLERGEN-1",
                    display="Synthetic allergen",
                    metadata={"semantic_type": "allergen"},
                ),
                LexicalConcept(
                    system_uri=SYSTEM_URI,
                    code="MEDICATION-1",
                    display="Synthetic medication",
                    metadata={"semantic_type": "medication"},
                ),
            ]
        }
    )


def test_same_term_is_filtered_to_section_appropriate_concept() -> None:
    matches = _matcher().lookup("shared term")

    allergies = apply_section_context(matches, ["Allergies"], "patient")
    medications = apply_section_context(matches, ["Medications"], "patient")

    assert [match.code for match in allergies] == ["ALLERGEN-1"]
    assert allergies[0].section == "allergies"
    assert allergies[0].experiencer == "patient"
    assert [match.code for match in medications] == ["MEDICATION-1"]
    assert medications[0].section == "medications"


def test_family_history_marks_and_downranks_non_patient_concepts() -> None:
    match = _match("COND-1", "Synthetic family condition", "condition")

    family = apply_section_context((match,), "Family History")
    patient = apply_section_context((match,), "Assessment", "patient")

    assert family[0].section == "family_history"
    assert family[0].experiencer == "family"
    assert family[0].metadata["non_patient"] is True
    assert family[0].metadata["patient_record_eligible"] is False
    assert family[0].ranking_score < patient[0].ranking_score
    assert family[0].context_score is not None


def test_section_rules_are_caller_configurable() -> None:
    rules = SectionContextConfig(
        {
            "custom review": {
                "semantic_type_biases": {"laboratory_test": 0.25},
                "excluded_semantic_types": ("medication",),
                "experiencer": "patient",
            }
        }
    )
    matches = (
        _match("LAB-1", "Synthetic lab", "laboratory test", score=0.8),
        _match("MED-1", "Synthetic medication", "medication", score=1.0),
    )

    result = apply_section_context(matches, "Custom Review", config=rules)

    assert [match.code for match in result] == ["LAB-1"]
    assert result[0].score == pytest.approx(0.8)
    assert result[0].ranking_score == pytest.approx(1.05)
    assert result[0].metadata["section"] == "custom_review"


def test_offset_sections_are_selected_per_match_without_copying_source_text() -> None:
    first = ConceptMatch(
        system_uri=SYSTEM_URI,
        code="A",
        display="Synthetic A",
        score=1.0,
        match_type="exact",
        matched_term="shared term",
        metadata={"semantic_type": "allergen", "start": 2, "end": 8},
    )
    second = ConceptMatch(
        system_uri=SYSTEM_URI,
        code="B",
        display="Synthetic B",
        score=1.0,
        match_type="exact",
        matched_term="shared term",
        metadata={"semantic_type": "medication", "start": 20, "end": 26},
    )

    result = apply_section_context(
        (first, second),
        (
            {"label": "Allergies", "start": 0, "end": 10},
            {"label": "Medications", "start": 10, "end": 30},
        ),
    )

    assert [(match.code, match.section) for match in result] == [
        ("A", "allergies"),
        ("B", "medications"),
    ]
    serialized = asdict(result[0])
    assert "shared term" not in serialized["metadata"]


def test_matcher_applies_context_before_limit() -> None:
    matches = _matcher().lookup(
        "shared term",
        sections="Medications",
        limit=1,
    )

    assert [match.code for match in matches] == ["MEDICATION-1"]


def test_excluded_matches_can_be_retained_for_audit() -> None:
    matches = _matcher().lookup("shared term")

    result = apply_section_context(matches, "Allergies", drop_excluded=False)

    by_code = {match.code: match for match in result}
    assert by_code["MEDICATION-1"].metadata["section_excluded"] is True


def test_apply_section_context_rejects_non_match_inputs() -> None:
    with pytest.raises(TypeError, match="ConceptMatch"):
        apply_section_context((object(),), "Assessment")
