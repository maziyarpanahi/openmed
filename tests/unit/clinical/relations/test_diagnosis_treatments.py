"""Tests for explicit diagnosis-to-treatment candidates."""

import json

from openmed.clinical.relations.diagnosis_treatments import (
    generate_diagnosis_treatment_candidates,
)


def _span(text: str, value: str, label: str, **extra):
    start = text.index(value)
    return {"label": label, "start": start, "end": start + len(value), **extra}


def test_requires_explicit_link_and_attaches_uncertainty() -> None:
    text = "Plan: possible pneumonia treated with ceftriaxone."
    spans = [
        _span(text, "pneumonia", "DIAGNOSIS", certainty="uncertain"),
        _span(text, "ceftriaxone", "MEDICATION"),
    ]

    (candidate,) = generate_diagnosis_treatment_candidates(text, spans)

    assert candidate.uncertainty == "possible"
    assert candidate.review_required is True
    assert candidate.treatment_recommendation is False
    assert candidate.treatment_evaluated is False
    payload = json.dumps(candidate.to_dict(), sort_keys=True)
    assert "pneumonia" not in payload
    assert "ceftriaxone" not in payload


def test_rejects_cooccurrence_distance_and_section_violations() -> None:
    text = "Pneumonia and ceftriaxone."
    spans = [
        _span(text, "Pneumonia", "DIAGNOSIS"),
        _span(text, "ceftriaxone", "TREATMENT"),
    ]
    assert generate_diagnosis_treatment_candidates(text, spans) == ()

    linked = "Pneumonia treated with ceftriaxone."
    linked_spans = [
        _span(linked, "Pneumonia", "DIAGNOSIS", section="assessment"),
        _span(linked, "ceftriaxone", "TREATMENT", section="plan"),
    ]
    assert generate_diagnosis_treatment_candidates(linked, linked_spans) == ()
    assert (
        generate_diagnosis_treatment_candidates(
            linked,
            [
                _span(linked, "Pneumonia", "DIAGNOSIS"),
                _span(linked, "ceftriaxone", "TREATMENT"),
            ],
            max_distance=2,
        )
        == ()
    )


def test_supports_treatment_first_language_and_section_allowlist() -> None:
    text = "Plan: ceftriaxone for pneumonia."
    spans = [
        _span(text, "pneumonia", "DIAGNOSIS"),
        _span(text, "ceftriaxone", "TREATMENT"),
    ]
    assert len(generate_diagnosis_treatment_candidates(text, spans)) == 1
    assert (
        generate_diagnosis_treatment_candidates(
            text,
            spans,
            allowed_sections={"assessment"},
        )
        == ()
    )
