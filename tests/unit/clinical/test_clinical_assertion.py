"""Tests for composed clinical assertion records."""

from __future__ import annotations

from openmed.clinical import (
    AFFIRMED,
    CERTAIN,
    HISTORICAL,
    NEGATED,
    RECENT,
    UNCERTAIN,
    ClinicalAssertion,
    assert_context,
    assert_context_axes,
)


def test_history_of_mi_assertion_is_historical_and_certain():
    assertion = assert_context_axes("history of MI")

    assert assertion.temporality == HISTORICAL
    assert assertion.certainty == CERTAIN


def test_possible_pneumonia_assertion_is_uncertain():
    assertion = assert_context_axes("possible pneumonia")

    assert assertion.temporality == RECENT
    assert assertion.certainty == UNCERTAIN


def test_to_dict_round_trips_and_omits_unset_axes():
    assertion = ClinicalAssertion(temporality=HISTORICAL, certainty=CERTAIN)

    data = assertion.to_dict()

    assert data == {"temporality": HISTORICAL, "certainty": CERTAIN}
    assert ClinicalAssertion(**data) == assertion


def test_assert_context_axes_includes_negation_and_leaves_experiencer_unset():
    assertion = assert_context_axes("no evidence of pneumonia")

    assert assertion.negation == NEGATED
    assert assertion.experiencer is None
    assert assertion.to_dict() == {
        "temporality": RECENT,
        "certainty": CERTAIN,
        "negation": NEGATED,
    }


def test_to_dict_includes_optional_axes_when_set():
    assertion = ClinicalAssertion(
        temporality=RECENT,
        certainty=CERTAIN,
        negation=AFFIRMED,
        experiencer="patient",
    )

    assert assertion.to_dict() == {
        "temporality": RECENT,
        "certainty": CERTAIN,
        "negation": AFFIRMED,
        "experiencer": "patient",
    }


def test_clinical_assertion_documents_mapping_and_disclaimer():
    docstring = ClinicalAssertion.__doc__ or ""
    normalized_docstring = " ".join(docstring.split())

    assert "clinicalStatus" in docstring
    assert "verificationStatus=provisional" in docstring
    assert "not asserted" in docstring
    assert "not clinical decisions" in normalized_docstring


def test_assert_context_attaches_all_axes_without_mutating_spans():
    text = "No evidence of pneumonia. Her mother had diabetes."
    spans = [
        {
            "text": "pneumonia",
            "start": text.index("pneumonia"),
            "end": text.index("pneumonia") + len("pneumonia"),
            "metadata": {"source": "synthetic"},
        },
        {
            "text": "diabetes",
            "start": text.index("diabetes"),
            "end": text.index("diabetes") + len("diabetes"),
        },
    ]

    enriched = assert_context(text, spans)

    assert "clinical_context" not in spans[0]["metadata"]
    assert enriched[0]["metadata"]["source"] == "synthetic"
    assert enriched[0]["metadata"]["clinical_context"] == {
        "negation": NEGATED,
        "uncertainty": CERTAIN,
        "experiencer": "patient",
        "temporality": RECENT,
    }
    assert enriched[1]["negation"] == AFFIRMED
    assert enriched[1]["uncertainty"] == CERTAIN
    assert enriched[1]["experiencer"] == "family"
    assert enriched[1]["temporality"] == RECENT
