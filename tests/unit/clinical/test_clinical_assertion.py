"""Tests for composed clinical assertion records."""

from __future__ import annotations

from openmed.clinical import (
    AFFIRMED,
    CERTAIN,
    HISTORICAL,
    RECENT,
    UNCERTAIN,
    ClinicalAssertion,
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


def test_optional_axes_default_to_none_until_wired():
    assertion = assert_context_axes("no evidence of pneumonia")

    assert assertion.negation is None
    assert assertion.experiencer is None
    assert assertion.to_dict() == {"temporality": RECENT, "certainty": CERTAIN}


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
