"""Tests for SDOH status vocabulary normalization."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    STATUS_NORMALIZATION_ADVISORY,
    load_status_vocab,
    normalize_employment_status,
    normalize_living_status,
    normalize_substance_status,
)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("smoker", "current"),
        ("active use", "current"),
        ("drinks daily", "current"),
        ("former smoker", "former"),
        ("quit 2010", "former"),
        ("substance use disorder in remission", "former"),
        ("status post alcohol use disorder", "former"),
        ("denies alcohol", "never"),
        ("never smoker", "never"),
        ("non-smoker", "never"),
    ],
)
def test_normalize_substance_status_from_surface_cues(phrase, expected):
    assert normalize_substance_status(phrase) == expected


def test_normalize_substance_status_folds_negation_axis_to_never():
    assert normalize_substance_status("alcohol", negated=True) == "never"
    assert normalize_substance_status("alcohol", negated="negated") == "never"


def test_normalize_substance_status_folds_historical_current_cue_to_former():
    assert normalize_substance_status("smoker", temporality="historical") == "former"


@pytest.mark.parametrize("phrase", ["", None, "social history not discussed"])
def test_normalize_substance_status_returns_unknown_without_cue(phrase):
    assert normalize_substance_status(phrase) == "unknown"


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("retired", "retired"),
        ("retired teacher", "retired"),
        ("currently employed", "employed"),
        ("previously employed", "former"),
        ("unemployed", "unemployed"),
        ("on disability", "disabled"),
        ("student", "student"),
        ("never employed", "never"),
    ],
)
def test_normalize_employment_status_from_table(phrase, expected):
    assert normalize_employment_status(phrase) == expected


def test_normalize_employment_status_folds_historical_current_cue():
    assert normalize_employment_status("employed", temporality="historical") == "former"


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("homeless x2 years", "homeless"),
        ("formerly homeless", "former"),
        ("lives alone", "lives_alone"),
        ("lives with family", "lives_with_family"),
        ("assisted living", "assisted_living"),
        ("stable housing", "housed"),
        ("denies homelessness", "never"),
    ],
)
def test_normalize_living_status_from_table(phrase, expected):
    assert normalize_living_status(phrase) == expected


def test_status_vocab_includes_provenance_and_advisory_disclaimer():
    payload = load_status_vocab()

    assert payload["schema_version"] >= 1
    assert payload["provenance"]["task"] == "OM-325"
    assert "clinical decision" in payload["provenance"]["disclaimer"]
    assert "not a clinical decision rule" in STATUS_NORMALIZATION_ADVISORY


def test_status_vocab_contains_required_tables():
    payload = load_status_vocab()

    assert set(payload["vocabularies"]) == {"substance", "employment", "living"}
    assert "former" in payload["vocabularies"]["substance"]["statuses"]
    assert "retired" in payload["vocabularies"]["employment"]["statuses"]
    assert "homeless" in payload["vocabularies"]["living"]["statuses"]
