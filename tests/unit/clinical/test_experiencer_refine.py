"""Tests for cue-based experiencer refinement."""

from __future__ import annotations

import unicodedata

import pytest

from openmed.clinical import (
    EXPERIENCER_REFINED_VALUES,
    EXPERIENCER_REFINEMENT_ADVISORY,
    FAMILY_EXPERIENCER,
    OTHER_EXPERIENCER,
    PATIENT_EXPERIENCER,
    ExperiencerAssignment,
    refine_experiencer,
)


def _span(text: str, sub: str, label: str = "CONDITION") -> dict:
    start = text.index(sub)
    return {"start": start, "end": start + len(sub), "label": label, "text": sub}


# --------------------------------------------------------------------------
# Cue-based subject detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "sub", "cue"),
    [
        ("Patient's mother has diabetes", "diabetes", "mother"),
        ("His brother was diagnosed with asthma", "asthma", "brother"),
        (
            "Family history of coronary artery disease",
            "coronary artery disease",
            "family history",
        ),
    ],
)
def test_family_cue_yields_family_experiencer(text, sub, cue):
    span = _span(text, sub)

    result = refine_experiencer(text, span)

    assert isinstance(result, ExperiencerAssignment)
    assert result.experiencer == FAMILY_EXPERIENCER
    assert result.cue == cue
    assert result.source == "cue"
    assert text[result.cue_offset[0] : result.cue_offset[1]].lower() == cue


@pytest.mark.parametrize(
    ("text", "sub", "cue"),
    [
        ("The donor was CMV-positive", "CMV-positive", "donor"),
        ("Her roommate had a similar rash", "rash", "roommate"),
    ],
)
def test_other_cue_yields_other_experiencer(text, sub, cue):
    span = _span(text, sub)

    result = refine_experiencer(text, span)

    assert result.experiencer == OTHER_EXPERIENCER
    assert result.cue == cue
    assert result.source == "cue"


def test_no_cue_defaults_to_patient():
    text = "Patient reports worsening chest pain"
    span = _span(text, "chest pain")

    result = refine_experiencer(text, span)

    assert result.experiencer == PATIENT_EXPERIENCER
    assert result.cue is None
    assert result.source == "default"


# --------------------------------------------------------------------------
# Section prior fallback and cue precedence
# --------------------------------------------------------------------------


def test_section_prior_used_when_no_cue():
    text = "Coronary artery disease and myocardial infarction"
    span = _span(text, "myocardial infarction")

    result = refine_experiencer(text, span, section_experiencer=FAMILY_EXPERIENCER)

    assert result.experiencer == FAMILY_EXPERIENCER
    assert result.source == "section"


def test_cue_overrides_section_prior():
    # Section says patient, but an explicit family cue near the span wins.
    text = "Patient's father also has hypertension"
    span = _span(text, "hypertension")

    result = refine_experiencer(text, span, section_experiencer=PATIENT_EXPERIENCER)

    assert result.experiencer == FAMILY_EXPERIENCER
    assert result.source == "cue"


# --------------------------------------------------------------------------
# Scope: a cue in a different clause must not attach
# --------------------------------------------------------------------------


def test_cue_in_prior_sentence_does_not_attach():
    text = "Mother is healthy. Patient has diabetes."
    span = _span(text, "diabetes")

    result = refine_experiencer(text, span)

    assert result.experiencer == PATIENT_EXPERIENCER
    assert result.source == "default"


def test_nearest_cue_wins_within_clause():
    text = "The sister of the organ donor developed sepsis"
    span = _span(text, "sepsis")

    result = refine_experiencer(text, span)

    # "donor" is closer to the finding than "sister".
    assert result.experiencer == OTHER_EXPERIENCER
    assert result.cue == "donor"


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def test_offsets_stable_under_nfc_normalization():
    raw = "Le père du patient a un cancer"
    text = unicodedata.normalize("NFC", raw)
    span = _span(text, "cancer")

    first = refine_experiencer(text, span)
    second = refine_experiencer(unicodedata.normalize("NFC", text), span)

    assert first == second


def test_refined_values_and_advisory_exposed():
    assert set(EXPERIENCER_REFINED_VALUES) == {
        PATIENT_EXPERIENCER,
        FAMILY_EXPERIENCER,
        OTHER_EXPERIENCER,
    }
    assert isinstance(EXPERIENCER_REFINEMENT_ADVISORY, str)
    assert EXPERIENCER_REFINEMENT_ADVISORY
