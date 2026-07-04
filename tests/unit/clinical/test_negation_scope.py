"""Tests for the negation-scope boundary detector."""

from __future__ import annotations

import unicodedata

import pytest

from openmed.clinical import (
    NEGATION_SCOPE_ADVISORY,
    NegationScope,
    detect_negation_scopes,
    negated_spans,
)


def _span(text: str, sub: str, label: str = "CONDITION") -> dict:
    start = text.index(sub)
    return {"start": start, "end": start + len(sub), "label": label, "text": sub}


# --------------------------------------------------------------------------
# Scope boundary computation
# --------------------------------------------------------------------------


def test_forward_cue_scope_stops_at_conjunction():
    text = "no chest pain but has fever"
    chest_pain = _span(text, "chest pain")
    fever = _span(text, "fever")

    scopes = detect_negation_scopes(text, [chest_pain, fever])

    assert len(scopes) == 1
    scope = scopes[0]
    assert isinstance(scope, NegationScope)
    assert scope.cue == "no"
    assert (scope.cue_start, scope.cue_end) == (0, 2)
    assert scope.direction == "forward"
    # Scope ends where "but" begins; fever sits past the terminator.
    assert scope.scope_start >= scope.cue_end
    assert scope.scope_end == text.index("but")
    assert (chest_pain["start"], chest_pain["end"]) in scope.governed
    assert (fever["start"], fever["end"]) not in scope.governed


def test_forward_cue_scope_stops_at_sentence_break():
    text = "no cough. Fever is present."
    cough = _span(text, "cough")
    fever = _span(text, "Fever")

    scopes = detect_negation_scopes(text, [cough, fever])

    assert len(scopes) == 1
    assert scopes[0].governed == ((cough["start"], cough["end"]),)


def test_backward_cue_scopes_to_the_left():
    text = "Pneumonia was ruled out today"
    pneumonia = _span(text, "Pneumonia")

    scopes = detect_negation_scopes(text, [pneumonia])

    assert len(scopes) == 1
    scope = scopes[0]
    assert scope.cue == "ruled out"
    assert scope.direction == "backward"
    assert scope.scope_end <= scope.cue_start
    assert (pneumonia["start"], pneumonia["end"]) in scope.governed


# --------------------------------------------------------------------------
# Pseudo-negation must not create a scope
# --------------------------------------------------------------------------


def test_pseudo_negation_creates_no_scope():
    text = "Pneumonia cannot be excluded"
    pneumonia = _span(text, "Pneumonia")

    scopes = detect_negation_scopes(text, [pneumonia])

    assert scopes == []


def test_pseudo_negation_not_ruled_out_is_masked():
    text = "Fracture not ruled out on imaging"
    fracture = _span(text, "Fracture")

    scopes = detect_negation_scopes(text, [fracture])

    assert scopes == []


# --------------------------------------------------------------------------
# Entity-level negation assignment (the core over/under-negation fix)
# --------------------------------------------------------------------------


def test_negated_spans_only_returns_in_scope_entities():
    text = "no nausea but reports vomiting"
    nausea = _span(text, "nausea")
    vomiting = _span(text, "vomiting")

    result = negated_spans(text, [nausea, vomiting])

    assert (nausea["start"], nausea["end"]) in result
    assert (vomiting["start"], vomiting["end"]) not in result


def test_entity_after_scope_terminator_is_not_negated():
    text = "denies fever and cough; reports chest pain"
    fever = _span(text, "fever")
    cough = _span(text, "cough")
    chest_pain = _span(text, "chest pain")

    result = negated_spans(text, [fever, cough, chest_pain])

    # "and" terminates the scope, so only fever is negated.
    assert result == ((fever["start"], fever["end"]),)


# --------------------------------------------------------------------------
# Offset exactness / determinism / Unicode stability
# --------------------------------------------------------------------------


def test_comma_separated_list_stays_in_scope():
    text = "no fever, chills, or cough"
    fever = _span(text, "fever")
    chills = _span(text, "chills")
    cough = _span(text, "cough")

    result = negated_spans(text, [fever, chills, cough])

    # Commas do not terminate: fever and chills are negated; "or" ends the run.
    assert (fever["start"], fever["end"]) in result
    assert (chills["start"], chills["end"]) in result
    assert (cough["start"], cough["end"]) not in result


def test_cue_offsets_index_back_to_the_cue_text():
    text = "no evidence of metastatic disease"
    disease = _span(text, "metastatic disease")

    scope = detect_negation_scopes(text, [disease])[0]

    assert text[scope.cue_start : scope.cue_end].lower() == "no evidence of"
    assert (disease["start"], disease["end"]) in scope.governed


def test_offsets_are_stable_under_nfc_normalization():
    raw = "no diseña findings but café noted"
    text = unicodedata.normalize("NFC", raw)
    finding = _span(text, "diseña findings")

    scopes_raw = detect_negation_scopes(text, [finding])
    scopes_again = detect_negation_scopes(unicodedata.normalize("NFC", text), [finding])

    assert scopes_raw == scopes_again
    assert (finding["start"], finding["end"]) in scopes_raw[0].governed


def test_detection_is_deterministic():
    text = "no fever, no chills, but has cough"
    spans = [_span(text, "fever"), _span(text, "chills"), _span(text, "cough")]

    assert detect_negation_scopes(text, spans) == detect_negation_scopes(text, spans)


def test_no_spans_returns_scopes_with_empty_governed():
    text = "no acute distress"
    scopes = detect_negation_scopes(text)
    assert len(scopes) == 1
    assert scopes[0].governed == ()


def test_advisory_is_exposed():
    assert isinstance(NEGATION_SCOPE_ADVISORY, str) and NEGATION_SCOPE_ADVISORY
