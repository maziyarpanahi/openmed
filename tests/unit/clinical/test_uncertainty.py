"""Tests for the ConText uncertainty axis (resolve_uncertainty)."""

from __future__ import annotations

from openmed.clinical.context import resolve_uncertainty


# ---------------------------------------------------------------------------
# Certain cases
# ---------------------------------------------------------------------------


def test_confirmed_finding_is_certain():
    assert resolve_uncertainty("pneumonia confirmed on CT", []) == "certain"


def test_plain_span_no_modifiers_is_certain():
    assert resolve_uncertainty("hypertension", []) == "certain"


def test_diagnosed_condition_is_certain():
    assert resolve_uncertainty("diabetes mellitus", []) == "certain"


def test_empty_modifier_hits_no_cue_in_span_is_certain():
    assert resolve_uncertainty("acute renal failure", []) == "certain"


# ---------------------------------------------------------------------------
# Uncertain: inline cues within the span text
# ---------------------------------------------------------------------------


def test_possible_in_span_text():
    assert resolve_uncertainty("possible pneumonia", []) == "uncertain"


def test_rule_out_in_span_text():
    assert resolve_uncertainty("rule out sepsis", []) == "uncertain"


def test_probable_in_span_text():
    assert resolve_uncertainty("probable PE", []) == "uncertain"


def test_likely_in_span_text():
    assert resolve_uncertainty("likely pneumonia", []) == "uncertain"


def test_concern_for_in_span_text():
    assert resolve_uncertainty("concern for MI", []) == "uncertain"


def test_cannot_exclude_in_span_text():
    assert resolve_uncertainty("cannot exclude lymphoma", []) == "uncertain"


def test_vs_in_span_text():
    assert resolve_uncertainty("pneumonia vs aspiration", []) == "uncertain"


def test_unlikely_in_span_text():
    assert resolve_uncertainty("unlikely pneumonia", []) == "uncertain"


def test_ro_abbreviation_in_span_text():
    assert resolve_uncertainty("r/o sepsis", []) == "uncertain"


def test_cant_exclude_in_span_text():
    assert resolve_uncertainty("can't exclude PE", []) == "uncertain"


# ---------------------------------------------------------------------------
# Uncertain: conditional 'if' scopes the concept as uncertain
# ---------------------------------------------------------------------------


def test_conditional_if_scopes_concept_as_uncertain():
    assert resolve_uncertainty("if fever returns, start abx", []) == "uncertain"


# ---------------------------------------------------------------------------
# Uncertain: modifier_hits provided by the ConText engine
# ---------------------------------------------------------------------------


def test_modifier_hit_possible():
    assert resolve_uncertainty("pneumonia", ["possible"]) == "uncertain"


def test_modifier_hit_rule_out():
    assert resolve_uncertainty("sepsis", ["rule out"]) == "uncertain"


def test_modifier_hit_cannot_exclude():
    assert resolve_uncertainty("PE", ["cannot exclude"]) == "uncertain"


def test_modifier_hit_probable():
    assert resolve_uncertainty("MI", ["probable"]) == "uncertain"


def test_modifier_hit_concern_for():
    assert resolve_uncertainty("malignancy", ["concern for"]) == "uncertain"


def test_modifier_hit_if_conditional():
    assert resolve_uncertainty("antibiotics", ["if"]) == "uncertain"


def test_modifier_hit_likely():
    assert resolve_uncertainty("DVT", ["likely"]) == "uncertain"


def test_modifier_hit_vs():
    assert resolve_uncertainty("sepsis", ["vs"]) == "uncertain"


def test_modifier_hit_unlikely():
    assert resolve_uncertainty("pneumonia", ["unlikely"]) == "uncertain"


def test_modifier_hit_ro_abbreviation():
    assert resolve_uncertainty("MI", ["r/o"]) == "uncertain"


# ---------------------------------------------------------------------------
# Cue matching: case-insensitive
# ---------------------------------------------------------------------------


def test_cues_matched_case_insensitively():
    assert resolve_uncertainty("Possible pneumonia", []) == "uncertain"
    assert resolve_uncertainty("RULE OUT sepsis", []) == "uncertain"
    assert resolve_uncertainty("pneumonia", ["Probable"]) == "uncertain"


# ---------------------------------------------------------------------------
# Span as an object with a .text attribute
# ---------------------------------------------------------------------------


class _Span:
    def __init__(self, text: str) -> None:
        self.text = text


def test_span_object_uncertain():
    assert resolve_uncertainty(_Span("possible pneumonia"), []) == "uncertain"


def test_span_object_certain():
    assert resolve_uncertainty(_Span("hypertension"), []) == "certain"


def test_span_object_certain_with_modifier_hit():
    assert resolve_uncertainty(_Span("sepsis"), ["rule out"]) == "uncertain"
