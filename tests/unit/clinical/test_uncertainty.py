"""Tests for the ConText uncertainty axis."""

from __future__ import annotations

import pytest

from openmed.clinical import CERTAIN, UNCERTAIN, resolve_uncertainty
from openmed.clinical.context import CERTAINTY_VALUES, UNCERTAINTY_CUES


@pytest.mark.parametrize(
    "text",
    [
        "pneumonia confirmed on CT",
        "hypertension",
        "diabetes mellitus",
        "acute renal failure",
    ],
)
def test_asserted_findings_are_certain(text):
    assert resolve_uncertainty(text) == CERTAIN


@pytest.mark.parametrize(
    "text",
    [
        "possible pneumonia",
        "probable PE",
        "likely pneumonia",
        "unlikely pneumonia",
        "rule out sepsis",
        "r/o sepsis",
        "to rule out MI",
        "cannot exclude lymphoma",
        "can't exclude PE",
        "concern for MI",
        "concerning for malignancy",
        "suspicious for abscess",
        "question of pneumonia",
        "pneumonia vs aspiration",
        "pneumonia versus aspiration",
        "may represent edema",
        "could reflect infection",
        "might indicate ischemia",
        "if fever returns, start abx",
        "should fever return, start abx",
        "unless fever resolves, start abx",
        "in case of chest pain, give nitro",
    ],
)
def test_uncertainty_cue_families_in_span_text(text):
    assert resolve_uncertainty(text) == UNCERTAIN


@pytest.mark.parametrize(
    "modifier",
    [
        "possible",
        "probable",
        "likely",
        "rule out",
        "r/o",
        "cannot exclude",
        "concern for",
        "vs",
        "may",
        "if",
        "unless",
    ],
)
def test_modifier_hits_carry_uncertainty_cues(modifier):
    assert resolve_uncertainty("pneumonia", [modifier]) == UNCERTAIN


def test_issue_examples_are_covered():
    assert resolve_uncertainty("possible pneumonia") == UNCERTAIN
    assert resolve_uncertainty("pneumonia confirmed on CT") == CERTAIN
    assert resolve_uncertainty("rule out sepsis") == UNCERTAIN
    assert resolve_uncertainty("if fever returns, start abx") == UNCERTAIN


def test_cues_matched_case_insensitively():
    assert resolve_uncertainty("Possible pneumonia") == UNCERTAIN
    assert resolve_uncertainty("RULE OUT sepsis") == UNCERTAIN
    assert resolve_uncertainty("pneumonia", ["Probable"]) == UNCERTAIN


def test_span_mapping_is_supported():
    assert resolve_uncertainty({"text": "possible pneumonia"}) == UNCERTAIN
    assert resolve_uncertainty({"text": "hypertension"}) == CERTAIN


class _Span:
    def __init__(self, text: str) -> None:
        self.text = text


def test_span_object_is_supported():
    assert resolve_uncertainty(_Span("possible pneumonia")) == UNCERTAIN
    assert resolve_uncertainty(_Span("hypertension")) == CERTAIN


def test_modifier_hits_as_mappings():
    hits = [{"text": "rule out"}]
    assert resolve_uncertainty({"text": "sepsis"}, hits) == UNCERTAIN


def test_single_modifier_hit_string_is_supported():
    assert resolve_uncertainty("sepsis", "rule out") == UNCERTAIN


def test_uncertainty_cues_do_not_cross_sentence_boundary():
    context = "Rule out sepsis. Sepsis confirmed on culture."
    span = _span_in_context(context, "Sepsis")

    assert resolve_uncertainty(span, ["rule out"]) == CERTAIN


def test_same_sentence_uncertainty_cue_still_applies_with_offsets():
    context = "Rule out sepsis today."
    span = _span_in_context(context, "sepsis")

    assert resolve_uncertainty(span, ["rule out"]) == UNCERTAIN


def test_modifier_hits_do_not_create_cues_across_fragments():
    assert resolve_uncertainty("rule", ["out sepsis"]) == CERTAIN


@pytest.mark.parametrize(
    "text",
    [
        "cardiovascular disease",
        "maybe pneumonia",
        "stiff neck",
        "ruleout typo",
        "suspiciousness alone",
    ],
)
def test_cue_matching_respects_word_boundaries(text):
    assert resolve_uncertainty(text) == CERTAIN


def test_uncertain_spans_are_flagged_not_filtered():
    span = {"text": "possible pneumonia", "start": 4, "end": 22}
    assert resolve_uncertainty(span) == UNCERTAIN
    assert span == {"text": "possible pneumonia", "start": 4, "end": 22}


def test_result_is_always_a_valid_certainty_value():
    for text in ("acute MI", "possible MI", "if chest pain recurs"):
        assert resolve_uncertainty(text) in CERTAINTY_VALUES


def test_required_uncertainty_cues_are_public():
    for cue in (
        "possible",
        "probable",
        "likely",
        "cannot exclude",
        "cannot be excluded",
        "rule out",
        "not ruled out",
        "if",
        "concern for",
        "vs",
    ):
        assert cue in UNCERTAINTY_CUES


def _span_in_context(context: str, target: str) -> dict[str, object]:
    start = context.index(target)
    return {
        "text": target,
        "context": context,
        "start": start,
        "end": start + len(target),
    }
