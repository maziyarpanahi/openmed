"""Tests for the ConText negation axis."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    AFFIRMED,
    NEGATED,
    ClinicalContextResult,
    resolve_negation,
    resolve_span_context,
)
from openmed.clinical.context import (
    NEGATION_CUES,
    NEGATION_VALUES,
    PSEUDO_NEGATION_CUES,
)


@pytest.mark.parametrize(
    "text",
    [
        "pneumonia",
        "pneumonia confirmed on CT",
        "acute renal failure",
    ],
)
def test_asserted_findings_are_affirmed(text):
    assert resolve_negation(text) == AFFIRMED


@pytest.mark.parametrize(
    "text",
    [
        "no evidence of pneumonia",
        "no signs of pneumonia",
        "negative for pneumonia",
        "patient denies chest pain",
        "without edema",
        "pneumonia ruled out",
        "absent breath sounds",
    ],
)
def test_true_negation_cues_negate_span_text(text):
    assert resolve_negation(text) == NEGATED


@pytest.mark.parametrize(
    "modifier",
    [
        "no evidence of",
        "negative for",
        "denies",
        "without",
        "ruled out",
    ],
)
def test_modifier_hits_carry_negation_cues(modifier):
    assert resolve_negation("pneumonia", [modifier]) == NEGATED


@pytest.mark.parametrize(
    "text",
    [
        "not ruled out pneumonia",
        "pneumonia cannot be excluded",
        "no increase in pulmonary edema",
        "no significant increase in pneumonia",
    ],
)
def test_pseudo_negation_cues_do_not_negate(text):
    assert resolve_negation(text) == AFFIRMED


def test_double_negation_is_deterministic():
    assert resolve_negation("not without pneumonia") == AFFIRMED
    assert resolve_negation("no evidence of pneumonia without edema") == AFFIRMED


def test_modifier_hits_as_mappings():
    hits = [{"text": "no evidence of"}]
    assert resolve_negation({"text": "sepsis"}, hits) == NEGATED


def test_single_modifier_hit_string_is_supported():
    assert resolve_negation("sepsis", "no evidence of") == NEGATED


def test_negation_cues_match_case_insensitively():
    assert resolve_negation("NO EVIDENCE OF pneumonia") == NEGATED
    assert resolve_negation("Pneumonia Cannot Be Excluded") == AFFIRMED


def test_modifier_hits_do_not_create_cues_across_fragments():
    assert resolve_negation("n", ["o evidence of pneumonia"]) == AFFIRMED


@pytest.mark.parametrize(
    "text",
    [
        "notable pneumonia",
        "nonelective surgery",
        "deniesing typo",
        "nose pain",
    ],
)
def test_short_negation_cues_respect_word_boundaries(text):
    assert resolve_negation(text) == AFFIRMED


def test_span_context_result_exposes_negation_field():
    context = resolve_span_context("no evidence of pneumonia")

    assert context.negation == NEGATED
    assert context.temporality == "recent"
    assert context.certainty == "certain"


def test_pseudo_negation_is_affirmed_in_span_context_result():
    context = resolve_span_context("pneumonia cannot be excluded")

    assert context.negation == AFFIRMED


def test_negated_maps_to_refuted_documented_on_context_result():
    assert "verificationStatus=refuted" in (ClinicalContextResult.__doc__ or "")


def test_result_is_always_a_valid_negation_value():
    for text in ("pneumonia", "no evidence of pneumonia", "not ruled out pneumonia"):
        assert resolve_negation(text) in NEGATION_VALUES


def test_required_negation_cues_are_public():
    for cue in (
        "no evidence of",
        "negative for",
        "denies",
        "without",
        "ruled out",
    ):
        assert cue in NEGATION_CUES


def test_required_pseudo_negation_cues_are_public():
    for cue in ("no increase", "not ruled out", "cannot be excluded"):
        assert cue in PSEUDO_NEGATION_CUES
