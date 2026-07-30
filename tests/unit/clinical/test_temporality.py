"""Tests for the ConText temporality axis (OM-141)."""

import pytest

from openmed.clinical import resolve_temporality
from openmed.clinical.context import (
    HISTORICAL,
    HYPOTHETICAL,
    RECENT,
    TEMPORALITY_VALUES,
)


def test_history_of_is_historical():
    assert resolve_temporality("history of MI") == "historical"


def test_acute_is_recent_by_default():
    assert resolve_temporality("acute MI") == "recent"


def test_conditional_is_hypothetical():
    assert resolve_temporality("if chest pain recurs") == "hypothetical"


@pytest.mark.parametrize(
    "text",
    [
        "h/o myocardial infarction",
        "hx of stroke",
        "status post CABG",
        "s/p appendectomy",
        "previous pneumonia",
        "pneumonia resolved",
        "fracture in the past",
        "prior DVT",
    ],
)
def test_historical_cue_families(text):
    assert resolve_temporality(text) == HISTORICAL


@pytest.mark.parametrize(
    "text",
    [
        "should fever develop",
        "in case of bleeding",
        "in the event of shock",
        "unless symptoms worsen",
    ],
)
def test_hypothetical_cue_families(text):
    assert resolve_temporality(text) == HYPOTHETICAL


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("history\nof MI", HISTORICAL),
        ("status\tpost CABG", HISTORICAL),
        ("in case\nof bleeding", HYPOTHETICAL),
    ],
)
def test_multiword_cues_tolerate_clinical_whitespace(text, expected):
    assert resolve_temporality(text) == expected


def test_default_is_recent_without_cues():
    assert resolve_temporality("chest pain") == RECENT
    assert resolve_temporality("acute MI", []) == RECENT


def test_modifier_hits_carry_the_cue():
    # The target span is just the concept; the cue arrives via the shared
    # ConText engine's modifier hits rather than being embedded in the span.
    assert resolve_temporality("MI", ["history of"]) == HISTORICAL
    assert resolve_temporality("chest pain", ["if"]) == HYPOTHETICAL


def test_modifier_hits_as_mappings():
    hits = [{"text": "status post"}]
    assert resolve_temporality({"text": "CABG"}, hits) == HISTORICAL


def test_modifier_hits_do_not_create_cues_across_fragments():
    assert resolve_temporality("in", ["case of bleeding"]) == RECENT


def test_span_mapping_is_supported():
    assert resolve_temporality({"text": "history of MI"}) == HISTORICAL


def test_hypothetical_takes_precedence_over_historical():
    # A conditional clause is not asserted to have occurred, so it wins.
    assert resolve_temporality("if history of MI recurs") == HYPOTHETICAL


@pytest.mark.parametrize(
    ("context", "target", "cue"),
    [
        (
            "History of MI. Patient has acute chest pain.",
            "chest pain",
            "history of",
        ),
        (
            "If chest pain recurs, call the clinic. Chest pain is active now.",
            "Chest pain",
            "if",
        ),
    ],
)
def test_temporal_cues_do_not_cross_sentence_boundary(context, target, cue):
    span = _span_in_context(context, target)

    assert resolve_temporality(span, [cue]) == RECENT


def test_temporal_cues_do_not_cross_clause_boundary():
    context = "History of asthma but pneumonia is present on exam."
    span = _span_in_context(context, "pneumonia")

    assert resolve_temporality(span, ["history of"]) == RECENT


def test_nested_temporal_cues_in_scope_use_most_specific_axis():
    context = "If history of migraine returns, call the clinic."
    span = _span_in_context(context, "migraine")
    hits = [
        _hit_in_context(context, "history of"),
        _hit_in_context(context, "If"),
    ]

    assert resolve_temporality(span, hits) == HYPOTHETICAL
    assert resolve_temporality(span, list(reversed(hits))) == HYPOTHETICAL


def test_cue_matching_respects_word_boundaries():
    # "if" must not fire inside "stiff"; "prior" stays a whole-word cue.
    assert resolve_temporality("stiff neck") == RECENT


def test_result_is_always_a_valid_temporality_value():
    for text in ("acute MI", "history of MI", "if chest pain recurs"):
        assert resolve_temporality(text) in TEMPORALITY_VALUES


def _span_in_context(context: str, target: str) -> dict[str, object]:
    start = context.index(target)
    return {
        "text": target,
        "context": context,
        "start": start,
        "end": start + len(target),
    }


def _hit_in_context(context: str, cue: str) -> dict[str, object]:
    start = context.index(cue)
    return {"text": cue, "start": start, "end": start + len(cue)}
