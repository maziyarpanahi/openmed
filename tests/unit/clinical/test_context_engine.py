"""Focused tests for the data-driven ConText rule engine."""

from __future__ import annotations

from importlib import resources

import pytest

from openmed.clinical.context import (
    DEFAULT_CONTEXT_RULES_RESOURCE,
    apply_context_rules,
    load_context_rules,
)


def _span(text: str, surface: str) -> dict[str, object]:
    start = text.index(surface)
    return {"text": surface, "start": start, "end": start + len(surface)}


def test_forward_negation_stops_at_conjunction() -> None:
    text = "No fever but cough persists."
    fever = _span(text, "fever")
    cough = _span(text, "cough")

    results = apply_context_rules(text, [fever, cough])

    assert results[0][0] is fever
    assert results[1][0] is cough
    assert len(results[0][1]) == 1
    hit = results[0][1][0]
    assert hit.category == "negation"
    assert hit.matched_cue == "No"
    assert hit.offset == (0, 2)
    assert results[1][1] == []


def test_backward_rule_scopes_preceding_concept() -> None:
    text = "Pneumonia was ruled out."
    pneumonia = _span(text, "Pneumonia")

    [(returned_span, hits)] = apply_context_rules(text, [pneumonia])

    assert returned_span is pneumonia
    assert len(hits) == 1
    assert hits[0].cue == "ruled out"
    assert hits[0].category == "negation"
    assert hits[0].direction == "backward"
    assert hits[0].offset == (
        text.index("ruled out"),
        text.index("ruled out") + len("ruled out"),
    )


def test_rule_scope_does_not_cross_sentence_boundary() -> None:
    text = "No evidence of fever. Pneumonia is present."
    pneumonia = _span(text, "Pneumonia")

    [(returned_span, hits)] = apply_context_rules(text, [pneumonia])

    assert returned_span is pneumonia
    assert hits == []


@pytest.mark.parametrize(
    ("text", "surface"),
    [
        ("Edema absent.", "Edema"),
        ("Absent edema.", "edema"),
    ],
)
def test_bidirectional_rule_scopes_on_either_side(text: str, surface: str) -> None:
    target = _span(text, surface)

    [(_, hits)] = apply_context_rules(text, [target])

    assert [(hit.cue.casefold(), hit.direction) for hit in hits] == [
        ("absent", "bidirectional")
    ]


def test_starter_rule_file_covers_all_axes_and_documents_provenance() -> None:
    rules = load_context_rules()
    rule_text = (
        resources.files("openmed.clinical")
        .joinpath(DEFAULT_CONTEXT_RULES_RESOURCE)
        .read_text(encoding="utf-8")
    )

    assert {rule.category for rule in rules} == {
        "negation",
        "uncertainty",
        "experiencer",
        "temporality",
    }
    assert all(rule.terminators and rule.max_scope_tokens > 0 for rule in rules)
    assert "NegEx" in rule_text
    assert "medspaCy ConText" in rule_text
    assert "Apache-2.0" in rule_text
    assert "under-perform" in rule_text
