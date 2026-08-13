"""Focused tests for the deterministic lightweight relation scaffold."""

from __future__ import annotations

import json

from openmed.clinical.relations_lite import (
    DRUG_DOSE,
    DRUG_ROUTE,
    FINDING_SEVERITY,
    PROBLEM_ANATOMY,
    extract_relation_candidates,
)


def _span(text: str, surface: str, label: str, **metadata: object) -> dict:
    start = text.index(surface)
    return {
        "start": start,
        "end": start + len(surface),
        "label": label,
        "text": surface,
        **metadata,
    }


def test_golden_fixture_emits_all_supported_pairs_and_rejects_incompatible_edges():
    text = (
        "Aspirin 81 mg orally was continued for knee pain in the left knee. "
        "Severe finding was noted."
    )
    spans = [
        _span(text, "Aspirin", "MEDICATION"),
        _span(text, "81 mg", "DOSAGE"),
        _span(text, "orally", "ROUTE"),
        _span(text, "knee pain", "PROBLEM"),
        _span(text, "left knee", "ANATOMY"),
        _span(text, "finding", "FINDING"),
        _span(text, "Severe", "SEVERITY"),
        # A compatible-looking proximity span with the wrong semantic type
        # must not create a drug/anatomy or problem/dose edge.
        _span(text, "continued", "PROCEDURE"),
    ]

    candidates = extract_relation_candidates(text, spans)

    assert {
        (candidate.relation_type, candidate.head_offset, candidate.tail_offset)
        for candidate in candidates
    } == {
        (
            DRUG_DOSE,
            (0, 7),
            (8, 13),
        ),
        (
            DRUG_ROUTE,
            (0, 7),
            (14, 20),
        ),
        (
            PROBLEM_ANATOMY,
            (39, 48),
            (56, 65),
        ),
        (
            FINDING_SEVERITY,
            (74, 81),
            (67, 73),
        ),
    }


def test_candidates_are_stable_offset_only_records():
    text = "Aspirin 81 mg was continued."
    spans = [_span(text, "81 mg", "DOSE"), _span(text, "Aspirin", "DRUG")]

    forward = extract_relation_candidates(text, spans)
    reversed_input = extract_relation_candidates(text, tuple(reversed(spans)))

    assert forward == reversed_input
    assert len(forward) == 1
    payload = json.dumps([candidate.to_dict() for candidate in forward])
    assert "Aspirin" not in payload
    assert "81 mg" not in payload
    assert {"start", "end", "label"} <= set(forward[0].to_dict()["head"])
    assert forward[0].confidence == forward[0].score


def test_sentence_boundary_requires_an_explicit_connective():
    disconnected = "Aspirin was continued. 81 mg was documented."
    spans = [
        _span(disconnected, "Aspirin", "DRUG"),
        _span(disconnected, "81 mg", "DOSAGE"),
    ]
    assert extract_relation_candidates(disconnected, spans) == ()

    connected = "Aspirin was continued. And 81 mg was documented."
    connected_spans = [
        _span(connected, "Aspirin", "DRUG"),
        _span(connected, "81 mg", "DOSAGE"),
    ]
    candidates = extract_relation_candidates(connected, connected_spans)

    assert len(candidates) == 1
    assert candidates[0].explicit_connective is True
    assert candidates[0].sentence_distance == 1


def test_section_and_assertion_scope_prevent_leaking_edges():
    text = "Assessment:\nPneumonia.\nPlan:\nright lung."
    spans = [
        _span(text, "Pneumonia", "PROBLEM"),
        _span(text, "right lung", "ANATOMY"),
    ]
    assert extract_relation_candidates(text, spans) == ()

    asserted_text = "No pneumonia in the right lung."
    asserted_spans = [
        _span(asserted_text, "pneumonia", "PROBLEM", negation="negated"),
        _span(asserted_text, "right lung", "ANATOMY", negation="affirmed"),
    ]
    assert extract_relation_candidates(asserted_text, asserted_spans) == ()


def test_inverse_argument_order_and_distance_aliases_are_supported():
    text = "Aspirin 81 mg was continued."
    spans = [_span(text, "Aspirin", "DRUG"), _span(text, "81 mg", "DOSAGE")]

    candidates = extract_relation_candidates(
        spans,
        text,
        max_character_distance=5,
        max_tokens=2,
    )

    assert len(candidates) == 1
