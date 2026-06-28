"""Tests for the sentence-windowed ConText cue scanner."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    HISTORICAL,
    UNCERTAIN,
    resolve_temporality,
    resolve_uncertainty,
    scan_context_cues,
)


def test_scan_context_cues_binds_forward_and_backward_cues_with_offsets() -> None:
    text = "No evidence of pneumonia. Edema was ruled out."

    hits = scan_context_cues(text, ["pneumonia", "Edema"])

    forward_hit = hits["pneumonia"][0]
    assert forward_hit.cue == "No evidence of"
    assert forward_hit.category == "negation"
    assert forward_hit.direction == "forward"
    assert forward_hit.start == text.index("No evidence of")
    assert forward_hit.end == forward_hit.start + len("No evidence of")

    backward_hit = hits["Edema"][0]
    assert backward_hit.cue == "ruled out"
    assert backward_hit.category == "negation"
    assert backward_hit.direction == "backward"
    assert backward_hit.start == text.index("ruled out")
    assert backward_hit.end == backward_hit.start + len("ruled out")


def test_scan_context_cues_does_not_cross_sentence_boundary() -> None:
    text = "No evidence of pneumonia. Effusion is present."

    hits = scan_context_cues(text, ["Effusion"])

    assert hits["Effusion"] == ()


@pytest.mark.parametrize("connector", ["but", "however", "although"])
def test_scan_context_cues_does_not_cross_conjunction_boundary(
    connector: str,
) -> None:
    text = f"Pneumonia {connector} it was ruled out."

    hits = scan_context_cues(text, ["Pneumonia"])

    assert hits["Pneumonia"] == ()


def test_resolve_uncertainty_consumes_scanner_hits() -> None:
    text = "Possible pneumonia remains on the differential."
    span = "pneumonia"

    hits = scan_context_cues(text, [span])

    assert [hit.cue for hit in hits[span]] == ["Possible"]
    assert resolve_uncertainty(span, hits[span]) == UNCERTAIN


def test_scan_context_cues_supports_span_mappings_with_offsets() -> None:
    text = "Possible pneumonia remains on the differential."
    start = text.index("pneumonia")
    span = {"text": "pneumonia", "start": start, "end": start + len("pneumonia")}

    hits = scan_context_cues(text, [span])

    assert span in hits
    assert [hit.cue for hit in hits[span]] == ["Possible"]


def test_resolve_temporality_consumes_scanner_hits() -> None:
    text = "History of MI with no current chest pain."
    span = "MI"

    hits = scan_context_cues(text, [span])

    assert [hit.cue for hit in hits[span]] == ["History of"]
    assert resolve_temporality(span, hits[span]) == HISTORICAL
