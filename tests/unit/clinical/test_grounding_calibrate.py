"""Focused tests for grounding confidence calibration and review reports."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.grounding import Candidate, GroundedSpan
from openmed.clinical.grounding.calibrate import (
    ACCEPT_BAND,
    UNCERTAIN_BAND,
    apply_grounding_calibration,
    fit_grounding_confidence_calibrator,
)
from openmed.clinical.grounding.review_report import build_review_report


def _calibrator():
    rows = [
        {"system": "RXNORM", "score": 0.10, "correct": False},
        {"system": "RXNORM", "score": 0.25, "correct": False},
        {"system": "RXNORM", "score": 0.75, "correct": True},
        {"system": "RXNORM", "score": 0.95, "correct": True},
        {"system": "LOINC", "score": 0.10, "correct": False},
        {"system": "LOINC", "score": 0.40, "correct": True},
        {"system": "LOINC", "score": 0.90, "correct": True},
    ]
    return fit_grounding_confidence_calibrator(rows, threshold=0.70)


def test_per_vocabulary_confidence_is_bounded_and_monotonic() -> None:
    calibrator = _calibrator()

    confidences = tuple(
        calibrator.predict("RXNORM", score) for score in (0.10, 0.25, 0.50, 0.75, 0.95)
    )

    assert all(0.0 <= confidence <= 1.0 for confidence in confidences)
    assert confidences == tuple(sorted(confidences))
    assert calibrator.predict("RXNORM", 0.75) > calibrator.predict("LOINC", 0.10)


def test_low_score_link_carries_an_uncertain_band() -> None:
    calibrator = _calibrator()
    span = GroundedSpan(
        text="mzx-alpha",
        start=12,
        end=21,
        candidates=(Candidate("RXNORM", "SYN-001", "Synthetic medicine", 0.25),),
    )

    calibrated = apply_grounding_calibration(span, calibrator)

    assert calibrated.calibrated_confidence == pytest.approx(0.0)
    assert calibrated.calibrated_score == calibrated.calibrated_confidence
    assert calibrated.confidence_band == UNCERTAIN_BAND
    assert calibrated.band == UNCERTAIN_BAND
    assert calibrated.abstained is False
    assert calibrated.provenance["grounding_calibration"]["band"] == UNCERTAIN_BAND


def test_review_report_pairs_each_code_with_source_offsets_and_serializes() -> None:
    calibrator = _calibrator()
    span = GroundedSpan(
        text="mzx-beta",
        start=30,
        end=38,
        candidates=(Candidate("RXNORM", "SYN-002", "Synthetic medicine", 0.95),),
    )
    calibrated = apply_grounding_calibration(span, calibrator)

    report = build_review_report(
        [calibrated],
        generated_at="2026-08-09T00:00:00Z",
    )

    assert len(report.entries) == 1
    entry = report.entries[0]
    assert entry.span_text == "mzx-beta"
    assert (entry.start, entry.end) == (30, 38)
    assert entry.system == "RXNORM"
    assert entry.system_uri == "http://www.nlm.nih.gov/research/umls/rxnorm"
    assert entry.code == "SYN-002"
    assert entry.raw_score == pytest.approx(0.95)
    assert entry.calibrated_confidence == pytest.approx(1.0)
    assert entry.band == ACCEPT_BAND
    assert report.reverse_index[(entry.system_uri, entry.code)] == ((30, 38),)

    payload = json.loads(report.to_json())
    assert payload["artifact_type"] == "openmed.grounding.review_report"
    assert payload["entries"][0]["offsets"] == [30, 38]
    assert payload["entries"][0]["calibrated_confidence"] == pytest.approx(1.0)
    markdown = report.to_markdown()
    assert "Grounding Review Report" in markdown
    assert "mzx-beta" in markdown
    assert "accept" in markdown
