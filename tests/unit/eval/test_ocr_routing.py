"""Focused offline tests for the synthetic OCR document-routing evaluation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from openmed.eval.ocr_routing import (
    OCR_DOCUMENT_FAMILIES,
    build_offset_projection,
    default_ocr_routing_fixtures,
    run_ocr_routing_eval,
)


def test_default_fixtures_cover_common_families_and_publish_no_source_text() -> None:
    fixtures = default_ocr_routing_fixtures()

    assert {fixture.document_family for fixture in fixtures} == set(
        OCR_DOCUMENT_FAMILIES
    )
    assert len({fixture.fixture_id for fixture in fixtures}) == len(fixtures)
    for fixture in fixtures:
        public = json.dumps(fixture.to_dict(), sort_keys=True)
        assert fixture.canonical_text not in public
        assert fixture.ocr_text not in public
        assert fixture.to_dict()["synthetic"] is True


def test_offset_projection_handles_insertions_and_replacements() -> None:
    source = "Header: Al pha\nTarget: Zeta value."
    target = "Header: Alpha\nTarget: Zeta value."
    projection = build_offset_projection(source, target)

    source_start = source.index("Target")
    source_end = len(source)
    target_start = target.index("Target")
    target_end = len(target)
    assert projection.project_span(source_start, source_end) == (
        target_start,
        target_end,
    )
    assert projection.project_offset(0) == 0
    assert projection.project_offset(len(source)) == len(target)


def test_default_eval_is_deterministic_and_passes_all_gates() -> None:
    first = run_ocr_routing_eval()
    second = run_ocr_routing_eval()

    assert first.passed is True
    assert first.to_dict() == second.to_dict()
    assert first.metrics.route_accuracy == 1.0
    assert first.metrics.profile_accuracy == 1.0
    assert first.metrics.offset_projection_accuracy == 1.0
    assert first.metrics.safe_fallback_rate == 1.0
    assert first.failures == ()

    serialized = first.to_json()
    markdown = first.to_markdown()
    for fixture in default_ocr_routing_fixtures():
        assert fixture.canonical_text not in serialized
        assert fixture.ocr_text not in serialized
        assert fixture.canonical_text not in markdown
        assert fixture.ocr_text not in markdown


def test_low_confidence_specialized_route_falls_back_without_dropping_sections() -> (
    None
):
    radiology = next(
        fixture
        for fixture in default_ocr_routing_fixtures()
        if fixture.fixture_id == "radiology-basic"
    )
    fallback_fixture = replace(
        radiology,
        expected_profile="generic",
        expect_fallback=True,
    )

    report = run_ocr_routing_eval(
        [fallback_fixture],
        classifier=lambda _text: {
            "type": "radiology_report",
            "confidence": 0.49,
        },
    )

    assert report.passed is True
    case = report.cases[0]
    assert case.predicted_document_type == "radiology_report"
    assert case.predicted_profile == "generic"
    assert case.observed_fallback is True
    assert case.fallback_safe is True
    assert case.offset_projection_correct is True


def test_failure_diagnostics_do_not_echo_fixture_text() -> None:
    fixture = default_ocr_routing_fixtures()[0]
    report = run_ocr_routing_eval(
        [fixture],
        classifier=lambda _text: {"type": "unknown", "confidence": 0.0},
    )

    assert report.passed is False
    assert report.failures
    diagnostics = json.dumps(report.to_dict(), sort_keys=True)
    assert fixture.canonical_text not in diagnostics
    assert fixture.ocr_text not in diagnostics
    with pytest.raises(AssertionError) as exc_info:
        from openmed.eval.ocr_routing import assert_ocr_routing_gate

        assert_ocr_routing_gate(
            [fixture],
            classifier=lambda _text: {"type": "unknown", "confidence": 0.0},
        )
    assert fixture.canonical_text not in str(exc_info.value)
    assert fixture.ocr_text not in str(exc_info.value)
