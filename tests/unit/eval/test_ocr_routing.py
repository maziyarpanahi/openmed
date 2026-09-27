"""Focused offline tests for the synthetic OCR document-routing evaluation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from openmed.eval.clinical_fixtures import generate_fixture
from openmed.eval.ocr_routing import (
    OCR_DOCUMENT_FAMILIES,
    OcrRoutingFixture,
    build_offset_projection,
    default_ocr_routing_fixtures,
    run_ocr_routing_eval,
)
from openmed.multimodal import (
    OcrResult,
    OcrWord,
    normalize_box,
    transform_ocr_result,
)
from openmed.multimodal.layout import parse_layout


def test_default_fixtures_cover_common_families_and_publish_no_source_text() -> None:
    fixtures = default_ocr_routing_fixtures()

    assert {fixture.document_family for fixture in fixtures} == set(
        OCR_DOCUMENT_FAMILIES
    )
    assert len({fixture.fixture_id for fixture in fixtures}) == len(fixtures)
    discharge = next(
        fixture for fixture in fixtures if fixture.fixture_id == "discharge-basic"
    )
    assert discharge.expected_profile == "discharge_summary"
    assert discharge.expect_fallback is False
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


def test_seeded_and_rotated_layout_documents_feed_routing_harness() -> None:
    seeded = generate_fixture("radiology", seed=7)
    rows = (
        ("Radiology", 20, 20, 85, 30),
        ("Report", 91, 20, 133, 30),
        ("Findings:", 30, 120, 112, 132),
        ("Synthetic", 30, 150, 74, 162),
        ("opacity", 80, 150, 110, 162),
        ("Impression:", 280, 120, 345, 132),
        ("Stable", 280, 150, 312, 162),
        ("finding", 318, 150, 355, 162),
        ("Test", 30, 350, 60, 362),
        ("Result", 200, 350, 243, 362),
        ("Reference", 350, 350, 416, 362),
        ("Hemoglobin", 30, 380, 102, 392),
        ("13", 200, 380, 215, 392),
        ("12-16", 350, 380, 386, 392),
        ("Sodium", 30, 410, 76, 422),
        ("140", 200, 410, 222, 422),
        ("135-145", 350, 410, 400, 422),
        ("Page", 30, 650, 59, 662),
        ("1", 65, 650, 72, 662),
    )
    words = tuple(
        OcrWord(text, (x0, y0, x1, y1), 0.4 if text == "opacity" else 0.99)
        for text, x0, y0, x1, y1 in rows
    )
    source = OcrResult(
        words=tuple(reversed(words)),
        metadata={"page_dimensions": {0: (500, 700)}, "source": "synthetic"},
    )
    rotated = transform_ocr_result(source, (500, 700), 90)
    restored = transform_ocr_result(rotated, (700, 500), 270)
    assert [word.bbox for word in restored.words] == [
        word.bbox for word in source.words
    ]
    assert [word.confidence for word in restored.words] == [
        word.confidence for word in source.words
    ]

    normalized = normalize_box(
        (125, 175, 250, 350),
        unit="pixel",
        page_size=(500, 700),
        source_ref="synthetic-region",
    )
    assert normalized.bbox == (0.25, 0.25, 0.5, 0.5)
    assert normalized.source_ref == "synthetic-region"

    layout = parse_layout(restored)
    assert len(layout.columns) == 2
    assert len(layout.tables) == 1
    assert any(word.confidence < 0.5 for word in restored.words)
    assert all(
        layout.offsets_for_bbox(span.page, span.bbox) == (span.offsets,)
        for span in layout.spans
    )
    report = run_ocr_routing_eval(
        (
            OcrRoutingFixture("seeded-radiology", seeded.profile, seeded.text),
            OcrRoutingFixture("layout-radiology", "radiology_report", layout.text),
        )
    )
    assert report.passed is True
    assert report.metrics.fixture_count == 2
    assert report.metrics.route_accuracy == 1.0
    assert report.metrics.offset_projection_accuracy == 1.0


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
