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


def test_complete_offset_miss_scores_zero_f1():
    from openmed.eval.ocr_routing import score_offset_projection

    score = score_offset_projection([("findings", 0, 1)], [("findings", 2, 3)])
    assert score.precision == score.recall == score.f1 == 0
    assert score_offset_projection([], []).f1 == 1


def test_untrusted_classifier_fields_do_not_escape_or_enter_reports():
    sentinel = "SYNTHETIC_PRIVATE_INPUT_123"

    class Hostile:
        @property
        def type(self):
            raise ValueError(sentinel)

    fixture = default_ocr_routing_fixtures()[0]
    report = run_ocr_routing_eval([fixture], classifier=lambda text: Hostile())
    assert not report.passed
    assert sentinel not in report.to_json()
    assert report.cases[0].classifier_error == "classifier_error"


def test_arbitrary_classifier_label_and_exception_names_are_not_published():
    sentinel = "SYNTHETIC_PRIVATE_INPUT_123"
    fixture = default_ocr_routing_fixtures()[0]
    report = run_ocr_routing_eval(
        [fixture], classifier=lambda text: {"type": sentinel, "confidence": 1}
    )
    assert sentinel not in report.to_json()
    assert report.cases[0].predicted_document_type == "unknown"

    def broken(text):
        raise type(sentinel, (ValueError,), {})(sentinel)

    report = run_ocr_routing_eval([fixture], classifier=broken, section_detector=broken)
    assert sentinel not in report.to_json()


def test_detector_internal_typeerror_does_not_retry_without_language():
    calls = []

    def detector(text, language="en"):
        calls.append(language)
        raise TypeError("SYNTHETIC_PRIVATE_INPUT_123")

    report = run_ocr_routing_eval(
        [default_ocr_routing_fixtures()[0]], section_detector=detector
    )
    assert calls == ["en"]
    assert report.cases[0].detector_error == "detector_error"


@pytest.mark.parametrize("confidence", [10**1000, 2.0, -1.0])
def test_invalid_classifier_confidence_fails_to_generic(confidence):
    fixture = default_ocr_routing_fixtures()[0]
    report = run_ocr_routing_eval(
        [fixture],
        classifier=lambda text: {"type": "radiology_report", "confidence": confidence},
    )
    assert report.cases[0].classifier_confidence == 0
    assert report.cases[0].predicted_profile == "generic"


def test_fixture_repr_does_not_include_text():
    fixture = OcrRoutingFixture("synthetic", "unknown", "SYNTHETIC_PRIVATE_INPUT_123")
    assert fixture.canonical_text not in repr(fixture)


def test_fixture_fallback_flag_requires_boolean():
    with pytest.raises(ValueError):
        OcrRoutingFixture("synthetic", "unknown", "A", expect_fallback="false")


def test_projection_lengths_reject_boolean_and_boundaries_are_immutable():
    from openmed.eval.ocr_routing import OffsetProjection

    with pytest.raises(ValueError):
        OffsetProjection(True, 1, (0, 1))
    boundaries = [0, 1]
    projection = OffsetProjection(1, 1, boundaries)
    boundaries[1] = 999
    assert projection.project_offset(1) == 1


def test_fixture_and_alignment_work_are_bounded():
    with pytest.raises(ValueError):
        OcrRoutingFixture("synthetic", "unknown", "a" * 4097)
    with pytest.raises(ValueError):
        build_offset_projection("a" * 4097, "b")
    with pytest.raises(ValueError):
        build_offset_projection("a" * 3000, "b" * 3000)


def test_section_iterator_failures_have_no_sensitive_exception_context():
    from openmed.eval.ocr_routing import score_offset_projection

    def broken():
        yield ("findings", 0, 1)
        raise RuntimeError("SYNTHETIC_PRIVATE_INPUT_123")

    with pytest.raises(ValueError) as raised:
        score_offset_projection(broken(), [])
    assert "SYNTHETIC_PRIVATE_INPUT_123" not in str(raised.value)
    assert raised.value.__context__ is None


def test_unbounded_fixture_iterator_stops_at_limit():
    fixture = default_ocr_routing_fixtures()[0]
    visited = []

    def fixtures():
        for index in range(600):
            visited.append(index)
            yield replace(fixture, fixture_id=f"synthetic-{index}")

    with pytest.raises(ValueError):
        run_ocr_routing_eval(fixtures())
    assert len(visited) <= 513


def test_unknown_detector_label_is_hashed_not_copied_to_report():
    sentinel = "SYNTHETIC_PRIVATE_INPUT_123"
    fixture = default_ocr_routing_fixtures()[0]
    report = run_ocr_routing_eval(
        [fixture], section_detector=lambda text: [(sentinel, 0, 1)]
    )
    assert sentinel not in report.to_json()
    assert report.cases[0].projected_sections[0].label.startswith("sha256:")


def test_threshold_conversion_failure_has_safe_context():
    class Hostile:
        def __float__(self):
            raise ValueError("SYNTHETIC_PRIVATE_INPUT_123")

    with pytest.raises(ValueError) as raised:
        run_ocr_routing_eval(min_route_accuracy=Hostile())
    assert "SYNTHETIC_PRIVATE_INPUT_123" not in str(raised.value)
    assert raised.value.__context__ is None
