"""Unit tests for the synthetic burned-in-PHI DICOM benchmark (OM-162).

The suite depends on ``pydicom`` and ``Pillow`` (the ``multimodal`` extra). The
tests skip cleanly when the extra is not installed, and every generated image
carries only synthetic PHI.
"""

from __future__ import annotations

import json

import pytest

# The whole suite needs the optional multimodal extra; skip cleanly otherwise.
pytest.importorskip("pydicom")
pytest.importorskip("PIL")
pytest.importorskip("numpy")

from openmed.core.labels import DATE, ID_NUM, PERSON  # noqa: E402
from openmed.eval.report import BenchmarkReport  # noqa: E402
from openmed.eval.suites import (  # noqa: E402
    MULTIMODAL_DICOM,
    load_suite_fixtures,
    suite_metadata,
)
from openmed.eval.suites.multimodal_dicom import (  # noqa: E402
    CORPUS_LICENSE,
    GoldBoxOcrEngine,
    SyntheticDicomCase,
    _offline_pixel_model,
    _word_is_covered,
    generate_synthetic_dicom_corpus,
    load_multimodal_dicom_fixtures,
    multimodal_dicom_metadata,
    run_multimodal_dicom,
)


def test_suite_is_registered() -> None:
    from openmed.eval.suites import DEFAULT_SUITES, validate_suite_name

    assert MULTIMODAL_DICOM in DEFAULT_SUITES
    assert validate_suite_name(MULTIMODAL_DICOM) == MULTIMODAL_DICOM


def test_generator_is_deterministic_with_span_and_bbox_truth(tmp_path) -> None:
    cases_a = generate_synthetic_dicom_corpus(tmp_path / "a", seed=4242, corpus_size=3)
    cases_b = generate_synthetic_dicom_corpus(tmp_path / "b", seed=4242, corpus_size=3)

    assert len(cases_a) == 3
    assert all(isinstance(case, SyntheticDicomCase) for case in cases_a)

    # Deterministic in the seed: same PHI, same layout, same gold truth.
    for left, right in zip(cases_a, cases_b):
        assert left.header_phi == right.header_phi
        assert left.pixel_text == right.pixel_text
        assert [w.bbox for w in left.pixel_words] == [w.bbox for w in right.pixel_words]

    case = cases_a[0]
    # Header PHI is present on disk and burned-in words carry gold spans+bboxes.
    assert case.header_phi["PatientID"].startswith("MRN-")
    assert case.pixel_words, "expected burned-in PHI words"
    labels = {word.label for word in case.pixel_words}
    assert labels <= {PERSON, ID_NUM, DATE}
    assert PERSON in labels and ID_NUM in labels and DATE in labels

    # Every gold span points at the right substring of the pixel text.
    for span in case.pixel_gold_spans:
        assert case.pixel_text[span.start : span.end] == span.text
        assert span.label in {PERSON, ID_NUM, DATE}

    # Each word has a valid, non-empty pixel bbox.
    for word in case.pixel_words:
        x0, y0, x1, y1 = word.bbox
        assert x0 < x1 and y0 < y1


def test_generator_writes_readable_dicoms_with_burned_in_pixels(tmp_path) -> None:
    import numpy as np
    import pydicom

    case = generate_synthetic_dicom_corpus(tmp_path, seed=7, corpus_size=1)[0]
    dataset = pydicom.dcmread(case.path)

    # Header PHI landed on the dataset.
    assert str(dataset.PatientName)
    assert dataset.PatientID == case.header_phi["PatientID"]
    assert dataset.BurnedInAnnotation == "YES"

    # The pixels actually carry inked (non-zero) text inside every gold bbox.
    pixels = np.asarray(dataset.pixel_array)
    for word in case.pixel_words:
        x0, y0, x1, y1 = word.bbox
        assert pixels[y0:y1, x0:x1].max() > 0


def test_run_scores_clean_redaction_with_leakage_first_metrics(tmp_path) -> None:
    report = run_multimodal_dicom(
        output_dir=tmp_path, seed=123, corpus_size=4, model_name="unit-test-system"
    )

    assert isinstance(report, BenchmarkReport)
    assert report.suite == MULTIMODAL_DICOM
    assert report.fixture_count == 4

    metrics = report.metrics
    # The redaction path fully clears burned-in and header PHI on the happy path.
    assert metrics["residual_phi_rate"] == 0.0
    assert metrics["leakage"]["overall"] == 0.0
    assert metrics["character_recall"]["rate"] == 1.0
    assert metrics["recall_slices"]["overall"] == 1.0
    assert metrics["header_residual_phi_rate"] == 0.0
    assert metrics["header_direct_identifier_count"] > 0
    assert metrics["header_residual_identifier_count"] == 0

    # Per-label recall is reported for every burned-in PHI class.
    recall_by_label = metrics["recall_slices"]["by_label"]
    for label in (PERSON, ID_NUM, DATE):
        assert recall_by_label[label] == 1.0

    # Provenance / license is documented and synthetic.
    assert report.metadata["license"] == CORPUS_LICENSE
    assert report.metadata["tcia_sampled"] is False
    assert report.metadata["requires_data_use_agreement"] is False


def test_run_report_never_contains_raw_phi(tmp_path) -> None:
    cases = generate_synthetic_dicom_corpus(tmp_path, seed=555, corpus_size=3)
    report = run_multimodal_dicom(cases=cases, model_name="unit-test-system")

    serialized = json.dumps(report.to_dict(), sort_keys=True)
    serialized += report.to_markdown()
    for case in cases:
        for word in case.pixel_words:
            assert word.text not in serialized
        assert case.header_phi["PatientID"] not in serialized
        assert str(case.header_phi["PatientName"]) not in serialized


def test_report_round_trips_through_report_format(tmp_path) -> None:
    report = run_multimodal_dicom(output_dir=tmp_path, seed=1, corpus_size=2)
    restored = BenchmarkReport.from_dict(report.to_dict())

    assert restored.suite == report.suite
    assert restored.fixture_count == report.fixture_count
    assert restored.metrics["residual_phi_rate"] == report.metrics["residual_phi_rate"]
    # Report renders to the harness's Markdown/JSON surfaces.
    assert "## Metrics" in report.to_markdown()
    assert json.loads(report.to_json())["suite"] == MULTIMODAL_DICOM


def test_benchmark_detects_leakage_under_partial_redaction(tmp_path) -> None:
    from openmed.eval.metrics import (
        EvalSpan,
        compute_character_recall,
        compute_leakage_rate,
    )
    from openmed.multimodal import redact_dicom_pixels

    case = generate_synthetic_dicom_corpus(tmp_path, seed=321, corpus_size=1)[0]
    # Drop the MRN word so the detector misses it -> that PHI must leak.
    missed = next(word for word in case.pixel_words if word.label == ID_NUM)
    partial = tuple(word for word in case.pixel_words if word is not missed)

    with _offline_pixel_model():
        result = redact_dicom_pixels(
            case.path,
            output_path=tmp_path / "partial.dcm",
            ocr_engine=GoldBoxOcrEngine(partial),
            model_name=None,
            verify_residual=True,
            fail_on_residual=False,
        )

    detected = [
        (finding.frame_index, tuple(int(v) for v in finding.bbox))
        for finding in result.findings
    ]
    predicted = []
    cursor = 0
    for word in case.pixel_words:
        start = case.pixel_text.index(word.text, cursor)
        cursor = start + len(word.text)
        if _word_is_covered(word, detected):
            predicted.append(
                EvalSpan(start, start + len(word.text), word.label, word.text)
            )

    leakage = compute_leakage_rate(
        list(case.pixel_gold_spans), predicted, source_text=case.pixel_text
    )
    recall = compute_character_recall(
        list(case.pixel_gold_spans), predicted, source_text=case.pixel_text
    )

    assert leakage.overall > 0.0
    assert recall.rate < 1.0
    # The whole MRN string leaked.
    assert leakage.leaked_chars_by_label[ID_NUM] == len(missed.text)


def test_load_fixtures_via_registry_and_direct(tmp_path) -> None:
    fixtures = load_multimodal_dicom_fixtures(
        output_dir=tmp_path, seed=88, corpus_size=2
    )
    assert len(fixtures) == 2
    first = fixtures[0]
    assert first.text
    assert first.gold_spans
    assert first.metadata["license"] == CORPUS_LICENSE
    assert "dicom_path" in first.metadata

    # The suite registry can load fixtures and metadata by name.
    registry_fixtures = load_suite_fixtures(
        MULTIMODAL_DICOM, output_dir=tmp_path / "reg", seed=88, corpus_size=2
    )
    assert len(registry_fixtures) == 2

    metadata = suite_metadata(MULTIMODAL_DICOM, seed=88, corpus_size=2)
    assert metadata["suite"] == MULTIMODAL_DICOM
    assert metadata == multimodal_dicom_metadata(seed=88, corpus_size=2)


def test_generate_rejects_non_positive_corpus_size(tmp_path) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_dicom_corpus(tmp_path, corpus_size=0)
