"""Unit tests for the synthetic burned-in-PHI DICOM benchmark (OM-162).

The suite depends on ``pydicom`` and ``Pillow`` (the ``multimodal`` extra). The
tests skip cleanly when the extra is not installed, and every generated image
carries only synthetic PHI.
"""

from __future__ import annotations

import hashlib
import json
import socket
import types
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

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
    DEFAULT_SYSTEM_NAME,
    SyntheticDicomCase,
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
    import pydicom

    cases_a = generate_synthetic_dicom_corpus(tmp_path / "a", seed=4242, corpus_size=3)
    cases_b = generate_synthetic_dicom_corpus(tmp_path / "b", seed=4242, corpus_size=3)

    assert len(cases_a) == 3
    assert all(isinstance(case, SyntheticDicomCase) for case in cases_a)

    # Deterministic in the seed: same PHI, same layout, same gold truth.
    for left, right in zip(cases_a, cases_b):
        assert left.header_phi == right.header_phi
        assert left.pixel_text == right.pixel_text
        assert [w.bbox for w in left.pixel_words] == [w.bbox for w in right.pixel_words]
        assert (
            left.metadata["source_dicom_sha256"]
            == right.metadata["source_dicom_sha256"]
        )
        assert (
            hashlib.sha256(left.path.read_bytes()).digest()
            == hashlib.sha256(right.path.read_bytes()).digest()
        )

        left_dataset = pydicom.dcmread(left.path)
        right_dataset = pydicom.dcmread(right.path)
        for keyword in (
            "SOPInstanceUID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
        ):
            assert getattr(left_dataset, keyword) == getattr(right_dataset, keyword)
        assert (
            left_dataset.file_meta.ImplementationClassUID
            == right_dataset.file_meta.ImplementationClassUID
        )

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


def test_final_artifact_hashes_and_report_are_deterministic(tmp_path) -> None:
    report_a = run_multimodal_dicom(
        output_dir=tmp_path / "a",
        seed=9191,
        corpus_size=2,
        generated_at="2026-01-01T00:00:00Z",
    )
    report_b = run_multimodal_dicom(
        output_dir=tmp_path / "b",
        seed=9191,
        corpus_size=2,
        generated_at="2026-01-01T00:00:00Z",
    )

    assert (
        report_a.metadata["final_artifact_sha256"]
        == report_b.metadata["final_artifact_sha256"]
    )
    assert report_a.to_dict() == report_b.to_dict()


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
        output_dir=tmp_path,
        seed=123,
        corpus_size=4,
        system_name="unit-test-system",
    )

    assert isinstance(report, BenchmarkReport)
    assert report.suite == MULTIMODAL_DICOM
    assert report.fixture_count == 4
    assert report.model_name == "unit-test-system"
    assert report.metadata["pii_model_name"].startswith("OpenMed/")
    assert report.metadata["pii_model_name"] != report.model_name

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


def test_default_benchmark_path_makes_no_network_connection(
    tmp_path, monkeypatch
) -> None:
    def reject_network(*_args, **_kwargs):
        raise AssertionError("benchmark attempted an outbound network connection")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    report = run_multimodal_dicom(output_dir=tmp_path, seed=404, corpus_size=1)

    assert report.metadata["offline"] is True
    assert report.metrics["residual_phi_rate"] == 0.0


def test_run_report_never_contains_raw_phi(tmp_path) -> None:
    cases = generate_synthetic_dicom_corpus(tmp_path, seed=555, corpus_size=3)
    sensitive_case_id = cases[0].header_phi["PatientID"]
    cases[0] = replace(cases[0], case_id=sensitive_case_id)
    report = run_multimodal_dicom(cases=cases, system_name="unit-test-system")

    serialized = json.dumps(report.to_dict(), sort_keys=True)
    serialized += report.to_markdown()
    for case in cases:
        for word in case.pixel_words:
            assert word.text not in serialized
        for value in case.header_phi.values():
            assert str(value) not in serialized
    assert "case_ids" not in report.metadata
    assert (
        report.metadata["case_id_sha256"][0]
        == hashlib.sha256(sensitive_case_id.encode()).hexdigest()
    )


def test_run_scores_one_combined_pixel_and_header_artifact(tmp_path) -> None:
    import numpy as np
    import pydicom

    case = generate_synthetic_dicom_corpus(tmp_path, seed=77, corpus_size=1)[0]
    source_dataset = pydicom.dcmread(case.path)
    source_uids = {
        keyword: str(getattr(source_dataset, keyword))
        for keyword in ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID")
    }
    assert any(element.tag.is_private for element in source_dataset.iterall())
    report = run_multimodal_dicom(cases=[case])

    final_path = case.path.with_name(f"{case.path.stem}.redacted.dcm")
    pixel_stage = case.path.with_name(f"{case.path.stem}.pixel-stage.dcm")
    assert final_path.exists()
    assert not pixel_stage.exists()

    final_dataset = pydicom.dcmread(final_path)
    for keyword, original in case.header_phi.items():
        assert str(original) not in str(getattr(final_dataset, keyword, ""))

    pixels = np.asarray(final_dataset.pixel_array)
    for word in case.pixel_words:
        x0, y0, x1, y1 = word.bbox
        assert not np.any(pixels[y0:y1, x0:x1])

    assert not any(element.tag.is_private for element in final_dataset.iterall())
    assert (
        final_dataset.SOPInstanceUID
        == final_dataset.file_meta.MediaStorageSOPInstanceUID
    )
    for keyword, original_uid in source_uids.items():
        assert str(getattr(final_dataset, keyword)) != original_uid
    expected_study_date = (
        datetime.strptime(case.header_phi["StudyDate"], "%Y%m%d") + timedelta(days=17)
    ).strftime("%Y%m%d")
    assert str(final_dataset.StudyDate) == expected_study_date
    assert str(final_dataset.PatientBirthDate) == ""

    assert report.metadata["artifact_pipeline"] == (
        "pixel-redaction-then-header-deidentification"
    )
    assert report.metadata["final_artifact_sha256"] == [
        hashlib.sha256(final_path.read_bytes()).hexdigest()
    ]


def test_report_round_trips_through_report_format(tmp_path) -> None:
    report = run_multimodal_dicom(output_dir=tmp_path, seed=1, corpus_size=2)
    restored = BenchmarkReport.from_dict(report.to_dict())

    assert restored.suite == report.suite
    assert restored.fixture_count == report.fixture_count
    assert restored.metrics["residual_phi_rate"] == report.metrics["residual_phi_rate"]
    # Report renders to the harness's Markdown/JSON surfaces.
    assert "## Metrics" in report.to_markdown()
    assert json.loads(report.to_json())["suite"] == MULTIMODAL_DICOM


def test_header_score_finds_phi_anywhere_when_header_stage_regresses(
    tmp_path, monkeypatch
) -> None:
    import shutil

    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    import openmed.multimodal as multimodal

    case = generate_synthetic_dicom_corpus(tmp_path, seed=707, corpus_size=1)[0]
    dataset = pydicom.dcmread(case.path)
    referenced_study = Dataset()
    referenced_study.StudyDescription = case.header_phi["PatientID"]
    dataset.ReferencedStudySequence = Sequence([referenced_study])
    dataset.save_as(case.path, enforce_file_format=True)
    case = _refresh_source_hash(case)

    def leaky_header_stage(path, *, policy):
        shutil.copyfile(path, policy.output_path)

    monkeypatch.setattr(multimodal, "deidentify_dicom_headers", leaky_header_stage)

    report = run_multimodal_dicom(cases=[case])
    final_path = case.path.with_name(f"{case.path.stem}.redacted.dcm")
    final_dataset = pydicom.dcmread(final_path)

    assert (
        str(final_dataset.ReferencedStudySequence[0].StudyDescription)
        == case.header_phi["PatientID"]
    )
    assert any(element.tag.is_private for element in final_dataset.iterall())
    assert report.metrics["header_residual_phi_rate"] == 1.0
    assert report.metrics["header_residual_identifier_count"] == 4
    assert report.metrics["pixel_residual_phi_rate"] == 0.0
    assert report.metrics["residual_phi_rate"] > 0.0


def test_compressed_rle_pixel_data_is_normalized_and_redacted(tmp_path) -> None:
    import numpy as np
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, RLELossless

    case = generate_synthetic_dicom_corpus(tmp_path, seed=808, corpus_size=1)[0]
    dataset = pydicom.dcmread(case.path)
    dataset.compress(RLELossless)
    dataset.save_as(case.path, enforce_file_format=True)
    case = _refresh_source_hash(case)

    report = run_multimodal_dicom(cases=[case])
    final_path = case.path.with_name(f"{case.path.stem}.redacted.dcm")
    final_dataset = pydicom.dcmread(final_path)

    assert final_dataset.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert report.metrics["residual_phi_rate"] == 0.0
    pixels = np.asarray(final_dataset.pixel_array)
    for word in case.pixel_words:
        x0, y0, x1, y1 = word.bbox
        assert not np.any(pixels[y0:y1, x0:x1])


def test_implicit_vr_little_endian_is_preserved_and_redacted(tmp_path) -> None:
    import pydicom
    from pydicom.uid import ImplicitVRLittleEndian

    case = generate_synthetic_dicom_corpus(tmp_path, seed=818, corpus_size=1)[0]
    dataset = pydicom.dcmread(case.path)
    dataset.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    pydicom.dcmwrite(
        case.path,
        dataset,
        implicit_vr=True,
        little_endian=True,
        enforce_file_format=True,
    )
    case = _refresh_source_hash(case)

    report = run_multimodal_dicom(cases=[case])
    final_path = case.path.with_name(f"{case.path.stem}.redacted.dcm")
    final_dataset = pydicom.dcmread(final_path)

    assert final_dataset.file_meta.TransferSyntaxUID == ImplicitVRLittleEndian
    assert report.metrics["residual_phi_rate"] == 0.0


def test_malformed_or_tampered_cases_fail_without_output_artifacts(tmp_path) -> None:
    case = generate_synthetic_dicom_corpus(tmp_path, seed=909, corpus_size=1)[0]
    case.path.write_bytes(b"not-a-dicom")
    malformed = _refresh_source_hash(case)

    with pytest.raises(ValueError, match="readable DICOM"):
        run_multimodal_dicom(cases=[malformed])

    assert not case.path.with_name(f"{case.path.stem}.pixel-stage.dcm").exists()
    assert not case.path.with_name(f"{case.path.stem}.redacted.dcm").exists()

    valid = generate_synthetic_dicom_corpus(
        tmp_path / "valid", seed=910, corpus_size=1
    )[0]
    valid.path.write_bytes(valid.path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash does not match truth"):
        run_multimodal_dicom(cases=[valid])


def test_supplied_cases_must_retain_synthetic_provenance(tmp_path) -> None:
    case = generate_synthetic_dicom_corpus(tmp_path, seed=1001, corpus_size=1)[0]
    metadata = dict(case.metadata)
    metadata["license"] = "unverified-external-data"

    with pytest.raises(ValueError, match="required synthetic provenance"):
        run_multimodal_dicom(cases=[replace(case, metadata=metadata)])


def test_partial_bbox_overlap_does_not_count_as_fully_redacted(
    tmp_path, monkeypatch
) -> None:
    import numpy as np
    import pydicom

    import openmed.multimodal as multimodal

    case = generate_synthetic_dicom_corpus(tmp_path, seed=321, corpus_size=1)[0]
    target = next(word for word in case.pixel_words if word.label == ID_NUM)
    received_models = []

    def partial_redact(path, *, output_path, model_name, **_kwargs):
        received_models.append(model_name)
        dataset = pydicom.dcmread(path)
        pixels = np.array(dataset.pixel_array, copy=True)
        for word in case.pixel_words:
            x0, y0, x1, y1 = word.bbox
            if word == target:
                # A sliver overlaps the gold box, but most PHI ink remains.
                pixels[y0:y1, x0 : x0 + 1] = 0
            else:
                pixels[y0:y1, x0:x1] = 0
        dataset.PixelData = np.ascontiguousarray(pixels).tobytes()
        dataset.save_as(output_path, enforce_file_format=True)
        return types.SimpleNamespace(
            findings=(
                types.SimpleNamespace(
                    frame_index=target.frame_index,
                    bbox=(
                        target.bbox[0],
                        target.bbox[1],
                        target.bbox[0] + 1,
                        target.bbox[3],
                    ),
                ),
            ),
            residual_report=types.SimpleNamespace(residual_entity_count=1),
        )

    monkeypatch.setattr(multimodal, "redact_dicom_pixels", partial_redact)
    report = run_multimodal_dicom(
        cases=[case],
        system_name="unit-test-system",
        pii_model_name="OpenMed/unit-test-pii",
    )

    assert received_models == ["OpenMed/unit-test-pii"]
    assert report.model_name == "unit-test-system"
    assert report.metadata["pii_model_name"] == "OpenMed/unit-test-pii"
    assert report.metrics["residual_phi_rate"] > 0.0
    assert report.metrics["character_recall"]["rate"] < 1.0
    assert report.metrics["leakage"]["leaked_chars_by_label"][ID_NUM] == len(
        target.text
    )


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


def test_default_fixture_temp_directory_is_cleaned(monkeypatch, tmp_path) -> None:
    import tempfile

    import openmed.eval.suites.multimodal_dicom as dicom_suite

    real_temporary_directory = tempfile.TemporaryDirectory

    def local_temporary_directory(*, prefix):
        return real_temporary_directory(prefix=prefix, dir=tmp_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", local_temporary_directory)
    with pytest.raises(ValueError, match="positive integer"):
        load_multimodal_dicom_fixtures(seed=111, corpus_size=0)
    assert not list(tmp_path.iterdir())

    fixtures = load_multimodal_dicom_fixtures(seed=111, corpus_size=1)
    dicom_path = Path(fixtures[0].metadata["dicom_path"])
    assert dicom_path.exists()

    dicom_suite._cleanup_fixture_temp_dirs()

    assert not dicom_path.exists()


def test_optional_pydicom_dependency_error_is_named(monkeypatch) -> None:
    import openmed.eval.suites.multimodal_dicom as dicom_suite
    from openmed.multimodal.exceptions import MissingDependencyError

    real_import_module = dicom_suite.importlib.import_module

    def import_without_pydicom(name):
        if name == "pydicom":
            raise ImportError("missing")
        return real_import_module(name)

    monkeypatch.setattr(dicom_suite.importlib, "import_module", import_without_pydicom)

    with pytest.raises(MissingDependencyError, match="pydicom"):
        dicom_suite._import_pydicom()


def test_generate_rejects_non_positive_corpus_size(tmp_path) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_dicom_corpus(tmp_path, corpus_size=0)


def test_run_rejects_an_empty_supplied_corpus() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_multimodal_dicom(cases=[])


def test_default_report_system_name_is_not_used_as_pii_checkpoint(tmp_path) -> None:
    from openmed.core.model_registry import get_default_pii_model

    report = run_multimodal_dicom(output_dir=tmp_path, seed=5, corpus_size=1)

    assert report.model_name == DEFAULT_SYSTEM_NAME
    assert report.metadata["system_name"] == DEFAULT_SYSTEM_NAME
    assert report.metadata["pii_model_name"] == get_default_pii_model("en")
    assert report.metadata["pii_model_name"] != DEFAULT_SYSTEM_NAME


def _refresh_source_hash(case: SyntheticDicomCase) -> SyntheticDicomCase:
    metadata = dict(case.metadata)
    metadata["source_dicom_sha256"] = hashlib.sha256(case.path.read_bytes()).hexdigest()
    return replace(case, metadata=metadata)
