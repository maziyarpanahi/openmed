"""Tests for DICOM PS3.15 header de-identification."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from openmed.multimodal import (
    DicomHeaderDeidPolicy,
    deidentify_dicom_headers,
    redact_document,
)

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

STUDY_UID = "1.2.826.0.1.3680043.10.543.1"
SERIES_UID = "1.2.826.0.1.3680043.10.543.2"
SOP_UID = "1.2.826.0.1.3680043.10.543.3"


def test_deidentify_dicom_headers_removes_phi_and_remaps_uids(tmp_path: Path):
    source = _write_synthetic_dicom(tmp_path / "phi.dcm")
    output = tmp_path / "deid.dcm"

    result = deidentify_dicom_headers(
        source,
        policy=DicomHeaderDeidPolicy(output_path=output, date_shift_days=10),
    )

    redacted = pydicom.dcmread(output)
    assert str(redacted.PatientName) == ""
    assert redacted.PatientID == ""
    assert redacted.PatientBirthDate == ""
    assert redacted.InstitutionName == ""
    assert str(redacted.ReferringPhysicianName) == ""
    assert (0x0011, 0x1010) not in redacted

    assert redacted.StudyInstanceUID != STUDY_UID
    assert redacted.SeriesInstanceUID != SERIES_UID
    assert redacted.SOPInstanceUID != SOP_UID
    assert redacted.file_meta.MediaStorageSOPInstanceUID == redacted.SOPInstanceUID
    assert (
        redacted.ReferencedStudySequence[0].ReferencedSOPInstanceUID
        == redacted.SOPInstanceUID
    )
    assert str(redacted.StudyInstanceUID).startswith("2.25.")

    assert redacted.StudyDate == "20200111"
    assert redacted.SeriesDate == "20200121"
    assert _interval_days(redacted.StudyDate, redacted.SeriesDate) == 10
    assert redacted.ContentDate == "20200210"
    assert redacted.LongitudinalTemporalInformationModified == "MODIFIED"
    assert redacted.PatientIdentityRemoved == "YES"
    assert "PS3.15" in redacted.DeidentificationMethod

    assert result.output_path == output
    assert result.uid_remap_count == 3
    assert result.private_tag_removed_count == 2


def test_dicom_provenance_lists_acted_tags_without_raw_phi(tmp_path: Path):
    source = _write_synthetic_dicom(tmp_path / "phi.dcm")
    result = deidentify_dicom_headers(
        source,
        policy={"output_path": tmp_path / "deid.dcm", "date_shift_days": 7},
    )

    report = result.to_audit_report()
    acted_tags = {action["tag"] for action in report["actions"]}
    assert {
        "(0010,0010)",
        "(0010,0020)",
        "(0010,0030)",
        "(0008,0080)",
        "(0008,0090)",
        "(0020,000D)",
        "(0020,000E)",
        "(0008,0018)",
    }.issubset(acted_tags)
    assert report["type"] == "dicom_header_deidentification"
    assert report["action_counts"]["replace_uid"] >= 3
    assert report["action_counts"]["shift_date"] >= 2

    serialized = json.dumps(report, sort_keys=True)
    assert "Jane" not in serialized
    assert "DOE" not in serialized
    assert "MRN-12345" not in serialized
    assert "OpenMed Clinic" not in serialized
    assert STUDY_UID not in serialized


def test_redact_document_dispatches_dicom_header_pass(tmp_path: Path):
    source = _write_synthetic_dicom(tmp_path / "phi.dcm")
    output = tmp_path / "dispatch.dcm"

    document = redact_document(
        source,
        policy=DicomHeaderDeidPolicy(output_path=output, date_shift_days=3),
    )

    assert document.text == ""
    assert document.metadata["format"] == "dicom"
    report = document.metadata["dicom_header_deid"]
    assert report["output_suffix"] == ".dcm"
    assert report["action_count"] > 0
    assert output.exists()


def test_dicom_missing_dependency_raises_named_error(monkeypatch, tmp_path: Path):
    import openmed.multimodal.dicom as dicom_mod

    source = tmp_path / "not-read.dcm"
    source.write_bytes(b"not a dicom")

    def missing_pydicom():
        raise dicom_mod.MissingDependencyError(
            dependency="pydicom",
            instruction='Install with: pip install "openmed[multimodal]".',
        )

    monkeypatch.setattr(dicom_mod, "_import_pydicom", missing_pydicom)
    with pytest.raises(dicom_mod.MissingDependencyError, match="pydicom"):
        deidentify_dicom_headers(source)


def _write_synthetic_dicom(path: Path) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = SOP_UID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.543.99"
    file_meta.SourceApplicationEntityTitle = "PHI_AE_TITLE"

    dataset = FileDataset(
        str(path),
        {},
        file_meta=file_meta,
        preamble=(b"PHI" * 42) + b"!!",
    )
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = SOP_UID
    dataset.StudyInstanceUID = STUDY_UID
    dataset.SeriesInstanceUID = SERIES_UID
    dataset.PatientName = "DOE^Jane"
    dataset.PatientID = "MRN-12345"
    dataset.PatientBirthDate = "19800102"
    dataset.InstitutionName = "OpenMed Clinic"
    dataset.ReferringPhysicianName = "Smith^Alice"
    dataset.StudyDate = "20200101"
    dataset.SeriesDate = "20200111"
    dataset.ContentDate = "20200131"
    dataset.StudyTime = "121314"
    dataset.ReferencedStudySequence = Sequence([Dataset()])
    dataset.ReferencedStudySequence[0].ReferencedSOPInstanceUID = SOP_UID
    dataset.add_new((0x0011, 0x0010), "LO", "OPENMED_PRIVATE")
    dataset.add_new((0x0011, 0x1010), "LO", "PRIVATE PATIENT NOTE")
    dataset.save_as(path, enforce_file_format=True)
    return path


def _interval_days(left: str, right: str) -> int:
    first = datetime.strptime(left, "%Y%m%d")
    second = datetime.strptime(right, "%Y%m%d")
    return (second - first).days
