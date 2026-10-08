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
    assert str(redacted.StudyDescription) == ""
    assert str(redacted.ReferencedStudySequence[0].StudyDescription) == ""
    assert str(redacted.ReferencedStudySequence[0].SeriesDescription) == ""
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
    dataset.StudyDescription = "Case for MRN-12345 / DOE Jane"
    dataset.ReferencedStudySequence = Sequence([Dataset()])
    dataset.ReferencedStudySequence[0].ReferencedSOPInstanceUID = SOP_UID
    dataset.ReferencedStudySequence[0].StudyDescription = "Copy of DOE^Jane"
    dataset.ReferencedStudySequence[0].SeriesDescription = "Copy of Smith Alice"
    dataset.add_new((0x0011, 0x0010), "LO", "OPENMED_PRIVATE")
    dataset.add_new((0x0011, 0x1010), "LO", "PRIVATE PATIENT NOTE")
    dataset.save_as(path, enforce_file_format=True)
    return path


def _interval_days(left: str, right: str) -> int:
    first = datetime.strptime(left, "%Y%m%d")
    second = datetime.strptime(right, "%Y%m%d")
    return (second - first).days


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("pixel_api", [False, True])
def test_dicom_carriers_are_removed_at_every_depth(tmp_path, nested, pixel_api):
    from openmed.multimodal import redact_dicom_pixels

    source = _write_synthetic_dicom(tmp_path / "carriers.dcm")
    dataset = pydicom.dcmread(source)
    carrier = Dataset() if nested else dataset
    sentinel = b"SYNTHETIC_CARRIER_SENTINEL"
    for group in (0x5000, 0x5002, 0x6000, 0x6002):
        carrier.add_new((group, 0x3000), "OW", sentinel)
        carrier.add_new((group, 0x0022), "LO", sentinel.decode())
    icon = Dataset()
    icon.add_new((0x7FE0, 0x0010), "OB", sentinel)
    carrier.IconImageSequence = Sequence([icon])
    if nested:
        dataset.ReferencedStudySequence = Sequence([carrier])
    dataset.save_as(source, enforce_file_format=True)
    output = tmp_path / "clean.dcm"
    if pixel_api:
        result = redact_dicom_pixels(source, output_path=output)
        actions = result.to_audit_report()["carrier_actions"]
    else:
        result = deidentify_dicom_headers(source, policy={"output_path": output})
        actions = result.to_audit_report()["actions"]
    assert sentinel not in output.read_bytes()
    removed = [a for a in actions if a["tag"].startswith(("(500", "(600", "(0088,"))]
    assert len(removed) == 9
    assert all(a["action"] == "remove" and a["ps315_action"] == "X" for a in removed)
    assert all("value_sha256" not in a and "value_length" not in a for a in removed)
    assert sentinel.decode() not in json.dumps(result.to_audit_report())
    assert pydicom.dcmread(output).SOPClassUID == CTImageStorage


@pytest.mark.parametrize("flag", ["YES", None, "UNKNOWN"])
def test_header_only_pixels_have_typed_unclean_outcome(tmp_path, flag):
    from openmed.multimodal import DicomPixelStatus

    source = _write_synthetic_dicom(tmp_path / "pixels.dcm")
    dataset = pydicom.dcmread(source)
    dataset.add_new((0x7FE0, 0x0010), "OB", b"unchanged image bytes")
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethodCodeSequence = Sequence([Dataset()])
    if flag is not None:
        dataset.BurnedInAnnotation = flag
    dataset.save_as(source, enforce_file_format=True)
    original_pixels = pydicom.dcmread(source).PixelData
    result = deidentify_dicom_headers(source)
    cleaned = pydicom.dcmread(source)
    assert result.pixel_status is DicomPixelStatus.NOT_CLEANED
    assert result.to_audit_report()["pixel_status"] == "pixels_not_cleaned"
    assert cleaned.PatientIdentityRemoved == "NO"
    assert "pixels not cleaned" in cleaned.DeidentificationMethod
    assert "DeidentificationMethodCodeSequence" not in cleaned
    assert cleaned.PixelData == original_pixels


def test_header_unclean_fail_closed_preserves_existing_files(tmp_path):
    from openmed.multimodal import DicomDeidentificationError

    source = _write_synthetic_dicom(tmp_path / "pixels.dcm")
    dataset = pydicom.dcmread(source)
    dataset.add_new((0x7FE0, 0x0010), "OB", b"not cleaned pixels")
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    output = tmp_path / "existing.dcm"
    output.write_bytes(b"existing destination")
    with pytest.raises(DicomDeidentificationError) as exc:
        deidentify_dicom_headers(
            source, policy={"output_path": output, "fail_on_unclean_pixels": True}
        )
    assert exc.value.reason_code == "pixels_not_cleaned"
    assert source.read_bytes() == original
    assert output.read_bytes() == b"existing destination"


@pytest.mark.parametrize("pixel_api", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
def test_encapsulated_payload_refusal_never_copies_source(
    tmp_path, pixel_api, nested, in_place
):
    from openmed.multimodal import DicomDeidentificationError, redact_dicom_pixels

    source = _write_synthetic_dicom(tmp_path / "encapsulated.dcm")
    dataset = pydicom.dcmread(source)
    carrier = Dataset() if nested else dataset
    carrier.EncapsulatedDocument = b"%PDF-SYNTHETIC_SECRET_PAYLOAD"
    carrier.MIMETypeOfEncapsulatedDocument = "application/pdf"
    if nested:
        dataset.ReferencedStudySequence = Sequence([carrier])
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    output = source if in_place else tmp_path / "existing.dcm"
    if not in_place:
        output.write_bytes(b"existing destination")
    with pytest.raises(DicomDeidentificationError) as exc:
        if pixel_api:
            redact_dicom_pixels(source, output_path=output)
        else:
            deidentify_dicom_headers(source, policy={"output_path": output})
    assert exc.value.reason_code == "encapsulated_document_requires_redaction"
    assert str(exc.value) == "encapsulated_document_requires_redaction"
    assert source.read_bytes() == original
    if not in_place:
        assert output.read_bytes() == b"existing destination"


@pytest.mark.parametrize("pixel_api", [False, True])
def test_encapsulated_redaction_uses_registered_handler_in_memory(
    monkeypatch, tmp_path, pixel_api
):
    from openmed.multimodal import (
        ExtractedDocument,
        base,
        redact_dicom_pixels,
        register_handler,
    )

    source = _write_synthetic_dicom(tmp_path / "encapsulated.dcm")
    dataset = pydicom.dcmread(source)
    dataset.EncapsulatedDocument = b"%PDF-SYNTHETIC_PAYLOAD_SENTINEL"
    dataset.MIMETypeOfEncapsulatedDocument = "application/pdf"
    dataset.save_as(source, enforce_file_format=True)
    calls = []

    def redact(stream, *, policy, models, lang):
        assert stream.read().startswith(b"%PDF-SYNTHETIC_PAYLOAD_SENTINEL")
        assert policy == {"return_bytes": True}
        assert callable(models)
        calls.append(stream.name)
        return ExtractedDocument(
            text="", metadata={"redacted_document_bytes": b"%PDF-CLEAN"}
        )

    monkeypatch.setitem(base._HANDLERS, ".pdf", [])
    register_handler(".pdf", redact, requires_multimodal=False)
    output = tmp_path / "clean.dcm"
    policy = {
        "output_path": output,
        "redact_encapsulated_documents": True,
        "document_policy": {"output_path": tmp_path / "must-not-write.pdf"},
        "document_models": lambda _: [],
    }
    result = (
        redact_dicom_pixels(source, policy=policy)
        if pixel_api
        else deidentify_dicom_headers(source, policy=policy)
    )
    cleaned = pydicom.dcmread(output)
    assert cleaned.EncapsulatedDocument == b"%PDF-CLEAN"
    assert cleaned.EncapsulatedDocumentLength == 10
    assert b"SYNTHETIC_PAYLOAD_SENTINEL" not in output.read_bytes()
    assert not (tmp_path / "must-not-write.pdf").exists()
    assert calls == ["encapsulated.pdf"]
    actions = result.to_audit_report()["carrier_actions" if pixel_api else "actions"]
    action = next(a for a in actions if a["tag"] == "(0042,0011)")
    assert action["action"] == "replace" and action["ps315_action"] == "D"
    assert "value_sha256" not in action


@pytest.mark.parametrize(
    "failure", ["no_bytes", "unchanged", "exception", "no_detector"]
)
def test_encapsulated_handler_must_prove_byte_replacement(
    monkeypatch, tmp_path, failure
):
    from openmed.multimodal import (
        DicomDeidentificationError,
        ExtractedDocument,
        base,
        register_handler,
    )

    source = _write_synthetic_dicom(tmp_path / "encapsulated.dcm")
    dataset = pydicom.dcmread(source)
    dataset.EncapsulatedDocument = b"%PDF-SYNTHETIC_PAYLOAD"
    dataset.MIMETypeOfEncapsulatedDocument = "application/pdf"
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()

    def handler(stream, **kwargs):
        if failure == "exception":
            raise ValueError("SYNTHETIC_PAYLOAD private handler details")
        payload = stream.read()
        return ExtractedDocument(
            text="",
            metadata={} if failure == "no_bytes" else {"redacted_pdf_bytes": payload},
        )

    monkeypatch.setitem(base._HANDLERS, ".pdf", [])
    register_handler(".pdf", handler, requires_multimodal=False)
    with pytest.raises(DicomDeidentificationError) as exc:
        deidentify_dicom_headers(
            source,
            policy={
                "redact_encapsulated_documents": True,
                "document_models": None if failure == "no_detector" else lambda _: [],
            },
        )
    assert str(exc.value) == "encapsulated_document_redaction_failed"
    assert source.read_bytes() == original


def test_external_pixel_provider_cannot_attest_header_only_deidentification(tmp_path):
    from openmed.multimodal import DicomDeidentificationError

    source = _write_synthetic_dicom(tmp_path / "external.dcm")
    dataset = pydicom.dcmread(source)
    dataset.PixelDataProviderURL = "https://example.invalid/synthetic-private"
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    with pytest.raises(
        DicomDeidentificationError, match="external_pixel_data_unsupported"
    ):
        deidentify_dicom_headers(source)
    assert source.read_bytes() == original
