"""Synthetic whole-slide instance refusal and content-bound retention tests."""

import hashlib
from dataclasses import replace

import pytest

from openmed.multimodal.dicom import (
    DicomHeaderDeidPolicy,
    DicomPixelRedactionResult,
    DicomResidualTextReport,
    _guard_slide_headers,
    _slide_instance_role,
    deidentify_dicom_headers,
    redact_dicom_pixels,
)

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian


def dataset(role):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"
    ds.SOPInstanceUID = "2.25.3817"
    ds.ImageType = ["ORIGINAL", "PRIMARY", role, "NONE"]
    ds.PatientName = "SYNTHETIC^NAME"
    ds.PixelData = b"\x00\x01\x02\x03"
    ds.Rows = 2
    ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    return ds


@pytest.mark.parametrize("role", ["LABEL", "OVERVIEW"])
def test_default_refusal_before_header_or_pixel_output(role, tmp_path):
    ds = dataset(role)
    source = tmp_path / "synthetic.dcm"
    ds.save_as(source, enforce_file_format=True)
    output = tmp_path / "copy.dcm"
    for action in (deidentify_dicom_headers, redact_dicom_pixels):
        with pytest.raises(ValueError, match="dicom_slide_associated_image_refused"):
            action(source, policy={"output_path": output})
        assert not output.exists()
        assert pydicom.dcmread(source).PatientName == "SYNTHETIC^NAME"


@pytest.mark.parametrize("role", ["LABEL", "OVERVIEW"])
def test_retention_requires_explicit_policy_review_and_verified_current_pixels(
    role, tmp_path
):
    ds = dataset(role)
    pixels = DicomPixelRedactionResult(
        tmp_path / "source",
        tmp_path / "copy",
        1,
        (),
        DicomResidualTextReport(1),
        residual_verified=True,
        pixel_sha256=hashlib.sha256(ds.PixelData).hexdigest(),
    )
    policy = DicomHeaderDeidPolicy(
        keep_slide_associated_images=True, reviewer_confirmed=True
    )
    _guard_slide_headers(ds, policy, pixels)
    for refused in [
        None,
        replace(pixels, frames_processed=0),
        replace(pixels, residual_verified=False),
        replace(pixels, pixel_sha256="0" * 64),
    ]:
        with pytest.raises(ValueError, match="dicom_slide_associated_image_refused"):
            _guard_slide_headers(ds, policy, refused)
    for invalid in [
        replace(policy, reviewer_confirmed=False),
        replace(policy, keep_slide_associated_images=False),
    ]:
        with pytest.raises(ValueError):
            _guard_slide_headers(ds, invalid, pixels)
    ds.PixelData = b"changed"
    with pytest.raises(ValueError):
        _guard_slide_headers(ds, policy, pixels)


@pytest.mark.parametrize("role", ["VOLUME", "THUMBNAIL"])
def test_volume_and_thumbnail_not_associated_unless_mislabeled(role):
    ds = dataset(role)
    _guard_slide_headers(ds, DicomHeaderDeidPolicy(), None)
    ds.SpecimenLabelInImage = "YES"
    with pytest.raises(ValueError, match="dicom_slide_image_type_unknown"):
        _slide_instance_role(ds)


def test_missing_and_unknown_wsi_image_type_refused():
    ds = dataset("UNKNOWN")
    with pytest.raises(ValueError, match="dicom_slide_image_type_unknown"):
        _slide_instance_role(ds)
    del ds.ImageType
    with pytest.raises(ValueError):
        _slide_instance_role(ds)


@pytest.mark.parametrize("combined", [False, True])
def test_retained_label_runs_actual_local_redaction_and_header_gate(
    monkeypatch, tmp_path, combined
):
    # Fully injected offline detector, no models or transports are loaded.
    import openmed.multimodal.dicom as module

    calls = []

    def detect(frame, **kwargs):
        calls.append(kwargs["frame_index"])
        from openmed.multimodal.dicom import DicomPixelFinding

        return (DicomPixelFinding(0, (0, 0, 2, 2), "ID_NUM", 1.0, "0" * 64, 4),)

    monkeypatch.setattr(module, "_detect_frame_pixel_findings", detect)

    def residual(frames, **kwargs):
        assert all(not frame.any() for frame in frames)
        return DicomResidualTextReport(len(frames))

    monkeypatch.setattr(module, "_residual_text_report", residual)
    source = tmp_path / "synthetic.dcm"
    output = tmp_path / "clean.dcm"
    dataset("LABEL").save_as(source, enforce_file_format=True)
    policy = {
        "output_path": output,
        "keep_slide_associated_images": True,
        "reviewer_confirmed": True,
        "ocr_engine": object(),
        "model_name": "offline-stub",
    }
    if combined:
        result = module._dicom_handler(source, policy=policy)
        assert result.metadata["dicom_pixel_redaction"]["residual_verified"]
    else:
        pixels = redact_dicom_pixels(source, policy=policy)
        assert pixels.residual_verified and pixels.pixel_sha256
        deidentify_dicom_headers(output, policy=policy, slide_pixel_redaction=pixels)
    assert calls == [0]
    assert pydicom.dcmread(output).PatientName == ""
    assert not pydicom.dcmread(output).pixel_array.any()
    # Same policy cannot bypass residual verification or explicit review.
    for changed in [
        {"verify_residual": False},
        {"fail_on_residual": False},
        {"reviewer_confirmed": False},
        {"keep_slide_associated_images": "true"},
    ]:
        with pytest.raises(ValueError, match="dicom_slide_associated_image_refused"):
            redact_dicom_pixels(source, policy={**policy, **changed})
