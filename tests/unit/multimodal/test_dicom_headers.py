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

import openmed.multimodal.dicom as dicom_module

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
    assert redacted.get("InstitutionName", "") == ""
    assert str(redacted.ReferringPhysicianName) == ""
    assert "StudyDescription" not in redacted
    assert "ReferencedStudySequence" not in redacted
    assert (0x0011, 0x1010) not in redacted

    assert redacted.StudyInstanceUID != STUDY_UID
    assert redacted.SeriesInstanceUID != SERIES_UID
    assert redacted.SOPInstanceUID != SOP_UID
    assert redacted.file_meta.MediaStorageSOPInstanceUID == redacted.SOPInstanceUID
    assert str(redacted.StudyInstanceUID).startswith("2.25.")

    assert redacted.StudyDate == "20200111"
    assert redacted.SeriesDate == "20200121"
    assert _interval_days(redacted.StudyDate, redacted.SeriesDate) == 10
    assert redacted.ContentDate == "20200210"
    assert redacted.LongitudinalTemporalInformationModified == "MODIFIED"
    assert redacted.PatientIdentityRemoved == "YES"
    assert "PS3.15" in str(redacted.DeidentificationMethod)

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
    assert "pixels not cleaned" in str(cleaned.DeidentificationMethod)
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


@pytest.mark.parametrize("tag_key,keyword,action", dicom_module._BASIC_PROFILE_CATALOG)
def test_pinned_basic_profile_catalog_conformance(tag_key, keyword, action):
    """Exercise every normative row's selected action on a synthetic value."""
    from pydicom.datadict import dictionary_VR
    from pydicom.dataelem import DataElement

    assert dicom_module._BASIC_PROFILE_VERSION == "2026d"
    assert len(dicom_module._BASIC_PROFILE_CATALOG) == 657
    tag = 0x00111010 if tag_key == "PRIVATE" else int(tag_key.replace("X", "0"), 16)
    try:
        vr = dictionary_VR(tag).split(" or ")[0]
    except KeyError:
        vr = "LO"
    values = {
        "UI": "1.2.826.0.1.3680043.10.543.123",
        "DA": "20001231",
        "DT": "20001231123456",
        "TM": "123456",
        "AS": "030Y",
        "DS": "123",
        "IS": "123",
        "US": 123,
        "SS": 123,
        "UL": 123,
        "SL": 123,
        "UV": 123,
        "SV": 123,
        "FL": 123.0,
        "FD": 123.0,
        "AT": 0x00100010,
    }
    if vr == "SQ":
        nested = Dataset()
        nested.PatientName = "Unseeded^Sentinel"
        value = [nested]
    elif vr in {"OB", "OW", "OF", "OD", "OL", "OV", "UN"}:
        value = b"SYNTHETIC_SENTINEL"
    else:
        value = values.get(vr, "SYNTHETIC")
    element = DataElement(tag, vr, value)
    dataset = Dataset()
    dataset.add(element)
    context = dicom_module._Context(
        date_shift_days=0, keep_year=False, uid_salt=b"test"
    )
    assert dicom_module._catalog_action(element, context) == action
    # This is the shared action executor. Carrier preprocessing is separately
    # exercised through both public APIs by the carrier regression tests above.
    dicom_module._apply_profile_action(
        dataset, element, context, action, location="Dataset"
    )
    selected = action.split("/")[0]
    if selected == "X":
        assert tag not in dataset
    elif selected == "Z":
        assert not dataset[tag].value
    elif selected == "D":
        assert dataset[tag].value != value
        assert dataset[tag].value != ""
        if vr == "SQ":
            assert len(dataset[tag].value) == 1
            assert "PatientName" not in dataset[tag].value[0]
    elif selected == "U":
        assert str(dataset[tag].value).startswith("2.25.")
        assert dataset[tag].value != value
    assert context.actions[-1].ps315_action == action


def test_default_profile_removes_unseeded_descriptors_and_sr_bytes(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "default.dcm")
    dataset = pydicom.dcmread(source)
    sentinel = "UNSEEDED_PHI"
    for keyword in (
        "StudyDescription",
        "SeriesDescription",
        "ImageComments",
        "ProtocolName",
        "DeviceSerialNumber",
        "StationName",
        "PatientAddress",
    ):
        setattr(dataset, keyword, sentinel)
    item = Dataset()
    item.ValueType = "TEXT"
    item.TextValue = sentinel
    item.PatientName = "UNSEEDED^PHI"
    dataset.ContentSequence = Sequence([item])
    dataset.Rows, dataset.Columns = 2, 3
    dataset.save_as(source, enforce_file_format=True)
    result = deidentify_dicom_headers(source)
    cleaned = pydicom.dcmread(source)
    assert sentinel.encode() not in source.read_bytes()
    assert b"UNSEEDED^PHI" not in source.read_bytes()
    assert cleaned.ContentSequence[0].TextValue == "[REMOVED]"
    assert len(cleaned.ContentSequence) == 1
    assert (cleaned.Rows, cleaned.Columns) == (2, 3)
    assert cleaned.LongitudinalTemporalInformationModified == "REMOVED"
    assert result.date_shift_days == 0
    assert result.profile_options == ()
    assert [item.CodeValue for item in cleaned.DeidentificationMethodCodeSequence] == [
        "113100"
    ]
    assert cleaned.DeidentificationMethod == "PS3.15 Basic Profile 2026d"


@pytest.mark.parametrize(
    "option,code",
    [
        ("clean_descriptors", "113105"),
        ("clean_structured_content", "113104"),
        ("retain_longitudinal_temporal_information", "113107"),
        ("retain_device_identity", "113109"),
        ("retain_patient_characteristics", "113108"),
        ("retain_uids", "113110"),
    ],
)
def test_profile_options_have_matching_codes_and_methods(tmp_path, option, code):
    source = _write_synthetic_dicom(tmp_path / "option.dcm")
    result = deidentify_dicom_headers(
        source, policy={option: True, "detector": lambda text: []}
    )
    cleaned = pydicom.dcmread(source)
    items = cleaned.DeidentificationMethodCodeSequence
    assert [item.CodeValue for item in items] == ["113100", code]
    assert all(item.CodingSchemeDesignator == "DCM" for item in items)
    assert dicom_module._PROFILE_OPTIONS[option][1] in cleaned.DeidentificationMethod
    assert result.profile_options == (option,)
    assert result.to_audit_report()["profile_version"] == "2026d"


def _sentinel_detector(text):
    token = "UNSEEDED_PHI"
    start = text.find(token)
    return [] if start < 0 else [{"start": start, "end": start + len(token)}]


def test_clean_options_use_independent_detector_at_nested_depth(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "clean-options.dcm")
    dataset = pydicom.dcmread(source)
    dataset.StudyDescription = "CT chest UNSEEDED_PHI"
    item = Dataset()
    item.ValueType = "TEXT"
    item.TextValue = "Stable nodule UNSEEDED_PHI"
    concept = Dataset()
    concept.CodeValue = "121071"
    concept.CodingSchemeDesignator = "DCM"
    concept.CodeMeaning = "Finding UNSEEDED_PHI"
    item.ConceptNameCodeSequence = Sequence([concept])
    dataset.ContentSequence = Sequence([item])
    dataset.save_as(source, enforce_file_format=True)
    deidentify_dicom_headers(
        source,
        policy={
            "clean_descriptors": True,
            "clean_structured_content": True,
            "detector": _sentinel_detector,
        },
    )
    cleaned = pydicom.dcmread(source)
    assert "UNSEEDED_PHI" not in source.read_bytes().decode("latin1")
    assert cleaned.StudyDescription == "CT chest [REMOVED]"
    assert cleaned.ContentSequence[0].TextValue == "Stable nodule [REMOVED]"
    assert (
        cleaned.ContentSequence[0].ConceptNameCodeSequence[0].CodeMeaning
        == "Finding [REMOVED]"
    )


@pytest.mark.parametrize(
    "detector",
    [
        None,
        lambda text: None,
        lambda text: "UNSEEDED_PHI",
        lambda text: {"wrong": []},
        lambda text: [{"start": -1, "end": 4}],
        lambda text: [{"start": 0, "end": len(text) + 1}],
    ],
)
def test_clean_detector_refusals_preserve_source_and_destination(tmp_path, detector):
    source = _write_synthetic_dicom(tmp_path / "refuse.dcm")
    dataset = pydicom.dcmread(source)
    dataset.StudyDescription = "UNSEEDED_PHI"
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    output = tmp_path / "existing.dcm"
    output.write_bytes(b"existing destination")
    with pytest.raises(dicom_module.DicomDeidentificationError) as exc:
        deidentify_dicom_headers(
            source,
            policy={
                "output_path": output,
                "clean_descriptors": True,
                "detector": detector,
            },
        )
    assert exc.value.reason_code in {
        "profile_detector_required",
        "profile_detector_failed",
    }
    assert "UNSEEDED_PHI" not in str(exc.value)
    assert source.read_bytes() == original
    assert output.read_bytes() == b"existing destination"


def test_detector_exception_is_value_free(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "exception.dcm")

    def detector(text):
        raise RuntimeError("UNSEEDED_PHI")

    with pytest.raises(dicom_module.DicomDeidentificationError) as exc:
        deidentify_dicom_headers(
            source, policy={"clean_descriptors": True, "detector": detector}
        )
    assert str(exc.value) == "profile_detector_failed"
    assert exc.value.__suppress_context__ is True


@pytest.mark.parametrize("option", dicom_module._PROFILE_OPTIONS)
@pytest.mark.parametrize("value", ["false", 1, None])
def test_retain_options_require_exact_booleans(tmp_path, option, value):
    source = _write_synthetic_dicom(tmp_path / "invalid.dcm")
    original = source.read_bytes()
    with pytest.raises(
        dicom_module.DicomDeidentificationError, match="invalid_profile_option"
    ):
        deidentify_dicom_headers(source, policy={option: value})
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "vr,value",
    [
        ("LO", "UNLISTED_PHI"),
        ("UI", "1.2.3.454545"),
        ("UN", b"UNLISTED_PHI"),
    ],
)
def test_unknown_public_attributes_are_removed_and_reported_by_tag(tmp_path, vr, value):
    source = _write_synthetic_dicom(tmp_path / "unknown.dcm")
    dataset = pydicom.dcmread(source)
    dataset.add_new((0x7776, 0x1000), vr, value)
    dataset.save_as(source, enforce_file_format=True)
    result = deidentify_dicom_headers(source)
    assert (0x7776, 0x1000) not in pydicom.dcmread(source)
    actions = [a.to_dict() for a in result.actions if a.tag == "(7776,1000)"]
    assert actions
    assert all(a["keyword"] == "" and a["vr"] == "" for a in actions)
    assert all("value_sha256" not in a and "value_length" not in a for a in actions)
    assert "UNLISTED_PHI" not in json.dumps(result.to_audit_report())


def test_file_meta_and_directory_values_cannot_preserve_identifiers(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "meta.dcm")
    dataset = pydicom.dcmread(source)
    dataset.file_meta.SendingApplicationEntityTitle = "SYNTHETIC_PHI"
    dataset.file_meta.ReceivingApplicationEntityTitle = "SYNTHETIC_PHI"
    dataset.file_meta.PrivateInformationCreatorUID = "1.2.3.777"
    dataset.file_meta.PrivateInformation = b"SYNTHETIC_PHI"
    dataset.add_new((0x0004, 0x1130), "CS", "SYNTHETIC_PHI")
    original_implementation = dataset.file_meta.ImplementationClassUID
    dataset.save_as(source, enforce_file_format=True)
    deidentify_dicom_headers(source)
    cleaned = pydicom.dcmread(source)
    assert b"SYNTHETIC_PHI" not in source.read_bytes()
    assert not any(int(e.tag) >> 16 == 4 for e in cleaned)
    assert cleaned.file_meta.ImplementationClassUID != original_implementation
    assert cleaned.file_meta.MediaStorageSOPInstanceUID == cleaned.SOPInstanceUID
    assert cleaned.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert cleaned.file_meta.MediaStorageSOPClassUID == CTImageStorage


def test_multivalue_uid_and_date_cardinality_and_consistency_are_preserved(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "multi.dcm")
    dataset = pydicom.dcmread(source)
    dataset.AcquisitionDate = ["20200101", "20200102"]
    dataset.FailedSOPInstanceUIDList = [SOP_UID, SERIES_UID]
    dataset.save_as(source, enforce_file_format=True)
    deidentify_dicom_headers(source, policy={"date_shift_days": 10})
    cleaned = pydicom.dcmread(source)
    assert list(cleaned.AcquisitionDate) == ["20200111", "20200112"]
    assert len(cleaned.FailedSOPInstanceUIDList) == 2
    assert cleaned.FailedSOPInstanceUIDList[0] == cleaned.SOPInstanceUID
    assert cleaned.FailedSOPInstanceUIDList[1] == cleaned.SeriesInstanceUID
    assert cleaned.StudyTime == "121314"


def test_retain_uids_preserves_references_and_meta_identity(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "retained.dcm")
    deidentify_dicom_headers(source, policy={"retain_uids": True})
    cleaned = pydicom.dcmread(source)
    assert cleaned.SOPInstanceUID == SOP_UID
    assert cleaned.file_meta.MediaStorageSOPInstanceUID == SOP_UID
    assert cleaned.ReferencedStudySequence[0].ReferencedSOPInstanceUID == SOP_UID
    assert "StudyDescription" not in cleaned.ReferencedStudySequence[0]


@pytest.mark.parametrize(
    "concept_key,rule", dicom_module._STRUCTURED_CONTENT_ACTIONS.items()
)
def test_structured_content_catalog_rules(concept_key, rule):
    scheme, code_value, value_type = concept_key
    action, _overrides = rule
    item = Dataset()
    item.ValueType = value_type
    concept = Dataset()
    concept.CodeValue, concept.CodingSchemeDesignator, concept.CodeMeaning = (
        code_value,
        scheme,
        "Synthetic concept",
    )
    item.ConceptNameCodeSequence = Sequence([concept])
    fields = {
        "TEXT": ("TextValue", "SYNTHETIC"),
        "PNAME": ("PersonName", "Synthetic^Person"),
        "UIDREF": ("UID", SOP_UID),
        "DATE": ("Date", "20001231"),
        "DATETIME": ("DateTime", "20001231123456"),
        "TIME": ("Time", "123456"),
    }
    keyword, value = fields.get(
        value_type, ("ReferencedSOPSequence", Sequence([Dataset()]))
    )
    if value_type == "NUM":
        keyword = "MeasuredValueSequence"
        value = Sequence([Dataset()])
        value[0].NumericValue = "123"
    if value_type == "CODE":
        keyword = "ConceptCodeSequence"
        value = Sequence([Dataset()])
        value[0].CodeValue, value[0].CodingSchemeDesignator = "123", "DCM"
        value[0].CodeMeaning = "SYNTHETIC"
    if isinstance(value, Sequence):
        value[0].PatientName = "Synthetic^Sentinel"
    setattr(item, keyword, value)
    dataset = Dataset()
    dataset.ContentSequence = Sequence([item])
    context = dicom_module._Context(
        date_shift_days=0,
        keep_year=False,
        uid_salt=b"test",
        profile_options=("clean_structured_content",),
        detector=lambda text: [],
    )
    dicom_module._deidentify_dataset(dataset, context, location="Dataset")
    if action.split("/")[0] == "X":
        assert len(dataset.ContentSequence) == 0
    else:
        assert len(dataset.ContentSequence) == 1
        actual = dataset.ContentSequence[0][keyword].value
        assert actual != value
        assert actual != ""


@pytest.mark.parametrize("retain", [False, True])
def test_structured_numeric_patient_characteristic_requires_retain_option(
    tmp_path, retain
):
    source = _write_synthetic_dicom(tmp_path / "characteristic.dcm")
    dataset = pydicom.dcmread(source)
    item = Dataset()
    item.ValueType = "NUM"
    concept = Dataset()
    concept.CodeValue, concept.CodingSchemeDesignator, concept.CodeMeaning = (
        "121033",
        "DCM",
        "Subject Age",
    )
    item.ConceptNameCodeSequence = Sequence([concept])
    value = Dataset()
    value.NumericValue = "47"
    item.MeasuredValueSequence = Sequence([value])
    dataset.ContentSequence = Sequence([item])
    dataset.save_as(source, enforce_file_format=True)
    deidentify_dicom_headers(
        source,
        policy={
            "clean_structured_content": True,
            "retain_patient_characteristics": retain,
            "detector": lambda text: [],
        },
    )
    cleaned = pydicom.dcmread(source)
    assert len(cleaned.ContentSequence) == int(retain)
    if retain:
        assert cleaned.ContentSequence[0].MeasuredValueSequence[0].NumericValue == "47"


@pytest.mark.parametrize("scheme", ["SRT", "SNM3", "99SDM", "UMLS", "99PRIVATE"])
def test_retired_and_unrecognized_structured_concepts_fail_closed(scheme):
    item = Dataset()
    item.ValueType = "NUM"
    concept = Dataset()
    concept.CodeValue, concept.CodingSchemeDesignator = "UNKNOWN", scheme
    item.ConceptNameCodeSequence = Sequence([concept])
    value = Dataset()
    value.NumericValue = "123456"
    item.MeasuredValueSequence = Sequence([value])
    context = dicom_module._Context(
        date_shift_days=0,
        keep_year=False,
        uid_salt=b"test",
        profile_options=("clean_structured_content",),
        detector=lambda text: [],
    )
    keep, _ = dicom_module._clean_structured_item(
        item, context, location="Dataset.ContentSequence[0]"
    )
    assert keep is False


def test_combined_dispatch_preserves_clean_options_and_detector_models(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "dispatch-options.dcm")
    dataset = pydicom.dcmread(source)
    dataset.StudyDescription = "Clinical description UNSEEDED_PHI"
    dataset.save_as(source, enforce_file_format=True)
    document = redact_document(
        source,
        policy={"clean_descriptors": True},
        models={"detector": _sentinel_detector},
    )
    assert pydicom.dcmread(source).StudyDescription == "Clinical description [REMOVED]"
    assert document.metadata["dicom_header_deid"]["profile_options"] == [
        "clean_descriptors"
    ]


def test_malformed_retained_temporal_values_never_copy_unknown_suffixes(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "temporal.dcm")
    dataset = pydicom.dcmread(source)
    with pytest.warns(UserWarning):
        dataset.AcquisitionDateTime = "20000101UNSEEDED_PHI"
    dataset.save_as(source, enforce_file_format=True)
    deidentify_dicom_headers(source, policy={"date_shift_days": 3})
    assert b"UNSEEDED_PHI" not in source.read_bytes()
    assert pydicom.dcmread(source).AcquisitionDateTime == ""


@pytest.mark.parametrize("result", [{"entities": ""}, {"entities": {}}, {"spans": b""}])
def test_empty_malformed_detector_seams_are_refused(tmp_path, result):
    source = _write_synthetic_dicom(tmp_path / "malformed-detector.dcm")
    original = source.read_bytes()
    with pytest.raises(
        dicom_module.DicomDeidentificationError, match="profile_detector_failed"
    ):
        deidentify_dicom_headers(
            source,
            policy={
                "clean_descriptors": True,
                "detector": lambda text: result,
            },
        )
    assert source.read_bytes() == original


@pytest.mark.parametrize("in_place", [False, True])
def test_binary_descriptor_cleaning_refuses_before_first_write(tmp_path, in_place):
    source = _write_synthetic_dicom(tmp_path / "binary-descriptor.dcm")
    dataset = pydicom.dcmread(source)
    dataset.DeviceSettingDescription = b"SYNTHETIC camera settings"
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    output = source if in_place else tmp_path / "existing.dcm"
    if not in_place:
        output.write_bytes(b"existing destination")
    with pytest.raises(
        dicom_module.DicomDeidentificationError,
        match="profile_binary_cleaning_unsupported",
    ):
        deidentify_dicom_headers(
            source,
            policy={
                "output_path": output,
                "clean_descriptors": True,
                "detector": lambda text: [],
            },
        )
    assert source.read_bytes() == original
    if not in_place:
        assert output.read_bytes() == b"existing destination"


def test_explicit_vr_cannot_hide_text_as_binary(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "wrong-vr.dcm")
    dataset = pydicom.dcmread(source)
    item = Dataset()
    item.add_new((0x0008, 0x0104), "OB", b"UNSEEDED_PHI")
    dataset.ProcedureCodeSequence = Sequence([item])
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    with pytest.raises(
        dicom_module.DicomDeidentificationError, match="profile_vr_mismatch"
    ):
        deidentify_dicom_headers(source)
    assert source.read_bytes() == original


@pytest.mark.parametrize("alternative", ["CodeValue", "LongCodeValue", "URNCodeValue"])
def test_alternate_sr_concept_value_encodings_enforce_characteristic_actions(
    tmp_path, alternative
):
    source = _write_synthetic_dicom(tmp_path / "alternate-concept.dcm")
    dataset = pydicom.dcmread(source)
    item = Dataset()
    item.ValueType = "NUM"
    concept = Dataset()
    concept.CodingSchemeDesignator = "DCM"
    setattr(concept, alternative, "121033")
    item.ConceptNameCodeSequence = Sequence([concept])
    measurement = Dataset()
    measurement.NumericValue = "47"
    item.MeasuredValueSequence = Sequence([measurement])
    dataset.ContentSequence = Sequence([item])
    dataset.save_as(source, enforce_file_format=True)
    deidentify_dicom_headers(
        source, policy={"clean_structured_content": True, "detector": lambda text: []}
    )
    assert len(pydicom.dcmread(source).ContentSequence) == 0


@pytest.mark.parametrize(
    "code,value_type", [(["121033", "121033"], "NUM"), ("121030", "NUM")]
)
def test_malformed_sr_concept_cannot_fall_through_to_numeric_retention(
    tmp_path, code, value_type
):
    source = _write_synthetic_dicom(tmp_path / "malformed-concept.dcm")
    dataset = pydicom.dcmread(source)
    item = Dataset()
    item.ValueType = value_type
    concept = Dataset()
    concept.CodingSchemeDesignator, concept.CodeValue = "DCM", code
    item.ConceptNameCodeSequence = Sequence([concept])
    measurement = Dataset()
    measurement.NumericValue = "123456"
    item.MeasuredValueSequence = Sequence([measurement])
    dataset.ContentSequence = Sequence([item])
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    policy = {"clean_structured_content": True, "detector": lambda text: []}
    if isinstance(code, list):
        with pytest.raises(
            dicom_module.DicomDeidentificationError, match="profile_concept_invalid"
        ):
            deidentify_dicom_headers(source, policy=policy)
        assert source.read_bytes() == original
    else:
        deidentify_dicom_headers(source, policy=policy)
        assert len(pydicom.dcmread(source).ContentSequence) == 0


def test_empty_required_uid_is_refused_before_writing(tmp_path):
    source = _write_synthetic_dicom(tmp_path / "empty-uid.dcm")
    dataset = pydicom.dcmread(source)
    dataset.SOPInstanceUID = ""
    dataset.save_as(source, enforce_file_format=True)
    original = source.read_bytes()
    with pytest.raises(
        dicom_module.DicomDeidentificationError, match="profile_uid_invalid"
    ):
        deidentify_dicom_headers(source)
    assert source.read_bytes() == original
