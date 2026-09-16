"""Tests for the DICOM SR content-tree structured-text extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.multimodal import (
    DICOM_SR_ADVISORY,
    extract_dicom_sr,
    redact_document,
    walk_sr_content_tree,
)

pydicom = pytest.importorskip("pydicom")

from tests.fixtures.multimodal.dicom_sr_synthetic import (  # noqa: E402
    PHI_TOKENS,
    build_synthetic_sr_dataset,
    write_synthetic_sr,
)

_EXPECTED_NODES = {
    "1": ("Imaging Measurement Report", "CONTAINER", "", None, None, "1500"),
    "1.1": (
        "Language of Content Item and Descendants",
        "CODE",
        "English",
        None,
        "HAS CONCEPT MOD",
        None,
    ),
    "1.2": (
        "Procedure reported",
        "CODE",
        "CT of chest",
        None,
        "HAS CONCEPT MOD",
        None,
    ),
    "1.3": ("Imaging Measurements", "CONTAINER", "", None, "CONTAINS", "1501"),
    "1.3.1": ("Measurement Group", "CONTAINER", "", None, "CONTAINS", "1502"),
    "1.3.1.1": (
        "Tracking Identifier",
        "TEXT",
        "Lesion 1",
        None,
        "HAS OBS CONTEXT",
        None,
    ),
    "1.3.1.2": (
        "Finding Site",
        "CODE",
        "Upper lobe of right lung",
        None,
        "HAS CONCEPT MOD",
        None,
    ),
    "1.3.1.3": ("Long Axis", "NUM", "12.5", "mm", "CONTAINS", None),
    "1.3.1.4": ("Short Axis", "NUM", "8.0", "mm", "CONTAINS", None),
    "1.4": ("Qualitative Evaluations", "CONTAINER", "", None, "CONTAINS", None),
    "1.4.1": (
        "Finding",
        "TEXT",
        "Solid pulmonary nodule, stable in size",
        None,
        "CONTAINS",
        None,
    ),
}


def _by_path(items):
    return {item["node_path"]: item for item in items}


def test_walk_sr_content_tree_reproduces_every_node():
    dataset = build_synthetic_sr_dataset()
    items = walk_sr_content_tree(dataset)

    by_path = _by_path(items)
    assert set(by_path) == set(_EXPECTED_NODES)
    for path, expected in _EXPECTED_NODES.items():
        item = by_path[path]
        concept, value_type, value, unit, relationship, template_id = expected
        assert item["concept_name"] == concept
        assert item["value_type"] == value_type
        assert item["value"] == value
        assert item["unit_code"] == unit
        assert item["relationship"] == relationship
        assert item["template_id"] == template_id


def test_num_items_carry_numeric_value_and_coded_unit(tmp_path: Path):
    source = write_synthetic_sr(tmp_path / "sr.dcm")
    document = extract_dicom_sr(source, policy={"date_shift_days": 5})
    by_path = _by_path(document.metadata["content_items"])

    long_axis = by_path["1.3.1.3"]
    assert long_axis["value_type"] == "NUM"
    assert long_axis["value"] == "12.5"
    assert long_axis["unit_code"] == "mm"

    short_axis = by_path["1.3.1.4"]
    assert short_axis["value_type"] == "NUM"
    assert short_axis["value"] == "8.0"
    assert short_axis["unit_code"] == "mm"


def test_nested_container_structure_and_node_paths_preserved(tmp_path: Path):
    source = write_synthetic_sr(tmp_path / "sr.dcm")
    document = extract_dicom_sr(source, policy={"date_shift_days": 5})
    items = document.metadata["content_items"]

    # Document order is preserved root-first with a stable, unique node_path.
    paths = [item["node_path"] for item in items]
    assert paths == list(_EXPECTED_NODES)
    assert len(set(paths)) == len(paths)

    # The measurement group is nested two containers deep under the root.
    group = _by_path(items)["1.3.1"]
    assert group["value_type"] == "CONTAINER"
    assert group["node_path"].count(".") == 2
    # Its children reference it via their node_path prefix.
    children = [p for p in paths if p.startswith("1.3.1.") and p.count(".") == 3]
    assert children == ["1.3.1.1", "1.3.1.2", "1.3.1.3", "1.3.1.4"]


def test_linearized_text_indents_by_depth_and_maps_spans(tmp_path: Path):
    source = write_synthetic_sr(tmp_path / "sr.dcm")
    document = extract_dicom_sr(source, policy={"date_shift_days": 5})

    assert document.text.splitlines()[0] == "Imaging Measurement Report"
    assert "  Procedure reported: CT of chest" in document.text
    assert "      Long Axis: 12.5 mm" in document.text
    assert "    Finding: Solid pulmonary nodule, stable in size" in document.text

    # Every node has a span mapping its rendered line back to its node_path.
    assert len(document.spans) == len(document.metadata["content_items"])
    for span in document.spans:
        assert document.text[span.start : span.end] == _line_for(document.text, span)
        assert span.metadata["node_path"] in _EXPECTED_NODES


def _line_for(text: str, span) -> str:
    return text[span.start : span.end]


def test_sr_header_phi_is_deidentified_before_text_emission(tmp_path: Path):
    source = write_synthetic_sr(tmp_path / "sr.dcm")
    document = extract_dicom_sr(source, policy={"date_shift_days": 9})

    serialized = json.dumps(
        {
            "text": document.text,
            "items": document.metadata["content_items"],
            "header": document.metadata.get("dicom_header_deid"),
        },
        sort_keys=True,
    )
    for token in PHI_TOKENS:
        assert token not in document.text
        assert token not in serialized

    assert document.metadata["headers_deidentified"] is True
    report = document.metadata["dicom_header_deid"]
    assert report["type"] == "dicom_header_deidentification"
    assert report["action_count"] > 0


def test_advisory_is_emitted(tmp_path: Path):
    source = write_synthetic_sr(tmp_path / "sr.dcm")
    document = extract_dicom_sr(source, policy={"date_shift_days": 3})

    assert document.metadata["advisory"] == DICOM_SR_ADVISORY
    assert "not a clinical interpretation" in DICOM_SR_ADVISORY
    assert "must not auto-trigger clinical decisions" in DICOM_SR_ADVISORY


def test_redact_document_dispatches_sr_extraction(tmp_path: Path):
    source = write_synthetic_sr(tmp_path / "sr.dcm")
    document = redact_document(source, policy={"date_shift_days": 4})

    assert document.metadata["format"] == "dicom_sr"
    assert document.metadata["node_count"] == len(_EXPECTED_NODES)
    assert "Imaging Measurement Report" in document.text


def test_extract_dicom_sr_rejects_non_sr_object(tmp_path: Path):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

    from openmed.multimodal.exceptions import UnsupportedDocumentError

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = "1.2.3.4.5"
    ds.StudyInstanceUID = "1.2.3.4.6"
    ds.SeriesInstanceUID = "1.2.3.4.7"
    ds.Modality = "CT"
    ds.PatientName = "DOE^Jane"
    ds.PatientID = "MRN-1"
    ds.PatientBirthDate = "19800101"
    ds.StudyDate = "20210101"
    path = tmp_path / "ct.dcm"
    ds.save_as(path, enforce_file_format=True)

    with pytest.raises(UnsupportedDocumentError, match="Structured Report"):
        extract_dicom_sr(path, deidentify_headers=False)


def test_missing_pydicom_raises_actionable_error(monkeypatch, tmp_path: Path):
    import openmed.multimodal.dicom_sr as sr_mod

    source = tmp_path / "sr.dcm"
    source.write_bytes(b"not a dicom")

    def missing_pydicom():
        raise sr_mod.MissingDependencyError(
            dependency="pydicom",
            instruction='Install with: pip install "openmed[multimodal]".',
        )

    monkeypatch.setattr(sr_mod, "_import_pydicom", missing_pydicom)
    with pytest.raises(sr_mod.MissingDependencyError, match="pydicom"):
        extract_dicom_sr(source)
