"""Tests for universal metadata scrubbing across images and documents."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

import openmed.multimodal.metadata_scrub as metadata_scrub_mod
from openmed.multimodal import (
    MetadataScrubResult,
    scrub_metadata,
    verify_metadata,
)
from openmed.multimodal.exceptions import MissingDependencyError

XMP_PACKET = b"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      dc:creator="Dr Jane Smith"
      xmp:CreateDate="2026-01-02T03:04:05Z" />
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def test_scrub_metadata_raises_clear_message_when_extra_missing(monkeypatch, tmp_path):
    source = tmp_path / "scan.jpg"
    source.write_bytes(b"not a real image")
    monkeypatch.setattr(
        metadata_scrub_mod,
        "_missing_dependencies",
        lambda requirements: ["piexif"],
    )

    with pytest.raises(MissingDependencyError) as excinfo:
        scrub_metadata(source)

    message = str(excinfo.value)
    assert "piexif" in message
    assert "openmed[multimodal]" in message


def test_jpeg_exif_xmp_are_removed_and_report_is_audit_safe(tmp_path):
    image_path = tmp_path / "synthetic_exif.jpg"
    _write_jpeg_with_phi_metadata(image_path)

    before = verify_metadata(image_path)
    assert {
        "image.exif.0th.artist",
        "image.exif.exif.datetimeoriginal",
        "image.exif.gps.gpslatitude",
        "image.xmp.packet",
    }.issubset({finding.identifier for finding in before.residual_metadata})

    result = scrub_metadata(image_path)

    assert isinstance(result, MetadataScrubResult)
    assert result.clean is True
    assert result.residual_report.residual_metadata == ()
    assert verify_metadata(image_path).clean is True

    removed_ids = {finding.identifier for finding in result.removed_metadata}
    assert "image.exif.0th.artist" in removed_ids
    assert "image.exif.gps.gpslatitude" in removed_ids
    assert "image.xmp.packet" in removed_ids

    audit_report = result.to_audit_report()
    assert audit_report["type"] == "metadata_scrub"
    assert audit_report["removed_count"] >= 4
    serialized = str(audit_report)
    assert "Dr Jane Smith" not in serialized
    assert "2026-01-02" not in serialized


def test_allowlisted_image_icc_profile_is_preserved(tmp_path):
    image_path = tmp_path / "synthetic_profile.jpg"
    icc_profile = b"synthetic-openmed-color-profile"
    _write_jpeg_with_phi_metadata(image_path, icc_profile=icc_profile)

    result = scrub_metadata(image_path, allowlist={"image.icc_profile"})
    report = verify_metadata(image_path, allowlist={"image.icc_profile"})

    assert result.clean is True
    assert report.clean is True
    assert {finding.identifier for finding in report.preserved_metadata} == {
        "image.icc_profile"
    }

    image_module, _piexif = _require_image_deps()
    with image_module.open(image_path) as image:
        assert image.info["icc_profile"] == icc_profile


def test_pdf_info_and_xmp_properties_are_cleared(tmp_path):
    pdf_path = tmp_path / "synthetic_meta.pdf"
    pikepdf = _write_pdf_with_phi_metadata(pdf_path)

    before = verify_metadata(pdf_path)
    assert {
        "pdf.info.author",
        "pdf.info.title",
        "pdf.info.subject",
        "pdf.xmp.description.creator",
    }.issubset({finding.identifier for finding in before.residual_metadata})

    result = scrub_metadata(pdf_path)

    assert result.clean is True
    assert verify_metadata(pdf_path).clean is True
    with pikepdf.open(pdf_path) as pdf:
        assert not dict(pdf.docinfo)
        assert pdf.Root.get("/Metadata") is None


def test_docx_core_app_and_custom_properties_are_cleared(tmp_path):
    docx_path = tmp_path / "synthetic_meta.docx"
    _write_docx_with_phi_metadata(docx_path)

    before = verify_metadata(docx_path)
    assert {
        "docx.core.creator",
        "docx.core.lastmodifiedby",
        "docx.app.company",
        "docx.custom.patientmrn",
    }.issubset({finding.identifier for finding in before.residual_metadata})

    result = scrub_metadata(docx_path)

    assert result.clean is True
    assert verify_metadata(docx_path).clean is True
    doc_props = _read_docx_props_text(docx_path)
    assert "Dr Jane Smith" not in doc_props
    assert "Nurse Example" not in doc_props
    assert "OpenMed Clinic" not in doc_props
    assert "MRN 12345" not in doc_props


def _require_image_deps():
    image_module = pytest.importorskip("PIL.Image")
    piexif = pytest.importorskip("piexif")
    return image_module, piexif


def _write_jpeg_with_phi_metadata(
    path: Path,
    *,
    icc_profile: bytes | None = None,
) -> None:
    image_module, piexif = _require_image_deps()
    image = image_module.new("RGB", (16, 12), color=(255, 255, 255))
    exif = {
        "0th": {
            piexif.ImageIFD.Artist: "Dr Jane Smith",
            piexif.ImageIFD.DateTime: "2026:01:02 03:04:05",
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2026:01:02 03:04:05",
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: ((37, 1), (46, 1), (30, 1)),
        },
    }
    save_kwargs = {"exif": piexif.dump(exif)}
    if icc_profile is not None:
        save_kwargs["icc_profile"] = icc_profile
    image.save(path, format="JPEG", **save_kwargs)
    _inject_jpeg_xmp(path, XMP_PACKET)


def _inject_jpeg_xmp(path: Path, xmp_packet: bytes) -> None:
    data = path.read_bytes()
    assert data[:2] == b"\xff\xd8"
    marker_payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp_packet
    segment = (
        b"\xff\xe1" + (len(marker_payload) + 2).to_bytes(2, "big") + marker_payload
    )
    path.write_bytes(data[:2] + segment + data[2:])


def _write_pdf_with_phi_metadata(path: Path):
    pikepdf = pytest.importorskip("pikepdf")
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(72, 72))
    pdf.docinfo["/Author"] = "Dr Jane Smith"
    pdf.docinfo["/Title"] = "John Doe visit"
    pdf.docinfo["/Subject"] = "MRN 12345"
    metadata = pdf.make_stream(XMP_PACKET)
    metadata.Type = pikepdf.Name("/Metadata")
    metadata.Subtype = pikepdf.Name("/XML")
    pdf.Root.Metadata = metadata
    pdf.save(path)
    return pikepdf


def _write_docx_with_phi_metadata(path: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Synthetic clinical note with no metadata PHI in body.")
    core = document.core_properties
    core.author = "Dr Jane Smith"
    core.last_modified_by = "Nurse Example"
    core.title = "John Doe visit"
    core.subject = "MRN 12345"
    document.save(path)
    _rewrite_docx_member(path, "docProps/app.xml", _set_docx_company)
    _rewrite_docx_member(
        path,
        "docProps/custom.xml",
        lambda _existing: _custom_docx_properties_xml(),
        create=True,
    )


def _rewrite_docx_member(
    path: Path,
    member: str,
    transform,
    *,
    create: bool = False,
) -> None:
    tmp_path = path.with_suffix(".tmp.docx")
    wrote_member = False
    with (
        zipfile.ZipFile(path) as source,
        zipfile.ZipFile(
            tmp_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as target,
    ):
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == member:
                data = transform(data)
                wrote_member = True
            target.writestr(info, data)
        if create and not wrote_member:
            target.writestr(member, transform(b""))
    os.replace(tmp_path, path)


def _set_docx_company(xml_bytes: bytes) -> bytes:
    namespace = (
        "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
    )
    root = ElementTree.fromstring(xml_bytes)
    company = root.find(f"{{{namespace}}}Company")
    if company is None:
        company = ElementTree.SubElement(root, f"{{{namespace}}}Company")
    company.text = "OpenMed Clinic"
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _custom_docx_properties_xml() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <property fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}" pid="2" name="PatientMRN">
    <vt:lpwstr>MRN 12345</vt:lpwstr>
  </property>
</Properties>"""


def _read_docx_props_text(path: Path) -> str:
    text_parts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in ("docProps/core.xml", "docProps/app.xml", "docProps/custom.xml"):
            try:
                text_parts.append(archive.read(member).decode("utf-8"))
            except KeyError:
                continue
    return "\n".join(text_parts)
