"""Tests for EPUB text extraction with source offset maps."""

from __future__ import annotations

import html
import zipfile
from pathlib import Path

import pytest

from openmed.multimodal import ExtractedDocument, extract_epub, redact_document
from openmed.multimodal.exceptions import UnsupportedDocumentError

CHAPTER_ONE = (
    '<html xmlns="http://www.w3.org/1999/xhtml">'
    "<head>"
    "<title>Ignore Jane Roe</title>"
    "<style>.hidden { display: none; }</style>"
    "</head>"
    "<body>"
    "<h1>First</h1>"
    "<p>Patient <strong>Jane Roe</strong></p>"
    "<script>MRN hidden</script>"
    "</body>"
    "</html>"
)

CHAPTER_TWO = (
    '<html xmlns="http://www.w3.org/1999/xhtml">'
    "<body>"
    "<p>MRN <span>A123</span></p>"
    "<p>Discharged &amp; stable</p>"
    "</body>"
    "</html>"
)

SECTION_SOURCES = {
    "OEBPS/chapter1.xhtml": CHAPTER_ONE,
    "OEBPS/chapter2.xhtml": CHAPTER_TWO,
}


def _write_synthetic_epub(path: Path) -> Path:
    manifest_items = "\n".join(
        [
            (
                '<item id="chapter-2" href="chapter2.xhtml" '
                'media-type="application/xhtml+xml"/>'
            ),
            (
                '<item id="chapter-1" href="chapter1.xhtml" '
                'media-type="application/xhtml+xml"/>'
            ),
        ]
    )
    spine_items = "\n".join(
        [
            '<itemref idref="chapter-1"/>',
            '<itemref idref="chapter-2"/>',
        ]
    )
    package = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Synthetic PHI</dc:title>
  </metadata>
  <manifest>
    {manifest_items}
  </manifest>
  <spine>
    {spine_items}
  </spine>
</package>
"""
    container = """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/package.opf"
      media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/package.opf", package)
        archive.writestr("OEBPS/chapter1.xhtml", CHAPTER_ONE)
        archive.writestr("OEBPS/chapter2.xhtml", CHAPTER_TWO)
    return path


def _raw_for_span(doc: ExtractedDocument, offset: int) -> str:
    span = doc.location_at(offset)
    assert span is not None
    source = SECTION_SOURCES[span.metadata["section_href"]]
    return source[span.metadata["source_start"] : span.metadata["source_end"]]


def test_extract_epub_reads_spine_order_and_preserves_sections(tmp_path: Path):
    path = _write_synthetic_epub(tmp_path / "synthetic_phi.epub")

    doc = extract_epub(path)

    expected_first = "First\nPatient Jane Roe"
    expected_second = "MRN A123\nDischarged & stable"
    assert doc.text == f"{expected_first}\n{expected_second}"
    assert "<strong>" not in doc.text
    assert "Ignore Jane Roe" not in doc.text
    assert "MRN hidden" not in doc.text
    assert doc.metadata["format"] == "epub"
    assert doc.metadata["package_path"] == "OEBPS/package.opf"
    assert doc.metadata["section_count"] == 2
    assert "source_text" not in doc.metadata

    sections = doc.metadata["sections"]
    assert sections == [
        {
            "index": 0,
            "id": "chapter-1",
            "href": "OEBPS/chapter1.xhtml",
            "start": 0,
            "end": len(expected_first),
        },
        {
            "index": 1,
            "id": "chapter-2",
            "href": "OEBPS/chapter2.xhtml",
            "start": len(expected_first) + 1,
            "end": len(doc.text),
        },
    ]


def test_epub_source_spans_map_back_to_xhtml_ranges(tmp_path: Path):
    path = _write_synthetic_epub(tmp_path / "synthetic_phi.epub")
    doc = extract_epub(path)

    jane = doc.location_at(doc.text.index("Jane Roe"))
    assert jane is not None
    assert jane.metadata["format"] == "epub"
    assert jane.metadata["section_index"] == 0
    assert jane.metadata["section_id"] == "chapter-1"
    assert jane.metadata["section_href"] == "OEBPS/chapter1.xhtml"
    assert _raw_for_span(doc, doc.text.index("Jane Roe")) == "Jane Roe"

    for span in doc.spans:
        source = SECTION_SOURCES[span.metadata["section_href"]]
        raw = source[span.metadata["source_start"] : span.metadata["source_end"]]
        assert html.unescape(raw) == doc.text_for(span)
        assert doc.location_at(span.start) == span


def test_epub_character_references_keep_source_ranges(tmp_path: Path):
    path = _write_synthetic_epub(tmp_path / "synthetic_phi.epub")
    doc = extract_epub(path)

    ampersand = doc.location_at(doc.text.index("&"))

    assert ampersand is not None
    assert doc.text_for(ampersand) == "&"
    assert _raw_for_span(doc, doc.text.index("&")) == "&amp;"


def test_redact_document_dispatches_epub(tmp_path: Path):
    path = _write_synthetic_epub(tmp_path / "synthetic_phi.epub")

    doc = redact_document(path)

    assert isinstance(doc, ExtractedDocument)
    assert "Patient Jane Roe" in doc.text


def test_epub_without_supported_spine_text_raises(tmp_path: Path):
    path = tmp_path / "image_only.epub"
    container = """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/package.opf"
      media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    package = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="cover" href="cover.png" media-type="image/png"/>
  </manifest>
  <spine>
    <itemref idref="cover"/>
  </spine>
</package>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/package.opf", package)
        archive.writestr("OEBPS/cover.png", b"not an image")

    with pytest.raises(UnsupportedDocumentError, match="supported XHTML or HTML"):
        extract_epub(path)


def test_encrypted_epub_entries_raise(tmp_path: Path):
    path = _write_synthetic_epub(tmp_path / "encrypted.epub")
    payload = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        header = payload.find(signature)
        assert header != -1
        payload[header + flag_offset] |= 0x1
    path.write_bytes(payload)

    with pytest.raises(UnsupportedDocumentError, match="Encrypted EPUB"):
        extract_epub(path)
