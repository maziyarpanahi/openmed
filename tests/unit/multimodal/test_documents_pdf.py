"""Tests for PDF text extraction and bbox projection."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import openmed.multimodal.base as base
from openmed.multimodal import extract_pdf, project_text_spans, redact_document


def _synthetic_pdf_bytes() -> bytes:
    stream = (
        b"BT\n"
        b"/F1 12 Tf\n"
        b"72 720 Td\n"
        b"(Patient John Doe) Tj\n"
        b"0 -18 Td\n"
        b"(MRN 12345) Tj\n"
        b"ET\n"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
    ]

    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")

    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(payload)


class _FakePage:
    def __init__(self, words):
        self._words = words

    def extract_words(self, **kwargs):
        return list(self._words)


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@pytest.fixture
def fake_pdfplumber(monkeypatch):
    pages = [
        _FakePage(
            [
                {
                    "text": "Patient",
                    "x0": 10,
                    "top": 20,
                    "x1": 42,
                    "bottom": 30,
                },
                {
                    "text": "John",
                    "x0": 45,
                    "top": 20,
                    "x1": 65,
                    "bottom": 30,
                },
                {
                    "text": "Doe",
                    "x0": 68,
                    "top": 20,
                    "x1": 84,
                    "bottom": 30,
                },
            ]
        ),
        _FakePage(
            [
                {
                    "text": "MRN",
                    "x0": 10,
                    "top": 50,
                    "x1": 30,
                    "bottom": 60,
                },
                {
                    "text": "12345",
                    "x0": 34,
                    "top": 50,
                    "x1": 70,
                    "bottom": 60,
                },
            ]
        ),
    ]
    module = SimpleNamespace(open=lambda path: _FakePdf(pages))
    monkeypatch.setitem(sys.modules, "pdfplumber", module)
    return module


@pytest.fixture
def multimodal_deps_present(monkeypatch):
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])


def test_extract_pdf_maps_words_to_offsets_and_bboxes(fake_pdfplumber):
    doc = extract_pdf("synthetic_phi.pdf")

    assert doc.text == "Patient John Doe\nMRN 12345"
    assert doc.metadata["format"] == "pdf"
    assert doc.metadata["page_count"] == 2
    assert doc.metadata["word_count"] == 5

    john = doc.location_at(doc.text.index("John"))
    assert john is not None
    assert john.page == 0
    assert john.bbox == (45.0, 20.0, 65.0, 30.0)
    assert doc.text_for(john) == "John"

    mrn = doc.location_at(doc.text.index("MRN"))
    assert mrn is not None
    assert mrn.page == 1
    assert mrn.bbox == (10.0, 50.0, 30.0, 60.0)


def test_synthetic_pdf_fixture_extracts_with_real_pdfplumber(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "synthetic_phi.pdf"
    path.write_bytes(_synthetic_pdf_bytes())

    doc = extract_pdf(path)
    assert "Patient John Doe" in doc.text
    start = doc.text.index("John")
    end = doc.text.index("Doe") + len("Doe")

    rectangles = project_text_spans(doc, [(start, end)])

    assert rectangles
    assert rectangles[0].page == 0
    assert rectangles[0].bbox[0] < rectangles[0].bbox[2]
    assert rectangles[0].bbox[1] < rectangles[0].bbox[3]


def test_project_text_spans_merges_same_line_words(fake_pdfplumber):
    doc = extract_pdf("synthetic_phi.pdf")
    start = doc.text.index("John")
    end = doc.text.index("Doe") + len("Doe")

    rectangles = project_text_spans(doc, [(start, end)])

    assert len(rectangles) == 1
    rectangle = rectangles[0]
    assert rectangle.page == 0
    assert rectangle.start == start
    assert rectangle.end == end
    assert rectangle.bbox == (45.0, 20.0, 84.0, 30.0)
    assert rectangle.metadata["source_span_count"] == 2
    assert "text_sha256" in rectangle.metadata


def test_redact_document_pdf_reports_detected_rectangles(
    fake_pdfplumber, multimodal_deps_present
):
    def detector(text, *, lang=None):
        return {
            "entities": [
                {
                    "start": text.index("John"),
                    "end": text.index("Doe") + len("Doe"),
                    "label": "PERSON",
                    "confidence": 0.93,
                }
            ]
        }

    doc = redact_document("synthetic_phi.pdf", models={"detector": detector}, lang="en")

    assert doc.text == "Patient John Doe\nMRN 12345"
    assert doc.metadata["detected_span_count"] == 1
    rectangles = doc.metadata["redaction_rectangles"]
    assert len(rectangles) == 1
    assert rectangles[0]["start"] == 8
    assert rectangles[0]["end"] == 16
    assert rectangles[0]["page"] == 0
    assert rectangles[0]["bbox"] == (45.0, 20.0, 84.0, 30.0)
    assert rectangles[0]["label"] == "PERSON"
    assert rectangles[0]["confidence"] == 0.93
    assert rectangles[0]["metadata"]["source_span_count"] == 2
    assert "text_sha256" in rectangles[0]["metadata"]
