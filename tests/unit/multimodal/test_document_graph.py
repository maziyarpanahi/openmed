"""Focused tests for the synthetic clinical document graph contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.multimodal.document_graph import (
    build_document_graph,
    extract_pdf_graph,
)
from openmed.multimodal.exceptions import (
    EncryptedDocumentError,
    MalformedDocumentError,
)

FIXTURE = (
    Path(__file__).parents[2] / "fixtures" / "documents" / "synthetic_form_blocks.json"
)


def _fixture_blocks() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_synthetic_columns_tables_and_offsets_are_deterministic() -> None:
    first = build_document_graph(_fixture_blocks())
    second = build_document_graph(_fixture_blocks())

    assert first.text == second.text
    assert first.text.splitlines() == [
        "Left column one",
        "Left column two",
        "Field: SYNTH-FIELD-01",
        "Result",
        "Value",
        "Glucose",
        "5.2",
        "Right column one",
        "Right column two",
    ]
    assert len(first.columns) == 2
    assert len(first.tables) == 1
    assert [(field.key, field.value) for field in first.form_fields] == [
        ("Field", "SYNTH-FIELD-01")
    ]
    assert [(cell.row, cell.column) for cell in first.tables[0].cells] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    start = first.text.index("5.2")
    end = start + len("5.2")
    regions = first.project_span(start, end)
    assert len(regions) == 1
    assert regions[0].page == 0
    assert regions[0].bbox == (120.0, 120.0, 190.0, 130.0)
    assert first.text[start:end] == "5.2"
    assert regions[0].start == start
    assert regions[0].end == end


def test_graph_preserves_explicit_form_key_value_regions() -> None:
    graph = build_document_graph(
        [
            {
                "kind": "form_field",
                "key": "Patient ID",
                "value": "SYNTH-PATIENT-01",
                "page": 1,
                "bbox": (10, 20, 200, 30),
                "value_bbox": (90, 20, 200, 30),
            }
        ]
    )

    assert len(graph.form_fields) == 1
    field = graph.form_fields[0]
    assert graph.text[field.value_start : field.value_end] == "SYNTH-PATIENT-01"
    assert field.value_region().page == 1
    assert field.value_region().bbox == (90.0, 20.0, 200.0, 30.0)


def test_malformed_pdf_fails_closed_before_optional_parser(tmp_path: Path) -> None:
    path = tmp_path / "malformed.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(MalformedDocumentError):
        extract_pdf_graph(path)


def test_encrypted_pdf_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "encrypted.pdf"
    path.write_bytes(b"%PDF-1.7 synthetic")

    fake_pdf = SimpleNamespace(
        is_encrypted=True,
        pages=(),
    )

    class _Context:
        def __enter__(self):
            return fake_pdf

        def __exit__(self, *_args):
            return False

    monkeypatch.setitem(
        sys.modules, "pdfplumber", SimpleNamespace(open=lambda _: _Context())
    )

    with pytest.raises(EncryptedDocumentError):
        extract_pdf_graph(path)


def test_pdf_words_use_column_reading_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "columns.pdf"
    path.write_bytes(b"%PDF-1.7 synthetic")

    words = [
        {"text": "right", "x0": 320, "top": 20, "x1": 350, "bottom": 30},
        {"text": "left", "x0": 40, "top": 20, "x1": 65, "bottom": 30},
        {"text": "right-two", "x0": 320, "top": 42, "x1": 380, "bottom": 52},
        {"text": "left-two", "x0": 40, "top": 42, "x1": 80, "bottom": 52},
    ]

    class _Page:
        width = 600
        height = 800

        def extract_words(self, **_kwargs):
            return words

    class _Pdf:
        is_encrypted = False
        pages = (_Page(),)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setitem(
        sys.modules, "pdfplumber", SimpleNamespace(open=lambda _: _Pdf())
    )

    graph = extract_pdf_graph(path)
    assert graph.text.splitlines() == ["left", "left-two", "right", "right-two"]
