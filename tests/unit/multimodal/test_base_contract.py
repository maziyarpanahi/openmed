"""Tests for the multimodal ingest/redact contract.

Covers the ``ExtractedDocument`` text<->offset round-trip and the
extension-based ``redact_document`` dispatcher with its lazy handler registry.
"""

from __future__ import annotations

import pytest

import openmed.multimodal.base as base
from openmed.multimodal import (
    ExtractedDocument,
    SourceSpan,
    redact_document,
    register_handler,
)
from openmed.multimodal.exceptions import UnsupportedDocumentError


@pytest.fixture
def clean_registry():
    """Snapshot and restore the handler registry around each test."""
    saved = dict(base._HANDLERS)
    base._HANDLERS.clear()
    try:
        yield
    finally:
        base._HANDLERS.clear()
        base._HANDLERS.update(saved)


@pytest.fixture
def deps_present(monkeypatch):
    """Pretend the multimodal extra is installed."""
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])


BLOCKS = [
    {"text": "Patient John Doe", "page": 1, "bbox": (0.0, 0.0, 1.0, 0.1)},
    {"text": "MRN 12345", "page": 1, "bbox": (0.0, 0.2, 1.0, 0.3)},
    {"text": "Discharged 2026-01-02", "page": 2},
]


class TestExtractedDocument:
    def test_from_blocks_concatenates_text(self):
        doc = ExtractedDocument.from_blocks(BLOCKS)
        assert doc.text == "Patient John Doe\nMRN 12345\nDischarged 2026-01-02"
        assert len(doc.spans) == 3

    def test_spans_round_trip_text(self):
        doc = ExtractedDocument.from_blocks(BLOCKS)
        for block, span in zip(BLOCKS, doc.spans):
            assert doc.text[span.start : span.end] == block["text"]
            assert doc.text_for(span) == block["text"]

    def test_location_at_resolves_source_location(self):
        doc = ExtractedDocument.from_blocks(BLOCKS)
        # An offset inside the second block resolves to page 1 with its bbox.
        offset = doc.text.index("MRN")
        span = doc.location_at(offset)
        assert span is not None
        assert span.page == 1
        assert span.bbox == (0.0, 0.2, 1.0, 0.3)
        assert doc.text_for(span) == "MRN 12345"

    def test_location_at_out_of_range_returns_none(self):
        doc = ExtractedDocument.from_blocks(BLOCKS)
        assert doc.location_at(-1) is None
        assert doc.location_at(len(doc.text) + 5) is None

    def test_source_span_is_frozen(self):
        span = SourceSpan(start=0, end=1)
        with pytest.raises(Exception):
            span.start = 5  # type: ignore[misc]


class TestRedactDocumentDispatcher:
    def test_dispatches_to_registered_handler(self, clean_registry, deps_present):
        received = {}

        def handler(path, *, policy=None, models=None):
            received["path"] = path
            received["policy"] = policy
            received["models"] = models
            return ExtractedDocument.from_blocks([{"text": "ok"}])

        register_handler(".txt", handler)
        doc = redact_document("note.txt", policy="hipaa_safe_harbor", models="m1")

        assert isinstance(doc, ExtractedDocument)
        assert received["path"] == "note.txt"
        assert received["policy"] == "hipaa_safe_harbor"
        assert received["models"] == "m1"

    def test_extension_match_is_case_insensitive(self, clean_registry, deps_present):
        register_handler(
            ".txt", lambda path, **_: ExtractedDocument.from_blocks([{"text": "x"}])
        )
        assert isinstance(redact_document("NOTE.TXT"), ExtractedDocument)

    def test_register_handler_accepts_multiple_extensions(
        self, clean_registry, deps_present
    ):
        register_handler(
            [".htm", ".html"],
            lambda path, **_: ExtractedDocument.from_blocks([{"text": "x"}]),
        )
        assert isinstance(redact_document("a.htm"), ExtractedDocument)
        assert isinstance(redact_document("a.html"), ExtractedDocument)

    def test_unknown_extension_raises_unsupported(self, clean_registry, deps_present):
        with pytest.raises(UnsupportedDocumentError):
            redact_document("mystery.xyz")
