"""Tests for Markdown/AsciiDoc extraction with raw-source offset maps."""

from __future__ import annotations

from pathlib import Path

import pytest

import openmed.multimodal.base as base
import openmed.multimodal.documents_markdown as documents_markdown
from openmed.multimodal import ExtractedDocument, redact_document
from openmed.multimodal.documents_markdown import (
    extract_asciidoc,
    extract_markdown,
    redact_source_text,
)
from openmed.multimodal.exceptions import MissingDependencyError

FIXTURES = Path(__file__).parent / "fixtures"
MARKDOWN_TARGET = "https://records.example.test/p/Jane_Roe?mrn=MRN-991"


@pytest.fixture
def markup_deps_present(monkeypatch):
    """Pretend optional markup parser dependencies are installed."""
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])
    monkeypatch.setattr(
        documents_markdown, "_ensure_markup_parser_available", lambda flavor: None
    )


def _assert_round_trips_source(doc: ExtractedDocument) -> None:
    source = doc.metadata["source_text"]
    assert isinstance(source, str)
    assert doc.spans
    for span in doc.spans:
        source_start = span.metadata["source_start"]
        source_end = span.metadata["source_end"]
        assert doc.text_for(span) == source[source_start:source_end]
        assert doc.location_at(span.start) == span


def _assert_single_source_rewrite(
    doc: ExtractedDocument,
    normalized_text: str,
    replacement: str,
) -> str:
    source = doc.metadata["source_text"]
    start = doc.text.index(normalized_text)
    end = start + len(normalized_text)
    span = doc.location_at(start)
    assert span is not None
    source_start = span.metadata["source_start"]
    source_end = span.metadata["source_end"]

    redacted = redact_source_text(doc, [(start, end, replacement)])

    assert redacted[:source_start] == source[:source_start]
    assert redacted[source_start : source_start + len(replacement)] == replacement
    assert redacted[source_start + len(replacement) :] == source[source_end:]
    return redacted


def test_markdown_extractor_round_trips_fixture_offsets(markup_deps_present):
    doc = extract_markdown(FIXTURES / "synthetic_phi.md")

    assert doc.metadata["format"] == "markdown"
    assert "Clinical handover" in doc.text
    assert "Jane Roe" in doc.text
    assert MARKDOWN_TARGET in doc.text
    assert "John Doe scan" in doc.text
    assert "John Doe radiograph" in doc.text
    assert "MRN-991" in doc.text
    _assert_round_trips_source(doc)

    target_span = doc.location_at(doc.text.index(MARKDOWN_TARGET))
    assert target_span is not None
    assert target_span.metadata["kind"] == "link_target"

    image_alt = doc.location_at(doc.text.index("John Doe scan"))
    assert image_alt is not None
    assert image_alt.metadata["kind"] == "image_alt"


def test_asciidoc_extractor_round_trips_fixture_offsets(markup_deps_present):
    doc = extract_asciidoc(FIXTURES / "synthetic_phi.adoc")

    assert doc.metadata["format"] == "asciidoc"
    assert "Clinical handover" in doc.text
    assert "Jane Roe" in doc.text
    assert MARKDOWN_TARGET in doc.text
    assert "John Doe scan" in doc.text
    assert "Raw code Patient John Doe" in doc.text
    _assert_round_trips_source(doc)

    target_span = doc.location_at(doc.text.index(MARKDOWN_TARGET))
    assert target_span is not None
    assert target_span.metadata["kind"] == "link_target"


def test_redacting_normalized_span_rewrites_only_source_range(markup_deps_present):
    doc = extract_markdown(FIXTURES / "synthetic_phi.md")

    redacted = _assert_single_source_rewrite(doc, "Jane Roe", "[PERSON]")

    assert "Patient **[PERSON]** reviewed" in redacted
    assert "Patient **Jane Roe** reviewed" not in redacted


def test_link_target_and_image_alt_are_included_and_redactable(markup_deps_present):
    doc = extract_markdown(FIXTURES / "synthetic_phi.md")

    target_redacted = _assert_single_source_rewrite(doc, MARKDOWN_TARGET, "[URL]")
    assert '[chart]([URL] "Jane Roe chart")' in target_redacted

    alt_redacted = _assert_single_source_rewrite(doc, "John Doe scan", "[IMAGE_ALT]")
    assert "![[IMAGE_ALT]](images/John_Doe.png" in alt_redacted

    title_redacted = _assert_single_source_rewrite(
        doc, "John Doe radiograph", "[IMAGE_TITLE]"
    )
    assert '"[IMAGE_TITLE]")' in title_redacted


@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        ("note.md", "# Note\nPatient **Jane Roe**", "Jane Roe"),
        ("note.markdown", "# Note\nPatient **Jane Roe**", "Jane Roe"),
        ("note.adoc", "= Note\nPatient *Jane Roe*", "Jane Roe"),
        ("note.asciidoc", "= Note\nPatient *Jane Roe*", "Jane Roe"),
    ],
)
def test_redact_document_dispatches_markup_extensions(
    tmp_path,
    markup_deps_present,
    filename,
    content,
    expected,
):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")

    doc = redact_document(path)

    assert isinstance(doc, ExtractedDocument)
    assert expected in doc.text


def test_missing_markup_parser_has_actionable_error(monkeypatch):
    def missing_markdown(name):
        if name == "markdown_it":
            return None
        return object()

    monkeypatch.setattr(
        documents_markdown.importlib.util, "find_spec", missing_markdown
    )

    with pytest.raises(MissingDependencyError) as excinfo:
        extract_markdown("# Note")

    message = str(excinfo.value)
    assert "markdown-it-py" in message
    assert "openmed[multimodal]" in message
