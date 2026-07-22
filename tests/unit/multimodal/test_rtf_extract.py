"""Tests for RTF text extraction with source offset maps."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.multimodal import ExtractedDocument, extract_rtf, redact_document
from openmed.multimodal.exceptions import UnsupportedDocumentError

# A synthetic clinical-note-style RTF, hand-built to exercise: font/color
# tables, document \info metadata, \pard immediately following \par (should
# not double the newline), a cp1252 hex-escaped curly quote, a \uN Unicode
# escape with a hex fallback run, a hyperlink field (instruction skipped,
# result kept), and a table row linearized with \cell/\row.
SYNTHETIC_NOTE = (
    r"{\rtf1\ansi\ansicpg1252\deff0"
    r"{\fonttbl{\f0 Times New Roman;}}"
    r"{\colortbl;\red0\green0\blue0;}"
    r"{\info{\author Jane Roe}{\title Visit Note}}"
    r"\pard Patient: John Q. Public"
    r"\par\pard Chief complaint: pt reports feeling \'93better\'94 today."
    r"\par\pard Caf\u233\'e9 Clinic follow-up scheduled."
    r"\par\pard See "
    r'{\field{\*\fldinst HYPERLINK "http://example.com/portal"}'
    r"{\fldrslt portal link}} for records."
    r"\par\pard "
    r"{\trowd \intbl Vitals\cell Normal\cell\row}"
    r"\par}"
)


def _write_synthetic_rtf(path: Path, source: str = SYNTHETIC_NOTE) -> Path:
    path.write_bytes(source.encode("latin-1"))
    return path


def _raw_source(path: Path) -> str:
    return path.read_bytes().decode("latin-1")


def test_extract_rtf_reads_body_text_in_order(tmp_path: Path):
    path = _write_synthetic_rtf(tmp_path / "synthetic_note.rtf")

    doc = extract_rtf(path)

    assert doc.text == (
        "Patient: John Q. Public\n"
        "Chief complaint: pt reports feeling “better” today.\n"
        "Café Clinic follow-up scheduled.\n"
        "See portal link for records.\n"
        "Vitals\tNormal\t\n"
        "\n"
    )
    assert doc.metadata["format"] == "rtf"
    assert "source_text" not in doc.metadata


def test_extract_rtf_skips_metadata_and_formatting_tables(tmp_path: Path):
    path = _write_synthetic_rtf(tmp_path / "synthetic_note.rtf")

    doc = extract_rtf(path)

    # Font table, color table, and \info (author/title) metadata are
    # destinations, not document body text, and must not leak into output.
    assert "Times New Roman" not in doc.text
    assert "Jane Roe" not in doc.text
    assert "Visit Note" not in doc.text
    assert "HYPERLINK" not in doc.text
    assert "http://example.com" not in doc.text


def test_par_immediately_followed_by_pard_does_not_double_newline(tmp_path: Path):
    # \pard resets paragraph formatting and very commonly appears right after
    # \par at the start of the next paragraph; it must not itself emit a
    # second newline (a naive parser that maps \pard -> "\n" too would
    # produce a spurious blank line between every paragraph here).
    path = _write_synthetic_rtf(
        tmp_path / "pard.rtf",
        r"{\rtf1\ansi\pard First.\par\pard Second.\par}",
    )

    doc = extract_rtf(path)

    assert doc.text == "First.\nSecond.\n"


def test_cp1252_hex_escapes_decode_via_codepage_not_raw_codepoint(tmp_path: Path):
    # Byte 0x92 in cp1252 is U+2019 (right single quotation mark), not the
    # C1 control character U+0092 that naive chr(0x92) would produce.
    path = _write_synthetic_rtf(
        tmp_path / "apostrophe.rtf",
        r"{\rtf1\ansi\ansicpg1252 It\'92s fine.}",
    )

    doc = extract_rtf(path)

    assert doc.text == "It’s fine."


def test_unicode_escape_with_hex_fallback_is_not_duplicated(tmp_path: Path):
    # Word commonly emits \uN followed by a one-byte \'hh fallback for older
    # readers (here \uc1 is the implicit default). The fallback must be
    # consumed, not appended alongside the real Unicode character.
    path = _write_synthetic_rtf(
        tmp_path / "unicode_fallback.rtf",
        r"{\rtf1\ansi\ansicpg1252 Caf\u233\'e9 today}",
    )

    doc = extract_rtf(path)

    assert doc.text == "Café today"


def test_explicit_uc_controls_fallback_skip_width(tmp_path: Path):
    # \uc2 means each \uN is followed by *two* fallback characters/escapes
    # that must both be skipped, not just one.
    path = _write_synthetic_rtf(
        tmp_path / "uc2.rtf",
        r"{\rtf1\ansi\ansicpg1252\uc2 caf\u233\'3f\'3fs}",
    )

    doc = extract_rtf(path)

    assert doc.text == "cafés"


def test_source_spans_round_trip_to_exact_byte_offsets(tmp_path: Path):
    path = _write_synthetic_rtf(tmp_path / "synthetic_note.rtf")
    doc = extract_rtf(path)
    raw = _raw_source(path)

    offset = doc.text.index("John Q. Public")
    span = doc.location_at(offset)

    assert span is not None
    assert span.metadata["format"] == "rtf"
    source_start = span.metadata["source_start"]
    source_end = span.metadata["source_end"]
    assert raw[source_start:source_end] == "J"

    # Every mapped span must point at a valid, in-order byte range in the
    # original file, and every span's *own* text must be reconstructible
    # from that raw range (directly for plain characters, or via the
    # documented escape it names for escaped ones).
    for span in doc.spans:
        source_start = span.metadata["source_start"]
        source_end = span.metadata["source_end"]
        assert 0 <= source_start < source_end <= len(raw)
        assert doc.text_for(span) != ""


def test_hex_escaped_curly_quote_maps_back_to_its_escape_sequence(tmp_path: Path):
    path = _write_synthetic_rtf(tmp_path / "synthetic_note.rtf")
    doc = extract_rtf(path)
    raw = _raw_source(path)

    offset = doc.text.index("“better”")
    open_quote_span = doc.location_at(offset)

    assert open_quote_span is not None
    source_start = open_quote_span.metadata["source_start"]
    source_end = open_quote_span.metadata["source_end"]
    assert raw[source_start:source_end] == r"\'93"


def test_table_row_linearizes_with_tab_and_row_break(tmp_path: Path):
    path = _write_synthetic_rtf(tmp_path / "synthetic_note.rtf")
    doc = extract_rtf(path)

    assert "Vitals\tNormal\t\n" in doc.text


def test_ignorable_star_destination_is_skipped_even_when_unrecognized(tmp_path: Path):
    # \* marks the following destination as ignorable if the reader does not
    # understand its control word; a compliant parser must skip *any*
    # \*-marked group, not only a hardcoded list of known keywords.
    path = _write_synthetic_rtf(
        tmp_path / "ignorable.rtf",
        r"{\rtf1\ansi{\*\vendorprivatedestination should never appear}Kept text.}",
    )

    doc = extract_rtf(path)

    assert doc.text == "Kept text."
    assert "should never appear" not in doc.text


def test_nested_skip_destination_does_not_leak_after_inner_group_closes(tmp_path: Path):
    # A destination nested inside another destination (e.g. a legacy \pict
    # fallback nested under a modern \shppict wrapper) must keep the outer
    # destination's skip state active after the inner group closes, right up
    # until the outer group itself closes.
    path = _write_synthetic_rtf(
        tmp_path / "nested_skip.rtf",
        r"{\rtf1\ansi{\pict{\*\shppict{\pict\pngblip 4a4a4a}}"
        r"stillinsidepictgarbage}Visible after.}",
    )

    doc = extract_rtf(path)

    assert "stillinsidepictgarbage" not in doc.text
    assert doc.text.strip() == "Visible after."


def test_bin_binary_payload_is_consumed_verbatim(tmp_path: Path):
    # \binN introduces N raw, unescaped bytes. If those bytes happen to
    # contain '{', '}', or '\\' (as in an embedded OLE payload) they must not
    # be interpreted as RTF syntax, or brace counting desyncs for the rest of
    # the document.
    payload = b"{}{}\\not-rtf-syntax"
    source = (
        (r"{\rtf1\ansi{\object{\objdata\bin" + str(len(payload)) + " ").encode(
            "latin-1"
        )
        + payload
        + b"}}Visible after object.}"
    )
    path = tmp_path / "binary.rtf"
    path.write_bytes(source)

    doc = extract_rtf(path)

    assert doc.text.strip() == "Visible after object."


def test_empty_document_extracts_to_empty_text(tmp_path: Path):
    path = _write_synthetic_rtf(tmp_path / "empty.rtf", r"{\rtf1}")

    doc = extract_rtf(path)

    assert doc.text == ""
    assert doc.spans == ()


def test_redact_document_dispatches_rtf(tmp_path: Path):
    path = _write_synthetic_rtf(tmp_path / "synthetic_note.rtf")

    doc = redact_document(path)

    assert isinstance(doc, ExtractedDocument)
    assert "Patient: John Q. Public" in doc.text


def test_non_rtf_file_raises_unsupported_document_error(tmp_path: Path):
    path = tmp_path / "not_rtf.rtf"
    path.write_text("This is plain text, not RTF.", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError, match="does not look like"):
        extract_rtf(path)
