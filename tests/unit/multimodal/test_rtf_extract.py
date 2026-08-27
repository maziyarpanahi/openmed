"""Tests for RTF text extraction with source offset maps."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.multimodal import (
    ExtractedDocument,
    extract_rtf,
    redact_document,
    write_redacted_rtf,
)
from openmed.multimodal.exceptions import UnsupportedDocumentError

DISCHARGE_NOTE = (
    "{\\rtf1\\ansi\\ansicpg1252\\deff0\n"
    "{\\fonttbl{\\f0\\froman Times New Roman;}}\n"
    "{\\colortbl;\\red0\\green0\\blue0;}\n"
    "{\\*\\generator Synthetic Writer 1.0;}\n"
    "{\\info{\\author Chart Clerk}{\\title Ignore This}}\n"
    "\\f0\\fs24 Patient {\\b Jane Roe}\\par\n"
    "MRN\\tab A123\\par\n"
    "Cl\\'e9ment reviewed the chart\\par\n"
    "Temp 38\\u176?C at 50\\'25 humidity\\par\n"
    "Braces \\{ and \\} stay literal\\par\n"
    "}\n"
)

EXPECTED_TEXT = (
    "Patient Jane Roe\n"
    "MRN\tA123\n"
    "Clément reviewed the chart\n"
    "Temp 38°C at 50% humidity\n"
    "Braces { and } stay literal"
)


def _write_rtf(path: Path, source: str) -> Path:
    path.write_bytes(source.encode("cp1252"))
    return path


def _raw_for_span(source: str, doc: ExtractedDocument, offset: int) -> str:
    span = doc.location_at(offset)
    assert span is not None
    return source[span.metadata["source_start"] : span.metadata["source_end"]]


def test_extract_rtf_reads_body_text_and_skips_control_groups(tmp_path: Path):
    path = _write_rtf(tmp_path / "synthetic_phi.rtf", DISCHARGE_NOTE)

    doc = extract_rtf(path)

    assert doc.text == EXPECTED_TEXT
    assert "Times New Roman" not in doc.text
    assert "Synthetic Writer" not in doc.text
    assert "Chart Clerk" not in doc.text
    assert "Ignore This" not in doc.text
    assert "\\par" not in doc.text
    assert doc.metadata["format"] == "rtf"
    assert doc.metadata["rtf_version"] == 1
    assert doc.metadata["encoding"] == "cp1252"
    assert doc.metadata["source_path"] == str(path)
    assert "source_text" not in doc.metadata


def test_rtf_source_spans_round_trip_to_source_offsets(tmp_path: Path):
    path = _write_rtf(tmp_path / "synthetic_phi.rtf", DISCHARGE_NOTE)
    doc = extract_rtf(path)

    jane = doc.location_at(doc.text.index("Jane Roe"))
    assert jane is not None
    assert jane.metadata["format"] == "rtf"
    assert _raw_for_span(DISCHARGE_NOTE, doc, doc.text.index("Jane Roe")) == "Jane Roe"

    previous_end = 0
    for span in doc.spans:
        assert 0 <= span.start < span.end <= len(doc.text)
        assert span.start >= previous_end
        previous_end = span.end
        source_start = span.metadata["source_start"]
        source_end = span.metadata["source_end"]
        assert 0 <= source_start < source_end <= len(DISCHARGE_NOTE)
        assert doc.location_at(span.start) == span


def test_rtf_hex_escape_decodes_and_keeps_source_range(tmp_path: Path):
    path = _write_rtf(tmp_path / "synthetic_phi.rtf", DISCHARGE_NOTE)
    doc = extract_rtf(path)

    offset = doc.text.index("é")
    accented = doc.location_at(offset)

    assert accented is not None
    assert doc.text_for(accented) == "é"
    assert _raw_for_span(DISCHARGE_NOTE, doc, offset) == "\\'e9"


def test_rtf_unicode_escape_drops_the_replacement_character(tmp_path: Path):
    path = _write_rtf(tmp_path / "synthetic_phi.rtf", DISCHARGE_NOTE)
    doc = extract_rtf(path)

    offset = doc.text.index("°")

    assert "38°C" in doc.text
    assert "?" not in doc.text
    assert _raw_for_span(DISCHARGE_NOTE, doc, offset) == "\\u176?"


def test_rtf_honors_the_group_scoped_unicode_skip_count(tmp_path: Path):
    source = "{\\rtf1\\ansi\\uc3 Dose 5\\u181???g daily}"
    path = _write_rtf(tmp_path / "unicode_skip.rtf", source)

    doc = extract_rtf(path)

    assert doc.text == "Dose 5µg daily"


def test_rtf_ansicpg_selects_the_declared_codepage(tmp_path: Path):
    source = "{\\rtf1\\ansi\\ansicpg1251 Clinic \\'cc\\'ee\\'f1\\'ea\\'e2\\'e0}"
    path = tmp_path / "cyrillic.rtf"
    path.write_bytes(source.encode("cp1251"))

    doc = extract_rtf(path)

    assert doc.metadata["encoding"] == "cp1251"
    assert doc.text == "Clinic Москва"


def test_rtf_binary_payloads_do_not_leak_into_text(tmp_path: Path):
    source = "{\\rtf1\\ansi Report\\bin6 SECRET end}"
    path = _write_rtf(tmp_path / "binary.rtf", source)

    doc = extract_rtf(path)

    assert doc.text == "Report end"


def test_rtf_skipped_destination_cannot_complete_body_surrogate(tmp_path: Path):
    source = "{\\rtf1\\ansi{\\*\\metadata \\u55357?}\\u56832?Visible}"
    path = _write_rtf(tmp_path / "skipped_surrogate.rtf", source)

    doc = extract_rtf(path)

    assert doc.text == "Visible"


def test_rtf_escaped_line_break_starts_a_new_paragraph(tmp_path: Path):
    source = "{\\rtf1\\ansi Vitals stable\\\r\nDischarge planned}"
    path = _write_rtf(tmp_path / "escaped_break.rtf", source)

    doc = extract_rtf(path)

    assert doc.text == "Vitals stable\nDischarge planned"


def test_redact_document_dispatches_rtf(tmp_path: Path):
    path = _write_rtf(tmp_path / "synthetic_phi.rtf", DISCHARGE_NOTE)

    doc = redact_document(path)

    assert isinstance(doc, ExtractedDocument)
    assert "Patient Jane Roe" in doc.text


def test_write_redacted_rtf_preserves_markup_and_escapes_replacement(tmp_path: Path):
    source = _write_rtf(tmp_path / "source.rtf", DISCHARGE_NOTE)
    output = tmp_path / "redacted.rtf"
    document = extract_rtf(source)
    start = document.text.index("Jane Roe")

    result = write_redacted_rtf(
        source,
        output,
        [(start, start + len("Jane Roe"), "A{B}\\C é")],
    )

    assert result == output
    assert extract_rtf(output).text.startswith("Patient A{B}\\C é\nMRN\tA123")
    raw = output.read_text(encoding="latin-1")
    assert "{\\fonttbl" in raw
    assert "A\\{B\\}\\\\C \\u233?" in raw
    assert "Chart Clerk" in raw


def test_write_redacted_rtf_projects_across_formatting_groups(tmp_path: Path):
    source = _write_rtf(
        tmp_path / "formatted.rtf",
        "{\\rtf1\\ansi Patient Jane {\\b Roe} MRN A123}",
    )
    output = tmp_path / "redacted.rtf"
    document = extract_rtf(source)
    start = document.text.index("Jane Roe")

    write_redacted_rtf(source, output, [(start, start + 8, "[NAME]")])

    assert extract_rtf(output).text == "Patient [NAME] MRN A123"
    assert "{\\b " in output.read_text(encoding="latin-1")


def test_write_redacted_rtf_preserves_partial_atomic_hex_run(tmp_path: Path):
    source = _write_rtf(
        tmp_path / "encoded.rtf",
        r"{\rtf1\ansi Patient \'4a\'61\'6e\'65 Roe}",
    )
    output = tmp_path / "redacted.rtf"
    document = extract_rtf(source)
    start = document.text.index("Jane")

    write_redacted_rtf(source, output, [(start, start + 4, "[NAME]")])

    assert extract_rtf(output).text == "Patient [NAME] Roe"


def test_rtf_handler_runs_detector_and_writes_redacted_copy(tmp_path: Path):
    source = _write_rtf(tmp_path / "source.rtf", DISCHARGE_NOTE)
    output = tmp_path / "redacted.rtf"
    observed: list[tuple[str, str | None]] = []

    def detector(text: str, *, lang: str | None = None):
        observed.append((text, lang))
        start = text.index("Jane Roe")
        return [(start, start + 8, "NAME")]

    document = redact_document(
        source,
        models=detector,
        lang="en",
        policy={"output_path": output},
    )

    assert observed == [(EXPECTED_TEXT, "en")]
    assert extract_rtf(output).text.startswith("Patient [NAME]")
    assert document.metadata["detected_span_count"] == 1
    assert document.metadata["redacted_rtf_path"] == str(output)


def test_write_redacted_rtf_rejects_structural_ranges_and_source_alias(
    tmp_path: Path,
):
    source = _write_rtf(tmp_path / "source.rtf", DISCHARGE_NOTE)
    document = extract_rtf(source)
    line_break = document.text.index("\n")

    with pytest.raises(ValueError, match="structural text"):
        write_redacted_rtf(
            source,
            tmp_path / "redacted.rtf",
            [(line_break - 1, line_break + 1, "x")],
        )
    with pytest.raises(ValueError, match="must differ"):
        write_redacted_rtf(source, source, [(0, 1, "x")])


def test_rtf_write_helper_is_exported_once():
    import openmed.multimodal as multimodal

    assert multimodal.write_redacted_rtf is write_redacted_rtf
    assert multimodal.__all__.count("write_redacted_rtf") == 1


def test_rtf_without_header_raises(tmp_path: Path):
    path = _write_rtf(tmp_path / "plain.rtf", "Patient Jane Roe\n")

    with pytest.raises(UnsupportedDocumentError, match="must start with"):
        extract_rtf(path)


def test_rtf_without_body_text_raises(tmp_path: Path):
    source = "{\\rtf1\\ansi{\\fonttbl{\\f0\\froman Times New Roman;}}\\f0\\fs24\\par}"
    path = _write_rtf(tmp_path / "empty.rtf", source)

    with pytest.raises(UnsupportedDocumentError, match="extractable text"):
        extract_rtf(path)


def test_rtf_unbalanced_groups_do_not_raise(tmp_path: Path):
    source = "{\\rtf1\\ansi Patient Jane Roe}}}\\par Follow-up"
    path = _write_rtf(tmp_path / "unbalanced.rtf", source)

    doc = extract_rtf(path)

    assert doc.text == "Patient Jane Roe\nFollow-up"
