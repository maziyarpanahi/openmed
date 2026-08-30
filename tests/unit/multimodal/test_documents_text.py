"""Tests for fixed-width plaintext extraction and write-back."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import openmed.multimodal as multimodal
from openmed.multimodal import base
from openmed.multimodal.documents_text import extract_text, write_redacted_text


def _write_text(path: Path, text: str) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


def test_extract_text_preserves_mixed_newlines_and_line_geometry(tmp_path: Path):
    raw = "NAME        MRN\r\nJane Roe    A123\nJosé Li     B456\r"
    source = _write_text(tmp_path / "report.txt", raw)

    document = extract_text(source)

    assert document.text == raw
    assert document.metadata["format"] == "text"
    assert document.metadata["encoding"] == "utf-8"
    assert document.metadata["source_path"] == str(source)
    assert document.metadata["lines"] == (
        {
            "line": 0,
            "start": 0,
            "content_end": 15,
            "end": 17,
            "columns": 15,
            "newline": "\r\n",
        },
        {
            "line": 1,
            "start": 17,
            "content_end": 33,
            "end": 34,
            "columns": 16,
            "newline": "\n",
        },
        {
            "line": 2,
            "start": 34,
            "content_end": 50,
            "end": 51,
            "columns": 16,
            "newline": "\r",
        },
    )
    jane = document.location_at(document.text.index("Jane"))
    assert jane is not None
    assert jane.metadata["line"] == 1
    assert jane.metadata["source_map_mode"] == "linear"


def test_write_redacted_text_preserves_fixed_width_columns(tmp_path: Path):
    raw = "NAME        MRN\nJane Roe    A123\nFollow-up   stable\n"
    source = _write_text(tmp_path / "report.txt", raw)
    output = tmp_path / "redacted.txt"
    start = raw.index("Jane Roe")
    mrn_column = raw.splitlines()[1].index("A123")

    result = write_redacted_text(
        source,
        output,
        [(start, start + len("Jane Roe"), "[NAME]")],
    )

    redacted = output.read_text(encoding="utf-8")
    assert result == output
    assert redacted.splitlines()[1] == "[NAME]      A123"
    assert redacted.splitlines()[1].index("A123") == mrn_column
    assert len(redacted.splitlines()[1]) == len(raw.splitlines()[1])
    assert redacted.splitlines()[2] == "Follow-up   stable"


def test_write_redacted_text_truncates_long_masks_to_source_width(tmp_path: Path):
    source = _write_text(tmp_path / "short.txt", "Li  123\n")
    output = tmp_path / "redacted.txt"

    write_redacted_text(source, output, [(0, 2, "[PERSON]")])

    assert output.read_text(encoding="utf-8") == "[P  123\n"


def test_text_handler_runs_detector_and_writes_redacted_copy(tmp_path: Path):
    source = _write_text(tmp_path / "report.txt", "Patient     MRN\nJane Roe    A123\n")
    output = tmp_path / "redacted.txt"
    observed: list[tuple[str, str | None]] = []

    def detector(text: str, *, lang: str | None = None):
        observed.append((text, lang))
        start = text.index("Jane Roe")
        return {"entities": [{"start": start, "end": start + 8, "label": "NAME"}]}

    document = multimodal.redact_document(
        source,
        models={"detector": detector},
        lang="en",
        policy={"output_path": output},
    )

    assert observed == [("Patient     MRN\nJane Roe    A123\n", "en")]
    assert output.read_text(encoding="utf-8").splitlines()[1] == "[PERSON]    A123"
    assert document.metadata["detected_span_count"] == 1
    assert document.metadata["redacted_text_path"] == str(output)
    assert base._HANDLERS[".txt"][-1].requires_multimodal is False


def test_text_handler_without_entities_does_not_create_output(tmp_path: Path):
    source = _write_text(tmp_path / "report.txt", "No identifiers\n")
    output = tmp_path / "unused.txt"

    document = multimodal.redact_document(
        source,
        models=lambda text: [],
        policy={"output_path": output},
    )

    assert document.metadata["detected_span_count"] == 0
    assert not output.exists()


def test_text_handler_accepts_object_entities_and_positional_only_lang(tmp_path: Path):
    source = _write_text(tmp_path / "report.txt", "Jane Roe\n")
    output = tmp_path / "redacted.txt"

    class Entity:
        start = 0
        end = 8
        entity_type = "PERSON"

    def detector(text: str, lang: str | None = None, /):
        assert text == "Jane Roe\n"
        assert lang == "da"
        return [Entity(), Entity()]

    document = multimodal.redact_document(
        source,
        models=detector,
        lang="da",
        policy={"output_path": output},
    )

    assert document.metadata["detected_span_count"] == 1
    assert output.read_text(encoding="utf-8") == "[PERSON]\n"


def test_detector_cannot_supply_raw_replacement_or_untrusted_label(tmp_path: Path):
    sentinel = "SYNTHETIC-RAW-PATIENT-VALUE"
    source = _write_text(tmp_path / "report.txt", f"{sentinel}\n")
    output = tmp_path / "redacted.txt"

    def detector(text: str):
        return [
            {
                "start": 0,
                "end": len(sentinel),
                "label": sentinel,
                "replacement": sentinel,
            }
        ]

    multimodal.redact_document(
        source,
        models=detector,
        policy={"output_path": output},
    )

    rendered = output.read_text(encoding="utf-8")
    assert sentinel not in rendered
    assert rendered.rstrip() == "[PHI]"


@pytest.mark.parametrize("offset", [True, 0.5])
def test_detector_offsets_must_be_integral(tmp_path: Path, offset: object):
    source = _write_text(tmp_path / "report.txt", "Jane Roe\n")

    with pytest.raises(ValueError, match="invalid detector entity offsets"):
        multimodal.redact_document(
            source,
            models=lambda text: [{"start": offset, "end": 8, "label": "NAME"}],
        )


def test_text_writer_rejects_line_crossings_and_source_aliases(tmp_path: Path):
    source = _write_text(tmp_path / "report.txt", "Jane\nRoe\n")
    output = tmp_path / "redacted.txt"

    with pytest.raises(ValueError, match="cannot cross line endings"):
        write_redacted_text(source, output, [(0, 8, "[NAME]")])
    with pytest.raises(ValueError, match="must differ"):
        write_redacted_text(source, source, [(0, 4, "mask")])
    hardlink = tmp_path / "hardlink.txt"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="must not alias"):
        write_redacted_text(source, hardlink, [(0, 4, "mask")])
    assert source.read_text(encoding="utf-8") == "Jane\nRoe\n"


def test_text_exports_are_public_and_unique():
    assert multimodal.extract_text is extract_text
    assert multimodal.write_redacted_text is write_redacted_text
    assert multimodal.__all__.count("extract_text") == 1
    assert multimodal.__all__.count("write_redacted_text") == 1
