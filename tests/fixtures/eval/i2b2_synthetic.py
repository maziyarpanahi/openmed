"""Synthetic i2b2 de-identification corpora for offline tests.

The real i2b2/DBMI de-identification corpora are DUA-gated and never
committed. Tests that need an i2b2-shaped credentialled directory can call
:func:`write_i2b2_synthetic_corpus`, which writes a small all-synthetic
corpus into a caller-supplied directory and returns the files it created.

The generated files use the same ``deIdi2b2`` XML shape the loader reads
(``TEXT`` plus ``TAGS`` with ``start``/``end``/``TYPE`` attributes). The
2006-style files use lowercase attribute names and the 2014-style files use
uppercase ones, matching the loader's case-insensitive parsing while keeping
the two year directories visibly distinct.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SUPPORTED_SYNTHETIC_YEARS: Final[tuple[int, ...]] = (2006, 2014)
SYNTHETIC_MARKER: Final[str] = "synthetic"


@dataclass(frozen=True)
class SyntheticSpan:
    """One synthetic PHI tag placed into a note's text."""

    category: str
    source_type: str | None
    text: str


@dataclass(frozen=True)
class SyntheticNote:
    """One synthetic note file made of PHI spans joined by filler text."""

    filename: str
    spans: tuple[SyntheticSpan, ...]


_CORPUS: Mapping[int, tuple[SyntheticNote, ...]] = {
    2006: (
        SyntheticNote(
            filename="note-2006-001.xml",
            spans=(
                SyntheticSpan("NAME", "PATIENT", "Synthetic Rowan Vale"),
                SyntheticSpan("DATE", None, "2044-03-12"),
                SyntheticSpan("AGE", None, "91"),
                SyntheticSpan("LOCATION", "STREET", "12 Example Lane"),
                SyntheticSpan("CONTACT", "PHONE", "555-0100"),
            ),
        ),
        SyntheticNote(
            filename="note-2006-002.xml",
            spans=(
                SyntheticSpan("ID", "MEDICALRECORD", "SYN-MRN-2006"),
                SyntheticSpan("NAME", "USERNAME", "synthetic.user"),
                SyntheticSpan("PROFESSION", None, "nurse"),
            ),
        ),
    ),
    2014: (
        SyntheticNote(
            filename="note-2014-001.xml",
            spans=(
                SyntheticSpan("NAME", "PATIENT", "Synthetic Alex Dune"),
                SyntheticSpan("DATE", None, "2045-07-04"),
                SyntheticSpan("AGE", None, "94"),
                SyntheticSpan("LOCATION", "CITY", "Exampleville"),
                SyntheticSpan("LOCATION", "ZIP", "00000"),
                SyntheticSpan("CONTACT", "EMAIL", "synthetic@example.invalid"),
                SyntheticSpan("CONTACT", "URL", "https://example.invalid/record"),
            ),
        ),
        SyntheticNote(
            filename="note-2014-002.xml",
            spans=(
                SyntheticSpan("ID", "SSN", "000-00-0000"),
                SyntheticSpan("LOCATION", "HOSPITAL", "Synthetic General"),
                SyntheticSpan("NAME", "DOCTOR", "Synthetic Dana Lee"),
            ),
        ),
    ),
}


def write_i2b2_synthetic_corpus(
    root: str | Path,
) -> dict[int, tuple[Path, ...]]:
    """Write synthetic 2006- and 2014-style i2b2 XML files under ``root``.

    Args:
        root: Directory that receives the corpus. It is created (including
            missing parents) when absent. Nothing is written outside it.

    Returns:
        Mapping from corpus year to the XML files written for that year.

    Raises:
        ValueError: If ``root`` exists and is not a directory.
    """
    base = Path(root).expanduser()
    if base.exists() and not base.is_dir():
        raise ValueError(f"synthetic i2b2 corpus root must be a directory: {base}")

    written: dict[int, list[Path]] = {}
    for year, notes in _CORPUS.items():
        directory = base / str(year)
        directory.mkdir(parents=True, exist_ok=True)
        for note in notes:
            path = directory / note.filename
            path.write_text(_render_note(note, year=year), encoding="utf-8")
            written.setdefault(year, []).append(path)
    return {year: tuple(paths) for year, paths in written.items()}


def _render_note(note: SyntheticNote, *, year: int) -> str:
    document = ET.Element("deIdi2b2")
    text_node = ET.SubElement(document, "TEXT")
    tags_node = ET.SubElement(document, "TAGS")

    text_parts: list[str] = []
    text_length = 0
    for index, span in enumerate(note.spans, start=1):
        if text_parts:
            text_parts.append(" ")
            text_length += 1
        start = text_length
        text_parts.append(span.text)
        text_length += len(span.text)

        attributes = {
            "start": str(start),
            "end": str(text_length),
            "id": f"P{index}",
            "text": span.text,
        }
        if span.source_type is not None:
            attributes["type"] = span.source_type
        if year == 2014:
            attributes = {name.upper(): value for name, value in attributes.items()}
        ET.SubElement(tags_node, span.category, attributes)

    text_node.text = "".join(text_parts)
    return ET.tostring(document, encoding="unicode")


__all__ = [
    "SUPPORTED_SYNTHETIC_YEARS",
    "SYNTHETIC_MARKER",
    "SyntheticNote",
    "SyntheticSpan",
    "write_i2b2_synthetic_corpus",
]
