"""Rules-first segmentation for history-family clinical sections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from openmed.clinical.lexicons import normalize_section_header

from .detect import UNSECTIONED_SECTION, SectionSpan, validate_section_spans

_HEADER_DELIMITERS = ":\uff1a\ufe55\ua789"


@cache
def _history_resources() -> tuple[frozenset[str], dict[str, str], tuple[str, ...]]:
    # Import lazily because context imports sections.detect while initializing.
    from openmed.clinical.context import (
        CANONICAL_SECTION_LABELS,
        SECTION_LABEL_ALIASES,
    )

    history_labels = frozenset(
        label for label in CANONICAL_SECTION_LABELS if "history" in label
    )
    aliases = {
        alias
        for alias, label in SECTION_LABEL_ALIASES.items()
        if label in history_labels
    }
    for label in history_labels:
        aliases.add(label.replace("_", " "))
        aliases.update(CANONICAL_SECTION_LABELS[label])
    sorted_aliases = tuple(
        sorted(aliases, key=lambda alias: (-len(alias), alias.casefold()))
    )
    return history_labels, dict(SECTION_LABEL_ALIASES), sorted_aliases


def _alias_pattern(alias: str) -> str:
    return r"[ \t]+".join(re.escape(part) for part in alias.split())


@cache
def _header_pattern() -> re.Pattern[str]:
    _, _, aliases = _history_resources()
    return re.compile(
        rf"^[ \t]*(?:[-*\u2022][ \t]+|\d+[.)][ \t]+)?"
        rf"(?P<header>{'|'.join(_alias_pattern(alias) for alias in aliases)})"
        rf"[ \t]*(?:[{_HEADER_DELIMITERS}][ \t]*(?P<body>.*))?$",
        re.IGNORECASE,
    )


@dataclass(frozen=True)
class _HeaderHit:
    label: str
    start: int


def segment_history_family(text: str) -> tuple[SectionSpan, ...]:
    """Partition *text* at recognized history-family headers.

    Header matching is anchored to complete lines and accepts indentation,
    bullets, numbered headings, inline colon-delimited content, and standalone
    headings separated from their content by a line break. Labels and aliases
    come from :mod:`openmed.clinical.context`, including ``HPI``, ``PMH``,
    ``FH``, and ``SH``. Other clinical section families are deliberately not
    detected by this focused segmenter.

    Args:
        text: Clinical note text. Character offsets refer to this exact string.

    Returns:
        A tuple of contiguous half-open section spans covering the complete
        input. Text before the first recognized header, or all headerless text,
        is labeled ``unsectioned``. Empty input returns an empty tuple.
    """

    if not text:
        return ()

    hits = _header_hits(text)
    if not hits:
        return (SectionSpan(label=UNSECTIONED_SECTION, start=0, end=len(text)),)

    spans: list[SectionSpan] = []
    if hits[0].start > 0:
        spans.append(
            SectionSpan(
                label=UNSECTIONED_SECTION,
                start=0,
                end=hits[0].start,
            )
        )

    for index, hit in enumerate(hits):
        end = hits[index + 1].start if index + 1 < len(hits) else len(text)
        spans.append(SectionSpan(label=hit.label, start=hit.start, end=end))

    result = tuple(spans)
    validate_section_spans(text, result)
    return result


def _header_hits(text: str) -> tuple[_HeaderHit, ...]:
    history_labels, section_label_aliases, _ = _history_resources()
    header_pattern = _header_pattern()
    hits: list[_HeaderHit] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        match = header_pattern.fullmatch(line)
        if match is not None:
            normalized = normalize_section_header(match.group("header"))
            label = section_label_aliases.get(normalized)
            if label in history_labels:
                hits.append(_HeaderHit(label=label, start=cursor))
        cursor += len(raw_line)
    return tuple(hits)


__all__ = ["segment_history_family"]
