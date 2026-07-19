"""Typed discharge-summary section structuring.

This module consumes section spans produced by a clinical section detector and
maps discharge-specific header aliases into four canonical slots. It does not
perform generic section detection or summarize section content. Emitted
``start``/``end`` offsets always index the returned ``content`` in the source
text so callers can retain provenance for downstream review and export.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypedDict

from openmed.clinical.lexicons import normalize_section_header

DischargeSlotName = Literal[
    "admission_diagnosis",
    "hospital_course",
    "discharge_medications",
    "follow_up",
]

REQUIRED_DISCHARGE_SLOTS: tuple[DischargeSlotName, ...] = (
    "admission_diagnosis",
    "hospital_course",
    "discharge_medications",
    "follow_up",
)

_SLOT_ALIASES: Mapping[DischargeSlotName, tuple[str, ...]] = {
    "admission_diagnosis": (
        "admission diagnosis",
        "admission diagnoses",
        "admission dx",
        "admitting diagnosis",
        "admitting diagnoses",
        "admitting dx",
        "diagnosis on admission",
        "diagnoses on admission",
    ),
    "hospital_course": (
        "hospital course",
        "brief hospital course",
        "clinical course",
        "course in hospital",
        "inpatient course",
        "summary of hospital course",
    ),
    "discharge_medications": (
        "discharge medication",
        "discharge medications",
        "discharge med",
        "discharge meds",
        "discharge med list",
        "d/c med",
        "d/c meds",
        "dc med",
        "dc meds",
        "medication at discharge",
        "medications at discharge",
        "medication on discharge",
        "medications on discharge",
        "meds at discharge",
        "meds on discharge",
    ),
    "follow_up": (
        "follow up",
        "followup",
        "follow up instructions",
        "follow up plan",
        "discharge follow up",
        "post discharge follow up",
        "follow up appointments",
    ),
}

_ALIAS_TO_SLOT: dict[str, DischargeSlotName] = {
    normalize_section_header(alias): slot
    for slot, aliases in _SLOT_ALIASES.items()
    for alias in (slot, *aliases)
}
_LABEL_FIELDS = (
    "label",
    "name",
    "section",
    "section_label",
    "section_name",
    "title",
)
_HEADER_DELIMITERS = frozenset(":：﹕꞉")


class DischargeSummarySection(TypedDict):
    """One canonical discharge-summary slot with source provenance."""

    slot: DischargeSlotName
    header: str
    content: str
    start: int
    end: int


class DischargeSummary(TypedDict):
    """Typed discharge-summary section record."""

    admission_diagnosis: DischargeSummarySection | None
    hospital_course: DischargeSummarySection | None
    discharge_medications: DischargeSummarySection | None
    follow_up: DischargeSummarySection | None
    missing_required_slots: list[DischargeSlotName]


def canonical_discharge_slot(header: str) -> DischargeSlotName | None:
    """Normalize a discharge-section header or label to its canonical slot.

    Args:
        header: Raw detector header or section label.

    Returns:
        A canonical discharge slot name, or ``None`` when the value is not a
        recognized discharge-section alias.
    """

    return _ALIAS_TO_SLOT.get(normalize_section_header(header))


def structure_discharge_summary(
    text: str,
    sections: Iterable[Any],
) -> DischargeSummary:
    """Map detected section spans into a typed discharge-summary record.

    A detected section may be a mapping or an object with attributes. The
    canonical slot is resolved from its ``header`` first and then from common
    label fields. Section ranges use half-open source offsets. When supplied,
    ``content_start`` and ``content_end`` delimit the body; otherwise the
    section range is used, with a supplied header excluded when it occurs at
    the beginning of that range. Surrounding whitespace is removed while the
    returned offsets continue to round-trip exactly to ``text``.

    If multiple detected sections map to the same slot, the earliest one in
    source order is retained deterministically.

    Args:
        text: Original discharge-summary text.
        sections: Section detector output containing labels and source ranges.

    Returns:
        Four fixed canonical slots plus required slot names that were not
        detected.

    Raises:
        ValueError: If a recognized section carries invalid source offsets.
    """

    slots: dict[DischargeSlotName, DischargeSummarySection | None] = {
        slot: None for slot in REQUIRED_DISCHARGE_SLOTS
    }
    recognized: list[tuple[int, DischargeSlotName, Any, str]] = []
    for section in sections:
        header = _text_field(section, "header")
        label = _first_text_field(section, _LABEL_FIELDS)
        slot = canonical_discharge_slot(header or "")
        if slot is None:
            slot = canonical_discharge_slot(label or "")
        if slot is None:
            continue

        section_start = _required_offset(section, "start")
        recognized.append((section_start, slot, section, header or label or slot))

    for _, slot, section, header in sorted(recognized, key=lambda item: item[0]):
        if slots[slot] is not None:
            continue
        start, end = _content_range(text, section, header)
        slots[slot] = DischargeSummarySection(
            slot=slot,
            header=header,
            content=text[start:end],
            start=start,
            end=end,
        )

    return DischargeSummary(
        admission_diagnosis=slots["admission_diagnosis"],
        hospital_course=slots["hospital_course"],
        discharge_medications=slots["discharge_medications"],
        follow_up=slots["follow_up"],
        missing_required_slots=[
            slot for slot in REQUIRED_DISCHARGE_SLOTS if slots[slot] is None
        ],
    )


def _field(section: Any, name: str) -> Any:
    if isinstance(section, Mapping):
        return section.get(name)
    return getattr(section, name, None)


def _text_field(section: Any, name: str) -> str | None:
    value = _field(section, name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_text_field(section: Any, names: Iterable[str]) -> str | None:
    for name in names:
        if value := _text_field(section, name):
            return value
    return None


def _required_offset(section: Any, name: str) -> int:
    value = _field(section, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"recognized discharge section {name} must be an integer")
    return value


def _optional_offset(section: Any, name: str, default: int) -> int:
    value = _field(section, name)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"recognized discharge section {name} must be an integer")
    return value


def _content_range(text: str, section: Any, header: str) -> tuple[int, int]:
    section_start = _required_offset(section, "start")
    section_end = _required_offset(section, "end")
    if not 0 <= section_start < section_end <= len(text):
        raise ValueError("recognized discharge section offsets are outside source text")

    explicit_content_start = _field(section, "content_start")
    start = _optional_offset(section, "content_start", section_start)
    end = _optional_offset(section, "content_end", section_end)
    if not section_start <= start <= end <= section_end:
        raise ValueError(
            "recognized discharge content offsets exceed its section range"
        )

    if explicit_content_start is None:
        header_end = _field(section, "header_end")
        if isinstance(header_end, int) and not isinstance(header_end, bool):
            if not section_start <= header_end <= section_end:
                raise ValueError(
                    "recognized discharge header_end exceeds its section range"
                )
            start = header_end
        else:
            header_start = text.find(header, section_start, section_end)
            if header_start >= section_start and not text[
                section_start:header_start
            ].strip(" \t-*•0123456789.)"):
                start = header_start + len(header)
        start = _skip_header_delimiter(text, start, end)

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _skip_header_delimiter(text: str, start: int, end: int) -> int:
    while start < end and text[start] in " \t":
        start += 1
    if start < end and text[start] in _HEADER_DELIMITERS:
        start += 1
    return start


__all__ = [
    "REQUIRED_DISCHARGE_SLOTS",
    "DischargeSlotName",
    "DischargeSummary",
    "DischargeSummarySection",
    "canonical_discharge_slot",
    "structure_discharge_summary",
]
