"""De-identified narrative extraction for common HL7 v2 messages.

The extractor builds on :mod:`openmed.interop.hl7v2` rather than parsing pipe
messages independently. Structured identifiers are redacted first, the
rendered narrative is passed through the text de-identification pipeline, and
the final visible value spans retain segment/field provenance.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from openmed.interop.hl7v2 import (
    FieldKey,
    HL7Message,
    HL7Segment,
    HL7V2Encoding,
    TextDeidentifier,
    parse_hl7v2,
    redact_hl7v2,
)

NarrativeMode = Literal["flat", "sectioned"]

_SECTION_ORDER = (
    "Message",
    "Patient",
    "Encounter",
    "Orders",
    "Observations",
    "Notes",
)

_SEX_LABELS = {
    "A": "Ambiguous",
    "F": "Female",
    "M": "Male",
    "N": "Not applicable",
    "O": "Other",
    "U": "Unknown",
}
_PATIENT_CLASS_LABELS = {
    "E": "Emergency",
    "I": "Inpatient",
    "O": "Outpatient",
    "P": "Preadmit",
    "R": "Recurring patient",
}
_RESULT_STATUS_LABELS = {
    "C": "Corrected",
    "F": "Final",
    "I": "Specimen in lab",
    "P": "Preliminary",
    "R": "Entered, not verified",
    "S": "Partial",
    "X": "Cannot obtain",
}


@dataclass(frozen=True)
class HL7V2FieldSource:
    """One source field in an HL7 v2 message.

    ``segment_index`` is zero-based in message order. ``segment_occurrence``
    is one-based among segments with the same three-character name.
    """

    segment: str
    segment_index: int
    segment_occurrence: int
    field_position: int

    @property
    def path(self) -> str:
        """Return a compact, stable source path such as ``OBX[2]-5``."""

        return f"{self.segment}[{self.segment_occurrence}]-{self.field_position}"


@dataclass(frozen=True)
class HL7V2FieldSpan:
    """A final narrative character range mapped to one source field."""

    start: int
    end: int
    source: HL7V2FieldSource
    section: str
    label: str

    @property
    def segment(self) -> str:
        """Return the source segment name."""

        return self.source.segment

    @property
    def field_position(self) -> int:
        """Return the one-based source field position."""

        return self.source.field_position


@dataclass(frozen=True)
class HL7V2NarrativeSection:
    """One logical section and its range in the final narrative."""

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class HL7V2Narrative:
    """De-identified narrative text plus source-field provenance."""

    text: str
    mode: NarrativeMode
    message_type: str | None
    spans: tuple[HL7V2FieldSpan, ...]
    sections: tuple[HL7V2NarrativeSection, ...]

    @property
    def field_mappings(self) -> tuple[HL7V2FieldSpan, ...]:
        """Return the field mappings (an explicit alias for ``spans``)."""

        return self.spans

    def provenance_at(self, offset: int) -> tuple[HL7V2FieldSpan, ...]:
        """Return every source-field span covering a narrative offset."""

        if offset < 0 or offset >= len(self.text):
            return ()
        return tuple(span for span in self.spans if span.start <= offset < span.end)

    def spans_for(
        self,
        segment: str,
        field_position: int,
        *,
        segment_occurrence: int | None = None,
    ) -> tuple[HL7V2FieldSpan, ...]:
        """Return narrative spans sourced from a segment/field coordinate."""

        normalized_segment = segment.upper()
        return tuple(
            span
            for span in self.spans
            if span.source.segment == normalized_segment
            and span.source.field_position == field_position
            and (
                segment_occurrence is None
                or span.source.segment_occurrence == segment_occurrence
            )
        )

    def text_for(self, span: HL7V2FieldSpan) -> str:
        """Return the final narrative text covered by ``span``."""

        return self.text[span.start : span.end]


@dataclass(frozen=True)
class _NarrativeItem:
    section: str
    label: str
    value: str
    source: HL7V2FieldSource


@dataclass(frozen=True)
class _RenderedNarrative:
    text: str
    spans: tuple[HL7V2FieldSpan, ...]
    sections: tuple[HL7V2NarrativeSection, ...]


def extract_hl7v2_narrative(
    message_or_path: str | Path | HL7Message,
    *,
    mode: NarrativeMode = "flat",
    field_map: Mapping[FieldKey | str, Any] | None = None,
    deidentifier: TextDeidentifier | None = None,
    deidentify_kwargs: Mapping[str, Any] | None = None,
    date_shift_days: int = 30,
    lang: str = "en",
    locale: str | None = None,
    seed: int | None = 0,
) -> HL7V2Narrative:
    """Render a safe narrative from a common ADT, ORU, or ORM message.

    The existing HL7 v2 redactor first handles structured fields such as
    patient identifiers and names. Its free-text hook is deliberately kept as
    a no-op during that pass because the complete rendered narrative is then
    sent through the supplied de-identification pipeline once. Final offsets
    are projected after de-identification, so provenance always indexes the
    returned text rather than the raw message or an intermediate narrative.

    Args:
        message_or_path: Pipe-delimited message text, UTF-8 path, or parsed
            :class:`~openmed.interop.hl7v2.HL7Message`.
        mode: ``"flat"`` for sentence-like text or ``"sectioned"`` for
            Markdown-style sections.
        field_map: Optional structured redaction rules forwarded to
            :func:`~openmed.interop.hl7v2.redact_hl7v2`.
        deidentifier: Optional narrative de-identifier. Defaults to
            :func:`openmed.core.pii.deidentify`.
        deidentify_kwargs: Extra keyword arguments for the narrative
            de-identifier. The defaults are ``method="mask"`` and ``lang``.
        date_shift_days: Stable structured date shift applied before narrative
            rendering. The default is 30 days; the complete narrative also
            passes through the privacy pipeline.
        lang: Language forwarded to structured and narrative de-identification.
        locale: Optional locale for deterministic structured surrogates.
        seed: Seed for stable structured surrogates.

    Returns:
        A de-identified narrative with section ranges and field provenance.

    Raises:
        ValueError: If ``mode`` is unsupported.
        TypeError: If the de-identifier does not return text in a supported
            shape.
    """

    if mode not in {"flat", "sectioned"}:
        raise ValueError("mode must be 'flat' or 'sectioned'")

    message_input: str | Path
    if isinstance(message_or_path, HL7Message):
        message_input = message_or_path.serialize()
    else:
        message_input = message_or_path

    structured_safe = redact_hl7v2(
        message_input,
        field_map=field_map,
        deidentifier=_identity_deidentifier,
        date_shift_days=date_shift_days,
        lang=lang,
        locale=locale,
        seed=seed,
    )
    parsed = parse_hl7v2(structured_safe)
    items = _collect_items(parsed)
    rendered = _render_items(items, mode=mode)

    deidentify = deidentifier or _load_deidentifier()
    kwargs = {"method": "mask", "lang": lang, **dict(deidentify_kwargs or {})}
    result = deidentify(rendered.text, **kwargs)
    safe_text = _deidentified_text(result)

    opcodes = SequenceMatcher(
        None,
        rendered.text,
        safe_text,
        autojunk=False,
    ).get_opcodes()
    spans = tuple(
        _project_field_span(span, opcodes, len(safe_text)) for span in rendered.spans
    )
    spans = tuple(span for span in spans if span.end > span.start)
    sections = tuple(
        _project_section(section, opcodes, len(safe_text))
        for section in rendered.sections
    )

    return HL7V2Narrative(
        text=safe_text,
        mode=mode,
        message_type=_message_type(parsed),
        spans=spans,
        sections=sections,
    )


def _identity_deidentifier(text: str, **_: Any) -> str:
    """Preserve HL7 free text until the complete narrative privacy pass."""

    return text


def _load_deidentifier() -> TextDeidentifier:
    from openmed.core.pii import deidentify

    return deidentify


def _deidentified_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping) and "deidentified_text" in result:
        return str(result["deidentified_text"])
    if hasattr(result, "deidentified_text"):
        return str(result.deidentified_text)
    raise TypeError(
        "narrative deidentifier must return a string, mapping, or object with "
        "deidentified_text"
    )


def _collect_items(message: HL7Message) -> tuple[_NarrativeItem, ...]:
    items: list[_NarrativeItem] = []
    occurrences: defaultdict[str, int] = defaultdict(int)
    observation_number = 0
    order_number = 0
    note_number = 0

    for segment_index, segment in enumerate(message.segments):
        occurrences[segment.name] += 1
        occurrence = occurrences[segment.name]
        source = _source_factory(segment, segment_index, occurrence)

        if segment.name == "MSH":
            _add(items, "Message", "Message type", segment, 9, source, _message_code)
            _add(items, "Message", "HL7 version", segment, 12, source)
        elif segment.name == "EVN":
            _add(items, "Encounter", "Event type", segment, 1, source)
            _add(items, "Encounter", "Event recorded", segment, 2, source, _date)
        elif segment.name == "PID":
            _add(items, "Patient", "Patient ID", segment, 3, source, _identifier)
            _add(items, "Patient", "Patient name", segment, 5, source, _name)
            _add(items, "Patient", "Date of birth", segment, 7, source, _date)
            _add(items, "Patient", "Administrative sex", segment, 8, source, _sex)
        elif segment.name == "PV1":
            _add(
                items,
                "Encounter",
                "Patient class",
                segment,
                2,
                source,
                _patient_class,
            )
            _add(items, "Encounter", "Location", segment, 3, source, _components)
            _add(
                items,
                "Encounter",
                "Attending provider",
                segment,
                7,
                source,
                _name,
            )
            _add(items, "Encounter", "Admitted", segment, 44, source, _date)
            _add(items, "Encounter", "Discharged", segment, 45, source, _date)
        elif segment.name == "ORC":
            _add(items, "Orders", "Order control", segment, 1, source)
            _add(items, "Orders", "Placer order ID", segment, 2, source, _identifier)
            _add(items, "Orders", "Filler order ID", segment, 3, source, _identifier)
        elif segment.name == "OBR":
            order_number += 1
            _add(
                items,
                "Orders",
                f"Requested test {order_number}",
                segment,
                4,
                source,
                _coded,
            )
            _add(
                items,
                "Orders",
                f"Observation requested {order_number}",
                segment,
                7,
                source,
                _date,
            )
        elif segment.name == "OBX":
            observation_number += 1
            _add(
                items,
                "Observations",
                f"Observation {observation_number}",
                segment,
                3,
                source,
                _coded,
            )
            _add(
                items,
                "Observations",
                f"Result {observation_number}",
                segment,
                5,
                source,
                _components,
            )
            _add(
                items,
                "Observations",
                f"Units {observation_number}",
                segment,
                6,
                source,
                _coded,
            )
            _add(
                items,
                "Observations",
                f"Reference range {observation_number}",
                segment,
                7,
                source,
            )
            _add(
                items,
                "Observations",
                f"Abnormal flags {observation_number}",
                segment,
                8,
                source,
                _components,
            )
            _add(
                items,
                "Observations",
                f"Result status {observation_number}",
                segment,
                11,
                source,
                _result_status,
            )
            _add(
                items,
                "Observations",
                f"Observed at {observation_number}",
                segment,
                14,
                source,
                _date,
            )
        elif segment.name == "NTE":
            note_number += 1
            _add(items, "Notes", f"Note {note_number}", segment, 3, source, _text)

    return tuple(items)


def _source_factory(
    segment: HL7Segment,
    segment_index: int,
    segment_occurrence: int,
) -> Callable[[int], HL7V2FieldSource]:
    def build(field_position: int) -> HL7V2FieldSource:
        return HL7V2FieldSource(
            segment=segment.name,
            segment_index=segment_index,
            segment_occurrence=segment_occurrence,
            field_position=field_position,
        )

    return build


def _add(
    items: list[_NarrativeItem],
    section: str,
    label: str,
    segment: HL7Segment,
    field_position: int,
    source: Callable[[int], HL7V2FieldSource],
    formatter: Callable[[str, HL7V2Encoding], str] | None = None,
) -> None:
    raw_value = segment.get_field(field_position)
    if raw_value is None or not raw_value.strip():
        return
    value = (formatter or _text)(raw_value, segment.encoding).strip()
    if not value:
        return
    items.append(
        _NarrativeItem(
            section=section,
            label=label,
            value=value,
            source=source(field_position),
        )
    )


def _render_items(
    items: Sequence[_NarrativeItem],
    *,
    mode: NarrativeMode,
) -> _RenderedNarrative:
    by_section = {
        section: [item for item in items if item.section == section]
        for section in _SECTION_ORDER
    }
    parts: list[str] = []
    spans: list[HL7V2FieldSpan] = []
    sections: list[HL7V2NarrativeSection] = []
    cursor = 0

    def append(value: str) -> None:
        nonlocal cursor
        parts.append(value)
        cursor += len(value)

    emitted_sections = 0
    for section in _SECTION_ORDER:
        section_items = by_section[section]
        if not section_items:
            continue

        if emitted_sections:
            append("\n\n" if mode == "sectioned" else " ")
        section_start = cursor
        if mode == "sectioned":
            append(f"## {section}\n")

        for item_index, item in enumerate(section_items):
            if item_index:
                append("\n" if mode == "sectioned" else " ")
            append(f"{item.label}: ")
            value_start = cursor
            append(item.value)
            value_end = cursor
            if mode == "flat":
                append(".")
            spans.append(
                HL7V2FieldSpan(
                    start=value_start,
                    end=value_end,
                    source=item.source,
                    section=item.section,
                    label=item.label,
                )
            )

        sections.append(
            HL7V2NarrativeSection(name=section, start=section_start, end=cursor)
        )
        emitted_sections += 1

    return _RenderedNarrative(
        text="".join(parts),
        spans=tuple(spans),
        sections=tuple(sections),
    )


def _message_type(message: HL7Message) -> str | None:
    msh = next((segment for segment in message.segments if segment.name == "MSH"), None)
    if msh is None:
        return None
    value = msh.get_field(9)
    return value or None


def _text(value: str, encoding: HL7V2Encoding) -> str:
    return _decode_escapes(value, encoding).strip()


def _components(value: str, encoding: HL7V2Encoding) -> str:
    repetitions: list[str] = []
    for repetition in value.split(encoding.repetition):
        components = [
            _decode_escapes(component, encoding).strip()
            for component in repetition.split(encoding.component)
            if component.strip()
        ]
        if components:
            repetitions.append(" / ".join(components))
    return "; ".join(repetitions)


def _coded(value: str, encoding: HL7V2Encoding) -> str:
    repetitions: list[str] = []
    for repetition in value.split(encoding.repetition):
        components = repetition.split(encoding.component)
        code = _decode_escapes(components[0], encoding).strip() if components else ""
        display = (
            _decode_escapes(components[1], encoding).strip()
            if len(components) > 1
            else ""
        )
        if display and code and display != code:
            repetitions.append(f"{display} ({code})")
        elif display or code:
            repetitions.append(display or code)
    return "; ".join(repetitions)


def _message_code(value: str, encoding: HL7V2Encoding) -> str:
    components = [
        _decode_escapes(component, encoding).strip()
        for component in value.split(encoding.component)
        if component.strip()
    ]
    return " ".join(components[:2])


def _identifier(value: str, encoding: HL7V2Encoding) -> str:
    first_repetition = value.split(encoding.repetition, 1)[0]
    first_component = first_repetition.split(encoding.component, 1)[0]
    return _decode_escapes(first_component, encoding).strip()


def _name(value: str, encoding: HL7V2Encoding) -> str:
    first_repetition = value.split(encoding.repetition, 1)[0]
    components = [
        _decode_escapes(component, encoding).strip()
        for component in first_repetition.split(encoding.component)
    ]
    family = _component(components, 0)
    given = _component(components, 1)
    middle = _component(components, 2)
    suffix = _component(components, 3)
    prefix = _component(components, 4)
    ordered = [prefix, given, middle, family, suffix]
    return " ".join(part for part in ordered if part)


def _component(components: Sequence[str], index: int) -> str:
    return components[index] if index < len(components) else ""


def _date(value: str, encoding: HL7V2Encoding) -> str:
    raw = _identifier(value, encoding)
    if len(raw) < 8 or not raw[:8].isdigit():
        return raw
    rendered = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    time = raw[8:14]
    if time:
        rendered += f" {time[0:2]}"
        if len(time) >= 4:
            rendered += f":{time[2:4]}"
        if len(time) >= 6:
            rendered += f":{time[4:6]}"
    return rendered


def _sex(value: str, encoding: HL7V2Encoding) -> str:
    code = _identifier(value, encoding).upper()
    label = _SEX_LABELS.get(code)
    return f"{label} ({code})" if label else code


def _patient_class(value: str, encoding: HL7V2Encoding) -> str:
    code = _identifier(value, encoding).upper()
    label = _PATIENT_CLASS_LABELS.get(code)
    return f"{label} ({code})" if label else code


def _result_status(value: str, encoding: HL7V2Encoding) -> str:
    code = _identifier(value, encoding).upper()
    label = _RESULT_STATUS_LABELS.get(code)
    return f"{label} ({code})" if label else code


def _decode_escapes(value: str, encoding: HL7V2Encoding) -> str:
    escape = encoding.escape
    replacements = {
        f"{escape}F{escape}": encoding.field,
        f"{escape}S{escape}": encoding.component,
        f"{escape}R{escape}": encoding.repetition,
        f"{escape}T{escape}": encoding.subcomponent,
        f"{escape}E{escape}": encoding.escape,
        f"{escape}.br{escape}": "\n",
    }
    rendered = value
    for encoded, decoded in replacements.items():
        rendered = rendered.replace(encoded, decoded)
    return rendered


def _project_field_span(
    span: HL7V2FieldSpan,
    opcodes: Sequence[tuple[str, int, int, int, int]],
    target_length: int,
) -> HL7V2FieldSpan:
    start, end = _project_range(span.start, span.end, opcodes, target_length)
    return HL7V2FieldSpan(
        start=start,
        end=end,
        source=span.source,
        section=span.section,
        label=span.label,
    )


def _project_section(
    section: HL7V2NarrativeSection,
    opcodes: Sequence[tuple[str, int, int, int, int]],
    target_length: int,
) -> HL7V2NarrativeSection:
    start, end = _project_range(section.start, section.end, opcodes, target_length)
    return HL7V2NarrativeSection(name=section.name, start=start, end=end)


def _project_range(
    start: int,
    end: int,
    opcodes: Sequence[tuple[str, int, int, int, int]],
    target_length: int,
) -> tuple[int, int]:
    projected_start = _project_boundary(start, opcodes, prefer_end=False)
    projected_end = _project_boundary(end, opcodes, prefer_end=True)
    projected_start = max(0, min(projected_start, target_length))
    projected_end = max(projected_start, min(projected_end, target_length))
    return projected_start, projected_end


def _project_boundary(
    position: int,
    opcodes: Sequence[tuple[str, int, int, int, int]],
    *,
    prefer_end: bool,
) -> int:
    for tag, source_start, source_end, target_start, target_end in opcodes:
        if tag == "insert":
            if position == source_start:
                return target_end if prefer_end else target_start
            continue
        if not source_start <= position <= source_end:
            continue
        if tag == "equal":
            return target_start + (position - source_start)
        if position == source_start:
            return target_start
        if position == source_end:
            return target_end
        if tag == "delete":
            return target_start
        source_width = source_end - source_start
        target_width = target_end - target_start
        relative = (position - source_start) / source_width
        return target_start + round(relative * target_width)
    return opcodes[-1][4] if opcodes else position


__all__ = [
    "HL7V2FieldSource",
    "HL7V2FieldSpan",
    "HL7V2Narrative",
    "HL7V2NarrativeSection",
    "NarrativeMode",
    "extract_hl7v2_narrative",
]
