"""Segment-aware X12 837 claim parsing and PHI redaction.

The parser preserves the interchange delimiter set, segment order, and any
whitespace between segments. It deliberately validates only the envelope
needed to identify an 837 transaction; companion-guide and full X12
conformance validation are outside this module's scope.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class X12Delimiters:
    """Separators declared by the fixed-width ISA interchange header."""

    element: str
    repetition: str
    component: str
    segment: str


@dataclass
class X12Element:
    """One X12 data element and its original raw-text character span."""

    value: str
    source_start: int
    source_end: int

    @property
    def source_span(self) -> tuple[int, int]:
        """Return the half-open source character span."""

        return (self.source_start, self.source_end)


@dataclass
class X12Segment:
    """A parsed X12 segment with one-based element access."""

    tag: str
    elements: list[X12Element]
    leading: str
    source_start: int
    source_end: int

    def get_element(self, position: int) -> X12Element | None:
        """Return a one-based element, or ``None`` when it is absent."""

        if position < 1:
            raise ValueError("X12 element positions are one-based")
        index = position - 1
        if index >= len(self.elements):
            return None
        return self.elements[index]

    def get_value(self, position: int) -> str | None:
        """Return a one-based element value, or ``None`` when absent."""

        element = self.get_element(position)
        return element.value if element is not None else None

    def set_value(self, position: int, value: str) -> None:
        """Replace an existing one-based element value."""

        element = self.get_element(position)
        if element is None:
            raise IndexError(f"{self.tag}{position:02d} is absent")
        element.value = str(value)


@dataclass(frozen=True)
class X12OffsetEntry:
    """Original and serialized character spans for one X12 element."""

    segment_index: int
    segment_tag: str
    element_position: int
    source_start: int
    source_end: int
    output_start: int
    output_end: int

    @property
    def source_span(self) -> tuple[int, int]:
        """Return the half-open span in the raw input."""

        return (self.source_start, self.source_end)

    @property
    def output_span(self) -> tuple[int, int]:
        """Return the half-open span in the serialized output."""

        return (self.output_start, self.output_end)


@dataclass(frozen=True)
class X12OffsetMap:
    """Bidirectional element-span projection between raw and output text.

    Projection is exact at element granularity. That is the unit used by the
    structured redactor and avoids inventing character-level correspondences
    when a replacement has a different length from its source value.
    """

    entries: tuple[X12OffsetEntry, ...]

    def for_element(
        self,
        segment_index: int,
        element_position: int,
    ) -> X12OffsetEntry:
        """Return the mapping for a segment/element coordinate."""

        matches = [
            entry
            for entry in self.entries
            if entry.segment_index == segment_index
            and entry.element_position == element_position
        ]
        if len(matches) != 1:
            raise KeyError(
                "no unique offset entry for "
                f"segment {segment_index}, element {element_position}"
            )
        return matches[0]

    def source_to_output(self, start: int, end: int) -> tuple[int, int]:
        """Project an exact source element span to serialized output."""

        entry = self._entry_for_span(start, end, source=True)
        return entry.output_span

    def output_to_source(self, start: int, end: int) -> tuple[int, int]:
        """Project an exact output element span back to the raw input."""

        entry = self._entry_for_span(start, end, source=False)
        return entry.source_span

    def _entry_for_span(
        self,
        start: int,
        end: int,
        *,
        source: bool,
    ) -> X12OffsetEntry:
        if start < 0 or end < start:
            raise ValueError("offset spans must satisfy 0 <= start <= end")

        matches = [
            entry
            for entry in self.entries
            if (entry.source_span if source else entry.output_span) == (start, end)
        ]
        if len(matches) != 1:
            side = "source" if source else "output"
            raise KeyError(f"no unique X12 element for {side} span {(start, end)}")
        return matches[0]


@dataclass
class X12837Message:
    """Parsed X12 837 message with lossless framing metadata."""

    segments: list[X12Segment]
    delimiters: X12Delimiters
    suffix: str = ""

    @classmethod
    def parse(cls, message: str) -> "X12837Message":
        """Parse an X12 837 interchange from raw text."""

        delimiters = _delimiters_from_isa(message)
        segments, suffix = _parse_segments(message, delimiters)
        _validate_837_envelope(segments)
        return cls(segments=segments, delimiters=delimiters, suffix=suffix)

    def serialize(self) -> str:
        """Serialize the message while preserving its original framing."""

        rendered, _ = self.serialize_with_offset_map()
        return rendered

    def serialize_with_offset_map(self) -> tuple[str, X12OffsetMap]:
        """Serialize and map every original element span into the output."""

        chunks: list[str] = []
        entries: list[X12OffsetEntry] = []
        cursor = 0
        element_separator = self.delimiters.element

        for segment_index, segment in enumerate(self.segments):
            chunks.append(segment.leading)
            cursor += len(segment.leading)
            chunks.append(segment.tag)
            cursor += len(segment.tag)

            for element_position, element in enumerate(segment.elements, start=1):
                chunks.append(element_separator)
                cursor += len(element_separator)
                output_start = cursor
                chunks.append(element.value)
                cursor += len(element.value)
                entries.append(
                    X12OffsetEntry(
                        segment_index=segment_index,
                        segment_tag=segment.tag,
                        element_position=element_position,
                        source_start=element.source_start,
                        source_end=element.source_end,
                        output_start=output_start,
                        output_end=cursor,
                    )
                )

            chunks.append(self.delimiters.segment)
            cursor += len(self.delimiters.segment)

        chunks.append(self.suffix)
        return "".join(chunks), X12OffsetMap(tuple(entries))

    def segment_names(self) -> tuple[str, ...]:
        """Return segment tags in message order."""

        return tuple(segment.tag for segment in self.segments)


@dataclass(frozen=True)
class X12837Redaction:
    """Audit-safe record for one redacted X12 element.

    The original value is intentionally omitted. Its SHA-256 digest and raw
    offsets provide provenance without copying PHI into audit artifacts.
    """

    segment_index: int
    segment_tag: str
    element_position: int
    entity_code: str
    label: str
    source_start: int
    source_end: int
    output_start: int
    output_end: int
    original_sha256: str
    replacement: str

    @property
    def source_span(self) -> tuple[int, int]:
        """Return the half-open raw-input span."""

        return (self.source_start, self.source_end)

    @property
    def output_span(self) -> tuple[int, int]:
        """Return the half-open redacted-output span."""

        return (self.output_start, self.output_end)


@dataclass(frozen=True)
class X12837RedactionResult:
    """Redacted X12 text plus audit-safe provenance and offset projection."""

    deidentified_text: str
    redactions: tuple[X12837Redaction, ...]
    offset_map: X12OffsetMap

    @property
    def redacted_text(self) -> str:
        """Return an explicit alias for :attr:`deidentified_text`."""

        return self.deidentified_text


@dataclass(frozen=True)
class _PendingRedaction:
    segment_index: int
    segment_tag: str
    element_position: int
    entity_code: str
    label: str
    source_start: int
    source_end: int
    original_sha256: str


SUBSCRIBER_PATIENT_ENTITY_CODES = frozenset({"IL", "QC"})
PROVIDER_ENTITY_CODES = frozenset(
    {
        "72",  # operating physician
        "77",  # service location
        "82",  # rendering provider
        "85",  # billing provider
        "87",  # pay-to provider
        "DN",  # referring provider
        "DQ",  # supervising provider
        "FA",  # facility
        "P3",  # primary care provider
        "PE",  # payee
    }
)
PHI_ENTITY_CODES = SUBSCRIBER_PATIENT_ENTITY_CODES | PROVIDER_ENTITY_CODES

_NM1_REDACTIONS = {
    3: "NAME",
    4: "NAME",
    5: "NAME",
    6: "NAME",
    7: "NAME",
    9: "IDENTIFIER",
}
_CONTEXT_REDACTIONS = {
    "N2": {1: "NAME", 2: "NAME"},
    "N3": {1: "ADDRESS", 2: "ADDRESS"},
    "N4": {1: "LOCATION", 2: "LOCATION", 3: "POSTAL_CODE"},
    "DMG": {2: "DATE_OF_BIRTH"},
    "REF": {2: "IDENTIFIER"},
}
_CONTEXT_PASSTHROUGH_SEGMENTS = frozenset({"PER", "PRV"})


def parse_x12_837(message: str) -> X12837Message:
    """Parse raw X12 837 text while preserving its separators and offsets."""

    return X12837Message.parse(message)


def redact_x12_837(
    message_or_path: str | Path,
    *,
    replacement: str = "REDACTED",
) -> X12837RedactionResult:
    """Redact structured PHI elements from an X12 837 claim.

    Patient, subscriber, and provider loops are identified from ``NM101``.
    Names and identifiers in ``NM1``, address elements in ``N2``/``N3``/``N4``,
    birth dates in ``DMG``, and loop-local identifiers in ``REF`` are replaced.
    Envelope, control, qualifier, and claim/service segments are not modified.

    Args:
        message_or_path: Raw X12 837 text or a UTF-8 file path.
        replacement: Token used for every non-empty PHI leaf. It must not
            contain any separator declared by the interchange.

    Returns:
        Redacted text, structured redaction records, and a bidirectional
        element-level offset map.
    """

    message_text = _read_message_or_path(message_or_path)
    message = X12837Message.parse(message_text)
    _validate_replacement(replacement, message.delimiters)

    pending: list[_PendingRedaction] = []
    active_entity_code: str | None = None

    for segment_index, segment in enumerate(message.segments):
        if segment.tag == "NM1":
            entity_code = (segment.get_value(1) or "").upper()
            active_entity_code = (
                entity_code if entity_code in PHI_ENTITY_CODES else None
            )
            if active_entity_code is not None:
                _redact_elements(
                    segment,
                    segment_index=segment_index,
                    entity_code=active_entity_code,
                    positions=_NM1_REDACTIONS,
                    replacement=replacement,
                    delimiters=message.delimiters,
                    pending=pending,
                )
            continue

        positions = _CONTEXT_REDACTIONS.get(segment.tag)
        if active_entity_code is not None and positions is not None:
            _redact_elements(
                segment,
                segment_index=segment_index,
                entity_code=active_entity_code,
                positions=positions,
                replacement=replacement,
                delimiters=message.delimiters,
                pending=pending,
            )
            continue

        if segment.tag not in _CONTEXT_PASSTHROUGH_SEGMENTS:
            active_entity_code = None

    deidentified_text, offset_map = message.serialize_with_offset_map()
    redactions = tuple(
        _finish_redaction(item, offset_map, deidentified_text) for item in pending
    )
    return X12837RedactionResult(
        deidentified_text=deidentified_text,
        redactions=redactions,
        offset_map=offset_map,
    )


def _delimiters_from_isa(message: str) -> X12Delimiters:
    if len(message) < 106 or not message.startswith("ISA"):
        raise ValueError("X12 837 messages must start with a fixed-width ISA segment")

    element = message[3]
    repetition = message[82]
    component = message[104]
    segment = message[105]
    delimiters = X12Delimiters(
        element=element,
        repetition=repetition,
        component=component,
        segment=segment,
    )
    values = (element, repetition, component, segment)
    if any(len(value) != 1 or value.isspace() for value in values):
        raise ValueError("X12 ISA separators must be non-whitespace characters")
    if len(set(values)) != len(values):
        raise ValueError("X12 ISA separators must be distinct")

    isa_content = message[:105]
    isa_parts = isa_content.split(element)
    if len(isa_parts) != 17 or isa_parts[0] != "ISA":
        raise ValueError("invalid fixed-width ISA segment")
    if len(isa_parts[11]) != 1 or len(isa_parts[16]) != 1:
        raise ValueError("ISA11 and ISA16 separators must be one character")
    if isa_parts[11] != repetition or isa_parts[16] != component:
        raise ValueError("ISA separator positions do not match ISA elements")
    return delimiters


def _parse_segments(
    message: str,
    delimiters: X12Delimiters,
) -> tuple[list[X12Segment], str]:
    segments: list[X12Segment] = []
    cursor = 0
    suffix = ""

    while cursor < len(message):
        leading_start = cursor
        while cursor < len(message) and message[cursor].isspace():
            cursor += 1
        leading = message[leading_start:cursor]

        terminator = message.find(delimiters.segment, cursor)
        if terminator < 0:
            suffix = message[leading_start:]
            if suffix.strip():
                raise ValueError("unterminated X12 segment")
            break

        segment_text = message[cursor:terminator]
        if not segment_text:
            raise ValueError("empty X12 segment")
        parts = segment_text.split(delimiters.element)
        tag = parts[0]
        if not 2 <= len(tag) <= 3 or not tag.isalnum():
            raise ValueError(f"invalid X12 segment tag {tag!r}")

        elements: list[X12Element] = []
        element_start = cursor + len(tag) + 1
        for value in parts[1:]:
            element_end = element_start + len(value)
            elements.append(
                X12Element(
                    value=value,
                    source_start=element_start,
                    source_end=element_end,
                )
            )
            element_start = element_end + 1

        segments.append(
            X12Segment(
                tag=tag,
                elements=elements,
                leading=leading,
                source_start=cursor,
                source_end=terminator + 1,
            )
        )
        cursor = terminator + 1

    if not segments:
        raise ValueError("empty X12 message")
    return segments, suffix


def _validate_837_envelope(segments: list[X12Segment]) -> None:
    tags = [segment.tag for segment in segments]
    if tags[0] != "ISA" or tags[-1] != "IEA":
        raise ValueError("X12 interchange must be framed by ISA and IEA segments")

    required = ("GS", "ST", "SE", "GE")
    missing = [tag for tag in required if tag not in tags]
    if missing:
        raise ValueError(f"X12 837 envelope is missing: {', '.join(missing)}")

    first_positions = [
        tags.index(tag) for tag in ("ISA", "GS", "ST", "SE", "GE", "IEA")
    ]
    if first_positions != sorted(first_positions):
        raise ValueError("X12 837 envelope segments are out of order")

    transaction_segments = [segment for segment in segments if segment.tag == "ST"]
    if any(segment.get_value(1) != "837" for segment in transaction_segments):
        raise ValueError("only X12 837 transaction sets are supported")


def _redact_elements(
    segment: X12Segment,
    *,
    segment_index: int,
    entity_code: str,
    positions: dict[int, str],
    replacement: str,
    delimiters: X12Delimiters,
    pending: list[_PendingRedaction],
) -> None:
    for position, label in positions.items():
        element = segment.get_element(position)
        if element is None or not element.value:
            continue

        original = element.value
        redacted = _replace_leaf_values(original, replacement, delimiters)
        if redacted == original:
            continue

        element.value = redacted
        pending.append(
            _PendingRedaction(
                segment_index=segment_index,
                segment_tag=segment.tag,
                element_position=position,
                entity_code=entity_code,
                label=label,
                source_start=element.source_start,
                source_end=element.source_end,
                original_sha256=hashlib.sha256(original.encode("utf-8")).hexdigest(),
            )
        )


def _replace_leaf_values(
    value: str,
    replacement: str,
    delimiters: X12Delimiters,
) -> str:
    leaf_separators = {delimiters.repetition, delimiters.component}
    chunks: list[str] = []
    leaf: list[str] = []

    def flush() -> None:
        raw_leaf = "".join(leaf)
        chunks.append(replacement if raw_leaf else "")
        leaf.clear()

    for character in value:
        if character in leaf_separators:
            flush()
            chunks.append(character)
        else:
            leaf.append(character)
    flush()
    return "".join(chunks)


def _finish_redaction(
    pending: _PendingRedaction,
    offset_map: X12OffsetMap,
    deidentified_text: str,
) -> X12837Redaction:
    entry = offset_map.for_element(
        pending.segment_index,
        pending.element_position,
    )
    return X12837Redaction(
        segment_index=pending.segment_index,
        segment_tag=pending.segment_tag,
        element_position=pending.element_position,
        entity_code=pending.entity_code,
        label=pending.label,
        source_start=pending.source_start,
        source_end=pending.source_end,
        output_start=entry.output_start,
        output_end=entry.output_end,
        original_sha256=pending.original_sha256,
        replacement=deidentified_text[entry.output_start : entry.output_end],
    )


def _validate_replacement(replacement: str, delimiters: X12Delimiters) -> None:
    if not isinstance(replacement, str):
        raise TypeError("replacement must be a string")
    separators = {
        delimiters.element,
        delimiters.repetition,
        delimiters.component,
        delimiters.segment,
    }
    if any(character in separators for character in replacement):
        raise ValueError("replacement must not contain X12 separators")
    if "\r" in replacement or "\n" in replacement:
        raise ValueError("replacement must not contain line breaks")


def _read_message_or_path(message_or_path: str | Path) -> str:
    if isinstance(message_or_path, Path):
        with message_or_path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()

    value = str(message_or_path)
    try:
        candidate = Path(value)
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                return handle.read()
    except OSError:
        pass
    return value


__all__ = [
    "PHI_ENTITY_CODES",
    "PROVIDER_ENTITY_CODES",
    "SUBSCRIBER_PATIENT_ENTITY_CODES",
    "X12837Message",
    "X12837Redaction",
    "X12837RedactionResult",
    "X12Delimiters",
    "X12Element",
    "X12OffsetEntry",
    "X12OffsetMap",
    "X12Segment",
    "parse_x12_837",
    "redact_x12_837",
]
