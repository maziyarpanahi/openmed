"""HL7 v2 segment-aware message de-identification.

This module implements a small HL7 v2 pipe-message parser for local
de-identification workflows. It is intentionally scoped to preserving message
framing while redacting common PHI-bearing fields in ADT, ORU, and ORM-style
messages. It is not a full HL7 conformance validator and never calls external
services.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

FieldAction = Literal["clear", "hash", "surrogate", "date-shift", "redact_text"]
FieldKey = tuple[str, int]
TextDeidentifier = Callable[..., Any]

SUPPORTED_MESSAGE_TYPES = ("ADT", "ORU", "ORM")
DEFAULT_NOTE_VALUE_TYPES = frozenset({"FT", "TX"})

_VALID_ACTIONS = frozenset({"clear", "hash", "surrogate", "date-shift", "redact_text"})
_HL7_DATE_RE = re.compile(
    r"^(?P<ymd>\d{8})(?P<time>\d{0,6})(?P<fraction>\.\d+)?(?P<tz>[+-]\d{4})?$"
)
_ISO_DATE_RE = re.compile(
    r"^(?P<year>\d{4})(?P<sep>[-/])(?P<month>\d{2})(?P=sep)(?P<day>\d{2})$"
)


@dataclass(frozen=True)
class HL7V2Encoding:
    """HL7 v2 delimiter set derived from ``MSH-1`` and ``MSH-2``."""

    field: str = "|"
    component: str = "^"
    repetition: str = "~"
    escape: str = "\\"
    subcomponent: str = "&"

    @classmethod
    def from_msh_segment(cls, segment_text: str) -> "HL7V2Encoding":
        """Build encoding characters from a raw MSH segment."""

        if len(segment_text) < 4 or not segment_text.startswith("MSH"):
            raise ValueError("HL7 v2 messages must start with an MSH segment")

        field_separator = segment_text[3]
        parts = segment_text.split(field_separator)
        encoding_chars = parts[1] if len(parts) > 1 and parts[1] else "^~\\&"
        return cls(
            field=field_separator,
            component=_char_at_or_default(encoding_chars, 0, "^"),
            repetition=_char_at_or_default(encoding_chars, 1, "~"),
            escape=_char_at_or_default(encoding_chars, 2, "\\"),
            subcomponent=_char_at_or_default(encoding_chars, 3, "&"),
        )

    def as_msh_2(self) -> str:
        """Return the four MSH-2 encoding characters."""

        return f"{self.component}{self.repetition}{self.escape}{self.subcomponent}"


@dataclass
class HL7Segment:
    """A parsed HL7 v2 segment with field-position helpers."""

    name: str
    fields: list[str]
    encoding: HL7V2Encoding

    @classmethod
    def parse(
        cls,
        segment_text: str,
        *,
        encoding: HL7V2Encoding | None = None,
    ) -> "HL7Segment":
        """Parse one segment using the message encoding."""

        if len(segment_text) < 3:
            raise ValueError(f"invalid HL7 segment: {segment_text!r}")

        name = segment_text[:3]
        if name == "MSH":
            segment_encoding = HL7V2Encoding.from_msh_segment(segment_text)
            parts = segment_text.split(segment_encoding.field)
            return cls(name=name, fields=parts[1:], encoding=segment_encoding)

        if encoding is None:
            raise ValueError("non-MSH segments require an HL7 encoding")

        parts = segment_text.split(encoding.field)
        return cls(name=parts[0], fields=parts[1:], encoding=encoding)

    def get_field(self, position: int) -> str | None:
        """Return the one-based HL7 field value, or ``None`` when absent."""

        index = _field_index(self.name, position)
        if index is None:
            return self.encoding.field
        if index < 0 or index >= len(self.fields):
            return None
        return self.fields[index]

    def set_field(self, position: int, value: str) -> None:
        """Set a one-based HL7 field value."""

        index = _field_index(self.name, position)
        if index is None:
            raise ValueError("MSH-1 is derived from the field separator")

        while len(self.fields) <= index:
            self.fields.append("")
        self.fields[index] = value

    def serialize(self) -> str:
        """Serialize the segment while preserving field delimiters."""

        separator = self.encoding.field
        if self.name == "MSH":
            fields = list(self.fields)
            if fields:
                fields[0] = self.encoding.as_msh_2()
            else:
                fields = [self.encoding.as_msh_2()]
            return f"MSH{separator}{separator.join(fields)}"

        return separator.join([self.name, *self.fields])


@dataclass
class HL7Message:
    """Parsed HL7 v2 message preserving segment order and separators."""

    segments: list[HL7Segment]
    encoding: HL7V2Encoding
    segment_separator: str = "\r"
    trailing_separator: bool = False

    @classmethod
    def parse(cls, message: str) -> "HL7Message":
        """Parse an HL7 v2 message into segment objects."""

        segment_separator = _detect_segment_separator(message)
        raw_segments = message.split(segment_separator)
        trailing_separator = bool(raw_segments and raw_segments[-1] == "")
        if trailing_separator:
            raw_segments = raw_segments[:-1]

        raw_segments = [segment for segment in raw_segments if segment]
        if not raw_segments:
            raise ValueError("empty HL7 v2 message")

        raw_segments[0] = raw_segments[0].lstrip("\ufeff")
        encoding = HL7V2Encoding.from_msh_segment(raw_segments[0])
        segments = [
            HL7Segment.parse(raw, encoding=encoding if index else None)
            for index, raw in enumerate(raw_segments)
        ]
        return cls(
            segments=segments,
            encoding=encoding,
            segment_separator=segment_separator,
            trailing_separator=trailing_separator,
        )

    def serialize(self) -> str:
        """Serialize the message to HL7 v2 pipe-delimited text."""

        rendered = self.segment_separator.join(
            segment.serialize() for segment in self.segments
        )
        if self.trailing_separator:
            rendered += self.segment_separator
        return rendered

    def segment_names(self) -> tuple[str, ...]:
        """Return segment names in message order."""

        return tuple(segment.name for segment in self.segments)


@dataclass(frozen=True)
class HL7FieldRule:
    """Data-driven redaction rule for one ``SEGMENT-field`` position.

    Component labels are one-based HL7 component positions. They let XPN-style
    fields preserve the component layout while redacting family, given, and
    middle names with label-aware surrogates.
    """

    action: FieldAction
    label: str = "ID_NUM"
    component_labels: Mapping[int, str] = field(default_factory=dict)
    type_field: int | None = None
    allowed_type_values: tuple[str, ...] = ()
    hash_salt: str = ""
    deidentify_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_action = str(self.action).lower()
        normalized_action = {
            "date_shift": "date-shift",
            "redact-text": "redact_text",
        }.get(raw_action, raw_action)
        if normalized_action not in _VALID_ACTIONS:
            known = ", ".join(sorted(_VALID_ACTIONS))
            raise ValueError(f"unknown HL7 field action {self.action!r}; use {known}")

        object.__setattr__(self, "action", normalized_action)
        object.__setattr__(
            self,
            "component_labels",
            {int(key): str(value) for key, value in self.component_labels.items()},
        )
        object.__setattr__(
            self,
            "allowed_type_values",
            tuple(str(value).upper() for value in self.allowed_type_values),
        )
        object.__setattr__(self, "deidentify_kwargs", dict(self.deidentify_kwargs))


DEFAULT_FIELD_MAP: Mapping[FieldKey, HL7FieldRule] = {
    ("PID", 3): HL7FieldRule("hash", label="ID_NUM"),
    ("PID", 5): HL7FieldRule(
        "surrogate",
        label="PERSON",
        component_labels={1: "LAST_NAME", 2: "FIRST_NAME", 3: "MIDDLE_NAME"},
    ),
    ("PID", 7): HL7FieldRule("date-shift", label="DATE_OF_BIRTH"),
    ("PID", 11): HL7FieldRule("surrogate", label="STREET_ADDRESS"),
    ("PID", 13): HL7FieldRule("hash", label="PHONE"),
    ("PID", 19): HL7FieldRule("hash", label="SSN"),
    ("PD1", 3): HL7FieldRule("surrogate", label="ORGANIZATION"),
    ("NK1", 2): HL7FieldRule(
        "surrogate",
        label="PERSON",
        component_labels={1: "LAST_NAME", 2: "FIRST_NAME", 3: "MIDDLE_NAME"},
    ),
    ("NK1", 4): HL7FieldRule("surrogate", label="STREET_ADDRESS"),
    ("NK1", 5): HL7FieldRule("hash", label="PHONE"),
    ("NK1", 13): HL7FieldRule("surrogate", label="ORGANIZATION"),
    ("GT1", 3): HL7FieldRule(
        "surrogate",
        label="PERSON",
        component_labels={1: "LAST_NAME", 2: "FIRST_NAME", 3: "MIDDLE_NAME"},
    ),
    ("GT1", 5): HL7FieldRule("surrogate", label="STREET_ADDRESS"),
    ("GT1", 6): HL7FieldRule("hash", label="PHONE"),
    ("GT1", 12): HL7FieldRule("hash", label="SSN"),
    ("GT1", 13): HL7FieldRule("date-shift", label="DATE_OF_BIRTH"),
    ("IN1", 16): HL7FieldRule(
        "surrogate",
        label="PERSON",
        component_labels={1: "LAST_NAME", 2: "FIRST_NAME", 3: "MIDDLE_NAME"},
    ),
    ("IN1", 18): HL7FieldRule("date-shift", label="DATE_OF_BIRTH"),
    ("IN1", 19): HL7FieldRule("surrogate", label="STREET_ADDRESS"),
    ("IN1", 36): HL7FieldRule("hash", label="ACCOUNT_NUMBER"),
    ("IN2", 1): HL7FieldRule("hash", label="ID_NUM"),
    ("IN2", 2): HL7FieldRule("hash", label="SSN"),
    ("OBX", 5): HL7FieldRule(
        "redact_text",
        label="OTHER",
        type_field=2,
        allowed_type_values=tuple(DEFAULT_NOTE_VALUE_TYPES),
    ),
    ("NTE", 3): HL7FieldRule("redact_text", label="OTHER"),
}


class _HL7V2Redactor:
    def __init__(
        self,
        *,
        field_map: Mapping[FieldKey | str, Any] | None,
        deidentifier: TextDeidentifier | None,
        deidentify_kwargs: Mapping[str, Any] | None,
        date_shift_days: int | None,
        lang: str,
        locale: str | None,
        seed: int | None,
    ) -> None:
        self.field_map = _normalize_field_map(field_map or DEFAULT_FIELD_MAP)
        self.deidentifier = deidentifier
        self.deidentify_kwargs = dict(deidentify_kwargs or {})
        self.date_shift_days = (
            date_shift_days if date_shift_days is not None else _random_nonzero_shift()
        )
        self.lang = lang
        self.locale = locale
        self.seed = seed
        self._anonymizer: Any | None = None

    def redact(self, message: HL7Message) -> HL7Message:
        """Apply configured rules in-place and return the message."""

        for segment in message.segments:
            for (segment_name, position), rules in self.field_map.items():
                if segment.name != segment_name:
                    continue

                current = segment.get_field(position)
                if current is None:
                    continue

                redacted = current
                for rule in rules:
                    redacted = self._apply_rule(redacted, segment=segment, rule=rule)
                if redacted != current:
                    segment.set_field(position, redacted)

        return message

    def _apply_rule(
        self,
        value: str,
        *,
        segment: HL7Segment,
        rule: HL7FieldRule,
    ) -> str:
        if not value or not _rule_applies(segment, rule):
            return value

        if rule.action == "clear":
            return ""
        if rule.action == "hash":
            return _transform_leaf_values(
                value,
                segment.encoding,
                lambda leaf, component: self._hash_leaf(leaf, rule, component),
            )
        if rule.action == "surrogate":
            return _transform_leaf_values(
                value,
                segment.encoding,
                lambda leaf, component: self._surrogate_leaf(leaf, rule, component),
            )
        if rule.action == "date-shift":
            return _transform_leaf_values(
                value,
                segment.encoding,
                lambda leaf, component: self._date_shift_leaf(leaf),
            )
        if rule.action == "redact_text":
            return _transform_leaf_values(
                value,
                segment.encoding,
                lambda leaf, component: self._redact_text_leaf(leaf, rule),
            )

        raise ValueError(f"unsupported HL7 field action: {rule.action}")

    def _hash_leaf(
        self,
        value: str,
        rule: HL7FieldRule,
        component: int,
    ) -> str:
        if value == "":
            return value

        label = rule.component_labels.get(component, rule.label)
        digest = hashlib.sha256(
            f"{rule.hash_salt}|{label}|{value}".encode("utf-8")
        ).hexdigest()
        return f"[{label}_HASH_{digest[:12]}]"

    def _surrogate_leaf(
        self,
        value: str,
        rule: HL7FieldRule,
        component: int,
    ) -> str:
        if value == "":
            return value

        label = rule.component_labels.get(component, rule.label)
        return str(
            self._anonymizer_or_default().surrogate(
                value,
                label,
                lang=self.lang,
                locale=self.locale,
            )
        )

    def _date_shift_leaf(self, value: str) -> str:
        if value == "":
            return value
        shifted = _shift_hl7_date(value, self.date_shift_days)
        return shifted if shifted is not None else ""

    def _redact_text_leaf(self, value: str, rule: HL7FieldRule) -> str:
        if value.strip() == "":
            return value

        kwargs = {"method": "mask", **self.deidentify_kwargs, **rule.deidentify_kwargs}
        result = self._deidentifier_or_default()(value, **kwargs)
        if isinstance(result, str):
            return result

        try:
            return str(result.deidentified_text)
        except AttributeError as exc:
            raise TypeError(
                "HL7 text deidentifier must return a string or an object with "
                "deidentified_text"
            ) from exc

    def _deidentifier_or_default(self) -> TextDeidentifier:
        if self.deidentifier is not None:
            return self.deidentifier

        from openmed.core.pii import deidentify

        return deidentify

    def _anonymizer_or_default(self) -> Any:
        if self._anonymizer is None:
            from openmed.core.anonymizer import Anonymizer

            self._anonymizer = Anonymizer(
                lang=self.lang,
                locale=self.locale,
                consistent=True,
                seed=self.seed,
            )
        return self._anonymizer


def parse_hl7v2(message: str) -> HL7Message:
    """Parse an HL7 v2 message while preserving delimiters and segment order."""

    return HL7Message.parse(message)


def redact_hl7v2(
    message_or_path: str | Path,
    *,
    field_map: Mapping[FieldKey | str, Any] | None = None,
    deidentifier: TextDeidentifier | None = None,
    deidentify_kwargs: Mapping[str, Any] | None = None,
    date_shift_days: int | None = None,
    lang: str = "en",
    locale: str | None = None,
    seed: int | None = 0,
) -> str:
    """Redact PHI from an HL7 v2 message or UTF-8 message file.

    Args:
        message_or_path: HL7 v2 message text, a path string, or a ``Path``.
        field_map: Optional replacement or extension map keyed by
            ``("PID", 5)`` or ``"PID-5"``.
        deidentifier: Optional text de-identifier used for OBX/NTE free text.
            Defaults to :func:`openmed.core.pii.deidentify`.
        deidentify_kwargs: Keyword arguments forwarded to the text
            de-identifier. ``method="mask"`` is used by default.
        date_shift_days: Optional date shift offset. When omitted, one non-zero
            offset is selected and reused for every shifted date in the message.
        lang: Language forwarded to surrogate generation.
        locale: Optional Faker locale override for surrogate generation.
        seed: Optional deterministic surrogate seed. The default keeps
            structured surrogates stable for repeatable pipelines.

    Returns:
        Redacted HL7 v2 message text preserving segment separators and
        field/component delimiters.
    """

    message_text = _read_message_or_path(message_or_path)
    message = HL7Message.parse(message_text)
    redactor = _HL7V2Redactor(
        field_map=field_map,
        deidentifier=deidentifier,
        deidentify_kwargs=deidentify_kwargs,
        date_shift_days=date_shift_days,
        lang=lang,
        locale=locale,
        seed=seed,
    )
    return redactor.redact(message).serialize()


def _normalize_field_map(
    field_map: Mapping[FieldKey | str, Any],
) -> dict[FieldKey, tuple[HL7FieldRule, ...]]:
    normalized: dict[FieldKey, tuple[HL7FieldRule, ...]] = {}
    for key, value in field_map.items():
        normalized[_normalize_field_key(key)] = tuple(_coerce_rules(value))
    return normalized


def _coerce_rules(value: Any) -> tuple[HL7FieldRule, ...]:
    if isinstance(value, HL7FieldRule) or isinstance(value, Mapping):
        return (_coerce_rule(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_coerce_rule(item) for item in value)
    raise TypeError(f"invalid HL7 field rule: {value!r}")


def _coerce_rule(value: Any) -> HL7FieldRule:
    if isinstance(value, HL7FieldRule):
        return value
    if isinstance(value, Mapping):
        return HL7FieldRule(**dict(value))
    raise TypeError(f"invalid HL7 field rule: {value!r}")


def _normalize_field_key(key: FieldKey | str) -> FieldKey:
    if isinstance(key, tuple) and len(key) == 2:
        segment, position = key
        return (str(segment).upper(), int(position))

    if isinstance(key, str):
        segment, separator, position = key.partition("-")
        if separator and segment and position:
            return (segment.upper(), int(position))

    raise ValueError(f"invalid HL7 field key {key!r}; use ('PID', 5) or 'PID-5'")


def _rule_applies(segment: HL7Segment, rule: HL7FieldRule) -> bool:
    if not rule.allowed_type_values:
        return True
    if rule.type_field is None:
        return True

    value = segment.get_field(rule.type_field) or ""
    value_type = value.split(segment.encoding.component, 1)[0].upper()
    return value_type in set(rule.allowed_type_values)


def _transform_leaf_values(
    value: str,
    encoding: HL7V2Encoding,
    transform: Callable[[str, int], str],
) -> str:
    repetitions = value.split(encoding.repetition)
    redacted_repetitions: list[str] = []
    for repetition in repetitions:
        components = repetition.split(encoding.component)
        redacted_components: list[str] = []
        for component_index, component in enumerate(components, start=1):
            subcomponents = component.split(encoding.subcomponent)
            redacted_subcomponents = [
                transform(subcomponent, component_index)
                for subcomponent in subcomponents
            ]
            redacted_components.append(
                encoding.subcomponent.join(redacted_subcomponents)
            )
        redacted_repetitions.append(encoding.component.join(redacted_components))
    return encoding.repetition.join(redacted_repetitions)


def _field_index(segment_name: str, position: int) -> int | None:
    if position < 1:
        raise ValueError("HL7 field positions are one-based")
    if segment_name == "MSH":
        if position == 1:
            return None
        return position - 2
    return position - 1


def _read_message_or_path(message_or_path: str | Path) -> str:
    if isinstance(message_or_path, Path):
        return message_or_path.read_text(encoding="utf-8")

    value = str(message_or_path)
    if _looks_like_hl7_message(value):
        return value

    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return value


def _looks_like_hl7_message(value: str) -> bool:
    stripped = value.lstrip("\ufeff")
    return len(stripped) >= 4 and stripped.startswith("MSH")


def _detect_segment_separator(message: str) -> str:
    if "\r\n" in message:
        return "\r\n"
    if "\r" in message:
        return "\r"
    if "\n" in message:
        return "\n"
    return "\r"


def _shift_hl7_date(value: str, days: int) -> str | None:
    match = _HL7_DATE_RE.match(value)
    if match:
        ymd = match.group("ymd")
        shifted = _shift_date_parts(
            int(ymd[0:4]),
            int(ymd[4:6]),
            int(ymd[6:8]),
            days,
        )
        if shifted is None:
            return None
        return (
            shifted.strftime("%Y%m%d")
            + match.group("time")
            + (match.group("fraction") or "")
            + (match.group("tz") or "")
        )

    match = _ISO_DATE_RE.match(value)
    if match:
        shifted = _shift_date_parts(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            days,
        )
        if shifted is None:
            return None
        separator = match.group("sep")
        return shifted.strftime(f"%Y{separator}%m{separator}%d")

    return None


def _shift_date_parts(year: int, month: int, day: int, days: int) -> date | None:
    try:
        return date(year, month, day) + timedelta(days=days)
    except ValueError:
        return None


def _random_nonzero_shift() -> int:
    choices = [value for value in range(-365, 366) if value != 0]
    return random.SystemRandom().choice(choices)


def _char_at_or_default(value: str, index: int, default: str) -> str:
    try:
        return value[index]
    except IndexError:
        return default


__all__ = [
    "DEFAULT_FIELD_MAP",
    "DEFAULT_NOTE_VALUE_TYPES",
    "SUPPORTED_MESSAGE_TYPES",
    "FieldAction",
    "FieldKey",
    "HL7FieldRule",
    "HL7Message",
    "HL7Segment",
    "HL7V2Encoding",
    "TextDeidentifier",
    "parse_hl7v2",
    "redact_hl7v2",
]
