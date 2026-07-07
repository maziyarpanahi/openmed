"""vCard and iCalendar PHI redaction.

The parser handles the shared RFC content-line shape used by ``.vcf`` and
``.ics`` files: unfold continuation lines, redact selected properties, then
fold output lines back to a standard client-friendly width.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.labels import (
    EMAIL,
    LOCATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    STREET_ADDRESS,
)

from .base import ExtractedDocument, register_handler

ACTION_MASK = "mask"
ACTION_HASH = "hash"
ACTION_REMOVE = "remove"
ACTION_KEEP = "keep"
ACTION_FREE_TEXT_REDACT = "free_text_redact"

SUPPORTED_ACTIONS = frozenset(
    {
        ACTION_MASK,
        ACTION_HASH,
        ACTION_REMOVE,
        ACTION_KEEP,
        ACTION_FREE_TEXT_REDACT,
    }
)

TextRedactor = Callable[[str], str]

_DIRECT_PROPERTY_LABELS = {
    "FN": PERSON,
    "N": PERSON,
    "TEL": PHONE,
    "EMAIL": EMAIL,
    "ATTENDEE": EMAIL,
    "ORGANIZER": EMAIL,
    "ADR": STREET_ADDRESS,
    "ORG": ORGANIZATION,
    "LOCATION": LOCATION,
}
_FREE_TEXT_PROPERTIES = frozenset({"NOTE", "SUMMARY", "DESCRIPTION"})
_STRUCTURED_PROPERTIES = frozenset({"N", "ADR"})
_CALENDAR_ADDRESS_PROPERTIES = frozenset({"ATTENDEE", "ORGANIZER"})
_CALENDAR_CN_PROPERTIES = frozenset({"ATTENDEE", "ORGANIZER"})
_URI_VALUE_PROPERTIES = frozenset({"EMAIL", "TEL", "ATTENDEE", "ORGANIZER"})
_PROPERTY_NAME_RE = re.compile(r"^[A-Za-z0-9-]+$")


@dataclass(frozen=True)
class _RedactionOptions:
    action_overrides: Mapping[str, str]
    text_redactor: TextRedactor
    lang: str


def redact_contacts_calendar(
    source: str | os.PathLike[str] | Any,
    *,
    policy: Any | None = None,
    models: Any | None = None,
    lang: str = "en",
) -> ExtractedDocument:
    """Redact PHI-bearing vCard and iCalendar properties.

    Args:
        source: A filesystem path, raw text string, or text file-like object.
        policy: Optional string de-identification policy, or mapping with
            ``action_overrides`` and ``deidentify_policy`` keys.
        models: Optional callable, mapping, or object supplying a text redactor.
        lang: OpenMed PII language code for free-text redaction.

    Returns:
        Redacted text in an :class:`ExtractedDocument`.
    """
    text, path = _read_source(source)
    newline = _detect_newline(text)
    had_final_newline = text.endswith(("\n", "\r"))
    options = _redaction_options(policy=policy, models=models, lang=lang)

    logical_lines = _unfold_lines(text)
    redacted_lines = [_redact_content_line(line, options) for line in logical_lines]
    folded = newline.join(_fold_line(line, newline=newline) for line in redacted_lines)
    if had_final_newline:
        folded += newline

    return ExtractedDocument(
        text=folded,
        metadata={
            "format": _format_name(text, path),
            "line_count": len(logical_lines),
        },
    )


def _contacts_calendar_handler(
    path: str | os.PathLike[str],
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return redact_contacts_calendar(
        path,
        policy=policy,
        models=models,
        lang=lang or "en",
    )


def _read_source(source: str | os.PathLike[str] | Any) -> tuple[str, str | None]:
    if hasattr(source, "read"):
        return str(source.read()), None
    if isinstance(source, os.PathLike):
        path = Path(source)
        return path.read_text(encoding="utf-8"), str(path)
    if isinstance(source, str):
        if "\n" in source or "\r" in source:
            return source, None
        path = Path(source)
        if path.exists():
            return path.read_text(encoding="utf-8"), str(path)
        return source, None
    raise TypeError("source must be a path, text content, or text file-like object")


def _detect_newline(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _unfold_lines(text: str) -> list[str]:
    logical_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith((" ", "\t")) and logical_lines:
            logical_lines[-1] += line[1:]
        else:
            logical_lines.append(line)
    return logical_lines


def _fold_line(line: str, *, newline: str) -> str:
    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0
    limit = 75

    for char in line:
        char_bytes = len(char.encode("utf-8"))
        if current and current_bytes + char_bytes > limit:
            chunks.append("".join(current))
            current = [char]
            current_bytes = char_bytes
            limit = 74
        else:
            current.append(char)
            current_bytes += char_bytes

    chunks.append("".join(current))
    if len(chunks) == 1:
        return chunks[0]
    return chunks[0] + "".join(f"{newline} {chunk}" for chunk in chunks[1:])


def _redact_content_line(line: str, options: _RedactionOptions) -> str:
    separator = _find_unquoted(line, ":")
    if separator is None:
        return line

    prefix = line[:separator]
    value = line[separator + 1 :]
    property_name = _property_name(prefix)
    if property_name is None:
        return line

    redacted_prefix = _redact_parameters(prefix, property_name, options)
    redacted_value = _redact_value(property_name, prefix, value, options)
    return f"{redacted_prefix}:{redacted_value}"


def _property_name(prefix: str) -> str | None:
    property_token = _split_unquoted(prefix, ";", maxsplit=1)[0]
    name = property_token.rsplit(".", 1)[-1]
    if not name or _PROPERTY_NAME_RE.fullmatch(name) is None:
        return None
    return name.upper()


def _redact_parameters(
    prefix: str,
    property_name: str,
    options: _RedactionOptions,
) -> str:
    if property_name not in _CALENDAR_CN_PROPERTIES:
        return prefix

    tokens = _split_unquoted(prefix, ";")
    redacted = [tokens[0]]
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if separator and key.upper() == "CN":
            action = _action_for("CN", PERSON, ACTION_MASK, options)
            value = _redact_parameter_value(
                value,
                label=PERSON,
                action=action,
                options=options,
            )
            redacted.append(f"{key}={value}")
        else:
            redacted.append(token)
    return ";".join(redacted)


def _redact_value(
    property_name: str,
    prefix: str,
    value: str,
    options: _RedactionOptions,
) -> str:
    if property_name in _FREE_TEXT_PROPERTIES:
        action = _action_for(property_name, None, ACTION_FREE_TEXT_REDACT, options)
        return _apply_action(value, label=None, action=action, options=options)

    label = _DIRECT_PROPERTY_LABELS.get(property_name)
    if label is None:
        return value

    action = _action_for(property_name, label, ACTION_MASK, options)
    if property_name in _STRUCTURED_PROPERTIES:
        return _redact_structured_value(
            value, label=label, action=action, options=options
        )
    if _is_uri_valued(property_name, prefix, value):
        return _redact_uri_value(
            property_name,
            value,
            label=label,
            action=action,
            options=options,
        )
    if property_name in _CALENDAR_ADDRESS_PROPERTIES:
        return _redact_calendar_address(
            value, label=label, action=action, options=options
        )
    return _apply_action(value, label=label, action=action, options=options)


def _redact_structured_value(
    value: str,
    *,
    label: str,
    action: str,
    options: _RedactionOptions,
) -> str:
    parts = _split_escaped_value(value, ";")
    return ";".join(
        _apply_action(part, label=label, action=action, options=options)
        if part
        else part
        for part in parts
    )


def _redact_calendar_address(
    value: str,
    *,
    label: str,
    action: str,
    options: _RedactionOptions,
) -> str:
    if value.lower().startswith("mailto:") and action != ACTION_REMOVE:
        return value[:7] + _apply_action(
            value[7:],
            label=label,
            action=action,
            options=options,
        )
    return _apply_action(value, label=label, action=action, options=options)


def _is_uri_valued(property_name: str, prefix: str, value: str) -> bool:
    if property_name not in _URI_VALUE_PROPERTIES:
        return False

    lower_value = value.lower()
    if property_name in _CALENDAR_ADDRESS_PROPERTIES:
        return lower_value.startswith("mailto:")
    if property_name == "EMAIL" and lower_value.startswith("mailto:"):
        return True
    if property_name == "TEL" and lower_value.startswith("tel:"):
        return True

    parameter = _parameter_value(prefix, "VALUE")
    return parameter is not None and parameter.lower() == "uri"


def _parameter_value(prefix: str, name: str) -> str | None:
    for token in _split_unquoted(prefix, ";")[1:]:
        key, separator, value = token.partition("=")
        if separator and key.upper() == name.upper():
            return _unquote_parameter_value(value)
    return None


def _unquote_parameter_value(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def _redact_uri_value(
    property_name: str,
    value: str,
    *,
    label: str,
    action: str,
    options: _RedactionOptions,
) -> str:
    if action == ACTION_KEEP or not value:
        return value
    if action == ACTION_REMOVE:
        return ""
    if action == ACTION_MASK:
        return _masked_uri_value(property_name)
    if action == ACTION_HASH:
        return _hashed_uri_value(property_name, value)
    return _apply_action(value, label=label, action=action, options=options)


def _masked_uri_value(property_name: str) -> str:
    if property_name == "TEL":
        return "tel:+10000000000"
    return "mailto:redacted@example.invalid"


def _hashed_uri_value(property_name: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if property_name == "TEL":
        digits = str(int(digest[:12], 16) % 10_000_000_000).zfill(10)
        return f"tel:+1{digits}"
    return f"mailto:email-{digest[:8]}@example.invalid"


def _redact_parameter_value(
    value: str,
    *,
    label: str,
    action: str,
    options: _RedactionOptions,
) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        inner = value[1:-1]
        redacted = _apply_action(inner, label=label, action=action, options=options)
        return f'"{redacted}"'
    return _apply_action(value, label=label, action=action, options=options)


def _apply_action(
    value: str,
    *,
    label: str | None,
    action: str,
    options: _RedactionOptions,
) -> str:
    if action == ACTION_KEEP or not value:
        return value
    if action == ACTION_FREE_TEXT_REDACT:
        return str(options.text_redactor(value))
    if action == ACTION_REMOVE:
        return ""
    if label is None:
        return value

    from openmed.core.pii import PIIEntity, _redact_entity

    entity = PIIEntity(
        text=value,
        label=label,
        entity_type=label,
        start=0,
        end=len(value),
        confidence=1.0,
        original_text=value,
    )
    return _redact_entity(entity, action, lang=options.lang)


def _action_for(
    property_name: str,
    label: str | None,
    default: str,
    options: _RedactionOptions,
) -> str:
    candidates = [property_name.upper()]
    if label is not None:
        candidates.append(label.upper())

    for candidate in candidates:
        override = options.action_overrides.get(candidate)
        if override is not None:
            action = str(override).lower()
            if action not in SUPPORTED_ACTIONS:
                raise ValueError(
                    f"unsupported contacts/calendar redaction action: {override!r}"
                )
            return action
    return default


def _redaction_options(
    *,
    policy: Any | None,
    models: Any | None,
    lang: str,
) -> _RedactionOptions:
    action_overrides: dict[str, str] = {}
    deidentify_policy = policy if isinstance(policy, str) else None

    if isinstance(policy, Mapping):
        raw_overrides = policy.get("action_overrides")
        if isinstance(raw_overrides, Mapping):
            action_overrides = {
                str(key).upper(): str(value) for key, value in raw_overrides.items()
            }
        raw_deidentify_policy = policy.get("deidentify_policy")
        if raw_deidentify_policy is not None:
            deidentify_policy = str(raw_deidentify_policy)

    text_redactor = _text_redactor_from_models(models) or _default_text_redactor(
        policy=deidentify_policy,
        lang=lang,
    )
    return _RedactionOptions(
        action_overrides=action_overrides,
        text_redactor=text_redactor,
        lang=lang,
    )


def _text_redactor_from_models(models: Any | None) -> TextRedactor | None:
    if callable(models):
        return lambda text: str(models(text))
    if isinstance(models, Mapping):
        for key in ("text_redactor", "deidentifier", "redactor"):
            candidate = models.get(key)
            if callable(candidate):
                return lambda text: str(candidate(text))
    for attribute in ("text_redactor", "deidentifier", "redactor"):
        candidate = getattr(models, attribute, None)
        if callable(candidate):
            return lambda text: str(candidate(text))
    return None


def _default_text_redactor(*, policy: str | None, lang: str) -> TextRedactor:
    def redactor(text: str) -> str:
        from openmed.core.pii import deidentify

        result = deidentify(text, method=ACTION_MASK, lang=lang, policy=policy)
        if hasattr(result, "deidentified_text"):
            return str(result.deidentified_text)
        return str(result)

    return redactor


def _find_unquoted(text: str, needle: str) -> int | None:
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if char == needle and not quoted:
            return index
    return None


def _split_unquoted(text: str, delimiter: str, *, maxsplit: int = -1) -> list[str]:
    parts: list[str] = []
    start = 0
    splits = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if char == delimiter and not quoted and (maxsplit < 0 or splits < maxsplit):
            parts.append(text[start:index])
            start = index + 1
            splits += 1
    parts.append(text[start:])
    return parts


def _split_escaped_value(text: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    escaped = False
    for char in text:
        if escaped:
            current.extend(("\\", char))
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == delimiter:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if escaped:
        current.append("\\")
    parts.append("".join(current))
    return parts


def _format_name(text: str, path: str | None) -> str:
    suffix = Path(path).suffix.lower() if path is not None else ""
    if suffix == ".vcf":
        return "vcard"
    if suffix == ".ics":
        return "icalendar"

    upper = text.upper()
    if "BEGIN:VCARD" in upper:
        return "vcard"
    if "BEGIN:VCALENDAR" in upper:
        return "icalendar"
    return "contacts_calendar"


register_handler(
    (".vcf", ".ics"),
    _contacts_calendar_handler,
    requires_multimodal=False,
)


__all__ = [
    "ACTION_FREE_TEXT_REDACT",
    "ACTION_HASH",
    "ACTION_KEEP",
    "ACTION_MASK",
    "ACTION_REMOVE",
    "SUPPORTED_ACTIONS",
    "redact_contacts_calendar",
]
