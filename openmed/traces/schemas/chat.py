"""Local-first redaction for role-based training message arrays.

The adapter only visits a message's ``content`` field.  Roles, tool-call
identifiers, and other message metadata remain outside the discovered content
paths and are copied byte-for-byte at the JSON value level.  A caller supplies
the text redactor so this module stays deterministic and never needs to load a
model or make a network request.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeGuard

PathPart: TypeAlias = str | int
ContentPath: TypeAlias = tuple[PathPart, ...]
TextRedactor: TypeAlias = Callable[[str], Any]
ContentItem: TypeAlias = tuple[ContentPath, str]

DEFAULT_MESSAGES_KEY = "messages"
DEFAULT_CONTENT_KEY = "content"

# These are the common text-bearing part types used by chat and response
# training exports.  Unknown part types are deliberately not traversed.
_TEXT_PART_TYPES = frozenset(
    {
        "input_text",
        "input_text_delta",
        "output_text",
        "output_text_delta",
        "text",
        "text_delta",
    }
)
_TOOL_RESULT_PART_TYPES = frozenset({"tool_result", "tool_response"})
_SAFE_PATH_KEYS = frozenset(
    {
        DEFAULT_MESSAGES_KEY,
        DEFAULT_CONTENT_KEY,
        "items",
        "parts",
        "text",
        "value",
    }
)
_HASHED_PATH_KEY = re.compile(r"^key_sha256_[0-9a-f]{12}$")
_INVALID_PART_TYPE = object()


class ChatSchemaError(ValueError):
    """Base class for safe role-message schema errors."""


class ChatMessageRedactionError(ChatSchemaError):
    """Raised when a role-message record cannot be safely redacted."""


# Discoverable aliases for callers using the shorter vocabulary.
ChatRedactionError = ChatMessageRedactionError
RoleMessageSchemaError = ChatSchemaError


def _plain_text(value: object) -> str | None:
    """Copy a string into a base ``str`` without calling subclass hooks."""

    if not isinstance(value, str):
        return None
    try:
        return str.encode(value, "utf-8").decode("utf-8")
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class ChatRedactionReport:
    """PHI-free counts and paths produced by one chat redaction."""

    message_count: int = 0
    text_value_count: int = 0
    redacted_text_count: int = 0
    structured_part_count: int = 0
    content_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            for name in (
                "message_count",
                "text_value_count",
                "redacted_text_count",
                "structured_part_count",
            ):
                value = getattr(self, name)
                if type(value) is not int or value < 0:
                    raise ValueError
            safe_paths = tuple(_safe_report_path(path) for path in self.content_paths)
        except Exception:  # noqa: BLE001 - report input may contain PHI
            raise ChatSchemaError("chat redaction report is invalid") from None
        object.__setattr__(self, "content_paths", safe_paths)

    @property
    def redacted_fields(self) -> int:
        """Return the number of text values changed by the redactor."""

        return self.redacted_text_count

    @property
    def text_count(self) -> int:
        """Return the number of discovered text values."""

        return self.text_value_count

    def to_dict(self) -> dict[str, Any]:
        """Return an audit-safe report without source or replacement text."""

        return {
            "message_count": self.message_count,
            "text_value_count": self.text_value_count,
            "redacted_text_count": self.redacted_text_count,
            "structured_part_count": self.structured_part_count,
            "content_paths": list(self.content_paths),
        }

    def summary(self) -> str:
        """Return a compact, value-free processing summary."""

        return (
            "Chat redaction: "
            f"{self.message_count} message(s), "
            f"{self.redacted_text_count} redacted text value(s), "
            f"{self.structured_part_count} structured part(s)."
        )


@dataclass(frozen=True, slots=True)
class ChatRedactionResult:
    """Redacted value paired with a PHI-free processing report."""

    value: Any
    report: ChatRedactionReport

    @property
    def messages(self) -> Any:
        """Return the redacted message sequence for list-oriented callers."""

        return self.value

    @property
    def record(self) -> Any:
        """Return the redacted record for record-oriented callers."""

        return self.value

    def to_dict(self) -> dict[str, Any]:
        """Return the result envelope with a safe report."""

        return {"value": self.value, "report": self.report.to_dict()}


@dataclass
class _RedactionState:
    redactor: TextRedactor
    text_value_count: int = 0
    redacted_text_count: int = 0
    structured_part_count: int = 0
    content_paths: list[str] = field(default_factory=list)

    def redact(self, value: str, path: ContentPath, *, structured: bool) -> str:
        """Apply the injected redactor and retain only aggregate state."""

        source_text = _plain_text(value)
        if source_text is None:
            raise ChatMessageRedactionError(
                f"text content is invalid at {_format_path(path)}"
            )
        self.text_value_count += 1
        if structured:
            self.structured_part_count += 1
        self.content_paths.append(_format_path(path))
        try:
            replacement = _coerce_redacted_text(self.redactor(source_text))
        except ChatMessageRedactionError:
            raise
        except Exception:
            raise ChatMessageRedactionError(
                f"text redactor failed at {_format_path(path)}"
            ) from None
        if replacement != source_text:
            self.redacted_text_count += 1
        return replacement

    def report(self, message_count: int) -> ChatRedactionReport:
        """Build an immutable, value-free report."""

        return ChatRedactionReport(
            message_count=message_count,
            text_value_count=self.text_value_count,
            redacted_text_count=self.redacted_text_count,
            structured_part_count=self.structured_part_count,
            content_paths=tuple(self.content_paths),
        )


def _is_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _walk_content(
    value: Any,
    path: ContentPath,
    *,
    structured: bool = False,
    active: set[int] | None = None,
) -> tuple[tuple[ContentPath, str, bool], ...]:
    """Discover only text slots in a known content value.

    Arbitrary nested metadata is intentionally opaque.  A direct ``text``
    field is safe to visit, while fields such as image URLs, tool names,
    identifiers, and tool inputs are preserved even when they contain strings.
    """

    if isinstance(value, str):
        text = _plain_text(value)
        if text is None:
            raise ChatSchemaError("message content text is invalid")
        return ((path, text, structured),)
    if not isinstance(value, (Mapping, Sequence)) or isinstance(
        value, (bytes, bytearray, str)
    ):
        return ()

    visited = active if active is not None else set()
    identifier = id(value)
    if identifier in visited:
        raise ChatSchemaError("cyclic message content is not supported")
    visited.add(identifier)
    try:
        if isinstance(value, Mapping):
            items: list[tuple[ContentPath, str, bool]] = []
            raw_part_type = value.get("type")
            if raw_part_type is None:
                part_type: object = None
            elif isinstance(raw_part_type, str):
                part_type = _plain_text(raw_part_type) or _INVALID_PART_TYPE
            else:
                part_type = _INVALID_PART_TYPE
            recognized_part = part_type is None or part_type in (
                _TEXT_PART_TYPES | _TOOL_RESULT_PART_TYPES
            )
            text_value = _plain_text(value.get("text"))
            if recognized_part and text_value is not None:
                items.append((path + ("text",), text_value, True))

            if part_type in _TEXT_PART_TYPES:
                for key in ("value", "content"):
                    if key in value:
                        items.extend(
                            _walk_content(
                                value[key],
                                path + (key,),
                                structured=True,
                                active=visited,
                            )
                        )
            elif part_type in _TOOL_RESULT_PART_TYPES and "content" in value:
                items.extend(
                    _walk_content(
                        value["content"],
                        path + ("content",),
                        structured=True,
                        active=visited,
                    )
                )
            elif "type" not in value and "content" in value:
                # Some exporters use {"content": ...} as a text wrapper
                # without a part type.  Keep all other keys opaque.
                items.extend(
                    _walk_content(
                        value["content"],
                        path + ("content",),
                        structured=True,
                        active=visited,
                    )
                )

            if part_type is None or part_type in (
                _TEXT_PART_TYPES | _TOOL_RESULT_PART_TYPES
            ):
                for key in ("parts", "items"):
                    if key in value and _is_sequence(value[key]):
                        items.extend(
                            _walk_content(
                                value[key],
                                path + (key,),
                                structured=True,
                                active=visited,
                            )
                        )
            return tuple(items)

        items = []
        for index, item in enumerate(value):
            items.extend(
                _walk_content(
                    item,
                    path + (index,),
                    structured=True,
                    active=visited,
                )
            )
        return tuple(items)
    finally:
        visited.remove(identifier)


def _walk_message_sequence(
    messages: Any,
    *,
    content_key: str,
    path_prefix: ContentPath = (),
) -> tuple[tuple[tuple[ContentPath, str, bool], ...], int]:
    """Return discovered content and message count for one message array."""

    normalized_content_key = _plain_text(content_key)
    if normalized_content_key is None or not normalized_content_key:
        raise ChatSchemaError("content_key must be a non-empty string")
    try:
        if not _is_sequence(messages):
            raise ChatSchemaError(
                "messages must be a list or tuple of message mappings"
            )

        items: list[tuple[ContentPath, str, bool]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ChatSchemaError("each message must be a mapping")
            if normalized_content_key in message:
                items.extend(
                    _walk_content(
                        message[normalized_content_key],
                        path_prefix + (index, normalized_content_key),
                    )
                )
        return tuple(items), len(messages)
    except ChatSchemaError:
        raise
    except Exception:
        raise ChatSchemaError("message content could not be read safely") from None


def _walk_record(
    record: Any,
    *,
    messages_key: str,
    content_key: str,
) -> tuple[tuple[tuple[ContentPath, str, bool], ...], int]:
    try:
        if not isinstance(record, Mapping):
            raise ChatSchemaError("record must be a mapping")
        if messages_key not in record:
            raise ChatSchemaError("record does not contain a messages field")
        return _walk_message_sequence(
            record[messages_key],
            content_key=content_key,
            path_prefix=(messages_key,),
        )
    except ChatSchemaError:
        raise
    except Exception:
        raise ChatSchemaError("message content could not be read safely") from None


def _normalize_path(raw_path: Any) -> ContentPath:
    if isinstance(raw_path, str):
        path_text = _plain_text(raw_path)
        if path_text is None:
            raise ChatSchemaError("content paths must be text")
        parts: list[PathPart] = []
        for part in path_text.split("."):
            if not part:
                raise ChatSchemaError("content paths must not contain empty parts")
            parts.append(int(part) if part.isdecimal() else part)
        return tuple(parts)

    if not isinstance(raw_path, Sequence) or isinstance(
        raw_path, (str, bytes, bytearray)
    ):
        raise ChatSchemaError("content paths must be sequences")
    try:
        raw_parts = tuple(raw_path)
    except Exception:
        raise ChatSchemaError("content paths could not be read") from None
    parts = []
    for part in raw_parts:
        if isinstance(part, str):
            normalized_part = _plain_text(part)
            if normalized_part is None or not normalized_part:
                raise ChatSchemaError("content path keys must not be empty")
            parts.append(normalized_part)
            continue
        if type(part) is not int:
            raise ChatSchemaError("content paths may contain only keys and indexes")
        if part < 0:
            raise ChatSchemaError("content path indexes must be non-negative")
        parts.append(part)
    if not parts:
        raise ChatSchemaError("content paths must not be empty")
    return tuple(parts)


def _path_sort_key(path: ContentPath) -> tuple[tuple[int, str], ...]:
    return tuple(
        (0, str(part)) if isinstance(part, int) else (1, part) for part in path
    )


def _validated_replacements(
    walked: Sequence[tuple[ContentPath, str, bool]],
    replacements: Any,
) -> dict[ContentPath, str]:
    if not isinstance(replacements, Mapping):
        raise ChatSchemaError("replacements must be a mapping")
    known_paths = {path for path, _text, _structured in walked}
    normalized: dict[ContentPath, str] = {}
    for raw_path, replacement in _replacement_entries(replacements):
        path = _normalize_path(raw_path)
        if path not in known_paths:
            raise ChatSchemaError("replacement path is not discovered content")
        if path in normalized:
            raise ChatSchemaError(
                "replacement paths must be unique after normalization"
            )
        replacement_text = _plain_text(replacement)
        if replacement_text is None:
            raise ChatSchemaError("replacement content must be text")
        normalized[path] = replacement_text
    return normalized


def _replacement_entries(
    replacements: Mapping[Any, Any],
) -> tuple[tuple[Any, Any], ...]:
    """Materialize replacement entries without exposing iterator failures."""

    try:
        raw_entries = tuple(replacements.items())
    except Exception:
        raise ChatSchemaError("replacements could not be read") from None
    entries: list[tuple[Any, Any]] = []
    for raw_entry in raw_entries:
        try:
            raw_path, replacement = raw_entry
        except Exception:
            raise ChatSchemaError("replacements could not be read") from None
        entries.append((raw_path, replacement))
    return tuple(entries)


def _replace_at_path(value: Any, path: ContentPath, replacement: str) -> Any:
    if not path:
        return replacement

    head, *tail = path
    remainder = tuple(tail)
    if isinstance(value, MutableMapping):
        if head not in value:
            raise ChatSchemaError("content path no longer exists")
        value[head] = _replace_at_path(value[head], remainder, replacement)
        return value
    if isinstance(value, Mapping):
        if head not in value:
            raise ChatSchemaError("content path no longer exists")
        copied = dict(value)
        copied[head] = _replace_at_path(copied[head], remainder, replacement)
        return copied
    if isinstance(value, list):
        if not isinstance(head, int) or isinstance(head, bool):
            raise ChatSchemaError("content path does not address a list")
        if head < 0 or head >= len(value):
            raise ChatSchemaError("content path index is out of range")
        value[head] = _replace_at_path(value[head], remainder, replacement)
        return value
    if isinstance(value, tuple):
        if not isinstance(head, int) or isinstance(head, bool):
            raise ChatSchemaError("content path does not address a tuple")
        if head < 0 or head >= len(value):
            raise ChatSchemaError("content path index is out of range")
        copied_items = list(value)
        copied_items[head] = _replace_at_path(
            copied_items[head], remainder, replacement
        )
        return tuple(copied_items)
    raise ChatSchemaError("content path enters a non-container value")


def _reconstruct(
    record: Any,
    walked: Sequence[tuple[ContentPath, str, bool]],
    replacements: Any,
) -> Any:
    normalized = _validated_replacements(walked, replacements)
    try:
        result = copy.deepcopy(record)
    except Exception:
        raise ChatSchemaError("record could not be copied safely") from None
    try:
        for path in sorted(normalized, key=_path_sort_key):
            result = _replace_at_path(result, path, normalized[path])
    except ChatSchemaError:
        raise
    except Exception:
        raise ChatSchemaError("record could not be reconstructed safely") from None
    return result


def _coerce_redacted_text(result: Any) -> str:
    text = _plain_text(result)
    if text is not None:
        return text
    if isinstance(result, Mapping):
        for key in ("deidentified_text", "redacted_text", "text"):
            try:
                candidate = result.get(key)
            except Exception:
                continue
            text = _plain_text(candidate)
            if text is not None:
                return text
    for attribute in ("deidentified_text", "redacted_text"):
        try:
            candidate = getattr(result, attribute)
        except Exception:
            continue
        text = _plain_text(candidate)
        if text is not None:
            return text
    raise ChatMessageRedactionError("text redactor must return text")


def _redactor_from_models(models: Any) -> TextRedactor | None:
    if callable(models):
        return models
    if isinstance(models, Mapping):
        for key in ("text_redactor", "redactor", "deidentifier"):
            try:
                candidate = models.get(key)
            except Exception:
                return None
            if callable(candidate):
                return candidate
        return None
    for attribute in ("text_redactor", "redactor", "deidentifier"):
        try:
            candidate = getattr(models, attribute, None)
        except Exception:
            continue
        if callable(candidate):
            return candidate
    return None


def _resolve_redactor(
    text_redactor: TextRedactor | None,
    redactor: TextRedactor | None,
    models: Any | None,
    configured: TextRedactor | None = None,
) -> TextRedactor:
    if text_redactor is not None and redactor is not None:
        raise ChatMessageRedactionError(
            "provide only one of text_redactor and redactor"
        )
    selected = (
        text_redactor
        if text_redactor is not None
        else redactor
        if redactor is not None
        else configured
    )
    if selected is None and models is not None:
        selected = _redactor_from_models(models)
    if not callable(selected):
        raise ChatMessageRedactionError(
            "a local text_redactor is required; no model is loaded automatically"
        )
    return selected


def _format_path(path: Sequence[PathPart]) -> str:
    if not path:
        return "$"
    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif part in _SAFE_PATH_KEYS:
            rendered += f".{part}"
        else:
            digest = hashlib.sha256(part.encode("utf-8")).hexdigest()[:12]
            rendered += f".key_sha256_{digest}"
    return rendered


def _safe_report_path(value: object) -> str:
    if isinstance(value, str) and value.startswith("$"):
        without_indexes = re.sub(r"\[\d+\]", "", value)
        parts = tuple(part for part in without_indexes[1:].split(".") if part)
        if without_indexes == "$" or all(
            part in _SAFE_PATH_KEYS or _HASHED_PATH_KEY.fullmatch(part)
            for part in parts
        ):
            return value
    raw = value if isinstance(value, str) else type(value).__qualname__
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"$.key_sha256_{digest}"


def _transform_walked(
    walked: Sequence[tuple[ContentPath, str, bool]],
    redactor: TextRedactor,
    *,
    message_count: int,
) -> tuple[dict[ContentPath, str], ChatRedactionReport]:
    state = _RedactionState(redactor=redactor)
    replacements: dict[ContentPath, str] = {}
    for path, text, structured in walked:
        replacements[path] = state.redact(text, path, structured=structured)
    return replacements, state.report(message_count)


class RoleMessageSchemaAdapter:
    """Schema adapter for records containing a ``messages`` role array.

    The structural methods implement the training-schema registry protocol:
    ``detect``, ``walk``, ``reconstruct``, and ``transform``.  Redaction
    methods require a caller-provided local callable and return a deep-copied
    value, so the source record is never modified.
    """

    name = "role_messages"

    def __init__(
        self,
        text_redactor: TextRedactor | None = None,
        *,
        redactor: TextRedactor | None = None,
        models: Any | None = None,
        messages_key: str = DEFAULT_MESSAGES_KEY,
        content_key: str = DEFAULT_CONTENT_KEY,
    ) -> None:
        normalized_messages_key = _plain_text(messages_key)
        if normalized_messages_key is None or not normalized_messages_key.strip():
            raise ChatSchemaError("messages_key must be a non-empty string")
        normalized_content_key = _plain_text(content_key)
        if normalized_content_key is None or not normalized_content_key.strip():
            raise ChatSchemaError("content_key must be a non-empty string")
        self.messages_key = normalized_messages_key.strip()
        self.content_key = normalized_content_key.strip()
        self._text_redactor = (
            _resolve_redactor(
                text_redactor,
                redactor,
                models,
            )
            if any(value is not None for value in (text_redactor, redactor, models))
            else None
        )

    def detect(self, record: Any) -> bool:
        """Return whether ``record`` has a structurally valid message array."""

        try:
            if not isinstance(record, Mapping):
                return False
            messages = record.get(self.messages_key)
            return _is_sequence(messages) and all(
                isinstance(message, Mapping) for message in messages
            )
        except Exception:
            return False

    def matches(self, record: Any) -> bool:
        """Alias for :meth:`detect`."""

        return self.detect(record)

    def walk(self, record: Any) -> tuple[ContentItem, ...]:
        """Return paths and text values in message content slots."""

        walked, _message_count = _walk_record(
            record,
            messages_key=self.messages_key,
            content_key=self.content_key,
        )
        return tuple((path, text) for path, text, _structured in walked)

    def iter_content(self, record: Any) -> tuple[ContentItem, ...]:
        """Alias for :meth:`walk`."""

        return self.walk(record)

    def reconstruct(
        self,
        record: Any,
        replacements: Mapping[ContentPath | str, str],
    ) -> Any:
        """Return a deep-copied record with validated content replacements."""

        walked, _message_count = _walk_record(
            record,
            messages_key=self.messages_key,
            content_key=self.content_key,
        )
        return _reconstruct(record, walked, replacements)

    def transform(
        self,
        record: Any,
        text_redactor: TextRedactor,
    ) -> Any:
        """Transform discovered text and return a structure-preserving copy."""

        if not callable(text_redactor):
            raise ChatMessageRedactionError("text_redactor must be callable")
        walked, message_count = _walk_record(
            record,
            messages_key=self.messages_key,
            content_key=self.content_key,
        )
        replacements, _report = _transform_walked(
            walked,
            text_redactor,
            message_count=message_count,
        )
        return _reconstruct(record, walked, replacements)

    def transform_with_report(
        self,
        record: Any,
        text_redactor: TextRedactor,
    ) -> ChatRedactionResult:
        """Transform a record and return its value-free processing report."""

        if not callable(text_redactor):
            raise ChatMessageRedactionError("text_redactor must be callable")
        walked, message_count = _walk_record(
            record,
            messages_key=self.messages_key,
            content_key=self.content_key,
        )
        replacements, report = _transform_walked(
            walked,
            text_redactor,
            message_count=message_count,
        )
        return ChatRedactionResult(
            value=_reconstruct(record, walked, replacements),
            report=report,
        )

    def redact(
        self,
        record: Any,
        text_redactor: TextRedactor | None = None,
        *,
        redactor: TextRedactor | None = None,
        models: Any | None = None,
    ) -> Any:
        """Redact a record using an explicit or constructor-configured callable."""

        selected = _resolve_redactor(
            text_redactor,
            redactor,
            models,
            configured=self._text_redactor,
        )
        return self.transform(record, selected)

    def redact_with_report(
        self,
        record: Any,
        text_redactor: TextRedactor | None = None,
        *,
        redactor: TextRedactor | None = None,
        models: Any | None = None,
    ) -> ChatRedactionResult:
        """Redact a record and return its value-free processing report."""

        selected = _resolve_redactor(
            text_redactor,
            redactor,
            models,
            configured=self._text_redactor,
        )
        return self.transform_with_report(record, selected)

    def adapt(
        self,
        record: Any,
        text_redactor: TextRedactor | None = None,
        *,
        redactor: TextRedactor | None = None,
        models: Any | None = None,
    ) -> Any:
        """Adapt ``record`` using an explicit or configured redactor."""

        return self.redact(
            record,
            text_redactor,
            redactor=redactor,
            models=models,
        )

    def __call__(
        self,
        record: Any,
        text_redactor: TextRedactor | None = None,
        *,
        redactor: TextRedactor | None = None,
        models: Any | None = None,
    ) -> Any:
        """Redact ``record`` when the adapter is used as a callable."""

        return self.redact(
            record,
            text_redactor,
            redactor=redactor,
            models=models,
        )


ChatMessageSchemaAdapter = RoleMessageSchemaAdapter
ChatSchemaAdapter = RoleMessageSchemaAdapter
RoleMessageAdapter = RoleMessageSchemaAdapter


def redact_chat_record(
    record: Any,
    text_redactor: TextRedactor | None = None,
    *,
    redactor: TextRedactor | None = None,
    models: Any | None = None,
    messages_key: str = DEFAULT_MESSAGES_KEY,
    content_key: str = DEFAULT_CONTENT_KEY,
) -> Any:
    """Return a redacted copy of one role-message training record."""

    adapter = RoleMessageSchemaAdapter(
        text_redactor=text_redactor,
        redactor=redactor,
        models=models,
        messages_key=messages_key,
        content_key=content_key,
    )
    return adapter.redact(record)


def redact_chat_record_with_report(
    record: Any,
    text_redactor: TextRedactor | None = None,
    *,
    redactor: TextRedactor | None = None,
    models: Any | None = None,
    messages_key: str = DEFAULT_MESSAGES_KEY,
    content_key: str = DEFAULT_CONTENT_KEY,
) -> ChatRedactionResult:
    """Redact one record and return a PHI-free processing report."""

    adapter = RoleMessageSchemaAdapter(
        text_redactor=text_redactor,
        redactor=redactor,
        models=models,
        messages_key=messages_key,
        content_key=content_key,
    )
    return adapter.redact_with_report(record)


def _transform_messages(
    messages: Any,
    *,
    content_key: str,
    redactor: TextRedactor,
) -> ChatRedactionResult:
    walked, message_count = _walk_message_sequence(
        messages,
        content_key=content_key,
    )
    replacements, report = _transform_walked(
        walked,
        redactor,
        message_count=message_count,
    )
    return ChatRedactionResult(
        value=_reconstruct(messages, walked, replacements),
        report=report,
    )


def redact_chat_messages(
    messages: Any,
    text_redactor: TextRedactor | None = None,
    *,
    redactor: TextRedactor | None = None,
    models: Any | None = None,
    content_key: str = DEFAULT_CONTENT_KEY,
) -> Any:
    """Return an ordered, redacted copy of a role-message array.

    Only string content and recognized text parts are passed to the redactor.
    The input array and every message mapping are left untouched.
    """

    selected = _resolve_redactor(text_redactor, redactor, models)
    return _transform_messages(
        messages,
        content_key=content_key,
        redactor=selected,
    ).value


def redact_chat_messages_with_report(
    messages: Any,
    text_redactor: TextRedactor | None = None,
    *,
    redactor: TextRedactor | None = None,
    models: Any | None = None,
    content_key: str = DEFAULT_CONTENT_KEY,
) -> ChatRedactionResult:
    """Redact a role-message array and return a PHI-free processing report."""

    selected = _resolve_redactor(text_redactor, redactor, models)
    return _transform_messages(
        messages,
        content_key=content_key,
        redactor=selected,
    )


def walk_chat_content(
    record: Any,
    *,
    messages_key: str = DEFAULT_MESSAGES_KEY,
    content_key: str = DEFAULT_CONTENT_KEY,
) -> tuple[ContentItem, ...]:
    """Return discovered content paths without changing ``record``."""

    return RoleMessageSchemaAdapter(
        messages_key=messages_key,
        content_key=content_key,
    ).walk(record)


# Role-oriented names are useful when a caller distinguishes this schema from
# other chat payloads.  They intentionally point to the same implementation.
redact_role_messages = redact_chat_messages
redact_role_messages_with_report = redact_chat_messages_with_report
rewrite_role_messages = redact_chat_messages


__all__ = [
    "ChatMessageRedactionError",
    "ChatMessageSchemaAdapter",
    "ChatRedactionError",
    "ChatRedactionReport",
    "ChatRedactionResult",
    "ChatSchemaAdapter",
    "ChatSchemaError",
    "ContentItem",
    "ContentPath",
    "DEFAULT_CONTENT_KEY",
    "DEFAULT_MESSAGES_KEY",
    "RoleMessageAdapter",
    "RoleMessageSchemaAdapter",
    "RoleMessageSchemaError",
    "TextRedactor",
    "redact_chat_messages",
    "redact_chat_messages_with_report",
    "redact_chat_record",
    "redact_chat_record_with_report",
    "redact_role_messages",
    "redact_role_messages_with_report",
    "rewrite_role_messages",
    "walk_chat_content",
]
