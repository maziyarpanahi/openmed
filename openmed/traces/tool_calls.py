"""Structure-aware redaction for tool-call arguments and results.

Tool traces commonly contain JSON objects in ``arguments`` and ``result``
fields. Redacting their serialized representation can change valid JSON into
plain text, so this module walks the configured content paths and only applies
the text redactor to string leaves. JSON-encoded argument/result strings are
decoded, transformed, and serialized deterministically; malformed JSON falls
back to the text redactor without including the payload in an exception or
report.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

ContentPath: TypeAlias = str | Sequence[str | int]
ContentPathSpec: TypeAlias = ContentPath | Sequence[ContentPath] | None
TextRedactor: TypeAlias = Callable[[str], Any]

DEFAULT_CONTENT_PATHS: tuple[str, ...] = (
    "arguments",
    "function.arguments",
    "result",
)


class ToolCallRedactionError(ValueError):
    """Raised when a tool-call payload cannot be safely redacted."""


@dataclass(frozen=True)
class ToolCallRedactionReport:
    """PHI-safe counts and locations produced by a tool-call redaction."""

    redacted_leaf_count: int = 0
    structured_payload_count: int = 0
    malformed_payload_count: int = 0
    redacted_paths: tuple[str, ...] = ()
    malformed_payload_paths: tuple[str, ...] = ()
    content_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report without source or replacement text."""

        return {
            "redacted_leaf_count": self.redacted_leaf_count,
            "structured_payload_count": self.structured_payload_count,
            "malformed_payload_count": self.malformed_payload_count,
            "redacted_paths": list(self.redacted_paths),
            "malformed_payload_paths": list(self.malformed_payload_paths),
            "content_paths": list(self.content_paths),
        }

    def summary(self) -> str:
        """Return a compact summary containing counts and no payload content."""

        return (
            "Tool-call redaction: "
            f"{self.redacted_leaf_count} string leaf(s), "
            f"{self.structured_payload_count} structured payload(s), "
            f"{self.malformed_payload_count} malformed payload(s)."
        )


@dataclass(frozen=True)
class ToolCallRedactionResult:
    """Redacted tool-call value paired with a PHI-safe processing report."""

    value: Any
    report: ToolCallRedactionReport

    def to_dict(self) -> dict[str, Any]:
        """Return the redacted value and report in a JSON-safe envelope."""

        return {"value": self.value, "report": self.report.to_dict()}


@dataclass
class _RedactionState:
    redactor: TextRedactor
    redacted_paths: set[str] = field(default_factory=set)
    malformed_payload_paths: set[str] = field(default_factory=set)
    visited_containers: set[int] = field(default_factory=set)
    visited_leaf_paths: set[tuple[str | int, ...]] = field(default_factory=set)
    redacted_leaf_count: int = 0
    structured_payload_count: int = 0


def redact_tool_call(
    tool_call: Any,
    *,
    content_paths: ContentPathSpec = None,
    text_redactor: TextRedactor | None = None,
    models: Any | None = None,
    policy: Any | None = None,
    lang: str = "en",
) -> Any:
    """Return a structure-preserving redaction of one tool-call payload.

    Args:
        tool_call: JSON-compatible tool-call mapping or trace value. The input
            is deep-copied and is never mutated.
        content_paths: Dot paths, JSON Pointer paths, or sequences of path
            segments whose values should be redacted. ``*`` traverses every
            object key or array item. The default covers top-level
            ``arguments``, ``function.arguments``, and ``result`` fields.
        text_redactor: Callable receiving each targeted string leaf. It may
            return a string or an OpenMed-style result exposing
            ``deidentified_text``. Supplying this callable keeps the workflow
            deterministic and offline-friendly.
        models: Optional callable, mapping, or object exposing a text redactor.
            This is a convenience alternative to ``text_redactor``.
        policy: Optional policy passed to OpenMed's lazy default redactor.
        lang: Language passed to OpenMed's lazy default redactor.

    Returns:
        A deep-copied payload with the same object, array, numeric, boolean,
        and null structure. Only string leaves under the configured paths are
        changed.

    Raises:
        ToolCallRedactionError: If the payload cannot be copied or the text
            redactor fails. Exception text contains only a safe content path.
    """

    return redact_tool_call_with_report(
        tool_call,
        content_paths=content_paths,
        text_redactor=text_redactor,
        models=models,
        policy=policy,
        lang=lang,
    ).value


def redact_tool_call_with_report(
    tool_call: Any,
    *,
    content_paths: ContentPathSpec = None,
    text_redactor: TextRedactor | None = None,
    models: Any | None = None,
    policy: Any | None = None,
    lang: str = "en",
) -> ToolCallRedactionResult:
    """Redact one tool-call payload and return a PHI-safe processing report."""

    paths = _normalize_content_paths(content_paths)
    redactor = _resolve_text_redactor(
        text_redactor,
        models=models,
        policy=policy,
        lang=lang,
    )

    try:
        value = copy.deepcopy(tool_call)
    except Exception:
        raise ToolCallRedactionError(
            "tool-call payload could not be copied safely"
        ) from None

    state = _RedactionState(redactor=redactor)
    for path in paths:
        value = _redact_path(value, path, (), state)

    report = ToolCallRedactionReport(
        redacted_leaf_count=state.redacted_leaf_count,
        structured_payload_count=state.structured_payload_count,
        malformed_payload_count=len(state.malformed_payload_paths),
        redacted_paths=tuple(sorted(state.redacted_paths)),
        malformed_payload_paths=tuple(sorted(state.malformed_payload_paths)),
        content_paths=tuple(_format_path(path) for path in paths),
    )
    return ToolCallRedactionResult(value=value, report=report)


def redact_tool_calls(
    tool_calls: Iterable[Any],
    *,
    content_paths: ContentPathSpec = None,
    text_redactor: TextRedactor | None = None,
    models: Any | None = None,
    policy: Any | None = None,
    lang: str = "en",
) -> list[Any]:
    """Redact an iterable of tool-call payloads independently and in order."""

    if isinstance(tool_calls, (str, bytes, Mapping)):
        raise TypeError("tool_calls must be an iterable of payloads")

    paths = _normalize_content_paths(content_paths)
    redactor = _resolve_text_redactor(
        text_redactor,
        models=models,
        policy=policy,
        lang=lang,
    )
    return [
        redact_tool_call(
            tool_call,
            content_paths=paths,
            text_redactor=redactor,
        )
        for tool_call in tool_calls
    ]


def _normalize_content_paths(
    content_paths: ContentPathSpec,
) -> tuple[tuple[str | int, ...], ...]:
    if content_paths is None:
        candidates: tuple[Any, ...] = DEFAULT_CONTENT_PATHS
    elif isinstance(content_paths, (str, bytes, int)):
        candidates = (content_paths,)
    else:
        supplied = tuple(content_paths)
        if supplied and any(isinstance(segment, int) for segment in supplied):
            candidates = (supplied,)
        else:
            candidates = supplied

    parsed = tuple(_parse_content_path(candidate) for candidate in candidates)
    unique = sorted(set(parsed), key=_path_sort_key)
    selected: list[tuple[str | int, ...]] = []
    for candidate in unique:
        if any(_path_covers(existing, candidate) for existing in selected):
            continue
        selected.append(candidate)
    return tuple(selected)


def _parse_content_path(path: Any) -> tuple[str | int, ...]:
    if isinstance(path, int) and not isinstance(path, bool):
        if path < 0:
            raise ValueError("content path indexes must be non-negative")
        return (path,)

    if isinstance(path, str):
        if path in ("", "$"):
            return ()
        if path.startswith("/"):
            parts = path[1:].split("/")
            return tuple(
                _parse_path_segment(_unescape_pointer_part(part)) for part in parts
            )
        raw_parts = path.split(".")
        if any(part == "" for part in raw_parts):
            raise ValueError("content paths must not contain empty segments")
        parts: list[str | int] = []
        for part in raw_parts:
            if part.endswith("[]"):
                base = part[:-2]
                if not base:
                    raise ValueError("content path array segments need a field name")
                parts.extend((_parse_path_segment(base), "*"))
            else:
                parts.append(_parse_path_segment(part))
        return tuple(parts)

    if isinstance(path, Sequence) and not isinstance(path, (str, bytes)):
        return tuple(_parse_path_segment(segment) for segment in path)

    raise TypeError("each content path must be a string or sequence of segments")


def _parse_path_segment(segment: Any) -> str | int:
    if isinstance(segment, bool):
        raise TypeError("content path segments cannot be booleans")
    if isinstance(segment, int):
        if segment < 0:
            raise ValueError("content path indexes must be non-negative")
        return segment
    if not isinstance(segment, str):
        raise TypeError("content path segments must be strings or integers")
    if not segment:
        raise ValueError("content path segments must not be empty")
    if segment == "**":
        raise ValueError("recursive ** content paths are not supported")
    return "*" if segment == "[]" else segment


def _unescape_pointer_part(part: str) -> str:
    return part.replace("~1", "/").replace("~0", "~")


def _path_sort_key(path: tuple[str | int, ...]) -> tuple[Any, ...]:
    return (len(path), tuple((type(part).__name__, str(part)) for part in path))


def _path_covers(prefix: tuple[str | int, ...], path: tuple[str | int, ...]) -> bool:
    if len(prefix) > len(path):
        return False
    return all(
        prefix_part == "*" or prefix_part == path_part
        for prefix_part, path_part in zip(prefix, path)
    )


def _redact_path(
    value: Any,
    path: tuple[str | int, ...],
    location: tuple[str | int, ...],
    state: _RedactionState,
) -> Any:
    if not path:
        return _redact_target(value, location, state)

    head, *tail = path
    remainder = tuple(tail)

    if isinstance(value, Mapping):
        target = value if isinstance(value, dict) else dict(value)
        if head == "*":
            keys = _stable_keys(target)
        else:
            key: Any = head
            if key not in target and isinstance(head, int) and str(head) in target:
                key = str(head)
            keys = [key] if key in target else []
        for key in keys:
            target[key] = _redact_path(target[key], remainder, location + (key,), state)
        return target

    if isinstance(value, (list, tuple)):
        target = list(value)
        if head == "*":
            indexes = range(len(target))
        else:
            index = _path_index(head)
            indexes = (index,) if index is not None and index < len(target) else ()
        for index in indexes:
            target[index] = _redact_path(
                target[index], remainder, location + (index,), state
            )
        return tuple(target) if isinstance(value, tuple) else target

    return value


def _redact_target(
    value: Any, location: tuple[str | int, ...], state: _RedactionState
) -> Any:
    if isinstance(value, str):
        return _redact_string_payload(value, location, state)
    return _redact_value(value, location, state)


def _redact_string_payload(
    value: str,
    location: tuple[str | int, ...],
    state: _RedactionState,
) -> str:
    stripped = value.lstrip()
    if not stripped or stripped[0] not in '[{"':
        return _redact_leaf(value, location, state)

    try:
        parsed = json.loads(stripped, parse_constant=_reject_nonstandard_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        if stripped[0] in "[{":
            state.malformed_payload_paths.add(_format_path(location))
        return _redact_leaf(value, location, state)

    state.structured_payload_count += 1
    redacted = _redact_value(parsed, location, state)
    try:
        return json.dumps(
            redacted,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        state.malformed_payload_paths.add(_format_path(location))
        return _redact_leaf(value, location, state)


def _reject_nonstandard_json(value: str) -> None:
    raise ValueError(f"unsupported JSON constant {value}")


def _redact_value(
    value: Any,
    location: tuple[str | int, ...],
    state: _RedactionState,
) -> Any:
    if isinstance(value, str):
        return _redact_leaf(value, location, state)

    if isinstance(value, Mapping):
        identifier = id(value)
        if identifier in state.visited_containers:
            return value
        state.visited_containers.add(identifier)
        target = value if isinstance(value, dict) else dict(value)
        for key in _stable_keys(target):
            target[key] = _redact_value(target[key], location + (key,), state)
        return target

    if isinstance(value, (list, tuple)):
        identifier = id(value)
        if identifier in state.visited_containers:
            return value
        state.visited_containers.add(identifier)
        target = [
            _redact_value(item, location + (index,), state)
            for index, item in enumerate(value)
        ]
        return tuple(target) if isinstance(value, tuple) else target

    return value


def _redact_leaf(
    value: str,
    location: tuple[str | int, ...],
    state: _RedactionState,
) -> str:
    if location in state.visited_leaf_paths or not value:
        state.visited_leaf_paths.add(location)
        return value
    state.visited_leaf_paths.add(location)

    try:
        redacted = _coerce_redacted_text(state.redactor(value))
    except Exception:
        raise ToolCallRedactionError(
            f"text redactor failed at {_format_path(location)}"
        ) from None

    if redacted != value:
        state.redacted_leaf_count += 1
        state.redacted_paths.add(_format_path(location))
    return redacted


def _stable_keys(value: Mapping[Any, Any]) -> list[Any]:
    return sorted(value, key=lambda key: (type(key).__name__, str(key)))


def _path_index(segment: str | int) -> int | None:
    if isinstance(segment, int):
        return segment
    if segment.isdecimal():
        return int(segment)
    return None


def _format_path(path: Sequence[str | int]) -> str:
    if not path:
        return "$"
    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif part == "*":
            rendered += ".*"
        elif isinstance(part, str) and part.isidentifier():
            rendered += f".{part}"
        else:
            rendered += "[" + json.dumps(str(part), ensure_ascii=False) + "]"
    return rendered


def _resolve_text_redactor(
    explicit: TextRedactor | None,
    *,
    models: Any | None,
    policy: Any | None,
    lang: str,
) -> TextRedactor:
    if explicit is not None:
        if not callable(explicit):
            raise TypeError("text_redactor must be callable")
        return explicit

    candidate: Any = None
    if callable(models):
        candidate = models
    elif isinstance(models, Mapping):
        for key in ("text_redactor", "deidentifier", "redactor"):
            if callable(models.get(key)):
                candidate = models[key]
                break
    else:
        for attribute in ("text_redactor", "deidentifier", "redactor"):
            possible = getattr(models, attribute, None)
            if callable(possible):
                candidate = possible
                break

    if candidate is not None:
        return candidate

    return _default_text_redactor(policy=policy, lang=lang)


def _default_text_redactor(*, policy: Any | None, lang: str) -> TextRedactor:
    def redact(text: str) -> str:
        from openmed.core.pii import deidentify

        result = deidentify(text, method="mask", lang=lang, policy=policy)
        return _coerce_redacted_text(result)

    return redact


def _coerce_redacted_text(result: Any) -> str:
    if hasattr(result, "deidentified_text"):
        return str(result.deidentified_text)
    if isinstance(result, Mapping):
        for key in ("deidentified_text", "redacted_text", "text"):
            if key in result:
                return str(result[key])
    return str(result)


__all__ = [
    "DEFAULT_CONTENT_PATHS",
    "ContentPath",
    "TextRedactor",
    "ToolCallRedactionError",
    "ToolCallRedactionReport",
    "ToolCallRedactionResult",
    "redact_tool_call",
    "redact_tool_call_with_report",
    "redact_tool_calls",
]
