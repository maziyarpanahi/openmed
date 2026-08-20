"""Stream schema-preserving content access for JSONL agent traces.

Trace records contain useful structure that must survive privacy processing:
roles, timestamps, tool calls, identifiers, and replay metadata. This module
therefore exposes only configured string locations to a caller and rewrites
the parsed record in place instead of flattening it into text.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

JsonPathSegment = str | int
JsonPath = tuple[JsonPathSegment, ...]
ContentPath = str | Sequence[JsonPathSegment]
TraceSource = str | os.PathLike[str] | Iterable[str] | TextIO
TextTransform = Callable[[str], str]

# The recursive wildcard limits the default surface to fields named
# "content". Callers handling a stricter trace schema should pass explicit
# paths instead.
DEFAULT_CONTENT_PATHS = ("**.content",)

_SAFE_PATH_KEYS = frozenset(
    {
        "arguments",
        "content",
        "event",
        "events",
        "input",
        "items",
        "messages",
        "output",
        "parts",
        "role",
        "text",
        "tool_calls",
        "value",
    }
)
_HASHED_PATH_KEY = re.compile(r"^key_sha256_[0-9a-f]{12}$")

__all__ = [
    "ContentPath",
    "DEFAULT_CONTENT_PATHS",
    "JsonPath",
    "JsonPathSegment",
    "TraceContentLocation",
    "TraceContentWalker",
    "TraceJSONLIOError",
    "TraceJSONLLineError",
    "TraceJSONLTransformError",
    "iter_jsonl_content",
    "iter_trace_content",
    "redact_trace_jsonl",
    "rewrite_jsonl_content",
    "rewrite_trace_jsonl",
    "walk_jsonl_content",
    "walk_trace_content",
    "write_trace_jsonl",
]


@dataclass(frozen=True)
class TraceContentLocation:
    """One configured string value in a JSONL trace record.

    Attributes:
        line_number: One-based physical line number in the source.
        path: Tuple of object keys and array indexes leading to the value.
        value: The string at path. It is exposed so a caller can redact it,
            but it is never copied into parser or transform diagnostics.
    """

    line_number: int
    path: JsonPath
    value: str

    @property
    def line(self) -> int:
        """Return the one-based source line number."""

        return self.line_number

    @property
    def json_path(self) -> str:
        """Return the location using a compact JSONPath-like notation."""

        return _format_json_path(self.path)

    @property
    def field_path(self) -> JsonPath:
        """Return the path as an immutable alias for callers using field terms."""

        return self.path

    def as_tuple(self) -> tuple[int, JsonPath, str]:
        """Return the location as line number, path, and value."""

        return self.line_number, self.path, self.value

    def __iter__(self) -> Iterator[Any]:
        """Allow convenient unpacking as line number, path, and value."""

        yield from self.as_tuple()


class TraceJSONLLineError(ValueError):
    """Value-free description of one invalid JSONL trace line."""

    def __init__(self, line_number: int, message: str) -> None:
        self.line_number = line_number
        self.message = message
        super().__init__(f"JSONL line {line_number}: {message}")

    @property
    def line(self) -> int:
        """Return the one-based line associated with the error."""

        return self.line_number


class TraceJSONLIOError(ValueError):
    """Value-free error for an unreadable source or unwritable destination."""


class TraceJSONLTransformError(TraceJSONLLineError):
    """Value-free error for a failed configured string transformation."""

    def __init__(self, line_number: int, path: JsonPath, message: str) -> None:
        self.path = _sanitize_error_path(path)
        self.json_path = _format_json_path(self.path)
        super().__init__(line_number, f"{message} at {self.json_path}")


class _DuplicateKeyError(ValueError):
    """Internal signal for a JSON object containing a duplicate key."""


class _NonStandardNumberError(ValueError):
    """Internal signal for NaN and infinity spellings rejected by JSONL."""


class TraceContentWalker:
    """Stream configured string locations and structure-preserving rewrites.

    content_paths accepts dotted paths, array indexes, and wildcards. The
    single-level wildcard "*" matches one object key or list item, while "**"
    matches zero or more nested object/list levels. For example,
    events.*.content and messages[*].content are valid.
    """

    def __init__(
        self,
        content_paths: Iterable[ContentPath] | ContentPath | None = None,
        *,
        paths: Iterable[ContentPath] | ContentPath | None = None,
    ) -> None:
        """Create a walker for configured redactable content paths.

        Args:
            content_paths: Dotted or tuple paths. If omitted, only fields
                named content at any depth are selected.
            paths: Alias for content_paths. Supplying both arguments is
                rejected to avoid ambiguous policy configuration.
        """

        if paths is not None:
            if content_paths is not None:
                raise TypeError("pass either content_paths or paths, not both")
            content_paths = paths
        self._content_paths = _normalise_content_paths(content_paths)

    @property
    def content_paths(self) -> tuple[JsonPath, ...]:
        """Return the immutable compiled content-path patterns."""

        return self._content_paths

    def walk(self, source: TraceSource) -> Iterator[TraceContentLocation]:
        """Yield configured string locations from a JSONL source lazily.

        Blank physical lines pass through the stream with no locations.
        Non-object JSON values and malformed JSON raise
        TraceJSONLLineError with a one-based line number.
        """

        for line_number, line in enumerate(_iter_source_lines(source), start=1):
            payload, _ = _split_line_ending(line)
            if not payload.strip():
                continue
            record = _parse_record(payload, line_number)
            yield from _locations_for_record(
                record,
                line_number=line_number,
                content_paths=self._content_paths,
            )

    def iter_locations(self, source: TraceSource) -> Iterator[TraceContentLocation]:
        """Yield locations; this is a descriptive alias for walk."""

        return self.walk(source)

    def rewrite(
        self,
        source: TraceSource,
        redactor: TextTransform,
    ) -> Iterator[str]:
        """Yield JSONL while changing only configured string values.

        The callback must return a string for every string it receives. If it
        raises or returns another type, a value-free TraceJSONLTransformError
        identifies the line and path. Object insertion order, scalar types,
        identifiers, and all unconfigured values are retained.
        """

        if not callable(redactor):
            raise TypeError("redactor must be callable")

        for line_number, line in enumerate(_iter_source_lines(source), start=1):
            payload, ending = _split_line_ending(line)
            if not payload.strip():
                yield line
                continue

            record = _parse_record(payload, line_number)
            seen_paths: set[JsonPath] = set()
            for location in _locations_for_record(
                record,
                line_number=line_number,
                content_paths=self._content_paths,
            ):
                if location.path in seen_paths:
                    continue
                seen_paths.add(location.path)
                replacement = _transform_value(redactor, location)
                _set_path(record, location.path, replacement)

            yield _dump_record(record) + ending

    def __call__(self, source: TraceSource) -> Iterator[TraceContentLocation]:
        """Yield locations when the walker is called as a function."""

        return self.walk(source)


def walk_trace_content(
    source: TraceSource,
    content_paths: Iterable[ContentPath] | ContentPath | None = None,
    *,
    paths: Iterable[ContentPath] | ContentPath | None = None,
) -> Iterator[TraceContentLocation]:
    """Yield configured redactable string locations from JSONL trace records."""

    return TraceContentWalker(content_paths, paths=paths).walk(source)


def rewrite_trace_jsonl(
    source: TraceSource,
    redactor: TextTransform,
    content_paths: Iterable[ContentPath] | ContentPath | None = None,
    *,
    paths: Iterable[ContentPath] | ContentPath | None = None,
) -> Iterator[str]:
    """Yield JSONL records after applying redactor to configured strings."""

    return TraceContentWalker(content_paths, paths=paths).rewrite(source, redactor)


def write_trace_jsonl(
    source: TraceSource,
    output: str | os.PathLike[str] | TextIO,
    redactor: TextTransform,
    content_paths: Iterable[ContentPath] | ContentPath | None = None,
    *,
    paths: Iterable[ContentPath] | ContentPath | None = None,
) -> int:
    """Stream a rewritten JSONL source into a path or text output.

    Returns:
        The number of physical lines written, including blank lines.

    Raises:
        ValueError: If path-based input and output resolve to the same file.
    """

    source_path = _path_source(source)
    output_path: Path | None = None
    if isinstance(output, (str, os.PathLike)):
        try:
            output_path = Path(output)
        except Exception:  # noqa: BLE001 - path errors may contain PHI
            raise TraceJSONLIOError("trace destination is invalid") from None
    if source_path is not None and isinstance(output, (str, os.PathLike)):
        if output_path is not None and _same_path(source_path, output_path):
            raise ValueError("input and output paths must be different")

    lines = rewrite_trace_jsonl(
        source,
        redactor,
        content_paths,
        paths=paths,
    )
    if output_path is not None:
        try:
            with output_path.open("w", encoding="utf-8", newline="") as destination:
                return _write_rewritten_lines(lines, destination)
        except (TraceJSONLIOError, TraceJSONLLineError):
            raise
        except Exception:  # noqa: BLE001 - I/O errors may contain PHI paths
            raise TraceJSONLIOError("trace destination could not be written") from None
    return _write_rewritten_lines(lines, cast(TextIO, output))


def _write_rewritten_lines(lines: Iterable[str], destination: TextIO) -> int:
    line_count = 0
    try:
        for line in lines:
            destination.write(line)
            line_count += 1
    except (TraceJSONLIOError, TraceJSONLLineError):
        raise
    except Exception:  # noqa: BLE001 - stream errors may contain PHI
        raise TraceJSONLIOError("trace destination could not be written") from None
    return line_count


# The aliases keep common descriptions of the same streaming surface
# discoverable without introducing separate implementations.
walk_jsonl_content = walk_trace_content
iter_trace_content = walk_trace_content
iter_jsonl_content = walk_trace_content
rewrite_jsonl_content = rewrite_trace_jsonl
redact_trace_jsonl = rewrite_trace_jsonl


def _normalise_content_paths(
    content_paths: Iterable[ContentPath] | ContentPath | None,
) -> tuple[JsonPath, ...]:
    if content_paths is None:
        raw_paths: Iterable[ContentPath] = DEFAULT_CONTENT_PATHS
    elif isinstance(content_paths, str):
        raw_paths = (content_paths,)
    elif isinstance(content_paths, Sequence) and any(
        isinstance(segment, int) and not isinstance(segment, bool)
        for segment in content_paths
    ):
        raw_paths = (cast(Sequence[JsonPathSegment], content_paths),)
    else:
        raw_paths = cast(Iterable[ContentPath], content_paths)

    compiled: list[JsonPath] = []
    seen: set[JsonPath] = set()
    for path in raw_paths:
        compiled_path = _compile_path(path)
        if compiled_path not in seen:
            compiled.append(compiled_path)
            seen.add(compiled_path)
    return tuple(compiled)


def _compile_path(path: ContentPath) -> JsonPath:
    if isinstance(path, str):
        text = path.strip()
        if text.startswith("$"):
            text = text[1:]
        text = re.sub(r"\[(?:\*|)\]", ".*", text)
        text = re.sub(r"\[(\d+)\]", r".\1", text)
        text = text.lstrip(".")
        if not text:
            raise ValueError("content paths must not be empty")
        parts: tuple[JsonPathSegment, ...] = tuple(text.split("."))
    else:
        parts = tuple(path)
        if not parts:
            raise ValueError("content paths must not be empty")

    for part in parts:
        if not isinstance(part, (str, int)) or isinstance(part, bool):
            raise TypeError("content path segments must be strings or integers")
        if isinstance(part, str) and not part:
            raise ValueError("content path segments must not be empty")
    return parts


def _locations_for_record(
    record: Mapping[str, Any],
    *,
    line_number: int,
    content_paths: Sequence[JsonPath],
) -> Iterator[TraceContentLocation]:
    seen_paths: set[JsonPath] = set()
    for content_path in content_paths:
        for path, value in _walk_path(record, content_path):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            yield TraceContentLocation(
                line_number=line_number,
                path=path,
                value=value,
            )


def _walk_path(
    node: Any,
    pattern: JsonPath,
    path: JsonPath = (),
) -> Iterator[tuple[JsonPath, str]]:
    if not pattern:
        if isinstance(node, str):
            yield path, node
        return

    segment, *tail_parts = pattern
    tail = tuple(tail_parts)
    if segment == "**":
        yield from _walk_path(node, tail, path)
        if isinstance(node, Mapping):
            for key, child in node.items():
                yield from _walk_path(child, pattern, path + (str(key),))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                yield from _walk_path(child, pattern, path + (index,))
        return

    if isinstance(node, Mapping):
        if segment == "*":
            for key, child in node.items():
                yield from _walk_path(child, tail, path + (str(key),))
        elif isinstance(segment, str) and segment in node:
            yield from _walk_path(node[segment], tail, path + (segment,))
        return

    if isinstance(node, list):
        if segment == "*":
            for index, child in enumerate(node):
                yield from _walk_path(child, tail, path + (index,))
        else:
            selected_index = _list_index(segment)
            if selected_index is not None and 0 <= selected_index < len(node):
                yield from _walk_path(
                    node[selected_index], tail, path + (selected_index,)
                )


def _list_index(segment: JsonPathSegment) -> int | None:
    if isinstance(segment, int):
        return segment
    if isinstance(segment, str) and segment.isdecimal():
        return int(segment)
    return None


def _set_path(record: dict[str, Any], path: JsonPath, value: str) -> None:
    node: Any = record
    for segment in path[:-1]:
        if isinstance(node, Mapping):
            node = node[segment]
        else:
            node = node[int(segment)]
    final_segment = path[-1]
    if isinstance(node, dict):
        node[final_segment] = value
    else:
        node[int(final_segment)] = value


def _transform_value(redactor: TextTransform, location: TraceContentLocation) -> str:
    try:
        replacement = redactor(location.value)
    except Exception:
        raise TraceJSONLTransformError(
            location.line_number,
            location.path,
            "content transform failed",
        ) from None
    if not isinstance(replacement, str):
        raise TraceJSONLTransformError(
            location.line_number,
            location.path,
            "content transform must return a string",
        )
    return replacement


def _parse_record(payload: str, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise TraceJSONLLineError(
            line_number,
            f"malformed JSON ({exc.msg})",
        ) from None
    except _DuplicateKeyError:
        raise TraceJSONLLineError(
            line_number,
            "malformed JSON (duplicate object key)",
        ) from None
    except _NonStandardNumberError:
        raise TraceJSONLLineError(
            line_number,
            "malformed JSON (non-standard numeric value)",
        ) from None
    except (TypeError, ValueError, RecursionError):
        raise TraceJSONLLineError(line_number, "malformed JSON") from None

    if not isinstance(record, dict):
        raise TraceJSONLLineError(
            line_number,
            "malformed trace record (expected a JSON object)",
        )
    return record


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise _DuplicateKeyError
        record[key] = value
    return record


def _reject_constant(_: str) -> None:
    raise _NonStandardNumberError


def _dump_record(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _iter_source_lines(source: TraceSource) -> Iterator[str]:
    if isinstance(source, os.PathLike):
        try:
            source_path = Path(source)
            with source_path.open("r", encoding="utf-8", newline="") as handle:
                yield from _iter_text_lines(handle)
        except TraceJSONLIOError:
            raise
        except Exception:  # noqa: BLE001 - path errors may contain PHI
            raise TraceJSONLIOError("trace source could not be read") from None
        return

    if isinstance(source, str):
        source_path = Path(source)
        if "\n" not in source and "\r" not in source:
            try:
                path_exists = source_path.exists()
            except OSError:
                path_exists = False
            if path_exists:
                try:
                    with source_path.open("r", encoding="utf-8", newline="") as handle:
                        yield from _iter_text_lines(handle)
                except TraceJSONLIOError:
                    raise
                except Exception:  # noqa: BLE001 - path errors may contain PHI
                    raise TraceJSONLIOError("trace source could not be read") from None
                return
        yield from _iter_text_lines(io.StringIO(source, newline=""))
        return

    if isinstance(source, Iterable):
        yield from _iter_text_lines(source)
        return

    raise TypeError("source must be a path, JSONL text, text stream, or line iterable")


def _path_source(source: TraceSource) -> Path | None:
    if isinstance(source, os.PathLike):
        try:
            return Path(source)
        except Exception:  # noqa: BLE001 - path errors may contain PHI
            raise TraceJSONLIOError("trace source is invalid") from None
    if isinstance(source, str) and "\n" not in source and "\r" not in source:
        source_path = Path(source)
        try:
            if source_path.exists():
                return source_path
        except OSError:
            return None
    return None


def _iter_text_lines(lines: Iterable[Any]) -> Iterator[str]:
    try:
        iterator = iter(lines)
    except Exception:  # noqa: BLE001 - stream errors may contain PHI
        raise TraceJSONLIOError("trace source could not be read") from None
    while True:
        try:
            line = next(iterator)
        except StopIteration:
            return
        except Exception:  # noqa: BLE001 - stream errors may contain PHI
            raise TraceJSONLIOError("trace source could not be read") from None
        yield _require_text_line(line)


def _same_path(first: Path, second: Path) -> bool:
    try:
        if first.samefile(second):
            return True
    except OSError:
        pass
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return first.absolute() == second.absolute()


def _require_text_line(line: Any) -> str:
    if not isinstance(line, str):
        raise TypeError("source lines must be text")
    return line


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1:]
    return line, ""


def _format_json_path(path: JsonPath) -> str:
    result = "$"
    for segment in path:
        if isinstance(segment, int):
            result += f"[{segment}]"
        elif segment in _SAFE_PATH_KEYS or _HASHED_PATH_KEY.fullmatch(segment):
            result += f".{segment}"
        else:
            result += f".{_safe_path_key(segment)}"
    return result


def _sanitize_error_path(path: JsonPath) -> JsonPath:
    return tuple(
        segment
        if isinstance(segment, int) or segment in _SAFE_PATH_KEYS
        else _safe_path_key(segment)
        for segment in path
    )


def _safe_path_key(value: str) -> str:
    if _HASHED_PATH_KEY.fullmatch(value):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"key_sha256_{digest}"
