"""Scrub a completed local trace in place.

The session-end hook is deliberately small and host-neutral.  It reads only
the path supplied by the caller, applies deterministic local redaction rules,
validates the rewritten JSON structure, and atomically replaces the source
file.  It does not load a model, inspect a trace store, or make a network call.

The module is executable as a quiet command-line hook::

    python -m openmed.guard.session_hook /path/to/completed-trace.json

Use ``--json`` when a host needs a machine-readable, value-free summary.  The
default success path emits nothing; callers can use the exit status alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})
_PLACEHOLDER_PATTERN = re.compile(r"^\[REDACTED:[A-Z_]+\]$")

_EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+(?![\w.-])"
)
_SSN_PATTERN = re.compile(r"(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)")
_CARD_PATTERN = re.compile(r"(?<!\w)(?:\d[ -]?){13,19}(?!\w)")
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d[\d().\-\s]{8,}\d)(?!\w)"
    r"|(?<!\w)\d{3}[.\-\s]\d{4}(?!\w)"
)
_IPV4_PATTERN = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_BEARER_PATTERN = re.compile(r"(?i)(?<!\w)Bearer\s+[A-Za-z0-9._~+/=-]{8,}(?![\w.-])")
_JWT_PATTERN = re.compile(
    r"(?<![\w.-])[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"
    r"(?![\w.-])"
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?P<prefix>\b(?:api[_ -]?key|access[_ -]?token|auth(?:orization)?|"
    r"secret|password|token)\b\s*[:=]\s*)"
    r"""(?P<quote>["']?)(?P<value>[^\s,;'" ]{8,})(?P=quote)"""
)
_IDENTIFIER_PATTERN = re.compile(
    r"(?i)\b(?:mrn|medical[ _-]?record(?:[ _-]?number)?|"
    r"patient[ _-]?id|member[ _-]?id)\b\s*[:#=]?\s*"
    r"[A-Za-z0-9][A-Za-z0-9./_-]{2,}"
)
_LABELED_DATE_PATTERN = re.compile(
    r"(?i)(?P<prefix>\b(?:dob|date[ _-]?of[ _-]?birth|birth[ _-]?date)\b"
    r"\s*[:=]?\s*)(?P<value>\d{1,4}[./-]\d{1,2}[./-]\d{1,4})"
)
_NAME_PHRASE_PATTERN = re.compile(
    r"(?P<prefix>(?i:\b(?:patient|subject|client|member|user)(?:\s+name)?"
    r"\s*(?:is\s+|[:=]\s*)?))"
    r"(?P<value>[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,3})"
)

_SENSITIVE_KEY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "SECRET",
        (
            "api_key",
            "access_token",
            "authorization",
            "cookie",
            "credential",
            "password",
            "secret",
            "token",
        ),
    ),
    (
        "EMAIL",
        ("email", "mail_address"),
    ),
    (
        "PHONE",
        ("phone", "mobile", "telephone", "fax"),
    ),
    (
        "DATE",
        ("birth_date", "date_of_birth", "dob"),
    ),
    (
        "ADDRESS",
        ("address", "street", "postal_code", "zip_code", "location"),
    ),
    (
        "ID",
        (
            "mrn",
            "medical_record_number",
            "patient_id",
            "member_id",
            "subject_id",
        ),
    ),
    (
        "NAME",
        (
            "name",
            "patient",
            "subject",
            "client",
            "member",
            "person",
            "user",
            "username",
        ),
    ),
    (
        "TEXT",
        (
            "body",
            "completion",
            "content",
            "description",
            "detail",
            "error",
            "exception",
            "input",
            "message",
            "narrative",
            "note",
            "output",
            "prompt",
            "request",
            "response",
            "stack",
            "stacktrace",
            "text",
            "trace",
            "traceback",
        ),
    ),
)


class _DuplicateKeyError(ValueError):
    """Raised when a trace object contains duplicate JSON keys."""


class SessionTraceError(Exception):
    """A value-free, stable failure from the session-end hook."""

    _MESSAGES = {
        "invalid_path": "the supplied trace path is invalid",
        "trace_not_found": "the supplied trace could not be read",
        "invalid_trace": "the supplied trace is not valid structured data",
        "unsupported_trace": "the supplied trace format is unsupported",
        "validation_failed": "the scrubbed trace failed structural validation",
        "concurrent_change": "the trace changed while it was being scrubbed",
        "write_failed": "the scrubbed trace could not be committed",
    }

    def __init__(self, code: str) -> None:
        if code not in self._MESSAGES:
            code = "write_failed"
        self.code = code
        super().__init__(self._MESSAGES[code])


@dataclass(frozen=True)
class SessionScrubResult:
    """Value-free evidence about one completed trace scrub."""

    format: str
    redaction_count: int
    changed: bool
    input_sha256: str
    output_sha256: str
    input_bytes: int
    output_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report containing no source values or paths."""
        return {
            "format": self.format,
            "redaction_count": self.redaction_count,
            "changed": self.changed,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
        }


def scrub_trace(path: str | os.PathLike[str]) -> SessionScrubResult:
    """Scrub one trace and expose only a value-free failure if it cannot finish."""
    try:
        return _scrub_trace(path)
    except SessionTraceError as exc:
        code = exc.code
    except Exception:
        code = "write_failed"
    raise SessionTraceError(code)


def _scrub_trace(path: str | os.PathLike[str]) -> SessionScrubResult:
    """Scrub one explicit completed JSON or JSONL trace in place.

    Args:
        path: The only trace path to read and, when needed, replace.  JSONL
            paths use one JSON value per non-empty line.

    Returns:
        A deterministic summary with counts and content hashes only.

    Raises:
        SessionTraceError: If the input cannot be safely read, validated, or
            atomically replaced.  The exception never includes a path or raw
            trace content.
    """

    trace_path = _coerce_trace_path(path)
    try:
        if trace_path.is_symlink() or not trace_path.is_file():
            raise SessionTraceError("trace_not_found")
        original_stat = trace_path.stat()
        original_bytes = trace_path.read_bytes()
    except SessionTraceError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SessionTraceError("trace_not_found") from exc

    try:
        text = original_bytes.decode("utf-8")
        payload, format_name = _parse_trace(text, trace_path.suffix.lower())
        scrubbed, redaction_count = _scrub_node(payload)
        _validate_scrubbed_payload(payload, scrubbed)
    except SessionTraceError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise SessionTraceError("invalid_trace") from exc

    if redaction_count == 0:
        return _result(
            format_name,
            redaction_count,
            original_bytes,
            original_bytes,
            changed=False,
        )

    try:
        scrubbed_text = _serialize_trace(scrubbed, format_name)
        scrubbed_bytes = scrubbed_text.encode("utf-8")
        final_payload, final_format = _parse_trace(
            scrubbed_text, _suffix_for_format(format_name)
        )
        _validate_scrubbed_payload(payload, final_payload)
        if final_format != format_name:
            raise SessionTraceError("validation_failed")
        _, residual_redactions = _scrub_node(final_payload)
        if residual_redactions:
            raise SessionTraceError("validation_failed")
    except SessionTraceError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise SessionTraceError("validation_failed") from exc

    try:
        current_stat = trace_path.stat()
    except OSError as exc:
        raise SessionTraceError("trace_not_found") from exc
    if not _same_file_snapshot(original_stat, current_stat):
        raise SessionTraceError("concurrent_change")

    try:
        _atomic_replace(trace_path, scrubbed_bytes, original_stat)
    except SessionTraceError:
        raise
    except OSError as exc:
        raise SessionTraceError("write_failed") from exc

    return _result(
        format_name,
        redaction_count,
        original_bytes,
        scrubbed_bytes,
        changed=True,
    )


def scrub_session_trace(path: str | os.PathLike[str]) -> SessionScrubResult:
    """Compatibility alias for :func:`scrub_trace`."""
    return scrub_trace(path)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone session-end hook parser."""
    parser = argparse.ArgumentParser(
        prog="openmed-session-end",
        description=(
            "Scrub one completed local JSON or JSONL trace in place. "
            "Success is silent unless --json is requested."
        ),
    )
    parser.add_argument(
        "trace_path",
        nargs="?",
        type=Path,
        help="Explicit completed trace path.",
    )
    parser.add_argument(
        "--trace",
        "--path",
        dest="trace_option",
        type=Path,
        metavar="PATH",
        help="Explicit completed trace path (alternative to the positional path).",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print a value-free result summary instead of remaining quiet.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the quiet command-line session-end hook."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.trace_path is None) == (args.trace_option is None):
        parser.error("provide exactly one explicit completed trace path")

    trace_path = args.trace_path or args.trace_option
    try:
        result = scrub_trace(trace_path)
    except SessionTraceError as exc:
        if args.json_output:
            print(json.dumps({"ok": False, "error": {"code": exc.code}}))
        else:
            print(f"session trace scrub failed: {exc.code}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps({"ok": True, "data": result.to_dict()}, sort_keys=True))
    return 0


def _coerce_trace_path(path: str | os.PathLike[str]) -> Path:
    try:
        if isinstance(path, Path):
            trace_path = path
        else:
            trace_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise SessionTraceError("invalid_path") from exc
    if not str(trace_path):
        raise SessionTraceError("invalid_path")
    return trace_path


def _parse_trace(text: str, suffix: str) -> tuple[Any, str]:
    if suffix in _JSONL_SUFFIXES:
        return _parse_jsonl(text), "jsonl"
    try:
        return _parse_json(text), "json"
    except _DuplicateKeyError as exc:
        raise SessionTraceError("invalid_trace") from exc
    except (TypeError, ValueError, json.JSONDecodeError) as json_exc:
        if "\n" not in text and "\r" not in text:
            raise SessionTraceError("invalid_trace") from json_exc
        try:
            return _parse_jsonl(text), "jsonl"
        except (TypeError, ValueError, json.JSONDecodeError) as jsonl_exc:
            raise SessionTraceError("unsupported_trace") from jsonl_exc


def _parse_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_object_pairs_without_duplicates,
        parse_constant=_reject_non_finite,
    )


def _parse_jsonl(text: str) -> list[Any]:
    values: list[Any] = []
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty trace")
    for line in lines:
        if not line.strip():
            raise ValueError("blank JSONL record")
        values.append(_parse_json(line))
    return values


def _object_pairs_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_non_finite(value: str) -> Any:
    raise ValueError(f"non-finite JSON value: {value}")


def _scrub_node(node: Any, context: str | None = None) -> tuple[Any, int]:
    if isinstance(node, str):
        return _scrub_string(node, context)
    if isinstance(node, list):
        scrubbed_items: list[Any] = []
        redactions = 0
        for item in node:
            scrubbed, count = _scrub_node(item, context)
            scrubbed_items.append(scrubbed)
            redactions += count
        return scrubbed_items, redactions
    if isinstance(node, dict):
        attribute_context = context
        for metadata_key in ("key", "field", "attribute", "attribute_name", "label"):
            metadata_value = node.get(metadata_key)
            if isinstance(metadata_value, str):
                attribute_context = (
                    _category_for_key(metadata_value) or attribute_context
                )
                if attribute_context:
                    break

        scrubbed_mapping: dict[str, Any] = {}
        redactions = 0
        for key, value in node.items():
            child_context = attribute_context
            if key in {"key", "field", "attribute", "attribute_name", "label"}:
                child_context = None
            else:
                child_context = _category_for_key(key) or child_context
            scrubbed_value, count = _scrub_node(value, child_context)
            scrubbed_mapping[key] = scrubbed_value
            redactions += count
        return scrubbed_mapping, redactions
    return node, 0


def _scrub_string(value: str, context: str | None) -> tuple[str, int]:
    if not value or _PLACEHOLDER_PATTERN.fullmatch(value):
        return value, 0
    if context:
        return _placeholder(value, context)

    scrubbed = value
    redactions = 0
    substitutions: tuple[
        tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...
    ] = (
        (_SECRET_ASSIGNMENT_PATTERN, _replace_secret_assignment),
        (_BEARER_PATTERN, lambda _match: "[REDACTED:SECRET]"),
        (_JWT_PATTERN, lambda _match: "[REDACTED:SECRET]"),
        (_IDENTIFIER_PATTERN, lambda _match: "[REDACTED:ID]"),
        (_EMAIL_PATTERN, lambda _match: "[REDACTED:EMAIL]"),
        (_SSN_PATTERN, lambda _match: "[REDACTED:ID]"),
        (_CARD_PATTERN, lambda _match: "[REDACTED:CARD]"),
        (_PHONE_PATTERN, lambda _match: "[REDACTED:PHONE]"),
        (_IPV4_PATTERN, _replace_ipv4),
        (_LABELED_DATE_PATTERN, _replace_labeled_date),
        (_NAME_PHRASE_PATTERN, _replace_name_phrase),
    )
    for pattern, replacement in substitutions:
        scrubbed, count = _substitute(scrubbed, pattern, replacement)
        redactions += count
    return scrubbed, redactions


def _placeholder(value: str, category: str) -> tuple[str, int]:
    if not value:
        return value, 0
    return f"[REDACTED:{category}]", 1


def _category_for_key(key: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    if not normalized:
        return None
    tokens = set(normalized.split("_"))
    for category, candidates in _SENSITIVE_KEY_GROUPS:
        for candidate in candidates:
            if normalized == candidate or candidate in tokens:
                return category
    if (
        normalized.endswith("_type")
        or normalized.endswith("_name")
        and normalized != "name"
    ):
        return None
    return None


def _substitute(
    text: str,
    pattern: re.Pattern[str],
    replacement: Callable[[re.Match[str]], str],
) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        value = replacement(match)
        if value != match.group(0):
            count += 1
        return value

    return pattern.sub(replace, text), count


def _replace_secret_assignment(match: re.Match[str]) -> str:
    quote = match.group("quote")
    placeholder = "[REDACTED:SECRET]"
    if quote:
        return f"{match.group('prefix')}{quote}{placeholder}{quote}"
    return f"{match.group('prefix')}{placeholder}"


def _replace_ipv4(match: re.Match[str]) -> str:
    octets = match.group(0).split(".")
    if all(int(octet) <= 255 for octet in octets):
        return "[REDACTED:IP]"
    return match.group(0)


def _replace_labeled_date(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}[REDACTED:DATE]"


def _replace_name_phrase(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}[REDACTED:NAME]"


def _validate_scrubbed_payload(original: Any, scrubbed: Any) -> None:
    if _structure_signature(original) != _structure_signature(scrubbed):
        raise SessionTraceError("validation_failed")


def _structure_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return (
            "object",
            tuple(
                (key, _structure_signature(child))
                for key, child in sorted(value.items(), key=lambda item: item[0])
            ),
        )
    if isinstance(value, list):
        return ("array", tuple(_structure_signature(item) for item in value))
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean",)
    if isinstance(value, int):
        return ("integer",)
    if isinstance(value, float):
        return ("number",)
    if isinstance(value, str):
        return ("string",)
    raise TypeError("unsupported JSON value")


def _serialize_trace(payload: Any, format_name: str) -> str:
    if format_name == "jsonl":
        return "".join(
            json.dumps(
                item,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in payload
        )
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _suffix_for_format(format_name: str) -> str:
    return ".jsonl" if format_name == "jsonl" else ".json"


def _same_file_snapshot(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )


def _atomic_replace(
    path: Path,
    payload: bytes,
    original_stat: os.stat_result,
) -> None:
    mode = stat.S_IMODE(original_stat.st_mode)
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=".openmed-session-scrub-",
            suffix=".tmp",
            dir=str(path.parent),
        )
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.chmod(temporary_name, mode)
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError:
            return
    finally:
        os.close(fd)


def _result(
    format_name: str,
    redaction_count: int,
    original: bytes,
    output: bytes,
    *,
    changed: bool,
) -> SessionScrubResult:
    return SessionScrubResult(
        format=format_name,
        redaction_count=redaction_count,
        changed=changed,
        input_sha256=hashlib.sha256(original).hexdigest(),
        output_sha256=hashlib.sha256(output).hexdigest(),
        input_bytes=len(original),
        output_bytes=len(output),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SessionScrubResult",
    "SessionTraceError",
    "build_parser",
    "main",
    "scrub_session_trace",
    "scrub_trace",
]
