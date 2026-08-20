"""Deterministic structural checks for redacted model traces.

Trace redaction is allowed to change declared content fields, but it must not
change the structure needed to replay or train from a trace.  This module
compares JSON-like values without logging or retaining source values.  It
checks mapping keys, sequence order, scalar types, identifiers, timestamps,
call references, and training labels while treating configured content paths as
value-flexible fields.

The verifier is deliberately a pure local operation.  It does not load a
model, inspect files, emit logs, or make network requests.  Reports and
exceptions contain paths, safe type names, and issue codes only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

TracePathPart: TypeAlias = str | int
TracePath: TypeAlias = tuple[TracePathPart, ...]
ContentPath: TypeAlias = str | Sequence[TracePathPart]
ContentPathSpec: TypeAlias = ContentPath | Iterable[ContentPath] | None

FIDELITY_SCHEMA_VERSION = 1
DEFAULT_CONTENT_FIELDS: tuple[str, ...] = ("**.content",)

_IDENTIFIER_FIELDS = frozenset(
    {
        "id",
        "trace_id",
        "span_id",
        "run_id",
        "request_id",
        "message_id",
        "example_id",
        "record_id",
        "parent_id",
        "source_id",
    }
)
_CALL_REFERENCE_FIELDS = frozenset(
    {
        "call_id",
        "tool_call_id",
        "tool_use_id",
        "parent_call_id",
        "parent_tool_call_id",
        "function_call_id",
    }
)
_TIMESTAMP_FIELDS = frozenset(
    {
        "timestamp",
        "timestamps",
        "created_at",
        "updated_at",
        "started_at",
        "ended_at",
        "observed_at",
        "completed_at",
        "event_time",
        "time",
        "ts",
    }
)
_LABEL_FIELDS = frozenset(
    {
        "label",
        "labels",
        "training_label",
        "training_labels",
        "reward",
        "preference",
        "target_label",
    }
)
_MESSAGE_SEQUENCE_NAMES = frozenset({"messages", "turns"})
_SAFE_PATH_KEYS = frozenset(
    {
        "arguments",
        "assistant",
        "calls",
        "content",
        "function",
        "function_call",
        "function_calls",
        "input",
        "items",
        "message",
        "messages",
        "name",
        "output",
        "parts",
        "role",
        "system",
        "text",
        "tool",
        "tool_call",
        "tool_calls",
        "turns",
        "type",
        "user",
        "value",
    }
    | _IDENTIFIER_FIELDS
    | _CALL_REFERENCE_FIELDS
    | _TIMESTAMP_FIELDS
    | _LABEL_FIELDS
)
_HASHED_PATH_KEY = re.compile(r"^key_sha256_[0-9a-f]{12}$")
_ISSUE_CODES = frozenset(
    {
        "call_linkage",
        "container_type",
        "identifier",
        "message_order",
        "scalar_type",
        "scalar_value",
        "sequence_length",
        "structure",
        "timestamp",
        "training_label",
    }
)
_TYPE_NAMES = frozenset(
    {
        "array",
        "boolean",
        "integer",
        "list",
        "missing",
        "null",
        "number",
        "object",
        "other",
        "string",
        "tuple",
    }
)


class TraceFidelityError(ValueError):
    """Raised by :func:`assert_trace_fidelity` when a check fails."""

    def __init__(self, report: "TraceFidelityReport") -> None:
        self.report = report
        super().__init__(report.summary())


@dataclass(frozen=True, slots=True, order=True)
class TraceFidelityIssue:
    """One value-free structural discrepancy.

    ``path`` is a JSONPath-like location.  ``expected_type`` and
    ``actual_type`` are coarse JSON-safe type names rather than representations
    of the values at that location.
    """

    path: str
    code: str
    expected_type: str | None = None
    actual_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _sanitize_report_path(self.path))
        object.__setattr__(
            self,
            "code",
            self.code if self.code in _ISSUE_CODES else "structure",
        )
        object.__setattr__(
            self,
            "expected_type",
            _sanitize_type_name(self.expected_type),
        )
        object.__setattr__(
            self,
            "actual_type",
            _sanitize_type_name(self.actual_type),
        )

    @property
    def kind(self) -> str:
        """Return ``code`` under the common issue-kind spelling."""

        return self.code

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable, value-free issue."""

        return {
            "path": self.path,
            "code": self.code,
            "expected_type": self.expected_type,
            "actual_type": self.actual_type,
        }


@dataclass(frozen=True, slots=True)
class TraceFidelityReport:
    """Deterministic result of comparing two traces.

    Reports retain only issue paths, issue codes, type names, counts, and the
    configured path patterns.  They never retain input or output values.
    """

    passed: bool
    issues: tuple[TraceFidelityIssue, ...] = ()
    allowed_content_paths: tuple[str, ...] = ()
    checked_scalar_count: int = 0
    content_field_count: int = 0
    schema_version: int = FIDELITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            normalized = tuple(sorted(self.issues))
            if not all(isinstance(issue, TraceFidelityIssue) for issue in normalized):
                raise TypeError
            allowed_paths = tuple(
                _sanitize_report_path(path) for path in self.allowed_content_paths
            )
            checked_scalar_count = _report_count(self.checked_scalar_count)
            content_field_count = _report_count(self.content_field_count)
            if type(self.schema_version) is not int or self.schema_version < 1:
                raise ValueError
        except Exception:  # noqa: BLE001 - report inputs may contain PHI
            raise ValueError("trace fidelity report is invalid") from None
        object.__setattr__(self, "issues", normalized)
        object.__setattr__(self, "passed", not normalized)
        object.__setattr__(self, "allowed_content_paths", allowed_paths)
        object.__setattr__(self, "checked_scalar_count", checked_scalar_count)
        object.__setattr__(self, "content_field_count", content_field_count)

    def __bool__(self) -> bool:
        """Allow a report to be used directly in a boolean check."""

        return self.passed

    @property
    def valid(self) -> bool:
        """Return whether the output preserves trace fidelity."""

        return self.passed

    @property
    def ok(self) -> bool:
        """Return whether the output preserves trace fidelity."""

        return self.passed

    @property
    def failures(self) -> tuple[TraceFidelityIssue, ...]:
        """Return discrepancies under the common ``failures`` spelling."""

        return self.issues

    @property
    def errors(self) -> tuple[TraceFidelityIssue, ...]:
        """Return discrepancies under the common ``errors`` spelling."""

        return self.issues

    @property
    def failure_count(self) -> int:
        """Return the number of reported discrepancies."""

        return len(self.issues)

    @property
    def failing_paths(self) -> tuple[str, ...]:
        """Return unique failing paths in deterministic order."""

        return tuple(dict.fromkeys(issue.path for issue in self.issues))

    def has_code(self, code: str) -> bool:
        """Return whether at least one issue has ``code``."""

        return any(issue.code == code for issue in self.issues)

    @property
    def message_order_valid(self) -> bool:
        """Return whether message sequence order was preserved."""

        return not self.has_code("message_order")

    @property
    def call_linkage_valid(self) -> bool:
        """Return whether call identifiers and references were preserved."""

        return not self.has_code("call_linkage")

    @property
    def identifiers_valid(self) -> bool:
        """Return whether identifier fields were preserved."""

        return not self.has_code("identifier")

    @property
    def timestamps_valid(self) -> bool:
        """Return whether timestamp fields were preserved."""

        return not self.has_code("timestamp")

    @property
    def training_labels_valid(self) -> bool:
        """Return whether training-label fields were preserved."""

        return not self.has_code("training_label")

    @property
    def scalar_types_valid(self) -> bool:
        """Return whether scalar types were preserved."""

        return not self.has_code("scalar_type")

    @property
    def structure_valid(self) -> bool:
        """Return whether keys, containers, and sequence lengths were preserved."""

        return not any(
            issue.code in {"structure", "container_type", "sequence_length"}
            for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible report without values."""

        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "failure_count": self.failure_count,
            "checked_scalar_count": self.checked_scalar_count,
            "content_field_count": self.content_field_count,
            "allowed_content_paths": list(self.allowed_content_paths),
            "checks": {
                "message_order": self.message_order_valid,
                "call_linkage": self.call_linkage_valid,
                "identifiers": self.identifiers_valid,
                "timestamps": self.timestamps_valid,
                "training_labels": self.training_labels_valid,
                "scalar_types": self.scalar_types_valid,
                "structure": self.structure_valid,
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the report deterministically as JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )

    def summary(self) -> str:
        """Return a compact summary containing no trace values."""

        if self.passed:
            return "Trace fidelity passed."
        details = ", ".join(f"{issue.path} ({issue.code})" for issue in self.issues[:8])
        if len(self.issues) > 8:
            details += ", ..."
        return f"Trace fidelity failed at {self.failure_count} path(s): {details}."


@dataclass
class _ComparisonState:
    patterns: tuple[TracePath, ...]
    issues: list[TraceFidelityIssue] = field(default_factory=list)
    checked_scalar_count: int = 0
    content_field_count: int = 0
    active_pairs: set[tuple[int, int]] = field(default_factory=set)

    def issue(
        self,
        path: TracePath,
        code: str,
        *,
        expected_type: str | None = None,
        actual_type: str | None = None,
    ) -> None:
        self.issues.append(
            TraceFidelityIssue(
                path=_format_path(path),
                code=code,
                expected_type=expected_type,
                actual_type=actual_type,
            )
        )


@dataclass(frozen=True, slots=True, init=False)
class TraceFidelityVerifier:
    """Reusable verifier configuration for a declared content policy."""

    content_paths: tuple[TracePath, ...]

    def __init__(
        self,
        content_paths: ContentPathSpec = None,
        *,
        content_fields: ContentPathSpec = None,
        allowed_content_fields: ContentPathSpec = None,
        allowed_paths: ContentPathSpec = None,
    ) -> None:
        selected = _select_content_spec(
            content_paths,
            content_fields,
            allowed_content_fields,
            allowed_paths,
        )
        object.__setattr__(self, "content_paths", _normalize_content_paths(selected))

    @property
    def allowed_content_fields(self) -> tuple[str, ...]:
        """Return configured content patterns in safe display form."""

        return tuple(_format_path(path) for path in self.content_paths)

    def verify(self, original: Any, output: Any) -> TraceFidelityReport:
        """Compare ``original`` and ``output`` using this verifier policy."""

        return _compare_traces(original, output, self.content_paths)

    def compare(self, original: Any, output: Any) -> TraceFidelityReport:
        """Alias for :meth:`verify`."""

        return self.verify(original, output)

    def __call__(self, original: Any, output: Any) -> TraceFidelityReport:
        """Verify traces when the configured verifier is called directly."""

        return self.verify(original, output)


def verify_trace_fidelity(
    original: Any,
    output: Any,
    *,
    content_paths: ContentPathSpec = None,
    content_fields: ContentPathSpec = None,
    allowed_content_fields: ContentPathSpec = None,
    allowed_paths: ContentPathSpec = None,
) -> TraceFidelityReport:
    """Verify structural fidelity after a trace transformation.

    Args:
        original: The pre-redaction JSON-like trace.
        output: The post-redaction JSON-like trace.
        content_paths: Dotted paths or path tuples whose scalar content may
            change. ``*`` matches one mapping key or sequence item and ``**``
            matches zero or more levels. The default allows fields named
            ``content`` at any depth.
        content_fields: Alias for ``content_paths``.
        allowed_content_fields: Descriptive alias for ``content_paths``.
        allowed_paths: Compatibility alias for ``content_paths``.

    Returns:
        A deterministic report.  Use ``report.passed`` or ``bool(report)`` to
        gate acceptance; the report itself never contains trace values.

    Raises:
        TypeError: If more than one content-path spelling is supplied.
        ValueError: If a content path is malformed.
    """

    selected = _select_content_spec(
        content_paths,
        content_fields,
        allowed_content_fields,
        allowed_paths,
    )
    return _compare_traces(original, output, _normalize_content_paths(selected))


def assert_trace_fidelity(
    original: Any,
    output: Any,
    **kwargs: Any,
) -> TraceFidelityReport:
    """Return a passing report or raise a value-free fidelity error."""

    report = verify_trace_fidelity(original, output, **kwargs)
    if not report.passed:
        raise TraceFidelityError(report)
    return report


def _compare_traces(
    original: Any,
    output: Any,
    patterns: tuple[TracePath, ...],
) -> TraceFidelityReport:
    state = _ComparisonState(patterns=patterns)
    try:
        _compare_value(original, output, (), state, content_mode=False)
    except Exception:  # noqa: BLE001 - trace errors may contain sensitive values
        state.issue((), "structure")
    issues = tuple(sorted(set(state.issues)))
    return TraceFidelityReport(
        passed=not issues,
        issues=issues,
        allowed_content_paths=tuple(_format_path(path) for path in patterns),
        checked_scalar_count=state.checked_scalar_count,
        content_field_count=state.content_field_count,
    )


def _compare_value(
    expected: Any,
    actual: Any,
    path: TracePath,
    state: _ComparisonState,
    *,
    content_mode: bool,
) -> None:
    matched_content = _path_matches(path, state.patterns)
    if matched_content:
        state.content_field_count += 1
    content_mode = content_mode or matched_content

    expected_kind = _json_kind(expected)
    actual_kind = _json_kind(actual)
    if expected_kind != actual_kind:
        state.issue(
            path,
            _type_issue_code(path, expected_kind, actual_kind),
            expected_type=expected_kind,
            actual_type=actual_kind,
        )
        return

    if expected_kind == "object":
        _compare_mapping(expected, actual, path, state, content_mode=content_mode)
        return
    if expected_kind == "array":
        _compare_sequence(expected, actual, path, state, content_mode=content_mode)
        return

    state.checked_scalar_count += 1
    if type(expected) is not type(actual):
        state.issue(
            path,
            "scalar_type",
            expected_type=_scalar_type(expected),
            actual_type=_scalar_type(actual),
        )
        return

    # Semantic fields remain immutable even when nested inside a declared
    # content object.  This prevents a broad content path from masking a
    # changed identifier, label, timestamp, or call reference.
    if content_mode and _semantic_code(path) is None:
        return
    if not _safe_equal(expected, actual):
        state.issue(
            path,
            _semantic_code(path) or "scalar_value",
            expected_type=_scalar_type(expected),
            actual_type=_scalar_type(actual),
        )


def _compare_mapping(
    expected: Mapping[Any, Any],
    actual: Mapping[Any, Any],
    path: TracePath,
    state: _ComparisonState,
    *,
    content_mode: bool,
) -> None:
    pair = (id(expected), id(actual))
    if pair in state.active_pairs:
        state.issue(path, "structure")
        return
    state.active_pairs.add(pair)
    try:
        expected_keys = _stable_keys(expected)
        actual_keys = _stable_keys(actual)
        for key in expected_keys:
            if not _contains_key(actual, key):
                key_path = path + (_path_key(key),)
                state.issue(
                    key_path,
                    _semantic_code(key_path) or "structure",
                    expected_type=_json_kind(expected[key]),
                    actual_type="missing",
                )
        for key in actual_keys:
            if not _contains_key(expected, key):
                key_path = path + (_path_key(key),)
                state.issue(
                    key_path,
                    _semantic_code(key_path) or "structure",
                    expected_type="missing",
                    actual_type=_json_kind(actual[key]),
                )
        for key in expected_keys:
            if _contains_key(actual, key):
                _compare_value(
                    expected[key],
                    actual[key],
                    path + (_path_key(key),),
                    state,
                    content_mode=content_mode,
                )
    finally:
        state.active_pairs.remove(pair)


def _compare_sequence(
    expected: Sequence[Any],
    actual: Sequence[Any],
    path: TracePath,
    state: _ComparisonState,
    *,
    content_mode: bool,
) -> None:
    if type(expected) is not type(actual):
        state.issue(
            path,
            "container_type",
            expected_type=_container_type(expected),
            actual_type=_container_type(actual),
        )
        return
    if len(expected) != len(actual):
        state.issue(
            path,
            "sequence_length",
            expected_type="array",
            actual_type="array",
        )
        return

    if _is_message_sequence(path, expected, actual):
        expected_fingerprints = [
            _fingerprint(value, path + (index,), state.patterns)
            for index, value in enumerate(expected)
        ]
        actual_fingerprints = [
            _fingerprint(value, path + (index,), state.patterns)
            for index, value in enumerate(actual)
        ]
        if expected_fingerprints != actual_fingerprints and _same_multiset(
            expected_fingerprints, actual_fingerprints
        ):
            state.issue(
                path, "message_order", expected_type="array", actual_type="array"
            )
            return

    pair = (id(expected), id(actual))
    if pair in state.active_pairs:
        state.issue(path, "structure")
        return
    state.active_pairs.add(pair)
    try:
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare_value(
                expected_item,
                actual_item,
                path + (index,),
                state,
                content_mode=content_mode,
            )
    finally:
        state.active_pairs.remove(pair)


def _fingerprint(
    value: Any,
    path: TracePath,
    patterns: tuple[TracePath, ...],
    *,
    content_mode: bool = False,
    active_ids: frozenset[int] = frozenset(),
) -> Any:
    """Build an internal-only shape fingerprint for message order checks."""

    content_mode = content_mode or _path_matches(path, patterns)
    kind = _json_kind(value)
    if kind == "object":
        identity = id(value)
        if identity in active_ids:
            return ("cycle",)
        nested_active_ids = active_ids | {identity}
        return (
            "object",
            tuple(
                (
                    _safe_key(key),
                    _fingerprint(
                        value[key],
                        path + (_path_key(key),),
                        patterns,
                        content_mode=content_mode,
                        active_ids=nested_active_ids,
                    ),
                )
                for key in _stable_keys(value)
            ),
        )
    if kind == "array":
        identity = id(value)
        if identity in active_ids:
            return ("cycle",)
        nested_active_ids = active_ids | {identity}
        return (
            "array",
            tuple(
                _fingerprint(
                    item,
                    path + (index,),
                    patterns,
                    content_mode=content_mode,
                    active_ids=nested_active_ids,
                )
                for index, item in enumerate(value)
            ),
        )
    if content_mode and _semantic_code(path) is None:
        return ("content", _scalar_type(value))
    return ("scalar", _scalar_type(value), _safe_internal_value(value))


def _same_multiset(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return Counter(left) == Counter(right)


def _is_message_sequence(
    path: TracePath,
    expected: Sequence[Any],
    actual: Sequence[Any],
) -> bool:
    if path and isinstance(path[-1], str) and path[-1] in _MESSAGE_SEQUENCE_NAMES:
        return True
    combined = tuple(expected) + tuple(actual)
    return bool(combined) and all(
        isinstance(item, Mapping) and "role" in item for item in combined
    )


def _select_content_spec(*specs: ContentPathSpec) -> ContentPathSpec:
    selected = [spec for spec in specs if spec is not None]
    if len(selected) > 1:
        raise TypeError("pass only one content path configuration")
    return selected[0] if selected else DEFAULT_CONTENT_FIELDS


def _normalize_content_paths(spec: ContentPathSpec) -> tuple[TracePath, ...]:
    if spec is None:
        candidates: Iterable[ContentPath] = DEFAULT_CONTENT_FIELDS
    elif isinstance(spec, str):
        candidates = (spec,)
    elif isinstance(spec, (bytes, bytearray)):
        raise TypeError("content paths must be strings or path sequences")
    elif isinstance(spec, Sequence):
        try:
            items = tuple(spec)
        except Exception:  # noqa: BLE001 - configuration may contain PHI
            raise ValueError("content paths could not be consumed") from None
        if len(items) > 1 and _looks_like_one_path(items):
            candidates = (items,)
        else:
            candidates = items
    elif isinstance(spec, Iterable):
        candidates = _safe_content_candidates(spec)
    else:
        raise TypeError("content paths must be strings or path sequences")

    compiled: list[TracePath] = []
    seen: set[TracePath] = set()
    for candidate in candidates:
        try:
            path = _compile_content_path(candidate)
        except Exception:  # noqa: BLE001 - configuration may contain PHI
            raise ValueError("content path is invalid") from None
        if path not in seen:
            compiled.append(path)
            seen.add(path)
    return tuple(sorted(compiled, key=_path_sort_key))


def _safe_content_candidates(
    candidates: Iterable[ContentPath],
) -> Iterator[ContentPath]:
    """Consume path configuration without exposing iterator exceptions."""

    try:
        items = iter(candidates)
    except Exception:  # noqa: BLE001 - configuration may contain PHI
        raise TypeError("content paths could not be consumed") from None
    while True:
        try:
            yield next(items)
        except StopIteration:
            return
        except Exception:  # noqa: BLE001 - configuration may contain PHI
            raise ValueError("content paths could not be consumed") from None


def _looks_like_one_path(items: Sequence[Any]) -> bool:
    if not items or not all(isinstance(item, (str, int)) for item in items):
        return False
    return any(isinstance(item, int) or item in {"*", "**"} for item in items)


def _compile_content_path(candidate: ContentPath) -> TracePath:
    if isinstance(candidate, str):
        text = candidate.strip()
        if text.startswith("$"):
            text = text[1:]
        if text.startswith("/"):
            parts = text[1:].split("/")
            parts = [part.replace("~1", "/").replace("~0", "~") for part in parts]
        else:
            text = re.sub(r"\[(?:\*|)\]", ".*", text)
            text = re.sub(r"\[(\d+)\]", r".\1", text)
            text = text.lstrip(".")
            if not text:
                raise ValueError("content paths must not be empty")
            parts = text.split(".")
        candidate_parts: Sequence[Any] = [
            int(part) if part.isdecimal() else part for part in parts
        ]
    elif isinstance(candidate, Sequence) and not isinstance(
        candidate, (bytes, bytearray, str)
    ):
        candidate_parts = tuple(candidate)
    else:
        raise TypeError("content paths must be strings or path sequences")

    if not candidate_parts:
        raise ValueError("content paths must not be empty")
    result: list[TracePathPart] = []
    for part in candidate_parts:
        if isinstance(part, bool) or not isinstance(part, (str, int)):
            raise TypeError("content path segments must be strings or integers")
        if isinstance(part, int):
            if part < 0:
                raise ValueError("content path indexes must be non-negative")
            result.append(part)
            continue
        if not part:
            raise ValueError("content path segments must not be empty")
        result.append(part)
    return tuple(result)


def _path_matches(path: TracePath, patterns: Sequence[TracePath]) -> bool:
    return any(_match_pattern(pattern, path) for pattern in patterns)


def _match_pattern(pattern: TracePath, path: TracePath) -> bool:
    if not pattern:
        return not path
    if pattern[0] == "**":
        return _match_pattern(pattern[1:], path) or (
            bool(path) and _match_pattern(pattern, path[1:])
        )
    if not path:
        return False
    head = pattern[0]
    if head != "*" and head != path[0]:
        return False
    return _match_pattern(pattern[1:], path[1:])


def _semantic_code(path: TracePath) -> str | None:
    fields = [part.lower() for part in reversed(path) if isinstance(part, str)]
    field = fields[0] if fields else ""
    parent_fields = {part.lower() for part in path if isinstance(part, str)}
    if any(part in _CALL_REFERENCE_FIELDS for part in fields) or (
        field == "id"
        and parent_fields.intersection(
            {
                "call",
                "calls",
                "tool_call",
                "tool_calls",
                "function_call",
                "function_calls",
            }
        )
    ):
        return "call_linkage"
    if any(
        part in _IDENTIFIER_FIELDS or part.endswith("_id") or part.endswith("_ids")
        for part in fields
    ):
        return "identifier"
    if any(part in _TIMESTAMP_FIELDS or part.endswith("_at") for part in fields):
        return "timestamp"
    if any(part in _LABEL_FIELDS for part in fields):
        return "training_label"
    return None


def _type_issue_code(path: TracePath, expected: str, actual: str) -> str:
    semantic = _semantic_code(path)
    if semantic is not None:
        return semantic
    if expected in {"object", "array"} and actual in {"object", "array"}:
        return "structure"
    return "scalar_type"


def _json_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "array"
    return "other"


def _scalar_type(value: Any) -> str:
    return _json_kind(value)


def _container_type(value: Sequence[Any]) -> str:
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    return "array"


def _safe_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
        return bool(result) if isinstance(result, bool) else False
    except Exception:
        return False


def _safe_internal_value(value: Any) -> Any:
    """Return a comparison-only scalar representation, never exposed."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return repr(value)
    except Exception:
        return (type(value).__module__, type(value).__qualname__)


def _stable_keys(value: Mapping[Any, Any]) -> list[Any]:
    return sorted(value, key=lambda key: (type(key).__name__, _safe_key(key)))


def _contains_key(value: Mapping[Any, Any], key: Any) -> bool:
    try:
        return key in value
    except Exception:
        return False


def _safe_key(value: Any) -> str:
    if isinstance(value, str) and value in _SAFE_PATH_KEYS:
        return value
    if isinstance(value, str) and _HASHED_PATH_KEY.fullmatch(value):
        return value
    internal_value = _safe_internal_value(value)
    digest = hashlib.sha256(repr(internal_value).encode("utf-8")).hexdigest()[:12]
    return f"key_sha256_{digest}"


def _path_key(value: Any) -> str:
    return value if isinstance(value, str) else _safe_key(value)


def _path_sort_key(path: TracePath) -> tuple[tuple[int, str], ...]:
    return tuple(
        (0, str(part)) if isinstance(part, int) else (1, _safe_key(part))
        for part in path
    )


def _format_path(path: Sequence[TracePathPart]) -> str:
    if not path:
        return "$"
    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif part == "*":
            rendered += ".*"
        elif part == "**":
            rendered += ".**"
        elif part in _SAFE_PATH_KEYS or _HASHED_PATH_KEY.fullmatch(part):
            rendered += f".{part}"
        else:
            rendered += f".{_safe_key(part)}"
    return rendered


def _sanitize_report_path(value: object) -> str:
    try:
        if value == "$":
            return "$"
        if not isinstance(value, str):
            return "$"
        return _format_path(_compile_content_path(value))
    except Exception:  # noqa: BLE001 - direct report input may contain PHI
        return "$"


def _sanitize_type_name(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) and value in _TYPE_NAMES else "other"


def _report_count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("trace fidelity report counts must be non-negative integers")
    return value


# Common descriptive spellings all use the same deterministic implementation.
compare_trace_fidelity = verify_trace_fidelity
check_trace_fidelity = verify_trace_fidelity
validate_trace_fidelity = verify_trace_fidelity
verify_trace = verify_trace_fidelity
TraceFidelityResult = TraceFidelityReport


__all__ = [
    "ContentPath",
    "ContentPathSpec",
    "DEFAULT_CONTENT_FIELDS",
    "FIDELITY_SCHEMA_VERSION",
    "TraceFidelityError",
    "TraceFidelityIssue",
    "TraceFidelityReport",
    "TraceFidelityResult",
    "TraceFidelityVerifier",
    "TracePath",
    "TracePathPart",
    "assert_trace_fidelity",
    "check_trace_fidelity",
    "compare_trace_fidelity",
    "validate_trace_fidelity",
    "verify_trace",
    "verify_trace_fidelity",
]
