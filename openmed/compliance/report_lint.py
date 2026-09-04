"""Deterministic, value-free linting for aggregate privacy reports.

Report schemas describe the only fields and representations that may cross a
report boundary.  The linter checks an in-memory mapping without copying it
into its result.  Findings contain stable reason codes, safe field paths, and
coarse value shapes only; rejected values are never included in diagnostics,
exceptions, or serialized output.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

REPORT_LINT_SCHEMA_VERSION: Final = 1
DEFAULT_CODE_MAX_LENGTH: Final = 64
DEFAULT_ARRAY_MAX_ITEMS: Final = 128
MAX_SCHEMA_FIELD_NAME_LENGTH: Final = 128
MAX_SCHEMA_CODE_LENGTH: Final = 256

_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_FIELD_NAME_RE = re.compile(
    rf"^[A-Za-z][A-Za-z0-9_.:-]{{0,{MAX_SCHEMA_FIELD_NAME_LENGTH - 1}}}$"
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NUMBER_KINDS = frozenset({"number", "ratio"})
_KIND_ALIASES = {
    "digest": "hash",
    "float": "number",
    "integer": "count",
    "list": "array",
    "mapping": "object",
    "numeric": "number",
    "string": "text",
}
_SUPPORTED_KINDS = frozenset(
    {
        "array",
        "boolean",
        "code",
        "count",
        "forbidden",
        "hash",
        "number",
        "object",
        "ratio",
        "text",
    }
)

__all__ = [
    "DEFAULT_ARRAY_MAX_ITEMS",
    "DEFAULT_CODE_MAX_LENGTH",
    "MAX_SCHEMA_CODE_LENGTH",
    "MAX_SCHEMA_FIELD_NAME_LENGTH",
    "REPORT_LINT_SCHEMA_VERSION",
    "ReportFieldRule",
    "ReportFieldSpec",
    "ReportLintError",
    "ReportLintFinding",
    "ReportLintResult",
    "ReportSchema",
    "lint_report",
    "require_valid_report",
    "validate_report",
]


def _normalise_kind(kind: str) -> str:
    """Return a supported, canonical field kind without exposing input data."""

    if not isinstance(kind, str):
        raise TypeError("report field kind must be a string")
    normalised = _KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
    if normalised not in _SUPPORTED_KINDS:
        raise ValueError("unsupported report field kind")
    return normalised


def _is_finite_number(value: object) -> bool:
    """Return whether ``value`` is a finite, non-boolean number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_optional_bound(value: object, *, allow_zero: bool) -> int | None:
    """Validate a schema length bound without interpolating untrusted input."""

    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("report schema bounds must be integers")
    minimum = 0 if allow_zero else 1
    if not minimum <= value <= MAX_SCHEMA_CODE_LENGTH:
        raise ValueError("report schema bound is outside the supported range")
    return value


def _normalise_allowed_values(
    values: object,
    *,
    maximum_length: int,
) -> tuple[str, ...]:
    """Normalize finite safe code values used by an enum-like field."""

    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("allowed report values must be a sequence of codes")

    normalised: set[str] = set()
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > maximum_length
            or not _CODE_RE.fullmatch(value)
        ):
            raise ValueError("allowed report values must be bounded safe codes")
        normalised.add(value)
    return tuple(sorted(normalised))


def _normalise_field_mapping(
    fields: object,
) -> MappingProxyType:
    """Validate and deterministically order a mapping of field specifications."""

    if not isinstance(fields, Mapping):
        raise TypeError("report schema fields must be a mapping")

    normalised: dict[str, ReportFieldSpec] = {}
    for name, specification in fields.items():
        if not isinstance(name, str) or not _FIELD_NAME_RE.fullmatch(name):
            raise ValueError("report schema field names must be safe bounded names")
        normalised[name] = _coerce_field_spec(specification)
    return MappingProxyType(dict(sorted(normalised.items())))


def _coerce_field_spec(specification: object) -> ReportFieldSpec:
    """Convert supported schema shorthands into a typed field specification."""

    if isinstance(specification, ReportFieldSpec):
        return specification
    if isinstance(specification, str):
        return ReportFieldSpec(specification)
    if specification is bool:
        return ReportFieldSpec("boolean")
    if specification is int:
        return ReportFieldSpec("count")
    if specification is float:
        return ReportFieldSpec("number")
    if specification is str:
        return ReportFieldSpec("text")
    if not isinstance(specification, Mapping):
        raise TypeError("report schema fields must use typed specifications")

    kind = specification.get("kind", specification.get("type"))
    if not isinstance(kind, str):
        raise TypeError("report schema field specifications require a kind")
    values = dict(specification)
    values.pop("kind", None)
    values.pop("type", None)
    if "max_value" in values and "maximum" not in values:
        values["maximum"] = values.pop("max_value")
    supported_keys = {
        "allowed_values",
        "fields",
        "item",
        "maximum",
        "max_items",
        "max_length",
        "minimum",
        "required",
    }
    if set(values) - supported_keys:
        raise ValueError("report schema field specification contains an unknown option")
    return ReportFieldSpec(kind, **values)


@dataclass(frozen=True)
class ReportFieldSpec:
    """Typed, privacy-safe rule for one report field.

    Supported scalar kinds are ``count``, ``hash``, ``code``, ``boolean``, and
    ``number``/``ratio``.  ``text`` and ``forbidden`` are explicit rejection
    rules.  ``object`` and ``array`` may contain nested typed rules.  Strings
    are never accepted as free text: use a bounded ``code`` or a canonical
    ``hash`` instead.
    """

    kind: str
    required: bool = False
    max_length: int | None = None
    max_items: int | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    allowed_values: tuple[str, ...] = ()
    fields: Mapping[str, ReportFieldSpec] = field(default_factory=dict)
    item: ReportFieldSpec | None = None

    def __post_init__(self) -> None:
        kind = _normalise_kind(self.kind)
        if type(self.required) is not bool:
            raise TypeError("report field required must be a boolean")
        max_length = _validate_optional_bound(self.max_length, allow_zero=False)
        max_items = _validate_optional_bound(self.max_items, allow_zero=True)
        effective_length = max_length or DEFAULT_CODE_MAX_LENGTH
        if effective_length > MAX_SCHEMA_CODE_LENGTH:
            raise ValueError("report field length bound is too large")

        allowed_values = _normalise_allowed_values(
            self.allowed_values,
            maximum_length=effective_length,
        )
        for bound in (self.minimum, self.maximum):
            if bound is not None and not _is_finite_number(bound):
                raise TypeError("report field numeric bounds must be finite numbers")
        if (
            self.minimum is not None
            and self.maximum is not None
            and float(self.minimum) > float(self.maximum)
        ):
            raise ValueError("report field minimum cannot exceed maximum")
        if kind == "count":
            for bound in (self.minimum, self.maximum):
                if bound is not None and type(bound) is not int:
                    raise TypeError("count bounds must be integers")
        if kind == "ratio":
            if self.minimum is not None and float(self.minimum) < 0:
                raise ValueError("ratio minimum cannot be negative")
            if self.maximum is not None and float(self.maximum) > 1:
                raise ValueError("ratio maximum cannot exceed one")

        fields = _normalise_field_mapping(self.fields)
        if kind == "object" and self.item is not None:
            raise ValueError("object fields cannot define an item specification")
        if kind != "object" and fields:
            raise ValueError("only object fields may define nested fields")
        if kind == "array" and self.item is None:
            raise ValueError("array fields require an item specification")
        if kind != "array" and self.item is not None:
            raise ValueError("only array fields may define an item specification")
        item = _coerce_field_spec(self.item) if self.item is not None else None

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "max_length", max_length)
        object.__setattr__(self, "max_items", max_items)
        object.__setattr__(self, "allowed_values", allowed_values)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "item", item)

    @property
    def effective_max_length(self) -> int:
        """Return the finite bound applied to code values."""

        return self.max_length or DEFAULT_CODE_MAX_LENGTH

    @property
    def effective_max_items(self) -> int:
        """Return the finite bound applied to array values."""

        return self.max_items if self.max_items is not None else DEFAULT_ARRAY_MAX_ITEMS


ReportFieldRule = ReportFieldSpec


@dataclass(frozen=True)
class ReportSchema(Mapping[str, ReportFieldSpec]):
    """Deterministic allowlist of fields accepted by :func:`lint_report`."""

    fields: Mapping[str, ReportFieldSpec]
    schema_version: int = REPORT_LINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("report schema version must be a positive integer")
        object.__setattr__(self, "fields", _normalise_field_mapping(self.fields))

    def __getitem__(self, key: str) -> ReportFieldSpec:
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)


@dataclass(frozen=True)
class ReportLintFinding:
    """Safe classification of one schema violation.

    ``path`` is generated only from schema-controlled names and array indexes.
    Unknown input keys are represented by ``<unknown>`` plus a digest, never by
    their source spelling.
    """

    code: str
    path: str
    expected_shape: str
    actual_shape: str
    field_digest: str | None = None

    @property
    def reason(self) -> str:
        """Compatibility alias for the stable classification code."""

        return self.code

    @property
    def category(self) -> str:
        """Return the stable classification category."""

        return self.code

    @property
    def field(self) -> str:
        """Return the safe path, never a rejected field value."""

        return self.path

    def to_dict(self) -> dict[str, str]:
        """Serialize only value-free diagnostic metadata."""

        payload = {
            "code": self.code,
            "path": self.path,
            "expected_shape": self.expected_shape,
            "actual_shape": self.actual_shape,
        }
        if self.field_digest is not None:
            payload["field_digest"] = self.field_digest
        return payload

    def __str__(self) -> str:
        """Return a safe, aggregate-only diagnostic string."""

        return (
            f"{self.code} at {self.path} "
            f"(expected {self.expected_shape}, got {self.actual_shape})"
        )


@dataclass(frozen=True)
class ReportLintResult:
    """Aggregate, value-free result returned by the report linter."""

    findings: tuple[ReportLintFinding, ...] = ()
    checked_field_count: int = 0
    input_field_count: int = 0
    schema_version: int = REPORT_LINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        for value in (self.checked_field_count, self.input_field_count):
            if type(value) is not int or value < 0:
                raise ValueError(
                    "report lint field counts must be non-negative integers"
                )

    @property
    def valid(self) -> bool:
        """Return whether the report satisfied every schema rule."""

        return not self.findings

    @property
    def ok(self) -> bool:
        """Short alias for :attr:`valid`."""

        return self.valid

    @property
    def passed(self) -> bool:
        """Return whether the report passed linting."""

        return self.valid

    @property
    def finding_count(self) -> int:
        """Return the number of classified violations."""

        return len(self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result without copying any report content."""

        by_code: dict[str, int] = {}
        for finding in self.findings:
            by_code[finding.code] = by_code.get(finding.code, 0) + 1
        return {
            "schema_version": self.schema_version,
            "valid": self.valid,
            "checked_field_count": self.checked_field_count,
            "input_field_count": self.input_field_count,
            "finding_count": self.finding_count,
            "findings": [finding.to_dict() for finding in self.findings],
            "by_code": dict(sorted(by_code.items())),
        }

    def __str__(self) -> str:
        """Return an aggregate-only result summary."""

        status = "passed" if self.valid else "failed"
        return f"privacy report lint {status} with {self.finding_count} finding(s)"


class ReportLintError(ValueError):
    """Safe exception raised when strict report linting finds violations."""

    def __init__(self, result: ReportLintResult) -> None:
        if not isinstance(result, ReportLintResult):
            raise TypeError("report lint errors require a lint result")
        self.result = result
        super().__init__(str(result))


def _shape_of(value: object) -> str:
    """Return a coarse, non-content-bearing description of a value."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(
        value,
        (bytes, bytearray, str),
    ):
        return "array"
    return "other"


def _opaque_digest(value: object) -> str:
    """Hash an unknown key without returning or logging its spelling."""

    if isinstance(value, str):
        payload = value.encode("utf-8", errors="surrogatepass")
    elif isinstance(value, bytes):
        payload = value
    else:
        value_type = type(value)
        payload = f"{value_type.__module__}.{value_type.__qualname__}".encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _path_for_field(parent: str, name: str) -> str:
    """Build a path from an already validated schema field name."""

    return f"{parent}.{name}"


def _unknown_key_sort_key(item: tuple[object, object]) -> tuple[str, str]:
    """Return a deterministic ordering key without exposing an unknown key."""

    key, value = item
    return (_opaque_digest(key), _shape_of(value))


def _append_finding(
    findings: list[ReportLintFinding],
    *,
    code: str,
    path: str,
    expected_shape: str,
    actual_shape: str,
    field_digest: str | None = None,
) -> None:
    """Append a safe finding using only static classifications and paths."""

    findings.append(
        ReportLintFinding(
            code=code,
            path=path,
            expected_shape=expected_shape,
            actual_shape=actual_shape,
            field_digest=field_digest,
        )
    )


def _lint_mapping(
    value: Mapping[object, object],
    fields: Mapping[str, ReportFieldSpec],
    *,
    path: str,
    findings: list[ReportLintFinding],
) -> int:
    """Lint one object and return how many allowlisted fields were present."""

    checked = 0
    for name, specification in fields.items():
        if name not in value:
            if specification.required:
                _append_finding(
                    findings,
                    code="missing_required",
                    path=_path_for_field(path, name),
                    expected_shape=specification.kind,
                    actual_shape="missing",
                )
            continue
        checked += 1
        _lint_value(
            value[name],
            specification,
            path=_path_for_field(path, name),
            findings=findings,
        )

    unknown = [(key, item) for key, item in value.items() if key not in fields]
    for key, item in sorted(unknown, key=_unknown_key_sort_key):
        _append_finding(
            findings,
            code="unknown_key",
            path=f"{path}.<unknown>",
            expected_shape="allowlisted field",
            actual_shape=_shape_of(item),
            field_digest=_opaque_digest(key),
        )
    return checked


def _lint_value(
    value: object,
    specification: ReportFieldSpec,
    *,
    path: str,
    findings: list[ReportLintFinding],
) -> None:
    """Classify one value without retaining or interpolating its content."""

    kind = specification.kind
    actual_shape = _shape_of(value)

    if kind == "text":
        _append_finding(
            findings,
            code="forbidden_text",
            path=path,
            expected_shape="hash, count, or bounded code",
            actual_shape=actual_shape,
        )
        return
    if kind == "forbidden":
        _append_finding(
            findings,
            code="forbidden_shape",
            path=path,
            expected_shape="forbidden",
            actual_shape=actual_shape,
        )
        return

    if kind == "count":
        if type(value) is not int:
            _append_finding(
                findings,
                code="invalid_count",
                path=path,
                expected_shape="non-negative integer",
                actual_shape=actual_shape,
            )
        elif (
            value < 0
            or (specification.minimum is not None and value < specification.minimum)
            or (specification.maximum is not None and value > specification.maximum)
        ):
            _append_finding(
                findings,
                code="count_out_of_bounds",
                path=path,
                expected_shape="bounded non-negative integer",
                actual_shape=actual_shape,
            )
        return

    if kind == "hash":
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            _append_finding(
                findings,
                code="invalid_hash",
                path=path,
                expected_shape="sha256 digest",
                actual_shape=actual_shape,
            )
        return

    if kind == "code":
        if not isinstance(value, str):
            _append_finding(
                findings,
                code="invalid_code",
                path=path,
                expected_shape="bounded safe code",
                actual_shape=actual_shape,
            )
        elif len(value) > specification.effective_max_length:
            _append_finding(
                findings,
                code="length_exceeded",
                path=path,
                expected_shape="bounded safe code",
                actual_shape=actual_shape,
            )
        elif _CODE_RE.fullmatch(value) is None or (
            specification.allowed_values and value not in specification.allowed_values
        ):
            _append_finding(
                findings,
                code="invalid_code",
                path=path,
                expected_shape="bounded safe code",
                actual_shape=actual_shape,
            )
        return

    if kind == "boolean":
        if type(value) is not bool:
            _append_finding(
                findings,
                code="invalid_boolean",
                path=path,
                expected_shape="boolean",
                actual_shape=actual_shape,
            )
        return

    if kind in _NUMBER_KINDS:
        if not _is_finite_number(value):
            _append_finding(
                findings,
                code="invalid_number",
                path=path,
                expected_shape="finite number",
                actual_shape=actual_shape,
            )
        elif (
            (
                specification.minimum is not None
                and float(value) < float(specification.minimum)
            )
            or (
                specification.maximum is not None
                and float(value) > float(specification.maximum)
            )
            or (kind == "ratio" and not 0 <= float(value) <= 1)
        ):
            _append_finding(
                findings,
                code="number_out_of_bounds",
                path=path,
                expected_shape="bounded finite number",
                actual_shape=actual_shape,
            )
        return

    if kind == "object":
        if not isinstance(value, Mapping):
            _append_finding(
                findings,
                code="invalid_object",
                path=path,
                expected_shape="object",
                actual_shape=actual_shape,
            )
        else:
            _lint_mapping(value, specification.fields, path=path, findings=findings)
        return

    if kind == "array":
        if not isinstance(value, Sequence) or isinstance(
            value,
            (bytes, bytearray, str),
        ):
            _append_finding(
                findings,
                code="invalid_array",
                path=path,
                expected_shape="bounded array",
                actual_shape=actual_shape,
            )
        elif len(value) > specification.effective_max_items:
            _append_finding(
                findings,
                code="too_many_items",
                path=path,
                expected_shape="bounded array",
                actual_shape=actual_shape,
            )
        else:
            assert specification.item is not None
            for index, item in enumerate(value):
                _lint_value(
                    item,
                    specification.item,
                    path=f"{path}[{index}]",
                    findings=findings,
                )
        return

    _append_finding(
        findings,
        code="forbidden_shape",
        path=path,
        expected_shape="supported report field",
        actual_shape=actual_shape,
    )


def lint_report(
    report: Mapping[object, object],
    schema: ReportSchema | Mapping[str, object],
    *,
    strict: bool = False,
) -> ReportLintResult:
    """Lint an aggregate report against a deterministic typed allowlist.

    The input mapping is inspected in place and is never copied into the
    result.  Unknown keys, raw text, malformed hashes, invalid counts, and
    other forbidden shapes are reported using stable codes only.  Set
    ``strict=True`` to raise :class:`ReportLintError` after producing the same
    safe result.
    """

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    if type(strict) is not bool:
        raise TypeError("strict must be a boolean")
    resolved_schema = (
        schema if isinstance(schema, ReportSchema) else ReportSchema(schema)
    )
    findings: list[ReportLintFinding] = []
    checked = _lint_mapping(
        report,
        resolved_schema.fields,
        path="$",
        findings=findings,
    )
    result = ReportLintResult(
        findings=tuple(findings),
        checked_field_count=checked,
        input_field_count=len(report),
        schema_version=resolved_schema.schema_version,
    )
    if strict and not result.valid:
        raise ReportLintError(result)
    return result


def validate_report(
    report: Mapping[object, object],
    schema: ReportSchema | Mapping[str, object],
    *,
    strict: bool = False,
) -> ReportLintResult:
    """Alias for :func:`lint_report` for validation-oriented call sites."""

    return lint_report(report, schema, strict=strict)


def require_valid_report(
    report: Mapping[object, object],
    schema: ReportSchema | Mapping[str, object],
) -> ReportLintResult:
    """Lint a report and raise a value-free error when it is not valid."""

    return lint_report(report, schema, strict=True)
