"""Build R4 ``OperationOutcome`` resources from internal result objects.

The FHIR-native way to report errors, warnings, and informational notes from an
operation or a validation pass is the ``OperationOutcome`` resource. OpenMed's
interop surfaces -- the ``$de-identify`` wrapper, the structural validator, and
the US Core conformance checks -- each return their own ad-hoc result objects.
This module is the single shared adapter that turns those internal results into
a standard ``OperationOutcome`` that any FHIR server or client understands.

Two entry points are exposed:

* :func:`to_operation_outcome` takes an iterable of *issues* (mappings or
  duck-typed objects carrying ``severity``/``code``/optional ``diagnostics``/
  optional ``expression`` or ``expressions``) and returns a valid R4
  ``OperationOutcome`` mapping.
* :func:`from_validation_result` adapts a validator/conformance result object
  (anything exposing an ``issues`` collection, or ``errors``/``warnings``/
  ``information`` buckets) into issues and delegates to
  :func:`to_operation_outcome`.

Severities are constrained to the FHIR ``issue-severity`` value set
(``fatal``/``error``/``warning``/``information``) and codes to the
``issue-type`` value set. An empty issue list yields a clean ``all-ok``
information outcome, because an ``OperationOutcome`` must always carry at least
one ``issue``.

The builder is purely mechanical: it never inspects PHI, only the structural
metadata of each issue (severity, code, a human-readable diagnostic string, and
a FHIRPath-style location).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["OperationOutcomeIssue", "to_operation_outcome", "from_validation_result"]


@dataclass(frozen=True)
class OperationOutcomeIssue:
    """OpenMed-owned issue shape that maps directly to ``OperationOutcome.issue``.

    ``severity`` and ``code`` must already be FHIR R4 issue-severity and
    issue-type codes. ``expression`` is a FHIRPath-style location and is emitted
    as R4 ``issue.expression``.
    """

    severity: str
    code: str
    diagnostics: str | None = None
    expression: str | Sequence[str] | None = None


# FHIR R4 ``issue-severity`` value set.
# https://hl7.org/fhir/R4/valueset-issue-severity.html
_SEVERITIES = frozenset({"fatal", "error", "warning", "information"})

# FHIR R4 ``issue-type`` value set (the codes used for ``OperationOutcome.issue.code``).
# https://hl7.org/fhir/R4/valueset-issue-type.html
_ISSUE_TYPES = frozenset(
    {
        "invalid",
        "structure",
        "required",
        "value",
        "invariant",
        "security",
        "login",
        "unknown",
        "expired",
        "forbidden",
        "suppressed",
        "processing",
        "not-supported",
        "duplicate",
        "multiple-matches",
        "not-found",
        "deleted",
        "too-long",
        "code-invalid",
        "extension",
        "too-costly",
        "business-rule",
        "conflict",
        "transient",
        "lock-error",
        "no-store",
        "exception",
        "timeout",
        "incomplete",
        "throttled",
        "informational",
    }
)

# Default issue type for validator/conformance findings (a structural problem).
_DEFAULT_VALIDATION_CODE = "invalid"

# The canonical "everything validated cleanly" outcome.
_ALL_OK_DIAGNOSTICS = "No issues detected."


def to_operation_outcome(issues: Any) -> dict[str, Any]:
    """Build an R4 ``OperationOutcome`` from an iterable of issues.

    Args:
        issues: An iterable of issue-like items, a single issue mapping, or
            ``None``. Each item may be an :class:`OperationOutcomeIssue`, a
            mapping, or a duck-typed object exposing ``severity`` and ``code``,
            optional ``diagnostics``/``message``, and an optional FHIRPath
            location as ``expression`` (str or list) or ``expressions`` (list).

    Returns:
        A ``resourceType=OperationOutcome`` mapping with one ``issue`` per input
        item. When ``issues`` is empty, a single ``information``/
        ``informational`` "all-ok" issue is returned.

    Raises:
        TypeError: If an issue item is not mapping/object shaped.
        ValueError: If an issue carries a severity outside the FHIR
            ``issue-severity`` value set, a code outside the ``issue-type``
            value set, or omits required ``severity``/``code`` fields.
    """

    built = [_build_issue(raw) for raw in _iter_issues(issues)]
    if not built:
        built = [_all_ok_issue()]
    return {"resourceType": "OperationOutcome", "issue": built}


def from_validation_result(result: Any) -> dict[str, Any]:
    """Adapt a validator/conformance result into an ``OperationOutcome``.

    Recognises two common result shapes via duck typing:

    * a result exposing an ``issues`` collection (mapping key or attribute),
      where each entry is itself an issue-like item; or
    * a result exposing ``fatals``/``errors``/``warnings``/``information``
      (a.k.a. ``informational``) buckets, where each entry is a string or an
      issue-like item. Entries in a bucket inherit that bucket's severity and a
      validation-appropriate default code (``invalid``) unless they override it.

    Args:
        result: The validator or US Core ``ConformanceResult`` object, or
            ``None``.

    Returns:
        A valid R4 ``OperationOutcome``; an all-ok information outcome when the
        result reports no issues.
    """

    return to_operation_outcome(_extract_issues(result))


# --- internal helpers -------------------------------------------------------

# (attribute/key, severity, default issue-type) for bucket-style results.
_RESULT_BUCKETS = (
    ("fatal", "fatal", _DEFAULT_VALIDATION_CODE),
    ("fatals", "fatal", _DEFAULT_VALIDATION_CODE),
    ("error", "error", _DEFAULT_VALIDATION_CODE),
    ("errors", "error", _DEFAULT_VALIDATION_CODE),
    ("warning", "warning", _DEFAULT_VALIDATION_CODE),
    ("warnings", "warning", _DEFAULT_VALIDATION_CODE),
    ("info", "information", "informational"),
    ("information", "information", "informational"),
    ("informational", "information", "informational"),
)

_DIAGNOSTIC_KEYS = ("diagnostics", "message", "detail")
_EXPRESSION_KEYS = (
    "expressions",
    "expression",
    "path",
    "fhir_path",
    "fhirpath",
    "field",
    "location",  # accepted as legacy input only; never emitted as R4 location
)
_MISSING = object()


def _iter_issues(issues: Any) -> list[Any]:
    """Normalise the ``issues`` argument into a list of raw issue items."""

    if issues is None:
        return []
    if isinstance(issues, Mapping):
        return [issues]
    if isinstance(issues, (str, bytes)):
        raise TypeError(
            "to_operation_outcome expects an iterable of issues, not a string"
        )
    if isinstance(issues, Iterable):
        return list(issues)
    raise TypeError(f"unsupported issues type: {type(issues).__name__!r}")


def _extract_issues(result: Any) -> list[Any]:
    """Pull issue-like items out of a validator/conformance result object."""

    if result is None:
        return []

    issues = _get(result, "issues")
    if issues is not None:
        return _iter_issues(issues)

    collected: list[Any] = []
    for attr, severity, default_code in _RESULT_BUCKETS:
        bucket = _get(result, attr)
        if not bucket:
            continue
        for item in _iter_bucket(bucket):
            collected.append(_bucket_issue(item, severity, default_code))
    return collected


def _iter_bucket(bucket: Any) -> list[Any]:
    """Normalise a severity bucket into raw entries without splitting strings."""

    if isinstance(bucket, Mapping):
        return [bucket]
    if isinstance(bucket, (str, bytes)):
        return [bucket.decode() if isinstance(bucket, bytes) else bucket]
    if isinstance(bucket, Iterable):
        return list(bucket)
    return [bucket]


def _bucket_issue(item: Any, severity: str, default_code: str) -> dict[str, Any]:
    """Coerce a single bucket entry into a raw issue mapping with a severity."""

    if isinstance(item, str):
        return {"severity": severity, "code": default_code, "diagnostics": item}
    if isinstance(item, Mapping):
        merged = dict(item)
        if merged.get("severity") is None:
            merged["severity"] = severity
        if merged.get("code") is None:
            merged["code"] = default_code
        return merged
    return {
        "severity": _get(item, "severity") or severity,
        "code": _get(item, "code") or default_code,
        "diagnostics": _first_present(item, _DIAGNOSTIC_KEYS),
        "expression": _first_present(item, _EXPRESSION_KEYS),
    }


def _build_issue(raw: Any) -> dict[str, Any]:
    """Build one validated ``OperationOutcome.issue`` from a raw issue item.

    The OpenMed-owned issue shape is ``severity`` (required), ``code``
    (required), optional ``diagnostics``, and an optional FHIRPath location
    given as either ``expression`` (str or list) or ``expressions`` (list). The
    emitted R4 issue always uses ``expression`` (``location`` is deprecated in
    R4 and never emitted).
    """

    if isinstance(raw, (str, bytes)) or not _is_issue_like(raw):
        raise TypeError(
            "each issue must be a mapping, OperationOutcomeIssue, or object "
            "with severity/code fields"
        )

    severity = _required(raw, "severity")
    code = _required(raw, "code")
    diagnostics = _first_present(raw, _DIAGNOSTIC_KEYS)
    expression = _first_present(raw, _EXPRESSION_KEYS)

    issue: dict[str, Any] = {
        "severity": _normalize_severity(severity),
        "code": _normalize_code(code),
    }
    if diagnostics is not None:
        issue["diagnostics"] = str(diagnostics)
    expressions = _normalize_expression(expression)
    if expressions:
        issue["expression"] = expressions
    return issue


def _is_issue_like(raw: Any) -> bool:
    """Return whether ``raw`` can reasonably represent one issue."""

    return isinstance(raw, (OperationOutcomeIssue, Mapping)) or (
        _get(raw, "severity", _MISSING) is not _MISSING
        or _get(raw, "code", _MISSING) is not _MISSING
    )


def _required(obj: Any, key: str) -> Any:
    """Read a required issue field from ``obj``."""

    value = _get(obj, key, _MISSING)
    if value is _MISSING or value is None:
        raise ValueError(f"issue {key!r} is required")
    return value


def _all_ok_issue() -> dict[str, Any]:
    """Return the canonical ``information``/``informational`` all-ok issue."""

    return {
        "severity": "information",
        "code": "informational",
        "diagnostics": _ALL_OK_DIAGNOSTICS,
    }


def _normalize_severity(severity: Any) -> str:
    """Validate ``severity`` against the FHIR ``issue-severity`` value set."""

    if not isinstance(severity, str):
        raise ValueError(f"issue severity must be a string, got {severity!r}")
    canonical = severity.strip().lower()
    if canonical not in _SEVERITIES:
        raise ValueError(
            f"invalid issue severity {severity!r}; "
            f"expected one of {sorted(_SEVERITIES)}"
        )
    return canonical


def _normalize_code(code: Any) -> str:
    """Validate ``code`` against the FHIR ``issue-type`` value set."""

    if not isinstance(code, str):
        raise ValueError(f"issue code must be a string, got {code!r}")
    canonical = code.strip().lower()
    if canonical not in _ISSUE_TYPES:
        raise ValueError(
            f"invalid issue code {code!r}; not in the FHIR issue-type value set"
        )
    return canonical


def _normalize_expression(expression: Any) -> list[str]:
    """Normalise a FHIRPath ``expression`` into a list of non-empty strings."""

    if expression is None:
        return []
    if isinstance(expression, str):
        expression = expression.strip()
        return [expression] if expression else []
    if isinstance(expression, (list, tuple)):
        return [
            normalized
            for item in expression
            if item is not None and (normalized := str(item).strip())
        ]
    normalized = str(expression).strip()
    return [normalized] if normalized else []


def _first_present(obj: Any, keys: Sequence[str]) -> Any:
    """Return the first non-``None`` key/attribute value from ``obj``."""

    for key in keys:
        value = _get(obj, key, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or an object's attribute, else ``default``."""

    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)
