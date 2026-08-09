"""Build deterministic, privacy-safe FHIR R4 validation reports.

This module is a boundary for validators that may receive protected values in
their error messages.  Callers provide findings with a category, an optional
FHIRPath expression, and optional diagnostic text.  The report never copies
that diagnostic text into an exception, log, or rendered resource.  Instead it
emits a category-level diagnostic and the structural location after removing
literal values from the expression.

The implementation is deliberately mechanical and local-only.  It does not
load a validator, contact a terminology service, or mutate the supplied
findings.  Structural findings map to ``structure`` and policy findings map to
``business-rule`` by default; callers can provide an explicit FHIR issue code
when a more specific code is appropriate.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...clinical.exporters.fhir import (
    OperationOutcomeIssue,
)
from ...clinical.exporters.fhir import (
    to_operation_outcome as _to_r4_operation_outcome,
)

__all__ = [
    "FHIRValidationReport",
    "OperationOutcomeReport",
    "ValidationFinding",
    "build_operation_outcome",
    "redact_diagnostic",
    "render_counts",
    "render_counts_text",
    "render_json",
    "render_operation_outcome_json",
    "to_operation_outcome",
]


_SEVERITIES = frozenset({"fatal", "error", "warning", "information"})
_SEVERITY_ALIASES = {
    "critical": "fatal",
    "fatal": "fatal",
    "error": "error",
    "err": "error",
    "warning": "warning",
    "warn": "warning",
    "information": "information",
    "informational": "information",
    "info": "information",
}

# FHIR R4 issue-type values accepted by the shared OperationOutcome builder.
_ISSUE_CODES = frozenset(
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

_CATEGORY_ALIASES = {
    "business": "policy",
    "business_rule": "policy",
    "businessrule": "policy",
    "business-rule": "policy",
    "code": "terminology",
    "coding": "terminology",
    "format": "structural",
    "invariant": "invariant",
    "missing": "required",
    "parse": "processing",
    "policy": "policy",
    "privacy": "security",
    "processing": "processing",
    "required": "required",
    "schema": "structural",
    "security": "security",
    "structure": "structural",
    "structural_failure": "structural",
    "structural": "structural",
    "terminology": "terminology",
    "type": "structural",
    "unsupported": "unsupported",
    "value": "value",
    "policy_failure": "policy",
    "policy_violation": "policy",
}

_CATEGORY_TO_CODE = {
    "invariant": "invariant",
    "policy": "business-rule",
    "processing": "processing",
    "required": "required",
    "security": "security",
    "structural": "structure",
    "terminology": "code-invalid",
    "unknown": "invalid",
    "unsupported": "not-supported",
    "value": "value",
}

_CATEGORY_LABELS = {
    "invariant": "Invariant",
    "policy": "Policy",
    "processing": "Processing",
    "required": "Required-element",
    "security": "Security-policy",
    "structural": "Structural",
    "terminology": "Terminology",
    "unknown": "Validation",
    "unsupported": "Unsupported-rule",
    "value": "Value",
}

_DIAGNOSTIC_BY_CODE = {
    "business-rule": "Policy validation failed; details redacted.",
    "code-invalid": "Terminology validation failed; details redacted.",
    "invariant": "Invariant validation failed; details redacted.",
    "not-supported": "Unsupported validation rule; details redacted.",
    "processing": "Validation processing failed; details redacted.",
    "required": "Required-element validation failed; details redacted.",
    "security": "Security-policy validation failed; details redacted.",
    "structure": "Structural validation failed; details redacted.",
    "value": "Value validation failed; details redacted.",
}

_RESULT_COLLECTION_KEYS = ("findings", "issues", "violations")
_RESULT_BUCKETS = (
    ("fatal", "fatal", "unknown"),
    ("fatals", "fatal", "unknown"),
    ("errors", "error", "unknown"),
    ("error", "error", "unknown"),
    ("warnings", "warning", "unknown"),
    ("warning", "warning", "unknown"),
    ("information", "information", "unknown"),
    ("informational", "information", "unknown"),
    ("structural", "error", "structural"),
    ("policy", "error", "policy"),
    ("structural_failures", "error", "structural"),
    ("policy_failures", "error", "policy"),
)

_PATH_LITERAL = re.compile(r"(['\"])(?:\\.|(?!\1).)*\1")
_PATH_SECRET = re.compile(
    r"(?i)\b(?:mrn|ssn|account|member|token|secret|api[_-]?key)\b"
    r"(?:\s*[-:=/]\s*|\s+)[a-z0-9][a-z0-9_.-]*"
)
_PATH_LONG_NUMBER = re.compile(r"(?<![\[\]])\b\d{4,}\b")
_PATH_RESOURCE_ID = re.compile(r"(?<![:/])\/([a-z0-9][a-z0-9_.-]*)\b", re.IGNORECASE)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """One validator finding accepted by :class:`FHIRValidationReport`.

    Args:
        category: Rule family, for example ``"structural"`` or ``"policy"``.
        path: FHIRPath-style expression identifying the affected element.
        severity: FHIR issue severity. It defaults to ``"error"``.
        code: Optional explicit FHIR issue-type code. When omitted, the code is
            derived from ``category``.
        diagnostics: Optional source diagnostic. It is intentionally ignored
            during rendering so protected values cannot cross the report
            boundary.
        rule: Optional rule identifier retained for caller-side bookkeeping;
            it is never rendered.

    ``expression`` and ``message`` are accepted as keyword aliases for
    ``path`` and ``diagnostics`` by the constructor for validator adapters.
    """

    category: str
    path: str | Sequence[str] | None = None
    severity: str = "error"
    code: str | None = None
    diagnostics: str | None = field(default=None, repr=False)
    rule: str | None = field(default=None, repr=False)

    def __init__(
        self,
        category: str,
        path: str | Sequence[str] | None = None,
        *,
        severity: str = "error",
        code: str | None = None,
        diagnostics: str | None = None,
        rule: str | None = None,
        expression: str | Sequence[str] | None = None,
        message: str | None = None,
    ) -> None:
        """Create a finding without retaining any source diagnostic in repr."""

        if path is None:
            path = expression
        if diagnostics is None:
            diagnostics = message
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "rule", rule)

    @property
    def expression(self) -> str | Sequence[str] | None:
        """Return ``path`` under the FHIR-native name ``expression``."""

        return self.path


@dataclass(frozen=True, slots=True)
class _NormalizedFinding:
    category: str
    severity: str
    code: str
    expressions: tuple[str, ...]
    diagnostics: str


def redact_diagnostic(
    diagnostic: Any = None,
    *,
    category: str = "unknown",
    code: str | None = None,
) -> str:
    """Return a safe, category-level diagnostic without echoing input text.

    The input is accepted for adapter ergonomics but is never interpolated or
    parsed. This conservative behavior is intentional: a generic redaction
    regex cannot reliably distinguish a person's name or a clinical value from
    ordinary prose. The returned sentence contains only canonical metadata.
    """

    del diagnostic
    normalized_category = _normalize_category(category)
    normalized_code = _normalize_code(code, normalized_category)
    return _DIAGNOSTIC_BY_CODE.get(
        normalized_code,
        f"{_CATEGORY_LABELS[normalized_category]} validation failed; details redacted.",
    )


class FHIRValidationReport:
    """Immutable, deterministic view of FHIR validation findings.

    The report normalizes and sorts findings when it is created. Its JSON form
    is an R4 ``OperationOutcome``; its text form contains aggregate counts only.
    """

    __slots__ = ("_findings",)

    def __init__(self, findings: Any = ()) -> None:
        self._findings = _normalize_findings(findings)

    @classmethod
    def from_findings(cls, findings: Any = ()) -> "FHIRValidationReport":
        """Build a report from an iterable or validator-result-like object."""

        return cls(findings)

    @property
    def findings(self) -> tuple[_NormalizedFinding, ...]:
        """Return normalized findings with diagnostics safe for inspection."""

        return self._findings

    @property
    def issue_count(self) -> int:
        """Return the number of validation issues, excluding all-ok output."""

        return len(self._findings)

    def to_operation_outcome(self) -> dict[str, Any]:
        """Return this report as a FHIR R4 ``OperationOutcome`` mapping."""

        issues = [
            OperationOutcomeIssue(
                severity=finding.severity,
                code=finding.code,
                diagnostics=finding.diagnostics,
                expression=list(finding.expressions) or None,
            )
            for finding in self._findings
        ]
        return _to_r4_operation_outcome(issues)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return a canonical JSON rendering with a trailing newline."""

        return (
            json.dumps(
                self.to_operation_outcome(),
                ensure_ascii=False,
                indent=indent,
                sort_keys=True,
            )
            + "\n"
        )

    def counts(self) -> dict[str, Any]:
        """Return deterministic aggregate counts without paths or diagnostics."""

        return {
            "total": self.issue_count,
            "by_category": _sorted_counts(
                finding.category for finding in self._findings
            ),
            "by_code": _sorted_counts(finding.code for finding in self._findings),
            "by_severity": _sorted_counts(
                finding.severity for finding in self._findings
            ),
        }

    def to_counts_text(self) -> str:
        """Return counts-only text with stable, line-oriented keys."""

        counts = self.counts()
        lines = [f"total={counts['total']}"]
        for group in ("by_category", "by_code", "by_severity"):
            prefix = group.removeprefix("by_")
            lines.extend(
                f"{prefix}.{key}={value}" for key, value in counts[group].items()
            )
        return "\n".join(lines) + "\n"

    def render(self, *, format: str = "json", indent: int | None = 2) -> str:
        """Render JSON or counts-only text.

        ``format`` accepts ``"json"`` and the aliases ``"counts"``,
        ``"counts-only"``, and ``"text"``.
        """

        normalized = format.strip().lower()
        if normalized == "json":
            return self.to_json(indent=indent)
        if normalized in {"counts", "counts-only", "text"}:
            return self.to_counts_text()
        raise ValueError("format must be 'json' or 'counts-only'")


OperationOutcomeReport = FHIRValidationReport


def build_operation_outcome(findings: Any = ()) -> dict[str, Any]:
    """Build a safe FHIR R4 ``OperationOutcome`` from validation findings."""

    return FHIRValidationReport(findings).to_operation_outcome()


def to_operation_outcome(findings: Any = ()) -> dict[str, Any]:
    """Alias for :func:`build_operation_outcome` for FHIR adapter callers."""

    return build_operation_outcome(findings)


def render_json(findings: Any = (), *, indent: int | None = 2) -> str:
    """Render findings as deterministic, privacy-safe OperationOutcome JSON."""

    report = (
        findings
        if isinstance(findings, FHIRValidationReport)
        else FHIRValidationReport(findings)
    )
    return report.to_json(indent=indent)


def render_operation_outcome_json(findings: Any = (), *, indent: int | None = 2) -> str:
    """Alias for :func:`render_json`."""

    return render_json(findings, indent=indent)


def render_counts(findings: Any = ()) -> str:
    """Render findings as counts-only text with no paths or diagnostics."""

    report = (
        findings
        if isinstance(findings, FHIRValidationReport)
        else FHIRValidationReport(findings)
    )
    return report.to_counts_text()


def render_counts_text(findings: Any = ()) -> str:
    """Alias for :func:`render_counts`."""

    return render_counts(findings)


def _normalize_findings(source: Any) -> tuple[_NormalizedFinding, ...]:
    raw_findings = list(_iter_findings(source))
    normalized = [
        _normalize_finding(item, category, severity)
        for item, category, severity in raw_findings
    ]
    return tuple(
        sorted(
            normalized,
            key=lambda finding: (
                finding.expressions[0] if finding.expressions else "",
                finding.code,
                _severity_rank(finding.severity),
                finding.category,
                finding.expressions,
            ),
        )
    )


def _iter_findings(
    source: Any,
    fallback_category: str | None = None,
    fallback_severity: str | None = None,
) -> Iterable[tuple[Any, str | None, str | None]]:
    if source is None:
        return

    if isinstance(source, Mapping):
        if _looks_like_finding(source):
            yield source, fallback_category, fallback_severity
            return

        found_collection = False
        for key in _RESULT_COLLECTION_KEYS:
            if key not in source:
                continue
            found_collection = True
            yield from _iter_findings(source[key], fallback_category, fallback_severity)
        for key, severity, category in _RESULT_BUCKETS:
            if key not in source:
                continue
            found_collection = True
            yield from _iter_bucket(source[key], category, severity)
        if not found_collection:
            yield source, fallback_category, fallback_severity
        return

    if isinstance(source, (str, bytes)):
        yield source, fallback_category, fallback_severity
        return

    if isinstance(source, Iterable):
        for item in source:
            yield item, fallback_category, fallback_severity
        return

    for key in _RESULT_COLLECTION_KEYS:
        collection = getattr(source, key, None)
        if collection is not None:
            yield from _iter_findings(collection, fallback_category, fallback_severity)
            return
    for key, severity, category in _RESULT_BUCKETS:
        bucket = getattr(source, key, None)
        if bucket is not None:
            yield from _iter_bucket(bucket, category, severity)
            return
    yield source, fallback_category, fallback_severity


def _iter_bucket(
    bucket: Any,
    category: str | None,
    severity: str | None,
) -> Iterable[tuple[Any, str | None, str | None]]:
    if isinstance(bucket, Mapping) or isinstance(bucket, (str, bytes)):
        yield bucket, category, severity
        return
    if isinstance(bucket, Iterable):
        for item in bucket:
            yield item, category, severity
        return
    yield bucket, category, severity


def _looks_like_finding(value: Mapping[str, Any]) -> bool:
    return bool(
        {
            "category",
            "kind",
            "rule_category",
            "type",
            "path",
            "expression",
            "location",
            "fhir_path",
            "fhirpath",
            "severity",
            "level",
            "code",
            "issue_code",
            "diagnostics",
            "message",
            "detail",
            "rule",
            "rule_id",
        }
        & value.keys()
    )


def _normalize_finding(
    raw: Any,
    fallback_category: str | None,
    fallback_severity: str | None,
) -> _NormalizedFinding:
    if isinstance(raw, ValidationFinding):
        category_value = raw.category
        path_value = raw.path
        severity_value = raw.severity
        code_value = raw.code
        diagnostic_value = raw.diagnostics
    elif isinstance(raw, Mapping):
        category_value = _first_value(
            raw, ("category", "kind", "rule_category", "type")
        )
        path_value = _first_value(
            raw, ("path", "expression", "location", "fhir_path", "fhirpath")
        )
        severity_value = _first_value(raw, ("severity", "level"))
        code_value = _first_value(raw, ("code", "issue_code"))
        diagnostic_value = _first_value(raw, ("diagnostics", "message", "detail"))
    elif isinstance(raw, (str, bytes)):
        category_value = None
        path_value = None
        severity_value = None
        code_value = None
        diagnostic_value = None
    else:
        category_value = _first_attribute(
            raw, ("category", "kind", "rule_category", "type")
        )
        path_value = _first_attribute(
            raw, ("path", "expression", "location", "fhir_path", "fhirpath")
        )
        severity_value = _first_attribute(raw, ("severity", "level"))
        code_value = _first_attribute(raw, ("code", "issue_code"))
        diagnostic_value = _first_attribute(raw, ("diagnostics", "message", "detail"))

    category = _normalize_category(category_value or fallback_category)
    severity = _normalize_severity(severity_value or fallback_severity or "error")
    code = _normalize_code(code_value, category)
    expressions = _safe_expressions(path_value)
    diagnostics = redact_diagnostic(
        diagnostic_value,
        category=category,
        code=code,
    )
    return _NormalizedFinding(category, severity, code, expressions, diagnostics)


def _normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    key = value.strip().lower().replace(" ", "_")
    return _CATEGORY_ALIASES.get(key, key if key in _CATEGORY_TO_CODE else "unknown")


def _normalize_severity(value: Any) -> str:
    if not isinstance(value, str):
        return "error"
    normalized = _SEVERITY_ALIASES.get(value.strip().lower())
    if normalized in _SEVERITIES:
        return normalized
    return "error"


def _normalize_code(value: Any, category: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "-")
        if normalized in _ISSUE_CODES:
            return normalized
    return _CATEGORY_TO_CODE[category]


def _safe_expressions(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Iterable[Any]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        values = (value,)

    safe: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        expression = _redact_expression(item)
        if expression:
            safe.add(expression)
    return tuple(sorted(safe))


def _redact_expression(value: str) -> str | None:
    if _CONTROL_CHARACTERS.search(value):
        return None
    expression = " ".join(value.split())
    if not expression or len(expression) > 256:
        return None
    expression = _PATH_LITERAL.sub('"[REDACTED]"', expression)
    expression = _PATH_SECRET.sub("[REDACTED]", expression)
    expression = _PATH_LONG_NUMBER.sub("[REDACTED]", expression)
    expression = _PATH_RESOURCE_ID.sub("/[REDACTED]", expression)
    return expression


def _severity_rank(severity: str) -> int:
    return {"fatal": 0, "error": 1, "warning": 2, "information": 3}[severity]


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _first_value(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    return None


def _first_attribute(value: Any, keys: Sequence[str]) -> Any:
    for key in keys:
        candidate = getattr(value, key, None)
        if candidate is not None:
            return candidate
    return None
