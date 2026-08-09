"""Counts-only renderers for comparator benchmark results.

The comparator matrix is useful while a run is in memory, but it can contain
fixture identifiers, adapter metadata, nested reports, and exception details.
This module projects that matrix onto a small, aggregate-only artifact.  The
projection is deliberately allowlisted: it does not serialize arbitrary input
metadata or failure messages.
"""

from __future__ import annotations

import json
import math
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.core.audit import stable_hash
from openmed.eval.comparators import (
    STATUS_NOT_AVAILABLE,
    STATUS_SCORED,
    ComparatorMatrixReport,
)

COMPARATOR_REPORT_ARTIFACT = "openmed.eval.comparator_report"
COMPARATOR_REPORT_SCHEMA_VERSION = 1

METRIC_NAMES = (
    "leakage_rate",
    "character_recall",
    "exact_span_f1",
    "relaxed_span_f1",
)
DEFAULT_METRIC_DEFINITIONS: Mapping[str, str] = {
    "character_recall": (
        "Grapheme-cluster recall over gold protected spans; higher is better."
    ),
    "exact_span_f1": (
        "F1 for exact predicted and gold span boundaries and labels; higher is better."
    ),
    "leakage_rate": (
        "Fraction of gold protected grapheme clusters left exposed; lower is better."
    ),
    "relaxed_span_f1": ("F1 under relaxed span-boundary matching; higher is better."),
}

_MISSING = object()
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/@+\-]{0,127}$")
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$"
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SENSITIVE_IDENTIFIER_RE = re.compile(
    r"(?:\d{5,}|@|\b(?:dob|mrn|patient|phone|ssn)\b)", re.IGNORECASE
)
_SAFE_STATUSES = frozenset({STATUS_SCORED, STATUS_NOT_AVAILABLE, "failed", "other"})
_FAILURE_CATEGORIES = frozenset(
    {"dependency", "execution", "not_available", "timeout", "validation", "other"}
)

_METRIC_PATHS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "leakage_rate": (
        ("leakage_rate",),
        ("metrics", "leakage_rate"),
        ("metrics", "leakage", "overall"),
        ("benchmark_report", "metrics", "leakage", "overall"),
    ),
    "character_recall": (
        ("character_recall",),
        ("metrics", "character_recall"),
        ("metrics", "character_recall", "rate"),
        ("benchmark_report", "metrics", "character_recall", "rate"),
    ),
    "exact_span_f1": (
        ("exact_span_f1",),
        ("metrics", "exact_span_f1"),
        ("metrics", "exact_span_f1", "f1"),
        ("benchmark_report", "metrics", "exact_span_f1", "f1"),
    ),
    "relaxed_span_f1": (
        ("relaxed_span_f1",),
        ("metrics", "relaxed_span_f1"),
        ("metrics", "relaxed_span_f1", "f1"),
        ("benchmark_report", "metrics", "relaxed_span_f1", "f1"),
    ),
}

_METRIC_COUNT_FIELDS: Mapping[str, frozenset[str]] = {
    "leakage_rate": frozenset(
        {
            "denominator",
            "leaked_chars",
            "leaked_graphemes",
            "numerator",
            "total_chars",
            "total_graphemes",
        }
    ),
    "character_recall": frozenset(
        {"covered_chars", "denominator", "matched", "numerator", "total_chars"}
    ),
    "exact_span_f1": frozenset(
        {"false_negatives", "false_positives", "true_positives"}
    ),
    "relaxed_span_f1": frozenset(
        {"false_negatives", "false_positives", "true_positives"}
    ),
}


def default_environment() -> dict[str, str]:
    """Return stable, non-content environment descriptors for local runs."""

    try:
        from openmed.__about__ import __version__
    except (ImportError, AttributeError):
        __version__ = "unknown"
    return {
        "implementation": sys.implementation.name,
        "machine": platform.machine() or "unknown",
        "openmed_version": str(__version__),
        "platform": platform.system() or "unknown",
        "python": platform.python_version(),
    }


def fingerprint_environment(environment: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint of environment metadata.

    The supplied values are used only as hash input.  They are never copied
    into a comparator report.
    """

    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping")
    return stable_hash(_hash_safe_value(environment))


@dataclass(frozen=True)
class ComparatorReportRow:
    """One aggregate comparator row with no source-derived detail."""

    system: str
    status: str
    fixture_count: int
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    metric_counts: Mapping[str, Mapping[str, int | float]] = field(default_factory=dict)
    failure_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "system", _safe_identifier(self.system, "system"))
        object.__setattr__(self, "status", _safe_status(self.status))
        object.__setattr__(self, "fixture_count", _safe_count(self.fixture_count))
        object.__setattr__(
            self,
            "metrics",
            {
                metric: _finite_float(self.metrics.get(metric))
                for metric in METRIC_NAMES
            },
        )
        object.__setattr__(
            self,
            "metric_counts",
            _safe_metric_counts(self.metric_counts),
        )
        object.__setattr__(self, "failure_count", _safe_count(self.failure_count))

    def to_dict(self) -> dict[str, Any]:
        """Return only aggregate metrics and counts for this system."""

        return {
            "failure_count": self.failure_count,
            "fixture_count": self.fixture_count,
            "metric_counts": {
                metric: dict(self.metric_counts[metric])
                for metric in sorted(self.metric_counts)
            },
            "metrics": dict(self.metrics),
            "status": self.status,
            "system": self.system,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ComparatorReport:
    """Deterministic, counts-only comparator report."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    rows: tuple[ComparatorReportRow, ...]
    environment_fingerprint: str = ""
    failure_summary: Mapping[str, Any] = field(default_factory=dict)
    generated_at: str | None = None
    schema_version: int = COMPARATOR_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "suite", _safe_identifier(self.suite, "comparator"))
        object.__setattr__(
            self, "model_name", _safe_identifier(self.model_name, "OpenMed")
        )
        object.__setattr__(self, "device", _safe_identifier(self.device, "unknown"))
        object.__setattr__(self, "fixture_count", _safe_count(self.fixture_count))
        object.__setattr__(
            self,
            "rows",
            tuple(
                row if isinstance(row, ComparatorReportRow) else _row_from_mapping(row)
                for row in self.rows
            ),
        )
        fingerprint = self.environment_fingerprint
        if not _DIGEST_RE.fullmatch(str(fingerprint)):
            fingerprint = fingerprint_environment(default_environment())
        object.__setattr__(self, "environment_fingerprint", str(fingerprint))
        object.__setattr__(
            self,
            "failure_summary",
            _safe_failure_summary(self.failure_summary),
        )
        object.__setattr__(self, "generated_at", _safe_timestamp(self.generated_at))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def summary(self) -> dict[str, int]:
        """Return aggregate system and failure counts."""

        status_counts = {
            status: sum(row.status == status for row in self.rows)
            for status in (STATUS_SCORED, STATUS_NOT_AVAILABLE, "failed", "other")
        }
        return {
            "comparator_count": len(self.rows),
            "failure_count": int(self.failure_summary["total"]),
            "failed_count": status_counts["failed"],
            "not_available_count": status_counts[STATUS_NOT_AVAILABLE],
            "scored_count": status_counts[STATUS_SCORED],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report without raw fixture values."""

        return {
            "artifact_type": COMPARATOR_REPORT_ARTIFACT,
            "device": self.device,
            "environment_fingerprint": self.environment_fingerprint,
            "failure_summary": dict(self.failure_summary),
            "fixture_count": self.fixture_count,
            "generated_at": self.generated_at,
            "metric_definitions": dict(DEFAULT_METRIC_DEFINITIONS),
            "model_name": self.model_name,
            "rows": [row.to_dict() for row in self.rows],
            "schema_version": self.schema_version,
            "summary": self.summary,
            "suite": self.suite,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComparatorReport":
        """Load and re-sanitize a counts-only report mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("comparator report payload must be a mapping")
        raw_rows = payload.get("rows") or ()
        if not isinstance(raw_rows, Sequence) or isinstance(
            raw_rows, (str, bytes, bytearray)
        ):
            raise ValueError("comparator report rows must be a sequence")
        return cls(
            suite=str(payload.get("suite") or "comparator"),
            model_name=str(payload.get("model_name") or "OpenMed"),
            device=str(payload.get("device") or "unknown"),
            fixture_count=_safe_count(payload.get("fixture_count")),
            rows=tuple(_row_from_mapping(row) for row in raw_rows),
            environment_fingerprint=str(payload.get("environment_fingerprint") or ""),
            failure_summary=payload.get("failure_summary") or {},
            generated_at=payload.get("generated_at"),
            schema_version=int(payload.get("schema_version", 1)),
        )

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render the report as deterministic Markdown."""

        summary = self.summary
        lines = [
            f"# Comparator Report: {_markdown_cell(self.suite)}",
            "",
            "## Summary",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Model | `{_markdown_cell(self.model_name)}` |",
            f"| Device | `{_markdown_cell(self.device)}` |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Comparators | {summary['comparator_count']} |",
            f"| Scored | {summary['scored_count']} |",
            f"| Not available | {summary['not_available_count']} |",
            f"| Failures | {summary['failure_count']} |",
            (f"| Environment fingerprint | `{self.environment_fingerprint}` |"),
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(
            [
                "",
                "## Systems",
                "",
                (
                    "| System | Status | Fixtures | Leakage rate | Character recall | "
                    "Exact span F1 | Relaxed span F1 | Failures |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in self.rows:
            lines.append(
                "| "
                f"`{_markdown_cell(row.system)}` | "
                f"{row.status} | "
                f"{row.fixture_count} | "
                f"{_format_percent(row.metrics['leakage_rate'])} | "
                f"{_format_percent(row.metrics['character_recall'])} | "
                f"{_format_percent(row.metrics['exact_span_f1'])} | "
                f"{_format_percent(row.metrics['relaxed_span_f1'])} | "
                f"{row.failure_count} |"
            )

        lines.extend(
            [
                "",
                "## Metric Definitions",
                "",
                "| Metric | Definition |",
                "|---|---|",
            ]
        )
        for metric in METRIC_NAMES:
            lines.append(
                f"| `{metric}` | {_markdown_cell(DEFAULT_METRIC_DEFINITIONS[metric])} |"
            )

        lines.extend(
            [
                "",
                "## Aggregate Failures",
                "",
                "| Category | Count |",
                "|---|---:|",
            ]
        )
        by_category = self.failure_summary["by_category"]
        if by_category:
            for category, count in by_category.items():
                lines.append(f"| `{_markdown_cell(category)}` | {count} |")
        else:
            lines.append("| _none_ | 0 |")
        lines.append(
            f"| Systems affected | {self.failure_summary['systems_affected']} |"
        )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ComparatorReportRenderer:
    """Reusable renderer configured with an optional local environment."""

    environment: Mapping[str, Any] | None = None
    environment_fingerprint: str | None = None

    def build(
        self,
        result: ComparatorMatrixReport | ComparatorReport | Mapping[str, Any],
        *,
        generated_at: str | None = None,
    ) -> ComparatorReport:
        """Build a counts-only report from comparator results."""

        return build_comparator_report(
            result,
            environment=self.environment,
            environment_fingerprint=self.environment_fingerprint,
            generated_at=generated_at,
        )

    def render(
        self,
        result: ComparatorMatrixReport | ComparatorReport | Mapping[str, Any],
        *,
        format: str = "markdown",
        generated_at: str | None = None,
    ) -> str:
        """Render comparator results as Markdown or JSON."""

        report = self.build(result, generated_at=generated_at)
        return _render_report(report, format=format)

    def render_json(
        self,
        result: ComparatorMatrixReport | ComparatorReport | Mapping[str, Any],
        *,
        generated_at: str | None = None,
        indent: int = 2,
    ) -> str:
        """Render comparator results as deterministic JSON."""

        return self.build(result, generated_at=generated_at).to_json(indent=indent)

    def render_markdown(
        self,
        result: ComparatorMatrixReport | ComparatorReport | Mapping[str, Any],
        *,
        generated_at: str | None = None,
    ) -> str:
        """Render comparator results as deterministic Markdown."""

        return self.build(result, generated_at=generated_at).to_markdown()


def build_comparator_report(
    result: ComparatorMatrixReport | ComparatorReport | Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None = None,
    environment_fingerprint: str | None = None,
    generated_at: str | None = None,
) -> ComparatorReport:
    """Project comparator results onto an aggregate-only report.

    ``result`` may be a :class:`ComparatorMatrixReport` or its JSON-ready
    mapping.  Fixture text, fixture identifiers, arbitrary metadata, nested
    benchmark reports, and exception messages are intentionally not copied.
    """

    if isinstance(result, ComparatorReport):
        if (
            environment is None
            and environment_fingerprint is None
            and generated_at is None
        ):
            return result
        source: Any = result.to_dict()
    elif isinstance(result, (ComparatorMatrixReport, Mapping)):
        source = result
    elif isinstance(result, Sequence) and not isinstance(
        result, (str, bytes, bytearray)
    ):
        source = {"rows": tuple(result)}
    else:
        raise TypeError("comparator result must be a matrix report or mapping")

    raw_rows = _source_rows(source)
    rows: list[ComparatorReportRow] = []
    row_failure_systems: set[str] = set()
    failure_categories: list[str] = []
    for raw_row in raw_rows:
        row, events = _project_row(raw_row)
        rows.append(row)
        if events:
            row_failure_systems.add(row.system)
            failure_categories.extend(events)

    failure_categories.extend(_top_level_failure_events(source))
    fixture_count = _safe_count(_value(source, "fixture_count", 0))
    if fixture_count == 0 and rows:
        fixture_count = max(row.fixture_count for row in rows)

    source_metadata = _value(source, "metadata", {})
    if not isinstance(source_metadata, Mapping):
        source_metadata = {}
    if environment is None:
        source_environment = _value(source, "environment", _MISSING)
        if not isinstance(source_environment, Mapping):
            source_environment = source_metadata.get("environment")
        environment = (
            source_environment
            if isinstance(source_environment, Mapping)
            else default_environment()
        )

    fingerprint = environment_fingerprint
    if fingerprint is None:
        source_fingerprint = _value(source, "environment_fingerprint", _MISSING)
        if source_fingerprint is _MISSING:
            source_fingerprint = source_metadata.get("environment_fingerprint")
        if source_fingerprint is not None and source_fingerprint is not _MISSING:
            fingerprint = str(source_fingerprint)
    if fingerprint is None or not _DIGEST_RE.fullmatch(str(fingerprint)):
        fingerprint = fingerprint_environment(environment)

    safe_generated_at = generated_at
    if safe_generated_at is None:
        safe_generated_at = _value(source, "generated_at", None)

    failure_summary = {
        "by_category": {
            category: failure_categories.count(category)
            for category in sorted(set(failure_categories))
        },
        "systems_affected": len(row_failure_systems),
        "total": len(failure_categories),
    }
    return ComparatorReport(
        suite=str(_value(source, "suite", "comparator")),
        model_name=str(_value(source, "model_name", "OpenMed")),
        device=str(_value(source, "device", "unknown")),
        fixture_count=fixture_count,
        rows=tuple(rows),
        environment_fingerprint=str(fingerprint),
        failure_summary=failure_summary,
        generated_at=(
            str(safe_generated_at) if safe_generated_at is not None else None
        ),
    )


def render_comparator_report(
    result: ComparatorMatrixReport | ComparatorReport | Mapping[str, Any],
    *,
    format: str = "markdown",
    output_format: str | None = None,
    environment: Mapping[str, Any] | None = None,
    environment_fingerprint: str | None = None,
    generated_at: str | None = None,
    indent: int = 2,
) -> str:
    """Render comparator results as deterministic Markdown or JSON."""

    report = build_comparator_report(
        result,
        environment=environment,
        environment_fingerprint=environment_fingerprint,
        generated_at=generated_at,
    )
    return _render_report(
        report,
        format=output_format if output_format is not None else format,
        indent=indent,
    )


def render_comparator_report_json(
    result: ComparatorMatrixReport | ComparatorReport | Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None = None,
    environment_fingerprint: str | None = None,
    generated_at: str | None = None,
    indent: int = 2,
) -> str:
    """Render comparator results as deterministic JSON."""

    return render_comparator_report(
        result,
        format="json",
        environment=environment,
        environment_fingerprint=environment_fingerprint,
        generated_at=generated_at,
        indent=indent,
    )


def render_comparator_report_markdown(
    result: ComparatorMatrixReport | ComparatorReport | Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None = None,
    environment_fingerprint: str | None = None,
    generated_at: str | None = None,
) -> str:
    """Render comparator results as deterministic Markdown."""

    return render_comparator_report(
        result,
        format="markdown",
        environment=environment,
        environment_fingerprint=environment_fingerprint,
        generated_at=generated_at,
    )


def write_comparator_report(
    result: ComparatorMatrixReport | ComparatorReport | Mapping[str, Any],
    path: str | Path,
    *,
    format: str | None = None,
    environment: Mapping[str, Any] | None = None,
    environment_fingerprint: str | None = None,
    generated_at: str | None = None,
    indent: int = 2,
) -> Path:
    """Write a counts-only comparator report, selecting format by suffix."""

    output_path = Path(path)
    selected_format = format or (
        "json" if output_path.suffix.lower() == ".json" else "markdown"
    )
    content = render_comparator_report(
        result,
        format=selected_format,
        environment=environment,
        environment_fingerprint=environment_fingerprint,
        generated_at=generated_at,
        indent=indent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        content + ("" if content.endswith("\n") else "\n"), encoding="utf-8"
    )
    return output_path


def _project_row(row: Any) -> tuple[ComparatorReportRow, list[str]]:
    status = _safe_status(_value(row, "status", "other"))
    reason = _value(row, "reason", None)
    events = (
        []
        if status == STATUS_SCORED and not reason
        else [_failure_category(status, reason)]
    )
    events.extend(_row_failure_events(row))
    projected = ComparatorReportRow(
        system=str(_value(row, "system", _value(row, "name", "system"))),
        status=status,
        fixture_count=_safe_count(_value(row, "fixture_count", 0)),
        metrics={metric: _metric_value(row, metric) for metric in METRIC_NAMES},
        metric_counts={
            metric: _metric_counts(row, metric)
            for metric in METRIC_NAMES
            if _metric_counts(row, metric)
        },
        failure_count=len(events),
    )
    return projected, events


def _source_rows(source: Any) -> tuple[Any, ...]:
    raw_rows = _value(source, "rows", _MISSING)
    if raw_rows is _MISSING:
        raw_rows = _value(source, "systems", ())
    if (
        raw_rows == ()
        and isinstance(source, Mapping)
        and not {"suite", "model_name", "device", "fixture_count"}.intersection(source)
    ):
        raw_rows = {
            name: value for name, value in source.items() if isinstance(value, Mapping)
        }
    if isinstance(raw_rows, Mapping):
        rows: list[Any] = []
        for name, value in raw_rows.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("system", name)
                rows.append(row)
        return tuple(rows)
    if isinstance(raw_rows, Sequence) and not isinstance(
        raw_rows, (str, bytes, bytearray)
    ):
        return tuple(raw_rows)
    return ()


def _metric_value(row: Any, metric: str) -> float | None:
    for path in _METRIC_PATHS[metric]:
        value = _path_value(row, path)
        if value is not _MISSING:
            if isinstance(value, Mapping) and "rate" in value:
                value = value["rate"]
            if isinstance(value, Mapping) and "f1" in value:
                value = value["f1"]
            result = _finite_float(value)
            if result is not None:
                return result
    return None


def _metric_counts(row: Any, metric: str) -> dict[str, int | float]:
    for path in _METRIC_PATHS[metric]:
        parent_path = path[:-1]
        candidate = _path_value(row, parent_path) if parent_path else _MISSING
        if isinstance(candidate, Mapping):
            counts = _safe_count_mapping(candidate, _METRIC_COUNT_FIELDS[metric])
            if counts:
                return counts
    return {}


def _row_failure_events(row: Any) -> list[str]:
    events: list[str] = []
    for path in (
        ("metrics",),
        ("benchmark_report", "metrics"),
    ):
        value = _path_value(row, path)
        if value is not _MISSING:
            events.extend(_failure_events_from_mapping(value))
    return events


def _top_level_failure_events(source: Any) -> list[str]:
    events: list[str] = []
    for key in ("failures", "errors", "exceptions"):
        value = _value(source, key, _MISSING)
        if value is not _MISSING:
            events.extend(_failure_events_from_mapping({key: value}))
    for key in ("failure_count", "error_count"):
        value = _value(source, key, _MISSING)
        if value is not _MISSING:
            count = _safe_count(value)
            events.extend("other" for _ in range(count))
    return events


def _failure_events_from_mapping(
    value: Any, *, _seen: set[int] | None = None
) -> list[str]:
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return []
    if isinstance(value, Mapping):
        _seen.add(id(value))
        events: list[str] = []
        for key, child in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if normalized_key in {"failures", "errors", "exceptions"}:
                if isinstance(child, Sequence) and not isinstance(
                    child, (str, bytes, bytearray)
                ):
                    events.extend(_failure_category("failed", item) for item in child)
                elif isinstance(child, Mapping):
                    events.extend(_failure_events_from_mapping(child, _seen=_seen))
                else:
                    events.append(_failure_category("failed", child))
            elif normalized_key in {"failure_count", "error_count"}:
                count = _safe_count(child)
                events.extend("other" for _ in range(count))
            elif isinstance(child, (Mapping, list, tuple)):
                events.extend(_failure_events_from_mapping(child, _seen=_seen))
        return events
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_failure_category("failed", item) for item in value]
    return []


def _failure_category(status: Any, reason: Any) -> str:
    normalized_status = _safe_status(status)
    if normalized_status == STATUS_NOT_AVAILABLE:
        return "not_available"
    text = str(reason or "").lower()
    if any(token in text for token in ("import", "module", "dependency", "extra")):
        return "dependency"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(token in text for token in ("validation", "schema", "invalid")):
        return "validation"
    if normalized_status == "failed" or text:
        return "execution"
    return "other"


def _row_from_mapping(value: Any) -> ComparatorReportRow:
    if isinstance(value, ComparatorReportRow):
        return value
    if not isinstance(value, Mapping):
        return ComparatorReportRow(
            system="system",
            status="other",
            fixture_count=0,
        )
    return ComparatorReportRow(
        system=str(value.get("system") or value.get("name") or "system"),
        status=str(value.get("status") or "other"),
        fixture_count=_safe_count(value.get("fixture_count")),
        metrics=value.get("metrics") or {},
        metric_counts=value.get("metric_counts") or {},
        failure_count=_safe_count(value.get("failure_count")),
    )


def _safe_failure_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    raw_categories = value.get("by_category", {}) if isinstance(value, Mapping) else {}
    categories: dict[str, int] = {}
    if isinstance(raw_categories, Mapping):
        for key, count in raw_categories.items():
            category = str(key) if str(key) in _FAILURE_CATEGORIES else "other"
            categories[category] = categories.get(category, 0) + _safe_count(count)
    return {
        "by_category": dict(sorted(categories.items())),
        "systems_affected": _safe_count(
            value.get("systems_affected", 0) if isinstance(value, Mapping) else 0
        ),
        "total": _safe_count(
            value.get("total", 0) if isinstance(value, Mapping) else 0
        ),
    }


def _safe_metric_counts(
    value: Mapping[str, Mapping[str, int | float]],
) -> dict[str, dict[str, int | float]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, dict[str, int | float]] = {}
    for metric, counts in value.items():
        if metric not in METRIC_NAMES or not isinstance(counts, Mapping):
            continue
        safe_counts = _safe_count_mapping(counts, _METRIC_COUNT_FIELDS[metric])
        if safe_counts:
            result[metric] = safe_counts
    return result


def _safe_count_mapping(
    value: Mapping[str, Any],
    allowed: frozenset[str],
) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key in sorted(allowed):
        if key not in value:
            continue
        number = _finite_float(value[key])
        if number is not None and number >= 0:
            result[key] = int(number) if number.is_integer() else number
    return result


def _safe_status(value: Any) -> str:
    text = str(value or "other").strip().lower().replace(" ", "_")
    if text in {"unavailable", "skipped"}:
        text = STATUS_NOT_AVAILABLE
    if text in {"error", "exception"}:
        text = "failed"
    return text if text in _SAFE_STATUSES else "other"


def _safe_identifier(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if _IDENTIFIER_RE.fullmatch(text) and not _SENSITIVE_IDENTIFIER_RE.search(text):
        return text
    if not text:
        return fallback
    return stable_hash({"identifier": text})


def _safe_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if _TIMESTAMP_RE.fullmatch(text) else None


def _safe_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return len(value) if isinstance(value, (list, tuple, set, frozenset)) else 0
    return max(0, number)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _value(source: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _path_value(source: Any, path: tuple[str, ...]) -> Any:
    current = source
    for key in path:
        current = _value(current, key, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current


def _hash_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _hash_safe_value(value[key]) for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_hash_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return type(value).__name__


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _render_report(report: ComparatorReport, *, format: str, indent: int = 2) -> str:
    normalized = str(format).strip().lower()
    if normalized in {"md", "markdown"}:
        return report.to_markdown()
    if normalized == "json":
        return report.to_json(indent=indent)
    raise ValueError("comparator report format must be 'json' or 'markdown'")


__all__ = [
    "COMPARATOR_REPORT_ARTIFACT",
    "COMPARATOR_REPORT_SCHEMA_VERSION",
    "DEFAULT_METRIC_DEFINITIONS",
    "METRIC_NAMES",
    "ComparatorReport",
    "ComparatorReportRenderer",
    "ComparatorReportRow",
    "build_comparator_report",
    "default_environment",
    "fingerprint_environment",
    "render_comparator_report",
    "render_comparator_report_json",
    "render_comparator_report_markdown",
    "write_comparator_report",
]
