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
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.core.audit import stable_hash
from openmed.eval.comparators import (
    DEFAULT_COMPARATOR_ADAPTERS,
    OPENMED_SYSTEM_NAME,
    STATUS_NOT_AVAILABLE,
    STATUS_SCORED,
    ComparatorMatrixReport,
    ComparatorMatrixRow,
)
from openmed.eval.report import BenchmarkReport

COMPARATOR_REPORT_ARTIFACT = "openmed.eval.comparator_report"
COMPARATOR_REPORT_SCHEMA_VERSION = 1

METRIC_NAMES = (
    "leakage_rate",
    "character_recall",
    "exact_span_f1",
    "relaxed_span_f1",
)
DEFAULT_METRIC_DEFINITIONS: Mapping[str, str] = MappingProxyType(
    {
        "character_recall": (
            "Grapheme-cluster recall over gold protected spans; higher is better."
        ),
        "exact_span_f1": (
            "F1 for exact predicted and gold span boundaries and labels; higher is better."
        ),
        "leakage_rate": (
            "Fraction of gold protected grapheme clusters left exposed; lower is better."
        ),
        "relaxed_span_f1": (
            "F1 under relaxed span-boundary matching; higher is better."
        ),
    }
)

_MISSING = object()
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$"
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_STATUSES = frozenset({STATUS_SCORED, STATUS_NOT_AVAILABLE, "failed", "other"})
_FAILURE_CATEGORIES = frozenset(
    {"dependency", "execution", "not_available", "timeout", "validation", "other"}
)
_PUBLIC_IDENTIFIERS = frozenset(
    {
        OPENMED_SYSTEM_NAME,
        *DEFAULT_COMPARATOR_ADAPTERS,
        "comparator",
        "cpu",
        "cuda",
        "metal",
        "mps",
        "system",
        "unknown",
        "xpu",
    }
)
_MAX_COUNT = 1_000_000_000
_MAX_REPORT_ROWS = 256
_MAX_FAILURE_DEPTH = 8
_MAX_FAILURE_ITEMS = 4096
_MAX_ENVIRONMENT_DEPTH = 8
_MAX_ENVIRONMENT_ITEMS = 256
_MAX_ENVIRONMENT_TEXT = 4096
_MAX_JSON_INDENT = 8

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

    if type(environment) is not dict:
        raise TypeError("environment must be a plain dictionary")
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
        raw_metrics = self.metrics if type(self.metrics) is dict else {}
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(
                {
                    metric: _safe_metric(raw_metrics.get(metric))
                    for metric in METRIC_NAMES
                }
            ),
        )
        safe_metric_counts = _safe_metric_counts(self.metric_counts)
        object.__setattr__(
            self,
            "metric_counts",
            MappingProxyType(
                {
                    metric: MappingProxyType(dict(counts))
                    for metric, counts in safe_metric_counts.items()
                }
            ),
        )
        failure_count = _safe_count(self.failure_count)
        if self.status != STATUS_SCORED and failure_count == 0:
            failure_count = 1
        object.__setattr__(self, "failure_count", failure_count)

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
        if type(self.rows) is not tuple or len(self.rows) > _MAX_REPORT_ROWS:
            raise ValueError("comparator report rows must be a bounded tuple")
        object.__setattr__(
            self,
            "rows",
            tuple(
                row if type(row) is ComparatorReportRow else _row_from_mapping(row)
                for row in self.rows
            ),
        )
        fixture_count = _safe_count(self.fixture_count)
        if self.rows:
            fixture_count = max(
                fixture_count,
                max(row.fixture_count for row in self.rows),
            )
        object.__setattr__(self, "fixture_count", fixture_count)
        fingerprint = self.environment_fingerprint
        if type(fingerprint) is not str or not _DIGEST_RE.fullmatch(fingerprint):
            fingerprint = fingerprint_environment(default_environment())
        object.__setattr__(self, "environment_fingerprint", fingerprint)
        safe_failure_summary = _safe_failure_summary(
            self.failure_summary,
            rows=self.rows,
        )
        object.__setattr__(
            self,
            "failure_summary",
            MappingProxyType(
                {
                    "by_category": MappingProxyType(
                        dict(safe_failure_summary["by_category"])
                    ),
                    "systems_affected": safe_failure_summary["systems_affected"],
                    "total": safe_failure_summary["total"],
                }
            ),
        )
        object.__setattr__(self, "generated_at", _safe_timestamp(self.generated_at))
        if type(self.schema_version) is not int or (
            self.schema_version != COMPARATOR_REPORT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported comparator report schema version")

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
            "failure_summary": {
                "by_category": dict(self.failure_summary["by_category"]),
                "systems_affected": self.failure_summary["systems_affected"],
                "total": self.failure_summary["total"],
            },
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

        if type(payload) is not dict:
            raise TypeError("comparator report payload must be a plain dictionary")
        artifact_type = payload.get("artifact_type", COMPARATOR_REPORT_ARTIFACT)
        if (
            type(artifact_type) is not str
            or artifact_type != COMPARATOR_REPORT_ARTIFACT
        ):
            raise ValueError("unsupported comparator report artifact type")
        raw_rows = payload.get("rows", ())
        if raw_rows is None:
            raw_rows = ()
        if type(raw_rows) not in (list, tuple) or len(raw_rows) > _MAX_REPORT_ROWS:
            raise ValueError("comparator report rows must be a bounded list or tuple")
        schema_version = payload.get("schema_version", COMPARATOR_REPORT_SCHEMA_VERSION)
        if type(schema_version) is not int or (
            schema_version != COMPARATOR_REPORT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported comparator report schema version")
        raw_fingerprint = payload.get("environment_fingerprint")
        safe_fingerprint = raw_fingerprint if type(raw_fingerprint) is str else ""
        raw_failure_summary = payload.get("failure_summary")
        safe_failure_summary: Mapping[str, Any] = (
            raw_failure_summary if type(raw_failure_summary) is dict else {}
        )
        return cls(
            suite=_string_or_fallback(payload.get("suite"), "comparator"),
            model_name=_string_or_fallback(payload.get("model_name"), "OpenMed"),
            device=_string_or_fallback(payload.get("device"), "unknown"),
            fixture_count=_safe_count(payload.get("fixture_count")),
            rows=tuple(_row_from_mapping(row) for row in raw_rows),
            environment_fingerprint=safe_fingerprint,
            failure_summary=safe_failure_summary,
            generated_at=payload.get("generated_at"),
            schema_version=schema_version,
        )

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=_safe_indent(indent),
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""

        return _write_report_text(path, self.to_json(indent=indent) + "\n")

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

        return _write_report_text(path, self.to_markdown())

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

    if type(result) is ComparatorReport:
        if (
            environment is None
            and environment_fingerprint is None
            and generated_at is None
        ):
            return result
        source: Any = result.to_dict()
    elif type(result) in (ComparatorMatrixReport, dict):
        source = result
    else:
        raise TypeError("comparator result must be a matrix report or plain dictionary")
    if environment is not None and type(environment) is not dict:
        raise TypeError("environment must be a plain dictionary")
    if environment_fingerprint is not None and type(environment_fingerprint) is not str:
        raise TypeError("environment fingerprint must be a string")

    raw_rows = _source_rows(source)
    rows: list[ComparatorReportRow] = []
    row_failure_systems: set[str] = set()
    failure_categories: Counter[str] = Counter()
    for raw_row in raw_rows:
        row, events = _project_row(raw_row)
        rows.append(row)
        if events:
            row_failure_systems.add(row.system)
            _update_failure_counts(failure_categories, events)

    _update_failure_counts(failure_categories, _top_level_failure_events(source))
    fixture_count = _safe_count(_value(source, "fixture_count", 0))
    if fixture_count == 0 and rows:
        fixture_count = max(row.fixture_count for row in rows)

    source_metadata = _value(source, "metadata", {})
    if type(source_metadata) is not dict:
        source_metadata = {}
    if environment is None:
        source_environment = _value(source, "environment", _MISSING)
        if type(source_environment) is not dict:
            source_environment = source_metadata.get("environment")
        environment = (
            source_environment
            if type(source_environment) is dict
            else default_environment()
        )

    fingerprint = environment_fingerprint
    if fingerprint is None:
        source_fingerprint = _value(source, "environment_fingerprint", _MISSING)
        if source_fingerprint is _MISSING:
            source_fingerprint = source_metadata.get("environment_fingerprint")
        if type(source_fingerprint) is str:
            fingerprint = source_fingerprint
    if fingerprint is None or not _DIGEST_RE.fullmatch(fingerprint):
        fingerprint = fingerprint_environment(environment)

    safe_generated_at = generated_at
    if safe_generated_at is None:
        safe_generated_at = _value(source, "generated_at", None)

    failure_summary = {
        "by_category": dict(sorted(failure_categories.items())),
        "systems_affected": len(row_failure_systems),
        "total": _bounded_total(failure_categories.values()),
    }
    return ComparatorReport(
        suite=_value(source, "suite", "comparator"),
        model_name=_value(source, "model_name", "OpenMed"),
        device=_value(source, "device", "unknown"),
        fixture_count=fixture_count,
        rows=tuple(rows),
        environment_fingerprint=fingerprint,
        failure_summary=failure_summary,
        generated_at=safe_generated_at,
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

    output_path = _output_path(path)
    selected_format = "json" if output_path.suffix.lower() == ".json" else "markdown"
    if format is not None:
        if type(format) is not str:
            raise TypeError("comparator report format must be a string")
        selected_format = format
    content = render_comparator_report(
        result,
        format=selected_format,
        environment=environment,
        environment_fingerprint=environment_fingerprint,
        generated_at=generated_at,
        indent=indent,
    )
    return _write_report_text(
        output_path,
        content + ("" if content.endswith("\n") else "\n"),
    )


def _project_row(row: Any) -> tuple[ComparatorReportRow, Counter[str]]:
    status = _safe_status(_value(row, "status", "other"))
    reason = _value(row, "reason", None)
    has_reason = reason is not None and (
        type(reason) is not str or bool(reason.strip())
    )
    events: Counter[str] = Counter()
    if status != STATUS_SCORED or has_reason:
        _add_failure_count(events, _failure_category(status, reason), 1)
    _update_failure_counts(events, _row_failure_events(row))
    metric_counts: dict[str, dict[str, int | float]] = {}
    for metric in METRIC_NAMES:
        counts = _metric_counts(row, metric)
        if counts:
            metric_counts[metric] = counts
    projected = ComparatorReportRow(
        system=_value(row, "system", _value(row, "name", "system")),
        status=status,
        fixture_count=_safe_count(_value(row, "fixture_count", 0)),
        metrics={metric: _metric_value(row, metric) for metric in METRIC_NAMES},
        metric_counts=metric_counts,
        failure_count=_bounded_total(events.values()),
    )
    return projected, events


def _source_rows(source: Any) -> tuple[Any, ...]:
    raw_rows = _value(source, "rows", _MISSING)
    if raw_rows is _MISSING:
        raw_rows = _value(source, "systems", ())
    if type(raw_rows) is dict:
        if len(raw_rows) > _MAX_REPORT_ROWS:
            raise ValueError("comparator report has too many rows")
        rows: list[Any] = []
        for name, value in raw_rows.items():
            if type(value) is not dict:
                raise ValueError("comparator rows must be plain dictionaries")
            row = dict(value)
            row.setdefault("system", name)
            rows.append(row)
        return tuple(rows)
    if type(raw_rows) in (list, tuple):
        if len(raw_rows) > _MAX_REPORT_ROWS:
            raise ValueError("comparator report has too many rows")
        if any(type(row) not in (dict, ComparatorMatrixRow) for row in raw_rows):
            raise ValueError("comparator rows have an unsupported type")
        return tuple(raw_rows)
    raise ValueError("comparator report rows must be a list, tuple, or dictionary")


def _metric_value(row: Any, metric: str) -> float | None:
    for path in _METRIC_PATHS[metric]:
        value = _path_value(row, path)
        if value is not _MISSING:
            if type(value) is dict and "rate" in value:
                value = value["rate"]
            if type(value) is dict and "f1" in value:
                value = value["f1"]
            result = _safe_metric(value)
            if result is not None:
                return result
    return None


def _metric_counts(row: Any, metric: str) -> dict[str, int | float]:
    for path in _METRIC_PATHS[metric]:
        parent_path = path[:-1]
        candidate = _path_value(row, parent_path) if parent_path else _MISSING
        if type(candidate) is dict:
            counts = _safe_count_mapping(candidate, _METRIC_COUNT_FIELDS[metric])
            if counts:
                return counts
    return {}


def _row_failure_events(row: Any) -> Counter[str]:
    events: Counter[str] = Counter()
    for path in (
        ("metrics",),
        ("benchmark_report", "metrics"),
    ):
        value = _path_value(row, path)
        if value is not _MISSING:
            _update_failure_counts(events, _failure_events_from_mapping(value))
    return events


def _top_level_failure_events(source: Any) -> Counter[str]:
    events: Counter[str] = Counter()
    for key in ("failures", "errors", "exceptions"):
        value = _value(source, key, _MISSING)
        if value is not _MISSING:
            _update_failure_counts(
                events,
                _failure_events_from_mapping({key: value}),
            )
    for key in ("failure_count", "error_count"):
        value = _value(source, key, _MISSING)
        if value is not _MISSING:
            _add_failure_count(events, "other", _safe_count(value))
    return events


def _failure_events_from_mapping(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _budget: list[int] | None = None,
    _depth: int = 0,
) -> Counter[str]:
    if _seen is None:
        _seen = set()
    if _budget is None:
        _budget = [_MAX_FAILURE_ITEMS]
    events: Counter[str] = Counter()
    if _depth > _MAX_FAILURE_DEPTH or _budget[0] <= 0:
        _add_failure_count(events, "other", 1)
        return events
    if id(value) in _seen:
        return events
    if type(value) is dict:
        _seen.add(id(value))
        for key, child in value.items():
            if _budget[0] <= 0:
                _add_failure_count(events, "other", 1)
                break
            _budget[0] -= 1
            normalized_key = (
                key.strip().lower().replace("-", "_")
                if type(key) is str and len(key) <= 128
                else ""
            )
            if normalized_key in {"failures", "errors", "exceptions"}:
                if type(child) in (list, tuple):
                    available = min(len(child), _budget[0])
                    for item in child[:available]:
                        _budget[0] -= 1
                        _add_failure_count(
                            events,
                            _failure_category_from_item(item),
                            1,
                        )
                    _add_failure_count(events, "other", len(child) - available)
                elif type(child) is dict:
                    _update_failure_counts(
                        events,
                        _failure_events_from_mapping(
                            child,
                            _seen=_seen,
                            _budget=_budget,
                            _depth=_depth + 1,
                        ),
                    )
                else:
                    _add_failure_count(
                        events,
                        _failure_category("failed", child),
                        1,
                    )
            elif normalized_key in {"failure_count", "error_count"}:
                _add_failure_count(events, "other", _safe_count(child))
            elif type(child) is dict:
                _update_failure_counts(
                    events,
                    _failure_events_from_mapping(
                        child,
                        _seen=_seen,
                        _budget=_budget,
                        _depth=_depth + 1,
                    ),
                )
        return events
    return events


def _failure_category_from_item(value: Any) -> str:
    if type(value) is str:
        return _failure_category("failed", value)
    if type(value) is dict:
        for key in ("category", "reason", "message", "error", "exception"):
            candidate = value.get(key)
            if type(candidate) is str:
                return _failure_category("failed", candidate)
    return "execution"


def _add_failure_count(
    target: Counter[str],
    category: str,
    count: int,
) -> None:
    safe_category = category if category in _FAILURE_CATEGORIES else "other"
    safe_count = _safe_count(count)
    if safe_count == 0:
        return
    target[safe_category] = min(
        _MAX_COUNT,
        target[safe_category] + safe_count,
    )


def _update_failure_counts(
    target: Counter[str],
    source: Mapping[str, int],
) -> None:
    for category, count in source.items():
        _add_failure_count(target, category, count)


def _bounded_total(values: Any) -> int:
    total = 0
    for value in values:
        total = min(_MAX_COUNT, total + _safe_count(value))
    return total


def _failure_category(status: Any, reason: Any) -> str:
    normalized_status = _safe_status(status)
    if normalized_status == STATUS_NOT_AVAILABLE:
        return "not_available"
    text = reason[:512].lower() if type(reason) is str else ""
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
    if type(value) is ComparatorReportRow:
        return value
    if type(value) is not dict:
        raise ValueError("comparator report rows must be plain dictionaries")
    system = value.get("system", _MISSING)
    if system is _MISSING or system is None:
        system = value.get("name", "system")
    status = value.get("status", "other")
    if status is None:
        status = "other"
    raw_metrics = value.get("metrics")
    metrics: Mapping[str, float | None] = (
        raw_metrics if type(raw_metrics) is dict else {}
    )
    raw_metric_counts = value.get("metric_counts")
    metric_counts: Mapping[str, Mapping[str, int | float]] = (
        raw_metric_counts if type(raw_metric_counts) is dict else {}
    )
    return ComparatorReportRow(
        system=_string_or_fallback(system, "system"),
        status=_string_or_fallback(status, "other"),
        fixture_count=_safe_count(value.get("fixture_count")),
        metrics=metrics,
        metric_counts=metric_counts,
        failure_count=_safe_count(value.get("failure_count")),
    )


def _safe_failure_summary(
    value: Mapping[str, Any],
    *,
    rows: tuple[ComparatorReportRow, ...],
) -> dict[str, Any]:
    raw_categories = value.get("by_category", {}) if type(value) is dict else {}
    categories: dict[str, int] = {}
    if type(raw_categories) is dict:
        for key, count in raw_categories.items():
            category = (
                key if type(key) is str and key in _FAILURE_CATEGORIES else "other"
            )
            categories[category] = min(
                _MAX_COUNT,
                categories.get(category, 0) + _safe_count(count),
            )
    category_total = _bounded_total(categories.values())
    row_total = _bounded_total(row.failure_count for row in rows)
    declared_total = _safe_count(value.get("total")) if type(value) is dict else 0
    target_total = max(category_total, row_total, declared_total)
    if category_total < target_total:
        categories["other"] = min(
            _MAX_COUNT,
            categories.get("other", 0) + target_total - category_total,
        )
    systems_with_failures = len({row.system for row in rows if row.failure_count > 0})
    declared_systems = (
        _safe_count(value.get("systems_affected")) if type(value) is dict else 0
    )
    return {
        "by_category": dict(sorted(categories.items())),
        "systems_affected": min(
            len(rows),
            max(systems_with_failures, declared_systems),
        ),
        "total": _bounded_total(categories.values()),
    }


def _safe_metric_counts(
    value: Mapping[str, Mapping[str, int | float]],
) -> dict[str, dict[str, int | float]]:
    if type(value) is not dict:
        return {}
    result: dict[str, dict[str, int | float]] = {}
    for metric, counts in value.items():
        if type(metric) is not str or metric not in METRIC_NAMES:
            continue
        if type(counts) is not dict:
            continue
        safe_counts = _safe_count_mapping(counts, _METRIC_COUNT_FIELDS[metric])
        if safe_counts:
            result[metric] = safe_counts
    return result


def _safe_count_mapping(
    value: Mapping[str, Any],
    allowed: frozenset[str],
) -> dict[str, int | float]:
    if type(value) is not dict:
        return {}
    result: dict[str, int | float] = {}
    for key in sorted(allowed):
        if key not in value:
            continue
        number = _finite_float(value[key])
        if number is not None and 0 <= number <= _MAX_COUNT:
            result[key] = int(number) if number.is_integer() else number
    return result


def _safe_status(value: Any) -> str:
    text = value.strip().lower().replace(" ", "_") if type(value) is str else "other"
    if text in {"unavailable", "skipped"}:
        text = STATUS_NOT_AVAILABLE
    if text in {"error", "exception"}:
        text = "failed"
    return text if text in _SAFE_STATUSES else "other"


def _string_or_fallback(value: Any, fallback: str) -> str:
    return value if type(value) is str else fallback


def _safe_identifier(value: Any, fallback: str) -> str:
    if type(value) is not str:
        return fallback
    text = value.strip()
    if text in _PUBLIC_IDENTIFIERS or _DIGEST_RE.fullmatch(text):
        return text
    if not text:
        return fallback
    hash_input = text if len(text) <= _MAX_ENVIRONMENT_TEXT else text[:256]
    return stable_hash(
        {
            "identifier": hash_input,
            "original_length": len(text),
        }
    )


def _safe_timestamp(value: Any) -> str | None:
    if type(value) is not str:
        return None
    text = value.strip()
    return text if _TIMESTAMP_RE.fullmatch(text) else None


def _safe_count(value: Any) -> int:
    if type(value) is bool:
        return 0
    if type(value) is int:
        number = value
    elif type(value) is float and math.isfinite(value) and value.is_integer():
        number = int(value)
    elif type(value) in (list, tuple, set, frozenset):
        number = len(value)
    else:
        return 0
    return min(_MAX_COUNT, max(0, number))


def _finite_float(value: Any) -> float | None:
    if type(value) not in (int, float) or type(value) is bool:
        return None
    if type(value) is int and abs(value) > _MAX_COUNT:
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _safe_metric(value: Any) -> float | None:
    number = _finite_float(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _safe_indent(value: Any) -> int:
    if type(value) is not int or type(value) is bool:
        raise TypeError("JSON indent must be an integer")
    if not 0 <= value <= _MAX_JSON_INDENT:
        raise ValueError(f"JSON indent must be between 0 and {_MAX_JSON_INDENT}")
    return value


def _value(source: Any, key: str, default: Any = _MISSING) -> Any:
    if type(source) is dict:
        return source.get(key, default)
    if type(source) in (
        BenchmarkReport,
        ComparatorMatrixReport,
        ComparatorMatrixRow,
        ComparatorReport,
        ComparatorReportRow,
    ):
        return object.__getattribute__(source, "__dict__").get(key, default)
    return default


def _path_value(source: Any, path: tuple[str, ...]) -> Any:
    current = source
    for key in path:
        current = _value(current, key, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current


def _hash_safe_value(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    if _seen is None:
        _seen = set()
    if _depth > _MAX_ENVIRONMENT_DEPTH:
        raise ValueError("environment nesting exceeds the supported limit")
    if type(value) is dict:
        if id(value) in _seen:
            return {"cycle": True}
        if len(value) > _MAX_ENVIRONMENT_ITEMS:
            raise ValueError("environment mapping exceeds the supported limit")
        _seen.add(id(value))
        mapping_result: dict[str, Any] = {}
        keys = list(value)
        if any(
            type(key) is not str or len(key) > _MAX_ENVIRONMENT_TEXT for key in keys
        ):
            raise ValueError("environment keys must be bounded strings")
        for key in sorted(keys):
            mapping_result[key] = _hash_safe_value(
                value[key],
                _depth=_depth + 1,
                _seen=_seen,
            )
        _seen.remove(id(value))
        return mapping_result
    if type(value) in (list, tuple):
        if id(value) in _seen:
            return ["cycle"]
        if len(value) > _MAX_ENVIRONMENT_ITEMS:
            raise ValueError("environment sequence exceeds the supported limit")
        _seen.add(id(value))
        sequence_result = [
            _hash_safe_value(item, _depth=_depth + 1, _seen=_seen) for item in value
        ]
        _seen.remove(id(value))
        return sequence_result
    if type(value) in (str, int, float, bool) or value is None:
        if type(value) is str and len(value) > _MAX_ENVIRONMENT_TEXT:
            raise ValueError("environment text exceeds the supported limit")
        if type(value) is int and abs(value) > _MAX_COUNT:
            raise ValueError("environment integer exceeds the supported limit")
        if type(value) is float and not math.isfinite(value):
            return str(value)
        return value
    return type(value).__name__


def _output_path(value: Any) -> Path:
    if type(value) is str:
        try:
            return Path(value)
        except (OSError, ValueError):
            raise ValueError("invalid comparator report output path") from None
    if isinstance(value, Path):
        return value
    raise TypeError("comparator report output path must be a string or Path")


def _write_report_text(path: str | Path, content: str) -> Path:
    output_path = _output_path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    except (OSError, ValueError):
        raise OSError("failed to write comparator report") from None
    return output_path


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _render_report(report: ComparatorReport, *, format: str, indent: int = 2) -> str:
    if type(format) is not str:
        raise TypeError("comparator report format must be a string")
    normalized = format.strip().lower()
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
