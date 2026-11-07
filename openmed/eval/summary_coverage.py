"""Deterministic, privacy-safe coverage for citation-anchored summaries.

The metric scores whether structured summary claims cite the source-fact
evidence supplied by a caller.  It deliberately matches opaque fact
identifiers or source offsets, never fact values or summary text.  Reports
therefore contain counts and reason codes only, even when the input records
carry ``value``, ``text``, or ``claim`` fields.

Source facts are the evidence set.  A source record needs either a stable
``id``/``fact_id`` (preferred) or a non-empty ``evidence``/span offset.  A
summary citation can reference that identifier with ``source_fact_id`` or
``fact_id``, or point to the same ``start``/``end`` source span.  A summary
record without a valid citation is an unsupported fact.  A source fact that
is never cited is an omission.

An empty or malformed evidence set never receives a vacuous perfect score:
the result is marked unavailable, ``recall`` is ``0.0``, and the report fails
closed.  This module is entirely local and has no model, filesystem, or
network side effects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUMMARY_FACT_COVERAGE = "summary_fact_coverage"
SUMMARY_FACT_COVERAGE_SCHEMA_VERSION = 1

REASON_MISSING_SOURCE_EVIDENCE = "missing_source_evidence"
REASON_INVALID_SOURCE_EVIDENCE = "invalid_source_evidence"
REASON_DUPLICATE_SOURCE_FACT = "duplicate_source_fact"
REASON_MISSING_SUMMARY_CITATION = "missing_summary_citation"
REASON_UNKNOWN_CITATION = "unknown_citation"
REASON_UNSUPPORTED_SUMMARY_FACT = "unsupported_summary_fact"
REASON_OMITTED_SOURCE_FACT = "omitted_source_fact"

_SOURCE_ID_KEYS = (
    "id",
    "fact_id",
    "source_fact_id",
    "source_id",
    "citation_id",
)
_SUMMARY_ID_KEYS = (
    "source_fact_id",
    "fact_id",
    "source_id",
    "citation_id",
    "evidence_id",
    "ref_id",
    "source_ref",
    "source_reference",
)
_NESTED_CITATION_KEYS = (
    "citation",
    "citations",
    "evidence",
    "source_evidence",
    "source_span",
    "supporting_span",
    "references",
    "source_refs",
)
_SPAN_KEYS = (
    "evidence",
    "source_evidence",
    "source_span",
    "supporting_span",
    "evidence_span",
    "span",
)
_CLAIM_KEYS = frozenset(
    {
        "claim",
        "summary",
        "summary_text",
        "text",
        "value",
        "surface",
        "surface_form",
    }
)
_RECORD_HINT_KEYS = frozenset(
    set(_SOURCE_ID_KEYS)
    | set(_SUMMARY_ID_KEYS)
    | set(_NESTED_CITATION_KEYS)
    | set(_SPAN_KEYS)
    | set(_CLAIM_KEYS)
    | {"start", "end"}
)

Reference = tuple[object, ...]


@dataclass(frozen=True)
class SummaryCoverageMetrics:
    """Aggregate citation coverage metrics with no raw-value diagnostics."""

    recall: float
    source_fact_count: int
    summary_fact_count: int
    cited_fact_count: int
    omission_count: int
    unsupported_fact_count: int
    invalid_citation_count: int
    missing_citation_count: int
    invalid_source_fact_count: int
    duplicate_source_fact_count: int
    source_evidence_count: int
    source_evidence_available: bool
    fail_closed: bool
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    schema_version: int = SUMMARY_FACT_COVERAGE_SCHEMA_VERSION

    @property
    def omissions(self) -> int:
        """Return the number of source facts not cited by the summary."""

        return self.omission_count

    @property
    def unsupported_facts(self) -> int:
        """Return the number of summary claims without valid support."""

        return self.unsupported_fact_count

    @property
    def coverage_available(self) -> bool:
        """Return whether the recall score is backed by valid evidence."""

        return self.source_evidence_available and not self.fail_closed

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic aggregate evidence without source values."""

        return {
            "schema_version": self.schema_version,
            "recall": float(self.recall),
            "source_fact_count": int(self.source_fact_count),
            "summary_fact_count": int(self.summary_fact_count),
            "cited_fact_count": int(self.cited_fact_count),
            "omission_count": int(self.omission_count),
            "omissions": int(self.omission_count),
            "unsupported_fact_count": int(self.unsupported_fact_count),
            "unsupported_facts": int(self.unsupported_fact_count),
            "invalid_citation_count": int(self.invalid_citation_count),
            "missing_citation_count": int(self.missing_citation_count),
            "invalid_source_fact_count": int(self.invalid_source_fact_count),
            "duplicate_source_fact_count": int(self.duplicate_source_fact_count),
            "source_evidence_count": int(self.source_evidence_count),
            "source_evidence_available": bool(self.source_evidence_available),
            "fail_closed": bool(self.fail_closed),
            "passed": bool(self.passed),
            "failure_reasons": list(self.failure_reasons),
        }

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized metric."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class SummaryCoverageReport:
    """Serializable summary fact-coverage report."""

    coverage: SummaryCoverageMetrics
    fixture_count: int = 1
    suite: str = SUMMARY_FACT_COVERAGE
    synthetic: bool = True

    @property
    def metrics(self) -> dict[str, dict[str, Any]]:
        """Return the metric under its stable report key."""

        return {SUMMARY_FACT_COVERAGE: self.coverage.to_dict()}

    @property
    def recall(self) -> float:
        """Return citation recall for the report."""

        return self.coverage.recall

    @property
    def passed(self) -> bool:
        """Return the strict all-source-facts-cited verdict."""

        return self.coverage.passed

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-value-free report mapping."""

        return {
            "suite": self.suite,
            "schema_version": SUMMARY_FACT_COVERAGE_SCHEMA_VERSION,
            "fixture_count": int(self.fixture_count),
            "synthetic": bool(self.synthetic),
            "passed": bool(self.coverage.passed),
            "metrics": self.metrics,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report deterministically as JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON evidence to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic aggregate-only Markdown report."""

        metric = self.coverage
        reasons = ", ".join(metric.failure_reasons) or "none"
        lines = [
            "# Summary Fact Coverage",
            "",
            "Aggregate citation evidence only; source and summary values are "
            "never rendered.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Recall | {metric.recall:.6f} |",
            f"| Source facts | {metric.source_fact_count} |",
            f"| Summary facts | {metric.summary_fact_count} |",
            f"| Cited facts | {metric.cited_fact_count} |",
            f"| Omissions | {metric.omission_count} |",
            f"| Unsupported facts | {metric.unsupported_fact_count} |",
            f"| Invalid citations | {metric.invalid_citation_count} |",
            f"| Evidence available | {metric.source_evidence_available} |",
            f"| Fail closed | {metric.fail_closed} |",
            f"| Verdict | {'pass' if metric.passed else 'fail'} |",
            f"| Failure reasons | `{reasons}` |",
        ]
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown evidence to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized report."""

        return self.to_dict()[key]


def compute_summary_fact_coverage(
    source_facts: Iterable[Any] | Mapping[Any, Any] | None = None,
    summary_citations: Iterable[Any] | Mapping[Any, Any] | None = None,
    *,
    source_evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
) -> SummaryCoverageMetrics:
    """Compute citation recall, omission, and unsupported-fact counts.

    Args:
        source_facts: The structured source evidence set. Records may contain
            ``id``/``fact_id`` and optional span evidence. Values and text are
            ignored by the matcher and never appear in the result.
        summary_citations: One item per summary claim. Items may be opaque
            source-fact identifiers, citation mappings, or mappings containing
            a ``citations`` collection. A claim with no valid citation is
            unsupported.
        source_evidence: Optional separate evidence collection. When supplied,
            it must be non-empty in addition to ``source_facts``; an empty
            collection fails closed. This lets callers distinguish a fact list
            from the evidence bundle that authorizes scoring.

    Returns:
        :class:`SummaryCoverageMetrics` containing aggregate counts only. A
        missing or malformed source evidence set returns ``recall=0.0`` and
        ``fail_closed=True`` rather than a vacuous perfect score.
    """

    source_items = _coerce_items(source_facts, "source_facts")
    summary_items = _coerce_items(summary_citations, "summary_citations")
    evidence_items = (
        _coerce_items(source_evidence, "source_evidence")
        if source_evidence is not None
        else source_items
    )

    failure_reasons: set[str] = set()
    if not source_items or not evidence_items:
        failure_reasons.add(REASON_MISSING_SOURCE_EVIDENCE)

    source_lookup: dict[Reference, Reference] = {}
    source_keys: set[Reference] = set()
    invalid_source_count = 0
    duplicate_source_count = 0

    for item in source_items:
        references = _source_references(item)
        if not references:
            invalid_source_count += 1
            continue
        if any(reference in source_lookup for reference in references):
            duplicate_source_count += 1
            continue
        canonical = references[0]
        source_keys.add(canonical)
        for reference in references:
            source_lookup[reference] = canonical

    if invalid_source_count:
        failure_reasons.add(REASON_INVALID_SOURCE_EVIDENCE)
    if duplicate_source_count:
        failure_reasons.add(REASON_DUPLICATE_SOURCE_FACT)

    evidence_available = bool(source_items and evidence_items) and not (
        invalid_source_count or duplicate_source_count
    )

    cited_source_keys: set[Reference] = set()
    unsupported_count = 0
    invalid_citation_count = 0
    missing_citation_count = 0
    for item in summary_items:
        references = _summary_references(item)
        valid_references = {
            source_lookup[reference]
            for reference in references
            if reference in source_lookup
        }
        if source_lookup:
            invalid_citation_count += sum(
                1 for reference in references if reference not in source_lookup
            )
        if not references:
            missing_citation_count += 1
        if not valid_references:
            unsupported_count += 1
        cited_source_keys.update(valid_references)

    omission_count = len(source_keys - cited_source_keys)
    if missing_citation_count:
        failure_reasons.add(REASON_MISSING_SUMMARY_CITATION)
    if invalid_citation_count:
        failure_reasons.add(REASON_UNKNOWN_CITATION)
    if unsupported_count:
        failure_reasons.add(REASON_UNSUPPORTED_SUMMARY_FACT)
    if omission_count:
        failure_reasons.add(REASON_OMITTED_SOURCE_FACT)

    fail_closed = not evidence_available
    recall = (
        len(cited_source_keys) / len(source_keys)
        if evidence_available and source_keys
        else 0.0
    )
    passed = bool(
        evidence_available
        and not omission_count
        and not unsupported_count
        and not invalid_citation_count
        and not missing_citation_count
    )
    return SummaryCoverageMetrics(
        recall=recall,
        source_fact_count=len(source_keys),
        summary_fact_count=len(summary_items),
        cited_fact_count=len(cited_source_keys),
        omission_count=omission_count,
        unsupported_fact_count=unsupported_count,
        invalid_citation_count=invalid_citation_count,
        missing_citation_count=missing_citation_count,
        invalid_source_fact_count=invalid_source_count,
        duplicate_source_fact_count=duplicate_source_count,
        source_evidence_count=len(evidence_items),
        source_evidence_available=evidence_available,
        fail_closed=fail_closed,
        passed=passed,
        failure_reasons=tuple(sorted(failure_reasons)),
    )


def summary_fact_coverage(
    source_facts: Iterable[Any] | Mapping[Any, Any] | None = None,
    summary_citations: Iterable[Any] | Mapping[Any, Any] | None = None,
    *,
    source_evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
) -> SummaryCoverageMetrics:
    """Convenience alias for :func:`compute_summary_fact_coverage`."""

    return compute_summary_fact_coverage(
        source_facts,
        summary_citations,
        source_evidence=source_evidence,
    )


def run_summary_coverage(
    source_facts: Iterable[Any] | Mapping[Any, Any] | None = None,
    summary_citations: Iterable[Any] | Mapping[Any, Any] | None = None,
    *,
    source_evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    fixture_count: int = 1,
) -> SummaryCoverageReport:
    """Build a serializable summary coverage report from local inputs."""

    if type(fixture_count) is not int or fixture_count < 0:
        raise ValueError("fixture_count must be a non-negative integer")
    coverage = compute_summary_fact_coverage(
        source_facts,
        summary_citations,
        source_evidence=source_evidence,
    )
    return SummaryCoverageReport(coverage=coverage, fixture_count=fixture_count)


def build_summary_coverage_report(
    source_facts: Iterable[Any] | Mapping[Any, Any] | None = None,
    summary_citations: Iterable[Any] | Mapping[Any, Any] | None = None,
    *,
    source_evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    fixture_count: int = 1,
) -> SummaryCoverageReport:
    """Return the report form of :func:`compute_summary_fact_coverage`."""

    return run_summary_coverage(
        source_facts,
        summary_citations,
        source_evidence=source_evidence,
        fixture_count=fixture_count,
    )


def assert_summary_coverage_gate(
    source_facts: Iterable[Any] | Mapping[Any, Any] | None = None,
    summary_citations: Iterable[Any] | Mapping[Any, Any] | None = None,
    *,
    source_evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    fixture_count: int = 1,
) -> SummaryCoverageReport:
    """Return a passing report or raise with aggregate-only diagnostics."""

    report = run_summary_coverage(
        source_facts,
        summary_citations,
        source_evidence=source_evidence,
        fixture_count=fixture_count,
    )
    if not report.passed:
        reasons = ", ".join(report.coverage.failure_reasons) or "coverage_failed"
        raise AssertionError(f"summary fact-coverage gate failed: {reasons}")
    return report


def summary_coverage_metadata() -> dict[str, Any]:
    """Return stable, raw-value-free metadata for the metric."""

    return {
        "suite": SUMMARY_FACT_COVERAGE,
        "schema_version": SUMMARY_FACT_COVERAGE_SCHEMA_VERSION,
        "synthetic": True,
        "matching": "opaque_fact_ids_or_source_offsets",
        "metrics": ["recall", "unsupported_fact_count", "omission_count"],
        "fail_closed_on_missing_source_evidence": True,
    }


def _coerce_items(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if _looks_like_record(value):
            return [value]
        records: list[Any] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                record = dict(item)
                record.setdefault("id", key)
            else:
                record = {"id": key, "evidence": item}
            records.append(record)
        return records
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable of structured records or identifiers"
        ) from exc


def _looks_like_record(value: Mapping[Any, Any]) -> bool:
    return any(str(key) in _RECORD_HINT_KEYS for key in value)


def _as_mapping(value: Any) -> Mapping[Any, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    data = getattr(value, "__dict__", None)
    return data if isinstance(data, Mapping) else None


def _identifier(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        result = str(value).strip()
        return result or None
    return None


def _mapping_identifier(value: Mapping[Any, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in value:
            result = _identifier(value.get(key))
            if result is not None:
                return result
    return None


def _span(value: Any, *, depth: int = 0) -> tuple[int, int] | None:
    if depth > 4:
        return None
    mapping = _as_mapping(value)
    if mapping is not None:
        start = mapping.get("start", mapping.get("start_char"))
        end = mapping.get("end", mapping.get("end_char"))
        if type(start) is int and type(end) is int and 0 <= start < end:
            return start, end
        for key in _SPAN_KEYS:
            if key in mapping:
                nested = _span(mapping[key], depth=depth + 1)
                if nested is not None:
                    return nested
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2 and all(type(item) is int for item in value):
            start, end = value
            if 0 <= start < end:
                return start, end
    return None


def _source_references(value: Any) -> list[Reference]:
    mapping = _as_mapping(value)
    references: list[Reference] = []
    if mapping is not None:
        identifier = _mapping_identifier(mapping, _SOURCE_ID_KEYS)
        if identifier is not None:
            references.append(("id", identifier))
        source_span = _span(mapping)
        if source_span is not None:
            references.append(("span", source_span[0], source_span[1]))
        return references
    identifier = _identifier(value)
    if identifier is not None:
        return [("id", identifier)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for part in value:
            identifier = _identifier(part)
            if identifier is not None:
                references.append(("id", identifier))
                break
        source_span = _span(value)
        if source_span is not None:
            references.append(("span", source_span[0], source_span[1]))
    return references


def _summary_references(value: Any, *, depth: int = 0) -> list[Reference]:
    if depth > 5 or value is None:
        return []
    mapping = _as_mapping(value)
    if mapping is not None:
        references: list[Reference] = []
        for key in _SUMMARY_ID_KEYS:
            if key in mapping:
                identifier = _identifier(mapping.get(key))
                if identifier is not None:
                    references.append(("id", identifier))
        source_span = _span(mapping)
        if source_span is not None:
            references.append(("span", source_span[0], source_span[1]))
        for key in _NESTED_CITATION_KEYS:
            if key in mapping:
                references.extend(_summary_references(mapping[key], depth=depth + 1))
        if not references and "id" in mapping:
            if not _CLAIM_KEYS.intersection(str(key) for key in mapping):
                identifier = _identifier(mapping.get("id"))
                if identifier is not None:
                    references.append(("id", identifier))
        return references
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        source_span = _span(value)
        if source_span is not None:
            return [("span", source_span[0], source_span[1])]
        references: list[Reference] = []
        for item in value:
            references.extend(_summary_references(item, depth=depth + 1))
        return references
    identifier = _identifier(value)
    return [("id", identifier)] if identifier is not None else []


__all__ = [
    "REASON_DUPLICATE_SOURCE_FACT",
    "REASON_INVALID_SOURCE_EVIDENCE",
    "REASON_MISSING_SOURCE_EVIDENCE",
    "REASON_MISSING_SUMMARY_CITATION",
    "REASON_OMITTED_SOURCE_FACT",
    "REASON_UNKNOWN_CITATION",
    "REASON_UNSUPPORTED_SUMMARY_FACT",
    "SUMMARY_FACT_COVERAGE",
    "SUMMARY_FACT_COVERAGE_SCHEMA_VERSION",
    "SummaryCoverageMetrics",
    "SummaryCoverageReport",
    "assert_summary_coverage_gate",
    "build_summary_coverage_report",
    "compute_summary_fact_coverage",
    "run_summary_coverage",
    "summary_fact_coverage",
    "summary_coverage_metadata",
]
