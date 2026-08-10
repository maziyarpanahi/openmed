"""Deterministic citation-consistency checks for guarded clinical summaries.

The checker accepts structured summary claims and a caller-supplied evidence
set.  Claims can cite opaque evidence identifiers or exact source spans.  It
never compares claim or evidence values, and its metrics and reports contain
only aggregate counts, booleans, and fixed reason codes.  This keeps the
review artifact useful without rendering summary text or sensitive source
values.

The module is deliberately local-only.  It performs no model inference,
filesystem discovery, telemetry, or network call.  It is a review aid, not a
clinical-quality certification or a clinical decision guarantee.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUMMARY_CITATION_CONSISTENCY = "summary_citation_consistency"
SUMMARY_CITATION_SCHEMA_VERSION = 1

REASON_DUPLICATE_CITATION = "duplicate_citation"
REASON_DUPLICATE_SOURCE_EVIDENCE = "duplicate_source_evidence"
REASON_INVALID_CITATION = "invalid_citation"
REASON_INVALID_SOURCE_EVIDENCE = "invalid_source_evidence"
REASON_MISSING_CITATION = "missing_citation"
REASON_MISSING_SOURCE_EVIDENCE = "missing_source_evidence"
REASON_MISSING_SOURCE_RECORD = "missing_source_record"
REASON_UNAVAILABLE_SPAN = "unavailable_span"
REASON_UNSUPPORTED_CLAIM = "unsupported_claim"

# Descriptive aliases make the reason vocabulary discoverable for callers that
# use "summary" terminology rather than the shorter internal names.
REASON_DUPLICATE_CITATIONS = REASON_DUPLICATE_CITATION
REASON_MISSING_EVIDENCE = REASON_MISSING_SOURCE_EVIDENCE
REASON_UNKNOWN_CITATION = REASON_MISSING_SOURCE_RECORD
REASON_UNSUPPORTED_SUMMARY_CLAIM = REASON_UNSUPPORTED_CLAIM

_SOURCE_ID_KEYS = (
    "id",
    "evidence_id",
    "source_id",
    "fact_id",
    "record_id",
)
_CITATION_ID_KEYS = (
    "evidence_id",
    "source_evidence_id",
    "source_id",
    "fact_id",
    "citation_id",
    "ref_id",
    "source_ref",
    "source_reference",
    "id",
)
_SPAN_CONTAINER_KEYS = (
    "span",
    "source_span",
    "evidence_span",
    "supporting_span",
    "source_evidence",
    "evidence",
    "offsets",
)
_CLAIM_CITATION_KEYS = (
    "citations",
    "citation",
    "references",
    "source_refs",
    "source_references",
    "evidence",
    "source_evidence",
)
_RECORD_HINT_KEYS = frozenset(
    set(_SOURCE_ID_KEYS)
    | set(_CITATION_ID_KEYS)
    | set(_SPAN_CONTAINER_KEYS)
    | set(_CLAIM_CITATION_KEYS)
    | {
        "claim",
        "summary",
        "summary_text",
        "text",
        "value",
        "surface",
        "start",
        "end",
        "start_char",
        "end_char",
    }
)
_DIRECT_SPAN_KEYS = frozenset({"start", "end", "start_char", "end_char"})

Reference = tuple[object, ...]


class SummaryCitationError(ValueError):
    """Raised when the citation-check input container is not usable."""


def _value(value: object, name: str) -> object | None:
    """Read a field without allowing source values into an error message."""

    if isinstance(value, Mapping):
        return value.get(name)
    try:
        return getattr(value, name, None)
    except Exception:
        raise SummaryCitationError("summary citation input is not accessible") from None


def _as_mapping(value: object) -> Mapping[Any, Any] | None:
    if isinstance(value, Mapping):
        return value
    try:
        to_dict = getattr(value, "to_dict", None)
    except Exception:
        raise SummaryCitationError("summary citation input is not accessible") from None
    if callable(to_dict):
        try:
            candidate = to_dict()
        except Exception:
            raise SummaryCitationError(
                "summary citation input is not accessible"
            ) from None
        if isinstance(candidate, Mapping):
            return candidate
    try:
        attributes = vars(value)
    except (TypeError, ValueError):
        return None
    return attributes if isinstance(attributes, Mapping) else None


def _identifier(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        normalized = str(value).strip()
        return normalized or None
    return None


def _span(value: object, *, depth: int = 0) -> tuple[int, int] | None:
    """Return an exact half-open span, without retaining any source text."""

    if depth > 4 or value is None:
        return None
    mapping = _as_mapping(value)
    if mapping is not None:
        start = mapping.get("start", mapping.get("start_char"))
        end = mapping.get("end", mapping.get("end_char"))
        if type(start) is int and type(end) is int and start >= 0 and end > start:
            return start, end
        for key in _SPAN_CONTAINER_KEYS:
            if key in mapping:
                nested = _span(mapping[key], depth=depth + 1)
                if nested is not None:
                    return nested
        return None

    start = _value(value, "start")
    end = _value(value, "end")
    if type(start) is int and type(end) is int and start >= 0 and end > start:
        return start, end
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2 and all(type(item) is int for item in value):
            start, end = value
            if start >= 0 and end > start:
                return start, end
    return None


def _mapping_identifier(value: object, keys: Sequence[str]) -> str | None:
    for key in keys:
        candidate = _identifier(_value(value, key))
        if candidate is not None:
            return candidate
    return None


def _source_references(value: object) -> tuple[Reference, ...]:
    mapping = _as_mapping(value)
    if mapping is not None:
        references: list[Reference] = []
        identifier = _mapping_identifier(mapping, _SOURCE_ID_KEYS)
        if identifier is not None:
            references.append(("id", identifier))
        source_span = _span(mapping)
        if source_span is not None:
            references.append(("span", source_span[0], source_span[1]))
        return tuple(dict.fromkeys(references))

    identifier = _identifier(value)
    if identifier is not None:
        return (("id", identifier),)
    source_span = _span(value)
    if source_span is not None:
        return (("span", source_span[0], source_span[1]),)
    return ()


def _citation_references(value: object, *, depth: int = 0) -> tuple[Reference, ...]:
    if depth > 5 or value is None:
        return ()
    mapping = _as_mapping(value)
    if mapping is not None:
        references: list[Reference] = []
        identifier = _mapping_identifier(mapping, _CITATION_ID_KEYS)
        if identifier is not None:
            references.append(("id", identifier))

        if _DIRECT_SPAN_KEYS.intersection(str(key) for key in mapping):
            direct_span = _span(mapping)
            if direct_span is not None:
                references.append(("span", direct_span[0], direct_span[1]))
        else:
            for key in _SPAN_CONTAINER_KEYS:
                if key in mapping:
                    nested_span = _span(mapping[key])
                    if nested_span is not None:
                        references.append(("span", nested_span[0], nested_span[1]))
                        break

        for key in _CLAIM_CITATION_KEYS:
            if key in mapping:
                references.extend(_citation_references(mapping[key], depth=depth + 1))
        return tuple(dict.fromkeys(references))

    identifier = _identifier(value)
    if identifier is not None:
        return (("id", identifier),)
    source_span = _span(value)
    if source_span is not None:
        return (("span", source_span[0], source_span[1]),)
    return ()


def _looks_like_record(value: Mapping[Any, Any]) -> bool:
    return bool(_RECORD_HINT_KEYS.intersection(str(key) for key in value))


def _coerce_records(value: object, name: str) -> list[object]:
    """Normalize sequence and keyed-record inputs without copying raw values."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        if _looks_like_record(value):
            return [value]
        records: list[object] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                record = dict(item)
                record.setdefault("id", key)
            elif name == "evidence":
                record = {"id": key, "value": item}
            else:
                record = {"id": key, "claim": item}
            records.append(record)
        return records
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        raise SummaryCitationError(
            f"{name} must be an iterable of structured records"
        ) from None


def _citation_items(value: object, *, depth: int = 0) -> list[object]:
    if depth > 5 or value is None:
        return []
    mapping = _as_mapping(value)
    if mapping is not None:
        direct = _citation_references(mapping)
        if direct:
            return [mapping]
        nested: list[object] = []
        for key in _CLAIM_CITATION_KEYS:
            if key in mapping:
                nested.extend(_citation_items(mapping[key], depth=depth + 1))
        return nested or [mapping]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if _span(value) is not None:
            return [value]
        return list(value)
    return [value]


def _claim_citations(claim: object) -> list[object]:
    mapping = _as_mapping(claim)
    if mapping is None:
        return []
    citations: list[object] = []
    for key in _CITATION_ID_KEYS:
        if key == "id":
            continue
        if key in mapping:
            citations.extend(_citation_items(mapping[key]))
    if _DIRECT_SPAN_KEYS.intersection(str(key) for key in mapping):
        citations.append(mapping)
    for key in _CLAIM_CITATION_KEYS:
        if key in mapping:
            citations.extend(_citation_items(mapping[key]))
    return citations


def _citation_key(
    references: tuple[Reference, ...], resolved_record: int | None
) -> tuple[object, ...] | None:
    if resolved_record is not None:
        return ("record", resolved_record)
    if references:
        return ("references", frozenset(references))
    return None


@dataclass(frozen=True)
class SummaryCitationMetrics:
    """Aggregate citation-consistency and abstention metrics.

    ``coverage`` is the fraction of claims with at least one valid citation.
    A claim with only unknown records, unavailable spans, or malformed
    citations is counted as an abstention.  Duplicate citations are reported
    separately and fail the strict consistency verdict.
    """

    coverage: float
    abstention_rate: float
    claim_count: int
    supported_claim_count: int
    unsupported_claim_count: int
    abstention_count: int
    citation_count: int
    valid_citation_count: int
    invalid_citation_count: int
    duplicate_citation_count: int
    missing_citation_count: int
    missing_source_record_count: int
    unavailable_span_count: int
    evidence_record_count: int
    valid_evidence_record_count: int
    invalid_evidence_record_count: int
    duplicate_evidence_record_count: int
    referenced_evidence_count: int
    evidence_coverage: float
    source_evidence_available: bool
    fail_closed: bool
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    schema_version: int = SUMMARY_CITATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "claim_count",
            "supported_claim_count",
            "unsupported_claim_count",
            "abstention_count",
            "citation_count",
            "valid_citation_count",
            "invalid_citation_count",
            "duplicate_citation_count",
            "missing_citation_count",
            "missing_source_record_count",
            "unavailable_span_count",
            "evidence_record_count",
            "valid_evidence_record_count",
            "invalid_evidence_record_count",
            "duplicate_evidence_record_count",
            "referenced_evidence_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{field_name} must be a non-negative integer")
        for field_name in ("coverage", "abstention_rate", "evidence_coverage"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be a finite ratio")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field_name} must be a finite ratio")
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("summary citation schema version is invalid")
        reasons = tuple(sorted(set(self.failure_reasons)))
        if not all(isinstance(reason, str) and reason for reason in reasons):
            raise ValueError("failure reasons must be non-empty strings")
        object.__setattr__(self, "failure_reasons", reasons)

    @property
    def citation_coverage(self) -> float:
        """Alias for the claim-level coverage ratio."""

        return self.coverage

    @property
    def abstained_claim_count(self) -> int:
        """Return the number of claims for which review must abstain."""

        return self.abstention_count

    @property
    def unsupported_count(self) -> int:
        """Return the number of claims without valid support."""

        return self.unsupported_claim_count

    @property
    def duplicate_citations(self) -> int:
        """Return the number of repeated citation entries."""

        return self.duplicate_citation_count

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate evidence without claims, identifiers, or text."""

        return {
            "schema_version": self.schema_version,
            "coverage": float(self.coverage),
            "abstention_rate": float(self.abstention_rate),
            "claim_count": self.claim_count,
            "supported_claim_count": self.supported_claim_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "abstention_count": self.abstention_count,
            "citation_count": self.citation_count,
            "valid_citation_count": self.valid_citation_count,
            "invalid_citation_count": self.invalid_citation_count,
            "duplicate_citation_count": self.duplicate_citation_count,
            "missing_citation_count": self.missing_citation_count,
            "missing_source_record_count": self.missing_source_record_count,
            "unavailable_span_count": self.unavailable_span_count,
            "evidence_record_count": self.evidence_record_count,
            "valid_evidence_record_count": self.valid_evidence_record_count,
            "invalid_evidence_record_count": self.invalid_evidence_record_count,
            "duplicate_evidence_record_count": self.duplicate_evidence_record_count,
            "referenced_evidence_count": self.referenced_evidence_count,
            "evidence_coverage": float(self.evidence_coverage),
            "source_evidence_available": bool(self.source_evidence_available),
            "fail_closed": bool(self.fail_closed),
            "passed": bool(self.passed),
            "failure_reasons": list(self.failure_reasons),
        }

    def to_json(self) -> str:
        """Serialize aggregate metrics deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to serialized metrics."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class SummaryCitationReport:
    """Serializable, aggregate-only summary citation review report."""

    metrics: SummaryCitationMetrics
    fixture_count: int = 1
    suite: str = SUMMARY_CITATION_CONSISTENCY
    synthetic: bool = True

    @property
    def coverage(self) -> float:
        """Return claim-level citation coverage."""

        return self.metrics.coverage

    @property
    def passed(self) -> bool:
        """Return the strict citation-consistency verdict."""

        return self.metrics.passed

    def __post_init__(self) -> None:
        if type(self.fixture_count) is not int or self.fixture_count < 0:
            raise ValueError("fixture_count must be a non-negative integer")
        if self.suite != SUMMARY_CITATION_CONSISTENCY:
            raise ValueError("summary citation report suite is invalid")
        if self.synthetic is not True:
            raise ValueError("summary citation reports must be synthetic")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report without raw claim or evidence data."""

        return {
            "suite": self.suite,
            "schema_version": SUMMARY_CITATION_SCHEMA_VERSION,
            "fixture_count": self.fixture_count,
            "synthetic": True,
            "passed": self.metrics.passed,
            "metrics": self.metrics.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report with stable key ordering."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a deterministic aggregate-only Markdown report."""

        metric = self.metrics
        reasons = ", ".join(metric.failure_reasons) or "none"
        lines = [
            "# Summary Citation Consistency",
            "",
            "Aggregate citation evidence only; claim and source values are "
            "never rendered.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Coverage | {metric.coverage:.6f} |",
            f"| Abstention rate | {metric.abstention_rate:.6f} |",
            f"| Claims | {metric.claim_count} |",
            f"| Supported claims | {metric.supported_claim_count} |",
            f"| Unsupported claims | {metric.unsupported_claim_count} |",
            f"| Citations | {metric.citation_count} |",
            f"| Duplicate citations | {metric.duplicate_citation_count} |",
            f"| Missing source records | {metric.missing_source_record_count} |",
            f"| Unavailable spans | {metric.unavailable_span_count} |",
            f"| Evidence available | {metric.source_evidence_available} |",
            f"| Fail closed | {metric.fail_closed} |",
            f"| Verdict | {'pass' if metric.passed else 'fail'} |",
            f"| Failure reasons | `{reasons}` |",
        ]
        return "\n".join(lines) + "\n"

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Write deterministic JSON evidence to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown evidence to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized report."""

        return self.to_dict()[key]


def _resolve_input_alias(
    primary: object | None,
    aliases: Sequence[tuple[str, object | None]],
    name: str,
) -> object | None:
    selected = primary
    for _alias_name, candidate in aliases:
        if candidate is None:
            continue
        if selected is not None:
            raise SummaryCitationError(f"{name} was supplied more than once")
        selected = candidate
    return selected


def compute_summary_citation_metrics(
    claims: Iterable[Any] | Mapping[Any, Any] | None = None,
    evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    *,
    summary_claims: Iterable[Any] | Mapping[Any, Any] | None = None,
    source_evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    source_records: Iterable[Any] | Mapping[Any, Any] | None = None,
) -> SummaryCitationMetrics:
    """Compute deterministic citation consistency for structured claims.

    Args:
        claims: Summary claim mappings.  Each claim may contain ``citations``,
            ``evidence_id``, or a direct ``start``/``end`` span.  Claim text
            and values are ignored.
        evidence: Source records.  Each record needs an opaque identifier or
            an exact half-open span.  Records may be passed as a sequence or
            as a mapping from identifier to record.
        summary_claims: Keyword alias for ``claims``.
        source_evidence: Keyword alias for ``evidence``.
        source_records: Additional keyword alias for ``evidence``.

    Returns:
        Aggregate metrics.  Missing or malformed source evidence fails closed
        with zero coverage; no input values are copied into the result.
    """

    resolved_claims = _resolve_input_alias(
        claims,
        (("summary_claims", summary_claims),),
        "claims",
    )
    resolved_evidence = _resolve_input_alias(
        evidence,
        (
            ("source_evidence", source_evidence),
            ("source_records", source_records),
        ),
        "evidence",
    )
    claim_items = _coerce_records(resolved_claims, "claims")
    evidence_items = _coerce_records(resolved_evidence, "evidence")

    evidence_index: dict[Reference, int] = {}
    seen_evidence_references: set[Reference] = set()
    invalid_evidence_count = 0
    duplicate_evidence_count = 0
    valid_evidence_count = 0

    for record in evidence_items:
        references = _source_references(record)
        if not references:
            invalid_evidence_count += 1
            continue
        if any(reference in seen_evidence_references for reference in references):
            duplicate_evidence_count += 1
            continue
        record_index = valid_evidence_count
        valid_evidence_count += 1
        for reference in references:
            seen_evidence_references.add(reference)
            evidence_index[reference] = record_index

    source_evidence_available = bool(valid_evidence_count) and not (
        invalid_evidence_count or duplicate_evidence_count
    )
    failure_reasons: set[str] = set()
    if not evidence_items or not valid_evidence_count:
        failure_reasons.add(REASON_MISSING_SOURCE_EVIDENCE)
    if invalid_evidence_count:
        failure_reasons.add(REASON_INVALID_SOURCE_EVIDENCE)
    if duplicate_evidence_count:
        failure_reasons.add(REASON_DUPLICATE_SOURCE_EVIDENCE)

    citation_count = 0
    valid_citation_count = 0
    invalid_citation_count = 0
    duplicate_citation_count = 0
    missing_citation_count = 0
    missing_source_record_count = 0
    unavailable_span_count = 0
    supported_claim_count = 0
    referenced_evidence: set[int] = set()

    for claim in claim_items:
        citations = _claim_citations(claim)
        citation_count += len(citations)
        if not citations:
            missing_citation_count += 1
            failure_reasons.add(REASON_MISSING_CITATION)

        claim_keys: set[tuple[object, ...]] = set()
        claim_supported = False
        for citation in citations:
            references = _citation_references(citation)
            resolved_records = {
                evidence_index[reference]
                for reference in references
                if reference in evidence_index
            }
            unresolved_references = [
                reference for reference in references if reference not in evidence_index
            ]
            resolved_record = (
                next(iter(resolved_records)) if len(resolved_records) == 1 else None
            )
            key = _citation_key(references, resolved_record)
            if key is not None:
                if key in claim_keys:
                    duplicate_citation_count += 1
                    failure_reasons.add(REASON_DUPLICATE_CITATION)
                claim_keys.add(key)

            citation_is_valid = bool(references) and not unresolved_references
            if citation_is_valid and len(resolved_records) == 1:
                valid_citation_count += 1
                claim_supported = True
                referenced_evidence.update(resolved_records)
                continue

            invalid_citation_count += 1
            failure_reasons.add(REASON_INVALID_CITATION)
            if any(reference[0] == "span" for reference in unresolved_references):
                unavailable_span_count += 1
                failure_reasons.add(REASON_UNAVAILABLE_SPAN)
            if any(reference[0] == "id" for reference in unresolved_references):
                missing_source_record_count += 1
                failure_reasons.add(REASON_MISSING_SOURCE_RECORD)

        if not claim_supported:
            failure_reasons.add(REASON_UNSUPPORTED_CLAIM)

        if claim_supported:
            supported_claim_count += 1

    unsupported_claim_count = len(claim_items) - supported_claim_count
    abstention_count = unsupported_claim_count
    if source_evidence_available:
        coverage = supported_claim_count / len(claim_items) if claim_items else 1.0
        evidence_coverage = (
            len(referenced_evidence) / valid_evidence_count
            if valid_evidence_count
            else 0.0
        )
    else:
        coverage = 0.0
        evidence_coverage = 0.0
    abstention_rate = abstention_count / len(claim_items) if claim_items else 0.0
    passed = bool(source_evidence_available and not failure_reasons)

    return SummaryCitationMetrics(
        coverage=coverage,
        abstention_rate=abstention_rate,
        claim_count=len(claim_items),
        supported_claim_count=supported_claim_count,
        unsupported_claim_count=unsupported_claim_count,
        abstention_count=abstention_count,
        citation_count=citation_count,
        valid_citation_count=valid_citation_count,
        invalid_citation_count=invalid_citation_count,
        duplicate_citation_count=duplicate_citation_count,
        missing_citation_count=missing_citation_count,
        missing_source_record_count=missing_source_record_count,
        unavailable_span_count=unavailable_span_count,
        evidence_record_count=len(evidence_items),
        valid_evidence_record_count=valid_evidence_count,
        invalid_evidence_record_count=invalid_evidence_count,
        duplicate_evidence_record_count=duplicate_evidence_count,
        referenced_evidence_count=len(referenced_evidence),
        evidence_coverage=evidence_coverage,
        source_evidence_available=source_evidence_available,
        fail_closed=not source_evidence_available,
        passed=passed,
        failure_reasons=tuple(sorted(failure_reasons)),
    )


def check_summary_citation_consistency(
    summary_claims: Iterable[Any] | Mapping[Any, Any] | None = None,
    source_evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    **kwargs: Any,
) -> SummaryCitationMetrics:
    """Validate claim-to-evidence references and return aggregate metrics."""

    return compute_summary_citation_metrics(
        summary_claims,
        source_evidence,
        **kwargs,
    )


def validate_summary_citations(
    claims: Iterable[Any] | Mapping[Any, Any] | None = None,
    evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    **kwargs: Any,
) -> SummaryCitationMetrics:
    """Alias for :func:`compute_summary_citation_metrics`."""

    return compute_summary_citation_metrics(claims, evidence, **kwargs)


def summary_citation_consistency(
    claims: Iterable[Any] | Mapping[Any, Any] | None = None,
    evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    **kwargs: Any,
) -> SummaryCitationMetrics:
    """Convenience alias for :func:`check_summary_citation_consistency`."""

    return compute_summary_citation_metrics(claims, evidence, **kwargs)


def run_summary_citation_check(
    claims: Iterable[Any] | Mapping[Any, Any] | None = None,
    evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    *,
    fixture_count: int = 1,
    **kwargs: Any,
) -> SummaryCitationReport:
    """Build an aggregate-only summary citation report."""

    metrics = compute_summary_citation_metrics(claims, evidence, **kwargs)
    return SummaryCitationReport(metrics=metrics, fixture_count=fixture_count)


def build_summary_citation_report(
    claims: Iterable[Any] | Mapping[Any, Any] | None = None,
    evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    *,
    fixture_count: int = 1,
    **kwargs: Any,
) -> SummaryCitationReport:
    """Return a serializable report for a synthetic local citation check."""

    return run_summary_citation_check(
        claims,
        evidence,
        fixture_count=fixture_count,
        **kwargs,
    )


def assert_summary_citation_gate(
    claims: Iterable[Any] | Mapping[Any, Any] | None = None,
    evidence: Iterable[Any] | Mapping[Any, Any] | None = None,
    *,
    fixture_count: int = 1,
    **kwargs: Any,
) -> SummaryCitationReport:
    """Return a passing report or raise with fixed aggregate reason codes."""

    report = run_summary_citation_check(
        claims,
        evidence,
        fixture_count=fixture_count,
        **kwargs,
    )
    if not report.passed:
        reasons = ", ".join(report.metrics.failure_reasons) or "consistency_failed"
        raise AssertionError(f"summary citation-consistency gate failed: {reasons}")
    return report


def summary_citation_metadata() -> dict[str, Any]:
    """Return stable, raw-value-free metadata for the checker."""

    return {
        "suite": SUMMARY_CITATION_CONSISTENCY,
        "schema_version": SUMMARY_CITATION_SCHEMA_VERSION,
        "synthetic": True,
        "matching": "opaque_evidence_ids_or_exact_source_spans",
        "metrics": ["coverage", "abstention_rate", "duplicate_citation_count"],
        "fail_closed_on_missing_source_evidence": True,
    }


# Backwards-friendly names for callers that use "check" or "metrics" wording.
check_summary_citations = check_summary_citation_consistency
compute_citation_consistency = compute_summary_citation_metrics
validate_summary_citation_consistency = check_summary_citation_consistency


__all__ = [
    "REASON_DUPLICATE_CITATION",
    "REASON_DUPLICATE_CITATIONS",
    "REASON_DUPLICATE_SOURCE_EVIDENCE",
    "REASON_INVALID_CITATION",
    "REASON_INVALID_SOURCE_EVIDENCE",
    "REASON_MISSING_CITATION",
    "REASON_MISSING_EVIDENCE",
    "REASON_MISSING_SOURCE_EVIDENCE",
    "REASON_MISSING_SOURCE_RECORD",
    "REASON_UNAVAILABLE_SPAN",
    "REASON_UNKNOWN_CITATION",
    "REASON_UNSUPPORTED_CLAIM",
    "REASON_UNSUPPORTED_SUMMARY_CLAIM",
    "SUMMARY_CITATION_CONSISTENCY",
    "SUMMARY_CITATION_SCHEMA_VERSION",
    "SummaryCitationError",
    "SummaryCitationMetrics",
    "SummaryCitationReport",
    "assert_summary_citation_gate",
    "build_summary_citation_report",
    "check_summary_citation_consistency",
    "check_summary_citations",
    "compute_citation_consistency",
    "compute_summary_citation_metrics",
    "run_summary_citation_check",
    "summary_citation_consistency",
    "summary_citation_metadata",
    "validate_summary_citation_consistency",
    "validate_summary_citations",
]
