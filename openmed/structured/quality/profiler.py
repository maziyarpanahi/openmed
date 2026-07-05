"""PHI-free quality profiling for extracted clinical result batches."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from openmed.clinical.lab_values import derive_abnormal_flag
from openmed.clinical.units import parse_measurement
from openmed.clinical.vital_signs import (
    VitalSignResult,
    normalize_vital_measurement,
    structure_vital_sign,
)
from openmed.core.quality_gates import (
    SpanValidationResult,
    validate_entity_spans_strict,
)

QualityCategory = Literal["completeness", "conformance", "plausibility"]
QualityStatus = Literal["pass", "fail"]

_MISSING = object()
_MIN_REASONABLE_DATE = date(1900, 1, 1)

_NOTE_ID_FIELDS = ("note_id", "document_id", "doc_id", "source_document_id", "id")
_NOTE_TEXT_FIELDS = ("note_text", "text", "document_text", "source_text")
_NOTE_HASH_FIELDS = ("source_note_hash", "note_hash", "document_hash")
_NOTE_DATE_FIELDS = ("note_date", "document_date", "date", "effective_date")
_PERSON_FIELDS = ("person_id", "patient_id", "person_source_value", "subject_id")
_ENTITY_FIELDS = ("entities", "clinical_entities", "spans", "grounded_spans")
_ENTITY_TEXT_FIELDS = (
    "lexical_variant",
    "normalized_text",
    "text",
    "entity_text",
    "word",
    "surface",
    "source_value",
)
_DOMAIN_FIELDS = (
    "domain_id",
    "omop_domain",
    "domain",
    "entity_label",
    "label",
    "entity_type",
    "entity_group",
)
_CONCEPT_ID_FIELDS = (
    "concept_id",
    "standard_concept_id",
    "target_concept_id",
    "note_nlp_concept_id",
)
_SOURCE_CONCEPT_ID_FIELDS = ("source_concept_id", "note_nlp_source_concept_id")
_VOCABULARY_FIELDS = (
    "vocabulary_id",
    "source_vocabulary_id",
    "system",
    "code_system",
    "coding_system",
)
_CODE_FIELDS = ("concept_code", "code", "code_value", "source_code", "coding_code")
_METADATA_FIELDS = ("metadata", "meta")
_CODING_FIELDS = ("coding", "codings", "code", "codeable_concept")
_VALUE_FIELDS = (
    "value",
    "numeric_value",
    "measurement_value",
    "value_as_number",
    "result_value",
)
_UNIT_FIELDS = ("unit", "units", "value_unit", "measurement_unit")
_REFERENCE_RANGE_FIELDS = ("reference_range", "ref_range", "normal_range")
_DATE_FIELDS = (
    "note_date",
    "document_date",
    "date",
    "effective_date",
    "event_date",
    "start_date",
    "end_date",
)

_NOTE_REQUIRED_FIELD_GROUPS: Mapping[str, tuple[str, ...]] = {
    "note_id": _NOTE_ID_FIELDS,
    "person_id": _PERSON_FIELDS,
    "note_text": _NOTE_TEXT_FIELDS,
    "entities": _ENTITY_FIELDS,
}
_ENTITY_REQUIRED_FIELD_GROUPS: Mapping[str, tuple[str, ...]] = {
    "start": ("start",),
    "end": ("end",),
    "domain": _DOMAIN_FIELDS,
}

_DOMAIN_ALIASES: Mapping[str, str] = {
    "condition": "condition",
    "condition_occurrence": "condition",
    "diagnosis": "condition",
    "disease": "condition",
    "disorder": "condition",
    "problem": "condition",
    "symptom": "condition",
    "drug": "drug",
    "drug_exposure": "drug",
    "medication": "drug",
    "medicine": "drug",
    "rx": "drug",
    "treatment": "drug",
    "lab": "measurement",
    "lab_value": "measurement",
    "laboratory": "measurement",
    "measurement": "measurement",
    "vital": "measurement",
    "vital_sign": "measurement",
    "procedure": "procedure",
    "procedure_occurrence": "procedure",
    "surgery": "procedure",
    "operation": "procedure",
    "observation": "observation",
    "social_history": "observation",
    "finding": "observation",
}
_GROUNDING_DOMAINS = ("condition", "drug", "measurement")


@dataclass
class _SpanProxy:
    label: str
    text: str
    start: int | None
    end: int | None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileIssue:
    """One PHI-free issue found while profiling a batch."""

    category: QualityCategory
    check: str
    reason: str
    source_note_hash: str
    span_index: int | None = None
    start: int | None = None
    end: int | None = None
    field: str | None = None
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible issue without raw values or note text."""

        return asdict(self)


@dataclass(frozen=True)
class QualityCheck:
    """Aggregate pass/fail evidence for one profiler check."""

    name: str
    category: QualityCategory
    passed: bool
    total: int
    failed: int
    issues: tuple[ProfileIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible check payload."""

        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "total": self.total,
            "failed": self.failed,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class DomainGroundingCoverage:
    """Grounding coverage for one vocabulary domain."""

    domain: str
    total: int
    grounded: int
    ungrounded: int
    coverage: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible coverage payload."""

        return asdict(self)


@dataclass(frozen=True)
class NoteQualityProfile:
    """PHI-free per-note profile summary."""

    index: int
    source_note_hash: str
    total_spans: int
    grounded_spans: int
    ungrounded_spans: int
    missing_required_fields: int
    required_fields: int
    null_fields: int
    profiled_fields: int
    invalid_spans: int
    residual_overlaps: int
    plausibility_issues: int
    completeness_score: float

    @property
    def status(self) -> QualityStatus:
        """Return pass when the note has no structural or plausibility issues."""

        if (
            self.missing_required_fields
            or self.invalid_spans
            or self.residual_overlaps
            or self.plausibility_issues
        ):
            return "fail"
        return "pass"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible per-note summary."""

        payload = asdict(self)
        payload["status"] = self.status
        return payload


@dataclass(frozen=True)
class QualityProfileReport:
    """Structured PHI-free data-quality report for extracted results."""

    status: QualityStatus
    completeness_floor: float
    overall_completeness_score: float
    category_scores: Mapping[str, float]
    pipeline_gate: Mapping[str, Any]
    totals: Mapping[str, int]
    domain_grounding: tuple[DomainGroundingCoverage, ...]
    notes: tuple[NoteQualityProfile, ...]
    checks: tuple[QualityCheck, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible report."""

        return {
            "status": self.status,
            "completeness_floor": self.completeness_floor,
            "overall_completeness_score": self.overall_completeness_score,
            "category_scores": dict(self.category_scores),
            "pipeline_gate": dict(self.pipeline_gate),
            "totals": dict(self.totals),
            "domain_grounding": [
                coverage.to_dict() for coverage in self.domain_grounding
            ],
            "notes": [note.to_dict() for note in self.notes],
            "checks": [check.to_dict() for check in self.checks],
            "summary": self.summary,
        }


class QualityGateError(ValueError):
    """Raised when a downstream load is blocked by a profile floor."""

    def __init__(self, report: QualityProfileReport) -> None:
        self.report = report
        super().__init__(
            "quality profile gate failed: "
            f"score {report.overall_completeness_score:.3f} is below "
            f"floor {report.completeness_floor:.3f}"
        )


def load_profile_jsonl(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Load extracted result records from a JSONL file."""

    records: list[Mapping[str, Any]] = []
    source = Path(path).expanduser()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{source}:{line_number} must contain a JSON object")
            records.append(payload)
    return tuple(records)


def load_profile_jsonl_text(text: str) -> tuple[Mapping[str, Any], ...]:
    """Load extracted result records from an in-memory JSONL payload."""

    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise ValueError(f"jsonl line {line_number} must contain a JSON object")
        records.append(payload)
    return tuple(records)


def profile_extracted_batch(
    records: Iterable[Any],
    *,
    completeness_floor: float = 0.8,
    athena_index: Mapping[str, Any] | None = None,
    reference_date: date | None = None,
) -> QualityProfileReport:
    """Profile extracted and grounded clinical results.

    The report intentionally contains only aggregate counts, hashes, offsets,
    field names, domains, and pass/fail evidence. Raw note text, entity text,
    source identifiers, and measured values are never copied into the report.
    """

    if completeness_floor < 0 or completeness_floor > 1:
        raise ValueError("completeness_floor must be between 0 and 1")

    today = reference_date or date.today()
    athena = _AthenaLookup(athena_index)
    note_profiles: list[NoteQualityProfile] = []
    issues: list[ProfileIssue] = []
    domain_counts: dict[str, Counter[str]] = {
        domain: Counter({"total": 0, "grounded": 0, "ungrounded": 0})
        for domain in _GROUNDING_DOMAINS
    }
    totals = Counter(
        {
            "records": 0,
            "spans": 0,
            "grounded_spans": 0,
            "ungrounded_spans": 0,
            "required_fields": 0,
            "missing_required_fields": 0,
            "profiled_fields": 0,
            "null_fields": 0,
            "invalid_spans": 0,
            "residual_overlaps": 0,
            "plausibility_candidates": 0,
            "plausibility_issues": 0,
        }
    )

    for index, record in enumerate(records):
        totals["records"] += 1
        note_hash = _note_hash(record, index)
        entities = _note_entities(record)
        note_text = _first_text((record,), _NOTE_TEXT_FIELDS)
        required_total, missing_required, required_issues = _required_field_issues(
            record,
            entities,
            note_hash=note_hash,
        )
        issues.extend(required_issues)

        field_total, null_fields = _field_stats(record)
        span_result = _span_result(entities, note_text)
        span_issues = _span_issues(span_result, note_hash=note_hash)
        issues.extend(span_issues)

        note_dates, note_date_candidates = _date_issues(
            record,
            note_hash=note_hash,
            span_index=None,
            reference_date=today,
        )
        issues.extend(note_dates)
        plausibility_candidates = note_date_candidates
        plausibility_issue_count = len(note_dates)

        grounded_spans = 0
        ungrounded_spans = 0
        for span_index, entity in enumerate(entities):
            domain = _entity_domain(entity, athena)
            grounded = _entity_grounded(entity, athena)
            if domain in _GROUNDING_DOMAINS:
                domain_counts[domain]["total"] += 1
                if grounded:
                    domain_counts[domain]["grounded"] += 1
                else:
                    domain_counts[domain]["ungrounded"] += 1
            if grounded:
                grounded_spans += 1
            else:
                ungrounded_spans += 1

            span_plausibility, candidates = _entity_plausibility_issues(
                entity,
                note_hash=note_hash,
                span_index=span_index,
                domain=domain,
                reference_date=today,
            )
            plausibility_candidates += candidates
            plausibility_issue_count += len(span_plausibility)
            issues.extend(span_plausibility)

        completeness_score = _note_completeness_score(
            required_total=required_total,
            missing_required=missing_required,
            grounded=grounded_spans,
            ungrounded=ungrounded_spans,
        )
        profile = NoteQualityProfile(
            index=index,
            source_note_hash=note_hash,
            total_spans=len(entities),
            grounded_spans=grounded_spans,
            ungrounded_spans=ungrounded_spans,
            missing_required_fields=missing_required,
            required_fields=required_total,
            null_fields=null_fields,
            profiled_fields=field_total,
            invalid_spans=span_result.invalid_spans,
            residual_overlaps=span_result.residual_overlaps,
            plausibility_issues=plausibility_issue_count,
            completeness_score=completeness_score,
        )
        note_profiles.append(profile)

        totals["spans"] += len(entities)
        totals["grounded_spans"] += grounded_spans
        totals["ungrounded_spans"] += ungrounded_spans
        totals["required_fields"] += required_total
        totals["missing_required_fields"] += missing_required
        totals["profiled_fields"] += field_total
        totals["null_fields"] += null_fields
        totals["invalid_spans"] += span_result.invalid_spans
        totals["residual_overlaps"] += span_result.residual_overlaps
        totals["plausibility_candidates"] += plausibility_candidates
        totals["plausibility_issues"] += plausibility_issue_count

    overall_score = _round_score(
        sum(note.completeness_score for note in note_profiles) / len(note_profiles)
        if note_profiles
        else 1.0
    )
    gate_passed = overall_score >= completeness_floor
    status: QualityStatus = "pass" if gate_passed else "fail"
    coverage = tuple(
        DomainGroundingCoverage(
            domain=domain,
            total=domain_counts[domain]["total"],
            grounded=domain_counts[domain]["grounded"],
            ungrounded=domain_counts[domain]["ungrounded"],
            coverage=_rate(
                domain_counts[domain]["grounded"],
                domain_counts[domain]["total"],
            ),
        )
        for domain in _GROUNDING_DOMAINS
    )
    checks = _checks_from_issues(totals, issues)
    category_scores = {
        "completeness": overall_score,
        "conformance": _round_score(
            1.0
            - _rate(
                totals["invalid_spans"] + totals["residual_overlaps"],
                totals["spans"],
            )
        ),
        "plausibility": _round_score(
            1.0
            - _rate(totals["plausibility_issues"], totals["plausibility_candidates"])
        ),
    }
    pipeline_gate = {
        "passed": gate_passed,
        "reason": "passed" if gate_passed else "completeness_floor",
        "score": overall_score,
        "floor": completeness_floor,
    }
    summary = _summary(
        status=status,
        records=totals["records"],
        spans=totals["spans"],
        score=overall_score,
        floor=completeness_floor,
        failed_checks=sum(1 for check in checks if not check.passed),
    )

    return QualityProfileReport(
        status=status,
        completeness_floor=completeness_floor,
        overall_completeness_score=overall_score,
        category_scores=category_scores,
        pipeline_gate=pipeline_gate,
        totals=dict(totals),
        domain_grounding=coverage,
        notes=tuple(note_profiles),
        checks=checks,
        summary=summary,
    )


def assert_profile_gate(report: QualityProfileReport) -> None:
    """Raise :class:`QualityGateError` when the pipeline gate failed."""

    if not bool(report.pipeline_gate.get("passed")):
        raise QualityGateError(report)


def render_profile_summary(report: QualityProfileReport) -> str:
    """Render a compact human-readable summary without raw input values."""

    lines = [report.summary]
    for coverage in report.domain_grounding:
        lines.append(
            f"{coverage.domain}: {coverage.grounded}/{coverage.total} grounded "
            f"({coverage.coverage:.3f})"
        )
    for check in report.checks:
        state = "pass" if check.passed else "fail"
        lines.append(f"{check.category}.{check.name}: {state} ({check.failed} failed)")
    return "\n".join(lines)


class _AthenaLookup:
    def __init__(self, index: Mapping[str, Any] | None) -> None:
        self._by_concept_id: dict[int, Mapping[str, Any]] = {}
        self._by_code: dict[tuple[str, str], Mapping[str, Any]] = {}
        if not index:
            return
        for vocabulary_id, concepts in index.items():
            if vocabulary_id == "_meta" or not isinstance(concepts, Mapping):
                continue
            for concept_code, record in concepts.items():
                if not isinstance(record, Mapping):
                    continue
                concept_id = _optional_int(record.get("concept_id"))
                if concept_id is not None:
                    self._by_concept_id[concept_id] = record
                self._by_code[(_norm(vocabulary_id), _norm(concept_code))] = record

    def record_for(self, entity: Any) -> Mapping[str, Any] | None:
        concept_id = _entity_concept_id(entity)
        if concept_id is not None and concept_id in self._by_concept_id:
            return self._by_concept_id[concept_id]

        vocabulary_id = _first_text(
            _entity_and_coding_sources(entity), _VOCABULARY_FIELDS
        )
        concept_code = _first_text(_entity_and_coding_sources(entity), _CODE_FIELDS)
        if vocabulary_id and concept_code:
            return self._by_code.get((_norm(vocabulary_id), _norm(concept_code)))
        return None


def _required_field_issues(
    note: Any,
    entities: Sequence[Any],
    *,
    note_hash: str,
) -> tuple[int, int, list[ProfileIssue]]:
    total = 0
    missing = 0
    issues: list[ProfileIssue] = []

    for field_name, field_group in _NOTE_REQUIRED_FIELD_GROUPS.items():
        total += 1
        if _first_value((note,), field_group) is _MISSING:
            missing += 1
            issues.append(
                ProfileIssue(
                    category="completeness",
                    check="required_fields",
                    reason="missing_required_field",
                    source_note_hash=note_hash,
                    field=field_name,
                )
            )

    for span_index, entity in enumerate(entities):
        sources = _entity_sources(entity)
        for field_name, field_group in _ENTITY_REQUIRED_FIELD_GROUPS.items():
            total += 1
            if _first_value(sources, field_group) is _MISSING:
                missing += 1
                start, end = _span_bounds(entity)
                issues.append(
                    ProfileIssue(
                        category="completeness",
                        check="required_fields",
                        reason="missing_required_field",
                        source_note_hash=note_hash,
                        span_index=span_index,
                        start=start,
                        end=end,
                        field=field_name,
                    )
                )
    return total, missing, issues


def _span_result(entities: Sequence[Any], note_text: str) -> SpanValidationResult:
    if not note_text:
        return validate_entity_spans_strict((), "")
    safe_entities = [_safe_span_entity(entity, note_text) for entity in entities]
    return validate_entity_spans_strict(safe_entities, note_text)


def _span_issues(
    result: SpanValidationResult,
    *,
    note_hash: str,
) -> list[ProfileIssue]:
    issues: list[ProfileIssue] = []
    for issue in result.offending_spans:
        issues.append(
            ProfileIssue(
                category="conformance",
                check="span_integrity",
                reason="invalid_span",
                source_note_hash=note_hash,
                span_index=issue.index,
                start=issue.start,
                end=issue.end,
                field=",".join(issue.problems),
            )
        )
    for finding in result.overlap_findings:
        first = finding.first
        issues.append(
            ProfileIssue(
                category="conformance",
                check="span_integrity",
                reason="overlapping_span",
                source_note_hash=note_hash,
                span_index=_optional_int(first.get("index")),
                start=_optional_int(first.get("start")),
                end=_optional_int(first.get("end")),
            )
        )
    return issues


def _safe_span_entity(entity: Any, note_text: str) -> _SpanProxy:
    start, end = _span_bounds(entity)
    span_text = _entity_text(entity)
    if not span_text and start is not None and end is not None:
        if 0 <= start <= end <= len(note_text):
            span_text = note_text[start:end]
    confidence = _first_numeric(_entity_sources(entity), ("confidence", "score"))
    metadata = _first_value(_entity_sources(entity), ("metadata", "meta"))
    return _SpanProxy(
        label=_first_text(_entity_sources(entity), _DOMAIN_FIELDS) or "UNKNOWN",
        text=span_text,
        start=start,
        end=end,
        confidence=confidence or 0.0,
        metadata=copy.deepcopy(dict(metadata)) if isinstance(metadata, Mapping) else {},
    )


def _entity_plausibility_issues(
    entity: Any,
    *,
    note_hash: str,
    span_index: int,
    domain: str | None,
    reference_date: date,
) -> tuple[list[ProfileIssue], int]:
    issues: list[ProfileIssue] = []
    candidates = 0
    start, end = _span_bounds(entity)

    date_issues, date_candidates = _date_issues(
        entity,
        note_hash=note_hash,
        span_index=span_index,
        reference_date=reference_date,
    )
    issues.extend(date_issues)
    candidates += date_candidates

    numeric_value = _first_numeric(_entity_sources(entity), _VALUE_FIELDS)
    unit = _first_text(_entity_sources(entity), _UNIT_FIELDS)
    reference_range = _first_value(_entity_sources(entity), _REFERENCE_RANGE_FIELDS)
    is_measurement = domain == "measurement" or _looks_like_measurement(entity)

    if is_measurement and (
        numeric_value is not None or unit or reference_range is not _MISSING
    ):
        candidates += 1
        if numeric_value is None:
            issues.append(
                ProfileIssue(
                    category="plausibility",
                    check="measurement_value",
                    reason="missing_numeric_measurement",
                    source_note_hash=note_hash,
                    span_index=span_index,
                    start=start,
                    end=end,
                    domain=domain,
                )
            )
        elif numeric_value < 0 or not math.isfinite(numeric_value):
            issues.append(
                ProfileIssue(
                    category="plausibility",
                    check="measurement_value",
                    reason="implausible_numeric_value",
                    source_note_hash=note_hash,
                    span_index=span_index,
                    start=start,
                    end=end,
                    domain=domain,
                )
            )
        if numeric_value is not None and unit:
            parsed = parse_measurement(numeric_value, unit)
            if parsed["status"] != "ok":
                issues.append(
                    ProfileIssue(
                        category="plausibility",
                        check="measurement_unit",
                        reason=f"unit_{parsed['status']}",
                        source_note_hash=note_hash,
                        span_index=span_index,
                        start=start,
                        end=end,
                        domain=domain,
                    )
                )
        if numeric_value is not None and reference_range is not _MISSING:
            flag = derive_abnormal_flag(numeric_value, reference_range, value_unit=unit)
            if flag in {"low", "high", "critical"}:
                issues.append(
                    ProfileIssue(
                        category="plausibility",
                        check="reference_range",
                        reason=f"out_of_reference_range_{flag}",
                        source_note_hash=note_hash,
                        span_index=span_index,
                        start=start,
                        end=end,
                        domain=domain,
                    )
                )

    vital = structure_vital_sign(_entity_text(entity))
    if vital["kind"] != "unknown":
        candidates += 1
        vital_reason = _vital_issue_reason(vital)
        if vital_reason is not None:
            issues.append(
                ProfileIssue(
                    category="plausibility",
                    check="vital_sign",
                    reason=vital_reason,
                    source_note_hash=note_hash,
                    span_index=span_index,
                    start=start,
                    end=end,
                    domain=domain,
                )
            )

    return issues, candidates


def _date_issues(
    source: Any,
    *,
    note_hash: str,
    span_index: int | None,
    reference_date: date,
) -> tuple[list[ProfileIssue], int]:
    issues: list[ProfileIssue] = []
    candidates = 0
    max_date = reference_date + timedelta(days=1)
    start, end = _span_bounds(source)
    for field_name in _DATE_FIELDS:
        raw = _first_value(_entity_sources(source), (field_name,))
        if raw is _MISSING or raw is None or raw == "":
            continue
        candidates += 1
        parsed = _parse_date(raw)
        if parsed is None:
            issues.append(
                ProfileIssue(
                    category="plausibility",
                    check="date_range",
                    reason="invalid_date",
                    source_note_hash=note_hash,
                    span_index=span_index,
                    start=start,
                    end=end,
                    field=field_name,
                )
            )
        elif parsed < _MIN_REASONABLE_DATE:
            issues.append(
                ProfileIssue(
                    category="plausibility",
                    check="date_range",
                    reason="date_before_reasonable_floor",
                    source_note_hash=note_hash,
                    span_index=span_index,
                    start=start,
                    end=end,
                    field=field_name,
                )
            )
        elif parsed > max_date:
            issues.append(
                ProfileIssue(
                    category="plausibility",
                    check="date_range",
                    reason="date_after_reference_window",
                    source_note_hash=note_hash,
                    span_index=span_index,
                    start=start,
                    end=end,
                    field=field_name,
                )
            )
    return issues, candidates


def _vital_issue_reason(vital: VitalSignResult) -> str | None:
    kind = vital["kind"]
    if kind == "blood_pressure":
        components = {
            item["kind"]: float(item["value"]) for item in vital["components"]
        }
        systolic = components.get("systolic")
        diastolic = components.get("diastolic")
        if systolic is None or diastolic is None:
            return "missing_blood_pressure_component"
        if systolic < 40 or systolic > 260 or diastolic < 20 or diastolic > 160:
            return "vital_value_out_of_plausible_range"
        if systolic <= diastolic:
            return "blood_pressure_components_inverted"
        return None

    value = vital.get("value")
    if value is None:
        return "missing_vital_value"
    numeric = float(value)
    if kind == "heart_rate" and not 20 <= numeric <= 250:
        return "vital_value_out_of_plausible_range"
    if kind == "respiratory_rate" and not 4 <= numeric <= 80:
        return "vital_value_out_of_plausible_range"
    if kind == "oxygen_saturation" and not 50 <= numeric <= 100:
        return "vital_value_out_of_plausible_range"
    if kind == "body_temperature":
        unit = vital.get("unit") or "Cel"
        converted = normalize_vital_measurement(numeric, unit, "Cel")
        if converted["status"] != "ok":
            return f"unit_{converted['status']}"
        temperature_c = converted["magnitude"]
        if temperature_c is None or not 30 <= float(temperature_c) <= 45:
            return "vital_value_out_of_plausible_range"
    return None


def _checks_from_issues(
    totals: Mapping[str, int],
    issues: Sequence[ProfileIssue],
) -> tuple[QualityCheck, ...]:
    by_check: dict[tuple[QualityCategory, str], list[ProfileIssue]] = {}
    for issue in issues:
        by_check.setdefault((issue.category, issue.check), []).append(issue)
    check_specs: tuple[tuple[QualityCategory, str, int, int], ...] = (
        (
            "completeness",
            "required_fields",
            totals["required_fields"],
            totals["missing_required_fields"],
        ),
        (
            "completeness",
            "grounding_coverage",
            totals["spans"],
            totals["ungrounded_spans"],
        ),
        (
            "completeness",
            "null_density",
            totals["profiled_fields"],
            totals["null_fields"],
        ),
        (
            "conformance",
            "span_integrity",
            totals["spans"],
            totals["invalid_spans"] + totals["residual_overlaps"],
        ),
        (
            "plausibility",
            "measurement_value",
            totals["plausibility_candidates"],
            len(by_check.get(("plausibility", "measurement_value"), ())),
        ),
        (
            "plausibility",
            "measurement_unit",
            totals["plausibility_candidates"],
            len(by_check.get(("plausibility", "measurement_unit"), ())),
        ),
        (
            "plausibility",
            "reference_range",
            totals["plausibility_candidates"],
            len(by_check.get(("plausibility", "reference_range"), ())),
        ),
        (
            "plausibility",
            "vital_sign",
            totals["plausibility_candidates"],
            len(by_check.get(("plausibility", "vital_sign"), ())),
        ),
        (
            "plausibility",
            "date_range",
            totals["plausibility_candidates"],
            len(by_check.get(("plausibility", "date_range"), ())),
        ),
    )
    return tuple(
        QualityCheck(
            name=name,
            category=category,
            passed=failed == 0,
            total=total,
            failed=failed,
            issues=tuple(by_check.get((category, name), ())),
        )
        for category, name, total, failed in check_specs
    )


def _note_completeness_score(
    *,
    required_total: int,
    missing_required: int,
    grounded: int,
    ungrounded: int,
) -> float:
    required_score = 1.0 - _rate(missing_required, required_total)
    groundable = grounded + ungrounded
    grounding_score = _rate(grounded, groundable) if groundable else 1.0
    return _round_score((required_score + grounding_score) / 2.0)


def _field_stats(value: Any) -> tuple[int, int]:
    total = 0
    nulls = 0
    if isinstance(value, Mapping):
        for item in value.values():
            total += 1
            if _is_null(item):
                nulls += 1
            child_total, child_nulls = _field_stats(item)
            total += child_total
            nulls += child_nulls
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            child_total, child_nulls = _field_stats(item)
            total += child_total
            nulls += child_nulls
    return total, nulls


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _note_hash(note: Any, index: int) -> str:
    explicit_hash = _first_text((note,), _NOTE_HASH_FIELDS)
    if explicit_hash:
        return _sha256_text(explicit_hash)
    note_text = _first_text((note,), _NOTE_TEXT_FIELDS)
    if note_text:
        return _sha256_text(note_text)
    source_id = _first_text((note,), _NOTE_ID_FIELDS)
    if source_id:
        return _sha256_text(source_id)
    return _sha256_text(f"record:{index}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _note_entities(note: Any) -> tuple[Any, ...]:
    value = _first_value((note,), _ENTITY_FIELDS)
    if value is _MISSING or value is None:
        return ()
    return tuple(_as_sequence(value))


def _entity_domain(entity: Any, athena: _AthenaLookup) -> str | None:
    record = athena.record_for(entity)
    if record is not None:
        domain = _normalize_domain(record.get("domain_id"))
        if domain is not None:
            return domain
    for source in _entity_sources(entity):
        domain = _normalize_domain(_first_text((source,), _DOMAIN_FIELDS))
        if domain is not None:
            return domain
    return None


def _entity_grounded(entity: Any, athena: _AthenaLookup) -> bool:
    concept_id = _entity_concept_id(entity)
    if concept_id is not None and concept_id > 0:
        return True
    record = athena.record_for(entity)
    if record is None:
        return False
    concept_id = _optional_int(record.get("concept_id"))
    return concept_id is not None and concept_id > 0


def _entity_concept_id(entity: Any) -> int | None:
    concept_id = _optional_int(
        _first_value(_entity_sources(entity), _CONCEPT_ID_FIELDS)
    )
    if concept_id is not None:
        return concept_id
    return _optional_int(
        _first_value(_entity_sources(entity), _SOURCE_CONCEPT_ID_FIELDS)
    )


def _entity_text(entity: Any) -> str:
    return _first_text(_entity_sources(entity), _ENTITY_TEXT_FIELDS)


def _looks_like_measurement(entity: Any) -> bool:
    label = _first_text(_entity_sources(entity), _DOMAIN_FIELDS).casefold()
    return any(term in label for term in ("lab", "measurement", "vital", "value"))


def _normalize_domain(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().replace("-", "_").replace(" ", "_").casefold()
    if not normalized:
        return None
    return _DOMAIN_ALIASES.get(normalized)


def _entity_sources(entity: Any) -> tuple[Any, ...]:
    sources: list[Any] = [entity]
    for source in tuple(sources):
        for name in _METADATA_FIELDS:
            metadata = _value(source, name)
            if metadata is not _MISSING and metadata is not None:
                sources.append(metadata)
    return tuple(sources)


def _entity_and_coding_sources(entity: Any) -> tuple[Any, ...]:
    sources = list(_entity_sources(entity))
    for source in tuple(sources):
        for name in _CODING_FIELDS:
            coding = _coerce_coding(_value(source, name))
            if coding is not None:
                sources.insert(0, coding)
    return tuple(sources)


def _coerce_coding(value: Any) -> Mapping[str, Any] | None:
    if value is _MISSING or value is None or isinstance(value, (str, bytes)):
        return None
    if isinstance(value, Mapping):
        nested = _coerce_coding(_value(value, "coding"))
        if nested is not None:
            return nested
        return value
    if isinstance(value, Sequence):
        for item in value:
            coding = _coerce_coding(item)
            if coding is not None:
                return coding
    return None


def _span_bounds(entity: Any) -> tuple[int | None, int | None]:
    return (
        _optional_int(_first_value(_entity_sources(entity), ("start",))),
        _optional_int(_first_value(_entity_sources(entity), ("end",))),
    )


def _first_value(sources: Iterable[Any], names: Iterable[str]) -> Any:
    for source in sources:
        for name in names:
            value = _value(source, name)
            if value is not _MISSING and value is not None:
                return value
    return _MISSING


def _first_text(sources: Iterable[Any], names: Iterable[str]) -> str:
    value = _first_value(sources, names)
    if value is _MISSING or value is None:
        return ""
    return str(value).strip()


def _first_numeric(sources: Iterable[Any], names: Iterable[str]) -> float | None:
    value = _first_value(sources, names)
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping) and name in item:
        return item[name]
    if hasattr(item, name):
        return getattr(item, name)
    return _MISSING


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return value
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _optional_int(value: Any) -> int | None:
    if value is _MISSING or value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _round_score(numerator / denominator)


def _round_score(value: float) -> float:
    return round(float(max(0.0, min(1.0, value))), 6)


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _summary(
    *,
    status: QualityStatus,
    records: int,
    spans: int,
    score: float,
    floor: float,
    failed_checks: int,
) -> str:
    return (
        f"{status}: {records} records, {spans} spans, "
        f"completeness={score:.3f}, floor={floor:.3f}, "
        f"failed_checks={failed_checks}"
    )


__all__ = [
    "DomainGroundingCoverage",
    "NoteQualityProfile",
    "ProfileIssue",
    "QualityCheck",
    "QualityGateError",
    "QualityProfileReport",
    "assert_profile_gate",
    "load_profile_jsonl",
    "load_profile_jsonl_text",
    "profile_extracted_batch",
    "render_profile_summary",
]
