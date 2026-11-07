"""Deterministic, PHI-free quality profiling for extracted clinical output."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openmed.clinical.lab_values import derive_abnormal_flag
from openmed.clinical.units import normalize_to, parse_measurement
from openmed.clinical.vital_signs import structure_vital_sign
from openmed.core.quality_gates import validate_entity_spans_strict

PROFILE_SCHEMA_VERSION = "openmed.quality.profile.v1"
DEFAULT_DATE_MIN = date(1900, 1, 1)
DEFAULT_DATE_MAX = date(2100, 12, 31)

_MISSING = object()
_SPAN_KEYS = (
    "entities",
    "clinical_entities",
    "spans",
    "grounded_spans",
    "grounded_entities",
    "mentions",
    "extractions",
    "results",
    "grounded",
)
_FIELD_KEYS = ("fields", "field_values", "extracted_fields", "extracted")
_DATE_KEYS = frozenset(
    {
        "date",
        "event_date",
        "note_date",
        "document_date",
        "observation_date",
        "measurement_date",
        "start_date",
        "end_date",
        "effective_date",
        "recorded_date",
        "onset_date",
        "resolution_date",
        "birth_date",
    }
)
_RESERVED_RECORD_KEYS = frozenset(
    {
        *_SPAN_KEYS,
        *_FIELD_KEYS,
        "note_text",
        "document_text",
        "source_text",
        "text",
        "document_id",
        "doc_id",
        "note_id",
        "patient_id",
        "person_id",
        "subject_id",
        "visit_id",
        "encounter_id",
        "required_fields",
    }
)
_DOMAIN_ALIASES = {
    "condition": "condition",
    "conditions": "condition",
    "clinical_finding": "condition",
    "clinical_findings": "condition",
    "diagnosis": "condition",
    "disease": "condition",
    "disorder": "condition",
    "problem": "condition",
    "problems": "condition",
    "symptom": "condition",
    "symptoms": "condition",
    "dx": "condition",
    "drug": "drug",
    "drugs": "drug",
    "medication": "drug",
    "medications": "drug",
    "medicine": "drug",
    "medicines": "drug",
    "rx": "drug",
    "treatment": "drug",
    "measurement": "measurement",
    "measurements": "measurement",
    "lab": "measurement",
    "labs": "measurement",
    "lab_value": "measurement",
    "lab_values": "measurement",
    "laboratory": "measurement",
    "vital": "measurement",
    "vitals": "measurement",
    "vital_sign": "measurement",
    "vital_signs": "measurement",
    "observation": "measurement",
    "observations": "measurement",
    "lab_test": "measurement",
    "lab_tests": "measurement",
}
_VITAL_LIMITS: Mapping[str, tuple[float, float]] = {
    "heart_rate": (20.0, 300.0),
    "respiratory_rate": (2.0, 80.0),
    "body_temperature": (25.0, 45.0),
    "oxygen_saturation": (0.0, 100.0),
    "systolic": (30.0, 300.0),
    "diastolic": (15.0, 200.0),
}
_VITAL_TARGET_UNITS = {
    "heart_rate": "beat/min",
    "respiratory_rate": "breath/min",
    "body_temperature": "Cel",
    "oxygen_saturation": "%",
    "systolic": "mmHg",
    "diastolic": "mmHg",
}


@dataclass
class _SpanAdapter:
    """Attribute-based span view required by the shared span gate."""

    text: str
    label: str
    start: Any
    end: Any
    confidence: Any = None
    score: Any = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _AthenaLookup:
    """Minimal standard-concept lookup derived from a caller-supplied index."""

    by_code: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]]
    by_any_code: Mapping[str, tuple[Mapping[str, Any], ...]]
    by_id: Mapping[int, tuple[Mapping[str, Any], ...]]
    vocabulary_ids: tuple[str, ...]

    @classmethod
    def from_index(cls, index: Mapping[str, Any]) -> "_AthenaLookup":
        by_code: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        by_any_code: dict[str, list[Mapping[str, Any]]] = {}
        by_id: dict[int, list[Mapping[str, Any]]] = {}
        vocabulary_ids: set[str] = set()
        for vocabulary_id, concepts in index.items():
            if vocabulary_id == "_meta" or not isinstance(concepts, Mapping):
                continue
            vocabulary = _normalise_name(vocabulary_id)
            if not vocabulary:
                continue
            vocabulary_ids.add(str(vocabulary_id))
            for code, raw_record in concepts.items():
                if not isinstance(raw_record, Mapping):
                    continue
                record = dict(raw_record)
                record_code = str(
                    _first_value((record,), ("concept_code", "code")) or code
                ).strip()
                if not record_code:
                    continue
                code_key = _normalise_name(record_code)
                by_code.setdefault((vocabulary, code_key), []).append(record)
                by_any_code.setdefault(code_key, []).append(record)
                concept_id = _positive_int(
                    _first_value((record,), ("concept_id", "conceptId"))
                )
                if concept_id is not None:
                    by_id.setdefault(concept_id, []).append(record)
        return cls(
            by_code={
                key: tuple(sorted(records, key=_athena_record_sort_key))
                for key, records in by_code.items()
            },
            by_any_code={
                key: tuple(sorted(records, key=_athena_record_sort_key))
                for key, records in by_any_code.items()
            },
            by_id={
                key: tuple(sorted(records, key=_athena_record_sort_key))
                for key, records in by_id.items()
            },
            vocabulary_ids=tuple(sorted(vocabulary_ids)),
        )

    def standard_match(self, span: Mapping[str, Any]) -> bool:
        """Return whether one span resolves to a standard Athena concept."""

        span_domain = _canonical_domain(_span_field(span))
        identifiers = _span_identifiers(span)
        records: list[Mapping[str, Any]] = []
        for vocabulary, code in identifiers["codes"]:
            if vocabulary:
                records.extend(self.by_code.get((vocabulary, code), ()))
            if not records:
                records.extend(self.by_any_code.get(code, ()))
        for concept_id in identifiers["concept_ids"]:
            records.extend(self.by_id.get(concept_id, ()))
        unique_records: list[Mapping[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for record in records:
            marker = (
                _first_value((record,), ("concept_id", "conceptId")),
                _first_value((record,), ("concept_code", "code")),
                _first_value((record,), ("vocabulary_id", "vocabularyId")),
            )
            if marker in seen:
                continue
            seen.add(marker)
            unique_records.append(record)
        return any(
            _is_standard_concept(record) and _record_matches_domain(record, span_domain)
            for record in unique_records
        )


class QualityGateError(ValueError):
    """Raised when an ETL quality floor rejects a batch."""

    def __init__(self, report: "QualityProfileReport") -> None:
        self.report = report
        score = report.overall_completeness_score
        floor = report["gate"]["completeness_floor"]
        if score < floor:
            message = (
                "Clinical data-quality gate failed: completeness score "
                f"{score:.6f} is below the configured floor {floor:.6f}."
            )
        else:
            message = "Clinical data-quality gate failed: a quality check did not pass."
        super().__init__(message)


class QualityProfileReport(Mapping[str, Any]):
    """Mapping wrapper for a deterministic quality report."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = _json_safe_copy(payload)

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    @property
    def passed(self) -> bool:
        """Return whether all checks and the configured gate passed."""

        return bool(self._payload["passed"])

    @property
    def status(self) -> str:
        """Return pass or fail for the complete profile."""

        return str(self._payload["status"])

    @property
    def overall_completeness_score(self) -> float:
        """Return the aggregate completeness score."""

        return float(self._payload["completeness"]["overall_score"])

    @property
    def human_summary(self) -> str:
        """Return the PHI-free human-readable summary."""

        return str(self._payload["human_summary"])

    @property
    def summary(self) -> str:
        """Compatibility alias for human_summary."""

        return self.human_summary

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible report mapping."""

        return _json_safe_copy(self._payload)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report with stable key ordering."""

        return json.dumps(
            self._payload,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def raise_for_gate(self) -> None:
        """Raise QualityGateError when this report did not pass."""

        if not self.passed:
            raise QualityGateError(self)


def profile_results(
    results: Iterable[Any] | Mapping[str, Any] | QualityProfileReport,
    *,
    required_fields: Iterable[str] | Mapping[str, Any] | None = None,
    completeness_floor: float = 0.0,
    quality_floor: float | None = None,
    athena_index: Mapping[str, Any] | str | Path | None = None,
    vocabulary_index: Mapping[str, Any] | str | Path | None = None,
    date_min: date | str = DEFAULT_DATE_MIN,
    date_max: date | str = DEFAULT_DATE_MAX,
    language: object | None = None,
) -> QualityProfileReport:
    """Profile a batch of extracted and grounded note results.

    A record may contain entities, spans, grounded spans, mentions, or a
    fields mapping. Spans can use domain/domain_id and the standard grounding
    fields emitted by OpenMed.
    """

    floor = _resolve_floor(completeness_floor, quality_floor)
    minimum_date = _coerce_date_bound(date_min, "date_min")
    maximum_date = _coerce_date_bound(date_max, "date_max")
    if minimum_date > maximum_date:
        raise ValueError("date_min must not be after date_max")
    records = _normalise_records(results)
    global_required = _normalise_required_fields(required_fields)
    vocabulary_source = athena_index if athena_index is not None else vocabulary_index
    lookup = _load_athena_lookup(vocabulary_source)

    domain_counts: dict[str, dict[str, int]] = {}
    field_counts: dict[str, dict[str, int]] = {}
    note_reports: list[dict[str, Any]] = []
    conformance_by_note: list[dict[str, Any]] = []
    plausibility_findings: list[dict[str, Any]] = []
    date_issue_count = 0
    measurement_issue_count = 0
    vital_issue_count = 0
    total_grounded = 0
    total_spans = 0

    for record_index, record in enumerate(records):
        spans = _extract_spans(record)
        note_text = _record_note_text(record)
        fields = _record_fields(record, spans)
        required = _record_required_fields(record, global_required)
        for field_name in required:
            fields.setdefault(field_name, None)
        missing_required = sorted(
            field_name for field_name in required if _is_null(fields.get(field_name))
        )
        grounded_for_note = 0
        fields_with_spans: dict[str, list[Mapping[str, Any]]] = {}
        for span in spans:
            domain = _span_field(span)
            fields_with_spans.setdefault(domain, []).append(span)
            grounded = _span_is_grounded(span, lookup)
            grounded_for_note += int(grounded)
            total_spans += 1
            total_grounded += int(grounded)
            domain_stat = domain_counts.setdefault(
                domain,
                {"total": 0, "grounded": 0, "ungrounded": 0},
            )
            domain_stat["total"] += 1
            domain_stat["grounded"] += int(grounded)
            domain_stat["ungrounded"] += int(not grounded)

        for field_name in set(fields) | set(fields_with_spans) | set(required):
            stat = field_counts.setdefault(
                field_name,
                {
                    "note_total": 0,
                    "present": 0,
                    "null": 0,
                    "span_total": 0,
                    "grounded": 0,
                    "ungrounded": 0,
                    "required": False,
                    "required_note_count": 0,
                    "missing_required": 0,
                },
            )
            stat["note_total"] += 1
            present = not _is_null(fields.get(field_name))
            stat["present"] += int(present)
            stat["null"] += int(not present)
            if field_name in required:
                stat["required"] = True
                stat["required_note_count"] += 1
                if not present:
                    stat["missing_required"] += 1
            for span in fields_with_spans.get(field_name, ()):
                grounded = _span_is_grounded(span, lookup)
                stat["span_total"] += 1
                stat["grounded"] += int(grounded)
                stat["ungrounded"] += int(not grounded)

        span_gate = _profile_span_integrity(record_index, spans, note_text)
        conformance_by_note.append(span_gate)
        findings, date_count, measurement_count, vital_count = _profile_plausibility(
            record_index,
            record,
            spans,
            fields,
            minimum_date,
            maximum_date,
            language=language,
        )
        plausibility_findings.extend(findings)
        date_issue_count += date_count
        measurement_issue_count += measurement_count
        vital_issue_count += vital_count

        required_score = (
            1.0
            if not required
            else _ratio(len(required) - len(missing_required), len(required))
        )
        grounding_score = (
            _ratio(grounded_for_note, len(spans)) if spans else (1.0 if fields else 0.0)
        )
        null_field_count = sum(int(_is_null(value)) for value in fields.values())
        field_count = len(fields)
        note_score = required_score * grounding_score
        note_reports.append(
            {
                "note_index": record_index,
                "completeness_score": note_score,
                "field_count": field_count,
                "null_field_count": null_field_count,
                "null_density": (
                    null_field_count / field_count if field_count else 0.0
                ),
                "required_field_count": len(required),
                "missing_required_field_count": len(missing_required),
                "missing_required_fields": missing_required,
                "missing_required_rate": (
                    len(missing_required) / len(required) if required else 0.0
                ),
                "span_count": len(spans),
                "grounded_span_count": grounded_for_note,
                "grounded_rate": grounding_score,
                "ungrounded_span_count": len(spans) - grounded_for_note,
                "ungrounded_rate": 1.0 - grounding_score,
                "passed": not missing_required and note_score >= floor,
            }
        )

    overall_score = (
        _ratio(
            sum(float(item["completeness_score"]) for item in note_reports),
            len(note_reports),
        )
        if note_reports
        else 0.0
    )
    grounding_by_domain = {
        domain: {
            **counts,
            "coverage": _ratio(counts["grounded"], counts["total"]),
            "grounded_rate": _ratio(counts["grounded"], counts["total"]),
            "passed": counts["ungrounded"] == 0,
        }
        for domain, counts in sorted(domain_counts.items())
    }
    for field_name, stat in field_counts.items():
        stat["null"] += len(records) - stat["note_total"]
        stat["note_total"] = len(records)
        stat["completeness"] = _ratio(stat["present"], stat["note_total"])
        stat["null_density"] = _ratio(stat["null"], stat["note_total"])
        stat["grounding_coverage"] = _ratio(stat["grounded"], stat["span_total"])
        stat["grounded_rate"] = stat["grounding_coverage"]
        stat["ungrounded_rate"] = _ratio(stat["ungrounded"], stat["span_total"])
        stat["missing_required_rate"] = _ratio(
            stat["missing_required"],
            stat["required_note_count"] or stat["note_total"],
        )
        stat["passed"] = stat["missing_required"] == 0

    completeness = {
        "overall_score": overall_score,
        "overall_completeness_score": overall_score,
        "note_count": len(records),
        "complete_note_count": sum(
            int(item["missing_required_field_count"] == 0) for item in note_reports
        ),
        "incomplete_note_count": sum(
            int(item["missing_required_field_count"] > 0) for item in note_reports
        ),
        "missing_required_field_count": sum(
            int(item["missing_required_field_count"]) for item in note_reports
        ),
        "per_note": note_reports,
        "per_field": {
            field_name: field_counts[field_name] for field_name in sorted(field_counts)
        },
    }
    conformance = _aggregate_conformance(conformance_by_note)
    plausibility = _build_plausibility_report(
        plausibility_findings,
        date_issue_count=date_issue_count,
        measurement_issue_count=measurement_issue_count,
        vital_issue_count=vital_issue_count,
    )
    gate_passed = overall_score >= floor
    checks = [
        {
            "name": "completeness_floor",
            "category": "completeness",
            "passed": gate_passed,
            "observed": overall_score,
            "threshold": floor,
        },
        {
            "name": "span_integrity",
            "category": "conformance",
            "passed": bool(conformance["passed"]),
            "invalid_count": conformance["invalid_spans"],
            "overlap_count": conformance["overlap_count"],
        },
        {
            "name": "clinical_plausibility",
            "category": "plausibility",
            "passed": bool(plausibility["passed"]),
            "invalid_count": plausibility["invalid_count"],
        },
    ]
    passed = all(bool(check["passed"]) for check in checks)
    payload: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "gate": {
            "completeness_floor": floor,
            "observed_completeness": overall_score,
            "passed": gate_passed,
        },
        "completeness": completeness,
        "grounding": {
            "total_spans": total_spans,
            "grounded_spans": total_grounded,
            "ungrounded_spans": total_spans - total_grounded,
            "coverage": _ratio(total_grounded, total_spans),
            "grounded_rate": _ratio(total_grounded, total_spans),
            "by_domain": grounding_by_domain,
        },
        "grounding_coverage": grounding_by_domain,
        "conformance": conformance,
        "plausibility": plausibility,
        "checks": checks,
        "provenance": {
            "local_only": True,
            "normalizers": [
                "openmed.clinical.lab_values",
                "openmed.clinical.units",
                "openmed.clinical.vital_signs",
                "openmed.core.quality_gates",
            ],
            "athena_index": {
                "provided": lookup is not None,
                "vocabulary_count": len(lookup.vocabulary_ids) if lookup else 0,
                "vocabulary_ids": list(lookup.vocabulary_ids) if lookup else [],
            },
            "date_bounds": {
                "minimum": minimum_date.isoformat(),
                "maximum": maximum_date.isoformat(),
            },
        },
    }
    payload["human_summary"] = _human_summary(payload)
    return QualityProfileReport(payload)


def profile_batch(
    results: Iterable[Any] | Mapping[str, Any],
    **kwargs: Any,
) -> QualityProfileReport:
    """Compatibility alias for profile_results."""

    return profile_results(results, **kwargs)


def profile_extracted_results(
    results: Iterable[Any] | Mapping[str, Any],
    **kwargs: Any,
) -> QualityProfileReport:
    """Compatibility alias for extraction-oriented callers."""

    return profile_results(results, **kwargs)


def profile(
    results: Iterable[Any] | Mapping[str, Any],
    **kwargs: Any,
) -> QualityProfileReport:
    """Short compatibility alias for profile_results."""

    return profile_results(results, **kwargs)


def profile_jsonl(source: str | Path, **kwargs: Any) -> QualityProfileReport:
    """Profile a JSONL string or local JSONL path."""

    return profile_results(_read_jsonl(source), **kwargs)


def render_human_summary(report: QualityProfileReport | Mapping[str, Any]) -> str:
    """Render the compact PHI-free summary from a report mapping."""

    if isinstance(report, QualityProfileReport):
        return report.human_summary
    return str(report.get("human_summary") or _human_summary(report))


def enforce_completeness_floor(
    results: Iterable[Any] | Mapping[str, Any],
    floor: float,
    *,
    report: QualityProfileReport | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> QualityProfileReport:
    """Profile and raise QualityGateError when a floor is not met."""

    current = (
        report
        if isinstance(report, QualityProfileReport)
        else QualityProfileReport(report)
        if isinstance(report, Mapping)
        else profile_results(results, completeness_floor=floor, **kwargs)
    )
    if current.overall_completeness_score < floor or not current.passed:
        raise QualityGateError(current)
    return current


def _normalise_records(
    results: Iterable[Any] | Mapping[str, Any] | QualityProfileReport,
) -> list[Mapping[str, Any]]:
    if isinstance(results, QualityProfileReport):
        return [results.to_dict()]
    if isinstance(results, Mapping):
        for key in ("records", "notes", "outputs", "items"):
            nested = results.get(key)
            if isinstance(nested, Sequence) and not isinstance(
                nested, (str, bytes, bytearray)
            ):
                return [_as_mapping(item) for item in nested]
        return [_as_mapping(results)]
    if isinstance(results, (str, bytes, bytearray)):
        raise TypeError("results must be mappings or an iterable of mappings")
    try:
        return [_as_mapping(item) for item in results]
    except TypeError as exc:
        raise TypeError("results must be mappings or an iterable of mappings") from exc


def _read_jsonl(source: str | Path) -> list[Mapping[str, Any]]:
    source_path: Path | None = None
    if isinstance(source, Path):
        source_path = source
    elif isinstance(source, str) and "\n" not in source and "\r" not in source:
        try:
            candidate = Path(source)
        except (OSError, ValueError):
            candidate = None
        if candidate is not None:
            try:
                if candidate.exists():
                    source_path = candidate
            except OSError:
                source_path = None
    if source_path is not None:
        try:
            lines = source_path.expanduser().read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError("quality input could not be read") from exc
    else:
        lines = str(source).splitlines()
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"quality JSONL line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"quality JSONL line {line_number} must be an object")
        records.append(value)
    if not records:
        raise ValueError("quality JSONL must contain at least one object")
    return records


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return attributes
    return {}


def _first_value(objects: Sequence[Any], keys: Sequence[str]) -> Any:
    for obj in objects:
        if isinstance(obj, Mapping):
            for key in keys:
                if key in obj:
                    return obj[key]
        else:
            for key in keys:
                value = getattr(obj, key, _MISSING)
                if value is not _MISSING:
                    return value
    return None


def _normalise_name(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _canonical_domain(value: Any) -> str:
    name = _normalise_name(value)
    if name in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[name]
    for suffix in ("_domain", "_label", "_type"):
        if name.endswith(suffix):
            candidate = name[: -len(suffix)]
            if candidate in _DOMAIN_ALIASES:
                return _DOMAIN_ALIASES[candidate]
    return name or "unknown"


def _canonical_field(value: Any) -> str:
    domain = _canonical_domain(value)
    return (
        domain
        if domain in {"condition", "drug", "measurement"}
        else (_normalise_name(value) or "unknown")
    )


def _iter_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _extract_spans(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    collected: list[Mapping[str, Any]] = []
    for key in _SPAN_KEYS:
        value = record.get(key, _MISSING)
        if value is _MISSING or value is None:
            continue
        for item in _iter_values(value):
            mapping = _as_mapping(item)
            if mapping:
                collected.append(mapping)
    if not collected and {"start", "end"} & set(record):
        collected.append(record)
    merged: list[Mapping[str, Any]] = []
    positions: dict[tuple[Any, ...], int] = {}
    for span in collected:
        start = span.get("start")
        end = span.get("end")
        label = _canonical_field(
            _first_value(
                (span,),
                ("field", "field_name", "slot", "domain", "domain_id", "label"),
            )
        )
        key = (start, end, label) if start is not None and end is not None else None
        if key is None or key not in positions:
            if key is not None:
                positions[key] = len(merged)
            merged.append(dict(span))
            continue
        index = positions[key]
        combined = dict(merged[index])
        for field_name, value in span.items():
            if field_name not in combined or _is_null(combined[field_name]):
                combined[field_name] = value
        merged[index] = combined
    return merged


def _record_note_text(record: Mapping[str, Any]) -> str | None:
    if not any(key in record for key in _SPAN_KEYS):
        value = record.get("text")
    else:
        value = _first_value(
            (record,), ("note_text", "document_text", "source_text", "text")
        )
    return value if isinstance(value, str) else None


def _record_fields(
    record: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in _FIELD_KEYS:
        value = record.get(key, _MISSING)
        if isinstance(value, Mapping):
            for field_name, field_value in value.items():
                fields[_canonical_field(field_name)] = field_value
            break
    for key, value in record.items():
        if key in _RESERVED_RECORD_KEYS:
            continue
        if _canonical_domain(key) in {"condition", "drug", "measurement"}:
            fields.setdefault(_canonical_field(key), value)
    for span in spans:
        field_name = _span_field(span)
        existing = fields.get(field_name, _MISSING)
        if existing is _MISSING or _is_null(existing):
            fields[field_name] = []
        elif not isinstance(existing, list):
            fields[field_name] = [existing]
        fields[field_name].append(span)
    return fields


def _record_required_fields(
    record: Mapping[str, Any],
    global_required: tuple[str, ...],
) -> tuple[str, ...]:
    value = record.get("required_fields", _MISSING)
    return (
        _normalise_required_fields(value) if value is not _MISSING else global_required
    )


def _normalise_required_fields(
    value: Iterable[str] | Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        values = [key for key, required in value.items() if required]
    elif isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    return tuple(
        sorted({_canonical_field(item) for item in values if str(item).strip()})
    )


def _span_field(span: Mapping[str, Any]) -> str:
    value = _first_value(
        (span,),
        (
            "field",
            "field_name",
            "slot",
            "domain",
            "domain_id",
            "canonical_label",
            "entity_label",
            "entity_group",
            "label",
            "type",
        ),
    )
    return _canonical_field(value)


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        if not value:
            return True
        if "value" in value and _is_null(value.get("value")):
            useful = set(value) - {
                "value",
                "grounded",
                "mapped",
                "concept_id",
                "standard_concept_id",
            }
            if not useful:
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) == 0
    return False


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return integer if integer > 0 and str(value).strip() not in {"0", "0.0"} else None


def _span_identifiers(span: Mapping[str, Any]) -> dict[str, Any]:
    codes: list[tuple[str, str]] = []
    concept_ids: set[int] = set()
    mappings: list[Mapping[str, Any]] = [span]
    for key in ("grounding", "coding", "concept", "mapping", "link"):
        value = span.get(key)
        if isinstance(value, Mapping):
            mappings.append(value)
            candidates = value.get("candidates")
            if isinstance(candidates, Sequence):
                mappings.extend(
                    item for item in candidates if isinstance(item, Mapping)
                )
    candidates = span.get("candidates")
    if isinstance(candidates, Sequence):
        mappings.extend(item for item in candidates if isinstance(item, Mapping))
    for mapping in mappings:
        concept_id = _positive_int(
            _first_value(
                (mapping,),
                (
                    "standard_concept_id",
                    "standardConceptId",
                    "target_concept_id",
                    "targetConceptId",
                    "concept_id",
                    "conceptId",
                    "code",
                ),
            )
        )
        if concept_id is not None:
            concept_ids.add(concept_id)
        code = _first_value(
            (mapping,),
            (
                "concept_code",
                "conceptCode",
                "source_code",
                "sourceCode",
                "code",
                "concept_id",
                "conceptId",
            ),
        )
        if code is None:
            continue
        code_text = str(code).strip()
        if not code_text or code_text.casefold() in {"none", "null", "unmapped"}:
            continue
        vocabulary = _normalise_name(
            _first_value(
                (mapping,),
                (
                    "vocabulary_id",
                    "vocabularyId",
                    "source_vocabulary_id",
                    "system",
                    "code_system",
                ),
            )
        )
        codes.append((vocabulary, _normalise_name(code_text)))
    return {
        "codes": tuple(dict.fromkeys(codes)),
        "concept_ids": tuple(sorted(concept_ids)),
    }


def _span_is_grounded(
    span: Mapping[str, Any],
    lookup: _AthenaLookup | None,
) -> bool:
    if bool(span.get("abstained", False)):
        return False
    explicit = span.get("grounded", _MISSING)
    if isinstance(explicit, bool):
        if not explicit:
            return False
        if lookup is None:
            return True
    if lookup is not None:
        return lookup.standard_match(span)
    for key in (
        "standard_concept_id",
        "standardConceptId",
        "target_concept_id",
        "targetConceptId",
        "concept_id",
        "conceptId",
        "code",
        "concept_code",
    ):
        value = span.get(key)
        if _positive_int(value) is not None or (
            isinstance(value, str)
            and value.strip()
            and value.casefold() not in {"unmapped", "none", "null", "0"}
        ):
            return True
    for key in ("grounding", "coding", "mapping", "concept"):
        nested = span.get(key)
        if isinstance(nested, Mapping) and _span_is_grounded(nested, None):
            return True
    candidates = span.get("candidates")
    return isinstance(candidates, Sequence) and any(
        isinstance(item, Mapping) and _span_is_grounded(item, None)
        for item in candidates
    )


def _is_standard_concept(record: Mapping[str, Any]) -> bool:
    value = _first_value((record,), ("standard_concept", "standardConcept"))
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {
        "s",
        "standard",
        "true",
        "1",
        "y",
    }


def _record_matches_domain(record: Mapping[str, Any], span_domain: str) -> bool:
    """Match optional Athena domain metadata without guessing missing data."""

    raw_domain = _first_value(
        (record,), ("domain_id", "domain", "omop_domain", "domainId")
    )
    if raw_domain is None or not str(raw_domain).strip():
        return True
    record_domain = _canonical_domain(raw_domain)
    return span_domain in {"unknown", record_domain} or record_domain == "unknown"


def _athena_record_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(_first_value((record,), ("concept_id", "conceptId")) or ""),
        str(_first_value((record,), ("concept_code", "code")) or ""),
        str(_first_value((record,), ("vocabulary_id", "vocabularyId")) or ""),
    )


def _load_athena_lookup(
    source: Mapping[str, Any] | str | Path | None,
) -> _AthenaLookup | None:
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        from openmed.interop.athena import load_athena_vocab

        source = load_athena_vocab(source)
    if not isinstance(source, Mapping):
        raise TypeError("athena_index must be a mapping or local export path")
    return _AthenaLookup.from_index(source)


def _profile_span_integrity(
    record_index: int,
    spans: Sequence[Mapping[str, Any]],
    note_text: str | None,
) -> dict[str, Any]:
    if not spans:
        return {
            "record_index": record_index,
            "total_spans": 0,
            "valid_spans": 0,
            "invalid_spans": 0,
            "offsetless_spans": 0,
            "overlap_count": 0,
            "overlaps_resolved": 0,
            "residual_overlaps": 0,
            "invalid_offsets": [],
            "overlap_offsets": [],
            "passed": True,
        }
    adapters: list[_SpanAdapter] = []
    max_end = 0
    for span in spans:
        start = span.get("start")
        end = span.get("end")
        if type(end) is int and end > max_end:
            max_end = end
        raw_text = span.get("text")
        if not isinstance(raw_text, str):
            if (
                isinstance(note_text, str)
                and type(start) is int
                and type(end) is int
                and 0 <= start <= end <= len(note_text)
            ):
                raw_text = note_text[start:end]
            elif type(start) is int and type(end) is int:
                raw_text = "x" * max(0, end - start)
            else:
                raw_text = ""
        adapters.append(
            _SpanAdapter(
                text=raw_text,
                label=str(
                    _first_value(
                        (span,),
                        ("label", "entity_label", "entity_group", "domain_id"),
                    )
                    or "unknown"
                ),
                start=start,
                end=end,
                confidence=span.get("confidence"),
                score=span.get("score"),
                metadata={},
            )
        )
    check_text = note_text if isinstance(note_text, str) else "x" * max_end
    result = validate_entity_spans_strict(adapters, check_text)
    invalid_offsets = [
        {
            "span_index": issue.index,
            "start": issue.start,
            "end": issue.end,
            "label": issue.label,
            "problem_count": len(issue.problems),
        }
        for issue in result.offending_spans
    ]
    overlap_offsets = []
    for finding in result.overlap_findings:
        overlap_offsets.append(
            {
                "first": {
                    "start": finding.first.get("start"),
                    "end": finding.first.get("end"),
                    "label": finding.first.get("label"),
                },
                "second": {
                    "start": finding.second.get("start"),
                    "end": finding.second.get("end"),
                    "label": finding.second.get("label"),
                },
            }
        )
    return {
        "record_index": record_index,
        "total_spans": result.total_spans,
        "valid_spans": result.valid_spans,
        "invalid_spans": result.invalid_spans,
        "offsetless_spans": result.offsetless_spans,
        "overlap_count": len(result.overlap_findings),
        "overlaps_resolved": result.overlaps_resolved,
        "residual_overlaps": result.residual_overlaps,
        "invalid_offsets": invalid_offsets,
        "overlap_offsets": overlap_offsets,
        "passed": result.passed and not result.overlap_findings,
    }


def _aggregate_conformance(
    per_note: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    invalid_offsets = [
        offset for item in per_note for offset in item.get("invalid_offsets", [])
    ]
    overlap_offsets = [
        offset for item in per_note for offset in item.get("overlap_offsets", [])
    ]
    total = sum(int(item["total_spans"]) for item in per_note)
    valid = sum(int(item["valid_spans"]) for item in per_note)
    invalid = sum(int(item["invalid_spans"]) for item in per_note)
    offsetless = sum(int(item["offsetless_spans"]) for item in per_note)
    overlaps = sum(int(item["overlap_count"]) for item in per_note)
    return {
        "passed": invalid == 0 and overlaps == 0,
        "total_spans": total,
        "valid_spans": valid,
        "invalid_spans": invalid,
        "offsetless_spans": offsetless,
        "overlap_count": overlaps,
        "overlaps_resolved": sum(int(item["overlaps_resolved"]) for item in per_note),
        "residual_overlaps": sum(int(item["residual_overlaps"]) for item in per_note),
        "offset_validity": _ratio(valid, total - offsetless),
        "invalid_offsets": invalid_offsets,
        "overlap_offsets": overlap_offsets,
        "per_note": list(per_note),
        "checks": [
            {
                "name": "offset_sanity",
                "passed": invalid == 0,
                "invalid_count": invalid,
                "total": total,
            },
            {
                "name": "overlap_sanity",
                "passed": overlaps == 0,
                "overlap_count": overlaps,
                "total": total,
            },
        ],
    }


def _profile_plausibility(
    record_index: int,
    record: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
    fields: Mapping[str, Any],
    minimum_date: date,
    maximum_date: date,
    *,
    language: object | None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    findings: list[dict[str, Any]] = []
    date_count = 0
    measurement_count = 0
    vital_count = 0
    for path, value in _iter_date_candidates(record, record_index):
        if not _date_is_valid(value, minimum_date, maximum_date):
            findings.append(
                {
                    "kind": "date",
                    "path": path,
                    "reason": "out_of_range_or_invalid",
                }
            )
            date_count += 1
    for span_index, span in enumerate(_plausibility_spans(spans, fields)):
        path = f"record[{record_index}].span[{span_index}]"
        for finding in _measurement_findings(span, path, language=language):
            findings.append(finding)
            measurement_count += 1
        for finding in _vital_findings(span, path, language=language):
            findings.append(finding)
            vital_count += 1
    return findings, date_count, measurement_count, vital_count


def _plausibility_spans(
    spans: Sequence[Mapping[str, Any]],
    fields: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Expose field-level measurements through the shared normalizer path."""

    candidates = list(spans)
    span_keys = {
        (_span_field(span), span.get("start"), span.get("end"))
        for span in spans
        if span.get("start") is not None and span.get("end") is not None
    }
    for field_name, value in fields.items():
        if _canonical_domain(field_name) != "measurement":
            continue
        for item in _iter_values(value):
            mapping = _as_mapping(item)
            candidate = dict(mapping) if mapping else {"value": item}
            candidate.setdefault("field", field_name)
            candidate_key = (
                _span_field(candidate),
                candidate.get("start"),
                candidate.get("end"),
            )
            if (
                candidate.get("start") is not None
                and candidate.get("end") is not None
                and candidate_key in span_keys
            ):
                continue
            candidates.append(candidate)
    return candidates


def _iter_date_candidates(
    record: Mapping[str, Any],
    record_index: int,
) -> Iterator[tuple[str, Any]]:
    def visit(
        value: Any,
        path: str,
        *,
        inspect_nested: bool = False,
    ) -> Iterator[tuple[str, Any]]:
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = _normalise_name(key)
                child_path = f"{path}.{_canonical_field(key)}"
                if name in _DATE_KEYS:
                    if isinstance(child, Mapping):
                        yield from visit(child, child_path, inspect_nested=True)
                    elif isinstance(child, Sequence) and not isinstance(
                        child, (str, bytes, bytearray)
                    ):
                        yield from visit(child, child_path, inspect_nested=True)
                    else:
                        yield child_path, child
                elif (
                    name in {"dates", "temporal", "timing"}
                    or name in _SPAN_KEYS
                    or name in _FIELD_KEYS
                    or inspect_nested
                ):
                    yield from visit(child, child_path, inspect_nested=True)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                yield from visit(
                    child,
                    f"{path}[{index}]",
                    inspect_nested=inspect_nested,
                )

    yield from visit(record, f"record[{record_index}]")


def _coerce_date_bound(value: date | str, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO date") from exc
    raise TypeError(f"{name} must be a date or ISO date string")


def _date_is_valid(value: Any, minimum: date, maximum: date) -> bool:
    if isinstance(value, datetime):
        candidate = value.date()
    elif isinstance(value, date):
        candidate = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            candidate = date.fromisoformat(text[:10])
        except ValueError:
            try:
                candidate = datetime.fromisoformat(text.replace("Z", "+00:00")).date()
            except ValueError:
                return False
    else:
        return False
    return minimum <= candidate <= maximum


def _measurement_findings(
    span: Mapping[str, Any],
    path: str,
    *,
    language: object | None,
) -> list[dict[str, Any]]:
    domain = _span_field(span)
    has_value = any(
        key in span
        for key in (
            "value",
            "numeric_value",
            "magnitude",
            "value_as_number",
            "result",
            "measurement",
        )
    )
    if domain != "measurement" and not has_value:
        return []
    value = _first_value(
        (span,),
        (
            "value_as_number",
            "numeric_value",
            "magnitude",
            "value",
            "result",
            "measurement",
        ),
    )
    unit = _first_value((span,), ("unit", "units", "value_unit"))
    reference_range = _first_value(
        (span,),
        ("reference_range", "referenceRange", "ref_range", "range"),
    )
    explicit_flag = _first_value(
        (span,),
        ("abnormal_flag", "abnormalFlag", "flag"),
    )
    if value is None:
        return []
    findings: list[dict[str, Any]] = []
    if unit is not None or isinstance(value, str):
        normalized = parse_measurement(value, unit, language=language)
        if normalized.get("status") != "ok":
            if unit is not None or reference_range is not None:
                return [
                    {
                        "kind": "measurement",
                        "path": path,
                        "reason": "value_or_unit_not_normalizable",
                    }
                ]
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            numeric = math.nan
        if not math.isfinite(numeric):
            return [
                {
                    "kind": "measurement",
                    "path": path,
                    "reason": "value_not_finite",
                }
            ]
    if reference_range is not None:
        flag = derive_abnormal_flag(
            value,
            reference_range,
            explicit_flag if isinstance(explicit_flag, str) else None,
            value_unit=unit,
            language=language,
        )
        if flag in {"low", "high", "critical"}:
            findings.append(
                {
                    "kind": "measurement",
                    "path": path,
                    "reason": "outside_reference_range",
                }
            )
    return findings


def _vital_findings(
    span: Mapping[str, Any],
    path: str,
    *,
    language: object | None,
) -> list[dict[str, Any]]:
    raw_vital = span.get("vital_sign")
    vital = dict(raw_vital) if isinstance(raw_vital, Mapping) else {}
    for key in ("kind", "value", "unit", "components"):
        if key not in vital and key in span:
            vital[key] = span[key]
    if not vital.get("kind"):
        parsed = structure_vital_sign(span.get("text"), language=language)
        if parsed.get("kind") != "unknown":
            vital = dict(parsed)
    kind = _normalise_name(vital.get("kind"))
    if kind not in {"blood_pressure", *_VITAL_TARGET_UNITS}:
        return []
    findings: list[dict[str, Any]] = []
    components = vital.get("components")
    if kind == "blood_pressure" and isinstance(components, Sequence):
        for component in components:
            if not isinstance(component, Mapping):
                continue
            component_kind = _normalise_name(component.get("kind"))
            finding = _check_vital_value(
                component.get("value"),
                component.get("unit") or vital.get("unit"),
                component_kind,
                path,
                language=language,
            )
            if finding is not None:
                findings.append(finding)
        return findings
    finding = _check_vital_value(
        vital.get("value"),
        vital.get("unit"),
        kind,
        path,
        language=language,
    )
    return [finding] if finding is not None else []


def _check_vital_value(
    value: Any,
    unit: Any,
    kind: str,
    path: str,
    *,
    language: object | None,
) -> dict[str, Any] | None:
    if kind not in _VITAL_LIMITS or value is None:
        return None
    numeric: float | None
    if unit:
        normalized = normalize_to(
            value,
            unit,
            _VITAL_TARGET_UNITS[kind],
            language=language,
        )
        if normalized.get("status") != "ok":
            return {
                "kind": "vital_sign",
                "path": path,
                "reason": "value_or_unit_not_normalizable",
            }
        numeric = normalized.get("magnitude")
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            numeric = None
    if numeric is None or not math.isfinite(float(numeric)):
        return {
            "kind": "vital_sign",
            "path": path,
            "reason": "value_not_finite",
        }
    minimum, maximum = _VITAL_LIMITS[kind]
    if not minimum <= float(numeric) <= maximum:
        return {
            "kind": "vital_sign",
            "path": path,
            "reason": "outside_plausible_range",
        }
    return None


def _build_plausibility_report(
    findings: Sequence[Mapping[str, Any]],
    *,
    date_issue_count: int,
    measurement_issue_count: int,
    vital_issue_count: int,
) -> dict[str, Any]:
    by_kind = Counter(str(item["kind"]) for item in findings)
    checks = [
        {
            "name": "date_range",
            "passed": date_issue_count == 0,
            "invalid_count": date_issue_count,
        },
        {
            "name": "laboratory_values",
            "passed": measurement_issue_count == 0,
            "invalid_count": measurement_issue_count,
        },
        {
            "name": "vital_signs",
            "passed": vital_issue_count == 0,
            "invalid_count": vital_issue_count,
        },
    ]
    return {
        "passed": not findings,
        "invalid_count": len(findings),
        "invalid_findings": list(findings),
        "findings": list(findings),
        "by_kind": dict(sorted(by_kind.items())),
        "checks": checks,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return 1.0 if denominator == 0 else float(numerator) / float(denominator)


def _resolve_floor(completeness_floor: float, quality_floor: float | None) -> float:
    value = completeness_floor if quality_floor is None else quality_floor
    try:
        floor = float(value)
    except (TypeError, ValueError):
        raise ValueError("completeness_floor must be a finite number between 0 and 1")
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("completeness_floor must be a finite number between 0 and 1")
    return floor


def _human_summary(payload: Mapping[str, Any]) -> str:
    completeness = payload["completeness"]
    grounding = payload["grounding"]
    conformance = payload["conformance"]
    plausibility = payload["plausibility"]
    gate = payload["gate"]
    return "\n".join(
        (
            f"Clinical data-quality profile: {str(payload['status']).upper()}",
            f"Records: {completeness['note_count']}",
            "Completeness: "
            f"{float(completeness['overall_score']):.6f} "
            f"(floor {float(gate['completeness_floor']):.6f})",
            "Grounding: "
            f"{grounding['grounded_spans']}/{grounding['total_spans']} spans "
            f"({float(grounding['coverage']):.6f})",
            "Conformance: "
            f"{'PASS' if conformance['passed'] else 'FAIL'} "
            f"({conformance['invalid_spans']} invalid, "
            f"{conformance['overlap_count']} overlaps)",
            "Plausibility: "
            f"{'PASS' if plausibility['passed'] else 'FAIL'} "
            f"({plausibility['invalid_count']} flagged)",
            f"Gate: {'PASS' if gate['passed'] else 'FAIL'}",
        )
    )


def _json_safe_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


__all__ = [
    "DEFAULT_DATE_MAX",
    "DEFAULT_DATE_MIN",
    "PROFILE_SCHEMA_VERSION",
    "QualityGateError",
    "QualityProfileReport",
    "enforce_completeness_floor",
    "profile",
    "profile_batch",
    "profile_extracted_results",
    "profile_jsonl",
    "profile_results",
    "render_human_summary",
]
