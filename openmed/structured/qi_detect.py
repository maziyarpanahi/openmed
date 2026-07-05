"""Automatic quasi-identifier detection for tabular datasets.

The detector samples bounded rows from CSV, JSONL/NDJSON, and Parquet files,
profiles column-level distributions, and ranks candidate quasi-identifier sets
by risk-report-compatible equivalence-class fragmentation. Emitted manifests
contain column names, aggregate counts, and hashes only; raw cell values are
never included in evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from openmed.core.labels import (
    AGE,
    API_KEY,
    CLINICAL_CONCEPT,
    CONDITION,
    DATE,
    DATE_OF_BIRTH,
    DIRECT_IDENTIFIER,
    EMAIL,
    GENDER,
    ID_NUM,
    LAB_TEST,
    LOCATION,
    MEDICATION,
    ORGANIZATION,
    OTHER,
    PASSWORD,
    PERSON,
    PHONE,
    PROCEDURE,
    QUASI_IDENTIFIER,
    SSN,
    STREET_ADDRESS,
    USERNAME,
    ZIPCODE,
    normalize_label,
    policy_label_for,
    risk_level_for,
    system_hints_for,
)
from openmed.risk import quasi_identifier_key

ROLE_DIRECT_ID = "direct-id"
ROLE_QUASI_ID = "quasi-id"
ROLE_SENSITIVE = "sensitive"
ROLE_SAFE = "safe"

SUPPORTED_SUFFIXES = frozenset({".csv", ".tsv", ".jsonl", ".ndjson", ".parquet"})
DEFAULT_MAX_ROWS = 10_000
DEFAULT_BATCH_SIZE = 4_096
DEFAULT_MAX_SET_SIZE = 4
DEFAULT_MAX_CANDIDATE_COLUMNS = 8

_HEADER_LABELS = {
    "name": PERSON,
    "fullname": PERSON,
    "patientname": PERSON,
    "membername": PERSON,
    "mrn": ID_NUM,
    "medicalrecordnumber": ID_NUM,
    "patientid": ID_NUM,
    "memberid": ID_NUM,
    "recordid": ID_NUM,
    "subjectid": ID_NUM,
    "identifier": ID_NUM,
    "ssn": SSN,
    "email": EMAIL,
    "emailaddress": EMAIL,
    "phone": PHONE,
    "telephone": PHONE,
    "username": USERNAME,
    "password": PASSWORD,
    "apikey": API_KEY,
    "address": STREET_ADDRESS,
    "streetaddress": STREET_ADDRESS,
    "dob": DATE_OF_BIRTH,
    "dateofbirth": DATE_OF_BIRTH,
    "birthdate": DATE_OF_BIRTH,
    "age": AGE,
    "patientage": AGE,
    "zip": ZIPCODE,
    "zipcode": ZIPCODE,
    "postalcode": ZIPCODE,
    "city": LOCATION,
    "county": LOCATION,
    "state": LOCATION,
    "region": LOCATION,
    "location": LOCATION,
    "admitdate": DATE,
    "admissiondate": DATE,
    "dischargedate": DATE,
    "encounterdate": DATE,
    "servicedate": DATE,
    "visitdate": DATE,
    "appointmentdate": DATE,
    "eventdate": DATE,
    "date": DATE,
    "sex": GENDER,
    "gender": GENDER,
    "provider": ORGANIZATION,
    "hospital": ORGANIZATION,
    "facility": ORGANIZATION,
    "clinic": ORGANIZATION,
    "diagnosis": CONDITION,
    "diagnosiscode": CONDITION,
    "dx": CONDITION,
    "raredx": CONDITION,
    "rarediagnosis": CONDITION,
    "condition": CONDITION,
    "disease": CONDITION,
    "medication": MEDICATION,
    "procedure": PROCEDURE,
    "lab": LAB_TEST,
    "labtest": LAB_TEST,
}

_DIRECT_VALUE_PATTERNS = (
    (SSN, re.compile(r"^\s*\d{3}-\d{2}-\d{4}\s*$"), 0.5),
    (EMAIL, re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"), 0.5),
    (
        PHONE,
        re.compile(
            r"^\s*(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)"
            r"\d{3}[\s.-]?\d{4}\s*$"
        ),
        0.6,
    ),
    (ID_NUM, re.compile(r"^\s*(?:MRN|MEDREC|MR)[\s:._-]*[A-Z0-9-]{3,}\s*$", re.I), 0.4),
)
_QI_VALUE_PATTERNS = (
    (
        DATE,
        re.compile(r"^\s*(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*$"),
        0.6,
    ),
    (ZIPCODE, re.compile(r"^\s*\d{5}(?:-\d{4})?\s*$"), 0.6),
    (AGE, re.compile(r"^\s*(?:1[01]\d|[1-9]?\d)\s*$"), 0.8),
)


@dataclass(frozen=True)
class _TableSample:
    path: Path
    format: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    max_rows: int
    source_rows: int | None = None


@dataclass(frozen=True)
class _ColumnProfile:
    name: str
    role: str
    confidence: float
    non_null_count: int
    null_count: int
    cardinality: int
    uniqueness_ratio: float
    dominant_value_ratio: float
    singleton_value_count: int
    canonical_label: str | None
    policy_label: str | None
    risk_level: str | None
    system_hints: tuple[str, ...]
    evidence: tuple[str, ...]

    def to_manifest(self, *, sampled_rows: int) -> dict[str, Any]:
        return {
            "role": self.role,
            "confidence": round(self.confidence, 6),
            "canonical_label": self.canonical_label,
            "policy_label": self.policy_label,
            "risk_level": self.risk_level,
            "system_hints": list(self.system_hints),
            "profile": {
                "sampled_rows": sampled_rows,
                "non_null_count": self.non_null_count,
                "null_count": self.null_count,
                "cardinality": self.cardinality,
                "uniqueness_ratio": round(self.uniqueness_ratio, 6),
                "dominant_value_ratio": round(self.dominant_value_ratio, 6),
                "singleton_value_count": self.singleton_value_count,
            },
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class _SetStats:
    columns: tuple[str, ...]
    keys: tuple[bytes, ...]
    counts: Counter[bytes]
    class_ratio: float
    singleton_ratio: float
    singleton_count: int
    min_class_size: int


def scan_table(
    path: str | Path,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_set_size: int = DEFAULT_MAX_SET_SIZE,
    max_candidate_columns: int = DEFAULT_MAX_CANDIDATE_COLUMNS,
) -> dict[str, Any]:
    """Profile a table and emit a PHI-safe quasi-identifier manifest.

    Args:
        path: CSV, TSV, JSONL/NDJSON, or Parquet file path.
        max_rows: Maximum rows to sample. Readers stop once this budget is met.
        batch_size: Parquet row-batch size. Also capped by ``max_rows``.
        max_set_size: Largest candidate quasi-identifier set to score.
        max_candidate_columns: Maximum role-eligible columns considered for
            combination scoring.

    Returns:
        A manifest with per-column roles and ranked quasi-identifier sets.
    """

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_set_size <= 0:
        raise ValueError("max_set_size must be positive")
    if max_candidate_columns <= 0:
        raise ValueError("max_candidate_columns must be positive")

    sample = _read_table_sample(Path(path), max_rows=max_rows, batch_size=batch_size)
    profiles = _profile_columns(sample.columns, sample.rows)
    qi_sets = _rank_quasi_identifier_sets(
        sample.rows,
        profiles,
        max_set_size=max_set_size,
        max_candidate_columns=max_candidate_columns,
    )
    confidence = max(
        (entry["confidence"] for entry in qi_sets),
        default=max((profile.confidence for profile in profiles), default=0.0),
    )

    return {
        "format": sample.format,
        "path": str(sample.path),
        "sample": {
            "sampled_rows": len(sample.rows),
            "max_rows": sample.max_rows,
            "source_rows": sample.source_rows,
            "bounded": sample.source_rows is None
            or sample.source_rows > len(sample.rows),
        },
        "columns": {
            profile.name: profile.to_manifest(sampled_rows=len(sample.rows))
            for profile in profiles
        },
        "column_roles": {profile.name: profile.role for profile in profiles},
        "quasi_identifier_sets": qi_sets,
        "confidence": round(confidence, 6),
    }


def _read_table_sample(path: Path, *, max_rows: int, batch_size: int) -> _TableSample:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported table format {suffix!r}; expected {supported}")
    if suffix in {".csv", ".tsv"}:
        return _read_delimited_sample(
            path, max_rows=max_rows, delimiter="\t" if suffix == ".tsv" else ","
        )
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl_sample(path, max_rows=max_rows)
    return _read_parquet_sample(path, max_rows=max_rows, batch_size=batch_size)


def _read_delimited_sample(
    path: Path,
    *,
    max_rows: int,
    delimiter: str,
) -> _TableSample:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = tuple(str(field) for field in (reader.fieldnames or ()))
        for row in reader:
            rows.append({key: value for key, value in row.items() if key is not None})
            if len(rows) >= max_rows:
                break
    return _TableSample(
        path=path,
        format="tsv" if delimiter == "\t" else "csv",
        columns=columns,
        rows=tuple(rows),
        max_rows=max_rows,
    )


def _read_jsonl_sample(path: Path, *, max_rows: int) -> _TableSample:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row {line_number} must be an object")
            row = {str(key): value for key, value in payload.items()}
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
            rows.append(row)
            if len(rows) >= max_rows:
                break
    return _TableSample(
        path=path,
        format="jsonl",
        columns=tuple(columns),
        rows=tuple(rows),
        max_rows=max_rows,
    )


def _read_parquet_sample(
    path: Path,
    *,
    max_rows: int,
    batch_size: int,
) -> _TableSample:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet quasi-identifier detection requires pyarrow. "
            "Install openmed[columnar] or install pyarrow directly."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    columns = tuple(str(name) for name in schema.names)
    source_rows = getattr(getattr(parquet_file, "metadata", None), "num_rows", None)
    rows: list[dict[str, Any]] = []
    resolved_batch_size = min(batch_size, max_rows)

    for record_batch in parquet_file.iter_batches(batch_size=resolved_batch_size):
        for row in record_batch.to_pylist():
            if isinstance(row, Mapping):
                rows.append({str(key): value for key, value in row.items()})
            else:
                raise ValueError("Parquet batches must yield row mappings")
            if len(rows) >= max_rows:
                return _TableSample(
                    path=path,
                    format="parquet",
                    columns=columns,
                    rows=tuple(rows),
                    max_rows=max_rows,
                    source_rows=int(source_rows) if source_rows is not None else None,
                )

    return _TableSample(
        path=path,
        format="parquet",
        columns=columns,
        rows=tuple(rows),
        max_rows=max_rows,
        source_rows=int(source_rows) if source_rows is not None else None,
    )


def _profile_columns(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[_ColumnProfile, ...]:
    return tuple(_profile_column(column, rows) for column in columns)


def _profile_column(column: str, rows: Sequence[Mapping[str, Any]]) -> _ColumnProfile:
    values = [_cell_text(row.get(column)) for row in rows]
    non_empty = [value for value in values if value]
    counts = Counter(_distribution_value(value) for value in non_empty)
    cardinality = len(counts)
    non_null_count = len(non_empty)
    null_count = len(values) - non_null_count
    uniqueness_ratio = _rate(cardinality, non_null_count)
    dominant_value_ratio = _rate(max(counts.values(), default=0), non_null_count)
    singleton_value_count = sum(1 for count in counts.values() if count == 1)

    label, source, source_confidence = _label_for_column(column, non_empty)
    policy_label = policy_label_for(label) if label is not None else None
    risk_level = risk_level_for(label) if label is not None else None
    system_hints = system_hints_for(label) if label is not None else ()
    role = _role_for_column(
        column,
        label=label,
        policy_label=policy_label,
        uniqueness_ratio=uniqueness_ratio,
        singleton_value_count=singleton_value_count,
    )
    confidence = _role_confidence(
        role,
        source_confidence=source_confidence,
        uniqueness_ratio=uniqueness_ratio,
        singleton_value_count=singleton_value_count,
        non_null_count=non_null_count,
    )
    evidence = _column_evidence(
        source=source,
        label=label,
        policy_label=policy_label,
        risk_level=risk_level,
        system_hints=system_hints,
        uniqueness_ratio=uniqueness_ratio,
        singleton_value_count=singleton_value_count,
        non_null_count=non_null_count,
        role=role,
    )

    return _ColumnProfile(
        name=column,
        role=role,
        confidence=confidence,
        non_null_count=non_null_count,
        null_count=null_count,
        cardinality=cardinality,
        uniqueness_ratio=uniqueness_ratio,
        dominant_value_ratio=dominant_value_ratio,
        singleton_value_count=singleton_value_count,
        canonical_label=label,
        policy_label=policy_label,
        risk_level=risk_level,
        system_hints=system_hints,
        evidence=evidence,
    )


def _label_for_column(
    column: str,
    values: Sequence[str],
) -> tuple[str | None, str, float]:
    header_key = _name_key(column)
    if header_key in _HEADER_LABELS:
        return _HEADER_LABELS[header_key], "header_name", 0.9

    canonical = normalize_label(column)
    if canonical != OTHER:
        return canonical, "label_taxonomy", 0.84

    for label, pattern, threshold in (*_DIRECT_VALUE_PATTERNS, *_QI_VALUE_PATTERNS):
        if not values:
            continue
        matches = sum(1 for value in values if pattern.fullmatch(value))
        ratio = _rate(matches, len(values))
        if matches and ratio >= threshold:
            return label, "value_sample", min(0.9, 0.6 + ratio * 0.3)

    return None, "statistics", 0.62


def _role_for_column(
    column: str,
    *,
    label: str | None,
    policy_label: str | None,
    uniqueness_ratio: float,
    singleton_value_count: int,
) -> str:
    if policy_label == DIRECT_IDENTIFIER:
        return ROLE_DIRECT_ID
    if policy_label == QUASI_IDENTIFIER:
        return ROLE_QUASI_ID
    if policy_label == CLINICAL_CONCEPT:
        if _rare_clinical_column(column) and (
            singleton_value_count > 0 or uniqueness_ratio >= 0.25
        ):
            return ROLE_QUASI_ID
        return ROLE_SENSITIVE
    if label is None and _name_key(column).endswith("id") and uniqueness_ratio >= 0.9:
        return ROLE_DIRECT_ID
    if label is None and uniqueness_ratio >= 0.75 and singleton_value_count > 0:
        return ROLE_QUASI_ID
    return ROLE_SAFE


def _role_confidence(
    role: str,
    *,
    source_confidence: float,
    uniqueness_ratio: float,
    singleton_value_count: int,
    non_null_count: int,
) -> float:
    if role == ROLE_SAFE:
        return min(0.95, max(0.7, source_confidence))
    statistical_boost = min(0.15, uniqueness_ratio * 0.15)
    if singleton_value_count:
        statistical_boost += 0.04
    if non_null_count >= 10:
        statistical_boost += 0.03
    return min(0.99, source_confidence + statistical_boost)


def _column_evidence(
    *,
    source: str,
    label: str | None,
    policy_label: str | None,
    risk_level: str | None,
    system_hints: Sequence[str],
    uniqueness_ratio: float,
    singleton_value_count: int,
    non_null_count: int,
    role: str,
) -> tuple[str, ...]:
    evidence = [
        f"source={source}",
        f"non_null_count={non_null_count}",
        f"uniqueness_ratio={uniqueness_ratio:.6f}",
        f"singleton_value_count={singleton_value_count}",
    ]
    if label is not None:
        evidence.append(f"canonical_label={label}")
    if policy_label is not None:
        evidence.append(f"policy_label={policy_label}")
    if risk_level is not None:
        evidence.append(f"risk_level={risk_level}")
    if system_hints:
        evidence.append(f"system_hints_count={len(system_hints)}")
    if role == ROLE_QUASI_ID and policy_label == CLINICAL_CONCEPT:
        evidence.append("clinical_column_fragments_equivalence_classes")
    return tuple(evidence)


def _rank_quasi_identifier_sets(
    rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[_ColumnProfile],
    *,
    max_set_size: int,
    max_candidate_columns: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    candidate_columns = _candidate_columns(profiles, limit=max_candidate_columns)
    if not candidate_columns:
        return []

    stats_by_columns: dict[tuple[str, ...], _SetStats] = {}
    ranked: list[dict[str, Any]] = []
    max_size = min(max_set_size, len(candidate_columns))

    for size in range(1, max_size + 1):
        for combo in combinations(candidate_columns, size):
            stats = _set_stats(rows, combo)
            if not stats.keys:
                continue
            stats_by_columns[combo] = stats
            if stats.singleton_count == 0 and stats.class_ratio < 0.2:
                continue
            ranked.append(
                _qi_set_manifest(
                    stats,
                    profiles=profiles,
                    prior_stats=stats_by_columns,
                )
            )

    ranked.sort(
        key=lambda item: (
            item["score"],
            item["singleton_count"],
            item["equivalence_class_count"],
            len(item["columns"]),
        ),
        reverse=True,
    )
    return ranked


def _candidate_columns(
    profiles: Sequence[_ColumnProfile],
    *,
    limit: int,
) -> tuple[str, ...]:
    candidates = [
        profile
        for profile in profiles
        if profile.non_null_count
        and (
            profile.role == ROLE_QUASI_ID
            or (
                profile.role == ROLE_SENSITIVE
                and profile.singleton_value_count > 0
                and profile.uniqueness_ratio >= 0.05
            )
        )
    ]
    candidates.sort(
        key=lambda profile: (
            profile.role == ROLE_QUASI_ID,
            profile.confidence,
            profile.uniqueness_ratio,
            profile.cardinality,
        ),
        reverse=True,
    )
    return tuple(profile.name for profile in candidates[:limit])


def _set_stats(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> _SetStats:
    keys = tuple(_risk_key_bytes(row, columns) for row in rows)
    non_empty_keys = tuple(key for key in keys if key != b"[]")
    counts = Counter(non_empty_keys)
    singleton_count = sum(1 for key in non_empty_keys if counts[key] == 1)
    return _SetStats(
        columns=tuple(columns),
        keys=non_empty_keys,
        counts=counts,
        class_ratio=_rate(len(counts), len(rows)),
        singleton_ratio=_rate(singleton_count, len(rows)),
        singleton_count=singleton_count,
        min_class_size=min(counts.values(), default=0),
    )


def _qi_set_manifest(
    stats: _SetStats,
    *,
    profiles: Sequence[_ColumnProfile],
    prior_stats: Mapping[tuple[str, ...], _SetStats],
) -> dict[str, Any]:
    profile_by_name = {profile.name: profile for profile in profiles}
    best_subset_singleton = 0.0
    best_subset_class = 0.0
    if len(stats.columns) > 1:
        for subset in combinations(stats.columns, len(stats.columns) - 1):
            subset_stats = prior_stats.get(tuple(subset))
            if subset_stats is None:
                continue
            best_subset_singleton = max(
                best_subset_singleton, subset_stats.singleton_ratio
            )
            best_subset_class = max(best_subset_class, subset_stats.class_ratio)

    marginal_uniqueness = max(0.0, stats.singleton_ratio - best_subset_singleton)
    marginal_fragmentation = max(0.0, stats.class_ratio - best_subset_class)
    confidence = min(
        0.99,
        sum(profile_by_name[column].confidence for column in stats.columns)
        / len(stats.columns),
    )
    score = min(
        1.0,
        0.4 * stats.singleton_ratio
        + 0.25 * stats.class_ratio
        + 0.2 * marginal_uniqueness
        + 0.1 * marginal_fragmentation
        + 0.05 * confidence,
    )
    singleton_keys = [key for key, count in stats.counts.items() if count == 1]

    return {
        "columns": list(stats.columns),
        "score": round(score, 6),
        "confidence": round(confidence, 6),
        "sampled_rows": len(stats.keys),
        "equivalence_class_count": len(stats.counts),
        "singleton_count": stats.singleton_count,
        "unique_row_ratio": round(stats.singleton_ratio, 6),
        "min_equivalence_class_size": stats.min_class_size,
        "marginal_uniqueness": round(marginal_uniqueness, 6),
        "marginal_fragmentation": round(marginal_fragmentation, 6),
        "key_fingerprints": sorted(_key_fingerprint(key) for key in stats.counts),
        "singleton_key_fingerprints": sorted(
            _key_fingerprint(key) for key in singleton_keys
        ),
        "evidence": [
            f"column_count={len(stats.columns)}",
            f"equivalence_class_count={len(stats.counts)}",
            f"singleton_count={stats.singleton_count}",
            f"marginal_uniqueness={marginal_uniqueness:.6f}",
            "keys=risk_report_compatible",
        ],
    }


def _risk_key_bytes(row: Mapping[str, Any], columns: Sequence[str]) -> bytes:
    key = quasi_identifier_key(row, fields=columns)
    return json.dumps(
        key,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _key_fingerprint(key: bytes) -> str:
    return f"sha256:{hashlib.sha256(key).hexdigest()}"


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip()
    return ""


def _distribution_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _rare_clinical_column(column: str) -> bool:
    key = _name_key(column)
    return any(
        token in key for token in ("rare", "diagnosis", "dx", "condition", "disease")
    )


def _rate(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_MAX_SET_SIZE",
    "ROLE_DIRECT_ID",
    "ROLE_QUASI_ID",
    "ROLE_SAFE",
    "ROLE_SENSITIVE",
    "SUPPORTED_SUFFIXES",
    "scan_table",
]
