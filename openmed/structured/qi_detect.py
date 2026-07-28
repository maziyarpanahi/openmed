"""Automatic quasi-identifier detection for tabular datasets.

The detector samples bounded rows from CSV, JSONL/NDJSON, and Parquet files,
profiles column-level distributions, and ranks candidate quasi-identifier sets
by equivalence-class fragmentation. Emitted manifests contain column names and
aggregate counts only; raw cell values and value-derived hashes are never
included in evidence.
"""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
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
from openmed.risk.reid import _normalize_qi_value

from .table_io import (
    _canonical_decimal,
    _DuplicateJsonKeyError,
    _materialize_row,
    _NonFiniteJsonNumberError,
    _strict_json_loads,
    _validate_arrow_temporal_precision,
    _validate_nonempty_schema,
    _validate_parquet_column_families,
    _validated_field_names,
)

ROLE_DIRECT_ID = "direct-id"
ROLE_QUASI_ID = "quasi-id"
ROLE_SENSITIVE = "sensitive"
ROLE_SAFE = "safe"
ROLE_INTERNAL_LINKAGE = "internal-linkage"
ROLE_FREE_TEXT = "free-text"

SUPPORTED_SUFFIXES = frozenset({".csv", ".tsv", ".jsonl", ".ndjson", ".parquet"})
DEFAULT_MAX_ROWS = 10_000
DEFAULT_BATCH_SIZE = 4_096
DEFAULT_MAX_SET_SIZE = 4
DEFAULT_MAX_CANDIDATE_COLUMNS = 8
DEFAULT_SEARCH_BUDGET = 1_000

_VALID_ROLES = frozenset(
    {
        ROLE_DIRECT_ID,
        ROLE_QUASI_ID,
        ROLE_SENSITIVE,
        ROLE_SAFE,
        ROLE_INTERNAL_LINKAGE,
        ROLE_FREE_TEXT,
    }
)
_PRIMARY_ROLE_ORDER = (
    ROLE_DIRECT_ID,
    ROLE_QUASI_ID,
    ROLE_SENSITIVE,
    ROLE_FREE_TEXT,
    ROLE_INTERNAL_LINKAGE,
    ROLE_SAFE,
)

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


class DiscoveryConfigurationError(ValueError):
    """Raised when explicit discovery roles do not match the input schema."""


@dataclass(frozen=True)
class _TableSample:
    format: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    max_rows: int | None
    source_rows: int | None = None
    complete: bool = False
    full_scan: bool = False


@dataclass(frozen=True)
class _ColumnProfile:
    name: str
    role: str
    roles: tuple[str, ...]
    explicit_roles: bool
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
            "roles": list(self.roles),
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
    counts: Counter[Hashable]
    analysis_unit_count: int
    record_count: int
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
    search_budget: int = DEFAULT_SEARCH_BUDGET,
    full_scan: bool = False,
    role_overrides: Mapping[str, str | Sequence[str]] | None = None,
    quasi_identifier_columns: Sequence[str] | None = None,
    sensitive_columns: Sequence[str] | None = None,
    privacy_unit: str | None = None,
    include_safe_candidates: bool = False,
) -> dict[str, Any]:
    """Profile a table and emit an aggregate-only quasi-identifier manifest.

    Args:
        path: CSV, TSV, JSONL/NDJSON, or Parquet file path.
        max_rows: Maximum rows to sample. Readers remain bounded unless
            ``full_scan`` is true.
        batch_size: Parquet row-batch size. Also capped by ``max_rows``.
        max_set_size: Largest candidate quasi-identifier set to score.
        max_candidate_columns: Maximum role-eligible columns considered for
            combination scoring.
        search_budget: Maximum candidate combinations to evaluate.
        full_scan: Read the complete input for final measurement. This can be
            expensive and intentionally ignores ``max_rows``.
        role_overrides: Explicit role or ordered roles for named columns.
            These replace heuristic roles.
        quasi_identifier_columns: Columns that must be treated as
            quasi-identifiers unless a complete ``role_overrides`` entry is
            supplied for the same column.
        sensitive_columns: Columns that must be treated as sensitive unless a
            complete ``role_overrides`` entry is supplied for the same column.
        privacy_unit: Column identifying the person or other privacy unit.
            Candidate equivalence classes are then measured across distinct
            units rather than repeated rows.
        include_safe_candidates: Include otherwise-safe scalar columns in the
            bounded combination search. This can find QIs that become
            identifying only in combination, but increases the search space
            and remains advisory until the resulting roles are reviewed.

    Returns:
        A manifest containing only schema names and aggregate statistics. It
        never includes the source path, cell values, record identifiers, tuple
        fingerprints, or hashes derived from low-entropy values.
    """

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_set_size <= 0:
        raise ValueError("max_set_size must be positive")
    if max_candidate_columns <= 0:
        raise ValueError("max_candidate_columns must be positive")
    if search_budget <= 0:
        raise ValueError("search_budget must be positive")
    if type(include_safe_candidates) is not bool:
        raise TypeError("include_safe_candidates must be a boolean")

    sample = _read_table_sample(
        Path(path),
        max_rows=None if full_scan else max_rows,
        batch_size=batch_size,
        full_scan=full_scan,
    )
    explicit_roles = _resolve_explicit_roles(
        sample.columns,
        role_overrides=role_overrides,
        quasi_identifier_columns=quasi_identifier_columns,
        sensitive_columns=sensitive_columns,
        privacy_unit=privacy_unit,
    )
    privacy_summary = _privacy_unit_summary(sample.rows, privacy_unit)
    if full_scan and privacy_unit is not None and privacy_summary["missing_unit_count"]:
        raise ValueError(
            f"privacy_unit {privacy_unit!r} contains "
            f"{privacy_summary['missing_unit_count']} missing values; "
            "complete subject-level measurement requires every row to identify "
            "its privacy unit"
        )
    profiles = _profile_columns(
        sample.columns,
        sample.rows,
        explicit_roles=explicit_roles,
        privacy_unit=privacy_unit,
    )
    qi_sets, search = _rank_quasi_identifier_sets(
        sample.rows,
        profiles,
        max_set_size=max_set_size,
        max_candidate_columns=max_candidate_columns,
        search_budget=search_budget,
        privacy_unit=privacy_unit,
        include_safe_candidates=include_safe_candidates,
    )
    confidence = max(
        (entry["confidence"] for entry in qi_sets),
        default=0.0,
    )
    dataset_complete = sample.complete
    sampled_discovery = not full_scan
    search_complete = search["complete"]
    no_candidates = not qi_sets
    if no_candidates:
        discovery_status = "insufficient-discovery"
    elif sampled_discovery or not dataset_complete or not search_complete:
        discovery_status = "advisory-candidates"
    else:
        discovery_status = "candidates-found"
    reasons: list[str] = []
    if sampled_discovery:
        reasons.append("sampled_discovery_requires_full_scan")
    if not dataset_complete:
        reasons.append("dataset_sampling_incomplete")
    if not search_complete:
        reasons.append("candidate_search_incomplete")
    if search["set_size_truncated"]:
        reasons.append("candidate_set_size_truncated")
    if no_candidates:
        reasons.append("no_candidate_qi_set_detected")

    return {
        "schema_version": "1.0",
        "format": sample.format,
        "sample": {
            "sampled_rows": len(sample.rows),
            "max_rows": sample.max_rows,
            "source_rows": sample.source_rows,
            "bounded": not sample.complete,
            "complete": sample.complete,
            "mode": "full-scan" if full_scan else "sample",
            "advisory": sampled_discovery,
        },
        "analysis_unit": privacy_summary,
        "columns": {
            profile.name: profile.to_manifest(sampled_rows=len(sample.rows))
            for profile in profiles
        },
        "column_roles": {profile.name: profile.role for profile in profiles},
        "column_role_sets": {profile.name: list(profile.roles) for profile in profiles},
        "quasi_identifier_sets": qi_sets,
        "search": search,
        "discovery": {
            "status": discovery_status,
            "advisory": sampled_discovery
            or not dataset_complete
            or not search_complete,
            "review_required": True,
            "final_measurement_ready": (
                full_scan and dataset_complete and search_complete and not no_candidates
            ),
            "reasons": reasons,
            "no_candidate_is_not_evidence_of_safety": no_candidates,
        },
        "confidence": round(confidence, 6),
    }


def _read_table_sample(
    path: Path,
    *,
    max_rows: int | None,
    batch_size: int,
    full_scan: bool,
) -> _TableSample:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported table format {suffix!r}; expected {supported}")
    if suffix in {".csv", ".tsv"}:
        return _read_delimited_sample(
            path,
            max_rows=max_rows,
            delimiter="\t" if suffix == ".tsv" else ",",
            full_scan=full_scan,
        )
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl_sample(path, max_rows=max_rows, full_scan=full_scan)
    return _read_parquet_sample(
        path,
        max_rows=max_rows,
        batch_size=batch_size,
        full_scan=full_scan,
    )


def _read_delimited_sample(
    path: Path,
    *,
    max_rows: int | None,
    delimiter: str,
    full_scan: bool,
) -> _TableSample:
    rows: list[dict[str, Any]] = []
    complete = True
    with path.open("r", encoding="utf-8", newline="") as handle:
        try:
            reader = csv.DictReader(handle, delimiter=delimiter, strict=True)
            if reader.fieldnames is None:
                raise ValueError("Delimited input must include a header row")
            columns = _validated_field_names(
                reader.fieldnames,
                source="Delimited input header",
            )
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"Delimited input row {row_number} has more cells "
                        "than its header"
                    )
                if any(row[field] is None for field in columns):
                    raise ValueError(
                        f"Delimited input row {row_number} has fewer cells "
                        "than its header"
                    )
                rows.append({field: row[field] for field in columns})
                if max_rows is not None and len(rows) >= max_rows:
                    complete = False
                    break
        except csv.Error:
            raise ValueError("Delimited input is malformed") from None
    return _TableSample(
        format="tsv" if delimiter == "\t" else "csv",
        columns=columns,
        rows=tuple(rows),
        max_rows=max_rows,
        source_rows=len(rows) if complete else None,
        complete=complete,
        full_scan=full_scan,
    )


def _read_jsonl_sample(
    path: Path,
    *,
    max_rows: int | None,
    full_scan: bool,
) -> _TableSample:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    seen: set[str] = set()
    complete = True
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = _strict_json_loads(stripped)
            except _DuplicateJsonKeyError:
                raise ValueError(
                    f"JSONL row {line_number} contains duplicate object keys"
                ) from None
            except (_NonFiniteJsonNumberError, json.JSONDecodeError):
                raise ValueError(f"JSONL row {line_number} is invalid") from None
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row {line_number} must be an object")
            row = _materialize_row(
                payload,
                row_index=line_number,
                format_name="JSONL",
                allow_arrow_scalars=False,
            )
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                complete = False
                break
    _validate_nonempty_schema(rows, source="JSONL input")
    return _TableSample(
        format="jsonl",
        columns=tuple(columns),
        rows=tuple(rows),
        max_rows=max_rows,
        source_rows=len(rows) if complete else None,
        complete=complete,
        full_scan=full_scan,
    )


def _read_parquet_sample(
    path: Path,
    *,
    max_rows: int | None,
    batch_size: int,
    full_scan: bool,
) -> _TableSample:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet quasi-identifier detection requires pyarrow. "
            "Install openmed[columnar] or install pyarrow directly."
        ) from exc

    try:
        parquet_file = pq.ParquetFile(path)
    except pa.ArrowException:
        raise ValueError("Parquet input could not be decoded safely") from None
    schema = parquet_file.schema_arrow
    _validate_arrow_temporal_precision(schema)
    columns = _validated_field_names(
        schema.names,
        source="Parquet input schema",
    )
    source_rows = getattr(getattr(parquet_file, "metadata", None), "num_rows", None)
    rows: list[dict[str, Any]] = []
    resolved_batch_size = batch_size if max_rows is None else min(batch_size, max_rows)

    try:
        batches = parquet_file.iter_batches(batch_size=resolved_batch_size)
        for record_batch in batches:
            for row_number, row in enumerate(
                record_batch.to_pylist(),
                start=len(rows) + 1,
            ):
                if not isinstance(row, Mapping):
                    raise ValueError("Parquet batches must yield row mappings")
                rows.append(
                    _materialize_row(
                        row,
                        row_index=row_number,
                        format_name="Parquet",
                        allow_arrow_scalars=True,
                    )
                )
                if max_rows is not None and len(rows) >= max_rows:
                    complete = source_rows is not None and int(source_rows) <= len(rows)
                    _validate_parquet_column_families(rows, columns)
                    return _TableSample(
                        format="parquet",
                        columns=columns,
                        rows=tuple(rows),
                        max_rows=max_rows,
                        source_rows=(
                            int(source_rows) if source_rows is not None else None
                        ),
                        complete=complete,
                        full_scan=full_scan,
                    )
    except pa.ArrowException:
        raise ValueError("Parquet input could not be decoded safely") from None

    _validate_parquet_column_families(rows, columns)
    return _TableSample(
        format="parquet",
        columns=columns,
        rows=tuple(rows),
        max_rows=max_rows,
        source_rows=int(source_rows) if source_rows is not None else None,
        complete=True,
        full_scan=full_scan,
    )


def _resolve_explicit_roles(
    columns: Sequence[str],
    *,
    role_overrides: Mapping[str, str | Sequence[str]] | None,
    quasi_identifier_columns: Sequence[str] | None,
    sensitive_columns: Sequence[str] | None,
    privacy_unit: str | None,
) -> dict[str, tuple[str, ...]]:
    available = set(columns)
    quasi_columns = (
        (quasi_identifier_columns,)
        if isinstance(quasi_identifier_columns, str)
        else tuple(quasi_identifier_columns or ())
    )
    sensitive_override_columns = (
        (sensitive_columns,)
        if isinstance(sensitive_columns, str)
        else tuple(sensitive_columns or ())
    )
    named_columns = set(role_overrides or ())
    named_columns.update(quasi_columns)
    named_columns.update(sensitive_override_columns)
    if privacy_unit is not None:
        named_columns.add(privacy_unit)
    unknown = sorted(named_columns - available)
    if unknown:
        raise DiscoveryConfigurationError(
            f"Unknown override columns: {', '.join(unknown)}"
        )

    resolved: dict[str, tuple[str, ...]] = {}
    for column, value in (role_overrides or {}).items():
        values = (value,) if isinstance(value, str) else tuple(value)
        if not values:
            raise DiscoveryConfigurationError(
                f"Role override for {column!r} must not be empty"
            )
        invalid = sorted(set(values) - _VALID_ROLES)
        if invalid:
            raise DiscoveryConfigurationError(
                f"Unsupported roles for {column!r}: {', '.join(invalid)}"
            )
        deduplicated = tuple(dict.fromkeys(values))
        if ROLE_SAFE in deduplicated and len(deduplicated) > 1:
            raise DiscoveryConfigurationError(
                f"Role override for {column!r} cannot combine safe with other roles"
            )
        resolved[column] = deduplicated

    for column in quasi_columns:
        if column not in resolved:
            resolved[column] = _ordered_roles(
                (*resolved.get(column, ()), ROLE_QUASI_ID)
            )
    for column in sensitive_override_columns:
        if column not in (role_overrides or {}):
            resolved[column] = _ordered_roles(
                (*resolved.get(column, ()), ROLE_SENSITIVE)
            )
    return resolved


def _ordered_roles(roles: Sequence[str]) -> tuple[str, ...]:
    role_set = set(roles)
    if ROLE_SAFE in role_set and len(role_set) > 1:
        role_set.remove(ROLE_SAFE)
    return tuple(role for role in _PRIMARY_ROLE_ORDER if role in role_set)


def _privacy_unit_summary(
    rows: Sequence[Mapping[str, Any]],
    privacy_unit: str | None,
) -> dict[str, Any]:
    if privacy_unit is None:
        return {
            "kind": "row",
            "method": "record-equivalence-classes",
            "record_count": len(rows),
            "unit_count": len(rows),
            "repeated_unit_count": 0,
            "missing_unit_count": 0,
            "max_records_per_unit": 1 if rows else 0,
        }

    unit_counts: Counter[Hashable] = Counter()
    missing_count = 0
    for index, row in enumerate(rows):
        token, missing = _privacy_unit_token(row, privacy_unit, index=index)
        unit_counts[token] += 1
        missing_count += int(missing)
    return {
        "kind": "subject",
        "method": "longitudinal-subject-profiles",
        "column": privacy_unit,
        "record_count": len(rows),
        "unit_count": len(unit_counts),
        "repeated_unit_count": sum(1 for count in unit_counts.values() if count > 1),
        "records_in_repeated_units": sum(
            count for count in unit_counts.values() if count > 1
        ),
        "missing_unit_count": missing_count,
        "max_records_per_unit": max(unit_counts.values(), default=0),
    }


def _privacy_unit_token(
    row: Mapping[str, Any],
    privacy_unit: str,
    *,
    index: int,
) -> tuple[Hashable, bool]:
    if privacy_unit not in row or row[privacy_unit] is None:
        return ("missing", index), True
    value = row[privacy_unit]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped == value:
            return ("string", value), False
        return ("missing", index), True
    if isinstance(value, bool):
        return ("boolean", value), False
    if isinstance(value, int):
        return ("integer", value), False
    if isinstance(value, float) and math.isfinite(value):
        return ("float", repr(value)), False
    return ("missing", index), True


def _profile_columns(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    explicit_roles: Mapping[str, tuple[str, ...]],
    privacy_unit: str | None,
) -> tuple[_ColumnProfile, ...]:
    return tuple(
        _profile_column(
            column,
            rows,
            explicit_roles=explicit_roles.get(column),
            is_privacy_unit=column == privacy_unit,
        )
        for column in columns
    )


def _profile_column(
    column: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    explicit_roles: tuple[str, ...] | None,
    is_privacy_unit: bool,
) -> _ColumnProfile:
    values = [row.get(column) for row in rows]
    rendered_values = [_cell_text(value) for value in values]
    non_empty = [
        (value, rendered)
        for value, rendered in zip(values, rendered_values)
        if rendered
    ]
    counts = Counter(
        _typed_distribution_value(value, rendered) for value, rendered in non_empty
    )
    cardinality = len(counts)
    non_null_count = len(non_empty)
    null_count = len(values) - non_null_count
    uniqueness_ratio = _rate(cardinality, non_null_count)
    dominant_value_ratio = _rate(max(counts.values(), default=0), non_null_count)
    singleton_value_count = sum(1 for count in counts.values() if count == 1)

    label, source, source_confidence = _label_for_column(
        column,
        [rendered for _value, rendered in non_empty],
    )
    policy_label = policy_label_for(label) if label is not None else None
    risk_level = risk_level_for(label) if label is not None else None
    system_hints = system_hints_for(label) if label is not None else ()
    roles = _roles_for_column(
        column,
        label=label,
        policy_label=policy_label,
        uniqueness_ratio=uniqueness_ratio,
        singleton_value_count=singleton_value_count,
    )
    if explicit_roles is not None:
        roles = explicit_roles
    if is_privacy_unit and ROLE_INTERNAL_LINKAGE not in roles:
        roles = _ordered_roles((*roles, ROLE_INTERNAL_LINKAGE))
    role = roles[0]
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
        roles=roles,
        explicit_roles=explicit_roles is not None,
        is_privacy_unit=is_privacy_unit,
    )

    return _ColumnProfile(
        name=column,
        role=role,
        roles=roles,
        explicit_roles=explicit_roles is not None,
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


def _roles_for_column(
    column: str,
    *,
    label: str | None,
    policy_label: str | None,
    uniqueness_ratio: float,
    singleton_value_count: int,
) -> tuple[str, ...]:
    if policy_label == DIRECT_IDENTIFIER:
        roles = [ROLE_DIRECT_ID]
        if _internal_linkage_column(column):
            roles.append(ROLE_INTERNAL_LINKAGE)
        return tuple(roles)
    if policy_label == QUASI_IDENTIFIER:
        return (ROLE_QUASI_ID,)
    if policy_label == CLINICAL_CONCEPT:
        if _rare_clinical_column(column) and (
            singleton_value_count > 0 or uniqueness_ratio >= 0.25
        ):
            return (ROLE_QUASI_ID, ROLE_SENSITIVE)
        return (ROLE_SENSITIVE,)
    if _free_text_column(column):
        return (ROLE_SENSITIVE, ROLE_FREE_TEXT)
    if label is None and _name_key(column).endswith("id") and uniqueness_ratio >= 0.9:
        return (ROLE_DIRECT_ID, ROLE_INTERNAL_LINKAGE)
    if label is None and uniqueness_ratio >= 0.75 and singleton_value_count > 0:
        return (ROLE_QUASI_ID,)
    return (ROLE_SAFE,)


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
    roles: Sequence[str],
    explicit_roles: bool,
    is_privacy_unit: bool,
) -> tuple[str, ...]:
    evidence = [
        f"source={source}",
        f"non_null_count={non_null_count}",
        f"uniqueness_ratio={uniqueness_ratio:.6f}",
        f"singleton_value_count={singleton_value_count}",
        f"role_count={len(roles)}",
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
    if ROLE_SENSITIVE in roles and ROLE_QUASI_ID in roles:
        evidence.append("overlapping_sensitive_and_quasi_identifier_roles")
    if explicit_roles:
        evidence.append("explicit_role_override")
    if is_privacy_unit:
        evidence.append("explicit_privacy_unit")
    return tuple(evidence)


def _rank_quasi_identifier_sets(
    rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[_ColumnProfile],
    *,
    max_set_size: int,
    max_candidate_columns: int,
    search_budget: int,
    privacy_unit: str | None,
    include_safe_candidates: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_candidate_columns = _candidate_columns(
        profiles,
        limit=len(profiles),
        excluded={privacy_unit} if privacy_unit is not None else set(),
        include_safe_candidates=include_safe_candidates,
    )
    eligible_count = len(all_candidate_columns)
    candidate_columns = all_candidate_columns[:max_candidate_columns]
    max_size = min(max_set_size, len(candidate_columns))
    possible_combinations = sum(
        math.comb(len(candidate_columns), size) for size in range(1, max_size + 1)
    )
    all_set_size_combinations = (2 ** len(candidate_columns)) - 1
    set_size_truncated = max_size < len(candidate_columns)
    search_metadata: dict[str, Any] = {
        "candidate_scope": (
            "all_reviewed_scalar_columns"
            if include_safe_candidates
            else "role_eligible_columns"
        ),
        "eligible_column_count": eligible_count,
        "candidate_column_count": len(candidate_columns),
        "max_candidate_columns": max_candidate_columns,
        "candidate_columns_truncated": eligible_count > len(candidate_columns),
        "max_set_size": max_set_size,
        "effective_max_set_size": max_size,
        "set_size_truncated": set_size_truncated,
        "combination_budget": search_budget,
        "combinations_possible": possible_combinations,
        "combinations_possible_all_set_sizes": all_set_size_combinations,
        "combinations_evaluated": 0,
        "budget_exhausted": False,
        "complete": False,
    }
    if not rows:
        search_metadata["complete"] = not (
            search_metadata["candidate_columns_truncated"] or set_size_truncated
        )
        return [], search_metadata
    if not candidate_columns:
        search_metadata["complete"] = not search_metadata["candidate_columns_truncated"]
        return [], search_metadata

    stats_by_columns: dict[tuple[str, ...], _SetStats] = {}
    ranked: list[dict[str, Any]] = []
    evaluated = 0

    for size in range(1, max_size + 1):
        for combo in combinations(candidate_columns, size):
            if evaluated >= search_budget:
                break
            evaluated += 1
            stats = _set_stats(rows, combo, privacy_unit=privacy_unit)
            if not stats.analysis_unit_count:
                continue
            stats_by_columns[combo] = stats
            ranked.append(
                _qi_set_manifest(
                    stats,
                    profiles=profiles,
                    prior_stats=stats_by_columns,
                )
            )
        if evaluated >= search_budget:
            break

    ranked.sort(
        key=lambda item: (
            item["score"],
            item["singleton_count"],
            item["equivalence_class_count"],
            len(item["columns"]),
        ),
        reverse=True,
    )
    budget_exhausted = evaluated < possible_combinations
    search_metadata.update(
        {
            "combinations_evaluated": evaluated,
            "budget_exhausted": budget_exhausted,
            "complete": (
                not budget_exhausted
                and not search_metadata["candidate_columns_truncated"]
                and not set_size_truncated
            ),
        }
    )
    return ranked, search_metadata


def _candidate_columns(
    profiles: Sequence[_ColumnProfile],
    *,
    limit: int,
    excluded: set[str],
    include_safe_candidates: bool,
) -> tuple[str, ...]:
    candidates = [
        profile
        for profile in profiles
        if (profile.non_null_count or profile.explicit_roles)
        and profile.name not in excluded
        and ROLE_DIRECT_ID not in profile.roles
        and ROLE_INTERNAL_LINKAGE not in profile.roles
        and ROLE_FREE_TEXT not in profile.roles
        and (
            ROLE_QUASI_ID in profile.roles
            or (
                ROLE_SENSITIVE in profile.roles
                and ROLE_FREE_TEXT not in profile.roles
                and profile.singleton_value_count > 0
                and profile.uniqueness_ratio >= 0.05
            )
            or include_safe_candidates
        )
    ]
    candidates.sort(
        key=lambda profile: (
            ROLE_QUASI_ID in profile.roles,
            ROLE_SENSITIVE in profile.roles,
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
    *,
    privacy_unit: str | None,
) -> _SetStats:
    if privacy_unit is None:
        keys = tuple(_risk_key_bytes(row, columns) for row in rows)
        non_empty_keys = tuple(key for key in keys if key != b"[]")
        counts: Counter[Hashable] = Counter(non_empty_keys)
        analysis_unit_count = len(non_empty_keys)
    else:
        profiles_by_unit: dict[Hashable, list[bytes]] = {}
        for index, row in enumerate(rows):
            key = _risk_key_bytes(row, columns)
            if key == b"[]":
                continue
            unit, _missing = _privacy_unit_token(row, privacy_unit, index=index)
            profiles_by_unit.setdefault(unit, []).append(key)
        longitudinal_profiles = tuple(
            tuple(sorted(profile)) for profile in profiles_by_unit.values()
        )
        counts = Counter(longitudinal_profiles)
        analysis_unit_count = len(longitudinal_profiles)
    singleton_count = sum(count for count in counts.values() if count == 1)
    return _SetStats(
        columns=tuple(columns),
        counts=counts,
        analysis_unit_count=analysis_unit_count,
        record_count=len(rows),
        class_ratio=_rate(len(counts), analysis_unit_count),
        singleton_ratio=_rate(singleton_count, analysis_unit_count),
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
    return {
        "columns": list(stats.columns),
        "score": round(score, 6),
        "confidence": round(confidence, 6),
        "sampled_rows": stats.record_count,
        "analysis_unit_count": stats.analysis_unit_count,
        "equivalence_class_count": len(stats.counts),
        "singleton_count": stats.singleton_count,
        "unique_row_ratio": round(stats.singleton_ratio, 6),
        "min_equivalence_class_size": stats.min_class_size,
        "marginal_uniqueness": round(marginal_uniqueness, 6),
        "marginal_fragmentation": round(marginal_fragmentation, 6),
        "evidence": [
            f"column_count={len(stats.columns)}",
            f"equivalence_class_count={len(stats.counts)}",
            f"singleton_count={stats.singleton_count}",
            f"marginal_uniqueness={marginal_uniqueness:.6f}",
            "equivalence_classes=aggregate_counts_only",
        ],
    }


def _risk_key_bytes(row: Mapping[str, Any], columns: Sequence[str]) -> bytes:
    key = [
        [field, *_typed_normalized_value(row, field)]
        for field in sorted(dict.fromkeys(columns))
    ]
    return json.dumps(
        key,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _typed_normalized_value(
    row: Mapping[str, Any],
    field: str,
) -> tuple[str, str]:
    if field not in row:
        return "missing", ""
    value = row[field]
    if value is None:
        return "null", ""
    if isinstance(value, bool):
        return "boolean", "true" if value else "false"
    if isinstance(value, int):
        return "integer", str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "float", "nan"
        if math.isinf(value):
            return "float", "infinity" if value > 0 else "-infinity"
        return "float", _exact_float_text(value)
    if isinstance(value, Decimal):
        return "decimal", str(_canonical_decimal(value))
    if isinstance(value, datetime):
        return "datetime", _exact_datetime_text(value)
    if isinstance(value, date):
        return "date", value.isoformat()
    if isinstance(value, time):
        return "time", value.isoformat()
    if isinstance(value, bytes):
        return "bytes", value.hex()
    if isinstance(value, str):
        return "string", value
    return f"unsupported:{type(value).__name__}", ""


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return _canonical_float_text(value)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value.strip())
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, Decimal):
        return str(_canonical_decimal(value))
    if isinstance(value, datetime):
        return _canonical_datetime_text(value)
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return "<binary>"
    return ""


def _typed_distribution_value(value: Any, rendered: str) -> tuple[str, str]:
    if isinstance(value, bool):
        return "boolean", rendered.casefold()
    if isinstance(value, int):
        return "integer", rendered
    if isinstance(value, float):
        return "float", _exact_float_text(value)
    if isinstance(value, Decimal):
        return "decimal", str(_canonical_decimal(value))
    if isinstance(value, datetime):
        return "datetime", _exact_datetime_text(value)
    if isinstance(value, date):
        return "date", value.isoformat()
    if isinstance(value, time):
        return "time", value.isoformat()
    if isinstance(value, bytes):
        return "bytes", value.hex()
    if isinstance(value, str):
        return "string", value
    return f"unsupported:{type(value).__name__}", ""


def _exact_datetime_text(value: datetime) -> str:
    if value.tzinfo is not None and value.utcoffset() is None:
        raise ValueError("datetime timezone offsets must be determinate")
    return value.isoformat()


def _canonical_datetime_text(value: datetime) -> str:
    if value.tzinfo is not None and value.utcoffset() is not None:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def _canonical_float_text(value: float) -> str:
    return "0" if value == 0.0 else format(value, ".17g")


def _exact_float_text(value: float) -> str:
    return repr(value)


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _rare_clinical_column(column: str) -> bool:
    key = _name_key(column)
    return any(
        token in key for token in ("rare", "diagnosis", "dx", "condition", "disease")
    )


def _internal_linkage_column(column: str) -> bool:
    key = _name_key(column)
    return key in {
        "id",
        "mrn",
        "medicalrecordnumber",
        "memberid",
        "patientid",
        "recordid",
        "subjectid",
    } or key.endswith(("patientid", "recordid", "subjectid"))


def _free_text_column(column: str) -> bool:
    key = _name_key(column)
    return any(
        token in key
        for token in (
            "comment",
            "description",
            "freeform",
            "freetext",
            "narrative",
            "note",
            "summary",
        )
    )


def _rate(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_MAX_SET_SIZE",
    "DEFAULT_SEARCH_BUDGET",
    "DiscoveryConfigurationError",
    "ROLE_DIRECT_ID",
    "ROLE_FREE_TEXT",
    "ROLE_INTERNAL_LINKAGE",
    "ROLE_QUASI_ID",
    "ROLE_SAFE",
    "ROLE_SENSITIVE",
    "SUPPORTED_SUFFIXES",
    "scan_table",
]
