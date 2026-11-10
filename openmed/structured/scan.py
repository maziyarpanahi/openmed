"""Deterministic column-role detection for tabular schemas.

``scan_table`` reads a CSV, Parquet, or JSONL table (or an in-memory frame)
and assigns every column one of four privacy roles -- ``direct_id``,
``quasi_id``, ``sensitive``, or ``safe`` -- from column-name normalization and
value-shape statistics. The result is a typed, reusable contract that the
downstream generalization and monotone-lattice k-anonymity search consume to
decide which columns to suppress, generalize, or leave untouched.

The detector is offline and reads only aggregate statistics (cardinality
ratios, date-likeness ratios, numeric-range flags, mean value lengths). No raw
cell value is ever placed in the returned classification or emitted to a log;
the payload carries names, roles, confidences, canonical labels, and derived
signal strings only.

This module is deliberately distinct from :mod:`openmed.structured.qi_detect`.
``qi_detect`` profiles equivalence-class fragmentation to rank candidate
quasi-identifier *sets* for re-identification review; ``scan`` produces a flat
per-column role map (the input contract for the lattice search) and does not
enumerate column combinations.
"""

from __future__ import annotations

import importlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from openmed.core.labels import (
    CLINICAL_CONCEPT,
    DATE,
    DATE_OF_BIRTH,
    DIRECT_IDENTIFIER,
    EMAIL,
    ID_NUM,
    LOCATION,
    OTHER,
    PERSON,
    PHONE,
    QUASI_IDENTIFIER,
    SENSITIVE_ATTRIBUTE,
    SSN,
    STREET_ADDRESS,
    normalize_label,
    policy_label_for,
)

from .table_io import read_table

__all__ = [
    "ColumnClassification",
    "ColumnRole",
    "ProfilerNotAvailableError",
    "ProfilerReportError",
    "RoleOverrideError",
    "TableRoleScan",
    "scan_table",
]


class ColumnRole(str, Enum):
    """Privacy role assigned to a single column.

    The vocabulary is intentionally flat and generalization-oriented: a
    ``direct_id`` is removed, a ``quasi_id`` is generalized by the lattice
    search, a ``sensitive`` column is protected but retained, and a ``safe``
    column needs no treatment. Members are plain strings so ``role ==
    "quasi_id"`` and JSON serialization both work without unwrapping.
    """

    DIRECT_ID = "direct_id"
    QUASI_ID = "quasi_id"
    SENSITIVE = "sensitive"
    SAFE = "safe"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class RoleOverrideError(ValueError):
    """Raised when a caller override names an unknown column or role."""


class ProfilerNotAvailableError(ImportError):
    """Raised when the explicitly requested DataProfiler backend is absent."""


class ProfilerReportError(ValueError):
    """Raised when DataProfiler cannot produce a usable structured report."""


# Canonical labels whose taxonomy policy is ``DIRECT_IDENTIFIER`` but which a
# k-anonymity generalization pipeline treats as quasi-identifiers, because the
# value is generalized (birth year, age band) rather than removed outright.
_QUASI_GENERALIZED_LABELS = frozenset({DATE_OF_BIRTH})

# Aggregate-statistic thresholds. These gate value-shape promotions of
# unlabeled columns; they are deliberately conservative so an unknown column
# stays ``safe`` unless a signal is unambiguous.
_DATE_LIKE_RATIO = 0.8
_NUMERIC_RANGE_MIN = 0
_NUMERIC_RANGE_MAX = 120
_FREE_TEXT_MEAN_LENGTH = 20.0
_FREE_TEXT_UNIQUENESS = 0.8

_DATE_PATTERN = re.compile(
    r"^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*$"
)
_COLUMN_OWNER_AFFIXES = (
    "patient",
    "person",
    "subject",
    "member",
    "provider",
    "clinician",
    "doctor",
)

_PROFILE_BACKENDS = frozenset({"auto", "native", "dataprofiler"})
_DATAPROFILER_LABELS = {
    "ADDRESS": STREET_ADDRESS,
    "BAN": ID_NUM,
    "CREDIT_CARD": ID_NUM,
    "DATE": DATE,
    "DATETIME": DATE,
    "DRIVERS_LICENSE": ID_NUM,
    "EMAIL_ADDRESS": EMAIL,
    "HASH_OR_KEY": ID_NUM,
    "PERSON": PERSON,
    "PHONE_NUMBER": PHONE,
    "SSN": SSN,
    "UUID": ID_NUM,
    "US_STATE": LOCATION,
}


@dataclass(frozen=True)
class ColumnClassification:
    """Typed role assignment for one column.

    Attributes:
        column: The source column name, preserved verbatim.
        role: The assigned :class:`ColumnRole`.
        confidence: Detector confidence in ``[0.0, 1.0]``.
        canonical_label: The core taxonomy label the name resolved to, or
            ``None`` when the name was unrecognized.
        overridden: Whether a caller override pinned this role.
        signals: Aggregate-only evidence strings; never raw cell values.
    """

    column: str
    role: ColumnRole
    confidence: float
    canonical_label: str | None
    overridden: bool
    signals: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping containing no raw cell values."""

        return {
            "column": self.column,
            "role": self.role.value,
            "confidence": round(self.confidence, 6),
            "canonical_label": self.canonical_label,
            "overridden": self.overridden,
            "signals": list(self.signals),
        }


@dataclass(frozen=True)
class TableRoleScan(Mapping[str, ColumnRole]):
    """Immutable ``{column: role}`` contract with per-column detail.

    The object *is* a mapping from column name to :class:`ColumnRole`, so
    ``dict(scan)`` and ``scan["zip"]`` work directly, while
    :attr:`classifications`, :attr:`confidence`, and the role-group properties
    expose the confidence scores and pre-grouped column lists that the
    k-anonymity lattice search consumes.
    """

    classifications: tuple[ColumnClassification, ...]

    def __getitem__(self, column: str) -> ColumnRole:
        for classification in self.classifications:
            if classification.column == column:
                return classification.role
        raise KeyError(column)

    def __iter__(self) -> Iterator[str]:
        return (classification.column for classification in self.classifications)

    def __len__(self) -> int:
        return len(self.classifications)

    @property
    def roles(self) -> dict[str, ColumnRole]:
        """Return the plain ``{column: role}`` map."""

        return {item.column: item.role for item in self.classifications}

    @property
    def confidence(self) -> dict[str, float]:
        """Return the ``{column: confidence}`` map."""

        return {item.column: item.confidence for item in self.classifications}

    def by_role(self, role: ColumnRole) -> tuple[str, ...]:
        """Return the columns assigned ``role`` in schema order."""

        return tuple(item.column for item in self.classifications if item.role is role)

    @property
    def direct_identifiers(self) -> tuple[str, ...]:
        return self.by_role(ColumnRole.DIRECT_ID)

    @property
    def quasi_identifiers(self) -> tuple[str, ...]:
        """Columns the lattice search should generalize."""

        return self.by_role(ColumnRole.QUASI_ID)

    @property
    def sensitive(self) -> tuple[str, ...]:
        return self.by_role(ColumnRole.SENSITIVE)

    @property
    def safe(self) -> tuple[str, ...]:
        return self.by_role(ColumnRole.SAFE)

    def to_dict(self) -> dict[str, str]:
        """Return ``{column: role_value}`` with plain string roles."""

        return {item.column: item.role.value for item in self.classifications}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready, PHI-safe view of the full scan."""

        return {
            "schema_version": "1.0",
            "columns": [item.as_dict() for item in self.classifications],
        }


@dataclass(frozen=True)
class _ColumnStats:
    non_null_count: int
    cardinality: int
    cardinality_ratio: float
    date_like_ratio: float
    numeric_in_demographic_range: bool
    mean_value_length: float


def scan_table(
    path_or_df: Any,
    *,
    overrides: Mapping[str, str | ColumnRole] | None = None,
    max_rows: int | None = None,
    profile_backend: str = "auto",
) -> TableRoleScan:
    """Classify every column of a table into a privacy role.

    Args:
        path_or_df: A CSV/Parquet/JSONL file path, a pandas- or polars-style
            frame, a columnar ``{name: sequence}`` mapping, or a sequence of
            row mappings.
        overrides: Optional ``{column: role}`` pins. A named column must exist
            and the role must be a valid :class:`ColumnRole`; a pinned column
            reports ``confidence`` 1.0 and ``overridden`` true.
        max_rows: Optional cap on the number of rows profiled. Statistics are
            derived from at most this many rows; ``None`` reads every row.
        profile_backend: ``"auto"`` uses Capital One DataProfiler when it is
            installed and otherwise keeps the native aggregate-statistics
            result. ``"native"`` disables the optional integration, while
            ``"dataprofiler"`` requires it. Only DataProfiler's column name,
            predicted label, and aggregate confidence are consumed; samples
            and raw values from its report are discarded.

    Returns:
        A :class:`TableRoleScan`; it behaves as a ``{column: role}`` mapping
        and carries per-column confidence and canonical-label detail.

    Raises:
        RoleOverrideError: If an override names an unknown column or role.
        ProfilerNotAvailableError: If ``profile_backend="dataprofiler"`` and
            the optional package is unavailable.
        ProfilerReportError: If an explicitly requested profiler returns an
            invalid report.
        ValueError: If the input type is unsupported or the table is empty.
    """

    if profile_backend not in _PROFILE_BACKENDS:
        choices = ", ".join(sorted(_PROFILE_BACKENDS))
        raise ValueError(f"profile_backend must be one of: {choices}")

    columns, columnar = _columnar_view(path_or_df, max_rows=max_rows)
    resolved_overrides = _resolve_overrides(columns, overrides)

    profiler_hints: dict[str, tuple[str, float]] = {}
    if profile_backend != "native":
        try:
            profiler_hints = _dataprofiler_hints(columnar)
        except (ProfilerNotAvailableError, ProfilerReportError):
            if profile_backend == "dataprofiler":
                raise

    classifications = tuple(
        _classify_column(
            column,
            columnar[column],
            override=resolved_overrides.get(column),
            profiler_hint=profiler_hints.get(column),
        )
        for column in columns
    )
    return TableRoleScan(classifications=classifications)


def _resolve_overrides(
    columns: Sequence[str],
    overrides: Mapping[str, str | ColumnRole] | None,
) -> dict[str, ColumnRole]:
    if not overrides:
        return {}
    available = set(columns)
    unknown = sorted(set(overrides) - available)
    if unknown:
        raise RoleOverrideError(f"Unknown override columns: {', '.join(unknown)}")
    resolved: dict[str, ColumnRole] = {}
    for column, role in overrides.items():
        try:
            resolved[column] = ColumnRole(role)
        except ValueError:
            valid = ", ".join(member.value for member in ColumnRole)
            raise RoleOverrideError(
                f"Unsupported role {role!r} for column {column!r}; expected one "
                f"of {valid}"
            ) from None
    return resolved


def _classify_column(
    column: str,
    values: Sequence[Any],
    *,
    override: ColumnRole | None,
    profiler_hint: tuple[str, float] | None = None,
) -> ColumnClassification:
    label = _normalize_column_label(column)
    canonical_label = None if label == OTHER else label
    stats = _column_stats(values)

    if override is not None:
        return ColumnClassification(
            column=column,
            role=override,
            confidence=1.0,
            canonical_label=canonical_label,
            overridden=True,
            signals=("override_pinned",),
        )

    role, confidence, signals = _role_from_label_and_shape(
        canonical_label=canonical_label,
        stats=stats,
    )
    if profiler_hint is not None:
        profiler_label, profiler_confidence = profiler_hint
        profiler_role = _role_for_label(profiler_label)
        # DataProfiler is an additional fail-closed signal. It may promote a
        # native safe classification or strengthen a direct identifier, but it
        # never downgrades a native privacy role.
        if role is ColumnRole.SAFE or profiler_role is ColumnRole.DIRECT_ID:
            role = profiler_role
            canonical_label = profiler_label
            confidence = max(confidence, profiler_confidence)
        signals = (
            *signals,
            f"dataprofiler_label={profiler_label}",
            f"dataprofiler_confidence={profiler_confidence:.6f}",
        )
    return ColumnClassification(
        column=column,
        role=role,
        confidence=confidence,
        canonical_label=canonical_label,
        overridden=False,
        signals=signals,
    )


def _dataprofiler_hints(
    columnar: Mapping[str, Sequence[Any]],
) -> dict[str, tuple[str, float]]:
    """Return PHI-free canonical-label hints from optional DataProfiler."""

    report = _dataprofiler_report(columnar)
    data_stats = report.get("data_stats", report.get("data stats"))
    if not isinstance(data_stats, Sequence) or isinstance(data_stats, (str, bytes)):
        raise ProfilerReportError("DataProfiler returned no column statistics")

    hints: dict[str, tuple[str, float]] = {}
    for item in data_stats:
        if not isinstance(item, Mapping):
            raise ProfilerReportError("DataProfiler returned invalid column statistics")
        column = item.get("column_name")
        raw_label = item.get("data_label")
        if not isinstance(column, str) or not isinstance(raw_label, str):
            continue
        canonical_label = _DATAPROFILER_LABELS.get(raw_label.strip().upper())
        if canonical_label is None:
            continue
        hints[column] = (canonical_label, _dataprofiler_confidence(item, raw_label))
    return hints


def _dataprofiler_report(
    columnar: Mapping[str, Sequence[Any]],
) -> Mapping[str, Any]:
    """Run the optional profiler without retaining its raw-value samples."""

    try:
        dataprofiler = importlib.import_module("dataprofiler")
        pandas = importlib.import_module("pandas")
    except ImportError as exc:
        raise ProfilerNotAvailableError(
            "DataProfiler support requires the optional DataProfiler package"
        ) from exc

    try:
        frame = pandas.DataFrame(
            {name: list(values) for name, values in columnar.items()}
        )
        profiler = dataprofiler.Profiler(frame)
        report = profiler.report(report_options={"output_format": "serializable"})
    except Exception:
        raise ProfilerReportError("DataProfiler could not profile the table") from None
    if not isinstance(report, Mapping):
        raise ProfilerReportError("DataProfiler returned an invalid report")
    return report


def _dataprofiler_confidence(item: Mapping[str, Any], label: str) -> float:
    statistics = item.get("statistics")
    if isinstance(statistics, Mapping):
        for key in ("data_label_representation", "avg_predictions"):
            probabilities = statistics.get(key)
            if isinstance(probabilities, Mapping):
                value = probabilities.get(label)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    confidence = float(value)
                    if math.isfinite(confidence):
                        return min(1.0, max(0.0, confidence))
    return 0.8


def _normalize_column_label(column: str) -> str:
    """Resolve a column name through the taxonomy and common owner affixes.

    Clinical schemas frequently qualify otherwise canonical identifier headers
    with an entity owner (for example ``patient_name`` or ``subject_id``).
    The core taxonomy intentionally normalizes entity labels rather than schema
    phrases, so this detector retries the canonical portion after removing one
    recognized owner affix. Unknown phrases such as ``patient_status`` still
    remain unknown and follow the documented safe-default path.
    """

    label = normalize_label(column)
    if label != OTHER:
        return label
    compact = re.sub(r"[^a-z0-9]", "", column.casefold())
    for affix in _COLUMN_OWNER_AFFIXES:
        candidates: list[str] = []
        if compact.startswith(affix):
            candidates.append(compact[len(affix) :])
        if compact.endswith(affix):
            candidates.append(compact[: -len(affix)])
        for candidate in candidates:
            resolved = normalize_label(candidate)
            if resolved != OTHER:
                return resolved
    return OTHER


def _role_from_label_and_shape(
    *,
    canonical_label: str | None,
    stats: _ColumnStats,
) -> tuple[ColumnRole, float, tuple[str, ...]]:
    base_signals = (
        f"cardinality_ratio={stats.cardinality_ratio:.6f}",
        f"non_null_count={stats.non_null_count}",
    )

    if canonical_label is not None:
        role = _role_for_label(canonical_label)
        signals = (
            f"name_label={canonical_label}",
            f"policy={policy_label_for(canonical_label)}",
            *base_signals,
        )
        return role, 0.9, signals

    # Unknown column name -> default to ``safe`` unless a value-shape signal
    # unambiguously indicates a quasi-identifier or sensitive free text.
    if stats.non_null_count:
        if stats.date_like_ratio >= _DATE_LIKE_RATIO:
            confidence = min(0.85, 0.6 + 0.3 * stats.date_like_ratio)
            return (
                ColumnRole.QUASI_ID,
                confidence,
                (
                    f"date_like_ratio={stats.date_like_ratio:.6f}",
                    *base_signals,
                ),
            )
        if stats.numeric_in_demographic_range and stats.cardinality > 1:
            return (
                ColumnRole.QUASI_ID,
                0.7,
                ("numeric_demographic_range=true", *base_signals),
            )
        if (
            stats.mean_value_length >= _FREE_TEXT_MEAN_LENGTH
            and stats.cardinality_ratio >= _FREE_TEXT_UNIQUENESS
        ):
            return (
                ColumnRole.SENSITIVE,
                0.7,
                (
                    f"mean_value_length={stats.mean_value_length:.2f}",
                    "free_text_shape=true",
                    *base_signals,
                ),
            )

    return ColumnRole.SAFE, 0.3, ("unrecognized_name_and_shape", *base_signals)


def _role_for_label(canonical_label: str) -> ColumnRole:
    if canonical_label in _QUASI_GENERALIZED_LABELS:
        return ColumnRole.QUASI_ID
    policy = policy_label_for(canonical_label)
    if policy == DIRECT_IDENTIFIER:
        return ColumnRole.DIRECT_ID
    if policy == QUASI_IDENTIFIER:
        return ColumnRole.QUASI_ID
    if policy in (SENSITIVE_ATTRIBUTE, CLINICAL_CONCEPT):
        return ColumnRole.SENSITIVE
    return ColumnRole.SAFE


def _column_stats(values: Sequence[Any]) -> _ColumnStats:
    rendered = [text for text in (_cell_text(value) for value in values) if text]
    non_null_count = len(rendered)
    counts = Counter(rendered)
    cardinality = len(counts)
    cardinality_ratio = _rate(cardinality, non_null_count)
    date_like = sum(1 for text in rendered if _DATE_PATTERN.match(text))
    date_like_ratio = _rate(date_like, non_null_count)
    mean_value_length = _rate(
        sum(len(text) for text in rendered),
        non_null_count,
    )
    numeric_in_range = bool(rendered) and all(
        _is_demographic_integer(text) for text in rendered
    )
    return _ColumnStats(
        non_null_count=non_null_count,
        cardinality=cardinality,
        cardinality_ratio=cardinality_ratio,
        date_like_ratio=date_like_ratio,
        numeric_in_demographic_range=numeric_in_range,
        mean_value_length=mean_value_length,
    )


def _is_demographic_integer(text: str) -> bool:
    if not re.fullmatch(r"-?\d{1,3}", text):
        return False
    number = int(text)
    return _NUMERIC_RANGE_MIN <= number <= _NUMERIC_RANGE_MAX


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return repr(value)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value.strip())
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, bytes):
        return "<binary>"
    return str(value).strip()


def _rate(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _columnar_view(
    source: Any,
    *,
    max_rows: int | None,
) -> tuple[tuple[str, ...], dict[str, list[Any]]]:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")

    if isinstance(source, (str, Path)):
        rows = read_table(source)
        return _pivot_rows(rows, max_rows=max_rows)

    # pandas / polars style frames expose ``columns`` and are column-indexable.
    # Column labels may be non-string (e.g. integer labels from a headerless
    # load); index with the original label but expose a stringified name.
    if not isinstance(source, (Mapping, Sequence)) and hasattr(source, "columns"):
        names = tuple(str(name) for name in source.columns)
        columnar = {str(name): list(source[name]) for name in source.columns}
        return _bound_columns(names, columnar, max_rows=max_rows)

    if isinstance(source, Mapping):
        names_list: list[str] = []
        columnar: dict[str, list[Any]] = {}
        for name in source:
            column = source[name]
            if isinstance(column, (str, bytes)) or not isinstance(column, Sequence):
                raise ValueError(
                    "Columnar mapping values must be sequences of cell values"
                )
            columnar[str(name)] = list(column)
            names_list.append(str(name))
        return _bound_columns(tuple(names_list), columnar, max_rows=max_rows)

    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        rows = list(source)
        if not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("Row sequences must contain only row mappings")
        return _pivot_rows(rows, max_rows=max_rows)

    raise ValueError(
        "Unsupported table source; expected a path, a dataframe, a columnar "
        "mapping, or a sequence of row mappings"
    )


def _pivot_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int | None,
) -> tuple[tuple[str, ...], dict[str, list[Any]]]:
    bounded = rows if max_rows is None else rows[:max_rows]
    names: list[str] = []
    seen: set[str] = set()
    ordered_keys: list[Any] = []
    for row in bounded:
        for name in row:
            key = str(name)
            if key not in seen:
                seen.add(key)
                names.append(key)
                ordered_keys.append(name)
    if not names:
        raise ValueError("Table source must include at least one column")
    columnar = {str(name): [row.get(name) for row in bounded] for name in ordered_keys}
    return tuple(names), columnar


def _bound_columns(
    names: tuple[str, ...],
    columnar: dict[str, list[Any]],
    *,
    max_rows: int | None,
) -> tuple[tuple[str, ...], dict[str, list[Any]]]:
    if not names:
        raise ValueError("Table source must include at least one column")
    if max_rows is None:
        return names, columnar
    return names, {name: values[:max_rows] for name, values in columnar.items()}
