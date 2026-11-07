"""Aggregate-only re-identification risk reports for tabular exports.

The report in this module is intentionally smaller than the detailed
``risk_report`` and release-assessment artifacts.  It is suitable for a
structured export manifest: row counts, schema descriptors, equivalence-class
size aggregates, caller-declared generalization coverage, suppression counts,
and threshold outcomes are retained, while source cells, class membership,
and suppression offsets are not.

All computation is local and deterministic.  Quasi-identifier values are
canonicalized only long enough to form in-process fingerprints; the public
artifact contains aggregate counts and a digest of those fingerprints.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Final

from openmed.core.audit import stable_hash

__all__ = [
    "MAX_TABULAR_RISK_CELL_STRING_CHARS",
    "MAX_TABULAR_RISK_COLUMNS",
    "MAX_TABULAR_RISK_ROWS",
    "MAX_TABULAR_RISK_TOTAL_CELLS",
    "TabularRiskReport",
    "TabularRiskThresholds",
    "build_tabular_risk_report",
    "compute_tabular_risk_report",
    "generate_tabular_risk_report",
    "render_tabular_risk_json",
    "render_tabular_risk_markdown",
    "tabular_risk_report",
]

_SCHEMA_VERSION: Final = 1
MAX_TABULAR_RISK_CELL_STRING_CHARS: Final = 65_536
MAX_TABULAR_RISK_COLUMNS: Final = 512
MAX_TABULAR_RISK_ROWS: Final = 10_000
MAX_TABULAR_RISK_TOTAL_CELLS: Final = 1_000_000
_MAX_TABULAR_RISK_INTEGER: Final = (1 << 63) - 1
_DIGEST_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_COLUMN_PATTERN: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
_SAFE_TITLE_PATTERN: Final = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,79}$")
_DIRECT_IDENTIFIER_NAME_PARTS: Final = frozenset(
    {
        "address",
        "email",
        "identifier",
        "id",
        "name",
        "phone",
        "secret",
        "ssn",
        "token",
    }
)
_SAFE_KINDS: Final = frozenset(
    {"boolean", "date", "datetime", "decimal", "float", "integer", "null", "string"}
)
_CAVEAT_LOCAL: Final = (
    "This is an aggregate local-sensitive diagnostic over the supplied rows and declared quasi-identifiers.",
    "The report does not estimate population risk or risk from external auxiliary data.",
    "Generalization and suppression are caller-declared evidence and are not independently validated here.",
    "Qualified expert review is required; this artifact is not a compliance certification, clinical decision, or guarantee.",
)
_INFERRED_QI_CAVEAT: Final = (
    "Quasi-identifiers were inferred from schema names; review the selection before release.",
)
_EMPTY_REPORT_CAVEAT: Final = (
    "No rows or schema columns were supplied; risk metrics are empty."
)
_THRESHOLD_ALIASES: Final = {
    "minimum_k": ("minimum_k", "target_k", "k"),
    "max_singleton_rate": ("max_singleton_rate", "maximum_singleton_rate"),
    "max_reidentification_risk": (
        "max_reidentification_risk",
        "maximum_reidentification_risk",
        "max_risk",
    ),
    "max_suppression_rate": (
        "max_suppression_rate",
        "maximum_suppression_rate",
    ),
    "min_generalization_coverage": (
        "min_generalization_coverage",
        "minimum_generalization_coverage",
    ),
}
_THRESHOLD_FIELDS: Final = frozenset(
    name for aliases in _THRESHOLD_ALIASES.values() for name in aliases
)
_SUPPRESSION_ALIASES: Final = {
    "indices": ("row_indices", "indices", "rows"),
    "count": ("count", "suppressed_count"),
}
_SUPPRESSION_FIELDS: Final = frozenset(
    name for aliases in _SUPPRESSION_ALIASES.values() for name in aliases
)


def _snapshot_mapping(
    value: Mapping[Any, Any],
    *,
    field_name: str,
    max_items: int,
) -> dict[str, Any]:
    try:
        items = list(itertools.islice(value.items(), max_items + 1))
    except Exception:  # noqa: BLE001 - mappings are caller-controlled protocols.
        raise ValueError(f"{field_name} could not be read") from None
    if len(items) > max_items:
        raise ValueError(f"{field_name} exceeds the supported item limit")

    result: dict[str, Any] = {}
    for item in items:
        if type(item) not in {list, tuple} or len(item) != 2:
            raise ValueError(f"{field_name} contains an invalid entry")
        key, field_value = item
        if type(key) is not str:
            raise ValueError(f"{field_name} keys must be strings")
        if key in result:
            raise ValueError(f"{field_name} keys must be unique")
        result[key] = field_value
    return result


def _safe_column_name(value: Any, *, field_name: str) -> str:
    if type(value) is not str or _SAFE_COLUMN_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must contain safe column identifiers")
    return value


def _normalize_cell_value(value: Any) -> Any:
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if abs(value) > _MAX_TABULAR_RISK_INTEGER:
            raise ValueError("rows contain an out-of-range integer value")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("rows contain a non-finite numeric value")
        return value
    if type(value) is str:
        if len(value) > MAX_TABULAR_RISK_CELL_STRING_CHARS:
            raise ValueError("rows contain an oversized string value")
        return value
    if type(value) in {date, datetime, time}:
        try:
            value.isoformat()
        except Exception:  # noqa: BLE001 - tzinfo callbacks may be caller-controlled.
            raise ValueError("rows contain an invalid temporal value") from None
        return value
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("rows contain a non-finite decimal value")
        if len(value.as_tuple().digits) > MAX_TABULAR_RISK_CELL_STRING_CHARS:
            raise ValueError("rows contain an oversized decimal value")
        return value
    raise TypeError("rows contain an unsupported scalar value")


@dataclass(frozen=True)
class TabularRiskThresholds:
    """Thresholds used to classify an aggregate tabular risk report.

    The defaults are intentionally advisory: a report requires at least
    2-anonymity and no singleton rows, while allowing callers to decide their
    own generalization and suppression budgets.
    """

    minimum_k: int = 2
    max_singleton_rate: float = 0.0
    max_reidentification_risk: float = 1.0
    max_suppression_rate: float = 1.0
    min_generalization_coverage: float = 0.0

    def __post_init__(self) -> None:
        if (
            type(self.minimum_k) is not int
            or not 1 <= self.minimum_k <= MAX_TABULAR_RISK_ROWS
        ):
            raise ValueError("minimum_k must be a bounded integer >= 1")
        for name in (
            "max_singleton_rate",
            "max_reidentification_risk",
            "max_suppression_rate",
            "min_generalization_coverage",
        ):
            value = getattr(self, name)
            if type(value) not in {int, float}:
                raise ValueError(f"{name} must be a number between 0 and 1")
            try:
                normalized = float(value)
            except (OverflowError, ValueError):
                raise ValueError(f"{name} must be a number between 0 and 1") from None
            if not math.isfinite(normalized) or not 0 <= normalized <= 1:
                raise ValueError(f"{name} must be a number between 0 and 1")
            object.__setattr__(self, name, normalized)

    @property
    def minimum_anonymity(self) -> int:
        """Alias for callers that describe the k threshold as anonymity."""

        return self.minimum_k

    @property
    def maximum_singleton_rate(self) -> float:
        """Return the long-form singleton-rate threshold."""

        return self.max_singleton_rate

    @property
    def maximum_reidentification_risk(self) -> float:
        """Return the long-form re-identification-risk threshold."""

        return self.max_reidentification_risk

    @property
    def maximum_suppression_rate(self) -> float:
        """Return the long-form suppression-rate threshold."""

        return self.max_suppression_rate

    @property
    def minimum_generalization_coverage(self) -> float:
        """Return the long-form generalization-coverage threshold."""

        return self.min_generalization_coverage

    def to_dict(self) -> dict[str, int | float]:
        """Return the canonical JSON-safe threshold mapping."""

        return {
            "minimum_k": self.minimum_k,
            "max_singleton_rate": self.max_singleton_rate,
            "max_reidentification_risk": self.max_reidentification_risk,
            "max_suppression_rate": self.max_suppression_rate,
            "min_generalization_coverage": self.min_generalization_coverage,
        }

    @classmethod
    def from_value(
        cls,
        value: TabularRiskThresholds | Mapping[str, Any] | None,
    ) -> TabularRiskThresholds:
        """Normalize a dataclass or a mapping with common threshold aliases."""

        if value is None:
            return cls()
        if type(value) is cls:
            return value
        if not isinstance(value, Mapping):
            raise TypeError("thresholds must be a mapping or TabularRiskThresholds")
        values = _snapshot_mapping(
            value,
            field_name="thresholds",
            max_items=len(_THRESHOLD_FIELDS),
        )
        if set(values) - _THRESHOLD_FIELDS:
            raise ValueError("thresholds contain unknown fields")

        def pick(*names: str, default: Any) -> Any:
            matches = [name for name in names if name in values]
            if len(matches) > 1:
                raise ValueError("thresholds must not use duplicate aliases")
            return default if not matches else values[matches[0]]

        return cls(
            minimum_k=pick("minimum_k", "target_k", "k", default=2),
            max_singleton_rate=pick(
                "max_singleton_rate",
                "maximum_singleton_rate",
                default=0.0,
            ),
            max_reidentification_risk=pick(
                "max_reidentification_risk",
                "maximum_reidentification_risk",
                "max_risk",
                default=1.0,
            ),
            max_suppression_rate=pick(
                "max_suppression_rate",
                "maximum_suppression_rate",
                default=1.0,
            ),
            min_generalization_coverage=pick(
                "min_generalization_coverage",
                "minimum_generalization_coverage",
                default=0.0,
            ),
        )


class TabularRiskReport(Mapping[str, Any]):
    """Immutable aggregate report with privacy-safe serialization helpers."""

    __slots__ = ("_payload",)
    _payload: dict[str, Any]

    def __init__(self, payload: Mapping[str, Any]) -> None:
        object.__setattr__(
            self, "_payload", copy.deepcopy(_project_report_payload(payload))
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TabularRiskReport is immutable")

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._payload[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-safe copy of the report."""

        return copy.deepcopy(self._payload)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the report deterministically as JSON."""

        return render_tabular_risk_json(self, indent=indent)

    def to_markdown(
        self, *, title: str = "Tabular re-identification risk report"
    ) -> str:
        """Render the report as deterministic, aggregate-only Markdown."""

        return render_tabular_risk_markdown(self, title=title)

    @property
    def row_count(self) -> int:
        """Return the analyzed (non-suppressed) row count."""

        return int(self["row_counts"]["analyzed"])

    @property
    def source_row_count(self) -> int:
        """Return the source row count before declared suppression."""

        return int(self["row_counts"]["source"])

    @property
    def suppressed_row_count(self) -> int:
        """Return the declared suppressed-row count."""

        return int(self["row_counts"]["suppressed"])

    @property
    def minimum_k(self) -> int:
        """Return the smallest observed equivalence-class size."""

        return int(self["equivalence_classes"]["minimum_k"])

    @property
    def risk_score(self) -> float:
        """Return the maximum exact-match risk indicator."""

        return float(self["risk"]["max_reidentification_risk"])

    @property
    def meets_thresholds(self) -> bool:
        """Return whether every configured threshold passed."""

        return bool(self["status"]["meets_thresholds"])

    @property
    def digest(self) -> str:
        """Return a digest over the aggregate report payload."""

        return stable_hash(self.to_dict())


def tabular_risk_report(
    rows: Any = None,
    schema: Mapping[str, Any] | Sequence[str] | None = None,
    *,
    quasi_identifiers: Sequence[str] | None = None,
    generalization: Mapping[str, Any] | Sequence[str] | str | None = None,
    generalized_columns: Mapping[str, Any] | Sequence[str] | str | None = None,
    suppressed_rows: int | Sequence[int] | Mapping[str, Any] | None = None,
    suppression: int | Sequence[int] | Mapping[str, Any] | None = None,
    suppression_count: int | None = None,
    thresholds: TabularRiskThresholds | Mapping[str, Any] | None = None,
) -> TabularRiskReport:
    """Build an aggregate re-identification risk report for tabular rows.

    Args:
        rows: A sequence of row mappings, a single row mapping, or a local
            DataFrame-like object exposing ``to_dict('records')`` or
            ``to_dicts()``.  Rows are copied only for the duration of this
            computation.
        schema: Optional schema mapping or sequence of column names.  When
            omitted, the union of row keys is used.
        quasi_identifiers: Columns forming the exact-match equivalence key.
            When omitted, likely direct-identifier columns are excluded and a
            caveat records that the remaining columns were inferred.
        generalization: Caller-declared generalized columns.  A mapping may
            carry arbitrary level metadata, but that metadata is never
            serialized; only the declared column names and coverage are kept.
        generalized_columns: Alias that can be used instead of
            ``generalization``.
        suppressed_rows: Row offsets to omit from analysis, an integer count
            for already-omitted rows, or a mapping with ``row_indices`` and/or
            ``count``.  ``suppression`` is an alias.
        suppression_count: Additional count for rows omitted before ``rows``.
        thresholds: A :class:`TabularRiskThresholds` or mapping with its
            field names (and common ``target_k``/``maximum_*`` aliases).

    Returns:
        A JSON-serializable :class:`TabularRiskReport` containing only
        aggregate evidence, digests, thresholds, and caveats.

    Raises:
        TypeError or ValueError: If the table, schema, declared columns, or
            thresholds are malformed.  Error messages contain metadata only;
            source cell values are never interpolated.
    """

    materialized_rows = _materialize_rows(rows)
    schema_columns, schema_kinds = _build_schema(materialized_rows, schema)
    qi_columns, inferred_qi = _normalize_quasi_identifiers(
        quasi_identifiers,
        schema_columns,
    )
    _validate_declared_columns(qi_columns, schema_columns, "quasi_identifiers")

    generalized = _normalize_generalization(
        generalization,
        generalized_columns,
    )
    _validate_declared_columns(
        tuple(generalized),
        schema_columns,
        "generalization",
    )

    suppression_info = _normalize_suppression(
        len(materialized_rows),
        suppressed_rows,
        suppression,
        suppression_count,
    )
    analyzed_rows = [
        row
        for index, row in enumerate(materialized_rows)
        if index not in suppression_info.indices
    ]
    source_row_count = suppression_info.source_row_count
    suppressed_count = suppression_info.suppressed_count
    analyzed_row_count = len(analyzed_rows)

    schema_payload = _schema_payload(
        materialized_rows,
        schema_columns,
        schema_kinds,
    )
    schema_digest = stable_hash(
        {
            "kind": "openmed-tabular-schema",
            "columns": schema_payload["columns"],
        }
    )

    class_counts = _equivalence_class_counts(analyzed_rows, qi_columns)
    class_sizes = sorted(class_counts.values())
    class_distribution = [
        {"size": size, "class_count": count}
        for size, count in sorted(Counter(class_sizes).items())
    ]
    minimum_k = min(class_sizes, default=0)
    singleton_class_count = sum(size == 1 for size in class_sizes)
    singleton_row_count = singleton_class_count
    singleton_rate = _rate(singleton_row_count, analyzed_row_count)
    risk_values = [1.0 / size for size in class_sizes for _ in range(size)]
    max_risk = max(risk_values, default=0.0)
    mean_risk = sum(risk_values) / len(risk_values) if risk_values else 0.0
    p95_risk = _percentile(risk_values, 0.95)

    threshold_values = TabularRiskThresholds.from_value(thresholds)
    generalization_columns = tuple(sorted(generalized))
    generalized_qi_count = len(set(generalization_columns) & set(qi_columns))
    generalization_coverage = _rate(generalized_qi_count, len(qi_columns))
    suppression_rate = _rate(suppressed_count, source_row_count)

    status: dict[str, Any] = {
        "meets_minimum_k": bool(analyzed_row_count)
        and minimum_k >= threshold_values.minimum_k,
        "meets_max_singleton_rate": singleton_rate
        <= threshold_values.max_singleton_rate,
        "meets_max_reidentification_risk": max_risk
        <= threshold_values.max_reidentification_risk,
        "meets_max_suppression_rate": suppression_rate
        <= threshold_values.max_suppression_rate,
        "meets_min_generalization_coverage": generalization_coverage
        >= threshold_values.min_generalization_coverage,
    }
    status["meets_thresholds"] = all(status.values())
    status["outcome"] = "pass" if status["meets_thresholds"] else "review"

    class_fingerprints = sorted(
        (fingerprint, size) for fingerprint, size in class_counts.items()
    )
    dataset_digest = stable_hash(
        {
            "kind": "openmed-tabular-risk-input",
            "schema_digest": schema_digest,
            "quasi_identifiers": list(qi_columns),
            "analyzed_row_count": analyzed_row_count,
            "suppressed_row_count": suppressed_count,
            "class_fingerprints": class_fingerprints,
        }
    )

    caveats = list(_CAVEAT_LOCAL)
    if inferred_qi:
        caveats.extend(_INFERRED_QI_CAVEAT)
    if not materialized_rows and not schema_columns:
        caveats.append(
            "No rows or schema columns were supplied; risk metrics are empty."
        )

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "artifact": "tabular_reidentification_risk_report",
        "detail_level": "aggregate_phi_safe",
        "row_count": analyzed_row_count,
        "source_row_count": source_row_count,
        "suppressed_row_count": suppressed_count,
        "row_counts": {
            "source": source_row_count,
            "analyzed": analyzed_row_count,
            "suppressed": suppressed_count,
            "suppression_rate": suppression_rate,
        },
        "schema": schema_payload,
        "quasi_identifiers": {
            "columns": list(qi_columns),
            "count": len(qi_columns),
            "inferred": inferred_qi,
        },
        "equivalence_classes": {
            "count": len(class_sizes),
            "minimum_k": minimum_k,
            "mean_size": (
                analyzed_row_count / len(class_sizes) if class_sizes else 0.0
            ),
            "size_distribution": class_distribution,
            "singleton_class_count": singleton_class_count,
            "singleton_row_count": singleton_row_count,
            "singleton_rate": singleton_rate,
        },
        "risk": {
            "attacker_model": "exact_match_on_declared_quasi_identifiers",
            "max_reidentification_risk": max_risk,
            "mean_reidentification_risk": mean_risk,
            "p95_reidentification_risk": p95_risk,
            "population_risk_estimated": False,
        },
        "generalization": {
            "declared_columns": list(generalization_columns),
            "declared_count": len(generalization_columns),
            "quasi_identifier_count": len(qi_columns),
            "quasi_identifier_coverage": generalization_coverage,
        },
        "suppression": {
            "declared_count": suppressed_count,
            "source_row_count": source_row_count,
            "analyzed_row_count": analyzed_row_count,
            "rate": suppression_rate,
        },
        "thresholds": threshold_values.to_dict(),
        "status": status,
        "schema_digest": schema_digest,
        "dataset_digest": dataset_digest,
        "caveats": caveats,
    }
    return TabularRiskReport(payload)


build_tabular_risk_report = tabular_risk_report
compute_tabular_risk_report = tabular_risk_report
generate_tabular_risk_report = tabular_risk_report


def render_tabular_risk_json(
    report: Mapping[str, Any] | TabularRiskReport,
    *,
    indent: int | None = 2,
) -> str:
    """Render an aggregate-only report as canonical JSON.

    Unknown fields are discarded when a plain mapping is supplied.  This
    prevents a caller from accidentally turning a safe renderer into a raw-row
    serializer by attaching arbitrary diagnostic fields to a report.
    """

    payload = _safe_report_payload(report)
    if indent is not None and (type(indent) is not int or not 0 <= indent <= 8):
        raise ValueError("tabular risk JSON indentation is invalid")
    return json.dumps(
        payload,
        ensure_ascii=True,
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
    )


def render_tabular_risk_markdown(
    report: Mapping[str, Any] | TabularRiskReport,
    *,
    title: str = "Tabular re-identification risk report",
) -> str:
    """Render aggregate risk evidence as deterministic Markdown."""

    payload = _safe_report_payload(report)
    if type(title) is not str or _SAFE_TITLE_PATTERN.fullmatch(title) is None:
        raise ValueError("tabular risk report title is invalid")
    rows = payload["row_counts"]
    schema = payload["schema"]
    qi = payload["quasi_identifiers"]
    classes = payload["equivalence_classes"]
    risk = payload["risk"]
    generalization = payload["generalization"]
    suppression = payload["suppression"]
    thresholds = payload["thresholds"]
    status = payload["status"]

    lines = [
        f"# {_markdown_cell(title)}",
        "",
        "> Aggregate evidence only; no source cells, class members, or suppression offsets are included.",
        "",
        "## Outcome",
        "",
        f"**{_markdown_cell(status['outcome'].upper())}** — "
        f"thresholds met: `{str(status['meets_thresholds']).lower()}`.",
        "",
        "| Check | Observed | Threshold | Result |",
        "| --- | ---: | ---: | :---: |",
        _markdown_row(
            "Minimum k",
            classes["minimum_k"],
            thresholds["minimum_k"],
            status["meets_minimum_k"],
        ),
        _markdown_row(
            "Singleton rate",
            _format_percent(classes["singleton_rate"]),
            _format_percent(thresholds["max_singleton_rate"]),
            status["meets_max_singleton_rate"],
        ),
        _markdown_row(
            "Maximum exact-match risk",
            _format_decimal(risk["max_reidentification_risk"]),
            _format_decimal(thresholds["max_reidentification_risk"]),
            status["meets_max_reidentification_risk"],
        ),
        _markdown_row(
            "Suppression rate",
            _format_percent(suppression["rate"]),
            _format_percent(thresholds["max_suppression_rate"]),
            status["meets_max_suppression_rate"],
        ),
        _markdown_row(
            "QI generalization coverage",
            _format_percent(generalization["quasi_identifier_coverage"]),
            _format_percent(thresholds["min_generalization_coverage"]),
            status["meets_min_generalization_coverage"],
        ),
        "",
        "## Row and schema summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Source rows | {rows['source']} |",
        f"| Analyzed rows | {rows['analyzed']} |",
        f"| Declared suppressed rows | {rows['suppressed']} |",
        f"| Suppression rate | {_format_percent(rows['suppression_rate'])} |",
        f"| Schema columns | {schema['column_count']} |",
        f"| Quasi-identifiers | {qi['count']} |",
        "",
        "### Schema",
        "",
        "| Column | Kind | Nullable | Missing rows | Distinct values |",
        "| --- | --- | :---: | ---: | ---: |",
    ]
    for column in schema["columns"]:
        lines.append(
            _markdown_row(
                column["name"],
                column["kind"],
                str(column["nullable"]).lower(),
                column["missing_count"],
                column["distinct_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Equivalence-class risk",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Classes | {classes['count']} |",
            f"| Minimum k | {classes['minimum_k']} |",
            f"| Mean class size | {_format_decimal(classes['mean_size'])} |",
            f"| Singleton classes | {classes['singleton_class_count']} |",
            f"| Singleton rows | {classes['singleton_row_count']} |",
            f"| Mean exact-match risk | {_format_decimal(risk['mean_reidentification_risk'])} |",
            f"| P95 exact-match risk | {_format_decimal(risk['p95_reidentification_risk'])} |",
            "",
            "### Class-size distribution",
            "",
            "| Class size | Class count |",
            "| ---: | ---: |",
        ]
    )
    for item in classes["size_distribution"]:
        lines.append(f"| {item['size']} | {item['class_count']} |")

    lines.extend(
        [
            "",
            "## Generalization and suppression",
            "",
            f"- Declared generalized columns: `{generalization['declared_count']}`.",
            f"- Generalization coverage over declared quasi-identifiers: `{_format_percent(generalization['quasi_identifier_coverage'])}`.",
            f"- Declared suppressed rows: `{suppression['declared_count']}`.",
            "- Generalization levels and suppression offsets are intentionally not retained.",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in payload["caveats"])
    lines.extend(
        [
            "",
            f"Schema digest: `{payload['schema_digest']}`.",
            f"Aggregate input digest: `{payload['dataset_digest']}`.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class _SuppressionInfo:
    indices: frozenset[int]
    source_row_count: int
    suppressed_count: int


def _materialize_rows(records: Any) -> list[dict[str, Any]]:
    if records is None:
        return []

    if not isinstance(records, Mapping) and type(records) not in {list, tuple}:
        try:
            to_dicts = getattr(records, "to_dicts", None)
            if callable(to_dicts):
                records = to_dicts()
            else:
                to_dict = getattr(records, "to_dict", None)
                if not callable(to_dict):
                    raise TypeError
                records = to_dict("records")
        except Exception:  # noqa: BLE001 - adapters are caller-controlled.
            raise TypeError(
                "DataFrame-like rows must expose a supported records conversion"
            ) from None

    if isinstance(records, Mapping):
        records = [records]

    if type(records) not in {list, tuple}:
        raise TypeError("rows must be a sequence of row mappings")
    if len(records) > MAX_TABULAR_RISK_ROWS:
        raise ValueError("rows exceed the supported item limit")

    materialized: list[dict[str, Any]] = []
    total_cells = 0
    for row in records:
        if not isinstance(row, Mapping):
            raise TypeError("rows must contain row mappings")
        raw_fields = _snapshot_mapping(
            row,
            field_name="row",
            max_items=MAX_TABULAR_RISK_COLUMNS,
        )
        total_cells += len(raw_fields)
        if total_cells > MAX_TABULAR_RISK_TOTAL_CELLS:
            raise ValueError("rows exceed the supported cell limit")
        fields = {
            _safe_column_name(field, field_name="row column names"): (
                _normalize_cell_value(value)
            )
            for field, value in raw_fields.items()
        }
        materialized.append(fields)
    return materialized


def _build_schema(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any] | Sequence[str] | None,
) -> tuple[tuple[str, ...], dict[str, str]]:
    row_columns = {field for row in rows for field in row}
    supplied_kinds: dict[str, str] = {}
    if schema is None:
        supplied_columns: set[str] = set()
    elif isinstance(schema, Mapping):
        schema_values = _snapshot_mapping(
            schema,
            field_name="schema",
            max_items=MAX_TABULAR_RISK_COLUMNS,
        )
        supplied_columns = set()
        for field, descriptor in schema_values.items():
            field = _safe_column_name(field, field_name="schema column names")
            supplied_columns.add(field)
            supplied_kinds[field] = _safe_kind_from_descriptor(descriptor)
    elif type(schema) in {list, tuple}:
        if len(schema) > MAX_TABULAR_RISK_COLUMNS:
            raise ValueError("schema exceeds the supported column limit")
        supplied_columns = set()
        for field in schema:
            field = _safe_column_name(field, field_name="schema columns")
            if field in supplied_columns:
                raise ValueError("schema columns must be unique")
            supplied_columns.add(field)
    else:
        raise TypeError("schema must be a column mapping or sequence of names")

    columns = tuple(sorted(row_columns | supplied_columns))
    if len(columns) > MAX_TABULAR_RISK_COLUMNS:
        raise ValueError("combined schema exceeds the supported column limit")
    return columns, supplied_kinds


def _schema_payload(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    supplied_kinds: Mapping[str, str],
) -> dict[str, Any]:
    descriptors: list[dict[str, Any]] = []
    for column in columns:
        values = [row[column] for row in rows if column in row]
        kinds = {_value_kind(value) for value in values if value is not None}
        kind = supplied_kinds.get(column)
        if kind is None:
            if not kinds:
                kind = "null"
            elif len(kinds) == 1:
                kind = next(iter(kinds))
            else:
                kind = "mixed"
        missing_count = sum(column not in row or row[column] is None for row in rows)
        distinct = {_value_fingerprint(value) for value in values}
        descriptors.append(
            {
                "name": column,
                "kind": kind,
                "nullable": bool(missing_count),
                "missing_count": missing_count,
                "distinct_count": len(distinct),
            }
        )
    return {"column_count": len(descriptors), "columns": descriptors}


def _safe_kind_from_descriptor(value: Any) -> str:
    if type(value) is not str:
        return "unknown"
    normalized = value.strip().lower()
    if normalized in _SAFE_KINDS:
        return normalized
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized in {"int", "int32", "int64", "integer"}:
        return "integer"
    if normalized in {"float32", "float64", "number", "numeric"}:
        return "float"
    if normalized in {"str", "string", "text", "category", "categorical"}:
        return "string"
    if normalized in {"datetime64", "timestamp"}:
        return "datetime"
    return "unknown"


def _value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("rows contain a non-finite numeric value")
        return "float"
    if type(value) is str:
        return "string"
    if type(value) is datetime:
        return "datetime"
    if type(value) is date:
        return "date"
    if type(value) is time:
        return "datetime"
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("rows contain a non-finite decimal value")
        return "decimal"
    return "unknown"


def _normalize_quasi_identifiers(
    quasi_identifiers: Sequence[str] | None,
    schema_columns: Sequence[str],
) -> tuple[tuple[str, ...], bool]:
    if quasi_identifiers is None:
        inferred = tuple(
            column
            for column in schema_columns
            if not _looks_like_direct_identifier(column)
        )
        return inferred, True
    if type(quasi_identifiers) not in {list, tuple}:
        raise TypeError("quasi_identifiers must be a sequence of column names")
    if len(quasi_identifiers) > MAX_TABULAR_RISK_COLUMNS:
        raise ValueError("quasi_identifiers exceed the supported column limit")
    columns: list[str] = []
    for column in quasi_identifiers:
        column = _safe_column_name(column, field_name="quasi_identifiers")
        if column in columns:
            raise ValueError("quasi_identifiers must be unique")
        columns.append(column)
    return tuple(sorted(columns)), False


def _looks_like_direct_identifier(column: str) -> bool:
    parts = {part for part in re.split(r"[^a-z0-9]+", column.lower()) if part}
    return bool(parts & _DIRECT_IDENTIFIER_NAME_PARTS) or column.lower().endswith(
        ("_id", "_email", "_phone")
    )


def _validate_declared_columns(
    declared: Sequence[str],
    available: Sequence[str],
    label: str,
) -> None:
    unknown = sorted(set(declared) - set(available))
    if unknown:
        raise ValueError(f"{label} contain unknown schema columns")


def _normalize_generalization(
    generalization: Mapping[str, Any] | Sequence[str] | str | None,
    generalized_columns: Mapping[str, Any] | Sequence[str] | str | None,
) -> dict[str, bool]:
    if generalization is not None and generalized_columns is not None:
        raise ValueError("provide only one generalization declaration")
    value = generalization if generalization is not None else generalized_columns
    if value is None:
        return {}
    if isinstance(value, Mapping):
        values = _snapshot_mapping(
            value,
            field_name="generalization",
            max_items=MAX_TABULAR_RISK_COLUMNS,
        )
        columns: dict[str, bool] = {}
        for column, level in values.items():
            column = _safe_column_name(column, field_name="generalization columns")
            if level is not False and level is not None:
                columns[column] = True
        return columns
    if type(value) is str:
        column = _safe_column_name(value, field_name="generalization columns")
        return {column: True}
    if type(value) in {list, tuple}:
        if len(value) > MAX_TABULAR_RISK_COLUMNS:
            raise ValueError("generalization exceeds the supported column limit")
        columns = {}
        for column in value:
            column = _safe_column_name(column, field_name="generalization columns")
            if column in columns:
                raise ValueError("generalization columns must be unique")
            columns[column] = True
        return columns
    raise TypeError("generalization must be a mapping or sequence of columns")


def _normalize_suppression(
    row_count: int,
    suppressed_rows: int | Sequence[int] | Mapping[str, Any] | None,
    suppression: int | Sequence[int] | Mapping[str, Any] | None,
    suppression_count: int | None,
) -> _SuppressionInfo:
    if suppressed_rows is not None and suppression is not None:
        raise ValueError("provide only one suppression declaration")
    declaration = suppressed_rows if suppressed_rows is not None else suppression
    indices: set[int] = set()
    count_only = 0

    if isinstance(declaration, Mapping):
        values = _snapshot_mapping(
            declaration,
            field_name="suppression",
            max_items=len(_SUPPRESSION_FIELDS),
        )
        if set(values) - _SUPPRESSION_FIELDS:
            raise ValueError("suppression contains unknown fields")

        def pick(group: str) -> Any:
            matches = [name for name in _SUPPRESSION_ALIASES[group] if name in values]
            if len(matches) > 1:
                raise ValueError("suppression must not use duplicate aliases")
            return None if not matches else values[matches[0]]

        raw_indices = pick("indices")
        if raw_indices is not None:
            indices.update(_validate_suppression_indices(raw_indices, row_count))
        raw_count = pick("count")
        count_only = _validate_suppression_count(raw_count)
    elif declaration is not None:
        if type(declaration) is int:
            count_only = _validate_suppression_count(declaration)
        else:
            indices.update(_validate_suppression_indices(declaration, row_count))

    count_only += _validate_suppression_count(suppression_count)
    if row_count + count_only > MAX_TABULAR_RISK_ROWS:
        raise ValueError("source row count exceeds the supported item limit")
    return _SuppressionInfo(
        indices=frozenset(indices),
        source_row_count=row_count + count_only,
        suppressed_count=len(indices) + count_only,
    )


def _validate_suppression_indices(value: Any, row_count: int) -> set[int]:
    if type(value) not in {list, tuple}:
        raise TypeError("suppressed row indices must be a sequence of integers")
    if len(value) > row_count:
        raise ValueError("suppressed row indices exceed the supplied rows")
    indices: set[int] = set()
    for index in value:
        if type(index) is not int or index < 0 or index >= row_count:
            raise ValueError("suppressed row indices are outside the supplied rows")
        if index in indices:
            raise ValueError("suppressed row indices must be unique")
        indices.add(index)
    return indices


def _validate_suppression_count(value: Any) -> int:
    if value is None:
        return 0
    if type(value) is not int or not 0 <= value <= MAX_TABULAR_RISK_ROWS:
        raise ValueError("suppressed row count must be a bounded integer")
    return value


def _equivalence_class_counts(
    rows: Sequence[Mapping[str, Any]],
    quasi_identifiers: Sequence[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        key = tuple(
            _value_fingerprint(row[field]) if field in row else _MISSING_FINGERPRINT
            for field in quasi_identifiers
        )
        fingerprint = stable_hash(
            {
                "kind": "openmed-tabular-quasi-identifier-class",
                "key": key,
            }
        )
        counts[fingerprint] += 1
    return counts


_MISSING_FINGERPRINT: Final = "missing-column"


def _value_fingerprint(value: Any) -> str:
    kind = _value_kind(value)
    if value is None:
        canonical: Any = None
    elif kind == "datetime":
        try:
            canonical = value.isoformat()
        except Exception:  # noqa: BLE001 - tzinfo callbacks may be caller-controlled.
            raise ValueError("rows contain an invalid temporal value") from None
    elif kind == "date":
        try:
            canonical = value.isoformat()
        except Exception:  # noqa: BLE001 - callbacks may be caller-controlled.
            raise ValueError("rows contain an invalid temporal value") from None
    elif kind == "decimal":
        canonical = str(value)
    elif kind == "float":
        canonical = repr(value)
    elif kind == "unknown":
        raise TypeError("rows contain an unsupported scalar value")
    else:
        canonical = value
    return stable_hash({"type": kind, "value": canonical})


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _safe_report_payload(
    report: Mapping[str, Any] | TabularRiskReport,
) -> dict[str, Any]:
    if type(report) is TabularRiskReport:
        return report.to_dict()
    return TabularRiskReport(report).to_dict()


def _project_report_payload(
    report: Mapping[str, Any] | TabularRiskReport,
) -> dict[str, Any]:
    if type(report) is TabularRiskReport:
        raw = report.to_dict()
    elif isinstance(report, Mapping):
        raw = _snapshot_mapping(
            report,
            field_name="tabular risk report",
            max_items=32,
        )
    else:
        raise TypeError("report must be a mapping or TabularRiskReport")

    # Derive every display metric and decision from bounded aggregate inputs.
    # Unknown top-level fields remain discarded so renderers cannot become a
    # raw-row serialization path.
    row_counts = _safe_mapping(
        raw.get("row_counts"), field_name="report row counts", max_items=4
    )
    schema = _safe_mapping(raw.get("schema"), field_name="report schema", max_items=2)
    qi = _safe_mapping(
        raw.get("quasi_identifiers"),
        field_name="report quasi-identifiers",
        max_items=3,
    )
    classes = _safe_mapping(
        raw.get("equivalence_classes"),
        field_name="report equivalence classes",
        max_items=7,
    )
    generalization = _safe_mapping(
        raw.get("generalization"),
        field_name="report generalization",
        max_items=4,
    )
    threshold_mapping = _safe_mapping(
        raw.get("thresholds"),
        field_name="report thresholds",
        max_items=len(_THRESHOLD_FIELDS),
    )

    source_row_count = _safe_nonnegative_int(
        row_counts.get("source"), maximum=MAX_TABULAR_RISK_ROWS
    )
    analyzed_row_count = _safe_nonnegative_int(
        row_counts.get("analyzed"), maximum=MAX_TABULAR_RISK_ROWS
    )
    suppressed_row_count = _safe_nonnegative_int(
        row_counts.get("suppressed"), maximum=MAX_TABULAR_RISK_ROWS
    )
    if source_row_count != analyzed_row_count + suppressed_row_count:
        raise ValueError("tabular risk report row counts are inconsistent")

    schema_columns: list[dict[str, Any]] = []
    raw_columns = schema.get("columns", [])
    if type(raw_columns) not in {list, tuple}:
        raise ValueError("tabular risk report schema columns are invalid")
    if len(raw_columns) > MAX_TABULAR_RISK_COLUMNS:
        raise ValueError("tabular risk report schema exceeds the column limit")
    seen_schema_columns: set[str] = set()
    for item in raw_columns:
        column = _safe_mapping(item, field_name="report schema column", max_items=5)
        name = _safe_column_name(column.get("name"), field_name="report schema columns")
        if name in seen_schema_columns:
            raise ValueError("tabular risk report schema columns must be unique")
        seen_schema_columns.add(name)
        nullable = column.get("nullable", False)
        schema_columns.append(
            {
                "name": name,
                "kind": _safe_kind_for_output(column.get("kind")),
                "nullable": nullable if type(nullable) is bool else False,
                "missing_count": _safe_nonnegative_int(
                    column.get("missing_count"), maximum=source_row_count
                ),
                "distinct_count": _safe_nonnegative_int(
                    column.get("distinct_count"), maximum=source_row_count
                ),
            }
        )
    schema_columns.sort(key=lambda item: item["name"])

    size_distribution: list[dict[str, int]] = []
    raw_distribution = classes.get("size_distribution", [])
    if type(raw_distribution) not in {list, tuple}:
        raise ValueError("tabular risk report class distribution is invalid")
    if len(raw_distribution) > MAX_TABULAR_RISK_ROWS:
        raise ValueError("tabular risk report class distribution is too large")
    seen_sizes: set[int] = set()
    for item in raw_distribution:
        entry = _safe_mapping(
            item, field_name="report class distribution entry", max_items=2
        )
        size = _safe_positive_int(entry.get("size"), maximum=MAX_TABULAR_RISK_ROWS)
        count = _safe_positive_int(
            entry.get("class_count"), maximum=MAX_TABULAR_RISK_ROWS
        )
        if size in seen_sizes:
            raise ValueError("tabular risk report class sizes must be unique")
        seen_sizes.add(size)
        size_distribution.append({"size": size, "class_count": count})
    size_distribution.sort(key=lambda item: item["size"])

    derived_row_count = sum(
        item["size"] * item["class_count"] for item in size_distribution
    )
    if derived_row_count != analyzed_row_count:
        raise ValueError("tabular risk report class distribution is inconsistent")
    class_count = sum(item["class_count"] for item in size_distribution)
    minimum_k = min((item["size"] for item in size_distribution), default=0)
    singleton_class_count = sum(
        item["class_count"] for item in size_distribution if item["size"] == 1
    )
    singleton_rate = _rate(singleton_class_count, analyzed_row_count)
    risk_values = [
        1.0 / item["size"]
        for item in size_distribution
        for _ in range(item["size"] * item["class_count"])
    ]
    max_risk = max(risk_values, default=0.0)
    mean_risk = sum(risk_values) / len(risk_values) if risk_values else 0.0
    p95_risk = _percentile(risk_values, 0.95)

    qi_columns = _safe_string_list(
        qi.get("columns"), field_name="report quasi-identifiers"
    )
    generalized_columns = _safe_string_list(
        generalization.get("declared_columns"),
        field_name="report generalized columns",
    )
    schema_names = {column["name"] for column in schema_columns}
    if set(qi_columns) - schema_names:
        raise ValueError("tabular risk report has unknown quasi-identifiers")
    if set(generalized_columns) - schema_names:
        raise ValueError("tabular risk report has unknown generalized columns")
    generalization_coverage = _rate(
        len(set(generalized_columns) & set(qi_columns)), len(qi_columns)
    )
    suppression_rate = _rate(suppressed_row_count, source_row_count)
    threshold_values = TabularRiskThresholds.from_value(threshold_mapping)

    status: dict[str, Any] = {
        "meets_minimum_k": bool(analyzed_row_count)
        and minimum_k >= threshold_values.minimum_k,
        "meets_max_singleton_rate": singleton_rate
        <= threshold_values.max_singleton_rate,
        "meets_max_reidentification_risk": max_risk
        <= threshold_values.max_reidentification_risk,
        "meets_max_suppression_rate": suppression_rate
        <= threshold_values.max_suppression_rate,
        "meets_min_generalization_coverage": generalization_coverage
        >= threshold_values.min_generalization_coverage,
    }
    status["meets_thresholds"] = all(status.values())
    status["outcome"] = "pass" if status["meets_thresholds"] else "review"

    schema_digest = stable_hash(
        {"kind": "openmed-tabular-schema", "columns": schema_columns}
    )
    supplied_schema_digest = raw.get("schema_digest")
    if supplied_schema_digest is not None and type(supplied_schema_digest) is not str:
        raise ValueError("tabular risk report schema digest is invalid")
    if supplied_schema_digest is not None and supplied_schema_digest != schema_digest:
        raise ValueError("tabular risk report schema digest is inconsistent")
    dataset_digest = raw.get("dataset_digest")
    if type(dataset_digest) is not str or not _DIGEST_PATTERN.fullmatch(dataset_digest):
        dataset_digest = "sha256:" + "0" * 64

    inferred_qi = qi.get("inferred", False)
    inferred_qi = inferred_qi if type(inferred_qi) is bool else False
    safe_caveats = list(_CAVEAT_LOCAL)
    if inferred_qi:
        safe_caveats.extend(_INFERRED_QI_CAVEAT)
    if source_row_count == 0 and not schema_columns:
        safe_caveats.append(_EMPTY_REPORT_CAVEAT)

    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact": "tabular_reidentification_risk_report",
        "detail_level": "aggregate_phi_safe",
        "row_count": analyzed_row_count,
        "source_row_count": source_row_count,
        "suppressed_row_count": suppressed_row_count,
        "row_counts": {
            "source": source_row_count,
            "analyzed": analyzed_row_count,
            "suppressed": suppressed_row_count,
            "suppression_rate": suppression_rate,
        },
        "schema": {
            "column_count": len(schema_columns),
            "columns": schema_columns,
        },
        "quasi_identifiers": {
            "columns": qi_columns,
            "count": len(qi_columns),
            "inferred": inferred_qi,
        },
        "equivalence_classes": {
            "count": class_count,
            "minimum_k": minimum_k,
            "mean_size": analyzed_row_count / class_count if class_count else 0.0,
            "size_distribution": size_distribution,
            "singleton_class_count": singleton_class_count,
            "singleton_row_count": singleton_class_count,
            "singleton_rate": singleton_rate,
        },
        "risk": {
            "attacker_model": "exact_match_on_declared_quasi_identifiers",
            "max_reidentification_risk": max_risk,
            "mean_reidentification_risk": mean_risk,
            "p95_reidentification_risk": p95_risk,
            "population_risk_estimated": False,
        },
        "generalization": {
            "declared_columns": generalized_columns,
            "declared_count": len(generalized_columns),
            "quasi_identifier_count": len(qi_columns),
            "quasi_identifier_coverage": generalization_coverage,
        },
        "suppression": {
            "declared_count": suppressed_row_count,
            "source_row_count": source_row_count,
            "analyzed_row_count": analyzed_row_count,
            "rate": suppression_rate,
        },
        "thresholds": threshold_values.to_dict(),
        "status": status,
        "schema_digest": schema_digest,
        "dataset_digest": dataset_digest,
        "caveats": safe_caveats,
    }


def _safe_mapping(value: Any, *, field_name: str, max_items: int) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return _snapshot_mapping(value, field_name=field_name, max_items=max_items)


def _safe_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if type(value) not in {list, tuple}:
        raise ValueError(f"{field_name} must be a sequence")
    if len(value) > MAX_TABULAR_RISK_COLUMNS:
        raise ValueError(f"{field_name} exceeds the column limit")
    result: list[str] = []
    for item in value:
        column = _safe_column_name(item, field_name=field_name)
        if column in result:
            raise ValueError(f"{field_name} must be unique")
        result.append(column)
    return sorted(result)


def _safe_nonnegative_int(value: Any, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("tabular risk report contains an invalid count")
    return value


def _safe_positive_int(value: Any, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError("tabular risk report contains an invalid class count")
    return value


def _safe_kind_for_output(value: Any) -> str:
    return (
        value
        if type(value) is str and value in _SAFE_KINDS | {"mixed", "unknown"}
        else "unknown"
    )


def _markdown_row(*values: Any) -> str:
    return "| " + " | ".join(_markdown_cell(value) for value in values) + " |"


def _markdown_cell(value: Any) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text.replace("|", "\\|")


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _format_decimal(value: float) -> str:
    return f"{float(value):.6f}"
