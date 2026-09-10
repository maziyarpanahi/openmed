"""Privacy-safe quasi-identifier profiling and generalization planning.

The profiler answers three questions for a structured table before release:
which columns are likely quasi-identifiers, how much each column increases
uniqueness, and which deterministic generalization or suppression plan reaches
the requested k-anonymity target.  Source values are used only in memory.
Public profile and plan payloads contain column names, labels, offsets, hashes,
and aggregate statistics; they never contain cell values or equivalence-class
keys.

This module deliberately delegates the generalization search to
``openmed.risk.kanon``.  The selected node is retained as level numbers and is
re-applied through the same in-process transforms, so applying a plan does not
need to retain a copy of the source table or re-run a data-dependent search.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Final

from openmed.core.audit import stable_hash

from . import kanon as _kanon
from .k_anonymity import EquivalenceClass, KAnonymityReport
from .reid import _field_category, _field_is_direct_identifier

__all__ = [
    "GeneralizationPlan",
    "QIColumnProfile",
    "QIGeneralization",
    "QIProfiler",
    "QIProfilerReport",
    "QuasiIdentifierProfiler",
    "apply_generalization_plan",
    "profile_quasi_identifier_risk",
    "profile_quasi_identifiers",
    "profile_qi",
]

_SCHEMA_VERSION: Final = 1
_DEFAULT_TARGET_K: Final = 2
_DEFAULT_MAX_LATTICE_NODES: Final = 100_000
_DEFAULT_MAX_SUPPRESSION_SUBSETS: Final = 100_000
_DATE_PATTERN: Final = re.compile(
    r"^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*$"
)
_QI_CATEGORIES: Final = frozenset({"age", "date", "geography", "provider_institution"})
_SENSITIVE_CATEGORIES: Final = frozenset({"rare_condition"})
_MISSING = object()


@dataclass(frozen=True)
class QIColumnProfile:
    """Aggregate profile for one source column.

    ``offsets`` identifies rows whose non-null value occurs exactly once in
    this column.  It is intentionally an offset list rather than a value list.
    ``marginal_uniqueness_contribution`` compares the singleton-row rate for
    the selected quasi-identifier set with and without this column.
    """

    column: str
    label: str
    role: str
    quasi_identifier_likelihood: float
    record_count: int
    non_null_count: int
    missing_count: int
    distinct_count: int
    distinctness: float
    unique_value_count: int
    uniqueness_rate: float
    offsets: tuple[int, ...]
    marginal_uniqueness_contribution: float
    marginal_reidentification_risk: float
    achieved_k_with_column: int
    achieved_k_without_column: int
    rank: int | None = None

    @property
    def uniqueness_contribution(self) -> float:
        """Return this column's standalone unique-row contribution."""

        return self.uniqueness_rate

    @property
    def unique_row_rate(self) -> float:
        """Return the fraction of source rows at unique values."""

        return self.uniqueness_rate

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic payload containing no source values."""

        return {
            "column": self.column,
            "label": self.label,
            "role": self.role,
            "quasi_identifier_likelihood": float(self.quasi_identifier_likelihood),
            "record_count": int(self.record_count),
            "non_null_count": int(self.non_null_count),
            "missing_count": int(self.missing_count),
            "distinct_count": int(self.distinct_count),
            "distinctness": float(self.distinctness),
            "unique_value_count": int(self.unique_value_count),
            "uniqueness_rate": float(self.uniqueness_rate),
            "offsets": list(self.offsets),
            "marginal_uniqueness_contribution": float(
                self.marginal_uniqueness_contribution
            ),
            "uniqueness_contribution": float(self.uniqueness_contribution),
            "marginal_reidentification_risk": float(
                self.marginal_reidentification_risk
            ),
            "achieved_k_with_column": int(self.achieved_k_with_column),
            "achieved_k_without_column": int(self.achieved_k_without_column),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class QIGeneralization:
    """One privacy-safe per-column generalization directive."""

    column: str
    label: str
    level: int
    loss: float
    action: str
    affected_offsets: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the directive without materialized generalized values."""

        return {
            "column": self.column,
            "label": self.label,
            "level": int(self.level),
            "loss": float(self.loss),
            "action": self.action,
            "affected_offsets": list(self.affected_offsets),
        }


@dataclass(frozen=True)
class GeneralizationPlan:
    """Executable risk-reduction plan containing only metadata and offsets.

    ``apply`` accepts the source table separately and returns transformed rows.
    The plan itself stores no source rows, raw cell values, or equivalence-class
    keys.  Row suppression offsets are positional and therefore require the
    same row order and record count as the table that was profiled.
    """

    target_k: int
    quasi_identifiers: tuple[str, ...]
    source_columns: tuple[str, ...]
    source_record_count: int
    columns: tuple[QIGeneralization, ...]
    suppressed_offsets: tuple[int, ...]
    before_achieved_k: int
    after_achieved_k: int
    _node: tuple[tuple[str, int], ...] = field(default=(), repr=False, compare=False)
    _remove_direct_identifiers: bool = field(
        default=False,
        repr=False,
        compare=False,
    )
    removed_columns: tuple[str, ...] = ()

    @property
    def achieved_k(self) -> int:
        """Return the achieved k after applying the planned changes."""

        return self.after_achieved_k

    @property
    def suppression_count(self) -> int:
        """Return the number of source rows selected for suppression."""

        return len(self.suppressed_offsets)

    @property
    def suppressed_rows(self) -> tuple[int, ...]:
        """Return suppressed row offsets as a compatibility alias."""

        return self.suppressed_offsets

    @property
    def levels(self) -> tuple[QIGeneralization, ...]:
        """Return per-column directives as a level-oriented alias."""

        return self.columns

    def apply(self, records: Any) -> list[dict[str, Any]]:
        """Apply this plan to a table with the profiled schema and row order.

        Args:
            records: A sequence of row mappings or a DataFrame-like object.

        Returns:
            Transformed row mappings with planned suppressed rows omitted.

        Raises:
            ValueError: If the source row count or schema does not match the
                positional plan.
        """

        rows, columns = _coerce_table(records)
        if len(rows) != self.source_record_count:
            raise ValueError(
                "generalization plan row count does not match the supplied table"
            )
        if set(columns) != set(self.source_columns):
            raise ValueError(
                "generalization plan columns do not match the supplied table"
            )

        node = dict(self._node)
        transformed = _transform_rows(
            rows,
            self.quasi_identifiers,
            node,
            remove_direct_identifiers=self._remove_direct_identifiers,
        )
        suppressed = set(self.suppressed_offsets)
        return [
            dict(row)
            for offset, row in enumerate(transformed)
            if offset not in suppressed
        ]

    def rescore(self, records: Any) -> KAnonymityReport:
        """Apply and re-score the plan using aggregate-only k evidence."""

        return _safe_k_anonymity_report(
            self.apply(records),
            self.quasi_identifiers,
            self.target_k,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe plan containing no raw source values."""

        node = dict(self._node)
        return {
            "schema_version": _SCHEMA_VERSION,
            "target_k": int(self.target_k),
            "quasi_identifiers": list(self.quasi_identifiers),
            "source_columns": list(self.source_columns),
            "source_record_count": int(self.source_record_count),
            "columns": [item.to_dict() for item in self.columns],
            "node": {column: int(level) for column, level in sorted(node.items())},
            "suppressed_offsets": list(self.suppressed_offsets),
            "suppression_count": self.suppression_count,
            "before_achieved_k": int(self.before_achieved_k),
            "after_achieved_k": int(self.after_achieved_k),
            "removed_columns": list(self.removed_columns),
        }


@dataclass(frozen=True)
class QIProfilerReport:
    """Complete privacy-safe profiler output and its executable plan."""

    record_count: int
    columns: tuple[QIColumnProfile, ...]
    quasi_identifiers: tuple[str, ...]
    before: KAnonymityReport
    plan: GeneralizationPlan
    after: KAnonymityReport

    @property
    def profiles(self) -> tuple[QIColumnProfile, ...]:
        """Return column profiles as a compatibility alias."""

        return self.columns

    @property
    def ranked_columns(self) -> tuple[QIColumnProfile, ...]:
        """Return all columns in deterministic risk rank order."""

        return self.columns

    @property
    def achieved_k(self) -> int:
        """Return the post-plan achieved k."""

        return self.after.achieved_k

    @property
    def generalization_plan(self) -> GeneralizationPlan:
        """Return the plan under its descriptive compatibility name."""

        return self.plan

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic offsets, labels, and statistics only."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "record_count": int(self.record_count),
            "quasi_identifiers": list(self.quasi_identifiers),
            "columns": [item.to_dict() for item in self.columns],
            "before": self.before.to_dict(),
            "plan": self.plan.to_dict(),
            "after": self.after.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the safe report deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )


@dataclass(frozen=True)
class QIProfiler:
    """Profile columns and propose a target-k generalization plan.

    Args:
        target_k: Minimum desired equivalence-class size.
        suppression_rate: Maximum fraction of rows eligible for suppression.
            The default permits the engine to choose minimal suppression when
            generalization alone cannot meet the target.
        suppression_limit: Optional absolute suppression cap.
        max_lattice_nodes: Search budget passed to the k-anonymity engine.
        max_suppression_subsets: Suppression-subset search budget.
        remove_direct_identifiers: Whether applying a plan also removes fields
            recognized as direct identifiers. It defaults to ``False`` because
            this profiler owns quasi-identifier transformations only.
    """

    target_k: int = _DEFAULT_TARGET_K
    suppression_rate: float = 1.0
    suppression_limit: int | None = None
    max_lattice_nodes: int = _DEFAULT_MAX_LATTICE_NODES
    max_suppression_subsets: int = _DEFAULT_MAX_SUPPRESSION_SUBSETS
    remove_direct_identifiers: bool = False

    def __post_init__(self) -> None:
        _validate_target_k(self.target_k)
        _validate_rate(self.suppression_rate)
        if self.suppression_limit is not None and (
            type(self.suppression_limit) is not int or self.suppression_limit < 0
        ):
            raise ValueError("suppression_limit must be an integer >= 0 or None")
        if type(self.max_lattice_nodes) is not int or self.max_lattice_nodes < 1:
            raise ValueError("max_lattice_nodes must be an integer >= 1")
        if (
            type(self.max_suppression_subsets) is not int
            or self.max_suppression_subsets < 1
        ):
            raise ValueError("max_suppression_subsets must be an integer >= 1")

    def profile(
        self,
        records: Any,
        *,
        quasi_identifiers: Sequence[str] | None = None,
    ) -> QIProfilerReport:
        """Profile a table and produce a privacy-safe executable plan."""

        rows, source_columns = _coerce_table(records)
        if self.target_k > len(rows):
            raise ValueError("target_k cannot exceed the source record count")

        measurements = {
            column: _measure_column(rows, column) for column in source_columns
        }
        classifications = {
            column: _classify_column(column, measurements[column])
            for column in source_columns
        }
        selected = _resolve_quasi_identifiers(
            source_columns,
            classifications,
            quasi_identifiers,
        )
        before = _safe_k_anonymity_report(rows, selected, self.target_k)
        profiles = _rank_profiles(
            rows,
            source_columns,
            measurements,
            classifications,
            selected,
        )

        if not selected:
            plan = GeneralizationPlan(
                target_k=self.target_k,
                quasi_identifiers=(),
                source_columns=source_columns,
                source_record_count=len(rows),
                columns=(),
                suppressed_offsets=(),
                before_achieved_k=before.achieved_k,
                after_achieved_k=before.achieved_k,
                removed_columns=(),
            )
            return QIProfilerReport(
                record_count=len(rows),
                columns=profiles,
                quasi_identifiers=(),
                before=before,
                plan=plan,
                after=before,
            )

        enforced = _kanon.enforce_kanon(
            rows,
            quasi_identifiers=selected,
            target_k=self.target_k,
            suppression_limit=self.suppression_limit,
            suppression_rate=self.suppression_rate,
            remove_direct_identifiers=self.remove_direct_identifiers,
            max_lattice_nodes=self.max_lattice_nodes,
            max_suppression_subsets=self.max_suppression_subsets,
        )
        node = {
            str(column): int(level)
            for column, level in enforced["generalization"]["node"].items()
        }
        transformed = _transform_rows(
            rows,
            selected,
            node,
            remove_direct_identifiers=self.remove_direct_identifiers,
        )
        suppressed_offsets = tuple(
            sorted(
                int(item.get("record_index", item.get("offset")))
                for item in enforced["suppressed_records"]
            )
        )
        level_metadata = enforced["generalization"]["levels"]
        directives = tuple(
            _directive_for_column(
                column,
                node[column],
                level_metadata[column],
                rows,
                transformed,
            )
            for column in selected
        )
        removed_columns = tuple(
            column
            for column in source_columns
            if self.remove_direct_identifiers and _field_is_direct_identifier(column)
        )
        plan = GeneralizationPlan(
            target_k=self.target_k,
            quasi_identifiers=selected,
            source_columns=source_columns,
            source_record_count=len(rows),
            columns=directives,
            suppressed_offsets=suppressed_offsets,
            before_achieved_k=before.achieved_k,
            after_achieved_k=int(enforced["kanon"]["k"]),
            _node=tuple((column, node[column]) for column in selected),
            _remove_direct_identifiers=self.remove_direct_identifiers,
            removed_columns=removed_columns,
        )
        released = [
            transformed[offset]
            for offset in range(len(transformed))
            if offset not in set(suppressed_offsets)
        ]
        after = _safe_k_anonymity_report(released, selected, self.target_k)
        if after.achieved_k != plan.after_achieved_k:
            raise RuntimeError("generalization plan re-score disagrees with engine")
        return QIProfilerReport(
            record_count=len(rows),
            columns=profiles,
            quasi_identifiers=selected,
            before=before,
            plan=plan,
            after=after,
        )


QuasiIdentifierProfiler = QIProfiler


def profile_quasi_identifiers(
    records: Any,
    *,
    quasi_identifiers: Sequence[str] | None = None,
    target_k: int = _DEFAULT_TARGET_K,
    suppression_rate: float = 1.0,
    suppression_limit: int | None = None,
    max_lattice_nodes: int = _DEFAULT_MAX_LATTICE_NODES,
    max_suppression_subsets: int = _DEFAULT_MAX_SUPPRESSION_SUBSETS,
    remove_direct_identifiers: bool = False,
) -> QIProfilerReport:
    """Profile a table and return a privacy-safe generalization plan."""

    return QIProfiler(
        target_k=target_k,
        suppression_rate=suppression_rate,
        suppression_limit=suppression_limit,
        max_lattice_nodes=max_lattice_nodes,
        max_suppression_subsets=max_suppression_subsets,
        remove_direct_identifiers=remove_direct_identifiers,
    ).profile(records, quasi_identifiers=quasi_identifiers)


profile_qi = profile_quasi_identifiers
profile_quasi_identifier_risk = profile_quasi_identifiers


def apply_generalization_plan(
    records: Any,
    plan: GeneralizationPlan,
) -> list[dict[str, Any]]:
    """Apply a :class:`GeneralizationPlan` to source rows."""

    if not isinstance(plan, GeneralizationPlan):
        raise TypeError("plan must be a GeneralizationPlan")
    return plan.apply(records)


@dataclass(frozen=True)
class _ColumnMeasurements:
    non_null_count: int
    missing_count: int
    distinct_count: int
    distinctness: float
    unique_value_count: int
    uniqueness_rate: float
    offsets: tuple[int, ...]
    date_like_ratio: float
    demographic_numeric: bool
    mean_value_length: float


def _rank_profiles(
    rows: Sequence[Mapping[str, Any]],
    source_columns: Sequence[str],
    measurements: Mapping[str, _ColumnMeasurements],
    classifications: Mapping[str, tuple[str, str, float]],
    selected: tuple[str, ...],
) -> tuple[QIColumnProfile, ...]:
    full_keys = _column_keys(rows, selected)
    full_singleton_rate = _singleton_rate(full_keys)
    full_k = _achieved_k_from_keys(full_keys)
    selected_metrics: dict[str, tuple[float, float, int]] = {}
    for column in selected:
        without = tuple(item for item in selected if item != column)
        without_keys = _column_keys(rows, without)
        without_singleton_rate = _singleton_rate(without_keys)
        without_k = _achieved_k_from_keys(without_keys)
        contribution = max(0.0, full_singleton_rate - without_singleton_rate)
        risk = max(
            0.0,
            (1.0 / full_k if full_k else 1.0) - (1.0 / without_k if without_k else 0.0),
        )
        selected_metrics[column] = (contribution, risk, without_k)

    def rank_key(column: str) -> tuple[Any, ...]:
        contribution, risk, without_k = selected_metrics.get(
            column,
            (0.0, 0.0, full_k),
        )
        label, role, likelihood = classifications[column]
        measurement = measurements[column]
        selected_first = 0 if column in selected else 1
        return (
            selected_first,
            -contribution,
            -risk,
            -(without_k - full_k),
            -measurement.uniqueness_rate,
            -measurement.distinctness,
            -likelihood,
            role,
            label,
            column,
        )

    ordered = tuple(sorted(source_columns, key=rank_key))
    rank_by_column = {column: index for index, column in enumerate(ordered, 1)}
    profiles: list[QIColumnProfile] = []
    for column in ordered:
        label, role, likelihood = classifications[column]
        measurement = measurements[column]
        contribution, risk, without_k = selected_metrics.get(
            column,
            (0.0, 0.0, full_k),
        )
        profiles.append(
            QIColumnProfile(
                column=column,
                label=label,
                role=role,
                quasi_identifier_likelihood=likelihood,
                record_count=len(rows),
                non_null_count=measurement.non_null_count,
                missing_count=measurement.missing_count,
                distinct_count=measurement.distinct_count,
                distinctness=measurement.distinctness,
                unique_value_count=measurement.unique_value_count,
                uniqueness_rate=measurement.uniqueness_rate,
                offsets=measurement.offsets,
                marginal_uniqueness_contribution=contribution,
                marginal_reidentification_risk=risk,
                achieved_k_with_column=full_k,
                achieved_k_without_column=without_k,
                rank=rank_by_column[column],
            )
        )
    return tuple(profiles)


def _classify_column(
    column: str,
    measurement: _ColumnMeasurements,
) -> tuple[str, str, float]:
    category = _field_category(column)
    if _field_is_direct_identifier(column):
        return "direct_identifier", "direct_identifier", 1.0
    if category in _SENSITIVE_CATEGORIES or (
        measurement.mean_value_length >= 20.0 and measurement.distinctness >= 0.8
    ):
        likelihood = 0.15 + 0.1 * measurement.distinctness
        return category or "free_text", "sensitive", likelihood
    if category in _QI_CATEGORIES:
        likelihood = min(1.0, 0.82 + 0.18 * measurement.distinctness)
        return category, "quasi_identifier", likelihood
    if measurement.date_like_ratio >= 0.8:
        likelihood = 0.65 + 0.25 * measurement.date_like_ratio
        return "date", "quasi_identifier", min(1.0, likelihood)
    if measurement.demographic_numeric and measurement.distinct_count > 1:
        likelihood = 0.58 + 0.3 * measurement.distinctness
        return "numeric_range", "quasi_identifier", min(1.0, likelihood)
    if measurement.distinctness >= 0.8:
        likelihood = 0.58 + 0.35 * measurement.distinctness
        return "high_cardinality", "quasi_identifier", min(1.0, likelihood)
    if measurement.distinctness >= 0.45:
        likelihood = 0.35 + 0.35 * measurement.distinctness
        return "moderate_cardinality", "safe", likelihood
    likelihood = 0.15 + 0.25 * measurement.distinctness
    return "unclassified", "safe", likelihood


def _resolve_quasi_identifiers(
    source_columns: Sequence[str],
    classifications: Mapping[str, tuple[str, str, float]],
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested is not None:
        selected = _normalize_columns(requested, name="quasi_identifiers")
        unknown = sorted(set(selected) - set(source_columns))
        if unknown:
            raise ValueError(f"Unknown quasi-identifier columns: {unknown!r}")
        return tuple(sorted(selected))
    return tuple(
        sorted(
            column
            for column in source_columns
            if classifications[column][1] == "quasi_identifier"
        )
    )


def _directive_for_column(
    column: str,
    level: int,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    transformed: Sequence[Mapping[str, Any]],
) -> QIGeneralization:
    label = str(metadata.get("name") or f"level_{level}")
    lower_label = label.casefold()
    if level == 0:
        action = "retain"
    elif "suppressed" in lower_label:
        action = "suppress_column"
    else:
        action = "generalize"
    affected = tuple(
        offset
        for offset, (source, output) in enumerate(zip(rows, transformed))
        if _value_token(source.get(column, _MISSING))
        != _value_token(output.get(column, _MISSING))
    )
    return QIGeneralization(
        column=column,
        label=label,
        level=int(level),
        loss=float(metadata.get("loss", 0.0)),
        action=action,
        affected_offsets=affected,
    )


def _transform_rows(
    rows: Sequence[Mapping[str, Any]],
    quasi_identifiers: Sequence[str],
    node: Mapping[str, int],
    *,
    remove_direct_identifiers: bool,
) -> list[dict[str, Any]]:
    if not quasi_identifiers:
        if not remove_direct_identifiers:
            return [dict(row) for row in rows]
        return [
            {
                field: value
                for field, value in row.items()
                if not _field_is_direct_identifier(field)
            }
            for row in rows
        ]
    records = _kanon._coerce_records(rows, source="deidentified")
    levels = _kanon._build_hierarchy_levels(
        records,
        tuple(quasi_identifiers),
        None,
    )
    level_node = tuple(int(node[column]) for column in quasi_identifiers)
    return [
        _kanon._transform_record(
            record,
            tuple(quasi_identifiers),
            levels,
            level_node,
            remove_direct_identifiers=remove_direct_identifiers,
        )
        for record in records
    ]


def _safe_k_anonymity_report(
    rows: Sequence[Mapping[str, Any]],
    quasi_identifiers: Sequence[str],
    target_k: int,
) -> KAnonymityReport:
    qis = tuple(quasi_identifiers)
    if not rows:
        return KAnonymityReport(
            record_count=0,
            quasi_identifiers=qis,
            target_k=target_k,
            achieved_k=0,
            smallest_class_size=0,
            equivalence_classes=(),
            violating_rows=(),
            meets_target=False,
        )
    if not qis:
        classes = (
            EquivalenceClass(
                class_hash=stable_hash(
                    {
                        "kind": "qi-profiler-equivalence-class",
                        "row_indices": list(range(len(rows))),
                    }
                ),
                size=len(rows),
                row_indices=tuple(range(len(rows))),
            ),
        )
        return KAnonymityReport(
            record_count=len(rows),
            quasi_identifiers=(),
            target_k=target_k,
            achieved_k=len(rows),
            smallest_class_size=len(rows),
            equivalence_classes=classes,
            violating_rows=(),
            meets_target=len(rows) >= target_k,
        )
    measurement = _kanon.kanon_report(rows, quasi_identifiers=qis)
    classes = tuple(
        sorted(
            (
                EquivalenceClass(
                    class_hash=stable_hash(
                        {
                            "kind": "qi-profiler-equivalence-class",
                            "row_indices": sorted(int(item) for item in cls["members"]),
                        }
                    ),
                    size=int(cls["size"]),
                    row_indices=tuple(sorted(int(item) for item in cls["members"])),
                )
                for cls in measurement["equivalence_classes"]
            ),
            key=lambda item: (item.row_indices[0], item.class_hash),
        )
    )
    violating = tuple(
        sorted(
            offset
            for item in classes
            if item.size < target_k
            for offset in item.row_indices
        )
    )
    achieved_k = int(measurement["k"])
    return KAnonymityReport(
        record_count=len(rows),
        quasi_identifiers=qis,
        target_k=target_k,
        achieved_k=achieved_k,
        smallest_class_size=achieved_k,
        equivalence_classes=classes,
        violating_rows=violating,
        meets_target=bool(rows) and not violating,
    )


def _column_keys(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(_value_token(row.get(column, _MISSING)) for column in columns)
        for row in rows
    )


def _singleton_rate(keys: Sequence[tuple[str, ...]]) -> float:
    counts = Counter(keys)
    if not keys:
        return 0.0
    return sum(counts[key] == 1 for key in keys) / len(keys)


def _achieved_k_from_keys(keys: Sequence[tuple[str, ...]]) -> int:
    if not keys:
        return 0
    return min(Counter(keys).values())


def _measure_column(
    rows: Sequence[Mapping[str, Any]],
    column: str,
) -> _ColumnMeasurements:
    tokens: list[str | None] = []
    non_null_values: list[Any] = []
    missing_count = 0
    for row_index, row in enumerate(rows):
        value = row.get(column, _MISSING)
        if value is _MISSING or value is None:
            missing_count += 1
            tokens.append(None)
            continue
        try:
            token = _value_token(value)
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"unsupported value in column {column!r} at row offset {row_index}"
            ) from None
        tokens.append(token)
        non_null_values.append(value)
    counts = Counter(token for token in tokens if token is not None)
    non_null_count = len(non_null_values)
    distinct_count = len(counts)
    uniqueness_count = sum(count == 1 for count in counts.values())
    distinctness = _rate(distinct_count, non_null_count)
    uniqueness_rate = _rate(uniqueness_count, non_null_count)
    unique_offsets = tuple(
        index
        for index, token in enumerate(tokens)
        if token is not None and counts[token] == 1
    )
    date_like = sum(_is_date_like(value) for value in non_null_values)
    numeric_values = [
        value
        for value in non_null_values
        if type(value) is not bool and isinstance(value, (int, float, Decimal))
    ]
    demographic_numeric = (
        bool(numeric_values)
        and len(numeric_values) == len(non_null_values)
        and all(0 <= float(value) <= 120 for value in numeric_values)
    )
    text_values = [value for value in non_null_values if isinstance(value, str)]
    mean_value_length = _rate(
        sum(len(value) for value in text_values),
        len(text_values),
    )
    return _ColumnMeasurements(
        non_null_count=non_null_count,
        missing_count=missing_count,
        distinct_count=distinct_count,
        distinctness=distinctness,
        unique_value_count=uniqueness_count,
        uniqueness_rate=uniqueness_rate,
        offsets=unique_offsets,
        date_like_ratio=_rate(date_like, non_null_count),
        demographic_numeric=demographic_numeric,
        mean_value_length=mean_value_length,
    )


def _coerce_table(records: Any) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    data = records
    to_dicts = getattr(data, "to_dicts", None)
    if callable(to_dicts):
        data = to_dicts()
    else:
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict) and not isinstance(data, Mapping):
            try:
                data = to_dict("records")
            except TypeError as exc:
                raise TypeError(
                    "DataFrame-like records must support to_dict('records')"
                ) from exc
    if isinstance(data, Mapping):
        for key in ("records", "rows", "items"):
            value = data.get(key)
            if _is_row_sequence(value):
                data = value
                break
        else:
            data = [data]
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise TypeError("records must be a sequence of row mappings")
    raw_rows = list(data)
    if not raw_rows:
        raise ValueError("records must contain at least one row")
    if not all(isinstance(row, Mapping) for row in raw_rows):
        raise TypeError("records must contain only row mappings")

    columns: list[str] = []
    seen: set[str] = set()
    for row in raw_rows:
        for field_name in row:
            if type(field_name) is not str:
                raise TypeError("table column names must be strings")
            if field_name not in seen:
                seen.add(field_name)
                columns.append(field_name)
    if not columns:
        raise ValueError("records must contain at least one column")
    normalized = [dict(row) for row in raw_rows]
    for row_index, row in enumerate(normalized):
        for column in columns:
            value = row.get(column, _MISSING)
            if value is _MISSING or value is None:
                continue
            try:
                _value_token(value)
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"unsupported value in column {column!r} at row offset {row_index}"
                ) from None
    return normalized, tuple(columns)


def _is_row_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(row, Mapping) for row in value)
    )


def _normalize_columns(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of column names")
    if not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of column names")
    columns: list[str] = []
    for column in value:
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"{name} must contain non-empty column names")
        normalized = column.strip()
        if normalized not in columns:
            columns.append(normalized)
    if not columns:
        raise ValueError(f"{name} must contain at least one column")
    return tuple(columns)


def _value_token(value: Any) -> str:
    if value is _MISSING:
        payload: Any = {"type": "missing", "value": None}
    else:
        payload = _kanon._canonical_scalar_payload(value)
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _is_date_like(value: Any) -> bool:
    if type(value) in (date, datetime):
        return True
    return isinstance(value, str) and bool(_DATE_PATTERN.match(value))


def _rate(numerator: int | float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validate_target_k(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("target_k must be an integer >= 1")


def _validate_rate(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("suppression_rate must be a finite number in [0, 1]")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("suppression_rate must be a finite number in [0, 1]")
