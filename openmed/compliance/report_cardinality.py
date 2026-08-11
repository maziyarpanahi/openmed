"""Deterministic, PHI-safe cardinality budgets for typed report shapes.

Only JSON-like report values are traversed.  The evaluator records paths,
constraint names, and counts, but never copies or formats report values.  An
unsupported, cyclic, or otherwise malformed shape fails closed.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

ReportScalar: TypeAlias = str | int | float | bool | None
ReportValue: TypeAlias = (
    ReportScalar
    | Mapping[str, "ReportValue"]
    | list["ReportValue"]
    | tuple["ReportValue", ...]
)

_DEFAULT_MAX_ITEMS_PER_FIELD: Final = 100
_DEFAULT_MAX_UNIQUE_KEYS: Final = 100
_DEFAULT_MAX_NESTING_DEPTH: Final = 8
_DEFAULT_MAX_AGGREGATE_ITEMS: Final = 1_000
_SCHEMA_FIELD_NAME_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


@dataclass(frozen=True)
class ReportCardinalityBudget:
    """Limits applied to a JSON-like report shape.

    ``max_items_per_field`` applies to every list, tuple, and mapping at one
    path. ``max_unique_keys`` applies to mapping keys. ``max_nesting_depth``
    counts containers from the root, where the root has depth zero.
    ``max_aggregate_items`` is the sum of items in every traversed container,
    including the root container.
    """

    max_items_per_field: int = _DEFAULT_MAX_ITEMS_PER_FIELD
    max_unique_keys: int = _DEFAULT_MAX_UNIQUE_KEYS
    max_nesting_depth: int = _DEFAULT_MAX_NESTING_DEPTH
    max_aggregate_items: int = _DEFAULT_MAX_AGGREGATE_ITEMS

    def __post_init__(self) -> None:
        """Reject ambiguous limits without echoing caller-supplied data."""

        for name, value in (
            ("max_items_per_field", self.max_items_per_field),
            ("max_unique_keys", self.max_unique_keys),
            ("max_nesting_depth", self.max_nesting_depth),
            ("max_aggregate_items", self.max_aggregate_items),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def max_total_items(self) -> int:
        """Return the aggregate limit under its common alternate name."""

        return self.max_aggregate_items


DEFAULT_REPORT_CARDINALITY_BUDGET: Final = ReportCardinalityBudget()


@dataclass(frozen=True)
class ReportCardinalityViolation:
    """A safe summary of one cardinality or shape violation."""

    path: str
    rule: str
    count: int
    limit: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return only safe path and numeric constraint metadata."""

        return {
            "count": self.count,
            "limit": self.limit,
            "path": self.path,
            "rule": self.rule,
        }


@dataclass(frozen=True)
class ReportCardinalityReport:
    """Deterministic result of evaluating one report shape."""

    allowed: bool
    aggregate_items: int
    max_depth: int
    violations: tuple[ReportCardinalityViolation, ...]
    schema_version: int = 1

    @property
    def within_budget(self) -> bool:
        """Return whether the report passed every fail-closed check."""

        return self.allowed

    @property
    def failed_closed(self) -> bool:
        """Return whether the report was rejected by the evaluator."""

        return not self.allowed

    def __bool__(self) -> bool:
        """Allow a report result to be used as its safe allow/deny decision."""

        return self.allowed

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report without report values."""

        return {
            "aggregate_items": self.aggregate_items,
            "allowed": self.allowed,
            "max_depth": self.max_depth,
            "schema_version": self.schema_version,
            "violations": [violation.to_dict() for violation in self.violations],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return deterministic JSON containing paths and counts only."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )


class _CardinalityEvaluator:
    """Stateful, single-use evaluator kept private to the public checker."""

    def __init__(self, budget: ReportCardinalityBudget) -> None:
        self._budget = budget
        self._active_containers: set[int] = set()
        self._aggregate_items = 0
        self._max_depth = 0
        self._stopped = False
        self._violations: set[ReportCardinalityViolation] = set()

    def evaluate(self, report: Any) -> ReportCardinalityReport:
        """Evaluate *report* without retaining any of its values."""

        self._walk(report, "$", 0)
        violations = tuple(
            sorted(
                self._violations,
                key=lambda item: (
                    item.path,
                    item.rule,
                    item.count,
                    -1 if item.limit is None else item.limit,
                ),
            )
        )
        return ReportCardinalityReport(
            allowed=not violations,
            aggregate_items=self._aggregate_items,
            max_depth=self._max_depth,
            violations=violations,
        )

    def _add_violation(
        self,
        *,
        path: str,
        rule: str,
        count: int,
        limit: int | None,
    ) -> None:
        self._violations.add(
            ReportCardinalityViolation(
                path=path,
                rule=rule,
                count=count,
                limit=limit,
            )
        )

    def _walk(self, value: Any, path: str, depth: int) -> None:
        if self._stopped:
            return

        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                self._add_violation(
                    path=path,
                    rule="non_finite_scalar",
                    count=1,
                    limit=0,
                )
            return

        if isinstance(value, Mapping):
            self._walk_mapping(value, path, depth)
            return
        if isinstance(value, (list, tuple)):
            self._walk_sequence(value, path, depth)
            return

        self._add_violation(
            path=path,
            rule="unsupported_shape",
            count=1,
            limit=0,
        )

    def _walk_mapping(self, value: Mapping[Any, Any], path: str, depth: int) -> None:
        container_id = id(value)
        if container_id in self._active_containers:
            self._add_violation(path=path, rule="cycle", count=1, limit=0)
            return

        try:
            pairs = list(value.items())
        except Exception:
            self._add_violation(
                path=path,
                rule="unsupported_shape",
                count=1,
                limit=0,
            )
            return

        invalid_key_count = sum(1 for key, _ in pairs if not isinstance(key, str))
        if invalid_key_count:
            self._add_violation(
                path=path,
                rule="non_string_key",
                count=invalid_key_count,
                limit=0,
            )
            return

        unique_key_count = len({key for key, _ in pairs})
        if not self._begin_container(
            path=path,
            depth=depth,
            item_count=len(pairs),
            unique_key_count=unique_key_count,
        ):
            return

        self._active_containers.add(container_id)
        try:
            for key, item in sorted(pairs, key=lambda pair: pair[0]):
                self._walk(item, _append_field(path, key), depth + 1)
                if self._stopped:
                    return
        finally:
            self._active_containers.remove(container_id)

    def _walk_sequence(
        self, value: list[Any] | tuple[Any, ...], path: str, depth: int
    ) -> None:
        container_id = id(value)
        if container_id in self._active_containers:
            self._add_violation(path=path, rule="cycle", count=1, limit=0)
            return

        try:
            item_count = len(value)
        except Exception:
            self._add_violation(
                path=path,
                rule="unsupported_shape",
                count=1,
                limit=0,
            )
            return

        if not self._begin_container(
            path=path,
            depth=depth,
            item_count=item_count,
            unique_key_count=None,
        ):
            return

        self._active_containers.add(container_id)
        try:
            for item in value:
                self._walk(item, _append_item(path), depth + 1)
                if self._stopped:
                    return
        finally:
            self._active_containers.remove(container_id)

    def _begin_container(
        self,
        *,
        path: str,
        depth: int,
        item_count: int,
        unique_key_count: int | None,
    ) -> bool:
        self._max_depth = max(self._max_depth, depth)
        self._aggregate_items += item_count

        blocked = False
        if depth > self._budget.max_nesting_depth:
            self._add_violation(
                path=path,
                rule="nesting_depth",
                count=depth,
                limit=self._budget.max_nesting_depth,
            )
            blocked = True
        if item_count > self._budget.max_items_per_field:
            self._add_violation(
                path=path,
                rule="items_per_field",
                count=item_count,
                limit=self._budget.max_items_per_field,
            )
            blocked = True
        if (
            unique_key_count is not None
            and unique_key_count > self._budget.max_unique_keys
        ):
            self._add_violation(
                path=path,
                rule="unique_keys",
                count=unique_key_count,
                limit=self._budget.max_unique_keys,
            )
            blocked = True

        if self._aggregate_items > self._budget.max_aggregate_items:
            self._add_violation(
                path=path,
                rule="aggregate_items",
                count=self._aggregate_items,
                limit=self._budget.max_aggregate_items,
            )
            self._stopped = True
            blocked = True

        return not blocked


def _append_field(path: str, key: str) -> str:
    """Append a schema-like key without exposing arbitrary map keys."""

    if _SCHEMA_FIELD_NAME_RE.fullmatch(key):
        return f"{path}.{key}"
    return f"{path}[key]"


def _append_item(path: str) -> str:
    """Append a wildcard item segment to a field path."""

    return f"{path}[*]"


def _failed_closed_report(rule: str) -> ReportCardinalityReport:
    return ReportCardinalityReport(
        allowed=False,
        aggregate_items=0,
        max_depth=0,
        violations=(ReportCardinalityViolation(path="$", rule=rule, count=1, limit=0),),
    )


def check_report_cardinality(
    report: ReportValue,
    budget: ReportCardinalityBudget | None = None,
) -> ReportCardinalityReport:
    """Return a deterministic, fail-closed cardinality decision for *report*.

    Accepted shapes contain JSON-like scalars, mappings with string keys, and
    lists or tuples.  Unknown objects, non-finite numbers, non-string mapping
    keys, and cycles are rejected.  The returned report contains no input
    values and performs no network or filesystem access.
    """

    if budget is None:
        budget = DEFAULT_REPORT_CARDINALITY_BUDGET
    if not isinstance(budget, ReportCardinalityBudget):
        return _failed_closed_report("invalid_budget")
    return _CardinalityEvaluator(budget).evaluate(report)


def evaluate_report_cardinality(
    report: ReportValue,
    budget: ReportCardinalityBudget | None = None,
) -> ReportCardinalityReport:
    """Alias for :func:`check_report_cardinality` with evaluation wording."""

    return check_report_cardinality(report, budget)


def enforce_report_cardinality(
    report: ReportValue,
    budget: ReportCardinalityBudget | None = None,
) -> ReportCardinalityReport:
    """Return the fail-closed enforcement result for *report*.

    Enforcement is represented as a structured decision so callers can reject
    a report without raising an exception that might include a raw value.
    """

    return check_report_cardinality(report, budget)


CardinalityBudget = ReportCardinalityBudget
CardinalityViolation = ReportCardinalityViolation
CardinalityReport = ReportCardinalityReport

__all__ = [
    "CardinalityBudget",
    "CardinalityReport",
    "CardinalityViolation",
    "DEFAULT_REPORT_CARDINALITY_BUDGET",
    "ReportCardinalityBudget",
    "ReportCardinalityReport",
    "ReportCardinalityViolation",
    "ReportScalar",
    "ReportValue",
    "check_report_cardinality",
    "enforce_report_cardinality",
    "evaluate_report_cardinality",
]
