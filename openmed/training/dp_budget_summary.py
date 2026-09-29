"""Bounded, identifier-free summaries of privacy-budget consumption.

Governance review needs to know how much validated privacy budget a training
program consumed, grouped by model family and composition method, without
learning which sites participated or what local data they held. This module
aggregates already-validated consumption metadata into deterministic summaries:

* identifying and value-bearing fields are rejected before aggregation,
* counts below the configured cell floor are suppressed rather than reported,
* epsilon values are additionally reported through fixed buckets, and
* the input order never changes the output.

It performs no privacy accounting of its own: it does not add noise, check a
composition bound, or certify differential privacy. Records are expected to
describe values already validated by the existing accounting surfaces
(``PrivacyBudgetSpend``, ``PrivacyBudgetDecision`` and
``PrivacyBudgetLedgerExceeded``).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum
from typing import Any, Final

DP_BUDGET_SUMMARY_SCHEMA_VERSION: Final = "openmed.training.dp_budget_summary.v1"

#: Smallest count reported exactly; anything below it is suppressed.
MIN_CELL_SIZE: Final = 5
MAX_MIN_CELL_SIZE: Final = 1_000
MAX_DP_BUDGET_CONSUMPTIONS: Final = 10_000
MAX_DP_BUDGET_ROUND_INDEX: Final = 1_000_000
MAX_DP_BUDGET_EPSILON: Final = 1_000_000.0

#: Composition methods accepted for a consumption record.
COMPOSITION_METHODS: Final[tuple[str, ...]] = ("basic", "advanced")

#: Aggregate outcomes accepted for a consumption record.
OUTCOMES: Final[tuple[str, ...]] = ("allowed", "denied", "exhausted")

#: Inclusive upper bounds mapping epsilon values onto stable bucket labels.
EPSILON_BUCKETS: Final[tuple[tuple[float, str], ...]] = (
    (0.5, "le_0.5"),
    (1.0, "le_1"),
    (3.0, "le_3"),
    (8.0, "le_8"),
)
EPSILON_OVERFLOW_BUCKET: Final = "gt_8"
SUPPRESSED_COUNT_BUCKET: Final = "<min_cell_size"

_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "composition",
        "delta",
        "epsilon",
        "model_family",
        "outcome",
        "round_index",
    }
)
_FAMILY_LABEL_RE: Final = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")


class DpBudgetSummaryError(ValueError):
    """Raised when consumption metadata cannot be summarized safely."""


@dataclass(frozen=True, slots=True)
class DpBudgetConsumption:
    """One already-validated privacy-budget consumption for one round.

    Attributes:
        round_index: One-based training round the charge belongs to.
        model_family: Lowercase model-family label; never a site or user id.
        epsilon: Validated epsilon charged for the round.
        delta: Validated delta charged for the round.
        composition: Composition method, one of :data:`COMPOSITION_METHODS`.
        outcome: Aggregate outcome, one of :data:`OUTCOMES`.
    """

    round_index: int
    model_family: str
    epsilon: float
    delta: float
    composition: str
    outcome: str

    def __post_init__(self) -> None:
        """Validate and normalize one consumption record."""

        object.__setattr__(
            self,
            "round_index",
            _bounded_positive_int(
                self.round_index,
                field_name="round_index",
                maximum=MAX_DP_BUDGET_ROUND_INDEX,
            ),
        )
        object.__setattr__(
            self,
            "model_family",
            _model_family(self.model_family),
        )
        object.__setattr__(
            self,
            "epsilon",
            _bounded_float(
                self.epsilon,
                field_name="epsilon",
                maximum=MAX_DP_BUDGET_EPSILON,
            ),
        )
        object.__setattr__(
            self,
            "delta",
            _bounded_float(self.delta, field_name="delta", maximum=1.0),
        )
        if self.delta >= 1.0:
            raise DpBudgetSummaryError("delta must be less than 1")
        if self.composition not in COMPOSITION_METHODS:
            raise DpBudgetSummaryError("composition must be a supported method")
        if self.outcome not in OUTCOMES:
            raise DpBudgetSummaryError("outcome must be a supported outcome")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DpBudgetConsumption:
        """Build a consumption record from a closed JSON-style mapping.

        Args:
            payload: Mapping holding exactly the record fields.

        Returns:
            The validated consumption record.

        Raises:
            DpBudgetSummaryError: If the mapping carries identifying or
                value-bearing fields, is missing fields, or holds invalid
                values. Rejection messages never echo input values.
        """

        if not isinstance(payload, Mapping):
            raise DpBudgetSummaryError("consumption record must be a mapping")
        keys = set(payload)
        if keys - _RECORD_FIELDS:
            raise DpBudgetSummaryError(
                "invalid differential-privacy budget consumption fields"
            )
        if _RECORD_FIELDS - keys:
            raise DpBudgetSummaryError(
                "differential-privacy budget consumption is incomplete"
            )
        return cls(
            round_index=payload["round_index"],
            model_family=payload["model_family"],
            epsilon=payload["epsilon"],
            delta=payload["delta"],
            composition=payload["composition"],
            outcome=payload["outcome"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the aggregate-only record representation."""

        return {
            "composition": self.composition,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "model_family": self.model_family,
            "outcome": self.outcome,
            "round_index": self.round_index,
        }


@dataclass(frozen=True, slots=True)
class DpBudgetCell:
    """One suppressed-or-reported ``(model_family, composition)`` cell."""

    model_family: str
    composition: str
    consumption_count: int
    suppressed: bool
    outcomes: Mapping[str, Mapping[str, Any]]
    epsilon_buckets: Mapping[str, Mapping[str, Any]]
    epsilon_total: float | None
    epsilon_max: float | None
    delta_total: float | None
    delta_max: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic cell payload."""

        return {
            "composition": self.composition,
            "consumption_count": self.consumption_count,
            "delta_max": self.delta_max,
            "delta_total": self.delta_total,
            "epsilon_buckets": {
                label: dict(self.epsilon_buckets[label]) for label in _bucket_labels()
            },
            "epsilon_max": self.epsilon_max,
            "epsilon_total": self.epsilon_total,
            "model_family": self.model_family,
            "outcomes": {outcome: dict(self.outcomes[outcome]) for outcome in OUTCOMES},
            "suppressed": self.suppressed,
        }


@dataclass(frozen=True, slots=True)
class DpBudgetSummary:
    """Deterministic, bounded summary of privacy-budget consumption."""

    cells: tuple[DpBudgetCell, ...]
    consumption_count: int
    min_cell_size: int
    schema_version: str = DP_BUDGET_SUMMARY_SCHEMA_VERSION

    @property
    def suppressed_cells(self) -> int:
        """Return how many cells were suppressed as too small to report."""

        return sum(1 for cell in self.cells if cell.suppressed)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready summary payload with stable key ordering."""

        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "consumption_count": self.consumption_count,
            "min_cell_size": self.min_cell_size,
            "schema_version": self.schema_version,
            "suppressed_cells": self.suppressed_cells,
        }

    def render_json(self) -> str:
        """Render the summary as deterministic, sorted JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    def render_markdown(self) -> str:
        """Render the summary as deterministic Markdown tables."""

        lines = [
            "# Differential-privacy budget summary",
            "",
            f"- schema: {self.schema_version}",
            f"- consumptions: {self.consumption_count}",
            f"- cells: {len(self.cells)}",
            f"- suppressed cells: {self.suppressed_cells}",
            f"- minimum reported cell: {self.min_cell_size}",
        ]
        for cell in self.cells:
            lines.extend(
                [
                    "",
                    f"## `{cell.model_family}` / `{cell.composition}`",
                    "",
                ]
            )
            if cell.suppressed:
                lines.append(
                    "Suppressed: fewer than the minimum reported cell of "
                    f"{self.min_cell_size} consumptions."
                )
                continue
            lines.extend(["| outcome | count |", "| --- | --- |"])
            lines.extend(
                f"| {outcome} | {_count_text(cell.outcomes[outcome])} |"
                for outcome in OUTCOMES
            )
            lines.extend(
                [
                    "",
                    "| epsilon bucket | count |",
                    "| --- | --- |",
                ]
            )
            lines.extend(
                f"| {label} | {_count_text(cell.epsilon_buckets[label])} |"
                for label in _bucket_labels()
            )
            lines.extend(
                [
                    "",
                    "| metric | value |",
                    "| --- | --- |",
                    f"| epsilon total | {_number_text(cell.epsilon_total)} |",
                    f"| epsilon max | {_number_text(cell.epsilon_max)} |",
                    f"| delta total | {_number_text(cell.delta_total)} |",
                    f"| delta max | {_number_text(cell.delta_max)} |",
                ]
            )
        return "\n".join(lines)


def build_dp_budget_summary(
    consumptions: Sequence[DpBudgetConsumption | Mapping[str, Any]],
    *,
    min_cell_size: int = MIN_CELL_SIZE,
) -> DpBudgetSummary:
    """Aggregate validated consumption metadata into a bounded summary.

    Args:
        consumptions: Consumption records, or mappings accepted by
            :meth:`DpBudgetConsumption.from_dict`.
        min_cell_size: Smallest count reported exactly; smaller counts and the
            numeric totals of smaller cells are suppressed.

    Returns:
        A deterministic summary whose cell order depends only on the
        ``(model_family, composition)`` labels.

    Raises:
        DpBudgetSummaryError: If any record is invalid, a round is duplicated
            within a model family, or the input exceeds the bounded limits.
    """

    floor = _min_cell_size(min_cell_size)
    if isinstance(consumptions, (str, bytes)) or not isinstance(consumptions, Sequence):
        raise DpBudgetSummaryError("consumptions must be a sequence of records")
    if len(consumptions) > MAX_DP_BUDGET_CONSUMPTIONS:
        raise DpBudgetSummaryError("too many consumption records to summarize")

    records: list[DpBudgetConsumption] = []
    for entry in consumptions:
        if isinstance(entry, DpBudgetConsumption):
            records.append(entry)
        else:
            records.append(DpBudgetConsumption.from_dict(entry))

    _reject_duplicate_rounds(records)
    groups: dict[tuple[str, str], list[DpBudgetConsumption]] = {}
    for record in records:
        groups.setdefault((record.model_family, record.composition), []).append(record)
    cells = tuple(_cell(key, groups[key], floor) for key in sorted(groups))
    return DpBudgetSummary(
        cells=cells,
        consumption_count=len(records),
        min_cell_size=floor,
    )


def build_dp_budget_summary_from_dicts(
    payloads: Sequence[Mapping[str, Any]],
    *,
    min_cell_size: int = MIN_CELL_SIZE,
) -> DpBudgetSummary:
    """Aggregate JSON-style consumption mappings into a bounded summary."""

    return build_dp_budget_summary(payloads, min_cell_size=min_cell_size)


def _cell(
    key: tuple[str, str],
    records: Sequence[DpBudgetConsumption],
    floor: int,
) -> DpBudgetCell:
    family, composition = key
    count = len(records)
    suppressed = count < floor
    outcome_counts = Counter(record.outcome for record in records)
    bucket_counts = Counter(_epsilon_bucket(record.epsilon) for record in records)
    epsilons = sorted(record.epsilon for record in records)
    deltas = sorted(record.delta for record in records)
    return DpBudgetCell(
        model_family=family,
        composition=composition,
        consumption_count=count,
        suppressed=suppressed,
        outcomes={
            outcome: _count_payload(outcome_counts.get(outcome, 0), floor)
            for outcome in OUTCOMES
        },
        epsilon_buckets={
            label: _count_payload(bucket_counts.get(label, 0), floor)
            for label in _bucket_labels()
        },
        epsilon_total=None if suppressed else _rounded(fsum(epsilons)),
        epsilon_max=None if suppressed else (epsilons[-1] if epsilons else 0.0),
        delta_total=None if suppressed else _rounded(fsum(deltas)),
        delta_max=None if suppressed else (deltas[-1] if deltas else 0.0),
    )


def _count_payload(count: int, floor: int) -> dict[str, Any]:
    if count < floor:
        return {
            "count": None,
            "count_bucket": SUPPRESSED_COUNT_BUCKET,
            "suppressed": True,
        }
    return {"count": count, "suppressed": False}


def _bucket_labels() -> tuple[str, ...]:
    return tuple(label for _, label in EPSILON_BUCKETS) + (EPSILON_OVERFLOW_BUCKET,)


def _epsilon_bucket(value: float) -> str:
    for upper_bound, label in EPSILON_BUCKETS:
        if value <= upper_bound:
            return label
    return EPSILON_OVERFLOW_BUCKET


def _reject_duplicate_rounds(records: Sequence[DpBudgetConsumption]) -> None:
    seen: set[tuple[str, int]] = set()
    for record in records:
        key = (record.model_family, record.round_index)
        if key in seen:
            raise DpBudgetSummaryError("duplicate round index for a model family")
        seen.add(key)


def _model_family(value: Any) -> str:
    if not isinstance(value, str) or not _FAMILY_LABEL_RE.fullmatch(value):
        raise DpBudgetSummaryError("model_family must be a lowercase label")
    return value


def _bounded_positive_int(value: Any, *, field_name: str, maximum: int) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise DpBudgetSummaryError(f"{field_name} must be a bounded positive integer")
    return value


def _bounded_float(value: Any, *, field_name: str, maximum: float) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise DpBudgetSummaryError(f"{field_name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > maximum:
        raise DpBudgetSummaryError(f"{field_name} must be a bounded finite number")
    return parsed


def _min_cell_size(value: Any) -> int:
    if type(value) is not int or value < 2 or value > MAX_MIN_CELL_SIZE:
        raise DpBudgetSummaryError("min_cell_size must be a bounded integer >= 2")
    return value


def _rounded(value: float) -> float:
    return round(value, 6)


def _count_text(payload: Mapping[str, Any]) -> str:
    return (
        f"suppressed ({SUPPRESSED_COUNT_BUCKET})"
        if payload.get("suppressed")
        else str(payload.get("count"))
    )


def _number_text(value: float | None) -> str:
    return "suppressed" if value is None else f"{value:.6g}"


__all__ = [
    "COMPOSITION_METHODS",
    "DP_BUDGET_SUMMARY_SCHEMA_VERSION",
    "DpBudgetCell",
    "DpBudgetConsumption",
    "DpBudgetSummary",
    "DpBudgetSummaryError",
    "EPSILON_BUCKETS",
    "EPSILON_OVERFLOW_BUCKET",
    "MAX_DP_BUDGET_CONSUMPTIONS",
    "MIN_CELL_SIZE",
    "OUTCOMES",
    "SUPPRESSED_COUNT_BUCKET",
    "build_dp_budget_summary",
    "build_dp_budget_summary_from_dicts",
]
