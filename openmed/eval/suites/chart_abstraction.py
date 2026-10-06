"""Offline chart-abstraction scoring with value-free aggregate reports."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from statistics import NormalDist
from typing import Any

from openmed.agent.workflows.abstraction_evidence import AbstractionEvidenceChain

_FIELD_ID = re.compile(r"[a-z][a-z0-9_]{0,63}(?:\.[a-z][a-z0-9_]{0,63}){0,7}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_Z95 = NormalDist().inv_cdf(0.975)
AbstractionValue = str | int | float | bool | None


class AbstractionFieldType(str, Enum):
    """Closed field-type slices; normalization never infers units or synonyms."""

    TEXT = "text"
    CATEGORICAL = "categorical"
    NUMBER = "number"
    BOOLEAN = "boolean"


def _validate_identity(case_id: str, field_id: str) -> None:
    if type(case_id) is not str or not case_id or len(case_id) > 128:
        raise ValueError("invalid_case_id")
    if type(field_id) is not str or _FIELD_ID.fullmatch(field_id) is None:
        raise ValueError("invalid_field_id")


def _validate_value(value: AbstractionValue) -> None:
    if value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("invalid_value")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("invalid_value")


def _normalize(value: AbstractionValue, kind: AbstractionFieldType) -> Any:
    if kind in (AbstractionFieldType.TEXT, AbstractionFieldType.CATEGORICAL):
        if type(value) is str:
            return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    elif kind is AbstractionFieldType.BOOLEAN:
        if type(value) is bool:
            return value
    elif kind is AbstractionFieldType.NUMBER:
        if type(value) in (str, int, float):
            try:
                number = Decimal(str(value))
                if number.is_finite():
                    return number
            except (InvalidOperation, ValueError):
                pass
    return None


@dataclass(frozen=True, slots=True)
class ChartAbstractionGoldField:
    """One private gold value; ``None`` marks an unanswerable field.

    Args:
        case_id: Private join key, never emitted in reports.
        field_id: Developer-authored schema identifier, never a patient identifier.
        field_type: Closed type controlling normalization and report slices.
        value: Scalar gold value, or None for an unanswerable field.
    """

    case_id: str = field(repr=False)
    field_id: str
    field_type: AbstractionFieldType
    value: AbstractionValue = field(repr=False)

    def __post_init__(self) -> None:
        _validate_identity(self.case_id, self.field_id)
        _validate_value(self.value)
        if type(self.field_type) is not AbstractionFieldType:
            raise ValueError("invalid_field_type")
        if self.value is not None and _normalize(self.value, self.field_type) is None:
            raise ValueError("invalid_gold_value")


@dataclass(frozen=True, slots=True)
class ChartAbstractionPrediction:
    """One private output and its existing normalized-fact digest binding.

    Args:
        case_id: Private join key matching a gold case.
        field_id: Schema identifier matching a gold field.
        value: Scalar output, or None for an explicit abstention.
        normalized_fact_digest: Producer's digest for the submitted fact. No fact
            encoding is prescribed here; use the evidence producer's encoding.
        evidence_chain: Existing source chain for that exact submitted fact.
    """

    case_id: str = field(repr=False)
    field_id: str
    value: AbstractionValue = field(repr=False)
    normalized_fact_digest: str | None = field(default=None, repr=False)
    evidence_chain: AbstractionEvidenceChain | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_identity(self.case_id, self.field_id)
        _validate_value(self.value)
        digest = self.normalized_fact_digest
        if digest is not None and (
            type(digest) is not str or _DIGEST.fullmatch(digest) is None
        ):
            raise ValueError("invalid_fact_digest")
        if self.evidence_chain is not None and (
            type(self.evidence_chain) is not AbstractionEvidenceChain
        ):
            raise ValueError("invalid_evidence_chain")


@dataclass(frozen=True, slots=True)
class AbstractionMetric:
    """Binomial success and trial counts with a deterministic 95% Wilson interval.

    Args:
        success_count: Observed successes (or events for an error metric).
        total_count: Eligible trials. Zero trials produce a null interval.
    """

    success_count: int
    total_count: int

    def __post_init__(self) -> None:
        if (
            type(self.success_count) is not int
            or type(self.total_count) is not int
            or not 0 <= self.success_count <= self.total_count
        ):
            raise ValueError("invalid_metric_counts")

    @property
    def interval(self) -> tuple[float, float] | None:
        """Return Wilson bounds, or None when no trials were observed."""
        n = self.total_count
        if not n:
            return None
        p = self.success_count / n
        z2 = _Z95 * _Z95
        denominator = 1 + z2 / n
        center = (p + z2 / (2 * n)) / denominator
        margin = _Z95 * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denominator
        return (max(0.0, center - margin), min(1.0, center + margin))

    def to_dict(self) -> dict[str, object]:
        """Return counts and interval only; rates are derived from the counts."""
        bounds = self.interval
        return {
            "success_count": self.success_count,
            "total_count": self.total_count,
            "interval": list(bounds) if bounds is not None else None,
        }


@dataclass(frozen=True, slots=True)
class AbstractionScores:
    """Counts and intervals for an aggregate, field, or field-type slice."""

    field_count: int
    answerable_count: int
    unanswerable_count: int
    missing_count: int
    answered_count: int
    abstained_count: int
    exact_agreement: AbstractionMetric
    normalized_agreement: AbstractionMetric
    abstention_correctness: AbstractionMetric
    answerable_abstention: AbstractionMetric
    unanswerable_answer: AbstractionMetric
    evidence_support: AbstractionMetric

    def to_dict(self) -> dict[str, object]:
        """Return only counts and intervals without retaining evaluated values."""
        return {
            "field_count": self.field_count,
            "answerable_count": self.answerable_count,
            "unanswerable_count": self.unanswerable_count,
            "missing_count": self.missing_count,
            "answered_count": self.answered_count,
            "abstained_count": self.abstained_count,
            "exact_agreement": self.exact_agreement.to_dict(),
            "normalized_agreement": self.normalized_agreement.to_dict(),
            "abstention_correctness": self.abstention_correctness.to_dict(),
            "answerable_abstention": self.answerable_abstention.to_dict(),
            "unanswerable_answer": self.unanswerable_answer.to_dict(),
            "evidence_support": self.evidence_support.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ChartAbstractionBenchmarkReport:
    """Value-free scores with sorted field IDs and closed field-type slices."""

    overall: AbstractionScores
    by_field: tuple[tuple[str, AbstractionScores], ...]
    by_field_type: tuple[tuple[AbstractionFieldType, AbstractionScores], ...]

    def to_dict(self) -> dict[str, object]:
        """Return field IDs, closed slice keys, counts and intervals only."""
        return {
            "overall": self.overall.to_dict(),
            "by_field": {key: scores.to_dict() for key, scores in self.by_field},
            "by_field_type": {
                key.value: scores.to_dict() for key, scores in self.by_field_type
            },
        }

    def to_json(self) -> str:
        """Return byte-stable JSON without clinical values or case identities."""
        import json

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _scores(
    rows: Sequence[ChartAbstractionGoldField],
    predictions: dict[tuple[str, str], ChartAbstractionPrediction],
) -> AbstractionScores:
    answerable = missing = answered = abstained = exact = normalized = 0
    abstention_correct = false_abstention = unsafe_answer = supported = 0
    for gold in rows:
        can_answer = gold.value is not None
        answerable += can_answer
        output = predictions.get((gold.case_id, gold.field_id))
        if output is None:
            missing += 1
            continue
        if output.value is None:
            abstained += 1
            false_abstention += can_answer
            abstention_correct += not can_answer
            continue
        answered += 1
        unsafe_answer += not can_answer
        abstention_correct += can_answer
        if can_answer:
            exact += (
                type(output.value) is type(gold.value) and output.value == gold.value
            )
            normalized += _normalize(output.value, gold.field_type) == _normalize(
                gold.value, gold.field_type
            )
        chain = output.evidence_chain
        supported += (
            chain is not None
            and chain.field_id == gold.field_id
            and chain.has_clinical_source
            and output.normalized_fact_digest == chain.normalized_fact_digest
        )
    unanswerable = len(rows) - answerable
    return AbstractionScores(
        field_count=len(rows),
        answerable_count=answerable,
        unanswerable_count=unanswerable,
        missing_count=missing,
        answered_count=answered,
        abstained_count=abstained,
        exact_agreement=AbstractionMetric(exact, answerable),
        normalized_agreement=AbstractionMetric(normalized, answerable),
        abstention_correctness=AbstractionMetric(abstention_correct, len(rows)),
        answerable_abstention=AbstractionMetric(false_abstention, answerable),
        unanswerable_answer=AbstractionMetric(unsafe_answer, unanswerable),
        evidence_support=AbstractionMetric(supported, answered),
    )


def run_chart_abstraction_benchmark(
    gold_fields: Sequence[ChartAbstractionGoldField],
    predictions: Sequence[ChartAbstractionPrediction],
) -> ChartAbstractionBenchmarkReport:
    """Score every gold field, treating absent outputs as missing, not abstained.

    Args:
        gold_fields: Nonempty synthetic or separately governed gold set. Field IDs
            must be developer-authored schema identifiers, not sensitive values.
        predictions: Outputs joined by case and field, with optional fact-bound
            source chains. Unknown or duplicate join keys are rejected.

    Returns:
        Deterministic overall, per-field and per-type counts and Wilson intervals.
        Exact/normalized denominators include all answerable gold fields;
        evidence-support denominators include all submitted non-abstained answers.

    Raises:
        ValueError: A controlled code for malformed or ambiguous inputs.
    """
    rows = tuple(gold_fields)
    outputs = tuple(predictions)
    if not rows or any(type(row) is not ChartAbstractionGoldField for row in rows):
        raise ValueError("invalid_gold_fields")
    if any(type(output) is not ChartAbstractionPrediction for output in outputs):
        raise ValueError("invalid_predictions")
    keys = {(row.case_id, row.field_id) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate_gold_field")
    types: dict[str, AbstractionFieldType] = {}
    fields: dict[str, list[ChartAbstractionGoldField]] = {}
    slices: dict[AbstractionFieldType, list[ChartAbstractionGoldField]] = {}
    for row in rows:
        if row.field_id in types and types[row.field_id] is not row.field_type:
            raise ValueError("inconsistent_field_type")
        types[row.field_id] = row.field_type
        fields.setdefault(row.field_id, []).append(row)
        slices.setdefault(row.field_type, []).append(row)
    by_key: dict[tuple[str, str], ChartAbstractionPrediction] = {}
    for output in outputs:
        key = (output.case_id, output.field_id)
        if key not in keys:
            raise ValueError("unknown_prediction_field")
        if key in by_key:
            raise ValueError("duplicate_prediction_field")
        by_key[key] = output
    return ChartAbstractionBenchmarkReport(
        overall=_scores(rows, by_key),
        by_field=tuple((key, _scores(fields[key], by_key)) for key in sorted(fields)),
        by_field_type=tuple(
            (key, _scores(slices[key], by_key))
            for key in sorted(slices, key=lambda kind: kind.value)
        ),
    )


def synthetic_chart_abstraction_gold() -> tuple[ChartAbstractionGoldField, ...]:
    """Return twelve offline synthetic fields across four types and three cases.

    Two cases are answerable; the third has no gold answer for any field. These
    invented values test scorer mechanics and establish no clinical validity.
    """
    fields = (
        ("registry.label", AbstractionFieldType.TEXT, ("Synthetic Alpha", "Test Beta")),
        ("registry.status", AbstractionFieldType.CATEGORICAL, ("code_a", "code_b")),
        ("registry.measurement", AbstractionFieldType.NUMBER, (12, 2.5)),
        ("registry.flag", AbstractionFieldType.BOOLEAN, (True, False)),
    )
    return tuple(
        ChartAbstractionGoldField(f"synthetic_{index}", field_id, kind, value)
        for field_id, kind, values in fields
        for index, value in enumerate((*values, None))
    )


__all__ = [
    "AbstractionFieldType",
    "AbstractionMetric",
    "AbstractionScores",
    "ChartAbstractionBenchmarkReport",
    "ChartAbstractionGoldField",
    "ChartAbstractionPrediction",
    "run_chart_abstraction_benchmark",
    "synthetic_chart_abstraction_gold",
]
