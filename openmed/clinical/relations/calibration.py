"""Per-type relation calibration, selective prediction, and abstention.

Calibration and empirical threshold selection are deliberately separate. The
operating points in this module are held-out point estimates; they are not
finite-sample conformal, Learn-then-Test, or RCPS guarantees. A future certified
layer can consume the emitted raw counts and empirical risks without changing
the runtime abstention representation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from openmed.core.audit import stable_hash
from openmed.core.decoding import SpanEdge
from openmed.eval.calibrate import (
    RELATION_REPORT_ARTIFACT,
    RELATION_REPORT_SCHEMA_VERSION,
    CalibrationSample,
    IsotonicCalibrationModel,
    fit_isotonic_probability_calibrator,
    fit_temperature_scaling,
    temperature_scale_probability,
)
from openmed.eval.metrics import (
    relation_reliability_report,
    risk_coverage_curve,
    selective_prediction_report,
)

RELATION_CALIBRATION_SCHEMA_VERSION = 1
RELATION_CALIBRATION_ADVISORY = (
    "Calibrated relation probabilities and empirical abstention thresholds are "
    "assistive outputs, not certified risk guarantees or autonomous clinical "
    "decisions. Uncertain relations require review."
)
RELATION_STATUS_ASSERTED = "asserted"
RELATION_STATUS_UNCERTAIN = "uncertain"
DEFAULT_MIN_ISOTONIC_SAMPLES = 8
DEFAULT_RELATION_RELIABILITY_BINS = 10
DEFAULT_MAX_RELATION_ABSTENTION_RATE = 0.35
DEFAULT_MIN_RETAINED_RELATION_ACCURACY = 0.80

RelationCalibrationMethod = Literal["isotonic", "pooled_temperature"]


@dataclass(frozen=True)
class RelationCalibrationRecord:
    """One labeled relation score used for calibration or held-out evaluation."""

    relation_type: str
    score: float
    correct: bool
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relation_type = _normalize_relation_type(self.relation_type)
        score = _bounded_probability(self.score, "score")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("relation calibration weight must be positive")
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "correct", bool(self.correct))
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RelationCalibrationRecord":
        """Parse a JSON-style relation calibration row."""

        relation_type = data.get("relation_type", data.get("label", data.get("type")))
        if relation_type is None:
            raise ValueError("relation calibration record requires relation_type")
        raw_score = data.get(
            "score",
            data.get("confidence", data.get("raw_relation_score")),
        )
        if raw_score is None:
            raise ValueError("relation calibration record requires score")
        correctness = data.get(
            "correct",
            data.get("target", data.get("matched", data.get("is_correct"))),
        )
        if correctness is None:
            raise ValueError("relation calibration record requires correctness")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        return cls(
            relation_type=str(relation_type),
            score=float(raw_score),
            correct=_coerce_bool(correctness),
            weight=float(data.get("weight", 1.0)),
            metadata=metadata,
        )

    def to_metric_record(self, *, confidence: float | None = None) -> dict[str, Any]:
        """Return a raw-text-free input for calibration/selective metrics."""

        return {
            "relation_type": self.relation_type,
            "confidence": self.score if confidence is None else confidence,
            "correct": self.correct,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class RelationTypeCalibrator:
    """Probability mapping for one relation type or the pooled fallback."""

    relation_type: str
    method: RelationCalibrationMethod
    sample_count: int
    positive_count: int
    total_weight: float
    isotonic: IsotonicCalibrationModel | None = None
    temperature: float | None = None

    def predict(self, score: float) -> float:
        """Return a calibrated probability for one raw relation score."""

        if self.method == "isotonic":
            if self.isotonic is None:
                raise ValueError("isotonic relation calibrator is missing its model")
            return self.isotonic.predict(score)
        if self.temperature is None:
            raise ValueError("temperature relation calibrator is missing temperature")
        return temperature_scale_probability(score, self.temperature)

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned, JSON-compatible calibrator payload."""

        payload: dict[str, Any] = {
            "relation_type": self.relation_type,
            "method": self.method,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "total_weight": self.total_weight,
        }
        if self.isotonic is not None:
            payload["isotonic"] = self.isotonic.to_dict()
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload


@dataclass(frozen=True)
class RelationCalibrator:
    """Independent per-type relation calibrators with a pooled fallback."""

    groups: Mapping[str, RelationTypeCalibrator]
    fallback: RelationTypeCalibrator
    min_isotonic_samples: int = DEFAULT_MIN_ISOTONIC_SAMPLES
    schema_version: int = RELATION_CALIBRATION_SCHEMA_VERSION

    def predict(self, *, relation_type: str, score: float) -> float:
        """Return a calibrated probability using the type-specific mapping."""

        group = self.groups.get(_normalize_relation_type(relation_type), self.fallback)
        return group.predict(score)

    def method_for(self, relation_type: str) -> RelationCalibrationMethod:
        """Return the selected calibration method for a relation type."""

        group = self.groups.get(_normalize_relation_type(relation_type), self.fallback)
        return group.method

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, versioned calibrator artifact section."""

        return {
            "schema_version": self.schema_version,
            "min_isotonic_samples": self.min_isotonic_samples,
            "groups": [
                group.to_dict()
                for _, group in sorted(self.groups.items(), key=lambda item: item[0])
            ],
            "fallback": self.fallback.to_dict(),
        }


@dataclass(frozen=True)
class RelationOperatingPoint:
    """Empirical relation threshold selected under a fixed abstention budget."""

    relation_type: str
    confidence_threshold: float
    accuracy: float
    empirical_risk: float
    coverage: float
    abstention_rate: float
    retained_count: int
    retained_weight: float
    total_count: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return an LTT/RCPS-ready empirical operating-point row."""

        return {
            "relation_type": self.relation_type,
            "confidence_threshold": self.confidence_threshold,
            "accuracy": self.accuracy,
            "empirical_risk": self.empirical_risk,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "retained_count": self.retained_count,
            "retained_weight": self.retained_weight,
            "total_count": self.total_count,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class CalibratedRelation:
    """One calibrated relation retained as asserted or marked uncertain."""

    relation: SpanEdge
    raw_score: float
    calibrated_probability: float
    confidence_threshold: float
    status: str
    calibration_method: RelationCalibrationMethod

    @property
    def asserted(self) -> bool:
        """Return whether this relation belongs in the asserted-fact set."""

        return self.status == RELATION_STATUS_ASSERTED

    def to_dict(self) -> dict[str, Any]:
        """Return the operational relation, including explicit uncertainty."""

        return {
            "relation": self.relation.to_dict(),
            "raw_score": self.raw_score,
            "calibrated_probability": self.calibrated_probability,
            "confidence_threshold": self.confidence_threshold,
            "status": self.status,
            "asserted": self.asserted,
            "calibration_method": self.calibration_method,
        }

    def to_audit_entry(self) -> dict[str, Any]:
        """Return a raw-text-free audit record with hashed endpoint identity."""

        relation_key_hash = stable_hash(
            {
                "head": self.relation.head,
                "tail": self.relation.tail,
                "relation_type": self.relation.label,
            }
        )
        payload: dict[str, Any] = {
            "relation_type": self.relation.label,
            "relation_key_hash": relation_key_hash,
            "raw_score": self.raw_score,
            "calibrated_probability": self.calibrated_probability,
            "confidence_threshold": self.confidence_threshold,
            "status": self.status,
            "asserted": self.asserted,
            "calibration_method": self.calibration_method,
            "schema_version": RELATION_CALIBRATION_SCHEMA_VERSION,
        }
        payload["content_hash"] = stable_hash(payload)
        return payload


@dataclass(frozen=True)
class RelationAbstentionResult:
    """All calibrated relations plus separated asserted and uncertain sets."""

    relations: tuple[CalibratedRelation, ...]
    asserted: tuple[CalibratedRelation, ...]
    uncertain: tuple[CalibratedRelation, ...]
    audit_trace: tuple[Mapping[str, Any], ...]

    @property
    def asserted_edges(self) -> tuple[SpanEdge, ...]:
        """Return only edges safe for the default asserted-fact graph."""

        return tuple(relation.relation for relation in self.asserted)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible abstention result."""

        return {
            "relations": [relation.to_dict() for relation in self.relations],
            "asserted": [relation.to_dict() for relation in self.asserted],
            "uncertain": [relation.to_dict() for relation in self.uncertain],
            "audit_trace": [dict(entry) for entry in self.audit_trace],
            "advisory": RELATION_CALIBRATION_ADVISORY,
        }


class RelationCalibrationConsistencyError(RuntimeError):
    """Raised when retained calibrated relation accuracy misses its floor."""


def coerce_relation_calibration_records(
    records: Iterable[RelationCalibrationRecord | Mapping[str, Any]],
) -> tuple[RelationCalibrationRecord, ...]:
    """Normalize relation calibration mappings into validated records."""

    return tuple(
        record
        if isinstance(record, RelationCalibrationRecord)
        else RelationCalibrationRecord.from_mapping(record)
        for record in records
    )


def fit_relation_calibrator(
    records: Iterable[RelationCalibrationRecord | Mapping[str, Any]],
    *,
    min_isotonic_samples: int = DEFAULT_MIN_ISOTONIC_SAMPLES,
    n_bins: int = DEFAULT_RELATION_RELIABILITY_BINS,
) -> RelationCalibrator:
    """Fit isotonic curves per type and pooled temperature for sparse types."""

    if min_isotonic_samples < 2:
        raise ValueError("min_isotonic_samples must be at least 2")
    normalized = coerce_relation_calibration_records(records)
    if not normalized:
        raise ValueError("relation calibration requires at least one record")

    pooled_samples = tuple(_as_calibration_sample(record) for record in normalized)
    pooled_temperature = fit_temperature_scaling(
        pooled_samples,
        n_bins=n_bins,
        default_model_id="relation-calibration",
    )
    fallback = RelationTypeCalibrator(
        relation_type="*",
        method="pooled_temperature",
        sample_count=len(normalized),
        positive_count=sum(1 for record in normalized if record.correct),
        total_weight=sum(record.weight for record in normalized),
        temperature=pooled_temperature.temperature,
    )

    grouped: dict[str, list[RelationCalibrationRecord]] = defaultdict(list)
    for record in normalized:
        grouped[record.relation_type].append(record)

    groups: dict[str, RelationTypeCalibrator] = {}
    for relation_type, group_records in sorted(grouped.items()):
        has_both_outcomes = any(record.correct for record in group_records) and any(
            not record.correct for record in group_records
        )
        if len(group_records) >= min_isotonic_samples and has_both_outcomes:
            isotonic = fit_isotonic_probability_calibrator(
                tuple(_as_calibration_sample(record) for record in group_records),
                default_model_id="relation-calibration",
            )
            groups[relation_type] = RelationTypeCalibrator(
                relation_type=relation_type,
                method="isotonic",
                sample_count=len(group_records),
                positive_count=sum(1 for record in group_records if record.correct),
                total_weight=sum(record.weight for record in group_records),
                isotonic=isotonic,
            )
        else:
            groups[relation_type] = RelationTypeCalibrator(
                relation_type=relation_type,
                method="pooled_temperature",
                sample_count=len(group_records),
                positive_count=sum(1 for record in group_records if record.correct),
                total_weight=sum(record.weight for record in group_records),
                temperature=pooled_temperature.temperature,
            )
    return RelationCalibrator(
        groups=groups,
        fallback=fallback,
        min_isotonic_samples=min_isotonic_samples,
    )


def calibrate_relation_scores(
    records: Iterable[RelationCalibrationRecord | Mapping[str, Any]],
    calibrator: RelationCalibrator,
) -> tuple[float, ...]:
    """Map raw relation scores to calibrated per-type probabilities."""

    normalized = coerce_relation_calibration_records(records)
    return tuple(
        calibrator.predict(
            relation_type=record.relation_type,
            score=record.score,
        )
        for record in normalized
    )


def select_relation_operating_points(
    records: Iterable[RelationCalibrationRecord | Mapping[str, Any]],
    probabilities: Sequence[float],
    *,
    max_abstention_rate: float = DEFAULT_MAX_RELATION_ABSTENTION_RATE,
    min_retained_accuracy: float = DEFAULT_MIN_RETAINED_RELATION_ACCURACY,
) -> dict[str, RelationOperatingPoint]:
    """Select empirical thresholds under a fixed abstention budget.

    The widest-coverage row meeting the accuracy floor and abstention budget is
    selected. These are held-out empirical operating points, not formal bounds.
    """

    max_abstention_rate = _bounded_probability(
        max_abstention_rate,
        "max_abstention_rate",
    )
    min_retained_accuracy = _bounded_probability(
        min_retained_accuracy,
        "min_retained_accuracy",
    )
    normalized = coerce_relation_calibration_records(records)
    if len(normalized) != len(probabilities):
        raise ValueError("records and probabilities must have the same length")
    bounded_probabilities = tuple(
        _bounded_probability(probability, "probability")
        for probability in probabilities
    )
    if not normalized:
        return {}

    grouped: dict[str, list[tuple[RelationCalibrationRecord, float]]] = defaultdict(
        list
    )
    grouped["*"].extend(zip(normalized, bounded_probabilities))
    for record, probability in zip(normalized, bounded_probabilities):
        grouped[record.relation_type].append((record, probability))

    points: dict[str, RelationOperatingPoint] = {}
    for relation_type, items in sorted(grouped.items()):
        curve = risk_coverage_curve(
            record.to_metric_record(confidence=probability)
            for record, probability in items
        )
        safe = [
            row
            for row in curve
            if float(row["abstention_rate"]) <= max_abstention_rate
            and float(row["accuracy"]) >= min_retained_accuracy
        ]
        candidates = safe or list(curve)
        chosen = max(
            candidates,
            key=lambda row: (
                float(row["coverage"]) if safe else float(row["accuracy"]),
                float(row["accuracy"]) if safe else float(row["coverage"]),
                -float(row["confidence_threshold"]),
            ),
        )
        points[relation_type] = RelationOperatingPoint(
            relation_type=relation_type,
            confidence_threshold=float(chosen["confidence_threshold"]),
            accuracy=float(chosen["accuracy"]),
            empirical_risk=float(chosen["empirical_risk"]),
            coverage=float(chosen["coverage"]),
            abstention_rate=float(chosen["abstention_rate"]),
            retained_count=int(chosen["retained_count"]),
            retained_weight=float(chosen["retained_weight"]),
            total_count=int(chosen["total_count"]),
            passed=bool(safe),
        )
    return points


def relation_calibration_report(
    records: Iterable[RelationCalibrationRecord | Mapping[str, Any]],
    *,
    calibrator: RelationCalibrator | None = None,
    n_bins: int = DEFAULT_RELATION_RELIABILITY_BINS,
    min_isotonic_samples: int = DEFAULT_MIN_ISOTONIC_SAMPLES,
    max_abstention_rate: float = DEFAULT_MAX_RELATION_ABSTENTION_RATE,
    min_retained_accuracy: float = DEFAULT_MIN_RETAINED_RELATION_ACCURACY,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a versioned per-type calibration and selective-risk report."""

    normalized = coerce_relation_calibration_records(records)
    if not normalized:
        raise ValueError("relation calibration report requires at least one record")
    fitted = calibrator or fit_relation_calibrator(
        normalized,
        min_isotonic_samples=min_isotonic_samples,
        n_bins=n_bins,
    )
    probabilities = calibrate_relation_scores(normalized, fitted)
    raw_metric_records = [record.to_metric_record() for record in normalized]
    calibrated_metric_records = [
        record.to_metric_record(confidence=probability)
        for record, probability in zip(normalized, probabilities)
    ]
    raw_reliability = relation_reliability_report(
        raw_metric_records,
        n_bins=n_bins,
    )
    calibrated_reliability = relation_reliability_report(
        calibrated_metric_records,
        n_bins=n_bins,
    )
    operating_points = select_relation_operating_points(
        normalized,
        probabilities,
        max_abstention_rate=max_abstention_rate,
        min_retained_accuracy=min_retained_accuracy,
    )
    gate = _relation_consistency_gate(
        operating_points,
        min_retained_accuracy=min_retained_accuracy,
    )
    return {
        "artifact_type": RELATION_REPORT_ARTIFACT,
        "schema_version": RELATION_REPORT_SCHEMA_VERSION,
        "suite": "relation_calibration",
        "generated_at": generated_at or _utc_now(),
        "advisory": RELATION_CALIBRATION_ADVISORY,
        "calibrator": fitted.to_dict(),
        "raw_reliability": raw_reliability,
        "calibrated_reliability": calibrated_reliability,
        "expected_calibration_error": {
            "raw": raw_reliability["expected_calibration_error"],
            "calibrated": calibrated_reliability["expected_calibration_error"],
        },
        "per_type_reliability": calibrated_reliability["per_type"],
        "selective_prediction": selective_prediction_report(calibrated_metric_records),
        "operating_points": {
            key: point.to_dict() for key, point in sorted(operating_points.items())
        },
        "consistency_gate": gate,
        "threshold_selection": {
            "method": "empirical_fixed_abstention_budget",
            "max_abstention_rate": max_abstention_rate,
            "min_retained_accuracy": min_retained_accuracy,
            "certified": False,
        },
    }


def evaluate_relation_consistency_gate(
    report: Mapping[str, Any],
    *,
    min_retained_accuracy: float = DEFAULT_MIN_RETAINED_RELATION_ACCURACY,
) -> dict[str, Any]:
    """Evaluate the pinned retained-accuracy floor from a relation report."""

    min_retained_accuracy = _bounded_probability(
        min_retained_accuracy,
        "min_retained_accuracy",
    )
    operating_payload = report.get("operating_points")
    if not isinstance(operating_payload, Mapping):
        raise ValueError("relation report is missing operating_points")
    operating_points: dict[str, RelationOperatingPoint] = {}
    for relation_type, value in operating_payload.items():
        if not isinstance(value, Mapping):
            raise ValueError("relation operating points must be mappings")
        operating_points[str(relation_type)] = RelationOperatingPoint(
            relation_type=str(value.get("relation_type", relation_type)),
            confidence_threshold=float(value["confidence_threshold"]),
            accuracy=float(value["accuracy"]),
            empirical_risk=float(value["empirical_risk"]),
            coverage=float(value["coverage"]),
            abstention_rate=float(value["abstention_rate"]),
            retained_count=int(value["retained_count"]),
            retained_weight=float(value["retained_weight"]),
            total_count=int(value["total_count"]),
            passed=bool(value.get("passed", False)),
        )
    return _relation_consistency_gate(
        operating_points,
        min_retained_accuracy=min_retained_accuracy,
    )


def assert_relation_consistency_gate(
    report: Mapping[str, Any],
    *,
    min_retained_accuracy: float = DEFAULT_MIN_RETAINED_RELATION_ACCURACY,
) -> None:
    """Raise when any retained relation slice falls below its pinned floor."""

    gate = evaluate_relation_consistency_gate(
        report,
        min_retained_accuracy=min_retained_accuracy,
    )
    if not gate["passed"]:
        raise RelationCalibrationConsistencyError(
            "retained relation accuracy fell below the pinned floor: "
            f"{gate['failed_relation_types']}"
        )


def apply_relation_abstention(
    relations: Sequence[SpanEdge],
    calibrator: RelationCalibrator,
    operating_points: Mapping[str, Any],
) -> RelationAbstentionResult:
    """Calibrate edges and retain below-threshold relations as uncertain.

    Uncertain relations remain in ``relations`` and ``audit_trace`` but are
    excluded from ``asserted`` and ``asserted_edges``.
    """

    calibrated: list[CalibratedRelation] = []
    for relation in relations:
        raw_score = _bounded_probability(relation.score, "relation score")
        relation_type = _normalize_relation_type(relation.label)
        probability = calibrator.predict(
            relation_type=relation_type,
            score=raw_score,
        )
        threshold = _relation_threshold(relation_type, operating_points)
        status = (
            RELATION_STATUS_ASSERTED
            if probability >= threshold
            else RELATION_STATUS_UNCERTAIN
        )
        method = calibrator.method_for(relation_type)
        calibration_metadata = {
            "schema_version": RELATION_CALIBRATION_SCHEMA_VERSION,
            "raw_score": raw_score,
            "calibrated_probability": probability,
            "confidence_threshold": threshold,
            "status": status,
            "calibration_method": method,
        }
        metadata = dict(relation.metadata)
        metadata["relation_calibration"] = calibration_metadata
        calibrated_edge = SpanEdge(
            head=relation.head,
            tail=relation.tail,
            label=relation.label,
            score=probability,
            metadata=metadata,
        )
        calibrated.append(
            CalibratedRelation(
                relation=calibrated_edge,
                raw_score=raw_score,
                calibrated_probability=probability,
                confidence_threshold=threshold,
                status=status,
                calibration_method=method,
            )
        )

    relations_tuple = tuple(calibrated)
    asserted = tuple(relation for relation in relations_tuple if relation.asserted)
    uncertain = tuple(relation for relation in relations_tuple if not relation.asserted)
    return RelationAbstentionResult(
        relations=relations_tuple,
        asserted=asserted,
        uncertain=uncertain,
        audit_trace=tuple(relation.to_audit_entry() for relation in relations_tuple),
    )


def _as_calibration_sample(record: RelationCalibrationRecord) -> CalibrationSample:
    return CalibrationSample(
        model_id="relation-calibration",
        label=record.relation_type,
        language="*",
        score=record.score,
        target=record.correct,
        weight=record.weight,
    )


def _relation_consistency_gate(
    operating_points: Mapping[str, RelationOperatingPoint],
    *,
    min_retained_accuracy: float,
) -> dict[str, Any]:
    failed = sorted(
        relation_type
        for relation_type, point in operating_points.items()
        if point.accuracy < min_retained_accuracy or not point.passed
    )
    return {
        "passed": not failed,
        "min_retained_accuracy": min_retained_accuracy,
        "failed_relation_types": failed,
        "retained_accuracy": {
            relation_type: point.accuracy
            for relation_type, point in sorted(operating_points.items())
        },
    }


def _relation_threshold(
    relation_type: str,
    operating_points: Mapping[str, Any],
) -> float:
    value = operating_points.get(relation_type)
    if value is None:
        value = operating_points.get("*")
    if isinstance(value, RelationOperatingPoint):
        return value.confidence_threshold
    if isinstance(value, Mapping):
        value = value.get("confidence_threshold", value.get("threshold"))
    if value is None:
        raise ValueError(f"missing relation operating point for {relation_type}")
    return _bounded_probability(value, "confidence_threshold")


def _normalize_relation_type(value: Any) -> str:
    relation_type = str(value or "").strip().lower()
    if not relation_type:
        raise ValueError("relation_type must be non-empty")
    return relation_type


def _bounded_probability(value: Any, name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return probability


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "matched", "correct"}:
            return True
        if normalized in {"0", "false", "no", "n", "incorrect"}:
            return False
    return bool(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_MAX_RELATION_ABSTENTION_RATE",
    "DEFAULT_MIN_ISOTONIC_SAMPLES",
    "DEFAULT_MIN_RETAINED_RELATION_ACCURACY",
    "DEFAULT_RELATION_RELIABILITY_BINS",
    "RELATION_CALIBRATION_ADVISORY",
    "RELATION_CALIBRATION_SCHEMA_VERSION",
    "RELATION_STATUS_ASSERTED",
    "RELATION_STATUS_UNCERTAIN",
    "CalibratedRelation",
    "RelationAbstentionResult",
    "RelationCalibrationConsistencyError",
    "RelationCalibrationRecord",
    "RelationCalibrator",
    "RelationOperatingPoint",
    "RelationTypeCalibrator",
    "apply_relation_abstention",
    "assert_relation_consistency_gate",
    "calibrate_relation_scores",
    "coerce_relation_calibration_records",
    "evaluate_relation_consistency_gate",
    "fit_relation_calibrator",
    "relation_calibration_report",
    "select_relation_operating_points",
]
