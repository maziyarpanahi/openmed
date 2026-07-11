"""Grounding score calibration, coverage gates, and abstention helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from openmed.eval.metrics import expected_calibration_error, reliability_bins

SCHEMA_VERSION = 1
GROUNDING_CALIBRATION_ARTIFACT = "openmed.grounding.calibration.report"
DEFAULT_GROUNDING_LABEL = "*"
DEFAULT_MIN_GROUNDING_ACCURACY = 0.85
DEFAULT_MIN_GROUNDING_COVERAGE = 0.70
DEFAULT_RELIABILITY_BINS = 10


@dataclass(frozen=True)
class GroundingCalibrationRecord:
    """One labeled grounding candidate used to fit or evaluate calibration."""

    system: str
    label: str
    score: float
    correct: bool
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready record."""

        return {
            "system": self.system,
            "label": self.label,
            "score": self.score,
            "correct": self.correct,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GroundingCalibrationGroup:
    """Piecewise-constant isotonic calibrator for one vocabulary/label pair."""

    system: str
    label: str
    knots: tuple[tuple[float, float], ...]
    sample_count: int
    positive_count: int
    total_weight: float

    def predict(self, score: float) -> float:
        """Return the calibrated probability for *score*."""

        value = _bounded_probability(score, "score")
        if not self.knots:
            return 0.0
        for max_score, probability in self.knots:
            if value <= max_score:
                return probability
        return self.knots[-1][1]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of the fitted group."""

        return {
            "system": self.system,
            "label": self.label,
            "knots": [
                {"max_score": score, "probability": probability}
                for score, probability in self.knots
            ],
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "total_weight": self.total_weight,
        }


@dataclass(frozen=True)
class GroundingCalibrator:
    """Per-``(system, label)`` grounding score calibrator."""

    groups: Mapping[tuple[str, str], GroundingCalibrationGroup]
    fallback: GroundingCalibrationGroup

    def predict(self, *, system: str, label: str, score: float) -> float:
        """Predict a calibrated probability for one grounding score."""

        key = (_normalize_system(system), _normalize_label(label))
        group = self.groups.get(key)
        if group is None:
            group = self.groups.get((key[0], DEFAULT_GROUNDING_LABEL))
        if group is None:
            group = self.fallback
        return group.predict(score)

    def predict_record(self, record: GroundingCalibrationRecord) -> float:
        """Predict a calibrated probability for *record*."""

        return self.predict(
            system=record.system,
            label=record.label,
            score=record.score,
        )

    def predict_many(
        self,
        scores: Sequence[Any],
        *,
        labels: Sequence[str] | None = None,
        systems: Sequence[str] | None = None,
    ) -> tuple[float, ...]:
        """Predict calibrated probabilities for records or raw scores."""

        records = coerce_grounding_calibration_records(
            scores,
            [False for _ in scores],
            labels=labels,
            systems=systems,
        )
        return tuple(self.predict_record(record) for record in records)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready fitted calibrator payload."""

        return {
            "groups": [
                group.to_dict()
                for _, group in sorted(self.groups.items(), key=lambda item: item[0])
            ],
            "fallback": self.fallback.to_dict(),
        }


@dataclass(frozen=True)
class GroundingOperatingPoint:
    """Per-system abstention threshold selected from a coverage curve."""

    system: str
    threshold: float
    accuracy: float
    coverage: float
    accepted_count: int
    total_count: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready operating point."""

        return {
            "system": self.system,
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "accepted_count": self.accepted_count,
            "total_count": self.total_count,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class GroundingCalibrationResult:
    """Calibrated probabilities plus reliability data for input records."""

    probabilities: tuple[float, ...]
    records: tuple[GroundingCalibrationRecord, ...]
    calibrator: GroundingCalibrator
    reliability: tuple[dict[str, Any], ...]
    expected_calibration_error: float
    n_bins: int

    def __iter__(self) -> Iterable[float]:
        return iter(self.probabilities)

    def __len__(self) -> int:
        return len(self.probabilities)

    def __getitem__(self, index: int) -> float:
        return self.probabilities[index]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready calibration result summary."""

        return {
            "probabilities": list(self.probabilities),
            "expected_calibration_error": self.expected_calibration_error,
            "n_bins": self.n_bins,
            "reliability": [dict(row) for row in self.reliability],
            "calibrator": self.calibrator.to_dict(),
        }


def calibrate_grounding(
    scores: Sequence[Any],
    gold: Sequence[Any] | None = None,
    *,
    labels: Sequence[str] | None = None,
    systems: Sequence[str] | None = None,
    n_bins: int = DEFAULT_RELIABILITY_BINS,
) -> GroundingCalibrationResult:
    """Fit per-vocabulary grounding calibration and return probabilities.

    ``scores`` may be raw floats, mappings with ``score``/``system``/``label``
    keys, or :class:`GroundingCalibrationRecord` instances. ``gold`` may be a
    boolean sequence, mappings with correctness keys, or omitted when
    ``scores`` already contain correctness labels.
    """

    records = coerce_grounding_calibration_records(
        scores,
        gold,
        labels=labels,
        systems=systems,
    )
    calibrator = fit_grounding_calibrator(records)
    probabilities = tuple(calibrator.predict_record(record) for record in records)
    reliability = tuple(
        reliability_bins(
            _probability_records(probabilities, records),
            n_bins=n_bins,
        )
    )
    return GroundingCalibrationResult(
        probabilities=probabilities,
        records=records,
        calibrator=calibrator,
        reliability=reliability,
        expected_calibration_error=expected_calibration_error(reliability),
        n_bins=n_bins,
    )


def fit_grounding_calibrator(
    scores: Sequence[Any],
    gold: Sequence[Any] | None = None,
    *,
    labels: Sequence[str] | None = None,
    systems: Sequence[str] | None = None,
) -> GroundingCalibrator:
    """Fit a weighted isotonic calibrator for each ``(system, label)`` group."""

    records = coerce_grounding_calibration_records(
        scores,
        gold,
        labels=labels,
        systems=systems,
    )
    if not records:
        raise ValueError("grounding calibration requires at least one record")

    grouped: dict[tuple[str, str], list[GroundingCalibrationRecord]] = defaultdict(list)
    by_system: dict[str, list[GroundingCalibrationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.system, record.label)].append(record)
        by_system[record.system].append(record)

    groups = {
        key: _fit_isotonic_group(system=key[0], label=key[1], records=value)
        for key, value in sorted(grouped.items())
    }
    for system, system_records in sorted(by_system.items()):
        key = (system, DEFAULT_GROUNDING_LABEL)
        groups.setdefault(
            key,
            _fit_isotonic_group(
                system=system,
                label=DEFAULT_GROUNDING_LABEL,
                records=system_records,
            ),
        )
    fallback = _fit_isotonic_group(
        system="*",
        label=DEFAULT_GROUNDING_LABEL,
        records=records,
    )
    return GroundingCalibrator(groups=groups, fallback=fallback)


def coerce_grounding_calibration_records(
    scores: Sequence[Any],
    gold: Sequence[Any] | None = None,
    *,
    labels: Sequence[str] | None = None,
    systems: Sequence[str] | None = None,
) -> tuple[GroundingCalibrationRecord, ...]:
    """Normalize grounding score/gold inputs into calibration records."""

    score_items = list(scores)
    gold_items = [None for _ in score_items] if gold is None else list(gold)
    if len(score_items) != len(gold_items):
        raise ValueError("scores and gold must have the same length")
    if labels is not None and len(labels) != len(score_items):
        raise ValueError("labels must have the same length as scores")
    if systems is not None and len(systems) != len(score_items):
        raise ValueError("systems must have the same length as scores")

    records: list[GroundingCalibrationRecord] = []
    for index, (score_item, gold_item) in enumerate(zip(score_items, gold_items)):
        score_map = score_item if isinstance(score_item, Mapping) else {}
        gold_map = gold_item if isinstance(gold_item, Mapping) else {}
        if isinstance(score_item, GroundingCalibrationRecord) and gold is None:
            records.append(score_item)
            continue

        score = _score_from_item(score_item)
        system = (
            systems[index]
            if systems is not None
            else _first_text(
                score_map,
                gold_map,
                keys=("system", "vocabulary", "vocab", "code_system"),
                default="GROUNDING",
            )
        )
        label = (
            labels[index]
            if labels is not None
            else _first_text(
                score_map,
                gold_map,
                keys=("label", "entity_type", "semantic_type", "kind"),
                default=DEFAULT_GROUNDING_LABEL,
            )
        )
        correct = _correct_from_items(score_item, gold_item)
        weight = _weight_from_items(score_item, gold_item)
        records.append(
            GroundingCalibrationRecord(
                system=_normalize_system(system),
                label=_normalize_label(label),
                score=score,
                correct=correct,
                weight=weight,
                metadata=_metadata_from_items(score_item, gold_item),
            )
        )
    return tuple(records)


def coverage_accuracy_curve(
    probabilities: Sequence[float],
    gold: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    """Return coverage-vs-accuracy rows for calibrated grounding scores."""

    if len(probabilities) != len(gold):
        raise ValueError("probabilities and gold must have the same length")
    if not probabilities:
        return ()

    pairs = [
        (_bounded_probability(probability, "probability"), _bool_from_value(item))
        for probability, item in zip(probabilities, gold)
    ]
    thresholds = sorted({probability for probability, _ in pairs}, reverse=True)
    if 0.0 not in thresholds:
        thresholds.append(0.0)

    rows: list[dict[str, Any]] = []
    total = len(pairs)
    for threshold in thresholds:
        accepted = [
            correct for probability, correct in pairs if probability >= threshold
        ]
        accepted_count = len(accepted)
        correct_count = sum(1 for correct in accepted if correct)
        rows.append(
            {
                "threshold": threshold,
                "coverage": accepted_count / total if total else 0.0,
                "accuracy": correct_count / accepted_count if accepted_count else 0.0,
                "accepted_count": accepted_count,
                "correct_count": correct_count,
                "total_count": total,
            }
        )
    return tuple(rows)


def select_grounding_operating_points(
    records: Sequence[GroundingCalibrationRecord],
    probabilities: Sequence[float],
    *,
    min_accuracy: float = DEFAULT_MIN_GROUNDING_ACCURACY,
    min_coverage: float = DEFAULT_MIN_GROUNDING_COVERAGE,
) -> dict[str, GroundingOperatingPoint]:
    """Select a per-system abstention threshold from coverage curves."""

    min_accuracy = _bounded_probability(min_accuracy, "min_accuracy")
    min_coverage = _bounded_probability(min_coverage, "min_coverage")
    if len(records) != len(probabilities):
        raise ValueError("records and probabilities must have the same length")

    by_system: dict[str, list[tuple[GroundingCalibrationRecord, float]]] = defaultdict(
        list
    )
    for record, probability in zip(records, probabilities):
        by_system[record.system].append((record, probability))

    points: dict[str, GroundingOperatingPoint] = {}
    for system, items in sorted(by_system.items()):
        curve = coverage_accuracy_curve(
            [probability for _, probability in items],
            [record.correct for record, _ in items],
        )
        points[system] = _operating_point_from_curve(
            system,
            curve,
            min_accuracy=min_accuracy,
            min_coverage=min_coverage,
        )
    return points


def grounding_calibration_report(
    scores: Sequence[Any],
    gold: Sequence[Any] | None = None,
    *,
    labels: Sequence[str] | None = None,
    systems: Sequence[str] | None = None,
    calibrator: GroundingCalibrator | None = None,
    n_bins: int = DEFAULT_RELIABILITY_BINS,
    min_accuracy: float = DEFAULT_MIN_GROUNDING_ACCURACY,
    min_coverage: float = DEFAULT_MIN_GROUNDING_COVERAGE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build an offline grounding calibration and coverage-gate report."""

    records = coerce_grounding_calibration_records(
        scores,
        gold,
        labels=labels,
        systems=systems,
    )
    if calibrator is None:
        calibrator = fit_grounding_calibrator(records)
    probabilities = tuple(calibrator.predict_record(record) for record in records)
    operating_points = select_grounding_operating_points(
        records,
        probabilities,
        min_accuracy=min_accuracy,
        min_coverage=min_coverage,
    )
    overall_reliability = tuple(
        reliability_bins(
            _probability_records(probabilities, records),
            n_bins=n_bins,
        )
    )
    vocabularies = _vocabulary_reports(
        records,
        probabilities,
        operating_points=operating_points,
        n_bins=n_bins,
    )
    gate = evaluate_grounding_coverage_gate(
        {"vocabularies": vocabularies},
        min_accuracy=min_accuracy,
        min_coverage=min_coverage,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": GROUNDING_CALIBRATION_ARTIFACT,
        "generated_at": generated_at or _utc_now(),
        "minimum_accuracy": min_accuracy,
        "minimum_coverage": min_coverage,
        "expected_calibration_error": expected_calibration_error(overall_reliability),
        "reliability_diagram": [dict(row) for row in overall_reliability],
        "vocabularies": vocabularies,
        "coverage_gate": gate,
        "calibrator": calibrator.to_dict(),
    }


def evaluate_grounding_coverage_gate(
    report: Mapping[str, Any],
    *,
    min_accuracy: float = DEFAULT_MIN_GROUNDING_ACCURACY,
    min_coverage: float = DEFAULT_MIN_GROUNDING_COVERAGE,
) -> dict[str, Any]:
    """Evaluate the release coverage gate for a grounding report."""

    min_accuracy = _bounded_probability(min_accuracy, "min_accuracy")
    min_coverage = _bounded_probability(min_coverage, "min_coverage")
    vocabularies = report.get("vocabularies", {})
    if not isinstance(vocabularies, Mapping) or not vocabularies:
        return {
            "passed": False,
            "minimum_accuracy": min_accuracy,
            "minimum_coverage": min_coverage,
            "vocabularies": {},
            "violations": {
                "*": "grounding coverage report requires vocabulary results"
            },
        }

    results: dict[str, Any] = {}
    violations: dict[str, Any] = {}
    for system, payload in sorted(vocabularies.items()):
        if not isinstance(payload, Mapping):
            violations[str(system)] = "vocabulary result must be a mapping"
            continue
        operating_point = _first_mapping(
            payload.get("operating_point"),
            payload.get("coverage_gate"),
        )
        accuracy = _optional_float(
            _first_value(
                operating_point.get("accuracy"),
                payload.get("accuracy"),
            )
        )
        coverage = _optional_float(
            _first_value(
                operating_point.get("coverage"),
                payload.get("coverage"),
            )
        )
        threshold = _optional_float(operating_point.get("threshold"))
        system_result = {
            "accuracy": accuracy if accuracy is not None else 0.0,
            "coverage": coverage if coverage is not None else 0.0,
            "threshold": threshold if threshold is not None else 0.0,
            "passed": bool(
                accuracy is not None
                and coverage is not None
                and accuracy >= min_accuracy
                and coverage >= min_coverage
            ),
        }
        results[str(system)] = system_result
        if not system_result["passed"]:
            violations[str(system)] = {
                "accuracy": system_result["accuracy"],
                "coverage": system_result["coverage"],
                "minimum_accuracy": min_accuracy,
                "minimum_coverage": min_coverage,
            }

    return {
        "passed": not violations,
        "minimum_accuracy": min_accuracy,
        "minimum_coverage": min_coverage,
        "vocabularies": results,
        "violations": violations,
    }


def apply_grounding_abstention(
    grounded_span: Any,
    calibrator: GroundingCalibrator,
    operating_points: Mapping[str, Any],
    *,
    label: str = DEFAULT_GROUNDING_LABEL,
) -> Any:
    """Return *grounded_span* with calibrated score and abstention metadata.

    Candidates are preserved on the returned span. Exporters decide whether to
    emit codings by checking the span's ``abstained`` flag.
    """

    candidates = tuple(getattr(grounded_span, "candidates", ()) or ())
    if not candidates:
        return replace(
            grounded_span,
            calibrated_score=None,
            abstained=False,
            provenance=_merge_grounding_provenance(
                getattr(grounded_span, "provenance", {}),
                {
                    "abstained": False,
                    "candidate_count": 0,
                    "reason": "no_candidates",
                },
            ),
        )

    top_candidate = candidates[0]
    system = _normalize_system(getattr(top_candidate, "system"))
    probability = calibrator.predict(
        system=system,
        label=label,
        score=float(getattr(top_candidate, "score")),
    )
    threshold = _threshold_for_system(system, operating_points)
    abstained = probability < threshold
    provenance = _merge_grounding_provenance(
        getattr(grounded_span, "provenance", {}),
        {
            "system": system,
            "label": _normalize_label(label),
            "raw_score": float(getattr(top_candidate, "score")),
            "calibrated_score": probability,
            "threshold": threshold,
            "abstained": abstained,
            "candidate_count": len(candidates),
        },
    )
    return replace(
        grounded_span,
        calibrated_score=probability,
        abstained=abstained,
        provenance=provenance,
    )


def _vocabulary_reports(
    records: Sequence[GroundingCalibrationRecord],
    probabilities: Sequence[float],
    *,
    operating_points: Mapping[str, GroundingOperatingPoint],
    n_bins: int,
) -> dict[str, Any]:
    by_system: dict[str, list[tuple[GroundingCalibrationRecord, float]]] = defaultdict(
        list
    )
    for record, probability in zip(records, probabilities):
        by_system[record.system].append((record, probability))

    reports: dict[str, Any] = {}
    for system, items in sorted(by_system.items()):
        system_records = tuple(record for record, _ in items)
        system_probabilities = tuple(probability for _, probability in items)
        reliability = tuple(
            reliability_bins(
                _probability_records(system_probabilities, system_records),
                n_bins=n_bins,
            )
        )
        curve = coverage_accuracy_curve(
            system_probabilities,
            [record.correct for record in system_records],
        )
        reports[system] = {
            "sample_count": len(system_records),
            "expected_calibration_error": expected_calibration_error(reliability),
            "reliability_diagram": [dict(row) for row in reliability],
            "coverage_accuracy_curve": [dict(row) for row in curve],
            "operating_point": operating_points[system].to_dict(),
        }
    return reports


@dataclass
class _Block:
    max_score: float
    weighted_correct: float
    total_weight: float
    sample_count: int

    @property
    def probability(self) -> float:
        if self.total_weight <= 0.0:
            return 0.0
        return self.weighted_correct / self.total_weight

    def merge(self, other: "_Block") -> "_Block":
        return _Block(
            max_score=other.max_score,
            weighted_correct=self.weighted_correct + other.weighted_correct,
            total_weight=self.total_weight + other.total_weight,
            sample_count=self.sample_count + other.sample_count,
        )


def _fit_isotonic_group(
    *,
    system: str,
    label: str,
    records: Sequence[GroundingCalibrationRecord],
) -> GroundingCalibrationGroup:
    buckets: dict[float, list[float]] = {}
    for record in records:
        bucket = buckets.setdefault(record.score, [0.0, 0.0, 0.0])
        bucket[0] += record.weight
        bucket[1] += record.weight if record.correct else 0.0
        bucket[2] += 1.0

    blocks: list[_Block] = []
    for score, (total_weight, weighted_correct, count) in sorted(buckets.items()):
        blocks.append(
            _Block(
                max_score=score,
                weighted_correct=weighted_correct,
                total_weight=total_weight,
                sample_count=int(count),
            )
        )
        while len(blocks) >= 2 and blocks[-2].probability > blocks[-1].probability:
            merged = blocks[-2].merge(blocks[-1])
            blocks[-2:] = [merged]

    knots = tuple(
        (block.max_score, _bounded_probability(block.probability, "probability"))
        for block in blocks
    )
    return GroundingCalibrationGroup(
        system=system,
        label=label,
        knots=knots,
        sample_count=len(records),
        positive_count=sum(1 for record in records if record.correct),
        total_weight=sum(record.weight for record in records),
    )


def _operating_point_from_curve(
    system: str,
    curve: Sequence[Mapping[str, Any]],
    *,
    min_accuracy: float,
    min_coverage: float,
) -> GroundingOperatingPoint:
    eligible = [
        row
        for row in curve
        if float(row.get("accuracy", 0.0)) >= min_accuracy
        and float(row.get("coverage", 0.0)) >= min_coverage
    ]
    passed = bool(eligible)
    candidates = eligible or list(curve)
    if not candidates:
        return GroundingOperatingPoint(
            system=system,
            threshold=1.0,
            accuracy=0.0,
            coverage=0.0,
            accepted_count=0,
            total_count=0,
            passed=False,
        )
    chosen = max(
        candidates,
        key=lambda row: (
            float(row.get("coverage", 0.0)),
            float(row.get("accuracy", 0.0)),
            -float(row.get("threshold", 0.0)),
        ),
    )
    return GroundingOperatingPoint(
        system=system,
        threshold=float(chosen.get("threshold", 0.0)),
        accuracy=float(chosen.get("accuracy", 0.0)),
        coverage=float(chosen.get("coverage", 0.0)),
        accepted_count=int(chosen.get("accepted_count", 0)),
        total_count=int(chosen.get("total_count", 0)),
        passed=passed,
    )


def _probability_records(
    probabilities: Sequence[float],
    records: Sequence[GroundingCalibrationRecord],
) -> list[tuple[float, bool]]:
    return [
        (_bounded_probability(probability, "probability"), record.correct)
        for probability, record in zip(probabilities, records)
    ]


def _score_from_item(item: Any) -> float:
    if isinstance(item, GroundingCalibrationRecord):
        return item.score
    if isinstance(item, Mapping):
        value = _first_value(
            item.get("score"),
            item.get("confidence"),
            item.get("similarity"),
            item.get("probability"),
        )
        if value is None:
            raise ValueError("grounding score mapping requires score")
        return _bounded_probability(value, "score")
    return _bounded_probability(item, "score")


def _correct_from_items(score_item: Any, gold_item: Any) -> bool:
    if gold_item is not None:
        if isinstance(gold_item, Mapping):
            value = _first_value(
                gold_item.get("correct"),
                gold_item.get("is_correct"),
                gold_item.get("matched"),
                gold_item.get("accurate"),
                gold_item.get("target"),
            )
            if value is not None:
                return _bool_from_value(value)
            predicted_code = _first_value(
                getattr(score_item, "code", None),
                score_item.get("code") if isinstance(score_item, Mapping) else None,
                score_item.get("predicted_code")
                if isinstance(score_item, Mapping)
                else None,
                score_item.get("candidate_code")
                if isinstance(score_item, Mapping)
                else None,
            )
            gold_code = _first_value(
                gold_item.get("gold_code"),
                gold_item.get("target_code"),
                gold_item.get("code"),
            )
            if predicted_code is not None and gold_code is not None:
                return str(predicted_code) == str(gold_code)
        return _bool_from_value(gold_item)

    if isinstance(score_item, GroundingCalibrationRecord):
        return score_item.correct
    if isinstance(score_item, Mapping):
        value = _first_value(
            score_item.get("correct"),
            score_item.get("is_correct"),
            score_item.get("matched"),
            score_item.get("accurate"),
            score_item.get("target"),
        )
        if value is not None:
            return _bool_from_value(value)
    raise ValueError("grounding calibration requires correctness labels")


def _weight_from_items(score_item: Any, gold_item: Any) -> float:
    value = None
    if isinstance(score_item, GroundingCalibrationRecord):
        value = score_item.weight
    elif isinstance(score_item, Mapping):
        value = _first_value(score_item.get("weight"), score_item.get("span_weight"))
    if value is None and isinstance(gold_item, Mapping):
        value = _first_value(gold_item.get("weight"), gold_item.get("span_weight"))
    if value is None:
        return 1.0
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("grounding calibration weight must be positive")
    return weight


def _metadata_from_items(score_item: Any, gold_item: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for item in (score_item, gold_item):
        if isinstance(item, GroundingCalibrationRecord):
            metadata.update(dict(item.metadata))
        elif isinstance(item, Mapping) and isinstance(item.get("metadata"), Mapping):
            metadata.update(dict(item["metadata"]))
    return metadata


def _threshold_for_system(system: str, operating_points: Mapping[str, Any]) -> float:
    value = operating_points.get(system)
    if value is None:
        value = operating_points.get(system.lower(), operating_points.get("*"))
    if isinstance(value, GroundingOperatingPoint):
        return value.threshold
    if isinstance(value, Mapping):
        threshold = value.get("threshold")
    else:
        threshold = value
    if threshold is None:
        raise ValueError(f"missing grounding operating point for {system}")
    return _bounded_probability(threshold, "threshold")


def _merge_grounding_provenance(
    provenance: Any,
    update: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(provenance) if isinstance(provenance, Mapping) else {}
    existing = merged.get("grounding_calibration")
    calibration = dict(existing) if isinstance(existing, Mapping) else {}
    calibration.update(update)
    merged["grounding_calibration"] = calibration
    return merged


def _normalize_system(value: Any) -> str:
    text = str(value or "GROUNDING").strip()
    return text.upper() if text else "GROUNDING"


def _normalize_label(value: Any) -> str:
    text = str(value or DEFAULT_GROUNDING_LABEL).strip()
    return text.upper() if text and text != DEFAULT_GROUNDING_LABEL else "*"


def _bounded_probability(value: Any, name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"{name} must be finite")
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "matched"}
    return bool(value)


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_text(
    *mappings: Mapping[str, Any],
    keys: Sequence[str],
    default: str,
) -> str:
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
    return default


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


__all__ = [
    "DEFAULT_GROUNDING_LABEL",
    "DEFAULT_MIN_GROUNDING_ACCURACY",
    "DEFAULT_MIN_GROUNDING_COVERAGE",
    "DEFAULT_RELIABILITY_BINS",
    "GROUNDING_CALIBRATION_ARTIFACT",
    "GroundingCalibrationGroup",
    "GroundingCalibrationRecord",
    "GroundingCalibrationResult",
    "GroundingCalibrator",
    "GroundingOperatingPoint",
    "apply_grounding_abstention",
    "calibrate_grounding",
    "coerce_grounding_calibration_records",
    "coverage_accuracy_curve",
    "evaluate_grounding_coverage_gate",
    "fit_grounding_calibrator",
    "grounding_calibration_report",
    "select_grounding_operating_points",
]
