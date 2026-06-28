"""Confidence-threshold sweep reports for scored PHI spans."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from openmed.eval.metrics import EvalSpan, compute_exact_span_f1, normalize_eval_spans


@dataclass(frozen=True)
class ThresholdSweepPoint:
    """Precision/recall counts at one confidence threshold."""

    threshold: float
    predicted_count: int
    gold_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable PHI-free point payload."""
        return {
            "threshold": self.threshold,
            "predicted_count": self.predicted_count,
            "gold_count": self.gold_count,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ThresholdSweepReport:
    """Full threshold sweep curve and deterministic operating points."""

    curve_points: tuple[ThresholdSweepPoint, ...]
    precision_floor: float
    recall_floor: float
    recall_maximizing_point: ThresholdSweepPoint | None
    precision_maximizing_point: ThresholdSweepPoint | None

    @property
    def curve(self) -> tuple[ThresholdSweepPoint, ...]:
        """Alias for callers that prefer precision-recall curve terminology."""
        return self.curve_points

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable PHI-free report payload."""
        return {
            "precision_floor": self.precision_floor,
            "recall_floor": self.recall_floor,
            "curve_points": [point.to_dict() for point in self.curve_points],
            "recall_maximizing_point": _point_to_dict(self.recall_maximizing_point),
            "precision_maximizing_point": _point_to_dict(
                self.precision_maximizing_point
            ),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a deterministic Markdown table of the sweep curve."""
        lines = [
            "# Confidence Threshold Sweep",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Precision Floor | {_format_float(self.precision_floor)} |",
            f"| Recall Floor | {_format_float(self.recall_floor)} |",
            "",
            "## Recommended Operating Points",
            "",
            "| Objective | Threshold | Precision | Recall | TP | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|",
            _recommendation_row(
                "Max recall at precision floor",
                self.recall_maximizing_point,
            ),
            _recommendation_row(
                "Max precision at recall floor",
                self.precision_maximizing_point,
            ),
            "",
            "## Curve",
            "",
            "| Threshold | Predicted | Gold | TP | FP | FN | Precision | Recall | F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for point in self.curve_points:
            lines.append(
                "| "
                f"{_format_float(point.threshold)} | "
                f"{point.predicted_count} | "
                f"{point.gold_count} | "
                f"{point.true_positives} | "
                f"{point.false_positives} | "
                f"{point.false_negatives} | "
                f"{_format_float(point.precision)} | "
                f"{_format_float(point.recall)} | "
                f"{_format_float(point.f1)} |"
            )
        return "\n".join(lines) + "\n"


def sweep_confidence_thresholds(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    thresholds: Sequence[int | float] | None = None,
    precision_floor: float = 0.0,
    recall_floor: float = 0.0,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> ThresholdSweepReport:
    """Sweep confidence thresholds and compute exact-span precision/recall.

    ``predicted_spans`` must include a bounded ``confidence`` or ``score`` value.
    The returned report intentionally carries only thresholds, counts, and
    aggregate rates; span text, labels, and offsets are not serialized.
    """

    precision_floor = _bounded_rate(precision_floor, "precision_floor")
    recall_floor = _bounded_rate(recall_floor, "recall_floor")
    raw_gold = list(gold_spans)
    raw_predictions = list(predicted_spans)
    confidences = tuple(_span_confidence(span) for span in raw_predictions)
    threshold_values = _threshold_values(confidences, thresholds=thresholds)
    gold = normalize_eval_spans(
        raw_gold,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predictions = normalize_eval_spans(
        raw_predictions,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )

    curve_points = tuple(
        _point_at_threshold(
            gold,
            predictions,
            confidences,
            threshold=threshold,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        )
        for threshold in threshold_values
    )
    return ThresholdSweepReport(
        curve_points=curve_points,
        precision_floor=precision_floor,
        recall_floor=recall_floor,
        recall_maximizing_point=_recall_maximizing_point(
            curve_points,
            precision_floor=precision_floor,
        ),
        precision_maximizing_point=_precision_maximizing_point(
            curve_points,
            recall_floor=recall_floor,
        ),
    )


def build_threshold_sweep_report(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    **kwargs: Any,
) -> ThresholdSweepReport:
    """Compatibility wrapper for :func:`sweep_confidence_thresholds`."""
    return sweep_confidence_thresholds(gold_spans, predicted_spans, **kwargs)


def threshold_sweep_report(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    **kwargs: Any,
) -> ThresholdSweepReport:
    """Compatibility wrapper for :func:`sweep_confidence_thresholds`."""
    return sweep_confidence_thresholds(gold_spans, predicted_spans, **kwargs)


def _point_at_threshold(
    gold: Sequence[EvalSpan],
    predictions: Sequence[EvalSpan],
    confidences: Sequence[float],
    *,
    threshold: float,
    default_language: str,
    default_device: str,
    source_text: str | None,
) -> ThresholdSweepPoint:
    filtered = [
        span
        for span, confidence in zip(predictions, confidences)
        if confidence >= threshold
    ]
    metrics = compute_exact_span_f1(
        gold,
        filtered,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    return ThresholdSweepPoint(
        threshold=threshold,
        predicted_count=len(filtered),
        gold_count=len(gold),
        true_positives=metrics.true_positives,
        false_positives=metrics.false_positives,
        false_negatives=metrics.false_negatives,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
    )


def _threshold_values(
    confidences: Sequence[float],
    *,
    thresholds: Sequence[int | float] | None,
) -> tuple[float, ...]:
    if thresholds is None:
        values = {0.0, 1.0, *confidences}
    else:
        raw_thresholds = list(thresholds)
        if not raw_thresholds:
            raise ValueError("thresholds must include at least one value")
        values = {
            _bounded_rate(value, f"thresholds[{index}]")
            for index, value in enumerate(raw_thresholds)
        }
    return tuple(sorted(values))


def _span_confidence(span: Any) -> float:
    if isinstance(span, EvalSpan):
        raw = span.metadata.get("confidence")
        if raw is None:
            raw = span.metadata.get("score")
    elif isinstance(span, Mapping):
        raw = span.get("confidence")
        if raw is None:
            raw = span.get("score")
        metadata = span.get("metadata")
        if raw is None and isinstance(metadata, Mapping):
            raw = metadata.get("confidence")
            if raw is None:
                raw = metadata.get("score")
    else:
        raw = getattr(span, "confidence", None)
        if raw is None:
            raw = getattr(span, "score", None)
        metadata = getattr(span, "metadata", None)
        if raw is None and isinstance(metadata, Mapping):
            raw = metadata.get("confidence")
            if raw is None:
                raw = metadata.get("score")

    if raw is None:
        raise ValueError("predicted span requires confidence or score")
    return _bounded_rate(raw, "confidence")


def _recall_maximizing_point(
    points: Sequence[ThresholdSweepPoint],
    *,
    precision_floor: float,
) -> ThresholdSweepPoint | None:
    candidates = [point for point in points if point.precision >= precision_floor]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda point: (
            point.recall,
            point.precision,
            point.f1,
            -point.false_positives,
            -point.threshold,
        ),
    )


def _precision_maximizing_point(
    points: Sequence[ThresholdSweepPoint],
    *,
    recall_floor: float,
) -> ThresholdSweepPoint | None:
    candidates = [point for point in points if point.recall >= recall_floor]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda point: (
            point.precision,
            point.recall,
            point.f1,
            -point.false_negatives,
            point.threshold,
        ),
    )


def _bounded_rate(value: Any, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return result


def _point_to_dict(point: ThresholdSweepPoint | None) -> dict[str, int | float] | None:
    return None if point is None else point.to_dict()


def _recommendation_row(objective: str, point: ThresholdSweepPoint | None) -> str:
    if point is None:
        return f"| {objective} | n/a | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {objective} | "
        f"{_format_float(point.threshold)} | "
        f"{_format_float(point.precision)} | "
        f"{_format_float(point.recall)} | "
        f"{point.true_positives} | "
        f"{point.false_positives} | "
        f"{point.false_negatives} |"
    )


def _format_float(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


__all__ = [
    "ThresholdSweepPoint",
    "ThresholdSweepReport",
    "build_threshold_sweep_report",
    "sweep_confidence_thresholds",
    "threshold_sweep_report",
]
