"""Per-label reliability and Expected Calibration Error (ECE).

Breaks calibration out per entity label so over- or under-confident labels are
visible and can drive per-label threshold tuning. Measurement-only: it does not
refit thresholds (OM-033) or duplicate the model-level reliability already in
the calibration harness (OM-264). Reuses :class:`CalibrationSample` as the input
type so scored predictions parse consistently.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from openmed.eval.calibrate import CalibrationSample

__all__ = ["per_label_calibration_report"]

_DEFAULT_NUM_BINS = 10


def per_label_calibration_report(
    samples: Iterable[CalibrationSample | Mapping[str, Any]],
    *,
    num_bins: int = _DEFAULT_NUM_BINS,
    ece_budget: float | None = None,
) -> dict[str, Any]:
    """Compute per-label reliability curves, ECE, and a budget flag.

    Args:
        samples: Scored predictions as :class:`CalibrationSample` instances or
            mappings (parsed via ``CalibrationSample.from_mapping``); each
            carries a canonical label, a confidence ``score``, a boolean
            ``target``, and an optional ``weight``.
        num_bins: Number of equal-width confidence bins in ``[0, 1]``.
        ece_budget: Optional Expected-Calibration-Error budget; labels whose ECE
            exceeds it are flagged.

    Returns:
        A deterministic, JSON-serializable mapping with per-label reliability
        curves, ECE and over-budget flags, plus the sorted flagged-label list.
        Contains only labels and numbers (no raw text).
    """
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")

    by_label: dict[str, list[CalibrationSample]] = defaultdict(list)
    for item in samples:
        sample = (
            item
            if isinstance(item, CalibrationSample)
            else (CalibrationSample.from_mapping(item, default_model_id="model"))
        )
        by_label[sample.label].append(sample)

    labels: dict[str, Any] = {}
    flagged: list[str] = []
    for label in sorted(by_label):
        report = _label_report(by_label[label], num_bins)
        over_budget = ece_budget is not None and report["ece"] > ece_budget
        report["over_budget"] = over_budget
        labels[label] = report
        if over_budget:
            flagged.append(label)

    return {
        "num_bins": num_bins,
        "ece_budget": ece_budget,
        "labels": labels,
        "flagged_labels": flagged,
    }


def _bin_index(score: float, num_bins: int) -> int:
    return min(max(int(score * num_bins), 0), num_bins - 1)


def _label_report(samples: list[CalibrationSample], num_bins: int) -> dict[str, Any]:
    weight = [0.0] * num_bins
    weighted_score = [0.0] * num_bins
    weighted_correct = [0.0] * num_bins
    count = [0] * num_bins
    total_weight = 0.0

    for sample in samples:
        index = _bin_index(sample.score, num_bins)
        weight[index] += sample.weight
        weighted_score[index] += sample.weight * sample.score
        weighted_correct[index] += sample.weight * (1.0 if sample.target else 0.0)
        count[index] += 1
        total_weight += sample.weight

    ece = 0.0
    reliability: list[dict[str, Any]] = []
    for index in range(num_bins):
        if weight[index] > 0.0:
            mean_confidence = weighted_score[index] / weight[index]
            accuracy = weighted_correct[index] / weight[index]
            if total_weight > 0.0:
                ece += (weight[index] / total_weight) * abs(accuracy - mean_confidence)
        else:
            mean_confidence = 0.0
            accuracy = 0.0
        reliability.append(
            {
                "bin_lower": index / num_bins,
                "bin_upper": (index + 1) / num_bins,
                "count": count[index],
                "weight": weight[index],
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    return {
        "sample_count": len(samples),
        "total_weight": total_weight,
        "ece": ece,
        "reliability": reliability,
    }
