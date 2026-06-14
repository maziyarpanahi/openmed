"""Detector arbitration and score calibration for privacy spans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .labels import (
    ACCOUNT_NUMBER,
    API_KEY,
    CREDIT_CARD,
    CVV,
    DATE,
    DATE_OF_BIRTH,
    EMAIL,
    FIRST_NAME,
    ID_NUM,
    IBAN,
    IP_ADDRESS,
    LAST_NAME,
    LOCATION,
    MIDDLE_NAME,
    OTHER,
    PASSWORD,
    PERSON,
    PHONE,
    PIN,
    SSN,
    STREET_ADDRESS,
    URL,
    USERNAME,
    normalize_label,
)
from .pii_entity_merger import is_more_specific
from .schemas.span import OpenMedSpan


CALIBRATION_VERSION = 1
MODE_BALANCED = "balanced"
MODE_HIGH_RECALL_UNION = "high_recall_union"
DEFAULT_BALANCED_FLOOR = 0.5
DEFAULT_HIGH_RECALL_FLOOR = 0.05

_SPECIFICITY_RANKS: Mapping[str, int] = {
    OTHER: 0,
    PERSON: 1,
    LOCATION: 1,
    DATE: 1,
    ID_NUM: 1,
    FIRST_NAME: 2,
    LAST_NAME: 2,
    MIDDLE_NAME: 2,
    USERNAME: 2,
    STREET_ADDRESS: 2,
    DATE_OF_BIRTH: 2,
    EMAIL: 2,
    PHONE: 2,
    URL: 2,
    SSN: 2,
    ACCOUNT_NUMBER: 2,
    PASSWORD: 2,
    PIN: 2,
    API_KEY: 2,
    CREDIT_CARD: 2,
    CVV: 2,
    IBAN: 2,
    IP_ADDRESS: 2,
}


@dataclass(frozen=True)
class DetectorCalibration:
    """Monotonic score calibration curve for one detector."""

    detector: str
    breakpoints: tuple[tuple[float, float], ...]
    method: str = "isotonic"

    def apply(self, score: float | None) -> float | None:
        if score is None:
            return None
        bounded_score = _clamp(float(score))
        if not self.breakpoints:
            return bounded_score
        if bounded_score <= self.breakpoints[0][0]:
            return self.breakpoints[0][1]
        if bounded_score >= self.breakpoints[-1][0]:
            return self.breakpoints[-1][1]

        previous_raw, previous_value = self.breakpoints[0]
        for raw, value in self.breakpoints[1:]:
            if bounded_score <= raw:
                width = raw - previous_raw
                if width <= 0:
                    return value
                fraction = (bounded_score - previous_raw) / width
                return _clamp(previous_value + fraction * (value - previous_value))
            previous_raw, previous_value = raw, value
        return self.breakpoints[-1][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "method": self.method,
            "breakpoints": [
                {"raw_score": raw, "calibrated_score": calibrated}
                for raw, calibrated in self.breakpoints
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DetectorCalibration":
        return cls(
            detector=str(payload["detector"]),
            method=str(payload.get("method") or "isotonic"),
            breakpoints=tuple(
                (
                    float(point["raw_score"]),
                    float(point["calibrated_score"]),
                )
                for point in payload.get("breakpoints", ())
            ),
        )


@dataclass(frozen=True)
class ScoreCalibrator:
    """Versioned per-detector score calibration table."""

    detectors: Mapping[str, DetectorCalibration]
    version: int = CALIBRATION_VERSION

    @classmethod
    def fit(cls, samples: Sequence[Any]) -> "ScoreCalibrator":
        grouped: dict[str, list[tuple[float, float]]] = {}
        for sample in samples:
            detector = _sample_value(sample, "detector")
            score = _sample_value(sample, "score", "raw_score")
            target = _sample_value(sample, "target", "label", "outcome", "is_true")
            if detector is None or score is None or target is None:
                raise ValueError("calibration samples require detector, score, and target")
            grouped.setdefault(str(detector), []).append(
                (_clamp(float(score)), 1.0 if bool(target) else 0.0)
            )

        return cls(
            detectors={
                detector: DetectorCalibration(
                    detector=detector,
                    breakpoints=_fit_isotonic(points),
                )
                for detector, points in grouped.items()
            }
        )

    def apply_score(self, detector: str | None, score: float | None) -> float | None:
        if score is None:
            return None
        if detector is None:
            return _clamp(float(score))
        calibration = self.detectors.get(detector)
        if calibration is None:
            return _clamp(float(score))
        return calibration.apply(score)

    def apply_span(self, span: OpenMedSpan) -> OpenMedSpan:
        calibrated = self.apply_score(span.detector, span.score)
        if calibrated == span.score:
            return span
        metadata = dict(span.metadata)
        metadata["score_calibration"] = {
            "version": self.version,
            "detector": span.detector,
            "raw_score": span.score,
            "calibrated_score": calibrated,
        }
        return replace(span, score=calibrated, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "detectors": {
                detector: calibration.to_dict()
                for detector, calibration in sorted(self.detectors.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScoreCalibrator":
        return cls(
            version=int(payload.get("version", CALIBRATION_VERSION)),
            detectors={
                detector: DetectorCalibration.from_dict(calibration)
                for detector, calibration in (payload.get("detectors") or {}).items()
            },
        )


def fit(samples: Sequence[Any]) -> ScoreCalibrator:
    """Fit a versioned monotonic score calibrator from fixture/eval samples."""

    return ScoreCalibrator.fit(samples)


def apply_calibration(
    spans: Sequence[OpenMedSpan],
    calibrator: ScoreCalibrator | None,
) -> tuple[OpenMedSpan, ...]:
    if calibrator is None:
        return tuple(spans)
    return tuple(calibrator.apply_span(span) for span in spans)


def arbitration_mode(*, strict_no_leak: bool = False, mode: str | None = None) -> str:
    if mode is not None:
        _validate_mode(mode)
        return mode
    return MODE_HIGH_RECALL_UNION if strict_no_leak else MODE_BALANCED


def arbitrate(
    spans: Sequence[OpenMedSpan],
    *,
    mode: str | None = None,
    strict_no_leak: bool = False,
    label_floors: Mapping[str, float] | None = None,
    high_recall_label_floors: Mapping[str, float] | None = None,
    calibrator: ScoreCalibrator | None = None,
) -> tuple[OpenMedSpan, ...]:
    """Resolve duplicate and overlapping detector spans."""

    selected_mode = arbitration_mode(strict_no_leak=strict_no_leak, mode=mode)
    calibrated = apply_calibration(spans, calibrator)
    if selected_mode == MODE_HIGH_RECALL_UNION:
        union_candidates = [
            span
            for span in calibrated
            if _meets_floor(
                span,
                high_recall_label_floors,
                default_floor=DEFAULT_HIGH_RECALL_FLOOR,
            )
        ]
        return _sort_spans(_collapse_exact_duplicates(union_candidates))

    candidates = [
        span
        for span in calibrated
        if _meets_floor(span, label_floors, default_floor=DEFAULT_BALANCED_FLOOR)
    ]
    deduped = _collapse_exact_duplicates(candidates)
    winners = [_choose_winner(cluster) for cluster in _overlap_clusters(deduped)]
    return _sort_spans(winners)


def specificity_rank(label: str) -> int:
    canonical = normalize_label(label)
    return _SPECIFICITY_RANKS.get(canonical, 1)


def is_more_specific_label(left: str, right: str) -> bool:
    left_canonical = normalize_label(left)
    right_canonical = normalize_label(right)
    left_rank = specificity_rank(left_canonical)
    right_rank = specificity_rank(right_canonical)
    if left_rank != right_rank:
        return left_rank > right_rank
    return is_more_specific(left_canonical.lower(), right_canonical.lower())


def _validate_mode(mode: str) -> None:
    if mode not in {MODE_BALANCED, MODE_HIGH_RECALL_UNION}:
        raise ValueError(
            f"mode must be {MODE_BALANCED!r} or {MODE_HIGH_RECALL_UNION!r}"
        )


def _sample_value(sample: Any, *names: str) -> Any:
    if isinstance(sample, Mapping):
        for name in names:
            if name in sample:
                return sample[name]
        return None
    for name in names:
        if hasattr(sample, name):
            return getattr(sample, name)
    if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
        index_by_name = {
            "detector": 0,
            "score": 1,
            "raw_score": 1,
            "target": 2,
            "label": 2,
            "outcome": 2,
            "is_true": 2,
        }
        for name in names:
            index = index_by_name.get(name)
            if index is not None and len(sample) > index:
                return sample[index]
    return None


def _fit_isotonic(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if not points:
        return ()

    blocks: list[dict[str, float]] = []
    for score, target in sorted(points):
        blocks.append({"min": score, "max": score, "sum": target, "count": 1.0})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left["sum"] / left["count"] <= right["sum"] / right["count"]:
                break
            merged = {
                "min": left["min"],
                "max": right["max"],
                "sum": left["sum"] + right["sum"],
                "count": left["count"] + right["count"],
            }
            blocks[-2:] = [merged]

    return tuple(
        (block["max"], _clamp(block["sum"] / block["count"]))
        for block in blocks
    )


def _collapse_exact_duplicates(
    spans: Sequence[OpenMedSpan],
) -> tuple[OpenMedSpan, ...]:
    grouped: dict[tuple[str, int, int, str], list[OpenMedSpan]] = {}
    for span in spans:
        grouped.setdefault(
            (span.doc_id, span.start, span.end, span.canonical_label),
            [],
        ).append(span)
    return tuple(_choose_winner(group) for group in grouped.values())


def _overlap_clusters(spans: Sequence[OpenMedSpan]) -> list[tuple[OpenMedSpan, ...]]:
    sorted_spans = _sort_spans(spans)
    clusters: list[list[OpenMedSpan]] = []
    current: list[OpenMedSpan] = []
    current_end: int | None = None

    for span in sorted_spans:
        if not current or current_end is None or span.start >= current_end:
            if current:
                clusters.append(current)
            current = [span]
            current_end = span.end
            continue
        current.append(span)
        current_end = max(current_end, span.end)

    if current:
        clusters.append(current)
    return [tuple(cluster) for cluster in clusters]


def _choose_winner(spans: Sequence[OpenMedSpan]) -> OpenMedSpan:
    if not spans:
        raise ValueError("cannot choose a winner from no spans")
    return max(spans, key=_winner_key)


def _winner_key(span: OpenMedSpan) -> tuple[int, int, int, float, int, str]:
    score = float(span.score or 0.0)
    return (
        1 if _is_rules_span(span) else 0,
        specificity_rank(span.canonical_label),
        span.end - span.start,
        score,
        -span.start,
        span.detector or "",
    )


def _is_rules_span(span: OpenMedSpan) -> bool:
    return bool(span.detector and span.detector.startswith("rules:"))


def _meets_floor(
    span: OpenMedSpan,
    label_floors: Mapping[str, float] | None,
    *,
    default_floor: float,
) -> bool:
    score = float(span.score or 0.0)
    floor = default_floor
    if label_floors:
        floor = float(
            label_floors.get(
                span.canonical_label,
                label_floors.get(span.entity_type, default_floor),
            )
        )
    return score >= floor


def _sort_spans(spans: Sequence[OpenMedSpan]) -> tuple[OpenMedSpan, ...]:
    return tuple(
        sorted(
            spans,
            key=lambda span: (
                span.doc_id,
                span.start,
                span.end,
                span.canonical_label,
                span.detector or "",
            ),
        )
    )


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


__all__ = [
    "CALIBRATION_VERSION",
    "DEFAULT_BALANCED_FLOOR",
    "DEFAULT_HIGH_RECALL_FLOOR",
    "DetectorCalibration",
    "MODE_BALANCED",
    "MODE_HIGH_RECALL_UNION",
    "ScoreCalibrator",
    "apply_calibration",
    "arbitrate",
    "arbitration_mode",
    "fit",
    "is_more_specific_label",
    "specificity_rank",
]
