"""Calibrate clinical grounding scores for human review.

Grounding linkers emit matcher scores, not probabilities. This module fits an
independent isotonic score-to-confidence mapping for each vocabulary and adds
an explicit ``accept``/``uncertain`` band to :class:`GroundedSpan` values.

This is deliberately separate from PII detection calibration. PII calibration
maps detector scores to redaction decisions; grounding calibration maps a
vocabulary-linker score to the probability that the selected code is correct.
Neither calibration path makes an autonomous clinical decision.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .calibration import (
    GroundingCalibrator,
    coerce_grounding_calibration_records,
)
from .calibration import (
    fit_grounding_calibrator as _fit_grounding_calibrator,
)
from .types import GROUNDING_CONFIDENCE_BANDS, GroundedSpan

CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_ACCEPT_THRESHOLD = 0.80
ACCEPT_BAND = "accept"
UNCERTAIN_BAND = "uncertain"
GROUNDING_CALIBRATION_ADVISORY = (
    "Grounding confidence is an assistive code-linking signal. It is distinct "
    "from PII detection calibration and requires human verification."
)


@dataclass(frozen=True)
class CalibratedGroundingScore:
    """One raw grounding score after calibration and band assignment."""

    vocabulary: str
    raw_score: float
    calibrated_confidence: float
    threshold: float
    band: str

    @property
    def accepted(self) -> bool:
        """Return whether the score is in the configured accept band."""

        return self.band == ACCEPT_BAND

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready score record without source text."""

        return {
            "vocabulary": self.vocabulary,
            "raw_score": self.raw_score,
            "calibrated_confidence": self.calibrated_confidence,
            "threshold": self.threshold,
            "band": self.band,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class GroundingConfidenceCalibrator:
    """Per-vocabulary grounding calibrator with configurable review bands.

    ``model`` contains the fitted isotonic curves. ``threshold`` is the default
    accept threshold and ``thresholds`` optionally overrides it per vocabulary.
    Thresholds are applied to calibrated confidence, never to raw matcher
    scores.
    """

    model: GroundingCalibrator
    threshold: float = DEFAULT_ACCEPT_THRESHOLD
    thresholds: Mapping[str, float] = field(default_factory=dict)
    label: str = "*"
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.model, GroundingCalibrator):
            raise TypeError("model must be a GroundingCalibrator")
        default_threshold = _bounded_probability(self.threshold, "threshold")
        normalized_thresholds = {
            _normalize_vocabulary(vocabulary): _bounded_probability(
                value,
                f"threshold for {vocabulary}",
            )
            for vocabulary, value in (self.thresholds or {}).items()
        }
        label = str(self.label or "*").strip() or "*"
        object.__setattr__(self, "threshold", default_threshold)
        object.__setattr__(self, "thresholds", normalized_thresholds)
        object.__setattr__(self, "label", label)

    def predict(
        self,
        vocabulary: str | None = None,
        score: float | None = None,
        *,
        system: str | None = None,
        label: str | None = None,
    ) -> float:
        """Return calibrated confidence for one vocabulary score.

        ``system`` is accepted as an alias for ``vocabulary`` so the method can
        be used directly with :class:`~openmed.clinical.grounding.Candidate`.
        """

        vocabulary = _resolve_vocabulary(vocabulary, system)
        if score is None:
            raise TypeError("score is required")
        return self.model.predict(
            system=vocabulary,
            label=label or self.label,
            score=_bounded_probability(score, "score"),
        )

    def confidence(
        self,
        vocabulary: str | None = None,
        score: float | None = None,
        *,
        system: str | None = None,
        label: str | None = None,
    ) -> float:
        """Alias for :meth:`predict` using the issue's confidence terminology."""

        return self.predict(
            vocabulary,
            score,
            system=system,
            label=label,
        )

    def threshold_for(
        self,
        vocabulary: str | None = None,
        *,
        system: str | None = None,
        threshold: float | None = None,
    ) -> float:
        """Return the explicit, per-vocabulary, or default threshold."""

        if threshold is not None:
            return _bounded_probability(threshold, "threshold")
        resolved = _resolve_vocabulary(vocabulary, system)
        return self.thresholds.get(resolved, self.threshold)

    def band_for(
        self,
        vocabulary: str,
        confidence: float,
        *,
        threshold: float | None = None,
    ) -> str:
        """Classify calibrated confidence as ``accept`` or ``uncertain``."""

        bounded_confidence = _bounded_probability(
            confidence,
            "calibrated_confidence",
        )
        configured_threshold = self.threshold_for(
            vocabulary,
            threshold=threshold,
        )
        return (
            ACCEPT_BAND
            if bounded_confidence >= configured_threshold
            else UNCERTAIN_BAND
        )

    def classify(
        self,
        vocabulary: str,
        score: float,
        *,
        label: str | None = None,
        threshold: float | None = None,
    ) -> CalibratedGroundingScore:
        """Calibrate and classify one raw grounding score."""

        confidence = self.predict(vocabulary, score, label=label)
        configured_threshold = self.threshold_for(
            vocabulary,
            threshold=threshold,
        )
        return CalibratedGroundingScore(
            vocabulary=_normalize_vocabulary(vocabulary),
            raw_score=_bounded_probability(score, "score"),
            calibrated_confidence=confidence,
            threshold=configured_threshold,
            band=self.band_for(
                vocabulary,
                confidence,
                threshold=configured_threshold,
            ),
        )

    def apply(
        self,
        grounded_span: GroundedSpan,
        *,
        label: str | None = None,
        threshold: float | None = None,
    ) -> GroundedSpan:
        """Attach per-candidate confidence metadata to a grounded span.

        The selected candidates remain intact, including uncertain candidates,
        so a review report can show the link that needs human adjudication.
        Existing ``abstained`` state is preserved; this layer only adds a
        confidence band.
        """

        if not isinstance(grounded_span, GroundedSpan):
            raise TypeError("grounded_span must be a GroundedSpan")
        candidates = tuple(grounded_span.candidates)
        if not candidates:
            return replace(
                grounded_span,
                calibrated_score=None,
                calibrated_confidence=None,
                confidence_band=None,
            )

        candidate_scores = []
        resolved_label = label or grounded_span.canonical_label or self.label
        for candidate in candidates:
            calibrated = self.classify(
                candidate.system,
                candidate.score,
                label=resolved_label,
                threshold=threshold,
            )
            candidate_scores.append(
                {
                    "system": candidate.system,
                    "code": candidate.code,
                    "display": candidate.display,
                    **calibrated.to_dict(),
                }
            )

        top = candidate_scores[0]
        calibration = {
            "schema_version": self.schema_version,
            "vocabulary": top["vocabulary"],
            "raw_score": top["raw_score"],
            "calibrated_confidence": top["calibrated_confidence"],
            "threshold": top["threshold"],
            "band": top["band"],
            "candidates": candidate_scores,
            "advisory": GROUNDING_CALIBRATION_ADVISORY,
        }
        provenance = _merge_calibration_mapping(
            grounded_span.provenance,
            calibration,
        )
        metadata = _merge_calibration_mapping(
            grounded_span.metadata,
            calibration,
            key="grounding_confidence_calibration",
        )
        return replace(
            grounded_span,
            calibrated_score=top["calibrated_confidence"],
            calibrated_confidence=top["calibrated_confidence"],
            confidence_band=top["band"],
            provenance=provenance,
            metadata=metadata,
        )

    def apply_many(
        self,
        grounded_spans: Iterable[GroundedSpan],
        *,
        label: str | None = None,
        threshold: float | None = None,
    ) -> tuple[GroundedSpan, ...]:
        """Apply this calibrator deterministically to a sequence of spans."""

        return tuple(
            self.apply(span, label=label, threshold=threshold)
            for span in grounded_spans
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, source-text-free calibrator artifact."""

        return {
            "schema_version": self.schema_version,
            "threshold": self.threshold,
            "thresholds": dict(sorted(self.thresholds.items())),
            "label": self.label,
            "model": self.model.to_dict(),
            "advisory": GROUNDING_CALIBRATION_ADVISORY,
        }


def fit_grounding_confidence_calibrator(
    scores: Sequence[Any],
    gold: Sequence[Any] | None = None,
    *,
    labels: Sequence[str] | None = None,
    systems: Sequence[str] | None = None,
    threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    thresholds: Mapping[str, float] | None = None,
    label: str = "*",
) -> GroundingConfidenceCalibrator:
    """Fit a per-vocabulary isotonic score-to-confidence calibrator.

    ``scores`` and ``gold`` use the same synthetic, offline record shapes as
    :func:`openmed.clinical.grounding.calibration.fit_grounding_calibrator`.
    The fitted model is intentionally vocabulary-specific and never consumes
    PII detector scores or PII labels.
    """

    model = _fit_grounding_calibrator(
        scores,
        gold,
        labels=labels,
        systems=systems,
    )
    return GroundingConfidenceCalibrator(
        model=model,
        threshold=threshold,
        thresholds={} if thresholds is None else thresholds,
        label=label,
    )


def calibrate_grounding_scores(
    scores: Sequence[Any],
    calibrator: GroundingConfidenceCalibrator,
    *,
    labels: Sequence[str] | None = None,
    systems: Sequence[str] | None = None,
) -> tuple[float, ...]:
    """Map labeled score-shaped inputs to bounded calibrated confidences."""

    records = coerce_grounding_calibration_records(
        scores,
        [False for _ in scores],
        labels=labels,
        systems=systems,
    )
    return tuple(
        calibrator.predict(
            vocabulary=record.system,
            score=record.score,
            label=record.label,
        )
        for record in records
    )


def apply_grounding_calibration(
    grounded_span: GroundedSpan,
    calibrator: GroundingConfidenceCalibrator,
    *,
    label: str | None = None,
    threshold: float | None = None,
) -> GroundedSpan:
    """Apply a fitted confidence calibrator to one grounded span."""

    return calibrator.apply(
        grounded_span,
        label=label,
        threshold=threshold,
    )


def calibrate_grounded_span(
    grounded_span: GroundedSpan,
    calibrator: GroundingConfidenceCalibrator,
    *,
    label: str | None = None,
    threshold: float | None = None,
) -> GroundedSpan:
    """Compatibility alias for :func:`apply_grounding_calibration`."""

    return apply_grounding_calibration(
        grounded_span,
        calibrator,
        label=label,
        threshold=threshold,
    )


def fit_calibrator(*args: Any, **kwargs: Any) -> GroundingConfidenceCalibrator:
    """Short alias for :func:`fit_grounding_confidence_calibrator`."""

    return fit_grounding_confidence_calibrator(*args, **kwargs)


# The task's public module is named ``calibrate.py`` while the older grounding
# coverage/report implementation remains in ``calibration.py``. Keep the
# obvious function name available in both paths without duplicating the model.
fit_grounding_calibrator = fit_grounding_confidence_calibrator


def _merge_calibration_mapping(
    value: Any,
    calibration: Mapping[str, Any],
    *,
    key: str = "grounding_calibration",
) -> dict[str, Any]:
    merged = dict(value) if isinstance(value, Mapping) else {}
    existing = merged.get(key)
    nested = dict(existing) if isinstance(existing, Mapping) else {}
    nested.update(calibration)
    merged[key] = nested
    return merged


def _resolve_vocabulary(
    vocabulary: str | None,
    system: str | None,
) -> str:
    if vocabulary is not None and system is not None:
        if _normalize_vocabulary(vocabulary) != _normalize_vocabulary(system):
            raise ValueError("vocabulary and system must identify the same system")
    return _normalize_vocabulary(vocabulary if vocabulary is not None else system)


def _normalize_vocabulary(value: Any) -> str:
    text = str(value or "GROUNDING").strip()
    return text.upper() if text else "GROUNDING"


def _bounded_probability(value: Any, name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


__all__ = [
    "ACCEPT_BAND",
    "CALIBRATION_SCHEMA_VERSION",
    "DEFAULT_ACCEPT_THRESHOLD",
    "GROUNDING_CALIBRATION_ADVISORY",
    "GROUNDING_CONFIDENCE_BANDS",
    "UNCERTAIN_BAND",
    "CalibratedGroundingScore",
    "GroundingConfidenceCalibrator",
    "apply_grounding_calibration",
    "calibrate_grounded_span",
    "calibrate_grounding_scores",
    "fit_calibrator",
    "fit_grounding_calibrator",
    "fit_grounding_confidence_calibrator",
]
