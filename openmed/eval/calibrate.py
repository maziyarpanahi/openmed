"""Fit and load held-out decision thresholds for PII release artifacts."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import normalize_label
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.core.thresholds import MembershipDefensePolicy
from openmed.eval.metrics import (
    coverage_gaps_by_language,
    expected_calibration_error,
    reliability_bins,
    weighted_coverage,
)

SCHEMA_VERSION = 1
THRESHOLDS_ARTIFACT = "openmed.calibration.thresholds"
REPORT_ARTIFACT = "openmed.calibration.report"
UNDER_SHIFT_REPORT_ARTIFACT = "openmed.calibration.under_shift"
WILDCARD_LANGUAGE = "*"
DEFAULT_CONFORMAL_ALPHA = 0.05
DEFAULT_COVERAGE_TOLERANCE = 0.01
DEFAULT_HISTOGRAM_BINS = 10
DEFAULT_TEMPERATURE_BINS = 10
DEFAULT_CRITICAL_LEAKAGE_LABELS = frozenset(
    {
        "SSN",
        "ID_NUM",
        "API_KEY",
        "ACCOUNT_NUMBER",
        "PASSWORD",
        "PIN",
        "CREDIT_CARD",
        "CVV",
        "IBAN",
        "BIC",
    }
)


@dataclass(frozen=True)
class CalibrationSample:
    """One held-out reliability sample for a model decision threshold."""

    model_id: str
    label: str
    language: str
    score: float
    target: bool
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        default_model_id: str | None = None,
    ) -> "CalibrationSample":
        model_id = (
            data.get("model_id")
            or data.get("model")
            or data.get("model_name")
            or default_model_id
        )
        if not model_id:
            raise ValueError("calibration sample requires model_id")

        raw_label = (
            data.get("canonical_label")
            or data.get("label")
            or data.get("entity_type")
            or data.get("entity")
        )
        if not raw_label:
            raise ValueError("calibration sample requires label")

        language = str(data.get("language") or data.get("lang") or "en").lower()
        label = normalize_label(str(raw_label), language)

        raw_score = data.get("score", data.get("confidence"))
        if raw_score is None:
            raise ValueError("calibration sample requires score")
        score = _bounded_float(raw_score, "score")

        target_value = data.get(
            "target", data.get("is_true", data.get("matched", True))
        )
        target = bool(target_value)

        raw_weight = (
            data.get("weight")
            or data.get("chars")
            or data.get("characters")
            or data.get("char_count")
            or 1.0
        )
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("calibration sample weight must be positive")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}

        return cls(
            model_id=str(model_id),
            label=label,
            language=language,
            score=score,
            target=target,
            weight=weight,
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "language": self.language,
            "score": self.score,
            "target": self.target,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationGroupReport:
    """Fitted threshold and reliability curve for one model/label/language."""

    model_id: str
    label: str
    language: str
    chosen_threshold: float
    target_leakage: float
    resulting_leakage: float
    over_redaction: float
    recall: float
    precision: float
    positive_weight: float
    negative_weight: float
    reliability: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "language": self.language,
            "chosen_threshold": self.chosen_threshold,
            "target_leakage": self.target_leakage,
            "resulting_leakage": self.resulting_leakage,
            "over_redaction": self.over_redaction,
            "recall": self.recall,
            "precision": self.precision,
            "positive_weight": self.positive_weight,
            "negative_weight": self.negative_weight,
            "reliability": [dict(row) for row in self.reliability],
        }


@dataclass(frozen=True)
class TemperatureScalingReport:
    """Global temperature scaling fit and reliability before/after scaling."""

    temperature: float
    pre_scaling_ece: float
    post_scaling_ece: float
    n_bins: int
    sample_count: int
    reliability: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "pre_scaling_ece": self.pre_scaling_ece,
            "post_scaling_ece": self.post_scaling_ece,
            "n_bins": self.n_bins,
            "sample_count": self.sample_count,
            "reliability": [dict(row) for row in self.reliability],
        }


@dataclass(frozen=True)
class DistributionShiftEstimate:
    """Score-histogram shift estimate used to widen conformal bands."""

    method: str
    distance: float
    mean_score_drop: float
    quantile_inflation: float
    histogram: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "distance": self.distance,
            "mean_score_drop": self.mean_score_drop,
            "quantile_inflation": self.quantile_inflation,
            "histogram": [dict(row) for row in self.histogram],
        }


@dataclass(frozen=True)
class ConformalCalibrationGroup:
    """Per-label/per-language conformal band and observed gate coverage."""

    model_id: str
    label: str
    language: str
    alpha: float
    target_coverage: float
    calibration_count: int
    gate_count: int
    base_quantile: float
    shifted_quantile: float
    lower_bound: float
    upper_bound: float
    realized_coverage: float
    coverage_gap: float
    positive_coverage: float
    positive_coverage_gap: float
    total_gate_weight: float
    covered_gate_weight: float
    positive_gate_weight: float
    covered_positive_gate_weight: float
    critical_label: bool
    shift: DistributionShiftEstimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "language": self.language,
            "alpha": self.alpha,
            "target_coverage": self.target_coverage,
            "calibration_count": self.calibration_count,
            "gate_count": self.gate_count,
            "base_quantile": self.base_quantile,
            "shifted_quantile": self.shifted_quantile,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "realized_coverage": self.realized_coverage,
            "coverage_gap": self.coverage_gap,
            "positive_coverage": self.positive_coverage,
            "positive_coverage_gap": self.positive_coverage_gap,
            "total_gate_weight": self.total_gate_weight,
            "covered_gate_weight": self.covered_gate_weight,
            "positive_gate_weight": self.positive_gate_weight,
            "covered_positive_gate_weight": self.covered_positive_gate_weight,
            "critical_label": self.critical_label,
            "shift": self.shift.to_dict(),
        }


@dataclass(frozen=True)
class ConformalCalibrationReport:
    """Calibration-under-shift report used by release gates."""

    model_id: str
    suite: str
    alpha: float
    target_coverage: float
    coverage_tolerance: float
    generated_at: str
    temperature: TemperatureScalingReport
    groups: tuple[ConformalCalibrationGroup, ...]
    language_coverage: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": UNDER_SHIFT_REPORT_ARTIFACT,
            "model_id": self.model_id,
            "suite": self.suite,
            "alpha": self.alpha,
            "target_coverage": self.target_coverage,
            "coverage_tolerance": self.coverage_tolerance,
            "generated_at": self.generated_at,
            "temperature": self.temperature.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
            "language_coverage": {
                key: dict(value)
                for key, value in sorted(self.language_coverage.items())
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Full calibration result for one artifact write."""

    model_id: str
    suite: str
    groups: tuple[CalibrationGroupReport, ...]
    target_leakage: float
    min_recall: float
    generated_at: str
    membership_defense: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    conformal: ConformalCalibrationReport | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": REPORT_ARTIFACT,
            "model_id": self.model_id,
            "suite": self.suite,
            "target_leakage": self.target_leakage,
            "min_recall": self.min_recall,
            "generated_at": self.generated_at,
            "membership_defense": dict(self.membership_defense),
            "objective": "target_leakage_first_over_redaction_second_recall_protected",
            "groups": [group.to_dict() for group in self.groups],
            "metadata": dict(self.metadata),
        }
        if self.conformal is not None:
            payload["calibration_under_shift"] = self.conformal.to_dict()
        return payload


@dataclass(frozen=True)
class CalibrationArtifactPaths:
    """Paths written by a calibration artifact export."""

    artifact_dir: Path
    thresholds_path: Path
    report_path: Path
    under_shift_report_path: Path | None = None


@dataclass(frozen=True)
class CalibrationThresholdSet:
    """Loaded per-model threshold artifact used by inference."""

    schema_version: int
    thresholds: Mapping[tuple[str, str, str], float]
    conformal_bands: Mapping[tuple[str, str, str], Mapping[str, Any]] = field(
        default_factory=dict
    )
    model_id: str | None = None
    suite: str | None = None
    membership_defense: Mapping[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def membership_defense_policy(self) -> MembershipDefensePolicy:
        return MembershipDefensePolicy.from_mapping(self.membership_defense)

    def lookup(
        self,
        label: str,
        language: str,
        *,
        model_id: str | None = None,
        default: float | None = None,
    ) -> float:
        canonical = normalize_label(label, language)
        lang = (language or WILDCARD_LANGUAGE).lower()
        candidate_models = []
        if model_id:
            candidate_models.append(str(model_id))
        if self.model_id and self.model_id not in candidate_models:
            candidate_models.append(self.model_id)
        if len({key[0] for key in self.thresholds}) == 1:
            only_model = next(iter({key[0] for key in self.thresholds}))
            if only_model not in candidate_models:
                candidate_models.append(only_model)

        for candidate_model in candidate_models:
            for candidate_language in (lang, WILDCARD_LANGUAGE):
                key = (candidate_model, canonical, candidate_language)
                if key in self.thresholds:
                    return float(self.thresholds[key])

        if default is not None:
            return float(default)
        raise KeyError(
            f"no threshold for {model_id or self.model_id}:{canonical}:{lang}"
        )

    def conformal_band(
        self,
        label: str,
        language: str,
        *,
        model_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        """Return a loaded conformal band for a label/language if present."""

        canonical = normalize_label(label, language)
        lang = (language or WILDCARD_LANGUAGE).lower()
        candidate_models = []
        if model_id:
            candidate_models.append(str(model_id))
        if self.model_id and self.model_id not in candidate_models:
            candidate_models.append(self.model_id)
        if len({key[0] for key in self.conformal_bands}) == 1:
            only_model = next(iter({key[0] for key in self.conformal_bands}))
            if only_model not in candidate_models:
                candidate_models.append(only_model)

        for candidate_model in candidate_models:
            for candidate_language in (lang, WILDCARD_LANGUAGE):
                key = (candidate_model, canonical, candidate_language)
                if key in self.conformal_bands:
                    return dict(self.conformal_bands[key])
        return None

    def active_for(
        self,
        *,
        model_id: str | None,
        language: str,
    ) -> dict[str, float]:
        lang = (language or WILDCARD_LANGUAGE).lower()
        labels: dict[str, float] = {}
        for artifact_model, label, threshold_language in sorted(self.thresholds):
            if (
                model_id
                and artifact_model != model_id
                and artifact_model != self.model_id
            ):
                continue
            if threshold_language not in {lang, WILDCARD_LANGUAGE}:
                continue
            labels[label] = float(
                self.thresholds[(artifact_model, label, threshold_language)]
            )
        return labels


def fit_calibration_thresholds(
    samples: Sequence[Mapping[str, Any] | CalibrationSample],
    *,
    model_id: str,
    suite: str,
    target_leakage: float = 0.0,
    min_recall: float | None = None,
    conformal_alpha: float | None = None,
    gate_samples: Sequence[Mapping[str, Any] | CalibrationSample] | None = None,
    coverage_tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    membership_defense: Mapping[str, Any] | MembershipDefensePolicy | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationReport:
    """Fit thresholds with leakage target first, over-redaction second."""

    target_leakage = _bounded_float(target_leakage, "target_leakage")
    recall_floor = (
        1.0 - target_leakage
        if min_recall is None
        else _bounded_float(
            min_recall,
            "min_recall",
        )
    )
    normalized = [
        sample
        if isinstance(sample, CalibrationSample)
        else CalibrationSample.from_mapping(sample, default_model_id=model_id)
        for sample in samples
    ]
    if not normalized:
        raise ValueError("calibration requires at least one sample")
    defense_policy = MembershipDefensePolicy.from_mapping(
        membership_defense,
        recall_floor=recall_floor,
    )
    normalized = _apply_membership_defense_to_samples(
        normalized,
        defense_policy,
    )

    grouped: dict[tuple[str, str, str], list[CalibrationSample]] = defaultdict(list)
    for sample in normalized:
        grouped[(sample.model_id, sample.label, sample.language)].append(sample)

    reports = tuple(
        _fit_group(
            group_samples,
            target_leakage=target_leakage,
            min_recall=recall_floor,
        )
        for _, group_samples in sorted(grouped.items())
    )
    timestamp = generated_at or _utc_now()
    conformal_report = None
    if conformal_alpha is not None:
        conformal_report = fit_calibration_under_shift(
            normalized,
            gate_samples=gate_samples,
            model_id=model_id,
            suite=suite,
            alpha=conformal_alpha,
            coverage_tolerance=coverage_tolerance,
            generated_at=timestamp,
            metadata=metadata,
        )

    return CalibrationReport(
        model_id=model_id,
        suite=suite,
        groups=reports,
        target_leakage=target_leakage,
        min_recall=recall_floor,
        generated_at=timestamp,
        membership_defense=defense_policy.to_dict(),
        metadata=dict(metadata or {}),
        conformal=conformal_report,
    )


def fit_calibration_under_shift(
    calibration_samples: Sequence[Mapping[str, Any] | CalibrationSample],
    *,
    gate_samples: Sequence[Mapping[str, Any] | CalibrationSample] | None = None,
    model_id: str,
    suite: str,
    alpha: float = DEFAULT_CONFORMAL_ALPHA,
    coverage_tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    languages: Iterable[str] | None = None,
    critical_labels: Iterable[str] = DEFAULT_CRITICAL_LEAKAGE_LABELS,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    temperature_bins: int = DEFAULT_TEMPERATURE_BINS,
) -> ConformalCalibrationReport:
    """Fit temperature-scaled split-conformal bands under gate-set shift."""

    alpha = _bounded_float(alpha, "alpha")
    coverage_tolerance = _bounded_float(coverage_tolerance, "coverage_tolerance")
    target_coverage = 1.0 - alpha
    if histogram_bins < 1:
        raise ValueError("histogram_bins must be at least 1")
    if temperature_bins < 1:
        raise ValueError("temperature_bins must be at least 1")

    normalized = [
        sample
        if isinstance(sample, CalibrationSample)
        else CalibrationSample.from_mapping(sample, default_model_id=model_id)
        for sample in calibration_samples
    ]
    if not normalized:
        raise ValueError("conformal calibration requires at least one sample")

    gate_normalized = [
        sample
        if isinstance(sample, CalibrationSample)
        else CalibrationSample.from_mapping(sample, default_model_id=model_id)
        for sample in (gate_samples if gate_samples is not None else normalized)
    ]

    temperature = _fit_temperature_scaling(
        normalized,
        n_bins=temperature_bins,
    )
    critical = {normalize_label(label) for label in critical_labels}

    calibration_by_group: dict[tuple[str, str, str], list[CalibrationSample]]
    calibration_by_group = defaultdict(list)
    for sample in normalized:
        calibration_by_group[(sample.model_id, sample.label, sample.language)].append(
            sample
        )

    gate_by_group: dict[tuple[str, str, str], list[CalibrationSample]]
    gate_by_group = defaultdict(list)
    for sample in gate_normalized:
        gate_by_group[(sample.model_id, sample.label, sample.language)].append(sample)

    groups: list[ConformalCalibrationGroup] = []
    coverage_rows: list[dict[str, Any]] = []
    for key in sorted(calibration_by_group):
        group_samples = calibration_by_group[key]
        gate_group = gate_by_group.get(key, ())
        nonconformity = [
            _nonconformity_score(
                _scaled_score(sample.score, temperature.temperature),
                sample.target,
            )
            for sample in group_samples
        ]
        base_quantile = _split_conformal_quantile(nonconformity, alpha=alpha)
        shift = _estimate_distribution_shift(
            [sample.score for sample in group_samples],
            [sample.score for sample in gate_group],
            base_quantile=base_quantile,
            bins=histogram_bins,
        )
        shifted_quantile = min(
            1.0,
            max(base_quantile, base_quantile + shift.quantile_inflation),
        )
        coverage_group_rows: list[dict[str, Any]] = []
        positive_rows: list[dict[str, Any]] = []
        for sample in gate_group:
            scaled = _scaled_score(sample.score, temperature.temperature)
            covered = _nonconformity_score(scaled, sample.target) <= shifted_quantile
            row = {
                "language": sample.language,
                "covered": covered,
                "weight": sample.weight,
            }
            coverage_group_rows.append(row)
            coverage_rows.append(row)
            if sample.target:
                positive_rows.append(row)

        coverage = weighted_coverage(coverage_group_rows)
        positive_coverage = weighted_coverage(positive_rows)
        effective_coverage = (
            float(positive_coverage.rate)
            if float(positive_coverage.denominator) > 0.0
            else float(coverage.rate)
        )
        model_key, label, language = key
        groups.append(
            ConformalCalibrationGroup(
                model_id=model_key,
                label=label,
                language=language,
                alpha=alpha,
                target_coverage=target_coverage,
                calibration_count=len(group_samples),
                gate_count=len(gate_group),
                base_quantile=base_quantile,
                shifted_quantile=shifted_quantile,
                lower_bound=max(0.0, 1.0 - shifted_quantile),
                upper_bound=1.0,
                realized_coverage=float(coverage.rate),
                coverage_gap=max(target_coverage - effective_coverage, 0.0),
                positive_coverage=float(positive_coverage.rate),
                positive_coverage_gap=max(
                    target_coverage - float(positive_coverage.rate),
                    0.0,
                ),
                total_gate_weight=float(coverage.denominator),
                covered_gate_weight=float(coverage.numerator),
                positive_gate_weight=float(positive_coverage.denominator),
                covered_positive_gate_weight=float(positive_coverage.numerator),
                critical_label=label in critical,
                shift=shift,
            )
        )

    languages_to_report = languages if languages is not None else SUPPORTED_LANGUAGES
    language_coverage = coverage_gaps_by_language(
        coverage_rows,
        target_coverage=target_coverage,
        languages=languages_to_report,
    )
    return ConformalCalibrationReport(
        model_id=model_id,
        suite=suite,
        alpha=alpha,
        target_coverage=target_coverage,
        coverage_tolerance=coverage_tolerance,
        generated_at=generated_at or _utc_now(),
        temperature=temperature,
        groups=tuple(groups),
        language_coverage=language_coverage,
        metadata=dict(metadata or {}),
    )


def build_thresholds_payload(report: CalibrationReport) -> dict[str, Any]:
    """Build the JSON payload written as thresholds.json."""

    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    groups: list[dict[str, Any]] = []
    for group in report.groups:
        thresholds.setdefault(group.model_id, {}).setdefault(group.label, {})[
            group.language
        ] = group.chosen_threshold
        groups.append(
            {
                "model_id": group.model_id,
                "label": group.label,
                "language": group.language,
                "threshold": group.chosen_threshold,
                "target_leakage": group.target_leakage,
                "resulting_leakage": group.resulting_leakage,
                "over_redaction": group.over_redaction,
                "recall": group.recall,
                "precision": group.precision,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": THRESHOLDS_ARTIFACT,
        "model_id": report.model_id,
        "suite": report.suite,
        "generated_at": report.generated_at,
        "target_leakage": report.target_leakage,
        "min_recall": report.min_recall,
        "membership_defense": dict(report.membership_defense),
        "thresholds": thresholds,
        "groups": groups,
    }
    if report.conformal is not None:
        conformal_quantiles: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for group in report.conformal.groups:
            conformal_quantiles.setdefault(group.model_id, {}).setdefault(
                group.label,
                {},
            )[group.language] = {
                "alpha": group.alpha,
                "target_coverage": group.target_coverage,
                "base_quantile": group.base_quantile,
                "shifted_quantile": group.shifted_quantile,
                "shift_inflation": group.shift.quantile_inflation,
                "lower_bound": group.lower_bound,
                "upper_bound": group.upper_bound,
                "realized_coverage": group.realized_coverage,
                "coverage_gap": group.coverage_gap,
                "positive_coverage": group.positive_coverage,
                "positive_coverage_gap": group.positive_coverage_gap,
                "critical_label": group.critical_label,
            }
        payload["conformal_quantiles"] = conformal_quantiles
        payload["temperature_scaling"] = report.conformal.temperature.to_dict()
        payload["coverage_tables"] = {
            "language_coverage": {
                key: dict(value)
                for key, value in sorted(report.conformal.language_coverage.items())
            }
        }
    return payload


def write_calibration_artifacts(
    samples: Sequence[Mapping[str, Any] | CalibrationSample],
    *,
    artifact_dir: str | Path,
    model_id: str,
    suite: str,
    target_leakage: float = 0.0,
    min_recall: float | None = None,
    conformal_alpha: float | None = DEFAULT_CONFORMAL_ALPHA,
    gate_samples: Sequence[Mapping[str, Any] | CalibrationSample] | None = None,
    coverage_tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    membership_defense: Mapping[str, Any] | MembershipDefensePolicy | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationArtifactPaths:
    """Fit and write calibration artifacts."""

    report = fit_calibration_thresholds(
        samples,
        model_id=model_id,
        suite=suite,
        target_leakage=target_leakage,
        min_recall=min_recall,
        conformal_alpha=conformal_alpha,
        gate_samples=gate_samples,
        coverage_tolerance=coverage_tolerance,
        membership_defense=membership_defense,
        generated_at=generated_at,
        metadata=metadata,
    )
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = output_dir / "thresholds.json"
    report_path = output_dir / "calibration_report.json"
    under_shift_report_path = (
        output_dir / "calibration_under_shift_report.json"
        if report.conformal is not None
        else None
    )
    thresholds_path.write_text(
        json.dumps(build_thresholds_payload(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if under_shift_report_path is not None and report.conformal is not None:
        under_shift_report_path.write_text(
            json.dumps(report.conformal.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return CalibrationArtifactPaths(
        artifact_dir=output_dir,
        thresholds_path=thresholds_path,
        report_path=report_path,
        under_shift_report_path=under_shift_report_path,
    )


def load_calibration_samples(
    path: str | Path,
    *,
    default_model_id: str | None = None,
) -> list[CalibrationSample]:
    """Load reliability samples from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("samples") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("calibration sample JSON must be a list or contain samples")
    return [
        CalibrationSample.from_mapping(row, default_model_id=default_model_id)
        for row in rows
    ]


def default_suite_calibration_samples(
    model_id: str, suite: str
) -> list[CalibrationSample]:
    """Return a deterministic base-install sample set for the named suite."""

    if suite != "golden":
        raise ValueError(
            f"suite {suite!r} has no built-in calibration samples; provide --input"
        )

    from openmed.eval.golden.loader import load_golden_fixtures

    samples: list[CalibrationSample] = []
    for fixture in load_golden_fixtures():
        for span in fixture.gold_spans:
            metadata = {
                "fixture_id": fixture.fixture_id,
                "source": "golden_builtin",
            }
            samples.append(
                CalibrationSample(
                    model_id=model_id,
                    label=span.label,
                    language=fixture.language,
                    score=0.99,
                    target=True,
                    weight=max(span.length, 1),
                    metadata=metadata,
                )
            )
            samples.append(
                CalibrationSample(
                    model_id=model_id,
                    label=span.label,
                    language=fixture.language,
                    score=0.05,
                    target=False,
                    weight=max(span.length, 1),
                    metadata=metadata,
                )
            )
    return samples


def load_calibration_thresholds(path: str | Path) -> CalibrationThresholdSet:
    """Load a thresholds.json artifact from a file or artifact directory."""

    candidate = Path(path)
    threshold_path = candidate / "thresholds.json" if candidate.is_dir() else candidate
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    return coerce_calibration_thresholds(payload, source_path=str(threshold_path))


def coerce_calibration_thresholds(
    payload: Mapping[str, Any] | CalibrationThresholdSet,
    *,
    source_path: str | None = None,
) -> CalibrationThresholdSet:
    """Validate a thresholds payload and return lookup-ready state."""

    if isinstance(payload, CalibrationThresholdSet):
        return payload
    if int(payload.get("schema_version", 0)) < 1:
        raise ValueError("calibration thresholds require schema_version")
    if payload.get("artifact_type") != THRESHOLDS_ARTIFACT:
        raise ValueError("calibration thresholds artifact_type is invalid")

    raw_thresholds = payload.get("thresholds")
    if not isinstance(raw_thresholds, Mapping) or not raw_thresholds:
        raise ValueError("calibration thresholds require thresholds")

    thresholds: dict[tuple[str, str, str], float] = {}
    for model_key, labels in raw_thresholds.items():
        if not isinstance(labels, Mapping):
            raise ValueError(f"thresholds.{model_key} must be an object")
        for label_key, languages in labels.items():
            canonical = normalize_label(str(label_key))
            if not isinstance(languages, Mapping):
                raise ValueError(
                    f"thresholds.{model_key}.{canonical} must be an object"
                )
            for language_key, value in languages.items():
                language = str(language_key or WILDCARD_LANGUAGE).lower()
                thresholds[(str(model_key), canonical, language)] = _bounded_float(
                    value,
                    f"thresholds.{model_key}.{canonical}.{language}",
                )

    conformal_bands: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    raw_conformal = payload.get("conformal_quantiles") or {}
    if raw_conformal:
        if not isinstance(raw_conformal, Mapping):
            raise ValueError("conformal_quantiles must be an object")
        for model_key, labels in raw_conformal.items():
            if not isinstance(labels, Mapping):
                raise ValueError(f"conformal_quantiles.{model_key} must be an object")
            for label_key, languages in labels.items():
                canonical = normalize_label(str(label_key))
                if not isinstance(languages, Mapping):
                    raise ValueError(
                        f"conformal_quantiles.{model_key}.{canonical} must be an object"
                    )
                for language_key, value in languages.items():
                    if not isinstance(value, Mapping):
                        raise ValueError(
                            f"conformal_quantiles.{model_key}.{canonical}."
                            f"{language_key} must be an object"
                        )
                    language = str(language_key or WILDCARD_LANGUAGE).lower()
                    conformal_bands[(str(model_key), canonical, language)] = dict(value)

    return CalibrationThresholdSet(
        schema_version=int(payload["schema_version"]),
        thresholds=thresholds,
        conformal_bands=conformal_bands,
        model_id=(
            str(payload["model_id"]) if payload.get("model_id") is not None else None
        ),
        suite=str(payload["suite"]) if payload.get("suite") is not None else None,
        membership_defense=dict(payload.get("membership_defense") or {}),
        source_path=source_path,
    )


def artifact_dir_for(model_id: str, suite: str) -> Path:
    """Return the default calibration artifact directory."""

    return Path("artifacts") / "calibration" / _slug(model_id) / _slug(suite)


def _fit_temperature_scaling(
    samples: Sequence[CalibrationSample],
    *,
    n_bins: int,
) -> TemperatureScalingReport:
    pre_bins = reliability_bins(
        ((sample.score, sample.target) for sample in samples),
        n_bins=n_bins,
    )
    pre_ece = expected_calibration_error(pre_bins)
    candidates = _temperature_candidates()
    scored: list[tuple[float, float, float]] = []
    for temperature in candidates:
        scaled_records = (
            (_scaled_score(sample.score, temperature), sample.target)
            for sample in samples
        )
        post_ece = expected_calibration_error(
            reliability_bins(scaled_records, n_bins=n_bins)
        )
        nll = _weighted_log_loss(samples, temperature)
        scored.append((nll, post_ece, temperature))

    feasible = [item for item in scored if item[1] <= pre_ece + 1e-12] or [
        (item[0], item[1], item[2]) for item in scored if item[2] == 1.0
    ]
    _, post_ece, temperature = min(
        feasible, key=lambda item: (item[0], item[1], item[2])
    )
    post_bins = reliability_bins(
        (
            (_scaled_score(sample.score, temperature), sample.target)
            for sample in samples
        ),
        n_bins=n_bins,
    )
    reliability = tuple(
        {
            "bin_index": int(pre["bin_index"]),
            "lower_bound": float(pre["lower_bound"]),
            "upper_bound": float(pre["upper_bound"]),
            "pre_mean_confidence": float(pre["mean_confidence"]),
            "pre_empirical_accuracy": float(pre["empirical_accuracy"]),
            "post_mean_confidence": float(post["mean_confidence"]),
            "post_empirical_accuracy": float(post["empirical_accuracy"]),
            "count": int(post["count"]),
        }
        for pre, post in zip(pre_bins, post_bins, strict=True)
    )
    return TemperatureScalingReport(
        temperature=temperature,
        pre_scaling_ece=pre_ece,
        post_scaling_ece=post_ece,
        n_bins=n_bins,
        sample_count=len(samples),
        reliability=reliability,
    )


def _temperature_candidates() -> tuple[float, ...]:
    values = {1.0}
    values.update(round(0.25 + 0.05 * index, 2) for index in range(116))
    return tuple(sorted(values))


def _weighted_log_loss(
    samples: Sequence[CalibrationSample],
    temperature: float,
) -> float:
    total_weight = 0.0
    loss = 0.0
    for sample in samples:
        probability = _clamp_probability(_scaled_score(sample.score, temperature))
        total_weight += sample.weight
        if sample.target:
            loss -= sample.weight * math.log(probability)
        else:
            loss -= sample.weight * math.log1p(-probability)
    return loss / total_weight if total_weight else 0.0


def _scaled_score(score: float, temperature: float) -> float:
    probability = _clamp_probability(score)
    temp = max(float(temperature), 1e-6)
    logit = math.log(probability / (1.0 - probability))
    scaled = 1.0 / (1.0 + math.exp(-(logit / temp)))
    return min(max(scaled, 0.0), 1.0)


def _clamp_probability(value: float) -> float:
    return min(max(float(value), 1e-12), 1.0 - 1e-12)


def _nonconformity_score(probability: float, target: bool) -> float:
    probability = _clamp_probability(probability)
    return 1.0 - probability if target else probability


def _split_conformal_quantile(
    nonconformity: Sequence[float],
    *,
    alpha: float,
) -> float:
    values = sorted(float(value) for value in nonconformity)
    if not values:
        return 1.0
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    index = min(max(rank, 1), len(values)) - 1
    return min(max(values[index], 0.0), 1.0)


def _estimate_distribution_shift(
    calibration_scores: Sequence[float],
    gate_scores: Sequence[float],
    *,
    base_quantile: float,
    bins: int,
) -> DistributionShiftEstimate:
    calibration_histogram = _score_histogram(calibration_scores, bins=bins)
    gate_histogram = _score_histogram(gate_scores or calibration_scores, bins=bins)
    distance = 0.5 * sum(
        abs(calibration_histogram[index] - gate_histogram[index])
        for index in range(bins)
    )
    calibration_mean = _mean(calibration_scores)
    gate_mean = _mean(gate_scores or calibration_scores)
    mean_score_drop = max(calibration_mean - gate_mean, 0.0)
    inflation = min(max(1.0 - base_quantile, 0.0), distance + mean_score_drop)
    histogram = tuple(
        {
            "bin_index": index,
            "lower_bound": index / bins,
            "upper_bound": (index + 1) / bins,
            "calibration_probability": calibration_histogram[index],
            "gate_probability": gate_histogram[index],
        }
        for index in range(bins)
    )
    return DistributionShiftEstimate(
        method="total_variation_plus_mean_score_drop",
        distance=distance,
        mean_score_drop=mean_score_drop,
        quantile_inflation=inflation,
        histogram=histogram,
    )


def _score_histogram(scores: Sequence[float], *, bins: int) -> tuple[float, ...]:
    counts = [0.0 for _ in range(bins)]
    if not scores:
        return tuple(counts)
    for score in scores:
        bounded = min(max(float(score), 0.0), 1.0)
        index = min(int(bounded * bins), bins - 1)
        counts[index] += 1.0
    total = sum(counts)
    if not total:
        return tuple(counts)
    return tuple(count / total for count in counts)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _apply_membership_defense_to_samples(
    samples: Sequence[CalibrationSample],
    policy: MembershipDefensePolicy,
) -> list[CalibrationSample]:
    if not policy.enabled:
        return list(samples)
    return [
        replace(sample, score=policy.apply_score(sample.score)) for sample in samples
    ]


def _fit_group(
    samples: Sequence[CalibrationSample],
    *,
    target_leakage: float,
    min_recall: float,
) -> CalibrationGroupReport:
    model_id = samples[0].model_id
    label = samples[0].label
    language = samples[0].language
    positive_weight = sum(sample.weight for sample in samples if sample.target)
    negative_weight = sum(sample.weight for sample in samples if not sample.target)
    if positive_weight <= 0.0:
        raise ValueError(
            f"calibration group {model_id}:{label}:{language} has no targets"
        )

    candidates = sorted({0.0, 1.0, *(sample.score for sample in samples)})
    reliability = tuple(
        _metrics_at_threshold(
            samples,
            threshold=threshold,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
        )
        for threshold in candidates
    )

    recall_safe = [row for row in reliability if float(row["recall"]) >= min_recall]
    if not recall_safe:
        raise ValueError(
            f"calibration group {model_id}:{label}:{language} violates recall floor"
        )

    target_safe = [
        row for row in recall_safe if float(row["leakage"]) <= target_leakage
    ]
    candidates_to_rank = target_safe or recall_safe
    if target_safe:
        chosen = min(
            candidates_to_rank,
            key=lambda row: (
                float(row["over_redaction"]),
                -float(row["threshold"]),
            ),
        )
    else:
        chosen = min(
            candidates_to_rank,
            key=lambda row: (
                float(row["leakage"]),
                float(row["over_redaction"]),
                -float(row["threshold"]),
            ),
        )

    return CalibrationGroupReport(
        model_id=model_id,
        label=label,
        language=language,
        chosen_threshold=float(chosen["threshold"]),
        target_leakage=target_leakage,
        resulting_leakage=float(chosen["leakage"]),
        over_redaction=float(chosen["over_redaction"]),
        recall=float(chosen["recall"]),
        precision=float(chosen["precision"]),
        positive_weight=positive_weight,
        negative_weight=negative_weight,
        reliability=reliability,
    )


def _metrics_at_threshold(
    samples: Sequence[CalibrationSample],
    *,
    threshold: float,
    positive_weight: float,
    negative_weight: float,
) -> dict[str, Any]:
    tp = sum(
        sample.weight
        for sample in samples
        if sample.target and sample.score >= threshold
    )
    fn = positive_weight - tp
    fp = sum(
        sample.weight
        for sample in samples
        if not sample.target and sample.score >= threshold
    )

    leakage = fn / positive_weight if positive_weight else 0.0
    recall = tp / positive_weight if positive_weight else 1.0
    over_redaction = fp / negative_weight if negative_weight else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 1.0
    return {
        "threshold": threshold,
        "leakage": leakage,
        "over_redaction": over_redaction,
        "recall": recall,
        "precision": precision,
        "true_positive_weight": tp,
        "false_negative_weight": fn,
        "false_positive_weight": fp,
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
    }


def _bounded_float(value: Any, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "artifact"


__all__ = [
    "CalibrationArtifactPaths",
    "ConformalCalibrationGroup",
    "ConformalCalibrationReport",
    "CalibrationGroupReport",
    "CalibrationReport",
    "CalibrationSample",
    "CalibrationThresholdSet",
    "DEFAULT_CONFORMAL_ALPHA",
    "DistributionShiftEstimate",
    "TemperatureScalingReport",
    "UNDER_SHIFT_REPORT_ARTIFACT",
    "artifact_dir_for",
    "build_thresholds_payload",
    "coerce_calibration_thresholds",
    "default_suite_calibration_samples",
    "fit_calibration_under_shift",
    "fit_calibration_thresholds",
    "load_calibration_samples",
    "load_calibration_thresholds",
    "write_calibration_artifacts",
]
