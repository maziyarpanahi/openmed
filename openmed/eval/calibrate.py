"""Fit and load held-out decision thresholds for PII release artifacts."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import normalize_label
from openmed.core.thresholds import MembershipDefensePolicy

SCHEMA_VERSION = 1
THRESHOLDS_ARTIFACT = "openmed.calibration.thresholds"
REPORT_ARTIFACT = "openmed.calibration.report"
WILDCARD_LANGUAGE = "*"


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
    nonconformity_quantile: float
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
            "nonconformity_quantile": self.nonconformity_quantile,
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

    def to_dict(self) -> dict[str, Any]:
        return {
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


@dataclass(frozen=True)
class CalibrationArtifactPaths:
    """Paths written by a calibration artifact export."""

    artifact_dir: Path
    thresholds_path: Path
    report_path: Path


@dataclass(frozen=True)
class CalibrationThresholdSet:
    """Loaded per-model threshold artifact used by inference."""

    schema_version: int
    thresholds: Mapping[tuple[str, str, str], float]
    nonconformity_quantiles: Mapping[tuple[str, str, str], float] = field(
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

    def lookup_nonconformity_quantile(
        self,
        label: str,
        language: str,
        *,
        model_id: str | None = None,
        default: float | None = None,
    ) -> float:
        """Return the conformal nonconformity quantile for a label/language."""

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

        quantiles = self.nonconformity_quantiles or {
            key: 1.0 - value for key, value in self.thresholds.items()
        }
        for candidate_model in candidate_models:
            for candidate_language in (lang, WILDCARD_LANGUAGE):
                key = (candidate_model, canonical, candidate_language)
                if key in quantiles:
                    return float(quantiles[key])

        if default is not None:
            return float(default)
        return 1.0 - self.lookup(
            label,
            language,
            model_id=model_id,
        )

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


@dataclass(frozen=True)
class SpanNonconformityScore:
    """Confidence-derived conformal score for one span decision."""

    label: str
    language: str
    confidence: float
    confidence_threshold: float
    nonconformity: float
    nonconformity_quantile: float
    accepted_by_quantile: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "language": self.language,
            "confidence": self.confidence,
            "confidence_threshold": self.confidence_threshold,
            "nonconformity": self.nonconformity,
            "nonconformity_quantile": self.nonconformity_quantile,
            "accepted_by_quantile": self.accepted_by_quantile,
        }


@dataclass(frozen=True)
class AbstentionRiskSample:
    """One calibration row used to select a risk-controlled abstention cutoff."""

    label: str
    language: str
    nonconformity: float
    residual_leakage: bool
    critical: bool = True
    weight: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | CalibrationSample,
        *,
        critical_labels: frozenset[str],
    ) -> "AbstentionRiskSample":
        if isinstance(data, CalibrationSample):
            label = data.label
            language = data.language
            confidence = data.score
            residual_leakage = not data.target
            weight = data.weight
        else:
            raw_label = (
                data.get("canonical_label")
                or data.get("label")
                or data.get("entity_type")
                or data.get("entity")
            )
            if not raw_label:
                raise ValueError("abstention risk sample requires label")
            language = str(data.get("language") or data.get("lang") or "en").lower()
            label = normalize_label(str(raw_label), language)

            confidence_value = data.get("score", data.get("confidence"))
            if confidence_value is not None:
                confidence = _bounded_float(confidence_value, "confidence")
                nonconformity = 1.0 - confidence
            else:
                nonconformity = _bounded_float(
                    data.get("nonconformity"),
                    "nonconformity",
                )

            if confidence_value is None:
                confidence = 1.0 - nonconformity

            if data.get("residual_leakage") is not None:
                residual_leakage = bool(data["residual_leakage"])
            elif data.get("leakage") is not None:
                residual_leakage = bool(data["leakage"])
            elif data.get("correct") is not None:
                residual_leakage = not bool(data["correct"])
            elif data.get("target") is not None:
                residual_leakage = not bool(data["target"])
            else:
                residual_leakage = False

            raw_weight = data.get("weight", 1.0)
            weight = float(raw_weight)

        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("abstention risk sample weight must be positive")

        return cls(
            label=label,
            language=language,
            nonconformity=1.0 - confidence,
            residual_leakage=residual_leakage,
            critical=label in critical_labels,
            weight=weight,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "language": self.language,
            "nonconformity": self.nonconformity,
            "residual_leakage": self.residual_leakage,
            "critical": self.critical,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class RiskControlledAbstentionThreshold:
    """Selected nonconformity cutoff with finite-sample risk evidence."""

    nonconformity_threshold: float
    target_risk: float
    confidence_level: float
    empirical_residual_risk: float
    residual_risk_upper_bound: float
    accepted_critical_weight: float
    residual_leakage_weight: float
    abstention_rate: float
    total_weight: float
    method: str = "hoeffding_one_sided"

    @property
    def confidence_threshold(self) -> float:
        """Return the equivalent minimum confidence cutoff."""

        return 1.0 - self.nonconformity_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "nonconformity_threshold": self.nonconformity_threshold,
            "confidence_threshold": self.confidence_threshold,
            "target_risk": self.target_risk,
            "confidence_level": self.confidence_level,
            "empirical_residual_risk": self.empirical_residual_risk,
            "residual_risk_upper_bound": self.residual_risk_upper_bound,
            "accepted_critical_weight": self.accepted_critical_weight,
            "residual_leakage_weight": self.residual_leakage_weight,
            "abstention_rate": self.abstention_rate,
            "total_weight": self.total_weight,
            "method": self.method,
        }


def fit_calibration_thresholds(
    samples: Sequence[Mapping[str, Any] | CalibrationSample],
    *,
    model_id: str,
    suite: str,
    target_leakage: float = 0.0,
    min_recall: float | None = None,
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
    return CalibrationReport(
        model_id=model_id,
        suite=suite,
        groups=reports,
        target_leakage=target_leakage,
        min_recall=recall_floor,
        generated_at=generated_at or _utc_now(),
        membership_defense=defense_policy.to_dict(),
        metadata=dict(metadata or {}),
    )


def build_thresholds_payload(report: CalibrationReport) -> dict[str, Any]:
    """Build the JSON payload written as thresholds.json."""

    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    conformal_quantiles: dict[str, dict[str, dict[str, float]]] = {}
    groups: list[dict[str, Any]] = []
    for group in report.groups:
        thresholds.setdefault(group.model_id, {}).setdefault(group.label, {})[
            group.language
        ] = group.chosen_threshold
        conformal_quantiles.setdefault(group.model_id, {}).setdefault(group.label, {})[
            group.language
        ] = group.nonconformity_quantile
        groups.append(
            {
                "model_id": group.model_id,
                "label": group.label,
                "language": group.language,
                "threshold": group.chosen_threshold,
                "nonconformity_quantile": group.nonconformity_quantile,
                "target_leakage": group.target_leakage,
                "resulting_leakage": group.resulting_leakage,
                "over_redaction": group.over_redaction,
                "recall": group.recall,
                "precision": group.precision,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": THRESHOLDS_ARTIFACT,
        "model_id": report.model_id,
        "suite": report.suite,
        "generated_at": report.generated_at,
        "target_leakage": report.target_leakage,
        "min_recall": report.min_recall,
        "membership_defense": dict(report.membership_defense),
        "thresholds": thresholds,
        "conformal_quantiles": conformal_quantiles,
        "groups": groups,
    }


def write_calibration_artifacts(
    samples: Sequence[Mapping[str, Any] | CalibrationSample],
    *,
    artifact_dir: str | Path,
    model_id: str,
    suite: str,
    target_leakage: float = 0.0,
    min_recall: float | None = None,
    membership_defense: Mapping[str, Any] | MembershipDefensePolicy | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationArtifactPaths:
    """Fit and write thresholds.json plus calibration_report.json."""

    report = fit_calibration_thresholds(
        samples,
        model_id=model_id,
        suite=suite,
        target_leakage=target_leakage,
        min_recall=min_recall,
        membership_defense=membership_defense,
        generated_at=generated_at,
        metadata=metadata,
    )
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = output_dir / "thresholds.json"
    report_path = output_dir / "calibration_report.json"
    thresholds_path.write_text(
        json.dumps(build_thresholds_payload(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CalibrationArtifactPaths(
        artifact_dir=output_dir,
        thresholds_path=thresholds_path,
        report_path=report_path,
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

    raw_quantiles = payload.get("conformal_quantiles") or {}
    quantiles: dict[tuple[str, str, str], float] = {}
    if isinstance(raw_quantiles, Mapping):
        for model_key, labels in raw_quantiles.items():
            if not isinstance(labels, Mapping):
                raise ValueError(f"conformal_quantiles.{model_key} must be an object")
            for label_key, languages in labels.items():
                canonical = normalize_label(str(label_key))
                if not isinstance(languages, Mapping):
                    raise ValueError(
                        f"conformal_quantiles.{model_key}.{canonical} must be an object"
                    )
                for language_key, value in languages.items():
                    language = str(language_key or WILDCARD_LANGUAGE).lower()
                    quantiles[(str(model_key), canonical, language)] = _bounded_float(
                        value,
                        (f"conformal_quantiles.{model_key}.{canonical}.{language}"),
                    )

    if not quantiles:
        quantiles = {key: 1.0 - value for key, value in thresholds.items()}

    return CalibrationThresholdSet(
        schema_version=int(payload["schema_version"]),
        thresholds=thresholds,
        nonconformity_quantiles=quantiles,
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

def score_span_nonconformity(
    span: Mapping[str, Any] | Any,
    thresholds: Mapping[str, Any] | CalibrationThresholdSet,
    *,
    model_id: str | None = None,
    default_language: str = "en",
    default_threshold: float = 0.0,
) -> SpanNonconformityScore:
    """Score one span against calibration-artifact conformal quantiles."""

    data = span if isinstance(span, Mapping) else vars(span)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    language = str(
        data.get("language")
        or data.get("lang")
        or metadata.get("language")
        or metadata.get("lang")
        or default_language
    ).lower()
    raw_label = (
        data.get("canonical_label")
        or data.get("label")
        or data.get("entity_type")
        or data.get("entity_group")
        or data.get("entity")
        or "OTHER"
    )
    label = normalize_label(str(raw_label), language)
    confidence = _bounded_float(
        data.get("confidence", data.get("score", metadata.get("confidence", 1.0))),
        "confidence",
    )
    confidence_threshold = _lookup_confidence_threshold(
        thresholds,
        label,
        language,
        model_id=model_id,
        default=default_threshold,
    )
    nonconformity_quantile = _lookup_nonconformity_quantile(
        thresholds,
        label,
        language,
        model_id=model_id,
        default=1.0 - confidence_threshold,
    )
    nonconformity = 1.0 - confidence
    return SpanNonconformityScore(
        label=label,
        language=language,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
        nonconformity=nonconformity,
        nonconformity_quantile=nonconformity_quantile,
        accepted_by_quantile=nonconformity <= nonconformity_quantile,
    )


def select_risk_controlled_abstention_threshold(
    samples: Sequence[Mapping[str, Any] | CalibrationSample | AbstentionRiskSample],
    *,
    target_risk: float,
    confidence_level: float = 0.95,
    critical_labels: frozenset[str] | None = None,
) -> RiskControlledAbstentionThreshold:
    """Select the loosest nonconformity cutoff whose risk bound is safe.

    The selector bounds residual leakage among accepted critical-label spans
    using a one-sided Hoeffding upper confidence bound. Larger target risks can
    only keep or loosen the selected cutoff, so abstention is monotonic in the
    target risk on a fixed calibration set.
    """

    target_risk = _bounded_float(target_risk, "target_risk")
    confidence_level = _bounded_float(confidence_level, "confidence_level")
    if confidence_level >= 1.0:
        raise ValueError("confidence_level must be less than 1.0")

    critical_set = frozenset(critical_labels or _DEFAULT_CRITICAL_LABELS)
    normalized = [
        sample
        if isinstance(sample, AbstentionRiskSample)
        else AbstentionRiskSample.from_mapping(sample, critical_labels=critical_set)
        for sample in samples
    ]
    if not normalized:
        raise ValueError("abstention risk control requires at least one sample")

    candidates = sorted({0.0, *(sample.nonconformity for sample in normalized)})
    safe_rows: list[dict[str, float]] = []
    for threshold in candidates:
        row = _risk_row_at_nonconformity(
            normalized,
            threshold=threshold,
            confidence_level=confidence_level,
        )
        if row["upper_bound"] <= target_risk:
            safe_rows.append(row)

    if safe_rows:
        chosen = max(safe_rows, key=lambda row: row["threshold"])
    else:
        chosen = _risk_row_at_nonconformity(
            normalized,
            threshold=0.0,
            confidence_level=confidence_level,
        )

    total_weight = sum(sample.weight for sample in normalized)
    accepted_weight = sum(
        sample.weight
        for sample in normalized
        if sample.nonconformity <= chosen["threshold"]
    )
    abstention_rate = 1.0 - (accepted_weight / total_weight)
    return RiskControlledAbstentionThreshold(
        nonconformity_threshold=float(chosen["threshold"]),
        target_risk=target_risk,
        confidence_level=confidence_level,
        empirical_residual_risk=float(chosen["empirical_risk"]),
        residual_risk_upper_bound=float(chosen["upper_bound"]),
        accepted_critical_weight=float(chosen["accepted_critical_weight"]),
        residual_leakage_weight=float(chosen["residual_leakage_weight"]),
        abstention_rate=abstention_rate,
        total_weight=total_weight,
    )


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
        nonconformity_quantile=1.0 - float(chosen["threshold"]),
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


def _lookup_confidence_threshold(
    thresholds: Mapping[str, Any] | CalibrationThresholdSet,
    label: str,
    language: str,
    *,
    model_id: str | None,
    default: float,
) -> float:
    if isinstance(thresholds, CalibrationThresholdSet):
        return thresholds.lookup(
            label,
            language,
            model_id=model_id,
            default=default,
        )
    return _lookup_nested_threshold(
        thresholds,
        label,
        language,
        model_id=model_id,
        default=default,
    )


def _lookup_nonconformity_quantile(
    thresholds: Mapping[str, Any] | CalibrationThresholdSet,
    label: str,
    language: str,
    *,
    model_id: str | None,
    default: float,
) -> float:
    if isinstance(thresholds, CalibrationThresholdSet):
        return thresholds.lookup_nonconformity_quantile(
            label,
            language,
            model_id=model_id,
            default=default,
        )
    conformal = thresholds.get("conformal_quantiles") if thresholds else None
    if isinstance(conformal, Mapping):
        return _lookup_nested_threshold(
            conformal,
            label,
            language,
            model_id=model_id,
            default=default,
        )
    return default


def _lookup_nested_threshold(
    payload: Mapping[str, Any],
    label: str,
    language: str,
    *,
    model_id: str | None,
    default: float,
) -> float:
    candidates: list[Any] = []
    if model_id and isinstance(payload.get(model_id), Mapping):
        candidates.append(payload[model_id])
    candidates.append(payload)
    canonical = normalize_label(label, language)
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        value = candidate[canonical] if canonical in candidate else candidate.get(label)
        if isinstance(value, Mapping):
            lang_value = (
                value[language] if language in value else value.get(WILDCARD_LANGUAGE)
            )
            if lang_value is not None:
                return _bounded_float(lang_value, "threshold")
        elif value is not None:
            return _bounded_float(value, "threshold")
    return default


def _risk_row_at_nonconformity(
    samples: Sequence[AbstentionRiskSample],
    *,
    threshold: float,
    confidence_level: float,
) -> dict[str, float]:
    accepted_critical = [
        sample
        for sample in samples
        if sample.critical and sample.nonconformity <= threshold
    ]
    accepted_weight = sum(sample.weight for sample in accepted_critical)
    residual_weight = sum(
        sample.weight for sample in accepted_critical if sample.residual_leakage
    )
    empirical = residual_weight / accepted_weight if accepted_weight else 0.0
    if accepted_weight:
        delta = max(1.0 - confidence_level, 1e-12)
        radius = math.sqrt(math.log(1.0 / delta) / (2.0 * accepted_weight))
        upper = min(1.0, empirical + radius)
    else:
        upper = 0.0
    return {
        "threshold": threshold,
        "empirical_risk": empirical,
        "upper_bound": upper,
        "accepted_critical_weight": accepted_weight,
        "residual_leakage_weight": residual_weight,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "artifact"


_DEFAULT_CRITICAL_LABELS = frozenset(
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


__all__ = [
    "AbstentionRiskSample",
    "CalibrationArtifactPaths",
    "CalibrationGroupReport",
    "CalibrationReport",
    "CalibrationSample",
    "CalibrationThresholdSet",
    "RiskControlledAbstentionThreshold",
    "SpanNonconformityScore",
    "artifact_dir_for",
    "build_thresholds_payload",
    "coerce_calibration_thresholds",
    "default_suite_calibration_samples",
    "fit_calibration_thresholds",
    "load_calibration_samples",
    "load_calibration_thresholds",
    "score_span_nonconformity",
    "select_risk_controlled_abstention_threshold",
    "write_calibration_artifacts",
]
