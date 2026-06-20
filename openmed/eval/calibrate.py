"""Fit and load held-out decision thresholds for PII release artifacts."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import normalize_label

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
class CalibrationReport:
    """Full calibration result for one artifact write."""

    model_id: str
    suite: str
    groups: tuple[CalibrationGroupReport, ...]
    target_leakage: float
    min_recall: float
    generated_at: str
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
    model_id: str | None = None
    suite: str | None = None
    source_path: str | None = None

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

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": THRESHOLDS_ARTIFACT,
        "model_id": report.model_id,
        "suite": report.suite,
        "generated_at": report.generated_at,
        "target_leakage": report.target_leakage,
        "min_recall": report.min_recall,
        "thresholds": thresholds,
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

    return CalibrationThresholdSet(
        schema_version=int(payload["schema_version"]),
        thresholds=thresholds,
        model_id=(
            str(payload["model_id"]) if payload.get("model_id") is not None else None
        ),
        suite=str(payload["suite"]) if payload.get("suite") is not None else None,
        source_path=source_path,
    )


def artifact_dir_for(model_id: str, suite: str) -> Path:
    """Return the default calibration artifact directory."""

    return Path("artifacts") / "calibration" / _slug(model_id) / _slug(suite)


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
    "CalibrationGroupReport",
    "CalibrationReport",
    "CalibrationSample",
    "CalibrationThresholdSet",
    "artifact_dir_for",
    "build_thresholds_payload",
    "coerce_calibration_thresholds",
    "default_suite_calibration_samples",
    "fit_calibration_thresholds",
    "load_calibration_samples",
    "load_calibration_thresholds",
    "write_calibration_artifacts",
]
