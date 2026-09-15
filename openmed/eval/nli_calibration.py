"""Offline threshold calibration for a clinical NLI entailment gate.

The evaluator treats the entailment score as an acceptance score. A fixture is
accepted when its score is at least the selected threshold and is otherwise
abstained. This makes the error trade-off explicit without pretending that a
threshold is a clinical decision guarantee.

Only aggregate evidence is retained in a report. Premise and hypothesis text
are used to bind the fixture fingerprint, but are never returned by a report,
serializer, or validation error.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

NLI_CALIBRATION_SCHEMA_VERSION = 1
NLI_CALIBRATION_ARTIFACT = "openmed.eval.nli_calibration"
NLI_CALIBRATION_SUITE = "clinical-nli-threshold-calibration"
NLI_ENTAILMENT_LABEL = "entailment"
NLI_NOT_ENTAILMENT_LABEL = "not_entailment"
NLI_GOLD_LABELS = ("contradiction", "entailment", "neutral", "not_entailment")
DEFAULT_NLI_MODEL_ID = "synthetic-nli-model"


@dataclass(frozen=True)
class NLIEntailmentFixture:
    """One synthetic NLI score and its gold entailment label.

    Premise and hypothesis are accepted for scoring provenance, but
    :meth:`to_dict` intentionally returns only hashed text metadata and labels.
    Callers should provide synthetic, offline fixtures for calibration.
    """

    fixture_id: str
    premise: str
    hypothesis: str
    gold_label: str
    entailment_score: float

    def __post_init__(self) -> None:
        """Validate direct construction without retaining unsafe metadata."""

        object.__setattr__(
            self, "fixture_id", _required_identifier(self.fixture_id, "fixture_id")
        )
        object.__setattr__(self, "premise", _required_text(self.premise, "premise"))
        object.__setattr__(
            self,
            "hypothesis",
            _required_text(self.hypothesis, "hypothesis"),
        )
        object.__setattr__(self, "gold_label", _normalise_gold_label(self.gold_label))
        object.__setattr__(
            self,
            "entailment_score",
            _bounded_score(self.entailment_score, "entailment_score"),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        index: int = 0,
    ) -> "NLIEntailmentFixture":
        """Build a fixture from a JSON-compatible mapping.

        Accepted score keys are ``entailment_score``, ``score``,
        ``confidence``, and ``probability``. Gold labels may be standard NLI
        labels or a boolean ``entailment``/``target`` field.
        """

        if not isinstance(payload, Mapping):
            raise ValueError("NLI calibration fixture must be a mapping")

        premise = _required_text(
            _first_present(payload, "premise", "text_a", "sentence1"),
            "premise",
        )
        hypothesis = _required_text(
            _first_present(payload, "hypothesis", "text_b", "sentence2"),
            "hypothesis",
        )

        label_value = _first_present(payload, "gold_label", "label", "gold")
        if label_value is None:
            label_value = _first_present(payload, "entailment", "target")
        gold_label = _normalise_gold_label(label_value)

        score_value = _first_present(
            payload,
            "entailment_score",
            "entailment_probability",
            "entailment_confidence",
            "score",
            "confidence",
            "probability",
        )
        if score_value is None:
            scores = payload.get("scores")
            if isinstance(scores, Mapping):
                score_value = scores.get(NLI_ENTAILMENT_LABEL)
        score = _bounded_score(score_value, "entailment_score")

        fixture_id_value = _first_present(payload, "fixture_id", "id", "uid")
        fixture_id = (
            str(fixture_id_value).strip()
            if fixture_id_value is not None
            else f"fixture-{index}"
        )
        if not fixture_id:
            fixture_id = f"fixture-{index}"

        return cls(
            fixture_id=fixture_id,
            premise=premise,
            hypothesis=hypothesis,
            gold_label=gold_label,
            entailment_score=score,
        )

    @property
    def score(self) -> float:
        """Return the entailment score using the shorter score terminology."""

        return self.entailment_score

    @property
    def is_entailment(self) -> bool:
        """Return whether the gold label is an entailment case."""

        return self.gold_label == NLI_ENTAILMENT_LABEL

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-free fixture summary without premise or hypothesis."""

        return {
            "fixture_id_sha256": _digest_text(self.fixture_id),
            "premise_sha256": _digest_text(self.premise),
            "hypothesis_sha256": _digest_text(self.hypothesis),
            "gold_label": self.gold_label,
            "entailment_score": self.entailment_score,
        }


@dataclass(frozen=True)
class NLIConfusionSummary:
    """Binary confusion counts for accepted entailment versus abstention."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def support(self) -> int:
        """Return the number of scored fixtures."""

        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

    @property
    def tp(self) -> int:
        """Short alias for true positives."""

        return self.true_positives

    @property
    def fp(self) -> int:
        """Short alias for false positives."""

        return self.false_positives

    @property
    def tn(self) -> int:
        """Short alias for true negatives."""

        return self.true_negatives

    @property
    def fn(self) -> int:
        """Short alias for false negatives."""

        return self.false_negatives

    def to_dict(self) -> dict[str, int]:
        """Return JSON-safe confusion counts."""

        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }


@dataclass(frozen=True)
class NLIThresholdPoint:
    """Aggregate NLI operating-point metrics for one score threshold."""

    threshold: float
    support: int
    gold_entailment_count: int
    gold_not_entailment_count: int
    accepted_count: int
    abstained_count: int
    abstention_rate: float
    abstention_by_gold_label: Mapping[str, float]
    confusion: NLIConfusionSummary
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float
    f1: float

    @property
    def accepted_entailment_count(self) -> int:
        """Return the number of accepted positive predictions."""

        return self.confusion.true_positives

    @property
    def accepted_non_entailment_count(self) -> int:
        """Return the number of accepted negative gold cases."""

        return self.confusion.false_positives

    @property
    def abstained_entailment_count(self) -> int:
        """Return the number of abstained positive gold cases."""

        return self.confusion.false_negatives

    @property
    def abstained_non_entailment_count(self) -> int:
        """Return the number of correctly abstained negative gold cases."""

        return self.confusion.true_negatives

    @property
    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        """Return a gold-label-by-decision matrix without raw fixture data."""

        return {
            NLI_ENTAILMENT_LABEL: {
                "accepted": self.confusion.true_positives,
                "abstained": self.confusion.false_negatives,
            },
            NLI_NOT_ENTAILMENT_LABEL: {
                "accepted": self.confusion.false_positives,
                "abstained": self.confusion.true_negatives,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic aggregate metrics only."""

        return {
            "threshold": self.threshold,
            "support": self.support,
            "gold_entailment_count": self.gold_entailment_count,
            "gold_not_entailment_count": self.gold_not_entailment_count,
            "accepted_count": self.accepted_count,
            "abstained_count": self.abstained_count,
            "abstention_rate": self.abstention_rate,
            "abstention_by_gold_label": {
                label: self.abstention_by_gold_label[label]
                for label in sorted(self.abstention_by_gold_label)
            },
            "confusion": self.confusion.to_dict(),
            "confusion_matrix": self.confusion_matrix,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "accuracy": self.accuracy,
            "f1": self.f1,
        }

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to serialized fields."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class NLICalibrationReport:
    """Deterministic, aggregate-only NLI threshold calibration evidence."""

    model_id: str
    model_fingerprint: str
    fixture_fingerprint: str
    fixture_count: int
    gold_label_counts: Mapping[str, int]
    threshold_points: tuple[NLIThresholdPoint, ...]
    recommended_point: NLIThresholdPoint
    selection: str
    selection_constraints: Mapping[str, float]
    model_revision: str | None = None
    suite: str = NLI_CALIBRATION_SUITE
    schema_version: int = NLI_CALIBRATION_SCHEMA_VERSION

    @property
    def recommended_threshold(self) -> float:
        """Return the selected operating threshold."""

        return self.recommended_point.threshold

    @property
    def curve(self) -> tuple[NLIThresholdPoint, ...]:
        """Return threshold points in ascending threshold order."""

        return self.threshold_points

    @property
    def curve_points(self) -> tuple[NLIThresholdPoint, ...]:
        """Compatibility alias for callers using curve terminology."""

        return self.threshold_points

    @property
    def operating_points(self) -> tuple[NLIThresholdPoint, ...]:
        """Return all evaluated operating points."""

        return self.threshold_points

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report without premise or hypothesis text."""

        return {
            "artifact_type": NLI_CALIBRATION_ARTIFACT,
            "schema_version": self.schema_version,
            "suite": self.suite,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_fingerprint": self.model_fingerprint,
            "fixture_fingerprint": self.fixture_fingerprint,
            "fixture_count": self.fixture_count,
            "gold_label_counts": {
                label: int(self.gold_label_counts[label])
                for label in sorted(self.gold_label_counts)
            },
            "threshold_points": [point.to_dict() for point in self.threshold_points],
            "recommended_threshold": self.recommended_threshold,
            "recommended_point": self.recommended_point.to_dict(),
            "selection": self.selection,
            "selection_constraints": {
                key: self.selection_constraints[key]
                for key in sorted(self.selection_constraints)
            },
            "premise_and_hypothesis_included": False,
        }

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to serialized fields."""

        return self.to_dict()[key]

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON evidence to ``path``."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic aggregate calibration table."""

        recommended = self.recommended_point
        lines = [
            "# Clinical NLI Threshold Calibration",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Model | `{self.model_id}` |",
            f"| Model fingerprint | `{self.model_fingerprint}` |",
            f"| Fixture fingerprint | `{self.fixture_fingerprint}` |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Recommended threshold | {_format_float(self.recommended_threshold)} |",
            f"| Selection | `{self.selection}` |",
            "",
            "Scores at or above a threshold are accepted as entailment; lower "
            "scores abstain.",
            "",
            "## Threshold trade-offs",
            "",
            "| Threshold | Accepted | Abstained | Abstention | Precision | "
            "Recall | FPR | F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for point in self.threshold_points:
            lines.append(
                f"| {_format_float(point.threshold)} | {point.accepted_count} | "
                f"{point.abstained_count} | {_format_float(point.abstention_rate)} | "
                f"{_format_float(point.precision)} | {_format_float(point.recall)} | "
                f"{_format_float(point.false_positive_rate)} | "
                f"{_format_float(point.f1)} |"
            )
        lines.extend(
            [
                "",
                "## Recommended confusion summary",
                "",
                "| TP | FP | TN | FN | Accuracy | Specificity |",
                "|---:|---:|---:|---:|---:|---:|",
                f"| {recommended.confusion.true_positives} | "
                f"{recommended.confusion.false_positives} | "
                f"{recommended.confusion.true_negatives} | "
                f"{recommended.confusion.false_negatives} | "
                f"{_format_float(recommended.accuracy)} | "
                f"{_format_float(recommended.specificity)} |",
                "",
                "_This report is evaluation evidence, not a clinical decision "
                "guarantee._",
                "",
            ]
        )
        return "\n".join(lines)

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown evidence to ``path``."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def calibrate_nli_thresholds(
    fixtures: Iterable[NLIEntailmentFixture | Mapping[str, Any]] | None = None,
    *,
    model_id: str = DEFAULT_NLI_MODEL_ID,
    model_name: str | None = None,
    model_revision: str | None = None,
    thresholds: Sequence[int | float] | None = None,
    precision_floor: float | None = None,
    recall_floor: float | None = None,
    false_positive_rate_ceiling: float | None = None,
    target_fpr: float | None = None,
) -> NLICalibrationReport:
    """Build an offline NLI threshold, abstention, and confusion report.

    By default, thresholds are the unique fixture scores plus ``0.0`` and
    ``1.0``. The recommended point maximizes F1, with recall, precision, and
    lower threshold used as deterministic tie-breakers. Optional floors or an
    FPR ceiling constrain the recommendation while preserving the same stable
    ordering. A supplied ``model_name`` is accepted as a compatibility alias
    for ``model_id``.

    Args:
        fixtures: NLI fixture mappings or :class:`NLIEntailmentFixture`
            instances. When omitted, the committed in-memory synthetic set is
            evaluated.
        model_id: Stable caller-provided model identifier.
        model_name: Compatibility alias for ``model_id``.
        model_revision: Optional non-text model revision or artifact version.
        thresholds: Explicit score thresholds. Values are deduplicated and
            evaluated in ascending order.
        precision_floor: Optional minimum precision for recommendation.
        recall_floor: Optional minimum recall for recommendation.
        false_positive_rate_ceiling: Optional maximum false-positive rate.
        target_fpr: Compatibility alias for the FPR ceiling.

    Returns:
        A deterministic report containing aggregate metrics and fingerprints.

    Raises:
        ValueError: If fixtures, scores, labels, thresholds, or constraints
            are invalid. Error messages never include fixture text.
    """

    active_model_id = model_name if model_name is not None else model_id
    active_model_id = _required_identifier(active_model_id, "model_id")
    revision = _optional_identifier(model_revision, "model_revision")

    normalized = _normalise_fixtures(fixtures)
    if not normalized:
        raise ValueError("NLI calibration requires at least one fixture")

    threshold_values = _threshold_values(normalized, thresholds=thresholds)
    points = tuple(
        _build_threshold_point(normalized, threshold) for threshold in threshold_values
    )

    if target_fpr is not None:
        if false_positive_rate_ceiling is not None:
            raise ValueError("provide only one false-positive-rate ceiling")
        false_positive_rate_ceiling = target_fpr
    constraints = _constraints(
        precision_floor=precision_floor,
        recall_floor=recall_floor,
        false_positive_rate_ceiling=false_positive_rate_ceiling,
    )
    recommended, selection = _select_recommended_point(points, constraints)

    counts = Counter(fixture.gold_label for fixture in normalized)
    return NLICalibrationReport(
        model_id=active_model_id,
        model_revision=revision,
        model_fingerprint=fingerprint_nli_model(active_model_id, revision),
        fixture_fingerprint=fingerprint_nli_fixtures(normalized),
        fixture_count=len(normalized),
        gold_label_counts=dict(counts),
        threshold_points=points,
        recommended_point=recommended,
        selection=selection,
        selection_constraints=constraints,
    )


def build_nli_calibration_report(
    fixtures: Iterable[NLIEntailmentFixture | Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> NLICalibrationReport:
    """Compatibility wrapper for :func:`calibrate_nli_thresholds`."""

    return calibrate_nli_thresholds(fixtures, **kwargs)


def nli_calibration_report(
    fixtures: Iterable[NLIEntailmentFixture | Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> NLICalibrationReport:
    """Compatibility wrapper for :func:`calibrate_nli_thresholds`."""

    return calibrate_nli_thresholds(fixtures, **kwargs)


def fingerprint_nli_model(model_id: str, model_revision: str | None = None) -> str:
    """Return a stable model fingerprint without exposing model internals."""

    identifier = _required_identifier(model_id, "model_id")
    revision = _optional_identifier(model_revision, "model_revision")
    return _digest_json({"model_id": identifier, "model_revision": revision})


def fingerprint_nli_fixtures(
    fixtures: Iterable[NLIEntailmentFixture | Mapping[str, Any]],
) -> str:
    """Return a content fingerprint for NLI fixtures without serializing text."""

    normalized = _normalise_fixtures(fixtures)
    if not normalized:
        raise ValueError("NLI fixture fingerprint requires at least one fixture")
    material = sorted(
        (
            {
                "fixture_id_sha256": _digest_text(fixture.fixture_id),
                "premise_sha256": _digest_text(fixture.premise),
                "hypothesis_sha256": _digest_text(fixture.hypothesis),
                "gold_label": fixture.gold_label,
                "entailment_score": fixture.entailment_score,
            }
            for fixture in normalized
        ),
        key=_canonical_json,
    )
    return _digest_json(material)


NLICalibrationFixture = NLIEntailmentFixture
NLIThresholdSummary = NLIThresholdPoint


def _normalise_fixtures(
    fixtures: Iterable[NLIEntailmentFixture | Mapping[str, Any]] | None,
) -> tuple[NLIEntailmentFixture, ...]:
    source = DEFAULT_NLI_FIXTURES if fixtures is None else fixtures
    normalized: list[NLIEntailmentFixture] = []
    try:
        for index, fixture in enumerate(source):
            if isinstance(fixture, NLIEntailmentFixture):
                normalized.append(fixture)
            else:
                normalized.append(
                    NLIEntailmentFixture.from_mapping(fixture, index=index)
                )
    except (TypeError, ValueError) as exc:
        if str(exc).startswith("NLI calibration fixture"):
            raise
        raise ValueError("invalid NLI calibration fixture") from exc
    return tuple(normalized)


def _threshold_values(
    fixtures: Sequence[NLIEntailmentFixture],
    *,
    thresholds: Sequence[int | float] | None,
) -> tuple[float, ...]:
    if thresholds is None:
        values: Iterable[Any] = (0.0, 1.0, *(fixture.score for fixture in fixtures))
    else:
        values = thresholds
    values_set: set[float] = set()
    for value in values:
        values_set.add(_bounded_score(value, "threshold"))
    if not values_set:
        raise ValueError("thresholds must include at least one value")
    return tuple(sorted(values_set))


def _build_threshold_point(
    fixtures: Sequence[NLIEntailmentFixture], threshold: float
) -> NLIThresholdPoint:
    positive_count = sum(1 for fixture in fixtures if fixture.is_entailment)
    negative_count = len(fixtures) - positive_count
    accepted = [fixture.score >= threshold for fixture in fixtures]
    true_positives = sum(
        decision and fixture.is_entailment
        for fixture, decision in zip(fixtures, accepted)
    )
    false_positives = sum(
        decision and not fixture.is_entailment
        for fixture, decision in zip(fixtures, accepted)
    )
    true_negatives = sum(
        not decision and not fixture.is_entailment
        for fixture, decision in zip(fixtures, accepted)
    )
    false_negatives = sum(
        not decision and fixture.is_entailment
        for fixture, decision in zip(fixtures, accepted)
    )
    confusion = NLIConfusionSummary(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )
    abstained_count = len(fixtures) - sum(accepted)
    by_label: dict[str, float] = {}
    label_counts = Counter(fixture.gold_label for fixture in fixtures)
    for label in sorted(label_counts):
        abstained = sum(
            not decision
            for fixture, decision in zip(fixtures, accepted)
            if fixture.gold_label == label
        )
        by_label[label] = _rate(abstained, label_counts[label], zero=0.0)

    return NLIThresholdPoint(
        threshold=threshold,
        support=len(fixtures),
        gold_entailment_count=positive_count,
        gold_not_entailment_count=negative_count,
        accepted_count=sum(accepted),
        abstained_count=abstained_count,
        abstention_rate=_rate(abstained_count, len(fixtures), zero=0.0),
        abstention_by_gold_label=by_label,
        confusion=confusion,
        precision=_rate(true_positives, true_positives + false_positives, zero=1.0),
        recall=_rate(true_positives, positive_count, zero=0.0),
        specificity=_rate(true_negatives, negative_count, zero=1.0),
        false_positive_rate=_rate(false_positives, negative_count, zero=0.0),
        false_negative_rate=_rate(false_negatives, positive_count, zero=0.0),
        accuracy=_rate(
            true_positives + true_negatives,
            len(fixtures),
            zero=0.0,
        ),
        f1=_f1(true_positives, false_positives, false_negatives),
    )


def _constraints(
    *,
    precision_floor: float | None,
    recall_floor: float | None,
    false_positive_rate_ceiling: float | None,
) -> dict[str, float]:
    raw = {
        "precision_floor": precision_floor,
        "recall_floor": recall_floor,
        "false_positive_rate_ceiling": false_positive_rate_ceiling,
    }
    return {
        key: _bounded_score(value, key)
        for key, value in raw.items()
        if value is not None
    }


def _select_recommended_point(
    points: Sequence[NLIThresholdPoint], constraints: Mapping[str, float]
) -> tuple[NLIThresholdPoint, str]:
    candidates = [
        point
        for point in points
        if point.precision >= constraints.get("precision_floor", 0.0)
        and point.recall >= constraints.get("recall_floor", 0.0)
        and point.false_positive_rate
        <= constraints.get("false_positive_rate_ceiling", 1.0)
    ]
    if candidates:
        if "false_positive_rate_ceiling" in constraints:
            selection = "max_recall_at_false_positive_rate_ceiling"
        elif "precision_floor" in constraints:
            selection = "max_recall_at_precision_floor"
        elif "recall_floor" in constraints:
            selection = "max_precision_at_recall_floor"
        else:
            selection = "max_f1"
        return _max_point(candidates, objective=selection), selection

    return (
        _max_point(points, objective="max_f1"),
        "max_f1_no_point_met_constraints",
    )


def _max_point(
    points: Sequence[NLIThresholdPoint], *, objective: str
) -> NLIThresholdPoint:
    if objective == "max_recall_at_false_positive_rate_ceiling":
        key = lambda point: (
            point.recall,
            point.precision,
            point.f1,
            -point.abstention_rate,
            -point.threshold,
        )
    elif objective == "max_recall_at_precision_floor":
        key = lambda point: (
            point.recall,
            point.precision,
            point.f1,
            -point.abstention_rate,
            -point.threshold,
        )
    elif objective == "max_precision_at_recall_floor":
        key = lambda point: (
            point.precision,
            point.recall,
            point.f1,
            -point.abstention_rate,
            -point.threshold,
        )
    else:
        key = lambda point: (
            point.f1,
            point.recall,
            point.precision,
            -point.abstention_rate,
            -point.threshold,
        )
    return max(points, key=key)


def _normalise_gold_label(value: Any) -> str:
    if isinstance(value, bool):
        return NLI_ENTAILMENT_LABEL if value else NLI_NOT_ENTAILMENT_LABEL
    if value is None:
        raise ValueError("NLI calibration fixture requires a gold label")
    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    aliases = {
        "entailment": NLI_ENTAILMENT_LABEL,
        "entailed": NLI_ENTAILMENT_LABEL,
        "positive": NLI_ENTAILMENT_LABEL,
        "true": NLI_ENTAILMENT_LABEL,
        "yes": NLI_ENTAILMENT_LABEL,
        "contradiction": "contradiction",
        "contradicted": "contradiction",
        "neutral": "neutral",
        "unknown": "neutral",
        "not_entailment": NLI_NOT_ENTAILMENT_LABEL,
        "not entailment": NLI_NOT_ENTAILMENT_LABEL,
        "non_entailment": NLI_NOT_ENTAILMENT_LABEL,
        "non-entailment": NLI_NOT_ENTAILMENT_LABEL,
        "negative": NLI_NOT_ENTAILMENT_LABEL,
        "false": NLI_NOT_ENTAILMENT_LABEL,
        "no": NLI_NOT_ENTAILMENT_LABEL,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "NLI calibration fixture has an unsupported gold label"
        ) from exc


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"NLI calibration fixture requires {field_name}")
    text = unicodedata.normalize("NFKC", str(value))
    if not text.strip():
        raise ValueError(f"NLI calibration fixture requires {field_name}")
    return text


def _required_identifier(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be a non-empty identifier")
    identifier = unicodedata.normalize("NFKC", str(value)).strip()
    if not identifier:
        raise ValueError(f"{field_name} must be a non-empty identifier")
    return identifier


def _optional_identifier(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_identifier(value, field_name)


def _bounded_score(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return result


def _rate(numerator: int, denominator: int, *, zero: float) -> float:
    return numerator / denominator if denominator else zero


def _f1(true_positives: int, false_positives: int, false_negatives: int) -> float:
    denominator = 2 * true_positives + false_positives + false_negatives
    return 2 * true_positives / denominator if denominator else 0.0


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_json(value: Any) -> str:
    return (
        "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    )


def _digest_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _format_float(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


DEFAULT_NLI_FIXTURES: tuple[NLIEntailmentFixture, ...] = (
    NLIEntailmentFixture(
        fixture_id="synthetic-alpha",
        premise="Synthetic case alpha contains marker cobalt.",
        hypothesis="The case contains marker cobalt.",
        gold_label=NLI_ENTAILMENT_LABEL,
        entailment_score=0.96,
    ),
    NLIEntailmentFixture(
        fixture_id="synthetic-beta",
        premise="Synthetic case beta contains marker ochre.",
        hypothesis="The case contains marker cobalt.",
        gold_label="contradiction",
        entailment_score=0.18,
    ),
    NLIEntailmentFixture(
        fixture_id="synthetic-gamma",
        premise="Synthetic case gamma contains marker jade.",
        hypothesis="The case contains a marker.",
        gold_label="neutral",
        entailment_score=0.44,
    ),
    NLIEntailmentFixture(
        fixture_id="synthetic-delta",
        premise="Synthetic case delta contains marker umber.",
        hypothesis="The case contains marker umber.",
        gold_label=NLI_ENTAILMENT_LABEL,
        entailment_score=0.79,
    ),
)


__all__ = [
    "DEFAULT_NLI_FIXTURES",
    "DEFAULT_NLI_MODEL_ID",
    "NLI_CALIBRATION_ARTIFACT",
    "NLI_CALIBRATION_SCHEMA_VERSION",
    "NLI_CALIBRATION_SUITE",
    "NLI_ENTAILMENT_LABEL",
    "NLI_GOLD_LABELS",
    "NLI_NOT_ENTAILMENT_LABEL",
    "NLICalibrationFixture",
    "NLICalibrationReport",
    "NLIConfusionSummary",
    "NLIEntailmentFixture",
    "NLIThresholdSummary",
    "NLIThresholdPoint",
    "build_nli_calibration_report",
    "calibrate_nli_thresholds",
    "fingerprint_nli_fixtures",
    "fingerprint_nli_model",
    "nli_calibration_report",
]
