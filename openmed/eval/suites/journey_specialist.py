"""Frozen-holdout evaluation and promotion gates for Journey specialists."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import canonical_json, sha256_digest
from openmed.clinical.model_packs import (
    CalibrationMetadata,
    FallbackPolicy,
    ModelPackEntry,
    QuantizationMetadata,
)
from openmed.eval.metrics import expected_calibration_error, reliability_bins
from openmed.structured.store import StoreResult, StoreState
from openmed.training.journey_specialist import (
    JOURNEY_SPECIALIST_COMPATIBILITY_POLICY,
    JOURNEY_SPECIALIST_SCHEMA_VERSION,
    JOURNEY_SPECIALIST_TASKS,
    GpuSpendLedger,
    JourneySpecialistConflictError,
    JourneySpecialistDeniedError,
    SpecialistRecipe,
    SpecialistRunManifest,
)

PROMOTION_DECISIONS = frozenset({"hold", "promote"})
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")


class JourneySpecialistEvaluationError(ValueError):
    """Raised when holdout evidence violates the public contract."""


@dataclass(frozen=True, slots=True)
class SpecialistPrediction:
    """One value-free frozen-holdout prediction and quantized counterpart."""

    example_id: str
    task: str
    gold_label: str
    scores: Mapping[str, float] = field(repr=False)
    quantized_scores: Mapping[str, float] | None = field(default=None, repr=False)
    subgroups: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _controlled(self.example_id, "example_id")
        if self.task not in JOURNEY_SPECIALIST_TASKS:
            raise JourneySpecialistEvaluationError("prediction task is unsupported")
        _controlled(self.gold_label, "gold_label")
        scores = _probabilities(self.scores, "scores")
        if self.gold_label not in scores:
            raise JourneySpecialistEvaluationError(
                "gold label is absent from prediction scores"
            )
        quantized = None
        if self.quantized_scores is not None:
            quantized = _probabilities(self.quantized_scores, "quantized_scores")
            if set(quantized) != set(scores):
                raise JourneySpecialistEvaluationError(
                    "quantized score labels differ from full precision"
                )
        subgroup_data = _mapping(self.subgroups, "subgroups")
        subgroups = MappingProxyType(
            {
                _controlled(key, "subgroup axis"): _controlled(value, "subgroup value")
                for key, value in sorted(subgroup_data.items())
            }
        )
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "quantized_scores", quantized)
        object.__setattr__(self, "subgroups", subgroups)

    @property
    def predicted_label(self) -> str:
        """Return the stable full-precision argmax label."""

        return _argmax(self.scores)

    @property
    def quantized_label(self) -> str | None:
        """Return the stable quantized argmax label when present."""

        return (
            _argmax(self.quantized_scores)
            if self.quantized_scores is not None
            else None
        )

    @property
    def confidence(self) -> float:
        """Return full-precision maximum probability."""

        return max(self.scores.values())


@dataclass(frozen=True, slots=True)
class SpecialistPromotionPolicy:
    """Named baseline, aliases, and abstention threshold for one evaluation."""

    task: str
    baseline_id: str
    baseline_macro_f1: float
    candidate_alias: str
    fallback_alias: str
    abstention_threshold: float = 0.7
    schema_version: str = JOURNEY_SPECIALIST_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_SPECIALIST_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.task not in JOURNEY_SPECIALIST_TASKS:
            raise JourneySpecialistEvaluationError("promotion task is unsupported")
        _controlled(self.baseline_id, "baseline_id")
        _controlled(self.candidate_alias, "candidate_alias")
        _controlled(self.fallback_alias, "fallback_alias")
        if self.candidate_alias == self.fallback_alias:
            raise JourneySpecialistEvaluationError(
                "candidate and fallback aliases must differ"
            )
        _unit_interval(self.baseline_macro_f1, "baseline_macro_f1")
        _unit_interval(self.abstention_threshold, "abstention_threshold")
        _contract_version(self.schema_version, self.compatibility_policy)

    @classmethod
    def from_recipe(
        cls,
        recipe: SpecialistRecipe,
        *,
        baseline_macro_f1: float,
        candidate_alias: str,
        fallback_alias: str,
        abstention_threshold: float = 0.7,
    ) -> "SpecialistPromotionPolicy":
        """Build the evaluation policy pinned by a training recipe."""

        return cls(
            task=recipe.task,
            baseline_id=recipe.baseline_id,
            baseline_macro_f1=baseline_macro_f1,
            candidate_alias=candidate_alias,
            fallback_alias=fallback_alias,
            abstention_threshold=abstention_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the public promotion policy."""

        return {
            "abstention_threshold": self.abstention_threshold,
            "baseline_id": self.baseline_id,
            "baseline_macro_f1": self.baseline_macro_f1,
            "candidate_alias": self.candidate_alias,
            "compatibility_policy": self.compatibility_policy,
            "fallback_alias": self.fallback_alias,
            "schema_version": self.schema_version,
            "task": self.task,
        }


@dataclass(frozen=True, slots=True)
class SpecialistHoldoutReport:
    """Raw-free metrics, slices, calibration, and promotion decision."""

    task: str
    dataset_digest: str
    calibration_digest: str
    holdout_digest: str
    recipe_digest: str
    policy: SpecialistPromotionPolicy
    sample_count: int
    metrics: Mapping[str, Any]
    per_class: Mapping[str, Mapping[str, Any]]
    calibration: Mapping[str, Any]
    abstention: Mapping[str, Any]
    subgroup_slices: Mapping[str, Mapping[str, Any]]
    quantized_delta: Mapping[str, Any]
    failure_slices: Mapping[str, int]
    promotion_decision: str
    promotion_reasons: tuple[str, ...]
    selected_alias: str
    schema_version: str = JOURNEY_SPECIALIST_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_SPECIALIST_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.task not in JOURNEY_SPECIALIST_TASKS:
            raise JourneySpecialistEvaluationError("report task is unsupported")
        _digest(self.dataset_digest, "dataset_digest")
        _digest(self.calibration_digest, "calibration_digest")
        _digest(self.holdout_digest, "holdout_digest")
        _digest(self.recipe_digest, "recipe_digest")
        if not isinstance(self.policy, SpecialistPromotionPolicy):
            raise TypeError("report policy must be SpecialistPromotionPolicy")
        if self.policy.task != self.task:
            raise JourneySpecialistEvaluationError("report policy task differs")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise JourneySpecialistEvaluationError("sample_count must be positive")
        if self.promotion_decision not in PROMOTION_DECISIONS:
            raise JourneySpecialistEvaluationError("promotion decision is unsupported")
        reasons = tuple(
            sorted(
                _controlled(item, "promotion reason") for item in self.promotion_reasons
            )
        )
        if self.promotion_decision == "promote" and reasons:
            raise JourneySpecialistEvaluationError(
                "promoted reports cannot contain blocking reasons"
            )
        expected_alias = (
            self.policy.candidate_alias
            if self.promotion_decision == "promote"
            else self.policy.fallback_alias
        )
        if self.selected_alias != expected_alias:
            raise JourneySpecialistEvaluationError(
                "selected alias does not match promotion decision"
            )
        _contract_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(
            self, "metrics", _freeze_json_mapping(self.metrics, "metrics")
        )
        object.__setattr__(
            self,
            "per_class",
            _freeze_nested_json_mapping(self.per_class, "per_class"),
        )
        object.__setattr__(
            self,
            "calibration",
            _freeze_json_mapping(self.calibration, "calibration"),
        )
        object.__setattr__(
            self,
            "abstention",
            _freeze_json_mapping(self.abstention, "abstention"),
        )
        object.__setattr__(
            self,
            "subgroup_slices",
            _freeze_nested_json_mapping(self.subgroup_slices, "subgroup_slices"),
        )
        object.__setattr__(
            self,
            "quantized_delta",
            _freeze_json_mapping(self.quantized_delta, "quantized_delta"),
        )
        object.__setattr__(
            self,
            "failure_slices",
            _count_mapping(self.failure_slices, "failure_slices"),
        )
        object.__setattr__(self, "promotion_reasons", reasons)

    @property
    def digest(self) -> str:
        """Return the canonical report digest."""

        return sha256_digest(canonical_json(self.to_dict()))

    @property
    def failure_slices_digest(self) -> str:
        """Return the canonical failure-slice digest."""

        return sha256_digest(canonical_json(self.failure_slices))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned holdout report."""

        return {
            "abstention": _plain(self.abstention),
            "calibration": _plain(self.calibration),
            "calibration_digest": self.calibration_digest,
            "compatibility_policy": self.compatibility_policy,
            "dataset_digest": self.dataset_digest,
            "failure_slices": dict(self.failure_slices),
            "holdout_digest": self.holdout_digest,
            "metrics": _plain(self.metrics),
            "per_class": _plain(self.per_class),
            "policy": self.policy.to_dict(),
            "promotion_decision": self.promotion_decision,
            "promotion_reasons": list(self.promotion_reasons),
            "quantized_delta": _plain(self.quantized_delta),
            "recipe_digest": self.recipe_digest,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "selected_alias": self.selected_alias,
            "subgroup_slices": _plain(self.subgroup_slices),
            "task": self.task,
        }

    def to_json(self) -> str:
        """Return deterministic compact JSON."""

        return canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class SpecialistRunCompletion:
    """Updated shared spend ledger and finalized run manifest."""

    ledger: GpuSpendLedger
    manifest: SpecialistRunManifest

    def __post_init__(self) -> None:
        if not isinstance(self.ledger, GpuSpendLedger):
            raise TypeError("completion ledger must be GpuSpendLedger")
        if not isinstance(self.manifest, SpecialistRunManifest):
            raise TypeError("completion manifest must be SpecialistRunManifest")
        if self.manifest.ledger_digest != self.ledger.digest:
            raise JourneySpecialistEvaluationError(
                "completion ledger and manifest disagree"
            )


def evaluate_journey_specialist_holdout(
    predictions: Sequence[SpecialistPrediction],
    *,
    recipe: SpecialistRecipe,
    policy: SpecialistPromotionPolicy,
    dataset_digest: str,
    calibration_digest: str,
    holdout_digest: str,
    calibration_bins: int = 10,
) -> StoreResult[SpecialistHoldoutReport]:
    """Evaluate one frozen holdout and choose candidate or safe fallback."""

    if not predictions:
        return StoreResult.outcome(StoreState.UNKNOWN, "holdout_predictions_missing")
    if policy.task != recipe.task or policy.baseline_id != recipe.baseline_id:
        return StoreResult.outcome(StoreState.CONFLICT, "promotion_policy_mismatch")
    try:
        _digest(dataset_digest, "dataset_digest")
        _digest(calibration_digest, "calibration_digest")
        _digest(holdout_digest, "holdout_digest")
        if calibration_digest == holdout_digest:
            raise JourneySpecialistEvaluationError(
                "calibration and holdout digests must differ"
            )
        if type(calibration_bins) is not int or calibration_bins < 2:
            raise JourneySpecialistEvaluationError(
                "calibration_bins must be at least two"
            )
    except JourneySpecialistEvaluationError:
        return StoreResult.outcome(StoreState.FAILURE, "holdout_contract_invalid")

    materialized = tuple(predictions)
    if any(not isinstance(item, SpecialistPrediction) for item in materialized):
        return StoreResult.outcome(StoreState.FAILURE, "holdout_prediction_invalid")
    tasks = {item.task for item in materialized}
    if tasks != {recipe.task}:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "holdout_task_unsupported")
    ids = tuple(item.example_id for item in materialized)
    if len(ids) != len(set(ids)):
        return StoreResult.outcome(StoreState.CONFLICT, "holdout_examples_duplicate")
    if any(set(item.scores) != set(recipe.labels) for item in materialized):
        return StoreResult.outcome(StoreState.CONFLICT, "holdout_labels_mismatch")
    quantized_present = tuple(
        item.quantized_scores is not None for item in materialized
    )
    if not all(quantized_present):
        return StoreResult.outcome(StoreState.PARTIAL, "quantized_holdout_incomplete")

    full_metrics = _classification_metrics(materialized, recipe.labels, quantized=False)
    quant_metrics = _classification_metrics(materialized, recipe.labels, quantized=True)
    bins = reliability_bins(
        (
            {
                "confidence": item.confidence,
                "correct": item.predicted_label == item.gold_label,
            }
            for item in materialized
        ),
        n_bins=calibration_bins,
    )
    ece = expected_calibration_error(bins)
    calibration = {
        "expected_calibration_error": ece,
        "method": recipe.calibration_method,
        "n_bins": calibration_bins,
        "reliability": bins,
    }
    abstention = _abstention_metrics(materialized, policy.abstention_threshold)
    subgroup_slices = _subgroup_metrics(materialized, recipe.labels)
    quantized_delta = _quantized_delta(full_metrics, quant_metrics, materialized)
    failure_slices = _failure_slices(materialized, policy.abstention_threshold)
    reasons = _promotion_reasons(
        recipe,
        policy,
        full_metrics,
        calibration,
        abstention,
        subgroup_slices,
        quantized_delta,
    )
    decision = "promote" if not reasons else "hold"
    report = SpecialistHoldoutReport(
        task=recipe.task,
        dataset_digest=dataset_digest,
        calibration_digest=calibration_digest,
        holdout_digest=holdout_digest,
        recipe_digest=recipe.digest,
        policy=policy,
        sample_count=len(materialized),
        metrics={
            "accuracy": full_metrics["accuracy"],
            "macro_f1": full_metrics["macro_f1"],
            "quantized_accuracy": quant_metrics["accuracy"],
            "quantized_macro_f1": quant_metrics["macro_f1"],
        },
        per_class=full_metrics["per_class"],
        calibration=calibration,
        abstention=abstention,
        subgroup_slices=subgroup_slices,
        quantized_delta=quantized_delta,
        failure_slices=failure_slices,
        promotion_decision=decision,
        promotion_reasons=tuple(reasons),
        selected_alias=(
            policy.candidate_alias if decision == "promote" else policy.fallback_alias
        ),
    )
    return StoreResult.success(report)


def finalize_journey_specialist_run(
    dry_manifest: SpecialistRunManifest,
    report: SpecialistHoldoutReport,
    ledger: GpuSpendLedger,
    *,
    actual_cost_usd: float,
    actual_gpu_hours: float,
    artifact_digests: Mapping[str, str],
    model_card_digest: str,
) -> StoreResult[SpecialistRunCompletion]:
    """Finalize a run only after spend, artifacts, and holdout agree."""

    if not dry_manifest.dry_run or dry_manifest.promotion_decision != "not_evaluated":
        return StoreResult.outcome(StoreState.CONFLICT, "run_manifest_not_dry")
    if dry_manifest.ledger_digest != ledger.digest:
        return StoreResult.outcome(StoreState.CONFLICT, "run_ledger_mismatch")
    if (
        report.task != dry_manifest.task
        or report.recipe_digest != dry_manifest.recipe.digest
        or report.dataset_digest != dry_manifest.dataset.content_digest
    ):
        return StoreResult.outcome(StoreState.CONFLICT, "run_evaluation_mismatch")
    try:
        measured_gpu_hours = _finite(actual_gpu_hours, "actual_gpu_hours")
        if measured_gpu_hours > dry_manifest.recipe.stop_rule.maximum_gpu_hours:
            return StoreResult.outcome(StoreState.DENIED, "training_stop_rule_exceeded")
        artifacts = _digest_mapping(artifact_digests, "artifact_digests")
        required = set(dry_manifest.recipe.export_formats) | set(
            dry_manifest.recipe.quantization_targets
        )
        if not required.issubset(artifacts):
            raise JourneySpecialistConflictError(
                "run artifacts do not include every declared export"
            )
        _digest(model_card_digest, "model_card_digest")
        updated_ledger = ledger.record_actual(
            dry_manifest.run_id,
            actual_cost_usd=actual_cost_usd,
            actual_gpu_hours=measured_gpu_hours,
        )
        manifest = replace(
            dry_manifest,
            ledger_digest=updated_ledger.digest,
            actual_cost_usd=float(actual_cost_usd),
            metrics_digest=report.digest,
            failure_slices_digest=report.failure_slices_digest,
            model_card_digest=model_card_digest,
            artifact_digests=artifacts,
            promotion_decision=report.promotion_decision,
            promotion_reason=(
                "all_promotion_gates_passed"
                if report.promotion_decision == "promote"
                else report.promotion_reasons[0]
            ),
            dry_run=False,
        )
        return StoreResult.success(
            SpecialistRunCompletion(ledger=updated_ledger, manifest=manifest)
        )
    except JourneySpecialistDeniedError:
        return StoreResult.outcome(StoreState.DENIED, "gpu_budget_exceeded")
    except (JourneySpecialistConflictError, JourneySpecialistEvaluationError):
        return StoreResult.outcome(StoreState.CONFLICT, "run_artifact_conflict")
    except (TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "run_completion_invalid")


def render_journey_specialist_model_card(
    manifest: SpecialistRunManifest,
    report: SpecialistHoldoutReport,
) -> str:
    """Render a deterministic task-level model card from measured evidence."""

    if (
        manifest.task != report.task
        or manifest.recipe.digest != report.recipe_digest
        or manifest.dataset.content_digest != report.dataset_digest
    ):
        raise JourneySpecialistEvaluationError("model-card evidence does not match")
    lines = [
        f"# {manifest.pack_id} / {manifest.task}",
        "",
        f"Promotion decision: **{report.promotion_decision}**",
        f"Selected runtime alias: `{report.selected_alias}`",
        "",
        "## Training lineage",
        "",
        f"- Backbone: `{manifest.recipe.backbone.model_id}`",
        f"- Backbone revision: `{manifest.recipe.backbone.revision}`",
        f"- Backbone license: `{manifest.recipe.backbone.license}`",
        f"- Dataset digest: `{manifest.dataset.content_digest}`",
        f"- Calibration digest: `{report.calibration_digest}`",
        f"- Split digest: `{manifest.split.assignment_digest}`",
        f"- Recipe digest: `{manifest.recipe.digest}`",
        f"- Code revision: `{manifest.code_revision}`",
        f"- Seed: `{manifest.recipe.seed}`",
        "",
        "## Frozen holdout",
        "",
        f"- Samples: `{report.sample_count}`",
        f"- Macro F1: `{float(report.metrics['macro_f1']):.6f}`",
        f"- Accuracy: `{float(report.metrics['accuracy']):.6f}`",
        f"- ECE: `{float(report.calibration['expected_calibration_error']):.6f}`",
        f"- Coverage: `{float(report.abstention['coverage']):.6f}`",
        f"- Quantized macro-F1 delta: "
        f"`{float(report.quantized_delta['macro_f1_delta']):.6f}`",
        "",
        "## Promotion reasons",
        "",
    ]
    if report.promotion_reasons:
        lines.extend(f"- `{item}`" for item in report.promotion_reasons)
    else:
        lines.append("- All declared promotion gates passed.")
    lines.extend(
        [
            "",
            "## Intended use and limitations",
            "",
            "This bounded specialist emits only its declared label set and must "
            "fall back or abstain when its evidence gate does not pass. It is not "
            "clinically validated and cannot autonomously diagnose, treat, enroll, "
            "contact, order, or otherwise act on a patient.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_journey_specialist_model_pack_entry(
    completion: SpecialistRunCompletion,
    report: SpecialistHoldoutReport,
    *,
    artifact_format: str = "int8",
    runtime: str = "onnx",
    languages: Sequence[str] = ("en",),
    domains: Sequence[str] = ("clinical",),
    priority: int = 10,
) -> StoreResult[ModelPackEntry]:
    """Bridge one promoted run into the versioned local model-pack contract."""

    manifest = completion.manifest
    if (
        report.promotion_decision != "promote"
        or manifest.promotion_decision != "promote"
    ):
        return StoreResult.outcome(StoreState.DENIED, "specialist_not_promoted")
    if (
        manifest.task != report.task
        or manifest.metrics_digest != report.digest
        or report.selected_alias != report.policy.candidate_alias
    ):
        return StoreResult.outcome(StoreState.CONFLICT, "model_pack_evidence_mismatch")
    artifact_digest = manifest.artifact_digests.get(artifact_format)
    if artifact_digest is None:
        return StoreResult.outcome(StoreState.CONFLICT, "model_pack_artifact_missing")
    try:
        entry = ModelPackEntry(
            alias=report.policy.candidate_alias,
            task=manifest.task,
            family="deberta_v2",
            artifact_id=f"local/{manifest.pack_id}.{manifest.task}",
            artifact_kind="model",
            revision=artifact_digest.removeprefix("sha256:"),
            artifact_digest=artifact_digest,
            license=manifest.recipe.backbone.license,
            runtime=runtime,
            model_kind="bounded",
            output_schema=manifest.recipe.output_schema,
            output_schema_version=JOURNEY_SPECIALIST_SCHEMA_VERSION,
            languages=tuple(languages),
            domains=tuple(domains),
            calibration=CalibrationMetadata(
                method=manifest.recipe.calibration_method,
                threshold=report.policy.abstention_threshold,
                calibration_dataset_digest=report.calibration_digest,
                holdout_dataset_digest=report.holdout_digest,
                seed=manifest.recipe.seed,
            ),
            quantization=QuantizationMetadata(
                mode=artifact_format,
                metric="macro_f1",
                observed_delta=float(report.quantized_delta["macro_f1_delta"]),
                maximum_delta=manifest.recipe.maximum_quantized_macro_f1_delta,
                evaluation_digest=report.digest,
            ),
            fallback=FallbackPolicy(mode="alias", alias=report.policy.fallback_alias),
            priority=priority,
        )
        return StoreResult.success(entry)
    except (TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "model_pack_entry_invalid")


def _classification_metrics(
    predictions: Sequence[SpecialistPrediction],
    labels: Sequence[str],
    *,
    quantized: bool,
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    correct = 0
    for label in labels:
        true_positive = false_positive = false_negative = support = 0
        for item in predictions:
            predicted = item.quantized_label if quantized else item.predicted_label
            if item.gold_label == label:
                support += 1
            if item.gold_label == label and predicted == label:
                true_positive += 1
            elif item.gold_label != label and predicted == label:
                false_positive += 1
            elif item.gold_label == label and predicted != label:
                false_negative += 1
        precision = _rate(true_positive, true_positive + false_positive)
        recall = _rate(true_positive, true_positive + false_negative)
        f1 = _f1(precision, recall)
        per_class[label] = {
            "f1": f1,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "precision": precision,
            "recall": recall,
            "support": support,
            "true_positive": true_positive,
        }
    for item in predictions:
        predicted = item.quantized_label if quantized else item.predicted_label
        correct += int(predicted == item.gold_label)
    return {
        "accuracy": _rate(correct, len(predictions)),
        "macro_f1": sum(value["f1"] for value in per_class.values()) / len(per_class),
        "per_class": per_class,
    }


def _abstention_metrics(
    predictions: Sequence[SpecialistPrediction], threshold: float
) -> dict[str, Any]:
    retained = tuple(item for item in predictions if item.confidence >= threshold)
    correct = sum(item.predicted_label == item.gold_label for item in retained)
    return {
        "abstained_count": len(predictions) - len(retained),
        "abstention_rate": 1.0 - _rate(len(retained), len(predictions)),
        "accuracy": _rate(correct, len(retained)),
        "coverage": _rate(len(retained), len(predictions)),
        "retained_count": len(retained),
        "threshold": threshold,
    }


def _subgroup_metrics(
    predictions: Sequence[SpecialistPrediction], labels: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    grouped: dict[str, list[SpecialistPrediction]] = defaultdict(list)
    for item in predictions:
        for axis, value in item.subgroups.items():
            grouped[f"{axis}:{value}"].append(item)
    result: dict[str, Mapping[str, Any]] = {}
    for key, items in sorted(grouped.items()):
        metrics = _classification_metrics(items, labels, quantized=False)
        supported_recalls = [
            value["recall"]
            for value in metrics["per_class"].values()
            if value["support"] > 0
        ]
        result[key] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "minimum_supported_recall": min(supported_recalls),
            "sample_count": len(items),
        }
    return result


def _quantized_delta(
    full: Mapping[str, Any],
    quantized: Mapping[str, Any],
    predictions: Sequence[SpecialistPrediction],
) -> dict[str, Any]:
    full_per_class = _mapping(full["per_class"], "full per_class")
    quant_per_class = _mapping(quantized["per_class"], "quantized per_class")
    recall_delta = {
        label: max(
            float(_mapping(value, "class metrics")["recall"])
            - float(_mapping(quant_per_class[label], "class metrics")["recall"]),
            0.0,
        )
        for label, value in full_per_class.items()
    }
    changes = sum(item.predicted_label != item.quantized_label for item in predictions)
    return {
        "macro_f1_delta": max(
            float(full["macro_f1"]) - float(quantized["macro_f1"]), 0.0
        ),
        "maximum_per_class_recall_delta": max(recall_delta.values(), default=0.0),
        "per_class_recall_delta": recall_delta,
        "prediction_change_rate": _rate(changes, len(predictions)),
    }


def _failure_slices(
    predictions: Sequence[SpecialistPrediction], threshold: float
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in predictions:
        if item.confidence < threshold:
            counts[f"abstained/{item.gold_label}"] += 1
        if item.predicted_label != item.gold_label:
            counts[f"confusion/{item.gold_label}/{item.predicted_label}"] += 1
        if item.quantized_label != item.predicted_label:
            counts[
                f"quantized_change/{item.predicted_label}/{item.quantized_label}"
            ] += 1
    return dict(sorted(counts.items()))


def _promotion_reasons(
    recipe: SpecialistRecipe,
    policy: SpecialistPromotionPolicy,
    metrics: Mapping[str, Any],
    calibration: Mapping[str, Any],
    abstention: Mapping[str, Any],
    subgroups: Mapping[str, Mapping[str, Any]],
    quantized_delta: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    macro_f1 = float(metrics["macro_f1"])
    if macro_f1 < recipe.minimum_macro_f1:
        reasons.append("macro_f1_below_floor")
    if macro_f1 < policy.baseline_macro_f1 + recipe.minimum_candidate_improvement:
        reasons.append("baseline_not_beaten")
    if (
        min(
            float(value["recall"])
            for value in _mapping(metrics["per_class"], "per_class").values()
        )
        < recipe.minimum_per_class_recall
    ):
        reasons.append("per_class_recall_below_floor")
    if float(calibration["expected_calibration_error"]) > recipe.maximum_ece:
        reasons.append("calibration_above_ceiling")
    if float(abstention["coverage"]) < recipe.minimum_coverage:
        reasons.append("coverage_below_floor")
    if (
        subgroups
        and min(
            float(value["minimum_supported_recall"]) for value in subgroups.values()
        )
        < recipe.minimum_subgroup_recall
    ):
        reasons.append("subgroup_recall_below_floor")
    if (
        float(quantized_delta["macro_f1_delta"])
        > recipe.maximum_quantized_macro_f1_delta
    ):
        reasons.append("quantized_delta_above_ceiling")
    return sorted(set(reasons))


def _argmax(scores: Mapping[str, float]) -> str:
    return min(scores, key=lambda label: (-scores[label], label))


def _probabilities(value: Mapping[str, Any], name: str) -> Mapping[str, float]:
    data = _mapping(value, name)
    if len(data) < 2:
        raise JourneySpecialistEvaluationError(f"{name} requires at least two labels")
    result: dict[str, float] = {}
    for label, score in data.items():
        _controlled(label, name)
        result[label] = _unit_interval(score, name)
    if abs(sum(result.values()) - 1.0) > 1e-6:
        raise JourneySpecialistEvaluationError(f"{name} must sum to one")
    return MappingProxyType(dict(sorted(result.items())))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JourneySpecialistEvaluationError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise JourneySpecialistEvaluationError(f"{name} keys must be strings")
    return value


def _controlled(value: Any, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise JourneySpecialistEvaluationError(f"{name} must be controlled")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JourneySpecialistEvaluationError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise JourneySpecialistEvaluationError(f"{name} must be finite")
    return number


def _unit_interval(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0 or number > 1:
        raise JourneySpecialistEvaluationError(f"{name} must be between zero and one")
    return number


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise JourneySpecialistEvaluationError(f"{name} must be a sha256 digest")
    return value


def _digest_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, str]:
    data = _mapping(value, name)
    return MappingProxyType(
        {
            _controlled(key, name): _digest(item, name)
            for key, item in sorted(data.items())
        }
    )


def _freeze_json_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        plain = _plain(_mapping(value, name))
    except (TypeError, ValueError) as exc:
        raise JourneySpecialistEvaluationError(f"{name} must be JSON-safe") from exc
    frozen = _freeze_json_value(plain)
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_nested_json_mapping(
    value: Mapping[str, Mapping[str, Any]], name: str
) -> Mapping[str, Mapping[str, Any]]:
    data = _mapping(value, name)
    return MappingProxyType(
        {
            _controlled(key, name): _freeze_json_mapping(_mapping(item, name), name)
            for key, item in sorted(data.items())
        }
    )


def _count_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, int]:
    data = _mapping(value, name)
    counts: dict[str, int] = {}
    for key, item in data.items():
        _controlled(key, name)
        if type(item) is not int or item < 0:
            raise JourneySpecialistEvaluationError(f"{name} values must be counts")
        counts[key] = item
    return MappingProxyType(dict(sorted(counts.items())))


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _contract_version(schema_version: str, compatibility_policy: str) -> None:
    if schema_version != JOURNEY_SPECIALIST_SCHEMA_VERSION:
        raise JourneySpecialistEvaluationError(
            "specialist schema version is unsupported"
        )
    if compatibility_policy != JOURNEY_SPECIALIST_COMPATIBILITY_POLICY:
        raise JourneySpecialistEvaluationError(
            "specialist compatibility policy is unsupported"
        )


__all__ = [
    "PROMOTION_DECISIONS",
    "JourneySpecialistEvaluationError",
    "SpecialistHoldoutReport",
    "SpecialistPrediction",
    "SpecialistPromotionPolicy",
    "SpecialistRunCompletion",
    "build_journey_specialist_model_pack_entry",
    "evaluate_journey_specialist_holdout",
    "finalize_journey_specialist_run",
    "render_journey_specialist_model_card",
]
