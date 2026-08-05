"""Reproducible Mode-A execution evidence for DirectID tiny candidates.

This module owns the bounded handoff from an already assembled DirectID dataset
to a locally produced floating-point checkpoint. Dataset rows and checkpoint
paths stay caller-local; the emitted artifacts contain only aggregate metrics,
stable references, and hashes for downstream quantization and gate evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.core.repro_hash import (
    build_training_provenance,
    compute_canonical_payload_hash,
    compute_environment_lock_digest,
    resolve_git_sha,
    verify_reproducibility,
)
from openmed.training.directid import (
    DIRECTID_STRUCTURED_ID_LABELS,
    DIRECTID_TINY_HEAD_CONTRACT,
    DirectIDHeadContract,
    validate_directid_contract,
    validate_directid_preset,
)
from openmed.training.distill import (
    ModeADistillationPipeline,
    build_distillation_report,
)
from openmed.training.hard_negatives import HARD_NEGATIVE_CATEGORIES
from openmed.training.recipe import TrainingRecipeConfig, config_hash, load_preset

DIRECTID_DISTILLATION_SCHEMA_VERSION = "openmed.training.directid_distillation.v1"
DIRECTID_TRAINING_REPORT_SCHEMA_VERSION = "openmed.training.directid_training_report.v1"
DIRECTID_RUN_MANIFEST_SCHEMA_VERSION = "openmed.training.directid_run_manifest.v1"
DIRECTID_CANDIDATE_SCHEMA_VERSION = "openmed.training.directid_candidate.v1"
DIRECTID_MODE_A_PIPELINE_REF = "openmed.training.distill:ModeADistillationPipeline@v1"
DIRECTID_CANDIDATE_REPO_ID = "OpenMed/OpenMed-PII-DirectID-Tiny"
DIRECTID_CANDIDATE_FORMAT = "pytorch-fp32"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_PHI_SHAPED_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{10,}\b"),
)
_FLOATING_REVISIONS = frozenset({"latest", "main", "master", "head"})
_DIRECTID_DATASET_EVIDENCE_SCHEMA_VERSION = (
    "openmed.training.directid_dataset_evidence.v1"
)
_DIRECTID_SPLITS = ("train", "validation", "test")


class DirectIDDistillationError(ValueError):
    """Raised when a DirectID training run cannot emit safe evidence."""


@dataclass(frozen=True)
class DirectIDLabelOutcome:
    """Aggregate model-only counts for one DirectID label."""

    label: str
    true_positive: int
    false_negative: int
    false_positive: int

    @property
    def gold_count(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def predicted_count(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def recall(self) -> float:
        return self.true_positive / self.gold_count

    @property
    def precision(self) -> float:
        if self.predicted_count == 0:
            return 0.0
        return self.true_positive / self.predicted_count

    def to_dict(self, *, critical: bool) -> dict[str, Any]:
        """Return deterministic per-label metric evidence."""

        return {
            "critical": critical,
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "gold_count": self.gold_count,
            "label": self.label,
            "precision": self.precision,
            "predicted_count": self.predicted_count,
            "recall": self.recall,
            "true_positive": self.true_positive,
        }


@dataclass(frozen=True)
class DirectIDHardNegativeOutcome:
    """Aggregate false-positive counts for one hard-negative category."""

    category: str
    example_count: int
    false_positive_count: int

    @property
    def false_positive_rate(self) -> float:
        return self.false_positive_count / self.example_count

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic hard-negative metric evidence."""

        return {
            "category": self.category,
            "example_count": self.example_count,
            "false_positive_count": self.false_positive_count,
            "false_positive_rate": self.false_positive_rate,
        }


@dataclass(frozen=True)
class DirectIDRunContext:
    """Validated, non-sensitive context passed to a local Mode-A trainer."""

    run_id: str
    recipe: TrainingRecipeConfig
    contract: DirectIDHeadContract
    rng_seeds: Mapping[str, int]
    preset_config_hash: str
    recipe_config_hash: str
    dataset_manifest_hash: str
    dataset_split_hashes: Mapping[str, str]
    temperature: float = 2.0
    alpha: float = 0.5
    span_loss_weight: float = 1.0
    mode_a_pipeline_ref: str = DIRECTID_MODE_A_PIPELINE_REF
    pipeline_type: type[ModeADistillationPipeline] = ModeADistillationPipeline


@dataclass(frozen=True)
class DirectIDTrainingOutput:
    """Aggregate output returned by a caller-supplied local training backend."""

    checkpoint_path: str | Path
    teacher_recall_by_label: Mapping[str, float]
    label_outcomes: Sequence[DirectIDLabelOutcome]
    hard_negative_outcomes: Sequence[DirectIDHardNegativeOutcome]
    training_steps: int
    completed_epochs: float
    final_loss: float


@dataclass(frozen=True)
class DirectIDCandidateCheckpoint:
    """Path-free reference to one floating-point DirectID candidate."""

    run_id: str
    checkpoint_ref: str
    artifact_hash: str
    reproducibility_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a quantization-ready candidate reference."""

        return {
            "artifact_hash": self.artifact_hash,
            "certified": False,
            "checkpoint_ref": self.checkpoint_ref,
            "family": DIRECTID_TINY_HEAD_CONTRACT.family,
            "format": DIRECTID_CANDIDATE_FORMAT,
            "published": False,
            "ready_for_quantization": True,
            "repo_id": DIRECTID_CANDIDATE_REPO_ID,
            "reproducibility_hash": self.reproducibility_hash,
            "run_id": self.run_id,
            "schema_version": DIRECTID_CANDIDATE_SCHEMA_VERSION,
            "tier": DIRECTID_TINY_HEAD_CONTRACT.tier,
        }


@dataclass(frozen=True)
class DirectIDRunArtifacts:
    """Files and in-memory evidence produced by one DirectID run."""

    checkpoint_path: Path
    candidate: DirectIDCandidateCheckpoint
    training_report: Mapping[str, Any]
    run_manifest: Mapping[str, Any]
    training_provenance: Mapping[str, Any]
    candidate_path: Path
    training_report_path: Path
    run_manifest_path: Path
    training_provenance_path: Path


DirectIDTrainer = Callable[[DirectIDRunContext], DirectIDTrainingOutput]


def run_directid_tiny_distillation(
    *,
    dataset_evidence: Mapping[str, Any],
    teacher_id: str,
    teacher_revision: str,
    trainer: DirectIDTrainer,
    output_dir: str | Path,
    recipe: TrainingRecipeConfig | None = None,
    contract: DirectIDHeadContract = DIRECTID_TINY_HEAD_CONTRACT,
    git_sha: str | None = None,
    env_lock_path: str | Path | None = None,
) -> DirectIDRunArtifacts:
    """Execute a local DirectID Mode-A backend and emit hash-only evidence.

    Args:
        dataset_evidence: Aggregate evidence produced by
            ``build_directid_dataset_evidence``. Raw records are not accepted.
        teacher_id: Stable local or registry identifier for the teacher model.
        teacher_revision: Pinned, non-floating teacher revision.
        trainer: Local backend that consumes the validated run context and
            returns aggregate counts plus a checkpoint path.
        output_dir: Directory in which to write evidence JSON files. Existing
            files are accepted only when byte-identical.
        recipe: Optional DirectID-compatible Mode-A recipe override.
        contract: DirectID head contract to enforce.
        git_sha: Optional 40- or 64-character source revision override.
        env_lock_path: Optional environment lock file. Defaults to ``uv.lock``
            at the repository root.

    Returns:
        Paths and parsed evidence for the candidate training run.

    Raises:
        DirectIDDistillationError: If inputs, backend output, checkpoint, or
            serialized evidence violate the DirectID run contract.
    """

    active_contract = validate_directid_contract(contract)
    active_recipe = recipe or load_preset("tiny_distill")
    validate_directid_preset(active_recipe, contract=active_contract)
    dataset_summary = _validate_dataset_evidence(
        dataset_evidence,
        contract=active_contract,
        expected_seed=active_recipe.seed,
    )
    safe_teacher_id = _safe_identifier(teacher_id, "teacher_id")
    safe_teacher_revision = _pinned_revision(teacher_revision, "teacher_revision")
    safe_student_backbone = _safe_identifier(
        active_recipe.backbone.model_ref,
        "student_backbone",
    )
    safe_student_revision = _pinned_revision(
        active_recipe.backbone.revision,
        "student_revision",
    )
    resolved_git_sha = git_sha or resolve_git_sha()
    if _GIT_SHA_RE.fullmatch(resolved_git_sha) is None:
        raise DirectIDDistillationError("git_sha must be a 40- or 64-hex revision")

    lock_path = (
        Path(env_lock_path)
        if env_lock_path is not None
        else Path(__file__).resolve().parents[2] / "uv.lock"
    )
    if not lock_path.is_file():
        raise DirectIDDistillationError("environment lock file does not exist")
    lock_digest = compute_environment_lock_digest(lock_path)
    preset_hash = config_hash(active_recipe)
    execution_hash = compute_canonical_payload_hash(
        {
            "contract_ref": active_contract.contract_ref,
            "dataset_manifest_hash": dataset_summary["manifest_hash"],
            "mode_a": {
                "alpha": 0.5,
                "pipeline_ref": DIRECTID_MODE_A_PIPELINE_REF,
                "span_loss_weight": 1.0,
                "temperature": 2.0,
            },
            "preset_config_hash": preset_hash,
            "schema_version": DIRECTID_DISTILLATION_SCHEMA_VERSION,
            "teacher": {
                "id": safe_teacher_id,
                "revision": safe_teacher_revision,
            },
        }
    )
    seeds = {
        "numpy": active_recipe.seed,
        "python": active_recipe.seed,
        "torch": active_recipe.seed,
    }
    initial_provenance = build_training_provenance(
        rng_seeds=seeds,
        data_manifest_hash=dataset_summary["manifest_hash"],
        recipe_config_hash=execution_hash,
        env_lock_digest=lock_digest,
        base_model=safe_student_backbone,
        base_model_revision=safe_student_revision,
        git_sha=resolved_git_sha,
        repo_id=DIRECTID_CANDIDATE_REPO_ID,
    )
    verify_reproducibility(initial_provenance)
    run_id = (
        "directid-tiny-"
        + initial_provenance["reproducibility_hash"].split(":", 1)[1][:16]
    )
    context = DirectIDRunContext(
        run_id=run_id,
        recipe=active_recipe,
        contract=active_contract,
        rng_seeds=seeds,
        preset_config_hash=preset_hash,
        recipe_config_hash=execution_hash,
        dataset_manifest_hash=dataset_summary["manifest_hash"],
        dataset_split_hashes=dataset_summary["split_hashes"],
    )

    _seed_local_rngs(active_recipe.seed)
    result = trainer(context)
    if not isinstance(result, DirectIDTrainingOutput):
        raise DirectIDDistillationError(
            "trainer must return DirectIDTrainingOutput aggregate evidence"
        )
    label_outcomes = _validate_label_outcomes(
        result.label_outcomes,
        contract=active_contract,
    )
    hard_negative_outcomes = _validate_hard_negative_outcomes(
        result.hard_negative_outcomes
    )
    teacher_recall = _validate_teacher_recall(
        result.teacher_recall_by_label,
        contract=active_contract,
    )
    _validate_training_progress(result)
    checkpoint_path = _validated_checkpoint_path(result.checkpoint_path)
    artifact_hash = _checkpoint_artifact_hash(checkpoint_path)
    checkpoint_ref = (
        "openmed://candidate/OpenMed-PII-DirectID-Tiny/"
        + artifact_hash.split(":", 1)[1]
    )
    provenance = build_training_provenance(
        rng_seeds=seeds,
        data_manifest_hash=dataset_summary["manifest_hash"],
        recipe_config_hash=execution_hash,
        env_lock_digest=lock_digest,
        base_model=safe_student_backbone,
        base_model_revision=safe_student_revision,
        git_sha=resolved_git_sha,
        repo_id=DIRECTID_CANDIDATE_REPO_ID,
        checkpoint_id=checkpoint_ref,
    )
    verify_reproducibility(provenance)
    candidate = DirectIDCandidateCheckpoint(
        run_id=run_id,
        checkpoint_ref=checkpoint_ref,
        artifact_hash=artifact_hash,
        reproducibility_hash=provenance["reproducibility_hash"],
    )

    student_recall = {outcome.label: outcome.recall for outcome in label_outcomes}
    distillation_report = build_distillation_report(
        teacher_id=safe_teacher_id,
        student_backbone=safe_student_backbone,
        temperature=context.temperature,
        alpha=context.alpha,
        teacher_recall_by_label=teacher_recall,
        student_recall_by_label=student_recall,
        critical_labels=active_contract.critical_labels,
    )
    training_report = _build_training_report(
        context=context,
        result=result,
        label_outcomes=label_outcomes,
        hard_negative_outcomes=hard_negative_outcomes,
        candidate=candidate,
        distillation_report=distillation_report.to_dict(),
    )
    report_hash = compute_canonical_payload_hash(training_report)
    run_manifest = _build_run_manifest(
        context=context,
        dataset_summary=dataset_summary,
        teacher_id=safe_teacher_id,
        teacher_revision=safe_teacher_revision,
        student_backbone=safe_student_backbone,
        student_revision=safe_student_revision,
        candidate=candidate,
        report_hash=report_hash,
        provenance=provenance,
    )

    destination = Path(output_dir)
    resolved_destination = destination.resolve()
    if (
        resolved_destination == checkpoint_path
        or resolved_destination in checkpoint_path.parents
        or checkpoint_path in resolved_destination.parents
    ):
        raise DirectIDDistillationError(
            "output_dir must be separate from the checkpoint tree"
        )
    candidate_path = _write_json(
        destination / "candidate_checkpoint.json", candidate.to_dict()
    )
    training_report_path = _write_json(
        destination / "training_report.json", training_report
    )
    run_manifest_path = _write_json(destination / "run_manifest.json", run_manifest)
    training_provenance_path = _write_json(
        destination / "training_provenance.json", provenance
    )
    return DirectIDRunArtifacts(
        checkpoint_path=checkpoint_path,
        candidate=candidate,
        training_report=training_report,
        run_manifest=run_manifest,
        training_provenance=provenance,
        candidate_path=candidate_path,
        training_report_path=training_report_path,
        run_manifest_path=run_manifest_path,
        training_provenance_path=training_provenance_path,
    )


def _validate_dataset_evidence(
    evidence: Mapping[str, Any],
    *,
    contract: DirectIDHeadContract,
    expected_seed: int,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise DirectIDDistillationError("dataset_evidence must be a mapping")
    if evidence.get("schema_version") != _DIRECTID_DATASET_EVIDENCE_SCHEMA_VERSION:
        raise DirectIDDistillationError("unsupported DirectID dataset evidence schema")
    if evidence.get("contract_ref") != contract.contract_ref:
        raise DirectIDDistillationError("dataset evidence contract_ref mismatch")
    if (
        evidence.get("family") != contract.family
        or evidence.get("tier") != contract.tier
    ):
        raise DirectIDDistillationError("dataset evidence family or tier mismatch")
    if evidence.get("raw_records_persisted") is not False:
        raise DirectIDDistillationError(
            "dataset evidence must declare raw_records_persisted=false"
        )
    manifest_hash = _require_digest(evidence.get("manifest_hash"), "manifest_hash")
    manifest_id = _safe_identifier(evidence.get("manifest_id"), "manifest_id")

    splits = evidence.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(_DIRECTID_SPLITS):
        raise DirectIDDistillationError(
            "dataset evidence must contain train, validation, and test splits"
        )
    split_hashes: dict[str, str] = {}
    for split_name in _DIRECTID_SPLITS:
        split = splits[split_name]
        if not isinstance(split, Mapping):
            raise DirectIDDistillationError(f"{split_name} evidence must be a mapping")
        split_hashes[split_name] = _require_digest(
            split.get("dataset_hash"), f"{split_name}.dataset_hash"
        )
        _positive_count_mapping(
            split.get("label_counts"),
            required=contract.labels,
            field=f"{split_name}.label_counts",
        )
        _positive_count_mapping(
            split.get("id_subtype_counts"),
            required=contract.id_subtypes,
            field=f"{split_name}.id_subtype_counts",
        )
        _positive_count_mapping(
            split.get("hard_negative_category_counts"),
            required=HARD_NEGATIVE_CATEGORIES,
            field=f"{split_name}.hard_negative_category_counts",
        )

    generation = evidence.get("synthetic_generation")
    if not isinstance(generation, Mapping):
        raise DirectIDDistillationError("synthetic_generation evidence is required")
    settings_hash = _require_digest(
        generation.get("settings_hash"),
        "synthetic settings hash",
    )
    settings = generation.get("settings")
    if not isinstance(settings, Mapping):
        raise DirectIDDistillationError("synthetic generation settings are required")
    if settings.get("contains_real_phi") is not False:
        raise DirectIDDistillationError(
            "synthetic settings must declare contains_real_phi=false"
        )
    if settings.get("seed") != expected_seed:
        raise DirectIDDistillationError(
            "synthetic generation seed must match the Mode-A recipe seed"
        )
    if compute_canonical_payload_hash(settings) != settings_hash:
        raise DirectIDDistillationError(
            "synthetic generation settings hash does not match its payload"
        )
    return {
        "manifest_hash": manifest_hash,
        "manifest_id": manifest_id,
        "split_hashes": split_hashes,
        "synthetic_settings_hash": settings_hash,
    }


def _positive_count_mapping(
    value: Any,
    *,
    required: Sequence[str],
    field: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise DirectIDDistillationError(
            f"{field} must cover exactly the required DirectID values"
        )
    counts: dict[str, int] = {}
    for key in required:
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise DirectIDDistillationError(f"{field}.{key} must be positive")
        counts[key] = count
    return counts


def _validate_label_outcomes(
    outcomes: Sequence[DirectIDLabelOutcome],
    *,
    contract: DirectIDHeadContract,
) -> tuple[DirectIDLabelOutcome, ...]:
    by_label: dict[str, DirectIDLabelOutcome] = {}
    for outcome in outcomes:
        if not isinstance(outcome, DirectIDLabelOutcome):
            raise DirectIDDistillationError(
                "label_outcomes must contain DirectIDLabelOutcome values"
            )
        if outcome.label in by_label:
            raise DirectIDDistillationError(
                f"duplicate label outcome for {outcome.label}"
            )
        for field_name in ("true_positive", "false_negative", "false_positive"):
            value = getattr(outcome, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DirectIDDistillationError(
                    f"{outcome.label}.{field_name} must be a non-negative integer"
                )
        if outcome.gold_count <= 0:
            raise DirectIDDistillationError(
                f"{outcome.label} must have positive evaluation support"
            )
        by_label[outcome.label] = outcome
    if set(by_label) != set(contract.labels):
        raise DirectIDDistillationError(
            "label_outcomes must cover exactly every DirectID contract label"
        )
    return tuple(by_label[label] for label in contract.labels)


def _validate_hard_negative_outcomes(
    outcomes: Sequence[DirectIDHardNegativeOutcome],
) -> tuple[DirectIDHardNegativeOutcome, ...]:
    by_category: dict[str, DirectIDHardNegativeOutcome] = {}
    for outcome in outcomes:
        if not isinstance(outcome, DirectIDHardNegativeOutcome):
            raise DirectIDDistillationError(
                "hard_negative_outcomes must contain DirectIDHardNegativeOutcome values"
            )
        if outcome.category in by_category:
            raise DirectIDDistillationError(
                f"duplicate hard-negative outcome for {outcome.category}"
            )
        if (
            isinstance(outcome.example_count, bool)
            or not isinstance(outcome.example_count, int)
            or outcome.example_count <= 0
        ):
            raise DirectIDDistillationError(
                f"{outcome.category}.example_count must be positive"
            )
        if (
            isinstance(outcome.false_positive_count, bool)
            or not isinstance(outcome.false_positive_count, int)
            or outcome.false_positive_count < 0
            or outcome.false_positive_count > outcome.example_count
        ):
            raise DirectIDDistillationError(
                f"{outcome.category}.false_positive_count is invalid"
            )
        by_category[outcome.category] = outcome
    if set(by_category) != set(HARD_NEGATIVE_CATEGORIES):
        raise DirectIDDistillationError(
            "hard_negative_outcomes must cover every required category"
        )
    return tuple(by_category[category] for category in HARD_NEGATIVE_CATEGORIES)


def _validate_teacher_recall(
    recall: Mapping[str, float],
    *,
    contract: DirectIDHeadContract,
) -> dict[str, float]:
    if not isinstance(recall, Mapping) or set(recall) != set(contract.labels):
        raise DirectIDDistillationError(
            "teacher_recall_by_label must cover every DirectID contract label"
        )
    normalized: dict[str, float] = {}
    for label in contract.labels:
        value = recall[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DirectIDDistillationError(
                f"teacher recall for {label} must be numeric"
            )
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise DirectIDDistillationError(
                f"teacher recall for {label} must be within [0, 1]"
            )
        normalized[label] = parsed
    return normalized


def _validate_training_progress(result: DirectIDTrainingOutput) -> None:
    if (
        isinstance(result.training_steps, bool)
        or not isinstance(result.training_steps, int)
        or result.training_steps <= 0
    ):
        raise DirectIDDistillationError("training_steps must be a positive integer")
    if (
        not math.isfinite(float(result.completed_epochs))
        or result.completed_epochs <= 0
    ):
        raise DirectIDDistillationError("completed_epochs must be positive and finite")
    if not math.isfinite(float(result.final_loss)) or result.final_loss < 0:
        raise DirectIDDistillationError("final_loss must be non-negative and finite")


def _build_training_report(
    *,
    context: DirectIDRunContext,
    result: DirectIDTrainingOutput,
    label_outcomes: Sequence[DirectIDLabelOutcome],
    hard_negative_outcomes: Sequence[DirectIDHardNegativeOutcome],
    candidate: DirectIDCandidateCheckpoint,
    distillation_report: Mapping[str, Any],
) -> dict[str, Any]:
    critical = set(context.contract.critical_labels)
    label_rows = [
        outcome.to_dict(critical=outcome.label in critical)
        for outcome in label_outcomes
    ]
    hard_negative_rows = [outcome.to_dict() for outcome in hard_negative_outcomes]
    hard_negative_examples = sum(row["example_count"] for row in hard_negative_rows)
    hard_negative_false_positives = sum(
        row["false_positive_count"] for row in hard_negative_rows
    )
    structured = [
        outcome
        for outcome in label_outcomes
        if outcome.label in DIRECTID_STRUCTURED_ID_LABELS
    ]
    structured_gold = sum(outcome.gold_count for outcome in structured)
    structured_true_positive = sum(outcome.true_positive for outcome in structured)
    critical_outcomes = [
        outcome for outcome in label_outcomes if outcome.label in critical
    ]
    critical_gold = sum(outcome.gold_count for outcome in critical_outcomes)
    critical_misses = sum(outcome.false_negative for outcome in critical_outcomes)
    per_label_recall = {outcome.label: outcome.recall for outcome in label_outcomes}
    per_label_precision = {
        outcome.label: outcome.precision for outcome in label_outcomes
    }
    return {
        "candidate_checkpoint": candidate.to_dict(),
        "critical_label_recall": {
            outcome.label: per_label_recall[outcome.label]
            for outcome in critical_outcomes
        },
        "critical_label_recall_min": min(
            per_label_recall[outcome.label] for outcome in critical_outcomes
        ),
        "dataset_manifest_hash": context.dataset_manifest_hash,
        "distillation": dict(distillation_report),
        "eval_set_hash": context.dataset_split_hashes["test"],
        "evaluation_split": "test",
        "family": context.contract.family,
        "hard_negative_false_positive_rate": (
            hard_negative_false_positives / hard_negative_examples
        ),
        "hard_negative_metrics": {
            "false_positive_count": hard_negative_false_positives,
            "false_positive_rate": (
                hard_negative_false_positives / hard_negative_examples
            ),
            "per_category": hard_negative_rows,
            "total_examples": hard_negative_examples,
        },
        "mode": context.recipe.mode,
        "model_only_critical_leakage_count": critical_misses,
        "model_only_residual_leakage_rate": critical_misses / critical_gold,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
        "per_label_metrics": label_rows,
        "raw_phi_persisted": False,
        "preset_config_hash": context.preset_config_hash,
        "recipe_config_hash": context.recipe_config_hash,
        "restricted_dataset_payloads_persisted": False,
        "run_id": context.run_id,
        "schema_version": DIRECTID_TRAINING_REPORT_SCHEMA_VERSION,
        "structured_id_recall": structured_true_positive / structured_gold,
        "tier": context.contract.tier,
        "training": {
            "completed_epochs": float(result.completed_epochs),
            "final_loss": float(result.final_loss),
            "steps": result.training_steps,
        },
    }


def _build_run_manifest(
    *,
    context: DirectIDRunContext,
    dataset_summary: Mapping[str, Any],
    teacher_id: str,
    teacher_revision: str,
    student_backbone: str,
    student_revision: str,
    candidate: DirectIDCandidateCheckpoint,
    report_hash: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_files": {
            "candidate_checkpoint": "candidate_checkpoint.json",
            "training_provenance": "training_provenance.json",
            "training_report": "training_report.json",
        },
        "candidate_checkpoint": candidate.to_dict(),
        "contract_ref": context.contract.contract_ref,
        "dataset": {
            "manifest_hash": dataset_summary["manifest_hash"],
            "manifest_id": dataset_summary["manifest_id"],
            "split_hashes": dict(dataset_summary["split_hashes"]),
            "synthetic_settings_hash": dataset_summary["synthetic_settings_hash"],
        },
        "family": context.contract.family,
        "final_certification_performed": False,
        "mode_a": {
            "alpha": context.alpha,
            "span_loss_weight": context.span_loss_weight,
            "temperature": context.temperature,
        },
        "mode": context.recipe.mode,
        "mode_a_pipeline_ref": context.mode_a_pipeline_ref,
        "preset_name": context.recipe.preset_name,
        "publishing_performed": False,
        "raw_phi_persisted": False,
        "ready_for_gate_evaluation": True,
        "ready_for_quantization": True,
        "preset_config_hash": context.preset_config_hash,
        "recipe_config_hash": context.recipe_config_hash,
        "restricted_dataset_payloads_persisted": False,
        "rng_seeds": dict(context.rng_seeds),
        "run_id": context.run_id,
        "schema_version": DIRECTID_RUN_MANIFEST_SCHEMA_VERSION,
        "student": {
            "backbone": student_backbone,
            "revision": student_revision,
        },
        "teacher": {"id": teacher_id, "revision": teacher_revision},
        "tier": context.contract.tier,
        "training_provenance": dict(provenance),
        "training_report_hash": report_hash,
    }


def _seed_local_rngs(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy
    except ImportError:
        pass
    else:
        numpy.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _validated_checkpoint_path(value: str | Path) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise DirectIDDistillationError("trainer checkpoint path must not be a symlink")
    path = unresolved.resolve()
    if not path.exists():
        raise DirectIDDistillationError("trainer checkpoint path does not exist")
    descendants = list(path.rglob("*")) if path.is_dir() else []
    if any(item.is_symlink() for item in descendants):
        raise DirectIDDistillationError("trainer checkpoint must not contain symlinks")
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in descendants if item.is_file())
    )
    if not files:
        raise DirectIDDistillationError("trainer checkpoint contains no files")
    return path


def _checkpoint_artifact_hash(path: Path) -> str:
    root = path if path.is_dir() else path.parent
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    entries = []
    for file_path in files:
        entries.append(
            {
                "path": file_path.relative_to(root).as_posix(),
                "sha256": _file_hash(file_path),
            }
        )
    return compute_canonical_payload_hash(entries)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DirectIDDistillationError(f"{field} must be a non-empty string")
    result = value.strip()
    if any(pattern.search(result) for pattern in _PHI_SHAPED_PATTERNS):
        raise DirectIDDistillationError(f"{field} contains a PHI-shaped value")
    return result


def _pinned_revision(value: Any, field: str) -> str:
    revision = _safe_identifier(value, field)
    if _REVISION_RE.fullmatch(revision) is None:
        raise DirectIDDistillationError(f"{field} has an invalid format")
    if revision.lower() in _FLOATING_REVISIONS:
        raise DirectIDDistillationError(f"{field} must be pinned")
    return revision


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DirectIDDistillationError(f"{field} must be a sha256 digest")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != serialized:
            raise DirectIDDistillationError(
                f"refusing to overwrite non-matching evidence file {path.name}"
            )
        return path
    path.write_text(serialized, encoding="utf-8")
    return path


__all__ = [
    "DIRECTID_CANDIDATE_FORMAT",
    "DIRECTID_CANDIDATE_REPO_ID",
    "DIRECTID_CANDIDATE_SCHEMA_VERSION",
    "DIRECTID_DISTILLATION_SCHEMA_VERSION",
    "DIRECTID_MODE_A_PIPELINE_REF",
    "DIRECTID_RUN_MANIFEST_SCHEMA_VERSION",
    "DIRECTID_TRAINING_REPORT_SCHEMA_VERSION",
    "DirectIDCandidateCheckpoint",
    "DirectIDDistillationError",
    "DirectIDHardNegativeOutcome",
    "DirectIDLabelOutcome",
    "DirectIDRunArtifacts",
    "DirectIDRunContext",
    "DirectIDTrainer",
    "DirectIDTrainingOutput",
    "run_directid_tiny_distillation",
]
