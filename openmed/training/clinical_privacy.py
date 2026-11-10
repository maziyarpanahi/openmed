"""Recall-first training contract for the clinical PHI flagship checkpoint.

The module resolves the dataset assembly manifest into immutable, metadata-only
input bindings and produces a checkpoint manifest after an external mode-C
training run. It never loads corpus rows or accepts credentialed dataset paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.manifest_schema import validate_manifest_row
from openmed.core.repro_hash import (
    build_training_provenance,
    compute_environment_lock_digest,
    compute_file_digest,
    verify_reproducibility,
)
from openmed.eval.datasets.clinical_phi import (
    CLINICAL_PHI_MANIFEST_ID,
    CLINICAL_PHI_MANIFEST_REF,
    CLINICAL_PRIVACY_MODEL_ID,
    ClinicalPHIDatasetManifest,
    ClinicalPHISource,
    clinical_phi_manifest_hash,
    load_clinical_phi_manifest,
    validate_clinical_phi_manifest,
)
from openmed.training.recipe import (
    CONFIG_DIR,
    TrainingRecipeConfig,
    load_preset,
)

CLINICAL_PRIVACY_TRAINING_SCHEMA_VERSION = "openmed.training.clinical_privacy.v1"
CLINICAL_PRIVACY_CHECKPOINT_SCHEMA_VERSION = (
    "openmed.training.clinical_privacy_checkpoint.v1"
)
CLINICAL_PRIVACY_CHECKPOINT_NAME = "OpenMed-ClinicalPrivacy-tier0"
CLINICAL_PRIVACY_FAMILY = "ClinicalPrivacy"
CLINICAL_PRIVACY_OBJECTIVE = "recall_first"
CLINICAL_PRIVACY_CONTRACT_REF = (
    "openmed.training.clinical_privacy:CLINICAL_PRIVACY_TIER0_CONTRACT@v1"
)
CLINICAL_PRIVACY_RECIPE_REF = "openmed/training/configs/large_teacher.yaml"
CLINICAL_PRIVACY_TRAINING_SOURCE_IDS = ("synthetic_golden_deid",)

_LARGE_TEACHER_CONFIG_PATH = CONFIG_DIR / "large_teacher.yaml"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,127}$")
_FORBIDDEN_CONTENT_KEYS = frozenset(
    {
        "content",
        "credentialed_path",
        "credentialed_paths",
        "fixture",
        "fixtures",
        "note_text",
        "passage",
        "raw_phi",
        "raw_text",
        "record",
        "records",
        "row",
        "rows",
        "text",
    }
)


class ClinicalPrivacyTrainingError(ValueError):
    """Raised when mode-C inputs or checkpoint metadata are incompatible."""


@dataclass(frozen=True)
class ClinicalPrivacyTier0Contract:
    """Stable identity and execution boundary for the flagship training run."""

    schema_version: str
    contract_ref: str
    checkpoint_name: str
    model_id: str
    family: str
    objective: str
    recipe_mode: str
    training_source_ids: tuple[str, ...]


CLINICAL_PRIVACY_TIER0_CONTRACT = ClinicalPrivacyTier0Contract(
    schema_version=CLINICAL_PRIVACY_TRAINING_SCHEMA_VERSION,
    contract_ref=CLINICAL_PRIVACY_CONTRACT_REF,
    checkpoint_name=CLINICAL_PRIVACY_CHECKPOINT_NAME,
    model_id=CLINICAL_PRIVACY_MODEL_ID,
    family=CLINICAL_PRIVACY_FAMILY,
    objective=CLINICAL_PRIVACY_OBJECTIVE,
    recipe_mode="C",
    training_source_ids=CLINICAL_PRIVACY_TRAINING_SOURCE_IDS,
)


@dataclass(frozen=True)
class ClinicalPrivacySourceBinding:
    """One revision-pinned source reference resolved without loading rows."""

    source_id: str
    dataset: str
    revision: str
    role: str
    access: str
    loader_ref: str
    license_id: str
    labels: tuple[str, ...]
    split: str
    source_url: str
    selected_for_training: bool
    requires_credentials: bool
    eval_only: bool
    synthetic: bool
    content_included: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return PHI-free source provenance for the training manifest."""

        return {
            "access": self.access,
            "content_included": self.content_included,
            "dataset": self.dataset,
            "eval_only": self.eval_only,
            "labels": list(self.labels),
            "license_id": self.license_id,
            "loader_ref": self.loader_ref,
            "requires_credentials": self.requires_credentials,
            "revision": self.revision,
            "role": self.role,
            "selected_for_training": self.selected_for_training,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "split": self.split,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class ClinicalPrivacyGateThreshold:
    """One manifest-derived release threshold recorded with the checkpoint."""

    gate: str
    metric: str
    comparator: str
    value: float
    labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable threshold representation."""

        return {
            "comparator": self.comparator,
            "gate": self.gate,
            "labels": list(self.labels),
            "metric": self.metric,
            "value": self.value,
        }


@dataclass(frozen=True)
class ClinicalPrivacyTrainingPlan:
    """Resolved mode-C configuration and metadata-only dataset bindings."""

    model_id: str
    manifest_id: str
    manifest_ref: str
    manifest_hash: str
    recipe: TrainingRecipeConfig
    recipe_config_hash: str
    label_weights: Mapping[str, float]
    sources: tuple[ClinicalPrivacySourceBinding, ...]
    thresholds: tuple[ClinicalPrivacyGateThreshold, ...]
    rng_seeds: Mapping[str, int]
    source_revisions_hash: str
    schema_version: str = CLINICAL_PRIVACY_TRAINING_SCHEMA_VERSION

    def training_sources(self) -> tuple[ClinicalPrivacySourceBinding, ...]:
        """Return only sources permitted to contribute training examples."""

        return tuple(source for source in self.sources if source.selected_for_training)

    def source_revisions(self) -> dict[str, str]:
        """Return source revisions keyed by manifest source id."""

        return {source.source_id: source.revision for source in self.sources}

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical PHI-free training plan."""

        return {
            "dataset_manifest": {
                "hash": self.manifest_hash,
                "id": self.manifest_id,
                "ref": self.manifest_ref,
            },
            "family": CLINICAL_PRIVACY_FAMILY,
            "label_weights": {
                "by_label": {
                    label: float(weight)
                    for label, weight in sorted(self.label_weights.items())
                },
                "default": 1.0,
            },
            "model_id": self.model_id,
            "objective": CLINICAL_PRIVACY_OBJECTIVE,
            "privacy": {
                "contains_gated_evaluation_content": False,
                "contains_raw_phi": False,
                "metadata_only_resolution": True,
            },
            "recipe": {
                "config": self.recipe.to_dict(),
                "config_hash": self.recipe_config_hash,
                "config_ref": CLINICAL_PRIVACY_RECIPE_REF,
                "mode": self.recipe.mode,
                "objective": CLINICAL_PRIVACY_OBJECTIVE,
                "preset_name": self.recipe.preset_name,
            },
            "rng_seeds": dict(sorted(self.rng_seeds.items())),
            "schema_version": self.schema_version,
            "source_revisions_hash": self.source_revisions_hash,
            "sources": [source.to_dict() for source in self.sources],
            "thresholds": [threshold.to_dict() for threshold in self.thresholds],
        }


@dataclass(frozen=True)
class ClinicalPrivacyCheckpointManifest:
    """Checkpoint row plus mode-C recipe and reproducibility metadata."""

    training_plan: ClinicalPrivacyTrainingPlan
    checkpoint_artifact_hash: str
    checkpoint_row: Mapping[str, Any]
    training_provenance: Mapping[str, Any]
    checkpoint_manifest_hash: str
    schema_version: str = CLINICAL_PRIVACY_CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a benchmark-compatible checkpoint manifest payload."""

        payload = self._payload_without_hash()
        payload["checkpoint_manifest_hash"] = self.checkpoint_manifest_hash
        return payload

    def _payload_without_hash(self) -> dict[str, Any]:
        plan = self.training_plan.to_dict()
        return {
            "checkpoint_artifact_hash": self.checkpoint_artifact_hash,
            "checkpoint_name": CLINICAL_PRIVACY_CHECKPOINT_NAME,
            "checkpoints": [dict(self.checkpoint_row)],
            "dataset_manifest": plan["dataset_manifest"],
            "model_id": self.training_plan.model_id,
            "privacy": plan["privacy"],
            "reproducibility": {
                "source_revisions_hash": self.training_plan.source_revisions_hash,
                "training_provenance": dict(self.training_provenance),
            },
            "schema_version": self.schema_version,
            "source_revisions": self.training_plan.source_revisions(),
            "thresholds": plan["thresholds"],
            "training_recipe": plan["recipe"],
        }


def resolve_clinical_privacy_training_plan(
    source_revisions: Mapping[str, str],
    *,
    recipe: TrainingRecipeConfig | None = None,
    manifest: ClinicalPHIDatasetManifest | None = None,
) -> ClinicalPrivacyTrainingPlan:
    """Resolve all clinical-PHI manifest sources into immutable references.

    Corpus rows are deliberately not loaded. Every source, including public
    comparison and DUA-held-out sources, must have an explicit revision so the
    resulting checkpoint record can state exactly which assembly contract was
    used. Only the manifest's synthetic training source is selected for mode C.

    Args:
        source_revisions: Revision or version identifier for every manifest source.
        recipe: Optional mode-C recipe override for validation tests.
        manifest: Optional dataset manifest override for validation tests.

    Returns:
        A validated metadata-only mode-C training plan.

    Raises:
        ClinicalPrivacyTrainingError: If source or recipe contracts are incomplete.
    """

    active_manifest = manifest or load_clinical_phi_manifest()
    validate_clinical_phi_manifest(active_manifest)
    active_recipe = recipe or load_preset("C")
    labels = _training_labels(active_manifest)
    _validate_mode_c_recipe(active_recipe, active_manifest, labels)

    expected_ids = tuple(source.source_id for source in active_manifest.sources)
    _validate_source_revisions(source_revisions, expected_ids)
    bindings = tuple(
        _source_binding(source, source_revisions[source.source_id])
        for source in active_manifest.sources
    )
    _validate_training_source_selection(bindings)

    label_weights = {
        label: active_recipe.loss.critical_label_weight for label in labels
    }
    thresholds = tuple(
        ClinicalPrivacyGateThreshold(
            gate=requirement.gate,
            metric=requirement.metric,
            comparator=requirement.comparator,
            value=requirement.threshold,
            labels=requirement.labels,
        )
        for requirement in active_manifest.gate_families
    )
    revisions = {source.source_id: source.revision for source in bindings}
    return ClinicalPrivacyTrainingPlan(
        model_id=active_manifest.model_id,
        manifest_id=active_manifest.manifest_id,
        manifest_ref=CLINICAL_PHI_MANIFEST_REF,
        manifest_hash=clinical_phi_manifest_hash(active_manifest),
        recipe=active_recipe,
        recipe_config_hash=compute_file_digest(_LARGE_TEACHER_CONFIG_PATH),
        label_weights=label_weights,
        sources=bindings,
        thresholds=thresholds,
        rng_seeds={
            "numpy": active_recipe.seed,
            "python": active_recipe.seed,
            "torch": active_recipe.seed,
        },
        source_revisions_hash=_canonical_hash(revisions),
    )


def build_clinical_privacy_checkpoint_manifest(
    source_revisions: Mapping[str, str],
    *,
    checkpoint_artifact_hash: str,
    git_sha: str,
    env_lock_digest: str | None = None,
    recipe: TrainingRecipeConfig | None = None,
    manifest: ClinicalPHIDatasetManifest | None = None,
    languages: Sequence[str] = ("en",),
    formats: Sequence[str] = ("pytorch",),
    param_count: int | None = None,
) -> ClinicalPrivacyCheckpointManifest:
    """Build the named checkpoint manifest from a completed external run.

    Args:
        source_revisions: Revision identifier for every dataset manifest source.
        checkpoint_artifact_hash: SHA-256 digest of the produced checkpoint.
        git_sha: Source revision used for the training run.
        env_lock_digest: Optional pinned environment lock digest. Defaults to uv.lock.
        recipe: Optional recipe override for validation tests.
        manifest: Optional dataset manifest override for validation tests.
        languages: Languages explicitly claimed by the checkpoint.
        formats: Produced checkpoint formats.
        param_count: Optional positive checkpoint parameter count.

    Returns:
        A validated checkpoint manifest containing no corpus content.
    """

    plan = resolve_clinical_privacy_training_plan(
        source_revisions,
        recipe=recipe,
        manifest=manifest,
    )
    artifact_hash = _require_sha256(
        checkpoint_artifact_hash, "checkpoint_artifact_hash"
    )
    source_sha = _require_revision(git_sha, "git_sha")
    lock_digest = _require_sha256(
        env_lock_digest or compute_environment_lock_digest(),
        "env_lock_digest",
    )
    provenance = build_training_provenance(
        rng_seeds=plan.rng_seeds,
        data_manifest_hash=plan.manifest_hash,
        recipe_config_hash=plan.recipe_config_hash,
        env_lock_digest=lock_digest,
        base_model=plan.recipe.backbone.model_ref,
        base_model_revision=plan.recipe.backbone.revision,
        git_sha=source_sha,
        repo_id=plan.model_id,
        checkpoint_id=CLINICAL_PRIVACY_CHECKPOINT_NAME,
    )
    verify_reproducibility(provenance)

    checkpoint_row: dict[str, Any] = {
        "architecture": plan.recipe.backbone.family,
        "arxiv": None,
        "base_model": plan.recipe.backbone.model_ref,
        "benchmark": {
            "dataset": "pending-certification",
            "micro_f1": None,
            "recall": None,
        },
        "canonical_labels": list(plan.label_weights),
        "family": CLINICAL_PRIVACY_FAMILY,
        "formats": list(formats),
        "languages": list(languages),
        "license": "apache-2.0",
        "param_count": param_count,
        "released": None,
        "repo_id": plan.model_id,
        "reproducibility_hash": provenance["reproducibility_hash"],
        "task": "token-classification",
        "tier": "Large",
        "training_provenance": dict(provenance),
    }
    violations = validate_manifest_row(checkpoint_row, line_number=1)
    if violations:
        raise ClinicalPrivacyTrainingError(
            "checkpoint row is invalid: "
            + "; ".join(violation.message for violation in violations)
        )

    result = ClinicalPrivacyCheckpointManifest(
        training_plan=plan,
        checkpoint_artifact_hash=artifact_hash,
        checkpoint_row=checkpoint_row,
        training_provenance=provenance,
        checkpoint_manifest_hash="",
    )
    result = replace(
        result,
        checkpoint_manifest_hash=_canonical_hash(result._payload_without_hash()),
    )
    validate_clinical_privacy_checkpoint_manifest(result)
    return result


def validate_clinical_privacy_checkpoint_manifest(
    manifest: ClinicalPrivacyCheckpointManifest,
) -> None:
    """Validate checkpoint identity, hashes, and the gated-content boundary."""

    payload = manifest.to_dict()
    expected_hash = _canonical_hash(manifest._payload_without_hash())
    if manifest.checkpoint_manifest_hash != expected_hash:
        raise ClinicalPrivacyTrainingError(
            "checkpoint_manifest_hash does not match checkpoint metadata"
        )
    if payload["checkpoint_name"] != CLINICAL_PRIVACY_CHECKPOINT_NAME:
        raise ClinicalPrivacyTrainingError("checkpoint name is not the tier0 flagship")
    if payload["model_id"] != CLINICAL_PRIVACY_MODEL_ID:
        raise ClinicalPrivacyTrainingError("checkpoint model_id is not the flagship")
    if payload["dataset_manifest"]["id"] != CLINICAL_PHI_MANIFEST_ID:
        raise ClinicalPrivacyTrainingError("checkpoint dataset manifest id is invalid")
    if payload["training_recipe"]["mode"] != "C":
        raise ClinicalPrivacyTrainingError("checkpoint must record recipe mode C")
    if payload["training_recipe"]["objective"] != CLINICAL_PRIVACY_OBJECTIVE:
        raise ClinicalPrivacyTrainingError(
            "checkpoint must record the recall-first objective"
        )
    if payload["training_recipe"]["config_ref"] != CLINICAL_PRIVACY_RECIPE_REF:
        raise ClinicalPrivacyTrainingError("checkpoint recipe reference is invalid")
    if payload["privacy"] != {
        "contains_gated_evaluation_content": False,
        "contains_raw_phi": False,
        "metadata_only_resolution": True,
    }:
        raise ClinicalPrivacyTrainingError("checkpoint privacy declaration is invalid")
    _assert_no_content_fields(payload)
    _require_sha256(manifest.checkpoint_artifact_hash, "checkpoint_artifact_hash")
    if manifest.training_plan.source_revisions_hash != _canonical_hash(
        manifest.training_plan.source_revisions()
    ):
        raise ClinicalPrivacyTrainingError(
            "source_revisions_hash does not match resolved source revisions"
        )
    for source in manifest.training_plan.sources:
        if source.content_included:
            raise ClinicalPrivacyTrainingError(
                f"source {source.source_id} must not embed corpus content"
            )
        if source.eval_only and source.selected_for_training:
            raise ClinicalPrivacyTrainingError(
                f"eval-only source {source.source_id} cannot be used for training"
            )
    verify_reproducibility(manifest.training_provenance)
    if manifest.training_provenance.get("data_manifest_hash") != (
        manifest.training_plan.manifest_hash
    ):
        raise ClinicalPrivacyTrainingError(
            "training provenance does not match the dataset manifest"
        )
    if manifest.training_provenance.get("recipe_config_hash") != (
        manifest.training_plan.recipe_config_hash
    ):
        raise ClinicalPrivacyTrainingError(
            "training provenance does not match the mode-C recipe"
        )
    if manifest.checkpoint_row.get("training_provenance") != (
        manifest.training_provenance
    ):
        raise ClinicalPrivacyTrainingError(
            "checkpoint row training provenance does not match the wrapper"
        )
    row_violations = validate_manifest_row(dict(manifest.checkpoint_row), line_number=1)
    if row_violations:
        raise ClinicalPrivacyTrainingError(
            "checkpoint row is invalid: "
            + "; ".join(violation.message for violation in row_violations)
        )


def write_clinical_privacy_checkpoint_manifest(
    path: str | Path,
    manifest: ClinicalPrivacyCheckpointManifest,
) -> Path:
    """Write a validated, PHI-free checkpoint manifest as canonical JSON."""

    validate_clinical_privacy_checkpoint_manifest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            manifest.to_dict(),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def _training_labels(manifest: ClinicalPHIDatasetManifest) -> tuple[str, ...]:
    labels = list(manifest.required_labels())
    for threshold in manifest.gate_families:
        labels.extend(threshold.labels)
    return tuple(dict.fromkeys(labels))


def _validate_mode_c_recipe(
    recipe: TrainingRecipeConfig,
    manifest: ClinicalPHIDatasetManifest,
    labels: Sequence[str],
) -> None:
    errors: list[str] = []
    if recipe.mode != "C" or recipe.preset_name != "large_teacher":
        errors.append("recipe must be the large_teacher mode-C preset")
    if manifest.recipe_mode != "C":
        errors.append("dataset manifest must declare recipe mode C")
    if recipe.dapt.corpus_ref != CLINICAL_PHI_MANIFEST_REF:
        errors.append("recipe must resolve the clinical PHI dataset manifest")
    if recipe.head_contract != CLINICAL_PRIVACY_CONTRACT_REF:
        errors.append("recipe must declare the clinical privacy tier0 contract")
    if recipe.output_tier != "teacher":
        errors.append("recipe output_tier must be teacher")
    if recipe.loss.class_weighting != "inverse_frequency":
        errors.append("recipe must use inverse-frequency class weighting")
    if recipe.loss.critical_label_weight <= 1:
        errors.append("clinical PHI label weight must be greater than one")

    configured_labels = set(recipe.loss.critical_labels)
    expected_labels = set(labels)
    missing_labels = sorted(expected_labels - configured_labels)
    extra_labels = sorted(configured_labels - expected_labels)
    if missing_labels:
        errors.append(
            "recipe is missing clinical PHI labels: " + ", ".join(missing_labels)
        )
    if extra_labels:
        errors.append(
            "recipe has labels outside the clinical PHI manifest: "
            + ", ".join(extra_labels)
        )
    if errors:
        raise ClinicalPrivacyTrainingError("; ".join(errors))


def _validate_source_revisions(
    source_revisions: Mapping[str, str],
    expected_ids: Sequence[str],
) -> None:
    if not isinstance(source_revisions, Mapping):
        raise ClinicalPrivacyTrainingError("source_revisions must be a mapping")
    expected = set(expected_ids)
    provided = set(source_revisions)
    missing = sorted(expected - provided)
    unknown = sorted(provided - expected)
    if missing:
        raise ClinicalPrivacyTrainingError(
            "source_revisions missing manifest source(s): " + ", ".join(missing)
        )
    if unknown:
        raise ClinicalPrivacyTrainingError(
            "source_revisions contains unknown source(s): " + ", ".join(unknown)
        )
    for source_id in expected_ids:
        _require_revision(source_revisions[source_id], f"source_revisions.{source_id}")


def _source_binding(
    source: ClinicalPHISource,
    revision: str,
) -> ClinicalPrivacySourceBinding:
    return ClinicalPrivacySourceBinding(
        source_id=source.source_id,
        dataset=source.dataset,
        revision=_require_revision(revision, f"source_revisions.{source.source_id}"),
        role=source.role,
        access=source.access,
        loader_ref=source.loader_ref,
        license_id=source.license_id,
        labels=source.labels,
        split=source.split,
        source_url=source.source_url,
        selected_for_training=(
            source.source_id in CLINICAL_PRIVACY_TRAINING_SOURCE_IDS
        ),
        requires_credentials=source.requires_credentials,
        eval_only=source.eval_only,
        synthetic=source.synthetic,
    )


def _validate_training_source_selection(
    bindings: Sequence[ClinicalPrivacySourceBinding],
) -> None:
    selected = tuple(
        source.source_id for source in bindings if source.selected_for_training
    )
    if selected != CLINICAL_PRIVACY_TRAINING_SOURCE_IDS:
        raise ClinicalPrivacyTrainingError(
            "mode-C training sources do not match the clinical PHI contract"
        )
    for source in bindings:
        if not source.selected_for_training:
            continue
        if not source.synthetic or source.eval_only or source.requires_credentials:
            raise ClinicalPrivacyTrainingError(
                f"training source {source.source_id} must be synthetic and non-gated"
            )


def _require_revision(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_REVISION_RE.fullmatch(value) is None:
        raise ClinicalPrivacyTrainingError(
            f"{field} must be a non-empty immutable revision identifier"
        )
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ClinicalPrivacyTrainingError(
            f"{field} must match sha256:<64 lowercase hex characters>"
        )
    return value


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _assert_no_content_fields(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).casefold()
            if normalized_key in _FORBIDDEN_CONTENT_KEYS:
                raise ClinicalPrivacyTrainingError(
                    f"{path}.{key} may not contain corpus or PHI content"
                )
            _assert_no_content_fields(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_no_content_fields(item, path=f"{path}[{index}]")


__all__ = [
    "CLINICAL_PRIVACY_CHECKPOINT_NAME",
    "CLINICAL_PRIVACY_CHECKPOINT_SCHEMA_VERSION",
    "CLINICAL_PRIVACY_CONTRACT_REF",
    "CLINICAL_PRIVACY_FAMILY",
    "CLINICAL_PRIVACY_OBJECTIVE",
    "CLINICAL_PRIVACY_RECIPE_REF",
    "CLINICAL_PRIVACY_TIER0_CONTRACT",
    "CLINICAL_PRIVACY_TRAINING_SCHEMA_VERSION",
    "CLINICAL_PRIVACY_TRAINING_SOURCE_IDS",
    "ClinicalPrivacyCheckpointManifest",
    "ClinicalPrivacyGateThreshold",
    "ClinicalPrivacySourceBinding",
    "ClinicalPrivacyTrainingError",
    "ClinicalPrivacyTrainingPlan",
    "ClinicalPrivacyTier0Contract",
    "build_clinical_privacy_checkpoint_manifest",
    "resolve_clinical_privacy_training_plan",
    "validate_clinical_privacy_checkpoint_manifest",
    "write_clinical_privacy_checkpoint_manifest",
]
