"""Offline donor-to-target adapter fine-tuning recipe.

The recipe describes how to initialize a target adapter from a donor adapter
while keeping the shared clinical backbone frozen. Its dry-run path only checks
local assets and emits aggregate, path-free metadata; it does not import a
training backend, download artifacts, or inspect synthetic-gold contents.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ADAPTER_RECIPE_SCHEMA_VERSION = "openmed.training.adapter_recipe.v1"
ADAPTER_METADATA_SCHEMA_VERSION = "openmed.training.adapter_metadata.v1"

DEFAULT_ADAPTER_TRAINING_DISCLAIMER = (
    "This adapter is for clinical decision support and research only. Its outputs "
    "require validation by qualified users and must not be used as the sole basis "
    "for diagnosis, treatment, or patient identification."
)

_PERMISSIVE_LICENSES = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "mit",
    }
)
_REMOTE_PATH_PREFIXES = (
    "ftp:/",
    "git+http:/",
    "git+https:/",
    "gs:/",
    "hf:/",
    "http:/",
    "https:/",
    "s3:/",
    "ssh:/",
)


class AdapterTrainingRecipeError(ValueError):
    """Raised when an adapter recipe cannot be validated or dry-run."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise AdapterTrainingRecipeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise AdapterTrainingRecipeError(f"{field_name} must not be empty")
    return normalized


def _normalize_language_code(value: object, field_name: str) -> str:
    normalized = _require_text(value, field_name).replace("_", "-").casefold()
    return normalized.split("-", 1)[0]


def _require_integer(
    value: object,
    field_name: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterTrainingRecipeError(f"{field_name} must be an integer")
    if value < minimum:
        raise AdapterTrainingRecipeError(
            f"{field_name} must be greater than or equal to {minimum}"
        )
    return value


def _require_finite_number(
    value: object,
    field_name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterTrainingRecipeError(f"{field_name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise AdapterTrainingRecipeError(
            f"{field_name} must be a finite number greater than or equal to {minimum}"
        )
    if maximum is not None and normalized > maximum:
        raise AdapterTrainingRecipeError(
            f"{field_name} must be less than or equal to {maximum}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class LocalTrainingAsset:
    """Logical identity and local path for one training input.

    The logical identity is emitted in provenance metadata. The filesystem path
    is used only during validation and is deliberately omitted from dry-run
    output so local paths cannot leak into portable artifacts.
    """

    asset_id: str
    path: str | Path

    def __post_init__(self) -> None:
        """Normalize the identity and reject remote asset references."""

        asset_id = _require_text(self.asset_id, "asset_id")
        raw_path = _require_text(str(self.path), "path")
        normalized_path = raw_path.casefold()
        if "://" in normalized_path or normalized_path.startswith(
            (*_REMOTE_PATH_PREFIXES, "git@")
        ):
            raise AdapterTrainingRecipeError("training asset paths must be local-only")
        object.__setattr__(self, "asset_id", asset_id)
        object.__setattr__(self, "path", Path(raw_path))

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable recipe representation of this asset."""

        return {
            "asset_id": self.asset_id,
            "local_files_only": True,
            "path": str(self.path),
        }


@dataclass(frozen=True, slots=True)
class ParameterEfficientAdapterConfig:
    """LoRA shape used to create the trainable target adapter."""

    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    method: str = "lora"

    def __post_init__(self) -> None:
        """Validate the adapter method and low-rank shape."""

        method = _require_text(self.method, "method").casefold()
        if method != "lora":
            raise AdapterTrainingRecipeError("method must be 'lora'")
        rank = _require_integer(self.rank, "rank", minimum=1)
        alpha = _require_integer(self.alpha, "alpha", minimum=1)
        dropout = _require_finite_number(
            self.dropout,
            "dropout",
            minimum=0.0,
            maximum=1.0,
        )
        if dropout == 1.0:
            raise AdapterTrainingRecipeError("dropout must be less than 1.0")
        if isinstance(self.target_modules, (str, bytes)):
            raise AdapterTrainingRecipeError(
                "target_modules must be an iterable of module names"
            )
        target_modules = tuple(
            _require_text(module, "target_modules") for module in self.target_modules
        )
        if not target_modules:
            raise AdapterTrainingRecipeError("target_modules must not be empty")
        if len(set(target_modules)) != len(target_modules):
            raise AdapterTrainingRecipeError(
                "target_modules must not contain duplicates"
            )

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "target_modules", target_modules)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable adapter hyperparameters."""

        return {
            "alpha": self.alpha,
            "dropout": self.dropout,
            "method": self.method,
            "rank": self.rank,
            "target_modules": list(self.target_modules),
        }


@dataclass(frozen=True, slots=True)
class AdapterTrainingSchedule:
    """Deterministic optimization schedule for target-language adaptation."""

    epochs: int
    batch_size: int
    learning_rate: float
    gradient_accumulation_steps: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate positive schedule values and a deterministic seed."""

        object.__setattr__(
            self,
            "epochs",
            _require_integer(self.epochs, "epochs", minimum=1),
        )
        object.__setattr__(
            self,
            "batch_size",
            _require_integer(self.batch_size, "batch_size", minimum=1),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _require_finite_number(
                self.learning_rate,
                "learning_rate",
                minimum=0.0,
            ),
        )
        if self.learning_rate == 0.0:
            raise AdapterTrainingRecipeError("learning_rate must be greater than 0")
        object.__setattr__(
            self,
            "gradient_accumulation_steps",
            _require_integer(
                self.gradient_accumulation_steps,
                "gradient_accumulation_steps",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "seed",
            _require_integer(self.seed, "seed", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serializable optimization settings."""

        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class AdapterParameterAccounting:
    """Aggregate trainable counts for a shared-backbone adapter run."""

    shared_backbone_parameter_count: int
    adapter_trainable_parameter_count: int
    task_head_trainable_parameter_count: int
    full_language_model_trainable_parameter_count: int

    def __post_init__(self) -> None:
        """Validate positive counts and enforce parameter efficiency."""

        object.__setattr__(
            self,
            "shared_backbone_parameter_count",
            _require_integer(
                self.shared_backbone_parameter_count,
                "shared_backbone_parameter_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "adapter_trainable_parameter_count",
            _require_integer(
                self.adapter_trainable_parameter_count,
                "adapter_trainable_parameter_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "task_head_trainable_parameter_count",
            _require_integer(
                self.task_head_trainable_parameter_count,
                "task_head_trainable_parameter_count",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "full_language_model_trainable_parameter_count",
            _require_integer(
                self.full_language_model_trainable_parameter_count,
                "full_language_model_trainable_parameter_count",
                minimum=1,
            ),
        )
        if (
            self.trainable_parameter_count
            >= self.full_language_model_trainable_parameter_count
        ):
            raise AdapterTrainingRecipeError(
                "adapter trainable parameter count must be lower than the full "
                "per-language model baseline"
            )
        if self.full_language_model_trainable_parameter_count < (
            self.shared_backbone_parameter_count
            + self.task_head_trainable_parameter_count
        ):
            raise AdapterTrainingRecipeError(
                "full per-language model baseline must include the shared "
                "backbone and task head"
            )

    @property
    def trainable_parameter_count(self) -> int:
        """Return adapter plus task-head parameters updated during training."""

        return (
            self.adapter_trainable_parameter_count
            + self.task_head_trainable_parameter_count
        )

    @property
    def frozen_parameter_count(self) -> int:
        """Return parameters kept frozen in the shared backbone."""

        return self.shared_backbone_parameter_count

    @property
    def trainable_fraction_of_full_model(self) -> float:
        """Return the adapter trainable count divided by the full baseline."""

        return (
            self.trainable_parameter_count
            / self.full_language_model_trainable_parameter_count
        )

    def to_dict(self) -> dict[str, Any]:
        """Return complete, serializable parameter accounting."""

        return {
            "adapter_trainable_parameter_count": (
                self.adapter_trainable_parameter_count
            ),
            "frozen_parameter_count": self.frozen_parameter_count,
            "full_language_model_trainable_parameter_count": (
                self.full_language_model_trainable_parameter_count
            ),
            "parameter_reduction_count": (
                self.full_language_model_trainable_parameter_count
                - self.trainable_parameter_count
            ),
            "shared_backbone_parameter_count": (self.shared_backbone_parameter_count),
            "task_head_trainable_parameter_count": (
                self.task_head_trainable_parameter_count
            ),
            "trainable_fraction_of_full_model": (self.trainable_fraction_of_full_model),
            "trainable_parameter_count": self.trainable_parameter_count,
        }


@dataclass(frozen=True, slots=True)
class DonorToTargetAdapterRecipe:
    """Validated recipe for adapting a donor adapter to a target language."""

    recipe_id: str
    donor_language: str
    target_language: str
    output_adapter_id: str
    backbone: LocalTrainingAsset
    donor_adapter: LocalTrainingAsset
    synthetic_gold: LocalTrainingAsset
    adapter: ParameterEfficientAdapterConfig
    schedule: AdapterTrainingSchedule
    parameter_accounting: AdapterParameterAccounting
    provenance: str
    license: str = "apache-2.0"
    disclaimer: str = DEFAULT_ADAPTER_TRAINING_DISCLAIMER
    schema_version: str = ADAPTER_RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize the transfer pair and validate recipe provenance."""

        schema_version = _require_text(self.schema_version, "schema_version")
        if schema_version != ADAPTER_RECIPE_SCHEMA_VERSION:
            raise AdapterTrainingRecipeError(
                f"schema_version must be {ADAPTER_RECIPE_SCHEMA_VERSION!r}"
            )
        donor_language = _normalize_language_code(
            self.donor_language,
            "donor_language",
        )
        target_language = _normalize_language_code(
            self.target_language,
            "target_language",
        )
        if donor_language == target_language:
            raise AdapterTrainingRecipeError(
                "donor_language must differ from target_language"
            )
        for field_name, expected_type in (
            ("backbone", LocalTrainingAsset),
            ("donor_adapter", LocalTrainingAsset),
            ("synthetic_gold", LocalTrainingAsset),
            ("adapter", ParameterEfficientAdapterConfig),
            ("schedule", AdapterTrainingSchedule),
            ("parameter_accounting", AdapterParameterAccounting),
        ):
            if not isinstance(getattr(self, field_name), expected_type):
                raise AdapterTrainingRecipeError(
                    f"{field_name} must be {expected_type.__name__}"
                )

        license_name = _require_text(self.license, "license").casefold()
        if license_name not in _PERMISSIVE_LICENSES:
            raise AdapterTrainingRecipeError(
                f"adapter license {license_name!r} is not permissive"
            )

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self, "recipe_id", _require_text(self.recipe_id, "recipe_id")
        )
        object.__setattr__(self, "donor_language", donor_language)
        object.__setattr__(self, "target_language", target_language)
        object.__setattr__(
            self,
            "output_adapter_id",
            _require_text(self.output_adapter_id, "output_adapter_id"),
        )
        object.__setattr__(
            self, "provenance", _require_text(self.provenance, "provenance")
        )
        object.__setattr__(self, "license", license_name)
        object.__setattr__(
            self,
            "disclaimer",
            _require_text(self.disclaimer, "disclaimer"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete serializable recipe, including local paths."""

        return {
            "adapter": self.adapter.to_dict(),
            "backbone": self.backbone.to_dict(),
            "disclaimer": self.disclaimer,
            "donor": {
                "adapter": self.donor_adapter.to_dict(),
                "language": self.donor_language,
            },
            "initialization_source": "donor_adapter",
            "license": self.license,
            "output_adapter_id": self.output_adapter_id,
            "parameter_accounting": self.parameter_accounting.to_dict(),
            "provenance": self.provenance,
            "recipe_id": self.recipe_id,
            "schedule": self.schedule.to_dict(),
            "schema_version": self.schema_version,
            "shared_backbone_frozen": True,
            "synthetic_gold": self.synthetic_gold.to_dict(),
            "target_language": self.target_language,
        }


@dataclass(frozen=True, slots=True)
class AdapterArtifactMetadata:
    """Path-free provenance metadata for the planned target adapter."""

    recipe_id: str
    adapter_id: str
    donor_language: str
    target_language: str
    backbone: str
    donor_adapter: str
    synthetic_gold_source: str
    license: str
    disclaimer: str
    provenance: Mapping[str, str]
    adapter_config: ParameterEfficientAdapterConfig
    schedule: AdapterTrainingSchedule
    parameter_accounting: AdapterParameterAccounting

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible adapter metadata with no local paths."""

        return {
            "adapter_config": self.adapter_config.to_dict(),
            "adapter_id": self.adapter_id,
            "backbone": self.backbone,
            "disclaimer": self.disclaimer,
            "donor_adapter": self.donor_adapter,
            "donor_language": self.donor_language,
            "initialization_source": self.donor_adapter,
            "license": self.license,
            "local_files_only": True,
            "offline_runnable": True,
            "parameter_accounting": self.parameter_accounting.to_dict(),
            "provenance": dict(self.provenance),
            "recipe_id": self.recipe_id,
            "schedule": self.schedule.to_dict(),
            "schema_version": ADAPTER_METADATA_SCHEMA_VERSION,
            "shared_backbone_frozen": True,
            "synthetic_gold_source": self.synthetic_gold_source,
            "target_language": self.target_language,
        }


@dataclass(frozen=True, slots=True)
class AdapterRecipeDryRun:
    """Deterministic result of validating an offline adapter recipe."""

    recipe_hash: str
    metadata: AdapterArtifactMetadata
    verified_local_assets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return serializable, PHI-free dry-run evidence."""

        return {
            "initialization_source": self.metadata.donor_adapter,
            "metadata": self.metadata.to_dict(),
            "network_accessed": False,
            "recipe_hash": self.recipe_hash,
            "training_started": False,
            "verified_local_assets": list(self.verified_local_assets),
        }


def build_donor_to_target_adapter_recipe(
    *,
    recipe_id: str,
    donor_language: str,
    target_language: str,
    output_adapter_id: str,
    backbone: LocalTrainingAsset,
    donor_adapter: LocalTrainingAsset,
    synthetic_gold: LocalTrainingAsset,
    parameter_accounting: AdapterParameterAccounting,
    adapter: ParameterEfficientAdapterConfig | None = None,
    schedule: AdapterTrainingSchedule | None = None,
    provenance: str = "Synthetic donor-to-target clinical adapter recipe",
    license: str = "apache-2.0",
    disclaimer: str = DEFAULT_ADAPTER_TRAINING_DISCLAIMER,
) -> DonorToTargetAdapterRecipe:
    """Build a donor-initialized, shared-backbone adapter recipe.

    Args:
        recipe_id: Stable identity for the recipe configuration.
        donor_language: Language of the configured initialization adapter.
        target_language: Language learned by the output adapter.
        output_adapter_id: Stable identity for the planned target adapter.
        backbone: Locally available shared clinical backbone.
        donor_adapter: Locally available donor adapter used for initialization.
        synthetic_gold: Local synthetic-gold training source.
        parameter_accounting: Measured aggregate parameter counts.
        adapter: Optional LoRA configuration. A conservative default is used
            when omitted.
        schedule: Optional deterministic optimization schedule.
        provenance: Human-readable recipe provenance statement.
        license: Permissive SPDX identifier for the output adapter.
        disclaimer: Clinical-use disclaimer carried into artifact metadata.

    Returns:
        A validated donor-to-target adapter recipe.
    """

    return DonorToTargetAdapterRecipe(
        recipe_id=recipe_id,
        donor_language=donor_language,
        target_language=target_language,
        output_adapter_id=output_adapter_id,
        backbone=backbone,
        donor_adapter=donor_adapter,
        synthetic_gold=synthetic_gold,
        adapter=adapter
        or ParameterEfficientAdapterConfig(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=("query", "value"),
        ),
        schedule=schedule
        or AdapterTrainingSchedule(
            epochs=3,
            batch_size=8,
            learning_rate=0.0002,
            seed=17,
        ),
        parameter_accounting=parameter_accounting,
        provenance=provenance,
        license=license,
        disclaimer=disclaimer,
    )


def dry_run_donor_to_target_adapter_recipe(
    recipe: DonorToTargetAdapterRecipe,
) -> AdapterRecipeDryRun:
    """Validate local inputs and emit path-free adapter metadata.

    This function never opens a network connection or imports optional trainer
    libraries. It verifies only that configured local assets exist and are
    non-empty; the synthetic-gold contents remain unread.

    Args:
        recipe: Validated donor-to-target recipe to inspect.

    Returns:
        Serializable dry-run evidence and target-adapter metadata.

    Raises:
        AdapterTrainingRecipeError: If an asset is missing or empty.
    """

    if not isinstance(recipe, DonorToTargetAdapterRecipe):
        raise AdapterTrainingRecipeError("recipe must be DonorToTargetAdapterRecipe")

    _validate_local_asset(recipe.backbone, role="backbone")
    _validate_local_asset(recipe.donor_adapter, role="donor adapter")
    _validate_local_asset(
        recipe.synthetic_gold,
        role="synthetic gold",
        require_file=True,
    )

    metadata = AdapterArtifactMetadata(
        recipe_id=recipe.recipe_id,
        adapter_id=recipe.output_adapter_id,
        donor_language=recipe.donor_language,
        target_language=recipe.target_language,
        backbone=recipe.backbone.asset_id,
        donor_adapter=recipe.donor_adapter.asset_id,
        synthetic_gold_source=recipe.synthetic_gold.asset_id,
        license=recipe.license,
        disclaimer=recipe.disclaimer,
        provenance={
            "initialization": "donor_adapter",
            "recipe": recipe.provenance,
            "shared_backbone": recipe.backbone.asset_id,
            "synthetic_gold_source": recipe.synthetic_gold.asset_id,
        },
        adapter_config=recipe.adapter,
        schedule=recipe.schedule,
        parameter_accounting=recipe.parameter_accounting,
    )
    return AdapterRecipeDryRun(
        recipe_hash=_portable_recipe_hash(recipe),
        metadata=metadata,
        verified_local_assets=(
            recipe.backbone.asset_id,
            recipe.donor_adapter.asset_id,
            recipe.synthetic_gold.asset_id,
        ),
    )


def _validate_local_asset(
    asset: LocalTrainingAsset,
    *,
    role: str,
    require_file: bool = False,
) -> None:
    path = Path(asset.path).expanduser()
    if not path.exists():
        raise AdapterTrainingRecipeError(
            f"{role} asset {asset.asset_id!r} is not available locally"
        )
    if require_file and not path.is_file():
        raise AdapterTrainingRecipeError(
            f"{role} asset {asset.asset_id!r} must be a local file"
        )
    if path.is_file():
        if path.stat().st_size == 0:
            raise AdapterTrainingRecipeError(
                f"{role} asset {asset.asset_id!r} must not be empty"
            )
        return
    if path.is_dir() and any(child.is_file() for child in path.rglob("*")):
        return
    raise AdapterTrainingRecipeError(
        f"{role} asset {asset.asset_id!r} must contain at least one local file"
    )


def _portable_recipe_hash(recipe: DonorToTargetAdapterRecipe) -> str:
    payload = {
        "adapter": recipe.adapter.to_dict(),
        "backbone": recipe.backbone.asset_id,
        "disclaimer": recipe.disclaimer,
        "donor_adapter": recipe.donor_adapter.asset_id,
        "donor_language": recipe.donor_language,
        "license": recipe.license,
        "output_adapter_id": recipe.output_adapter_id,
        "parameter_accounting": recipe.parameter_accounting.to_dict(),
        "provenance": recipe.provenance,
        "recipe_id": recipe.recipe_id,
        "schedule": recipe.schedule.to_dict(),
        "schema_version": recipe.schema_version,
        "synthetic_gold_source": recipe.synthetic_gold.asset_id,
        "target_language": recipe.target_language,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "ADAPTER_METADATA_SCHEMA_VERSION",
    "ADAPTER_RECIPE_SCHEMA_VERSION",
    "DEFAULT_ADAPTER_TRAINING_DISCLAIMER",
    "AdapterArtifactMetadata",
    "AdapterParameterAccounting",
    "AdapterRecipeDryRun",
    "AdapterTrainingRecipeError",
    "AdapterTrainingSchedule",
    "DonorToTargetAdapterRecipe",
    "LocalTrainingAsset",
    "ParameterEfficientAdapterConfig",
    "build_donor_to_target_adapter_recipe",
    "dry_run_donor_to_target_adapter_recipe",
]
