"""Reproducible, budget-bounded training plans for Journey specialists.

This module resolves local configuration and synthetic data into immutable dry-
run manifests. It never downloads a model, opens a network connection, or
launches a GPU job. A separate caller may execute the pinned recipe, but it
must preserve the resulting manifest and shared spend ledger.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import resources
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import (
    canonical_json,
    derived_opaque_id,
    sha256_digest,
)
from openmed.structured.store import StoreResult, StoreState

JOURNEY_SPECIALIST_SCHEMA_VERSION = "1.0.0"
JOURNEY_SPECIALIST_COMPATIBILITY_POLICY = "same_major"
JOURNEY_SPECIALIST_CONFIG_PACKAGE = "openmed.training.configs"
JOURNEY_SPECIALIST_CONFIG_NAME = "journey_specialist_pack.json"
JOURNEY_SPECIALIST_DATA_PACKAGE = "openmed.training.data"
JOURNEY_SPECIALIST_SCHEMA_PACKAGE = "openmed.core.schemas.json"
JOURNEY_SPECIALIST_SCHEMA_NAME = "journey_specialist_training"
JOURNEY_SPECIALIST_GPU_BUDGET_USD = 1000.0

JOURNEY_SPECIALIST_TASKS = (
    "assertion",
    "temporality",
    "pair_scoring",
    "classification",
)
JOURNEY_SPECIALIST_ARCHITECTURES = frozenset(
    {"sequence_classifier", "span_pair_classifier"}
)
JOURNEY_SPECIALIST_ADAPTER_METHODS = frozenset({"head_only", "ia3", "lora"})
JOURNEY_SPECIALIST_PRECISIONS = frozenset({"bf16", "fp16"})
JOURNEY_SPECIALIST_EXPORT_FORMATS = frozenset(
    {"coreml", "onnx", "safetensors", "torchscript"}
)
JOURNEY_SPECIALIST_QUANTIZATION_TARGETS = frozenset({"int8", "int4"})
JOURNEY_SPECIALIST_PERMISSIVE_LICENSES = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by-4.0",
        "cc0-1.0",
        "isc",
        "mit",
        "unlicense",
    }
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_SAFE_RESOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CONFIG_KEYS = frozenset(
    {
        "aggregate_gpu_budget_usd",
        "compatibility_policy",
        "dataset",
        "pack_id",
        "recipes",
        "schema_version",
    }
)
_RECIPE_KEYS = frozenset(
    {
        "adapter",
        "backbone",
        "baseline_id",
        "calibration_method",
        "estimated_gpu_hours",
        "export_formats",
        "hourly_cost_usd",
        "labels",
        "maximum_ece",
        "maximum_quantized_macro_f1_delta",
        "minimum_candidate_improvement",
        "minimum_coverage",
        "minimum_macro_f1",
        "minimum_per_class_recall",
        "minimum_subgroup_recall",
        "mixed_precision",
        "output_schema",
        "quantization_targets",
        "seed",
        "stop_rule",
        "task",
    }
)


class JourneySpecialistError(ValueError):
    """Base error for invalid specialist-pack contracts."""


class JourneySpecialistDeniedError(JourneySpecialistError):
    """Raised when a license, privacy, or budget policy denies a plan."""


class JourneySpecialistConflictError(JourneySpecialistError):
    """Raised when pinned content and observed content disagree."""


class JourneySpecialistUnsupportedError(JourneySpecialistError):
    """Raised when a requested recipe capability is unsupported."""


@dataclass(frozen=True, slots=True)
class SpecialistDatasetLineage:
    """Pinned dataset lineage without source records or raw text."""

    source_id: str
    revision: str
    content_digest: str
    license: str
    record_count: int
    resource: str
    usage_lane: str
    synthetic: bool
    bundled: bool
    schema_version: str = JOURNEY_SPECIALIST_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_SPECIALIST_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _controlled(self.source_id, "source_id")
        _version(self.revision, "dataset revision")
        _digest(self.content_digest, "dataset content_digest")
        license_name = _controlled(self.license, "dataset license")
        if self.usage_lane not in {"distributable", "user_supplied_eval"}:
            raise JourneySpecialistUnsupportedError("dataset usage lane is unsupported")
        if type(self.record_count) is not int or self.record_count < 1:
            raise JourneySpecialistError("dataset record_count must be positive")
        if (
            not isinstance(self.resource, str)
            or _SAFE_RESOURCE_RE.fullmatch(self.resource) is None
        ):
            raise JourneySpecialistError("dataset resource must be a safe local name")
        if type(self.synthetic) is not bool or type(self.bundled) is not bool:
            raise JourneySpecialistError("dataset flags must be booleans")
        if self.bundled and (
            not self.synthetic
            or self.usage_lane != "distributable"
            or license_name not in JOURNEY_SPECIALIST_PERMISSIVE_LICENSES
        ):
            raise JourneySpecialistDeniedError(
                "bundled datasets must be synthetic, distributable, and permissive"
            )
        _contract_version(self.schema_version, self.compatibility_policy)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic value-free lineage metadata."""

        return {
            "bundled": self.bundled,
            "compatibility_policy": self.compatibility_policy,
            "content_digest": self.content_digest,
            "license": self.license,
            "record_count": self.record_count,
            "resource": self.resource,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "synthetic": self.synthetic,
            "usage_lane": self.usage_lane,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpecialistDatasetLineage":
        """Build validated lineage from the public configuration shape."""

        data = _mapping(value, "dataset")
        _exact_keys(
            data,
            {
                "bundled",
                "content_digest",
                "license",
                "record_count",
                "resource",
                "revision",
                "source_id",
                "synthetic",
                "usage_lane",
            },
            {"schema_version", "compatibility_policy"},
            "dataset",
        )
        return cls(
            source_id=_text(data["source_id"], "dataset source_id"),
            revision=_text(data["revision"], "dataset revision"),
            content_digest=_text(data["content_digest"], "dataset content_digest"),
            license=_text(data["license"], "dataset license"),
            record_count=_integer(data["record_count"], "dataset record_count"),
            resource=_text(data["resource"], "dataset resource"),
            usage_lane=_text(data["usage_lane"], "dataset usage_lane"),
            synthetic=_boolean(data["synthetic"], "dataset synthetic"),
            bundled=_boolean(data["bundled"], "dataset bundled"),
            schema_version=_text(
                data.get("schema_version", JOURNEY_SPECIALIST_SCHEMA_VERSION),
                "dataset schema_version",
            ),
            compatibility_policy=_text(
                data.get(
                    "compatibility_policy",
                    JOURNEY_SPECIALIST_COMPATIBILITY_POLICY,
                ),
                "dataset compatibility_policy",
            ),
        )


@dataclass(frozen=True, slots=True)
class SpecialistBackbone:
    """Immutable, license-aware encoder backbone reference."""

    model_id: str
    revision: str
    license: str
    architecture: str

    def __post_init__(self) -> None:
        _controlled(self.model_id, "backbone model_id")
        if _SHA_RE.fullmatch(self.revision) is None:
            raise JourneySpecialistError("backbone revision must be an immutable hash")
        license_name = _controlled(self.license, "backbone license")
        if license_name not in JOURNEY_SPECIALIST_PERMISSIVE_LICENSES:
            raise JourneySpecialistDeniedError("backbone license is not permissive")
        if self.architecture not in JOURNEY_SPECIALIST_ARCHITECTURES:
            raise JourneySpecialistUnsupportedError(
                "backbone architecture is unsupported"
            )

    def to_dict(self) -> dict[str, str]:
        """Return the pinned backbone record."""

        return {
            "architecture": self.architecture,
            "license": self.license,
            "model_id": self.model_id,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpecialistBackbone":
        """Build a backbone from configuration."""

        data = _mapping(value, "backbone")
        _exact_keys(
            data,
            {"architecture", "license", "model_id", "revision"},
            set(),
            "backbone",
        )
        return cls(**{key: _text(data[key], f"backbone {key}") for key in data})


@dataclass(frozen=True, slots=True)
class SpecialistAdapter:
    """Parameter-efficient adapter configuration for one specialist."""

    method: str
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    trainable_ratio_ceiling: float

    def __post_init__(self) -> None:
        if self.method not in JOURNEY_SPECIALIST_ADAPTER_METHODS:
            raise JourneySpecialistUnsupportedError("adapter method is unsupported")
        if type(self.rank) is not int or self.rank < 1 or self.rank > 256:
            raise JourneySpecialistError("adapter rank is invalid")
        if type(self.alpha) is not int or self.alpha < 1 or self.alpha > 1024:
            raise JourneySpecialistError("adapter alpha is invalid")
        _unit_interval(self.dropout, "adapter dropout")
        modules = _controlled_values(self.target_modules, "adapter target_modules")
        if not modules:
            raise JourneySpecialistError("adapter target_modules cannot be empty")
        ceiling = _finite(self.trainable_ratio_ceiling, "trainable_ratio_ceiling")
        if ceiling <= 0 or ceiling > 0.05:
            raise JourneySpecialistError(
                "adapter trainable ratio must be greater than zero and at most 0.05"
            )
        object.__setattr__(self, "target_modules", modules)
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "trainable_ratio_ceiling", ceiling)

    def to_dict(self) -> dict[str, Any]:
        """Return the parameter-efficient tuning contract."""

        return {
            "alpha": self.alpha,
            "dropout": self.dropout,
            "method": self.method,
            "rank": self.rank,
            "target_modules": list(self.target_modules),
            "trainable_ratio_ceiling": self.trainable_ratio_ceiling,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpecialistAdapter":
        """Build an adapter contract from configuration."""

        data = _mapping(value, "adapter")
        _exact_keys(
            data,
            {
                "alpha",
                "dropout",
                "method",
                "rank",
                "target_modules",
                "trainable_ratio_ceiling",
            },
            set(),
            "adapter",
        )
        return cls(
            method=_text(data["method"], "adapter method"),
            rank=_integer(data["rank"], "adapter rank"),
            alpha=_integer(data["alpha"], "adapter alpha"),
            dropout=_finite(data["dropout"], "adapter dropout"),
            target_modules=_text_sequence(
                data["target_modules"], "adapter target_modules"
            ),
            trainable_ratio_ceiling=_finite(
                data["trainable_ratio_ceiling"], "trainable_ratio_ceiling"
            ),
        )


@dataclass(frozen=True, slots=True)
class SpecialistStopRule:
    """Predeclared bounded stopping rule for one training run."""

    metric: str
    mode: str
    patience: int
    minimum_delta: float
    maximum_steps: int
    maximum_gpu_hours: float

    def __post_init__(self) -> None:
        _controlled(self.metric, "stop metric")
        if self.mode not in {"min", "max"}:
            raise JourneySpecialistError("stop mode must be min or max")
        if type(self.patience) is not int or self.patience < 1:
            raise JourneySpecialistError("stop patience must be positive")
        minimum_delta = _finite(self.minimum_delta, "stop minimum_delta")
        maximum_hours = _finite(self.maximum_gpu_hours, "maximum_gpu_hours")
        if minimum_delta < 0 or maximum_hours <= 0:
            raise JourneySpecialistError("stop-rule bounds are invalid")
        if type(self.maximum_steps) is not int or self.maximum_steps < 1:
            raise JourneySpecialistError("stop maximum_steps must be positive")
        object.__setattr__(self, "minimum_delta", minimum_delta)
        object.__setattr__(self, "maximum_gpu_hours", maximum_hours)

    def to_dict(self) -> dict[str, Any]:
        """Return the bounded stop rule."""

        return {
            "maximum_gpu_hours": self.maximum_gpu_hours,
            "maximum_steps": self.maximum_steps,
            "metric": self.metric,
            "minimum_delta": self.minimum_delta,
            "mode": self.mode,
            "patience": self.patience,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpecialistStopRule":
        """Build a stop rule from configuration."""

        data = _mapping(value, "stop_rule")
        _exact_keys(
            data,
            {
                "maximum_gpu_hours",
                "maximum_steps",
                "metric",
                "minimum_delta",
                "mode",
                "patience",
            },
            set(),
            "stop_rule",
        )
        return cls(
            metric=_text(data["metric"], "stop metric"),
            mode=_text(data["mode"], "stop mode"),
            patience=_integer(data["patience"], "stop patience"),
            minimum_delta=_finite(data["minimum_delta"], "stop minimum_delta"),
            maximum_steps=_integer(data["maximum_steps"], "stop maximum_steps"),
            maximum_gpu_hours=_finite(
                data["maximum_gpu_hours"], "stop maximum_gpu_hours"
            ),
        )


@dataclass(frozen=True, slots=True)
class SpecialistRecipe:
    """One fully pinned low-cost encoder or span-model training recipe."""

    task: str
    labels: tuple[str, ...]
    output_schema: str
    backbone: SpecialistBackbone
    adapter: SpecialistAdapter
    mixed_precision: str
    seed: int
    stop_rule: SpecialistStopRule
    estimated_gpu_hours: float
    hourly_cost_usd: float
    baseline_id: str
    calibration_method: str
    export_formats: tuple[str, ...]
    quantization_targets: tuple[str, ...]
    minimum_macro_f1: float
    minimum_per_class_recall: float
    minimum_subgroup_recall: float
    minimum_coverage: float
    minimum_candidate_improvement: float
    maximum_ece: float
    maximum_quantized_macro_f1_delta: float
    schema_version: str = JOURNEY_SPECIALIST_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_SPECIALIST_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.task not in JOURNEY_SPECIALIST_TASKS:
            raise JourneySpecialistUnsupportedError("specialist task is unsupported")
        labels = _controlled_values(self.labels, "recipe labels")
        if len(labels) < 2:
            raise JourneySpecialistError("recipe must declare at least two labels")
        _controlled(self.output_schema, "output_schema")
        if not isinstance(self.backbone, SpecialistBackbone):
            raise TypeError("recipe backbone must be SpecialistBackbone")
        if not isinstance(self.adapter, SpecialistAdapter):
            raise TypeError("recipe adapter must be SpecialistAdapter")
        if self.mixed_precision not in JOURNEY_SPECIALIST_PRECISIONS:
            raise JourneySpecialistUnsupportedError("mixed precision is unsupported")
        if type(self.seed) is not int or self.seed < 0:
            raise JourneySpecialistError("recipe seed must be non-negative")
        if not isinstance(self.stop_rule, SpecialistStopRule):
            raise TypeError("recipe stop_rule must be SpecialistStopRule")
        gpu_hours = _finite(self.estimated_gpu_hours, "estimated_gpu_hours")
        hourly_cost = _finite(self.hourly_cost_usd, "hourly_cost_usd")
        if gpu_hours <= 0 or hourly_cost <= 0:
            raise JourneySpecialistError("GPU hours and hourly cost must be positive")
        if gpu_hours > self.stop_rule.maximum_gpu_hours:
            raise JourneySpecialistError("estimated GPU hours exceed the stop rule")
        _controlled(self.baseline_id, "baseline_id")
        if self.calibration_method not in {"isotonic", "temperature"}:
            raise JourneySpecialistUnsupportedError("calibration method is unsupported")
        exports = _controlled_values(self.export_formats, "export_formats")
        if not exports or set(exports) - JOURNEY_SPECIALIST_EXPORT_FORMATS:
            raise JourneySpecialistUnsupportedError("export format is unsupported")
        quantization = _controlled_values(
            self.quantization_targets, "quantization_targets"
        )
        if (
            not quantization
            or set(quantization) - JOURNEY_SPECIALIST_QUANTIZATION_TARGETS
        ):
            raise JourneySpecialistUnsupportedError(
                "quantization target is unsupported"
            )
        for name in (
            "minimum_macro_f1",
            "minimum_per_class_recall",
            "minimum_subgroup_recall",
            "minimum_coverage",
            "minimum_candidate_improvement",
            "maximum_ece",
            "maximum_quantized_macro_f1_delta",
        ):
            _unit_interval(getattr(self, name), name)
        _contract_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "export_formats", exports)
        object.__setattr__(self, "quantization_targets", quantization)
        object.__setattr__(self, "estimated_gpu_hours", gpu_hours)
        object.__setattr__(self, "hourly_cost_usd", hourly_cost)

    @property
    def estimated_cost_usd(self) -> float:
        """Return the declared maximum estimated run cost."""

        return round(self.estimated_gpu_hours * self.hourly_cost_usd, 6)

    @property
    def digest(self) -> str:
        """Return the canonical recipe digest."""

        return sha256_digest(canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned recipe record."""

        return {
            "adapter": self.adapter.to_dict(),
            "backbone": self.backbone.to_dict(),
            "baseline_id": self.baseline_id,
            "calibration_method": self.calibration_method,
            "compatibility_policy": self.compatibility_policy,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_gpu_hours": self.estimated_gpu_hours,
            "export_formats": list(self.export_formats),
            "hourly_cost_usd": self.hourly_cost_usd,
            "labels": list(self.labels),
            "maximum_ece": self.maximum_ece,
            "maximum_quantized_macro_f1_delta": (self.maximum_quantized_macro_f1_delta),
            "minimum_candidate_improvement": self.minimum_candidate_improvement,
            "minimum_coverage": self.minimum_coverage,
            "minimum_macro_f1": self.minimum_macro_f1,
            "minimum_per_class_recall": self.minimum_per_class_recall,
            "minimum_subgroup_recall": self.minimum_subgroup_recall,
            "mixed_precision": self.mixed_precision,
            "output_schema": self.output_schema,
            "quantization_targets": list(self.quantization_targets),
            "schema_version": self.schema_version,
            "seed": self.seed,
            "stop_rule": self.stop_rule.to_dict(),
            "task": self.task,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpecialistRecipe":
        """Build a validated recipe from configuration."""

        data = _mapping(value, "recipe")
        _exact_keys(
            data,
            set(_RECIPE_KEYS),
            {
                "schema_version",
                "compatibility_policy",
                "estimated_cost_usd",
            },
            "recipe",
        )
        recipe = cls(
            task=_text(data["task"], "recipe task"),
            labels=_text_sequence(data["labels"], "recipe labels"),
            output_schema=_text(data["output_schema"], "recipe output_schema"),
            backbone=SpecialistBackbone.from_dict(
                _mapping(data["backbone"], "backbone")
            ),
            adapter=SpecialistAdapter.from_dict(_mapping(data["adapter"], "adapter")),
            mixed_precision=_text(data["mixed_precision"], "mixed_precision"),
            seed=_integer(data["seed"], "recipe seed"),
            stop_rule=SpecialistStopRule.from_dict(
                _mapping(data["stop_rule"], "stop_rule")
            ),
            estimated_gpu_hours=_finite(
                data["estimated_gpu_hours"], "estimated_gpu_hours"
            ),
            hourly_cost_usd=_finite(data["hourly_cost_usd"], "hourly_cost_usd"),
            baseline_id=_text(data["baseline_id"], "baseline_id"),
            calibration_method=_text(data["calibration_method"], "calibration_method"),
            export_formats=_text_sequence(data["export_formats"], "export_formats"),
            quantization_targets=_text_sequence(
                data["quantization_targets"], "quantization_targets"
            ),
            minimum_macro_f1=_finite(data["minimum_macro_f1"], "minimum_macro_f1"),
            minimum_per_class_recall=_finite(
                data["minimum_per_class_recall"], "minimum_per_class_recall"
            ),
            minimum_subgroup_recall=_finite(
                data["minimum_subgroup_recall"], "minimum_subgroup_recall"
            ),
            minimum_coverage=_finite(data["minimum_coverage"], "minimum_coverage"),
            minimum_candidate_improvement=_finite(
                data["minimum_candidate_improvement"],
                "minimum_candidate_improvement",
            ),
            maximum_ece=_finite(data["maximum_ece"], "maximum_ece"),
            maximum_quantized_macro_f1_delta=_finite(
                data["maximum_quantized_macro_f1_delta"],
                "maximum_quantized_macro_f1_delta",
            ),
            schema_version=_text(
                data.get("schema_version", JOURNEY_SPECIALIST_SCHEMA_VERSION),
                "recipe schema_version",
            ),
            compatibility_policy=_text(
                data.get(
                    "compatibility_policy",
                    JOURNEY_SPECIALIST_COMPATIBILITY_POLICY,
                ),
                "recipe compatibility_policy",
            ),
        )
        if (
            "estimated_cost_usd" in data
            and _finite(data["estimated_cost_usd"], "estimated_cost_usd")
            != recipe.estimated_cost_usd
        ):
            raise JourneySpecialistConflictError(
                "serialized estimated cost differs from recipe inputs"
            )
        return recipe


@dataclass(frozen=True, slots=True)
class SpecialistPackPlan:
    """One shared dataset and four specialist recipes under a hard cap."""

    pack_id: str
    dataset: SpecialistDatasetLineage
    recipes: tuple[SpecialistRecipe, ...]
    aggregate_gpu_budget_usd: float = JOURNEY_SPECIALIST_GPU_BUDGET_USD
    schema_version: str = JOURNEY_SPECIALIST_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_SPECIALIST_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _controlled(self.pack_id, "pack_id")
        if not isinstance(self.dataset, SpecialistDatasetLineage):
            raise TypeError("pack dataset must be SpecialistDatasetLineage")
        if self.dataset.usage_lane != "distributable":
            raise JourneySpecialistDeniedError(
                "eval-only datasets cannot enter a training plan"
            )
        recipes = tuple(self.recipes)
        if any(not isinstance(item, SpecialistRecipe) for item in recipes):
            raise TypeError("pack recipes must be SpecialistRecipe values")
        tasks = tuple(item.task for item in recipes)
        if set(tasks) != set(JOURNEY_SPECIALIST_TASKS) or len(tasks) != len(
            JOURNEY_SPECIALIST_TASKS
        ):
            raise JourneySpecialistError(
                "specialist pack must contain exactly one recipe for every task"
            )
        cap = _finite(self.aggregate_gpu_budget_usd, "aggregate GPU budget")
        if cap <= 0 or cap > JOURNEY_SPECIALIST_GPU_BUDGET_USD:
            raise JourneySpecialistDeniedError(
                "aggregate GPU budget exceeds the hard USD 1000 cap"
            )
        if sum(item.estimated_cost_usd for item in recipes) > cap + 1e-9:
            raise JourneySpecialistDeniedError(
                "planned specialist runs exceed the aggregate GPU budget"
            )
        _contract_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(
            self, "recipes", tuple(sorted(recipes, key=lambda item: item.task))
        )
        object.__setattr__(self, "aggregate_gpu_budget_usd", cap)

    @property
    def digest(self) -> str:
        """Return the canonical plan digest."""

        return sha256_digest(canonical_json(self.to_dict()))

    def recipe_for(self, task: str) -> SpecialistRecipe:
        """Return the recipe for one supported task."""

        for recipe in self.recipes:
            if recipe.task == task:
                return recipe
        raise JourneySpecialistUnsupportedError("specialist task is unsupported")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned pack plan."""

        return {
            "aggregate_gpu_budget_usd": self.aggregate_gpu_budget_usd,
            "compatibility_policy": self.compatibility_policy,
            "dataset": self.dataset.to_dict(),
            "pack_id": self.pack_id,
            "recipes": [item.to_dict() for item in self.recipes],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpecialistPackPlan":
        """Build the public specialist pack plan from JSON-compatible data."""

        data = _mapping(value, "specialist pack")
        if set(data) != set(_CONFIG_KEYS):
            raise JourneySpecialistError("specialist pack fields do not match contract")
        recipes = data["recipes"]
        if not isinstance(recipes, Sequence) or isinstance(recipes, (str, bytes)):
            raise JourneySpecialistError("specialist recipes must be an array")
        return cls(
            pack_id=_text(data["pack_id"], "pack_id"),
            dataset=SpecialistDatasetLineage.from_dict(
                _mapping(data["dataset"], "dataset")
            ),
            recipes=tuple(
                SpecialistRecipe.from_dict(_mapping(item, "recipe")) for item in recipes
            ),
            aggregate_gpu_budget_usd=_finite(
                data["aggregate_gpu_budget_usd"], "aggregate GPU budget"
            ),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )


@dataclass(frozen=True, slots=True)
class SpecialistTrainingExample:
    """One synthetic or caller-supplied example kept out of manifests."""

    example_id: str
    task: str
    label: str
    text: str = field(repr=False)
    subgroups: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _controlled(self.example_id, "example_id")
        if self.task not in JOURNEY_SPECIALIST_TASKS:
            raise JourneySpecialistUnsupportedError("example task is unsupported")
        _controlled(self.label, "example label")
        if not isinstance(self.text, str) or not self.text:
            raise JourneySpecialistError("training example text cannot be empty")
        subgroup_map = _mapping(self.subgroups, "example subgroups")
        normalized = {
            _controlled(key, "subgroup axis"): _controlled(value, "subgroup value")
            for key, value in subgroup_map.items()
        }
        object.__setattr__(self, "subgroups", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class SpecialistSplitManifest:
    """Seeded stratified split evidence containing no training text."""

    task: str
    seed: int
    strategy: str
    dataset_digest: str
    assignment_digest: str
    split_counts: Mapping[str, int]
    label_counts: Mapping[str, Mapping[str, int]]
    schema_version: str = JOURNEY_SPECIALIST_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_SPECIALIST_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.task not in JOURNEY_SPECIALIST_TASKS:
            raise JourneySpecialistUnsupportedError("split task is unsupported")
        if type(self.seed) is not int or self.seed < 0:
            raise JourneySpecialistError("split seed must be non-negative")
        if self.strategy != "seeded_stratified_hash_v1":
            raise JourneySpecialistUnsupportedError("split strategy is unsupported")
        _digest(self.dataset_digest, "split dataset_digest")
        _digest(self.assignment_digest, "split assignment_digest")
        counts = _count_mapping(self.split_counts, "split_counts")
        if set(counts) != {"train", "validation", "holdout"}:
            raise JourneySpecialistError("split counts are incomplete")
        labels = _nested_count_mapping(self.label_counts, "label_counts")
        if any(set(value) != set(counts) for value in labels.values()):
            raise JourneySpecialistError("per-label split counts are incomplete")
        _contract_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(self, "split_counts", counts)
        object.__setattr__(self, "label_counts", labels)

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free split manifest."""

        return {
            "assignment_digest": self.assignment_digest,
            "compatibility_policy": self.compatibility_policy,
            "dataset_digest": self.dataset_digest,
            "label_counts": {
                key: dict(value) for key, value in self.label_counts.items()
            },
            "schema_version": self.schema_version,
            "seed": self.seed,
            "split_counts": dict(self.split_counts),
            "strategy": self.strategy,
            "task": self.task,
        }


@dataclass(frozen=True, slots=True)
class GpuSpendEntry:
    """One reserved or completed GPU spend line without billing credentials."""

    run_id: str
    task: str
    estimated_cost_usd: float
    estimated_gpu_hours: float
    hourly_cost_usd: float
    actual_cost_usd: float | None = None
    actual_gpu_hours: float | None = None
    state: str = "reserved"

    def __post_init__(self) -> None:
        _controlled(self.run_id, "run_id")
        if self.task not in JOURNEY_SPECIALIST_TASKS:
            raise JourneySpecialistUnsupportedError("spend task is unsupported")
        for name in (
            "estimated_cost_usd",
            "estimated_gpu_hours",
            "hourly_cost_usd",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise JourneySpecialistError("spend estimates must be positive")
            object.__setattr__(self, name, value)
        if self.state not in {"reserved", "completed", "cancelled"}:
            raise JourneySpecialistError("spend state is unsupported")
        if self.state == "completed":
            if self.actual_cost_usd is None or self.actual_gpu_hours is None:
                raise JourneySpecialistError("completed spend requires actual totals")
        if (self.actual_cost_usd is None) != (self.actual_gpu_hours is None):
            raise JourneySpecialistError(
                "actual spend fields must be supplied together"
            )
        if self.actual_cost_usd is not None:
            actual_cost = _finite(self.actual_cost_usd, "actual_cost_usd")
            actual_hours = _finite(self.actual_gpu_hours, "actual_gpu_hours")
            if actual_cost < 0 or actual_hours < 0:
                raise JourneySpecialistError("actual spend cannot be negative")
            object.__setattr__(self, "actual_cost_usd", actual_cost)
            object.__setattr__(self, "actual_gpu_hours", actual_hours)

    @property
    def committed_cost_usd(self) -> float:
        """Return the cap-relevant cost for this run."""

        if self.actual_cost_usd is not None:
            return self.actual_cost_usd
        if self.state == "cancelled":
            return 0.0
        return self.estimated_cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Return one deterministic spend line."""

        return {
            "actual_cost_usd": self.actual_cost_usd,
            "actual_gpu_hours": self.actual_gpu_hours,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_gpu_hours": self.estimated_gpu_hours,
            "hourly_cost_usd": self.hourly_cost_usd,
            "run_id": self.run_id,
            "state": self.state,
            "task": self.task,
        }


@dataclass(frozen=True, slots=True)
class GpuSpendLedger:
    """Immutable shared ledger enforcing the aggregate USD 1,000 cap."""

    entries: tuple[GpuSpendEntry, ...] = ()
    cap_usd: float = JOURNEY_SPECIALIST_GPU_BUDGET_USD
    schema_version: str = JOURNEY_SPECIALIST_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_SPECIALIST_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        cap = _finite(self.cap_usd, "GPU spend cap")
        if cap <= 0 or cap > JOURNEY_SPECIALIST_GPU_BUDGET_USD:
            raise JourneySpecialistDeniedError(
                "GPU spend cap exceeds the hard USD 1000 maximum"
            )
        entries = tuple(self.entries)
        if any(not isinstance(item, GpuSpendEntry) for item in entries):
            raise TypeError("spend entries must be GpuSpendEntry values")
        run_ids = tuple(item.run_id for item in entries)
        if len(run_ids) != len(set(run_ids)):
            raise JourneySpecialistConflictError("spend ledger run IDs must be unique")
        if sum(item.committed_cost_usd for item in entries) > cap + 1e-9:
            raise JourneySpecialistDeniedError("GPU spend ledger exceeds its cap")
        _contract_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(
            self, "entries", tuple(sorted(entries, key=lambda x: x.run_id))
        )
        object.__setattr__(self, "cap_usd", cap)

    @property
    def committed_cost_usd(self) -> float:
        """Return reserved or actual spend counted against the cap."""

        return round(sum(item.committed_cost_usd for item in self.entries), 6)

    @property
    def remaining_cost_usd(self) -> float:
        """Return uncommitted budget."""

        return round(self.cap_usd - self.committed_cost_usd, 6)

    @property
    def digest(self) -> str:
        """Return the canonical ledger digest."""

        return sha256_digest(canonical_json(self.to_dict()))

    def reserve(self, run_id: str, recipe: SpecialistRecipe) -> "GpuSpendLedger":
        """Reserve one predeclared run or deny the entire update."""

        if any(item.run_id == run_id for item in self.entries):
            raise JourneySpecialistConflictError("GPU spend run is already reserved")
        entry = GpuSpendEntry(
            run_id=run_id,
            task=recipe.task,
            estimated_cost_usd=recipe.estimated_cost_usd,
            estimated_gpu_hours=recipe.estimated_gpu_hours,
            hourly_cost_usd=recipe.hourly_cost_usd,
        )
        return replace(self, entries=(*self.entries, entry))

    def record_actual(
        self,
        run_id: str,
        *,
        actual_cost_usd: float,
        actual_gpu_hours: float,
    ) -> "GpuSpendLedger":
        """Replace one reservation with actual totals while retaining the cap."""

        actual_cost_usd = _finite(actual_cost_usd, "actual_cost_usd")
        actual_gpu_hours = _finite(actual_gpu_hours, "actual_gpu_hours")
        found = False
        updated: list[GpuSpendEntry] = []
        for entry in self.entries:
            if entry.run_id != run_id:
                updated.append(entry)
                continue
            found = True
            if entry.state != "reserved":
                if (
                    entry.state == "completed"
                    and entry.actual_cost_usd == actual_cost_usd
                    and entry.actual_gpu_hours == actual_gpu_hours
                ):
                    updated.append(entry)
                    continue
                raise JourneySpecialistConflictError(
                    "terminal GPU spend cannot be rewritten"
                )
            updated.append(
                replace(
                    entry,
                    actual_cost_usd=actual_cost_usd,
                    actual_gpu_hours=actual_gpu_hours,
                    state="completed",
                )
            )
        if not found:
            raise JourneySpecialistConflictError("GPU spend run is not reserved")
        return replace(self, entries=tuple(updated))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned ledger."""

        return {
            "cap_usd": self.cap_usd,
            "committed_cost_usd": self.committed_cost_usd,
            "compatibility_policy": self.compatibility_policy,
            "entries": [item.to_dict() for item in self.entries],
            "remaining_cost_usd": self.remaining_cost_usd,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class SpecialistRunManifest:
    """Pinned dry-run or completed-run provenance for one specialist."""

    run_id: str
    pack_id: str
    task: str
    code_revision: str
    dataset: SpecialistDatasetLineage
    split: SpecialistSplitManifest
    recipe: SpecialistRecipe
    ledger_digest: str
    hardware: str
    estimated_cost_usd: float
    actual_cost_usd: float | None
    metrics_digest: str | None
    failure_slices_digest: str | None
    model_card_digest: str | None
    artifact_digests: Mapping[str, str]
    promotion_decision: str
    promotion_reason: str
    dry_run: bool
    schema_version: str = JOURNEY_SPECIALIST_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_SPECIALIST_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _controlled(self.run_id, "run_id")
        _controlled(self.pack_id, "pack_id")
        if self.task not in JOURNEY_SPECIALIST_TASKS:
            raise JourneySpecialistUnsupportedError("run task is unsupported")
        if _SHA_RE.fullmatch(self.code_revision) is None:
            raise JourneySpecialistError("code revision must be an immutable hash")
        if not isinstance(self.dataset, SpecialistDatasetLineage):
            raise TypeError("run dataset must be SpecialistDatasetLineage")
        if not isinstance(self.split, SpecialistSplitManifest):
            raise TypeError("run split must be SpecialistSplitManifest")
        if not isinstance(self.recipe, SpecialistRecipe):
            raise TypeError("run recipe must be SpecialistRecipe")
        if self.task != self.split.task or self.task != self.recipe.task:
            raise JourneySpecialistConflictError("run task provenance is inconsistent")
        _digest(self.ledger_digest, "ledger_digest")
        _controlled(self.hardware, "hardware")
        estimated = _finite(self.estimated_cost_usd, "estimated_cost_usd")
        if estimated != self.recipe.estimated_cost_usd:
            raise JourneySpecialistConflictError("run cost differs from recipe")
        if self.actual_cost_usd is not None:
            actual = _finite(self.actual_cost_usd, "actual_cost_usd")
            if actual < 0:
                raise JourneySpecialistError("actual cost cannot be negative")
            object.__setattr__(self, "actual_cost_usd", actual)
        for name in (
            "metrics_digest",
            "failure_slices_digest",
            "model_card_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)
        artifacts = _digest_mapping(self.artifact_digests, "artifact_digests")
        if self.promotion_decision not in {"not_evaluated", "hold", "promote"}:
            raise JourneySpecialistError("promotion decision is unsupported")
        _controlled(self.promotion_reason, "promotion_reason")
        if type(self.dry_run) is not bool:
            raise JourneySpecialistError("dry_run must be boolean")
        if self.dry_run and (
            self.actual_cost_usd is not None
            or self.metrics_digest is not None
            or artifacts
            or self.promotion_decision != "not_evaluated"
        ):
            raise JourneySpecialistConflictError(
                "dry-run manifests cannot claim execution or promotion evidence"
            )
        _contract_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(self, "artifact_digests", artifacts)
        object.__setattr__(self, "estimated_cost_usd", estimated)

    @property
    def digest(self) -> str:
        """Return the canonical run-manifest digest."""

        return sha256_digest(canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        """Return complete, raw-free training provenance."""

        return {
            "actual_cost_usd": self.actual_cost_usd,
            "artifact_digests": dict(self.artifact_digests),
            "code_revision": self.code_revision,
            "compatibility_policy": self.compatibility_policy,
            "dataset": self.dataset.to_dict(),
            "dry_run": self.dry_run,
            "estimated_cost_usd": self.estimated_cost_usd,
            "failure_slices_digest": self.failure_slices_digest,
            "hardware": self.hardware,
            "ledger_digest": self.ledger_digest,
            "metrics_digest": self.metrics_digest,
            "model_card_digest": self.model_card_digest,
            "pack_id": self.pack_id,
            "promotion_decision": self.promotion_decision,
            "promotion_reason": self.promotion_reason,
            "recipe": self.recipe.to_dict(),
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "split": self.split.to_dict(),
            "task": self.task,
        }


@dataclass(frozen=True, slots=True)
class SpecialistPackDryRun:
    """Offline dry-run result for all four specialist recipes."""

    pack_id: str
    plan_digest: str
    dataset_digest: str
    ledger: GpuSpendLedger
    manifests: tuple[SpecialistRunManifest, ...]
    schema_version: str = JOURNEY_SPECIALIST_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_SPECIALIST_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _controlled(self.pack_id, "pack_id")
        _digest(self.plan_digest, "plan_digest")
        _digest(self.dataset_digest, "dataset_digest")
        if not isinstance(self.ledger, GpuSpendLedger):
            raise TypeError("dry-run ledger must be GpuSpendLedger")
        manifests = tuple(self.manifests)
        if len(manifests) != len(JOURNEY_SPECIALIST_TASKS) or any(
            not isinstance(item, SpecialistRunManifest) for item in manifests
        ):
            raise JourneySpecialistError("dry run must contain every task manifest")
        if {item.task for item in manifests} != set(JOURNEY_SPECIALIST_TASKS):
            raise JourneySpecialistError("dry-run task manifests are incomplete")
        if any(item.ledger_digest != self.ledger.digest for item in manifests):
            raise JourneySpecialistConflictError("manifest ledger digests disagree")
        _contract_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(
            self, "manifests", tuple(sorted(manifests, key=lambda item: item.task))
        )

    @property
    def digest(self) -> str:
        """Return the canonical dry-run digest."""

        return sha256_digest(canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned dry-run report."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "dataset_digest": self.dataset_digest,
            "ledger": self.ledger.to_dict(),
            "manifests": [item.to_dict() for item in self.manifests],
            "pack_id": self.pack_id,
            "plan_digest": self.plan_digest,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return deterministic compact JSON."""

        return canonical_json(self.to_dict())


def load_journey_specialist_pack(
    payload: str | bytes | Mapping[str, Any] | None = None,
) -> SpecialistPackPlan:
    """Load the bundled or caller-supplied specialist-pack configuration."""

    if payload is None:
        resource = resources.files(JOURNEY_SPECIALIST_CONFIG_PACKAGE).joinpath(
            JOURNEY_SPECIALIST_CONFIG_NAME
        )
        payload = resource.read_bytes()
    if isinstance(payload, Mapping):
        value = dict(payload)
    else:
        try:
            raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
            if not isinstance(raw, str):
                raise TypeError
            value = json.loads(raw, object_pairs_hook=_no_duplicate_object)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JourneySpecialistError(
                "specialist pack is not valid UTF-8 JSON"
            ) from exc
    return SpecialistPackPlan.from_dict(_mapping(value, "specialist pack"))


def load_journey_specialist_examples(
    lineage: SpecialistDatasetLineage,
    payload: bytes | None = None,
) -> tuple[SpecialistTrainingExample, ...]:
    """Load a digest-checked local dataset without network resolution."""

    if payload is None:
        if not lineage.bundled:
            raise JourneySpecialistUnsupportedError(
                "user-supplied dataset bytes are required"
            )
        resource = resources.files(JOURNEY_SPECIALIST_DATA_PACKAGE).joinpath(
            lineage.resource
        )
        payload = resource.read_bytes()
    if sha256_digest(payload) != lineage.content_digest:
        raise JourneySpecialistConflictError("dataset content digest does not match")
    examples: list[SpecialistTrainingExample] = []
    try:
        text = payload.decode("utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            value = json.loads(line, object_pairs_hook=_no_duplicate_object)
            data = _mapping(value, "training example")
            _exact_keys(
                data,
                {"example_id", "label", "subgroups", "task", "text"},
                set(),
                "training example",
            )
            examples.append(
                SpecialistTrainingExample(
                    example_id=_text(data["example_id"], "example_id"),
                    task=_text(data["task"], "example task"),
                    label=_text(data["label"], "example label"),
                    text=_text(data["text"], "example text"),
                    subgroups=_mapping(data["subgroups"], "example subgroups"),
                )
            )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise JourneySpecialistError("training dataset is malformed") from exc
    if len(examples) != lineage.record_count:
        raise JourneySpecialistConflictError("dataset record count does not match")
    ids = tuple(item.example_id for item in examples)
    if len(ids) != len(set(ids)):
        raise JourneySpecialistConflictError("training example IDs must be unique")
    return tuple(sorted(examples, key=lambda item: item.example_id))


def build_specialist_split(
    recipe: SpecialistRecipe,
    examples: Sequence[SpecialistTrainingExample],
    *,
    dataset_digest: str,
) -> StoreResult[SpecialistSplitManifest]:
    """Build a deterministic stratified train/validation/holdout manifest."""

    _digest(dataset_digest, "dataset_digest")
    selected = tuple(item for item in examples if item.task == recipe.task)
    if not selected:
        return StoreResult.outcome(StoreState.UNKNOWN, "task_examples_missing")
    observed_labels = {item.label for item in selected}
    if observed_labels != set(recipe.labels):
        return StoreResult.outcome(StoreState.CONFLICT, "task_labels_mismatch")
    grouped: dict[str, list[SpecialistTrainingExample]] = defaultdict(list)
    for item in selected:
        grouped[item.label].append(item)
    if any(len(items) < 3 for items in grouped.values()):
        return StoreResult.outcome(StoreState.PARTIAL, "task_split_insufficient")

    assignments: dict[str, str] = {}
    label_counts: dict[str, dict[str, int]] = {}
    total_counts = {"train": 0, "validation": 0, "holdout": 0}
    for label, items in sorted(grouped.items()):
        ranked = sorted(
            items,
            key=lambda item: sha256_digest(
                f"{recipe.seed}:{recipe.task}:{label}:{item.example_id}"
            ),
        )
        counts = {"train": len(ranked) - 2, "validation": 1, "holdout": 1}
        label_counts[label] = counts
        for index, item in enumerate(ranked):
            split = "holdout" if index == 0 else "validation" if index == 1 else "train"
            assignments[item.example_id] = split
            total_counts[split] += 1
    manifest = SpecialistSplitManifest(
        task=recipe.task,
        seed=recipe.seed,
        strategy="seeded_stratified_hash_v1",
        dataset_digest=dataset_digest,
        assignment_digest=sha256_digest(canonical_json(assignments)),
        split_counts=total_counts,
        label_counts=label_counts,
    )
    return StoreResult.success(manifest)


def dry_run_journey_specialist_pack(
    *,
    code_revision: str,
    config_payload: str | bytes | Mapping[str, Any] | None = None,
    dataset_payload: bytes | None = None,
    hardware: str = "single_gpu_24gb",
) -> StoreResult[SpecialistPackDryRun]:
    """Resolve every manifest and reservation without GPU or network access."""

    try:
        plan = load_journey_specialist_pack(config_payload)
        examples = load_journey_specialist_examples(plan.dataset, dataset_payload)
        _controlled(hardware, "hardware")
        if _SHA_RE.fullmatch(code_revision) is None:
            raise JourneySpecialistError("code revision must be an immutable hash")
        ledger = GpuSpendLedger(cap_usd=plan.aggregate_gpu_budget_usd)
        splits: dict[str, SpecialistSplitManifest] = {}
        run_ids: dict[str, str] = {}
        for recipe in plan.recipes:
            split_result = build_specialist_split(
                recipe,
                examples,
                dataset_digest=plan.dataset.content_digest,
            )
            if not split_result.ok or split_result.value is None:
                return StoreResult.outcome(
                    split_result.state, split_result.code or "split_failed"
                )
            splits[recipe.task] = split_result.value
            run_id = derived_opaque_id(
                "run",
                plan.pack_id,
                recipe.digest,
                split_result.value.assignment_digest,
                code_revision,
            )
            run_ids[recipe.task] = run_id
            ledger = ledger.reserve(run_id, recipe)
        manifests = tuple(
            SpecialistRunManifest(
                run_id=run_ids[recipe.task],
                pack_id=plan.pack_id,
                task=recipe.task,
                code_revision=code_revision,
                dataset=plan.dataset,
                split=splits[recipe.task],
                recipe=recipe,
                ledger_digest=ledger.digest,
                hardware=hardware,
                estimated_cost_usd=recipe.estimated_cost_usd,
                actual_cost_usd=None,
                metrics_digest=None,
                failure_slices_digest=None,
                model_card_digest=None,
                artifact_digests={},
                promotion_decision="not_evaluated",
                promotion_reason="dry_run_only",
                dry_run=True,
            )
            for recipe in plan.recipes
        )
        return StoreResult.success(
            SpecialistPackDryRun(
                pack_id=plan.pack_id,
                plan_digest=plan.digest,
                dataset_digest=plan.dataset.content_digest,
                ledger=ledger,
                manifests=manifests,
            )
        )
    except JourneySpecialistDeniedError:
        return StoreResult.outcome(StoreState.DENIED, "specialist_plan_denied")
    except JourneySpecialistConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "specialist_plan_conflict")
    except JourneySpecialistUnsupportedError:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "specialist_plan_unsupported"
        )
    except (JourneySpecialistError, OSError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "specialist_plan_invalid")


def load_journey_specialist_schema() -> dict[str, Any]:
    """Load the bundled specialist training/evaluation JSON Schema."""

    resource = resources.files(JOURNEY_SPECIALIST_SCHEMA_PACKAGE).joinpath(
        f"{JOURNEY_SPECIALIST_SCHEMA_NAME}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_journey_specialist_pack_card(
    plan: SpecialistPackPlan,
    dry_run: SpecialistPackDryRun,
) -> str:
    """Render a raw-free pack card for review before any training spend."""

    if plan.pack_id != dry_run.pack_id or plan.digest != dry_run.plan_digest:
        raise JourneySpecialistConflictError("pack card inputs do not match")
    lines = [
        f"# {plan.pack_id}",
        "",
        "Status: no trained artifact is promoted by this dry run.",
        "",
        "## Reproducibility",
        "",
        f"- Plan digest: `{plan.digest}`",
        f"- Dataset digest: `{plan.dataset.content_digest}`",
        f"- Split strategy: `seeded_stratified_hash_v1`",
        f"- Shared GPU budget: `${plan.aggregate_gpu_budget_usd:.2f}`",
        f"- Reserved estimate: `${dry_run.ledger.committed_cost_usd:.2f}`",
        "",
        "## Specialists",
        "",
        "| Task | Backbone revision | Adapter | Precision | Cost ceiling | Exports |",
        "|---|---|---|---|---:|---|",
    ]
    for recipe in plan.recipes:
        lines.append(
            f"| {recipe.task} | `{recipe.backbone.revision}` | "
            f"{recipe.adapter.method} | {recipe.mixed_precision} | "
            f"${recipe.estimated_cost_usd:.2f} | "
            f"{', '.join(recipe.export_formats)} |"
        )
    lines.extend(
        [
            "",
            "## Promotion policy",
            "",
            "Every candidate remains on hold until a frozen holdout proves that "
            "it beats its named baseline and meets per-class recall, calibration, "
            "abstention coverage, subgroup, and quantized-delta floors.",
            "",
            "## Limitations",
            "",
            "This pack is not clinically validated and cannot authorize diagnosis, "
            "treatment, enrollment, outreach, ordering, or another patient-care "
            "action. Credentialed datasets remain user supplied and eval only.",
        ]
    )
    return "\n".join(lines) + "\n"


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise JourneySpecialistError("JSON contains a duplicate key")
        value[key] = item
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JourneySpecialistError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise JourneySpecialistError(f"{name} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    if not required.issubset(value) or set(value) - required - optional:
        raise JourneySpecialistError(f"{name} fields do not match contract")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise JourneySpecialistError(f"{name} must be non-empty text")
    return value


def _controlled(value: Any, name: str) -> str:
    text = _text(value, name)
    if _CONTROLLED_RE.fullmatch(text) is None:
        raise JourneySpecialistError(f"{name} must be controlled")
    return text


def _version(value: Any, name: str) -> str:
    text = _text(value, name)
    if _VERSION_RE.fullmatch(text) is None:
        raise JourneySpecialistError(f"{name} must be semantic")
    return text


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if _DIGEST_RE.fullmatch(text) is None:
        raise JourneySpecialistError(f"{name} must be a sha256 digest")
    return text


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise JourneySpecialistError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise JourneySpecialistError(f"{name} must be boolean")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JourneySpecialistError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise JourneySpecialistError(f"{name} must be a finite number")
    return number


def _unit_interval(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0 or number > 1:
        raise JourneySpecialistError(f"{name} must be between zero and one")
    return number


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise JourneySpecialistError(f"{name} must be an array")
    return tuple(_text(item, name) for item in value)


def _controlled_values(value: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(_controlled(item, name) for item in value)
    if len(items) != len(set(items)):
        raise JourneySpecialistConflictError(f"{name} must be unique")
    return tuple(sorted(items))


def _count_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, int]:
    data = _mapping(value, name)
    counts: dict[str, int] = {}
    for key, item in data.items():
        _controlled(key, name)
        count = _integer(item, name)
        if count < 0:
            raise JourneySpecialistError(f"{name} cannot contain negative counts")
        counts[key] = count
    return MappingProxyType(dict(sorted(counts.items())))


def _nested_count_mapping(
    value: Mapping[str, Mapping[str, Any]], name: str
) -> Mapping[str, Mapping[str, int]]:
    data = _mapping(value, name)
    result = {
        _controlled(key, name): _count_mapping(_mapping(item, name), name)
        for key, item in data.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


def _digest_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, str]:
    data = _mapping(value, name)
    result = {_controlled(key, name): _digest(item, name) for key, item in data.items()}
    return MappingProxyType(dict(sorted(result.items())))


def _contract_version(schema_version: str, compatibility_policy: str) -> None:
    if schema_version != JOURNEY_SPECIALIST_SCHEMA_VERSION:
        raise JourneySpecialistUnsupportedError(
            "specialist schema version is unsupported"
        )
    if compatibility_policy != JOURNEY_SPECIALIST_COMPATIBILITY_POLICY:
        raise JourneySpecialistUnsupportedError(
            "specialist compatibility policy is unsupported"
        )


__all__ = [
    "JOURNEY_SPECIALIST_ADAPTER_METHODS",
    "JOURNEY_SPECIALIST_ARCHITECTURES",
    "JOURNEY_SPECIALIST_COMPATIBILITY_POLICY",
    "JOURNEY_SPECIALIST_CONFIG_NAME",
    "JOURNEY_SPECIALIST_EXPORT_FORMATS",
    "JOURNEY_SPECIALIST_GPU_BUDGET_USD",
    "JOURNEY_SPECIALIST_PERMISSIVE_LICENSES",
    "JOURNEY_SPECIALIST_PRECISIONS",
    "JOURNEY_SPECIALIST_QUANTIZATION_TARGETS",
    "JOURNEY_SPECIALIST_SCHEMA_NAME",
    "JOURNEY_SPECIALIST_SCHEMA_VERSION",
    "JOURNEY_SPECIALIST_TASKS",
    "GpuSpendEntry",
    "GpuSpendLedger",
    "JourneySpecialistConflictError",
    "JourneySpecialistDeniedError",
    "JourneySpecialistError",
    "JourneySpecialistUnsupportedError",
    "SpecialistAdapter",
    "SpecialistBackbone",
    "SpecialistDatasetLineage",
    "SpecialistPackDryRun",
    "SpecialistPackPlan",
    "SpecialistRecipe",
    "SpecialistRunManifest",
    "SpecialistSplitManifest",
    "SpecialistStopRule",
    "SpecialistTrainingExample",
    "build_specialist_split",
    "dry_run_journey_specialist_pack",
    "load_journey_specialist_examples",
    "load_journey_specialist_pack",
    "load_journey_specialist_schema",
    "render_journey_specialist_pack_card",
]
