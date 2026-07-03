"""Versioned training recipe validation and dry-run entrypoint.

The package owns the reproducible recipe contract for future training jobs.
It validates the intended configuration surface, records stable hashes, and
imports the core runtime helpers that training code must reuse. It does not
launch GPU training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.core.anonymizer import AnonymizerConfig
from openmed.core.decoding import build_label_info
from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_entity_merger import merge_entities_with_semantic_units
from openmed.eval.metrics import compute_leakage_rate, compute_recall_slices

CONFIG_SCHEMA_VERSION = "openmed.training.recipe.v1"
QLORA_CONFIG_SCHEMA_VERSION = "openmed.training.qlora_recipe.v1"
QLORA_SMOKE_RESULT_SCHEMA_VERSION = "openmed.training.qlora_smoke_result.v1"
MAX_LORA_TRAINABLE_RATIO = 0.015
CONFIG_DIR = Path(__file__).with_name("configs")
QLORA_SMOKE_PRESET = "qlora_smoke"
PRESET_BY_MODE = {
    "A": "tiny_distill",
    "B": "laptop_lora",
    "C": "large_teacher",
}
MODE_BY_PRESET = {preset: mode for mode, preset in PRESET_BY_MODE.items()}

_REQUIRED_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "preset_name",
        "mode",
        "backbone",
        "dapt",
        "lora",
        "label_set_ref",
        "loss",
        "hard_negatives_required",
        "output_tier",
        "quantization",
        "seed",
    }
)
_OPTIONAL_ROOT_FIELDS = frozenset({"head_contract"})
_QLORA_REQUIRED_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "preset_name",
        "base_model",
        "adapter",
        "heads",
        "data",
        "gates",
        "evidence",
        "seed",
    }
)
_ALLOWED_QLORA_HEADS = frozenset({"token_classification", "generative_pii"})
_REMOTE_REF_PREFIXES = ("http://", "https://", "s3://", "gs://", "hf://")


class RecipeConfigError(ValueError):
    """Raised when a recipe config violates the versioned schema."""


class NetworkEgressBlockedError(RuntimeError):
    """Raised when QLoRA dry-run code attempts network egress."""


@dataclass(frozen=True)
class BackboneConfig:
    model_ref: str
    revision: str
    family: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model_ref": self.model_ref,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class DaptConfig:
    corpus_ref: str
    sequence_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_ref": self.corpus_ref,
            "sequence_length": self.sequence_length,
        }


@dataclass(frozen=True)
class LoraConfig:
    target_trainable_ratio: float
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "dropout": self.dropout,
            "rank": self.rank,
            "target_modules": list(self.target_modules),
            "target_trainable_ratio": self.target_trainable_ratio,
        }


@dataclass(frozen=True)
class LossConfig:
    name: str
    focal_gamma: float
    class_weighted: bool
    class_weighting: str
    critical_label_weight: float
    critical_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_weighted": self.class_weighted,
            "class_weighting": self.class_weighting,
            "critical_label_weight": self.critical_label_weight,
            "critical_labels": list(self.critical_labels),
            "focal_gamma": self.focal_gamma,
            "name": self.name,
        }


@dataclass(frozen=True)
class QuantizationConfig:
    default: str
    allow_fp32_fallback: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_fp32_fallback": self.allow_fp32_fallback,
            "default": self.default,
        }


@dataclass(frozen=True)
class TrainingRecipeConfig:
    schema_version: str
    preset_name: str
    mode: str
    backbone: BackboneConfig
    dapt: DaptConfig
    lora: LoraConfig
    label_set_ref: str
    loss: LossConfig
    hard_negatives_required: bool
    output_tier: str
    quantization: QuantizationConfig
    seed: int
    head_contract: str | None = None

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> "TrainingRecipeConfig":
        data = _copy_mapping(raw_config)
        _reject_root_shape(data)

        schema_version = _require_str(data, "schema_version")
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise RecipeConfigError(
                f"schema_version must be {CONFIG_SCHEMA_VERSION!r}, got {schema_version!r}"
            )

        mode = _require_str(data, "mode").upper()
        if mode not in PRESET_BY_MODE:
            raise RecipeConfigError("mode must be one of A, B, or C")

        preset_name = _require_str(data, "preset_name")
        expected_preset = PRESET_BY_MODE[mode]
        if preset_name != expected_preset:
            raise RecipeConfigError(
                f"preset_name {preset_name!r} does not match mode {mode!r}"
            )

        hard_negatives_required = data["hard_negatives_required"]
        if hard_negatives_required is not True:
            raise RecipeConfigError("hard_negatives_required must be present and true")

        seed = _require_int(data, "seed")
        if seed < 0:
            raise RecipeConfigError("seed must be a non-negative integer")

        config = cls(
            schema_version=schema_version,
            preset_name=preset_name,
            mode=mode,
            backbone=_parse_backbone(_require_mapping(data, "backbone")),
            dapt=_parse_dapt(_require_mapping(data, "dapt")),
            lora=_parse_lora(_require_mapping(data, "lora")),
            label_set_ref=_require_str(data, "label_set_ref"),
            loss=_parse_loss(_require_mapping(data, "loss")),
            hard_negatives_required=hard_negatives_required,
            output_tier=_require_str(data, "output_tier"),
            quantization=_parse_quantization(_require_mapping(data, "quantization")),
            seed=seed,
            head_contract=_optional_str(data, "head_contract"),
        )
        _validate_output_tier(config.output_tier)
        return config

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backbone": self.backbone.to_dict(),
            "dapt": self.dapt.to_dict(),
            "hard_negatives_required": self.hard_negatives_required,
            "label_set_ref": self.label_set_ref,
            "lora": self.lora.to_dict(),
            "loss": self.loss.to_dict(),
            "mode": self.mode,
            "output_tier": self.output_tier,
            "preset_name": self.preset_name,
            "quantization": self.quantization.to_dict(),
            "schema_version": self.schema_version,
            "seed": self.seed,
        }
        if self.head_contract is not None:
            payload["head_contract"] = self.head_contract
        return payload


@dataclass(frozen=True)
class RuntimeDependencies:
    anonymizer_config: type[AnonymizerConfig]
    merger: Callable[..., Any]
    decoder: Callable[..., Any]

    def module_names(self) -> dict[str, str]:
        return {
            "anonymizer": self.anonymizer_config.__module__,
            "decoding": self.decoder.__module__,
            "merger": self.merger.__module__,
        }


@dataclass(frozen=True)
class DryRunResult:
    preset_name: str
    mode: str
    seed: int
    config_hash: str
    output_tier: str
    quant_default: str
    dependency_modules: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "dependency_modules": dict(self.dependency_modules),
            "mode": self.mode,
            "output_tier": self.output_tier,
            "preset_name": self.preset_name,
            "quant_default": self.quant_default,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class QloraBaseModelConfig:
    """Local-only base checkpoint loading contract for QLoRA smoke runs."""

    model_ref: str
    revision: str
    load_in_4bit: bool
    quantization_type: str
    double_quant: bool
    local_files_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "double_quant": self.double_quant,
            "load_in_4bit": self.load_in_4bit,
            "local_files_only": self.local_files_only,
            "model_ref": self.model_ref,
            "quantization_type": self.quantization_type,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class QloraAdapterConfig:
    """Low-rank adapter shape for a QLoRA smoke recipe."""

    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    target_trainable_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "dropout": self.dropout,
            "rank": self.rank,
            "target_modules": list(self.target_modules),
            "target_trainable_ratio": self.target_trainable_ratio,
        }


@dataclass(frozen=True)
class QloraDataConfig:
    """Local synthetic smoke corpus reference for QLoRA gating."""

    corpus_path: str
    heldout_split: str
    local_files_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_path": self.corpus_path,
            "heldout_split": self.heldout_split,
            "local_files_only": self.local_files_only,
        }


@dataclass(frozen=True)
class QloraGateConfig:
    """Recall and leakage budgets for QLoRA promotion."""

    max_recall_drop: float
    max_leakage_increase: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_leakage_increase": self.max_leakage_increase,
            "max_recall_drop": self.max_recall_drop,
        }


@dataclass(frozen=True)
class QloraEvidenceConfig:
    """Evidence destination and signing requirement for QLoRA runs."""

    output_dir: str
    signing_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "signing_required": self.signing_required,
        }


@dataclass(frozen=True)
class QloraRecipeConfig:
    """Validated QLoRA smoke recipe that never imports optional trainers."""

    schema_version: str
    preset_name: str
    base_model: QloraBaseModelConfig
    adapter: QloraAdapterConfig
    heads: tuple[str, ...]
    data: QloraDataConfig
    gates: QloraGateConfig
    evidence: QloraEvidenceConfig
    seed: int
    config_path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        raw_config: Mapping[str, Any],
        *,
        config_path: str | Path | None = None,
    ) -> "QloraRecipeConfig":
        data = _copy_mapping(raw_config)
        _reject_qlora_root_shape(data)

        schema_version = _require_str(data, "schema_version")
        if schema_version != QLORA_CONFIG_SCHEMA_VERSION:
            raise RecipeConfigError(
                "schema_version must be "
                f"{QLORA_CONFIG_SCHEMA_VERSION!r}, got {schema_version!r}"
            )

        seed = _require_int(data, "seed")
        if seed < 0:
            raise RecipeConfigError("seed must be a non-negative integer")

        config = cls(
            schema_version=schema_version,
            preset_name=_require_str(data, "preset_name"),
            base_model=_parse_qlora_base_model(_require_mapping(data, "base_model")),
            adapter=_parse_qlora_adapter(_require_mapping(data, "adapter")),
            heads=_parse_qlora_heads(data["heads"]),
            data=_parse_qlora_data(_require_mapping(data, "data")),
            gates=_parse_qlora_gates(_require_mapping(data, "gates")),
            evidence=_parse_qlora_evidence(_require_mapping(data, "evidence")),
            seed=seed,
            config_path=Path(config_path) if config_path is not None else None,
        )
        _validate_qlora_local_only(config)
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter.to_dict(),
            "base_model": self.base_model.to_dict(),
            "data": self.data.to_dict(),
            "evidence": self.evidence.to_dict(),
            "gates": self.gates.to_dict(),
            "heads": list(self.heads),
            "preset_name": self.preset_name,
            "schema_version": self.schema_version,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class QloraEntityDelta:
    """Per-entity recall and leakage deltas between base and adapted outputs."""

    label: str
    base_recall: float
    adapted_recall: float
    recall_delta: float
    base_leakage: float
    adapted_leakage: float
    leakage_delta: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapted_leakage": self.adapted_leakage,
            "adapted_recall": self.adapted_recall,
            "base_leakage": self.base_leakage,
            "base_recall": self.base_recall,
            "label": self.label,
            "leakage_delta": self.leakage_delta,
            "passed": self.passed,
            "recall_delta": self.recall_delta,
        }


@dataclass(frozen=True)
class QloraSmokeResult:
    """PHI-free evidence returned by a deterministic QLoRA smoke dry run."""

    preset_name: str
    seed: int
    config_hash: str
    corpus_hash: str
    example_count: int
    gold_span_count: int
    heads: tuple[str, ...]
    labels: tuple[str, ...]
    base_overall_recall: float
    adapted_overall_recall: float
    base_overall_leakage: float
    adapted_overall_leakage: float
    per_entity_deltas: tuple[QloraEntityDelta, ...]
    gate_passed: bool
    gate_violations: tuple[Mapping[str, Any], ...]
    network_egress_blocked: bool
    egress_probe_blocked: bool
    contains_raw_phi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapted_overall_leakage": self.adapted_overall_leakage,
            "adapted_overall_recall": self.adapted_overall_recall,
            "base_overall_leakage": self.base_overall_leakage,
            "base_overall_recall": self.base_overall_recall,
            "config_hash": self.config_hash,
            "contains_raw_phi": self.contains_raw_phi,
            "corpus_hash": self.corpus_hash,
            "example_count": self.example_count,
            "gate": {
                "passed": self.gate_passed,
                "violations": [dict(item) for item in self.gate_violations],
            },
            "gold_span_count": self.gold_span_count,
            "heads": list(self.heads),
            "labels": list(self.labels),
            "network": {
                "egress_blocked": self.network_egress_blocked,
                "egress_probe_blocked": self.egress_probe_blocked,
                "local_files_only": True,
            },
            "per_entity_deltas": {
                delta.label: delta.to_dict() for delta in self.per_entity_deltas
            },
            "preset_name": self.preset_name,
            "schema_version": QLORA_SMOKE_RESULT_SCHEMA_VERSION,
            "seed": self.seed,
        }


def load_preset(mode_or_preset: str) -> TrainingRecipeConfig:
    """Load and validate a committed preset by mode (A/B/C) or preset name."""

    preset_name = _preset_name_for(mode_or_preset)
    path = CONFIG_DIR / f"{preset_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Training preset not found: {path}")
    return TrainingRecipeConfig.from_mapping(load_config_file(path))


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a preset file using the repository's supported YAML subset."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    return _parse_yaml_subset(text)


def load_qlora_preset(name: str = QLORA_SMOKE_PRESET) -> QloraRecipeConfig:
    """Load a committed QLoRA smoke preset without importing trainer backends."""

    if _has_remote_prefix(name):
        raise RecipeConfigError("qlora preset must be a local preset name or path")
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"QLoRA preset not found: {path}")
    return QloraRecipeConfig.from_mapping(load_config_file(path), config_path=path)


def dry_run_recipe(config: TrainingRecipeConfig) -> DryRunResult:
    """Validate a recipe and return deterministic dry-run metadata."""

    dependencies = runtime_dependencies()
    return DryRunResult(
        preset_name=config.preset_name,
        mode=config.mode,
        seed=config.seed,
        config_hash=config_hash(config),
        output_tier=config.output_tier,
        quant_default=config.quantization.default,
        dependency_modules=dependencies.module_names(),
    )


def run_recipe(mode_or_preset: str, *, dry_run: bool = True) -> DryRunResult:
    """Single recipe entrypoint for modes A, B, and C."""

    config = load_preset(mode_or_preset)
    if dry_run:
        return dry_run_recipe(config)
    raise NotImplementedError(
        "Training execution is out of scope for this recipe package"
    )


def run_qlora_smoke(
    preset_or_path: str | Path | QloraRecipeConfig = QLORA_SMOKE_PRESET,
    *,
    dry_run: bool = True,
    seeded_regression: bool = False,
    egress_probe: Callable[[], Any] | None = None,
) -> QloraSmokeResult:
    """Run the offline QLoRA smoke recipe and return PHI-free gate evidence.

    The smoke path is deterministic and intentionally lightweight. It validates
    the QLoRA config, reads only a local synthetic JSONL corpus, computes
    per-entity recall/leakage deltas, and evaluates the promotion gate. It does
    not import or execute optional training backends.
    """

    if not dry_run:
        raise NotImplementedError(
            "Full QLoRA training is tracked separately from the smoke recipe"
        )

    config = (
        preset_or_path
        if isinstance(preset_or_path, QloraRecipeConfig)
        else load_qlora_preset(str(preset_or_path))
    )
    corpus_path = _resolve_qlora_corpus_path(config)

    egress_probe_blocked = False
    with block_network_egress():
        if egress_probe is not None:
            try:
                egress_probe()
            except NetworkEgressBlockedError:
                egress_probe_blocked = True
            else:
                raise RecipeConfigError("egress_probe was not blocked")
        examples = _load_qlora_smoke_corpus(corpus_path)

    metrics = _compute_qlora_smoke_metrics(
        examples,
        use_regression_predictions=seeded_regression,
    )
    deltas, violations = _qlora_entity_deltas(
        labels=metrics["labels"],
        base_recall=metrics["base_recall"].by_label,
        adapted_recall=metrics["adapted_recall"].by_label,
        base_leakage=metrics["base_leakage"].by_label,
        adapted_leakage=metrics["adapted_leakage"].by_label,
        gates=config.gates,
    )

    return QloraSmokeResult(
        preset_name=config.preset_name,
        seed=config.seed,
        config_hash=config_hash(config.to_dict()),
        corpus_hash=_file_sha256(corpus_path),
        example_count=len(examples),
        gold_span_count=metrics["gold_span_count"],
        heads=config.heads,
        labels=metrics["labels"],
        base_overall_recall=metrics["base_recall"].overall,
        adapted_overall_recall=metrics["adapted_recall"].overall,
        base_overall_leakage=metrics["base_leakage"].overall,
        adapted_overall_leakage=metrics["adapted_leakage"].overall,
        per_entity_deltas=deltas,
        gate_passed=not violations,
        gate_violations=tuple(violations),
        network_egress_blocked=True,
        egress_probe_blocked=egress_probe_blocked,
    )


def config_hash(config: TrainingRecipeConfig | Mapping[str, Any]) -> str:
    """Return a deterministic hash over the canonical recipe config."""

    payload = (
        config.to_dict()
        if isinstance(config, TrainingRecipeConfig)
        else _copy_mapping(config)
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def runtime_dependencies() -> RuntimeDependencies:
    """Return imported core helpers that future training code must reuse."""

    return RuntimeDependencies(
        anonymizer_config=AnonymizerConfig,
        merger=merge_entities_with_semantic_units,
        decoder=build_label_info,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an OpenMed training recipe preset."
    )
    parser.add_argument(
        "mode", choices=tuple(PRESET_BY_MODE) + tuple(PRESET_BY_MODE.values())
    )
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args(argv)

    result = run_recipe(args.mode, dry_run=args.dry_run)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


def _reject_root_shape(data: Mapping[str, Any]) -> None:
    keys = set(data)
    missing = sorted(_REQUIRED_ROOT_FIELDS - keys)
    if missing:
        raise RecipeConfigError(f"missing required field(s): {', '.join(missing)}")
    unknown = sorted(keys - _REQUIRED_ROOT_FIELDS - _OPTIONAL_ROOT_FIELDS)
    if unknown:
        raise RecipeConfigError(f"unknown field(s): {', '.join(unknown)}")


def _reject_qlora_root_shape(data: Mapping[str, Any]) -> None:
    keys = set(data)
    missing = sorted(_QLORA_REQUIRED_ROOT_FIELDS - keys)
    if missing:
        raise RecipeConfigError(f"missing required field(s): {', '.join(missing)}")
    unknown = sorted(keys - _QLORA_REQUIRED_ROOT_FIELDS)
    if unknown:
        raise RecipeConfigError(f"unknown field(s): {', '.join(unknown)}")


def _parse_backbone(data: Mapping[str, Any]) -> BackboneConfig:
    _require_exact_fields(data, "backbone", {"model_ref", "revision", "family"})
    return BackboneConfig(
        model_ref=_require_str(data, "model_ref"),
        revision=_require_str(data, "revision"),
        family=_require_str(data, "family"),
    )


def _parse_dapt(data: Mapping[str, Any]) -> DaptConfig:
    _require_exact_fields(data, "dapt", {"corpus_ref", "sequence_length"})
    sequence_length = _require_int(data, "sequence_length")
    if sequence_length <= 0:
        raise RecipeConfigError("dapt.sequence_length must be positive")
    return DaptConfig(
        corpus_ref=_require_str(data, "corpus_ref"),
        sequence_length=sequence_length,
    )


def _parse_lora(data: Mapping[str, Any]) -> LoraConfig:
    _require_exact_fields(
        data,
        "lora",
        {"target_trainable_ratio", "rank", "alpha", "dropout", "target_modules"},
    )
    ratio = _require_float(data, "target_trainable_ratio")
    if ratio <= 0 or ratio >= MAX_LORA_TRAINABLE_RATIO:
        raise RecipeConfigError("lora.target_trainable_ratio must be > 0 and < 0.015")
    rank = _require_int(data, "rank")
    alpha = _require_int(data, "alpha")
    dropout = _require_float(data, "dropout")
    if rank <= 0 or alpha <= 0:
        raise RecipeConfigError("lora.rank and lora.alpha must be positive")
    if dropout < 0 or dropout >= 1:
        raise RecipeConfigError("lora.dropout must be >= 0 and < 1")
    target_modules = _require_str_tuple(data, "target_modules")
    if not target_modules:
        raise RecipeConfigError("lora.target_modules must not be empty")
    return LoraConfig(
        target_trainable_ratio=ratio,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )


def _parse_loss(data: Mapping[str, Any]) -> LossConfig:
    _require_exact_fields(
        data,
        "loss",
        {
            "name",
            "focal_gamma",
            "class_weighted",
            "class_weighting",
            "critical_label_weight",
            "critical_labels",
        },
    )
    name = _require_str(data, "name")
    if name != "focal_class_weighted":
        raise RecipeConfigError("loss.name must be 'focal_class_weighted'")
    class_weighted = data["class_weighted"]
    if class_weighted is not True:
        raise RecipeConfigError("loss.class_weighted must be true")
    focal_gamma = _require_float(data, "focal_gamma")
    if focal_gamma <= 0:
        raise RecipeConfigError("loss.focal_gamma must be positive")
    critical_label_weight = _require_float(data, "critical_label_weight")
    if critical_label_weight <= 1:
        raise RecipeConfigError("loss.critical_label_weight must be greater than 1")
    critical_labels = _require_str_tuple(data, "critical_labels")
    if not critical_labels:
        raise RecipeConfigError("loss.critical_labels must not be empty")
    unknown_labels = sorted(set(critical_labels) - CANONICAL_LABELS)
    if unknown_labels:
        raise RecipeConfigError(
            f"loss.critical_labels contains unknown label(s): {', '.join(unknown_labels)}"
        )
    return LossConfig(
        name=name,
        focal_gamma=focal_gamma,
        class_weighted=class_weighted,
        class_weighting=_require_str(data, "class_weighting"),
        critical_label_weight=critical_label_weight,
        critical_labels=critical_labels,
    )


def _parse_quantization(data: Mapping[str, Any]) -> QuantizationConfig:
    _require_exact_fields(data, "quantization", {"default", "allow_fp32_fallback"})
    allow_fp32_fallback = data["allow_fp32_fallback"]
    if not isinstance(allow_fp32_fallback, bool):
        raise RecipeConfigError("quantization.allow_fp32_fallback must be a boolean")
    return QuantizationConfig(
        default=_require_str(data, "default"),
        allow_fp32_fallback=allow_fp32_fallback,
    )


def _parse_qlora_base_model(data: Mapping[str, Any]) -> QloraBaseModelConfig:
    _require_exact_fields(
        data,
        "base_model",
        {
            "model_ref",
            "revision",
            "load_in_4bit",
            "quantization_type",
            "double_quant",
            "local_files_only",
        },
    )
    load_in_4bit = data["load_in_4bit"]
    double_quant = data["double_quant"]
    local_files_only = data["local_files_only"]
    if load_in_4bit is not True:
        raise RecipeConfigError("base_model.load_in_4bit must be true")
    if not isinstance(double_quant, bool):
        raise RecipeConfigError("base_model.double_quant must be a boolean")
    if local_files_only is not True:
        raise RecipeConfigError("base_model.local_files_only must be true")
    quantization_type = _require_str(data, "quantization_type")
    if quantization_type not in {"nf4", "fp4"}:
        raise RecipeConfigError("base_model.quantization_type must be nf4 or fp4")
    return QloraBaseModelConfig(
        model_ref=_require_str(data, "model_ref"),
        revision=_require_str(data, "revision"),
        load_in_4bit=load_in_4bit,
        quantization_type=quantization_type,
        double_quant=double_quant,
        local_files_only=local_files_only,
    )


def _parse_qlora_adapter(data: Mapping[str, Any]) -> QloraAdapterConfig:
    _require_exact_fields(
        data,
        "adapter",
        {"rank", "alpha", "dropout", "target_modules", "target_trainable_ratio"},
    )
    rank = _require_int(data, "rank")
    alpha = _require_int(data, "alpha")
    dropout = _require_float(data, "dropout")
    ratio = _require_float(data, "target_trainable_ratio")
    if rank <= 0 or alpha <= 0:
        raise RecipeConfigError("adapter.rank and adapter.alpha must be positive")
    if dropout < 0 or dropout >= 1:
        raise RecipeConfigError("adapter.dropout must be >= 0 and < 1")
    if ratio <= 0 or ratio >= MAX_LORA_TRAINABLE_RATIO:
        raise RecipeConfigError(
            "adapter.target_trainable_ratio must be > 0 and < 0.015"
        )
    target_modules = _require_str_tuple(data, "target_modules")
    if not target_modules:
        raise RecipeConfigError("adapter.target_modules must not be empty")
    return QloraAdapterConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        target_trainable_ratio=ratio,
    )


def _parse_qlora_heads(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RecipeConfigError("heads must be a list of strings")
    heads = tuple(str(item) for item in value if isinstance(item, str) and item)
    if len(heads) != len(value):
        raise RecipeConfigError("heads must be a list of non-empty strings")
    unknown = sorted(set(heads) - _ALLOWED_QLORA_HEADS)
    if unknown:
        raise RecipeConfigError(
            f"heads contains unknown value(s): {', '.join(unknown)}"
        )
    if not heads:
        raise RecipeConfigError("heads must not be empty")
    return heads


def _parse_qlora_data(data: Mapping[str, Any]) -> QloraDataConfig:
    _require_exact_fields(
        data,
        "data",
        {"corpus_path", "heldout_split", "local_files_only"},
    )
    local_files_only = data["local_files_only"]
    if local_files_only is not True:
        raise RecipeConfigError("data.local_files_only must be true")
    return QloraDataConfig(
        corpus_path=_require_str(data, "corpus_path"),
        heldout_split=_require_str(data, "heldout_split"),
        local_files_only=local_files_only,
    )


def _parse_qlora_gates(data: Mapping[str, Any]) -> QloraGateConfig:
    _require_exact_fields(data, "gates", {"max_recall_drop", "max_leakage_increase"})
    max_recall_drop = _require_float(data, "max_recall_drop")
    max_leakage_increase = _require_float(data, "max_leakage_increase")
    if max_recall_drop < 0:
        raise RecipeConfigError("gates.max_recall_drop must be non-negative")
    if max_leakage_increase < 0:
        raise RecipeConfigError("gates.max_leakage_increase must be non-negative")
    return QloraGateConfig(
        max_recall_drop=max_recall_drop,
        max_leakage_increase=max_leakage_increase,
    )


def _parse_qlora_evidence(data: Mapping[str, Any]) -> QloraEvidenceConfig:
    _require_exact_fields(data, "evidence", {"output_dir", "signing_required"})
    signing_required = data["signing_required"]
    if signing_required is not True:
        raise RecipeConfigError("evidence.signing_required must be true")
    return QloraEvidenceConfig(
        output_dir=_require_str(data, "output_dir"),
        signing_required=signing_required,
    )


def _validate_output_tier(output_tier: str) -> None:
    if output_tier not in {"tiny", "laptop", "teacher"}:
        raise RecipeConfigError("output_tier must be one of tiny, laptop, or teacher")


def _validate_qlora_local_only(config: QloraRecipeConfig) -> None:
    if _has_remote_prefix(config.base_model.model_ref):
        raise RecipeConfigError("base_model.model_ref must be local-only")
    if _has_remote_prefix(config.data.corpus_path):
        raise RecipeConfigError("data.corpus_path must be local-only")


def _has_remote_prefix(value: str) -> bool:
    return value.casefold().startswith(_REMOTE_REF_PREFIXES)


def _resolve_qlora_corpus_path(config: QloraRecipeConfig) -> Path:
    raw_path = Path(config.data.corpus_path)
    if raw_path.is_absolute():
        path = raw_path
    else:
        base_dir = config.config_path.parent if config.config_path else CONFIG_DIR
        path = base_dir / raw_path
    if not path.exists():
        raise FileNotFoundError(f"QLoRA smoke corpus not found: {path}")
    return path


def _load_qlora_smoke_corpus(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecipeConfigError(
                    f"invalid QLoRA smoke JSONL at line {line_number}"
                ) from exc
            _validate_qlora_smoke_example(item, line_number=line_number)
            examples.append(item)
    if not examples:
        raise RecipeConfigError("QLoRA smoke corpus must contain at least one example")
    return examples


def _validate_qlora_smoke_example(
    item: Mapping[str, Any],
    *,
    line_number: int,
) -> None:
    required = {
        "id",
        "language",
        "text",
        "gold_spans",
        "base_predictions",
        "adapted_predictions",
        "regressed_predictions",
    }
    missing = sorted(required - set(item))
    if missing:
        raise RecipeConfigError(
            f"QLoRA smoke example line {line_number} missing: {', '.join(missing)}"
        )
    if not isinstance(item["text"], str) or not item["text"]:
        raise RecipeConfigError(f"QLoRA smoke example line {line_number} needs text")
    for key in (
        "gold_spans",
        "base_predictions",
        "adapted_predictions",
        "regressed_predictions",
    ):
        spans = item[key]
        if not isinstance(spans, list):
            raise RecipeConfigError(
                f"QLoRA smoke example line {line_number} {key} must be a list"
            )
        for span in spans:
            _validate_qlora_span(
                span,
                text=item["text"],
                language=str(item["language"]),
                line_number=line_number,
                key=key,
            )


def _validate_qlora_span(
    span: Any,
    *,
    text: str,
    language: str,
    line_number: int,
    key: str,
) -> None:
    if not isinstance(span, Mapping):
        raise RecipeConfigError(f"QLoRA smoke line {line_number} {key} span invalid")
    label = span.get("label")
    start = span.get("start")
    end = span.get("end")
    if label not in CANONICAL_LABELS:
        raise RecipeConfigError(
            f"QLoRA smoke line {line_number} {key} has unknown label {label!r}"
        )
    if not isinstance(start, int) or not isinstance(end, int) or start >= end:
        raise RecipeConfigError(
            f"QLoRA smoke line {line_number} {key} span offsets are invalid"
        )
    if start < 0 or end > len(text):
        raise RecipeConfigError(
            f"QLoRA smoke line {line_number} {key} span exceeds text length"
        )
    span_language = span.get("language", language)
    if not isinstance(span_language, str) or not span_language:
        raise RecipeConfigError(
            f"QLoRA smoke line {line_number} {key} span language is invalid"
        )


def _compute_qlora_smoke_metrics(
    examples: Sequence[Mapping[str, Any]],
    *,
    use_regression_predictions: bool,
) -> dict[str, Any]:
    gold_spans: list[dict[str, Any]] = []
    base_predictions: list[dict[str, Any]] = []
    adapted_predictions: list[dict[str, Any]] = []
    labels: set[str] = set()
    offset = 0
    adapted_key = (
        "regressed_predictions" if use_regression_predictions else "adapted_predictions"
    )

    for item in examples:
        language = str(item["language"])
        text = str(item["text"])
        gold = _offset_qlora_spans(item["gold_spans"], offset, language)
        base = _offset_qlora_spans(item["base_predictions"], offset, language)
        adapted = _offset_qlora_spans(item[adapted_key], offset, language)
        gold_spans.extend(gold)
        base_predictions.extend(base)
        adapted_predictions.extend(adapted)
        labels.update(span["label"] for span in gold)
        offset += len(text) + 1

    return {
        "adapted_leakage": compute_leakage_rate(gold_spans, adapted_predictions),
        "adapted_recall": compute_recall_slices(gold_spans, adapted_predictions),
        "base_leakage": compute_leakage_rate(gold_spans, base_predictions),
        "base_recall": compute_recall_slices(gold_spans, base_predictions),
        "gold_span_count": len(gold_spans),
        "labels": tuple(sorted(labels)),
    }


def _offset_qlora_spans(
    spans: Sequence[Mapping[str, Any]],
    offset: int,
    default_language: str,
) -> list[dict[str, Any]]:
    return [
        {
            "end": int(span["end"]) + offset,
            "label": str(span["label"]),
            "language": str(span.get("language") or default_language),
            "start": int(span["start"]) + offset,
        }
        for span in spans
    ]


def _qlora_entity_deltas(
    *,
    labels: Sequence[str],
    base_recall: Mapping[str, float],
    adapted_recall: Mapping[str, float],
    base_leakage: Mapping[str, float],
    adapted_leakage: Mapping[str, float],
    gates: QloraGateConfig,
) -> tuple[tuple[QloraEntityDelta, ...], list[dict[str, Any]]]:
    deltas: list[QloraEntityDelta] = []
    violations: list[dict[str, Any]] = []
    for label in sorted(labels):
        recall_delta = float(adapted_recall[label]) - float(base_recall[label])
        leakage_delta = float(adapted_leakage[label]) - float(base_leakage[label])
        recall_passed = recall_delta >= -gates.max_recall_drop
        leakage_passed = leakage_delta <= gates.max_leakage_increase
        passed = recall_passed and leakage_passed
        deltas.append(
            QloraEntityDelta(
                label=label,
                base_recall=float(base_recall[label]),
                adapted_recall=float(adapted_recall[label]),
                recall_delta=recall_delta,
                base_leakage=float(base_leakage[label]),
                adapted_leakage=float(adapted_leakage[label]),
                leakage_delta=leakage_delta,
                passed=passed,
            )
        )
        if not recall_passed:
            violations.append(
                {
                    "budget": gates.max_recall_drop,
                    "delta": recall_delta,
                    "label": label,
                    "metric": "recall_delta",
                }
            )
        if not leakage_passed:
            violations.append(
                {
                    "budget": gates.max_leakage_increase,
                    "delta": leakage_delta,
                    "label": label,
                    "metric": "leakage_delta",
                }
            )
    return tuple(deltas), violations


@contextmanager
def block_network_egress():
    """Temporarily fail socket creation in local-only QLoRA smoke paths."""

    original_socket = socket.socket
    original_create_connection = socket.create_connection

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise NetworkEgressBlockedError(
            "network egress is disabled for QLoRA smoke runs"
        )

    socket.socket = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_exact_fields(data: Mapping[str, Any], name: str, fields: set[str]) -> None:
    keys = set(data)
    missing = sorted(fields - keys)
    unknown = sorted(keys - fields)
    if missing:
        raise RecipeConfigError(
            f"{name} missing required field(s): {', '.join(missing)}"
        )
    if unknown:
        raise RecipeConfigError(f"{name} has unknown field(s): {', '.join(unknown)}")


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data[key]
    if not isinstance(value, Mapping):
        raise RecipeConfigError(f"{key} must be a mapping")
    return value


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise RecipeConfigError(f"{key} must be a non-empty string")
    return value


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RecipeConfigError(f"{key} must be a non-empty string when present")
    return value


def _require_int(data: Mapping[str, Any], key: str) -> int:
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecipeConfigError(f"{key} must be an integer")
    return value


def _require_float(data: Mapping[str, Any], key: str) -> float:
    value = data[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RecipeConfigError(f"{key} must be a number")
    return float(value)


def _require_str_tuple(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RecipeConfigError(f"{key} must be a list of strings")
    if not all(isinstance(item, str) and item for item in value):
        raise RecipeConfigError(f"{key} must be a list of non-empty strings")
    return tuple(value)


def _preset_name_for(mode_or_preset: str) -> str:
    key = mode_or_preset.strip()
    mode = key.upper()
    if mode in PRESET_BY_MODE:
        return PRESET_BY_MODE[mode]
    if key in MODE_BY_PRESET:
        return key
    raise RecipeConfigError("preset must be mode A/B/C or a committed preset name")


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2:
            raise RecipeConfigError(f"invalid indentation at line {line_number}")
        stripped = raw_line.strip()
        key, separator, value = stripped.partition(":")
        if not separator:
            raise RecipeConfigError(f"expected key/value pair at line {line_number}")
        key = key.strip()
        if not key:
            raise RecipeConfigError(f"empty key at line {line_number}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise RecipeConfigError(f"invalid indentation at line {line_number}")
        parent = stack[-1][1]
        scalar = value.strip()
        if not scalar:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(scalar)

    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
