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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.core.anonymizer import AnonymizerConfig
from openmed.core.decoding import build_label_info
from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_entity_merger import merge_entities_with_semantic_units

CONFIG_SCHEMA_VERSION = "openmed.training.recipe.v1"
MAX_LORA_TRAINABLE_RATIO = 0.015
CONFIG_DIR = Path(__file__).with_name("configs")
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


class RecipeConfigError(ValueError):
    """Raised when a recipe config violates the versioned schema."""


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


def _validate_output_tier(output_tier: str) -> None:
    if output_tier not in {"tiny", "laptop", "teacher"}:
        raise RecipeConfigError("output_tier must be one of tiny, laptop, or teacher")


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
