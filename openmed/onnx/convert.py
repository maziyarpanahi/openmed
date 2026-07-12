"""Convert token-classification checkpoints to ONNX runtime artifacts.

Use the default profile for a standard fp32 ONNX export plus the WebGPU fp16
variant. Use ``--profile android`` for Android ONNX Runtime Mobile artifacts:
``model.onnx`` and ``model_fp16.onnx`` are exported at the fixed Android opset
with `input_ids` and `attention_mask` inputs, optional `token_type_ids`, a
`logits` output, and named dynamic `batch` and `sequence` axes. The fixed opset
keeps the graph stable for the later `.ort` mobile-format conversion step.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.hf_publish import publish_artifact
from openmed.eval.quant_delta import evaluate_onnx_logit_parity
from openmed.mlx.artifact import find_tokenizer_files
from openmed.onnx.android_profile import (
    ANDROID_FP16_FILENAME,
    ANDROID_ONNX_FORMAT,
    ANDROID_ONNX_OPSET,
    ANDROID_PROFILE_NAME,
    export_android_fp16,
    validate_android_profile,
)
from openmed.onnx.openvino_export import (
    OPENVINO_FORMAT,
    OPENVINO_IR_DIRNAME,
    OPENVINO_PROFILE_NAME,
    build_synthetic_token_inputs,
    export_openvino_ir,
    run_onnx_reference_logits,
)
from openmed.onnx.ort_mobile import (
    ORT_ANDROID_FORMAT,
    convert_android_onnx_to_ort,
)
from openmed.onnx.quantize_int8 import (
    INT8_ONNX_FILENAME,
    ONNX_INT8_FORMAT,
    apply_int8_recall_certification,
    int8_artifact_metadata,
    quantize_dynamic_int8,
    write_int8_recall_delta_report,
)
from openmed.onnx.transformersjs import (
    DEFAULT_BUNDLE_DIRNAME,
    TRANSFORMERSJS_FORMAT,
    export_transformersjs_bundle,
)
from openmed.processing.tokenizer_cache import get_tokenizer_with_loader

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "openmed-onnx.json"
MANIFEST_FORMAT = "openmed-onnx"
MANIFEST_VERSION = 1
DEFAULT_ONNX_FILENAME = "model.onnx"
DEFAULT_UNOPTIMIZED_ONNX_FILENAME = "model.unoptimized.onnx"
DEFAULT_WEBGPU_FILENAME = "model.webgpu.onnx"
DEFAULT_ONNX_OPSET = 18
DEFAULT_PROFILE_NAME = "default"
DEFAULT_SAMPLE_TEXT = "Patient John Doe visited the clinic on 2024-01-15."
MINIMUM_TOKEN_CLASSIFICATION_OPSET = 18
DEFAULT_SHAPE_BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_DYNAMIC_VALIDATION_LENGTHS = DEFAULT_SHAPE_BUCKETS
DEFAULT_EXECUTION_PROVIDERS = ("CPUExecutionProvider",)

_FUSION_OPTION_ATTRS = {
    "attention_fusion": ("enable_attention",),
    "layer_norm_fusion": ("enable_layer_norm", "enable_layer_norm_fusion"),
    "gelu_fusion": ("enable_gelu", "enable_gelu_approximation"),
}


@dataclass(frozen=True)
class ExportArtifact:
    """One exported runtime artifact inside an ONNX conversion output."""

    format: str
    path: Path
    precision: str
    metadata: Mapping[str, Any] | None = None

    def to_manifest(self, root: Path) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "path": self.path.relative_to(root).as_posix(),
            "precision": self.precision,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class OnnxConversionResult:
    """Paths and manifest data produced by an ONNX conversion."""

    output_dir: Path
    manifest_path: Path
    artifacts: tuple[ExportArtifact, ...]

    @property
    def formats(self) -> list[str]:
        return _dedupe_keep_order([artifact.format for artifact in self.artifacts])


@dataclass(frozen=True)
class OnnxOptimizationConfig:
    """Post-export graph optimization settings for token-classification ONNX."""

    constant_folding: bool = True
    attention_fusion: bool = True
    layer_norm_fusion: bool = True
    gelu_fusion: bool = True
    required_latency_improvement: float = 0.20
    providers: tuple[str, ...] = DEFAULT_EXECUTION_PROVIDERS

    @property
    def enabled_passes(self) -> tuple[str, ...]:
        return tuple(name for name, enabled in self._passes().items() if enabled)

    @property
    def disabled_passes(self) -> tuple[str, ...]:
        return tuple(name for name, enabled in self._passes().items() if not enabled)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "passes": self._passes(),
            "enabled_passes": list(self.enabled_passes),
            "disabled_passes": list(self.disabled_passes),
            "required_latency_improvement": self.required_latency_improvement,
            "providers": list(self.providers),
        }

    def _passes(self) -> dict[str, bool]:
        return {
            "constant_folding": self.constant_folding,
            "attention_fusion": self.attention_fusion,
            "layer_norm_fusion": self.layer_norm_fusion,
            "gelu_fusion": self.gelu_fusion,
        }


@dataclass(frozen=True)
class ShapeBucketConfig:
    """Sequence-length buckets used for dynamic-shape ONNX runtime feeds."""

    buckets: tuple[int, ...] = DEFAULT_SHAPE_BUCKETS
    min_length: int = 8
    max_length: int = 2048

    def __post_init__(self) -> None:
        cleaned = tuple(sorted({int(bucket) for bucket in self.buckets}))
        if not cleaned:
            raise ValueError("shape bucket config must include at least one bucket")
        if any(bucket <= 0 for bucket in cleaned):
            raise ValueError("shape buckets must be positive integers")
        object.__setattr__(self, "buckets", cleaned)
        if self.min_length <= 0 or self.max_length < self.min_length:
            raise ValueError("invalid dynamic-shape validation length range")

    def bucket_for(self, sequence_length: int) -> int:
        length = int(sequence_length)
        if length <= 0:
            raise ValueError("sequence length must be positive")
        for bucket in self.buckets:
            if length <= bucket:
                return bucket
        return length

    def to_manifest(self) -> dict[str, Any]:
        return {
            "strategy": "next_configured_bucket_or_exact",
            "axes": {"input": ["batch", "sequence"], "output": ["batch", "sequence"]},
            "buckets": list(self.buckets),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "overflow": "exact_length",
        }


def export_onnx(
    model_id: str,
    output_path: str | Path,
    *,
    max_seq_length: int = 512,
    opset: int | None = None,
    profile: str = DEFAULT_PROFILE_NAME,
    cache_dir: str | None = None,
    sample_text: str = DEFAULT_SAMPLE_TEXT,
) -> Path:
    """Export a Hugging Face token-classification checkpoint to ONNX."""

    profile = _normalize_profile(profile)
    resolved_opset = _resolve_opset(profile=profile, opset=opset)
    if resolved_opset < MINIMUM_TOKEN_CLASSIFICATION_OPSET:
        raise ValueError(
            "token-classification ONNX export requires opset "
            f">= {MINIMUM_TOKEN_CLASSIFICATION_OPSET}; got {resolved_opset}"
        )

    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "torch and transformers are required for ONNX export. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer_with_loader(
        model_id,
        AutoTokenizer.from_pretrained,
        cache_dir=cache_dir,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    model.to(dtype=torch.float32)
    model.eval()

    sample = tokenizer(
        [sample_text, sample_text],
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    use_token_type_ids = "token_type_ids" in sample

    class TokenClassificationWrapper(torch.nn.Module):
        def __init__(self, base_model: Any, include_token_type_ids: bool) -> None:
            super().__init__()
            self.base_model = base_model
            self.include_token_type_ids = include_token_type_ids

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any = None,
        ) -> Any:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if self.include_token_type_ids:
                inputs["token_type_ids"] = token_type_ids
            return self.base_model(**inputs).logits

    wrapper = TokenClassificationWrapper(model, use_token_type_ids)
    wrapper.eval()

    input_names = ["input_ids", "attention_mask"]
    example_inputs: tuple[Any, ...] = (sample["input_ids"], sample["attention_mask"])
    if use_token_type_ids:
        input_names.append("token_type_ids")
        example_inputs = (*example_inputs, sample["token_type_ids"])

    with torch.no_grad():
        export_kwargs = {
            "input_names": input_names,
            "output_names": ["logits"],
            "opset_version": resolved_opset,
            "do_constant_folding": True,
        }
        export_kwargs.update(
            _dynamic_export_kwargs(
                torch,
                export_fn=torch.onnx.export,
                input_names=input_names,
                max_seq_length=max_seq_length,
                profile=profile,
            )
        )
        with _onnx_export_compatibility(model, torch):
            torch.onnx.export(
                wrapper,
                example_inputs,
                str(output_path),
                **export_kwargs,
            )

    _consolidate_external_onnx_data(output_path)
    _check_onnx_model(output_path)
    return output_path


def export_webgpu(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    keep_io_types: bool = True,
) -> Path:
    """Create a WebGPU-targeted fp16 ONNX artifact from a fp32 ONNX model."""

    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        raise ImportError(
            "onnx and onnxruntime are required for WebGPU export. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path), load_external_data=False)
    fp16_model = convert_float_to_float16(
        str(onnx_path),
        keep_io_types=keep_io_types,
    )
    _copy_missing_model_metadata(fp16_model, model)
    onnx.save(fp16_model, str(output_path))
    _check_onnx_model(output_path)
    return output_path


def convert(
    model_id: str,
    output_dir: str | Path,
    *,
    include_webgpu: bool = True,
    include_transformersjs: bool = False,
    include_int8: bool = True,
    max_seq_length: int = 512,
    opset: int | None = None,
    profile: str = DEFAULT_PROFILE_NAME,
    optimize_onnx: bool = True,
    optimization_config: OnnxOptimizationConfig | None = None,
    shape_bucket_config: ShapeBucketConfig | None = None,
    validation_texts: Sequence[str] | None = None,
    validation_lengths: Sequence[int] | None = None,
    cache_dir: str | None = None,
    sample_text: str = DEFAULT_SAMPLE_TEXT,
    eval_suite_path: str | Path | None = None,
    recall_delta_report_path: str | Path | None = None,
    publish_to_hub: bool = False,
    publish_repo_id: str | None = None,
    publish_org: str = "OpenMed",
    publish_version: int = 1,
    publish_manifest_path: str | Path | None = None,
    publish_token_env: str = "HF_WRITE_TOKEN",
    publish_private: bool = False,
    publish_overwrite_existing: bool = False,
) -> OnnxConversionResult:
    """Export ONNX artifacts and optionally publish the resulting directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = _normalize_profile(profile)
    shape_bucket_config = shape_bucket_config or ShapeBucketConfig()
    optimization_config = _normalise_optimization_config(optimization_config)
    should_optimize_onnx = optimize_onnx and profile not in {
        ANDROID_PROFILE_NAME,
        OPENVINO_PROFILE_NAME,
    }

    optimization_manifest: dict[str, Any]
    validation_manifest: dict[str, Any] | None = None
    operator_fallbacks: list[dict[str, Any]] = []
    if should_optimize_onnx:
        unoptimized_path = export_onnx(
            model_id,
            output_dir / DEFAULT_UNOPTIMIZED_ONNX_FILENAME,
            max_seq_length=max_seq_length,
            opset=opset,
            profile=profile,
            cache_dir=cache_dir,
        )
        onnx_path = output_dir / DEFAULT_ONNX_FILENAME
    else:
        onnx_path = export_onnx(
            model_id,
            output_dir / DEFAULT_ONNX_FILENAME,
            max_seq_length=max_seq_length,
            opset=opset,
            profile=profile,
            cache_dir=cache_dir,
        )
        unoptimized_path = None
        optimization_manifest = {
            "enabled": False,
            "minimum_opset": MINIMUM_TOKEN_CLASSIFICATION_OPSET,
        }

    config, tokenizer_files = save_source_assets(
        model_id,
        output_dir,
        cache_dir=cache_dir,
        max_seq_length=max_seq_length,
        require_id2label=profile in {ANDROID_PROFILE_NAME, OPENVINO_PROFILE_NAME},
    )
    if should_optimize_onnx and unoptimized_path is not None:
        optimization_manifest = optimize_onnx_graph(
            unoptimized_path,
            onnx_path,
            config=optimization_config,
            model_type=_model_type_for_optimizer(config),
            minimum_opset=MINIMUM_TOKEN_CLASSIFICATION_OPSET,
        )
        validation_manifest = validate_optimized_onnx_export(
            unoptimized_path,
            onnx_path,
            model_id=model_id,
            cache_dir=cache_dir,
            config=config,
            shape_bucket_config=shape_bucket_config,
            optimization_config=optimization_config,
            validation_texts=validation_texts,
            validation_lengths=validation_lengths,
        )
        operator_fallbacks = list(validation_manifest.get("operator_fallbacks") or [])

    int8_path: Path | None = None
    int8_validation_metadata: Mapping[str, Any] | None = None
    if profile == ANDROID_PROFILE_NAME:
        android_validation = validate_android_profile(onnx_path)
        android_fp16_path = export_android_fp16(
            onnx_path,
            output_dir / ANDROID_FP16_FILENAME,
            validate=False,
        )
        android_fp16_validation = validate_android_profile(android_fp16_path)
        if include_int8:
            int8_path = quantize_dynamic_int8(
                onnx_path,
                output_dir / INT8_ONNX_FILENAME,
            )
            int8_validation = validate_android_profile(int8_path)
            int8_validation_metadata = int8_validation.to_metadata()
        artifacts = [
            ExportArtifact(
                format=ANDROID_ONNX_FORMAT,
                path=onnx_path,
                precision="float32",
                metadata=android_validation.to_metadata(),
            ),
            ExportArtifact(
                format=ANDROID_ONNX_FORMAT,
                path=android_fp16_path,
                precision="float16",
                metadata=android_fp16_validation.to_metadata(),
            ),
        ]
        if int8_path is not None:
            artifacts.append(
                ExportArtifact(
                    format=ONNX_INT8_FORMAT,
                    path=int8_path,
                    precision="int8",
                    metadata=int8_artifact_metadata(
                        validation_metadata=int8_validation_metadata,
                    ),
                )
            )
        ort_result = convert_android_onnx_to_ort(
            onnx_path,
            output_dir=output_dir,
            validation=android_validation,
        )
        if not ort_result.skipped and ort_result.ort_path is not None:
            artifacts.append(
                ExportArtifact(
                    format=ORT_ANDROID_FORMAT,
                    path=ort_result.ort_path,
                    precision="float32",
                    metadata=ort_result.to_metadata(output_dir),
                )
            )
    elif profile == OPENVINO_PROFILE_NAME:
        config, tokenizer_files = save_source_assets(
            model_id,
            output_dir,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            require_id2label=True,
        )
        tokenizer = get_tokenizer_with_loader(
            model_id,
            _transformers_tokenizer_loader(cache_dir=cache_dir),
            cache_dir=cache_dir,
        )
        synthetic_inputs = build_synthetic_token_inputs(
            tokenizer,
            text=sample_text,
            max_seq_length=max_seq_length,
        )
        reference_logits = run_onnx_reference_logits(onnx_path, synthetic_inputs)
        openvino_result = export_openvino_ir(
            onnx_path,
            output_dir / OPENVINO_IR_DIRNAME,
            sample_inputs=synthetic_inputs,
            reference_logits=reference_logits,
            id2label=config["id2label"],
            sample_text=sample_text,
        )
        artifacts = [
            ExportArtifact(
                format=OPENVINO_FORMAT,
                path=openvino_result.model_xml_path,
                precision="float32",
                metadata=openvino_result.to_metadata(output_dir),
            )
        ]
    else:
        if eval_suite_path is not None:
            raise ValueError("eval_suite_path requires profile='android'.")
        artifacts = [
            ExportArtifact(format="onnx", path=onnx_path, precision="float32"),
        ]

    if profile not in {ANDROID_PROFILE_NAME, OPENVINO_PROFILE_NAME} and include_webgpu:
        webgpu_path = export_webgpu(
            onnx_path,
            output_dir / DEFAULT_WEBGPU_FILENAME,
        )
        artifacts.append(
            ExportArtifact(format="webgpu", path=webgpu_path, precision="float16")
        )

    if include_transformersjs:
        transformersjs_result = export_transformersjs_bundle(
            output_dir,
            output_dir / DEFAULT_BUNDLE_DIRNAME,
            tokenizer_source=output_dir,
            config_source=output_dir / "config.json",
            update_manifest=False,
        )
        artifacts.append(
            ExportArtifact(
                format=TRANSFORMERSJS_FORMAT,
                path=transformersjs_result.output_dir,
                precision="int8",
            )
        )

    recall_report: dict[str, Any] | None = None
    if eval_suite_path is not None:
        if int8_path is None:
            raise ValueError("eval_suite_path requires Android INT8 export.")
        recall_report = write_int8_recall_delta_report(
            source_model_id=model_id,
            artifact_dir=output_dir,
            eval_suite_path=eval_suite_path,
            fp_model_path=onnx_path,
            int8_model_path=int8_path,
            output_path=recall_delta_report_path,
            cache_dir=cache_dir,
        )
        artifacts = [
            (
                ExportArtifact(
                    format=artifact.format,
                    path=artifact.path,
                    precision=artifact.precision,
                    metadata=int8_artifact_metadata(
                        validation_metadata=int8_validation_metadata,
                        recall_report=recall_report,
                    ),
                )
                if artifact.path == int8_path
                else artifact
            )
            for artifact in artifacts
        ]

    manifest_path = write_export_manifest(
        output_dir,
        source_model_id=model_id,
        config=config,
        artifacts=artifacts,
        tokenizer_files=tokenizer_files,
        minimum_opset=MINIMUM_TOKEN_CLASSIFICATION_OPSET,
        optimization=optimization_manifest,
        shape_buckets=shape_bucket_config,
        validation=validation_manifest,
        operator_fallbacks=operator_fallbacks,
    )
    if recall_report is not None:
        apply_int8_recall_certification(
            output_dir,
            recall_report,
            report_relpath=str(recall_report["report_path"]),
        )
    result = OnnxConversionResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        artifacts=tuple(artifacts),
    )

    if publish_to_hub:
        publish_result = publish_artifact(
            artifact_dir=output_dir,
            source_model_id=model_id,
            format_name=result.formats[0],
            formats=result.formats,
            repo_id=publish_repo_id,
            org=publish_org,
            version=publish_version,
            token_env=publish_token_env,
            manifest_path=publish_manifest_path,
            private=publish_private,
            skip_existing=not publish_overwrite_existing,
        )
        if publish_result.skipped:
            logger.info("Skipped existing Hub repo %s", publish_result.repo_id)
        else:
            logger.info("Published ONNX artifacts to %s", publish_result.repo_id)

    return result


def save_source_assets(
    model_id: str,
    output_dir: str | Path,
    *,
    cache_dir: str | None = None,
    max_seq_length: int = 512,
    require_id2label: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Save config, labels, and tokenizer assets beside exported artifacts."""

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for ONNX export metadata. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    output_dir = Path(output_dir)
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir).to_dict()
    config["max_sequence_length"] = max_seq_length
    if require_id2label:
        config["id2label"] = _ensure_id2label(config)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    id2label = config.get("id2label")
    if isinstance(id2label, Mapping):
        with (output_dir / "id2label.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {str(key): value for key, value in id2label.items()}, handle, indent=2
            )

    tokenizer_files: list[str] = []
    try:
        tokenizer = get_tokenizer_with_loader(
            model_id,
            AutoTokenizer.from_pretrained,
            cache_dir=cache_dir,
        )
        tokenizer.save_pretrained(output_dir)
        tokenizer_files = find_tokenizer_files(output_dir)
    except Exception as exc:
        logger.warning(
            "Could not save tokenizer assets for %s into %s: %s",
            model_id,
            output_dir,
            exc,
        )

    return config, tokenizer_files


def write_export_manifest(
    output_dir: str | Path,
    *,
    source_model_id: str,
    config: Mapping[str, Any],
    artifacts: Sequence[ExportArtifact],
    tokenizer_files: Sequence[str] | None = None,
    minimum_opset: int = MINIMUM_TOKEN_CLASSIFICATION_OPSET,
    optimization: Mapping[str, Any] | None = None,
    shape_buckets: ShapeBucketConfig | Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    operator_fallbacks: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Write an ONNX/WebGPU artifact manifest into *output_dir*."""

    output_dir = Path(output_dir)
    formats = _dedupe_keep_order([artifact.format for artifact in artifacts])
    task = str(config.get("_mlx_task") or config.get("task") or "token-classification")
    family = str(
        config.get("_mlx_family")
        or config.get("_mlx_model_type")
        or config.get("model_type")
        or "unknown"
    )

    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "formats": formats,
        "task": task,
        "family": family,
        "source_model_id": source_model_id,
        "config_path": "config.json",
        "label_map_path": "id2label.json"
        if (output_dir / "id2label.json").exists()
        else None,
        "artifacts": [artifact.to_manifest(output_dir) for artifact in artifacts],
        "minimum_opset": minimum_opset,
        "max_sequence_length": config.get("max_sequence_length")
        or config.get("max_position_embeddings", 512),
        "dynamic_shapes": {
            "batch_axis": "dynamic",
            "sequence_axis": "dynamic",
            "shape_buckets": _shape_buckets_manifest(shape_buckets),
        },
        "optimization": dict(optimization or {"enabled": False}),
        "validation": dict(validation or {}),
        "operator_fallbacks": [dict(item) for item in (operator_fallbacks or [])],
        "tokenizer": {
            "path": ".",
            "files": list(tokenizer_files or find_tokenizer_files(output_dir)),
        },
    }

    manifest_path = output_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def optimize_onnx_graph(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: OnnxOptimizationConfig | None = None,
    model_type: str = "bert",
    minimum_opset: int = MINIMUM_TOKEN_CLASSIFICATION_OPSET,
) -> dict[str, Any]:
    """Optimize an exported ONNX graph and return manifest-ready metadata."""

    config = _normalise_optimization_config(config)
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"ONNX input does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = "copy"
    if not config.enabled_passes:
        _copy_model(input_path, output_path)
    else:
        backend = _try_transformer_optimizer(
            input_path,
            output_path,
            config=config,
            model_type=model_type,
        ) or _optimize_with_ort_session(input_path, output_path, config=config)

    _check_onnx_model(output_path)
    manifest = config.to_manifest()
    manifest.update(
        {
            "backend": backend,
            "input_path": input_path.name,
            "output_path": output_path.name,
            "minimum_opset": minimum_opset,
        }
    )
    return manifest


def validate_optimized_onnx_export(
    unoptimized_path: str | Path,
    optimized_path: str | Path,
    *,
    model_id: str,
    cache_dir: str | None = None,
    config: Mapping[str, Any] | None = None,
    shape_bucket_config: ShapeBucketConfig | None = None,
    optimization_config: OnnxOptimizationConfig | None = None,
    validation_texts: Sequence[str] | None = None,
    validation_lengths: Sequence[int] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Validate optimized ONNX parity, dynamic shapes, fallback, and latency."""

    try:
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "numpy, onnxruntime, and transformers are required for optimized "
            "ONNX validation. Install with: pip install openmed[onnx]"
        ) from exc

    shape_bucket_config = shape_bucket_config or ShapeBucketConfig()
    optimization_config = _normalise_optimization_config(optimization_config)
    providers = _runtime_providers(ort, optimization_config.providers)
    tokenizer = get_tokenizer_with_loader(
        model_id,
        AutoTokenizer.from_pretrained,
        cache_dir=cache_dir,
    )
    unoptimized_session = ort.InferenceSession(
        str(unoptimized_path),
        providers=providers,
    )
    optimized_session = ort.InferenceSession(
        str(optimized_path),
        providers=providers,
    )

    lengths = tuple(
        int(length)
        for length in (validation_lengths or DEFAULT_DYNAMIC_VALIDATION_LENGTHS)
    )
    id2label = (config or {}).get("id2label")
    dynamic_results: list[dict[str, Any]] = []
    parity_results: list[dict[str, Any]] = []
    all_dynamic_passed = True
    all_parity_passed = True
    for length in lengths:
        feed = _runtime_feed_for_length(
            optimized_session.get_inputs(),
            tokenizer=tokenizer,
            sequence_length=length,
            np_module=np,
        )
        try:
            baseline_logits = unoptimized_session.run(None, feed)[0]
            optimized_logits = optimized_session.run(None, feed)[0]
        except Exception as exc:  # pragma: no cover - depends on runtime kernels.
            all_dynamic_passed = False
            all_parity_passed = False
            dynamic_results.append(
                {
                    "requested_length": length,
                    "bucket": shape_bucket_config.bucket_for(length),
                    "passed": False,
                    "error": str(exc),
                }
            )
            continue

        parity = evaluate_onnx_logit_parity(
            baseline_logits,
            optimized_logits,
            id2label=id2label if isinstance(id2label, Mapping) else None,
            rtol=rtol,
            atol=atol,
        )
        all_parity_passed = all_parity_passed and parity.passed
        dynamic_results.append(
            {
                "requested_length": length,
                "bucket": shape_bucket_config.bucket_for(length),
                "passed": True,
            }
        )
        parity_payload = parity.to_dict()
        parity_payload["requested_length"] = length
        parity_results.append(parity_payload)

    benchmark_length = shape_bucket_config.bucket_for(
        min(128, max(lengths) if lengths else 128)
    )
    benchmark_feed = _runtime_feed_for_length(
        optimized_session.get_inputs(),
        tokenizer=tokenizer,
        sequence_length=benchmark_length,
        np_module=np,
    )
    baseline_ms = _average_latency_ms(unoptimized_session, benchmark_feed)
    optimized_ms = _average_latency_ms(optimized_session, benchmark_feed)
    improvement = (baseline_ms - optimized_ms) / baseline_ms if baseline_ms > 0 else 0.0
    latency_passed = improvement >= optimization_config.required_latency_improvement
    operator_fallbacks = _detect_operator_fallbacks(
        optimized_path,
        benchmark_feed,
        providers=optimization_config.providers,
        ort_module=ort,
    )
    operator_fallbacks.extend(
        _provider_fallbacks(
            requested=optimization_config.providers,
            used=providers,
        )
    )
    if validation_texts:
        parity_results.extend(
            _validate_text_corpus_parity(
                unoptimized_session,
                optimized_session,
                tokenizer=tokenizer,
                texts=validation_texts,
                id2label=id2label if isinstance(id2label, Mapping) else None,
                rtol=rtol,
                atol=atol,
            )
        )
        all_parity_passed = all(item.get("passed") for item in parity_results)

    passed = all_dynamic_passed and all_parity_passed and latency_passed
    manifest = {
        "passed": passed,
        "dynamic_shapes": {
            "passed": all_dynamic_passed,
            "lengths": dynamic_results,
        },
        "numeric_parity": {
            "passed": all_parity_passed,
            "rtol": rtol,
            "atol": atol,
            "results": parity_results,
        },
        "latency": {
            "passed": latency_passed,
            "required_improvement": optimization_config.required_latency_improvement,
            "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms,
            "improvement": improvement,
            "benchmark_length": benchmark_length,
            "provider": providers[0] if providers else None,
        },
        "operator_fallbacks": operator_fallbacks,
    }
    if not passed:
        raise RuntimeError(
            "optimized ONNX validation failed; see validation manifest evidence"
        )
    return manifest


def _normalise_optimization_config(
    config: OnnxOptimizationConfig | None,
) -> OnnxOptimizationConfig:
    if config is None:
        return OnnxOptimizationConfig()
    providers = tuple(config.providers or DEFAULT_EXECUTION_PROVIDERS)
    if providers == config.providers:
        return config
    return OnnxOptimizationConfig(
        constant_folding=config.constant_folding,
        attention_fusion=config.attention_fusion,
        layer_norm_fusion=config.layer_norm_fusion,
        gelu_fusion=config.gelu_fusion,
        required_latency_improvement=config.required_latency_improvement,
        providers=providers,
    )


def _model_type_for_optimizer(config: Mapping[str, Any]) -> str:
    value = str(config.get("model_type") or "bert").strip().lower()
    return value.replace("_", "-") or "bert"


def _try_transformer_optimizer(
    input_path: Path,
    output_path: Path,
    *,
    config: OnnxOptimizationConfig,
    model_type: str,
) -> str | None:
    try:
        from onnxruntime.transformers.fusion_options import FusionOptions
        from onnxruntime.transformers.optimizer import optimize_model
    except ImportError:
        return None

    try:
        fusion_options = FusionOptions(model_type)
        _configure_fusion_options(fusion_options, config)
        optimized = optimize_model(
            str(input_path),
            model_type=model_type,
            num_heads=0,
            hidden_size=0,
            optimization_options=fusion_options,
        )
        optimized.save_model_to_file(
            str(output_path),
            use_external_data_format=False,
        )
    except Exception as exc:  # pragma: no cover - depends on optional ORT tools.
        logger.debug("Transformer ONNX optimizer fallback: %s", exc)
        return None

    if not output_path.exists():
        return None
    return "onnxruntime-transformers"


def _configure_fusion_options(
    fusion_options: Any,
    config: OnnxOptimizationConfig,
) -> None:
    for pass_name, attrs in _FUSION_OPTION_ATTRS.items():
        enabled = config._passes()[pass_name]
        for attr in attrs:
            if hasattr(fusion_options, attr):
                setattr(fusion_options, attr, enabled)


def _optimize_with_ort_session(
    input_path: Path,
    output_path: Path,
    *,
    config: OnnxOptimizationConfig,
) -> str:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for post-export graph optimization. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = _graph_optimization_level(ort, config)
    session_options.optimized_model_filepath = str(output_path)
    ort.InferenceSession(
        str(input_path),
        sess_options=session_options,
        providers=_runtime_providers(ort, config.providers),
    )
    if not output_path.exists():
        _copy_model(input_path, output_path)
        return "onnxruntime-session-copy"
    return "onnxruntime-session"


def _graph_optimization_level(
    ort_module: Any,
    config: OnnxOptimizationConfig,
) -> Any:
    if not config.enabled_passes:
        return ort_module.GraphOptimizationLevel.ORT_DISABLE_ALL
    if config.constant_folding and not (
        config.attention_fusion or config.layer_norm_fusion or config.gelu_fusion
    ):
        return ort_module.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort_module.GraphOptimizationLevel.ORT_ENABLE_EXTENDED


def _runtime_providers(ort_module: Any, requested: Sequence[str]) -> list[str]:
    requested_providers = [str(provider) for provider in requested if provider]
    if not requested_providers:
        requested_providers = list(DEFAULT_EXECUTION_PROVIDERS)
    available_getter = getattr(ort_module, "get_available_providers", None)
    available = (
        set(available_getter())
        if callable(available_getter)
        else set(requested_providers)
    )
    selected = [provider for provider in requested_providers if provider in available]
    if not selected and "CPUExecutionProvider" in available:
        selected = ["CPUExecutionProvider"]
    return selected or requested_providers


def _runtime_feed_for_length(
    session_inputs: Sequence[Any],
    *,
    tokenizer: Any,
    sequence_length: int,
    np_module: Any,
) -> dict[str, Any]:
    token_id = _token_id_for_feed(tokenizer)
    feed: dict[str, Any] = {}
    for input_info in session_inputs:
        name = getattr(input_info, "name", "")
        if name == "input_ids":
            feed[name] = np_module.full(
                (1, sequence_length),
                token_id,
                dtype=np_module.int64,
            )
        elif name == "attention_mask":
            feed[name] = np_module.ones((1, sequence_length), dtype=np_module.int64)
        elif name == "token_type_ids":
            feed[name] = np_module.zeros((1, sequence_length), dtype=np_module.int64)
        else:
            feed[name] = np_module.zeros((1, sequence_length), dtype=np_module.int64)
    return feed


def _token_id_for_feed(tokenizer: Any) -> int:
    for attr in ("unk_token_id", "pad_token_id", "cls_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 1


def _average_latency_ms(
    session: Any, feed: Mapping[str, Any], repeats: int = 5
) -> float:
    session.run(None, dict(feed))
    started = time.perf_counter()
    for _ in range(repeats):
        session.run(None, dict(feed))
    elapsed = time.perf_counter() - started
    return (elapsed / repeats) * 1000.0


def _validate_text_corpus_parity(
    unoptimized_session: Any,
    optimized_session: Any,
    *,
    tokenizer: Any,
    texts: Sequence[str],
    id2label: Mapping[Any, str] | None,
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="np",
    )
    offsets = encoded.pop("offset_mapping", None)
    feed = _feed_for_session_inputs(optimized_session.get_inputs(), encoded)
    parity = evaluate_onnx_logit_parity(
        unoptimized_session.run(None, feed)[0],
        optimized_session.run(None, feed)[0],
        id2label=id2label,
        offsets=offsets,
        rtol=rtol,
        atol=atol,
    )
    payload = parity.to_dict()
    payload["corpus_examples"] = len(texts)
    payload["source"] = "validation_texts"
    return [payload]


def _feed_for_session_inputs(
    session_inputs: Sequence[Any],
    encoded: Mapping[str, Any],
) -> dict[str, Any]:
    feed: dict[str, Any] = {}
    for input_info in session_inputs:
        name = getattr(input_info, "name", "")
        if name in encoded:
            feed[name] = encoded[name]
    return feed


def _detect_operator_fallbacks(
    model_path: str | Path,
    feed: Mapping[str, Any],
    *,
    providers: Sequence[str],
    ort_module: Any,
) -> list[dict[str, Any]]:
    try:
        session_options = ort_module.SessionOptions()
        session_options.enable_profiling = True
        session = ort_module.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=_runtime_providers(ort_module, providers),
        )
        session.run(None, _feed_for_session_inputs(session.get_inputs(), feed))
        profile_path = session.end_profiling()
        with open(profile_path, "r", encoding="utf-8") as handle:
            events = json.load(handle)
        try:
            os.unlink(profile_path)
        except OSError:
            pass
    except Exception as exc:  # pragma: no cover - depends on runtime profiling.
        return [
            {
                "requested_provider": providers[0] if providers else None,
                "execution_provider": None,
                "op_type": None,
                "node_name": None,
                "reason": f"profiling_unavailable: {exc}",
            }
        ]
    return _operator_fallbacks_from_profile_events(
        events,
        preferred_provider=providers[0] if providers else None,
    )


def _operator_fallbacks_from_profile_events(
    events: Sequence[Mapping[str, Any]],
    *,
    preferred_provider: str | None,
) -> list[dict[str, Any]]:
    fallbacks: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for event in events:
        args = event.get("args") if isinstance(event, Mapping) else None
        if not isinstance(args, Mapping):
            continue
        provider = args.get("provider")
        if not provider or provider == preferred_provider:
            continue
        op_type = args.get("op_name") or args.get("op_type")
        node_name = args.get("node_name") or event.get("name")
        key = (str(provider), str(op_type), str(node_name))
        if key in seen:
            continue
        seen.add(key)
        fallbacks.append(
            {
                "requested_provider": preferred_provider,
                "execution_provider": str(provider),
                "op_type": str(op_type) if op_type is not None else None,
                "node_name": str(node_name) if node_name is not None else None,
                "reason": "profiled_on_fallback_provider",
            }
        )
    return fallbacks


def _provider_fallbacks(
    *,
    requested: Sequence[str],
    used: Sequence[str],
) -> list[dict[str, Any]]:
    used_set = set(used)
    fallback_provider = used[0] if used else None
    return [
        {
            "requested_provider": provider,
            "execution_provider": fallback_provider,
            "op_type": None,
            "node_name": None,
            "reason": "provider_unavailable",
        }
        for provider in requested
        if provider not in used_set
    ]


def _shape_buckets_manifest(
    value: ShapeBucketConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(value, ShapeBucketConfig):
        return value.to_manifest()
    if isinstance(value, Mapping):
        return dict(value)
    return ShapeBucketConfig().to_manifest()


def _copy_model(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def _check_onnx_model(path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is required to validate exported artifacts. "
            "Install with: pip install openmed[onnx]"
        ) from exc
    onnx.checker.check_model(str(path))


def _consolidate_external_onnx_data(path: Path) -> None:
    """Store large-model external tensors in one deterministic sidecar file."""

    try:
        import onnx
    except ImportError:
        return

    metadata_model = onnx.load(str(path), load_external_data=False)
    old_locations = {
        item.value
        for tensor in metadata_model.graph.initializer
        for item in tensor.external_data
        if item.key == "location" and item.value
    }
    if not old_locations:
        return

    sidecar_name = f"{path.name}.data"
    sidecar_path = path.parent / sidecar_name
    temporary_dir = path.parent / f".{path.name}.consolidated"
    temporary_path = temporary_dir / path.name
    temporary_sidecar = temporary_dir / sidecar_name
    backup_sidecar = temporary_dir / f"{sidecar_name}.original"
    model = onnx.load(str(path), load_external_data=True)
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir()

    try:
        onnx.save_model(
            model,
            str(temporary_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=sidecar_name,
            size_threshold=0,
            convert_attribute=False,
        )
        if not temporary_path.is_file() or not temporary_sidecar.is_file():
            raise RuntimeError("ONNX external-data consolidation was incomplete")

        if sidecar_path.exists():
            sidecar_path.replace(backup_sidecar)
        try:
            temporary_sidecar.replace(sidecar_path)
            temporary_path.replace(path)
        except Exception:
            sidecar_path.unlink(missing_ok=True)
            if backup_sidecar.exists():
                backup_sidecar.replace(sidecar_path)
            raise
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    for location in old_locations - {sidecar_name}:
        old_path = (path.parent / location).resolve(strict=False)
        if old_path.parent == path.parent.resolve(strict=False):
            old_path.unlink(missing_ok=True)


def _copy_missing_model_metadata(target: Any, source: Any) -> None:
    if not getattr(target, "ir_version", None) and getattr(source, "ir_version", None):
        target.ir_version = source.ir_version

    for field in ("producer_name", "producer_version", "domain", "doc_string"):
        if not getattr(target, field, None) and getattr(source, field, None):
            setattr(target, field, getattr(source, field))
    if not getattr(target, "model_version", None) and getattr(
        source, "model_version", None
    ):
        target.model_version = source.model_version

    target_opsets = getattr(target, "opset_import", None)
    source_opsets = getattr(source, "opset_import", None)
    if target_opsets is not None and source_opsets is not None and not target_opsets:
        target_opsets.extend(source_opsets)

    target_graph = getattr(target, "graph", None)
    source_graph = getattr(source, "graph", None)
    if (
        target_graph is not None
        and source_graph is not None
        and not getattr(target_graph, "name", None)
        and getattr(source_graph, "name", None)
    ):
        target_graph.name = source_graph.name


def _dedupe_keep_order(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("expected at least one integer")
    return result


def _normalize_profile(profile: str) -> str:
    normalized = profile.lower().replace("_", "-")
    if normalized in {DEFAULT_PROFILE_NAME, "onnx"}:
        return DEFAULT_PROFILE_NAME
    if normalized == ANDROID_PROFILE_NAME:
        return ANDROID_PROFILE_NAME
    if normalized == OPENVINO_PROFILE_NAME:
        return OPENVINO_PROFILE_NAME
    raise ValueError(
        "unsupported ONNX export profile "
        f"{profile!r}; expected default, android, or openvino"
    )


def _resolve_opset(*, profile: str, opset: int | None) -> int:
    if profile == ANDROID_PROFILE_NAME:
        if opset is not None and opset != ANDROID_ONNX_OPSET:
            raise ValueError(
                "Android ONNX profile requires fixed opset "
                f"{ANDROID_ONNX_OPSET}; got {opset}"
            )
        return ANDROID_ONNX_OPSET
    return DEFAULT_ONNX_OPSET if opset is None else opset


def _transformers_tokenizer_loader(*, cache_dir: str | None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for OpenVINO export verification. "
            "Install with: pip install openmed[openvino]"
        ) from exc

    def load_tokenizer(model_id: str, **kwargs: Any) -> Any:
        options = dict(kwargs)
        if cache_dir is not None:
            options.setdefault("cache_dir", cache_dir)
        return AutoTokenizer.from_pretrained(model_id, **options)

    return load_tokenizer


def _ensure_id2label(config: Mapping[str, Any]) -> dict[str, str]:
    id2label = config.get("id2label")
    if isinstance(id2label, Mapping) and id2label:
        return {str(key): str(value) for key, value in id2label.items()}

    label2id = config.get("label2id")
    if isinstance(label2id, Mapping) and label2id:
        labels = sorted(
            ((int(index), str(label)) for label, index in label2id.items()),
            key=lambda item: item[0],
        )
        return {str(index): label for index, label in labels}

    num_labels = config.get("num_labels")
    if isinstance(num_labels, int) and num_labels > 0:
        return {str(index): f"LABEL_{index}" for index in range(num_labels)}

    raise ValueError("Android ONNX profile requires config.json id2label metadata")


def _dynamic_export_kwargs(
    torch_module: Any,
    *,
    export_fn: Any,
    input_names: Sequence[str],
    max_seq_length: int,
    profile: str = DEFAULT_PROFILE_NAME,
) -> dict[str, Any]:
    """Return Torch ONNX dynamic-shape kwargs for the installed exporter."""

    parameters = inspect.signature(export_fn).parameters
    if profile == ANDROID_PROFILE_NAME:
        kwargs: dict[str, Any] = {
            "dynamic_axes": {
                name: {0: "batch", 1: "sequence"} for name in [*input_names, "logits"]
            }
        }
        if "dynamo" in parameters:
            kwargs["dynamo"] = False
        return kwargs

    if "dynamo" in parameters and "dynamic_shapes" in parameters:
        batch_dim = torch_module.export.Dim("batch", min=1)
        sequence_dim = torch_module.export.Dim("sequence", min=1)
        return {
            "dynamo": True,
            "dynamic_shapes": {
                name: {0: batch_dim, 1: sequence_dim} for name in input_names
            },
        }

    dynamic_axes = {
        name: {0: "batch", 1: "sequence"} for name in [*input_names, "logits"]
    }
    return {"dynamic_axes": dynamic_axes}


@contextmanager
def _onnx_export_compatibility(model: Any, torch_module: Any):
    """Apply narrowly scoped exporter compatibility fixes for known families."""

    if getattr(getattr(model, "config", None), "model_type", None) != "longformer":
        yield
        return

    from transformers.models.longformer import modeling_longformer

    original = modeling_longformer.create_bidirectional_mask

    def legacy_bidirectional_mask(
        *,
        inputs_embeds: Any,
        attention_mask: Any,
        **_: Any,
    ) -> Any:
        mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
        return (1.0 - mask) * torch_module.finfo(inputs_embeds.dtype).min

    modeling_longformer.create_bidirectional_mask = legacy_bidirectional_mask
    try:
        yield
    finally:
        modeling_longformer.create_bidirectional_mask = original


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face token-classification model to ONNX artifacts",
    )
    parser.add_argument(
        "--model", required=True, help="Source model ID or local directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for artifacts"
    )
    parser.add_argument(
        "--no-webgpu",
        action="store_true",
        help="Only emit the fp32 ONNX artifact",
    )
    parser.add_argument(
        "--include-transformersjs",
        action="store_true",
        help="Also emit a Transformers.js bundle with model_quantized.onnx",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Do not emit model_int8.onnx for the android profile",
    )
    parser.add_argument(
        "--profile",
        choices=[DEFAULT_PROFILE_NAME, ANDROID_PROFILE_NAME, OPENVINO_PROFILE_NAME],
        default=DEFAULT_PROFILE_NAME,
        help=(
            "Export profile. Use android for ONNX Runtime Mobile artifacts "
            "or openvino for Intel edge runtimes."
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version; android always uses its fixed mobile opset",
    )
    parser.add_argument(
        "--no-onnx-optimization",
        action="store_true",
        help="Skip post-export ONNX graph optimization and validation",
    )
    parser.add_argument(
        "--disable-constant-folding",
        action="store_true",
        help="Disable the post-export constant-folding optimization pass",
    )
    parser.add_argument(
        "--disable-attention-fusion",
        action="store_true",
        help="Disable attention fusion in the post-export optimization pass",
    )
    parser.add_argument(
        "--disable-layernorm-fusion",
        action="store_true",
        help="Disable layernorm fusion in the post-export optimization pass",
    )
    parser.add_argument(
        "--disable-gelu-fusion",
        action="store_true",
        help="Disable GELU fusion in the post-export optimization pass",
    )
    parser.add_argument(
        "--shape-buckets",
        default=",".join(str(bucket) for bucket in DEFAULT_SHAPE_BUCKETS),
        help="Comma-separated sequence length buckets for dynamic-shape runs",
    )
    parser.add_argument(
        "--validation-lengths",
        default=",".join(str(length) for length in DEFAULT_DYNAMIC_VALIDATION_LENGTHS),
        help="Comma-separated sequence lengths used for optimized graph validation",
    )
    parser.add_argument(
        "--validation-text",
        action="append",
        default=None,
        help="Additional real text sample for optimized-vs-unoptimized parity",
    )
    parser.add_argument(
        "--execution-provider",
        action="append",
        default=None,
        help="ONNX Runtime execution provider to request, repeatable",
    )
    parser.add_argument(
        "--required-latency-improvement",
        type=float,
        default=0.20,
        help="Required optimized CPU latency improvement ratio",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Hugging Face cache directory"
    )
    parser.add_argument(
        "--sample-text",
        default=DEFAULT_SAMPLE_TEXT,
        help="Synthetic note used for export and runtime verification",
    )
    parser.add_argument(
        "--eval-suite",
        default=None,
        help="Benchmark fixture JSON/JSONL used to certify android INT8 recall",
    )
    parser.add_argument(
        "--recall-delta-report",
        default=None,
        help="Output path for the INT8 recall_delta.json report",
    )
    parser.add_argument(
        "--publish-to-hub",
        action="store_true",
        help="Publish the converted artifact after a successful conversion",
    )
    parser.add_argument(
        "--publish-repo-id", default=None, help="Explicit target repo id"
    )
    parser.add_argument("--publish-org", default="OpenMed", help="Target organization")
    parser.add_argument(
        "--publish-version",
        type=int,
        default=1,
        help="Version suffix used when the source repo is not already versioned",
    )
    parser.add_argument(
        "--publish-manifest",
        default=None,
        help="JSONL manifest path to append or update after publishing",
    )
    parser.add_argument(
        "--publish-token-env",
        default="HF_WRITE_TOKEN",
        help="Environment variable containing the Hub write token",
    )
    parser.add_argument(
        "--publish-private",
        action="store_true",
        help="Create the target repo as private when it does not exist",
    )
    parser.add_argument(
        "--publish-overwrite-existing",
        action="store_true",
        help="Upload into an existing target repo instead of skipping it",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    optimization_config = OnnxOptimizationConfig(
        constant_folding=not args.disable_constant_folding,
        attention_fusion=not args.disable_attention_fusion,
        layer_norm_fusion=not args.disable_layernorm_fusion,
        gelu_fusion=not args.disable_gelu_fusion,
        required_latency_improvement=args.required_latency_improvement,
        providers=tuple(args.execution_provider or DEFAULT_EXECUTION_PROVIDERS),
    )
    shape_bucket_config = ShapeBucketConfig(
        buckets=_parse_int_tuple(args.shape_buckets),
        max_length=max(_parse_int_tuple(args.validation_lengths)),
    )
    convert(
        args.model,
        args.output,
        include_webgpu=not args.no_webgpu,
        include_transformersjs=args.include_transformersjs,
        include_int8=not args.no_int8,
        max_seq_length=args.max_seq_length,
        opset=args.opset,
        profile=args.profile,
        optimize_onnx=not args.no_onnx_optimization,
        optimization_config=optimization_config,
        shape_bucket_config=shape_bucket_config,
        validation_texts=args.validation_text,
        validation_lengths=_parse_int_tuple(args.validation_lengths),
        cache_dir=args.cache_dir,
        sample_text=args.sample_text,
        eval_suite_path=args.eval_suite,
        recall_delta_report_path=args.recall_delta_report,
        publish_to_hub=args.publish_to_hub,
        publish_repo_id=args.publish_repo_id,
        publish_org=args.publish_org,
        publish_version=args.publish_version,
        publish_manifest_path=args.publish_manifest,
        publish_token_env=args.publish_token_env,
        publish_private=args.publish_private,
        publish_overwrite_existing=args.publish_overwrite_existing,
    )


if __name__ == "__main__":
    main()
