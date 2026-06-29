"""Convert HuggingFace token-classification models to CoreML format.

Produces a ``.mlpackage`` suitable for iOS 16+ and macOS 13+ deployment.

Usage::

    python -m openmed.coreml.convert \\
        --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \\
        --output ./OpenMedPIISmall.mlpackage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import plistlib
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from openmed.core.decoding import build_label_info, labels_to_token_spans
from openmed.core.hf_publish import publish_artifact
from openmed.eval.metrics import compute_recall_slices, normalize_eval_spans
from openmed.eval.quant_delta import (
    COREML_RECALL_DELTA_LIMIT,
    evaluate_coreml_span_parity,
)
from openmed.processing.tokenizer_cache import get_tokenizer_with_loader

logger = logging.getLogger(__name__)

SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES = (
    "bert",
    "distilbert",
    "electra",
    "roberta",
    "xlm-roberta",
    "deberta-v2",
)

COMPUTE_UNIT_CHOICES = ("all", "cpuAndNeuralEngine", "cpuOnly")
QUANTIZATION_CHOICES = ("int8", "int4", "all")
COREML_MANIFEST_FILENAME = "openmed-coreml.json"
COREML_MANIFEST_VERSION = 1
COREML_RESIDENCY_THRESHOLD = 0.90
COREML_PARITY_REPORT_VERSION = 1

_ARCHITECTURE_TYPE_HINTS = (
    ("DebertaV2", "deberta-v2"),
    ("XLMRoberta", "xlm-roberta"),
    ("Roberta", "roberta"),
    ("DistilBert", "distilbert"),
    ("Electra", "electra"),
    ("Bert", "bert"),
)

_COMPUTE_UNIT_ATTRS = {
    "all": "ALL",
    "cpuAndNeuralEngine": "CPU_AND_NE",
    "cpuOnly": "CPU_ONLY",
}

_COMPUTE_PLAN_FILENAMES = (
    "ane_residency.json",
    "compute_plan.json",
    "coreml_compute_plan.json",
    "compute_units.json",
)

CoreMLRunner = Callable[[Any, str, str], Iterable[Any]]


@dataclass(frozen=True)
class CoreMLLayerResidency:
    """One CoreML operation's compute-unit assignment."""

    name: str
    op_type: str = ""
    compute_unit: str = "unknown"
    weight: float = 1.0

    @property
    def ane_resident(self) -> bool:
        return _is_ane_compute_unit(self.compute_unit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "compute_unit": self.compute_unit,
            "weight": self.weight,
            "ane_resident": self.ane_resident,
        }


@dataclass(frozen=True)
class CoreMLResidencyReport:
    """ANE residency summary for a compiled CoreML model."""

    artifact_path: str
    ane_residency_percentage: float
    ane_resident_ops: int
    total_ops: int
    cpu_fallback_layers: tuple[CoreMLLayerResidency, ...] = ()
    threshold: float = COREML_RESIDENCY_THRESHOLD
    source: str = "compute_plan"

    @property
    def passed(self) -> bool:
        return (
            self.ane_residency_percentage >= self.threshold
            and not self.cpu_fallback_layers
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "ane_residency_percentage": self.ane_residency_percentage,
            "ane_resident_ops": self.ane_resident_ops,
            "total_ops": self.total_ops,
            "cpu_fallback_layers": [
                layer.to_dict() for layer in self.cpu_fallback_layers
            ],
            "threshold": self.threshold,
            "passed": self.passed,
            "source": self.source,
        }


@dataclass(frozen=True)
class CoreMLVariantRecord:
    """One converted CoreML package variant in the conversion manifest."""

    name: str
    path: str
    precision: str
    quantization: str
    size_bytes: int
    latency_ms: Mapping[str, Any] = field(default_factory=dict)
    residency: Mapping[str, Any] = field(default_factory=dict)
    parity: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "precision": self.precision,
            "quantization": self.quantization,
            "size_bytes": self.size_bytes,
            "latency_ms": dict(self.latency_ms),
            "ane_residency_percentage": self.residency.get("ane_residency_percentage"),
            "cpu_fallback_layers": self.residency.get("cpu_fallback_layers", []),
            "residency": dict(self.residency),
            "parity": dict(self.parity),
        }


def convert(
    model_id: str,
    output_path: str | Path,
    max_seq_length: int = 512,
    compute_precision: str = "float16",
    compute_units: str = "all",
    quantize: str | None = None,
    quantized_output_path: str | Path | None = None,
    cache_dir: Optional[str] = None,
    publish_to_hub: bool = False,
    publish_repo_id: str | None = None,
    publish_org: str = "OpenMed",
    publish_version: int = 1,
    publish_manifest_path: str | Path | None = None,
    publish_token_env: str = "HF_WRITE_TOKEN",
    publish_private: bool = False,
    publish_overwrite_existing: bool = False,
    conversion_manifest_path: str | Path | None = None,
    residency_plan_path: str | Path | None = None,
    eval_suite_path: str | Path | None = None,
    parity_report_path: str | Path | None = None,
    swift_parity_corpus_path: str | Path | None = None,
    latency_iterations: int = 3,
    optimize_for_ane: bool = True,
) -> Path:
    """Convert a HuggingFace token-classification model to CoreML.

    Args:
        model_id: HuggingFace model identifier.
        output_path: Destination for the ``.mlpackage`` file.
        max_seq_length: Maximum input sequence length.
        compute_precision: ``"float16"`` (Neural Engine) or ``"float32"`` (CPU).
        compute_units: Core ML compute unit selector: ``"all"``,
            ``"cpuAndNeuralEngine"``, or ``"cpuOnly"``.
        quantize: Optional quantization mode. Use ``"int8"``, ``"int4"``, or
            ``"all"`` to emit palettized sibling ``.mlpackage`` variants.
        quantized_output_path: Optional destination for a single palettized
            package. Defaults to ``<output stem>_<quantize>.mlpackage``.
        cache_dir: HuggingFace cache directory.
        conversion_manifest_path: Optional JSON manifest path. Defaults to
            ``<output stem>_coreml_manifest.json``.
        residency_plan_path: Optional compiled-model compute assignment export
            to parse for ANE residency. Defaults to the saved package path.
        eval_suite_path: Optional benchmark fixture corpus for PyTorch/CoreML
            span parity.
        parity_report_path: Optional JSON path for span parity evidence.
        swift_parity_corpus_path: Optional JSON path for Swift smoke parity
            fixtures that reference Python spans by fixture id and text hash.
        latency_iterations: Number of prediction calls used for latency
            evidence when the CoreML object supports local prediction.
        optimize_for_ane: Use static rank-2 input shapes for fp16 ANE exports.

    Returns:
        Path to the created ``.mlpackage``.
    """
    output_path = Path(output_path)
    _validate_compute_precision(compute_precision)
    _validate_compute_units_choice(compute_units)
    quantize = _validate_quantize(quantize)
    quantization_variants = _quantization_variants(quantize)
    quantized_outputs = _resolve_quantized_output_paths(
        output_path,
        quantization_variants,
        quantized_output_path,
    )
    manifest_path = _resolve_conversion_manifest_path(
        output_path,
        conversion_manifest_path,
    )

    try:
        import coremltools as ct
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Install with: pip install openmed[coreml]"
        )

    source_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    model_type = resolve_supported_model_type(source_config)

    # 1. Load model and tokenizer
    logger.info("Loading HuggingFace model %s ...", model_id)
    tokenizer = get_tokenizer_with_loader(
        model_id,
        AutoTokenizer.from_pretrained,
        cache_dir=cache_dir,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    model.eval()

    num_labels = model.config.num_labels
    id2label = model.config.id2label

    # 2. Create wrapper that returns only logits (not ModelOutput)
    class TokenClassificationWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, attention_mask):
            output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return output.logits

    wrapper = TokenClassificationWrapper(model)
    wrapper.eval()

    # 3. Trace with sample inputs
    logger.info("Tracing model with sequence length %d ...", max_seq_length)
    sample_text = "Patient John Doe visited the clinic on 2024-01-15."
    sample = tokenizer(
        sample_text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    traced = torch.jit.trace(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
    )

    # 4. Convert to CoreML
    logger.info("Converting to CoreML (%s precision) ...", compute_precision)
    ct_precision = (
        ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32
    )
    ct_compute_units = _resolve_compute_units(ct, compute_units)
    input_shape = _coreml_input_shape(
        ct,
        max_seq_length=max_seq_length,
        optimize_for_ane=optimize_for_ane,
        compute_precision=compute_precision,
        compute_units=compute_units,
    )

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=input_shape,
                dtype=int,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=input_shape,
                dtype=int,
            ),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        compute_precision=ct_precision,
        compute_units=ct_compute_units,
        minimum_deployment_target=ct.target.iOS16,
    )

    sample_inputs = _latency_sample_inputs(sample)

    # 5. Add metadata
    _apply_metadata(
        mlmodel,
        model_id=model_id,
        model_type=model_type,
        num_labels=num_labels,
        id2label=id2label,
        max_seq_length=max_seq_length,
        compute_precision=compute_precision,
        compute_units=compute_units,
        quantization="none",
        optimize_for_ane=optimize_for_ane,
    )

    # 6. Save
    logger.info("Saving to %s ...", output_path)
    mlmodel.save(str(output_path))
    _write_id2label(output_path, id2label)
    variant_models: dict[str, Any] = {"coreml-fp16": mlmodel}
    variant_paths: dict[str, Path] = {"coreml-fp16": output_path}

    for variant in quantization_variants:
        quantized_output = quantized_outputs[variant]
        bits = _quantization_bits(variant)
        logger.info("Palettizing CoreML weights to %s ...", variant.upper())
        quantized_model = _palettize(ct, mlmodel, bits=bits)
        _apply_metadata(
            quantized_model,
            model_id=model_id,
            model_type=model_type,
            num_labels=num_labels,
            id2label=id2label,
            max_seq_length=max_seq_length,
            compute_precision=compute_precision,
            compute_units=compute_units,
            quantization=variant,
            optimize_for_ane=optimize_for_ane,
        )
        logger.info(
            "Saving %s CoreML package to %s ...",
            variant.upper(),
            quantized_output,
        )
        quantized_model.save(str(quantized_output))
        _write_id2label(quantized_output, id2label)
        variant_models[f"coreml-{variant}"] = quantized_model
        variant_paths[f"coreml-{variant}"] = quantized_output

        float_size = _artifact_size_bytes(output_path)
        quantized_size = _artifact_size_bytes(quantized_output)
        if quantized_size >= float_size:
            logger.warning(
                "%s CoreML package is not smaller than the float package "
                "(float=%d bytes, quantized=%d bytes).",
                variant.upper(),
                float_size,
                quantized_size,
            )

    residency_by_variant = _analyze_residency_for_variants(
        variant_paths,
        residency_plan_path=residency_plan_path,
    )
    parity_by_variant: dict[str, Mapping[str, Any]] = {}
    if eval_suite_path is not None:
        parity_report = write_coreml_variant_parity_report(
            source_model_id=model_id,
            variant_paths=variant_paths,
            eval_suite_path=eval_suite_path,
            output_path=parity_report_path,
            swift_corpus_path=swift_parity_corpus_path,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
        )
        parity_by_variant = {
            str(item["format"]): dict(item)
            for item in parity_report.get("variants", [])
            if isinstance(item, Mapping)
        }

    records = []
    for name, path in variant_paths.items():
        quantization = name.removeprefix("coreml-")
        if quantization == "fp16":
            quantization = "none"
        records.append(
            CoreMLVariantRecord(
                name=name,
                path=_artifact_relative_path(path, manifest_path.parent),
                precision=compute_precision,
                quantization=quantization,
                size_bytes=_artifact_size_bytes(path),
                latency_ms=_measure_prediction_latency(
                    variant_models[name],
                    sample_inputs,
                    iterations=latency_iterations,
                ),
                residency=residency_by_variant.get(name, {}),
                parity=parity_by_variant.get(name, {}),
            )
        )
    _write_coreml_conversion_manifest(
        manifest_path,
        model_id=model_id,
        model_type=model_type,
        compute_units=compute_units,
        max_seq_length=max_seq_length,
        optimize_for_ane=optimize_for_ane,
        variants=records,
    )

    logger.info("CoreML model saved to %s", output_path)
    if publish_to_hub:
        result = publish_artifact(
            artifact_dir=output_path,
            source_model_id=model_id,
            format_name="coreml",
            repo_id=publish_repo_id,
            org=publish_org,
            version=publish_version,
            token_env=publish_token_env,
            manifest_path=publish_manifest_path,
            private=publish_private,
            skip_existing=not publish_overwrite_existing,
        )
        if result.skipped:
            logger.info("Skipped existing Hub repo %s", result.repo_id)
        else:
            logger.info("Published CoreML artifact to %s", result.repo_id)
    return output_path


def resolve_supported_model_type(config) -> str:
    """Resolve and validate a supported CoreML token-classification family."""
    raw_model_type = _config_get(config, "model_type")
    model_type = _normalize_model_type(raw_model_type)
    if model_type is None:
        model_type = _infer_model_type_from_architectures(
            _config_get(config, "architectures", []) or []
        )

    if model_type in SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES:
        return model_type

    supported = ", ".join(SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES)
    raise ValueError(
        "Unsupported CoreML token-classification architecture "
        f"{model_type or raw_model_type!r}. Supported families: {supported}."
    )


def _config_get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_model_type(model_type: str | None) -> str | None:
    if model_type is None:
        return None
    normalized = str(model_type).replace("_", "-").lower()
    return normalized or None


def _infer_model_type_from_architectures(architectures) -> str | None:
    for needle, model_type in _ARCHITECTURE_TYPE_HINTS:
        if any(needle in str(architecture) for architecture in architectures):
            return model_type
    return None


def _resolve_compute_units(ct, compute_units: str):
    _validate_compute_units_choice(compute_units)

    attr = _COMPUTE_UNIT_ATTRS[compute_units]
    try:
        return getattr(ct.ComputeUnit, attr)
    except AttributeError as exc:
        raise ValueError(
            "Installed coremltools does not expose compute unit "
            f"{attr!r}; update coremltools or choose another compute_units value."
        ) from exc


def _validate_compute_precision(compute_precision: str) -> None:
    if compute_precision not in {"float16", "float32"}:
        raise ValueError("compute_precision must be either 'float16' or 'float32'.")


def _validate_compute_units_choice(compute_units: str) -> None:
    if compute_units not in _COMPUTE_UNIT_ATTRS:
        choices = ", ".join(COMPUTE_UNIT_CHOICES)
        raise ValueError(f"compute_units must be one of: {choices}.")


def _validate_quantize(quantize: str | None) -> str | None:
    if quantize is None:
        return None
    normalized = str(quantize).lower()
    if normalized not in QUANTIZATION_CHOICES:
        choices = ", ".join(QUANTIZATION_CHOICES)
        raise ValueError(f"quantize must be one of: {choices}.")
    return normalized


def _quantization_variants(quantize: str | None) -> tuple[str, ...]:
    if quantize is None:
        return ()
    if quantize == "all":
        return ("int8", "int4")
    return (quantize,)


def _quantization_bits(quantize: str) -> int:
    if quantize == "int8":
        return 8
    if quantize == "int4":
        return 4
    raise ValueError(f"unsupported quantization variant: {quantize}")


def _resolve_quantized_output_paths(
    output_path: Path,
    quantization_variants: Sequence[str],
    quantized_output_path: str | Path | None,
) -> dict[str, Path]:
    if not quantization_variants:
        if quantized_output_path is not None:
            raise ValueError("quantized_output_path requires quantize.")
        return {}
    if len(quantization_variants) > 1 and quantized_output_path is not None:
        raise ValueError("quantized_output_path can only be used with one variant.")

    outputs: dict[str, Path] = {}
    for quantize in quantization_variants:
        outputs[quantize] = _resolve_quantized_output_path(
            output_path,
            quantize,
            quantized_output_path,
        )
    return outputs


def _resolve_quantized_output_path(
    output_path: Path,
    quantize: str,
    quantized_output_path: str | Path | None,
) -> Path:
    if quantized_output_path is not None:
        return Path(quantized_output_path)
    suffix = output_path.suffix or ".mlpackage"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}_{quantize}{suffix}")


def _resolve_conversion_manifest_path(
    output_path: Path,
    conversion_manifest_path: str | Path | None,
) -> Path:
    if conversion_manifest_path is not None:
        return Path(conversion_manifest_path)
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}_coreml_manifest.json")


def _coreml_input_shape(
    ct,
    *,
    max_seq_length: int,
    optimize_for_ane: bool,
    compute_precision: str,
    compute_units: str,
):
    if (
        optimize_for_ane
        and compute_precision == "float16"
        and compute_units in {"all", "cpuAndNeuralEngine"}
    ):
        return ct.Shape(shape=(1, max_seq_length))
    return ct.Shape(
        shape=(
            1,
            ct.RangeDim(lower_bound=1, upper_bound=max_seq_length, default=128),
        ),
    )


def _apply_metadata(
    mlmodel,
    *,
    model_id: str,
    model_type: str,
    num_labels: int,
    id2label,
    max_seq_length: int,
    compute_precision: str,
    compute_units: str,
    quantization: str,
    optimize_for_ane: bool,
) -> None:
    mlmodel.short_description = (
        f"OpenMed Token Classification: {model_id} "
        f"({num_labels} labels, max_seq={max_seq_length}, "
        f"quantization={quantization})"
    )
    mlmodel.author = "OpenMed"
    mlmodel.license = "Apache-2.0"
    mlmodel.user_defined_metadata["id2label"] = json.dumps(_id2label_dict(id2label))
    mlmodel.user_defined_metadata["num_labels"] = str(num_labels)
    mlmodel.user_defined_metadata["max_seq_length"] = str(max_seq_length)
    mlmodel.user_defined_metadata["source_model"] = model_id
    mlmodel.user_defined_metadata["source_model_type"] = model_type
    mlmodel.user_defined_metadata["compute_precision"] = compute_precision
    mlmodel.user_defined_metadata["compute_units"] = compute_units
    mlmodel.user_defined_metadata["quantization"] = quantization
    mlmodel.user_defined_metadata["ane_optimization_profile"] = (
        "static-rank2-fp16" if optimize_for_ane else "dynamic"
    )


def _id2label_dict(id2label) -> dict[str, str]:
    return {str(key): str(value) for key, value in id2label.items()}


def _write_id2label(output_path: Path, id2label) -> None:
    id2label_path = output_path.parent / f"{output_path.stem}_id2label.json"
    with open(id2label_path, "w") as f:
        json.dump(_id2label_dict(id2label), f, indent=2)


def _palettize(ct, mlmodel, *, bits: int):
    optimizer = getattr(getattr(ct, "optimize", None), "coreml", None)
    if optimizer is None:
        raise ImportError(
            "coremltools.optimize.coreml is required for palettization. "
            "Install or upgrade openmed[coreml]."
        )

    op_config = optimizer.OpPalettizerConfig(mode="kmeans", nbits=bits)
    config = optimizer.OptimizationConfig(global_config=op_config)
    return optimizer.palettize_weights(mlmodel, config=config)


def analyze_ane_residency(
    compiled_model_path: str | Path,
    *,
    threshold: float = COREML_RESIDENCY_THRESHOLD,
) -> CoreMLResidencyReport:
    """Parse CoreML compute-unit assignments and summarize ANE residency.

    The parser accepts JSON/plist compute-plan exports produced by Core ML
    tooling and simple text dumps used by local smoke harnesses. Unknown or
    missing plans fail closed so release gates do not infer ANE residency from
    the requested compute-unit setting alone.
    """

    path = Path(compiled_model_path)
    plan_path = _find_compute_plan(path)
    if plan_path is None:
        return CoreMLResidencyReport(
            artifact_path=str(path),
            ane_residency_percentage=0.0,
            ane_resident_ops=0,
            total_ops=0,
            threshold=threshold,
            source="missing_compute_plan",
        )

    layers = tuple(_load_compute_plan_layers(plan_path))
    if not layers:
        return CoreMLResidencyReport(
            artifact_path=str(path),
            ane_residency_percentage=0.0,
            ane_resident_ops=0,
            total_ops=0,
            threshold=threshold,
            source=str(plan_path),
        )

    total_weight = sum(max(layer.weight, 0.0) for layer in layers)
    if total_weight <= 0.0:
        total_weight = float(len(layers))
    ane_weight = sum(
        max(layer.weight, 0.0) if layer.weight > 0.0 else 1.0
        for layer in layers
        if layer.ane_resident
    )
    fallback_layers = tuple(layer for layer in layers if not layer.ane_resident)
    return CoreMLResidencyReport(
        artifact_path=str(path),
        ane_residency_percentage=ane_weight / total_weight,
        ane_resident_ops=len([layer for layer in layers if layer.ane_resident]),
        total_ops=len(layers),
        cpu_fallback_layers=fallback_layers,
        threshold=threshold,
        source=str(plan_path),
    )


def write_coreml_variant_parity_report(
    *,
    source_model_id: str,
    variant_paths: Mapping[str, str | Path],
    eval_suite_path: str | Path,
    output_path: str | Path | None = None,
    swift_corpus_path: str | Path | None = None,
    cache_dir: str | None = None,
    max_seq_length: int = 512,
    parent_runner: CoreMLRunner | None = None,
    candidate_runners: Mapping[str, CoreMLRunner] | None = None,
    span_tolerance: int = 0,
    recall_delta_limit: float = COREML_RECALL_DELTA_LIMIT,
) -> dict[str, Any]:
    """Run PyTorch-vs-CoreML parity for all variants and write evidence JSON."""

    from openmed.eval.harness import load_fixtures

    eval_suite_path = Path(eval_suite_path)
    report_path = (
        Path(output_path)
        if output_path is not None
        else Path(next(iter(variant_paths.values()))).with_name("coreml_parity.json")
    )
    generated_at = _utc_now()
    fixtures = load_fixtures(eval_suite_path)
    parent_runner = parent_runner or _hf_token_classification_runner(
        source_model_id,
        cache_dir=cache_dir,
    )
    reference_spans = _capture_spans(
        fixtures,
        runner=parent_runner,
        model_name=source_model_id,
        device="pytorch-reference",
    )
    reference_recall = _recall_by_label(fixtures, reference_spans)

    candidate_runners = dict(candidate_runners or {})
    variants: list[dict[str, Any]] = []
    for format_name, variant_path in sorted(variant_paths.items()):
        runner = candidate_runners.get(format_name) or _coreml_artifact_runner(
            variant_path,
            source_model_id=source_model_id,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
        )
        candidate_spans = _capture_spans(
            fixtures,
            runner=runner,
            model_name=str(variant_path),
            device=format_name,
        )
        candidate_recall = _recall_by_label(fixtures, candidate_spans)
        parity = evaluate_coreml_span_parity(
            format_name=format_name,
            reference_spans=reference_spans,
            candidate_spans=candidate_spans,
            reference_recall=reference_recall,
            candidate_recall=candidate_recall,
            recall_delta_limit=recall_delta_limit,
            span_tolerance=span_tolerance,
            rejectable="int4" in format_name,
        )
        payload = parity.to_dict()
        payload["path"] = str(variant_path)
        payload["candidate_per_label_recall"] = candidate_recall
        variants.append(payload)

    swift_payload: dict[str, Any] | None = None
    if swift_corpus_path is not None:
        swift_payload = write_swift_parity_corpus(
            swift_corpus_path,
            fixtures=fixtures,
            reference_spans=reference_spans,
        )

    payload = {
        "schema_version": COREML_PARITY_REPORT_VERSION,
        "generated_at": generated_at,
        "source_model_id": source_model_id,
        "eval_suite_path": str(eval_suite_path),
        "fixture_count": len(fixtures),
        "metric": "character_recall",
        "recall_delta_limit": recall_delta_limit,
        "span_tolerance": span_tolerance,
        "reference_per_label_recall": reference_recall,
        "variants": variants,
        "swift_smoke_corpus": swift_payload,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def write_swift_parity_corpus(
    output_path: str | Path,
    *,
    fixtures: Sequence[Any],
    reference_spans: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Write a privacy-safe Swift smoke corpus keyed by fixture text hashes."""

    output_path = Path(output_path)
    rows = []
    for fixture in fixtures:
        fixture_id = str(fixture.fixture_id)
        rows.append(
            {
                "fixture_id": fixture_id,
                "text_sha256": hashlib.sha256(fixture.text.encode("utf-8")).hexdigest(),
                "python_reference_spans": [
                    _plain_span(span) for span in reference_spans.get(fixture_id, [])
                ],
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "fixtures": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "path": str(output_path),
        "fixture_count": len(rows),
        "contains_raw_text": False,
    }


def _analyze_residency_for_variants(
    variant_paths: Mapping[str, Path],
    *,
    residency_plan_path: str | Path | None,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for name, path in variant_paths.items():
        source_path = Path(residency_plan_path) if residency_plan_path else path
        reports[name] = analyze_ane_residency(source_path).to_dict()
    return reports


def _find_compute_plan(path: Path) -> Path | None:
    if path.is_file():
        return path
    if not path.exists():
        return None
    for filename in _COMPUTE_PLAN_FILENAMES:
        candidate = path / filename
        if candidate.exists():
            return candidate
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        lowered = candidate.name.lower()
        if "compute" in lowered and candidate.suffix.lower() in {
            ".json",
            ".plist",
            ".txt",
        }:
            return candidate
    return None


def _load_compute_plan_layers(path: Path) -> list[CoreMLLayerResidency]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return _layers_from_compute_payload(json.load(handle))
    if suffix == ".plist":
        with path.open("rb") as handle:
            return _layers_from_compute_payload(plistlib.load(handle))
    return _layers_from_compute_text(path.read_text(encoding="utf-8"))


def _layers_from_compute_payload(payload: Any) -> list[CoreMLLayerResidency]:
    rows = _extract_compute_rows(payload)
    layers: list[CoreMLLayerResidency] = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            name = str(row.get("name") or row.get("layer") or f"op_{index}")
            op_type = str(row.get("op_type") or row.get("type") or "")
            compute_unit = str(
                row.get("compute_unit")
                or row.get("computeUnit")
                or row.get("unit")
                or row.get("device")
                or "unknown"
            )
            weight = _optional_float(
                row.get("flops") or row.get("weight") or row.get("cost") or 1.0
            )
            layers.append(
                CoreMLLayerResidency(
                    name=name,
                    op_type=op_type,
                    compute_unit=compute_unit,
                    weight=weight if weight is not None else 1.0,
                )
            )
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            values = list(row)
            if len(values) >= 2:
                layers.append(
                    CoreMLLayerResidency(
                        name=str(values[0]),
                        compute_unit=str(values[1]),
                    )
                )
    return layers


def _extract_compute_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in (
            "layers",
            "operations",
            "operators",
            "op_assignments",
            "compute_plan",
            "computeUnits",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if all(isinstance(value, str) for value in payload.values()):
            return [
                {"name": str(name), "compute_unit": str(unit)}
                for name, unit in payload.items()
            ]
    return []


def _layers_from_compute_text(text: str) -> list[CoreMLLayerResidency]:
    layers: list[CoreMLLayerResidency] = []
    pattern = re.compile(
        r"(?P<name>[\w./:-]+).*?(?:compute[_ -]?unit|unit|device)"
        r"\s*[:=]\s*(?P<unit>[\w -]+)",
        flags=re.IGNORECASE,
    )
    for index, line in enumerate(text.splitlines()):
        match = pattern.search(line)
        if match is None:
            continue
        layers.append(
            CoreMLLayerResidency(
                name=match.group("name") or f"op_{index}",
                compute_unit=match.group("unit").strip(),
            )
        )
    return layers


def _is_ane_compute_unit(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return (
        "ane" in normalized
        or "neuralengine" in normalized
        or normalized in {"ne", "cpuandne", "cpuandneuralengine"}
    )


def _latency_sample_inputs(sample: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        return {}

    inputs: dict[str, Any] = {}
    for name in ("input_ids", "attention_mask"):
        value = sample[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        try:
            inputs[name] = np.asarray(value, dtype=np.int32)
        except (TypeError, ValueError):
            return {}
    return inputs


def _measure_prediction_latency(
    mlmodel,
    sample_inputs: Mapping[str, Any],
    *,
    iterations: int,
) -> dict[str, Any]:
    if iterations <= 0:
        return {"measured": False, "reason": "disabled"}
    if not sample_inputs:
        return {"measured": False, "reason": "sample_inputs_unavailable"}
    predict = getattr(mlmodel, "predict", None)
    if predict is None:
        return {"measured": False, "reason": "predict_unavailable"}

    values: list[float] = []
    try:
        predict(dict(sample_inputs))
        for _ in range(iterations):
            started = time.perf_counter()
            predict(dict(sample_inputs))
            values.append((time.perf_counter() - started) * 1000.0)
    except Exception as exc:
        return {"measured": False, "reason": str(exc)}

    if not values:
        return {"measured": False, "reason": "no_iterations"}
    return {
        "measured": True,
        "p50": statistics.median(values),
        "p95": _p95(values),
        "iterations": len(values),
    }


def _write_coreml_conversion_manifest(
    manifest_path: Path,
    *,
    model_id: str,
    model_type: str,
    compute_units: str,
    max_seq_length: int,
    optimize_for_ane: bool,
    variants: Sequence[CoreMLVariantRecord],
) -> Path:
    payload = {
        "schema_version": COREML_MANIFEST_VERSION,
        "format": "openmed-coreml",
        "generated_at": _utc_now(),
        "source_model_id": model_id,
        "source_model_type": model_type,
        "compute_units": compute_units,
        "max_sequence_length": max_seq_length,
        "ane_optimization_profile": (
            "static-rank2-fp16" if optimize_for_ane else "dynamic"
        ),
        "variants": [variant.to_dict() for variant in variants],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for variant in variants:
        package_path = (manifest_path.parent / variant.path).resolve()
        if package_path.is_dir():
            package_manifest = package_path / COREML_MANIFEST_FILENAME
            with package_manifest.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
    return manifest_path


def _capture_spans(
    fixtures: Sequence[Any],
    *,
    runner: CoreMLRunner,
    model_name: str,
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    captured: dict[str, list[dict[str, Any]]] = {}
    for fixture in fixtures:
        raw_spans = list(runner(fixture, model_name, device))
        spans = normalize_eval_spans(
            raw_spans,
            default_language=fixture.language,
            default_device=device,
            source_text=fixture.text,
        )
        captured[str(fixture.fixture_id)] = [_plain_span(span) for span in spans]
    return captured


def _recall_by_label(
    fixtures: Sequence[Any],
    spans_by_fixture: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    gold = []
    predicted = []
    offset = 0
    for fixture in fixtures:
        fixture_id = str(fixture.fixture_id)
        for span in fixture.gold_spans:
            gold.append(
                {
                    "label": span.label,
                    "start": span.start + offset,
                    "end": span.end + offset,
                    "text": span.text,
                    "language": span.language,
                }
            )
        for span in spans_by_fixture.get(fixture_id, []):
            predicted.append(
                {
                    **dict(span),
                    "start": int(span["start"]) + offset,
                    "end": int(span["end"]) + offset,
                    "language": fixture.language,
                }
            )
        offset += len(fixture.text) + 1
    return compute_recall_slices(gold, predicted).by_label


def _hf_token_classification_runner(
    model_id: str,
    *,
    cache_dir: str | None = None,
) -> CoreMLRunner:
    pipeline_instance: Any | None = None

    def run_fixture(fixture: Any, model_name: str, device: str) -> Iterable[Any]:
        del model_name, device
        nonlocal pipeline_instance
        if pipeline_instance is None:
            try:
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                    pipeline,
                )
            except ImportError as exc:
                raise ImportError(
                    "transformers is required to certify CoreML parity. "
                    "Install with: pip install transformers"
                ) from exc

            tokenizer = get_tokenizer_with_loader(
                model_id,
                AutoTokenizer.from_pretrained,
                cache_dir=cache_dir,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            pipeline_instance = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
        return pipeline_instance(fixture.text)

    return run_fixture


def _coreml_artifact_runner(
    model_path: str | Path,
    *,
    source_model_id: str,
    max_seq_length: int,
    cache_dir: str | None = None,
) -> CoreMLRunner:
    pipeline_state: dict[str, Any] = {}
    model_path = Path(model_path)

    def run_fixture(fixture: Any, model_name: str, device: str) -> Iterable[Any]:
        del model_name, device
        if not pipeline_state:
            try:
                import coremltools as ct
                import numpy as np
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "coremltools, numpy, and transformers are required to run "
                    "CoreML parity fixtures."
                ) from exc

            id2label = _load_id2label_for_package(model_path)
            pipeline_state["np"] = np
            pipeline_state["model"] = ct.models.MLModel(str(model_path))
            pipeline_state["label_info"] = build_label_info(id2label)
            pipeline_state["tokenizer"] = get_tokenizer_with_loader(
                source_model_id,
                AutoTokenizer.from_pretrained,
                cache_dir=cache_dir,
            )

        tokenizer = pipeline_state["tokenizer"]
        np = pipeline_state["np"]
        encoded = tokenizer(
            fixture.text,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        input_ids = np.asarray([encoded["input_ids"]], dtype=np.int32)
        attention_mask = np.asarray([encoded["attention_mask"]], dtype=np.int32)
        outputs = pipeline_state["model"].predict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
        logits = np.asarray(outputs["logits"])
        label_ids = logits[0].argmax(axis=-1).tolist()
        offsets = encoded.get("offset_mapping") or []
        label_info = pipeline_state["label_info"]
        labels_by_index = {
            index: int(label_id)
            for index, label_id in enumerate(label_ids)
            if index < len(offsets) and tuple(offsets[index]) != (0, 0)
        }
        spans = []
        for span_label, token_start, token_end in labels_to_token_spans(
            labels_by_index,
            label_info,
        ):
            if token_start >= len(offsets) or token_end - 1 >= len(offsets):
                continue
            start = int(offsets[token_start][0])
            end = int(offsets[token_end - 1][1])
            label = label_info.span_class_names[span_label]
            spans.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": fixture.text[start:end],
                    "score": 1.0,
                }
            )
        return spans

    return run_fixture


def _load_id2label_for_package(model_path: Path) -> dict[int, str]:
    candidates = [
        model_path.parent / f"{model_path.stem}_id2label.json",
        model_path / "id2label.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return {int(key): str(value) for key, value in data.items()}
    raise FileNotFoundError(f"Could not find id2label JSON for {model_path}")


def _plain_span(span: Any) -> dict[str, Any]:
    if isinstance(span, Mapping):
        data = span
        return {
            "label": str(
                data.get("label")
                or data.get("entity_group")
                or data.get("entity_type")
                or data.get("entity")
                or "OTHER"
            ),
            "start": int(data.get("start", 0)),
            "end": int(data.get("end", 0)),
            "text": str(data.get("text") or data.get("word") or ""),
        }
    return {
        "label": str(getattr(span, "label", getattr(span, "entity_group", "OTHER"))),
        "start": int(getattr(span, "start")),
        "end": int(getattr(span, "end")),
        "text": str(getattr(span, "text", "")),
    }


def _artifact_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except (OSError, ValueError):
        return path.name


def _p95(values: Sequence[float]) -> float:
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return float(ordered[index])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _artifact_size_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return path.stat().st_size


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace token-classification model to CoreML format",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for .mlpackage file",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum input sequence length (default: 512)",
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Compute precision (default: float16 for Neural Engine)",
    )
    parser.add_argument(
        "--compute-units",
        choices=COMPUTE_UNIT_CHOICES,
        default="all",
        help="Core ML compute units (default: all)",
    )
    parser.add_argument(
        "--quantize",
        choices=QUANTIZATION_CHOICES,
        default=None,
        help=(
            "Optional palettization pass. Use int8, int4, or all to emit "
            "sibling packages."
        ),
    )
    parser.add_argument(
        "--quantized-output",
        default=None,
        help=(
            "Output path for a single palettized .mlpackage. Not valid with "
            "--quantize all."
        ),
    )
    parser.add_argument(
        "--conversion-manifest",
        default=None,
        help="Output path for the CoreML conversion manifest JSON",
    )
    parser.add_argument(
        "--residency-plan",
        default=None,
        help=(
            "Compiled CoreML compute-unit assignment file or directory to parse "
            "for ANE residency evidence"
        ),
    )
    parser.add_argument(
        "--eval-suite",
        default=None,
        help="Benchmark fixture JSON/JSONL used for PyTorch/CoreML span parity",
    )
    parser.add_argument(
        "--parity-report",
        default=None,
        help="Output path for CoreML variant parity evidence JSON",
    )
    parser.add_argument(
        "--swift-parity-corpus",
        default=None,
        help="Output path for privacy-safe Swift parity smoke corpus JSON",
    )
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=3,
        help="Prediction iterations used for conversion latency evidence",
    )
    parser.add_argument(
        "--disable-ane-static-shapes",
        action="store_true",
        help="Keep dynamic sequence shapes instead of ANE-optimized rank-2 shapes",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace model cache directory",
    )
    parser.add_argument(
        "--publish-to-hub",
        action="store_true",
        help="Publish the converted artifact after a successful conversion",
    )
    parser.add_argument(
        "--publish-repo-id",
        default=None,
        help="Explicit target repo id for publishing",
    )
    parser.add_argument(
        "--publish-org",
        default="OpenMed",
        help="Target organization for derived publish repo ids",
    )
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
    convert(
        args.model,
        args.output,
        max_seq_length=args.max_seq_length,
        compute_precision=args.precision,
        compute_units=args.compute_units,
        quantize=args.quantize,
        quantized_output_path=args.quantized_output,
        cache_dir=args.cache_dir,
        publish_to_hub=args.publish_to_hub,
        publish_repo_id=args.publish_repo_id,
        publish_org=args.publish_org,
        publish_version=args.publish_version,
        publish_manifest_path=args.publish_manifest,
        publish_token_env=args.publish_token_env,
        publish_private=args.publish_private,
        publish_overwrite_existing=args.publish_overwrite_existing,
        conversion_manifest_path=args.conversion_manifest,
        residency_plan_path=args.residency_plan,
        eval_suite_path=args.eval_suite,
        parity_report_path=args.parity_report,
        swift_parity_corpus_path=args.swift_parity_corpus,
        latency_iterations=args.latency_iterations,
        optimize_for_ane=not args.disable_ane_static_shapes,
    )


if __name__ == "__main__":
    main()
