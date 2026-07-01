"""Convert HuggingFace token-classification models to CoreML format.

Produces a ``.mlpackage`` suitable for iOS 16+ and macOS 13+ deployment.

Usage::

    python -m openmed.coreml.convert \\
        --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \\
        --output ./OpenMedPIISmall.mlpackage
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from openmed.core.hf_publish import publish_artifact
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
QUANTIZATION_CHOICES = ("int8",)

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
) -> Path:
    """Convert a HuggingFace token-classification model to CoreML.

    Args:
        model_id: HuggingFace model identifier.
        output_path: Destination for the ``.mlpackage`` file.
        max_seq_length: Maximum input sequence length.
        compute_precision: ``"float16"`` (Neural Engine) or ``"float32"`` (CPU).
        compute_units: Core ML compute unit selector: ``"all"``,
            ``"cpuAndNeuralEngine"``, or ``"cpuOnly"``.
        quantize: Optional quantization mode. Use ``"int8"`` to emit an
            INT8-palettized sibling ``.mlpackage``.
        quantized_output_path: Optional destination for the INT8 package.
            Defaults to ``<output stem>_int8.mlpackage`` when ``quantize`` is
            ``"int8"``.
        cache_dir: HuggingFace cache directory.

    Returns:
        Path to the created ``.mlpackage``.
    """
    output_path = Path(output_path)
    _validate_compute_precision(compute_precision)
    _validate_compute_units_choice(compute_units)
    quantize = _validate_quantize(quantize)
    quantized_output = _resolve_quantized_output_path(
        output_path,
        quantize,
        quantized_output_path,
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
        ) from e

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

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape(
                    shape=(
                        1,
                        ct.RangeDim(
                            lower_bound=1, upper_bound=max_seq_length, default=128
                        ),
                    ),
                ),
                dtype=int,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=ct.Shape(
                    shape=(
                        1,
                        ct.RangeDim(
                            lower_bound=1, upper_bound=max_seq_length, default=128
                        ),
                    ),
                ),
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
    )

    # 6. Save
    logger.info("Saving to %s ...", output_path)
    mlmodel.save(str(output_path))
    _write_id2label(output_path, id2label)

    if quantize == "int8" and quantized_output is not None:
        logger.info("Palettizing CoreML weights to INT8 ...")
        quantized_model = _palettize_int8(ct, mlmodel)
        _apply_metadata(
            quantized_model,
            model_id=model_id,
            model_type=model_type,
            num_labels=num_labels,
            id2label=id2label,
            max_seq_length=max_seq_length,
            compute_precision=compute_precision,
            compute_units=compute_units,
            quantization="int8",
        )
        logger.info("Saving INT8 CoreML package to %s ...", quantized_output)
        quantized_model.save(str(quantized_output))
        _write_id2label(quantized_output, id2label)

        float_size = _artifact_size_bytes(output_path)
        int8_size = _artifact_size_bytes(quantized_output)
        if int8_size >= float_size:
            logger.warning(
                "INT8 CoreML package is not smaller than the float package "
                "(float=%d bytes, int8=%d bytes).",
                float_size,
                int8_size,
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


def _resolve_quantized_output_path(
    output_path: Path,
    quantize: str | None,
    quantized_output_path: str | Path | None,
) -> Path | None:
    if quantize is None:
        if quantized_output_path is not None:
            raise ValueError("quantized_output_path requires quantize='int8'.")
        return None
    if quantized_output_path is not None:
        return Path(quantized_output_path)
    suffix = output_path.suffix or ".mlpackage"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}_{quantize}{suffix}")


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


def _id2label_dict(id2label) -> dict[str, str]:
    return {str(key): str(value) for key, value in id2label.items()}


def _write_id2label(output_path: Path, id2label) -> None:
    id2label_path = output_path.parent / f"{output_path.stem}_id2label.json"
    with open(id2label_path, "w", encoding="utf-8") as f:
        json.dump(_id2label_dict(id2label), f, indent=2)


def _palettize_int8(ct, mlmodel):
    optimizer = getattr(getattr(ct, "optimize", None), "coreml", None)
    if optimizer is None:
        raise ImportError(
            "coremltools.optimize.coreml is required for INT8 palettization. "
            "Install or upgrade openmed[coreml]."
        )

    op_config = optimizer.OpPalettizerConfig(mode="kmeans", nbits=8)
    config = optimizer.OptimizationConfig(global_config=op_config)
    return optimizer.palettize_weights(mlmodel, config=config)


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
        help="Optional quantization pass. Use int8 to emit a sibling INT8 package.",
    )
    parser.add_argument(
        "--quantized-output",
        default=None,
        help="Output path for the INT8 .mlpackage when --quantize int8 is used",
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
    )


if __name__ == "__main__":
    main()
