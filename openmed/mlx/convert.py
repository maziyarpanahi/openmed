"""Convert Hugging Face token-classification models to OpenMed MLX artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from openmed.mlx.artifact import find_tokenizer_files, write_manifest
from openmed.mlx.models import (
    build_model,
    normalize_model_config,
    normalize_model_type,
    resolve_model_type,
)

logger = logging.getLogger(__name__)

# ---- Weight key remapping ---------------------------------------------------

_BERT_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    (".attention.self.query.", ".attention.query_proj."),
    (".attention.self.key.", ".attention.key_proj."),
    (".attention.self.value.", ".attention.value_proj."),
    (".attention.output.dense.", ".attention.out_proj."),
    (".attention.output.LayerNorm.", ".ln1."),
    (".intermediate.dense.", ".linear1."),
    (".output.dense.", ".linear2."),
    (".output.LayerNorm.", ".ln2."),
    ("bert.encoder.layer.", "encoder.layers."),
    ("bert.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("bert.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("bert.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("bert.embeddings.LayerNorm.", "embeddings.norm."),
    ("classifier.", "classifier."),
    ("bert.pooler.", "_pooler."),
]

_DEBERTA_V2_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    (".attention.output.dense.", ".attention.out_proj."),
    (".attention.output.LayerNorm.", ".ln1."),
    (".intermediate.dense.", ".linear1."),
    (".output.dense.", ".linear2."),
    (".output.LayerNorm.", ".ln2."),
]

_ROBERTA_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    ("roberta.encoder.layer.", "encoder.layers."),
    ("xlm_roberta.encoder.layer.", "encoder.layers."),
    ("roberta.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("roberta.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("roberta.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("roberta.embeddings.LayerNorm.", "embeddings.norm."),
    ("xlm_roberta.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("xlm_roberta.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("xlm_roberta.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("xlm_roberta.embeddings.LayerNorm.", "embeddings.norm."),
    ("roberta.pooler.", "_pooler."),
    ("xlm_roberta.pooler.", "_pooler."),
]

_DISTILBERT_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    (".attention.q_lin.", ".attention.query_proj."),
    (".attention.k_lin.", ".attention.key_proj."),
    (".attention.v_lin.", ".attention.value_proj."),
    (".attention.out_lin.", ".attention.out_proj."),
    (".sa_layer_norm.", ".ln1."),
    (".output_layer_norm.", ".ln2."),
    (".ffn.lin1.", ".linear1."),
    (".ffn.lin2.", ".linear2."),
    ("distilbert.transformer.layer.", "encoder.layers."),
    ("distilbert.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("distilbert.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("distilbert.embeddings.LayerNorm.", "embeddings.norm."),
]

_ELECTRA_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    (".attention.self.query.", ".attention.query_proj."),
    (".attention.self.key.", ".attention.key_proj."),
    (".attention.self.value.", ".attention.value_proj."),
    (".attention.output.dense.", ".attention.out_proj."),
    (".attention.output.LayerNorm.", ".ln1."),
    (".intermediate.dense.", ".linear1."),
    (".output.dense.", ".linear2."),
    (".output.LayerNorm.", ".ln2."),
    ("electra.encoder.layer.", "encoder.layers."),
    ("electra.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("electra.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("electra.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("electra.embeddings.LayerNorm.", "embeddings.norm."),
    ("classifier.", "classifier."),
]

def _infer_source_model_type(key: str, model_type: str | None) -> str:
    normalized = normalize_model_type(model_type)
    if normalized is not None:
        return normalized

    if key.startswith("deberta."):
        return "deberta-v2"
    if key.startswith("distilbert."):
        return "distilbert"
    if key.startswith("roberta.") or key.startswith("xlm_roberta."):
        return "roberta"
    if key.startswith("electra."):
        return "electra"
    return "bert"


def remap_key(key: str, model_type: str | None = None) -> str:
    """Remap a HuggingFace state-dict key to the MLX model namespace."""
    source_model_type = _infer_source_model_type(key, model_type)
    resolved_model_type = resolve_model_type(source_model_type)

    if resolved_model_type == "deberta-v2":
        replacements = _DEBERTA_V2_KEY_REPLACEMENTS
    elif source_model_type == "distilbert":
        replacements = _DISTILBERT_KEY_REPLACEMENTS
    elif source_model_type in {"roberta", "xlm-roberta", "xlm_roberta"}:
        replacements = _ROBERTA_KEY_REPLACEMENTS + _BERT_KEY_REPLACEMENTS
    elif source_model_type == "electra":
        replacements = _ELECTRA_KEY_REPLACEMENTS
    else:
        replacements = _BERT_KEY_REPLACEMENTS

    for hf_pattern, mlx_pattern in replacements:
        key = key.replace(hf_pattern, mlx_pattern)
    return key


def _to_numpy(tensor: Any) -> Any:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return tensor


def convert_weights(
    model_id: str,
    cache_dir: str | None = None,
) -> Tuple[Dict[str, Any], dict[str, Any]]:
    """Load HF token-classification weights and config, then remap for MLX."""
    try:
        from transformers import AutoConfig, AutoModelForTokenClassification
    except ImportError:
        raise ImportError(
            "transformers is required for model conversion. "
            "Install with: pip install transformers"
        )

    logger.info("Loading Hugging Face token-classification model %s ...", model_id)
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    source_model_type = normalize_model_type(config.model_type)
    state_dict = model.state_dict()
    mlx_weights = {}
    skipped = []

    for hf_key, tensor in state_dict.items():
        mlx_key = remap_key(hf_key, source_model_type)
        if mlx_key.startswith("_"):
            skipped.append(hf_key)
            continue
        mlx_weights[mlx_key] = _to_numpy(tensor)

    if skipped:
        logger.info("Skipped %d keys (pooler, etc.): %s", len(skipped), skipped[:5])

    config_dict = normalize_model_config(config.to_dict())
    config_dict["_mlx_model_type"] = resolve_model_type(config_dict)
    config_dict["_mlx_task"] = "token-classification"
    config_dict.setdefault("num_labels", config.num_labels)
    return mlx_weights, config_dict


def save_mlx_model(
    weights: Dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
    quantize_bits: int | None = None,
    source_model_id: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Save converted weights and config to *output_dir*."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
    except ImportError:
        raise ImportError("MLX is required. Install with: pip install openmed[mlx]")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_to_save = dict(config)

    mlx_weights = {k: mx.array(v) for k, v in weights.items()}

    if quantize_bits is not None:
        logger.info("Quantizing to %d bits ...", quantize_bits)
        model = build_model(config)
        model.load_weights(list(mlx_weights.items()))
        nn.quantize(model, bits=quantize_bits)
        mlx_weights = dict(tree_flatten(model.parameters()))
        config_to_save["_mlx_quantization"] = {"bits": quantize_bits}
    else:
        config_to_save.pop("_mlx_quantization", None)

    def _cleanup_other_weight_files(keep_path: Path) -> None:
        for candidate in (output_dir / "weights.safetensors", output_dir / "weights.npz"):
            if candidate != keep_path and candidate.exists():
                candidate.unlink()

    weights_format = "npz"
    weights_path = output_dir / "weights.npz"
    metadata = {
        "format": "mlx",
        "openmed_task": config_to_save.get("_mlx_task", "token-classification"),
    }
    try:
        weights_path = output_dir / "weights.safetensors"
        mx.save_safetensors(weights_path, mlx_weights, metadata=metadata)
        weights_format = "safetensors"
    except Exception as exc:
        logger.warning(
            "Could not save MLX weights as safetensors; falling back to npz: %s",
            exc,
        )
        weights_path = output_dir / "weights.npz"
        mx.savez(str(weights_path), **mlx_weights)
        weights_format = "npz"

    _cleanup_other_weight_files(weights_path)
    config_to_save["_mlx_weights_format"] = weights_format

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    if "id2label" in config_to_save:
        with open(output_dir / "id2label.json", "w") as f:
            json.dump(config_to_save["id2label"], f, indent=2)

    _finalize_artifact(
        output_dir,
        source_model_id=source_model_id,
        config=config_to_save,
        cache_dir=cache_dir,
    )

    logger.info("Saved MLX model to %s", output_dir)
    return output_dir


def save_numpy_model(
    weights: Dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
    source_model_id: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Save converted weights without MLX, preferring ``.safetensors``."""
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_to_save = dict(config)

    def _cleanup_other_weight_files(keep_path: Path) -> None:
        for candidate in (output_dir / "weights.safetensors", output_dir / "weights.npz"):
            if candidate != keep_path and candidate.exists():
                candidate.unlink()

    weights_format = "npz"
    weights_path = output_dir / "weights.npz"
    metadata = {
        "format": "mlx",
        "openmed_task": config_to_save.get("_mlx_task", "token-classification"),
    }
    try:
        from safetensors.numpy import save_file

        weights_path = output_dir / "weights.safetensors"
        safe_weights = {k: np.ascontiguousarray(v) for k, v in weights.items()}
        save_file(safe_weights, str(weights_path), metadata=metadata)
        weights_format = "safetensors"
    except Exception as exc:
        logger.warning(
            "Could not save NumPy weights as safetensors; falling back to npz: %s",
            exc,
        )
        weights_path = output_dir / "weights.npz"
        np.savez(str(weights_path), **weights)
        weights_format = "npz"

    _cleanup_other_weight_files(weights_path)
    config_to_save["_mlx_weights_format"] = weights_format

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    if "id2label" in config_to_save:
        with open(output_dir / "id2label.json", "w") as f:
            json.dump(config_to_save["id2label"], f, indent=2)

    _finalize_artifact(
        output_dir,
        source_model_id=source_model_id,
        config=config_to_save,
        cache_dir=cache_dir,
    )

    logger.info(
        "Saved MLX-compatible model to %s using %s weights",
        output_dir,
        weights_format,
    )
    return output_dir


def _finalize_artifact(
    output_dir: str | Path,
    *,
    source_model_id: str | None,
    config: dict[str, Any],
    cache_dir: str | None,
) -> None:
    if source_model_id is None:
        return

    output_dir = Path(output_dir)
    tokenizer_files: list[str] = []

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source_model_id, cache_dir=cache_dir)
        tokenizer.save_pretrained(output_dir)
        tokenizer_files = find_tokenizer_files(output_dir)
    except Exception as exc:
        logger.warning(
            "Could not save tokenizer assets for %s into %s: %s",
            source_model_id,
            output_dir,
            exc,
        )

    write_manifest(
        output_dir,
        source_model_id=source_model_id,
        config=config,
        tokenizer_files=tokenizer_files,
    )


def convert(
    model_id: str,
    output_dir: str | Path,
    quantize_bits: int | None = None,
    cache_dir: str | None = None,
) -> Path:
    """End-to-end: download a model, remap it, and save an OpenMed MLX artifact."""
    weights, config = convert_weights(model_id, cache_dir=cache_dir)

    try:
        import mlx.core  # noqa: F401
        return save_mlx_model(
            weights,
            config,
            output_dir,
            quantize_bits,
            source_model_id=model_id,
            cache_dir=cache_dir,
        )
    except ImportError:
        if quantize_bits is not None:
            logger.warning(
                "MLX not available — skipping quantization. "
                "Install mlx for quantization support."
            )
        return save_numpy_model(
            weights,
            config,
            output_dir,
            source_model_id=model_id,
            cache_dir=cache_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face token-classification model to OpenMed MLX format",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Source model ID or local directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for MLX model files",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantize weights to N bits (4 or 8)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert(args.model, args.output, args.quantize, args.cache_dir)


if __name__ == "__main__":
    main()
