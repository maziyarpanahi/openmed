"""Convert HuggingFace token-classification models to MLX format.

Usage::

    python3 -m openmed.mlx.convert \\
        --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \\
        --output ./mlx-models/pii-small \\
        --quantize 8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

from openmed.mlx.models import (
    build_model,
    normalize_model_config,
    normalize_model_type,
    resolve_model_type,
)

logger = logging.getLogger(__name__)

# ---- Weight key remapping ---------------------------------------------------

_BERT_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    # Attention projections
    (".attention.self.query.", ".attention.query_proj."),
    (".attention.self.key.", ".attention.key_proj."),
    (".attention.self.value.", ".attention.value_proj."),
    (".attention.output.dense.", ".attention.out_proj."),
    # Attention LayerNorm → ln1
    (".attention.output.LayerNorm.", ".ln1."),
    # Feed-forward
    (".intermediate.dense.", ".linear1."),
    (".output.dense.", ".linear2."),
    # Output LayerNorm → ln2
    (".output.LayerNorm.", ".ln2."),
    # Encoder layers
    ("bert.encoder.layer.", "encoder.layers."),
    # Embeddings
    ("bert.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("bert.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("bert.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("bert.embeddings.LayerNorm.", "embeddings.norm."),
    # Classification head (keep as-is)
    ("classifier.", "classifier."),
    # Pooler (not needed for TC, but remap to avoid warnings)
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
    # Encoder layers
    ("roberta.encoder.layer.", "encoder.layers."),
    ("xlm_roberta.encoder.layer.", "encoder.layers."),
    # Embeddings
    ("roberta.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("roberta.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("roberta.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("roberta.embeddings.LayerNorm.", "embeddings.norm."),
    ("xlm_roberta.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("xlm_roberta.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("xlm_roberta.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("xlm_roberta.embeddings.LayerNorm.", "embeddings.norm."),
    # Pooler
    ("roberta.pooler.", "_pooler."),
    ("xlm_roberta.pooler.", "_pooler."),
]

_DISTILBERT_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    # Attention projections
    (".attention.q_lin.", ".attention.query_proj."),
    (".attention.k_lin.", ".attention.key_proj."),
    (".attention.v_lin.", ".attention.value_proj."),
    (".attention.out_lin.", ".attention.out_proj."),
    # Norms
    (".sa_layer_norm.", ".ln1."),
    (".output_layer_norm.", ".ln2."),
    # Feed-forward
    (".ffn.lin1.", ".linear1."),
    (".ffn.lin2.", ".linear2."),
    # Encoder layers
    ("distilbert.transformer.layer.", "encoder.layers."),
    # Embeddings
    ("distilbert.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("distilbert.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("distilbert.embeddings.LayerNorm.", "embeddings.norm."),
]

_ELECTRA_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    # Attention projections
    (".attention.self.query.", ".attention.query_proj."),
    (".attention.self.key.", ".attention.key_proj."),
    (".attention.self.value.", ".attention.value_proj."),
    (".attention.output.dense.", ".attention.out_proj."),
    # Attention LayerNorm → ln1
    (".attention.output.LayerNorm.", ".ln1."),
    # Feed-forward
    (".intermediate.dense.", ".linear1."),
    (".output.dense.", ".linear2."),
    # Output LayerNorm → ln2
    (".output.LayerNorm.", ".ln2."),
    # Encoder layers
    ("electra.encoder.layer.", "encoder.layers."),
    # Embeddings
    ("electra.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("electra.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("electra.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("electra.embeddings.LayerNorm.", "embeddings.norm."),
    # Classification head (keep as-is)
    ("classifier.", "classifier."),
]


def _infer_source_model_type(key: str, model_type: str | None) -> str:
    """Infer the original HF model type for key remapping purposes."""
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


def convert_weights(model_id: str, cache_dir: str | None = None) -> Tuple[Dict, dict]:
    """Load HF weights and config, remap keys for MLX.

    Returns:
        ``(weights_dict, config_dict)`` where *weights_dict* maps MLX key
        names to numpy arrays.
    """
    try:
        from transformers import AutoConfig, AutoModelForTokenClassification
    except ImportError:
        raise ImportError(
            "transformers is required for model conversion. "
            "Install with: pip install transformers"
        )

    logger.info("Loading HuggingFace model %s ...", model_id)
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, cache_dir=cache_dir,
    )

    source_model_type = normalize_model_type(config.model_type)
    resolved_model_type = resolve_model_type(config.to_dict())
    state_dict = model.state_dict()
    mlx_weights = {}
    skipped = []

    for hf_key, tensor in state_dict.items():
        mlx_key = remap_key(hf_key, source_model_type)
        if mlx_key.startswith("_"):
            skipped.append(hf_key)
            continue
        mlx_weights[mlx_key] = tensor.detach().cpu().numpy()

    if skipped:
        logger.info("Skipped %d keys (pooler, etc.): %s", len(skipped), skipped[:5])

    config_dict = normalize_model_config(config.to_dict())
    config_dict["_mlx_model_type"] = resolved_model_type
    # Ensure num_labels is present
    config_dict.setdefault("num_labels", config.num_labels)

    return mlx_weights, config_dict


def save_mlx_model(
    weights: Dict,
    config: dict,
    output_dir: str | Path,
    quantize_bits: int | None = None,
) -> Path:
    """Save converted weights and config to *output_dir*.

    Args:
        weights: Remapped weight dict (key → numpy array).
        config: Model config dict.
        output_dir: Destination directory.
        quantize_bits: If set (4 or 8), quantize weights.

    Returns:
        Path to the output directory.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
    except ImportError:
        raise ImportError("MLX is required. Install with: pip install openmed[mlx]")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_to_save = dict(config)

    # Convert numpy arrays to MLX arrays
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}

    if quantize_bits is not None:
        logger.info("Quantizing to %d bits ...", quantize_bits)
        model = build_model(config)
        model.load_weights(list(mlx_weights.items()))
        nn.quantize(model, bits=quantize_bits)
        # Re-extract weights after quantization (tree_flatten returns flat list)
        mlx_weights = dict(tree_flatten(model.parameters()))
        config_to_save["_mlx_quantization"] = {"bits": quantize_bits}
    else:
        config_to_save.pop("_mlx_quantization", None)

    # Save weights
    weights_path = output_dir / "weights.npz"
    mx.savez(str(weights_path), **mlx_weights)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)

    # Save id2label if present
    if "id2label" in config_to_save:
        id2label_path = output_dir / "id2label.json"
        with open(id2label_path, "w") as f:
            json.dump(config_to_save["id2label"], f, indent=2)

    logger.info("Saved MLX model to %s", output_dir)
    return output_dir


def save_numpy_model(
    weights: Dict,
    config: dict,
    output_dir: str | Path,
) -> Path:
    """Save converted weights as plain NumPy ``.npz`` — no MLX required.

    This fallback is useful when converting on a machine without MLX
    (e.g., Linux CI).  The resulting files are identical in structure to
    :func:`save_mlx_model` and can be loaded by the MLX backend.
    """
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "weights.npz"
    np.savez(str(weights_path), **weights)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if "id2label" in config:
        id2label_path = output_dir / "id2label.json"
        with open(id2label_path, "w") as f:
            json.dump(config["id2label"], f, indent=2)

    logger.info("Saved NumPy model to %s (MLX-compatible)", output_dir)
    return output_dir


def convert(
    model_id: str,
    output_dir: str | Path,
    quantize_bits: int | None = None,
    cache_dir: str | None = None,
) -> Path:
    """End-to-end: download HF model → remap → save MLX format.

    If MLX is installed, uses MLX for saving (and optional quantization).
    Otherwise, falls back to plain NumPy format (no quantization).

    Args:
        model_id: HuggingFace model identifier.
        output_dir: Destination directory for MLX model.
        quantize_bits: Optional quantization (4 or 8 bits, requires MLX).
        cache_dir: HuggingFace cache directory.

    Returns:
        Path to the output directory.
    """
    weights, config = convert_weights(model_id, cache_dir=cache_dir)

    try:
        import mlx.core  # noqa: F401
        return save_mlx_model(weights, config, output_dir, quantize_bits)
    except ImportError:
        if quantize_bits is not None:
            logger.warning(
                "MLX not available — skipping quantization. "
                "Install mlx for quantization support."
            )
        return save_numpy_model(weights, config, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace token-classification model to MLX format",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID (e.g. OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for MLX model files",
    )
    parser.add_argument(
        "--quantize", type=int, choices=[4, 8], default=None,
        help="Quantize weights to N bits (4 or 8)",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace model cache directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert(args.model, args.output, args.quantize, args.cache_dir)


if __name__ == "__main__":
    main()
