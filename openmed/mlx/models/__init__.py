"""MLX model implementations for token classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SUPPORTED_MODEL_TYPES = {
    "bert": "bert",
    "distilbert": "bert",
    "electra": "bert",
    "roberta": "bert",
    "xlm-roberta": "bert",
    "xlm_roberta": "bert",
    "deberta": "deberta-v2",
    "deberta-v2": "deberta-v2",
}

_ARCHITECTURE_TYPE_HINTS = [
    ("ModernBert", "modernbert"),
    ("Longformer", "longformer"),
    ("EuroBert", "eurobert"),
    ("Qwen3", "qwen3"),
    ("DebertaV2", "deberta-v2"),
    ("Deberta", "deberta"),
    ("XLMRoberta", "xlm-roberta"),
    ("Roberta", "roberta"),
    ("DistilBert", "distilbert"),
    ("Electra", "electra"),
    ("Bert", "bert"),
]


def normalize_model_type(model_type: str | None) -> str | None:
    """Normalize Hugging Face model-type strings for internal dispatch."""
    if model_type is None:
        return None
    return model_type.replace("_", "-").lower()


def resolve_model_type(config_or_type: dict[str, Any] | str | None) -> str:
    """Resolve a config dict or model_type string to a supported MLX family."""
    model_type: str | None

    if isinstance(config_or_type, dict):
        model_type = config_or_type.get("_mlx_model_type") or config_or_type.get("model_type")
        if model_type is None:
            architectures = config_or_type.get("architectures", [])
            for needle, inferred_type in _ARCHITECTURE_TYPE_HINTS:
                if any(needle in architecture for architecture in architectures):
                    model_type = inferred_type
                    break
    else:
        model_type = config_or_type

    model_type = normalize_model_type(model_type)
    resolved = _SUPPORTED_MODEL_TYPES.get(model_type)
    if resolved is None:
        supported = ", ".join(sorted(_SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"Unsupported MLX model type: {model_type!r}. Supported types: {supported}."
        )
    return resolved


def normalize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill architecture-specific config aliases needed by the MLX backends."""
    normalized = dict(config)
    source_model_type = normalize_model_type(
        normalized.get("model_type") or normalized.get("_mlx_model_type")
    )

    normalized.setdefault("hidden_size", normalized.get("dim"))
    normalized.setdefault("num_attention_heads", normalized.get("n_heads"))
    normalized.setdefault("num_hidden_layers", normalized.get("n_layers"))
    normalized.setdefault("intermediate_size", normalized.get("hidden_dim"))
    normalized.setdefault(
        "hidden_dropout_prob",
        normalized.get("dropout", normalized.get("hidden_dropout_prob", 0.1)),
    )
    normalized.setdefault(
        "attention_probs_dropout_prob",
        normalized.get(
            "attention_dropout",
            normalized.get("attention_probs_dropout_prob", 0.1),
        ),
    )
    normalized.setdefault("layer_norm_eps", normalized.get("layer_norm_eps", 1e-12))

    if source_model_type == "distilbert":
        normalized.setdefault("type_vocab_size", 0)
        normalized.setdefault("_mlx_position_offset", 0)
    elif source_model_type in {"roberta", "xlm-roberta"}:
        normalized.setdefault("type_vocab_size", 1)
        normalized.setdefault("_mlx_position_offset", int(normalized.get("pad_token_id", 1)) + 1)
    else:
        normalized.setdefault("type_vocab_size", normalized.get("type_vocab_size", 2))
        normalized.setdefault("_mlx_position_offset", 0)

    return normalized


def build_model(config: dict[str, Any]):
    """Instantiate the appropriate MLX model for *config*."""
    config = normalize_model_config(config)
    model_type = resolve_model_type(config)

    if model_type == "bert":
        from openmed.mlx.models.bert_tc import BertForTokenClassification

        return BertForTokenClassification(config)

    if model_type == "deberta-v2":
        from openmed.mlx.models.deberta_v2_tc import DebertaV2ForTokenClassification

        return DebertaV2ForTokenClassification(config)

    raise AssertionError(f"Unhandled MLX model type: {model_type}")


def _is_quantized_checkpoint(weights: dict[str, Any]) -> bool:
    """Detect MLX quantized checkpoints by their auxiliary scale tensors."""
    return any(key.endswith(".scales") for key in weights)


def _quantize_model_for_weights(config: dict[str, Any], weights: dict[str, Any]):
    """Instantiate a model matching the quantized checkpoint layout."""
    import mlx.nn as nn

    quantization = config.get("_mlx_quantization") or {}
    candidate_bits = []

    bits = quantization.get("bits")
    if bits is not None:
        candidate_bits.append(bits)

    for fallback_bits in (8, 4):
        if fallback_bits not in candidate_bits:
            candidate_bits.append(fallback_bits)

    last_error: Exception | None = None
    for bits in candidate_bits:
        model = build_model(config)
        nn.quantize(model, bits=bits)
        try:
            model.load_weights(list(weights.items()))
            return model
        except ValueError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    return build_model(config)


def load_model(model_path: str | Path):
    """Load a converted MLX token-classification model from *model_path*."""
    try:
        import mlx.core as mx
    except ImportError:
        raise ImportError(
            "MLX is required for this module. "
            "Install with: pip install openmed[mlx]"
        )

    model_path = Path(model_path)

    with open(model_path / "config.json") as f:
        config = json.load(f)
    config = normalize_model_config(config)

    weights_npz = model_path / "weights.npz"
    weights_sf = model_path / "weights.safetensors"
    if weights_sf.exists():
        from mlx.utils import load as mlx_load

        weights = dict(mlx_load(str(weights_sf)))
    elif weights_npz.exists():
        weights = dict(mx.load(str(weights_npz)))
    else:
        raise FileNotFoundError(
            f"No weights found in {model_path}. "
            "Expected weights.npz or weights.safetensors."
        )

    if _is_quantized_checkpoint(weights):
        model = _quantize_model_for_weights(config, weights)
    else:
        model = build_model(config)
        model.load_weights(list(weights.items()))

    model.eval()
    mx.eval(model.parameters())
    return model
