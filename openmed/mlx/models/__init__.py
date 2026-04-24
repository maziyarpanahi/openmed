"""MLX model implementations for token classification and zero-shot tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openmed.mlx.artifact import load_artifact_config, resolve_weight_candidates

_SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES = {
    "bert": "bert",
    "distilbert": "bert",
    "electra": "bert",
    "openai-privacy-filter": "openai-privacy-filter",
    "privacy-filter": "openai-privacy-filter",
    "roberta": "bert",
    "xlm-roberta": "bert",
    "xlm_roberta": "bert",
    "deberta": "deberta-v2",
    "deberta-v2": "deberta-v2",
}

_CUSTOM_FAMILIES = {
    "gliner-uni-encoder-span",
    "gliclass-uni-encoder",
    "gliner-uni-encoder-token-relex",
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


def resolve_artifact_task(
    config: dict[str, Any] | None,
    manifest: dict[str, Any] | None = None,
) -> str:
    """Resolve an MLX artifact task name with backward-compatible defaults."""
    if manifest is not None:
        task = manifest.get("task")
        if task:
            return str(task)
    if config is not None:
        task = config.get("_mlx_task")
        if task:
            return str(task)
    return "token-classification"


def resolve_artifact_family(
    config: dict[str, Any] | str | None,
    manifest: dict[str, Any] | None = None,
) -> str:
    """Resolve a config dict, manifest, or family string to an MLX family."""
    if manifest is not None:
        family = normalize_model_type(manifest.get("family"))
        if family in _CUSTOM_FAMILIES:
            return family
        resolved_family = _SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES.get(family)
        if resolved_family is not None:
            return resolved_family

    model_type: str | None = None
    if isinstance(config, dict):
        family = normalize_model_type(config.get("_mlx_family"))
        if family in _CUSTOM_FAMILIES:
            return family
        resolved_family = _SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES.get(family)
        if resolved_family is not None:
            return resolved_family

        model_type = config.get("_mlx_model_type") or config.get("model_type")
        if model_type is None:
            architectures = config.get("architectures", [])
            for needle, inferred_type in _ARCHITECTURE_TYPE_HINTS:
                if any(needle in architecture for architecture in architectures):
                    model_type = inferred_type
                    break
    else:
        model_type = config

    normalized = normalize_model_type(model_type)
    resolved = _SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES.get(normalized)
    if resolved is None:
        supported = ", ".join(
            sorted(_CUSTOM_FAMILIES | set(_SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES))
        )
        raise ValueError(
            f"Unsupported MLX model family/type: {normalized!r}. Supported: {supported}."
        )
    return resolved


def resolve_model_type(
    config: dict[str, Any] | str | None,
    manifest: dict[str, Any] | None = None,
) -> str:
    """Backward-compatible alias for architecture-family resolution."""
    return resolve_artifact_family(config, manifest=manifest)


def normalize_model_config(
    config: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill architecture-specific config aliases needed by the MLX backends."""
    normalized = dict(config)
    task = resolve_artifact_task(normalized, manifest=manifest)
    family = resolve_artifact_family(normalized, manifest=manifest)

    normalized.setdefault("_mlx_task", task)
    normalized.setdefault("_mlx_family", family)

    if family in _CUSTOM_FAMILIES:
        normalized.setdefault(
            "encoder_hidden_size",
            normalized.get("encoder_hidden_size", normalized.get("hidden_size", 768)),
        )
        normalized.setdefault("dropout", normalized.get("hidden_dropout_prob", 0.1))
        normalized.setdefault("embed_ent_token", normalized.get("embed_ent_token", True))
        normalized.setdefault("embed_class_token", normalized.get("embed_class_token", True))
        normalized.setdefault("num_rnn_layers", normalized.get("num_rnn_layers", 0) or 0)
        normalized.setdefault("class_token_index", normalized.get("class_token_index"))
        normalized.setdefault("rel_token_index", normalized.get("rel_token_index"))
        normalized.setdefault("text_token_index", normalized.get("text_token_index"))
        normalized.setdefault("example_token_index", normalized.get("example_token_index"))
        normalized.setdefault("max_width", normalized.get("max_width", 12))
        normalized.setdefault("pooling_strategy", normalized.get("pooling_strategy", "first"))
        normalized.setdefault(
            "class_token_pooling",
            normalized.get("class_token_pooling", "first"),
        )
        normalized.setdefault(
            "logit_scale_init_value",
            normalized.get("logit_scale_init_value", 1.0),
        )

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
    elif family == "openai-privacy-filter":
        normalized.setdefault("num_experts", normalized.get("num_local_experts", 128))
        normalized.setdefault("experts_per_token", normalized.get("num_experts_per_tok", 4))
        normalized.setdefault("swiglu_limit", normalized.get("swiglu_limit", 7.0))
        normalized.setdefault("rms_norm_eps", normalized.get("rms_norm_eps", 1e-5))
        rope_parameters = normalized.get("rope_parameters") or {}
        normalized.setdefault("rope_theta", rope_parameters.get("rope_theta", 150000.0))
        normalized.setdefault("rope_scaling_factor", rope_parameters.get("factor", 1.0))
        normalized.setdefault("rope_ntk_alpha", rope_parameters.get("beta_slow", 1.0))
        normalized.setdefault("rope_ntk_beta", rope_parameters.get("beta_fast", 32.0))
        normalized.setdefault(
            "initial_context_length",
            rope_parameters.get(
                "original_max_position_embeddings",
                normalized.get("initial_context_length", 4096),
            ),
        )
        sliding_window = normalized.get("sliding_window")
        default_context = (
            max(0, (int(sliding_window) - 1) // 2)
            if sliding_window is not None
            else 128
        )
        normalized.setdefault(
            "bidirectional_left_context",
            normalized.get("bidirectional_left_context", default_context),
        )
        normalized.setdefault(
            "bidirectional_right_context",
            normalized.get("bidirectional_right_context", default_context),
        )
    else:
        normalized.setdefault("type_vocab_size", normalized.get("type_vocab_size", 2))
        normalized.setdefault("_mlx_position_offset", 0)

    return normalized


def build_model(
    config: dict[str, Any],
    manifest: dict[str, Any] | None = None,
):
    """Instantiate the appropriate MLX model for *config* and *manifest*."""
    config = normalize_model_config(config, manifest=manifest)
    family = resolve_artifact_family(config, manifest=manifest)

    if family == "bert":
        from openmed.mlx.models.bert_tc import BertForTokenClassification

        return BertForTokenClassification(config)

    if family == "deberta-v2" and resolve_artifact_task(config, manifest=manifest) == "token-classification":
        from openmed.mlx.models.deberta_v2_tc import DebertaV2ForTokenClassification

        return DebertaV2ForTokenClassification(config)

    if family == "openai-privacy-filter":
        from openmed.mlx.models.privacy_filter import OpenAIPrivacyFilterForTokenClassification

        return OpenAIPrivacyFilterForTokenClassification(config)

    if family == "gliner-uni-encoder-span":
        from openmed.mlx.models.gliner_span import GLiNERSpanModel

        return GLiNERSpanModel(config)

    if family == "gliclass-uni-encoder":
        from openmed.mlx.models.gliclass_uni import GLiClassUniEncoderModel

        return GLiClassUniEncoderModel(config)

    if family == "gliner-uni-encoder-token-relex":
        from openmed.mlx.models.gliner_relex import GLiNERRelexModel

        return GLiNERRelexModel(config)

    raise AssertionError(f"Unhandled MLX model family: {family}")


def _is_quantized_checkpoint(weights: dict[str, Any]) -> bool:
    """Detect MLX quantized checkpoints by their auxiliary scale tensors."""
    return any(key.endswith(".scales") for key in weights)


def _quantize_model_for_weights(
    config: dict[str, Any],
    weights: dict[str, Any],
    manifest: dict[str, Any] | None = None,
):
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
        model = build_model(config, manifest=manifest)
        nn.quantize(model, bits=bits)
        try:
            model.load_weights(list(weights.items()))
            return model
        except ValueError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    return build_model(config, manifest=manifest)


def load_model(model_path: str | Path):
    """Load a converted MLX model from *model_path*."""
    try:
        import mlx.core as mx
    except ImportError:
        raise ImportError(
            "MLX is required for this module. "
            "Install with: pip install openmed[mlx]"
        )

    model_path = Path(model_path)

    manifest, config = load_artifact_config(model_path)
    config = normalize_model_config(config, manifest=manifest)

    candidate_paths = resolve_weight_candidates(model_path, config=config, manifest=manifest)
    weights_path = next((path for path in candidate_paths if path.exists()), None)
    if weights_path is None:
        raise FileNotFoundError(
            f"No weights found in {model_path}. "
            "Expected weights.safetensors or weights.npz."
        )

    if weights_path.suffix in {".safetensors", ".npz"}:
        weights = dict(mx.load(str(weights_path)))
    else:
        raise FileNotFoundError(
            f"Unsupported MLX weight file: {weights_path}. "
            "Expected weights.safetensors or weights.npz."
        )

    if _is_quantized_checkpoint(weights):
        model = _quantize_model_for_weights(config, weights, manifest=manifest)
    else:
        model = build_model(config, manifest=manifest)
        model.load_weights(list(weights.items()))

    model.eval()
    mx.eval(model.parameters())
    return model
