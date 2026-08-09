"""ModernBERT token classification implemented in Apple MLX.

The module mirrors the encoder used by Hugging Face ``ModernBertForTokenClassification``.
ModernBERT combines full-attention and bidirectional sliding-attention layers,
uses rotary position embeddings instead of learned position embeddings, and
places a small prediction head before the token classifier.
"""

from __future__ import annotations

from typing import Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError(
        "MLX is required for this module. Install with: pip install openmed[mlx]"
    )


def _layer_type(config: dict, layer_idx: int) -> str:
    """Return a normalized ModernBERT attention type for one layer."""
    layer_types = config.get("layer_types") or []
    if layer_idx < len(layer_types):
        value = str(layer_types[layer_idx]).lower()
        if value in {"sliding_attention", "sliding", "local"}:
            return "sliding_attention"
    return "full_attention"


def _rope_theta(config: dict, layer_type: str) -> float:
    """Resolve the rotary base for a full or sliding attention layer."""
    rope_parameters = config.get("rope_parameters") or {}
    parameters = rope_parameters
    if isinstance(rope_parameters, dict):
        parameters = (
            rope_parameters.get(layer_type)
            or rope_parameters.get(
                "global" if layer_type == "full_attention" else "local"
            )
            or rope_parameters
        )
    if not isinstance(parameters, dict):
        parameters = {}

    default = 160000.0 if layer_type == "full_attention" else 10000.0
    return float(
        parameters.get(
            "rope_theta",
            config.get(
                "global_rope_theta"
                if layer_type == "full_attention"
                else "local_rope_theta",
                config.get("rope_theta", default),
            ),
        )
    )


def _rotary_embeddings(
    position_ids: mx.array,
    head_dim: int,
    config: dict,
    layer_type: str,
) -> tuple[mx.array, mx.array]:
    """Build the cosine and sine tables used by ModernBERT attention."""
    if head_dim % 2:
        raise ValueError("ModernBERT rotary attention requires an even head dimension")

    theta = _rope_theta(config, layer_type)
    frequencies = mx.arange(0, head_dim, 2, dtype=mx.float32)
    inverse_frequency = 1.0 / mx.power(theta, frequencies / float(head_dim))
    positions = position_ids.astype(mx.float32)[..., None]
    angles = positions * inverse_frequency[None, None, :]
    angles = mx.concatenate((angles, angles), axis=-1)
    return mx.cos(angles), mx.sin(angles)


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate the final dimension by half, matching Hugging Face RoPE."""
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply a rotary embedding while preserving the projection dtype."""
    original_dtype = x.dtype
    x = x.astype(mx.float32)
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (x * cos + _rotate_half(x) * sin).astype(original_dtype)


class ModernBertEmbeddings(nn.Module):
    """ModernBERT token embeddings and input normalization."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            int(config["vocab_size"]), int(config["hidden_size"])
        )
        self.norm = nn.LayerNorm(
            int(config["hidden_size"]),
            eps=float(config.get("norm_eps", 1e-5)),
            bias=bool(config.get("norm_bias", False)),
        )
        self.dropout = nn.Dropout(p=float(config.get("embedding_dropout", 0.0)))

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds is required")
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.word_embeddings(input_ids)
        )
        return self.dropout(self.norm(hidden_states))


class ModernBertAttention(nn.Module):
    """Fused-QKV rotary attention for one ModernBERT layer."""

    def __init__(self, config: dict, layer_idx: int) -> None:
        super().__init__()
        self.num_heads = int(config["num_attention_heads"])
        self.hidden_size = int(config["hidden_size"])
        self.head_dim = int(config.get("head_dim", self.hidden_size // self.num_heads))
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("ModernBERT hidden size must equal heads times head dim")

        attention_bias = bool(config.get("attention_bias", False))
        self.qkv_proj = nn.Linear(
            self.hidden_size, 3 * self.num_heads * self.head_dim, bias=attention_bias
        )
        self.out_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=attention_bias
        )
        self.layer_type = _layer_type(config, layer_idx)
        self.sliding_window = max(
            0,
            int(
                config.get(
                    "sliding_window",
                    int(config.get("local_attention", 128)) // 2,
                )
            ),
        )
        self.dropout = nn.Dropout(p=float(config.get("attention_dropout", 0.0)))
        self.output_dropout = nn.Dropout(p=float(config.get("attention_dropout", 0.0)))

    def _mask_scores(
        self,
        scores: mx.array,
        attention_mask: Optional[mx.array],
    ) -> mx.array:
        """Apply padding and, for local layers, sliding-window masks."""
        _, _, seq_len, _ = scores.shape
        if attention_mask is None:
            valid_keys = None
        elif attention_mask.ndim == 2:
            valid_keys = attention_mask > 0
        elif attention_mask.ndim == 4:
            return scores + attention_mask
        else:
            raise ValueError("ModernBERT attention_mask must have rank 2 or 4")

        if self.layer_type == "sliding_attention":
            positions = mx.arange(seq_len, dtype=mx.int32)
            local = (
                mx.abs(positions[:, None] - positions[None, :]) <= self.sliding_window
            )
            allowed = local[None, None, :, :]
        else:
            allowed = mx.ones((1, 1, seq_len, seq_len), dtype=mx.bool_)

        if valid_keys is not None:
            allowed = allowed & valid_keys[:, None, None, :]

        min_value = mx.array(mx.finfo(scores.dtype).min, dtype=scores.dtype)
        return mx.where(allowed, scores, min_value)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        query = qkv[:, :, 0].transpose(0, 2, 1, 3)
        key = qkv[:, :, 1].transpose(0, 2, 1, 3)
        value = qkv[:, :, 2].transpose(0, 2, 1, 3)
        cos, sin = position_embeddings
        query = _apply_rotary(query, cos, sin)
        key = _apply_rotary(key, cos, sin)

        scores = (query @ key.transpose(0, 1, 3, 2)) * (self.head_dim**-0.5)
        scores = self._mask_scores(scores, attention_mask)
        probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
            scores.dtype
        )
        probabilities = self.dropout(probabilities)
        output = (probabilities @ value).transpose(0, 2, 1, 3)
        return self.output_dropout(
            self.out_proj(output.reshape(batch_size, seq_len, self.hidden_size))
        )


def _activation(x: mx.array, name: str) -> mx.array:
    """Apply one of the activations exposed by ModernBERT configs."""
    normalized = name.lower().replace("-", "_")
    if normalized in {"gelu", "gelu_new", "gelu_fast"}:
        return nn.gelu(x) if normalized == "gelu" else nn.gelu_approx(x)
    if normalized in {"silu", "swish"}:
        return nn.silu(x)
    if normalized == "relu":
        return nn.relu(x)
    raise ValueError(f"Unsupported ModernBERT activation: {name!r}")


class ModernBertMLP(nn.Module):
    """Gated ModernBERT feed-forward block."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        intermediate_size = int(config["intermediate_size"])
        bias = bool(config.get("mlp_bias", False))
        self.wi_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=bias)
        self.wo_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.activation = str(
            config.get("hidden_activation", config.get("hidden_act", "gelu"))
        )
        self.dropout = nn.Dropout(p=float(config.get("mlp_dropout", 0.0)))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_part, gate = mx.split(self.wi_proj(hidden_states), 2, axis=-1)
        hidden_states = _activation(input_part, self.activation) * gate
        return self.wo_proj(self.dropout(hidden_states))


class ModernBertEncoderLayer(nn.Module):
    """One pre-normalized ModernBERT encoder layer."""

    def __init__(self, config: dict, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = (
            nn.Identity()
            if layer_idx == 0
            else nn.LayerNorm(
                int(config["hidden_size"]),
                eps=float(config.get("norm_eps", 1e-5)),
                bias=bool(config.get("norm_bias", False)),
            )
        )
        self.attention = ModernBertAttention(config, layer_idx)
        self.mlp_norm = nn.LayerNorm(
            int(config["hidden_size"]),
            eps=float(config.get("norm_eps", 1e-5)),
            bias=bool(config.get("norm_bias", False)),
        )
        self.mlp = ModernBertMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        attention_output = self.attention(
            self.attn_norm(hidden_states), position_embeddings, attention_mask
        )
        hidden_states = hidden_states + attention_output
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class ModernBertEncoder(nn.Module):
    """Stack of ModernBERT encoder layers."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.layers = [
            ModernBertEncoderLayer(config, layer_idx)
            for layer_idx in range(int(config["num_hidden_layers"]))
        ]
        self.config = config

    def __call__(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        for layer_idx, layer in enumerate(self.layers):
            layer_type = _layer_type(self.config, layer_idx)
            position_embeddings = _rotary_embeddings(
                position_ids,
                layer.attention.head_dim,
                self.config,
                layer_type,
            )
            hidden_states = layer(
                hidden_states, position_embeddings, attention_mask=attention_mask
            )
        return hidden_states


class ModernBertPredictionHead(nn.Module):
    """ModernBERT's dense, activation, and normalization prediction head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.dense = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bool(config.get("classifier_bias", False)),
        )
        self.activation = str(
            config.get("classifier_activation", config.get("hidden_act", "gelu"))
        )
        self.norm = nn.LayerNorm(
            hidden_size,
            eps=float(config.get("norm_eps", 1e-5)),
            bias=bool(config.get("norm_bias", False)),
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.norm(_activation(self.dense(hidden_states), self.activation))


class ModernBertModel(nn.Module):
    """ModernBERT encoder without a task-specific head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embeddings = ModernBertEmbeddings(config)
        self.encoder = ModernBertEncoder(config)
        self.final_norm = nn.LayerNorm(
            int(config["hidden_size"]),
            eps=float(config.get("norm_eps", 1e-5)),
            bias=bool(config.get("norm_bias", False)),
        )
        self.config = config

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.embeddings(input_ids, inputs_embeds)
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = mx.broadcast_to(
                mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len)
            )
        hidden_states = self.encoder(hidden_states, position_ids, attention_mask)
        return self.final_norm(hidden_states)


class ModernBertForTokenClassification(nn.Module):
    """ModernBERT with a token-classification logits head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.dropout = nn.Dropout(p=float(config.get("classifier_dropout", 0.0)))
        self.classifier = nn.Linear(
            int(config["hidden_size"]), int(config["num_labels"])
        )
        self.config = config

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **_: object,
    ) -> mx.array:
        del token_type_ids
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = self.dropout(self.head(hidden_states))
        return self.classifier(hidden_states)
