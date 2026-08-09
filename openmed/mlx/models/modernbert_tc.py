"""ModernBERT token classification implemented in Apple MLX.

The implementation mirrors the inference path of Hugging Face
``ModernBertForTokenClassification`` for converted checkpoints. ModernBERT
uses rotary positions, alternating global and sliding-window attention, and a
gated MLP rather than the absolute-position BERT block.
"""

from __future__ import annotations

import math
from typing import Any, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError(
        "MLX is required for this module. Install with: pip install openmed[mlx]"
    )


class _Identity(nn.Module):
    """Parameter-free identity layer used by ModernBERT's first block."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        return x


class _LayerNorm(nn.Module):
    """LayerNorm with the optional bias used by ModernBERT checkpoints."""

    def __init__(self, dimensions: int, eps: float, bias: bool) -> None:
        super().__init__()
        self.weight = mx.ones((dimensions,))
        self.use_bias = bool(bias)
        if self.use_bias:
            self.bias = mx.zeros((dimensions,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        normalized = centered * mx.rsqrt(
            mx.mean(centered * centered, axis=-1, keepdims=True) + self.eps
        )
        normalized = normalized * self.weight
        if self.use_bias:
            normalized = normalized + self.bias
        return normalized


def _activation(x: mx.array, name: str) -> mx.array:
    """Apply a Hugging Face-compatible ModernBERT activation."""
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"gelu", "gelu_exact"}:
        return nn.gelu(x)
    if normalized in {"gelu_new", "gelu_fast", "gelu_pytorch_tanh"}:
        return (
            0.5
            * x
            * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
        )
    if normalized in {"silu", "swish"}:
        return nn.silu(x)
    if normalized == "relu":
        return nn.relu(x)
    raise ValueError(f"Unsupported ModernBERT activation: {name!r}")


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate the first and second halves of the head dimension."""
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


class ModernBertRotaryEmbedding(nn.Module):
    """Generate the non-interleaved rotary embeddings used by ModernBERT."""

    def __init__(self, dimension: int, base: float) -> None:
        super().__init__()
        self.dimension = dimension
        self.base = float(base)

    def __call__(
        self,
        x: mx.array,
        position_ids: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        """Return cosine and sine tensors broadcastable over attention heads."""
        seq_len = x.shape[-3]
        if position_ids is None:
            position_ids = mx.arange(seq_len, dtype=mx.float32)[None, :]
        else:
            position_ids = position_ids.astype(mx.float32)

        exponent = mx.arange(
            0,
            self.dimension,
            2,
            dtype=mx.float32,
        ) / float(self.dimension)
        inv_freq = mx.exp(-mx.log(mx.array(self.base, dtype=mx.float32)) * exponent)
        freqs = position_ids[..., None] * inv_freq
        embedding = mx.concatenate((freqs, freqs), axis=-1)
        return mx.cos(embedding).astype(x.dtype), mx.sin(embedding).astype(x.dtype)


class ModernBertEmbeddings(nn.Module):
    """Token embeddings followed by ModernBERT's input normalization."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            config["vocab_size"],
            config["hidden_size"],
        )
        self.norm = _LayerNorm(
            config["hidden_size"],
            eps=config.get("layer_norm_eps", 1e-5),
            bias=config.get("norm_bias", False),
        )
        self.drop = nn.Dropout(p=float(config.get("embedding_dropout", 0.0)))

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


class ModernBertMLP(nn.Module):
    """Gated ModernBERT feed-forward block."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        bias = bool(config.get("mlp_bias", False))
        self.Wi = nn.Linear(hidden_size, intermediate_size * 2, bias=bias)
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.activation = config.get(
            "hidden_act",
            config.get("hidden_activation", "gelu"),
        )
        self.drop = nn.Dropout(p=float(config.get("mlp_dropout", 0.0)))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        projected = self.Wi(hidden_states)
        split = projected.shape[-1] // 2
        input_states = projected[..., :split]
        gate_states = projected[..., split:]
        activated = _activation(input_states, self.activation) * gate_states
        return self.Wo(self.drop(activated))


class ModernBertAttention(nn.Module):
    """Multi-head rotary self-attention with optional local masking."""

    def __init__(self, config: dict, layer_id: int) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        self.num_heads = int(config["num_attention_heads"])
        self.head_dim = hidden_size // self.num_heads
        if self.head_dim * self.num_heads != hidden_size:
            raise ValueError(
                "ModernBERT hidden_size must be divisible by num_attention_heads"
            )

        attention_bias = bool(config.get("attention_bias", False))
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=attention_bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.attention_dropout = float(config.get("attention_dropout", 0.0))
        self.drop = nn.Dropout(p=self.attention_dropout)
        self.out_drop = nn.Dropout(p=self.attention_dropout)

        global_every = max(1, int(config.get("global_attn_every_n_layers", 3)))
        is_local = layer_id % global_every != 0
        if is_local:
            local_window = int(config.get("local_attention", 128))
            self.local_radius = max(0, local_window // 2)
            rope_theta = config.get("local_rope_theta") or config.get(
                "global_rope_theta", 160000.0
            )
        else:
            self.local_radius = None
            rope_theta = config.get("global_rope_theta") or 160000.0
        self.rotary_emb = ModernBertRotaryEmbedding(self.head_dim, rope_theta)

    def _attention_mask(
        self,
        attention_mask: Optional[mx.array],
        sequence_length: int,
        dtype: Any,
    ) -> Optional[mx.array]:
        if attention_mask is None and self.local_radius is None:
            return None

        negative = mx.array(mx.finfo(dtype).min, dtype=dtype)
        if attention_mask is None:
            mask = mx.zeros((1, 1, 1, sequence_length), dtype=dtype)
        else:
            valid = attention_mask > 0
            mask = mx.where(
                valid[:, None, None, :],
                mx.array(0.0, dtype=dtype),
                negative,
            )

        if self.local_radius is not None:
            positions = mx.arange(sequence_length, dtype=mx.int32)
            distance = mx.abs(positions[:, None] - positions[None, :])
            local = distance <= self.local_radius
            mask = mx.where(local[None, None, :, :], mask, negative)
        return mask

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.transpose(0, 3, 2, 1, 4)
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        cos, sin = self.rotary_emb(query, position_ids=position_ids)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin

        scale = self.head_dim**-0.5
        scores = (query @ key.transpose(0, 1, 3, 2)) * scale
        mask = self._attention_mask(
            attention_mask,
            sequence_length,
            scores.dtype,
        )
        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        weights = self.drop(weights)
        output = (
            (weights @ value)
            .transpose(0, 2, 1, 3)
            .reshape(
                batch_size,
                sequence_length,
                self.num_heads * self.head_dim,
            )
        )
        return self.out_drop(self.Wo(output))


class ModernBertEncoderLayer(nn.Module):
    """One ModernBERT pre-normalized residual encoder block."""

    def __init__(self, config: dict, layer_id: int) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        eps = config.get("layer_norm_eps", 1e-5)
        self.attn_norm = (
            _Identity()
            if layer_id == 0
            else _LayerNorm(hidden_size, eps=eps, bias=config.get("norm_bias", False))
        )
        self.attn = ModernBertAttention(config, layer_id)
        self.mlp_norm = _LayerNorm(
            hidden_size,
            eps=eps,
            bias=config.get("norm_bias", False),
        )
        self.mlp = ModernBertMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class ModernBertModel(nn.Module):
    """ModernBERT encoder without the token-classification head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [
            ModernBertEncoderLayer(config, layer_id)
            for layer_id in range(int(config["num_hidden_layers"]))
        ]
        self.final_norm = _LayerNorm(
            config["hidden_size"],
            eps=config.get("layer_norm_eps", 1e-5),
            bias=config.get("norm_bias", False),
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        sequence_length = input_ids.shape[1]
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.float32)
        if position_ids is None:
            position_ids = mx.arange(sequence_length, dtype=mx.int32)[None, :]

        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return self.final_norm(hidden_states)


class ModernBertPredictionHead(nn.Module):
    """Token-classification projection head used by ModernBERT."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        self.dense = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bool(config.get("classifier_bias", False)),
        )
        self.activation = config.get(
            "classifier_activation",
            config.get("hidden_act", "gelu"),
        )
        self.norm = _LayerNorm(
            hidden_size,
            eps=config.get("layer_norm_eps", 1e-5),
            bias=config.get("norm_bias", False),
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.norm(_activation(self.dense(hidden_states), self.activation))


class ModernBertForTokenClassification(nn.Module):
    """ModernBERT with a token-classification head.

    Output shape is ``(batch, sequence_length, num_labels)``.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = nn.Dropout(p=float(config.get("classifier_dropout", 0.0)))
        self.classifier = nn.Linear(config["hidden_size"], config["num_labels"])
        self.config = config

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        del token_type_ids
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = self.drop(self.head(hidden_states))
        return self.classifier(hidden_states)
