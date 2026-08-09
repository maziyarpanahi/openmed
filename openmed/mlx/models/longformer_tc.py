"""Longformer token classification implemented in Apple MLX.

The local attention path is expressed as a vectorized mask so it works for
arbitrary sequence lengths. Global tokens use Longformer's separate query,
key, and value projections and attend to every non-padding token.
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


def _attention_windows(config: dict) -> list[int]:
    """Return one validated even attention window per encoder layer."""
    num_layers = int(config["num_hidden_layers"])
    configured = config.get("attention_window", 512)
    if isinstance(configured, (list, tuple)):
        windows = [int(value) for value in configured]
        if len(windows) != num_layers:
            raise ValueError(
                "Longformer attention_window must have one value per layer"
            )
    else:
        windows = [int(configured)] * num_layers
    if any(window <= 0 or window % 2 for window in windows):
        raise ValueError("Longformer attention_window values must be positive and even")
    return windows


class LongformerEmbeddings(nn.Module):
    """Longformer token, position, and segment embeddings."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.padding_idx = int(config.get("pad_token_id", 0) or 0)
        self.word_embeddings = nn.Embedding(int(config["vocab_size"]), hidden_size)
        self.position_embeddings = nn.Embedding(
            int(config["max_position_embeddings"]), hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            int(config.get("type_vocab_size", 2)), hidden_size
        )
        self.norm = nn.LayerNorm(
            hidden_size, eps=float(config.get("layer_norm_eps", 1e-12))
        )
        self.dropout = nn.Dropout(p=float(config.get("hidden_dropout_prob", 0.1)))

    def _position_ids(
        self,
        input_ids: Optional[mx.array],
        inputs_embeds: Optional[mx.array],
    ) -> mx.array:
        if input_ids is None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            positions = mx.arange(
                self.padding_idx + 1,
                self.padding_idx + seq_len + 1,
                dtype=mx.int32,
            )
            return mx.broadcast_to(positions[None, :], (batch_size, seq_len))

        non_padding = input_ids != self.padding_idx
        incremental = mx.cumsum(non_padding.astype(mx.int32), axis=1)
        return incremental * non_padding.astype(mx.int32) + self.padding_idx

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds is required")
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if position_ids is None:
            position_ids = self._position_ids(input_ids, inputs_embeds)
        if token_type_ids is None:
            token_type_ids = mx.zeros(position_ids.shape, dtype=mx.int32)
        hidden_states = (
            inputs_embeds
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.norm(hidden_states))


class LongformerAttention(nn.Module):
    """Local attention plus Longformer's separate global-token projections."""

    def __init__(self, config: dict, attention_window: int) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.head_dim = hidden_size // self.num_heads
        if hidden_size != self.num_heads * self.head_dim:
            raise ValueError("Longformer hidden size must divide evenly by heads")
        self.hidden_size = hidden_size
        self.window = attention_window // 2
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.query_global_proj = nn.Linear(hidden_size, hidden_size)
        self.key_global_proj = nn.Linear(hidden_size, hidden_size)
        self.value_global_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(
            p=float(config.get("attention_probs_dropout_prob", 0.1))
        )

    def _softmax_attention(
        self,
        scores: mx.array,
        allowed: mx.array,
        query_active: mx.array,
    ) -> mx.array:
        min_value = mx.array(mx.finfo(scores.dtype).min, dtype=scores.dtype)
        scores = mx.where(allowed, scores, min_value)
        probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
            scores.dtype
        )
        probabilities = mx.where(
            query_active[:, None, :, None], probabilities, mx.zeros_like(probabilities)
        )
        return self.dropout(probabilities)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        query = (
            self.query_proj(hidden_states)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        key = (
            self.key_proj(hidden_states)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        value = (
            self.value_proj(hidden_states)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        active = attention_mask > 0
        global_tokens = attention_mask > 1
        positions = mx.arange(seq_len, dtype=mx.int32)
        local = (mx.abs(positions[:, None] - positions[None, :]) <= self.window)[
            None, None, :, :
        ]
        allowed = local | global_tokens[:, None, None, :]
        allowed = allowed & active[:, None, None, :]
        scores = (query @ key.transpose(0, 1, 3, 2)) * (self.head_dim**-0.5)
        probabilities = self._softmax_attention(scores, allowed, active)
        local_output = probabilities @ value

        global_query = (
            self.query_global_proj(hidden_states)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        global_key = (
            self.key_global_proj(hidden_states)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        global_value = (
            self.value_global_proj(hidden_states)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        global_scores = (global_query @ global_key.transpose(0, 1, 3, 2)) * (
            self.head_dim**-0.5
        )
        global_allowed = active[:, None, None, :]
        global_probabilities = self._softmax_attention(
            global_scores, global_allowed, active
        )
        global_output = global_probabilities @ global_value
        output = mx.where(global_tokens[:, None, :, None], global_output, local_output)
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.hidden_size
        )
        return self.out_proj(output)


def _longformer_activation(x: mx.array, name: str) -> mx.array:
    """Apply the activation named by a Longformer config."""
    normalized = name.lower().replace("-", "_")
    if normalized == "gelu":
        return nn.gelu(x)
    if normalized in {"gelu_new", "gelu_fast"}:
        return nn.gelu_approx(x)
    if normalized in {"silu", "swish"}:
        return nn.silu(x)
    if normalized == "relu":
        return nn.relu(x)
    raise ValueError(f"Unsupported Longformer activation: {name!r}")


class LongformerLayer(nn.Module):
    """One Longformer encoder layer."""

    def __init__(self, config: dict, attention_window: int) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.attention = LongformerAttention(config, attention_window)
        self.ln1 = nn.LayerNorm(
            hidden_size, eps=float(config.get("layer_norm_eps", 1e-12))
        )
        self.linear1 = nn.Linear(hidden_size, int(config["intermediate_size"]))
        self.linear2 = nn.Linear(int(config["intermediate_size"]), hidden_size)
        self.ln2 = nn.LayerNorm(
            hidden_size, eps=float(config.get("layer_norm_eps", 1e-12))
        )
        self.attention_dropout = nn.Dropout(
            p=float(config.get("hidden_dropout_prob", 0.1))
        )
        self.output_dropout = nn.Dropout(
            p=float(config.get("hidden_dropout_prob", 0.1))
        )
        self.activation = str(config.get("hidden_act", "gelu"))

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.ln1(
            hidden_states + self.attention_dropout(attention_output)
        )
        intermediate = _longformer_activation(
            self.linear1(hidden_states), self.activation
        )
        output = self.linear2(intermediate)
        return self.ln2(hidden_states + self.output_dropout(output))


class LongformerEncoder(nn.Module):
    """Stack of Longformer layers."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        windows = _attention_windows(config)
        self.layers = [LongformerLayer(config, window) for window in windows]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class LongformerModel(nn.Module):
    """Longformer encoder without a task-specific head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.config = config

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        global_attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        if attention_mask is None:
            attention_mask = mx.ones(hidden_states.shape[:2], dtype=mx.float32)
        if global_attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        return self.encoder(hidden_states, attention_mask)


class LongformerForTokenClassification(nn.Module):
    """Longformer with a linear token-classification logits head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(p=float(config.get("hidden_dropout_prob", 0.1)))
        self.classifier = nn.Linear(
            int(config["hidden_size"]), int(config["num_labels"])
        )
        self.config = config

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        global_attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **_: object,
    ) -> mx.array:
        hidden_states = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return self.classifier(self.dropout(hidden_states))
