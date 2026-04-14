"""BERT for Token Classification implemented in Apple MLX.

This is a pure-MLX implementation of BERT with a classification head
on top, suitable for NER / token-classification tasks.  The architecture
follows the original Devlin et al. (2019) design and is weight-compatible
with HuggingFace ``BertForTokenClassification`` after key remapping via
:mod:`openmed.mlx.convert`.

Reference: ``ml-explore/mlx-examples/bert/model.py`` for the base
encoder architecture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError(
        "MLX is required for this module. "
        "Install with: pip install openmed[mlx]"
    )


class BertEmbeddings(nn.Module):
    """Token + position + segment embeddings with LayerNorm."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        layer_norm_eps = config.get("layer_norm_eps", 1e-12)
        type_vocab_size = int(config.get("type_vocab_size", 2) or 0)

        self.word_embeddings = nn.Embedding(
            config["vocab_size"], config["hidden_size"],
        )
        self.position_embeddings = nn.Embedding(
            config["max_position_embeddings"], config["hidden_size"],
        )
        self.token_type_embeddings = (
            nn.Embedding(type_vocab_size, config["hidden_size"])
            if type_vocab_size > 0
            else None
        )
        self.position_offset = int(config.get("_mlx_position_offset", 0))
        self.norm = nn.LayerNorm(config["hidden_size"], eps=layer_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        seq_len = input_ids.shape[1]
        position_ids = mx.arange(seq_len, dtype=input_ids.dtype) + self.position_offset

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = mx.zeros_like(input_ids)
            x = x + self.token_type_embeddings(token_type_ids)
        return self.norm(x)


class BertAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_heads = config["num_attention_heads"]
        self.hidden_size = config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads

        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.query_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.key_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.value_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        scores = (queries @ keys.transpose(0, 1, 3, 2)) / scale

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = mx.softmax(scores, axis=-1)
        out = (weights @ values).transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return self.out_proj(out)


class BertLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        layer_norm_eps = config.get("layer_norm_eps", 1e-12)
        self.attention = BertAttention(config)
        self.ln1 = nn.LayerNorm(config["hidden_size"], eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(config["hidden_size"], eps=layer_norm_eps)
        self.linear1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.linear2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        attn_out = self.attention(x, attention_mask)
        x = self.ln1(x + attn_out)
        ff_out = self.linear2(nn.gelu(self.linear1(x)))
        return self.ln2(x + ff_out)


class BertEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.layers = [BertLayer(config) for _ in range(config["num_hidden_layers"])]

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class BertForTokenClassification(nn.Module):
    """BERT with a linear token-classification head.

    Output shape: ``(batch, seq_len, num_labels)`` logits.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dropout = nn.Dropout(p=config.get("hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(
            config["hidden_size"], config["num_labels"],
        )
        self.config = config

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # Convert [B, L] mask to [B, 1, 1, L] additive mask
            mask = (1.0 - attention_mask[:, None, None, :]) * -1e9
        else:
            mask = None

        x = self.encoder(x, mask)
        x = self.dropout(x)
        return self.classifier(x)


def load_model(model_path: str | Path) -> BertForTokenClassification:
    """Load a converted MLX BERT-TC model from *model_path*.

    Expects the directory to contain ``config.json`` and MLX weights in
    ``weights.safetensors`` or ``weights.npz``.
    """
    model_path = Path(model_path)

    with open(model_path / "config.json") as f:
        config = json.load(f)

    model = BertForTokenClassification(config)

    preferred_format = config.get("_mlx_weights_format")
    weights_npz = model_path / "weights.npz"
    weights_sf = model_path / "weights.safetensors"
    candidate_paths = []
    if preferred_format == "safetensors":
        candidate_paths.append(weights_sf)
    elif preferred_format == "npz":
        candidate_paths.append(weights_npz)
    candidate_paths.extend([weights_sf, weights_npz])

    weights_path = next((path for path in candidate_paths if path.exists()), None)
    if weights_path is None:
        raise FileNotFoundError(
            f"No weights found in {model_path}. "
            "Expected weights.safetensors or weights.npz."
        )
    weights = dict(mx.load(str(weights_path)))

    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model
