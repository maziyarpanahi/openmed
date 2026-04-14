"""DeBERTa-v2 for token classification implemented in Apple MLX.

This mirrors the converted weight layout produced by :mod:`openmed.mlx.convert`
for DeBERTa-v2 checkpoints:

- ``deberta.embeddings.*``
- ``deberta.encoder.layer.{i}.attention.self.*``
- ``deberta.encoder.layer.{i}.attention.out_proj.*``
- ``deberta.encoder.layer.{i}.ln1.*``
- ``deberta.encoder.layer.{i}.linear1.*``
- ``deberta.encoder.layer.{i}.linear2.*``
- ``deberta.encoder.layer.{i}.ln2.*``
- ``deberta.encoder.rel_embeddings.weight``
- ``deberta.encoder.LayerNorm.*``
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError(
        "MLX is required for this module. "
        "Install with: pip install openmed[mlx]"
    )


def make_log_bucket_position(
    relative_pos: mx.array,
    bucket_size: int,
    max_position: int,
) -> mx.array:
    """Bucketize relative positions using DeBERTa's logarithmic scheme."""
    sign = mx.sign(relative_pos)
    mid = bucket_size // 2
    dtype = relative_pos.dtype

    abs_pos = mx.where(
        (relative_pos < mid) & (relative_pos > -mid),
        mx.array(mid - 1, dtype=dtype),
        mx.abs(relative_pos),
    )

    abs_pos_f = abs_pos.astype(mx.float32)
    mid_f = mx.array(float(mid), dtype=mx.float32)
    log_base = mx.log(mx.array((max_position - 1) / mid, dtype=mx.float32))
    log_pos = mx.ceil(mx.log(abs_pos_f / mid_f) / log_base * (mid - 1)) + mid
    bucket_pos = mx.where(
        abs_pos <= mid,
        relative_pos.astype(log_pos.dtype),
        log_pos * sign.astype(log_pos.dtype),
    )
    return bucket_pos.astype(dtype)


def build_relative_position(
    query_layer: mx.array,
    key_layer: mx.array,
    bucket_size: int = -1,
    max_position: int = -1,
) -> mx.array:
    """Build the query-to-key relative position matrix."""
    query_size = query_layer.shape[-2]
    key_size = key_layer.shape[-2]

    q_ids = mx.arange(query_size, dtype=mx.int32)
    k_ids = mx.arange(key_size, dtype=mx.int32)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]

    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)

    return rel_pos_ids.astype(mx.int32)[None, :, :]


def build_rpos(
    query_layer: mx.array,
    key_layer: mx.array,
    relative_pos: mx.array,
    position_buckets: int,
    max_relative_positions: int,
) -> mx.array:
    """Build relative positions for position-to-content attention."""
    if key_layer.shape[-2] != query_layer.shape[-2]:
        return build_relative_position(
            key_layer,
            key_layer,
            bucket_size=position_buckets,
            max_position=max_relative_positions,
        )
    return relative_pos


def _repeat_batches(x: mx.array, batch_size: int) -> mx.array:
    """Repeat a [heads, length, dim] tensor across batches."""
    return mx.broadcast_to(x[None, ...], (batch_size,) + x.shape).reshape(
        batch_size * x.shape[0], x.shape[1], x.shape[2]
    )


class DebertaV2Embeddings(nn.Module):
    """Word/token embeddings plus optional positional/type embeddings."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embedding_size = config.get("embedding_size", config["hidden_size"])
        self.position_biased_input = config.get("position_biased_input", True)

        self.word_embeddings = nn.Embedding(config["vocab_size"], self.embedding_size)

        if self.position_biased_input:
            self.position_embeddings = nn.Embedding(
                config["max_position_embeddings"], self.embedding_size,
            )
        else:
            self.position_embeddings = None

        if config.get("type_vocab_size", 0) > 0:
            self.token_type_embeddings = nn.Embedding(
                config["type_vocab_size"], self.embedding_size,
            )
        else:
            self.token_type_embeddings = None

        if self.embedding_size != config["hidden_size"]:
            self.embed_proj = nn.Linear(self.embedding_size, config["hidden_size"], bias=False)
        else:
            self.embed_proj = None

        self.LayerNorm = nn.LayerNorm(
            config["hidden_size"], eps=config.get("layer_norm_eps", 1e-7),
        )
        self.dropout = nn.Dropout(p=config.get("hidden_dropout_prob", 0.1))

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        seq_len = input_ids.shape[1]
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds

        if self.position_embeddings is not None:
            position_ids = mx.arange(seq_len, dtype=input_ids.dtype)[None, :]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = mx.zeros_like(input_ids)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        if self.embed_proj is not None:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if attention_mask is not None:
            mask = attention_mask
            if mask.ndim == 4:
                mask = mask[:, 0, 0, :]
            if mask.ndim == 2:
                mask = mask[:, :, None]
            embeddings = embeddings * mask.astype(embeddings.dtype)

        return self.dropout(embeddings)


class DisentangledSelfAttention(nn.Module):
    """DeBERTa-v2 disentangled self-attention with relative position bias."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_attention_heads = config["num_attention_heads"]
        default_head_size = config["hidden_size"] // self.num_attention_heads
        self.attention_head_size = config.get("attention_head_size", default_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_proj = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key_proj = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value_proj = nn.Linear(config["hidden_size"], self.all_head_size)

        self.share_att_key = config.get("share_att_key", False)
        self.pos_att_type = config.get("pos_att_type") or []
        self.relative_attention = config.get("relative_attention", False)

        if self.relative_attention:
            self.position_buckets = config.get("position_buckets", -1)
            self.max_relative_positions = config.get("max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config["max_position_embeddings"]
            self.pos_ebd_size = (
                self.position_buckets if self.position_buckets > 0 else self.max_relative_positions
            )
            self.pos_dropout = nn.Dropout(p=config.get("hidden_dropout_prob", 0.1))

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config["hidden_size"], self.all_head_size)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config["hidden_size"], self.all_head_size)

        self.dropout = nn.Dropout(p=config.get("attention_probs_dropout_prob", 0.1))

    def transpose_for_scores(self, x: mx.array) -> mx.array:
        """Reshape [B, L, H] projections to [B*heads, L, head_dim]."""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size * self.num_attention_heads, seq_len, self.attention_head_size)

    def disentangled_attention_bias(
        self,
        query_layer: mx.array,
        key_layer: mx.array,
        relative_pos: Optional[mx.array],
        rel_embeddings: mx.array,
        scale_factor: int,
    ) -> mx.array:
        """Compute DeBERTa's content-to-position and position-to-content bias."""
        if relative_pos is None:
            relative_pos = build_relative_position(
                query_layer,
                key_layer,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )

        if relative_pos.ndim == 2:
            relative_pos = relative_pos[None, None, :, :]
        elif relative_pos.ndim == 3:
            relative_pos = relative_pos[:, None, :, :]
        elif relative_pos.ndim != 4:
            raise ValueError(
                f"Relative position ids must be of dim 2, 3, or 4. Got {relative_pos.ndim}."
            )

        relative_pos = relative_pos.astype(mx.int32)
        att_span = self.pos_ebd_size
        rel_embeddings = rel_embeddings[: att_span * 2, :][None, :, :]

        batch_size = query_layer.shape[0] // self.num_attention_heads
        if self.share_att_key:
            pos_query_layer = _repeat_batches(
                self.transpose_for_scores(self.query_proj(rel_embeddings)), batch_size,
            )
            pos_key_layer = _repeat_batches(
                self.transpose_for_scores(self.key_proj(rel_embeddings)), batch_size,
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = _repeat_batches(
                    self.transpose_for_scores(self.pos_key_proj(rel_embeddings)), batch_size,
                )
            if "p2c" in self.pos_att_type:
                pos_query_layer = _repeat_batches(
                    self.transpose_for_scores(self.pos_query_proj(rel_embeddings)), batch_size,
                )

        score = None

        if "c2p" in self.pos_att_type:
            c2p_att = query_layer @ pos_key_layer.transpose(0, 2, 1)
            c2p_pos = mx.clip(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_index = mx.broadcast_to(
                mx.squeeze(c2p_pos, axis=0),
                (query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1]),
            ).astype(mx.int32)
            c2p_score = mx.take_along_axis(c2p_att, c2p_index, axis=-1)
            c2p_score = c2p_score / math.sqrt(pos_key_layer.shape[-1] * scale_factor)
            score = c2p_score if score is None else score + c2p_score

        if "p2c" in self.pos_att_type:
            r_pos = build_rpos(
                query_layer,
                key_layer,
                relative_pos,
                self.position_buckets,
                self.max_relative_positions,
            )
            p2c_pos = mx.clip(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = key_layer @ pos_query_layer.transpose(0, 2, 1)
            p2c_index = mx.broadcast_to(
                mx.squeeze(p2c_pos, axis=0),
                (query_layer.shape[0], key_layer.shape[-2], key_layer.shape[-2]),
            ).astype(mx.int32)
            p2c_score = mx.take_along_axis(p2c_att, p2c_index, axis=-1).transpose(0, 2, 1)
            p2c_score = p2c_score / math.sqrt(pos_query_layer.shape[-1] * scale_factor)
            score = p2c_score if score is None else score + p2c_score

        if score is None:
            return mx.zeros((query_layer.shape[0], query_layer.shape[1], key_layer.shape[1]))
        return score

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        query_states: Optional[mx.array] = None,
        relative_pos: Optional[mx.array] = None,
        rel_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if query_states is None:
            query_states = hidden_states

        query_layer = self.transpose_for_scores(self.query_proj(query_states))
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states))

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1

        scale = math.sqrt(query_layer.shape[-1] * scale_factor)
        attention_scores = query_layer @ (key_layer.transpose(0, 2, 1) / scale)

        if self.relative_attention and rel_embeddings is not None:
            rel_att = self.disentangled_attention_bias(
                query_layer,
                key_layer,
                relative_pos,
                self.pos_dropout(rel_embeddings),
                scale_factor,
            )
            attention_scores = attention_scores + rel_att

        batch_size = hidden_states.shape[0]
        query_len = query_layer.shape[1]
        key_len = key_layer.shape[1]
        attention_scores = attention_scores.reshape(
            batch_size, self.num_attention_heads, query_len, key_len,
        )

        mask = attention_mask > 0
        min_value = mx.array(mx.finfo(attention_scores.dtype).min, dtype=attention_scores.dtype)
        attention_scores = mx.where(mask, attention_scores, min_value)

        attention_probs = self.dropout(mx.softmax(attention_scores, axis=-1))
        context_layer = attention_probs.reshape(
            batch_size * self.num_attention_heads, query_len, key_len,
        ) @ value_layer
        context_layer = context_layer.reshape(
            batch_size, self.num_attention_heads, query_len, self.attention_head_size,
        )
        context_layer = context_layer.transpose(0, 2, 1, 3)
        return context_layer.reshape(batch_size, query_len, self.all_head_size)


class DebertaV2Attention(nn.Module):
    """Attention block matching the converted DeBERTa weight layout."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.out_proj = nn.Linear(config["hidden_size"], config["hidden_size"])

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        query_states: Optional[mx.array] = None,
        relative_pos: Optional[mx.array] = None,
        rel_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        context = self.self(
            hidden_states,
            attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        return self.out_proj(context)


class DebertaV2Layer(nn.Module):
    """Single DeBERTa-v2 encoder layer."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        eps = config.get("layer_norm_eps", 1e-7)
        self.attention = DebertaV2Attention(config)
        self.ln1 = nn.LayerNorm(config["hidden_size"], eps=eps)
        self.linear1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.linear2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.ln2 = nn.LayerNorm(config["hidden_size"], eps=eps)
        self.dropout = nn.Dropout(p=config.get("hidden_dropout_prob", 0.1))
        self.hidden_act = config.get("hidden_act", "gelu")

    def _apply_activation(self, x: mx.array) -> mx.array:
        if self.hidden_act == "gelu":
            return nn.gelu(x)
        raise ValueError(f"Unsupported DeBERTa activation: {self.hidden_act!r}")

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        query_states: Optional[mx.array] = None,
        relative_pos: Optional[mx.array] = None,
        rel_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states if query_states is None else query_states

        attn_out = self.attention(
            hidden_states,
            attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        hidden_states = self.ln1(residual + self.dropout(attn_out))

        ff_out = self.linear2(self._apply_activation(self.linear1(hidden_states)))
        return self.ln2(hidden_states + self.dropout(ff_out))


class DebertaV2Encoder(nn.Module):
    """Stack of DeBERTa-v2 encoder layers with relative attention."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.layer = [DebertaV2Layer(config) for _ in range(config["num_hidden_layers"])]
        self.relative_attention = config.get("relative_attention", False)
        self.position_buckets = config.get("position_buckets", -1)
        self.max_relative_positions = config.get("max_relative_positions", -1)
        if self.max_relative_positions < 1:
            self.max_relative_positions = config["max_position_embeddings"]

        if self.relative_attention:
            pos_ebd_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config["hidden_size"])

        self.norm_rel_ebd = [
            value.strip()
            for value in config.get("norm_rel_ebd", "none").lower().split("|")
        ]
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(
                config["hidden_size"], eps=config.get("layer_norm_eps", 1e-7),
            )

    def get_rel_embedding(self) -> Optional[mx.array]:
        """Return relative embeddings, normalized when configured."""
        if not self.relative_attention:
            return None

        rel_embeddings = self.rel_embeddings.weight
        if "layer_norm" in self.norm_rel_ebd:
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask: mx.array) -> mx.array:
        """Expand a [B, L] mask to DeBERTa's pairwise [B, 1, L, L] mask."""
        if attention_mask.ndim <= 2:
            return attention_mask[:, None, :, None] * attention_mask[:, None, None, :]
        if attention_mask.ndim == 3:
            return attention_mask[:, None, :, :]
        return attention_mask

    def get_rel_pos(
        self,
        hidden_states: mx.array,
        query_states: Optional[mx.array] = None,
        relative_pos: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Build relative positions on demand."""
        if self.relative_attention and relative_pos is None:
            if query_states is not None:
                return build_relative_position(
                    query_states,
                    hidden_states,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
            return build_relative_position(
                hidden_states,
                hidden_states,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        return relative_pos

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states)
        rel_embeddings = self.get_rel_embedding()

        for layer in self.layer:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )
        return hidden_states


class DebertaV2Model(nn.Module):
    """Backbone DeBERTa-v2 encoder."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.float32)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return self.encoder(embedding_output, attention_mask)


class DebertaV2ForTokenClassification(nn.Module):
    """DeBERTa-v2 with a linear token-classification head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(p=config.get("hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(config["hidden_size"], config["num_labels"])
        self.config = config

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = self.dropout(hidden_states)
        return self.classifier(hidden_states)
