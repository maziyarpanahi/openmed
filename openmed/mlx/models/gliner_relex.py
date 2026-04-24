"""MLX runtime for GLiNER relation-extraction checkpoints."""

from __future__ import annotations

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError(
        "MLX is required for this module. "
        "Install with: pip install openmed[mlx]"
    )

from openmed.mlx.models.deberta_v2_tc import DebertaV2Model
from openmed.mlx.models.gliner_common import (
    BidirectionalLSTM,
    ProjectionMLP,
    build_all_entity_pairs,
    extract_marker_embeddings,
    extract_word_embeddings,
)
from openmed.mlx.models.gliner_span import SpanMarkerV0


class GLiNERTokenScorer(nn.Module):
    """Token-label scorer for GLiNER token-level checkpoints."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.proj_token = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_label = nn.Linear(hidden_size, hidden_size * 2)
        self.out_linear1 = nn.Linear(hidden_size * 3, hidden_size * 4)
        self.dropout = nn.Dropout(p=dropout)
        self.out_linear2 = nn.Linear(hidden_size * 4, 3)

    def __call__(self, token_rep: mx.array, label_rep: mx.array) -> mx.array:
        batch_size, seq_len, hidden_size = token_rep.shape
        num_classes = label_rep.shape[1]

        token_proj = self.proj_token(token_rep).reshape(batch_size, seq_len, 1, 2, hidden_size)
        label_proj = self.proj_label(label_rep).reshape(batch_size, 1, num_classes, 2, hidden_size)

        token_left = mx.broadcast_to(token_proj[:, :, :, 0, :], (batch_size, seq_len, num_classes, hidden_size))
        token_right = mx.broadcast_to(token_proj[:, :, :, 1, :], (batch_size, seq_len, num_classes, hidden_size))
        label_left = mx.broadcast_to(label_proj[:, :, :, 0, :], (batch_size, seq_len, num_classes, hidden_size))
        label_right = mx.broadcast_to(label_proj[:, :, :, 1, :], (batch_size, seq_len, num_classes, hidden_size))

        combined = mx.concatenate([token_left, label_left, token_right * label_right], axis=-1)
        hidden = self.out_linear1(combined)
        hidden = self.dropout(hidden)
        hidden = nn.relu(hidden)
        return self.out_linear2(hidden)


class GLiNERRelexModel(nn.Module):
    """Token-level GLiNER relation-extraction model."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        encoder_hidden_size = config.get("encoder_hidden_size", config["hidden_size"])
        hidden_size = config["hidden_size"]
        backbone_config = dict(config)
        backbone_config["hidden_size"] = encoder_hidden_size
        self.deberta = DebertaV2Model(backbone_config)
        if encoder_hidden_size != hidden_size:
            self.token_projection = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.token_projection = None
        self.rnn = BidirectionalLSTM(hidden_size)
        self.scorer = GLiNERTokenScorer(hidden_size, config.get("dropout", 0.0))
        self.span_rep_layer = SpanMarkerV0(hidden_size, config.get("dropout", 0.0))
        self.prompt_rep_layer = ProjectionMLP(
            hidden_size,
            hidden_size,
            dropout=config.get("dropout", 0.0),
        )
        self.pair_rep_layer = ProjectionMLP(
            hidden_size * 2,
            hidden_size,
            dropout=config.get("dropout", 0.0),
        )

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        words_mask: mx.array | None = None,
    ) -> dict[str, mx.array]:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.float32)
        if words_mask is None:
            raise ValueError("GLiNER relation-extraction models require words_mask.")

        hidden_states = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        entity_prompts, entity_prompt_mask = extract_marker_embeddings(
            hidden_states,
            input_ids,
            self.config["class_token_index"],
            include_marker_token=self.config.get("embed_ent_token", True),
        )
        relation_prompts, relation_prompt_mask = extract_marker_embeddings(
            hidden_states,
            input_ids,
            self.config["rel_token_index"],
            include_marker_token=self.config.get(
                "embed_rel_token",
                self.config.get("embed_ent_token", True),
            ),
        )
        words_embedding, word_mask = extract_word_embeddings(hidden_states, words_mask)

        if self.token_projection is not None:
            entity_prompts = self.token_projection(entity_prompts)
            relation_prompts = self.token_projection(relation_prompts)
            words_embedding = self.token_projection(words_embedding)

        if self.config.get("num_rnn_layers", 0) > 0:
            words_embedding = self.rnn(words_embedding, word_mask)

        return {
            "entity_prompts": entity_prompts,
            "entity_prompt_mask": entity_prompt_mask,
            "relation_prompts": relation_prompts,
            "relation_prompt_mask": relation_prompt_mask,
            "words_embedding": words_embedding,
            "word_mask": word_mask,
        }

    def entity_scores(self, encoded: dict[str, mx.array]) -> mx.array:
        entity_prompts = self.prompt_rep_layer(encoded["entity_prompts"])
        return self.scorer(encoded["words_embedding"], entity_prompts)

    def relation_scores(
        self,
        encoded: dict[str, mx.array],
        span_idx: mx.array,
        span_mask: mx.array,
    ) -> dict[str, mx.array]:
        span_idx = span_idx * span_mask[:, :, None].astype(span_idx.dtype)
        span_rep = self.span_rep_layer(encoded["words_embedding"], span_idx)
        pair_idx, pair_mask, head_rep, tail_rep = build_all_entity_pairs(span_rep, span_mask)
        pair_rep = mx.concatenate([head_rep, tail_rep], axis=-1)
        pair_rep = self.pair_rep_layer(pair_rep)
        pair_scores = mx.einsum("bnd,bcd->bnc", pair_rep, encoded["relation_prompts"])
        return {
            "pair_scores": pair_scores,
            "pair_idx": pair_idx,
            "pair_mask": pair_mask,
            "span_rep": span_rep,
            "relation_prompt_mask": encoded["relation_prompt_mask"],
        }

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        words_mask: mx.array | None = None,
    ) -> dict[str, mx.array]:
        encoded = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_mask=words_mask,
        )
        return {
            "entity_scores": self.entity_scores(encoded),
            "entity_prompt_mask": encoded["entity_prompt_mask"],
            "word_mask": encoded["word_mask"],
        }
