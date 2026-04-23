"""MLX runtime for uni-encoder span-based GLiNER checkpoints."""

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
    extract_marker_embeddings,
    extract_word_embeddings,
    gather_span_endpoints,
)


class SpanMarkerV0(nn.Module):
    """SpanMarkerV0 head matching official GLiNER checkpoints."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.project_start = ProjectionMLP(hidden_size, hidden_size, dropout=dropout)
        self.project_end = ProjectionMLP(hidden_size, hidden_size, dropout=dropout)
        self.out_project = ProjectionMLP(
            hidden_size * 2,
            hidden_size,
            dropout=dropout,
        )

    def __call__(self, hidden_states: mx.array, span_idx: mx.array) -> mx.array:
        start_rep = self.project_start(hidden_states)
        end_rep = self.project_end(hidden_states)
        start_span_rep, end_span_rep = gather_span_endpoints(
            start_rep,
            end_rep,
            span_idx,
        )
        combined = mx.concatenate([start_span_rep, end_span_rep], axis=-1)
        combined = nn.relu(combined)
        return self.out_project(combined)


class GLiNERSpanModel(nn.Module):
    """GLiNER uni-encoder span model with DeBERTa backbone."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        encoder_hidden_size = config.get("encoder_hidden_size", config["hidden_size"])
        hidden_size = config["hidden_size"]
        backbone_config = dict(config)
        backbone_config["hidden_size"] = encoder_hidden_size
        self.deberta = DebertaV2Model(backbone_config)
        self.token_projection = nn.Linear(encoder_hidden_size, hidden_size)
        self.rnn = BidirectionalLSTM(hidden_size)
        self.span_rep_layer = SpanMarkerV0(hidden_size, config.get("dropout", 0.0))
        self.prompt_rep_layer = ProjectionMLP(
            hidden_size,
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
            raise ValueError("GLiNER span models require words_mask.")

        hidden_states = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeddings, prompt_mask = extract_marker_embeddings(
            hidden_states,
            input_ids,
            self.config["class_token_index"],
            include_marker_token=self.config.get("embed_ent_token", True),
        )
        words_embedding, word_mask = extract_word_embeddings(hidden_states, words_mask)

        words_embedding = self.token_projection(words_embedding)
        prompt_embeddings = self.token_projection(prompt_embeddings)

        if self.config.get("num_rnn_layers", 0) > 0:
            words_embedding = self.rnn(words_embedding, word_mask)

        return {
            "prompt_embeddings": prompt_embeddings,
            "prompt_mask": prompt_mask,
            "words_embedding": words_embedding,
            "word_mask": word_mask,
        }

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        words_mask: mx.array | None = None,
        span_idx: mx.array | None = None,
        span_mask: mx.array | None = None,
    ) -> dict[str, mx.array]:
        if span_idx is None:
            raise ValueError("GLiNER span models require span_idx.")

        if span_mask is not None:
            span_idx = span_idx * span_mask[:, :, None].astype(span_idx.dtype)

        encoded = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_mask=words_mask,
        )
        span_rep = self.span_rep_layer(encoded["words_embedding"], span_idx)
        prompt_rep = self.prompt_rep_layer(encoded["prompt_embeddings"])
        logits = mx.einsum("bsd,bcd->bsc", span_rep, prompt_rep)
        if span_mask is not None:
            logits = mx.where(
                span_mask[:, :, None],
                logits,
                mx.full(logits.shape, -1e9, dtype=logits.dtype),
            )
        return {
            "logits": logits,
            "span_rep": span_rep,
            "span_idx": span_idx,
            "span_mask": span_mask if span_mask is not None else mx.ones(logits.shape[:2], dtype=mx.bool_),
            "prompt_mask": encoded["prompt_mask"],
            "word_mask": encoded["word_mask"],
        }
