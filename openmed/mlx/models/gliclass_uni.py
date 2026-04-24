"""MLX runtime for uni-encoder GLiClass checkpoints."""

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


class GLiClassFeaturesProjector(nn.Module):
    """Feature projector matching the official GLiClass layout."""

    def __init__(self, encoder_hidden_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(encoder_hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_size, encoder_hidden_size)

    def __call__(self, features: mx.array) -> mx.array:
        hidden = self.linear1(features)
        hidden = nn.gelu(hidden)
        hidden = self.dropout(hidden)
        return self.linear2(hidden)


class GLiClassMLPScorer(nn.Module):
    """MLP scorer matching GLiClass uni-encoder checkpoints."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size * 2, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)

    def __call__(self, text_rep: mx.array, label_rep: mx.array) -> mx.array:
        batch_size, num_labels, dim = label_rep.shape
        expanded_text = mx.broadcast_to(text_rep[:, None, :], (batch_size, num_labels, dim))
        combined = mx.concatenate([expanded_text, label_rep], axis=-1)
        hidden = nn.relu(self.linear1(combined))
        hidden = nn.relu(self.linear2(hidden))
        return self.linear3(hidden).squeeze(-1)


class GLiClassUniEncoderModel(nn.Module):
    """Uni-encoder GLiClass model backed by DeBERTa-v3."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.deberta = DebertaV2Model(config)
        self.classes_projector = GLiClassFeaturesProjector(
            config["encoder_hidden_size"],
            config["hidden_size"],
            config.get("dropout", 0.0),
        )
        self.text_projector = GLiClassFeaturesProjector(
            config["encoder_hidden_size"],
            config["hidden_size"],
            config.get("dropout", 0.0),
        )
        self.segment_embeddings = nn.Embedding(3, config["encoder_hidden_size"])
        self.dropout = nn.Dropout(p=config.get("dropout", 0.0))
        self.scorer = GLiClassMLPScorer(config["encoder_hidden_size"])
        self.logit_scale = mx.array(
            config.get("logit_scale_init_value", 1.0),
            dtype=mx.float32,
        )

    def _create_segment_ids(self, input_ids: mx.array) -> mx.array:
        batch_size, seq_length = input_ids.shape
        segment_rows = []
        for batch_index in range(batch_size):
            row = [0] * seq_length
            tokens = input_ids[batch_index].tolist()
            text_positions = [idx for idx, token_id in enumerate(tokens) if token_id == self.config["text_token_index"]]
            if text_positions:
                text_start = text_positions[0]
                example_positions = [
                    idx for idx, token_id in enumerate(tokens)
                    if token_id == self.config["example_token_index"]
                ]
                if example_positions:
                    example_start = example_positions[0]
                    for index in range(text_start, example_start):
                        row[index] = 1
                    for index in range(example_start, seq_length):
                        row[index] = 2
                else:
                    for index in range(text_start, seq_length):
                        row[index] = 1
            segment_rows.append(row)
        return mx.array(segment_rows, dtype=mx.int32)

    def _pad_rows(self, rows: list[mx.array], width: int) -> mx.array:
        if not rows:
            return mx.zeros((0, width, self.config["encoder_hidden_size"]), dtype=mx.float32)

        padded: list[mx.array] = []
        for row in rows:
            pad_len = width - row.shape[0]
            if pad_len > 0:
                padding = mx.zeros((pad_len, row.shape[1]), dtype=row.dtype)
                row = mx.concatenate([row, padding], axis=0)
            padded.append(row)
        return mx.stack(padded, axis=0)

    def _pad_mask_rows(self, rows: list[list[int]], width: int) -> mx.array:
        return mx.array([row + [0] * (width - len(row)) for row in rows], dtype=mx.bool_)

    def _extract_class_features(
        self,
        encoder_layer: mx.array,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        batch_size, seq_length, embed_dim = encoder_layer.shape
        class_rows: list[mx.array] = []
        class_mask_rows: list[list[int]] = []
        max_classes = 0

        for batch_index in range(batch_size):
            token_ids = input_ids[batch_index].tolist()
            class_positions = [
                idx for idx, token_id in enumerate(token_ids)
                if token_id == self.config["class_token_index"]
            ]
            text_positions = [
                idx for idx, token_id in enumerate(token_ids)
                if token_id == self.config["text_token_index"]
            ]
            text_start = text_positions[0] if text_positions else seq_length
            row_embeddings: list[mx.array] = []

            for class_index, class_pos in enumerate(class_positions):
                start_pos = class_pos if self.config.get("embed_class_token", True) else min(class_pos + 1, seq_length - 1)
                end_pos = (
                    class_positions[class_index + 1]
                    if class_index + 1 < len(class_positions)
                    else text_start
                )
                if start_pos >= end_pos:
                    row_embeddings.append(encoder_layer[batch_index][start_pos])
                    continue

                class_tokens = encoder_layer[batch_index][start_pos:end_pos]
                class_attention = attention_mask[batch_index][start_pos:end_pos].astype(class_tokens.dtype)
                denom = mx.sum(class_attention)
                if float(denom.item()) > 0:
                    pooled = mx.sum(class_tokens * class_attention[:, None], axis=0) / denom
                else:
                    pooled = mx.mean(class_tokens, axis=0)
                row_embeddings.append(pooled)

            if row_embeddings:
                row = mx.stack(row_embeddings, axis=0)
            else:
                row = mx.zeros((0, embed_dim), dtype=encoder_layer.dtype)

            class_rows.append(row)
            max_classes = max(max_classes, row.shape[0])
            class_mask_rows.append([1] * row.shape[0])

        max_classes = max(max_classes, 1)
        classes_embedding = self._pad_rows(class_rows, max_classes)
        classes_embedding_mask = self._pad_mask_rows(class_mask_rows, max_classes)

        if self.config.get("extract_text_features", False):
            raise NotImplementedError(
                "GLiClass MLX support currently assumes extract_text_features=False."
            )

        return classes_embedding, classes_embedding_mask, encoder_layer, attention_mask

    def _pool_text(self, text_embeddings: mx.array, text_mask: mx.array) -> mx.array:
        pooling_strategy = self.config.get("pooling_strategy", "first")
        if pooling_strategy == "first":
            return text_embeddings[:, 0, :]

        if pooling_strategy == "mean":
            mask = text_mask.astype(text_embeddings.dtype)
            denom = mx.maximum(mx.sum(mask, axis=1, keepdims=True), 1.0)
            return mx.sum(text_embeddings * mask[:, :, None], axis=1) / denom

        raise ValueError(f"Unsupported GLiClass pooling strategy: {pooling_strategy!r}")

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> dict[str, mx.array]:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.float32)

        embedded = self.deberta.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if self.config.get("use_segment_embeddings", False):
            segment_ids = self._create_segment_ids(input_ids)
            embedded = embedded + self.segment_embeddings(segment_ids)

        hidden_states = self.deberta.encoder(embedded, attention_mask)

        classes_embedding, classes_mask, text_embeddings, text_mask = self._extract_class_features(
            hidden_states,
            input_ids,
            attention_mask,
        )
        pooled_output = self._pool_text(text_embeddings, text_mask)
        pooled_output = self.text_projector(pooled_output)
        pooled_output = self.dropout(pooled_output)

        classes_embedding = self.classes_projector(classes_embedding)

        if self.config.get("normalize_features", False):
            pooled_norm = mx.maximum(
                mx.linalg.norm(pooled_output, axis=-1, keepdims=True),
                1e-8,
            )
            class_norm = mx.maximum(
                mx.linalg.norm(classes_embedding, axis=-1, keepdims=True),
                1e-8,
            )
            pooled_output = pooled_output / pooled_norm
            classes_embedding = classes_embedding / class_norm

        logits = self.scorer(pooled_output, classes_embedding)
        if self.config.get("normalize_features", False):
            logits = logits * self.logit_scale

        return {
            "logits": logits,
            "classes_mask": classes_mask,
            "class_embeddings": classes_embedding,
            "text_embeddings": pooled_output,
        }
