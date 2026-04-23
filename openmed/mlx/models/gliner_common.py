"""Shared MLX helpers for GLiNER-style zero-shot models."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError(
        "MLX is required for this module. "
        "Install with: pip install openmed[mlx]"
    )


class ProjectionMLP(nn.Module):
    """Two-layer projection block matching GLiNER / GLiClass checkpoints."""

    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_size = output_size or input_size
        self.linear1 = nn.Linear(input_size, self.output_size * 4)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(self.output_size * 4, self.output_size)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.linear1(x)
        hidden = nn.relu(hidden)
        hidden = self.dropout(hidden)
        return self.linear2(hidden)


class BidirectionalLSTM(nn.Module):
    """Small bidirectional LSTM wrapper backed by MLX single-direction LSTMs."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        half_hidden = hidden_size // 2
        self.forward_lstm = nn.LSTM(hidden_size, half_hidden)
        self.backward_lstm = nn.LSTM(hidden_size, half_hidden)

    @staticmethod
    def _reverse_padded(x: mx.array, lengths: mx.array) -> mx.array:
        steps = mx.arange(x.shape[1], dtype=mx.int32)[None, :]
        gather_index = mx.where(
            steps < lengths[:, None],
            lengths[:, None] - 1 - steps,
            steps,
        )
        return mx.take_along_axis(x, gather_index[:, :, None], axis=1)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        lengths = mx.sum(mask.astype(mx.int32), axis=1)
        forward_out, _ = self.forward_lstm(x)

        reversed_x = self._reverse_padded(x, lengths)
        backward_reversed, _ = self.backward_lstm(reversed_x)
        backward_out = self._reverse_padded(backward_reversed, lengths)

        output = mx.concatenate([forward_out, backward_out], axis=-1)
        return output * mask[:, :, None].astype(output.dtype)


def _pad_embeddings(rows: list[mx.array], width: int) -> mx.array:
    if not rows:
        return mx.zeros((0, width, 0), dtype=mx.float32)

    embed_dim = rows[0].shape[-1]
    padded_rows: list[mx.array] = []
    for row in rows:
        pad_len = width - row.shape[0]
        if pad_len > 0:
            padding = mx.zeros((pad_len, embed_dim), dtype=row.dtype)
            row = mx.concatenate([row, padding], axis=0)
        padded_rows.append(row)
    return mx.stack(padded_rows, axis=0)


def _pad_masks(rows: list[list[int]], width: int) -> mx.array:
    padded = [row + [0] * (width - len(row)) for row in rows]
    return mx.array(padded, dtype=mx.bool_)


def extract_marker_embeddings(
    token_embeds: mx.array,
    input_ids: mx.array,
    marker_token_id: int,
    *,
    include_marker_token: bool = True,
) -> tuple[mx.array, mx.array]:
    """Extract prompt embeddings from repeated special marker tokens."""
    batch_size, seq_len, embed_dim = token_embeds.shape
    rows: list[mx.array] = []
    masks: list[list[int]] = []
    max_items = 0

    for batch_index in range(batch_size):
        row_ids = input_ids[batch_index].tolist()
        raw_positions = [idx for idx, token_id in enumerate(row_ids) if token_id == marker_token_id]
        if not raw_positions:
            rows.append(mx.zeros((0, embed_dim), dtype=token_embeds.dtype))
            masks.append([])
            continue

        positions = mx.array(raw_positions, dtype=mx.int32)
        if not include_marker_token:
            positions = mx.minimum(
                positions + 1,
                mx.array(seq_len - 1, dtype=positions.dtype),
            )
        row = token_embeds[batch_index][positions]
        rows.append(row)
        count = row.shape[0]
        max_items = max(max_items, count)
        masks.append([1] * count)

    max_items = max(max_items, 1)
    return _pad_embeddings(rows, max_items), _pad_masks(masks, max_items)


def extract_word_embeddings(
    token_embeds: mx.array,
    words_mask: mx.array,
) -> tuple[mx.array, mx.array]:
    """Gather first-subtoken embeddings into word-level representations."""
    batch_size, _, embed_dim = token_embeds.shape
    rows: list[mx.array] = []
    masks: list[list[int]] = []
    max_words = 0

    for batch_index in range(batch_size):
        word_mask = words_mask[batch_index].tolist()
        token_positions = [idx for idx, word_index in enumerate(word_mask) if word_index > 0]
        if not token_positions:
            rows.append(mx.zeros((0, embed_dim), dtype=token_embeds.dtype))
            masks.append([])
            continue

        positions = mx.array(token_positions, dtype=mx.int32)
        row = token_embeds[batch_index][positions]
        rows.append(row)
        count = row.shape[0]
        max_words = max(max_words, count)
        masks.append([1] * count)

    max_words = max(max_words, 1)
    return _pad_embeddings(rows, max_words), _pad_masks(masks, max_words)


def build_candidate_span_indices(
    lengths: list[int],
    max_width: int,
) -> tuple[mx.array, mx.array]:
    """Enumerate all candidate spans up to ``max_width`` for each example."""
    span_rows: list[list[list[int]]] = []
    mask_rows: list[list[int]] = []
    max_spans = 0

    for length in lengths:
        spans: list[list[int]] = []
        for start in range(max(length, 1)):
            for width in range(max_width):
                end = start + width
                spans.append([start, end])
        max_spans = max(max_spans, len(spans))
        span_rows.append(spans)
        mask_rows.append([1 if end < length else 0 for _, end in spans])

    max_spans = max(max_spans, 1)
    padded_spans = [
        spans + [[0, 0]] * (max_spans - len(spans))
        for spans in span_rows
    ]
    padded_masks = [
        masks + [0] * (max_spans - len(masks))
        for masks in mask_rows
    ]
    return (
        mx.array(padded_spans, dtype=mx.int32),
        mx.array(padded_masks, dtype=mx.bool_),
    )


def gather_span_endpoints(
    start_hidden_states: mx.array,
    end_hidden_states: mx.array,
    span_idx: mx.array,
) -> tuple[mx.array, mx.array]:
    """Gather start/end embeddings for batched spans."""
    hidden_size = start_hidden_states.shape[-1]
    start_idx = mx.broadcast_to(
        span_idx[:, :, 0][:, :, None],
        (*span_idx.shape[:2], hidden_size),
    )
    end_idx = mx.broadcast_to(
        span_idx[:, :, 1][:, :, None],
        (*span_idx.shape[:2], hidden_size),
    )
    start_rep = mx.take_along_axis(start_hidden_states, start_idx, axis=1)
    end_rep = mx.take_along_axis(end_hidden_states, end_idx, axis=1)
    return start_rep, end_rep


def build_all_entity_pairs(
    span_rep: mx.array,
    span_mask: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Build every directed entity pair for relation extraction."""
    batch_size, _, embed_dim = span_rep.shape
    pair_rows: list[list[list[int]]] = []
    pair_mask_rows: list[list[int]] = []
    head_rows: list[mx.array] = []
    tail_rows: list[mx.array] = []
    max_pairs = 0

    for batch_index in range(batch_size):
        entity_count = int(mx.sum(span_mask[batch_index].astype(mx.int32)).item())
        pairs = [
            [head, tail]
            for head in range(entity_count)
            for tail in range(entity_count)
            if head != tail
        ]
        max_pairs = max(max_pairs, len(pairs))
        pair_rows.append(pairs)
        pair_mask_rows.append([1] * len(pairs))

        if not pairs:
            head_rows.append(mx.zeros((0, embed_dim), dtype=span_rep.dtype))
            tail_rows.append(mx.zeros((0, embed_dim), dtype=span_rep.dtype))
            continue

        pair_array = mx.array(pairs, dtype=mx.int32)
        head_rows.append(span_rep[batch_index][pair_array[:, 0]])
        tail_rows.append(span_rep[batch_index][pair_array[:, 1]])

    max_pairs = max(max_pairs, 1)
    padded_pairs = [
        pairs + [[0, 0]] * (max_pairs - len(pairs))
        for pairs in pair_rows
    ]
    padded_pair_masks = [
        mask + [0] * (max_pairs - len(mask))
        for mask in pair_mask_rows
    ]

    return (
        mx.array(padded_pairs, dtype=mx.int32),
        mx.array(padded_pair_masks, dtype=mx.bool_),
        _pad_embeddings(head_rows, max_pairs),
        _pad_embeddings(tail_rows, max_pairs),
    )


@dataclass(slots=True)
class TokenLevelSpan:
    start: int
    end: int
    label_index: int
    score: float


@dataclass(slots=True)
class TokenLevelSpanResult:
    spans: list[TokenLevelSpan]

    @property
    def span_idx(self) -> list[tuple[int, int]]:
        return [(span.start, span.end) for span in self.spans]

    @property
    def span_mask(self) -> list[bool]:
        return [True] * len(self.spans)


def _has_flat_overlap(span: TokenLevelSpan, selected: list[TokenLevelSpan], multi_label: bool) -> bool:
    for existing in selected:
        if span.start == existing.start and span.end == existing.end:
            if not multi_label:
                return True
            continue
        if not (span.start > existing.end or existing.start > span.end):
            return True
    return False


def _has_nested_overlap(span: TokenLevelSpan, selected: list[TokenLevelSpan], multi_label: bool) -> bool:
    for existing in selected:
        if span.start == existing.start and span.end == existing.end:
            if not multi_label:
                return True
            continue
        is_disjoint = span.start > existing.end or existing.start > span.end
        is_nested = (
            (span.start <= existing.start and span.end >= existing.end)
            or (existing.start <= span.start and existing.end >= span.end)
        )
        if not (is_disjoint or is_nested):
            return True
    return False


def _greedy_token_spans(
    spans: list[TokenLevelSpan],
    *,
    flat_ner: bool,
    multi_label: bool,
) -> list[TokenLevelSpan]:
    selected: list[TokenLevelSpan] = []
    overlap_checker = _has_flat_overlap if flat_ner else _has_nested_overlap
    for span in sorted(spans, key=lambda item: -item.score):
        if not overlap_checker(span, selected, multi_label):
            selected.append(span)
    return sorted(selected, key=lambda item: (item.start, item.end, item.label_index))


def decode_token_level_spans(
    token_scores: list[list[list[list[float]]]],
    *,
    threshold: float,
    flat_ner: bool = True,
    multi_label: bool = False,
) -> list[TokenLevelSpanResult]:
    """Decode token-level [start, end, inside] scores into spans."""
    results: list[TokenLevelSpanResult] = []
    for sample_scores in token_scores:
        sequence_length = len(sample_scores)
        num_classes = len(sample_scores[0]) if sample_scores else 0
        sample_spans: list[TokenLevelSpan] = []

        for class_index in range(num_classes):
            starts: list[int] = []
            ends: list[int] = []
            inside_scores: list[float] = []
            for token_index in range(sequence_length):
                start_score, end_score, inside_score = sample_scores[token_index][class_index]
                if start_score > threshold:
                    starts.append(token_index)
                if end_score > threshold:
                    ends.append(token_index)
                inside_scores.append(float(inside_score))

            for start in starts:
                for end in ends:
                    if end < start:
                        continue
                    span_inside = inside_scores[start : end + 1]
                    if any(score < threshold for score in span_inside):
                        continue
                    start_score = float(sample_scores[start][class_index][0])
                    end_score = float(sample_scores[end][class_index][1])
                    score = min([start_score, end_score] + span_inside)
                    sample_spans.append(
                        TokenLevelSpan(
                            start=start,
                            end=end,
                            label_index=class_index,
                            score=score,
                        )
                    )

        results.append(
            TokenLevelSpanResult(
                spans=_greedy_token_spans(
                    sample_spans,
                    flat_ner=flat_ner,
                    multi_label=multi_label,
                ),
            )
        )
    return results
