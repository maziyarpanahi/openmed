"""BIOES-aware Viterbi decoder for token-classification logits.

Used by both the MLX and PyTorch privacy-filter pipelines. Pure-Python,
no array-framework dependencies — operates on lists of floats produced
by ``logits.tolist()`` from either backend.

The algorithm enforces the standard BIOES (B-egin / I-nside / E-nd /
S-ingleton, plus background ``O``) state machine and supports six learned
transition biases consumed by the OpenAI privacy-filter family.
"""

from __future__ import annotations

import math
from typing import Final, Sequence


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

VITERBI_BIAS_KEYS: Final = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


class TokenLabelInfo:
    """Resolves a flat ``id2label`` map into BIOES tag / span-label tables.

    Token labels arrive as strings like ``B-NAME``, ``I-NAME``, ``E-NAME``,
    ``S-NAME``, plus the background ``O``. ``TokenLabelInfo`` precomputes
    the mappings used during decoding:

    - ``span_class_names``: deduplicated list of underlying span labels
      (``["O", "NAME", "EMAIL", ...]``) where index 0 is always background.
    - ``span_label_lookup``: ``{base_label: span_index}``.
    - ``token_to_span_label``: ``{token_label_id: span_index}``.
    - ``token_boundary_tags``: ``{token_label_id: "B" | "I" | "E" | "S" | None}``
      with ``None`` reserved for the background token.
    - ``background_token_label`` / ``background_span_label``: indices of
      the ``O`` class in token-label and span-label space respectively.
    """

    def __init__(self, class_names: Sequence[str]) -> None:
        self.span_class_names: list[str] = ["O"]
        self.span_label_lookup: dict[str, int] = {"O": 0}
        self.token_to_span_label: dict[int, int] = {}
        self.token_boundary_tags: dict[int, str | None] = {}
        self.background_token_label = 0
        self.background_span_label = 0

        for index, label in enumerate(class_names):
            if label == "O":
                self.background_token_label = index
                self.token_to_span_label[index] = self.background_span_label
                self.token_boundary_tags[index] = None
                continue

            boundary, base_label = _split_boundary_label(label)
            span_label = self.span_label_lookup.get(base_label)
            if span_label is None:
                span_label = len(self.span_class_names)
                self.span_class_names.append(base_label)
                self.span_label_lookup[base_label] = span_label
            self.token_to_span_label[index] = span_label
            self.token_boundary_tags[index] = boundary


def build_label_info(id2label: dict[int, str]) -> TokenLabelInfo:
    """Construct a ``TokenLabelInfo`` from a ``{id: label_string}`` map."""
    class_names = [id2label[index] for index in sorted(id2label)]
    return TokenLabelInfo(class_names)


def zero_viterbi_biases() -> dict[str, float]:
    """Return a zero-initialized bias dict keyed by ``VITERBI_BIAS_KEYS``."""
    return {key: 0.0 for key in VITERBI_BIAS_KEYS}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_NEG_INF: Final = -1e9


def _split_boundary_label(label: str) -> tuple[str, str]:
    if len(label) > 2 and label[1] == "-" and label[0] in {"B", "I", "E", "S"}:
        return label[0], label[2:]
    return "B", label


def _transition_bias(
    *,
    previous_tag: str | None,
    previous_span: int | None,
    next_tag: str | None,
    next_span: int | None,
    background_token_idx: int,
    background_span_idx: int,
    previous_idx: int,
    next_idx: int,
    biases: dict[str, float],
) -> float:
    previous_is_background = (
        previous_span == background_span_idx or previous_idx == background_token_idx
    )
    next_is_background = next_span == background_span_idx or next_idx == background_token_idx

    if previous_is_background:
        if next_is_background:
            return biases["transition_bias_background_stay"]
        if next_tag in {"B", "S"}:
            return biases["transition_bias_background_to_start"]
        return 0.0

    if previous_tag in {"B", "I"}:
        if next_tag == "I" and previous_span == next_span:
            return biases["transition_bias_inside_to_continue"]
        if next_tag == "E" and previous_span == next_span:
            return biases["transition_bias_inside_to_end"]
        return 0.0

    if previous_tag in {"E", "S"}:
        if next_is_background:
            return biases["transition_bias_end_to_background"]
        if next_tag in {"B", "S"}:
            return biases["transition_bias_end_to_start"]
        return 0.0

    return 0.0


def _is_valid_transition(
    *,
    previous_tag: str | None,
    previous_span: int | None,
    next_tag: str | None,
    next_span: int | None,
    background_token_idx: int,
    background_span_idx: int,
    next_idx: int,
) -> bool:
    next_is_background = next_span == background_span_idx or next_idx == background_token_idx
    if (next_span is None or next_tag is None) and not next_is_background:
        return False

    if previous_span is None or previous_tag is None:
        return next_is_background or next_tag in {"B", "S"}

    previous_is_background = previous_span == background_span_idx
    if previous_is_background:
        return next_is_background or next_tag in {"B", "S"}
    if previous_tag in {"E", "S"}:
        return next_is_background or next_tag in {"B", "S"}
    if previous_tag in {"B", "I"}:
        return previous_span == next_span and next_tag in {"I", "E"}
    return False


def _build_viterbi_scores(
    label_info: TokenLabelInfo,
    biases: dict[str, float],
) -> tuple[list[float], list[float], list[list[float]]]:
    num_classes = len(label_info.token_to_span_label)
    start_scores = [_NEG_INF] * num_classes
    end_scores = [_NEG_INF] * num_classes
    transition_scores = [[_NEG_INF] * num_classes for _ in range(num_classes)]

    background_token_idx = label_info.background_token_label
    background_span_idx = label_info.background_span_label
    for previous_idx in range(num_classes):
        previous_tag = label_info.token_boundary_tags.get(previous_idx)
        previous_span = label_info.token_to_span_label.get(previous_idx)
        if previous_tag in {"B", "S"} or previous_idx == background_token_idx:
            start_scores[previous_idx] = 0.0
        if previous_tag in {"E", "S"} or previous_idx == background_token_idx:
            end_scores[previous_idx] = 0.0

        for next_idx in range(num_classes):
            next_tag = label_info.token_boundary_tags.get(next_idx)
            next_span = label_info.token_to_span_label.get(next_idx)
            if _is_valid_transition(
                previous_tag=previous_tag,
                previous_span=previous_span,
                next_tag=next_tag,
                next_span=next_span,
                background_token_idx=background_token_idx,
                background_span_idx=background_span_idx,
                next_idx=next_idx,
            ):
                transition_scores[previous_idx][next_idx] = _transition_bias(
                    previous_tag=previous_tag,
                    previous_span=previous_span,
                    next_tag=next_tag,
                    next_span=next_span,
                    background_token_idx=background_token_idx,
                    background_span_idx=background_span_idx,
                    previous_idx=previous_idx,
                    next_idx=next_idx,
                    biases=biases,
                )
    return start_scores, end_scores, transition_scores


# ---------------------------------------------------------------------------
# Public decoders
# ---------------------------------------------------------------------------

def viterbi_decode(
    token_logprobs: list[list[float]],
    *,
    label_info: TokenLabelInfo,
    biases: dict[str, float],
) -> list[int]:
    """Decode token-class indices via constrained BIOES Viterbi.

    Args:
        token_logprobs: One inner list of class log-probabilities per token.
        label_info: Tag/span mapping built from the model's ``id2label``.
        biases: Optional learned transition biases keyed by
            ``VITERBI_BIAS_KEYS``. Unknown keys are ignored.

    Returns:
        A list of class indices (one per token) representing the most
        likely BIOES-valid path. Falls back to a per-token argmax when
        no finite-score path exists (e.g. degenerate label space).
    """
    if not token_logprobs:
        return []

    resolved_biases = zero_viterbi_biases()
    resolved_biases.update({key: float(value) for key, value in biases.items() if key in resolved_biases})
    start_scores, end_scores, transition_scores = _build_viterbi_scores(
        label_info,
        resolved_biases,
    )

    num_classes = len(label_info.token_to_span_label)
    if any(len(row) < num_classes for row in token_logprobs):
        raise ValueError("token_logprobs has fewer classes than the configured label space")
    scores = [token_logprobs[0][idx] + start_scores[idx] for idx in range(num_classes)]
    backpointers: list[list[int]] = []

    for token_scores in token_logprobs[1:]:
        next_scores: list[float] = []
        paths: list[int] = []
        for next_idx in range(num_classes):
            best_idx = 0
            best_score = -math.inf
            for previous_idx, previous_score in enumerate(scores):
                score = previous_score + transition_scores[previous_idx][next_idx]
                if score > best_score:
                    best_score = score
                    best_idx = previous_idx
            next_scores.append(best_score + token_scores[next_idx])
            paths.append(best_idx)
        scores = next_scores
        backpointers.append(paths)

    final_scores = [score + end_scores[idx] for idx, score in enumerate(scores)]
    if not any(math.isfinite(score) for score in final_scores):
        return [max(range(num_classes), key=lambda idx: row[idx]) for row in token_logprobs]

    last_label = max(range(num_classes), key=lambda idx: final_scores[idx])
    path = [last_label]
    for paths in reversed(backpointers):
        last_label = paths[last_label]
        path.append(last_label)
    path.reverse()
    return path


def labels_to_token_spans(
    labels_by_index: dict[int, int],
    label_info: TokenLabelInfo,
) -> list[tuple[int, int, int]]:
    """Convert per-token class indices into ``(span_label, start, end)`` triples.

    The decoder honours BIOES boundary semantics:
      - ``S`` produces a singleton ``[idx, idx+1)`` span.
      - ``B`` opens a new span; mismatched continuations close + restart.
      - ``E`` closes the current span at ``[start, idx+1)``.
      - The background token (``O``) closes any open span without emitting it.

    Gaps in ``labels_by_index`` (non-contiguous keys) are treated as a hard
    span break — they typically indicate the caller is filtering tokens
    pre-decoding (e.g. dropping special-token positions).
    """
    spans: list[tuple[int, int, int]] = []
    current_label: int | None = None
    start_idx: int | None = None
    previous_idx: int | None = None

    for token_idx in sorted(labels_by_index):
        label_id = labels_by_index[token_idx]
        span_label = label_info.token_to_span_label.get(label_id)
        boundary_tag = label_info.token_boundary_tags.get(label_id)

        if previous_idx is not None and token_idx != previous_idx + 1:
            if current_label is not None and start_idx is not None:
                spans.append((current_label, start_idx, previous_idx + 1))
            current_label = None
            start_idx = None

        if span_label is None:
            previous_idx = token_idx
            continue

        if span_label == label_info.background_span_label:
            if current_label is not None and start_idx is not None:
                spans.append((current_label, start_idx, token_idx))
            current_label = None
            start_idx = None
            previous_idx = token_idx
            continue

        if boundary_tag == "S":
            if current_label is not None and start_idx is not None and previous_idx is not None:
                spans.append((current_label, start_idx, previous_idx + 1))
            spans.append((span_label, token_idx, token_idx + 1))
            current_label = None
            start_idx = None
        elif boundary_tag == "B":
            if current_label is not None and start_idx is not None and previous_idx is not None:
                spans.append((current_label, start_idx, previous_idx + 1))
            current_label = span_label
            start_idx = token_idx
        elif boundary_tag == "I":
            if current_label is None or current_label != span_label:
                if current_label is not None and start_idx is not None and previous_idx is not None:
                    spans.append((current_label, start_idx, previous_idx + 1))
                current_label = span_label
                start_idx = token_idx
        elif boundary_tag == "E":
            if current_label is None or current_label != span_label or start_idx is None:
                if current_label is not None and start_idx is not None and previous_idx is not None:
                    spans.append((current_label, start_idx, previous_idx + 1))
                spans.append((span_label, token_idx, token_idx + 1))
                current_label = None
                start_idx = None
            else:
                spans.append((current_label, start_idx, token_idx + 1))
                current_label = None
                start_idx = None

        previous_idx = token_idx

    if current_label is not None and start_idx is not None and previous_idx is not None:
        spans.append((current_label, start_idx, previous_idx + 1))
    return spans
