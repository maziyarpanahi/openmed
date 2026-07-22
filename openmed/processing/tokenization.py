"""Tokenization utilities for OpenMed."""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from openmed.core.decoding.spans import (
    is_grapheme_boundary,
    is_indic_text,
    iter_grapheme_clusters,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)

_UNSET_MAX_LENGTH_SENTINEL = 1_000_000
DEFAULT_MEDICAL_EXCEPTIONS = [
    "COVID-19",
    "SARS-CoV-2",
    "IL-6",
    "IL-2",
    "TNF-alpha",
    "BCR-ABL1",
    "CAR-T",
    "post-CAR-T",
    "t(8;21)",
    "t(15;17)",
]

_MEDICAL_TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:°?[CFK])?"  # numbers/decimals with optional temperature unit (39.8C)
    r"|[A-Za-z]+(?:-[A-Za-z0-9]+)*"  # words, allowing hyphen chains (IL-6-mediated, COVID-19)
    r"|[A-Za-z0-9]+(?:/[A-Za-z0-9µ]+)+"  # ratios like mg/kg, mmHg/...
    r"|[^ \t\r\n]"  # any other non-space char as its own token
)
_NUMERIC_CONNECTORS = frozenset({"/", "-", ".", ",", ":"})


@dataclass(frozen=True)
class SpanToken:
    text: str
    start: int
    end: int


def grapheme_tokenize(text: str) -> List[SpanToken]:
    """Return one non-whitespace token per extended grapheme cluster.

    This output-oriented tokenizer preserves exact source offsets and keeps
    Indic aksharas, emoji ZWJ sequences, and combining-mark sequences intact.
    It does not create model input IDs.

    Args:
        text: Original, unnormalized Unicode text.

    Returns:
        Non-whitespace tokens with exact source code-point offsets.
    """

    return [
        SpanToken(text[start:end], start, end)
        for start, end in iter_grapheme_clusters(text)
        if not text[start:end].isspace()
    ]


def indic_grapheme_tokenize(text: str) -> List[SpanToken]:
    """Return grapheme-aligned tokens for Indic runs in *text*.

    Non-Indic clusters and whitespace are omitted. Offsets always refer to the
    original string, allowing callers to combine this producer with their
    existing tokenization for other scripts.

    Args:
        text: Original, unnormalized Unicode text.

    Returns:
        Indic grapheme tokens with exact source code-point offsets.
    """

    return [token for token in grapheme_tokenize(text) if is_indic_text(token.text)]


def _append_token(tokens: List[SpanToken], text: str, start: int, end: int) -> None:
    if start < end and not text[start:end].isspace():
        tokens.append(SpanToken(text[start:end], start, end))


def _cluster_kind(cluster: str) -> str:
    if cluster.isspace():
        return "space"
    if all(char.isdecimal() for char in cluster):
        return "number"
    if any(unicodedata.category(char).startswith("L") for char in cluster):
        return "word"
    if cluster in _NUMERIC_CONNECTORS:
        return "connector"
    return "punctuation"


def indic_word_tokenize(text: str) -> List[SpanToken]:
    """Tokenize mixed Indic text without bisecting a grapheme cluster.

    Letter clusters are grouped into words, native or ASCII digit runs retain
    internal date/number separators, and punctuation is emitted at its own
    boundary. Returned spans always index ``text`` exactly.
    """

    clusters = list(iter_grapheme_clusters(text))
    tokens: List[SpanToken] = []
    token_start: Optional[int] = None
    token_end = 0
    token_kind: Optional[str] = None

    for cluster_index, (start, end) in enumerate(clusters):
        cluster = text[start:end]
        kind = _cluster_kind(cluster)
        next_kind = (
            _cluster_kind(text[slice(*clusters[cluster_index + 1])])
            if cluster_index + 1 < len(clusters)
            else None
        )

        if kind == "connector" and token_kind == "number" and next_kind == "number":
            token_end = end
            continue

        if kind in {"word", "number"}:
            if token_start is not None and token_kind != kind:
                _append_token(tokens, text, token_start, token_end)
                token_start = None
            if token_start is None:
                token_start = start
                token_kind = kind
            token_end = end
            continue

        if token_start is not None:
            _append_token(tokens, text, token_start, token_end)
            token_start = None
            token_kind = None

        if kind == "punctuation" or kind == "connector":
            _append_token(tokens, text, start, end)

    if token_start is not None:
        _append_token(tokens, text, token_start, token_end)

    return tokens


def _regex_medical_tokens(
    text: str,
    *,
    offset: int = 0,
) -> List[SpanToken]:
    return [
        SpanToken(match.group(0), offset + match.start(), offset + match.end())
        for match in _MEDICAL_TOKEN_PATTERN.finditer(text)
    ]


def _medical_tokens_in_segment(text: str, *, offset: int = 0) -> List[SpanToken]:
    tokens: List[SpanToken] = []
    non_indic_start = 0

    for cluster_start, cluster_end in iter_grapheme_clusters(text):
        cluster = text[cluster_start:cluster_end]
        if not is_indic_text(cluster):
            continue
        if non_indic_start < cluster_start:
            tokens.extend(
                _regex_medical_tokens(
                    text[non_indic_start:cluster_start],
                    offset=offset + non_indic_start,
                )
            )
        if not cluster.isspace():
            tokens.append(
                SpanToken(cluster, offset + cluster_start, offset + cluster_end)
            )
        non_indic_start = cluster_end

    if non_indic_start < len(text):
        tokens.extend(
            _regex_medical_tokens(
                text[non_indic_start:],
                offset=offset + non_indic_start,
            )
        )
    return tokens


def medical_tokenize(
    text: str,
    *,
    exceptions: Optional[Iterable[str]] = None,
) -> List[SpanToken]:
    """Tokenize clinical text into stable span tokens for output remapping.

    This tokenizer is **not** used to create model input ids. It is used to produce
    user-facing token boundaries and to remap model predictions back onto medical-friendly
    spans.
    """
    exceptions_set = {e for e in (exceptions or []) if e}
    if not exceptions_set:
        return _medical_tokens_in_segment(text)

    protected: List[Tuple[int, int]] = []
    grapheme_boundaries = {0, len(text)}
    grapheme_boundaries.update(end for _, end in iter_grapheme_clusters(text))
    for exc in sorted(exceptions_set, key=len, reverse=True):
        start = 0
        while True:
            idx = text.find(exc, start)
            if idx == -1:
                break
            span = (idx, idx + len(exc))
            if not (
                is_grapheme_boundary(span[0], text)
                and is_grapheme_boundary(span[1], text)
                and span[0] in grapheme_boundaries
                and span[1] in grapheme_boundaries
            ):
                start = idx + 1
                continue
            if any(not (span[1] <= a or span[0] >= b) for a, b in protected):
                start = idx + 1
                continue
            protected.append(span)
            start = idx + len(exc)

    if not protected:
        return _medical_tokens_in_segment(text)

    protected.sort()
    tokens: List[SpanToken] = []
    cursor = 0
    for s, e in protected:
        if cursor < s:
            tokens.extend(_medical_tokens_in_segment(text[cursor:s], offset=cursor))
        tokens.append(SpanToken(text[s:e], s, e))
        cursor = e
    if cursor < len(text):
        tokens.extend(_medical_tokens_in_segment(text[cursor:], offset=cursor))

    return [
        t for t in sorted(tokens, key=lambda x: (x.start, x.end)) if t.end > t.start
    ]


def remap_predictions_to_tokens(
    predictions: List[Dict[str, Any]],
    text: str,
    tokens: List[SpanToken],
    *,
    gap: int = 1,
) -> List[Dict[str, Any]]:
    """Remap model predictions (char spans) onto custom tokens and merge contiguous tokens.

    Returns a list of prediction dicts compatible with OutputFormatter.
    """
    if not predictions or not tokens:
        return predictions

    token_labels: List[Optional[str]] = [None] * len(tokens)
    token_scores: List[float] = [0.0] * len(tokens)
    token_meta: List[Optional[Dict[str, Any]]] = [None] * len(tokens)

    for idx, tok in enumerate(tokens):
        best_label: Optional[str] = None
        best_score = -1.0
        best_meta: Optional[Dict[str, Any]] = None
        for pred in predictions:
            start = pred.get("start")
            end = pred.get("end")
            if not (isinstance(start, int) and isinstance(end, int)):
                continue
            if end <= tok.start or start >= tok.end:
                continue
            raw_label = pred.get("entity_group") or pred.get("entity") or ""
            if not raw_label:
                continue
            clean = str(raw_label).replace("B-", "").replace("I-", "")
            score = float(pred.get("score", 0.0) or 0.0)
            if score > best_score:
                best_label = clean
                best_score = score
                meta = pred.get("metadata")
                best_meta = dict(meta) if isinstance(meta, dict) else None
        if best_label is not None:
            token_labels[idx] = best_label
            token_scores[idx] = best_score
            token_meta[idx] = best_meta

    remapped: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        label = token_labels[i]
        if label is None:
            i += 1
            continue
        start = tokens[i].start
        end = tokens[i].end
        scores = [token_scores[i]]
        meta = token_meta[i] or {}
        j = i + 1
        while (
            j < len(tokens)
            and token_labels[j] == label
            and tokens[j].start <= end + gap
        ):
            end = tokens[j].end
            scores.append(token_scores[j])
            if not meta and token_meta[j]:
                meta = token_meta[j] or {}
            j += 1

        remapped.append(
            {
                "start": start,
                "end": end,
                "score": sum(scores) / len(scores),
                "entity_group": label,
                "word": text[start:end],
                "metadata": meta,
            }
        )
        i = j

    return remapped


def _is_reasonable_length(
    value: Optional[int], threshold: int = _UNSET_MAX_LENGTH_SENTINEL
) -> bool:
    if value is None:
        return False
    try:
        # Some tokenizers return special sentinel values like `int(1e30)`
        as_int = int(value)
    except (TypeError, ValueError):
        return False
    if as_int <= 0:
        return False
    return as_int < threshold


def infer_tokenizer_max_length(
    tokenizer: "PreTrainedTokenizer",
    *,
    fallback: Optional[int] = None,
    threshold: int = _UNSET_MAX_LENGTH_SENTINEL,
) -> Optional[int]:
    """Infer a sensible maximum sequence length for a tokenizer.

    Many Hugging Face tokenizers expose very large placeholder values (e.g., ``int(1e30)``,
    ``2**63 - 1``) when the underlying model does not specify an explicit limit. This helper
    collapses the common hints into a single manageable integer suitable for truncation.

    Args:
        tokenizer: Hugging Face tokenizer instance.
        fallback: Optional value to return if no reasonable limit is discovered.
        threshold: Maximum value considered a realistic limit.

    Returns:
        Inferred maximum length or ``None`` if unknown.
    """
    candidates: List[Optional[int]] = []

    max_length = getattr(tokenizer, "model_max_length", None)
    if _is_reasonable_length(max_length, threshold):
        return int(max_length)
    candidates.append(max_length)  # record for debugging

    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    kw_max = init_kwargs.get("model_max_length")
    if _is_reasonable_length(kw_max, threshold):
        return int(kw_max)
    candidates.append(kw_max)

    config = getattr(tokenizer, "config", None)
    if config is not None:
        for attr in (
            "model_max_length",
            "max_position_embeddings",
            "n_positions",
            "seq_length",
        ):
            candidate = getattr(config, attr, None)
            if _is_reasonable_length(candidate, threshold):
                return int(candidate)
            candidates.append(candidate)

    if fallback is not None and _is_reasonable_length(fallback, threshold):
        return int(fallback)

    logger.debug(
        "Tokenizer %s did not expose a reasonable max_length; candidates=%s",
        getattr(tokenizer, "name_or_path", "<unknown>"),
        candidates,
    )
    return None


class TokenizationHelper:
    """Helper class for tokenization operations in medical text."""

    def __init__(self, tokenizer: Optional["PreTrainedTokenizer"] = None):
        """Initialize tokenization helper.

        Args:
            tokenizer: HuggingFace tokenizer instance.
        """
        self.tokenizer = tokenizer

    def tokenize_with_alignment(
        self, text: str, return_word_ids: bool = True
    ) -> Dict[str, Any]:
        """Tokenize text while maintaining word alignment.

        Args:
            text: Input text to tokenize.
            return_word_ids: Whether to return word ID mappings.

        Returns:
            Dictionary containing tokenization results and alignments.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")

        # Tokenize with special tokens
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True,
        )

        result = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "tokens": self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0]),
            "offset_mapping": encoding.get("offset_mapping"),
            "special_tokens_mask": encoding.get("special_tokens_mask"),
        }

        if return_word_ids:
            result["word_ids"] = encoding.word_ids()

        return result

    def align_predictions_to_words(
        self,
        predictions: List[Any],
        word_ids: List[Optional[int]],
        text: str,
        aggregation_strategy: str = "first",
    ) -> List[Tuple[str, Any]]:
        """Align token-level predictions to word-level predictions.

        Args:
            predictions: Token-level predictions.
            word_ids: Word ID mappings from tokenizer.
            text: Original text.
            aggregation_strategy: How to aggregate subword predictions
                                 ("first", "max", "average").

        Returns:
            List of (word, prediction) tuples.
        """
        if len(predictions) != len(word_ids):
            raise ValueError("Predictions and word_ids must have same length")

        words = text.split()
        word_predictions = {}

        for i, (pred, word_id) in enumerate(zip(predictions, word_ids)):
            if word_id is None:  # Special tokens
                continue

            if word_id not in word_predictions:
                word_predictions[word_id] = []
            word_predictions[word_id].append(pred)

        # Aggregate predictions for each word
        result = []
        for word_id in sorted(word_predictions.keys()):
            if word_id < len(words):
                word = words[word_id]
                preds = word_predictions[word_id]

                if aggregation_strategy == "first":
                    final_pred = preds[0]
                elif aggregation_strategy == "max":
                    final_pred = (
                        max(preds) if isinstance(preds[0], (int, float)) else preds[0]
                    )
                elif aggregation_strategy == "average":
                    if isinstance(preds[0], (int, float)):
                        final_pred = sum(preds) / len(preds)
                    else:
                        final_pred = preds[0]  # Can't average non-numeric
                else:
                    final_pred = preds[0]

                result.append((word, final_pred))

        return result

    def create_attention_masks(
        self, input_ids: List[List[int]], pad_token_id: int
    ) -> List[List[int]]:
        """Create attention masks for batched inputs.

        Args:
            input_ids: Batched input token IDs.
            pad_token_id: ID of the padding token.

        Returns:
            Attention masks.
        """
        attention_masks = []
        for ids in input_ids:
            mask = [1 if token_id != pad_token_id else 0 for token_id in ids]
            attention_masks.append(mask)
        return attention_masks

    def truncate_sequences(
        self,
        sequences: List[str],
        max_length: int,
        truncation_strategy: str = "longest_first",
    ) -> List[str]:
        """Truncate sequences to fit within max_length.

        Args:
            sequences: List of text sequences.
            max_length: Maximum sequence length.
            truncation_strategy: How to truncate ("longest_first", "do_not_truncate").

        Returns:
            Truncated sequences.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")

        truncated = []
        for seq in sequences:
            tokens = self.tokenizer.tokenize(seq)
            if len(tokens) <= max_length:
                truncated.append(seq)
            else:
                if truncation_strategy == "longest_first":
                    truncated_tokens = tokens[:max_length]
                    truncated_text = self.tokenizer.convert_tokens_to_string(
                        truncated_tokens
                    )
                    truncated.append(truncated_text)
                else:
                    truncated.append(seq)  # Don't truncate

        return truncated

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, Any]:
        """Batch encode multiple texts.

        Args:
            texts: List of texts to encode.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.

        Returns:
            Encoded batch.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")

        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
