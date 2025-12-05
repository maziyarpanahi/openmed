"""Tokenization utilities for OpenMed."""

from typing import List, Dict, Any, Optional, Tuple, Union, Iterable
import logging
import re

try:
    from transformers import PreTrainedTokenizer
    HF_AVAILABLE = True
except (ImportError, OSError):
    HF_AVAILABLE = False

try:
    from tokenizers import pre_tokenizers
    from tokenizers.pre_tokenizers import Split, Sequence as PreSeq
    TOKENIZERS_AVAILABLE = True
except (ImportError, OSError):
    TOKENIZERS_AVAILABLE = False

logger = logging.getLogger(__name__)

_UNSET_MAX_LENGTH_SENTINEL = 1_000_000

def build_medical_pretokenizer(exceptions: Optional[Iterable[str]] = None):
    """Return a safe Bert-style pretokenizer. Exceptions are handled via added tokens."""
    if not TOKENIZERS_AVAILABLE:
        logger.warning("tokenizers package not available; skipping medical pretokenizer")
        return None
    try:
        return pre_tokenizers.BertPreTokenizer()
    except AttributeError:  # pragma: no cover - older tokenizers versions
        return pre_tokenizers.Whitespace()


def apply_medical_pretokenizer(tokenizer: Any, exceptions: Optional[Iterable[str]] = None) -> bool:
    """Attach the medical pretokenizer to a Hugging Face fast tokenizer.

    Returns True if applied, False otherwise.
    """
    if not TOKENIZERS_AVAILABLE:
        return False
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return False
    pre_tok = build_medical_pretokenizer(exceptions=exceptions)
    if pre_tok is None:
        return False
    try:
        backend.pre_tokenizer = pre_tok  # type: ignore[attr-defined]
        if exceptions:
            try:
                tokenizer.add_tokens(list(exceptions), special_tokens=False)
            except Exception:
                pass
        return True
    except Exception as exc:
        logger.warning("Medical pre-tokenizer not applied: %s", exc)
        return False


def _is_reasonable_length(value: Optional[int], threshold: int = _UNSET_MAX_LENGTH_SENTINEL) -> bool:
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
        self,
        text: str,
        return_word_ids: bool = True
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
            return_special_tokens_mask=True
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
        aggregation_strategy: str = "first"
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
                    final_pred = max(preds) if isinstance(preds[0], (int, float)) else preds[0]
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
        self,
        input_ids: List[List[int]],
        pad_token_id: int
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
        truncation_strategy: str = "longest_first"
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
                    truncated_text = self.tokenizer.convert_tokens_to_string(truncated_tokens)
                    truncated.append(truncated_text)
                else:
                    truncated.append(seq)  # Don't truncate

        return truncated

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
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
            return_tensors="pt"
        )
