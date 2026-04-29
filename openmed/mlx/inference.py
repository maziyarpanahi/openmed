"""MLX token-classification inference pipeline.

Produces output in the same format as HuggingFace ``pipeline("token-classification")``,
so all downstream OpenMed code (OutputFormatter, entity merging, quality gates) works
unchanged.
"""

from __future__ import annotations

import math
import json
import logging
import re
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openmed.core.decoding import (
    TokenLabelInfo,
    build_label_info,
    labels_to_token_spans,
    refine_privacy_filter_span,
    trim_span_whitespace,
    viterbi_decode,
    zero_viterbi_biases,
)
from openmed.mlx.artifact import (
    MANIFEST_FILENAME,
    has_local_tokenizer,
    load_artifact_config,
    read_manifest,
    resolve_tokenizer_reference,
)

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def _tokenizer_has_list_extra_special_tokens(reference: str | Path) -> bool:
    """Return True for local tokenizer configs with legacy list-shaped extras."""
    tokenizer_dir = Path(reference)
    if not tokenizer_dir.exists() or not tokenizer_dir.is_dir():
        return False

    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return False

    try:
        with open(tokenizer_config) as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return isinstance(config.get("extra_special_tokens"), list)


def _load_auto_tokenizer(reference: str | Path) -> Any:
    """Load a tokenizer, tolerating older exported tokenizer metadata."""
    from transformers import AutoTokenizer

    def _from_pretrained(**kwargs: Any) -> Any:
        try:
            return AutoTokenizer.from_pretrained(
                reference,
                fix_mistral_regex=True,
                **kwargs,
            )
        except TypeError as exc:
            if "fix_mistral_regex" not in str(exc):
                raise
            return AutoTokenizer.from_pretrained(reference, **kwargs)

    try:
        return _from_pretrained()
    except AttributeError as exc:
        if (
            "'list' object has no attribute 'keys'" not in str(exc)
            or not _tokenizer_has_list_extra_special_tokens(reference)
        ):
            raise
        return _from_pretrained(extra_special_tokens={})


def _resolve_model_max_length(tokenizer: Any, config: dict[str, Any]) -> int | None:
    """Resolve a practical tokenizer max length from artifact metadata."""
    candidates = (
        config.get("max_position_embeddings"),
        config.get("model_max_length"),
        getattr(tokenizer, "model_max_length", None),
    )
    for value in candidates:
        try:
            max_length = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < max_length < 1_000_000:
            return max_length
    return None


def _tokenize_with_optional_max_length(
    tokenizer: Any,
    max_length: int | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call a tokenizer while avoiding truncation warnings for local artifacts."""
    if max_length is not None and kwargs.get("truncation"):
        kwargs.setdefault("max_length", max_length)
    return tokenizer(*args, **kwargs)


class MLXTokenClassificationPipeline:
    """NER inference pipeline backed by an MLX token-classification model.

    Designed to be a drop-in replacement for HuggingFace's
    ``pipeline("token-classification", aggregation_strategy=...)``.

    Args:
        model_path: Directory containing MLX ``weights.safetensors`` or
            ``weights.npz`` and ``config.json``.
        tokenizer_name: HuggingFace tokenizer to use (usually same as original model).
        aggregation_strategy: How to aggregate sub-word tokens.
            ``None`` for raw per-token output, ``"simple"`` for grouped entities.
    """

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_name: Optional[str] = None,
        aggregation_strategy: Optional[str] = "simple",
    ) -> None:
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required. Install with: pip install openmed[mlx]")

        from openmed.mlx.models import load_model

        self.model_path = Path(model_path)
        self.model = load_model(self.model_path)
        self.aggregation_strategy = aggregation_strategy

        manifest, config = load_artifact_config(self.model_path)

        self.id2label: Dict[int, str] = {
            int(k): v for k, v in config.get("id2label", {}).items()
        }

        # Load HuggingFace tokenizer (framework-agnostic)
        try:
            tok_name = tokenizer_name or resolve_tokenizer_reference(
                self.model_path,
                config=config,
                manifest=manifest,
            )
            self.tokenizer = _load_auto_tokenizer(tok_name)
            self.max_length = _resolve_model_max_length(self.tokenizer, config)
        except ImportError:
            raise ImportError(
                "HuggingFace tokenizers is required for MLX inference. "
                "Install with: pip install tokenizers transformers"
            )

    @staticmethod
    def _is_special_offset(offset: Any) -> bool:
        """Return True for special-token/padding offsets like ``(0, 0)``."""
        return len(offset) >= 2 and offset[0] == 0 and offset[1] == 0

    def __call__(
        self,
        text: str | list[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]] | List[List[Dict[str, Any]]]:
        """Run token classification on *text* or a batch of texts.

        Returns a list of entity dicts matching the HuggingFace format::

            [{"entity_group": "NAME", "score": 0.95,
              "word": "John Doe", "start": 8, "end": 16}, ...]
        """
        if isinstance(text, (list, tuple)):
            return [self._predict_single(item) for item in text]

        return self._predict_single(text)

    def _predict_single(self, text: str) -> List[Dict[str, Any]]:
        """Run token classification for a single input string."""
        # 1. Tokenize
        encoding = _tokenize_with_optional_max_length(
            self.tokenizer,
            self.max_length,
            text,
            return_offsets_mapping=True,
            return_tensors=None,  # plain Python lists
            truncation=True,
            padding=False,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"]

        # 2. Convert to MLX arrays
        input_ids_mx = mx.array([input_ids])
        attention_mask_mx = mx.array([attention_mask], dtype=mx.float32)

        # 3. Forward pass
        logits = self.model(input_ids_mx, attention_mask=attention_mask_mx)
        mx.eval(logits)

        # 4. Softmax → per-token probabilities
        probs = mx.softmax(logits, axis=-1)[0]  # (seq_len, num_labels)
        pred_ids = mx.argmax(probs, axis=-1)  # (seq_len,)

        # Convert to Python
        probs_py = probs.tolist()
        pred_ids_py = pred_ids.tolist()

        # 5. Decode to entity list
        if self.aggregation_strategy is None:
            return self._decode_raw(pred_ids_py, probs_py, offset_mapping, text)
        else:
            return self._decode_grouped(pred_ids_py, probs_py, offset_mapping, text)

    def _decode_raw(
        self,
        pred_ids: List[int],
        probs: List[List[float]],
        offsets: List[List[int]],
        text: str,
    ) -> List[Dict[str, Any]]:
        """Return one dict per token (no aggregation)."""
        results = []
        for i, (label_id, offset) in enumerate(zip(pred_ids, offsets)):
            if self._is_special_offset(offset):
                continue  # skip [CLS], [SEP], padding
            start, end = offset
            label = self.id2label.get(label_id, f"LABEL_{label_id}")
            score = probs[i][label_id]
            if label == "O":
                continue
            results.append({
                "entity": label,
                "score": score,
                "word": text[start:end],
                "start": start,
                "end": end,
                "index": i,
            })
        return results

    def _decode_grouped(
        self,
        pred_ids: List[int],
        probs: List[List[float]],
        offsets: List[List[int]],
        text: str,
    ) -> List[Dict[str, Any]]:
        """Group BIO-tagged tokens into entity spans.

        Mimics HuggingFace ``aggregation_strategy="simple"``.
        """
        entities: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for i, (label_id, offset) in enumerate(zip(pred_ids, offsets)):
            if self._is_special_offset(offset):
                continue  # skip special tokens

            start, end = offset
            label = self.id2label.get(label_id, f"LABEL_{label_id}")
            score = probs[i][label_id]

            if label == "O":
                if current is not None:
                    entities.append(current)
                    current = None
                continue

            # Parse BIO prefix
            if label.startswith("B-") or label.startswith("I-"):
                tag_prefix = label[:2]
                entity_type = label[2:]
            else:
                tag_prefix = "B-"
                entity_type = label

            if tag_prefix == "B-" or current is None or current["_type"] != entity_type:
                # Start new entity
                if current is not None:
                    entities.append(current)
                current = {
                    "entity_group": entity_type,
                    "score": score,
                    "word": text[start:end],
                    "start": start,
                    "end": end,
                    "_type": entity_type,
                    "_scores": [score],
                }
            else:
                # Continue current entity
                current["end"] = end
                current["word"] = text[current["start"]:end]
                current["_scores"].append(score)

        if current is not None:
            entities.append(current)

        # Finalize: average scores, remove internal fields
        for ent in entities:
            scores = ent.pop("_scores")
            ent.pop("_type")
            if self.aggregation_strategy == "first":
                ent["score"] = scores[0]
            elif self.aggregation_strategy == "max":
                ent["score"] = max(scores)
            else:  # "simple" / "average"
                ent["score"] = sum(scores) / len(scores)

        return entities


def _decode_tiktoken_offsets(token_ids: Sequence[int], encoding: Any) -> tuple[str, list[int], list[int]]:
    token_bytes = [encoding.decode_single_token_bytes(int(token_id)) for token_id in token_ids]
    decoded_text = b"".join(token_bytes).decode("utf-8", errors="replace")
    if not token_bytes:
        return decoded_text, [], []

    char_byte_starts: list[int] = []
    char_byte_ends: list[int] = []
    byte_cursor = 0
    for char in decoded_text:
        char_byte_starts.append(byte_cursor)
        byte_cursor += len(char.encode("utf-8"))
        char_byte_ends.append(byte_cursor)

    char_starts: list[int] = []
    char_ends: list[int] = []
    token_byte_cursor = 0
    for raw_bytes in token_bytes:
        token_byte_start = token_byte_cursor
        token_byte_end = token_byte_start + len(raw_bytes)
        token_byte_cursor = token_byte_end
        start_idx = bisect_right(char_byte_ends, token_byte_start)
        end_idx = bisect_left(char_byte_starts, token_byte_end)
        if end_idx < start_idx:
            end_idx = start_idx
        char_starts.append(start_idx)
        char_ends.append(end_idx)

    return decoded_text, char_starts, char_ends


# Span-refinement helpers moved to openmed.core.decoding.spans for reuse
# across MLX and Torch privacy-filter pipelines. Keep local aliases so the
# rest of this module reads unchanged.
_trim_span_whitespace = trim_span_whitespace
_refine_privacy_filter_span = refine_privacy_filter_span


class PrivacyFilterMLXPipeline:
    """OpenAI Privacy Filter inference with tiktoken offsets and BIOES Viterbi."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_name: Optional[str] = None,
        aggregation_strategy: Optional[str] = "simple",
    ) -> None:
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required. Install with: pip install openmed[mlx]")

        from openmed.mlx.models import load_model

        self.model_path = Path(model_path)
        self.model = load_model(self.model_path)
        self.aggregation_strategy = aggregation_strategy
        self.manifest, self.config = load_artifact_config(self.model_path)
        self.id2label: dict[int, str] = {
            int(key): value for key, value in self.config.get("id2label", {}).items()
        }
        self.label_info = build_label_info(self.id2label)
        self.viterbi_biases = self.config.get("_mlx_viterbi_biases") or {}

        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "tiktoken is required for Privacy Filter MLX inference. "
                "Install with: pip install tiktoken"
            ) from exc

        encoding_name = self.config.get("encoding", "o200k_base")
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.tokenizer_name = tokenizer_name

    def __call__(
        self,
        text: str | list[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]] | List[List[Dict[str, Any]]]:
        if isinstance(text, (list, tuple)):
            return [self._predict_single(item) for item in text]
        return self._predict_single(text)

    def _predict_single(self, text: str) -> List[Dict[str, Any]]:
        token_ids = [int(token_id) for token_id in self.encoding.encode(text, allowed_special="all")]
        if not token_ids:
            return []

        input_ids = mx.array([token_ids], dtype=mx.int32)
        attention_mask = mx.ones(input_ids.shape, dtype=mx.bool_)
        logits = self.model(input_ids, attention_mask=attention_mask)
        mx.eval(logits)

        logits = logits.astype(mx.float32)
        log_probs = (logits - mx.logsumexp(logits, axis=-1, keepdims=True))[0]
        probs = mx.exp(log_probs)
        log_probs_py = log_probs.tolist()
        probs_py = probs.tolist()
        pred_ids = viterbi_decode(
            log_probs_py,
            label_info=self.label_info,
            biases=self.viterbi_biases,
        )

        decoded_text, char_starts, char_ends = _decode_tiktoken_offsets(token_ids, self.encoding)
        source_text = decoded_text if decoded_text != text else text

        if self.aggregation_strategy is None:
            return self._decode_raw(pred_ids, probs_py, char_starts, char_ends, source_text)
        return self._decode_grouped(pred_ids, probs_py, char_starts, char_ends, source_text)

    def _decode_raw(
        self,
        pred_ids: list[int],
        probs: list[list[float]],
        char_starts: list[int],
        char_ends: list[int],
        text: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, label_id in enumerate(pred_ids):
            label = self.id2label.get(label_id, f"LABEL_{label_id}")
            if label == "O":
                continue
            start, end = char_starts[index], char_ends[index]
            start, end = _refine_privacy_filter_span(label, start, end, text)
            if end <= start:
                continue
            results.append(
                {
                    "entity": label,
                    "score": probs[index][label_id],
                    "word": text[start:end],
                    "start": start,
                    "end": end,
                    "index": index,
                }
            )
        return results

    def _decode_grouped(
        self,
        pred_ids: list[int],
        probs: list[list[float]],
        char_starts: list[int],
        char_ends: list[int],
        text: str,
    ) -> list[dict[str, Any]]:
        labels_by_index = {index: int(label_id) for index, label_id in enumerate(pred_ids)}
        token_spans = labels_to_token_spans(labels_by_index, self.label_info)

        entities: list[dict[str, Any]] = []
        for span_label, token_start, token_end in token_spans:
            if not (0 <= token_start < token_end <= len(char_starts)):
                continue
            start = char_starts[token_start]
            end = char_ends[token_end - 1]
            start, end = _trim_span_whitespace(start, end, text)
            if end <= start:
                continue
            label = (
                self.label_info.span_class_names[span_label]
                if 0 <= span_label < len(self.label_info.span_class_names)
                else f"label_{span_label}"
            )
            token_scores = [
                probs[index][pred_ids[index]]
                for index in range(token_start, token_end)
                if 0 <= index < len(probs)
            ]
            start, end = _refine_privacy_filter_span(label, start, end, text)
            if end <= start:
                continue
            score = sum(token_scores) / len(token_scores) if token_scores else 0.0
            entities.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": text[start:end],
                    "start": start,
                    "end": end,
                }
            )
        return entities


def _sigmoid(logits: mx.array) -> mx.array:
    return 1.0 / (1.0 + mx.exp(-logits))


def _split_words_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\w+(?:[-_]\w+)*|\S", text):
        words.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return words, offsets


def _prepare_word_mask(
    tokenized_inputs: Any,
    *,
    skip_first_words: int,
) -> list[int]:
    mask: list[int] = []
    seen_words = 0
    previous_word_id: int | None = None
    for word_id in tokenized_inputs.word_ids(0):
        if word_id is None:
            mask.append(0)
        elif word_id != previous_word_id:
            seen_words += 1
            if seen_words <= skip_first_words:
                mask.append(0)
            else:
                mask.append(seen_words - skip_first_words)
        else:
            mask.append(0)
        previous_word_id = word_id
    return mask


def _as_batched_lists(values: Any) -> list[list[int]]:
    if values and isinstance(values[0], list):
        return values
    return [values]


def _build_ragged_span_batch(spans: list[list[tuple[int, int]]]) -> tuple[mx.array, mx.array]:
    max_spans = max((len(sample) for sample in spans), default=0)
    max_spans = max(max_spans, 1)
    padded_spans = [
        [[start, end] for start, end in sample] + [[0, 0]] * (max_spans - len(sample))
        for sample in spans
    ]
    padded_mask = [
        [True] * len(sample) + [False] * (max_spans - len(sample))
        for sample in spans
    ]
    return (
        mx.array(padded_spans, dtype=mx.int32),
        mx.array(padded_mask, dtype=mx.bool_),
    )


def _suppress_overlaps(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for entity in sorted(entities, key=lambda item: (-item["score"], item["start"], item["end"])):
        overlaps = any(
            not (entity["end"] <= current["start"] or entity["start"] >= current["end"])
            for current in selected
        )
        if not overlaps:
            selected.append(entity)
    return sorted(selected, key=lambda item: (item["start"], item["end"], item["label"]))


class _BaseExperimentalMLXPipeline:
    """Shared loader/tokenizer plumbing for experimental MLX tasks."""

    expected_task: str | None = None
    expected_family: str | None = None

    def __init__(self, model_path: str | Path, tokenizer_name: Optional[str] = None) -> None:
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required. Install with: pip install openmed[mlx]")

        from openmed.mlx.models import load_model

        self.model_path = Path(model_path)
        self.model = load_model(self.model_path)
        self.manifest, self.config = load_artifact_config(self.model_path)
        self.task = (self.manifest or {}).get("task") or self.config.get("_mlx_task")
        self.family = (self.manifest or {}).get("family") or self.config.get("_mlx_family")

        if self.expected_task is not None and self.task != self.expected_task:
            raise ValueError(
                f"Expected MLX task {self.expected_task!r}, got {self.task!r} from {self.model_path}."
            )
        if self.expected_family is not None and self.family != self.expected_family:
            raise ValueError(
                f"Expected MLX family {self.expected_family!r}, got {self.family!r} from {self.model_path}."
            )

        tok_name = tokenizer_name or resolve_tokenizer_reference(
            self.model_path,
            config=self.config,
            manifest=self.manifest,
        )
        try:
            self.tokenizer = _load_auto_tokenizer(tok_name)
        except ImportError:
            raise ImportError(
                "HuggingFace tokenizers is required for MLX inference. "
                "Install with: pip install tokenizers transformers"
            )
        self.max_length = _resolve_model_max_length(self.tokenizer, self.config)
        self.prompt_spec = (self.manifest or {}).get("prompt_spec") or self.config.get("_mlx_prompt_spec") or {}


class GLiNERMLXPipeline(_BaseExperimentalMLXPipeline):
    """Experimental MLX zero-shot NER pipeline for GLiNER span checkpoints."""

    expected_task = "zero-shot-ner"
    expected_family = "gliner-uni-encoder-span"

    def predict_entities(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[dict[str, Any]]:
        if not labels:
            return []

        from openmed.mlx.models.gliner_common import build_candidate_span_indices

        words, word_offsets = _split_words_with_offsets(text)
        if not words:
            return []

        entity_token = self.prompt_spec.get("entity_token", "<<ENT>>")
        separator_token = self.prompt_spec.get("separator_token", "<<SEP>>")
        prompt_words: list[str] = []
        for label in labels:
            prompt_words.extend([entity_token, str(label)])
        prompt_words.append(separator_token)

        tokenized = _tokenize_with_optional_max_length(
            self.tokenizer,
            self.max_length,
            [prompt_words + words],
            is_split_into_words=True,
            return_tensors=None,
            truncation=True,
            padding=False,
        )
        words_mask = _prepare_word_mask(tokenized, skip_first_words=len(prompt_words))
        span_idx, span_mask = build_candidate_span_indices([len(words)], self.config.get("max_width", 12))

        outputs = self.model(
            input_ids=mx.array(_as_batched_lists(tokenized["input_ids"]), dtype=mx.int32),
            attention_mask=mx.array(_as_batched_lists(tokenized["attention_mask"]), dtype=mx.float32),
            words_mask=mx.array([words_mask], dtype=mx.int32),
            span_idx=span_idx,
            span_mask=span_mask,
        )

        scores = _sigmoid(outputs["logits"])[0].tolist()
        valid_prompt_count = min(len(labels), int(mx.sum(outputs["prompt_mask"][0].astype(mx.int32)).item()))
        candidate_spans = outputs["span_idx"][0].tolist()
        candidate_mask = outputs["span_mask"][0].tolist()

        entities: list[dict[str, Any]] = []
        for span_index, is_valid in enumerate(candidate_mask):
            if not is_valid:
                continue
            start_word, end_word = candidate_spans[span_index]
            if end_word >= len(word_offsets):
                continue
            start_char = word_offsets[start_word][0]
            end_char = word_offsets[end_word][1]
            for label_index in range(valid_prompt_count):
                score = float(scores[span_index][label_index])
                if score < threshold:
                    continue
                entities.append(
                    {
                        "text": text[start_char:end_char],
                        "label": str(labels[label_index]),
                        "score": score,
                        "start": start_char,
                        "end": end_char,
                    }
                )

        if flat_ner:
            return _suppress_overlaps(entities)

        return sorted(entities, key=lambda item: (item["start"], item["end"], item["label"]))


class GLiClassMLXPipeline(_BaseExperimentalMLXPipeline):
    """Experimental MLX zero-shot classification pipeline for GLiClass."""

    expected_task = "zero-shot-sequence-classification"
    expected_family = "gliclass-uni-encoder"

    def classify(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        if not labels:
            return []

        label_token = self.prompt_spec.get("label_token", "<<LABEL>>")
        separator_token = self.prompt_spec.get("separator_token", "<<SEP>>")
        prompt_first = bool(self.prompt_spec.get("prompt_first", True))

        prompt_parts = [f"{label_token}{label}" for label in labels]
        prompt_parts.append(separator_token)
        if prompt:
            prompt_parts.append(prompt)

        if prompt_first:
            input_text = "".join(prompt_parts) + text
        else:
            input_text = text + "".join(prompt_parts)

        tokenized = _tokenize_with_optional_max_length(
            self.tokenizer,
            self.max_length,
            input_text,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        outputs = self.model(
            input_ids=mx.array([tokenized["input_ids"]], dtype=mx.int32),
            attention_mask=mx.array([tokenized["attention_mask"]], dtype=mx.float32),
        )

        scores = _sigmoid(outputs["logits"])[0].tolist()
        valid_labels = min(len(labels), int(mx.sum(outputs["classes_mask"][0].astype(mx.int32)).item()))
        predictions = []
        for index in range(valid_labels):
            score = float(scores[index])
            if score >= threshold:
                predictions.append({"label": str(labels[index]), "score": score})
        return sorted(predictions, key=lambda item: item["score"], reverse=True)


class GLiNERRelexMLXPipeline(_BaseExperimentalMLXPipeline):
    """Experimental MLX relation-extraction pipeline for GLiNER relex checkpoints."""

    expected_task = "zero-shot-relation-extraction"
    expected_family = "gliner-uni-encoder-token-relex"

    def inference(
        self,
        text: str,
        labels: Sequence[str],
        relations: Sequence[str],
        *,
        threshold: float = 0.5,
        relation_threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        if not labels:
            return {"entities": [], "relations": []}

        from openmed.mlx.models.gliner_common import decode_token_level_spans

        words, word_offsets = _split_words_with_offsets(text)
        if not words:
            return {"entities": [], "relations": []}

        entity_token = self.prompt_spec.get("entity_token", "<<ENT>>")
        relation_token = self.prompt_spec.get("relation_token", "<<REL>>")
        separator_token = self.prompt_spec.get("separator_token", "<<SEP>>")

        prompt_words: list[str] = []
        for label in labels:
            prompt_words.extend([entity_token, str(label)])
        prompt_words.append(separator_token)
        for relation in relations:
            prompt_words.extend([relation_token, str(relation)])
        prompt_words.append(separator_token)

        tokenized = _tokenize_with_optional_max_length(
            self.tokenizer,
            self.max_length,
            [prompt_words + words],
            is_split_into_words=True,
            return_tensors=None,
            truncation=True,
            padding=False,
        )
        words_mask = _prepare_word_mask(tokenized, skip_first_words=len(prompt_words))
        encoded = self.model.encode(
            input_ids=mx.array(_as_batched_lists(tokenized["input_ids"]), dtype=mx.int32),
            attention_mask=mx.array(_as_batched_lists(tokenized["attention_mask"]), dtype=mx.float32),
            words_mask=mx.array([words_mask], dtype=mx.int32),
        )
        entity_scores = _sigmoid(self.model.entity_scores(encoded))
        entity_prompt_count = min(
            len(labels),
            int(mx.sum(encoded["entity_prompt_mask"][0].astype(mx.int32)).item()),
        )
        decoded_result = decode_token_level_spans(
            entity_scores.tolist(),
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )[0]
        decoded_spans = [
            span for span in decoded_result.spans
            if span.label_index < entity_prompt_count
        ]
        span_idx, span_mask = _build_ragged_span_batch([
            [(span.start, span.end) for span in decoded_spans]
        ])
        relation_outputs = self.model.relation_scores(encoded, span_idx, span_mask)

        relation_prompt_count = min(
            len(relations),
            int(mx.sum(relation_outputs["relation_prompt_mask"][0].astype(mx.int32)).item()),
        )

        entities: list[dict[str, Any]] = []
        for entity_index, span in enumerate(decoded_spans):
            start_word = span.start
            end_word = span.end
            if end_word >= len(word_offsets):
                continue
            start_char = word_offsets[start_word][0]
            end_char = word_offsets[end_word][1]

            entities.append(
                {
                    "id": entity_index,
                    "text": text[start_char:end_char],
                    "label": str(labels[span.label_index]),
                    "score": span.score,
                    "start": start_char,
                    "end": end_char,
                }
            )

        pair_scores = _sigmoid(relation_outputs["pair_scores"])[0].tolist()
        pair_idx = relation_outputs["pair_idx"][0].tolist()
        pair_mask = relation_outputs["pair_mask"][0].tolist()

        extracted_relations: list[dict[str, Any]] = []
        for pair_index, is_valid in enumerate(pair_mask):
            if not is_valid:
                continue
            head_index, tail_index = pair_idx[pair_index]
            if head_index >= len(entities) or tail_index >= len(entities):
                continue
            for relation_index in range(relation_prompt_count):
                score = float(pair_scores[pair_index][relation_index])
                if score < relation_threshold:
                    continue
                extracted_relations.append(
                    {
                        "label": str(relations[relation_index]),
                        "score": score,
                        "head": entities[head_index],
                        "tail": entities[tail_index],
                    }
                )

        return {
            "entities": entities,
            "relations": extracted_relations,
        }


# -- MLX model registry -------------------------------------------------------

_MLX_MODEL_MAP: Dict[str, str] = {
    "OpenMed/privacy-filter-mlx": "OpenMed/privacy-filter-mlx",
    "OpenMed/privacy-filter-mlx-8bit": "OpenMed/privacy-filter-mlx-8bit",
    "OpenMed/privacy-filter-nemotron-mlx": "OpenMed/privacy-filter-nemotron-mlx",
    "OpenMed/privacy-filter-nemotron-mlx-8bit": "OpenMed/privacy-filter-nemotron-mlx-8bit",
}


def _download_preconverted_mlx_model(
    repo_id: str,
    cache_dir: Optional[str] = None,
) -> str:
    """Download a pre-converted MLX model snapshot from the Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Install with: pip install openmed[mlx]"
        ) from e

    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=[
            MANIFEST_FILENAME,
            "config.json",
            "id2label.json",
            "weights.safetensors",
            "weights.npz",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "spm.model",
            "sentencepiece.bpe.model",
            "added_tokens.json",
        ],
    )


def _resolve_mlx_model(
    model_name: str,
    config: Any = None,
) -> tuple[str, str]:
    """Resolve a model name to (mlx_model_path, tokenizer_name).

    Tries in order:
    1. Pre-converted MLX model from _MLX_MODEL_MAP
    2. Local path if model_name is a directory with config.json or openmed-mlx.json
    3. On-the-fly conversion from HuggingFace
    """
    from openmed.core.model_registry import OPENMED_MODELS

    # Resolve registry key to full model ID
    if model_name in OPENMED_MODELS:
        full_model_id = OPENMED_MODELS[model_name].model_id
    else:
        full_model_id = model_name

    cache_dir = None
    if config is not None:
        cache_dir = getattr(config, "cache_dir", None)

    # Check pre-converted registry
    if full_model_id in _MLX_MODEL_MAP:
        repo_id = _MLX_MODEL_MAP[full_model_id]
        try:
            mlx_path = _download_preconverted_mlx_model(repo_id, cache_dir=cache_dir)
            return mlx_path, full_model_id
        except Exception as exc:
            logger.warning(
                "Unable to download pre-converted MLX model %s for %s; "
                "falling back to local conversion: %s",
                repo_id,
                full_model_id,
                exc,
            )

    # Check local path
    local = Path(model_name)
    if local.is_dir() and ((local / "config.json").exists() or (local / MANIFEST_FILENAME).exists()):
        manifest = read_manifest(local)
        if manifest is not None or has_local_tokenizer(local):
            try:
                _, local_config = load_artifact_config(local)
            except Exception:
                local_config = {}
            return str(local), resolve_tokenizer_reference(local, config=local_config, manifest=manifest)

        try:
            with open(local / "config.json") as f:
                local_config = json.load(f)
        except Exception:
            local_config = {}

        tokenizer_name = local_config.get("_name_or_path") or model_name
        return str(local), tokenizer_name

    # On-the-fly conversion
    mlx_cache = Path(cache_dir or "~/.cache/openmed/mlx").expanduser()
    safe_name = full_model_id.replace("/", "_")
    output_dir = mlx_cache / safe_name

    if (output_dir / "config.json").exists():
        logger.info("Using cached MLX model at %s", output_dir)
        return str(output_dir), full_model_id

    logger.info("Converting %s to MLX format (one-time) ...", full_model_id)
    from openmed.mlx.convert import convert
    convert(full_model_id, output_dir, cache_dir=cache_dir)
    return str(output_dir), full_model_id


def create_mlx_pipeline(
    model_name: str,
    aggregation_strategy: Optional[str] = "simple",
    config: Any = None,
    **kwargs: Any,
) -> Any:
    """Create an MLX inference pipeline for *model_name*.

    This is the entry point called by :class:`openmed.core.backends.MLXBackend`.
    """
    model_path, tokenizer_name = _resolve_mlx_model(model_name, config)
    manifest, artifact_config = load_artifact_config(model_path)
    from openmed.mlx.models import resolve_artifact_family

    task = (manifest or {}).get("task") or artifact_config.get("_mlx_task", "token-classification")
    raw_family = (manifest or {}).get("family") or artifact_config.get("_mlx_family")
    try:
        family = resolve_artifact_family(artifact_config, manifest=manifest)
    except ValueError:
        family = raw_family

    if family == "openai-privacy-filter":
        return PrivacyFilterMLXPipeline(
            model_path=model_path,
            tokenizer_name=tokenizer_name,
            aggregation_strategy=aggregation_strategy,
        )
    if task == "token-classification":
        return MLXTokenClassificationPipeline(
            model_path=model_path,
            tokenizer_name=tokenizer_name,
            aggregation_strategy=aggregation_strategy,
        )
    if task == "zero-shot-ner" or family == "gliner-uni-encoder-span":
        return GLiNERMLXPipeline(
            model_path=model_path,
            tokenizer_name=tokenizer_name,
        )
    if task == "zero-shot-sequence-classification" or family == "gliclass-uni-encoder":
        return GLiClassMLXPipeline(
            model_path=model_path,
            tokenizer_name=tokenizer_name,
        )
    if task == "zero-shot-relation-extraction" or family == "gliner-uni-encoder-token-relex":
        return GLiNERRelexMLXPipeline(
            model_path=model_path,
            tokenizer_name=tokenizer_name,
        )

    raise ValueError(
        f"Unsupported MLX experimental task {task!r} for family {family!r} at {model_path}."
    )
