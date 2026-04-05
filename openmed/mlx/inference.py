"""MLX token-classification inference pipeline.

Produces output in the same format as HuggingFace ``pipeline("token-classification")``,
so all downstream OpenMed code (OutputFormatter, entity merging, quality gates) works
unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


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

        # Load config for id2label
        with open(self.model_path / "config.json") as f:
            config = json.load(f)

        self.id2label: Dict[int, str] = {
            int(k): v for k, v in config.get("id2label", {}).items()
        }

        # Load HuggingFace tokenizer (framework-agnostic)
        try:
            from transformers import AutoTokenizer
            tok_name = tokenizer_name or config.get("_name_or_path", str(self.model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
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
        encoding = self.tokenizer(
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


# -- MLX model registry -------------------------------------------------------

_MLX_MODEL_MAP: Dict[str, str] = {
    # Public runtime defaults currently prefer local/on-the-fly conversion.
    # Private pre-converted snapshots can be wired here later if needed.
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
            "config.json",
            "id2label.json",
            "weights.safetensors",
            "weights.npz",
        ],
    )


def _resolve_mlx_model(
    model_name: str,
    config: Any = None,
) -> tuple[str, str]:
    """Resolve a model name to (mlx_model_path, tokenizer_name).

    Tries in order:
    1. Pre-converted MLX model from _MLX_MODEL_MAP
    2. Local path if model_name is a directory with config.json
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
    if local.is_dir() and (local / "config.json").exists():
        bundled_tokenizer_files = (
            local / "tokenizer.json",
            local / "tokenizer_config.json",
            local / "special_tokens_map.json",
            local / "vocab.txt",
            local / "merges.txt",
            local / "spm.model",
        )
        if any(path.exists() for path in bundled_tokenizer_files):
            return str(local), str(local)

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
) -> MLXTokenClassificationPipeline:
    """Create an MLX inference pipeline for *model_name*.

    This is the entry point called by :class:`openmed.core.backends.MLXBackend`.
    """
    model_path, tokenizer_name = _resolve_mlx_model(model_name, config)
    return MLXTokenClassificationPipeline(
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        aggregation_strategy=aggregation_strategy,
    )
