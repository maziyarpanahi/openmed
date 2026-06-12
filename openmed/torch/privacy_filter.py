"""PyTorch wrapper for the OpenAI ``privacy-filter`` token classifier.

This is the cross-platform path used when MLX is unavailable (Linux,
Windows, Intel Mac). It loads ``openai/privacy-filter`` (or any compatible
HuggingFace fine-tune) via ``transformers.AutoModelForTokenClassification``
and produces the same entity-dict shape as the MLX pipeline:

    {"entity_group": str, "score": float, "word": str, "start": int, "end": int}

So downstream code (``extract_pii``, smart-merging, deidentification) can
consume MLX and Torch results interchangeably.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from openmed.core.decoding import refine_privacy_filter_span, trim_span_whitespace

logger = logging.getLogger(__name__)


# First-party privacy-filter repos that legitimately require
# trust_remote_code=True (they ship modeling_openai_privacy_filter.py and
# friends in the repo and rely on Transformers' auto_map import).
# Stored in lower-case because HuggingFace namespaces are case-insensitive
# (OpenMed and openmed resolve to the same org).  Normalising both sides
# prevents a user-supplied "openmed/privacy-filter-multilingual" from failing
# to match the hard-coded "OpenMed/..." entry.
TRUSTED_REMOTE_CODE_MODELS = frozenset({
    "openai/privacy-filter",
    "openmed/privacy-filter-multilingual",
    "openmed/privacy-filter-nemotron",
})

# Operators with custom fine-tunes can extend the allowlist with a
# comma-separated list of HuggingFace repo IDs. Empty entries are ignored.
_ALLOWLIST_ENV_VAR = "OPENMED_TRUSTED_REMOTE_CODE_MODELS"


def _env_allowlist() -> frozenset[str]:
    raw = os.getenv(_ALLOWLIST_ENV_VAR, "")
    return frozenset(
        part.strip().lower() for part in raw.split(",") if part.strip()
    )


def is_trusted_for_remote_code(model_name: str) -> bool:
    """Return True if *model_name* may be loaded with ``trust_remote_code=True``.

    Trusted sources:

    - ``TRUSTED_REMOTE_CODE_MODELS`` — first-party OpenAI/OpenMed
      privacy-filter repos that ship custom modeling code.
    - The ``OPENMED_TRUSTED_REMOTE_CODE_MODELS`` env var — operator-extensible
      comma-separated list of repo IDs for custom fine-tunes.
    - Local filesystem paths that identify as a privacy-filter artifact
      via on-disk metadata (the ``_is_privacy_filter_artifact_path`` check
      already used by the dispatcher).
    """
    if not model_name:
        return False
    normalized = model_name.lower()
    if normalized in TRUSTED_REMOTE_CODE_MODELS:
        return True
    if normalized in _env_allowlist():
        return True
    # Local path check is deferred (it touches the filesystem) and imported
    # lazily to avoid a circular import with openmed.core.pii.
    from openmed.core.pii import _is_privacy_filter_artifact_path
    return _is_privacy_filter_artifact_path(model_name)


class PrivacyFilterTorchPipeline:
    """Run ``openai/privacy-filter`` (or compatible) via Transformers.

    Output shape matches :class:`openmed.mlx.inference.PrivacyFilterMLXPipeline`
    — both pipelines emit a list of ``{entity_group, score, word, start, end}``
    dicts so the rest of OpenMed's privacy machinery is backend-agnostic.

    Args:
        model_name: HuggingFace model ID or local path. Default
            ``openai/privacy-filter``.
        device: Torch device string (``cpu``, ``cuda``, ``cuda:0``, ``mps``).
            ``None`` autodetects: CUDA if available, else CPU.
        dtype: Optional torch dtype (e.g. ``"float16"``, ``"bfloat16"``).
            Defaults to model native.
        aggregation_strategy: Passed through to HF's pipeline. ``"simple"``
            (the default) groups BIOES tokens into spans and matches MLX
            output shape.
        local_files_only: When True, never download from the Hub — only
            use a cached copy. Mirrors the demo's offline-first default.
        trust_remote_code: When True, the loader permits Transformers to
            execute custom Python shipped inside the model repo via
            ``auto_map``. This is required by the first-party Privacy
            Filter models (which ship ``modeling_openai_privacy_filter.py``)
            but is dangerous for arbitrary HuggingFace repositories.
            Defaults to ``False``. When True, ``model_name`` must be in the
            allowlist resolved by :func:`is_trusted_for_remote_code` —
            otherwise a :class:`ValueError` is raised before any download.
    """

    DEFAULT_MODEL_ID = "openai/privacy-filter"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        aggregation_strategy: str = "simple",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        if trust_remote_code and not is_trusted_for_remote_code(model_name):
            raise ValueError(
                f"Refusing to load {model_name!r} with trust_remote_code=True: "
                "model is not in the OpenMed trusted-remote-code allowlist. "
                "Trusted repos are listed in "
                "openmed.torch.privacy_filter.TRUSTED_REMOTE_CODE_MODELS; "
                f"to extend, set {_ALLOWLIST_ENV_VAR} to a comma-separated "
                "list of repo IDs you control."
            )
        try:
            import torch
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:  # pragma: no cover - hf is optional extra
            raise ImportError(
                "PrivacyFilterTorchPipeline requires `transformers` and "
                "`torch`. Install with: pip install openmed[hf]"
            ) from exc

        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy

        resolved_device = device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        load_kwargs: Dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if dtype is not None:
            torch_dtype = getattr(torch, dtype, None)
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, **load_kwargs,
        )
        self.model.to(resolved_device)
        self.model.eval()

        # HF pipeline accepts a string device for "cpu"/"cuda"/"mps"; for
        # "cpu" it expects -1 in older versions, which still works.
        pipeline_device = -1 if resolved_device == "cpu" else resolved_device

        self._pipeline = pipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=pipeline_device,
        )
        self.device = resolved_device

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]] | List[List[Dict[str, Any]]]:
        """Run inference and emit MLX-compatible entity dicts."""
        call_kwargs: Dict[str, Any] = {}
        if batch_size is not None:
            call_kwargs["batch_size"] = batch_size
        if num_workers is not None:
            call_kwargs["num_workers"] = num_workers

        if isinstance(text, (list, tuple)):
            texts = list(text)
            non_empty = [item for item in texts if item and item.strip()]
            if not non_empty:
                return [[] for _ in texts]

            raw_batch = self._pipeline(non_empty, **call_kwargs)
            normalized_batch = self._normalize_batch_output(raw_batch, len(non_empty))
            normalized_iter = iter(
                [
                    [self._normalize_entity(entity, source) for entity in raw]
                    for raw, source in zip(normalized_batch, non_empty)
                ]
            )
            return [
                next(normalized_iter) if item and item.strip() else []
                for item in texts
            ]

        if not text or not text.strip():
            return []

        raw = self._pipeline(text, **call_kwargs)
        return [self._normalize_entity(item, text) for item in raw]

    @staticmethod
    def _normalize_batch_output(
        raw_batch: Any,
        expected_count: int,
    ) -> List[List[Dict[str, Any]]]:
        """Normalize HuggingFace output for a batch of input texts."""
        if expected_count == 1:
            if raw_batch == []:
                return [[]]
            if isinstance(raw_batch, list) and raw_batch and isinstance(raw_batch[0], dict):
                return [raw_batch]
            if isinstance(raw_batch, list) and len(raw_batch) == 1:
                return [raw_batch[0] or []]

        if isinstance(raw_batch, list) and len(raw_batch) == expected_count:
            return [item or [] for item in raw_batch]

        raise ValueError(
            "Privacy-filter batch output length did not match input length "
            f"({expected_count})"
        )

    @staticmethod
    def _normalize_entity(item: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Refine spans, coerce types, and ensure the schema matches MLX."""
        # HF pipeline emits either ``entity_group`` (when aggregating) or
        # ``entity`` (when not). We always normalise to ``entity_group``.
        label = item.get("entity_group") or item.get("entity") or ""
        start = int(item.get("start", 0))
        end = int(item.get("end", 0))
        score = float(item.get("score", 0.0))

        start, end = trim_span_whitespace(start, end, text)
        if label:
            start, end = refine_privacy_filter_span(label, start, end, text)
        if end <= start:
            return {
                "entity_group": label,
                "score": score,
                "word": "",
                "start": start,
                "end": end,
            }
        return {
            "entity_group": label,
            "score": score,
            "word": text[start:end],
            "start": start,
            "end": end,
        }


__all__ = [
    "PrivacyFilterTorchPipeline",
    "TRUSTED_REMOTE_CODE_MODELS",
    "is_trusted_for_remote_code",
]
