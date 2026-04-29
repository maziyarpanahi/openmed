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
from typing import Any, Dict, List, Optional

from openmed.core.decoding import refine_privacy_filter_span, trim_span_whitespace

logger = logging.getLogger(__name__)


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
        trust_remote_code: The OpenAI Privacy Filter family ships with
            custom modeling code (``modeling_openai_privacy_filter.py``) in
            the model repo, which transformers needs permission to import.
            Defaults to ``True`` because this pipeline is *specifically*
            for that family — set to ``False`` to opt out (and accept that
            loading will fail without an upstream registration).
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
        trust_remote_code: bool = True,
    ) -> None:
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

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        """Run inference and emit MLX-compatible entity dicts."""
        if not text or not text.strip():
            return []

        raw = self._pipeline(text)
        return [self._normalize_entity(item, text) for item in raw]

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


__all__ = ["PrivacyFilterTorchPipeline"]
