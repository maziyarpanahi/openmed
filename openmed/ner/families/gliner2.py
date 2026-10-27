"""Local GLiNER2 / GLiNER2.5 loading and native extraction adapters."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

from openmed.core.capabilities import is_backend_available

from ..exceptions import MissingDependencyError
from .gliner import _resolve_native_device

_PRIMARY_IMPORT = "gliner2"
_GLINER2_HINT = (
    "Install with `pip install openmed[gliner2]` "
    "(requires gliner2[local]>=2.1.0 for local span and boundary checkpoints)."
)


def is_gliner2_available(force_refresh: bool = False) -> bool:
    """Return whether the GLiNER2 local inference stack is installed."""

    return is_backend_available("gliner2")


def ensure_gliner2_available() -> None:
    """Require GLiNER2's local runtime, including its optional ML stack."""

    if is_gliner2_available(force_refresh=False):
        return

    raise MissingDependencyError(
        dependency="gliner2[local]>=2.1.0", instruction=_GLINER2_HINT
    )


@dataclass
class GLiNER2Handle:
    """Adapt native GLiNER2 dictionaries to the common span-list contract."""

    model_id: str
    model: Any

    def predict_entities(
        self,
        text: str,
        labels: Optional[Sequence[str]] = None,
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Extract all occurrences with native offsets and confidence scores."""

        result = self.model.extract_entities(
            text,
            list(labels) if labels else [],
            threshold=threshold,
            include_spans=True,
            include_confidence=True,
            overlap_policy="disallow" if flat_ner else "allow",
            **kwargs,
        )
        if not isinstance(result, dict) or not isinstance(result.get("entities"), dict):
            raise ValueError("GLiNER2 returned an invalid entities mapping")
        entities = []
        for label, spans in result["entities"].items():
            if not isinstance(spans, list):
                raise ValueError("GLiNER2 returned an invalid span list")
            for span in spans:
                if not isinstance(span, dict) or not {
                    "text",
                    "start",
                    "end",
                    "confidence",
                }.issubset(span):
                    raise ValueError("GLiNER2 must return spans with confidence")
                entities.append({**span, "label": label, "score": span["confidence"]})
        return sorted(entities, key=lambda item: (item["start"], item["end"]))

    def predict_relations(
        self,
        text: str,
        labels: Optional[Sequence[str]] = None,
        *,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Any:
        """Optional relation extraction hook if the model provides it."""
        ensure_gliner2_available()
        if hasattr(self.model, "extract_relations"):
            predict_kwargs = dict(kwargs)
            predict_kwargs.setdefault("threshold", threshold)
            return self.model.extract_relations(
                text, list(labels or []), **predict_kwargs
            )
        raise AttributeError("GLiNER2 checkpoint does not support relation extraction")


def load_gliner2_handle(
    model_id: str,
    *,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    device: Optional[str] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> GLiNER2Handle:
    """Load a local GLiNER2 extractor with cache, authentication and device options.

    ``AutoExtractor`` selects the span or boundary architecture from checkpoint
    configuration; it never constructs the package's hosted API client.
    """

    ensure_gliner2_available()
    model = _load_model(
        model_id,
        cache_dir or None,
        token or None,
        _resolve_native_device(device),
        local_files_only,
        revision,
    )

    return GLiNER2Handle(model_id=model_id, model=model)


def clear_gliner2_cache() -> None:
    """Clear cached GLiNER v2 model instances."""

    _load_model.cache_clear()  # type: ignore[attr-defined]


@lru_cache(maxsize=4)
def _load_model(
    model_id: str,
    cache_dir: Optional[str],
    token: Optional[str],
    device: Optional[str],
    local_files_only: bool,
    revision: Optional[str],
) -> Any:
    ensure_gliner2_available()
    module = importlib.import_module(_PRIMARY_IMPORT)

    loader = getattr(module, "AutoExtractor", None)
    if loader is None:
        raise MissingDependencyError(
            dependency="gliner2[local]>=2.1.0",
            instruction=_GLINER2_HINT,
        )

    reference = model_id
    if not Path(reference).is_dir():
        hub = importlib.import_module("huggingface_hub")
        # AutoExtractor 2.1 consumes Hub options while resolving config but
        # does not forward them to the weight loader. Resolve one snapshot so
        # authentication, cache, offline mode and revision bind both stages.
        reference = hub.snapshot_download(
            repo_id=model_id,
            repo_type="model",
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.model",
                "*.txt",
                "*.vocab",
                "*.merges",
            ],
        )
    model = loader.from_pretrained(reference, local_files_only=True)
    if device:
        model = model.to(device)
    model.eval()
    return model


__all__ = [
    "GLiNER2Handle",
    "is_gliner2_available",
    "ensure_gliner2_available",
    "load_gliner2_handle",
    "clear_gliner2_cache",
]
