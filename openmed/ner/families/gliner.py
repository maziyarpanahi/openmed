"""Integration helpers for GLiNER-based zero-shot NER models."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from ..exceptions import MissingDependencyError

_GLINER_HINT = "Run `pip install .[gliner]` to enable GLiNER support."
_PRIMARY_IMPORT = "gliner"
_ANCILLARY_IMPORTS = ("torch", "transformers")

_availability_cache: Optional[bool] = None


def is_gliner_available(force_refresh: bool = False) -> bool:
    """Return True when GLiNER and its ancillary dependencies are present."""

    global _availability_cache
    if force_refresh or _availability_cache is None:
        _availability_cache = _check_dependencies()
    return bool(_availability_cache)


def ensure_gliner_available() -> None:
    """Ensure GLiNER is importable, otherwise raise an informative error."""

    if is_gliner_available(force_refresh=False):  # quick path
        return

    missing = _missing_dependency()
    dependency = missing or _PRIMARY_IMPORT
    raise MissingDependencyError(dependency=dependency, instruction=_GLINER_HINT)


def _check_dependencies() -> bool:
    primary_ok = _safe_import(_PRIMARY_IMPORT)
    if not primary_ok:
        return False

    for module_name in _ANCILLARY_IMPORTS:
        _safe_import(module_name)
    return True


def _safe_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _missing_dependency() -> Optional[str]:
    if not _safe_import(_PRIMARY_IMPORT):
        return _PRIMARY_IMPORT
    for module_name in _ANCILLARY_IMPORTS:
        if not _safe_import(module_name):
            return module_name
    return None


@dataclass
class GLiNERHandle:
    """Lightweight wrapper around a GLiNER model instance."""

    model_id: str
    model: Any

    def predict_entities(
        self,
        text: str,
        labels: Optional[list[str]] = None,
        *,
        flat_ner: bool = True,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Any:
        ensure_gliner_available()
        gliner_labels = labels or []
        predict_kwargs = dict(kwargs)
        if "threshold" not in predict_kwargs:
            predict_kwargs["threshold"] = threshold
        if "flat_ner" not in predict_kwargs:
            predict_kwargs["flat_ner"] = flat_ner
        return self.model.predict_entities(text, gliner_labels, **predict_kwargs)


def load_gliner_handle(
    model_id: str,
    *,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    device: Optional[str] = None,
) -> GLiNERHandle:
    """Load a GLiNER model and wrap it in ``GLiNERHandle``."""

    ensure_gliner_available()
    model = _load_model(model_id, cache_dir or None, token or None)

    if device and hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:  # pragma: no cover - defensive path
            pass

    return GLiNERHandle(model_id=model_id, model=model)


def clear_gliner_cache() -> None:
    """Clear cached GLiNER model instances."""

    _load_model.cache_clear()  # type: ignore[attr-defined]


@lru_cache(maxsize=4)
def _load_model(model_id: str, cache_dir: Optional[str], token: Optional[str]) -> Any:
    ensure_gliner_available()
    module = importlib.import_module(_PRIMARY_IMPORT)
    loader = getattr(module, "GLiNER")
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if token:
        kwargs["token"] = token
    return loader.from_pretrained(model_id, **kwargs)


__all__ = [
    "ensure_gliner_available",
    "is_gliner_available",
    "GLiNERHandle",
    "load_gliner_handle",
    "clear_gliner_cache",
]
