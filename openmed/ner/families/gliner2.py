"""Integration helpers for GLiNER v2 / Fastino-style multi-task models.

This is a draft loader that mirrors the v1 GLiNER helpers but isolates
imports so the base package stays lightweight. It favours graceful
degradation: if dependencies are missing, callers get a clear hint to
install the optional extras.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Sequence

from ..exceptions import MissingDependencyError

_PRIMARY_IMPORT = "gliner"
_ANCILLARY_IMPORTS = ("torch", "transformers")
_GLINER2_HINT = (
    "Run `pip install .[gliner]` to enable GLiNER v2 support "
    "(requires gliner>=0.3.0 with Fastino/GLiNER2 checkpoints)."
)

_availability_cache: Optional[bool] = None


def is_gliner2_available(force_refresh: bool = False) -> bool:
    """Return True when GLiNER>=0.3 and its deps are importable."""

    global _availability_cache
    if force_refresh or _availability_cache is None:
        _availability_cache = _check_dependencies()
    return bool(_availability_cache)


def ensure_gliner2_available() -> None:
    """Ensure GLiNER v2 stack is importable, else raise a helpful error."""

    if is_gliner2_available(force_refresh=False):
        return

    missing = _missing_dependency()
    dependency = missing or _PRIMARY_IMPORT
    raise MissingDependencyError(dependency=dependency, instruction=_GLINER2_HINT)


def _check_dependencies() -> bool:
    primary_ok = _safe_import(_PRIMARY_IMPORT, require_v2=True)
    if not primary_ok:
        return False

    for module_name in _ANCILLARY_IMPORTS:
        _safe_import(module_name)
    return True


def _safe_import(module_name: str, *, require_v2: bool = False) -> bool:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False

    if require_v2 and module_name == _PRIMARY_IMPORT:
        # GLiNER2 landed in 0.3.x; guard on the attribute that's new there.
        version = getattr(module, "__version__", "0.0")
        if version and version.split(".")[0].isdigit():
            major = int(version.split(".")[0])
            minor = int(version.split(".")[1]) if len(version.split(".")) > 1 else 0
            if major == 0 and minor < 3:
                return False
        # Heuristic: v2 exposes GLiNERMultiTask / FastGLiNER classes.
        if not any(
            hasattr(module, attr) for attr in ("GLiNERMultiTask", "FastGLiNER", "GLiNER")
        ):
            return False
    return True


def _missing_dependency() -> Optional[str]:
    if not _safe_import(_PRIMARY_IMPORT, require_v2=True):
        return _PRIMARY_IMPORT
    for module_name in _ANCILLARY_IMPORTS:
        if not _safe_import(module_name):
            return module_name
    return None


@dataclass
class GLiNER2Handle:
    """Wrapper around a GLiNER v2 model instance with multi-task helpers."""

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
        ensure_gliner2_available()
        gliner_labels = list(labels) if labels else []
        predict_kwargs = dict(kwargs)
        predict_kwargs.setdefault("threshold", threshold)
        predict_kwargs.setdefault("flat_ner", flat_ner)
        if hasattr(self.model, "predict_entities"):
            return self.model.predict_entities(text, gliner_labels, **predict_kwargs)
        if hasattr(self.model, "predict"):
            return self.model.predict(text, labels=gliner_labels, **predict_kwargs)
        raise AttributeError("GLiNER v2 model lacks predict_entities/predict")

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
        if hasattr(self.model, "predict_relations"):
            predict_kwargs = dict(kwargs)
            predict_kwargs.setdefault("threshold", threshold)
            return self.model.predict_relations(text, labels=labels, **predict_kwargs)
        raise MissingDependencyError(
            dependency="gliner>=0.3.0",
            instruction="Install a GLiNER2 checkpoint that supports relation extraction.",
        )


def load_gliner2_handle(
    model_id: str,
    *,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    device: Optional[str] = None,
) -> GLiNER2Handle:
    """Load a GLiNER v2 model and wrap it in ``GLiNER2Handle``."""

    ensure_gliner2_available()
    model = _load_model(model_id, cache_dir or None, token or None)

    if device and hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:  # pragma: no cover - defensive path
            pass

    return GLiNER2Handle(model_id=model_id, model=model)


def clear_gliner2_cache() -> None:
    """Clear cached GLiNER v2 model instances."""

    _load_model.cache_clear()  # type: ignore[attr-defined]


@lru_cache(maxsize=4)
def _load_model(model_id: str, cache_dir: Optional[str], token: Optional[str]) -> Any:
    ensure_gliner2_available()
    module = importlib.import_module(_PRIMARY_IMPORT)

    # Prefer explicit v2 classes when present, otherwise fall back to GLiNER.
    loader_name = None
    for candidate in ("GLiNERMultiTask", "FastGLiNER", "GLiNER"):
        if hasattr(module, candidate):
            loader_name = candidate
            break
    if loader_name is None:
        raise MissingDependencyError(
            dependency="gliner>=0.3.0",
            instruction=_GLINER2_HINT,
        )

    loader = getattr(module, loader_name)
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if token:
        kwargs["token"] = token
    return loader.from_pretrained(model_id, **kwargs)


__all__ = [
    "GLiNER2Handle",
    "is_gliner2_available",
    "ensure_gliner2_available",
    "load_gliner2_handle",
    "clear_gliner2_cache",
]
