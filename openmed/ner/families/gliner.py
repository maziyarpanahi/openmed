"""Native GLiNER and GLiFormer zero-shot NER integrations."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from openmed.core.capabilities import is_backend_available

from ..exceptions import MissingDependencyError

_GLINER_HINT = "Install with `pip install openmed[gliner]`."
_PRIMARY_IMPORT = "gliner"
_ANCILLARY_IMPORTS = ("torch", "transformers")

_availability_cache: Optional[bool] = None


def is_gliner_available(force_refresh: bool = False) -> bool:
    """Return True when GLiNER and its ancillary dependencies are present."""

    global _availability_cache
    if force_refresh or _availability_cache is None:
        _availability_cache = is_backend_available("gliner")
    return bool(_availability_cache)


def ensure_gliner_available() -> None:
    """Ensure GLiNER is importable, otherwise raise an informative error."""

    if is_gliner_available(force_refresh=False):  # quick path
        return

    missing = _missing_dependency()
    dependency = missing or _PRIMARY_IMPORT
    raise MissingDependencyError(dependency=dependency, instruction=_GLINER_HINT)


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
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> GLiNERHandle:
    """Load a GLiNER model and wrap it in ``GLiNERHandle``."""

    ensure_gliner_available()
    model = _load_model(
        model_id,
        cache_dir or None,
        token or None,
        device or None,
        local_files_only,
        revision,
    )

    return GLiNERHandle(model_id=model_id, model=model)


def clear_gliner_cache() -> None:
    """Clear cached GLiNER model instances."""

    _load_model.cache_clear()  # type: ignore[attr-defined]


@lru_cache(maxsize=4)
def _load_model(
    model_id: str,
    cache_dir: Optional[str],
    token: Optional[str],
    device: Optional[str] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> Any:
    ensure_gliner_available()
    module = importlib.import_module(_PRIMARY_IMPORT)
    loader = getattr(module, "GLiNER")
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if token:
        kwargs["token"] = token
    if local_files_only:
        kwargs["local_files_only"] = True
    if revision:
        kwargs["revision"] = revision
    model = loader.from_pretrained(model_id, **kwargs)
    # Place each cached instance once; another device must not move live handles.
    if device and hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:  # pragma: no cover - defensive path
            pass
    return model


def is_gliformer_available(force_refresh: bool = False) -> bool:
    """Return whether the optional local GLiFormer stack is installed."""

    return is_backend_available("gliformer")


def ensure_gliformer_available() -> None:
    """Raise an actionable installation error for a missing GLiFormer stack."""

    if not is_gliformer_available():
        raise MissingDependencyError(
            dependency="gliformer>=0.1.3",
            instruction="Install with `pip install openmed[gliformer]`.",
        )


@dataclass
class GLiFormerHandle:
    """Wrap a native GLiFormer model without a Transformers NER pipeline."""

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
        """Predict native half-open character spans using the requested labels."""

        return self.model.predict_entities(
            text, labels or [], threshold=threshold, flat_ner=flat_ner, **kwargs
        )


def load_gliformer_handle(
    model_id: str,
    *,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    device: Optional[str] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> GLiFormerHandle:
    """Load a GLiFormer checkpoint, caching independently for each device.

    Args:
        model_id: Hugging Face identifier or local checkpoint directory.
        cache_dir: Optional Hugging Face cache directory.
        token: Optional Hugging Face authentication token.
        device: Optional PyTorch device; invalid devices fail visibly.
        local_files_only: Restrict loading to local files and cached artifacts.
        revision: Optional checkpoint revision.

    Returns:
        A handle to an evaluation-mode native GLiFormer model.
    """

    ensure_gliformer_available()
    model = _load_gliformer_model(
        model_id,
        cache_dir or None,
        token or None,
        _resolve_native_device(device),
        local_files_only,
        revision,
    )
    return GLiFormerHandle(model_id=model_id, model=model)


def clear_gliformer_cache() -> None:
    """Release cached GLiFormer model instances."""

    _load_gliformer_model.cache_clear()


def _resolve_native_device(device: Optional[str]) -> str:
    from openmed.torch.device import apply_mps_tuning, resolve_torch_device

    resolved = resolve_torch_device(device)
    if resolved.startswith("mps"):
        apply_mps_tuning()
    return resolved


@lru_cache(maxsize=4)
def _load_gliformer_model(
    model_id: str,
    cache_dir: Optional[str],
    token: Optional[str],
    device: Optional[str],
    local_files_only: bool,
    revision: Optional[str],
) -> Any:
    module = importlib.import_module("gliformer")
    model = module.GLiFormer.from_pretrained(
        model_id,
        load_tokenizer=True,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        revision=revision,
    )
    if device:
        model = model.to(device)
    model.eval()
    return model


__all__ = [
    "ensure_gliner_available",
    "is_gliner_available",
    "GLiNERHandle",
    "load_gliner_handle",
    "clear_gliner_cache",
    "GLiFormerHandle",
    "is_gliformer_available",
    "ensure_gliformer_available",
    "load_gliformer_handle",
    "clear_gliformer_cache",
]
