"""Inference backend abstraction for OpenMed.

Provides a protocol for pluggable inference backends (HuggingFace, MLX, etc.)
and auto-detection logic for selecting the best available backend on the
current platform.
"""

from __future__ import annotations

import logging
import platform
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol for inference backends.

    Each backend must be able to report availability and create a callable
    pipeline that accepts text and returns a list of entity dicts in the
    HuggingFace ``token-classification`` output format::

        [{"entity_group": str, "score": float, "word": str,
          "start": int, "end": int}, ...]
    """

    def is_available(self) -> bool:
        """Return True if this backend's dependencies are installed."""
        ...

    def create_pipeline(
        self,
        model_name: str,
        task: str = "token-classification",
        aggregation_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Callable:
        """Create an inference pipeline for *model_name*.

        Returns a callable ``pipeline(text, **kw) -> List[Dict]``.
        """
        ...


class HuggingFaceBackend:
    """Backend using HuggingFace Transformers + PyTorch."""

    def __init__(self, config: Any = None) -> None:
        self._config = config

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def create_pipeline(
        self,
        model_name: str,
        task: str = "token-classification",
        aggregation_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Callable:
        from openmed.core.models import ModelLoader
        loader = ModelLoader(self._config)
        return loader.create_pipeline(
            model_name,
            task=task,
            aggregation_strategy=aggregation_strategy,
            **kwargs,
        )


class MLXBackend:
    """Backend using Apple MLX for hardware-accelerated inference."""

    def __init__(self, config: Any = None) -> None:
        self._config = config

    def is_available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        try:
            import mlx.core  # noqa: F401
            return True
        except ImportError:
            return False

    def create_pipeline(
        self,
        model_name: str,
        task: str = "token-classification",
        aggregation_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Callable:
        from openmed.mlx.inference import create_mlx_pipeline
        return create_mlx_pipeline(
            model_name,
            aggregation_strategy=aggregation_strategy,
            config=self._config,
            **kwargs,
        )


# -- Backend registry and auto-detection ------------------------------------

_BACKENDS: Dict[str, type] = {
    "hf": HuggingFaceBackend,
    "mlx": MLXBackend,
}


def get_backend(
    name: Optional[str] = None,
    config: Any = None,
) -> InferenceBackend:
    """Return the requested backend, or auto-detect the best available one.

    Args:
        name: ``"hf"``, ``"mlx"``, or ``None`` for auto-detect.
        config: OpenMedConfig to pass to the backend.

    Auto-detection order:
        1. MLX — if on Apple Silicon *and* ``mlx`` is importable.
        2. HuggingFace — default fallback.
    """
    if name is not None:
        if name not in _BACKENDS:
            raise ValueError(
                f"Unknown backend {name!r}. Available: {sorted(_BACKENDS)}"
            )
        backend = _BACKENDS[name](config)
        if not backend.is_available():
            raise RuntimeError(
                f"Backend {name!r} is not available. "
                f"Install its dependencies first."
            )
        return backend

    # Auto-detect: prefer MLX on Apple Silicon
    for candidate_name in ("mlx", "hf"):
        candidate = _BACKENDS[candidate_name](config)
        if candidate.is_available():
            logger.info("Auto-selected inference backend: %s", candidate_name)
            return candidate

    raise RuntimeError(
        "No inference backend available. "
        "Install at least one: pip install openmed[hf] or pip install openmed[mlx]"
    )
