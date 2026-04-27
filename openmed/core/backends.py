"""Inference backend abstraction for OpenMed.

Provides a protocol for pluggable inference backends (HuggingFace, MLX, etc.)
and auto-detection logic for selecting the best available backend on the
current platform.
"""

from __future__ import annotations

import logging
import platform
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)
_warned_substitutions: set[str] = set()


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
        from openmed.core.models import HF_AVAILABLE
        return HF_AVAILABLE

    def create_pipeline(
        self,
        model_name: str,
        task: str = "token-classification",
        aggregation_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Callable:
        from openmed.core.models import ModelLoader
        loader = ModelLoader(self._config)
        return loader._create_hf_pipeline(
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


# -- Privacy-filter routing ------------------------------------------------

# The MLX-only artifacts (``OpenMed/privacy-filter-mlx``,
# ``OpenMed/privacy-filter-mlx-8bit``) cannot run on non-Apple-Silicon
# machines. When a user passes one of these IDs from Linux/Windows we
# silently fall back to the upstream PyTorch model and emit a one-time
# warning so they understand the substitution.
PRIVACY_FILTER_TORCH_FALLBACK = "openai/privacy-filter"


def select_privacy_filter_backend(
    model_name: str,
) -> Literal["mlx", "torch"]:
    """Pick MLX or Torch for a privacy-filter-family ``model_name``.

    Returns ``"mlx"`` only when (a) MLX is importable on the current
    machine, and (b) the requested model is itself an MLX artifact
    (its name contains ``"mlx"`` or its on-disk metadata identifies as
    one). Otherwise returns ``"torch"`` — including when an MLX-only
    model name is requested on a non-Mac host, in which case the caller
    should substitute :data:`PRIVACY_FILTER_TORCH_FALLBACK` for the
    actual download.
    """
    name_lc = (model_name or "").lower()
    is_mlx_artifact = "mlx" in name_lc

    if not is_mlx_artifact:
        # Some artifacts identify as MLX only via their on-disk metadata.
        try:
            from .pii import _is_privacy_filter_artifact_path
            is_mlx_artifact = _is_privacy_filter_artifact_path(model_name)
        except ImportError:  # pragma: no cover
            is_mlx_artifact = False

    if is_mlx_artifact and MLXBackend().is_available():
        return "mlx"
    return "torch"


def resolve_privacy_filter_model(
    model_name: str,
    backend: Literal["mlx", "torch"],
) -> str:
    """Map a privacy-filter ``model_name`` to the actual artifact for ``backend``.

    On Linux/Windows where MLX is unavailable, an ``OpenMed/privacy-filter-mlx*``
    request needs to download the upstream PyTorch model instead. This
    helper performs that substitution and emits a one-time UserWarning
    so the user understands the swap.
    """
    if backend == "mlx":
        return model_name

    if "mlx" in (model_name or "").lower():
        if model_name not in _warned_substitutions:
            warnings.warn(
                f"OpenMed: {model_name!r} is an MLX-only artifact and "
                f"cannot run on this host. Substituting "
                f"{PRIVACY_FILTER_TORCH_FALLBACK!r} via Transformers. "
                "To silence, request the PyTorch model directly.",
                UserWarning,
                stacklevel=3,
            )
            _warned_substitutions.add(model_name)
        return PRIVACY_FILTER_TORCH_FALLBACK
    return model_name


def create_privacy_filter_pipeline(model_name: str) -> Callable:
    """Build a privacy-filter pipeline appropriate for the host.

    Returns a callable ``pipeline(text) -> List[Dict]`` whose output
    schema matches the HuggingFace ``token-classification`` pipeline so
    downstream OpenMed code is backend-agnostic.
    """
    backend = select_privacy_filter_backend(model_name)
    actual_model = resolve_privacy_filter_model(model_name, backend)

    if backend == "mlx":
        from openmed.mlx.inference import create_mlx_pipeline
        return create_mlx_pipeline(actual_model)

    from openmed.torch.privacy_filter import PrivacyFilterTorchPipeline
    return PrivacyFilterTorchPipeline(actual_model)
