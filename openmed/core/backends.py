"""Inference backend abstraction for OpenMed.

Provides a protocol for pluggable inference backends (HuggingFace, MLX, etc.)
and auto-detection logic for selecting the best available backend on the
current platform.
"""

from __future__ import annotations

import logging
import platform
import sys
import warnings
from importlib.util import find_spec
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

from .offline import configure_offline_mode, is_local_only

logger = logging.getLogger(__name__)
_warned_substitutions: set[str] = set()
_MISSING_MODULE = object()


def _module_available(module_name: str) -> bool:
    """Return whether a module is loaded or importable without importing it."""
    loaded = sys.modules.get(module_name, _MISSING_MODULE)
    if loaded is not _MISSING_MODULE:
        return loaded is not None
    try:
        return find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


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

        return HF_AVAILABLE and _module_available("torch")

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


class OnnxTokenClassificationPipeline:
    """Transform :class:`OnnxModel` output into the standard pipeline schema."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.tokenizer = model.tokenizer
        self.variant = model.variant

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        """Run one or more inputs without importing a Torch runtime."""
        threshold = float(kwargs.pop("threshold", 0.0))
        max_length = kwargs.pop("max_length", None)
        kwargs.pop("batch_size", None)
        kwargs.pop("num_workers", None)
        if kwargs:
            logger.debug(
                "Ignoring unsupported ONNX pipeline options: %s", sorted(kwargs)
            )

        single = isinstance(inputs, str)
        texts = [inputs] if single else list(inputs)
        predictions = [
            [
                {
                    "entity_group": entity.label,
                    "score": entity.score,
                    "word": entity.text,
                    "start": entity.start,
                    "end": entity.end,
                }
                for entity in self.model.predict(
                    text,
                    threshold=threshold,
                    max_length=max_length,
                )
            ]
            for text in texts
        ]
        return predictions[0] if single else predictions


class OnnxBackend:
    """CPU-only ONNX Runtime backend with an INT8-capable model loader."""

    _RUNTIME_MODULES = ("huggingface_hub", "numpy", "onnxruntime", "tokenizers")

    def __init__(self, config: Any = None) -> None:
        self._config = config

    def is_available(self) -> bool:
        return all(_module_available(name) for name in self._RUNTIME_MODULES)

    def create_pipeline(
        self,
        model_name: str,
        task: str = "token-classification",
        aggregation_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Callable:
        """Create a CPU ONNX token-classification pipeline."""
        del aggregation_strategy
        if task not in {"ner", "token-classification"}:
            raise ValueError(
                "The ONNX backend supports only token-classification tasks"
            )

        from openmed.onnx.inference import load_onnx_model

        kwargs.pop("use_fast_tokenizer", None)
        variant = getattr(self._config, "onnx_variant", "auto")
        threads = getattr(self._config, "onnx_intra_op_num_threads", None)
        session_options = None
        if threads is not None:
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = int(threads)
            session_options.inter_op_num_threads = 1

        model = load_onnx_model(
            model_name,
            variant=variant,
            revision=getattr(self._config, "pii_model_revision", None) or "main",
            cache_dir=getattr(self._config, "cache_dir", None),
            token=getattr(self._config, "hf_token", None),
            local_files_only=is_local_only(self._config),
            providers=("CPUExecutionProvider",),
            session_options=session_options,
        )
        if variant == "int8" and model.variant != "int8":
            raise RuntimeError(
                f"Low-resource profile requires model_int8.onnx; got {model.variant!r}"
            )
        return OnnxTokenClassificationPipeline(model)


# -- Backend registry and auto-detection ------------------------------------

_BACKENDS: Dict[str, type] = {
    "hf": HuggingFaceBackend,
    "mlx": MLXBackend,
    "onnx": OnnxBackend,
}


def get_backend(
    name: Optional[str] = None,
    config: Any = None,
) -> InferenceBackend:
    """Return the requested backend, or auto-detect the best available one.

    Args:
        name: ``"hf"``, ``"mlx"``, ``"onnx"``, or ``None`` for auto-detect.
        config: OpenMedConfig to pass to the backend.

    Auto-detection order:
        1. MLX — if on Apple Silicon *and* ``mlx`` is importable.
        2. HuggingFace — default fallback.
        3. ONNX Runtime — CPU fallback when only ``onnx-runtime`` is installed.
    """
    configured_name = getattr(config, "backend", None)
    if name is None and configured_name is not None:
        name = configured_name

    if name is not None:
        if name not in _BACKENDS:
            raise ValueError(
                f"Unknown backend {name!r}. Available: {sorted(_BACKENDS)}"
            )
        backend = _BACKENDS[name](config)
        if not backend.is_available():
            if name == "onnx":
                raise RuntimeError(
                    "Backend 'onnx' is not available. Install the CPU runtime "
                    "with: pip install 'openmed[onnx-runtime]'. The low_resource "
                    "profile does not fall back to a Torch backend."
                )
            raise RuntimeError(
                f"Backend {name!r} is not available. Install its dependencies first."
            )
        return backend

    # Auto-detect: prefer MLX on Apple Silicon
    for candidate_name in ("mlx", "hf", "onnx"):
        candidate = _BACKENDS[candidate_name](config)
        if candidate.is_available():
            logger.info("Auto-selected inference backend: %s", candidate_name)
            return candidate

    raise RuntimeError(
        "No inference backend available. "
        "Install at least one: pip install openmed[hf], openmed[mlx], "
        "or openmed[onnx-runtime]"
    )


# -- Privacy-filter routing ------------------------------------------------

# Default Torch fallback for the original OpenAI Privacy Filter MLX artifacts
# (``OpenMed/privacy-filter-mlx``, ``OpenMed/privacy-filter-mlx-8bit``). When
# a user passes one of these IDs on a non-Apple-Silicon host we silently fall
# back to the upstream PyTorch model and emit a one-time warning so they
# understand the substitution.
PRIVACY_FILTER_TORCH_FALLBACK = "openai/privacy-filter"


# Family-aware Torch fallbacks. Order matters: the first matching marker
# wins. Add new privacy-filter families here as they're introduced so an
# MLX-only request from Linux falls back to the same family's PyTorch model
# (not the unrelated default).
_TORCH_FALLBACK_BY_FAMILY: tuple[tuple[str, str], ...] = (
    ("multilingual", "OpenMed/privacy-filter-multilingual"),
    ("nemotron", "OpenMed/privacy-filter-nemotron"),
)


def _torch_fallback_for(model_name: str) -> str:
    """Pick the Torch fallback that matches ``model_name``'s family.

    Substring-based to keep adding new families a one-line change.
    """
    name_lc = (model_name or "").lower()
    for marker, repo in _TORCH_FALLBACK_BY_FAMILY:
        if marker in name_lc:
            return repo
    return PRIVACY_FILTER_TORCH_FALLBACK


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
        target = _torch_fallback_for(model_name)
        if model_name not in _warned_substitutions:
            warnings.warn(
                f"OpenMed: {model_name!r} is an MLX-only artifact and "
                f"cannot run on this host. Substituting "
                f"{target!r} via Transformers. "
                "To silence, request the PyTorch model directly.",
                UserWarning,
                stacklevel=3,
            )
            _warned_substitutions.add(model_name)
        return target
    return model_name


def create_privacy_filter_pipeline(model_name: str, config: Any = None) -> Callable:
    """Build a privacy-filter pipeline appropriate for the host.

    Returns a callable ``pipeline(text) -> List[Dict]`` whose output
    schema matches the HuggingFace ``token-classification`` pipeline so
    downstream OpenMed code is backend-agnostic.
    """
    configure_offline_mode(config)
    backend = select_privacy_filter_backend(model_name)
    actual_model = resolve_privacy_filter_model(model_name, backend)

    if backend == "mlx":
        from openmed.mlx.inference import create_mlx_pipeline

        if config is None:
            return create_mlx_pipeline(actual_model)
        return create_mlx_pipeline(actual_model, config=config)

    from openmed.torch.privacy_filter import (
        PrivacyFilterTorchPipeline,
        is_trusted_for_remote_code,
    )

    # ``trust_remote_code=True`` is required to import the custom modeling
    # code shipped inside first-party privacy-filter repos. Only enable it
    # when the resolved model is on the allowlist; the pipeline itself
    # double-checks and raises ``ValueError`` if the gate is bypassed.
    pipeline_kwargs: Dict[str, Any] = {
        "trust_remote_code": is_trusted_for_remote_code(actual_model),
    }
    if is_local_only(config):
        pipeline_kwargs["local_files_only"] = True

    return PrivacyFilterTorchPipeline(actual_model, **pipeline_kwargs)
