"""Model loading functionality for OpenMed."""

import gc
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoConfig,
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    HF_AVAILABLE = True
except (ImportError, OSError) as e:
    HF_AVAILABLE = False
    logger.warning(
        "HuggingFace transformers could not be imported (%s). "
        "Install with: pip install transformers",
        e,
    )

    AutoTokenizer = None  # type: ignore[assignment]
    AutoModelForTokenClassification = None  # type: ignore[assignment]
    AutoConfig = None  # type: ignore[assignment]
    pipeline = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .config import OpenMedConfig

from .config import get_config
from .model_registry import (
    OPENMED_MODELS,
    get_all_models,
    get_model_info,
    get_model_suggestions,
    get_models_by_category,
)
from .model_registry import (
    ModelInfo as RegistryModelInfo,
)


class ModelLoader:
    """Handles loading and managing OpenMed models from HuggingFace Hub."""

    def __init__(self, config: Optional["OpenMedConfig"] = None):
        """Initialize the model loader.

        Args:
            config: OpenMed configuration. If None, uses global config.
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers is required. "
                "Install with: pip install transformers"
            )

        self.config = config or get_config()
        self._models = {}  # Cache for loaded models
        self._tokenizers = {}  # Cache for loaded tokenizers
        self._pipelines = {}  # Cache for created pipelines

    def list_available_models(
        self,
        include_registry: bool = True,
        include_remote: bool = True,
    ) -> List[str]:
        """List all available TokenClassification models from OpenMed org.

        Args:
            include_registry: Whether to include models from local registry
            include_remote: Whether to query Hugging Face Hub for additional models

        Returns:
            List of model names available for loading.
        """
        models = []

        if include_registry:
            registry_models = [info.model_id for info in get_all_models().values()]
            models.extend(registry_models)

        if include_remote:
            logger.debug(
                "include_remote is retained for compatibility; model discovery "
                "uses the committed manifest snapshot."
            )

        return sorted(set(models))

    def load_model(
        self, model_name: str, force_reload: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Load a TokenClassification model and tokenizer.

        Args:
            model_name: Name of the model to load. Can be just the model name
                       (will prepend org) or full model path.
            force_reload: Whether to force reload even if cached.
            **kwargs: Additional arguments to pass to model loading.

        Returns:
            Dictionary containing 'model', 'tokenizer', and 'config'.

        Raises:
            ValueError: If model is not found or not a TokenClassification model.
        """
        full_model_name = self._resolve_model_name(model_name)

        # Check cache
        if not force_reload and full_model_name in self._models:
            logger.info(f"Using cached model: {full_model_name}")
            return {
                "model": self._models[full_model_name],
                "tokenizer": self._tokenizers[full_model_name],
                "config": self._models[full_model_name].config,
            }

        try:
            logger.info(f"Loading model: {full_model_name}")

            auth_kwargs = self._hub_auth_kwargs()
            local_loading_kwargs = self._local_loading_kwargs(full_model_name, kwargs)
            pretrained_kwargs = {**auth_kwargs, **local_loading_kwargs}
            load_kwargs = dict(kwargs)
            device_preference = load_kwargs.pop("device", None)
            resolved_device = self._resolve_torch_device(device_preference)

            # Load config first to verify it's a token classification model
            config = AutoConfig.from_pretrained(
                full_model_name,
                cache_dir=self.config.cache_dir,
                **pretrained_kwargs,
            )

            # Verify model type
            if (
                not hasattr(config, "num_labels")
                or config.problem_type != "token_classification"
            ):
                # Try to infer from architecture
                if not any(
                    arch in config.architectures[0].lower()
                    for arch in ["tokenclassification", "ner", "pos"]
                ):
                    logger.warning(
                        f"Model {full_model_name} may not be a TokenClassification model"
                    )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                full_model_name,
                cache_dir=self.config.cache_dir,
                **pretrained_kwargs,
            )

            # Note: medical tokenization (if enabled) is applied at the output remapping layer,
            # not by modifying the model tokenizer/vocabulary.

            # Load model
            model_kwargs = {**pretrained_kwargs, **load_kwargs}
            model = AutoModelForTokenClassification.from_pretrained(
                full_model_name,
                cache_dir=self.config.cache_dir,
                **model_kwargs,
            )
            if hasattr(model, "to"):
                model.to(resolved_device)

            # Cache the loaded model and tokenizer
            self._models[full_model_name] = model
            self._tokenizers[full_model_name] = tokenizer

            logger.info(f"Successfully loaded model: {full_model_name}")

            return {
                "model": model,
                "tokenizer": tokenizer,
                "config": config,
            }

        except Exception as e:
            logger.error(f"Failed to load model {full_model_name}: {e}")
            raise ValueError(f"Could not load model {full_model_name}: {e}")

    def create_pipeline(
        self,
        model_name: str,
        task: str = "token-classification",
        aggregation_strategy: Optional[str] = None,
        use_fast_tokenizer: bool = True,
        **kwargs,
    ):
        """Create an inference pipeline for the configured backend."""
        from openmed.core.backends import HuggingFaceBackend, get_backend

        backend = get_backend(getattr(self.config, "backend", None), config=self.config)
        if isinstance(backend, HuggingFaceBackend):
            return self._create_hf_pipeline(
                model_name,
                task=task,
                aggregation_strategy=aggregation_strategy,
                use_fast_tokenizer=use_fast_tokenizer,
                **kwargs,
            )

        try:
            return backend.create_pipeline(
                model_name,
                task=task,
                aggregation_strategy=aggregation_strategy,
                use_fast_tokenizer=use_fast_tokenizer,
                **kwargs,
            )
        except Exception:
            if getattr(self.config, "backend", None) is not None:
                raise

            logger.warning(
                "Auto-selected backend %s failed for %s; falling back to HuggingFace.",
                type(backend).__name__,
                model_name,
                exc_info=True,
            )
            return self._create_hf_pipeline(
                model_name,
                task=task,
                aggregation_strategy=aggregation_strategy,
                use_fast_tokenizer=use_fast_tokenizer,
                **kwargs,
            )

    def _create_hf_pipeline(
        self,
        model_name: str,
        task: str = "token-classification",
        aggregation_strategy: Optional[str] = None,
        use_fast_tokenizer: bool = True,
        **kwargs,
    ):
        """Create a HuggingFace pipeline for the model.

        Args:
            model_name: Name of the model to use.
            task: Task type (default: "token-classification").
            aggregation_strategy: How to aggregate tokens (None for raw tokens,
                                "simple", "first", "average", "max").
            use_fast_tokenizer: Whether to use fast tokenizer.
            **kwargs: Additional arguments for pipeline creation.

        Returns:
            HuggingFace pipeline object.
        """
        # Resolve model name if it's a registry key
        full_model_name = self._resolve_model_name(model_name)
        cache_key = self._build_pipeline_cache_key(
            full_model_name,
            task=task,
            aggregation_strategy=aggregation_strategy,
            use_fast_tokenizer=use_fast_tokenizer,
            kwargs=kwargs,
        )

        if cache_key in self._pipelines:
            logger.info(f"Using cached pipeline for {full_model_name}")
            return self._pipelines[cache_key]

        try:
            # Create pipeline directly with model name for better caching
            pipeline_device = kwargs.get("device", self._get_device_id())
            pipeline_kwargs = {
                "model": full_model_name,
                "aggregation_strategy": aggregation_strategy,
                "device": pipeline_device,
                "use_fast": use_fast_tokenizer,
            }
            pipeline_kwargs.update(self._hub_auth_kwargs())
            pipeline_kwargs.update(self._local_loading_kwargs(full_model_name, kwargs))
            pipeline_kwargs.update(kwargs)

            ner_pipeline = pipeline(task, **pipeline_kwargs)
            self._pipelines[cache_key] = ner_pipeline

            logger.info(f"Created pipeline for {full_model_name}")
            return ner_pipeline

        except Exception as e:
            logger.error(f"Failed to create pipeline for {full_model_name}: {e}")
            # Fall back to loading model components manually
            model_data = self.load_model(model_name)

            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("device", None)
            if aggregation_strategy is not None:
                fallback_kwargs["aggregation_strategy"] = aggregation_strategy

            ner_pipeline = pipeline(
                task,
                model=model_data["model"],
                tokenizer=model_data["tokenizer"],
                device=kwargs.get("device", self._get_device_id()),
                **fallback_kwargs,
            )
            self._pipelines[cache_key] = ner_pipeline
            return ner_pipeline

    def resolve_model_name(self, model_name: str) -> str:
        """Resolve a registry alias, local path, or full model id."""
        return self._resolve_model_name(model_name)

    def loaded_models(self) -> Dict[str, Dict[str, int]]:
        """Return cache counts grouped by resolved model name."""
        model_names = set(self._models) | set(self._tokenizers)
        model_names.update(key[0] for key in self._pipelines if key)

        loaded = {}
        for model_name in sorted(model_names):
            loaded[model_name] = {
                "models": int(model_name in self._models),
                "tokenizers": int(model_name in self._tokenizers),
                "pipelines": sum(
                    1 for key in self._pipelines if key and key[0] == model_name
                ),
            }
        return loaded

    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Release cached model, tokenizer, and pipelines for one model."""
        full_model_name = self._resolve_model_name(model_name)
        pipeline_keys = [
            key for key in self._pipelines if key and key[0] == full_model_name
        ]

        for key in pipeline_keys:
            self._pipelines.pop(key, None)

        removed_models = int(self._models.pop(full_model_name, None) is not None)
        removed_tokenizers = int(
            self._tokenizers.pop(full_model_name, None) is not None
        )
        released = {
            "model_name": full_model_name,
            "models": removed_models,
            "tokenizers": removed_tokenizers,
            "pipelines": len(pipeline_keys),
        }
        if any(released[name] for name in ("models", "tokenizers", "pipelines")):
            self._release_cached_memory()
        return released

    def unload_all_models(self) -> Dict[str, Any]:
        """Release all cached models, tokenizers, and pipelines."""
        released = {
            "models": len(self._models),
            "tokenizers": len(self._tokenizers),
            "pipelines": len(self._pipelines),
        }
        self._models.clear()
        self._tokenizers.clear()
        self._pipelines.clear()
        if any(released.values()):
            self._release_cached_memory()
        return released

    def get_max_sequence_length(
        self,
        model_name: str,
        *,
        tokenizer: Optional[Any] = None,
    ) -> Optional[int]:
        """Infer the maximum supported sequence length for a model/tokenizer."""
        if not HF_AVAILABLE:
            return None

        from ..processing.tokenization import infer_tokenizer_max_length

        full_model_name = self._resolve_model_name(model_name)
        auth_kwargs = self._hub_auth_kwargs()
        local_loading_kwargs = self._local_loading_kwargs(full_model_name)
        pretrained_kwargs = {**auth_kwargs, **local_loading_kwargs}

        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    full_model_name,
                    cache_dir=self.config.cache_dir,
                    use_fast=True,
                    **pretrained_kwargs,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to load tokenizer for %s when inferring max length: %s",
                    full_model_name,
                    exc,
                )
                tokenizer = None

        if tokenizer is not None:
            inferred = infer_tokenizer_max_length(tokenizer)
            if inferred is not None:
                return inferred

        try:
            config = AutoConfig.from_pretrained(
                full_model_name,
                cache_dir=self.config.cache_dir,
                **pretrained_kwargs,
            )
            for attr in ("max_position_embeddings", "n_positions", "seq_length"):
                value = getattr(config, attr, None)
                if isinstance(value, int) and 0 < value < 1_000_000:
                    return value
        except Exception as exc:
            logger.debug(
                "Failed to load config for %s when inferring max length: %s",
                full_model_name,
                exc,
            )

        return None

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model name from registry key or return full model name."""
        local_path = self._as_existing_local_path(model_name)
        if local_path is not None:
            return str(local_path)

        # Check if it's a registry key
        registry_info = get_model_info(model_name)
        if registry_info:
            return registry_info.model_id

        # Check if it needs org prefix
        if "/" not in model_name:
            return f"{self.config.default_org}/{model_name}"

        return model_name

    def _get_device_id(self) -> Union[int, str]:
        """Get device ID for pipeline."""
        resolved_device = self._resolve_torch_device()
        if resolved_device == "cpu":
            return -1
        if resolved_device == "cuda":
            return 0
        return resolved_device

    def _resolve_torch_device(self, prefer: Optional[str] = None) -> str:
        """Resolve and prepare the torch device for this loader."""
        from openmed.torch.device import apply_mps_tuning, resolve_torch_device

        device_preference = prefer if prefer is not None else self.config.device
        resolved_device = resolve_torch_device(device_preference)
        if resolved_device.startswith("mps"):
            apply_mps_tuning()
        return resolved_device

    def get_registry_info(self, model_key: str) -> Optional[RegistryModelInfo]:
        """Get information from model registry."""
        return get_model_info(model_key)

    def get_model_suggestions(
        self, text: str
    ) -> List[Tuple[str, RegistryModelInfo, str]]:
        """Get model suggestions based on text content."""
        return get_model_suggestions(text)

    def list_models_by_category(self, category: str) -> List[RegistryModelInfo]:
        """List models by category."""
        return get_models_by_category(category)

    def get_model_info(self, model_name: str):
        """Get detailed information about a model.

        Args:
            model_name: Name of the model.

        Returns:
            ModelInfo object or None if not found.
        """
        full_model_name = self._resolve_model_name(model_name)
        return get_model_info(model_name) or get_model_info(full_model_name)

    def _hub_auth_kwargs(self) -> Dict[str, Any]:
        """Return authentication kwargs for Hugging Face Hub calls."""
        if getattr(self.config, "hf_token", None):
            return {"token": self.config.hf_token}
        return {}

    def _as_existing_local_path(self, model_name: str) -> Optional[Path]:
        """Return a filesystem path when ``model_name`` points to local files."""
        try:
            path = Path(model_name).expanduser()
        except (TypeError, ValueError, OSError):
            return None

        try:
            if path.exists():
                return path
        except OSError:
            return None
        return None

    def _local_loading_kwargs(
        self,
        model_name: str,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Force local-only Hub loading for filesystem-backed model names."""
        if kwargs and "local_files_only" in kwargs:
            return {"local_files_only": kwargs["local_files_only"]}
        if self._as_existing_local_path(model_name) is not None:
            return {"local_files_only": True}
        return {}

    def _build_pipeline_cache_key(
        self,
        full_model_name: str,
        *,
        task: str,
        aggregation_strategy: Optional[str],
        use_fast_tokenizer: bool,
        kwargs: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """Return a hashable cache key for pipeline creation."""
        return (
            full_model_name,
            task,
            aggregation_strategy,
            use_fast_tokenizer,
            self._get_device_id(),
            self._freeze_cache_value(kwargs),
        )

    def _freeze_cache_value(self, value: Any) -> Any:
        """Convert nested kwargs into a hashable representation."""
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    (str(key), self._freeze_cache_value(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_cache_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze_cache_value(item) for item in value))
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return repr(value)

    def _release_cached_memory(self) -> None:
        """Nudge Python and torch runtimes after cache references are dropped."""
        gc.collect()
        try:
            import torch
        except Exception:
            return

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_available", None)):
            try:
                if cuda.is_available():
                    cuda.empty_cache()
            except Exception:
                logger.debug(
                    "Failed to clear CUDA cache after unloading model",
                    exc_info=True,
                )

        mps = getattr(torch, "mps", None)
        if mps is not None and callable(getattr(mps, "empty_cache", None)):
            try:
                mps.empty_cache()
            except Exception:
                logger.debug(
                    "Failed to clear MPS cache after unloading model", exc_info=True
                )


# Convenience function for quick model loading
def load_model(
    model_name: str, config: Optional["OpenMedConfig"] = None, **kwargs
) -> Dict[str, Any]:
    """Convenience function to quickly load an OpenMed model.

    Args:
        model_name: Name of the model to load.
        config: Optional configuration object.
        **kwargs: Additional arguments for model loading.

    Returns:
        Dictionary containing 'model', 'tokenizer', and 'config'.
    """
    loader = ModelLoader(config)
    return loader.load_model(model_name, **kwargs)
