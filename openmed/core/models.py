"""Model loading functionality for OpenMed."""

import logging
from typing import Optional, Union, List, Dict, Any, Tuple, TYPE_CHECKING


logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        AutoConfig,
        pipeline,
    )
    from huggingface_hub import list_models, model_info as hf_model_info
    try:
        from huggingface_hub import ModelFilter
    except ImportError:  # pragma: no cover - older hub versions
        ModelFilter = None  # type: ignore[assignment]

    HF_AVAILABLE = True
except ImportError as e:
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

    def list_models(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("HuggingFace Hub is not available")

    def hf_model_info(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("HuggingFace Hub is not available")

    ModelFilter = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .config import OpenMedConfig

from .config import get_config
from .model_registry import (
    OPENMED_MODELS,
    get_model_info,
    get_models_by_category,
    get_model_suggestions,
    get_all_models,
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

        # Add models from local registry
        if include_registry:
            registry_models = [info.model_id for info in get_all_models().values()]
            models.extend(registry_models)

        if include_remote:
            auth_kwargs = self._hub_auth_kwargs()

            try:
                hf_models = list_models(
                    **self._build_model_filter_kwargs(),
                    **auth_kwargs,
                )
                hf_model_ids = [model.modelId for model in hf_models]

                for model_id in hf_model_ids:
                    if model_id not in models:
                        models.append(model_id)

            except Exception as e:
                logger.warning(f"Failed to fetch models from HuggingFace Hub: {e}")
                if not models:  # If no registry models either
                    return []

        return sorted(models)

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

            # Load config first to verify it's a token classification model
            config = AutoConfig.from_pretrained(
                full_model_name,
                cache_dir=self.config.cache_dir,
                **auth_kwargs,
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
                **auth_kwargs,
            )

            # Load model
            model_kwargs = {**auth_kwargs, **kwargs}
            model = AutoModelForTokenClassification.from_pretrained(
                full_model_name,
                cache_dir=self.config.cache_dir,
                **model_kwargs,
            )

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

        try:
            # Create pipeline directly with model name for better caching
            pipeline_kwargs = {
                "model": full_model_name,
                "aggregation_strategy": aggregation_strategy,
                "device": self._get_device_id(),
                "use_fast": use_fast_tokenizer,
            }
            pipeline_kwargs.update(self._hub_auth_kwargs())
            pipeline_kwargs.update(kwargs)

            ner_pipeline = pipeline(task, **pipeline_kwargs)

            logger.info(f"Created pipeline for {full_model_name}")
            return ner_pipeline

        except Exception as e:
            logger.error(f"Failed to create pipeline for {full_model_name}: {e}")
            # Fall back to loading model components manually
            model_data = self.load_model(model_name)

            return pipeline(
                task,
                model=model_data["model"],
                tokenizer=model_data["tokenizer"],
                device=self._get_device_id(),
                **kwargs,
            )

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model name from registry key or return full model name."""
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
        if self.config.device is None:
            return -1  # CPU
        elif self.config.device.lower() == "cpu":
            return -1
        elif self.config.device.lower() in ["cuda", "gpu"]:
            return 0
        else:
            return self.config.device

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

        try:
            return hf_model_info(full_model_name, **self._hub_auth_kwargs())
        except Exception as e:
            logger.error(f"Failed to get model info for {full_model_name}: {e}")
            return None

    def _build_model_filter_kwargs(self) -> Dict[str, Any]:
        """Build keyword arguments for huggingface_hub.list_models."""
        if HF_AVAILABLE and ModelFilter is not None:  # type: ignore[name-defined]
            return {
                "filter": ModelFilter(
                    author=self.config.default_org, task="token-classification"
                )
            }

        # Fallback for older huggingface_hub versions without ModelFilter import.
        logger.debug(
            "huggingface_hub.ModelFilter unavailable; using deprecated author/task arguments."
        )
        return {
            "author": self.config.default_org,
            "task": "token-classification",
        }

    def _hub_auth_kwargs(self) -> Dict[str, Any]:
        """Return authentication kwargs for Hugging Face Hub calls."""
        if getattr(self.config, "hf_token", None):
            return {"token": self.config.hf_token}
        return {}


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
