"""MLX language-model support for OpenMed.

The PII/NER MLX backend uses OpenMed's artifact contract. Causal language
models use the MLX-LM artifact contract so OpenMed can load custom `model_file`
implementations such as Laneformer without bundling model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

LANEFORMER_SOURCE_MODEL = "kogai/laneformer-2b-it"
LANEFORMER_MLX_MODEL = "OpenMed/laneformer-2b-it-q4-mlx"

_MLX_LANGUAGE_MODEL_MAP: dict[str, str] = {
    "laneformer-2b-it": LANEFORMER_MLX_MODEL,
    LANEFORMER_SOURCE_MODEL: LANEFORMER_MLX_MODEL,
    LANEFORMER_MLX_MODEL: LANEFORMER_MLX_MODEL,
}

_MLX_LM_ALLOW_PATTERNS = [
    "README.md",
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "*.py",
]


def resolve_mlx_language_model(model_name: str, config: Any = None) -> str:
    """Resolve an OpenMed/source/local name to an MLX-LM artifact path.

    Args:
        model_name: Registry alias, source model id, OpenMed MLX repo id, or
            local artifact directory.
        config: Optional OpenMed config. If present, ``cache_dir`` is passed to
            Hugging Face snapshot downloads.

    Returns:
        Local directory path containing an MLX-LM-compatible artifact.
    """

    local_path = Path(model_name).expanduser()
    if local_path.is_dir() and (local_path / "config.json").exists():
        return str(local_path)

    from openmed.core.model_registry import OPENMED_MODELS

    if model_name in OPENMED_MODELS:
        model_name = OPENMED_MODELS[model_name].model_id

    repo_id = _MLX_LANGUAGE_MODEL_MAP.get(model_name, model_name)
    if "/" not in repo_id:
        repo_id = f"OpenMed/{repo_id}"

    cache_dir = getattr(config, "cache_dir", None) if config is not None else None
    return _download_mlx_lm_artifact(repo_id, cache_dir=cache_dir)


def _download_mlx_lm_artifact(repo_id: str, cache_dir: str | None = None) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required for OpenMed MLX language models. "
            "Install with: pip install openmed[mlx]"
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=_MLX_LM_ALLOW_PATTERNS,
    )


class OpenMedMLXLanguageModel:
    """Small OpenMed wrapper around an MLX-LM causal language model."""

    def __init__(self, model_name: str = LANEFORMER_MLX_MODEL, config: Any = None):
        """Load an MLX-LM artifact.

        Args:
            model_name: Source model id, OpenMed MLX repo id, registry alias, or
                local artifact directory.
            config: Optional OpenMed config. If present, its ``cache_dir`` is
                passed to Hugging Face snapshot downloads.
        """

        self.model_name = model_name
        self.config = config
        self.model_path = resolve_mlx_language_model(model_name, config=config)

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for OpenMed MLX language models. "
                "Install with: pip install openmed[mlx]"
            ) from exc

        self.model, self.tokenizer = load(self.model_path)

    def format_chat_prompt(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render chat messages with the model tokenizer's chat template.

        Args:
            messages: Chat messages accepted by the tokenizer chat template.
            add_generation_prompt: Whether to append the assistant generation
                marker when rendering the prompt.

        Returns:
            Rendered prompt text.
        """

        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("The loaded tokenizer does not expose a chat template.")
        return tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    def generate(
        self,
        prompt: str | Sequence[int] | None = None,
        *,
        messages: Sequence[Mapping[str, str]] | None = None,
        max_tokens: int = 256,
        temp: float = 0.0,
        top_p: float = 1.0,
        verbose: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generate text with the loaded MLX language model.

        Args:
            prompt: Plain prompt text or token ids. Mutually exclusive with
                ``messages``.
            messages: Chat messages to render with the model chat template.
            max_tokens: Maximum generated token count.
            temp: Sampling temperature forwarded to ``mlx_lm.generate``.
            top_p: Top-p sampling value forwarded to ``mlx_lm.generate``.
            verbose: Whether MLX-LM should print generation details.
            **kwargs: Additional keyword arguments forwarded to
                ``mlx_lm.generate``.

        Returns:
            Generated text from MLX-LM.
        """

        if messages is not None:
            if prompt is not None:
                raise ValueError("Pass either prompt or messages, not both.")
            prompt = self.format_chat_prompt(messages)
        if prompt is None:
            raise ValueError("prompt or messages is required.")

        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for OpenMed MLX language models. "
                "Install with: pip install openmed[mlx]"
            ) from exc

        generate_kwargs = dict(kwargs)
        if "sampler" not in generate_kwargs and (temp > 0.0 or top_p < 1.0):
            try:
                from mlx_lm.sample_utils import make_sampler
            except ImportError as exc:
                raise ImportError(
                    "mlx-lm sampler utilities are required for non-greedy "
                    "OpenMed MLX language-model generation. Install with: "
                    "pip install openmed[mlx]"
                ) from exc

            sampler_top_p = top_p if top_p < 1.0 else 0.0
            generate_kwargs["sampler"] = make_sampler(temp=temp, top_p=sampler_top_p)

        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=verbose,
            **generate_kwargs,
        )


def generate_text(
    prompt: str | Sequence[int] | None = None,
    *,
    messages: Sequence[Mapping[str, str]] | None = None,
    model_name: str = LANEFORMER_MLX_MODEL,
    config: Any = None,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = False,
    **kwargs: Any,
) -> str:
    """Load an OpenMed MLX language model and generate text.

    Args:
        prompt: Plain prompt text or token ids. Mutually exclusive with
            ``messages``.
        messages: Chat messages to render with the model chat template.
        model_name: Source model id, OpenMed MLX repo id, registry alias, or
            local artifact directory.
        config: Optional OpenMed config for cache placement.
        max_tokens: Maximum generated token count.
        temp: Sampling temperature forwarded to ``mlx_lm.generate``.
        top_p: Top-p sampling value forwarded to ``mlx_lm.generate``.
        verbose: Whether MLX-LM should print generation details.
        **kwargs: Additional keyword arguments forwarded to ``mlx_lm.generate``.

    Returns:
        Generated text from MLX-LM.
    """

    runner = OpenMedMLXLanguageModel(model_name=model_name, config=config)
    return runner.generate(
        prompt,
        messages=messages,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        verbose=verbose,
        **kwargs,
    )


__all__ = [
    "LANEFORMER_MLX_MODEL",
    "LANEFORMER_SOURCE_MODEL",
    "OpenMedMLXLanguageModel",
    "generate_text",
    "resolve_mlx_language_model",
]
