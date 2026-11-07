"""Optional MLX runtime for local, immutable SmolVLM checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

from openmed.multimodal.vision.providers.local_assets import LocalVLMAssets


@dataclass(frozen=True, slots=True)
class SmolVLMGeneration:
    """Transient generation; never serialize text or token identifiers."""

    text: str = field(repr=False)
    output_tokens: int


class SmolVLMRuntime:
    """Load a verified local checkpoint without remote code or Hub resolution.

    Args:
        assets: Immutable local model and tokenizer assets already verified by caller.

    Construct in a dedicated local worker for hard resource/network isolation.
    The adapter supplies only in-memory images and has no URL or upload interface.
    """

    def __init__(self, assets: LocalVLMAssets) -> None:
        import mlx.core as mx
        from mlx_vlm.utils import load_model, load_processor

        assets.verify_current()
        self._mx = mx
        self._config = json.loads((assets.directory / "config.json").read_bytes())
        self._model = load_model(assets.directory, lazy=False, strict=True)
        self._processor = load_processor(
            assets.directory,
            local_files_only=True,
            trust_remote_code=False,
            eos_token_ids=self._model.config.eos_token_id,
        )
        self.versions = {
            name: version(name)
            for name in (
                "mlx",
                "mlx-vlm",
                "transformers",
                "tokenizers",
                "Pillow",
                "numpy",
            )
        }

    def generate(
        self, image: Any, prompt: str, max_tokens: int, seed: int
    ) -> SmolVLMGeneration:
        """Run greedy generation with no conversation, disk or vision-feature cache.

        Args:
            image: Fresh decoded RGB Pillow image held only for this call.
            prompt: Protected text, never printed.
            max_tokens: Hard decoder token cap.
            seed: Explicit MLX random seed; temperature remains zero.
        """
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self._mx.random.seed(seed)
        formatted = apply_chat_template(
            self._processor, self._config, prompt, num_images=1
        )
        result = generate(
            self._model,
            self._processor,
            formatted,
            image=[image],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            verbose=False,
        )
        self._mx.synchronize()
        return SmolVLMGeneration(result.text, result.generation_tokens)
