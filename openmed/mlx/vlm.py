"""First-class OpenMed MLX vision-language inference.

Heavy MLX imports stay lazy so importing :mod:`openmed.mlx` still works when
the optional ``mlx`` extra is not installed.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_NORTH_MICRO_VISION_MODEL = "OpenMed/North-Micro-Vision-Instruct-4bit-mlx"
SUPPORTED_COMPASS_MODEL_TYPE = "cohere_compass"


class OpenMedMLXVLMError(RuntimeError):
    """Base error raised by the OpenMed MLX multimodal runtime."""


class OpenMedMLXVLMArtifactError(OpenMedMLXVLMError):
    """Raised when an MLX VLM artifact violates the runtime contract."""


@dataclass(frozen=True)
class CompassImageBatch:
    """Native-resolution image patches and their time/height/width grids."""

    pixel_values: Any
    image_grid_thw: Any


@dataclass(frozen=True)
class CompassPreparedInput:
    """Tokenized prompt plus optional native image tensors."""

    input_ids: Any
    attention_mask: Any
    pixel_values: Any | None
    image_grid_thw: Any | None
    formatted_prompt: str


@dataclass(frozen=True)
class VisionLanguageGeneration:
    """Text plus deterministic generation and performance metadata."""

    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generation_tokens: int
    prompt_seconds: float
    generation_seconds: float
    peak_memory_gb: float


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 32,
    min_pixels: int = 16_384,
    max_pixels: int = 3_868_706,
) -> tuple[int, int]:
    """Resize dimensions to a patch multiple within the Compass pixel budget.

    Args:
        height: Source image height in pixels.
        width: Source image width in pixels.
        factor: Required divisibility, normally patch size times merge size.
        min_pixels: Minimum resized pixel count.
        max_pixels: Maximum resized pixel count.

    Returns:
        The resized ``(height, width)`` pair.

    Raises:
        ValueError: If dimensions or the absolute aspect ratio are invalid.
    """

    if height <= 0 or width <= 0 or factor <= 0:
        raise ValueError("height, width, and factor must be positive")
    aspect = max(height, width) / min(height, width)
    if aspect > 200:
        raise ValueError(f"absolute aspect ratio must not exceed 200 (got {aspect})")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


def resolve_mlx_vlm_model(
    model: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve an OpenMed MLX VLM from disk or Hugging Face Hub.

    Args:
        model: Local artifact directory or Hugging Face repository ID.
        revision: Optional immutable Hub revision.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        The validated local artifact directory.
    """

    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return _validate_compass_artifact(candidate.resolve())
    if candidate.exists():
        raise OpenMedMLXVLMArtifactError(
            f"MLX VLM artifact must be a directory: {candidate}"
        )

    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=str(model),
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "model*.safetensors",
            "tokenizer.model",
            "*.tiktoken",
        ],
    )
    return _validate_compass_artifact(Path(downloaded))


def _validate_compass_artifact(directory: Path) -> Path:
    required = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if not list(directory.glob("model*.safetensors")):
        missing.append("model*.safetensors")
    if missing:
        raise OpenMedMLXVLMArtifactError(
            f"Compass artifact is incomplete ({', '.join(missing)}): {directory}"
        )
    try:
        config = json.loads((directory / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OpenMedMLXVLMArtifactError(
            f"Invalid Compass config.json in {directory}: {error}"
        ) from error
    model_type = config.get("model_type")
    if model_type != SUPPORTED_COMPASS_MODEL_TYPE:
        raise OpenMedMLXVLMArtifactError(
            f"Unsupported MLX VLM model_type {model_type!r}; expected "
            f"{SUPPORTED_COMPASS_MODEL_TYPE!r}"
        )
    manifest_path = directory / "openmed-mlx.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise OpenMedMLXVLMArtifactError(
                f"Invalid openmed-mlx.json in {directory}: {error}"
            ) from error
        family = str(manifest.get("family", "")).replace("_", "-")
        if family and family != "cohere-compass":
            raise OpenMedMLXVLMArtifactError(
                f"Manifest family {family!r} does not match cohere-compass"
            )
        task = manifest.get("task")
        if task and task != "image-text-to-text":
            raise OpenMedMLXVLMArtifactError(
                f"Manifest task {task!r} is not image-text-to-text"
            )
    return directory


class CohereCompassProcessor:
    """Torch-free native-resolution Compass text and image processor."""

    image_token = "<|IMAGE_PAD|>"

    def __init__(
        self,
        tokenizer: Any,
        model_config: dict[str, Any],
        processor_config: dict[str, Any] | None = None,
    ) -> None:
        processor_config = processor_config or {}
        vision = model_config.get("vision_config") or {}
        self.tokenizer = tokenizer
        self.patch_size = int(
            processor_config.get("patch_size", vision.get("patch_size", 16))
        )
        self.temporal_patch_size = int(
            processor_config.get(
                "temporal_patch_size", vision.get("temporal_patch_size", 2)
            )
        )
        self.merge_size = int(
            processor_config.get("merge_size", vision.get("spatial_merge_size", 2))
        )
        self.min_pixels = int(
            model_config.get("min_pixels")
            or processor_config.get("min_pixels")
            or 16_384
        )
        self.max_pixels = int(
            model_config.get("max_pixels")
            or processor_config.get("max_pixels")
            or 3_868_706
        )
        self.do_resize = bool(processor_config.get("do_resize", True))
        self.do_rescale = bool(processor_config.get("do_rescale", True))
        self.rescale_factor = float(processor_config.get("rescale_factor", 1 / 255))
        self.do_normalize = bool(processor_config.get("do_normalize", True))
        self.image_mean = processor_config.get("image_mean") or [0.5, 0.5, 0.5]
        self.image_std = processor_config.get("image_std") or [0.5, 0.5, 0.5]

    def format_prompt(
        self,
        prompt: str | Sequence[dict[str, Any]],
        *,
        image_count: int = 0,
    ) -> str:
        """Apply the checkpoint template with image items before user text."""

        if isinstance(prompt, str):
            if image_count:
                content: str | list[dict[str, str]] = [
                    *({"type": "image"} for _ in range(image_count)),
                    {"type": "text", "text": prompt, "content": prompt},
                ]
            else:
                content = prompt
            messages = [{"role": "user", "content": content}]
        else:
            messages = [dict(message) for message in prompt]
            if image_count:
                user_indexes = [
                    index
                    for index, message in enumerate(messages)
                    if message.get("role", "user") == "user"
                ]
                if not user_indexes:
                    raise ValueError("image prompts require a user message")
                index = user_indexes[-1]
                existing = messages[index].get("content", "")
                if isinstance(existing, str):
                    existing = [
                        {
                            "type": "text",
                            "text": existing,
                            "content": existing,
                        }
                    ]
                messages[index]["content"] = [
                    *({"type": "image"} for _ in range(image_count)),
                    *existing,
                ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def process_images(self, images: Sequence[Any]) -> CompassImageBatch:
        """Convert PIL/path/array images into packed native-resolution patches."""

        import mlx.core as mx
        import numpy as np
        from PIL import Image

        all_patches = []
        all_grids = []
        for value in images:
            if isinstance(value, (str, Path)):
                image = Image.open(value).convert("RGB")
            elif hasattr(value, "convert"):
                image = value.convert("RGB")
            else:
                pixels = np.asarray(value)
                if pixels.ndim == 2:
                    pixels = np.repeat(pixels[..., None], 3, axis=-1)
                if pixels.ndim != 3:
                    raise ValueError(
                        f"expected a three-dimensional image, got {pixels.shape}"
                    )
                if pixels.shape[0] in (1, 3, 4) and pixels.shape[-1] not in (
                    1,
                    3,
                    4,
                ):
                    pixels = pixels.transpose(1, 2, 0)
                image = Image.fromarray(pixels[..., :3].astype(np.uint8)).convert("RGB")

            width, height = image.size
            if self.do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = image.resize(
                    (resized_width, resized_height), Image.Resampling.BICUBIC
                )
            else:
                resized_height, resized_width = height, width

            pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
            if self.do_rescale:
                pixels = pixels * self.rescale_factor
            if self.do_normalize:
                mean = np.asarray(self.image_mean, dtype=np.float32)[:, None, None]
                std = np.asarray(self.image_std, dtype=np.float32)[:, None, None]
                pixels = (pixels - mean) / std

            temporal = self.temporal_patch_size
            patches = np.repeat(pixels[None, None], temporal, axis=1)
            grid_time = 1
            grid_height = resized_height // self.patch_size
            grid_width = resized_width // self.patch_size
            channels = patches.shape[2]
            patches = patches.reshape(
                1,
                grid_time,
                temporal,
                channels,
                grid_height // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_width // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            patches = patches.reshape(
                grid_time * grid_height * grid_width,
                channels * temporal * self.patch_size * self.patch_size,
            )
            all_patches.append(patches)
            all_grids.append([grid_time, grid_height, grid_width])
        return CompassImageBatch(
            pixel_values=mx.array(np.concatenate(all_patches, axis=0)),
            image_grid_thw=mx.array(np.asarray(all_grids, dtype=np.int32)),
        )

    def prepare(
        self,
        prompt: str | Sequence[dict[str, Any]],
        *,
        images: Sequence[Any] | None = None,
        formatted: bool = False,
    ) -> CompassPreparedInput:
        """Format, patchify, expand image tokens, and tokenize one request."""

        import mlx.core as mx

        image_values = list(images or [])
        formatted_prompt = (
            str(prompt)
            if formatted
            else self.format_prompt(prompt, image_count=len(image_values))
        )
        image_batch = self.process_images(image_values) if image_values else None
        if image_batch is not None:
            grids = image_batch.image_grid_thw.tolist()
            image_index = 0
            expanded = formatted_prompt
            while self.image_token in expanded:
                if image_index >= len(grids):
                    raise ValueError("more image placeholders than supplied images")
                time_value, height, width = grids[image_index]
                token_count = int(time_value * height * width) // self.merge_size**2
                replacement = "<|openmed_image_slot|>" * token_count
                expanded = expanded.replace(self.image_token, replacement, 1)
                image_index += 1
            if image_index != len(grids):
                raise ValueError("more supplied images than image placeholders")
            formatted_prompt = expanded.replace(
                "<|openmed_image_slot|>", self.image_token
            )

        encoded = self.tokenizer(
            formatted_prompt,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        input_ids = mx.array([encoded["input_ids"]], dtype=mx.int32)
        attention_mask = mx.array(
            [encoded.get("attention_mask", [1] * input_ids.shape[1])],
            dtype=mx.int32,
        )
        return CompassPreparedInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=image_batch.pixel_values if image_batch else None,
            image_grid_thw=image_batch.image_grid_thw if image_batch else None,
            formatted_prompt=formatted_prompt,
        )


class OpenMedMLXVisionLanguageModel:
    """Load and run an OpenMed Cohere Compass MLX model.

    Args:
        model: Local artifact directory or Hugging Face repository ID.
        revision: Optional immutable Hub revision.
        cache_dir: Optional Hugging Face cache directory.
        lazy: Defer weight evaluation until first use.
        strict: Require every checkpoint weight to match the OpenMed model.
    """

    def __init__(
        self,
        model: str | Path = DEFAULT_NORTH_MICRO_VISION_MODEL,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        lazy: bool = False,
        strict: bool = True,
    ) -> None:
        self.model_path = resolve_mlx_vlm_model(
            model, revision=revision, cache_dir=cache_dir
        )
        self.strict = strict
        self.lazy = lazy
        self.model: Any = None
        self.processor: CohereCompassProcessor | None = None
        self.config: dict[str, Any] | None = None
        self._load()

    def _load(self) -> None:
        import mlx_lm.utils
        from transformers import AutoTokenizer

        from openmed.mlx.models.cohere_compass import Model, ModelConfig

        def compass_classes(config: dict[str, Any]) -> tuple[type, type]:
            del config
            return Model, ModelConfig

        loaded_model, config = mlx_lm.utils.load_model(
            self.model_path,
            lazy=self.lazy,
            strict=self.strict,
            get_model_classes=compass_classes,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        template_path = self.model_path / "chat_template.jinja"
        if template_path.is_file():
            tokenizer.chat_template = template_path.read_text()
        processor_config = json.loads(
            (self.model_path / "preprocessor_config.json").read_text()
        )
        self.model = loaded_model
        self.config = config
        self.processor = CohereCompassProcessor(tokenizer, config, processor_config)

    def generate(
        self,
        prompt: str | Sequence[dict[str, Any]],
        *,
        image: Any | None = None,
        images: Sequence[Any] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        formatted: bool = False,
    ) -> str:
        """Generate a text response from text and optional image input.

        Args:
            prompt: User text or a chat-message sequence.
            image: Optional single PIL image, path, or image array.
            images: Optional image sequence; mutually exclusive with ``image``.
            max_tokens: Maximum number of generated tokens.
            temperature: Sampling temperature; zero performs greedy decoding.
            top_p: Optional nucleus-sampling cutoff.
            top_k: Optional top-k sampling cutoff.
            formatted: Treat ``prompt`` as an already-rendered chat template.

        Returns:
            The decoded assistant response.
        """

        return self.generate_with_metadata(
            prompt,
            image=image,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            formatted=formatted,
        ).text

    def generate_with_metadata(
        self,
        prompt: str | Sequence[dict[str, Any]],
        *,
        image: Any | None = None,
        images: Sequence[Any] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        formatted: bool = False,
    ) -> VisionLanguageGeneration:
        """Generate text and return token, latency, and memory metadata."""

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if image is not None and images is not None:
            raise ValueError("pass either image or images, not both")
        image_values = list(images) if images is not None else []
        if image is not None:
            image_values = [image]
        assert self.processor is not None
        assert self.config is not None

        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        prepared = self.processor.prepare(
            prompt,
            images=image_values,
            formatted=formatted,
        )
        self.model.reset_generation_state()
        cache = self.model.make_cache()
        prompt_started = time.perf_counter()
        logits = self.model(
            prepared.input_ids,
            pixel_values=prepared.pixel_values,
            image_grid_thw=prepared.image_grid_thw,
            attention_mask=prepared.attention_mask,
            cache=cache,
        )
        mx.eval(logits)
        prompt_seconds = time.perf_counter() - prompt_started

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        eos_value = self.config.get("eos_token_id", [])
        eos_values = eos_value if isinstance(eos_value, list) else [eos_value]
        eos_ids = {int(value) for value in eos_values if value is not None}
        generated: list[int] = []
        generation_started = time.perf_counter()
        for _ in range(max_tokens):
            last_logits = logits[:, -1, :]
            log_probabilities = last_logits - mx.logsumexp(
                last_logits, axis=-1, keepdims=True
            )
            token = sampler(log_probabilities)
            mx.eval(token)
            token_id = int(token.item())
            if token_id in eos_ids:
                break
            generated.append(token_id)
            logits = self.model(
                mx.array([[token_id]], dtype=mx.int32),
                cache=cache,
            )
            mx.eval(logits)
        generation_seconds = time.perf_counter() - generation_started
        text = self.processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()
        return VisionLanguageGeneration(
            text=text,
            token_ids=tuple(generated),
            prompt_tokens=int(prepared.input_ids.shape[1]),
            generation_tokens=len(generated),
            prompt_seconds=prompt_seconds,
            generation_seconds=generation_seconds,
            peak_memory_gb=float(mx.get_peak_memory()) / 1e9,
        )


def generate_vision_text(
    prompt: str,
    *,
    image: Any | None = None,
    model: str | Path = DEFAULT_NORTH_MICRO_VISION_MODEL,
    max_tokens: int = 256,
    **kwargs: Any,
) -> str:
    """Load an OpenMed MLX VLM and generate one text or image response."""

    runner = OpenMedMLXVisionLanguageModel(model)
    return runner.generate(prompt, image=image, max_tokens=max_tokens, **kwargs)


__all__ = [
    "CohereCompassProcessor",
    "CompassImageBatch",
    "CompassPreparedInput",
    "DEFAULT_NORTH_MICRO_VISION_MODEL",
    "OpenMedMLXVLMArtifactError",
    "OpenMedMLXVLMError",
    "OpenMedMLXVisionLanguageModel",
    "VisionLanguageGeneration",
    "generate_vision_text",
    "resolve_mlx_vlm_model",
    "smart_resize",
]
