"""Native MLX implementation of the Cohere Compass multimodal architecture.

The implementation is intentionally owned by OpenMed and uses only MLX/MLX-LM
primitives.  It mirrors the public checkpoint architecture rather than loading
remote model code or delegating inference to a separate VLM runtime.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache


@dataclass
class VisionConfig:
    """Configuration for the native-resolution Compass vision encoder."""

    model_type: str = "cohere_compass_vision"
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])
    use_rope: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> VisionConfig:
        """Build a vision configuration while ignoring HF metadata fields."""

        values = values or {}
        allowed = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class TextConfig:
    """Configuration for the Compass autoregressive decoder."""

    model_type: str = "cohere_compass_text"
    vocab_size: int = 131072
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    layer_norm_eps: float = 1e-5
    rms_norm_eps: float | None = None
    norm_type: str = "layer_norm"
    transformer_block_type: str = "parallel"
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    swa_rope_theta: float | None = None
    rope_parameters: dict[str, Any] | None = None
    rope_style: str = "split"
    rope_on_all_layers: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    sliding_window: int | None = None
    layer_types: list[str] | None = None
    logit_scale: float | None = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.swa_rope_theta is None:
            self.swa_rope_theta = self.rope_theta
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> TextConfig:
        """Build a text configuration while ignoring HF metadata fields."""

        values = values or {}
        allowed = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class ModelConfig:
    """Top-level configuration for ``cohere_compass`` checkpoints."""

    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig | None = field(default_factory=VisionConfig)
    model_type: str = "cohere_compass"
    image_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> ModelConfig:
        """Build a model configuration from a Hugging Face config mapping."""

        values = dict(values or {})
        values["text_config"] = TextConfig.from_dict(values.get("text_config"))
        raw_vision = values.get("vision_config")
        values["vision_config"] = (
            VisionConfig.from_dict(raw_vision) if raw_vision is not None else None
        )
        allowed = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in values.items() if key in allowed})


class CompassLayerNorm(nn.Module):
    """Bias-free LayerNorm with float32 statistics."""

    def __init__(self, dimensions: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dimensions,))
        self.eps = eps

    def __call__(self, values: mx.array) -> mx.array:
        source_dtype = values.dtype
        values = values.astype(mx.float32)
        mean = mx.mean(values, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(values - mean), axis=-1, keepdims=True)
        normalized = (values - mean) * mx.rsqrt(variance + self.eps)
        return (normalized * self.weight.astype(mx.float32)).astype(source_dtype)


class CompassRMSNorm(nn.Module):
    """Bias-free RMSNorm with float32 statistics."""

    def __init__(self, dimensions: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dimensions,))
        self.eps = eps

    def __call__(self, values: mx.array) -> mx.array:
        source_dtype = values.dtype
        values = values.astype(mx.float32)
        normalized = values * mx.rsqrt(
            mx.mean(mx.square(values), axis=-1, keepdims=True) + self.eps
        )
        return (normalized * self.weight.astype(mx.float32)).astype(source_dtype)


def _normalization(config: TextConfig) -> nn.Module:
    if config.norm_type == "rms_norm":
        return CompassRMSNorm(config.hidden_size, config.rms_norm_eps or 1e-6)
    return CompassLayerNorm(config.hidden_size, config.layer_norm_eps)


def _rotate_split(values: mx.array) -> mx.array:
    half = values.shape[-1] // 2
    return mx.concatenate([-values[..., half:], values[..., :half]], axis=-1)


def _rotate_interleaved(values: mx.array) -> mx.array:
    pairs = mx.stack([-values[..., 1::2], values[..., ::2]], axis=-1)
    return pairs.flatten(-2)


class CompassRotaryEmbedding(nn.Module):
    """Per-layer Compass RoPE with interleaved three-axis position selection."""

    def __init__(self, config: TextConfig, layer_type: str) -> None:
        super().__init__()
        assert config.head_dim is not None
        assert config.layer_types is not None
        self.dimensions = config.head_dim
        self.style = config.rope_style

        raw = config.rope_parameters
        per_layer = isinstance(raw, dict) and any(
            kind in raw for kind in config.layer_types
        )
        parameters = raw.get(layer_type) if per_layer and raw else raw
        self.enabled = (
            parameters is not None if per_layer else config.rope_on_all_layers
        )
        parameters = parameters or {}
        theta = parameters.get(
            "rope_theta",
            config.swa_rope_theta
            if layer_type == "sliding_attention"
            else config.rope_theta,
        )
        exponents = mx.arange(0, self.dimensions, 2, dtype=mx.float32)
        self._inverse_frequencies = 1.0 / (
            float(theta) ** (exponents / self.dimensions)
        )

        section = parameters.get("mrope_section")
        expected = self._inverse_frequencies.shape[0]
        if (
            isinstance(section, list)
            and len(section) == 3
            and all(isinstance(size, int) and size >= 0 for size in section)
            and sum(section) == expected
        ):
            selector = [0] * expected
            for axis in (1, 2):
                for index in range(axis, min(section[axis] * 3, expected), 3):
                    selector[index] = axis
            self._position_selector = mx.array(selector, dtype=mx.int32)
        else:
            self._position_selector = None

    @property
    def has_multimodal_positions(self) -> bool:
        """Whether this RoPE instance consumes temporal/row/column positions."""

        return self._position_selector is not None

    def __call__(
        self, reference: mx.array, position_ids: mx.array
    ) -> tuple[mx.array, mx.array]:
        if position_ids.ndim == 3 and self._position_selector is not None:
            positions = mx.take(position_ids, self._position_selector, axis=0)
            frequencies = positions.transpose(1, 2, 0).astype(mx.float32)
            frequencies = frequencies * self._inverse_frequencies
        else:
            frequencies = (
                position_ids.astype(mx.float32)[..., None] * self._inverse_frequencies
            )
        if self.style == "interleave":
            angles = mx.repeat(frequencies, 2, axis=-1)
        else:
            angles = mx.concatenate([frequencies, frequencies], axis=-1)
        return (
            mx.cos(angles).astype(reference.dtype),
            mx.sin(angles).astype(reference.dtype),
        )


class CompassAttention(nn.Module):
    """Grouped-query attention used by Compass text layers."""

    def __init__(self, config: TextConfig, layer_index: int) -> None:
        super().__init__()
        assert config.head_dim is not None
        assert config.num_key_value_heads is not None
        assert config.layer_types is not None
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.layer_type = config.layer_types[layer_index]
        self.rotary_emb = CompassRotaryEmbedding(config, self.layer_type)
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.query_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.query_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None,
        cache: Any,
        position_ids: mx.array,
        position_embeddings: tuple[mx.array, mx.array] | None,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            batch, length, self.query_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            batch, length, self.kv_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            batch, length, self.kv_heads, self.head_dim
        )
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.rotary_emb.enabled:
            cos_values, sin_values = position_embeddings or self.rotary_emb(
                hidden_states, position_ids
            )
            cos_values = mx.expand_dims(cos_values, axis=1)
            sin_values = mx.expand_dims(sin_values, axis=1)
            rotate = (
                _rotate_interleaved
                if self.rotary_emb.style == "interleave"
                else _rotate_split
            )
            queries = (queries * cos_values + rotate(queries) * sin_values).astype(
                queries.dtype
            )
            keys = (keys * cos_values + rotate(keys) * sin_values).astype(keys.dtype)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        if isinstance(mask, mx.array) and mask.shape[-1] != keys.shape[-2]:
            mask = mask[..., -keys.shape[-2] :]
        attended = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(attended)


class CompassMLP(nn.Module):
    """SwiGLU feed-forward network with checkpoint-compatible names."""

    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, values: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(values)) * self.up_proj(values))


class CompassDecoderLayer(nn.Module):
    """Parallel or sequential Compass decoder block."""

    def __init__(self, config: TextConfig, layer_index: int) -> None:
        super().__init__()
        assert config.layer_types is not None
        self.attention_type = config.layer_types[layer_index]
        self.input_layernorm = _normalization(config)
        self.self_attn = CompassAttention(config, layer_index)
        self.mlp = CompassMLP(config)
        self.parallel = config.transformer_block_type == "parallel"
        if not self.parallel:
            self.post_attention_layernorm = _normalization(config)

    def __call__(
        self,
        values: mx.array,
        mask: mx.array | str | None,
        cache: Any,
        position_ids: mx.array,
        position_embeddings: tuple[mx.array, mx.array] | None,
    ) -> mx.array:
        normalized = self.input_layernorm(values)
        attention = self.self_attn(
            normalized, mask, cache, position_ids, position_embeddings
        )
        if self.parallel:
            return values + attention + self.mlp(normalized)
        residual = values + attention
        return residual + self.mlp(self.post_attention_layernorm(residual))


class CompassTextModel(nn.Module):
    """Embedding, decoder stack, and final norm for Compass."""

    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        assert config.layer_types is not None
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            CompassDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ]
        self.norm = _normalization(config)
        layer_types = list(dict.fromkeys(config.layer_types))
        self.rotary_embeddings = {
            kind: CompassRotaryEmbedding(config, kind) for kind in layer_types
        }
        self.primary_rotary = next(
            (rotary for rotary in self.rotary_embeddings.values() if rotary.enabled),
            next(iter(self.rotary_embeddings.values())),
        )

    def __call__(
        self,
        input_ids: mx.array,
        *,
        inputs_embeds: mx.array | None = None,
        cache: list[Any] | None = None,
        mask: mx.array | str | None = None,
        position_ids: mx.array | None = None,
        visual_mask: mx.array | None = None,
        deepstack_features: list[mx.array] | None = None,
    ) -> mx.array:
        hidden = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        caches = cache or [None] * len(self.layers)
        assert self.config.layer_types is not None

        global_index = next(
            (
                index
                for index, kind in enumerate(self.config.layer_types)
                if kind == "full_attention"
            ),
            None,
        )
        sliding_index = next(
            (
                index
                for index, kind in enumerate(self.config.layer_types)
                if kind == "sliding_attention"
            ),
            None,
        )
        global_cache = caches[global_index] if global_index is not None else None
        sliding_cache = caches[sliding_index] if sliding_index is not None else None
        if mask is None:
            global_mask = create_attention_mask(hidden, global_cache)
            sliding_mask = create_attention_mask(
                hidden,
                sliding_cache,
                window_size=self.config.sliding_window,
            )
        else:
            global_mask = sliding_mask = mask

        if position_ids is None:
            position_ids = mx.arange(hidden.shape[1], dtype=mx.int32)[None, :]
            if self.primary_rotary.has_multimodal_positions:
                position_ids = mx.broadcast_to(
                    position_ids[None, ...],
                    (3, hidden.shape[0], hidden.shape[1]),
                )
        rotary_values = {
            kind: rotary(hidden, position_ids) if rotary.enabled else None
            for kind, rotary in self.rotary_embeddings.items()
        }

        for index, (layer, layer_cache) in enumerate(zip(self.layers, caches)):
            layer_mask = (
                global_mask
                if layer.attention_type == "full_attention"
                else sliding_mask
            )
            hidden = layer(
                hidden,
                layer_mask,
                layer_cache,
                position_ids,
                rotary_values[layer.attention_type],
            )
            if deepstack_features is not None and index < len(deepstack_features):
                hidden = _inject_visual_rows(
                    hidden, visual_mask, deepstack_features[index], add=True
                )
        return self.norm(hidden)


class CompassLanguageModel(nn.Module):
    """Checkpoint-compatible Compass language-model wrapper."""

    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = CompassTextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def layers(self) -> list[CompassDecoderLayer]:
        """Return decoder layers for MLX-LM cache introspection."""

        return self.model.layers

    def make_cache(self) -> list[Any]:
        """Create full and rotating KV caches matching each layer type."""

        assert self.config.layer_types is not None
        return [
            RotatingKVCache(max_size=self.config.sliding_window, keep=0)
            if kind == "sliding_attention"
            else KVCache()
            for kind in self.config.layer_types
        ]

    def __call__(
        self,
        input_ids: mx.array,
        *,
        inputs_embeds: mx.array | None = None,
        cache: list[Any] | None = None,
        mask: mx.array | str | None = None,
        position_ids: mx.array | None = None,
        visual_mask: mx.array | None = None,
        deepstack_features: list[mx.array] | None = None,
    ) -> mx.array:
        hidden = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            mask=mask,
            position_ids=position_ids,
            visual_mask=visual_mask,
            deepstack_features=deepstack_features,
        )
        projection = (
            self.model.embed_tokens.as_linear
            if self.config.tie_word_embeddings
            else self.lm_head
        )
        logits = projection(hidden)
        if self.config.logit_scale is not None:
            logits = logits * self.config.logit_scale
        return logits


class VisionRotaryEmbedding(nn.Module):
    """Two-dimensional rotary frequency table for vision patches."""

    def __init__(self, dimensions: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.theta = theta

    def __call__(self, positions: mx.array) -> mx.array:
        steps = mx.arange(0, self.dimensions, 2, dtype=mx.float32)
        inverse = 1.0 / (self.theta ** (steps / self.dimensions))
        return (positions[..., None] * inverse).reshape(positions.shape[0], -1)


class CompassPatchEmbedding(nn.Module):
    """Spatiotemporal Conv3D patch projection."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        kernel = (self.temporal_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )

    def __call__(self, patches: mx.array) -> mx.array:
        patches = patches.reshape(
            -1,
            self.in_channels,
            self.temporal_size,
            self.patch_size,
            self.patch_size,
        ).transpose(0, 2, 3, 4, 1)
        return self.proj(patches).reshape(patches.shape[0], -1)


class CompassPatchMerger(nn.Module):
    """Merge each 2x2 patch group into one language-model embedding."""

    def __init__(self, config: VisionConfig, *, postshuffle_norm: bool = False) -> None:
        super().__init__()
        merged = config.hidden_size * config.spatial_merge_size**2
        self.merged_size = merged
        self.postshuffle_norm = postshuffle_norm
        self.norm = nn.LayerNorm(
            merged if postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(merged, merged)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(merged, config.out_hidden_size)

    def __call__(self, values: mx.array) -> mx.array:
        if self.postshuffle_norm:
            values = self.norm(values.reshape(-1, self.merged_size))
        else:
            values = self.norm(values).reshape(-1, self.merged_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(values)))


def _vision_rotate(values: mx.array) -> mx.array:
    half = values.shape[-1] // 2
    return mx.concatenate([-values[..., half:], values[..., :half]], axis=-1)


class CompassVisionAttention(nn.Module):
    """Independent attention over each image/frame patch sequence."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        values: mx.array,
        sequence_ends: mx.array,
        rotary: mx.array | None,
    ) -> mx.array:
        length = values.shape[0]
        qkv = self.qkv(values).reshape(length, 3, self.num_heads, self.head_dim)
        query, key, value = [part[:, 0] for part in mx.split(qkv, 3, axis=1)]
        if rotary is not None:
            cos_values = mx.concatenate([mx.cos(rotary), mx.cos(rotary)], axis=-1)
            sin_values = mx.concatenate([mx.sin(rotary), mx.sin(rotary)], axis=-1)
            cos_values = mx.expand_dims(cos_values.astype(mx.float32), axis=-2)
            sin_values = mx.expand_dims(sin_values.astype(mx.float32), axis=-2)
            query_type, key_type = query.dtype, key.dtype
            query = query.astype(mx.float32)
            key = key.astype(mx.float32)
            query = query * cos_values + _vision_rotate(query) * sin_values
            key = key * cos_values + _vision_rotate(key) * sin_values
            query, key = query.astype(query_type), key.astype(key_type)

        query = query.transpose(1, 0, 2)[None]
        key = key.transpose(1, 0, 2)[None]
        value = value.transpose(1, 0, 2)[None]
        split_points = sequence_ends[1:-1].tolist()
        query_parts = mx.split(query, split_points, axis=2)
        key_parts = mx.split(key, split_points, axis=2)
        value_parts = mx.split(value, split_points, axis=2)
        attended = [
            mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            for q, k, v in zip(query_parts, key_parts, value_parts)
        ]
        output = mx.concatenate(attended, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(length, -1)
        return self.proj(output)


class CompassVisionMLP(nn.Module):
    """GELU MLP for the vision transformer."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = nn.GELU(approx="tanh")

    def __call__(self, values: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(values)))


class CompassVisionBlock(nn.Module):
    """Pre-normalized vision-transformer block."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = CompassVisionAttention(config)
        self.mlp = CompassVisionMLP(config)

    def __call__(
        self, values: mx.array, sequence_ends: mx.array, rotary: mx.array | None
    ) -> mx.array:
        values = values + self.attn(self.norm1(values), sequence_ends, rotary)
        return values + self.mlp(self.norm2(values))


class CompassVisionModel(nn.Module):
    """Packed native-resolution Compass vision tower."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        if config.model_type != "cohere_compass_vision":
            raise ValueError(f"Unsupported Compass vision type: {config.model_type}")
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = CompassPatchEmbedding(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.grid_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = [CompassVisionBlock(config) for _ in range(config.depth)]
        self.merger = CompassPatchMerger(config)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = [
            CompassPatchMerger(config, postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        ]
        self.use_rope = config.use_rope

    def _interpolated_positions(self, grids: mx.array) -> mx.array:
        index_sets: list[list[int]] = [[] for _ in range(4)]
        weight_sets: list[list[float]] = [[] for _ in range(4)]
        for _, height, width in grids.tolist():
            height, width = int(height), int(width)
            row_values = mx.linspace(0, self.grid_side - 1, height)
            column_values = mx.linspace(0, self.grid_side - 1, width)
            row_floor = row_values.astype(mx.int32)
            column_floor = column_values.astype(mx.int32)
            row_ceil = mx.minimum(row_floor + 1, self.grid_side - 1)
            column_ceil = mx.minimum(column_floor + 1, self.grid_side - 1)
            row_fraction = row_values - row_floor.astype(mx.float32)
            column_fraction = column_values - column_floor.astype(mx.float32)
            row_base = row_floor * self.grid_side
            row_ceil_base = row_ceil * self.grid_side
            indices = [
                (row_base[:, None] + column_floor[None, :]).flatten(),
                (row_base[:, None] + column_ceil[None, :]).flatten(),
                (row_ceil_base[:, None] + column_floor[None, :]).flatten(),
                (row_ceil_base[:, None] + column_ceil[None, :]).flatten(),
            ]
            weights = [
                (
                    (1 - row_fraction)[:, None] * (1 - column_fraction)[None, :]
                ).flatten(),
                ((1 - row_fraction)[:, None] * column_fraction[None, :]).flatten(),
                (row_fraction[:, None] * (1 - column_fraction)[None, :]).flatten(),
                (row_fraction[:, None] * column_fraction[None, :]).flatten(),
            ]
            for corner in range(4):
                index_sets[corner].extend(indices[corner].tolist())
                weight_sets[corner].extend(weights[corner].tolist())

        indices = mx.array(index_sets, dtype=mx.int32)
        weights = mx.array(weight_sets, dtype=mx.float32)
        interpolated = (self.pos_embed(indices) * weights[..., None]).sum(axis=0)
        lengths = [int(height * width) for _, height, width in grids.tolist()]
        split_points = list(accumulate(lengths[:-1]))
        parts = (
            mx.split(interpolated, split_points, axis=0)
            if split_points
            else [interpolated]
        )
        reordered = []
        merge = self.spatial_merge_size
        for part, (time, height, width) in zip(parts, grids.tolist()):
            time, height, width = int(time), int(height), int(width)
            part = mx.tile(part, (time, 1)).reshape(time, height, width, -1)
            part = part.reshape(
                time,
                height // merge,
                merge,
                width // merge,
                merge,
                part.shape[-1],
            )
            reordered.append(
                part.transpose(0, 1, 3, 2, 4, 5).reshape(-1, part.shape[-1])
            )
        return mx.concatenate(reordered, axis=0)

    def _rotary_positions(self, grids: mx.array) -> mx.array:
        max_side = int(mx.max(grids[:, 1:]).item())
        table = self.rotary_pos_emb(mx.arange(max_side, dtype=mx.int32))
        positions = []
        merge = self.spatial_merge_size
        for time, height, width in grids.tolist():
            time, height, width = int(time), int(height), int(width)
            merged_height, merged_width = height // merge, width // merge
            rows = mx.arange(merged_height)[:, None, None, None] * merge
            rows = rows + mx.arange(merge)[None, None, :, None]
            columns = mx.arange(merged_width)[None, :, None, None] * merge
            columns = columns + mx.arange(merge)[None, None, None, :]
            shape = (merged_height, merged_width, merge, merge)
            rows = mx.broadcast_to(rows, shape).reshape(-1)
            columns = mx.broadcast_to(columns, shape).reshape(-1)
            coords = mx.stack([rows, columns], axis=-1)
            positions.append(mx.tile(coords, (time, 1)) if time > 1 else coords)
        position_ids = mx.concatenate(positions, axis=0)
        return mx.concatenate(
            [table[position_ids[:, 0]], table[position_ids[:, 1]]], axis=-1
        )

    def _single(
        self, patches: mx.array, grids: mx.array
    ) -> tuple[mx.array, list[mx.array]]:
        hidden = self.patch_embed(patches)
        hidden = hidden + self._interpolated_positions(grids).astype(hidden.dtype)
        rotary = self._rotary_positions(grids) if self.use_rope else None
        sequence_lengths = []
        for time, height, width in grids.tolist():
            sequence_lengths.extend([int(height) * int(width)] * int(time))
        sequence_ends = mx.pad(
            mx.cumsum(mx.array(sequence_lengths, dtype=mx.int32)), (1, 0)
        )
        deepstack = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, sequence_ends, rotary)
            if index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(index)
                deepstack.append(self.deepstack_merger_list[merger_index](hidden))
        return self.merger(hidden), deepstack

    def __call__(
        self, patches: mx.array, grids: mx.array
    ) -> tuple[mx.array, list[mx.array]]:
        if grids.shape[0] == 1:
            return self._single(patches, grids)
        lengths = [int(t * h * w) for t, h, w in grids.tolist()]
        split_points = list(accumulate(lengths[:-1]))
        patch_groups = mx.split(patches, split_points, axis=0)
        outputs = []
        deep_outputs: list[list[mx.array]] = [[] for _ in self.deepstack_visual_indexes]
        for image_patches, grid in zip(patch_groups, grids):
            output, deepstack = self._single(image_patches, grid[None])
            outputs.append(output)
            for index, features in enumerate(deepstack):
                deep_outputs[index].append(features)
        return mx.concatenate(outputs, axis=0), [
            mx.concatenate(features, axis=0) for features in deep_outputs
        ]


def _visual_indices(mask: mx.array) -> mx.array:
    flat = np.asarray(mask.tolist(), dtype=bool).reshape(-1)
    return mx.array(np.flatnonzero(flat), dtype=mx.uint32)


def _inject_visual_rows(
    hidden: mx.array,
    mask: mx.array | None,
    visual: mx.array,
    *,
    add: bool,
) -> mx.array:
    if mask is None:
        return hidden
    indices = _visual_indices(mask)
    if indices.shape[0] == 0:
        return hidden
    flattened = hidden.reshape(-1, hidden.shape[-1])
    if add:
        flattened = flattened.at[indices].add(visual[: indices.shape[0]])
    else:
        flattened[indices] = visual[: indices.shape[0]]
    return flattened.reshape(hidden.shape)


class Model(nn.Module):
    """Native OpenMed MLX model for Cohere Compass checkpoints."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.language_model = CompassLanguageModel(config.text_config)
        self.vision_tower = (
            CompassVisionModel(config.vision_config)
            if config.vision_config is not None
            else None
        )
        self._rope_delta: mx.array | None = None

    @property
    def layers(self) -> list[CompassDecoderLayer]:
        """Return language layers for MLX-LM cache introspection."""

        return self.language_model.layers

    def make_cache(self) -> list[Any]:
        """Create a fresh mixed sliding/global KV cache."""

        return self.language_model.make_cache()

    def reset_generation_state(self) -> None:
        """Clear per-request multimodal position state."""

        self._rope_delta = None

    def _rope_positions(
        self,
        input_ids: mx.array,
        grids: mx.array | None,
        attention_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        rotary = self.language_model.model.primary_rotary
        if not rotary.has_multimodal_positions:
            positions = mx.arange(input_ids.shape[1], dtype=mx.int32)[None, :]
            return mx.broadcast_to(positions, input_ids.shape), mx.zeros(
                (input_ids.shape[0], 1), dtype=input_ids.dtype
            )
        attention_mask = (
            mx.ones_like(input_ids) if attention_mask is None else attention_mask
        )
        if grids is None:
            positions = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1
            positions = mx.where(
                attention_mask == 0, mx.ones_like(positions), positions
            )
            delta = positions.max(axis=-1, keepdims=True) + 1 - input_ids.shape[1]
            return mx.broadcast_to(
                positions[None, ...], (3, input_ids.shape[0], input_ids.shape[1])
            ), delta

        assert self.config.image_token_id is not None
        assert self.config.vision_start_token_id is not None
        assert self.config.vision_config is not None
        rows = []
        deltas = []
        image_index = 0
        merge = self.config.vision_config.spatial_merge_size
        for token_row, mask_row in zip(input_ids.tolist(), attention_mask.tolist()):
            valid = [index for index, keep in enumerate(mask_row) if keep]
            tokens = [token_row[index] for index in valid]
            image_count = sum(
                tokens[index + 1] == self.config.image_token_id
                for index, token in enumerate(tokens[:-1])
                if token == self.config.vision_start_token_id
            )
            pieces = []
            start = 0
            for _ in range(image_count):
                end = tokens.index(self.config.image_token_id, start)
                time, height, width = [
                    int(value) for value in grids[image_index].tolist()
                ]
                image_index += 1
                text_length = end - start
                origin = int(pieces[-1].max().item()) + 1 if pieces else 0
                pieces.append(
                    mx.broadcast_to(
                        mx.arange(text_length, dtype=mx.int32)[None, :],
                        (3, text_length),
                    )
                    + origin
                )
                grid_height, grid_width = height // merge, width // merge
                temporal = mx.broadcast_to(
                    mx.arange(time, dtype=mx.int32)[:, None],
                    (time, grid_height * grid_width),
                ).reshape(-1)
                row_ids = mx.broadcast_to(
                    mx.arange(grid_height, dtype=mx.int32)[None, :, None],
                    (time, grid_height, grid_width),
                ).reshape(-1)
                column_ids = mx.broadcast_to(
                    mx.arange(grid_width, dtype=mx.int32)[None, None, :],
                    (time, grid_height, grid_width),
                ).reshape(-1)
                pieces.append(
                    mx.stack([temporal, row_ids, column_ids]) + text_length + origin
                )
                start = end + time * grid_height * grid_width
            if start < len(tokens):
                origin = int(pieces[-1].max().item()) + 1 if pieces else 0
                text_length = len(tokens) - start
                pieces.append(
                    mx.broadcast_to(
                        mx.arange(text_length, dtype=mx.int32)[None, :],
                        (3, text_length),
                    )
                    + origin
                )
            compact = mx.concatenate(pieces, axis=1)
            padded = [[1] * input_ids.shape[1] for _ in range(3)]
            for compact_index, original_index in enumerate(valid):
                for axis in range(3):
                    padded[axis][original_index] = int(
                        compact[axis, compact_index].item()
                    )
            rows.append(mx.array(padded, dtype=mx.int32))
            deltas.append(int(compact.max().item()) + 1 - input_ids.shape[1])
        return mx.stack(rows, axis=1), mx.array(deltas, dtype=input_ids.dtype)[:, None]

    def __call__(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        attention_mask: mx.array | None = None,
        cache: list[Any] | None = None,
    ) -> mx.array:
        inputs_embeds = None
        visual_mask = None
        deepstack = None
        if pixel_values is not None:
            if self.vision_tower is None or image_grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values")
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
            image_features, deepstack = self.vision_tower(
                pixel_values.astype(self.vision_tower.patch_embed.proj.weight.dtype),
                image_grid_thw,
            )
            visual_mask = input_ids == self.config.image_token_id
            expected = int(visual_mask.sum().item())
            if expected != image_features.shape[0]:
                raise ValueError(
                    "Image feature/token mismatch: "
                    f"{image_features.shape[0]} features for {expected} tokens"
                )
            inputs_embeds = _inject_visual_rows(
                inputs_embeds, visual_mask, image_features, add=False
            )

        cache_offset = int(cache[0].offset) if cache else 0
        if cache_offset == 0:
            position_ids, self._rope_delta = self._rope_positions(
                input_ids, image_grid_thw, attention_mask
            )
        else:
            delta = self._rope_delta
            if delta is None:
                delta = mx.zeros((input_ids.shape[0], 1), dtype=mx.int32)
            positions = mx.arange(input_ids.shape[1], dtype=mx.int32)[None, :]
            positions = (
                mx.broadcast_to(positions, input_ids.shape) + cache_offset + delta
            )
            position_ids = mx.broadcast_to(
                positions[None, ...], (3, input_ids.shape[0], input_ids.shape[1])
            )
        return self.language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            position_ids=position_ids,
            visual_mask=visual_mask,
            deepstack_features=deepstack,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Normalize source names while preserving native MLX tensor layouts."""

        sanitized = {}
        for key, value in weights.items():
            if key == "lm_head.weight" and self.config.text_config.tie_word_embeddings:
                continue
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif key.startswith("model.visual."):
                key = key.replace("model.visual", "vision_tower", 1)
            elif key.startswith("lm_head."):
                key = f"language_model.{key}"
            sanitized[key] = value
        return sanitized


__all__ = ["Model", "ModelConfig", "TextConfig", "VisionConfig"]
