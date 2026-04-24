"""OpenAI Privacy Filter implemented in Apple MLX."""

from __future__ import annotations

import math
from typing import Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError(
        "MLX is required for this module. Install with: pip install openmed[mlx]"
    )


def _param_dtype(config: dict) -> mx.Dtype:
    dtype = str(config.get("param_dtype") or config.get("dtype") or "bfloat16").lower()
    if dtype in {"bf16", "bfloat16"}:
        return mx.bfloat16
    return mx.float32


def _linear_input(x: mx.array, layer: nn.Module) -> mx.array:
    """Cast inputs for normal Linear layers, but not for quantized weights.

    MLX quantized linear weights are packed ``uint32`` arrays. Casting
    activations to that storage dtype silently breaks inference, so quantized
    modules must receive floating-point inputs.
    """
    if hasattr(layer, "scales"):
        return x
    weight = getattr(layer, "weight", None)
    if weight is None:
        return x
    return x.astype(weight.dtype)


class PrivacyFilterRMSNorm(nn.Module):
    """RMSNorm with the original ``scale`` parameter name."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = mx.ones((hidden_size,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        t = x.astype(mx.float32)
        t = t * mx.rsqrt(mx.mean(t * t, axis=-1, keepdims=True) + self.eps)
        return (t * self.scale).astype(dtype)


def _apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    dtype = x.dtype
    cos = cos[None, :, None, :].astype(x.dtype)
    sin = sin[None, :, None, :].astype(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return mx.stack([out1, out2], axis=-1).reshape(x.shape).astype(dtype)


class PrivacyFilterRotaryEmbedding:
    """RoPE cache computation with the YaRN-style scaling used by OPF."""

    def __init__(
        self,
        *,
        head_dim: int,
        base: float,
        initial_context_length: int,
        scaling_factor: float,
        ntk_alpha: float,
        ntk_beta: float,
    ) -> None:
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.initial_context_length = int(initial_context_length)
        self.scaling_factor = float(scaling_factor)
        self.ntk_alpha = float(ntk_alpha)
        self.ntk_beta = float(ntk_beta)

    def _cos_sin(self, num_tokens: int) -> tuple[mx.array, mx.array]:
        freq = self.base ** (
            mx.arange(0, self.head_dim, 2, dtype=mx.float32) / float(self.head_dim)
        )
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            d_half = self.head_dim / 2.0
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (mx.arange(int(d_half), dtype=mx.float32) - low) / (high - low)
            mask = 1.0 - mx.clip(ramp, 0.0, 1.0)
            inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        positions = mx.arange(num_tokens, dtype=mx.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        return mx.cos(freqs) * concentration, mx.sin(freqs) * concentration

    def __call__(self, query: mx.array, key: mx.array) -> tuple[mx.array, mx.array]:
        cos, sin = self._cos_sin(query.shape[1])
        return _apply_rotary_emb(query, cos, sin), _apply_rotary_emb(key, cos, sin)


def _topk(values: mx.array, k: int) -> tuple[mx.array, mx.array]:
    indices = mx.argpartition(-values, kth=k - 1, axis=-1)[..., :k]
    top_values = mx.take_along_axis(values, indices, axis=-1)
    order = mx.argsort(-top_values, axis=-1)
    indices = mx.take_along_axis(indices, order, axis=-1).astype(mx.int32)
    top_values = mx.take_along_axis(top_values, order, axis=-1)
    return top_values, indices


def _swiglu(x: mx.array, *, alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"SwiGLU expects an even final dimension, got {x.shape[-1]}.")
    half = x.shape[-1] // 2
    x_glu = mx.minimum(x[..., :half], limit)
    x_linear = mx.clip(x[..., half:], -limit, limit)
    return (x_glu / (1.0 + mx.exp(-alpha * x_glu))) * (x_linear + 1.0)


class PrivacyFilterExpertLinear(nn.Module):
    """Batched expert linear layer with OPF's ``[experts, in, out]`` weights."""

    def __init__(self, num_experts: int, in_features: int, out_features: int, dtype: mx.Dtype) -> None:
        super().__init__()
        self.weight = mx.zeros((num_experts, in_features, out_features), dtype=dtype)
        self.bias = mx.zeros((num_experts, out_features), dtype=dtype)

    def __call__(self, x: mx.array, expert_indices: mx.array) -> mx.array:
        weight = mx.take(self.weight, expert_indices, axis=0)
        bias = mx.take(self.bias, expert_indices, axis=0)
        out = mx.matmul(x[..., None, :].astype(weight.dtype), weight).squeeze(-2)
        return out + bias

    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
    ) -> "PrivacyFilterQuantizedExpertLinear":
        return PrivacyFilterQuantizedExpertLinear.from_expert_linear(
            self,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )


class PrivacyFilterQuantizedExpertLinear(nn.Module):
    """Quantized expert linear layer using packed ``[experts, out, in]`` weights."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
    ) -> None:
        super().__init__()
        self.group_size = 64 if group_size is None else int(group_size)
        self.bits = 4 if bits is None else int(bits)
        self.mode = mode
        packed_features = (in_features * self.bits) // 32
        scale_groups = in_features // self.group_size
        self.weight = mx.zeros((num_experts, out_features, packed_features), dtype=mx.uint32)
        self.scales = mx.ones((num_experts, out_features, scale_groups), dtype=mx.float16)
        self.biases = mx.zeros((num_experts, out_features, scale_groups), dtype=mx.float16)
        self.bias = mx.zeros((num_experts, out_features), dtype=mx.float16)
        self.freeze()

    def __call__(self, x: mx.array, expert_indices: mx.array) -> mx.array:
        input_shape = x.shape
        flat_x = x.reshape(-1, 1, input_shape[-1])
        flat_indices = expert_indices.reshape(-1).astype(mx.int32)
        out = mx.gather_qmm(
            flat_x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=flat_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        ).squeeze(-2)
        if "bias" in self:
            out = out + mx.take(self["bias"], flat_indices, axis=0)
        return out.reshape(*input_shape[:-1], out.shape[-1])

    @classmethod
    def from_expert_linear(
        cls,
        expert_layer: PrivacyFilterExpertLinear,
        *,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
    ) -> "PrivacyFilterQuantizedExpertLinear":
        num_experts, in_features, out_features = expert_layer.weight.shape
        quantized = cls(
            num_experts,
            in_features,
            out_features,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        transposed_weight = mx.swapaxes(expert_layer.weight, -1, -2)
        quantized.weight, quantized.scales, *biases = mx.quantize(
            transposed_weight,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        quantized.biases = biases[0] if biases else None
        quantized.bias = expert_layer.bias
        return quantized


def _as_bool_attention_mask(attention_mask: Optional[mx.array], shape: tuple[int, int]) -> Optional[mx.array]:
    if attention_mask is None:
        return None
    if attention_mask.shape != shape:
        raise ValueError(
            f"attention_mask shape mismatch: expected {shape}, got {attention_mask.shape}."
        )
    return attention_mask.astype(mx.bool_)


def _local_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    sinks: mx.array,
    *,
    left_context: int,
    right_context: int,
    attention_mask: Optional[mx.array],
) -> mx.array:
    """Run bidirectional local attention with OPF sink logits."""
    batch_size, num_tokens, num_kv_heads, query_mult, head_dim = query.shape
    window = left_context + right_context + 1
    key_padded = mx.pad(key, [(0, 0), (left_context, right_context), (0, 0), (0, 0)])
    value_padded = mx.pad(value, [(0, 0), (left_context, right_context), (0, 0), (0, 0)])
    padded_tokens = num_tokens + left_context + right_context
    strides = (
        padded_tokens * num_kv_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
        head_dim,
        1,
    )
    key_windows = mx.as_strided(
        key_padded,
        shape=(batch_size, num_tokens, window, num_kv_heads, head_dim),
        strides=strides,
    )
    value_windows = mx.as_strided(
        value_padded,
        shape=(batch_size, num_tokens, window, num_kv_heads, head_dim),
        strides=strides,
    )

    scores = mx.einsum("bthqd,btwhd->bthqw", query, key_windows).astype(mx.float32)
    offsets = mx.arange(window, dtype=mx.int32) - int(left_context)
    positions = mx.arange(num_tokens, dtype=mx.int32)[:, None] + offsets[None, :]
    valid = (positions >= 0) & (positions < num_tokens)
    score_valid = valid[None, :, None, None, :]

    if attention_mask is not None:
        mask_padded = mx.pad(
            attention_mask,
            [(0, 0), (left_context, right_context)],
            constant_values=False,
        )
        mask_windows = mx.as_strided(
            mask_padded,
            shape=(batch_size, num_tokens, window),
            strides=(padded_tokens, 1, 1),
        )
        score_valid = score_valid & mask_windows[:, :, None, None, :]

    scores = mx.where(score_valid, scores, mx.array(-1e9, dtype=scores.dtype))
    sink_scores = (sinks * math.log(2.0)).reshape(num_kv_heads, query_mult)
    sink_scores = mx.broadcast_to(
        sink_scores[None, None, :, :, None],
        (batch_size, num_tokens, num_kv_heads, query_mult, 1),
    )
    weights = mx.softmax(mx.concatenate([scores, sink_scores], axis=-1), axis=-1)[..., :-1]
    attn = mx.einsum("bthqw,btwhd->bthqd", weights.astype(value.dtype), value_windows)
    return attn.reshape(batch_size, num_tokens, num_kv_heads * query_mult * head_dim)


class PrivacyFilterAttentionBlock(nn.Module):
    """Attention block with RMSNorm, fused QKV, RoPE, local GQA, and sinks."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        dtype = _param_dtype(config)
        hidden_size = int(config["hidden_size"])
        self.head_dim = int(config["head_dim"])
        self.num_attention_heads = int(config["num_attention_heads"])
        self.num_key_value_heads = int(config["num_key_value_heads"])
        self.query_mult = self.num_attention_heads // self.num_key_value_heads
        self.left_context = int(config.get("bidirectional_left_context", config.get("sliding_window", 0)))
        self.right_context = int(config.get("bidirectional_right_context", 0))
        self.norm = PrivacyFilterRMSNorm(hidden_size, eps=float(config.get("rms_norm_eps", 1e-5)))
        qkv_dim = self.head_dim * (self.num_attention_heads + 2 * self.num_key_value_heads)
        self.qkv = nn.Linear(hidden_size, qkv_dim)
        self.out = nn.Linear(self.head_dim * self.num_attention_heads, hidden_size)
        self.sinks = mx.zeros((self.num_attention_heads,), dtype=mx.float32)
        self.qk_scale = 1.0 / math.sqrt(math.sqrt(self.head_dim))
        self.rope = PrivacyFilterRotaryEmbedding(
            head_dim=self.head_dim,
            base=float(config.get("rope_theta", 150000.0)),
            initial_context_length=int(config.get("initial_context_length", 4096)),
            scaling_factor=float(config.get("rope_scaling_factor", 1.0)),
            ntk_alpha=float(config.get("rope_ntk_alpha", 1.0)),
            ntk_beta=float(config.get("rope_ntk_beta", 32.0)),
        )
        self.qkv.weight = self.qkv.weight.astype(dtype)
        self.qkv.bias = self.qkv.bias.astype(dtype)
        self.out.weight = self.out.weight.astype(dtype)
        self.out.bias = self.out.bias.astype(dtype)

    def __call__(self, x: mx.array, *, attention_mask: Optional[mx.array] = None) -> mx.array:
        batch_size, num_tokens, _ = x.shape
        t = _linear_input(self.norm(x), self.qkv)
        qkv = self.qkv(t)
        q_end = self.num_attention_heads * self.head_dim
        k_end = q_end + self.num_key_value_heads * self.head_dim
        q = qkv[:, :, :q_end].reshape(batch_size, num_tokens, self.num_attention_heads, self.head_dim)
        k = qkv[:, :, q_end:k_end].reshape(batch_size, num_tokens, self.num_key_value_heads, self.head_dim)
        v = qkv[:, :, k_end:].reshape(batch_size, num_tokens, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        q = (q * self.qk_scale).reshape(
            batch_size,
            num_tokens,
            self.num_key_value_heads,
            self.query_mult,
            self.head_dim,
        )
        k = k * self.qk_scale
        attn_out = _local_attention(
            q,
            k,
            v,
            self.sinks,
            left_context=self.left_context,
            right_context=self.right_context,
            attention_mask=attention_mask,
        )
        return x + self.out(_linear_input(attn_out, self.out)).astype(x.dtype)


class PrivacyFilterMLPBlock(nn.Module):
    """Sparse MoE feed-forward block used by Privacy Filter."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        dtype = _param_dtype(config)
        hidden_size = int(config["hidden_size"])
        intermediate_size = int(config["intermediate_size"])
        num_experts = int(config["num_experts"])
        self.experts_per_token = int(config["experts_per_token"])
        self.swiglu_limit = float(config.get("swiglu_limit", 7.0))
        self.norm = PrivacyFilterRMSNorm(hidden_size, eps=float(config.get("rms_norm_eps", 1e-5)))
        self.gate = nn.Linear(hidden_size, num_experts)
        self.swiglu = PrivacyFilterExpertLinear(
            num_experts,
            hidden_size,
            intermediate_size * 2,
            dtype,
        )
        self.out = PrivacyFilterExpertLinear(
            num_experts,
            intermediate_size,
            hidden_size,
            dtype,
        )
        self.gate.weight = self.gate.weight.astype(dtype)
        self.gate.bias = self.gate.bias.astype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        batch_shape = x.shape[:-1]
        hidden_size = x.shape[-1]
        t = self.norm(x).reshape(-1, hidden_size)
        gate_logits = self.gate(_linear_input(t, self.gate)).astype(mx.float32)
        expert_values, expert_indices = _topk(gate_logits, self.experts_per_token)
        expert_weights = mx.softmax(expert_values, axis=-1) / float(self.experts_per_token)
        expanded = mx.broadcast_to(
            _linear_input(t, self.swiglu)[:, None, :],
            (t.shape[0], self.experts_per_token, hidden_size),
        )
        hidden = self.swiglu(expanded, expert_indices).astype(mx.float32)
        hidden = _swiglu(hidden, limit=self.swiglu_limit)
        out = self.out(_linear_input(hidden, self.out), expert_indices).astype(mx.float32)
        out = mx.sum(out * expert_weights[..., None], axis=1) * float(self.experts_per_token)
        return x + out.reshape(*batch_shape, hidden_size).astype(x.dtype)


class PrivacyFilterTransformerBlock(nn.Module):
    """One attention block followed by one MoE block."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.attn = PrivacyFilterAttentionBlock(config)
        self.mlp = PrivacyFilterMLPBlock(config)

    def __call__(self, x: mx.array, *, attention_mask: Optional[mx.array] = None) -> mx.array:
        x = self.attn(x, attention_mask=attention_mask)
        return self.mlp(x)


class OpenAIPrivacyFilterForTokenClassification(nn.Module):
    """OpenAI Privacy Filter token classifier."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        dtype = _param_dtype(config)
        self.config = config
        self.embedding = nn.Embedding(int(config["vocab_size"]), int(config["hidden_size"]))
        self.embedding.weight = self.embedding.weight.astype(dtype)
        self.block = [
            PrivacyFilterTransformerBlock(config)
            for _ in range(int(config["num_hidden_layers"]))
        ]
        self.norm = PrivacyFilterRMSNorm(
            int(config["hidden_size"]),
            eps=float(config.get("rms_norm_eps", 1e-5)),
        )
        self.unembedding = nn.Linear(
            int(config["hidden_size"]),
            int(config["num_labels"]),
            bias=False,
        )
        self.unembedding.weight = self.unembedding.weight.astype(dtype)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        del token_type_ids
        if input_ids.ndim != 2:
            raise ValueError(
                "OpenAIPrivacyFilterForTokenClassification expects input_ids "
                "with shape [batch, tokens]."
            )
        attention_mask = _as_bool_attention_mask(attention_mask, input_ids.shape)
        x = self.embedding(input_ids)
        for block in self.block:
            x = block(x, attention_mask=attention_mask)
        x = self.norm(x)
        return self.unembedding(_linear_input(x, self.unembedding))
