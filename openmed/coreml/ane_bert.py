"""BERT token-classification graph rewritten for Apple Neural Engine.

The Hugging Face BERT encoder is NCHW-unfriendly: ``nn.Linear`` becomes a
matmul, LayerNorm runs over the last dim, and attention materializes
``(batch, heads, seq, seq)`` in channels-last layout. The ANE's fast path is
``BC1S`` plus 1x1 convolution (see Apple's ml-ane-transformers and the M4 ANE
write-ups). This module copies a trained ``BertForTokenClassification`` into
that layout so CoreML can keep the encoder on the Neural Engine.

The public ``forward`` still takes ``input_ids`` / ``attention_mask`` and
returns ``(batch, seq, labels)`` logits so existing CoreML I/O is unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# fp16 ANE cannot host the 1e-12 eps used by stock BERT LayerNorm.
ANE_LAYER_NORM_EPS = 1e-7
# fp16 softmax saturates at ~1e4, not the 1e9 BERT uses on CPU/GPU.
ANE_ATTENTION_MASK_VALUE = -1e4


def _conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)


def _linear_weight_to_conv(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 linear weight, got {tuple(weight.shape)}")
    return weight.detach().to(dtype=torch.float32).contiguous()[:, :, None, None]


def _copy_linear_to_conv(conv: nn.Conv2d, linear: nn.Module) -> None:
    conv.weight.data.copy_(_linear_weight_to_conv(linear.weight))
    if conv.bias is None or getattr(linear, "bias", None) is None:
        raise ValueError("ANE conv replacement requires a bias on both modules")
    conv.bias.data.copy_(linear.bias.detach().to(dtype=torch.float32))


class LayerNormANE(nn.Module):
    """Channel-first LayerNorm: ``(x + bias) * weight`` over BC1S."""

    def __init__(self, num_channels: int, eps: float = ANE_LAYER_NORM_EPS) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = float(max(eps, ANE_LAYER_NORM_EPS))
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def load_from_layer_norm(self, layer_norm: nn.Module) -> None:
        weight = layer_norm.weight.detach().to(dtype=torch.float32)
        bias = layer_norm.bias.detach().to(dtype=torch.float32)
        # Invert nn.LayerNorm's ``x * weight + bias`` into ``(x + bias) * weight``.
        self.weight.data.copy_(weight)
        self.bias.data.copy_(bias / weight.clamp(min=1e-6))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, channels, 1, seq)
        mean = inputs.mean(dim=1, keepdim=True)
        centered = inputs - mean
        denom = (centered * centered).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        normed = centered * denom
        scale = self.weight.view(1, self.num_channels, 1, 1)
        shift = self.bias.view(1, self.num_channels, 1, 1)
        return (normed + shift) * scale


class ANEBertEmbeddings(nn.Module):
    """Word/position/type lookup, then BC1S LayerNorm."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        self.word_embeddings = nn.Embedding(int(config.vocab_size), hidden)
        self.position_embeddings = nn.Embedding(
            int(config.max_position_embeddings), hidden
        )
        self.token_type_embeddings = nn.Embedding(int(config.type_vocab_size), hidden)
        self.LayerNorm = LayerNormANE(hidden, eps=float(config.layer_norm_eps))
        self.register_buffer(
            "position_ids",
            torch.arange(int(config.max_position_embeddings)).unsqueeze(0),
            persistent=False,
        )

    def load_from_hf(self, embeddings: nn.Module) -> None:
        self.word_embeddings.weight.data.copy_(
            embeddings.word_embeddings.weight.detach()
        )
        self.position_embeddings.weight.data.copy_(
            embeddings.position_embeddings.weight.detach()
        )
        self.token_type_embeddings.weight.data.copy_(
            embeddings.token_type_embeddings.weight.detach()
        )
        self.LayerNorm.load_from_layer_norm(embeddings.LayerNorm)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        position_ids = self.position_ids[:, :seq].expand(batch, seq)
        token_type_ids = torch.zeros_like(input_ids)
        hidden = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        # (B, S, C) -> (B, C, 1, S)
        hidden = hidden.transpose(1, 2).unsqueeze(2)
        return self.LayerNorm(hidden)


class ANEBertSelfAttention(nn.Module):
    """Scaled dot-product attention using 1x1 conv projections and per-head einsum."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        heads = int(config.num_attention_heads)
        if hidden % heads != 0:
            raise ValueError(f"hidden_size {hidden} is not divisible by heads {heads}")
        self.n_heads = heads
        self.head_dim = hidden // heads
        self.scale = float(self.head_dim) ** -0.5
        self.query = _conv1x1(hidden, hidden)
        self.key = _conv1x1(hidden, hidden)
        self.value = _conv1x1(hidden, hidden)
        self.out = _conv1x1(hidden, hidden)

    def load_from_hf(self, attention: nn.Module) -> None:
        self_attn = attention.self
        output = attention.output
        _copy_linear_to_conv(self.query, self_attn.query)
        _copy_linear_to_conv(self.key, self_attn.key)
        _copy_linear_to_conv(self.value, self_attn.value)
        _copy_linear_to_conv(self.out, output.dense)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        mh_q = query.split(self.head_dim, dim=1)
        mh_k = key.transpose(1, 3).split(self.head_dim, dim=3)
        mh_v = value.split(self.head_dim, dim=1)
        context = []
        for qi, ki, vi in zip(mh_q, mh_k, mh_v):
            scores = torch.einsum("bchq,bkhc->bkhq", qi, ki) * self.scale
            if mask is not None:
                scores = scores + mask
            weights = torch.softmax(scores, dim=1)
            context.append(torch.einsum("bkhq,bchk->bchq", weights, vi))
        return self.out(torch.cat(context, dim=1))


class ANEBertLayer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        intermediate = int(config.intermediate_size)
        self.attention = ANEBertSelfAttention(config)
        self.attention_norm = LayerNormANE(hidden, eps=float(config.layer_norm_eps))
        self.ffn_in = _conv1x1(hidden, intermediate)
        self.ffn_out = _conv1x1(intermediate, hidden)
        self.output_norm = LayerNormANE(hidden, eps=float(config.layer_norm_eps))
        self.hidden_act = str(getattr(config, "hidden_act", "gelu"))

    def load_from_hf(self, layer: nn.Module) -> None:
        self.attention.load_from_hf(layer.attention)
        self.attention_norm.load_from_layer_norm(layer.attention.output.LayerNorm)
        _copy_linear_to_conv(self.ffn_in, layer.intermediate.dense)
        _copy_linear_to_conv(self.ffn_out, layer.output.dense)
        self.output_norm.load_from_layer_norm(layer.output.LayerNorm)

    def _activate(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.hidden_act in {"gelu", "gelu_new", "gelu_fast"}:
            return F.gelu(tensor)
        if self.hidden_act == "relu":
            return F.relu(tensor)
        raise ValueError(f"unsupported hidden_act {self.hidden_act!r} for ANE BERT")

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        attn = self.attention(hidden, mask)
        hidden = self.attention_norm(hidden + attn)
        ff = self.ffn_out(self._activate(self.ffn_in(hidden)))
        return self.output_norm(hidden + ff)


class ANEBertForTokenClassification(nn.Module):
    """Drop-in logits wrapper for CoreML conversion of BERT token classifiers."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embeddings = ANEBertEmbeddings(config)
        self.encoder = nn.ModuleList(
            [ANEBertLayer(config) for _ in range(int(config.num_hidden_layers))]
        )
        self.classifier = _conv1x1(int(config.hidden_size), int(config.num_labels))

    @classmethod
    def from_hf(cls, model: nn.Module) -> ANEBertForTokenClassification:
        """Copy weights from a Hugging Face ``BertForTokenClassification``."""

        if not hasattr(model, "bert"):
            raise TypeError("ANE BERT wrapper requires a model with a .bert encoder")
        wrapper = cls(model.config)
        wrapper.embeddings.load_from_hf(model.bert.embeddings)
        src_layers = model.bert.encoder.layer
        if len(src_layers) != len(wrapper.encoder):
            raise ValueError("encoder layer count mismatch")
        for dest, src in zip(wrapper.encoder, src_layers):
            dest.load_from_hf(src)
        _copy_linear_to_conv(wrapper.classifier, model.classifier)
        wrapper.eval()
        for parameter in wrapper.parameters():
            parameter.requires_grad_(False)
        return wrapper

    def _attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # (B, S) -> (B, S, 1, 1) additive mask in Apple's attention layout.
        mask = attention_mask.to(dtype=torch.float32)
        return (1.0 - mask).unsqueeze(2).unsqueeze(3) * ANE_ATTENTION_MASK_VALUE

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.embeddings(input_ids)
        mask = self._attention_mask(attention_mask)
        for layer in self.encoder:
            hidden = layer(hidden, mask)
        logits = self.classifier(hidden)
        return logits.squeeze(2).transpose(1, 2)
