"""PyTorch reference graphs for the GLiNER span and GLiClass export profiles.

OpenMed's GLiNER span and GLiClass runtimes are MLX custom families. These
modules re-express the same computation in PyTorch, parameter for parameter
and with the same configuration defaults, so that one set of converted
weights drives three paths: the MLX runtime, the PyTorch eager reference, and
the traced ONNX graph. Every data-dependent step of the MLX implementation
(label marker gathering, first-subtoken word gathering, the length-aware
backward LSTM, label span pooling, segment ids) is written as plain tensor
arithmetic, so tracing records it instead of freezing one example's shapes.

The modules return logits only:

- GLiNER span: ``[batch, words, max_width, labels]`` for the inputs
  ``input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask``
  (the tensor names used by portable GLiNER runtimes);
- GLiClass: ``[batch, labels]`` for ``input_ids, attention_mask``.

Nothing here imports PyTorch at module import time; builders take the
imported ``torch`` module, like :mod:`openmed.torch.export_adapters`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from openmed.eval.export_qualification import ExportQualificationError, ExportReason
from openmed.ner.families.zero_shot_export import (
    TASK_ZERO_SHOT_NER,
    ZeroShotModelSpec,
)

# Masked span logits match the MLX runtime exactly.
SPAN_MASK_VALUE = -1.0e9
_DEBERTA_LAYER_PREFIX = "deberta.encoder.layer."
# Inverse of openmed.mlx.convert._DEBERTA_V2_KEY_REPLACEMENTS.
_MLX_TO_HF_DEBERTA = (
    (".attention.out_proj.", ".attention.output.dense."),
    (".ln1.", ".attention.output.LayerNorm."),
    (".linear1.", ".intermediate.dense."),
    (".linear2.", ".output.dense."),
    (".ln2.", ".output.LayerNorm."),
)
_DEBERTA_CONFIG_KEYS = (
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "hidden_act",
    "max_position_embeddings",
    "type_vocab_size",
    "layer_norm_eps",
    "relative_attention",
    "max_relative_positions",
    "position_buckets",
    "pos_att_type",
    "norm_rel_ebd",
    "share_att_key",
    "position_biased_input",
    "embedding_size",
    "attention_head_size",
    "pad_token_id",
)


def deberta_config(config: Mapping[str, Any], *, hidden_size: int) -> Any:
    """Return a ``DebertaV2Config`` matching the MLX backbone configuration.

    Dropout is zeroed: the reference and the exported graph are inference
    graphs, and the MLX runtime evaluates in eval mode.
    """

    from transformers import DebertaV2Config

    values = {key: config[key] for key in _DEBERTA_CONFIG_KEYS if key in config}
    values.update(
        hidden_size=int(hidden_size),
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return DebertaV2Config(**values)


def build_zero_shot_module(
    torch_module: Any,
    config: Mapping[str, Any],
    spec: ZeroShotModelSpec,
) -> Any:
    """Return an eval-mode reference module for *spec*'s profile.

    Args:
        torch_module: The imported ``torch`` module.
        config: Normalized MLX configuration of the checkpoint.
        spec: Validated architecture facts from
            :func:`~openmed.ner.families.zero_shot_export.validate_zero_shot_config`.
    """

    classes = _module_classes(torch_module)
    if spec.profile.task == TASK_ZERO_SHOT_NER:
        module = classes["gliner"](config, spec)
    else:
        if spec.encoder_hidden_size != spec.hidden_size:
            # The MLX GLiClass backbone and heads share one hidden size.
            raise ExportQualificationError(ExportReason.UNSUPPORTED_MODEL_VARIANT)
        module = classes["gliclass"](config, spec)
    module.eval()
    return module


def mlx_weights_to_torch_state(weights: Mapping[str, Any]) -> dict[str, Any]:
    """Map MLX parameter names onto the reference module's state-dict names.

    DeBERTa layer keys get their Hugging Face names back, and MLX LSTM
    ``Wx``/``Wh``/``bias`` become PyTorch ``weight_ih_l0``/``weight_hh_l0``
    plus a fused input bias with a zero hidden bias. Every other name is
    shared by both runtimes.
    """

    import numpy as np

    state: dict[str, Any] = {}
    for key, value in weights.items():
        array = np.asarray(value)
        if key.startswith(_DEBERTA_LAYER_PREFIX):
            for mlx_name, hf_name in _MLX_TO_HF_DEBERTA:
                key = key.replace(mlx_name, hf_name)
        if ".forward_lstm." in key or ".backward_lstm." in key:
            prefix, _, name = key.rpartition(".")
            if name == "Wx":
                state[f"{prefix}.weight_ih_l0"] = array
            elif name == "Wh":
                state[f"{prefix}.weight_hh_l0"] = array
            elif name == "bias":
                state[f"{prefix}.bias_ih_l0"] = array
                state[f"{prefix}.bias_hh_l0"] = np.zeros_like(array)
            else:
                state[key] = array
            continue
        state[key] = array
    return state


def load_reference_weights(torch_module: Any, module: Any, weights: Mapping) -> None:
    """Load MLX-layout *weights* into *module*, requiring an exact match.

    Raises:
        ExportQualificationError: ``unsupported_model_variant`` when a
            parameter is missing, unexpected, or has the wrong shape.
    """

    state = mlx_weights_to_torch_state(weights)
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_MODEL_VARIANT)
    tensors = {}
    for key, value in state.items():
        tensor = torch_module.as_tensor(value, dtype=torch_module.float32)
        if tuple(tensor.shape) != tuple(expected[key].shape):
            raise ExportQualificationError(ExportReason.UNSUPPORTED_MODEL_VARIANT)
        tensors[key] = tensor
    module.load_state_dict(tensors, strict=True)


def read_mlx_weights(path: str | Path) -> dict[str, Any]:
    """Read an MLX ``weights.safetensors`` file as numpy arrays.

    Raises:
        ExportQualificationError: ``unsupported_quantization`` for a
            quantized checkpoint (it carries ``.scales`` tensors).
    """

    from safetensors.numpy import load_file

    weights = load_file(str(path))
    if any(key.endswith(".scales") for key in weights):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_QUANTIZATION)
    return dict(weights)


def zero_shot_reference_runner(
    torch_module: Any, module: Any
) -> Callable[[Mapping[str, Any]], Any]:
    """Return a runner mapping one integer feed to float32 numpy logits."""

    import numpy as np

    def run(feed: Mapping[str, Any]) -> Any:
        tensors = {}
        for name, value in feed.items():
            array = np.asarray(value)
            dtype = torch_module.bool if array.dtype == bool else torch_module.int64
            tensors[name] = torch_module.as_tensor(array, dtype=dtype)
        with torch_module.no_grad():
            logits = module(**tensors)
        return logits.detach().to(torch_module.float32).cpu().numpy()

    return run


def example_inputs(torch_module: Any, feed: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return tracing inputs for *feed* in the profile's argument order."""
    import numpy as np

    values = []
    for value in feed.values():
        array = np.asarray(value)
        dtype = torch_module.bool if array.dtype == bool else torch_module.int64
        values.append(torch_module.as_tensor(array, dtype=dtype))
    return tuple(values)


@lru_cache(maxsize=None)
def _module_classes(torch_module: Any) -> dict[str, Any]:
    """Define the reference ``nn.Module`` classes for *torch_module* once."""

    nn = torch_module.nn
    functional = torch_module.nn.functional

    class ProjectionMLP(nn.Module):
        """``linear1 -> relu -> linear2`` with a 4x hidden expansion."""

        def __init__(self, input_size: int, output_size: int) -> None:
            super().__init__()
            self.linear1 = nn.Linear(input_size, output_size * 4)
            self.linear2 = nn.Linear(output_size * 4, output_size)

        def forward(self, x: Any) -> Any:
            return self.linear2(functional.relu(self.linear1(x)))

    class BidirectionalLSTM(nn.Module):
        """Two unidirectional LSTMs; the backward pass reverses each length."""

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            half = hidden_size // 2
            self.forward_lstm = nn.LSTM(hidden_size, half, batch_first=True)
            self.backward_lstm = nn.LSTM(hidden_size, half, batch_first=True)

        @staticmethod
        def reverse_padded(x: Any, lengths: Any) -> Any:
            steps = torch_module.arange(x.shape[1], device=x.device)[None, :]
            limit = lengths[:, None]
            index = torch_module.where(steps < limit, limit - 1 - steps, steps)
            index = index[:, :, None].expand(-1, -1, x.shape[-1])
            return torch_module.gather(x, 1, index)

        def forward(self, x: Any, mask: Any) -> Any:
            lengths = mask.to(torch_module.int64).sum(dim=1)
            forward_out, _ = self.forward_lstm(x)
            reversed_x = self.reverse_padded(x, lengths)
            backward_reversed, _ = self.backward_lstm(reversed_x)
            backward_out = self.reverse_padded(backward_reversed, lengths)
            output = torch_module.cat([forward_out, backward_out], dim=-1)
            return output * mask[:, :, None].to(output.dtype)

    class SpanMarkerV0(nn.Module):
        """GLiNER ``markerV0`` span representation."""

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.project_start = ProjectionMLP(hidden_size, hidden_size)
            self.project_end = ProjectionMLP(hidden_size, hidden_size)
            self.out_project = ProjectionMLP(hidden_size * 2, hidden_size)

        def forward(self, words: Any, span_idx: Any) -> Any:
            start_rep = self.project_start(words)
            end_rep = self.project_end(words)
            size = start_rep.shape[-1]
            start_index = span_idx[:, :, 0:1].expand(-1, -1, size)
            end_index = span_idx[:, :, 1:2].expand(-1, -1, size)
            start_span = torch_module.gather(start_rep, 1, start_index)
            end_span = torch_module.gather(end_rep, 1, end_index)
            combined = functional.relu(torch_module.cat([start_span, end_span], -1))
            return self.out_project(combined)

    def marker_selection(input_ids: Any, marker_id: int) -> tuple[Any, Any, Any]:
        is_marker = input_ids == marker_id
        count = is_marker.to(torch_module.int64).sum(dim=-1)
        rank = is_marker.to(torch_module.int64).cumsum(dim=-1) - 1
        slots = torch_module.arange(count.max(), device=input_ids.device)
        select = (rank[:, None, :] == slots[None, :, None]) & is_marker[:, None, :]
        return select, count, slots

    class GLiNERSpanExport(nn.Module):
        """GLiNER uni-encoder span model returning span logits."""

        def __init__(self, config: Mapping[str, Any], spec: ZeroShotModelSpec):
            super().__init__()
            from transformers import DebertaV2Model

            hidden = spec.hidden_size
            self.deberta = DebertaV2Model(
                deberta_config(config, hidden_size=spec.encoder_hidden_size)
            )
            self.token_projection = nn.Linear(spec.encoder_hidden_size, hidden)
            self.rnn = BidirectionalLSTM(hidden)
            self.span_rep_layer = SpanMarkerV0(hidden)
            self.prompt_rep_layer = ProjectionMLP(hidden, hidden)
            self.marker_id = int(spec.marker_token_index)
            self.embed_marker = bool(spec.options.get("embed_ent_token", True))
            self.use_rnn = int(spec.options.get("num_rnn_layers", 0)) > 0
            self.span_width = int(spec.span_width or 1)

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            words_mask: Any,
            text_lengths: Any,
            span_idx: Any,
            span_mask: Any,
        ) -> Any:
            hidden = self.deberta(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            select, _, _ = marker_selection(input_ids, self.marker_id)
            source = hidden
            if not self.embed_marker:
                source = torch_module.cat([hidden[:, 1:], hidden[:, -1:]], dim=1)
            prompts = torch_module.matmul(select.to(hidden.dtype), source)

            lengths = text_lengths[:, 0]
            word_slots = torch_module.arange(lengths.max(), device=input_ids.device)
            word_select = words_mask[:, None, :] == (word_slots + 1)[None, :, None]
            words = torch_module.matmul(word_select.to(hidden.dtype), hidden)
            word_mask = word_slots[None, :] < lengths[:, None]

            words = self.token_projection(words)
            prompts = self.token_projection(prompts)
            if self.use_rnn:
                words = self.rnn(words, word_mask)

            span_idx = span_idx * span_mask[:, :, None].to(span_idx.dtype)
            span_rep = self.span_rep_layer(words, span_idx)
            prompt_rep = self.prompt_rep_layer(prompts)
            logits = torch_module.einsum("bsd,bcd->bsc", span_rep, prompt_rep)
            logits = logits.masked_fill(~span_mask[:, :, None], SPAN_MASK_VALUE)
            return logits.reshape(
                logits.shape[0], -1, self.span_width, logits.shape[-1]
            )

    class GLiClassFeaturesProjector(nn.Module):
        """``linear1 -> gelu -> linear2`` projector."""

        def __init__(self, encoder_hidden: int, hidden: int) -> None:
            super().__init__()
            self.linear1 = nn.Linear(encoder_hidden, hidden)
            self.linear2 = nn.Linear(hidden, encoder_hidden)

        def forward(self, x: Any) -> Any:
            return self.linear2(functional.gelu(self.linear1(x)))

    class GLiClassMLPScorer(nn.Module):
        """MLP scorer over ``[text, label]`` pairs."""

        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.linear1 = nn.Linear(hidden * 2, 256)
            self.linear2 = nn.Linear(256, 128)
            self.linear3 = nn.Linear(128, 1)

        def forward(self, text_rep: Any, label_rep: Any) -> Any:
            expanded = text_rep[:, None, :].expand(-1, label_rep.shape[1], -1)
            combined = torch_module.cat([expanded, label_rep], dim=-1)
            hidden = functional.relu(self.linear1(combined))
            hidden = functional.relu(self.linear2(hidden))
            return self.linear3(hidden).squeeze(-1)

    class GLiClassUniEncoderExport(nn.Module):
        """GLiClass uni-encoder returning one logit per label marker."""

        def __init__(self, config: Mapping[str, Any], spec: ZeroShotModelSpec):
            super().__init__()
            from transformers import DebertaV2Model

            encoder_hidden = spec.encoder_hidden_size
            self.deberta = DebertaV2Model(
                deberta_config(config, hidden_size=spec.hidden_size)
            )
            self.classes_projector = GLiClassFeaturesProjector(
                encoder_hidden, spec.hidden_size
            )
            self.text_projector = GLiClassFeaturesProjector(
                encoder_hidden, spec.hidden_size
            )
            self.segment_embeddings = nn.Embedding(3, encoder_hidden)
            self.scorer = GLiClassMLPScorer(encoder_hidden)
            self.logit_scale = nn.Parameter(
                torch_module.tensor(float(config.get("logit_scale_init_value", 1.0)))
            )
            options = spec.options
            self.marker_id = int(spec.marker_token_index)
            self.text_id = int(spec.text_token_index or 0)
            self.example_id = spec.example_token_index
            self.embed_marker = bool(options.get("embed_class_token", True))
            self.use_segments = bool(options.get("use_segment_embeddings"))
            self.normalize = bool(options.get("normalize_features"))
            self.mean_pool_text = options.get("pooling_strategy") == "mean"

        def first_position(self, input_ids: Any, token_id: int, dtype: Any) -> Any:
            matches = input_ids == token_id
            found = matches.any(dim=-1)
            first = matches.to(dtype).argmax(dim=-1)
            return found, first

        def segment_ids(self, input_ids: Any, dtype: Any) -> Any:
            positions = torch_module.arange(input_ids.shape[1], device=input_ids.device)
            has_text, text_start = self.first_position(input_ids, self.text_id, dtype)
            segments = has_text[:, None] & (positions[None, :] >= text_start[:, None])
            segments = segments.to(torch_module.int64)
            if self.example_id is not None:
                has_example, example_start = self.first_position(
                    input_ids, int(self.example_id), dtype
                )
                in_examples = (
                    has_text[:, None]
                    & has_example[:, None]
                    & (positions[None, :] >= example_start[:, None])
                )
                segments = torch_module.where(
                    in_examples, torch_module.full_like(segments, 2), segments
                )
            return segments

        def class_features(
            self, hidden: Any, input_ids: Any, attention_mask: Any
        ) -> Any:
            length = input_ids.shape[1]
            positions = torch_module.arange(length, device=input_ids.device)
            select, count, slots = marker_selection(input_ids, self.marker_id)
            marker_pos = (select.to(torch_module.int64) * positions).sum(dim=-1)
            has_text, text_first = self.first_position(
                input_ids, self.text_id, hidden.dtype
            )
            text_start = torch_module.where(
                has_text, text_first, torch_module.full_like(text_first, length)
            )
            following = torch_module.cat([marker_pos[:, 1:], marker_pos[:, :1]], 1)
            is_last = (slots[None, :] + 1) >= count[:, None]
            end = torch_module.where(is_last, text_start[:, None], following)
            start = marker_pos
            if not self.embed_marker:
                start = torch_module.clamp(marker_pos + 1, max=length - 1)
            grid = positions[None, None, :]
            member = (grid >= start[:, :, None]) & (grid < end[:, :, None])
            member_f = member.to(hidden.dtype)
            weights = member_f * attention_mask[:, None, :].to(hidden.dtype)
            denom = weights.sum(dim=-1, keepdim=True)
            weighted = torch_module.matmul(weights, hidden) / torch_module.clamp(
                denom, min=1e-12
            )
            plain = torch_module.matmul(member_f, hidden) / torch_module.clamp(
                member_f.sum(dim=-1, keepdim=True), min=1.0
            )
            single = torch_module.matmul(
                (grid == start[:, :, None]).to(hidden.dtype), hidden
            )
            pooled = torch_module.where(denom > 0, weighted, plain)
            empty = (start >= end)[:, :, None]
            return torch_module.where(empty, single, pooled)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            embedded = self.deberta.embeddings(input_ids=input_ids, mask=attention_mask)
            if self.use_segments:
                segments = self.segment_ids(input_ids, embedded.dtype)
                embedded = embedded + self.segment_embeddings(segments)
            hidden = self.deberta.encoder(
                embedded,
                attention_mask,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            classes = self.class_features(hidden, input_ids, attention_mask)
            if self.mean_pool_text:
                mask = attention_mask.to(hidden.dtype)
                denom = torch_module.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
                text = (hidden * mask[:, :, None]).sum(dim=1) / denom
            else:
                text = hidden[:, 0, :]
            text = self.text_projector(text)
            classes = self.classes_projector(classes)
            if self.normalize:
                text = text / torch_module.clamp(
                    torch_module.linalg.norm(text, dim=-1, keepdim=True), min=1e-8
                )
                classes = classes / torch_module.clamp(
                    torch_module.linalg.norm(classes, dim=-1, keepdim=True), min=1e-8
                )
            logits = self.scorer(text, classes)
            if self.normalize:
                logits = logits * self.logit_scale
            return logits

    return {"gliner": GLiNERSpanExport, "gliclass": GLiClassUniEncoderExport}


__all__ = [
    "SPAN_MASK_VALUE",
    "build_zero_shot_module",
    "deberta_config",
    "example_inputs",
    "load_reference_weights",
    "mlx_weights_to_torch_state",
    "read_mlx_weights",
    "zero_shot_reference_runner",
]
