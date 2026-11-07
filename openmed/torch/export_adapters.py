"""Family-aware PyTorch tracing adapters for the Core ML and ONNX exporters.

The generic exporters trace an ``(input_ids, attention_mask) -> logits``
wrapper around a Hugging Face token classifier. Families whose stock forward
pass does not trace faithfully register an :class:`ExportFamilyAdapter` here.
An adapter declares:

- how to load the model for tracing and how to wrap it;
- a scoped compatibility context active only while tracing;
- the runtime support table qualification may claim for Core ML and ONNX;
- a pre-load request check that rejects unsupported precision, provider,
  sequence, and converter-environment combinations with stable, value-free
  :class:`~openmed.eval.export_qualification.ExportReason` codes.

Families with extra graph inputs or bounded sequence variants also declare
the graph inputs per runtime and profile, the global-attention fill policies,
attention-window-aligned sequence buckets with memory estimates, the
boundary lengths a qualification sweep must cover, and whether long
documents are windowed.

ModernBERT and Longformer are registered here; later families (Privacy
Filter, GLiNER) register another :class:`ExportFamilyAdapter`.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable, ContextManager, Iterable, Iterator, Mapping

from openmed.eval.export_qualification import (
    GLOBAL_ATTENTION_FIRST_TOKEN,
    GLOBAL_ATTENTION_INPUT,
    GLOBAL_ATTENTION_NONE,
    QUANTIZATION_NONE,
    SHAPE_MODE_DYNAMIC,
    SHAPE_MODE_STATIC,
    ExportQualificationError,
    ExportReason,
    RuntimeSupport,
)

COREML_RUNTIME = "coreml"
ONNX_RUNTIME = "onnx"
ANDROID_PROFILE = "android"
BASE_INPUT_NAMES = ("input_ids", "attention_mask")
MODERNBERT_FAMILY = "modernbert"
MODERNBERT_ADAPTER_VERSION = 1
MODERNBERT_MODEL_TYPE_ALIASES = ("modernbert", "modern-bert", "modern_bert")
MODERNBERT_MIN_TRANSFORMERS_MAJOR = 5
# Additive mask value for blocked keys. It is finite in float16, so masked
# scores never become ``0 * -inf = NaN`` after Core ML fp16 conversion, and
# ``exp(-1e4)`` is exactly zero in both float16 and float32 softmax.
MODERNBERT_MASK_VALUE = -1.0e4

MODERNBERT_COREML_SUPPORT = RuntimeSupport(
    runtime=COREML_RUNTIME,
    precisions=("float32", "float16"),
    quantizations=(QUANTIZATION_NONE, "int8", "int4"),
    providers=("all", "cpuAndNeuralEngine", "cpuOnly"),
    shape_modes=(SHAPE_MODE_STATIC,),
    target="iOS16",
)
MODERNBERT_ONNX_SUPPORT = RuntimeSupport(
    runtime=ONNX_RUNTIME,
    precisions=("float32", "float16"),
    quantizations=(QUANTIZATION_NONE, "int8"),
    providers=("CPUExecutionProvider",),
    shape_modes=(SHAPE_MODE_DYNAMIC,),
    target="opset>=18",
)


@dataclass(frozen=True)
class SequenceLimits:
    """Declared sequence buckets and memory estimates for one export target.

    Attributes:
        runtime: Runtime the limits apply to.
        precision: Compute precision the memory estimate assumes.
        buckets: Every declared (attention-window-aligned) bucket up to the
            checkpoint's position limit, ascending.
        position_limit: Largest length the position-embedding table allows.
        attention_windows: Full attention window of each encoder layer.
        memory_ceiling_bytes: Declared ceiling for the estimated working set.
        bucket_memory_bytes: Estimated attention working set per bucket.
        memory_estimate: Name of the estimate formula, for manifests.
    """

    runtime: str
    precision: str
    buckets: tuple[int, ...]
    position_limit: int
    attention_windows: tuple[int, ...]
    memory_ceiling_bytes: int
    bucket_memory_bytes: Mapping[int, int] = field(default_factory=dict)
    memory_estimate: str = "attention_working_set"

    @property
    def supported_buckets(self) -> tuple[int, ...]:
        """Return the declared buckets that fit the memory ceiling."""
        return tuple(
            bucket
            for bucket in self.buckets
            if self.bucket_memory_bytes.get(bucket, 0) <= self.memory_ceiling_bytes
        )

    @property
    def max_supported_length(self) -> int | None:
        """Return the largest bucket that fits the memory ceiling."""
        supported = self.supported_buckets
        return supported[-1] if supported else None

    def check(self, length: int) -> None:
        """Reject a requested bucket before any weights load.

        Raises:
            ExportQualificationError: ``unsupported_sequence_length`` past the
                position limit, ``bucket_unavailable`` for a length that is not
                a declared bucket, or ``memory_budget_exceeded`` when the
                bucket's estimated working set exceeds the ceiling.
        """

        value = int(length)
        if value <= 0 or value > self.position_limit:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_SEQUENCE_LENGTH)
        if value not in self.buckets:
            raise ExportQualificationError(ExportReason.BUCKET_UNAVAILABLE)
        if self.bucket_memory_bytes.get(value, 0) > self.memory_ceiling_bytes:
            raise ExportQualificationError(ExportReason.MEMORY_BUDGET_EXCEEDED)

    def to_dict(self, *, selected: int | None = None) -> dict[str, Any]:
        """Return the value-free manifest form, optionally for one bound."""
        payload: dict[str, Any] = {
            "runtime": self.runtime,
            "precision": self.precision,
            "declared_buckets": list(self.buckets),
            "supported_buckets": list(self.supported_buckets),
            "position_limit": self.position_limit,
            "attention_windows": list(self.attention_windows),
            "memory_ceiling_bytes": self.memory_ceiling_bytes,
            "memory_estimate": self.memory_estimate,
            "estimated_bytes_by_bucket": {
                str(bucket): int(self.bucket_memory_bytes.get(bucket, 0))
                for bucket in self.buckets
            },
        }
        if selected is not None:
            payload["selected_max_sequence_length"] = int(selected)
            payload["selected_estimated_bytes"] = int(
                self.bucket_memory_bytes.get(int(selected), 0)
            )
        return payload


def _default_graph_inputs(runtime: str, profile: str | None) -> tuple[str, ...]:
    del runtime, profile
    return BASE_INPUT_NAMES


def _no_sequence_limits(config: Any, runtime: str, precision: str) -> None:
    del config, runtime, precision
    return None


@dataclass(frozen=True)
class ExportFamilyAdapter:
    """Tracing recipe and declared runtime support for one model family.

    Attributes:
        family: Canonical family name recorded in manifests.
        version: Adapter version; bump when the traced graph changes.
        model_types: Hugging Face ``model_type`` spellings handled here.
        support: Runtime name to declared :class:`RuntimeSupport`.
        prepare_model: Configures a loaded model for tracing in place, for
            example by selecting eager attention.
        build_wrapper: ``(torch_module, model) -> torch.nn.Module`` returning
            logits for the inputs named by :attr:`graph_inputs`.
        trace_context: ``(torch_module, model) -> context manager`` active
            only while tracing.
        check_environment: Raises when installed converters are unqualified.
        max_sequence_length: Returns the largest qualifiable length for a
            configuration.
        graph_inputs: ``(runtime, profile) -> input names`` in feed order.
        global_attention_policies: Declared ``global_attention_mask`` fill
            policies; the first is the default. Empty for families without
            global attention.
        sequence_limits: ``(config, runtime, precision) -> SequenceLimits``
            or ``None`` when any length up to the maximum is allowed.
        sweep_boundaries: ``config -> lengths`` (for example attention
            windows) a qualification sweep must cover, or ``None`` to skip the
            sweep.
        windowed_documents: Whether qualification checks long documents
            through overlapping windows.
        forbidden_onnx_operators: ``domain:op_type`` names that must not
            appear in an exported ONNX graph (for example data-dependent
            gathers the adapter is meant to remove); a graph that uses one
            fails with ``unsupported_operator`` before qualification runs.
    """

    family: str
    version: int
    model_types: tuple[str, ...]
    support: Mapping[str, RuntimeSupport]
    prepare_model: Callable[[Any], None]
    build_wrapper: Callable[[Any, Any], Any]
    trace_context: Callable[[Any, Any], ContextManager[None]]
    check_environment: Callable[[], None]
    max_sequence_length: Callable[[Any], int | None]
    graph_inputs: Callable[[str, str | None], tuple[str, ...]] = _default_graph_inputs
    global_attention_policies: tuple[str, ...] = ()
    sequence_limits: Callable[[Any, str, str], SequenceLimits | None] = (
        _no_sequence_limits
    )
    sweep_boundaries: Callable[[Any], tuple[int, ...]] | None = None
    windowed_documents: bool = False
    forbidden_onnx_operators: tuple[str, ...] = ()

    def inputs_for(self, runtime: str, profile: str | None = None) -> tuple[str, ...]:
        """Return the graph input names for *runtime* and *profile*."""
        return tuple(self.graph_inputs(runtime, profile))

    def resolve_global_attention(
        self,
        requested: str | None,
        *,
        runtime: str,
        profile: str | None = None,
    ) -> str | None:
        """Return the effective global attention policy for one export.

        Families without global attention return ``None``. A graph without a
        ``global_attention_mask`` input can only run the default policy, which
        is then fixed in the graph.

        Raises:
            ExportQualificationError: ``unsupported_global_attention``.
        """

        if not self.global_attention_policies:
            if requested is not None:
                raise ExportQualificationError(
                    ExportReason.UNSUPPORTED_GLOBAL_ATTENTION
                )
            return None
        default = self.global_attention_policies[0]
        policy = default if requested is None else str(requested)
        if policy not in self.global_attention_policies:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_GLOBAL_ATTENTION)
        if policy != default and GLOBAL_ATTENTION_INPUT not in self.inputs_for(
            runtime, profile
        ):
            raise ExportQualificationError(ExportReason.UNSUPPORTED_GLOBAL_ATTENTION)
        return policy

    def limits_for(
        self, config: Any, runtime: str, precision: str
    ) -> SequenceLimits | None:
        """Return declared sequence limits for one runtime and precision."""
        return self.sequence_limits(config, runtime, precision)

    def check_operators(self, operator_types: Iterable[str]) -> None:
        """Reject an exported graph that uses a forbidden operator.

        Raises:
            ExportQualificationError: ``unsupported_operator``.
        """

        forbidden = set(self.forbidden_onnx_operators)
        if forbidden & {str(item) for item in operator_types}:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_OPERATOR)

    def support_for(self, runtime: str) -> RuntimeSupport:
        """Return the declared support table for *runtime*.

        Raises:
            ExportQualificationError: ``unsupported_runtime``.
        """

        support = self.support.get(runtime)
        if support is None:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_RUNTIME)
        return support

    def validate_request(
        self,
        *,
        runtime: str,
        config: Any,
        max_sequence_length: int,
        precisions: tuple[str, ...] = (),
        quantizations: tuple[str, ...] = (),
        providers: tuple[str, ...] = (),
        shape_mode: str | None = None,
        global_attention: str | None = None,
        profile: str | None = None,
    ) -> RuntimeSupport:
        """Reject an export request before any weights are loaded.

        Raises:
            ExportQualificationError: With a stable reason code for the first
                unsupported runtime, environment, precision, quantization,
                provider, shape mode, global attention policy, sequence
                length, sequence bucket, or memory budget.
        """

        support = self.support_for(runtime)
        self.check_environment()
        for precision in precisions:
            support.validate(precision=precision)
        for quantization in quantizations:
            support.validate(quantization=quantization)
        for provider in providers:
            support.validate(provider=provider)
        support.validate(shape_mode=shape_mode)
        self.resolve_global_attention(
            global_attention, runtime=runtime, profile=profile
        )
        limit = self.max_sequence_length(config)
        if max_sequence_length <= 0 or (
            limit is not None and max_sequence_length > limit
        ):
            raise ExportQualificationError(ExportReason.UNSUPPORTED_SEQUENCE_LENGTH)
        for precision in precisions or support.precisions[:1]:
            limits = self.limits_for(config, runtime, precision)
            if limits is not None:
                limits.check(max_sequence_length)
        return support

    def describe(self) -> dict[str, Any]:
        """Return a JSON-ready description for manifests and docs."""
        payload: dict[str, Any] = {
            "family": self.family,
            "version": self.version,
            "model_types": list(self.model_types),
            "support": {
                name: support.to_dict()
                for name, support in sorted(self.support.items())
            },
        }
        if self.global_attention_policies:
            payload["global_attention_policies"] = list(self.global_attention_policies)
        return payload


_ADAPTERS: dict[str, ExportFamilyAdapter] = {}


def register_export_adapter(adapter: ExportFamilyAdapter) -> ExportFamilyAdapter:
    """Register *adapter* for each of its model types and return it."""
    for model_type in adapter.model_types:
        _ADAPTERS[_normalize(model_type)] = adapter
    return adapter


def resolve_export_adapter(model_type: str | None) -> ExportFamilyAdapter | None:
    """Return the adapter for a Hugging Face ``model_type``, if one exists."""
    if not model_type:
        return None
    return _ADAPTERS.get(_normalize(model_type))


def registered_export_families() -> tuple[str, ...]:
    """Return the canonical family names with a registered adapter."""
    return tuple(sorted({adapter.family for adapter in _ADAPTERS.values()}))


def _normalize(model_type: str) -> str:
    return str(model_type).strip().lower().replace("_", "-")


# ---------------------------------------------------------------------------
# ModernBERT
# ---------------------------------------------------------------------------


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def modernbert_sliding_window(config: Any) -> int:
    """Return the inclusive half-window of ModernBERT local attention.

    Transformers stores ``sliding_window = local_attention // 2``; a token
    attends to keys whose distance is at most this value.
    """

    window = _config_value(config, "sliding_window")
    if window is None:
        local_attention = int(_config_value(config, "local_attention", 128))
        window = local_attention // 2
    return int(window)


def _modernbert_max_sequence_length(config: Any) -> int | None:
    value = _config_value(config, "max_position_embeddings")
    return int(value) if value else None


def _check_transformers_major(minimum: int) -> None:
    try:
        version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        raise ExportQualificationError(
            ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT
        ) from None
    major = version.split(".", 1)[0]
    if not major.isdigit() or int(major) < minimum:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT)


def _check_modernbert_environment() -> None:
    # Transformers 4.x rebuilds ModernBERT masks internally, so the
    # precomputed per-layer masks below would be ignored.
    _check_transformers_major(MODERNBERT_MIN_TRANSFORMERS_MAJOR)


def _use_eager_attention(model: Any) -> None:
    """Select eager attention so masks are plain additive tensors.

    Every ModernBERT layer reads the shared config at call time, so setting
    the config attribute switches the whole model. ``set_attn_implementation``
    is not used: Transformers refuses it for ModernBERT.
    """
    config = getattr(model, "config", None)
    if config is not None:
        config._attn_implementation = "eager"


def torch_reference_runner(torch_module: Any, model: Any) -> Callable[[Any], Any]:
    """Return a logits runner for a stock PyTorch reference checkpoint.

    The runner accepts the same integer feed as the exported artifacts and
    returns float32 numpy logits. It is family-neutral.
    """

    import numpy as np

    def run(feed: Mapping[str, Any]) -> Any:
        tensors = {
            name: torch_module.as_tensor(np.asarray(value, dtype=np.int64))
            for name, value in feed.items()
        }
        with torch_module.no_grad():
            logits = model(**tensors).logits
        return logits.detach().to(torch_module.float32).cpu().numpy()

    return run


def _build_modernbert_wrapper(torch_module: Any, model: Any) -> Any:
    """Return a trace-friendly ``(input_ids, attention_mask) -> logits`` module.

    ModernBERT alternates full and bidirectional sliding-window attention.
    The stock forward builds both masks through ``masking_utils`` (vmap,
    boolean SDPA masks, and data-dependent skips) that Core ML cannot convert.
    The wrapper builds the two additive masks with plain tensor arithmetic and
    passes them through the per-layer-type mask mapping Transformers 5 accepts,
    so every layer still sees exactly the reference masking semantics.
    """

    config = model.config
    window = modernbert_sliding_window(config)
    try:
        dtype = next(model.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        dtype = getattr(torch_module, "float32", None)
    mask_value = MODERNBERT_MASK_VALUE

    class ModernBertTokenClassificationExport(torch_module.nn.Module):
        """Exportable ModernBERT token classifier returning logits only."""

        def __init__(self, base_model: Any) -> None:
            super().__init__()
            self.base_model = base_model
            self.sliding_window = window

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            keep = attention_mask.to(dtype)
            key_bias = (1.0 - keep)[:, None, None, :] * mask_value
            positions = torch_module.arange(input_ids.shape[1], device=input_ids.device)
            distance = (positions[None, :] - positions[:, None]).abs()
            outside = (distance > self.sliding_window).to(dtype)
            window_bias = outside[None, None, :, :] * mask_value
            masks = {
                "full_attention": key_bias,
                "sliding_attention": key_bias + window_bias,
            }
            output = self.base_model(input_ids=input_ids, attention_mask=masks)
            return output.logits

    wrapper = ModernBertTokenClassificationExport(model)
    wrapper.eval()
    return wrapper


@contextmanager
def _modernbert_trace_context(torch_module: Any, model: Any) -> Iterator[None]:
    """Make ModernBERT rotary embeddings trace to static slices.

    ``rotate_half`` computes ``x.shape[-1] // 2`` at run time. TorchScript
    records that as a tensor ``floor_divide`` followed by ``aten::Int``, which
    coremltools cannot fold. The head dimension is fixed by the config, so the
    patched function slices at the same constant and is numerically identical.
    The patch is restored on exit.
    """

    from transformers.models.modernbert import modeling_modernbert

    config = model.config
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    half = head_dim // 2
    original = modeling_modernbert.rotate_half

    def static_rotate_half(x: Any) -> Any:
        return torch_module.cat((-x[..., half:], x[..., :half]), dim=-1)

    modeling_modernbert.rotate_half = static_rotate_half
    try:
        yield
    finally:
        modeling_modernbert.rotate_half = original


MODERNBERT_EXPORT_ADAPTER = register_export_adapter(
    ExportFamilyAdapter(
        family=MODERNBERT_FAMILY,
        version=MODERNBERT_ADAPTER_VERSION,
        model_types=MODERNBERT_MODEL_TYPE_ALIASES,
        support={
            COREML_RUNTIME: MODERNBERT_COREML_SUPPORT,
            ONNX_RUNTIME: MODERNBERT_ONNX_SUPPORT,
        },
        prepare_model=_use_eager_attention,
        build_wrapper=_build_modernbert_wrapper,
        trace_context=_modernbert_trace_context,
        check_environment=_check_modernbert_environment,
        max_sequence_length=_modernbert_max_sequence_length,
    )
)


# ---------------------------------------------------------------------------
# Longformer
# ---------------------------------------------------------------------------

LONGFORMER_FAMILY = "longformer"
LONGFORMER_ADAPTER_VERSION = 1
LONGFORMER_MODEL_TYPE_ALIASES = ("longformer",)
LONGFORMER_MIN_TRANSFORMERS_MAJOR = 4
# Finite in float16 like the ModernBERT mask, so masked scores never produce
# ``0 * -inf = NaN`` after Core ML fp16 conversion.
LONGFORMER_MASK_VALUE = -1.0e4
# The export graph materializes local and global ``[heads, S, S]`` scores and
# probabilities per layer, so the estimate counts four such tensors.
LONGFORMER_ATTENTION_TENSORS = 4
LONGFORMER_MEMORY_CEILING_BYTES: Mapping[str, int] = {
    COREML_RUNTIME: 512 * 1024**2,
    ONNX_RUNTIME: 2 * 1024**3,
}
LONGFORMER_GLOBAL_ATTENTION_POLICIES = (
    GLOBAL_ATTENTION_NONE,
    GLOBAL_ATTENTION_FIRST_TOKEN,
)
LONGFORMER_MEMORY_ESTIMATE = "dense_local_global_attention_per_layer"
# The stock forward's data-dependent global-token gathers and control flow;
# the adapter graph never needs them, so their presence means it was bypassed.
LONGFORMER_FORBIDDEN_ONNX_OPERATORS = ("ai.onnx:NonZero", "ai.onnx:If", "ai.onnx:Loop")

LONGFORMER_COREML_SUPPORT = RuntimeSupport(
    runtime=COREML_RUNTIME,
    precisions=("float32", "float16"),
    quantizations=(QUANTIZATION_NONE, "int8", "int4"),
    providers=("all", "cpuAndNeuralEngine", "cpuOnly"),
    shape_modes=(SHAPE_MODE_STATIC,),
    target="iOS16",
)
LONGFORMER_ONNX_SUPPORT = RuntimeSupport(
    runtime=ONNX_RUNTIME,
    precisions=("float32", "float16"),
    quantizations=(QUANTIZATION_NONE, "int8"),
    providers=("CPUExecutionProvider",),
    shape_modes=(SHAPE_MODE_DYNAMIC,),
    target="opset>=18",
)


def longformer_attention_windows(config: Any) -> tuple[int, ...]:
    """Return the full attention window of every Longformer encoder layer.

    Raises:
        ExportQualificationError: ``io_contract_mismatch`` for a window list
            whose length does not match the layer count, or an odd or
            non-positive window.
    """

    layers = int(_config_value(config, "num_hidden_layers", 0) or 0)
    configured = _config_value(config, "attention_window", 512)
    if isinstance(configured, (list, tuple)):
        windows = tuple(int(value) for value in configured)
        if layers and len(windows) != layers:
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
    else:
        windows = (int(configured),) * max(layers, 1)
    if not windows or any(window <= 0 or window % 2 for window in windows):
        raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
    return windows


def longformer_position_limit(config: Any) -> int:
    """Return the longest sequence the position-embedding table can index.

    Longformer numbers positions from ``pad_token_id + 1`` (RoBERTa style),
    so ``max_position_embeddings=4098`` with ``pad_token_id=1`` allows 4096
    tokens.
    """

    positions = int(_config_value(config, "max_position_embeddings", 0) or 0)
    pad = int(_config_value(config, "pad_token_id", 0) or 0)
    return max(positions - pad - 1, 0)


def longformer_attention_bytes(config: Any, bucket: int, precision: str) -> int:
    """Estimate the per-layer attention working set of one ``[1, bucket]`` row.

    The export graph evaluates local and global attention densely, so it holds
    ``LONGFORMER_ATTENTION_TENSORS`` tensors of ``heads * bucket**2`` values.
    """

    heads = int(_config_value(config, "num_attention_heads", 1) or 1)
    element_bytes = 2 if precision == "float16" else 4
    return LONGFORMER_ATTENTION_TENSORS * heads * int(bucket) ** 2 * element_bytes


def longformer_sequence_limits(
    config: Any, runtime: str, precision: str
) -> SequenceLimits:
    """Return the declared Longformer buckets and memory bounds.

    Buckets are every multiple of the largest attention window up to the
    position limit; a bucket is supported only when its estimated attention
    working set fits the runtime's declared ceiling.
    """

    windows = longformer_attention_windows(config)
    step = max(windows)
    position_limit = longformer_position_limit(config)
    buckets = tuple(range(step, position_limit + 1, step))
    return SequenceLimits(
        runtime=runtime,
        precision=precision,
        buckets=buckets,
        position_limit=position_limit,
        attention_windows=windows,
        memory_ceiling_bytes=int(LONGFORMER_MEMORY_CEILING_BYTES.get(runtime, 0)),
        bucket_memory_bytes={
            bucket: longformer_attention_bytes(config, bucket, precision)
            for bucket in buckets
        },
        memory_estimate=LONGFORMER_MEMORY_ESTIMATE,
    )


def _longformer_graph_inputs(runtime: str, profile: str | None) -> tuple[str, ...]:
    # The Android ONNX profile keeps the two-input mobile contract, so global
    # attention is fixed to the default policy inside the graph.
    if runtime == ONNX_RUNTIME and profile == ANDROID_PROFILE:
        return BASE_INPUT_NAMES
    return (*BASE_INPUT_NAMES, GLOBAL_ATTENTION_INPUT)


def _longformer_max_sequence_length(config: Any) -> int | None:
    limit = longformer_position_limit(config)
    return limit or None


def _longformer_sweep_boundaries(config: Any) -> tuple[int, ...]:
    boundaries: set[int] = set()
    for window in longformer_attention_windows(config):
        boundaries.update({window // 2, window})
    return tuple(sorted(boundaries))


def _check_longformer_environment() -> None:
    _check_transformers_major(LONGFORMER_MIN_TRANSFORMERS_MAJOR)


def _prepare_longformer(model: Any) -> None:
    """Longformer needs no model changes; the wrapper owns the attention."""
    del model


def _longformer_self_attention(
    torch_module: Any,
    attention: Any,
    hidden: Any,
    *,
    allowed: Any,
    key_keep: Any,
    query_keep: Any,
    is_global: Any | None,
    mask_value: float,
) -> Any:
    """Dense, mask-based Longformer self-attention for one layer.

    Local queries attend to keys inside their window plus every global key,
    with the local projections. Global queries attend to every non-padding key
    with the separate global projections. Padding queries produce zeros, as in
    the reference. No shape is read at run time, so the graph traces to Core
    ML static buckets and to ONNX dynamic sequence axes alike.
    """

    heads = int(attention.num_heads)
    head_dim = int(attention.head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    def split(values: Any) -> Any:
        return values.unflatten(-1, (heads, head_dim)).transpose(1, 2)

    query = split(attention.query(hidden) * scale)
    key = split(attention.key(hidden))
    value = split(attention.value(hidden))
    scores = query @ key.transpose(-1, -2) + (1.0 - allowed) * mask_value
    output = torch_module.softmax(scores, dim=-1) @ value
    if is_global is not None:
        global_query = split(attention.query_global(hidden) * scale)
        global_key = split(attention.key_global(hidden))
        global_value = split(attention.value_global(hidden))
        global_scores = (
            global_query @ global_key.transpose(-1, -2) + (1.0 - key_keep) * mask_value
        )
        global_output = torch_module.softmax(global_scores, dim=-1) @ global_value
        selector = is_global[:, None, :, None]
        output = output + selector * (global_output - output)
    output = output * query_keep[:, None, :, None]
    return output.transpose(1, 2).flatten(2)


def _build_longformer_wrapper(torch_module: Any, model: Any) -> Any:
    """Return a trace-friendly Longformer token classifier returning logits.

    The stock forward pads to a multiple of the attention window, gathers
    global tokens with data-dependent ``nonzero`` indices, and reads
    ``.item()`` to choose a code path, none of which traces faithfully. The
    wrapper reuses every Longformer weight (embeddings, local and global
    projections, output, feed-forward, and classifier) but evaluates each
    layer's banded-plus-global attention with explicit masks. The forward is
    ``(input_ids, attention_mask, global_attention_mask=None)``; without a
    global mask the graph has local attention only, which is the stock
    token-classification behavior.
    """

    windows = longformer_attention_windows(model.config)
    try:
        dtype = next(model.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        dtype = getattr(torch_module, "float32", None)
    mask_value = LONGFORMER_MASK_VALUE

    class LongformerTokenClassificationExport(torch_module.nn.Module):
        """Exportable Longformer token classifier returning logits only."""

        def __init__(self, base_model: Any) -> None:
            super().__init__()
            self.base_model = base_model
            self.one_sided_windows = tuple(window // 2 for window in windows)

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            global_attention_mask: Any = None,
        ) -> Any:
            longformer = self.base_model.longformer
            keep = attention_mask.to(dtype)
            hidden = longformer.embeddings(
                input_ids=input_ids,
                token_type_ids=torch_module.zeros_like(input_ids),
            )
            # Positions from an integer cumulative sum: no run-time shape
            # reads, and nothing for fp16 graph conversion to rewrite.
            positions = (
                torch_module.cumsum(torch_module.ones_like(attention_mask[0]), dim=0)
                - 1
            ).to(dtype)
            offset = positions[None, :] - positions[:, None]
            key_keep = keep[:, None, None, :]
            is_global = (
                None
                if global_attention_mask is None
                else global_attention_mask.to(dtype) * keep
            )
            for layer, half in zip(longformer.encoder.layer, self.one_sided_windows):
                limit = float(half) + 0.5
                local = ((offset < limit) & (-offset < limit)).to(dtype)[None, None]
                if is_global is None:
                    allowed = local * key_keep
                else:
                    global_key = is_global[:, None, None, :]
                    allowed = (local + global_key - local * global_key) * key_keep
                attended = _longformer_self_attention(
                    torch_module,
                    layer.attention.self,
                    hidden,
                    allowed=allowed,
                    key_keep=key_keep,
                    query_keep=keep,
                    is_global=is_global,
                    mask_value=mask_value,
                )
                attention_output = layer.attention.output(attended, hidden)
                hidden = layer.output(
                    layer.intermediate(attention_output), attention_output
                )
            return self.base_model.classifier(hidden)

    wrapper = LongformerTokenClassificationExport(model)
    wrapper.eval()
    return wrapper


def _longformer_trace_context(torch_module: Any, model: Any) -> ContextManager[None]:
    del torch_module, model
    return nullcontext()


LONGFORMER_EXPORT_ADAPTER = register_export_adapter(
    ExportFamilyAdapter(
        family=LONGFORMER_FAMILY,
        version=LONGFORMER_ADAPTER_VERSION,
        model_types=LONGFORMER_MODEL_TYPE_ALIASES,
        support={
            COREML_RUNTIME: LONGFORMER_COREML_SUPPORT,
            ONNX_RUNTIME: LONGFORMER_ONNX_SUPPORT,
        },
        prepare_model=_prepare_longformer,
        build_wrapper=_build_longformer_wrapper,
        trace_context=_longformer_trace_context,
        check_environment=_check_longformer_environment,
        max_sequence_length=_longformer_max_sequence_length,
        graph_inputs=_longformer_graph_inputs,
        global_attention_policies=LONGFORMER_GLOBAL_ATTENTION_POLICIES,
        sequence_limits=longformer_sequence_limits,
        sweep_boundaries=_longformer_sweep_boundaries,
        windowed_documents=True,
        forbidden_onnx_operators=LONGFORMER_FORBIDDEN_ONNX_OPERATORS,
    )
)


__all__ = [
    "ANDROID_PROFILE",
    "BASE_INPUT_NAMES",
    "COREML_RUNTIME",
    "LONGFORMER_ADAPTER_VERSION",
    "LONGFORMER_ATTENTION_TENSORS",
    "LONGFORMER_COREML_SUPPORT",
    "LONGFORMER_EXPORT_ADAPTER",
    "LONGFORMER_FORBIDDEN_ONNX_OPERATORS",
    "LONGFORMER_FAMILY",
    "LONGFORMER_GLOBAL_ATTENTION_POLICIES",
    "LONGFORMER_MASK_VALUE",
    "LONGFORMER_MEMORY_CEILING_BYTES",
    "LONGFORMER_ONNX_SUPPORT",
    "MODERNBERT_ADAPTER_VERSION",
    "MODERNBERT_COREML_SUPPORT",
    "MODERNBERT_EXPORT_ADAPTER",
    "MODERNBERT_FAMILY",
    "MODERNBERT_MASK_VALUE",
    "MODERNBERT_ONNX_SUPPORT",
    "ONNX_RUNTIME",
    "ExportFamilyAdapter",
    "SequenceLimits",
    "longformer_attention_bytes",
    "longformer_attention_windows",
    "longformer_position_limit",
    "longformer_sequence_limits",
    "modernbert_sliding_window",
    "register_export_adapter",
    "registered_export_families",
    "resolve_export_adapter",
    "torch_reference_runner",
]
