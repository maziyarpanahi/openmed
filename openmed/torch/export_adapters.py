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
  :class:`~openmed.eval.export_qualification.ExportReason` codes;
- optionally, a pre-load source check (eligible checkpoints, remote code,
  configuration shape, pinned revision), a reference loader, a qualification
  binding (canonical decoder, decoder profile, default suite), and the ONNX
  operator set the family's graphs are qualified for.

The registry is versioned: every qualified artifact records its family and
adapter version, and :func:`require_registered_adapter` rejects unknown
families and stale adapter versions at load time. Families register another
:class:`ExportFamilyAdapter` (ModernBERT here, Privacy Filter in
:mod:`openmed.torch.privacy_filter_export`).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, ContextManager, Iterable, Iterator, Mapping

from openmed.eval.export_qualification import (
    QUANTIZATION_NONE,
    SHAPE_MODE_DYNAMIC,
    SHAPE_MODE_STATIC,
    ExportQualificationError,
    ExportReason,
    QualificationBinding,
    RuntimeSupport,
)

COREML_RUNTIME = "coreml"
ONNX_RUNTIME = "onnx"
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
            logits for ``(input_ids, attention_mask)``.
        trace_context: ``(torch_module, model) -> context manager`` active
            only while tracing.
        check_environment: Raises when installed converters are unqualified.
        max_sequence_length: Returns the largest qualifiable length for a
            configuration.
        check_source: Optional ``(config, *, model_id, expected_revision) ->
            dict`` pre-load check. It raises a stable reason code for
            ineligible checkpoints, repository code, unsupported configuration
            shapes, or a missing pin, and returns a value-free eligibility
            record for manifests.
        load_reference: Optional ``(torch_module, model_id, cache_dir) ->
            model`` reference loader for qualification. Defaults to a stock
            float32 ``AutoModelForTokenClassification`` without repository
            code.
        qualification_binding: Optional ``(config, model_id, cache_dir) ->
            QualificationBinding`` with the family's canonical decoder.
        onnx_operators: Optional ``domain:op`` set the family's ONNX graphs
            are qualified for; any other operator fails closed.
        coreml_fp32_ops: Optional Core ML MIL op types kept in float32 when a
            package is converted with float16 compute precision, matching
            reference sections that compute in float32.
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
    check_source: Callable[..., Mapping[str, Any]] | None = None
    load_reference: Callable[[Any, str, str | None], Any] | None = None
    qualification_binding: (
        Callable[[Any, str, str | None], QualificationBinding] | None
    ) = None
    onnx_operators: frozenset[str] | None = None
    coreml_fp32_ops: frozenset[str] | None = None

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
    ) -> RuntimeSupport:
        """Reject an export request before any weights are loaded.

        Raises:
            ExportQualificationError: With a stable reason code for the first
                unsupported runtime, environment, precision, quantization,
                provider, shape mode, or sequence length.
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
        limit = self.max_sequence_length(config)
        if max_sequence_length <= 0 or (
            limit is not None and max_sequence_length > limit
        ):
            raise ExportQualificationError(ExportReason.UNSUPPORTED_SEQUENCE_LENGTH)
        return support

    def source_eligibility(
        self,
        config: Any,
        *,
        model_id: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any] | None:
        """Run the family's pre-load source check, if it declares one.

        Returns:
            A value-free eligibility record, or ``None`` for families without
            a source check.

        Raises:
            ExportQualificationError: With the family's stable reason code.
        """

        if self.check_source is None:
            return None
        return dict(
            self.check_source(
                config, model_id=model_id, expected_revision=expected_revision
            )
        )

    def reference_model(
        self, torch_module: Any, model_id: str, cache_dir: str | None = None
    ) -> Any:
        """Load the float32 PyTorch reference checkpoint for qualification."""

        if self.load_reference is not None:
            return self.load_reference(torch_module, model_id, cache_dir)
        from transformers import AutoModelForTokenClassification

        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
        model.to(dtype=torch_module.float32)
        model.eval()
        return model

    def binding(
        self, config: Any, model_id: str, cache_dir: str | None = None
    ) -> QualificationBinding | None:
        """Return the family's qualification binding, if it declares one."""

        if self.qualification_binding is None:
            return None
        return self.qualification_binding(config, model_id, cache_dir)

    def check_onnx_operators(self, operator_types: Iterable[str]) -> None:
        """Reject an ONNX graph that uses an operator the family lacks.

        Raises:
            ExportQualificationError: ``unsupported_operator``.
        """

        if self.onnx_operators is None:
            return
        if set(operator_types) - set(self.onnx_operators):
            raise ExportQualificationError(ExportReason.UNSUPPORTED_OPERATOR)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-ready description for manifests and docs."""
        return {
            "family": self.family,
            "version": self.version,
            "model_types": list(self.model_types),
            "support": {
                name: support.to_dict()
                for name, support in sorted(self.support.items())
            },
        }


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


def export_adapter_for_family(family: str | None) -> ExportFamilyAdapter | None:
    """Return the adapter registered under canonical *family*, if any."""
    if not family:
        return None
    wanted = _normalize(family)
    for adapter in _ADAPTERS.values():
        if _normalize(adapter.family) == wanted:
            return adapter
    return None


def require_registered_adapter(
    family: str | None, adapter_version: Any
) -> ExportFamilyAdapter:
    """Return the adapter that qualified an artifact, or fail closed.

    Args:
        family: Family recorded in a qualification report or manifest.
        adapter_version: Adapter version recorded alongside it.

    Raises:
        ExportQualificationError: ``unsupported_family`` when no reviewed
            adapter is registered for *family*, or ``adapter_version_mismatch``
            when the artifact was qualified by a different adapter version.
    """

    adapter = export_adapter_for_family(family)
    if adapter is None:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_FAMILY)
    if isinstance(adapter_version, bool) or adapter_version != adapter.version:
        raise ExportQualificationError(ExportReason.ADAPTER_VERSION_MISMATCH)
    return adapter


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


def _check_modernbert_environment() -> None:
    try:
        version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        raise ExportQualificationError(
            ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT
        ) from None
    major = version.split(".", 1)[0]
    if not major.isdigit() or int(major) < MODERNBERT_MIN_TRANSFORMERS_MAJOR:
        # Transformers 4.x rebuilds ModernBERT masks internally, so the
        # precomputed per-layer masks below would be ignored.
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT)


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


# Registers the Privacy Filter adapter on import. It comes last because that
# module builds on the registry defined above; it is never read here, so an
# import that starts from that module resolves cleanly too.
from openmed.torch import privacy_filter_export  # noqa: E402, F401

__all__ = [
    "COREML_RUNTIME",
    "MODERNBERT_ADAPTER_VERSION",
    "MODERNBERT_COREML_SUPPORT",
    "MODERNBERT_EXPORT_ADAPTER",
    "MODERNBERT_FAMILY",
    "MODERNBERT_MASK_VALUE",
    "MODERNBERT_ONNX_SUPPORT",
    "ONNX_RUNTIME",
    "ExportFamilyAdapter",
    "export_adapter_for_family",
    "modernbert_sliding_window",
    "register_export_adapter",
    "registered_export_families",
    "require_registered_adapter",
    "resolve_export_adapter",
    "torch_reference_runner",
]
