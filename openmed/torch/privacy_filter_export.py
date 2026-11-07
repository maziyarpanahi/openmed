"""Core ML and ONNX export adapter for the OpenAI Privacy Filter family.

The Privacy Filter is a bidirectional token classifier with a custom
architecture: fused grouped-query attention over a banded (sliding-window)
mask with learned attention sinks, interleaved YaRN rotary embeddings, and a
top-k mixture-of-experts feed-forward block with clamped SwiGLU experts. The
generic token-classification exporters cannot trace the Transformers
implementation faithfully (its expert loop is data dependent), so this module
owns the family's export contract:

- the declared checkpoint families eligible for portable export
  (:data:`PRIVACY_FILTER_CHECKPOINTS`) and their label spaces;
- configuration normalization that rejects repository code, unknown model
  types, and unsupported configuration shapes before weights load;
- the custom weight mapping from the Transformers checkpoint to a
  trace-friendly module (fused QKV, flattened expert projections, optional
  classifier bias);
- a trace-friendly re-implementation that evaluates the router exactly like
  the reference (top-k, softmax, ``1 / k`` scaling) and computes every expert
  densely with zero routing weight for unselected experts, so the graph has no
  data-dependent control flow;
- the canonical BIOES Viterbi decoder, threshold profile, and span
  construction shared with the MLX and PyTorch pipelines, recorded in every
  exported manifest as a decoder profile.

The adapter is registered with :mod:`openmed.torch.export_adapters` under the
``openai-privacy-filter`` family. Unknown families fail closed with
``unsupported_family``; checkpoints that need repository code fail with
``unsupported_remote_code``. Export requests must name a pinned source
revision.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from importlib import metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.eval.export_qualification import (
    QUANTIZATION_NONE,
    SHAPE_MODE_DYNAMIC,
    SHAPE_MODE_STATIC,
    ExportQualificationError,
    ExportReason,
    QualificationBinding,
    RuntimeSupport,
    SpanDecoder,
    label_map_identity,
)
from openmed.torch.export_adapters import (
    COREML_RUNTIME,
    ONNX_RUNTIME,
    ExportFamilyAdapter,
    register_export_adapter,
)

PRIVACY_FILTER_FAMILY = "openai-privacy-filter"
PRIVACY_FILTER_ADAPTER_VERSION = 1
# Only the Transformers-native spelling is accepted. The original OPF
# checkpoint format (``model_type: privacy_filter``) and MLX artifacts are not
# export sources and fail closed with ``unsupported_family``.
PRIVACY_FILTER_MODEL_TYPE_ALIASES = ("openai-privacy-filter",)
PRIVACY_FILTER_NATIVE_MODULE = "transformers.models.openai_privacy_filter"
PRIVACY_FILTER_MIN_TRANSFORMERS = (5, 6)
# Dense expert evaluation keeps ``tokens * experts * 2 * intermediate``
# activations per layer, so qualified exports are bounded to 512 tokens.
PRIVACY_FILTER_MAX_EXPORT_SEQUENCE_LENGTH = 512
# Additive mask value for blocked keys; finite in float16 (see ModernBERT).
PRIVACY_FILTER_MASK_VALUE = -1.0e4
PRIVACY_FILTER_DECODER = "bioes-viterbi"
PRIVACY_FILTER_CONFIDENCE_THRESHOLD = 0.5
DECODER_PROFILE_FORMAT = "openmed-decoder-profile"
DECODER_PROFILE_SCHEMA_VERSION = 1
DECODER_PROFILE_FILENAME = "openmed-decoder-profile.json"
VITERBI_CALIBRATION_FILENAME = "viterbi_calibration.json"
PRIVACY_FILTER_QUALIFICATION_SUITE = (
    Path(__file__).resolve().parents[1]
    / "eval"
    / "fixtures"
    / "export_qualification_privacy_filter.jsonl"
)

OPENAI_PRIVACY_FILTER_LABEL_SPACE = "openai-privacy-filter-v1"
OPENAI_PRIVACY_FILTER_SPAN_LABELS = (
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)
# The upstream label names normalize to ``OTHER`` in the canonical taxonomy;
# qualification maps them so recall and leakage gates see direct identifiers.
OPENAI_PRIVACY_FILTER_LABEL_ALIASES: Mapping[str, str] = {
    "account_number": "ACCOUNT_NUMBER",
    "private_address": "STREET_ADDRESS",
    "private_date": "DATE",
    "private_email": "EMAIL",
    "private_person": "PERSON",
    "private_phone": "PHONE",
    "private_url": "URL",
    "secret": "PASSWORD",
}
_LABEL_SPACE_ALIASES: Mapping[str, Mapping[str, str]] = {
    OPENAI_PRIVACY_FILTER_LABEL_SPACE: OPENAI_PRIVACY_FILTER_LABEL_ALIASES,
}
_SUPPORTED_ROPE_TYPES = ("yarn", "default")

PRIVACY_FILTER_COREML_SUPPORT = RuntimeSupport(
    runtime=COREML_RUNTIME,
    precisions=("float32", "float16"),
    quantizations=(QUANTIZATION_NONE, "int8"),
    providers=("all", "cpuAndNeuralEngine", "cpuOnly"),
    shape_modes=(SHAPE_MODE_STATIC,),
    target="iOS16",
)
PRIVACY_FILTER_ONNX_SUPPORT = RuntimeSupport(
    runtime=ONNX_RUNTIME,
    precisions=("float32", "float16"),
    quantizations=(QUANTIZATION_NONE, "int8"),
    providers=("CPUExecutionProvider",),
    shape_modes=(SHAPE_MODE_DYNAMIC,),
    target="opset>=18",
)

# The reference computes the attention softmax in float32; fp16 Core ML
# packages keep its MIL ops in float32 too (``exp`` alone drifts by ~0.1).
PRIVACY_FILTER_COREML_FP32_OPS = frozenset({"exp", "real_div", "reduce_sum"})

# Operators the qualified Privacy Filter graphs may contain: the float graph
# from either Torch exporter, its fp16 conversion, and dynamic INT8
# quantization. Control flow, data-dependent shape operators (``Loop``,
# ``If``, ``Scan``, ``NonZero``, ``Unique``), and contrib or fused operators
# are deliberately absent; any of them fails closed.
PRIVACY_FILTER_ONNX_OPERATORS = frozenset(
    f"ai.onnx:{name}"
    for name in (
        "Abs",
        "Add",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Cos",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Exp",
        "Expand",
        "Gather",
        "Gemm",
        "Greater",
        "Identity",
        "Less",
        "MatMul",
        "MatMulInteger",
        "Max",
        "Min",
        "Mul",
        "Neg",
        "Pow",
        "Range",
        "Reciprocal",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
    )
)


# ---------------------------------------------------------------------------
# Declared checkpoint families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrivacyFilterCheckpoint:
    """One Privacy Filter checkpoint family declared eligible for export.

    Attributes:
        repo_id: Hugging Face repository id of the Transformers-format source.
        label_space: Label-space identifier (``opf_metadata.category_version``
            or :data:`OPENAI_PRIVACY_FILTER_LABEL_SPACE`).
        num_labels: Size of the BIOES token label map.
        revision: Reviewed source revision, or ``None`` while no revision has
            been reviewed. Requests must still pin a revision.
        qualified_variants: Runtime name to variant names with published
            qualification evidence. Empty until release evidence lands.
        evidence_sha256: Runtime name to the ``stable_sha256`` of the
            published qualification report backing ``qualified_variants``.
    """

    repo_id: str
    label_space: str
    num_labels: int
    revision: str | None = None
    qualified_variants: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    evidence_sha256: Mapping[str, str] = field(default_factory=dict)

    def qualified_for(self, runtime: str, variants: Sequence[str]) -> bool:
        """Return whether every name in *variants* has published evidence."""
        declared = set(self.qualified_variants.get(runtime, ()))
        return (
            self.revision is not None
            and bool(self.evidence_sha256.get(runtime))
            and bool(variants)
            and set(variants) <= declared
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready declaration."""
        return {
            "repo_id": self.repo_id,
            "label_space": self.label_space,
            "num_labels": self.num_labels,
            "revision": self.revision,
            "qualified_variants": {
                key: list(value)
                for key, value in sorted(self.qualified_variants.items())
            },
            "evidence_sha256": dict(sorted(self.evidence_sha256.items())),
        }


PRIVACY_FILTER_CHECKPOINTS: tuple[PrivacyFilterCheckpoint, ...] = (
    PrivacyFilterCheckpoint(
        "openai/privacy-filter", OPENAI_PRIVACY_FILTER_LABEL_SPACE, 33
    ),
    PrivacyFilterCheckpoint("OpenMed/privacy-filter-nemotron", "nemotron_fine_v1", 221),
    PrivacyFilterCheckpoint(
        "OpenMed/privacy-filter-nemotron-v2", "nemotron_fine_v1", 221
    ),
    PrivacyFilterCheckpoint(
        "OpenMed/privacy-filter-multilingual", "openmed_pii54_multi_v1", 217
    ),
    PrivacyFilterCheckpoint(
        "OpenMed/privacy-filter-multilingual-v2", "openmed_pii54_multi_v1", 217
    ),
)


def privacy_filter_checkpoint(repo_id: str | None) -> PrivacyFilterCheckpoint | None:
    """Return the declared checkpoint for a Hub *repo_id*, if any.

    Hub namespaces are case-insensitive, so the lookup is too.
    """

    if not repo_id:
        return None
    wanted = str(repo_id).strip().lower()
    for checkpoint in PRIVACY_FILTER_CHECKPOINTS:
        if checkpoint.repo_id.lower() == wanted:
            return checkpoint
    return None


# ---------------------------------------------------------------------------
# Configuration normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrivacyFilterArchitecture:
    """Normalized, validated Privacy Filter architecture parameters."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_experts: int
    experts_per_token: int
    sliding_window: int
    rms_norm_eps: float
    swiglu_alpha: float
    swiglu_limit: float
    attention_bias: bool
    num_labels: int
    max_position_embeddings: int
    rope_type: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready architecture description."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
            "sliding_window": self.sliding_window,
            "rms_norm_eps": self.rms_norm_eps,
            "swiglu_alpha": self.swiglu_alpha,
            "swiglu_limit": self.swiglu_limit,
            "attention_bias": self.attention_bias,
            "num_labels": self.num_labels,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_type": self.rope_type,
        }

    @property
    def sha256(self) -> str:
        """Return a digest of the normalized architecture."""
        return _canonical_digest(self.to_dict())


_MISSING = object()


def _value(config: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(config, Mapping):
            if name in config and config[name] is not None:
                return config[name]
        else:
            value = getattr(config, name, None)
            if value is not None:
                return value
    if default is _MISSING:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    return default


def _positive_int(config: Any, *names: str, default: Any = _MISSING) -> int:
    value = _value(config, *names, default=default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    return int(value)


def _finite_float(config: Any, *names: str, default: Any = _MISSING) -> float:
    value = _value(config, *names, default=default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    if not math.isfinite(float(value)):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    return float(value)


def _normalize_model_type(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def reject_remote_code(config: Any) -> None:
    """Fail closed when a configuration asks for repository code.

    Raises:
        ExportQualificationError: ``unsupported_remote_code`` when the
            configuration declares ``auto_map`` or custom pipelines.
    """

    for key in ("auto_map", "custom_pipelines"):
        if _value(config, key, default=None):
            raise ExportQualificationError(ExportReason.UNSUPPORTED_REMOTE_CODE)


def privacy_filter_architecture(config: Any) -> PrivacyFilterArchitecture:
    """Normalize and validate a Transformers Privacy Filter configuration.

    Args:
        config: A ``PretrainedConfig`` or the ``config.json`` mapping.

    Returns:
        The validated architecture.

    Raises:
        ExportQualificationError: ``unsupported_remote_code`` for repository
            code, ``unsupported_family`` for another model type, and
            ``unsupported_configuration`` for missing or unsupported shape
            parameters.
    """

    reject_remote_code(config)
    if _normalize_model_type(_value(config, "model_type", default="")) not in (
        PRIVACY_FILTER_MODEL_TYPE_ALIASES
    ):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_FAMILY)

    hidden_size = _positive_int(config, "hidden_size")
    num_attention_heads = _positive_int(config, "num_attention_heads")
    num_key_value_heads = _positive_int(
        config, "num_key_value_heads", default=num_attention_heads
    )
    head_dim = _positive_int(
        config, "head_dim", default=hidden_size // num_attention_heads or 1
    )
    num_experts = _positive_int(config, "num_local_experts", "num_experts")
    experts_per_token = _positive_int(
        config, "num_experts_per_tok", "experts_per_token"
    )
    sliding_window = _value(config, "sliding_window")
    if isinstance(sliding_window, bool) or not isinstance(sliding_window, int):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    labels = privacy_filter_labels(config)
    rope = _value(config, "rope_parameters", "rope_scaling", default={}) or {}
    if not isinstance(rope, Mapping):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    rope_type = str(rope.get("rope_type") or rope.get("type") or "default")
    partial = rope.get("partial_rotary_factor", 1.0)
    attention_bias = _value(config, "attention_bias", default=True)
    if (
        num_attention_heads % num_key_value_heads
        or head_dim % 2
        or experts_per_token > num_experts
        or sliding_window < 0
        or rope_type not in _SUPPORTED_ROPE_TYPES
        or partial not in (None, 1, 1.0)
        or not isinstance(attention_bias, bool)
    ):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    num_labels = _value(config, "num_labels", default=len(labels))
    if num_labels != len(labels):
        raise ExportQualificationError(ExportReason.LABEL_MAP_MISMATCH)
    return PrivacyFilterArchitecture(
        vocab_size=_positive_int(config, "vocab_size"),
        hidden_size=hidden_size,
        num_hidden_layers=_positive_int(config, "num_hidden_layers"),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=_positive_int(config, "intermediate_size"),
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        sliding_window=int(sliding_window),
        rms_norm_eps=_finite_float(config, "rms_norm_eps", default=1e-5),
        swiglu_alpha=_finite_float(config, "swiglu_alpha", default=1.702),
        swiglu_limit=_finite_float(config, "swiglu_limit", default=7.0),
        attention_bias=attention_bias,
        num_labels=len(labels),
        max_position_embeddings=_positive_int(
            config, "max_position_embeddings", default=131072
        ),
        rope_type=rope_type,
    )


def privacy_filter_labels(config: Any) -> tuple[str, ...]:
    """Return the ordered BIOES token labels of a configuration.

    Raises:
        ExportQualificationError: ``missing_label_map`` for an absent or
            non-contiguous map, ``label_map_mismatch`` for one that is not a
            well-formed ``O`` plus ``B/I/E/S`` label space.
    """

    raw = _value(config, "id2label", default=None)
    if not isinstance(raw, Mapping) or not raw:
        raise ExportQualificationError(ExportReason.MISSING_LABEL_MAP)
    try:
        ordered = {int(key): str(value) for key, value in raw.items()}
    except (TypeError, ValueError):
        raise ExportQualificationError(ExportReason.MISSING_LABEL_MAP) from None
    if sorted(ordered) != list(range(len(ordered))):
        raise ExportQualificationError(ExportReason.MISSING_LABEL_MAP)
    labels = tuple(ordered[index] for index in range(len(ordered)))
    _span_classes(labels)
    return labels


def _span_classes(labels: Sequence[str]) -> tuple[str, ...]:
    if not labels or labels[0] != "O" or (len(labels) - 1) % 4:
        raise ExportQualificationError(ExportReason.LABEL_MAP_MISMATCH)
    classes: list[str] = []
    for index in range(1, len(labels), 4):
        quad = labels[index : index + 4]
        name = quad[0][2:]
        if not name or quad != tuple(f"{tag}-{name}" for tag in "BIES"):
            raise ExportQualificationError(ExportReason.LABEL_MAP_MISMATCH)
        classes.append(name)
    if len(set(classes)) != len(classes):
        raise ExportQualificationError(ExportReason.LABEL_MAP_MISMATCH)
    return tuple(classes)


def privacy_filter_label_space(config: Any) -> str:
    """Return the declared label-space identifier of a configuration.

    Raises:
        ExportQualificationError: ``label_map_mismatch`` when the label map
            disagrees with ``opf_metadata``, ``unsupported_configuration`` for
            an unknown inference contract, or ``unsupported_family`` for a
            label space no checkpoint declares.
    """

    labels = privacy_filter_labels(config)
    metadata_block = _value(config, "opf_metadata", default=None) or {}
    if not isinstance(metadata_block, Mapping):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    contract = metadata_block.get("inference_contract_version")
    if contract not in (None, 1):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    names = metadata_block.get("ner_class_names")
    if names is not None and tuple(str(name) for name in names) != labels:
        raise ExportQualificationError(ExportReason.LABEL_MAP_MISMATCH)
    version = metadata_block.get("category_version")
    if version:
        label_space = str(version)
    elif _span_classes(labels) == OPENAI_PRIVACY_FILTER_SPAN_LABELS:
        label_space = OPENAI_PRIVACY_FILTER_LABEL_SPACE
    else:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_FAMILY)
    if not any(item.label_space == label_space for item in PRIVACY_FILTER_CHECKPOINTS):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_FAMILY)
    return label_space


def check_privacy_filter_source(
    config: Any,
    *,
    model_id: str,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Check that a source checkpoint is an eligible, pinned Privacy Filter.

    Hub ids must be declared in :data:`PRIVACY_FILTER_CHECKPOINTS` with a
    matching label space. Local directories must carry a declared label
    space. Every request must pin the source revision (a Hub commit, or the
    ``sha256:`` content digest of a local directory).

    Args:
        config: Source configuration (object or mapping).
        model_id: Hub repository id or local checkpoint directory.
        expected_revision: The pinned source revision.

    Returns:
        A value-free eligibility record; local paths are never included.

    Raises:
        ExportQualificationError: ``unsupported_remote_code``,
            ``unsupported_family``, ``unsupported_configuration``,
            ``label_map_mismatch``, ``source_revision_unavailable`` (no pin),
            or ``source_revision_mismatch`` (pin differs from a reviewed one).
    """

    architecture = privacy_filter_architecture(config)
    label_space = privacy_filter_label_space(config)
    declared: PrivacyFilterCheckpoint | None = None
    if Path(str(model_id)).expanduser().is_dir():
        source_kind = "local_directory"
        if not any(
            item.label_space == label_space
            and item.num_labels == architecture.num_labels
            for item in PRIVACY_FILTER_CHECKPOINTS
        ):
            raise ExportQualificationError(ExportReason.LABEL_MAP_MISMATCH)
    else:
        source_kind = "hub"
        declared = privacy_filter_checkpoint(str(model_id))
        if declared is None:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_FAMILY)
        if (
            declared.label_space != label_space
            or declared.num_labels != architecture.num_labels
        ):
            raise ExportQualificationError(ExportReason.LABEL_MAP_MISMATCH)
    if not expected_revision:
        raise ExportQualificationError(ExportReason.SOURCE_REVISION_UNAVAILABLE)
    if (
        declared is not None
        and declared.revision is not None
        and expected_revision != declared.revision
    ):
        raise ExportQualificationError(ExportReason.SOURCE_REVISION_MISMATCH)
    return {
        "family": PRIVACY_FILTER_FAMILY,
        "adapter_version": PRIVACY_FILTER_ADAPTER_VERSION,
        "source_kind": source_kind,
        "declared_checkpoint": declared.repo_id if declared is not None else None,
        "declared_revision": declared.revision if declared is not None else None,
        "label_space": label_space,
        "num_labels": architecture.num_labels,
        "architecture_sha256": architecture.sha256,
        "revision_pinned": True,
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _check_privacy_filter_environment() -> None:
    """Require a Transformers release that ships the native implementation."""

    try:
        version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        raise ExportQualificationError(
            ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT
        ) from None
    parts = version.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        raise ExportQualificationError(
            ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT
        ) from None
    if (major, minor) < PRIVACY_FILTER_MIN_TRANSFORMERS:
        # Older releases only load the family through repository code.
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT)
    try:
        found = importlib_util.find_spec(PRIVACY_FILTER_NATIVE_MODULE) is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT)


def _privacy_filter_max_sequence_length(config: Any) -> int:
    limit = _value(config, "max_position_embeddings", default=None)
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        return min(int(limit), PRIVACY_FILTER_MAX_EXPORT_SEQUENCE_LENGTH)
    return PRIVACY_FILTER_MAX_EXPORT_SEQUENCE_LENGTH


def require_native_privacy_filter(model: Any) -> None:
    """Fail closed unless *model* is the reviewed Transformers implementation.

    Raises:
        ExportQualificationError: ``unsupported_remote_code`` for a class
            loaded from repository code (or any other module).
    """

    module = type(model).__module__ or ""
    if not module.startswith(PRIVACY_FILTER_NATIVE_MODULE):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_REMOTE_CODE)


def _prepare_privacy_filter_model(model: Any) -> None:
    require_native_privacy_filter(model)
    config = getattr(model, "config", None)
    if config is not None:
        config._attn_implementation = "eager"


def load_privacy_filter_reference(
    torch_module: Any, model_id: str, cache_dir: str | None = None
) -> Any:
    """Load the native float32 eager reference without repository code.

    Raises:
        ExportQualificationError: ``unsupported_remote_code`` when the loaded
            class is not the native implementation.
    """

    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    require_native_privacy_filter(model)
    model.to(dtype=torch_module.float32)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Weight mapping
# ---------------------------------------------------------------------------


def expected_source_weight_shapes(
    architecture: PrivacyFilterArchitecture,
    *,
    classifier_bias: bool,
) -> dict[str, tuple[int, ...]]:
    """Return the Transformers state-dict keys and shapes of a checkpoint."""

    a = architecture
    q_dim = a.num_attention_heads * a.head_dim
    kv_dim = a.num_key_value_heads * a.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (a.vocab_size, a.hidden_size),
        "model.norm.weight": (a.hidden_size,),
        "score.weight": (a.num_labels, a.hidden_size),
    }
    if classifier_bias:
        shapes["score.bias"] = (a.num_labels,)
    for index in range(a.num_hidden_layers):
        prefix = f"model.layers.{index}."
        shapes.update(
            {
                f"{prefix}input_layernorm.weight": (a.hidden_size,),
                f"{prefix}post_attention_layernorm.weight": (a.hidden_size,),
                f"{prefix}self_attn.q_proj.weight": (q_dim, a.hidden_size),
                f"{prefix}self_attn.k_proj.weight": (kv_dim, a.hidden_size),
                f"{prefix}self_attn.v_proj.weight": (kv_dim, a.hidden_size),
                f"{prefix}self_attn.o_proj.weight": (a.hidden_size, q_dim),
                f"{prefix}self_attn.sinks": (a.num_attention_heads,),
                f"{prefix}mlp.router.weight": (a.num_experts, a.hidden_size),
                f"{prefix}mlp.router.bias": (a.num_experts,),
                f"{prefix}mlp.experts.gate_up_proj": (
                    a.num_experts,
                    a.hidden_size,
                    2 * a.intermediate_size,
                ),
                f"{prefix}mlp.experts.gate_up_proj_bias": (
                    a.num_experts,
                    2 * a.intermediate_size,
                ),
                f"{prefix}mlp.experts.down_proj": (
                    a.num_experts,
                    a.intermediate_size,
                    a.hidden_size,
                ),
                f"{prefix}mlp.experts.down_proj_bias": (a.num_experts, a.hidden_size),
            }
        )
        if a.attention_bias:
            shapes.update(
                {
                    f"{prefix}self_attn.q_proj.bias": (q_dim,),
                    f"{prefix}self_attn.k_proj.bias": (kv_dim,),
                    f"{prefix}self_attn.v_proj.bias": (kv_dim,),
                    f"{prefix}self_attn.o_proj.bias": (a.hidden_size,),
                }
            )
    return shapes


def map_privacy_filter_weights(
    torch_module: Any,
    state_dict: Mapping[str, Any],
    architecture: PrivacyFilterArchitecture,
) -> dict[str, Any]:
    """Map a Transformers Privacy Filter state dict to the export module.

    The mapping fuses ``q/k/v`` projections into one ``qkv`` linear (the MLX
    layout), flattens each layer's ``[experts, hidden, 2 * intermediate]``
    expert projection into one ``[hidden, experts * 2 * intermediate]``
    matrix, keeps the classifier bias when the checkpoint has one, and casts
    every tensor to float32.

    Raises:
        ExportQualificationError: ``unsupported_configuration`` for missing,
            unexpected, or mis-shaped weights. The error is value-free.
    """

    torch = torch_module
    classifier_bias = "score.bias" in state_dict
    expected = expected_source_weight_shapes(
        architecture, classifier_bias=classifier_bias
    )
    if set(state_dict) != set(expected):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    for key, shape in expected.items():
        if tuple(state_dict[key].shape) != shape:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)

    def tensor(key: str) -> Any:
        return state_dict[key].detach().to(torch.float32).clone()

    a = architecture
    mapped: dict[str, Any] = {
        "embedding.weight": tensor("model.embed_tokens.weight"),
        "norm.weight": tensor("model.norm.weight"),
        "classifier.weight": tensor("score.weight"),
    }
    if classifier_bias:
        mapped["classifier.bias"] = tensor("score.bias")
    for index in range(a.num_hidden_layers):
        source = f"model.layers.{index}."
        target = f"layers.{index}."
        mapped[f"{target}attn_norm.weight"] = tensor(f"{source}input_layernorm.weight")
        mapped[f"{target}mlp_norm.weight"] = tensor(
            f"{source}post_attention_layernorm.weight"
        )
        mapped[f"{target}qkv.weight"] = torch.cat(
            [tensor(f"{source}self_attn.{name}_proj.weight") for name in "qkv"], dim=0
        )
        mapped[f"{target}out.weight"] = tensor(f"{source}self_attn.o_proj.weight")
        if a.attention_bias:
            mapped[f"{target}qkv.bias"] = torch.cat(
                [tensor(f"{source}self_attn.{name}_proj.bias") for name in "qkv"],
                dim=0,
            )
            mapped[f"{target}out.bias"] = tensor(f"{source}self_attn.o_proj.bias")
        mapped[f"{target}sinks"] = tensor(f"{source}self_attn.sinks")
        mapped[f"{target}router.weight"] = tensor(f"{source}mlp.router.weight")
        mapped[f"{target}router.bias"] = tensor(f"{source}mlp.router.bias")
        gate_up = tensor(f"{source}mlp.experts.gate_up_proj")
        mapped[f"{target}gate_up_weight"] = gate_up.permute(1, 0, 2).reshape(
            a.hidden_size, a.num_experts * 2 * a.intermediate_size
        )
        mapped[f"{target}gate_up_bias"] = tensor(
            f"{source}mlp.experts.gate_up_proj_bias"
        ).reshape(a.num_experts * 2 * a.intermediate_size)
        mapped[f"{target}down_weight"] = tensor(f"{source}mlp.experts.down_proj")
        mapped[f"{target}down_bias"] = tensor(f"{source}mlp.experts.down_proj_bias")
    return mapped


# ---------------------------------------------------------------------------
# Trace-friendly module
# ---------------------------------------------------------------------------


def build_privacy_filter_export_module(
    torch_module: Any,
    architecture: PrivacyFilterArchitecture,
    weights: Mapping[str, Any],
    *,
    inv_freq: Any,
    attention_scaling: float,
) -> Any:
    """Build the trace-friendly ``(input_ids, attention_mask) -> logits`` module.

    Args:
        torch_module: The imported ``torch`` module.
        architecture: Normalized architecture.
        weights: Output of :func:`map_privacy_filter_weights`.
        inv_freq: Rotary inverse frequencies of the reference implementation.
        attention_scaling: Rotary attention scaling of the reference.

    Returns:
        An ``nn.Module`` in eval mode with the mapped weights loaded strictly.
    """

    torch = torch_module
    nn = torch.nn
    a = architecture
    mask_value = PRIVACY_FILTER_MASK_VALUE

    class RMSNorm(nn.Module):
        """RMSNorm in the traced (float32) precision, like the reference.

        The module is traced in float32, so the reference's float32 upcast is
        implicit. Float-to-float casts are deliberately absent: the ONNX fp16
        converter leaves them targeting float32 and breaks the graph types.
        """

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(a.hidden_size))
            self.eps = a.rms_norm_eps

        def forward(self, hidden: Any) -> Any:
            # ``x * x`` rather than ``x.pow(2)``: ONNX Runtime's fused
            # SimplifiedLayerNorm pass mis-handles fp16 graphs that share the
            # exponent constant across layers, so the pattern is not offered.
            variance = (hidden * hidden).mean(-1, keepdim=True)
            return self.weight * (hidden * torch.rsqrt(variance + self.eps))

    def rotate(x: Any, cos: Any, sin: Any) -> Any:
        first, second = x[..., ::2], x[..., 1::2]
        return torch.stack(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        ).flatten(-2)

    class PrivacyFilterExportLayer(nn.Module):
        """Attention with sinks followed by the dense-routed MoE block."""

        def __init__(self) -> None:
            super().__init__()
            self.q_dim = a.num_attention_heads * a.head_dim
            self.kv_dim = a.num_key_value_heads * a.head_dim
            self.groups = a.num_attention_heads // a.num_key_value_heads
            self.scale = float(a.head_dim) ** -0.25
            self.attn_norm = RMSNorm()
            self.qkv = nn.Linear(
                a.hidden_size, self.q_dim + 2 * self.kv_dim, bias=a.attention_bias
            )
            self.out = nn.Linear(self.q_dim, a.hidden_size, bias=a.attention_bias)
            self.sinks = nn.Parameter(torch.zeros(a.num_attention_heads))
            self.mlp_norm = RMSNorm()
            self.router = nn.Linear(a.hidden_size, a.num_experts)
            width = a.num_experts * 2 * a.intermediate_size
            self.gate_up_weight = nn.Parameter(torch.zeros(a.hidden_size, width))
            self.gate_up_bias = nn.Parameter(torch.zeros(width))
            self.down_weight = nn.Parameter(
                torch.zeros(a.num_experts, a.intermediate_size, a.hidden_size)
            )
            self.down_bias = nn.Parameter(torch.zeros(a.num_experts, a.hidden_size))
            self.register_buffer(
                "expert_ids", torch.arange(a.num_experts), persistent=False
            )

        def forward(self, hidden: Any, cos: Any, sin: Any, bias: Any) -> Any:
            batch, seq = hidden.shape[0], hidden.shape[1]
            heads, kv_heads, dim = (
                a.num_attention_heads,
                a.num_key_value_heads,
                a.head_dim,
            )
            qkv = self.qkv(self.attn_norm(hidden))
            q_end = self.q_dim
            k_end = self.q_dim + self.kv_dim
            query = qkv[..., :q_end].reshape(batch, seq, heads, dim).transpose(1, 2)
            key = qkv[..., q_end:k_end].reshape(batch, seq, kv_heads, dim)
            value = qkv[..., k_end:].reshape(batch, seq, kv_heads, dim)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            query = rotate(query, cos, sin) * self.scale
            key = rotate(key, cos, sin) * self.scale
            if self.groups > 1:
                key = (
                    key[:, :, None, :, :]
                    .expand(batch, kv_heads, self.groups, seq, dim)
                    .reshape(batch, heads, seq, dim)
                )
                value = (
                    value[:, :, None, :, :]
                    .expand(batch, kv_heads, self.groups, seq, dim)
                    .reshape(batch, heads, seq, dim)
                )
            scores = torch.matmul(query, key.transpose(2, 3)) + bias
            # Softmax over [scores, sink] without materializing the sink
            # column: identical to the reference concat-softmax-drop. The
            # running maximum uses ``where`` because the ONNX fp16 converter
            # mis-orders the casts it inserts around ``Max``.
            sink = self.sinks.reshape(1, heads, 1, 1)
            row_peak = scores.amax(dim=-1, keepdim=True)
            peak = torch.where(row_peak > sink, row_peak, sink)
            weights = torch.exp(scores - peak)
            total = weights.sum(dim=-1, keepdim=True) + torch.exp(sink - peak)
            attended = torch.matmul(weights / total, value)
            attended = attended.transpose(1, 2).reshape(batch, seq, self.q_dim)
            hidden = hidden + self.out(attended)

            tokens = self.mlp_norm(hidden).reshape(-1, a.hidden_size)
            router_logits = self.router(tokens)
            top_values, top_index = torch.topk(
                router_logits, a.experts_per_token, dim=-1
            )
            top_weights = torch.softmax(top_values, dim=-1) / a.experts_per_token
            selected = (top_index[..., None] == self.expert_ids).to(tokens.dtype)
            routing = (selected * top_weights[..., None]).sum(dim=1)
            gate_up = torch.matmul(tokens, self.gate_up_weight) + self.gate_up_bias
            gate_up = gate_up.reshape(-1, a.num_experts, 2 * a.intermediate_size)
            gate = torch.clamp(gate_up[..., : a.intermediate_size], max=a.swiglu_limit)
            up = torch.clamp(
                gate_up[..., a.intermediate_size :],
                min=-a.swiglu_limit,
                max=a.swiglu_limit,
            )
            activated = (up + 1.0) * (gate * torch.sigmoid(gate * a.swiglu_alpha))
            experts = torch.matmul(activated.transpose(0, 1), self.down_weight)
            experts = experts + self.down_bias[:, None, :]
            mixed = (experts * routing.transpose(0, 1)[..., None]).sum(dim=0)
            mixed = mixed * a.experts_per_token
            return hidden + mixed.reshape(batch, seq, a.hidden_size)

    class PrivacyFilterTokenClassificationExport(nn.Module):
        """Exportable Privacy Filter token classifier returning logits only."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(a.vocab_size, a.hidden_size)
            self.layers = nn.ModuleList(
                [PrivacyFilterExportLayer() for _ in range(a.num_hidden_layers)]
            )
            self.norm = RMSNorm()
            self.classifier = nn.Linear(
                a.hidden_size, a.num_labels, bias="classifier.bias" in weights
            )
            self.sliding_window = a.sliding_window
            self.attention_scaling = float(attention_scaling)
            self.register_buffer(
                "inv_freq",
                inv_freq.detach().to(torch.float32).clone(),
                persistent=False,
            )

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden = self.embedding(input_ids)
            seq = input_ids.shape[1]
            positions = torch.arange(seq, device=input_ids.device)
            freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
            cos = torch.cos(freqs) * self.attention_scaling
            sin = torch.sin(freqs) * self.attention_scaling
            keep = attention_mask.to(hidden.dtype)
            key_bias = (1.0 - keep)[:, None, None, :] * mask_value
            distance = (positions[None, :] - positions[:, None]).abs()
            outside = (distance > self.sliding_window).to(hidden.dtype)
            bias = key_bias + outside[None, None, :, :] * mask_value
            for layer in self.layers:
                hidden = layer(hidden, cos, sin, bias)
            return self.classifier(self.norm(hidden))

    module = PrivacyFilterTokenClassificationExport()
    module.load_state_dict(dict(weights), strict=True)
    module.eval()
    return module


def _build_privacy_filter_wrapper(torch_module: Any, model: Any) -> Any:
    """Return the export module built from a loaded native reference model."""

    require_native_privacy_filter(model)
    architecture = privacy_filter_architecture(model.config)
    weights = map_privacy_filter_weights(torch_module, model.state_dict(), architecture)
    rotary = model.model.rotary_emb
    return build_privacy_filter_export_module(
        torch_module,
        architecture,
        weights,
        inv_freq=rotary.inv_freq,
        attention_scaling=float(rotary.attention_scaling),
    )


def _privacy_filter_trace_context(torch_module: Any, model: Any) -> Any:
    # The export module is already trace-friendly; nothing is patched.
    return nullcontext()


# ---------------------------------------------------------------------------
# Canonical decoder and threshold profile
# ---------------------------------------------------------------------------


def privacy_filter_viterbi_biases(
    model_id: str,
    config: Any,
    *,
    cache_dir: str | None = None,
) -> tuple[dict[str, float], str]:
    """Resolve the default Viterbi operating point without network access.

    Sources, in order: ``_mlx_viterbi_biases`` in the configuration, the
    checkpoint's ``viterbi_calibration.json`` (local directory, or the local
    Hugging Face cache only), else zero biases.

    Returns:
        ``(biases, source)`` where *source* is ``"config"``,
        ``"viterbi_calibration.json"``, or ``"default_zero"``.

    Raises:
        ExportQualificationError: ``unsupported_configuration`` for a
            malformed calibration file or non-finite biases.
    """

    from openmed.core.decoding import resolve_viterbi_biases

    configured = _value(config, "_mlx_viterbi_biases", default=None)
    if configured:
        return _resolve_biases(resolve_viterbi_biases, configured), "config"

    calibration: Path | None = None
    local = Path(str(model_id)).expanduser()
    if local.is_dir():
        candidate = local / VITERBI_CALIBRATION_FILENAME
        calibration = candidate if candidate.is_file() else None
    else:
        calibration = _cached_hub_file(
            str(model_id),
            VITERBI_CALIBRATION_FILENAME,
            cache_dir=cache_dir,
            revision=_value(config, "_commit_hash", default=None),
        )
    if calibration is None:
        return _resolve_biases(resolve_viterbi_biases, {}), "default_zero"
    try:
        payload = json.loads(calibration.read_text(encoding="utf-8"))
        biases = payload["operating_points"]["default"]["biases"]
    except (OSError, ValueError, KeyError, TypeError):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION) from None
    return _resolve_biases(resolve_viterbi_biases, biases), VITERBI_CALIBRATION_FILENAME


def _resolve_biases(resolver: Any, biases: Any) -> dict[str, float]:
    if not isinstance(biases, Mapping):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    try:
        return dict(resolver(dict(biases)))
    except (TypeError, ValueError):
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION) from None


def _cached_hub_file(
    repo_id: str,
    filename: str,
    *,
    cache_dir: str | None,
    revision: str | None,
) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        found = try_to_load_from_cache(
            repo_id, filename, cache_dir=cache_dir, revision=revision
        )
    except Exception:  # noqa: BLE001 - a cache miss never fails export.
        return None
    return Path(found) if isinstance(found, (str, Path)) else None


def privacy_filter_decoder_profile(
    config: Any,
    *,
    viterbi_biases: Mapping[str, float] | None = None,
    biases_source: str = "default_zero",
    confidence_threshold: float = PRIVACY_FILTER_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Return the value-free decoder and threshold profile for manifests.

    The profile names the canonical decoder (BIOES Viterbi over log-softmax
    emissions with the six learned transition biases), the threshold profile
    (mean token probability ``>=`` the confidence threshold, the
    ``extract_pii`` default), the span construction, and the label space. It
    carries a digest so consumers can pin it.

    Raises:
        ExportQualificationError: ``unsupported_configuration`` for an
            out-of-range threshold or invalid biases.
    """

    from openmed.core.decoding import resolve_viterbi_biases

    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_CONFIGURATION)
    labels = privacy_filter_labels(config)
    label_space = privacy_filter_label_space(config)
    metadata_block = _value(config, "opf_metadata", default=None) or {}
    body: dict[str, Any] = {
        "format": DECODER_PROFILE_FORMAT,
        "schema_version": DECODER_PROFILE_SCHEMA_VERSION,
        "family": PRIVACY_FILTER_FAMILY,
        "adapter_version": PRIVACY_FILTER_ADAPTER_VERSION,
        "task": "token-classification",
        "decoder": PRIVACY_FILTER_DECODER,
        "emissions": "log_softmax",
        "label_space": label_space,
        "label_map_sha256": label_map_identity(dict(enumerate(labels)))["sha256"],
        "num_labels": len(labels),
        "span_label_aliases": dict(
            sorted(_LABEL_SPACE_ALIASES.get(label_space, {}).items())
        ),
        "viterbi": {
            "operating_point": "default",
            "biases": _resolve_biases(resolve_viterbi_biases, viterbi_biases or {}),
            "source": str(biases_source),
        },
        "threshold_profile": {
            "confidence_threshold": float(confidence_threshold),
            "score": "mean_token_probability",
            "comparison": "greater_or_equal",
        },
        "span_construction": {
            "aggregation": "simple",
            "offsets": "tokenizer_offset_mapping",
            "grapheme_snapping": True,
            "whitespace_trim": True,
            "refinement": "refine_privacy_filter_span",
        },
        "encoding": (
            str(metadata_block.get("encoding"))
            if isinstance(metadata_block, Mapping) and metadata_block.get("encoding")
            else None
        ),
    }
    body["sha256"] = _canonical_digest(body)
    return body


def verify_decoder_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a loaded decoder profile and its digest.

    Raises:
        ExportQualificationError: ``report_invalid`` for a malformed profile,
            ``adapter_version_mismatch`` for another adapter version.
    """

    if (
        not isinstance(profile, Mapping)
        or profile.get("format") != DECODER_PROFILE_FORMAT
        or profile.get("schema_version") != DECODER_PROFILE_SCHEMA_VERSION
        or profile.get("decoder") != PRIVACY_FILTER_DECODER
        or not isinstance(profile.get("viterbi"), Mapping)
        or not isinstance(profile.get("threshold_profile"), Mapping)
    ):
        raise ExportQualificationError(ExportReason.REPORT_INVALID)
    body = {key: value for key, value in profile.items() if key != "sha256"}
    if profile.get("sha256") != _canonical_digest(body):
        raise ExportQualificationError(ExportReason.REPORT_INVALID)
    if profile.get("adapter_version") != PRIVACY_FILTER_ADAPTER_VERSION:
        raise ExportQualificationError(ExportReason.ADAPTER_VERSION_MISMATCH)
    return dict(profile)


def decode_privacy_filter_logits(
    logit_rows: Sequence[Sequence[float]],
    offsets: Sequence[Sequence[int]],
    text: str,
    *,
    id2label: Mapping[Any, str],
    biases: Mapping[str, float],
    confidence_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Decode real-token logits with the canonical Privacy Filter decoder.

    Mirrors the MLX pipeline: log-softmax emissions, constrained BIOES
    Viterbi with the transition biases, token spans mapped to grapheme-safe
    character spans, whitespace trimming, ``refine_privacy_filter_span``, and
    the mean token probability as the span score. Spans scoring below
    *confidence_threshold* are dropped.

    Args:
        logit_rows: One logit row per real (non-padding) token.
        offsets: Character offsets for the same tokens.
        text: Source text the offsets index; kept in memory only.
        id2label: Token label map.
        biases: Viterbi transition biases.
        confidence_threshold: Minimum span score to keep.

    Returns:
        ``{"label", "start", "end", "score"}`` spans with raw model labels.
    """

    from openmed.core.decoding import (
        build_label_info,
        labels_to_token_spans,
        refine_privacy_filter_span,
        token_spans_to_char_spans,
        trim_span_whitespace,
        viterbi_decode,
    )
    from openmed.core.decoding.spans import grapheme_break_checker

    if not logit_rows:
        return []
    label_info = build_label_info({int(k): str(v) for k, v in id2label.items()})
    log_probs = [_log_softmax(row) for row in logit_rows]
    predicted = viterbi_decode(log_probs, label_info=label_info, biases=dict(biases))
    token_spans = labels_to_token_spans(
        {index: int(label) for index, label in enumerate(predicted)}, label_info
    )
    token_offsets = [(int(item[0]), int(item[1])) for item in offsets]
    char_spans = token_spans_to_char_spans(token_spans, token_offsets, text)
    spans: list[dict[str, Any]] = []
    break_checker = None
    for (_, token_start, token_end), (span_label, start, end) in zip(
        token_spans, char_spans
    ):
        if break_checker is None:
            break_checker = grapheme_break_checker(text)
        start, end = trim_span_whitespace(start, end, text, break_checker=break_checker)
        if end <= start:
            continue
        label = (
            label_info.span_class_names[span_label]
            if 0 <= span_label < len(label_info.span_class_names)
            else f"label_{span_label}"
        )
        scores = [
            math.exp(log_probs[index][predicted[index]])
            for index in range(token_start, token_end)
            if 0 <= index < len(log_probs)
        ]
        start, end = refine_privacy_filter_span(
            label, start, end, text, break_checker=break_checker
        )
        if end <= start:
            continue
        score = sum(scores) / len(scores) if scores else 0.0
        if score < confidence_threshold:
            continue
        spans.append({"label": label, "start": start, "end": end, "score": score})
    return spans


def privacy_filter_span_decoder(profile: Mapping[str, Any]) -> SpanDecoder:
    """Return the qualification span decoder bound to a decoder profile.

    The decoder applies the profile's biases and confidence threshold to the
    reference and to every candidate alike, and maps span labels through the
    profile's canonical aliases so recall and leakage gates see direct
    identifiers.
    """

    biases = dict(profile["viterbi"]["biases"])
    threshold = float(profile["threshold_profile"]["confidence_threshold"])
    aliases = dict(profile.get("span_label_aliases") or {})

    def decode(
        rows: Sequence[Sequence[float]],
        *,
        offsets: Sequence[Sequence[int]],
        attention: Sequence[int],
        id2label: Mapping[Any, str],
        text: str,
    ) -> list[dict[str, Any]]:
        keep = [
            index
            for index in range(min(len(rows), len(offsets), len(attention)))
            if attention[index]
        ]
        spans = decode_privacy_filter_logits(
            [rows[index] for index in keep],
            [offsets[index] for index in keep],
            text,
            id2label=id2label,
            biases=biases,
            confidence_threshold=threshold,
        )
        return [
            {
                "label": aliases.get(span["label"], span["label"]),
                "start": span["start"],
                "end": span["end"],
            }
            for span in spans
        ]

    return decode


def _privacy_filter_binding(
    config: Any, model_id: str, cache_dir: str | None = None
) -> QualificationBinding:
    biases, source = privacy_filter_viterbi_biases(
        model_id, config, cache_dir=cache_dir
    )
    profile = privacy_filter_decoder_profile(
        config, viterbi_biases=biases, biases_source=source
    )
    return QualificationBinding(
        decoder_profile=profile,
        span_decoder=privacy_filter_span_decoder(profile),
        fixtures_path=PRIVACY_FILTER_QUALIFICATION_SUITE,
        gate_languages=True,
    )


def write_decoder_profile(profile: Mapping[str, Any], path: str | Path) -> Path:
    """Write a decoder profile as sorted JSON and return the path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(profile), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def load_decoder_profile(path: str | Path) -> dict[str, Any]:
    """Load and verify a decoder profile written by an export.

    Raises:
        ExportQualificationError: ``report_invalid`` or
            ``adapter_version_mismatch``.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ExportQualificationError(ExportReason.REPORT_INVALID) from None
    return verify_decoder_profile(payload)


def _log_softmax(row: Sequence[float]) -> list[float]:
    values = [float(value) for value in row]
    peak = max(values)
    total = math.log(sum(math.exp(value - peak) for value in values))
    return [value - peak - total for value in values]


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


PRIVACY_FILTER_EXPORT_ADAPTER = register_export_adapter(
    ExportFamilyAdapter(
        family=PRIVACY_FILTER_FAMILY,
        version=PRIVACY_FILTER_ADAPTER_VERSION,
        model_types=PRIVACY_FILTER_MODEL_TYPE_ALIASES,
        support={
            COREML_RUNTIME: PRIVACY_FILTER_COREML_SUPPORT,
            ONNX_RUNTIME: PRIVACY_FILTER_ONNX_SUPPORT,
        },
        prepare_model=_prepare_privacy_filter_model,
        build_wrapper=_build_privacy_filter_wrapper,
        trace_context=_privacy_filter_trace_context,
        check_environment=_check_privacy_filter_environment,
        max_sequence_length=_privacy_filter_max_sequence_length,
        check_source=check_privacy_filter_source,
        load_reference=load_privacy_filter_reference,
        qualification_binding=_privacy_filter_binding,
        onnx_operators=PRIVACY_FILTER_ONNX_OPERATORS,
        coreml_fp32_ops=PRIVACY_FILTER_COREML_FP32_OPS,
    )
)


__all__ = [
    "DECODER_PROFILE_FILENAME",
    "DECODER_PROFILE_FORMAT",
    "DECODER_PROFILE_SCHEMA_VERSION",
    "OPENAI_PRIVACY_FILTER_LABEL_ALIASES",
    "OPENAI_PRIVACY_FILTER_LABEL_SPACE",
    "OPENAI_PRIVACY_FILTER_SPAN_LABELS",
    "PRIVACY_FILTER_ADAPTER_VERSION",
    "PRIVACY_FILTER_CHECKPOINTS",
    "PRIVACY_FILTER_CONFIDENCE_THRESHOLD",
    "PRIVACY_FILTER_COREML_FP32_OPS",
    "PRIVACY_FILTER_COREML_SUPPORT",
    "PRIVACY_FILTER_DECODER",
    "PRIVACY_FILTER_EXPORT_ADAPTER",
    "PRIVACY_FILTER_FAMILY",
    "PRIVACY_FILTER_MAX_EXPORT_SEQUENCE_LENGTH",
    "PRIVACY_FILTER_ONNX_OPERATORS",
    "PRIVACY_FILTER_ONNX_SUPPORT",
    "PRIVACY_FILTER_QUALIFICATION_SUITE",
    "PrivacyFilterArchitecture",
    "PrivacyFilterCheckpoint",
    "build_privacy_filter_export_module",
    "check_privacy_filter_source",
    "decode_privacy_filter_logits",
    "expected_source_weight_shapes",
    "load_decoder_profile",
    "load_privacy_filter_reference",
    "map_privacy_filter_weights",
    "privacy_filter_architecture",
    "privacy_filter_checkpoint",
    "privacy_filter_decoder_profile",
    "privacy_filter_label_space",
    "privacy_filter_labels",
    "privacy_filter_span_decoder",
    "privacy_filter_viterbi_biases",
    "reject_remote_code",
    "require_native_privacy_filter",
    "verify_decoder_profile",
    "write_decoder_profile",
]
